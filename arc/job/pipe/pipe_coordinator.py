"""
Pipe run lifecycle coordinator.

Manages the active pipe run registry, eligibility checks, submission,
reconstruction, polling, resubmission, and ingestion dispatch.

This module owns the lifecycle of pipe runs once they are created.
Family-specific task planning lives in ``pipe_planner.py``.
"""

import time
from typing import TYPE_CHECKING, Dict, List

from arc.common import get_logger
from arc.imports import settings

from arc.job.pipe.pipe_run import PipeRun, ingest_completed_task
from arc.job.pipe.pipe_state import PipeRunState, TaskState, TaskSpec, read_task_state

if TYPE_CHECKING:
    from arc.scheduler import Scheduler

logger = get_logger()

pipe_settings = settings['pipe_settings']


class PipeCoordinator:
    """
    Manages the lifecycle of active pipe runs for a Scheduler instance.

    Owns:
      - pipe eligibility checks
      - run creation / submission / reconstruction
      - polling / resubmission
      - terminal ingestion dispatch

    Args:
        sched: The owning Scheduler instance, providing ``project_directory``,
               ``species_dict``, and ``output``.
    """

    def __init__(self, sched: 'Scheduler'):
        self.sched = sched
        self.active_pipes: Dict[str, PipeRun] = {}
        self._pipe_poll_failures: Dict[str, int] = {}
        self._last_pipe_summary: Dict[str, str] = {}

    def should_use_pipe(self, tasks: List[TaskSpec]) -> bool:
        """
        Determine whether a list of tasks is eligible for pipe-mode execution.

        Returns ``True`` only if:
          1. Pipe mode is enabled.
          2. There are at least ``min_tasks`` tasks.
          3. All tasks are homogeneous in engine, task_family, owner_type,
             level, required_cores, and required_memory_mb.
        """
        if not pipe_settings.get('enabled', True):
            return False
        if not tasks:
            return False
        min_tasks = pipe_settings.get('min_tasks', 10)
        if len(tasks) < min_tasks:
            return False
        ref = tasks[0]
        return all(t.engine == ref.engine
                   and t.task_family == ref.task_family
                   and t.owner_type == ref.owner_type
                   and t.level == ref.level
                   and t.required_cores == ref.required_cores
                   and t.required_memory_mb == ref.required_memory_mb
                   for t in tasks[1:])

    def submit_pipe_run(self, run_id: str, tasks: List[TaskSpec],
                        cluster_software: str = 'slurm') -> PipeRun:
        """
        Create, stage, and register a new pipe run.

        Attempts to write a submit script and submit the array job.
        On submission failure, the run is still registered as STAGED.

        Returns:
            PipeRun: The created pipe run.
        """
        pipe = PipeRun(
            project_directory=self.sched.project_directory,
            run_id=run_id,
            tasks=tasks,
            cluster_software=cluster_software,
            max_workers=pipe_settings.get('max_workers', 100),
            max_attempts=pipe_settings.get('max_attempts', 3),
        )
        pipe.stage()
        try:
            pipe.write_submit_script()
        except NotImplementedError:
            logger.warning(f'Pipe run {run_id}: submit script generation not yet implemented '
                           f'for {cluster_software}. Tasks are staged but must be submitted manually.')
            self.active_pipes[run_id] = pipe
            return pipe
        try:
            job_status, job_id = pipe.submit_to_scheduler()
            if job_id and job_status in ('submitted', 'running'):
                pipe.scheduler_job_id = job_id
                pipe.status = PipeRunState.SUBMITTED
                pipe.submitted_at = time.time()
                pipe._save_run_metadata()
                logger.info(f'Pipe run {run_id} submitted as job {job_id}.')
            else:
                logger.warning(f'Pipe run {run_id}: submission returned status={job_status}. '
                               f'Tasks are staged at {pipe.pipe_root}.')
        except Exception as e:
            logger.warning(f'Pipe run {run_id}: submission failed ({e}). '
                           f'Tasks are staged at {pipe.pipe_root} but not running.')
        self.active_pipes[run_id] = pipe
        return pipe

    def register_pipe_run_from_dir(self, pipe_root: str) -> PipeRun:
        """Reconstruct and register an existing pipe run from disk."""
        pipe = PipeRun.from_dir(pipe_root)
        self.active_pipes[pipe.run_id] = pipe
        return pipe

    def poll_pipes(self) -> None:
        """
        Reconcile all active pipe runs.

        Detects orphans, schedules retries, resubmits if needed, ingests
        terminal runs, and removes completed/failed runs from the registry.

        Tolerates up to 3 consecutive reconciliation failures per run before
        marking it as FAILED and removing it.
        """
        max_consecutive_failures = 3
        for run_id in list(self.active_pipes.keys()):
            pipe = self.active_pipes[run_id]
            try:
                counts = pipe.reconcile()
            except Exception:
                n_failures = self._pipe_poll_failures.get(run_id, 0) + 1
                self._pipe_poll_failures[run_id] = n_failures
                logger.error(f'Pipe run {run_id}: reconciliation failed '
                             f'({n_failures}/{max_consecutive_failures})', exc_info=True)
                if n_failures >= max_consecutive_failures:
                    logger.error(f'Pipe run {run_id}: {max_consecutive_failures} consecutive polling failures. '
                                 f'Marking as FAILED and removing from active pipes.')
                    try:
                        pipe.status = PipeRunState.FAILED
                        pipe._save_run_metadata()
                    except Exception as e:
                        logger.debug(f'Pipe run {run_id}: best-effort FAILED persist failed: {e}')
                    del self.active_pipes[run_id]
                    self._pipe_poll_failures.pop(run_id, None)
                continue
            self._pipe_poll_failures.pop(run_id, None)
            summary = ', '.join(f'{state}: {n}' for state, n in sorted(counts.items()) if n > 0)
            if summary != self._last_pipe_summary.get(run_id):
                logger.info(f'Pipe run {run_id}: {summary}')
                self._last_pipe_summary[run_id] = summary
            if pipe.needs_resubmission:
                logger.info(f'Pipe run {run_id}: resubmitting to pick up retried tasks.')
                try:
                    job_status, job_id = pipe.submit_to_scheduler()
                    if job_id and job_status in ('submitted', 'running'):
                        pipe.scheduler_job_id = job_id
                        pipe.status = PipeRunState.SUBMITTED
                        pipe.submitted_at = time.time()
                        pipe._needs_resubmission = False
                        pipe._save_run_metadata()
                        logger.info(f'Pipe run {run_id}: resubmitted as job {job_id}.')
                    else:
                        pipe._needs_resubmission = False
                except Exception:
                    logger.warning(f'Pipe run {run_id}: resubmission failed.', exc_info=True)
            if pipe.status in (PipeRunState.COMPLETED, PipeRunState.COMPLETED_PARTIAL):
                self.ingest_pipe_results(pipe)
                del self.active_pipes[run_id]
            elif pipe.status == PipeRunState.FAILED:
                logger.error(f'Pipe run {run_id} has FAILED status. '
                             f'Ingesting any available results and removing from active pipes.')
                self.ingest_pipe_results(pipe)
                del self.active_pipes[run_id]

    def ingest_pipe_results(self, pipe: PipeRun) -> None:
        """
        Ingest results from a terminal pipe run.

        Dispatches by task_family. One broken task does not abort
        ingestion of remaining tasks. After all per-task ingestion,
        triggers family-specific post-processing (e.g., selecting
        the best conformer and spawning the next job).
        """
        for spec in pipe.tasks:
            try:
                state = read_task_state(pipe.pipe_root, spec.task_id)
            except (FileNotFoundError, ValueError, KeyError):
                logger.error(f'Pipe run {pipe.run_id}, task {spec.task_id}: '
                             f'could not read state, skipping.')
                continue
            if state.status == TaskState.COMPLETED.value:
                ingest_completed_task(pipe.run_id, pipe.pipe_root, spec, state,
                                      self.sched.species_dict, self.sched.output)
            elif state.status == TaskState.FAILED_TERMINAL.value:
                logger.error(f'Pipe run {pipe.run_id}, task {spec.task_id}: '
                             f'failed terminally (failure_class={state.failure_class}). '
                             f'Manual troubleshooting required.')
            elif state.status == TaskState.CANCELLED.value:
                logger.warning(f'Pipe run {pipe.run_id}, task {spec.task_id}: '
                               f'was cancelled.')
        self._post_ingest_pipe_run(pipe)

    def _post_ingest_pipe_run(self, pipe: PipeRun) -> None:
        """
        Trigger family-specific post-processing after all tasks in a pipe run
        have been individually ingested.

        Families requiring post-processing:
          - ts_opt: determine best TS conformer, then run opt job
          - conf_opt: determine most stable conformer, then run opt job
          - conf_sp: determine most stable conformer (sp_flag), then run opt job

        Other families (species_sp, species_freq, irc, rotor_scan_1d) are
        leaf jobs with no batch-level post-processing.
        """
        if not pipe.tasks:
            return
        task_family = pipe.tasks[0].task_family
        label = pipe.tasks[0].owner_key
        if not label or label not in self.sched.species_dict:
            return
        if task_family == 'ts_opt':
            self._post_ingest_ts_opt(label)
        elif task_family == 'conf_opt':
            self._post_ingest_conf_opt(label)
        elif task_family == 'conf_sp':
            self._post_ingest_conf_sp(label)

    def _post_ingest_ts_opt(self, label: str) -> None:
        """After all TS opt tasks, pick the best conformer and run proper opt."""
        ts_species = self.sched.species_dict[label]
        if not ts_species.is_ts:
            logger.warning(f'_post_ingest_ts_opt called for non-TS species {label}, skipping.')
            return
        if all(tsg.energy is None for tsg in ts_species.ts_guesses):
            logger.error(f'No ts_opt task converged for TS {label}.')
            return
        logger.info(f'\nConformer jobs for {label} successfully terminated (pipe mode).\n')
        try:
            self.sched.determine_most_likely_ts_conformer(label)
        except Exception:
            logger.error(f'Failed to determine most likely TS conformer for {label}.', exc_info=True)
            return
        if ts_species.initial_xyz is not None:
            if not self.sched.composite_method:
                self.sched.run_opt_job(label, fine=self.sched.fine_only)
            else:
                self.sched.run_composite_job(label)

    def _post_ingest_conf_opt(self, label: str) -> None:
        """After all conformer opt tasks, pick the best conformer and run opt."""
        logger.info(f'\nConformer opt jobs for {label} successfully terminated (pipe mode).\n')
        try:
            if self.sched.species_dict[label].is_ts:
                self.sched.determine_most_likely_ts_conformer(label)
            else:
                self.sched.determine_most_stable_conformer(label, sp_flag=False)
        except Exception:
            logger.error(f'Failed to determine most stable conformer for {label}.', exc_info=True)
            return
        if self.sched.species_dict[label].initial_xyz is not None:
            if not self.sched.composite_method:
                self.sched.run_opt_job(label, fine=self.sched.fine_only)
            else:
                self.sched.run_composite_job(label)

    def _post_ingest_conf_sp(self, label: str) -> None:
        """After all conformer SP tasks, pick the best conformer and run opt."""
        logger.info(f'\nConformer SP jobs for {label} successfully terminated (pipe mode).\n')
        try:
            self.sched.determine_most_stable_conformer(label, sp_flag=True)
        except Exception:
            logger.error(f'Failed to determine most stable conformer for {label}.', exc_info=True)
            return
        if self.sched.species_dict[label].initial_xyz is not None:
            if not self.sched.composite_method:
                self.sched.run_opt_job(label, fine=self.sched.fine_only)
            else:
                self.sched.run_composite_job(label)
