"""
Pipe run lifecycle coordinator.

Manages the active pipe run registry, eligibility checks, submission,
reconstruction, polling, resubmission, and ingestion dispatch.

This module owns the lifecycle of pipe runs once they are created.
Family-specific task planning lives in ``pipe_planner.py``.
"""

import os
import time
from typing import TYPE_CHECKING, Dict, List

from arc.common import get_logger
from arc.imports import settings

from arc.job.pipe.pipe_run import PipeRun, ingest_completed_task
from arc.job.pipe.pipe_state import (
    TASK_FAMILY_TO_JOB_TYPE, PipeRunState, TaskState, TaskSpec,
    TaskStateRecord, read_task_state,
)

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

    def _compute_pipe_root(self, run_id: str, tasks: List[TaskSpec]) -> str:
        """
        Compute the pipe_root path under ``calcs/``, following ARC's directory convention.

        Per-species runs: ``calcs/{TSs|Species}/<label>/pipe_<family>_<N>/``
        Cross-species batches: ``calcs/batches/pipe_<run_id>_<N>/``
        The index ``<N>`` auto-increments if a previous run directory exists.
        """
        calcs_dir = os.path.join(self.sched.project_directory, 'calcs')
        owner_keys = {t.owner_key for t in tasks} if tasks else set()
        if len(owner_keys) == 1:
            label = owner_keys.pop()
            species = self.sched.species_dict.get(label)
            if species is not None:
                folder = 'TSs' if species.is_ts else 'Species'
            else:
                folder = 'Species'
            base_dir = os.path.join(calcs_dir, folder, label)
            prefix = f'pipe_{tasks[0].task_family}'
        else:
            base_dir = os.path.join(calcs_dir, 'batches')
            prefix = f'pipe_{run_id}'
        return self._next_indexed_dir(base_dir, prefix)

    @staticmethod
    def _next_indexed_dir(base_dir: str, prefix: str) -> str:
        """Find the next available auto-incremented directory name."""
        if not os.path.isdir(base_dir):
            return os.path.join(base_dir, f'{prefix}_0')
        max_idx = -1
        for entry in os.listdir(base_dir):
            if entry.startswith(prefix + '_') and os.path.isdir(os.path.join(base_dir, entry)):
                suffix = entry[len(prefix) + 1:]
                if suffix.isdigit():
                    max_idx = max(max_idx, int(suffix))
        return os.path.join(base_dir, f'{prefix}_{max_idx + 1}')

    @staticmethod
    def _write_task_summary(pipe: PipeRun) -> None:
        """Write a task_summary.txt mapping each task to its worker and outcome."""
        tasks_dir = os.path.join(pipe.pipe_root, 'tasks')
        if not os.path.isdir(tasks_dir):
            return
        lines = [f'{"Task":<30} {"Worker":<8} {"Status":<20} {"Failure Class"}']
        lines.append('-' * 80)
        for task_id in sorted(os.listdir(tasks_dir)):
            try:
                state = read_task_state(pipe.pipe_root, task_id)
                worker = state.claimed_by or '?'
                status = state.status
                fc = state.failure_class or ''
            except (FileNotFoundError, ValueError, KeyError):
                worker, status, fc = '?', '?', ''
            lines.append(f'{task_id:<30} {worker:<8} {status:<20} {fc}')
        try:
            with open(os.path.join(pipe.pipe_root, 'task_summary.txt'), 'w') as f:
                f.write('\n'.join(lines) + '\n')
        except OSError as e:
            logger.warning(f'Could not write task_summary.txt for {pipe.run_id}: {e}')

    def submit_pipe_run(self, run_id: str, tasks: List[TaskSpec],
                        cluster_software: str = 'slurm') -> PipeRun:
        """
        Create, stage, and register a new pipe run.

        The pipe_root is placed under ``calcs/`` alongside regular job output,
        with auto-incrementing index to avoid collisions with prior runs.

        Returns:
            PipeRun: The created pipe run.
        """
        pipe_root = self._compute_pipe_root(run_id, tasks)
        pipe = PipeRun(
            project_directory=self.sched.project_directory,
            run_id=run_id,
            tasks=tasks,
            cluster_software=cluster_software,
            max_workers=pipe_settings.get('max_workers', 100),
            max_attempts=pipe_settings.get('max_attempts', 3),
            pipe_root=pipe_root,
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
        the best conformer and spawning the next job) — but only if
        no tasks were ejected to the Scheduler for troubleshooting.
        Ejected tasks will complete through the Scheduler's normal
        pipeline, and the Scheduler's main loop will trigger the
        next workflow steps when all conformer jobs are done.
        """
        # Write a task summary mapping tasks to workers and outcomes.
        self._write_task_summary(pipe)
        ejected_count = 0
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
            elif state.status == TaskState.FAILED_ESS.value:
                self._eject_to_scheduler(pipe, spec, state)
                ejected_count += 1
            elif state.status == TaskState.FAILED_TERMINAL.value:
                logger.error(f'Pipe run {pipe.run_id}, task {spec.task_id}: '
                             f'failed terminally (failure_class={state.failure_class}). '
                             f'Manual troubleshooting required.')
            elif state.status == TaskState.CANCELLED.value:
                logger.warning(f'Pipe run {pipe.run_id}, task {spec.task_id}: '
                               f'was cancelled.')
        if ejected_count > 0:
            logger.info(f'Pipe run {pipe.run_id}: {ejected_count} task(s) ejected to Scheduler '
                        f'for troubleshooting. Deferring post-ingestion workflow.')
        else:
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

    def _eject_to_scheduler(self, pipe: 'PipeRun', spec: TaskSpec,
                            state: 'TaskStateRecord') -> None:
        """
        Eject a failed pipe task to the Scheduler as an individual job.

        Translates the TaskSpec back into a ``Scheduler.run_job()`` call so that
        the Scheduler's existing ``troubleshoot_ess()`` pipeline handles it.
        """
        label = spec.owner_key
        if label not in self.sched.species_dict:
            logger.warning(f'Pipe run {pipe.run_id}, task {spec.task_id}: '
                           f'species "{label}" not in species_dict, cannot eject.')
            return
        # Map task_family to the Scheduler's job_type. Note: ts_opt pipe tasks
        # are TS conformer optimizations (at the guess level), not proper-level
        # optimizations. The Scheduler uses 'conf_opt' for these, not 'opt'.
        family_to_sched_job_type = {
            'ts_opt': 'conf_opt',
            'conf_opt': 'conf_opt',
            'conf_sp': 'conf_sp',
            'species_sp': 'sp',
            'species_freq': 'freq',
            'irc': 'irc',
            'rotor_scan_1d': 'scan',
        }
        job_type = family_to_sched_job_type.get(spec.task_family)
        if job_type is None:
            logger.warning(f'Pipe run {pipe.run_id}, task {spec.task_id}: '
                           f'unknown task_family "{spec.task_family}", cannot eject.')
            return
        payload = spec.input_payload or {}
        meta = spec.ingestion_metadata or {}
        kwargs = {
            'job_type': job_type,
            'label': label,
            'level_of_theory': spec.level,
            'job_adapter': spec.engine,
            'xyz': payload.get('xyz'),
            'conformer': meta.get('conformer_index'),
        }
        if spec.task_family == 'irc':
            kwargs['irc_direction'] = meta.get('irc_direction')
        elif spec.task_family == 'rotor_scan_1d':
            kwargs['rotor_index'] = meta.get('rotor_index')
            kwargs['torsions'] = payload.get('torsions')
        try:
            logger.info(f'Pipe run {pipe.run_id}, task {spec.task_id}: '
                        f'ejecting to Scheduler as individual {job_type} job for {label}.')
            self.sched.run_job(**kwargs)
        except Exception:
            logger.error(f'Pipe run {pipe.run_id}, task {spec.task_id}: '
                         f'failed to eject to Scheduler.', exc_info=True)
