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
            if job_status == 'submitted' and job_id:
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
            logger.info(f'Pipe run {run_id}: {summary}')
            if pipe.needs_resubmission:
                logger.info(f'Pipe run {run_id}: resubmitting to pick up retried tasks.')
                try:
                    job_status, job_id = pipe.submit_to_scheduler()
                    if job_status == 'submitted' and job_id:
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
        ingestion of remaining tasks.
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
