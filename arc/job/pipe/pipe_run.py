"""
A module for the PipeRun orchestrator, task-spec routing, and result ingestion.

Contains:
  - ``PipeRun``: manages the lifecycle of a pipe run (staging, submit-script
    generation, reconciliation with orphan detection and retry scheduling).
  - Ingestion helpers: dispatch completed pipe task results back into ARC's
    species/output state by task family.
  - Routing helpers: build ``TaskSpec`` objects and decide whether to submit
    a pipe run for various task families.

All QA, troubleshooting, and downstream branching remain in mother ARC.
"""

import json
import os
import stat
import sys
import time
from numbers import Integral
from typing import Dict, List, Optional

import arc.parser.parser as parser
from arc.common import get_logger
from arc.imports import pipe_submit, settings

from arc.job.pipe.pipe_state import (
    PipeRunState,
    TaskState,
    TaskSpec,
    get_task_attempt_dir,
    initialize_task,
    read_task_state,
    update_task_state,
)

logger = get_logger()

RESUBMIT_GRACE = 120  # seconds – grace period after resubmission before flagging again

pipe_settings = settings['pipe_settings']
default_job_settings = settings['default_job_settings']
servers_dict = settings['servers']


class PipeRun:
    """
    Orchestrator for a pipe run.

    Args:
        project_directory (str): Path to the ARC project directory.
        run_id (str): Unique identifier for this pipe run.
        tasks (List[TaskSpec]): Task specifications to execute.
        cluster_software (str): Cluster scheduler type.
        max_workers (int): Maximum total array worker slots (array size).
        max_concurrent (Optional[int]): Max workers running simultaneously
            (like PBS ``%N``). Must be ``None`` (unthrottled) or a positive
            integer; ``0``, negatives, and non-integers raise ``ValueError``.
        max_attempts (int): Maximum retry attempts per task.
    """

    @staticmethod
    def _validate_max_concurrent(max_concurrent: Optional[int]) -> None:
        """Accept ``None`` or a positive integer for throttle settings."""
        if max_concurrent is None:
            return
        if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, Integral):
            raise ValueError('PipeRun max_concurrent must be None or a positive integer.')
        if max_concurrent > 0:
            return
        raise ValueError('PipeRun max_concurrent must be None or a positive integer.')

    def __init__(self,
                 project_directory: str,
                 run_id: str,
                 tasks: List[TaskSpec],
                 cluster_software: str,
                 max_workers: int = 100,
                 max_concurrent: Optional[int] = None,
                 max_attempts: int = 3,
                 pipe_root: Optional[str] = None,
                 ):
        self.project_directory = project_directory
        self.run_id = run_id
        self.tasks = tasks
        self.cluster_software = cluster_software
        self.max_workers = max_workers
        self._validate_max_concurrent(max_concurrent)
        self.max_concurrent = None if max_concurrent is None else int(max_concurrent)
        self.max_attempts = max_attempts
        self.pipe_root = pipe_root if pipe_root is not None \
            else os.path.join(project_directory, 'calcs', 'pipe_' + run_id)
        self.status = PipeRunState.CREATED
        self.created_at = time.time()
        self.submitted_at = None
        self.completed_at = None
        self.scheduler_job_id = None

    def _save_run_metadata(self) -> None:
        """Write run-level metadata to ``run.json`` under ``self.pipe_root``."""
        os.makedirs(self.pipe_root, exist_ok=True)
        run_path = os.path.join(self.pipe_root, 'run.json')
        # Derive homogeneous fields from tasks when all tasks agree.
        task_family = None
        engine = None
        level = None
        if self.tasks:
            families = {t.task_family for t in self.tasks}
            if len(families) == 1:
                task_family = families.pop()
            engines = {t.engine for t in self.tasks}
            if len(engines) == 1:
                engine = engines.pop()
            levels = [t.level for t in self.tasks]
            if levels and all(l == levels[0] for l in levels):
                level = levels[0]
        data = {
            'run_id': self.run_id,
            'project_directory': self.project_directory,
            'pipe_root': self.pipe_root,
            'status': self.status.value,
            'cluster_software': self.cluster_software,
            'max_workers': self.max_workers,
            'max_concurrent': self.max_concurrent,
            'max_attempts': self.max_attempts,
            'task_family': task_family,
            'engine': engine,
            'level': level,
            'created_at': self.created_at,
            'submitted_at': self.submitted_at,
            'completed_at': self.completed_at,
            'scheduler_job_id': self.scheduler_job_id,
        }
        tmp_path = run_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, run_path)

    @classmethod
    def from_dir(cls, pipe_root: str) -> 'PipeRun':
        """
        Reconstruct a PipeRun from an existing run directory.

        Args:
            pipe_root: Path to the pipe run root directory.

        Returns:
            PipeRun: The reconstructed run object.
        """
        run_path = os.path.join(pipe_root, 'run.json')
        with open(run_path, 'r') as f:
            data = json.load(f)
        tasks = []
        tasks_dir = os.path.join(pipe_root, 'tasks')
        if os.path.isdir(tasks_dir):
            for task_id in sorted(os.listdir(tasks_dir)):
                spec_path = os.path.join(tasks_dir, task_id, 'spec.json')
                if os.path.isfile(spec_path):
                    with open(spec_path, 'r') as f:
                        tasks.append(TaskSpec.from_dict(json.load(f)))
        project_directory = data['project_directory']
        run = cls(
            project_directory=project_directory,
            run_id=data['run_id'],
            tasks=tasks,
            cluster_software=data['cluster_software'],
            max_workers=data.get('max_workers', 100),
            max_concurrent=data.get('max_concurrent'),
            max_attempts=data.get('max_attempts', 3),
            pipe_root=pipe_root,
        )
        run.status = PipeRunState(data['status'])
        run.created_at = data.get('created_at', 0)
        run.submitted_at = data.get('submitted_at')
        run.completed_at = data.get('completed_at')
        run.scheduler_job_id = data.get('scheduler_job_id')
        return run

    def stage(self) -> None:
        """
        Create the pipe_root directory tree and initialize all tasks on disk.

        Validates that all tasks are homogeneous in ``task_family``, ``engine``,
        and ``level`` before staging. Mixed conformer runs are rejected early.
        """
        if len(self.tasks) > 1:
            ref = self.tasks[0]
            for t in self.tasks[1:]:
                if t.task_family != ref.task_family:
                    raise ValueError(f'PipeRun tasks must be homogeneous in task_family: '
                                     f'{ref.task_family} vs {t.task_family}')
                if t.engine != ref.engine:
                    raise ValueError(f'PipeRun tasks must be homogeneous in engine: '
                                     f'{ref.engine} vs {t.engine}')
                if t.level != ref.level:
                    raise ValueError(f'PipeRun tasks must be homogeneous in level: '
                                     f'{ref.level} vs {t.level}')
                if t.required_cores != ref.required_cores:
                    raise ValueError(f'PipeRun tasks must be homogeneous in required_cores: '
                                     f'{ref.required_cores} vs {t.required_cores}')
                if t.required_memory_mb != ref.required_memory_mb:
                    raise ValueError(f'PipeRun tasks must be homogeneous in required_memory_mb: '
                                     f'{ref.required_memory_mb} vs {t.required_memory_mb}')
        os.makedirs(os.path.join(self.pipe_root, 'tasks'), exist_ok=True)
        for spec in self.tasks:
            initialize_task(self.pipe_root, spec, max_attempts=self.max_attempts)
        self.status = PipeRunState.STAGED
        self._save_run_metadata()

    def _submission_resources(self):
        """
        Derive resource settings from the homogeneous task list.

        Returns:
            Tuple[int, int, int, Optional[int]]:
                ``(cpus, memory_mb, array_size, throttle)`` where ``throttle``
                caps workers running simultaneously (clamped to ``array_size``),
                or ``None`` if unthrottled.
        """
        cpus = self.tasks[0].required_cores if self.tasks else 1
        memory_mb = self.tasks[0].required_memory_mb if self.tasks else 4096
        array_size = min(self.max_workers, len(self.tasks)) if self.tasks else self.max_workers
        throttle = None
        if self.max_concurrent is not None:
            throttle = min(self.max_concurrent, array_size)
        return cpus, memory_mb, array_size, throttle

    def _render_throttle(self, array_size: int, throttle: Optional[int]) -> Dict[str, str]:
        """
        Render scheduler-specific array-range and extra-directives strings.

        SLURM/PBS encode the throttle as an inline ``%K`` suffix on the range.
        SGE uses a separate ``-tc`` directive. HTCondor uses ``max_materialize``
        and takes a bare count (not a range) for ``queue``.
        """
        cs = 'sge' if self.cluster_software == 'oge' else self.cluster_software
        if cs == 'htcondor':
            array_range = str(array_size)
            extra = f'max_materialize = {throttle}' if throttle else ''
        elif cs == 'sge':
            array_range = f'1-{array_size}'
            extra = f'#$ -tc {throttle}' if throttle else ''
        elif cs in ('slurm', 'pbs'):
            suffix = f'%{throttle}' if throttle else ''
            array_range = f'1-{array_size}{suffix}'
            extra = ''
        else:
            raise NotImplementedError(f'No throttle rendering for {self.cluster_software}')
        return {'array_range': array_range, 'extra_directives': extra}

    def write_submit_script(self) -> str:
        """
        Generate an array submission script for the configured cluster scheduler.

        Formats a template from ``arc/settings/submit.py`` (the ``pipe_submit``
        dict, keyed by cluster scheduler type) and writes it under
        ``self.pipe_root``. Rerunning safely overwrites the file.

        Returns:
            str: Absolute path to the generated submit script.
        """
        template_key = 'sge' if self.cluster_software == 'oge' else self.cluster_software
        if template_key not in pipe_submit:
            raise NotImplementedError(
                f'No pipe submit template for cluster software: {self.cluster_software}. '
                f'Available templates: {list(pipe_submit.keys())}')
        cpus, memory_mb, array_size, throttle = self._submission_resources()
        throttle_fields = self._render_throttle(array_size, throttle)
        server = servers_dict.get('local', {})
        queue, _ = next(iter(server.get('queues', {}).items()), ('', None))
        engine = self.tasks[0].engine if self.tasks else ''
        env_setup = pipe_settings.get('env_setup', {}).get(engine, '')
        scratch_base = pipe_settings.get('scratch_base', '')
        if scratch_base:
            scratch_export = f'export TMPDIR="{scratch_base}/${{PBS_JOBID%%[*}}/$PBS_ARRAY_INDEX"\nmkdir -p "$TMPDIR"'
            env_setup = f'{env_setup}\n{scratch_export}' if env_setup else scratch_export
        content = pipe_submit[template_key].format(
            name=f'pipe_{self.run_id}',
            array_range=throttle_fields['array_range'],
            extra_directives=throttle_fields['extra_directives'],
            pipe_root=self.pipe_root,
            python_exe=sys.executable,
            cpus=cpus,
            memory=memory_mb,
            queue=queue,
            env_setup=env_setup,
        )
        filename = 'submit.sub' if self.cluster_software == 'htcondor' else 'submit.sh'
        submit_path = os.path.join(self.pipe_root, filename)
        tmp_path = submit_path + '.tmp'
        with open(tmp_path, 'w') as f:
            f.write(content)
        os.replace(tmp_path, submit_path)
        # Make shell scripts executable (not HTCondor .sub files).
        if self.cluster_software != 'htcondor':
            st = os.stat(submit_path)
            os.chmod(submit_path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP)
        return submit_path

    def submit_to_scheduler(self):
        """
        Submit the generated array script to the cluster scheduler.

        Uses ``arc.job.local.submit_job`` with the cluster software mapped
        to the canonical casing expected by ``submit_command`` in settings.

        Returns:
            Tuple[str, str]: ``(job_status, job_id)`` — ``'submitted'`` on
                success, ``'errored'`` on failure.
        """
        import shutil as _shutil
        from arc.imports import settings as _settings
        submit_command = _settings['submit_command']
        # Map lowercase cluster_software to the casing used in settings.submit_command
        cs_map = {'slurm': 'Slurm', 'pbs': 'PBS', 'sge': 'OGE', 'oge': 'OGE', 'htcondor': 'HTCondor'}
        canonical_cs = cs_map.get(self.cluster_software.lower(), self.cluster_software)
        if canonical_cs not in submit_command:
            logger.warning(f'No submit command configured for {canonical_cs}. Cannot submit.')
            return 'errored', None
        cmd_path = submit_command[canonical_cs].split()[0]
        if not os.path.isfile(cmd_path) and _shutil.which(os.path.basename(cmd_path)) is None:
            logger.warning(f'Submit command {cmd_path} not found. Cannot submit pipe run.')
            return 'errored', None
        from arc.job.local import submit_job as local_submit_job
        filename = 'submit.sub' if self.cluster_software == 'htcondor' else 'submit.sh'
        job_status, job_id = local_submit_job(
            path=self.pipe_root,
            cluster_soft=canonical_cs,
            submit_filename=filename,
        )
        return job_status, job_id

    def reconcile(self) -> Dict[str, int]:
        """
        Poll all tasks, detect orphans, schedule retries, and check for completion.
        Does not regress an already-terminal run status.

        Returns:
            Dict[str, int]: Counts of tasks in each state.
        """
        if self.status in (PipeRunState.COMPLETED, PipeRunState.COMPLETED_PARTIAL, PipeRunState.FAILED):
            return self._count_task_states()

        self.status = PipeRunState.RECONCILING
        self._save_run_metadata()
        tasks_dir = os.path.join(self.pipe_root, 'tasks')
        if not os.path.isdir(tasks_dir):
            return {}

        now = time.time()
        counts: Dict[str, int] = {s.value: 0 for s in TaskState}
        retried_pending = 0  # PENDING tasks with attempt_index > 0 (genuinely retried)
        fresh_pending = 0    # PENDING tasks with attempt_index == 0 (awaiting initial workers)
        task_ids = sorted(os.listdir(tasks_dir))

        for task_id in task_ids:
            if not os.path.isdir(os.path.join(tasks_dir, task_id)):
                continue
            try:
                state = read_task_state(self.pipe_root, task_id)
            except (FileNotFoundError, ValueError, KeyError):
                continue
            current = TaskState(state.status)
            if current in (TaskState.CLAIMED, TaskState.RUNNING) \
                    and state.lease_expires_at is not None \
                    and now > state.lease_expires_at:
                try:
                    update_task_state(self.pipe_root, task_id,
                                     new_status=TaskState.ORPHANED,
                                     claimed_by=None, claim_token=None,
                                     claimed_at=None, lease_expires_at=None)
                    current = TaskState.ORPHANED
                except (ValueError, TimeoutError) as e:
                    logger.debug(f'Could not mark task {task_id} as ORPHANED '
                                 f'(another process may be handling it): {e}')
            if current == TaskState.PENDING:
                if state.attempt_index > 0:
                    retried_pending += 1
                else:
                    fresh_pending += 1
            counts[current.value] += 1

        active_workers = counts[TaskState.CLAIMED.value] + counts[TaskState.RUNNING.value]
        retryable = counts[TaskState.FAILED_RETRYABLE.value] + counts[TaskState.ORPHANED.value]
        total = sum(counts.values())

        if active_workers == 0 and retryable > 0:
            for task_id in task_ids:
                if not os.path.isdir(os.path.join(tasks_dir, task_id)):
                    continue
                try:
                    state = read_task_state(self.pipe_root, task_id)
                except (FileNotFoundError, ValueError, KeyError):
                    continue
                current = TaskState(state.status)
                if current not in (TaskState.FAILED_RETRYABLE, TaskState.ORPHANED):
                    continue
                try:
                    # FAILED_ESS tasks are handled separately (ejected to Scheduler).
                    # Only FAILED_RETRYABLE and ORPHANED reach here.
                    if state.attempt_index + 1 < state.max_attempts:
                        update_task_state(self.pipe_root, task_id,
                                          new_status=TaskState.PENDING,
                                          attempt_index=state.attempt_index + 1,
                                          claimed_by=None, claim_token=None,
                                          claimed_at=None, lease_expires_at=None,
                                          started_at=None, ended_at=None,
                                          failure_class=None, retry_disposition=None)
                        counts[current.value] -= 1
                        counts[TaskState.PENDING.value] += 1
                        retried_pending += 1
                    else:
                        ended = state.ended_at or now
                        update_task_state(self.pipe_root, task_id,
                                          new_status=TaskState.FAILED_TERMINAL,
                                          ended_at=ended)
                        counts[current.value] -= 1
                        counts[TaskState.FAILED_TERMINAL.value] += 1
                except (ValueError, TimeoutError) as e:
                    logger.debug(f'Could not promote task {task_id} to FAILED_TERMINAL '
                                 f'(lock contention or concurrent state change): {e}')

        # Only flag resubmission for genuinely retried tasks (attempt_index > 0).
        # Fresh PENDING tasks (attempt_index == 0) are waiting for the initial
        # submission's workers to start — don't resubmit for those.
        # Crucially: if fresh_pending > 0, scheduler workers are still queued
        # (PBS Q state) and will claim retried tasks when they start.
        # After a resubmission, allow a grace period for workers to start before
        # flagging again (prevents duplicate submissions).
        time_since_submit = (now - self.submitted_at) if self.submitted_at else float('inf')
        if retried_pending > 0 and active_workers == 0 \
                and fresh_pending == 0 and time_since_submit > RESUBMIT_GRACE:
            self._needs_resubmission = True
            logger.info(f'Pipe run {self.run_id}: {retried_pending} retried tasks '
                        f'need workers. Resubmission needed.')
        else:
            if retried_pending > 0 and fresh_pending > 0:
                logger.debug(f'Pipe run {self.run_id}: {retried_pending} retried tasks '
                             f'waiting, but {fresh_pending} fresh tasks still pending — '
                             f'scheduler workers still starting, skipping resubmission.')
            self._needs_resubmission = False

        terminal = (counts[TaskState.COMPLETED.value]
                    + counts[TaskState.FAILED_ESS.value]
                    + counts[TaskState.FAILED_TERMINAL.value]
                    + counts[TaskState.CANCELLED.value])

        if total > 0 and terminal == total:
            failed = (counts[TaskState.FAILED_ESS.value]
                      + counts[TaskState.FAILED_TERMINAL.value]
                      + counts[TaskState.CANCELLED.value])
            if failed > 0:
                self.status = PipeRunState.COMPLETED_PARTIAL
            else:
                self.status = PipeRunState.COMPLETED
            self.completed_at = time.time()
            self._save_run_metadata()

        return counts

    @property
    def needs_resubmission(self) -> bool:
        """Whether the run has PENDING retried tasks but no active workers."""
        return getattr(self, '_needs_resubmission', False)

    def _count_task_states(self) -> Dict[str, int]:
        """Read all task states and return counts without modifying anything."""
        counts: Dict[str, int] = {s.value: 0 for s in TaskState}
        tasks_dir = os.path.join(self.pipe_root, 'tasks')
        if not os.path.isdir(tasks_dir):
            return counts
        for task_id in sorted(os.listdir(tasks_dir)):
            if not os.path.isdir(os.path.join(tasks_dir, task_id)):
                continue
            try:
                state = read_task_state(self.pipe_root, task_id)
                counts[state.status] += 1
            except (FileNotFoundError, ValueError, KeyError):
                continue
        return counts


# ===========================================================================
# Ingestion helpers
# ===========================================================================

def find_output_file(attempt_dir: str, engine: str, task_id: str = '') -> Optional[str]:
    """
    Find the output file for a completed task.

    Prefers the ``canonical_output_path`` stored in ``result.json`` (written
    by the worker) before falling back to a filesystem walk through the
    ``calcs/`` tree.  This keeps ingestion fast and consistent with the
    worker's own output discovery.

    Returns:
        Path to the output file, or ``None`` if not found.
    """
    # 1. Prefer result.json canonical path (written by worker)
    result_path = os.path.join(attempt_dir, 'result.json')
    if os.path.isfile(result_path):
        try:
            with open(result_path) as f:
                result_data = json.load(f)
            canonical = result_data.get('canonical_output_path')
            if canonical and os.path.isfile(canonical):
                return canonical
        except (json.JSONDecodeError, OSError):
            pass  # Fall through to filesystem walk.

    # 2. Fallback: walk calcs/ tree for engine-specific output filename
    output_filenames = settings.get('output_filenames', {})
    target_name = output_filenames.get(engine, 'output.out')
    calcs_dir = os.path.join(attempt_dir, 'calcs')
    if not os.path.isdir(calcs_dir):
        logger.warning(f'Task {task_id}: no calcs/ directory in {attempt_dir} '
                       f'(engine={engine}, expected={target_name})')
        return None
    for root, dirs, files in os.walk(calcs_dir):
        if target_name in files:
            return os.path.join(root, target_name)
    logger.warning(f'Task {task_id}: {target_name} not found under {calcs_dir} '
                   f'(engine={engine})')
    return None


def _check_ess_convergence(pipe_run_id: str, spec: TaskSpec, output_file: str, label: str) -> bool:
    """
    Check whether an ESS job converged by inspecting the output file.

    Returns ``True`` if the job converged (status == 'done'), ``False`` otherwise.
    Families that don't run ESS (e.g., ts_guess_batch_method) should skip this check.
    """
    from arc.job.trsh import determine_ess_status
    try:
        status, keywords, error, line = determine_ess_status(
            output_path=output_file, species_label=label,
            job_type='opt', software=spec.engine)
    except Exception as e:
        logger.warning(f'Pipe run {pipe_run_id}, task {spec.task_id}: '
                       f'could not determine ESS status: {type(e).__name__}: {e}')
        return False
    if status != 'done':
        logger.warning(f'Pipe run {pipe_run_id}, task {spec.task_id}: '
                       f'ESS job did not converge (status={status}, keywords={keywords}). Skipping.')
        return False
    return True


def ingest_completed_task(pipe_run_id: str, pipe_root: str, spec: TaskSpec,
                          state: 'TaskStateRecord', species_dict: dict,
                          output: dict) -> None:
    """
    Ingest a single completed task, dispatched by ``task_family``.

    Called from ``Scheduler.ingest_pipe_results()`` for each completed task.
    Mutates ``species_dict`` and ``output`` in place.
    """
    label = spec.owner_key
    if not label:
        logger.warning(f'Pipe run {pipe_run_id}, task {spec.task_id}: '
                       f'missing owner_key, skipping.')
        return

    if spec.task_family in ('conf_opt', 'conf_sp'):
        if label not in species_dict:
            logger.warning(f'Pipe run {pipe_run_id}, task {spec.task_id}: '
                           f'species "{label}" not in species_dict, skipping.')
            return
        meta = spec.ingestion_metadata or {}
        conformer_index = meta.get('conformer_index')
        if conformer_index is None:
            logger.warning(f'Pipe run {pipe_run_id}, task {spec.task_id}: '
                           f'missing conformer_index in ingestion_metadata, skipping.')
            return
        if spec.task_family == 'conf_opt':
            _ingest_conf_opt(pipe_run_id, pipe_root, spec, state, species_dict, label, conformer_index)
        else:
            _ingest_conf_sp(pipe_run_id, pipe_root, spec, state, species_dict, label, conformer_index)
    elif spec.task_family == 'ts_guess_batch_method':
        _ingest_ts_guess_batch(pipe_run_id, pipe_root, spec, state, species_dict, label)
    elif spec.task_family == 'ts_opt':
        _ingest_ts_opt(pipe_run_id, pipe_root, spec, state, species_dict, label)
    elif spec.task_family == 'species_sp':
        _ingest_species_sp(pipe_run_id, pipe_root, spec, state, species_dict, label)
    elif spec.task_family == 'species_freq':
        _ingest_species_freq(pipe_run_id, pipe_root, spec, state, species_dict, label, output)
    elif spec.task_family == 'irc':
        _ingest_irc(pipe_run_id, pipe_root, spec, state, species_dict, label, output)
    elif spec.task_family == 'rotor_scan_1d':
        _ingest_rotor_scan_1d(pipe_run_id, pipe_root, spec, state, species_dict, label)


def _ingest_conf_opt(run_id, pipe_root, spec, state, species_dict, label, conformer_index):
    """Ingest a completed conf_opt task: update geometry and opt-level energy."""
    attempt_dir = get_task_attempt_dir(pipe_root, spec.task_id, state.attempt_index)
    species = species_dict[label]
    try:
        output_file = find_output_file(attempt_dir, spec.engine, spec.task_id)
        if output_file is None:
            return
        if not _check_ess_convergence(run_id, spec, output_file, label):
            return
        xyz = parser.parse_geometry(log_file_path=output_file)
        e_elect = parser.parse_e_elect(log_file_path=output_file)
    except Exception as e:
        logger.error(f'Pipe run {run_id}, task {spec.task_id}: '
                     f'parsing failed for {attempt_dir}: {type(e).__name__}: {e}')
        return
    if conformer_index < len(species.conformers) and xyz is not None:
        species.conformers[conformer_index] = xyz
    if conformer_index < len(species.conformer_energies) and e_elect is not None:
        species.conformer_energies[conformer_index] = e_elect


def _ingest_conf_sp(run_id, pipe_root, spec, state, species_dict, label, conformer_index):
    """Ingest a completed conf_sp task: update energy only."""
    attempt_dir = get_task_attempt_dir(pipe_root, spec.task_id, state.attempt_index)
    species = species_dict[label]
    try:
        output_file = find_output_file(attempt_dir, spec.engine, spec.task_id)
        if output_file is None:
            return
        if not _check_ess_convergence(run_id, spec, output_file, label):
            return
        e_elect = parser.parse_e_elect(log_file_path=output_file)
    except Exception as e:
        logger.error(f'Pipe run {run_id}, task {spec.task_id}: '
                     f'parsing failed for {attempt_dir}: {type(e).__name__}: {e}')
        return
    if conformer_index < len(species.conformer_energies) and e_elect is not None:
        species.conformer_energies[conformer_index] = e_elect


def _ingest_ts_guess_batch(run_id, pipe_root, spec, state, species_dict, label):
    if label not in species_dict:
        logger.warning(f'Pipe run {run_id}, task {spec.task_id}: '
                       f'TS species "{label}" not in species_dict, skipping.')
        return
    attempt_dir = get_task_attempt_dir(pipe_root, spec.task_id, state.attempt_index)
    try:
        output_file = find_output_file(attempt_dir, spec.engine, spec.task_id)
    except Exception as e:
        logger.error(f'Pipe run {run_id}, task {spec.task_id}: '
                     f'output lookup failed: {type(e).__name__}: {e}')
        return
    ts_species = species_dict[label]
    if output_file is not None and hasattr(ts_species, 'process_completed_tsg_queue_jobs'):
        try:
            ts_species.process_completed_tsg_queue_jobs(path=output_file)
        except Exception as e:
            logger.error(f'Pipe run {run_id}, task {spec.task_id}: '
                         f'TSG processing failed: {type(e).__name__}: {e}')


def _ingest_ts_opt(run_id, pipe_root, spec, state, species_dict, label):
    """Ingest a completed ts_opt task: update the matching TSGuess's opt_xyz and energy."""
    if label not in species_dict:
        logger.warning(f'Pipe run {run_id}, task {spec.task_id}: '
                       f'TS species "{label}" not in species_dict, skipping.')
        return
    meta = spec.ingestion_metadata or {}
    conformer_index = meta.get('conformer_index')
    if conformer_index is None:
        logger.warning(f'Pipe run {run_id}, task {spec.task_id}: '
                       f'missing conformer_index in ingestion_metadata, skipping.')
        return
    attempt_dir = get_task_attempt_dir(pipe_root, spec.task_id, state.attempt_index)
    ts_species = species_dict[label]
    try:
        output_file = find_output_file(attempt_dir, spec.engine, spec.task_id)
        if output_file is None:
            return
        if not _check_ess_convergence(run_id, spec, output_file, label):
            return
        xyz = parser.parse_geometry(log_file_path=output_file)
        e_elect = parser.parse_e_elect(log_file_path=output_file)
    except Exception as e:
        logger.error(f'Pipe run {run_id}, task {spec.task_id}: '
                     f'parsing failed for {attempt_dir}: {type(e).__name__}: {e}')
        return
    for tsg in ts_species.ts_guesses:
        if getattr(tsg, 'conformer_index', None) == conformer_index:
            if xyz is not None:
                tsg.opt_xyz = xyz
            if e_elect is not None:
                tsg.energy = e_elect
            tsg.index = conformer_index
            break
    else:
        logger.warning(f'Pipe run {run_id}, task {spec.task_id}: '
                       f'no TSGuess with conformer_index={conformer_index} for {label}.')


def _ingest_species_sp(run_id, pipe_root, spec, state, species_dict, label):
    if label not in species_dict:
        logger.warning(f'Pipe run {run_id}, task {spec.task_id}: '
                       f'species "{label}" not in species_dict, skipping.')
        return
    attempt_dir = get_task_attempt_dir(pipe_root, spec.task_id, state.attempt_index)
    species = species_dict[label]
    try:
        output_file = find_output_file(attempt_dir, spec.engine, spec.task_id)
        if output_file is None:
            return
        if not _check_ess_convergence(run_id, spec, output_file, label):
            return
        e_elect = parser.parse_e_elect(log_file_path=output_file)
    except Exception as e:
        logger.error(f'Pipe run {run_id}, task {spec.task_id}: '
                     f'parsing failed for {attempt_dir}: {type(e).__name__}: {e}')
        return
    if e_elect is not None:
        species.e_elect = e_elect


def _ingest_species_freq(run_id, pipe_root, spec, state, species_dict, label, output):
    if label not in species_dict:
        logger.warning(f'Pipe run {run_id}, task {spec.task_id}: '
                       f'species "{label}" not in species_dict, skipping.')
        return
    attempt_dir = get_task_attempt_dir(pipe_root, spec.task_id, state.attempt_index)
    try:
        output_file = find_output_file(attempt_dir, spec.engine, spec.task_id)
    except Exception as e:
        logger.error(f'Pipe run {run_id}, task {spec.task_id}: '
                     f'output lookup failed: {type(e).__name__}: {e}')
        return
    if output_file is not None and _check_ess_convergence(run_id, spec, output_file, label):
        if label not in output:
            output[label] = {'paths': {}}
        elif 'paths' not in output[label]:
            output[label]['paths'] = {}
        output[label]['paths']['freq'] = output_file


def _ingest_irc(run_id, pipe_root, spec, state, species_dict, label, output):
    if label not in species_dict:
        logger.warning(f'Pipe run {run_id}, task {spec.task_id}: '
                       f'TS species "{label}" not in species_dict, skipping.')
        return
    attempt_dir = get_task_attempt_dir(pipe_root, spec.task_id, state.attempt_index)
    try:
        output_file = find_output_file(attempt_dir, spec.engine, spec.task_id)
    except Exception as e:
        logger.error(f'Pipe run {run_id}, task {spec.task_id}: '
                     f'output lookup failed: {type(e).__name__}: {e}')
        return
    if output_file is not None and _check_ess_convergence(run_id, spec, output_file, label):
        if label not in output:
            output[label] = {'paths': {'irc': []}}
        elif 'paths' not in output[label]:
            output[label]['paths'] = {'irc': []}
        irc_paths = output[label]['paths'].get('irc', [])
        irc_paths.append(output_file)
        output[label]['paths']['irc'] = irc_paths


def _ingest_rotor_scan_1d(run_id, pipe_root, spec, state, species_dict, label):
    if label not in species_dict:
        logger.warning(f'Pipe run {run_id}, task {spec.task_id}: '
                       f'species "{label}" not in species_dict, skipping.')
        return
    attempt_dir = get_task_attempt_dir(pipe_root, spec.task_id, state.attempt_index)
    try:
        output_file = find_output_file(attempt_dir, spec.engine, spec.task_id)
    except Exception as e:
        logger.error(f'Pipe run {run_id}, task {spec.task_id}: '
                     f'output lookup failed: {type(e).__name__}: {e}')
        return
    if output_file is None:
        return
    if not _check_ess_convergence(run_id, spec, output_file, label):
        return
    meta = spec.ingestion_metadata or {}
    rotor_index = meta.get('rotor_index')
    if rotor_index is None:
        logger.warning(f'Pipe run {run_id}, task {spec.task_id}: '
                       f'missing rotor_index in ingestion_metadata for species "{label}", skipping.')
        return
    species = species_dict[label]
    if not hasattr(species, 'rotors_dict') or not isinstance(species.rotors_dict, dict):
        logger.warning(f'Pipe run {run_id}, task {spec.task_id}: '
                       f'species "{label}" has no valid rotors_dict, skipping rotor_index={rotor_index}.')
        return
    if rotor_index not in species.rotors_dict:
        logger.warning(f'Pipe run {run_id}, task {spec.task_id}: '
                       f'rotor_index={rotor_index} not found in rotors_dict for species "{label}", skipping.')
        return
    species.rotors_dict[rotor_index]['scan_path'] = output_file


# ===========================================================================
# Routing helpers
# ===========================================================================

def derive_cluster_software(ess_settings: dict, job_adapter: str) -> str:
    """
    Heuristic: derive cluster software from the first server configured
    for this engine in ess_settings. Mirrors how run_job() picks its server.

    Returns a lowercase identifier matching the ``pipe_submit`` template keys
    (e.g., ``'slurm'``, ``'pbs'``, ``'sge'``, ``'htcondor'``).
    Maps ``'oge'`` to ``'sge'`` for template compatibility.
    """
    cs_alias = {'oge': 'sge'}
    for server_name in ess_settings.get(job_adapter, []):
        if server_name in servers_dict and 'cluster_soft' in servers_dict[server_name]:
            raw = servers_dict[server_name]['cluster_soft'].lower()
            return cs_alias.get(raw, raw)
    return 'slurm'


def build_conformer_pipe_tasks(species, label: str, task_family: str,
                               level_dict: dict, job_adapter: str,
                               memory_mb: int,
                               conformer_indices: Optional[List[int]] = None,
                               ) -> List[TaskSpec]:
    """
    Build TaskSpec objects for conformer pipe tasks (conf_opt or conf_sp).

    Args:
        conformer_indices: If given, build tasks only for these indices.
            If ``None``, build tasks for all conformers.
    """
    cores = default_job_settings.get('job_cpu_cores', 8)
    species_dict_payload = species.as_dict()
    indices = conformer_indices if conformer_indices is not None else list(range(len(species.conformers)))
    tasks = []
    for i in indices:
        tasks.append(TaskSpec(
            task_id=f'{label}_{task_family}_{i}',
            task_family=task_family,
            owner_type='species',
            owner_key=label,
            input_fingerprint=f'{label}_{task_family}_{i}',
            engine=job_adapter,
            level=level_dict,
            required_cores=cores,
            required_memory_mb=memory_mb,
            input_payload={
                'species_dicts': [species_dict_payload],
                'xyz': species.conformers[i],
                'conformer': i,
            },
            ingestion_metadata={'conformer_index': i},
        ))
    return tasks


def build_species_leaf_task(species, label: str, task_family: str,
                            level_dict: dict, job_adapter: str,
                            memory_mb: int,
                            extra_ingestion: Optional[dict] = None) -> TaskSpec:
    """Build a single TaskSpec for a species-side leaf job (sp, freq, irc)."""
    cores = default_job_settings.get('job_cpu_cores', 8)
    meta = extra_ingestion or {}
    return TaskSpec(
        task_id=f'{label}_{task_family}',
        task_family=task_family,
        owner_type='species',
        owner_key=label,
        input_fingerprint=f'{label}_{task_family}',
        engine=job_adapter,
        level=level_dict,
        required_cores=cores,
        required_memory_mb=memory_mb,
        input_payload={'species_dicts': [species.as_dict()]},
        ingestion_metadata=meta,
    )


def build_tsg_tasks(ts_label: str, method: str, count: int,
                    rxn_dict: dict, memory_mb: int) -> List[TaskSpec]:
    """
    Build TaskSpec objects for one TSG method batch.

    Contract:
      - ``engine`` is set to ``method`` (the TSG method name, e.g. 'heuristics'),
        which is a registered ARC adapter — not a computational engine like 'gaussian'.
      - ``level`` is ``{'method': method}`` by convention for TSG tasks.
      - ``owner_key`` is the TS species label (not a reaction key), consistent
        with the species-ownership model used throughout the pipe system.
      - Each task represents one method-batch member for one TS species/method group.
    """
    cores = default_job_settings.get('job_cpu_cores', 8)
    tasks = []
    for i in range(count):
        tasks.append(TaskSpec(
            task_id=f'{ts_label}_tsg_{method}_{i}',
            task_family='ts_guess_batch_method',
            owner_type='species',
            owner_key=ts_label,
            input_fingerprint=f'{ts_label}_tsg_{method}_{i}',
            engine=method,
            level={'method': method},
            required_cores=cores,
            required_memory_mb=memory_mb,
            input_payload={'reactions_dicts': [rxn_dict]},
            ingestion_metadata={'tsg_index': i, 'method': method},
        ))
    return tasks


def build_ts_opt_tasks(species, label: str, xyzs: List[dict],
                       level_dict: dict, job_adapter: str,
                       memory_mb: int) -> List[TaskSpec]:
    """Build TaskSpec objects for TS optimization tasks."""
    cores = default_job_settings.get('job_cpu_cores', 8)
    species_dict_payload = species.as_dict()
    tasks = []
    for i, xyz in enumerate(xyzs):
        tasks.append(TaskSpec(
            task_id=f'{label}_ts_opt_{i}',
            task_family='ts_opt',
            owner_type='species',
            owner_key=label,
            input_fingerprint=f'{label}_ts_opt_{i}',
            engine=job_adapter,
            level=level_dict,
            required_cores=cores,
            required_memory_mb=memory_mb,
            input_payload={
                'species_dicts': [species_dict_payload],
                'xyz': xyz,
                'conformer': i,
            },
            ingestion_metadata={'conformer_index': i},
        ))
    return tasks


def build_rotor_scan_1d_tasks(species, label: str, rotor_indices: List[int],
                              level_dict: dict, job_adapter: str,
                              memory_mb: int) -> List[TaskSpec]:
    """Build TaskSpec objects for 1D rotor scan tasks."""
    cores = default_job_settings.get('job_cpu_cores', 8)
    species_dict_payload = species.as_dict()
    tasks = []
    for ri in rotor_indices:
        rotor = species.rotors_dict[ri]
        torsions = rotor['torsion']
        if isinstance(torsions[0], int):
            torsions = [torsions]
        tasks.append(TaskSpec(
            task_id=f'{label}_scan_r{ri}',
            task_family='rotor_scan_1d',
            owner_type='species',
            owner_key=label,
            input_fingerprint=f'{label}_scan_r{ri}',
            engine=job_adapter,
            level=level_dict,
            required_cores=cores,
            required_memory_mb=memory_mb,
            input_payload={
                'species_dicts': [species_dict_payload],
                'torsions': torsions,
                'rotor_index': ri,
            },
            ingestion_metadata={'rotor_index': ri},
        ))
    return tasks
