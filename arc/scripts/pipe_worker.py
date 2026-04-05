#!/usr/bin/env python3
"""
Pipe-mode worker script.

A lightweight consumer that runs inside a single slot of a SLURM/PBS/OGE/HTCondor
job array. It scans the task directory for claimable work, executes tasks
using an ARC job adapter in ``incore`` mode, and records the outcome.
The worker loops until no more PENDING tasks are available.

Usage::

    python -m arc.scripts.pipe_worker --pipe_root /path/to/pipe_run --worker_id 7
"""

import argparse
import logging
import os
import shutil
import tempfile
import time
from typing import Optional

from arc.imports import settings
from arc.job.factory import job_factory
from arc.job.pipe.pipe_state import (
    TASK_FAMILY_TO_JOB_TYPE,
    TaskState,
    TaskSpec,
    TaskStateRecord,
    generate_claim_token,
    get_task_attempt_dir,
    read_task_spec,
    read_task_state,
    update_task_state,
    write_result_json,
)
from arc.level import Level
from arc.reaction import ARCReaction
from arc.species import ARCSpecies

pipe_settings, output_filenames = settings['pipe_settings'], settings.get('output_filenames', {})


logger = logging.getLogger('pipe_worker')


def setup_logging(log_path: str) -> None:
    """Configure logging. Safe to call multiple times."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(stderr_handler)
    logger.setLevel(logging.INFO)


def claim_task(pipe_root: str, worker_id: str):
    """
    Scan for a PENDING task and attempt to claim it.
    Returns ``(task_id, TaskStateRecord, claim_token)`` on success,
    or ``(None, None, None)`` if no tasks are available.
    """
    tasks_dir = os.path.join(pipe_root, 'tasks')
    if not os.path.isdir(tasks_dir):
        return None, None, None
    for task_id in sorted(os.listdir(tasks_dir)):
        if not os.path.isdir(os.path.join(tasks_dir, task_id)):
            continue
        try:
            state = read_task_state(pipe_root, task_id)
            current_status = TaskState(state.status)
        except (FileNotFoundError, ValueError, KeyError):
            continue  # Skip tasks with unreadable or corrupted state.
        if current_status != TaskState.PENDING:
            continue
        try:
            now = time.time()
            token = generate_claim_token()
            updated = update_task_state(pipe_root, task_id,
                                        new_status=TaskState.CLAIMED,
                                        claimed_by=worker_id,
                                        claim_token=token,
                                        claimed_at=now,
                                        lease_expires_at=now + pipe_settings.get('lease_duration_s', 86400))
            logger.info(f'Claimed task {task_id}')
            return task_id, updated, token
        except (ValueError, TimeoutError):
            continue
    return None, None, None


def run_task(pipe_root: str, task_id: str, state: TaskStateRecord,
             worker_id: str, claim_token: str) -> None:
    """
    Execute a claimed task: transition to RUNNING, dispatch by task_family,
    copy outputs, write result.json, and mark COMPLETED or FAILED.
    """
    attempt_dir = get_task_attempt_dir(pipe_root, task_id, state.attempt_index)
    os.makedirs(attempt_dir, exist_ok=True)
    setup_logging(os.path.join(attempt_dir, 'worker.log'))

    started_at = time.time()
    try:
        update_task_state(pipe_root, task_id, new_status=TaskState.RUNNING, started_at=started_at)
    except (ValueError, TimeoutError) as e:
        logger.warning(f'Task {task_id}: could not transition to RUNNING ({e}), skipping.')
        return

    spec = read_task_spec(pipe_root, task_id)
    scratch_dir = tempfile.mkdtemp(prefix=f'pipe_{task_id}_')
    result = _make_result_template(task_id, state.attempt_index, started_at)
    try:
        _dispatch_execution(spec, scratch_dir)
        _copy_outputs(scratch_dir, attempt_dir)
        ended_at = time.time()
        result['ended_at'] = ended_at
        result['status'] = 'COMPLETED'
        result['canonical_output_path'] = _find_canonical_output(attempt_dir, spec.engine)
        write_result_json(attempt_dir, result)
        if not _verify_ownership(pipe_root, task_id, worker_id, claim_token):
            return
        try:
            update_task_state(pipe_root, task_id, new_status=TaskState.COMPLETED, ended_at=ended_at)
        except (ValueError, TimeoutError) as e:
            logger.warning(f'Task {task_id}: could not mark COMPLETED ({e}). '
                           f'Task may have been orphaned concurrently.')
            return
        logger.info(f'Task {task_id} completed successfully')
    except Exception as e:
        failure_class = type(e).__name__
        ended_at = time.time()
        logger.error(f'Task {task_id} failed: {failure_class}: {e}')
        _copy_outputs(scratch_dir, attempt_dir)
        result['ended_at'] = ended_at
        result['status'] = 'FAILED'
        result['failure_class'] = failure_class
        write_result_json(attempt_dir, result)
        if not _verify_ownership(pipe_root, task_id, worker_id, claim_token):
            return
        try:
            current_state = read_task_state(pipe_root, task_id)
            target = TaskState.FAILED_RETRYABLE if current_state.attempt_index + 1 < current_state.max_attempts \
                else TaskState.FAILED_TERMINAL
            update_task_state(pipe_root, task_id, new_status=target,
                              ended_at=ended_at, failure_class=failure_class)
        except (ValueError, TimeoutError) as e:
            logger.warning(f'Task {task_id}: could not mark failed ({e}). '
                           f'Task may have been orphaned concurrently.')
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def _make_result_template(task_id: str, attempt_index: int, started_at: float) -> dict:
    return {
        'task_id': task_id,
        'attempt_index': attempt_index,
        'started_at': started_at,
        'ended_at': None,
        'status': None,
        'canonical_output_path': None,
        'exit_code': None,
        'failure_class': None,
        'parser_summary': None,
        'result_fields': {},
    }


# ---------------------------------------------------------------------------
# Task-family execution dispatch
# ---------------------------------------------------------------------------

def _get_family_extra_kwargs(spec: TaskSpec) -> dict:
    """
    Extract family-specific kwargs needed by the adapter beyond the base job_type.

    The adapter-facing job_type comes from TASK_FAMILY_TO_JOB_TYPE (the central
    mapping in pipe_state.py). This helper supplies only the extra parameters
    that certain families need (e.g. irc_direction, torsions, rotor_index).
    """
    kwargs = {}
    payload = spec.input_payload or {}
    meta = spec.ingestion_metadata or {}

    if spec.task_family == 'irc':
        irc_direction = meta.get('irc_direction')
        if irc_direction:
            kwargs['irc_direction'] = irc_direction
    elif spec.task_family == 'rotor_scan_1d':
        torsions = payload.get('torsions')
        rotor_index = payload.get('rotor_index')
        if torsions is not None:
            kwargs['torsions'] = torsions
        if rotor_index is not None:
            kwargs['rotor_index'] = rotor_index

    return kwargs


def _dispatch_execution(spec: TaskSpec, scratch_dir: str) -> None:
    """
    Dispatch execution by task_family.

    The adapter-facing job_type is derived from the central
    ``TASK_FAMILY_TO_JOB_TYPE`` mapping in ``pipe_state.py``.
    Family-specific extra kwargs (e.g. irc_direction, torsions)
    are extracted by ``_get_family_extra_kwargs``.
    """
    job_type = TASK_FAMILY_TO_JOB_TYPE.get(spec.task_family)
    if job_type is None:
        raise ValueError(f'Unsupported task_family for execution: {spec.task_family}')
    extra = _get_family_extra_kwargs(spec)
    _run_adapter(spec, scratch_dir, job_type=job_type, **extra)


def _run_adapter(spec: TaskSpec, scratch_dir: str, job_type: str, **extra_kwargs) -> None:
    """
    Reconstruct ARC objects and run the adapter incore with the given job_type.

    Args:
        spec: The task specification.
        scratch_dir: Temporary working directory for the adapter.
        job_type: The adapter-facing job type (e.g. 'sp', 'freq', 'irc').
        **extra_kwargs: Additional keyword arguments passed to job_factory
                        (e.g. ``irc_direction`` for IRC jobs).
    """
    species_list = None
    reactions_list = None
    payload = spec.input_payload or {}
    species_dicts = payload.get('species_dicts')
    reactions_dicts = payload.get('reactions_dicts')
    if species_dicts:
        species_list = [ARCSpecies(species_dict=_fix_int_keys(d)) for d in species_dicts]
    if reactions_dicts:
        reactions_list = [ARCReaction(reaction_dict=_fix_int_keys(d)) for d in reactions_dicts]
    level_info = spec.level
    if not level_info:
        raise ValueError(f'Task {spec.task_id}: missing level information')
    level = Level(repr=level_info)
    # Pass per-task xyz and conformer/tsg index from input_payload so
    # each task operates on its specific geometry, not the species default.
    xyz = payload.get('xyz')
    conformer = payload.get('conformer')
    tsg = payload.get('tsg')
    job = job_factory(
        job_adapter=spec.engine,
        execution_type='incore',
        project='pipe_run',
        project_directory=scratch_dir,
        job_type=job_type,
        level=level,
        species=species_list,
        reactions=reactions_list,
        xyz=xyz,
        conformer=conformer,
        tsg=tsg,
        testing=False,
        **extra_kwargs,
    )
    job.execute()
    output_file = getattr(job, 'local_path_to_output_file', None)
    if output_file and not os.path.isfile(output_file):
        raise RuntimeError(f'{spec.engine} produced no output file at {output_file}. '
                           f'The engine may not be installed or configured on this node.')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verify_ownership(pipe_root: str, task_id: str,
                      worker_id: str, claim_token: str) -> bool:
    """
    Verify this worker still owns the task.

    Checks claimed_by, claim_token, AND that the current status is still
    RUNNING or CLAIMED (not ORPHANED by the coordinator due to lease expiry).
    """
    try:
        current = read_task_state(pipe_root, task_id)
    except (FileNotFoundError, ValueError, KeyError):
        logger.warning(f'Task {task_id}: could not read state for ownership check')
        return False
    if current.claimed_by != worker_id or current.claim_token != claim_token:
        logger.warning(f'Task {task_id}: ownership lost '
                       f'(claimed_by={current.claimed_by}, token={current.claim_token}, '
                       f'expected={worker_id}/{claim_token}). Not writing terminal state.')
        return False
    current_status = TaskState(current.status)
    if current_status not in (TaskState.RUNNING, TaskState.CLAIMED):
        logger.warning(f'Task {task_id}: status is {current_status.value} (expected RUNNING/CLAIMED). '
                       f'Task may have been orphaned. Not writing terminal state.')
        return False
    return True


def _find_canonical_output(attempt_dir: str, engine: str) -> Optional[str]:
    """Try to find the canonical output file path within the attempt calcs tree."""
    target = output_filenames.get(engine, 'output.out')
    calcs_dir = os.path.join(attempt_dir, 'calcs')
    if os.path.isdir(calcs_dir):
        for root, dirs, files in os.walk(calcs_dir):
            if target in files:
                return os.path.join(root, target)
    return None


def _fix_int_keys(obj):
    """Recursively convert string dict keys that represent integers back to int."""
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            try:
                k = int(k)
            except (ValueError, TypeError):
                pass  # Key is not numeric; keep it as a string.
            new[k] = _fix_int_keys(v)
        return new
    elif isinstance(obj, list):
        return [_fix_int_keys(x) for x in obj]
    return obj


def _copy_outputs(src_dir: str, dst_dir: str) -> None:
    calcs_dir = os.path.join(src_dir, 'calcs')
    if not os.path.isdir(calcs_dir):
        return
    shutil.copytree(calcs_dir, os.path.join(dst_dir, 'calcs'), dirs_exist_ok=True)


def main(argv=None):
    """Entry point. Loops claiming and executing PENDING tasks until none remain."""
    parser = argparse.ArgumentParser(description='Pipe-mode worker: claim and execute tasks.')
    parser.add_argument('--pipe_root', required=True, help='Root directory of the pipe run.')
    parser.add_argument('--worker_id', required=True, help='Worker identifier.')
    args = parser.parse_args(argv)

    tasks_completed = 0
    while True:
        task_id, state, token = claim_task(args.pipe_root, args.worker_id)
        if task_id is None:
            break
        run_task(args.pipe_root, task_id, state, args.worker_id, token)
        tasks_completed += 1

    if tasks_completed == 0:
        print('No claimable tasks found. Exiting.')
    else:
        print(f'Worker {args.worker_id} completed {tasks_completed} task(s). No more work remaining.')


if __name__ == '__main__':
    main()
