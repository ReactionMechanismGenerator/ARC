"""
A module for pipe-mode task state management.

Defines the state machines, data models, and filesystem I/O utilities for
orchestrating subjobs within a single SLURM/PBS/HTCondor array allocation.
All task metadata is persisted as JSON files under a structured directory
tree, with file-level locking for safe concurrent access from multiple
worker processes.

Directory layout for a task::

    <pipe_root>/
        tasks/
            <task_id>/
                spec.json          # immutable task specification
                state.json         # mutable state record (locked on update)
                state.json.lock    # lock file for state.json
                attempts/
                    <attempt_index>/
                        result.json    # worker-written result metadata
                        calcs/         # preserved adapter output tree
                        worker.log     # per-attempt log
"""

import fcntl
import json
import os
import time
import uuid
from enum import Enum

class TaskState(str, Enum):
    """
    States for an individual task within a pipe run.

    Lifecycle::

        PENDING ──► CLAIMED ──► RUNNING ──► COMPLETED
                        │           │
                        │           ├──► FAILED_RETRYABLE ──► PENDING (retry)
                        │           │                     └──► FAILED_TERMINAL
                        │           ├──► FAILED_ESS ──► (ejected to Scheduler)
                        │           └──► ORPHANED ──► PENDING (retry)
                        └──► ORPHANED

    PENDING:            Awaiting a worker. Fresh tasks start here (attempt_index=0).
                        Retried tasks return here with attempt_index incremented.
    CLAIMED:            A worker has claimed this task (file-locked).
    RUNNING:            The worker is executing the ESS adapter.
    COMPLETED:          ESS ran and converged successfully. Results ready for ingestion.
    FAILED_RETRYABLE:   Transient failure (node crash, NoOutput, disk issue).
                        Will be retried with identical input on a different node.
    FAILED_ESS:         Deterministic ESS error (SCF, MaxOptCycles, InternalCoordinateError).
                        Blind retry won't help — ejected to Scheduler for troubleshooting
                        with modified input (different algorithm, keywords, etc.).
    FAILED_TERMINAL:    Exhausted all retry attempts. No further action by pipe system.
    ORPHANED:           Worker lease expired (likely killed by PBS). Reset to PENDING.
    CANCELLED:          Manually cancelled. Terminal state.
    """
    PENDING = 'PENDING'
    CLAIMED = 'CLAIMED'
    RUNNING = 'RUNNING'
    COMPLETED = 'COMPLETED'
    FAILED_RETRYABLE = 'FAILED_RETRYABLE'
    FAILED_ESS = 'FAILED_ESS'
    FAILED_TERMINAL = 'FAILED_TERMINAL'
    ORPHANED = 'ORPHANED'
    CANCELLED = 'CANCELLED'

class PipeRunState(str, Enum):
    """States for the overall pipe run."""
    CREATED = 'CREATED'
    STAGED = 'STAGED'
    SUBMITTED = 'SUBMITTED'
    ACTIVE = 'ACTIVE'
    RECONCILING = 'RECONCILING'
    COMPLETED = 'COMPLETED'
    COMPLETED_PARTIAL = 'COMPLETED_PARTIAL'
    FAILED = 'FAILED'

# Task families currently supported by the pipe system.
# Only families listed here pass TaskSpec validation.
SUPPORTED_TASK_FAMILIES = (
    'conf_opt', 'conf_sp',
    'ts_guess_batch_method', 'ts_opt',
    'species_sp', 'species_freq', 'irc',
    'rotor_scan_1d',
)

# Owner types mapping to ARC object categories.
SUPPORTED_OWNER_TYPES = ('species', 'reaction')

# Mapping from task_family to the adapter-facing job_type.
# Kept explicit so that task_family is not blindly used as job_type.
TASK_FAMILY_TO_JOB_TYPE = {
    'conf_opt': 'conf_opt',
    'conf_sp': 'conf_sp',
    'ts_guess_batch_method': 'tsg',
    'ts_opt': 'opt',
    'species_sp': 'sp',
    'species_freq': 'freq',
    'irc': 'irc',
    'rotor_scan_1d': 'scan',
}

# Allowed transitions: maps each state to the set of states it may transition to.
TASK_TRANSITIONS: dict[TaskState, tuple[TaskState, ...]] = {
    TaskState.PENDING: (TaskState.CLAIMED, TaskState.CANCELLED),
    TaskState.CLAIMED: (TaskState.RUNNING, TaskState.ORPHANED, TaskState.CANCELLED),
    TaskState.RUNNING: (TaskState.COMPLETED, TaskState.FAILED_RETRYABLE, TaskState.FAILED_ESS,
                        TaskState.FAILED_TERMINAL, TaskState.ORPHANED, TaskState.CANCELLED),
    TaskState.COMPLETED: (),
    TaskState.FAILED_RETRYABLE: (TaskState.PENDING, TaskState.FAILED_TERMINAL),
    TaskState.FAILED_ESS: (),  # Terminal within pipe — ejected to Scheduler for troubleshooting.
    TaskState.FAILED_TERMINAL: (),
    TaskState.ORPHANED: (TaskState.PENDING, TaskState.FAILED_TERMINAL),
    TaskState.CANCELLED: (),
}

PIPE_RUN_TRANSITIONS: dict[PipeRunState, tuple[PipeRunState, ...]] = {
    PipeRunState.CREATED: (PipeRunState.STAGED, PipeRunState.FAILED),
    PipeRunState.STAGED: (PipeRunState.SUBMITTED, PipeRunState.FAILED),
    PipeRunState.SUBMITTED: (PipeRunState.ACTIVE, PipeRunState.FAILED),
    PipeRunState.ACTIVE: (PipeRunState.RECONCILING, PipeRunState.FAILED),
    PipeRunState.RECONCILING: (PipeRunState.COMPLETED, PipeRunState.COMPLETED_PARTIAL, PipeRunState.FAILED),
    PipeRunState.COMPLETED: (),
    PipeRunState.COMPLETED_PARTIAL: (),
    PipeRunState.FAILED: (),
}

def check_valid_transition(current_state: TaskState | PipeRunState,
                           new_state: TaskState | PipeRunState,
                           ) -> None:
    """
    Validate that a state transition is allowed.

    Args:
        current_state: The current state.
        new_state: The proposed new state.

    Raises:
        ValueError: If the transition is not allowed.
        TypeError: If the two states are not of the same enum type.
    """
    if type(current_state) is not type(new_state):
        raise TypeError(f'Cannot transition between different state types: '
                        f'{type(current_state).__name__} -> {type(new_state).__name__}')
    if isinstance(current_state, TaskState):
        allowed = TASK_TRANSITIONS
    elif isinstance(current_state, PipeRunState):
        allowed = PIPE_RUN_TRANSITIONS
    else:
        raise TypeError(f'Unsupported state type: {type(current_state).__name__}')
    if new_state not in allowed[current_state]:
        raise ValueError(f'Invalid state transition: {current_state.value} -> {new_state.value}')

def _validate_task_spec(spec: 'TaskSpec') -> None:
    """
    Validate required fields on a TaskSpec.

    Raises:
        ValueError: If any required field is missing or invalid.
    """
    if not spec.task_family:
        raise ValueError('TaskSpec.task_family is required')
    if spec.task_family not in SUPPORTED_TASK_FAMILIES:
        raise ValueError(f'TaskSpec.task_family must be one of {SUPPORTED_TASK_FAMILIES}, '
                         f'got {spec.task_family!r}')
    if not spec.owner_type:
        raise ValueError('TaskSpec.owner_type is required')
    if spec.owner_type not in SUPPORTED_OWNER_TYPES:
        raise ValueError(f'TaskSpec.owner_type must be one of {SUPPORTED_OWNER_TYPES}, '
                         f'got {spec.owner_type!r}')
    if not spec.owner_key:
        raise ValueError('TaskSpec.owner_key is required')
    if spec.level is None:
        raise ValueError('TaskSpec.level is required')
    if spec.input_payload is None:
        raise ValueError('TaskSpec.input_payload is required')
    if spec.ingestion_metadata is None:
        raise ValueError('TaskSpec.ingestion_metadata is required')

class TaskSpec:
    """
    Immutable specification for a single pipe task.

    Written once to ``spec.json`` and never modified.

    Args:
        task_id (str): Unique identifier for this task.
        task_family (str): Pipe task family (e.g. ``'conf_opt'``, ``'conf_sp'``).
        owner_type (str): Owner kind — ``'species'`` or ``'reaction'``.
        owner_key (str): Stable key identifying the owning ARC object.
        input_fingerprint (str): Hash or fingerprint of the input for deduplication.
        engine (str): Computational engine (e.g. ``'gaussian'``, ``'orca'``).
        level (dict): Level-of-theory payload (``Level.as_dict()`` output).
        required_cores (int): Number of CPU cores required.
        required_memory_mb (int): Memory requirement in MB.
        input_payload (dict): Task-family-specific execution inputs.
        ingestion_metadata (dict): Task-family-specific data for reattaching results.
        args (dict, optional): Legacy/extra arguments.
    """

    def __init__(self,
                 task_id: str,
                 task_family: str,
                 owner_type: str,
                 owner_key: str,
                 input_fingerprint: str,
                 engine: str,
                 level: dict,
                 required_cores: int,
                 required_memory_mb: int,
                 input_payload: dict,
                 ingestion_metadata: dict,
                 args: dict | None = None,
                 ):
        self.task_id = task_id
        self.task_family = task_family
        self.owner_type = owner_type
        self.owner_key = owner_key
        self.input_fingerprint = input_fingerprint
        self.engine = engine
        self.level = level
        self.required_cores = required_cores
        self.required_memory_mb = required_memory_mb
        self.input_payload = input_payload
        self.ingestion_metadata = ingestion_metadata
        self.args = args or {}
        _validate_task_spec(self)

    def as_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        return {
            'task_id': self.task_id,
            'task_family': self.task_family,
            'owner_type': self.owner_type,
            'owner_key': self.owner_key,
            'input_fingerprint': self.input_fingerprint,
            'engine': self.engine,
            'level': self.level,
            'required_cores': self.required_cores,
            'required_memory_mb': self.required_memory_mb,
            'input_payload': self.input_payload,
            'ingestion_metadata': self.ingestion_metadata,
            'args': self.args,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TaskSpec':
        """
        Reconstruct a TaskSpec from a dictionary.

        Bypasses validation so that specs already persisted on disk can be
        read back even if the supported-families list has changed.

        Contract:
          - **Producers** (``build_*_tasks`` helpers) must construct valid specs
            through ``__init__``, which enforces validation.
          - **Deserializers** (this method) read persisted specs leniently so
            that evolving family definitions don't break restart.
          - **Execution/routing** paths (worker dispatch, ingestion) must still
            fail safely if a task_family is unsupported at runtime.
        """
        obj = object.__new__(cls)
        obj.task_id = d['task_id']
        obj.task_family = d['task_family']
        obj.owner_type = d['owner_type']
        obj.owner_key = d['owner_key']
        obj.input_fingerprint = d['input_fingerprint']
        obj.engine = d['engine']
        obj.level = d['level']
        obj.required_cores = d['required_cores']
        obj.required_memory_mb = d['required_memory_mb']
        obj.input_payload = d['input_payload']
        obj.ingestion_metadata = d['ingestion_metadata']
        obj.args = d.get('args', {})
        return obj

class TaskStateRecord:
    """
    Mutable state record for a single pipe task.

    Persisted in ``state.json`` and updated under a file lock.

    Args:
        status (str): Current task state (a TaskState value).
        attempt_index (int): Current attempt number (0-indexed).
        max_attempts (int): Maximum allowed attempts before terminal failure.
        claimed_by (str, optional): Worker identifier that claimed this task.
        claimed_at (float, optional): Timestamp (epoch seconds) when claimed.
        lease_expires_at (float, optional): Timestamp when the lease expires.
        started_at (float, optional): Timestamp when execution started.
        ended_at (float, optional): Timestamp when execution ended.
        failure_class (str, optional): Classification of the failure (e.g. 'oom', 'timeout', 'ess_error').
        retry_disposition (str, optional): How the retry was decided (e.g. 'auto', 'manual').
    """

    def __init__(self,
                 status: str = TaskState.PENDING.value,
                 attempt_index: int = 0,
                 max_attempts: int = 3,
                 claimed_by: str | None = None,
                 claim_token: str | None = None,
                 claimed_at: float | None = None,
                 lease_expires_at: float | None = None,
                 started_at: float | None = None,
                 ended_at: float | None = None,
                 failure_class: str | None = None,
                 retry_disposition: str | None = None,
                 ):
        self.status = status
        self.attempt_index = attempt_index
        self.max_attempts = max_attempts
        self.claimed_by = claimed_by
        self.claim_token = claim_token
        self.claimed_at = claimed_at
        self.lease_expires_at = lease_expires_at
        self.started_at = started_at
        self.ended_at = ended_at
        self.failure_class = failure_class
        self.retry_disposition = retry_disposition

    def as_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        return {
            'status': self.status,
            'attempt_index': self.attempt_index,
            'max_attempts': self.max_attempts,
            'claimed_by': self.claimed_by,
            'claim_token': self.claim_token,
            'claimed_at': self.claimed_at,
            'lease_expires_at': self.lease_expires_at,
            'started_at': self.started_at,
            'ended_at': self.ended_at,
            'failure_class': self.failure_class,
            'retry_disposition': self.retry_disposition,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TaskStateRecord':
        """Reconstruct a TaskStateRecord from a dictionary."""
        return cls(
            status=d['status'],
            attempt_index=d['attempt_index'],
            max_attempts=d['max_attempts'],
            claimed_by=d.get('claimed_by'),
            claim_token=d.get('claim_token'),
            claimed_at=d.get('claimed_at'),
            lease_expires_at=d.get('lease_expires_at'),
            started_at=d.get('started_at'),
            ended_at=d.get('ended_at'),
            failure_class=d.get('failure_class'),
            retry_disposition=d.get('retry_disposition'),
        )

def generate_claim_token() -> str:
    """Generate a unique claim token for ownership verification."""
    return uuid.uuid4().hex[:16]

# ---------------------------------------------------------------------------
# Directory & I/O Utilities
# ---------------------------------------------------------------------------

def get_task_dir(pipe_root: str, task_id: str) -> str:
    """
    Get the directory path for a task.

    Args:
        pipe_root (str): Root directory of the pipe run.
        task_id (str): The task identifier.

    Returns:
        str: Absolute path to the task directory.
    """
    return os.path.join(pipe_root, 'tasks', task_id)

def get_task_attempt_dir(pipe_root: str, task_id: str, attempt_index: int) -> str:
    """
    Get the working directory for a specific attempt of a task.

    Args:
        pipe_root (str): Root directory of the pipe run.
        task_id (str): The task identifier.
        attempt_index (int): The 0-indexed attempt number.

    Returns:
        str: Absolute path to the attempt directory.
    """
    return os.path.join(pipe_root, 'tasks', task_id, 'attempts', str(attempt_index))

def initialize_task(pipe_root: str, spec: TaskSpec, max_attempts: int = 3,
                    overwrite: bool = False) -> str:
    """
    Create the directory structure and initial files for a new task.

    Args:
        pipe_root: Root directory of the pipe run.
        spec: The task specification.
        max_attempts: Maximum retry attempts.
        overwrite: If False, raise FileExistsError if the task already exists.

    Returns:
        str: Path to the created task directory.
    """
    task_dir = get_task_dir(pipe_root, spec.task_id)
    spec_path = os.path.join(task_dir, 'spec.json')
    state_path = os.path.join(task_dir, 'state.json')
    if not overwrite and (os.path.isfile(spec_path) or os.path.isfile(state_path)):
        raise FileExistsError(f'Task {spec.task_id} already initialized at {task_dir}')
    os.makedirs(os.path.join(task_dir, 'attempts'), exist_ok=True)
    with open(spec_path, 'w') as f:
        json.dump(spec.as_dict(), f, indent=2)
    state = TaskStateRecord(max_attempts=max_attempts)
    with open(state_path, 'w') as f:
        json.dump(state.as_dict(), f, indent=2)
    return task_dir

def read_task_spec(pipe_root: str, task_id: str) -> TaskSpec:
    """
    Read the immutable task specification from disk.

    Args:
        pipe_root (str): Root directory of the pipe run.
        task_id (str): The task identifier.

    Returns:
        TaskSpec: The deserialized task specification.
    """
    spec_path = os.path.join(get_task_dir(pipe_root, task_id), 'spec.json')
    with open(spec_path, 'r') as f:
        return TaskSpec.from_dict(json.load(f))

def read_task_state(pipe_root: str, task_id: str) -> TaskStateRecord:
    """
    Read the current task state from disk.

    Args:
        pipe_root (str): Root directory of the pipe run.
        task_id (str): The task identifier.

    Returns:
        TaskStateRecord: The deserialized task state.
    """
    state_path = os.path.join(get_task_dir(pipe_root, task_id), 'state.json')
    with open(state_path, 'r') as f:
        return TaskStateRecord.from_dict(json.load(f))

def write_result_json(attempt_dir: str, result: dict) -> str:
    """Write a ``result.json`` file in the attempt directory. Returns the path."""
    result_path = os.path.join(attempt_dir, 'result.json')
    tmp_path = result_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, result_path)
    return result_path

def _validate_state_invariants(state: TaskStateRecord) -> None:
    """Validate lightweight invariants on a TaskStateRecord before persisting."""
    if state.attempt_index < 0:
        raise ValueError(f'attempt_index must be >= 0, got {state.attempt_index}')
    if state.max_attempts < 1:
        raise ValueError(f'max_attempts must be >= 1, got {state.max_attempts}')
    status = TaskState(state.status)
    if status == TaskState.CLAIMED:
        if state.claimed_by is None:
            raise ValueError('Transition to CLAIMED requires claimed_by')
        if state.claim_token is None:
            raise ValueError('Transition to CLAIMED requires claim_token')
        if state.claimed_at is None:
            raise ValueError('Transition to CLAIMED requires claimed_at')
        if state.lease_expires_at is None:
            raise ValueError('Transition to CLAIMED requires lease_expires_at')
    if status == TaskState.RUNNING:
        if state.started_at is None:
            raise ValueError('Transition to RUNNING requires started_at')
    if status in (TaskState.COMPLETED, TaskState.FAILED_TERMINAL, TaskState.CANCELLED):
        if state.ended_at is None:
            raise ValueError(f'Transition to {status.value} requires ended_at')
    if state.lease_expires_at is not None and state.claimed_at is not None:
        if state.lease_expires_at < state.claimed_at:
            raise ValueError(f'lease_expires_at ({state.lease_expires_at}) '
                             f'must be >= claimed_at ({state.claimed_at})')

def update_task_state(pipe_root: str,
                      task_id: str,
                      new_status: TaskState | None = None,
                      lock_timeout: float = 30.0,
                      **fields,
                      ) -> TaskStateRecord:
    """
    Atomically update a task's state record under a file lock.

    Acquires an exclusive lock on ``state.json.lock``, reads the current state,
    validates any status transition and field invariants, applies updates, and
    writes the result atomically (write to temp file, then rename).

    Args:
        pipe_root (str): Root directory of the pipe run.
        task_id (str): The task identifier.
        new_status (TaskState, optional): If provided, transition to this status
                                          (validated against allowed transitions).
        lock_timeout (float): Maximum seconds to wait for the lock.
        **fields: Additional fields to update on the TaskStateRecord
                  (e.g., ``claimed_by='worker-3'``, ``lease_expires_at=1234567890.0``).

    Returns:
        TaskStateRecord: The updated state record.

    Raises:
        ValueError: If the state transition or field invariants are invalid.
        TimeoutError: If the lock cannot be acquired within ``lock_timeout``.
    """
    task_dir = get_task_dir(pipe_root, task_id)
    state_path = os.path.join(task_dir, 'state.json')
    lock_path = state_path + '.lock'
    lock_fd = open(lock_path, 'w')
    try:
        _acquire_lock(lock_fd, lock_timeout)
        with open(state_path, 'r') as f:
            state = TaskStateRecord.from_dict(json.load(f))
        if new_status is not None:
            current = TaskState(state.status)
            check_valid_transition(current, new_status)
            state.status = new_status.value
        valid_fields = set(TaskStateRecord().__dict__.keys()) - {'status'}
        for key, value in fields.items():
            if key not in valid_fields:
                raise ValueError(f'Unknown TaskStateRecord field: {key}')
            setattr(state, key, value)
        _validate_state_invariants(state)
        tmp_path = state_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(state.as_dict(), f, indent=2)
        os.replace(tmp_path, state_path)
        return state
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

def _acquire_lock(lock_fd, timeout: float) -> None:
    """
    Acquire an exclusive file lock with a timeout.

    Args:
        lock_fd: Open file descriptor for the lock file.
        timeout (float): Maximum seconds to wait.

    Raises:
        TimeoutError: If the lock is not acquired within the timeout.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except (OSError, BlockingIOError):
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Could not acquire lock within {timeout}s')
            time.sleep(0.10)
