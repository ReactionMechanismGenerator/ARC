"""
A module deciding whether a queue status command (``squeue``, ``qstat``, ``condor_q``) was answered.

A command that fails on the cluster, e.g., Slurm's "Socket timed out on send/recv operation", prints nothing on
stdout, which is also what an empty queue looks like. Reading that as "no jobs running" ends every live job, so a
failed query is retried here, after each delay in ``QUEUE_QUERY_RETRY_DELAYS`` (seconds), and then reported as
unknown, never as empty. The local and the SSH paths both query through ``run_queue_query()``.

The command's exit status decides: 0 is an answer, whatever stderr says (many servers print a module warning or a
login banner on every command), and any other status is a failure. The shell's 127 is a missing command. Only a
result that carries no exit status falls back to reading stdout and stderr.
"""

import math
import time
from collections.abc import Callable

from arc.common import get_logger
from arc.exceptions import QueueQueryError
from arc.imports import settings


logger = get_logger()

QUEUE_QUERY_RETRY_DELAYS = (5, 15)
QUEUE_OUTAGE_LIMIT_HRS = 4
QUEUE_QUERY_TIMEOUT = 120
MISSING_COMMAND_EXIT_STATUS = 127


def queue_query_failed(stdout: list[str] | None,
                       stderr: list[str] | None,
                       exit_status: int | None = None,
                       missing_command_reads_empty: bool = False,
                       ) -> bool:
    """
    Decide whether a queue status command failed, as opposed to answering with an empty queue.

    An exit status of 0 is an answer and any other is a failure. A missing queue command (exit status 127) may be
    read as no jobs instead: machines without a queue, and incore-only runs, have always read it so on the local
    machine. Where no exit status exists, the query failed if it printed nothing on stdout and an error on stderr.

    Args:
        stdout (list[str] | None): The stdout lines of the queue status command.
        stderr (list[str] | None): The stderr lines of the queue status command.
        exit_status (int | None, optional): The command's exit status, ``None`` where the transport has none.
        missing_command_reads_empty (bool, optional): Whether a missing queue command reads as an empty queue.

    Returns:
        bool: Whether the query failed.
    """
    if exit_status is not None:
        if exit_status == MISSING_COMMAND_EXIT_STATUS:
            return not missing_command_reads_empty
        return exit_status != 0
    if any(line.strip() for line in stdout or list()):
        return False
    return any(line.strip() for line in stderr or list())


def run_queue_query(run: Callable[[], tuple[list | None, list | None, int | None]],
                    server: str,
                    retry_delays: tuple[float, ...] = QUEUE_QUERY_RETRY_DELAYS,
                    missing_command_reads_empty: bool = False,
                    on_missing_command: Callable[[], None] | None = None,
                    ) -> list[str]:
    """
    Run a queue status command, retrying it with a bounded backoff while it fails.

    An ``OSError`` raised while running the command, e.g., a failed fork, is a failed attempt.

    Args:
        run (Callable[[], tuple[list | None, list | None, int | None]]): Runs the command once, returns its
                                                                         (stdout, stderr, exit status).
        server (str): The server name, for the error message.
        retry_delays (tuple[float, ...], optional): Seconds to wait before each retry.
        missing_command_reads_empty (bool, optional): Whether a missing queue command reads as an empty queue.
        on_missing_command (Callable[[], None], optional): Called when a missing queue command was read as an empty
                                                           queue, so a caller can tell it from a real answer.

    Raises:
        QueueQueryError: If every attempt failed.

    Returns:
        list[str]: The stdout lines of the first answered attempt.
    """
    error, exit_status = '', None
    for attempt, delay in enumerate((None, ) + tuple(retry_delays), start=1):
        if delay is not None:
            time.sleep(delay)
        try:
            stdout, stderr, exit_status = run()
        except OSError as e:
            stdout, stderr, exit_status = list(), [f'{type(e).__name__}: {e}'], None
        if not queue_query_failed(stdout, stderr, exit_status=exit_status,
                                  missing_command_reads_empty=missing_command_reads_empty):
            if exit_status == MISSING_COMMAND_EXIT_STATUS and on_missing_command is not None:
                on_missing_command()
            return stdout or list()
        error = ' '.join(line.strip() for line in list(stderr or list()) + list(stdout or list()) if line.strip())
        logger.debug(f'Attempt {attempt} of the queue status command on {server} failed '
                     f'(exit status {exit_status}) with: {error}')
    raise QueueQueryError(f'The queue status command on {server} failed {len(retry_delays) + 1} times, '
                          f'the last time with exit status {exit_status}: {error}',
                          server=server, output=error, exit_status=exit_status)


def positive_setting(value: object, name: str, default: float, unit: str, scale: float = 1.0) -> float:
    """
    Read a setting that must be a positive, finite number.

    Args:
        value (object): The setting's value; ``None`` where it is not set.
        name (str): The setting's name, for the warning.
        default (float): The value to use where the setting is absent or malformed.
        unit (str): The setting's unit, for the warning.
        scale (float, optional): The factor that turns the unit into the returned one.

    Returns:
        float: The setting times ``scale``, or ``default`` times ``scale`` if the setting is absent or malformed (a
               malformed one is warned about).
    """
    if value is None:
        return default * scale
    try:
        number = float(value) * scale if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    except OverflowError:
        number = None
    if number is None or not math.isfinite(number) or number <= 0:
        logger.warning(f'The {name} setting must be a positive number of {unit}, got {value!r}. '
                       f'Using {default} {unit}.')
        return default * scale
    return number


def queue_outage_limit(hours: object) -> float:
    """
    Get the time a server's queue may stay unreadable before ARC leaves its jobs for a restart.

    Args:
        hours (object): The ``queue_outage_limit_hrs`` setting; ``None`` where it is not set.

    Returns:
        float: The limit in seconds. An absent or malformed setting gives ``QUEUE_OUTAGE_LIMIT_HRS``, never no limit.
    """
    return positive_setting(hours, 'queue_outage_limit_hrs', QUEUE_OUTAGE_LIMIT_HRS, 'hours', scale=3600.0)


def queue_query_timeout(seconds: object) -> float:
    """
    Get the time one attempt of a queue status command may take before it counts as a failed attempt.

    Args:
        seconds (object): The ``queue_query_timeout`` setting; ``None`` where it is not set.

    Returns:
        float: The timeout in seconds. An absent or malformed setting gives ``QUEUE_QUERY_TIMEOUT``, never no timeout.
    """
    return float(positive_setting(seconds, 'queue_query_timeout', QUEUE_QUERY_TIMEOUT, 'seconds'))


_resolved_query_timeout = dict()


def configured_queue_query_timeout() -> float:
    """
    Get the ``queue_query_timeout`` setting as ARC runs with it, warning about a malformed value only once.

    Returns:
        float: The timeout in seconds.
    """
    value = settings.get('queue_query_timeout')
    resolved = _resolved_query_timeout.get('last')
    if resolved is None or resolved[0] is not value:
        resolved = _resolved_query_timeout['last'] = (value, queue_query_timeout(value))
    return resolved[1]
