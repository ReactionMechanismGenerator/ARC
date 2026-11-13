"""
A module deciding whether a queue status command (``squeue``, ``qstat``, ``condor_q``) was answered.

A command that fails on the cluster, e.g., Slurm's "Socket timed out on send/recv operation", prints nothing on
stdout, which is also what an empty queue looks like. Reading that as "no jobs running" ends every live job, so a
failed query is retried here and then reported as unknown, never as empty. The local and the SSH paths both query
through ``run_queue_query()``.

The command's exit status decides: 0 is an answer, whatever stderr says (many servers print a module warning or a
login banner on every command), and any other status is a failure. The shell's 127 is a missing command. Only a
result that carries no exit status falls back to reading stdout and stderr.
"""

import math
import time
from collections.abc import Callable

from arc.common import get_logger
from arc.exceptions import QueueQueryError


logger = get_logger()

QUEUE_QUERY_RETRY_DELAYS = (5, 15)  # Seconds to wait before each retry of a failed queue query.
QUEUE_OUTAGE_LIMIT_HRS = 4  # Hours a server's queue may stay unreadable before ARC leaves its jobs for a restart.
MISSING_COMMAND_EXIT_STATUS = 127  # The POSIX shell's exit status for a command it cannot find.


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
                    ) -> list[str]:
    """
    Run a queue status command, retrying it with a bounded backoff while it fails.

    Args:
        run (Callable[[], tuple[list | None, list | None, int | None]]): Runs the command once, returns its
                                                                         (stdout, stderr, exit status).
        server (str): The server name, for the error message.
        retry_delays (tuple[float, ...], optional): Seconds to wait before each retry.
        missing_command_reads_empty (bool, optional): Whether a missing queue command reads as an empty queue.

    Raises:
        QueueQueryError: If every attempt failed.

    Returns:
        list[str]: The stdout lines of the first answered attempt.
    """
    error, exit_status = '', None
    for attempt, delay in enumerate((None, ) + tuple(retry_delays), start=1):
        if delay is not None:
            time.sleep(delay)
        stdout, stderr, exit_status = run()
        if not queue_query_failed(stdout, stderr, exit_status=exit_status,
                                  missing_command_reads_empty=missing_command_reads_empty):
            return stdout or list()
        error = ' '.join(line.strip() for line in list(stderr or list()) + list(stdout or list()) if line.strip())
        logger.debug(f'Attempt {attempt} of the queue status command on {server} failed '
                     f'(exit status {exit_status}) with: {error}')
    raise QueueQueryError(f'The queue status command on {server} failed {len(retry_delays) + 1} times, '
                          f'the last time with exit status {exit_status}: {error}',
                          server=server, stderr=error, exit_status=exit_status)


def queue_outage_limit(hours: object) -> float:
    """
    Get the time a server's queue may stay unreadable before ARC leaves its jobs for a restart.

    Args:
        hours (object): The ``queue_outage_limit_hrs`` setting; ``None`` where it is not set.

    Returns:
        float: The limit in seconds. An absent or malformed setting gives ``QUEUE_OUTAGE_LIMIT_HRS``, never no limit.
    """
    if hours is None:
        return QUEUE_OUTAGE_LIMIT_HRS * 3600.0
    try:
        seconds = float(hours) * 3600.0 if isinstance(hours, (int, float)) and not isinstance(hours, bool) else None
    except OverflowError:
        seconds = None
    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        logger.warning(f'The queue_outage_limit_hrs setting must be a positive number of hours, got {hours!r}. '
                       f'Using {QUEUE_OUTAGE_LIMIT_HRS} hours.')
        return QUEUE_OUTAGE_LIMIT_HRS * 3600.0
    return seconds
