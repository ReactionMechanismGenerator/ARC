"""
A module deciding whether a queue status command (``squeue``, ``qstat``, ``condor_q``) was answered.

A command that fails on the cluster, e.g., Slurm's "Socket timed out on send/recv operation", prints nothing on
stdout, which is also what an empty queue looks like. Reading that as "no jobs running" ends every live job, so a
failed query is retried here and then reported as unknown, never as empty. The local and the SSH paths both query
through ``run_queue_query()``.
"""

import re
import time
from collections.abc import Callable

from arc.common import get_logger
from arc.exceptions import QueueQueryError


logger = get_logger()

QUEUE_QUERY_RETRY_DELAYS = (5, 15)  # Seconds to wait before each retry of a failed queue query.
# The shell's own report of a command it cannot run, e.g., "/bin/sh: 1: condor_q: not found".
MISSING_COMMAND_PATTERN = re.compile(r'^\S*sh: (?:line \d+: |\d+: )?\S+: (?:command not found|not found|'
                                     r'No such file or directory)$')


def queue_query_failed(stdout: list[str] | None,
                       stderr: list[str] | None,
                       missing_command_reads_empty: bool = False,
                       ) -> bool:
    """
    Decide whether a queue status command failed, as opposed to answering with an empty queue.

    A query fails when it printed nothing on stdout and an error on stderr. On the local machine a queue command
    that is not installed at all may be read as no jobs instead: machines without a queue, and incore-only runs,
    have always read it so. That holds only when every stderr line is the shell's own missing-command report.

    Args:
        stdout (list[str] | None): The stdout lines of the queue status command.
        stderr (list[str] | None): The stderr lines of the queue status command.
        missing_command_reads_empty (bool, optional): Whether a missing queue command reads as an empty queue.

    Returns:
        bool: Whether the query failed.
    """
    if any(line.strip() for line in stdout or list()):
        return False
    error_lines = [line.strip() for line in stderr or list() if line.strip()]
    if not error_lines:
        return False
    return not (missing_command_reads_empty and all(MISSING_COMMAND_PATTERN.match(line) for line in error_lines))


def run_queue_query(run: Callable[[], tuple[list | None, list | None]],
                    server: str,
                    retry_delays: tuple[float, ...] = QUEUE_QUERY_RETRY_DELAYS,
                    missing_command_reads_empty: bool = False,
                    ) -> list[str]:
    """
    Run a queue status command, retrying it with a bounded backoff while it fails.

    Args:
        run (Callable[[], tuple[list | None, list | None]]): Runs the command once, returns its (stdout, stderr).
        server (str): The server name, for the error message.
        retry_delays (tuple[float, ...], optional): Seconds to wait before each retry.
        missing_command_reads_empty (bool, optional): Whether a missing queue command reads as an empty queue.

    Raises:
        QueueQueryError: If every attempt failed.

    Returns:
        list[str]: The stdout lines of the first answered attempt.
    """
    error = ''
    for attempt, delay in enumerate((None, ) + tuple(retry_delays), start=1):
        if delay is not None:
            time.sleep(delay)
        stdout, stderr = run()
        if not queue_query_failed(stdout, stderr, missing_command_reads_empty=missing_command_reads_empty):
            return stdout or list()
        error = ' '.join(line.strip() for line in stderr if line.strip())
        logger.debug(f'Attempt {attempt} of the queue status command on {server} failed with: {error}')
    raise QueueQueryError(f'The queue status command on {server} failed {len(retry_delays) + 1} times, '
                          f'the last time with: {error}')
