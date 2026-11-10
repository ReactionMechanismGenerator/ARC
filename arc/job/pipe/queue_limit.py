"""
Sizing a pipe job array against the per-user job limit of the queue.

A pipe run submits one job array whose subjobs are workers, and every array subjob counts
against the per-user queue limit that the scheduler judges when the array is submitted. The
limit is rarely known to the user, so it is read from the scheduler when possible
(:func:`discover_job_allowance`), and :func:`next_array_size` shrinks the array when the scheduler
refuses it regardless. Every discovery failure yields ``None`` (unknown), never an exception.
"""

import getpass
import os
import re

from arc.common import get_logger
from arc.imports import settings
from arc.job.local import execute_command


logger = get_logger()

PBS_LIMIT_ENTRY = re.compile(r'\[([ugpo]):([^=\]\s]+)=(\d+)\]')
PBS_GENERIC_USER = 'PBS_GENERIC'
PBS_JOB_ID = re.compile(r'^(\d+)(\[\d*\])?\.')
PBS_INACTIVE_STATES = ('F', 'X')


def next_array_size(size: int) -> int:
    """
    The array size to try after the scheduler refused ``size`` workers.

    Args:
        size (int): The refused array size.

    Returns:
        int: Half of ``size``, but at least 1.
    """
    return max(1, size // 2)


def parse_pbs_per_user_limit(text: str, user: str) -> int | None:
    """
    Parse the per-user ``max_queued`` limit from the output of ``qstat -Qf <queue>`` or ``qstat -Bf``.

    A limit set for the user by name wins over the generic one (``PBS_GENERIC``). Limits on
    groups, projects, or the whole queue/server are not per-user limits and are ignored, as are
    resource limits such as ``max_queued_res.ncpus``.

    Args:
        text (str): The ``qstat -Qf`` or ``qstat -Bf`` output.
        user (str): The user name.

    Returns:
        int | None: The limit, or ``None`` if the text defines no per-user job-count limit.
    """
    generic, specific = None, None
    for line in text.replace('\n\t', '').splitlines():
        key, sep, value = line.partition('=')
        if not sep or key.strip() != 'max_queued':
            continue
        for kind, name, number in PBS_LIMIT_ENTRY.findall(value):
            if kind != 'u':
                continue
            if name == user:
                specific = int(number)
            elif name == PBS_GENERIC_USER:
                generic = int(number)
    return specific if specific is not None else generic


def parse_pbs_max_array_size(text: str) -> int | None:
    """
    Parse the largest allowed job array size (``max_array_size``) from ``qstat -Bf`` or ``qstat -Qf``.

    Args:
        text (str): The ``qstat -Bf`` or ``qstat -Qf`` output.

    Returns:
        int | None: The largest number of subjobs of an array, or ``None`` if not set.
    """
    match = re.search(r'^\s*max_array_size\s*=\s*(\d+)\s*$', text, re.MULTILINE)
    return int(match.group(1)) if match else None


def _pbs_queue_matches(listed: str, queue: str) -> bool:
    """
    Whether a queue column entry of ``qstat -u`` names ``queue``.

    The default (narrow) output truncates a long name and marks it with ``*`` (``mafat_n*``);
    such an entry matches every queue it is a prefix of, which can only overcount.
    """
    if listed.endswith('*'):
        return queue.startswith(listed[:-1])
    return listed == queue


def count_pbs_user_jobs(stdout: list[str], queue: str) -> int:
    """
    Count the user's active jobs (queued, running, held, ...) in a queue from ``qstat -w -t -u <user>``.

    Each listed subjob counts as one job. An array's parent row counts as one job only if none of
    its subjobs is listed, as is the case for a queued array that has not started.

    Args:
        stdout (list[str]): The ``qstat -w -t -u <user>`` stdout lines (``-w`` avoids truncated names).
        queue (str): The queue name.

    Returns:
        int: The number of active jobs in that queue.
    """
    count = 0
    parents, parents_with_subjobs = set(), set()
    for line in stdout:
        tokens = line.split()
        match = PBS_JOB_ID.match(tokens[0]) if tokens else None
        if len(tokens) < 10 or match is None or not _pbs_queue_matches(tokens[2], queue) \
                or tokens[-2] in PBS_INACTIVE_STATES:
            continue
        array_index = match.group(2)
        if array_index == '[]':
            parents.add(match.group(1))
        else:
            count += 1
            if array_index is not None:
                parents_with_subjobs.add(match.group(1))
    return count + len(parents - parents_with_subjobs)


def parse_slurm_max_submit_jobs(text: str) -> int | None:
    """
    Parse the per-user submit limit from ``sacctmgr -n -P show assoc user=<user> format=MaxSubmitJobs``.

    Args:
        text (str): The sacctmgr output, one association per line, empty if unlimited.

    Returns:
        int | None: The smallest positive limit, or ``None`` if there is none.
    """
    limits = [int(line.strip()) for line in text.splitlines() if line.strip().isdigit() and int(line.strip()) > 0]
    return min(limits) if limits else None


def parse_slurm_max_array_size(text: str) -> int | None:
    """
    Parse ``MaxArraySize`` from ``scontrol show config``.

    Slurm accepts array indices below ``MaxArraySize``, so the largest array has ``MaxArraySize - 1`` tasks.

    Args:
        text (str): The ``scontrol show config`` output.

    Returns:
        int | None: The largest allowed number of array tasks, or ``None`` if not found.
    """
    match = re.search(r'^\s*MaxArraySize\s*=\s*(\d+)', text, re.MULTILINE)
    return max(int(match.group(1)) - 1, 1) if match else None


def _run(command: str) -> str:
    """
    Run a read-only scheduler query and return its stdout, or an empty string on any failure.
    """
    try:
        stdout, _ = execute_command(command)
    except Exception as e:
        logger.debug(f'The scheduler query "{command}" failed: {e}')
        return ''
    return '\n'.join(stdout or list())


def _command_in_dir_of(status_command: str, name: str) -> str:
    """
    A scheduler command ``name`` expected beside the configured status command.
    """
    binary = status_command.split()[0]
    return os.path.join(os.path.dirname(binary), name) if os.path.dirname(binary) else name


def discover_job_allowance(cluster_software: str,
                           queue: str,
                           limit_override: int | None = None,
                           ) -> int | None:
    """
    How many more jobs the user may submit to a queue, as far as the scheduler says.

    Args:
        cluster_software (str): ``'pbs'`` or ``'slurm'``; any other scheduler is not queried.
        queue (str): The queue (partition) name.
        limit_override (int, optional): The per-user job limit, if the user states it
                                        (``pipe_settings['max_queued_jobs']``), instead of reading it.

    Returns:
        int | None: The allowance (0 if the user is at the limit), or ``None`` if unknown.
    """
    software = cluster_software.lower()
    status_commands = settings.get('check_status_command') or dict()
    user = getpass.getuser()
    if software == 'pbs':
        status_command = status_commands.get('PBS')
        if not status_command:
            return None
        qstat = status_command.split()[0]
        server_text = _run(f'{qstat} -Bf')
        limits = [limit_override] if limit_override else [
            parse_pbs_per_user_limit(_run(f'{qstat} -Qf {queue}'), user),
            parse_pbs_per_user_limit(server_text, user)]
        limits = [limit for limit in limits if limit is not None]
        max_array = parse_pbs_max_array_size(server_text)
        allowances = [max_array] if max_array is not None else list()
        if limits:
            used = count_pbs_user_jobs(_run(f'{qstat} -w -t -u {user}').splitlines(), queue)
            allowances.append(max(min(limits) - used, 0))
        return min(allowances) if allowances else None
    if software == 'slurm':
        status_command = status_commands.get('Slurm')
        if not status_command:
            return None
        limit = limit_override or parse_slurm_max_submit_jobs(
            _run(f'{_command_in_dir_of(status_command, "sacctmgr")} -n -P show assoc user={user} format=MaxSubmitJobs'))
        max_array = parse_slurm_max_array_size(_run(f'{_command_in_dir_of(status_command, "scontrol")} show config'))
        allowances = list()
        if limit is not None:
            used = len([line for line in _run(f'{status_command.split()[0]} -h -r -u {user} -p {queue} -t PD,R '
                                              f'-o %i').splitlines() if line.strip()])
            allowances.append(max(limit - used, 0))
        if max_array is not None:
            allowances.append(max_array)
        return min(allowances) if allowances else None
    return None
