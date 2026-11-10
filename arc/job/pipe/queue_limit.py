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
PBS_JOB_ID = re.compile(r'^\d+(\[\d*\])?\.')
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


def count_pbs_user_jobs(stdout: list[str], queue: str) -> int:
    """
    Count the user's active jobs (queued, running, held, ...) in a queue from ``qstat -t -u <user>``.

    Args:
        stdout (list[str]): The ``qstat -t -u <user>`` stdout lines.
        queue (str): The queue name.

    Returns:
        int: The number of active job rows in that queue.
    """
    count = 0
    for line in stdout:
        tokens = line.split()
        if len(tokens) > 9 and PBS_JOB_ID.match(tokens[0]) and tokens[2] == queue \
                and tokens[-2] not in PBS_INACTIVE_STATES:
            count += 1
    return count


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
        limits = [limit_override] if limit_override else [
            parse_pbs_per_user_limit(_run(f'{qstat} -Qf {queue}'), user),
            parse_pbs_per_user_limit(_run(f'{qstat} -Bf'), user)]
        limits = [limit for limit in limits if limit is not None]
        if not limits:
            return None
        used = count_pbs_user_jobs(_run(f'{qstat} -t -u {user}').splitlines(), queue)
        return max(min(limits) - used, 0)
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
