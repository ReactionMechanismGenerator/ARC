"""
A module for running jobs on the local machine.
When transitioning to Python 3, use
`subprocess.run() <https://docs.python.org/3/library/subprocess.html#subprocess.run>`_
"""

import datetime
import errno
import os
import re
import shutil
import subprocess
import time

from arc.common import get_logger
from arc.exceptions import ServerError
from arc.imports import settings
from arc.job.ssh import check_job_status_in_stdout, queue_query_failed, register_queue_query


logger = get_logger()

TRANSIENT_SPAWN_ERRNOS = frozenset({errno.EAGAIN, errno.ENOMEM, errno.EMFILE, errno.ENFILE})

servers, check_status_command, submit_command, submit_filenames, delete_command, output_filenames = \
    settings['servers'], settings['check_status_command'], settings['submit_command'], settings['submit_filenames'],\
    settings['delete_command'], settings['output_filenames']


def execute_command(command: str | list[str],
                    shell: bool = True,
                    no_fail: bool = False,
                    executable: str | None = None,
                    return_code: bool = False,
                    ) -> tuple[list | None, list | None] | tuple[list | None, list | None, int | None]:
    """
    Execute a command.

    Notes:
        If ``no_fail`` is ``True``, then a warning is logged and ``None`` values are returned
        so that the calling function can debug the situation.

        A non-zero exit status is not treated as an error, and is only reported to callers
        that request ``return_code``.

        A failure to spawn a process for the command, e.g., when the machine cannot allocate
        memory or has no free process slots, is retried with an escalating delay.

    Args:
        command (str | list[str]): An array of string commands to send.
        shell (bool, optional): Specifies whether the command should be executed using bash instead of Python.
        no_fail (bool, optional): If ``True`` then ARC will not crash if an error is encountered.
        executable (str, optional): Select a specific shell to run with, e.g., '/bin/bash'.
                                    Default shell of the subprocess command is '/bin/sh'.
        return_code (bool, optional): Whether to also return the exit status of the command.

    Raises:
        ServerError: If a process for the command could not be spawned in ``max_times_to_try`` attempts
                     and ``no_fail`` is ``False``.

    Returns: tuple[list, list] | tuple[list, list, int]:
        - A list of lines of standard output stream.
        - A list of lines of the standard error stream.
        - The exit status of the command, only if ``return_code`` is ``True``.
    """
    error = None
    if not isinstance(command, list):
        command = [command]
    command = [' && '.join(command)]
    i, max_times_to_try = 1, 10
    sleep_time = 60  # Seconds
    while i <= max_times_to_try:
        try:
            if executable is None:
                completed_process = subprocess.run(command, shell=shell, capture_output=True)
            else:
                completed_process = subprocess.run(command, shell=shell, capture_output=True, executable=executable)
        except OSError as e:
            if e.errno not in TRANSIENT_SPAWN_ERRNOS:
                raise
            error = e
            _output_command_error_message(command, e, logger.error)
            if i == max_times_to_try:
                break
            logger.error(f'ARC could not spawn a process for the above command (attempt {i} of {max_times_to_try}), '
                         f'and is sleeping for {sleep_time * i} seconds before retrying.')
            logger.info('ZZZZZ..... ZZZZZ.....')
            time.sleep(sleep_time * i)
            i += 1
            continue
        stdout, stderr = _format_stdout(completed_process.stdout), _format_stdout(completed_process.stderr)
        if return_code:
            return stdout, stderr, completed_process.returncode
        return stdout, stderr
    message = (f'Could not spawn a process for the command "{command}" in {max_times_to_try} attempts, got:\n{error}'
               f'\nThe machine could not allocate the resources required for another process, which usually means '
               f'that too many processes are running on it concurrently, or that it ran out of memory.'
               f'\nRun fewer concurrent ARC instances or incore jobs on this machine, or use a machine with more '
               f'memory.')
    if no_fail:
        logger.warning(message)
        return (None, None, None) if return_code else (None, None)
    raise ServerError(message)


def _output_command_error_message(command: list[str],
                                  error: OSError,
                                  logging_func,
                                  ) -> None:
    """
    Formats and logs an error raised while trying to run a command at the desired logging level

    Args:
        command (list[str]): The command that threw the error.
        error (OSError): The exception raised when trying to run the command.
        logging_func: ``logging.warning`` or ``logging.error`` as a function object.
    """
    logging_func('The server command is erroneous.')
    logging_func(f'Tried to run the following command:\n{command}')
    logging_func(f'And got the following error (errno, message, file): '
                 f'{error.errno}, {error.strerror}, {error.filename}')
    logging_func(error)


def _format_stdout(stdout: bytes) -> list[str]:
    """
    Format the stdout as a list of unicode strings

    Args:
        stdout (bytes): The standard output.

    Returns:
        List(str): The decoded lines from stdout.
    """
    lines, list_of_strs = stdout.splitlines(), list()
    for line in lines:
        list_of_strs.append(line.decode())
    return list_of_strs


def check_job_status(job_id: int) -> str:
    """
    Possible status values: ``before_submission``, ``running``, ``errored on node xx``, ``done``
    Status line formats:

    OGE::

        540420 0.45326 xq1340b    user_name       r     10/26/2018 11:08:30 long1@node18.cluster

    Slurm::

        14428     debug xq1371m2   user_name  R 50-04:04:46      1 node06

    PBS (taken from zeldo.dow.com)::
                                                                                         Req'd       Req'd       Elap
        Job ID                  Username    Queue    Jobname         SessID  NDS   TSK   Memory      Time    S   Time
        ----------------------- ----------- -------- --------------- ------ ----- ------ --------- --------- - ---------
        2016614.zeldo.local     u780444     workq    scan.pbs         75380     1     10       --  730:00:00 R  00:00:20
        2016616.zeldo.local     u780444     workq    scan.pbs         75380     1     10       --  730:00:00 R  00:00:20

    HTCondor (using ARC's modified condor_q command)::

        3261.0 R 10 28161 a2719 56
        3263.0 R 10 28161 a2721 23
        3268.0 R 10 28161 a2726 18
        3269.0 R 10 28161 a2727 17
        3270.0 P 10 28161 a2728 23
    """
    server = 'local'
    cmd = check_status_command[servers[server]['cluster_soft']]
    stdout, stderr, code = execute_command(cmd, return_code=True)
    failed = queue_query_failed(return_code=code, stderr=stderr)
    register_queue_query(failed=failed, server=server, return_code=code, stderr=stderr)
    return check_job_status_in_stdout(job_id=job_id, stdout=stdout, server=server,
                                      return_code=code, stderr=stderr)


def delete_job(job_id: int | str):
    """
    Deletes a running job.
    """
    cmd = f"{delete_command[servers['local']['cluster_soft']]} {job_id}"
    success = not bool(execute_command(cmd, no_fail=True)[1])
    if not success:
        logger.warning(f'Detected possible error when trying to delete job {job_id}. Checking to see if the job is '
                       f'still running...')
        running_jobs = check_running_jobs_ids()
        if running_jobs is None:
            logger.error(f'Job {job_id} was scheduled for deletion, but the deletion command appeared to have errored '
                         f'and the queue could not be queried. It is unknown whether the job was deleted.')
        elif job_id in running_jobs:
            logger.error(f'Job {job_id} was scheduled for deletion, but the deletion command has appeared to errored. '
                         f'The job is still running.')
            raise RuntimeError(f'Could not delete job {job_id}')
        else:
            logger.info(f'Job {job_id} is no longer running.')


def check_running_jobs_ids() -> list[str] | None:
    """
    Check which jobs are still running on the server for this user.
    The output of a query which failed is not parsed.

    Returns:
        list[str] | None: A list of job IDs, or ``None`` if the queue could not be queried, in which
                          case the queue must not be assumed to be empty.
    """
    cluster_soft = servers['local']['cluster_soft'].lower()
    if cluster_soft not in ['slurm', 'oge', 'sge', 'pbs', 'htcondor']:
        raise ValueError(f"Server cluster software {servers['local']['cluster_soft']} is not supported.")
    cmd = check_status_command[servers['local']['cluster_soft']]
    stdout, stderr, code = execute_command(cmd, return_code=True)
    failed = queue_query_failed(return_code=code, stderr=stderr)
    register_queue_query(failed=failed, server='local', return_code=code, stderr=stderr)
    if failed:
        return None
    return parse_running_jobs_ids(stdout, cluster_soft=cluster_soft)


def parse_running_jobs_ids(stdout: list[str],
                           cluster_soft: str | None = None,
                           ) -> list[str]:
    """
    A helper function for parsing job IDs from the stdout of a job status command.

    Args:
        stdout (list[str]): The stdout of a job status command.
        cluster_soft (str | None): The cluster software.

    Returns:
        List(str): List of job IDs.
    """
    cluster_soft = cluster_soft or servers['local']['cluster_soft'].lower()
    i_dict = {'slurm': 0, 'oge': 1, 'sge': 1, 'pbs': 4, 'htcondor': -1}
    split_by_dict = {'slurm': ' ', 'oge': ' ', 'sge': ' ', 'pbs': '.', 'htcondor': '.'}
    running_job_ids = list()
    for i, status_line in enumerate(stdout):
        if i > i_dict[cluster_soft]:
            job_id = status_line.strip().split(split_by_dict[cluster_soft])[0]
            job_id = f'{job_id}'  # job_id is sometimes a byte, this transforms b'bytes' into "b'bytes'"
            if "b'" in job_id:
                job_id = job_id.split("b'")[1].split("'")[0]
            running_job_ids.append(job_id)
    return running_job_ids


def submit_job(path: str,
               cluster_soft: str | None = None,
               submit_cmd: str | None = None,
               submit_filename: str | None = None,
               recursion: bool = False,
               ) -> tuple[str | None, str | None]:
    """
    Submit a job.

    Args:
        path (str): The job's folder path, where the submit script is located (just the folder path, w/o the filename).
        cluster_soft (str, optional): The server cluster software.
        submit_cmd (str, optional): The submit command.
        submit_filename (str, optional): The submit script file name.
        recursion (bool, optional): Whether this call is within a recursion.

    Returns:
        tuple[str | None, str | None]: job_status, job_id
    """
    cluster_soft = cluster_soft or servers['local']['cluster_soft']
    job_status, job_id = '', ''
    submit_cmd = submit_cmd or submit_command[cluster_soft]
    submit_filename = submit_filename or submit_filenames[cluster_soft]
    cmd = f'cd "{path}"; {submit_cmd} {submit_filename}'
    stdout, stderr = execute_command(cmd)
    if not len(stdout):
        time.sleep(10)
        stdout, stderr = execute_command(cmd)
    if stderr:
        if cluster_soft.lower() == 'slurm' and any('AssocMaxSubmitJobLimit' in err_line for err_line in stderr):
            logger.warning(f'Max number of submitted jobs was reached, sleeping...')
            time.sleep(5 * 60)
            submit_job(path=path,
                       cluster_soft=cluster_soft,
                       submit_cmd=submit_cmd,
                       submit_filename=submit_filename,
                       recursion=True,
                       )
        if cluster_soft.lower() == 'pbs' and  (any('qsub: would exceed' in err_line for err_line in stderr ) or any('qsub: Maximum number of jobs' in err_line for err_line in stderr)):
            logger.warning(f'Max number of submitted jobs was reached, sleeping...')
            time.sleep(5 * 60)
            submit_job(path=path,
                       cluster_soft=cluster_soft,
                       submit_cmd=submit_cmd,
                       submit_filename=submit_filename,
                       recursion=True,
                       )
        elif cluster_soft.lower() == 'pbs' and any('qsub: Illegal attribute or resource value' in err_line for err_line in stderr):
            raise ValueError(f'Got the following error when trying to submit job:\n{stderr}. Please check your submit script')
        elif cluster_soft.lower() == 'pbs' and (
                any('Please do NOT submit jobs on compute nodes' in err_line for err_line in stderr)
                or any('Jobs should be submitted on login server' in err_line for err_line in stderr)
        ):
            raise ValueError('PBS job submission attempted from a compute node. Submit jobs from the login server.')
    if not len(stdout) or recursion:
        return None, None
    if len(stderr) > 0 or len(stdout) == 0:
        logger.warning(f'Got the following error when trying to submit job:\n{stderr}.')
        job_status = 'errored'
    else:
        job_id = _determine_job_id(stdout=stdout, cluster_soft=cluster_soft)
    job_status = 'running' if job_id else job_status
    return job_status, job_id


def _determine_job_id(stdout: list[str],
                      cluster_soft: str | None = None
                      ) -> str:
    """
    Determine the job ID right after it was submitted from the stdout.

    Args:
        stdout (list[str]): The stdout got from submitting a job.
        cluster_soft (str, optional): The server cluster software.

    Returns:
        str: The determined job ID.
    """
    job_id = ''
    cluster_soft = cluster_soft or servers['local']['cluster_soft']
    cluster_soft = cluster_soft.lower() if cluster_soft is not None else None
    if cluster_soft in ['oge', 'sge'] and 'submitted' in stdout[0].lower():
        job_id = stdout[0].split()[2]
    elif cluster_soft == 'slurm' and 'submitted' in stdout[0].lower():
        job_id = stdout[0].split()[3]
    elif cluster_soft == 'pbs':
        job_id = stdout[0].split('.')[0]
    elif cluster_soft == 'htcondor' and 'submitting' in stdout[0].lower():
        if len(stdout) and len(stdout[1].split()) and len(stdout[1].split()[-1].split('.')):
            job_id = stdout[1].split()[-1].split('.')[0]
    else:
        raise ValueError(f'Unrecognized cluster software: {cluster_soft}')
    return job_id


def get_last_modified_time(file_path_1: str,
                           file_path_2: str | None = None,
                           ) -> datetime.datetime | None:
    """
    Returns the last modified time of ``file_path_1`` if the file exists,
    else returns the last modified time of ``file_path_2`` if the file exists.

    Args:
        file_path_1 (str): The path to file 1.
        file_path_2 (str, optional): The path to file 2.
    """
    timestamp = None
    if os.path.isfile(file_path_1):
        try:
            timestamp = os.stat(file_path_1).st_mtime
        except (IOError, OSError):
            pass
    if timestamp is None and file_path_2 is not None:
        try:
            timestamp = os.stat(file_path_2).st_mtime
        except (IOError, OSError):
            return None
    return datetime.datetime.fromtimestamp(timestamp) if timestamp is not None else None


def write_file(file_path: str, file_string: str) -> None:
    """
    Write ``file_string`` as the file's content in ``file_path``.

    Args:
        file_path (str): The file path.
        file_string (str): The content to be written into the file.
    """
    with open(file_path, 'w') as f:
        f.write(file_string)


def rename_output(local_file_path: str,
                  software: str,
                  ) -> None:
    """
    Rename the output file to "output.out" for consistency between software.

    Args:
        local_file_path (str): The full path to the output.out file.
        software (str): The software used for the job by which the original output file name was determined.
    """
    software = software.lower()

    for i in range(5):
        if not os.path.isfile(local_file_path) \
                and not os.path.isfile(os.path.join(os.path.dirname(local_file_path), output_filenames[software])):
            # Wait for file to be transferred on the server (the head node might be busy).
            time.sleep(6)
        else:
            break
    else:
        # Nothing to rename.
        return None

    if os.path.isfile(os.path.join(os.path.dirname(local_file_path), output_filenames[software])):
        shutil.move(src=os.path.join(os.path.dirname(local_file_path), output_filenames[software]), dst=local_file_path)


def change_mode(mode: str,
                file_name: str,
                recursive: bool = False,
                path: str = '',
                ) -> None:
    """
    Change the mode of a file or a directory.

    Args:
        mode (str): The mode change to be applied, can be either octal or symbolic.
        file_name (str): The path to the file or the directory to be changed.
        recursive (bool, optional): Whether to recursively change the mode to all files
                                    under a directory. ``True`` for recursively change.
        path (str, optional): The directory path at which the command will be executed.
    """
    if os.path.isfile(path):
        path = os.path.dirname(path)
    recursive = ' -R' if recursive else ''
    command = [f'cd "{path}"'] if path else []
    command.append(f'chmod{recursive} {mode} {file_name}')
    execute_command(command=command)


def delete_all_local_arc_jobs(jobs: list[str | int] | None = None) -> None:
    """
    Delete all ARC-spawned jobs (with job name starting with `a` and a digit) from the local server.
    Make sure you know what you're doing, so unrelated jobs won't be deleted...
    Useful when terminating ARC while some (ghost) jobs are still running.

    Args:
        jobs (list[str | int], optional): Specific ARC job IDs to delete.
    """
    server = 'local'
    if server in servers:
        print('\nDeleting all ARC jobs from local server...')
        cmd = check_status_command[servers[server]['cluster_soft']]
        stdout = execute_command(cmd, no_fail=True)[0]
        for status_line in stdout:
            s = re.search(r' a\d+', status_line)
            if s is not None:
                job_name = s.group()[1:]
                cluster_soft = servers[server]['cluster_soft'].lower()
                server_job_id = None
                if jobs is None or job_name in jobs:
                    if cluster_soft == 'slurm':
                        server_job_id = status_line.split()[0]
                        delete_job(server_job_id)
                    elif cluster_soft == 'pbs':
                        server_job_id = status_line.split()[0]
                        delete_job(server_job_id)
                    elif cluster_soft in ['oge', 'sge']:
                        delete_job(job_name)
                    elif cluster_soft == 'htcondor':
                        server_job_id = status_line.split()[0].split('.')[0]
                        delete_job(server_job_id)
                    else:
                        raise ValueError(f'Unrecognized cluster software {cluster_soft}.')
                    aux_text = f' ({server_job_id} on server)' if server_job_id is not None else ''
                    print(f'deleted job {job_name}{aux_text}.')
        print('\ndone.')
