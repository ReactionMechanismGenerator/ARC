"""
A module for running jobs on the local machine.
When transitioning to Python 3, use
`subprocess.run() <https://docs.python.org/3/library/subprocess.html#subprocess.run>`_
"""

import datetime
import os
import re
import shutil
import subprocess
import time
from typing import List, Optional, Tuple, Union

from arc.common import get_logger
from arc.exceptions import SettingsError
from arc.imports import settings
from arc.job.ssh import check_job_status_in_stdout


logger = get_logger()

servers, check_status_command, submit_command, submit_filenames, delete_command, output_filenames = \
    settings['servers'], settings['check_status_command'], settings['submit_command'], settings['submit_filenames'],\
    settings['delete_command'], settings['output_filenames']


def execute_command(command: Union[str, List[str]],
                    shell: bool = True,
                    no_fail: bool = False,
                    ) -> Tuple[Optional[list], Optional[list]]:
    """
    Execute a command.

    Notes:
        If ``no_fail`` is ``True``, then a warning is logged and ``False`` is returned
        so that the calling function can debug the situation.

    Args:
        command (Union[str, List[str]]): An array of string commands to send.
        shell (bool, optional): Specifies whether the command should be executed using bash instead of Python.
        no_fail (bool, optional): If ``True`` then ARC will not crash if an error is encountered.

    Returns: Tuple[list, list]:
        - A list of lines of standard output stream.
        - A list of lines of the standard error stream.
    """
    error = None
    if not isinstance(command, list):
        command = [command]
    command = [' && '.join(command)]
    i, max_times_to_try = 1, 30
    sleep_time = 60  # Seconds
    while i < max_times_to_try:
        try:
            completed_process = subprocess.run(command, shell=shell, capture_output=True)
            return _format_stdout(completed_process.stdout), _format_stdout(completed_process.stderr)
        except subprocess.CalledProcessError as e:
            error = e  # Store the error so we can raise a SettingsError if needed.
            if no_fail:
                _output_command_error_message(command, e, logger.warning)
                return None, None
            else:
                _output_command_error_message(command, e, logger.error)
                logger.error(f'ARC is sleeping for {sleep_time * i} seconds before retrying.\nPlease check whether '
                             f'this is a server issue by executing the command manually on the server.')
                logger.info('ZZZZZ..... ZZZZZ.....')
                time.sleep(sleep_time * i)  # In seconds
                i += 1
    # If unsuccessful:
    raise SettingsError(f'The command "{command}" is erroneous, got: \n{error}'
                        f'\nThis maybe either a server issue or the command is wrong.'
                        f'\nTo check if this is a server issue, please run the command on server and restart ARC.'
                        f'\nTo correct the command, modify settings.py'
                        f'\nTips: use "which" command to locate cluster software commands on server.'
                        f'\nExample: type "which sbatch" on a server running Slurm to find the correct '
                        f'sbatch path required in the submit_command dictionary.')


def _output_command_error_message(command: List[str],
                                  error: subprocess.CalledProcessError,
                                  logging_func: Union[logger.warning, logger.error],
                                  ) -> None:
    """
    Formats and logs the error message returned from a command at the desired logging level

    Args:
        command (List[str]): The command that threw the error.
        error (subprocess.CalledProcessError): The exception caught by python from subprocess.
        logging_func: ``logging.warning`` or ``logging.error`` as a function object.
    """
    logging_func('The server command is erroneous.')
    logging_func(f'Tried to submit the following command:\n{command}')
    logging_func('And got the following status (cmd, message, output, return code)')
    logging_func(error.cmd)
    logger.info('\n')
    logging_func(error)
    logger.info('\n')
    logging_func(error.output)
    logger.info('\n')
    logging_func(error.returncode)


def _format_stdout(stdout: bytes) -> List[str]:
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
    stdout = execute_command(cmd)[0]
    return check_job_status_in_stdout(job_id=job_id, stdout=stdout, server=server)


def delete_job(job_id: Union[int, str]):
    """
    Deletes a running job.
    """
    cmd = f"{delete_command[servers['local']['cluster_soft']]} {job_id}"
    success = not bool(execute_command(cmd, no_fail=True)[1])
    if not success:
        logger.warning(f'Detected possible error when trying to delete job {job_id}. Checking to see if the job is '
                       f'still running...')
        running_jobs = check_running_jobs_ids()
        if job_id in running_jobs:
            logger.error(f'Job {job_id} was scheduled for deletion, but the deletion command has appeared to errored. '
                         f'The job is still running.')
            raise RuntimeError(f'Could not delete job {job_id}')
        else:
            logger.info(f'Job {job_id} is no longer running.')


def check_running_jobs_ids() -> List[str]:
    """
    Check which jobs are still running on the server for this user.

    Returns:
        List[str]: List of job IDs.
    """
    cluster_soft = servers['local']['cluster_soft'].lower()
    if cluster_soft not in ['slurm', 'oge', 'sge', 'pbs', 'htcondor']:
        raise ValueError(f"Server cluster software {servers['local']['cluster_soft']} is not supported.")
    cmd = check_status_command[servers['local']['cluster_soft']]
    stdout = execute_command(cmd)[0]
    running_job_ids = parse_running_jobs_ids(stdout, cluster_soft=cluster_soft)
    return running_job_ids


def parse_running_jobs_ids(stdout: List[str],
                           cluster_soft: Optional[str] = None,
                           ) -> List[str]:
    """
    A helper function for parsing job IDs from the stdout of a job status command.

    Args:
        stdout (List[str]): The stdout of a job status command.
        cluster_soft (Optional[str]): The cluster software.

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
               ) -> Tuple[Optional[str], Optional[str]]:
    """
    Submit a job.

    Args:
        path (str): The job's folder path, where the submit script is located (just the folder path, w/o the filename).

    Returns:
        Tuple[Optional[str], Optional[str]]: job_status, job_id
    """
    job_status, job_id = '', ''
    cmd = f"cd {path}; {submit_command[servers['local']['cluster_soft']]} " \
          f"{submit_filenames[servers['local']['cluster_soft']]}"
    stdout, stderr = execute_command(cmd)
    if not len(stdout):
        time.sleep(10)
        stdout, stderr = execute_command(cmd)
    if not len(stdout):
        return None, None
    if len(stderr) > 0 or len(stdout) == 0:
        logger.warning(f'Got the following error when trying to submit job:\n{stderr}.')
        job_status = 'errored'
    else:
        job_id = _determine_job_id
    job_status = 'running' if job_id else job_status
    return job_status, job_id


def _determine_job_id(stdout: List[str],
                      cluster_soft: Optional[str] = None
                      ) -> str:
    """
    Determine the job ID right after it was submitted from the stdout.

    Args:
        stdout (List[str]): The stdout got from submitting a job.
    """
    job_id = ''
    cluster_soft = cluster_soft or servers['local']['cluster_soft'].lower()
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
                           file_path_2: Optional[str] = None,
                           ) -> Optional[datetime.datetime]:
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


def rename_output(local_file_path: str, software: str) -> None:
    """
    Rename the output file to "output.out" for consistency between software.

    Args:
        local_file_path (str): The full path to the output.out file.
        software (str): The software used for the job by which the original output file name was determined.
    """
    software = software.lower()
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
                                    under a directory.``True`` for recursively change.
        path (str, optional): The directory path at which the command will be executed.
    """
    if os.path.isfile(path):
        path = os.path.dirname(path)
    recursive = ' -R' if recursive else ''
    command = [f'cd {path}'] if path else []
    command.append(f'chmod{recursive} {mode} {file_name}')
    execute_command(command=command)


def delete_all_local_arc_jobs(jobs: Optional[List[Union[str, int]]] = None) -> None:
    """
    Delete all ARC-spawned jobs (with job name starting with `a` and a digit) from the local server.
    Make sure you know what you're doing, so unrelated jobs won't be deleted...
    Useful when terminating ARC while some (ghost) jobs are still running.

    Args:
        jobs (List[Union[str, int]], optional): Specific ARC job IDs to delete.
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
