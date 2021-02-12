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
from typing import List, Optional, Union

from arc.common import get_logger
from arc.exceptions import SettingsError
from arc.imports import settings
from arc.job.ssh import check_job_status_in_stdout


logger = get_logger()

servers, check_status_command, submit_command, submit_filenames, delete_command, output_filenames = \
    settings['servers'], settings['check_status_command'], settings['submit_command'], settings['submit_filenames'],\
    settings['delete_command'], settings['output_filenames']


def execute_command(command, shell=True, no_fail=False):
    """
    Execute a command.

    Notes:
        If `no_fail` == True, then a warning is logged and `False` is returned so that the calling function can debug
        the situation.

    Args:
        command: An array of string commands to send.
        shell (bool): Specifies whether the command should be executed using bash instead of Python
        no_fail (bool): If `True` then ARC will not crash if an error is encountered.

    Returns: list
        lists of stdin, stdout, stderr corresponding to the commands sent
    """
    # Initialize variables
    error = None

    if not isinstance(command, list) and not shell:
        command = [command]
    i, max_times_to_try = 1, 30
    sleep_time = 60  # seconds
    while i < max_times_to_try:
        try:
            stdout = subprocess.check_output(command, shell=shell)
            return _format_command_stdout(stdout), ''
        except subprocess.CalledProcessError as e:
            error = e  # Store the error so we can raise the SettingsError if need be
            if no_fail:
                _output_command_error_message(command, e, logger.warning)
                return False
            else:
                _output_command_error_message(command, e, logger.error)
                logger.error(f'ARC is sleeping for {sleep_time * i} seconds before retrying.\nPlease check whether '
                             f'this is a server issue by executing the command manually on the server.')
                logger.info('ZZZZZ..... ZZZZZ.....')
                time.sleep(sleep_time * i)  # in seconds
                i += 1

    # If not success
    raise SettingsError(f'The command "{command}" is erroneous, got: \n{error}'
                        f'\nThis maybe either a server issue or the command is wrong.'
                        f'\nTo check if this is a server issue, please run the command on server and restart ARC.'
                        f'\nTo correct the command, modify settings.py'
                        f'\nTips: use "which" command to locate cluster software commands on server.'
                        f'\nExample: type "which sbatch" on a server running Slurm to find the correct '
                        f'sbatch path required in the submit_command dictionary.')


def _output_command_error_message(command, error, logging_func):
    """
    Formats and logs the error message returned from a command at the desired logging level

    Args:
        command: Command that threw the error
        error: Exception caught by python from subprocess
        logging_func: `logging.warning` or `logging.error` as a python function object
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


def _format_command_stdout(stdout):
    """
    Formats the output from stdout returned from subprocess
    """
    lines, list_of_strs = stdout.splitlines(), list()
    for line in lines:
        list_of_strs.append(line.decode())

    return list_of_strs


def check_job_status(job_id):
    """
    Possible statuses: `before_submission`, `running`, `errored on node xx`, `done`
    Status line formats:

    OGE::

        540420 0.45326 xq1340b    user_name       r     10/26/2018 11:08:30 long1@node18.cluster

    Slurm::

        14428     debug xq1371m2   user_name  R 50-04:04:46      1 node06

    PBS (taken from zeldo.dow.com)::

                                                                                          Req'd       Req'd       Elap
        Job ID                  Username    Queue    Jobname          SessID  NDS   TSK   Memory      Time    S   Time
        ----------------------- ----------- -------- ---------------- ------ ----- ------ --------- --------- - ---------
        2016614.zeldo.local     u780444     workq    scan.pbs          75380     1     10       --  730:00:00 R  00:00:20
        2016616.zeldo.local     u780444     workq    scan.pbs          75380     1     10       --  730:00:00 R  00:00:20
    """
    server = 'local'
    cmd = check_status_command[servers[server]['cluster_soft']] + ' -u $USER'
    stdout = execute_command(cmd)[0]
    return check_job_status_in_stdout(job_id=job_id, stdout=stdout, server=server)


def delete_job(job_id):
    """
    Deletes a running job
    """
    cmd = delete_command[servers['local']['cluster_soft']] + ' ' + str(job_id)
    success = bool(execute_command(cmd, no_fail=True))
    if not success:  # Check if the job is still running. If not then this failure does not matter
        logger.warning(f'Detected possible error when trying to delete job {job_id}. Checking to see if the job is '
                       f'still running...')
        running_jobs = check_running_jobs_ids()
        if job_id in running_jobs:
            logger.error(f'Job {job_id} was scheduled for deletion, but the deletion command has appeared to errored, '
                         f'and is still running')
            raise RuntimeError(f'Could not delete job {job_id}')
        else:  # The job seems to have been deleted.
            logger.warning(f'Job {job_id} is no longer running, so we can continue.')


def check_running_jobs_ids():
    """
    Return a list of ``int`` representing job IDs of all jobs submitted by the user on a server
    """
    running_jobs_ids = list()
    cmd = check_status_command[servers['local']['cluster_soft']] + ' -u $USER'
    stdout = execute_command(cmd)[0]
    for i, status_line in enumerate(stdout):
        if servers['local']['cluster_soft'].lower() == 'slurm' and i > 0:
            running_jobs_ids.append(int(status_line.split()[0]))
        elif servers['local']['cluster_soft'].lower() == 'oge' and i > 1:
            running_jobs_ids.append(int(status_line.split()[0]))
        elif servers['local']['cluster_soft'].lower() == 'pbs' and i > 4:
            running_jobs_ids.append(int(status_line.split('.')[0]))

    return running_jobs_ids


def submit_job(path):
    """
    Submit a job
    `path` is the job's folder path, where the submit script is located (without the submit script file name)
    """
    job_status = ''
    job_id = 0
    cmd = 'cd ' + path + '; ' + submit_command[servers['local']['cluster_soft']] + ' '\
        + submit_filenames[servers['local']['cluster_soft']]
    stdout = execute_command(cmd)[0]
    if servers['local']['cluster_soft'].lower() in ['oge', 'sge'] and 'submitted' in stdout[0].lower():
        job_id = int(stdout[0].split()[2])
        job_status = 'running'
    elif servers['local']['cluster_soft'].lower() == 'slurm' and 'submitted' in stdout[0].lower():
        job_id = int(stdout[0].split()[3])
        job_status = 'running'
    elif servers['local']['cluster_soft'].lower() == 'pbs':
        job_id = int(stdout[0].split('.')[0])
        job_status = 'running'
    else:
        raise ValueError('Unrecognized cluster software {0}'.format(servers['local']['cluster_soft']))
    return job_status, job_id


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


def write_file(file_path, file_string):
    """
    Write `file_string` as the file's content in `file_path`
    """
    with open(file_path, 'w') as f:
        f.write(file_string)


def rename_output(local_file_path, software):
    """
    Rename the output file to "output.out" for consistency between software
    `local_file_path` is the full path to the output.out file,
    `software` is the software used for the job by which the original output file name is determined
    """
    software = software.lower()
    if os.path.isfile(os.path.join(os.path.dirname(local_file_path), output_filenames[software])):
        shutil.move(src=os.path.join(os.path.dirname(local_file_path), output_filenames[software]), dst=local_file_path)


def delete_all_local_arc_jobs(jobs: Optional[List[Union[str, int]]] = None):
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
        cmd = check_status_command[servers[server]['cluster_soft']] + ' -u $USER'
        stdout = execute_command(cmd, no_fail=True)[0]
        for status_line in stdout:
            s = re.search(r' a\d+', status_line)
            if s is not None:
                job_id = s.group()[1:]
                if jobs is None or job_id in jobs:
                    if servers[server]['cluster_soft'].lower() == 'slurm':
                        server_job_id = status_line.split()[0]
                        delete_job(server_job_id)
                        print(f'deleted job {job_id} ({server_job_id} on server)')
                    elif servers[server]['cluster_soft'].lower() == 'pbs':
                        server_job_id = status_line.split()[0]
                        delete_job(server_job_id)
                        print(f'deleted job {job_id} ({server_job_id} on server)')
                    elif servers[server]['cluster_soft'].lower() in ['oge', 'sge']:
                        delete_job(job_id)
                        print(f'deleted job {job_id}')
        print('\ndone.')
