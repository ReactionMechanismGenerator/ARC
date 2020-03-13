#!/usr/bin/env python3
# encoding: utf-8

"""
A module for SSHing into servers.
Used for giving commands, uploading, and downloading files.

Todo:
    * delete scratch files of a failed job: ssh nodeXX; rm scratch/dhdhdhd/job_number
"""

import datetime
import logging
import os
import re
import time

import paramiko

from arc.common import get_logger
from arc.exceptions import InputError, ServerError
from arc.settings import check_status_command, delete_command, list_available_nodes_command, \
                         servers, submit_command, submit_filename


logger = get_logger()


class SSHClient(object):
    """
    This is a class for communicating with remote servers via SSH.

    Args:
        server (str): The server name as specified in ARCs's settings file under ``servers`` as a key.

    Attributes:
        server (str): The server name as specified in ARCs's settings file under ``servers`` as a key.
        address (str): The server's address.
        un (str): The username to use on the server.
        key (str): A path to a file containing the RSA SSH private key to the server.
    """
    def __init__(self, server=''):
        if server == '':
            raise ValueError('A server name must be specified')
        if server not in servers.keys():
            raise ValueError(f'Server name invalid. Currently defined servers are: {servers.keys()}')
        self.server = server
        self.address = servers[server]['address']
        self.un = servers[server]['un']
        self.key = servers[server]['key']
        self._sftp = None
        self._ssh = None
        logging.getLogger("paramiko").setLevel(logging.WARNING)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def _send_command_to_server(self, command: str, remote_path: str='') -> (list, list):
        """
        A wapper for exec_command in paramiko. SSHClient. Send commands to the server. 

        Args:
            command (str or list): A string or an array of string commands to send.
            remote_path (str, optional): The directory path at which the command will be executed.

        Returns:
            list: A list of lines of standard output stream.

        Returns:
            list: A list of lines of standard error stream.
        """
        if isinstance(command, list):
            command = '; '.join(command)
        if remote_path != '':
            # execute command in remote_path directory.
            # Check remote path existence, otherwise the cmd will be invalid
            # and even yield different behaviors.
            # Make sure to change directory back after the command is executed
            if self._check_dir_exists(remote_path):
                command = f'cd "{remote_path}"; {command}; cd '
            else:
                raise InputError(
                    f'Cannot execute command at given remote_path({remote_path})')
        try:
            _, stdout, stderr = self._ssh.exec_command(command)
        except:  # SSHException: Timeout opening channel.
            try:  # try again
                _, stdout, stderr = self._ssh.exec_command(command)
            except:
                return '', 'ssh timed-out after two trials'
        stdout = stdout.readlines()
        stderr = stderr.readlines()
        return stdout, stderr

    def upload_file(self, remote_file_path: str, local_file_path: str='', file_string: str=''):
        """
        A modulator method of _upload_file(). Upload a local file or contents
        from a string to the remote server.

        Args:
            remote_file_path (str): The path to write into on the remote server.
            local_file_path (str, optional): The local file path to be copied to the remote location.
            file_string (str, optional): The file content to be copied and saved as the remote file.

        Raises:
            InputError: If both `local_file_path` or `file_string` are invalid,
                        or `local_file_path` does not exists.
            ServerError: If the file cannot be uploaded with maximum times to try
        """
        if not local_file_path and not file_string:
            raise InputError('Cannot not upload file to server. Either `file_string` or `local_file_path`'
                             ' must be specified')
        if local_file_path and not os.path.isfile(local_file_path):
            raise InputError(f'Cannot upload a non-existing file. '
                             f'Check why file in path {local_file_path} is missing.')
        # If the directory does not exist, _upload_file cannot create a file based on the given path
        remote_dir_path = os.path.dirname(remote_file_path)
        if not self._check_dir_exists(remote_dir_path):
            self._create_dir(remote_dir_path)

        i, max_times_to_try = 1, 30
        success = False
        sleep_time = 10  # seconds
        while i < max_times_to_try and not success:
            try:
                self._upload_file(remote_file_path,
                                  local_file_path, file_string)
            except IOError:
                logger.error(f'Could not upload file {local_file_path} to {self.server}!')
                logger.error(f'ARC is sleeping for {sleep_time * i} seconds before re-trying, '
                             f'please check your connectivity.')
                logger.info('ZZZZZ..... ZZZZZ.....')
                time.sleep(sleep_time * i)  # in seconds
                i += 1
            else:
                success = True
        if not success:
            raise ServerError(f'Could not write file {remote_file_path} on {self.server}. '
                              f'Tried {max_times_to_try} times.')

    def _upload_file(self, remote_file_path: str, local_file_path: str = '', file_string: str = ''):
        """
        Upload a file. If `file_string` is given, write it as the content of the file.
        Else, if `local_file_path` is given, copy it to `remote_file_path`.

        Args:
            remote_file_path (str): The path to write into on the remote server.
            local_file_path (str, optional): The local file path to be copied to the remote location.
            file_string (str, optional): The file content to be copied and saved as the remote file.
        """
        try:
            if file_string:
                with self._sftp.open(remote_file_path, 'w') as f_remote:
                    f_remote.write(file_string)
            else:
                self._sftp.put(localpath=local_file_path,
                               remotepath=remote_file_path)
        except IOError:
            logger.debug(
                f'Got an IOError when trying to upload file {remote_file_path} from {self.server}')
            raise IOError(
                f'Got an IOError when trying to upload file {remote_file_path} from {self.server}')

    def download_file(self, remote_file_path: str, local_file_path: str):
        """
        A modulator function of _download_file(). Download a file from the server.

        Args:
            remote_file_path (str): The remote path to be downloaded from.
            local_file_path (str): The local path to be downloaded to.

        Raises:
            ServerError: If the file cannot be downloaded with maximum times to try
        """
        i, max_times_to_try = 1, 30
        success = False
        sleep_time = 10  # seconds

        if not self._check_file_exists(remote_file_path):
            # Check if a file exists
            # This doesn't have a real impact now to avoid screwing up ESS trsh
            # but introduce an opportunity for better troubleshooting.
            # The current behavior is that if the remote path does not exist
            # an empty file will be created at the local path
            logger.debug(
                f'{remote_file_path} does not exist on {self.server}.')

        while i < max_times_to_try and not success:
            self._download_file(remote_file_path, local_file_path)
            if os.path.isfile(local_file_path):
                success = True
            else:
                logger.error(f'Could not download file {remote_file_path} from {self.server}!')
                logger.error(f'ARC is sleeping for {sleep_time * i} seconds before re-trying, '
                             f'please check your connectivity.')
                logger.info('ZZZZZ..... ZZZZZ.....')
                time.sleep(sleep_time * i)  # in seconds
            i += 1
        if not success:
            raise ServerError(f'Could not download file {remote_file_path} from {self.server}. '
                              f'Tried {max_times_to_try} times.')

    def _download_file(self, remote_file_path: str, local_file_path: str):
        """
        Download a file from the server.

        Args:
            remote_file_path (str): The remote path to be downloaded from.
            local_file_path (str): The local path to be downloaded to.

        Raises:
            IOError: Cannot download file via sftp.
        """
        try:
            self._sftp.get(remotepath=remote_file_path,
                           localpath=local_file_path)
        except IOError:
            logger.debug(
                f'Got an IOError when trying to download file {remote_file_path} from {self.server}')

    def read_remote_file(self, remote_file_path: str) -> list:
        """
        Read a remote file.

        Args:
            remote_file_path (str): The remote path to be read.
        
        Returns:
            list: A list of lines read from the file.
        """
        with self._sftp.open(remote_file_path, 'r') as f_remote:
            content = f_remote.readlines()
        return content

    def check_job_status(self, job_id: int) -> str:
        """
        A modulator method of _check_job_status(). Check job's status.

        Args:
            job_id (int): The job's ID.

        Returns:
            str: Possible statuses: `before_submission`, `running`, `errored on node xx`,
                 `done`, and `connection error`
        """
        i = 1
        sleep_time = 1  # minutes
        while i < 30:
            result = self._check_job_status(job_id)
            if result == 'connection error':
                logger.error(f'ARC is sleeping for {sleep_time * i} min before re-trying, '
                             f'please check your connectivity.')
                logger.info('ZZZZZ..... ZZZZZ.....')
                time.sleep(sleep_time * i * 60)  # in seconds
            else:
                i = 1000
            i += 1
        return result

    def _check_job_status(self, job_id: int) -> str:
        """
        Check job's status.

        Args:
            job_id (int): The job's ID.
        
        Returns:
            str: Possible statuses: `before_submission`, `running`, `errored on node xx`,
                 `done`, and `connection error`
        """
        cmd = check_status_command[servers[self.server]['cluster_soft']] + ' -u $USER'
        stdout, stderr = self._send_command_to_server(cmd)
        # Status line formats:
        # OGE: '540420 0.45326 xq1340b    user_name       r     10/26/2018 11:08:30 long1@node18.cluster'
        # SLURM: '14428     debug xq1371m2   user_name  R 50-04:04:46      1 node06'
        if stderr:
            logger.info('\n\n')
            logger.error(f'Could not check status of job {job_id} due to {stderr}')
            return 'connection error'
        return check_job_status_in_stdout(job_id=job_id, stdout=stdout, server=self.server)

    def delete_job(self, job_id: int):
        """
        Deletes a running job.

        Args:
            job_id (int): The job's ID.
        """
        cmd = delete_command[servers[self.server]['cluster_soft']] + ' ' + str(job_id)
        self._send_command_to_server(cmd)

    def check_running_jobs_ids(self) -> list:
        """
        Check all jobs submitted by the user on a server.

        Returns:
            list: A list of job IDs
        """
        running_jobs_ids = list()
        cmd = check_status_command[servers[self.server]['cluster_soft']] + ' -u $USER'
        stdout = self._send_command_to_server(cmd)[0]
        for i, status_line in enumerate(stdout):
            if (servers[self.server]['cluster_soft'].lower() == 'slurm' and i > 0)\
                    or (servers[self.server]['cluster_soft'].lower() == 'oge' and i > 1):
                running_jobs_ids.append(int(status_line.split()[0]))
        return running_jobs_ids

    def submit_job(self, remote_path: str) -> (str, int):
        """
        Submit a job to the server.
        
        Args:
            remote_path (str): The remote path contains the input file and the submission script.

        Returns:
            str: A string indicate the status of job submission. Either `errored` or `submitted`.

        Returns:
            int: the job ID of the submitted job.
        """
        job_status = ''
        job_id = 0
        cluster_soft = servers[self.server]['cluster_soft']
        cmd = submit_command[cluster_soft] + ' ' + submit_filename[cluster_soft]
        stdout, stderr = self._send_command_to_server(cmd, remote_path)
        if len(stderr) > 0 or len(stdout) == 0:
            logger.warning(f'Got stderr when submitting job:\n{stderr}')
            job_status = 'errored'
            for line in stderr:
                if 'Requested node configuration is not available' in line:
                    logger.warning(f'User may be requesting more resources than are available. Please check server '
                                   f'settings, such as cpus and memory, in ARC/arc/settings.py')
        elif 'submitted' in stdout[0].lower():
            job_status = 'running'
            if cluster_soft.lower() == 'oge':
                job_id = int(stdout[0].split()[2])
            elif cluster_soft.lower() == 'slurm':
                job_id = int(stdout[0].split()[3])
            else:
                raise ValueError(f'Unrecognized cluster software {servers[self.server]["cluster_soft"]}')
        return job_status, job_id

    def connect(self):
        """
        A modulator function for _connect(). Connect to the server.

        Raises:
            ServerError: Cannot connect to the server with maximum times to try
            
        Returns:
            paramiko.sftp_client.SFTPClient

        Returns:
            paramiko.SSHClient
        """
        times_tried = 0
        max_times_to_try = 1440  # continue trying for 24 hrs (24 hr * 60 min/hr)...
        interval = 60  # wait 60 sec between trials
        while times_tried < max_times_to_try:
            times_tried += 1
            try:
                self._sftp, self._ssh = self._connect()
            except Exception as e:
                if not times_tried % 10:
                    logger.info(f'Tried connecting to {self.server} {times_tried} times with no success...'
                                f'\nGot: {e}')
                else:
                    print(f'Tried connecting to {self.server} {times_tried} times with no success...'
                          f'\nGot: {e}')
            else:
                logger.debug(f'Successfully connected to {self.server} at the {times_tried} trial.')
                return
            time.sleep(interval)
        raise ServerError(f'Could not connect to server {self.server} even after {times_tried} trials.')

    def _connect(self):
        """
        Connect via paramiko, and open a SSH session as well as a SFTP session.

        Returns:
            paramiko.sftp_client.SFTPClient

        Returns:
            paramiko.SSHClient
        """
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.load_system_host_keys(filename=self.key)
        try:
            ssh.connect(hostname=self.address, username=self.un)
        except:
            # This sometimes gives "SSHException: Error reading SSH protocol banner[Error 104] Connection reset by peer"
            # Try again:
            ssh.connect(hostname=self.address, username=self.un)
        sftp = ssh.open_sftp()
        return sftp, ssh

    def close(self):
        """
        Close the connection to paramiko SSHClient and SFTPClient
        """
        if self._sftp is not None:
            self._sftp.close()
        if self._ssh is not None:
            self._ssh.close()


    def get_last_modified_time(self, remote_file_path: str):
        """
        Get the last modified time of a remote file.

        Args:
            remote_file_path (str): The remote file path to check.

        Returns:
            datetime.datetime: the last modified time of the file
        """
        try:
            timestamp = self._sftp.stat(remote_file_path).st_mtime
        except IOError:
            return None
        return datetime.datetime.fromtimestamp(timestamp)

    def list_dir(self, remote_path: str = '') -> list:
        """
        List directory contents.

        Args:
            mode (str): The mode change to be applied, can be either octal or symbolic.
            remote_path (str, optional): The directory path at which the command will be executed.
        """
        command = f'ls -alF'
        return self._send_command_to_server(command, remote_path)[0]

    def find_package(self, package_name: str) -> list:
        """
        Find the path to the package.

        Args:
            package_name (str): The name of the package to search for.
        """
        command = f'. ~/.bashrc; which {package_name}'
        return self._send_command_to_server(command)[0]

    def list_available_nodes(self) -> list:
        """
        List available nodes on the server.

        Args:
            mode (str): The mode change to be applied, can be either octal or symbolic.

        Returns:
            list: lines of the node hostnames.
        """
        cluster_soft = servers[self.server]['cluster_soft']
        cmd = list_available_nodes_command[cluster_soft]
        stdout = self._send_command_to_server(command=cmd)[0]
        if cluster_soft.lower() == 'oge':
            # Stdout line example:
            # long1@node01.cluster           BIP   0/0/8          -NA-     lx24-amd64    aAdu
            nodes = [line.split()[0].split('@')[1]
                     for line in stdout if '0/0/8' in line]
        elif cluster_soft.lower() == 'slurm':
            # Stdout line example:
            # node01 alloc 1.00 none
            nodes = [line.split()[0] for line in stdout
                     if line.split()[1] in ['mix', 'alloc', 'idle']]
        return nodes

    def change_mode(self,
                    mode: str,
                    path: str,
                    recursive: bool = False,
                    remote_path: str = ''):
        """
        Change the mode to a file or a directory.

        Args:
            mode (str): The mode change to be applied, can be either octal or symbolic.
            path (str): The path to the file or the directory to be changed.
            recursive (bool, optional): Whether to recursively change the mode to all files
                                        under a directory.``True`` for recursively change.
            remote_path (str, optional): The directory path at which the command will be executed.
        """
        recursive = '-R' if recursive else ''
        command = f'chmod {recursive} {mode} {path}'
        self._send_command_to_server(command, remote_path)

    def _check_file_exists(self, remote_file_path: str) -> bool:
        """
        Check if a file exists on the remote server.

        Args:
            remote_file_path (str): The path to the file on the remote server.

        Returs:
            bool: If the file exists on the remote server. ``True`` if exist.
        """
        command = f'[ -f "{remote_file_path}" ] && echo "File exists"'
        stdout, _ = self._send_command_to_server(command, remote_path='')
        if len(stdout):
            return True

    def _check_dir_exists(self,
                          remote_dir_path: str) -> bool:
        """
        Check if a directory exists on the remote server.

        Args:
            remote_dir_path (str): The path to the directory on the remote server.

        Returns:
            bool: If the directory exists on the remote server. ``True`` if exist.
        """
        command = f'[ -d "{remote_dir_path}" ] && echo "Dir exists"'
        stdout, _ = self._send_command_to_server(command)
        if len(stdout):
            return True

    def _create_dir(self, remote_path: str):
        """
        Create a new directory on the server.

        Args:
            remote_path (str): The path to the directory to create on the remote server.
        """
        command = f'mkdir -p "{remote_path}"'
        _, stderr = self._send_command_to_server(command)
        if stderr:
            raise ServerError(
                f'Cannot create dir for the given path ({remote_path}).\nGot: {stderr}')


def check_job_status_in_stdout(job_id, stdout, server):
    """
    A helper function for checking job status.

    Args:
        job_id (int): the job ID recognized by the server.
        stdout (list, str): The output of a queue status check.
        server (str): The server name.

    Returns:
        str: The job status on the server ('running', 'done', or 'errored').
    """
    if not isinstance(stdout, list):
        stdout = stdout.splitlines()
    for status_line in stdout:
        if str(job_id) in status_line:
            break
    else:
        return 'done'
    status = status_line.split()[4]
    if status.lower() in ['r', 'qw', 't']:
        return 'running'
    else:
        if servers[server]['cluster_soft'].lower() == 'oge':
            if '.cluster' in status_line:
                try:
                    return 'errored on node ' + status_line.split()[-1].split('@')[1].split('.')[0][-2:]
                except IndexError:
                    return 'errored'
            else:
                return 'errored'
        elif servers[server]['cluster_soft'].lower() == 'slurm':
            return 'errored on node ' + status_line.split()[-1][-2:]
        else:
            raise ValueError(f'Unknown cluster software {servers[server]["cluster_soft"]}')


def delete_all_arc_jobs(server_list, jobs=None):
    """
    Delete all ARC-spawned jobs (with job name starting with `a` and a digit) from :list:servers
    (`servers` could also be a string of one server name)
    Make sure you know what you're doing, so unrelated jobs won't be deleted...
    Useful when terminating ARC while some (ghost) jobs are still running.

    Args:
        server_list (list): List of servers to delete ARC jobs from.
        jobs (Optional[List[str]]): Specific ARC job IDs to delete.
    """
    if isinstance(server_list, str):
        server_list = [server_list]
    for server in server_list:
        jobs_message = f'{len(jobs)}' if jobs is not None else 'all'
        print(f'\nDeleting {jobs_message} ARC jobs from {server}...')
        cmd = check_status_command[servers[server]['cluster_soft']] + ' -u $USER'
        ssh = SSHClient(server)
        stdout = ssh._send_command_to_server(cmd)[0]
        for status_line in stdout:
            s = re.search(r' a\d+', status_line)
            if s is not None:
                job_id = s.group()[1:]
                if job_id in jobs or jobs is None:
                    if servers[server]['cluster_soft'].lower() == 'slurm':
                        server_job_id = status_line.split()[0]
                        ssh.delete_job(server_job_id)
                        print(f'deleted job {job_id} ({server_job_id} on server)')
                    elif servers[server]['cluster_soft'].lower() == 'oge':
                        ssh.delete_job(job_id)
                        print(f'deleted job {job_id}')
    if server_list:
        print('\ndone.')
