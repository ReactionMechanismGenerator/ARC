"""
A module for SSHing into servers.
Used for giving commands, uploading, and downloading files.

Todo:
    * delete scratch files of a failed job: ssh nodeXX; rm scratch/dhdhdhd/job_number
"""

import datetime
import logging
import os
import time
from typing import Any, Callable, List, Optional, Tuple, Union

import paramiko

from arc.common import get_logger, is_str_int
from arc.exceptions import InputError, ServerError
from arc.imports import settings


logger = get_logger()


check_status_command, delete_command, list_available_nodes_command, servers, submit_command, submit_filenames = \
    settings['check_status_command'], settings['delete_command'], settings['list_available_nodes_command'], \
    settings['servers'], settings['submit_command'], settings['submit_filenames']


def check_connections(function: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator designned for ``SSHClient``to check SSH connections before
    calling a method. It first checks if ``self._ssh`` is available in a 
    SSHClient instance and then checks if you can send ``ls`` and get response
    to make sure your connection still alive. If connection is bad, this
    decorator will reconnect the SSH channel, to avoid connection related
    error when executing the method.
    """
    def decorator(*args, **kwargs) -> Any:
        self = args[0]
        if self._ssh is None:  # not sure if some status may cause False
            self._sftp, self._ssh = self.connect()
        # test connection, reference:
        # https://stackoverflow.com/questions/
        # 20147902/how-to-know-if-a-paramiko-ssh-channel-is-disconnected
        # According to author, maybe no better way
        try:
            self._ssh.exec_command('ls')
        except Exception as e:
            logger.debug(f'The connection is no longer valid. {e}')
            self.connect()
        return function(*args, **kwargs)
    return decorator


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
        _ssh (paramiko.SSHClient): A high-level representation of a session with an SSH server.
        _sftp (paramiko.sftp_client.SFTPClient): SFTP client used to perform remote file operations. 
    """
    def __init__(self, server: str = '') -> None:
        if server == '':
            raise ValueError('A server name must be specified')
        if server not in servers.keys():
            raise ValueError(f'Server name "{server}" is invalid. Currently defined servers are: {list(servers.keys())}')
        self.server = server
        self.address = servers[server]['address']
        self.un = servers[server]['un']
        self.key = servers[server]['key']
        self._sftp = None
        self._ssh = None
        logging.getLogger("paramiko").setLevel(logging.WARNING)

    def __enter__(self) -> 'SSHClient':
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.close()

    @check_connections
    def _send_command_to_server(self, 
                                command: Union[str, list], 
                                remote_path: str = '',
                                ) -> Tuple[list, list]:
        """
        A wrapper for exec_command in paramiko.SSHClient. Send commands to the server. 

        Args:
            command (Union[str, list]): A string or an array of string commands to send.
            remote_path (Optional[str]): The directory path at which the command will be executed.

        Returns: Tuple[list, list]:
            - A list of lines of standard output stream.
            - A list of lines of the standard error stream.
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
        except Exception as e:  # SSHException: Timeout opening channel.
            logger.debug(f'ssh timed-out in the first trial. Got: {e}')
            try:  # try again
                _, stdout, stderr = self._ssh.exec_command(command)
            except Exception as e:
                logger.debug(f'ssh timed-out after two trials. Got: {e}')
                return ['', ], ['ssh timed-out after two trials', ]
        stdout = stdout.readlines()
        stderr = stderr.readlines()
        return stdout, stderr

    def upload_file(self,
                    remote_file_path: str,
                    local_file_path: str = '',
                    file_string: str = '',
                    ) -> None:
        """
        Upload a local file or contents from a string to the remote server.

        Args:
            remote_file_path (str): The path to write into on the remote server.
            local_file_path (Optional[str]): The local file path to be copied to the remote location.
            file_string (Optional[str]): The file content to be copied and saved as the remote file.

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

        try:
            if file_string:
                with self._sftp.open(remote_file_path, 'w') as f_remote:
                    f_remote.write(file_string)
            else:
                self._sftp.put(localpath=local_file_path,
                               remotepath=remote_file_path)
        except IOError:
            logger.debug(f'Could not upload file {local_file_path} to {self.server}!')
            raise ServerError(f'Could not write file {remote_file_path} on {self.server}. ')

    def download_file(self,
                      remote_file_path: str,
                      local_file_path: str,
                      ) -> None:
        """
        Download a file from the server.

        Args:
            remote_file_path (str): The remote path to be downloaded from.
            local_file_path (str): The local path to be downloaded to.

        Raises:
            ServerError: If the file cannot be downloaded with maximum times to try
        """
        if not self._check_file_exists(remote_file_path):
            # Check if a file exists
            # This doesn't have a real impact now to avoid screwing up ESS trsh
            # but introduce an opportunity for better troubleshooting.
            # The current behavior is that if the remote path does not exist
            # an empty file will be created at the local path
            logger.debug(f'{remote_file_path} does not exist on {self.server}.')
        try:
            self._sftp.get(remotepath=remote_file_path,
                           localpath=local_file_path)
        except IOError:
            logger.warning(f'Got an IOError when trying to download file '
                           f'{remote_file_path} from {self.server}')

    @check_connections
    def read_remote_file(self, remote_file_path: str) -> list:
        """
        Read a remote file.

        Args:
            remote_file_path (str): The remote path to be read.
        
        Returns: list
            A list of lines read from the file.
        """
        with self._sftp.open(remote_file_path, 'r') as f_remote:
            content = f_remote.readlines()
        return content

    def check_job_status(self, job_id: int) -> str:
        """
        Check job's status.

        Args:
            job_id (int): The job's ID.

        Returns: str
            Possible statuses: `before_submission`, `running`, `errored on node xx`,
            `done`, and `errored: ...`
        """
        cmd = check_status_command[servers[self.server]['cluster_soft']]
        stdout, stderr = self._send_command_to_server(cmd)
        # Status line formats:
        # OGE: '540420 0.45326 xq1340b    user_name       r     10/26/2018 11:08:30 long1@node18.cluster'
        # SLURM: '14428     debug xq1371m2   user_name  R 50-04:04:46      1 node06'
        if stderr:
            logger.info('\n\n')
            logger.error(f'Could not check status of job {job_id} due to {stderr}')
            return f'errored: {stderr}'
        return check_job_status_in_stdout(job_id=job_id, stdout=stdout, server=self.server)

    def delete_job(self, job_id: Union[int, str]) -> None:
        """
        Deletes a running job.

        Args:
            job_id (Union[int, str]): The job's ID.
        """
        cmd = f"{delete_command[servers[self.server]['cluster_soft']]} {job_id}"
        self._send_command_to_server(cmd)

    def delete_jobs(self,
                    jobs: Optional[List[Union[str, int]]] = None
                    ) -> None:
        """
        Delete all of the jobs on a specific server.

        Args:
            jobs (List[Union[str, int]], optional): Specific ARC job IDs to delete.
        """
        jobs_message = f'{len(jobs)}' if jobs is not None else 'all'
        print(f'\nDeleting {jobs_message} ARC jobs from {self.server}...')
        
        running_job_ids = self.check_running_jobs_ids()
        for job_id in running_job_ids:
            if jobs is None or str(job_id) in jobs:
                self.delete_job(job_id)
                print(f'deleted job {job_id}')

    def check_running_jobs_ids(self) -> list:
        """
        Check all jobs submitted by the user on a server.

        Returns: list
            A list of job IDs.
        """
        if servers[self.server]['cluster_soft'].lower() not in ['slurm', 'oge', 'sge', 'pbs', 'htcondor']:
            raise ValueError(f"Server cluster software {servers['local']['cluster_soft']} is not supported.")
        running_job_ids = list()
        cmd = check_status_command[servers[self.server]['cluster_soft']]
        stdout = self._send_command_to_server(cmd)[0]
        i_dict = {'slurm': 0, 'oge': 1, 'sge': 1, 'pbs': 4, 'htcondor': -1}
        split_by_dict = {'slurm': ' ', 'oge': ' ', 'sge': ' ', 'pbs': '.', 'htcondor': ' '}
        cluster_soft = servers[self.server]['cluster_soft'].lower()
        for i, status_line in enumerate(stdout):
            if i > i_dict[cluster_soft]:
                job_id = status_line.split(split_by_dict[cluster_soft])[0]
                job_id = job_id.split('.')[0] if '.' in job_id else job_id
                running_job_ids.append(job_id)
        return running_job_ids

    def submit_job(self, remote_path: str) -> Tuple[str, int]:
        """
        Submit a job to the server.
        
        Args:
            remote_path (str): The remote path contains the input file
                               and the submission script.

        Returns: Tuple[str, int]
            - A string indicate the status of job submission.
              Either `errored` or `submitted`.
            - The job ID of the submitted job.
        """
        job_status = ''
        job_id = 0
        cluster_soft = servers[self.server]['cluster_soft']
        cmd = f'{submit_command[cluster_soft]} {submit_filenames[cluster_soft]}'
        stdout, stderr = self._send_command_to_server(cmd, remote_path)
        if len(stderr) > 0 or len(stdout) == 0:
            logger.warning(f'Got stderr when submitting job:\n{stderr}')
            job_status = 'errored'
            for line in stderr:
                if 'Requested node configuration is not available' in line:
                    logger.warning(f'User may be requesting more resources than are available. Please check server '
                                   f'settings, such as cpus and memory, in ARC/arc/settings/settings.py')
        elif servers[self.server]['cluster_soft'].lower() in ['oge', 'sge'] and 'submitted' in stdout[0].lower():
            job_id = stdout[0].split()[2]
        elif servers[self.server]['cluster_soft'].lower() == 'slurm' and 'submitted' in stdout[0].lower():
            job_id = stdout[0].split()[3]
        elif servers[self.server]['cluster_soft'].lower() == 'pbs':
            job_id = stdout[0].split('.')[0]
        elif servers[self.server]['cluster_soft'].lower() == 'htcondor' and 'submitting' in stdout[0].lower():
            # Submitting job(s).
            # 1 job(s) submitted to cluster 443069.
            if len(stdout) and len(stdout[1].split()) and len(stdout[1].split()[-1].split('.')):
                job_id = stdout[1].split()[-1][:-1]
        else:
            raise ValueError(f'Unrecognized cluster software: {cluster_soft}')
        job_status = 'running' if job_id else job_status
        return job_status, job_id

    def connect(self) -> None:
        """
        A modulator function for _connect(). Connect to the server.

        Raises:
            ServerError: Cannot connect to the server with maximum times to try
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

    def _connect(self) -> Tuple[paramiko.sftp_client.SFTPClient, paramiko.SSHClient]:
        """
        Connect via paramiko, and open a SSH session as well as a SFTP session.

        Returns: Tuple[paramiko.sftp_client.SFTPClient, paramiko.SSHClient]
            - An SFTP client used to perform remote file operations.
            - A high-level representation of a session with an SSH server.
        """
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.load_system_host_keys(filename=self.key)
        try:
            # If the server accepts the connection but the SSH daemon doesn't respond in 
            # 15 seconds (default in paramiko) due to network congestion, faulty switches, 
            # etc..., common solution is to enlarging the timeout variable.
            ssh.connect(hostname=self.address, username=self.un, banner_timeout=200)
        except:
            # This sometimes gives "SSHException: Error reading SSH protocol banner[Error 104] Connection reset by peer"
            # Try again:
            ssh.connect(hostname=self.address, username=self.un, banner_timeout=200)
        sftp = ssh.open_sftp()
        return sftp, ssh

    def close(self) -> None:
        """
        Close the connection to paramiko SSHClient and SFTPClient
        """
        if self._sftp is not None:
            self._sftp.close()
        if self._ssh is not None:
            self._ssh.close()

    @check_connections
    def get_last_modified_time(self, 
                               remote_file_path_1: str,
                               remote_file_path_2: Optional[str],
                               ) -> Optional[datetime.datetime]:
        """
        Returns the last modified time of ``remote_file_path_1`` if the file exists,
        else returns the last modified time of ``remote_file_path_2`` if the file exists.

        Args:
            remote_file_path_1 (str): The remote path to file 1.
            remote_file_path_2 (str, optional): The remote path to file .

        Returns: datetime.datetime
            The last modified time of the file.
        """
        timestamp = None
        try:
            timestamp = self._sftp.stat(remote_file_path_1).st_mtime
        except IOError:
            pass
        if timestamp is None and remote_file_path_2 is not None:
            try:
                timestamp = self._sftp.stat(remote_file_path_2).st_mtime
            except IOError:
                return None
        return datetime.datetime.fromtimestamp(timestamp) if timestamp is not None else None

    def list_dir(self, remote_path: str = '') -> list:
        """
        List directory contents.

        Args:
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

        Returns:
            list: lines of the node hostnames.
        """
        cluster_soft = servers[self.server]['cluster_soft'].lower()
        if cluster_soft == 'htcondor':
            return list()
        cmd = list_available_nodes_command[cluster_soft]
        stdout = self._send_command_to_server(command=cmd)[0]
        nodes = list()
        if cluster_soft.lower() in ['oge', 'sge']:
            # Stdout line example:
            # long1@node01.cluster           BIP   0/0/8          -NA-     lx24-amd64    aAdu
            nodes = [line.split()[0].split('@')[1]
                     for line in stdout if '0/0/8' in line]
        elif cluster_soft.lower() == 'slurm':
            # Stdout line example:
            # node01 alloc 1.00 none
            nodes = [line.split()[0] for line in stdout
                     if line.split()[1] in ['mix', 'alloc', 'idle']]
        elif cluster_soft.lower() in ['pbs', 'htcondor']:
            logger.warning(f'Listing available nodes is not yet implemented for {cluster_soft}.')
        return nodes

    def change_mode(self,
                    mode: str,
                    file_name: str,
                    recursive: bool = False,
                    remote_path: str = '',
                    ) -> None:
        """
        Change the mode of a file or a directory.

        Args:
            mode (str): The mode change to be applied, can be either octal or symbolic.
            file_name (str): The path to the file or the directory to be changed.
            recursive (bool, optional): Whether to recursively change the mode to all files
                                        under a directory.``True`` for recursively change.
            remote_path (str, optional): The directory path at which the command will be executed.
        """
        if os.path.isfile(remote_path):
            remote_path = os.path.dirname(remote_path)
        recursive = ' -R' if recursive else ''
        command = f'chmod{recursive} {mode} {file_name}'
        self._send_command_to_server(command, remote_path)

    def _check_file_exists(self, 
                           remote_file_path: str,
                           ) -> bool:
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
                          remote_dir_path: str,
                          ) -> bool:
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

    def _create_dir(self, remote_path: str) -> None:
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


def check_job_status_in_stdout(job_id: int, 
                               stdout: Union[list, str],
                               server: str,
                               ) -> str:
    """
    A helper function for checking job status.

    Args:
        job_id (int): the job ID recognized by the server.
        stdout (Union[list, str]): The output of a queue status check.
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
    if servers[server]['cluster_soft'].lower() == 'slurm':
        status = status_line.split()[4]
        if status.lower() in ['r', 'qw', 't', 'cg', 'pd']:
            return 'running'
        elif status.lower() in ['bf', 'ca', 'f', 'nf', 'st', 'oom']:
            return 'errored'
    elif servers[server]['cluster_soft'].lower() == 'pbs':
        status = status_line.split()[-2]
        if status.lower() in ['r', 'q', 'c', 'e', 'w']:
            return 'running'
        elif status.lower() in ['h', 's']:
            return 'errored'
    elif servers[server]['cluster_soft'].lower() in ['oge', 'sge']:
        status = status_line.split()[4]
        if status.lower() in ['r', 'qw', 't']:
            return 'running'
        elif status.lower() in ['e',]:
            return 'errored'
    elif servers[server]['cluster_soft'].lower() == 'htcondor':
        return 'running'
    else:
        raise ValueError(f'Unknown cluster software {servers[server]["cluster_soft"]}')


def delete_all_arc_jobs(server_list: list,
                        jobs: Optional[List[str]] = None,
                        ) -> None:
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
        with SSHClient(server) as ssh:
            ssh.delete_jobs(jobs)
    if server_list:
        print('\ndone.')
