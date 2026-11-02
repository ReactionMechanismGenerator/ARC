"""
A module for SSHing into servers.
Used for giving commands, uploading, and downloading files.

Todo:
    * delete scratch files of a failed job: ssh nodeXX; rm scratch/dhdhdhd/job_number
"""

import base64
import datetime
import hashlib
import logging
import os
import shlex
import time
from collections import Counter
from typing import Any
from collections.abc import Callable

import paramiko

from arc.common import get_logger
from arc.exceptions import InputError, ServerError
from arc.imports import settings


logger = get_logger()

check_status_command, delete_command, list_available_nodes_command, servers, submit_command, submit_filenames = \
    settings['check_status_command'], settings['delete_command'], settings['list_available_nodes_command'], \
    settings['servers'], settings['submit_command'], settings['submit_filenames']

KNOWN_HOSTS_PATH = '~/.ssh/known_hosts'
PLACEHOLDER_ADDRESS_SUFFIX = '.host.edu'
PLACEHOLDER_USERNAME = '<username>'


class UnknownHostKeyError(ServerError, paramiko.SSHException):
    """
    Raised when a server's host key is absent from ``known_hosts`` and the server sets
    ``strict_host_key_checking``.

    An ARC :class:`~arc.exceptions.ServerError`, so ARC's server error handling covers it, and a
    ``paramiko.SSHException``, which is what a missing host key policy is expected to raise. Being
    a distinct type, a refused host key is told apart from a transport failure without matching on
    paramiko's message text.
    """


def get_host_key_fingerprint(key: paramiko.PKey) -> str:
    """
    Return the OpenSSH-style SHA256 fingerprint of a host key.

    Args:
        key (paramiko.PKey): The host key to fingerprint.

    Returns: str
        The fingerprint, formatted as ``SHA256:<unpadded base64>``, as reported by
        ``ssh-keygen -lf`` and by OpenSSH when it prompts about an unknown host.
    """
    digest = hashlib.sha256(key.asbytes()).digest()
    return f'SHA256:{base64.b64encode(digest).decode().rstrip("=")}'


class LogAndAcceptHostKeyPolicy(paramiko.MissingHostKeyPolicy):
    """
    A missing host key policy that reports the unknown key through ARC's logger, then accepts it.

    paramiko's ``WarningPolicy`` emits through ``warnings.warn()``, which ARC's
    :func:`arc.common.initialize_log` filters out for the ``paramiko`` module, so nothing of it
    reaches the log file or the terminal. This policy logs the host and the key's fingerprint at
    the warning level, and connects.
    """

    def missing_host_key(self,
                         client: paramiko.SSHClient,
                         hostname: str,
                         key: paramiko.PKey,
                         ) -> None:
        """
        Log the unknown host key and accept it.

        Args:
            client (paramiko.SSHClient): The client the key was presented to.
            hostname (str): The address of the server that presented the key.
            key (paramiko.PKey): The host key that is not in ``known_hosts``.
        """
        logger.warning(f'Connecting to {hostname} with an unverified host key: '
                       f'{key.get_name()} {get_host_key_fingerprint(key)} is not in '
                       f'{KNOWN_HOSTS_PATH}, so a first-ever connection cannot be told apart '
                       f'from an interception. Verify the fingerprint against a trusted source '
                       f'and add the key with:\n'
                       f'    ssh-keyscan -H {hostname} >> {KNOWN_HOSTS_PATH}')


class RejectUnknownHostKeyPolicy(paramiko.RejectPolicy):
    """
    A missing host key policy that refuses the connection, raising :class:`UnknownHostKeyError`.
    """

    def missing_host_key(self,
                         client: paramiko.SSHClient,
                         hostname: str,
                         key: paramiko.PKey,
                         ) -> None:
        """
        Refuse the unknown host key.

        Args:
            client (paramiko.SSHClient): The client the key was presented to.
            hostname (str): The address of the server that presented the key.
            key (paramiko.PKey): The host key that is not in ``known_hosts``.

        Raises:
            UnknownHostKeyError: Always.
        """
        raise UnknownHostKeyError(f'The host key of {hostname} '
                                  f'({key.get_name()} {get_host_key_fingerprint(key)}) is not in '
                                  f'{KNOWN_HOSTS_PATH}, and this server sets '
                                  f'strict_host_key_checking. Verify the fingerprint against a '
                                  f'trusted source and add the key with:\n'
                                  f'    ssh-keyscan -H {hostname} >> {KNOWN_HOSTS_PATH}')


class HostKeyMismatchError(ServerError, paramiko.SSHException):
    """
    Raised when the host key a server presents contradicts the one stored in ``known_hosts``.

    Told apart from :class:`UnknownHostKeyError`, which is the absence of a stored key, because
    the two mean different things: an absent key is a host never connected to before, while a
    contradicted key is either a re-keyed server or an interception, and only the second of
    those is a security event. Being an ARC :class:`~arc.exceptions.ServerError` keeps it inside
    ARC's server error handling, and being a ``paramiko.SSHException`` keeps it catchable
    alongside the exception it is raised from.
    """


PERMANENT_CONNECTION_ERRORS = (paramiko.AuthenticationException,
                               paramiko.BadHostKeyException,
                               UnknownHostKeyError,
                               )


def check_connections(function: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator designned for ``SSHClient``to check SSH connections before
    calling a method. It first checks if ``self._ssh`` is available in a
    SSHClient instance and then checks if you can send ``ls`` and get response
    to make sure your connection still alive. If connection is bad, this
    decorator will reconnect the SSH channel, to avoid connection related
    error when executing the method.

    ``connect()`` assigns ``self._sftp`` and ``self._ssh`` itself and returns nothing, so its
    result is not unpacked into them. Unpacking it raised ``TypeError: cannot unpack
    non-iterable NoneType object`` for any client that had not connected yet, which is the one
    case this branch exists to serve.
    """
    def decorator(*args, **kwargs) -> Any:
        self = args[0]
        if self._ssh is None:  # not sure if some status may cause False
            self.connect()
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
        connection_attempts (int, optional): The number of times to try connecting to the server,
                                             waiting a minute between attempts. The default keeps trying
                                             for 24 hours, which is appropriate while jobs are running.
                                             Pass a low number where blocking is worse than giving up.
                                             A permanent failure raises on the first attempt whatever
                                             this is set to, see :meth:`connect`.

    Attributes:
        server (str): The server name as specified in ARCs's settings file under ``servers`` as a key.
        address (str): The server's address.
        un (str): The username to use on the server.
        key (str | None): A path to a file containing the SSH private key to the server.
                          Optional: when it is not set (or set to an empty string), no explicit
                          identity is offered and paramiko falls back to a running ssh-agent and
                          then to the default key paths (``~/.ssh/id_rsa``, ``~/.ssh/id_ecdsa``,
                          ``~/.ssh/id_ed25519``), which is how agent-based setups authenticate.
        connection_attempts (int): The number of times to try connecting to the server.
        _ssh (paramiko.SSHClient): A high-level representation of a session with an SSH server.
        _sftp (paramiko.sftp_client.SFTPClient): SFTP client used to perform remote file operations.
        _keepalive_interval (int | None): The keepalive interval this client was asked for, which
                                          :meth:`connect` re-applies to every transport it opens.
                                          ``None`` until :func:`arc.job.ssh_pool.set_keepalive`
                                          sets it, since a client that is not pooled is not held
                                          open long enough to be dropped while idle.
    """
    def __init__(self,
                 server: str = '',
                 connection_attempts: int = 1440,
                 ) -> None:
        if server == '':
            raise ValueError('A server name must be specified')
        if server not in servers.keys():
            raise ValueError(f'Server name "{server}" is invalid. Currently defined servers are: {list(servers.keys())}')
        self.server = server
        self.address = servers[server]['address']
        self.un = servers[server]['un']
        self.key = servers[server].get('key') or None
        self.connection_attempts = connection_attempts
        self._sftp = None
        self._ssh = None
        self._keepalive_interval = None
        logging.getLogger("paramiko").setLevel(logging.WARNING)

    def __enter__(self) -> SSHClient:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.close()

    @check_connections
    def _send_command_to_server(self,
                                command: str | list,
                                remote_path: str = '',
                                ) -> tuple[list, list]:
        """
        A wrapper for exec_command in paramiko.SSHClient. Send commands to the server.

        Args:
            command (str | list): A string or an array of string commands to send.
            remote_path (str | None): The directory path at which the command will be executed.

        Returns: tuple[list, list]:
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
                command = f'cd -- {shlex.quote(remote_path)}; {command}; cd '
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

    @check_connections
    def upload_file(self,
                    remote_file_path: str,
                    local_file_path: str = '',
                    file_string: str = '',
                    ) -> None:
        """
        Upload a local file or contents from a string to the remote server.

        Args:
            remote_file_path (str): The path to write into on the remote server.
            local_file_path (str | None): The local file path to be copied to the remote location.
            file_string (str | None): The file content to be copied and saved as the remote file.

        Raises:
            InputError: If both `local_file_path` or `file_string` are invalid,
                        or `local_file_path` does not exist.
            ServerError: If the file cannot be uploaded with maximum times to try
        """
        if not local_file_path and not file_string:
            raise InputError('Cannot upload file to server. Either `file_string` or `local_file_path`'
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

    @check_connections
    def download_file(self,
                      remote_file_path: str,
                      local_file_path: str,
                      ) -> None:
        """
        Download a file from the server.

        The existence of the remote file is checked up to three times, one second apart, since
        scheduler epilogues may flush a job's stdout and stderr to the work directory a second or
        two after the job has left the queue.

        If the remote file does not exist, the local path is emptied instead of being downloaded
        to: it is created if it is absent, and truncated if a file is already there. A file left
        at that path by an earlier job therefore never survives to be read as this job's output
        (see :meth:`arc.job.adapter.JobAdapter._get_additional_job_info`), and the miss is
        reported at the warning level.

        Args:
            remote_file_path (str): The remote path to be downloaded from.
            local_file_path (str): The local path to be downloaded to.

        Raises:
            ServerError: If the file cannot be downloaded with maximum times to try
        """
        for attempt in range(3):
            if self._check_file_exists(remote_file_path):
                break
            if attempt < 2:
                time.sleep(1.0)
        else:
            logger.warning(f'{remote_file_path} does not exist on {self.server}. '
                           f'Emptied {local_file_path} instead of downloading it.')
            self._empty_local_file(local_file_path)
            return
        try:
            self._sftp.get(remotepath=remote_file_path,
                           localpath=local_file_path)
        except IOError:
            logger.warning(f'Got an IOError when trying to download file '
                           f'{remote_file_path} from {self.server}')

    @staticmethod
    def _empty_local_file(local_file_path: str) -> None:
        """
        Create ``local_file_path`` if it does not exist, and truncate it if it does.

        Args:
            local_file_path (str): The local path to empty.
        """
        try:
            with open(local_file_path, 'wb'):
                pass
        except OSError as e:
            logger.warning(f'Could not empty the local file {local_file_path}. '
                           f'Got {type(e).__name__}: {e}')

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

    def delete_job(self, job_id: int | str) -> None:
        """
        Deletes a running job.

        Args:
            job_id (int | str): The job's ID.
        """
        cmd = f"{delete_command[servers[self.server]['cluster_soft']]} {job_id}"
        self._send_command_to_server(cmd)

    def delete_jobs(self,
                    jobs: list[str | int] | None = None
                    ) -> None:
        """
        Delete all of the jobs on a specific server.

        Args:
            jobs (list[str | int], optional): Specific ARC job IDs to delete.
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
                job_id = status_line.lstrip().split(split_by_dict[cluster_soft])[0]
                job_id = job_id.split('.')[0] if '.' in job_id else job_id
                running_job_ids.append(job_id)
        return running_job_ids

    def submit_job(self, remote_path: str,
                   recursion: bool = False,
                   ) -> tuple[str | None, str | None]:
        """
        Submit a job to the server.

        Args:
            remote_path (str): The remote path contains the input file and the submission script.
            recursion (bool, optional): Whether this call is within a recursion.

        Returns: tuple[str, int]
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
                    logger.warning('User may be requesting more resources than are available. Please check server '
                                   'settings, such as cpus and memory, in ARC/arc/settings/settings.py')
                if 'Memory specification can not be satisfied' in line:
                    logger.warning('User may be requesting more memory than is available. Please check server '
                                   'settings, such as cpus and memory, in ARC/arc/settings/settings.py.')
                if cluster_soft.lower() == 'slurm' and 'AssocMaxSubmitJobLimit' in line:
                    logger.warning(f'Max number of submitted jobs was reached, sleeping...')
                    time.sleep(5 * 60)
                    self.submit_job(remote_path=remote_path, recursion=True)
        if recursion:
            return None, None
        elif cluster_soft.lower() in ['oge', 'sge'] and stdout and 'submitted' in stdout[0].lower():
            job_id = stdout[0].split()[2]
        elif cluster_soft.lower() == 'slurm' and stdout and 'submitted' in stdout[0].lower():
            job_id = stdout[0].split()[3]
        elif cluster_soft.lower() == 'pbs' and stdout:
            job_id = stdout[0].split('.')[0]
        elif cluster_soft.lower() == 'htcondor' and stdout and 'submitting' in stdout[0].lower():
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

        Failures that retrying cannot resolve -- a rejected authentication, a host key that does
        not match ``known_hosts``, and an unknown host key on a server that sets
        ``strict_host_key_checking`` (:data:`PERMANENT_CONNECTION_ERRORS`) -- raise a
        ``ServerError`` on the first attempt, carrying the paramiko exception as its cause -- this
        holds whatever ``connection_attempts`` is set to, since no number of retries can resolve
        them. A configured ``key`` file that does not exist is permanent for the same reason, and
        is told apart from the transport-level ``OSError``s by :meth:`_is_a_missing_key_file`.
        Every other failure is transport-level, and is retried once a minute until
        ``connection_attempts`` attempts have been made (24 hours by default). No interval is
        waited out after the last attempt, so a client asked for a single attempt fails at once.

        A contradicted host key raises the :class:`HostKeyMismatchError` subclass of
        ``ServerError`` rather than a plain one, and is reported at the error level naming both
        fingerprints, so it is told apart from a wrong password in the log rather than reading as
        one more failed connection.

        Raises:
            HostKeyMismatchError: The server's host key contradicts the one in ``known_hosts``.
            ServerError: Cannot connect to the server with maximum times to try,
                         or the failure is permanent.
        """
        times_tried = 0
        interval = 60  # wait 60 sec between trials
        while times_tried < self.connection_attempts:
            times_tried += 1
            try:
                self._sftp, self._ssh = self._connect()
            except paramiko.BadHostKeyException as e:
                raise self._host_key_mismatch_error(e) from e
            except PERMANENT_CONNECTION_ERRORS as e:
                raise ServerError(f'Could not connect to server {self.server}, and retrying '
                                  f'cannot resolve it. Got {type(e).__name__}: {e}') from e
            except OSError as e:
                if self._is_a_missing_key_file(e):
                    raise ServerError(f'Could not connect to server {self.server}, and retrying '
                                      f'cannot resolve it: the SSH key file {self.key!r} does not '
                                      f'exist. Either correct the "key" entry of this server in the '
                                      f'settings, or remove it to authenticate via a running '
                                      f'ssh-agent or the default key paths.') from e
                self._report_a_failed_connection_attempt(e, times_tried)
            except Exception as e:
                self._report_a_failed_connection_attempt(e, times_tried)
            else:
                self._apply_keepalive()
                logger.debug(f'Successfully connected to {self.server} at the {times_tried} trial.')
                return
            if times_tried < self.connection_attempts:
                time.sleep(interval)
        raise ServerError(f'Could not connect to server {self.server} even after {times_tried} trials.')

    def _apply_keepalive(self) -> bool:
        """
        Re-apply the keepalive interval this client was asked for to its current transport.

        A keepalive is a property of a paramiko ``Transport``, not of the client that holds it, so
        it is lost whenever a new transport is opened. :func:`check_connections` reconnects a
        client in place when its socket has gone half-open, which replaces the transport under a
        pooled client that outlives many such reconnects, so the interval is re-applied here for
        every transport rather than only for the first.

        Returns: bool
            Whether a keepalive was applied, ``False`` when none was asked for or there is no
            live transport to apply it to.
        """
        if self._keepalive_interval is None or self._ssh is None:
            return False
        transport = self._ssh.get_transport()
        if transport is None:
            return False
        transport.set_keepalive(self._keepalive_interval)
        return True

    def _report_a_failed_connection_attempt(self, error: Exception, times_tried: int) -> None:
        """
        Report a connection attempt that failed and will be retried.

        Every tenth attempt goes to the log, and the ones in between are printed, so that a run
        that spends hours retrying leaves a bounded trail in the log file while still showing
        progress on the terminal.

        Args:
            error (Exception): The failure to report.
            times_tried (int): The number of attempts made so far, including this one.
        """
        message = f'Tried connecting to {self.server} {times_tried} times with no success...' \
                  f'\nGot: {error}'
        if not times_tried % 10:
            logger.info(message)
        else:
            print(message)

    def _is_a_missing_key_file(self, error: OSError) -> bool:
        """
        Whether ``error`` reports that this client's configured ``key`` file does not exist.

        paramiko reads the identity it was asked to authenticate with outside the ``SSHException``
        it guards that read with, so an absent key file surfaces as a ``FileNotFoundError``. That
        is an ``OSError``, as a refused or reset connection is, and the two must not be treated
        alike: a missing file is permanent, while a network failure is exactly what the retry loop
        exists for. They are told apart by the error being a ``FileNotFoundError`` that names the
        configured key path.

        Args:
            error (OSError): The error raised while connecting.

        Returns: bool
            Whether the error is the absence of the configured key file.
        """
        if self.key is None or not isinstance(error, FileNotFoundError):
            return False
        filename = getattr(error, 'filename', None)
        if filename is None:
            return True
        return os.path.expanduser(str(filename)) == os.path.expanduser(self.key)

    def _host_key_mismatch_error(self, error: paramiko.BadHostKeyException) -> HostKeyMismatchError:
        """
        Report a contradicted host key and build the error to raise for it.

        The report goes out at the error level and names the stored and the presented
        fingerprints, since which of the two the reader recognises is what decides whether the
        server was re-keyed or the session was intercepted.

        Args:
            error (paramiko.BadHostKeyException): The mismatch paramiko raised.

        Returns: HostKeyMismatchError
            The error to raise, carrying the same report as its message.
        """
        path = os.path.expanduser(KNOWN_HOSTS_PATH)
        message = f'The host key {self.address} presented does not match the key stored for it ' \
                  f'in {KNOWN_HOSTS_PATH}, so server {self.server} was not connected to.\n' \
                  f'    stored:    {error.expected_key.get_name()} ' \
                  f'{get_host_key_fingerprint(error.expected_key)}\n' \
                  f'    presented: {error.key.get_name()} {get_host_key_fingerprint(error.key)}\n' \
                  f'A re-keyed or rebuilt server and an intercepted session look exactly like ' \
                  f'this, and the fingerprints are what tells them apart. Verify the presented ' \
                  f'fingerprint against a trusted source. Only once it is confirmed to be the ' \
                  f'server\'s own key, replace the stored one with:\n' \
                  f'    ssh-keygen -R {self.address} -f {path}\n' \
                  f'    ssh-keyscan -H {self.address} >> {KNOWN_HOSTS_PATH}'
        logger.error(message)
        return HostKeyMismatchError(message)

    def _connect(self) -> tuple[paramiko.sftp_client.SFTPClient, paramiko.SSHClient]:
        """
        Connect via paramiko, and open an SSH session as well as a SFTP session.

        ``self.key`` is passed as paramiko's ``key_filename``, i.e. as the identity to
        authenticate with. It may be ``None``, in which case paramiko looks for a running
        ssh-agent and then for the default key paths; that is the only way an agent-forwarded
        session can be used, and it also avoids paramiko raising on a configured key path that
        does not exist on this machine.

        Note that ARC never parses ``~/.ssh/config``: paramiko only does so when an application
        builds a ``paramiko.SSHConfig`` itself, and ARC does not. Directives such as
        ``IdentityFile``, ``ProxyJump`` and ``ProxyCommand`` therefore have no effect here.

        Host key policy applies to an *unknown* key only. A key that is known and contradicted
        is refused by paramiko whatever the policy, and :meth:`connect` turns that into a
        :class:`HostKeyMismatchError` naming both fingerprints.

        An unknown host key means either a first-ever connection or a
        machine-in-the-middle, and the two are indistinguishable from here. The default policy
        is :class:`LogAndAcceptHostKeyPolicy`, which logs the unknown key through ARC's logger
        and connects; the key is never added to ``known_hosts``, so the warning repeats on every
        connection. Setting ``strict_host_key_checking: True`` on the server selects
        :class:`RejectUnknownHostKeyPolicy` instead, which raises :class:`UnknownHostKeyError`
        for any host that is not already in ``known_hosts``.

        Warning rather than rejecting is the default because ARC is a scheduler that runs
        unattended for days. Rejecting an unknown host does not fail once: every job submission,
        status poll and download for that server fails while the driver stays alive, so the run
        keeps going and produces nothing, and the cause surfaces only when someone reads the log.
        Recovering then means stopping ARC, running ``ssh-keyscan``, and restarting. To make the
        default policy's residual risk visible before that cost is paid,
        :func:`check_servers_known_hosts` reports every configured server that is absent from
        ``known_hosts`` at startup, before any calculation is submitted.

        The timeout is enlarged from paramiko's 15 second default because a server may accept
        the connection while its SSH daemon takes longer to answer, e.g. under network
        congestion.

        The retry covers transport-level failures only, such as "SSHException: Error reading SSH
        protocol banner[Error 104] Connection reset by peer". A bad key, a bad username and a
        refused host key (:data:`PERMANENT_CONNECTION_ERRORS`) are raised without a second
        attempt. A bare ``except`` here also swallowed KeyboardInterrupt/SystemExit, and
        discarded the first exception so that a bad key or username surfaced as the retry's error
        instead of its own.

        Returns: tuple[paramiko.sftp_client.SFTPClient, paramiko.SSHClient]
            - An SFTP client used to perform remote file operations.
            - A high-level representation of a session with an SSH server.
        """
        ssh = paramiko.SSHClient()
        ssh.load_system_host_keys()
        if servers[self.server].get('strict_host_key_checking', False):
            ssh.set_missing_host_key_policy(RejectUnknownHostKeyPolicy())
        else:
            ssh.set_missing_host_key_policy(LogAndAcceptHostKeyPolicy())
        try:
            ssh.connect(hostname=self.address, username=self.un, banner_timeout=200, key_filename=self.key)
        except PERMANENT_CONNECTION_ERRORS:
            raise
        except (paramiko.SSHException, OSError) as e:
            if isinstance(e, OSError) and self._is_a_missing_key_file(e):
                raise
            logger.debug(f'First SSH connection attempt to {self.server} failed with '
                         f'{type(e).__name__}: {e}. Retrying once.')
            ssh.connect(hostname=self.address, username=self.un, banner_timeout=200, key_filename=self.key)
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
                               remote_file_path_2: str | None = None,
                               ) -> datetime.datetime | None:
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
        command = 'ls -alF'
        return self._send_command_to_server(command, remote_path)[0]

    def find_package(self, package_name: str) -> list:
        """
        Find the path to the package.

        Args:
            package_name (str): The name of the package to search for.
        """
        command = f'. ~/.bashrc; which {shlex.quote(package_name)}'
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
        cmd = list_available_nodes_command[servers[self.server]['cluster_soft']]
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
        command = f'chmod{recursive} -- {mode} {shlex.quote(file_name)}'
        self._send_command_to_server(command, remote_path)

    def remove_dir(self, remote_path: str) -> None:
        """
        Remove a directory, and everything under it, on the server.

        This is the remote-cleanup primitive. ARC's own job flow does not call it: no job removes
        its remote work directory today, and the caller that will is added separately. It is
        reached through :meth:`arc.job.adapter.JobAdapter.remove_remote_files`, which supplies
        the job's remote path.

        Args:
            remote_path (str): The path to the directory to remove on the remote server.

        Raises:
            ServerError: If the directory could not be removed.
        """
        command = f'rm -rf -- {shlex.quote(remote_path)}'
        _, stderr = self._send_command_to_server(command)
        if stderr:
            raise ServerError(
                f'Cannot remove dir for the given path ({remote_path}).\nGot: {stderr}')

    def delete_remote_check_files(self, remote_path: str) -> None:
        """
        Delete ESS checkfiles under a remote directory (recursively).
        They usually take up lots of space and are not needed after ARC terminates.
        Pass ``True`` to the ``keep_checks`` flag in ARC to avoid deleting check files.
        The local counterpart of this method is ``arc.common.delete_check_files()``.

        Unlike :meth:`remove_dir`, this keeps the remote directory and everything in it that is
        not a checkfile, so a project's outputs remain on the server after the cleanup.
        A failure is logged rather than raised, see :func:`delete_check_files_on_servers`.

        Args:
            remote_path (str): The remote directory path under which checkfiles will be deleted.
        """
        if not remote_path or not self._check_dir_exists(remote_path):
            return
        command = f'find {shlex.quote(remote_path)} -type f -name "*.chk" -delete'
        _, stderr = self._send_command_to_server(command)
        if stderr:
            logger.warning(f'Could not delete all check files under {remote_path} on {self.server}.\nGot: {stderr}')

    def _check_file_exists(self,
                           remote_file_path: str,
                           ) -> bool:
        """
        Check if a file exists on the remote server.

        Args:
            remote_file_path (str): The path to the file on the remote server.

        Returs:
            bool: Whether the file exists on the remote server. ``True`` if it exists.
        """
        command = f'[ -f {shlex.quote(remote_file_path)} ] && echo "File exists"'
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
            bool: Whether the directory exists on the remote server. ``True`` if it exists.
        """
        command = f'[ -d {shlex.quote(remote_dir_path)} ] && echo "Dir exists"'
        stdout, _ = self._send_command_to_server(command)
        if len(stdout):
            return True

    def _create_dir(self, remote_path: str) -> None:
        """
        Create a new directory on the server.

        Args:
            remote_path (str): The path to the directory to create on the remote server.
        """
        command = f'mkdir -p -- {shlex.quote(remote_path)}'
        _, stderr = self._send_command_to_server(command)
        if stderr:
            raise ServerError(
                f'Cannot create dir for the given path ({remote_path}).\nGot: {stderr}')


def _addresses_worth_checking(server_dict: dict) -> dict[str, str]:
    """
    Return the address of every server whose host key is worth looking up.

    Servers named ``local``, entries that are not dictionaries, servers without an ``address``,
    and servers still carrying ARC's shipped placeholder address or username are left out. The
    placeholders cannot be reached at all, and reporting them would fire on every run made with
    the repository's default settings.

    Args:
        server_dict (dict): The servers to filter.

    Returns: dict[str, str]
        The address of each server to check, keyed by server name.
    """
    addresses = dict()
    for server_name, server_settings in server_dict.items():
        if server_name == 'local' or not isinstance(server_settings, dict):
            continue
        address = server_settings.get('address')
        if not address or address.endswith(PLACEHOLDER_ADDRESS_SUFFIX) \
                or server_settings.get('un') == PLACEHOLDER_USERNAME:
            continue
        addresses[server_name] = address
    return addresses


def _load_known_hosts(path: str) -> paramiko.HostKeys:
    """
    Read a ``known_hosts`` file, returning an empty set of host keys if it cannot be read.

    Args:
        path (str): The expanded path of the ``known_hosts`` file to read.

    Returns: paramiko.HostKeys
        The host keys the file holds, empty when the file is absent or unreadable.
    """
    try:
        return paramiko.HostKeys(filename=path)
    except (OSError, paramiko.SSHException) as e:
        logger.debug(f'Could not read the host keys in {path}: {type(e).__name__}: {e}')
        return paramiko.HostKeys()


def get_servers_missing_host_keys(server_dict: dict | None = None,
                                  known_hosts_path: str | None = None,
                                  ) -> dict[str, str]:
    """
    Determine which of the configured servers have no host key on this machine.

    The lookup is local and offline: the ``known_hosts`` file is read, nothing is resolved and
    no connection is opened. paramiko's ``HostKeys`` performs the lookup, so hashed entries
    (``ssh-keyscan -H``) and ``[host]:port`` entries are matched as OpenSSH matches them.

    This reports an absent key only. Whether a stored key still matches the one a server
    presents is not knowable from this machine, since only the server can present it; that
    comparison is made by paramiko while connecting, and raises
    :class:`HostKeyMismatchError`. What can be checked offline alongside an absent key is a
    ``known_hosts`` file that contradicts itself, which is
    :func:`get_servers_with_conflicting_host_keys`.

    Args:
        server_dict (dict, optional): The servers to check. Defaults to the configured servers.
        known_hosts_path (str, optional): The ``known_hosts`` file to read.
                                          Defaults to :data:`KNOWN_HOSTS_PATH`.

    Returns: dict[str, str]
        The address of each server that has no host key, keyed by server name.
    """
    server_dict = servers if server_dict is None else server_dict
    path = os.path.expanduser(known_hosts_path if known_hosts_path is not None else KNOWN_HOSTS_PATH)
    host_keys = _load_known_hosts(path)
    missing = dict()
    for server_name, address in _addresses_worth_checking(server_dict).items():
        if host_keys.lookup(address) is None:
            missing[server_name] = address
    return missing


def get_servers_with_conflicting_host_keys(server_dict: dict | None = None,
                                           known_hosts_path: str | None = None,
                                           ) -> dict[str, list[str]]:
    """
    Determine which of the configured servers have contradictory host keys on this machine.

    A server legitimately has one host key per key type, and ``ssh-keyscan`` writes one line per
    type. More than one entry of the *same* type for one address means the file disagrees with
    itself about what that server's key is, which is what a stale entry left behind by a rebuilt
    server looks like, and equally what an entry prepended to shadow the real key looks like.
    Only the first matching entry is ever consulted -- by OpenSSH, and by the ``HostKeys.lookup``
    paramiko authenticates with -- so a shadowed key is trusted silently while the server's real
    key is reported as a mismatch.

    The check is local and offline: the ``known_hosts`` file is read, nothing is resolved and no
    connection is opened. It therefore cannot say *which* of the recorded keys is the server's;
    answering that requires the key the server presents, which is compared while connecting and
    raises :class:`HostKeyMismatchError`.

    Args:
        server_dict (dict, optional): The servers to check. Defaults to the configured servers.
        known_hosts_path (str, optional): The ``known_hosts`` file to read.
                                          Defaults to :data:`KNOWN_HOSTS_PATH`.

    Returns: dict[str, list[str]]
        The key types recorded more than once, sorted, keyed by server name. Servers whose
        entries do not contradict each other are absent.
    """
    server_dict = servers if server_dict is None else server_dict
    path = os.path.expanduser(known_hosts_path if known_hosts_path is not None else KNOWN_HOSTS_PATH)
    host_keys = _load_known_hosts(path)
    conflicting = dict()
    for server_name, address in _addresses_worth_checking(server_dict).items():
        entries = host_keys.lookup(address)
        if entries is None:
            continue
        repeated = sorted(key_type for key_type, count in Counter(entries.keys()).items() if count > 1)
        if repeated:
            conflicting[server_name] = repeated
    return conflicting


def check_servers_known_hosts(server_dict: dict | None = None,
                              known_hosts_path: str | None = None,
                              ) -> dict[str, str]:
    """
    Report configured servers whose host keys need attention before any job is submitted.

    Two offline conditions are reported, at two levels. A server with no host key at all is a
    warning: ARC connects to an unknown host anyway (see :meth:`SSHClient._connect`), so without
    this the first sign of an unseeded ``known_hosts`` is a per-connection warning buried in a
    running job's log, or -- for a server with ``strict_host_key_checking`` -- a run that appears
    to hang while every connection is refused. A server with contradictory entries
    (:func:`get_servers_with_conflicting_host_keys`) is an error: the file records two different
    keys as that server's, only one of them is consulted, and which one is trusted is decided by
    line order rather than by anything the reader chose.

    A stored key that no longer matches the key the server presents is not reported here and
    cannot be, since the comparison needs the server. paramiko makes it while connecting, and it
    surfaces as :class:`HostKeyMismatchError`.

    Args:
        server_dict (dict, optional): The servers to check. Defaults to the configured servers.
        known_hosts_path (str, optional): The ``known_hosts`` file to read.
                                          Defaults to :data:`KNOWN_HOSTS_PATH`.

    Returns: dict[str, str]
        The address of each server that has no host key, keyed by server name.
    """
    server_dict = servers if server_dict is None else server_dict
    path = os.path.expanduser(known_hosts_path if known_hosts_path is not None else KNOWN_HOSTS_PATH)
    missing = get_servers_missing_host_keys(server_dict=server_dict, known_hosts_path=path)
    for server_name, address in missing.items():
        if server_dict[server_name].get('strict_host_key_checking', False):
            consequence = f'server "{server_name}" sets strict_host_key_checking, so every ' \
                          f'connection to it will be refused'
        else:
            consequence = f'ARC will connect to server "{server_name}" anyway and warn on every ' \
                          f'connection, and cannot tell a first-ever connection from an interception'
        logger.warning(f'The host key of {address} is not in {path}; {consequence}. '
                       f'Verify the fingerprint against a trusted source and add it with:\n'
                       f'    ssh-keyscan -H {address} >> {path}')
    conflicting = get_servers_with_conflicting_host_keys(server_dict=server_dict, known_hosts_path=path)
    for server_name, key_types in conflicting.items():
        address = server_dict[server_name]['address']
        logger.error(f'{path} records more than one {", ".join(key_types)} host key for '
                     f'{address}, the address of server "{server_name}", so it disagrees with '
                     f'itself about that server\'s identity. Only the first entry is used, which '
                     f'means a stale key left by a rebuilt server, or a key placed there to '
                     f'impersonate it, would be trusted in place of the real one. Verify the '
                     f'server\'s fingerprint against a trusted source, then leave only that key:\n'
                     f'    ssh-keygen -R {address} -f {path}\n'
                     f'    ssh-keyscan -H {address} >> {path}')
    return missing


def delete_check_files_on_servers(remote_project_paths: dict) -> None:
    """
    Delete ESS checkfiles from an ARC project's directory on all servers it ran jobs on.
    The local counterpart of this function is ``arc.common.delete_check_files()``.
    Errors are only logged and never raised: this runs once ARC is done with the science,
    an unreachable server at that point is an inconvenience, not a reason to lose a run.

    Each server is reached through its own single-attempt client rather than through the
    connection pool (:mod:`arc.job.ssh_pool`), for the same reason: both the pool's factory and
    the fallback in :func:`~arc.job.ssh_pool.borrow_ssh_client` build a client with the default
    24-hour retry, so borrowing here would let a server that has gone away hold up the end of a
    run indefinitely. A cleanup that cannot reach a server must give up, not wait.

    Args:
        remote_project_paths (dict): Keys are server names, values are the respective remote paths
                                     of the project's directory on that server.
    """
    for server, remote_project_path in remote_project_paths.items():
        if not server or server.lower() == 'local' or not remote_project_path:
            continue
        try:
            with SSHClient(server, connection_attempts=1) as ssh:
                ssh.delete_remote_check_files(remote_path=remote_project_path)
        except Exception as e:
            logger.warning(f'Could not delete the check files under {remote_project_path} on {server}.\nGot: {e}')


def check_job_status_in_stdout(job_id: int,
                               stdout: list | str,
                               server: str,
                               ) -> str:
    """
    A helper function for checking job status.

    Args:
        job_id (int): the job ID recognized by the server.
        stdout (list | str): The output of a queue status check.
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
        elif status.lower() in ['e']:
            return 'errored'
    elif servers[server]['cluster_soft'].lower() == 'htcondor':
        return 'running'
    raise ValueError(f'Unknown cluster software {servers[server]["cluster_soft"]}')


def delete_all_arc_jobs(server_list: list,
                        jobs: list[str] | None = None,
                        ) -> None:
    """
    Delete all ARC-spawned jobs (with job name starting with `a` and a digit) from :list:servers
    (`servers` could also be a string of one server name)
    Make sure you know what you're doing, so unrelated jobs won't be deleted...
    Useful when terminating ARC while some (ghost) jobs are still running.

    Args:
        server_list (list): List of servers to delete ARC jobs from.
        jobs (list[str] | None): Specific ARC job IDs to delete.
    """
    if isinstance(server_list, str):
        server_list = [server_list]
    for server in server_list:
        with SSHClient(server) as ssh:
            ssh.delete_jobs(jobs)
    if server_list:
        print('\ndone.')
