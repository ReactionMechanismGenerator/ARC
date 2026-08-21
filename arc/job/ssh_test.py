#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.ssh module
"""

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import arc.job.ssh as ssh
from arc.exceptions import ServerError
from arc.job.ssh import SSHClient, delete_check_files_on_servers


class TestSSH(unittest.TestCase):
    """
    Contains unit tests for the SSH module
    """

    def test_check_job_status_in_stdout(self):
        """Test checking the job status in stdout"""
        # OGE
        stdout_1 = """job-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID 
-----------------------------------------------------------------------------------------------------------------
 582682 0.45451 a9654      alongd       e     04/17/2019 16:22:14 long5@node93.cluster              48
 588334 0.45451 pf1005a    alongd       r     05/07/2019 16:24:31 long3@node67.cluster              48
 588345 0.45451 a14121     alongd       r     05/08/2019 02:11:42 long3@node69.cluster              48    """
        status1 = ssh.check_job_status_in_stdout(job_id=588345, stdout=stdout_1, server='server1')
        self.assertEqual(status1, 'running')
        status2 = ssh.check_job_status_in_stdout(job_id=582682, stdout=stdout_1, server='server1')
        self.assertEqual(status2, 'errored')
        status3 = ssh.check_job_status_in_stdout(job_id=582600, stdout=stdout_1, server='server1')
        self.assertEqual(status3, 'done')

        # HTCondor
        stdout_2 = ['5231.0 R 10 7885 a20596 130',
                    '5232.0 R 10 7885 a20597 130',
                    '5233.0 R 10 7885 a20598 130',
                    '5241.0 P 10 7885 a20616 0']
        status1 = ssh.check_job_status_in_stdout(job_id=5231, stdout=stdout_2, server='local')
        self.assertEqual(status1, 'running')
        status1 = ssh.check_job_status_in_stdout(job_id=5241, stdout=stdout_2, server='local')
        self.assertEqual(status1, 'running')
        status1 = ssh.check_job_status_in_stdout(job_id=4000, stdout=stdout_2, server='local')
        self.assertEqual(status1, 'done')


class TestSSHClientConnect(unittest.TestCase):
    """
    Contains unit tests for connecting to a server.
    """

    def setUp(self):
        """
        A method that is run before each unit test in this class.
        Count the connection trials instead of attempting to reach a server.
        """
        self.trials, self.intervals = list(), list()
        for patch in [mock.patch.object(SSHClient, '_connect', lambda ssh_client: self.fail_to_connect()),
                      mock.patch.object(ssh.time, 'sleep', lambda interval: self.intervals.append(interval))]:
            patch.start()
            self.addCleanup(patch.stop)

    def fail_to_connect(self):
        """
        Record a connection trial and fail it, as an unreachable server would.

        Raises:
            ServerError: Always.
        """
        self.trials.append(len(self.trials) + 1)
        raise ServerError('Could not connect.')

    def test_connect_gives_up_after_a_single_requested_trial(self):
        """Test that a single connection trial is not followed by an interval, so teardown cannot block"""
        ssh_client = SSHClient('server2', connection_attempts=1)
        with self.assertRaises(ServerError):
            ssh_client.connect()
        self.assertEqual(len(self.trials), 1)
        self.assertEqual(self.intervals, list())

    def test_connect_does_not_wait_after_its_last_trial(self):
        """Test that connecting waits between trials, but not after the last one"""
        ssh_client = SSHClient('server2', connection_attempts=3)
        with self.assertRaises(ServerError):
            ssh_client.connect()
        self.assertEqual(len(self.trials), 3)
        self.assertEqual(len(self.intervals), 2)

    def test_connect_defaults_to_the_long_haul(self):
        """Test that the default number of connection trials, used while jobs are running, is unchanged"""
        self.assertEqual(SSHClient('server2').connection_attempts, 1440)


class TestDeleteCheckFilesOnServers(unittest.TestCase):
    """
    Contains unit tests for deleting ESS checkfiles on the servers a project ran on.
    """

    def setUp(self):
        """
        A method that is run before each unit test in this class.
        Set up a fake remote server: a temporary directory in which the commands ARC would have sent
        to a server are actually executed, so that the real cleanup code path is exercised.
        """
        self.remote_root = tempfile.mkdtemp()
        self.commands = list()
        self.server = 'server2'
        self.project_path = os.path.join('runs', 'ARC_Projects', 'a_project')
        self.other_project_path = os.path.join('runs', 'ARC_Projects', 'an_unrelated_project')
        for patch in [mock.patch.object(SSHClient, 'connect', lambda ssh_client: None),
                      mock.patch.object(SSHClient, '_send_command_to_server',
                                        lambda ssh_client, command, remote_path='':
                                        self.execute_on_fake_server(command, remote_path))]:
            patch.start()
            self.addCleanup(patch.stop)
        self.check_file = self.write_remote_file(self.project_path, 'spc1', 'opt_a1', 'check.chk')
        self.output_file = self.write_remote_file(self.project_path, 'spc1', 'opt_a1', 'input.log')
        self.other_check_file = self.write_remote_file(self.other_project_path, 'spc2', 'opt_a1', 'check.chk')

    def execute_on_fake_server(self, command: str, remote_path: str = '') -> tuple:
        """
        Execute a command in the fake remote server directory instead of sending it to a server.

        Args:
            command (str): The command to execute.
            remote_path (str, optional): The directory path at which the command will be executed.

        Returns: tuple[list, list]
            The lines of the standard output and of the standard error streams.
        """
        self.commands.append(command)
        result = subprocess.run(command, shell=True, capture_output=True, text=True,
                                cwd=os.path.join(self.remote_root, remote_path))
        return result.stdout.splitlines(True), result.stderr.splitlines(True)

    def write_remote_file(self, *args) -> str:
        """
        Write a file in the fake remote server directory.

        Args:
            args: The path of the file relative to the fake remote server directory.

        Returns: str
            The path of the written file.
        """
        path = os.path.join(self.remote_root, *args)
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w') as f:
            f.write('dummy file content')
        return path

    def test_only_check_files_of_the_given_project_are_deleted(self):
        """Test that check files are deleted, and that nothing else on the server is"""
        delete_check_files_on_servers({self.server: self.project_path})
        self.assertFalse(os.path.isfile(self.check_file))
        self.assertTrue(os.path.isfile(self.output_file))
        self.assertTrue(os.path.isfile(self.other_check_file))

    def test_a_local_server_is_skipped(self):
        """Test that a server named 'local' is skipped, its check files are deleted by the local cleanup"""
        delete_check_files_on_servers({'local': self.project_path})
        self.assertTrue(os.path.isfile(self.check_file))

    def test_a_missing_remote_directory_is_a_no_op(self):
        """Test that a project directory which does not exist on the server is silently skipped"""
        delete_check_files_on_servers({self.server: os.path.join('runs', 'ARC_Projects', 'no_such_project')})
        self.assertTrue(os.path.isfile(self.check_file))
        self.assertTrue(os.path.isfile(self.other_check_file))

    def test_a_path_with_shell_characters_is_not_interpreted(self):
        """Test that the remote path is quoted, so that its content cannot become part of the command"""
        wild_path = os.path.join('runs', 'ARC_Projects', 'a project $(touch injected.txt)')
        wild_check_file = self.write_remote_file(wild_path, 'spc1', 'opt_a1', 'check.chk')
        delete_check_files_on_servers({self.server: wild_path})
        self.assertEqual(len(self.commands), 1)
        self.assertEqual([token for token in shlex.split(self.commands[0]) if token == wild_path], [wild_path] * 2)
        self.assertFalse(os.path.isfile(wild_check_file))
        self.assertFalse(os.path.isfile(os.path.join(self.remote_root, 'injected.txt')))

    def test_an_unreachable_server_is_only_logged(self):
        """Test that a server which cannot be reached does not raise, ARC is done with the science by now"""
        def raise_server_error(ssh_client):
            raise ServerError(f'Could not connect to server {ssh_client.server}.')

        with mock.patch.object(SSHClient, 'connect', raise_server_error):
            delete_check_files_on_servers({self.server: self.project_path})  # Must not raise.
        self.assertTrue(os.path.isfile(self.check_file))

    def tearDown(self):
        """
        A method that is run after each unit test in this class.
        """
        shutil.rmtree(self.remote_root, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
