#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.ssh module
"""

import unittest
from unittest.mock import patch

import arc.job.ssh as ssh


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

    @staticmethod
    def _make_client(server: str) -> ssh.SSHClient:
        """
        Create an SSHClient instance bound to a server name without connecting.

        Args:
            server (str): The server name to bind the client to.

        Returns:
            SSHClient: A client instance that has not opened any connection.
        """
        client = ssh.SSHClient.__new__(ssh.SSHClient)
        client.server = server
        client._ssh, client._sftp = None, None
        return client

    @staticmethod
    def _fail_send(command: str) -> tuple:
        """
        A stand-in for ``_send_command_to_server`` that must not be reached.

        Args:
            command (str): The command that would have been sent.

        Returns:
            tuple: Never returns, always raises.
        """
        raise AssertionError(f'_send_command_to_server should not have been called (command: {command})')

    def test_list_available_nodes_pbs_and_htcondor_return_empty(self):
        """Test that PBS and HTCondor early-return an empty node list without sending a command"""
        for cluster_soft in ['PBS', 'HTCondor', 'pbs', 'htcondor']:
            with patch.dict(ssh.servers, {'node_test_server': {'cluster_soft': cluster_soft}}):
                client = self._make_client('node_test_server')
                client._send_command_to_server = self._fail_send
                self.assertEqual(client.list_available_nodes(), list())

    def test_list_available_nodes_slurm_command_resolution(self):
        """Test command-key resolution and stdout parsing for a canonical-case Slurm config"""
        sent_commands = list()
        stdout = ['node01 alloc 1.00 none\n',
                  'node02 idle 0.00 none\n',
                  'node03 down 0.00 none\n']

        def send(command: str) -> tuple:
            sent_commands.append(command)
            return stdout, list()

        with patch.dict(ssh.servers, {'node_test_server': {'cluster_soft': 'Slurm'}}):
            client = self._make_client('node_test_server')
            client._send_command_to_server = send
            nodes = client.list_available_nodes()
        self.assertEqual(sent_commands, [ssh.list_available_nodes_command['Slurm']])
        self.assertEqual(nodes, ['node01', 'node02'])

    def test_list_available_nodes_lowercase_cluster_soft(self):
        """Test that a lowercase user-configured cluster_soft ('slurm') resolves the command key"""
        sent_commands = list()

        def send(command: str) -> tuple:
            sent_commands.append(command)
            return ['node05 mix 1.00 none\n'], list()

        with patch.dict(ssh.servers, {'node_test_server': {'cluster_soft': 'slurm'}}):
            client = self._make_client('node_test_server')
            client._send_command_to_server = send
            nodes = client.list_available_nodes()
        self.assertEqual(sent_commands, [ssh.list_available_nodes_command['Slurm']])
        self.assertEqual(nodes, ['node05'])


class TestCheckRunningJobsIdsAndStates(unittest.TestCase):
    """
    Contains unit tests for queue-state-aware job listing.
    """

    @staticmethod
    def _make_client(server: str, stdout: list, sent_commands: list) -> ssh.SSHClient:
        """
        Create an SSHClient bound to a server with a canned _send_command_to_server.

        Args:
            server (str): The server name to bind the client to.
            stdout (list): The canned stdout lines the fake command returns.
            sent_commands (list): A list collecting the commands sent.

        Returns:
            SSHClient: A client instance that has not opened any connection.
        """
        client = ssh.SSHClient.__new__(ssh.SSHClient)
        client.server = server
        client._ssh, client._sftp = None, None

        def send(command: str, remote_path: str = '') -> tuple:
            sent_commands.append(command)
            return stdout, list()

        client._send_command_to_server = send
        return client

    def test_normalize_queue_state_trivial(self):
        """Test normalizing a plain Slurm running state"""
        self.assertEqual(ssh.normalize_queue_state('R', 'Slurm'), 'running')

    def test_normalize_queue_state_per_cluster_soft(self):
        """Test state normalization across cluster softwares"""
        self.assertEqual(ssh.normalize_queue_state('PD', 'Slurm'), 'pending')
        self.assertEqual(ssh.normalize_queue_state('CG', 'Slurm'), 'exiting')
        self.assertEqual(ssh.normalize_queue_state('XX', 'Slurm'), 'unknown')
        self.assertEqual(ssh.normalize_queue_state('r', 'OGE'), 'running')
        self.assertEqual(ssh.normalize_queue_state('qw', 'OGE'), 'pending')
        self.assertEqual(ssh.normalize_queue_state('hqw', 'OGE'), 'held')
        self.assertEqual(ssh.normalize_queue_state('Eqw', 'OGE'), 'pending')
        self.assertEqual(ssh.normalize_queue_state('R', 'PBS'), 'running')
        self.assertEqual(ssh.normalize_queue_state('Q', 'PBS'), 'pending')
        self.assertEqual(ssh.normalize_queue_state('H', 'PBS'), 'held')
        self.assertEqual(ssh.normalize_queue_state('E', 'PBS'), 'exiting')
        self.assertEqual(ssh.normalize_queue_state('P', 'HTCondor'), 'pending')
        self.assertEqual(ssh.normalize_queue_state('R', 'HTCondor'), 'running')
        self.assertEqual(ssh.normalize_queue_state('H', 'HTCondor'), 'held')

    def test_check_running_jobs_ids_and_states_slurm(self):
        """Test parsing job IDs and states from squeue output"""
        stdout = ['JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)\n',
                  '14428 debug xq1371m2 user R 50-04:04:46 1 node06\n',
                  '14429 debug xq1371m3 user PD 0:00 1 (Priority)\n',
                  '14430 debug xq1371m4 user CG 0:01 1 node07\n']
        sent = list()
        with patch.dict(ssh.servers, {'state_test_server': {'cluster_soft': 'Slurm'}}):
            client = self._make_client('state_test_server', stdout, sent)
            states = client.check_running_jobs_ids_and_states()
        self.assertEqual(states, {'14428': 'running', '14429': 'pending', '14430': 'exiting'})
        self.assertEqual(sent, [ssh.check_status_command['Slurm']])

    def test_check_running_jobs_ids_and_states_oge(self):
        """Test parsing job IDs and states from OGE qstat output"""
        stdout = ['job-ID prior name user state submit/start at queue slots\n',
                  '----------------------------------------------------------\n',
                  '582682 0.45451 a9654 alongd r 04/17/2019 16:22:14 long5@node93 48\n',
                  '588334 0.45451 pf1005a alongd qw 05/07/2019 16:24:31 48\n',
                  '588345 0.45451 a14121 alongd hqw 05/08/2019 02:11:42 48\n']
        sent = list()
        with patch.dict(ssh.servers, {'state_test_server': {'cluster_soft': 'OGE'}}):
            client = self._make_client('state_test_server', stdout, sent)
            states = client.check_running_jobs_ids_and_states()
        self.assertEqual(states, {'582682': 'running', '588334': 'pending', '588345': 'held'})

    def test_check_running_jobs_ids_and_states_pbs(self):
        """Test parsing job IDs and states from PBS qstat output"""
        stdout = ['server:\n',
                  '\n',
                  'Job ID Username Queue Jobname SessID NDS TSK Memory Time S Time\n',
                  '------ -------- ----- ------- ------ --- --- ------ ---- - ----\n',
                  'filler\n',
                  '123456.zeus user workq spc1 1234 1 8 -- 24:00 R 01:23\n',
                  '123457.zeus user workq spc2 -- 1 8 -- 24:00 Q --\n',
                  '123458.zeus user workq spc3 -- 1 8 -- 24:00 H --\n',
                  '123459.zeus user workq spc4 5678 1 8 -- 24:00 E 23:59\n']
        sent = list()
        with patch.dict(ssh.servers, {'state_test_server': {'cluster_soft': 'PBS'}}):
            client = self._make_client('state_test_server', stdout, sent)
            states = client.check_running_jobs_ids_and_states()
        self.assertEqual(states, {'123456': 'running', '123457': 'pending',
                                  '123458': 'held', '123459': 'exiting'})

    def test_check_running_jobs_ids_and_states_htcondor(self):
        """Test parsing job IDs and states from condor_q output"""
        stdout = ['5231.0 R 10 7885 a20596 130\n',
                  '5241.0 P 10 7885 a20616 0\n',
                  '5250.0 H 10 7885 a20620 0\n']
        sent = list()
        with patch.dict(ssh.servers, {'state_test_server': {'cluster_soft': 'HTCondor'}}):
            client = self._make_client('state_test_server', stdout, sent)
            states = client.check_running_jobs_ids_and_states()
        self.assertEqual(states, {'5231': 'running', '5241': 'pending', '5250': 'held'})

    def test_check_running_jobs_ids_delegates(self):
        """Test that check_running_jobs_ids returns the same IDs in the same order"""
        stdout = ['JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)\n',
                  '14428 debug xq1371m2 user R 50-04:04:46 1 node06\n',
                  '14429 debug xq1371m3 user PD 0:00 1 (Priority)\n']
        sent = list()
        with patch.dict(ssh.servers, {'state_test_server': {'cluster_soft': 'Slurm'}}):
            client = self._make_client('state_test_server', stdout, sent)
            self.assertEqual(client.check_running_jobs_ids(), ['14428', '14429'])

    def test_check_running_jobs_ids_and_states_unsupported_cluster_soft(self):
        """Test that an unsupported cluster software raises a ValueError"""
        with patch.dict(ssh.servers, {'state_test_server': {'cluster_soft': 'LSF'},
                                      'local': {'cluster_soft': 'LSF'}}):
            client = self._make_client('state_test_server', list(), list())
            with self.assertRaises(ValueError):
                client.check_running_jobs_ids_and_states()
            with self.assertRaises(ValueError):
                client.check_running_jobs_ids()


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
