#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.local module
"""

import datetime
import os
import shutil
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import arc.job.local as local
from arc.common import ARC_PATH
from arc.exceptions import SettingsError


def process_is_alive(pid: int) -> bool:
    """
    Whether a process with the given pid exists and is not a zombie.

    Args:
        pid (int): The process ID to check.

    Returns:
        bool: Whether the process is alive.
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f'/proc/{pid}/stat', 'r') as f:
            return f.read().rsplit(')', 1)[-1].split()[0] != 'Z'
    except (OSError, IndexError):
        return True


def get_parent_pid(pid: int) -> int | None:
    """
    Get the parent pid of a process, used to report whether a process was reparented to init.

    Args:
        pid (int): The process ID to check.

    Returns:
        int | None: The parent process ID, or ``None`` if it could not be determined.
    """
    try:
        with open(f'/proc/{pid}/stat', 'r') as f:
            return int(f.read().rsplit(')', 1)[-1].split()[1])
    except (OSError, IndexError, ValueError):
        return None


def _read_pid(path: str) -> int | None:
    """
    Read a pid from a file, returning ``None`` if it is absent or unreadable.

    Args:
        path (str): The path of the file holding the pid.

    Returns:
        int | None: The pid, or ``None``.
    """
    try:
        with open(path, 'r') as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


class TestLocal(unittest.TestCase):
    """
    Contains unit tests for the local module
    """

    def test_execute_command(self):
        """Test executing a local command"""
        command1 = 'ls'
        out1 = local.execute_command(command1)
        self.assertIsInstance(out1, tuple)
        self.assertIsInstance(out1[0], list)
        self.assertIsInstance(out1[0][0], str)
        self.assertEqual(out1[1], [])
        if 'arc' in out1[0]:
            # Running from root
            self.assertIn('ARC.py', out1[0])
            self.assertIn('environment.yml', out1[0])
        else:
            # Running directly
            self.assertIn('adapter.py', out1[0])
            self.assertIn('ssh.py', out1[0])

    def test_execute_command_without_a_timeout_is_unchanged(self):
        """Test that not passing a timeout leaves the subprocess call exactly as it was"""
        with patch('arc.job.local.subprocess.run') as mock_run:
            mock_run.return_value.stdout, mock_run.return_value.stderr = b'ok\n', b''
            local.execute_command('ls')
            local.execute_command('ls', executable='/bin/bash')
        self.assertEqual(len(mock_run.call_args_list), 2)
        for call in mock_run.call_args_list:
            self.assertNotIn('timeout', call.kwargs)
            self.assertNotIn('start_new_session', call.kwargs)
        self.assertEqual(mock_run.call_args_list[0].kwargs, {'shell': True, 'capture_output': True})
        self.assertEqual(mock_run.call_args_list[1].kwargs,
                         {'shell': True, 'capture_output': True, 'executable': '/bin/bash'})

    def test_execute_command_with_a_generous_timeout(self):
        """Test that a command which completes in time is unaffected by a timeout"""
        stdout, stderr = local.execute_command('echo hello', timeout=60)
        self.assertEqual(stdout, ['hello'])
        self.assertEqual(stderr, [])

    def test_execute_command_timeout_kills_spawned_children(self):
        """Test that a timing out command is killed along with the processes it spawned"""
        # The grandchild ignores SIGTERM, so this pins both halves of the kill:
        # signalling the whole process group rather than only the direct child, and escalating
        # to SIGKILL. A grandchild that dies on SIGTERM cannot tell the escalation apart.
        # The direct child here is a plain ``sleep`` that does die on SIGTERM, which is exactly
        # why the escalation must not be conditioned on the direct child having survived.
        temp_dir = tempfile.mkdtemp()
        pid_path = os.path.join(temp_dir, 'grandchild.pid')
        script_path = os.path.join(temp_dir, 'stubborn_grandchild.py')
        with open(script_path, 'w') as f:
            f.write('import os, signal, sys, time\n'
                    'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
                    "with open(sys.argv[1], 'w') as f:\n"
                    "    f.write(str(os.getpid()))\n"
                    'time.sleep(300)\n')
        # The shell waits for the grandchild to record its pid before starting the long sleep, so a
        # loaded worker cannot have the timeout fire while the grandchild is still starting up and
        # fail the assertion below even though process group cleanup worked correctly.
        command = f'{sys.executable} {script_path} {pid_path} & ' \
                  f'while [ ! -s {pid_path} ]; do sleep 0.05; done; sleep 300'
        try:
            with self.assertRaises(SettingsError):
                local.execute_command(command, timeout=3)
            self.assertTrue(os.path.isfile(pid_path), 'The grandchild never recorded its pid.')
            with open(pid_path, 'r') as f:
                grandchild_pid = int(f.read().strip())
            for _ in range(200):
                if not process_is_alive(grandchild_pid):
                    break
                time.sleep(0.1)
            self.assertFalse(process_is_alive(grandchild_pid),
                             f'The spawned process {grandchild_pid} was orphaned instead of killed '
                             f'(parent pid {get_parent_pid(grandchild_pid)}).')
        finally:
            for pid in [_read_pid(pid_path)]:
                if pid is not None and process_is_alive(pid):
                    os.kill(pid, signal.SIGKILL)
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_execute_command_timeout_with_no_fail(self):
        """Test that a timing out command returns None, None when no_fail is True"""
        stdout, stderr = local.execute_command('sleep 300', no_fail=True, timeout=2)
        self.assertIsNone(stdout)
        self.assertIsNone(stderr)

    def test_determine_job_id(self):
        """Test determining a job ID from the stdout of a job submission command."""
        # Slurm
        stdout_1 = ['Submitted batch job 17670585']
        job_id = local._determine_job_id(stdout_1, cluster_soft='slurm')
        self.assertEqual(job_id, '17670585')

        # HTCondor
        stdout_2 = ['Submitting job(s).', '1 job(s) submitted to cluster 5263.']
        job_id = local._determine_job_id(stdout_2)
        self.assertEqual(job_id, '5263')

        # Wrong server name
        with self.assertRaises(ValueError):
            local._determine_job_id(stdout_2, cluster_soft='wrong')

    def test_get_last_modified_time(self):
        """Test the get_last_modified_time() function"""
        path = os.path.join(ARC_PATH, 'ARC.py')
        t = local.get_last_modified_time(path)
        self.assertIsInstance(t, datetime.datetime)
        t = local.get_last_modified_time('no file', path)
        self.assertIsInstance(t, datetime.datetime)

    def test_rename_output(self):
        """Test the rename_output() function"""
        path1 = os.path.join(ARC_PATH, 'scratch', 'input.log')
        path2 = os.path.join(ARC_PATH, 'scratch', 'output.out')
        if not os.path.exists(os.path.join(ARC_PATH, 'scratch')):
            os.makedirs(os.path.join(ARC_PATH, 'scratch'))
        with open(path1, 'w'):
            pass
        local.rename_output(local_file_path=path2, software='gaussian')
        self.assertFalse(os.path.isfile(path1))
        self.assertTrue(os.path.isfile(path2))
        shutil.rmtree(os.path.join(ARC_PATH, 'scratch'), ignore_errors=True)

    def test_parse_running_jobs_ids(self):
        """Test the parse_running_jobs_ids() function"""
        # Slurm:
        stdout = ['             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)',
                  '          10990729    normal     a207   alongd PD       0:00      1 (None)',
                  '          10990728   xeon-p8  xa1001d   alongd  R       0:05      1 d-19-14-2',
                  ]
        running_job_ids = local.parse_running_jobs_ids(stdout, cluster_soft='slurm')
        self.assertEqual(running_job_ids, ['10990729', '10990728'])

        # HTCondor:
        stdout = ['11224.0 R 8 6759 a2495 7',
                  '11225.0 R 8 6759 a2496 6',
                  '11226.0 R 8 6759 a2497 7',
                  '11227.0 R 8 6759 a2498 7',
                  '11228.0 R 8 6759 a2499 7',
                  '11229.0 P 8 6759 a2500 14',
                  '11230.0 P 8 6759 a2501 13',
                  '11231.0 P 8 6759 a2502 13',
                  ]
        running_job_ids = local.parse_running_jobs_ids(stdout, cluster_soft='htcondor')
        self.assertEqual(running_job_ids, ['11224', '11225', '11226', '11227', '11228', '11229', '11230', '11231'])

    def test_submit_job_pbs_compute_node_error(self):
        """Test submit_job() error handling for PBS compute node submissions."""
        stderr = ['qsub: Unauthorized Request: Please do NOT submit jobs on compute nodes!',
                  'Jobs should be submitted on login server, i.e. ZEUS.']
        with patch('arc.job.local.execute_command', side_effect=[([], stderr), ([], stderr)]):
            with patch('time.sleep', return_value=None):
                with self.assertRaises(ValueError) as cm:
                    local.submit_job(path='.', cluster_soft='pbs', submit_cmd='qsub', submit_filename='submit.sh')
        self.assertIn('compute node', str(cm.exception))

    def test_check_running_jobs_ids_without_a_queueing_system(self):
        """Test that a machine with no queueing system reports no queue job IDs"""
        for cluster_soft in ['local', 'Local']:
            with patch.dict(local.servers, {'local': {'cluster_soft': cluster_soft}}):
                with patch('arc.job.local.execute_command') as mock_execute:
                    self.assertEqual(local.check_running_jobs_ids(), list())
                mock_execute.assert_not_called()

    def test_check_running_jobs_ids_unsupported_cluster_software(self):
        """Test that an unrecognized cluster software is still rejected"""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'no_such_scheduler'}}):
            with self.assertRaises(ValueError):
                local.check_running_jobs_ids()


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
