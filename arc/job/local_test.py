#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.local module
"""

import datetime
import os
import shutil
import unittest
from unittest.mock import patch

import arc.job.local as local
from arc.common import ARC_PATH
from arc.exceptions import ServerError


SOCKET_TIMEOUT = ['squeue: error: slurm_receive_msg: Socket timed out on send/recv operation',
                  'slurm_load_jobs error: Socket timed out on send/recv operation']
SLURM_LISTING = ['             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)',
                 '              4242     short      a42     user  R       0:01      1 node01']


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

    def _submit_with_cap(self, cluster_soft, cap_stderr, success_stdout, n_refusals):
        """Submit through a scheduler that refuses ``n_refusals`` times, then accepts."""
        responses = [([], list(cap_stderr))] * n_refusals + [(success_stdout, [])]
        with patch('arc.job.local.execute_command', side_effect=responses) as mock_execute:
            with patch('time.sleep', return_value=None) as mock_sleep:
                result = local.submit_job(path='.', cluster_soft=cluster_soft, submit_cmd='sub',
                                          submit_filename='submit.sh')
        return result, mock_execute, mock_sleep

    def test_submit_job_pbs_cap_retries_until_accepted(self):
        """Test that a PBS per-user cap keeps retrying and then tracks the job that finally got submitted"""
        for cap_line in ('qsub: would exceed queue generic per-user limit of 40 jobs',
                         'qsub: Maximum number of jobs already in queue'):
            with self.subTest(cap_line=cap_line):
                result, mock_execute, mock_sleep = self._submit_with_cap(
                    'pbs', [cap_line], ['12345.zeus'], n_refusals=3)
                self.assertEqual(result, ('running', '12345'))
                self.assertEqual(mock_execute.call_count, 4)
                cap_sleeps = [call.args[0] for call in mock_sleep.call_args_list
                              if call.args[0] == local.SUBMISSION_CAP_WAIT_SECONDS]
                self.assertEqual(len(cap_sleeps), 3)

    def test_submit_job_slurm_cap_retries_until_accepted(self):
        """Test that a Slurm per-user cap keeps retrying and then tracks the job that finally got submitted"""
        result, mock_execute, mock_sleep = self._submit_with_cap(
            'slurm', ['sbatch: error: AssocMaxSubmitJobLimit'], ['Submitted batch job 777'], n_refusals=2)
        self.assertEqual(result, ('running', '777'))
        self.assertEqual(mock_execute.call_count, 3)
        cap_sleeps = [call.args[0] for call in mock_sleep.call_args_list
                      if call.args[0] == local.SUBMISSION_CAP_WAIT_SECONDS]
        self.assertEqual(len(cap_sleeps), 2)

    def test_submit_job_returns_the_cap_refusal_when_not_waiting(self):
        """Test that wait_on_cap=False hands a per-user cap refusal back at once, for every scheduler"""
        for cluster_soft, cap_line in (('pbs', "qsub: would exceed queue generic's per-user limit"),
                                       ('slurm', 'sbatch: error: AssocMaxSubmitJobLimit')):
            with self.subTest(cluster_soft=cluster_soft):
                with patch('arc.job.local.execute_command', return_value=([], [cap_line])) as mock_execute, \
                        patch('arc.job.local.time.sleep') as mock_sleep:
                    result = local.submit_job(path='.', cluster_soft=cluster_soft, submit_cmd='sub',
                                              submit_filename='submit.sh', wait_on_cap=False)
                self.assertEqual(result, (local.SUBMISSION_CAP_REACHED, None))
                self.assertEqual(mock_execute.call_count, 1)
                mock_sleep.assert_not_called()

    def test_submit_job_wait_on_cap_false_still_accepts_a_submission(self):
        """Test that wait_on_cap=False does not change an accepted submission"""
        with patch('arc.job.local.execute_command', return_value=(['999.zeus'], [])):
            result = local.submit_job(path='.', cluster_soft='pbs', submit_cmd='qsub',
                                      submit_filename='submit.sh', wait_on_cap=False)
        self.assertEqual(result, ('running', '999'))

    def test_submit_job_without_cap_does_not_wait(self):
        """Test that an accepted submission neither waits for the cap nor retries"""
        result, mock_execute, mock_sleep = self._submit_with_cap('pbs', [], ['999.zeus'], n_refusals=0)
        self.assertEqual(result, ('running', '999'))
        self.assertEqual(mock_execute.call_count, 1)
        mock_sleep.assert_not_called()

    def test_submit_job_pbs_illegal_attribute_error(self):
        """Test that an illegal PBS attribute is still reported as an error"""
        stderr = ['qsub: Illegal attribute or resource value']
        with patch('arc.job.local.execute_command', return_value=(['x'], stderr)):
            with self.assertRaises(ValueError) as cm:
                local.submit_job(path='.', cluster_soft='pbs', submit_cmd='qsub', submit_filename='submit.sh')
        self.assertIn('check your submit script', str(cm.exception))

    def test_submit_job_empty_stdout_without_cap_returns_nothing(self):
        """Test that an empty stdout without a cap error retries once after 10 s and then gives up"""
        with patch('arc.job.local.execute_command', return_value=([], [])) as mock_execute:
            with patch('time.sleep', return_value=None) as mock_sleep:
                result = local.submit_job(path='.', cluster_soft='pbs', submit_cmd='qsub',
                                          submit_filename='submit.sh')
        self.assertEqual(result, (None, None))
        self.assertEqual(mock_execute.call_count, 2)
        mock_sleep.assert_called_once_with(10)

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

    def test_check_job_status_does_not_read_a_socket_timeout_as_done(self):
        """A queue that cannot be read must not make a running job look finished"""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'Slurm'}}), \
                patch('arc.job.local.execute_command', return_value=([], SOCKET_TIMEOUT)), \
                patch('time.sleep'):
            with self.assertRaises(ServerError):
                local.check_job_status(4242)

    def test_check_running_jobs_ids_does_not_read_a_socket_timeout_as_no_jobs(self):
        """A queue that cannot be read must not report an empty job list"""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'Slurm'}}), \
                patch('arc.job.local.execute_command', return_value=([], SOCKET_TIMEOUT)), \
                patch('time.sleep'):
            with self.assertRaises(ServerError):
                local.check_running_jobs_ids()

    def test_check_running_jobs_ids_retries_a_failed_query(self):
        """A query that fails once and then succeeds gives the true running list"""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'Slurm'}}), \
                patch('arc.job.local.execute_command',
                      side_effect=[([], SOCKET_TIMEOUT), (SLURM_LISTING, [])]) as mock_execute, \
                patch('time.sleep'):
            self.assertEqual(local.check_running_jobs_ids(), ['4242'])
        self.assertEqual(mock_execute.call_count, 2)

    def test_a_missing_queue_command_still_reads_as_no_jobs(self):
        """A queue command that is not installed is not a transient failure, and is not retried"""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'Slurm'}}), \
                patch('arc.job.local.execute_command',
                      return_value=([], ['/bin/sh: 1: /usr/bin/squeue: not found'])), \
                patch('time.sleep') as mock_sleep:
            self.assertEqual(local.check_running_jobs_ids(), list())
            self.assertEqual(local.check_job_status(4242), 'done')
        mock_sleep.assert_not_called()

    def test_delete_job_does_not_claim_a_job_ended_when_the_queue_cannot_be_read(self):
        """A failed deletion followed by an unreadable queue is reported as unknown, not as ended"""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'Slurm'}}), \
                patch('arc.job.local.execute_command',
                      side_effect=[([], ['scancel: error: Socket timed out'])] + [([], SOCKET_TIMEOUT)] * 3), \
                patch('time.sleep'):
            with self.assertLogs('arc', level='WARNING') as logs:
                local.delete_job(4242)
        self.assertTrue(any('could not verify' in line for line in logs.output))

    def test_delete_all_local_arc_jobs_does_not_read_a_socket_timeout_as_an_empty_queue(self):
        """Deleting every ARC job must not report success when the queue could not be listed"""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'Slurm'}}), \
                patch('arc.job.local.execute_command', return_value=([], SOCKET_TIMEOUT)), \
                patch('time.sleep'):
            with self.assertRaises(ServerError):
                local.delete_all_local_arc_jobs()


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
