#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.local module
"""

import datetime
import errno
import os
import shutil
import subprocess
import unittest
from unittest.mock import patch

import arc.job.local as local
import arc.job.ssh as ssh
from arc.common import ARC_PATH
from arc.exceptions import ServerError


class TestLocal(unittest.TestCase):
    """
    Contains unit tests for the local module
    """

    def setUp(self):
        """A function that is run before every unit test in this class"""
        ssh.reset_queue_query_history()
        self.addCleanup(ssh.reset_queue_query_history)

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

    def test_execute_command_return_code(self):
        """Test that execute_command() reports the exit status only when asked to"""
        self.assertEqual(len(local.execute_command('echo hello')), 2)
        stdout, stderr, return_code = local.execute_command('echo hello', return_code=True)
        self.assertEqual(stdout, ['hello'])
        self.assertEqual(return_code, 0)
        stdout, stderr, return_code = local.execute_command('exit 3', return_code=True)
        self.assertEqual(return_code, 3)
        stdout, stderr, return_code = local.execute_command('a_command_that_does_not_exist', return_code=True)
        self.assertEqual(return_code, 127)
        self.assertEqual(stdout, [])
        self.assertTrue(len(stderr))

    def test_execute_command_retries_a_process_spawn_failure(self):
        """Test that a command whose process could not be spawned is retried instead of raising"""
        completed_process = subprocess.CompletedProcess(args=['echo hello'], returncode=0, stdout=b'hello\n', stderr=b'')
        side_effect = [OSError(errno.ENOMEM, 'Cannot allocate memory', '/bin/sh'),
                       OSError(errno.EAGAIN, 'Resource temporarily unavailable', '/bin/sh'),
                       completed_process]
        with patch('arc.job.local.subprocess.run', side_effect=side_effect) as run_mock:
            with patch('time.sleep', return_value=None) as sleep_mock:
                stdout, stderr = local.execute_command('echo hello')
        self.assertEqual(stdout, ['hello'])
        self.assertEqual(stderr, [])
        self.assertEqual(run_mock.call_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)

    def test_execute_command_returns_the_exit_status_after_a_process_spawn_failure(self):
        """Test that a retried command still reports its exit status when asked to"""
        completed_process = subprocess.CompletedProcess(args=['echo hello'], returncode=3, stdout=b'hello\n', stderr=b'')
        side_effect = [OSError(errno.ENOMEM, 'Cannot allocate memory', '/bin/sh'), completed_process]
        with patch('arc.job.local.subprocess.run', side_effect=side_effect):
            with patch('time.sleep', return_value=None):
                result = local.execute_command('echo hello', return_code=True)
        self.assertEqual(result, (['hello'], [], 3))

    def test_execute_command_gives_up_on_a_persistent_process_spawn_failure(self):
        """Test that a command whose process can never be spawned fails with an actionable error"""
        error = OSError(errno.ENOMEM, 'Cannot allocate memory', '/bin/sh')
        with patch('arc.job.local.subprocess.run', side_effect=error) as run_mock:
            with patch('time.sleep', return_value=None):
                with self.assertRaises(ServerError) as cm:
                    local.execute_command('echo hello')
        self.assertEqual(run_mock.call_count, 10)
        self.assertIn('too many processes', str(cm.exception))
        self.assertIn('ran out of memory', str(cm.exception))

    def test_execute_command_does_not_crash_a_no_fail_call_on_a_process_spawn_failure(self):
        """Test that a no_fail command whose process can never be spawned returns the expected arity"""
        error = OSError(errno.ENOMEM, 'Cannot allocate memory', '/bin/sh')
        with patch('arc.job.local.subprocess.run', side_effect=error):
            with patch('time.sleep', return_value=None):
                self.assertEqual(local.execute_command('echo hello', no_fail=True), (None, None))
                self.assertEqual(local.execute_command('echo hello', no_fail=True, return_code=True),
                                 (None, None, None))

    def test_execute_command_does_not_retry_a_non_transient_spawn_failure(self):
        """Test that an error which retrying cannot resolve is raised without retrying"""
        error = FileNotFoundError(errno.ENOENT, 'No such file or directory', '/bin/no_such_shell')
        with patch('arc.job.local.subprocess.run', side_effect=error) as run_mock:
            with patch('time.sleep', return_value=None):
                with self.assertRaises(FileNotFoundError):
                    local.execute_command('echo hello', executable='/bin/no_such_shell')
        self.assertEqual(run_mock.call_count, 1)

    def test_check_job_status_of_a_failed_query(self):
        """Test that a job is reported as running when the queue could not be queried"""
        with patch('arc.job.local.execute_command', return_value=([], ['qstat: cannot connect to server'], 1)):
            self.assertEqual(local.check_job_status(4556708), 'running')

    def test_check_running_jobs_ids_of_a_failed_query(self):
        """Test that a queue which could not be queried is not reported as an empty queue"""
        with patch('arc.job.local.execute_command', return_value=([], ['qstat: cannot connect to server'], 1)):
            self.assertIsNone(local.check_running_jobs_ids())

    def test_check_running_jobs_ids_does_not_parse_a_partial_failed_query(self):
        """Test that the output of a query which failed is not parsed"""
        stdout = ['5231.0 R 10 7885 a20596 130', '5232.0 R 10 7885 a20597 130']
        with patch('arc.job.local.execute_command', return_value=(stdout, ['qstat: cannot connect'], 1)):
            self.assertIsNone(local.check_running_jobs_ids())

    def test_check_running_jobs_ids_of_a_silently_failed_query(self):
        """Test that a queue status command which failed without a diagnostic is not an empty queue"""
        for return_code in (124, -9, 2):
            ssh.reset_queue_query_history()
            with patch('arc.job.local.execute_command', return_value=([], [], return_code)):
                self.assertIsNone(local.check_running_jobs_ids(),
                                  msg=f'A query which exited with {return_code} was read as an empty queue')

    def test_check_running_jobs_ids_of_an_answered_query(self):
        """Test that a query which answered is parsed, whether or not its exit status is zero"""
        stdout = ['5231.0 R 10 7885 a20596 130', '5232.0 R 10 7885 a20597 130']
        with patch('arc.job.local.execute_command', return_value=(stdout, [], 0)):
            self.assertEqual(local.check_running_jobs_ids(), ['5231', '5232'])
        with patch('arc.job.local.execute_command', return_value=([], [], 1)):
            self.assertEqual(local.check_running_jobs_ids(), [])

    def test_check_running_jobs_ids_gives_up_after_the_tolerance(self):
        """Test that a queue which is unanswerable for longer than the tolerance stops the run"""
        with patch('arc.job.local.execute_command', return_value=([], ['qstat: cannot connect to server'], 1)):
            self.assertIsNone(local.check_running_jobs_ids())
            ssh._queue_query_history['local']['failing_since'] = \
                datetime.datetime.now() - ssh.QUEUE_QUERY_TOLERANCE - datetime.timedelta(minutes=1)
            with self.assertRaises(ServerError):
                local.check_running_jobs_ids()

    def test_delete_job_does_not_claim_deletion_on_an_unanswerable_query(self):
        """Test that a failed deletion is not reported as successful when the queue cannot be queried"""
        with patch('arc.job.local.execute_command', return_value=([], ['qdel: request rejected'], 1)):
            with self.assertLogs(logger='arc', level='ERROR') as cm:
                local.delete_job(4556708)
        self.assertTrue(any('unknown whether the job was deleted' in message for message in cm.output))

    def test_check_running_jobs_ids_without_a_queueing_system(self):
        """Test that a machine with no queueing system reports no queue job IDs"""
        for cluster_soft in ['local', 'Local']:
            with patch.dict(local.servers, {'local': {'cluster_soft': cluster_soft}}):
                with patch('arc.job.local.execute_command') as mock_execute:
                    self.assertEqual(local.check_running_jobs_ids(), list())
                mock_execute.assert_not_called()

    def test_check_running_jobs_ids_unsupported_cluster_software(self):
        """Test that an unrecognized cluster software is still rejected, naming the server"""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'no_such_scheduler'}}):
            with self.assertRaises(ValueError) as cm:
                local.check_running_jobs_ids()
        self.assertIn('local', str(cm.exception))
        self.assertIn('no_such_scheduler', str(cm.exception))


class TestClusterSoftNormalisation(unittest.TestCase):
    """``check_running_jobs_ids()`` and ``submit_job()`` must resolve a non-canonical
    ``cluster_soft`` spelling via their real ``get_canonical_cluster_soft('local')`` fallback call
    sites, alias SGE to OGE, refuse an unsupported value naming the server, and leave a canonical
    spelling working unchanged."""

    STATUS_STDOUT = {
        'slurm': ['header', '10990729 rest of line'],
        'oge': ['h0', 'h1', '540420 rest of line'],
        'pbs': ['h0', 'h1', 'h2', 'h3', 'h4', '2016614.zeldo.local rest of line'],
        'htcondor': ['3261.0 R 10 28161 a2719 56'],
    }
    STATUS_EXPECTED_FIRST_ID = {'slurm': '10990729', 'oge': '540420', 'pbs': '2016614', 'htcondor': '3261'}

    def test_check_running_jobs_ids_resolves_non_canonical_spelling_per_software(self):
        """The status path resolves a non-canonical spelling for every supported cluster software."""
        non_canonical = {'slurm': 'SLURM', 'oge': 'Oge', 'pbs': 'Pbs', 'htcondor': 'HTCONDOR'}
        for canonical_lower, raw in non_canonical.items():
            with self.subTest(raw=raw):
                with patch.dict(local.servers, {'local': {'cluster_soft': raw}}):
                    with patch('arc.job.local.execute_command',
                               return_value=(self.STATUS_STDOUT[canonical_lower], [], 0)):
                        job_ids = local.check_running_jobs_ids()
                self.assertEqual(job_ids[0], self.STATUS_EXPECTED_FIRST_ID[canonical_lower])

    def test_check_running_jobs_ids_aliases_sge_to_oge(self):
        """The status path treats SGE as OGE."""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'sge'}}):
            with patch('arc.job.local.execute_command', return_value=(self.STATUS_STDOUT['oge'], [], 0)):
                job_ids = local.check_running_jobs_ids()
        self.assertEqual(job_ids[0], self.STATUS_EXPECTED_FIRST_ID['oge'])

    def test_check_running_jobs_ids_canonical_value_still_works(self):
        """A canonical spelling on the status path still resolves and parses as before."""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'PBS'}}):
            with patch('arc.job.local.execute_command', return_value=(self.STATUS_STDOUT['pbs'], [], 0)):
                job_ids = local.check_running_jobs_ids()
        self.assertEqual(job_ids[0], self.STATUS_EXPECTED_FIRST_ID['pbs'])

    SUBMIT_STDOUT = {
        'slurm': (['Submitted batch job 17670585'], []),
        'oge': (['Your job 540420 ("job") has been submitted'], []),
        'pbs': (['2016614.zeldo.local'], []),
        'htcondor': (['Submitting job(s).', '1 job(s) submitted to cluster 5263.'], []),
    }
    SUBMIT_EXPECTED_ID = {'slurm': '17670585', 'oge': '540420', 'pbs': '2016614', 'htcondor': '5263'}

    def test_submit_job_resolves_non_canonical_spelling_per_software(self):
        """The submission path resolves a non-canonical spelling for every supported cluster software."""
        non_canonical = {'slurm': 'SLURM', 'oge': 'Oge', 'pbs': 'Pbs', 'htcondor': 'HTCONDOR'}
        for canonical_lower, raw in non_canonical.items():
            with self.subTest(raw=raw):
                with patch.dict(local.servers, {'local': {'cluster_soft': raw}}):
                    with patch('arc.job.local.execute_command',
                               return_value=self.SUBMIT_STDOUT[canonical_lower]):
                        status, job_id = local.submit_job(path='.')
                self.assertEqual(status, 'running')
                self.assertEqual(job_id, self.SUBMIT_EXPECTED_ID[canonical_lower])

    def test_submit_job_aliases_sge_to_oge(self):
        """The submission path treats SGE as OGE."""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'sge'}}):
            with patch('arc.job.local.execute_command', return_value=self.SUBMIT_STDOUT['oge']):
                status, job_id = local.submit_job(path='.')
        self.assertEqual(status, 'running')
        self.assertEqual(job_id, self.SUBMIT_EXPECTED_ID['oge'])

    def test_submit_job_rejects_unsupported_value_naming_the_server(self):
        """The submission path refuses an unrecognised cluster_soft, naming the actual server."""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'sun grid engine'}}):
            with self.assertRaises(ValueError) as cm:
                local.submit_job(path='.')
        self.assertIn('local', str(cm.exception))
        self.assertIn('sun grid engine', str(cm.exception))

    def test_submit_job_canonical_value_still_works(self):
        """A canonical spelling on the submission path still resolves and parses as before."""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'Slurm'}}):
            with patch('arc.job.local.execute_command', return_value=self.SUBMIT_STDOUT['slurm']):
                status, job_id = local.submit_job(path='.')
        self.assertEqual(status, 'running')
        self.assertEqual(job_id, self.SUBMIT_EXPECTED_ID['slurm'])


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
