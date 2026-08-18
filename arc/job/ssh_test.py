#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.ssh module
"""

import datetime
import unittest

import arc.job.ssh as ssh
from arc.exceptions import ServerError, SettingsError


class TestSSH(unittest.TestCase):
    """
    Contains unit tests for the SSH module
    """

    def setUp(self):
        """A function that is run before every unit test in this class"""
        ssh.reset_queue_query_history()
        self.addCleanup(ssh.reset_queue_query_history)

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

    def test_check_job_status_in_stdout_failed_query(self):
        """Test that a queue status command which failed is not reported as a finished job"""
        stdout = ['5231.0 R 10 7885 a20596 130']
        self.assertEqual(ssh.check_job_status_in_stdout(job_id=4000, stdout=stdout, server='local'), 'done')
        self.assertEqual(ssh.check_job_status_in_stdout(job_id=4000, stdout=stdout, server='local',
                                                        return_code=0), 'done')
        self.assertEqual(ssh.check_job_status_in_stdout(job_id=4000, stdout=[], server='local', return_code=2,
                                                        stderr=['qstat: cannot connect to server']), 'running')
        self.assertEqual(ssh.check_job_status_in_stdout(job_id=4000, stdout=[], server='local',
                                                        return_code=ssh.COMMAND_NOT_FOUND_RETURN_CODE,
                                                        stderr=['/usr/local/bin/qstat: not found']), 'running')
        self.assertEqual(ssh.check_job_status_in_stdout(job_id=5231, stdout=stdout, server='local',
                                                        return_code=0), 'running')

    def test_queue_query_failed(self):
        """Test identifying a queue status query which failed to answer"""
        self.assertFalse(ssh.queue_query_failed())
        self.assertFalse(ssh.queue_query_failed(return_code=0, stderr=[]))
        self.assertFalse(ssh.queue_query_failed(return_code=0, stderr=['a warning was printed']))
        self.assertFalse(ssh.queue_query_failed(return_code=1, stderr=[]))
        self.assertFalse(ssh.queue_query_failed(return_code=1, stderr=['\n', '  ']))
        self.assertTrue(ssh.queue_query_failed(return_code=1, stderr=['qstat: cannot connect to server']))
        self.assertTrue(ssh.queue_query_failed(return_code=127, stderr='/usr/local/bin/qstat: not found'))

    def test_queue_query_failed_without_a_diagnostic_on_stderr(self):
        """Test identifying a queue status query which failed silently"""
        self.assertTrue(ssh.queue_query_failed(return_code=124, stderr=[]))
        self.assertTrue(ssh.queue_query_failed(return_code=-9, stderr=[]))
        self.assertTrue(ssh.queue_query_failed(return_code=2, stderr=[]))
        self.assertTrue(ssh.queue_query_failed(return_code=ssh.COMMAND_NOT_FOUND_RETURN_CODE, stderr=[]))
        self.assertFalse(ssh.queue_query_failed(return_code=ssh.AMBIGUOUS_RETURN_CODE, stderr=[]))

    def test_register_queue_query_tolerates_a_transient_outage(self):
        """Test that a queue which has been failing for less than the tolerance is tolerated"""
        stderr = ['qstat: cannot connect to server']
        ssh.register_queue_query(failed=False, server='local', return_code=0)
        for _ in range(5):
            ssh.register_queue_query(failed=True, server='local', return_code=1, stderr=stderr)
        history = ssh._queue_query_history['local']
        self.assertEqual(history['consecutive_failures'], 5)
        self.assertIsNotNone(history['failing_since'])
        ssh.register_queue_query(failed=False, server='local', return_code=0)
        self.assertEqual(ssh._queue_query_history['local']['consecutive_failures'], 0)
        self.assertIsNone(ssh._queue_query_history['local']['failing_since'])

    def test_register_queue_query_gives_up_after_the_tolerance(self):
        """Test that a queue which has been failing for longer than the tolerance stops the run"""
        stderr = ['qstat: cannot connect to server']
        ssh.register_queue_query(failed=False, server='local', return_code=0)
        ssh.register_queue_query(failed=True, server='local', return_code=1, stderr=stderr)
        self.assertGreater(ssh.QUEUE_QUERY_TOLERANCE, datetime.timedelta(hours=2))
        ssh._queue_query_history['local']['failing_since'] = \
            datetime.datetime.now() - ssh.QUEUE_QUERY_TOLERANCE - datetime.timedelta(minutes=1)
        with self.assertRaises(ServerError) as cm:
            ssh.register_queue_query(failed=True, server='local', return_code=1, stderr=stderr)
        self.assertIn('cannot tell which of its jobs are still running', str(cm.exception))

    def test_register_queue_query_command_not_found(self):
        """Test that a queue status command which was never found is reported as a settings error"""
        stderr = ['/usr/local/bin/qstat: not found']
        ssh.register_queue_query(failed=True, server='local', return_code=ssh.COMMAND_NOT_FOUND_RETURN_CODE,
                                 stderr=stderr)
        ssh._queue_query_history['local']['failing_since'] = \
            datetime.datetime.now() - ssh.COMMAND_NOT_FOUND_TOLERANCE - datetime.timedelta(minutes=1)
        with self.assertRaises(SettingsError) as cm:
            ssh.register_queue_query(failed=True, server='local', return_code=ssh.COMMAND_NOT_FOUND_RETURN_CODE,
                                     stderr=stderr)
        self.assertIn('check_status_command', str(cm.exception))
        self.assertIn('arc/settings/settings.py', str(cm.exception))
        self.assertIn('~/.arc/settings.py', str(cm.exception))
        self.assertIn(ssh.get_check_status_command('local'), str(cm.exception))

    def test_register_queue_query_tolerates_a_transient_command_not_found(self):
        """Test that a queue status command which was not found is tolerated for a short while"""
        self.assertLess(ssh.COMMAND_NOT_FOUND_TOLERANCE, ssh.QUEUE_QUERY_TOLERANCE)
        for _ in range(10):
            ssh.register_queue_query(failed=True, server='local', return_code=ssh.COMMAND_NOT_FOUND_RETURN_CODE,
                                     stderr=['/opt/sge/bin/lx24-amd64/qstat: No such file or directory'])
        self.assertEqual(ssh._queue_query_history['local']['consecutive_failures'], 10)

    def test_register_queue_query_does_not_repeat_the_warning(self):
        """Test that a persistently failing queue is not warned about on every query"""
        stderr = ['qstat: cannot connect to server']
        with self.assertLogs(logger='arc', level='WARNING') as cm:
            for _ in range(20):
                ssh.register_queue_query(failed=True, server='local', return_code=2, stderr=stderr)
        self.assertEqual(len(cm.output), 1)
        ssh._queue_query_history['local']['last_warned'] = \
            datetime.datetime.now() - ssh.QUEUE_QUERY_WARNING_INTERVAL - datetime.timedelta(minutes=1)
        with self.assertLogs(logger='arc', level='WARNING') as cm:
            ssh.register_queue_query(failed=True, server='local', return_code=2, stderr=stderr)
        self.assertEqual(len(cm.output), 1)

    def test_register_queue_query_command_not_found_after_the_queue_answered(self):
        """Test that a command not found error is tolerated once the server has answered before"""
        ssh.register_queue_query(failed=False, server='local', return_code=0)
        ssh.register_queue_query(failed=True, server='local', return_code=ssh.COMMAND_NOT_FOUND_RETURN_CODE,
                                 stderr=['/usr/local/bin/qstat: not found'])
        self.assertEqual(ssh._queue_query_history['local']['consecutive_failures'], 1)

    def test_register_queue_query_tracks_servers_separately(self):
        """Test that an answering server does not clear the failures recorded for another server"""
        ssh.register_queue_query(failed=True, server='server1', return_code=1, stderr=['qstat: cannot connect'])
        ssh.register_queue_query(failed=False, server='local', return_code=0)
        self.assertEqual(ssh._queue_query_history['server1']['consecutive_failures'], 1)
        self.assertEqual(ssh._queue_query_history['local']['consecutive_failures'], 0)

    def test_get_check_status_command(self):
        """Test getting the configured queue status command without raising on an unknown server"""
        self.assertIsNotNone(ssh.get_check_status_command('local'))
        self.assertIsNone(ssh.get_check_status_command('a_server_that_is_not_configured'))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
