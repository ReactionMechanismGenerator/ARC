#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.queue_status module
"""

import unittest
from unittest.mock import patch

from arc.exceptions import QueueQueryError, ServerError
from arc.job.queue_status import (QUEUE_OUTAGE_LIMIT_HRS, QUEUE_QUERY_RETRY_DELAYS, queue_outage_limit,
                                  queue_query_failed, run_queue_query)


SOCKET_TIMEOUT = ['squeue: error: slurm_receive_msg: Socket timed out on send/recv operation\n',
                  'slurm_load_jobs error: Socket timed out on send/recv operation\n']
SLURM_LISTING = ['             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)\n',
                 '              4242     short      a42     user  R       0:01      1 node01\n']
BANNER = ['** WARNING: connection is not using a post-quantum key exchange algorithm.\n']


class TestQueueQueryFailed(unittest.TestCase):
    """
    Contains unit tests for queue_query_failed()
    """

    def test_exit_zero_with_an_empty_stdout_is_an_empty_queue(self):
        """A module warning or a login banner on stderr does not turn an empty queue into a failure"""
        for stderr in (BANNER, ['lmod: warning: module foo is deprecated\n'], SOCKET_TIMEOUT):
            self.assertFalse(queue_query_failed(list(), stderr, exit_status=0), stderr)
        self.assertFalse(queue_query_failed(None, None, exit_status=0))

    def test_exit_zero_with_a_listing_is_answered(self):
        self.assertFalse(queue_query_failed(SLURM_LISTING, BANNER, exit_status=0))

    def test_a_non_zero_exit_is_a_failure(self):
        """Even with text on stdout and nothing on stderr"""
        for exit_status in (1, 2, 255, -9):
            self.assertTrue(queue_query_failed(list(), SOCKET_TIMEOUT, exit_status=exit_status), exit_status)
            self.assertTrue(queue_query_failed(SLURM_LISTING, list(), exit_status=exit_status), exit_status)
            self.assertTrue(queue_query_failed(list(), list(), exit_status=exit_status), exit_status)

    def test_exit_127_is_a_missing_command_in_any_wording(self):
        """The exit status decides, so a message in another language or an empty stderr still counts"""
        for stderr in (['/bin/sh: 1: squeue: introuvable\n'], list(), ['whatever the shell says']):
            self.assertFalse(queue_query_failed(list(), stderr, exit_status=127, missing_command_reads_empty=True))
            self.assertTrue(queue_query_failed(list(), stderr, exit_status=127))

    def test_missing_command_text_without_exit_127_is_not_exempt(self):
        """The shell's wording alone exempts nothing"""
        self.assertTrue(queue_query_failed(list(), ['/bin/sh: 1: condor_q: not found'], exit_status=1,
                                           missing_command_reads_empty=True))
        self.assertTrue(queue_query_failed(list(), ['/bin/sh: 1: condor_q: not found'], exit_status=None,
                                           missing_command_reads_empty=True))

    def test_the_stdout_stderr_rule_applies_only_without_an_exit_status(self):
        self.assertTrue(queue_query_failed(list(), SOCKET_TIMEOUT, exit_status=None))
        self.assertFalse(queue_query_failed(list(), SOCKET_TIMEOUT, exit_status=0))
        self.assertTrue(queue_query_failed([''], ['ssh timed-out after two trials'], exit_status=None))
        self.assertFalse(queue_query_failed(SLURM_LISTING, ['squeue: warning: deprecated option'], exit_status=None))
        self.assertFalse(queue_query_failed(list(), ['', '\n'], exit_status=None))
        self.assertFalse(queue_query_failed(None, None, exit_status=None))


class TestQueueOutageLimit(unittest.TestCase):
    """
    Contains unit tests for queue_outage_limit()
    """

    def test_the_default_is_four_hours(self):
        self.assertEqual(QUEUE_OUTAGE_LIMIT_HRS, 4)

    def test_an_absent_setting_uses_the_default_silently(self):
        with self.assertNoLogs('arc', level='WARNING'):
            self.assertEqual(queue_outage_limit(None), QUEUE_OUTAGE_LIMIT_HRS * 3600)

    def test_a_valid_setting_is_used(self):
        self.assertEqual(queue_outage_limit(12), 12 * 3600)
        self.assertEqual(queue_outage_limit(0.5), 1800)

    def test_a_malformed_setting_warns_and_uses_the_default(self):
        """It never fails open to an unbounded wait"""
        for value in ('4', True, False, 0, -1, float('nan'), float('inf'), 1e308, 10 ** 400, [4], dict()):
            with self.assertLogs('arc', level='WARNING') as logged:
                self.assertEqual(queue_outage_limit(value), QUEUE_OUTAGE_LIMIT_HRS * 3600, value)
            self.assertIn('queue_outage_limit_hrs', logged.output[0])


class TestRunQueueQuery(unittest.TestCase):
    """
    Contains unit tests for run_queue_query()
    """

    def setUp(self):
        sleep = patch('time.sleep')
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)

    @staticmethod
    def _run(*replies):
        """A command runner answering ``replies`` in turn, counting its calls."""
        replies = list(replies)
        calls = list()

        def run():
            calls.append(None)
            return replies[len(calls) - 1]
        return run, calls

    def test_an_answered_query_runs_once(self):
        run, calls = self._run((SLURM_LISTING, list(), 0))
        self.assertEqual(run_queue_query(run, server='srv'), SLURM_LISTING)
        self.assertEqual(len(calls), 1)
        self.sleep.assert_not_called()

    def test_a_query_that_fails_once_is_retried(self):
        run, calls = self._run((list(), SOCKET_TIMEOUT, 1), (SLURM_LISTING, list(), 0))
        self.assertEqual(run_queue_query(run, server='srv'), SLURM_LISTING)
        self.assertEqual(len(calls), 2)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [QUEUE_QUERY_RETRY_DELAYS[0]])

    def test_a_query_that_keeps_failing_raises_after_every_delay(self):
        run, calls = self._run(*[(list(), SOCKET_TIMEOUT, 1)] * (len(QUEUE_QUERY_RETRY_DELAYS) + 1))
        with self.assertRaises(QueueQueryError) as cm:
            run_queue_query(run, server='srv')
        self.assertEqual(len(calls), len(QUEUE_QUERY_RETRY_DELAYS) + 1)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], list(QUEUE_QUERY_RETRY_DELAYS))
        self.assertIn('srv', str(cm.exception))
        self.assertIn('Socket timed out', str(cm.exception))
        self.assertIsInstance(cm.exception, ServerError)
        self.assertEqual(cm.exception.server, 'srv')
        self.assertEqual(cm.exception.exit_status, 1)
        self.assertIn('Socket timed out', cm.exception.stderr)

    def test_a_failure_reported_on_stdout_is_kept(self):
        """A wrapper may print its error on stdout; the diagnostic must not come out empty"""
        run, _ = self._run(*[(['queue wrapper: scheduler unreachable'], BANNER, 1)] * 3)
        with self.assertRaises(QueueQueryError) as cm:
            run_queue_query(run, server='srv')
        self.assertIn('scheduler unreachable', cm.exception.stderr)
        self.assertIn('post-quantum', cm.exception.stderr)

    def test_a_zero_exit_with_a_banner_is_not_retried(self):
        run, calls = self._run((list(), BANNER, 0))
        self.assertEqual(run_queue_query(run, server='srv'), list())
        self.assertEqual(len(calls), 1)
        self.sleep.assert_not_called()

    def test_every_failed_attempt_is_logged(self):
        """An early, specific error is kept in the log even when a later attempt fails differently"""
        run, _ = self._run((list(), ['slurm_load_jobs error: Invalid user'], 1), (list(), SOCKET_TIMEOUT, 1),
                           (list(), SOCKET_TIMEOUT, 1))
        with self.assertLogs('arc', level='DEBUG') as logged, self.assertRaises(QueueQueryError):
            run_queue_query(run, server='srv')
        self.assertEqual(len(logged.output), 3)
        self.assertIn('Invalid user', logged.output[0])

    def test_a_missing_command_is_not_retried_where_allowed(self):
        run, calls = self._run((None, ['/bin/sh: 1: condor_q: not found'], 127))
        self.assertEqual(run_queue_query(run, server='local', missing_command_reads_empty=True), list())
        self.assertEqual(len(calls), 1)
        self.sleep.assert_not_called()

    def test_no_retry_delays_means_a_single_attempt(self):
        run, calls = self._run((list(), SOCKET_TIMEOUT, 1))
        with self.assertRaises(QueueQueryError):
            run_queue_query(run, server='srv', retry_delays=tuple())
        self.assertEqual(len(calls), 1)
        self.sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
