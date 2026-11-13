#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.queue_status module
"""

import unittest
from unittest.mock import patch

from arc.exceptions import QueueQueryError, ServerError
from arc.job.queue_status import QUEUE_QUERY_RETRY_DELAYS, queue_query_failed, run_queue_query


SOCKET_TIMEOUT = ['squeue: error: slurm_receive_msg: Socket timed out on send/recv operation\n',
                  'slurm_load_jobs error: Socket timed out on send/recv operation\n']
SLURM_LISTING = ['             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)\n',
                 '              4242     short      a42     user  R       0:01      1 node01\n']


class TestQueueQueryFailed(unittest.TestCase):
    """
    Contains unit tests for queue_query_failed()
    """

    def test_a_clean_listing_is_answered(self):
        """A listing with no stderr is an answered query"""
        self.assertFalse(queue_query_failed(SLURM_LISTING, list()))

    def test_an_empty_queue_is_answered(self):
        """No output at all is an empty queue (OGE prints nothing when no jobs run)"""
        self.assertFalse(queue_query_failed(list(), list()))
        self.assertFalse(queue_query_failed(None, None))

    def test_a_socket_timeout_is_a_failure(self):
        """Slurm's socket timeout leaves stdout empty and the error on stderr"""
        self.assertTrue(queue_query_failed(list(), SOCKET_TIMEOUT))

    def test_blank_stdout_lines_do_not_hide_a_failure(self):
        """ARC's own SSH timeout reply carries a blank stdout line"""
        self.assertTrue(queue_query_failed([''], ['ssh timed-out after two trials']))
        self.assertTrue(queue_query_failed(['\n', '  '], SOCKET_TIMEOUT))

    def test_a_warning_beside_a_listing_is_answered(self):
        """A listing is believed even when a warning came with it"""
        self.assertFalse(queue_query_failed(SLURM_LISTING, ['squeue: warning: deprecated option\n']))

    def test_blank_stderr_lines_are_not_an_error(self):
        self.assertFalse(queue_query_failed(list(), ['', '\n']))

    def test_a_missing_queue_command_is_not_a_failure_where_allowed(self):
        """A local queue command that is not installed reads as no jobs, as it always has, in every shell's wording"""
        for stderr in (['/bin/sh: 1: condor_q: not found\n'],
                       ['bash: squeue: command not found\n'],
                       ['bash: line 1: squeue: command not found\n'],
                       ['bash: /usr/bin/squeue: No such file or directory\n']):
            self.assertFalse(queue_query_failed(list(), stderr, missing_command_reads_empty=True), stderr)

    def test_a_missing_queue_command_is_a_failure_by_default(self):
        """On a remote queue server a missing command is a broken query, e.g., after a PATH or module change"""
        self.assertTrue(queue_query_failed(list(), ['bash: squeue: command not found\n']))

    def test_an_error_that_merely_mentions_not_found_is_a_failure(self):
        """Only the shell's own missing-command line is exempt"""
        for stderr in (['slurm_load_jobs error: host not found in the middle of it'],
                       ['squeue: error: Munge encode failed: Failed to access "/var/run/munge/munge.socket.2": '
                        'No such file or directory'],
                       ['slurm_load_jobs error: Unable to contact slurm controller: not found']):
            self.assertTrue(queue_query_failed(list(), stderr, missing_command_reads_empty=True), stderr)

    def test_a_missing_command_line_does_not_hide_another_error(self):
        stderr = ['/bin/sh: 1: module: not found\n'] + SOCKET_TIMEOUT
        self.assertTrue(queue_query_failed(list(), stderr, missing_command_reads_empty=True))


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
        run, calls = self._run((SLURM_LISTING, list()))
        self.assertEqual(run_queue_query(run, server='srv'), SLURM_LISTING)
        self.assertEqual(len(calls), 1)
        self.sleep.assert_not_called()

    def test_a_query_that_fails_once_is_retried(self):
        run, calls = self._run((list(), SOCKET_TIMEOUT), (SLURM_LISTING, list()))
        self.assertEqual(run_queue_query(run, server='srv'), SLURM_LISTING)
        self.assertEqual(len(calls), 2)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [QUEUE_QUERY_RETRY_DELAYS[0]])

    def test_a_query_that_keeps_failing_raises_after_every_delay(self):
        run, calls = self._run(*[(list(), SOCKET_TIMEOUT)] * (len(QUEUE_QUERY_RETRY_DELAYS) + 1))
        with self.assertRaises(QueueQueryError) as cm:
            run_queue_query(run, server='srv')
        self.assertEqual(len(calls), len(QUEUE_QUERY_RETRY_DELAYS) + 1)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], list(QUEUE_QUERY_RETRY_DELAYS))
        self.assertIn('srv', str(cm.exception))
        self.assertIn('Socket timed out', str(cm.exception))
        self.assertIsInstance(cm.exception, ServerError)

    def test_every_failed_attempt_is_logged(self):
        """An early, specific error is kept in the log even when a later attempt fails differently"""
        run, _ = self._run((list(), ['slurm_load_jobs error: Invalid user']), (list(), SOCKET_TIMEOUT),
                           (list(), SOCKET_TIMEOUT))
        with self.assertLogs('arc', level='DEBUG') as logged, self.assertRaises(QueueQueryError):
            run_queue_query(run, server='srv')
        self.assertEqual(len(logged.output), 3)
        self.assertIn('Invalid user', logged.output[0])

    def test_a_missing_command_is_not_retried_where_allowed(self):
        run, calls = self._run((None, ['/bin/sh: 1: condor_q: not found']))
        self.assertEqual(run_queue_query(run, server='local', missing_command_reads_empty=True), list())
        self.assertEqual(len(calls), 1)
        self.sleep.assert_not_called()

    def test_no_retry_delays_means_a_single_attempt(self):
        run, calls = self._run((list(), SOCKET_TIMEOUT))
        with self.assertRaises(QueueQueryError):
            run_queue_query(run, server='srv', retry_delays=tuple())
        self.assertEqual(len(calls), 1)
        self.sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
