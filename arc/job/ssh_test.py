#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.ssh module
"""

import unittest
from unittest.mock import patch, MagicMock

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
        self.assertEqual(status1, 'queued')
        status1 = ssh.check_job_status_in_stdout(job_id=4000, stdout=stdout_2, server='local')
        self.assertEqual(status1, 'done')

    def test_get_job_history_local_pbs(self):
        """Ensure get_job_history runs locally for PBS when server=='local'."""
        orig_cluster = ssh.servers['local'].get('cluster_soft')
        orig_address = ssh.servers['local'].get('address')
        orig_un = ssh.servers['local'].get('un')
        orig_key = ssh.servers['local'].get('key')
        ssh.servers['local']['cluster_soft'] = 'PBS'
        ssh.servers['local']['address'] = 'localhost'
        ssh.servers['local']['un'] = 'user'
        ssh.servers['local']['key'] = ''
        try:
            mock_completed = MagicMock(returncode=0, stdout='line1\nline2\n', stderr='')
            with patch('arc.job.ssh.subprocess.run', return_value=mock_completed) as sub_run:
                stdout, stderr = ssh.get_job_history_for_server('12345', 'local')
                sub_run.assert_called_once()
                self.assertEqual(stdout, ['line1', 'line2'])
                self.assertEqual(stderr, [])
        finally:
            if orig_cluster is not None:
                ssh.servers['local']['cluster_soft'] = orig_cluster
            else:
                ssh.servers['local'].pop('cluster_soft', None)
            if orig_address is not None:
                ssh.servers['local']['address'] = orig_address
            else:
                ssh.servers['local'].pop('address', None)
            if orig_un is not None:
                ssh.servers['local']['un'] = orig_un
            else:
                ssh.servers['local'].pop('un', None)
            if orig_key is not None:
                ssh.servers['local']['key'] = orig_key
            else:
                ssh.servers['local'].pop('key', None)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
