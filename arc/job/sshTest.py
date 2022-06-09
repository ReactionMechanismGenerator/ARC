#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.ssh module
"""

import unittest

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
 582682 0.45451 a9654      un       e     04/17/2019 16:22:14 long5@node93.cluster              48
 588334 0.45451 pf1005a    un       r     05/07/2019 16:24:31 long3@node67.cluster              48
 588345 0.45451 a14121     un       r     05/08/2019 02:11:42 long3@node69.cluster              48    """
        status = ssh.check_job_status_in_stdout(job_id=588345, stdout=stdout_1, server='server1')
        self.assertEqual(status, 'running')
        status = ssh.check_job_status_in_stdout(job_id=582682, stdout=stdout_1, server='server1')
        self.assertEqual(status, 'errored')
        status = ssh.check_job_status_in_stdout(job_id=582600, stdout=stdout_1, server='server1')
        self.assertEqual(status, 'done')

        # Slurm
        stdout_2 = ['14428     debug xq1371m2   user_name  R 50-04:04:46      1 node06',
                    '14529     debug xq1371m2   user_name  F 50-04:04:46      1 node06',
                    '14429     debug xq1371m2   user_name PD 50-04:04:46      1 node06']
        status = ssh.check_job_status_in_stdout(job_id=14428, stdout=stdout_2, cluster_soft='Slurm')
        self.assertEqual(status, 'running')
        status = ssh.check_job_status_in_stdout(job_id='14428', stdout=stdout_2, cluster_soft='Slurm')
        self.assertEqual(status, 'running')
        status = ssh.check_job_status_in_stdout(job_id='14429', stdout=stdout_2, cluster_soft='Slurm')
        self.assertEqual(status, 'running')
        status = ssh.check_job_status_in_stdout(job_id=14430, stdout=stdout_2, cluster_soft='Slurm')
        self.assertEqual(status, 'done')
        status = ssh.check_job_status_in_stdout(job_id=14529, stdout=stdout_2, cluster_soft='Slurm')
        self.assertEqual(status, 'errored')

        # HTCondor
        stdout_3 = ['5231.0 R 10 7885 a20596 130',
                    '5232.0 R 10 7885 a20597 130',
                    '5233.0 R 10 7885 a20598 130',
                    '5241.0 P 10 7885 a20616 0']
        status = ssh.check_job_status_in_stdout(job_id=5231, stdout=stdout_3, server='local')
        self.assertEqual(status, 'running')
        status = ssh.check_job_status_in_stdout(job_id=5241, stdout=stdout_3, cluster_soft='HTCondor')
        self.assertEqual(status, 'running')
        status = ssh.check_job_status_in_stdout(job_id=4000, stdout=stdout_3, server='local')
        self.assertEqual(status, 'done')

        # Cobalt
        stdout_4 = ['JobID   User       WallTime  Nodes  State     Location',
                    '========================================================',
                    '602991  user_name  00:01:00  1      queued    None',
                    '602992  user_name  00:01:00  1      starting  0',
                    '602993  user_name  00:01:00  1      running   0',
                    '602994  user_name  00:01:00  1      exiting   0',
                    '602995  user_name  00:01:00  1      errored   0',
                    '602996  user_name  00:01:00  1      killing   0']
        status = ssh.check_job_status_in_stdout(job_id=602991, stdout=stdout_4, cluster_soft='Cobalt')
        self.assertEqual(status, 'running')
        status = ssh.check_job_status_in_stdout(job_id=602992, stdout=stdout_4, cluster_soft='Cobalt')
        self.assertEqual(status, 'running')
        status = ssh.check_job_status_in_stdout(job_id=602993, stdout=stdout_4, cluster_soft='Cobalt')
        self.assertEqual(status, 'running')
        status = ssh.check_job_status_in_stdout(job_id=602994, stdout=stdout_4, cluster_soft='Cobalt')
        self.assertEqual(status, 'running')
        status = ssh.check_job_status_in_stdout(job_id=602995, stdout=stdout_4, cluster_soft='Cobalt')
        self.assertEqual(status, 'errored')
        status = ssh.check_job_status_in_stdout(job_id=602996, stdout=stdout_4, cluster_soft='Cobalt')
        self.assertEqual(status, 'running')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
