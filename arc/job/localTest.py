#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.local module
"""

import datetime
import os
import shutil
import unittest

import arc.job.local as local
from arc.common import ARC_PATH


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
        stdout = ['Submitted batch job 17670585']
        job_id = local._determine_job_id(stdout, cluster_soft='slurm')
        self.assertEqual(job_id, '17670585')

        # HTCondor
        stdout = ['Submitting job(s).', '1 job(s) submitted to cluster 5263.']
        job_id = local._determine_job_id(stdout)
        self.assertEqual(job_id, '5263')

        # Cobalt
        stdout = ['Job routed to queue "debug-flat-quad".',
                  'Memory mode set to flat quad for queue debug-flat-quad',
                  'WARNING: Filesystem attribute not set for this job submission.',
                  'This job will be set to request all filesystems.  In the event',
                  'of a filesystem outage, this job may be put on hold unnecessarily.',
                  "Setting attrs to:  {'numa': 'quad', 'mcdram': 'flat', 'filesystems': 'home,grand,eagle,theta-fs0'}",
                  '603441']
        job_id = local._determine_job_id(stdout, cluster_soft='Cobalt')
        self.assertEqual(job_id, '603441')

        # Wrong server name
        with self.assertRaises(ValueError):
            local._determine_job_id(stdout, cluster_soft='wrong')

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
        # OGE
        stdout = [
            'job-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID',
            '-----------------------------------------------------------------------------------------------------------------',
            ' 582682 0.45451 a9654      un       e     04/17/2019 16:22:14 long5@node93.cluster              48',
            ' 588334 0.45451 pf1005a    un       r     05/07/2019 16:24:31 long3@node67.cluster              48',
            ' 588345 0.45451 a14121     un       r     05/08/2019 02:11:42 long3@node69.cluster              48', ]
        running_job_ids = local.parse_running_jobs_ids(stdout, cluster_soft='OGE')
        self.assertEqual(running_job_ids, ['582682', '588334', '588345'])
        stdout = """job-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID
 -----------------------------------------------------------------------------------------------------------------
 582685 0.45451 a9654      un       e     04/17/2019 16:22:14 long5@node93.cluster              48
 588336 0.45451 pf1005a    un       r     05/07/2019 16:24:31 long3@node67.cluster              48
 588347 0.45451 a14121     un       r     05/08/2019 02:11:42 long3@node69.cluster              48 """
        running_job_ids = local.parse_running_jobs_ids(stdout, cluster_soft='OGE')
        self.assertEqual(running_job_ids, ['582685', '588336', '588347'])

        # Slurm:
        stdout = ['             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)',
                  '          10990729    normal     a207   un PD       0:00      1 (None)',
                  '          10990728   xeon-p8  xa1001d   un  R       0:05      1 d-19-14-2',
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
        running_job_ids = local.parse_running_jobs_ids(stdout, cluster_soft='HTCondor')
        self.assertEqual(running_job_ids, ['11224', '11225', '11226', '11227', '11228', '11229', '11230', '11231'])

        # Cobalt
        stdout = ['JobID   User       WallTime  Nodes  State     Location',
                  '========================================================',
                  '602991  user_name  00:01:00  1      queued    None',
                  '602992  user_name  00:01:00  1      starting  0',
                  '602993  user_name  00:01:00  1      running   0',
                  '602994  user_name  00:01:00  1      exiting   0',
                  '602995  user_name  00:01:00  1      errored   0',
                  '602996  user_name  00:01:00  1      killing   0']
        running_job_ids = local.parse_running_jobs_ids(stdout, cluster_soft='Cobalt')
        self.assertEqual(running_job_ids, ['602991', '602992', '602993', '602994', '602995', '602996'])


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
