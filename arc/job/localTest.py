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
        self.assertEqual(out1[1], '')
        self.assertIn('arc', out1[0])
        self.assertIn('ARC.py', out1[0])
        self.assertIn('environment.yml', out1[0])

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


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
