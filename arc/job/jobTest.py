#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests of the arc.job.job module
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest
import os
import datetime

from arc.job.job import Job
from arc.settings import arc_path

################################################################################


class TestJob(unittest.TestCase):
    """
    Contains unit tests for the Job class
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        settings = {'gaussian': 'server1', 'molpro': 'server2', 'qchem': 'server1', 'ssh': False}
        cls.job1 = Job(project='project_test', settings=settings, species_name='tst_spc', xyz='C 0.0 0.0 0.0',
                       job_type='opt', level_of_theory='b3lyp/6-31+g(d)', multiplicity=1,
                       project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        cls.job1.initial_time = datetime.datetime(2019, 3, 15, 19, 53, 7, 0)
        cls.job1.final_time = datetime.datetime(2019, 3, 15, 19, 53, 8, 0)
        cls.job1.determine_run_time()

    def test_as_dict(self):
        """Test Job.as_dict()"""
        job_dict = self.job1.as_dict()
        initial_time = job_dict['initial_time']
        final_time = job_dict['final_time']
        expected_dict = {'initial_time': initial_time,
                         'final_time': final_time,
                         'run_time': 1.0,
                         'ess_trsh_methods': [],
                         'trsh': '',
                         'initial_trsh': {},
                         'fine': True,
                         'job_id': 0,
                         'job_name': 'opt_a100',
                         'job_num': 100,
                         'job_server_name': 'a100',
                         'job_status': ['initializing', 'initializing'],
                         'job_type': 'opt',
                         'level_of_theory': 'b3lyp/6-31+g(d)',
                         'memory': 1500,
                         'occ': None,
                         'pivots': [],
                         'project_directory': os.path.join(arc_path, 'Projects', 'project_test'),
                         'scan': '',
                         'server': None,
                         'shift': '',
                         'max_job_time': 120,
                         'comments': '',
                         'scan_res': 8.0,
                         'scan_trsh': '',
                         'software': 'gaussian',
                         'xyz': 'C 0.0 0.0 0.0'}
        self.assertEqual(job_dict, expected_dict)


################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
