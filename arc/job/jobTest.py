#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests of the arc.job.job module
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest
import os

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

    def test_as_dict(self):
        """Test Job.as_dict()"""
        job_dict = self.job1.as_dict()
        dt = job_dict['date_time']
        expected_dict = {u'date_time': dt,
                         u'ess_trsh_methods': [],
                         u'fine': True,
                         u'job_id': 0,
                         u'job_name': u'opt_a100',
                         u'job_num': 100,
                         u'job_server_name': u'a100',
                         u'job_status': [u'initializing', u'initializing'],
                         u'job_type': u'opt',
                         u'level_of_theory': u'b3lyp/6-31+g(d)',
                         u'memory': 1500,
                         u'occ': None,
                         u'pivots': [],
                         u'project_directory': os.path.join(arc_path, 'Projects', 'project_test'),
                         u'run_time': u'',
                         u'scan': u'',
                         u'server': None,
                         u'shift': u'',
                         u'trsh': u'',
                         u'xyz': u'C 0.0 0.0 0.0'}
        self.assertEqual(job_dict, expected_dict)


################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
