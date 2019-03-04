#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the arc.scheduler module
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest
import os

import arc.rmgdb as rmgdb
from arc.scheduler import Scheduler
from arc.job.job import Job
from arc.species.species import ARCSpecies
import arc.parser as parser
from arc.settings import arc_path, default_levels_of_theory

################################################################################


class TestScheduler(unittest.TestCase):
    """
    Contains unit tests for the Scheduler class
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        settings = {'gaussian': 'server1', 'molpro': 'server2', 'qchem': 'server1', 'ssh': False}
        project_directory = os.path.join(arc_path, 'Projects', 'project_test')
        cls.spc1 = ARCSpecies(label=str('methylamine'), smiles=str('CN'))
        cls.spc2 = ARCSpecies(label=str('C2H6'), smiles=str('CC'))
        cls.job1 = Job(project='project_test', settings=settings, species_name='methylamine', xyz='C 0.0 0.0 0.0',
                       job_type='conformer', conformer=0, level_of_theory='b97-d3/6-311+g(d,p)', multiplicity=1,
                       project_directory=project_directory, job_num=101)
        cls.job2 = Job(project='project_test', settings=settings, species_name='methylamine', xyz='C 0.0 0.0 0.0',
                       job_type='conformer', conformer=1, level_of_theory='b97-d3/6-311+g(d,p)', multiplicity=1,
                       project_directory=project_directory, job_num=102)
        cls.job3 = Job(project='project_test', settings=settings, species_name='C2H6', xyz='C 0.0 0.0 0.0',
                       job_type='freq', level_of_theory='wb97x-d3/6-311+g(d,p)', multiplicity=1,
                       project_directory=project_directory, software='qchem', job_num=103)
        cls.rmgdb = rmgdb.make_rmg_database_object()
        cls.sched1 = Scheduler(project='project_test', settings=settings, species_list=[cls.spc1, cls.spc2],
                               composite_method='', conformer_level=default_levels_of_theory['conformer'],
                               opt_level=default_levels_of_theory['opt'], freq_level=default_levels_of_theory['freq'],
                               sp_level=default_levels_of_theory['sp'], scan_level=default_levels_of_theory['scan'],
                               ts_guess_level=default_levels_of_theory['ts_guesses'], rmgdatabase=cls.rmgdb,
                               project_directory=project_directory, generate_conformers=True, testing=True)

    def test_conformers(self):
        """Test the parse_conformer_energy() and determine_most_stable_conformer() methods"""
        label = 'methylamine'
        self.job1.local_path_to_output_file = os.path.join(arc_path, 'arc', 'testing', 'methylamine_conformer_0.out')
        self.job1.job_status = ['done', 'done']
        self.job2.local_path_to_output_file = os.path.join(arc_path, 'arc', 'testing', 'methylamine_conformer_1.out')
        self.job2.job_status = ['done', 'done']
        self.sched1.job_dict[label] = dict()
        self.sched1.job_dict[label]['conformers'] = dict()
        self.sched1.job_dict[label]['conformers'][0] = self.job1
        self.sched1.job_dict[label]['conformers'][1] = self.job2
        self.sched1.parse_conformer_energy(job=self.job1, label=label, i=0)
        self.sched1.parse_conformer_energy(job=self.job1, label=label, i=1)
        expecting = [-251596443.5088726, -251596443.5088726]
        self.assertEqual(self.sched1.species_dict[label].conformer_energies, expecting)

        self.sched1.determine_most_stable_conformer(label=label)
        expecting = """N      -0.75556320   -0.12937340    0.00000000
C       0.70855047    0.03886715    0.00000000
H       1.12733096   -0.45978028   -0.88432893
H       1.12733096   -0.45978028    0.88432893
H       1.06394036    1.08710265    0.00000000
H      -1.16566523    0.32024044    0.81630398
H      -1.16566523    0.32024044   -0.81630398
"""
        self.assertEqual(self.sched1.species_dict[label].initial_xyz, expecting)

    def test_check_negative_freq(self):
        """Test the check_negative_freq() method"""
        label='C2H6'
        self.job3.local_path_to_output_file = os.path.join(arc_path, 'arc', 'testing', 'C2H6_freq_Qchem.out')
        self.job3.job_status = ['done', 'done']
        vibfreqs = parser.parse_frequencies(path=str(self.job3.local_path_to_output_file), software=self.job3.software)
        self.assertTrue(self.sched1.check_negative_freq(label=label, job=self.job3, vibfreqs=vibfreqs))

################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
