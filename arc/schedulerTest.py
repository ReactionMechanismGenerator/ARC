#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the arc.scheduler module
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest
import os
import shutil

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
        cls.ess_settings = {'gaussian': ['server1'], 'molpro': ['server2', 'server1'], 'qchem': ['server1']}
        cls.project_directory = os.path.join(arc_path, 'Projects', 'arc_project_for_testing_delete_after_usage3')
        cls.spc1 = ARCSpecies(label=str('methylamine'), smiles=str('CN'))
        cls.spc2 = ARCSpecies(label=str('C2H6'), smiles=str('CC'))
        cls.job1 = Job(project='project_test', ess_settings=cls.ess_settings, species_name='methylamine',
                       xyz='C 0.0 0.0 0.0', job_type='conformer', conformer=0, level_of_theory='b97-d3/6-311+g(d,p)',
                       multiplicity=1, project_directory=cls.project_directory, job_num=101)
        cls.job2 = Job(project='project_test', ess_settings=cls.ess_settings, species_name='methylamine',
                       xyz='C 0.0 0.0 0.0', job_type='conformer', conformer=1, level_of_theory='b97-d3/6-311+g(d,p)',
                       multiplicity=1, project_directory=cls.project_directory, job_num=102)
        cls.job3 = Job(project='project_test', ess_settings=cls.ess_settings, species_name='C2H6', xyz='C 0.0 0.0 0.0',
                       job_type='freq', level_of_theory='wb97x-d3/6-311+g(d,p)', multiplicity=1,
                       project_directory=cls.project_directory, software='qchem', job_num=103)
        cls.rmgdb = rmgdb.make_rmg_database_object()
        cls.job_types1 = {'conformers': True,
                          'opt': True,
                          'fine_grid': False,
                          'freq': True,
                          'sp': True,
                          '1d_rotors': False,
                          'orbitals': False,
                          'lennard_jones': False,
                          }
        cls.sched1 = Scheduler(project='project_test', ess_settings=cls.ess_settings, species_list=[cls.spc1, cls.spc2],
                               composite_method='', conformer_level=default_levels_of_theory['conformer'],
                               opt_level=default_levels_of_theory['opt'], freq_level=default_levels_of_theory['freq'],
                               sp_level=default_levels_of_theory['sp'], scan_level=default_levels_of_theory['scan'],
                               ts_guess_level=default_levels_of_theory['ts_guesses'], rmgdatabase=cls.rmgdb,
                               project_directory=cls.project_directory, testing=True, job_types=cls.job_types1,
                               orbitals_level=default_levels_of_theory['orbitals'], adaptive_levels=None)

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
        self.sched1.species_dict[label].conformer_energies = [None, None]
        self.sched1.species_dict[label].conformers = [None, None]
        self.sched1.parse_conformer_energy(job=self.job1, label=label, i=0)
        self.sched1.parse_conformer_energy(job=self.job2, label=label, i=1)
        expecting = [-251596.4435088726, -254221.9433698632]
        self.assertEqual(self.sched1.species_dict[label].conformer_energies, expecting)
        self.sched1.species_dict[label].conformers[0] = parser.parse_xyz_from_file(self.job1.local_path_to_output_file)
        self.sched1.species_dict[label].conformers[1] = parser.parse_xyz_from_file(self.job2.local_path_to_output_file)

        self.sched1.determine_most_stable_conformer(label=label)
        expecting = """N      -0.75555952   -0.12937106    0.00000000
C       0.70855440    0.03887206    0.00000000
H       1.06395135    1.08711266    0.00000000
H       1.12732348   -0.45978507    0.88433277
H       1.12732348   -0.45978507   -0.88433277
H      -1.16566701    0.32023496    0.81630508
H      -1.16566701    0.32023496   -0.81630508"""
        self.assertEqual(self.sched1.species_dict[label].initial_xyz, expecting)
        methylamine_conf_path = os.path.join(self.sched1.project_directory, 'output', 'Species', 'methylamine',
                                             'geometry', 'conformers_after_optimization.txt')
        self.assertTrue(os.path.isfile(methylamine_conf_path))
        with open(methylamine_conf_path, 'r') as f:
            lines = f.readlines()
        self.assertTrue('conformers optimized at' in lines[0])
        self.assertEqual(lines[10], 'SMILES: CN\n')
        self.assertTrue('Relative Energy:' in lines[11])
        self.assertEqual(lines[15][0], 'N')

        self.sched1.output['C2H6'] = {'status': ''}  # otherwise confs won't be generated due to the presence of 'geo'
        self.sched1.run_conformer_jobs()
        self.sched1.save_conformers_file(label='C2H6')
        c2h6_conf_path = os.path.join(self.sched1.project_directory, 'output', 'Species', 'C2H6', 'geometry',
                                      'conformers_before_optimization.txt')
        self.assertTrue(os.path.isfile(c2h6_conf_path))
        with open(c2h6_conf_path, 'r') as f:
            lines = f.readlines()
        self.assertEqual(lines[1][0], 'C')
        self.assertEqual(lines[9], '\n')
        self.assertEqual(lines[10], 'SMILES: CC\n')

    def test_check_negative_freq(self):
        """Test the check_negative_freq() method"""
        label = 'C2H6'
        self.job3.local_path_to_output_file = os.path.join(arc_path, 'arc', 'testing', 'C2H6_freq_Qchem.out')
        self.job3.job_status = ['done', 'done']
        vibfreqs = parser.parse_frequencies(path=str(self.job3.local_path_to_output_file), software=self.job3.software)
        self.assertTrue(self.sched1.check_negative_freq(label=label, job=self.job3, vibfreqs=vibfreqs))

    def test_determine_adaptive_level(self):
        """Test the determine_adaptive_level() method"""
        adaptive_levels = {(1, 5):      {'optfreq': 'wb97xd/6-311+g(2d,2p)',
                                         'sp': 'ccsd(t)-f12/aug-cc-pvtz-f12'},
                           (6, 15):     {'optfreq': 'b3lyp/cbsb7',
                                         'sp': 'dlpno-ccsd(t)/def2-tzvp/c'},
                           (16, 30):    {'optfreq': 'b3lyp/6-31g(d,p)',
                                         'sp': 'wb97xd/6-311+g(2d,2p)'},
                           (31, 'inf'): {'optfreq': 'b3lyp/6-31g(d,p)',
                                         'sp': 'b3lyp/6-311+g(d,p)'}}

        sched2 = Scheduler(project='project_test', ess_settings=self.ess_settings, species_list=[self.spc1, self.spc2],
                           composite_method='', conformer_level=default_levels_of_theory['conformer'],
                           opt_level=default_levels_of_theory['opt'], freq_level=default_levels_of_theory['freq'],
                           sp_level=default_levels_of_theory['sp'], scan_level=default_levels_of_theory['scan'],
                           ts_guess_level=default_levels_of_theory['ts_guesses'], rmgdatabase=self.rmgdb,
                           project_directory=self.project_directory, testing=True, job_types=self.job_types1,
                           orbitals_level=default_levels_of_theory['orbitals'], adaptive_levels=adaptive_levels)
        level1 = sched2.determine_adaptive_level(original_level_of_theory='some_original_level',
                                                 job_type='opt', heavy_atoms=5)
        level2 = sched2.determine_adaptive_level(original_level_of_theory='some_original_level',
                                                 job_type='freq', heavy_atoms=5)
        level3 = sched2.determine_adaptive_level(original_level_of_theory='some_original_level',
                                                 job_type='opt', heavy_atoms=20)
        level4 = sched2.determine_adaptive_level(original_level_of_theory='some_original_level',
                                                 job_type='composite', heavy_atoms=50)
        level5 = sched2.determine_adaptive_level(original_level_of_theory='some_original_level',
                                                 job_type='orbitals', heavy_atoms=5)
        level6 = sched2.determine_adaptive_level(original_level_of_theory='some_original_level',
                                                 job_type='sp', heavy_atoms=7)
        level7 = sched2.determine_adaptive_level(original_level_of_theory='some_original_level',
                                                 job_type='sp', heavy_atoms=25)
        self.assertEqual(level1, 'wb97xd/6-311+g(2d,2p)')
        self.assertEqual(level2, 'wb97xd/6-311+g(2d,2p)')
        self.assertEqual(level3, 'b3lyp/6-31g(d,p)')
        self.assertEqual(level4, 'b3lyp/6-31g(d,p)')
        self.assertEqual(level5, 'some_original_level')
        self.assertEqual(level6, 'dlpno-ccsd(t)/def2-tzvp/c')
        self.assertEqual(level7, 'wb97xd/6-311+g(2d,2p)')

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        projects = ['arc_project_for_testing_delete_after_usage3']
        for project in projects:
            project_directory = os.path.join(arc_path, 'Projects', project)
            shutil.rmtree(project_directory)


################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
