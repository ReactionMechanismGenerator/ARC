#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.scheduler module
"""

import unittest
import os
import shutil

import arc.rmgdb as rmgdb
import arc.parser as parser
from arc.common import ARC_PATH, almost_equal_coords_lists
from arc.job.factory import job_factory
from arc.level import Level
from arc.plotter import save_conformers_file
from arc.scheduler import Scheduler
from arc.imports import settings
from arc.species.species import ARCSpecies


default_levels_of_theory = settings['default_levels_of_theory']


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
        cls.project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_for_testing_delete_after_usage3')
        cls.spc1 = ARCSpecies(label='methylamine', smiles='CN')
        cls.spc2 = ARCSpecies(label='C2H6', smiles='CC')
        xyz3 = """C       1.11424367   -0.01231165   -0.11493630
C      -0.07257945   -0.17830906   -0.16010022
O      -1.38500471   -0.36381519   -0.20928090
H       2.16904830    0.12689206   -0.07152274
H      -1.82570782    0.42754384   -0.56130718"""
        cls.spc3 = ARCSpecies(label='CtripCO', smiles='C#CO', xyz=xyz3)
        xyz2 = {'symbols': ('C',), 'isotopes': (12,), 'coords': ((0.0, 0.0, 0.0),)}
        cls.job1 = job_factory(job_adapter='gaussian', project='project_test', ess_settings=cls.ess_settings,
                               species=[cls.spc1], xyz=xyz2, job_type='conformers',
                               conformer=0, level=Level(repr={'method': 'b97-d3', 'basis': '6-311+g(d,p)'}),
                               project_directory=cls.project_directory, job_num=101)
        cls.job2 = job_factory(job_adapter='gaussian', project='project_test', ess_settings=cls.ess_settings,
                               species=[cls.spc1], xyz=xyz2, job_type='conformers',
                               conformer=1, level=Level(repr={'method': 'b97-d3', 'basis': '6-311+g(d,p)'}),
                               project_directory=cls.project_directory, job_num=102)
        cls.job3 = job_factory(job_adapter='qchem', project='project_test', ess_settings=cls.ess_settings,
                               species=[cls.spc2], job_type='freq',
                               level=Level(repr={'method': 'wb97x-d3', 'basis': '6-311+g(d,p)'}),
                               project_directory=cls.project_directory, job_num=103)
        cls.rmg_database = rmgdb.make_rmg_database_object()
        cls.job_types1 = {'conformers': True,
                          'opt': True,
                          'fine_grid': False,
                          'freq': True,
                          'sp': True,
                          'rotors': False,
                          'orbitals': False,
                          'lennard_jones': False,
                          }
        cls.sched1 = Scheduler(project='project_test', ess_settings=cls.ess_settings,
                               species_list=[cls.spc1, cls.spc2, cls.spc3],
                               composite_method=None,
                               conformer_level=Level(repr=default_levels_of_theory['conformer']),
                               opt_level=Level(repr=default_levels_of_theory['opt']),
                               freq_level=Level(repr=default_levels_of_theory['freq']),
                               sp_level=Level(repr=default_levels_of_theory['sp']),
                               scan_level=Level(repr=default_levels_of_theory['scan']),
                               ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                               rmg_database=cls.rmg_database,
                               project_directory=cls.project_directory,
                               testing=True,
                               job_types=cls.job_types1,
                               orbitals_level=default_levels_of_theory['orbitals'],
                               adaptive_levels=None,
                               )

    def test_conformers(self):
        """Test the parse_conformer_energy() and determine_most_stable_conformer() methods"""
        label = 'methylamine'
        self.job1.local_path_to_output_file = os.path.join(ARC_PATH, 'arc', 'testing', 'methylamine_conformer_0.out')
        self.job1.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]
        self.job2.local_path_to_output_file = os.path.join(ARC_PATH, 'arc', 'testing', 'methylamine_conformer_1.out')
        self.job2.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]
        self.sched1.job_dict[label] = dict()
        self.sched1.job_dict[label]['conformers'] = dict()
        self.sched1.job_dict[label]['conformers'][0] = self.job1
        self.sched1.job_dict[label]['conformers'][1] = self.job2
        self.sched1.species_dict[label].conformer_energies = [None, None]
        self.sched1.species_dict[label].conformers = [None, None]
        self.sched1.parse_conformer(job=self.job1, label=label, i=0)
        self.sched1.parse_conformer(job=self.job2, label=label, i=1)
        expecting = [-251596.4435088726, -254221.9433698632]
        self.assertAlmostEqual(self.sched1.species_dict[label].conformer_energies[0], expecting[0], 5)
        self.assertAlmostEqual(self.sched1.species_dict[label].conformer_energies[1], expecting[1], 5)
        self.sched1.species_dict[label].conformers[0] = parser.parse_xyz_from_file(self.job1.local_path_to_output_file)
        self.sched1.species_dict[label].conformers[1] = parser.parse_xyz_from_file(self.job2.local_path_to_output_file)

        self.sched1.determine_most_stable_conformer(label=label)
        expecting = {'symbols': ('N', 'C', 'H', 'H', 'H', 'H', 'H'), 'isotopes': (14, 12, 1, 1, 1, 1, 1),
                     'coords': ((-0.75555952, -0.12937106, 0.0), (0.7085544, 0.03887206, 0.0),
                                (1.06395135, 1.08711266, 0.0), (1.12732348, -0.45978507, 0.88433277),
                                (1.12732348, -0.45978507, -0.88433277), (-1.16566701, 0.32023496, 0.81630508),
                                (-1.16566701, 0.32023496, -0.81630508))}
        self.assertTrue(almost_equal_coords_lists(self.sched1.species_dict[label].initial_xyz, expecting))
        methylamine_conf_path = os.path.join(self.sched1.project_directory, 'output', 'Species', 'methylamine',
                                             'geometry', 'conformers', 'conformers_after_optimization.txt')
        self.assertTrue(os.path.isfile(methylamine_conf_path))
        with open(methylamine_conf_path, 'r') as f:
            lines = f.readlines()
        self.assertTrue('Conformers for methylamine, optimized at the wb97xd/def2svp level' in lines[0])
        self.assertEqual(lines[11], 'SMILES: CN\n')
        self.assertTrue('Relative Energy:' in lines[12])
        self.assertEqual(lines[16][0], 'N')

        self.sched1.output['C2H6'] = {'info': '',
                                      'paths': {'composite': '', 'freq': '', 'geo': ''},
                                      'isomorphism': '',
                                      'warnings': '',
                                      'errors': '',
                                      'job_types': {'opt': False, 'composite': False, 'sp': False, 'fine_grid': False,
                                                    'freq': False, 'conformers': False},
                                      'convergence': True, 'conformers': '', 'restart': ''}
        self.sched1.run_conformer_jobs()
        save_conformers_file(project_directory=self.sched1.project_directory,
                             label='C2H6',
                             xyzs=self.sched1.species_dict['C2H6'].conformers,
                             level_of_theory=Level(method='CBS-QB3'),
                             multiplicity=1,
                             charge=0)
        c2h6_conf_path = os.path.join(self.sched1.project_directory, 'output', 'Species', 'C2H6', 'geometry',
                                      'conformers', 'conformers_before_optimization.txt')
        self.assertTrue(os.path.isfile(c2h6_conf_path))
        with open(c2h6_conf_path, 'r') as f:
            lines = f.readlines()
        self.assertEqual(lines[0], 'conformer 0:\n')
        self.assertEqual(lines[1][0], 'C')
        self.assertEqual(lines[9], '\n')
        self.assertEqual(lines[10], 'SMILES: CC\n')

    def test_check_negative_freq(self):
        """Test the check_negative_freq() method"""
        label = 'C2H6'
        self.job3.local_path_to_output_file = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'C2H6_freq_QChem.out')
        self.job3.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]
        vibfreqs = parser.parse_frequencies(path=self.job3.local_path_to_output_file, software=self.job3.job_adapter)
        self.assertTrue(self.sched1.check_negative_freq(label=label, job=self.job3, vibfreqs=vibfreqs))

    def test_determine_adaptive_level(self):
        """Test the determine_adaptive_level() method"""
        # adaptive_levels get converted to ``Level`` objects in main, but here we skip main and test Scheduler directly
        adaptive_levels = {(1, 5):      {('opt', 'freq'): Level(repr='wb97xd/6-311+g(2d,2p)'),
                                         ('sp',): Level(repr='ccsd(t)-f12/aug-cc-pvtz-f12')},
                           (6, 15):     {('opt', 'freq'): Level(repr='b3lyp/cbsb7'),
                                         ('sp',): Level(repr='dlpno-ccsd(t)/def2-tzvp')},
                           (16, 30):    {('opt', 'freq'): Level(repr='b3lyp/6-31g(d,p)'),
                                         ('sp',): Level(repr='wb97xd/6-311+g(2d,2p)')},
                           (31, 'inf'): {('opt', 'freq'): Level(repr='b3lyp/6-31g(d,p)'),
                                         ('sp',): Level(repr='b3lyp/6-311+g(d,p)')}}

        sched2 = Scheduler(project='project_test',
                           ess_settings=self.ess_settings,
                           species_list=[self.spc1, self.spc2],
                           composite_method=None,
                           conformer_level=default_levels_of_theory['conformer'],
                           opt_level=default_levels_of_theory['opt'],
                           freq_level=default_levels_of_theory['freq'],
                           sp_level=default_levels_of_theory['sp'],
                           scan_level=default_levels_of_theory['scan'],
                           ts_guess_level=default_levels_of_theory['ts_guesses'],
                           rmg_database=self.rmg_database,
                           project_directory=self.project_directory,
                           testing=True, job_types=self.job_types1,
                           orbitals_level=default_levels_of_theory['orbitals'],
                           adaptive_levels=adaptive_levels)
        original_level = Level(method='CBS-QB3')
        level1 = sched2.determine_adaptive_level(original_level_of_theory=original_level,
                                                 job_type='opt',
                                                 heavy_atoms=5)
        level2 = sched2.determine_adaptive_level(original_level_of_theory=original_level,
                                                 job_type='freq',
                                                 heavy_atoms=5)
        level3 = sched2.determine_adaptive_level(original_level_of_theory=original_level,
                                                 job_type='opt',
                                                 heavy_atoms=20)
        level4 = sched2.determine_adaptive_level(original_level_of_theory=original_level,
                                                 job_type='composite',
                                                 heavy_atoms=50)
        level5 = sched2.determine_adaptive_level(original_level_of_theory=original_level,
                                                 job_type='orbitals',
                                                 heavy_atoms=5)
        level6 = sched2.determine_adaptive_level(original_level_of_theory=original_level,
                                                 job_type='sp',
                                                 heavy_atoms=7)
        level7 = sched2.determine_adaptive_level(original_level_of_theory=original_level,
                                                 job_type='sp',
                                                 heavy_atoms=25)
        self.assertEqual(level1.simple(), 'wb97xd/6-311+g(2d,2p)')
        self.assertEqual(level2.simple(), 'wb97xd/6-311+g(2d,2p)')
        self.assertEqual(level3.simple(), 'b3lyp/6-31g(d,p)')
        self.assertEqual(level4.simple(), 'cbs-qb3')
        self.assertEqual(level5.simple(), 'cbs-qb3')
        self.assertEqual(level6.simple(), 'dlpno-ccsd(t)/def2-tzvp')
        self.assertEqual(level7.simple(), 'wb97xd/6-311+g(2d,2p)')

    def test_initialize_output_dict(self):
        """Test Scheduler.initialize_output_dict"""
        self.assertTrue(self.sched1._does_output_dict_contain_info())
        self.sched1.output = dict()
        self.assertEqual(self.sched1.output, dict())
        self.sched1.initialize_output_dict()
        self.assertFalse(self.sched1._does_output_dict_contain_info())
        empty_species_dict = {'conformers': '',
                              'convergence': None,
                              'errors': '',
                              'info': '',
                              'isomorphism': '',
                              'job_types': {'rotors': True,
                                            'composite': False,
                                            'conformers': False,
                                            'fine_grid': False,
                                            'freq': False,
                                            'lennard_jones': False,
                                            'onedmin': False,
                                            'opt': False,
                                            'orbitals': False,
                                            'sp': False},
                              'paths': {'composite': '', 'freq': '', 'geo': '', 'sp': ''},
                              'restart': '', 'warnings': ''}
        initialized_output_dict = {'C2H6': empty_species_dict,
                                   'CtripCO': empty_species_dict,
                                   'methylamine': empty_species_dict,
                                   }
        self.assertEqual(self.sched1.output, initialized_output_dict)

    def test_does_output_dict_contain_info(self):
        """Test Scheduler.does_output_dict_contain_info"""
        self.sched1.output = dict()
        self.sched1.initialize_output_dict()
        self.assertFalse(self.sched1._does_output_dict_contain_info())

        self.sched1.output['C2H6']['info'] = 'some text'
        self.sched1.output['C2H6']['job_types']['freq'] = True
        self.sched1.output['C2H6']['paths']['sp'] = 'some/path/out.out'
        self.assertTrue(self.sched1._does_output_dict_contain_info())

    def test_non_rotor(self):
        """Test that a 180 degree angle on either side of a torsion is not considered as a rotor"""
        self.sched1.species_dict['CtripCO'].rotors_dict = {
            0: {'torsion': [1, 2, 3, 4], 'top': [3, 5], 'scan': [1, 2, 3, 5], 'number_of_running_jobs': 0,
                'success': None, 'invalidation_reason': '', 'times_dihedral_set': 0, 'trsh_methods': [], 'scan_path': '',
                'directed_scan_type': '', 'directed_scan': {}, 'dimensions': 1, 'original_dihedrals': [],
                'cont_indices': []}}
        self.sched1.species_dict['CtripCO'].number_of_rotors = 1
        self.sched1.job_types['rotors'] = True
        self.sched1.run_scan_jobs(label='CtripCO')
        self.assertEqual(self.sched1.species_dict['CtripCO'].rotors_dict[0]['invalidation_reason'],
                         'not a torsional mode (angles = 0.20, 13.03 degrees)')
        self.assertFalse(self.sched1.species_dict['CtripCO'].rotors_dict[0]['success'])

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        projects = ['arc_project_for_testing_delete_after_usage3']
        for project in projects:
            project_directory = os.path.join(ARC_PATH, 'Projects', project)
            shutil.rmtree(project_directory, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
