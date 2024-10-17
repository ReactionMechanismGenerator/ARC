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
from arc.checks.ts import check_ts
from arc.common import ARC_PATH, almost_equal_coords_lists, initialize_job_types, read_yaml_file
from arc.job.factory import job_factory
from arc.level import Level
from arc.plotter import save_conformers_file
from arc.scheduler import Scheduler, species_has_freq, species_has_geo, species_has_sp, species_has_sp_and_freq
from arc.imports import settings
from arc.reaction import ARCReaction
from arc.species.converter import str_to_xyz
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
        xyz1 = str_to_xyz("""C      -0.57422867   -0.01669771    0.01229213
N       0.82084044    0.08279104   -0.37769346
H      -1.05737005   -0.84067772   -0.52007494
H      -1.10211468    0.90879867   -0.23383011
H      -0.66133128   -0.19490562    1.08785111
H       0.88047852    0.26966160   -1.37780789
H       1.27889520   -0.81548721   -0.22940984""")
        cls.spc1 = ARCSpecies(label='methylamine', smiles='CN', xyz=xyz1)
        cls.spc2 = ARCSpecies(label='C2H6', smiles='CC')
        xyz3 = """C       1.11424367   -0.01231165   -0.11493630
C      -0.07257945   -0.17830906   -0.16010022
O      -1.38500471   -0.36381519   -0.20928090
H       2.16904830    0.12689206   -0.07152274
H      -1.82570782    0.42754384   -0.56130718"""
        cls.spc3 = ARCSpecies(label='CtripCO', smiles='C#CO', xyz=xyz3)
        cls.job1 = job_factory(job_adapter='gaussian', project='project_test', ess_settings=cls.ess_settings,
                               species=[cls.spc1], xyz=xyz1, job_type='conf_opt',
                               conformer=0, level=Level(repr={'method': 'b97-d3', 'basis': '6-311+g(d,p)'}),
                               project_directory=cls.project_directory, job_num=101)
        cls.job2 = job_factory(job_adapter='gaussian', project='project_test', ess_settings=cls.ess_settings,
                               species=[cls.spc1], xyz=xyz1, job_type='conf_opt',
                               conformer=1, level=Level(repr={'method': 'b97-d3', 'basis': '6-311+g(d,p)'}),
                               project_directory=cls.project_directory, job_num=102)
        cls.job3 = job_factory(job_adapter='qchem', project='project_test', ess_settings=cls.ess_settings,
                               species=[cls.spc2], job_type='freq',
                               level=Level(repr={'method': 'wb97x-d3', 'basis': '6-311+g(d,p)'}),
                               project_directory=cls.project_directory, job_num=103)
        cls.job4 = job_factory(job_adapter='gaussian', project='project_test_4', ess_settings=cls.ess_settings,
                               species=[cls.spc1], xyz=xyz1, job_type='scan', torsions=[[3, 1, 2, 6]], rotor_index=0,
                               level=Level(repr={'method': 'b3lyp', 'basis': 'cbsb7'}),
                               project_directory=cls.project_directory, job_num=104)
        cls.rmg_database = rmgdb.make_rmg_database_object()
        cls.job_types1 = {'conf_opt': True,
                          'conf_sp': False,
                          'opt': True,
                          'fine': False,
                          'freq': True,
                          'sp': True,
                          'rotors': False,
                          'orbitals': False,
                          'lennard_jones': False,
                          }
        cls.job_types2 = {'conf_opt': True,
                          'conf_sp': False,
                          'opt': True,
                          'fine': False,
                          'freq': True,
                          'sp': True,
                          'rotors': True,
                          }
        cls.sched1 = Scheduler(project='project_test_1', ess_settings=cls.ess_settings,
                               species_list=[cls.spc1, cls.spc2, cls.spc3],
                               composite_method=None,
                               conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
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
        cls.sched2 = Scheduler(project='project_test_2', ess_settings=cls.ess_settings,
                               species_list=[cls.spc1, cls.spc2, cls.spc3],
                               composite_method=None,
                               conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
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
        cls.sched3 = Scheduler(project='project_test_4', ess_settings=cls.ess_settings,
                               species_list=[cls.spc1],
                               composite_method=Level(repr='CBS-QB3'),
                               conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                               opt_level=Level(repr=default_levels_of_theory['freq_for_composite']),
                               freq_level=Level(repr=default_levels_of_theory['freq_for_composite']),
                               scan_level=Level(repr=default_levels_of_theory['scan_for_composite']),
                               ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                               rmg_database=cls.rmg_database,
                               project_directory=cls.project_directory,
                               testing=True,
                               job_types=cls.job_types2,
                               )

    def test_conformers(self):
        """Test the parse_conformer_energy() and determine_most_stable_conformer() methods"""
        label = 'methylamine'
        self.job1.local_path_to_output_file = os.path.join(ARC_PATH, 'arc', 'testing', 'methylamine_conformer_0.out')
        self.job1.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]
        self.job2.local_path_to_output_file = os.path.join(ARC_PATH, 'arc', 'testing', 'methylamine_conformer_1.out')
        self.job2.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]
        self.sched1.job_dict[label] = dict()
        self.sched1.job_dict[label]['conf_opt'] = dict()
        self.sched1.job_dict[label]['conf_opt'][0] = self.job1
        self.sched1.job_dict[label]['conf_opt'][1] = self.job2
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
                                      'job_types': {'opt': False, 'composite': False, 'sp': False, 'fine': False,
                                                    'freq': False, 'conf_opt': False, 'conf_sp': False},
                                      'convergence': False, 'conformers': '', 'restart': ''}
        self.sched1.run_conformer_jobs()
        save_conformers_file(project_directory=self.sched1.project_directory,
                             label='C2H6',
                             xyzs=self.sched1.species_dict['C2H6'].conformers,
                             level_of_theory=Level(method='CBS-QB3'),
                             multiplicity=1,
                             charge=0,
                             before_optimization=True,)
        c2h6_conf_path = os.path.join(self.sched1.project_directory, 'output', 'Species', 'C2H6', 'geometry',
                                      'conformers', 'conformers_before_optimization.txt')
        self.assertTrue(os.path.isfile(c2h6_conf_path))
        with open(c2h6_conf_path, 'r') as f:
            lines = f.readlines()
        self.assertEqual(lines[0], 'Conformers for C2H6, computed using a force field:\n')
        self.assertEqual(lines[2], 'conformer 0:\n')
        self.assertEqual(lines[3][0], 'C')
        self.assertEqual(lines[11], '\n')
        self.assertEqual(lines[12], 'SMILES: CC\n')

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
                           conformer_opt_level=default_levels_of_theory['conformer'],
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
                                            'conf_opt': False,
                                            'conf_sp': False,
                                            'fine': False,
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
        """Test that a 180 degree angle on either side of a torsion is not considered as a rotor."""
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

    def test_deduce_job_adapter(self):
        """Test the deduce_job_adapter() method."""
        level_1 = Level(method='CBS-QB3')
        job_type_1 = 'composite'
        job_adapter_1 = self.sched1.deduce_job_adapter(level=level_1, job_type=job_type_1)
        self.assertEqual(job_adapter_1, 'gaussian')

        level_2 = Level(repr='dlpno-ccsd(t)/def2-svp')
        job_type_2 = 'sp'
        job_adapter_2 = self.sched1.deduce_job_adapter(level=level_2, job_type=job_type_2)
        self.assertEqual(job_adapter_2, 'orca')

        level_3 = Level(repr='ccsd(t)/cc-pvtz')
        job_type_3 = 'sp'
        job_adapter_3 = self.sched1.deduce_job_adapter(level=level_3, job_type=job_type_3)
        self.assertEqual(job_adapter_3, 'molpro')

        level_4 = Level(repr='m06-2x/def2-svp')
        job_type_4 = 'freq'
        job_adapter_4 = self.sched1.deduce_job_adapter(level=level_4, job_type=job_type_4)
        self.assertEqual(job_adapter_4, 'qchem')

        level_5 = Level(repr='pbe/def2-svp')
        job_type_5 = 'freq'
        job_adapter_5 = self.sched1.deduce_job_adapter(level=level_5, job_type=job_type_5)
        self.assertEqual(job_adapter_5, 'terachem')

    def test_check_scan_job(self):
        """Test the check_scan_job() method."""
        self.job4.job_status[1]['status'] = 'done'
        self.job4.local_path_to_output_file = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'N2O3.out')
        self.sched3.check_scan_job(label='methylamine', job=self.job4)
        self.assertTrue(self.sched3.species_dict['methylamine'].rotors_dict[self.job4.rotor_index]['success'])

        self.job4.local_path_to_output_file = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'l103_err.out')
        self.job4.job_status[1]['status'] = 'errored'
        self.job4.job_status[1]['error'] = 'Internal coordinate error'
        self.sched3.check_scan_job(label='methylamine', job=self.job4)
        self.assertFalse(self.sched3.species_dict['methylamine'].rotors_dict[self.job4.rotor_index]['success'])
        self.assertIn('Internal coordinate error', self.sched3.species_dict['methylamine'].rotors_dict[self.job4.rotor_index]['invalidation_reason'])

    def test_check_rxn_e0_by_spc(self):
        """Test the check_rxn_e0_by_spc() method."""
        rxn_dict = \
            {'label': 'nC3H7 <=> iC3H7', 'index': 0, 'multiplicity': 2, 'charge': 0, 'reactants': ['nC3H7'], 'products': ['iC3H7'],
             'r_species': [{'force_field': 'MMFF94s', 'is_ts': False, 'label': 'nC3H7',
                            'long_thermo_description': "Bond corrections: {'C-C': 2, 'C-H': 7}\n", 'multiplicity': 2,
                            'charge': 0, 'compute_thermo': True, 'number_of_rotors': 0, 'arkane_file': None,
                            'consider_all_diastereomers': True, 'e_elect': -311073.1524474179, 'run_time': 64.0,
                            'opt_level': 'b3lyp/6-31g(d,p)', 'conf_is_isomorphic': True,
                            'bond_corrections': {'C-C': 2, 'C-H': 7}, 'mol': {'atoms': [
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 1, 'charge': 0, 'label': '*1',
                      'lone_pairs': 0, 'id': -26782, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-26781: 1.0, -26779: 1.0, -26778: 1.0}},
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '*2',
                      'lone_pairs': 0, 'id': -26781, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-26782: 1.0, -26780: 1.0, -26777: 1.0, -26776: 1.0}},
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26780, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-26781: 1.0, -26775: 1.0, -26774: 1.0, -26773: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26779, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26782: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26778, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26782: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26777, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26781: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '*3',
                      'lone_pairs': 0, 'id': -26776, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26781: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26775, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26780: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26774, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26780: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26773, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26780: 1.0}}], 'multiplicity': 2, 'props': {}, 'atom_order': [-26782, -26781, -26780, -26779, -26778, -26777, -26776, -26775, -26774, -26773]},
                            'initial_xyz': 'C       1.37804355    0.27791700   -0.19511840\nC       0.17557158   -0.34036318    0.43265003\nC      -0.83187173    0.70418067    0.88324591\nH       2.32472110   -0.25029805   -0.17789388\nH       1.28332450    1.14667614   -0.83695597\nH      -0.29365298   -1.02042821   -0.28596734\nH       0.48922284   -0.93756983    1.29560539\nH      -1.19281782    1.29832390    0.03681748\nH      -1.69636720    0.21982441    1.34850246\nH      -0.39178710    1.38838724    1.61666119',
                            'final_xyz': 'C       1.39393700    0.26537900   -0.20838600\nC       0.19342400   -0.33106400    0.44496000\nC      -0.84902400    0.70694400    0.88292100\nH       2.32817400   -0.28300300   -0.27001000\nH       1.31393200    1.18780400   -0.77595800\nH      -0.29199000   -1.04919300   -0.24230300\nH       0.50265500   -0.93665200    1.30845500\nH      -1.19276500    1.30046200    0.02844400\nH      -1.72540600    0.22697800    1.32987900\nH      -0.42854900    1.39899700    1.61954600',
                            'checkfile': '/storage/ce_dana/alongd/runs/ARC/debug13/calcs/Species/nC3H7/opt_a23998/check.chk',
                            'cheap_conformer': 'C       1.33903242    0.28849749    0.51672185\nC       0.18657092   -0.40960576   -0.12107655\nC      -1.14634812    0.08737020    0.41314525\nH       2.30155135   -0.20706406    0.57574068\nH       1.28914995    1.34778675    0.74274705\nH       0.23056488   -0.25673734   -1.20448178\nH       0.27283709   -1.48573243    0.06394873\nH      -1.22296487   -0.06954352    1.49444959\nH      -1.97028260   -0.45101088   -0.06611972\nH      -1.28011103    1.15603956    0.21368858',
                            'conformers': [
                                'C       1.37804355    0.27791700   -0.19511840\nC       0.17557158   -0.34036318    0.43265003\nC      -0.83187173    0.70418067    0.88324591\nH       2.32472110   -0.25029805   -0.17789388\nH       1.28332450    1.14667614   -0.83695597\nH      -0.29365298   -1.02042821   -0.28596734\nH       0.48922284   -0.93756983    1.29560539\nH      -1.19281782    1.29832390    0.03681748\nH      -1.69636720    0.21982441    1.34850246\nH      -0.39178710    1.38838724    1.61666119'],
                            'conformer_energies': [None], 'conformers_before_opt': [
                     'C       1.37804355    0.27791700   -0.19511840\nC       0.17557158   -0.34036318    0.43265003\nC      -0.83187173    0.70418067    0.88324591\nH       2.32472110   -0.25029805   -0.17789388\nH       1.28332450    1.14667614   -0.83695597\nH      -0.29365298   -1.02042821   -0.28596734\nH       0.48922284   -0.93756983    1.29560539\nH      -1.19281782    1.29832390    0.03681748\nH      -1.69636720    0.21982441    1.34850246\nH      -0.39178710    1.38838724    1.61666119']}],
             'p_species': [{'force_field': 'MMFF94s', 'is_ts': False, 'label': 'iC3H7',
                            'long_thermo_description': "Bond corrections: {'C-C': 2, 'C-H': 7}\n", 'multiplicity': 2,
                            'charge': 0, 'compute_thermo': True, 'number_of_rotors': 0, 'arkane_file': None,
                            'consider_all_diastereomers': True, 'e_elect': -311090.81145707075, 'run_time': 61.0,
                            'opt_level': 'b3lyp/6-31g(d,p)', 'conf_is_isomorphic': True,
                            'bond_corrections': {'C-C': 2, 'C-H': 7}, 'mol': {'atoms': [
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32758, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-32757: 1.0, -32755: 1.0, -32754: 1.0, -32753: 1.0}},
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 1, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32757, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-32758: 1.0, -32756: 1.0, -32752: 1.0}},
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32756, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-32757: 1.0, -32751: 1.0, -32750: 1.0, -32749: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32755, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-32758: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32754, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-32758: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32753, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-32758: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32752, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-32757: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32751, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-32756: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32750, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-32756: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32749, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-32756: 1.0}}], 'multiplicity': 2, 'props': {}, 'atom_order': [-32758, -32757, -32756, -32755, -32754, -32753, -32752, -32751, -32750, -32749]},
                            'initial_xyz': 'C       1.29196387    0.15815210    0.32047503\nC      -0.03887789   -0.17543467    0.89494533\nC      -1.26222918    0.47039644    0.34836510\nH       1.40933232    1.23955428    0.20511486\nH       2.08593721   -0.19903577    0.98301313\nH       1.41699441   -0.31973461   -0.65525752\nH      -0.13933823   -1.05339936    1.52398873\nH      -1.51964710    0.03926484   -0.62319221\nH      -2.10441807    0.31322346    1.02876738\nH      -1.11812298    1.54852996    0.23271515',
                            'final_xyz': 'C       1.30445700    0.14930600    0.33486100\nC      -0.03852500   -0.17292600    0.89922000\nC      -1.27617300    0.46478200    0.36303900\nH       1.43314300    1.23075500    0.19226600\nH       2.11562900   -0.20650200    0.97825700\nH       1.46118100   -0.30946900   -0.65803600\nH      -0.13925700   -1.05320700    1.52930100\nH      -1.56011100    0.05987600   -0.62504600\nH      -2.13512900    0.31314300    1.02467200\nH      -1.14362100    1.54575800    0.22040200',
                            'checkfile': '/storage/ce_dana/alongd/runs/ARC/debug13/calcs/Species/iC3H7/opt_a23999/check.chk',
                            'cheap_conformer': 'C      -1.28873024    0.06292844    0.10889819\nC       0.01096161   -0.45756396   -0.39342150\nC       1.28410310    0.11324608    0.12206177\nH      -1.49844465    1.04581965   -0.32238736\nH      -1.28247249    0.14649430    1.19953628\nH      -2.09838469   -0.61664655   -0.17318515\nH       0.02736023   -1.06013834   -1.29522253\nH       2.12255117   -0.53409831   -0.15158596\nH       1.26342625    0.19628892    1.21256167\nH       1.45962973    1.10366979   -0.30725541',
                            'conformers': [
                                'C       1.29196387    0.15815210    0.32047503\nC      -0.03887789   -0.17543467    0.89494533\nC      -1.26222918    0.47039644    0.34836510\nH       1.40933232    1.23955428    0.20511486\nH       2.08593721   -0.19903577    0.98301313\nH       1.41699441   -0.31973461   -0.65525752\nH      -0.13933823   -1.05339936    1.52398873\nH      -1.51964710    0.03926484   -0.62319221\nH      -2.10441807    0.31322346    1.02876738\nH      -1.11812298    1.54852996    0.23271515'],
                            'conformer_energies': [None], 'conformers_before_opt': [
                     'C       1.29196387    0.15815210    0.32047503\nC      -0.03887789   -0.17543467    0.89494533\nC      -1.26222918    0.47039644    0.34836510\nH       1.40933232    1.23955428    0.20511486\nH       2.08593721   -0.19903577    0.98301313\nH       1.41699441   -0.31973461   -0.65525752\nH      -0.13933823   -1.05339936    1.52398873\nH      -1.51964710    0.03926484   -0.62319221\nH      -2.10441807    0.31322346    1.02876738\nH      -1.11812298    1.54852996    0.23271515']}],
             'ts_species': {'force_field': 'MMFF94s', 'is_ts': True, 'label': 'TS0', 'long_thermo_description': '',
                            'multiplicity': 2, 'charge': 0, 'compute_thermo': False, 'number_of_rotors': 0,
                            'arkane_file': None, 'consider_all_diastereomers': True, 'ts_guesses': [
                     {'t0': '2022-05-26T23:41:03.211794', 'method': 'autotst', 'method_index': 0,
                      'method_direction': 'F', 'success': True, 'energy': 457.09273842390394, 'index': 0,
                      'imaginary_freqs': None, 'conformer_index': 0, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:17.672822',
                      'initial_xyz': 'C       0.06870000   -0.52310000   -0.65000000\nC       1.32690000   -0.17800000    0.12310000\nC      -1.61580000    0.23640000    0.43190000\nH      -0.94590000   -0.88470000    0.05810000\nH       0.00080000   -0.05630000   -1.65490000\nH       1.36660000   -0.77570000    1.05730000\nH       2.21840000   -0.40920000   -0.49590000\nH       1.32650000    0.90240000    0.37660000\nH      -1.31000000    1.13360000    1.01690000\nH      -2.43630000    0.55470000   -0.26310000',
                      'opt_xyz': 'C       0.61418600   -0.57125700   -1.11315700\nC       1.52769300   -0.13581800   -0.03895800\nC      -1.76088500    0.17894600    0.38021300\nH      -1.71341100   -0.89798100    0.54776400\nH      -0.18837200    0.14692200   -1.38061400\nH       1.10728900   -0.50824200    0.91631600\nH       2.52104100   -0.60562200   -0.12803100\nH       1.64035400    0.95840500    0.07522100\nH      -1.26555200    0.85783800    1.07567800\nH      -2.48244300    0.57690900   -0.33443300'},
                     {'t0': '2022-05-26T23:41:03.211794', 'method': 'autotst', 'method_index': 1,
                      'method_direction': 'F', 'success': True, 'energy': None, 'index': 1, 'imaginary_freqs': None,
                      'conformer_index': 1, 'successful_irc': None, 'successful_normal_mode': None,
                      'execution_time': '0:00:17.672822',
                      'initial_xyz': 'C       1.27140000   -0.19880000    0.27060000\nC      -1.52300000    0.74650000   -0.35470000\nC       0.02370000   -0.73290000   -0.41700000\nH      -0.96230000   -0.12240000   -1.12430000\nH       1.61070000    0.72270000   -0.24570000\nH       1.04290000    0.03790000    1.33060000\nH       2.07780000   -0.96010000    0.22660000\nH      -1.03890000    1.73950000   -0.25260000\nH      -2.20860000    0.44390000    0.46320000\nH      -0.29360000   -1.67630000    0.10330000'},
                     {'t0': '2022-05-26T23:41:20.911151', 'method': 'autotst', 'method_index': 2,
                      'method_direction': 'R', 'success': True, 'energy': None, 'index': 2, 'imaginary_freqs': None,
                      'conformer_index': 2, 'successful_irc': None, 'successful_normal_mode': None,
                      'execution_time': '0:00:15.846009',
                      'initial_xyz': 'C       1.71270000   -0.29390000    0.04290000\nC      -1.29740000   -0.16230000   -0.10640000\nC      -0.10790000    0.76300000   -0.32970000\nH       1.08610000    0.51490000   -0.81510000\nH       1.48670000   -1.37900000   -0.00500000\nH       2.09490000    0.08260000    1.01380000\nH      -2.24090000    0.38520000   -0.31090000\nH      -1.29900000   -0.52400000    0.94300000\nH      -1.22320000   -1.03060000   -0.79300000\nH      -0.21210000    1.64410000    0.36030000'},
                     {'t0': '2022-05-26T23:41:20.911151', 'method': 'autotst', 'method_index': 3,
                      'method_direction': 'R', 'success': True, 'energy': None, 'index': 3, 'imaginary_freqs': None,
                      'conformer_index': 3, 'successful_irc': None, 'successful_normal_mode': None,
                      'execution_time': '0:00:15.846009',
                      'initial_xyz': 'C       0.14170000    0.77420000   -0.31580000\nC       1.33000000   -0.02950000    0.17580000\nC      -1.64460000   -0.24370000    0.27000000\nH      -0.90890000    0.81700000    0.50710000\nH       0.07300000    0.86010000   -1.42040000\nH       1.38540000    0.02680000    1.28260000\nH       2.26370000    0.38470000   -0.25780000\nH       1.22000000   -1.08930000   -0.13430000\nH      -2.43100000   -0.17220000   -0.52760000\nH      -1.42930000   -1.32810000    0.42060000'},
                     {'t0': '2022-05-26T23:41:37.414768', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'F', 'success': True, 'energy': 180.8585423058248, 'index': 4,
                      'imaginary_freqs': [-1962.3757], 'conformer_index': 4, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:02.875351',
                      'initial_xyz': 'C      -1.06700206   -0.98290563    0.09666771\nC      -0.01429677    0.14576414   -0.26970389\nC       1.07411957    0.53558373    0.44180292\nH      -0.90097213   -1.99955773   -0.00804090\nH      -1.92791498   -0.83538461    0.61150122\nH      -0.72827876   -0.72893214   -0.99451387\nH      -0.19050112    0.61306989   -1.05721593\nH       0.65847605    1.37783504    0.86201239\nH       1.92377913    0.72413480   -0.06474017\nH       1.29896247   -0.04988281    1.22807050',
                      'opt_xyz': 'C      -0.94073000   -1.08641400   -0.13521400\nC      -0.33120900    0.27400500   -0.10738200\nC       1.03297600    0.48550500    0.47374300\nH      -0.34195200   -1.93903700    0.19367700\nH      -2.02636000   -1.19201700   -0.17452300\nH      -0.43853800   -0.46541200   -1.16876000\nH      -1.02644700    1.11801000   -0.14324700\nH       1.47599100    1.43717700    0.14587900\nH       1.72184900   -0.32388500    0.18160800\nH       1.00079200    0.49179200    1.58006000'},
                     {'t0': '2022-05-26T23:41:40.296647', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': 28.47611632832559, 'index': 5,
                      'imaginary_freqs': [-247.9856], 'conformer_index': 5, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.891656',
                      'initial_xyz': 'C      -1.17498302    0.20234825    0.61815524\nC      -0.14451513   -0.22229944   -0.36327118\nC       1.37613642   -0.20638606   -0.29276791\nH      -1.91179907    0.93071961    0.36481556\nH      -1.65566945   -0.34783489    1.24894297\nH      -0.69110036   -0.96965396    0.09287225\nH      -0.52582997   -0.29021457   -1.37846231\nH       1.47940874    0.65462989   -0.58598584\nH       1.83152962   -0.71277982   -1.11679804\nH       1.74064636    0.29004735    0.63243920',
                      'opt_xyz': 'C      -1.20591200    0.47424500    0.36946100\nC      -0.29394000   -0.44521000   -0.36471500\nC       1.20715500   -0.13952500   -0.20416700\nH      -2.28344900    0.44537500    0.18865600\nH      -0.84180200    1.09602600    1.19197200\nH      -0.49608300   -1.48771500   -0.04130900\nH      -0.57224000   -0.44812600   -1.43454100\nH       1.37268200    0.73806100    0.43870000\nH       1.68512500    0.07671500   -1.17052800\nH       1.75228700   -0.98126900    0.24641000'},
                     {'t0': '2022-05-26T23:41:42.194417', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'F', 'success': True, 'energy': 180.85896501125535, 'index': 6,
                      'imaginary_freqs': None, 'conformer_index': 6, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.913523',
                      'initial_xyz': 'C      -1.20873415    0.71530855   -0.52397704\nC       0.31399596    0.81271130   -0.77977943\nC       1.24493992    0.12856162    0.05752292\nH      -1.95132613    0.96989143    0.11211920\nH      -1.76489258    0.25215778   -1.29827285\nH      -0.50961399    1.76985371   -0.06161229\nH       0.62333548    1.23630238   -1.62280464\nH       1.41772580    0.03376916    0.95344961\nH       2.18181777    0.58830410   -0.08182961\nH       0.75694025   -0.81274462    0.46448511',
                      'opt_xyz': 'C      -1.22745100    0.78774200   -0.50893200\nC       0.23979800    0.78871300   -0.77355000\nC       1.19005200    0.16594100    0.20252100\nH      -1.59024000    0.40579300    0.44822900\nH      -1.92750900    0.86269900   -1.34296200\nH      -0.44232200    1.82484300   -0.39144600\nH       0.54372500    0.88104400   -1.82053600\nH       0.90894700    0.40388700    1.24163400\nH       2.22267500    0.51098300    0.04740600\nH       1.18651400   -0.93753000    0.11693700'},
                     {'t0': '2022-05-26T23:41:44.113571', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': 28.47585902933497, 'index': 7,
                      'imaginary_freqs': [-247.9818], 'conformer_index': 7, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.908796',
                      'initial_xyz': 'C       1.02964175    0.68634045   -0.24918917\nC       0.06535231   -0.13109583    0.30977562\nC      -1.42600191   -0.11903072    0.61585003\nH       1.62296820    1.51068902    0.02148751\nH       1.33659041    0.45585936   -1.11817455\nH       0.58185875    0.60811055    0.84054208\nH       0.52493042   -1.22503757    0.64394557\nH      -1.53707004    0.36833912    1.51066422\nH      -1.94065011   -0.88619411    0.56052530\nH      -1.86324120    0.63528162   -0.11346494',
                      'opt_xyz': 'C       0.95062100    0.66915600   -0.44795600\nC       0.21199900   -0.19479100    0.51349900\nC      -1.28747400    0.12620900    0.65892300\nH       0.54583300    1.63209800   -0.77168200\nH       1.97180500    0.41800500   -0.74581000\nH       0.70253200   -0.12992100    1.50713500\nH       0.35123500   -1.25292900    0.22542900\nH      -1.54934400    0.40978500    1.68835900\nH      -1.91697900   -0.73384400    0.38894600\nH      -1.58584900    0.95949400    0.00511800'},
                     {'t0': '2022-05-26T23:41:46.029238', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'F', 'success': True, 'energy': 180.8588232342736, 'index': 8,
                      'imaginary_freqs': None, 'conformer_index': 8, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.874294',
                      'initial_xyz': 'C      -1.48518515   -0.23374367   -0.23082997\nC      -0.24483547    0.51183116    0.41401744\nC       1.12809086    0.09508162    0.25930798\nH      -2.24081373    0.35100281   -0.63667858\nH      -1.89986777   -1.13838661   -0.37233800\nH      -1.16789770   -0.34802154    0.82655811\nH      -0.53055906    1.36634922    0.83735430\nH       1.39702737    0.53136593   -0.54898417\nH       1.63798177    0.33128169    1.15132225\nH       1.27656150   -0.84899682   -0.05066128',
                      'opt_xyz': 'C      -1.50785500   -0.17356000   -0.23397900\nC      -0.37774200    0.49776600    0.46979900\nC       1.03237900    0.04062300    0.25540200\nH      -2.43960000    0.36960000   -0.40168300\nH      -1.30039300   -1.05924100   -0.83912300\nH      -1.21188200   -0.32485400    1.02906500\nH      -0.55506800    1.52491200    0.80243500\nH       1.43111700    0.40933600   -0.70936600\nH       1.70382800    0.39261800    1.05206300\nH       1.09571900   -1.05943700    0.22445500'},
                     {'t0': '2022-05-26T23:41:47.908778', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': 180.86181630415376, 'index': 9,
                      'imaginary_freqs': None, 'conformer_index': 9, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.874496',
                      'initial_xyz': 'C       1.76142025   -0.32615915   -0.31378934\nC       0.26012418   -0.50720125   -0.57989764\nC      -0.83998406    0.51821351   -0.61445594\nH       2.41904354   -0.49318668   -1.15896177\nH       2.07220912   -0.43000120    0.63903868\nH       0.91324747   -0.90038979   -0.88085961\nH      -0.25748059   -1.59517562   -0.60023719\nH      -0.67072332    1.01107574    0.27245176\nH      -1.69623446    0.36441237   -1.05993485\nH      -0.47145543    1.33631945   -1.17536044',
                      'opt_xyz': 'C       1.65863100   -0.47380300   -0.35467500\nC       0.17468800   -0.61083700   -0.40103100\nC      -0.68874000    0.56815100   -0.72949700\nH       2.11011800    0.47530200   -0.65302000\nH       2.24747600   -1.16958900    0.24557700\nH       0.98476000   -1.05412600   -1.31295700\nH      -0.25460900   -1.43358700    0.17871800\nH      -0.78230000    1.24962700    0.13753100\nH      -1.70193000    0.26482300   -1.03037000\nH      -0.25792800    1.16194800   -1.55228100'},
                     {'t0': '2022-05-26T23:41:51.692281', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': 180.85910678829532, 'index': 10,
                      'imaginary_freqs': None, 'conformer_index': 10, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.857693',
                      'initial_xyz': 'C       1.45932961   -0.40568867   -0.48573068\nC       0.07347362    0.17359543   -0.56615287\nC      -1.32016015   -0.44909847   -0.37993887\nH       2.03022957   -1.01502204   -0.05810530\nH       2.13141823    0.20080915   -1.03013027\nH       0.64928532   -0.32624221   -1.27710116\nH      -0.01041280    1.17878938   -0.71969837\nH      -1.24753046   -1.07629967    0.34866881\nH      -2.06527519    0.17695132   -0.33415791\nH      -1.42948318   -1.16617382   -1.21987271',
                      'opt_xyz': 'C       1.39402000   -0.37546100   -0.68083900\nC       0.05132900    0.27274800   -0.69128200\nC      -1.17344800   -0.50633300   -0.32188400\nH       1.45964400   -1.44843800   -0.48606700\nH       2.28898600    0.23638300   -0.55441700\nH       0.70827900   -0.08260100   -1.75281000\nH       0.03733000    1.36144800   -0.58346900\nH      -1.25420900   -0.63011100    0.77497600\nH      -2.09322700   -0.01591200   -0.67250300\nH      -1.14782900   -1.52010300   -0.75392300'},
                     {'t0': '2022-05-26T23:41:53.557121', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'F', 'success': True, 'energy': 180.85898338980041, 'index': 11,
                      'imaginary_freqs': None, 'conformer_index': 11, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.856500',
                      'initial_xyz': 'C      -1.51046324   -0.05654161   -0.17449787\nC      -0.19565231   -0.49037039    0.45278808\nC       0.89220119    0.38680810    0.88834977\nH      -2.35916400   -0.09211956    0.43497366\nH      -1.94329524    0.31062192   -0.99955177\nH      -0.51100838   -0.25559592   -0.80780500\nH      -0.27175251   -1.40853167    0.85534734\nH       1.03489912    1.27390730    0.29284352\nH       1.72903490   -0.00521671    0.87521076\nH       0.39971116    0.88248992    1.71725774',
                      'opt_xyz': 'C      -1.55102600   -0.01764800   -0.15063200\nC      -0.27961500   -0.49037900    0.46836100\nC       0.81488200    0.47842700    0.79522200\nH      -2.46365400   -0.60315800   -0.02583100\nH      -1.63676300    1.03145600   -0.44337600\nH      -0.60079400   -0.65705600   -0.77833000\nH      -0.32952200   -1.43561000    1.01706800\nH       0.94016900    1.22520500   -0.00580600\nH       1.78194900   -0.02655600    0.93690600\nH       0.58888500    1.04077100    1.72133500'},
                     {'t0': '2022-05-26T23:41:55.421723', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': 180.8587628477835, 'index': 12,
                      'imaginary_freqs': None, 'conformer_index': 12, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.853634',
                      'initial_xyz': 'C      -0.58550024    1.08561718   -0.15663983\nC       0.42889541    0.45084977    0.59194607\nC       1.37000704   -0.60319155    0.74652404\nH      -1.47753155    1.46616268    0.40266326\nH      -0.57472807    1.33417463   -1.12147713\nH      -0.71895623   -0.17481887    0.10822438\nH       0.81476825    1.35811055    1.05770040\nH       1.98313403   -0.47356719   -0.09324066\nH       1.39554203   -1.56005931    0.87285095\nH       1.42157912   -0.48325348    1.76832342',
                      'opt_xyz': 'C      -0.63997100    0.88380900   -0.20145300\nC       0.34594100    0.56011900    0.86922900\nC       1.40902800   -0.46981300    0.64022500\nH      -1.13365300    1.85720800   -0.20589100\nH      -0.63149900    0.29502200   -1.12181100\nH      -0.89396500    0.17556700    0.86618200\nH       0.51614700    1.33529500    1.62234300\nH       2.23406900   -0.06376700    0.02428000\nH       1.00940700   -1.34313100    0.09886900\nH       1.84170500   -0.83028400    1.58490200'},
                     {'t0': '2022-05-26T23:41:59.196027', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': None, 'index': 13, 'imaginary_freqs': None,
                      'conformer_index': 13, 'successful_irc': None, 'successful_normal_mode': None,
                      'execution_time': '0:00:01.878886',
                      'initial_xyz': 'C       0.96748012   -0.81678355   -0.31229910\nC       0.37854999    0.36768591    0.51071781\nC      -1.07459605    0.66234928    0.62251425\nH       1.73308313   -0.68185002   -0.92448866\nH       1.04368222   -1.71272838    0.00532236\nH       1.10471821    0.25733542    0.11611196\nH       1.05092859    1.07043862    1.24027312\nH      -1.39371753    0.75815308   -0.36373425\nH      -1.41272283    1.44293737    1.03271341\nH      -1.48376942   -0.15894234    1.14175749'},
                     {'t0': '2022-05-26T23:42:01.079910', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'F', 'success': True, 'energy': 180.85898601531517, 'index': 14,
                      'imaginary_freqs': None, 'conformer_index': 14, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.857904',
                      'initial_xyz': 'C       0.76262105   -0.76400435    0.93583715\nC      -0.65082604   -0.60366839    0.43407044\nC      -1.19700181    0.60047364   -0.20144393\nH       1.00494814   -1.72844446    1.08477473\nH       1.24139726   -0.43856725    1.77211285\nH       0.63145959   -0.97310996   -0.35562319\nH      -1.31025076   -1.13134599    0.93603009\nH      -0.49987090    1.15099120   -0.75667942\nH      -1.96762645    0.30284613   -0.80094039\nH      -1.43324780    1.20295048    0.58301079',
                      'opt_xyz': 'C       0.72155900   -0.86117000    0.97318200\nC      -0.61848700   -0.72093700    0.33453400\nC      -1.12815500    0.62572200   -0.07815300\nH       0.92710900   -1.71883100    1.61623600\nH       1.36170600    0.02052200    1.05393400\nH       0.43775100   -1.18715300   -0.25927600\nH      -1.33931300   -1.51820200    0.53930300\nH      -0.33104900    1.23030900   -0.54085700\nH      -1.95289300    0.54866000   -0.80154200\nH      -1.49662600    1.19920100    0.79378800'},
                     {'t0': '2022-05-26T23:42:02.943259', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': 0.000761395029257983, 'index': 15,
                      'imaginary_freqs': None, 'conformer_index': 15, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.972401',
                      'initial_xyz': 'C      -0.88464719   -0.15515247   -0.44346783\nC       0.51426482    0.36782986    0.08452535\nC       1.67532301   -0.16636546    0.76834404\nH      -1.73940670   -0.05415071    0.08939913\nH      -0.95850170   -0.52282810   -1.42392790\nH      -0.21972930   -0.35693800   -0.12551412\nH       0.56886059    1.42613912   -0.16336975\nH       1.98881960   -1.04647338    0.11560042\nH       2.46397924    0.41142815    0.83529609\nH       1.35643530   -0.47905838    1.72272897',
                      'opt_xyz': 'C      -0.59745500   -0.27793600   -0.57857500\nC       0.48467300    0.52323400    0.05729400\nC       1.53324200   -0.09835600    0.91430400\nH      -1.12677200   -0.90979500    0.15898600\nH      -1.34483800    0.35715900   -1.07491300\nH      -0.19950600   -0.97597300   -1.34239600\nH       0.54166300    1.59877300   -0.13597200\nH       1.41852300   -1.19214200    0.96826900\nH       2.55256900    0.11174200    0.54188200\nH       1.50329900    0.28772400    1.95073600'},
                     {'t0': '2022-05-26T23:42:04.922705', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'F', 'success': True, 'energy': 180.8585974413436, 'index': 16,
                      'imaginary_freqs': None, 'conformer_index': 16, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.981361',
                      'initial_xyz': 'C       1.45752501   -0.34142691   -1.12588692\nC       0.28476441    0.55947429   -0.89261472\nC      -0.58748847    0.63957649    0.22596583\nH       2.07569289   -1.14329469   -0.79948431\nH       2.20992613    0.01069019   -1.45111012\nH       0.37588233   -0.65742445   -1.53954506\nH       0.08357757    1.29397428   -1.73200774\nH      -0.80598485   -0.35379100    0.63491660\nH      -1.45885408    0.91999155    0.06410865\nH       0.05302338    0.88666624    1.01776028',
                      'opt_xyz': 'C       1.50202500   -0.33601100   -1.14219900\nC       0.28744200    0.51648300   -0.99711700\nC      -0.52374300    0.47896400    0.26147300\nH       1.73964600   -1.05373100   -0.35347400\nH       2.29035900   -0.04080700   -1.83732000\nH       0.35207100   -0.55979200   -1.71961800\nH       0.24456900    1.41057000   -1.62633000\nH      -0.63348600   -0.55198700    0.63573000\nH      -1.53267100    0.89056300    0.11241000\nH      -0.03814700    1.06018300    1.06854700'},
                     {'t0': '2022-05-26T23:42:06.912255', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': 180.8586131943157, 'index': 17,
                      'imaginary_freqs': None, 'conformer_index': 17, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.880731',
                      'initial_xyz': 'C      -0.74663436   -0.55736071    0.14056355\nC       0.56897283    0.02896419    0.40835238\nC       1.86595774   -0.83959478    0.71856529\nH      -1.21906948   -0.27343133   -0.69941294\nH      -1.41548634   -1.06072807    0.73646659\nH      -0.18981230    0.43102172    1.16182208\nH       0.84268308    0.82189310   -0.03522815\nH       1.67278290   -1.53900778    1.42446065\nH       2.71231127   -0.30402660    0.73884761\nH       1.88601792   -1.52979362   -0.16393194',
                      'opt_xyz': 'C      -0.75934400   -0.43268700    0.22653800\nC       0.65504000    0.03833200    0.25318700\nC       1.75696600   -0.88370100    0.67631100\nH      -1.48104000    0.05952800   -0.42780900\nH      -0.98946500   -1.43166100    0.60438900\nH      -0.24707700    0.36024700    1.12928800\nH       0.89598200    0.88661000   -0.39432800\nH       1.45676900   -1.49377900    1.54388300\nH       2.66878100   -0.33436700    0.95240100\nH       2.02111100   -1.59058500   -0.13335500'},
                     {'t0': '2022-05-26T23:42:08.797819', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'F', 'success': True, 'energy': 0.0, 'index': 18, 'imaginary_freqs': None,
                      'conformer_index': 18, 'successful_irc': None, 'successful_normal_mode': None,
                      'execution_time': '0:00:01.927177',
                      'initial_xyz': 'C       1.25889516   -0.08437126    0.41225553\nC      -0.00654572   -0.50146621   -0.40311623\nC      -1.15623581    0.17680863   -0.62774318\nH       1.95039654    0.60666728    0.25720629\nH       1.42539263   -0.28843823    1.43446612\nH       1.29356039   -0.91786581   -0.35060552\nH       0.01991384   -1.51818204   -0.68589437\nH      -1.04161429    0.91045481   -1.20867670\nH      -1.89031172   -0.47944239   -0.92619979\nH      -1.32774293    0.66987532    0.21154730',
                      'opt_xyz': 'C       1.26777600   -0.27870900    0.29276500\nC       0.07463000   -0.62022400   -0.53010300\nC      -1.16029300    0.21360700   -0.52169200\nH       1.59040700    0.76699400    0.13219700\nH       1.05951400   -0.36638400    1.37791700\nH       2.12334900   -0.93227100    0.07125200\nH       0.08167100   -1.53252500   -1.13428800\nH      -1.37845000    0.64026200   -1.51890000\nH      -2.05479500   -0.37150500   -0.23997200\nH      -1.07810100    1.05479500    0.18406400'},
                     {'t0': '2022-05-26T23:42:10.729805', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': None, 'index': 19, 'imaginary_freqs': None,
                      'conformer_index': 19, 'successful_irc': None, 'successful_normal_mode': None,
                      'execution_time': '0:00:01.908651',
                      'initial_xyz': 'C      -1.39099145   -0.51042563    0.51720619\nC      -0.23563087    0.22043005   -0.48801798\nC       1.02234614    0.22202557   -0.15813138\nH      -1.66820598   -1.36889911    0.67907137\nH      -2.04905176    0.37884614    0.97399098\nH      -0.64659941    0.19653773    0.31738496\nH      -0.70008379    0.71301955   -1.30898690\nH       1.47499561   -0.65745252   -0.29022256\nH       1.59206700    0.78563404   -0.95010835\nH       1.42435002    0.75069726    0.67192549'},
                     {'t0': '2022-05-26T23:42:12.647837', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'F', 'success': True, 'energy': 180.8588179833023, 'index': 20,
                      'imaginary_freqs': None, 'conformer_index': 20, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.894893',
                      'initial_xyz': 'C       1.00167012    0.51517284   -0.06591963\nC      -0.18289231    0.17261295   -0.82903284\nC      -1.55816829   -0.17413202   -0.13559744\nH       1.52568972    1.36995649    0.10040943\nH       1.80391240   -0.14884450    0.13658467\nH       0.23577479    1.34846580   -0.45981669\nH      -0.07382769   -0.08149067   -1.80809546\nH      -1.44916737   -1.09953022    0.21490166\nH      -2.30217791    0.14279181   -0.75846672\nH      -1.60130811    0.19376379    0.86936384',
                      'opt_xyz': 'C       1.07037200    0.51932700   -0.08930100\nC      -0.17470700    0.14173400   -0.81755900\nC      -1.45969100   -0.06121300   -0.07563200\nH       1.01161600    0.73188100    0.98080900\nH       2.04355700    0.28507500   -0.52448500\nH       0.35222500    1.32806600   -0.82117000\nH      -0.03975400   -0.35171200   -1.78471400\nH      -1.47658900   -1.04259500    0.43586800\nH      -2.33157300   -0.01684200   -0.74442900\nH      -1.59595100    0.70504400    0.70494300'},
                     {'t0': '2022-05-26T23:42:14.548687', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': 180.85891512676608, 'index': 21,
                      'imaginary_freqs': None, 'conformer_index': 21, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.856999',
                      'initial_xyz': 'C      -1.29100406   -0.56441748   -0.56283945\nC      -0.69772559    0.45237696    0.12930320\nC       0.48780710    0.53718090    1.09389055\nH      -2.13487411   -1.01646090   -0.37586549\nH      -0.74074459   -1.03662610   -1.36385643\nH      -0.88092804    0.78589129   -0.95387864\nH      -1.42911565    1.25710464    0.56285071\nH       1.13139296    1.01143909    0.52348405\nH       1.22643602    0.13242149    1.36027193\nH       0.07100621    0.09331925    1.99217510',
                      'opt_xyz': 'C      -1.28361900   -0.58490300   -0.52884500\nC      -0.77840100    0.59669300    0.22753600\nC       0.50959100    0.51909800    0.98833400\nH      -2.35094600   -0.67802000   -0.73703700\nH      -0.65628400   -1.47706100   -0.59368100\nH      -0.73787900    0.47431800   -1.06384700\nH      -1.52697700    1.33501600    0.52991200\nH       0.91155600    1.51658700    1.21846100\nH       1.27761800   -0.02903900    0.41908400\nH       0.37758900   -0.02045900    1.94561900'},
                     {'t0': '2022-05-26T23:42:16.557742', 'method': 'kinbot', 'method_index': 0,
                      'method_direction': 'F', 'success': True, 'energy': 17.085965428326745, 'index': 22,
                      'imaginary_freqs': [-90.065], 'conformer_index': 22, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:00.006560',
                      'initial_xyz': 'C       1.39393700    0.26537900   -0.20838600\nC       0.19342400   -0.33106400    0.44496000\nC      -0.84902400    0.70694400    0.88292100\nH       2.32817400   -0.28300300   -0.27001000\nH       1.31393200    1.18780400   -0.77595800\nH      -0.29199000   -1.04919300   -0.24230300\nH       0.50265500   -0.93665200    1.30845500\nH      -1.19276500    1.30046200    0.02844400\nH      -1.72540600    0.22697800    1.32987900\nH      -0.42854900    1.39899700    1.61954600',
                      'opt_xyz': 'C       1.37026500    0.21375700   -0.27043800\nC       0.15134000   -0.35818800    0.36799100\nC      -0.80019700    0.70757700    0.93794300\nH       2.33530800    0.22065000    0.24218900\nH       1.29190400    0.76724000   -1.21086100\nH      -0.40898300   -0.95723900   -0.37328200\nH       0.43817300   -1.05326200    1.17318700\nH      -1.13112000    1.40316500    0.15146200\nH      -1.69586400    0.24180000    1.37707100\nH      -0.30643800    1.30115200    1.72228500'},
                     {'t0': '2022-05-26T23:42:16.606748', 'method': 'kinbot', 'method_index': 1,
                      'method_direction': 'R', 'success': True, 'energy': 6.82630343362689e-05, 'index': 23,
                      'imaginary_freqs': None, 'conformer_index': 23, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:00.005740',
                      'initial_xyz': 'C       1.30445700    0.14930600    0.33486100\nC      -0.03852500   -0.17292600    0.89922000\nC      -1.27617300    0.46478200    0.36303900\nH       1.43314300    1.23075500    0.19226600\nH       2.11562900   -0.20650200    0.97825700\nH       1.46118100   -0.30946900   -0.65803600\nH      -0.13925700   -1.05320700    1.52930100\nH      -1.56011100    0.05987600   -0.62504600\nH      -2.13512900    0.31314300    1.02467200\nH      -1.14362100    1.54575800    0.22040200',
                      'opt_xyz': 'C       1.31569700    0.21321300    0.38752400\nC      -0.03847200   -0.25392000    0.79404500\nC      -1.28436600    0.38955400    0.29002800\nH       1.53200500    1.23187800    0.76714600\nH       2.11024900   -0.44892600    0.75976300\nH       1.41460600    0.27511900   -0.71237600\nH      -0.13099100   -1.07201200    1.51484000\nH      -1.90661200   -0.31435900   -0.29433400\nH      -1.92744900    0.75124200    1.11310300\nH      -1.06307300    1.24972500   -0.36080500'}],
                            'ts_conf_spawned': True, 'ts_guesses_exhausted': False, 'ts_number': 0, 'ts_report': '',
                            'rxn_label': 'nC3H7 <=> iC3H7', 'rxn_index': 0,
                            'successful_methods': ['autotst', 'autotst', 'autotst', 'autotst', 'gcn', 'gcn', 'gcn',
                                                   'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'gcn',
                                                   'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'kinbot', 'kinbot'],
                            'unsuccessful_methods': [], 'chosen_ts_method': 'gcn', 'chosen_ts': 4,
                            'rxn_zone_atom_indices': None, 'chosen_ts_list': [18, 23, 15, 22, 7, 5, 4],
                            'ts_checks': {'E0': None, 'e_elect': True, 'IRC': None, 'freq': True,
                                          'NMD': None, 'warnings': ''},
                            'e_elect': -310902.61556421133, 'tsg_spawned': True, 'opt_level': 'b3lyp/6-31g(d,p)',
                            'bond_corrections': {}, 'mol': {'atoms': [
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26582, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-26581: 1.0, -26579: 1.0, -26578: 1.0, -26577: 1.0}},
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 1, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26581, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-26582: 1.0, -26580: 1.0, -26576: 1.0}},
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26580, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-26581: 1.0, -26575: 1.0, -26574: 1.0, -26573: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26579, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26582: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26578, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26582: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26577, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26582: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26576, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26581: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26575, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26580: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26574, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26580: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26573, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26580: 1.0}}], 'multiplicity': 2, 'props': {},
                     'atom_order': [-26582, -26581, -26580, -26579, -26578, -26577, -26576, -26575, -26574, -26573]},
                            'initial_xyz': 'C      -0.94073000   -1.08641400   -0.13521400\nC      -0.33120900    0.27400500   -0.10738200\nC       1.03297600    0.48550500    0.47374300\nH      -0.34195200   -1.93903700    0.19367700\nH      -2.02636000   -1.19201700   -0.17452300\nH      -0.43853800   -0.46541200   -1.16876000\nH      -1.02644700    1.11801000   -0.14324700\nH       1.47599100    1.43717700    0.14587900\nH       1.72184900   -0.32388500    0.18160800\nH       1.00079200    0.49179200    1.58006000',
                            'final_xyz': 'C      -0.94403900   -1.08919600   -0.13528100\nC      -0.33161500    0.27504900   -0.10775400\nC       1.03465100    0.48772700    0.47568400\nH      -0.35007600   -1.93768200    0.18473700\nH      -2.02160800   -1.19512500   -0.17500700\nH      -0.44246400   -0.46665500   -1.16369500\nH      -1.02252600    1.11305500   -0.14369600\nH       1.47717100    1.43311000    0.14664900\nH       1.72023500   -0.31816900    0.18700200\nH       1.00664300    0.49760900    1.57720100',
                            'checkfile': '/storage/ce_dana/alongd/runs/ARC/debug13/calcs/TSs/TS0/opt_a24061/check.chk'},
             'done_opt_r_n_p': True, 'family': 'intra_H_migration', 'family_own_reverse': True, 'ts_label': 'TS0'}
        rxn = ARCReaction(reaction_dict=rxn_dict)
        output = {'nC3H7': {'paths': {'geo': os.path.join(ARC_PATH, 'arc', 'testing', 'opt', 'nC3H7.out'),
                                      'freq': os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'nC3H7.out'),
                                      'sp': os.path.join(ARC_PATH, 'arc', 'testing', 'opt', 'nC3H7.out'),
                                      'composite': ''},
                            'restart': '', 'convergence': True,
                            'job_types': {'conf_opt': True, 'conf_sp': False, 'opt': True, 'freq': True, 'sp': True, 'rotors': True, 'irc': True, 'fine': True},
                            },
                  'iC3H7': {'paths': {'geo': os.path.join(ARC_PATH, 'arc', 'testing', 'opt', 'iC3H7.out'),
                                      'freq': os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'iC3H7.out'),
                                      'sp': os.path.join(ARC_PATH, 'arc', 'testing', 'opt', 'iC3H7.out'),
                                      'composite': ''},
                            'restart': '', 'convergence': True,
                            'job_types': {'conf_opt': True, 'conf_sp': False, 'opt': True, 'freq': True, 'sp': True, 'rotors': True, 'irc': True, 'fine': True},
                            },
                  'TS0': {'paths': {'geo': os.path.join(ARC_PATH, 'arc', 'testing', 'opt', 'TS_nC3H7-iC3H7.out'),
                                    'freq': os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'TS_nC3H7-iC3H7.out'),
                                    'sp': os.path.join(ARC_PATH, 'arc', 'testing', 'opt', 'TS_nC3H7-iC3H7.out'),
                                    'composite': ''},
                          'restart': '', 'convergence': True,
                          'job_types': {'conf_opt': True, 'conf_sp': False, 'opt': True, 'freq': True, 'sp': True, 'rotors': True, 'irc': True, 'fine': True},
                            },
                  }
        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_for_testing_delete_after_usage6')
        os.makedirs(os.path.join(project_directory, 'output', 'Species', 'nC3H7', 'geometry'))
        os.makedirs(os.path.join(project_directory, 'output', 'Species', 'iC3H7', 'geometry'))
        os.makedirs(os.path.join(project_directory, 'output', 'rxns', 'TS0', 'geometry'))
        shutil.copy(src=os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'nC3H7.out'),
                    dst=os.path.join(project_directory, 'output', 'Species', 'nC3H7', 'geometry', 'freq.out'))
        shutil.copy(src=os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'iC3H7.out'),
                    dst=os.path.join(project_directory, 'output', 'Species', 'iC3H7', 'geometry', 'freq.out'))
        shutil.copy(src=os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'TS_nC3H7-iC3H7.out'),
                    dst=os.path.join(project_directory, 'output', 'rxns', 'TS0', 'geometry', 'freq.out'))
        sched = Scheduler(project='test_rxn_e0_check',
                          ess_settings=self.ess_settings,
                          project_directory=os.path.join(ARC_PATH, 'Projects', 'arc_project_for_testing_delete_after_usage6'),
                          rxn_list=[rxn],
                          species_list=rxn.r_species + rxn.p_species + [rxn.ts_species],
                          kinetics_adapter='arkane',
                          freq_scale_factor=1.0,
                          sp_level=Level(repr='B3LYP/6-31G(d,p)'),
                          job_types=initialize_job_types(),
                          restart_dict={'output': output},
                          )
        self.assertEqual(rxn.ts_species.ts_checks,
                         {'E0': None, 'e_elect': True, 'IRC': None, 'freq': True, 'NMD': None, 'warnings': ''})

        job_1 = job_factory(job_adapter='gaussian',
                            species=[ARCSpecies(label='SPC', smiles='C')],
                            job_type='freq',
                            level=Level(repr='B3LYP/6-31G(d,p)'),
                            project='test_project',
                            project_directory=os.path.join(ARC_PATH,
                                                           'Projects',
                                                           'arc_project_for_testing_delete_after_usage6'),
                            )
        job_1.local_path_to_output_file = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'TS_nC3H7-iC3H7.out')
        check_ts(reaction=rxn, verbose=True, job=job_1, checks=['freq'])
        self.assertEqual(rxn.ts_species.ts_checks,
                         {'E0': None, 'e_elect': True, 'IRC': None, 'freq': True, 'NMD': True, 'warnings': ''})

        sched.check_rxn_e0_by_spc('TS0')
        self.assertEqual(rxn.ts_species.ts_checks,
                         {'E0': True, 'e_elect': True, 'IRC': None, 'freq': True, 'NMD': True, 'warnings': ''})

    def test_save_e_elect(self):
        """Test the save_e_elect() method."""
        project_directory = os.path.join(ARC_PATH, 'Projects', 'save_e_elect')
        e_elect_summary_path = os.path.join(project_directory, 'output', 'e_elect_summary.yml')
        self.assertFalse(os.path.isfile(os.path.join(project_directory, 'output', 'e_elect_summary.yml')))
        sched = Scheduler(project='test_save_e_elect',
                          ess_settings=self.ess_settings,
                          project_directory=project_directory,
                          species_list=[ARCSpecies(label='formaldehyde', smiles='C=O'),
                                        ARCSpecies(label='mehylamine', smiles='CN')],
                          freq_scale_factor=1.0,
                          opt_level=Level(method='B3LYP', basis='6-31G(d,p)', software='gaussian'),
                          sp_level=Level(method='B3LYP', basis='6-31G(d,p)', software='gaussian'),
                          job_types={'opt': True, 'fine_grid': False, 'freq': False, 'sp': True, 'rotors': False,
                                     'conf_opt': False, 'conf_sp': False, 'irc': False},
                          report_e_elect=True,
                          testing=True,
                          )
        sched.post_sp_actions(label='formaldehyde',
                              sp_path=os.path.join(ARC_PATH, 'arc', 'testing', 'sp', 'formaldehyde_sp_terachem_output.out'))
        self.assertTrue(os.path.isfile(e_elect_summary_path))
        content = read_yaml_file(e_elect_summary_path)
        self.assertEqual(content, {'formaldehyde': -300621.95378630824})

        sched.post_sp_actions(label='mehylamine',
                              sp_path=os.path.join(ARC_PATH, 'arc', 'testing', 'sp', 'mehylamine_CCSD(T).out'))
        content = read_yaml_file(e_elect_summary_path)
        self.assertEqual(content, {'formaldehyde': -300621.95378630824,
                                    'mehylamine': -251377.49160993524})
        shutil.rmtree(project_directory, ignore_errors=True)

    def test_species_has_geo_sp_freq(self):
        """Test the species_has_geo() / species_has_sp() / species_has_freq() functions."""
        for property_, species_has_property in zip(['geo', 'sp', 'freq'], [species_has_geo, species_has_sp, species_has_freq]):
            species_output_dict = {'paths': {property_: False, 'composite': False}}
            self.assertFalse(species_has_property((species_output_dict)))
            species_output_dict = {'paths': {property_: True, 'composite': False}}
            self.assertTrue(species_has_property((species_output_dict)))
            species_output_dict = {'paths': {property_: False, 'composite': True}}
            self.assertTrue(species_has_property((species_output_dict)))
            species_output_dict = {'paths': {property_: True, 'composite': True}}
            self.assertTrue(species_has_property((species_output_dict)))
        yml_path=os.path.join(ARC_PATH, 'arc', 'testing', 'yml_testing', 'N4H6.yml')
        species_output_dict = {'paths': {'geo': False, 'sp': False, 'freq': False, 'composite': False}}
        self.assertTrue(species_has_freq(species_output_dict=species_output_dict, yml_path=yml_path))
        self.assertTrue(species_has_geo(species_output_dict=species_output_dict, yml_path=yml_path))
        self.assertTrue(species_has_sp(species_output_dict=species_output_dict, yml_path=yml_path))
        self.assertTrue(species_has_sp_and_freq(species_output_dict=species_output_dict, yml_path=yml_path))

    def test_add_label_to_unique_species_labels(self):
        """Test the add_label_to_unique_species_labels() method."""
        self.assertEqual(self.sched2.unique_species_labels, ['methylamine', 'C2H6', 'CtripCO'])
        unique_label = self.sched2.add_label_to_unique_species_labels(label='new_species_15')
        self.assertEqual(unique_label, 'new_species_15')
        self.assertEqual(self.sched2.unique_species_labels, ['methylamine', 'C2H6', 'CtripCO', 'new_species_15'])
        unique_label = self.sched2.add_label_to_unique_species_labels(label='new_species_15')
        self.assertEqual(unique_label, 'new_species_15_0')
        self.assertEqual(self.sched2.unique_species_labels, ['methylamine', 'C2H6', 'CtripCO', 'new_species_15', 'new_species_15_0'])
        unique_label = self.sched2.add_label_to_unique_species_labels(label='new_species_15')
        self.assertEqual(unique_label, 'new_species_15_1')
        self.assertEqual(self.sched2.unique_species_labels, ['methylamine', 'C2H6', 'CtripCO', 'new_species_15', 'new_species_15_0', 'new_species_15_1'])

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        projects = ['arc_project_for_testing_delete_after_usage3', 'arc_project_for_testing_delete_after_usage6']
        for project in projects:
            project_directory = os.path.join(ARC_PATH, 'Projects', project)
            shutil.rmtree(project_directory, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
