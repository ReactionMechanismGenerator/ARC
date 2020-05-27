#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for ARC's common module
"""

import copy
import os
import time
import unittest

import pandas as pd

from rmgpy.molecule.molecule import Molecule

import arc.common as common
from arc.exceptions import InputError, SettingsError
from arc.settings import arc_path, servers
import arc.species.converter as converter


class TestCommon(unittest.TestCase):
    """
    Contains unit tests for ARC's common module
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.default_job_types = {'conformers': True,
                                 'opt': True,
                                 'fine': True,
                                 'freq': True,
                                 'sp': True,
                                 'rotors': True,
                                 'irc': True,
                                 'orbitals': False,
                                 'onedmin': False,
                                 'bde': False,
                                 }

    def test_read_yaml_file(self):
        """Test the read_yaml_file() function"""
        restart_path = os.path.join(arc_path, 'arc', 'testing', 'restart', '1_restart_thermo', 'restart.yml')
        input_dict = common.read_yaml_file(restart_path)
        self.assertIsInstance(input_dict, dict)
        self.assertTrue('reactions' in input_dict)
        self.assertTrue('freq_level' in input_dict)
        self.assertTrue('use_bac' in input_dict)
        self.assertTrue('ts_guess_level' in input_dict)
        self.assertTrue('running_jobs' in input_dict)

        with self.assertRaises(InputError):
            common.read_yaml_file('nopath')

    def test_get_git_commit(self):
        """Test the get_git_commit() function"""
        git_commit = common.get_git_commit()
        # output format: ['fafdb957049917ede565cebc58b29899f597fb5a', 'Fri Mar 29 11:09:50 2019 -0400']
        self.assertEqual(len(git_commit[0]), 40)
        self.assertEqual(len(git_commit[1].split()), 6)
        self.assertIsInstance(git_commit[0], str)
        self.assertIsInstance(git_commit[1], str)

    def test_get_git_branch(self):
        """Test the get_git_branch() function"""
        git_branch = common.get_git_branch()
        self.assertIsInstance(git_branch, str)

    def test_time_lapse(self):
        """Test the time_lapse() function"""
        t0 = time.time()
        time.sleep(2)
        lap = common.time_lapse(t0)
        self.assertEqual(lap, '00:00:02')

    def test_colliding_atoms(self):
        """Check that we correctly determine when atoms collide in xyz"""
        xyz_no_0 = """C	0.0000000	0.0000000	0.6505570"""  # monoatomic
        xyz_no_1 = """C      -0.84339557   -0.03079260   -0.13110478
N       0.53015060    0.44534713   -0.25006000
O       1.33245258   -0.55134720    0.44204567
H      -1.12632103   -0.17824612    0.91628291
H      -1.52529493    0.70480833   -0.56787044
H      -0.97406455   -0.97317212   -0.67214713
H       0.64789210    1.26863944    0.34677470
H       1.98414750   -0.79355889   -0.24492049"""  # no colliding atoms
        xyz_no_2 = """C      0.0 0.0 0.0
H       0.0 0.0 1.09"""  # no colliding atoms
        xyz_no_3 = """N      -0.29070308    0.26322835    0.48770927
N       0.29070351   -0.26323281   -0.48771096
N      -2.61741263    1.38275080    2.63428181
N       2.61742270   -1.38276006   -2.63427425
C      -1.77086206    0.18100754    0.43957605
C       1.77086254   -0.18101028   -0.43957552
C      -2.22486176   -1.28143567    0.45202312
C      -2.30707039    0.92407663   -0.78734681
C       2.30707074   -0.92407071    0.78735246
C       2.22485929    1.28143406   -0.45203080
C      -2.23868798    0.85547218    1.67084736
C       2.23869247   -0.85548109   -1.67084185
H      -1.90398693   -1.81060764   -0.45229645
H      -3.31681639   -1.35858536    0.51240600
H      -1.80714051   -1.81980551    1.31137107
H      -3.40300863    0.95379538   -0.78701415
H      -1.98806037    0.44494681   -1.71978670
H      -1.94802915    1.96005927   -0.81269573
H       1.98805486   -0.44493850    1.71978893
H       1.94803425   -1.96005464    0.81270509
H       3.40300902   -0.95378386    0.78702431
H       1.90398036    1.81061002    0.45228426
H       3.31681405    1.35858667   -0.51241516
H       1.80713611    1.81979843   -1.31138136"""  # check that N=N and C#N do not collide

        self.assertFalse(common.colliding_atoms(converter.str_to_xyz(xyz_no_0)))
        self.assertFalse(common.colliding_atoms(converter.str_to_xyz(xyz_no_1)))
        self.assertFalse(common.colliding_atoms(converter.str_to_xyz(xyz_no_2)))
        self.assertFalse(common.colliding_atoms(converter.str_to_xyz(xyz_no_3)))

        xyz_0 = """C      0.0 0.0 0.0
H       0.0 0.0 0.5"""  # colliding atoms
        xyz_1 = """C      -0.84339557   -0.03079260   -0.13110478
N       0.53015060    0.44534713   -0.25006000
O       1.33245258   -0.55134720    0.44204567
H      -1.12632103   -0.17824612    0.91628291
H      -1.52529493    0.70480833   -0.56787044
H      -0.97406455   -0.97317212   -0.67214713
H       1.33245258   -0.55134720    0.48204567
H       1.98414750   -0.79355889   -0.24492049"""  # colliding atoms
        xyz_2 = """ N                 -0.29070308    0.26322835    0.48770927
 N                  0.29070351   -0.26323281   -0.48771096
 N                 -2.48318439    1.19587180    2.29281971
 N                  2.61742270   -1.38276006   -2.63427425
 C                 -1.77086206    0.18100754    0.43957605
 C                  1.77086254   -0.18101028   -0.43957552
 C                 -2.22486176   -1.28143567    0.45202312
 C                 -2.30707039    0.92407663   -0.78734681
 C                  2.30707074   -0.92407071    0.78735246
 C                  2.22485929    1.28143406   -0.45203080
 C                 -2.23868798    0.85547218    1.67084736
 C                  2.23869247   -0.85548109   -1.67084185
 H                 -1.90398693   -1.81060764   -0.45229645
 H                 -3.31681639   -1.35858536    0.51240600
 H                 -1.80714051   -1.81980551    1.31137107
 H                 -3.40300863    0.95379538   -0.78701415
 H                 -1.98806037    0.44494681   -1.71978670
 H                 -1.94802915    1.96005927   -0.81269573
 H                  1.98805486   -0.44493850    1.71978893
 H                  1.94803425   -1.96005464    0.81270509
 H                  3.40300902   -0.95378386    0.78702431
 H                  1.90398036    1.81061002    0.45228426
 H                  3.31681405    1.35858667   -0.51241516
 H                  1.80713611    1.81979843   -1.31138136"""  # check that C-N collide
        xyz_3 = """ N                 -0.29070308    0.26322835    0.48770927
 N                  0.29070351   -0.26323281   -0.48771096
 N                 -2.61741263    1.38275080    2.63428181
 N                  2.61742270   -1.38276006   -2.63427425
 C                 -1.77086206    0.18100754    0.43957605
 C                  1.77086254   -0.18101028   -0.43957552
 C                 -2.22486176   -1.28143567    0.45202312
 C                 -2.30707039    0.92407663   -0.78734681
 C                  2.30707074   -0.92407071    0.78735246
 C                  2.22485929    1.28143406   -0.45203080
 C                 -2.23868798    0.85547218    1.67084736
 C                  2.23869247   -0.85548109   -1.67084185
 H                 -1.90398693   -1.81060764   -0.45229645
 H                 -2.77266137   -1.32013927    0.48231533
 H                 -1.80714051   -1.81980551    1.31137107
 H                 -3.40300863    0.95379538   -0.78701415
 H                 -1.98806037    0.44494681   -1.71978670
 H                 -1.94802915    1.96005927   -0.81269573
 H                  1.98805486   -0.44493850    1.71978893
 H                  1.94803425   -1.96005464    0.81270509
 H                  3.40300902   -0.95378386    0.78702431
 H                  1.90398036    1.81061002    0.45228426
 H                  3.31681405    1.35858667   -0.51241516
 H                  1.80713611    1.81979843   -1.31138136"""  # check that C-H collide

        self.assertTrue(common.colliding_atoms(converter.str_to_xyz(xyz_0)))
        self.assertTrue(common.colliding_atoms(converter.str_to_xyz(xyz_1)))
        self.assertTrue(common.colliding_atoms(converter.str_to_xyz(xyz_2)))
        self.assertTrue(common.colliding_atoms(converter.str_to_xyz(xyz_3)))

    def test_check_ess_settings(self):
        """Test the check_ess_settings function"""
        server_names = list(servers.keys())
        ess_settings1 = {'gaussian': [server_names[0]], 'molpro': [server_names[1], server_names[0]],
                         'qchem': [server_names[0]], 'orca': [server_names[0]]}
        ess_settings2 = {'gaussian': server_names[0], 'molpro': server_names[1], 'qchem': server_names[0],
                         'orca': [server_names[1]]}
        ess_settings3 = {'gaussian': server_names[0], 'molpro': [server_names[1], server_names[0]],
                         'qchem': server_names[0], 'orca': 'local'}
        ess_settings4 = {'gaussian': server_names[0], 'molpro': server_names[1], 'qchem': server_names[0],
                         'orca': [server_names[1], server_names[0]]}
        ess_settings5 = {'gaussian': 'local', 'molpro': server_names[1], 'qchem': server_names[0],
                         'orca': [server_names[0]], 'terachem': server_names[0]}

        ess_settings1 = common.check_ess_settings(ess_settings1)
        ess_settings2 = common.check_ess_settings(ess_settings2)
        ess_settings3 = common.check_ess_settings(ess_settings3)
        ess_settings4 = common.check_ess_settings(ess_settings4)
        ess_settings5 = common.check_ess_settings(ess_settings5)

        ess_list = [ess_settings1, ess_settings2, ess_settings3, ess_settings4, ess_settings5]

        for ess in ess_list:
            for soft, server_list in ess.items():
                self.assertTrue(soft in ['gaussian', 'molpro', 'orca', 'qchem', 'terachem'])
                self.assertIsInstance(server_list, list)

        with self.assertRaises(SettingsError):
            ess_settings6 = {'nosoft': ['server1']}
            common.check_ess_settings(ess_settings6)
        with self.assertRaises(SettingsError):
            ess_settings7 = {'gaussian': ['noserver']}
            common.check_ess_settings(ess_settings7)

    def test_determine_top_group_indices(self):
        """Test determining the top group in a molecule"""
        mol = Molecule(smiles='c1cc(OC)ccc1OC(CC)SF')
        atom1 = mol.atoms[9]  # this is the C atom at the S, O, H, and C junction
        atom2a = mol.atoms[10]  # C
        atom2b = mol.atoms[8]  # O
        atom2c = mol.atoms[12]  # S
        atom2d = mol.atoms[21]  # H

        top, top_has_heavy_atoms = common.determine_top_group_indices(mol, atom1, atom2a)
        self.assertEqual(len(top), 7)
        self.assertTrue(top_has_heavy_atoms)

        top, top_has_heavy_atoms = common.determine_top_group_indices(mol, atom1, atom2b)
        self.assertEqual(len(top), 16)
        self.assertTrue(top_has_heavy_atoms)

        top, top_has_heavy_atoms = common.determine_top_group_indices(mol, atom1, atom2c)
        self.assertEqual(len(top), 2)
        self.assertTrue(top_has_heavy_atoms)

        top, top_has_heavy_atoms = common.determine_top_group_indices(mol, atom1, atom2d)
        self.assertEqual(top, [22])
        self.assertFalse(top_has_heavy_atoms)  # H

    def test_extermum_list(self):
        """Test the extermum_list() function"""
        lst = []
        min_lst = common.extermum_list(lst)
        self.assertEqual(min_lst, None)

        lst = [None]
        min_lst = common.extermum_list(lst)
        self.assertEqual(min_lst, None)

        lst = [None, None]
        min_lst = common.extermum_list(lst)
        self.assertEqual(min_lst, None)

        lst = [0]
        min_lst = common.extermum_list(lst)
        self.assertEqual(min_lst, 0)

        lst = [-8]
        min_lst = common.extermum_list(lst)
        self.assertEqual(min_lst, -8)

        lst = [-8, -80]
        min_lst = common.extermum_list(lst)
        self.assertEqual(min_lst, -80)

        lst = [-8, None]
        min_lst = common.extermum_list(lst)
        self.assertEqual(min_lst, -8)

        lst = [-8, -8, -8, -8]
        min_lst = common.extermum_list(lst)
        self.assertEqual(min_lst, -8)

        lst = [-8, None, None, 100, -79, None]
        min_lst = common.extermum_list(lst)
        self.assertEqual(min_lst, -79)

        lst = [-8, None, None, 100, -79, None]
        max_lst = common.extermum_list(lst, return_min=False)
        self.assertEqual(max_lst, 100)

    def test_key_by_val(self):
        d = {1: 5, 2: 8}
        self.assertEqual(common.key_by_val(d, 8), 2)

        d = {1: 5, 2: 8, 5: 8}
        self.assertIn(common.key_by_val(d, 8), [2, 5])

        d = {1: 5, 2: None, 3: 9}
        self.assertEqual(common.key_by_val(d, 9), 3)
        self.assertEqual(common.key_by_val(d, None), 2)

        d = {1: 5, 2: 'X', 3: 9}
        self.assertEqual(common.key_by_val(d, 9), 3)
        self.assertEqual(common.key_by_val(d, 'X'), 2)

        with self.assertRaises(ValueError):
            common.key_by_val(d, 10)

    def test_initialize_job_with_given_job_type(self):
        """Test the initialize_job_types() function"""
        job_types = {'conformers': False, 'opt': True, 'fine': True, 'freq': True, 'sp': False, 'rotors': False, 'irc': True}
        job_types_expected = copy.deepcopy(self.default_job_types)
        job_types_expected.update(job_types)
        job_types_initialized = common.initialize_job_types(job_types)
        self.assertEqual(job_types_expected, job_types_initialized)

    def test_initialize_job_with_empty_job_type(self):
        """Test the initialize_job_types() function"""
        job_types = {}
        job_types_expected = copy.deepcopy(self.default_job_types)
        job_types_expected.update(job_types)
        job_types_initialized = common.initialize_job_types(job_types)
        self.assertEqual(job_types_expected, job_types_initialized)

    def test_initialize_job_with_bde_job_type(self):
        """Test the initialize_job_types() function"""
        job_types = {'bde': True}
        job_types_expected = copy.deepcopy(self.default_job_types)
        job_types_expected.update(job_types)
        job_types_initialized = common.initialize_job_types(job_types)
        self.assertEqual(job_types_expected, job_types_initialized)

    def test_initialize_job_with_specific_job_type(self):
        """Test the initialize_job_types() function"""
        specific_job_type = 'freq'
        specific_job_type_expected = {job_type: False for job_type in self.default_job_types.keys()}
        specific_job_type_expected[specific_job_type] = True
        specific_job_type_initialized = common.initialize_job_types({}, specific_job_type=specific_job_type)
        self.assertEqual(specific_job_type_expected, specific_job_type_initialized)

    def test_initialize_job_with_conflict_job_type(self):
        """Test the initialize_job_types() function"""
        specific_job_type = 'sp'
        conflict_job_type = {'bde': True}
        specific_job_type_expected = {job_type: False for job_type in self.default_job_types.keys()}
        specific_job_type_expected[specific_job_type] = True
        specific_job_type_initialized = common.initialize_job_types(conflict_job_type,
                                                                    specific_job_type=specific_job_type)
        self.assertEqual(specific_job_type_expected, specific_job_type_initialized)

    def test_initialize_job_with_specific_bde_job_type(self):
        """Test the initialize_job_types() function"""
        specific_job_type = 'bde'
        bde_default = {'opt': True, 'fine': True, 'freq': True, 'sp': True, 'bde': True}
        specific_job_type_expected = {job_type: False for job_type in self.default_job_types.keys()}
        specific_job_type_expected.update(bde_default)
        specific_job_type_initialized = common.initialize_job_types({}, specific_job_type=specific_job_type)
        self.assertEqual(specific_job_type_expected, specific_job_type_initialized)

    def test_initialize_job_with_not_supported_job_type(self):
        """Test the initialize_job_types() function"""
        with self.assertRaises(InputError):
            job_types = {'fake_job': True}
            common.initialize_job_types(job_types)
        with self.assertRaises(InputError):
            specific_job_type = 'fake_job_type'
            common.initialize_job_types({}, specific_job_type=specific_job_type)

    def test_determine_ess(self):
        """Test the determine_ess function"""
        gaussian = os.path.join(arc_path, 'arc', 'testing', 'composite', 'SO2OO_CBS-QB3.log')
        qchem = os.path.join(arc_path, 'arc', 'testing', 'freq', 'C2H6_freq_QChem.out')
        molpro = os.path.join(arc_path, 'arc', 'testing', 'freq', 'CH2O_freq_molpro.out')

        self.assertEqual(common.determine_ess(gaussian), 'gaussian')
        self.assertEqual(common.determine_ess(qchem), 'qchem')
        self.assertEqual(common.determine_ess(molpro), 'molpro')

    def test_sort_two_lists_by_the_first(self):
        """Test the sort_two_lists_by_the_first function"""
        list1 = [5, 2, 8, 1, 0]
        list2 = ['D', 'C', 'E', 'B', 'A']
        list1, list2 = common.sort_two_lists_by_the_first(list1, list2)
        self.assertEqual(list1, [0, 1, 2, 5, 8])
        self.assertEqual(list2, ['A', 'B', 'C', 'D', 'E'])

        list1 = [-402175.42413054925, -402175.42413054925, -402175.42413054925]
        list2 = [
            {'symbols': ('O', 'C', 'C', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1),
             'coords': ((1.289877, -1.027889, -0.472673), (-0.705088, 0.018405, 0.167196),
                        (0.681617, 0.004981, -0.157809), (-1.21258, 0.938155, 0.439627),
                        (-1.264105, -0.911112, 0.141353), (1.210279, 0.97746, -0.117693))},
            {'symbols': ('O', 'C', 'C', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1),
             'coords': ((1.438259, -0.905929, 0.233373), (-0.722557, -0.021223, 0.05388),
                        (0.695838, 0.042156, -0.059689), (-1.184869, -0.935184, 0.411808),
                        (-1.348775, 0.824904, -0.209994), (1.122104, 0.995276, -0.429377))},
            {'symbols': ('O', 'C', 'C', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1),
             'coords': ((-1.097258, 1.288817, 0.280596), (0.66579, -0.163586, -0.235383),
                        (-0.646392, 0.135474, 0.230984), (1.308294, 0.646397, -0.564452),
                        (1.03492, -1.183579, -0.268027), (-1.265354, -0.723524, 0.556282))}]
        list1, list2 = common.sort_two_lists_by_the_first(list1, list2)
        self.assertEqual(list1, [-402175.42413054925, -402175.42413054925, -402175.42413054925])
        self.assertEqual(list2, [
            {'symbols': ('O', 'C', 'C', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1),
             'coords': ((1.289877, -1.027889, -0.472673), (-0.705088, 0.018405, 0.167196),
                        (0.681617, 0.004981, -0.157809), (-1.21258, 0.938155, 0.439627),
                        (-1.264105, -0.911112, 0.141353), (1.210279, 0.97746, -0.117693))},
            {'symbols': ('O', 'C', 'C', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1),
             'coords': ((1.438259, -0.905929, 0.233373), (-0.722557, -0.021223, 0.05388),
                        (0.695838, 0.042156, -0.059689), (-1.184869, -0.935184, 0.411808),
                        (-1.348775, 0.824904, -0.209994), (1.122104, 0.995276, -0.429377))},
            {'symbols': ('O', 'C', 'C', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1),
             'coords': ((-1.097258, 1.288817, 0.280596), (0.66579, -0.163586, -0.235383),
                        (-0.646392, 0.135474, 0.230984), (1.308294, 0.646397, -0.564452),
                        (1.03492, -1.183579, -0.268027), (-1.265354, -0.723524, 0.556282))}])

        list1 = [-293712.44825524034, -293712.4484442763, -293719.9392868027]
        list2 = [
            {'symbols': ('N', 'N', 'H', 'H', 'H', 'H'), 'isotopes': (14, 14, 1, 1, 1, 1),
             'coords': ((0.626959, 0.049055, 0.398851), (-0.627579, -0.002911, -0.40088),
                        (1.135191, 0.833831, -0.015525), (1.156572, -0.755054, 0.054181),
                        (-1.135811, -0.787687, 0.013496), (-1.157192, 0.801198, -0.056209))},
            {'symbols': ('N', 'N', 'H', 'H', 'H', 'H'), 'isotopes': (14, 14, 1, 1, 1, 1),
             'coords': ((0.650715, 0.034647, 0.359727), (-0.650716, -0.034614, -0.359727),
                        (1.181164, -0.741554, -0.042786), (1.107, 0.847165, -0.061582),
                        (-1.106969, -0.847181, 0.061521), (-1.181193, 0.741537, 0.042846))},
            {'symbols': ('N', 'N', 'H', 'H', 'H', 'H'), 'isotopes': (14, 14, 1, 1, 1, 1),
             'coords': ((-0.679412, 0.224819, -0.236977), (0.680753, -0.233287, -0.22462),
                        (-0.974999, 0.570328, 0.676502), (-1.260937, -0.580211, -0.451942),
                        (1.263604, 0.56338, -0.465582), (0.970991, -0.54503, 0.702618))}]
        list1, list2 = common.sort_two_lists_by_the_first(list1, list2)
        self.assertEqual(list1, [-293719.9392868027, -293712.4484442763, -293712.44825524034])
        self.assertEqual(list2, [
            {'isotopes': (14, 14, 1, 1, 1, 1), 'symbols': ('N', 'N', 'H', 'H', 'H', 'H'),
             'coords': ((-0.679412, 0.224819, -0.236977), (0.680753, -0.233287, -0.22462),
                        (-0.974999, 0.570328, 0.676502), (-1.260937, -0.580211, -0.451942),
                        (1.263604, 0.56338, -0.465582), (0.970991, -0.54503, 0.702618))},
            {'isotopes': (14, 14, 1, 1, 1, 1), 'symbols': ('N', 'N', 'H', 'H', 'H', 'H'),
             'coords': ((0.650715, 0.034647, 0.359727), (-0.650716, -0.034614, -0.359727),
                        (1.181164, -0.741554, -0.042786), (1.107, 0.847165, -0.061582),
                        (-1.106969, -0.847181, 0.061521), (-1.181193, 0.741537, 0.042846))},
            {'isotopes': (14, 14, 1, 1, 1, 1), 'symbols': ('N', 'N', 'H', 'H', 'H', 'H'),
             'coords': ((0.626959, 0.049055, 0.398851), (-0.627579, -0.002911, -0.40088),
                        (1.135191, 0.833831, -0.015525), (1.156572, -0.755054, 0.054181),
                        (-1.135811, -0.787687, 0.013496), (-1.157192, 0.801198, -0.056209))}])

        list1 = [5, None, 1]
        list2 = [1, 2, 3]
        list1, list2 = common.sort_two_lists_by_the_first(list1, list2)
        self.assertEqual(list1, [1, 5])
        self.assertEqual(list2, [3, 1])

        list1 = [None, None, None]
        list2 = [1, 2, 3]
        list1, list2 = common.sort_two_lists_by_the_first(list1, list2)
        self.assertEqual(list1, [])
        self.assertEqual(list2, [])

    def test_determine_model_chemistry_type(self):
        """Test that the type (e.g., DFT, wavefunction ...) of a model chemistry can be determined properly."""

        # The special case: has `hf` keyword but is a DFT method
        method = 'm06-hf'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'dft'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

        # Test family of wavefunction methods
        method = 'hf'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'wavefunction'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

        method = 'DLPNO-CCSD(T)'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'wavefunction'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

        method = 'DLPNO-MP2-F12'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'wavefunction'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

        method = 'QCISD'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'wavefunction'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

        # Test family of force field (a.k.a molecular dynamics) methods
        method = 'ANI-1x'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'force_field'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

        method = 'MMFF94'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'force_field'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

        # Test family of semi-empirical methods
        method = 'ZINDO/S'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'semiempirical'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

        method = 'pm7'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'semiempirical'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

        # Test family of DFT methods
        method = 'mPW1PW'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'dft'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

        method = 'b3lyp'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'dft'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

        method = 'wb97x-d3'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'dft'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

        method = 'apfd'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'dft'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

        method = 'M06-2X'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'dft'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

        method = 'B2PLYP'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'dft'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

        method = 'CBS-QB3'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'composite'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

        method = 'G4'
        model_chemistry_class = common.determine_model_chemistry_type(method)
        model_chemistry_class_expected = 'composite'
        self.assertEqual(model_chemistry_class, model_chemistry_class_expected)

    def test_format_level_of_theory_inputs(self):
        """Test formatting the job model chemistry inputs"""
        # Test illegal input (list)
        with self.assertRaises(InputError):
            common.format_level_of_theory_inputs(['b3lyp', 'def2tzvp'])

        # Test illegal input (not exactly three pipes)
        with self.assertRaises(InputError):
            common.format_level_of_theory_inputs('b3lyp|def2tzvp')
        with self.assertRaises(InputError):
            common.format_level_of_theory_inputs('wb97xd|def2tzvp|||')

        # Test illegal input (empty space)
        with self.assertRaises(InputError):
            common.format_level_of_theory_inputs('b3 lyp')
        with self.assertRaises(InputError):
            common.format_level_of_theory_inputs('dlpno-ccsd(t)/def2-svp def2-svp/c')
        with self.assertRaises(InputError):
            common.format_level_of_theory_inputs('dlpno-ccsd(t)/def2-svp aug-def2-svp')

        # Test illegal input (multiple slashes)
        with self.assertRaises(InputError):
            common.format_level_of_theory_inputs('dlpno-ccsd(t)/def2-svp/def2-svp/c')
        with self.assertRaises(InputError):
            common.format_level_of_theory_inputs('b3lyp/def2-svp/aug-def2-svp')

        # Test illegal input ('method' is not a key)
        with self.assertRaises(InputError):
            common.format_level_of_theory_inputs({'basis': '6-31g'})

        # Test illegal input (illegal key)
        with self.assertRaises(InputError):
            common.format_level_of_theory_inputs({'random': 'something'})

        # Test parsing string inputs
        output_dict_0, output_str_0 = common.format_level_of_theory_inputs('cbs-qb3')
        expected_dict_0 = {'method': 'cbs-qb3', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}
        expected_str_0 = 'cbs-qb3|||'
        self.assertEqual(output_dict_0, expected_dict_0)
        self.assertEqual(output_str_0, expected_str_0)

        output_dict_1, output_str_1 = common.format_level_of_theory_inputs('b3lyp/def2-TZVP')
        expected_dict_1 = {'method': 'b3lyp', 'basis': 'def2-tzvp', 'auxiliary_basis': '', 'dispersion': ''}
        expected_str_1 = 'b3lyp|def2-tzvp||'
        self.assertEqual(output_dict_1, expected_dict_1)
        self.assertEqual(output_str_1, expected_str_1)

        output_dict_2, output_str_2 = common.format_level_of_theory_inputs('|||')
        expected_dict_2 = {'method': '', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}
        expected_str_2 = '|||'
        self.assertEqual(output_dict_2, expected_dict_2)
        self.assertEqual(output_str_2, expected_str_2)

        output_dict_3, output_str_3 = common.format_level_of_theory_inputs('b3lyp|def2tzvp||')
        expected_dict_3 = {'method': 'b3lyp', 'basis': 'def2tzvp', 'auxiliary_basis': '', 'dispersion': ''}
        expected_str_3 = 'b3lyp|def2tzvp||'
        self.assertEqual(output_dict_3, expected_dict_3)
        self.assertEqual(output_str_3, expected_str_3)

        output_dict_4, output_str_4 = common.format_level_of_theory_inputs('b3lyp|def2tzvp|aug-def2-svp|gd3bj')
        expected_dict_4 = {'method': 'b3lyp', 'basis': 'def2tzvp', 'auxiliary_basis': 'aug-def2-svp',
                           'dispersion': 'gd3bj'}
        expected_str_4 = 'b3lyp|def2tzvp|aug-def2-svp|gd3bj'
        self.assertEqual(output_dict_4, expected_dict_4)
        self.assertEqual(output_str_4, expected_str_4)

        output_dict_5, output_str_5 = common.format_level_of_theory_inputs('b3lyp|def2tzvp||gd3bj')
        expected_dict_5 = {'method': 'b3lyp', 'basis': 'def2tzvp', 'auxiliary_basis': '', 'dispersion': 'gd3bj'}
        expected_str_5 = 'b3lyp|def2tzvp||gd3bj'
        self.assertEqual(output_dict_5, expected_dict_5)
        self.assertEqual(output_str_5, expected_str_5)

        # Test parsing dictionary inputs
        output_dict_6, output_str_6 = common.format_level_of_theory_inputs({'method': 'wb97xd', 'basis': '6-31g'})
        expected_dict_6 = {'method': 'wb97xd', 'basis': '6-31g', 'auxiliary_basis': '', 'dispersion': ''}
        expected_str_6 = 'wb97xd|6-31g||'
        self.assertEqual(output_dict_6, expected_dict_6)
        self.assertEqual(output_str_6, expected_str_6)

        output_dict_7, output_str_7 = common.format_level_of_theory_inputs({'method': 'b3lyp', 'basis': 'def2tzvp',
                                                                            'auxiliary_basis': 'aug-def2-svp',
                                                                            'dispersion': 'gd3bj'})
        expected_dict_7 = {'method': 'b3lyp', 'basis': 'def2tzvp', 'auxiliary_basis': 'aug-def2-svp',
                           'dispersion': 'gd3bj'}
        expected_str_7 = 'b3lyp|def2tzvp|aug-def2-svp|gd3bj'
        self.assertEqual(output_dict_7, expected_dict_7)
        self.assertEqual(output_str_7, expected_str_7)

        # Test parsing empty inputs
        output_dict_8, output_str_8 = common.format_level_of_theory_inputs('')
        expected_dict_8 = {'method': '', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}
        expected_str_8 = ''
        self.assertEqual(output_dict_8, expected_dict_8)
        self.assertEqual(output_str_8, expected_str_8)

        output_dict_9, output_str_9 = common.format_level_of_theory_inputs({'method': '', 'basis': '',
                                                                            'auxiliary_basis': '', 'dispersion': ''})
        expected_dict_9 = {}
        expected_str_9 = ''
        self.assertEqual(output_dict_9, expected_dict_9)
        self.assertEqual(output_str_9, expected_str_9)

        output_dict_10, output_str_10 = common.format_level_of_theory_inputs({})
        expected_dict_10 = {'method': '', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}
        expected_str_10 = ''
        self.assertEqual(output_dict_10, expected_dict_10)
        self.assertEqual(output_str_10, expected_str_10)

    def test_format_level_of_theory_for_logging(self):
        """Test format level of theory dictionary to string for logging purposes."""
        level_of_theory = {}
        expected_str = ''
        formatted_str = common.format_level_of_theory_for_logging(level_of_theory)
        self.assertEqual(expected_str, formatted_str)

        level_of_theory = {'method': 'cbs-qb3'}
        expected_str = 'cbs-qb3'
        formatted_str = common.format_level_of_theory_for_logging(level_of_theory)
        self.assertEqual(expected_str, formatted_str)

        level_of_theory = {'method': 'cbs-qb3', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}
        expected_str = 'cbs-qb3'
        formatted_str = common.format_level_of_theory_for_logging(level_of_theory)
        self.assertEqual(expected_str, formatted_str)

        level_of_theory = {'method': 'apfd', 'basis': 'def2svp', 'auxiliary_basis': '', 'dispersion': ''}
        expected_str = 'apfd/def2svp'
        formatted_str = common.format_level_of_theory_for_logging(level_of_theory)
        self.assertEqual(expected_str, formatted_str)

        level_of_theory = {'method': 'b3lyp', 'basis': '6-31g', 'auxiliary_basis': '', 'dispersion': 'gd3bj'}
        expected_str = 'b3lyp/6-31g gd3bj'
        formatted_str = common.format_level_of_theory_for_logging(level_of_theory)
        self.assertEqual(expected_str, formatted_str)

        level_of_theory = {'method': 'DLPNO-CCSD(T)-F12', 'basis': 'cc-pVTZ-F12',
                           'auxiliary_basis': 'aug-cc-pVTZ/C cc-pVTZ-F12-CABS', 'dispersion': ''}
        expected_str = 'dlpno-ccsd(t)-f12/cc-pvtz-f12/aug-cc-pvtz/c cc-pvtz-f12-cabs'
        formatted_str = common.format_level_of_theory_for_logging(level_of_theory)
        self.assertEqual(expected_str, formatted_str)

        level_of_theory = 'PM6'
        expected_str = 'pm6'
        formatted_str = common.format_level_of_theory_for_logging(level_of_theory)
        self.assertEqual(expected_str, formatted_str)

        level_of_theory = 'wb97xd3/6-31G+'
        expected_str = 'wb97xd3/6-31g+'
        formatted_str = common.format_level_of_theory_for_logging(level_of_theory)
        self.assertEqual(expected_str, formatted_str)

    def test_is_notebook(self):
        """Test whether ARC is being called from an IPython notebook"""
        is_notebook = common.is_notebook()
        self.assertFalse(is_notebook)

    def test_is_str_float(self):
        """Test the is_str_float() function"""
        self.assertTrue(common.is_str_float('123'))
        self.assertTrue(common.is_str_float('.2'))
        self.assertTrue(common.is_str_float('125.84'))
        self.assertTrue(common.is_str_float('6e02'))
        self.assertTrue(common.is_str_float('+1e1'))
        self.assertFalse(common.is_str_float('+1e1e'))
        self.assertFalse(common.is_str_float('text 34'))
        self.assertFalse(common.is_str_float(' '))
        self.assertFalse(common.is_str_float('R1'))
        self.assertFalse(common.is_str_float('D_3_5_7_4'))

    def test_is_str_int(self):
        """Test the is_str_int() function"""
        self.assertTrue(common.is_str_int('0'))
        self.assertTrue(common.is_str_int('123'))
        self.assertTrue(common.is_str_int('+123'))
        self.assertTrue(common.is_str_int('-123'))
        self.assertFalse(common.is_str_int('6e02'))
        self.assertFalse(common.is_str_int('+1e1'))
        self.assertFalse(common.is_str_int('+1e1e'))
        self.assertFalse(common.is_str_int('text 34'))
        self.assertFalse(common.is_str_int(' '))
        self.assertFalse(common.is_str_int('R1'))
        self.assertFalse(common.is_str_int('D_3_5_7_4'))
        self.assertFalse(common.is_str_int('.2'))
        self.assertFalse(common.is_str_int('.0'))
        self.assertFalse(common.is_str_int('125.84'))
        self.assertFalse(common.is_str_int('0.0'))


    def test_get_atom_radius(self):
        """Test determining the covalent radius of an atom"""
        self.assertEqual(common.get_atom_radius('C'), 0.76)
        self.assertEqual(common.get_atom_radius('S'), 1.05)
        self.assertEqual(common.get_atom_radius('O'), 0.66)
        self.assertEqual(common.get_atom_radius('H'), 0.31)
        self.assertEqual(common.get_atom_radius('N'), 0.71)
        self.assertIsNone(common.get_atom_radius('wrong'))

    def test_get_single_bond_length(self):
        """Test getting an approximation for a single bond length"""
        self.assertEqual(common.get_single_bond_length('C', 'C'), 1.54)
        self.assertEqual(common.get_single_bond_length('C', 'O'), 1.43)
        self.assertEqual(common.get_single_bond_length('O', 'C'), 1.43)
        self.assertEqual(common.get_single_bond_length('P', 'Si'), 2.5)

    def test_globalize_paths(self):
        """Test modifying a file's contents to correct absolute file paths"""
        project_directory = os.path.join(arc_path, 'arc', 'testing', 'restart')
        restart_path = os.path.join(arc_path, 'arc', 'testing', 'restart', 'restart_paths.yml')
        common.globalize_paths(file_path=restart_path, project_directory=project_directory)
        globalized_restart_path = os.path.join(arc_path, 'arc', 'testing', 'restart', 'restart_paths_globalized.yml')
        content = common.read_yaml_file(globalized_restart_path)
        self.assertEqual(content['restart'], 'Restarted ARC at 2020-02-28 12:51:14.446086; ')
        self.assertIn('ARC/arc/testing/restart/calcs/Species/HCN/freq_a38229/output.out', content['paths']['freq'])
        self.assertIn('ARC/arc/testing/restart/calcs/Species/HCN/sp_a38230/output.out', content['paths']['sp'])
        self.assertNotIn('gpfs/workspace/users/user', content['paths']['freq'])
        self.assertNotIn('gpfs/workspace/users/user', content['paths']['sp'])

    def test_globalize_path(self):
        """Test rebasing a single path to the current ARC project"""
        project_directory = 'project/directory/'
        for string in ['/gpfs/workspace/users/user/runs/ARC/Soup/R_Add_HCN/calcs/Species/HCN/sp_a38230/output.out',
                       '/gpfs/workspace/users/user/runs/ARC/Soup/R_Add_HCN/calcs/TSs/HCN/sp_a38230/output.out']:
            globalized_string = common.globalize_path(string=string, project_directory=project_directory)
            self.assertIn('project/directory/calcs/', globalized_string)
            self.assertIn('/HCN/sp_a38230/output.out', globalized_string)
            self.assertFalse('/gpfs/workspace/users/user' in globalized_string)

        string = 'some non-path string'
        globalized_string = common.globalize_path(string=string, project_directory='')
        self.assertEqual(globalized_string, string)

    def test_estimate_orca_mem_cpu_requirement(self):
        """Test estimating memory and cpu requirements for an Orca job."""
        num_heavy_atoms_0 = 0
        est_cpu_0, est_memory_0 = common.estimate_orca_mem_cpu_requirement(num_heavy_atoms_0)
        expected_cpu_0, expected_memory_0 = 2, 4000.0
        self.assertEqual(est_cpu_0, expected_cpu_0)
        self.assertEqual(est_memory_0, expected_memory_0)

        num_heavy_atoms_1 = 12
        est_cpu_1, est_memory_1 = common.estimate_orca_mem_cpu_requirement(num_heavy_atoms_1)
        expected_cpu_1, expected_memory_1 = 50, 100000.0
        self.assertEqual(est_cpu_1, expected_cpu_1)
        self.assertEqual(est_memory_1, expected_memory_1)

        num_heavy_atoms_2 = 12
        est_cpu_2, est_memory_2 = common.estimate_orca_mem_cpu_requirement(num_heavy_atoms_2, 'server2', True)
        expected_cpu_2, expected_memory_2 = 48, 96000.0
        self.assertEqual(est_cpu_2, expected_cpu_2)
        self.assertEqual(est_memory_2, expected_memory_2)

    def test_check_torsion_change(self):
        """Test checking if torsion changes are significant"""
        ics = {0: [120], 1: [126], 2: [176], 3: [-174]}
        torsions = pd.DataFrame(data=ics, index=['D1'])
        self.assertFalse(common.check_torsion_change(
            torsions, 1, 0, threshold=20.0, delta=0.0)['D1'])
        self.assertTrue(common.check_torsion_change(
            torsions, 1, 0, threshold=5.0, delta=0.0)['D1'])
        self.assertFalse(common.check_torsion_change(
            torsions, 1, 0, threshold=5.0, delta=8.0)['D1'])
        self.assertTrue(common.check_torsion_change(
            torsions, 2, 1, threshold=20.0, delta=8.0)['D1'])
        self.assertTrue(common.check_torsion_change(
            torsions, 3, 1, threshold=20.0, delta=8.0)['D1'])
        self.assertFalse(common.check_torsion_change(
            torsions, 3, 2, threshold=20.0, delta=8.0)['D1'])

    def test_is_same_pivot(self):
        """Test whether two torsions have the same pivot"""
        self.assertTrue(common.is_same_pivot([1, 2, 3, 4], [5, 2, 3, 4]))
        self.assertTrue(common.is_same_pivot([1, 2, 3, 4], [4, 3, 2, 5]))
        self.assertFalse(common.is_same_pivot([1, 2, 3, 4], [5, 4, 3, 2]))
        self.assertTrue(common.is_same_pivot("[1, 2, 3, 4]", [5, 2, 3, 4]))
        self.assertTrue(common.is_same_pivot([1, 2, 3, 4], "[4, 3, 2, 5]"))
        self.assertFalse(common.is_same_pivot("[1, 2, 3, 4]", "[5, 4, 3, 2]"))

    def test_is_same_sequence_sublist(self):
        """Test whether a sequence appears in a list"""
        self.assertTrue(common.is_same_sequence_sublist([1, 2, 3], [1, 2, 3, 4]))
        self.assertTrue(common.is_same_sequence_sublist([2, 3, 4], [1, 2, 3, 4]))
        self.assertFalse(common.is_same_sequence_sublist([1, 3, 4], [1, 2, 3, 4]))
        self.assertFalse(common.is_same_sequence_sublist([4, 3, 2], [1, 2, 3, 4]))

    def test_get_ordered_intersection_of_two_lists(self):
        """Test get ordered intersection of two lists."""
        l1 = [1, 2, 3, 3, 5, 6]
        l2 = [6, 3, 5, 5, 1]

        l3_out_0 = common.get_ordered_intersection_of_two_lists(l1, l2, order_by_first_list=True, return_unique=True)
        l3_expected_0 = [1, 3, 5, 6]
        self.assertEqual(l3_out_0, l3_expected_0)

        l3_out_1 = common.get_ordered_intersection_of_two_lists(l1, l2, order_by_first_list=True, return_unique=False)
        l3_expected_1 = [1, 3, 3, 5, 6]
        self.assertEqual(l3_out_1, l3_expected_1)

        l3_out_2 = common.get_ordered_intersection_of_two_lists(l1, l2, order_by_first_list=False, return_unique=True)
        l3_expected_2 = [6, 3, 5, 1]
        self.assertEqual(l3_out_2, l3_expected_2)

        l3_out_3 = common.get_ordered_intersection_of_two_lists(l1, l2, order_by_first_list=False, return_unique=False)
        l3_expected_3 = [6, 3, 5, 5, 1]
        self.assertEqual(l3_out_3, l3_expected_3)

        l1 = [1, 2, 3, 3, 5, 6]
        l2 = [7]
        self.assertEqual(common.get_ordered_intersection_of_two_lists(l1, l2), list())


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
