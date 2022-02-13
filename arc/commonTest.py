#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for ARC's common module
"""

import copy
import datetime
import os
import time
import unittest

import numpy as np
import pandas as pd

from rmgpy.molecule.molecule import Molecule

import arc.common as common
from arc.exceptions import InputError, SettingsError
from arc.imports import settings
from arc.reaction import ARCReaction
from arc.rmgdb import make_rmg_database_object, load_families_only
from arc.species.mapping import get_rmg_reactions_from_arc_reaction
from arc.species.species import ARCSpecies
import arc.species.converter as converter
from arc.reaction import ARCReaction
from arc.species.species import ARCSpecies


servers = settings['servers']


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
        cls.rmgdb = make_rmg_database_object()
        load_families_only(cls.rmgdb)
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
        restart_path = os.path.join(common.ARC_PATH, 'arc', 'testing', 'restart', '1_restart_thermo', 'restart.yml')
        input_dict = common.read_yaml_file(restart_path)
        self.assertIsInstance(input_dict, dict)
        self.assertTrue('reactions' in input_dict)
        self.assertTrue('freq_level' in input_dict)
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

    def test_extremum_list(self):
        """Test the extremum_list() function"""
        lst = []
        min_lst = common.extremum_list(lst)
        self.assertEqual(min_lst, None)

        lst = [None]
        min_lst = common.extremum_list(lst)
        self.assertEqual(min_lst, None)

        lst = [None, None]
        min_lst = common.extremum_list(lst)
        self.assertEqual(min_lst, None)

        lst = [0]
        min_lst = common.extremum_list(lst)
        self.assertEqual(min_lst, 0)

        lst = [-8]
        min_lst = common.extremum_list(lst)
        self.assertEqual(min_lst, -8)

        lst = [-8, -80]
        min_lst = common.extremum_list(lst)
        self.assertEqual(min_lst, -80)

        lst = [-8, None]
        min_lst = common.extremum_list(lst)
        self.assertEqual(min_lst, -8)

        lst = [-8, -8, -8, -8]
        min_lst = common.extremum_list(lst)
        self.assertEqual(min_lst, -8)

        lst = [-8, None, None, 100, -79, None]
        min_lst = common.extremum_list(lst)
        self.assertEqual(min_lst, -79)

        lst = [-8, None, None, 100, -79, None]
        max_lst = common.extremum_list(lst, return_min=False)
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

        d = {1: 5, 2: 'X8', 3: 9}
        self.assertEqual(common.key_by_val(d, 8), 2)

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
        gaussian = os.path.join(common.ARC_PATH, 'arc', 'testing', 'composite', 'SO2OO_CBS-QB3.log')
        qchem = os.path.join(common.ARC_PATH, 'arc', 'testing', 'freq', 'C2H6_freq_QChem.out')
        molpro = os.path.join(common.ARC_PATH, 'arc', 'testing', 'freq', 'CH2O_freq_molpro.out')

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
        self.assertEqual(common.get_single_bond_length('N', 'N'), 1.45)
        self.assertEqual(common.get_single_bond_length('N', 'N', 1, 1), 1.81)
        self.assertEqual(common.get_single_bond_length('N', 'O', 1, -1), 1.2)

    def test_globalize_paths(self):
        """Test modifying a file's contents to correct absolute file paths"""
        project_directory = os.path.join(common.ARC_PATH, 'arc', 'testing', 'restart', '4_globalized_paths')
        restart_path = os.path.join(project_directory, 'restart_paths.yml')
        common.globalize_paths(file_path=restart_path, project_directory=project_directory)
        globalized_restart_path = os.path.join(project_directory, 'restart_paths_globalized.yml')
        content = common.read_yaml_file(globalized_restart_path)
        self.assertEqual(content['output']['restart'], 'Restarted ARC at 2020-02-28 12:51:14.446086; ')
        self.assertIn('ARC/arc/testing/restart/4_globalized_paths/calcs/Species/HCN/freq_a38229/output.out',
                      content['output']['spc']['paths']['freq'])
        self.assertNotIn('gpfs/workspace/users/user', content['output']['spc']['paths']['freq'])

        path = '/home/user/runs/ARC/ARC_Project/calcs/Species/H/sp_a4339/output.out'
        new_path = common.globalize_path(path, project_directory)
        self.assertIn('/ARC/arc/testing/restart/4_globalized_paths/calcs/Species/H/sp_a4339/output.out', new_path)

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
        expected_cpu_2, expected_memory_2 = 24, 48000.0
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
        """Test get ordered intersection of two lists"""
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

    def test_get_angle_in_180_range(self):
        """Test the getting a corresponding angle in the -180 to +180 range"""
        self.assertEqual(common.get_angle_in_180_range(0), 0)
        self.assertEqual(common.get_angle_in_180_range(10), 10)
        self.assertEqual(common.get_angle_in_180_range(-5.364589, round_to=None), -5.364589)
        self.assertEqual(common.get_angle_in_180_range(-5.364589), -5.36)
        self.assertEqual(common.get_angle_in_180_range(-120), -120)
        self.assertEqual(common.get_angle_in_180_range(179.999), 180)
        self.assertEqual(common.get_angle_in_180_range(180), -180)
        self.assertEqual(common.get_angle_in_180_range(-180), -180)
        self.assertEqual(common.get_angle_in_180_range(181), -179)
        self.assertEqual(common.get_angle_in_180_range(360), 0)
        self.assertEqual(common.get_angle_in_180_range(362), 2)
        self.assertEqual(common.get_angle_in_180_range(1000), -80)
        self.assertEqual(common.get_angle_in_180_range(-1000), 80)
        self.assertEqual(common.get_angle_in_180_range(247.62), -112.38)

    def test_get_close_tuple(self):
        """Test getting a close tuple of strings from a list of tuples for a given tuple"""
        keys = [('121.1', '-180.0'), ('129.1', '-180.0'), ('137.1', '-180.0'), ('145.1', '-180.0'), ('153.1', '-180.0'),
                ('161.1', '-180.0'), ('169.1', '-180.0'), ('177.1', '-180.0'), ('-174.9', '-180.0'),
                ('-166.9', '-180.0'), ('-158.9', '-180.0')]
        self.assertEqual(('161.1', '-180.0'), common.get_close_tuple(key_1=('161.1', '-180.0'), keys=keys))
        self.assertEqual(('161.1', '-180.0'), common.get_close_tuple(key_1=('161.12', '-180.0'), keys=keys))
        self.assertEqual(('161.1', '-180.0'), common.get_close_tuple(key_1=('161.075', '-180.0'), keys=keys))
        self.assertEqual(('161.1', '-180.0'), common.get_close_tuple(key_1=(161.075, '-180.0'), keys=keys))
        self.assertEqual(('161.1', '-180.0'), common.get_close_tuple(key_1=(161.075, -180.03), keys=keys))
        self.assertEqual(('161.1', '-150.0'), common.get_close_tuple(key_1=('161.12', '-150.0'), keys=keys))
        self.assertIsNone(common.get_close_tuple(key_1=(1.075, -150.03), keys=keys))
        with self.assertRaises(ValueError):
            common.get_close_tuple(key_1=(1.075, -150.03), keys=keys, raise_error=True)

    def test_get_number_with_ordinal_indicator(self):
        """Test the get_number_with_ordinal_indicator() function"""
        self.assertEqual(common.get_number_with_ordinal_indicator(1), '1st')
        self.assertEqual(common.get_number_with_ordinal_indicator(2), '2nd')
        self.assertEqual(common.get_number_with_ordinal_indicator(3), '3rd')
        self.assertEqual(common.get_number_with_ordinal_indicator(4), '4th')
        self.assertEqual(common.get_number_with_ordinal_indicator(50), '50th')
        self.assertEqual(common.get_number_with_ordinal_indicator(23), '23rd')
        self.assertEqual(common.get_number_with_ordinal_indicator(31), '31st')
        self.assertEqual(common.get_number_with_ordinal_indicator(22), '22nd')
        self.assertEqual(common.get_number_with_ordinal_indicator(100), '100th')

    def test_timedelta_from_str(self):
        """Test reconstructing a timedelta object from its string representation"""
        t0 = datetime.datetime.now()
        time.sleep(0.5)
        delta = datetime.datetime.now() - t0
        str_delta = str(delta)
        self.assertIn('0:00:00.5', str_delta)
        reconstructed_delta = common.timedelta_from_str(str_delta)
        self.assertIsInstance(reconstructed_delta, datetime.timedelta)

    def test_torsions_to_scans(self):
        """Test the torsions_to_scans() function"""
        self.assertEqual(common.torsions_to_scans([0, 1, 2, 3]), [[1, 2, 3, 4]])
        self.assertEqual(common.torsions_to_scans([1, 2, 3, 4], direction=-1), [[0, 1, 2, 3]])
        self.assertEqual(common.torsions_to_scans([1, 2, 3, 4], direction=2), [[0, 1, 2, 3]])
        self.assertEqual(common.torsions_to_scans([[0, 1, 2, 3], [5, 7, 8, 9]]), [[1, 2, 3, 4], [6, 8, 9, 10]])
        with self.assertRaises(TypeError):
            common.torsions_to_scans('4, 3, 5, 6')
        with self.assertRaises(ValueError):
            common.torsions_to_scans([[0, 1, 2, 3], [6, 8, 9, 10]], direction=-1)

    def test_convert_list_index_0_to_1(self):
        """Test the convert_list_index_0_to_1() function"""
        self.assertEqual(common.convert_list_index_0_to_1([]), [])
        self.assertEqual(common.convert_list_index_0_to_1([0]), [1])
        self.assertEqual(common.convert_list_index_0_to_1([1], direction=-1), [0])
        self.assertEqual(common.convert_list_index_0_to_1([0, 5, 8]), [1, 6, 9])  # test list
        self.assertEqual(common.convert_list_index_0_to_1((0, 5, 8), direction=1), (1, 6, 9))  # test tuple
        self.assertEqual(common.convert_list_index_0_to_1([1, 5, 8], direction=-1), [0, 4, 7])
        with self.assertRaises(ValueError):
            common.convert_list_index_0_to_1([-9])
        with self.assertRaises(ValueError):
            common.convert_list_index_0_to_1([0], direction=-1)

    def test_rmg_mol_to_dict_repr(self):
        """Test the rmg_mol_to_dict_repr() function."""
        mol = Molecule(smiles='CC')
        for atom in mol.atoms:
            atom.id = -1
        representation = common.rmg_mol_to_dict_repr(mol, testing=True)
        expected_repr = {'atoms': [{'element': {'number': 6, 'isotope': -1}, 'atomtype': 'Cs',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 0,
                                    'props': {'inRing': False}, 'edges': {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}},
                                   {'element': {'number': 6, 'isotope': -1}, 'atomtype': 'Cs',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 1,
                                    'props': {'inRing': False}, 'edges': {0: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}},
                                   {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 2,
                                    'props': {'inRing': False}, 'edges': {0: 1.0}},
                                   {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 3,
                                    'props': {'inRing': False}, 'edges': {0: 1.0}},
                                   {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 4,
                                    'props': {'inRing': False}, 'edges': {0: 1.0}},
                                   {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 5,
                                    'props': {'inRing': False}, 'edges': {1: 1.0}},
                                   {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 6,
                                    'props': {'inRing': False}, 'edges': {1: 1.0}},
                                   {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 7,
                                    'props': {'inRing': False}, 'edges': {1: 1.0}}],
                         'multiplicity': 1, 'props': {},
                         'atom_order': [0, 1, 2, 3, 4, 5, 6, 7],
                         }
        self.assertEqual(representation, expected_repr)

        mol = Molecule(smiles='NCC')
        representation = common.rmg_mol_to_dict_repr(mol, testing=True)
        expected_repr = {'atoms': [{'element': {'number': 7, 'isotope': -1}, 'atomtype': 'N3s',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 1, 'id': 0,
                                    'props': {'inRing': False}, 'edges': {1: 1.0, 3: 1.0, 4: 1.0}},
                                   {'element': {'number': 6, 'isotope': -1}, 'atomtype': 'Cs',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 1,
                                    'props': {'inRing': False}, 'edges': {0: 1.0, 2: 1.0, 5: 1.0, 6: 1.0}},
                                   {'element': {'number': 6, 'isotope': -1}, 'atomtype': 'Cs',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 2,
                                    'props': {'inRing': False}, 'edges': {1: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}},
                                   {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 3,
                                    'props': {'inRing': False}, 'edges': {0: 1.0}},
                                   {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 4,
                                    'props': {'inRing': False}, 'edges': {0: 1.0}},
                                   {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 5,
                                    'props': {'inRing': False}, 'edges': {1: 1.0}},
                                   {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 6,
                                    'props': {'inRing': False}, 'edges': {1: 1.0}},
                                   {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 7,
                                    'props': {'inRing': False}, 'edges': {2: 1.0}},
                                   {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 8,
                                    'props': {'inRing': False}, 'edges': {2: 1.0}},
                                   {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                                    'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': 9,
                                    'props': {'inRing': False}, 'edges': {2: 1.0}}],
                         'multiplicity': 1, 'props': {},
                         'atom_order': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                         }
        self.assertEqual(representation, expected_repr)

    def test_rmg_mol_from_dict_repr(self):
        """Test the rmg_mol_from_dict_repr() function."""
        representation = {'atoms':
                          [{'element': {'number': 7, 'isotope': -1}, 'atomtype': 'N3s',
                            'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 1, 'id': -32768,
                            'props': {'inRing': False}, 'edges': {-32767: 1.0, -32765: 1.0, -32764: 1.0}},
                           {'element': {'number': 6, 'isotope': -1}, 'atomtype': 'Cs',
                            'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': -32767,
                            'props': {'inRing': False}, 'edges': {-32768: 1.0, -32766: 1.0, -32763: 1.0, -32762: 1.0}},
                           {'element': {'number': 6, 'isotope': -1}, 'atomtype': 'Cs',
                            'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': -32766,
                            'props': {'inRing': False}, 'edges': {-32767: 1.0, -32761: 1.0, -32760: 1.0, -32759: 1.0}},
                           {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                            'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': -32765,
                            'props': {'inRing': False}, 'edges': {-32768: 1.0}},
                           {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                            'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': -32764,
                            'props': {'inRing': False}, 'edges': {-32768: 1.0}},
                           {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                            'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': -32763,
                            'props': {'inRing': False}, 'edges': {-32767: 1.0}},
                           {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                            'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': -32762,
                            'props': {'inRing': False}, 'edges': {-32767: 1.0}},
                           {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                            'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': -32761,
                            'props': {'inRing': False}, 'edges': {-32766: 1.0}},
                           {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                            'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': -32760,
                            'props': {'inRing': False}, 'edges': {-32766: 1.0}},
                           {'element': {'number': 1, 'isotope': -1}, 'atomtype': 'H',
                            'radical_electrons': 0, 'charge': 0, 'label': '', 'lone_pairs': 0, 'id': -32759,
                            'props': {'inRing': False}, 'edges': {-32766: 1.0}}],
                          'multiplicity': 1, 'props': {},
                          'atom_order': [-32768, -32767, -32766, -32765, -32764, -32763, -32762, -32761, -32760, -32759],
                          }
        mol = common.rmg_mol_from_dict_repr(representation=representation, is_ts=False)
        smiles = mol.to_smiles()
        self.assertEqual(len(smiles), 3)
        self.assertEqual(smiles.count('C'), 2)
        self.assertEqual(smiles.count('N'), 1)

        # Test round trip:
        mol = Molecule(smiles='CC')
        representation = common.rmg_mol_to_dict_repr(mol)
        new_mol = common.rmg_mol_from_dict_repr(representation, is_ts=False)
        self.assertEqual(new_mol.to_smiles(), 'CC')

    def test_calc_rmsd(self):
        """Test compute the root-mean-square deviation between two matrices."""
        # Test a np.array type input:
        a_1 = np.array([1, 2, 3, 4])
        b_1 = np.array([1, 2, 3, 4])
        rmsd_1 = common.calc_rmsd(a_1, b_1)
        self.assertEqual(rmsd_1, 0.0)

        # Test a list type input:
        a_2 = [1, 2, 3, 4]
        b_2 = [1, 2, 3, 4]
        rmsd_2 = common.calc_rmsd(a_2, b_2)
        self.assertEqual(rmsd_2, 0.0)

        # Test a 1-length list:
        a_3 = [1]
        b_3 = [2]
        rmsd_3 = common.calc_rmsd(a_3, b_3)
        self.assertEqual(rmsd_3, 1.0)

        a_4 = np.array([1, 2, 3, 4])
        b_4 = np.array([4, 3, 2, 1])
        rmsd_4 = common.calc_rmsd(a_4, b_4)
        self.assertAlmostEqual(rmsd_4, 2.23606797749979)

        a_5 = np.array([[1, 2], [3, 4]])
        b_5 = np.array([[4, 3], [2, 1]])
        rmsd_5 = common.calc_rmsd(a_5, b_5)
        self.assertAlmostEqual(rmsd_5, 3.1622776601683795)

    def test_check_r_n_p_symbols_between_rmg_and_arc_rxns(self):
        """Test the _check_r_n_p_symbols_between_rmg_and_arc_rxns() function"""
        arc_rxn = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='OH', smiles='[OH]')],
                              p_species=[ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='H2O', smiles='O')])
        rmg_reactions = get_rmg_reactions_from_arc_reaction(arc_reaction=arc_rxn, db=self.rmgdb)
        self.assertTrue(common._check_r_n_p_symbols_between_rmg_and_arc_rxns(arc_rxn, rmg_reactions))

    def test_almost_equal_coords(self):
        """Test the almost_equal_coords() function"""
        with self.assertRaises(TypeError):
            common.almost_equal_coords([1], [2])
        ch4_a = {'symbols': ('C', 'H', 'H', 'H', 'H'),
                 'isotopes': (12, 1, 1, 1, 1),
                 'coords': ((0.0, 0.0, 0.0),
                            (0.6300326, 0.6300326, 0.6300326),
                            (-0.6300326, -0.6300326, 0.6300326),
                            (-0.6300326, 0.6300326, -0.6300326),
                            (0.6300326, -0.6300326, -0.6300326))}
        ch4_b = {'symbols': ('H', 'C', 'H', 'H', 'H'),
                 'isotopes': (1, 12, 1, 1, 1),
                 'coords': ((0.6300326, 0.6300326, 0.6300326),
                            (0.0, 0.0, 0.0),
                            (-0.6300326, -0.6300326, 0.6300326),
                            (-0.6300326, 0.6300326, -0.6300326),
                            (0.6300326, -0.6300326, -0.6300326))}
        ch4_c = {'symbols': ('C', 'H', 'H', 'H', 'H'),
                 'isotopes': (12, 1, 1, 1, 1),
                 'coords': ((0.0, 0.0, 0.0),
                            (0.6300324, 0.6300324, 0.6300324),
                            (-0.6300324, -0.6300324, 0.6300324),
                            (-0.6300324, 0.6300324, -0.6300324),
                            (0.6300324, -0.6300324, -0.6300324))}
        ch4_d = {'symbols': ('C', 'H', 'H', 'H', 'H'),
                 'isotopes': (12, 1, 1, 1, 1),
                 'coords': ((0.0, 0.0, 0.0),
                            (-0.6300324, -0.6300324, -0.6300324),
                            (0.6300324, 0.6300324, -0.6300324),
                            (0.6300324, -0.6300324, 0.6300324),
                            (-0.6300324, 0.6300324, 0.6300324))}
        with self.assertRaises(ValueError):
            common.almost_equal_coords(ch4_a, ch4_b)
        self.assertTrue(common.almost_equal_coords(ch4_a, ch4_a))
        self.assertTrue(common.almost_equal_coords(ch4_a, ch4_c))
        self.assertFalse(common.almost_equal_coords(ch4_a, ch4_d))


    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        """
        globalized_restart_path = os.path.join(common.ARC_PATH, 'arc', 'testing', 'restart', '4_globalized_paths',
                                               'restart_paths_globalized.yml')
        os.remove(path=globalized_restart_path)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
