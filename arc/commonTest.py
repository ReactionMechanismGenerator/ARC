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
from arc.rmgdb import make_rmg_database_object, load_families_only
from arc.species.mapping import get_rmg_reactions_from_arc_reaction
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
        spc_1 = ARCSpecies(label='spc_1', smiles='CCO', xyz="""C      -0.97459464    0.29181710    0.10303882
                                                               C       0.39565894   -0.35143697    0.10221676
                                                               O       0.30253309   -1.63748710   -0.49196889
                                                               H      -1.68942501   -0.32359616    0.65926091
                                                               H      -0.93861751    1.28685508    0.55523033
                                                               H      -1.35943743    0.38135479   -0.91822428
                                                               H       0.76858330   -0.46187184    1.12485643
                                                               H       1.10301149    0.25256708   -0.47388355
                                                               H       1.19485981   -2.02360458   -0.47786539""")
        top, top_has_heavy_atoms = common.determine_top_group_indices(spc_1.mol,
                                                                      atom1=spc_1.mol.atoms[0],
                                                                      atom2=spc_1.mol.atoms[1],
                                                                      )
        self.assertEqual(top, [2, 7, 8, 3, 9])
        top, top_has_heavy_atoms = common.determine_top_group_indices(spc_1.mol,
                                                                      atom1=spc_1.mol.atoms[0],
                                                                      atom2=spc_1.mol.atoms[1],
                                                                      index=0,
                                                                      )
        self.assertEqual(top, [1, 6, 7, 2, 8])

        spc_2 = ARCSpecies(label='spc_2', smiles='CCl', xyz="""C      -0.13907898   -0.00452380   -0.00236673
                                                               Cl      1.62669770    0.05291133    0.02768175
                                                               H      -0.50777491   -0.18308484    1.01009636
                                                               H      -0.45879122   -0.81430981   -0.66186117
                                                               H      -0.52105258    0.94900712   -0.37355021""")
        top, top_has_heavy_atoms = common.determine_top_group_indices(spc_2.mol,
                                                                      atom1=spc_2.mol.atoms[0],
                                                                      atom2=spc_2.mol.atoms[1],
                                                                      index=0,
                                                                      )
        self.assertEqual(top, [1])

        spc_3 = ARCSpecies(label='spc_3', smiles='[O]Cl', xyz="""O       0.84074010    0.00000000    0.00000000
                                                                 Cl     -0.84074010    0.00000000    0.00000000""")
        top, top_has_heavy_atoms = common.determine_top_group_indices(spc_3.mol,
                                                                      atom1=spc_3.mol.atoms[0],
                                                                      atom2=spc_3.mol.atoms[1],
                                                                      index=0,
                                                                      )
        self.assertEqual(top, [1])
        top, top_has_heavy_atoms = common.determine_top_group_indices(spc_3.mol,
                                                                      atom1=spc_3.mol.atoms[1],
                                                                      atom2=spc_3.mol.atoms[0],
                                                                      index=0,
                                                                      )
        self.assertEqual(top, [0])

        spc_4 = ARCSpecies(label='spc_4', smiles='c1cc(OC)ccc1OC(CC)SF',
                           xyz="""C       0.94215818    0.68259790   -0.88468248
                                  C       2.26853480    0.67405892   -0.43827136
                                  C       2.67171279   -0.24624190    0.52623293
                                  O       3.92802249   -0.36692583    1.05121570
                                  C       4.91605664    0.53301876    0.56522390
                                  C       1.73721797   -1.15343774    1.03003385
                                  C       0.41050395   -1.14495773    0.58353699
                                  C       0.00403820   -0.21696773   -0.37187302
                                  O      -1.24786249   -0.09420914   -0.90395655
                                  C      -2.34211052   -0.74644555   -0.24812824
                                  C      -3.60183081    0.09941937   -0.49599944
                                  C      -3.48543146    1.48913951    0.12201008
                                  S      -2.63174622   -2.42257099   -0.92600337
                                  F      -3.71723126   -2.90971886    0.13619095
                                  H       0.63778273    1.40294019   -1.64072806
                                  H       2.95487566    1.39638959   -0.86636549
                                  H       4.65183250    1.57078604    0.79400990
                                  H       5.90677644    0.28085546    0.95761333
                                  H       4.95383303    0.43163142   -0.52369005
                                  H       2.04709640   -1.88218094    1.77575511
                                  H      -0.26284396   -1.88840238    0.99556971
                                  H      -2.18639696   -0.81327590    0.83508536
                                  H      -3.77036870    0.22560086   -1.57312199
                                  H      -4.48612013   -0.39141797   -0.07174634
                                  H      -2.64439268    2.04641598   -0.30206953
                                  H      -4.39788184    2.06307772   -0.06860114
                                  H      -3.34536255    1.42469609    1.20573587""")
        atom1 = spc_4.mol.atoms[9]  # this is the C atom at the S, O, H, and C junction
        atom2a = spc_4.mol.atoms[10]  # C
        atom2b = spc_4.mol.atoms[8]  # O
        atom2c = spc_4.mol.atoms[12]  # S
        atom2d = spc_4.mol.atoms[21]  # H

        top, top_has_heavy_atoms = common.determine_top_group_indices(spc_4.mol, atom1, atom2a)
        self.assertEqual(len(top), 7)
        self.assertTrue(top_has_heavy_atoms)

        top, top_has_heavy_atoms = common.determine_top_group_indices(spc_4.mol, atom1, atom2b)
        self.assertEqual(len(top), 16)
        self.assertTrue(top_has_heavy_atoms)

        top, top_has_heavy_atoms = common.determine_top_group_indices(spc_4.mol, atom1, atom2c)
        self.assertEqual(len(top), 2)
        self.assertTrue(top_has_heavy_atoms)

        top, top_has_heavy_atoms = common.determine_top_group_indices(spc_4.mol, atom1, atom2d)
        self.assertEqual(top, [22])
        self.assertFalse(top_has_heavy_atoms)  # H

    def test_extremum_list(self):
        """Test the extremum_list() function"""
        lst = []
        min_lst = common.extremum_list(lst)
        self.assertIsNone(min_lst)

        lst = [None]
        min_lst = common.extremum_list(lst)
        self.assertIsNone(min_lst)

        lst = [None, None]
        min_lst = common.extremum_list(lst)
        self.assertIsNone(min_lst)

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

    def test_get_get_extremum_index(self):
        """Test the get_extremum_index() function."""
        lst = []
        extremum_index = common.get_extremum_index(lst)
        self.assertIsNone(extremum_index)

        lst = [None]
        extremum_index = common.get_extremum_index(lst)
        self.assertIsNone(extremum_index)

        lst = [None, None]
        extremum_index = common.get_extremum_index(lst)
        self.assertIsNone(extremum_index)

        lst = [100]
        extremum_index = common.get_extremum_index(lst)
        self.assertEqual(extremum_index, 0)

        lst = [-8, -80]
        extremum_index = common.get_extremum_index(lst)
        self.assertEqual(extremum_index, 1)

        lst = [-8, None]
        extremum_index = common.get_extremum_index(lst)
        self.assertEqual(extremum_index, 0)

        lst = [-8, -8, -8, -8]
        extremum_index = common.get_extremum_index(lst)
        self.assertEqual(extremum_index, 0)

        lst = [-8, None, None, 100, -79, None]
        extremum_index = common.get_extremum_index(lst)
        self.assertEqual(extremum_index, 4)

        lst = [-8, None, None, 100, -79, None]
        extremum_index = common.get_extremum_index(lst, return_min=False)
        self.assertEqual(extremum_index, 3)

        lst = [8, None, 0, 100, 79, None]
        extremum_index = common.get_extremum_index(lst, skip_values=[0])
        self.assertEqual(extremum_index, 0)

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

    def test_get_bonds_from_dmat(self):
        """test getting bonds from a distance matrix"""
        h2_xyz = {'symbols': ('H', 'H'), 'isotopes': (1, 1), 'coords': ((0.0, 0.0, 0.371517), (0.0, 0.0, -0.371517))}
        bonds = common.get_bonds_from_dmat(dmat=converter.xyz_to_dmat(h2_xyz), elements=h2_xyz['symbols'])
        self.assertEqual(bonds, [(0, 1)])

        nh3_xyz = {'symbols': ('N', 'H', 'H', 'H'), 'isotopes': (14, 1, 1, 1),
                   'coords': ((0.0006492354002636227, -0.0009969784288894215, 0.2955929244020652),
                              (-0.4178660616416419, 0.842103963871788, -0.09477452075659776),
                              (-0.5203922802597125, -0.7822529247012627, -0.10002797449860866),
                              (0.9376091065010891, -0.05885406074163403, -0.10079042914685925))}
        nh3_dmat = converter.xyz_to_dmat(nh3_xyz)
        bonds = common.get_bonds_from_dmat(dmat=nh3_dmat, elements=nh3_xyz['symbols'])
        self.assertTrue(common.check_that_all_entries_are_in_list(bonds, [(0, 1), (0, 2), (0, 3)]))

        with self.assertRaises(ValueError):
            common.get_bonds_from_dmat(dmat=nh3_dmat, elements=h2_xyz['symbols'])

        c5diol_xyz = {'symbols': ('O', 'C', 'C', 'C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H',
                                  'H', 'H', 'H', 'H', 'H', 'H'),
                      'isotopes': (16, 12, 12, 12, 12, 12, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                      'coords': ((-0.7424236292317568, 1.8880948990185775, 0.7094655459117919),
                                 (-1.69936658455501, 0.8765955319978836, 0.4418752644358258),
                                 (-2.319641524664334, 0.9851238322799277, -0.9492998396013165),
                                 (-1.3604476725233865, 0.7009204927719811, -2.110341289097591),
                                 (-0.17304644597340518, 1.6519684539247455, -2.3046019185178146),
                                 (-0.5560182128159811, 3.1121210696379262, -2.521968614455213),
                                 (-0.9988904149050924, 3.70111326206134, -1.3039774797343486),
                                 (-0.9305849561093252, 2.65049516982463, 0.11982419502552191),
                                 (-1.2151825911022744, -0.09577045040706897, 0.5762170967451388),
                                 (-2.486206440387856, 0.9626947162342055, 1.1982005901051487),
                                 (-2.7806693858193388, 1.9695584826317185, -1.0803901927484965),
                                 (-3.136574179993242, 0.254842775339988, -1.0090088874510745),
                                 (-0.9675452279527998, -0.3176192411324763, -1.9980855151799954),
                                 (-1.9486132279848114, 0.696520542998205, -3.037240645079648),
                                 (0.5124481026165748, 1.565276995656951, -1.4550836995590979),
                                 (0.3805701966789545, 1.309740628333202, -3.187871325120396),
                                 (0.31874961995347606, 3.6737255646101694, -2.865553968557727),
                                 (-1.348690242126151, 3.20647349995194, -3.2707873365181532),
                                 (-1.150962458565406, 4.645286501517968, -1.4926167602923543))}
        bonds = common.get_bonds_from_dmat(dmat=converter.xyz_to_dmat(c5diol_xyz), elements=c5diol_xyz['symbols'])
        self.assertTrue(common.check_that_all_entries_are_in_list(
            bonds, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 7), (1, 8), (1, 9), (2, 10), (2, 11), (3, 12),
                    (3, 13), (4, 14), (4, 15), (5, 16), (5, 17), (6, 18)]))

        b_butenyl_xyz = {'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                                     'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                         'isotopes': (12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                         'coords': ((0.8655718620447994, 0.8820146741298858, 0.6342160188049014),
                                    (2.109490186955037, 1.5144936515720024, 0.7124665459461635),
                                    (3.2522460147647636, 0.8618658053950043, 0.2591221025405695),
                                    (3.149570117424174, -0.4205928124241902, -0.27342876680717765),
                                    (1.9042194677600208, -1.0525100004662034, -0.35083860728394395),
                                    (0.7469984057387592, -0.4111809393224403, 0.10593633654400356),
                                    (-0.5847931386940496, -1.0193653172210266, 0.04460380361543761),
                                    (-0.8598047575202332, -2.429670891753131, -0.3614475690227426),
                                    (-1.1995427451661047, -2.5222594273301895, -1.820341604424373),
                                    (-2.3740862642155256, -2.9533459439072502, -2.2991437127263556),
                                    (-0.012465784856138728, 1.4130728619474153, 0.9952277243928566),
                                    (2.1829285341042826, 2.5163305471162896, 1.1271343450678533),
                                    (4.220236118638911, 1.3520410174671271, 0.31838735197853474),
                                    (4.038805315598044, -0.9321031172214398, -0.6326693440561406),
                                    (1.8625691696491857, -2.0484987888810378, -0.781876812936419),
                                    (-1.4405845087338476, -0.38162927917308864, 0.24463988010176366),
                                    (-0.014567136918820167, -3.089411321381447, -0.1405634825429265),
                                    (-1.6921909786924387, -2.8018003350298124, 0.24897564165176247),
                                    (-0.42576149750456516, -2.219250874579489, -2.5232375475370903),
                                    (-2.5542078891776994, -2.996143835485984, -3.368970805863449),
                                    (-3.1787270916846255, -3.26780621963705, -1.641942110891295))}
        bonds = common.get_bonds_from_dmat(dmat=converter.xyz_to_dmat(b_butenyl_xyz), elements=b_butenyl_xyz['symbols'])
        self.assertTrue(common.check_that_all_entries_are_in_list(
            bonds, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (0, 10),
                    (1, 11), (2, 12), (3, 13), (4, 14), (6, 15), (7, 16), (7, 17), (8, 18), (9, 19), (9, 20)]))

        # TS N2H3 + NH2 perturbation 1
        ts_n3h5_1_xyz = {'symbols': ('N', 'H', 'H', 'N', 'H', 'H', 'N', 'H'), 'isotopes': (14, 1, 1, 14, 1, 1, 14, 1),
                         'coords': ((-0.424886, 0.694403, -0.092695), (-0.473966, 1.196384, 0.779312),
                                    (0.050017, 0.609187, -0.289804), (-1.230175, -0.49518, -0.005872),
                                    (-1.816882, -0.464956, 0.807105), (-1.846181, -0.503768, -0.816157),
                                    (1.943069, -0.169246, -0.067924), (1.737758, -0.876687, 0.671382))}
        bonds = common.get_bonds_from_dmat(dmat=converter.xyz_to_dmat(ts_n3h5_1_xyz), elements=ts_n3h5_1_xyz['symbols'])
        self.assertTrue(common.check_that_all_entries_are_in_list(bonds, [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5), (6, 7)]))

        # TS N2H3 + NH2 perturbation 2
        ts_n3h5_2_xyz = {'symbols': ('N', 'H', 'H', 'N', 'H', 'H', 'N', 'H'), 'isotopes': (14, 1, 1, 14, 1, 1, 14, 1),
                         'coords': ((-0.468986, 0.665003, -0.092695), (-0.415166, 1.078784, 0.852812),
                                    (1.402417, 0.109387, -0.172204), (-1.230175, -0.43638, -0.005872),
                                    (-1.816882, -0.538456, 0.821805), (-1.713881, -0.636068, -0.874957),
                                    (1.869569, -0.139846, -0.082624), (1.737758, -0.832587, 0.656682))}
        bonds = common.get_bonds_from_dmat(dmat=converter.xyz_to_dmat(ts_n3h5_2_xyz), elements=ts_n3h5_2_xyz['symbols'])
        self.assertTrue(common.check_that_all_entries_are_in_list(bonds, [(0, 1), (0, 3), (3, 4), (3, 5), (2, 6), (6, 7)]))

        # TS C3 intra H migration 1
        ts_c3_intra_h_1_xyz = {'symbols': ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                               'isotopes': (12, 12, 12, 1, 1, 1, 1, 1, 1, 1),
                               'coords': ((1.524852, 0.591965, -0.276141), (0.15624659, -0.07605135, -0.22609488),
                                          (-1.00575771, 0.59324294, 0.32280653), (0.11884265, -1.16827906, -0.35326306),
                                          (2.19484, 0.138175, -1.010532), (1.441557, 1.657406, -0.52194235),
                                          (-0.290061, 0.790224, -0.88231488), (2.00176129, 0.53979971, 0.708471),
                                          (-1.84853706, 0.023846, 0.73488606), (-0.89616871, 1.61707229, 0.73243706))}
        bonds = common.get_bonds_from_dmat(dmat=converter.xyz_to_dmat(ts_c3_intra_h_1_xyz), elements=ts_c3_intra_h_1_xyz['symbols'])
        self.assertTrue(common.check_that_all_entries_are_in_list(
            bonds, [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5), (1, 6), (0, 7), (2, 8), (2, 9)]))

        # TS C3 intra H migration 2
        ts_c3_intra_h_2_xyz = {'symbols': ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                               'isotopes': (12, 12, 12, 1, 1, 1, 1, 1, 1, 1),
                               'coords': ((1.524852, 0.591965, -0.276141), (0.20330541, -0.06428665, -0.29668312),
                                          (-0.98222829, 0.62853706, 0.40515947), (0.13060735, -1.13298494, -0.31796894),
                                          (2.19484, 0.138175, -1.010532), (1.441557, 1.657406, -0.51017765),
                                          (-1.290061, 0.190224, -0.95290312), (2.02529071, 0.51627029, 0.708471),
                                          (-1.81324294, 0.023846, 0.69959194), (-0.87263929, 1.64060171, 0.69714294))}
        bonds = common.get_bonds_from_dmat(dmat=converter.xyz_to_dmat(ts_c3_intra_h_2_xyz), elements=ts_c3_intra_h_2_xyz['symbols'])
        self.assertTrue(common.check_that_all_entries_are_in_list(
            bonds, [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5), (0, 7), (2, 8), (2, 9), (2, 6)]))

        # TS C3 intra H migration 3
        ts_c3_intra_h_3_xyz = {'symbols': ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                               'isotopes': (12, 12, 12, 1, 1, 1, 1, 1, 1, 1),
                               'coords': ((0.52243, -0.940297, -0.553226),
                                          (0.0853459676972449, 0.19778650807568876, 0.2683129515458674),
                                          (-0.9372640161513774, 1.2163625242270664, 0.11038855652982141),
                                          (0.3409084755398544, 0.08505557338043682, 1.4489635733804367),
                                          (1.487662, -1.382626, -0.294747), (0.581946, -0.596182, -1.5863334755398544),
                                          (0.8060955791123762, 1.1427712525325744, -0.04576614676087361),
                                          (-0.2413660489202912, -1.7526210489202914, -0.530005),
                                          (-1.4459875733804366, 1.71243, 0.7424384266195632),
                                          (-1.3369720489202912, 1.1281310489202914, -1.0204525733804366))}
        bonds = common.get_bonds_from_dmat(dmat=converter.xyz_to_dmat(ts_c3_intra_h_3_xyz), elements=ts_c3_intra_h_3_xyz['symbols'])
        self.assertTrue(common.check_that_all_entries_are_in_list(
            bonds, [(0, 1), (1, 2), (0, 4), (0, 5), (0, 7), (2, 8), (1, 3), (1, 6), (2, 9)]))
        ts_c3_intra_h_4_xyz = {'symbols': ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                               'isotopes': (12, 12, 12, 1, 1, 1, 1, 1, 1, 1),
                               'coords': ((0.52243, -0.940297, -0.553226),
                                          (0.22391003230275508, 0.16314549192431121, 0.47615904845413265),
                                          (-0.8679819838486225, 1.1124394757729337, -0.13209855652982141),
                                          (0.3509475244601456, 0.0549384266195632, 1.4188464266195633),
                                          (1.487662, -1.382626, -0.294747), (0.581946, -0.596182, -1.5963725244601457),
                                          (-0.04722357911237618, 1.6547627474674258, 0.014468146760873612),
                                          (-0.2212879510797088, -1.7325429510797088, -0.530005),
                                          (-1.4158704266195632, 1.71243, 0.7725555733804368),
                                          (-1.3168939510797086, 1.1080529510797088, -0.9903354266195631))}
        bonds = common.get_bonds_from_dmat(dmat=converter.xyz_to_dmat(ts_c3_intra_h_4_xyz), elements=ts_c3_intra_h_4_xyz['symbols'])
        self.assertTrue(common.check_that_all_entries_are_in_list(
            bonds, [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5), (2, 6), (0, 7), (2, 9), (2, 8)]))
        bonds = common.get_bonds_from_dmat(dmat=converter.xyz_to_dmat(ts_c3_intra_h_4_xyz),
                                           elements=ts_c3_intra_h_4_xyz['symbols'], tolerance=1.5)
        self.assertTrue(common.check_that_all_entries_are_in_list(
            bonds, [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5), (2, 6), (0, 7), (2, 9), (2, 8)]))

        ts_c3_intra_h_4_dmat = np.array([[0.0, 1.53828901, 2.51482024, 2.21561943, 1.09276674, 1.10005089, 2.71682111, 1.08688051, 3.54283303, 2.78745668],
                                         [1.53828901, 0.0, 1.56951093, 0.95734361, 2.14027444, 2.23610266, 1.58480092, 2.19184899, 2.27530511, 2.32755947],
                                         [2.51482024, 1.56951093, 0.0, 2.23819748, 3.43523869, 2.67689685, 0.9946058, 2.94456579, 1.21596442, 0.96856168],
                                         [2.21561943, 0.95734361, 2.23819748, 0.0, 2.50900652, 3.09335795, 2.16569999, 2.70565426, 2.50731248, 3.11366385],
                                         [1.09276674, 2.14027444, 3.43523869, 2.50900652, 0.0, 1.77004086, 3.41719449, 1.76019832, 4.37595784, 3.81482096],
                                         [1.10005089, 2.23610266, 2.67689685, 3.09335795, 1.77004086, 0.0, 2.83855849, 1.75318016, 3.86429556, 2.62245894],
                                         [2.71682111, 1.58480092, 0.9946058, 2.16569999, 3.41719449, 2.83855849, 0.0, 3.43519858, 1.56563605, 1.70897182],
                                         [1.08688051, 2.19184899, 2.94456579, 2.70565426, 1.76019832, 1.75318016, 3.43519858, 0.0, 3.87188972, 3.0791625],
                                         [3.54283303, 2.27530511, 1.21596442, 2.50731248, 4.37595784, 3.86429556, 1.56563605, 3.87188972, 0.0, 1.86624024],
                                         [2.78745668, 2.32755947, 0.96856168, 3.11366385, 3.81482096, 2.62245894, 1.70897182, 3.0791625, 1.86624024, 0.0]], np.float64)
        bonds = common.get_bonds_from_dmat(dmat=ts_c3_intra_h_4_dmat, elements=ts_c3_intra_h_4_xyz['symbols'])
        self.assertTrue(common.check_that_all_entries_are_in_list(
            bonds, [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5), (2, 6), (0, 7), (2, 9), (2, 8)]))

        # TS CH3CH2NH2 + H <=> CH3CHNH2 + H2, dmats generated from two perturbed geometries each in a different direction
        # *1: atom 1 (C atom), *2: atom 5 (abstracted H), *3: atom 10 (abstracting H)
        elements = ['N', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']
        dmat_1 = [
            [0.0, 1.5090259346402959, 2.5172759431530243, 0.7950770533587272, 1.0474365896134326, 1.8368939426118225,
             1.9540650655179468, 2.8114651959035895, 3.440516794254548, 2.8104289394195154, 3.598392289463991],
            [1.5090259346402959, 0.0, 1.710363447648614, 2.116480215311915, 2.151404555387096, 0.7810219174357843,
             0.969135237808477, 2.376042234476513, 2.2500088284432445, 2.3083024527780127, 2.552810584578915],
            [2.5172759431530243, 1.710363447648614, 0.0, 2.9968112724257634, 2.592021808092268, 1.7983085854153713,
             2.1295983024833713, 1.1037689520168332, 1.1163540804384426, 1.0441400855329384, 3.548417909916891],
            [0.7950770533587272, 2.116480215311915, 2.9968112724257634, 0.0, 1.4595141462779728, 2.5811532604031697,
             2.2626774056563317, 3.017019658938956, 3.945141916042076, 3.3682074787593317, 4.317904523789208],
            [1.0474365896134326, 2.151404555387096, 2.592021808092268, 1.4595141462779728, 0.0, 2.1806906406498605,
             2.81247658444124, 2.9040510974355214, 3.634597633190848, 2.49248375571737, 3.974590190698698],
            [1.8368939426118225, 0.7810219174357843, 1.7983085854153713, 2.5811532604031697, 2.1806906406498605, 0.0,
             1.7052235065299475, 2.7148348360817702, 2.2600210076633855, 2.076081769677702, 2.0384600329293936],
            [1.9540650655179468, 0.969135237808477, 2.1295983024833713, 2.2626774056563317, 2.81247658444124,
             1.7052235065299475, 0.0, 2.459312625277458, 2.4734530668131622, 2.9860874132892476, 3.080568847606993],
            [2.8114651959035895, 2.376042234476513, 1.1037689520168332, 3.017019658938956, 2.9040510974355214,
             2.7148348360817702, 2.459312625277458, 0.0, 1.7845662476633397, 1.7877716361611289, 4.560922031971599],
            [3.440516794254548, 2.2500088284432445, 1.1163540804384426, 3.945141916042076, 3.634597633190848,
             2.2600210076633855, 2.4734530668131622, 1.7845662476633397, 0.0, 1.7487964829584288, 3.4532709952115606],
            [2.8104289394195154, 2.3083024527780127, 1.0441400855329384, 3.3682074787593317, 2.49248375571737,
             2.076081769677702, 2.9860874132892476, 1.7877716361611289, 1.7487964829584288, 0.0, 3.7110656385670673],
            [3.598392289463991, 2.552810584578915, 3.548417909916891, 4.317904523789208, 3.974590190698698,
             2.0384600329293936, 3.080568847606993, 4.560922031971599, 3.4532709952115606, 3.7110656385670673, 0.0]]
        bonds = common.get_bonds_from_dmat(dmat=np.array(dmat_1, np.float64), elements=elements)
        self.assertTrue(common.check_that_all_entries_are_in_list(
            bonds, [(0, 1), (1, 2), (0, 3), (0, 4), (1, 5), (1, 6), (2, 7), (2, 8), (2, 9)]))  # No (5, 10) bond.

        dmat_2 = [
            [0.0, 1.4123925549714023, 2.5389708109479114, 1.2581822445168058, 0.9941687157160606, 2.5543277703490532,
             2.2049537515899096, 2.9030994024137327, 3.453642408332476, 2.7717454994874133, 2.6912643033410175],
            [1.4123925549714023, 0.0, 1.3603746862851966, 2.073403892754172, 1.9718898158695954, 1.890462207541803,
             1.29496367545339, 1.972351759907336, 2.1141037631581976, 2.0399458679182443, 2.1295382501089968],
            [2.5389708109479114, 1.3603746862851966, 0.0, 3.1821091260680543, 2.6558074350323992, 2.6229559820411184,
             2.2428217672699176, 1.0910956578496303, 1.0712616831372197, 1.1464296245426653, 2.62863252978441],
            [1.2581822445168058, 2.073403892754172, 3.1821091260680543, 0.0, 1.8593041168733335, 3.4060875852877266,
             2.3052644860561964, 3.195595147656094, 4.074584144225373, 3.6707302807822617, 3.733972817045404],
            [0.9941687157160606, 1.9718898158695954, 2.6558074350323992, 1.8593041168733335, 0.0, 3.2105016838046323,
             3.0392594195236327, 2.8927491738202695, 3.6746277646231484, 2.5178093015058636, 3.1401710236858857],
            [2.5543277703490532, 1.890462207541803, 2.6229559820411184, 3.4060875852877266, 3.2105016838046323, 0.0,
             2.00073767150549, 3.5958390798940223, 2.7423274409012826, 3.021922340267745, 0.7523699800467102],
            [2.2049537515899096, 1.29496367545339, 2.2428217672699176, 2.3052644860561964, 3.0392594195236327,
             2.00073767150549, 0.0, 2.6816273919677567, 2.570219463496617, 3.1934763809576823, 2.5958884959836257],
            [2.9030994024137327, 1.972351759907336, 1.0910956578496303, 3.195595147656094, 2.8927491738202695,
             3.5958390798940223, 2.6816273919677567, 0.0, 1.7677230177500438, 1.754715704027976, 3.673784860949396],
            [3.453642408332476, 2.1141037631581976, 1.0712616831372197, 4.074584144225373, 3.6746277646231484,
             2.7423274409012826, 2.570219463496617, 1.7677230177500438, 0.0, 1.783887124931921, 2.7650434189370765],
            [2.7717454994874133, 2.0399458679182443, 1.1464296245426653, 3.6707302807822617, 2.5178093015058636,
             3.021922340267745, 3.1934763809576823, 1.754715704027976, 1.783887124931921, 0.0, 2.7359815651907393],
            [2.6912643033410175, 2.1295382501089968, 2.62863252978441, 3.733972817045404, 3.1401710236858857,
             0.7523699800467102, 2.5958884959836257, 3.673784860949396, 2.7650434189370765, 2.7359815651907393, 0.0]]
        bonds = common.get_bonds_from_dmat(dmat=np.array(dmat_2, np.float64), elements=elements)
        self.assertTrue(common.check_that_all_entries_are_in_list(
            bonds, [(0, 1), (1, 2), (0, 3), (0, 4), (1, 6), (2, 7), (2, 8), (2, 9), (5, 10)]))  # No (1, 5) bond.

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
        self.assertFalse(common.almost_equal_coords(ch4_a, ch4_b))
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
        if os.path.isfile(globalized_restart_path):
            os.remove(path=globalized_restart_path)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
