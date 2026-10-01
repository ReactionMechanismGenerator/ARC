#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.checks.ts module
"""

import unittest
import os
import shutil
import tempfile
from unittest.mock import Mock, patch

import networkx as nx
import numpy as np

import arc.checks.ts as ts
from arc.common import (ARC_PATH, ARC_TESTING_PATH, NUMBER_BY_SYMBOL, almost_equal_lists, get_bonds_from_dmat,
                        get_test_project_directory)
from arc.exceptions import ReactionError
from arc.job.factory import job_factory
from arc.level import Level
from arc.checks.common import IRC_START_GEOMETRY_TOLERANCE
from arc.parser.parser import (parse_irc_path, parse_irc_start_geometry, parse_irc_traj, parse_normal_mode_displacement,
                               parse_geometry)
from arc.reaction import ARCReaction
from arc.species.converter import kabsch, xyz_from_data, xyz_to_dmat
from arc.species.species import ARCSpecies, TSGuess


class TestTSChecks(unittest.TestCase):
    """
    Contains unit tests for the check module.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None

        cls.rms_list_1 = [0.01414213562373095, 0.05, 0.04, 0.5632938842203065, 0.7993122043357026, 0.08944271909999159,
                          0.10677078252031312, 0.09000000000000001, 0.05, 0.09433981132056604]
        path_1 = os.path.join(ARC_TESTING_PATH, 'freq', 'C3H7_intra_h_TS.out')
        cls.freqs_1, cls.normal_modes_disp_1 = parse_normal_mode_displacement(path_1)
        cls.ts_1 = ARCSpecies(label='TS', is_ts=True)
        cls.ts_1.ts_guesses = [TSGuess(family='intra_H_migration', xyz='C 0 0 0'),
                               TSGuess(family='intra_H_migration', xyz='C 0 0 0'),
                               ]
        cls.ts_xyz_1 = """O      -0.63023600    0.92494700    0.43958200
                          C       0.14513500   -0.07880000   -0.04196400
                          C      -0.97050300   -1.02992900   -1.65916600
                          N      -0.75664700   -2.16458700   -1.81286400
                          H      -1.25079800    0.57954500    1.08412300
                          H       0.98208300    0.28882200   -0.62114100
                          H       0.30969500   -0.94370100    0.59100600
                          H      -1.47626400   -0.10694600   -1.88883800"""  # 'N#[CH].[CH2][OH]'

        cls.ts_xyz_2 = """C        1.279906   -0.191149   -0.024558
                          C       -0.040637    0.517073    0.025028
                          C       -1.318249   -0.255157   -0.038482
                          H       -0.091811    1.556222   -0.280736
                          H        2.096169    0.442456    0.330967
                          H        1.269507   -1.096591    0.591664
                          H       -0.823137    0.291596    1.036018
                          H        1.524901   -0.510401   -1.049451
                          H       -2.222433    0.228641   -0.382279
                          H       -1.279319   -1.336527   -0.018107"""  # C[CH]C <=> [CH2]CC
        cls.r_xyz_2a = """C                  0.50180491   -0.93942231   -0.57086745
                          C                  0.01278145    0.13148427    0.42191407
                          C                 -0.86874485    1.29377369   -0.07163907
                          H                  0.28549447    0.06799101    1.45462711
                          H                  1.44553946   -1.32386345   -0.24456986
                          H                  0.61096295   -0.50262210   -1.54153222
                          H                 -0.24653265    2.11136864   -0.37045418
                          H                 -0.21131163   -1.73585284   -0.61629002
                          H                 -1.51770930    1.60958621    0.71830245
                          H                 -1.45448167    0.96793094   -0.90568876"""
        cls.r_xyz_2b = """C                  0.50180491   -0.93942231   -0.57086745
                          C                  0.01278145    0.13148427    0.42191407
                          H                  0.28549447    0.06799101    1.45462711
                          H                  1.44553946   -1.32386345   -0.24456986
                          H                  0.61096295   -0.50262210   -1.54153222
                          H                 -0.24653265    2.11136864   -0.37045418
                          C                 -0.86874485    1.29377369   -0.07163907
                          H                 -0.21131163   -1.73585284   -0.61629002
                          H                 -1.51770930    1.60958621    0.71830245
                          H                 -1.45448167    0.96793094   -0.90568876"""
        cls.p_xyz_2 = """C                  0.48818717   -0.94549701   -0.55196729
                         C                  0.35993708    0.29146456    0.35637075
                         C                 -0.91834764    1.06777042   -0.01096751
                         H                  0.30640232   -0.02058840    1.37845537
                         H                  1.37634603   -1.48487836   -0.29673876
                         H                  0.54172192   -0.63344406   -1.57405191
                         H                  1.21252186    0.92358349    0.22063264
                         H                 -0.36439762   -1.57761595   -0.41622918
                         H                 -1.43807526    1.62776079    0.73816131
                         H                 -1.28677889    1.04716138   -1.01532486"""
        cls.ts_spc_2 = ARCSpecies(label='TS', is_ts=True, xyz=cls.ts_xyz_2)
        cls.ts_spc_2.mol_from_xyz()
        cls.reactant_2a = ARCSpecies(label='C[CH]C', smiles='C[CH]C', xyz=cls.r_xyz_2a)
        cls.reactant_2b = ARCSpecies(label='C[CH]C', smiles='C[CH]C', xyz=cls.r_xyz_2b)  # Shuffled order, != ts_xyz_2
        cls.product_2 = ARCSpecies(label='[CH2]CC', smiles='[CH2]CC', xyz=cls.p_xyz_2)
        cls.rxn_2a = ARCReaction(r_species=[cls.reactant_2a], p_species=[cls.product_2])
        cls.rxn_2a.ts_species = cls.ts_spc_2
        cls.rxn_2b = ARCReaction(r_species=[cls.reactant_2b], p_species=[cls.product_2])
        cls.rxn_2b.ts_species = cls.ts_spc_2
        cls.job1 = job_factory(job_adapter='gaussian',
                               species=[ARCSpecies(label='SPC', smiles='C')],
                               job_type='composite',
                               level=Level(method='CBS-QB3'),
                               project='test_project',
                               project_directory=get_test_project_directory('arc_project_for_testing_delete_after_usage4'),
                               )

        cls.rxn_3 = ARCReaction(r_species=[ARCSpecies(label='NH3', smiles='N'), ARCSpecies(label='H', smiles='[H]')],
                                p_species=[ARCSpecies(label='NH2', smiles='[NH2]'), ARCSpecies(label='H2', smiles='[H][H]')])
        cls.rxn_3.ts_species = ARCSpecies(label='TS3', is_ts=True,
                                          xyz=os.path.join(ARC_TESTING_PATH, 'freq', 'TS_NH3+H=NH2+H2.out'))

        ccooj_xyz = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H'),
                     'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1),
                     'coords': ((-1.10653, -0.06552, 0.042602),
                                (0.385508, 0.205048, 0.049674),
                                (0.759622, 1.114927, -1.032928),
                                (0.675395, 0.525342, -2.208593),
                                (-1.671503, 0.860958, 0.166273),
                                (-1.396764, -0.534277, -0.898851),
                                (-1.36544, -0.740942, 0.862152),
                                (0.97386, -0.704577, -0.082293),
                                (0.712813, 0.732272, 0.947293),
                                )}
        ccooj = ARCSpecies(label='CCOOj', smiles='CCO[O]', xyz=ccooj_xyz)
        cls.rxn_4 = ARCReaction(r_species=[ccooj, ARCSpecies(label='CC', smiles='CC')],
                                p_species=[ARCSpecies(label='CCOOH', smiles='CCOO'), ARCSpecies(label='CCj', smiles='[CH2]C')])
        cls.rxn_4.ts_species = ARCSpecies(label='TS4', is_ts=True,
                                          xyz=os.path.join(ARC_TESTING_PATH, 'composite', 'TS0_composite_2043.out'))

        ea_xyz = """N                  1.27511929   -0.21413688   -0.09829069
                    C                  0.04568411    0.51479456    0.24529057
                    C                 -1.17314611   -0.39875221    0.01838707
                    H                  1.35437220   -1.02559828    0.48071654
                    H                  1.24076865   -0.49175940   -1.05836661
                    H                 -0.03911651    1.38305825   -0.37424716
                    H                  0.08243929    0.81185065    1.27257181
                    H                 -1.08834550   -1.26701591    0.63792481
                    H                 -2.06804111    0.13183054    0.26847684
                    H                 -1.20990129   -0.69580830   -1.00889416"""
        ch3chnh2_xyz = {'symbols': ('C', 'C', 'N', 'H', 'H', 'H', 'H', 'H', 'H'), 'isotopes': (12, 12, 14, 1, 1, 1, 1, 1, 1),
                        'coords': ((-1.126885721359211, 0.0525078449047029, 0.16096122288992248),
                                   (0.3110254709748632, 0.43198882113724923, 0.25317384633324647),
                                   (0.8353039548590167, 1.612144179849946, -0.43980881000393546),
                                   (-1.2280793873408098, -0.9063788436936245, -0.3556390245681601),
                                   (-1.5506369749373963, -0.04972561706616586, 1.1642482560565997),
                                   (-1.7098257757667992, 0.801238249703059, -0.3836181889319467),
                                   (0.9893789736981147, -0.22608070229597396, 0.7909844523882721),
                                   (1.6782473720416846, 1.9880253918558228, -0.007820540824533973),
                                   (0.15476721028478654, 2.3655991320212193, -0.527262390251838))}
        ea = ARCSpecies(label='EA', smiles='NCC', xyz=ea_xyz)
        ch3chnh2 = ARCSpecies(label='CH3CHNH2', smiles='C[CH]N', xyz=ch3chnh2_xyz)
        cls.rxn_5 = ARCReaction(r_species=[ea, ARCSpecies(label='H', smiles='[H]')],
                                p_species=[ch3chnh2, ARCSpecies(label='H2', smiles='[H][H]')])
        cls.rxn_5.ts_species = ARCSpecies(label='TS5', is_ts=True,
                                          xyz=os.path.join(ARC_TESTING_PATH, 'composite', 'TS0_composite_2044.out'))
        cls.rxn_6 = ARCReaction(r_species=[ea, ARCSpecies(label='H', smiles='[H]')],
                                p_species=[ARCSpecies(label='CH2CH2NH2', smiles='[CH2]CN'), ARCSpecies(label='H2', smiles='[H][H]')])
        cls.rxn_6.ts_species = ARCSpecies(label='TS6', is_ts=True,
                                          xyz=os.path.join(ARC_TESTING_PATH, 'composite', 'TS1_composite_695.out'))

        cls.c2h5no2_xyz = """O                  0.62193295    1.59121319   -0.58381518
                             N                  0.43574593    0.41740669    0.07732982
                             O                  1.34135576   -0.35713755    0.18815532
                             C                 -0.87783860    0.10001361    0.65582554
                             C                 -1.73002357   -0.64880063   -0.38564362
                             H                 -1.37248469    1.00642547    0.93625873
                             H                 -0.74723653   -0.51714586    1.52009245
                             H                 -1.23537748   -1.55521250   -0.66607681
                             H                 -2.68617014   -0.87982825    0.03543830
                             H                 -1.86062564   -0.03164117   -1.24991054"""
        cls.rxn_7 = ARCReaction(r_species=[ARCSpecies(label='C2H5NO2', smiles='[O-][N+](=O)CC', xyz=cls.c2h5no2_xyz)],
                                p_species=[ARCSpecies(label='C2H5ONO', smiles='CCON=O')])
        xyz_7 = """O        0.520045    1.026544   -0.223307
                   N        0.818877   -0.207900   -0.075436
                   O        1.964221   -0.523711   -0.014266
                   C       -0.968581    0.050866    0.695117
                   C       -1.903603   -0.321292   -0.395596
                   H       -1.145584    1.019535    1.170709
                   H       -0.740906   -0.730110    1.427000
                   H       -1.628826   -1.274421   -0.863423
                   H       -2.906412   -0.425097    0.055493
                   H       -1.951439    0.465285   -1.158262"""
        cls.rxn_7.ts_species = ARCSpecies(label='TS7', is_ts=True, xyz=xyz_7)
        cls.rxn_8 = ARCReaction(r_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC')],
                                p_species=[ARCSpecies(label='iC3H7', smiles='C[CH]C')])
        cls.rxn_8.ts_species = ARCSpecies(label='TS8', is_ts=True,
                                          xyz=os.path.join(ARC_TESTING_PATH, 'freq', 'TS_nC3H7-iC3H7.out'))
        cls.rxn_8.ts_label = cls.rxn_8.ts_species.label

        cls.ccooj_xyz = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H'),
                         'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1),
                         'coords': ((-1.0558210286905791, -0.033295741345331475, -0.10080257427276477),
                                    (0.4179269451258209, 0.1783120494760899, 0.21035514057712598),
                                    (1.192340198620975, -0.6538968331819118, -0.6111144272932062),
                                    (2.447496844713327, -0.41401220066030225, -0.2838136270165004),
                                    (-1.3361400206798786, -1.0915178337292122, 0.0871488160949016),
                                    (-1.2595361812141086, 0.21489046050976668, -1.164118972112424),
                                    (-1.674103962566004, 0.6234141912977972, 0.5469951414070585),
                                    (0.5956635031421627, -0.06437685811832683, 1.2825664049784615),
                                    (0.6725467605584472, 1.2467632926898817, 0.02676370288619609))}
        cls.c2h6_xyz = {'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'), 'isotopes': (12, 12, 1, 1, 1, 1, 1, 1),
                        'coords': ((0.7556043653036945, 0.024826347908153884, 0.005052054400891163),
                                   (-0.7556043714831743, -0.02482635425984135, -0.005052060396341506),
                                   (1.1738008157989488, -0.9310955032458506, -0.32410611474882567),
                                   (1.118910949297216, 0.8090395514407392, -0.6657965895380061),
                                   (1.1265685376671182, 0.23440078247573526, 1.0127643787045777),
                                   (-1.1738008745316266, 0.9310955099502647, 0.3241060408257655),
                                   (-1.1189110146736851, -0.8090395034382198, 0.6657966599789673),
                                   (-1.1265684046717404, -0.2344009055503307, -1.0127644068816903))}

        cls.species_dict_8 = {spc.label: spc for spc in cls.rxn_8.r_species + cls.rxn_8.p_species + [cls.rxn_8.ts_species]}
        cls.project_directory_5 = get_test_project_directory('arc_project_for_testing_delete_after_usage5')
        cls.output_dict_8 = {'iC3H7': {'paths': {'freq': os.path.join(ARC_TESTING_PATH, 'freq', 'iC3H7.out'),
                                                 'sp': os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out'),
                                                 'opt': os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out'),
                                                 'composite': ''},
                                       'convergence': True},
                             'nC3H7': {'paths': {'freq': os.path.join(ARC_TESTING_PATH, 'freq', 'nC3H7.out'),
                                                 'sp': os.path.join(ARC_TESTING_PATH, 'opt', 'nC3H7.out'),
                                                 'opt': os.path.join(ARC_TESTING_PATH, 'opt', 'nC3H7.out'),
                                                 'composite': ''},
                                       'convergence': True},
                             'TS8': {'paths': {'freq': os.path.join(ARC_TESTING_PATH, 'freq', 'TS_nC3H7-iC3H7.out'),
                                               'sp': os.path.join(ARC_TESTING_PATH, 'opt', 'TS_nC3H7-iC3H7.out'),
                                               'opt': os.path.join(ARC_TESTING_PATH, 'opt', 'TS_nC3H7-iC3H7.out'),
                                               'composite': ''},
                                     'convergence': True}}

    def test_analyze_ts_normal_mode_displacement(self):
        """Test checking for NMD."""
        # iC3H7 <=> nC3H7
        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'TS_C3_intraH_8.out')
        self.rxn_2a.ts_species.populate_ts_checks()
        self.rxn_2a.ts_species.final_xyz = parse_geometry(log_file_path=self.job1.local_path_to_output_file)
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['NMD'])
        ts.check_ts(reaction=self.rxn_2a, job=self.job1, checks=['NMD'])
        self.assertTrue(self.rxn_2a.ts_species.ts_checks['NMD'])

        # C2H5NO2 <=> C2H5ONO
        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'C2H5NO2__C2H5ONO.out')
        self.rxn_7.ts_species.populate_ts_checks()
        self.assertFalse(self.rxn_7.ts_species.ts_checks['NMD'])
        ts.check_ts(reaction=self.rxn_7, job=self.job1, checks=['NMD'])
        self.assertTrue(self.rxn_7.ts_species.ts_checks['NMD'])

        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'TS_nC3H7-iC3H7.out')
        self.rxn_8.ts_species.populate_ts_checks()
        self.assertFalse(self.rxn_8.ts_species.ts_checks['NMD'])
        ts.check_ts(reaction=self.rxn_8, job=self.job1, checks=['NMD'])
        self.assertTrue(self.rxn_8.ts_species.ts_checks['NMD'])

    def test_did_ts_pass_all_checks(self):
        """Test the did_ts_pass_all_checks() function."""
        spc = ARCSpecies(label='TS', is_ts=True)
        spc.populate_ts_checks()
        self.assertFalse(ts.ts_passed_checks(spc))

        self.ts_checks = {'E0': False,
                          'e_elect': False,
                          'IRC': False,
                          'freq': False,
                          'NMD': False,
                          'warnings': '',
                          }
        for key in ['E0', 'e_elect', 'IRC', 'freq']:
            spc.ts_checks[key] = True
        self.assertFalse(ts.ts_passed_checks(spc))
        self.assertTrue(ts.ts_passed_checks(spc, exemptions=['NMD', 'warnings']))
        spc.ts_checks['e_elect'] = False

    def test_ts_passed_checks_irc(self):
        """Test that ts_passed_checks() treats the three-valued IRC check correctly."""
        spc = ARCSpecies(label='TS', is_ts=True)
        spc.populate_ts_checks()
        for key in ['E0', 'e_elect', 'freq', 'NMD']:
            spc.ts_checks[key] = True

        spc.ts_checks['IRC'] = True
        self.assertTrue(ts.ts_passed_checks(spc, exemptions=['warnings']))

        spc.ts_checks['IRC'] = None
        self.assertTrue(ts.ts_passed_checks(spc, exemptions=['warnings']))

        spc.ts_checks['IRC'] = False
        self.assertFalse(ts.ts_passed_checks(spc, exemptions=['warnings']))
        self.assertTrue(ts.ts_passed_checks(spc, exemptions=['warnings', 'IRC']))

    def test_check_rxn_e_elect(self):
        """Test the check_rxn_e_elect() function."""
        rxn1 = ARCReaction(r_species=[ARCSpecies(label='s1', smiles='C')], p_species=[ARCSpecies(label='s2', smiles='C')])
        rxn1.ts_species = ARCSpecies(label='TS', is_ts=True)
        # no data
        rxn1.ts_species.populate_ts_checks()
        ts.check_rxn_e_elect(reaction=rxn1)
        self.assertIsNone(rxn1.ts_species.ts_checks['e_elect'])
        # only E0 (incorrect)
        rxn1.r_species[0].e0 = 2
        rxn1.p_species[0].e0 = 50
        rxn1.ts_species.e0 = -100
        rxn1.ts_species.populate_ts_checks()
        ts.check_rxn_e_elect(reaction=rxn1)
        self.assertFalse(rxn1.ts_species.ts_checks['E0'])
        # only E0 (partial data)
        rxn1.r_species[0].e0 = 2
        rxn1.p_species[0].e0 = None
        rxn1.ts_species.e0 = -100
        rxn1.ts_species.populate_ts_checks()
        ts.check_rxn_e_elect(reaction=rxn1)
        self.assertIsNone(rxn1.ts_species.ts_checks['e_elect'])
        # check e_elect (correct)
        rxn1.r_species[0].e_elect = 2
        rxn1.p_species[0].e_elect = 50
        rxn1.ts_species.e_elect = 100
        rxn1.ts_species.populate_ts_checks()
        ts.check_rxn_e_elect(reaction=rxn1)
        self.assertTrue(rxn1.ts_species.ts_checks['e_elect'])
        # incorrect e_elect
        rxn1.r_species[0].e_elect = 2
        rxn1.p_species[0].e_elect = 50
        rxn1.ts_species.e_elect = -100
        rxn1.ts_species.populate_ts_checks()
        ts.check_rxn_e_elect(reaction=rxn1)
        self.assertFalse(rxn1.ts_species.ts_checks['e_elect'])

    def test_determine_changing_bond(self):
        """Test the determine_changing_bond() function."""
        dmat_2 = np.array([[0, 1, 2],
                           [1, 0, 5],
                           [2, 5, 0]])
        dmat_bonds_1 = ts.get_bonds_from_dmat(dmat=dmat_2, elements=['C', 'C', 'H'], tolerance=1.5)
        dmat_bonds_2 = ts.get_bonds_from_dmat(dmat=dmat_2, elements=['C', 'C', 'H'], tolerance=1.5)
        change = ts.determine_changing_bond(bond=(0, 1), dmat_bonds_1=dmat_bonds_1, dmat_bonds_2=dmat_bonds_2)
        self.assertIsNone(change)
        dmat_1 = np.array([[0, 5, 2],
                           [5, 0, 5],
                           [2, 5, 0]])
        dmat_bonds_1 = ts.get_bonds_from_dmat(dmat=dmat_1, elements=['C', 'C', 'H'], tolerance=1.5)
        change = ts.determine_changing_bond(bond=(0, 1), dmat_bonds_1=dmat_bonds_1, dmat_bonds_2=dmat_bonds_2)
        self.assertEqual(change, 'forming')
        change = ts.determine_changing_bond(bond=(0, 1), dmat_bonds_1=dmat_bonds_2, dmat_bonds_2=dmat_bonds_1)
        self.assertEqual(change, 'breaking')

    def test_compute_rxn_e0(self):
        """Test the compute_rxn_e0() function."""
        for spc_label in self.rxn_8.reactants + self.rxn_8.products + [self.rxn_8.ts_label]:
            folder = 'rxns' if self.species_dict_8[spc_label].is_ts else 'Species'
            base_path = os.path.join(self.project_directory_5, 'output', folder, spc_label, 'geometry')
            os.makedirs(base_path, exist_ok=True)
            freq_path = os.path.join(self.project_directory_5, 'output', folder, spc_label, 'geometry', 'freq.out')
            shutil.copy(src=self.output_dict_8[spc_label]['paths']['freq'], dst=freq_path)

        self.assertIsNone(self.rxn_8.r_species[0].e0)
        self.assertIsNone(self.rxn_8.p_species[0].e0)
        self.assertIsNone(self.rxn_8.ts_species.e0)

        rxn_copy = ts.compute_rxn_e0(reaction=self.rxn_8,
                                     species_dict=self.species_dict_8,
                                     project_directory=self.project_directory_5,
                                     kinetics_adapter='arkane',
                                     output=self.output_dict_8,
                                     sp_level=Level(repr='cbs-qb3'),
                                     freq_scale_factor=1.0,
                                     )
        self.assertAlmostEqual(rxn_copy.r_species[0].e0, 3523.18, places=1)
        self.assertAlmostEqual(rxn_copy.p_species[0].e0, 3514.12, places=1)
        self.assertAlmostEqual(rxn_copy.ts_species.e0, 3760.7, places=1)

    def test_check_rxn_e0(self):
        """Test the check_rxn_e0() function."""
        for spc_label in self.rxn_8.reactants + self.rxn_8.products + [self.rxn_8.ts_label]:
            folder = 'rxns' if self.species_dict_8[spc_label].is_ts else 'Species'
            base_path = os.path.join(self.project_directory_5, 'output', folder, spc_label, 'geometry')
            os.makedirs(base_path, exist_ok=True)
            freq_path = os.path.join(self.project_directory_5, 'output', folder, spc_label, 'geometry', 'freq.out')
            shutil.copy(src=self.output_dict_8[spc_label]['paths']['freq'], dst=freq_path)
        rxn_copy = ts.compute_rxn_e0(reaction=self.rxn_8,
                                     species_dict=self.species_dict_8,
                                     project_directory=self.project_directory_5,
                                     kinetics_adapter='arkane',
                                     output=self.output_dict_8,
                                     sp_level=Level(repr='CBS-QB3'),
                                     freq_scale_factor=1.0,
                                     )
        self.assertIsNone(rxn_copy.ts_species.ts_checks['E0'])
        ts.check_rxn_e0(reaction=rxn_copy, verbose=True)
        self.assertTrue(rxn_copy.ts_species.ts_checks['E0'])

    def _make_project_with_freq_files(self, labels, species_dict, output_dict, copy_files=True):
        """Create a temporary project directory that holds a real freq.out for each of ``labels``."""
        project_directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        if copy_files:
            for label in labels:
                folder = 'rxns' if species_dict[label].is_ts else 'Species'
                base_path = os.path.join(project_directory, 'output', folder, label, 'geometry')
                os.makedirs(base_path)
                shutil.copy(src=next(iter(output_dict.values()))['paths']['freq'],
                            dst=os.path.join(base_path, 'freq.out'))
        return project_directory

    def _fake_statmech_factory(self, e0_by_label, calls):
        """Return a ``statmech_factory`` replacement whose ``compute_thermo()`` sets E0 values by species label."""
        def factory(**kwargs):
            calls.append(kwargs)
            adapter = Mock()
            def compute_thermo(e0_only=False, skip_rotors=False):
                for spc in kwargs['species']:
                    if spc.label in e0_by_label:
                        spc.e0 = e0_by_label[spc.label]
            adapter.compute_thermo.side_effect = compute_thermo
            return adapter
        return factory

    def _rxn_8_with_preset_e0(self, well_e0=500.0, ts_e0=None):
        """Return a copy of rxn_8 whose wells (and optionally TS) carry preset E0 values."""
        rxn = self.rxn_8.copy()
        rxn.ts_species.populate_ts_checks()
        for spc in rxn.r_species + rxn.p_species:
            spc.e0 = well_e0
        rxn.ts_species.e0 = ts_e0
        return rxn

    def _compute_rxn_8_e0(self, rxn, e0_by_label, calls, project_directory):
        with patch.object(ts, 'statmech_factory', side_effect=self._fake_statmech_factory(e0_by_label, calls)):
            return ts.compute_rxn_e0(reaction=rxn,
                                     species_dict=self.species_dict_8,
                                     project_directory=project_directory,
                                     kinetics_adapter='arkane',
                                     output=self.output_dict_8,
                                     sp_level=Level(repr='cbs-qb3'),
                                     freq_scale_factor=1.0,
                                     )

    def test_compute_rxn_e0_ignores_preset_e0_values(self):
        """Test that compute_rxn_e0() computes E0 for all participants and leaves the given reaction untouched."""
        rxn = self._rxn_8_with_preset_e0()
        project_directory = self._make_project_with_freq_files(['nC3H7', 'iC3H7', 'TS8'],
                                                               self.species_dict_8, self.output_dict_8)
        calls = list()
        rxn_copy = self._compute_rxn_8_e0(rxn, {'nC3H7': 100.0, 'iC3H7': 90.0, 'TS8': 200.0}, calls,
                                          project_directory)
        self.assertEqual(len(calls), 1)
        self.assertEqual(sorted(spc.label for spc in calls[0]['species']), ['TS8', 'iC3H7', 'nC3H7'])
        self.assertIsNone(calls[0]['bac_type'])
        self.assertAlmostEqual(rxn_copy.r_species[0].e0, 100.0)
        self.assertAlmostEqual(rxn_copy.p_species[0].e0, 90.0)
        self.assertAlmostEqual(rxn_copy.ts_species.e0, 200.0)
        self.assertAlmostEqual(rxn.r_species[0].e0, 500.0)
        self.assertAlmostEqual(rxn.p_species[0].e0, 500.0)
        self.assertIsNone(rxn.ts_species.e0)

    def test_compute_rxn_e0_does_not_keep_a_preset_e0_the_statmech_run_did_not_produce(self):
        """Test that a preset well E0 is dropped, not kept, when the statmech run yields no E0 for that well."""
        rxn = self._rxn_8_with_preset_e0()
        project_directory = self._make_project_with_freq_files(['nC3H7', 'iC3H7', 'TS8'],
                                                               self.species_dict_8, self.output_dict_8)
        rxn_copy = self._compute_rxn_8_e0(rxn, {'iC3H7': 90.0, 'TS8': 200.0}, list(), project_directory)
        self.assertIsNone(rxn_copy.r_species[0].e0)
        self.assertAlmostEqual(rxn_copy.p_species[0].e0, 90.0)
        self.assertAlmostEqual(rxn.r_species[0].e0, 500.0)

    def test_compute_rxn_e0_carries_the_correction_stamps_and_the_aec_digest(self):
        """Test that the stamps and the AEC.yml digest of the statmech run reach the copy, and a stale one does not."""
        rxn = self._rxn_8_with_preset_e0()
        for spc in rxn.r_species + rxn.p_species + [rxn.ts_species]:
            spc.e0_atom_corrections_applied = False
            spc.e0_bond_corrections_applied = True
            spc.e0_aec_yml_sha256 = 'f' * 64
        project_directory = self._make_project_with_freq_files(['nC3H7', 'iC3H7', 'TS8'],
                                                               self.species_dict_8, self.output_dict_8)

        def factory(**kwargs):
            adapter = Mock()

            def compute_thermo(e0_only=False, skip_rotors=False):
                for spc in kwargs['species']:
                    if spc.label != 'iC3H7':
                        spc.e0 = 100.0
                        spc.e0_atom_corrections_applied = True
                        spc.e0_bond_corrections_applied = False
                        spc.e0_aec_yml_sha256 = 'a' * 64
            adapter.compute_thermo.side_effect = compute_thermo
            return adapter

        with patch.object(ts, 'statmech_factory', side_effect=factory):
            rxn_copy = ts.compute_rxn_e0(reaction=rxn, species_dict=self.species_dict_8,
                                         project_directory=project_directory, kinetics_adapter='arkane',
                                         output=self.output_dict_8, sp_level=Level(repr='cbs-qb3'),
                                         freq_scale_factor=1.0)
        for spc in (rxn_copy.r_species[0], rxn_copy.ts_species):
            self.assertIs(spc.e0_atom_corrections_applied, True)
            self.assertIs(spc.e0_bond_corrections_applied, False)
            self.assertEqual(spc.e0_aec_yml_sha256, 'a' * 64)
        well_without_e0 = rxn_copy.p_species[0]
        self.assertIsNone(well_without_e0.e0)
        self.assertIsNone(well_without_e0.e0_atom_corrections_applied)
        self.assertIsNone(well_without_e0.e0_aec_yml_sha256)
        self.assertEqual(rxn.ts_species.e0_aec_yml_sha256, 'f' * 64)

    def test_compute_rxn_e0_passes_each_label_once(self):
        """Test that compute_rxn_e0() gives the statmech run one object per label for an identity reaction."""
        ch3, ch4, ts_spc = (ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='CH4', smiles='C'),
                            ARCSpecies(label='TS_id', is_ts=True))
        rxn = ARCReaction(r_species=[ch3, ch4],
                          p_species=[ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='CH3', smiles='[CH3]')])
        rxn.ts_species = ts_spc
        rxn.ts_label = 'TS_id'
        species_dict = {'CH3': ch3, 'CH4': ch4, 'TS_id': ts_spc}
        output_dict = {label: {'paths': {'freq': self.output_dict_8['nC3H7']['paths']['freq']}}
                       for label in species_dict}
        project_directory = self._make_project_with_freq_files(list(species_dict), species_dict, output_dict)
        calls = list()
        with patch.object(ts, 'statmech_factory',
                          side_effect=self._fake_statmech_factory({'CH3': 10.0, 'CH4': 20.0, 'TS_id': 30.0}, calls)):
            rxn_copy = ts.compute_rxn_e0(reaction=rxn, species_dict=species_dict,
                                         project_directory=project_directory, kinetics_adapter='arkane',
                                         output=output_dict, sp_level=Level(repr='cbs-qb3'), freq_scale_factor=1.0)
        self.assertEqual(sorted(spc.label for spc in calls[0]['species']), ['CH3', 'CH4', 'TS_id'])
        self.assertEqual([spc.e0 for spc in rxn_copy.r_species + rxn_copy.p_species], [10.0, 20.0, 20.0, 10.0])

    def test_check_ts_energy_compares_e0_values_computed_with_the_same_settings(self):
        """Test that the E0 check ignores well and TS E0 values that came from a run with different corrections."""
        rxn = self._rxn_8_with_preset_e0(ts_e0=999.0)
        project_directory = self._make_project_with_freq_files(['nC3H7', 'iC3H7', 'TS8'],
                                                               self.species_dict_8, self.output_dict_8)
        with patch.object(ts, 'statmech_factory',
                          side_effect=self._fake_statmech_factory({'nC3H7': 100.0, 'iC3H7': 90.0, 'TS8': 200.0},
                                                                  list())):
            ts.check_ts(reaction=rxn,
                        checks=['energy'],
                        species_dict=self.species_dict_8,
                        project_directory=project_directory,
                        kinetics_adapter='arkane',
                        output=self.output_dict_8,
                        sp_level=Level(repr='cbs-qb3'),
                        freq_scale_factor=1.0,
                        verbose=False,
                        )
        self.assertIs(rxn.ts_species.ts_checks['E0'], True)
        self.assertAlmostEqual(rxn.r_species[0].e0, 500.0)
        self.assertAlmostEqual(rxn.p_species[0].e0, 500.0)
        self.assertAlmostEqual(rxn.ts_species.e0, 200.0)

    def test_check_ts_energy_does_not_compare_preset_e0_values_when_they_cannot_be_recomputed(self):
        """Test that preset E0 values are never compared if the consistent E0 values cannot be computed."""
        rxn = self._rxn_8_with_preset_e0(ts_e0=200.0)
        project_directory = self._make_project_with_freq_files(['nC3H7', 'iC3H7', 'TS8'],
                                                               self.species_dict_8, self.output_dict_8,
                                                               copy_files=False)
        with patch.object(ts, 'statmech_factory') as factory, patch.object(ts.logger, 'warning') as warning:
            ts.check_ts(reaction=rxn,
                        checks=['energy'],
                        species_dict=self.species_dict_8,
                        project_directory=project_directory,
                        kinetics_adapter='arkane',
                        output=self.output_dict_8,
                        sp_level=Level(repr='cbs-qb3'),
                        freq_scale_factor=1.0,
                        verbose=False,
                        )
        factory.assert_not_called()
        self.assertIsNone(rxn.ts_species.ts_checks['E0'])
        self.assertTrue(any('left undetermined' in str(call.args[0]) for call in warning.call_args_list))

    def test_check_ts_energy_does_not_compare_preset_e0_values_when_statmech_yields_nothing(self):
        """Test that an empty statmech result leaves the E0 check undetermined instead of failing it."""
        rxn = self._rxn_8_with_preset_e0(ts_e0=200.0)
        project_directory = self._make_project_with_freq_files(['nC3H7', 'iC3H7', 'TS8'],
                                                               self.species_dict_8, self.output_dict_8)
        with patch.object(ts, 'statmech_factory', side_effect=self._fake_statmech_factory(dict(), list())):
            ts.check_ts(reaction=rxn,
                        checks=['energy'],
                        species_dict=self.species_dict_8,
                        project_directory=project_directory,
                        kinetics_adapter='arkane',
                        output=self.output_dict_8,
                        sp_level=Level(repr='cbs-qb3'),
                        freq_scale_factor=1.0,
                        verbose=False,
                        )
        self.assertIsNone(rxn.ts_species.ts_checks['E0'])

    def test_check_normal_mode_displacement(self):
        """Test the check_normal_mode_displacement() function."""
        self.rxn_2a.ts_species.populate_ts_checks()
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['NMD'])
        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'TS_intra_H_migration_CBS-QB3.out')
        self.rxn_2a.ts_species.final_xyz = parse_geometry(log_file_path=self.job1.local_path_to_output_file)
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertTrue(self.rxn_2a.ts_species.ts_checks['NMD'])
        self.rxn_2a.ts_species.populate_ts_checks()

        # like rxn_2a, shuffled reactant atom order, expect this to return False for the same TS as in self.job1
        self.rxn_2b.ts_species.populate_ts_checks()
        self.assertFalse(self.rxn_2b.ts_species.ts_checks['NMD'])
        ts.check_normal_mode_displacement(reaction=self.rxn_2b, job=self.job1)
        self.assertFalse(self.rxn_2b.ts_species.ts_checks['NMD'])

        # Wrong TS for intra H migration [CH2]CC <=> C[CH]C
        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'C3H7', 'TS1.log')  # A wrong TS.
        self.rxn_2a.ts_species.final_xyz = parse_geometry(log_file_path=self.job1.local_path_to_output_file)
        self.rxn_2a.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['NMD'])

        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'C3H7', 'TS2.log')  # A wrong TS.
        self.rxn_2a.ts_species.final_xyz = parse_geometry(log_file_path=self.job1.local_path_to_output_file)
        self.rxn_2a.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['NMD'])

        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'C3H7', 'TS3.log')  # ** The correct TS. **
        self.rxn_2a.ts_species.final_xyz = parse_geometry(log_file_path=self.job1.local_path_to_output_file)
        self.rxn_2a.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertTrue(self.rxn_2a.ts_species.ts_checks['NMD'])

        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'C3H7', 'TS4.log')  # A wrong TS.
        self.rxn_2a.ts_species.final_xyz = parse_geometry(log_file_path=self.job1.local_path_to_output_file)
        self.rxn_2a.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['NMD'])

        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'C3H7', 'TS5.log')  # A wrong TS.
        self.rxn_2a.ts_species.final_xyz = parse_geometry(log_file_path=self.job1.local_path_to_output_file)
        self.rxn_2a.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['NMD'])

        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'C3H7', 'TS6.log')  # A wrong TS.
        self.rxn_2a.ts_species.final_xyz = parse_geometry(log_file_path=self.job1.local_path_to_output_file)
        self.rxn_2a.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['NMD'])

        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'C3H7', 'TS7.log')  # A wrong TS.
        self.rxn_2a.ts_species.final_xyz = parse_geometry(log_file_path=self.job1.local_path_to_output_file)
        self.rxn_2a.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['NMD'])

        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'TS_C3_intraH_8.out')  # Correct TS (freq run, not composite).
        self.rxn_2a.ts_species.final_xyz = parse_geometry(log_file_path=self.job1.local_path_to_output_file)
        self.rxn_2a.ts_species.populate_ts_checks()
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['NMD'])
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertTrue(self.rxn_2a.ts_species.ts_checks['NMD'])

        # CCO[O] + CC <=> CCOO + [CH2]C, incorrect TS:
        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'TS0_composite_2043.out')
        self.rxn_4.ts_species.final_xyz = parse_geometry(log_file_path=self.job1.local_path_to_output_file)
        self.rxn_4.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_4, job=self.job1)
        self.assertFalse(self.rxn_4.ts_species.ts_checks['NMD'])

        # CCO[O] + CC <=> CCOO + [CH2]C, correct TS:
        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'TS0_composite_2102.out')
        self.rxn_4.ts_species.final_xyz = parse_geometry(log_file_path=self.job1.local_path_to_output_file)
        self.rxn_4.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_4, job=self.job1)
        self.assertTrue(self.rxn_4.ts_species.ts_checks['NMD'])

        # NCC + H <=> CH3CHNH2 + H2, correct TS:
        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'TS0_composite_2044.out')
        self.rxn_5.ts_species.final_xyz = parse_geometry(log_file_path=self.job1.local_path_to_output_file)
        self.rxn_5.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_5, job=self.job1)
        self.assertTrue(self.rxn_5.ts_species.ts_checks['NMD'])

        # NH2 + N2H3 <=> NH + N2H4:
        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'TS_NH2+N2H3.out')
        rxn_6 = ARCReaction(r_species=[ARCSpecies(label='NH2', xyz="""N       0.00000000   -0.00000000    0.14115400
                                                                      H      -0.80516800    0.00000000   -0.49355600
                                                                      H       0.80516800   -0.00000000   -0.49355600"""),
                                       ARCSpecies(label='N2H3', xyz="""N       0.59115400    0.02582600   -0.07080800
                                                                       H       1.01637000    0.90287000    0.19448100
                                                                       H       1.13108000   -0.79351700    0.15181600
                                                                       N      -0.73445800   -0.15245400    0.02565700
                                                                       H      -1.14969800    0.77790100   -0.02540800""")],
                            p_species=[ARCSpecies(label='NH', xyz="""N       0.00000000    0.00000000    0.13025700
                                                                     H       0.00000000    0.00000000   -0.90825700"""),
                                       ARCSpecies(label='N2H4', xyz="""N       0.70348300    0.09755100   -0.07212500
                                                                       N      -0.70348300   -0.09755100   -0.07212500
                                                                       H       1.05603900    0.38865300    0.83168200
                                                                       H      -1.05603900   -0.38865300    0.83168200
                                                                       H       1.14245100   -0.77661300   -0.32127200
                                                                       H      -1.14245100    0.77661300   -0.32127200""")])
        rxn_6.ts_species = ARCSpecies(label='TS6', is_ts=True, xyz="""N      -0.44734500    0.68033000   -0.09191900
                                                                      H      -0.45257300    1.14463200    0.81251500
                                                                      H       0.67532500    0.38185200   -0.23044400
                                                                      N      -1.22777700   -0.47121500   -0.00284000
                                                                      H      -1.81516400   -0.50310400    0.81640600
                                                                      H      -1.78119500   -0.57249600   -0.84071000
                                                                      N       1.91083300   -0.14543600   -0.06636000
                                                                      H       1.73701100   -0.85419700    0.66460600""")
        rxn_6.ts_species.mol_from_xyz()
        rxn_6.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=rxn_6, job=self.job1)
        self.assertTrue(rxn_6.ts_species.ts_checks['NMD'])

        # [CH2]CC=C <=> CCC=[CH] butylene intra H migration:
        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'TS_butylene_intra_H_migration.out')
        rxn_7 = ARCReaction(r_species=[
            ARCSpecies(label='butylene',
                       xyz={'symbols': ('C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                            'isotopes': (12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1),
                            'coords': ((-1.5025309111564664, -0.534274223668814, -0.8036222996901808),
                                       (-0.7174177387201146, -0.023936112728414158, 0.35370258735369786),
                                       (-1.5230462996626752, -0.05695435961481443, 1.6163349692848272),
                                       (-1.8634470313869078, 1.0277715224421244, 2.324841919574016),
                                       (-0.984024423978003, -0.9539130636653048, -1.6577859414775906),
                                       (-2.550807526086091, -0.2789561000296545, -0.9131030981780086),
                                       (-0.3724512697624012, 0.9914237465990766, 0.12894489304781925),
                                       (0.1738420368001901, -0.6466414881716757, 0.48830614688365104),
                                       (-1.8352343593831375, -1.0368501719961523, 1.9724902744574715),
                                       (-1.57401834878684, 2.026695960278519, 2.0137658090390858),
                                       (-2.446426657980167, 0.9347672870076474, 3.235948559430434))})],
                            p_species=[ARCSpecies(label='CCC=[CH]', smiles='CCC=[CH]')])
        rxn_7.ts_species = ARCSpecies(label='TS7', is_ts=True,
                                      xyz="""C                 -1.21222600   -0.64083500    0.00000300
                                             C                 -0.63380200    0.77863500   -0.00000300
                                             C                  0.87097000    0.58302100    0.00000400
                                             C                  1.24629100   -0.68545200   -0.00000300
                                             H                 -1.72740700   -0.95796100    0.90446200
                                             H                 -1.72743700   -0.95796100   -0.90443900
                                             H                 -0.95478600    1.35296500    0.87649200
                                             H                 -0.95477600    1.35295100   -0.87651200
                                             H                  1.55506600    1.42902600    0.00001400
                                             H                  2.20977700   -1.17852900   -0.00000300
                                             H                 -0.02783300   -1.25271100   -0.00001000""")
        rxn_7.ts_species.mol_from_xyz()
        rxn_7.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=rxn_7, job=self.job1)
        self.assertTrue(rxn_7.ts_species.ts_checks['NMD'])

        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'TS_NH3+H=NH2+H2.out')  # NH3 + H <=> NH2 + H2
        self.rxn_3.ts_species.populate_ts_checks()
        self.rxn_3.ts_species.final_xyz = parse_geometry(log_file_path=self.job1.local_path_to_output_file)
        self.assertFalse(self.rxn_3.ts_species.ts_checks['NMD'])
        ts.check_normal_mode_displacement(reaction=self.rxn_3, job=self.job1)
        self.assertTrue(self.rxn_3.ts_species.ts_checks['NMD'])

    def make_c3h7_intra_h_rxn(self):
        """A helper for building the C[CH]C <=> [CH2]CC reaction with a fresh TS species."""
        rxn = ARCReaction(r_species=[ARCSpecies(label='C[CH]C', smiles='C[CH]C', xyz=self.r_xyz_2a)],
                          p_species=[ARCSpecies(label='[CH2]CC', smiles='[CH2]CC', xyz=self.p_xyz_2)])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=self.ts_xyz_2)
        rxn.ts_species.mol_from_xyz()
        return rxn

    def test_check_normal_mode_displacement_only_defaults_a_none_amplitude(self):
        """Test that only ``None`` selects the default amplitude, while other values reach the analysis as given."""
        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite',
                                                           'TS_intra_H_migration_CBS-QB3.out')
        requested_amplitudes = [None, 0, 0.0, [], [0.25, 0.5]]
        forwarded = list()

        def record_amplitude(reaction, job, amplitude):
            forwarded.append(amplitude)
            return None

        with patch('arc.checks.ts.analyze_ts_normal_mode_displacement', side_effect=record_amplitude):
            for requested in requested_amplitudes:
                rxn = self.make_c3h7_intra_h_rxn()
                rxn.ts_species.populate_ts_checks()
                ts.check_normal_mode_displacement(reaction=rxn, job=self.job1, amplitude=requested)
        self.assertEqual(forwarded, [ts.DEFAULT_AMPLITUDE, 0, 0.0, [], [0.25, 0.5]])

    def test_check_normal_mode_displacement_with_an_amplitude_that_probes_nothing(self):
        """Test that an amplitude leaving no displacement to probe does not fall through to the default."""
        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite',
                                                           'TS_intra_H_migration_CBS-QB3.out')
        rxn = self.make_c3h7_intra_h_rxn()
        rxn.ts_species.final_xyz = parse_geometry(log_file_path=self.job1.local_path_to_output_file)
        rxn.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=rxn, job=self.job1)
        self.assertIs(rxn.ts_species.ts_checks['NMD'], True)
        for amplitude in [0, []]:
            rxn.ts_species.populate_ts_checks()
            ts.check_normal_mode_displacement(reaction=rxn, job=self.job1, amplitude=amplitude)
            self.assertIs(rxn.ts_species.ts_checks['NMD'], False, msg=f'amplitude {amplitude}')

    def test_check_ts_does_not_promote_an_unknown_nmd_verdict_when_skipping(self):
        """Test that skip_nmd promotes a failed NMD check and leaves an unperformed one unknown."""
        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'orca_neg_freq_ts.out')
        rxn = self.make_c3h7_intra_h_rxn()
        rxn.ts_species.populate_ts_checks()
        ts.check_ts(reaction=rxn, job=self.job1, checks=['NMD'], skip_nmd=True)
        self.assertIsNone(rxn.ts_species.ts_checks['NMD'])

        def fail_the_check(reaction, job, amplitude):
            return False

        rxn = self.make_c3h7_intra_h_rxn()
        rxn.ts_species.populate_ts_checks()
        with patch('arc.checks.ts.analyze_ts_normal_mode_displacement', side_effect=fail_the_check):
            ts.check_ts(reaction=rxn, job=self.job1, checks=['NMD'], skip_nmd=True)
        self.assertIs(rxn.ts_species.ts_checks['NMD'], True)

    def test_check_ts_records_whether_a_passing_nmd_verdict_was_forced(self):
        """Test that only a failed check promoted by skip_nmd is recorded as forced."""
        for verdict, skip_nmd, forced in ((False, True, True), (True, True, False), (True, False, False),
                                         (False, False, False)):
            rxn = self.make_c3h7_intra_h_rxn()
            rxn.ts_species.populate_ts_checks()
            with patch('arc.checks.ts.analyze_ts_normal_mode_displacement', return_value=verdict):
                ts.check_ts(reaction=rxn, job=self.job1, checks=['NMD'], skip_nmd=skip_nmd)
            self.assertIs(rxn.ts_species.nmd_record.get('forced', False), forced,
                          msg=f'verdict={verdict}, skip_nmd={skip_nmd}')
        rxn.ts_species.nmd_record = {'forced': True, 'mode_index': 1}
        with patch('arc.checks.ts.analyze_ts_normal_mode_displacement', return_value=True):
            ts.check_normal_mode_displacement(reaction=rxn, job=self.job1)
        self.assertEqual(rxn.ts_species.nmd_record, dict())

    def test_check_ts_reaches_the_rotor_block_for_an_ess_reporting_no_normal_modes(self):
        """Test that a TS whose ESS reports no normal mode displacements does not end the run."""
        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'orca_neg_freq_ts.out')
        for skip_nmd in [True, False]:
            rxn = self.make_c3h7_intra_h_rxn()
            rxn.ts_species.populate_ts_checks()
            for check in ['E0', 'e_elect', 'freq']:
                rxn.ts_species.ts_checks[check] = True
            ts.check_ts(reaction=rxn, job=self.job1, checks=['NMD'], skip_nmd=skip_nmd)
            self.assertIsNone(rxn.ts_species.ts_checks['NMD'], msg=f'skip_nmd={skip_nmd}')
            self.assertTrue(ts.ts_passed_checks(species=rxn.ts_species, exemptions=['E0', 'warnings']))

    def test_get_rxn_zone_atom_indices_for_an_ess_reporting_no_normal_modes(self):
        """Test that a log file yielding no normal mode displacements yields an empty reaction zone."""
        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'orca_neg_freq_ts.out')
        rxn = self.make_c3h7_intra_h_rxn()
        with patch.object(ts.parser, 'get_normal_mode_displacement', return_value=None):
            self.assertEqual(ts.get_rxn_zone_atom_indices(reaction=rxn, job=self.job1), list())
            ts.invalidate_rotors_with_both_pivots_in_a_reactive_zone(rxn, self.job1)
        self.assertEqual([key for key, rotor in rxn.ts_species.rotors_dict.items()
                          if 'pivTS' in rotor['invalidation_reason']], list())

    def test_ts_passed_checks_nmd(self):
        """Test that ts_passed_checks() treats the three-valued NMD check correctly."""
        spc = ARCSpecies(label='TS', is_ts=True)
        spc.populate_ts_checks()
        for key in ['E0', 'e_elect', 'IRC', 'freq']:
            spc.ts_checks[key] = True

        spc.ts_checks['NMD'] = True
        self.assertTrue(ts.ts_passed_checks(spc, exemptions=['warnings']))

        spc.ts_checks['NMD'] = None
        self.assertTrue(ts.ts_passed_checks(spc, exemptions=['warnings']))

        spc.ts_checks['NMD'] = False
        self.assertFalse(ts.ts_passed_checks(spc, exemptions=['warnings']))

    def test_invalidate_rotors_with_both_pivots_in_a_reactive_zone(self):
        """Test the invalidate_rotors_with_both_pivots_in_a_reactive_zone() function."""
        ts_spc_1 = ARCSpecies(label='TS', is_ts=True, xyz=self.ts_xyz_1, multiplicity=2)
        ts_spc_1.mol_from_xyz()
        ts_spc_1.determine_rotors()
        # Manually add the rotor that breaks the TS, it is not identified automatically:
        ts_spc_1.rotors_dict[1] = {'pivots': [2, 3],
                                   'top': [4, 8],
                                   'scan': [1, 2, 3, 4],
                                   'torsion': [0, 1, 2, 3],
                                   'success': None,
                                   'invalidation_reason': '',
                                   'dimensions': 1}
        rxn = ARCReaction(r_species=[ARCSpecies(label='N#[CH]', smiles='N#C'), ARCSpecies(label='[CH2][OH]', smiles='[CH2]O')],
                          p_species=[ARCSpecies(label='N=CCO', smiles='[N]=CCO')])
        rxn.ts_species = ts_spc_1
        rxn_zone_atom_indices = [1, 2]
        ts.invalidate_rotors_with_both_pivots_in_a_reactive_zone(reaction=rxn,
                                                                 job=self.job1,
                                                                 rxn_zone_atom_indices=rxn_zone_atom_indices)
        self.assertEqual(ts_spc_1.rotors_dict[0]['pivots'], [1, 2])
        self.assertEqual(ts_spc_1.rotors_dict[0]['invalidation_reason'], '')
        self.assertIsNone(ts_spc_1.rotors_dict[0]['success'])
        self.assertEqual(ts_spc_1.rotors_dict[1]['pivots'], [2, 3])
        self.assertEqual(ts_spc_1.rotors_dict[1]['scan'], [1, 2, 3, 4])
        self.assertEqual(ts_spc_1.rotors_dict[1]['invalidation_reason'],
                         'Pivots participate in the TS reaction zone (code: pivTS). ')
        self.assertEqual(ts_spc_1.rotors_dict[1]['success'], False)

        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'TS_intra_H_migration_CBS-QB3.out')
        self.rxn_2a.ts_species.populate_ts_checks()
        ts.invalidate_rotors_with_both_pivots_in_a_reactive_zone(reaction=self.rxn_2a,
                                                                 job=self.job1)
        self.assertEqual(self.rxn_2a.ts_species.rotors_dict[0]['pivots'], [1, 2])
        self.assertEqual(self.rxn_2a.ts_species.rotors_dict[0]['scan'], [5, 1, 2, 3])
        self.assertEqual(self.rxn_2a.ts_species.rotors_dict[0]['invalidation_reason'], '')
        self.assertIsNone(self.rxn_2a.ts_species.rotors_dict[0]['success'])
        self.assertEqual(self.rxn_2a.ts_species.rotors_dict[1]['pivots'], [2, 3])
        self.assertEqual(self.rxn_2a.ts_species.rotors_dict[1]['scan'], [1, 2, 3, 9])
        self.assertEqual(self.rxn_2a.ts_species.rotors_dict[1]['invalidation_reason'],
                         'Pivots participate in the TS reaction zone (code: pivTS). ')
        self.assertEqual(self.rxn_2a.ts_species.rotors_dict[1]['success'], False)

    def test_get_rxn_zone_atom_indices(self):
        """Test the get_rxn_zone_atom_indices() function."""
        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'TS0_composite_2102.out')
        rxn_zone_atom_indices = ts.get_rxn_zone_atom_indices(reaction=self.rxn_4, job=self.job1)
        self.assertEqual(rxn_zone_atom_indices, [14, 10, 3, 2, 16])

        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'TS0_composite_2044.out')
        rxn_zone_atom_indices = ts.get_rxn_zone_atom_indices(reaction=self.rxn_5, job=self.job1)
        self.assertEqual(rxn_zone_atom_indices, [5, 10, 1, 3, 0])

        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'composite', 'TS1_composite_695.out')
        rxn_zone_atom_indices = ts.get_rxn_zone_atom_indices(reaction=self.rxn_6, job=self.job1)
        self.assertEqual(rxn_zone_atom_indices, [7, 2, 9, 8, 1])  # Atom 10, the abstractor, is not moving

    def test_get_rms_from_normal_modes_disp(self):
        """Test the get_rms_from_normal_modes_disp() function."""
        rms = ts.get_rms_from_normal_mode_disp(normal_mode_disp=self.normal_modes_disp_1,
                                               freqs=np.array([-1000.3, 320.5], np.float64))
        self.assertEqual(rms, [0.07874007874011811, 0.07280109889280519, 0.0, 0.9914635646356349, 0.03605551275463989,
                               0.034641016151377546, 0.0, 0.033166247903554, 0.01414213562373095, 0.0])

        freqs_2, normal_modes_disp_2 = parse_normal_mode_displacement(
            os.path.join(ARC_TESTING_PATH, 'composite', 'TS0_composite_2102.out'))
        rms = ts.get_rms_from_normal_mode_disp(normal_mode_disp=normal_modes_disp_2,
                                               freqs=freqs_2,
                                               reaction=ARCReaction(r_species=[ARCSpecies(label='CCOOj', xyz=self.ccooj_xyz),
                                                                               ARCSpecies(label='C2H6', xyz=self.c2h6_xyz)],
                                                                    p_species=[ARCSpecies(label='CCOOH', smiles='CCOO'),
                                                                               ARCSpecies(label='C2H5', smiles='C[CH2]')])
                                               )
        self.assertTrue(almost_equal_lists(rms, [0.0,                    # 0
                                                 0.039223801763523365,   # 1
                                                 0.15236541607924428,    # 2
                                                 0.20024738798047312,    # 3 * (the abstracting O)
                                                 0.0,                    # 4
                                                 0.0,                    # 5
                                                 0.010042962189013961,   # 6
                                                 0.0,                    # 7
                                                 0.010042962189013961,   # 8
                                                 0.039223801763523365,   # 9
                                                 0.33968808760216107,    # 10 * (the abstracting C)
                                                 0.010042962189013961,   # 11
                                                 0.0,                    # 12
                                                 0.010042962189013961,   # 13
                                                 0.9804112316882941,     # 14 * (the abstracted H)
                                                 0.12176444538905733,    # 15
                                                 0.12462988320468919]))  # 16

    def test_get_expected_num_atoms_with_largest_normal_mode_disp(self):
        """Test the get_expected_num_atoms_with_largest_normal_mode_disp() function"""
        normal_disp_mode_rms = [0.01414213562373095, 0.05, 0.04, 0.5632938842203065, 0.7993122043357026,
                                0.08944271909999159, 0.10677078252031312, 0.09000000000000001, 0.05, 0.09433981132056604]
        num_of_atoms = ts.get_expected_num_atoms_with_largest_normal_mode_disp(normal_mode_disp_rms=normal_disp_mode_rms,
                                                                               ts_guesses=self.ts_1.ts_guesses)
        self.assertEqual(num_of_atoms, 4)

    def test_get_rxn_normal_mode_disp_atom_number(self):
        """Test the get_rxn_normal_mode_disp_atom_number() function."""
        with self.assertRaises(TypeError):
            ts.get_rxn_normal_mode_disp_atom_number('family', rms_list='family')
        with self.assertRaises(TypeError):
            ts.get_rxn_normal_mode_disp_atom_number('family', rms_list=['family'])
        with self.assertRaises(TypeError):
            ts.get_rxn_normal_mode_disp_atom_number('family', rms_list=15.215)
        with self.assertRaises(ValueError):
            self.assertEqual(ts.get_rxn_normal_mode_disp_atom_number(), 3)
        self.assertEqual(ts.get_rxn_normal_mode_disp_atom_number('default'), 3)
        self.assertEqual(ts.get_rxn_normal_mode_disp_atom_number('intra_H_migration'), 3)
        self.assertEqual(ts.get_rxn_normal_mode_disp_atom_number('intra_H_migration', rms_list=self.rms_list_1), 4)

    def test_check_irc_species_and_rxn(self):
        """Test the check_irc_species_and_rxn() function."""
        xyz_1 = """ O                 -0.41278100    1.32451200    1.69341300
                    C                 -0.36012400    0.71815900    0.68711000
                    C                  0.07168300   -0.73373200    0.45498600
                    O                 -0.09882100   -1.07908500   -0.89888400
                    O                  0.69058000   -0.12998800   -1.65496000
                    H                  0.05586000    0.59945600   -1.74796200
                    H                  1.11467600   -0.83984800    0.77895200
                    H                 -0.57663000   -1.40626900    1.02593400"""
        xyz_2 = """ O                 -0.43783600    1.29437500    1.69860500
                    C                 -0.27875600    0.77603900    0.63081800
                    C                  0.05866100   -0.70395300    0.45930700
                    O                 -0.05939300   -1.06149500   -0.93456800
                    O                  0.65685500   -0.22958000   -1.67443200
                    H                 -0.35607400    1.32805100   -0.32564200
                    H                  1.08243900   -0.89713600    0.79060300
                    H                 -0.63911300   -1.34159000    1.00373600"""
        rxn = ARCReaction(r_species=[ARCSpecies(label='R', smiles='O=[C]COO', xyz=xyz_1)],
                          p_species=[ARCSpecies(label='P', smiles='O=CCO[O]', xyz=xyz_2)])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        self.assertIsNone(rxn.ts_species.ts_checks['IRC'])
        ts.check_irc_species_and_rxn(xyz_1=parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out')),
                                     xyz_2=parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_2.out')),
                                     rxn=rxn,
                                     )
        self.assertTrue(rxn.ts_species.ts_checks['IRC'])

    def test_check_imaginary_frequencies(self):
        """Test the check_imaginary_frequencies() function."""
        imaginary_freqs = None
        self.assertTrue(ts.check_imaginary_frequencies(imaginary_freqs))
        imaginary_freqs = list()
        self.assertFalse(ts.check_imaginary_frequencies(imaginary_freqs))
        imaginary_freqs = [-300.14]
        self.assertTrue(ts.check_imaginary_frequencies(imaginary_freqs))
        imaginary_freqs = [-3.14]
        self.assertFalse(ts.check_imaginary_frequencies(imaginary_freqs))
        imaginary_freqs = [-500.80, -3.14]
        self.assertTrue(ts.check_imaginary_frequencies(imaginary_freqs))

    def test_perceive_irc_fragments_single_fragment(self):
        """Test _perceive_irc_fragments for a single connected molecule (O=[C]COO)."""
        xyz_1 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out'))
        frags = ts._perceive_irc_fragments(xyz_1, charge=0)
        self.assertIsNotNone(frags)
        self.assertEqual(len(frags), 1)
        self.assertTrue(frags[0].is_isomorphic(ARCSpecies(label='R', smiles='O=[C]COO').mol))

    def test_perceive_irc_fragments_two_fragments(self):
        """Test _perceive_irc_fragments for well-separated water + methane."""
        coords = (
            (0.0000, 0.0000, 0.1173),     # O
            (0.0000, 0.7572, -0.4692),     # H
            (0.0000, -0.7572, -0.4692),    # H
            (10.0000, 0.0000, 0.0000),     # C
            (10.6276, 0.6276, 0.6276),     # H
            (10.6276, -0.6276, -0.6276),   # H
            (9.3724, 0.6276, -0.6276),     # H
            (9.3724, -0.6276, 0.6276),     # H
        )
        symbols = ('O', 'H', 'H', 'C', 'H', 'H', 'H', 'H')
        xyz = xyz_from_data(coords=coords, symbols=symbols)
        frags = ts._perceive_irc_fragments(xyz, charge=0)
        self.assertIsNotNone(frags)
        self.assertEqual(len(frags), 2)
        water_mol = ARCSpecies(label='water', smiles='O').mol
        methane_mol = ARCSpecies(label='methane', smiles='C').mol
        # Fragment order follows atom indices: water (atoms 0-2) then methane (atoms 3-7)
        self.assertTrue(frags[0].is_isomorphic(water_mol))
        self.assertTrue(frags[1].is_isomorphic(methane_mol))

    def test_perceive_irc_fragments_charge_handling(self):
        """Test that charge is correctly handled for both single and multi-fragment systems."""
        # 1. Single fragment case: Ensure the +1 charge is passed through
        # Using the same XYZ but pretending it's a cation
        xyz_1 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out'))
        frags_single = ts._perceive_irc_fragments(xyz_1, charge=1)

        self.assertIsNotNone(frags_single)
        self.assertEqual(len(frags_single), 1)
        # Verify the perceived molecule actually inherited the +1 charge
        self.assertEqual(frags_single[0].get_net_charge(), 1)

        # 2. Multi-fragment case: charge is distributed across fragments
        # H2 + H2O with total charge=0 should give two neutral fragments
        xyz_multi = {
            'symbols': ('H', 'H', 'O', 'H', 'H'),
            'coords': ((0, 0, 0), (0, 0, 0.74),  # H2
                       (5, 0, 0), (5.7, 0.7, 0), (5.7, -0.7, 0)) # H2O
        }
        frags_multi = ts._perceive_irc_fragments(xyz_multi, charge=0)

        self.assertIsNotNone(frags_multi)
        self.assertEqual(len(frags_multi), 2)
        for frag in frags_multi:
            self.assertEqual(frag.get_net_charge(), 0)

    def test_assign_fragments_to_species_single(self):
        """Test _assign_fragments_to_species with a single fragment."""
        r_spc = ARCSpecies(label='R', smiles='O=[C]COO')
        p_spc = ARCSpecies(label='P', smiles='O=CCO[O]')
        xyz_1 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out'))
        frags = ts._perceive_irc_fragments(xyz_1, charge=0)
        self.assertIsNotNone(frags)
        self.assertEqual(ts._assign_fragments_to_species(frags, [r_spc.mol]), [0])
        self.assertIsNone(ts._assign_fragments_to_species(frags, [p_spc.mol]))

    def test_assign_fragments_to_species_multi(self):
        """Test _assign_fragments_to_species with two well-separated fragments."""
        coords = (
            (0.0000, 0.0000, 0.1173),
            (0.0000, 0.7572, -0.4692),
            (0.0000, -0.7572, -0.4692),
            (10.0000, 0.0000, 0.0000),
            (10.6276, 0.6276, 0.6276),
            (10.6276, -0.6276, -0.6276),
            (9.3724, 0.6276, -0.6276),
            (9.3724, -0.6276, 0.6276),
        )
        symbols = ('O', 'H', 'H', 'C', 'H', 'H', 'H', 'H')
        xyz = xyz_from_data(coords=coords, symbols=symbols)
        frags = ts._perceive_irc_fragments(xyz, charge=0)
        self.assertIsNotNone(frags)

        water_mol = ARCSpecies(label='water', smiles='O').mol
        methane_mol = ARCSpecies(label='methane', smiles='C').mol
        nh3_mol = ARCSpecies(label='NH3', smiles='N').mol

        # Correct match (either order of expected species should work due to permutations)
        self.assertIsNotNone(ts._assign_fragments_to_species(frags, [water_mol, methane_mol]))
        self.assertIsNotNone(ts._assign_fragments_to_species(frags, [methane_mol, water_mol]))
        self.assertIsNone(ts._assign_fragments_to_species(frags, [water_mol, nh3_mol]))
        self.assertIsNone(ts._assign_fragments_to_species(frags, [water_mol]))
        self.assertIsNone(ts._assign_fragments_to_species(frags, [water_mol, methane_mol, nh3_mol]))

    def test_assign_fragments_to_species_empty(self):
        """Test _assign_fragments_to_species edge cases."""
        self.assertEqual(ts._assign_fragments_to_species([], []), [])
        water_mol = ARCSpecies(label='water', smiles='O').mol
        self.assertIsNone(ts._assign_fragments_to_species([], [water_mol]))

    def test_check_irc_isomorphism_path(self):
        """Test that the full check_irc_species_and_rxn uses isomorphism when mol objects are available."""
        xyz_1 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out'))
        xyz_2 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_2.out'))
        rxn = ARCReaction(r_species=[ARCSpecies(label='R', smiles='O=[C]COO', xyz=xyz_1)],
                          p_species=[ARCSpecies(label='P', smiles='O=CCO[O]', xyz=xyz_2)])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        # Both species have mol objects, so isomorphism path should be used
        self.assertIsNotNone(rxn.r_species[0].mol)
        self.assertIsNotNone(rxn.p_species[0].mol)
        ts.check_irc_species_and_rxn(xyz_1=xyz_1, xyz_2=xyz_2, rxn=rxn)
        self.assertTrue(rxn.ts_species.ts_checks['IRC'])

    def test_check_irc_swapped_endpoints(self):
        """Test that check_irc_species_and_rxn works when IRC endpoints are swapped."""
        xyz_1 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out'))
        xyz_2 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_2.out'))
        rxn = ARCReaction(r_species=[ARCSpecies(label='R', smiles='O=[C]COO', xyz=xyz_1)],
                          p_species=[ARCSpecies(label='P', smiles='O=CCO[O]', xyz=xyz_2)])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        # Pass endpoints in reverse order (xyz_2 as first, xyz_1 as second)
        ts.check_irc_species_and_rxn(xyz_1=xyz_2, xyz_2=xyz_1, rxn=rxn)
        self.assertTrue(rxn.ts_species.ts_checks['IRC'])

    def test_check_irc_wrong_species(self):
        """Test that check_irc_species_and_rxn returns False for mismatched species."""
        xyz_1 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out'))
        xyz_2 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_2.out'))
        # Use wrong product species
        rxn = ARCReaction(r_species=[ARCSpecies(label='R', smiles='O=[C]COO', xyz=xyz_1)],
                          p_species=[ARCSpecies(label='P_wrong', smiles='O=C(O)C[O]', multiplicity=2)])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        ts.check_irc_species_and_rxn(xyz_1=xyz_1, xyz_2=xyz_2, rxn=rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], False)

    def test_check_irc_bond_list_tier_does_not_bond_a_dissociated_atom(self):
        """
        Test that the bond-list tier does not bond a dissociated fragment to its nearest neighbor.

        The bond-list tier is the only tier allowed to reject a TS, so it must perceive bonds
        without assuming a single connected molecule. Otherwise a bare H in an endpoint is bonded
        to whatever atom happens to be closest, no matter how far away, and correct IRC endpoints
        are rejected. Both reactions below reach the bond-list tier because the triplet atom is
        not perceived as a triplet, so the isomorphism tier cannot match it.
        """
        for r_smiles, r_multiplicity, p_smiles, symbols, coords_r, coords_p in [
                ('[O]', 3, '[OH]', ('O', 'H', 'H'),
                 ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 0.0, 0.74)),
                 ((0.0, 0.0, 0.0), (3.2, 0.0, 0.0), (0.0, 0.0, 0.97))),
                ('[S]', 3, '[SH]', ('S', 'H', 'H'),
                 ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 0.0, 0.74)),
                 ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 0.0, 1.34)))]:
            with self.subTest(r_smiles=r_smiles):
                rxn = ARCReaction(r_species=[ARCSpecies(label='X', smiles=r_smiles, multiplicity=r_multiplicity),
                                             ARCSpecies(label='H2', smiles='[H][H]')],
                                  p_species=[ARCSpecies(label='XH', smiles=p_smiles),
                                             ARCSpecies(label='H', smiles='[H]')])
                rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
                xyz_1 = xyz_from_data(coords=coords_r, symbols=symbols)
                xyz_2 = xyz_from_data(coords=coords_p, symbols=symbols)
                ts.check_irc_species_and_rxn(xyz_1=xyz_1, xyz_2=xyz_2, rxn=rxn)
                self.assertIs(rxn.ts_species.ts_checks['IRC'], True)

    def test_check_irc_identical_endpoints(self):
        """
        Test that two identical IRC endpoints are a positive IRC failure (False, not None).

        Both endpoints re-perceive as the product, i.e., the "TS" connects P <=> P.
        """
        xyz_1 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out'))
        xyz_2 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_2.out'))
        rxn = ARCReaction(r_species=[ARCSpecies(label='R', smiles='O=[C]COO', xyz=xyz_1)],
                          p_species=[ARCSpecies(label='P', smiles='O=CCO[O]', xyz=xyz_2)])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        ts.check_irc_species_and_rxn(xyz_1=xyz_2, xyz_2=xyz_2, rxn=rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], False)

    @staticmethod
    def _make_ch4_oh_endpoints():
        """Endpoint geometries of CH4 + OH <=> CH3 + H2O, in one atom order: C0 H1 H2 H3 H4 O5 H6."""
        symbols = ('C', 'H', 'H', 'H', 'H', 'O', 'H')
        reactant_coords = ((0.0, 0.0, 0.0), (0.629, 0.629, 0.629), (-0.629, -0.629, 0.629),
                           (-0.629, 0.629, -0.629), (0.629, -0.629, -0.629), (6.0, 0.0, 0.0), (6.97, 0.0, 0.0))
        product_coords = ((0.0, 0.0, 0.0), (1.08, 0.0, 0.0), (-0.54, 0.935, 0.0), (-0.54, -0.935, 0.0),
                          (5.7597, 0.9294, 0.0), (6.0, 0.0, 0.0), (6.96, 0.0, 0.0))
        return (xyz_from_data(coords=reactant_coords, symbols=symbols),
                xyz_from_data(coords=product_coords, symbols=symbols))

    def _make_ch4_oh_rxn(self):
        rxn = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C'),
                                     ARCSpecies(label='OH', smiles='[OH]')],
                          p_species=[ARCSpecies(label='CH3', smiles='[CH3]'),
                                     ARCSpecies(label='H2O', smiles='O')])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        return rxn

    def test_check_irc_records_which_endpoint_atoms_belong_to_which_participant(self):
        """Test that an isomorphism verdict records the atom indices of each participant, per endpoint geometry."""
        xyz_r, xyz_p = self._make_ch4_oh_endpoints()
        expected = {'reactants': {'endpoint': 1, 'endpoint_label': 'IRC_TS_1',
                                  'participants': [{'label': 'CH4', 'position': 1, 'occurrence': 1,
                                                    'atom_indices': [0, 1, 2, 3, 4]},
                                                   {'label': 'OH', 'position': 2, 'occurrence': 1,
                                                    'atom_indices': [5, 6]}]},
                    'products': {'endpoint': 2, 'endpoint_label': 'IRC_TS_2',
                                 'participants': [{'label': 'CH3', 'position': 1, 'occurrence': 1,
                                                   'atom_indices': [0, 1, 2, 3]},
                                                  {'label': 'H2O', 'position': 2, 'occurrence': 1,
                                                   'atom_indices': [4, 5, 6]}]},
                    'sides_distinguishable': True, 'atom_order_matches_ts': None}
        rxn = self._make_ch4_oh_rxn()
        ts.check_irc_species_and_rxn(xyz_1=xyz_r, xyz_2=xyz_p, rxn=rxn, endpoint_labels=('IRC_TS_1', 'IRC_TS_2'))
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertEqual(rxn.ts_species.irc_participant_mapping, expected)

        rxn = self._make_ch4_oh_rxn()
        ts.check_irc_species_and_rxn(xyz_1=xyz_p, xyz_2=xyz_r, rxn=rxn, endpoint_labels=('IRC_TS_1', 'IRC_TS_2'))
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        mapping = rxn.ts_species.irc_participant_mapping
        self.assertEqual((mapping['reactants']['endpoint'], mapping['reactants']['endpoint_label']), (2, 'IRC_TS_2'))
        self.assertEqual((mapping['products']['endpoint'], mapping['products']['endpoint_label']), (1, 'IRC_TS_1'))
        self.assertEqual([p['atom_indices'] for p in mapping['reactants']['participants']], [[0, 1, 2, 3, 4], [5, 6]])
        self.assertEqual([p['atom_indices'] for p in mapping['products']['participants']], [[0, 1, 2, 3], [4, 5, 6]])

        rxn = self._make_ch4_oh_rxn()
        ts.check_irc_species_and_rxn(xyz_1=xyz_r, xyz_2=xyz_p, rxn=rxn)
        self.assertIsNone(rxn.ts_species.irc_participant_mapping['reactants']['endpoint_label'])

    def test_check_irc_participant_mapping_expands_a_repeated_species_per_occurrence(self):
        """Test that CH3 + CH3 <=> C2H6 lists the two CH3 occurrences as separate participants."""
        symbols = ('C', 'H', 'H', 'H', 'C', 'H', 'H', 'H')
        xyz_r = xyz_from_data(coords=((0.0, 0.0, 0.0), (1.08, 0.0, 0.0), (-0.54, 0.935, 0.0), (-0.54, -0.935, 0.0),
                                      (8.0, 0.0, 0.0), (9.08, 0.0, 0.0), (7.46, 0.935, 0.0), (7.46, -0.935, 0.0)),
                              symbols=symbols)
        xyz_p = xyz_from_data(coords=((0.0, 0.0, 0.0), (-0.36, 1.03, 0.0), (-0.36, -0.51, 0.89),
                                      (-0.36, -0.51, -0.89), (1.54, 0.0, 0.0), (1.90, 1.03, 0.0),
                                      (1.90, -0.51, 0.89), (1.90, -0.51, -0.89)),
                              symbols=symbols)
        rxn = ARCReaction(label='CH3 + CH3 <=> C2H6',
                          r_species=[ARCSpecies(label='CH3', smiles='[CH3]')],
                          p_species=[ARCSpecies(label='C2H6', smiles='CC')])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        ts.check_irc_species_and_rxn(xyz_1=xyz_r, xyz_2=xyz_p, rxn=rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        mapping = rxn.ts_species.irc_participant_mapping
        self.assertEqual(mapping['reactants']['participants'],
                         [{'label': 'CH3', 'position': 1, 'occurrence': 1, 'atom_indices': [0, 1, 2, 3]},
                          {'label': 'CH3', 'position': 2, 'occurrence': 2, 'atom_indices': [4, 5, 6, 7]}])
        self.assertEqual(mapping['products']['participants'],
                         [{'label': 'C2H6', 'position': 1, 'occurrence': 1, 'atom_indices': list(range(8))}])

    def test_check_irc_participant_mapping_states_whether_the_irc_geometries_follow_from_the_ts(self):
        """Test atom_order_matches_ts: true only for congruent starts, a followed endpoint chain and agreeing
        elements; false for a contradiction; null when unknown."""
        xyz_r, xyz_p = self._make_ch4_oh_endpoints()
        scratch = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, scratch, True)
        shifted = xyz_from_data(coords=tuple((x + 0.5, y, z) for x, y, z in xyz_r['coords']), symbols=xyz_r['symbols'])
        swapped_coords = list(xyz_r['coords'])
        swapped_coords[0], swapped_coords[5] = swapped_coords[5], swapped_coords[0]
        swapped = xyz_from_data(coords=tuple(swapped_coords), symbols=xyz_r['symbols'])
        swapped_endpoint_coords = list(xyz_p['coords'])
        swapped_endpoint_coords[1], swapped_endpoint_coords[4] = swapped_endpoint_coords[4], swapped_endpoint_coords[1]
        swapped_endpoint = xyz_from_data(coords=tuple(swapped_endpoint_coords), symbols=xyz_p['symbols'])
        reordered = [5, 6, 0, 1, 2, 3, 4]
        reordered_r = xyz_from_data(coords=tuple(xyz_r['coords'][i] for i in reordered),
                                    symbols=tuple(xyz_r['symbols'][i] for i in reordered))
        reordered_p = xyz_from_data(coords=tuple(xyz_p['coords'][i] for i in reordered),
                                    symbols=tuple(xyz_p['symbols'][i] for i in reordered))

        def write(name, start, last, first=None, final=None):
            paths = {'irc': [], 'endpoint': []}
            for i, (irc_last, endpoint_final) in enumerate(zip(last, final or last)):
                irc_path, endpoint_path = (os.path.join(scratch, f'{name}_{kind}_{i}.out') for kind in ('irc', 'opt'))
                TestTsAtomMap._gaussian_irc_log(irc_path, start, last=irc_last)
                TestTsAtomMap._gaussian_irc_log(endpoint_path, (first or last)[i], last=endpoint_final)
                paths['irc'].append(irc_path)
                paths['endpoint'].append(endpoint_path)
            return paths

        good = write('good', xyz_r, (xyz_r, xyz_p))
        cases = (
            ('congruent', xyz_r, (xyz_r, xyz_p), good['irc'], good['endpoint'], True),
            ('reoriented start', xyz_r, (xyz_r, xyz_p), write('shift', shifted, (xyz_r, xyz_p))['irc'],
             good['endpoint'], True),
            ('differing start', xyz_r, (xyz_r, xyz_p), write('swap', swapped, (xyz_r, xyz_p))['irc'],
             good['endpoint'], False),
            ('endpoint swapped after the optimization', xyz_r, (xyz_r, swapped_endpoint), good['irc'],
             good['endpoint'], False),
            ('optimization started elsewhere', xyz_r, (xyz_r, xyz_p), good['irc'],
             write('moved', xyz_r, (xyz_r, xyz_p), first=(xyz_r, swapped_endpoint))['endpoint'], False),
            ('no logs', xyz_r, (xyz_r, xyz_p), None, None, None),
            ('no endpoint logs', xyz_r, (xyz_r, xyz_p), good['irc'], None, None),
            ('no TS', None, (xyz_r, xyz_p), good['irc'], good['endpoint'], None),
            ('endpoint elements differ from the TS', xyz_r, (reordered_r, reordered_p),
             write('elements', xyz_r, (reordered_r, reordered_p))['irc'],
             write('elements', xyz_r, (reordered_r, reordered_p))['endpoint'], False),
        )
        for name, ts_xyz, endpoints, irc_paths, endpoint_paths, expected in cases:
            with self.subTest(case=name):
                rxn = self._make_ch4_oh_rxn()
                rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=ts_xyz)
                ts.check_irc_species_and_rxn(xyz_1=endpoints[0], xyz_2=endpoints[1], rxn=rxn, irc_log_paths=irc_paths,
                                             endpoint_log_paths=endpoint_paths)
                self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
                self.assertIs(rxn.ts_species.irc_participant_mapping['atom_order_matches_ts'], expected)

    def test_check_irc_participant_mapping_states_whether_the_sides_are_distinguishable(self):
        """Test that graph-isomorphic reactants and products are flagged, since the endpoint naming is a convention."""
        xyz_r, xyz_p = self._make_ch4_oh_endpoints()
        rxn = self._make_ch4_oh_rxn()
        ts.check_irc_species_and_rxn(xyz_1=xyz_r, xyz_2=xyz_p, rxn=rxn)
        self.assertIs(rxn.ts_species.irc_participant_mapping['sides_distinguishable'], True)
        coords = ((0.0, 0.0, 0.0), (1.08, 0.0, 0.0), (-0.54, 0.935, 0.0), (-0.54, -0.935, 0.0),
                  (8.0, 0.0, 0.0), (8.629, 0.629, 0.629), (7.371, -0.629, 0.629), (7.371, 0.629, -0.629),
                  (8.629, -0.629, -0.629))
        xyz = xyz_from_data(coords=coords, symbols=('C', 'H', 'H', 'H', 'C', 'H', 'H', 'H', 'H'))
        rxn = ARCReaction(label='CH3 + CH4 <=> CH4 + CH3',
                          r_species=[ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='CH4', smiles='C')],
                          p_species=[ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='CH3', smiles='[CH3]')])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        ts.check_irc_species_and_rxn(xyz_1=xyz, xyz_2=xyz, rxn=rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertIs(rxn.ts_species.irc_participant_mapping['sides_distinguishable'], False)

    def test_a_participant_mapping_failure_keeps_the_verdict(self):
        """Test that a failure while building the mapping leaves the isomorphism verdict, and no mapping."""
        xyz_r, xyz_p = self._make_ch4_oh_endpoints()
        for target in ('_get_irc_participant_mapping', 'get_irc_start_geometry_reason'):
            with self.subTest(failing=target):
                rxn = self._make_ch4_oh_rxn()
                rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=xyz_r)
                rxn.ts_species.irc_participant_mapping = {'stale': True}
                with patch.object(ts, target, side_effect=RuntimeError('boom')):
                    ts.check_irc_species_and_rxn(xyz_1=xyz_r, xyz_2=xyz_p, rxn=rxn)
                self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
                self.assertIsNone(rxn.ts_species.irc_participant_mapping)

    def test_check_irc_participant_mapping_is_none_without_an_isomorphism_verdict(self):
        """Test that the bond-list fallback, a failed check and a stale record all leave no participant mapping."""
        for r_smiles, r_multiplicity, p_smiles, symbols, coords_r, coords_p in [
                ('[O]', 3, '[OH]', ('O', 'H', 'H'),
                 ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 0.0, 0.74)),
                 ((0.0, 0.0, 0.0), (3.2, 0.0, 0.0), (0.0, 0.0, 0.97)))]:
            with self.subTest(case='bond-list fallback'):
                rxn = ARCReaction(r_species=[ARCSpecies(label='X', smiles=r_smiles, multiplicity=r_multiplicity),
                                             ARCSpecies(label='H2', smiles='[H][H]')],
                                  p_species=[ARCSpecies(label='XH', smiles=p_smiles),
                                             ARCSpecies(label='H', smiles='[H]')])
                rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
                rxn.ts_species.irc_participant_mapping = {'stale': True}
                ts.check_irc_species_and_rxn(xyz_1=xyz_from_data(coords=coords_r, symbols=symbols),
                                             xyz_2=xyz_from_data(coords=coords_p, symbols=symbols),
                                             rxn=rxn)
                self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
                self.assertIsNone(rxn.ts_species.irc_participant_mapping)
        xyz_r, xyz_p = self._make_ch4_oh_endpoints()
        rxn = self._make_ch4_oh_rxn()
        rxn.ts_species.irc_participant_mapping = {'stale': True}
        ts.check_irc_species_and_rxn(xyz_1=xyz_r, xyz_2=xyz_r, rxn=rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], False)
        self.assertIsNone(rxn.ts_species.irc_participant_mapping)

    def test_check_irc_verdict_is_that_of_the_boolean_isomorphism_logic(self):
        """Test that the verdict equals the one the boolean fragment matching decides, for every endpoint pairing."""
        xyz_r, xyz_p = self._make_ch4_oh_endpoints()
        rxn = self._make_ch4_oh_rxn()
        reactants, products = rxn.get_reactants_and_products(return_copies=True)
        r_mols, p_mols = [r.mol for r in reactants], [p.mol for p in products]
        for first, second in ((xyz_r, xyz_p), (xyz_p, xyz_r), (xyz_r, xyz_r), (xyz_p, xyz_p)):
            frags_1, frags_2 = ts._perceive_irc_fragments(first), ts._perceive_irc_fragments(second)

            def match(fragments, mols):
                return ts._assign_fragments_to_species(fragments, mols) is not None

            expected = (match(frags_1, r_mols) and match(frags_2, p_mols)) \
                or (match(frags_1, p_mols) and match(frags_2, r_mols))
            rxn = self._make_ch4_oh_rxn()
            ts.check_irc_species_and_rxn(xyz_1=first, xyz_2=second, rxn=rxn)
            self.assertIs(rxn.ts_species.ts_checks['IRC'], expected)
            self.assertEqual(rxn.ts_species.irc_participant_mapping is not None, expected)

    def test_assign_fragments_to_species(self):
        """Test that _assign_fragments_to_species returns the expected index of each fragment."""
        coords = ((0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692),
                  (10.0, 0.0, 0.0), (10.6276, 0.6276, 0.6276), (10.6276, -0.6276, -0.6276),
                  (9.3724, 0.6276, -0.6276), (9.3724, -0.6276, 0.6276))
        xyz = xyz_from_data(coords=coords, symbols=('O', 'H', 'H', 'C', 'H', 'H', 'H', 'H'))
        frags = ts._perceive_irc_fragments(xyz, charge=0)
        self.assertEqual(ts._get_irc_fragment_atom_indices(xyz), [[0, 1, 2], [3, 4, 5, 6, 7]])
        water_mol = ARCSpecies(label='water', smiles='O').mol
        methane_mol = ARCSpecies(label='methane', smiles='C').mol
        self.assertEqual(ts._assign_fragments_to_species(frags, [water_mol, methane_mol]), [0, 1])
        self.assertEqual(ts._assign_fragments_to_species(frags, [methane_mol, water_mol]), [1, 0])
        self.assertIsNone(ts._assign_fragments_to_species(frags, [water_mol, water_mol]))
        self.assertEqual(ts._assign_fragments_to_species([], []), [])

    def test_check_irc_unknown_if_no_comparison_was_performed(self):
        """
        Test that the IRC check is None (unknown), not False, if no comparison could be performed.

        Neither the isomorphism check nor the bond-list fallback can be carried out here.
        """
        xyz_1 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out'))
        xyz_2 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_2.out'))
        rxn = ARCReaction(r_species=[ARCSpecies(label='R', smiles='O=[C]COO', xyz=xyz_1)],
                          p_species=[ARCSpecies(label='P', smiles='O=CCO[O]', xyz=xyz_2)])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        with patch.object(ts, '_perceive_irc_fragments', return_value=None), \
                patch.object(rxn, 'get_bonds', side_effect=ReactionError('Cannot get bonds without an atom map.')):
            ts.check_irc_species_and_rxn(xyz_1=xyz_1, xyz_2=xyz_2, rxn=rxn)
        self.assertIsNone(rxn.ts_species.ts_checks['IRC'])

    def test_check_irc_isomorphism_mismatch_alone_is_not_a_failure(self):
        """
        Test that a negative isomorphism result alone leaves the IRC check undetermined.

        The isomorphism check is only the first tier, and the code falls back to the bond-list
        comparison whenever it does not match. If that fallback cannot run, nothing was
        conclusively compared, so the verdict must stay None rather than reject the TS.
        """
        xyz_1 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out'))
        xyz_2 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_2.out'))
        rxn = ARCReaction(r_species=[ARCSpecies(label='R', smiles='O=[C]COO', xyz=xyz_1)],
                          p_species=[ARCSpecies(label='P', smiles='O=CCO[O]', xyz=xyz_2)])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        with patch.object(rxn, 'get_bonds', side_effect=ReactionError('Cannot get bonds without an atom map.')):
            ts.check_irc_species_and_rxn(xyz_1=xyz_2, xyz_2=xyz_2, rxn=rxn)
        self.assertIsNone(rxn.ts_species.ts_checks['IRC'])

    def test_check_irc_identical_endpoints(self):
        """Test that two identical IRC endpoints are a positive IRC failure (False, not None)."""
        xyz_1 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out'))
        xyz_2 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_2.out'))
        rxn = ARCReaction(r_species=[ARCSpecies(label='R', smiles='O=[C]COO', xyz=xyz_1)],
                          p_species=[ARCSpecies(label='P', smiles='O=CCO[O]', xyz=xyz_2)])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        # Both endpoints re-perceive as the product, the "TS" connects P <=> P.
        ts.check_irc_species_and_rxn(xyz_1=xyz_2, xyz_2=xyz_2, rxn=rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], False)

    def test_check_irc_unknown_if_no_comparison_was_performed(self):
        """Test that the IRC check is None (unknown), not False, if no comparison could be performed."""
        xyz_1 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out'))
        xyz_2 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_2.out'))
        rxn = ARCReaction(r_species=[ARCSpecies(label='R', smiles='O=[C]COO', xyz=xyz_1)],
                          p_species=[ARCSpecies(label='P', smiles='O=CCO[O]', xyz=xyz_2)])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        # Neither the isomorphism check nor the bond-list fallback can be carried out.
        with patch.object(ts, '_perceive_irc_fragments', return_value=None), \
                patch.object(rxn, 'get_bonds', side_effect=ReactionError('Cannot get bonds without an atom map.')):
            ts.check_irc_species_and_rxn(xyz_1=xyz_1, xyz_2=xyz_2, rxn=rxn)
        self.assertIsNone(rxn.ts_species.ts_checks['IRC'])

    @staticmethod
    def _make_irc_endpoints(rxn, permutation=None):
        """
        Build IRC endpoint geometries in the reactant atom order from the species of ``rxn``, optionally permuted.

        The reactant endpoint holds the separated reactants, and the product endpoint holds the separated products
        with their atoms reordered according to the atom map of the reaction.
        """
        def concatenate(species_list):
            symbols, coords = list(), list()
            for i, spc in enumerate(species_list):
                xyz = spc.get_xyz()
                symbols.extend(xyz['symbols'])
                coords.extend((x + 8.0 * i, y, z) for x, y, z in xyz['coords'])
            return symbols, coords
        r_symbols, r_coords = concatenate(rxn.r_species)
        p_symbols, p_coords = concatenate(rxn.p_species)
        order = permutation if permutation is not None else list(range(len(r_symbols)))
        xyz_r = xyz_from_data(coords=[r_coords[i] for i in order], symbols=[r_symbols[i] for i in order])
        xyz_p = xyz_from_data(coords=[p_coords[rxn.atom_map[i]] for i in order],
                              symbols=[p_symbols[rxn.atom_map[i]] for i in order])
        return xyz_r, xyz_p

    def _check_irc_fallback(self, rxn, xyz_1, xyz_2):
        """Run check_irc_species_and_rxn() with molecule perception forced to fail and return the IRC verdict."""
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        with patch.object(ts, '_perceive_irc_fragments', return_value=None):
            ts.check_irc_species_and_rxn(xyz_1=xyz_1, xyz_2=xyz_2, rxn=rxn)
        return rxn.ts_species.ts_checks['IRC']

    def test_check_irc_bond_list_fallback_is_independent_of_atom_order(self):
        """
        Test that the bond-list fallback accepts IRC endpoints listed in a different atom order than the reactants.

        The IRC endpoints keep the TS atom order, which no TS method guarantees to be the reactant order.
        """
        xyz_1 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out'))
        xyz_2 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_2.out'))
        rxn = ARCReaction(r_species=[ARCSpecies(label='R', smiles='O=[C]COO', xyz=xyz_1)],
                          p_species=[ARCSpecies(label='P', smiles='O=CCO[O]', xyz=xyz_2)])
        order = list(reversed(range(len(xyz_1['symbols']))))
        permuted_1 = xyz_from_data(coords=[xyz_1['coords'][i] for i in order],
                                   symbols=[xyz_1['symbols'][i] for i in order])
        permuted_2 = xyz_from_data(coords=[xyz_2['coords'][i] for i in order],
                                   symbols=[xyz_2['symbols'][i] for i in order])
        for endpoints in ((xyz_1, xyz_2), (permuted_1, permuted_2), (permuted_2, permuted_1)):
            self.assertIs(self._check_irc_fallback(rxn, *endpoints), True)

    def test_check_irc_bond_list_fallback_still_rejects_wrong_endpoints(self):
        """Guard: the bond-list fallback rejects endpoints that are not the reactant and product, in any order."""
        xyz_1 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out'))
        xyz_2 = parse_geometry(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_2.out'))
        rxn = ARCReaction(r_species=[ARCSpecies(label='R', smiles='O=[C]COO', xyz=xyz_1)],
                          p_species=[ARCSpecies(label='P', smiles='O=CCO[O]', xyz=xyz_2)])
        order = list(reversed(range(len(xyz_2['symbols']))))
        permuted_2 = xyz_from_data(coords=[xyz_2['coords'][i] for i in order],
                                   symbols=[xyz_2['symbols'][i] for i in order])
        self.assertIs(self._check_irc_fallback(rxn, permuted_2, permuted_2), False)

    def test_check_irc_bond_list_fallback_rejects_an_irc_that_stays_in_one_well(self):
        """
        Test that the bond-list fallback rejects endpoints that are both the same well of a degenerate reaction.

        The reactant and product graphs of the ethyl 1,2-H shift are isomorphic, so each endpoint matches each side
        by itself. Only the comparison of the condensed graph of reaction distinguishes a real shift.
        """
        rxn = ARCReaction(r_species=[ARCSpecies(label='ethyl', smiles='C[CH2]')],
                          p_species=[ARCSpecies(label='ethyl_p', smiles='[CH2]C')])
        xyz_r, xyz_p = self._make_irc_endpoints(rxn)
        self.assertIs(self._check_irc_fallback(rxn, xyz_r, xyz_r), False)
        self.assertIs(self._check_irc_fallback(rxn, xyz_p, xyz_p), False)
        self.assertIs(self._check_irc_fallback(rxn, xyz_r, xyz_p), True)
        self.assertIs(self._check_irc_fallback(rxn, xyz_p, xyz_r), True)
        order = list(reversed(range(len(xyz_r['symbols']))))
        permuted_r, permuted_p = self._make_irc_endpoints(rxn, permutation=order)
        self.assertIs(self._check_irc_fallback(rxn, permuted_r, permuted_p), True)
        self.assertIs(self._check_irc_fallback(rxn, permuted_r, permuted_r), False)

    def test_check_irc_bond_list_fallback_with_different_element_order_in_the_products(self):
        """
        Test the bond-list fallback of H + CH4 <=> H2 + CH3, whose products list the elements in another order.

        The reaction graph must be labelled with the reactant elements, because the reaction bonds are in the
        reactant atom indices.
        """
        rxn = ARCReaction(r_species=[ARCSpecies(label='H', smiles='[H]'), ARCSpecies(label='CH4', smiles='C')],
                          p_species=[ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='H2', smiles='[H][H]')])
        self.assertNotEqual([a.element.symbol for spc in rxn.r_species for a in spc.mol.atoms],
                            [a.element.symbol for spc in rxn.p_species for a in spc.mol.atoms])
        xyz_r, xyz_p = self._make_irc_endpoints(rxn)
        self.assertIs(self._check_irc_fallback(rxn, xyz_r, xyz_p), True)
        self.assertIs(self._check_irc_fallback(rxn, xyz_p, xyz_r), True)
        order = [3, 0, 5, 1, 4, 2]
        permuted_r, permuted_p = self._make_irc_endpoints(rxn, permutation=order)
        self.assertIs(self._check_irc_fallback(rxn, permuted_r, permuted_p), True)
        self.assertIs(self._check_irc_fallback(rxn, permuted_r, permuted_r), False)

    def test_find_cgr_isomorphism(self):
        """Test the _get_condensed_graph_of_reaction() and _find_cgr_isomorphism() functions."""
        graph_1 = ts._get_condensed_graph_of_reaction(['C', 'O', 'H'], [(0, 1)], [(1, 2)])
        graph_2 = ts._get_condensed_graph_of_reaction(['H', 'O', 'C'], [(1, 2)], [(0, 1)])
        graph_3 = ts._get_condensed_graph_of_reaction(['H', 'O', 'C'], [(0, 1)], [(1, 2)])
        self.assertEqual(ts._find_cgr_isomorphism(graph_1, graph_1), [0, 1, 2])
        self.assertEqual(ts._find_cgr_isomorphism(graph_1, graph_2), [2, 1, 0])
        self.assertIsNone(ts._find_cgr_isomorphism(graph_1, graph_3))

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        for project_directory in [get_test_project_directory('arc_project_for_testing_delete_after_usage4'),
                                  cls.project_directory_5]:
            shutil.rmtree(project_directory, ignore_errors=True)
        file_paths = [os.path.join(ARC_PATH, 'arc', 'checks', 'nul'), os.path.join(ARC_PATH, 'arc', 'checks', 'run.out')]
        for file_path in file_paths:
            if os.path.isfile(file_path):
                os.remove(file_path)


class TestTsAtomMap(unittest.TestCase):
    """
    Test the TS atom map the IRC check records: the TS atom of every reactant atom and every product atom, from an
    isomorphism of the condensed graphs of reaction of the reaction and of the TS, checked on the TS geometry.
    Each case is built in the atom indices of the concatenated reactants, then permuted into the TS atom order.
    """

    CH3 = ((0.0, 0.0, 0.0), (1.08, 0.0, 0.0), (-0.54, 0.935, 0.0), (-0.54, -0.935, 0.0))
    CH4_OH_TS = ((0.0, 0.0, 0.0), (-0.36, 1.03, 0.0), (-0.36, -0.51, 0.89), (-0.36, -0.51, -0.89),
                 (1.3, 0.0, 0.0), (2.6, 0.0, 0.0), (3.4, 0.55, 0.0))

    def setUp(self):
        self.scratch = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.scratch, True)

    @staticmethod
    def _gaussian_irc_log(path, xyz, last=None):
        """Write a minimal Gaussian log whose first Input orientation table holds ``xyz`` and, when given, whose
        last one holds ``last``."""
        rule = ' ' + '-' * 69 + '\n'

        def table(geometry):
            rows = ''.join(f'{i + 1:7d}{NUMBER_BY_SYMBOL[symbol]:11d}{0:12d}{x:16.6f}{y:12.6f}{z:12.6f}\n'
                           for i, (symbol, (x, y, z)) in enumerate(zip(geometry['symbols'], geometry['coords'])))
            return ('                          Input orientation:                          \n' + rule
                    + ' Center     Atomic      Atomic             Coordinates (Angstroms)\n'
                    + ' Number     Number       Type             X           Y           Z\n' + rule + rows + rule)

        with open(path, 'w') as f:
            f.write(' Entering Gaussian System, Link 0=g16\n Current Structure is TS -> form Hessian eigenvectors.\n')
            f.write(table(xyz) + ' Point Number:   0          Path Number:   1\n')
            if last is not None:
                f.write(table(last))

    def _irc_logs(self, rxn, xyz=None, frames=(None, None), last=None):
        """Write the two IRC logs of a reaction's TS (or of ``xyz``), the second optionally reoriented, each ending at
        the geometry of ``last`` of the same index (the start geometry when not given)."""
        xyz = xyz or rxn.ts_species.get_xyz(generate=False)
        paths = list()
        for i, frame in enumerate(frames):
            coords = np.array(xyz['coords'], dtype=float)
            if frame is not None:
                coords = coords @ frame[0].T + frame[1]
            path = os.path.join(self.scratch, f'irc_{i}.out')
            start = xyz_from_data(coords=coords, symbols=xyz['symbols'])
            self._gaussian_irc_log(path, start, last=last[i] if last is not None else start)
            paths.append(path)
        return paths

    def _endpoint_logs(self, endpoint_1, endpoint_2, first=None):
        """Write the optimization logs of the two endpoint species, each starting at ``first`` of the same index (the
        endpoint geometry when not given) and ending at the endpoint geometry."""
        paths = list()
        for i, endpoint in enumerate((endpoint_1, endpoint_2)):
            path = os.path.join(self.scratch, f'endpoint_{i}.out')
            self._gaussian_irc_log(path, first[i] if first is not None else endpoint, last=endpoint)
            paths.append(path)
        return paths

    @staticmethod
    def _verdict_inputs(endpoint_1, endpoint_2):
        """The endpoint geometries with the fragments the IRC verdict perceives on them."""
        return {'endpoint_xyzs': {1: endpoint_1, 2: endpoint_2},
                'endpoint_fragments': {1: ts._perceive_irc_fragments(endpoint_1),
                                       2: ts._perceive_irc_fragments(endpoint_2)}}

    def _check(self, endpoint_1, endpoint_2, rxn, irc_log_paths='written', endpoint_log_paths='written'):
        """Run the IRC check with IRC logs started from the TS geometry and ending at the endpoints, and endpoint
        optimization logs that follow them, unless other logs are given."""
        has_ts = rxn.ts_species is not None and rxn.ts_species.get_xyz(generate=False) is not None
        if irc_log_paths == 'written':
            irc_log_paths = self._irc_logs(rxn, last=(endpoint_1, endpoint_2)) if has_ts else None
        if endpoint_log_paths == 'written':
            endpoint_log_paths = self._endpoint_logs(endpoint_1, endpoint_2)
        ts.check_irc_species_and_rxn(xyz_1=endpoint_1, xyz_2=endpoint_2, rxn=rxn, irc_log_paths=irc_log_paths,
                                     endpoint_log_paths=endpoint_log_paths)

    @staticmethod
    def _shift(coords, dx):
        return tuple((x + dx, y, z) for x, y, z in coords)

    def _build(self, symbols, r_coords, p_coords, ts_coords, reactants, products, ts_order, label=None, crossed=None):
        """
        Build a reaction with a declared atom map, the TS species, and the two IRC endpoints in the TS atom order.

        Args:
            symbols: The element of every atom, in the order of the concatenated reactants.
            r_coords: The reactant endpoint coordinates in that order.
            p_coords: The product endpoint coordinates in that order.
            ts_coords: The TS coordinates in that order.
            reactants: ``(label, smiles, multiplicity, atoms, count)`` per reactant species, ``atoms`` being the
                       indices of the first occurrence in the species' own atom order.
            products: ``(label, smiles, multiplicity, atoms)`` per product species.
            ts_order: TS atom ``k`` is the reactant atom ``ts_order[k]``.
            label: The reaction label (needed for a repeated species).
            crossed: A pair of reactant atoms whose images under the atom map are swapped.
        """
        def _species(label_, smiles, multiplicity, atoms, coords):
            xyz = xyz_from_data(coords=tuple(coords[i] for i in atoms), symbols=tuple(symbols[i] for i in atoms))
            return ARCSpecies(label=label_, smiles=smiles, multiplicity=multiplicity, xyz=xyz)

        r_species = [_species(lbl, smiles, mult, atoms, r_coords) for lbl, smiles, mult, atoms, _ in reactants]
        p_species = [_species(lbl, smiles, mult, atoms, p_coords) for lbl, smiles, mult, atoms in products]
        product_order = [i for _, _, _, atoms in products for i in atoms]
        atom_map = [product_order.index(i) for i in range(len(symbols))]
        if crossed is not None:
            a, b = crossed
            atom_map[a], atom_map[b] = atom_map[b], atom_map[a]
        rxn = ARCReaction(label=label, r_species=r_species, p_species=p_species)
        rxn._set_atom_map(atom_map, 'declared')
        ts_symbols = tuple(symbols[i] for i in ts_order)
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=xyz_from_data(
            coords=tuple(ts_coords[i] for i in ts_order), symbols=ts_symbols))
        endpoint_r = xyz_from_data(coords=tuple(r_coords[i] for i in ts_order), symbols=ts_symbols)
        endpoint_p = xyz_from_data(coords=tuple(p_coords[i] for i in ts_order), symbols=ts_symbols)
        return rxn, endpoint_r, endpoint_p, atom_map

    def _ch4_oh(self, ts_order, ts_coords=None, crossed=None):
        symbols = ('C', 'H', 'H', 'H', 'H', 'O', 'H')
        r_coords = ((0.0, 0.0, 0.0), (0.629, 0.629, 0.629), (-0.629, -0.629, 0.629), (-0.629, 0.629, -0.629),
                    (0.629, -0.629, -0.629), (6.0, 0.0, 0.0), (6.97, 0.0, 0.0))
        p_coords = ((0.0, 0.0, 0.0), (1.08, 0.0, 0.0), (-0.54, 0.935, 0.0), (-0.54, -0.935, 0.0),
                    (5.7597, 0.9294, 0.0), (6.0, 0.0, 0.0), (6.96, 0.0, 0.0))
        return self._build(symbols, r_coords, p_coords, ts_coords or self.CH4_OH_TS,
                           reactants=[('CH4', 'C', 1, [0, 1, 2, 3, 4], 1), ('OH', '[OH]', 2, [5, 6], 1)],
                           products=[('CH3', '[CH3]', 2, [0, 1, 2, 3]), ('H2O', 'O', 1, [5, 4, 6])],
                           ts_order=ts_order, crossed=crossed)

    def _assert_consistent(self, rxn, ts_atom_map, atom_map):
        """The recorded map is a bijection onto the TS atoms that conserves the element and the atom map."""
        n_atoms = len(atom_map)
        ts_symbols = rxn.ts_species.get_xyz(generate=False)['symbols']
        reactants, products = rxn.get_reactants_and_products(return_copies=False)
        r_symbols = [s for spc in reactants for s in spc.get_xyz(generate=False)['symbols']]
        self.assertEqual(sorted(ts_atom_map['reactants']), list(range(n_atoms)))
        self.assertEqual(sorted(ts_atom_map['products']), list(range(n_atoms)))
        for i in range(n_atoms):
            self.assertEqual(r_symbols[i], ts_symbols[ts_atom_map['reactants'][i]])
            self.assertEqual(ts_atom_map['products'][atom_map[i]], ts_atom_map['reactants'][i])
        self.assertEqual(ts_atom_map['ts_atom_order_follows_reactants'],
                         ts_atom_map['reactants'] == list(range(n_atoms)))
        self.assertEqual(ts_atom_map['method'], 'irc_endpoint_cgr_isomorphism')
        self.assertEqual(ts_atom_map['ts_label'], 'TS')
        self.assertIn(ts_atom_map['reactant_endpoint'], (1, 2))

    def test_an_h_abstraction_with_a_permuted_ts_order_pins_the_transferring_hydrogen(self):
        """Test CH4 + OH with the TS atoms listed O, H, C, H(transferring), H, H, H"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=[5, 6, 0, 4, 1, 2, 3])
        self.assertEqual(atom_map, [0, 1, 2, 3, 5, 4, 6])
        self._check(endpoint_r, endpoint_p, rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        ts_atom_map = rxn.ts_species.ts_atom_map
        self.assertIsNone(rxn.ts_species.ts_atom_map_unavailable_reason)
        self.assertEqual(rxn.atom_map, atom_map)
        self._assert_consistent(rxn, ts_atom_map, atom_map)
        self._assert_both_legs_are_graph_isomorphisms(rxn, ts_atom_map, endpoint_r, endpoint_p)
        reactants = ts_atom_map['reactants']
        self.assertEqual((reactants[0], reactants[4], reactants[5], reactants[6]), (2, 3, 0, 1))
        self.assertEqual(sorted(reactants[1:4]), [4, 5, 6])
        self.assertIs(ts_atom_map['ts_atom_order_follows_reactants'], False)
        self.assertEqual(ts_atom_map['products'][5], 3)
        self.assertEqual(ts_atom_map['reactant_endpoint'], 1)

    def test_the_endpoints_in_either_order_give_a_map_with_the_same_pinned_atoms(self):
        """Test that swapping which endpoint is given first does not change the pinned atoms"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=[5, 6, 0, 4, 1, 2, 3])
        self._check(endpoint_p, endpoint_r, rxn)
        ts_atom_map = rxn.ts_species.ts_atom_map
        self._assert_consistent(rxn, ts_atom_map, atom_map)
        self.assertEqual([ts_atom_map['reactants'][i] for i in (0, 4, 5, 6)], [2, 3, 0, 1])
        self.assertEqual(ts_atom_map['reactant_endpoint'], 2)

    def test_a_ts_in_reactant_order_states_it(self):
        """Test that the identity is chosen, and the flag is true, when the TS follows the reactant order"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        self._check(endpoint_r, endpoint_p, rxn)
        ts_atom_map = rxn.ts_species.ts_atom_map
        self._assert_consistent(rxn, ts_atom_map, atom_map)
        self.assertEqual(ts_atom_map['reactants'], list(range(7)))
        self.assertEqual(ts_atom_map['products'], [0, 1, 2, 3, 5, 4, 6])
        self.assertIs(ts_atom_map['ts_atom_order_follows_reactants'], True)

    def test_a_swap_of_two_hydrogens_between_the_ts_and_the_irc_start_gives_no_map(self):
        """Test that IRC logs started from a TS whose transferring H sits elsewhere than in the recorded TS give none"""
        ts_coords = list(self.CH4_OH_TS)
        ts_coords[1], ts_coords[4] = ts_coords[4], ts_coords[1]
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)), ts_coords=tuple(ts_coords))
        symbols = rxn.ts_species.get_xyz(generate=False)['symbols']
        logs = self._irc_logs(rxn, xyz=xyz_from_data(coords=self.CH4_OH_TS, symbols=symbols))
        with self.assertLogs('arc', level='WARNING') as captured:
            self._check(endpoint_r, endpoint_p, rxn, irc_log_paths=logs)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertIsNone(rxn.ts_species.ts_atom_map)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'irc_start_geometry_differs')
        self.assertTrue(any('does not start from the TS geometry' in message for message in captured.output))

    def test_a_swap_of_two_heavy_atoms_between_the_ts_and_the_irc_start_gives_no_map(self):
        """Test that exchanging the positions of the carbon and the oxygen is refused the same way"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        ts_xyz = rxn.ts_species.get_xyz(generate=False)
        coords = [list(c) for c in ts_xyz['coords']]
        coords[0], coords[5] = coords[5], coords[0]
        logs = self._irc_logs(rxn, xyz=xyz_from_data(coords=coords, symbols=ts_xyz['symbols']))
        with self.assertLogs('arc', level='WARNING'):
            self._check(endpoint_r, endpoint_p, rxn, irc_log_paths=logs)
        self.assertIsNone(rxn.ts_species.ts_atom_map)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'irc_start_geometry_differs')

    def test_one_irc_log_that_starts_elsewhere_is_enough_to_refuse_the_map(self):
        """Test that the second log differing from the TS refuses the map although the first one matches"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        ts_xyz = rxn.ts_species.get_xyz(generate=False)
        coords = [list(c) for c in ts_xyz['coords']]
        coords[1][0] += 0.2
        good = self._irc_logs(rxn)[0]
        bad = os.path.join(self.scratch, 'bad.out')
        self._gaussian_irc_log(bad, xyz_from_data(coords=coords, symbols=ts_xyz['symbols']))
        with self.assertLogs('arc', level='WARNING'):
            self._check(endpoint_r, endpoint_p, rxn, irc_log_paths=[good, bad])
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'irc_start_geometry_differs')

    def test_a_reoriented_irc_start_frame_still_gives_the_map(self):
        """Test that a log that prints the TS rotated and translated is the TS geometry, atom for atom"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=[5, 6, 0, 4, 1, 2, 3])
        theta = 0.7
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0],
                             [0.0, 0.0, 1.0]])
        logs = self._irc_logs(rxn, frames=(None, (rotation, np.array([1.5, -2.0, 0.3]))), last=(endpoint_r, endpoint_p))
        self._check(endpoint_r, endpoint_p, rxn, irc_log_paths=logs)
        self.assertIsNone(rxn.ts_species.ts_atom_map_unavailable_reason)
        self._assert_consistent(rxn, rxn.ts_species.ts_atom_map, atom_map)

    def test_a_mirror_image_start_is_not_the_ts_geometry(self):
        """Test that a reflected start geometry, which no rotation superimposes, is refused"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        mirror = np.diag([1.0, 1.0, -1.0])
        logs = self._irc_logs(rxn, frames=((mirror, np.zeros(3)), (mirror, np.zeros(3))))
        with self.assertLogs('arc', level='WARNING'):
            self._check(endpoint_r, endpoint_p, rxn, irc_log_paths=logs)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'irc_start_geometry_differs')

    def test_an_unreadable_irc_log_gives_no_map(self):
        """Test that no logs, a missing log, a log without a geometry and an unknown program are all unavailable"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        empty = os.path.join(self.scratch, 'empty.out')
        open(empty, 'w').close()
        no_table = os.path.join(self.scratch, 'no_table.out')
        with open(no_table, 'w') as f:
            f.write(' Entering Gaussian System, Link 0=g16\n Point Number:   0          Path Number:   1\n')
        good = self._irc_logs(rxn)[0]
        for logs in (None, [], [good, None], [good, os.path.join(self.scratch, 'missing.out')], [good, empty],
                     [good, no_table]):
            with self.subTest(logs=logs), self.assertLogs('arc', level='WARNING'):
                self._check(endpoint_r, endpoint_p, rxn, irc_log_paths=logs)
                self.assertIsNone(rxn.ts_species.ts_atom_map)
                self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'irc_start_geometry_unavailable')

    def test_real_irc_logs_of_one_ts_start_from_the_same_geometry(self):
        """Test the start of two real Gaussian IRC logs: each is the first input orientation, and they agree"""
        logs = [os.path.join(ARC_TESTING_PATH, 'irc', f'rxn_1_irc_{i}.out') for i in (1, 2)]
        starts = [parse_irc_start_geometry(log_file_path=log) for log in logs]
        self.assertEqual(starts[0]['symbols'], ('O', 'C', 'C', 'O', 'O', 'H', 'H', 'H'))
        self.assertEqual(starts[0]['coords'][0], (-0.975117, 1.693544, 1.019828))
        self.assertIsNone(ts.get_irc_start_geometry_reason(starts[0], logs))
        moved = dict(starts[0], coords=tuple((x + 0.01, y, z) for x, y, z in starts[0]['coords']))
        self.assertIsNone(ts.get_irc_start_geometry_reason(moved, logs))
        distorted = dict(starts[0], coords=(starts[0]['coords'][0][:1] + (5.0,) + starts[0]['coords'][0][2:],)
                         + tuple(starts[0]['coords'][1:]))
        self.assertEqual(ts.get_irc_start_geometry_reason(distorted, logs), 'irc_start_geometry_differs')

    def test_the_start_of_a_real_irc_log_is_its_first_input_orientation_and_one_step_from_point_one(self):
        """Test that the start is the first Input orientation block, 0.036 A RSSD from point 1, and not traj[0]"""
        log = os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out')
        with open(log) as f:
            lines = f.readlines()
        first = next(i for i, line in enumerate(lines) if 'Input orientation:' in line)
        rows = [line.split() for line in lines[first + 5:first + 13]]
        expected = tuple((float(row[3]), float(row[4]), float(row[5])) for row in rows)
        start = parse_irc_start_geometry(log_file_path=log)
        self.assertEqual(tuple(start['coords']), expected)
        point_1 = parse_irc_path(log_file_path=log)[0]['xyz']
        self.assertAlmostEqual(kabsch(start, point_1), 0.036, delta=0.002)
        self.assertGreater(kabsch(start, point_1), 10 * IRC_START_GEOMETRY_TOLERANCE)
        self.assertNotEqual(tuple(start['coords']), tuple(parse_irc_traj(log_file_path=log)[0]['coords']))

    def test_the_rssd_separates_a_rigid_copy_rounding_and_a_real_difference(self):
        """Test the congruence criterion on a rigid copy, 1e-6 rounding, one IRC step and a same-element swap"""
        log = os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out')
        start = parse_irc_start_geometry(log_file_path=log)
        theta = 0.4
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0],
                             [0.0, 0.0, 1.0]])
        coords = np.array(start['coords'], dtype=float)
        rigid = dict(start, coords=tuple(map(tuple, coords @ rotation.T + np.array([1.0, 2.0, 3.0]))))
        rounded = dict(start, coords=tuple(map(tuple, np.round(coords, 6) + 1e-6)))
        swapped = [list(c) for c in start['coords']]
        swapped[5], swapped[6] = swapped[6], swapped[5]
        step = parse_irc_path(log_file_path=log)[0]['xyz']
        self.assertTrue(ts.are_geometries_congruent(start, rigid))
        self.assertTrue(ts.are_geometries_congruent(start, rounded))
        self.assertFalse(ts.are_geometries_congruent(start, step))
        self.assertFalse(ts.are_geometries_congruent(start, dict(start, coords=tuple(map(tuple, swapped)))))

    @staticmethod
    def _bond_set(xyz):
        """The distance-perceived bonds of a geometry, independently of the IRC verdict's perception."""
        bonds = get_bonds_from_dmat(dmat=xyz_to_dmat(xyz), elements=xyz['symbols'], n_fragments=0)
        return {tuple(sorted(bond)) for bond in bonds}

    def _assert_both_legs_are_graph_isomorphisms(self, rxn, ts_atom_map, endpoint_r, endpoint_p):
        """The map sends the reactant bonds onto the reactant-endpoint bonds and the product bonds onto the
        product-endpoint bonds, bijectively, and conserves atoms through the atom map."""
        r_bonds, p_bonds = rxn.get_bonds()
        endpoints = {1: endpoint_r, 2: endpoint_p}
        r_side = self._bond_set(endpoints[ts_atom_map['reactant_endpoint']])
        p_side = self._bond_set(endpoints[3 - ts_atom_map['reactant_endpoint']])
        m = ts_atom_map['reactants']
        self.assertEqual(sorted(m), list(range(len(m))))
        self.assertEqual({tuple(sorted((m[a], m[b]))) for a, b in r_bonds}, r_side)
        self.assertEqual({tuple(sorted((m[a], m[b]))) for a, b in p_bonds}, p_side)
        for i, j in enumerate(rxn.atom_map):
            self.assertEqual(ts_atom_map['products'][j], m[i])

    def test_a_stale_ts_geometry_refuses_the_map(self):
        """Test that a TS whose recorded geometry changed after the IRC was started no longer matches the IRC logs"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        logs = self._irc_logs(rxn)
        ts_xyz = rxn.ts_species.get_xyz(generate=False)
        coords = [list(c) for c in ts_xyz['coords']]
        coords[1][1] += 0.3
        rxn.ts_species.final_xyz = xyz_from_data(coords=tuple(map(tuple, coords)), symbols=ts_xyz['symbols'])
        with self.assertLogs('arc', level='WARNING'):
            self._check(endpoint_r, endpoint_p, rxn, irc_log_paths=logs)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertIsNone(rxn.ts_species.ts_atom_map)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'irc_start_geometry_differs')
        self.assertIs(rxn.ts_species.irc_participant_mapping['atom_order_matches_ts'], False)

    def test_the_participant_mapping_states_whether_the_irc_started_from_the_ts(self):
        """Test atom_order_matches_ts: true for congruent starts, false for a differing one, null when unreadable"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        self._check(endpoint_r, endpoint_p, rxn)
        self.assertIs(rxn.ts_species.irc_participant_mapping['atom_order_matches_ts'], True)
        with self.assertLogs('arc', level='WARNING'):
            self._check(endpoint_r, endpoint_p, rxn, irc_log_paths=None)
        self.assertIsNone(rxn.ts_species.irc_participant_mapping['atom_order_matches_ts'])

    def test_a_c3_relabelling_of_equivalent_hydrogens_leaves_the_seed_congruent_and_the_map_valid(self):
        """Test a C3-symmetric TS whose IRC input lists its three spectator hydrogens cyclically permuted"""
        angles = (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
        spectators = tuple((-0.36, 1.03 * np.cos(a), 1.03 * np.sin(a)) for a in angles)
        ts_coords = ((0.0, 0.0, 0.0),) + spectators + ((1.3, 0.0, 0.0), (2.6, 0.0, 0.0), (3.4, 0.0, 0.0))
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)), ts_coords=ts_coords)
        ts_xyz = rxn.ts_species.get_xyz(generate=False)
        relabelled = [ts_xyz['coords'][i] for i in (0, 2, 3, 1, 4, 5, 6)]
        log_xyz = xyz_from_data(coords=tuple(map(tuple, relabelled)), symbols=ts_xyz['symbols'])
        self.assertIsNone(ts.get_irc_start_geometry_reason(ts_xyz, self._irc_logs(rxn, xyz=log_xyz)))
        self._check(endpoint_r, endpoint_p, rxn,
                    irc_log_paths=self._irc_logs(rxn, xyz=log_xyz, last=(endpoint_r, endpoint_p)))
        self.assertIsNone(rxn.ts_species.ts_atom_map_unavailable_reason)
        self._assert_consistent(rxn, rxn.ts_species.ts_atom_map, atom_map)
        self._assert_both_legs_are_graph_isomorphisms(rxn, rxn.ts_species.ts_atom_map, endpoint_r, endpoint_p)

    def test_an_atom_map_wrong_only_by_an_automorphism_still_gives_a_valid_map(self):
        """Test that exchanging the images of two equivalent spectator hydrogens leaves a valid exported map"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)), crossed=(1, 2))
        self._check(endpoint_r, endpoint_p, rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertIsNone(rxn.ts_species.ts_atom_map_unavailable_reason)
        self._assert_consistent(rxn, rxn.ts_species.ts_atom_map, atom_map)
        self._assert_both_legs_are_graph_isomorphisms(rxn, rxn.ts_species.ts_atom_map, endpoint_r, endpoint_p)

    def test_an_atom_map_wrong_for_a_non_equivalent_hydrogen_is_contradicted_and_left_alone(self):
        """Test that exchanging the images of a CH4 hydrogen and of the hydroxyl hydrogen gives no map"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)), crossed=(1, 6))
        declared = list(rxn.atom_map)
        with self.assertLogs('arc', level='WARNING'):
            self._check(endpoint_r, endpoint_p, rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertIsNone(rxn.ts_species.ts_atom_map)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'atom_map_contradicts_ts')
        self.assertEqual(rxn.atom_map, declared)

    def test_a_perceived_molecule_that_cannot_be_tied_to_the_geometry_gives_a_perception_reason(self):
        """Test that a fragment whose atom coordinates are not those of the endpoint geometry refuses the map"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        original = ts._perceive_irc_fragments

        def perceive_and_disturb(xyz, charge=0):
            fragments = original(xyz, charge=charge)
            fragments[0].atoms[0].coords = fragments[0].atoms[0].coords + 1e-9
            return fragments

        with patch.object(ts, '_perceive_irc_fragments', side_effect=perceive_and_disturb), \
                self.assertLogs('arc', level='WARNING'):
            self._check(endpoint_r, endpoint_p, rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertIsNone(rxn.ts_species.ts_atom_map)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'endpoint_perception_mismatch')

    def test_the_endpoint_connectivity_comes_from_the_fragments_and_ignores_bond_orders(self):
        """Test that the bonds are the fragments' own, in endpoint atom indices, for a permuted TS order"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=[5, 6, 0, 4, 1, 2, 3])
        inputs = self._verdict_inputs(endpoint_r, endpoint_p)
        bonds = ts._get_endpoint_bonds_from_fragments(endpoint_r, inputs['endpoint_fragments'][1])
        self.assertEqual({tuple(sorted(bond)) for bond in bonds}, self._bond_set(endpoint_r))
        self.assertIsNone(ts._get_endpoint_bonds_from_fragments(endpoint_r, inputs['endpoint_fragments'][1][:1]))

    def test_reordered_fragment_atoms_are_tied_to_the_endpoint_by_exact_coordinates(self):
        """Test that a perception that reorders the atoms of its fragments still maps, and is valid"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=[5, 6, 0, 4, 1, 2, 3])
        original = ts._perceive_irc_fragments

        def perceive_reordered(xyz, charge=0):
            fragments = original(xyz, charge=charge)
            for fragment in fragments:
                fragment.atoms.reverse()
            return fragments

        with patch.object(ts, '_perceive_irc_fragments', side_effect=perceive_reordered):
            self._check(endpoint_r, endpoint_p, rxn)
        self.assertIsNone(rxn.ts_species.ts_atom_map_unavailable_reason)
        self._assert_consistent(rxn, rxn.ts_species.ts_atom_map, atom_map)
        self._assert_both_legs_are_graph_isomorphisms(rxn, rxn.ts_species.ts_atom_map, endpoint_r, endpoint_p)

    def test_fragment_atoms_without_coordinates_or_without_a_one_to_one_tie_refuse_the_map(self):
        """Test that a fragment atom with no coordinates, a repeated atom and an uncovered endpoint atom are refused"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        fragments = ts._perceive_irc_fragments(endpoint_r)
        self.assertIsNotNone(ts._get_endpoint_bonds_from_fragments(endpoint_r, fragments))
        fragments[0].atoms[0].coords = None
        self.assertIsNone(ts._get_endpoint_bonds_from_fragments(endpoint_r, fragments))
        fragments = ts._perceive_irc_fragments(endpoint_r)
        fragments[0].atoms[1].coords = np.array(fragments[0].atoms[0].coords)
        self.assertIsNone(ts._get_endpoint_bonds_from_fragments(endpoint_r, fragments))
        self.assertIsNone(ts._get_endpoint_bonds_from_fragments(endpoint_r, ts._perceive_irc_fragments(endpoint_r)[:0]))

    def test_an_endpoint_whose_elements_differ_from_the_ts_is_refused_by_the_element_guard_alone(self):
        """Test the guard in isolation: consistent fragments and no IRC geometry reason, yet permuted endpoint elements"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        reactants, products = rxn.get_reactants_and_products(return_copies=False)
        inputs = self._verdict_inputs(endpoint_r, endpoint_p)
        bonds_r = ts._get_endpoint_bonds_from_fragments(endpoint_r, inputs['endpoint_fragments'][1])
        bonds_p = ts._get_endpoint_bonds_from_fragments(endpoint_p, inputs['endpoint_fragments'][2])
        permuted = dict(endpoint_r, symbols=tuple(reversed(endpoint_r['symbols'])))
        inputs['endpoint_xyzs'] = {1: permuted, 2: endpoint_p}
        with patch.object(ts, '_get_endpoint_bonds_from_fragments', side_effect=[bonds_r, bonds_p]), \
                self.assertLogs('arc', level='WARNING'):
            ts_atom_map, reason = ts.get_ts_atom_map(
                rxn=rxn, reactants=reactants, reactant_endpoint=1, ts_xyz=rxn.ts_species.get_xyz(generate=False),
                irc_geometry_reason=None, **inputs)
        self.assertIsNone(ts_atom_map)
        self.assertEqual(reason, 'endpoint_perception_mismatch')

    def test_a_swap_in_an_endpoint_after_the_irc_is_refused_through_the_endpoint_chain(self):
        """Test that exchanging a spectator and a transferring hydrogen in the product endpoint gives no map"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        coords = list(endpoint_p['coords'])
        coords[1], coords[4] = coords[4], coords[1]
        swapped = xyz_from_data(coords=tuple(coords), symbols=endpoint_p['symbols'])
        irc_logs = self._irc_logs(rxn, last=(endpoint_r, endpoint_p))
        with self.assertLogs('arc', level='WARNING'):
            self._check(endpoint_r, swapped, rxn, irc_log_paths=irc_logs,
                        endpoint_log_paths=self._endpoint_logs(endpoint_r, endpoint_p))
        self.assertIsNone(rxn.ts_species.ts_atom_map)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'irc_endpoint_geometry_differs')
        self.assertIs(rxn.ts_species.irc_participant_mapping['atom_order_matches_ts'], False)
        with self.assertLogs('arc', level='WARNING'):
            self._check(endpoint_r, endpoint_p, rxn, irc_log_paths=irc_logs,
                        endpoint_log_paths=self._endpoint_logs(endpoint_r, endpoint_p,
                                                               first=(endpoint_r, swapped)))
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'irc_endpoint_geometry_differs')

    def test_endpoints_in_another_element_order_than_the_ts_state_it_and_give_no_map(self):
        """Test that endpoints O, H, C, H, H, H, H against a TS C, H, H, H, H, O, H are flagged false and refused"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=[5, 6, 0, 1, 2, 3, 4])
        natural = self._ch4_oh(ts_order=list(range(7)))[0].ts_species.get_xyz(generate=False)
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=natural)
        self.assertEqual(endpoint_r['symbols'], ('O', 'H', 'C', 'H', 'H', 'H', 'H'))
        self.assertEqual(natural['symbols'], ('C', 'H', 'H', 'H', 'H', 'O', 'H'))
        with self.assertLogs('arc', level='WARNING'):
            self._check(endpoint_r, endpoint_p, rxn)
        self.assertIs(rxn.ts_species.irc_participant_mapping['atom_order_matches_ts'], False)
        self.assertIsNone(rxn.ts_species.ts_atom_map)

    def test_a_deuterated_ts_is_congruent_with_its_parsed_start_geometry(self):
        """Test that the isotopes of the TS reach the superposition of a parsed geometry, which states none"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        ts_xyz = rxn.ts_species.get_xyz(generate=False)
        isotopes = list(ts_xyz['isotopes'])
        isotopes[1] = 2
        deuterated = xyz_from_data(coords=ts_xyz['coords'], symbols=ts_xyz['symbols'], isotopes=tuple(isotopes))
        parsed = xyz_from_data(coords=tuple(map(tuple, np.round(np.array(ts_xyz['coords'], dtype=float), 6))),
                               symbols=ts_xyz['symbols'])
        self.assertGreater(kabsch(deuterated, parsed), IRC_START_GEOMETRY_TOLERANCE)
        self.assertTrue(ts.are_geometries_congruent(deuterated, parsed))
        rxn.ts_species.final_xyz = deuterated
        self._check(endpoint_r, endpoint_p, rxn)
        self.assertIsNone(rxn.ts_species.ts_atom_map_unavailable_reason)

    def test_a_crossed_spectator_hydrogen_contradicts_the_ts_and_records_no_map(self):
        """Test C2H6 + OH with an atom map that swaps a spectator H of each carbon: no isomorphism exists"""
        symbols = ('C', 'H', 'H', 'H', 'C', 'H', 'H', 'H', 'O', 'H')
        r_coords = ((0.0, 0.0, 0.0), (-0.38, 1.03, 0.0), (-0.38, -0.51, 0.89), (-0.38, -0.51, -0.89),
                    (1.53, 0.0, 0.0), (1.91, 1.03, 0.0), (1.91, -0.51, 0.89), (1.91, -0.51, -0.89),
                    (6.0, 0.0, 0.0), (6.97, 0.0, 0.0))
        p_coords = list(r_coords)
        p_coords[1] = (5.4, 0.8, 0.0)
        ts_coords = ((0.0, 0.0, 0.0), (-0.43, 1.23, 0.0), (-0.38, -0.51, 0.89), (-0.38, -0.51, -0.89),
                     (1.53, 0.0, 0.0), (1.91, 1.03, 0.0), (1.91, -0.51, 0.89), (1.91, -0.51, -0.89),
                     (-0.86, 2.46, 0.0), (-1.7, 2.9, 0.0))
        reactants = [('C2H6', 'CC', 1, [0, 1, 2, 3, 4, 5, 6, 7], 1), ('OH', '[OH]', 2, [8, 9], 1)]
        products = [('C2H5', 'C[CH2]', 2, [0, 4, 2, 3, 5, 6, 7]), ('H2O', 'O', 1, [8, 1, 9])]
        rxn, endpoint_r, endpoint_p, atom_map = self._build(symbols, r_coords, tuple(p_coords), ts_coords,
                                                            reactants, products, ts_order=list(range(10)))
        self._check(endpoint_r, endpoint_p, rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self._assert_consistent(rxn, rxn.ts_species.ts_atom_map, atom_map)

        rxn, endpoint_r, endpoint_p, atom_map = self._build(symbols, r_coords, tuple(p_coords), ts_coords,
                                                            reactants, products, ts_order=list(range(10)),
                                                            crossed=(2, 5))
        with self.assertLogs('arc', level='WARNING') as logs:
            self._check(endpoint_r, endpoint_p, rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertIsNone(rxn.ts_species.ts_atom_map)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'atom_map_contradicts_ts')
        self.assertTrue(any('contradicts' in message for message in logs.output))
        self.assertEqual(rxn.atom_map, atom_map)

    def test_two_methyls_forming_ethane_state_one_block_per_occurrence(self):
        """Test CH3 + CH3 <=> C2H6, once with the TS in reactant order and once with the first two atoms exchanged"""
        symbols = ('C', 'H', 'H', 'H', 'C', 'H', 'H', 'H')
        r_coords = self.CH3 + self._shift(self.CH3, 8.0)
        p_coords = ((0.0, 0.0, 0.0), (-0.36, 1.03, 0.0), (-0.36, -0.51, 0.89), (-0.36, -0.51, -0.89),
                    (1.54, 0.0, 0.0), (1.9, 1.03, 0.0), (1.9, -0.51, 0.89), (1.9, -0.51, -0.89))
        ts_coords = ((0.0, 0.0, 0.0), (-0.36, 1.03, 0.0), (-0.36, -0.51, 0.89), (-0.36, -0.51, -0.89),
                     (2.2, 0.0, 0.0), (2.56, 1.03, 0.0), (2.56, -0.51, 0.89), (2.56, -0.51, -0.89))
        for ts_order, follows in ((list(range(8)), True), ([1, 0, 2, 3, 4, 5, 6, 7], False)):
            with self.subTest(ts_order=ts_order):
                rxn, endpoint_r, endpoint_p, atom_map = self._build(
                    symbols, r_coords, p_coords, ts_coords,
                    reactants=[('CH3', '[CH3]', 2, [0, 1, 2, 3], 2)],
                    products=[('C2H6', 'CC', 1, list(range(8)))],
                    ts_order=ts_order, label='CH3 + CH3 <=> C2H6')
                self._check(endpoint_r, endpoint_p, rxn)
                self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
                ts_atom_map = rxn.ts_species.ts_atom_map
                self._assert_consistent(rxn, ts_atom_map, atom_map)
                self.assertIs(ts_atom_map['ts_atom_order_follows_reactants'], follows)
                self.assertEqual(len(ts_atom_map['reactants']), 8)

    def test_two_hydroperoxyls_forming_peroxide_and_oxygen(self):
        """Test HO2 + HO2 <=> H2O2 + O2 with the TS atoms in an order of neither side"""
        symbols = ('O', 'O', 'H', 'O', 'O', 'H')
        r_coords = ((0.0, 0.0, 0.0), (1.33, 0.0, 0.0), (1.65, 0.92, 0.0),
                    (6.0, 0.0, 0.0), (7.33, 0.0, 0.0), (7.65, 0.92, 0.0))
        p_coords = ((0.0, 0.0, 0.0), (1.45, 0.0, 0.0), (1.8, 0.9, 0.0),
                    (6.0, 0.0, 0.0), (7.21, 0.0, 0.0), (-0.35, 0.9, 0.0))
        ts_coords = ((0.0, 0.0, 0.0), (1.33, 0.0, 0.0), (1.65, 0.92, 0.0),
                     (-1.8, 3.1, 0.0), (-0.6, 2.36, 0.0), (-0.3, 1.2, 0.0))
        rxn, endpoint_r, endpoint_p, atom_map = self._build(
            symbols, r_coords, p_coords, ts_coords,
            reactants=[('HO2', 'O[O]', 2, [0, 1, 2], 2)],
            products=[('H2O2', 'OO', 1, [0, 1, 5, 2]), ('O2', '[O][O]', 3, [3, 4])],
            ts_order=[5, 4, 3, 2, 1, 0], label='HO2 + HO2 <=> H2O2 + O2')
        self.assertEqual(atom_map, [0, 1, 3, 4, 5, 2])
        self._check(endpoint_r, endpoint_p, rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        ts_atom_map = rxn.ts_species.ts_atom_map
        self._assert_consistent(rxn, ts_atom_map, atom_map)
        self.assertEqual((ts_atom_map['reactants'][5], ts_atom_map['reactants'][0]), (0, 5))

    def test_an_isomerization_pins_the_migrating_hydrogen(self):
        """Test CH3O <=> CH2OH, whose migrating hydrogen is the only atom that changes its heavy neighbor"""
        symbols = ('C', 'O', 'H', 'H', 'H')
        r_coords = ((0.0, 0.0, 0.0), (1.37, 0.0, 0.0), (-0.36, 1.03, 0.0), (-0.36, -0.51, 0.89), (-0.36, -0.51, -0.89))
        p_coords = ((0.0, 0.0, 0.0), (1.36, 0.0, 0.0), (1.7, 0.9, 0.0), (-0.55, 0.93, 0.0), (-0.55, -0.93, 0.0))
        ts_coords = ((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.65, 1.0, 0.0), (-0.45, -0.5, 0.85), (-0.45, -0.5, -0.85))
        for ts_order in ([2, 0, 1, 3, 4], [0, 1, 2, 3, 4]):
            with self.subTest(ts_order=ts_order):
                rxn, endpoint_r, endpoint_p, atom_map = self._build(
                    symbols, r_coords, p_coords, ts_coords,
                    reactants=[('CH3O', 'C[O]', 2, [0, 1, 2, 3, 4], 1)],
                    products=[('CH2OH', '[CH2]O', 2, [0, 1, 3, 4, 2])],
                    ts_order=ts_order)
                self._check(endpoint_r, endpoint_p, rxn)
                self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
                ts_atom_map = rxn.ts_species.ts_atom_map
                self._assert_consistent(rxn, ts_atom_map, atom_map)
                self.assertEqual(ts_atom_map['reactants'][2], ts_order.index(2))
                self.assertEqual(ts_atom_map['products'][4], ts_order.index(2))

    def test_a_ts_in_product_order_is_mapped_by_its_bonds_not_by_the_reactant_order(self):
        """Test H + C2H4 <=> C2H5 with the TS atoms listed in the atom order of the product"""
        symbols = ('H', 'C', 'C', 'H', 'H', 'H', 'H')
        r_coords = ((6.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.33, 0.0, 0.0), (-0.55, 0.94, 0.0), (-0.55, -0.94, 0.0),
                    (1.88, 0.94, 0.0), (1.88, -0.94, 0.0))
        p_coords = ((1.88, -0.51, -0.88), (0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (-0.55, 0.94, 0.0), (-0.55, -0.94, 0.0),
                    (1.88, 1.02, 0.0), (1.88, -0.51, 0.88))
        ts_coords = ((1.4, 0.0, 1.5), (0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (-0.55, 0.94, 0.0), (-0.55, -0.94, 0.0),
                     (1.95, 0.94, 0.0), (1.95, -0.94, 0.0))
        rxn, endpoint_r, endpoint_p, atom_map = self._build(
            symbols, r_coords, p_coords, ts_coords,
            reactants=[('H', '[H]', 2, [0], 1), ('C2H4', 'C=C', 1, [1, 2, 3, 4, 5, 6], 1)],
            products=[('C2H5', 'C[CH2]', 2, [2, 1, 5, 6, 3, 4, 0])],
            ts_order=[1, 2, 3, 4, 5, 6, 0])
        self._check(endpoint_r, endpoint_p, rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        ts_atom_map = rxn.ts_species.ts_atom_map
        self._assert_consistent(rxn, ts_atom_map, atom_map)
        self.assertEqual(ts_atom_map['reactants'][0], 6)
        self.assertEqual([ts_atom_map['reactants'][1], ts_atom_map['reactants'][2]], [0, 1])
        self.assertIs(ts_atom_map['ts_atom_order_follows_reactants'], False)

    def test_every_reason_a_map_is_not_recorded(self):
        """Test the reason recorded for each way the check cannot state the map"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        self._check(endpoint_r, endpoint_p, rxn)
        self.assertIsNone(rxn.ts_species.ts_atom_map)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'no_ts')

        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        symbols = rxn.ts_species.get_xyz(generate=False)['symbols']
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=xyz_from_data(
            coords=endpoint_r['coords'], symbols=symbols[::-1]))
        self._check(endpoint_r, endpoint_p, rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'endpoint_perception_mismatch')

        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        rxn._atom_map = None
        self._check(endpoint_r, endpoint_p, rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'no_atom_map')
        self.assertIsNone(rxn._atom_map)

        coords_r, coords_p = ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 0.0, 0.74)), \
            ((0.0, 0.0, 0.0), (3.2, 0.0, 0.0), (0.0, 0.0, 0.97))
        rxn = ARCReaction(r_species=[ARCSpecies(label='X', smiles='[O]', multiplicity=3),
                                     ARCSpecies(label='H2', smiles='[H][H]')],
                          p_species=[ARCSpecies(label='XH', smiles='[OH]'), ARCSpecies(label='H', smiles='[H]')])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        rxn.ts_species.ts_atom_map = {'stale': True}
        ts.check_irc_species_and_rxn(xyz_1=xyz_from_data(coords=coords_r, symbols=('O', 'H', 'H')),
                                     xyz_2=xyz_from_data(coords=coords_p, symbols=('O', 'H', 'H')), rxn=rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertIsNone(rxn.ts_species.ts_atom_map)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'irc_fallback_path')

    def test_endpoint_bonds_that_are_not_those_of_the_species_give_a_perception_reason(self):
        """Test that an endpoint whose bond graph is not that of its side gives endpoint_perception_mismatch"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        reactants, products = rxn.get_reactants_and_products(return_copies=False)
        distorted = dict(endpoint_r)
        coords = [list(c) for c in endpoint_r['coords']]
        coords[6] = [30.0, 0.0, 0.0]
        distorted['coords'] = tuple(tuple(c) for c in coords)
        with self.assertLogs('arc', level='WARNING') as logs:
            ts_atom_map, reason = ts.get_ts_atom_map(
                rxn=rxn, reactants=reactants, reactant_endpoint=1, ts_xyz=rxn.ts_species.get_xyz(generate=False),
                irc_geometry_reason=None, **self._verdict_inputs(distorted, endpoint_p))
        self.assertIsNone(ts_atom_map)
        self.assertEqual(reason, 'endpoint_perception_mismatch')
        self.assertTrue(any('perceived' in message for message in logs.output))

    def test_the_opposite_endpoint_roles_are_tried_when_the_sides_are_isomorphic(self):
        """Test that, when the sides are flagged isomorphic, the endpoints given in the wrong roles still map"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        reactants, products = rxn.get_reactants_and_products(return_copies=False)
        kwargs = dict(rxn=rxn, reactants=reactants, reactant_endpoint=1, ts_xyz=rxn.ts_species.get_xyz(generate=False),
                      irc_geometry_reason=None, **self._verdict_inputs(endpoint_p, endpoint_r))
        with self.assertLogs('arc', level='WARNING'):
            self.assertEqual(ts.get_ts_atom_map(sides_distinguishable=True, **kwargs),
                             (None, 'endpoint_perception_mismatch'))
        ts_atom_map, reason = ts.get_ts_atom_map(sides_distinguishable=False, **kwargs)
        self.assertIsNone(reason)
        self.assertEqual(ts_atom_map['reactant_endpoint'], 2)
        self._assert_consistent(rxn, ts_atom_map, atom_map)

    def test_a_failure_while_mapping_keeps_the_verdict_and_the_participant_mapping(self):
        """Test that an error in the TS atom map is its own reason and leaves the verdict and the participants"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        with patch.object(ts, 'get_ts_atom_map', side_effect=RuntimeError('boom')), \
                self.assertLogs('arc', level='WARNING'):
            self._check(endpoint_r, endpoint_p, rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertIsNotNone(rxn.ts_species.irc_participant_mapping)
        self.assertIsNone(rxn.ts_species.ts_atom_map)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'computation_failed')
        self.assertEqual(rxn.atom_map, atom_map)

    def test_a_participant_mapping_failure_does_not_suppress_the_ts_atom_map(self):
        """Test that the two records are built independently"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        with patch.object(ts, '_get_irc_participant_mapping', side_effect=RuntimeError('boom')), \
                self.assertLogs('arc', level='WARNING'):
            self._check(endpoint_r, endpoint_p, rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertIsNone(rxn.ts_species.irc_participant_mapping)
        self._assert_consistent(rxn, rxn.ts_species.ts_atom_map, atom_map)

    def test_a_failure_comparing_the_endpoints_with_the_ts_fails_both_records(self):
        """Test that, with no atom order verdict, there is no participant mapping and the reason is computation_failed"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        with patch.object(ts, 'get_irc_start_geometry_reason', side_effect=RuntimeError('boom')), \
                self.assertLogs('arc', level='WARNING'):
            self._check(endpoint_r, endpoint_p, rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertIsNone(rxn.ts_species.irc_participant_mapping)
        self.assertIsNone(rxn.ts_species.ts_atom_map)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'computation_failed')

    def test_the_cgr_isomorphism_prefers_the_identity_even_when_the_matcher_finds_another_first(self):
        """Test that the identity is returned although the first isomorphism the matcher yields is not the identity"""
        symbols, bonds = ['C', 'H', 'H', 'H', 'H'], [(0, 1), (0, 2), (0, 3), (0, 4)]
        graph = ts._get_condensed_graph_of_reaction(symbols, bonds, bonds)
        reversed_graph = nx.Graph()
        for index in (0, 4, 3, 2, 1):
            reversed_graph.add_node(index, element=symbols[index])
        for bond in bonds:
            reversed_graph.add_edge(*bond, kind=(True, True))
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            graph, reversed_graph, node_match=lambda a, b: a['element'] == b['element'],
            edge_match=lambda a, b: a['kind'] == b['kind'])
        first = next(matcher.isomorphisms_iter())
        self.assertNotEqual([first[i] for i in range(5)], list(range(5)))
        self.assertEqual(ts._find_cgr_isomorphism(graph, reversed_graph), list(range(5)))

    def test_the_cgr_isomorphism_finds_one_for_isomorphic_graphs_and_none_otherwise(self):
        """Test the CGR isomorphism helper on a relabelled graph and on graphs that differ in an edge label"""
        graph = ts._get_condensed_graph_of_reaction(['C', 'H', 'H', 'H', 'H'],
                                                   [(0, 1), (0, 2), (0, 3), (0, 4)], [(0, 1), (0, 2), (0, 3), (0, 4)])
        other = ts._get_condensed_graph_of_reaction(['H', 'C', 'H', 'H', 'H'],
                                                   [(1, 0), (1, 2), (1, 3), (1, 4)], [(1, 0), (1, 2), (1, 3), (1, 4)])
        self.assertEqual(ts._find_cgr_isomorphism(graph, other)[0], 1)
        different = ts._get_condensed_graph_of_reaction(['C', 'H', 'H', 'H', 'H'],
                                                       [(0, 1), (0, 2), (0, 3), (0, 4)], [(0, 1), (0, 2), (0, 3)])
        self.assertIsNone(ts._find_cgr_isomorphism(graph, different))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
