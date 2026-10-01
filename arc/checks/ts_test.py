#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.checks.ts module
"""

import unittest
import os
import shutil
from unittest.mock import patch

import networkx as nx
import numpy as np

import arc.checks.ts as ts
from arc.common import ARC_PATH, ARC_TESTING_PATH, almost_equal_lists, get_test_project_directory
from arc.exceptions import ReactionError
from arc.job.factory import job_factory
from arc.level import Level
from arc.parser.parser import parse_normal_mode_displacement, parse_geometry
from arc.reaction import ARCReaction
from arc.species.converter import xyz_from_data
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
        rxn.ts_species.nmd_record = {'forced': True, 'frequency_cm1': -1000.0}
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

    def test_check_irc_participant_mapping_states_whether_the_endpoints_follow_the_ts_atom_order(self):
        """Test that the endpoint element sequences are compared with the TS geometry the IRC was run from."""
        xyz_r, xyz_p = self._make_ch4_oh_endpoints()
        shuffled_symbols = ('H', 'H', 'H', 'H', 'C', 'O', 'H')
        for ts_xyz, expected in ((xyz_r, True),
                                 (xyz_from_data(coords=xyz_r['coords'], symbols=xyz_r['symbols'][::-1]), False),
                                 (xyz_from_data(coords=xyz_r['coords'], symbols=shuffled_symbols), False),
                                 (None, None)):
            with self.subTest(expected=expected):
                rxn = self._make_ch4_oh_rxn()
                rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=ts_xyz)
                ts.check_irc_species_and_rxn(xyz_1=xyz_r, xyz_2=xyz_p, rxn=rxn)
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
        for target in ('_get_irc_participant_mapping', '_get_irc_endpoints_atom_order_matches_ts'):
            with self.subTest(failing=target):
                rxn = self._make_ch4_oh_rxn()
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
        rxn.declare_atom_map(atom_map)
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
        ts.check_irc_species_and_rxn(xyz_1=endpoint_r, xyz_2=endpoint_p, rxn=rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        ts_atom_map = rxn.ts_species.ts_atom_map
        self.assertIsNone(rxn.ts_species.ts_atom_map_unavailable_reason)
        self.assertEqual(rxn.atom_map, atom_map)
        self._assert_consistent(rxn, ts_atom_map, atom_map)
        reactants = ts_atom_map['reactants']
        self.assertEqual((reactants[0], reactants[4], reactants[5], reactants[6]), (2, 3, 0, 1))
        self.assertEqual(sorted(reactants[1:4]), [4, 5, 6])
        self.assertIs(ts_atom_map['ts_atom_order_follows_reactants'], False)
        self.assertEqual(ts_atom_map['products'][5], 3)
        self.assertEqual(ts_atom_map['reactant_endpoint'], 1)

    def test_the_endpoints_in_either_order_give_a_map_with_the_same_pinned_atoms(self):
        """Test that swapping which endpoint is given first does not change the pinned atoms"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=[5, 6, 0, 4, 1, 2, 3])
        ts.check_irc_species_and_rxn(xyz_1=endpoint_p, xyz_2=endpoint_r, rxn=rxn)
        ts_atom_map = rxn.ts_species.ts_atom_map
        self._assert_consistent(rxn, ts_atom_map, atom_map)
        self.assertEqual([ts_atom_map['reactants'][i] for i in (0, 4, 5, 6)], [2, 3, 0, 1])
        self.assertEqual(ts_atom_map['reactant_endpoint'], 2)

    def test_a_ts_in_reactant_order_states_it(self):
        """Test that the identity is chosen, and the flag is true, when the TS follows the reactant order"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        ts.check_irc_species_and_rxn(xyz_1=endpoint_r, xyz_2=endpoint_p, rxn=rxn)
        ts_atom_map = rxn.ts_species.ts_atom_map
        self._assert_consistent(rxn, ts_atom_map, atom_map)
        self.assertEqual(ts_atom_map['reactants'], list(range(7)))
        self.assertEqual(ts_atom_map['products'], [0, 1, 2, 3, 5, 4, 6])
        self.assertIs(ts_atom_map['ts_atom_order_follows_reactants'], True)

    def test_a_swap_of_two_hydrogens_between_the_ts_and_the_endpoints_is_caught_on_the_ts_geometry(self):
        """Test that the transferring hydrogen sitting in a spectator position of the TS gives no map"""
        ts_coords = list(self.CH4_OH_TS)
        ts_coords[1], ts_coords[4] = ts_coords[4], ts_coords[1]
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)), ts_coords=tuple(ts_coords))
        with self.assertLogs('arc', level='WARNING') as logs:
            ts.check_irc_species_and_rxn(xyz_1=endpoint_r, xyz_2=endpoint_p, rxn=rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertIsNone(rxn.ts_species.ts_atom_map)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'ts_geometry_contradicts_map')
        self.assertTrue(any('TS geometry' in message for message in logs.output))

    def test_the_ts_geometry_check_uses_bonded_and_partial_bond_distances(self):
        """Test the distance limits: an intact bond is bonded, a forming or breaking one may be partial"""
        r_bonds, p_bonds = [(0, 1), (0, 2)], [(0, 1), (1, 2)]
        symbols = ('C', 'H', 'O')
        mapping = [0, 1, 2]

        def supported(coords):
            return ts._does_ts_geometry_support_map(xyz_from_data(coords=coords, symbols=symbols), mapping,
                                                    r_bonds, p_bonds)

        self.assertTrue(supported(((0.0, 0.0, 0.0), (1.09, 0.0, 0.0), (2.4, 0.0, 0.0))))
        self.assertFalse(supported(((0.0, 0.0, 0.0), (1.9, 0.0, 0.0), (2.4, 0.0, 0.0))))
        self.assertFalse(supported(((0.0, 0.0, 0.0), (1.09, 0.0, 0.0), (5.0, 0.0, 0.0))))

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
        ts.check_irc_species_and_rxn(xyz_1=endpoint_r, xyz_2=endpoint_p, rxn=rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self._assert_consistent(rxn, rxn.ts_species.ts_atom_map, atom_map)

        rxn, endpoint_r, endpoint_p, atom_map = self._build(symbols, r_coords, tuple(p_coords), ts_coords,
                                                            reactants, products, ts_order=list(range(10)),
                                                            crossed=(2, 5))
        with self.assertLogs('arc', level='WARNING') as logs:
            ts.check_irc_species_and_rxn(xyz_1=endpoint_r, xyz_2=endpoint_p, rxn=rxn)
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
                ts.check_irc_species_and_rxn(xyz_1=endpoint_r, xyz_2=endpoint_p, rxn=rxn)
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
        ts.check_irc_species_and_rxn(xyz_1=endpoint_r, xyz_2=endpoint_p, rxn=rxn)
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
                ts.check_irc_species_and_rxn(xyz_1=endpoint_r, xyz_2=endpoint_p, rxn=rxn)
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
        ts.check_irc_species_and_rxn(xyz_1=endpoint_r, xyz_2=endpoint_p, rxn=rxn)
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
        ts.check_irc_species_and_rxn(xyz_1=endpoint_r, xyz_2=endpoint_p, rxn=rxn)
        self.assertIsNone(rxn.ts_species.ts_atom_map)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'no_ts')

        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        symbols = rxn.ts_species.get_xyz(generate=False)['symbols']
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=xyz_from_data(
            coords=endpoint_r['coords'], symbols=symbols[::-1]))
        ts.check_irc_species_and_rxn(xyz_1=endpoint_r, xyz_2=endpoint_p, rxn=rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertEqual(rxn.ts_species.ts_atom_map_unavailable_reason, 'atom_order_mismatch')

        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        rxn._atom_map = None
        ts.check_irc_species_and_rxn(xyz_1=endpoint_r, xyz_2=endpoint_p, rxn=rxn)
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
                rxn=rxn, reactants=reactants, endpoint_xyzs={1: distorted, 2: endpoint_p}, reactant_endpoint=1,
                atom_order_matches_ts=True, ts_xyz=rxn.ts_species.get_xyz(generate=False))
        self.assertIsNone(ts_atom_map)
        self.assertEqual(reason, 'endpoint_perception_mismatch')
        self.assertTrue(any('perceived' in message for message in logs.output))

    def test_the_opposite_endpoint_roles_are_tried_when_the_sides_are_isomorphic(self):
        """Test that, when the sides are flagged isomorphic, the endpoints given in the wrong roles still map"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        reactants, products = rxn.get_reactants_and_products(return_copies=False)
        kwargs = dict(rxn=rxn, reactants=reactants, endpoint_xyzs={1: endpoint_p, 2: endpoint_r}, reactant_endpoint=1,
                      atom_order_matches_ts=True, ts_xyz=rxn.ts_species.get_xyz(generate=False))
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
            ts.check_irc_species_and_rxn(xyz_1=endpoint_r, xyz_2=endpoint_p, rxn=rxn)
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
            ts.check_irc_species_and_rxn(xyz_1=endpoint_r, xyz_2=endpoint_p, rxn=rxn)
        self.assertIs(rxn.ts_species.ts_checks['IRC'], True)
        self.assertIsNone(rxn.ts_species.irc_participant_mapping)
        self._assert_consistent(rxn, rxn.ts_species.ts_atom_map, atom_map)

    def test_a_failure_comparing_the_endpoints_with_the_ts_fails_both_records(self):
        """Test that, with no atom order verdict, there is no participant mapping and the reason is computation_failed"""
        rxn, endpoint_r, endpoint_p, atom_map = self._ch4_oh(ts_order=list(range(7)))
        with patch.object(ts, '_get_irc_endpoints_atom_order_matches_ts', side_effect=RuntimeError('boom')), \
                self.assertLogs('arc', level='WARNING'):
            ts.check_irc_species_and_rxn(xyz_1=endpoint_r, xyz_2=endpoint_p, rxn=rxn)
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
