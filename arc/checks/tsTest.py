#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.checks.ts module
"""

import unittest
import os
import shutil

import numpy as np

import arc.checks.ts as ts
from arc.common import ARC_PATH
from arc.job.factory import job_factory
from arc.level import Level
from arc.parser import parse_normal_mode_displacement
from arc.reaction import ARCReaction
from arc.rmgdb import load_families_only, make_rmg_database_object
from arc.species.species import ARCSpecies, TSGuess
from arc.utils.wip import work_in_progress


class TestChecks(unittest.TestCase):
    """
    Contains unit tests for the check module.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.rmgdb = make_rmg_database_object()
        load_families_only(cls.rmgdb)

        cls.rms_list_1 = [0.01414213562373095, 0.05, 0.04, 0.5632938842203065, 0.7993122043357026, 0.08944271909999159,
                          0.10677078252031312, 0.09000000000000001, 0.05, 0.09433981132056604]
        path_1 = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'freq', 'C3H7_intra_h_TS.out')
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

        cls.ts_xyz_2 = """C       0.52123900   -0.93806900   -0.55301700
C       0.15387500    0.18173100    0.37122900
C      -0.89554000    1.16840700   -0.01362800
H       0.33997700    0.06424800    1.44287100
H       1.49602200   -1.37860200   -0.29763200
H       0.57221700   -0.59290500   -1.59850500
H       0.39006800    1.39857900   -0.01389600
H      -0.23302200   -1.74751100   -0.52205400
H      -1.43670700    1.71248300    0.76258900
H      -1.32791000    1.11410600   -1.01554900"""  # C[CH]C <=> [CH2]CC
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
                               project_directory=os.path.join(ARC_PATH,
                                                              'Projects',
                                                              'arc_project_for_testing_delete_after_usage4'),
                               )

        cls.rxn_3 = ARCReaction(r_species=[ARCSpecies(label='NH3', smiles='N'), ARCSpecies(label='H', smiles='[H]')],
                                p_species=[ARCSpecies(label='NH2', smiles='[NH2]'), ARCSpecies(label='H2', smiles='[H][H]')])
        cls.rxn_3.ts_species = ARCSpecies(label='TS3', is_ts=True,
                                          xyz=os.path.join(ts.ARC_PATH, 'arc', 'testing', 'freq', 'TS_NH3+H=NH2+H2.out'))

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
                                          xyz=os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite', 'TS0_composite_2043.out'))

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
                                          xyz=os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite', 'TS0_composite_2044.out'))
        cls.rxn_6 = ARCReaction(r_species=[ea, ARCSpecies(label='H', smiles='[H]')],
                                p_species=[ARCSpecies(label='CH2CH2NH2', smiles='[CH2]CN'), ARCSpecies(label='H2', smiles='[H][H]')])
        cls.rxn_6.ts_species = ARCSpecies(label='TS6', is_ts=True,
                                          xyz=os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite', 'TS1_composite_695.out'))

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
        cls.rxn_7.ts_species = ARCSpecies(label='TS7', is_ts=True,
                                          xyz=os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite', 'keto_enol_ts.out'))

        cls.rxn_2a.determine_family(rmg_database=cls.rmgdb, save_order=True)
        cls.rxn_2b.determine_family(rmg_database=cls.rmgdb, save_order=True)
        cls.rxn_3.determine_family(rmg_database=cls.rmgdb, save_order=True)
        cls.rxn_4.determine_family(rmg_database=cls.rmgdb, save_order=True)
        cls.rxn_5.determine_family(rmg_database=cls.rmgdb, save_order=True)
        cls.rxn_6.determine_family(rmg_database=cls.rmgdb, save_order=True)
        cls.rxn_7.determine_family(rmg_database=cls.rmgdb, save_order=True)

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

    def test_check_ts(self):
        """Test the check_ts() function."""
        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'freq', 'TS_C3_intraH_8.out')
        self.rxn_2a.ts_species.populate_ts_checks()
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['normal_mode_displacement'])
        ts.check_ts(reaction=self.rxn_2a, job=self.job1)
        self.assertTrue(self.rxn_2a.ts_species.ts_checks['normal_mode_displacement'])

        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite', 'keto_enol_ts.out')
        self.rxn_7.ts_species.populate_ts_checks()
        self.assertFalse(self.rxn_7.ts_species.ts_checks['normal_mode_displacement'])
        ts.check_ts(reaction=self.rxn_7, job=self.job1)
        self.assertTrue(self.rxn_7.ts_species.ts_checks['normal_mode_displacement'])

    def test_did_ts_pass_all_checks(self):
        """Test the did_ts_pass_all_checks() function."""
        spc = ARCSpecies(label='TS', is_ts=True)
        spc.populate_ts_checks()
        self.assertFalse(ts.ts_passed_all_checks(spc))

        self.ts_checks = {'E0': False,
                          'e_elect': False,
                          'IRC': False,
                          'freq': False,
                          'normal_mode_displacement': False,
                          'warnings': '',
                          }
        for key in ['E0', 'e_elect', 'IRC', 'freq']:
            spc.ts_checks[key] = True
        self.assertFalse(ts.ts_passed_all_checks(spc))
        self.assertTrue(ts.ts_passed_all_checks(spc, exemptions=['normal_mode_displacement', 'warnings']))
        spc.ts_checks['e_elect'] = False

    def test_check_ts_energy(self):
        """Test the check_ts_energy() function."""
        rxn1 = ARCReaction(r_species=[ARCSpecies(label='s1', smiles='C')], p_species=[ARCSpecies(label='s2', smiles='C')])
        rxn1.ts_species = ARCSpecies(label='TS', is_ts=True)
        # no data
        rxn1.ts_species.populate_ts_checks()
        ts.check_ts_energy(reaction=rxn1)
        self.assertIsNone(rxn1.ts_species.ts_checks['e_elect'])
        # only E0 (incorrect)
        rxn1.r_species[0].e0 = 2
        rxn1.p_species[0].e0 = 50
        rxn1.ts_species.e0 = -100
        rxn1.ts_species.populate_ts_checks()
        ts.check_ts_energy(reaction=rxn1)
        self.assertFalse(rxn1.ts_species.ts_checks['E0'])
        # only E0 (partial data)
        rxn1.r_species[0].e0 = 2
        rxn1.p_species[0].e0 = None
        rxn1.ts_species.e0 = -100
        rxn1.ts_species.populate_ts_checks()
        ts.check_ts_energy(reaction=rxn1)
        self.assertIsNone(rxn1.ts_species.ts_checks['e_elect'])
        # check e_elect (correct)
        rxn1.r_species[0].e_elect = 2
        rxn1.p_species[0].e_elect = 50
        rxn1.ts_species.e_elect = 100
        rxn1.ts_species.populate_ts_checks()
        ts.check_ts_energy(reaction=rxn1)
        self.assertTrue(rxn1.ts_species.ts_checks['e_elect'])
        # incorrect e_elect
        rxn1.r_species[0].e_elect = 2
        rxn1.p_species[0].e_elect = 50
        rxn1.ts_species.e_elect = -100
        rxn1.ts_species.populate_ts_checks()
        ts.check_ts_energy(reaction=rxn1)
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

    def test_check_normal_mode_displacement(self):
        """Test the check_normal_mode_displacement() function."""
        self.rxn_2a.ts_species.populate_ts_checks()
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['normal_mode_displacement'])
        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                           'TS_intra_H_migration_CBS-QB3.out')
        self.rxn_2a.determine_family(rmg_database=self.rmgdb)
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertTrue(self.rxn_2a.ts_species.ts_checks['normal_mode_displacement'])
        self.rxn_2a.ts_species.populate_ts_checks()

        self.rxn_2b.ts_species.populate_ts_checks()
        self.assertFalse(self.rxn_2b.ts_species.ts_checks['normal_mode_displacement'])
        ts.check_normal_mode_displacement(reaction=self.rxn_2b, job=self.job1)
        self.assertFalse(self.rxn_2b.ts_species.ts_checks['normal_mode_displacement'])

        # Wrong TS for intra H migration [CH2]CC <=> C[CH]C
        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                           'TS_C3_intraH_1.out')  # A wrong TS.
        self.rxn_2a.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['normal_mode_displacement'])

        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                           'TS_C3_intraH_2.out')  # A wrong TS.
        self.rxn_2a.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['normal_mode_displacement'])

        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                           'TS_C3_intraH_3.out')  # ** The correct TS. **
        self.rxn_2a.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertTrue(self.rxn_2a.ts_species.ts_checks['normal_mode_displacement'])

        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                           'TS_C3_intraH_4.out')  # A wrong TS.
        self.rxn_2a.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['normal_mode_displacement'])

        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                           'TS_C3_intraH_5.out')  # A wrong TS.
        self.rxn_2a.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['normal_mode_displacement'])

        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                           'TS_C3_intraH_6.out')  # A wrong TS.
        self.rxn_2a.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['normal_mode_displacement'])

        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                           'TS_C3_intraH_7.out')  # A wrong TS.
        self.rxn_2a.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['normal_mode_displacement'])

        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'freq',
                                                           'TS_C3_intraH_8.out')  # Correct TS (freq run, not composite).
        self.rxn_2a.ts_species.populate_ts_checks()
        self.assertFalse(self.rxn_2a.ts_species.ts_checks['normal_mode_displacement'])
        ts.check_normal_mode_displacement(reaction=self.rxn_2a, job=self.job1)
        self.assertTrue(self.rxn_2a.ts_species.ts_checks['normal_mode_displacement'])

        # CCO[O] + CC <=> CCOO + [CH2]C, incorrect TS:
        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                           'TS0_composite_2043.out')
        self.rxn_4.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_4, job=self.job1)
        self.assertFalse(self.rxn_4.ts_species.ts_checks['normal_mode_displacement'])

        # CCO[O] + CC <=> CCOO + [CH2]C, correct TS:
        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                           'TS0_composite_2102.out')
        self.rxn_4.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_4, job=self.job1)
        self.assertTrue(self.rxn_4.ts_species.ts_checks['normal_mode_displacement'])

        # NCC + H <=> CH3CHNH2 + H2, correct TS:
        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                           'TS0_composite_2044.out')
        self.rxn_5.ts_species.populate_ts_checks()
        ts.check_normal_mode_displacement(reaction=self.rxn_5, job=self.job1)
        self.assertTrue(self.rxn_5.ts_species.ts_checks['normal_mode_displacement'])

        # NH2 + N2H3 <=> NH + N2H4:
        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'freq', 'TS_NH2+N2H3.out')
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
        self.assertTrue(rxn_6.ts_species.ts_checks['normal_mode_displacement'])

        # [CH2]CC=C <=> CCC=[CH] butylene intra H migration:
        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                           'TS_butylene_intra_H_migration.out')
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
        self.assertTrue(rxn_7.ts_species.ts_checks['normal_mode_displacement'])

    @work_in_progress
    def test_check_normal_mode_displacement_wip(self):
        """Test the check_normal_mode_displacement() function."""
        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'freq',
                                                           'TS_NH3+H=NH2+H2.out')  # NH3 + H <=> NH2 + H2
        self.rxn_3.ts_species.populate_ts_checks()
        self.assertFalse(self.rxn_3.ts_species.ts_checks['normal_mode_displacement'])
        ts.check_normal_mode_displacement(reaction=self.rxn_3, job=self.job1)
        self.assertTrue(self.rxn_3.ts_species.ts_checks['normal_mode_displacement'])

    def test_invalidate_rotors_with_both_pivots_in_a_reactive_zone(self):
        """Test the invalidate_rotors_with_both_pivots_in_a_reactive_zone() function."""
        ts_spc_1 = ARCSpecies(label='TS', is_ts=True, xyz=self.ts_xyz_1)
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

        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                           'TS_intra_H_migration_CBS-QB3.out')
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
        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                           'TS0_composite_2102.out')
        rxn_zone_atom_indices = ts.get_rxn_zone_atom_indices(reaction=self.rxn_4, job=self.job1)
        self.assertEqual(rxn_zone_atom_indices, [14, 10, 3, 2, 16])

        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                           'TS0_composite_2044.out')
        rxn_zone_atom_indices = ts.get_rxn_zone_atom_indices(reaction=self.rxn_5, job=self.job1)
        self.assertEqual(rxn_zone_atom_indices, [5, 10, 1, 3, 0])

        self.job1.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                           'TS1_composite_695.out')
        rxn_zone_atom_indices = ts.get_rxn_zone_atom_indices(reaction=self.rxn_6, job=self.job1)
        self.assertEqual(rxn_zone_atom_indices, [7, 2, 9, 8, 1])  # Atom 10, the abstractor, is not moving

    def test_get_rms_from_normal_modes_disp(self):
        """Test the get_rms_from_normal_modes_disp() function."""
        rms = ts.get_rms_from_normal_mode_disp(normal_mode_disp=self.normal_modes_disp_1,
                                               freqs=np.array([-1000.3, 320.5], np.float64))
        self.assertEqual(rms, [0.07874007874011811, 0.07280109889280519, 0.0, 0.9914635646356349, 0.03605551275463989,
                               0.034641016151377546, 0.0, 0.033166247903554, 0.01414213562373095, 0.0])

        freqs_2, normal_modes_disp_2 = parse_normal_mode_displacement(
            os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite', 'TS0_composite_2102.out'))
        rms = ts.get_rms_from_normal_mode_disp(normal_mode_disp=normal_modes_disp_2,
                                               freqs=freqs_2,
                                               reaction=ARCReaction(r_species=[ARCSpecies(label='CCOOj', xyz=self.ccooj_xyz),
                                                                               ARCSpecies(label='C2H6', xyz=self.c2h6_xyz)],
                                                                    p_species=[ARCSpecies(label='CCOOH', smiles='CCOO'),
                                                                               ARCSpecies(label='C2H5', smiles='C[CH2]')])
                                               )
        self.assertEqual(rms, [0.0,                     # 0
                               0.0008785397004602187,   # 1
                               0.0034115627396496605,   # 2
                               0.004483671853664259,    # 3 * (the abstracting O)
                               0.0,                     # 4
                               0.0,                     # 5
                               0.00022485182747281122,  # 6
                               0.0,                     # 7
                               0.00022485182747281122,  # 8
                               0.0008785397004602187,    # 9
                               0.0076083769883172085,   # 10 * (the abstracting C)
                               0.00022485182747281122,  # 11
                               0.0,                     # 12
                               0.00022485182747281122,  # 13
                               0.021950421894561232,    # 14 * (the abstracted H)
                               0.0027261835255033436,   # 15
                               0.002790337797642228])   # 16

    def test_get_index_of_abs_largest_neg_freq(self):
        """Test the get_index_of_abs_largest_neg_freq() function."""
        self.assertIsNone(ts.get_index_of_abs_largest_neg_freq(np.array([], np.float64)))
        self.assertIsNone(ts.get_index_of_abs_largest_neg_freq(np.array([1, 320.5], np.float64)))
        self.assertEqual(ts.get_index_of_abs_largest_neg_freq(np.array([-1], np.float64)), 0)
        self.assertEqual(ts.get_index_of_abs_largest_neg_freq(np.array([-1, 320.5], np.float64)), 0)
        self.assertEqual(ts.get_index_of_abs_largest_neg_freq(np.array([320.5, -1], np.float64)), 1)
        self.assertEqual(ts.get_index_of_abs_largest_neg_freq(np.array([320.5, -1, -80, -90, 5000], np.float64)), 3)
        self.assertEqual(ts.get_index_of_abs_largest_neg_freq(np.array([-320.5, -1, -80, -90, 5000], np.float64)), 0)

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
        self.assertEqual(ts.get_rxn_normal_mode_disp_atom_number(), 3)
        self.assertEqual(ts.get_rxn_normal_mode_disp_atom_number('default'), 3)
        self.assertEqual(ts.get_rxn_normal_mode_disp_atom_number('intra_H_migration'), 3)
        self.assertEqual(ts.get_rxn_normal_mode_disp_atom_number('intra_H_migration', rms_list=self.rms_list_1), 4)

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

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        projects = ['arc_project_for_testing_delete_after_usage4']
        for project in projects:
            project_directory = os.path.join(ARC_PATH, 'Projects', project)
            shutil.rmtree(project_directory, ignore_errors=True)
        file_paths = [os.path.join(ARC_PATH, 'arc', 'checks', 'nul'), os.path.join(ARC_PATH, 'arc', 'checks', 'run.out')]
        for file_path in file_paths:
            if os.path.isfile(file_path):
                os.remove(file_path)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
