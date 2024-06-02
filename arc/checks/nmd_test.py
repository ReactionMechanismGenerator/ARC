#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.checks.ts module
"""

import unittest
import os
import shutil

import numpy as np

import arc.checks.nmd as nmd
from arc.common import ARC_PATH, almost_equal_lists
from arc.job.factory import job_factory
from arc.level import Level
from arc.parser import parse_normal_mode_displacement, parse_xyz_from_file
from arc.reaction import ARCReaction
from arc.rmgdb import load_families_only, make_rmg_database_object
from arc.reaction import Reaction
from arc.species.species import ARCSpecies, TSGuess
from arc.utils.wip import work_in_progress


class TestNMD(unittest.TestCase):
    """
    Contains unit tests for the nmd module.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.xyz_1 = {'symbols': ('C', 'N', 'H', 'H', 'H', 'H'),
                     'isotopes': (13, 14, 1, 1, 1, 1),
                     'coords': ((0.6616514836, 0.4027481525, -0.4847382281),
                                (-0.6039793084, 0.6637270105, 0.0671637135),
                                (-1.4226865648, -0.4973210697, -0.2238712255),
                                (-0.4993010635, 0.6531020442, 1.0853092315),
                                (-2.2115796924, -0.4529256762, 0.4144516252),
                                (-1.8113671395, -0.3268900681, -1.1468957003))}
        cls.weights_1 = np.array([[3.60601648], [3.74206815], [1.00390489], [1.00390489], [1.00390489], [1.00390489]])
        cls.nmd_1 = np.array([[-0.5, 0.0, -0.09], [-0.0, 0.0, -0.1], [0.0, 0.0, -0.01],
                              [0.0, 0.0, -0.07], [0.0, 0.0, -0.2], [-0.0, -0.0, 0.28]], np.float64)

        ch4_xyz = """C      -0.00000000    0.00000000    0.00000000
                     H      -0.65055201   -0.77428020   -0.41251879
                     H      -0.34927558    0.98159583   -0.32768232
                     H      -0.02233792   -0.04887375    1.09087665
                     H       1.02216551   -0.15844188   -0.35067554"""
        oh_xyz = """O       0.48890387    0.00000000    0.00000000
                    H      -0.48890387    0.00000000    0.00000000"""
        ch3_xyz = """C       0.00000000    0.00000001   -0.00000000
                     H       1.06690511   -0.17519582    0.05416493
                     H      -0.68531716   -0.83753536   -0.02808565
                     H      -0.38158795    1.01273118   -0.02607927"""
        h2o_xyz = """O      -0.00032832    0.39781490    0.00000000
                     H      -0.76330345   -0.19953755    0.00000000
                     H       0.76363177   -0.19827735    0.00000000"""
        cls.rxn_1 = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C', xyz=ch4_xyz),
                                           ARCSpecies(label='OH', smiles='[OH]', xyz=oh_xyz)],
                                p_species=[ARCSpecies(label='CH3', smiles='[CH3]', xyz=ch3_xyz),
                                           ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz)])
        # cls.rmgdb = make_rmg_database_object()
        # load_families_only(cls.rmgdb)

    def test_analyze_ts_normal_mode_displacement(self):
        """Test the analyze_ts_normal_mode_displacement() function."""
        # 27, 32, 33
        job = job_factory(job_adapter='gaussian',
                          species=[ARCSpecies(label='SPC', smiles='C')],
                          job_type='composite',
                          level=Level(method='CBS-QB3'),
                          project='test_project',
                          project_directory=os.path.join(ARC_PATH, 'Projects'),
                          )
        path = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'HO2+N2H4_H2O2+N2H3_freq.out')

    def test_get_weights_from_xyz(self):
        """Test the get_weights_from_xyz() function."""
        weights = nmd.get_weights_from_xyz(xyz=self.xyz_1, weights=False)
        np.testing.assert_array_equal(weights, np.array([[1], [1], [1], [1], [1], [1]]))

        weights = nmd.get_weights_from_xyz(xyz=self.xyz_1, weights=True)
        np.testing.assert_almost_equal(weights, self.weights_1)

        w_array = np.array([[10], [1], [1], [3], [1], [1]])
        weights = nmd.get_weights_from_xyz(xyz=self.xyz_1, weights=w_array)
        np.testing.assert_equal(weights, w_array)

    def test_get_bond_length_changes_baseline_and_std(self):
        """Test the get_bond_length_changes_baseline_and_std() function."""
        baseline_max, std = nmd.get_bond_length_changes_baseline_and_std(reaction=self.rxn_1)
        self.assertAlmostEqual(baseline_max, 0.00965, 4)
        self.assertAlmostEqual(std, 0.000365, 5)

    def test_get_bond_length_in_reaction(self):
        """Test the get_bond_length_in_reaction() function."""
        bond_length = nmd.get_bond_length_in_reaction(species=[self.rxn_1.r_species[0]], bond=(0, 1))
        self.assertAlmostEqual(bond_length, 1.0922, 4)

        bond_length = nmd.get_bond_length_in_reaction(species=[self.rxn_1.r_species[1]], bond=(0, 1))
        self.assertAlmostEqual(bond_length, 0.9778, 4)

        reactants, products = self.rxn_1.get_reactants_and_products()
        bond_length = nmd.get_bond_length_in_reaction(species=reactants, bond=(5, 6))
        self.assertAlmostEqual(bond_length, 0.9778, 4)

    def test_get_displaced_xyzs(self):
        """Test the get_displaced_xyzs() function."""
        xyzs = nmd.get_displaced_xyzs(xyz=self.xyz_1, amplitude=0.1, normal_mode_disp=self.nmd_1, weights=self.weights_1)
        expected_xyz_1 = np.array([[0.84195231, 0.40274815, -0.45228408],
                                   [-0.60397931, 0.66372701, 0.1045844],
                                   [-1.42268656, -0.49732107, -0.22286732],
                                   [-0.49930106, 0.65310204, 1.09233657],
                                   [-2.21157969, -0.45292568, 0.43452972],
                                   [-1.81136714, -0.32689007, -1.17500504]])
        expected_xyz_2 = np.array([[0.48135066, 0.40274815, -0.51719238],
                                   [-0.60397931, 0.66372701, 0.02974303],
                                   [-1.42268656, -0.49732107, -0.22487513],
                                   [-0.49930106, 0.65310204, 1.0782819],
                                   [-2.21157969, -0.45292568, 0.39437353],
                                   [-1.81136714, -0.32689007, -1.11878636]])
        np.testing.assert_almost_equal(xyzs[0], expected_xyz_1)
        np.testing.assert_almost_equal(xyzs[1], expected_xyz_2)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        pass


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
