#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.checks.nmd module
"""

import unittest
import os

import numpy as np

import arc.checks.nmd as nmd
from arc.common import ARC_PATH
from arc.job.factory import job_factory
from arc.level import Level
from arc.parser import parse_normal_mode_displacement
from arc.reaction import ARCReaction
from arc.species.species import ARCSpecies
from arc.species.converter import check_xyz_dict, xyz_to_str
from rmgpy.molecule import Molecule


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

        cls.ch4_xyz = """C      -0.00000000    0.00000000    0.00000000
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
        cls.rxn_1 = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C', xyz=cls.ch4_xyz),
                                           ARCSpecies(label='OH', smiles='[OH]', xyz=oh_xyz)],
                                p_species=[ARCSpecies(label='CH3', smiles='[CH3]', xyz=ch3_xyz),
                                           ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz)])
        cls.ts_1_xyz = check_xyz_dict("""C                    -1.212192   -0.010161    0.000000
                                         H                     0.010122    0.150115    0.000001
                                         H                    -1.460491   -0.555461   -0.907884
                                         H                    -1.460499   -0.555391    0.907924
                                         H                    -1.600535    1.006984   -0.000041
                                         O                     1.295714    0.108618   -0.000000
                                         H                     1.418838   -0.854225    0.000000""")
        cls.rxn_1.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=cls.ts_1_xyz)
        cls.freq_log_path_1 = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'TS_CH4_OH.log')

        cls.ho2_xyz = """O       1.00509800   -0.18331500   -0.00000000
                         O      -0.16548400    0.44416100    0.00000000
                         H      -0.83961400   -0.26084600    0.00000000"""
        cls.n2h4_xyz = """N      -0.66510800   -0.10671700   -0.25444200
                          N       0.63033400    0.04211900    0.34557500
                          H      -1.16070500    0.76768900   -0.12511600
                          H      -1.21272700   -0.83945300    0.19196500
                          H       1.26568700   -0.57247200   -0.14993500
                          H       0.63393800   -0.23649100    1.32457000"""
        cls.h2o2_xyz = """O       0.60045000   -0.40342400    0.24724100
                          O      -0.59754500    0.41963800    0.22641300
                          H       1.20401100    0.16350100   -0.25009400
                          H      -1.20691600   -0.17971500   -0.22356000"""
        cls.n2h3_xyz = """N       0.74263400   -0.29604200    0.40916100
                          N      -0.39213800   -0.13735700   -0.31177100
                          H       1.49348100    0.07315400   -0.18245700
                          H      -1.18274100   -0.63578900    0.07132400
                          H      -0.36438800   -0.12591900   -1.32684600"""
        cls.rxn_2 = ARCReaction(r_species=[ARCSpecies(label='HO2', smiles='O[O]', xyz=cls.ho2_xyz),
                                           ARCSpecies(label='N2H4', smiles='NN', xyz=cls.n2h4_xyz)],
                                p_species=[ARCSpecies(label='H2O2', smiles='OO', xyz=cls.h2o2_xyz),
                                           ARCSpecies(label='N2H3', smiles='N[NH]', xyz=cls.n2h3_xyz)])
        cls.ts_2_xyz = check_xyz_dict("""O      -1.275006   -0.656458   -0.217271
                                         O      -1.463976    0.650010    0.244741
                                         H      -1.239724    1.193859   -0.524436
                                         N       1.456296    0.587219   -0.119103
                                         N       1.134258   -0.749491    0.081665
                                         H       1.953072    0.715591   -0.992228
                                         H       1.926641    1.040462    0.653661
                                         H      -0.004455   -0.848273   -0.183117
                                         H       1.142454   -0.914148    1.088425""")
        cls.rxn_2.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=cls.ts_2_xyz)
        cls.freq_log_path_2 = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'HO2+N2H4_H2O2+N2H3_freq.out')

        cls.displaced_xyz_3 = ({'symbols': ('C', 'N', 'H', 'H', 'H', 'H'),
                                'isotopes': (13, 14, 1, 1, 1, 1),
                                'coords': ((0.8419523076000001, 0.4027481525, -0.45228407978),
                                           (-0.6039793084, 0.6637270105, 0.10458439500000001),
                                           (-1.4226865648, -0.4973210697, -0.22286732061),
                                           (-0.4993010635, 0.6531020442, 1.09233656573),
                                           (-2.2115796924, -0.4529256762, 0.43452972300000003),
                                           (-1.8113671395, -0.3268900681, -1.17500503722))},
                               {'symbols': ('C', 'N', 'H', 'H', 'H', 'H'),
                                'isotopes': (13, 14, 1, 1, 1, 1),
                                'coords': ((0.4813506596, 0.4027481525, -0.51719237642),
                                           (-0.6039793084, 0.6637270105, 0.02974303199999999),
                                           (-1.4226865648, -0.4973210697, -0.22487513038999998),
                                           (-0.4993010635, 0.6531020442, 1.07828189727),
                                           (-2.2115796924, -0.4529256762, 0.3943735274),
                                           (-1.8113671395, -0.3268900681, -1.11878636338))})

    def test_analyze_ts_normal_mode_displacement(self):
        """Test the analyze_ts_normal_mode_displacement() function."""
        job_1 = job_factory(job_adapter='gaussian',
                            species=[self.rxn_1.ts_species],
                            job_type='composite',
                            level=Level(method='CBS-QB3'),
                            project='test_project',
                            project_directory=os.path.join(ARC_PATH, 'Projects'),
                            )
        job_1.local_path_to_output_file = self.freq_log_path_1
        self.assertTrue(nmd.analyze_ts_normal_mode_displacement(reaction=self.rxn_1,
                                                                job=job_1,
                                                                amplitude=0.1,
                                                                weights=True))
        self.assertTrue(nmd.analyze_ts_normal_mode_displacement(reaction=self.rxn_1,
                                                                job=job_1,
                                                                amplitude=[0, 0.001, 0.5],
                                                                weights=True))

        job_2 = job_factory(job_adapter='gaussian',
                            species=[self.rxn_2.ts_species],
                            job_type='composite',
                            level=Level(method='CBS-QB3'),
                            project='test_project',
                            project_directory=os.path.join(ARC_PATH, 'Projects'),
                            )
        job_2.local_path_to_output_file = self.freq_log_path_2
        valid = nmd.analyze_ts_normal_mode_displacement(reaction=self.rxn_2, job=job_2, amplitude=0.1, weights=True)
        self.assertTrue(valid)

    def test_translate_all_tuples_simultaneously(self):
        """Test the translate_all_tuples_simultaneously() function."""
        translated_tuples = nmd.translate_all_tuples_simultaneously(list_1=[(0, 1)],
                                                                    list_2=[],
                                                                    equivalences=[])
        self.assertEqual(translated_tuples, [([(0, 1)], [])])

        translated_tuples = nmd.translate_all_tuples_simultaneously(list_1=[(0, 1)],
                                                                    list_2=[],
                                                                    equivalences=[[0, 3, 4]])
        self.assertEqual(translated_tuples, [([(0, 1)], []), ([(3, 1)], []), ([(4, 1)], [])])

        translated_tuples = nmd.translate_all_tuples_simultaneously(list_1=[(0, 1)],
                                                                    list_2=[(1, 5)],
                                                                    equivalences=[[1, 2, 3, 4]])
        self.assertEqual(translated_tuples, [([(0, 1)], [(1, 5)]),
                                             ([(0, 2)], [(2, 5)]),
                                             ([(0, 3)], [(3, 5)]),
                                             ([(0, 4)], [(4, 5)])])

        translated_tuples = nmd.translate_all_tuples_simultaneously(list_1=[(0, 1)],
                                                                    list_2=[(1, 5)],
                                                                    equivalences=[[1, 2, 3, 4], [5, 6]])
        self.assertEqual(translated_tuples, [([(0, 1)], [(1, 5)]),
                                             ([(0, 1)], [(1, 6)]),
                                             ([(0, 2)], [(2, 5)]),
                                             ([(0, 2)], [(2, 6)]),
                                             ([(0, 3)], [(3, 5)]),
                                             ([(0, 3)], [(3, 6)]),
                                             ([(0, 4)], [(4, 5)]),
                                             ([(0, 4)], [(4, 6)])])

        translated_tuples = nmd.translate_all_tuples_simultaneously(list_1=[(0, 1), (0, 2)],
                                                                    list_2=[(1, 5)],
                                                                    equivalences=[[1, 2, 3, 4], [5, 6]])
        self.assertEqual(translated_tuples, [([(0, 1), (0, 2)], [(1, 5)]),
                                             ([(0, 1), (0, 2)], [(1, 6)]),
                                             ([(0, 1), (0, 3)], [(1, 5)]),
                                             ([(0, 1), (0, 3)], [(1, 6)]),
                                             ([(0, 1), (0, 4)], [(1, 5)]),
                                             ([(0, 1), (0, 4)], [(1, 6)]),
                                             ([(0, 2), (0, 1)], [(2, 5)]),
                                             ([(0, 2), (0, 1)], [(2, 6)]),
                                             ([(0, 2), (0, 3)], [(2, 5)]),
                                             ([(0, 2), (0, 3)], [(2, 6)]),
                                             ([(0, 2), (0, 4)], [(2, 5)]),
                                             ([(0, 2), (0, 4)], [(2, 6)]),
                                             ([(0, 3), (0, 1)], [(3, 5)]),
                                             ([(0, 3), (0, 1)], [(3, 6)]),
                                             ([(0, 3), (0, 2)], [(3, 5)]),
                                             ([(0, 3), (0, 2)], [(3, 6)]),
                                             ([(0, 3), (0, 4)], [(3, 5)]),
                                             ([(0, 3), (0, 4)], [(3, 6)]),
                                             ([(0, 4), (0, 1)], [(4, 5)]),
                                             ([(0, 4), (0, 1)], [(4, 6)]),
                                             ([(0, 4), (0, 2)], [(4, 5)]),
                                             ([(0, 4), (0, 2)], [(4, 6)]),
                                             ([(0, 4), (0, 3)], [(4, 5)]),
                                             ([(0, 4), (0, 3)], [(4, 6)])])

    def test_create_equivalence_mapping(self):
        """Test the create_equivalence_mapping() function."""
        mapping = nmd.create_equivalence_mapping(equivalences=[[1, 2, 3, 4]])
        self.assertEqual(mapping, {1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4], 4: [1, 2, 3, 4]})
        mapping = nmd.create_equivalence_mapping(equivalences=[[1, 2, 3, 4], [5, 6]])
        self.assertEqual(mapping, {1: [1, 2, 3, 4],
                                   2: [1, 2, 3, 4],
                                   3: [1, 2, 3, 4],
                                   4: [1, 2, 3, 4],
                                   5: [5, 6],
                                   6: [5, 6]})

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
        weights = nmd.get_weights_from_xyz(xyz=self.ts_2_xyz, weights=False)
        freqs, normal_mode_disp = parse_normal_mode_displacement(path=self.freq_log_path_2)
        np.testing.assert_array_almost_equal(freqs[0], -1350.1119)
        xyzs = nmd.get_displaced_xyzs(xyz=self.ts_2_xyz,
                                      amplitude=0.1,
                                      normal_mode_disp=normal_mode_disp[0],
                                      weights=weights,
                                      )
        baseline_max, std = nmd.get_bond_length_changes_baseline_and_std(
            non_reactive_bonds=[(0, 1), (1, 2), (4, 7), (4, 8), (3, 6), (3, 4)],
            xyzs=xyzs)
        self.assertAlmostEqual(baseline_max, 0.18455072, 4)
        self.assertAlmostEqual(std, 0.0664312, 4)

    def test_get_bond_length_in_reaction(self):
        """Test the get_bond_length_in_reaction() function."""
        bond_length = nmd.get_bond_length_in_reaction(bond=(0, 1), xyz=self.rxn_1.r_species[0].get_xyz())
        self.assertAlmostEqual(bond_length, 1.0922, 4)
        bond_length = nmd.get_bond_length_in_reaction(bond=(0, 1), xyz=self.rxn_1.r_species[1].get_xyz())
        self.assertAlmostEqual(bond_length, 0.9778, 4)

    def test_get_displaced_xyzs(self):
        """Test the get_displaced_xyzs() function."""
        xyzs = nmd.get_displaced_xyzs(xyz=self.xyz_1,
                                      amplitude=0.1,
                                      normal_mode_disp=self.nmd_1,
                                      weights=self.weights_1,
                                      )
        self.assertEqual(xyzs[0], self.displaced_xyz_3[0])
        self.assertEqual(xyzs[1], self.displaced_xyz_3[1])

        normal_mode_disp = np.array([[-0.06, 0.04, 0.01], [-0., -0.03, -0.01], [0.08, 0., 0.02],
                                     [-0.02, 0.04, -0.02], [-0., -0.03, 0.01], [0.14, -0.05, 0.06],
                                     [0.04, -0.07, 0.01], [0.96, -0.16, 0.02], [0.04, -0.06, -0.01]])
        weights = np.array([[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]])
        xyzs = nmd.get_displaced_xyzs(xyz=self.ts_2_xyz,
                                      amplitude=0.3,
                                      normal_mode_disp=normal_mode_disp,
                                      weights=weights,
                                      )
        expected_xyzs = ({'symbols': ('O', 'O', 'H', 'N', 'N', 'H', 'H', 'H', 'H'),
                          'isotopes': (16, 16, 1, 14, 14, 1, 1, 1, 1),
                          'coords': ((-1.257006, -0.668458, -0.220271), (-1.463976, 0.65901, 0.247741),
                                     (-1.263724, 1.193859, -0.530436), (1.462296, 0.575219, -0.113103),
                                     (1.134258, -0.740491, 0.078665), (1.9110719999999999, 0.730591, -1.010228),
                                     (1.914641, 1.061462, 0.650661), (-0.29245499999999996, -0.800273, -0.189117),
                                     (1.130454, -0.896148, 1.0914249999999999))},
                         {'symbols': ('O', 'O', 'H', 'N', 'N', 'H', 'H', 'H', 'H'),
                          'isotopes': (16, 16, 1, 14, 14, 1, 1, 1, 1),
                          'coords': ((-1.293006, -0.644458, -0.214271), (-1.463976, 0.64101, 0.24174099999999998),
                                     (-1.215724, 1.193859, -0.518436), (1.450296, 0.5992190000000001, -0.125103),
                                     (1.134258, -0.758491, 0.084665), (1.995072, 0.700591, -0.974228),
                                     (1.938641, 1.019462, 0.656661), (0.283545, -0.8962730000000001, -0.177117),
                                     (1.154454, -0.932148, 1.085425))})
        self.assertEqual(xyzs[0], expected_xyzs[0])
        self.assertEqual(xyzs[1], expected_xyzs[1])

    def test_find_equivalent_atoms(self):
        """Test the find_equivalent_atoms() function."""
        r_eq_atoms, p_eq_atoms = nmd.find_equivalent_atoms(reaction=self.rxn_1, reactant_only=False)
        self.assertEqual(r_eq_atoms, [[1, 2, 3, 4]])
        self.assertEqual(p_eq_atoms, [[2, 3, 4], [6, 1]])

        r_eq_atoms, p_eq_atoms = nmd.find_equivalent_atoms(reaction=self.rxn_1, reactant_only=True)
        self.assertEqual(r_eq_atoms, [[1, 2, 3, 4]])
        self.assertEqual(p_eq_atoms, [])

        r_eq_atoms, _ = nmd.find_equivalent_atoms(reaction=self.rxn_2, reactant_only=True)
        self.assertEqual(r_eq_atoms, [[3, 4], [5, 6, 7, 8]])

    def test_identify_equivalent_atoms_in_molecule(self):
        """Test the identify_equivalent_atoms_in_molecule() function."""
        equivalent_atoms = nmd.identify_equivalent_atoms_in_molecule(molecule=Molecule(smiles='C'))
        self.assertEqual(equivalent_atoms, [[1, 2, 3, 4]])

        equivalent_atoms = nmd.identify_equivalent_atoms_in_molecule(molecule=Molecule(smiles='OO'))
        self.assertEqual(equivalent_atoms, [[0, 1], [2, 3]])

        equivalent_atoms = nmd.identify_equivalent_atoms_in_molecule(molecule=Molecule(smiles='CC'))
        self.assertEqual(equivalent_atoms, [[0, 1], [2, 3, 4, 5, 6, 7]])

        equivalent_atoms = nmd.identify_equivalent_atoms_in_molecule(molecule=Molecule(smiles='NCC(CN)CN'))
        self.assertEqual(equivalent_atoms, [[0, 4, 6], [3, 5], [7, 8, 14, 15, 18, 19], [9, 10], [12, 13, 16, 17]])

        equivalent_atoms = nmd.identify_equivalent_atoms_in_molecule(molecule=Molecule(smiles='OO'), atom_map=[3, 2, 1, 0])
        self.assertEqual(equivalent_atoms, [[3, 2], [1, 0]])

    def test_fingerprint_atom(self):
        """Test the fingerprint_atom() function."""
        fp = nmd.fingerprint_atom(atom_index=0, molecule=Molecule(smiles='[H]'))
        self.assertEqual(fp, [1])

        fp = nmd.fingerprint_atom(atom_index=0, molecule=Molecule(smiles='[H][H]'))
        self.assertEqual(fp, [1, [1]])

        fp = nmd.fingerprint_atom(atom_index=0, molecule=ARCSpecies(label='CH4', smiles='C', xyz=self.ch4_xyz).mol)
        self.assertEqual(fp, [6, [1, 1, 1, 1]])

        fp = nmd.fingerprint_atom(atom_index=1, molecule=ARCSpecies(label='CH4', smiles='C', xyz=self.ch4_xyz).mol)
        self.assertEqual(fp, [1, [6, [1, 1, 1]]])

        fp = nmd.fingerprint_atom(atom_index=0, molecule=ARCSpecies(label='H2O2', smiles='OO', xyz=self.h2o2_xyz).mol)
        self.assertEqual(fp, [8, [8, [1], 1]])

        fp = nmd.fingerprint_atom(atom_index=0, molecule=Molecule(smiles='CCCCCCCCCCCC'))
        self.assertEqual(fp, [6, [6, [6, [1, 1, 6], 1, 1], 1, 1, 1]])

        fp = nmd.fingerprint_atom(atom_index=1, molecule=Molecule(smiles='CCCCCCCCCCCC'))
        self.assertEqual(fp, [6, [6, [1, 1, 1]], [6, [6, [1, 1, 6], 1, 1], 1, 1]])

        fp = nmd.fingerprint_atom(atom_index=15, molecule=Molecule(smiles='CCCCCCCCCCCC'))
        self.assertEqual(fp, [1, [6, [6, [1, 1, 1]], [6, [1, 1, 6], 1]]])


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
