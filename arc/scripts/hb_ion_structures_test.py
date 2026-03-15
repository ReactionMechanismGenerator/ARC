#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the hydrogen bond structures around an ion.
"""

import copy
import unittest

import arc.scripts.hb_ion_structures as structures


class TestHBStructures(unittest.TestCase):
    """
    Contains unit tests for the hydrogen bond structures around an ion.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.oh_xyz = {'symbols': ('O', 'H'), 'isotopes': (16, 1),
                      'coords': ((0.0, 0.0, 0.106631), (0.0, 0.0, -0.85305))}
        cls.oh_zmat = {'symbols': ('O', 'H'),
                       'coords': ((None, None, None), ('R_1_0', None, None)),
                       'vars': {'R_1_0': 0.9596809607593241},
                       'map': {0: 0, 1: 1}}

    def test_check_args(self):
        """
        Test the check_args function.
        """
        with self.assertRaises(ValueError):  # wrong ion
            structures.check_args(hydration_level=2, ion='Li+', e_threshold=1, min_hb_per_ion=1, t_list=[300])
        with self.assertRaises(ValueError):  # hydration_level < 1
            structures.check_args(hydration_level=0, ion='OH-', e_threshold=1, min_hb_per_ion=1, t_list=[300])
        with self.assertRaises(ValueError):  # hydration_level > 8
            structures.check_args(hydration_level=10, ion='OH-', e_threshold=1, min_hb_per_ion=1, t_list=[300])
        with self.assertRaises(ValueError):  # e_threshold < 1
            structures.check_args(hydration_level=2, ion='OH-', e_threshold=0, min_hb_per_ion=1, t_list=[300])
        with self.assertRaises(ValueError):  # min_hb_per_ion > hydration_level
            structures.check_args(hydration_level=2, ion='OH-', e_threshold=1, min_hb_per_ion=3, t_list=[300])
        with self.assertRaises(ValueError):  # min_hb_per_ion < 1
            structures.check_args(hydration_level=2, ion='OH-', e_threshold=1, min_hb_per_ion=0, t_list=[300])
        with self.assertRaises(ValueError):  # t_list not a list
            structures.check_args(hydration_level=2, ion='OH-', e_threshold=1, min_hb_per_ion=1, t_list=300)
        with self.assertRaises(ValueError):  # t_list empty
            structures.check_args(hydration_level=2, ion='OH-', e_threshold=1, min_hb_per_ion=1, t_list=[])
        with self.assertRaises(ValueError):  # t_list has wrong T
            structures.check_args(hydration_level=2, ion='OH-', e_threshold=1, min_hb_per_ion=1, t_list=[-1])

    def test_get_ion_zmat(self):
        """
        Test the get_ion_zmat function.
        """
        zmat = structures.get_ion_zmat('OH-')
        self.assertEqual(self.oh_zmat, zmat)


    def test_get_last_added_water_o_index(self):
        """
        Test the get_last_added_water_o_index function.
        """
        # dict xyz with OH and one water molecule in atom order OHHOH:
        xyz = {'symbols': ('O', 'H', 'H', 'O', 'H'), 'isotopes': (16, 1, 1, 16, 1),
               'coords': ((0.0, 0.0, 0.106631), (0.0, 0.0, -0.85305), (0.0, 0.0, 1.106631),
                          (0.0, 0.0, 2.106631), (0.0, 0.0, 3.106631))}
        last_o_index = structures.get_last_added_water_o_index(xyz, self.oh_zmat)
        self.assertEqual(last_o_index, 3)


    def test_add_water_molecule(self):
        """
        Test the add_water_molecule function.
        """
        new_zmat_1 = structures.add_water_molecule(zmat=copy.deepcopy(self.oh_zmat), ion_zmat=self.oh_zmat, dihedarl=0, beta=110)
        expected_zmat_1 = {'symbols': ('O', 'H', 'H', 'O', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_3', 'A_4_3_2', 'D_4_3_2_1')),
                           'vars': {'R_1_0': 0.9596809607593241, 'R_2_0': 1.5, 'A_2_0_1': 110, 'R_3_0': 2.4699999999999998,
                                    'A_3_0_1': 110, 'D_3_0_1_2': 0, 'R_4_3': 0.97, 'A_4_3_2': 109.5, 'D_4_3_2_1': 90},
                           'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}}
        self.assertEqual(new_zmat_1, expected_zmat_1)

        new_zmat_2 = structures.add_water_molecule(zmat=expected_zmat_1, ion_zmat=self.oh_zmat, dihedarl=180, beta=110)
        expected_zmat_2 = {'symbols': ('O', 'H', 'H', 'O', 'H', 'H', 'O', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_3', 'A_4_3_2', 'D_4_3_2_1'),
                                      ('R_5_0', 'A_5_0_1', 'D_5_0_1_3'), ('R_6_0', 'A_6_0_1', 'D_6_0_1_3'),
                                      ('R_7_6', 'A_7_6_5', 'D_7_6_5_1')),
                           'vars': {'R_1_0': 0.9596809607593241, 'R_2_0': 1.5, 'A_2_0_1': 110, 'R_3_0': 2.4699999999999998,
                                    'A_3_0_1': 110, 'D_3_0_1_2': 0, 'R_4_3': 0.97, 'A_4_3_2': 109.5, 'D_4_3_2_1': 90,
                                    'R_5_0': 1.5, 'A_5_0_1': 110, 'R_6_0': 2.4699999999999998, 'A_6_0_1': 110, 'D_6_0_1_3': 180,
                                    'R_7_6': 0.97, 'A_7_6_5': 109.5, 'D_7_6_5_1': 90, 'D_5_0_1_3': 180},
                           'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}}
        self.assertEqual(new_zmat_2, expected_zmat_2)

    def test_get_skeletal_structure(self):
        """
        Test the get_skeletal_structure function.
        """
        zmat_0 = structures.get_skeletal_structure(ion_zmat=self.oh_zmat, num_w_mols=0)
        expected_zmat_0 = {'symbols': ('O', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None)),
                           'vars': {'R_1_0': 0.9596809607593241}, 'map': {0: 0, 1: 1}}
        self.assertEqual(zmat_0, expected_zmat_0)

        zmat_1 = structures.get_skeletal_structure(ion_zmat=self.oh_zmat, num_w_mols=1)
        expected_zmat_1 = {'symbols': ('O', 'H', 'H', 'O', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_3', 'A_4_3_2', 'D_4_3_2_1')),
                           'vars': {'R_1_0': 0.9596809607593241, 'R_2_0': 1.5, 'A_2_0_1': 110, 'R_3_0': 2.4699999999999998,
                                    'A_3_0_1': 110, 'D_3_0_1_2': 0.0, 'R_4_3': 0.97, 'A_4_3_2': 109.5, 'D_4_3_2_1': 90},
                           'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}}
        self.assertEqual(zmat_1, expected_zmat_1)

        zmat_2 = structures.get_skeletal_structure(ion_zmat=self.oh_zmat, num_w_mols=2)
        expected_zmat_2 = {'symbols': ('O', 'H', 'H', 'O', 'H', 'H', 'O', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_3', 'A_4_3_2', 'D_4_3_2_1'),
                                      ('R_5_0', 'A_5_0_1', 'D_5_0_1_3'), ('R_6_0', 'A_6_0_1', 'D_6_0_1_3'),
                                      ('R_7_6', 'A_7_6_5', 'D_7_6_5_1')),
                           'vars': {'R_1_0': 0.9596809607593241, 'R_2_0': 1.5, 'A_2_0_1': 110, 'R_3_0': 2.4699999999999998,
                                    'A_3_0_1': 110, 'D_3_0_1_2': 0.0, 'R_4_3': 0.97, 'A_4_3_2': 109.5, 'D_4_3_2_1': 90,
                                    'R_5_0': 1.5, 'A_5_0_1': 110, 'R_6_0': 2.4699999999999998, 'A_6_0_1': 110, 'D_6_0_1_3': 180.0,
                                    'R_7_6': 0.97, 'A_7_6_5': 109.5, 'D_7_6_5_1': 90, 'D_5_0_1_3': 180.0},
                           'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}}
        self.assertEqual(zmat_2, expected_zmat_2)

        zmat_3 = structures.get_skeletal_structure(ion_zmat=self.oh_zmat, num_w_mols=3)
        expected_zmat_3 = {'symbols': ('O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_3', 'A_4_3_2', 'D_4_3_2_1'),
                                      ('R_5_0', 'A_5_0_1', 'D_5_0_1_3'), ('R_6_0', 'A_6_0_1', 'D_6_0_1_3'),
                                      ('R_7_6', 'A_7_6_5', 'D_7_6_5_1'), ('R_8_0', 'A_8_0_1', 'D_8_0_1_3'),
                                      ('R_9_0', 'A_9_0_1', 'D_9_0_1_3'), ('R_10_9', 'A_10_9_8', 'D_10_9_8_1')),
                           'vars': {'R_1_0': 0.9596809607593241, 'R_2_0': 1.5, 'A_2_0_1': 110, 'R_3_0': 2.4699999999999998,
                                    'A_3_0_1': 110, 'D_3_0_1_2': 0.0, 'R_4_3': 0.97, 'A_4_3_2': 109.5, 'D_4_3_2_1': 90,
                                    'R_5_0': 1.5, 'A_5_0_1': 110, 'R_6_0': 2.4699999999999998, 'A_6_0_1': 110, 'D_6_0_1_3': 120.0,
                                    'R_7_6': 0.97, 'A_7_6_5': 109.5, 'D_7_6_5_1': 90, 'D_5_0_1_3': 120.0, 'R_8_0': 1.5, 'A_8_0_1': 110,
                                    'R_9_0': 2.4699999999999998, 'A_9_0_1': 110, 'D_9_0_1_3': 240.0, 'R_10_9': 0.97, 'A_10_9_8': 109.5,
                                    'D_10_9_8_1': 90, 'D_8_0_1_3': 240.0},
                           'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}}
        self.assertEqual(zmat_3, expected_zmat_3)

        zmat_4 = structures.get_skeletal_structure(ion_zmat=self.oh_zmat, num_w_mols=4)
        expected_zmat_4 = {'symbols': ('O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_3', 'A_4_3_2', 'D_4_3_2_1'),
                                      ('R_5_0', 'A_5_0_1', 'D_5_0_1_3'), ('R_6_0', 'A_6_0_1', 'D_6_0_1_3'),
                                      ('R_7_6', 'A_7_6_5', 'D_7_6_5_1'), ('R_8_0', 'A_8_0_1', 'D_8_0_1_3'),
                                      ('R_9_0', 'A_9_0_1', 'D_9_0_1_3'), ('R_10_9', 'A_10_9_8', 'D_10_9_8_1'),
                                      ('R_11_0', 'A_11_0_1', 'D_11_0_1_3'), ('R_12_0', 'A_12_0_1', 'D_12_0_1_3'),
                                      ('R_13_12', 'A_13_12_11', 'D_13_12_11_1')),
                           'vars': {'R_1_0': 0.9596809607593241, 'R_2_0': 1.5, 'A_2_0_1': 110, 'R_3_0': 2.4699999999999998,
                                    'A_3_0_1': 110, 'D_3_0_1_2': 0.0, 'R_4_3': 0.97, 'A_4_3_2': 109.5, 'D_4_3_2_1': 90,
                                    'R_5_0': 1.5, 'A_5_0_1': 110, 'R_6_0': 2.4699999999999998, 'A_6_0_1': 110, 'D_6_0_1_3': 90.0,
                                    'R_7_6': 0.97, 'A_7_6_5': 109.5, 'D_7_6_5_1': 90, 'D_5_0_1_3': 90.0, 'R_8_0': 1.5, 'A_8_0_1': 110,
                                    'R_9_0': 2.4699999999999998, 'A_9_0_1': 110, 'D_9_0_1_3': 180.0, 'R_10_9': 0.97, 'A_10_9_8': 109.5,
                                    'D_10_9_8_1': 90, 'D_8_0_1_3': 180.0, 'R_11_0': 1.5, 'A_11_0_1': 110, 'R_12_0': 2.4699999999999998,
                                    'A_12_0_1': 110, 'D_12_0_1_3': 270.0, 'R_13_12': 0.97, 'A_13_12_11': 109.5, 'D_13_12_11_1': 90, 'D_11_0_1_3': 270.0},
                           'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13}}
        self.assertEqual(zmat_4, expected_zmat_4)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
