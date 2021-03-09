#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.utils.scale module
"""

import os
import shutil
import unittest

from arc.common import almost_equal_coords_lists, arc_path
from arc.utils.scale import (calculate_truhlar_scaling_factors,
                             get_species_list,
                             rename_level,
                             summarize_results,
                             )


class TestScale(unittest.TestCase):
    """
    Contains unit tests for the arc.utils.scale module
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.zpe_dicts = [{'C2H2': 16 * 4184,
                          'CH4': 27 * 4184,
                          'CO2': 7 * 4184,
                          'CO': 3 * 4184,
                          'F2': 1 * 4184,
                          'CH2O': 16 * 4184,
                          'H2O': 13 * 4184,
                          'H2': 6 * 4184,
                          'HCN': 10 * 4184,
                          'HF': 5 * 4184,
                          'N2O': 6 * 4184,
                          'N2': 3 * 4184,
                          'NH3': 21 * 4184,
                          'OH': 5 * 4184,
                          'Cl2': 0 * 4184},
                         {'C2H2': 15 * 4184,
                          'CH4': 27 * 4184,
                          'CO2': None,
                          'CO': 3 * 4184,
                          'F2': 1 * 4184,
                          'CH2O': 16 * 4184,
                          'H2O': 13 * 4184,
                          'H2': 6 * 4184,
                          'HCN': 10 * 4184,
                          'HF': 5 * 4184,
                          'N2O': 6 * 4184,
                          'N2': 3 * 4184,
                          'NH3': 21 * 4184,
                          'OH': 5 * 4184}]

    def test_calculate_truhlar_scaling_factors(self):
        """Test the scale calculate_truhlar_scaling_factors() function"""
        lambda_zpe = calculate_truhlar_scaling_factors(zpe_dict=self.zpe_dicts[0], level='test')
        self.assertAlmostEqual(lambda_zpe, 1.0241661, 5)

    def test_summarize_results(self):
        """Test the scale summarize_results() function"""
        lambda_zpes = [0.95, 0.98]
        levels_of_theory = ['level1', 'level2']
        times = ['3', '5']
        overall_time = '8.5'
        base_path = os.path.join(arc_path, 'Projects', 'scaling_factors_arc_testing_delete_after_usage')

        summarize_results(lambda_zpes=lambda_zpes,
                          levels=levels_of_theory,
                          zpe_dicts=self.zpe_dicts,
                          times=times,
                          overall_time=overall_time,
                          base_path=base_path)

        info_file_path = os.path.join(base_path, 'scaling_factors_0.info')
        self.assertTrue(os.path.isfile(info_file_path))
        with open(info_file_path, 'r') as f:
            lines = f.readlines()
        self.assertIn('CITATIONS:\n', lines)
        self.assertIn('Level of theory: level1\n', lines)
        self.assertIn('Level of theory: level2\n', lines)
        self.assertIn('The following species from the standard set did not converge at this level:\n', lines)
        self.assertIn(" ['CO2']\n", lines)
        self.assertIn('Scale Factor for Fundamental Frequencies = 0.955\n', lines)
        self.assertIn('Scale Factor for Harmonic Frequencies    = 0.994\n', lines)
        self.assertIn('Scaling factors calculation for 2 levels of theory completed (elapsed time: 8.5).\n', lines)
        self.assertIn('You may copy-paste the following harmonic frequencies scaling factor/s to Arkane\n', lines)
        self.assertIn("                 'level1': 0.963,  # [4]\n", lines)

    def test_get_species_list(self):
        """Test the scale get_species_list() function"""
        species_list = get_species_list()
        self.assertEqual(len(species_list), 15)
        labels = ['C2H2', 'CH4', 'CO2', 'CO', 'F2', 'CH2O', 'H2O', 'H2', 'HCN', 'HF', 'N2O', 'N2', 'NH3', 'OH',  'Cl2']
        for spc in species_list:
            self.assertIn(spc.label, labels)
            self.assertTrue(spc.initial_xyz)
        c2h2_xyz = {'symbols': ('C', 'C', 'H', 'H'), 'isotopes': (12, 12, 1, 1),
                    'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 1.203142), (0.0, -0.0, 2.265747), (-0.0, -0.0, -1.062605))}
        self.assertTrue(almost_equal_coords_lists(species_list[0].initial_xyz, c2h2_xyz))

    def test_rename_level(self):
        """Test the scale rename_level() function"""
        level1 = 'b3lyp/6-311+G**'
        level2 = 'wb97xd/6-311+G(2d,2p)'
        renamed_level1 = rename_level(level1)
        renamed_level2 = rename_level(level2)
        self.assertEqual(renamed_level1, 'b3lyp_6-311pGss')
        self.assertEqual(renamed_level2, 'wb97xd_6-311pGb2d,2pb')

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all directories created during these unit tests
        """
        path = os.path.join(arc_path, 'Projects', 'scaling_factors_arc_testing_delete_after_usage')
        shutil.rmtree(path, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
