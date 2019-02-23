#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the parser functions
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest
import os
import numpy as np

from arc.settings import arc_path
from arc import parser

################################################################################


class TestParser(unittest.TestCase):
    """
    Contains unit tests for the parser functions
    """

    def test_parse_frequencies(self):
        """Test frequency parsing"""
        no3_path = os.path.join(arc_path, 'arc', 'testing', 'NO3_freq_QChem_fails_on_cclib.out')
        c2h6_path = os.path.join(arc_path, 'arc', 'testing', 'C2H6_freq_Qchem.out')
        so2oo_path = os.path.join(arc_path, 'arc', 'testing', 'SO2OO_CBS-QB3.log')
        no3_freqs = parser.parse_frequencies(path=no3_path, software='QChem')
        c2h6_freqs = parser.parse_frequencies(path=c2h6_path, software='QChem')
        so2oo_freqs = parser.parse_frequencies(path=so2oo_path, software='Gaussian')
        self.assertTrue(np.array_equal(no3_freqs,
                                       np.array([-390.08, -389.96, 822.75, 1113.23, 1115.24, 1195.35], np.float64)))
        self.assertTrue(np.array_equal(c2h6_freqs,
                                       np.array([352.37, 847.01, 861.68, 1023.23, 1232.66, 1235.04, 1425.48, 1455.31,
                                                 1513.67, 1518.02, 1526.18, 1526.56, 3049.78, 3053.32, 3111.61, 3114.2,
                                                 3134.14, 3136.8], np.float64)))
        self.assertTrue(np.array_equal(so2oo_freqs,
                                       np.array([302.51, 468.1488, 469.024, 484.198, 641.0067, 658.6316,
                                                 902.2888, 1236.9268, 1419.0826], np.float64)))

    def test_parse_xyz_from_file(self):
        """Test parsing xyz from a file"""
        path1 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'CH3C(O)O.gjf')
        path2 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'CH3C(O)O.xyz')
        path3 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'AIBN.gjf')
        path4 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'molpro.in')
        path5 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'qchem.in')
        path6 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'qchem_output.out')

        xyz1 = parser.parse_xyz_from_file(path1)
        xyz2 = parser.parse_xyz_from_file(path2)
        xyz3 = parser.parse_xyz_from_file(path3)
        xyz4 = parser.parse_xyz_from_file(path4)
        xyz5 = parser.parse_xyz_from_file(path5)
        xyz6 = parser.parse_xyz_from_file(path6)

        self.assertEqual(xyz1.rstrip(), xyz2.rstrip())
        self.assertTrue('C       1.40511900    0.21728200    0.07675200' in xyz1)
        self.assertTrue('O      -0.79314200    1.04818800    0.18134200' in xyz1)
        self.assertTrue('H      -0.43701200   -1.34990600    0.92900600' in xyz2)
        self.assertTrue(' C                  2.12217963   -0.66843078    1.04808732' in xyz3)
        self.assertTrue(' N                  2.41731872   -1.07916417    2.08039935' in xyz3)
        self.assertTrue(' S         -0.4204682221       -0.3909949822        0.0245352116' in xyz4)
        self.assertTrue(' N                 -1.99742564    0.38106573    0.09139807' in xyz5)
        self.assertTrue('N      -1.17538406    0.34366165    0.03265021' in xyz6)

    def test_parse_t1(self):
        """Test T1 diagnostic parsing"""
        path = os.path.join(arc_path, 'arc', 'testing', 'mehylamine_CCSD(T).out')
        t1 = parser.parse_t1(path)
        self.assertEqual(t1, 0.0086766)

################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
