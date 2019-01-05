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
        """Test parsing of QChem frequencies"""
        no3_path = os.path.join(arc_path, 'arc', 'testing', 'NO3_freq_QChem_fails_on_cclib.out')
        c2h6_path = os.path.join(arc_path, 'arc', 'testing', 'C2H6_freq_Qchem.out')
        no3_freqs = parser.parse_frequencies(path=no3_path, software='QChem')
        c2h6_freqs = parser.parse_frequencies(path=c2h6_path, software='QChem')
        self.assertTrue(np.array_equal(no3_freqs,
                                       np.array([-390.08, -389.96, 822.75, 1113.23, 1115.24, 1195.35], np.float64)))
        self.assertTrue(np.array_equal(c2h6_freqs,
                                       np.array([352.37, 847.01, 861.68, 1023.23, 1232.66, 1235.04, 1425.48, 1455.31,
                                                 1513.67, 1518.02, 1526.18, 1526.56, 3049.78, 3053.32, 3111.61, 3114.2,
                                                 3134.14, 3136.8], np.float64)))

################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
