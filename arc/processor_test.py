#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.processor module
"""

import unittest

import arc.processor as processor
from arc.species.species import ARCSpecies


class TestProcessor(unittest.TestCase):
    """
    Contains unit tests for the Processor class
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.ch4 = ARCSpecies(label='CH4', smiles='C')
        cls.nh3 = ARCSpecies(label='NH3', smiles='N')
        cls.h = ARCSpecies(label='H', smiles='[H]')
        cls.ch4_bde_1_2_a = ARCSpecies(label='CH4_BDE_1_2_A', smiles='[CH3]')
        cls.ch4.e0, cls.h.e0, cls.ch4_bde_1_2_a.e0 = 10, 25, 35
        cls.ch4.bdes = [(1, 2)]

    def test_process_bdes(self):
        """Test the process_bdes method"""
        bde_report = processor.process_bdes(label='CH4',
                                            species_dict={'CH4': self.ch4,
                                                          'NH3': self.nh3,
                                                          'H': self.h,
                                                          'CH4_BDE_1_2_A': self.ch4_bde_1_2_a})
        self.assertEqual(bde_report, {(1, 2): 50})


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
