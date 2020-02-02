#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for TS guess generation methods
"""

import unittest

from arc.ts.train import read_data


class TestTrain(unittest.TestCase):
    """
    Contains unit tests for training TS search by heuristics
    """

    @classmethod
    def setUpClass(cls):
        """
        A function run ONCE before all unit tests in this class.
        """
        cls.maxDiff = None

    def test_read_data(self):
        """Test reading the data"""
        results = read_data('H_Abstraction')
        for entry in results:
            self.assertEqual(entry['family'], 'H_Abstraction')
            self.assertIsInstance(entry['labels'], list)
            self.assertIsInstance(entry['stretch'], list)
            self.assertIsInstance(entry['xyz'], dict)
            if entry['name'] == 'O+HCl':
                self.assertEqual(entry['labels'], [0, 1, 2])
                self.assertEqual(entry['level'], 'QCISD/MG3')
                self.assertEqual(entry['reaction'], 'O+HCl=OH+Cl')
                self.assertEqual(entry['source'], 'HTBH38/08 (https://comp.chem.umn.edu/db/dbs/htbh38.html)')
                self.assertEqual(entry['stretch'], [1.1482, 1.2345])
                self.assertEqual(entry['xyz'], {'coords': ((0.01882, -0.8173, 0.0),
                                                           (-0.47049, 0.56948, 0.0),
                                                           (0.01882, 1.66558, 0.0)),
                                                'isotopes': (35, 1, 16),
                                                'symbols': ('Cl', 'H', 'O')})


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
