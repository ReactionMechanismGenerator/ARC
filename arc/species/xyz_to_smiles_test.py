#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.species.xyz_to_smiles module
"""

import unittest
from arc.species.xyz_to_smiles import xyz_to_smiles

class TestXYZToSMILES(unittest.TestCase):
    """
    Contains unit tests for the xyz_to_smiles function
    """

    def test_water(self):
        """Test water perception"""
        xyz = {'symbols': ('O', 'H', 'H'),
               'coords': ((0.0000, 0.0000, 0.1173),
                          (0.0000, 0.7572, -0.4692),
                          (0.0000, -0.7572, -0.4692))}
        smiles = xyz_to_smiles(xyz)
        self.assertIn('O', smiles)

    def test_methane(self):
        """Test methane perception"""
        xyz = {'symbols': ('C', 'H', 'H', 'H', 'H'),
               'coords': ((0.0000, 0.0000, 0.0000),
                          (0.6291, 0.6291, 0.6291),
                          (-0.6291, -0.6291, 0.6291),
                          (0.6291, -0.6291, -0.6291),
                          (-0.6291, 0.6291, -0.6291))}
        smiles = xyz_to_smiles(xyz)
        self.assertIn('C', smiles)

    def test_ethylene(self):
        """Test ethylene perception"""
        xyz = {'symbols': ('C', 'C', 'H', 'H', 'H', 'H'),
               'coords': ((0.0000, 0.0000, 0.6650),
                          (0.0000, 0.0000, -0.6650),
                          (0.0000, 0.9229, 1.2327),
                          (0.0000, -0.9229, 1.2327),
                          (0.0000, 0.9229, -1.2327),
                          (0.0000, -0.9229, -1.2327))}
        smiles = xyz_to_smiles(xyz)
        self.assertIn('C=C', smiles)

    def test_charged_species(self):
        """Test OH- perception"""
        xyz = {'symbols': ('O', 'H'),
               'coords': ((0.0000, 0.0000, 0.0000),
                          (0.0000, 0.0000, 0.9600))}
        smiles = xyz_to_smiles(xyz, charge=-1)
        self.assertIn('[OH-]', smiles)

    def test_huckel(self):
        """Test with use_huckel=False"""
        xyz = {'symbols': ('O', 'H', 'H'),
               'coords': ((0.0000, 0.0000, 0.1173),
                          (0.0000, 0.7572, -0.4692),
                          (0.0000, -0.7572, -0.4692))}
        smiles = xyz_to_smiles(xyz, use_huckel=False)
        self.assertIn('O', smiles)

    def test_acetylene(self):
        """Test acetylene perception (triple bond)"""
        xyz = {'symbols': ('C', 'C', 'H', 'H'),
               'coords': ((0.0000, 0.0000, 0.6000),
                          (0.0000, 0.0000, -0.6000),
                          (0.0000, 0.0000, 1.6600),
                          (0.0000, 0.0000, -1.6600))}
        smiles = xyz_to_smiles(xyz)
        self.assertIn('C#C', smiles)

    def test_chiral_center(self):
        """Test chirality perception"""
        # (S)-1-fluoro-1-chloroethane
        xyz = {'symbols': ('C', 'C', 'F', 'Cl', 'H', 'H', 'H', 'H'),
               'coords': ((0.0000, 0.0000, 0.0000),
                          (1.5000, 0.0000, 0.0000),
                          (-0.5000, 1.2000, 0.0000),
                          (-0.5000, -0.6000, 1.4000),
                          (-0.4000, -0.6000, -0.8000),
                          (1.9000, 0.5000, 0.8000),
                          (1.9000, 0.5000, -0.8000),
                          (1.9000, -1.0000, 0.0000))}
        smiles = xyz_to_smiles(xyz, embed_chiral=True)
        self.assertTrue(any('@' in s for s in smiles))

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
