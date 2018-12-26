#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests of the arc.species module
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest

from rmgpy.species import Species
from rmgpy.molecule.molecule import Molecule

from arc.species import ARCSpecies

################################################################################


class TestARCSpecies(unittest.TestCase):
    """
    Contains unit tests for the ARCSpecies class
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        # Method 1: RMG Species object (here by SMILES)
        spc1 = Species(molecule=[Molecule().fromSMILES("C=C[O]")])  # delocalized radical + amine
        spc1.label = 'vinoxy'
        cls.spc1 = ARCSpecies(rmg_species=spc1)

        # Method 2: ARCSpecies object by XYZ (also give SMILES for thermo BAC)
        oh_xyz = """O       0.00000000    0.00000000   -0.12002167
        H       0.00000000    0.00000000    0.85098324"""
        cls.spc2 = ARCSpecies(label='OH', xyz=oh_xyz, smiles='[OH]', multiplicity=2, charge=0)

        # Method 3: ARCSpecies object by SMILES
        cls.spc3 = ARCSpecies(label='methylamine', smiles='CN', multiplicity=1, charge=0)

        # Method 4: ARCSpecies object by RMG Molecule object
        mol4 = Molecule().fromSMILES("C=CC")
        cls.spc4 = ARCSpecies(label='propene', mol=mol4, multiplicity=1, charge=0)

        # Method 5: ARCSpecies by AdjacencyList (to generate AdjLists, see https://rmg.mit.edu/molecule_search)
        n2h4_adj = """1 N u0 p1 c0 {2,S} {3,S} {4,S}
        2 N u0 p1 c0 {1,S} {5,S} {6,S}
        3 H u0 p0 c0 {1,S}
        4 H u0 p0 c0 {1,S}
        5 H u0 p0 c0 {2,S}
        6 H u0 p0 c0 {2,S}"""
        cls.spc5 = ARCSpecies(label='N2H4', adjlist=n2h4_adj, multiplicity=1, charge=0)

    def test_conformers(self):
        """Test conformer generation"""
        self.spc1.generate_conformers()  # vinoxy has two res. structures, each is assgined two conformers (rdkit/ob)
        self.assertEqual(len(self.spc1.conformers), 4)
        self.assertEqual(len(self.spc1.conformers), len(self.spc1.conformer_energies))


################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
