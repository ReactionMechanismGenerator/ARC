#!/usr/bin/env python3
# encoding: utf-8

import unittest

from arc.molecule.molecule import Molecule
from arc.molecule.kekulize import AromaticRing


class KekulizeTest(unittest.TestCase):

    def setUp(self):
        """To be run before each test."""
        molecule = Molecule().from_adjacency_list("""
1  C u0 p0 c0 {2,B} {6,B} {7,S}
2  C u0 p0 c0 {1,B} {3,B} {8,S}
3  C u0 p0 c0 {2,B} {4,B} {9,S}
4  C u0 p0 c0 {3,B} {5,B} {10,S}
5  C u0 p0 c0 {4,B} {6,B} {11,S}
6  C u0 p0 c0 {1,B} {5,B} {12,S}
7  H u0 p0 c0 {1,S}
8  H u0 p0 c0 {2,S}
9  H u0 p0 c0 {3,S}
10 H u0 p0 c0 {4,S}
11 H u0 p0 c0 {5,S}
12 H u0 p0 c0 {6,S}
""")
        bonds = set()
        for atom in molecule.atoms:
            bonds.update(list(atom.bonds.values()))

        ring_atoms, ring_bonds = molecule.get_aromatic_rings()

        self.aromatic_ring = AromaticRing(ring_atoms[0], set(ring_bonds[0]), bonds - set(ring_bonds[0]))

    def test_aromatic_ring(self):
        """Test that the AromaticRing class works properly for kekulization."""
        self.aromatic_ring.update()

        self.assertEqual(self.aromatic_ring.endo_dof, 6)
        self.assertEqual(self.aromatic_ring.exo_dof, 0)

        result = self.aromatic_ring.kekulize()

        self.assertTrue(result)

    def test_aromatic_bond(self):
        """Test that the AromaticBond class works properly for kekulization."""
        self.aromatic_ring.process_bonds()
        resolved, unresolved = self.aromatic_ring.resolved, self.aromatic_ring.unresolved

        self.assertEqual(len(resolved), 0)
        self.assertEqual(len(unresolved), 6)

        for bond in unresolved:
            bond.update()
            self.assertEqual(bond.endo_dof, 2)
            self.assertEqual(bond.exo_dof, 0)
            self.assertTrue(bond.double_possible)
            self.assertFalse(bond.double_required)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
