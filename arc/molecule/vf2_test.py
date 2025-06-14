#!/usr/bin/env python3
# encoding: utf-8


import unittest

from numpy import testing

from arc.molecule.graph import get_vertex_connectivity_value
from arc.molecule.molecule import Molecule
from arc.molecule.vf2 import VF2


class TestVF2(unittest.TestCase):
    """
    Contains unit tests of the methods for computing symmetry numbers for a
    given Molecule object.
    """

    def setUp(self):
        self.vf2 = VF2()
        self.mol = Molecule().from_smiles("CC(=O)C[CH2]")
        self.mol2 = self.mol.copy(deep=True)

    def test_import_graph(self):
        """Test that we can add graphs to the object and that they are sorted"""

        self.mol.sort_vertices()

        ordered_original_connectivity_order = [get_vertex_connectivity_value(atom) for atom in self.mol.atoms]

        self.vf2.graphA = self.mol
        self.vf2.graphB = self.mol2

        final_connectivity_order = [get_vertex_connectivity_value(atom) for atom in self.vf2.graphA.atoms]
        final_connectivity_order2 = [get_vertex_connectivity_value(atom) for atom in self.vf2.graphB.atoms]

        testing.assert_array_equal(final_connectivity_order, final_connectivity_order2)
        testing.assert_array_equal(final_connectivity_order, ordered_original_connectivity_order)

    def test_feasible(self):
        """
        Test that feasibility returns correct values on highly functional molecule
        
        `feasible` method isn't perfect in assigning values, but it should do a good
        job on highly functional values
        """

        self.vf2.graphA = self.mol
        self.vf2.graphB = self.mol2

        for atom1 in self.vf2.graphA.atoms:
            for atom2 in self.vf2.graphB.atoms:
                # same connectivity values should result in `feasible` being true
                if get_vertex_connectivity_value(atom1) == get_vertex_connectivity_value(atom2):
                    self.assertTrue(self.vf2.feasible(atom1, atom2))
                else:  # different connectivity values should return false
                    self.assertFalse(self.vf2.feasible(atom1, atom2))

    def test_clear_mapping(self):
        """Test that vertex mapping is cleared after isomorphism."""
        self.vf2.is_isomorphic(self.mol, self.mol2, dict())

        for atom in self.mol.atoms:
            self.assertIsNone(atom.mapping)
            self.assertFalse(atom.terminal)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
