#!/usr/bin/env python3
# encoding: utf-8

import unittest

from arc.molecule.filtration import (get_octet_deviation_list, get_octet_deviation, filter_structures,
                                      charge_filtration, get_charge_span_list, aromaticity_filtration)
from arc.molecule.molecule import Molecule
from arc.molecule.resonance import generate_resonance_structures, analyze_molecule


class FiltrationTest(unittest.TestCase):

    def test_basic_filtration(self):
        """Test that structures with higher octet deviation get filtered out"""
        adj1 = """
        multiplicity 2
        1 N u0 p1 c0 {2,D} {3,S}
        2 O u0 p2 c0 {1,D}
        3 O u1 p2 c0 {1,S}
        """
        adj2 = """
        multiplicity 2
        1 N u1 p1 c0 {2,S} {3,S}
        2 O u0 p2 c+1 {1,S}
        3 O u0 p3 c-1 {1,S}
        """
        adj3 = """
        multiplicity 2
        1 O u1 p2 c0 {3,S}
        2 O u0 p3 c-1 {3,S}
        3 N u0 p1 c+1 {1,S} {2,S}
        """

        mol1 = Molecule().from_adjacency_list(adj1)
        mol2 = Molecule().from_adjacency_list(adj2)
        mol3 = Molecule().from_adjacency_list(adj3)

        mol_list = [mol1, mol2, mol3]
        octet_deviation_list = get_octet_deviation_list(mol_list)
        filtered_list = filter_structures(mol_list)

        self.assertEqual(octet_deviation_list, [1, 3, 3])
        self.assertEqual(len(filtered_list), 1)
        self.assertTrue(all([atom.charge == 0 for atom in filtered_list[0].vertices]))

    def test_penalty_for_o4tc(self):
        """Test that an O4tc atomtype with octet 8 gets penalized in the electronegativity heuristic"""
        adj = """
        1 S u0 p1 c0 {2,S} {3,T}
        2 O u0 p3 c-1 {1,S}
        3 O u0 p1 c+1 {1,T}
        """
        mol = Molecule().from_adjacency_list(adj)
        octet_deviation = get_octet_deviation(mol)
        self.assertEqual(octet_deviation, 0)
        self.assertEqual(mol.vertices[2].atomtype.label, 'O4tc')
        mol_list = generate_resonance_structures(mol)
        self.assertEqual(len(mol_list), 2)
        for mol in mol_list:
            if mol.reactive:
                for atom in mol.vertices:
                    self.assertEqual(atom.charge, 0)

    def test_penalty_birads_replacing_lone_pairs(self):
        """Test that birads on `S u2 p0` are penalized"""
        adj = """
        multiplicity 3
        1 S u2 p0 c0 {2,D} {3,D}
        2 O u0 p2 c0 {1,D}
        3 O u0 p2 c0 {1,D}
        """
        mol = Molecule().from_adjacency_list(adj)
        mol.update()
        mol_list = generate_resonance_structures(mol, keep_isomorphic=False, filter_structures=True)
        for mol in mol_list:
            if mol.reactive:
                for atom in mol.vertices:
                    if atom.is_sulfur():
                        self.assertNotEqual(atom.radical_electrons, 2)
        self.assertEqual(len(mol_list), 3)
        self.assertEqual(sum([1 for mol in mol_list if mol.reactive]), 2)

    def test_penalty_for_s_triple_s(self):
        """Test that an S#S substructure in a molecule gets penalized in the octet deviation score"""
        adj = """
        1  C u0 p0 c0 {3,S} {5,S} {6,S} {7,S}
        2  C u0 p0 c0 {4,S} {8,S} {9,S} {10,S}
        3  S u0 p0 c0 {1,S} {4,T} {11,D}
        4  S u0 p1 c0 {2,S} {3,T}
        5  H u0 p0 c0 {1,S}
        6  H u0 p0 c0 {1,S}
        7  H u0 p0 c0 {1,S}
        8  H u0 p0 c0 {2,S}
        9  H u0 p0 c0 {2,S}
        10 H u0 p0 c0 {2,S}
        11 O u0 p2 c0 {3,D}
        """
        mol = Molecule().from_adjacency_list(adj)
        octet_deviation = get_octet_deviation(mol)
        self.assertEqual(octet_deviation, 1.0)

    def test_radical_site(self):
        """Test that a charged molecule isn't filtered if it introduces new radical site"""
        adj1 = """
        multiplicity 2
        1 O u1 p2 c0 {3,S}
        2 O u0 p2 c0 {3,D}
        3 N u0 p1 c0 {1,S} {2,D}
        """
        adj2 = """
        multiplicity 2
        1 O u0 p3 c-1 {3,S}
        2 O u0 p2 c0 {3,D}
        3 N u1 p0 c+1 {1,S} {2,D}
        """
        adj3 = """
        multiplicity 2
        1 O u1 p2 c0 {3,S}
        2 O u0 p3 c-1 {3,S}
        3 N u0 p1 c+1 {1,S} {2,S}
        """

        mol_list = [Molecule().from_adjacency_list(adj1),
                    Molecule().from_adjacency_list(adj2),
                    Molecule().from_adjacency_list(adj3)]

        for mol in mol_list:
            mol.update()  # the charge_filtration uses the atom.sorting_label attribute

        filtered_list = charge_filtration(mol_list, get_charge_span_list(mol_list))
        self.assertEqual(len(filtered_list), 2)
        self.assertTrue(any([mol.get_charge_span() == 1 for mol in filtered_list]))
        for mol in filtered_list:
            if mol.get_charge_span() == 1:
                for atom in mol.vertices:
                    if atom.charge == -1:
                        self.assertTrue(atom.is_oxygen())
                    if atom.charge == 1:
                        self.assertTrue(atom.is_nitrogen())

    def test_electronegativity(self):
        """Test that structures with charge separation are only kept if they obey the electronegativity rule

        (If a structure must have charge separation, negative charges will be assigned to more electronegative atoms,
        whereas positive charges will be assigned to less electronegative atoms)

        In this test, only the three structures with no charge separation and the structure where both partial charges
        are on the nitrogen atoms should be kept."""
        adj1 = """
        multiplicity 2
        1 N u0 p1 c0 {2,S} {3,D}
        2 N u1 p1 c0 {1,S} {4,S}
        3 S u0 p1 c0 {1,D} {5,D}
        4 H u0 p0 c0 {2,S}
        5 O u0 p2 c0 {3,D}
        """
        adj2 = """
        multiplicity 2
        1 N u0 p1 c0 {2,D} {3,S}
        2 N u0 p1 c0 {1,D} {4,S}
        3 S u1 p1 c0 {1,S} {5,D}
        4 H u0 p0 c0 {2,S}
        5 O u0 p2 c0 {3,D}
        """
        adj3 = """
        multiplicity 2
        1 N u0 p1 c0 {2,D} {3,S}
        2 N u0 p1 c0 {1,D} {4,S}
        3 S u0 p2 c0 {1,S} {5,S}
        4 H u0 p0 c0 {2,S}
        5 O u1 p2 c0 {3,S}
        """
        adj4 = """
        multiplicity 2
        1 N u1 p0 c+1 {2,S} {3,D}
        2 N u0 p2 c-1 {1,S} {4,S}
        3 S u0 p1 c0 {1,D} {5,D}
        4 H u0 p0 c0 {2,S}
        5 O u0 p2 c0 {3,D}
        """
        adj5 = """
        multiplicity 2
        1 N u1 p0 c+1 {2,D} {3,S}
        2 N u0 p1 c0 {1,D} {4,S}
        3 S u0 p2 c-1 {1,S} {5,D}
        4 H u0 p0 c0 {2,S}
        5 O u0 p2 c0 {3,D}
        """
        adj6 = """
        multiplicity 2
        1 N u1 p1 c0 {2,S} {3,S}
        2 N u0 p2 c-1 {1,S} {4,S}
        3 S u0 p1 c+1 {1,S} {5,D}
        4 H u0 p0 c0 {2,S}
        5 O u0 p2 c0 {3,D}
        """
        adj7 = """
        multiplicity 2
        1 N u1 p0 c+1 {2,D} {3,S}
        2 N u0 p1 c0 {1,D} {4,S}
        3 S u0 p2 c0 {1,S} {5,S}
        4 H u0 p0 c0 {2,S}
        5 O u0 p3 c-1 {3,S}
        """

        mol_list = [Molecule().from_adjacency_list(adj1),
                    Molecule().from_adjacency_list(adj2),
                    Molecule().from_adjacency_list(adj3),
                    Molecule().from_adjacency_list(adj4),
                    Molecule().from_adjacency_list(adj5),
                    Molecule().from_adjacency_list(adj6),
                    Molecule().from_adjacency_list(adj7)]

        for mol in mol_list:
            mol.update()  # the charge_filtration uses the atom.sorting_label attribute

        filtered_list = charge_filtration(mol_list, get_charge_span_list(mol_list))
        self.assertEqual(len(filtered_list), 4)
        self.assertTrue(any([mol.get_charge_span() == 1 for mol in filtered_list]))
        for mol in filtered_list:
            if mol.get_charge_span() == 1:
                for atom in mol.vertices:
                    if abs(atom.charge) == 1:
                        self.assertTrue(atom.is_nitrogen())

    def test_aromaticity(self):
        """Test that aromatics are properly filtered."""
        adj1 = """multiplicity 2
1  C u0 p0 c0 {2,D} {3,S} {7,S}
2  C u0 p0 c0 {1,D} {4,S} {8,S}
3  C u0 p0 c0 {1,S} {6,D} {12,S}
4  C u0 p0 c0 {2,S} {5,D} {9,S}
5  C u0 p0 c0 {4,D} {6,S} {10,S}
6  C u0 p0 c0 {3,D} {5,S} {11,S}
7  C u1 p0 c0 {1,S} {13,S} {14,S}
8  H u0 p0 c0 {2,S}
9  H u0 p0 c0 {4,S}
10 H u0 p0 c0 {5,S}
11 H u0 p0 c0 {6,S}
12 H u0 p0 c0 {3,S}
13 H u0 p0 c0 {7,S}
14 H u0 p0 c0 {7,S}
"""
        adj2 = """multiplicity 2
1  C u0 p0 c0 {2,B} {3,B} {7,S}
2  C u0 p0 c0 {1,B} {4,B} {8,S}
3  C u0 p0 c0 {1,B} {6,B} {12,S}
4  C u0 p0 c0 {2,B} {5,B} {9,S}
5  C u0 p0 c0 {4,B} {6,B} {10,S}
6  C u0 p0 c0 {3,B} {5,B} {11,S}
7  C u1 p0 c0 {1,S} {13,S} {14,S}
8  H u0 p0 c0 {2,S}
9  H u0 p0 c0 {4,S}
10 H u0 p0 c0 {5,S}
11 H u0 p0 c0 {6,S}
12 H u0 p0 c0 {3,S}
13 H u0 p0 c0 {7,S}
14 H u0 p0 c0 {7,S}
"""
        adj3 = """multiplicity 2
1  C u0 p0 c0 {2,S} {3,S} {7,D}
2  C u1 p0 c0 {1,S} {4,S} {8,S}
3  C u0 p0 c0 {1,S} {6,D} {12,S}
4  C u0 p0 c0 {2,S} {5,D} {9,S}
5  C u0 p0 c0 {4,D} {6,S} {10,S}
6  C u0 p0 c0 {3,D} {5,S} {11,S}
7  C u0 p0 c0 {1,D} {13,S} {14,S}
8  H u0 p0 c0 {2,S}
9  H u0 p0 c0 {4,S}
10 H u0 p0 c0 {5,S}
11 H u0 p0 c0 {6,S}
12 H u0 p0 c0 {3,S}
13 H u0 p0 c0 {7,S}
14 H u0 p0 c0 {7,S}
"""
        adj4 = """multiplicity 2
1  C u0 p0 c0 {2,S} {3,S} {7,D}
2  C u0 p0 c0 {1,S} {5,D} {8,S}
3  C u0 p0 c0 {1,S} {6,D} {12,S}
4  C u1 p0 c0 {5,S} {6,S} {10,S}
5  C u0 p0 c0 {2,D} {4,S} {9,S}
6  C u0 p0 c0 {3,D} {4,S} {11,S}
7  C u0 p0 c0 {1,D} {13,S} {14,S}
8  H u0 p0 c0 {2,S}
9  H u0 p0 c0 {5,S}
10 H u0 p0 c0 {4,S}
11 H u0 p0 c0 {6,S}
12 H u0 p0 c0 {3,S}
13 H u0 p0 c0 {7,S}
14 H u0 p0 c0 {7,S}
"""

        mol_list = [
            Molecule().from_adjacency_list(adj1),
            Molecule().from_adjacency_list(adj2),
            Molecule().from_adjacency_list(adj3),
            Molecule().from_adjacency_list(adj4),
        ]

        filtered_list = aromaticity_filtration(mol_list, analyze_molecule(mol_list[0]))
        self.assertEqual(len(filtered_list), 3)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
