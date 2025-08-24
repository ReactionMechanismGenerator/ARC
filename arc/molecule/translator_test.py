#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit test for the translator module.
"""

import unittest
from unittest.mock import patch

from arc.molecule.atomtype import ATOMTYPES
from arc.molecule.molecule import Molecule
from arc.molecule.translator import *


class TranslatorTest(unittest.TestCase):

    def test_empty_molecule(self):
        """Test that we can safely return a blank identifier for an empty molecule."""
        mol = Molecule()

        self.assertEqual(mol.to_smiles(), '')
        self.assertEqual(mol.to_inchi(), '')


class SMILESGenerationTest(unittest.TestCase):
    def compare(self, adjlist, smiles):
        mol = Molecule().from_adjacency_list(adjlist)
        self.assertEqual(smiles, mol.to_smiles())

    def test_ch4(self):
        "Test the SMILES generation for methane"

        adjlist = """
1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {1,S}
        """
        smiles = "C"
        self.compare(adjlist, smiles)

    def test_c(self):
        "Test the SMILES generation for atomic carbon mult=(1,3,5)"
        adjlist = "1 C u0 p2 c0"
        smiles = "[C]"
        self.compare(adjlist, smiles)

        adjlist = "multiplicity 3\n1 C u2 p1 c0"
        smiles = "[C]"
        self.compare(adjlist, smiles)

        adjlist = "multiplicity 5\n1 C u4 p0 c0"
        smiles = "[C]"
        self.compare(adjlist, smiles)

    def test_various(self):
        "Test the SMILES generation for various molecules and radicals"

        # Test N2
        adjlist = '''
        1 N u0 p1 c0 {2,T}
        2 N u0 p1 c0 {1,T}
        '''
        smiles = 'N#N'
        self.compare(adjlist, smiles)

        # Test CH4
        adjlist = '''
        1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
        2 H u0 p0 c0 {1,S}
        3 H u0 p0 c0 {1,S}
        4 H u0 p0 c0 {1,S}
        5 H u0 p0 c0 {1,S}
        '''
        smiles = 'C'
        self.compare(adjlist, smiles)

        # Test H2O
        adjlist = '''
        1 O u0 p2 c0 {2,S} {3,S}
        2 H u0 p0 c0 {1,S}
        3 H u0 p0 c0 {1,S}
        '''
        smiles = 'O'
        self.compare(adjlist, smiles)

        # Test C2H6
        adjlist = '''
        1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
        2 C u0 p0 c0 {1,S} {6,S} {7,S} {8,S}
        3 H u0 p0 c0 {1,S}
        4 H u0 p0 c0 {1,S}
        5 H u0 p0 c0 {1,S}
        6 H u0 p0 c0 {2,S}
        7 H u0 p0 c0 {2,S}
        8 H u0 p0 c0 {2,S}
        '''
        smiles = 'CC'
        self.compare(adjlist, smiles)

        # Test H2
        adjlist = '''
        1 H u0 p0 c0 {2,S}
        2 H u0 p0 c0 {1,S}
        '''
        smiles = '[H][H]'
        self.compare(adjlist, smiles)

        # Test H2O2
        adjlist = '''
        1 O u0 p2 c0 {2,S} {3,S}
        2 O u0 p2 c0 {1,S} {4,S}
        3 H u0 p0 c0 {1,S}
        4 H u0 p0 c0 {2,S}
        '''
        smiles = 'OO'
        self.compare(adjlist, smiles)

        # Test C3H8
        adjlist = '''
        1  C u0 p0 c0 {2,S} {4,S} {5,S} {6,S}
        2  C u0 p0 c0 {1,S} {3,S} {7,S} {8,S}
        3  C u0 p0 c0 {2,S} {9,S} {10,S} {11,S}
        4  H u0 p0 c0 {1,S}
        5  H u0 p0 c0 {1,S}
        6  H u0 p0 c0 {1,S}
        7  H u0 p0 c0 {2,S}
        8  H u0 p0 c0 {2,S}
        9  H u0 p0 c0 {3,S}
        10 H u0 p0 c0 {3,S}
        11 H u0 p0 c0 {3,S}
        '''
        smiles = 'CCC'
        self.compare(adjlist, smiles)

        # Test Ar
        adjlist = '''
        1 Ar u0 p4 c0
        '''
        smiles = '[Ar]'
        self.compare(adjlist, smiles)

        # Test He
        adjlist = '''
        1 He u0 p1 c0
        '''
        smiles = '[He]'
        self.compare(adjlist, smiles)

        # Test CH4O
        adjlist = '''
        1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
        2 O u0 p2 c0 {1,S} {6,S}
        3 H u0 p0 c0 {1,S}
        4 H u0 p0 c0 {1,S}
        5 H u0 p0 c0 {1,S}
        6 H u0 p0 c0 {2,S}
        '''
        smiles = 'CO'
        self.compare(adjlist, smiles)

        # Test CO2
        adjlist = '''
        1 O u0 p2 c0 {2,D}
        2 C u0 p0 c0 {1,D} {3,D}
        3 O u0 p2 c0 {2,D}
        '''
        smiles = 'O=C=O'
        self.compare(adjlist, smiles)

        # Test CO
        adjlist = '''
        1 C u0 p1 c-1 {2,T}
        2 O u0 p1 c+1 {1,T}
        '''
        smiles = '[C-]#[O+]'
        self.compare(adjlist, smiles)

        # Test C2H4
        adjlist = '''
        1 C u0 p0 c0 {2,D} {3,S} {4,S}
        2 C u0 p0 c0 {1,D} {5,S} {6,S}
        3 H u0 p0 c0 {1,S}
        4 H u0 p0 c0 {1,S}
        5 H u0 p0 c0 {2,S}
        6 H u0 p0 c0 {2,S}
        '''
        smiles = 'C=C'
        self.compare(adjlist, smiles)

        # Test O2
        adjlist = '''
        1 O u0 p2 c0 {2,D}
        2 O u0 p2 c0 {1,D}
        '''
        smiles = 'O=O'
        self.compare(adjlist, smiles)

        # Test CH3
        adjlist = '''
        multiplicity 2
        1 C u1 p0 c0 {2,S} {3,S} {4,S}
        2 H u0 p0 c0 {1,S}
        3 H u0 p0 c0 {1,S}
        4 H u0 p0 c0 {1,S}
        '''
        smiles = '[CH3]'
        self.compare(adjlist, smiles)

        # Test HO
        adjlist = '''
        multiplicity 2
        1 O u1 p2 c0 {2,S}
        2 H u0 p0 c0 {1,S}
        '''
        smiles = '[OH]'
        self.compare(adjlist, smiles)

        # Test C2H5
        adjlist = '''
        multiplicity 2
        1 C u0 p0 c0 {2,S} {5,S} {6,S} {7,S}
        2 C u1 p0 c0 {1,S} {3,S} {4,S}
        3 H u0 p0 c0 {2,S}
        4 H u0 p0 c0 {2,S}
        5 H u0 p0 c0 {1,S}
        6 H u0 p0 c0 {1,S}
        7 H u0 p0 c0 {1,S}
        '''
        smiles = 'C[CH2]'
        self.compare(adjlist, smiles)

        # Test O
        adjlist = '''
        multiplicity 3
        1 O u2 p2 c0
        '''
        smiles = '[O]'
        self.compare(adjlist, smiles)

        # Test HO2
        adjlist = '''
        multiplicity 2
        1 O u1 p2 c0 {2,S}
        2 O u0 p2 c0 {1,S} {3,S}
        3 H u0 p0 c0 {2,S}
        '''
        smiles = '[O]O'
        self.compare(adjlist, smiles)

        # Test CH
        adjlist = '''
        multiplicity 4
        1 C u3 p0 c0 {2,S}
        2 H u0 p0 c0 {1,S}
        '''
        smiles = '[CH]'
        self.compare(adjlist, smiles)

        # Test H
        adjlist = '''
        multiplicity 2
        1 H u1 p0 c0
        '''
        smiles = '[H]'
        self.compare(adjlist, smiles)

        # Test C
        adjlist = '''
        multiplicity 5
        1 C u4 p0 c0
        '''
        smiles = '[C]'
        self.compare(adjlist, smiles)

        # Test O2
        adjlist = '''
        multiplicity 3
        1 O u1 p2 c0 {2,S}
        2 O u1 p2 c0 {1,S}
        '''
        smiles = '[O][O]'
        self.compare(adjlist, smiles)

        # Test CF
        adjlist = '''
        multiplicity 2
        1 C u1 p1 c0 {2,S}
        2 F u0 p3 c0 {1,S}
        '''
        smiles = '[C]F'
        self.compare(adjlist, smiles)

        # Test CCl
        adjlist = '''
        multiplicity 2
        1 C u1 p1 c0 {2,S}
        2 Cl u0 p3 c0 {1,S}
        '''
        smiles = '[C]Cl'
        self.compare(adjlist, smiles)

        # Test CBr
        adjlist = '''
        multiplicity 2
        1 C u1 p1 c0 {2,S}
        2 Br u0 p3 c0 {1,S}
        '''
        smiles = '[C]Br'
        self.compare(adjlist, smiles)

    def test_aromatics(self):
        """Test that different aromatics representations returns different SMILES."""
        mol1 = Molecule().from_adjacency_list("""
1  O u0 p2 c0 {6,S} {9,S}
2  C u0 p0 c0 {3,D} {5,S} {11,S}
3  C u0 p0 c0 {2,D} {4,S} {12,S}
4  C u0 p0 c0 {3,S} {6,D} {13,S}
5  C u0 p0 c0 {2,S} {7,D} {10,S}
6  C u0 p0 c0 {1,S} {4,D} {7,S}
7  C u0 p0 c0 {5,D} {6,S} {8,S}
8  C u0 p0 c0 {7,S} {14,S} {15,S} {16,S}
9  H u0 p0 c0 {1,S}
10 H u0 p0 c0 {5,S}
11 H u0 p0 c0 {2,S}
12 H u0 p0 c0 {3,S}
13 H u0 p0 c0 {4,S}
14 H u0 p0 c0 {8,S}
15 H u0 p0 c0 {8,S}
16 H u0 p0 c0 {8,S}
""")
        mol2 = Molecule().from_adjacency_list("""
1  O u0 p2 c0 {6,S} {9,S}
2  C u0 p0 c0 {3,S} {5,D} {11,S}
3  C u0 p0 c0 {2,S} {4,D} {12,S}
4  C u0 p0 c0 {3,D} {6,S} {13,S}
5  C u0 p0 c0 {2,D} {7,S} {10,S}
6  C u0 p0 c0 {1,S} {4,S} {7,D}
7  C u0 p0 c0 {5,S} {6,D} {8,S}
8  C u0 p0 c0 {7,S} {14,S} {15,S} {16,S}
9  H u0 p0 c0 {1,S}
10 H u0 p0 c0 {5,S}
11 H u0 p0 c0 {2,S}
12 H u0 p0 c0 {3,S}
13 H u0 p0 c0 {4,S}
14 H u0 p0 c0 {8,S}
15 H u0 p0 c0 {8,S}
16 H u0 p0 c0 {8,S}
""")
        mol3 = Molecule().from_adjacency_list("""
1  O u0 p2 c0 {6,S} {9,S}
2  C u0 p0 c0 {3,B} {5,B} {11,S}
3  C u0 p0 c0 {2,B} {4,B} {12,S}
4  C u0 p0 c0 {3,B} {6,B} {13,S}
5  C u0 p0 c0 {2,B} {7,B} {10,S}
6  C u0 p0 c0 {1,S} {4,B} {7,B}
7  C u0 p0 c0 {5,B} {6,B} {8,S}
8  C u0 p0 c0 {7,S} {14,S} {15,S} {16,S}
9  H u0 p0 c0 {1,S}
10 H u0 p0 c0 {5,S}
11 H u0 p0 c0 {2,S}
12 H u0 p0 c0 {3,S}
13 H u0 p0 c0 {4,S}
14 H u0 p0 c0 {8,S}
15 H u0 p0 c0 {8,S}
16 H u0 p0 c0 {8,S}
""")

        smiles1 = mol1.to_smiles()
        smiles2 = mol2.to_smiles()
        smiles3 = mol3.to_smiles()

        self.assertNotEqual(smiles1, smiles2)
        self.assertNotEqual(smiles2, smiles3)
        self.assertNotEqual(smiles1, smiles3)

    def test_surface_molecule_rdkit(self):
        """Test InChI generation for a surface molecule using RDKit"""
        mol = Molecule().from_adjacency_list("""
1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 X u0 p0 c0 {1,S}
""")
        smiles = '[CH3][Pt]'

        self.assertEqual(to_smiles(mol, backend='rdkit'), smiles)

    def test_surface_molecule_ob(self):
        """Test InChI generation for a surface molecule using OpenBabel"""
        mol = Molecule().from_adjacency_list("""
1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 X u0 p0 c0 {1,S}
""")
        smiles = 'C[Pt]'

        self.assertEqual(to_smiles(mol, backend='openbabel'), smiles)


class ParsingTest(unittest.TestCase):
    def setUp(self):
        self.methane = Molecule().from_adjacency_list("""
1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {1,S}
""")
        self.methylamine = Molecule().from_adjacency_list("""
1 N u0 p1 c0 {2,S} {3,S} {4,S}
2 C u0 p0 c0 {1,S} {5,S} {6,S} {7,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {2,S}
6 H u0 p0 c0 {2,S}
7 H u0 p0 c0 {2,S}
""")

    def test_from_augmented_inchi(self):
        aug_inchi = 'InChI=1S/CH4/h1H4'
        mol = from_augmented_inchi(Molecule(), aug_inchi)
        self.assertTrue(not mol.inchi == '')
        self.assertTrue(mol.is_isomorphic(self.methane))

        aug_inchi = 'InChI=1/CH4/h1H4'
        mol = from_augmented_inchi(Molecule(), aug_inchi)
        self.assertTrue(not mol.inchi == '')
        self.assertTrue(mol.is_isomorphic(self.methane))

    def compare(self, adjlist, smiles):
        """
        Compare result of parsing an adjacency list and a SMILES string.

        The adjacency list is presumed correct and this is to test the SMILES parser.
        """
        mol1 = Molecule().from_adjacency_list(adjlist)
        mol2 = Molecule(smiles=smiles)
        self.assertTrue(mol1.is_isomorphic(mol2),
                        "Parsing SMILES={!r} gave unexpected molecule\n{}".format(smiles, mol2.to_adjacency_list()))

    def test_from_smiles(self):
        smiles = 'C'
        mol = from_smiles(Molecule(), smiles)
        self.assertTrue(mol.is_isomorphic(self.methane))

        # Test that atomtypes that rely on lone pairs for identity are typed correctly
        smiles = 'CN'
        mol = from_smiles(Molecule(), smiles)
        self.assertEqual(mol.atoms[1].atomtype, ATOMTYPES['N3s'])

        # Test N2
        adjlist = '''
        1 N u0 p1 c0 {2,T}
        2 N u0 p1 c0 {1,T}
        '''
        smiles = 'N#N'
        self.compare(adjlist, smiles)

        # Test CH4
        adjlist = '''
        1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
        2 H u0 p0 c0 {1,S}
        3 H u0 p0 c0 {1,S}
        4 H u0 p0 c0 {1,S}
        5 H u0 p0 c0 {1,S}
        '''
        smiles = 'C'
        self.compare(adjlist, smiles)

        # Test H2O
        adjlist = '''
        1 O u0 p2 c0 {2,S} {3,S}
        2 H u0 p0 c0 {1,S}
        3 H u0 p0 c0 {1,S}
        '''
        smiles = 'O'
        self.compare(adjlist, smiles)

        # Test C2H6
        adjlist = '''
        1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
        2 C u0 p0 c0 {1,S} {6,S} {7,S} {8,S}
        3 H u0 p0 c0 {1,S}
        4 H u0 p0 c0 {1,S}
        5 H u0 p0 c0 {1,S}
        6 H u0 p0 c0 {2,S}
        7 H u0 p0 c0 {2,S}
        8 H u0 p0 c0 {2,S}
        '''
        smiles = 'CC'
        self.compare(adjlist, smiles)

        # Test H2
        adjlist = '''
        1 H u0 p0 c0 {2,S}
        2 H u0 p0 c0 {1,S}
        '''
        smiles = '[H][H]'
        self.compare(adjlist, smiles)

        # Test H2O2
        adjlist = '''
        1 O u0 p2 c0 {2,S} {3,S}
        2 O u0 p2 c0 {1,S} {4,S}
        3 H u0 p0 c0 {1,S}
        4 H u0 p0 c0 {2,S}
        '''
        smiles = 'OO'
        self.compare(adjlist, smiles)

        # Test C3H8
        adjlist = '''
        1  C u0 p0 c0 {2,S} {4,S} {5,S} {6,S}
        2  C u0 p0 c0 {1,S} {3,S} {7,S} {8,S}
        3  C u0 p0 c0 {2,S} {9,S} {10,S} {11,S}
        4  H u0 p0 c0 {1,S}
        5  H u0 p0 c0 {1,S}
        6  H u0 p0 c0 {1,S}
        7  H u0 p0 c0 {2,S}
        8  H u0 p0 c0 {2,S}
        9  H u0 p0 c0 {3,S}
        10 H u0 p0 c0 {3,S}
        11 H u0 p0 c0 {3,S}
        '''
        smiles = 'CCC'
        self.compare(adjlist, smiles)

        # Test Ar
        adjlist = '''
        1 Ar u0 p4 c0
        '''
        smiles = '[Ar]'
        self.compare(adjlist, smiles)

        # Test He
        adjlist = '''
        1 He u0 p1 c0
        '''
        smiles = '[He]'
        self.compare(adjlist, smiles)

        # Test CH4O
        adjlist = '''
        1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
        2 O u0 p2 c0 {1,S} {6,S}
        3 H u0 p0 c0 {1,S}
        4 H u0 p0 c0 {1,S}
        5 H u0 p0 c0 {1,S}
        6 H u0 p0 c0 {2,S}
        '''
        smiles = 'CO'
        self.compare(adjlist, smiles)

        # Test CO2
        adjlist = '''
        1 O u0 p2 c0 {2,D}
        2 C u0 p0 c0 {1,D} {3,D}
        3 O u0 p2 c0 {2,D}
        '''
        smiles = 'O=C=O'
        self.compare(adjlist, smiles)

        # Test CO
        adjlist = '''
        1 C u0 p1 c-1 {2,T}
        2 O u0 p1 c+1 {1,T}
        '''
        smiles = '[C-]#[O+]'
        self.compare(adjlist, smiles)

        # Test C2H4
        adjlist = '''
        1 C u0 p0 c0 {2,D} {3,S} {4,S}
        2 C u0 p0 c0 {1,D} {5,S} {6,S}
        3 H u0 p0 c0 {1,S}
        4 H u0 p0 c0 {1,S}
        5 H u0 p0 c0 {2,S}
        6 H u0 p0 c0 {2,S}
        '''
        smiles = 'C=C'
        self.compare(adjlist, smiles)

        # Test O2
        adjlist = '''
        1 O u0 p2 c0 {2,D}
        2 O u0 p2 c0 {1,D}
        '''
        smiles = 'O=O'
        self.compare(adjlist, smiles)

        # Test CH3
        adjlist = '''
        multiplicity 2
        1 C u1 p0 c0 {2,S} {3,S} {4,S}
        2 H u0 p0 c0 {1,S}
        3 H u0 p0 c0 {1,S}
        4 H u0 p0 c0 {1,S}
        '''
        smiles = '[CH3]'
        self.compare(adjlist, smiles)

        # Test HO
        adjlist = '''
        multiplicity 2
        1 O u1 p2 c0 {2,S}
        2 H u0 p0 c0 {1,S}
        '''
        smiles = '[OH]'
        self.compare(adjlist, smiles)

        # Test C2H5
        adjlist = '''
        multiplicity 2
        1 C u0 p0 c0 {2,S} {5,S} {6,S} {7,S}
        2 C u1 p0 c0 {1,S} {3,S} {4,S}
        3 H u0 p0 c0 {2,S}
        4 H u0 p0 c0 {2,S}
        5 H u0 p0 c0 {1,S}
        6 H u0 p0 c0 {1,S}
        7 H u0 p0 c0 {1,S}
        '''
        smiles = 'C[CH2]'
        self.compare(adjlist, smiles)

        # Test O
        adjlist = '''
        multiplicity 3
        1 O u2 p2 c0
        '''
        smiles = '[O]'
        self.compare(adjlist, smiles)

        # Test HO2
        adjlist = '''
        multiplicity 2
        1 O u1 p2 c0 {2,S}
        2 O u0 p2 c0 {1,S} {3,S}
        3 H u0 p0 c0 {2,S}
        '''
        smiles = '[O]O'
        self.compare(adjlist, smiles)

        # Test CH, methylidyne.
        # Wikipedia reports:
        # The ground state is a doublet radical with one unpaired electron,
        # and the first two excited states are a quartet radical with three
        # unpaired electrons and a doublet radical with one unpaired electron.
        # With the quartet radical only 71 kJ above the ground state, a sample
        # of methylidyne exists as a mixture of electronic states even at
        # room temperature, giving rise to complex reactions.
        #
        adjlist = '''
        multiplicity 2
        1 C u1 p1 c0 {2,S}
        2 H u0 p0 c0 {1,S}
        '''
        smiles = '[CH]'
        self.compare(adjlist, smiles)

        # Test H
        adjlist = '''
        multiplicity 2
        1 H u1 p0 c0
        '''
        smiles = '[H]'
        self.compare(adjlist, smiles)

        # Test atomic C, which is triplet in ground state
        adjlist = '''
        multiplicity 3
        1 C u2 p1 c0
        '''
        smiles = '[C]'
        self.compare(adjlist, smiles)

        # Test O2
        adjlist = '''
        multiplicity 3
        1 O u1 p2 c0 {2,S}
        2 O u1 p2 c0 {1,S}
        '''
        smiles = '[O][O]'
        self.compare(adjlist, smiles)

    def test_from_inchi(self):
        inchi = 'InChI=1S/CH4/h1H4'
        mol = from_inchi(Molecule(), inchi)
        self.assertTrue(mol.is_isomorphic(self.methane))
        # Test that atomtypes that rely on lone pairs for identity are typed correctly
        inchi = "InChI=1S/CH5N/c1-2/h2H2,1H3"
        mol = from_inchi(Molecule(), inchi)
        self.assertEqual(mol.atoms[1].atomtype, ATOMTYPES['N3s'])

    # current implementation of SMARTS is broken
    def test_from_smarts(self):
        smarts = '[CH4]'
        mol = from_smarts(Molecule(), smarts)
        self.assertTrue(mol.is_isomorphic(self.methane))

    def test_incorrect_identifier_type(self):
        """Test that the appropriate error is raised for identifier/type mismatch."""
        with self.assertRaises(ValueError) as cm:
            Molecule().from_smiles('InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H')

        self.assertTrue('Improper identifier type' in str(cm.exception))

    def test_read_inchikey_error(self):
        """Test that the correct error is raised when reading an InChIKey"""
        with self.assertRaises(ValueError) as cm:
            Molecule().from_inchi('InChIKey=UHOVQNZJYSORNB-UHFFFAOYSA-N')

        self.assertTrue('InChIKey is a write-only format' in str(cm.exception))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
