#!/usr/bin/env python3
# encoding: utf-8

import unittest

from arc.molecule.adjlist import InvalidAdjacencyListError
from arc.molecule.group import Group
from arc.molecule.molecule import Molecule


class TestGroupAdjLists(unittest.TestCase):
    """
    Contains adjacency list unit tests of the Graph class.
    """
    def setUp(self):
        pass

    def test_from_adjacency_list(self):
        """
        adjlist: Test the Group.from_adjacency_list() method.
        """
        adjlist = """
1 *2 [Cs,Cd]   u0 {2,[S,D]} {3,S}
2 *1 [O2s,O2d] u0 {1,[S,D]}
3    R!H       u0 {1,S}
            """
        group = Group().from_adjacency_list(adjlist)

        atom1, atom2, atom3 = group.atoms
        self.assertTrue(group.has_bond(atom1, atom2))
        self.assertTrue(group.has_bond(atom1, atom3))
        self.assertFalse(group.has_bond(atom2, atom3))
        bond12 = atom1.bonds[atom2]
        bond13 = atom1.bonds[atom3]

        self.assertEqual(atom1.label, '*2')
        self.assertIn(atom1.atomtype[0].label, ['Cs', 'Cd'])
        self.assertIn(atom1.atomtype[1].label, ['Cs', 'Cd'])
        self.assertEqual(atom1.radical_electrons, [0])

        self.assertEqual(atom2.label, '*1')
        self.assertIn(atom2.atomtype[0].label, ['O2s', 'O2d'])
        self.assertIn(atom2.atomtype[1].label, ['O2s', 'O2d'])
        self.assertEqual(atom2.radical_electrons, [0])

        self.assertEqual(atom3.label, '')
        self.assertEqual(atom3.atomtype[0].label, 'R!H')
        self.assertEqual(atom3.radical_electrons, [0])

        self.assertEqual(bond12.order, [1, 2])
        self.assertTrue(bond13.is_single())

    def test_from_adjacency_list_multiplicity(self):
        gp = Group().from_adjacency_list(
            """
            multiplicity [1]
            1 C u0 p0 c0
            """
        )
        self.assertEqual(len(gp.multiplicity), 1)
        self.assertEqual(gp.multiplicity[0], 1)

    def test_from_adjacency_list_multiplicity_list(self):
        gp = Group().from_adjacency_list(
            """
            multiplicity [ 1, 3, 5 ]
            1 C u0 p0 c0
            """
        )
        self.assertEqual(len(gp.multiplicity), 3)
        self.assertEqual(gp.multiplicity[0], 1)
        self.assertEqual(gp.multiplicity[1], 3)
        self.assertEqual(gp.multiplicity[2], 5)

    def test_to_adjacency_list(self):
        """
        adjlist: Test the Group.to_adjacency_list() method.
        """
        adjlist = """
1 *2 [Cs,Cd]   u0 {2,[S,D]} {3,S}
2 *1 [O2s,O2d] u0 {1,[S,D]}
3    R!H       u0 {1,S}
            """
        group = Group().from_adjacency_list(adjlist)
        adjlist2 = group.to_adjacency_list()

        self.assertEqual(adjlist.strip(), adjlist2.strip())

    def test_atom_props(self):
        """Test that the atom props attribute can be properly read and written."""
        adjlist = """
1 *1 R!H u1 r0 {2,S}
2 *4 R!H u0 r0 {1,S} {3,S}
3 *2 Cb  u0 r1 {2,S} {4,B}
4 *3 Cb  u0 r1 {3,B}
        """
        group = Group().from_adjacency_list(adjlist)
        for atom in group.atoms:
            if atom.atomtype[0].label == 'R!H':
                self.assertFalse(atom.props['inRing'])
            elif atom.atomtype[0].label == 'Cb':
                self.assertTrue(atom.props['inRing'])
        adjlist2 = group.to_adjacency_list()

        self.assertEqual(adjlist.strip(), adjlist2.strip())
        
    def test_metal_facet_site_morphology(self):
        adjlist1 = """metal Cu
facet 111
1 *3 X u0 p0 c0 s"atop" m"terrace"  {2,S} {4,S}
2 *1 O u0 p2 c0 {1,S} {3,R}
3 *2 H u0 p0 c0 {2,R} {4,R}
4 *4 X u0 p0 c0 s"hcp" m"terrace" {3,R} {1,S}"""
        
        adjlist2 = """multiplicity [1]
metal [Cu, Fe, CuO2 ]
facet [111, 211, 1101, 110, ]
1 *3 X u0 p0 c0 s["atop","fcc"] m"terrace"  {2,S} {4,S}
2 *1 O u0 p2 c0 {1,S} {3,R}
3 *2 H u0 p0 c0 {2,R} {4,R}
4 *4 X u0 p0 c0 s"hcp" m["terrace","sc"] {3,R} {1,S}"""
        
        adjlist1test = """metal Cu
facet 111
1 *3 X u0 p0 c0 s"atop" m"terrace" {2,S} {4,S}
2 *1 O u0 p2 c0 {1,S} {3,R}
3 *2 H u0 p0 c0 {2,R} {4,R}
4 *4 X u0 p0 c0 s"hcp" m"terrace" {1,S} {3,R}"""
        
        adjlist2test = """multiplicity [1]
metal [Cu,Fe,CuO2]
facet [111,211,1101,110]
1 *3 X u0 p0 c0 s["atop","fcc"] m"terrace" {2,S} {4,S}
2 *1 O u0 p2 c0 {1,S} {3,R}
3 *2 H u0 p0 c0 {2,R} {4,R}
4 *4 X u0 p0 c0 s"hcp" m["terrace","sc"] {1,S} {3,R}"""
        mol = Molecule().from_adjacency_list(adjlist1,check_consistency=False)
        group = Group().from_adjacency_list(adjlist2)
        
        self.assertEqual(mol.metal,"Cu")
        self.assertEqual(mol.facet,"111")
        self.assertEqual(group.metal,["Cu","Fe","CuO2"])
        self.assertEqual(group.facet,["111","211","1101","110"])
        
        self.assertEqual(mol.atoms[0].site,"atop")
        self.assertEqual(mol.atoms[3].site,"hcp")
        self.assertEqual(mol.atoms[0].morphology, "terrace")
        self.assertEqual(mol.atoms[3].morphology, "terrace")
        
        self.assertEqual(group.atoms[0].site,["atop","fcc"])
        self.assertEqual(group.atoms[3].site,["hcp"])
        self.assertEqual(group.atoms[0].morphology, ["terrace"])
        self.assertEqual(group.atoms[3].morphology,["terrace","sc"])
        
        self.assertEqual(mol.to_adjacency_list().strip(),adjlist1test.strip())
        self.assertEqual(group.to_adjacency_list().strip(), adjlist2test.strip())



class TestMoleculeAdjLists(unittest.TestCase):
    """
    adjlist: Contains adjacency list unit tests of the Molecule class.
    """

    def setUp(self):
        pass

    def test_from_adjacency_list1(self):
        """
        adjlist: Test the Molecule.from_adjacency_list() method 1.
        """
        # molecule 1
        adjlist = """
1 *1 C u1 p0 c0  {2,S} {3,S} {4,S}
2    H u0 p0 c0  {1,S}
3    H u0 p0 c0  {1,S}
4 *2 N u0 p0 c+1 {1,S} {5,S} {6,D}
5    O u0 p3 c-1 {4,S}
6    O u0 p2 c0  {4,D}
            """
        molecule = Molecule().from_adjacency_list(adjlist)

        self.assertEqual(molecule.multiplicity, 2)

        atom1 = molecule.atoms[0]
        atom2 = molecule.atoms[3]
        atom3 = molecule.atoms[4]
        atom4 = molecule.atoms[5]
        self.assertTrue(molecule.has_bond(atom2, atom1))
        self.assertTrue(molecule.has_bond(atom2, atom3))
        self.assertTrue(molecule.has_bond(atom2, atom4))
        self.assertFalse(molecule.has_bond(atom1, atom3))
        self.assertFalse(molecule.has_bond(atom1, atom4))
        bond21 = atom2.bonds[atom1]
        bond23 = atom2.bonds[atom3]
        bond24 = atom2.bonds[atom4]

        self.assertEqual(atom1.label, '*1')
        self.assertEqual(atom1.element.symbol, 'C')
        self.assertEqual(atom1.radical_electrons, 1)
        self.assertEqual(atom1.charge, 0)

        self.assertEqual(atom2.label, '*2')
        self.assertEqual(atom2.element.symbol, 'N')
        self.assertEqual(atom2.radical_electrons, 0)
        self.assertEqual(atom2.charge, 1)

        self.assertEqual(atom3.label, '')
        self.assertEqual(atom3.element.symbol, 'O')
        self.assertEqual(atom3.radical_electrons, 0)
        self.assertEqual(atom3.charge, -1)

        self.assertEqual(atom4.label, '')
        self.assertEqual(atom4.element.symbol, 'O')
        self.assertEqual(atom4.radical_electrons, 0)
        self.assertEqual(atom4.charge, 0)

        self.assertTrue(bond21.is_single())
        self.assertTrue(bond23.is_single())
        self.assertTrue(bond24.is_double())

    def test_from_adjacency_list2(self):
        """
        adjlist: Test the Molecule.from_adjacency_list() method 2.
        """
        # molecule 2
        adjlist = """
1 *1 C u1 {2,S} {3,S} {4,S}
2    H u0 {1,S}
3    H u0 {1,S}
4 *2 N u0 p0 c+1 {1,S} {5,S} {6,D}
5    O u0 p3 c-1 {4,S}
6    O u0 p2 {4,D}
            """
        molecule = Molecule().from_adjacency_list(adjlist)

        self.assertEqual(molecule.multiplicity, 2)

        atom1 = molecule.atoms[0]
        atom2 = molecule.atoms[3]
        atom3 = molecule.atoms[4]
        atom4 = molecule.atoms[5]
        self.assertTrue(molecule.has_bond(atom2, atom1))
        self.assertTrue(molecule.has_bond(atom2, atom3))
        self.assertTrue(molecule.has_bond(atom2, atom4))
        self.assertFalse(molecule.has_bond(atom1, atom3))
        self.assertFalse(molecule.has_bond(atom1, atom4))
        bond21 = atom2.bonds[atom1]
        bond23 = atom2.bonds[atom3]
        bond24 = atom2.bonds[atom4]

        self.assertEqual(atom1.label, '*1')
        self.assertEqual(atom1.element.symbol, 'C')
        self.assertEqual(atom1.radical_electrons, 1)
        self.assertEqual(atom1.charge, 0)

        self.assertEqual(atom2.label, '*2')
        self.assertEqual(atom2.element.symbol, 'N')
        self.assertEqual(atom2.radical_electrons, 0)
        self.assertEqual(atom2.charge, 1)

        self.assertEqual(atom3.label, '')
        self.assertEqual(atom3.element.symbol, 'O')
        self.assertEqual(atom3.radical_electrons, 0)
        self.assertEqual(atom3.charge, -1)

        self.assertEqual(atom4.label, '')
        self.assertEqual(atom4.element.symbol, 'O')
        self.assertEqual(atom4.radical_electrons, 0)
        self.assertEqual(atom4.charge, 0)

        self.assertTrue(bond21.is_single())
        self.assertTrue(bond23.is_single())
        self.assertTrue(bond24.is_double())

    def test_from_adjacency_list3(self):
        """
        adjlist: Test the Molecule.from_adjacency_list() method 3.
        """
        # molecule 3
        adjlist = """
1 *1 C u1 {2,S} {3,S} {4,S}
2    H u0 {1,S}
3    H u0 {1,S}
4 *2 N u0 p0 c+1 {1,S} {5,S} {6,D}
5    O u0 p3 c-1 {4,S}
6    O u0 p2 {4,D}
            """
        molecule = Molecule().from_adjacency_list(adjlist)

        self.assertEqual(molecule.multiplicity, 2)

        atom1 = molecule.atoms[0]
        atom2 = molecule.atoms[3]
        atom3 = molecule.atoms[4]
        atom4 = molecule.atoms[5]
        self.assertTrue(molecule.has_bond(atom2, atom1))
        self.assertTrue(molecule.has_bond(atom2, atom3))
        self.assertTrue(molecule.has_bond(atom2, atom4))
        self.assertFalse(molecule.has_bond(atom1, atom3))
        self.assertFalse(molecule.has_bond(atom1, atom4))
        bond21 = atom2.bonds[atom1]
        bond23 = atom2.bonds[atom3]
        bond24 = atom2.bonds[atom4]

        self.assertEqual(atom1.label, '*1')
        self.assertEqual(atom1.element.symbol, 'C')
        self.assertEqual(atom1.radical_electrons, 1)
        self.assertEqual(atom1.charge, 0)

        self.assertEqual(atom2.label, '*2')
        self.assertEqual(atom2.element.symbol, 'N')
        self.assertEqual(atom2.radical_electrons, 0)
        self.assertEqual(atom2.charge, 1)

        self.assertEqual(atom3.label, '')
        self.assertEqual(atom3.element.symbol, 'O')
        self.assertEqual(atom3.radical_electrons, 0)
        self.assertEqual(atom3.charge, -1)

        self.assertEqual(atom4.label, '')
        self.assertEqual(atom4.element.symbol, 'O')
        self.assertEqual(atom4.radical_electrons, 0)
        self.assertEqual(atom4.charge, 0)

        self.assertTrue(bond21.is_single())
        self.assertTrue(bond23.is_single())
        self.assertTrue(bond24.is_double())

    def test_from_adjacency_list4(self):
        """
        adjlist: Test the Molecule.from_adjacency_list() method 4.
        """
        # molecule 4
        adjlist = """
1 *1 C u1 {2,S}
2 *2 N u0 p0 c+1 {1,S} {3,S} {4,D}
3    O u0 p3 c-1 {2,S}
4    O u0 p2 {2,D}
            """
        molecule = Molecule().from_adjacency_list(adjlist, saturate_h=True)

        self.assertEqual(molecule.multiplicity, 2)

        atom1 = molecule.atoms[0]
        atom2 = molecule.atoms[1]
        atom3 = molecule.atoms[2]
        atom4 = molecule.atoms[3]
        self.assertTrue(molecule.has_bond(atom2, atom1))
        self.assertTrue(molecule.has_bond(atom2, atom3))
        self.assertTrue(molecule.has_bond(atom2, atom4))
        self.assertFalse(molecule.has_bond(atom1, atom3))
        self.assertFalse(molecule.has_bond(atom1, atom4))
        bond21 = atom2.bonds[atom1]
        bond23 = atom2.bonds[atom3]
        bond24 = atom2.bonds[atom4]

        self.assertEqual(atom1.label, '*1')
        self.assertEqual(atom1.element.symbol, 'C')
        self.assertEqual(atom1.radical_electrons, 1)
        self.assertEqual(atom1.charge, 0)

        self.assertEqual(atom2.label, '*2')
        self.assertEqual(atom2.element.symbol, 'N')
        self.assertEqual(atom2.radical_electrons, 0)
        self.assertEqual(atom2.charge, 1)

        self.assertEqual(atom3.label, '')
        self.assertEqual(atom3.element.symbol, 'O')
        self.assertEqual(atom3.radical_electrons, 0)
        self.assertEqual(atom3.charge, -1)

        self.assertEqual(atom4.label, '')
        self.assertEqual(atom4.element.symbol, 'O')
        self.assertEqual(atom4.radical_electrons, 0)
        self.assertEqual(atom4.charge, 0)

        self.assertTrue(bond21.is_single())
        self.assertTrue(bond23.is_single())
        self.assertTrue(bond24.is_double())

    def test_from_adjacency_list5(self):
        """
        adjlist: Test if from_adjacency_list works when saturateH is turned on
        and test molecule is fused aromatics.
        """
        # molecule 5
        adjlist = """
1  * C u0 p0 c0 {2,B} {3,B} {4,B}
2    C u0 p0 c0 {1,B} {5,B} {6,B}
3    C u0 p0 c0 {1,B} {8,B} {13,S}
4    C u0 p0 c0 {1,B} {9,B}
5    C u0 p0 c0 {2,B} {10,B}
6    C u0 p0 c0 {2,B} {7,B}
7    C u0 p0 c0 {6,B} {8,B} {11,S}
8    C u0 p0 c0 {3,B} {7,B} {12,S}
9    C u0 p0 c0 {4,B} {10,B}
10   C u0 p0 c0 {5,B} {9,B}
11   H u0 p0 c0 {7,S}
12   H u0 p0 c0 {8,S}
13   H u0 p0 c0 {3,S}
            """
        molecule = Molecule().from_adjacency_list(adjlist, saturate_h=True)

        self.assertEqual(molecule.multiplicity, 1)

        atom1 = molecule.atoms[0]
        atom2 = molecule.atoms[1]
        atom3 = molecule.atoms[2]
        atom7 = molecule.atoms[6]
        atom11 = molecule.atoms[10]
        bond21 = atom2.bonds[atom1]
        bond13 = atom1.bonds[atom3]
        bond7_11 = atom7.bonds[atom11]

        self.assertEqual(atom1.label, '*')
        self.assertEqual(atom1.element.symbol, 'C')
        self.assertEqual(atom1.radical_electrons, 0)
        self.assertEqual(atom1.charge, 0)

        self.assertEqual(atom2.label, '')
        self.assertEqual(atom2.element.symbol, 'C')
        self.assertEqual(atom2.radical_electrons, 0)
        self.assertEqual(atom2.charge, 0)

        self.assertTrue(bond21.is_benzene())
        self.assertTrue(bond13.is_benzene())
        self.assertTrue(bond7_11.is_single())

    def test_wildcard_adjlists(self):
        """
        adjlist: Test that molecule adjlists containing wildcards raise an InvalidAdjacencyListError.
        """
        # A molecule with a wildcard assignment
        wildcard_adjlist1 = "1 C u1 px c0"
        wildcard_adjlist2 = "1 C ux p2 c0"
        wildcard_adjlist3 = "1 C u1 p2 cx"
        wildcard_adjlist4 = "1 [C,N] u1 p2 c0"

        with self.assertRaises(InvalidAdjacencyListError):
            Molecule().from_adjacency_list(wildcard_adjlist1)
        with self.assertRaises(InvalidAdjacencyListError):
            Molecule().from_adjacency_list(wildcard_adjlist2)
        with self.assertRaises(InvalidAdjacencyListError):
            Molecule().from_adjacency_list(wildcard_adjlist3)
        with self.assertRaises(InvalidAdjacencyListError):
            Molecule().from_adjacency_list(wildcard_adjlist4)

    def test_incorrect_adjlists(self):
        """
        adjlist: Test that improperly formed adjlists raise an InvalidAdjacencyListError.
        """
        # Carbon with 1 radical and 2 lone pairs = 5 total electrons.  Should have -1 charge but doesn't
        adjlist1 = "1 C u1 p2 c0"

        with self.assertRaises(InvalidAdjacencyListError):
            Molecule().from_adjacency_list(adjlist1)

    def test_helium(self):
        """
        adjlist: Test that the adjlist reading and writing works with Helium.
        """
        smiles = '[He]'
        inchi = 'InChI=1S/He'
        adjlist = '1 He u0 p1 c0'

        mol_smiles = Molecule().from_smiles(smiles)
        mol_inchi = Molecule().from_inchi(inchi)
        mol = Molecule().from_adjacency_list(adjlist)

        # Isomorphic check
        self.assertTrue(mol_smiles.is_isomorphic(mol))
        self.assertTrue(mol_smiles.is_isomorphic(mol_inchi))

        # Adjlist check
        self.assertEqual(mol_smiles.to_adjacency_list().strip(), adjlist)
        self.assertEqual(mol_inchi.to_adjacency_list().strip(), adjlist)
        self.assertEqual(mol.to_adjacency_list().strip(), adjlist)

        self.assertEqual(mol.to_smiles(), smiles)
        self.assertEqual(mol.to_inchi(), 'InChI=1S/He')

    def test_to_adjacency_list(self):
        """
        adjlist: Test the Molecule.to_adjacency_list() method.
        """
        adjlist = """
1 *1 C u1 p0 c0  {2,S} {3,S} {4,S}
2    H u0 p0 c0  {1,S}
3    H u0 p0 c0  {1,S}
4 *2 N u0 p0 c+1 {1,S} {5,S} {6,D}
5    O u0 p3 c-1 {4,S}
6    O u0 p2 c0  {4,D}
            """
        molecule = Molecule().from_adjacency_list(adjlist)
        adjlist_1 = molecule.to_adjacency_list(remove_h=False)
        new_molecule = Molecule().from_adjacency_list(adjlist_1)
        self.assertTrue(molecule.is_isomorphic(new_molecule))

    def test_to_adjacency_list_for_non_integer_bonds(self):
        """
        Test the adjacency list can be created for molecules with bond orders
        that don't fit into single, double, triple, or benzene
        """
        from arc.molecule.molecule import Atom, Bond, Molecule
        atom1 = Atom(element='H', lone_pairs=0)
        atom2 = Atom(element='H', lone_pairs=0)
        bond = Bond(atom1, atom2, 0.5)
        mol = Molecule(multiplicity=1)
        mol.add_atom(atom1)
        mol.add_atom(atom2)
        mol.add_bond(bond)
        adjlist = mol.to_adjacency_list()
        self.assertIn('H', adjlist)
        self.assertIn('{1,0.5}', adjlist)

    def test_adjacency_list(self):
        """
        adjlist: Check the adjacency list read/write functions for a full molecule.
        """
        molecule1 = Molecule().from_adjacency_list("""
        1  C u0 {2,D} {7,S} {8,S}
        2  C u0 {1,D} {3,S} {9,S}
        3  C u0 {2,S} {4,D} {10,S}
        4  C u0 {3,D} {5,S} {11,S}
        5  C u1 {4,S} {6,S} {12,S}
        6  C u0 {5,S} {13,S} {14,S} {15,S}
        7  H u0 {1,S}
        8  H u0 {1,S}
        9  H u0 {2,S}
        10 H u0 {3,S}
        11 H u0 {4,S}
        12 H u0 {5,S}
        13 H u0 {6,S}
        14 H u0 {6,S}
        15 H u0 {6,S}
        """)
        molecule2 = Molecule().from_smiles('C=CC=C[CH]C')
        self.assertTrue(molecule1.is_isomorphic(molecule2))
        self.assertTrue(molecule2.is_isomorphic(molecule1))

        # Test that charges are correctly stored and written with adjacency lists
        adjlist3 = """
1 C u0 p1 c-1 {2,T}
2 O u0 p1 c+1 {1,T}
"""
        molecule3 = Molecule().from_adjacency_list(adjlist3)
        self.assertEqual(molecule3.atoms[0].charge, -1)
        self.assertEqual(molecule3.atoms[1].charge, 1)
        adjlist4 = molecule3.to_adjacency_list()
        self.assertEqual(adjlist3.strip(), adjlist4.strip())

    def test_group_adjacency_list(self):
        """
        adjlist: Check the adjacency list read/write functions for a full molecule.
        """
        adjlist = """1 C u0 {2,D}
2 O u1 p1 c[-1,0,+1] {1,D}
"""
        group = Group().from_adjacency_list("""
        1 C u0 {2,D} 
        2 O u1 p1 c[-1,0,+1] {1,D}
        """)
        self.assertEqual(adjlist, group.to_adjacency_list())


class TestConsistencyChecker(unittest.TestCase):
    def test_check_hund_rule_fail(self):
        with self.assertRaises(InvalidAdjacencyListError):
            Molecule().from_adjacency_list("""
            multiplicity 1
            1 C u2 p0 c0
            """, saturate_h=True)

    def test_check_hund_rule_success(self):
        try:
            Molecule().from_adjacency_list("""
            multiplicity 3
            1 C u2 p0 c0
            """, saturate_h=True)
        except InvalidAdjacencyListError:
            self.fail('InvalidAdjacencyListError thrown unexpectedly!')

    def test_check_multiplicity(self):
        """
        adjlist: Check that RMG allows different electron spins in the same molecule with multiplicity = 2s + 1
        """
        # [N] radical:
        try:
            Molecule().from_adjacency_list('''multiplicity 4
                                            1 N u3 p1 c0''')
        except InvalidAdjacencyListError:
            self.fail('InvalidAdjacencyListError thrown unexpectedly for N tri-rad!')

        # A general molecule with 4 radicals, multiplicity 5:
        try:
            Molecule().from_adjacency_list('''multiplicity 5
                                            1 O u1 p2 c0 {2,S}
                                            2 C u1 p0 c0 {1,S} {3,S} {4,S}
                                            3 H u0 p0 c0 {2,S}
                                            4 N u1 p1 c0 {2,S} {5,S}
                                            5 O u1 p2 c0 {4,S}''')
        except InvalidAdjacencyListError:
            self.fail('InvalidAdjacencyListError thrown unexpectedly for a molecule with 4 radicals, multiplicity 5')

        # A general molecule with 4 radicals, multiplicity 3:
        try:
            Molecule().from_adjacency_list('''multiplicity 3
                                            1 O u1 p2 c0 {2,S}
                                            2 C u1 p0 c0 {1,S} {3,S} {4,S}
                                            3 H u0 p0 c0 {2,S}
                                            4 N u1 p1 c0 {2,S} {5,S}
                                            5 O u1 p2 c0 {4,S}''')
        except InvalidAdjacencyListError:
            self.fail('InvalidAdjacencyListError thrown unexpectedly for a molecule with 4 radicals, multiplicity 3')

        # [N]=C=[N] singlet:
        try:
            Molecule().from_adjacency_list('''multiplicity 1
                                            1 N u1 p1 c0 {2,D}
                                            2 C u0 p0 c0 {1,D} {3,D}
                                            3 N u1 p1 c0 {2,D}''')
        except InvalidAdjacencyListError:
            self.fail('InvalidAdjacencyListError thrown unexpectedly for singlet [N]=C=[N]!')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=3))
