#!/usr/bin/env python3
# encoding: utf-8

import unittest

from arc.exceptions import InchiException
from arc.molecule.inchi import (InChI, AugmentedInChI, remove_inchi_prefix, create_augmented_layers,
                                compose_aug_inchi, decompose_aug_inchi, INCHI_PREFIX, P_LAYER_PREFIX, U_LAYER_PREFIX)
from arc.molecule.inchi import (_has_unexpected_lone_pairs, _parse_e_layer, _parse_h_layer,
                                 _parse_n_layer, _reset_lone_pairs)
from arc.molecule.molecule import Atom, Molecule


class InChITest(unittest.TestCase):

    def test_constructor(self):
        inchi1 = InChI('InChI=1S/foo')
        self.assertTrue(inchi1 is not None)

        with self.assertRaises(InchiException):
            InChI('foo')

    def test_compare(self):
        inchi1 = InChI('InChI=1S/foo')
        inchi2 = InChI('InChI=1/foo')
        inchi3 = InChI('InChI=1/bar')

        self.assertTrue(inchi1 == inchi2)
        self.assertTrue(not inchi1 != inchi2)

        self.assertTrue((inchi1 < inchi2) is (inchi1 > inchi2))
        self.assertTrue((inchi1 < inchi3) is not (inchi1 > inchi3))


class AugmentedInChITest(unittest.TestCase):

    def test_constructor(self):
        aug_inchi1 = AugmentedInChI('InChI=1S/foo')

        self.assertTrue(aug_inchi1 == 'foo', aug_inchi1)
        self.assertTrue(aug_inchi1.u_indices is None)

        aug_inchi2 = AugmentedInChI('InChI=1S/foo')

        self.assertTrue(aug_inchi2 == 'foo', aug_inchi2)
        self.assertTrue(aug_inchi2.u_indices is None)

        aug_inchi3 = AugmentedInChI('InChI=1S/foo/u1,3')

        self.assertTrue(aug_inchi3 == 'foo/u1,3', aug_inchi3)
        self.assertTrue(aug_inchi3.u_indices == [1, 3])

    def test_compare(self):
        aug_inchi1 = AugmentedInChI('InChI=1S/foo')
        aug_inchi2 = AugmentedInChI('InChI=1/foo')

        self.assertTrue(aug_inchi1 == aug_inchi2)
        self.assertTrue(not aug_inchi1 != aug_inchi2)


class IgnorePrefixTest(unittest.TestCase):

    def test_ignore(self):
        string = 'InChI=1S/foo'
        self.assertTrue(remove_inchi_prefix(string) == 'foo')

        with self.assertRaises(InchiException):
            remove_inchi_prefix('foo')


class ComposeTest(unittest.TestCase):

    def test_compose_aug_inchi(self):
        inchi = 'C2H5/c1-2/h1H2,2H3'
        mult = 2

        aug_inchi = compose_aug_inchi(inchi, U_LAYER_PREFIX + str(mult))
        self.assertTrue(aug_inchi == INCHI_PREFIX + '/' + inchi + U_LAYER_PREFIX + str(mult), aug_inchi)


class Parse_H_LayerTest(unittest.TestCase):

    def test_oco(self):
        smi = 'O=C-O'
        inchi = Molecule().from_smiles(smi).to_inchi()
        mobile_hs = _parse_h_layer(inchi)
        expected = [[2, 3]]
        self.assertTrue(mobile_hs == expected)


class Parse_E_LayerTest(unittest.TestCase):
    def test_no_equivalence_layer(self):
        """Test that the absence of an E-layer results in an empty list."""

        auxinfo = "AuxInfo=1/0/N:1/rA:1C/rB:/rC:;"
        e_layer = _parse_e_layer(auxinfo)
        self.assertFalse(e_layer)

    def test_c8h22(self):
        auxinfo = "AuxInfo=1/0/N:1,8,4,6,2,7,3,5/E:(1,2)(3,4)(5,6)(7,8)/rA:8C.2C.2CCCCCC/rB:s1;s2;s3;s3;s5;s5;d7;/rC:;;;;;;;;"
        e_layer = _parse_e_layer(auxinfo)
        expected = [[1, 2], [3, 4], [5, 6], [7, 8]]
        self.assertTrue(e_layer == expected)

    def test_c7h17(self):
        auxinfo = "AuxInfo=1/0/N:3,5,7,2,4,6,1/E:(1,2,3)(4,5,6)/rA:7CCCCCCC/rB:s1;d2;s1;d4;s1;d6;/rC:;;;;;;;"
        e_layer = _parse_e_layer(auxinfo)
        expected = [[1, 2, 3], [4, 5, 6]]
        self.assertTrue(e_layer == expected)


class ParseNLayerTest(unittest.TestCase):
    def test_occc(self):
        auxinfo = "AuxInfo=1/0/N:4,3,2,1/rA:4OCCC/rB:s1;s2;s3;/rC:;;;;"
        n_layer = _parse_n_layer(auxinfo)
        expected = [4, 3, 2, 1]
        self.assertTrue(n_layer == expected)


class DecomposeTest(unittest.TestCase):

    def test_inchi(self):
        string = 'InChI=1S/XXXX/cXXX/hXXX'

        _, u_indices, _ = decompose_aug_inchi(string)
        self.assertEqual([], u_indices)

    def test_inchi_u_layer(self):
        string = 'InChI=1S/XXXX/cXXX/hXXX/u1,2'

        _, u_indices, _ = decompose_aug_inchi(string)
        self.assertEqual([1, 2], u_indices)

    def test_inchi_p_layer(self):
        string = 'InChI=1S/XXXX/cXXX/hXXX/lp1,2'
        _, _, p_indices = decompose_aug_inchi(string)
        self.assertEqual([1, 2], p_indices)

    def test_inchi_u_layer_p_layer(self):
        string = 'InChI=1S/XXXX/cXXX/hXXX/u1,2/lp3,4'
        _, u_indices, p_indices = decompose_aug_inchi(string)
        self.assertEqual([1, 2], u_indices)
        self.assertEqual([3, 4], p_indices)

    def test_inchi_p_layer_zero_lp(self):
        """
        Test that the p-layer containing an element with zero lone 
        pairs can be read correctly.
        """
        string = 'InChI=1S/XXXX/cXXX/hXXX/lp1(0)'
        _, _, p_indices = decompose_aug_inchi(string)
        self.assertEqual([(1, 0)], p_indices)


class CreateULayerTest(unittest.TestCase):
    def test_c4h6(self):
        """
        Test that 3-butene-1,2-diyl biradical is always resulting in the
        same u-layer, regardless of the original order.
        """

        # radical positions 3 and 4
        adjlist1 = """
1  C u0 p0 c0 {2,D} {5,S} {6,S}
2  C u0 p0 c0 {1,D} {3,S} {7,S}
3  C u1 p0 c0 {2,S} {4,S} {8,S}
4  C u1 p0 c0 {3,S} {9,S} {10,S}
5  H u0 p0 c0 {1,S}
6  H u0 p0 c0 {1,S}
7  H u0 p0 c0 {2,S}
8  H u0 p0 c0 {3,S}
9  H u0 p0 c0 {4,S}
10 H u0 p0 c0 {4,S}

        """

        # radical positions 1 and 2
        adjlist2 = """
1  C u1 p0 c0 {2,S} {5,S} {6,S}
2  C u1 p0 c0 {1,S} {3,S} {7,S}
3  C u0 p0 c0 {2,S} {4,D} {8,S}
4  C u0 p0 c0 {3,D} {9,S} {10,S}
5  H u0 p0 c0 {1,S}
6  H u0 p0 c0 {1,S}
7  H u0 p0 c0 {2,S}
8  H u0 p0 c0 {3,S}
9  H u0 p0 c0 {4,S}
10 H u0 p0 c0 {4,S}
        """

        u_layers = []
        for adjlist in [adjlist1, adjlist2]:
            mol = Molecule().from_adjacency_list(adjlist)
            u_layer = create_augmented_layers(mol)[0]
            u_layers.append(u_layer)

        self.assertEqual(u_layers[0], u_layers[1])


class ExpectedLonePairsTest(unittest.TestCase):
    def test_singlet_carbon(self):
        mol = Molecule(atoms=[Atom(element='C', lone_pairs=1)])
        unexpected = _has_unexpected_lone_pairs(mol)
        self.assertTrue(unexpected)

    def test_normal_carbon(self):
        mol = Molecule(atoms=[Atom(element='C', lone_pairs=0)])
        unexpected = _has_unexpected_lone_pairs(mol)
        self.assertFalse(unexpected)

    def test_normal_oxygen(self):
        mol = Molecule(atoms=[Atom(element='O', lone_pairs=2)])
        unexpected = _has_unexpected_lone_pairs(mol)
        self.assertFalse(unexpected)

    def test_oxygen_3_lone_pairs(self):
        mol = Molecule(atoms=[Atom(element='O', lone_pairs=3)])
        unexpected = _has_unexpected_lone_pairs(mol)
        self.assertTrue(unexpected)


class CreateAugmentedLayersTest(unittest.TestCase):
    def test_methane(self):
        smi = 'C'
        mol = Molecule().from_smiles(smi)
        ulayer, player = create_augmented_layers(mol)
        self.assertTrue(not ulayer)
        self.assertTrue(not player)

    def test_singlet_methylene(self):
        adjlist = """
multiplicity 1
1 C u0 p1 c0 {2,S} {3,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
"""
        mol = Molecule().from_adjacency_list(adjlist)
        ulayer, player = create_augmented_layers(mol)
        self.assertTrue(not ulayer)
        self.assertEqual(P_LAYER_PREFIX + '1', player)

    def test_triplet_methylene(self):
        adjlist = """
multiplicity 3
1 C u2 p0 c0 {2,S} {3,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
"""
        mol = Molecule().from_adjacency_list(adjlist)
        ulayer, player = create_augmented_layers(mol)
        self.assertEqual(U_LAYER_PREFIX + '1,1', ulayer)
        self.assertTrue(not player)


class ResetLonePairsTest(unittest.TestCase):

    def test_methane(self):
        smi = 'C'
        mol = Molecule().from_smiles(smi)
        p_indices = []

        _reset_lone_pairs(mol, p_indices)

        for at in mol.atoms:
            self.assertEqual(at.lone_pairs, 0)

    def test_singlet_methylene(self):
        adjlist = """
multiplicity 1
1 C u0 p1 c0 {2,S} {3,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
"""
        mol = Molecule().from_adjacency_list(adjlist)
        p_indices = [1]

        _reset_lone_pairs(mol, p_indices)

        for at in mol.atoms:
            if at.symbol == 'C':
                self.assertEqual(at.lone_pairs, 1)
            else:
                self.assertEqual(at.lone_pairs, 0)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
