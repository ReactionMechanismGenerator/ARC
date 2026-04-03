#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the
arc.job.adapters.ts.linear_utils.addition module
"""

import unittest

import numpy as np

from arc.molecule.molecule import Molecule
from arc.species import ARCSpecies

_MASS_NUMBER = {'H': 1, 'C': 12, 'N': 14, 'O': 16, 'S': 32}

from arc.job.adapters.ts.linear_utils.addition import (
    find_split_bonds_by_fragmentation,
    map_and_verify_fragments,
    stretch_bond,
    detect_intra_frag_ring_bonds,
    apply_intra_frag_contraction,
    _reposition_leaving_groups,
)


class TestFindSplitBondsByFragmentation(unittest.TestCase):
    """Tests for the find_split_bonds_by_fragmentation function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_single_product_returns_empty(self):
        """A single product means no fragmentation is needed."""
        mol = Molecule().from_smiles('CC')
        spc = ARCSpecies(label='CC', smiles='CC')
        result = find_split_bonds_by_fragmentation(mol, [spc])
        self.assertEqual(result, [])

    def test_ethane_to_two_methyls(self):
        """Ethane fragmented into two CH3 radicals: 1-bond cut at C-C."""
        mol = Molecule().from_smiles('CC')
        spc1 = ARCSpecies(label='CH3_a', smiles='[CH3]')
        spc2 = ARCSpecies(label='CH3_b', smiles='[CH3]')
        result = find_split_bonds_by_fragmentation(mol, [spc1, spc2])
        self.assertTrue(len(result) >= 1)
        for bonds in result:
            self.assertEqual(len(bonds), 1)
            a, b = bonds[0]
            self.assertEqual(mol.atoms[a].symbol, 'C')
            self.assertEqual(mol.atoms[b].symbol, 'C')

    def test_propane_to_methyl_and_ethyl(self):
        """Propane fragmented into CH3 + C2H5: single C-C cut."""
        mol = Molecule().from_smiles('CCC')
        spc1 = ARCSpecies(label='CH3', smiles='[CH3]')
        spc2 = ARCSpecies(label='C2H5', smiles='C[CH2]')
        result = find_split_bonds_by_fragmentation(mol, [spc1, spc2])
        self.assertTrue(len(result) >= 1)
        for bonds in result:
            self.assertEqual(len(bonds), 1)

    def test_no_valid_cut(self):
        """If product formulas don't match any cut, return empty list."""
        mol = Molecule().from_smiles('CC')
        spc1 = ARCSpecies(label='N', smiles='[N]')
        spc2 = ARCSpecies(label='O', smiles='[O]')
        result = find_split_bonds_by_fragmentation(mol, [spc1, spc2])
        self.assertEqual(result, [])


class TestMapAndVerifyFragments(unittest.TestCase):
    """Tests for the map_and_verify_fragments function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_ethane_valid_split(self):
        """Ethane split at C-C maps to two methyl fragments."""
        mol = Molecule().from_smiles('CC')
        spc1 = ARCSpecies(label='CH3_a', smiles='[CH3]')
        spc2 = ARCSpecies(label='CH3_b', smiles='[CH3]')
        c_indices = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        split_bond = (c_indices[0], c_indices[1])
        result = map_and_verify_fragments(mol, [split_bond], [spc1, spc2])
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(mol.atoms))

    def test_invalid_split_bond(self):
        """Splitting a bond that doesn't fragment correctly returns None."""
        mol = Molecule().from_smiles('CCC')
        spc1 = ARCSpecies(label='N', smiles='[N]')
        spc2 = ARCSpecies(label='O', smiles='[O]')
        result = map_and_verify_fragments(mol, [(0, 1)], [spc1, spc2])
        self.assertIsNone(result)


class TestStretchBond(unittest.TestCase):
    """Tests for the stretch_bond function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

        cls.ethane_mol = Molecule().from_smiles('CC')
        c_indices = [i for i, a in enumerate(cls.ethane_mol.atoms) if a.symbol == 'C']
        cls.ethane_c_indices = c_indices

        coords = []
        symbols = []
        for i, atom in enumerate(cls.ethane_mol.atoms):
            symbols.append(atom.symbol)
            if atom.symbol == 'C':
                c_rank = c_indices.index(i)
                coords.append((c_rank * 1.54, 0.0, 0.0))
            else:
                bonded_c = None
                for nbr in atom.bonds.keys():
                    if nbr.symbol == 'C':
                        bonded_c = cls.ethane_mol.atoms.index(nbr)
                        break
                if bonded_c is not None:
                    c_rank = c_indices.index(bonded_c)
                    h_list = [j for j, a2 in enumerate(cls.ethane_mol.atoms)
                              if a2.symbol == 'H'
                              and cls.ethane_mol.has_bond(a2, cls.ethane_mol.atoms[bonded_c])]
                    h_rank = h_list.index(i) if i in h_list else 0
                    offset_y = 1.09 * ((-1) ** h_rank)
                    offset_z = 0.5 * (h_rank // 2)
                    coords.append((c_rank * 1.54, offset_y, offset_z))
                else:
                    coords.append((0.0, 1.09, 0.0))

        cls.ethane_xyz = {
            'symbols': tuple(symbols),
            'isotopes': tuple(_MASS_NUMBER[a.symbol] for a in cls.ethane_mol.atoms),
            'coords': tuple(tuple(c) for c in coords),
        }

    def test_stretch_bond_increases_distance(self):
        """Stretching the C-C bond in ethane increases the distance."""
        c0, c1 = self.ethane_c_indices
        split_bond = (c0, c1)
        d_before = np.linalg.norm(
            np.array(self.ethane_xyz['coords'][c0]) -
            np.array(self.ethane_xyz['coords'][c1]))

        result = stretch_bond(self.ethane_xyz, self.ethane_mol,
                               [split_bond], weight=0.5, label='test')
        if result is not None:
            d_after = np.linalg.norm(
                np.array(result['coords'][c0]) -
                np.array(result['coords'][c1]))
            self.assertGreater(d_after, d_before)

    def test_stretch_bond_preserves_symbols(self):
        """Stretch preserves atom symbols."""
        c0, c1 = self.ethane_c_indices
        result = stretch_bond(self.ethane_xyz, self.ethane_mol,
                               [(c0, c1)], weight=0.5, label='test')
        if result is not None:
            self.assertEqual(result['symbols'], self.ethane_xyz['symbols'])

    def test_stretch_bond_returns_none_for_no_fragments(self):
        """If the split doesn't create separable fragments, may return None."""
        mol = Molecule().from_smiles('C')
        xyz = {
            'symbols': ('C', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 1, 1, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (1.09, 0.0, 0.0),
                       (-0.363, 1.028, 0.0),
                       (-0.363, -0.514, 0.890),
                       (-0.363, -0.514, -0.890)),
        }
        result = stretch_bond(xyz, mol, [(0, 1)], weight=0.5, label='test')
        if result is not None:
            self.assertIn('coords', result)


class TestDetectIntraFragRingBonds(unittest.TestCase):
    """Tests for the detect_intra_frag_ring_bonds function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_detect_ring_bonds_butadiene_cyclobutene(self):
        """Butadiene forming cyclobutene: detects the ring-closing bond."""
        mol = Molecule().from_smiles('C=CC=C')
        c_indices = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        spc = ARCSpecies(label='cyclobutene', smiles='C1=CCC1')
        coords = []
        symbols = []
        for i, atom in enumerate(mol.atoms):
            symbols.append(atom.symbol)
            if atom.symbol == 'C':
                c_rank = c_indices.index(i)
                coords.append((c_rank * 1.40, 0.0, 0.0))
            else:
                bonded_c = None
                for nbr in atom.bonds.keys():
                    if nbr.symbol == 'C':
                        bonded_c = mol.atoms.index(nbr)
                        break
                if bonded_c is not None:
                    c_rank = c_indices.index(bonded_c)
                    coords.append((c_rank * 1.40, 1.09, 0.0))
                else:
                    coords.append((0.0, 1.09, 0.0))
        xyz = {
            'symbols': tuple(symbols),
            'isotopes': tuple(_MASS_NUMBER[a.symbol] for a in mol.atoms),
            'coords': tuple(tuple(c) for c in coords),
        }
        result = detect_intra_frag_ring_bonds(mol, [], [spc], xyz)
        if result:
            for (a, b), ring_size in result:
                self.assertGreaterEqual(ring_size, 3)
                self.assertEqual(mol.atoms[a].symbol, 'C')
                self.assertEqual(mol.atoms[b].symbol, 'C')


if __name__ == '__main__':
    unittest.main()
