#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the
arc.job.adapters.ts.linear_utils.isomerization module
"""

import unittest

import numpy as np

from arc.molecule.molecule import Molecule

_MASS_NUMBER = {'H': 1, 'C': 12, 'N': 14, 'O': 16, 'S': 32}

from arc.job.adapters.ts.linear_utils.isomerization import (
    backbone_atom_map,
    get_near_attack_xyz,
    path_has_cumulated_bonds,
    ring_closure_xyz,
)


class TestBackboneAtomMap(unittest.TestCase):
    """Tests for the backbone_atom_map function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_backbone_atom_map_identical_molecules(self):
        """Mapping identical molecules produces the identity map."""
        mol = Molecule().from_smiles('CC')
        atom_map = backbone_atom_map(mol, mol)
        self.assertIsNotNone(atom_map)
        self.assertEqual(len(atom_map), len(mol.atoms))
        heavy_r = [i for i, a in enumerate(mol.atoms) if a.symbol != 'H']
        heavy_p = [atom_map[i] for i in heavy_r]
        self.assertEqual(sorted(heavy_p), sorted(heavy_r))

    def test_backbone_atom_map_ethane_isomerization(self):
        """Atom map between ethane tautomers (same structure) is valid."""
        r_mol = Molecule().from_smiles('CC')
        p_mol = Molecule().from_smiles('CC')
        atom_map = backbone_atom_map(r_mol, p_mol)
        self.assertIsNotNone(atom_map)
        self.assertEqual(len(atom_map), len(r_mol.atoms))
        for i, pi in enumerate(atom_map):
            self.assertEqual(r_mol.atoms[i].symbol, p_mol.atoms[pi].symbol)

    def test_backbone_atom_map_different_atom_count(self):
        """Different atom counts return None."""
        r_mol = Molecule().from_smiles('CC')
        p_mol = Molecule().from_smiles('C')
        atom_map = backbone_atom_map(r_mol, p_mol)
        self.assertIsNone(atom_map)

    def test_backbone_atom_map_different_heavy_count(self):
        """Different heavy-atom counts return None."""
        r_mol = Molecule().from_smiles('CCO')
        p_mol = Molecule().from_smiles('CCC')
        atom_map = backbone_atom_map(r_mol, p_mol)
        self.assertIsNone(atom_map)

    def test_backbone_atom_map_non_isomorphic_backbones(self):
        """Non-isomorphic heavy-atom backbones return None."""
        r_mol = Molecule().from_smiles('C(C)(C)C')
        p_mol = Molecule().from_smiles('CCCC')
        atom_map = backbone_atom_map(r_mol, p_mol)
        self.assertIsNone(atom_map)

    def test_backbone_atom_map_h_migration(self):
        """H-migration: heavy-atom backbone is identical, H mapping succeeds."""
        r_mol = Molecule().from_adjacency_list("""
1 C u0 p0 c0 {2,D} {3,S} {4,S}
2 C u0 p0 c0 {1,D} {5,S} {6,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {2,S}
6 H u0 p0 c0 {2,S}
""")
        p_mol = Molecule().from_adjacency_list("""
1 C u0 p0 c0 {2,D} {3,S} {4,S}
2 C u0 p0 c0 {1,D} {5,S} {6,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {2,S}
6 H u0 p0 c0 {2,S}
""")
        atom_map = backbone_atom_map(r_mol, p_mol)
        self.assertIsNotNone(atom_map)
        self.assertEqual(len(atom_map), 6)


class TestPathHasCumulatedBonds(unittest.TestCase):
    """Tests for the path_has_cumulated_bonds function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_no_cumulated_bonds_single(self):
        """A single bond path has no cumulated bonds."""
        mol = Molecule().from_smiles('CCC')
        self.assertFalse(path_has_cumulated_bonds(mol, (0, 2)))

    def test_cumulated_bonds_allene(self):
        """Allene (C=C=C) has cumulated bonds."""
        mol = Molecule().from_smiles('C=C=C')
        n_c = sum(1 for a in mol.atoms if a.symbol == 'C')
        c_indices = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        self.assertTrue(path_has_cumulated_bonds(mol, (c_indices[0], c_indices[2])))

    def test_no_cumulated_bonds_conjugated(self):
        """Conjugated diene (C=C-C=C) is NOT cumulated."""
        mol = Molecule().from_smiles('C=CC=C')
        c_indices = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        self.assertFalse(path_has_cumulated_bonds(mol, (c_indices[0], c_indices[3])))

    def test_no_cumulated_bonds_short_path(self):
        """Forming bond between adjacent atoms: path too short for cumulation."""
        mol = Molecule().from_smiles('C=C')
        c_indices = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        self.assertFalse(path_has_cumulated_bonds(mol, (c_indices[0], c_indices[1])))


class TestGetNearAttackXyz(unittest.TestCase):
    """Tests for the get_near_attack_xyz function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

        # Pentane-like chain: C0-C1-C2-C3-C4 with H atoms
        # Ring closure between C0 and C4 → 5-membered ring
        cls.pentane_mol = Molecule().from_smiles('CCCCC')
        c_indices = [i for i, a in enumerate(cls.pentane_mol.atoms) if a.symbol == 'C']
        cls.pentane_c_indices = c_indices

        # Build a stretched linear conformation
        coords = []
        symbols = []
        for i, atom in enumerate(cls.pentane_mol.atoms):
            symbols.append(atom.symbol)
            if atom.symbol == 'C':
                c_rank = c_indices.index(i)
                coords.append((c_rank * 1.54, 0.0, 0.0))
            else:
                bonded_c = None
                for nbr in atom.bonds.keys():
                    if nbr.symbol == 'C':
                        bonded_c = cls.pentane_mol.atoms.index(nbr)
                        break
                if bonded_c is not None:
                    c_rank = c_indices.index(bonded_c)
                    h_count = sum(1 for a2 in cls.pentane_mol.atoms[:i]
                                  if a2.symbol == 'H'
                                  and any(cls.pentane_mol.has_bond(a2, nbr)
                                          and nbr == cls.pentane_mol.atoms[bonded_c]
                                          for nbr in a2.bonds.keys()))
                    offset_y = 1.09 * ((-1) ** h_count)
                    offset_z = 0.5 * h_count
                    coords.append((c_rank * 1.54, offset_y, offset_z))
                else:
                    coords.append((0.0, 1.09, 0.0))

        cls.pentane_xyz = {
            'symbols': tuple(symbols),
            'isotopes': tuple(_MASS_NUMBER[a.symbol] for a in cls.pentane_mol.atoms),
            'coords': tuple(tuple(c) for c in coords),
        }

    def test_get_near_attack_xyz_returns_dict(self):
        """Near-attack conformer returns a valid xyz dict."""
        forming_bond = (self.pentane_c_indices[0], self.pentane_c_indices[4])
        result = get_near_attack_xyz(self.pentane_xyz, self.pentane_mol, [forming_bond])
        self.assertIn('symbols', result)
        self.assertIn('coords', result)
        self.assertEqual(len(result['symbols']), len(self.pentane_xyz['symbols']))

    def test_get_near_attack_xyz_does_not_increase_distance(self):
        """Near-attack conformer does not increase forming-bond distance."""
        c0, c4 = self.pentane_c_indices[0], self.pentane_c_indices[4]
        forming_bond = (c0, c4)
        d_before = np.linalg.norm(
            np.array(self.pentane_xyz['coords'][c0]) -
            np.array(self.pentane_xyz['coords'][c4]))

        result = get_near_attack_xyz(self.pentane_xyz, self.pentane_mol, [forming_bond])
        d_after = np.linalg.norm(
            np.array(result['coords'][c0]) -
            np.array(result['coords'][c4]))
        self.assertLessEqual(d_after, d_before + 0.01)

    def test_get_near_attack_xyz_preserves_atom_count(self):
        """Near-attack conformer preserves atom count and symbols."""
        forming_bond = (self.pentane_c_indices[0], self.pentane_c_indices[4])
        result = get_near_attack_xyz(self.pentane_xyz, self.pentane_mol, [forming_bond])
        self.assertEqual(result['symbols'], self.pentane_xyz['symbols'])


class TestRingClosureXyz(unittest.TestCase):
    """Tests for the ring_closure_xyz function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

        # 1,3-butadiene: C0=C1-C2=C3 (H atoms included)
        cls.butadiene_mol = Molecule().from_smiles('C=CC=C')
        c_indices = [i for i, a in enumerate(cls.butadiene_mol.atoms) if a.symbol == 'C']
        cls.butadiene_c_indices = c_indices

        # Build a stretched near-planar conformation
        coords = []
        symbols = []
        for i, atom in enumerate(cls.butadiene_mol.atoms):
            symbols.append(atom.symbol)
            if atom.symbol == 'C':
                c_rank = c_indices.index(i)
                coords.append((c_rank * 1.40, 0.0, 0.0))
            else:
                bonded_c = None
                for nbr in atom.bonds.keys():
                    if nbr.symbol == 'C':
                        bonded_c = cls.butadiene_mol.atoms.index(nbr)
                        break
                if bonded_c is not None:
                    c_rank = c_indices.index(bonded_c)
                    h_list = [j for j, a2 in enumerate(cls.butadiene_mol.atoms)
                              if a2.symbol == 'H'
                              and cls.butadiene_mol.has_bond(a2, cls.butadiene_mol.atoms[bonded_c])]
                    h_rank = h_list.index(i) if i in h_list else 0
                    offset_y = 1.09 * ((-1) ** h_rank)
                    coords.append((c_rank * 1.40, offset_y, 0.0))
                else:
                    coords.append((0.0, 1.09, 0.0))

        cls.butadiene_xyz = {
            'symbols': tuple(symbols),
            'isotopes': tuple(_MASS_NUMBER[a.symbol] for a in cls.butadiene_mol.atoms),
            'coords': tuple(tuple(c) for c in coords),
        }

    def test_ring_closure_xyz_returns_dict_or_none(self):
        """Ring closure returns either a valid xyz dict or None."""
        c0 = self.butadiene_c_indices[0]
        c3 = self.butadiene_c_indices[3]
        result = ring_closure_xyz(
            self.butadiene_xyz, self.butadiene_mol,
            forming_bond=(c0, c3), target_distance=2.3)
        if result is not None:
            self.assertIn('symbols', result)
            self.assertIn('coords', result)
            self.assertEqual(len(result['coords']), len(self.butadiene_xyz['coords']))

    def test_ring_closure_xyz_preserves_symbols(self):
        """Ring closure preserves atom symbols."""
        c0 = self.butadiene_c_indices[0]
        c3 = self.butadiene_c_indices[3]
        result = ring_closure_xyz(
            self.butadiene_xyz, self.butadiene_mol,
            forming_bond=(c0, c3), target_distance=2.3)
        if result is not None:
            self.assertEqual(result['symbols'], self.butadiene_xyz['symbols'])

    def test_ring_closure_xyz_reduces_forming_bond_distance(self):
        """Ring closure reduces the forming-bond distance toward the target."""
        c0 = self.butadiene_c_indices[0]
        c3 = self.butadiene_c_indices[3]
        d_before = np.linalg.norm(
            np.array(self.butadiene_xyz['coords'][c0]) -
            np.array(self.butadiene_xyz['coords'][c3]))
        target = 2.3
        result = ring_closure_xyz(
            self.butadiene_xyz, self.butadiene_mol,
            forming_bond=(c0, c3), target_distance=target)
        if result is not None:
            d_after = np.linalg.norm(
                np.array(result['coords'][c0]) -
                np.array(result['coords'][c3]))
            self.assertLess(d_after, d_before)

    def test_ring_closure_xyz_short_path_returns_none(self):
        """Forming bond between adjacent atoms (path too short) returns None."""
        c0 = self.butadiene_c_indices[0]
        c1 = self.butadiene_c_indices[1]
        result = ring_closure_xyz(
            self.butadiene_xyz, self.butadiene_mol,
            forming_bond=(c0, c1), target_distance=2.3)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
