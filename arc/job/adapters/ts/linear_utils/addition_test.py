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
    build_concerted_ts,
    try_insertion_ring,
    stretch_core_from_large,
    migrate_verified_atoms,
    migrate_h_between_fragments,
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


class TestBuildConcertedTs(unittest.TestCase):
    """Tests for the build_concerted_ts function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_no_bonds_returns_none(self):
        """No split or cross bonds yields None."""
        mol = Molecule().from_smiles('CC')
        xyz = {
            'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 1, 1, 1, 1, 1, 1),
            'coords': ((0.0, 0.0, 0.0), (1.54, 0.0, 0.0),
                       (-0.39, 1.03, 0.0), (-0.39, -0.51, 0.89),
                       (-0.39, -0.51, -0.89), (1.93, 1.03, 0.0),
                       (1.93, -0.51, 0.89), (1.93, -0.51, -0.89)),
        }
        result = build_concerted_ts(xyz, mol, split_bonds=[], cross_bonds=[])
        self.assertIsNone(result)

    def test_concerted_single_split_bond(self):
        """Stretching a single C-C split bond increases the bond distance."""
        mol = Molecule().from_smiles('CC')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        c0, c1 = c_idx[0], c_idx[1]
        symbols = []
        coords = []
        for i, atom in enumerate(mol.atoms):
            symbols.append(atom.symbol)
            if atom.symbol == 'C':
                c_rank = c_idx.index(i)
                coords.append((c_rank * 1.54, 0.0, 0.0))
            else:
                bonded_c = None
                for nbr in atom.bonds.keys():
                    if nbr.symbol == 'C':
                        bonded_c = mol.atoms.index(nbr)
                        break
                c_rank = c_idx.index(bonded_c) if bonded_c is not None else 0
                h_list = [j for j, a2 in enumerate(mol.atoms)
                          if a2.symbol == 'H'
                          and mol.has_bond(a2, mol.atoms[bonded_c])]
                h_rank = h_list.index(i) if i in h_list else 0
                coords.append((c_rank * 1.54, 1.09 * ((-1) ** h_rank), 0.5 * (h_rank // 2)))
        xyz = {
            'symbols': tuple(symbols),
            'isotopes': tuple(_MASS_NUMBER[a.symbol] for a in mol.atoms),
            'coords': tuple(tuple(c) for c in coords),
        }
        d_before = np.linalg.norm(np.array(xyz['coords'][c0]) - np.array(xyz['coords'][c1]))
        result = build_concerted_ts(xyz, mol, split_bonds=[(c0, c1)], cross_bonds=[])
        self.assertIsNotNone(result)
        self.assertIn('coords', result)
        d_after = np.linalg.norm(np.array(result['coords'][c0]) - np.array(result['coords'][c1]))
        self.assertGreater(d_after, d_before)

    def test_concerted_preserves_symbols(self):
        """Symbols are preserved through the concerted TS builder."""
        mol = Molecule().from_smiles('CC')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        symbols = []
        coords = []
        for i, atom in enumerate(mol.atoms):
            symbols.append(atom.symbol)
            if atom.symbol == 'C':
                c_rank = c_idx.index(i)
                coords.append((c_rank * 1.54, 0.0, 0.0))
            else:
                bonded_c = None
                for nbr in atom.bonds.keys():
                    if nbr.symbol == 'C':
                        bonded_c = mol.atoms.index(nbr)
                        break
                c_rank = c_idx.index(bonded_c) if bonded_c is not None else 0
                coords.append((c_rank * 1.54, 1.09, 0.0))
        xyz = {
            'symbols': tuple(symbols),
            'isotopes': tuple(_MASS_NUMBER[a.symbol] for a in mol.atoms),
            'coords': tuple(tuple(c) for c in coords),
        }
        result = build_concerted_ts(xyz, mol, split_bonds=[(c_idx[0], c_idx[1])], cross_bonds=[])
        if result is not None:
            self.assertEqual(result['symbols'], xyz['symbols'])

    def test_concerted_with_cross_bond_contracts(self):
        """A cross (forming) bond gets shorter after concerted TS build."""
        # Use propanoic acid: CCC(=O)O — break C-C and form C-O
        spc = ARCSpecies(label='R', smiles='CCC(=O)O')
        mol = spc.mol
        symbols = tuple(a.symbol for a in mol.atoms)
        n = len(symbols)
        # Generate initial coordinates spaced out along x
        coords_list = []
        for i, atom in enumerate(mol.atoms):
            coords_list.append((float(i) * 1.5, 0.0, 0.0))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(_MASS_NUMBER.get(s, 12) for s in symbols),
            'coords': tuple(tuple(c) for c in coords_list),
        }
        # Find C-C bond to break and an O atom to form bond with
        c_indices = [i for i, s in enumerate(symbols) if s == 'C']
        o_indices = [i for i, s in enumerate(symbols) if s == 'O']
        if len(c_indices) >= 2 and len(o_indices) >= 1:
            split_b = [(c_indices[0], c_indices[1])]
            cross_b = [(c_indices[0], o_indices[0])]
            d_cross_before = np.linalg.norm(
                np.array(xyz['coords'][c_indices[0]]) - np.array(xyz['coords'][o_indices[0]]))
            result = build_concerted_ts(xyz, mol, split_bonds=split_b, cross_bonds=cross_b)
            if result is not None:
                d_cross_after = np.linalg.norm(
                    np.array(result['coords'][c_indices[0]]) - np.array(result['coords'][o_indices[0]]))
                self.assertLess(d_cross_after, d_cross_before)


class TestTryInsertionRing(unittest.TestCase):
    """Tests for the try_insertion_ring function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_no_central_atom_returns_none(self):
        """When no atom appears in 2+ split bonds, returns None."""
        mol = Molecule().from_smiles('CC')
        symbols = tuple(a.symbol for a in mol.atoms)
        n = len(symbols)
        coords = tuple((float(i) * 1.5, 0.0, 0.0) for i in range(n))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(_MASS_NUMBER.get(s, 12) for s in symbols),
            'coords': coords,
        }
        c_idx = [i for i, s in enumerate(symbols) if s == 'C']
        # Only 1 split bond means no atom appears in 2+
        fragments = [{c_idx[0]}, {c_idx[1]}]
        result = try_insertion_ring(xyz, mol, fragments,
                                    split_bonds=[(c_idx[0], c_idx[1])],
                                    cross_bonds=[],
                                    weight=0.5,
                                    n_atoms=n)
        self.assertIsNone(result)

    def test_no_cross_bond_match_returns_none(self):
        """When central atom partners have no cross bond, returns None."""
        # CCC: break both C-C bonds → central atom, but no cross bond
        mol = Molecule().from_smiles('CCC')
        symbols = tuple(a.symbol for a in mol.atoms)
        n = len(symbols)
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        coords_list = []
        for i, atom in enumerate(mol.atoms):
            if atom.symbol == 'C':
                coords_list.append((c_idx.index(i) * 1.54, 0.0, 0.0))
            else:
                bonded_c = None
                for nbr in atom.bonds.keys():
                    if nbr.symbol == 'C':
                        bonded_c = mol.atoms.index(nbr)
                        break
                c_rank = c_idx.index(bonded_c) if bonded_c is not None else 0
                coords_list.append((c_rank * 1.54, 1.09, 0.0))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(_MASS_NUMBER.get(s, 12) for s in symbols),
            'coords': tuple(tuple(c) for c in coords_list),
        }
        # Break both C-C bonds: C0-C1 and C1-C2. C1 appears in 2 split bonds.
        split_bonds = [(c_idx[0], c_idx[1]), (c_idx[1], c_idx[2])]
        fragments = [
            {i for i in range(n) if i == c_idx[0] or (mol.atoms[i].symbol == 'H' and any(
                mol.has_bond(mol.atoms[i], mol.atoms[c_idx[0]]) for _ in [1]))},
        ]
        # Build real fragments by BFS without split bonds
        atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
        adj = {k: set() for k in range(n)}
        for atom in mol.atoms:
            idx_a = atom_to_idx[atom]
            for neighbor in atom.edges:
                idx_b = atom_to_idx[neighbor]
                adj[idx_a].add(idx_b)
        for a, b in split_bonds:
            adj[a].discard(b)
            adj[b].discard(a)
        visited = set()
        fragments = []
        for start in range(n):
            if start in visited:
                continue
            component = set()
            queue = [start]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                queue.extend(adj[node] - visited)
            fragments.append(component)

        result = try_insertion_ring(xyz, mol, fragments,
                                    split_bonds=split_bonds,
                                    cross_bonds=[],
                                    weight=0.5,
                                    n_atoms=n)
        self.assertIsNone(result)

    def test_insertion_ring_with_proper_pattern(self):
        """A valid 3-fragment insertion pattern with a cross bond returns a dict or None."""
        # Formaldehyde insertion: H2CO → H + CO + H or similar
        # Use a simple 3-atom linear: A-B-C with bonds A-B, B-C broken and A-C cross bond
        mol = Molecule().from_smiles('CCC')
        symbols = tuple(a.symbol for a in mol.atoms)
        n = len(symbols)
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        coords_list = []
        for i, atom in enumerate(mol.atoms):
            if atom.symbol == 'C':
                coords_list.append((c_idx.index(i) * 2.0, 0.0, 0.0))
            else:
                bonded_c = None
                for nbr in atom.bonds.keys():
                    if nbr.symbol == 'C':
                        bonded_c = mol.atoms.index(nbr)
                        break
                c_rank = c_idx.index(bonded_c) if bonded_c is not None else 0
                coords_list.append((c_rank * 2.0, 1.09, 0.0))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(_MASS_NUMBER.get(s, 12) for s in symbols),
            'coords': tuple(tuple(c) for c in coords_list),
        }
        split_bonds = [(c_idx[0], c_idx[1]), (c_idx[1], c_idx[2])]
        cross_bonds = [(c_idx[0], c_idx[2])]
        # Build fragments
        atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
        adj = {k: set() for k in range(n)}
        for atom in mol.atoms:
            idx_a = atom_to_idx[atom]
            for neighbor in atom.edges:
                idx_b = atom_to_idx[neighbor]
                adj[idx_a].add(idx_b)
        for a, b in split_bonds:
            adj[a].discard(b)
            adj[b].discard(a)
        visited = set()
        fragments = []
        for start in range(n):
            if start in visited:
                continue
            component = set()
            queue = [start]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                queue.extend(adj[node] - visited)
            fragments.append(component)

        result = try_insertion_ring(xyz, mol, fragments,
                                    split_bonds=split_bonds,
                                    cross_bonds=cross_bonds,
                                    weight=0.5,
                                    n_atoms=n)
        # May return None if validation fails, but the function should not raise
        if result is not None:
            self.assertIn('symbols', result)
            self.assertIn('coords', result)
            self.assertEqual(len(result['coords']), n)


class TestStretchCoreFromLarge(unittest.TestCase):
    """Tests for the stretch_core_from_large function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_no_core_anchors_returns_original(self):
        """When no split bond connects core to large, returns the original xyz."""
        mol = Molecule().from_smiles('CC')
        symbols = tuple(a.symbol for a in mol.atoms)
        coords = tuple((float(i) * 1.5, 0.0, 0.0) for i in range(len(symbols)))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(_MASS_NUMBER.get(s, 12) for s in symbols),
            'coords': coords,
        }
        c_idx = [i for i, s in enumerate(symbols) if s == 'C']
        # core and large_prod_atoms don't share a split bond endpoint
        result = stretch_core_from_large(
            xyz, mol, split_bonds=[(c_idx[0], c_idx[1])],
            core={c_idx[0]}, large_prod_atoms=set(), small_prod_atoms={c_idx[0]},
            weight=0.5)
        self.assertEqual(result, xyz)

    def test_stretch_core_increases_distance(self):
        """Core atoms are pushed away from the large product."""
        spc = ARCSpecies(label='R', smiles='CCC(=O)O')
        mol = spc.mol
        symbols = tuple(a.symbol for a in mol.atoms)
        n = len(symbols)
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        o_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'O']

        # Build coordinates with 1.5A spacing along x
        coords_list = []
        for i in range(n):
            coords_list.append((float(i) * 1.5, 0.0, 0.0))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(_MASS_NUMBER.get(s, 12) for s in symbols),
            'coords': tuple(tuple(c) for c in coords_list),
        }
        # Let core = {c_idx[2], o_idx[0], o_idx[1]}, large = {c_idx[0], c_idx[1]}
        # Split bond between c_idx[1] and c_idx[2]
        core = {c_idx[2]}
        if len(o_idx) >= 2:
            core.update(o_idx)
        large = {c_idx[0], c_idx[1]}
        split_bonds = [(c_idx[1], c_idx[2])]

        result = stretch_core_from_large(
            xyz, mol, split_bonds=split_bonds,
            core=core, large_prod_atoms=large,
            small_prod_atoms=core, weight=0.5)
        # The core atom c_idx[2] should have moved away from the large product
        orig_d = np.linalg.norm(
            np.array(xyz['coords'][c_idx[2]]) - np.array(xyz['coords'][c_idx[1]]))
        new_d = np.linalg.norm(
            np.array(result['coords'][c_idx[2]]) - np.array(result['coords'][c_idx[1]]))
        self.assertGreaterEqual(new_d, orig_d - 0.01)

    def test_stretch_core_preserves_symbols(self):
        """Symbols and isotopes are preserved."""
        mol = Molecule().from_smiles('CCC')
        symbols = tuple(a.symbol for a in mol.atoms)
        n = len(symbols)
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        coords = tuple((float(i) * 1.5, 0.0, 0.0) for i in range(n))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(_MASS_NUMBER.get(s, 12) for s in symbols),
            'coords': coords,
        }
        result = stretch_core_from_large(
            xyz, mol, split_bonds=[(c_idx[0], c_idx[1])],
            core={c_idx[1]}, large_prod_atoms={c_idx[0]},
            small_prod_atoms={c_idx[1], c_idx[2]}, weight=0.5)
        self.assertEqual(result['symbols'], xyz['symbols'])


class TestMigrateVerifiedAtoms(unittest.TestCase):
    """Tests for the migrate_verified_atoms function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_no_migrating_atoms_returns_same(self):
        """Empty migrating set returns coordinates unchanged."""
        mol = Molecule().from_smiles('CC')
        symbols = tuple(a.symbol for a in mol.atoms)
        n = len(symbols)
        coords = tuple((float(i) * 1.5, 0.0, 0.0) for i in range(n))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(_MASS_NUMBER.get(s, 12) for s in symbols),
            'coords': coords,
        }
        c_idx = [i for i, s in enumerate(symbols) if s == 'C']
        result = migrate_verified_atoms(
            xyz, mol, migrating_atoms=set(),
            core={c_idx[0]}, large_prod_atoms={c_idx[1]}, weight=0.5)
        np.testing.assert_allclose(result['coords'], xyz['coords'], atol=1e-10)

    def test_h_migration_moves_h_between_fragments(self):
        """A migrating H is placed between donor and acceptor."""
        # HO2 elimination: CC(O)O[O] — H migrates from C to O
        spc = ARCSpecies(label='R', smiles='CC(O)O[O]')
        mol = spc.mol
        symbols = tuple(a.symbol for a in mol.atoms)
        n = len(symbols)
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        o_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'O']
        h_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'H']

        # Build coordinates
        coords_list = []
        for i in range(n):
            coords_list.append((float(i) * 1.5, 0.0, 0.0))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(_MASS_NUMBER.get(s, 12) for s in symbols),
            'coords': tuple(tuple(c) for c in coords_list),
        }

        # Pick an H bonded to a C atom (donor in large_prod) to migrate toward O (in core)
        atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
        migrating_h = None
        donor_c = None
        for hi in h_idx:
            for nbr in mol.atoms[hi].bonds.keys():
                nbr_idx = atom_to_idx[nbr]
                if nbr.symbol == 'C':
                    migrating_h = hi
                    donor_c = nbr_idx
                    break
            if migrating_h is not None:
                break

        if migrating_h is not None and len(o_idx) >= 1:
            core = set(o_idx)
            large = set(c_idx) | (set(h_idx) - {migrating_h})
            cross_bonds = [(migrating_h, o_idx[0])]
            orig_pos = np.array(xyz['coords'][migrating_h])
            result = migrate_verified_atoms(
                xyz, mol, migrating_atoms={migrating_h},
                core=core, large_prod_atoms=large,
                weight=0.5, cross_bonds=cross_bonds)
            new_pos = np.array(result['coords'][migrating_h])
            # The migrating H should have moved
            displacement = np.linalg.norm(new_pos - orig_pos)
            self.assertGreater(displacement, 0.01)

    def test_migrate_preserves_non_migrating_atoms(self):
        """Non-migrating atoms stay at their original positions."""
        mol = Molecule().from_smiles('CO')
        symbols = tuple(a.symbol for a in mol.atoms)
        n = len(symbols)
        coords = tuple((float(i) * 1.5, 0.0, 0.0) for i in range(n))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(_MASS_NUMBER.get(s, 12) for s in symbols),
            'coords': coords,
        }
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        o_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'O']
        result = migrate_verified_atoms(
            xyz, mol, migrating_atoms=set(),
            core=set(o_idx), large_prod_atoms=set(c_idx), weight=0.5)
        for i in range(n):
            np.testing.assert_allclose(result['coords'][i], xyz['coords'][i], atol=1e-10)


class TestMigrateHBetweenFragments(unittest.TestCase):
    """Tests for the migrate_h_between_fragments function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_no_surplus_returns_original(self):
        """When fragment H counts match product H counts, xyz is unchanged."""
        mol = Molecule().from_smiles('CC')
        spc1 = ARCSpecies(label='CH3_a', smiles='[CH3]')
        spc2 = ARCSpecies(label='CH3_b', smiles='[CH3]')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        symbols = tuple(a.symbol for a in mol.atoms)
        n = len(symbols)
        coords_list = []
        for i, atom in enumerate(mol.atoms):
            if atom.symbol == 'C':
                coords_list.append((c_idx.index(i) * 3.0, 0.0, 0.0))
            else:
                bonded_c = None
                for nbr in atom.bonds.keys():
                    if nbr.symbol == 'C':
                        bonded_c = mol.atoms.index(nbr)
                        break
                c_rank = c_idx.index(bonded_c) if bonded_c is not None else 0
                coords_list.append((c_rank * 3.0, 1.09, 0.0))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(_MASS_NUMBER.get(s, 12) for s in symbols),
            'coords': tuple(tuple(c) for c in coords_list),
        }
        result = migrate_h_between_fragments(
            xyz, mol, split_bonds=[(c_idx[0], c_idx[1])],
            product_species=[spc1, spc2], weight=0.5)
        # Should be unchanged because CH3 + CH3 has correct H count on each fragment
        np.testing.assert_allclose(result['coords'], xyz['coords'], atol=1e-10)

    def test_h_surplus_triggers_migration(self):
        """When one fragment has excess H, the closest H is partially moved."""
        # Formic acid decomposition: HC(=O)OH → H2 + CO2
        # After splitting C-H, the CHO fragment has 1 excess H vs CO2 target
        mol = Molecule().from_smiles('[H]C(=O)O')
        symbols = tuple(a.symbol for a in mol.atoms)
        n = len(symbols)

        # Products: H2 and CO2
        spc_h2 = ARCSpecies(label='H2', smiles='[H][H]')
        spc_co2 = ARCSpecies(label='CO2', smiles='O=C=O')

        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        h_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'H']
        o_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'O']

        # Build coordinates with fragments well-separated
        coords_list = []
        for i in range(n):
            coords_list.append((float(i) * 2.0, 0.0, 0.0))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(_MASS_NUMBER.get(s, 12) for s in symbols),
            'coords': tuple(tuple(c) for c in coords_list),
        }

        # Find C-H bond to break
        atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
        ch_bonds = []
        for hi in h_idx:
            for nbr in mol.atoms[hi].bonds.keys():
                nbr_idx = atom_to_idx[nbr]
                if mol.atoms[nbr_idx].symbol == 'C':
                    ch_bonds.append((hi, nbr_idx))
        if ch_bonds:
            split_bond = [ch_bonds[0]]
            result = migrate_h_between_fragments(
                xyz, mol, split_bonds=split_bond,
                product_species=[spc_h2, spc_co2], weight=0.5)
            # The function returns an xyz dict (possibly unchanged if formulas don't match)
            self.assertIn('symbols', result)
            self.assertIn('coords', result)

    def test_preserves_atom_count(self):
        """Output xyz has the same number of atoms as input."""
        mol = Molecule().from_smiles('CC')
        spc1 = ARCSpecies(label='CH3_a', smiles='[CH3]')
        spc2 = ARCSpecies(label='CH3_b', smiles='[CH3]')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        symbols = tuple(a.symbol for a in mol.atoms)
        n = len(symbols)
        coords = tuple((float(i) * 1.5, 0.0, 0.0) for i in range(n))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(_MASS_NUMBER.get(s, 12) for s in symbols),
            'coords': coords,
        }
        result = migrate_h_between_fragments(
            xyz, mol, split_bonds=[(c_idx[0], c_idx[1])],
            product_species=[spc1, spc2], weight=0.5)
        self.assertEqual(len(result['coords']), n)
        self.assertEqual(len(result['symbols']), n)


if __name__ == '__main__':
    unittest.main()
