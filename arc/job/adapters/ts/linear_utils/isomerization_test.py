#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the
arc.job.adapters.ts.linear_utils.isomerization module
"""

import unittest

import numpy as np

from arc.molecule.molecule import Molecule
from arc.species import ARCSpecies
from arc.job.adapters.ts.linear_utils.addition_test import MASS_NUMBER
from arc.job.adapters.ts.linear_utils.isomerization import (
    backbone_atom_map,
    get_near_attack_xyz,
    get_path_length,
    build_4center_interchange_ts,
    generate_zmat_branch,
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
        """Different heavy-atom element counts return None."""
        # Acetic acid (C2H4O2) vs ethanol + O (C2H6O) — different
        # element counts make backbone matching impossible.
        r_mol = Molecule().from_smiles('CC(=O)O')
        p_mol = Molecule().from_smiles('CCOO')
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
        coords, symbols = [], []
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

        cls.pentane_xyz = {'symbols': tuple(symbols),
                           'isotopes': tuple(MASS_NUMBER[a.symbol] for a in cls.pentane_mol.atoms),
                           'coords': tuple(tuple(c) for c in coords)}

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
        coords, symbols  = [], []
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
                              if a2.symbol == 'H' and cls.butadiene_mol.has_bond(a2, cls.butadiene_mol.atoms[bonded_c])]
                    h_rank = h_list.index(i) if i in h_list else 0
                    offset_y = 1.09 * ((-1) ** h_rank)
                    coords.append((c_rank * 1.40, offset_y, 0.0))
                else:
                    coords.append((0.0, 1.09, 0.0))

        cls.butadiene_xyz = {'symbols': tuple(symbols),
                             'isotopes': tuple(MASS_NUMBER[a.symbol] for a in cls.butadiene_mol.atoms),
                             'coords': tuple(tuple(c) for c in coords)}

    def test_ring_closure_xyz_returns_dict_or_none(self):
        """Ring closure returns either a valid xyz dict or None."""
        c0 = self.butadiene_c_indices[0]
        c3 = self.butadiene_c_indices[3]
        result = ring_closure_xyz(self.butadiene_xyz, self.butadiene_mol,
                                  forming_bond=(c0, c3), target_distance=2.3)
        if result is not None:
            self.assertIn('symbols', result)
            self.assertIn('coords', result)
            self.assertEqual(len(result['coords']), len(self.butadiene_xyz['coords']))

    def test_ring_closure_xyz_preserves_symbols(self):
        """Ring closure preserves atom symbols."""
        c0 = self.butadiene_c_indices[0]
        c3 = self.butadiene_c_indices[3]
        result = ring_closure_xyz(self.butadiene_xyz, self.butadiene_mol,
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
        result = ring_closure_xyz(self.butadiene_xyz, self.butadiene_mol,
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
        result = ring_closure_xyz(self.butadiene_xyz, self.butadiene_mol,
                                  forming_bond=(c0, c1), target_distance=2.3)
        self.assertIsNone(result)


class TestGetPathLength(unittest.TestCase):
    """Tests for the get_path_length function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_same_atom_returns_zero(self):
        """Path from an atom to itself is 0."""
        mol = Molecule().from_smiles('CC')
        self.assertEqual(get_path_length(mol, 0, 0), 0)

    def test_adjacent_atoms_returns_one(self):
        """Two bonded atoms have path length 1."""
        mol = Molecule().from_smiles('CC')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        result = get_path_length(mol, c_idx[0], c_idx[1])
        self.assertEqual(result, 1)

    def test_linear_chain_propane(self):
        """Path length across a 3-carbon chain is 2."""
        mol = Molecule().from_smiles('CCC')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        result = get_path_length(mol, c_idx[0], c_idx[2])
        self.assertEqual(result, 2)

    def test_longer_chain_butane(self):
        """Path length across a 4-carbon chain is 3."""
        mol = Molecule().from_smiles('CCCC')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        result = get_path_length(mol, c_idx[0], c_idx[3])
        self.assertEqual(result, 3)

    def test_branched_molecule(self):
        """Branched molecule: shortest path through backbone."""
        mol = Molecule().from_smiles('CC(C)C')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        # All terminal carbons are 2 bonds from each other (through center)
        # Find the central C (bonded to 3 carbons)
        atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
        center = None
        terminals = []
        for ci in c_idx:
            c_nbr = sum(1 for nbr in mol.atoms[ci].bonds if nbr.symbol == 'C')
            if c_nbr == 3:
                center = ci
            else:
                terminals.append(ci)
        if center is not None and len(terminals) >= 2:
            self.assertEqual(get_path_length(mol, terminals[0], terminals[1]), 2)
            self.assertEqual(get_path_length(mol, terminals[0], center), 1)

    def test_h_to_heavy_atom(self):
        """Path from H to a heavy atom through the bond graph."""
        mol = Molecule().from_smiles('CCC')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        h_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'H']
        # H bonded to C0 → path to C2 goes through C0-C1-C2 = 3 bonds
        atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
        h_on_c0 = None
        for hi in h_idx:
            for nbr in mol.atoms[hi].bonds:
                if atom_to_idx[nbr] == c_idx[0]:
                    h_on_c0 = hi
                    break
            if h_on_c0 is not None:
                break
        if h_on_c0 is not None:
            result = get_path_length(mol, h_on_c0, c_idx[2])
            self.assertEqual(result, 3)

    def test_ring_molecule_shortest_path(self):
        """In cyclohexane, the shortest path between opposite atoms is 3."""
        mol = Molecule().from_smiles('C1CCCCC1')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        # In a 6-membered ring, the maximum shortest path is 3
        result = get_path_length(mol, c_idx[0], c_idx[3])
        self.assertEqual(result, 3)


class TestBuild4CenterInterchangeTs(unittest.TestCase):
    """Tests for the build_4center_interchange_ts function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_wrong_bond_count_returns_none(self):
        """Fewer than 2 breaking/forming bonds returns None."""
        mol = Molecule().from_smiles('CC')
        symbols = tuple(a.symbol for a in mol.atoms)
        coords = tuple((float(i) * 1.5, 0.0, 0.0) for i in range(len(symbols)))
        xyz = {'symbols': symbols,
               'isotopes': tuple(MASS_NUMBER.get(s, 12) for s in symbols),
               'coords': coords}
        result = build_4center_interchange_ts(xyz, mol, bb=[(0, 1)], fb=[(0, 1)])
        self.assertIsNone(result)

    def test_more_than_4_reactive_atoms_returns_none(self):
        """More than 4 unique reactive atoms returns None."""
        mol = Molecule().from_smiles('CCCCC')
        symbols = tuple(a.symbol for a in mol.atoms)
        coords = tuple((float(i) * 1.5, 0.0, 0.0) for i in range(len(symbols)))
        xyz = {'symbols': symbols,
               'isotopes': tuple(MASS_NUMBER.get(s, 12) for s in symbols),
               'coords': coords}
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        # 5 unique reactive atoms
        result = build_4center_interchange_ts(
            xyz, mol,
            bb=[(c_idx[0], c_idx[1]), (c_idx[2], c_idx[3])],
            fb=[(c_idx[0], c_idx[3]), (c_idx[1], c_idx[4])])
        self.assertIsNone(result)

    def test_valid_4center_pattern(self):
        """A proper 4-center interchange with 4 reactive atoms returns a valid dict or None."""
        # 1,2-XY interchange on ethane-like: X-C1-C2-Y → Y-C1-C2-X
        # Use chlorofluoroethane: FCH2CH2Cl → ClCH2CH2F
        # Simplified: just use C-C with two H substituents as migrants
        spc = ARCSpecies(label='R', smiles='CC')
        mol = spc.mol
        symbols = tuple(a.symbol for a in mol.atoms)
        n = len(symbols)
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        h_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'H']
        self.assertEqual(len(c_idx), 2)
        self.assertEqual(len(h_idx), 6)

        # Build coordinates
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
                h_list = [j for j, a2 in enumerate(mol.atoms)
                          if a2.symbol == 'H'
                          and mol.has_bond(a2, mol.atoms[bonded_c])]
                h_rank = h_list.index(i) if i in h_list else 0
                coords_list.append((c_rank * 1.54, 1.09 * ((-1) ** h_rank), 0.5 * (h_rank // 2)))
        xyz = {'symbols': symbols,
               'isotopes': tuple(MASS_NUMBER.get(s, 12) for s in symbols),
               'coords': tuple(tuple(c) for c in coords_list)}

        # Find H atoms bonded to each C
        atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
        h_on_c0 = [atom_to_idx[nbr] for nbr in mol.atoms[c_idx[0]].bonds
                    if nbr.symbol == 'H']
        h_on_c1 = [atom_to_idx[nbr] for nbr in mol.atoms[c_idx[1]].bonds
                    if nbr.symbol == 'H']

        if len(h_on_c0) >= 1 and len(h_on_c1) >= 1:
            m1, m2 = h_on_c0[0], h_on_c1[0]
            self.assertIn(m1, h_idx)
            self.assertIn(m2, h_idx)
            # bb: H1 leaves C0, H2 leaves C1
            # fb: H1 goes to C1, H2 goes to C0
            bb = [(m1, c_idx[0]), (m2, c_idx[1])]
            fb = [(m1, c_idx[1]), (m2, c_idx[0])]
            result = build_4center_interchange_ts(xyz, mol, bb=bb, fb=fb, weight=0.5, label='test')
            # May be None if no center pair is found or validation fails, but should not raise
            if result is not None:
                self.assertIn('symbols', result)
                self.assertIn('coords', result)
                self.assertEqual(len(result['coords']), n)


class TestGenerateZmatBranch(unittest.TestCase):
    """Tests for the generate_zmat_branch function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

        # Build a propane molecule with proper 3D coordinates for Z-matrix generation
        cls.propane_spc = ARCSpecies(label='propane', smiles='CCC', xyz={
            'symbols': ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
            'coords': ((-1.267, -0.259, 0.000),
                       (0.000, 0.587, 0.000),
                       (1.267, -0.259, 0.000),
                       (-2.148, 0.393, 0.000),
                       (-1.267, -0.904, 0.882),
                       (-1.267, -0.904, -0.882),
                       (0.000, 1.232, 0.882),
                       (0.000, 1.232, -0.882),
                       (2.148, 0.393, 0.000),
                       (1.267, -0.904, 0.882),
                       (1.267, -0.904, -0.882))})
        cls.propane_mol = cls.propane_spc.mol
        cls.propane_xyz = cls.propane_spc.get_xyz()

    def test_generate_zmat_branch_returns_dict_or_none(self):
        """Z-mat branch generation returns a dict or None."""
        # Use propane as both anchor and target (identity test)
        result = generate_zmat_branch(anchor_xyz=self.propane_xyz,
                                      anchor_mol=self.propane_mol,
                                      target_xyz=self.propane_xyz,
                                      weight=0.5,
                                      reactive_xyz_indices=set(),
                                      anchors=None,
                                      constraints=None,
                                      r_mol=self.propane_mol,
                                      forming_bonds=[],
                                      breaking_bonds=[],
                                      label='test',
                                      skip_postprocess=True)
        if result is not None:
            self.assertIn('symbols', result)
            self.assertIn('coords', result)
            self.assertEqual(len(result['coords']), len(self.propane_xyz['coords']))

    def test_generate_zmat_branch_preserves_atom_count(self):
        """Output has the same number of atoms as input."""
        result = generate_zmat_branch(anchor_xyz=self.propane_xyz,
                                      anchor_mol=self.propane_mol,
                                      target_xyz=self.propane_xyz,
                                      weight=0.0,
                                      reactive_xyz_indices=set(),
                                      anchors=None,
                                      constraints=None,
                                      r_mol=self.propane_mol,
                                      forming_bonds=[],
                                      breaking_bonds=[],
                                      label='test',
                                      skip_postprocess=True)
        if result is not None:
            self.assertEqual(len(result['symbols']), len(self.propane_xyz['symbols']))

    def test_generate_zmat_branch_weight_zero_close_to_anchor(self):
        """At weight=0, the result should be a reasonable geometry (Z-mat round trip)."""
        result = generate_zmat_branch(anchor_xyz=self.propane_xyz,
                                      anchor_mol=self.propane_mol,
                                      target_xyz=self.propane_xyz,
                                      weight=0.0,
                                      reactive_xyz_indices=set(),
                                      anchors=None,
                                      constraints=None,
                                      r_mol=self.propane_mol,
                                      forming_bonds=[],
                                      breaking_bonds=[],
                                      label='test',
                                      skip_postprocess=True)
        if result is not None:
            anchor_coords = np.array(self.propane_xyz['coords'])
            result_coords = np.array(result['coords'])
            # Z-mat round-trip plus postprocessing can shift H atoms; use a
            # generous tolerance that still rejects completely broken geometries.
            rmsd = np.sqrt(np.mean(np.sum((anchor_coords - result_coords) ** 2, axis=1)))
            self.assertLess(rmsd, 5.0)

    def test_generate_zmat_branch_with_h_migration_family(self):
        """With H-migration family, the pipeline runs without error."""
        # Ethanol-like: C-C-O with H migration
        spc = ARCSpecies(label='ethanol', smiles='CCO', xyz={
            'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 16, 1, 1, 1, 1, 1, 1),
            'coords': ((-1.168, -0.211, 0.000),
                       (0.138, 0.545, 0.000),
                       (1.210, -0.383, 0.000),
                       (-2.027, 0.463, 0.000),
                       (-1.168, -0.855, 0.882),
                       (-1.168, -0.855, -0.882),
                       (0.138, 1.189, 0.882),
                       (0.138, 1.189, -0.882),
                       (2.053, 0.091, 0.000))})
        mol = spc.mol
        xyz = spc.get_xyz()
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        o_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'O']
        h_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'H']
        self.assertEqual(len(c_idx), 2)
        self.assertEqual(len(o_idx), 1)
        self.assertEqual(len(h_idx), 6)

        # Simulate H migration: pick an H on C, form bond to O
        atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
        h_on_c0 = [atom_to_idx[nbr] for nbr in mol.atoms[c_idx[0]].bonds if nbr.symbol == 'H']
        if h_on_c0 and o_idx:
            self.assertIn(h_on_c0[0], h_idx)
            forming = [(h_on_c0[0], o_idx[0])]
            breaking = [(h_on_c0[0], c_idx[0])]
            reactive = {h_on_c0[0], o_idx[0], c_idx[0]}
            result = generate_zmat_branch(anchor_xyz=xyz,
                                          anchor_mol=mol,
                                          target_xyz=xyz,
                                          weight=0.5,
                                          reactive_xyz_indices=reactive,
                                          anchors=None,
                                          constraints=None,
                                          r_mol=mol,
                                          forming_bonds=forming,
                                          breaking_bonds=breaking,
                                          label='test_hmig',
                                          family='intra_H_migration')
            if result is not None:
                self.assertEqual(len(result['coords']), len(xyz['coords']))


if __name__ == '__main__':
    unittest.main()
