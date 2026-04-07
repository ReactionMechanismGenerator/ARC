#!/usr/bin/env python3
# encoding: utf-8

"""Phase 3b unit tests for local reactive-center geometry helpers."""

import unittest

import numpy as np

from arc.common import get_single_bond_length
from arc.species import ARCSpecies

from arc.job.adapters.ts.linear_utils.local_geometry import (
    clean_migrating_h,
    identify_h_migration_pairs,
    orient_h_away_from_axis,
    regularize_terminal_h_geometry,
    restore_terminal_h_symmetry,
)
from arc.job.adapters.ts.linear_utils.postprocess import PAULING_DELTA


class TestRegularizeTerminalHGeometry(unittest.TestCase):
    """``regularize_terminal_h_geometry`` snaps distorted terminal H
    bond lengths back to ``get_single_bond_length(C, H)`` while
    preserving direction."""

    def setUp(self):
        # Methane CH4 — center is C[0], 4 H atoms.
        self.sp = ARCSpecies(label='CH4', smiles='C')
        self.xyz = self.sp.get_xyz()
        self.mol = self.sp.mol
        # Identify the C atom and the H atoms.
        self.c_idx = next(i for i, s in enumerate(self.xyz['symbols']) if s == 'C')
        self.h_idxs = [i for i, s in enumerate(self.xyz['symbols']) if s == 'H']

    def test_no_op_when_within_range(self):
        """A fresh CH4 should not be touched."""
        out = regularize_terminal_h_geometry(self.xyz, self.mol, self.c_idx)
        self.assertIs(out, self.xyz)

    def test_pulls_overstretched_h_back(self):
        """An H atom stretched far past sbl is snapped back along its bond."""
        coords = list(map(list, self.xyz['coords']))
        h = self.h_idxs[0]
        c_pos = np.array(coords[self.c_idx])
        h_pos = np.array(coords[h])
        # Stretch this H to 2.0 Å along the same direction.
        direction = (h_pos - c_pos) / float(np.linalg.norm(h_pos - c_pos))
        coords[h] = (c_pos + direction * 2.0).tolist()
        bad = {**self.xyz, 'coords': tuple(tuple(row) for row in coords)}
        out = regularize_terminal_h_geometry(bad, self.mol, self.c_idx)
        sbl = get_single_bond_length('C', 'H')
        new_d = float(np.linalg.norm(
            np.array(out['coords'][h]) - np.array(out['coords'][self.c_idx])))
        self.assertAlmostEqual(new_d, sbl, places=4)

    def test_exclude_atoms_skips_listed_hydrogen(self):
        coords = list(map(list, self.xyz['coords']))
        h = self.h_idxs[0]
        c_pos = np.array(coords[self.c_idx])
        h_pos = np.array(coords[h])
        direction = (h_pos - c_pos) / float(np.linalg.norm(h_pos - c_pos))
        coords[h] = (c_pos + direction * 2.5).tolist()
        bad = {**self.xyz, 'coords': tuple(tuple(row) for row in coords)}
        out = regularize_terminal_h_geometry(
            bad, self.mol, self.c_idx, exclude_atoms={h})
        # Excluded H is not touched — its distance is still 2.5.
        self.assertAlmostEqual(
            float(np.linalg.norm(
                np.array(out['coords'][h]) - np.array(out['coords'][self.c_idx]))),
            2.5, places=4)


class TestOrientHAwayFromAxis(unittest.TestCase):
    """The blocking-H reorientation helper reflects an inward H to the
    opposite side of the donor–parent–acceptor axis when it sits in the
    blocking pocket."""

    def test_inward_h_is_reflected(self):
        """An H bonded to donor and sitting in the inward blocking
        pocket (small angle to the donor–acceptor axis AND close to the
        acceptor) is reflected across that axis to the outward side."""
        # Build a hand-crafted donor–acceptor pair with one H sitting
        # IN the blocking pocket: short distance to acceptor (< 1.60 Å)
        # AND small angle to the donor–acceptor axis (< 85°).
        xyz = {
            'symbols': ('C', 'C', 'H'),
            'isotopes': (12, 12, 1),
            'coords': (
                (0.0, 0.0, 0.0),     # donor
                (2.5, 0.0, 0.0),     # acceptor — closer than before
                (1.4, 0.10, 0.0),    # inward H: ~1.10 Å from acceptor
            ),
        }
        class _A:
            def __init__(self, sym, bonds):
                self.bonds = bonds
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
        c0 = _A('C', {})
        c1 = _A('C', {})
        h2 = _A('H', {})
        c0.bonds = {h2: None}
        h2.bonds = {c0: None}
        c1.bonds = {}
        class _M:
            atoms = [c0, c1, h2]

        out = orient_h_away_from_axis(xyz, _M(), donor=0, acceptor=1, exclude_h=set())
        h_after = np.array(out['coords'][2])
        # Reflected x component flips sign of the axis-parallel piece
        # (1.4 → -1.4 after subtracting 2 × 1.4 × axis_hat from h_vec).
        self.assertLess(h_after[0], 0.0)
        # Heavy atoms unchanged.
        self.assertEqual(out['coords'][0], xyz['coords'][0])
        self.assertEqual(out['coords'][1], xyz['coords'][1])

    def test_excluded_h_is_left_alone(self):
        xyz = {
            'symbols': ('C', 'C', 'H'),
            'isotopes': (12, 12, 1),
            'coords': (
                (0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.5, 0.05, 0.0),
            ),
        }
        class _A:
            def __init__(self, sym, bonds):
                self.bonds = bonds
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
        c0 = _A('C', {})
        c1 = _A('C', {})
        h2 = _A('H', {})
        c0.bonds = {h2: None}
        h2.bonds = {c0: None}
        class _M:
            atoms = [c0, c1, h2]
        out = orient_h_away_from_axis(
            xyz, _M(), donor=0, acceptor=1, exclude_h={2})
        # The migrating H (excluded) is untouched, so the result should be
        # identical to the input.
        self.assertEqual(out['coords'], xyz['coords'])

    def test_outward_h_is_left_alone(self):
        # H sitting OUTSIDE the blocking pocket — should not be moved.
        xyz = {
            'symbols': ('C', 'C', 'H'),
            'isotopes': (12, 12, 1),
            'coords': (
                (0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (-0.7, 0.7, 0.0),
            ),
        }
        class _A:
            def __init__(self, sym, bonds):
                self.bonds = bonds
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
        c0 = _A('C', {})
        c1 = _A('C', {})
        h2 = _A('H', {})
        c0.bonds = {h2: None}
        h2.bonds = {c0: None}
        class _M:
            atoms = [c0, c1, h2]
        out = orient_h_away_from_axis(
            xyz, _M(), donor=0, acceptor=1, exclude_h=set())
        self.assertEqual(out['coords'], xyz['coords'])


class TestCleanMigratingH(unittest.TestCase):
    """``clean_migrating_h`` re-places the migrating H at the donor–
    acceptor triangulated TS position and is idempotent."""

    def test_idempotent_on_triangulated_position(self):
        # Two heavy atoms 2.5 Å apart, an H roughly between them.
        xyz = {
            'symbols': ('C', 'N', 'H'),
            'isotopes': (12, 14, 1),
            'coords': (
                (0.0, 0.0, 0.0), (2.5, 0.0, 0.0), (1.0, 0.5, 0.0),
            ),
        }
        class _A:
            def __init__(self, sym):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = {}
        c = _A('C'); n = _A('N'); h = _A('H')
        class _M:
            atoms = [c, n, h]
        out = clean_migrating_h(xyz, _M(), donor=0, acceptor=1, h_idx=2)
        h_pos_1 = np.array(out['coords'][2])
        # Run again — should be a no-op (idempotent).
        out2 = clean_migrating_h(out, _M(), donor=0, acceptor=1, h_idx=2)
        h_pos_2 = np.array(out2['coords'][2])
        self.assertTrue(np.allclose(h_pos_1, h_pos_2, atol=1e-10))

    def test_h_distance_to_donor_matches_target(self):
        xyz = {
            'symbols': ('C', 'N', 'H'),
            'isotopes': (12, 14, 1),
            'coords': (
                (0.0, 0.0, 0.0), (2.5, 0.0, 0.0), (1.0, 0.5, 0.0),
            ),
        }
        class _A:
            def __init__(self, sym):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = {}
        c = _A('C'); n = _A('N'); h = _A('H')
        class _M:
            atoms = [c, n, h]
        out = clean_migrating_h(xyz, _M(), donor=0, acceptor=1, h_idx=2)
        h_pos = np.array(out['coords'][2])
        d_dh = float(np.linalg.norm(h_pos - np.array(out['coords'][0])))
        sbl_ch = get_single_bond_length('C', 'H')
        # The triangulation places H at distance d_DH = sbl + PAULING_DELTA from donor.
        self.assertAlmostEqual(d_dh, sbl_ch + PAULING_DELTA, places=4)


class TestRestoreTerminalHSymmetry(unittest.TestCase):
    """``restore_terminal_h_symmetry`` averages H positions about the
    parent–center axis when the center is clearly CH₂ or CH₃."""

    def test_ch3_distorted_h_is_resymmetrized(self):
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        # Find the first C and its H neighbors.
        c0 = next(i for i, s in enumerate(symbols) if s == 'C')
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        h_neighbors = [
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c0].bonds.keys()
            if nbr.element.symbol == 'H'
        ]
        # Distort one H by ~0.02 Å (well within the bail-out tolerance).
        coords = list(map(list, xyz['coords']))
        coords[h_neighbors[0]][1] += 0.02
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        out = restore_terminal_h_symmetry(bad, sp.mol, c0)
        new_coords = np.asarray(out['coords'])
        # The three H atoms around c0 should have nearly equal distances
        # to c0 after symmetrization.
        c_pos = new_coords[c0]
        dists = [float(np.linalg.norm(new_coords[h] - c_pos)) for h in h_neighbors]
        self.assertAlmostEqual(max(dists) - min(dists), 0.0, places=2)

    def test_skips_non_terminal_centers(self):
        # The central C of propane has 2 heavy neighbors → not a clear terminal.
        sp = ARCSpecies(label='propane', smiles='CCC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        # The middle C is the one with 2 C neighbors.
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        middle_c = None
        for atom in sp.mol.atoms:
            if atom.element.symbol == 'C':
                heavy_nbrs = sum(1 for n in atom.bonds.keys() if n.element.symbol != 'H')
                if heavy_nbrs == 2:
                    middle_c = atom_to_idx[atom]
                    break
        self.assertIsNotNone(middle_c)
        out = restore_terminal_h_symmetry(xyz, sp.mol, middle_c)
        # No-op: the function returns the original input.
        self.assertIs(out, xyz)

    def test_does_not_introduce_h_crowding(self):
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        out = restore_terminal_h_symmetry(
            xyz, sp.mol,
            next(i for i, s in enumerate(xyz['symbols']) if s == 'C'),
        )
        # Verify pairwise H distances are not closer than 1.5 Å.
        coords = np.asarray(out['coords'])
        h_idxs = [i for i, s in enumerate(out['symbols']) if s == 'H']
        for i in range(len(h_idxs)):
            for j in range(i + 1, len(h_idxs)):
                d = float(np.linalg.norm(coords[h_idxs[i]] - coords[h_idxs[j]]))
                self.assertGreater(d, 1.50)


class TestIdentifyHMigrationPairs(unittest.TestCase):
    """``identify_h_migration_pairs`` recovers (h, donor, acceptor)
    triples mirroring ``migrate_verified_atoms``'s internal selection."""

    def test_returns_cross_bond_acceptor_when_available(self):
        # Mock: H4 is the migrating atom, donor is C0, acceptor is N1.
        xyz = {
            'symbols': ('C', 'N', 'C', 'H', 'H'),
            'isotopes': (12, 14, 12, 1, 1),
            'coords': (
                (0.0, 0.0, 0.0),     # C0 (donor, in large)
                (2.5, 0.0, 0.0),     # N1 (acceptor, in core)
                (1.0, 1.5, 0.0),     # C2 (in core)
                (0.5, -1.0, 0.0),    # H3 (non-migrating, on C0)
                (1.0, 0.5, 0.0),     # H4 (migrating)
            ),
        }
        class _A:
            def __init__(self, sym, bonds):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = bonds
        c0 = _A('C', {}); n1 = _A('N', {}); c2 = _A('C', {})
        h3 = _A('H', {}); h4 = _A('H', {})
        c0.bonds = {h3: None, h4: None}
        h3.bonds = {c0: None}
        h4.bonds = {c0: None}
        class _M:
            atoms = [c0, n1, c2, h3, h4]
        recs = identify_h_migration_pairs(
            xyz, _M(), migrating_atoms={4},
            core={1, 2}, large_prod_atoms={0, 3},
            cross_bonds=[(4, 1)],
        )
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]['h_idx'], 4)
        self.assertEqual(recs[0]['donor'], 0)
        self.assertEqual(recs[0]['acceptor'], 1)
        self.assertEqual(recs[0]['source'], 'cross_bond')

    def test_falls_back_to_nearest_core_when_no_cross_bond(self):
        xyz = {
            'symbols': ('C', 'N', 'C', 'H'),
            'isotopes': (12, 14, 12, 1),
            'coords': (
                (0.0, 0.0, 0.0), (2.5, 0.0, 0.0),
                (1.0, 1.5, 0.0), (1.0, 0.5, 0.0),
            ),
        }
        class _A:
            def __init__(self, sym, bonds):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = bonds
        c0 = _A('C', {}); n1 = _A('N', {}); c2 = _A('C', {}); h3 = _A('H', {})
        c0.bonds = {h3: None}
        h3.bonds = {c0: None}
        class _M:
            atoms = [c0, n1, c2, h3]
        recs = identify_h_migration_pairs(
            xyz, _M(), migrating_atoms={3},
            core={1, 2}, large_prod_atoms={0}, cross_bonds=None,
        )
        self.assertEqual(len(recs), 1)
        # H3 is closest to C2 (1.0 Å away), not N1 (2.0 Å away).
        self.assertEqual(recs[0]['acceptor'], 2)
        self.assertEqual(recs[0]['source'], 'nearest_core')


if __name__ == '__main__':
    unittest.main()
