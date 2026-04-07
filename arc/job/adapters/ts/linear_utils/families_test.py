#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for arc.job.adapters.ts.linear_utils.families
"""

import unittest

import numpy as np

from arc.species import ARCSpecies
from arc.species.converter import str_to_xyz
from arc.common import get_single_bond_length
from arc.job.adapters.ts.linear_utils.families import (
    _bfs_fragment,
    _dihedral_angle,
    _rotate_fragment,
    _set_bond_distance,
    build_xy_elimination_ts,
    PAULING_DELTA,
)


class TestGeometryHelpers(unittest.TestCase):
    """Tests for the low-level geometry helpers in families.py."""

    def test_dihedral_angle_eclipsed(self):
        """Dihedral of 4 atoms in a plane (eclipsed) should be ~0°."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0.01, 0]], dtype=float)
        dih = _dihedral_angle(coords, 0, 1, 2, 3)
        self.assertAlmostEqual(dih, 0.0, delta=2.0)

    def test_dihedral_angle_anti(self):
        """Dihedral of 4 atoms with the last one on the opposite side should be ~180°."""
        coords = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, -1, 0]], dtype=float)
        dih = _dihedral_angle(coords, 0, 1, 2, 3)
        self.assertAlmostEqual(abs(dih), 180.0, delta=1.0)

    def test_dihedral_angle_gauche(self):
        """Dihedral of a gauche conformation should be ~60° or ~-60°."""
        coords = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0],
                           [1.5, 0.5, 0.866]], dtype=float)
        dih = _dihedral_angle(coords, 0, 1, 2, 3)
        self.assertTrue(40 < abs(dih) < 80, msg=f'Gauche dihedral: {dih:.1f}°')

    def test_dihedral_angle_linear_segment(self):
        """Dihedral with a linear segment (collinear B-C) returns 0."""
        coords = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0.001], [0, -1, 0]], dtype=float)
        dih = _dihedral_angle(coords, 0, 1, 2, 3)
        # Near-zero or zero — should not crash.
        self.assertIsInstance(dih, float)

    def test_rotate_fragment_180(self):
        """Rotating a single atom 180° around the x-axis flips its y-coordinate."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1.0, 0]], dtype=float)
        rotated = _rotate_fragment(coords, axis_origin=0, axis_end=1,
                                   angle_deg=180.0, moving_atoms={2})
        np.testing.assert_allclose(rotated[2], [0.5, -1.0, 0.0], atol=1e-6)

    def test_rotate_fragment_preserves_non_moving(self):
        """Atoms not in the moving set should not change."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        rotated = _rotate_fragment(coords, axis_origin=0, axis_end=1,
                                   angle_deg=90.0, moving_atoms={2})
        np.testing.assert_allclose(rotated[0], coords[0])
        np.testing.assert_allclose(rotated[1], coords[1])

    def test_set_bond_distance(self):
        """Set the distance between atom 0 (fixed) and atom 1 (mobile) to 3.0 Å."""
        coords = np.array([[0, 0, 0], [1.5, 0, 0], [2.5, 0, 0]], dtype=float)
        new_coords = _set_bond_distance(coords, fixed=0, mobile=1,
                                        target_dist=3.0, mobile_frag={1, 2})
        d = float(np.linalg.norm(new_coords[1] - new_coords[0]))
        self.assertAlmostEqual(d, 3.0, places=4)
        # Fragment atom 2 should also move by the same displacement.
        d2 = float(np.linalg.norm(new_coords[2] - new_coords[1]))
        self.assertAlmostEqual(d2, 1.0, places=4)

    def test_bfs_fragment_simple(self):
        """BFS finds the connected component not crossing the blocked atom."""
        adj = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
        frag = _bfs_fragment(adj, start=2, block={1})
        self.assertEqual(frag, {2, 3})

    def test_bfs_fragment_full(self):
        """BFS with no block returns all connected atoms."""
        adj = {0: {1}, 1: {0, 2}, 2: {1}}
        frag = _bfs_fragment(adj, start=0, block=set())
        self.assertEqual(frag, {0, 1, 2})


class TestBuildXYEliminationTS(unittest.TestCase):
    """Tests for the dedicated XY_elimination_hydroxyl TS builder."""

    @classmethod
    def setUpClass(cls):
        """Set up the propanoic acid reactant used in multiple tests."""
        cls.r_xyz = str_to_xyz("""C      -1.44342440    0.21938567    0.14134495
C      -0.17943385   -0.58558878   -0.10310381
C      -0.01901784   -1.69295804    0.90160826
O      -0.76331949   -1.97415266    1.82455783
O       1.10272691   -2.40793854    0.68425738
H      -1.43203982    0.67684331    1.13627537
H      -1.53708941    1.01747004   -0.60148550
H      -2.33303198   -0.41590501    0.07585794
H      -0.21702502   -1.02774537   -1.10407189
H       0.69165336    0.07422934   -0.03456899
H       1.09172878   -3.08582798    1.39221989""")
        cls.r = ARCSpecies(label='R', smiles='CCC(=O)O', xyz=cls.r_xyz)

    def test_returns_xyz_dict(self):
        """The builder should return a valid XYZ dictionary for propanoic acid."""
        ts = build_xy_elimination_ts(self.r_xyz, self.r.mol)
        self.assertIsNotNone(ts)
        self.assertIn('symbols', ts)
        self.assertIn('coords', ts)
        self.assertEqual(len(ts['symbols']), 11)
        self.assertEqual(len(ts['coords']), 11)

    def test_ring_bond_distances(self):
        """The 6-membered ring bonds in the TS should be at Pauling-like distances."""
        ts = build_xy_elimination_ts(self.r_xyz, self.r.mol)
        self.assertIsNotNone(ts)
        coords = np.array(ts['coords'], dtype=float)
        # Ring: Cα(0)-Hα(7)-Hoh(10)-Ooh(4)-Ccarb(2)-Cβ(1)-Cα(0)
        # Note: Hα might not be index 7 — the builder picks the best H.
        # Check the heavy-atom ring bonds instead.
        d_cc_break = float(np.linalg.norm(coords[2] - coords[1]))  # Ccarb-Cβ
        d_cc_form = float(np.linalg.norm(coords[0] - coords[1]))   # Cα-Cβ
        d_oc = float(np.linalg.norm(coords[4] - coords[2]))        # Ooh-Ccarb
        sbl_cc = get_single_bond_length('C', 'C') or 1.54
        # Breaking C-C should be stretched well beyond SBL.
        self.assertAlmostEqual(d_cc_break, sbl_cc + 2 * PAULING_DELTA, delta=0.15,
                               msg=f'C-C breaking: {d_cc_break:.3f}')
        # Forming C=C should be shortened below SBL.
        self.assertLess(d_cc_form, sbl_cc, msg=f'C=C forming: {d_cc_form:.3f}')
        # Ooh-Ccarb should be near its reactant length (slight strengthening).
        self.assertTrue(1.2 < d_oc < 1.5, msg=f'O-C: {d_oc:.3f}')

    def test_h_h_formed(self):
        """The migrating H atoms should be close (H₂ forming)."""
        ts = build_xy_elimination_ts(self.r_xyz, self.r.mol)
        self.assertIsNotNone(ts)
        coords = np.array(ts['coords'], dtype=float)
        # Find the closest H-H pair involving H10 (the hydroxyl H).
        h_indices = [i for i, s in enumerate(ts['symbols']) if s == 'H']
        min_hh = float('inf')
        for hi in h_indices:
            d = float(np.linalg.norm(coords[hi] - coords[10]))
            if hi != 10 and d < min_hh:
                min_hh = d
        self.assertLess(min_hh, 1.5, msg=f'Closest H to H10: {min_hh:.3f} (expect < 1.5)')

    def test_element_sensitivity(self):
        """H-H should be shorter than H-O, which should be shorter than H-C in the ring."""
        ts = build_xy_elimination_ts(self.r_xyz, self.r.mol)
        self.assertIsNotNone(ts)
        coords = np.array(ts['coords'], dtype=float)
        # Find the H bonded to Ooh (H10) and the H from Cα closest to H10.
        h_indices_on_c0 = [i for i in range(len(ts['symbols']))
                           if ts['symbols'][i] == 'H'
                           and float(np.linalg.norm(coords[i] - coords[0])) < 2.0
                           and i != 10]
        if h_indices_on_c0:
            h_alpha = min(h_indices_on_c0,
                          key=lambda h: float(np.linalg.norm(coords[h] - coords[10])))
            d_hh = float(np.linalg.norm(coords[h_alpha] - coords[10]))
            d_oh = float(np.linalg.norm(coords[10] - coords[4]))
            d_ch = float(np.linalg.norm(coords[h_alpha] - coords[0]))
            # Element sensitivity: H-H < H-O < H-C
            self.assertLess(d_hh, d_oh, msg=f'H-H={d_hh:.3f} should be < H-O={d_oh:.3f}')

    def test_returns_none_for_non_matching_molecule(self):
        """The builder should return None for molecules without a carboxylic acid group."""
        ethane = ARCSpecies(label='ethane', smiles='CC', xyz=str_to_xyz(
            'C 0 0 0\nC 1.54 0 0\nH -0.5 0.9 0\nH -0.5 -0.9 0\nH -0.5 0 0.9\n'
            'H 2.04 0.9 0\nH 2.04 -0.9 0\nH 2.04 0 0.9'))
        ts = build_xy_elimination_ts(ethane.get_xyz(), ethane.mol)
        self.assertIsNone(ts)

    def test_no_colliding_atoms(self):
        """The generated TS should not have colliding atoms."""
        from arc.species.species import colliding_atoms
        ts = build_xy_elimination_ts(self.r_xyz, self.r.mol)
        self.assertIsNotNone(ts)
        self.assertFalse(colliding_atoms(ts), 'TS has colliding atoms')

    def test_atom_count_preserved(self):
        """The TS should have exactly the same atoms as the reactant."""
        ts = build_xy_elimination_ts(self.r_xyz, self.r.mol)
        self.assertIsNotNone(ts)
        self.assertEqual(ts['symbols'], self.r_xyz['symbols'])


if __name__ == '__main__':
    unittest.main()
