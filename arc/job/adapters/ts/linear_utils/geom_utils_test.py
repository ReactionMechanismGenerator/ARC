#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the
arc.job.adapters.ts.linear_utils.geom_utils module
"""

import math
import unittest

import numpy as np

from arc.job.adapters.ts.linear_utils.geom_utils import (
    bfs_path,
    dihedral_deg,
    downstream,
    rotate_atoms,
)


class TestGeomUtils(unittest.TestCase):
    """Contains unit tests for the geometry helper functions."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

        # Simple linear chain: 0-1-2-3
        cls.adj_linear = [[1], [0, 2], [1, 3], [2]]

        # Branched: 0-1-2, 1-3
        cls.adj_branched = [[1], [0, 2, 3], [1], [1]]

        # Cyclic: 0-1-2-3-0 (square)
        cls.adj_cyclic = [[1, 3], [0, 2], [1, 3], [2, 0]]

        # Disconnected: 0-1, 2-3
        cls.adj_disconnected = [[1], [0], [3], [2]]

        # Single atom
        cls.adj_single = [[]]

    # ------------------------------------------------------------------
    # bfs_path
    # ------------------------------------------------------------------
    def test_bfs_path_trivial_same_node(self):
        """BFS path from a node to itself returns a single-element list."""
        result = bfs_path(self.adj_linear, 2, 2)
        self.assertEqual(result, [2])

    def test_bfs_path_adjacent(self):
        """BFS path between adjacent nodes returns a two-element path."""
        result = bfs_path(self.adj_linear, 0, 1)
        self.assertEqual(result, [0, 1])

    def test_bfs_path_linear_chain(self):
        """BFS on a linear chain returns the unique path."""
        result = bfs_path(self.adj_linear, 0, 3)
        self.assertEqual(result, [0, 1, 2, 3])

    def test_bfs_path_branched(self):
        """BFS on a branched graph finds the shortest path through the hub."""
        result = bfs_path(self.adj_branched, 0, 3)
        self.assertEqual(result, [0, 1, 3])

    def test_bfs_path_cyclic_shortest(self):
        """BFS on a cyclic graph returns the shortest of two possible paths."""
        result = bfs_path(self.adj_cyclic, 0, 2)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[-1], 2)

    def test_bfs_path_unreachable(self):
        """BFS returns None when source and destination are in different components."""
        result = bfs_path(self.adj_disconnected, 0, 2)
        self.assertIsNone(result)

    def test_bfs_path_single_atom_self(self):
        """BFS from the only atom to itself returns [0]."""
        result = bfs_path(self.adj_single, 0, 0)
        self.assertEqual(result, [0])

    # ------------------------------------------------------------------
    # downstream
    # ------------------------------------------------------------------
    def test_downstream_terminal_atom(self):
        """Downstream of a terminal atom includes only that atom."""
        result = downstream(self.adj_linear, cut_a=1, cut_b=0)
        self.assertEqual(result, {0})

    def test_downstream_linear_chain(self):
        """Downstream from cut 1-2 on the 2-side includes atoms 2 and 3."""
        result = downstream(self.adj_linear, cut_a=1, cut_b=2)
        self.assertEqual(result, {2, 3})

    def test_downstream_branched(self):
        """Downstream from the hub toward a leaf is just the leaf."""
        result = downstream(self.adj_branched, cut_a=1, cut_b=3)
        self.assertEqual(result, {3})

    def test_downstream_branched_hub_side(self):
        """Downstream from a leaf toward the hub includes hub and all other branches."""
        result = downstream(self.adj_branched, cut_a=0, cut_b=1)
        self.assertEqual(result, {1, 2, 3})

    def test_downstream_cyclic(self):
        """Downstream in a cyclic graph includes all atoms reachable without crossing the cut."""
        result = downstream(self.adj_cyclic, cut_a=0, cut_b=1)
        self.assertEqual(result, {1, 2, 3})

    def test_downstream_includes_cut_b(self):
        """cut_b itself is always included in the result."""
        result = downstream(self.adj_linear, cut_a=2, cut_b=3)
        self.assertIn(3, result)

    # ------------------------------------------------------------------
    # rotate_atoms
    # ------------------------------------------------------------------
    def test_rotate_atoms_zero_angle(self):
        """Rotation by 0 radians does not change coordinates."""
        coords = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        original = coords.copy()
        rotate_atoms(coords, origin=np.array([0.0, 0.0, 0.0]),
                     axis=np.array([0.0, 0.0, 1.0]),
                     indices={0, 1, 2}, angle=0.0)
        np.testing.assert_allclose(coords, original, atol=1e-12)

    def test_rotate_atoms_90_degrees_z_axis(self):
        """Rotating (1,0,0) by 90 degrees around z-axis gives (0,1,0)."""
        coords = np.array([[1.0, 0.0, 0.0]])
        rotate_atoms(coords, origin=np.array([0.0, 0.0, 0.0]),
                     axis=np.array([0.0, 0.0, 1.0]),
                     indices={0}, angle=math.pi / 2)
        np.testing.assert_allclose(coords[0], [0.0, 1.0, 0.0], atol=1e-12)

    def test_rotate_atoms_180_degrees(self):
        """Rotating (1,0,0) by 180 degrees around z-axis gives (-1,0,0)."""
        coords = np.array([[1.0, 0.0, 0.0]])
        rotate_atoms(coords, origin=np.array([0.0, 0.0, 0.0]),
                     axis=np.array([0.0, 0.0, 1.0]),
                     indices={0}, angle=math.pi)
        np.testing.assert_allclose(coords[0], [-1.0, 0.0, 0.0], atol=1e-12)

    def test_rotate_atoms_full_revolution(self):
        """Rotation by 2*pi returns the original position."""
        coords = np.array([[1.0, 2.0, 3.0]])
        original = coords.copy()
        rotate_atoms(coords, origin=np.array([0.0, 0.0, 0.0]),
                     axis=np.array([0.0, 0.0, 1.0]),
                     indices={0}, angle=2 * math.pi)
        np.testing.assert_allclose(coords, original, atol=1e-12)

    def test_rotate_atoms_subset_only(self):
        """Only atoms in the indices set are rotated; others are unchanged."""
        coords = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
        rotate_atoms(coords, origin=np.array([0.0, 0.0, 0.0]),
                     axis=np.array([0.0, 0.0, 1.0]),
                     indices={0}, angle=math.pi / 2)
        np.testing.assert_allclose(coords[0], [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(coords[1], [0.0, 1.0, 0.0], atol=1e-12)

    def test_rotate_atoms_around_nonzero_origin(self):
        """Rotation around an offset origin shifts the rotation center."""
        coords = np.array([[2.0, 0.0, 0.0]])
        rotate_atoms(coords, origin=np.array([1.0, 0.0, 0.0]),
                     axis=np.array([0.0, 0.0, 1.0]),
                     indices={0}, angle=math.pi / 2)
        np.testing.assert_allclose(coords[0], [1.0, 1.0, 0.0], atol=1e-12)

    def test_rotate_atoms_empty_indices(self):
        """Passing empty indices set does nothing."""
        coords = np.array([[1.0, 0.0, 0.0]])
        original = coords.copy()
        rotate_atoms(coords, origin=np.array([0.0, 0.0, 0.0]),
                     axis=np.array([0.0, 0.0, 1.0]),
                     indices=set(), angle=math.pi / 2)
        np.testing.assert_allclose(coords, original, atol=1e-12)

    def test_rotate_atoms_preserves_distances(self):
        """Rotation preserves interatomic distances (is an isometry)."""
        coords = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        d_before = np.linalg.norm(coords[0] - coords[1])
        rotate_atoms(coords, origin=np.array([0.0, 0.0, 0.0]),
                     axis=np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
                     indices={0, 1, 2}, angle=1.23)
        d_after = np.linalg.norm(coords[0] - coords[1])
        self.assertAlmostEqual(d_before, d_after, places=10)

    # ------------------------------------------------------------------
    # dihedral_deg
    # ------------------------------------------------------------------
    def test_dihedral_deg_cis_planar(self):
        """Cis-planar arrangement (all atoms in the same plane, same side) gives 0 degrees."""
        p1 = np.array([1.0, 1.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        p3 = np.array([0.0, 0.0, 0.0])
        p4 = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(dihedral_deg(p1, p2, p3, p4), 0.0, places=5)

    def test_dihedral_deg_trans_planar(self):
        """Trans-planar arrangement gives +-180 degrees."""
        p1 = np.array([1.0, 1.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        p3 = np.array([0.0, 0.0, 0.0])
        p4 = np.array([-1.0, 0.0, 0.0])
        self.assertAlmostEqual(abs(dihedral_deg(p1, p2, p3, p4)), 180.0, places=5)

    def test_dihedral_deg_90_degrees(self):
        """A known 90-degree dihedral arrangement."""
        p1 = np.array([1.0, 1.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        p3 = np.array([0.0, 0.0, 0.0])
        p4 = np.array([0.0, 0.0, 1.0])
        self.assertAlmostEqual(abs(dihedral_deg(p1, p2, p3, p4)), 90.0, places=5)

    def test_dihedral_deg_l_shape_signed_90(self):
        """L-shape with the final arm along +z yields a signed 90 degree dihedral."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        p4 = np.array([0.0, 1.0, 1.0])
        self.assertAlmostEqual(dihedral_deg(p1, p2, p3, p4), -90.0, places=5)

    def test_dihedral_deg_l_shape_mirror_signed_90(self):
        """Mirror L-shape (final arm along -z) flips the sign of the 90 degree dihedral."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        p4 = np.array([0.0, 1.0, -1.0])
        self.assertAlmostEqual(dihedral_deg(p1, p2, p3, p4), 90.0, places=5)

    def test_dihedral_deg_collinear_returns_zero(self):
        """When three consecutive points are collinear, dihedral returns 0.0."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([2.0, 0.0, 0.0])
        p4 = np.array([3.0, 1.0, 0.0])
        self.assertAlmostEqual(dihedral_deg(p1, p2, p3, p4), 0.0)

    def test_dihedral_deg_sign_convention(self):
        """Verify IUPAC sign convention: positive vs negative dihedral."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        p4_pos = np.array([0.0, 1.0, 1.0])
        p4_neg = np.array([0.0, 1.0, -1.0])
        d_pos = dihedral_deg(p1, p2, p3, p4_pos)
        d_neg = dihedral_deg(p1, p2, p3, p4_neg)
        self.assertAlmostEqual(d_pos, -d_neg, places=5)

    def test_dihedral_deg_range(self):
        """Result is always in [-180, 180]."""
        for _ in range(20):
            points = [np.random.randn(3) for _ in range(4)]
            d = dihedral_deg(*points)
            self.assertGreaterEqual(d, -180.0)
            self.assertLessEqual(d, 180.0)


if __name__ == '__main__':
    unittest.main()
