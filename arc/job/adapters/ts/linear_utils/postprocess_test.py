#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the
arc.job.adapters.ts.linear_utils.postprocess module
"""

import unittest

import numpy as np

from arc.molecule.molecule import Molecule

from arc.job.adapters.ts.linear_utils.postprocess import (
    PAULING_DELTA,
    _has_detached_hydrogen,
    _has_h_close_contact,
    _has_too_many_fragments,
    _has_misoriented_migrating_h,
    _has_migrating_h_nearer_to_nonreactive,
    _has_bad_ts_motif,
    has_excessive_backbone_drift,
    fix_forming_bond_distances,
    fix_nonreactive_h_distances,
    fix_crowded_h_atoms,
    fix_h_nonbonded_clashes,
    has_inward_migrating_group_h,
    fix_migrating_group_umbrella,
    stagger_donor_terminal_h,
)


class TestRejectionFilters(unittest.TestCase):
    """Tests for the geometry rejection filter functions."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

        # A compact water molecule: O at origin, H's nearby
        cls.water_xyz = {
            'symbols': ('O', 'H', 'H'),
            'isotopes': (16, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (0.96, 0.0, 0.0),
                       (-0.24, 0.93, 0.0)),
        }

        # Water with one detached H (10 A away)
        cls.water_detached_h = {
            'symbols': ('O', 'H', 'H'),
            'isotopes': (16, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (0.96, 0.0, 0.0),
                       (10.0, 0.0, 0.0)),
        }

        # Three separate atoms far apart (3 fragments)
        cls.three_fragments = {
            'symbols': ('C', 'H', 'N', 'H', 'O', 'H'),
            'isotopes': (12, 1, 14, 1, 16, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0),
                       (10.0, 0.0, 0.0),
                       (11.0, 0.0, 0.0),
                       (20.0, 0.0, 0.0),
                       (21.0, 0.0, 0.0)),
        }

        # Two fragments: reasonable TS (stretched bond)
        cls.two_fragments = {
            'symbols': ('C', 'H', 'N', 'H'),
            'isotopes': (12, 1, 14, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0),
                       (10.0, 0.0, 0.0),
                       (11.0, 0.0, 0.0)),
        }

        # Two H atoms too close (H-H sbl ~0.74 A * 0.85 ~ 0.63 A)
        cls.h_close_contact = {
            'symbols': ('C', 'H', 'H'),
            'isotopes': (12, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (1.09, 0.0, 0.0),
                       (1.09, 0.50, 0.0)),
        }

        # Single heavy atom
        cls.single_atom = {
            'symbols': ('C',),
            'isotopes': (12,),
            'coords': ((0.0, 0.0, 0.0),),
        }

    # ------------------------------------------------------------------
    # _has_detached_hydrogen
    # ------------------------------------------------------------------
    def test_has_detached_hydrogen_false_compact(self):
        """Compact water has no detached hydrogens."""
        self.assertFalse(_has_detached_hydrogen(self.water_xyz))

    def test_has_detached_hydrogen_true(self):
        """Water with H at 10 A has a detached hydrogen."""
        self.assertTrue(_has_detached_hydrogen(self.water_detached_h))

    def test_has_detached_hydrogen_custom_threshold(self):
        """Custom threshold can make a moderately distant H count as detached."""
        xyz = {
            'symbols': ('C', 'H'),
            'isotopes': (12, 1),
            'coords': ((0.0, 0.0, 0.0), (2.5, 0.0, 0.0)),
        }
        self.assertFalse(_has_detached_hydrogen(xyz, max_h_heavy_dist=3.0))
        self.assertTrue(_has_detached_hydrogen(xyz, max_h_heavy_dist=2.0))

    def test_has_detached_hydrogen_exempt_indices(self):
        """Exempt indices are skipped."""
        self.assertFalse(_has_detached_hydrogen(self.water_detached_h,
                                                 exempt_indices={2}))

    def test_has_detached_hydrogen_no_heavy_atoms(self):
        """An H-only molecule returns False (no heavy atoms to measure against)."""
        h2 = {'symbols': ('H', 'H'), 'isotopes': (1, 1),
               'coords': ((0.0, 0.0, 0.0), (0.74, 0.0, 0.0))}
        self.assertFalse(_has_detached_hydrogen(h2))

    # ------------------------------------------------------------------
    # _has_too_many_fragments
    # ------------------------------------------------------------------
    def test_has_too_many_fragments_false_single_atom(self):
        """A single atom is one fragment."""
        self.assertFalse(_has_too_many_fragments(self.single_atom))

    def test_has_too_many_fragments_false_compact(self):
        """Compact water is one fragment."""
        self.assertFalse(_has_too_many_fragments(self.water_xyz))

    def test_has_too_many_fragments_false_two_fragments(self):
        """Two fragments is acceptable for a TS."""
        self.assertFalse(_has_too_many_fragments(self.two_fragments))

    def test_has_too_many_fragments_true(self):
        """Three well-separated groups are detected."""
        self.assertTrue(_has_too_many_fragments(self.three_fragments))

    def test_has_too_many_fragments_custom_thresholds(self):
        """Increasing the thresholds can merge fragments."""
        self.assertFalse(_has_too_many_fragments(self.three_fragments,
                                                  max_heavy_heavy=25.0,
                                                  max_heavy_h=25.0))

    # ------------------------------------------------------------------
    # _has_h_close_contact
    # ------------------------------------------------------------------
    def test_has_h_close_contact_false_compact(self):
        """Normal water geometry has no close contacts."""
        self.assertFalse(_has_h_close_contact(self.water_xyz))

    def test_has_h_close_contact_true(self):
        """Two H atoms 0.50 A apart triggers close contact."""
        self.assertTrue(_has_h_close_contact(self.h_close_contact))

    def test_has_h_close_contact_heavy_only_skipped(self):
        """Pairs of only heavy atoms are not checked."""
        xyz = {
            'symbols': ('C', 'N'),
            'isotopes': (12, 14),
            'coords': ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        }
        self.assertFalse(_has_h_close_contact(xyz))

    def test_has_h_close_contact_custom_threshold(self):
        """Custom threshold changes detection sensitivity."""
        xyz_slightly_close = {
            'symbols': ('C', 'H'),
            'isotopes': (12, 1),
            'coords': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        }
        self.assertFalse(_has_h_close_contact(xyz_slightly_close, threshold=0.85))
        self.assertTrue(_has_h_close_contact(xyz_slightly_close, threshold=1.0))

    # ------------------------------------------------------------------
    # has_excessive_backbone_drift
    # ------------------------------------------------------------------
    def test_has_excessive_backbone_drift_false_no_drift(self):
        """Identical geometries have zero drift."""
        self.assertFalse(has_excessive_backbone_drift(self.water_xyz, self.water_xyz))

    def test_has_excessive_backbone_drift_true(self):
        """Large shift in heavy atoms triggers drift detection."""
        shifted = {
            'symbols': ('O', 'H', 'H'),
            'isotopes': (16, 1, 1),
            'coords': ((5.0, 5.0, 5.0),
                       (5.96, 5.0, 5.0),
                       (4.76, 5.93, 5.0)),
        }
        self.assertTrue(has_excessive_backbone_drift(shifted, self.water_xyz,
                                                       max_mean_heavy_disp=1.0))

    def test_has_excessive_backbone_drift_reactive_excluded(self):
        """Reactive atoms are excluded from drift calculation."""
        shifted = {
            'symbols': ('O', 'H', 'H'),
            'isotopes': (16, 1, 1),
            'coords': ((5.0, 5.0, 5.0),
                       (0.96, 0.0, 0.0),
                       (-0.24, 0.93, 0.0)),
        }
        self.assertFalse(has_excessive_backbone_drift(
            shifted, self.water_xyz, max_mean_heavy_disp=1.0,
            reactive_indices={0}))


class TestGeometryFixers(unittest.TestCase):
    """Tests for the geometry fixer functions."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

        # Methane molecule for testing H distance fixes.
        # CH4: C at origin, 4 H atoms at tetrahedral positions.
        cls.ch4_mol = Molecule().from_smiles('[CH4]')
        cls.ch4_xyz = {
            'symbols': ('C', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 1, 1, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (1.09, 0.0, 0.0),
                       (-0.363, 1.028, 0.0),
                       (-0.363, -0.514, 0.890),
                       (-0.363, -0.514, -0.890)),
        }

        # Ethane molecule for fix_crowded_h test.
        cls.ethane_mol = Molecule().from_smiles('CC')
        cls.ethane_xyz = {
            'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 1, 1, 1, 1, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (1.54, 0.0, 0.0),
                       (-0.39, 1.03, 0.0),
                       (-0.39, -0.51, 0.89),
                       (-0.39, -0.51, -0.89),
                       (1.93, 1.03, 0.0),
                       (1.93, -0.51, 0.89),
                       (1.93, -0.51, -0.89)),
        }

    # ------------------------------------------------------------------
    # fix_nonreactive_h_distances
    # ------------------------------------------------------------------
    def test_fix_nonreactive_h_distances_no_change_needed(self):
        """H atoms at equilibrium distance are not moved."""
        result = fix_nonreactive_h_distances(self.ch4_xyz, self.ch4_mol,
                                              migrating_h_indices=set())
        coords_orig = np.array(self.ch4_xyz['coords'])
        coords_new = np.array(result['coords'])
        for i in range(1, 5):
            d_orig = np.linalg.norm(coords_orig[i] - coords_orig[0])
            d_new = np.linalg.norm(coords_new[i] - coords_new[0])
            self.assertAlmostEqual(d_orig, d_new, places=1)

    def test_fix_nonreactive_h_distances_stretched_h(self):
        """An H atom at 2.0 A is corrected to near-equilibrium distance."""
        stretched_xyz = {
            'symbols': ('C', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 1, 1, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (2.0, 0.0, 0.0),
                       (-0.363, 1.028, 0.0),
                       (-0.363, -0.514, 0.890),
                       (-0.363, -0.514, -0.890)),
        }
        result = fix_nonreactive_h_distances(stretched_xyz, self.ch4_mol,
                                              migrating_h_indices=set())
        new_d = np.linalg.norm(
            np.array(result['coords'][1]) - np.array(result['coords'][0]))
        self.assertAlmostEqual(new_d, 1.09, places=1)

    def test_fix_nonreactive_h_distances_migrating_skipped(self):
        """Migrating H indices are not modified."""
        stretched_xyz = {
            'symbols': ('C', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 1, 1, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (2.0, 0.0, 0.0),
                       (-0.363, 1.028, 0.0),
                       (-0.363, -0.514, 0.890),
                       (-0.363, -0.514, -0.890)),
        }
        result = fix_nonreactive_h_distances(stretched_xyz, self.ch4_mol,
                                              migrating_h_indices={1})
        self.assertAlmostEqual(result['coords'][1][0], 2.0, places=5)

    # ------------------------------------------------------------------
    # fix_forming_bond_distances
    # ------------------------------------------------------------------
    def test_fix_forming_bond_distances_no_h_bond(self):
        """Forming bonds not involving H are unaffected."""
        mol = Molecule().from_smiles('CC')
        xyz = {
            'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 1, 1, 1, 1, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (3.0, 0.0, 0.0),
                       (-0.39, 1.03, 0.0),
                       (-0.39, -0.51, 0.89),
                       (-0.39, -0.51, -0.89),
                       (3.39, 1.03, 0.0),
                       (3.39, -0.51, 0.89),
                       (3.39, -0.51, -0.89)),
        }
        result = fix_forming_bond_distances(xyz, mol, [(0, 1)])
        np.testing.assert_allclose(result['coords'], xyz['coords'], atol=1e-10)

    # ------------------------------------------------------------------
    # _has_misoriented_migrating_h
    # ------------------------------------------------------------------
    def test_has_misoriented_migrating_h_false(self):
        """When migrating H is closer to acceptor than to any H on acceptor, returns False."""
        mol = Molecule().from_adjacency_list("""
1  C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2  C u0 p0 c0 {1,S} {6,S} {7,S} {8,S}
3  H u0 p0 c0 {1,S}
4  H u0 p0 c0 {1,S}
5  H u0 p0 c0 {1,S}
6  H u0 p0 c0 {2,S}
7  H u0 p0 c0 {2,S}
8  H u0 p0 c0 {2,S}
""")
        xyz = {
            'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 1, 1, 1, 1, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (2.5, 0.0, 0.0),
                       (1.3, 0.0, 0.0),
                       (-0.39, 1.03, 0.0),
                       (-0.39, -0.51, -0.89),
                       (2.89, 1.03, 0.0),
                       (2.89, -0.51, 0.89),
                       (2.89, -0.51, -0.89)),
        }
        self.assertFalse(_has_misoriented_migrating_h(xyz, [(2, 1)], mol))

    # ------------------------------------------------------------------
    # _has_bad_ts_motif
    # ------------------------------------------------------------------
    def test_has_bad_ts_motif_good_geometry(self):
        """A well-formed D-H-A geometry passes."""
        mol = Molecule().from_adjacency_list("""
1  C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2  H u0 p0 c0 {1,S}
3  H u0 p0 c0 {1,S}
4  H u0 p0 c0 {1,S}
5  N u0 p1 c0 {1,S} {6,S} {7,S}
6  H u0 p0 c0 {5,S}
7  H u0 p0 c0 {5,S}
""")
        xyz = {
            'symbols': ('C', 'H', 'H', 'H', 'N', 'H', 'H'),
            'isotopes': (12, 1, 1, 1, 14, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (1.5, 0.0, 0.0),
                       (-0.39, 1.03, 0.0),
                       (-0.39, -0.51, -0.89),
                       (3.0, 0.0, 0.0),
                       (3.39, 1.03, 0.0),
                       (3.39, -0.51, -0.89)),
        }
        self.assertFalse(_has_bad_ts_motif(xyz, [(1, 4)], mol))

    def test_has_bad_ts_motif_h_too_far(self):
        """H too far from both donor and acceptor fails."""
        mol = Molecule().from_adjacency_list("""
1  C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2  H u0 p0 c0 {1,S}
3  H u0 p0 c0 {1,S}
4  H u0 p0 c0 {1,S}
5  N u0 p1 c0 {1,S} {6,S} {7,S}
6  H u0 p0 c0 {5,S}
7  H u0 p0 c0 {5,S}
""")
        xyz = {
            'symbols': ('C', 'H', 'H', 'H', 'N', 'H', 'H'),
            'isotopes': (12, 1, 1, 1, 14, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (5.0, 0.0, 0.0),
                       (-0.39, 1.03, 0.0),
                       (-0.39, -0.51, -0.89),
                       (10.0, 0.0, 0.0),
                       (10.39, 1.03, 0.0),
                       (10.39, -0.51, -0.89)),
        }
        self.assertTrue(_has_bad_ts_motif(xyz, [(1, 4)], mol))

    # ------------------------------------------------------------------
    # _has_migrating_h_nearer_to_nonreactive
    # ------------------------------------------------------------------
    def test_has_migrating_h_nearer_to_nonreactive_false(self):
        """Migrating H between donor and acceptor: no non-reactive atom is closer."""
        mol = Molecule().from_adjacency_list("""
1  C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2  H u0 p0 c0 {1,S}
3  H u0 p0 c0 {1,S}
4  H u0 p0 c0 {1,S}
5  N u0 p1 c0 {1,S} {6,S} {7,S}
6  H u0 p0 c0 {5,S}
7  H u0 p0 c0 {5,S}
""")
        xyz = {
            'symbols': ('C', 'H', 'H', 'H', 'N', 'H', 'H'),
            'isotopes': (12, 1, 1, 1, 14, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (1.5, 0.0, 0.0),
                       (-0.5, 1.0, 0.0),
                       (-0.5, -1.0, 0.0),
                       (3.0, 0.0, 0.0),
                       (3.5, 1.0, 0.0),
                       (3.5, -1.0, 0.0)),
        }
        self.assertFalse(_has_migrating_h_nearer_to_nonreactive(
            xyz, [(1, 4)], mol))

    # ------------------------------------------------------------------
    # fix_crowded_h_atoms
    # ------------------------------------------------------------------
    def test_fix_crowded_h_atoms_no_crowding(self):
        """Well-separated H atoms are not repositioned."""
        result = fix_crowded_h_atoms(self.ethane_xyz, self.ethane_mol)
        coords_orig = np.array(self.ethane_xyz['coords'])
        coords_new = np.array(result['coords'])
        np.testing.assert_allclose(coords_new, coords_orig, atol=0.05)

    def test_fix_crowded_h_atoms_bunched_h(self):
        """Two H atoms at 0.5 A apart on the same heavy atom get redistributed."""
        bunched_xyz = {
            'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 1, 1, 1, 1, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (1.54, 0.0, 0.0),
                       (-0.39, 0.50, 0.0),
                       (-0.39, 0.70, 0.0),
                       (-0.39, 0.60, 0.1),
                       (1.93, 1.03, 0.0),
                       (1.93, -0.51, 0.89),
                       (1.93, -0.51, -0.89)),
        }
        result = fix_crowded_h_atoms(bunched_xyz, self.ethane_mol)
        coords_new = np.array(result['coords'])
        d_h2_h3 = np.linalg.norm(coords_new[2] - coords_new[3])
        self.assertGreater(d_h2_h3, 1.0)

    # ------------------------------------------------------------------
    # fix_h_nonbonded_clashes
    # ------------------------------------------------------------------
    def test_fix_h_nonbonded_clashes_no_clash(self):
        """Normal geometry has no clashes."""
        result = fix_h_nonbonded_clashes(self.ethane_xyz, self.ethane_mol)
        coords_orig = np.array(self.ethane_xyz['coords'])
        coords_new = np.array(result['coords'])
        np.testing.assert_allclose(coords_new, coords_orig, atol=0.01)


class TestPostprocessConstants(unittest.TestCase):
    """Tests for module-level constants."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_pauling_delta_value(self):
        """PAULING_DELTA is approximately 0.6*ln(2) ~ 0.42."""
        self.assertAlmostEqual(PAULING_DELTA, 0.42, places=2)


if __name__ == '__main__':
    unittest.main()
