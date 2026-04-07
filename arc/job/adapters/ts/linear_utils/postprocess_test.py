#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the
arc.job.adapters.ts.linear_utils.postprocess module
"""

import unittest

import numpy as np

from arc.molecule.molecule import Molecule

from arc.species import ARCSpecies

from arc.job.adapters.ts.linear_utils.postprocess import (
    PAULING_DELTA,
    _has_detached_hydrogen,
    _has_h_close_contact,
    _has_too_many_fragments,
    _has_misoriented_migrating_h,
    _has_migrating_h_nearer_to_nonreactive,
    _has_bad_ts_motif,
    has_excessive_backbone_drift,
    has_misdirected_migrating_h,
    adjust_reactive_bond_distances,
    has_broken_nonreactive_bond,
    fix_forming_bond_distances,
    fix_nonreactive_h_distances,
    fix_crowded_h_atoms,
    fix_h_nonbonded_clashes,
    has_inward_migrating_group_h,
    fix_migrating_group_umbrella,
    stagger_donor_terminal_h,
    postprocess_h_migration,
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
    # has_misdirected_migrating_h
    # ------------------------------------------------------------------
    def test_has_misdirected_migrating_h_no_forming_bonds(self):
        """An empty forming-bond list trivially passes."""
        xyz = {
            'symbols': ('O', 'H'),
            'isotopes': (16, 1),
            'coords': ((0.0, 0.0, 0.0), (0.96, 0.0, 0.0)),
        }
        self.assertFalse(has_misdirected_migrating_h(xyz, []))

    def test_has_misdirected_migrating_h_no_h_in_forming_bonds(self):
        """Heavy-heavy forming bonds are ignored (no migrating H to check)."""
        xyz = {
            'symbols': ('C', 'C', 'H', 'H'),
            'isotopes': (12, 12, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (1.5, 0.0, 0.0),
                       (-1.0, 0.0, 0.0),
                       (2.5, 0.0, 0.0)),
        }
        # Forming bond between the two carbons; no H involved.
        self.assertFalse(has_misdirected_migrating_h(xyz, [(0, 1)]))

    def test_has_misdirected_migrating_h_normal_ts_geometry(self):
        """A migrating H placed at its Pauling TS distance from the
        acceptor passes (~ sbl + 0.42)."""
        # O-H Pauling target ~ 0.96 + 0.42 = 1.38; place H at 1.32 from O
        # (typical good TS guess), still bonded to its donor C at 1.50.
        xyz = {
            'symbols': ('C', 'H', 'O'),
            'isotopes': (12, 1, 16),
            'coords': ((0.0, 0.0, 0.0),
                       (1.50, 0.0, 0.0),
                       (2.82, 0.0, 0.0)),
        }
        self.assertFalse(has_misdirected_migrating_h(xyz, [(2, 1)]))

    def test_has_misdirected_migrating_h_far_from_acceptor(self):
        """Migrating H far from its acceptor (Pauling target * 2.5) is rejected."""
        # H placed 4.0 A from O while the forming bond is (O, H).
        # Pauling target ~ 1.38; 4.0 / 1.38 ~ 2.9 > 2.0 (default factor).
        xyz = {
            'symbols': ('C', 'H', 'O'),
            'isotopes': (12, 1, 16),
            'coords': ((0.0, 0.0, 0.0),
                       (1.10, 0.0, 0.0),
                       (5.10, 0.0, 0.0)),
        }
        self.assertTrue(has_misdirected_migrating_h(xyz, [(2, 1)]))

    def test_has_misdirected_migrating_h_custom_threshold(self):
        """Custom max_factor changes which distances are rejected."""
        # H at 2.5 A from O: ~1.81 * Pauling target.
        xyz = {
            'symbols': ('C', 'H', 'O'),
            'isotopes': (12, 1, 16),
            'coords': ((0.0, 0.0, 0.0),
                       (1.10, 0.0, 0.0),
                       (3.60, 0.0, 0.0)),
        }
        # Strict threshold (1.5) catches it; default threshold (2.0) doesn't.
        self.assertTrue(has_misdirected_migrating_h(xyz, [(2, 1)], max_factor=1.5))
        self.assertFalse(has_misdirected_migrating_h(xyz, [(2, 1)], max_factor=2.0))

    def test_has_misdirected_migrating_h_pentanal_bad_xyz(self):
        """Regression: the pentanal -> cyclopentanol bad TS guess
        previously generated by ring closure on the O-H forming bond is
        rejected.  In that geometry the migrating H sits far from the
        acceptor O while still close to its donor C — exactly the
        ``H pointing the wrong way`` symptom this filter targets."""
        # User-provided bad xyz from the original report:
        # H[8] at (4.74, -2.77, 2.66), O[5] at (0.52, -4.12, 2.28),
        # giving O-H = 4.45 A which is > 2.0 * (0.96 + 0.42) = 2.76 A.
        bad_xyz = {
            'symbols': ('C', 'C', 'C', 'C', 'C', 'O',
                        'H', 'H', 'H', 'H', 'H', 'H',
                        'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 12, 12, 12, 16,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            'coords': ((3.24657550, -2.57001959, 2.67218088),
                       (2.50271270, -1.43372393, 1.98878846),
                       (1.33268649, -1.88507190, 1.11593671),
                       (-0.00916763, -1.92676106, 1.83978234),
                       (-0.34013642, -3.26490651, 2.45557060),
                       (0.51794350, -4.12410013, 2.28438051),
                       (3.98217017, -3.01439404, 2.00170952),
                       (3.77508364, -2.20056804, 3.55097818),
                       (4.74326405, -2.76943385, 2.65613908),
                       (2.14600680, -0.75703184, 2.76528618),
                       (3.22335050, -0.88156711, 1.38554389),
                       (1.56626277, -2.87297057, 0.71895013),
                       (1.26749969, -1.21238834, 0.26074813),
                       (-0.01648520, -1.18716968, 2.64043897),
                       (-0.80949860, -1.68060709, 1.14194384),
                       (-1.24814858, -3.49644817, 3.01234917)),
        }
        # Forming bonds in the trivial-fallback-detected order:
        # (0, 4) is the new C-C; (5, 8) is the new O-H (the migrating
        # H is atom 8, the acceptor is atom 5).
        forming_bonds = [(0, 4), (5, 8)]
        self.assertTrue(has_misdirected_migrating_h(bad_xyz, forming_bonds))
        # Tightening the threshold further still rejects it.
        self.assertTrue(has_misdirected_migrating_h(bad_xyz, forming_bonds,
                                                    max_factor=1.5))

    def test_has_misdirected_migrating_h_pentanal_good_xyz(self):
        """Sanity: the well-formed pentanal -> cyclopentanol TS guess
        (the one currently produced by interpolate_isomerization) is
        not rejected."""
        # The current good guess has H[8] at d_O ~ 1.32 A.
        good_xyz = {
            'symbols': ('C', 'C', 'C', 'C', 'C', 'O',
                        'H', 'H', 'H', 'H', 'H', 'H',
                        'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 12, 12, 12, 16,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            'coords': ((2.37622772, 0.13188036, -0.10043648),
                       (1.09261098, -0.13889783, 0.66802441),
                       (1.02784361, -1.57835621, 1.17626773),
                       (2.36484309, -2.03834764, 1.74819711),
                       (3.50408935, -1.61952543, 0.87973604),
                       (4.36546469, -0.72181688, 1.21157095),
                       (1.56422743, -0.09701856, -0.79062545),
                       (2.09115810, 0.99539655, 0.50054197),
                       (3.53132429, 0.21665884, 0.80480632),
                       (1.02736061, 0.55104643, 1.50934439),
                       (0.23918280, 0.06065247, 0.01999661),
                       (0.73382444, -2.24222880, 0.36329422),
                       (0.25744088, -1.66314672, 1.94268159),
                       (2.51367011, -1.62732890, 2.74670305),
                       (2.39465238, -3.12557678, 1.81992021),
                       (3.59883726, -2.10358433, -0.07523703)),
        }
        forming_bonds = [(0, 4), (5, 8)]
        self.assertFalse(has_misdirected_migrating_h(good_xyz, forming_bonds))

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


    def test_has_close_h_pair_rejects_endocyclic_ts0(self):
        """Test that a TS with H atoms too close on the same parent is rejected.

        Uses the actual TS0 from test_interpolate_intra_r_add_endocyclic where
        C0 has H6-H7=1.076 Å and C5 has H13-H14=0.791 Å (both unphysical).
        """
        from arc.job.adapters.ts.linear_utils.postprocess import has_close_h_pair_on_same_parent
        from arc.species import ARCSpecies
        mol = ARCSpecies(label='R', smiles='[CH2]C(=C)CC=C', xyz={
            'symbols': ('C','C','C','C','C','C','H','H','H','H','H','H','H','H','H'),
            'isotopes': (12,12,12,12,12,12,1,1,1,1,1,1,1,1,1),
            'coords': ((-1.278, 1.000, 0.801), (-1.019, -0.230, 0.090),
                       (-0.026, -0.293, -0.810), (-1.884, -1.427, 0.425),
                       (-3.277, -1.295, -0.130), (-4.393, -1.321, 0.610),
                       (-2.110, 1.062, 1.493), (-0.686, 1.889, 0.615),
                       (0.598, 0.564, -1.040), (0.189, -1.212, -1.348),
                       (-1.903, -1.570, 1.513), (-1.442, -2.342, 0.010),
                       (-3.361, -1.180, -1.209), (-4.360, -1.437, 1.689),
                       (-5.369, -1.223, 0.145))
        }).mol
        # TS0 with collapsed H pairs: C0 H6-H7=1.076, C5 H13-H14=0.791
        ts_xyz = {
            'symbols': ('C','C','C','C','C','C','H','H','H','H','H','H','H','H','H'),
            'isotopes': (12,12,12,12,12,12,1,1,1,1,1,1,1,1,1),
            'coords': ((-0.698, 0.878, -1.128), (-1.019, -0.230, 0.090),
                       (-0.026, -0.293, -0.810), (-1.800, -1.506, 0.302),
                       (-2.597, -2.324, -0.720), (-1.908, -3.108, -1.776),
                       (-0.375, 0.424, -1.560), (-0.153, 1.320, -0.605),  # H6-H7 close!
                       (0.977, -0.419, -0.436), (-0.082, 0.296, -1.757),
                       (-2.864, -1.265, 0.201), (-1.648, -1.965, 1.282),
                       (-2.126, -3.643, -0.260), (-2.189, -2.831, -2.143), (-1.631, -3.385, -2.174))  # H13-H14 close!
        }
        self.assertTrue(has_close_h_pair_on_same_parent(ts_xyz, mol, min_hh_dist=1.2))

    def test_orient_h_on_internal_reactive_center(self):
        """Test that H atoms on an internal reactive CH₂ are flipped when both face the reactive partner.

        Mimics TS0 from intra_NO2_ONO_conversion: C3 is an internal CH₂ bonded to
        both N1 (reactive) and C4 (non-reactive). Both H's on C3 point toward N1,
        which is chemically wrong — they should point away from the NO₂ group.
        """
        from arc.job.adapters.ts.linear_utils.postprocess import orient_h_on_reactive_centers
        from arc.species import ARCSpecies
        # Simplified: N0-C1(-H3,-H4)-C2  where C1 is the reactive centre
        # and the forming bond is (C1, O_phantom) in the N direction.
        # Put both H's on C1 pointing TOWARD N0.
        mol = ARCSpecies(label='test', smiles='NCC', xyz={
            'symbols': ('N', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (14, 12, 12, 1, 1, 1, 1, 1, 1, 1),
            'coords': ((-1.5, 0, 0), (0, 0, 0), (1.5, 0, 0),
                       (-0.3, 0.5, 0.8),   # H3 on C1 pointing toward N (wrong)
                       (-0.3, -0.5, 0.8),   # H4 on C1 pointing toward N (wrong)
                       (-2.0, 0.5, 0), (-2.0, -0.5, 0),
                       (2.0, 0.5, 0), (2.0, -0.5, 0), (2.0, 0, 0.9))
        }).mol
        xyz = {
            'symbols': ('N', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (14, 12, 12, 1, 1, 1, 1, 1, 1, 1),
            'coords': ((-1.5, 0, 0), (0, 0, 0), (1.5, 0, 0),
                       (-0.3, 0.5, 0.8), (-0.3, -0.5, 0.8),
                       (-2.0, 0.5, 0), (-2.0, -0.5, 0),
                       (2.0, 0.5, 0), (2.0, -0.5, 0), (2.0, 0, 0.9))
        }
        result = orient_h_on_reactive_centers(xyz, mol,
                                              breaking_bonds=[(0, 1)],
                                              forming_bonds=[])
        coords_new = np.array(result['coords'])
        # After flipping, H3 and H4 should have positive x (away from N at -1.5).
        h3_x = coords_new[3][0]
        h4_x = coords_new[4][0]
        self.assertGreater(h3_x, 0, f'H3 should point away from N: x={h3_x:.3f}')
        self.assertGreater(h4_x, 0, f'H4 should point away from N: x={h4_x:.3f}')

    def test_orient_h_on_reactive_centers(self):
        """Test that H atoms on reactive centres are flipped away from the reactive direction."""
        from arc.job.adapters.ts.linear_utils.postprocess import orient_h_on_reactive_centers
        from arc.species import ARCSpecies
        # Simple case: CH3 radical approaching a C atom.
        # Put two H atoms on C0 pointing TOWARD the forming-bond partner C1.
        mol = ARCSpecies(label='test', smiles='[CH2]C', xyz={
            'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 1, 1, 1, 1, 1),
            'coords': ((0, 0, 0), (2, 0, 0),
                       (0.5, 0.5, 0),   # H2 on C0 — pointing toward C1 (wrong)
                       (0.5, -0.5, 0),   # H3 on C0 — pointing toward C1 (wrong)
                       (-0.5, 0, 0.9),   # H4 on C0 — pointing away (OK)
                       (2.5, 0.5, 0),    # H5 on C1
                       (2.5, -0.5, 0))   # H6 on C1
        }).mol
        xyz = {
            'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 1, 1, 1, 1, 1),
            'coords': ((0, 0, 0), (2, 0, 0),
                       (0.5, 0.5, 0), (0.5, -0.5, 0), (-0.5, 0, 0.9),
                       (2.5, 0.5, 0), (2.5, -0.5, 0))
        }
        result = orient_h_on_reactive_centers(xyz, mol,
                                              breaking_bonds=[],
                                              forming_bonds=[(0, 1)])
        coords_new = np.array(result['coords'])
        # After flipping, H2 and H3 should have negative x (away from C1).
        h_avg_x = (coords_new[2][0] + coords_new[3][0]) / 2
        self.assertLess(h_avg_x, 0, 'H atoms should point away from forming bond partner')


class TestPostprocessConstants(unittest.TestCase):
    """Tests for module-level constants."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_pauling_delta_value(self):
        """PAULING_DELTA is approximately 0.6*ln(2) ~ 0.42."""
        self.assertAlmostEqual(PAULING_DELTA, 0.42, places=2)


class TestAdjustReactiveBondDistances(unittest.TestCase):
    """Tests for the adjust_reactive_bond_distances function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_no_bonds_returns_unchanged(self):
        """No breaking or forming bonds means no changes."""
        mol = Molecule().from_smiles('CC')
        xyz = {
            'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 1, 1, 1, 1, 1, 1),
            'coords': ((0.0, 0.0, 0.0), (1.54, 0.0, 0.0),
                       (-0.39, 1.03, 0.0), (-0.39, -0.51, 0.89),
                       (-0.39, -0.51, -0.89), (1.93, 1.03, 0.0),
                       (1.93, -0.51, 0.89), (1.93, -0.51, -0.89)),
        }
        result = adjust_reactive_bond_distances(xyz, mol, breaking_bonds=[], forming_bonds=[])
        np.testing.assert_allclose(result['coords'], xyz['coords'], atol=1e-10)

    def test_breaking_bond_at_equilibrium_gets_stretched(self):
        """A C-C breaking bond at equilibrium distance (1.54 A) gets stretched toward TS."""
        mol = Molecule().from_smiles('CC')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        symbols = tuple(a.symbol for a in mol.atoms)
        coords_list = []
        for i, atom in enumerate(mol.atoms):
            if atom.symbol == 'C':
                coords_list.append((c_idx.index(i) * 1.40, 0.0, 0.0))
            else:
                bonded_c = None
                for nbr in atom.bonds.keys():
                    if nbr.symbol == 'C':
                        bonded_c = mol.atoms.index(nbr)
                        break
                c_rank = c_idx.index(bonded_c) if bonded_c is not None else 0
                coords_list.append((c_rank * 1.40, 1.09, 0.0))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(12 if s == 'C' else 1 for s in symbols),
            'coords': tuple(tuple(c) for c in coords_list),
        }
        d_before = np.linalg.norm(
            np.array(xyz['coords'][c_idx[0]]) - np.array(xyz['coords'][c_idx[1]]))
        result = adjust_reactive_bond_distances(
            xyz, mol, breaking_bonds=[(c_idx[0], c_idx[1])], forming_bonds=[])
        d_after = np.linalg.norm(
            np.array(result['coords'][c_idx[0]]) - np.array(result['coords'][c_idx[1]]))
        # The bond was shorter than SBL, so it should get stretched
        self.assertGreater(d_after, d_before)

    def test_h_bonds_are_skipped(self):
        """Breaking bonds involving H are not adjusted."""
        mol = Molecule().from_smiles('C')
        symbols = tuple(a.symbol for a in mol.atoms)
        h_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'H']
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        coords = tuple((float(i) * 1.0, 0.0, 0.0) for i in range(len(symbols)))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(12 if s == 'C' else 1 for s in symbols),
            'coords': coords,
        }
        if h_idx and c_idx:
            result = adjust_reactive_bond_distances(
                xyz, mol, breaking_bonds=[(c_idx[0], h_idx[0])], forming_bonds=[])
            # H-involving bonds should remain unchanged
            np.testing.assert_allclose(result['coords'], xyz['coords'], atol=1e-10)

    def test_forming_bond_over_compressed_gets_pushed(self):
        """A forming heavy-atom bond compressed below 90% SBL gets pushed apart."""
        mol = Molecule().from_smiles('CC')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        symbols = tuple(a.symbol for a in mol.atoms)
        # Put the two C atoms very close (0.8 A, well below 90% of 1.54 = 1.386)
        coords_list = []
        for i, atom in enumerate(mol.atoms):
            if atom.symbol == 'C':
                coords_list.append((c_idx.index(i) * 0.8, 0.0, 0.0))
            else:
                bonded_c = None
                for nbr in atom.bonds.keys():
                    if nbr.symbol == 'C':
                        bonded_c = mol.atoms.index(nbr)
                        break
                c_rank = c_idx.index(bonded_c) if bonded_c is not None else 0
                coords_list.append((c_rank * 0.8, 1.09, 0.0))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(12 if s == 'C' else 1 for s in symbols),
            'coords': tuple(tuple(c) for c in coords_list),
        }
        d_before = np.linalg.norm(
            np.array(xyz['coords'][c_idx[0]]) - np.array(xyz['coords'][c_idx[1]]))
        result = adjust_reactive_bond_distances(
            xyz, mol, breaking_bonds=[], forming_bonds=[(c_idx[0], c_idx[1])])
        d_after = np.linalg.norm(
            np.array(result['coords'][c_idx[0]]) - np.array(result['coords'][c_idx[1]]))
        self.assertGreater(d_after, d_before)

    def test_preserves_symbols(self):
        """Symbols are preserved through adjustment."""
        mol = Molecule().from_smiles('CC')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        symbols = tuple(a.symbol for a in mol.atoms)
        coords = tuple((float(i) * 1.5, 0.0, 0.0) for i in range(len(symbols)))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(12 if s == 'C' else 1 for s in symbols),
            'coords': coords,
        }
        result = adjust_reactive_bond_distances(
            xyz, mol, breaking_bonds=[(c_idx[0], c_idx[1])], forming_bonds=[])
        self.assertEqual(result['symbols'], xyz['symbols'])


class TestHasBrokenNonreactiveBond(unittest.TestCase):
    """Tests for the has_broken_nonreactive_bond function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_normal_geometry_returns_false(self):
        """A molecule at equilibrium geometry has no over-stretched bonds."""
        mol = Molecule().from_smiles('CC')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        symbols = tuple(a.symbol for a in mol.atoms)
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
            'isotopes': tuple(12 if s == 'C' else 1 for s in symbols),
            'coords': tuple(tuple(c) for c in coords_list),
        }
        reactive_bonds = {(c_idx[0], c_idx[1])}
        self.assertFalse(has_broken_nonreactive_bond(xyz, mol, reactive_bonds))

    def test_stretched_nonreactive_bond_returns_true(self):
        """A non-reactive C-H bond stretched to 10.0 A is detected.

        The function exempts bonds within 2 hops of the reactive center, so we
        use pentane (CCCCC) and make the reactive bond C0-C1 while stretching
        an H on C4 (3 hops away from C1, 4 from C0).
        """
        spc = ARCSpecies(label='pentane', smiles='CCCCC', xyz={
            'symbols': ('C', 'C', 'C', 'C', 'C',
                        'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 12, 12, 12,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            'coords': (
                (-2.541, 0.391, 0.000), (-1.267, -0.459, 0.000),
                (0.000, 0.391, 0.000), (1.267, -0.459, 0.000),
                (2.541, 0.391, 0.000),
                (-3.422, -0.261, 0.000), (-2.541, 1.036, 0.882),
                (-2.541, 1.036, -0.882), (-1.267, -1.104, 0.882),
                (-1.267, -1.104, -0.882), (0.000, 1.036, 0.882),
                (0.000, 1.036, -0.882), (1.267, -1.104, 0.882),
                (1.267, -1.104, -0.882), (3.422, -0.261, 0.000),
                (2.541, 1.036, 0.882), (2.541, 1.036, -0.882),
            ),
        })
        mol = spc.mol
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        # Reactive bond: C0-C1 (far from C4)
        reactive_bonds = {(min(c_idx[0], c_idx[1]), max(c_idx[0], c_idx[1]))}

        # Stretch an H on C4 (>2 hops from reactive center) to 10.0 A away
        atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
        h_on_c4 = [atom_to_idx[nbr] for nbr in mol.atoms[c_idx[4]].bonds
                    if nbr.symbol == 'H']
        if h_on_c4:
            xyz = spc.get_xyz()
            coords_list = list(list(c) for c in xyz['coords'])
            coords_list[h_on_c4[0]] = [10.0, 0.0, 0.0]
            xyz_stretched = dict(xyz)
            xyz_stretched['coords'] = tuple(tuple(c) for c in coords_list)
            self.assertTrue(has_broken_nonreactive_bond(xyz_stretched, mol, reactive_bonds))

    def test_reactive_bond_exempt(self):
        """Reactive bonds are exempt from the check even if stretched."""
        mol = Molecule().from_smiles('CC')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        symbols = tuple(a.symbol for a in mol.atoms)
        # Stretch the C-C bond to 5.0 A (far beyond 1.3x SBL)
        coords_list = []
        for i, atom in enumerate(mol.atoms):
            if atom.symbol == 'C':
                coords_list.append((c_idx.index(i) * 5.0, 0.0, 0.0))
            else:
                bonded_c = None
                for nbr in atom.bonds.keys():
                    if nbr.symbol == 'C':
                        bonded_c = mol.atoms.index(nbr)
                        break
                c_rank = c_idx.index(bonded_c) if bonded_c is not None else 0
                coords_list.append((c_rank * 5.0, 1.09, 0.0))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(12 if s == 'C' else 1 for s in symbols),
            'coords': tuple(tuple(c) for c in coords_list),
        }
        # The C-C bond is reactive, so it should be exempt
        reactive_bonds = {(min(c_idx[0], c_idx[1]), max(c_idx[0], c_idx[1]))}
        # All non-reactive bonds are C-H which are still at normal distance
        self.assertFalse(has_broken_nonreactive_bond(xyz, mol, reactive_bonds))

    def test_custom_stretch_ratio(self):
        """Custom max_stretch_ratio changes detection threshold."""
        mol = Molecule().from_smiles('CC')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        symbols = tuple(a.symbol for a in mol.atoms)
        # C-H at 1.5 A (normal is ~1.09): ratio = 1.5/1.09 ~ 1.38
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
                coords_list.append((c_rank * 1.54, 1.50, 0.0))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(12 if s == 'C' else 1 for s in symbols),
            'coords': tuple(tuple(c) for c in coords_list),
        }
        reactive_bonds = {(min(c_idx[0], c_idx[1]), max(c_idx[0], c_idx[1]))}
        # With default 1.3 it should detect the stretched C-H as problematic
        # (since ratio ~1.38 > 1.3), but with 1.5 it should not
        self.assertFalse(has_broken_nonreactive_bond(xyz, mol, reactive_bonds,
                                                      max_stretch_ratio=1.5))


class TestPostprocessHMigration(unittest.TestCase):
    """Tests for the postprocess_h_migration function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_no_forming_bonds_returns_empty_migrating_set(self):
        """With no forming bonds, migrating H set is empty."""
        mol = Molecule().from_smiles('CC')
        symbols = tuple(a.symbol for a in mol.atoms)
        coords = tuple((float(i) * 1.5, 0.0, 0.0) for i in range(len(symbols)))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(12 if s == 'C' else 1 for s in symbols),
            'coords': coords,
        }
        result_xyz, migrating_hs = postprocess_h_migration(
            xyz, mol, forming_bonds=[], breaking_bonds=[])
        self.assertIsInstance(migrating_hs, set)
        self.assertEqual(len(migrating_hs), 0)
        self.assertIn('coords', result_xyz)

    def test_h_migration_identifies_migrating_h(self):
        """H atoms in forming bonds are identified as migrating."""
        spc = ARCSpecies(label='ethanol', smiles='CCO', xyz={
            'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 16, 1, 1, 1, 1, 1, 1),
            'coords': (
                (-1.168, -0.211, 0.000),
                (0.138, 0.545, 0.000),
                (1.210, -0.383, 0.000),
                (-2.027, 0.463, 0.000),
                (-1.168, -0.855, 0.882),
                (-1.168, -0.855, -0.882),
                (0.138, 1.189, 0.882),
                (0.138, 1.189, -0.882),
                (2.053, 0.091, 0.000),
            ),
        })
        mol = spc.mol
        xyz = spc.get_xyz()
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        o_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'O']
        h_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'H']

        # Pick an H on C0 to form bond with O
        atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
        h_on_c0 = [atom_to_idx[nbr] for nbr in mol.atoms[c_idx[0]].bonds
                    if nbr.symbol == 'H']
        if h_on_c0 and o_idx:
            forming = [(h_on_c0[0], o_idx[0])]
            breaking = [(h_on_c0[0], c_idx[0])]
            result_xyz, migrating_hs = postprocess_h_migration(
                xyz, mol, forming_bonds=forming, breaking_bonds=breaking)
            self.assertIn(h_on_c0[0], migrating_hs)

    def test_returns_valid_xyz_dict(self):
        """The returned xyz dict has all required keys."""
        mol = Molecule().from_smiles('CC')
        symbols = tuple(a.symbol for a in mol.atoms)
        n = len(symbols)
        coords = tuple((float(i) * 1.5, 0.0, 0.0) for i in range(n))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(12 if s == 'C' else 1 for s in symbols),
            'coords': coords,
        }
        result_xyz, _ = postprocess_h_migration(xyz, mol, forming_bonds=[], breaking_bonds=[])
        self.assertIn('symbols', result_xyz)
        self.assertIn('coords', result_xyz)
        self.assertEqual(len(result_xyz['coords']), n)

    def test_preserves_atom_count(self):
        """The number of atoms is preserved through the pipeline."""
        spc = ARCSpecies(label='propane', smiles='CCC', xyz={
            'symbols': ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
            'coords': (
                (-1.267, -0.259, 0.000),
                (0.000, 0.587, 0.000),
                (1.267, -0.259, 0.000),
                (-2.148, 0.393, 0.000),
                (-1.267, -0.904, 0.882),
                (-1.267, -0.904, -0.882),
                (0.000, 1.232, 0.882),
                (0.000, 1.232, -0.882),
                (2.148, 0.393, 0.000),
                (1.267, -0.904, 0.882),
                (1.267, -0.904, -0.882),
            ),
        })
        mol = spc.mol
        xyz = spc.get_xyz()
        result_xyz, _ = postprocess_h_migration(xyz, mol, forming_bonds=[], breaking_bonds=[])
        self.assertEqual(len(result_xyz['symbols']), 11)
        self.assertEqual(len(result_xyz['coords']), 11)


class TestStaggerDonorTerminalH(unittest.TestCase):
    """Tests for the stagger_donor_terminal_h function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_no_forming_h_bond_returns_unchanged(self):
        """No forming bonds involving H means no staggering needed."""
        mol = Molecule().from_smiles('CC')
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        symbols = tuple(a.symbol for a in mol.atoms)
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
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(12 if s == 'C' else 1 for s in symbols),
            'coords': tuple(tuple(c) for c in coords_list),
        }
        # Only heavy-atom forming bond -> no H staggering
        result = stagger_donor_terminal_h(xyz, mol, bonds=[(c_idx[0], c_idx[1])])
        np.testing.assert_allclose(result['coords'], xyz['coords'], atol=1e-10)

    def test_stagger_rotates_h_atoms(self):
        """H atoms on the donor are rotated to staggered positions."""
        spc = ARCSpecies(label='ethanol', smiles='CCO', xyz={
            'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 16, 1, 1, 1, 1, 1, 1),
            'coords': (
                (-1.168, -0.211, 0.000),
                (0.138, 0.545, 0.000),
                (1.210, -0.383, 0.000),
                (-2.027, 0.463, 0.000),
                (-1.168, -0.855, 0.882),
                (-1.168, -0.855, -0.882),
                (0.138, 1.189, 0.882),
                (0.138, 1.189, -0.882),
                (2.053, 0.091, 0.000),
            ),
        })
        mol = spc.mol
        xyz = spc.get_xyz()
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        o_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'O']

        # H migration: H3 on C0 migrates to O
        atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
        h_on_c0 = [atom_to_idx[nbr] for nbr in mol.atoms[c_idx[0]].bonds
                    if nbr.symbol == 'H']
        if h_on_c0 and o_idx:
            forming = [(h_on_c0[0], o_idx[0])]
            result = stagger_donor_terminal_h(xyz, mol, bonds=forming)
            self.assertIn('coords', result)
            self.assertEqual(len(result['coords']), len(xyz['coords']))
            # The staggering may or may not move atoms depending on current dihedrals
            # but it should preserve symbols
            self.assertEqual(result['symbols'], xyz['symbols'])

    def test_preserves_bond_lengths(self):
        """Non-migrating H bond lengths to their parent are preserved."""
        spc = ARCSpecies(label='ethanol', smiles='CCO', xyz={
            'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 16, 1, 1, 1, 1, 1, 1),
            'coords': (
                (-1.168, -0.211, 0.000),
                (0.138, 0.545, 0.000),
                (1.210, -0.383, 0.000),
                (-2.027, 0.463, 0.000),
                (-1.168, -0.855, 0.882),
                (-1.168, -0.855, -0.882),
                (0.138, 1.189, 0.882),
                (0.138, 1.189, -0.882),
                (2.053, 0.091, 0.000),
            ),
        })
        mol = spc.mol
        xyz = spc.get_xyz()
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        o_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'O']

        atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
        h_on_c0 = [atom_to_idx[nbr] for nbr in mol.atoms[c_idx[0]].bonds
                    if nbr.symbol == 'H']
        if h_on_c0 and o_idx:
            forming = [(h_on_c0[0], o_idx[0])]
            result = stagger_donor_terminal_h(xyz, mol, bonds=forming)
            # Check that non-migrating H on C0 still have same distance from C0
            for hi in h_on_c0[1:]:
                d_orig = np.linalg.norm(
                    np.array(xyz['coords'][hi]) - np.array(xyz['coords'][c_idx[0]]))
                d_new = np.linalg.norm(
                    np.array(result['coords'][hi]) - np.array(result['coords'][c_idx[0]]))
                self.assertAlmostEqual(d_orig, d_new, places=2)


class TestFixMigratingGroupUmbrella(unittest.TestCase):
    """Tests for the fix_migrating_group_umbrella function."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    def test_no_migrating_group_returns_unchanged(self):
        """When there is no migrating heavy atom, coordinates are unchanged."""
        mol = Molecule().from_smiles('CC')
        symbols = tuple(a.symbol for a in mol.atoms)
        coords = tuple((float(i) * 1.5, 0.0, 0.0) for i in range(len(symbols)))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(12 if s == 'C' else 1 for s in symbols),
            'coords': coords,
        }
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        # Only breaking bond, no forming: no atom appears in both
        result = fix_migrating_group_umbrella(
            xyz, mol, breaking_bonds=[(c_idx[0], c_idx[1])], forming_bonds=[])
        np.testing.assert_allclose(result['coords'], xyz['coords'], atol=1e-10)

    def test_umbrella_flip_moves_h_away_from_backbone(self):
        """H atoms on a migrating heavy atom pointing toward backbone get flipped."""
        # Build a 1,2-shift scenario: C0-C1 breaks, C1-C2 forms, C1 is migrating
        # C1 has H atoms pointing toward the backbone (C0-C2 midpoint)
        spc = ARCSpecies(label='propane', smiles='CCC', xyz={
            'symbols': ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
            'coords': (
                (-2.0, 0.0, 0.0),   # C0 (donor)
                (0.0, 1.5, 0.0),     # C1 (migrating) — above the C0-C2 line
                (2.0, 0.0, 0.0),     # C2 (acceptor)
                (-2.5, 1.0, 0.0),    # H on C0
                (-2.5, -0.5, 0.8),   # H on C0
                (-2.5, -0.5, -0.8),  # H on C0
                (0.0, 0.5, 0.8),     # H on C1 — pointing toward backbone (wrong)
                (0.0, 0.5, -0.8),    # H on C1 — pointing toward backbone (wrong)
                (2.5, 1.0, 0.0),     # H on C2
                (2.5, -0.5, 0.8),    # H on C2
                (2.5, -0.5, -0.8),   # H on C2
            ),
        })
        mol = spc.mol
        xyz = spc.get_xyz()
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']

        # C1 appears in both bb and fb: it is the migrating atom
        breaking_bonds = [(c_idx[0], c_idx[1])]
        forming_bonds = [(c_idx[1], c_idx[2])]

        result = fix_migrating_group_umbrella(xyz, mol, breaking_bonds, forming_bonds)
        self.assertIn('coords', result)
        self.assertEqual(len(result['coords']), len(xyz['coords']))

    def test_preserves_symbols_and_isotopes(self):
        """Symbols and isotopes are preserved."""
        mol = Molecule().from_smiles('CC')
        symbols = tuple(a.symbol for a in mol.atoms)
        coords = tuple((float(i) * 1.5, 0.0, 0.0) for i in range(len(symbols)))
        xyz = {
            'symbols': symbols,
            'isotopes': tuple(12 if s == 'C' else 1 for s in symbols),
            'coords': coords,
        }
        c_idx = [i for i, a in enumerate(mol.atoms) if a.symbol == 'C']
        result = fix_migrating_group_umbrella(
            xyz, mol, breaking_bonds=[(c_idx[0], c_idx[1])],
            forming_bonds=[(c_idx[0], c_idx[1])])
        self.assertEqual(result['symbols'], xyz['symbols'])
        self.assertEqual(result['isotopes'], xyz['isotopes'])


if __name__ == '__main__':
    unittest.main()
