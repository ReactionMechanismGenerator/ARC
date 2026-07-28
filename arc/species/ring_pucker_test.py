#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.species.ring_pucker module
"""

import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import arc.species.ring_pucker as ring_pucker


class TestRingPucker(unittest.TestCase):
    """
    Contains unit tests for the ring_pucker module
    """

    def test_puckering_amplitude_planar_ring_is_zero(self):
        """A perfectly planar ring has zero Cremer-Pople puckering amplitude."""
        # A regular planar hexagon in the z=0 plane.
        angles = np.linspace(0.0, 2.0 * np.pi, num=6, endpoint=False)
        ring_coords = [[float(np.cos(a)), float(np.sin(a)), 0.0] for a in angles]
        self.assertAlmostEqual(ring_pucker.puckering_amplitude(ring_coords), 0.0, places=6)

    def test_puckering_amplitude_invariant_under_rigid_motion(self):
        """The total puckering amplitude Q is invariant under rotation and translation."""
        ring_coords = ring_pucker.ideal_pucker_geometry(6, 'chair')
        q_before = ring_pucker.puckering_amplitude(ring_coords)

        theta = 0.7
        rotation = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        translation = np.array([3.5, -2.1, 7.2])
        moved_coords = ring_coords @ rotation.T + translation

        q_after = ring_pucker.puckering_amplitude(moved_coords)
        self.assertAlmostEqual(q_before, q_after, places=6)

    def test_amplitude_consistent_with_cremer_pople_params(self):
        """For a 6-ring, puckering_amplitude matches sqrt(q2^2 + q3^2) from cremer_pople_params."""
        ring_coords = ring_pucker.ideal_pucker_geometry(6, 'twist-boat')
        q_direct = ring_pucker.puckering_amplitude(ring_coords)
        params = ring_pucker.cremer_pople_params(ring_coords)
        q_from_params = np.sqrt(params.q2 ** 2 + params.q3 ** 2)
        self.assertAlmostEqual(q_direct, q_from_params, places=6)

    def test_classify_pucker_chair_like_hexagon(self):
        """A chair-seeded hexagon classifies as a chair."""
        ring_coords = ring_pucker.ideal_pucker_geometry(6, 'chair')
        self.assertEqual(ring_pucker.classify_pucker(ring_coords), 'chair')

    def test_canonical_pucker_states_for_6_and_5_rings(self):
        """canonical_pucker_states returns the expected discrete labels for 6- and 5-rings."""
        self.assertEqual(ring_pucker.canonical_pucker_states(6), ['chair', 'boat', 'twist-boat'])
        self.assertEqual(ring_pucker.canonical_pucker_states(5), ['envelope', 'twist'])

    def test_canonical_pucker_states_raises_for_unsupported_ring_size(self):
        """canonical_pucker_states raises RingPuckerError for ring sizes other than 5 or 6."""
        with self.assertRaises(ring_pucker.RingPuckerError):
            ring_pucker.canonical_pucker_states(4)
        with self.assertRaises(ring_pucker.RingPuckerError):
            ring_pucker.canonical_pucker_states(7)

    def test_ideal_pucker_geometry_round_trips_for_6_ring_labels(self):
        """Every canonical 6-ring pucker label round-trips through ideal_pucker_geometry and classify_pucker."""
        for label in ring_pucker.canonical_pucker_states(6):
            ring_coords = ring_pucker.ideal_pucker_geometry(6, label)
            self.assertEqual(ring_pucker.classify_pucker(ring_coords), label)

    def test_ideal_pucker_geometry_round_trips_for_5_ring_labels(self):
        """Every canonical 5-ring pucker label round-trips through ideal_pucker_geometry and classify_pucker."""
        for label in ring_pucker.canonical_pucker_states(5):
            ring_coords = ring_pucker.ideal_pucker_geometry(5, label)
            self.assertEqual(ring_pucker.classify_pucker(ring_coords), label)

    def test_planar_hexagon_classifies_as_planar(self):
        """A perfectly planar hexagon classifies as 'planar' rather than a definite pucker state."""
        angles = np.linspace(0.0, 2.0 * np.pi, num=6, endpoint=False)
        ring_coords = [[float(np.cos(a)), float(np.sin(a)), 0.0] for a in angles]
        self.assertEqual(ring_pucker.classify_pucker(ring_coords), 'planar')

    def test_near_planar_hexagon_with_tiny_noise_classifies_as_planar(self):
        """A hexagon with a negligible (~1e-6 Angstrom) out-of-plane perturbation is still 'planar'."""
        rng = np.random.default_rng(0)
        angles = np.linspace(0.0, 2.0 * np.pi, num=6, endpoint=False)
        ring_coords = np.array([[np.cos(a), np.sin(a), 0.0] for a in angles])
        ring_coords = ring_coords + rng.normal(scale=1e-6, size=ring_coords.shape)
        self.assertEqual(ring_pucker.classify_pucker(ring_coords), 'planar')

    def test_hexagon_between_chair_and_boat_windows_classifies_as_half_chair(self):
        """A hexagon with theta strictly between the tightened chair and boat/twist-boat windows
        is classified as a half-chair rather than being lumped into 'chair'."""
        ring_size = 6
        amplitude = 0.6
        theta_deg = 40.0
        theta_rad = np.radians(theta_deg)
        q2 = amplitude * np.sin(theta_rad)
        q_half = amplitude * np.cos(theta_rad)
        idx = np.arange(ring_size)
        xy_angles = 2.0 * np.pi * idx / ring_size
        x = ring_pucker.DEFAULT_RING_RADIUS * np.cos(xy_angles)
        y = ring_pucker.DEFAULT_RING_RADIUS * np.sin(xy_angles)
        z = (np.sqrt(2.0 / ring_size) * q2 * np.cos(2.0 * np.pi * 2 * idx / ring_size)
             + (q_half / np.sqrt(ring_size)) * ((-1.0) ** idx))
        ring_coords = np.column_stack([x, y, z])
        self.assertEqual(ring_pucker.classify_pucker(ring_coords), 'half-chair')

    def test_phi_exactly_on_bin_boundary_bins_deterministically(self):
        """A phi value placed exactly on a 30-degree hexagon bin boundary bins deterministically."""
        ring_size = 6
        amplitude = 0.6
        phi2_deg = 15.0
        idx = np.arange(ring_size)
        xy_angles = 2.0 * np.pi * idx / ring_size
        x = ring_pucker.DEFAULT_RING_RADIUS * np.cos(xy_angles)
        y = ring_pucker.DEFAULT_RING_RADIUS * np.sin(xy_angles)
        phi2_rad = np.radians(phi2_deg)
        z = np.sqrt(2.0 / ring_size) * amplitude * np.cos(phi2_rad + 2.0 * np.pi * 2 * idx / ring_size)
        ring_coords = np.column_stack([x, y, z])
        label = ring_pucker.classify_pucker(ring_coords)
        self.assertIn(label, ('boat', 'twist-boat'))
        # Pin the exact deterministic outcome of the half-open bin logic at the boundary.
        self.assertEqual(label, 'boat')

    def test_degenerate_collinear_ring_raises_ring_pucker_error(self):
        """A collinear ring of points has no well-defined ring normal and raises RingPuckerError."""
        ring_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        with self.assertRaises(ring_pucker.RingPuckerError):
            ring_pucker.puckering_amplitude(ring_coords)

    def test_invalid_ring_shape_raises_ring_pucker_error(self):
        """Ring coordinates that are not an (N x 3) array with N >= 3 raise RingPuckerError."""
        with self.assertRaises(ring_pucker.RingPuckerError):
            ring_pucker.puckering_amplitude([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        with self.assertRaises(ring_pucker.RingPuckerError):
            ring_pucker.puckering_amplitude([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def test_validate_ring_order_raises_for_scrambled_ring(self):
        """A ring reindexed out of connectivity order fails the consecutive-bond-length check."""
        ring_coords = np.asarray(ring_pucker.ideal_pucker_geometry(6, 'chair'))
        scrambled = ring_coords[[0, 2, 4, 1, 3, 5]]
        with self.assertRaises(ring_pucker.RingPuckerError):
            ring_pucker.validate_ring_order(scrambled)
        with self.assertRaises(ring_pucker.RingPuckerError):
            ring_pucker.cremer_pople_params(scrambled)
        with self.assertRaises(ring_pucker.RingPuckerError):
            ring_pucker.puckering_amplitude(scrambled)

    def test_validate_ring_order_passes_for_ideal_geometries(self):
        """Ideal, connectivity-ordered ring geometries pass the consecutive-bond-length check."""
        for ring_size, label in [(6, 'chair'), (6, 'boat'), (6, 'twist-boat'), (5, 'envelope'), (5, 'twist')]:
            ring_coords = ring_pucker.ideal_pucker_geometry(ring_size, label)
            ring_pucker.validate_ring_order(ring_coords)

    def test_rdkit_cyclohexane_conformers_classify_honestly(self):
        """RDKit-embedded and MMFF-optimized cyclohexane conformers, across several seeds, each get
        an honest pucker label (whatever local minimum the embedding actually settled into), and at
        least one of them settles into a chair with the expected amplitude. Seeds are not
        cherry-picked to force a particular outcome."""
        valid_labels = {'chair', 'boat', 'twist-boat', 'half-chair'}
        found_chair_with_expected_amplitude = False
        for seed in (0, 1, 2, 7, 42):
            mol = Chem.MolFromSmiles('C1CCCCC1')
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=seed)
            AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
            conf = mol.GetConformer()
            ring_coords = [list(conf.GetAtomPosition(i)) for i in range(6)]

            params = ring_pucker.cremer_pople_params(ring_coords)
            label = ring_pucker.classify_pucker(ring_coords)
            self.assertIn(label, valid_labels)
            if label == 'chair' and 0.55 <= params.amplitude <= 0.65:
                found_chair_with_expected_amplitude = True

        self.assertTrue(found_chair_with_expected_amplitude)

    def test_hardcoded_chair_geometry_classifies_as_chair_independently(self):
        """A hand-built idealized chair cyclohexane geometry, constructed independently of
        ideal_pucker_geometry (i.e. not by calling any function in this module), classifies as a
        chair with a physically sensible puckering amplitude. This guards against a shared
        sign/phase bug that could hide identically in both the generator and the analyzer."""
        ring_radius = 1.46  # Angstrom, plausible chair cyclohexane ring circumradius.
        half_height = 0.25  # Angstrom, plausible alternating out-of-plane displacement per atom.
        ring_coords = []
        for i in range(6):
            angle_rad = np.radians(60.0 * i)
            x = ring_radius * np.cos(angle_rad)
            y = ring_radius * np.sin(angle_rad)
            z = half_height if i % 2 == 0 else -half_height
            ring_coords.append([x, y, z])

        self.assertEqual(ring_pucker.classify_pucker(ring_coords), 'chair')
        amplitude = ring_pucker.puckering_amplitude(ring_coords)
        self.assertTrue(0.5 <= amplitude <= 0.7)


if __name__ == '__main__':
    unittest.main()
