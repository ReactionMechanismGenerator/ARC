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

    def test_rdkit_cyclohexane_classifies_as_chair(self):
        """An RDKit-embedded and MMFF-optimized cyclohexane classifies as a chair."""
        mol = Chem.MolFromSmiles('C1CCCCC1')
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=0)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        conf = mol.GetConformer()
        ring_coords = [list(conf.GetAtomPosition(i)) for i in range(6)]

        params = ring_pucker.cremer_pople_params(ring_coords)
        self.assertEqual(ring_pucker.classify_pucker(ring_coords), 'chair')
        self.assertTrue(0.55 <= params.amplitude <= 0.65)
        self.assertTrue(params.theta_deg <= 45.0 or params.theta_deg >= 135.0)


if __name__ == '__main__':
    unittest.main()
