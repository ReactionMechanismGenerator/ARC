#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.species.ring_pucker module
"""

import unittest

import numpy as np

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


if __name__ == '__main__':
    unittest.main()
