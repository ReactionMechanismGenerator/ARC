#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.checks.spin module
"""

import unittest

from arc.checks.spin import (BROKEN_SYMMETRY_S2_THRESHOLD,
                             MIN_S2_SEPARATION,
                             get_spin_projection,
                             yamaguchi_projected_energy,
                             )


class TestYamaguchiProjectedEnergy(unittest.TestCase):
    """
    Contains unit tests for the Yamaguchi approximate spin projection.
    """

    def test_fully_broken_pair_reproduces_the_standard_limit(self):
        """Test that <S**2>_BS = 1, <S**2>_HS = 2 gives the standard 2*E_BS - E_HS limit"""
        e_bs, e_hs = -195.2885, -195.2500
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=e_bs, e_hs=e_hs, s2_bs=1.0, s2_hs=2.0),
                               2 * e_bs - e_hs, places=10)
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=e_bs, e_hs=e_hs, s2_bs=1.0, s2_hs=2.0),
                               -195.3270, places=10)

    def test_no_spin_contamination_reduces_to_the_broken_symmetry_energy(self):
        """Test that <S**2>_BS = 0 returns E_BS unchanged"""
        self.assertEqual(yamaguchi_projected_energy(e_bs=-195.2885, e_hs=-195.25, s2_bs=0.0, s2_hs=2.0),
                         -195.2885)
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=1e-6, s2_hs=2.0),
                               -100.0, places=5)

    def test_a_hand_checkable_intermediate_case(self):
        """Test a partially contaminated case against the formula evaluated by hand"""
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=0.5, s2_hs=2.0),
                               (2.0 * -100.0 - 0.5 * -99.0) / 1.5, places=10)
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=0.5, s2_hs=2.0),
                               -100.3333333333, places=9)

    def test_the_projection_lies_below_the_broken_symmetry_energy(self):
        """Test that removing high-spin contamination lowers the energy when the HS state is higher"""
        projected = yamaguchi_projected_energy(e_bs=-195.2885, e_hs=-195.2500, s2_bs=1.0, s2_hs=2.0)
        self.assertLess(projected, -195.2885)

    def test_degenerate_denominator_returns_none(self):
        """Test that references not separated in <S**2> yield None rather than a wild number"""
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=2.0, s2_hs=2.0))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=2.0,
                                                     s2_hs=2.0 + MIN_S2_SEPARATION / 2))

    def test_a_more_contaminated_bs_than_hs_reference_returns_none(self):
        """Test that an inverted <S**2> ordering is refused instead of extrapolated"""
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=2.0, s2_hs=1.0))

    def test_missing_inputs_return_none(self):
        """Test that any missing argument yields None"""
        self.assertIsNone(yamaguchi_projected_energy(e_bs=None, e_hs=-0.9, s2_bs=1.0, s2_hs=2.0))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=None, s2_bs=1.0, s2_hs=2.0))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=None, s2_hs=2.0))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=1.0, s2_hs=None))


class TestGetSpinProjection(unittest.TestCase):
    """
    Contains unit tests for assembling a spin projection record.
    """

    def test_the_record_carries_every_quantity_the_projection_used(self):
        """Test that the record allows the projected energy to be recomputed from it"""
        record = get_spin_projection(e_bs=-195.2885, e_hs=-195.2500, s2_bs=1.0, s2_hs=2.0,
                                     e_restricted=-195.2693)
        self.assertEqual(record['e_bs'], -195.2885)
        self.assertEqual(record['e_hs'], -195.2500)
        self.assertEqual(record['s2_bs'], 1.0)
        self.assertEqual(record['s2_hs'], 2.0)
        self.assertEqual(record['e_restricted'], -195.2693)
        self.assertEqual(record['scheme'], 'yamaguchi_ap')
        self.assertAlmostEqual(record['e_projected'],
                               yamaguchi_projected_energy(e_bs=record['e_bs'], e_hs=record['e_hs'],
                                                          s2_bs=record['s2_bs'], s2_hs=record['s2_hs']),
                               places=10)

    def test_the_r_u_gap_is_positive_when_the_broken_symmetry_solution_is_lower(self):
        """Test the restricted minus broken-symmetry energy gap"""
        record = get_spin_projection(e_bs=-195.2885, e_hs=-195.25, s2_bs=1.0, s2_hs=2.0,
                                     e_restricted=-195.2693)
        self.assertAlmostEqual(record['r_u_gap'], 0.0192, places=10)
        self.assertGreater(record['r_u_gap'], 0)

    def test_a_collapsed_broken_symmetry_solution_is_reported_as_such(self):
        """Test that a BS optimization that fell back onto the closed-shell solution is flagged"""
        record = get_spin_projection(e_bs=-195.2885, e_hs=-195.25, s2_bs=0.0, s2_hs=2.0,
                                     e_restricted=-195.2885)
        self.assertFalse(record['broken_symmetry'])
        self.assertEqual(record['e_projected'], record['e_bs'])
        self.assertEqual(record['r_u_gap'], 0.0)

    def test_a_genuinely_broken_solution_is_reported_as_such(self):
        """Test that a spin-contaminated BS reference is flagged as symmetry broken"""
        record = get_spin_projection(e_bs=-195.2885, e_hs=-195.25,
                                     s2_bs=BROKEN_SYMMETRY_S2_THRESHOLD * 10, s2_hs=2.0)
        self.assertTrue(record['broken_symmetry'])

    def test_missing_inputs_leave_the_record_undecided(self):
        """Test that absent quantities yield None entries rather than raising"""
        record = get_spin_projection(e_bs=None, e_hs=None, s2_bs=None, s2_hs=None)
        self.assertIsNone(record['e_projected'])
        self.assertIsNone(record['r_u_gap'])
        self.assertIsNone(record['broken_symmetry'])
        self.assertEqual(record['scheme'], 'yamaguchi_ap')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
