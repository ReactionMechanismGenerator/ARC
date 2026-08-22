#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.checks.spin module
"""

import math
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

    def test_a_non_singlet_target_state_is_projected_to_its_own_spin_purity(self):
        """Test the general form against a hand-evaluated broken-symmetry doublet / high-spin quartet pair"""
        projected = yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.9, s2_bs=1.0, s2_hs=3.80, s2_ls=0.75)
        self.assertAlmostEqual(projected, -100.0 + ((1.0 - 0.75) / 2.80) * (-100.0 - -99.9), places=10)
        self.assertAlmostEqual(projected, -100.0089285714, places=9)

    def test_assuming_a_singlet_target_misprojects_a_doublet_by_a_chemically_large_amount(self):
        """Test that the default singlet target and a doublet target differ by tens of kJ/mol"""
        doublet = yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.9, s2_bs=1.0, s2_hs=3.80, s2_ls=0.75)
        singlet = yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.9, s2_bs=1.0, s2_hs=3.80)
        self.assertAlmostEqual(singlet, -100.0357142857, places=9)
        self.assertAlmostEqual(doublet - singlet, 0.0267857142, places=9)

    def test_a_triplet_target_state(self):
        """Test the general form for a triplet target, whose spin-pure <S**2> is 2"""
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=-50.0, e_hs=-49.5, s2_bs=2.4, s2_hs=6.0, s2_ls=2.0),
                               -50.0 + (0.4 / 3.6) * (-50.0 - -49.5), places=10)

    def test_a_target_state_equal_to_the_broken_symmetry_reference_returns_the_bs_energy(self):
        """Test that an uncontaminated BS reference of a non-singlet target returns E_BS"""
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=0.75, s2_hs=3.75,
                                                          s2_ls=0.75),
                               -100.0, places=10)

    def test_references_too_close_in_s_squared_return_none(self):
        """Test that a pair whose <S**2> values nearly coincide is refused rather than amplified"""
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=2.0, s2_hs=2.0))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=2.0, s2_hs=2.002))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-195.2885, e_hs=-195.2500, s2_bs=0.7540, s2_hs=0.7560))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=1.0, s2_hs=1.05))

    def test_a_separation_at_the_floor_is_projected(self):
        """Test that a pair separated by more than the floor is projected rather than refused"""
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=1.0, s2_hs=1.2),
                               -100.0 + (1.0 / 0.2) * (-100.0 - -99.0), places=10)

    def test_a_refused_projection_never_lies_far_below_the_broken_symmetry_energy(self):
        """Test that no near-degenerate pair yields an energy displaced by more than the energy gap"""
        for s2_hs in [2.0, 2.0005, 2.002, 2.05, 2.09]:
            projected = yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=2.0, s2_hs=s2_hs)
            self.assertIsNone(projected)

    def test_an_inverted_pair_is_reported_as_a_mismatched_calculation(self):
        """Test that <S**2>_HS below <S**2>_BS returns None and warns, the pair being inconsistent"""
        with self.assertLogs('arc', level='WARNING') as captured:
            self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=2.0, s2_hs=1.0))
        self.assertTrue(any('below the broken-symmetry' in record for record in captured.output))

    def test_a_near_degenerate_pair_is_not_reported_as_a_mismatched_calculation(self):
        """Test that the benign near-degenerate case is kept off the warning channel"""
        with self.assertNoLogs('arc', level='WARNING'):
            self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=2.0, s2_hs=2.002))

    def test_a_bs_reference_below_the_target_spin_purity_is_refused(self):
        """Test that a BS <S**2> below the target's S(S+1) returns None and warns"""
        with self.assertLogs('arc', level='WARNING') as captured:
            self.assertIsNone(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=0.5, s2_hs=3.5,
                                                          s2_ls=2.0))
        self.assertTrue(any('target low-spin state' in record for record in captured.output))

    def test_missing_inputs_return_none(self):
        """Test that any missing argument yields None"""
        self.assertIsNone(yamaguchi_projected_energy(e_bs=None, e_hs=-0.9, s2_bs=1.0, s2_hs=2.0))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=None, s2_bs=1.0, s2_hs=2.0))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=None, s2_hs=2.0))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=1.0, s2_hs=None))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=1.0, s2_hs=2.0, s2_ls=None))

    def test_non_finite_inputs_return_none(self):
        """Test that NaN and infinite arguments yield None rather than propagating"""
        for bad in [float('nan'), float('inf'), float('-inf')]:
            self.assertIsNone(yamaguchi_projected_energy(e_bs=bad, e_hs=-0.9, s2_bs=1.0, s2_hs=2.0))
            self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=bad, s2_bs=1.0, s2_hs=2.0))
            self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=bad, s2_hs=2.0))
            self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=1.0, s2_hs=bad))
            self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=1.0, s2_hs=2.0, s2_ls=bad))

    def test_a_negative_s_squared_returns_none(self):
        """Test that a negative <S**2>, which no expectation value can be, yields None"""
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=-1.0, s2_hs=2.0))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=1.0, s2_hs=-2.0))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=1.0, s2_hs=2.0, s2_ls=-0.75))


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
        self.assertEqual(record['s2_ls'], 0.0)
        self.assertEqual(record['e_restricted'], -195.2693)
        self.assertEqual(record['scheme'], 'yamaguchi_ap')
        self.assertAlmostEqual(record['e_projected'],
                               yamaguchi_projected_energy(e_bs=record['e_bs'], e_hs=record['e_hs'],
                                                          s2_bs=record['s2_bs'], s2_hs=record['s2_hs'],
                                                          s2_ls=record['s2_ls']),
                               places=10)

    def test_the_target_spin_purity_is_carried_and_applied(self):
        """Test that a non-singlet target is recorded and used by the projection"""
        record = get_spin_projection(e_bs=-100.0, e_hs=-99.9, s2_bs=1.0, s2_hs=3.80, s2_ls=0.75)
        self.assertEqual(record['s2_ls'], 0.75)
        self.assertAlmostEqual(record['e_projected'], -100.0089285714, places=9)

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
        record = get_spin_projection(e_bs=-195.2885, e_hs=-195.25, s2_bs=0.1, s2_hs=2.0)
        self.assertTrue(record['broken_symmetry'])
        record = get_spin_projection(e_bs=-195.2885, e_hs=-195.25, s2_bs=0.001, s2_hs=2.0)
        self.assertFalse(record['broken_symmetry'])

    def test_symmetry_breaking_is_judged_against_the_target_state_not_against_zero(self):
        """Test that a clean doublet is not flagged as broken merely for having <S**2> near 0.75"""
        record = get_spin_projection(e_bs=-100.0, e_hs=-99.0, s2_bs=0.7536, s2_hs=3.75, s2_ls=0.75)
        self.assertFalse(record['broken_symmetry'])
        record = get_spin_projection(e_bs=-100.0, e_hs=-99.0, s2_bs=1.0, s2_hs=3.75, s2_ls=0.75)
        self.assertTrue(record['broken_symmetry'])

    def test_missing_inputs_leave_the_record_undecided(self):
        """Test that absent quantities yield None entries rather than raising"""
        record = get_spin_projection(e_bs=None, e_hs=None, s2_bs=None, s2_hs=None)
        self.assertIsNone(record['e_projected'])
        self.assertIsNone(record['r_u_gap'])
        self.assertIsNone(record['broken_symmetry'])
        self.assertEqual(record['scheme'], 'yamaguchi_ap')

    def test_a_non_finite_s_squared_leaves_symmetry_breaking_undecided(self):
        """Test that an unreadable <S**2> yields None, never False, for broken_symmetry"""
        for bad in [float('nan'), float('inf'), float('-inf')]:
            record = get_spin_projection(e_bs=-195.2885, e_hs=-195.25, s2_bs=bad, s2_hs=2.0)
            self.assertIsNone(record['broken_symmetry'])
            self.assertIsNone(record['e_projected'])
        record = get_spin_projection(e_bs=float('nan'), e_hs=-195.25, s2_bs=1.0, s2_hs=2.0,
                                     e_restricted=-195.2693)
        self.assertIsNone(record['r_u_gap'])
        self.assertIsNone(record['e_projected'])

    def test_the_constants_are_the_ones_the_module_documents(self):
        """Test the physical floors themselves, so a change to either is a deliberate one"""
        self.assertEqual(MIN_S2_SEPARATION, 0.1)
        self.assertEqual(BROKEN_SYMMETRY_S2_THRESHOLD, 1e-2)
        self.assertFalse(math.isnan(MIN_S2_SEPARATION))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
