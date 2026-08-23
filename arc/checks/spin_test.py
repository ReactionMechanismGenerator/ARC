#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.checks.spin module
"""

import math
import unittest

from arc.checks.spin import (BROKEN_SYMMETRY_S2_THRESHOLD,
                             MAX_PROJECTION_AMPLIFICATION,
                             MIN_S2_SEPARATION,
                             get_spin_projection,
                             target_low_spin_s_squared,
                             yamaguchi_projected_energy,
                             )

LEVEL = 'wb97xd/def2tzvp'
XYZ = {'symbols': ('H', 'H'), 'isotopes': (1, 1),
       'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.74))}


class TestTargetLowSpinSSquared(unittest.TestCase):
    """
    Contains unit tests for the spin-pure <S**2> of a target low-spin state.
    """

    def test_the_first_three_multiplicities(self):
        """Test S(S+1) for a singlet, a doublet and a triplet"""
        self.assertEqual(target_low_spin_s_squared(1), 0.0)
        self.assertEqual(target_low_spin_s_squared(2), 0.75)
        self.assertEqual(target_low_spin_s_squared(3), 2.0)

    def test_a_missing_multiplicity_is_refused_rather_than_assumed_to_be_a_singlet(self):
        """Test that no multiplicity yields None and a warning, never the singlet value"""
        with self.assertLogs('arc', level='WARNING') as captured:
            self.assertIsNone(target_low_spin_s_squared(None))
        self.assertTrue(any('is not a spin multiplicity' in record for record in captured.output))

    def test_a_value_that_names_no_spin_state_is_refused(self):
        """Test that a multiplicity below one, non-numeric, non-finite or fractional yields None"""
        for multiplicity in [0, -2, 'doublet', float('nan'), float('inf'), float('-inf'), 1.5, 2.5, 0.5]:
            with self.assertLogs('arc', level='WARNING'):
                self.assertIsNone(target_low_spin_s_squared(multiplicity),
                                  msg=f'{multiplicity!r} was accepted as a multiplicity')


class TestYamaguchiProjectedEnergy(unittest.TestCase):
    """
    Contains unit tests for the Yamaguchi approximate spin projection.
    """

    def test_fully_broken_pair_reproduces_the_standard_limit(self):
        """Test that <S**2>_BS = 1, <S**2>_HS = 2 gives the standard 2*E_BS - E_HS limit"""
        e_bs, e_hs = -195.2885, -195.2500
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=e_bs, e_hs=e_hs, s2_bs=1.0, s2_hs=2.0,
                                                          multiplicity=1),
                               2 * e_bs - e_hs, places=10)
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=e_bs, e_hs=e_hs, s2_bs=1.0, s2_hs=2.0,
                                                          multiplicity=1),
                               -195.3270, places=10)

    def test_no_spin_contamination_reduces_to_the_broken_symmetry_energy(self):
        """Test that <S**2>_BS = 0 returns E_BS unchanged"""
        self.assertEqual(yamaguchi_projected_energy(e_bs=-195.2885, e_hs=-195.25, s2_bs=0.0, s2_hs=2.0,
                                                    multiplicity=1),
                         -195.2885)
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=1e-6, s2_hs=2.0,
                                                          multiplicity=1),
                               -100.0, places=5)

    def test_a_hand_checkable_intermediate_case(self):
        """Test a partially contaminated case against the formula evaluated by hand"""
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=0.5, s2_hs=2.0,
                                                          multiplicity=1),
                               (2.0 * -100.0 - 0.5 * -99.0) / 1.5, places=10)
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=0.5, s2_hs=2.0,
                                                          multiplicity=1),
                               -100.3333333333, places=9)

    def test_the_projection_lies_below_the_broken_symmetry_energy(self):
        """Test that removing high-spin contamination lowers the energy when the HS state is higher"""
        projected = yamaguchi_projected_energy(e_bs=-195.2885, e_hs=-195.2500, s2_bs=1.0, s2_hs=2.0,
                                               multiplicity=1)
        self.assertLess(projected, -195.2885)

    def test_a_non_singlet_target_state_is_projected_to_its_own_spin_purity(self):
        """Test the general form against a hand-evaluated broken-symmetry doublet / high-spin quartet pair"""
        projected = yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.9, s2_bs=1.0, s2_hs=3.80, multiplicity=2)
        self.assertAlmostEqual(projected, -100.0 + ((1.0 - 0.75) / 2.80) * (-100.0 - -99.9), places=10)
        self.assertAlmostEqual(projected, -100.0089285714, places=9)

    def test_projecting_a_doublet_onto_a_singlet_target_misses_by_a_chemically_large_amount(self):
        """Test that the target multiplicity moves the answer by tens of kJ/mol"""
        doublet = yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.9, s2_bs=1.0, s2_hs=3.80, multiplicity=2)
        singlet = yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.9, s2_bs=1.0, s2_hs=3.80, multiplicity=1)
        self.assertAlmostEqual(singlet, -100.0357142857, places=9)
        self.assertAlmostEqual(doublet - singlet, 0.0267857142, places=9)

    def test_a_triplet_target_state(self):
        """Test the general form for a triplet target, whose spin-pure <S**2> is 2"""
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=-50.0, e_hs=-49.5, s2_bs=2.4, s2_hs=6.0,
                                                          multiplicity=3),
                               -50.0 + (0.4 / 3.6) * (-50.0 - -49.5), places=10)

    def test_a_target_state_equal_to_the_broken_symmetry_reference_returns_the_bs_energy(self):
        """Test that an uncontaminated BS reference of a non-singlet target returns E_BS"""
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=0.75, s2_hs=3.75,
                                                          multiplicity=2),
                               -100.0, places=10)

    def test_references_too_close_in_s_squared_return_none(self):
        """Test that a pair whose <S**2> values nearly coincide is refused rather than amplified"""
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=2.0, s2_hs=2.0,
                                                     multiplicity=1))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=2.0, s2_hs=2.002,
                                                     multiplicity=1))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-195.2885, e_hs=-195.2500, s2_bs=0.7540,
                                                     s2_hs=0.7560, multiplicity=1))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=1.0, s2_hs=1.05,
                                                     multiplicity=1))

    def test_a_separation_exactly_at_the_floor_is_projected(self):
        """Test that the floor itself is inside the accepted range and anything below it is not"""
        s2_bs, s2_hs = 0.1, 0.2
        self.assertEqual(s2_hs - s2_bs, MIN_S2_SEPARATION)
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=s2_bs, s2_hs=s2_hs,
                                                          multiplicity=1),
                               -100.0 + (s2_bs / MIN_S2_SEPARATION) * (-100.0 - -99.0), places=10)
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=s2_bs,
                                                     s2_hs=s2_hs - 1e-9, multiplicity=1))

    def test_a_pair_separated_by_exactly_the_floor_and_spin_pure_returns_the_bs_energy(self):
        """Test the floor with a BS reference carrying no contamination to remove"""
        self.assertEqual(MIN_S2_SEPARATION - 0.0, MIN_S2_SEPARATION)
        self.assertEqual(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=0.0,
                                                    s2_hs=MIN_S2_SEPARATION, multiplicity=1),
                         -100.0)

    def test_a_refusal_below_the_floor_is_warned_about_rather_than_silent(self):
        """Test that a near-degenerate pair says why it was refused on the warning channel"""
        with self.assertLogs('arc', level='WARNING') as captured:
            self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=2.0, s2_hs=2.002,
                                                         multiplicity=1))
        message = ' '.join(captured.output)
        self.assertIn('are separated by', message)
        self.assertNotIn('below the broken-symmetry', message)

    def test_a_refused_projection_never_lies_far_below_the_broken_symmetry_energy(self):
        """Test that no near-degenerate pair yields an energy displaced by more than the energy gap"""
        for s2_hs in [2.0, 2.0005, 2.002, 2.05, 2.09]:
            projected = yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=2.0, s2_hs=s2_hs,
                                                   multiplicity=1)
            self.assertIsNone(projected)

    def test_an_over_amplified_pair_is_refused(self):
        """Test that a BS reference more high-spin than target-spin is refused rather than amplified"""
        with self.assertLogs('arc', level='WARNING') as captured:
            self.assertIsNone(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=1.0, s2_hs=1.2,
                                                         multiplicity=1))
        self.assertTrue(any('an amplification of' in record for record in captured.output))

    def test_an_amplification_exactly_at_the_cap_is_projected(self):
        """Test that the cap itself is inside the accepted range and anything above it is not"""
        s2_bs, s2_hs = 1.0, 1.5
        self.assertEqual(s2_bs / (s2_hs - s2_bs), MAX_PROJECTION_AMPLIFICATION)
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=s2_bs, s2_hs=s2_hs,
                                                          multiplicity=1),
                               -102.0, places=10)
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=s2_bs, s2_hs=1.49,
                                                     multiplicity=1))

    def test_the_amplification_cap_bounds_how_far_the_projection_moves_the_energy(self):
        """Test that every accepted projection stays within the cap times the BS to HS gap"""
        e_bs, e_hs = -100.0, -99.0
        for s2_bs, s2_hs in [(1.0, 2.0), (0.5, 2.0), (1.0, 1.5), (1.2, 1.8), (0.0, 0.1)]:
            projected = yamaguchi_projected_energy(e_bs=e_bs, e_hs=e_hs, s2_bs=s2_bs, s2_hs=s2_hs,
                                                   multiplicity=1)
            if projected is not None:
                self.assertLessEqual(abs(projected - e_bs),
                                     MAX_PROJECTION_AMPLIFICATION * abs(e_bs - e_hs) + 1e-10,
                                     msg=f'({s2_bs}, {s2_hs}) moved the energy past the cap')

    def test_a_bs_reference_spin_pure_to_within_the_tolerance_is_not_moved_upward(self):
        """Test that a BS <S**2> marginally below the target projects to E_BS rather than above it"""
        e_bs, e_hs = -100.0, -99.0
        for s2_bs in [0.75, 0.70, 0.6501]:
            projected = yamaguchi_projected_energy(e_bs=e_bs, e_hs=e_hs, s2_bs=s2_bs, s2_hs=2.0,
                                                   multiplicity=2)
            self.assertIsNotNone(projected, msg=f'a BS <S**2> of {s2_bs} was refused')
            self.assertLessEqual(projected, e_bs,
                                 msg=f'a BS <S**2> of {s2_bs} projected above the BS energy')
        self.assertAlmostEqual(yamaguchi_projected_energy(e_bs=e_bs, e_hs=e_hs, s2_bs=0.7, s2_hs=2.0,
                                                          multiplicity=2),
                               e_bs, places=10)

    def test_an_inverted_pair_is_reported_as_a_mismatched_calculation(self):
        """Test that <S**2>_HS below <S**2>_BS returns None and warns, the pair being inconsistent"""
        with self.assertLogs('arc', level='WARNING') as captured:
            self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=2.0, s2_hs=1.0,
                                                         multiplicity=1))
        self.assertTrue(any('below the broken-symmetry' in record for record in captured.output))

    def test_a_bs_reference_below_the_target_spin_purity_is_refused(self):
        """Test that a BS <S**2> below the target's S(S+1) returns None and warns"""
        with self.assertLogs('arc', level='WARNING') as captured:
            self.assertIsNone(yamaguchi_projected_energy(e_bs=-100.0, e_hs=-99.0, s2_bs=0.5, s2_hs=3.5,
                                                         multiplicity=3))
        self.assertTrue(any('target low-spin state' in record for record in captured.output))

    def test_missing_inputs_return_none(self):
        """Test that any missing argument yields None"""
        self.assertIsNone(yamaguchi_projected_energy(e_bs=None, e_hs=-0.9, s2_bs=1.0, s2_hs=2.0,
                                                     multiplicity=1))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=None, s2_bs=1.0, s2_hs=2.0,
                                                     multiplicity=1))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=None, s2_hs=2.0,
                                                     multiplicity=1))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=1.0, s2_hs=None,
                                                     multiplicity=1))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=1.0, s2_hs=2.0,
                                                     multiplicity=None))

    def test_non_finite_inputs_return_none(self):
        """Test that NaN and infinite arguments yield None rather than propagating"""
        for bad in [float('nan'), float('inf'), float('-inf')]:
            self.assertIsNone(yamaguchi_projected_energy(e_bs=bad, e_hs=-0.9, s2_bs=1.0, s2_hs=2.0,
                                                         multiplicity=1))
            self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=bad, s2_bs=1.0, s2_hs=2.0,
                                                         multiplicity=1))
            self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=bad, s2_hs=2.0,
                                                         multiplicity=1))
            self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=1.0, s2_hs=bad,
                                                         multiplicity=1))
            self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=1.0, s2_hs=2.0,
                                                         multiplicity=bad))

    def test_a_negative_s_squared_returns_none(self):
        """Test that a negative <S**2>, which no expectation value can be, yields None"""
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=-1.0, s2_hs=2.0,
                                                     multiplicity=1))
        self.assertIsNone(yamaguchi_projected_energy(e_bs=-1.0, e_hs=-0.9, s2_bs=1.0, s2_hs=-2.0,
                                                     multiplicity=1))


class TestGetSpinProjection(unittest.TestCase):
    """
    Contains unit tests for assembling a spin projection record.
    """

    def test_the_record_carries_every_quantity_the_projection_used(self):
        """Test that the record allows the projected energy to be recomputed from it"""
        record = get_spin_projection(e_bs=-195.2885, e_hs=-195.2500, s2_bs=1.0, s2_hs=2.0,
                                     multiplicity=1, level=LEVEL, xyz=XYZ, e_restricted=-195.2693)
        self.assertEqual(record['e_bs'], -195.2885)
        self.assertEqual(record['e_hs'], -195.2500)
        self.assertEqual(record['s2_bs'], 1.0)
        self.assertEqual(record['s2_hs'], 2.0)
        self.assertEqual(record['s2_ls'], 0.0)
        self.assertEqual(record['multiplicity'], 1)
        self.assertEqual(record['e_restricted'], -195.2693)
        self.assertEqual(record['scheme'], 'yamaguchi_ap')
        self.assertAlmostEqual(record['e_projected'],
                               yamaguchi_projected_energy(e_bs=record['e_bs'], e_hs=record['e_hs'],
                                                          s2_bs=record['s2_bs'], s2_hs=record['s2_hs'],
                                                          multiplicity=record['multiplicity']),
                               places=10)

    def test_the_record_names_the_level_and_the_geometry_the_energies_came_from(self):
        """Test that the two energies are bound to the level and geometry that produced them"""
        record = get_spin_projection(e_bs=-195.2885, e_hs=-195.2500, s2_bs=1.0, s2_hs=2.0,
                                     multiplicity=1, level=LEVEL, xyz=XYZ)
        self.assertEqual(record['level'], LEVEL)
        self.assertEqual(record['xyz'], XYZ)

    def test_an_absent_provenance_is_reported_as_absent(self):
        """Test that a record built without a level or a geometry says so rather than omitting the keys"""
        record = get_spin_projection(e_bs=-195.2885, e_hs=-195.2500, s2_bs=1.0, s2_hs=2.0,
                                     multiplicity=1, level=None, xyz=None)
        self.assertIsNone(record['level'])
        self.assertIsNone(record['xyz'])

    def test_the_target_spin_purity_is_carried_and_applied(self):
        """Test that a non-singlet target is recorded and used by the projection"""
        record = get_spin_projection(e_bs=-100.0, e_hs=-99.9, s2_bs=1.0, s2_hs=3.80, multiplicity=2,
                                     level=LEVEL, xyz=XYZ)
        self.assertEqual(record['s2_ls'], 0.75)
        self.assertEqual(record['multiplicity'], 2)
        self.assertAlmostEqual(record['e_projected'], -100.0089285714, places=9)

    def test_the_r_u_gap_is_positive_when_the_broken_symmetry_solution_is_lower(self):
        """Test the restricted minus broken-symmetry energy gap"""
        record = get_spin_projection(e_bs=-195.2885, e_hs=-195.25, s2_bs=1.0, s2_hs=2.0, multiplicity=1,
                                     level=LEVEL, xyz=XYZ, e_restricted=-195.2693)
        self.assertAlmostEqual(record['r_u_gap'], 0.0192, places=10)
        self.assertGreater(record['r_u_gap'], 0)

    def test_a_collapsed_broken_symmetry_solution_is_reported_as_such(self):
        """Test that a BS optimization that fell back onto the closed-shell solution is flagged"""
        record = get_spin_projection(e_bs=-195.2885, e_hs=-195.25, s2_bs=0.0, s2_hs=2.0, multiplicity=1,
                                     level=LEVEL, xyz=XYZ, e_restricted=-195.2885)
        self.assertFalse(record['broken_symmetry'])
        self.assertEqual(record['e_projected'], record['e_bs'])
        self.assertEqual(record['r_u_gap'], 0.0)

    def test_a_genuinely_broken_solution_is_reported_as_such(self):
        """Test that a spin-contaminated BS reference is flagged as symmetry broken"""
        record = get_spin_projection(e_bs=-195.2885, e_hs=-195.25, s2_bs=0.1, s2_hs=2.0, multiplicity=1,
                                     level=LEVEL, xyz=XYZ)
        self.assertTrue(record['broken_symmetry'])
        record = get_spin_projection(e_bs=-195.2885, e_hs=-195.25, s2_bs=0.001, s2_hs=2.0, multiplicity=1,
                                     level=LEVEL, xyz=XYZ)
        self.assertFalse(record['broken_symmetry'])

    def test_a_deviation_exactly_at_the_symmetry_breaking_threshold_is_not_broken(self):
        """Test that the threshold itself lies outside the range reported as symmetry broken"""
        self.assertEqual(BROKEN_SYMMETRY_S2_THRESHOLD - 0.0, BROKEN_SYMMETRY_S2_THRESHOLD)
        record = get_spin_projection(e_bs=-195.2885, e_hs=-195.25, s2_bs=BROKEN_SYMMETRY_S2_THRESHOLD,
                                     s2_hs=2.0, multiplicity=1, level=LEVEL, xyz=XYZ)
        self.assertFalse(record['broken_symmetry'])
        record = get_spin_projection(e_bs=-195.2885, e_hs=-195.25,
                                     s2_bs=BROKEN_SYMMETRY_S2_THRESHOLD * 1.001,
                                     s2_hs=2.0, multiplicity=1, level=LEVEL, xyz=XYZ)
        self.assertTrue(record['broken_symmetry'])

    def test_symmetry_breaking_is_judged_against_the_target_state_not_against_zero(self):
        """Test that a clean doublet is not flagged as broken merely for having <S**2> near 0.75"""
        record = get_spin_projection(e_bs=-100.0, e_hs=-99.0, s2_bs=0.7536, s2_hs=3.75, multiplicity=2,
                                     level=LEVEL, xyz=XYZ)
        self.assertFalse(record['broken_symmetry'])
        record = get_spin_projection(e_bs=-100.0, e_hs=-99.0, s2_bs=1.0, s2_hs=3.75, multiplicity=2,
                                     level=LEVEL, xyz=XYZ)
        self.assertTrue(record['broken_symmetry'])

    def test_missing_inputs_leave_the_record_undecided(self):
        """Test that absent quantities yield None entries rather than raising"""
        record = get_spin_projection(e_bs=None, e_hs=None, s2_bs=None, s2_hs=None, multiplicity=None,
                                     level=None, xyz=None)
        self.assertIsNone(record['e_projected'])
        self.assertIsNone(record['r_u_gap'])
        self.assertIsNone(record['broken_symmetry'])
        self.assertIsNone(record['s2_ls'])
        self.assertEqual(record['scheme'], 'yamaguchi_ap')

    def test_a_non_finite_s_squared_leaves_symmetry_breaking_undecided(self):
        """Test that an unreadable <S**2> yields None, never False, for broken_symmetry"""
        for bad in [float('nan'), float('inf'), float('-inf')]:
            record = get_spin_projection(e_bs=-195.2885, e_hs=-195.25, s2_bs=bad, s2_hs=2.0,
                                         multiplicity=1, level=LEVEL, xyz=XYZ)
            self.assertIsNone(record['broken_symmetry'])
            self.assertIsNone(record['e_projected'])
        record = get_spin_projection(e_bs=float('nan'), e_hs=-195.25, s2_bs=1.0, s2_hs=2.0,
                                     multiplicity=1, level=LEVEL, xyz=XYZ, e_restricted=-195.2693)
        self.assertIsNone(record['r_u_gap'])
        self.assertIsNone(record['e_projected'])

    def test_the_constants_are_the_ones_the_module_documents(self):
        """Test the physical floors themselves, so a change to either is a deliberate one"""
        self.assertEqual(MIN_S2_SEPARATION, 0.1)
        self.assertEqual(MAX_PROJECTION_AMPLIFICATION, 2.0)
        self.assertEqual(BROKEN_SYMMETRY_S2_THRESHOLD, 1e-2)
        self.assertFalse(math.isnan(MIN_S2_SEPARATION))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
