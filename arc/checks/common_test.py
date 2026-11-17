#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.checks.common module
"""

import datetime
import unittest

import numpy as np

import arc.checks.common as common
from arc.species import ARCSpecies


class TestChecks(unittest.TestCase):
    """
    Contains unit tests for the check module.
    """
    def test_sum_time_delta(self):
        """Test the sum_time_delta() function"""
        dt1 = datetime.timedelta(days=0, minutes=0, seconds=0)
        dt2 = datetime.timedelta(days=0, minutes=0, seconds=0)
        dt3 = datetime.timedelta(days=0, minutes=1, seconds=15)
        dt4 = datetime.timedelta(days=10, minutes=1, seconds=15, microseconds=300)
        fake_dt5 = None
        fake_dt6 = 'fake'
        fake_dt7 = 18.52
        self.assertEqual(common.sum_time_delta([]), datetime.timedelta(days=0, minutes=0, seconds=0))
        self.assertEqual(common.sum_time_delta([dt1]), datetime.timedelta(days=0, minutes=0, seconds=0))
        self.assertEqual(common.sum_time_delta([dt1, dt2]), datetime.timedelta(days=0, minutes=0, seconds=0))
        self.assertEqual(common.sum_time_delta([dt1, dt3]), datetime.timedelta(days=0, minutes=1, seconds=15))
        self.assertEqual(common.sum_time_delta([dt3, dt4]), datetime.timedelta(days=10, minutes=2, seconds=30, microseconds=300))
        self.assertEqual(common.sum_time_delta([dt3, fake_dt5, fake_dt6, fake_dt7]),
                         datetime.timedelta(days=0, minutes=1, seconds=15))

    def test_get_index_of_abs_largest_neg_freq(self):
        """Test the get_index_of_abs_largest_neg_freq() function."""
        self.assertIsNone(common.get_index_of_abs_largest_neg_freq(None))
        self.assertIsNone(common.get_index_of_abs_largest_neg_freq(np.array([], np.float64)))
        self.assertIsNone(common.get_index_of_abs_largest_neg_freq(np.array([1, 320.5], np.float64)))
        self.assertEqual(common.get_index_of_abs_largest_neg_freq(np.array([-1], np.float64)), 0)
        self.assertEqual(common.get_index_of_abs_largest_neg_freq(np.array([-1, 320.5], np.float64)), 0)
        self.assertEqual(common.get_index_of_abs_largest_neg_freq(np.array([320.5, -1], np.float64)), 1)
        self.assertEqual(common.get_index_of_abs_largest_neg_freq(np.array([320.5, -1, -80, -90, 5000],
                                                                          np.float64)), 3)
        self.assertEqual(common.get_index_of_abs_largest_neg_freq(np.array([-320.5, -1, -80, -90, 5000],
                                                                          np.float64)), 0)

    def test_get_i_from_job_name(self):
        """Test the get_i_from_job_name() function"""
        self.assertIsNone(common.get_i_from_job_name(''))
        self.assertIsNone(common.get_i_from_job_name('some_job_name'))
        self.assertEqual(common.get_i_from_job_name('conf_opt_3'), 3)
        self.assertEqual(common.get_i_from_job_name('conf_opt_33'), 33)
        self.assertEqual(common.get_i_from_job_name('conf_opt_3355'), 3355)
        self.assertEqual(common.get_i_from_job_name('tsg2'), 2)

    def test_get_conformer_job_name(self):
        """Test the get_conformer_job_name() function"""
        self.assertEqual(common.get_conformer_job_name('conf_opt', 0), 'conf_opt_0')
        self.assertEqual(common.get_conformer_job_name('conf_sp', 3), 'conf_sp_3')
        self.assertEqual(common.get_conformer_job_name('conf_opt', 3355), 'conf_opt_3355')
        self.assertTrue(common.is_conformer_job(common.get_conformer_job_name('conf_sp', 12)))
        for job_type in common.CONFORMER_JOB_TYPES:
            for i in [0, 7, 99999]:
                self.assertEqual(common.get_i_from_job_name(common.get_conformer_job_name(job_type, i)), i)

    def test_is_ts_check_exempt(self):
        """
        Test the is_ts_check_exempt() function.
        """
        self.assertFalse(common.is_ts_check_exempt('NMD', {'NMD': False, 'E0': True}))
        self.assertFalse(common.is_ts_check_exempt('E0', {'e_elect': False, 'E0': True}))
        self.assertTrue(common.is_ts_check_exempt('e_elect', {'e_elect': False, 'E0': True}))
        self.assertFalse(common.is_ts_check_exempt('e_elect', {'e_elect': False, 'E0': False}))
        self.assertFalse(common.is_ts_check_exempt('e_elect', {'e_elect': False, 'E0': None}))
        self.assertFalse(common.is_ts_check_exempt('e_elect', dict()))

    def test_get_ts_validation_comment(self):
        """
        Test the get_ts_validation_comment() function.
        """
        self.assertIsNone(common.get_ts_validation_comment(None))
        ts = ARCSpecies(label='TS0', is_ts=True)
        self.assertIsNone(common.get_ts_validation_comment(ts))
        ts.ts_checks['IRC'] = True
        self.assertIsNone(common.get_ts_validation_comment(ts))
        ts.ts_checks['IRC'] = None
        ts.ts_checks['NMD'] = False
        self.assertIsNone(common.get_ts_validation_comment(ts))
        ts.ts_checks['IRC'] = False
        comment = common.get_ts_validation_comment(ts)
        self.assertIn(common.TS_IRC_FAILED_MARKER, comment)
        self.assertIn('NMD', comment)
        ts.ts_checks['NMD'] = True
        comment = common.get_ts_validation_comment(ts)
        self.assertIn(common.TS_IRC_FAILED_MARKER, comment)
        self.assertNotIn('NMD', comment)
        ts.ts_checks['e_elect'] = False
        ts.ts_checks['E0'] = True
        self.assertNotIn('e_elect', common.get_ts_validation_comment(ts))
        ts.ts_checks['E0'] = False
        comment = common.get_ts_validation_comment(ts)
        self.assertIn('e_elect', comment)
        self.assertIn('E0', comment)


    def test_get_ts_validation_caution(self):
        """Test that only a TS whose IRC is unverified because an IRC job failed gets a caution"""
        self.assertIsNone(common.get_ts_validation_caution(None))
        ts = ARCSpecies(label='TS0', is_ts=True)
        self.assertIsNone(common.get_ts_validation_caution(ts))
        ts.ts_checks['warnings'] = 'NMD check could not run; '
        self.assertIsNone(common.get_ts_validation_caution(ts))
        ts.ts_checks['warnings'] += f'{common.IRC_JOB_FAILED_WARNING}IRC job irc_a1 (forward) failed: x; '
        caution = common.get_ts_validation_caution(ts)
        self.assertTrue(caution.startswith(f'{common.TS_IRC_NOT_CONFIRMED_MARKER} ({common.IRC_CAUSE_JOB_FAILED})'))
        self.assertIsNone(common.get_ts_validation_comment(ts))
        for verdict in [True, False]:
            with self.subTest(irc=verdict):
                ts.ts_checks['IRC'] = verdict
                self.assertIsNone(common.get_ts_validation_caution(ts))
        ts.ts_checks['warnings'] = None
        ts.ts_checks['IRC'] = None
        self.assertIsNone(common.get_ts_validation_caution(ts))

    def test_get_ts_validation_caution_names_its_cause(self):
        """Test that a TS whose requested IRC reached no verdict gets a caution naming why"""
        converged = {'paths': {'geo': 'opt.log'}}
        unconverged = {'paths': {'geo': ''}}
        cases = {common.IRC_CAUSE_JOB_FAILED: (f'{common.IRC_JOB_FAILED_WARNING}IRC job irc_a1 failed: x; ',
                                               'IRC_TS0_1', {'IRC_TS0_1': converged}),
                 common.IRC_CAUSE_ENDPOINT_OPT_NOT_CONVERGED: ('', 'IRC_TS0_1 IRC_TS0_2',
                                                               {'IRC_TS0_1': converged, 'IRC_TS0_2': unconverged}),
                 common.IRC_CAUSE_COMPARISON_INCONCLUSIVE: ('', 'IRC_TS0_1 IRC_TS0_2',
                                                            {'IRC_TS0_1': converged, 'IRC_TS0_2': converged}),
                 common.IRC_CAUSE_NOT_COMPLETED: ('', None, dict())}
        for cause, (warnings, irc_label, output_dict) in cases.items():
            with self.subTest(cause=cause):
                ts = ARCSpecies(label='TS0', is_ts=True)
                ts.ts_checks['warnings'], ts.irc_label = warnings, irc_label
                caution = common.get_ts_validation_caution(ts, irc_requested=True, output_dict=output_dict)
                self.assertTrue(caution.startswith(f'{common.TS_IRC_NOT_CONFIRMED_MARKER} ({cause}): '))
                self.assertIn(common.IRC_NOT_CONFIRMED_CAUSES[cause], caution)
                for verdict in [True, False]:
                    with self.subTest(cause=cause, irc=verdict):
                        ts.ts_checks['IRC'] = verdict
                        self.assertIsNone(common.get_ts_validation_caution(ts, irc_requested=True,
                                                                           output_dict=output_dict))
        ts = ARCSpecies(label='TS0', is_ts=True)
        ts.irc_label = 'IRC_TS0_1 IRC_TS0_2'
        self.assertIsNone(common.get_ts_validation_caution(ts, irc_requested=False,
                                                           output_dict={'IRC_TS0_1': unconverged}))
        self.assertIn(common.IRC_CAUSE_ENDPOINT_OPT_NOT_CONVERGED,
                      common.get_ts_validation_caution(ts, irc_requested=True))
        self.assertIsNone(common.get_ts_validation_caution(None, irc_requested=True))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
