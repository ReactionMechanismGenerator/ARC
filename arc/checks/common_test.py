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


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
