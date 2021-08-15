#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.checks.common module
"""

import datetime
import unittest

import arc.checks.common as common


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
        self.assertEqual(common.sum_time_delta([]), datetime.timedelta(days=0, minutes=0, seconds=0))
        self.assertEqual(common.sum_time_delta([dt1]), datetime.timedelta(days=0, minutes=0, seconds=0))
        self.assertEqual(common.sum_time_delta([dt1, dt2]), datetime.timedelta(days=0, minutes=0, seconds=0))
        self.assertEqual(common.sum_time_delta([dt1, dt3]), datetime.timedelta(days=0, minutes=1, seconds=15))
        self.assertEqual(common.sum_time_delta([dt3, dt4]), datetime.timedelta(days=10, minutes=2, seconds=30, microseconds=300))

    def test_get_i_from_job_name(self):
        """Test the get_i_from_job_name() function"""
        self.assertIsNone(common.get_i_from_job_name(''))
        self.assertIsNone(common.get_i_from_job_name('some_job_name'))
        self.assertEqual(common.get_i_from_job_name('conformer3'), 3)
        self.assertEqual(common.get_i_from_job_name('conformer33'), 33)
        self.assertEqual(common.get_i_from_job_name('conformer3355'), 3355)
        self.assertEqual(common.get_i_from_job_name('tsg2'), 2)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
