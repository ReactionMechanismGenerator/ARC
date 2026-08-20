#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.factory module
"""

import unittest

from arc.job.adapter import JobAdapter, JobEnum
from arc.job.factory import get_registered_job_adapters


class TestFactory(unittest.TestCase):
    """
    Contains unit tests for the arc.job.factory module
    """

    def test_get_registered_job_adapters(self):
        """Test that get_registered_job_adapters returns the live registry populated by
        importing arc.job.adapters (registration decorators run at import time)"""
        registered_job_adapters = get_registered_job_adapters()
        self.assertIsInstance(registered_job_adapters, dict)
        self.assertIn(JobEnum('gaussian'), registered_job_adapters)
        for job_adapter_class in registered_job_adapters.values():
            self.assertTrue(issubclass(job_adapter_class, JobAdapter))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
