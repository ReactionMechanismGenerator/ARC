#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.submit module
"""

import unittest

from arc.job.submit import submit_scripts


class TestSubmit(unittest.TestCase):
    """
    Contains unit tests for the submit module
    """

    def test_servers(self):
        """Test server keys in submit_scripts"""
        for server in submit_scripts.keys():
            self.assertTrue(server in ['pharos', 'c3ddb', 'rmg'])


################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
