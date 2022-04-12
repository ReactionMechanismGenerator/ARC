#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.common module
"""

import unittest

import arc.job.adapters.common as common


class TestJobCommon(unittest.TestCase):
    """
    Contains unit tests for the job.adapters.common module.
    """

    def test_which(self):
        """Test the which() function"""
        ans = common.which(command='python', return_bool=True, raise_error=False)
        self.assertTrue(ans)

        ans = common.which(command='python', return_bool=False, raise_error=False)
        self.assertIn('arc_env/bin/python', ans)

        ans = common.which(command='ls', return_bool=True, raise_error=False)
        self.assertTrue(ans)

        ans = common.which(command='fake_command_1', return_bool=True, raise_error=False)
        self.assertFalse(ans)

        ans = common.which(command=['fake_command_1', 'ARC.py', 'python'], return_bool=False, raise_error=False)
        self.assertIn('ARC.py', ans)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
