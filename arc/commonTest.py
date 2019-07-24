#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests for ARC's common module
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest
import os
import time

import arc.common as common
from arc.settings import arc_path, servers
from arc.arc_exceptions import InputError, SettingsError

################################################################################


class TestARC(unittest.TestCase):
    """
    Contains unit tests for ARC's common module
    """

    def test_read_yaml_file(self):
        """Test the read_yaml_file() function"""
        restart_path = os.path.join(arc_path, 'arc', 'testing', 'restart(H,H2O2,N2H3,CH3CO2).yml')
        input_dict = common.read_yaml_file(restart_path)
        self.assertIsInstance(input_dict, dict)
        self.assertTrue('reactions' in input_dict)
        self.assertTrue('freq_level' in input_dict)
        self.assertTrue('use_bac' in input_dict)
        self.assertTrue('ts_guess_level' in input_dict)
        self.assertTrue('running_jobs' in input_dict)

        with self.assertRaises(InputError):
            common.read_yaml_file('nopath')

    def test_get_git_commit(self):
        """Test the get_git_commit() function"""
        git_commit = common.get_git_commit()
        # output format: ['fafdb957049917ede565cebc58b29899f597fb5a', 'Fri Mar 29 11:09:50 2019 -0400']
        self.assertEqual(len(git_commit[0]), 40)
        self.assertEqual(len(git_commit[1].split()), 6)

    def test_time_lapse(self):
        """Test the time_lapse() function"""
        t0 = time.time()
        time.sleep(2)
        lap = common.time_lapse(t0)
        self.assertEqual(lap, '00:00:02')

    def test_check_ess_settings(self):
        """Test the check_ess_settings function"""
        server_names = servers.keys()
        ess_settings1 = {'gaussian': [server_names[0]], 'molpro': [server_names[1], server_names[0]],
                         'qchem': [server_names[0]]}
        ess_settings2 = {'gaussian': server_names[0], 'molpro': server_names[1], 'qchem': server_names[0]}
        ess_settings3 = {'gaussian': server_names[0], 'molpro': [server_names[1], server_names[0]],
                         'qchem': server_names[0]}
        ess_settings4 = {'gaussian': server_names[0], 'molpro': server_names[1], 'qchem': server_names[0]}
        ess_settings5 = {'gaussian': 'local', 'molpro': server_names[1], 'qchem': server_names[0]}

        ess_settings1 = common.check_ess_settings(ess_settings1)
        ess_settings2 = common.check_ess_settings(ess_settings2)
        ess_settings3 = common.check_ess_settings(ess_settings3)
        ess_settings4 = common.check_ess_settings(ess_settings4)
        ess_settings5 = common.check_ess_settings(ess_settings5)

        ess_list = [ess_settings1, ess_settings2, ess_settings3, ess_settings4, ess_settings5]

        for ess in ess_list:
            for soft, server_list in ess.items():
                self.assertTrue(soft in ['gaussian', 'molpro', 'qchem'])
                self.assertIsInstance(server_list, list)

        with self.assertRaises(SettingsError):
            ess_settings6 = {'nosoft': ['server1']}
            common.check_ess_settings(ess_settings6)
        with self.assertRaises(SettingsError):
            ess_settings7 = {'gaussian': ['noserver']}
            common.check_ess_settings(ess_settings7)

    def test_min_list(self):
        """Test the min_list() function"""
        lst = []
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, None)

        lst = [None]
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, None)

        lst = [None, None]
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, None)

        lst = [0]
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, 0)

        lst = [-8]
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, -8)

        lst = [-8, -80]
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, -80)

        lst = [-8, None]
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, -8)

        lst = [-8, -8, -8, -8]
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, -8)

        lst = [-8, None, None, 100, -79, None]
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, -79)

################################################################################


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
