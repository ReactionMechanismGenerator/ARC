#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests for ARC's common module
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest
import os
import time

from arc.common import read_file, get_git_commit, time_lapse, check_ess_settings, get_atom_radius, get_center_of_mass
from arc.settings import arc_path, servers
from arc.arc_exceptions import InputError, SettingsError

################################################################################


class TestARC(unittest.TestCase):
    """
    Contains unit tests for ARC's common module
    """

    def test_read_file(self):
        """Test the read_file() function"""
        restart_path = os.path.join(arc_path, 'arc', 'testing', 'restart(H,H2O2,N2H3,CH3CO2).yml')
        input_dict = read_file(restart_path)
        self.assertIsInstance(input_dict, dict)
        self.assertTrue('reactions' in input_dict)
        self.assertTrue('freq_level' in input_dict)
        self.assertTrue('use_bac' in input_dict)
        self.assertTrue('ts_guess_level' in input_dict)
        self.assertTrue('running_jobs' in input_dict)

        with self.assertRaises(InputError):
            read_file('nopath')

    def test_get_git_commit(self):
        """Test the get_git_commit() function"""
        git_commit = get_git_commit()
        # output format: ['fafdb957049917ede565cebc58b29899f597fb5a', 'Fri Mar 29 11:09:50 2019 -0400']
        self.assertEqual(len(git_commit[0]), 40)
        self.assertEqual(len(git_commit[1].split()), 6)

    def test_time_lapse(self):
        """Test the time_lapse() function"""
        t0 = time.time()
        time.sleep(2)
        lap = time_lapse(t0)
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

        ess_settings1 = check_ess_settings(ess_settings1)
        ess_settings2 = check_ess_settings(ess_settings2)
        ess_settings3 = check_ess_settings(ess_settings3)
        ess_settings4 = check_ess_settings(ess_settings4)
        ess_settings5 = check_ess_settings(ess_settings5)

        ess_list = [ess_settings1, ess_settings2, ess_settings3, ess_settings4, ess_settings5]

        for ess in ess_list:
            for soft, server_list in ess.items():
                self.assertTrue(soft in ['gaussian', 'molpro', 'qchem'])
                self.assertIsInstance(server_list, list)

        with self.assertRaises(SettingsError):
            ess_settings6 = {'nosoft': ['server1']}
            check_ess_settings(ess_settings6)
        with self.assertRaises(SettingsError):
            ess_settings7 = {'gaussian': ['noserver']}
            check_ess_settings(ess_settings7)

    def test_get_atom_radius(self):
        """Test determining the covalent radius of an atom"""
        self.assertEqual(get_atom_radius('H'), 0.31)
        self.assertEqual(get_atom_radius('Al'), 1.21)
        self.assertIsNone(get_atom_radius('no atom'))
        with self.assertRaises(InputError):
            get_atom_radius(['H', 'C'])

    def test_get_center_of_mass(self):
        """Test identification of the center of mass for given 3D coordinates"""
        xyz = 'H 0.0 1.0 2.0'
        cm = get_center_of_mass(xyz)
        self.assertEqual(cm, (0.0, 1.0, 2.0))

        xyz = """
        C       0.01110400    0.73099100    0.00000000
        H       0.27332400    1.21632500    0.93134200
        H       0.27332400    1.21632500   -0.93134200
        N       0.01045700   -0.65996900   -0.00000000
        H      -0.34564700   -1.09923900    0.83510100
        H      -0.34564700   -1.09923900   -0.83510100
        """
        cm = get_center_of_mass(xyz)
        self.assertAlmostEqual(cm[0], 0.004458, 5)
        self.assertAlmostEqual(cm[1], -0.007781, 5)
        self.assertAlmostEqual(cm[2], 0.0, 5)

################################################################################


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
