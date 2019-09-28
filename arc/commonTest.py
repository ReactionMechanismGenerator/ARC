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
from arc.exceptions import InputError, SettingsError
from arc.species.converter import str_to_xyz
from arc.settings import arc_path, servers


class TestARC(unittest.TestCase):
    """
    Contains unit tests for ARC's common module
    """

    def test_read_yaml_file(self):
        """Test the read_yaml_file() function"""
        restart_path = os.path.join(arc_path, 'arc', 'testing', 'restart', 'restart(H,H2O2,N2H3,CH3CO2).yml')
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

    def test_initialize_job_types(self):
        """Test the initialize_job_types() function"""
        job_types_0 = {'conformers': False, 'opt': True, 'fine': True, 'freq': True, 'sp': False, 'rotors': False}
        job_types_0_expected = {'rotors': False, 'bde': False, 'conformers': False, 'fine': True,
                                'freq': True, 'onedmin': False, 'opt': True, 'orbitals': False, 'sp': False}
        job_types_0_initialized = common.initialize_job_types(job_types_0)
        self.assertEqual(job_types_0_expected, job_types_0_initialized)

        job_types_1 = {}
        job_types_1_expected = {'rotors': True, 'bde': False, 'conformers': True, 'fine': True, 'freq': True,
                                'onedmin': False, 'opt': True, 'orbitals': False, 'sp': True}
        job_types_1_initialized = common.initialize_job_types(job_types_1)
        self.assertEqual(job_types_1_expected, job_types_1_initialized)

        job_types_2 = {'bde': True}
        job_types_2_expected = {'rotors': True, 'bde': True, 'conformers': True, 'fine': True, 'freq': True,
                                'onedmin': False, 'opt': True, 'orbitals': False, 'sp': True}
        job_types_2_initialized = common.initialize_job_types(job_types_2)
        self.assertEqual(job_types_2_expected, job_types_2_initialized)
        with self.assertRaises(InputError):
            job_types_3 = {'fake_job': True}
            common.initialize_job_types(job_types_3)

    def test_calculate_dihedral_angle(self):
        """Test calculating a dihedral angle"""
        propene = str_to_xyz("""C       1.22905000   -0.16449200    0.00000000
C      -0.13529200    0.45314000    0.00000000
C      -1.27957200   -0.21983000    0.00000000
H       1.17363000   -1.25551200    0.00000000
H       1.79909600    0.15138400    0.87934300
H       1.79909600    0.15138400   -0.87934300
H      -0.16831500    1.54137600    0.00000000
H      -2.23664600    0.28960500    0.00000000
H      -1.29848800   -1.30626200    0.00000000""")
        hydrazine = str_to_xyz("""N       0.70683700   -0.07371000   -0.21400700
N      -0.70683700    0.07371000   -0.21400700
H       1.11984200    0.81113900   -0.47587600
H       1.07456200   -0.35127300    0.68988300
H      -1.11984200   -0.81113900   -0.47587600
H      -1.07456200    0.35127300    0.68988300""")
        cj_11974 = str_to_xyz("""C 	5.675	2.182	1.81
O 	4.408	1.923	1.256
C 	4.269	0.813	0.479
C 	5.303	-0.068	0.178
C 	5.056	-1.172	-0.639
C 	3.794	-1.414	-1.169
C 	2.77	-0.511	-0.851
C 	2.977	0.59	-0.032
C 	1.872	1.556	0.318
N 	0.557	1.029	-0.009
C 	-0.537	1.879	0.448
C 	-0.535	3.231	-0.298
C 	-1.831	3.983	0.033
C 	-3.003	3.199	-0.61
N 	-2.577	1.854	-0.99
C 	-1.64	1.962	-2.111
C 	-0.501	2.962	-1.805
C 	-1.939	1.236	0.178
C 	-1.971	-0.305	0.069
C 	-3.385	-0.794	-0.209
C 	-4.336	-0.893	0.81
C 	-5.631	-1.324	0.539
C 	-5.997	-1.673	-0.759
C 	-5.056	-1.584	-1.781
C 	-3.764	-1.147	-1.505
C 	-1.375	-1.024	1.269
C 	-1.405	-0.508	2.569
C 	-0.871	-1.226	3.638
C 	-0.296	-2.475	3.429
C 	-0.259	-3.003	2.14
C 	-0.794	-2.285	1.078
C 	3.533	-2.614	-2.056
C 	2.521	-3.574	-1.424
C 	3.087	-2.199	-3.461
H 	5.569	3.097	2.395
H 	6.433	2.338	1.031
H 	6.003	1.368	2.47
H 	6.302	0.091	0.57
H 	5.874	-1.854	-0.864
H 	1.772	-0.654	-1.257
H 	1.963	1.832	1.384
H 	2.033	2.489	-0.239
H 	0.469	0.13	0.461
H 	-0.445	2.089	1.532
H 	0.328	3.83	0.012
H 	-1.953	4.059	1.122
H 	-1.779	5.008	-0.352
H 	-3.365	3.702	-1.515
H 	-3.856	3.118	0.074
H 	-1.226	0.969	-2.31
H 	-2.211	2.259	-2.999
H 	-0.639	3.906	-2.348
H 	0.466	2.546	-2.105
H 	-2.586	1.501	1.025
H 	-1.36	-0.582	-0.799
H 	-4.057	-0.647	1.831
H 	-6.355	-1.396	1.347
H 	-7.006	-2.015	-0.97
H 	-5.329	-1.854	-2.798
H 	-3.038	-1.07	-2.311
H 	-1.843	0.468	2.759
H 	-0.904	-0.802	4.638
H 	0.125	-3.032	4.262
H 	0.189	-3.977	1.961
H 	-0.772	-2.708	0.075
H 	4.484	-3.155	-2.156
H 	1.543	-3.093	-1.308
H 	2.383	-4.464	-2.049
H 	2.851	-3.899	-0.431
H 	3.826	-1.542	-3.932
H 	2.134	-1.659	-3.429
H 	2.951	-3.078	-4.102""")

        dihedral0 = common.calculate_dihedral_angle(coords=propene['coords'], torsion=[9, 3, 2, 7])
        dihedral1 = common.calculate_dihedral_angle(coords=propene['coords'], torsion=[5, 1, 2, 7])
        self.assertAlmostEqual(dihedral0, 180, 2)
        self.assertAlmostEqual(dihedral1, 59.26447, 2)

        dihedral2 = common.calculate_dihedral_angle(coords=hydrazine['coords'], torsion=[3, 1, 2, 5])
        self.assertAlmostEqual(dihedral2, 148.31829, 2)

        dihedral3 = common.calculate_dihedral_angle(coords=cj_11974['coords'], torsion=[15, 18, 19, 20])
        self.assertAlmostEqual(dihedral3, 308.04758, 2)

    def test_determine_ess(self):
        """Test the determine_ess function"""
        gaussian = os.path.join(arc_path, 'arc', 'testing', 'SO2OO_CBS-QB3.log')
        qchem = os.path.join(arc_path, 'arc', 'testing', 'C2H6_freq_QChem.out')
        molpro = os.path.join(arc_path, 'arc', 'testing', 'CH2O_freq_molpro.out')

        self.assertEqual(common.determine_ess(gaussian), 'gaussian')
        self.assertEqual(common.determine_ess(qchem), 'qchem')
        self.assertEqual(common.determine_ess(molpro), 'molpro')

################################################################################


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
