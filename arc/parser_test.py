#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the parser functions
"""

import numpy as np
import os
import unittest

import arc.parser as parser
from arc.common import ARC_PATH, almost_equal_coords
from arc.species import ARCSpecies
from arc.species.converter import str_to_xyz, xyz_to_str


class TestParser(unittest.TestCase):
    """
    Contains unit tests for the parser functions
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None

    def test_parse_frequencies(self):
        """Test frequency parsing"""
        no3_path = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'NO3_freq_QChem_fails_on_cclib.out')
        c2h6_path = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'C2H6_freq_QChem.out')
        so2oo_path = os.path.join(ARC_PATH, 'arc', 'testing', 'composite', 'SO2OO_CBS-QB3.log')
        ch2o_path_molpro = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'CH2O_freq_molpro.out')
        ch2o_path_terachem = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'CH2O_freq_terachem.dat')
        ch2o_path_terachem_output = os.path.join(ARC_PATH, 'arc', 'testing', 'freq',
                                                 'formaldehyde_freq_terachem_output.out')
        ncc_path_terachem_output = os.path.join(ARC_PATH, 'arc', 'testing', 'freq',
                                                'ethylamine_freq_terachem_output.out')
        orca_path = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'orca_example_freq.log')
        dual_freq_path = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'dual_freq_output.out')
        co2_xtb_freqs_path = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'CO2_xtb.out')
        ts_xtb_freqs_path = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'TS_NH2+N2H3_xtb.out')
        yml_freqs_path = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'output.yml')
        vibspectrum_path = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'vibspectrum')
        mock_path = os.path.join(ARC_PATH, 'arc', 'testing', 'mockter.yml')

        no3_freqs = parser.parse_frequencies(path=no3_path, software='QChem')
        c2h6_freqs = parser.parse_frequencies(path=c2h6_path, software='QChem')
        so2oo_freqs = parser.parse_frequencies(path=so2oo_path, software='Gaussian')
        ch2o_molpro_freqs = parser.parse_frequencies(path=ch2o_path_molpro, software='Molpro')
        ch2o_terachem_freqs = parser.parse_frequencies(path=ch2o_path_terachem, software='TeraChem')
        ch2o_terachem_output_freqs = parser.parse_frequencies(path=ch2o_path_terachem_output, software='TeraChem')
        ncc_terachem_output_freqs = parser.parse_frequencies(path=ncc_path_terachem_output, software='TeraChem')
        orca_freqs = parser.parse_frequencies(path=orca_path, software='Orca')
        dual_freqs = parser.parse_frequencies(path=dual_freq_path, software='Gaussian')
        co2_xtb_freqs = parser.parse_frequencies(path=co2_xtb_freqs_path)
        ts_xtb_freqs = parser.parse_frequencies(path=ts_xtb_freqs_path)
        yml_freqs = parser.parse_frequencies(path=yml_freqs_path)
        vibspectrum_freqs = parser.parse_frequencies(path=vibspectrum_path, software='xTB')
        mock_freqs = parser.parse_frequencies(path=mock_path, software='Mockter')

        np.testing.assert_almost_equal(no3_freqs,
                                       np.array([-390.08, -389.96, 822.75, 1113.23, 1115.24, 1195.35], np.float64))
        np.testing.assert_almost_equal(c2h6_freqs,
                                       np.array([352.37, 847.01, 861.68, 1023.23, 1232.66, 1235.04, 1425.48, 1455.31,
                                                 1513.67, 1518.02, 1526.18, 1526.56, 3049.78, 3053.32, 3111.61, 3114.2,
                                                 3134.14, 3136.8], np.float64))
        np.testing.assert_almost_equal(so2oo_freqs,
                                       np.array([302.51, 468.1488, 469.024, 484.198, 641.0067, 658.6316,
                                                 902.2888, 1236.9268, 1419.0826], np.float64))
        np.testing.assert_almost_equal(ch2o_molpro_freqs,
                                       np.array([1181.01, 1261.34, 1529.25, 1764.47, 2932.15, 3000.10], np.float64))
        np.testing.assert_almost_equal(ch2o_terachem_freqs,
                                       np.array([1198.228, 1271.913, 1562.435, 1900.334, 2918.771, 2966.569],
                                                np.float64))
        np.testing.assert_almost_equal(ch2o_terachem_output_freqs,
                                       np.array([1198.63520807, 1276.19910582, 1563.62759321, 1893.24407646,
                                                 2916.39175334, 2965.86839559], np.float64))
        np.testing.assert_almost_equal(ncc_terachem_output_freqs,
                                       np.array([170.56668709, 278.52007409, 406.49102131, 765.91960508, 861.6118189,
                                                 910.16404036, 1010.63529045, 1052.86795614, 1160.15911873,
                                                 1275.00946008, 1386.75755192, 1406.08828477, 1425.90872097,
                                                 1506.47789418, 1522.65901736, 1527.41841768, 1710.89393731,
                                                 3020.79869151, 3035.66348773, 3061.21808688, 3085.3062489,
                                                 3087.60678739, 3447.41720077, 3529.23879182], np.float64))
        np.testing.assert_almost_equal(orca_freqs,
                                       np.array([1151.03, 1250.19, 1526.12, 1846.4, 3010.49, 3070.82], np.float64))
        np.testing.assert_almost_equal(dual_freqs,
                                       np.array([-1617.8276, 56.9527, 76.681, 121.4038, 182.1572, 194.9796,
                                                 202.4056, 209.9621, 273.506, 342.468, 431.985, 464.0768,
                                                 577.758, 594.4119, 615.5216, 764.1286, 962.2969, 968.0013,
                                                 1004.7852, 1098.0136, 1129.3888, 1137.0454, 1150.7824, 1185.4531,
                                                 1249.0746, 1387.4803, 1401.5073, 1413.8079, 1420.6471, 1453.6296,
                                                 1481.9425, 1487.0125, 1496.0713, 1498.382, 1507.7379, 2280.9881,
                                                 3015.0638, 3018.8871, 3030.1281, 3074.8208, 3079.5256, 3103.8434,
                                                 3109.1728, 3156.4352, 3783.7315], np.float64))
        np.testing.assert_almost_equal(co2_xtb_freqs,
                                       np.array([600.7, 600.7, 1424.29, 2592.18], np.float64))
        np.testing.assert_almost_equal(ts_xtb_freqs,
                                       np.array([-781.89, 139.33, 236.79, 327.73, 471.51, 690.72, 827.09, 915.23,
                                                 1056.62, 1185.04, 1315.85, 1347.33, 1424.36, 1497.04, 3181.14,
                                                 3367.54, 3433.32, 3467.59], np.float64))
        np.testing.assert_almost_equal(yml_freqs,
                                       np.array([2.4532480913713e-05, 204.48157765807244, 410.8720268963782,
                                                 811.6875778091901, 930.1054760588398, 1063.4363767201394,
                                                 1125.4142473593588, 1205.9470708729282, 1275.6731702161098,
                                                 1300.2253451543897, 1422.443746117271, 1472.318099536584,
                                                 1515.083361599387, 1528.8135856445965, 1553.9744435034322,
                                                 3085.8337759988153, 3128.1504569691283, 3135.666024988286,
                                                 3230.936983192379, 3235.2332367908975, 3922.9230982968807],
                                                np.float64))
        np.testing.assert_almost_equal(vibspectrum_freqs, np.array([4225.72], np.float64))

        np.testing.assert_almost_equal(mock_freqs, np.array([-500.,  520.,  540.], np.float64))

    def test_parse_normal_mode_displacement(self):
        """Test parsing frequencies and normal mode displacements"""
        freq_path = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'Gaussian_neg_freq.out')
        freqs, normal_modes_disp = parser.parse_normal_mode_displacement(path=freq_path)
        expected_freqs = np.array([-18.0696, 127.6948, 174.9499, 207.585, 228.8421, 281.2939, 292.4101,
                                   308.0345, 375.4493, 486.8396, 498.6986, 537.6196, 564.0223, 615.3762,
                                   741.8843, 749.3428, 777.1524, 855.3031, 871.055, 962.7075, 977.6181,
                                   1050.3147, 1051.8134, 1071.7234, 1082.0731, 1146.8729, 1170.0212, 1179.6722,
                                   1189.1581, 1206.0905, 1269.8371, 1313.4043, 1355.1081, 1380.155, 1429.7095,
                                   1464.5357, 1475.5996, 1493.6501, 1494.3533, 1500.9964, 1507.5851, 1532.7927,
                                   1587.7095, 1643.0702, 2992.7203, 3045.3662, 3068.6577, 3123.9646, 3158.2579,
                                   3159.3532, 3199.1684, 3211.4927, 3223.942, 3233.9201], np.float64)
        np.testing.assert_almost_equal(freqs, expected_freqs)
        expected_normal_modes_disp_0 = np.array(
            [[-0.0, 0.0, -0.09], [-0.0, 0.0, -0.1], [0.0, 0.0, -0.01], [0.0, 0.0, -0.07], [0.0, 0.0, -0.2],
             [-0.0, -0.0, 0.28], [0.0, -0.0, -0.08], [0.0, -0.0, 0.01], [0.0, -0.0, 0.12], [0.0, -0.0, 0.12],
             [0.08, -0.02, -0.04], [-0.08, 0.02, -0.04], [-0.0, 0.0, -0.18], [-0.3, -0.03, 0.41], [-0.0, -0.0, 0.4],
             [0.3, 0.03, 0.41], [0.0, 0.0, -0.15], [0.0, -0.0, 0.01], [0.0, -0.0, 0.21], [0.0, -0.0, 0.19]], np.float64)
        np.testing.assert_almost_equal(normal_modes_disp[0], expected_normal_modes_disp_0)

        freq_path = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'CHO_neg_freq.out')
        freqs, normal_modes_disp = parser.parse_normal_mode_displacement(path=freq_path)
        expected_freqs = np.array([-1612.8294, 840.8655, 1883.4822, 3498.091], np.float64)
        np.testing.assert_almost_equal(freqs, expected_freqs)
        expected_normal_modes_disp_1 = np.array(
            [[[0.05, 0.03, -0.0], [-0.13, -0.09, 0.0], [0.8, 0.57, -0.0]],
             [[-0.03, 0.05, -0.0], [0.09, -0.13, 0.0], [-0.57, 0.8, 0.0]],
             [[0.0, -0.0, -0.41], [-0.0, 0.0, 0.49], [0.0, -0.0, 0.77]],
             [[0.0, -0.0, 0.02], [-0.0, 0.0, -0.11], [0.0, -0.0, 0.99]]], np.float64)
        np.testing.assert_almost_equal(normal_modes_disp, expected_normal_modes_disp_1)

        freq_path = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'CH3OO_freq_gaussian.out')
        freqs, normal_modes_disp = parser.parse_normal_mode_displacement(path=freq_path)
        expected_freqs = np.array([136.4446, 494.1267, 915.7812, 1131.4603, 1159.9315, 1225.148, 1446.5652,
                                   1474.8065, 1485.6423, 3046.2186, 3134.8026, 3147.5619], np.float64)
        np.testing.assert_almost_equal(freqs, expected_freqs)
        expected_normal_modes_disp_2 = np.array(
            [[[-0.0, -0.0, 0.02], [0.0, 0.0, -0.11], [0.0, -0.0, 0.09],
              [0.32, 0.39, -0.23], [0.0, 0.0, 0.61], [-0.32, -0.39, -0.23]],
             [[-0.24, -0.06, -0.0], [-0.02, 0.22, 0.0], [0.26, -0.13, -0.0],
              [-0.54, -0.11, 0.01], [0.12, -0.43, -0.0], [-0.54, -0.11, -0.01]],
             [[0.41, -0.16, -0.0], [-0.25, 0.25, 0.0], [-0.1, -0.09, 0.0],
              [-0.03, -0.16, -0.0], [0.66, -0.42, -0.0], [-0.03, -0.16, 0.0]],
             [[0.0, 0.0, 0.13], [0.0, 0.0, -0.06], [-0.0, -0.0, -0.0],
              [-0.52, 0.38, -0.16], [-0.0, 0.0, -0.29], [0.52, -0.38, -0.16]],
             [[-0.01, 0.17, -0.0], [0.21, 0.02, 0.0], [-0.2, -0.12, 0.0],
              [-0.35, -0.01, 0.1], [0.62, -0.46, 0.0], [-0.35, -0.01, -0.1]],
             [[-0.07, -0.14, -0.0], [0.13, 0.2, 0.0], [-0.11, -0.12, -0.0],
              [0.52, 0.06, -0.1], [-0.48, 0.3, 0.0], [0.52, 0.06, 0.1]],
             [[-0.09, 0.05, 0.0], [-0.04, -0.0, 0.0], [0.02, 0.02, -0.0],
              [0.52, -0.16, 0.17], [0.39, -0.44, -0.0], [0.52, -0.16, -0.17]],
             [[-0.0, 0.0, -0.06], [-0.0, 0.0, -0.01], [0.0, -0.0, 0.0],
              [-0.44, -0.19, 0.07], [0.0, 0.0, 0.73], [0.44, 0.19, 0.07]],
             [[-0.03, -0.05, -0.0], [-0.01, -0.02, -0.0], [0.0, 0.01, 0.0],
              [0.06, 0.53, -0.37], [0.24, -0.29, 0.0], [0.06, 0.53, 0.37]],
             [[-0.03, 0.02, -0.0], [-0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
              [0.02, -0.33, -0.5], [0.37, 0.37, -0.0], [0.02, -0.33, 0.5]],
             [[0.0, 0.0, -0.1], [0.0, 0.0, 0.0], [-0.0, -0.0, 0.0],
              [-0.02, 0.4, 0.58], [-0.0, -0.0, -0.02], [0.02, -0.4, 0.58]],
             [[-0.05, -0.08, -0.0], [-0.0, -0.0, -0.0], [0.0, 0.0, 0.0],
              [-0.02, 0.19, 0.31], [0.61, 0.6, -0.0], [-0.02, 0.19, -0.31]]], np.float64)
        np.testing.assert_almost_equal(normal_modes_disp, expected_normal_modes_disp_2)

        freq_path = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'TS_NH2+N2H3.out')
        freqs, normal_modes_disp = parser.parse_normal_mode_displacement(path=freq_path)
        expected_freqs = np.array([-1745.4843, 64.9973, 168.1583, 234.1226, 453.2505, 657.1672, 737.7965, 844.5179,
                                   1156.12, 1177.1321, 1390.4004, 1454.281, 1565.3214, 1680.0987, 3367.2838, 3512.739,
                                   3550.219, 3652.1575], np.float64)
        np.testing.assert_almost_equal(freqs, expected_freqs)
        expected_normal_modes_disp_2 = np.array(
            [[[-0.03, -0.02, -0.0], [0.04, -0.08, 0.05], [0.92, -0.34, 0.08], [0.0, 0.04, 0.0],
              [0.0, -0.05, 0.01], [0.09, -0.09, -0.04], [-0.05, 0.02, -0.01], [0.0, 0.03, -0.01]],
             [[0.0, -0.01, -0.05], [0.01, 0.05, -0.08], [0.02, 0.0, 0.03], [-0.01, 0.0, 0.03],
              [0.03, 0.04, 0.07], [-0.06, -0.02, 0.07], [0.03, 0.05, 0.06], [-0.28, -0.65, -0.68]],
             [[-0.05, 0.18, -0.0], [-0.15, 0.14, 0.02], [-0.0, 0.32, 0.01], [0.29, -0.07, 0.02],
              [0.19, -0.14, -0.06], [0.4, -0.29, -0.03], [-0.23, -0.1, -0.01], [-0.59, -0.07, -0.06]],
             [[-0.01, 0.01, 0.03], [-0.06, -0.02, 0.04], [-0.03, 0.01, 0.07], [-0.02, -0.01, 0.04],
              [-0.55, 0.43, -0.33], [0.46, -0.3, -0.24], [0.03, -0.0, -0.03], [0.09, -0.07, -0.09]],
             [[0.09, -0.04, -0.01], [0.55, -0.39, 0.18], [-0.04, -0.05, -0.42], [-0.0, -0.03, 0.01],
              [-0.14, 0.29, -0.07], [-0.04, 0.15, 0.01], [-0.08, 0.06, 0.02], [-0.4, 0.15, 0.03]],
             [[0.19, -0.12, 0.04], [-0.33, 0.48, -0.28], [-0.04, -0.03, 0.3], [0.03, -0.01, -0.01],
              [-0.08, 0.22, -0.07], [-0.04, 0.28, -0.01], [-0.22, 0.07, -0.01], [0.47, -0.14, -0.06]],
             [[-0.06, 0.1, -0.06], [0.29, -0.31, 0.17], [-0.06, -0.16, -0.0], [0.04, -0.06, 0.0],
              [0.01, 0.16, 0.0], [-0.14, 0.12, 0.09], [-0.05, -0.01, 0.04], [0.77, -0.28, -0.04]],
             [[0.03, -0.05, 0.0], [0.26, -0.09, 0.02], [-0.29, -0.08, -0.2], [-0.02, 0.13, 0.0],
              [0.17, -0.45, 0.08], [0.36, -0.56, -0.14], [-0.06, 0.02, 0.02], [0.25, -0.11, -0.03]],
             [[0.26, 0.24, -0.05], [-0.09, -0.12, 0.16], [0.2, 0.09, 0.36], [-0.23, -0.18, 0.01],
              [0.01, -0.42, 0.16], [-0.07, -0.59, -0.02], [-0.02, 0.01, -0.02], [-0.08, 0.06, 0.01]],
             [[-0.01, 0.04, 0.01], [0.33, 0.45, -0.2], [0.02, -0.2, -0.23], [-0.02, -0.05, 0.04],
              [-0.26, -0.47, -0.14], [0.21, 0.4, -0.16], [0.01, -0.0, 0.01], [0.02, -0.02, -0.01]],
             [[-0.01, 0.02, 0.08], [0.02, 0.38, -0.11], [0.24, 0.23, -0.68], [-0.03, -0.04, -0.06],
              [0.12, 0.27, 0.06], [-0.17, -0.32, 0.06], [0.01, -0.02, 0.02], [0.18, -0.08, -0.02]],
             [[0.01, -0.03, -0.05], [-0.19, -0.27, 0.06], [0.25, 0.76, -0.21], [-0.02, 0.02, 0.05],
              [-0.11, -0.22, -0.02], [0.16, 0.27, -0.08], [-0.01, -0.02, 0.01], [0.18, -0.06, -0.0]],
             [[-0.04, -0.03, 0.01], [0.66, 0.21, -0.1], [0.09, 0.45, 0.5], [-0.02, -0.01, -0.03],
              [0.1, 0.12, 0.06], [0.01, -0.09, -0.04], [-0.0, -0.0, -0.01], [0.01, 0.02, 0.02]],
             [[-0.01, 0.01, 0.0], [-0.08, -0.02, 0.01], [-0.04, -0.11, -0.05], [-0.05, -0.04, 0.0],
              [0.51, 0.3, 0.38], [0.51, 0.27, -0.38], [0.0, 0.0, 0.0], [-0.01, -0.0, -0.0]],
             [[-0.0, 0.0, 0.0], [-0.0, -0.0, -0.01], [0.02, 0.01, -0.01], [-0.0, -0.0, -0.0],
              [-0.0, -0.0, 0.0], [0.0, -0.0, 0.0], [0.01, 0.05, -0.05], [-0.16, -0.68, 0.71]],
             [[0.0, -0.03, -0.06], [-0.02, 0.45, 0.88], [-0.02, 0.01, -0.02], [-0.01, 0.0, 0.01],
              [0.08, 0.01, -0.12], [0.0, 0.0, 0.01], [0.0, 0.0, -0.0], [-0.0, -0.0, 0.01]],
             [[0.0, -0.0, -0.01], [-0.0, 0.04, 0.09], [-0.02, 0.01, -0.0], [0.05, 0.01, 0.01],
              [-0.35, -0.03, 0.54], [-0.39, -0.08, -0.65], [0.0, -0.0, 0.0], [0.0, -0.0, 0.0]],
             [[0.0, 0.0, 0.01], [0.0, -0.05, -0.09], [0.01, -0.0, 0.0], [-0.01, 0.0, 0.08],
              [0.43, 0.01, -0.61], [-0.36, -0.06, -0.54], [-0.0, 0.0, -0.0], [-0.0, -0.0, 0.0]]], np.float64)
        np.testing.assert_almost_equal(normal_modes_disp, expected_normal_modes_disp_2)

        path = os.path.join(ARC_PATH, 'arc', 'testing', 'normal_mode', 'HO2', 'output.out')
        freqs, normal_modes_disp = parser.parse_normal_mode_displacement(path=path)
        expected_freqs = np.array([1224.9751, 1355.2709, 3158.763], np.float64)
        np.testing.assert_almost_equal(freqs, expected_freqs)
        expected_normal_modes_disp_3 = np.array(
            [[[0.57, -0.41, -0.], [-0.51, 0.36, 0.], [-0.25, 0.22, 0.]],
             [[-0.22, - 0.11, 0.], [0.39, -0.03, -0.], [-0.68, 0.57, 0.]],
             [[0.15, 0.19, -0.], [0.02, -0.01, -0.], [-0.66, -0.71, 0.]]], np.float64)
        np.testing.assert_almost_equal(normal_modes_disp, expected_normal_modes_disp_3)

        path = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'output.yml')
        freqs, normal_modes_disp = parser.parse_normal_mode_displacement(path=path)
        self.assertEqual(freqs[-1], 3922.9230982968807)
        expected_normal_modes_disp_4_0 = np.array(
            [[0.008599578508578239, 0.01787645439208711, -0.04175706756233052],
             [-0.008909310426849138, -0.018556163013860285, 0.04330724268285693],
             [0.011241853539959802, 0.023478488708442366, -0.054747714680216726],
             [-0.05238385033814685, -0.019847498073382798, -0.15577070366458387],
             [-0.009756669586238532, -0.020314852256014822, 0.04741197775795493],
             [0.09823056279991693, 0.11520121377966115, -0.06694055318283264],
             [-0.1273765184024777, -0.09250300723936498, 0.07996788342871741],
             [0.07839541036186716, -0.009452306143447562, 0.15793005333991175],
             [-0.16184923713199378, -0.3376354950974596, 0.787886990928027]], np.float64)
        np.testing.assert_almost_equal(normal_modes_disp[0], expected_normal_modes_disp_4_0)

    def test_parse_xyz_from_file(self):
        """Test parsing xyz from a file"""
        path1 = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'CH3C(O)O.gjf')
        path2 = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'CH3C(O)O.xyz')
        path3 = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'AIBN.gjf')
        path4 = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'molpro.in')
        path5 = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'qchem.in')
        path6 = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'qchem_output.out')
        path7 = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'TS.gjf')
        path8 = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'formaldehyde_coords.xyz')
        path9 = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'optim_traj_terachem.xyz')  # test trajectories
        path10 = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'ethane_minimize_terachem_output.out')
        path11 = os.path.join(ARC_PATH, 'arc', 'testing', 'orca_example_opt.log')
        path12 = os.path.join(ARC_PATH, 'arc', 'testing', 'tani_output.yml')

        xyz1 = parser.parse_xyz_from_file(path1)
        xyz2 = parser.parse_xyz_from_file(path2)
        xyz3 = parser.parse_xyz_from_file(path3)
        xyz4 = parser.parse_xyz_from_file(path4)
        xyz5 = parser.parse_xyz_from_file(path5)
        xyz6 = parser.parse_xyz_from_file(path6)
        xyz7 = parser.parse_xyz_from_file(path7)
        xyz8 = parser.parse_xyz_from_file(path8)
        xyz9 = parser.parse_xyz_from_file(path9)
        xyz10 = parser.parse_xyz_from_file(path10)
        xyz11 = parser.parse_xyz_from_file(path11)
        xyz12 = parser.parse_xyz_from_file(path12)

        self.assertEqual(xyz1, xyz2)
        xyz1_str = xyz_to_str(xyz1)
        xyz2_str = xyz_to_str(xyz2)
        xyz3_str = xyz_to_str(xyz3)
        xyz4_str = xyz_to_str(xyz4)
        xyz5_str = xyz_to_str(xyz5)
        xyz6_str = xyz_to_str(xyz6)
        xyz9_str = xyz_to_str(xyz9)
        xyz11_str = xyz_to_str(xyz11)

        self.assertTrue('C       1.40511900    0.21728200    0.07675200' in xyz1_str)
        self.assertTrue('O      -0.79314200    1.04818800    0.18134200' in xyz1_str)
        self.assertTrue('H      -0.43701200   -1.34990600    0.92900600' in xyz2_str)
        self.assertTrue('C       2.12217963   -0.66843078    1.04808732' in xyz3_str)
        self.assertTrue('N       2.41731872   -1.07916417    2.08039935' in xyz3_str)
        spc3 = ARCSpecies(label='AIBN', xyz=xyz3)
        self.assertEqual(len(spc3.mol.atoms), 24)
        self.assertTrue('S      -0.42046822   -0.39099498    0.02453521' in xyz4_str)
        self.assertTrue('N      -1.99742564    0.38106573    0.09139807' in xyz5_str)
        self.assertTrue('N      -1.17538406    0.34366165    0.03265021' in xyz6_str)
        self.assertEqual(len(xyz7['symbols']), 34)
        self.assertEqual(len(xyz8['symbols']), 4)
        expected_xyz_9 = """N      -0.67665958    0.74524340   -0.41319355
H      -1.26179357    1.52577220   -0.13687665
H       0.28392722    1.06723640   -0.44163375
N      -0.75345799   -0.33268278    0.51180786
H      -0.97153041   -0.02416219    1.45398654
H      -1.48669570   -0.95874053    0.20627423
N       2.28178508   -0.42455356    0.14404399
H       1.32677989   -0.80557411    0.33156013"""
        self.assertEqual(xyz9_str, expected_xyz_9)
        self.assertIsNone(xyz10)
        expected_xyz_11 = """C       0.00917900   -0.00000000   -0.00000000
O       1.20814900   -0.00000000    0.00000000
H      -0.59436200    0.94730400    0.00000000
H      -0.59436200   -0.94730400    0.00000000"""
        self.assertEqual(xyz11_str, expected_xyz_11)
        expected_xyz_12 = """
C       0.76543810    1.12187162    0.30492610
C       1.35782656   -0.27242561    0.13987256
O       0.40260198   -1.25859876    0.48175081
C      -0.76543825   -1.12187192   -0.30492599
C      -1.35782634    0.27242561   -0.13987266
O      -0.40260197    1.25859858   -0.48175076
H       1.46909034    1.88883246   -0.03480069
H       0.53541546    1.29972688    1.36777761
H       1.69381294   -0.40788846   -0.90078084
H       2.21458405   -0.41511654    0.80648738
H      -1.46909026   -1.88883253    0.03480063
H      -0.53541537   -1.29972706   -1.36777773
H      -2.21458420    0.41511639   -0.80648746
H      -1.69381305    0.40788834    0.90078104"""
        self.assertTrue(almost_equal_coords(xyz12, str_to_xyz(expected_xyz_12)))

    def test_parse_geometry(self):
        """Test parse_geometry()"""
        # Test parsing xyz from a Gaussina file with more than 50 atoms where the iop(2/9=2000) keyword is not specified
        path_1 = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'Gaussian_large.log')
        xyz_1 = parser.parse_geometry(path=path_1)
        self.assertIsInstance(xyz_1, dict)
        self.assertEqual(len(xyz_1['symbols']), 53)

        path_2 = os.path.join(ARC_PATH, 'arc', 'testing', 'opt', 'xtb_opt_1.out')  # Turbomol format
        xyz_2 = parser.parse_geometry(path=path_2)
        expected_xyz_2 = {'symbols': ('C', 'C', 'C', 'O', 'H', 'H', 'H', 'H'),
                          'isotopes': (12, 12, 12, 16, 1, 1, 1, 1),
                          'coords': ((-2.1908361683609, -0.0875055545944093, -0.712358508847116),
                                     (-0.170663821576948, -0.721009080780027, 0.6339514359292),
                                     (2.27150340995404, 0.539200343733434, 0.305747355748416),
                                     (4.16185183882216, 0.0456870795397582, 1.46958543963615),
                                     (-2.10906550336131, 1.39281538085596, -2.11159862590894),
                                     (-3.98822887666127, -1.01150815322938, -0.474092464462499),
                                     (-0.230883305078883, -2.20128348308961, 2.04098984698359),
                                     (2.25632242626311, 2.04360346756428, -1.15222446018155))}
        self.assertTrue(almost_equal_coords(xyz_2, expected_xyz_2))

        path_3 = os.path.join(ARC_PATH, 'arc', 'testing', 'opt', 'xtb_opt_2.out')  # SDF format
        xyz_3 = parser.parse_geometry(path=path_3)
        expected_xyz_3 = {'symbols': ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                          'isotopes': (12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                          'coords': ((1.2769, -0.0581, -0.1002), (-0.1342, -0.602, 0.0867), (-1.1766, 0.5082, 0.0354),
                                     (1.3672, 0.442, -1.0623), (2.008, -0.8624, -0.0611), (1.514, 0.6603, 0.6818),
                                     (-0.1997, -1.1163, 1.0482), (-0.3465, -1.3346, -0.6954),
                                     (-1.1394, 1.0205, -0.9237), (-0.9926, 1.2388, 0.8204), (-2.1772, 0.1036, 0.1703))}
        self.assertTrue(almost_equal_coords(xyz_3, expected_xyz_3))

        path_3 = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'output.yml')
        xyz_3 = parser.parse_geometry(path=path_3)
        expected_xyz_3 = {'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
                          'isotopes': (12, 12, 16, 1, 1, 1, 1, 1, 1),
                          'coords': ((-0.946290349145905, 0.26780700385778744, 0.09754742423589319),
                                     (0.42975981142499586, -0.353915094763013, 0.11437888547377144),
                                     (0.30820117912595224, -1.6514516474848426, -0.4667184698053586),
                                     (-1.6438832997484905, -0.34260791258158996, 0.6728214694472372),
                                     (-0.9219930220681609, 1.2701448632614325, 0.5318298475513369),
                                     (-1.3149024252390442, 0.34248035073781935, -0.9265204691350346),
                                     (0.8039042577310339, -0.4257131823561406, 1.140962515296026),
                                     (1.1332670535632363, 0.2595738062901037, -0.45809194757240435),
                                     (1.1691669229762556, -2.0726946137332924, -0.4703870247902347))}
        self.assertTrue(almost_equal_coords(xyz_3, expected_xyz_3))

        path_4 = os.path.join(ARC_PATH, 'arc', 'testing', 'mockter.yml')
        xyz_4 = parser.parse_geometry(path=path_4)
        expected_xyz_4 = {'symbols': ('O', 'H', 'H'), 'isotopes': (16, 1, 1),
                          'coords': ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))}
        self.assertTrue(almost_equal_coords(xyz_4, expected_xyz_4))

        path_5 = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'TS_NH3+H=NH2+H2.out')
        xyz_5 = parser.parse_geometry(path=path_5)
        expected_xyz_5 = {'symbols': ('N', 'H', 'H', 'H', 'H'), 'isotopes': (14, 1, 1, 1, 1),
                          'coords': ((-0.0, 0.317177, 0.0), (-0.624513, 0.203027, 0.807493),
                                     (-0.624513, 0.203027, -0.807493), (0.59, -0.871703, -0.0),
                                     (0.659027, -1.754591, -0.0))}
        self.assertTrue(almost_equal_coords(xyz_5, expected_xyz_5))

    def test_parse_trajectory(self):
        """Test parsing trajectories"""
        path = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'scan_optim.xyz')
        trajectory = parser.parse_trajectory(path)
        self.assertEqual(len(trajectory), 46)
        self.assertIsInstance(trajectory[0], dict)
        self.assertEqual(len(trajectory[0]['symbols']), 9)

        path = os.path.join(ARC_PATH, 'arc', 'testing', 'irc', 'cyano_irc_1.out')
        trajectory = parser.parse_trajectory(path)
        self.assertEqual(len(trajectory), 58)
        self.assertIsInstance(trajectory[0], dict)
        self.assertEqual(len(trajectory[0]['symbols']), 16)

        path = os.path.join(ARC_PATH, 'arc', 'testing', 'irc', 'irc_failed.out')
        trajectory = parser.parse_trajectory(path)
        self.assertEqual(len(trajectory), 21)
        self.assertIsInstance(trajectory[0], dict)
        self.assertEqual(len(trajectory[0]['symbols']), 17)

        path = os.path.join(ARC_PATH, 'arc', 'testing', 'stringfile.xyz0000')
        trajectory = parser.parse_trajectory(path)
        self.assertEqual(len(trajectory), 9)
        self.assertIsInstance(trajectory[0], dict)
        self.assertEqual(len(trajectory[0]['symbols']), 3)

    def test_parse_1d_scan_coords(self):
        """Test parsing the optimized coordinates of a torsion scan at each optimization point"""
        path_1 = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'H2O2.out')
        traj_1 = parser.parse_1d_scan_coords(path_1)
        self.assertEqual(len(traj_1), 37)
        self.assertEqual(traj_1[10], {'coords': ((-0.715582, -0.140909, 0.383809),
                                                 (0.715582, 0.140909, 0.383809),
                                                 (-1.043959, 0.678384, -0.010288),
                                                 (1.043959, -0.678384, -0.010288)),
                                      'isotopes': (16, 16, 1, 1),
                                      'symbols': ('O', 'O', 'H', 'H')})

        # Test that the function doesn't crush on an incomplete scan (errored during opt):
        path_2 = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'TS_errored.out')
        traj_2 = parser.parse_1d_scan_coords(path_2)
        self.assertEqual(len(traj_2), 2)

        path_3 = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'xtb_1', 'output.out')
        traj_3 = parser.parse_1d_scan_coords(path_3)
        self.assertEqual(len(traj_3), 45)
        self.assertEqual(traj_3[0]['symbols'], ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'))

        path_4 = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'TS_scan.out')
        traj_4 = parser.parse_1d_scan_coords(path_4)
        self.assertEqual(len(traj_4), 8)  # output file trimmed
        self.assertEqual(traj_4[0]['symbols'], ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                                                'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                                                'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'))

    def test_parse_t1(self):
        """Test T1 diagnostic parsing"""
        path = os.path.join(ARC_PATH, 'arc', 'testing', 'sp', 'mehylamine_CCSD(T).out')
        t1 = parser.parse_t1(path)
        self.assertEqual(t1, 0.0086766)

        path = os.path.join(ARC_PATH, 'arc', 'testing', 'mockter.yml')
        t1 = parser.parse_t1(path)
        self.assertEqual(t1, 0.0002)

    def test_parse_e_elect(self):
        """Test parsing the electronic energy from a single-point job output file"""
        path = os.path.join(ARC_PATH, 'arc', 'testing', 'sp', 'mehylamine_CCSD(T).out')
        e_elect = parser.parse_e_elect(path)
        self.assertAlmostEqual(e_elect, -251377.49160993524)

        path = os.path.join(ARC_PATH, 'arc', 'testing', 'composite', 'SO2OO_CBS-QB3.log')
        e_elect = parser.parse_e_elect(path, zpe_scale_factor=0.99)
        self.assertAlmostEqual(e_elect, -1833127.0939478774)

        path = os.path.join(ARC_PATH, 'arc', 'testing', 'sp', 'formaldehyde_sp_terachem_output.out')
        e_elect = parser.parse_e_elect(path)
        self.assertAlmostEqual(e_elect, -300621.95378630824)

        path = os.path.join(ARC_PATH, 'arc', 'testing', 'sp', 'formaldehyde_sp_terachem_results.dat')
        e_elect = parser.parse_e_elect(path)
        self.assertAlmostEqual(e_elect, -300621.95378630824)

        path = os.path.join(ARC_PATH, 'arc', 'testing', 'sp', 'NCC_xTB.out')
        e_elect = parser.parse_e_elect(path, software='xtb')
        self.assertAlmostEqual(e_elect, -28229.880775867754)

        path = os.path.join(ARC_PATH, 'arc', 'testing', 'sp', 'NCC_xTB.out')
        e_elect = parser.parse_e_elect(path)  # not specifying software
        self.assertAlmostEqual(e_elect, -28229.880775867754)

        path = os.path.join(ARC_PATH, 'arc', 'testing', 'sp', 'output.yml')
        e_elect = parser.parse_e_elect(path)
        self.assertAlmostEqual(e_elect, -40692.56663699465)

        path = os.path.join(ARC_PATH, 'arc', 'testing', 'mockter.yml')
        e_elect = parser.parse_e_elect(path)
        self.assertAlmostEqual(e_elect, 50)

    def test_identify_ess(self):
        """Test the identify_ess() function."""
        ess = parser.identify_ess(os.path.join(ARC_PATH, 'arc', 'testing', 'sp', 'NCC_xTB.out'))
        self.assertEqual(ess, 'xtb')
        ess = parser.identify_ess(os.path.join(ARC_PATH, 'arc', 'testing', 'mockter.yml'))
        self.assertEqual(ess, 'mockter')
        ess = parser.identify_ess(os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'TS_NH3+H=NH2+H2.out'))
        self.assertEqual(ess, 'gaussian')

    def test_parse_zpe(self):
        """Test the parse_zpe() function for parsing zero point energies"""
        path1 = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'C2H6_freq_QChem.out')
        path2 = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'CH2O_freq_molpro.out')
        path3 = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'NO3_freq_QChem_fails_on_cclib.out')
        path4 = os.path.join(ARC_PATH, 'arc', 'testing', 'composite', 'SO2OO_CBS-QB3.log')
        path5 = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'TS_NH2+N2H3_xtb.out')
        zpe1, zpe2, zpe3, zpe4, zpe5 = parser.parse_zpe(path1), parser.parse_zpe(path2), parser.parse_zpe(path3), \
            parser.parse_zpe(path4), parser.parse_zpe(path5)
        self.assertAlmostEqual(zpe1, 198.08311200000, 5)
        self.assertAlmostEqual(zpe2, 69.793662734869, 5)
        self.assertAlmostEqual(zpe3, 25.401064000000, 5)
        self.assertAlmostEqual(zpe4, 39.368057626223, 5)
        self.assertAlmostEqual(zpe5, -29058.854452910982, 5)

    def test_parse_1d_scan_energies(self):
        """Test parsing a 1D scan output file"""
        path1 = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'sBuOH.out')
        energies, angles = parser.parse_1d_scan_energies(path=path1)
        expected_energies = np.array([1.57530564e-05, 3.98826556e-01, 1.60839959e+00, 3.49030801e+00,
                                      5.74358812e+00, 8.01124810e+00, 9.87649510e+00, 1.10079306e+01,
                                      1.11473788e+01, 1.02373175e+01, 8.49330826e+00, 6.23697731e+00,
                                      3.89294941e+00, 1.87096796e+00, 5.13009545e-01, 1.86410533e-04,
                                      4.16146979e-01, 1.66269755e+00, 3.59565619e+00, 5.90306099e+00,
                                      8.19668453e+00, 1.00329329e+01, 1.10759678e+01, 1.10923247e+01,
                                      1.00763770e+01, 8.28078980e+00, 6.04456755e+00, 3.77500671e+00,
                                      1.83344694e+00, 5.20014378e-01, 2.21067093e-03, 3.70723206e-01,
                                      1.56091218e+00, 3.44323279e+00, 5.73505787e+00, 8.04497265e+00,
                                      9.93330041e+00, 1.10426686e+01, 1.11168469e+01, 1.01271857e+01,
                                      8.32729265e+00, 6.06336876e+00, 3.76108631e+00, 1.80461632e+00,
                                      4.94715062e-01, 0.00000000e+00], np.float64)
        expected_angles = np.array([0., 8., 16., 24., 32., 40., 48., 56., 64., 72., 80., 88., 96., 104.,
                                    112., 120., 128., 136., 144., 152., 160., 168., 176., 184., 192., 200., 208., 216.,
                                    224., 232., 240., 248., 256., 264., 272., 280., 288., 296., 304., 312., 320., 328.,
                                    336., 344., 352., 360.], np.float64)
        np.testing.assert_almost_equal(energies, expected_energies)
        np.testing.assert_almost_equal(angles, expected_angles, 3)

        path2 = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'scan_1d_curvilinear_error.out')
        energies_2, angles_2 = parser.parse_1d_scan_energies(path=path2)
        self.assertEqual(energies_2, 0.)
        self.assertEqual(angles_2, 360.)

        path3 = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'xtb_1', 'output.out')
        energies_3, angles_3 = parser.parse_1d_scan_energies(path=path3)
        self.assertEqual(energies_3,
                         [0.0, 0.0, 0.8745195798946952, 1.9753081586022745, 3.4582153578448924, 5.221739195556438,
                          7.077549993919092, 8.399646521429531, 8.289930474034918, 6.795010313322564, 4.870396973175957,
                          3.0985765217774315, 1.6695973495516228, 0.6705322400630394, 0.11805330382048851,
                          0.01705542561467155, 0.3780853801545163, 1.1885756662632048, 2.435098248941358,
                          4.048450670245074, 5.945599850980216, 7.739232931569859, 8.563762981029868, 7.732003775239718,
                          5.937498125524144, 4.0434600886110275, 2.415860321434593, 1.1742907836305676,
                          0.37071732437834726, 0.018443422370182816, 0.12250622509236564, 0.6827384540738421,
                          1.6915252329381474, 3.101438046192925, 4.87996864773595, 6.803885765704763, 8.29145483137836,
                          8.397975306757871, 7.071194732583535, 5.2144584560301155, 3.4362616106591304,
                          1.9581994720974762, 0.8766537489609618, 0.22504858509637415, 0.0004629911454685498])
        self.assertEqual(angles_3,
                         [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0,
                          120.0, 128.0, 136.0, 144.0, 152.0, 160.0, 168.0, 176.0, 184.0, 192.0, 200.0, 208.0, 216.0,
                          224.0, 232.0, 240.0, 248.0, 256.0, 264.0, 272.0, 280.0, 288.0, 296.0, 304.0, 312.0, 320.0,
                          328.0, 336.0, 344.0, 352.0])

    def test_parse_nd_scan_energies(self):
        """Test parsing an ND scan output file"""
        path1 = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'scan_2D_relaxed_OCdOO.log')
        results = parser.parse_nd_scan_energies(path=path1, software='gaussian')
        self.assertEqual(results[0]['directed_scan_type'], 'ess_gaussian')
        self.assertEqual(results[0]['scans'], [(4, 1, 2, 5), (4, 1, 3, 6)])
        self.assertEqual(len(list(results[0].keys())), 3)
        self.assertEqual(len(list(results[0]['directed_scan'].keys())), 36 * 36 + 1)  # 1297
        self.assertAlmostEqual(results[0]['directed_scan']['170.00', '40.00']['energy'], 26.09747088)

    def test_parse_dipole_moment(self):
        """Test parsing the dipole moment from an opt job output file"""
        path1 = os.path.join(ARC_PATH, 'arc', 'testing', 'composite', 'SO2OO_CBS-QB3.log')
        dm1 = parser.parse_dipole_moment(path1)
        self.assertEqual(dm1, 0.63)

        path2 = os.path.join(ARC_PATH, 'arc', 'testing', 'N2H4_opt_QChem.out')
        dm2 = parser.parse_dipole_moment(path2)
        self.assertEqual(dm2, 2.0664)

        path3 = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'CH2O_freq_molpro.out')
        dm3 = parser.parse_dipole_moment(path3)
        self.assertAlmostEqual(dm3, 2.8840, 4)

        path4 = os.path.join(ARC_PATH, 'arc', 'testing', 'orca_example_opt.log')
        dm4 = parser.parse_dipole_moment(path4)
        self.assertEqual(dm4, 2.11328)

        path5 = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'ethane_minimize_terachem_output.out')
        dm5 = parser.parse_dipole_moment(path5)
        self.assertAlmostEqual(dm5, 0.000179036, 4)

    def test_parse_polarizability(self):
        """Test parsing the polarizability moment from a freq job output file"""
        path1 = os.path.join(ARC_PATH, 'arc', 'testing', 'composite', 'SO2OO_CBS-QB3.log')
        polar1 = parser.parse_polarizability(path1)
        self.assertAlmostEqual(polar1, 3.99506, 4)

    def test_process_conformers_file(self):
        """Test processing ARC conformer files"""
        path1 = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'conformers_before_optimization.txt')
        path2 = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'conformers_after_optimization.txt')
        path3 = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'conformers_file.txt')

        xyzs, energies = parser.process_conformers_file(path1)
        self.assertEqual(len(xyzs), 3)
        self.assertEqual(len(energies), 3)
        self.assertTrue(all([e is None for e in energies]))

        spc1 = ARCSpecies(label='tst1', xyz=xyzs[0])
        self.assertEqual(len(spc1.conformers), 1)

        xyzs, energies = parser.process_conformers_file(path2)
        self.assertEqual(len(xyzs), 3)
        self.assertEqual(len(energies), 3)
        self.assertEqual(energies, [0.0, 10.271, 10.288])

        spc2 = ARCSpecies(label='tst2', xyz=xyzs[:2])
        self.assertEqual(len(spc2.conformers), 2)
        self.assertEqual(len(spc2.conformer_energies), 2)

        xyzs, energies = parser.process_conformers_file(path3)
        self.assertEqual(len(xyzs), 4)
        self.assertEqual(len(energies), 4)
        self.assertEqual(energies, [0.0, 0.005, None, 0.005])

        spc3 = ARCSpecies(label='tst3', xyz=xyzs)
        self.assertEqual(len(spc3.conformers), 4)
        self.assertEqual(len(spc3.conformer_energies), 4)

        spc4 = ARCSpecies(label='tst4', xyz=path1)
        self.assertEqual(len(spc4.conformers), 3)
        self.assertTrue(all([e is None for e in spc4.conformer_energies]))
        spc5 = ARCSpecies(label='tst5', xyz=path2)
        self.assertEqual(len(spc5.conformers), 3)
        self.assertTrue(all([e is not None for e in spc5.conformer_energies]))
        spc6 = ARCSpecies(label='tst6', xyz=path3)
        self.assertEqual(len(spc6.conformers), 4)

    def test_parse_str_blocks(self):
        """Test parsing str blocks"""
        path = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'H2O2.out')
        str_blks = parser.parse_str_blocks(
            path, 'Initial Parameters', '--------', regex=False, tail_count=3)
        desire_str_lists = [
            '                           !    Initial Parameters    !\n',
            '                           ! (Angstroms and Degrees)  !\n',
            ' --------------------------                            --------------------------\n',
            ' ! Name  Definition              Value          Derivative Info.                !\n',
            ' --------------------------------------------------------------------------------\n',
            ' ! R1    R(1,2)                  1.4252         calculate D2E/DX2 analytically  !\n',
            ' ! R2    R(1,3)                  0.9628         calculate D2E/DX2 analytically  !\n',
            ' ! R3    R(2,4)                  0.9628         calculate D2E/DX2 analytically  !\n',
            ' ! A1    A(2,1,3)              101.2687         calculate D2E/DX2 analytically  !\n',
            ' ! A2    A(1,2,4)              101.2687         calculate D2E/DX2 analytically  !\n',
            ' ! D1    D(3,1,2,4)            118.8736         Scan                            !\n',
            ' --------------------------------------------------------------------------------\n']
        self.assertEqual(len(str_blks), 1)
        self.assertEqual(str_blks[0], desire_str_lists)

    def test_parse_scan_args(self):
        """Test parsing scan arguments"""
        path = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'CH2OOH.out')
        scan_args = parser.parse_scan_args(path)
        self.assertEqual(scan_args['scan'], [4, 1, 2, 3])
        self.assertEqual(scan_args['freeze'], [[1, 2, 3, 6], [2, 3]])
        self.assertEqual(scan_args['step'], 90)
        self.assertEqual(scan_args['step_size'], 4.0)
        self.assertEqual(scan_args['n_atom'], 6)

    def test_parse_ic_info(self):
        """Test parsing internal coordinates information"""
        path = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'CH2OOH.out')
        ic_info = parser.parse_ic_info(path)
        expected_labels = ['R1', 'R2', 'R3', 'R4', 'R5', 'A1', 'A2',
                           'A3', 'A4', 'A5', 'D1', 'D2', 'D3', 'D4']
        expected_types = ['R', 'R', 'R', 'R', 'R', 'A',
                          'A', 'A', 'A', 'A', 'D', 'D', 'D', 'D']
        expected_atoms = [[1, 2], [1, 4], [1, 5], [2, 3], [3, 6], [2, 1, 4],
                          [2, 1, 5], [4, 1, 5], [1, 2, 3], [2, 3, 6], [4, 1, 2, 3],
                          [5, 1, 2, 3], [2, 1, 4, 5], [1, 2, 3, 6]]
        expected_redundant = [False] * 14
        expected_scan = [False, False, False, False, False, False, False,
                         False, False, False, True, True, False, False]
        self.assertEqual(expected_labels, ic_info.index.to_list())
        self.assertEqual(expected_types, ic_info.type.to_list())
        self.assertEqual(expected_atoms, ic_info.atoms.to_list())
        self.assertEqual(expected_redundant, ic_info.redundant.to_list())
        self.assertEqual(expected_scan, ic_info.scan.to_list())

    def test_parse_ic_values(self):
        """Test parsing internal coordinate values"""
        ic_blk = [
            ' ! R1    R(1,2)                  1.4535         -DE/DX =    0.0                 !\n',
            ' ! R2    R(1,3)                  0.9674         -DE/DX =    0.0                 !\n',
            ' ! R3    R(2,4)                  0.9674         -DE/DX =    0.0                 !\n',
            ' ! A1    A(2,1,3)              100.563          -DE/DX =    0.0                 !\n',
            ' ! A2    A(1,2,4)              100.563          -DE/DX =    0.0                 !\n',
            ' ! D1    D(3,1,2,4)            118.8736         -DE/DX =    0.0003              !\n']
        software = 'gaussian'
        ic_values = parser.parse_ic_values(ic_blk, software)
        expected_labels = ['R1', 'R2', 'R3', 'A1', 'A2', 'D1']
        expected_values = [1.4535, 0.9674, 0.9674, 100.563, 100.563, 118.8736]
        self.assertEqual(expected_labels, ic_values.index.to_list())
        self.assertEqual(expected_values, ic_values.value.to_list())

    def test_parse_conformers(self):
        """Test parsing internal coordinates of all intermediate conformers in a scan job"""
        path = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'H2O2.out')
        scan_conformers = parser.parse_scan_conformers(path)
        expected_labels = ['R1', 'R2', 'R3', 'A1', 'A2', 'D1']
        expected_types = ['R', 'R', 'R', 'A', 'A', 'D']
        expected_atoms = [[1, 2], [1, 3], [2, 4], [2, 1, 3], [1, 2, 4], [3, 1, 2, 4]]
        expected_redundant = [False] * 6
        expected_scan = [False] * 5 + [True]
        expected_conf_0 = [1.4535, 0.9674, 0.9674, 100.563, 100.563, 118.8736]
        expected_conf_18 = [1.4512, 0.9688, 0.9688, 103.2599, 103.2599, -61.1264]
        expected_conf_36 = [1.4536, 0.9673, 0.9673, 100.5586, 100.5586, 118.8736]
        self.assertEqual(expected_labels, scan_conformers.index.to_list())
        self.assertEqual(expected_types, scan_conformers.type.to_list())
        self.assertEqual(expected_atoms, scan_conformers.atoms.to_list())
        self.assertEqual(expected_redundant, scan_conformers.redundant.to_list())
        self.assertEqual(expected_scan, scan_conformers.scan.to_list())
        self.assertEqual((6, 41), scan_conformers.shape)
        self.assertEqual(expected_conf_0, scan_conformers[0].to_list())
        self.assertEqual(expected_conf_18, scan_conformers[18].to_list())
        self.assertEqual(expected_conf_36, scan_conformers[36].to_list())


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
