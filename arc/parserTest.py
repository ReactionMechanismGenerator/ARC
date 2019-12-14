#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the parser functions
"""

import numpy as np
import os
import unittest

import arc.parser as parser
from arc.settings import arc_path
from arc.species import ARCSpecies
from arc.species.converter import xyz_to_str


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
        no3_path = os.path.join(arc_path, 'arc', 'testing', 'NO3_freq_QChem_fails_on_cclib.out')
        c2h6_path = os.path.join(arc_path, 'arc', 'testing', 'C2H6_freq_QChem.out')
        so2oo_path = os.path.join(arc_path, 'arc', 'testing', 'SO2OO_CBS-QB3.log')
        ch2o_path = os.path.join(arc_path, 'arc', 'testing', 'CH2O_freq_molpro.out')
        no3_freqs = parser.parse_frequencies(path=no3_path, software='QChem')
        c2h6_freqs = parser.parse_frequencies(path=c2h6_path, software='QChem')
        so2oo_freqs = parser.parse_frequencies(path=so2oo_path, software='Gaussian')
        ch2o_freqs = parser.parse_frequencies(path=ch2o_path, software='Molpro')
        self.assertTrue(np.array_equal(no3_freqs,
                                       np.array([-390.08, -389.96, 822.75, 1113.23, 1115.24, 1195.35], np.float64)))
        self.assertTrue(np.array_equal(c2h6_freqs,
                                       np.array([352.37, 847.01, 861.68, 1023.23, 1232.66, 1235.04, 1425.48, 1455.31,
                                                 1513.67, 1518.02, 1526.18, 1526.56, 3049.78, 3053.32, 3111.61, 3114.2,
                                                 3134.14, 3136.8], np.float64)))
        self.assertTrue(np.array_equal(so2oo_freqs,
                                       np.array([302.51, 468.1488, 469.024, 484.198, 641.0067, 658.6316,
                                                 902.2888, 1236.9268, 1419.0826], np.float64)))
        self.assertTrue(np.array_equal(ch2o_freqs,
                                       np.array([1181.01, 1261.34, 1529.25, 1764.47, 2932.15, 3000.10], np.float64)))

    def test_parse_normal_displacement_modes(self):
        """Test parsing frequencies and normal displacement modes"""
        gaussian_neg_freq_path = os.path.join(arc_path, 'arc', 'testing', 'Gaussian_neg_freq.out')
        freqs, normal_disp_modes = parser.parse_normal_displacement_modes(path=gaussian_neg_freq_path)
        expected_freqs = np.array([-18.0696, 127.6948, 174.9499, 207.585, 228.8421, 281.2939, 292.4101,
                                   308.0345, 375.4493, 486.8396, 498.6986, 537.6196, 564.0223, 615.3762,
                                   741.8843, 749.3428, 777.1524, 855.3031, 871.055, 962.7075, 977.6181,
                                   1050.3147, 1051.8134, 1071.7234, 1082.0731, 1146.8729, 1170.0212, 1179.6722,
                                   1189.1581, 1206.0905, 1269.8371, 1313.4043, 1355.1081, 1380.155, 1429.7095,
                                   1464.5357, 1475.5996, 1493.6501, 1494.3533, 1500.9964, 1507.5851, 1532.7927,
                                   1587.7095, 1643.0702, 2992.7203, 3045.3662, 3068.6577, 3123.9646, 3158.2579,
                                   3159.3532, 3199.1684, 3211.4927, 3223.942, 3233.9201], np.float64)
        np.testing.assert_almost_equal(freqs, expected_freqs)
        expected_normal_disp_modes_0 = np.array(
            [[-0.0, 0.0, -0.09], [-0.0, 0.0, -0.1], [0.0, 0.0, -0.01], [0.0, 0.0, -0.07], [0.0, 0.0, -0.2],
             [-0.0, -0.0, 0.28], [0.0, -0.0, -0.08], [0.0, -0.0, 0.01], [0.0, -0.0, 0.12], [0.0, -0.0, 0.12],
             [0.08, -0.02, -0.04], [-0.08, 0.02, -0.04], [-0.0, 0.0, -0.18], [-0.3, -0.03, 0.41], [-0.0, -0.0, 0.4],
             [0.3, 0.03, 0.41], [0.0, 0.0, -0.15], [0.0, -0.0, 0.01], [0.0, -0.0, 0.21], [0.0, -0.0, 0.19]], np.float64)
        np.testing.assert_almost_equal(normal_disp_modes[0], expected_normal_disp_modes_0)

    def test_parse_xyz_from_file(self):
        """Test parsing xyz from a file"""
        path1 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'CH3C(O)O.gjf')
        path2 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'CH3C(O)O.xyz')
        path3 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'AIBN.gjf')
        path4 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'molpro.in')
        path5 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'qchem.in')
        path6 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'qchem_output.out')
        path7 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'TS.gjf')
        path8 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'optim_traj_terachem.xyz')

        xyz1 = parser.parse_xyz_from_file(path1)
        xyz2 = parser.parse_xyz_from_file(path2)
        xyz3 = parser.parse_xyz_from_file(path3)
        xyz4 = parser.parse_xyz_from_file(path4)
        xyz5 = parser.parse_xyz_from_file(path5)
        xyz6 = parser.parse_xyz_from_file(path6)
        xyz7 = parser.parse_xyz_from_file(path7)
        xyz8 = parser.parse_xyz_from_file(path8)

        self.assertEqual(xyz1, xyz2)
        xyz1_str = xyz_to_str(xyz1)
        xyz2_str = xyz_to_str(xyz2)
        xyz3_str = xyz_to_str(xyz3)
        xyz4_str = xyz_to_str(xyz4)
        xyz5_str = xyz_to_str(xyz5)
        xyz6_str = xyz_to_str(xyz6)
        xyz8_str = xyz_to_str(xyz8)
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
        expected_xyz_8 = """N      -0.67665958    0.74524340   -0.41319355
H      -1.26179357    1.52577220   -0.13687665
H       0.28392722    1.06723640   -0.44163375
N      -0.75345799   -0.33268278    0.51180786
H      -0.97153041   -0.02416219    1.45398654
H      -1.48669570   -0.95874053    0.20627423
N       2.28178508   -0.42455356    0.14404399
H       1.32677989   -0.80557411    0.33156013"""
        self.assertEqual(xyz8_str, expected_xyz_8)

    def test_parse_t1(self):
        """Test T1 diagnostic parsing"""
        path = os.path.join(arc_path, 'arc', 'testing', 'mehylamine_CCSD(T).out')
        t1 = parser.parse_t1(path)
        self.assertEqual(t1, 0.0086766)

    def test_parse_e_elect(self):
        """Test parsing E0 from an sp job output file"""
        path1 = os.path.join(arc_path, 'arc', 'testing', 'mehylamine_CCSD(T).out')
        e_elect = parser.parse_e_elect(path1)
        self.assertEqual(e_elect, -251377.49160993524)

        path2 = os.path.join(arc_path, 'arc', 'testing', 'SO2OO_CBS-QB3.log')
        e_elect = parser.parse_e_elect(path2, zpe_scale_factor=0.99)
        self.assertEqual(e_elect, -1833127.0939478774)

    def test_parse_zpe(self):
        """Test the parse_zpe() function for parsing zero point energies"""
        path1 = os.path.join(arc_path, 'arc', 'testing', 'C2H6_freq_QChem.out')
        path2 = os.path.join(arc_path, 'arc', 'testing', 'CH2O_freq_molpro.out')
        path3 = os.path.join(arc_path, 'arc', 'testing', 'NO3_freq_QChem_fails_on_cclib.out')
        path4 = os.path.join(arc_path, 'arc', 'testing', 'SO2OO_CBS-QB3.log')
        zpe1, zpe2, zpe3, zpe4 = parser.parse_zpe(path1), parser.parse_zpe(path2), parser.parse_zpe(path3), \
            parser.parse_zpe(path4)
        self.assertAlmostEqual(zpe1, 198.08311200000, 5)
        self.assertAlmostEqual(zpe2, 69.793662734869, 5)
        self.assertAlmostEqual(zpe3, 25.401064000000, 5)
        self.assertAlmostEqual(zpe4, 39.368057626223, 5)

    def test_parse_1d_scan_energies(self):
        """Test parsing a 1D scan output file"""
        path1 = os.path.join(arc_path, 'arc', 'testing', 'rotor_scans', 'sBuOH.out')
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
        np.testing.assert_almost_equal(angles, expected_angles)

    def test_parse_nd_scan_energies(self):
        """Test parsing an ND scan output file"""
        path1 = os.path.join(arc_path, 'arc', 'testing', 'rotor_scans', 'scan_2D_relaxed_OCdOO.log')
        results = parser.parse_nd_scan_energies(path=path1, software='gaussian')
        self.assertEqual(results['directed_scan_type'], 'ess_gaussian')
        self.assertEqual(results['scans'], [(4, 1, 2, 5), (4, 1, 3, 6)])
        self.assertEqual(len(list(results.keys())), 3)
        self.assertEqual(len(list(results['directed_scan'].keys())), 36 * 36 + 1)  # 1297
        self.assertAlmostEqual(results['directed_scan']['170.00', '40.00']['energy'], 26.09747088)

    def test_parse_dipole_moment(self):
        """Test parsing the dipole moment from an opt job output file"""
        path1 = os.path.join(arc_path, 'arc', 'testing', 'SO2OO_CBS-QB3.log')
        dm1 = parser.parse_dipole_moment(path1)
        self.assertEqual(dm1, 0.63)

        path2 = os.path.join(arc_path, 'arc', 'testing', 'N2H4_opt_QChem.out')
        dm2 = parser.parse_dipole_moment(path2)
        self.assertEqual(dm2, 2.0664)

        path3 = os.path.join(arc_path, 'arc', 'testing', 'CH2O_freq_molpro.out')
        dm3 = parser.parse_dipole_moment(path3)
        self.assertAlmostEqual(dm3, 2.8840, 4)

    def test_parse_polarizability(self):
        """Test parsing the polarizability moment from a freq job output file"""
        path1 = os.path.join(arc_path, 'arc', 'testing', 'SO2OO_CBS-QB3.log')
        polar1 = parser.parse_polarizability(path1)
        self.assertAlmostEqual(polar1, 3.99506, 4)

    def test_process_conformers_file(self):
        """Test processing ARC conformer files"""
        path1 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'conformers_before_optimization.txt')
        path2 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'conformers_after_optimization.txt')
        path3 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'conformers_file.txt')

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


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
