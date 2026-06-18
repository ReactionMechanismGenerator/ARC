#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.adapters.ase module.
These tests verify IO and logic without executing the external ASE script in CI.
"""

import os
import shutil
import unittest
from unittest.mock import patch
import numpy as np

from arc.common import ARC_TESTING_PATH, read_yaml_file, save_yaml_file
from arc.job.adapters.ase_adapter import ASEAdapter
from arc.species.species import ARCSpecies
from arc.job.adapters.scripts.ase_script import to_kJmol, numpy_vibrational_analysis, is_linear


class TestASEAdapter(unittest.TestCase):
    """
    Contains unit tests for the ASEAdapter class and ase_script utility functions.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.project_directory = os.path.join(ARC_TESTING_PATH, 'test_ASEAdapter')
        if not os.path.exists(cls.project_directory):
            os.makedirs(cls.project_directory)

        xyz = {'symbols': ('O', 'H', 'H'),
               'isotopes': (16, 1, 1),
               'coords': ((0.0, 0.0, 0.0),
                          (0.0, 0.75, 0.58),
                          (0.0, -0.75, 0.58))}
        
        cls.job_1 = ASEAdapter(execution_type='incore',
                               job_type='sp',
                               project='test_1',
                               project_directory=os.path.join(cls.project_directory, 'test_1'),
                               species=[ARCSpecies(label='H2O', xyz=xyz)],
                               args={'keyword': {'calculator': 'torchani', 'model': 'ANI2x'}},
                               testing=True)

        cls.job_2 = ASEAdapter(execution_type='queue',
                               job_type='opt',
                               project='test_2',
                               project_directory=os.path.join(cls.project_directory, 'test_2'),
                               species=[ARCSpecies(label='H2O', xyz=xyz)],
                               args={'keyword': {'calculator': 'xtb', 'method': 'GFN2-xTB'}},
                               testing=True)
                               
        cls.job_1.local_path = os.path.join(cls.project_directory, 'test_1')
        cls.job_2.local_path = os.path.join(cls.project_directory, 'test_2')
        cls.job_2.remote_path = '/path/to/remote'
        os.makedirs(cls.job_1.local_path, exist_ok=True)
        os.makedirs(cls.job_2.local_path, exist_ok=True)

    def test_get_python_executable(self):
        """Test resolving the python executable environment"""
        with patch('arc.job.adapters.ase_adapter.settings', {'TANI_PYTHON': '/path/to/tani_python'}):
            exe = self.job_1.get_python_executable()
            self.assertEqual(exe, '/path/to/tani_python')

        with patch('arc.job.adapters.ase_adapter.settings', {'XTB_PYTHON': '/path/to/xtb_python'}):
            exe = self.job_2.get_python_executable()
            self.assertEqual(exe, '/path/to/xtb_python')

    def test_write_input_file(self):
        """Test writing the YAML input file for the ASE script"""
        self.job_1.write_input_file()
        input_path = os.path.join(self.job_1.local_path, 'input.yml')
        self.assertTrue(os.path.isfile(input_path))
        data = read_yaml_file(input_path)
        self.assertEqual(data['job_type'], 'sp')
        self.assertEqual(data['settings']['calculator'], 'torchani')
        self.assertEqual(data['settings']['model'], 'ANI2x')
        self.assertEqual(data['xyz']['symbols'], ('O', 'H', 'H'))

    def test_write_submit_script(self):
        """Test writing the submission script for queue execution"""
        self.job_2.python_executable = '/fake/python'
        self.job_2.write_submit_script()
        submit_path = os.path.join(self.job_2.local_path, 'submit.sh')
        self.assertTrue(os.path.isfile(submit_path))
        with open(submit_path, 'r') as f:
            content = f.read()
        self.assertIn('/fake/python', content)
        self.assertIn('--yml_path /path/to/remote', content)
        self.assertIn('ase_script.py', content)

    def test_set_files(self):
        """Test properly assigning upload and download files"""
        self.job_2.set_files()
        self.assertTrue(any('submit.sh' in f['local'] for f in self.job_2.files_to_upload))
        self.assertTrue(any('input.yml' in f['local'] for f in self.job_2.files_to_upload))
        self.assertTrue(any('ase_script.py' in f['local'] for f in self.job_2.files_to_upload))
        self.assertTrue(any('output.yml' in f['local'] for f in self.job_2.files_to_download))

    def test_parse_results(self):
        """Test parsing dummy output YAML back into object attributes"""
        output_data = {
            'sp': -76.0,
            'opt_xyz': {'symbols': ('O', 'H', 'H'), 'coords': ((0.0, 0.0, 0.0), (0.0, 0.76, 0.59), (0.0, -0.76, 0.59))},
            'freqs': [1500.0, 3600.0, 3700.0],
            'modes': [[[0.0, 0.0, 0.1]]],
            'reduced_masses': [1.0, 1.0, 1.0],
            'force_constants': [1.0, 2.0, 3.0]
        }
        save_yaml_file(os.path.join(self.job_1.local_path, 'output.yml'), output_data)
        self.job_1.parse_results()
        self.assertEqual(self.job_1.electronic_energy, -76.0)
        self.assertEqual(self.job_1.frequencies, [1500.0, 3600.0, 3700.0])
        self.assertEqual(self.job_1.force_constants, [1.0, 2.0, 3.0])
        self.assertIsNotNone(self.job_1.xyz_out)
        self.assertAlmostEqual(self.job_1.xyz_out['coords'][1][1], 0.76)

    def test_to_kJmol(self):
        """Test utility conversion function to_kJmol"""
        self.assertAlmostEqual(to_kJmol(1.0), 96.48534, places=5)
        self.assertAlmostEqual(to_kJmol(27.21138), 2625.49937, places=5)

    def test_is_linear(self):
        """Test the is_linear helper function in ase_script"""
        from ase import Atoms
        # 1. Monatomic (H)
        h = Atoms('H', positions=[(0.0, 0.0, 0.0)])
        self.assertFalse(is_linear(h))

        # 2. Diatomic (H2)
        h2 = Atoms('H2', positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)])
        self.assertTrue(is_linear(h2))

        # 3. Linear triatomic (CO2)
        co2 = Atoms('CO2', positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.16), (0.0, 0.0, -1.16)])
        self.assertTrue(is_linear(co2))

        # 4. Non-linear triatomic (H2O)
        h2o = Atoms('H2O', positions=[(0.0, 0.0, 0.0), (0.0, 0.75, 0.58), (0.0, -0.75, 0.58)])
        self.assertFalse(is_linear(h2o))

    def test_numpy_vibrational_analysis(self):
        """Test fallback numpy vibrational analysis directly"""
        masses = np.array([16.0, 1.0, 1.0])
        n_atoms = len(masses)
        # Create a hessian with some very small eigenvalues (for translations/rotations)
        # and some large ones.
        hessian = np.zeros((3 * n_atoms, 3 * n_atoms))
        for i in range(6, 9):
            hessian[i, i] = 10.0
        
        results = numpy_vibrational_analysis(masses, hessian)
        self.assertIn('freqs', results)
        self.assertIn('modes', results)
        self.assertIn('force_constants', results)
        self.assertIn('reduced_masses', results)
        print(results['freqs'])
        # nonlinear (len > 2), filters out first 6 modes
        self.assertEqual(len(results['freqs']), 3)
        self.assertEqual(len(results['modes']), 3)
        self.assertEqual(len(results['force_constants']), 3)
        self.assertEqual(len(results['reduced_masses']), 3)

        # 3 atom linear species, actual hessian from computation in Orca 6.0.0 r2scan-3c for O=C=O
        hessian = np.array(\
            [[ 1.47625806e-01,  2.29436182e-05,  1.34550341e-05, -7.38129035e-02,
            -7.97506445e-06, -4.70698398e-06, -7.38129030e-02, -7.97743334e-06,
            -4.70838645e-06],
            [ 2.29436182e-05,  1.49024440e+00,  7.76950445e-01, -1.49650473e-05,
            -7.45122209e-01, -3.88386820e-01, -1.49696914e-05, -7.45122187e-01,
            -3.88386839e-01],
            [ 1.34550341e-05,  7.76950445e-01,  5.96226997e-01, -8.74599081e-06,
            -3.88563609e-01, -2.98113486e-01, -8.74870706e-06, -3.88563621e-01,
            -2.98113511e-01],
            [-7.38129035e-02, -1.49650473e-05, -8.74599081e-06,  3.68163144e-02,
            9.08896394e-06,  5.35084314e-06,  3.69965890e-02,  2.38052672e-06,
            1.37531791e-06],
            [-7.97506445e-06, -7.45122209e-01, -3.88563609e-01,  9.08896394e-06,
            7.93484601e-01,  4.37753696e-01,  2.38165713e-06, -4.83623921e-02,
            -4.92784793e-02],
            [-4.70698398e-06, -3.88386820e-01, -2.98113486e-01,  5.35084314e-06,
            4.37753696e-01,  2.89551995e-01,  1.37597060e-06, -4.92784828e-02,
            8.56149142e-03],
            [-7.38129030e-02, -1.49696914e-05, -8.74870706e-06,  3.69965890e-02,
            2.38165713e-06,  1.37597060e-06,  3.68163139e-02,  9.09247043e-06,
            5.35290250e-06],
            [-7.97743334e-06, -7.45122187e-01, -3.88563621e-01,  2.38052672e-06,
            -4.83623921e-02, -4.92784828e-02,  9.09247043e-06,  7.93484579e-01,
            4.37753711e-01],
            [-4.70838645e-06, -3.88386839e-01, -2.98113511e-01,  1.37531791e-06,
            -4.92784793e-02,  8.56149142e-03,  5.35290250e-06,  4.37753711e-01,
            2.89552020e-01],])
        masses = np.array([12.0, 16.0, 16.0])
        freqs = np.array([0., 0., 0., 0., 0., 666.85873322, 668.56887375, 1362.1172728, 2423.3776014])

        conv_factor = 27.211386245988 / (0.529177210903 ** 2)
        results = numpy_vibrational_analysis(masses, hessian * conv_factor, is_linear=True)
        self.assertEqual(len(results['freqs']), 4)
        for i, val in enumerate(freqs[5:]):
            self.assertAlmostEqual(results['freqs'][i], val, delta=1e-3)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        shutil.rmtree(cls.project_directory, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
