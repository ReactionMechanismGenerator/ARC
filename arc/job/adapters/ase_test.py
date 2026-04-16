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
from arc.job.adapters.ase import ASEAdapter
from arc.species.species import ARCSpecies
from arc.job.adapters.scripts.ase_script import to_kJmol, numpy_vibrational_analysis


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
        with patch('arc.job.adapters.ase.settings', {'TANI_PYTHON': '/path/to/tani_python'}):
            exe = self.job_1.get_python_executable()
            self.assertEqual(exe, '/path/to/tani_python')

        with patch('arc.job.adapters.ase.settings', {'XTB_PYTHON': '/path/to/xtb_python'}):
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
        self.assertAlmostEqual(to_kJmol(1.0), 96.48533, places=5)
        self.assertAlmostEqual(to_kJmol(27.21138), 2625.499015202655, places=5)

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
        
        # nonlinear (len > 2), filters out first 6 modes
        self.assertEqual(len(results['freqs']), 3)
        self.assertEqual(len(results['modes']), 3)
        self.assertEqual(len(results['force_constants']), 3)
        self.assertEqual(len(results['reduced_masses']), 3)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        shutil.rmtree(cls.project_directory, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
