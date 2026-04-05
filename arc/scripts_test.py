#!/usr/bin/env python3
# encoding: utf-8

"""
Tests for ARC helper scripts that run in the RMG conda environment.

These tests call the scripts as subprocesses in rmg_env (matching production usage).
Tests are skipped if rmg_env is not available.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

from arc.common import ARC_PATH, ARC_TESTING_PATH, read_yaml_file


def _rmg_env_available() -> bool:
    """Check whether rmg_env conda environment is available."""
    try:
        result = subprocess.run(
            ['conda', 'run', '-n', 'rmg_env', 'python', '-c', 'import rmgpy'],
            capture_output=True, timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


RMG_ENV = _rmg_env_available()


@unittest.skipUnless(RMG_ENV, 'rmg_env not available')
class TestSaveArkaneThermo(unittest.TestCase):
    """Test the save_arkane_thermo.py script."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        # Copy the test thermo library
        src = os.path.join(ARC_TESTING_PATH, 'statmech', 'thermo', 'RMG_libraries')
        dst = os.path.join(self.tmp_dir, 'RMG_libraries')
        shutil.copytree(src, dst)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_produces_thermo_yaml(self):
        """Run the script and verify it produces a valid thermo.yaml."""
        script = os.path.join(ARC_PATH, 'arc', 'scripts', 'save_arkane_thermo.py')
        result = subprocess.run(
            ['conda', 'run', '-n', 'rmg_env', 'python', script],
            capture_output=True, text=True, cwd=self.tmp_dir, timeout=60,
        )
        self.assertEqual(result.returncode, 0, f'Script failed: {result.stderr}')

        yaml_path = os.path.join(self.tmp_dir, 'thermo.yaml')
        self.assertTrue(os.path.isfile(yaml_path))

        data = read_yaml_file(yaml_path)
        self.assertIsInstance(data, dict)
        self.assertIn('CHO', data)
        self.assertIn('CH4', data)
        self.assertIn('CH2O', data)
        self.assertIn('CH3', data)

    def test_h298_s298_values(self):
        """Verify H298 and S298 are reasonable."""
        script = os.path.join(ARC_PATH, 'arc', 'scripts', 'save_arkane_thermo.py')
        subprocess.run(
            ['conda', 'run', '-n', 'rmg_env', 'python', script],
            capture_output=True, cwd=self.tmp_dir, timeout=60,
        )
        data = read_yaml_file(os.path.join(self.tmp_dir, 'thermo.yaml'))

        # CHO: H298 ~ 41 kJ/mol (radical), S298 ~ 224 J/(mol*K)
        self.assertAlmostEqual(data['CHO']['H298'], 41.3, delta=1.0)
        self.assertAlmostEqual(data['CHO']['S298'], 224.1, delta=1.0)

        # CH4: H298 ~ -79 kJ/mol, S298 ~ 186 J/(mol*K)
        self.assertAlmostEqual(data['CH4']['H298'], -78.8, delta=1.0)
        self.assertAlmostEqual(data['CH4']['S298'], 186.1, delta=1.0)

    def test_nasa_polynomials_present(self):
        """Verify NASA polynomial data is extracted."""
        script = os.path.join(ARC_PATH, 'arc', 'scripts', 'save_arkane_thermo.py')
        subprocess.run(
            ['conda', 'run', '-n', 'rmg_env', 'python', script],
            capture_output=True, cwd=self.tmp_dir, timeout=60,
        )
        data = read_yaml_file(os.path.join(self.tmp_dir, 'thermo.yaml'))

        for label in ['CHO', 'CH4', 'CH2O', 'CH3']:
            self.assertIn('nasa_low', data[label], f'Missing nasa_low for {label}')
            self.assertIn('nasa_high', data[label], f'Missing nasa_high for {label}')
            self.assertIsNotNone(data[label]['nasa_low'])
            self.assertIsNotNone(data[label]['nasa_high'])
            self.assertEqual(len(data[label]['nasa_low']['coeffs']), 7)
            self.assertEqual(len(data[label]['nasa_high']['coeffs']), 7)

    def test_cp_data_present(self):
        """Verify tabulated Cp data is extracted."""
        script = os.path.join(ARC_PATH, 'arc', 'scripts', 'save_arkane_thermo.py')
        subprocess.run(
            ['conda', 'run', '-n', 'rmg_env', 'python', script],
            capture_output=True, cwd=self.tmp_dir, timeout=60,
        )
        data = read_yaml_file(os.path.join(self.tmp_dir, 'thermo.yaml'))

        for label in ['CHO', 'CH4', 'CH2O', 'CH3']:
            self.assertIn('cp_data', data[label], f'Missing cp_data for {label}')
            cp = data[label]['cp_data']
            self.assertIsInstance(cp, list)
            self.assertGreater(len(cp), 0)
            self.assertIn('temperature_k', cp[0])
            self.assertIn('cp_j_mol_k', cp[0])


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
