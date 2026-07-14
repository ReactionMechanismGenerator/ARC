#!/usr/bin/env python3
# encoding: utf-8

"""
Tests for ARC helper scripts that run in the RMG conda environment.

These tests call the scripts as subprocesses in rmg_env (matching production usage).
Tests are skipped if rmg_env is not available.
"""

import math
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

R_J_MOL_K = 8.314462618


def _nasa_enthalpy(coeffs: list, t: float) -> float:
    """Return H(T) in J/mol from the 7 NASA coefficients of a single temperature range."""
    a1, a2, a3, a4, a5, a6, _ = coeffs
    return R_J_MOL_K * t * (a1 + a2 * t / 2 + a3 * t ** 2 / 3 + a4 * t ** 3 / 4 + a5 * t ** 4 / 5 + a6 / t)


def _nasa_entropy(coeffs: list, t: float) -> float:
    """Return S(T) in J/(mol*K) from the 7 NASA coefficients of a single temperature range."""
    a1, a2, a3, a4, a5, _, a7 = coeffs
    return R_J_MOL_K * (a1 * math.log(t) + a2 * t + a3 * t ** 2 / 2 + a4 * t ** 3 / 3 + a5 * t ** 4 / 4 + a7)


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


@unittest.skipUnless(RMG_ENV, 'rmg_env not available')
class TestSaveArkaneThermoOutputPyFallback(unittest.TestCase):
    """
    The motivating scenario end to end: Arkane's save_thermo_lib crashed on the two identical
    reactants of an A+A reaction, so it left output.py behind but never wrote
    RMG_libraries/thermo.py. Running the real script must still produce thermo.yaml for every
    species, including both duplicates.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='test_aa_reload_')
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        src = os.path.join(ARC_TESTING_PATH, 'statmech', 'thermo_aa', 'output.py')
        shutil.copy(src, os.path.join(self.tmp_dir, 'output.py'))
        self.assertFalse(os.path.isdir(os.path.join(self.tmp_dir, 'RMG_libraries')))

    def _run_script(self):
        script = os.path.join(ARC_PATH, 'arc', 'scripts', 'save_arkane_thermo.py')
        result = subprocess.run(['conda', 'run', '-n', 'rmg_env', 'python', script],
                                capture_output=True, text=True, cwd=self.tmp_dir, timeout=300)
        self.assertEqual(result.returncode, 0, f'Script failed: {result.stderr}')
        yaml_path = os.path.join(self.tmp_dir, 'thermo.yaml')
        self.assertTrue(os.path.isfile(yaml_path),
                        'thermo.yaml must be recovered from output.py when the library is absent')
        return read_yaml_file(yaml_path)

    def test_both_duplicates_recover_their_own_thermo(self):
        """R1 and R2 are the same species submitted twice; each recovers thermo from its own block."""
        data = self._run_script()
        self.assertEqual(set(data), {'R1', 'R2', 'P1'})
        for label in ('R1', 'R2', 'P1'):
            self.assertIsNotNone(data[label]['H298'])
            self.assertIsNotNone(data[label]['S298'])
            self.assertIsNotNone(data[label]['nasa_low'])
            self.assertIsNotNone(data[label]['nasa_high'])
        self.assertAlmostEqual(data['R1']['H298'], data['R2']['H298'])
        self.assertAlmostEqual(data['R1']['S298'], data['R2']['S298'])

    def test_recovered_values_match_reference_thermochemistry(self):
        """P1 is H2O; the recovered values are checked against JANAF, at 298 K and at 2000 K.

        The high-temperature check evaluates the serialized nasa_high coefficients, which is what
        ARC consumers use up to 3000 K, so a reload that corrupted them cannot pass.
        """
        data = self._run_script()
        self.assertAlmostEqual(data['P1']['H298'], -241.8, delta=3.0)
        self.assertAlmostEqual(data['P1']['S298'], 188.8, delta=2.0)
        nasa_high = data['P1']['nasa_high']
        self.assertGreaterEqual(nasa_high['tmax_k'], 3000.0)
        h_2000 = _nasa_enthalpy(nasa_high['coeffs'], 2000.0) / 1000.0
        s_2000 = _nasa_entropy(nasa_high['coeffs'], 2000.0)
        self.assertAlmostEqual(h_2000 - data['P1']['H298'], 72.79, delta=3.0)
        self.assertAlmostEqual(s_2000, 264.77, delta=3.0)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
