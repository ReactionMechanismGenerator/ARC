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
import textwrap
import unittest

from arc.common import ARC_PATH, ARC_TESTING_PATH, read_yaml_file, save_yaml_file
from arc.scripts.common import parse_command_line_arguments


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

    def test_thermo_points_present(self):
        """Verify tabulated thermo points (Cp + H + S + G per T) are extracted."""
        script = os.path.join(ARC_PATH, 'arc', 'scripts', 'save_arkane_thermo.py')
        subprocess.run(
            ['conda', 'run', '-n', 'rmg_env', 'python', script],
            capture_output=True, cwd=self.tmp_dir, timeout=60,
        )
        data = read_yaml_file(os.path.join(self.tmp_dir, 'thermo.yaml'))

        for label in ['CHO', 'CH4', 'CH2O', 'CH3']:
            self.assertIn('thermo_points', data[label],
                          f'Missing thermo_points for {label}')
            points = data[label]['thermo_points']
            self.assertIsInstance(points, list)
            self.assertGreater(len(points), 0)
            first = points[0]
            for key in ('temperature_k', 'cp_j_mol_k', 'h_kj_mol',
                        's_j_mol_k', 'g_kj_mol'):
                self.assertIn(key, first,
                              f'{label} thermo_points[0] missing {key}')
                self.assertIsInstance(first[key], (int, float))


class TestCommonArgparse(unittest.TestCase):
    """Test the shared CLI parser used by the standalone scripts."""

    def test_positional_file_only(self):
        """Without ``--output`` the parser exposes ``args.output is None``."""
        args = parse_command_line_arguments(['/tmp/in.yml'])
        self.assertEqual(args.file, '/tmp/in.yml')
        self.assertIsNone(args.output)

    def test_output_long_form(self):
        """``--output`` populates ``args.output`` so callers can avoid overwriting input."""
        args = parse_command_line_arguments(['/tmp/in.yml', '--output', '/tmp/out.yml'])
        self.assertEqual(args.file, '/tmp/in.yml')
        self.assertEqual(args.output, '/tmp/out.yml')

    def test_output_short_form(self):
        """``-o`` is an accepted short form."""
        args = parse_command_line_arguments(['/tmp/in.yml', '-o', '/tmp/out.yml'])
        self.assertEqual(args.output, '/tmp/out.yml')


@unittest.skipUnless(RMG_ENV, 'rmg_env not available')
class TestRmgKineticsHelpers(unittest.TestCase):
    """
    Unit tests for ``rmg_kinetics.py`` helpers that don't need a full RMG database load.

    Each test runs a tiny ``python -c`` snippet inside ``rmg_env`` so we can import
    rmgpy and the script module directly. Stdout is parsed as JSON.
    """

    SCRIPT_DIR = os.path.join(ARC_PATH, 'arc', 'scripts')

    def _run_in_rmg_env(self, snippet: str) -> str:
        """Execute ``snippet`` inside rmg_env and return stripped stdout."""
        result = subprocess.run(
            ['conda', 'run', '-n', 'rmg_env', 'python', '-c', snippet],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0,
                         f'snippet failed: stderr={result.stderr}\nstdout={result.stdout}')
        return result.stdout.strip()

    def test_get_kinetics_from_reactions_arrhenius(self):
        """``get_kinetics_from_reactions`` reports A/n/Ea (Ea in kJ/mol) for an Arrhenius rxn."""
        snippet = textwrap.dedent(f"""
            import sys, json
            sys.path.insert(0, {self.SCRIPT_DIR!r})
            from rmg_kinetics import get_kinetics_from_reactions
            from rmgpy.kinetics import Arrhenius
            from rmgpy.reaction import Reaction
            rxn = Reaction()
            rxn.kinetics = Arrhenius(A=(1.5e13, 'cm^3/(mol*s)'), n=0.0, Ea=(20.0, 'kJ/mol'),
                                     Tmin=(300.0, 'K'), Tmax=(2500.0, 'K'))
            rxn.comment = 'unit-test'
            out = get_kinetics_from_reactions([rxn])
            print(json.dumps(out[0]))
        """)
        import json
        entry = json.loads(self._run_in_rmg_env(snippet))
        self.assertEqual(entry['comment'], 'unit-test')
        self.assertAlmostEqual(entry['A'], 1.5e13, delta=1e7)
        self.assertEqual(entry['n'], 0.0)
        self.assertAlmostEqual(entry['Ea'], 20.0, places=6)  # kJ/mol
        self.assertEqual(entry['T_min'], 300.0)
        self.assertEqual(entry['T_max'], 2500.0)

    def test_get_kinetics_from_reactions_handles_missing_T_bounds(self):
        """Tmin/Tmax may be absent; helper should yield None rather than crashing."""
        snippet = textwrap.dedent(f"""
            import sys, json
            sys.path.insert(0, {self.SCRIPT_DIR!r})
            from rmg_kinetics import get_kinetics_from_reactions
            from rmgpy.kinetics import Arrhenius
            from rmgpy.reaction import Reaction
            rxn = Reaction()
            rxn.kinetics = Arrhenius(A=(1.0, 's^-1'), n=1.0, Ea=(0.0, 'J/mol'))
            rxn.comment = 'no-T-bounds'
            print(json.dumps(get_kinetics_from_reactions([rxn])[0]))
        """)
        import json
        entry = json.loads(self._run_in_rmg_env(snippet))
        self.assertIsNone(entry['T_min'])
        self.assertIsNone(entry['T_max'])

    def test_scale_kinetics_by_degeneracy_skips_non_arrhenius(self):
        """``scale_kinetics_by_degeneracy`` scales Arrhenius A by the degeneracy but
        leaves non-Arrhenius forms (e.g. Chebyshev) untouched. Dropping the guard would
        let ``change_rate`` corrupt the Chebyshev coefficients and fail this test."""
        snippet = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {self.SCRIPT_DIR!r})
            from rmg_kinetics import scale_kinetics_by_degeneracy
            from rmgpy.kinetics import Arrhenius, Chebyshev
            arr = Arrhenius(A=(1.0, 's^-1'), n=0.0, Ea=(0.0, 'J/mol'))
            scale_kinetics_by_degeneracy(arr, 2)
            assert abs(arr.A.value_si - 2.0) < 1e-9, arr.A.value_si
            cheb = Chebyshev(coeffs=[[1.0, 0.0], [0.0, 0.0]],
                             kunits='cm^3/(mol*s)',
                             Tmin=(300.0, 'K'), Tmax=(2000.0, 'K'),
                             Pmin=(0.01, 'bar'), Pmax=(100.0, 'bar'))
            before = cheb.coeffs.value_si.tolist()
            scale_kinetics_by_degeneracy(cheb, 2)
            assert cheb.coeffs.value_si.tolist() == before, 'Chebyshev coeffs were mutated'
            print('ok')
        """)
        self.assertEqual(self._run_in_rmg_env(snippet), 'ok')


@unittest.skipUnless(RMG_ENV, 'rmg_env not available')
class TestRmgScriptsOutputFlag(unittest.TestCase):
    """Verify ``--output`` writes to a fresh path and leaves the input file untouched."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='rmg_scripts_test_')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _h2_adjlist(self) -> str:
        return '1 H u0 p0 c0 {2,S}\n2 H u0 p0 c0 {1,S}\n'

    def test_rmg_thermo_output_does_not_overwrite_input(self):
        """The thermo script writes the augmented YAML to ``--output`` and preserves input."""
        input_path = os.path.join(self.tmp_dir, 'in.yml')
        output_path = os.path.join(self.tmp_dir, 'out.yml')
        original = [{'label': 'H2', 'adjlist': self._h2_adjlist()}]
        save_yaml_file(path=input_path, content=original)
        with open(input_path, 'rb') as f:
            input_bytes_before = f.read()

        script = os.path.join(ARC_PATH, 'arc', 'scripts', 'rmg_thermo.py')
        result = subprocess.run(
            ['conda', 'run', '-n', 'rmg_env', 'python', script, input_path, '--output', output_path],
            capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(result.returncode, 0, f'thermo script failed: {result.stderr}')

        # Input must be byte-identical (the script must not overwrite it).
        with open(input_path, 'rb') as f:
            self.assertEqual(f.read(), input_bytes_before)
        # Output must contain the new keys.
        out = read_yaml_file(output_path)
        self.assertEqual(len(out), 1)
        self.assertIn('h298', out[0])
        self.assertIn('s298', out[0])
        self.assertIn('comment', out[0])


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
