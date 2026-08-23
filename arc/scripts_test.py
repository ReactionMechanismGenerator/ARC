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
import textwrap
import unittest

from arc.common import ARC_PATH, ARC_TESTING_PATH, read_yaml_file, save_yaml_file
from arc.job.env_run import rmg_env_command
from arc.scripts.common import parse_command_line_arguments


def _run_rmg_env(py_args, cwd=None, timeout=120, text=True):
    """
    Run ``python <py_args>`` inside rmg_env via ARC's production launcher helper.

    Routing through ``rmg_env_command`` (rather than a hardcoded ``conda run``) matches
    how ARC invokes these scripts in production and selects the right launcher
    (micromamba/mamba/conda) on whatever host the tests run on.

    Args:
        py_args (list[str]): Everything after ``python`` (e.g. ``[script, '--output', out]``
                             or ``['-c', snippet]``).
        cwd (str, optional): Directory to run in.
        timeout (int): Subprocess timeout in seconds.
        text (bool): Whether to decode stdout/stderr as text.

    Returns:
        subprocess.CompletedProcess: The finished process.
    """
    command = rmg_env_command(py_args=py_args, cwd=cwd)
    return subprocess.run(command, shell=True, executable='/bin/bash',
                          capture_output=True, text=text, timeout=timeout)


def _rmg_env_available() -> bool:
    """Check whether the rmg_env conda environment is available."""
    try:
        return _run_rmg_env(['-c', 'import rmgpy'], timeout=30, text=False).returncode == 0
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
        result = _run_rmg_env([script], cwd=self.tmp_dir, timeout=60)
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
        _run_rmg_env([script], cwd=self.tmp_dir, timeout=60)
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
        _run_rmg_env([script], cwd=self.tmp_dir, timeout=60)
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
        _run_rmg_env([script], cwd=self.tmp_dir, timeout=60)
        data = read_yaml_file(os.path.join(self.tmp_dir, 'thermo.yaml'))

        for label in ['CHO', 'CH4', 'CH2O', 'CH3']:
            self.assertIn('cp_data', data[label], f'Missing cp_data for {label}')
            cp = data[label]['cp_data']
            self.assertIsInstance(cp, list)
            self.assertGreater(len(cp), 0)
            self.assertIn('temperature_k', cp[0])
            self.assertIn('cp_j_mol_k', cp[0])


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
        result = _run_rmg_env(['-c', snippet], timeout=120)
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

    def test_get_kinetics_from_reactions_converts_si_to_cm(self):
        """``get_kinetics_from_reactions`` reports A in the cm/mol/s convention.

        A bimolecular Arrhenius built in SI (``m^3/(mol*s)``) must be reported ×1e6
        (``cm^3/(mol*s)``) via ``A.get_conversion_factor_from_si_to_cm_mol_s()`` — this
        exercises the units contract directly rather than reading a cm value straight back.
        """
        snippet = textwrap.dedent(f"""
            import sys, json
            sys.path.insert(0, {self.SCRIPT_DIR!r})
            from rmg_kinetics import get_kinetics_from_reactions
            from rmgpy.kinetics import Arrhenius
            from rmgpy.reaction import Reaction
            rxn = Reaction()
            rxn.kinetics = Arrhenius(A=(1.0, 'm^3/(mol*s)'), n=0.0, Ea=(0.0, 'J/mol'))
            rxn.comment = 'si-input'
            print(json.dumps(get_kinetics_from_reactions([rxn])[0]))
        """)
        import json
        entry = json.loads(self._run_in_rmg_env(snippet))
        self.assertAlmostEqual(entry['A'], 1.0e6, delta=1.0)  # m^3/(mol*s) -> cm^3/(mol*s)


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
        result = _run_rmg_env([script, input_path, '--output', output_path], timeout=300)
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
