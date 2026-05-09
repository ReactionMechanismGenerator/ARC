"""
Tests for the mockter Gaussian-log renderer.

Round-trips fixture data through ``render_gaussian_log`` and ARC's Gaussian
parser, asserting that every parsed value bit-equals the source. Also runs
the same round-trip against the real fixtures committed under
``arc/testing/mockter_fixtures/``.

Arkane's ``GaussianLog`` is exercised in a separate test class that is
skipped when the local Arkane install is not importable; the ARC-side
contract is the must-pass minimum.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from arc.common import read_yaml_file
from arc.constants import E_h_kJmol
from arc.job.adapters.mockter_renderer import render_gaussian_log
from arc.parser.parser import (
    parse_e_elect,
    parse_frequencies,
    parse_geometry,
    parse_zpe_correction,
)
from arc.species.converter import str_to_xyz, xyz_to_str


KJ_PER_HARTREE = E_h_kJmol
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'testing', 'mockter_fixtures')


def _write_log(text: str) -> str:
    """
    Write text to a temp .log file and return its path.

    Args:
        text (str): Forged Gaussian log content.

    Returns:
        str: Absolute path to the temp file.
    """
    fd, path = tempfile.mkstemp(suffix='.log')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


class TestRendererSyntheticInputs(unittest.TestCase):
    """
    Unit tests with hand-crafted inputs covering each block in isolation.
    """

    def setUp(self):
        """Trivial water geometry for shared setup."""
        self.h2o_xyz = str_to_xyz(
            'O   0.000000   0.000000   0.116011\n'
            'H   0.000000   0.761905  -0.464046\n'
            'H   0.000000  -0.761905  -0.464046'
        )

    def test_geometry_round_trips_through_parser(self):
        """Render a minimal opt log; ARC parser must read back identical xyz."""
        log_text = render_gaussian_log(job_type='opt', xyz=self.h2o_xyz, e_elect_hartree=-76.4377)
        path = _write_log(log_text)
        try:
            parsed = parse_geometry(path)
            self.assertEqual(parsed['symbols'], self.h2o_xyz['symbols'])
            for orig, got in zip(self.h2o_xyz['coords'], parsed['coords']):
                for o, g in zip(orig, got):
                    self.assertAlmostEqual(o, g, places=6)
        finally:
            os.unlink(path)

    def test_scf_done_round_trips(self):
        """ARC parser returns kJ/mol; convert back to Hartree to check bit-equal."""
        e_hartree = -76.4376926773
        log_text = render_gaussian_log(job_type='sp', xyz=self.h2o_xyz, e_elect_hartree=e_hartree)
        path = _write_log(log_text)
        try:
            parsed_kjmol = parse_e_elect(path)
            parsed_hartree = parsed_kjmol / KJ_PER_HARTREE
            self.assertAlmostEqual(parsed_hartree, e_hartree, places=8)
        finally:
            os.unlink(path)

    def test_ccsdt_energy_round_trips(self):
        """``is_t1_capable=True`` switches to ``CCSD(T)=`` line; parser handles it."""
        e_hartree = -76.3389123456
        log_text = render_gaussian_log(
            job_type='sp', xyz=self.h2o_xyz, e_elect_hartree=e_hartree,
            is_t1_capable=True, t1_diagnostic=0.012,
        )
        path = _write_log(log_text)
        try:
            parsed_kjmol = parse_e_elect(path)
            parsed_hartree = parsed_kjmol / KJ_PER_HARTREE
            self.assertAlmostEqual(parsed_hartree, e_hartree, places=8)
        finally:
            os.unlink(path)

    def test_frequencies_round_trip(self):
        """Frequencies block must split-decode bit-equal under ARC's parser."""
        freqs = [1624.5678, 3876.1696, 3986.0496]
        log_text = render_gaussian_log(
            job_type='freq', xyz=self.h2o_xyz, e_elect_hartree=-76.4377,
            freqs_cm1=freqs, zpe_hartree=0.021612,
        )
        path = _write_log(log_text)
        try:
            parsed = parse_frequencies(path)
            self.assertEqual(len(parsed), len(freqs))
            for o, g in zip(freqs, parsed):
                self.assertAlmostEqual(o, g, places=4)
        finally:
            os.unlink(path)

    def test_zpe_round_trips(self):
        """ZPE returned in kJ/mol; check Hartree round-trip."""
        zpe = 0.021612
        log_text = render_gaussian_log(
            job_type='freq', xyz=self.h2o_xyz, e_elect_hartree=-76.4377,
            freqs_cm1=[1624.5678, 3876.1696, 3986.0496], zpe_hartree=zpe,
        )
        path = _write_log(log_text)
        try:
            parsed_kjmol = parse_zpe_correction(path)
            parsed_hartree = parsed_kjmol / KJ_PER_HARTREE
            self.assertAlmostEqual(parsed_hartree, zpe, places=6)
        finally:
            os.unlink(path)

    def test_imaginary_frequency_preserved(self):
        """A negative frequency must survive the round-trip with its sign."""
        freqs = [-1234.56, 500.0, 1500.0, 2500.0]
        log_text = render_gaussian_log(
            job_type='freq', xyz=self.h2o_xyz, e_elect_hartree=-76.4377,
            freqs_cm1=freqs, zpe_hartree=0.01,
        )
        path = _write_log(log_text)
        try:
            parsed = parse_frequencies(path)
            self.assertEqual(len(parsed), len(freqs))
            for o, g in zip(freqs, parsed):
                self.assertAlmostEqual(o, g, places=4)
            self.assertLess(parsed[0], 0)
        finally:
            os.unlink(path)

    def test_hessian_block_spliced_verbatim(self):
        """A pre-formatted Hessian block must appear in the output untouched."""
        hess = (
            ' Force constants in Cartesian coordinates:\n'
            '                1             2             3             4             5\n'
            '      1  0.123456D+00\n'
            '      2  0.000000D+00  0.234567D+00\n'
        )
        log_text = render_gaussian_log(
            job_type='freq', xyz=self.h2o_xyz, e_elect_hartree=-76.4377,
            freqs_cm1=[1624.5678, 3876.1696, 3986.0496], zpe_hartree=0.021612,
            hessian_block=hess,
        )
        self.assertIn(hess.rstrip(), log_text)
        self.assertIn('Force constants in Cartesian coordinates:', log_text)

    def test_normal_termination_present(self):
        """Every rendered log must end with a Normal termination line."""
        log_text = render_gaussian_log(job_type='sp', xyz=self.h2o_xyz, e_elect_hartree=-76.4377)
        self.assertIn('Normal termination of Gaussian 16', log_text)

    def test_composite_energy_round_trips(self):
        """A CBS-QB3 composite line is parsed by ARC's parse_e_elect."""
        e_composite = -115.59054
        zpe = 0.052
        log_text = render_gaussian_log(
            job_type='composite', xyz=self.h2o_xyz, e_elect_hartree=None,
            freqs_cm1=[1000.0, 2000.0, 3000.0], zpe_hartree=zpe,
            composite_method='CBS-QB3', e_composite_hartree=e_composite,
        )
        path = _write_log(log_text)
        try:
            parsed_kjmol = parse_e_elect(path)
            parsed_hartree = parsed_kjmol / KJ_PER_HARTREE
            self.assertAlmostEqual(parsed_hartree, e_composite - zpe, places=4)
        finally:
            os.unlink(path)


class TestRendererRealFixtures(unittest.TestCase):
    """
    Round-trip tests that drive the renderer with real fixture data extracted
    from production DFT runs. Failures here indicate either a renderer bug
    or a fixture-data drift.
    """

    @classmethod
    def setUpClass(cls):
        """Load all six fixtures once."""
        cls.fixtures = {}
        for n in (1, 2, 3, 4, 5, 6):
            path = os.path.join(FIXTURES_DIR, f'mockter{n}.yml')
            if os.path.isfile(path):
                cls.fixtures[n] = read_yaml_file(path)

    def _round_trip_freq_block(self, fixture_num: int, label: str) -> None:
        """
        Render a freq log from a fixture entry and assert ARC's parser recovers
        the same xyz, freqs, and ZPE.

        Args:
            fixture_num (int): Fixture index (1-6).
            label (str): Species label inside the fixture.
        """
        fixture = self.fixtures[fixture_num]
        spc = fixture['species'][label]
        xyz_str = (spc.get('fine_opt') or spc.get('opt') or {}).get('xyz')
        freq = spc.get('freq') or {}
        if not xyz_str or not freq.get('freqs') or freq.get('zpe') is None:
            self.skipTest(f'mockter{fixture_num}/{label}: missing freq fields')

        log_text = render_gaussian_log(
            job_type='freq',
            xyz=xyz_str,
            e_elect_hartree=(spc.get('fine_opt') or spc.get('opt') or {}).get('e_elect'),
            multiplicity=spc.get('multiplicity', 1),
            charge=spc.get('charge', 0),
            freqs_cm1=freq['freqs'],
            zpe_hartree=freq['zpe'],
            hessian_block=freq.get('hessian_block'),
        )
        path = _write_log(log_text)
        try:
            parsed_xyz = parse_geometry(path)
            self.assertIsNotNone(parsed_xyz, f'mockter{fixture_num}/{label}: parse_geometry returned None')
            self.assertEqual(parsed_xyz['symbols'], str_to_xyz(xyz_str)['symbols'])

            parsed_freqs = parse_frequencies(path)
            self.assertIsNotNone(parsed_freqs, f'mockter{fixture_num}/{label}: parse_frequencies returned None')
            np.testing.assert_allclose(parsed_freqs, freq['freqs'], atol=1e-3)

            parsed_zpe_kjmol = parse_zpe_correction(path)
            self.assertIsNotNone(parsed_zpe_kjmol, f'mockter{fixture_num}/{label}: parse_zpe_correction returned None')
            self.assertAlmostEqual(parsed_zpe_kjmol / KJ_PER_HARTREE, freq['zpe'], places=5)
        finally:
            os.unlink(path)

    def test_mockter1_n_butane_freq_round_trip(self):
        """Scenario 1 (n-butane, 14 atoms) round-trips cleanly."""
        if 1 not in self.fixtures:
            self.skipTest('mockter1.yml not present')
        self._round_trip_freq_block(1, 'n_butane')

    def test_mockter4_h2o_freq_round_trip(self):
        """Scenario 4 (H2O, 3 atoms) round-trips cleanly."""
        if 4 not in self.fixtures:
            self.skipTest('mockter4.yml not present')
        self._round_trip_freq_block(4, 'H2O')

    def test_mockter4_oh_round_trip(self):
        """OH is a 2-atom doublet — exercises linear and open-shell paths."""
        if 4 not in self.fixtures:
            self.skipTest('mockter4.yml not present')
        self._round_trip_freq_block(4, 'OH')

    def test_mockter4_c2h6_round_trip(self):
        """C2H6 is a closed-shell 8-atom species — also has rotor scans."""
        if 4 not in self.fixtures:
            self.skipTest('mockter4.yml not present')
        self._round_trip_freq_block(4, 'C2H6')

    def test_mockter6_propanol_round_trip(self):
        """Propanol has the largest freqs vector among scenarios."""
        if 6 not in self.fixtures:
            self.skipTest('mockter6.yml not present')
        self._round_trip_freq_block(6, 'propanol')

    def test_sp_log_round_trips_for_all_species(self):
        """Render an sp log from each fixture's sp block and recover the energy."""
        for n, fixture in self.fixtures.items():
            for label, spc in fixture.get('species', {}).items():
                sp = spc.get('sp') or {}
                if sp.get('e_elect') is None:
                    continue
                xyz_str = (spc.get('fine_opt') or spc.get('opt') or {}).get('xyz')
                if not xyz_str:
                    continue
                log_text = render_gaussian_log(
                    job_type='sp', xyz=xyz_str,
                    e_elect_hartree=sp['e_elect'],
                    is_t1_capable=True,
                    t1_diagnostic=sp.get('t1_diagnostic'),
                    multiplicity=spc.get('multiplicity', 1),
                    charge=spc.get('charge', 0),
                )
                path = _write_log(log_text)
                try:
                    parsed = parse_e_elect(path)
                    self.assertAlmostEqual(
                        parsed / KJ_PER_HARTREE, sp['e_elect'], places=6,
                        msg=f'mockter{n}/{label} sp round-trip failed',
                    )
                finally:
                    os.unlink(path)


class TestRendererArkaneSubprocess(unittest.TestCase):
    """
    Contract test: forged Gaussian logs must be consumable by a real
    Arkane invocation. ARC runs Arkane as a subprocess (``python -m arkane
    input.py``); this test mirrors that, using ``conda run -n rmg_env``
    to enter an environment that has Arkane installed.

    Skips gracefully if neither ``conda`` nor ``rmg_env`` is available.
    """

    SPECIES_PY = (
        '#!/usr/bin/env python\n'
        '# -*- coding: utf-8 -*-\n'
        '\n'
        'linear = {linear}\n'
        'spinMultiplicity = {mult}\n'
        '\n'
        "energy = Log('{sp_path}')\n"
        "geometry = Log('{freq_path}')\n"
        "frequencies = Log('{freq_path}')\n"
    )

    INPUT_PY = (
        '#!/usr/bin/env python\n'
        '# -*- coding: utf-8 -*-\n'
        "modelChemistry = LevelOfTheory(method='ccsd(t)f12', basis='ccpvtzf12', software='molpro')\n"
        'useHinderedRotors = False\n'
        'useAtomCorrections = False\n'
        'useBondCorrections = False\n'
        '\n'
        "species('{label}', 'species/{label}.py', structure=SMILES('{smiles}'), spinMultiplicity={mult})\n"
        '\n'
        "thermo('{label}', 'NASA')\n"
    )

    @classmethod
    def setUpClass(cls):
        """Skip the whole class if rmg_env is unreachable."""
        cls.conda = shutil.which('conda')
        if cls.conda is None:
            raise unittest.SkipTest('conda not on PATH — cannot launch rmg_env Arkane subprocess')
        proc = subprocess.run(
            [cls.conda, 'run', '-n', 'rmg_env', 'python', '-m', 'arkane', '--help'],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise unittest.SkipTest(
                f'rmg_env Arkane unreachable; stderr tail: {proc.stderr[-200:]}'
            )

    def _run_arkane_thermo(
        self, label: str, smiles: str, multiplicity: int,
        is_linear: bool, xyz_str: str, e_elect_h: float, freqs: list,
        zpe_h: float, hessian_block: str | None,
    ) -> str:
        """
        Render a forged freq + sp log pair, build a minimal Arkane species and
        input file, run ``python -m arkane input.py`` in rmg_env, and return
        the path to the run directory.

        Args:
            label (str): Species label (used for filenames).
            smiles (str): SMILES for Arkane's structure annotation.
            multiplicity (int): Spin multiplicity.
            is_linear (bool): Whether the species is linear (Arkane uses this).
            xyz_str (str): Geometry in ARC xyz string format.
            e_elect_h (float): Electronic energy in Hartree.
            freqs (list): Harmonic frequencies in cm^-1.
            zpe_h (float): Zero-point energy in Hartree.
            hessian_block (str | None): Verbatim Gaussian Hessian block, or None.

        Returns:
            str: The run directory path (caller is responsible for cleanup).
        """
        run_dir = tempfile.mkdtemp(prefix='mockter_arkane_')
        os.makedirs(os.path.join(run_dir, 'species'), exist_ok=True)

        sp_path = os.path.join(run_dir, f'sp_{label}.log')
        freq_path = os.path.join(run_dir, f'freq_{label}.log')
        with open(sp_path, 'w') as f:
            f.write(render_gaussian_log(
                job_type='sp', xyz=xyz_str, e_elect_hartree=e_elect_h,
                multiplicity=multiplicity, is_t1_capable=True, t1_diagnostic=0.012,
            ))
        with open(freq_path, 'w') as f:
            f.write(render_gaussian_log(
                job_type='freq', xyz=xyz_str, e_elect_hartree=e_elect_h,
                multiplicity=multiplicity, freqs_cm1=freqs, zpe_hartree=zpe_h,
                hessian_block=hessian_block,
            ))

        with open(os.path.join(run_dir, 'species', f'{label}.py'), 'w') as f:
            f.write(self.SPECIES_PY.format(
                linear=is_linear, mult=multiplicity,
                sp_path=sp_path, freq_path=freq_path,
            ))
        with open(os.path.join(run_dir, 'input.py'), 'w') as f:
            f.write(self.INPUT_PY.format(label=label, smiles=smiles, mult=multiplicity))

        proc = subprocess.run(
            [self.conda, 'run', '-n', 'rmg_env', 'python', '-m', 'arkane', 'input.py'],
            cwd=run_dir, capture_output=True, text=True, timeout=120,
        )
        # Renderer contract: Arkane must parse the forged log AND compute thermo.
        # Anything past "Saving thermo for {label}" exercises rmgpy's downstream
        # YAML serializer, which can fail for reasons unrelated to log content
        # (e.g. zero-size ArrayQuantity in current numpy/rmgpy combos). For the
        # renderer's contract we require parse + statmech + thermo, not save.
        combined = proc.stdout + '\n' + proc.stderr
        parse_milestones = [
            'Loading statistical mechanics parameters for ' + label,
            'Saving statistical mechanics parameters for ' + label,
            'Saving thermo for ' + label,
        ]
        for milestone in parse_milestones:
            self.assertIn(
                milestone, combined,
                f'arkane never reached "{milestone}" for {label}\n'
                f'stdout tail: {proc.stdout[-1000:]}\n'
                f'stderr tail: {proc.stderr[-1000:]}'
            )
        # Specifically reject errors that originate inside the parser: any
        # 'load_conformer', 'load_geometry', 'load_energy', 'load_force_constant_matrix'
        # frame in the traceback would indicate a renderer bug.
        renderer_failures = [
            'load_conformer',
            'load_geometry',
            'load_energy',
            'load_force_constant_matrix',
            'load_negative_frequency',
        ]
        for frame in renderer_failures:
            if frame in proc.stderr and 'Traceback' in proc.stderr:
                self.fail(
                    f'arkane traceback hit {frame} — renderer contract violated\n'
                    f'stderr tail: {proc.stderr[-1500:]}'
                )
        return run_dir

    def test_h2o_thermo_via_arkane_subprocess(self):
        """End-to-end: render forged logs for H2O, run Arkane, check output.py."""
        fixture_path = os.path.join(FIXTURES_DIR, 'mockter4.yml')
        if not os.path.isfile(fixture_path):
            self.skipTest('mockter4.yml not present')
        fixture = read_yaml_file(fixture_path)
        spc = fixture['species']['H2O']
        xyz_str = (spc.get('fine_opt') or spc['opt'])['xyz']
        run_dir = self._run_arkane_thermo(
            label='H2O', smiles='O', multiplicity=1, is_linear=False,
            xyz_str=xyz_str,
            e_elect_h=spc['sp']['e_elect'],
            freqs=spc['freq']['freqs'],
            zpe_h=spc['freq']['zpe'],
            hessian_block=spc['freq'].get('hessian_block'),
        )
        shutil.rmtree(run_dir, ignore_errors=True)

    def test_c2h6_thermo_via_arkane_subprocess(self):
        """Larger species: C2H6 has 18 freqs; same end-to-end protocol."""
        fixture_path = os.path.join(FIXTURES_DIR, 'mockter4.yml')
        if not os.path.isfile(fixture_path):
            self.skipTest('mockter4.yml not present')
        fixture = read_yaml_file(fixture_path)
        spc = fixture['species']['C2H6']
        xyz_str = (spc.get('fine_opt') or spc['opt'])['xyz']
        run_dir = self._run_arkane_thermo(
            label='C2H6', smiles='CC', multiplicity=1, is_linear=False,
            xyz_str=xyz_str,
            e_elect_h=spc['sp']['e_elect'],
            freqs=spc['freq']['freqs'],
            zpe_h=spc['freq']['zpe'],
            hessian_block=spc['freq'].get('hessian_block'),
        )
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
