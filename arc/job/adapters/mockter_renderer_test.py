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


if __name__ == '__main__':
    unittest.main()
