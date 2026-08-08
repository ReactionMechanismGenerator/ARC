"""
Tests for the arc.output module (consolidated output.yml writer).
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

from arc.common import ARC_PATH, read_yaml_file
from arc.constants import E_h_kJmol
from arc.level import Level
from arc.common import ARC_TESTING_PATH
from arc.output import (
    EnergyCorrections,
    _build_energy_corrections_for_species,
    _build_rotor_scans,
    _build_scan_result_for_rotor,
    _compute_point_groups,
    _compute_species_corrections,
    _evidence_status_counts,
    _get_arkane_git_commit,
    _get_energy_corrections,
    _get_ess_software,
    _get_ess_versions,
    _get_rejected_torsions,
    _get_rotor_barrier,
    _get_torsions,
    _get_imaginary_freqs,
    _flat_parameter_values,
    _level_to_dict,
    _make_rel_path,
    _parse_opt_log,
    _parse_spin_diagnostic,
    _parse_zpe,
    _resolve_freq_scale_factor_source,
    _rxn_to_dict,
    _spc_to_dict,
    _statmech_to_dict,
    _thermo_to_dict,
    write_output_yml,
)
from arc.species.species import TSGuess, ThermoData


class TestLevelToDict(unittest.TestCase):
    """Tests for _level_to_dict."""

    def test_none_input(self):
        self.assertIsNone(_level_to_dict(None))

    def test_level_object(self):
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        result = _level_to_dict(level)
        self.assertEqual(result['method'], 'wb97xd')
        self.assertEqual(result['basis'], 'def2tzvp')
        self.assertEqual(result['software'], 'gaussian')

    def test_level_no_basis(self):
        level = Level(method='cbs-qb3')
        result = _level_to_dict(level)
        self.assertEqual(result['method'], 'cbs-qb3')
        self.assertNotIn('basis', result)  # as_dict omits None fields

    def test_level_with_solvent(self):
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian',
                      solvent='water', solvation_method='smd')
        result = _level_to_dict(level)
        self.assertEqual(result['solvent'], 'water')
        self.assertEqual(result['solvation_method'], 'smd')

    def test_solvation_scheme_level_is_yaml_safe(self):
        """A nested solvation scheme level is emitted as a plain dict that ``yaml.safe_load`` reads."""
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian',
                      solvent='water', solvation_method='smd',
                      solvation_scheme_level=Level(method='b3lyp', basis='6-31g'))
        result = _level_to_dict(level)
        self.assertIsInstance(result['solvation_scheme_level'], dict)
        self.assertEqual(result['solvation_scheme_level']['method'], 'b3lyp')
        round_tripped = yaml.safe_load(yaml.dump({'level': result}))
        self.assertEqual(round_tripped['level']['solvation_scheme_level']['basis'], '6-31g')


class TestMakeRelPath(unittest.TestCase):
    """Tests for _make_rel_path."""

    def test_none_input(self):
        self.assertIsNone(_make_rel_path(None, '/some/dir'))

    def test_empty_string(self):
        self.assertIsNone(_make_rel_path('', '/some/dir'))

    def test_absolute_to_relative(self):
        result = _make_rel_path('/home/user/project/calcs/sp.log', '/home/user/project')
        self.assertEqual(result, 'calcs/sp.log')

    def test_same_dir(self):
        result = _make_rel_path('/home/user/project/file.log', '/home/user/project')
        self.assertEqual(result, 'file.log')


class TestResolveFreqScaleFactorSource(unittest.TestCase):
    """Tests for _resolve_freq_scale_factor_source."""

    def test_none_level(self):
        self.assertIsNone(_resolve_freq_scale_factor_source(None))

    def test_missing_level(self):
        """A level not in the YAML should return None."""
        level = Level(method='totally_fake_method', basis='fake_basis')
        self.assertIsNone(_resolve_freq_scale_factor_source(level))

    def test_known_level_returns_citation(self):
        """wb97xd/def2tzvp should resolve to [4] citation."""
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        result = _resolve_freq_scale_factor_source(level)
        self.assertIsNotNone(result)
        self.assertIn('10.1021/ct100326h', result)


class TestParseThermoDataBlock(unittest.TestCase):
    """Tests for parse_thermo_data_block in arkane.py."""

    def test_parses_all_fields(self):
        from arc.statmech.arkane import parse_thermo_data_block
        block = """
            Tdata = ([300, 400, 500, 600, 800, 1000, 1500], 'K'),
            Cpdata = ([35.5, 39.0, 43.2, 47.5, 55.1, 61.1, 70.7], 'J/(mol*K)'),
            H298 = (-108.9, 'kJ/mol'),
            S298 = (218.4, 'J/(mol*K)'),
            Tmin = (10.0, 'K'),
            Tmax = (3000.0, 'K'),
            Cp0 = (33.3, 'J/(mol*K)'),
            CpInf = (83.3, 'J/(mol*K)'),
        """
        result = parse_thermo_data_block(block)
        self.assertAlmostEqual(result['H298'], -108.9)
        self.assertAlmostEqual(result['S298'], 218.4)
        self.assertAlmostEqual(result['Tmin'], 10.0)
        self.assertAlmostEqual(result['Tmax'], 3000.0)
        self.assertIsInstance(result['Tmin'], float)
        self.assertIsInstance(result['Tmax'], float)
        self.assertEqual(len(result['Tdata']), 7)
        self.assertEqual(len(result['Cpdata']), 7)

    def test_handles_missing_fields(self):
        from arc.statmech.arkane import parse_thermo_data_block
        block = "H298 = (-50.0, 'kJ/mol')"
        result = parse_thermo_data_block(block)
        self.assertAlmostEqual(result['H298'], -50.0)
        self.assertNotIn('Tmin', result)
        self.assertNotIn('Cpdata', result)

    def test_handles_scientific_notation(self):
        from arc.statmech.arkane import parse_thermo_data_block
        block = "H298 = (-1.089e+02, 'kJ/mol'), S298 = (2.184e+02, 'J/(mol*K)')"
        result = parse_thermo_data_block(block)
        self.assertAlmostEqual(result['H298'], -108.9, places=1)
        self.assertAlmostEqual(result['S298'], 218.4, places=1)


class TestGetArkaneGitCommit(unittest.TestCase):
    """Tests for _get_arkane_git_commit."""

    @patch('arc.output.settings', {'RMG_PATH': '/fake/RMG-Py'})
    @patch('arc.output.get_git_commit', return_value=('abc1234', '2026-01-01'))
    def test_returns_hash(self, mock_git):
        result = _get_arkane_git_commit()
        self.assertEqual(result, 'abc1234')

    @patch('arc.output.settings', {'RMG_PATH': '/fake/RMG-Py'})
    @patch('arc.output.get_git_commit', side_effect=Exception('no repo'))
    def test_returns_none_on_error(self, mock_git):
        self.assertIsNone(_get_arkane_git_commit())

    @patch('arc.output.settings', {'RMG_PATH': '/fake/RMG-Py'})
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_returns_none_for_empty(self, mock_git):
        self.assertIsNone(_get_arkane_git_commit())

    @patch('arc.output.settings', {})
    def test_returns_none_no_rmg_path(self):
        self.assertIsNone(_get_arkane_git_commit())


class TestThermoToDict(unittest.TestCase):
    """Tests for _thermo_to_dict."""

    def test_basic_thermo(self):
        thermo = ThermoData(H298=-50.2, S298=230.1, Tmin=(300, 'K'), Tmax=(3000, 'K'))
        result = _thermo_to_dict(thermo)
        self.assertEqual(result['h298_kj_mol'], -50.2)
        self.assertEqual(result['s298_j_mol_k'], 230.1)
        self.assertEqual(result['tmin_k'], 300)
        self.assertEqual(result['tmax_k'], 3000)
        self.assertIsNone(result['thermo_points'])
        self.assertIsNone(result['nasa_low'])
        self.assertIsNone(result['nasa_high'])

    def test_thermo_with_nasa(self):
        nasa_low = {'tmin_k': 300.0, 'tmax_k': 1000.0, 'coeffs': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
        nasa_high = {'tmin_k': 1000.0, 'tmax_k': 3000.0, 'coeffs': [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]}
        thermo = ThermoData(H298=-50.2, S298=230.1, Tmin=(300, 'K'), Tmax=(3000, 'K'),
                            nasa_low=nasa_low, nasa_high=nasa_high)
        result = _thermo_to_dict(thermo)
        self.assertEqual(result['nasa_low'], nasa_low)
        self.assertEqual(result['nasa_high'], nasa_high)

    def test_thermo_with_cp_data(self):
        cp = [{'temperature_k': 300.0, 'cp_j_mol_k': 35.1}, {'temperature_k': 400.0, 'cp_j_mol_k': 40.5}]
        thermo = ThermoData(H298=-10.0, S298=200.0, Tmin=(300, 'K'), Tmax=(2000, 'K'), thermo_points=cp)
        result = _thermo_to_dict(thermo)
        self.assertEqual(result['thermo_points'], cp)

    def test_tmin_tmax_scalar(self):
        """Tmin/Tmax can be plain numbers (not tuples)."""
        thermo = ThermoData(H298=-10.0, S298=200.0, Tmin=300, Tmax=3000)
        result = _thermo_to_dict(thermo)
        self.assertEqual(result['tmin_k'], 300)
        self.assertEqual(result['tmax_k'], 3000)


class TestGetTsImagFreq(unittest.TestCase):
    """Tests for _get_imaginary_freqs falling back to the chosen TS guess."""

    def test_no_ts_guesses(self):
        spc = MagicMock()
        spc.freqs = None
        spc.chosen_ts = None
        spc.ts_guesses = []
        self.assertIsNone(_get_imaginary_freqs(spc))

    def test_valid_imag_freq(self):
        ts_guess = MagicMock()
        ts_guess.imaginary_freqs = [-1500.0, -200.0]
        spc = MagicMock()
        spc.freqs = None
        spc.chosen_ts = 0
        spc.ts_guesses = [ts_guess]
        self.assertEqual(_get_imaginary_freqs(spc), [-1500.0, -200.0])

    def test_chosen_ts_out_of_range(self):
        spc = MagicMock()
        spc.freqs = None
        spc.chosen_ts = 5
        spc.ts_guesses = [MagicMock()]
        self.assertIsNone(_get_imaginary_freqs(spc))


class TestStatmechToDict(unittest.TestCase):
    """Tests for _statmech_to_dict."""

    def _make_spc(self, is_ts=False, is_linear=False, freqs=None):
        spc = MagicMock()
        spc.is_ts = is_ts
        spc._is_linear = is_linear
        spc.is_monoatomic.return_value = False
        spc.e0 = 100.5
        spc.multiplicity = 1
        spc.optical_isomers = 1
        spc.external_symmetry = 2
        spc.freqs = freqs
        spc.rotors_dict = None
        return spc

    def test_nonlinear_species(self):
        spc = self._make_spc(freqs=[100.0, 200.0, 300.0])
        result = _statmech_to_dict(spc, '/tmp/project')
        self.assertEqual(result['rigid_rotor_kind'], 'asymmetric_top')
        self.assertFalse(result['is_linear'])
        self.assertEqual(result['harmonic_frequencies_cm1'], [100.0, 200.0, 300.0])
        self.assertEqual(result['spin_multiplicity'], 1)
        self.assertEqual(result['external_symmetry'], 2)
        self.assertIsNone(result['point_group'])

    def test_with_point_group(self):
        spc = self._make_spc(freqs=[100.0])
        result = _statmech_to_dict(spc, '/tmp/project', point_group='C2v')
        self.assertEqual(result['point_group'], 'C2v')

    def test_linear_species(self):
        spc = self._make_spc(is_linear=True, freqs=[500.0, 600.0])
        result = _statmech_to_dict(spc, '/tmp/project')
        self.assertEqual(result['rigid_rotor_kind'], 'linear')
        self.assertTrue(result['is_linear'])

    def test_ts_filters_imaginary(self):
        spc = self._make_spc(is_ts=True, freqs=[-1500.0, 100.0, 200.0])
        result = _statmech_to_dict(spc, '/tmp/project')
        self.assertEqual(result['harmonic_frequencies_cm1'], [100.0, 200.0])

    def test_no_freqs(self):
        spc = self._make_spc(freqs=None)
        result = _statmech_to_dict(spc, '/tmp/project')
        self.assertIsNone(result['harmonic_frequencies_cm1'])

    def test_empty_torsions(self):
        spc = self._make_spc()
        result = _statmech_to_dict(spc, '/tmp/project')
        self.assertEqual(result['torsions'], [])

    def test_empty_rejected_torsions(self):
        spc = self._make_spc()
        result = _statmech_to_dict(spc, '/tmp/project')
        self.assertEqual(result['rejected_torsions'], [])

    def test_mixed_rotors_split_between_torsions_and_rejected_torsions(self):
        spc = self._make_spc()
        spc.rotors_dict = {
            0: {'success': True, 'scan': [1, 2, 3, 4], 'pivots': [2, 3], 'symmetry': 3,
                'type': 'HinderedRotor', 'scan_path': ''},
            1: {'success': False, 'scan': [2, 3, 4, 5], 'pivots': [3, 4],
                'invalidation_reason': 'rotor set too many (5) times'},
        }
        result = _statmech_to_dict(spc, '/tmp/project')
        self.assertEqual(len(result['torsions']), 1)
        self.assertEqual(len(result['rejected_torsions']), 1)
        self.assertEqual(result['rejected_torsions'][0]['rotor_index'], 1)
        self.assertEqual(result['rejected_torsions'][0]['invalidation_reason'],
                         'rotor set too many (5) times')


class TestGetTorsions(unittest.TestCase):
    """Tests for _get_torsions."""

    def test_no_rotors_dict(self):
        spc = MagicMock()
        spc.rotors_dict = None
        self.assertEqual(_get_torsions(spc, '/tmp'), [])

    def test_empty_rotors_dict(self):
        spc = MagicMock()
        spc.rotors_dict = {}
        self.assertEqual(_get_torsions(spc, '/tmp'), [])

    def test_failed_rotor_skipped(self):
        spc = MagicMock()
        spc.rotors_dict = {0: {'success': False, 'scan': [1, 2, 3, 4], 'pivots': [2, 3]}}
        self.assertEqual(_get_torsions(spc, '/tmp'), [])

    def test_successful_rotor(self):
        spc = MagicMock()
        spc.rotors_dict = {
            0: {
                'success': True,
                'scan': [1, 2, 3, 4],
                'pivots': [2, 3],
                'symmetry': 3,
                'type': 'HinderedRotor',
                'scan_path': '',
            }
        }
        result = _get_torsions(spc, '/tmp')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['atom_indices'], [1, 2, 3, 4])
        self.assertEqual(result[0]['pivot_atoms'], [2, 3])
        self.assertEqual(result[0]['symmetry_number'], 3)
        self.assertEqual(result[0]['treatment'], 'hindered_rotor')
        self.assertIsNone(result[0]['barrier_kj_mol'])

    def test_free_rotor(self):
        spc = MagicMock()
        spc.rotors_dict = {
            0: {
                'success': True,
                'scan': [1, 2, 3, 4],
                'pivots': [2, 3],
                'symmetry': 1,
                'type': 'FreeRotor',
                'scan_path': '',
            }
        }
        result = _get_torsions(spc, '/tmp')
        self.assertEqual(result[0]['treatment'], 'free_rotor')


class TestGetRejectedTorsions(unittest.TestCase):
    """Tests for _get_rejected_torsions."""

    def test_no_rotors_dict(self):
        spc = MagicMock()
        spc.rotors_dict = None
        self.assertEqual(_get_rejected_torsions(spc, '/tmp/project'), [])

    def test_empty_rotors_dict(self):
        spc = MagicMock()
        spc.rotors_dict = {}
        self.assertEqual(_get_rejected_torsions(spc, '/tmp/project'), [])

    def test_successful_rotor_excluded(self):
        spc = MagicMock()
        spc.rotors_dict = {
            0: {'success': True, 'scan': [1, 2, 3, 4], 'pivots': [2, 3],
                'invalidation_reason': ''},
        }
        self.assertEqual(_get_rejected_torsions(spc, '/tmp/project'), [])

    def test_rejected_rotor_with_reason(self):
        spc = MagicMock()
        spc.rotors_dict = {
            0: {'success': False, 'scan': [1, 2, 3, 4], 'pivots': [2, 3],
                'invalidation_reason': 'rotor set too many (5) times'},
        }
        result = _get_rejected_torsions(spc, '/tmp/project')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['rotor_index'], 0)
        self.assertEqual(result[0]['invalidation_reason'], 'rotor set too many (5) times')
        self.assertEqual(result[0]['atom_indices'], [1, 2, 3, 4])
        self.assertEqual(result[0]['pivot_atoms'], [2, 3])
        self.assertEqual(result[0]['dimension'], 1)
        # No 'scan_path' on the rotor dict -> no scan log -> no dangling reference.
        self.assertNotIn('source_log', result[0])

    def test_rejected_rotor_with_empty_invalidation_reason(self):
        """ARC's default is an empty string; it must be carried as-is, not fabricated."""
        spc = MagicMock()
        spc.rotors_dict = {
            0: {'success': False, 'scan': [1, 2, 3, 4], 'pivots': [2, 3],
                'invalidation_reason': ''},
        }
        result = _get_rejected_torsions(spc, '/tmp/project')
        self.assertEqual(result[0]['invalidation_reason'], '')

    def test_pending_rotor_excluded(self):
        """``success is None`` is pending (never attempted / mid-troubleshooting), not rejected.

        A species' convergence does not depend on its rotors completing
        (``job_types['rotors']`` is initialised to ``True`` and never set
        False), so an ordinary converged species can carry rotors that are
        still ``None``. Those must not be published as reason-less
        rejections -- they simply are not decided yet, exactly like
        ``torsions`` already omits them.
        """
        spc = MagicMock()
        spc.rotors_dict = {
            0: {'scan': [1, 2, 3, 4], 'pivots': [2, 3]},  # no 'success' key -> .get() is None
            1: {'success': None, 'scan': [2, 3, 4, 5], 'pivots': [3, 4]},
        }
        result = _get_rejected_torsions(spc, '/tmp/project')
        self.assertEqual(result, [])

    def test_mix_of_successful_pending_and_rejected_rotors(self):
        spc = MagicMock()
        spc.rotors_dict = {
            0: {'success': True, 'scan': [1, 2, 3, 4], 'pivots': [2, 3],
                'invalidation_reason': ''},
            1: {'success': False, 'scan': [2, 3, 4, 5], 'pivots': [3, 4],
                'invalidation_reason': 'rotor set too many (5) times'},
            2: {'success': False, 'scan': [3, 4, 5, 6], 'pivots': [4, 5],
                'invalidation_reason': ''},
            3: {'success': None, 'scan': [4, 5, 6, 7], 'pivots': [5, 6]},
        }
        result = _get_rejected_torsions(spc, '/tmp/project')
        self.assertEqual({entry['rotor_index'] for entry in result}, {1, 2})

    def test_only_rejected_rotors(self):
        """A species where ARC found rotors but rejected every one of them."""
        spc = MagicMock()
        spc.rotors_dict = {
            0: {'success': False, 'scan': [1, 2, 3, 4], 'pivots': [2, 3],
                'invalidation_reason': 'rotor set too many (5) times'},
            1: {'success': False, 'scan': [2, 3, 4, 5], 'pivots': [3, 4],
                'invalidation_reason': 'rotor set too many (3) times'},
        }
        result = _get_rejected_torsions(spc, '/tmp/project')
        self.assertEqual(len(result), 2)
        self.assertEqual({entry['rotor_index'] for entry in result}, {0, 1})


class TestGetRotorBarrier(unittest.TestCase):
    """Tests for _get_rotor_barrier."""

    def test_no_scan_path(self):
        self.assertIsNone(_get_rotor_barrier({}, '/tmp'))
        self.assertIsNone(_get_rotor_barrier({'scan_path': ''}, '/tmp'))

    def test_missing_file(self):
        self.assertIsNone(_get_rotor_barrier({'scan_path': '/nonexistent/file.log'}, '/tmp'))

    @patch('arc.output.parse_1d_scan_energies', return_value=([0.0, 5.2, 10.1, 3.3], [0, 90, 180, 270]))
    def test_valid_barrier(self, mock_parse):
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            f.write(b'dummy scan data')
            tmp_path = f.name
        try:
            result = _get_rotor_barrier({'scan_path': tmp_path}, '/tmp')
            self.assertAlmostEqual(result, 10.1)
        finally:
            os.unlink(tmp_path)

    @patch('arc.output.parse_1d_scan_energies', side_effect=Exception('parse error'))
    def test_parse_failure(self, mock_parse):
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            f.write(b'dummy')
            tmp_path = f.name
        try:
            self.assertIsNone(_get_rotor_barrier({'scan_path': tmp_path}, '/tmp'))
        finally:
            os.unlink(tmp_path)


class TestParseOptLog(unittest.TestCase):
    """Tests for _parse_opt_log and the Gaussian parse_opt_steps adapter."""

    def test_gaussian_opt_log(self):
        """Parse a real Gaussian opt log for step count and final energy."""
        opt_path = os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')
        n_steps, e_hartree, final_xyz = _parse_opt_log(opt_path, '/dummy')
        self.assertEqual(n_steps, 4)
        self.assertIsNotNone(e_hartree)
        self.assertAlmostEqual(e_hartree, -116.986089069, places=6)
        self.assertIsNotNone(final_xyz)

    def test_missing_file(self):
        n_steps, e_hartree, final_xyz = _parse_opt_log('/nonexistent/file.log', '/tmp')
        self.assertIsNone(n_steps)
        self.assertIsNone(e_hartree)
        self.assertIsNone(final_xyz)

    def test_none_path(self):
        n_steps, e_hartree, final_xyz = _parse_opt_log(None, '/tmp')
        self.assertIsNone(n_steps)
        self.assertIsNone(e_hartree)
        self.assertIsNone(final_xyz)

    def test_parse_zpe_from_freq_log(self):
        """Parse ZPE from a real Gaussian freq log."""
        freq_path = os.path.join(ARC_TESTING_PATH, 'freq', 'iC3H7.out')
        zpe = _parse_zpe(freq_path, '/dummy')
        self.assertIsNotNone(zpe)
        self.assertAlmostEqual(zpe, 0.0945, places=3)  # ~0.0945 Hartree for iC3H7

    def test_gaussian_adapter_parse_opt_steps(self):
        """Test the Gaussian adapter method directly."""
        from arc.parser.adapters.gaussian import GaussianParser
        opt_path = os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')
        parser = GaussianParser(log_file_path=opt_path)
        self.assertEqual(parser.parse_opt_steps(), 4)

    def test_parse_opt_steps_via_make_parser(self):
        """Test the top-level parse_opt_steps function."""
        from arc.parser.parser import parse_opt_steps
        opt_path = os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')
        self.assertEqual(parse_opt_steps(opt_path), 4)


class TestParseSpinDiagnostic(unittest.TestCase):
    """Tests for _parse_spin_diagnostic (output.yml S**2 plumbing)."""

    def test_open_shell_gaussian_doublet(self):
        """Open-shell doublet: block with s_squared, expected (from mult), annihilated."""
        sp = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'Species', 'NH2_freq.out')
        sd = _parse_spin_diagnostic(sp, None, None, multiplicity=2, project_directory='/dummy')
        self.assertIsNotNone(sd)
        self.assertAlmostEqual(sd['s_squared'], 0.7535)
        self.assertAlmostEqual(sd['s_squared_expected'], 0.75)
        self.assertAlmostEqual(sd['s_squared_annihilated'], 0.75)

    def test_expected_recomputed_from_arc_multiplicity(self):
        """s_squared_expected is authoritative from ARC's multiplicity (triplet -> 2.0)."""
        sp = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'TSs', 'TS_freq.out')
        sd = _parse_spin_diagnostic(sp, None, None, multiplicity=3, project_directory='/dummy')
        self.assertIsNotNone(sd)
        self.assertAlmostEqual(sd['s_squared'], 2.0153)
        self.assertAlmostEqual(sd['s_squared_expected'], 2.0)

    def test_closed_shell_returns_none(self):
        """Restricted/closed-shell log (no <S**2>) -> None (block omitted)."""
        sp = os.path.join(ARC_TESTING_PATH, 'composite', 'C2H5NO2__C2H5ONO.out')
        self.assertIsNone(_parse_spin_diagnostic(sp, None, None, multiplicity=1, project_directory='/dummy'))

    def test_fallback_to_freq_when_sp_absent(self):
        """When the sp log is absent, falls back to the freq log."""
        freq = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'Species', 'NH2_freq.out')
        sd = _parse_spin_diagnostic(None, freq, None, multiplicity=2, project_directory='/dummy')
        self.assertIsNotNone(sd)
        self.assertAlmostEqual(sd['s_squared'], 0.7535)

    def test_no_paths_returns_none(self):
        self.assertIsNone(_parse_spin_diagnostic(None, None, None, multiplicity=2, project_directory='/dummy'))

    def test_orca_open_shell_no_annihilation_key(self):
        """ORCA: annihilated is None -> the key is omitted from the emitted block."""
        sp = os.path.join(ARC_TESTING_PATH, 'neb', 'neb_res.out')
        sd = _parse_spin_diagnostic(sp, None, None, multiplicity=2, project_directory='/dummy')
        self.assertIsNotNone(sd)
        self.assertNotIn('s_squared_annihilated', sd)
        self.assertAlmostEqual(sd['s_squared_expected'], 0.75)


class TestParseEssVersion(unittest.TestCase):
    """Tests for parse_ess_version across ESS adapters."""

    def test_gaussian(self):
        from arc.parser.parser import parse_ess_version
        path = os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')
        self.assertEqual(parse_ess_version(path), 'Gaussian 09, Revision D.01')

    def test_orca(self):
        from arc.parser.parser import parse_ess_version
        path = os.path.join(ARC_TESTING_PATH, 'freq', 'orca_neg_freq_ts.out')
        self.assertEqual(parse_ess_version(path), 'ORCA 5.0.4')

    def test_qchem(self):
        from arc.parser.parser import parse_ess_version
        path = os.path.join(ARC_TESTING_PATH, 'N2H4_opt_QChem.out')
        self.assertEqual(parse_ess_version(path), 'Q-Chem 4.4')

    def test_molpro(self):
        from arc.parser.parser import parse_ess_version
        path = os.path.join(ARC_TESTING_PATH, 'freq', 'CH2O_freq_molpro.out')
        self.assertEqual(parse_ess_version(path), 'Molpro 2015.1.37')


class TestGetEssVersions(unittest.TestCase):
    """Tests for _get_ess_versions."""

    def test_gaussian_log(self):
        paths = {'sp': os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')}
        result = _get_ess_versions(paths, '/dummy')
        self.assertIn('sp', result)
        self.assertIn('Gaussian 09', result['sp'])

    def test_shared_log_file_reports_all_job_types(self):
        """When sp and geo point to the same file, both job types should appear with the same version."""
        log = os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')
        paths = {'sp': log, 'geo': log}
        result = _get_ess_versions(paths, '/dummy')
        self.assertEqual(len(result), 2)
        self.assertIn('sp', result)
        self.assertIn('opt', result)
        self.assertEqual(result['sp'], result['opt'])

    def test_no_paths(self):
        self.assertIsNone(_get_ess_versions({}, '/dummy'))

    def test_missing_files(self):
        paths = {'sp': '/nonexistent.log', 'geo': '/also_missing.log'}
        self.assertIsNone(_get_ess_versions(paths, '/dummy'))


class TestGetEssSoftware(unittest.TestCase):
    """Tests for _get_ess_software."""

    gaussian_log = os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')
    orca_log = os.path.join(ARC_TESTING_PATH, 'orca_example_opt.log')

    def test_mixed_program_run(self):
        """A run whose opt and sp used different programs reports both, keyed by job type."""
        paths = {'geo': self.gaussian_log, 'sp': self.orca_log}
        self.assertEqual(_get_ess_software(paths, '/dummy'),
                         {'opt': 'gaussian', 'sp': 'orca'})

    def test_pairs_with_ess_versions_for_a_mixed_program_run(self):
        """Each version banner is paired with the program that actually produced it."""
        paths = {'geo': self.gaussian_log, 'sp': self.orca_log}
        software = _get_ess_software(paths, '/dummy')
        versions = _get_ess_versions(paths, '/dummy')
        self.assertEqual(software['opt'], 'gaussian')
        self.assertIn('Gaussian', versions['opt'])
        self.assertEqual(software['sp'], 'orca')
        self.assertIn('ORCA', versions['sp'])

    def test_key_set_is_a_superset_of_the_version_key_set(self):
        paths = {'geo': self.gaussian_log, 'sp': self.orca_log}
        software = _get_ess_software(paths, '/dummy')
        versions = _get_ess_versions(paths, '/dummy')
        self.assertLessEqual(set(versions), set(software))

    def test_shared_log_file_reports_all_job_types(self):
        paths = {'sp': self.gaussian_log, 'geo': self.gaussian_log}
        self.assertEqual(_get_ess_software(paths, '/dummy'),
                         {'sp': 'gaussian', 'opt': 'gaussian'})

    def test_relative_paths_are_resolved_against_the_project_directory(self):
        paths = {'geo': os.path.join('opt', 'iC3H7.out')}
        self.assertEqual(_get_ess_software(paths, ARC_TESTING_PATH), {'opt': 'gaussian'})

    def test_no_paths(self):
        self.assertIsNone(_get_ess_software({}, '/dummy'))

    def test_missing_files(self):
        paths = {'sp': '/nonexistent.log', 'geo': '/also_missing.log'}
        self.assertIsNone(_get_ess_software(paths, '/dummy'))


class TestGetEnergyCorrections(unittest.TestCase):
    """Tests for _get_energy_corrections."""

    def test_none_level(self):
        corrections = _get_energy_corrections(None, 'p')
        self.assertIsNone(corrections.aec)
        self.assertIsNone(corrections.bac)
        self.assertIsNone(corrections.aec_key)
        self.assertIsNone(corrections.bac_key)

    def test_known_level(self):
        lot = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        corrections = _get_energy_corrections(lot, 'p')
        if corrections.aec is not None:  # only if RMG-database is available
            self.assertIn('H', corrections.aec)
            self.assertIn('C', corrections.aec)
            self.assertIsInstance(corrections.aec['H'], float)
        if corrections.bac is not None:
            self.assertIn('C-H', corrections.bac)
            self.assertIsInstance(corrections.bac['C-H'], float)

    def test_no_bac_when_type_none(self):
        lot = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        corrections = _get_energy_corrections(lot, None)
        self.assertIsNone(corrections.bac)
        self.assertIsNone(corrections.bac_key)

    def test_independent_aec_and_bac_keys(self):
        """AEC and BAC keys should be resolved independently, not reusing the AEC key for BAC."""
        lot = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        aec_key = "LevelOfTheory(method='wb97xd',basis='def2tzvp',software='gaussian')"
        bac_key = "LevelOfTheory(method='wb97xd',basis='def2tzvp')"  # different key (no software)

        calls = []
        def mock_find_best(level, files, start, end):
            calls.append(start)
            if 'atom_energies' in start:
                return aec_key
            elif 'pbac' in start:
                return bac_key
            return None

        with patch('arc.output.find_best_across_files', side_effect=mock_find_best), \
             patch('arc.output.get_qm_corrections_files', return_value=['/fake/data.py']), \
             patch('arc.output.execute_command', return_value=('', '')), \
             patch('arc.output.read_yaml_file', return_value={'aec': {'H': -0.5}, 'bac': {'C-H': -0.06}}), \
             patch('arc.output.save_yaml_file') as mock_save:
            corrections = _get_energy_corrections(lot, 'p')

        self.assertTrue(any('atom_energies' in c for c in calls))
        self.assertTrue(any('pbac' in c for c in calls))
        save_call = mock_save.call_args
        saved_content = save_call[1].get('content') or save_call[0][1]
        self.assertEqual(saved_content['aec_key'], aec_key)
        self.assertEqual(saved_content['bac_key'], bac_key)
        self.assertIsNotNone(corrections.aec)
        self.assertIsNotNone(corrections.bac)
        self.assertEqual(corrections.aec_key, aec_key)
        self.assertEqual(corrections.bac_key, bac_key)

    def test_matched_keys_reported_when_only_the_aec_section_matches(self):
        """A missing BAC key must be reported as None rather than reusing the AEC key."""
        lot = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        aec_key = "LevelOfTheory(method='wb97xd',basis='def2tzvp',software='gaussian')"

        def mock_find_best(level, files, start, end):
            return aec_key if 'atom_energies' in start else None

        with patch('arc.output.find_best_across_files', side_effect=mock_find_best), \
             patch('arc.output.get_qm_corrections_files', return_value=['/fake/data.py']), \
             patch('arc.output.execute_command', return_value=('', '')), \
             patch('arc.output.read_yaml_file', return_value={'aec': {'H': -0.5}, 'bac': None}), \
             patch('arc.output.save_yaml_file'):
            corrections = _get_energy_corrections(lot, 'p')

        self.assertEqual(corrections.aec_key, aec_key)
        self.assertIsNone(corrections.bac_key)

    def test_bac_table_looked_up_when_only_the_bac_section_matches(self):
        """A BAC-section match with no atom-energy match still yields the BAC table and its key."""
        lot = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        bac_key = "LevelOfTheory(method='wb97xd',basis='def2tzvp')"

        def mock_find_best(level, files, start, end):
            return bac_key if 'pbac' in start else None

        with patch('arc.output.find_best_across_files', side_effect=mock_find_best), \
             patch('arc.output.get_qm_corrections_files', return_value=['/fake/data.py']), \
             patch('arc.output.execute_command', return_value=('', '')) as mock_exec, \
             patch('arc.output.read_yaml_file', return_value={'aec': None, 'bac': {'C-H': -0.06}}), \
             patch('arc.output.save_yaml_file') as mock_save:
            corrections = _get_energy_corrections(lot, 'p')

        mock_exec.assert_called_once()
        saved_content = mock_save.call_args[1].get('content') or mock_save.call_args[0][1]
        self.assertIsNone(saved_content['aec_key'])
        self.assertEqual(saved_content['bac_key'], bac_key)
        self.assertIsNone(corrections.aec)
        self.assertIsNone(corrections.aec_key)
        self.assertEqual(corrections.bac, {'C-H': -0.06})
        self.assertEqual(corrections.bac_key, bac_key)

    def test_aec_table_lookup_failure_preserves_the_matched_keys(self):
        """The subprocess failing loses the tables, not the provenance that was observed."""
        lot = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        aec_key = "LevelOfTheory(method='wb97xd',basis='def2tzvp',software='gaussian')"
        bac_key = "LevelOfTheory(method='wb97xd',basis='def2tzvp')"

        def mock_find_best(level, files, start, end):
            return aec_key if 'atom_energies' in start else bac_key

        with patch('arc.output.find_best_across_files', side_effect=mock_find_best), \
             patch('arc.output.get_qm_corrections_files', return_value=['/fake/data.py']), \
             patch('arc.output.execute_command', side_effect=RuntimeError('boom')), \
             patch('arc.output.save_yaml_file'):
            corrections = _get_energy_corrections(lot, 'p')

        self.assertIsNone(corrections.aec)
        self.assertIsNone(corrections.bac)
        self.assertEqual(corrections.aec_key, aec_key)
        self.assertEqual(corrections.bac_key, bac_key)


class TestGetTsImagFreqFromFreqs(unittest.TestCase):
    """Tests for _get_imaginary_freqs using spc.freqs as the primary source."""

    def test_from_spc_freqs(self):
        spc = MagicMock()
        spc.freqs = [-1500.0, 100.0, 200.0, 300.0]
        spc.chosen_ts = None
        spc.ts_guesses = []
        self.assertEqual(_get_imaginary_freqs(spc), [-1500.0])

    def test_most_negative_first(self):
        spc = MagicMock()
        spc.freqs = [-200.0, -1500.0, 100.0, 300.0]
        spc.chosen_ts = None
        spc.ts_guesses = []
        self.assertEqual(_get_imaginary_freqs(spc), [-1500.0, -200.0])

    def test_no_negative_freqs(self):
        spc = MagicMock()
        spc.freqs = [100.0, 200.0, 300.0]
        spc.chosen_ts = None
        spc.ts_guesses = []
        self.assertEqual(_get_imaginary_freqs(spc), [])

    def test_a_spurious_small_mode_is_reported_alongside_the_major_one(self):
        spc = MagicMock()
        spc.freqs = [-1379.0, -42.0, 120.0, 800.0]
        spc.chosen_ts = None
        spc.ts_guesses = []
        self.assertEqual(_get_imaginary_freqs(spc), [-1379.0, -42.0])

    def test_an_imaginary_mode_on_a_stable_species_is_reported(self):
        spc = MagicMock()
        spc.freqs = [-31.0, 120.0, 800.0]
        spc.chosen_ts = None
        spc.ts_guesses = []
        self.assertEqual(_get_imaginary_freqs(spc), [-31.0])


class TestParseConformerStatmech(unittest.TestCase):
    """Tests for _parse_conformer_statmech in arkane.py."""

    def test_parses_symmetry_and_optical_isomers(self):
        from arc.statmech.arkane import _parse_conformer_statmech
        content = """
conformer(
    label = 'CH2O',
    E0 = (-118.911, 'kJ/mol'),
    modes = [
        NonlinearRotor(
            inertia = ([1.0, 2.0, 3.0], 'amu*angstrom^2'),
            symmetry = 2,
        ),
        HarmonicOscillator(frequencies=([1200.0, 1500.0], 'cm^-1')),
    ],
    spin_multiplicity = 1,
    optical_isomers = 1,
)
"""
        spc = MagicMock()
        spc.label = 'CH2O'
        spc.optical_isomers = None
        spc.external_symmetry = None
        _parse_conformer_statmech(spc, content)
        self.assertEqual(spc.optical_isomers, 1)
        self.assertEqual(spc.external_symmetry, 2)

    def test_does_not_overwrite_existing(self):
        from arc.statmech.arkane import _parse_conformer_statmech
        content = "conformer(label='X', E0=(0,'kJ/mol'), modes=[NonlinearRotor(symmetry=4)], optical_isomers=2)"
        spc = MagicMock()
        spc.label = 'X'
        spc.optical_isomers = 1  # already set
        spc.external_symmetry = 12  # already set
        _parse_conformer_statmech(spc, content)
        self.assertEqual(spc.optical_isomers, 1)
        self.assertEqual(spc.external_symmetry, 12)


class TestKineticsCommentParsing(unittest.TestCase):
    """Tests for dA/dn/dEa parsing from Arkane kinetics comment."""

    def test_parse_uncertainties(self):
        from arc.statmech.arkane import parse_reaction_kinetics
        content = """
kinetics(
    label = 'A + B <=> C + D',
    kinetics = Arrhenius(
        A = (1.2e10, 'cm^3/(mol*s)'),
        n = 2.5,
        Ea = (45.6, 'kJ/mol'),
        T0 = (1, 'K'),
        Tmin = (300, 'K'),
        Tmax = (3000, 'K'),
        comment = 'Fitted to 50 data points; dA = *|/ 1.48, dn = +|- 0.05, dEa = +|- 0.29 kJ/mol',
    ),
)
"""
        rxn = MagicMock()
        rxn.label = 'A + B <=> C + D'
        rxn.ts_species = MagicMock()
        rxn.ts_species.label = 'TS0'
        rxn.ts_species.e0 = None
        parse_reaction_kinetics(rxn, content)
        self.assertAlmostEqual(rxn.kinetics['dA'], 1.48)
        self.assertAlmostEqual(rxn.kinetics['dn'], 0.05)
        self.assertAlmostEqual(rxn.kinetics['dEa'], 0.29)
        self.assertEqual(rxn.kinetics['dEa_units'], 'kJ/mol')
        self.assertEqual(rxn.kinetics['n_data_points'], 50)

    def test_rxn_to_dict_with_uncertainties(self):
        rxn = MagicMock()
        rxn.label = 'A <=> B'
        rxn.reactants = ['A']
        rxn.products = ['B']
        rxn.family = 'intra_H_migration'
        rxn.multiplicity = 1
        rxn.ts_label = 'TS0'
        rxn.kinetics = {
            'A': (1.2e10, 's^-1'), 'n': 2.5, 'Ea': (45.6, 'kJ/mol'),
            'Tmin': (300, 'K'), 'Tmax': (3000, 'K'),
            'dA': 1.48, 'dn': 0.05, 'dEa': 0.29, 'dEa_units': 'kJ/mol',
            'n_data_points': 50,
        }
        result = _rxn_to_dict(rxn)
        self.assertAlmostEqual(result['kinetics']['dA'], 1.48)
        self.assertAlmostEqual(result['kinetics']['dn'], 0.05)
        self.assertAlmostEqual(result['kinetics']['dEa'], 0.29)
        self.assertEqual(result['kinetics']['dEa_units'], 'kJ/mol')
        self.assertEqual(result['kinetics']['n_data_points'], 50)


class TestTsWithSmiles(unittest.TestCase):
    """Test that TS species get SMILES when mol is available."""

    def test_ts_smiles_null_formula_from_mol(self):
        """TS SMILES should be null; formula comes from spc.mol."""
        spc = MagicMock()
        spc.label = 'TS0'
        spc.original_label = None
        spc.charge = 0
        spc.multiplicity = 2
        spc.is_ts = True
        spc.mol = MagicMock()
        spc.mol.get_formula.return_value = 'C2H6O'
        spc.final_xyz = {'symbols': ('C',), 'isotopes': (12,), 'coords': ((0, 0, 0),)}
        spc.initial_xyz = None
        spc.is_monoatomic.return_value = False
        spc.e_elect = -100.0
        spc.e0 = -95.0
        spc._is_linear = False
        spc.optical_isomers = 1
        spc.external_symmetry = 1
        spc.freqs = [-1500.0, 100.0]
        spc.rotors_dict = None
        spc.thermo = None
        spc.rxn_label = 'CHO + CH4 <=> CH2O + CH3'
        spc.chosen_ts_method = 'heuristics'
        spc.successful_methods = ['heuristics']
        output_dict = {'TS0': {'convergence': True, 'paths': {'irc': []}, 'job_types': {'opt': True, 'irc': True}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNone(result['smiles'])
        self.assertEqual(result['formula'], 'C2H6O')

    def test_ts_without_mol(self):
        spc = MagicMock()
        spc.label = 'TS1'
        spc.original_label = None
        spc.charge = 0
        spc.multiplicity = 2
        spc.is_ts = True
        spc.mol = None
        spc.final_xyz = {'symbols': ('C',), 'isotopes': (12,), 'coords': ((0, 0, 0),)}
        spc.initial_xyz = None
        spc.is_monoatomic.return_value = False
        spc.e_elect = -100.0
        spc.e0 = -95.0
        spc._is_linear = False
        spc.optical_isomers = 1
        spc.external_symmetry = 1
        spc.freqs = [-1500.0, 100.0]
        spc.rotors_dict = None
        spc.thermo = None
        spc.rxn_label = 'A <=> B'
        spc.chosen_ts_method = None
        spc.successful_methods = []
        output_dict = {'TS1': {'convergence': True, 'paths': {'irc': []}, 'job_types': {}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNone(result['smiles'])
        self.assertIsNone(result['formula'])

    def test_ts_path_logs_and_irc_directions(self):
        spc = MagicMock()
        spc.label = 'TS_paths'
        spc.original_label = None
        spc.charge = 0
        spc.multiplicity = 1
        spc.is_ts = True
        spc.mol = None
        spc.final_xyz = {'symbols': ('C',), 'isotopes': (12,), 'coords': ((0, 0, 0),)}
        spc.initial_xyz = None
        spc.is_monoatomic.return_value = False
        spc.e_elect = None
        spc._is_linear = False
        spc.optical_isomers = 1
        spc.external_symmetry = 1
        spc.freqs = [-1000.0, 100.0]
        spc.rotors_dict = None
        spc.thermo = None
        spc.rxn_label = 'A <=> B'
        spc.chosen_ts_method = 'xTB-GSM'
        spc.successful_methods = ['xTB-GSM']
        spc.chosen_ts = None
        spc.ts_guesses = []
        output_dict = {'TS_paths': {
            'convergence': True,
            'paths': {
                'gsm': '/run/gsm/stringfile.xyz0000',
                'neb': '',
                'irc': ['/run/irc/forward.log', '/run/irc/reverse.log'],
                'irc_directions': ['forward', 'reverse'],
            },
            'job_types': {'irc': True},
        }}
        result = _spc_to_dict(spc, output_dict, '/run')
        self.assertEqual(result['gsm_log'], 'gsm/stringfile.xyz0000')
        self.assertIsNone(result['neb_log'])
        self.assertEqual(result['irc_log_directions'], ['forward', 'reverse'])

    def test_merged_ts_guess_recovers_path_artifact_by_index(self):
        spc = MagicMock()
        spc.label = 'TS_merged'
        spc.original_label = None
        spc.charge = 0
        spc.multiplicity = 1
        spc.is_ts = True
        spc.mol = None
        spc.final_xyz = {'symbols': ('C',), 'isotopes': (12,), 'coords': ((0, 0, 0),)}
        spc.initial_xyz = None
        spc.is_monoatomic.return_value = False
        spc.e_elect = None
        spc._is_linear = False
        spc.optical_isomers = 1
        spc.external_symmetry = 1
        spc.freqs = [-1000.0, 100.0]
        spc.rotors_dict = None
        spc.thermo = None
        spc.rxn_label = 'A <=> B'
        spc.chosen_ts_method = 'gcn'
        spc.successful_methods = ['gcn', 'xTB-GSM']
        guess = TSGuess(index=7, method='gcn', success=True, xyz='C 0 0 0')
        guess.method_sources = ['gcn', 'xtb-gsm']
        guess.method_source_paths = {'xtb-gsm': '/run/gsm/stringfile.xyz0000'}
        spc.chosen_ts = 7
        spc.ts_guesses = [guess]
        output_dict = {'TS_merged': {
            'convergence': True,
            'paths': {'gsm': '', 'neb': '', 'irc': []},
            'job_types': {},
        }}
        result = _spc_to_dict(spc, output_dict, '/run')
        self.assertEqual(result['gsm_log'], 'gsm/stringfile.xyz0000')
        self.assertIsNone(result['neb_log'])
        self.assertEqual(result['ts_guesses'], [{
            'index': 7,
            'chosen': True,
            'method': 'gcn',
            'method_sources': ['gcn', 'xtb-gsm'],
        }])


class TestEvidenceStatusCounts(unittest.TestCase):
    """Tests concise evidence-write diagnostics."""

    def test_counts_only_known_evidence_envelopes(self):
        evidence = {'records': [
            {'record_kind': 'species', 'label': 'A',
             'freq_hessian': {'status': 'available'}},
            {'record_kind': 'transition_state', 'label': 'TS0',
             'freq_hessian': {'status': 'unavailable'},
             'irc': {'status': 'available'},
             'gsm': {'status': 'unavailable'}},
            {'record_kind': 'species', 'label': 'B', 'other': {'status': 'available'}},
        ]}
        self.assertEqual(_evidence_status_counts(evidence), {
            'freq_hessian': {'available': 1, 'unavailable': 1},
            'irc': {'available': 1, 'unavailable': 0},
            'gsm': {'available': 0, 'unavailable': 1},
        })


class TestRxnToDict(unittest.TestCase):
    """Tests for _rxn_to_dict."""

    def test_no_kinetics(self):
        rxn = MagicMock()
        rxn.label = 'CH4 + OH <=> CH3 + H2O'
        rxn.reactants = ['CH4', 'OH']
        rxn.products = ['CH3', 'H2O']
        rxn.family = 'H_Abstraction'
        rxn.multiplicity = 2
        rxn.ts_label = 'TS0'
        rxn.kinetics = None
        result = _rxn_to_dict(rxn)
        self.assertEqual(result['label'], 'CH4 + OH <=> CH3 + H2O')
        self.assertEqual(result['reactant_labels'], ['CH4', 'OH'])
        self.assertEqual(result['product_labels'], ['CH3', 'H2O'])
        self.assertEqual(result['family'], 'H_Abstraction')
        self.assertEqual(result['ts_label'], 'TS0')
        self.assertIsNone(result['kinetics'])

    def test_with_kinetics(self):
        rxn = MagicMock()
        rxn.label = 'A <=> B'
        rxn.reactants = ['A']
        rxn.products = ['B']
        rxn.family = 'intra_H_migration'
        rxn.multiplicity = 1
        rxn.ts_label = 'TS1'
        rxn.kinetics = {
            'A': (1.2e10, 's^-1'),
            'n': 0.5,
            'Ea': (45.6, 'kJ/mol'),
            'Tmin': (300, 'K'),
            'Tmax': (2000, 'K'),
            'dA': None,
            'dn': None,
            'dEa': None,
        }
        result = _rxn_to_dict(rxn)
        self.assertAlmostEqual(result['kinetics']['A'], 1.2e10)
        self.assertEqual(result['kinetics']['A_units'], 's^-1')
        self.assertAlmostEqual(result['kinetics']['Ea'], 45.6)
        self.assertEqual(result['kinetics']['Ea_units'], 'kJ/mol')
        self.assertEqual(result['kinetics']['n'], 0.5)


class TestSpcToDict(unittest.TestCase):
    """Tests for _spc_to_dict."""

    def _make_spc_mock(self, label='CH4', is_ts=False, converged=True, monoatomic=False):
        spc = MagicMock()
        spc.label = label
        spc.original_label = label
        spc.charge = 0
        spc.multiplicity = 1
        spc.is_ts = is_ts
        spc.mol = MagicMock() if not is_ts else None
        if spc.mol is not None:
            mol_copy = MagicMock()
            mol_copy.to_smiles.return_value = 'C'
            mol_copy.to_inchi.return_value = 'InChI=1S/CH4/h1H4'
            mol_copy.to_inchi_key.return_value = 'VNWKTOKETHGBQD-UHFFFAOYSA-N'
            spc.mol.copy.return_value = mol_copy
            spc.mol.get_formula.return_value = 'CH4'
        spc.final_xyz = {'symbols': ('C', 'H', 'H', 'H', 'H'),
                         'isotopes': (12, 1, 1, 1, 1),
                         'coords': ((0.0, 0.0, 0.0),
                                    (0.63, 0.63, 0.63),
                                    (-0.63, -0.63, 0.63),
                                    (-0.63, 0.63, -0.63),
                                    (0.63, -0.63, -0.63))}
        spc.initial_xyz = None
        spc.is_monoatomic.return_value = monoatomic
        spc.e_elect = -105236.6  # kJ/mol
        spc.e0 = -105136.6      # kJ/mol (e_elect + ZPE in kJ/mol)
        spc._is_linear = False
        spc.optical_isomers = 1
        spc.external_symmetry = 12
        spc.freqs = [1300.0, 1500.0, 3000.0, 3100.0]
        spc.rotors_dict = None
        spc.thermo = ThermoData(H298=-74.6, S298=186.3, Tmin=(300, 'K'), Tmax=(3000, 'K'))
        spc.rxn_label = None
        spc.ts_guesses = []
        spc.chosen_ts = None
        return spc

    def test_converged_species(self):
        spc = self._make_spc_mock()
        output_dict = {'CH4': {'convergence': True, 'paths': {'freq': '/abs/freq.log', 'sp': '/abs/sp.log'},
                                'job_types': {'opt': True}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertEqual(result['label'], 'CH4')
        self.assertTrue(result['converged'])
        self.assertFalse(result['is_ts'])
        self.assertEqual(result['smiles'], 'C')
        self.assertEqual(result['formula'], 'CH4')
        self.assertIsNotNone(result['sp_energy_hartree'])
        # zpe_hartree is parsed from the freq log file (which doesn't exist in this mock)
        self.assertIsNone(result['zpe_hartree'])
        self.assertIsNotNone(result['thermo'])
        self.assertIsNotNone(result['statmech'])
        self.assertEqual(result['freq_n_imag'], 0)
        self.assertIsNone(result['imag_freq_cm1'])
        self.assertEqual(result['freq_log'], 'freq.log')
        self.assertEqual(result['sp_log'], 'sp.log')

    def test_non_converged_species(self):
        spc = self._make_spc_mock(converged=False)
        output_dict = {'CH4': {'convergence': False, 'paths': {}, 'job_types': {}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertFalse(result['converged'])
        self.assertIsNone(result['sp_energy_hartree'])
        self.assertIsNone(result['zpe_hartree'])
        self.assertIsNone(result['freq_n_imag'])
        self.assertIsNone(result['thermo'])
        self.assertIsNone(result['statmech'])

    def test_monoatomic_species(self):
        spc = self._make_spc_mock(label='Ar', monoatomic=True)
        spc.final_xyz = {'symbols': ('Ar',), 'isotopes': (40,), 'coords': ((0.0, 0.0, 0.0),)}
        spc.freqs = None
        spc.thermo = None
        output_dict = {'Ar': {'convergence': True, 'paths': {}, 'job_types': {}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNone(result['freq_n_imag'])
        self.assertIsNone(result['imag_freq_cm1'])
        self.assertIsNone(result['zpe_hartree'])
        self.assertIsNone(result['statmech'])

    def test_ts_species(self):
        spc = self._make_spc_mock(label='TS0', is_ts=True)
        spc.rxn_label = 'CH4 + OH <=> CH3 + H2O'
        spc.thermo = None
        ts_guess = MagicMock()
        ts_guess.imaginary_freqs = [-1500.0]
        spc.ts_guesses = [ts_guess]
        spc.chosen_ts = 0
        output_dict = {'TS0': {'convergence': True, 'paths': {'freq': '/abs/freq.log', 'irc': ['/abs/irc_f.log', '/abs/irc_r.log']},
                                'job_types': {'opt': True, 'irc': True}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertTrue(result['is_ts'])
        self.assertEqual(result['freq_n_imag'], 1)
        self.assertAlmostEqual(result['imag_freq_cm1'], -1500.0)
        self.assertIsNone(result['thermo'])
        self.assertIsNone(result['smiles'])
        self.assertEqual(result['rxn_label'], 'CH4 + OH <=> CH3 + H2O')
        self.assertEqual(len(result['irc_logs']), 2)
        self.assertTrue(result['irc_converged'])

    def test_ts_irc_not_requested(self):
        spc = self._make_spc_mock(label='TS1', is_ts=True)
        spc.rxn_label = 'A <=> B'
        spc.thermo = None
        output_dict = {'TS1': {'convergence': True, 'paths': {'irc': []}, 'job_types': {}}}
        result = _spc_to_dict(spc, output_dict, '/abs', irc_requested=False)
        self.assertIsNone(result['irc_converged'])

    def test_ts_irc_failed(self):
        spc = self._make_spc_mock(label='TS2', is_ts=True)
        spc.rxn_label = 'A <=> B'
        spc.thermo = None
        output_dict = {'TS2': {'convergence': True, 'paths': {'irc': ['/abs/irc_f.log']},
                                'job_types': {'irc': False}}}
        result = _spc_to_dict(spc, output_dict, '/abs', irc_requested=True)
        self.assertFalse(result['irc_converged'])

    def test_point_groups_threaded(self):
        spc = self._make_spc_mock()
        output_dict = {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}
        pg = {'CH4': 'Td'}
        result = _spc_to_dict(spc, output_dict, '/abs', point_groups=pg)
        self.assertEqual(result['statmech']['point_group'], 'Td')

    def test_no_point_groups(self):
        spc = self._make_spc_mock()
        output_dict = {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNone(result['statmech']['point_group'])


class TestFlatCalculationContract(unittest.TestCase):
    """Pin the flat per-calculation shape that downstream uploaders consume.

    ARC emits every calculation result as a flat, prefixed scalar field on the
    species record, and never as a wrapped ``opt_result``/``freq_result``/
    ``sp_result`` object. Field names, flatness and units are asserted here.
    """

    FLAT_CALC_FIELDS = (
        'opt_converged',
        'opt_n_steps',
        'opt_final_energy_hartree',
        'freq_n_imag',
        'imag_freq_cm1',
        'zpe_hartree',
        'sp_energy_hartree',
    )

    WRAPPED_CALC_FIELDS = ('opt_result', 'freq_result', 'sp_result', 'irc_result', 'neb_result')

    def _converged(self, **kwargs):
        maker = TestSpcToDict()
        spc = maker._make_spc_mock(**kwargs)
        output_dict = {spc.label: {'convergence': True,
                                   'paths': {'freq': '/abs/freq.log', 'sp': '/abs/sp.log'},
                                   'job_types': {'opt': True}}}
        return spc, _spc_to_dict(spc, output_dict, '/abs')

    def test_calculation_fields_are_flat_and_present(self):
        """Every flat calc field is emitted on a converged species, wrapped ones never are."""
        _, record = self._converged()
        for field in self.FLAT_CALC_FIELDS:
            self.assertIn(field, record)
        for field in self.WRAPPED_CALC_FIELDS:
            self.assertNotIn(field, record)

    def test_flat_calc_fields_hold_scalars_not_result_objects(self):
        """The flat fields carry scalars or None, never a nested result mapping."""
        _, record = self._converged()
        for field in self.FLAT_CALC_FIELDS:
            with self.subTest(field=field):
                self.assertIsInstance(record[field], (bool, int, float, type(None)))

    def test_energy_fields_are_hartree_not_kj_mol(self):
        """``*_hartree`` fields are converted out of ARC's internal kJ/mol."""
        spc, record = self._converged()
        self.assertAlmostEqual(record['sp_energy_hartree'], spc.e_elect / E_h_kJmol)
        self.assertLess(abs(record['sp_energy_hartree']), abs(spc.e_elect))

    def test_imaginary_frequency_is_cm1_and_negative_for_a_ts(self):
        """A TS reports one imaginary mode, in cm^-1, carrying its negative sign."""
        maker = TestSpcToDict()
        spc = maker._make_spc_mock(label='TS0', is_ts=True)
        spc.thermo = None
        spc.rxn_label = 'A <=> B'
        guess = MagicMock()
        guess.imaginary_freqs = [-1500.0]
        spc.ts_guesses = [guess]
        spc.chosen_ts = 0
        output_dict = {'TS0': {'convergence': True, 'paths': {}, 'job_types': {'opt': True}}}
        record = _spc_to_dict(spc, output_dict, '/abs')
        self.assertEqual(record['freq_n_imag'], 1)
        self.assertAlmostEqual(record['imag_freq_cm1'], -1500.0)

    def test_unconverged_species_nulls_every_calc_field(self):
        """An unconverged species emits the same keys, all null, never a missing key."""
        maker = TestSpcToDict()
        spc = maker._make_spc_mock(converged=False)
        record = _spc_to_dict(spc, {'CH4': {'convergence': False, 'paths': {}, 'job_types': {}}}, '/abs')
        for field in self.FLAT_CALC_FIELDS:
            with self.subTest(field=field):
                self.assertIn(field, record)
                self.assertIsNone(record[field])


class TestComputePointGroups(unittest.TestCase):
    """Tests for _compute_point_groups."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp_dir, 'output'), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_spc(self, label, symbols, coords):
        spc = MagicMock()
        spc.label = label
        spc.final_xyz = {'symbols': tuple(symbols), 'coords': tuple(tuple(c) for c in coords)}
        spc.initial_xyz = None
        return spc

    @patch('arc.output.execute_command')
    @patch('arc.job.env_run.settings', {'RMG_ENV_NAME': 'rmg_env'})
    def test_returns_point_groups(self, mock_exec):
        mock_exec.return_value = ([], [])
        species_dict = {
            'H2O': self._make_spc('H2O', ['O', 'H', 'H'], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]]),
            'NH3': self._make_spc('NH3', ['N', 'H', 'H', 'H'], [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        }
        with patch('arc.output.read_yaml_file', return_value={'H2O': 'C2v', 'NH3': 'C3v'}):
            result = _compute_point_groups(species_dict, self.tmp_dir)
        self.assertEqual(result['H2O'], 'C2v')
        self.assertEqual(result['NH3'], 'C3v')

    @patch('arc.output.execute_command')
    @patch('arc.job.env_run.settings', {'RMG_ENV_NAME': 'rmg_env'})
    def test_null_point_group(self, mock_exec):
        mock_exec.return_value = ([], [])
        species_dict = {
            'Ar': self._make_spc('Ar', ['Ar'], [[0, 0, 0]]),
        }
        with patch('arc.output.read_yaml_file', return_value={'Ar': None}):
            result = _compute_point_groups(species_dict, self.tmp_dir)
        self.assertIsNone(result.get('Ar'))

    def test_empty_species_dict(self):
        result = _compute_point_groups({}, self.tmp_dir)
        self.assertEqual(result, {})

    def test_species_without_xyz(self):
        spc = MagicMock()
        spc.final_xyz = None
        spc.initial_xyz = None
        result = _compute_point_groups({'X': spc}, self.tmp_dir)
        self.assertEqual(result, {})

    @patch('arc.output.execute_command', side_effect=Exception('conda not found'))
    @patch('arc.job.env_run.settings', {'RMG_ENV_NAME': 'rmg_env'})
    def test_graceful_failure(self, mock_exec):
        species_dict = {
            'H2O': self._make_spc('H2O', ['O', 'H', 'H'], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]]),
        }
        result = _compute_point_groups(species_dict, self.tmp_dir)
        self.assertEqual(result, {})

    @patch('arc.output.execute_command')
    @patch('arc.job.env_run.settings', {'RMG_ENV_NAME': 'rmg_env'})
    def test_uses_initial_xyz_fallback(self, mock_exec):
        mock_exec.return_value = ([], [])
        spc = MagicMock()
        spc.label = 'CH4'
        spc.final_xyz = None
        spc.initial_xyz = {'symbols': ('C', 'H', 'H', 'H', 'H'),
                           'coords': ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, -1, -1))}
        with patch('arc.output.read_yaml_file', return_value={'CH4': 'Td'}):
            result = _compute_point_groups({'CH4': spc}, self.tmp_dir)
        self.assertEqual(result['CH4'], 'Td')

    @patch('arc.output.execute_command')
    @patch('arc.job.env_run.settings', {'RMG_ENV_NAME': 'rmg_env'})
    def test_writes_input_yaml(self, mock_exec):
        """Verify the input YAML is written and the script path is passed to execute_command."""
        mock_exec.return_value = ([], [])
        species_dict = {
            'H2O': self._make_spc('H2O', ['O', 'H', 'H'], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]]),
        }
        with patch('arc.output.read_yaml_file', return_value={'H2O': 'C2v'}):
            _compute_point_groups(species_dict, self.tmp_dir)
        mock_exec.assert_called_once()
        cmd = mock_exec.call_args[1].get('command') or mock_exec.call_args[0][0]
        cmd_str = str(cmd)
        self.assertIn('get_point_groups.py', cmd_str)
        self.assertIn('rmg_env', cmd_str)


class TestWriteOutputYml(unittest.TestCase):
    """Tests for write_output_yml (integration-level)."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp_dir, 'output'), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_spc_mock(self, label='CH4'):
        spc = MagicMock()
        spc.label = label
        spc.original_label = label
        spc.charge = 0
        spc.multiplicity = 1
        spc.is_ts = False
        mol_copy = MagicMock()
        mol_copy.to_smiles.return_value = 'C'
        mol_copy.to_inchi.return_value = 'InChI=1S/CH4/h1H4'
        mol_copy.to_inchi_key.return_value = 'VNWKTOKETHGBQD-UHFFFAOYSA-N'
        spc.mol = MagicMock()
        spc.mol.copy.return_value = mol_copy
        spc.mol.get_formula.return_value = 'CH4'
        spc.final_xyz = {'symbols': ('C', 'H', 'H', 'H', 'H'),
                         'isotopes': (12, 1, 1, 1, 1),
                         'coords': ((0.0, 0.0, 0.0), (0.63, 0.63, 0.63),
                                    (-0.63, -0.63, 0.63), (-0.63, 0.63, -0.63),
                                    (0.63, -0.63, -0.63))}
        spc.initial_xyz = None
        spc.is_monoatomic.return_value = False
        spc.e_elect = -105236.6
        spc.e0 = -105136.6
        spc._is_linear = False
        spc.optical_isomers = 1
        spc.external_symmetry = 12
        spc.freqs = [1300.0, 1500.0, 3000.0]
        spc.rotors_dict = None
        spc.thermo = ThermoData(H298=-74.6, S298=186.3, Tmin=(300, 'K'), Tmax=(3000, 'K'))
        return spc

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_git_commit', return_value='abc123')
    @patch('arc.output.get_git_commit', return_value=('def456', '2026-01-01'))
    def test_writes_file_atomically(self, mock_arc_git, mock_arkane_git, mock_pg):
        from arc.common import read_yaml_file
        spc = self._make_spc_mock()
        species_dict = {'CH4': spc}
        output_dict = {'CH4': {'convergence': True, 'paths': {}, 'job_types': {'opt': True}}}

        write_output_yml(
            project='test_project',
            project_directory=self.tmp_dir,
            species_dict=species_dict,
            reactions=[],
            output_dict=output_dict,
        )

        out_path = os.path.join(self.tmp_dir, 'output', 'output.yml')
        self.assertTrue(os.path.isfile(out_path))
        doc = read_yaml_file(out_path)
        self.assertEqual(doc['schema_version'], '1.1')
        self.assertEqual(doc['tckdb_evidence']['path'], 'tckdb_evidence.json')
        evidence_path = os.path.join(self.tmp_dir, 'output', 'tckdb_evidence.json')
        self.assertTrue(os.path.isfile(evidence_path))
        with open(evidence_path) as handle:
            evidence = json.load(handle)
        self.assertEqual(evidence['document_id'], doc['tckdb_evidence']['document_id'])
        self.assertEqual(doc['project'], 'test_project')
        self.assertEqual(doc['arc_git_commit'], 'def456')
        self.assertEqual(doc['arkane_git_commit'], 'abc123')
        self.assertIsInstance(doc['species'], list)
        self.assertEqual(len(doc['species']), 1)
        self.assertEqual(doc['species'][0]['label'], 'CH4')
        self.assertEqual(doc['reactions'], [])
        self.assertEqual(doc['transition_states'], [])

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_git_commit', return_value=None)
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_ess_software_is_emitted_and_pairs_with_ess_versions(self, mock_arc_git, mock_arkane_git, mock_pg):
        """``ess_software`` names the program per job type even when the level declares another one."""
        from arc.common import read_yaml_file
        spc = self._make_spc_mock()
        paths = {'geo': os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out'),
                 'sp': os.path.join(ARC_TESTING_PATH, 'orca_example_opt.log')}

        write_output_yml(
            project='test_ess_software',
            project_directory=self.tmp_dir,
            species_dict={'CH4': spc},
            reactions=[],
            output_dict={'CH4': {'convergence': True, 'paths': paths, 'job_types': {}}},
            opt_level=Level(method='wb97xd', basis='def2tzvp', software='gaussian'),
            freq_level=Level(method='wb97xd', basis='def2tzvp', software='gaussian'),
            sp_level=Level(method='wb97xd', basis='def2tzvp', software='gaussian'),
        )

        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        entry = doc['species'][0]
        self.assertEqual(entry['ess_software'], {'opt': 'gaussian', 'sp': 'orca'})
        self.assertIn('Gaussian', entry['ess_versions']['opt'])
        self.assertIn('ORCA', entry['ess_versions']['sp'])

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_git_commit', return_value=None)
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_ess_software_is_null_for_a_non_converged_species(self, mock_arc_git, mock_arkane_git, mock_pg):
        from arc.common import read_yaml_file
        spc = self._make_spc_mock()
        paths = {'geo': os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')}

        write_output_yml(
            project='test_ess_software_unconverged',
            project_directory=self.tmp_dir,
            species_dict={'CH4': spc},
            reactions=[],
            output_dict={'CH4': {'convergence': False, 'paths': paths, 'job_types': {}}},
        )

        entry = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))['species'][0]
        self.assertIn('ess_software', entry)
        self.assertIsNone(entry['ess_software'])
        self.assertIsNone(entry['ess_versions'])

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_git_commit', return_value=None)
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_levels_of_theory(self, mock_arc_git, mock_arkane_git, mock_pg):
        from arc.common import read_yaml_file
        spc = self._make_spc_mock()
        opt_level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        freq_level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        sp_level = Level(method='dlpno-ccsd(t)', basis='cc-pvtz', software='orca')

        write_output_yml(
            project='test_lot',
            project_directory=self.tmp_dir,
            species_dict={'CH4': spc},
            reactions=[],
            output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
            opt_level=opt_level,
            freq_level=freq_level,
            sp_level=sp_level,
            freq_scale_factor=0.975,
            freq_scale_factor_user_provided=True,
            bac_type='p',
        )

        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        self.assertEqual(doc['opt_level']['method'], 'wb97xd')
        self.assertEqual(doc['sp_level']['method'], 'dlpno-ccsd(t)')
        self.assertAlmostEqual(doc['freq_scale_factor'], 0.975)
        self.assertIsNone(doc['freq_scale_factor_source'])  # user-provided
        self.assertEqual(doc['bac_type'], 'p')

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_git_commit', return_value=None)
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_arkane_level_of_theory(self, mock_arc_git, mock_arkane_git, mock_pg):
        from arc.common import read_yaml_file
        spc = self._make_spc_mock()
        sp_level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        arkane_lot = Level(method='cbs-qb3', software='gaussian')

        write_output_yml(
            project='test_arkane_lot',
            project_directory=self.tmp_dir,
            species_dict={'CH4': spc},
            reactions=[],
            output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
            sp_level=sp_level,
            arkane_level_of_theory=arkane_lot,
        )

        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        self.assertEqual(doc['arkane_level_of_theory']['method'], 'cbs-qb3')
        self.assertEqual(doc['sp_level']['method'], 'wb97xd')

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_git_commit', return_value=None)
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_arkane_level_none_when_not_set(self, mock_arc_git, mock_arkane_git, mock_pg):
        from arc.common import read_yaml_file
        spc = self._make_spc_mock()

        write_output_yml(
            project='test_no_arkane',
            project_directory=self.tmp_dir,
            species_dict={'CH4': spc},
            reactions=[],
            output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
        )

        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        self.assertIsNone(doc['arkane_level_of_theory'])

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_git_commit', return_value=None)
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_energy_correction_records_carry_the_matched_arkane_key(self, mock_arc_git, mock_arkane_git, mock_pg):
        """The writer threads the matched Arkane key onto every emitted correction record."""
        aec_key = "LevelOfTheory(method='wb97xd',basis='def2tzvp',software='gaussian')"
        corrections = EnergyCorrections(aec={'C': -37.8, 'H': -0.5}, bac={'C-H': -0.17},
                                        aec_key=aec_key, bac_key=aec_key)
        species_corrections = {'CH4': {
            'aec': {'value': -0.02, 'value_unit': 'hartree', 'components': []},
            'bac': {'value': -0.7, 'value_unit': 'kcal_mol', 'components': []},
        }}
        with patch('arc.output._get_energy_corrections', return_value=corrections), \
             patch('arc.output._compute_species_corrections', return_value=species_corrections):
            write_output_yml(
                project='matched_key_wiring',
                project_directory=self.tmp_dir,
                species_dict={'CH4': self._make_spc_mock()},
                reactions=[],
                output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
                sp_level=Level(method='wb97xd', basis='def2tzvp', software='gaussian'),
                arkane_level_of_theory=Level(method='wb97xd', basis='def2tzvp', software='gaussian'),
                bac_type='p',
            )

        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        records = doc['species'][0]['energy_corrections']
        self.assertEqual(sorted(r['correction_type'] for r in records),
                         ['atom_energy', 'bond_additivity'])
        for record in records:
            self.assertEqual(record['matched_arkane_key'], aec_key)
        bac = next(r for r in records if r['correction_type'] == 'bond_additivity')
        self.assertEqual(bac['parameter_table']['values'], {'C-H': -0.17})

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_git_commit', return_value=None)
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_no_tmp_files_left(self, mock_arc_git, mock_arkane_git, mock_pg):
        """After a successful write, no .tmp files should remain."""
        spc = self._make_spc_mock()
        write_output_yml(
            project='cleanup_test',
            project_directory=self.tmp_dir,
            species_dict={'CH4': spc},
            reactions=[],
            output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
        )
        out_dir = os.path.join(self.tmp_dir, 'output')
        leftover = [f for f in os.listdir(out_dir) if f.endswith('.tmp')]
        self.assertEqual(leftover, [])

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output.build_tckdb_evidence', side_effect=RuntimeError('evidence failed'))
    def test_evidence_failure_still_writes_output(self, mock_build, mock_pg):
        spc = self._make_spc_mock()
        write_output_yml(
            project='evidence_failure',
            project_directory=self.tmp_dir,
            species_dict={'CH4': spc},
            reactions=[],
            output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
        )
        from arc.common import read_yaml_file
        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        self.assertEqual(doc['schema_version'], '1.1')
        self.assertNotIn('tckdb_evidence', doc)

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output.build_tckdb_evidence', side_effect=RuntimeError('evidence failed'))
    def test_evidence_failure_removes_the_previous_runs_sidecar(self, mock_build, mock_pg):
        """A run that cannot produce evidence must not leave the last run's beside its output.

        The sidecar is only interpretable against the ``output.yml`` naming its
        ``document_id``. A surviving sidecar from an earlier run sits beside a
        document with no descriptor at all, so nothing marks it stale and a
        consumer reads last week's Hessians as this run's.
        """
        out_dir = os.path.join(self.tmp_dir, 'output')
        os.makedirs(out_dir, exist_ok=True)
        stale_path = os.path.join(out_dir, 'tckdb_evidence.json')
        with open(stale_path, 'w') as handle:
            handle.write('{"document_id": "from_a_previous_run"}\n')

        spc = self._make_spc_mock()
        write_output_yml(
            project='evidence_failure',
            project_directory=self.tmp_dir,
            species_dict={'CH4': spc},
            reactions=[],
            output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
        )
        doc = read_yaml_file(os.path.join(out_dir, 'output.yml'))
        self.assertNotIn('tckdb_evidence', doc)
        self.assertFalse(os.path.exists(stale_path))

    @patch('arc.output._compute_point_groups', return_value={})
    def test_evidence_is_replaced_before_output(self, mock_pg):
        spc = self._make_spc_mock()
        real_replace = os.replace
        destinations = []

        def recording_replace(source, destination):
            destinations.append(os.path.basename(destination))
            return real_replace(source, destination)

        with patch('os.replace', side_effect=recording_replace):
            write_output_yml(
                project='replace_order',
                project_directory=self.tmp_dir,
                species_dict={'CH4': spc},
                reactions=[],
                output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
            )
        self.assertEqual(destinations[-2:], ['tckdb_evidence.json', 'output.yml'])


class TestGetPointGroupsScript(unittest.TestCase):
    """Tests for arc/scripts/get_point_groups.py helper functions (imported directly)."""

    def test_point_group_for_monoatomic(self):
        """Monoatomic species should return 'Kh' without calling the binary."""
        import importlib.util
        script_path = os.path.join(ARC_PATH, 'arc', 'scripts', 'get_point_groups.py')
        spec = importlib.util.spec_from_file_location('get_point_groups', script_path,
                                                       submodule_search_locations=[])
        # The script imports from 'common' which is in the scripts dir — add it to path
        import sys
        scripts_dir = os.path.join(ARC_PATH, 'arc', 'scripts')
        added = scripts_dir not in sys.path
        if added:
            sys.path.insert(0, scripts_dir)
        try:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            result = mod._point_group_for_species(['Ar'], [[0.0, 0.0, 0.0]])
            self.assertEqual(result, 'Kh')
        finally:
            if added:
                sys.path.remove(scripts_dir)

    def test_point_group_for_empty(self):
        import importlib.util
        import sys
        script_path = os.path.join(ARC_PATH, 'arc', 'scripts', 'get_point_groups.py')
        spec = importlib.util.spec_from_file_location('get_point_groups', script_path)
        scripts_dir = os.path.join(ARC_PATH, 'arc', 'scripts')
        added = scripts_dir not in sys.path
        if added:
            sys.path.insert(0, scripts_dir)
        try:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.assertIsNone(mod._point_group_for_species([], []))
            self.assertIsNone(mod._point_group_for_species(None, None))
        finally:
            if added:
                sys.path.remove(scripts_dir)

    def test_point_group_unknown_element(self):
        import importlib.util
        import sys
        script_path = os.path.join(ARC_PATH, 'arc', 'scripts', 'get_point_groups.py')
        spec = importlib.util.spec_from_file_location('get_point_groups', script_path)
        scripts_dir = os.path.join(ARC_PATH, 'arc', 'scripts')
        added = scripts_dir not in sys.path
        if added:
            sys.path.insert(0, scripts_dir)
        try:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # 'Uuo' is not in the lookup table
            result = mod._point_group_for_species(['Uuo', 'H'], [[0, 0, 0], [1, 0, 0]])
            self.assertIsNone(result)
        finally:
            if added:
                sys.path.remove(scripts_dir)


class TestBuildAppliedCorrectionsForSpecies(unittest.TestCase):
    """Direct tests for `_build_energy_corrections_for_species`.

    Stubs the rmg_env script's per-label result so we don't depend on the
    Arkane subprocess; the helper's job is purely shape-translation.
    """

    def _lot(self):
        return Level(method='wb97xd3', basis='def2tzvp', software='qchem')

    def _aec_block(self):
        return {
            'value': -0.0234,
            'value_unit': 'hartree',
            'components': [
                {'component_kind': 'atom', 'key': 'C', 'multiplicity': 1,
                 'parameter_value': -37.84993993, 'parameter_unit': 'hartree',
                 'contribution_value': -0.015},
                {'component_kind': 'atom', 'key': 'H', 'multiplicity': 4,
                 'parameter_value': -0.49991749, 'parameter_unit': 'hartree',
                 'contribution_value': -0.008},
            ],
        }

    def _pbac_block(self):
        return {
            'value': -0.694,
            'value_unit': 'kcal_mol',
            'bac_type': 'p',
            'components': [
                {'component_kind': 'bond', 'key': 'C-H', 'multiplicity': 4,
                 'parameter_value': -0.1735, 'parameter_unit': 'kcal_mol',
                 'contribution_value': -0.694},
            ],
        }

    def _mbac_block(self):
        return {
            'value': -0.056,
            'value_unit': 'kcal_mol',
            'bac_type': 'm',
        }

    def test_aec_total_emitted(self):
        sc = {'CH4': {'aec': self._aec_block()}}
        out = _build_energy_corrections_for_species('CH4', sc, self._lot(), 'p')
        roles = [e['correction_type'] for e in out]
        self.assertIn('atom_energy', roles)
        aec = next(e for e in out if e['correction_type'] == 'atom_energy')
        self.assertAlmostEqual(aec['total']['value'], -0.0234)
        self.assertEqual(aec['total']['unit'], 'hartree')
        self.assertEqual(aec['model'], 'arkane_atom_energy')

    def test_aec_components_sum_to_total(self):
        # Use values that arithmetically sum exactly so the test
        # asserts the producer doesn't drop or rescale rows.
        block = {
            'value': -0.030,
            'value_unit': 'hartree',
            'components': [
                {'component_kind': 'atom', 'key': 'C', 'multiplicity': 1,
                 'parameter_value': -37.85, 'parameter_unit': 'hartree',
                 'contribution_value': -0.018},
                {'component_kind': 'atom', 'key': 'H', 'multiplicity': 4,
                 'parameter_value': -0.5, 'parameter_unit': 'hartree',
                 'contribution_value': -0.012},
            ],
        }
        sc = {'X': {'aec': block}}
        out = _build_energy_corrections_for_species('X', sc, self._lot(), None)
        aec = next(e for e in out if e['correction_type'] == 'atom_energy')
        total = sum(c['contribution_value'] for c in aec['components'])
        self.assertAlmostEqual(total, aec['total']['value'], places=6)

    def test_pbac_total_and_components(self):
        sc = {'CH4': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_energy_corrections_for_species('CH4', sc, self._lot(), 'p')
        bac = next(e for e in out if e['correction_type'] == 'bond_additivity')
        self.assertEqual(bac['model'], 'petersson')
        self.assertEqual(bac['total']['unit'], 'kcal_mol')
        self.assertEqual(len(bac['components']), 1)
        self.assertEqual(bac['components'][0]['key'], 'C-H')

    def test_mbac_total_only_no_components(self):
        sc = {'CH4': {'aec': self._aec_block(), 'bac': self._mbac_block()}}
        out = _build_energy_corrections_for_species('CH4', sc, self._lot(), 'm')
        bac = next(e for e in out if e['correction_type'] == 'bond_additivity')
        self.assertEqual(bac['model'], 'melius')
        self.assertEqual(bac['components'], [])

    def test_pbac_omits_components_when_param_missing(self):
        block = self._pbac_block()
        block['components'][0]['parameter_value'] = None
        sc = {'X': {'aec': self._aec_block(), 'bac': block}}
        out = _build_energy_corrections_for_species('X', sc, self._lot(), 'p')
        bac = next(e for e in out if e['correction_type'] == 'bond_additivity')
        # Components dropped entirely (partial decomposition would mislead).
        self.assertEqual(bac['components'], [])

    def test_units_are_explicit(self):
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_energy_corrections_for_species('X', sc, self._lot(), 'p')
        units = {e['correction_type']: e['total']['unit'] for e in out}
        self.assertEqual(units['atom_energy'], 'hartree')
        self.assertEqual(units['bond_additivity'], 'kcal_mol')

    def test_missing_correction_omits_silently(self):
        # AEC failed (no 'aec' key), BAC succeeded → only BAC emitted.
        sc = {'X': {'bac': self._pbac_block()}}
        out = _build_energy_corrections_for_species('X', sc, self._lot(), 'p')
        roles = [e['correction_type'] for e in out]
        self.assertEqual(roles, ['bond_additivity'])

    def test_no_data_returns_empty_list(self):
        out = _build_energy_corrections_for_species('X', {}, self._lot(), 'p')
        self.assertEqual(out, [])

    def test_bac_type_none_omits_bac(self):
        # Even if a BAC block is present, bac_type=None means no BAC role.
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_energy_corrections_for_species('X', sc, self._lot(), None)
        roles = [e['correction_type'] for e in out]
        self.assertEqual(roles, ['atom_energy'])

    # ---- provenance of the parameters each record was computed from ----

    def test_matched_arkane_key_is_the_atom_energy_key_on_every_record(self):
        """Both records report the atom-energy key, which is the key ARC hands Arkane."""
        aec_key = "LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='qchem')"
        bac_key = "LevelOfTheory(method='wb97xd3',basis='def2tzvp')"
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_energy_corrections_for_species(
            'X', sc, self._lot(), 'p', aec_key=aec_key, bac_key=bac_key,
        )
        self.assertEqual([e['matched_arkane_key'] for e in out], [aec_key, aec_key])

    def test_matched_arkane_key_unchanged_by_an_unmatched_bac_key(self):
        """An unmatched BAC key leaves the reported key as the atom-energy key."""
        aec_key = "LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='qchem')"
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_energy_corrections_for_species(
            'X', sc, self._lot(), 'p', aec_key=aec_key, bac_key=None,
        )
        bac = next(e for e in out if e['correction_type'] == 'bond_additivity')
        self.assertEqual(bac['matched_arkane_key'], aec_key)

    def test_matched_arkane_key_is_none_when_the_caller_supplies_no_keys(self):
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_energy_corrections_for_species('X', sc, self._lot(), 'p')
        for correction in out:
            self.assertIsNone(correction['matched_arkane_key'])

    def test_level_of_theory_and_matched_arkane_key_are_not_conflated(self):
        """``level_of_theory`` is what ARC ran; ``matched_arkane_key`` is what Arkane matched."""
        aec_key = "LevelOfTheory(method='wb97xd',basis='def2tzvp',software='gaussian')"
        sc = {'X': {'aec': self._aec_block()}}
        out = _build_energy_corrections_for_species(
            'X', sc, self._lot(), None, aec_key=aec_key,
        )
        aec = next(e for e in out if e['correction_type'] == 'atom_energy')
        self.assertIsInstance(aec['level_of_theory'], dict)
        self.assertEqual(aec['level_of_theory']['method'], 'wb97xd3')
        self.assertEqual(aec['level_of_theory']['software'], 'qchem')
        self.assertIsInstance(aec['matched_arkane_key'], str)
        self.assertEqual(aec['matched_arkane_key'], aec_key)
        self.assertNotEqual(aec['level_of_theory'], aec['matched_arkane_key'])

    # ---- scheme parameter tables (atom_params / bond_params) ----

    def test_aec_scheme_includes_atom_params_from_run_table(self):
        # ARC's run-level atom_energy_corrections dict is the source of
        # truth for AEC scheme parameters; without atom_params the
        # downstream energy_correction_scheme_atom_param table never gets
        # populated even though the applied row lands. Sorted-by-element
        # for deterministic output.yml.
        aec_table = {'C': -37.84706, 'H': -0.50066}
        sc = {'X': {'aec': self._aec_block()}}
        out = _build_energy_corrections_for_species(
            'X', sc, self._lot(), 'p', aec_table=aec_table, bac_table=None,
        )
        aec = next(e for e in out if e['correction_type'] == 'atom_energy')
        self.assertEqual(
            aec['reference_atom_energies']['values'],
            {'C': -37.84706, 'H': -0.50066},
        )

    def test_pbac_scheme_includes_bond_params_from_run_table(self):
        bac_table = {'C-H': -0.17350, 'C=O': -2.63454}
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_energy_corrections_for_species(
            'X', sc, self._lot(), 'p', aec_table=None, bac_table=bac_table,
        )
        bac = next(e for e in out if e['correction_type'] == 'bond_additivity')
        self.assertEqual(
            bac['parameter_table']['values'],
            {'C-H': -0.17350, 'C=O': -2.63454},
        )

    def test_pbac_parameter_table_emitted_when_the_bac_key_is_the_applied_key(self):
        """The BAC table is emitted when the BAC-matched key is the key the numbers came from."""
        key = "LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='qchem')"
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_energy_corrections_for_species(
            'X', sc, self._lot(), 'p', bac_table={'C-H': -0.17350},
            aec_key=key, bac_key=key,
        )
        bac = next(e for e in out if e['correction_type'] == 'bond_additivity')
        self.assertEqual(bac['parameter_table']['values'], {'C-H': -0.17350})

    def test_pbac_parameter_table_omitted_when_the_bac_key_differs_from_the_applied_key(self):
        """A BAC table matched under a different key than the applied one is omitted."""
        aec_key = "LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='qchem')"
        bac_key = "LevelOfTheory(method='wb97xd3',basis='def2tzvp')"
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_energy_corrections_for_species(
            'X', sc, self._lot(), 'p', bac_table={'C-H': -0.17350},
            aec_key=aec_key, bac_key=bac_key,
        )
        bac = next(e for e in out if e['correction_type'] == 'bond_additivity')
        self.assertNotIn('parameter_table', bac)
        self.assertEqual(bac['matched_arkane_key'], aec_key)
        self.assertAlmostEqual(bac['total']['value'], -0.694)

    def test_pbac_parameter_table_omitted_when_no_bac_key_matched(self):
        """An unmatched BAC key omits the BAC table while the total and components stay."""
        aec_key = "LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='qchem')"
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_energy_corrections_for_species(
            'X', sc, self._lot(), 'p', bac_table={'C-H': -0.17350},
            aec_key=aec_key, bac_key=None,
        )
        bac = next(e for e in out if e['correction_type'] == 'bond_additivity')
        self.assertNotIn('parameter_table', bac)
        self.assertEqual(len(bac['components']), 1)

    def test_parameter_table_unit_is_the_table_unit_not_the_total_unit(self):
        """Parameter tables carry their own source unit, not the total block's.

        The values come from the Arkane AEC/BAC tables (Hartree and kcal/mol
        respectively) while ``total.unit`` describes the per-species total. The
        two are independent, so a divergent total unit must not relabel the
        per-parameter values.
        """
        aec_block = self._aec_block()
        aec_block['value_unit'] = 'kj_mol'
        bac_block = self._pbac_block()
        bac_block['value_unit'] = 'kj_mol'
        sc = {'X': {'aec': aec_block, 'bac': bac_block}}
        out = _build_energy_corrections_for_species(
            'X', sc, self._lot(), 'p',
            aec_table={'C': -37.84706}, bac_table={'C-H': -0.17350},
        )
        aec = next(e for e in out if e['correction_type'] == 'atom_energy')
        bac = next(e for e in out if e['correction_type'] == 'bond_additivity')
        self.assertEqual(aec['reference_atom_energies']['unit'], 'hartree')
        self.assertEqual(aec['reference_atom_energies']['applied_as'], 'subtracted')
        self.assertEqual(bac['parameter_table']['unit'], 'kcal_mol')
        self.assertEqual(aec['total']['unit'], 'kj_mol')
        self.assertEqual(bac['total']['unit'], 'kj_mol')

    def test_mbac_scheme_omits_params(self):
        # Per spec: Melius BAC parameters are atom-pair / length / neighbor /
        # molecular and don't fit SchemeBondParamPayload's bond-key shape.
        # The producer must NOT fabricate or coerce them — emit total only.
        bac_table = {'C-H': -0.17350}  # would coerce, but we must not
        sc = {'X': {'aec': self._aec_block(), 'bac': self._mbac_block()}}
        out = _build_energy_corrections_for_species(
            'X', sc, self._lot(), 'm', aec_table=None, bac_table=bac_table,
        )
        bac = next(e for e in out if e['correction_type'] == 'bond_additivity')
        self.assertEqual(bac['model'], 'melius')
        self.assertNotIn('parameter_table', bac)

    def test_a_nested_melius_table_degrades_instead_of_raising(self):
        """Arkane's real Melius table is nested and must not reach the flattener.

        ``write_output_yml``'s caller logs and discards any exception, taking
        the whole document with it, so a ``TypeError`` here would silently cost
        an entire run its ``output.yml``. The record omits the parameter table
        rather than raising, whichever ``bac_type`` is in play.
        """
        nested = {
            'atom_corr': {'C': 0.20491, 'H': -2.06284},
            'bond_corr_length': {'C': 0.05644, 'H': 1.07830},
            'bond_corr_neighbor': {'C': -0.09439, 'H': -0.17906},
            'mol_corr': -3.782190737782739,
        }
        self.assertIsNone(_flat_parameter_values(nested))
        self.assertIsNone(_flat_parameter_values(None))
        self.assertEqual(_flat_parameter_values({'C-H': -0.1735}), {'C-H': -0.1735})
        for bac_type, block in (('m', self._mbac_block()), ('p', self._pbac_block())):
            with self.subTest(bac_type=bac_type):
                sc = {'X': {'aec': self._aec_block(), 'bac': block}}
                out = _build_energy_corrections_for_species(
                    'X', sc, self._lot(), bac_type, aec_table=None, bac_table=nested,
                    aec_key='k', bac_key='k',
                )
                bac = next(e for e in out if e['correction_type'] == 'bond_additivity')
                self.assertNotIn('parameter_table', bac)

    def test_aec_scheme_omits_atom_params_when_table_missing(self):
        # Backward compat: when aec_table isn't supplied (caller predates
        # this fix, or output.yml was written without it), the scheme still
        # has identity but no atom_params field — schema treats it as []
        # via the default factory.
        sc = {'X': {'aec': self._aec_block()}}
        out = _build_energy_corrections_for_species(
            'X', sc, self._lot(), 'p', aec_table=None, bac_table=None,
        )
        aec = next(e for e in out if e['correction_type'] == 'atom_energy')
        self.assertNotIn('reference_atom_energies', aec)

    def test_atom_params_sorted_for_determinism(self):
        # Stable insertion order matters for the idempotency hash
        # downstream consumers compute over the payload.
        aec_table = {'O': -75.07, 'H': -0.5, 'C': -37.85}
        sc = {'X': {'aec': self._aec_block()}}
        out = _build_energy_corrections_for_species(
            'X', sc, self._lot(), None, aec_table=aec_table, bac_table=None,
        )
        aec = next(e for e in out if e['correction_type'] == 'atom_energy')
        self.assertEqual(list(aec['reference_atom_energies']['values']), ['C', 'H', 'O'])


class TestComputeSpeciesCorrections(unittest.TestCase):
    """Tests for `_compute_species_corrections` orchestration (subprocess call)."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        os.makedirs(os.path.join(self.tmp_dir, 'output'), exist_ok=True)

    def _spc(self, label='CH4'):
        spc = MagicMock()
        spc.label = label
        spc.multiplicity = 1
        spc.bond_corrections = {'C-H': 4}
        spc.final_xyz = {'symbols': ('C', 'H', 'H', 'H', 'H'),
                         'isotopes': (12, 1, 1, 1, 1),
                         'coords': ((0, 0, 0), (0.6, 0.6, 0.6),
                                    (-0.6, -0.6, 0.6), (-0.6, 0.6, -0.6),
                                    (0.6, -0.6, -0.6))}
        spc.initial_xyz = None
        return spc

    AEC_KEY = "LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='qchem')"
    BAC_KEY = "LevelOfTheory(method='wb97xd3',basis='def2tzvp')"

    @staticmethod
    def _script_result(label='CH4'):
        return {'species': [
            {'label': label,
             'aec': {'value': -0.02, 'value_unit': 'hartree', 'components': []},
             'bac': {'value': -0.7, 'value_unit': 'kcal_mol', 'components': []}},
        ]}

    def test_returns_empty_when_no_atom_energy_key_matched(self):
        """Without an atom-energy key Arkane applies no corrections, so none are computed."""
        with patch('arc.output.execute_command') as mock_exec:
            out = _compute_species_corrections({'CH4': self._spc()}, None, 'p', self.tmp_dir)
        self.assertEqual(out, {})
        mock_exec.assert_not_called()

    def test_returns_empty_when_no_species_have_xyz(self):
        spc = self._spc()
        spc.final_xyz = None
        spc.initial_xyz = None
        with patch('arc.output.execute_command') as mock_exec:
            out = _compute_species_corrections({'CH4': spc}, self.AEC_KEY, 'p', self.tmp_dir)
        self.assertEqual(out, {})
        mock_exec.assert_not_called()

    def test_invokes_subprocess_with_batched_input(self):
        with patch('arc.output.execute_command', return_value=('', '')) as mock_exec, \
             patch('arc.output.read_yaml_file', return_value=self._script_result()), \
             patch('arc.output.save_yaml_file') as mock_save:
            out = _compute_species_corrections(
                {'CH4': self._spc()}, self.AEC_KEY, 'p', self.tmp_dir,
            )
        self.assertEqual(mock_exec.call_count, 1)
        self.assertIn('CH4', out)
        self.assertEqual(out['CH4']['aec']['value'], -0.02)
        self.assertEqual(out['CH4']['bac']['value'], -0.7)
        save_call = mock_save.call_args
        content = save_call[1].get('content') or save_call[0][1]
        self.assertEqual(content['level_of_theory'], self.AEC_KEY)
        self.assertEqual(content['bac_type'], 'p')
        self.assertEqual(len(content['species']), 1)
        self.assertEqual(content['species'][0]['label'], 'CH4')
        self.assertEqual(content['species'][0]['atoms'], {'C': 1, 'H': 4})
        self.assertEqual(content['species'][0]['bonds'], {'C-H': 4})
        self.assertEqual(content['species'][0]['multiplicity'], 1)

    def test_both_corrections_computed_from_the_atom_energy_key_in_one_pass(self):
        """The AEC and the BAC come from a single invocation keyed on the atom-energy key."""
        with patch('arc.output.execute_command', return_value=('', '')) as mock_exec, \
             patch('arc.output.read_yaml_file', return_value=self._script_result()), \
             patch('arc.output.save_yaml_file') as mock_save:
            out = _compute_species_corrections(
                {'CH4': self._spc()}, self.AEC_KEY, 'p', self.tmp_dir,
            )
        self.assertEqual(mock_exec.call_count, 1)
        contents = [(c[1].get('content') or c[0][1]) for c in mock_save.call_args_list]
        self.assertEqual([c['level_of_theory'] for c in contents], [self.AEC_KEY])
        self.assertEqual(contents[0]['bac_type'], 'p')
        self.assertEqual(contents[0]['species'][0]['atoms'], {'C': 1, 'H': 4})
        self.assertEqual(out['CH4']['aec']['value'], -0.02)
        self.assertEqual(out['CH4']['bac']['value'], -0.7)

    def test_no_pass_is_keyed_on_the_bac_section_key(self):
        """A BAC-section key that differs from the atom-energy key drives no invocation."""
        with patch('arc.output.execute_command', return_value=('', '')), \
             patch('arc.output.read_yaml_file', return_value=self._script_result()), \
             patch('arc.output.save_yaml_file') as mock_save:
            _compute_species_corrections({'CH4': self._spc()}, self.AEC_KEY, 'p', self.tmp_dir)
        contents = [(c[1].get('content') or c[0][1]) for c in mock_save.call_args_list]
        self.assertNotIn(self.BAC_KEY, [c['level_of_theory'] for c in contents])

    def test_subprocess_failure_returns_empty(self):
        with patch('arc.output.execute_command', side_effect=RuntimeError('boom')):
            out = _compute_species_corrections(
                {'CH4': self._spc()}, self.AEC_KEY, 'p', self.tmp_dir,
            )
        self.assertEqual(out, {})


class TestScanCalculations(unittest.TestCase):
    """Tests for tool-neutral rotor-scan result export.

    Covers two layers:
    - ``_build_scan_result_for_rotor`` preserves scientific scan facts,
      returning ``None`` when the input is unusable.
    - ``_build_rotor_scans`` aggregates across ``rotors_dict`` and
      filters non-1D / failed / unparseable rotors.
    - ``_get_torsions`` attaches ``source_scan_key`` only when
      the corresponding scan log is on disk.
    """

    SCAN_LOG = os.path.join(ARC_TESTING_PATH, 'rotor_scans', 'sBuOH.out')

    def _rotor(self, **overrides) -> dict:
        """Build a rotor-dict with sensible defaults; override per-test."""
        rotor: dict = {
            'success': True,
            'scan': [1, 2, 3, 4],
            'pivots': [2, 3],
            'symmetry': 3,
            'type': 'HinderedRotor',
            'scan_path': self.SCAN_LOG,
            'dimensions': 1,
        }
        rotor.update(overrides)
        return rotor

    def test_build_scan_result_happy_path(self):
        """Real Gaussian scan log → fully populated scan_result dict."""
        result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        self.assertEqual(result['dimension'], 1)
        self.assertTrue(result['relaxed'])
        # zero_energy_reference_hartree = min absolute energy on the curve.
        self.assertIsInstance(result['zero_energy_reference_hartree'], float)
        # One coordinate, dihedral, atoms 1–4, 1-based, with symmetry.
        coord = result['coordinate']
        self.assertEqual(coord['coordinate_type'], 'dihedral')
        self.assertEqual(coord['atom_indices'], [1, 2, 3, 4])
        self.assertEqual(coord['index_base'], 1)
        self.assertEqual(coord['unit'], 'degree')
        self.assertEqual(coord['symmetry_number'], 3)
        self.assertEqual(coord['sample_count'], len(result['samples']))
        # Samples carry source index, angle, energies, and ARC-format geometry
        # text straight from ``parse_1d_scan_full_result()['geometries']``.
        self.assertGreater(len(result['samples']), 0)
        first = result['samples'][0]
        self.assertEqual(first['source_index'], 0)
        self.assertIn('angle_degrees', first)
        self.assertIn('relative_energy_kj_mol', first)
        self.assertIn('electronic_energy_hartree', first)
        self.assertNotIn('xyz', first)
        # First point's relative energy ≈ 0 by zero-shift convention.
        self.assertAlmostEqual(first['relative_energy_kj_mol'], 1.5753056e-05,
                               places=6)

    def test_sample_absolute_energies_are_index_aligned(self):
        """Each sample's absolute energy is the one at its own index, not a neighbour's.

        ``relative_energy_kj_mol`` was pinned to an exact value but
        ``electronic_energy_hartree`` was only asserted present, so an off-by-one
        in the absolute-energy lookup attached the wrong point's energy to every
        exported sample without failing anything. Pin the correspondence itself:
        the absolute energies must reproduce the relative curve.
        """
        result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        samples = result['samples']
        absolute = [s['electronic_energy_hartree'] for s in samples]
        relative = [s['relative_energy_kj_mol'] for s in samples]
        self.assertEqual(len(absolute), len(relative))
        # The zero reference is the minimum of the absolute curve, and
        # rel = (abs - min) * E_h_kJmol, point for point.
        self.assertAlmostEqual(result['zero_energy_reference_hartree'], min(absolute), places=10)
        for index, (abs_e, rel_e) in enumerate(zip(absolute, relative)):
            self.assertAlmostEqual((abs_e - min(absolute)) * E_h_kJmol, rel_e, places=4,
                                   msg=f'absolute/relative energies disagree at sample {index}')
        # And the ordering is genuinely non-trivial, so the check has teeth.
        self.assertGreater(max(relative) - min(relative), 1.0)

    def test_build_scan_result_no_log(self):
        """Empty scan_path → None, never an exception."""
        rotor = self._rotor(scan_path='')
        self.assertIsNone(_build_scan_result_for_rotor(rotor, '/tmp/project'))

    def test_build_scan_result_missing_log(self):
        """Path that doesn't resolve to a real file → None."""
        rotor = self._rotor(scan_path='/nonexistent/does/not/exist.log')
        self.assertIsNone(_build_scan_result_for_rotor(rotor, '/tmp/project'))

    def test_build_scan_result_malformed_atom_indices(self):
        """Non-quartet ``scan`` field → None (no fabricated atom list)."""
        rotor = self._rotor(scan=[1, 2, 3])  # only 3 atoms
        self.assertIsNone(_build_scan_result_for_rotor(rotor, '/tmp/project'))

    def test_build_scan_result_parser_failure_returns_none(self):
        """Exceptions in the scan-result parser surface as ``None`` (no crash)."""
        with patch('arc.output.parse_1d_scan_full_result',
                   side_effect=Exception('boom')):
            result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNone(result)

    def test_build_scan_result_missing_relative_energies_returns_none(self):
        """Parser returning empty energies → None, even with angles present."""
        with patch('arc.output.parse_1d_scan_full_result',
                   return_value={
                       'angles_deg': [0.0, 90.0],
                       'relative_energies_kj_mol': None,
                       'absolute_energies_hartree': None,
                       'zero_energy_reference_hartree': None,
                       'geometries': None,
                   }):
            result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNone(result)

    def test_build_scan_calculations_emits_one_per_rotor(self):
        """Two successful 1D rotors → two scan_rotor_<i> entries."""
        spc = MagicMock()
        spc.rotors_dict = {0: self._rotor(), 1: self._rotor()}
        calcs = _build_rotor_scans(spc, '/tmp/project')
        self.assertEqual(len(calcs), 2)
        self.assertEqual(calcs[0]['key'], 'scan_rotor_0')
        self.assertEqual(calcs[1]['key'], 'scan_rotor_1')
        self.assertIn('source_log', calcs[0])
        self.assertEqual(calcs[0]['result']['dimension'], 1)

    def test_build_scan_calculations_skips_failed_rotor(self):
        """An unsuccessful rotor (success is not True) is filtered, even with a real scan log.

        ``_build_rotor_scan_entry`` is the only place that decides whether a
        rotor yields a ``rotor_scans`` record, so the consumer cannot drift
        from the producer: a rejected rotor never gets a scan-calculation
        record here, full stop. (Its evidence, when a log survives on disk,
        is instead recorded directly on ``statmech.rejected_torsions[].source_log`` --
        see ``_get_rejected_torsions`` -- without going through this gate.)
        """
        spc = MagicMock()
        spc.rotors_dict = {
            0: self._rotor(),                     # ok
            1: self._rotor(success=False),        # filtered
        }
        calcs = _build_rotor_scans(spc, '/tmp/project')
        self.assertEqual([c['key'] for c in calcs], ['scan_rotor_0'])

    def test_build_scan_calculations_skips_pending_rotor(self):
        """A pending rotor (success=None) is filtered even with a leftover scan path.

        ARC has not decided its fate yet -- a stray ``scan_path`` (e.g. from a
        previous conformer or mid-troubleshooting) is not evidence of
        anything final, so no record is emitted for it.
        """
        spc = MagicMock()
        spc.rotors_dict = {
            0: self._rotor(),                       # success=True, ok
            1: self._rotor(success=None),           # pending -> filtered
        }
        calcs = _build_rotor_scans(spc, '/tmp/project')
        self.assertEqual([c['key'] for c in calcs], ['scan_rotor_0'])

    def test_build_scan_calculations_skips_nd(self):
        """ND rotors are deferred — only 1D scans are emitted today."""
        spc = MagicMock()
        spc.rotors_dict = {
            0: self._rotor(),                  # 1D, ok
            1: self._rotor(dimensions=2),      # ND, skipped
        }
        calcs = _build_rotor_scans(spc, '/tmp/project')
        self.assertEqual([c['key'] for c in calcs], ['scan_rotor_0'])

    def test_build_scan_calculations_skips_unparseable(self):
        """Unparseable scan (no log on disk) → no calc, no exception."""
        spc = MagicMock()
        spc.rotors_dict = {
            0: self._rotor(scan_path=''),  # log missing → skipped
            1: self._rotor(),              # ok
        }
        calcs = _build_rotor_scans(spc, '/tmp/project')
        self.assertEqual([c['key'] for c in calcs], ['scan_rotor_1'])

    def test_get_torsions_attaches_scan_key_when_log_present(self):
        """``source_scan_calculation_key`` matches the scan calc key only when log resolves."""
        spc = MagicMock()
        spc.rotors_dict = {
            0: {
                'success': True,
                'scan': [1, 2, 3, 4],
                'pivots': [2, 3],
                'symmetry': 3,
                'type': 'HinderedRotor',
                'scan_path': self.SCAN_LOG,
                'dimensions': 1,
            },
            7: {  # intentional non-contiguous index — keys must use the dict key.
                'success': True,
                'scan': [5, 6, 7, 8],
                'pivots': [6, 7],
                'symmetry': 1,
                'type': 'HinderedRotor',
                'scan_path': '',
                'dimensions': 1,
            },
        }
        torsions = _get_torsions(spc, '/tmp/project')
        self.assertEqual(len(torsions), 2)
        self.assertEqual(torsions[0]['source_scan_key'], 'scan_rotor_0')
        # Second rotor has no scan log on disk → no fabricated key.
        self.assertIsNone(torsions[1]['source_scan_key'])

    def test_get_rejected_torsions_attaches_source_log_when_invalidated_rotor_has_real_log(self):
        """A rejected rotor with a genuine on-disk scan carries its evidence path directly.

        This is recorded from ``rotors_dict``'s own ``scan_path`` (via
        ``_resolve_scan_path`` / ``_make_rel_path``, the same relative-path
        treatment ``rotor_scans[].source_log`` gets) -- not via a
        ``rotor_scans`` key, since rejected rotors never get a
        ``rotor_scans`` record (see ``test_build_scan_calculations_skips_failed_rotor``).
        """
        spc = MagicMock()
        spc.rotors_dict = {
            0: {
                'success': False,
                'scan': [1, 2, 3, 4],
                'pivots': [2, 3],
                'scan_path': self.SCAN_LOG,
                'dimensions': 1,
                'invalidation_reason': 'rotor set too many (5) times',
            },
            7: {  # intentional non-contiguous index — keys must use the dict key.
                'success': False,
                'scan': [5, 6, 7, 8],
                'pivots': [6, 7],
                'scan_path': '',
                'dimensions': 1,
                'invalidation_reason': '',
            },
        }
        # Use a project_directory that is a real ancestor of SCAN_LOG (rather
        # than the unrelated '/tmp/project' used elsewhere in this class) so
        # the expected relative path is a fixed, portable literal -- not a
        # value recomputed with the implementation's own os.path.relpath
        # call, which would pass no matter what _make_rel_path actually does.
        project_directory = os.path.dirname(ARC_TESTING_PATH)  # .../arc
        rejected = _get_rejected_torsions(spc, project_directory)
        self.assertEqual(len(rejected), 2)
        by_index = {entry['rotor_index']: entry for entry in rejected}
        self.assertEqual(by_index[0]['source_log'], os.path.join('testing', 'rotor_scans', 'sBuOH.out'))
        self.assertEqual(by_index[0]['dimension'], 1)
        # Second rejected rotor has no scan log on disk -> field is omitted, not null.
        self.assertNotIn('source_log', by_index[7])

    def test_get_rejected_torsions_survives_a_missing_on_disk_log(self):
        """``scan_path`` set but the file is gone -> ``source_log`` omitted, rejection still emitted.

        This is D1/(a) from round 4 review: a restart-restored rejected
        rotor's ``scan_path`` is only repaired (see ``arc/main.py``) for
        rotors where ``success`` is truthy, so a rejected rotor's stale
        relative path routinely fails ``_resolve_scan_path``'s
        ``os.path.isfile`` check even though a scan genuinely happened.
        Losing the evidence pointer must never lose the rejection itself.
        """
        spc = MagicMock()
        spc.rotors_dict = {
            0: {
                'success': False,
                'scan': [1, 2, 3, 4],
                'pivots': [2, 3],
                'scan_path': 'rotors/does_not_exist_on_disk.out',
                'dimensions': 1,
                'invalidation_reason': 'rotor scan did not converge after 3 attempts',
            },
        }
        rejected = _get_rejected_torsions(spc, '/tmp/project')
        self.assertEqual(len(rejected), 1)
        entry = rejected[0]
        self.assertEqual(entry['rotor_index'], 0)
        self.assertEqual(entry['invalidation_reason'], 'rotor scan did not converge after 3 attempts')
        self.assertEqual(entry['atom_indices'], [1, 2, 3, 4])
        self.assertEqual(entry['pivot_atoms'], [2, 3])
        self.assertNotIn('source_log', entry)

    def test_get_rejected_torsions_survives_a_directed_scan_failure(self):
        """A directed-scan-failure-shaped rotor is emitted with its reason, despite no ``source_log``.

        This is D1/(b) from round 4 review: ``Scheduler.check_directed_scan``
        (``arc/scheduler.py``) deliberately blanks ``scan_path`` to ``''``
        after a real directed scan fails, while still recording a non-empty
        ``invalidation_reason`` and an ND (list-of-lists) ``scan``/``pivots``
        shape. The record looks identical, by ``source_log`` alone, to a
        rotor that was never scanned -- but the rejection itself, and its
        reason, must still come through.
        """
        spc = MagicMock()
        spc.rotors_dict = {
            0: {
                'success': False,
                'scan': [[1, 2, 3, 4], [5, 6, 7, 8]],
                'pivots': [[2, 3], [6, 7]],
                'scan_path': '',
                'dimensions': 2,
                'invalidation_reason': 'Directed scan is inconsistent. ',
            },
        }
        rejected = _get_rejected_torsions(spc, '/tmp/project')
        self.assertEqual(len(rejected), 1)
        entry = rejected[0]
        self.assertEqual(entry['rotor_index'], 0)
        self.assertEqual(entry['invalidation_reason'], 'Directed scan is inconsistent. ')
        self.assertEqual(entry['atom_indices'], [[1, 2, 3, 4], [5, 6, 7, 8]])
        self.assertEqual(entry['pivot_atoms'], [[2, 3], [6, 7]])
        self.assertEqual(entry['dimension'], 2)
        self.assertNotIn('source_log', entry)

    def test_get_torsions_no_scan_key_for_multidimensional_rotor(self):
        """An ND rotor emits no scan record, so its torsion must not reference one."""
        spc = MagicMock()
        spc.rotors_dict = {0: self._rotor(dimensions=2)}
        self.assertEqual([c['key'] for c in _build_rotor_scans(spc, '/tmp/project')], [])
        torsions = _get_torsions(spc, '/tmp/project')
        self.assertEqual(len(torsions), 1)
        self.assertIsNone(torsions[0]['source_scan_key'])

    def test_get_torsions_no_scan_key_when_scan_log_unparseable(self):
        """A 1D rotor whose log exists but does not parse must not reference a record."""
        handle = tempfile.NamedTemporaryFile(mode='w', suffix='.out', delete=False)
        handle.write('not a scan log\n')
        handle.close()
        self.addCleanup(os.remove, handle.name)
        spc = MagicMock()
        spc.rotors_dict = {0: self._rotor(scan_path=handle.name)}
        self.assertEqual([c['key'] for c in _build_rotor_scans(spc, '/tmp/project')], [])
        torsions = _get_torsions(spc, '/tmp/project')
        self.assertEqual(len(torsions), 1)
        self.assertIsNone(torsions[0]['source_scan_key'])

    def test_every_torsion_scan_key_resolves_to_a_record(self):
        """The contract invariant: no source_scan_key without a matching rotor_scans record."""
        handle = tempfile.NamedTemporaryFile(mode='w', suffix='.out', delete=False)
        handle.write('not a scan log\n')
        handle.close()
        self.addCleanup(os.remove, handle.name)
        spc = MagicMock()
        spc.rotors_dict = {
            0: self._rotor(),                          # 1D, parseable → record
            1: self._rotor(dimensions=2),              # ND → no record
            2: self._rotor(scan_path=handle.name),     # unparseable → no record
            3: self._rotor(scan_path=''),              # no log → no record
        }
        scans = _build_rotor_scans(spc, '/tmp/project')
        emitted = {entry['key'] for entry in scans}
        self.assertEqual(emitted, {'scan_rotor_0'})
        referenced = {t['source_scan_key'] for t in _get_torsions(spc, '/tmp/project')
                      if t['source_scan_key'] is not None}
        self.assertEqual(referenced, emitted)
        self.assertTrue(referenced.issubset(emitted))

    # ---- per-sample scan geometries ----
    #
    # ARC's parser wrapper already returns aligned per-step xyz dicts.
    # ``_build_scan_result_for_rotor`` preserves them as ARC-format text.

    def _stub_parsed(self, *, n_points=3, geometries='aligned', start_dihedral=0.0):
        """Build a parser-wrapper return value with controllable geometry alignment.

        Each frame is a 5-atom geometry whose 1-2-3-4 dihedral is
        ``start_dihedral + 30 * i`` degrees, so the stub carries a real, measurable
        sweep rather than a placeholder.

        ``geometries`` is one of:
          - ``'aligned'``   : list of length n_points, each a valid xyz dict.
          - ``'mismatch'``  : list of length n_points + 1.
          - ``'none'``      : ``None`` (parser returned no geometries).
          - ``'malformed'`` : valid count, but one entry is malformed.
        """
        def frames(count):
            return [self._input_xyz_for_dihedral(start_dihedral + i * 30.0)
                    for i in range(count)]
        if geometries == 'aligned':
            geom_list = frames(n_points)
        elif geometries == 'mismatch':
            geom_list = frames(n_points + 1)
        elif geometries == 'none':
            geom_list = None
        elif geometries == 'malformed':
            geom_list = frames(n_points)
            geom_list[1] = {'symbols': (), 'isotopes': (), 'coords': ()}
        else:
            raise ValueError(geometries)
        return {
            'angles_deg': [i * 30.0 for i in range(n_points)],
            'relative_energies_kj_mol': [0.0] * n_points,
            'absolute_energies_hartree': [-100.0] * n_points,
            'zero_energy_reference_hartree': -100.0,
            'geometries': geom_list,
        }

    def test_scan_points_include_geometry_when_aligned(self):
        """Aligned geometries → every point carries ``geometry.xyz_text``."""
        with patch('arc.output.parse_1d_scan_full_result',
                   return_value=self._stub_parsed(n_points=3)):
            result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        self.assertEqual(len(result['samples']), 3)
        for point in result['samples']:
            self.assertIn('geometry_xyz', point)
            xyz_text = point['geometry_xyz']
            lines = xyz_text.splitlines()
            self.assertEqual(len(lines), 5)

    def test_scan_point_geometry_uses_only_xyz_text_no_db_id(self):
        """No ``geometry_id`` (or any DB id) anywhere under scan_result."""
        with patch('arc.output.parse_1d_scan_full_result',
                   return_value=self._stub_parsed(n_points=3)):
            result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        forbidden = {'geometry_id', 'existing_geometry_id', 'id'}
        for point in result['samples']:
            geom = point.get('geometry_xyz')
            self.assertIsInstance(geom, str)
            for k in forbidden:
                self.assertNotIn(k, point, msg=f"{k} leaked onto scan point")
        # Top-level scan_result also has no DB ids.
        for k in forbidden:
            self.assertNotIn(k, result)

    def test_scan_is_dropped_when_the_parser_returned_no_geometries(self):
        """``angle_degrees`` is measured per point, so no geometries means no scan."""
        with patch('arc.output.parse_1d_scan_full_result',
                   return_value=self._stub_parsed(n_points=3, geometries='none')):
            with self.assertLogs('arc', level='WARNING') as cm:
                result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNone(result)
        self.assertTrue(any('absolute dihedral cannot be measured' in m for m in cm.output),
                        msg=f"expected an unmeasurable-dihedral warning, got: {cm.output}")

    def test_scan_is_dropped_when_geometries_do_not_align_with_the_points(self):
        """A geometry list of the wrong length cannot be paired point-for-point."""
        with patch('arc.output.parse_1d_scan_full_result',
                   return_value=self._stub_parsed(n_points=3, geometries='mismatch')):
            with self.assertLogs('arc', level='WARNING') as cm:
                result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNone(result)
        self.assertTrue(any('do not align' in m for m in cm.output),
                        msg=f"expected an alignment warning, got: {cm.output}")

    # ---- requested scan-grid metadata (TCKDB calc_scan_coordinate fields) ----

    def test_scan_coord_includes_step_size_from_gaussian_header(self):
        """Real Gaussian scan log → step_size + resolution_degrees populated
        from the parsed ModRedundant header (not from the completed-point
        spacing). The fixture log has ``S N 8.0`` in its header."""
        result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        coord = result['coordinate']
        self.assertIn('requested_step_size', coord)
        # ARC writes ``S 360/scan_res scan_res``; for the sBuOH fixture the
        # requested step size is 8 degrees.
        self.assertAlmostEqual(coord['requested_step_size'], 8.0, places=6)

    def test_scan_coord_step_size_independent_from_completed_count(self):
        """``step_count`` reflects the *completed* points, ``step_size`` the
        *requested* grid — they're sourced separately and must not be
        coupled (a partially-failed scan would otherwise emit a misleading
        derived step_size). Spot-check both come from independent data."""
        result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        coord = result['coordinate']
        # step_count is the point count we actually parsed; step_size is
        # the requested grid spacing. Their product covers the requested
        # range only when no points dropped.
        self.assertEqual(coord['sample_count'], len(result['samples']))
        self.assertGreater(coord['requested_step_size'], 0.0)

    def test_scan_coord_omits_grid_metadata_for_non_gaussian(self):
        """``parse_scan_args`` raising NotImplementedError (ORCA, etc.) →
        step_size / resolution_degrees absent, no exception, scan_result
        still produced."""
        with patch('arc.output.parse_scan_args',
                   side_effect=NotImplementedError('ORCA path')):
            result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        coord = result['coordinate']
        self.assertNotIn('requested_step_size', coord)
        self.assertNotIn('resolution_degrees', coord)

    def test_scan_coord_omits_grid_metadata_when_parser_raises(self):
        """Generic parser failure (corrupt log, etc.) → grid fields absent,
        no exception."""
        with patch('arc.output.parse_scan_args',
                   side_effect=RuntimeError('boom')):
            result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        coord = result['coordinate']
        self.assertNotIn('requested_step_size', coord)
        self.assertNotIn('resolution_degrees', coord)

    def test_scan_coord_omits_grid_metadata_when_step_size_zero(self):
        """``parse_scan_args`` returns step_size=0 by default when the
        ModRedundant block isn't matched — must be treated as 'unknown',
        not as a literal 0-degree step (which would be nonsense and
        violate the schema's intent)."""
        stub = {'scan': [1, 2, 3, 4], 'freeze': [], 'step': 0,
                'step_size': 0, 'n_atom': 0}
        with patch('arc.output.parse_scan_args', return_value=stub):
            result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        coord = result['coordinate']
        self.assertNotIn('requested_step_size', coord)
        self.assertNotIn('resolution_degrees', coord)

    def test_scan_coord_grid_metadata_does_not_affect_points(self):
        """Independence: completed-point coordinate_values aren't touched
        by the requested-grid plumbing."""
        with patch('arc.output.parse_scan_args',
                   return_value={'scan': [1, 2, 3, 4], 'freeze': [],
                                 'step': 36, 'step_size': 10.0, 'n_atom': 0}):
            result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        # Points still carry their actual coordinate_value list; step_size
        # didn't propagate into per-point data.
        self.assertGreater(len(result['samples']), 0)
        for point in result['samples']:
            self.assertIn('angle_degrees', point)

    # ---- start_value / end_value (TCKDB requested-grid endpoints) ----
    #
    # The dihedral is read from the input geometry the rotor scan was
    # launched against. Gaussian's ModRedundant ``S`` syntax encodes
    # ``end_value = start_value + step_size * (step_count - 1)``;
    # we emit both values continuous (no [-180, 180] wrap) so a full
    # rotation lands at start + 360, not back at start.

    @staticmethod
    def _input_xyz_for_dihedral(start_dihedral_deg: float):
        """Build a minimal 5-atom xyz whose 1-2-3-4 dihedral, as
        measured by :func:`calculate_dihedral_angle` (the same helper
        the production code uses), equals ``start_dihedral_deg`` in the
        0-360 convention. The internal rotation is offset by -270° to
        compensate for the helper's right-hand-rule sign choice.
        """
        import math as _math
        rad = _math.radians(start_dihedral_deg - 270.0)
        return {
            'symbols': ('C', 'C', 'C', 'C', 'H'),
            'isotopes': (12, 12, 12, 12, 1),
            'coords': (
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, _math.cos(rad), _math.sin(rad)),
                (4.0, 0.0, 0.0),
            ),
        }

    def test_scan_start_value_computed_from_input_geometry(self):
        """Input geometry → ``start_value`` matches the dihedral on the
        scan atom quartet."""
        rotor = self._rotor(scan=[1, 2, 3, 4])
        xyz = self._input_xyz_for_dihedral(60.0)
        result = _build_scan_result_for_rotor(
            rotor, '/tmp/project', input_xyz=xyz,
        )
        coord = result['coordinate']
        self.assertIn('requested_start', coord)
        # The fixture's offset compensates for the helper's right-hand
        # rule, so 60° in → 60° out.
        self.assertAlmostEqual(coord['requested_start'], 60.0, places=4)

    def test_scan_end_value_extends_continuously_from_start(self):
        """``end_value = start_value + step_size * (step_count - 1)`` —
        not wrapped, so a 46-point 8° scan from 60° lands at 60 + 360 = 420°,
        NOT back at 60° and NOT mod-360'd to 60°."""
        rotor = self._rotor(scan=[1, 2, 3, 4])
        xyz = self._input_xyz_for_dihedral(60.0)
        result = _build_scan_result_for_rotor(
            rotor, '/tmp/project', input_xyz=xyz,
        )
        coord = result['coordinate']
        # Real Gaussian fixture: step_size=8, len(points)=46 → +360 span.
        expected_end = coord['requested_start'] + coord['requested_step_size'] * (
            coord['sample_count'] - 1
        )
        self.assertAlmostEqual(coord['requested_end'], expected_end, places=6)
        # Not wrapped: a full-rotation scan exceeds 360°, never re-folds.
        self.assertGreater(coord['requested_end'], 360.0)

    def test_scan_end_value_is_not_wrapped_into_minus_180_180(self):
        """Continuity contract: even when start is near 180°, the end
        value must not flip sign by wrapping into [-180, 180]."""
        rotor = self._rotor(scan=[1, 2, 3, 4])
        xyz = self._input_xyz_for_dihedral(170.0)
        result = _build_scan_result_for_rotor(
            rotor, '/tmp/project', input_xyz=xyz,
        )
        coord = result['coordinate']
        # 170 + 360 = 530 — must NOT have folded to -190 or 170.
        self.assertGreater(coord['requested_end'], 360.0)
        self.assertGreater(coord['requested_end'] - coord['requested_start'],
                           coord['requested_step_size'] * 0.99)

    def test_scan_is_dropped_when_the_real_log_yields_no_geometries(self):
        """Killing the parser's geometries drops the whole scan, not just start/end."""
        rotor = self._rotor(scan=[1, 2, 3, 4])
        with patch('arc.output.parse_1d_scan_full_result') as p:
            from arc.parser.parser import parse_1d_scan_full_result as real
            parsed = real(self.SCAN_LOG)
            parsed['geometries'] = None
            p.return_value = parsed
            result = _build_scan_result_for_rotor(
                rotor, '/tmp/project', input_xyz=None,
            )
        self.assertIsNone(result)

    def test_scan_start_end_absent_when_step_size_unknown(self):
        """Without step_size we can't compute end_value, so we omit BOTH
        rather than emit a half-populated range."""
        rotor = self._rotor(scan=[1, 2, 3, 4])
        xyz = self._input_xyz_for_dihedral(60.0)
        with patch('arc.output.parse_scan_args',
                   side_effect=NotImplementedError('non-Gaussian')):
            result = _build_scan_result_for_rotor(
                rotor, '/tmp/project', input_xyz=xyz,
            )
        coord = result['coordinate']
        self.assertNotIn('requested_step_size', coord)  # confirms the precondition
        self.assertNotIn('requested_start', coord)
        self.assertNotIn('requested_end', coord)

    def test_scan_start_end_absent_when_step_size_zero(self):
        """``parse_scan_args`` returning step_size=0 → no end_value
        possible, omit both."""
        rotor = self._rotor(scan=[1, 2, 3, 4])
        xyz = self._input_xyz_for_dihedral(60.0)
        stub = {'scan': [1, 2, 3, 4], 'freeze': [], 'step': 0,
                'step_size': 0, 'n_atom': 0}
        with patch('arc.output.parse_scan_args', return_value=stub):
            result = _build_scan_result_for_rotor(
                rotor, '/tmp/project', input_xyz=xyz,
            )
        coord = result['coordinate']
        self.assertNotIn('requested_start', coord)
        self.assertNotIn('requested_end', coord)

    def test_scan_is_dropped_when_the_dihedral_calculation_raises(self):
        """No measurable dihedral means no absolute axis, so the scan is not published."""
        rotor = self._rotor(scan=[1, 2, 3, 4])
        xyz = self._input_xyz_for_dihedral(60.0)
        with patch('arc.output.calculate_dihedral_angle',
                   side_effect=RuntimeError('atom missing')):
            with self.assertLogs('arc', level='WARNING') as cm:
                result = _build_scan_result_for_rotor(
                    rotor, '/tmp/project', input_xyz=xyz,
                )
        self.assertIsNone(result)
        self.assertTrue(any('dihedral calculation failed' in m for m in cm.output),
                        msg=f"expected dihedral-failure warning, got: {cm.output}")

    def test_scan_point_coordinate_values_unchanged_by_start_end_addition(self):
        """``points[i].coordinate_values`` must remain whatever
        ``parse_1d_scan_full_result`` reported, regardless of start/end."""
        rotor = self._rotor(scan=[1, 2, 3, 4])
        xyz = self._input_xyz_for_dihedral(60.0)
        result_with = _build_scan_result_for_rotor(
            rotor, '/tmp/project', input_xyz=xyz,
        )
        result_without = _build_scan_result_for_rotor(
            rotor, '/tmp/project', input_xyz=None,
        )
        # Same number of points, same coordinate_values per point.
        self.assertEqual(len(result_with['samples']), len(result_without['samples']))
        for p_with, p_without in zip(result_with['samples'], result_without['samples']):
            self.assertEqual(p_with['angle_degrees'], p_without['angle_degrees'])

    def test_scan_falls_back_to_parsed_first_frame_when_input_xyz_missing(self):
        """When ``input_xyz`` is None but the parser returned aligned
        geometries, the first frame is a documented fallback for the
        input dihedral (Gaussian ModRedundant freezes the scan dihedral
        at the input value, so the first frame's dihedral IS the
        requested start)."""
        rotor = self._rotor(scan=[1, 2, 3, 4])
        result = _build_scan_result_for_rotor(
            rotor, '/tmp/project', input_xyz=None,
        )
        coord = result['coordinate']
        # Real Gaussian fixture has ``geometries`` populated, so the
        # fallback resolves and start/end are emitted.
        self.assertIn('requested_start', coord)
        self.assertIn('requested_end', coord)

    # ---- the samples' angle is the absolute internal coordinate ----

    def test_sample_angle_is_the_absolute_dihedral_of_its_own_geometry(self):
        """Each ``angle_degrees`` equals the dihedral recomputed from that sample's
        geometry, modulo a full turn — the same check a consumer performs on deposit."""
        from arc.parser.parser import parse_1d_scan_full_result
        from arc.species.vectors import calculate_dihedral_angle
        rotor = self._rotor(scan=[1, 2, 3, 4])
        result = _build_scan_result_for_rotor(rotor, '/tmp/project')
        geometries = parse_1d_scan_full_result(self.SCAN_LOG)['geometries']
        self.assertEqual(len(result['samples']), len(geometries))
        for sample, geometry in zip(result['samples'], geometries):
            measured = calculate_dihedral_angle(coords=geometry, torsion=[1, 2, 3, 4], index=1)
            self.assertAlmostEqual(sample['angle_degrees'] % 360.0, measured % 360.0, places=6)

    def test_sample_angle_is_not_the_displacement_from_the_first_sample(self):
        """The published axis must not start at 0 for a scan whose coordinate does not."""
        from arc.parser.parser import parse_1d_scan_full_result
        from arc.species.vectors import calculate_dihedral_angle
        rotor = self._rotor(scan=[1, 2, 3, 4])
        result = _build_scan_result_for_rotor(rotor, '/tmp/project')
        first_geometry = parse_1d_scan_full_result(self.SCAN_LOG)['geometries'][0]
        first_geometry_dihedral = calculate_dihedral_angle(
            coords=first_geometry, torsion=[1, 2, 3, 4], index=1)
        self.assertNotAlmostEqual(result['samples'][0]['angle_degrees'], 0.0, places=3)
        self.assertAlmostEqual(result['samples'][0]['angle_degrees'],
                               first_geometry_dihedral, places=6)

    def test_a_sweep_past_a_full_turn_is_not_wrapped(self):
        """A 360-degree sweep from 350 must end near 710, never fold back below 350."""
        with patch('arc.output.parse_1d_scan_full_result',
                   return_value=self._stub_parsed(n_points=13, start_dihedral=350.0)):
            result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        angles = [sample['angle_degrees'] for sample in result['samples']]
        self.assertEqual(angles, sorted(angles))
        self.assertAlmostEqual(angles[0], 350.0, places=4)
        self.assertAlmostEqual(angles[-1], 710.0, places=4)
        self.assertGreater(angles[-1], 360.0)

    def test_a_reversed_sweep_goes_below_its_start(self):
        """A scan towards decreasing values keeps descending instead of folding to 360."""
        parsed = self._stub_parsed(n_points=5)
        parsed['geometries'] = [self._input_xyz_for_dihedral(60.0 - i * 30.0) for i in range(5)]
        with patch('arc.output.parse_1d_scan_full_result', return_value=parsed):
            result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        angles = [sample['angle_degrees'] for sample in result['samples']]
        self.assertEqual(angles, sorted(angles, reverse=True))
        self.assertAlmostEqual(angles[0], 60.0, places=4)
        self.assertAlmostEqual(angles[-1], -60.0, places=4)
        self.assertLess(angles[-1], 0.0)

    # ---- a reversed Gaussian scan keeps its requested grid ----

    def test_reversed_scan_records_a_signed_step_size_and_a_descending_range(self):
        """A negative ModRedundant step must populate the grid, not blank it out.

        ``requested_step_size`` keeps its sign, so ``requested_end`` falls below
        ``requested_start`` and the scan's direction survives into the document.
        """
        rotor = self._rotor(scan=[1, 2, 3, 4])
        xyz = self._input_xyz_for_dihedral(60.0)
        stub = {'scan': [1, 2, 3, 4], 'freeze': [], 'step': 45,
                'step_size': -8.0, 'n_atom': 0}
        with patch('arc.output.parse_scan_args', return_value=stub):
            result = _build_scan_result_for_rotor(rotor, '/tmp/project', input_xyz=xyz)
        coord = result['coordinate']
        self.assertIn('requested_step_size', coord)
        self.assertIn('requested_start', coord)
        self.assertIn('requested_end', coord)
        self.assertAlmostEqual(coord['requested_step_size'], -8.0, places=6)
        self.assertAlmostEqual(coord['requested_start'], 60.0, places=4)
        self.assertAlmostEqual(coord['requested_end'], -300.0, places=4)
        self.assertLess(coord['requested_end'], coord['requested_start'])

    def test_requested_start_and_end_are_untouched_by_the_sample_axis(self):
        """start/end stay grid metadata read off the launch geometry.

        They describe the requested extent, so they must not be re-derived from, or
        shifted by, the samples' absolute angles.
        """
        rotor = self._rotor(scan=[1, 2, 3, 4])
        xyz = self._input_xyz_for_dihedral(60.0)
        result = _build_scan_result_for_rotor(rotor, '/tmp/project', input_xyz=xyz)
        coord = result['coordinate']
        self.assertAlmostEqual(coord['requested_start'], 60.0, places=4)
        self.assertAlmostEqual(
            coord['requested_end'],
            60.0 + coord['requested_step_size'] * (coord['sample_count'] - 1),
            places=4)
        self.assertNotAlmostEqual(coord['requested_start'],
                                  result['samples'][0]['angle_degrees'], places=3)

    def test_scan_points_omit_geometry_uniformly_on_serialization_failure(self):
        """Unserializable geometry text → drop ``geometry_xyz`` from ALL points, warn
        once, and still publish the scan: the dihedrals were measurable from the same
        geometry dicts, so the absolute axis is intact."""
        with patch('arc.output.parse_1d_scan_full_result',
                   return_value=self._stub_parsed(n_points=3)):
            with patch('arc.output._xyz_dict_to_output_text',
                       side_effect=ValueError('unserializable')):
                with self.assertLogs('arc', level='WARNING') as cm:
                    result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        for point in result['samples']:
            self.assertNotIn('geometry_xyz', point)
            self.assertIn('angle_degrees', point)
        self.assertTrue(
            any('serialization failed' in m or 'empty text' in m for m in cm.output),
            msg=f"expected serialization warning, got: {cm.output}",
        )

    def test_scan_is_dropped_when_one_point_geometry_is_unusable(self):
        """A single unmeasurable frame sinks the scan: a partial axis is not published."""
        with patch('arc.output.parse_1d_scan_full_result',
                   return_value=self._stub_parsed(n_points=3, geometries='malformed')):
            with self.assertLogs('arc', level='WARNING') as cm:
                result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNone(result)
        self.assertTrue(any('Could not measure the dihedral' in m for m in cm.output),
                        msg=f"expected a per-point dihedral warning, got: {cm.output}")


class TestScanConstraintDispatch(unittest.TestCase):
    """Software-aware dispatch for rotor-scan constraint extraction.

    The scheduler stamps ``scan_software`` onto each rotor when a scan
    job completes (``arc/scheduler.py``). ``_parse_scan_constraints``
    consumes that hint to call the right parser; everything else
    degrades gracefully without failing payload generation.
    """

    SCAN_LOG = os.path.join(ARC_TESTING_PATH, 'rotor_scans', 'sBuOH.out')

    def _rotor(self, **overrides) -> dict:
        rotor: dict = {
            'success': True,
            'scan': [1, 2, 3, 4],
            'pivots': [2, 3],
            'symmetry': 1,
            'type': 'HinderedRotor',
            'scan_path': self.SCAN_LOG,
            'dimensions': 1,
            'scan_software': '',
        }
        rotor.update(overrides)
        return rotor

    def test_gaussian_hint_routes_to_gaussian_parser(self):
        from arc.output import _parse_scan_constraints
        sentinel = [{'coordinate_type': 'distance', 'atom_indices': [1, 2],
                     'index_base': 1, 'target_value': None}]
        with patch('arc.output.parse_gaussian_constraints',
                   return_value=sentinel) as gauss, \
             patch('arc.output.parse_orca_constraints') as orca:
            result = _parse_scan_constraints(
                self._rotor(scan_software='gaussian'), '/tmp/project',
            )
        self.assertEqual(result, sentinel)
        gauss.assert_called_once_with(self.SCAN_LOG)
        orca.assert_not_called()

    def test_orca_hint_routes_to_orca_parser(self):
        from arc.output import _parse_scan_constraints
        sentinel = [{'coordinate_type': 'dihedral',
                     'atom_indices': [0, 1, 2, 3], 'index_base': 0,
                     'target_value': 90.0}]
        with patch('arc.output.parse_orca_constraints',
                   return_value=sentinel) as orca, \
             patch('arc.output.parse_gaussian_constraints') as gauss:
            result = _parse_scan_constraints(
                self._rotor(scan_software='orca'), '/tmp/project',
            )
        self.assertEqual(result, sentinel)
        orca.assert_called_once_with(self.SCAN_LOG)
        gauss.assert_not_called()

    def test_missing_software_falls_back_to_gaussian(self):
        # Empty / missing ``scan_software`` preserves the historical
        # behavior: try Gaussian (the only software with ModRedundant
        # emission). Restart files written before this field landed
        # therefore keep producing constraints rather than silently
        # losing them.
        from arc.output import _parse_scan_constraints
        with patch('arc.output.parse_gaussian_constraints',
                   return_value=[]) as gauss:
            rotor_no_field = self._rotor()
            rotor_no_field.pop('scan_software', None)
            _parse_scan_constraints(rotor_no_field, '/tmp/project')
            _parse_scan_constraints(self._rotor(scan_software=''), '/tmp/project')
        self.assertEqual(gauss.call_count, 2)

    def test_unknown_software_returns_empty_list_no_parser_call(self):
        from arc.output import _parse_scan_constraints
        with patch('arc.output.parse_gaussian_constraints') as gauss, \
             patch('arc.output.parse_orca_constraints') as orca:
            result = _parse_scan_constraints(
                self._rotor(scan_software='qchem'), '/tmp/project',
            )
        self.assertEqual(result, [])
        gauss.assert_not_called()
        orca.assert_not_called()

    def test_parser_exception_degrades_to_empty_list(self):
        from arc.output import _parse_scan_constraints
        with patch('arc.output.parse_gaussian_constraints',
                   side_effect=RuntimeError('parser crashed')):
            result = _parse_scan_constraints(
                self._rotor(scan_software='gaussian'), '/tmp/project',
            )
        self.assertEqual(result, [])

    def test_missing_scan_path_returns_empty_list(self):
        # Defensive: never invoke a parser without a real path.
        from arc.output import _parse_scan_constraints
        rotor = self._rotor(scan_path='', scan_software='gaussian')
        with patch('arc.output.parse_gaussian_constraints') as gauss:
            self.assertEqual(_parse_scan_constraints(rotor, '/tmp/project'), [])
            gauss.assert_not_called()


if __name__ == '__main__':
    unittest.main()
