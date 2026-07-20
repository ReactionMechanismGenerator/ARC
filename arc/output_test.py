"""
Tests for the arc.output module (consolidated output.yml writer).
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from arc.common import ARC_PATH
from arc.level import Level
from arc.common import ARC_TESTING_PATH
from arc.output import (
    _build_applied_corrections_for_species,
    _build_scan_calculations,
    _build_scan_result_for_rotor,
    _compute_point_groups,
    _compute_species_corrections,
    _get_arkane_git_commit,
    _get_energy_corrections,
    _get_ess_versions,
    _get_rotor_barrier,
    _get_torsions,
    _get_ts_imag_freq,
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

    def test_thermo_with_thermo_points(self):
        points = [
            {'temperature_k': 300.0, 'cp_j_mol_k': 35.1,
             'h_kj_mol': -50.0, 's_j_mol_k': 200.0, 'g_kj_mol': -110.0},
            {'temperature_k': 400.0, 'cp_j_mol_k': 40.5,
             'h_kj_mol': -45.2, 's_j_mol_k': 215.0, 'g_kj_mol': -131.2},
        ]
        thermo = ThermoData(H298=-10.0, S298=200.0, Tmin=(300, 'K'),
                            Tmax=(2000, 'K'), thermo_points=points)
        result = _thermo_to_dict(thermo)
        self.assertEqual(result['thermo_points'], points)

    def test_tmin_tmax_scalar(self):
        """Tmin/Tmax can be plain numbers (not tuples)."""
        thermo = ThermoData(H298=-10.0, S298=200.0, Tmin=300, Tmax=3000)
        result = _thermo_to_dict(thermo)
        self.assertEqual(result['tmin_k'], 300)
        self.assertEqual(result['tmax_k'], 3000)


class TestGetTsImagFreq(unittest.TestCase):
    """Tests for _get_ts_imag_freq."""

    def test_no_ts_guesses(self):
        spc = MagicMock()
        spc.chosen_ts = None
        spc.ts_guesses = []
        self.assertIsNone(_get_ts_imag_freq(spc))

    def test_valid_imag_freq(self):
        ts_guess = MagicMock()
        ts_guess.imaginary_freqs = [-1500.0, -200.0]
        spc = MagicMock()
        spc.chosen_ts = 0
        spc.ts_guesses = [ts_guess]
        self.assertAlmostEqual(_get_ts_imag_freq(spc), -1500.0)

    def test_chosen_ts_out_of_range(self):
        spc = MagicMock()
        spc.chosen_ts = 5
        spc.ts_guesses = [MagicMock()]
        self.assertIsNone(_get_ts_imag_freq(spc))


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
        """Parse a real Gaussian opt log for step count, final energy, and final xyz."""
        opt_path = os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')
        n_steps, e_hartree, final_xyz = _parse_opt_log(opt_path, '/dummy')
        self.assertEqual(n_steps, 4)
        self.assertIsNotNone(e_hartree)
        self.assertAlmostEqual(e_hartree, -116.986089069, places=6)
        # The geometry is parsed via the shared parse_geometry dispatcher;
        # we just check it produced a non-empty atom-only string.
        self.assertIsNotNone(final_xyz)
        self.assertGreaterEqual(len(final_xyz.splitlines()), 3,
                                msg=f"expected several atom lines, got {final_xyz!r}")

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


class TestGetEnergyCorrections(unittest.TestCase):
    """Tests for _get_energy_corrections."""

    def test_none_level(self):
        aec, bac = _get_energy_corrections(None, 'p')
        self.assertIsNone(aec)
        self.assertIsNone(bac)

    def test_known_level(self):
        lot = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        aec, bac = _get_energy_corrections(lot, 'p')
        if aec is not None:  # only if RMG-database is available
            self.assertIn('H', aec)
            self.assertIn('C', aec)
            self.assertIsInstance(aec['H'], float)
        if bac is not None:
            self.assertIn('C-H', bac)
            self.assertIsInstance(bac['C-H'], float)

    def test_no_bac_when_type_none(self):
        lot = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        aec, bac = _get_energy_corrections(lot, None)
        self.assertIsNone(bac)

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
            aec, bac = _get_energy_corrections(lot, 'p')

        # Verify both sections were searched independently
        self.assertTrue(any('atom_energies' in c for c in calls))
        self.assertTrue(any('pbac' in c for c in calls))
        # Verify the script received separate keys
        save_call = mock_save.call_args
        saved_content = save_call[1].get('content') or save_call[0][1]
        self.assertEqual(saved_content['aec_key'], aec_key)
        self.assertEqual(saved_content['bac_key'], bac_key)
        # Verify results returned
        self.assertIsNotNone(aec)
        self.assertIsNotNone(bac)


class TestGetTsImagFreqFromFreqs(unittest.TestCase):
    """Tests for _get_ts_imag_freq using spc.freqs as primary source."""

    def test_from_spc_freqs(self):
        spc = MagicMock()
        spc.freqs = [-1500.0, 100.0, 200.0, 300.0]
        spc.chosen_ts = None
        spc.ts_guesses = []
        result = _get_ts_imag_freq(spc)
        self.assertAlmostEqual(result, -1500.0)

    def test_most_negative_selected(self):
        spc = MagicMock()
        spc.freqs = [-200.0, -1500.0, 100.0, 300.0]
        spc.chosen_ts = None
        spc.ts_guesses = []
        result = _get_ts_imag_freq(spc)
        self.assertAlmostEqual(result, -1500.0)

    def test_no_negative_freqs(self):
        spc = MagicMock()
        spc.freqs = [100.0, 200.0, 300.0]
        spc.chosen_ts = None
        spc.ts_guesses = []
        self.assertIsNone(_get_ts_imag_freq(spc))


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

    @staticmethod
    def _minimal_ts_mock(label, chosen):
        """Build a ``MagicMock`` species shaped enough for ``_spc_to_dict``
        to walk the TS branch and emit ``neb_log``/``gsm_log``."""
        spc = MagicMock()
        spc.label = label
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
        spc.chosen_ts_method = chosen
        spc.successful_methods = [chosen] if chosen else []
        return spc

    def test_ts_emits_gsm_log_when_paths_gsm_set(self):
        # When the scheduler routed an xtb_gsm log to ``paths['gsm']``
        # (separate slot from ``paths['neb']``), the TS record carries
        # a ``gsm_log`` field populated with the run-relative path. The
        # ``neb_log`` field stays empty/None for the same record so the
        # TCKDB adapter's method-aware gate doesn't see cross-pollination.
        spc = self._minimal_ts_mock(label='TS_gsm', chosen='xTB-GSM')
        gsm_abs = '/abs/calcs/TS_gsm/gsm/stringfile.xyz0000'
        output_dict = {'TS_gsm': {
            'convergence': True,
            'paths': {'irc': [], 'neb': '', 'gsm': gsm_abs},
            'job_types': {},
        }}
        result = _spc_to_dict(spc, output_dict, '/abs')
        # ``_make_rel_path`` strips the project_directory prefix.
        self.assertEqual(result['gsm_log'],
                         'calcs/TS_gsm/gsm/stringfile.xyz0000')
        self.assertIsNone(result['neb_log'])

    def test_ts_emits_neb_log_when_paths_neb_set(self):
        # Mirror of the GSM test: ``paths['neb']`` populated → ``neb_log``
        # filled, ``gsm_log`` stays None. Guards against a regression
        # that would emit both fields from the same path slot.
        spc = self._minimal_ts_mock(label='TS_neb', chosen='orca_neb')
        neb_abs = '/abs/calcs/TS_neb/neb/input.log'
        output_dict = {'TS_neb': {
            'convergence': True,
            'paths': {'irc': [], 'neb': neb_abs, 'gsm': ''},
            'job_types': {},
        }}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertEqual(result['neb_log'], 'calcs/TS_neb/neb/input.log')
        self.assertIsNone(result['gsm_log'])

    def test_ts_emits_neither_log_when_paths_empty(self):
        # Geometry-only TS guess (heuristics/AutoTST/user XYZ): both
        # slots empty, both ``*_log`` fields end up None. The TCKDB
        # adapter's gate then leaves ts_opt edge-less.
        spc = self._minimal_ts_mock(label='TS_geom', chosen='Heuristics')
        # Explicitly: no ts_guesses/chosen_ts → fallback path inert.
        spc.ts_guesses = []
        spc.chosen_ts = None
        output_dict = {'TS_geom': {
            'convergence': True,
            'paths': {'irc': [], 'neb': '', 'gsm': ''},
            'job_types': {},
        }}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNone(result['neb_log'])
        self.assertIsNone(result['gsm_log'])

    # ------------------------------------------------------------------
    # Restart-restored fallback: chosen TSGuess.log_path → *_log when
    # the scheduler's TS-selection write sites didn't re-fire.
    # ------------------------------------------------------------------

    @staticmethod
    def _ts_guess_mock(*, method, log_path):
        """Build a TSGuess-shaped mock with just the attrs the
        ``_spc_to_dict`` fallback reads."""
        g = MagicMock()
        g.method = method
        g.log_path = log_path
        return g

    def test_restart_chosen_gsm_falls_back_to_tsguess_log_path(self):
        # The restart-restored scenario: paths['gsm'] is empty (the
        # scheduler write site bypassed) but the in-memory TSGuess
        # carries the stringfile path. Output must still emit gsm_log
        # so the TCKDB adapter can fire the path_search gate.
        spc = self._minimal_ts_mock(label='TS_gsm_restart', chosen='xtb-gsm')
        spc.ts_guesses = [
            self._ts_guess_mock(method='autotst', log_path=None),
            self._ts_guess_mock(method='xTB-GSM',
                                log_path='/abs/calcs/TS/tsg1/stringfile.xyz0000'),
        ]
        spc.chosen_ts = 1
        output_dict = {'TS_gsm_restart': {
            'convergence': True,
            'paths': {'irc': [], 'neb': '', 'gsm': ''},  # both empty
            'job_types': {},
        }}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertEqual(result['gsm_log'],
                         'calcs/TS/tsg1/stringfile.xyz0000')
        self.assertIsNone(result['neb_log'])

    def test_restart_chosen_neb_falls_back_to_tsguess_log_path(self):
        # Mirror for orca_neb: paths['neb'] empty, TSGuess.log_path set,
        # output must populate neb_log.
        spc = self._minimal_ts_mock(label='TS_neb_restart', chosen='orca_neb')
        spc.ts_guesses = [
            self._ts_guess_mock(method='heuristics', log_path=None),
            self._ts_guess_mock(method='orca_neb',
                                log_path='/abs/calcs/TS/tsg2/input.log'),
        ]
        spc.chosen_ts = 1
        output_dict = {'TS_neb_restart': {
            'convergence': True,
            'paths': {'irc': [], 'neb': '', 'gsm': ''},
            'job_types': {},
        }}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertEqual(result['neb_log'], 'calcs/TS/tsg2/input.log')
        self.assertIsNone(result['gsm_log'])

    def test_chosen_gsm_does_not_populate_neb_log_from_log_path(self):
        # Cross-pollination guard: a GSM stringfile must never end up
        # in neb_log even if neb_log slot is empty and a GSM log_path
        # exists. The TCKDB adapter's gate would otherwise mis-emit
        # type=path_search method=neb pointing at a GSM stringfile.
        spc = self._minimal_ts_mock(label='TS_xpoll', chosen='xtb-gsm')
        spc.ts_guesses = [
            self._ts_guess_mock(method='xTB-GSM',
                                log_path='/abs/calcs/TS/tsg1/stringfile.xyz0000'),
        ]
        spc.chosen_ts = 0
        output_dict = {'TS_xpoll': {
            'convergence': True,
            'paths': {'irc': [], 'neb': '', 'gsm': ''},
            'job_types': {},
        }}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNone(result['neb_log'])
        self.assertEqual(result['gsm_log'],
                         'calcs/TS/tsg1/stringfile.xyz0000')

    def test_chosen_neb_does_not_populate_gsm_log_from_log_path(self):
        # Mirror cross-pollination guard.
        spc = self._minimal_ts_mock(label='TS_xpoll2', chosen='orca_neb')
        spc.ts_guesses = [
            self._ts_guess_mock(method='orca_neb',
                                log_path='/abs/calcs/TS/tsg2/input.log'),
        ]
        spc.chosen_ts = 0
        output_dict = {'TS_xpoll2': {
            'convergence': True,
            'paths': {'irc': [], 'neb': '', 'gsm': ''},
            'job_types': {},
        }}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNone(result['gsm_log'])
        self.assertEqual(result['neb_log'], 'calcs/TS/tsg2/input.log')

    def test_geometry_only_chosen_method_does_not_populate_either_log(self):
        # Even if a TSGuess has a stray log_path, a non-path-search
        # chosen method (heuristics/AutoTST/user/GCN/KinBot) must keep
        # both fields null — the gate in adapter.py would otherwise
        # never see them, but defense-in-depth at the producer side.
        for chosen_method in ('Heuristics', 'AutoTST', 'KinBot', 'GCN', 'user guess 0'):
            spc = self._minimal_ts_mock(label='TS_geom_log',
                                        chosen=chosen_method)
            spc.ts_guesses = [
                self._ts_guess_mock(method=chosen_method,
                                    log_path='/abs/some/stray.log'),
            ]
            spc.chosen_ts = 0
            output_dict = {'TS_geom_log': {
                'convergence': True,
                'paths': {'irc': [], 'neb': '', 'gsm': ''},
                'job_types': {},
            }}
            result = _spc_to_dict(spc, output_dict, '/abs')
            self.assertIsNone(result['neb_log'],
                              msg=f'neb_log leaked for {chosen_method}')
            self.assertIsNone(result['gsm_log'],
                              msg=f'gsm_log leaked for {chosen_method}')

    def test_restart_merged_geometry_primary_recovers_gsm_source(self):
        # Dedup-merged (benchmark reaction_06): the chosen guess's primary
        # method is geometry-only (gcn) but xtb-gsm merged into it during
        # clustering, carrying a preserved log in ``method_source_paths``.
        # On restart (paths empty), output must recover ``gsm_log`` from
        # the merged source, not from the geometry-only primary method.
        spc = self._minimal_ts_mock(label='TS_merged', chosen='gcn')
        gcn = TSGuess(index=0, method='gcn', success=True, xyz='C 0.0 0.0 0.0')
        gcn.method_sources = ['gcn', 'xtb-gsm']
        gcn.method_source_paths = {'xtb-gsm': '/abs/calcs/TS/tsg/stringfile.xyz0000'}
        spc.ts_guesses = [gcn]
        spc.chosen_ts = 0
        output_dict = {'TS_merged': {
            'convergence': True,
            'paths': {'irc': [], 'neb': '', 'gsm': ''},
            'job_types': {},
        }}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertEqual(result['gsm_log'], 'calcs/TS/tsg/stringfile.xyz0000')
        self.assertIsNone(result['neb_log'])

    def test_restart_merged_sets_exactly_one_path_log_field(self):
        # A guess merging BOTH xtb-gsm and orca_neb must not populate two
        # path-log fields — mirror the scheduler's single-slot invariant
        # (first path source in method_sources order wins).
        spc = self._minimal_ts_mock(label='TS_merged2', chosen='gcn')
        gcn = TSGuess(index=0, method='gcn', success=True, xyz='C 0.0 0.0 0.0')
        gcn.method_sources = ['gcn', 'xtb-gsm', 'orca_neb']
        gcn.method_source_paths = {'xtb-gsm': '/abs/calcs/TS/tsg/stringfile.xyz0000',
                                   'orca_neb': '/abs/calcs/TS/tsg/input.log'}
        spc.ts_guesses = [gcn]
        spc.chosen_ts = 0
        output_dict = {'TS_merged2': {
            'convergence': True,
            'paths': {'irc': [], 'neb': '', 'gsm': ''},
            'job_types': {},
        }}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertEqual(result['gsm_log'], 'calcs/TS/tsg/stringfile.xyz0000')
        self.assertIsNone(result['neb_log'])

    def test_paths_slot_wins_over_tsguess_log_path(self):
        # When both are populated (live-scheduler scenario), the
        # ``paths`` slot is the source of truth (it's what the
        # scheduler explicitly wrote). The fallback only fires when
        # the slot is empty.
        spc = self._minimal_ts_mock(label='TS_both', chosen='xtb-gsm')
        spc.ts_guesses = [
            self._ts_guess_mock(method='xTB-GSM',
                                log_path='/abs/from/tsguess.xyz0000'),
        ]
        spc.chosen_ts = 0
        output_dict = {'TS_both': {
            'convergence': True,
            'paths': {'irc': [], 'neb': '',
                      'gsm': '/abs/from/scheduler.xyz0000'},
            'job_types': {},
        }}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertEqual(result['gsm_log'], 'from/scheduler.xyz0000')


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

    def test_tunneling_method_defaults_to_arkane_template_constant(self):
        # ARC writes ``tunneling='Eckart'`` into every Arkane reaction()
        # block (see ARKANE_TUNNELING_METHOD). _rxn_to_dict must surface
        # that decision in the kinetics block so downstream consumers
        # (TCKDB, analysis) know which correction was applied to the fit.
        from arc.statmech.arkane import ARKANE_TUNNELING_METHOD
        rxn = MagicMock()
        rxn.label = 'A <=> B'
        rxn.reactants = ['A']
        rxn.products = ['B']
        rxn.family = None
        rxn.multiplicity = 1
        rxn.ts_label = 'TS0'
        rxn.kinetics = {
            'A': (1.0e10, 's^-1'),
            'n': 0.0,
            'Ea': (10.0, 'kJ/mol'),
            'Tmin': (300, 'K'), 'Tmax': (2000, 'K'),
        }
        result = _rxn_to_dict(rxn)
        self.assertEqual(result['kinetics']['tunneling'], ARKANE_TUNNELING_METHOD)

    def test_tunneling_method_from_parsed_kinetics_wins(self):
        # If Arkane ever surfaces an explicit tunneling marker on the
        # parsed kinetics dict, prefer that over the template constant.
        # Future-proofs the producer against per-reaction tunneling
        # configs without forcing a template-constant change.
        rxn = MagicMock()
        rxn.label = 'A <=> B'
        rxn.reactants = ['A']
        rxn.products = ['B']
        rxn.family = None
        rxn.multiplicity = 1
        rxn.ts_label = 'TS0'
        rxn.kinetics = {
            'A': (1.0e10, 's^-1'), 'n': 0.0, 'Ea': (10.0, 'kJ/mol'),
            'Tmin': (300, 'K'), 'Tmax': (2000, 'K'),
            'tunneling': 'Wigner',
        }
        result = _rxn_to_dict(rxn)
        self.assertEqual(result['kinetics']['tunneling'], 'Wigner')


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

    def test_monoatomic_synthesizes_xyz_when_geometry_missing(self):
        # ARC skips opt for atoms (nothing to optimize), so final_xyz/initial_xyz
        # may both be None even when the species converged via SP. The producer
        # must still emit a usable xyz so downstream consumers (e.g. TCKDB) can
        # build a geometry payload.
        spc = self._make_spc_mock(label='H_atom', monoatomic=True)
        spc.final_xyz = None
        spc.initial_xyz = None
        spc.freqs = None
        spc.thermo = None
        atom = MagicMock()
        atom.element.symbol = 'H'
        spc.mol.atoms = [atom]
        spc.mol.get_formula.return_value = 'H'
        output_dict = {'H_atom': {'convergence': True, 'paths': {}, 'job_types': {}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNotNone(result['xyz'])
        self.assertIn('H', result['xyz'])
        self.assertIn('0.00000000', result['xyz'])

    def test_polyatomic_with_missing_geometry_keeps_xyz_none(self):
        # Negative case: a non-monoatomic species missing both final_xyz and
        # initial_xyz must NOT get a synthesized geometry — there's no unique
        # one to choose.
        spc = self._make_spc_mock(label='CH4')
        spc.final_xyz = None
        spc.initial_xyz = None
        # _make_spc_mock leaves spc.mol.atoms unset (a MagicMock attribute), so
        # set len-able multi-atom contents to make the negative case explicit.
        spc.mol.atoms = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        output_dict = {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNone(result['xyz'])

    def test_unconverged_monoatomic_keeps_xyz_none(self):
        # Negative case: don't synthesize geometry for unconverged species, even
        # if monoatomic — output.yml only carries geometry for results we trust.
        spc = self._make_spc_mock(label='H_atom', monoatomic=True, converged=False)
        spc.final_xyz = None
        spc.initial_xyz = None
        spc.freqs = None
        spc.thermo = None
        atom = MagicMock()
        atom.element.symbol = 'H'
        spc.mol.atoms = [atom]
        output_dict = {'H_atom': {'convergence': False, 'paths': {}, 'job_types': {}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNone(result['xyz'])

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

    # ------------------------------------------------------------------
    # Input-deck path emission (`<job>_input` keys).
    # ------------------------------------------------------------------

    def test_input_paths_default_none_when_no_software_info(self):
        """Back-compat: callers that don't pass ``software_by_job`` get None."""
        spc = self._make_spc_mock()
        output_dict = {'CH4': {
            'convergence': True,
            'paths': {'geo': '/abs/opt.log', 'freq': '/abs/freq.log', 'sp': '/abs/sp.log'},
            'job_types': {'opt': True},
        }}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNone(result['opt_input'])
        self.assertIsNone(result['freq_input'])
        self.assertIsNone(result['sp_input'])

    def test_input_path_emitted_when_file_exists(self):
        """Gaussian's input.gjf next to opt.log → opt_input populated, project-relative."""
        proj = tempfile.mkdtemp(prefix='arc-output-test-')
        self.addCleanup(shutil.rmtree, proj, ignore_errors=True)
        opt_dir = os.path.join(proj, 'calcs', 'CH4', 'opt')
        os.makedirs(opt_dir, exist_ok=True)
        opt_log = os.path.join(opt_dir, 'input.log')
        opt_inp = os.path.join(opt_dir, 'input.gjf')
        for p in (opt_log, opt_inp):
            with open(p, 'w') as f:
                f.write('x')
        spc = self._make_spc_mock()
        output_dict = {'CH4': {
            'convergence': True,
            'paths': {'geo': opt_log},
            'job_types': {'opt': True},
        }}
        result = _spc_to_dict(
            spc, output_dict, proj,
            software_by_job={'opt': 'gaussian', 'freq': None, 'sp': None},
        )
        self.assertEqual(result['opt_input'], 'calcs/CH4/opt/input.gjf')

    def test_input_path_none_when_input_file_missing(self):
        """Software is known, log is on disk, but input deck isn't → None (no ghost path)."""
        proj = tempfile.mkdtemp(prefix='arc-output-test-')
        self.addCleanup(shutil.rmtree, proj, ignore_errors=True)
        opt_dir = os.path.join(proj, 'calcs', 'CH4', 'opt')
        os.makedirs(opt_dir, exist_ok=True)
        opt_log = os.path.join(opt_dir, 'input.log')
        with open(opt_log, 'w') as f:
            f.write('x')
        # no input.gjf written
        spc = self._make_spc_mock()
        output_dict = {'CH4': {
            'convergence': True,
            'paths': {'geo': opt_log},
            'job_types': {'opt': True},
        }}
        result = _spc_to_dict(
            spc, output_dict, proj,
            software_by_job={'opt': 'gaussian'},
        )
        self.assertIsNone(result['opt_input'])

    def test_input_path_uses_software_specific_filename(self):
        """orca → input.in, cfour → ZMAT — driven by settings['input_filenames']."""
        proj = tempfile.mkdtemp(prefix='arc-output-test-')
        self.addCleanup(shutil.rmtree, proj, ignore_errors=True)
        # opt: orca run, deck is input.in
        opt_dir = os.path.join(proj, 'calcs', 'CH4', 'opt')
        os.makedirs(opt_dir, exist_ok=True)
        opt_log = os.path.join(opt_dir, 'input.log')
        opt_inp = os.path.join(opt_dir, 'input.in')
        for p in (opt_log, opt_inp):
            open(p, 'w').close()
        # sp: cfour run, deck is ZMAT
        sp_dir = os.path.join(proj, 'calcs', 'CH4', 'sp')
        os.makedirs(sp_dir, exist_ok=True)
        sp_log = os.path.join(sp_dir, 'output.out')
        sp_inp = os.path.join(sp_dir, 'ZMAT')
        for p in (sp_log, sp_inp):
            open(p, 'w').close()
        spc = self._make_spc_mock()
        output_dict = {'CH4': {
            'convergence': True,
            'paths': {'geo': opt_log, 'sp': sp_log},
            'job_types': {'opt': True},
        }}
        result = _spc_to_dict(
            spc, output_dict, proj,
            software_by_job={'opt': 'orca', 'sp': 'cfour'},
        )
        self.assertEqual(result['opt_input'], 'calcs/CH4/opt/input.in')
        self.assertEqual(result['sp_input'], 'calcs/CH4/sp/ZMAT')

    def test_input_path_none_when_log_missing(self):
        """No log path → no input path, regardless of software."""
        spc = self._make_spc_mock()
        output_dict = {'CH4': {
            'convergence': True,
            'paths': {},  # no geo/freq/sp
            'job_types': {'opt': True},
        }}
        result = _spc_to_dict(
            spc, output_dict, '/abs',
            software_by_job={'opt': 'gaussian', 'freq': 'gaussian', 'sp': 'gaussian'},
        )
        self.assertIsNone(result['opt_input'])
        self.assertIsNone(result['freq_input'])
        self.assertIsNone(result['sp_input'])

    def test_input_path_none_when_software_unknown(self):
        """Software not in settings['input_filenames'] (e.g., gcn) → None."""
        proj = tempfile.mkdtemp(prefix='arc-output-test-')
        self.addCleanup(shutil.rmtree, proj, ignore_errors=True)
        opt_dir = os.path.join(proj, 'calcs', 'CH4', 'opt')
        os.makedirs(opt_dir, exist_ok=True)
        opt_log = os.path.join(opt_dir, 'output.yml')
        open(opt_log, 'w').close()
        spc = self._make_spc_mock()
        output_dict = {'CH4': {
            'convergence': True,
            'paths': {'geo': opt_log},
            'job_types': {'opt': True},
        }}
        result = _spc_to_dict(
            spc, output_dict, proj,
            software_by_job={'opt': 'gcn'},  # gcn has no entry in input_filenames
        )
        self.assertIsNone(result['opt_input'])

    # ------------------------------------------------------------------
    # opt_input_xyz: pre-opt geometry surfaced for opt's input-geometry
    # provenance. freq + sp share the conformer's converged geometry by
    # ARC's invariant; only opt has a distinct input.
    # ------------------------------------------------------------------

    def test_opt_input_xyz_emitted_from_initial_xyz(self):
        """``spc.initial_xyz`` lands as ``opt_input_xyz`` in xyz_to_str format."""
        spc = self._make_spc_mock()
        spc.initial_xyz = {
            'symbols': ('C', 'H'),
            'isotopes': (12, 1),
            'coords': ((0.001, 0.002, 0.003), (1.090, 0.000, 0.000)),
        }
        output_dict = {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNotNone(result['opt_input_xyz'])
        # Same atom-only string format as the existing ``xyz`` field.
        lines = result['opt_input_xyz'].splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith('C '))
        self.assertTrue(lines[1].startswith('H '))

    def test_opt_input_xyz_distinct_from_xyz_when_both_present(self):
        """``xyz`` carries final, ``opt_input_xyz`` carries initial. Different
        coordinates → different strings; the bundle's `input_geometries`
        link for opt is genuinely separate from the conformer geometry."""
        spc = self._make_spc_mock()
        spc.initial_xyz = {
            'symbols': ('C',),
            'isotopes': (12,),
            'coords': ((0.001, 0.0, 0.0),),
        }
        spc.final_xyz = {
            'symbols': ('C',),
            'isotopes': (12,),
            'coords': ((0.500, 0.0, 0.0),),
        }
        output_dict = {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNotNone(result['xyz'])
        self.assertIsNotNone(result['opt_input_xyz'])
        self.assertNotEqual(result['xyz'], result['opt_input_xyz'])

    def test_opt_input_xyz_none_when_initial_xyz_absent(self):
        """Species with no initial_xyz set → ``opt_input_xyz`` is null."""
        spc = self._make_spc_mock()
        spc.initial_xyz = None
        output_dict = {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNone(result['opt_input_xyz'])

    def test_opt_input_xyz_emitted_independently_of_convergence(self):
        """Opt's input is meaningful even on failed runs — surface it
        regardless of the species convergence flag."""
        spc = self._make_spc_mock(converged=False)
        spc.initial_xyz = {
            'symbols': ('O',),
            'isotopes': (16,),
            'coords': ((0.0, 0.0, 0.0),),
        }
        output_dict = {'CH4': {'convergence': False, 'paths': {}, 'job_types': {}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNotNone(result['opt_input_xyz'])

    # ------------------------------------------------------------------
    # coarse → fine opt geometry chain. When coarse runs, opt_input_xyz
    # changes meaning from "pre-everything" to "fine opt's input" =
    # coarse opt's output.
    # ------------------------------------------------------------------

    def test_no_coarse_opt_keeps_single_stage_semantics(self):
        """No coarse log → coarse_opt_* fields are null; opt_input_xyz
        reflects the species' initial xyz (the single-stage opt's input)."""
        spc = self._make_spc_mock()
        spc.initial_xyz = {'symbols': ('C',), 'isotopes': (12,),
                           'coords': ((0.0, 0.0, 0.0),)}
        output_dict = {'CH4': {'convergence': True,
                                'paths': {},   # no geo_coarse
                                'job_types': {'opt': True}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNone(result['coarse_opt_log'])
        self.assertIsNone(result['coarse_opt_input_xyz'])
        self.assertIsNone(result['coarse_opt_output_xyz'])
        # Single-stage: opt_input_xyz comes from spc.initial_xyz.
        self.assertIsNotNone(result['opt_input_xyz'])
        self.assertEqual(result['opt_input_xyz'].split()[0], 'C')

    def test_coarse_opt_chains_geometries_to_fine(self):
        """When coarse log parses cleanly, the geometry chain is:
        spc.initial_xyz → coarse_opt_input_xyz → coarse_opt_output_xyz =
        opt_input_xyz → xyz."""
        # Use the shipped Gaussian iC3H7 opt log — parse_geometry handles
        # it via the per-ESS adapter, so we get a real xyz back.
        coarse_log = os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')
        spc = self._make_spc_mock()
        spc.initial_xyz = {'symbols': ('C',), 'isotopes': (12,),
                           'coords': ((9.999, 9.999, 9.999),)}  # distinctive
        output_dict = {'CH4': {'convergence': True,
                                'paths': {'geo_coarse': coarse_log,
                                          'geo': coarse_log},  # both point at same file for this test
                                'job_types': {'opt': True}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        # Coarse fields populated.
        self.assertIsNotNone(result['coarse_opt_log'])
        self.assertIsNotNone(result['coarse_opt_input_xyz'])
        self.assertIsNotNone(result['coarse_opt_output_xyz'])
        # Coarse input == species initial xyz.
        self.assertIn('9.999', result['coarse_opt_input_xyz'])
        # Coarse output != initial xyz (it was actually parsed from the log).
        self.assertNotIn('9.999', result['coarse_opt_output_xyz'])
        # opt_input_xyz now points at the coarse output, not initial xyz.
        self.assertEqual(result['opt_input_xyz'], result['coarse_opt_output_xyz'])

    def test_final_settings_emitted_when_coarse_stage_ran(self):
        """When a coarse opt ran, the fine opt is the ``"fine"`` stage
        and the coarse opt is the ``"coarse"`` stage of ARC's two-stage
        convention. Surfacing it as ``optimization_stage`` (rather than
        an ESS-shaped ``fine: bool``) keeps the meaning self-evident
        and avoids collision with future ESS "fine" keywords landing
        in the same dict.
        """
        coarse_log = os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')
        spc = self._make_spc_mock()
        spc.initial_xyz = {'symbols': ('C',), 'isotopes': (12,),
                           'coords': ((0.0, 0.0, 0.0),)}
        output_dict = {'CH4': {'convergence': True,
                                'paths': {'geo_coarse': coarse_log,
                                          'geo': coarse_log},
                                'job_types': {'opt': True}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertEqual(result['opt_final_settings'],
                         {'optimization_stage': 'fine'})
        self.assertEqual(result['coarse_opt_final_settings'],
                         {'optimization_stage': 'coarse'})

    def test_final_settings_null_when_single_stage_opt(self):
        """Single-stage opt: filesystem state alone can't prove whether
        ``fine=True`` or ``fine=False`` was passed to the job, so the
        producer leaves the field None — better than fabricating a
        default."""
        spc = self._make_spc_mock()
        spc.initial_xyz = {'symbols': ('C',), 'isotopes': (12,),
                           'coords': ((0.0, 0.0, 0.0),)}
        output_dict = {'CH4': {'convergence': True,
                                'paths': {},
                                'job_types': {'opt': True}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNone(result['opt_final_settings'])
        self.assertIsNone(result['coarse_opt_final_settings'])
        # freq/sp have no producer-side honest source today either.
        self.assertIsNone(result['freq_final_settings'])
        self.assertIsNone(result['sp_final_settings'])

    def test_coarse_opt_unparseable_geometry_falls_back_safely(self):
        """If the coarse log exists but its geometry can't be parsed, we
        emit no coarse_opt_output_xyz and fall back to single-stage
        semantics for opt_input_xyz (= spc.initial_xyz). A bundle
        downstream won't emit a structured opt_coarse calc in this case."""
        # An unparseable file (empty), but its existence triggers the
        # coarse-opt-ran branch.
        proj = tempfile.mkdtemp(prefix='arc-coarse-fallback-')
        self.addCleanup(shutil.rmtree, proj, ignore_errors=True)
        empty_log = os.path.join(proj, 'coarse.log')
        open(empty_log, 'w').close()
        spc = self._make_spc_mock()
        spc.initial_xyz = {'symbols': ('C',), 'isotopes': (12,),
                           'coords': ((0.0, 0.0, 0.0),)}
        output_dict = {'CH4': {'convergence': True,
                                'paths': {'geo_coarse': empty_log},
                                'job_types': {'opt': True}}}
        result = _spc_to_dict(spc, output_dict, proj)
        # coarse_opt_log path is recorded (it exists) but the parsed
        # geometry is None, so the chain-aware fields stay null.
        self.assertIsNotNone(result['coarse_opt_log'])
        self.assertIsNone(result['coarse_opt_output_xyz'])
        self.assertIsNone(result['coarse_opt_input_xyz'])
        # Fallback: opt_input_xyz comes from initial_xyz, not from the
        # missing coarse output.
        self.assertIsNotNone(result['opt_input_xyz'])


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
    @patch('arc.output.settings', {'RMG_ENV_NAME': 'rmg_env'})
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
    @patch('arc.output.settings', {'RMG_ENV_NAME': 'rmg_env'})
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
    @patch('arc.output.settings', {'RMG_ENV_NAME': 'rmg_env'})
    def test_graceful_failure(self, mock_exec):
        species_dict = {
            'H2O': self._make_spc('H2O', ['O', 'H', 'H'], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]]),
        }
        result = _compute_point_groups(species_dict, self.tmp_dir)
        self.assertEqual(result, {})

    @patch('arc.output.execute_command')
    @patch('arc.output.settings', {'RMG_ENV_NAME': 'rmg_env'})
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
    @patch('arc.output.settings', {'RMG_ENV_NAME': 'rmg_env'})
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
    def test_evidence_failure_still_writes_output_without_descriptor(self, mock_build, mock_pg):
        spc = self._make_spc_mock()
        write_output_yml(
            project='evidence_failure', project_directory=self.tmp_dir,
            species_dict={'CH4': spc}, reactions=[],
            output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
        )
        from arc.common import read_yaml_file
        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        self.assertEqual(doc['schema_version'], '1.1')
        self.assertNotIn('tckdb_evidence', doc)

    @patch('arc.output._compute_point_groups', return_value={})
    def test_evidence_replaced_before_output(self, mock_pg):
        spc = self._make_spc_mock()
        real_replace = os.replace
        destinations = []

        def recording_replace(source, destination):
            destinations.append(os.path.basename(destination))
            return real_replace(source, destination)

        with patch('os.replace', side_effect=recording_replace):
            write_output_yml(
                project='replace_order', project_directory=self.tmp_dir,
                species_dict={'CH4': spc}, reactions=[],
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
    """Direct tests for `_build_applied_corrections_for_species`.

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
        out = _build_applied_corrections_for_species('CH4', sc, self._lot(), 'p')
        roles = [e['application_role'] for e in out]
        self.assertIn('aec_total', roles)
        aec = next(e for e in out if e['application_role'] == 'aec_total')
        self.assertAlmostEqual(aec['value'], -0.0234)
        self.assertEqual(aec['value_unit'], 'hartree')
        self.assertEqual(aec['scheme']['kind'], 'atom_energy')
        self.assertEqual(aec['scheme']['name'], 'atom_energy')

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
        out = _build_applied_corrections_for_species('X', sc, self._lot(), None)
        aec = next(e for e in out if e['application_role'] == 'aec_total')
        total = sum(c['contribution_value'] for c in aec['components'])
        self.assertAlmostEqual(total, aec['value'], places=6)

    def test_pbac_total_and_components(self):
        sc = {'CH4': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_applied_corrections_for_species('CH4', sc, self._lot(), 'p')
        bac = next(e for e in out if e['application_role'] == 'bac_total')
        self.assertEqual(bac['scheme']['kind'], 'bac_petersson')
        self.assertEqual(bac['value_unit'], 'kcal_mol')
        self.assertEqual(len(bac['components']), 1)
        self.assertEqual(bac['components'][0]['key'], 'C-H')

    def test_mbac_total_only_no_components(self):
        sc = {'CH4': {'aec': self._aec_block(), 'bac': self._mbac_block()}}
        out = _build_applied_corrections_for_species('CH4', sc, self._lot(), 'm')
        bac = next(e for e in out if e['application_role'] == 'bac_total')
        self.assertEqual(bac['scheme']['kind'], 'bac_melius')
        self.assertEqual(bac['components'], [])

    def test_pbac_omits_components_when_param_missing(self):
        block = self._pbac_block()
        block['components'][0]['parameter_value'] = None
        sc = {'X': {'aec': self._aec_block(), 'bac': block}}
        out = _build_applied_corrections_for_species('X', sc, self._lot(), 'p')
        bac = next(e for e in out if e['application_role'] == 'bac_total')
        # Components dropped entirely (partial decomposition would mislead).
        self.assertEqual(bac['components'], [])

    def test_units_are_explicit(self):
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_applied_corrections_for_species('X', sc, self._lot(), 'p')
        units = {e['application_role']: e['value_unit'] for e in out}
        self.assertEqual(units['aec_total'], 'hartree')
        self.assertEqual(units['bac_total'], 'kcal_mol')

    def test_missing_correction_omits_silently(self):
        # AEC failed (no 'aec' key), BAC succeeded → only BAC emitted.
        sc = {'X': {'bac': self._pbac_block()}}
        out = _build_applied_corrections_for_species('X', sc, self._lot(), 'p')
        roles = [e['application_role'] for e in out]
        self.assertEqual(roles, ['bac_total'])

    def test_no_data_returns_empty_list(self):
        out = _build_applied_corrections_for_species('X', {}, self._lot(), 'p')
        self.assertEqual(out, [])

    def test_bac_type_none_omits_bac(self):
        # Even if a BAC block is present, bac_type=None means no BAC role.
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_applied_corrections_for_species('X', sc, self._lot(), None)
        roles = [e['application_role'] for e in out]
        self.assertEqual(roles, ['aec_total'])

    # ---- scheme parameter tables (atom_params / bond_params) ----

    def test_aec_scheme_includes_atom_params_from_run_table(self):
        # ARC's run-level atom_energy_corrections dict is the source of
        # truth for AEC scheme parameters; without atom_params the
        # downstream energy_correction_scheme_atom_param table never gets
        # populated even though the applied row lands. Sorted-by-element
        # for deterministic output.yml.
        aec_table = {'C': -37.84706, 'H': -0.50066}
        sc = {'X': {'aec': self._aec_block()}}
        out = _build_applied_corrections_for_species(
            'X', sc, self._lot(), 'p', aec_table=aec_table, bac_table=None,
        )
        aec = next(e for e in out if e['application_role'] == 'aec_total')
        self.assertEqual(
            aec['scheme']['atom_params'],
            [{'element': 'C', 'value': -37.84706},
             {'element': 'H', 'value': -0.50066}],
        )

    def test_pbac_scheme_includes_bond_params_from_run_table(self):
        bac_table = {'C-H': -0.17350, 'C=O': -2.63454}
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_applied_corrections_for_species(
            'X', sc, self._lot(), 'p', aec_table=None, bac_table=bac_table,
        )
        bac = next(e for e in out if e['application_role'] == 'bac_total')
        self.assertEqual(
            bac['scheme']['bond_params'],
            [{'bond_key': 'C-H', 'value': -0.17350},
             {'bond_key': 'C=O', 'value': -2.63454}],
        )

    def test_mbac_scheme_omits_params(self):
        # Per spec: Melius BAC parameters are atom-pair / length / neighbor /
        # molecular and don't fit SchemeBondParamPayload's bond-key shape.
        # The producer must NOT fabricate or coerce them — emit total only.
        bac_table = {'C-H': -0.17350}  # would coerce, but we must not
        sc = {'X': {'aec': self._aec_block(), 'bac': self._mbac_block()}}
        out = _build_applied_corrections_for_species(
            'X', sc, self._lot(), 'm', aec_table=None, bac_table=bac_table,
        )
        bac = next(e for e in out if e['application_role'] == 'bac_total')
        self.assertEqual(bac['scheme']['kind'], 'bac_melius')
        self.assertNotIn('bond_params', bac['scheme'])
        self.assertNotIn('atom_params', bac['scheme'])
        self.assertNotIn('component_params', bac['scheme'])

    def test_aec_scheme_omits_atom_params_when_table_missing(self):
        # Backward compat: when aec_table isn't supplied (caller predates
        # this fix, or output.yml was written without it), the scheme still
        # has identity but no atom_params field — schema treats it as []
        # via the default factory.
        sc = {'X': {'aec': self._aec_block()}}
        out = _build_applied_corrections_for_species(
            'X', sc, self._lot(), 'p', aec_table=None, bac_table=None,
        )
        aec = next(e for e in out if e['application_role'] == 'aec_total')
        self.assertNotIn('atom_params', aec['scheme'])
        self.assertNotIn('bond_params', aec['scheme'])

    def test_atom_params_sorted_for_determinism(self):
        # Stable insertion order matters for the idempotency hash
        # downstream consumers compute over the payload.
        aec_table = {'O': -75.07, 'H': -0.5, 'C': -37.85}
        sc = {'X': {'aec': self._aec_block()}}
        out = _build_applied_corrections_for_species(
            'X', sc, self._lot(), None, aec_table=aec_table, bac_table=None,
        )
        aec = next(e for e in out if e['application_role'] == 'aec_total')
        elements = [p['element'] for p in aec['scheme']['atom_params']]
        self.assertEqual(elements, ['C', 'H', 'O'])  # sorted


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

    def test_returns_empty_when_lot_is_none(self):
        out = _compute_species_corrections({'CH4': self._spc()}, None, 'p', self.tmp_dir)
        self.assertEqual(out, {})

    def _patch_lot_key(self, key="LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='qchem')"):
        return [
            patch('arc.output.get_qm_corrections_files', return_value=['/fake/data.py']),
            patch('arc.output.find_best_across_files', return_value=key),
        ]

    def test_returns_empty_when_no_species_have_xyz(self):
        spc = self._spc()
        spc.final_xyz = None
        spc.initial_xyz = None
        lot = Level(method='wb97xd3', basis='def2tzvp', software='qchem')
        patches = self._patch_lot_key()
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        with patch('arc.output.execute_command') as mock_exec:
            out = _compute_species_corrections({'CH4': spc}, lot, 'p', self.tmp_dir)
        self.assertEqual(out, {})
        mock_exec.assert_not_called()

    def test_invokes_subprocess_with_batched_input(self):
        lot = Level(method='wb97xd3', basis='def2tzvp', software='qchem')
        patches = self._patch_lot_key()
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        with patch('arc.output.execute_command', return_value=('', '')) as mock_exec, \
             patch('arc.output.read_yaml_file', return_value={'species': [
                 {'label': 'CH4',
                  'aec': {'value': -0.02, 'value_unit': 'hartree', 'components': []},
                  'bac': {'value': -0.7, 'value_unit': 'kcal_mol', 'components': []}}
             ]}), \
             patch('arc.output.save_yaml_file') as mock_save:
            out = _compute_species_corrections(
                {'CH4': self._spc()}, lot, 'p', self.tmp_dir,
            )
        # Subprocess was called once
        self.assertEqual(mock_exec.call_count, 1)
        # Result keyed by label
        self.assertIn('CH4', out)
        self.assertEqual(out['CH4']['aec']['value'], -0.02)
        self.assertEqual(out['CH4']['bac']['value'], -0.7)
        # Subprocess input batched all species
        save_call = mock_save.call_args
        content = save_call[1].get('content') or save_call[0][1]
        self.assertEqual(content['level_of_theory'],
                          "LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='qchem')")
        self.assertEqual(content['bac_type'], 'p')
        self.assertEqual(len(content['species']), 1)
        self.assertEqual(content['species'][0]['label'], 'CH4')
        self.assertEqual(content['species'][0]['atoms'], {'C': 1, 'H': 4})
        self.assertEqual(content['species'][0]['bonds'], {'C-H': 4})
        self.assertEqual(content['species'][0]['multiplicity'], 1)

    def test_returns_empty_when_lot_key_not_in_database(self):
        lot = Level(method='unknown', basis='unknown')
        patches = [patch('arc.output.get_qm_corrections_files', return_value=['/fake/data.py']),
                   patch('arc.output.find_best_across_files', return_value=None)]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        with patch('arc.output.execute_command') as mock_exec:
            out = _compute_species_corrections({'CH4': self._spc()}, lot, 'p', self.tmp_dir)
        self.assertEqual(out, {})
        mock_exec.assert_not_called()

    def test_subprocess_failure_returns_empty(self):
        lot = Level(method='wb97xd3', basis='def2tzvp', software='qchem')
        patches = self._patch_lot_key()
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        with patch('arc.output.execute_command', side_effect=RuntimeError('boom')):
            out = _compute_species_corrections(
                {'CH4': self._spc()}, lot, 'p', self.tmp_dir,
            )
        self.assertEqual(out, {})


class TestScanCalculations(unittest.TestCase):
    """Tests for the rotor scan → ``additional_calculations`` plumbing.

    Covers two layers:
    - ``_build_scan_result_for_rotor`` shapes one rotor into a TCKDB-like
      ``scan_result`` dict, returning ``None`` when the input is unusable.
    - ``_build_scan_calculations`` aggregates across ``rotors_dict`` and
      filters non-1D / failed / unparseable rotors.
    - ``_get_torsions`` attaches ``source_scan_calculation_key`` only when
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
        self.assertTrue(result['is_relaxed'])
        # zero_energy_reference_hartree = min absolute energy on the curve.
        self.assertIsInstance(result['zero_energy_reference_hartree'], float)
        # One coordinate, dihedral, atoms 1–4, 1-based, with symmetry.
        self.assertEqual(len(result['coordinates']), 1)
        coord = result['coordinates'][0]
        self.assertEqual(coord['coordinate_index'], 1)
        self.assertEqual(coord['coordinate_kind'], 'dihedral')
        self.assertEqual(
            (coord['atom1_index'], coord['atom2_index'],
             coord['atom3_index'], coord['atom4_index']),
            (1, 2, 3, 4),
        )
        self.assertEqual(coord['value_unit'], 'degree')
        self.assertEqual(coord['symmetry_number'], 3)
        self.assertEqual(coord['step_count'], len(result['points']))
        # Points carry index, energies, coordinate_values. Per-point
        # geometries are now emitted under ``geometry.xyz_text`` so
        # TCKDB can persist them into ``calc_scan_point.geometry_id``;
        # geometries come straight from
        # ``parse_1d_scan_full_result()['geometries']`` and are
        # serialized in the TCKDB count-headered xyz convention.
        self.assertGreater(len(result['points']), 0)
        first = result['points'][0]
        self.assertEqual(first['point_index'], 1)
        self.assertEqual(first['coordinate_values'][0]['value_unit'], 'degree')
        self.assertIn('relative_energy_kj_mol', first)
        self.assertIn('electronic_energy_hartree', first)
        self.assertNotIn('xyz', first)
        # First point's relative energy ≈ 0 by zero-shift convention.
        self.assertAlmostEqual(first['relative_energy_kj_mol'], 1.5753056e-05,
                               places=6)

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
        calcs = _build_scan_calculations(spc, '/tmp/project')
        self.assertEqual(len(calcs), 2)
        self.assertEqual(calcs[0]['key'], 'scan_rotor_0')
        self.assertEqual(calcs[0]['type'], 'scan')
        self.assertEqual(calcs[1]['key'], 'scan_rotor_1')
        self.assertIn('scan_result', calcs[0])
        self.assertEqual(calcs[0]['scan_result']['dimension'], 1)

    def test_build_scan_calculations_skips_failed_rotor(self):
        spc = MagicMock()
        spc.rotors_dict = {
            0: self._rotor(),                     # ok
            1: self._rotor(success=False),        # filtered
        }
        calcs = _build_scan_calculations(spc, '/tmp/project')
        self.assertEqual([c['key'] for c in calcs], ['scan_rotor_0'])

    def test_build_scan_calculations_skips_nd(self):
        """ND rotors are deferred — only 1D scans are emitted today."""
        spc = MagicMock()
        spc.rotors_dict = {
            0: self._rotor(),                  # 1D, ok
            1: self._rotor(dimensions=2),      # ND, skipped
        }
        calcs = _build_scan_calculations(spc, '/tmp/project')
        self.assertEqual([c['key'] for c in calcs], ['scan_rotor_0'])

    def test_build_scan_calculations_skips_unparseable(self):
        """Unparseable scan (no log on disk) → no calc, no exception."""
        spc = MagicMock()
        spc.rotors_dict = {
            0: self._rotor(scan_path=''),  # log missing → skipped
            1: self._rotor(),              # ok
        }
        calcs = _build_scan_calculations(spc, '/tmp/project')
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
        self.assertEqual(torsions[0]['source_scan_calculation_key'], 'scan_rotor_0')
        # Second rotor has no scan log on disk → no fabricated key.
        self.assertIsNone(torsions[1]['source_scan_calculation_key'])

    # ---- per-point scan geometries (TCKDB calc_scan_point.geometry_id) ----
    #
    # ARC's parser wrapper already returns aligned per-step xyz dicts.
    # ``_build_scan_result_for_rotor`` previously dropped them; now it
    # passes them through as ``points[i].geometry.xyz_text`` so TCKDB's
    # bundle workflow can resolve and persist a geometry per scan point.

    def _stub_parsed(self, *, n_points=3, geometries='aligned'):
        """Build a parser-wrapper return value with controllable geometry alignment.

        ``geometries`` is one of:
          - ``'aligned'``   : list of length n_points, each a valid xyz dict.
          - ``'mismatch'``  : list of length n_points + 1.
          - ``'none'``      : ``None`` (parser returned no geometries).
          - ``'malformed'`` : valid count, but one entry is malformed.
        """
        valid_xyz = {
            'symbols': ('C', 'H'),
            'isotopes': (12, 1),
            'coords': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        }
        if geometries == 'aligned':
            geom_list = [valid_xyz for _ in range(n_points)]
        elif geometries == 'mismatch':
            geom_list = [valid_xyz for _ in range(n_points + 1)]
        elif geometries == 'none':
            geom_list = None
        elif geometries == 'malformed':
            geom_list = [valid_xyz for _ in range(n_points)]
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
        self.assertEqual(len(result['points']), 3)
        for point in result['points']:
            self.assertIn('geometry', point)
            self.assertIn('xyz_text', point['geometry'])
            xyz_text = point['geometry']['xyz_text']
            # TCKDB count-headered convention: "<n>\\n<comment>\\n<atoms>".
            lines = xyz_text.splitlines()
            self.assertEqual(int(lines[0].strip()), 2,
                             msg=f"first line must be atom count, got {lines[0]!r}")
            # Body has the right atom count.
            self.assertEqual(len(lines), 4)  # count + comment + 2 atom rows

    def test_scan_point_geometry_uses_only_xyz_text_no_db_id(self):
        """No ``geometry_id`` (or any DB id) anywhere under scan_result."""
        with patch('arc.output.parse_1d_scan_full_result',
                   return_value=self._stub_parsed(n_points=3)):
            result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        forbidden = {'geometry_id', 'existing_geometry_id', 'id'}
        for point in result['points']:
            geom = point.get('geometry') or {}
            self.assertEqual(set(geom.keys()), {'xyz_text'},
                             msg=f"geometry must carry only xyz_text, got {geom}")
            for k in forbidden:
                self.assertNotIn(k, point, msg=f"{k} leaked onto scan point")
                self.assertNotIn(k, geom)
        # Top-level scan_result also has no DB ids.
        for k in forbidden:
            self.assertNotIn(k, result)

    def test_scan_points_omit_geometry_when_geometries_missing(self):
        """Parser returned ``geometries=None`` → no point carries a geometry,
        no warning, scan_result still emitted with energies and angles."""
        with patch('arc.output.parse_1d_scan_full_result',
                   return_value=self._stub_parsed(n_points=3, geometries='none')):
            result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        for point in result['points']:
            self.assertNotIn('geometry', point)

    def test_scan_points_omit_geometry_uniformly_on_length_mismatch(self):
        """Length mismatch → drop geometries from ALL points (not partial)
        and log a warning. The scan_result itself still uploads."""
        with patch('arc.output.parse_1d_scan_full_result',
                   return_value=self._stub_parsed(n_points=3, geometries='mismatch')):
            with self.assertLogs('arc', level='WARNING') as cm:
                result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        for point in result['points']:
            self.assertNotIn('geometry', point)
        self.assertTrue(any('does not match scan-point count' in m for m in cm.output),
                        msg=f"expected mismatch warning, got: {cm.output}")

    # ---- requested scan-grid metadata (TCKDB calc_scan_coordinate fields) ----

    def test_scan_coord_includes_step_size_from_gaussian_header(self):
        """Real Gaussian scan log → step_size + resolution_degrees populated
        from the parsed ModRedundant header (not from the completed-point
        spacing). The fixture log has ``S N 8.0`` in its header."""
        result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        coord = result['coordinates'][0]
        self.assertIn('step_size', coord)
        self.assertIn('resolution_degrees', coord)
        # 1D dihedral torsion scan: resolution == step size.
        self.assertEqual(coord['step_size'], coord['resolution_degrees'])
        # ARC writes ``S 360/scan_res scan_res``; for the sBuOH fixture the
        # requested step size is 8 degrees.
        self.assertAlmostEqual(coord['step_size'], 8.0, places=6)

    def test_scan_coord_step_size_independent_from_completed_count(self):
        """``step_count`` reflects the *completed* points, ``step_size`` the
        *requested* grid — they're sourced separately and must not be
        coupled (a partially-failed scan would otherwise emit a misleading
        derived step_size). Spot-check both come from independent data."""
        result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        coord = result['coordinates'][0]
        # step_count is the point count we actually parsed; step_size is
        # the requested grid spacing. Their product covers the requested
        # range only when no points dropped.
        self.assertEqual(coord['step_count'], len(result['points']))
        self.assertGreater(coord['step_size'], 0.0)

    def test_scan_coord_omits_grid_metadata_for_non_gaussian(self):
        """``parse_scan_args`` raising NotImplementedError (ORCA, etc.) →
        step_size / resolution_degrees absent, no exception, scan_result
        still produced."""
        with patch('arc.output.parse_scan_args',
                   side_effect=NotImplementedError('ORCA path')):
            result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        coord = result['coordinates'][0]
        self.assertNotIn('step_size', coord)
        self.assertNotIn('resolution_degrees', coord)

    def test_scan_coord_omits_grid_metadata_when_parser_raises(self):
        """Generic parser failure (corrupt log, etc.) → grid fields absent,
        no exception."""
        with patch('arc.output.parse_scan_args',
                   side_effect=RuntimeError('boom')):
            result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        coord = result['coordinates'][0]
        self.assertNotIn('step_size', coord)
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
        coord = result['coordinates'][0]
        self.assertNotIn('step_size', coord)
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
        self.assertGreater(len(result['points']), 0)
        for point in result['points']:
            self.assertIn('coordinate_values', point)
            self.assertEqual(point['coordinate_values'][0]['value_unit'], 'degree')

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
        coord = result['coordinates'][0]
        self.assertIn('start_value', coord)
        # The fixture's offset compensates for the helper's right-hand
        # rule, so 60° in → 60° out.
        self.assertAlmostEqual(coord['start_value'], 60.0, places=4)

    def test_scan_end_value_extends_continuously_from_start(self):
        """``end_value = start_value + step_size * (step_count - 1)`` —
        not wrapped, so a 46-point 8° scan from 60° lands at 60 + 360 = 420°,
        NOT back at 60° and NOT mod-360'd to 60°."""
        rotor = self._rotor(scan=[1, 2, 3, 4])
        xyz = self._input_xyz_for_dihedral(60.0)
        result = _build_scan_result_for_rotor(
            rotor, '/tmp/project', input_xyz=xyz,
        )
        coord = result['coordinates'][0]
        # Real Gaussian fixture: step_size=8, len(points)=46 → +360 span.
        expected_end = coord['start_value'] + coord['step_size'] * (
            coord['step_count'] - 1
        )
        self.assertAlmostEqual(coord['end_value'], expected_end, places=6)
        # Not wrapped: a full-rotation scan exceeds 360°, never re-folds.
        self.assertGreater(coord['end_value'], 360.0)

    def test_scan_end_value_is_not_wrapped_into_minus_180_180(self):
        """Continuity contract: even when start is near 180°, the end
        value must not flip sign by wrapping into [-180, 180]."""
        rotor = self._rotor(scan=[1, 2, 3, 4])
        xyz = self._input_xyz_for_dihedral(170.0)
        result = _build_scan_result_for_rotor(
            rotor, '/tmp/project', input_xyz=xyz,
        )
        coord = result['coordinates'][0]
        # 170 + 360 = 530 — must NOT have folded to -190 or 170.
        self.assertGreater(coord['end_value'], 360.0)
        self.assertGreater(coord['end_value'] - coord['start_value'],
                           coord['step_size'] * 0.99)

    def test_scan_start_end_absent_when_input_geometry_missing(self):
        """No ``input_xyz`` and no parser fallback → fields stay absent."""
        rotor = self._rotor(scan=[1, 2, 3, 4])
        # Patch out both sources: input_xyz=None and parsed geometries=None.
        with patch('arc.output.parse_1d_scan_full_result') as p:
            from arc.parser.parser import parse_1d_scan_full_result as real
            parsed = real(self.SCAN_LOG)
            parsed['geometries'] = None  # kill the fallback
            p.return_value = parsed
            result = _build_scan_result_for_rotor(
                rotor, '/tmp/project', input_xyz=None,
            )
        coord = result['coordinates'][0]
        self.assertNotIn('start_value', coord)
        self.assertNotIn('end_value', coord)

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
        coord = result['coordinates'][0]
        self.assertNotIn('step_size', coord)  # confirms the precondition
        self.assertNotIn('start_value', coord)
        self.assertNotIn('end_value', coord)

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
        coord = result['coordinates'][0]
        self.assertNotIn('start_value', coord)
        self.assertNotIn('end_value', coord)

    def test_scan_start_end_absent_when_dihedral_calc_raises(self):
        """A failed dihedral calculation logs a warning, omits start/end,
        and does NOT abort the rest of scan_result emission."""
        rotor = self._rotor(scan=[1, 2, 3, 4])
        xyz = self._input_xyz_for_dihedral(60.0)
        with patch('arc.output.calculate_dihedral_angle',
                   side_effect=RuntimeError('atom missing')):
            with self.assertLogs('arc', level='WARNING') as cm:
                result = _build_scan_result_for_rotor(
                    rotor, '/tmp/project', input_xyz=xyz,
                )
        # scan_result is still emitted (energies + step_size + points all there).
        self.assertIsNotNone(result)
        coord = result['coordinates'][0]
        self.assertNotIn('start_value', coord)
        self.assertNotIn('end_value', coord)
        self.assertIn('step_size', coord)  # other grid fields untouched
        self.assertGreater(len(result['points']), 0)
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
        self.assertEqual(len(result_with['points']), len(result_without['points']))
        for p_with, p_without in zip(result_with['points'], result_without['points']):
            self.assertEqual(p_with['coordinate_values'],
                             p_without['coordinate_values'])

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
        coord = result['coordinates'][0]
        # Real Gaussian fixture has ``geometries`` populated, so the
        # fallback resolves and start/end are emitted.
        self.assertIn('start_value', coord)
        self.assertIn('end_value', coord)

    def test_scan_points_omit_geometry_uniformly_on_serialization_failure(self):
        """One unserializable xyz dict → drop geometries from ALL points,
        warn once. Energies/angles still flow through."""
        with patch('arc.output.parse_1d_scan_full_result',
                   return_value=self._stub_parsed(n_points=3, geometries='malformed')):
            with self.assertLogs('arc', level='WARNING') as cm:
                result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        # No partial coverage: nothing carries a geometry.
        for point in result['points']:
            self.assertNotIn('geometry', point)
        self.assertTrue(
            any('serialization failed' in m or 'empty text' in m for m in cm.output),
            msg=f"expected serialization warning, got: {cm.output}",
        )


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
        sentinel = [{'constraint_kind': 'bond', 'atoms': [1, 2],
                     'target_value': None}]
        with patch('arc.parser.adapters.gaussian.parse_gaussian_constraints',
                   return_value=sentinel) as gauss, \
             patch('arc.parser.adapters.orca.parse_orca_constraints') as orca:
            result = _parse_scan_constraints(
                self._rotor(scan_software='gaussian'), '/tmp/project',
            )
        self.assertEqual(result, sentinel)
        gauss.assert_called_once_with(self.SCAN_LOG)
        orca.assert_not_called()

    def test_orca_hint_routes_to_orca_parser(self):
        from arc.output import _parse_scan_constraints
        sentinel = [{'constraint_kind': 'dihedral',
                     'atoms': [1, 2, 3, 4], 'target_value': 90.0}]
        with patch('arc.parser.adapters.orca.parse_orca_constraints',
                   return_value=sentinel) as orca, \
             patch('arc.parser.adapters.gaussian.parse_gaussian_constraints') as gauss:
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
        with patch('arc.parser.adapters.gaussian.parse_gaussian_constraints',
                   return_value=[]) as gauss:
            rotor_no_field = self._rotor()
            rotor_no_field.pop('scan_software', None)
            _parse_scan_constraints(rotor_no_field, '/tmp/project')
            _parse_scan_constraints(self._rotor(scan_software=''), '/tmp/project')
        self.assertEqual(gauss.call_count, 2)

    def test_unknown_software_returns_empty_list_no_parser_call(self):
        from arc.output import _parse_scan_constraints
        with patch('arc.parser.adapters.gaussian.parse_gaussian_constraints') as gauss, \
             patch('arc.parser.adapters.orca.parse_orca_constraints') as orca:
            result = _parse_scan_constraints(
                self._rotor(scan_software='qchem'), '/tmp/project',
            )
        self.assertEqual(result, [])
        gauss.assert_not_called()
        orca.assert_not_called()

    def test_parser_exception_degrades_to_empty_list(self):
        from arc.output import _parse_scan_constraints
        with patch('arc.parser.adapters.gaussian.parse_gaussian_constraints',
                   side_effect=RuntimeError('parser crashed')):
            result = _parse_scan_constraints(
                self._rotor(scan_software='gaussian'), '/tmp/project',
            )
        self.assertEqual(result, [])

    def test_missing_scan_path_returns_empty_list(self):
        # Defensive: never invoke a parser without a real path.
        from arc.output import _parse_scan_constraints
        rotor = self._rotor(scan_path='', scan_software='gaussian')
        with patch('arc.parser.adapters.gaussian.parse_gaussian_constraints') as gauss:
            self.assertEqual(_parse_scan_constraints(rotor, '/tmp/project'), [])
            gauss.assert_not_called()


if __name__ == '__main__':
    unittest.main()
