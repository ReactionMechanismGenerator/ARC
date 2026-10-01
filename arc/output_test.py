"""
Tests for the arc.output module (consolidated output.yml writer).
"""

import copy
import datetime
import hashlib
import json
import math
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

from arc.common import ARC_PATH, read_yaml_file
from arc.constants import E_h_kJmol, bohr_to_angstrom
from arc.exceptions import InputError
from arc.level import Level, adaptive_levels_as_list, set_recorded_irc_log_level, set_recorded_level
from arc.common import ARC_TESTING_PATH
from arc.output import (
    EnergyCorrections,
    GAUSSIAN_CORRELATED_METHOD_REGEX,
    _bac_is_applied_to,
    _build_energy_corrections_for_species,
    _build_rotor_scans,
    _build_scan_result_for_rotor,
    _compute_cost_metrics,
    _compute_point_groups,
    _compute_species_corrections,
    _evidence_status_counts,
    _get_arkane_provenance,
    _get_arkane_quantum_corrections_path,
    _has_default_isotopes,
    _get_rmg_database_identity,
    _get_rmg_py_git_commit,
    _get_energy_corrections,
    _match_arkane_correction_keys,
    _get_ess_route,
    _get_gaussian_dipole_moment,
    _get_gaussian_polarizability,
    _get_log_stated_isotopes,
    _get_ess_software,
    _get_reversible,
    _get_t1_diagnostic,
    _gaussian_route_text,
    _get_freq_hessian_method,
    _get_ess_versions,
    _get_gsm_ess,
    _gsm_level,
    _get_rejected_torsions,
    _get_rotor_barrier,
    _get_torsions,
    _get_imaginary_freqs,
    _flat_parameter_values,
    LEVEL_DICT_STRIPPED_KEYS,
    _level_to_dict,
    _recorded_level_to_dict,
    _make_rel_path,
    _orca_route_text,
    _parse_arkane_log_provenance,
    _parse_opt_log,
    _parse_rmg_py_version,
    _parse_spin_diagnostic,
    _parse_wavefunction_stability,
    _parse_zpe,
    _resolve_freq_scale_factor_entry,
    _rxn_to_dict,
    _spc_to_dict,
    _statmech_to_dict,
    _thermo_to_dict,
    _timedelta_to_seconds,
    _xyz_isotopes,
    write_output_yml,
)
from arc.reaction import ARCReaction
from arc.species.converter import xyz_from_data
from arc.species.species import ARCSpecies, TSGuess, ThermoData


def setUpModule():
    """Keep ``write_output_yml`` from starting the RMG environment to ask where Arkane loaded its tables from."""
    global _arkane_path_patcher
    _arkane_path_patcher = patch('arc.output._get_arkane_quantum_corrections_path', return_value=None)
    _arkane_path_patcher.start()


def tearDownModule():
    _arkane_path_patcher.stop()


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

    def test_year_survives_and_is_omitted_when_unset(self):
        """``year`` is carried through when the level has one and omitted when it does not."""
        self.assertEqual(_level_to_dict(Level(method='wb97xd', basis='def2tzvp',
                                              software='gaussian', year=2023))['year'], 2023)
        self.assertNotIn('year', _level_to_dict(Level(method='wb97xd', basis='def2tzvp',
                                                      software='gaussian')))

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

    def test_a_log_outside_the_project_directory_is_dropped(self):
        """An export is shared, so a path that walks out of the project — which would
        carry the writer's own directory layout and resolve nowhere for a reader — is
        reported as absent rather than as a ``../..`` path."""
        for path in ('/home/someone/elsewhere/sp.log',
                     '/home/user/project_sibling/sp.log',
                     '/home/user/sp.log'):
            with self.subTest(path=path):
                self.assertIsNone(_make_rel_path(path, '/home/user/project'))

    def test_a_dropped_path_never_leaks_a_directory_name(self):
        """The guard is on the whole result, not only on a leading ``..``."""
        result = _make_rel_path('/home/quiteidentifiable/run/sp.log', '/srv/projects/p1')
        self.assertIsNone(result)


class TestResolveFreqScaleFactorEntry(unittest.TestCase):
    """Tests for _resolve_freq_scale_factor_entry."""

    def test_none_level(self):
        self.assertEqual(_resolve_freq_scale_factor_entry(None), (None, None))

    def test_missing_level(self):
        """A level not in the YAML has neither a key nor a source."""
        level = Level(method='totally_fake_method', basis='fake_basis')
        self.assertEqual(_resolve_freq_scale_factor_entry(level), (None, None))

    def test_known_level_returns_its_key_and_citation(self):
        """wb97xd/def2tzvp names the block it was read from and that block's [4] citation."""
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        key, source = _resolve_freq_scale_factor_entry(level)
        self.assertEqual(key, 'wb97xd/def2tzvp, software: gaussian')
        self.assertIsNotNone(source)
        self.assertIn('10.1021/ct100326h', source)

    def test_the_key_is_the_block_the_factor_was_read_from(self):
        """The returned key indexes the entry whose factor ARC would use."""
        level = Level(method='b3lyp', basis='6-311g(d,p)', software='gaussian')
        key, source = _resolve_freq_scale_factor_entry(level)
        data = read_yaml_file(os.path.join(ARC_PATH, 'data', 'freq_scale_factors.yml'))
        self.assertIn(key, data['freq_scale_factors'])
        self.assertEqual(source, data['sources'][data['freq_scale_factors'][key]['source']])

    def test_a_string_level_is_used_verbatim(self):
        key, source = _resolve_freq_scale_factor_entry('wb97xd/def2tzvp, software: gaussian')
        self.assertEqual(key, 'wb97xd/def2tzvp, software: gaussian')
        self.assertIsNotNone(source)

    def test_an_entry_without_a_source_still_reports_its_key(self):
        """A matched block with no source index is named, with a null citation."""
        database = {'sources': {1: 'a citation'},
                    'freq_scale_factors': {'fake/level': {'factor': 0.99}}}
        with patch('arc.output.read_yaml_file', return_value=database):
            self.assertEqual(_resolve_freq_scale_factor_entry('fake/level'), ('fake/level', None))

    def test_an_unlisted_source_index_still_reports_its_key(self):
        database = {'sources': {1: 'a citation'},
                    'freq_scale_factors': {'fake/level': {'factor': 0.99, 'source': 7}}}
        with patch('arc.output.read_yaml_file', return_value=database):
            self.assertEqual(_resolve_freq_scale_factor_entry('fake/level'), ('fake/level', None))

    def test_an_unreadable_or_malformed_database_yields_no_key(self):
        """The failures read_yaml_file can raise are guarded, one at a time."""
        for error in (InputError('missing'), OSError('unreadable'), yaml.YAMLError('malformed')):
            with self.subTest(error=type(error).__name__):
                with patch('arc.output.read_yaml_file', side_effect=error):
                    self.assertEqual(_resolve_freq_scale_factor_entry('fake/level'), (None, None))

    def test_an_unexpected_error_is_not_swallowed(self):
        """The guard is narrow: anything read_yaml_file cannot raise propagates."""
        with patch('arc.output.read_yaml_file', side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                _resolve_freq_scale_factor_entry('fake/level')

    def test_a_database_that_is_not_a_mapping_yields_no_key(self):
        with patch('arc.output.read_yaml_file', return_value=['not', 'a', 'mapping']):
            self.assertEqual(_resolve_freq_scale_factor_entry('fake/level'), (None, None))


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


class TestGetRmgPyGitCommit(unittest.TestCase):
    """Tests for _get_rmg_py_git_commit."""

    def setUp(self):
        self.rmg_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.rmg_dir, ignore_errors=True)

    @patch('arc.output.get_git_commit', return_value=('abc1234', '2026-01-01'))
    def test_returns_hash(self, mock_git):
        self.assertEqual(_get_rmg_py_git_commit(self.rmg_dir), 'abc1234')
        self.assertEqual(mock_git.call_args[0][0], self.rmg_dir)

    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_returns_none_for_empty(self, mock_git):
        self.assertIsNone(_get_rmg_py_git_commit(self.rmg_dir))

    def test_returns_none_no_rmg_path(self):
        self.assertIsNone(_get_rmg_py_git_commit(None))

    def test_a_misconfigured_rmg_path_is_reported(self):
        """A path that is set but names no directory is logged as a warning."""
        missing = os.path.join(self.rmg_dir, 'does_not_exist')
        with self.assertLogs('arc', level='WARNING') as captured:
            self.assertIsNone(_get_rmg_py_git_commit(missing))
        self.assertIn(missing, '\n'.join(captured.output))

    @patch('arc.output.get_git_commit', side_effect=ValueError('not enough values to unpack'))
    def test_returns_none_when_git_output_is_unreadable(self, mock_git):
        """Output that does not split into a hash and a date is absorbed."""
        self.assertIsNone(_get_rmg_py_git_commit(self.rmg_dir))


ARKANE_LOG_HEADER = """Arkane execution initiated at Sat Sep 13 10:00:00 2026

################################################################
#                                                              #
# Automated Reaction Kinetics and Network Exploration (Arkane) #
#                                                              #
#   Version: {version:49s} #
#   Authors: RMG Developers (rmg_dev@mit.edu)                  #
#                                                              #
################################################################

The current git HEAD for RMG-Py is:
\t{head}
\tTue Feb 17 08:33:23 2026 -0500

The current git HEAD for RMG-database is:
\t0000000000000000000000000000000000000000
\tTue Feb 17 08:33:23 2026 -0500
"""


def arkane_log_header(version: str = '3.3.0',
                      head: str = '6b1368de6c19204c7ce4fda6fbecb05da4a0fe0e') -> str:
    """Render an Arkane log header in the layout ``arkane.main.log_header`` prints."""
    return ARKANE_LOG_HEADER.format(version=version, head=head)


class TestArkaneProvenance(unittest.TestCase):
    """Tests for _get_arkane_provenance and the sources it reads."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def _write_arkane_log(self, subdir: str, file_name: str, content: str) -> str:
        from arc.statmech.arkane import create_statmech_dir
        log_dir = create_statmech_dir(calcs_directory=os.path.join(self.tmp_dir, 'calcs'),
                                      subdir=subdir)
        log_path = os.path.join(log_dir, file_name)
        with open(log_path, 'w') as handle:
            handle.write(content)
        return log_path

    def _write_rmg_py_tree(self, version_line: str) -> str:
        rmg_path = os.path.join(self.tmp_dir, 'RMG-Py')
        os.makedirs(os.path.join(rmg_path, 'rmgpy'), exist_ok=True)
        with open(os.path.join(rmg_path, 'rmgpy', 'version.py'), 'w') as handle:
            handle.write(f'"""Docstring."""\n\n{version_line}\n')
        return rmg_path

    def test_the_statmech_layout_matches_create_statmech_dir(self):
        """``create_statmech_dir`` writes the directory this module reads, not a copy of it."""
        self._write_arkane_log('thermo', 'arkane.log', arkane_log_header())
        with patch('arc.output.settings', {}):
            provenance = _get_arkane_provenance(self.tmp_dir)
        self.assertEqual(provenance.version, '3.3.0')
        self.assertEqual(provenance.git_commit, '6b1368de6c19204c7ce4fda6fbecb05da4a0fe0e')

    def test_the_rmg_database_commit_is_not_mistaken_for_rmg_py(self):
        log_path = self._write_arkane_log('thermo', 'arkane.log', arkane_log_header())
        self.assertEqual(_parse_arkane_log_provenance(log_path).git_commit,
                         '6b1368de6c19204c7ce4fda6fbecb05da4a0fe0e')

    def test_an_appended_log_reports_the_first_run_whole(self):
        """Both fields come from the first header, so an appended log pairs one run's own values."""
        log_path = self._write_arkane_log(
            'thermo', 'arkane.log',
            arkane_log_header(version='3.3.0', head='a' * 40)
            + arkane_log_header(version='3.4.0', head='b' * 40))
        self.assertEqual(_parse_arkane_log_provenance(log_path), ('3.3.0', 'a' * 40))

    def test_an_arkane_run_that_produced_no_output_is_still_identified(self):
        """Arkane prints its header at startup, so a failed run is named the same as a complete one."""
        self._write_arkane_log('thermo', 'arkane.log', arkane_log_header())
        output_py = os.path.join(self.tmp_dir, 'calcs', 'statmech', 'thermo', 'output.py')
        self.assertFalse(os.path.isfile(output_py))
        with patch('arc.output.settings', {}):
            self.assertEqual(_get_arkane_provenance(self.tmp_dir),
                             ('3.3.0', '6b1368de6c19204c7ce4fda6fbecb05da4a0fe0e'))

    def test_an_empty_version_field_does_not_yield_the_box_character(self):
        """An empty version renders as padding between two ``#``; neither is a version."""
        log_path = self._write_arkane_log('thermo', 'arkane.log', arkane_log_header(version=''))
        self.assertIsNone(_parse_arkane_log_provenance(log_path).version)

    def test_a_multi_token_version_is_kept_whole(self):
        """A version carrying a qualifier is returned whole, not truncated at the first space."""
        log_path = self._write_arkane_log('thermo', 'arkane.log',
                                          arkane_log_header(version='3.3.0 (dev)'))
        self.assertEqual(_parse_arkane_log_provenance(log_path).version, '3.3.0 (dev)')

    def test_a_log_without_a_git_head_keeps_the_version(self):
        """Arkane prints no HEAD on an install with no repository; the version still stands."""
        header = arkane_log_header().split('The current git HEAD')[0]
        log_path = self._write_arkane_log('thermo', 'arkane.log', header)
        provenance = _parse_arkane_log_provenance(log_path)
        self.assertEqual(provenance.version, '3.3.0')
        self.assertIsNone(provenance.git_commit)

    def test_missing_and_headerless_logs_yield_nothing(self):
        self.assertEqual(_parse_arkane_log_provenance(os.path.join(self.tmp_dir, 'absent.log')),
                         (None, None))
        log_path = self._write_arkane_log('thermo', 'stdout.log', 'no header here\n')
        self.assertEqual(_parse_arkane_log_provenance(log_path), (None, None))

    def test_a_header_beyond_the_scan_limit_is_not_read(self):
        log_path = self._write_arkane_log('thermo', 'arkane.log',
                                          'filler\n' * 200 + arkane_log_header())
        self.assertEqual(_parse_arkane_log_provenance(log_path), (None, None))

    def test_a_log_older_than_the_run_is_ignored(self):
        """A statmech directory persists between runs; a stale log is not this run's Arkane."""
        log_path = self._write_arkane_log('thermo', 'arkane.log', arkane_log_header())
        os.utime(log_path, (1000.0, 1000.0))
        self.assertEqual(_parse_arkane_log_provenance(log_path, t0=2000.0), (None, None))
        self.assertEqual(_parse_arkane_log_provenance(log_path, t0=500.0).version, '3.3.0')

    def test_the_running_arkane_is_preferred_over_the_source_tree(self):
        self._write_arkane_log('thermo', 'arkane.log', arkane_log_header())
        rmg_path = self._write_rmg_py_tree("__version__ = '9.9.9'")
        with patch('arc.output.settings', {'RMG_PATH': rmg_path}):
            self.assertEqual(_get_arkane_provenance(self.tmp_dir).version, '3.3.0')

    def test_the_kinetics_log_is_read_when_no_thermo_log_exists(self):
        self._write_arkane_log('kinetics', 'stdout.log', arkane_log_header())
        with patch('arc.output.settings', {}):
            self.assertEqual(_get_arkane_provenance(self.tmp_dir).version, '3.3.0')

    @patch('arc.output.get_git_commit', return_value=('def4567', '2026-01-01'))
    def test_both_fields_fall_back_to_the_same_rmg_py_tree(self, mock_git):
        """The pair never mixes installs: when the version falls back, so does the commit."""
        rmg_path = self._write_rmg_py_tree("__version__ = '4.0.0'")
        with patch('arc.output.settings', {'RMG_PATH': rmg_path}):
            self.assertEqual(_get_arkane_provenance(self.tmp_dir), ('4.0.0', 'def4567'))

    def test_returns_nothing_when_no_source_is_resolvable(self):
        with patch('arc.output.settings', {'RMG_PATH': None}):
            self.assertEqual(_get_arkane_provenance(self.tmp_dir), (None, None))

    def test_rmg_py_version_file_edge_cases(self):
        mismatched_quotes = "__version__ = '4.0.0" + chr(34)
        self.assertIsNone(_parse_rmg_py_version(None))
        self.assertIsNone(_parse_rmg_py_version(os.path.join(self.tmp_dir, 'absent')))
        self.assertIsNone(_parse_rmg_py_version(self._write_rmg_py_tree('__version__ = 4.0.0')))
        self.assertIsNone(_parse_rmg_py_version(self._write_rmg_py_tree(mismatched_quotes)))


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

    def test_correction_flags_are_null_when_no_arkane_run_recorded_them(self):
        """Thermo no Arkane run stamped carries the keys as null, never a guessed bool."""
        result = _thermo_to_dict(ThermoData(H298=-10.0, S298=200.0))
        self.assertIn('atom_corrections_applied', result)
        self.assertIn('bond_corrections_applied', result)
        self.assertIsNone(result['atom_corrections_applied'])
        self.assertIsNone(result['bond_corrections_applied'])

    def test_correction_flags_are_exported_as_recorded(self):
        """The flags an Arkane run stamped on the thermo are exported unchanged, false included."""
        for aec, bac in [(True, True), (True, False), (False, False)]:
            thermo = ThermoData(H298=-10.0, S298=200.0)
            thermo.atom_corrections_applied = aec
            thermo.bond_corrections_applied = bac
            result = _thermo_to_dict(thermo)
            self.assertIs(result['atom_corrections_applied'], aec)
            self.assertIs(result['bond_corrections_applied'], bac)

    def test_atom_corrections_level_uses_the_document_level_shape(self):
        """The level is serialized exactly as the document's sp_level is, so the two compare directly."""
        level = Level(method='bmk', basis='cbsb7', software='gaussian')
        thermo = ThermoData(H298=-10.0, S298=200.0)
        thermo.atom_corrections_applied = True
        thermo.bond_corrections_applied = False
        thermo.atom_corrections_level = level
        result = _thermo_to_dict(thermo)
        self.assertEqual(result['atom_corrections_level'], _level_to_dict(level))
        self.assertEqual(result['atom_corrections_level']['method'], 'bmk')

    def test_atom_corrections_level_is_null_unless_the_correction_was_on(self):
        """No level is named for atom energies that were not subtracted, or not known to be."""
        self.assertIsNone(_thermo_to_dict(ThermoData(H298=-10.0, S298=200.0))['atom_corrections_level'])
        for applied in (False, None):
            thermo = ThermoData(H298=-10.0, S298=200.0)
            thermo.atom_corrections_applied = applied
            thermo.atom_corrections_level = Level(method='bmk', basis='cbsb7')
            self.assertIsNone(_thermo_to_dict(thermo)['atom_corrections_level'])


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
        ts_guess.index = 0
        ts_guess.imaginary_freqs = [-1500.0, -200.0]
        spc = MagicMock()
        spc.freqs = None
        spc.chosen_ts = 0
        spc.ts_guesses = [ts_guess]
        self.assertEqual(_get_imaginary_freqs(spc), [-1500.0, -200.0])

    def test_chosen_ts_is_an_index_not_a_position(self):
        """``chosen_ts`` is the chosen TSGuess.index; the list position must not be used."""
        other_guess, chosen_guess = MagicMock(), MagicMock()
        other_guess.index, other_guess.imaginary_freqs = 0, [-100.0]
        chosen_guess.index, chosen_guess.imaginary_freqs = 4, [-1500.0]
        spc = MagicMock()
        spc.freqs = None
        spc.chosen_ts = 4
        spc.ts_guesses = [other_guess, chosen_guess]
        self.assertEqual(_get_imaginary_freqs(spc), [-1500.0])

    def test_chosen_ts_matches_no_guess(self):
        guess = MagicMock()
        guess.index, guess.imaginary_freqs = 0, [-100.0]
        spc = MagicMock()
        spc.freqs = None
        spc.chosen_ts = 5
        spc.ts_guesses = [guess]
        self.assertIsNone(_get_imaginary_freqs(spc))


class TestStatmechToDict(unittest.TestCase):
    """Tests for _statmech_to_dict."""

    def _make_spc(self, is_ts=False, is_linear=False, freqs=None):
        spc = MagicMock()
        spc.is_ts = is_ts
        spc._is_linear = is_linear
        spc.is_monoatomic.return_value = False
        spc.e0 = 100.5
        spc.e0_atom_corrections_applied = None
        spc.e0_bond_corrections_applied = None
        spc.arkane_rotor_modes = None
        spc.multiplicity = 1
        spc.optical_isomers = 1
        spc.external_symmetry = 2
        spc.freqs = freqs
        spc.rotors_dict = None
        return spc

    def test_nonlinear_species(self):
        spc = self._make_spc(freqs=[100.0, 200.0, 300.0])
        spc.final_xyz = _xyz_dict(*_water_coords())
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
        spc.e0_atom_corrections_applied = None
        spc.e0_bond_corrections_applied = None
        spc.arkane_rotor_modes = None
        spc.rotors_dict = None
        self.assertEqual(_get_torsions(spc, '/tmp'), [])

    def test_empty_rotors_dict(self):
        spc = MagicMock()
        spc.e0_atom_corrections_applied = None
        spc.e0_bond_corrections_applied = None
        spc.arkane_rotor_modes = None
        spc.rotors_dict = {}
        self.assertEqual(_get_torsions(spc, '/tmp'), [])

    def test_failed_rotor_skipped(self):
        spc = MagicMock()
        spc.e0_atom_corrections_applied = None
        spc.e0_bond_corrections_applied = None
        spc.arkane_rotor_modes = None
        spc.rotors_dict = {0: {'success': False, 'scan': [1, 2, 3, 4], 'pivots': [2, 3]}}
        self.assertEqual(_get_torsions(spc, '/tmp'), [])

    @staticmethod
    def make_spc(rotor_modes, rotor_count=1):
        """A species with ``rotor_count`` successful rotors, each labelled ``HinderedRotor``, and the rotor modes
        Arkane is recorded as having kept."""
        spc = MagicMock()
        spc.rotors_dict = {index: {'success': True, 'scan': [1, 2, 3, 4], 'pivots': [2, 3], 'symmetry': 3,
                                   'type': 'HinderedRotor', 'scan_path': ''} for index in range(rotor_count)}
        spc.arkane_rotor_modes = rotor_modes
        return spc

    def test_successful_rotor(self):
        result = _get_torsions(self.make_spc(['HinderedRotor']), '/tmp')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['atom_indices'], [1, 2, 3, 4])
        self.assertEqual(result[0]['pivot_atoms'], [2, 3])
        self.assertEqual(result[0]['symmetry_number'], 3)
        self.assertEqual(result[0]['treatment'], 'hindered_rotor')
        self.assertIsNone(result[0]['barrier_kj_mol'])

    def test_free_rotor(self):
        result = _get_torsions(self.make_spc(['FreeRotor']), '/tmp')
        self.assertEqual(result[0]['treatment'], 'free_rotor')

    def test_treatment_is_taken_per_rotor_from_the_arkane_modes(self):
        result = _get_torsions(self.make_spc(['FreeRotor', 'HinderedRotor'], rotor_count=2), '/tmp')
        self.assertEqual([torsion['treatment'] for torsion in result], ['free_rotor', 'hindered_rotor'])

    def test_treatment_is_null_unless_arkane_is_known_to_have_kept_every_rotor(self):
        """ARC's own ``type`` never decides the treatment. With no parsed Arkane output, with Arkane having
        dropped every rotor, with a different number of rotors kept, or with a multi-dimensional mode, it is null."""
        for rotor_modes, rotor_count in ((None, 1), ([], 1), (['HinderedRotor'], 2), (['Mode'], 1),
                                         (['HinderedRotorClassicalND'], 1)):
            with self.subTest(rotor_modes=rotor_modes, rotor_count=rotor_count):
                result = _get_torsions(self.make_spc(rotor_modes, rotor_count), '/tmp')
                self.assertEqual(len(result), rotor_count)
                self.assertEqual([torsion['treatment'] for torsion in result], [None] * rotor_count)


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
        n_steps, e_hartree, final_xyz, final_isotopes = _parse_opt_log(opt_path, '/dummy')
        self.assertEqual(n_steps, 4)
        self.assertIsNotNone(e_hartree)
        self.assertAlmostEqual(e_hartree, -116.986089069, places=6)
        self.assertIsNotNone(final_xyz)

    def test_missing_file(self):
        n_steps, e_hartree, final_xyz, final_isotopes = _parse_opt_log('/nonexistent/file.log', '/tmp')
        self.assertIsNone(n_steps)
        self.assertIsNone(e_hartree)
        self.assertIsNone(final_xyz)

    def test_none_path(self):
        n_steps, e_hartree, final_xyz, final_isotopes = _parse_opt_log(None, '/tmp')
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
        """Test that an open-shell doublet yields s_squared, expected and annihilated"""
        sp = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'Species', 'NH2_freq.out')
        sd = _parse_spin_diagnostic(sp, None, None, multiplicity=2, project_directory='/dummy')
        self.assertIsNotNone(sd)
        self.assertAlmostEqual(sd['s_squared'], 0.7535)
        self.assertAlmostEqual(sd['s_squared_expected'], 0.75)
        self.assertAlmostEqual(sd['s_squared_annihilated'], 0.75)

    def test_expected_recomputed_from_arc_multiplicity(self):
        """Test that s_squared_expected comes from ARC's multiplicity, a triplet giving 2.0"""
        sp = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'TSs', 'TS_freq.out')
        sd = _parse_spin_diagnostic(sp, None, None, multiplicity=3, project_directory='/dummy')
        self.assertIsNotNone(sd)
        self.assertAlmostEqual(sd['s_squared'], 2.0153)
        self.assertAlmostEqual(sd['s_squared_expected'], 2.0)

    def test_closed_shell_returns_none(self):
        """Test that a restricted log, which prints no <S**2>, yields None"""
        sp = os.path.join(ARC_TESTING_PATH, 'composite', 'C2H5NO2__C2H5ONO.out')
        self.assertIsNone(_parse_spin_diagnostic(sp, None, None, multiplicity=1, project_directory='/dummy'))

    def test_fallback_to_freq_when_sp_absent(self):
        """Test that an absent sp log falls back to the freq log"""
        freq = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'Species', 'NH2_freq.out')
        sd = _parse_spin_diagnostic(None, freq, None, multiplicity=2, project_directory='/dummy')
        self.assertIsNotNone(sd)
        self.assertAlmostEqual(sd['s_squared'], 0.7535)

    def test_no_paths_returns_none(self):
        """Test that no candidate log yields None"""
        self.assertIsNone(_parse_spin_diagnostic(None, None, None, multiplicity=2, project_directory='/dummy'))

    def test_orca_open_shell_no_annihilation_key(self):
        """Test that an ESS reporting no annihilated value omits the key from the block"""
        sp = os.path.join(ARC_TESTING_PATH, 'neb', 'neb_res.out')
        sd = _parse_spin_diagnostic(sp, None, None, multiplicity=2, project_directory='/dummy')
        self.assertIsNotNone(sd)
        self.assertNotIn('s_squared_annihilated', sd)
        self.assertAlmostEqual(sd['s_squared_expected'], 0.75)

    def test_a_stability_log_is_not_read_off_its_eigenvectors(self):
        """Test that a Stable job's log yields the wavefunction's <S**2>, not a root's"""
        sp = os.path.join(ARC_TESTING_PATH, 'stability', 'stable_unrestricted_doublet_ts.out')
        sd = _parse_spin_diagnostic(sp, None, None, multiplicity=2, project_directory='/dummy')
        self.assertIsNotNone(sd)
        self.assertAlmostEqual(sd['s_squared'], 0.7536)
        restricted = os.path.join(ARC_TESTING_PATH, 'stability', 'stable_restricted_singlet_ts.out')
        self.assertIsNone(_parse_spin_diagnostic(restricted, None, None, multiplicity=1,
                                                 project_directory='/dummy'))

    def test_the_sp_log_is_preferred_over_the_freq_and_opt_logs(self):
        """Test that the sp log wins when several candidate logs exist"""
        sp = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'Species', 'NH2_freq.out')
        freq = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'TSs', 'TS_freq.out')
        opt = os.path.join(ARC_TESTING_PATH, 'freq', 'CH3OO_freq_gaussian.out')
        sd = _parse_spin_diagnostic(sp, freq, opt, multiplicity=2, project_directory=ARC_TESTING_PATH)
        self.assertAlmostEqual(sd['s_squared'], 0.7535)
        self.assertNotAlmostEqual(sd['s_squared'], 2.0153)
        self.assertNotAlmostEqual(sd['s_squared'], 0.7544)

    def test_the_log_the_value_was_read_from_is_recorded(self):
        """Test that the block names the log its <S**2> came from"""
        freq = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'Species', 'NH2_freq.out')
        sd = _parse_spin_diagnostic(None, freq, None, multiplicity=2, project_directory=ARC_TESTING_PATH)
        self.assertEqual(sd['log'], os.path.join('restart', '2_restart_rate', 'calcs', 'Species',
                                                 'NH2_freq.out'))

    def test_an_sp_log_that_holds_no_s_squared_is_not_replaced_by_another_job(self):
        """Test that a present sp log yielding no <S**2> ends the search rather than falling through"""
        sp = os.path.join(ARC_TESTING_PATH, 'composite', 'C2H5NO2__C2H5ONO.out')
        freq = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'Species', 'NH2_freq.out')
        self.assertIsNone(_parse_spin_diagnostic(sp, freq, None, multiplicity=2,
                                                 project_directory=ARC_TESTING_PATH))

    def test_the_expected_value_falls_back_to_the_one_the_ess_reported(self):
        """Test that ORCA's Ideal value S*(S+1) is used when ARC's multiplicity is unknown"""
        sp = os.path.join(ARC_TESTING_PATH, 'neb', 'neb_res.out')
        sd = _parse_spin_diagnostic(sp, None, None, multiplicity=None, project_directory=ARC_TESTING_PATH)
        self.assertIsNotNone(sd)
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


class TestGetFreqHessianMethod(unittest.TestCase):
    """Tests for _get_freq_hessian_method."""

    orca_bare_freq_log = os.path.join(ARC_TESTING_PATH, 'freq', 'orca_neg_freq_ts.out')
    orca_anfreq_log = os.path.join(ARC_TESTING_PATH, 'freq', 'orca_example_freq.log')
    gaussian_wrapped_route_log = os.path.join(ARC_TESTING_PATH, 'composite',
                                              'TS_butylene_intra_H_migration.out')

    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.scratch, True)

    def deck(self, name: str, text: str) -> str:
        """Write ``text`` to ``name`` under the class scratch directory and return the name."""
        with open(os.path.join(self.scratch, name), 'w') as handle:
            handle.write(text)
        return name

    def gaussian(self, name: str, route: str, level) -> str | None:
        """Resolve the Hessian method of a Gaussian deck holding ``route``."""
        deck = self.deck(name, self.gaussian_deck_text(route))
        return _get_freq_hessian_method(deck, None, 'gaussian', level, None, self.scratch)

    @staticmethod
    def gaussian_deck_text(route: str) -> str:
        """Return a minimal Gaussian input deck carrying ``route``."""
        return f'%chk=check.chk\n{route}\n\ntitle\n\n0 1\nC 0.0 0.0 0.0\n'

    @staticmethod
    def gaussian_log_text(route: str) -> str:
        """Return a minimal Gaussian log whose echoed route section holds ``route``."""
        return ('  Gaussian 16:  ES64L-G16RevC.01\n'
                ' ' + '-' * 70 + '\n'
                f' {route}\n'
                ' ' + '-' * 70 + '\n'
                ' Normal termination of Gaussian 16.\n')

    def test_gaussian_bare_freq_with_a_dft_level_is_analytic(self):
        """The bare ``freq`` ARC writes plus a DFT functional resolves to analytic."""
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        route = '#P guess=read uwb97xd/def2tzvp freq IOp(7/33=1) scf=(tight, direct)'
        self.assertEqual(self.gaussian('dft.gjf', route, level), 'analytic')

    def test_gaussian_freq_numer_is_a_gradient_finite_difference(self):
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        route = '#P uwb97xd/def2tzvp freq=(numer) scf=tight'
        self.assertEqual(self.gaussian('numer.gjf', route, level), 'finite_difference_gradient')

    def test_gaussian_freq_enonly_is_an_energy_finite_difference(self):
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        route = '#P uwb97xd/def2tzvp freq=enonly'
        self.assertEqual(self.gaussian('enonly.gjf', route, level), 'finite_difference_energy')

    def test_gaussian_ccsd_t_has_no_analytic_gradients(self):
        """CCSD(T) has neither analytic second derivatives nor analytic gradients."""
        level = Level(method='ccsd(t)', basis='cc-pvtz', software='gaussian')
        route = '#P uccsd(t)/cc-pvtz freq IOp(7/33=1)'
        self.assertEqual(self.gaussian('ccsdt.gjf', route, level), 'finite_difference_energy')

    def test_gaussian_mp2_has_analytic_second_derivatives(self):
        level = Level(method='mp2', basis='cc-pvtz', software='gaussian')
        route = '#P ump2/cc-pvtz freq IOp(7/33=1)'
        self.assertEqual(self.gaussian('mp2.gjf', route, level), 'analytic')

    def test_gaussian_bare_mp4_means_mp4_sdtq(self):
        """``mp4`` alone is MP4(SDTQ), which has no analytic gradients."""
        level = Level(method='mp4', basis='cc-pvtz', software='gaussian')
        route = '#P mp4/cc-pvtz freq IOp(7/33=1)'
        self.assertEqual(self.gaussian('mp4.gjf', route, level), 'finite_difference_energy')

    def test_gaussian_mp4_sdq_has_analytic_gradients(self):
        level = Level(method='mp4(sdq)', basis='cc-pvtz', software='gaussian')
        route = '#P mp4(sdq)/cc-pvtz freq IOp(7/33=1)'
        self.assertEqual(self.gaussian('mp4sdq.gjf', route, level), 'finite_difference_gradient')

    def test_gaussian_cisd_is_not_cis(self):
        """CIS has analytic second derivatives; CISD only has analytic gradients."""
        cis_level = Level(method='cis', basis='cc-pvtz', software='gaussian')
        cisd_level = Level(method='cisd', basis='cc-pvtz', software='gaussian')
        self.assertEqual(self.gaussian('cis.gjf', '#P cis/cc-pvtz freq', cis_level), 'analytic')
        self.assertEqual(self.gaussian('cisd.gjf', '#P cisd/cc-pvtz freq', cisd_level),
                         'finite_difference_gradient')

    def test_gaussian_rohf_gets_numerical_frequencies(self):
        """Gaussian has analytic second derivatives for RHF/UHF but not for ROHF."""
        level = Level(method='rohf', basis='cc-pvtz', software='gaussian')
        route = '#P rohf/cc-pvtz freq IOp(7/33=1)'
        self.assertEqual(self.gaussian('rohf.gjf', route, level), 'finite_difference_gradient')

    def test_gaussian_uhf_is_analytic(self):
        level = Level(method='uhf', basis='cc-pvtz', software='gaussian')
        route = '#P uhf/cc-pvtz freq IOp(7/33=1)'
        self.assertEqual(self.gaussian('uhf.gjf', route, level), 'analytic')

    def test_gaussian_ro_correlated_methods_are_energies_only(self):
        level = Level(method='romp2', basis='cc-pvtz', software='gaussian')
        route = '#P romp2/cc-pvtz freq IOp(7/33=1)'
        self.assertEqual(self.gaussian('romp2.gjf', route, level), 'finite_difference_energy')

    def test_gaussian_ro_dft_is_undocumented(self):
        level = Level(method='rob3lyp', basis='def2tzvp', software='gaussian')
        route = '#P rob3lyp/def2tzvp freq IOp(7/33=1)'
        self.assertIsNone(self.gaussian('rob3lyp.gjf', route, level))

    def test_gaussian_unlisted_wavefunction_method_is_null(self):
        level = Level(method='nevpt2', basis='cc-pvtz', software='gaussian')
        route = '#P nevpt2/cc-pvtz freq IOp(7/33=1)'
        self.assertIsNone(self.gaussian('nevpt2.gjf', route, level))

    def test_gaussian_composite_level_is_null(self):
        """A composite method's internal frequency step is not ARC's request to describe."""
        level = Level(method='cbs-qb3', software='gaussian')
        route = '#P cbs-qb3 freq IOp(7/33=1)'
        self.assertIsNone(self.gaussian('composite.gjf', route, level))

    def test_gaussian_undocumented_double_hybrid_is_null(self):
        level = Level(method='dsdpbep86', basis='def2tzvp', software='gaussian')
        route = '#P dsdpbep86/def2tzvp freq IOp(7/33=1)'
        self.assertIsNone(self.gaussian('dsd.gjf', route, level))

    def test_gaussian_route_without_a_freq_keyword_is_null(self):
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        route = '#P uwb97xd/def2tzvp opt=(calcfc,tight)'
        self.assertIsNone(self.gaussian('opt_only.gjf', route, level))

    def test_gaussian_freq_tolerates_whitespace_around_the_equals_sign(self):
        """Gaussian's route is free-format, so ``Freq = Numer`` is the same keyword."""
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        self.assertEqual(self.gaussian('ws_numer.gjf', '#P uwb97xd/def2tzvp Freq = Numer', level),
                         'finite_difference_gradient')
        self.assertEqual(self.gaussian('ws_enonly.gjf', '#P uwb97xd/def2tzvp Freq = EnOnly', level),
                         'finite_difference_energy')
        self.assertEqual(self.gaussian('ws_paren.gjf',
                                       '#P uwb97xd/def2tzvp Freq = (NoRaman, Numer)', level),
                         'finite_difference_gradient')

    def test_gaussian_freq_numerical_spelling_is_a_gradient_finite_difference(self):
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        route = '#P uwb97xd/def2tzvp Freq=Numerical scf=tight'
        self.assertEqual(self.gaussian('numerical.gjf', route, level),
                         'finite_difference_gradient')

    def test_gaussian_freq_keyword_neighbours_are_not_mistaken_for_options(self):
        """A following keyword, a parenthesised ``scf`` and ``opt=calcnumerfc`` leave freq bare."""
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        for name, route in (('nb_scf.gjf', '#P uwb97xd/def2tzvp freq scf=(numer)'),
                            ('nb_calcnumerfc.gjf', '#P uwb97xd/def2tzvp opt=calcnumerfc freq'),
                            ('nb_iop.gjf', '#P uwb97xd/def2tzvp freq IOp(7/33=1)')):
            self.assertEqual(self.gaussian(name, route, level), 'analytic')

    def test_gaussian_freq_keyword_does_not_match_a_longer_word(self):
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        for name, route in (('freqchk.gjf', '#P uwb97xd/def2tzvp freqchk'),
                            ('anharm.gjf', '#P uwb97xd/def2tzvp anharmonicfreq')):
            self.assertIsNone(self.gaussian(name, route, level))

    def test_gaussian_correlated_option_forms_are_null_not_analytic(self):
        """An MP4/MP5 spelling outside the tables is not a DFT functional."""
        for index, method in enumerate(('mp4(full)', 'MP4(SDTQ,Full)', 'mp4(sdq,full)',
                                        'mp5(full)', 'mp4(fc)', 'mp4(dq,full)', 'mp4(t)',
                                        'mp4sdq(full)')):
            level = Level(method=method, basis='cc-pvtz', software='gaussian')
            self.assertEqual(level.method_type, 'dft')
            self.assertIsNone(self.gaussian(f'correlated_{index}.gjf',
                                            f'#P {method}/cc-pvtz freq', level))

    def test_gaussian_dft_functionals_starting_like_a_correlated_method_stay_analytic(self):
        """A DFT functional whose name opens with a correlated-method prefix stays analytic."""
        for index, method in enumerate(('b3lyp', 'mpw3lyp', 'mpw2plyp', 'b2plyp', 'b2plypd3',
                                        'b97d3', 'wb97xd')):
            level = Level(method=method, basis='def2tzvp', software='gaussian')
            self.assertEqual(level.method_type, 'dft')
            self.assertEqual(self.gaussian(f'dftlike_{index}.gjf',
                                           f'#P {method}/def2tzvp freq', level), 'analytic')

    def test_gaussian_correlated_guard_spares_dft_exchange_functionals(self):
        """``HFS``/``HFB``/``HF-3c`` are not Hartree-Fock spellings the guard may claim."""
        for method in ('hfs', 'hfb', 'hf3c', 'cam-b3lyp', 'b3lyp', 'mpw3lyp', 'mpw2plyp',
                       'b2plyp', 'wb97xd', 'mn15', 'm06-2x', 'hse06', 'bhandhlyp', 'cid'):
            self.assertIsNone(GAUSSIAN_CORRELATED_METHOD_REGEX.match(method))
        for method in ('mp4(full)', 'mp4sdq(full)', 'mp5(full)', 'ccsd(full)', 'cisd(full)',
                       'qcisd(full)', 'casscf(8,8)', 'bd(tq)', 'hf'):
            self.assertIsNotNone(GAUSSIAN_CORRELATED_METHOD_REGEX.match(method))

    def test_gaussian_double_hybrid_exclusion_matches_a_substring(self):
        """Spelling variants of the excluded double hybrids are excluded too."""
        for index, method in enumerate(('revdsdpbep86', 'dsd-blyp', 'pwpb95', 'b2gpplyp',
                                        'wb97x-2', 'pbe-qidh')):
            level = Level(method=method, basis='def2tzvp', software='gaussian')
            self.assertIsNone(self.gaussian(f'dh_{index}.gjf',
                                            f'#P {method}/def2tzvp freq', level))

    def test_gaussian_deck_wins_over_the_log(self):
        """The deck is ARC's request; the log only echoes it."""
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        deck = self.deck('precedence.gjf',
                         self.gaussian_deck_text('#P uwb97xd/def2tzvp freq'))
        log = self.deck('precedence.log',
                        self.gaussian_log_text('#P uwb97xd/def2tzvp freq=numer'))
        self.assertEqual(_get_freq_hessian_method(deck, log, 'gaussian', level, None, self.scratch),
                         'analytic')

    def test_gaussian_log_only_resolves_from_the_logs_route(self):
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        log = self.deck('log_only.log', self.gaussian_log_text('#P uwb97xd/def2tzvp freq=numer'))
        self.assertEqual(_get_freq_hessian_method(None, log, 'gaussian', level, None, self.scratch),
                         'finite_difference_gradient')

    def test_gaussian_absent_deck_falls_back_to_the_log(self):
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        log = self.deck('fallback.log', self.gaussian_log_text('#P uwb97xd/def2tzvp freq=enonly'))
        self.assertEqual(_get_freq_hessian_method('no_such_deck.gjf', log, 'gaussian',
                                                  level, None, self.scratch),
                         'finite_difference_energy')

    def test_gaussian_empty_deck_is_null(self):
        """A deck holding no route yields no value, even though the file is on disk."""
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        deck = self.deck('empty.gjf', '')
        self.assertIsNone(_get_freq_hessian_method(deck, None, 'gaussian', level,
                                                   None, self.scratch))

    def test_gaussian_wrapped_log_route_is_rejoined(self):
        """A real log wraps the route mid-token; the halves rejoin without a space."""
        route = _gaussian_route_text(self.gaussian_wrapped_route_log)
        self.assertIn('integral=(grid=ultrafine', route)
        level = Level(method='b3lyp', basis='cbsb7', software='gaussian')
        self.assertEqual(_get_freq_hessian_method(None, self.gaussian_wrapped_route_log,
                                                  'gaussian', level, None, self.scratch),
                         'analytic')

    def test_orca_numfreq_deck_is_a_gradient_finite_difference(self):
        level = Level(method='wb97xd', basis='def2tzvp', software='orca')
        deck = self.deck('numfreq.in', '! wb97x-d3 def2-tzvp tightscf NumFreq\n* xyz 0 1\n*\n')
        self.assertEqual(_get_freq_hessian_method(deck, None, 'orca', level, None, self.scratch),
                         'finite_difference_gradient')

    def test_orca_numfreq_over_numgrad_is_an_energy_finite_difference(self):
        """Differentiating a numerical gradient makes the Hessian an energy difference."""
        level = Level(method='wb97xd', basis='def2tzvp', software='orca')
        deck = self.deck('numfreq_numgrad.in', '! wb97x-d3 def2-tzvp NumFreq NumGrad\n')
        self.assertEqual(_get_freq_hessian_method(deck, None, 'orca', level, None, self.scratch),
                         'finite_difference_energy')

    def test_orca_anfreq_deck_is_analytic(self):
        level = Level(method='b3lyp', basis='def2tzvp', software='orca')
        deck = self.deck('anfreq.in', '! RKS B3LYP def2-tzvp TightSCF AnFreq\n')
        self.assertEqual(_get_freq_hessian_method(deck, None, 'orca', level, None, self.scratch),
                         'analytic')

    def test_orca_bare_freq_log_with_a_dft_level_is_analytic(self):
        """A real Orca log whose echoed keyword line is a bare ``!Freq`` (Freq aliases AnFreq)."""
        level = Level(method='wb97xd3', basis='def2tzvp', software='orca')
        self.assertEqual(_get_freq_hessian_method(None, self.orca_bare_freq_log, 'orca',
                                                  level, None, self.scratch),
                         'analytic')

    def test_orca_anfreq_log_is_analytic(self):
        """A real Orca log whose echoed keyword line carries ``AnFreq``."""
        level = Level(method='b3lyp', basis='svp', software='orca')
        self.assertEqual(_get_freq_hessian_method(None, self.orca_anfreq_log, 'orca',
                                                  level, None, self.scratch),
                         'analytic')

    def test_orca_analytic_hessian_marker_is_present_in_the_reference_logs(self):
        """The ``ORCA SCF HESSIAN`` header the log fallback keys on is real, not assumed."""
        for path in (self.orca_bare_freq_log, self.orca_anfreq_log):
            with open(path, 'r', errors='ignore') as handle:
                self.assertIn('ORCA SCF HESSIAN', handle.read())

    def test_orca_marker_fallback_is_analytic_when_no_keyword_line_survives(self):
        """A log whose echoed keyword lines are gone still states an analytic Hessian."""
        with open(self.orca_anfreq_log, 'r', errors='ignore') as handle:
            lines = handle.readlines()
        tail = [line for line in lines if not line.strip().startswith('!')]
        marker_index = next(i for i, line in enumerate(tail) if 'ORCA SCF HESSIAN' in line)
        tail = tail[marker_index - 5:]
        self.assertNotIn('!', ''.join(line.strip()[:1] for line in tail))
        log = self.deck('orca_marker_tail.log', ''.join(tail))
        abs_log = os.path.join(self.scratch, log)
        self.assertIsNone(_orca_route_text(abs_log))
        level = Level(method='b3lyp', basis='svp', software='orca')
        self.assertEqual(_get_freq_hessian_method(None, log, 'orca', level, None, self.scratch),
                         'analytic')

    def test_orca_marker_fallback_is_null_without_the_marker(self):
        """No keyword line and no analytic-Hessian header leaves nothing to state."""
        log = self.deck('orca_no_marker.log',
                        '                       ORCA property calculations\n'
                        '  Total Energy       :         -76.12345678 Eh\n'
                        '  ****ORCA TERMINATED NORMALLY****\n')
        level = Level(method='b3lyp', basis='svp', software='orca')
        self.assertIsNone(_get_freq_hessian_method(None, log, 'orca', level, None, self.scratch))

    def test_orca_route_scan_stops_at_the_end_of_the_echoed_input(self):
        """Keyword-looking lines after ``****END OF INPUT****`` are not part of the route."""
        log = self.deck('orca_end_of_input.log',
                        '| 1> ! RKS B3LYP def2-tzvp AnFreq\n'
                        '| 2>                          ****END OF INPUT****\n'
                        '! NumFreq NumGrad\n')
        abs_log = os.path.join(self.scratch, log)
        self.assertEqual(_orca_route_text(abs_log), 'rks b3lyp def2-tzvp anfreq')
        level = Level(method='b3lyp', basis='def2tzvp', software='orca')
        self.assertEqual(_get_freq_hessian_method(None, log, 'orca', level, None, self.scratch),
                         'analytic')

    def test_orca_bare_freq_with_mp2_is_null(self):
        """Orca deprecated the analytic MP2 Hessian and documents no fallback."""
        level = Level(method='mp2', basis='def2tzvp', software='orca')
        self.assertIsNone(_get_freq_hessian_method(None, self.orca_bare_freq_log, 'orca',
                                                   level, None, self.scratch))

    def test_orca_bare_freq_with_ri_jk_is_null(self):
        level = Level(method='b3lyp', basis='def2tzvp', software='orca')
        deck = self.deck('rijk.in', '! B3LYP def2-tzvp RI-JK def2/JK Freq\n')
        self.assertIsNone(_get_freq_hessian_method(deck, None, 'orca', level, None, self.scratch))

    def test_orca_bare_freq_with_a_double_hybrid_is_null(self):
        level = Level(method='b2plyp', basis='def2tzvp', software='orca')
        deck = self.deck('b2plyp.in', '! B2PLYP def2-tzvp Freq\n')
        self.assertIsNone(_get_freq_hessian_method(deck, None, 'orca', level, None, self.scratch))

    def test_pyscf_closed_shell_is_analytic(self):
        """pyscf_script builds an analytic RKS Hessian for multiplicity 1."""
        level = Level(method='wb97m-v', basis='def2tzvp', software='pyscf')
        spc = ARCSpecies(label='CH4', smiles='C')
        deck = self.deck('pyscf_singlet.yml', 'job_type: freq\n')
        self.assertEqual(_get_freq_hessian_method(deck, None, 'pyscf', level, spc, self.scratch),
                         'analytic')

    def test_pyscf_open_shell_is_a_gradient_finite_difference(self):
        """pyscf_script differentiates analytic gradients for any multiplicity above 1."""
        level = Level(method='wb97m-v', basis='def2tzvp', software='pyscf')
        spc = ARCSpecies(label='CH3', smiles='[CH3]')
        deck = self.deck('pyscf_doublet.yml', 'job_type: freq\n')
        self.assertEqual(_get_freq_hessian_method(deck, None, 'pyscf', level, spc, self.scratch),
                         'finite_difference_gradient')

    def test_pyscf_unusable_multiplicity_is_null(self):
        """A multiplicity that is not an integer value yields no Hessian method."""
        level = Level(method='wb97m-v', basis='def2tzvp', software='pyscf')
        deck = self.deck('pyscf_bad_mult.yml', 'job_type: freq\n')
        for value in ('doublet', None, [2]):
            spc = MagicMock(multiplicity=value)
            self.assertIsNone(_get_freq_hessian_method(deck, None, 'pyscf', level,
                                                       spc, self.scratch))

    def test_unknown_ess_is_null(self):
        level = Level(method='ccsd(t)', basis='cc-pvtz', software='molpro')
        deck = self.deck('molpro.in', '#P ccsd(t)/cc-pvtz freq\n')
        self.assertIsNone(_get_freq_hessian_method(deck, None, 'molpro', level, None, self.scratch))

    def test_no_software_is_null(self):
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        deck = self.deck('no_software.gjf', '#P uwb97xd/def2tzvp freq\n')
        self.assertIsNone(_get_freq_hessian_method(deck, None, None, level, None, self.scratch))

    def test_missing_deck_and_log_is_null(self):
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        self.assertIsNone(_get_freq_hessian_method('nonexistent.gjf', '/also/missing.log',
                                                   'gaussian', level, None, self.scratch))


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

    def test_matched_keys_respect_dispersion_and_solvation(self):
        """The matched Arkane keys fold a separate dispersion field into the method, and a solvated level has none,
        so no gas-phase or non-dispersion table is looked up for it."""
        with patch('arc.output.execute_command') as mock_exec:
            for lot in (Level(method='b3lyp', basis='def2tzvp', software='gaussian', dispersion='gd3bj'),
                        Level(method='wb97xd', basis='def2tzvp', software='gaussian',
                              solvation_method='smd', solvent='water')):
                with self.subTest(level=str(lot)):
                    self.assertEqual(_match_arkane_correction_keys(lot, 'p'), (None, None))
                    self.assertEqual(_get_energy_corrections(lot, 'p'), EnergyCorrections(None, None, None, None))
        mock_exec.assert_not_called()
        key = "LevelOfTheory(method='b2plypd32023',basis='def2tzvp',software='gaussian')"
        self.assertEqual(_match_arkane_correction_keys(
            Level(method='b2plyp', basis='def2tzvp', software='gaussian', dispersion='gd3'), 'p'), (key, key))

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
        spc.e0_atom_corrections_applied = None
        spc.e0_bond_corrections_applied = None
        spc.arkane_rotor_modes = None
        spc._is_linear = False
        spc.optical_isomers = 1
        spc.external_symmetry = 1
        spc.freqs = [-1500.0, 100.0]
        spc.rotors_dict = None
        spc.thermo = None
        spc.rxn_label = 'CHO + CH4 <=> CH2O + CH3'
        spc.chosen_ts_method = 'heuristics'
        spc.successful_methods = ['heuristics']
        spc.number_of_radicals = None
        spc.derived_stability_verdict = None
        spc.scf_references = dict()
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
        spc.e0_atom_corrections_applied = None
        spc.e0_bond_corrections_applied = None
        spc.arkane_rotor_modes = None
        spc._is_linear = False
        spc.optical_isomers = 1
        spc.external_symmetry = 1
        spc.freqs = [-1500.0, 100.0]
        spc.rotors_dict = None
        spc.thermo = None
        spc.rxn_label = 'A <=> B'
        spc.chosen_ts_method = None
        spc.successful_methods = []
        spc.number_of_radicals = None
        spc.derived_stability_verdict = None
        spc.scf_references = dict()
        output_dict = {'TS1': {'convergence': True, 'paths': {'irc': []}, 'job_types': {}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIsNone(result['smiles'])
        self.assertIsNone(result['formula'])

    def test_ts_path_logs_and_irc_directions(self):
        spc = MagicMock()
        spc.e0_atom_corrections_applied = None
        spc.e0_bond_corrections_applied = None
        spc.arkane_rotor_modes = None
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
        spc.number_of_radicals = None
        spc.derived_stability_verdict = None
        spc.scf_references = dict()
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
        spc.e0_atom_corrections_applied = None
        spc.e0_bond_corrections_applied = None
        spc.arkane_rotor_modes = None
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
        spc.number_of_radicals = None
        spc.derived_stability_verdict = None
        spc.scf_references = dict()
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
        spc.e0_atom_corrections_applied = None
        spc.e0_bond_corrections_applied = None
        spc.arkane_rotor_modes = None
        spc._is_linear = False
        spc.optical_isomers = 1
        spc.external_symmetry = 12
        spc.freqs = [1300.0, 1500.0, 3000.0, 3100.0]
        spc.rotors_dict = None
        spc.thermo = ThermoData(H298=-74.6, S298=186.3, Tmin=(300, 'K'), Tmax=(3000, 'K'))
        spc.rxn_label = None
        spc.ts_guesses = []
        spc.chosen_ts = None
        spc.number_of_radicals = None
        spc.derived_stability_verdict = None
        spc.scf_references = dict()
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

    def test_freq_hessian_method_is_emitted_for_a_converged_gaussian_freq_job(self):
        """The species record carries the freq job's Hessian method beside its deck path."""
        scratch = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, scratch, True)
        freq_dir = os.path.join(scratch, 'calcs', 'Species', 'CH4', 'freq')
        os.makedirs(freq_dir)
        with open(os.path.join(freq_dir, 'input.gjf'), 'w') as handle:
            handle.write('%chk=check.chk\n#P uwb97xd/def2tzvp freq IOp(7/33=1)\n\nCH4\n\n0 1\nC 0.0 0.0 0.0\n')
        spc = self._make_spc_mock()
        output_dict = {'CH4': {'convergence': True,
                               'paths': {'freq': os.path.join(freq_dir, 'input.log')},
                               'job_types': {'opt': True}}}
        result = _spc_to_dict(spc, output_dict, scratch,
                              software_by_job={'freq': 'gaussian'},
                              freq_level=Level(method='wb97xd', basis='def2tzvp', software='gaussian'))
        self.assertEqual(result['freq_hessian_method'], 'analytic')

    def test_freq_hessian_method_is_null_without_a_freq_level(self):
        """Without the freq level ARC cannot place a bare ``freq`` route, so it says nothing."""
        spc = self._make_spc_mock()
        output_dict = {'CH4': {'convergence': True, 'paths': {'freq': '/abs/freq.log'},
                               'job_types': {'opt': True}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertIn('freq_hessian_method', result)
        self.assertIsNone(result['freq_hessian_method'])

    def test_non_converged_species(self):
        spc = self._make_spc_mock(converged=False)
        output_dict = {'CH4': {'convergence': False, 'paths': {}, 'job_types': {}}}
        result = _spc_to_dict(spc, output_dict, '/abs')
        self.assertFalse(result['converged'])
        self.assertIsNone(result['freq_hessian_method'])
        self.assertIsNone(result['sp_energy_hartree'])
        self.assertIsNone(result['zpe_hartree'])
        self.assertIsNone(result['freq_n_imag'])
        self.assertIsNone(result['thermo'])
        self.assertIsNone(result['statmech'])

    def test_a_non_converged_species_carries_no_spin_diagnostic(self):
        """Test that a readable sp log does not give an unconverged species an sp_spin_diagnostic"""
        sp = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'Species', 'NH2_freq.out')
        spc = self._make_spc_mock(label='NH2')
        spc.multiplicity = 2
        output_dict = {'NH2': {'convergence': True, 'paths': {'sp': sp}, 'job_types': {}}}
        self.assertIsNotNone(_spc_to_dict(spc, output_dict, ARC_TESTING_PATH)['sp_spin_diagnostic'])
        output_dict['NH2']['convergence'] = False
        self.assertIsNone(_spc_to_dict(spc, output_dict, ARC_TESTING_PATH)['sp_spin_diagnostic'])

    def test_scf_reference_records_a_declared_source(self):
        """Test that a user-declared number_of_radicals is reported as the deciding source"""
        spc = self._make_spc_mock()
        spc.number_of_radicals = 2
        output_dict = {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}
        block = _spc_to_dict(spc, output_dict, '/abs')['scf_reference']
        self.assertEqual(block['source'], 'declared')
        self.assertEqual(block['declared_number_of_radicals'], 2)
        self.assertIsNone(block['verdict'])

    def test_scf_reference_records_a_derived_source_and_its_verdict(self):
        """Test that an adopted measured verdict is reported as the deciding source"""
        spc = self._make_spc_mock(is_ts=True)
        spc.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        output_dict = {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}
        block = _spc_to_dict(spc, output_dict, '/abs')['scf_reference']
        self.assertEqual(block['source'], 'derived')
        self.assertEqual(block['verdict'], 'external_instability')
        self.assertIs(block['verdict_restricted'], True)
        self.assertIsNone(block['declared_number_of_radicals'])

    def test_a_well_records_its_verdict_without_being_credited_with_a_decision(self):
        """Test that a species that is not a TS records the same verdict under no deciding source"""
        spc = self._make_spc_mock()
        spc.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        output_dict = {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}
        entry = _spc_to_dict(spc, output_dict, '/abs')
        self.assertFalse(entry['is_ts'])
        block = entry['scf_reference']
        self.assertEqual(block['verdict'], 'external_instability')
        self.assertIs(block['verdict_restricted'], True)
        self.assertIsNone(block['source'])

    def test_a_well_records_the_stability_log_it_was_measured_from(self):
        """Test that the stability diagnostic reaches output.yml for a species that is not a TS"""
        stability = os.path.join(ARC_TESTING_PATH, 'stability', 'rhf_uhf_instability_singlet_ts.out')
        spc = self._make_spc_mock()
        output_dict = {'CH4': {'convergence': True, 'paths': {'stability': stability}, 'job_types': {}}}
        entry = _spc_to_dict(spc, output_dict, ARC_TESTING_PATH)
        self.assertFalse(entry['is_ts'])
        self.assertEqual(entry['wavefunction_stability']['verdict'], 'external_instability')
        self.assertEqual(entry['scf_reference']['log'],
                         os.path.join('stability', 'rhf_uhf_instability_singlet_ts.out'))

    def test_scf_reference_reports_a_declared_source_over_a_contradicting_verdict(self):
        """Test that a declared value is still the reported source where the verdict disagrees"""
        spc = self._make_spc_mock(is_ts=True)
        spc.number_of_radicals = 2
        spc.derived_stability_verdict = {'verdict': 'stable', 'restricted': True}
        output_dict = {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}
        block = _spc_to_dict(spc, output_dict, '/abs')['scf_reference']
        self.assertEqual(block['source'], 'declared')
        self.assertEqual(block['verdict'], 'stable')

    def test_a_declaration_attributing_no_open_shell_character_names_no_source(self):
        """Test that a declared 1 blocks the verdict and is reported without being called the source"""
        spc = self._make_spc_mock(is_ts=True)
        spc.number_of_radicals = 1
        spc.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        output_dict = {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}
        block = _spc_to_dict(spc, output_dict, '/abs')['scf_reference']
        self.assertIsNone(block['source'])
        self.assertEqual(block['declared_number_of_radicals'], 1)
        self.assertEqual(block['verdict'], 'external_instability')

    def test_scf_reference_reports_a_mixed_reference(self):
        """Test that an electronic energy and a ZPE from different references are recorded as such"""
        spc = self._make_spc_mock()
        output_dict = {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}
        spc.scf_references = {'sp': 'unrestricted', 'freq': 'restricted'}
        block = _spc_to_dict(spc, output_dict, '/abs')['scf_reference']
        self.assertEqual(block['sp_reference'], 'unrestricted')
        self.assertEqual(block['freq_reference'], 'restricted')
        self.assertTrue(block['reference_mismatch'])
        spc.scf_references = {'sp': 'unrestricted', 'freq': 'unrestricted'}
        self.assertIs(_spc_to_dict(spc, output_dict, '/abs')['scf_reference']['reference_mismatch'], False)

    def test_an_unrecorded_reference_is_reported_as_unknown_rather_than_consistent(self):
        """Test that a missing sp or freq reference is None, not the False that means checked and equal"""
        spc = self._make_spc_mock()
        output_dict = {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}
        for references in [dict(), {'freq': 'restricted'}, {'sp': 'restricted'}, 'not a dict']:
            spc.scf_references = references
            block = _spc_to_dict(spc, output_dict, '/abs')['scf_reference']
            self.assertIsNone(block['reference_mismatch'], msg=f'{references} reported a mismatch verdict')

    def test_scf_reference_names_the_ts_guess_a_carried_verdict_was_measured_on(self):
        """Test that a verdict carried over from an abandoned TS guess says which guess it came from"""
        spc = self._make_spc_mock()
        spc.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True,
                                         'measured_on_ts_guess': 3}
        output_dict = {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}
        self.assertEqual(_spc_to_dict(spc, output_dict, '/abs')['scf_reference']['measured_on_ts_guess'], 3)

    def test_a_carried_verdict_still_names_the_analysis_that_decided_the_reference(self):
        """Test that a TS switch, which resets the stability path, leaves no source without a log"""
        stability = os.path.join(ARC_TESTING_PATH, 'stability', 'rhf_uhf_instability_singlet_ts.out')
        spc = self._make_spc_mock(is_ts=True)
        spc.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True,
                                         'measured_on_ts_guess': 3, 'log': stability}
        output_dict = {'CH4': {'convergence': True, 'paths': {'stability': ''}, 'job_types': {}}}
        entry = _spc_to_dict(spc, output_dict, ARC_TESTING_PATH)
        self.assertIsNone(entry['wavefunction_stability'])
        block = entry['scf_reference']
        self.assertEqual(block['source'], 'derived')
        self.assertEqual(block['measured_on_ts_guess'], 3)
        self.assertEqual(block['log'], os.path.join('stability', 'rhf_uhf_instability_singlet_ts.out'))

    def test_a_verdict_carrying_no_log_reports_none_rather_than_raising(self):
        """Test that a verdict restored from a restart written without a log path is still reported"""
        spc = self._make_spc_mock(is_ts=True)
        spc.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        output_dict = {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}
        block = _spc_to_dict(spc, output_dict, ARC_TESTING_PATH)['scf_reference']
        self.assertEqual(block['source'], 'derived')
        self.assertIsNone(block['log'])

    def test_a_non_converged_species_still_carries_its_scf_reference_block(self):
        """Test that the record of what ARC decided survives the species failing to converge"""
        spc = self._make_spc_mock(converged=False, is_ts=True)
        spc.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        spc.scf_references = {'freq': 'restricted', 'sp': 'unrestricted'}
        output_dict = {'CH4': {'convergence': False, 'paths': {}, 'job_types': {}}}
        entry = _spc_to_dict(spc, output_dict, '/abs')
        self.assertFalse(entry['converged'])
        self.assertIsNone(entry['sp_spin_diagnostic'])
        block = entry['scf_reference']
        self.assertEqual(block['source'], 'derived')
        self.assertEqual(block['verdict'], 'external_instability')
        self.assertIs(block['reference_mismatch'], True)

    def test_scf_reference_reads_a_verdict_off_the_log_without_calling_it_the_source(self):
        """Test that a verdict only the log holds is reported but is not credited with the decision"""
        stability = os.path.join(ARC_TESTING_PATH, 'stability', 'rhf_uhf_instability_singlet_ts.out')
        spc = self._make_spc_mock()
        output_dict = {'CH4': {'convergence': True, 'paths': {'stability': stability}, 'job_types': {}}}
        block = _spc_to_dict(spc, output_dict, ARC_TESTING_PATH)['scf_reference']
        self.assertEqual(block['verdict'], 'external_instability')
        self.assertEqual(block['log'], os.path.join('stability', 'rhf_uhf_instability_singlet_ts.out'))
        self.assertIsNone(block['source'])

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
        ts_guess.index = 0
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


class TestSpeciesLevelsAndIrcEndpoints(unittest.TestCase):
    """Tests for the per-record ``levels`` object, the header ``adaptive_levels`` and the IRC endpoint markers."""

    OPT = {'method': 'wb97xd', 'basis': 'def2-tzvp', 'software': 'gaussian'}
    FREQ = {'method': 'wb97xd', 'basis': 'def2-svp', 'software': 'gaussian'}
    SP = {'method': 'dlpno-ccsd(t)', 'basis': 'cc-pvtz', 'software': 'orca', 'repr': 'dlpno-ccsd(t)/cc-pvtz',
          'compatible_ess': ['orca']}

    def make_spc(self, label='CH4', is_ts=False, irc_label=None):
        spc = TestSpcToDict._make_spc_mock(self, label=label, is_ts=is_ts)
        spc.irc_label = irc_label
        if is_ts:
            spc.thermo = None
            spc.rxn_label = 'A <=> B'
        return spc

    def entry(self, label='CH4', **kwargs):
        return {label: {'convergence': True, 'paths': {}, 'job_types': {}, **kwargs}}

    def test_a_non_adaptive_species_exports_the_recorded_levels(self):
        """Test that every level key is present, recorded ones are level dicts without repr and compatible_ess"""
        result = _spc_to_dict(self.make_spc(), self.entry(levels={'opt': self.OPT, 'freq': self.OPT, 'sp': self.SP}),
                              '/abs')
        self.assertEqual(list(result['levels']), ['opt', 'freq', 'sp', 'composite', 'irc'])
        self.assertEqual(result['levels']['opt'], {'method': 'wb97xd', 'basis': 'def2-tzvp'})
        self.assertEqual(result['levels']['sp'], {'method': 'dlpno-ccsd(t)', 'basis': 'cc-pvtz'})
        self.assertIsNone(result['levels']['composite'])
        self.assertIsNone(result['levels']['irc'])

    def test_levels_differ_between_species_and_a_ts_copy_under_adaptive_levels(self):
        """Test that each record exports its own levels, including an adaptive-level ``<label>_TS<i>`` copy"""
        plain = _spc_to_dict(self.make_spc('CH4'), self.entry('CH4', levels={'opt': self.OPT}), '/abs')
        ts_copy = _spc_to_dict(self.make_spc('CH4_TS0'), self.entry('CH4_TS0', levels={'opt': self.FREQ}), '/abs')
        self.assertEqual(plain['levels']['opt'], {'method': 'wb97xd', 'basis': 'def2-tzvp'})
        self.assertEqual(ts_copy['levels']['opt'], {'method': 'wb97xd', 'basis': 'def2-svp'})

    def test_levels_are_null_where_nothing_was_recorded(self):
        """Test that an entry with no levels (an older restart), an empty one, or no entry at all exports all null"""
        for output_dict in (self.entry(), self.entry(levels=dict()), self.entry(levels=None), dict()):
            with self.subTest(output_dict=output_dict):
                result = _spc_to_dict(self.make_spc(), output_dict, '/abs')
                self.assertEqual(result['levels'], {key: None for key in ('opt', 'freq', 'sp', 'composite', 'irc')})

    def test_irc_level_is_exported_for_a_ts_only(self):
        """Test that the irc level is exported on a TS record and is null on a non-TS record"""
        levels = {'opt': self.OPT, 'irc': self.FREQ}
        ts = _spc_to_dict(self.make_spc('TS0', is_ts=True), self.entry('TS0', levels=levels), '/abs')
        species = _spc_to_dict(self.make_spc(), self.entry(levels=levels), '/abs')
        self.assertEqual(ts['levels']['irc'], {'method': 'wb97xd', 'basis': 'def2-svp'})
        self.assertIsNone(species['levels']['irc'])

    def test_a_level_stored_by_the_scheduler_exports_without_the_scheme_and_the_software(self):
        """Test the shape the scheduler stores for a solvation scheme level: no scheme, no deduced software"""
        level = Level(method='wb97xd', basis='def2-tzvp', software='gaussian', solvation_method='smd',
                      solvent='water', solvation_scheme_level=Level(method='b3lyp', basis='6-31g'))
        levels = dict()
        set_recorded_level(levels, 'sp', level)
        self.assertNotIn('solvation_scheme_level', levels['sp'])
        result = _spc_to_dict(self.make_spc(), self.entry(levels=levels), '/abs')
        self.assertEqual(result['levels']['sp'], {'method': 'wb97xd', 'basis': 'def2-tzvp', 'method_type': 'dft',
                                                  'solvation_method': 'smd', 'solvent': 'water'})

    def test_the_deduced_software_is_not_exported_in_a_recorded_level(self):
        """Test that a record's levels never state a software, whatever the stored level carries"""
        result = _spc_to_dict(self.make_spc(), self.entry(levels={'opt': self.OPT, 'sp': self.SP}), '/abs')
        for key in ('opt', 'sp'):
            self.assertNotIn('software', result['levels'][key])

    def test_an_irc_endpoint_species_names_its_ts_and_direction(self):
        """Test the endpoint markers of a species created from an IRC, and that <n> in the label is not the direction"""
        result = _spc_to_dict(self.make_spc('IRC_TS0_1', irc_label='TS0'),
                              self.entry('IRC_TS0_1', irc_direction='reverse'), '/abs')
        self.assertEqual(result['irc_endpoint_of'], 'TS0')
        self.assertEqual(result['irc_endpoint_direction'], 'reverse')

    def test_an_irc_endpoint_without_a_recorded_direction_has_a_null_direction(self):
        """Test that an endpoint restored from a restart with no recorded direction exports null, not a guess"""
        for entry in (self.entry('IRC_TS0_2'), self.entry('IRC_TS0_2', irc_direction=None),
                      self.entry('IRC_TS0_2', irc_direction='sideways')):
            with self.subTest(entry=entry):
                result = _spc_to_dict(self.make_spc('IRC_TS0_2', irc_label='TS0'), entry, '/abs')
                self.assertEqual(result['irc_endpoint_of'], 'TS0')
                self.assertIsNone(result['irc_endpoint_direction'])

    def test_an_ordinary_species_is_not_an_endpoint(self):
        """Test that a species with no irc_label has null markers, whatever a stale output entry says"""
        result = _spc_to_dict(self.make_spc(), self.entry(irc_direction='forward'), '/abs')
        self.assertIsNone(result['irc_endpoint_of'])
        self.assertIsNone(result['irc_endpoint_direction'])

    def test_a_ts_carries_no_endpoint_markers_although_its_irc_label_lists_endpoints(self):
        """Test that the TS's own irc_label (its endpoint labels) is not exported as irc_endpoint_of"""
        result = _spc_to_dict(self.make_spc('TS0', is_ts=True, irc_label='IRC_TS0_1 IRC_TS0_2'),
                              self.entry('TS0'), '/abs')
        self.assertNotIn('irc_endpoint_of', result)
        self.assertNotIn('irc_endpoint_direction', result)

    def test_a_stored_level_that_is_not_a_dict_exports_null(self):
        """Test that a corrupt stored level exports null rather than a schema-invalid stub"""
        result = _spc_to_dict(self.make_spc(), self.entry(levels={'opt': 'wb97xd/def2-tzvp', 'freq': 7}), '/abs')
        self.assertIsNone(result['levels']['opt'])
        self.assertIsNone(result['levels']['freq'])

    def test_the_adaptive_levels_header_uses_the_restart_list_form(self):
        """Test the header list form, the 'inf' bound, joined job types, and null for a non-adaptive run"""
        levels = {(1, 5): {('opt', 'freq'): Level(method='wb97xd', basis='def2-svp'),
                           ('sp',): Level(method='dlpno-ccsd(t)', basis='cc-pvtz')},
                  (6, 'inf'): {('opt', 'freq'): Level(method='b3lyp', basis='6-31g')}}
        result = adaptive_levels_as_list(levels, strip=LEVEL_DICT_STRIPPED_KEYS)
        self.assertEqual([entry['atom_range'] for entry in result], [[1, 5], [6, 'inf']])
        self.assertEqual(set(result[0]['levels']), {'opt freq', 'sp'})
        self.assertEqual(result[0]['levels']['opt freq']['method'], 'wb97xd')
        self.assertNotIn('repr', result[1]['levels']['opt freq'])
        self.assertIsNone(adaptive_levels_as_list(None))

    def test_the_writer_puts_the_adaptive_levels_in_the_header(self):
        """Test that write_output_yml writes adaptive_levels, and null when it is not given"""
        scratch = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, scratch, True)
        documents = dict()
        for name, adaptive in (('adaptive', {(1, 'inf'): {('opt',): Level(method='wb97xd', basis='def2-svp')}}),
                               ('plain', None)):
            with patch('arc.output._compute_point_groups', return_value={}), \
                    patch('arc.output._get_arkane_provenance', return_value=('3.3.0', 'abc123')), \
                    patch('arc.output.get_git_commit', return_value=('def456', '2026-01-01')):
                write_output_yml(project=name, project_directory=scratch, species_dict=dict(), reactions=[],
                                 output_dict=dict(), adaptive_levels=adaptive)
            documents[name] = read_yaml_file(os.path.join(scratch, 'output', 'output.yml'))
        self.assertEqual(documents['adaptive']['adaptive_levels'][0]['atom_range'], [1, 'inf'])
        self.assertIsNone(documents['plain']['adaptive_levels'])

    def test_a_stored_level_dict_is_a_valid_input_of_level_to_dict(self):
        """Test that _level_to_dict accepts the plain dict the scheduler stores"""
        self.assertEqual(_level_to_dict(self.SP), {'method': 'dlpno-ccsd(t)', 'basis': 'cc-pvtz', 'software': 'orca'})
        self.assertIsNone(_level_to_dict(dict()))
        self.assertEqual(_recorded_level_to_dict(self.SP), {'method': 'dlpno-ccsd(t)', 'basis': 'cc-pvtz'})


class TestIrcAndCompositeEssPrograms(unittest.TestCase):
    """Tests for the ``irc`` and ``composite`` keys of ``ess_versions`` and ``ess_software``, and ``composite_*``."""

    gaussian_log = os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')
    orca_log = os.path.join(ARC_TESTING_PATH, 'orca_example_opt.log')

    def test_irc_logs_that_agree_state_the_program_and_banner(self):
        """Both IRC logs from one program export that program and banner under ``irc``"""
        paths = {'irc': [self.gaussian_log, self.gaussian_log]}
        self.assertEqual(_get_ess_software(paths, '/dummy'), {'irc': 'gaussian'})
        self.assertIn('Gaussian 09', _get_ess_versions(paths, '/dummy')['irc'])

    def test_irc_logs_that_disagree_state_nothing(self):
        """Two IRC logs from different programs cannot be stated as one program, so ``irc`` is absent"""
        paths = {'irc': [self.gaussian_log, self.orca_log], 'geo': self.gaussian_log}
        self.assertEqual(_get_ess_software(paths, '/dummy'), {'opt': 'gaussian'})
        self.assertEqual(list(_get_ess_versions(paths, '/dummy')), ['opt'])

    def test_irc_logs_with_different_banners_state_no_version_but_the_program(self):
        """Two logs of one program at different versions agree on the program and not on the banner"""
        paths = {'irc': [self.gaussian_log, self.gaussian_log]}
        with patch('arc.output.parse_ess_version', side_effect=['Gaussian 09, Revision D.01',
                                                                   'Gaussian 16, Revision C.01']), \
                patch('arc.output._existing_log_path', side_effect=lambda path, _: path):
            self.assertIsNone(_get_ess_versions({'irc': ['/a.log', '/b.log']}, '/dummy'))
        self.assertEqual(_get_ess_software(paths, '/dummy'), {'irc': 'gaussian'})

    def test_a_missing_irc_log_leaves_the_program_unstated(self):
        """One recorded IRC log that cannot be read leaves ``irc`` absent, since the pair is not known to agree"""
        paths = {'irc': [self.gaussian_log, '/nonexistent.log']}
        self.assertIsNone(_get_ess_software(paths, '/dummy'))
        self.assertIsNone(_get_ess_versions(paths, '/dummy'))

    def test_a_single_irc_log_states_its_program(self):
        """A run with one IRC log exports it"""
        self.assertEqual(_get_ess_software({'irc': [self.orca_log]}, '/dummy'), {'irc': 'orca'})

    def test_every_exported_irc_log_counts(self):
        """``irc`` covers every IRC log the record exports, so a third log from another program leaves it absent"""
        paths = {'irc': [self.gaussian_log, self.gaussian_log, self.orca_log]}
        self.assertIsNone(_get_ess_software(paths, '/dummy'))
        paths = {'irc': [self.gaussian_log, self.gaussian_log, self.gaussian_log]}
        self.assertEqual(_get_ess_software(paths, '/dummy'), {'irc': 'gaussian'})

    def test_composite_log_states_its_program(self):
        """A composite run's only log is exported under ``composite``"""
        paths = {'composite': self.gaussian_log}
        self.assertEqual(_get_ess_software(paths, '/dummy'), {'composite': 'gaussian'})
        self.assertIn('Gaussian', _get_ess_versions(paths, '/dummy')['composite'])

    def test_composite_log_and_input_are_exported_and_null_for_a_non_composite_run(self):
        """``composite_log`` and ``composite_input`` come from ``paths['composite']`` and are null otherwise"""
        scratch = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, scratch, True)
        calc_dir = os.path.join(scratch, 'calcs', 'Species', 'CH4', 'composite')
        os.makedirs(calc_dir)
        shutil.copyfile(self.gaussian_log, os.path.join(calc_dir, 'output.out'))
        with open(os.path.join(calc_dir, 'input.gjf'), 'w') as f:
            f.write('#p cbs-qb3\n')
        spc = TestSpcToDict._make_spc_mock(self)
        composite = _spc_to_dict(spc, {'CH4': {'convergence': True, 'job_types': {},
                                               'paths': {'composite': os.path.join(calc_dir, 'output.out')}}},
                                 scratch)
        self.assertEqual(composite['composite_log'], os.path.join('calcs', 'Species', 'CH4', 'composite', 'output.out'))
        self.assertEqual(composite['composite_input'], os.path.join('calcs', 'Species', 'CH4', 'composite', 'input.gjf'))
        self.assertEqual(composite['ess_software'], {'composite': 'gaussian'})
        plain = _spc_to_dict(spc, {'CH4': {'convergence': True, 'job_types': {}, 'paths': {'composite': ''}}}, scratch)
        self.assertIsNone(plain['composite_log'])
        self.assertIsNone(plain['composite_input'])
        os.remove(os.path.join(calc_dir, 'input.gjf'))
        no_deck = _spc_to_dict(spc, {'CH4': {'convergence': True, 'job_types': {},
                                             'paths': {'composite': os.path.join(calc_dir, 'output.out')}}}, scratch)
        self.assertIsNone(no_deck['composite_input'])
        self.assertIsNotNone(no_deck['composite_log'])


class TestIrcLogLevels(unittest.TestCase):
    """Tests for the TS-only ``irc_log_levels``."""

    FORWARD = {'method': 'wb97xd', 'basis': 'def2-svp', 'software': 'gaussian', 'repr': 'wb97xd/def2svp'}
    REVERSE = {'method': 'b3lyp', 'basis': '6-31g', 'software': 'gaussian'}

    def _ts(self, **paths):
        spc = TestSpeciesLevelsAndIrcEndpoints.make_spc(self, 'TS0', is_ts=True)
        entry = {'TS0': {'convergence': True, 'job_types': {}, 'paths': paths}}
        return _spc_to_dict(spc, entry, '/abs')

    make_spc = TestSpeciesLevelsAndIrcEndpoints.make_spc

    def test_each_log_keeps_its_own_level_without_the_software(self):
        """Two IRC logs at different levels export both, where ``levels.irc`` is null for the pair"""
        result = self._ts(irc=['/abs/f.log', '/abs/r.log'], irc_levels=[self.FORWARD, self.REVERSE])
        self.assertEqual(result['irc_log_levels'], [{'method': 'wb97xd', 'basis': 'def2-svp'},
                                                    {'method': 'b3lyp', 'basis': '6-31g'}])

    def test_the_list_is_in_lockstep_with_the_logs(self):
        """A level not recorded (a restart) is null and the list is as long as ``irc_logs``"""
        self.assertEqual(self._ts(irc=['/abs/f.log', '/abs/r.log'], irc_levels=[self.FORWARD])['irc_log_levels'],
                         [{'method': 'wb97xd', 'basis': 'def2-svp'}, None])
        self.assertEqual(self._ts(irc=['/abs/f.log'])['irc_log_levels'], [None])
        self.assertEqual(self._ts()['irc_log_levels'], [])
        self.assertEqual(self._ts(irc=['/abs/f.log'], irc_levels=['wb97xd/def2svp'])['irc_log_levels'], [None])

    def test_a_species_record_has_no_irc_log_levels(self):
        """The key is TS-only"""
        result = _spc_to_dict(TestSpeciesLevelsAndIrcEndpoints.make_spc(self), {}, '/abs')
        self.assertNotIn('irc_log_levels', result)

    def test_the_recorder_keeps_the_list_aligned_with_the_irc_paths(self):
        """``set_recorded_irc_log_level`` appends one entry per log, padding entries for logs recorded without one"""
        paths = {'irc': ['f.log']}
        set_recorded_irc_log_level(paths, Level(method='wb97xd', basis='def2svp', software='gaussian',
                                                solvation_scheme_level=Level(method='b3lyp', basis='6-31g')))
        self.assertEqual(len(paths['irc_levels']), 1)
        self.assertEqual(paths['irc_levels'][0]['method'], 'wb97xd')
        self.assertNotIn('solvation_scheme_level', paths['irc_levels'][0])
        paths['irc'].append('r.log')
        set_recorded_irc_log_level(paths, None)
        self.assertEqual([bool(level) for level in paths['irc_levels']], [True, False])
        late = {'irc': ['f.log', 'r.log']}
        set_recorded_irc_log_level(late, Level(method='b3lyp', basis='6-31g'))
        self.assertEqual([bool(level) for level in late['irc_levels']], [False, True])


class TestGsmLevel(unittest.TestCase):
    """Tests for the header ``gsm_level`` and the ``gsm`` program key, both read from archived xtb outputs."""

    BANNER = """      -----------------------------------------------------------
     |                           x T B                           |
      -----------------------------------------------------------

   * xtb version 6.5.1 (b24c23e) compiled by 'conda@728c89f4b128' on 2022-07-12

           -------------------------------------------------
          |                Calculation Setup                |
           -------------------------------------------------

          program call               : xtb orcain1.in.xyz --grad {flags}
          coordinate file            : orcain1.in.xyz
"""
    GFN2 = "          :  Hamiltonian                  GFN2-xTB          :\n"
    GFN1 = "          :  Hamiltonian                  GFN1-xTB          :\n"

    def _run(self, *xtbouts, multiplicity=1, charge=0):
        """A project holding a GSM stringfile with the given xtb output texts archived beside it."""
        scratch = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, scratch, True)
        os.makedirs(os.path.join(scratch, 'gsm', 'gsm_node_outputs'))
        for index, text in enumerate(xtbouts):
            with open(os.path.join(scratch, 'gsm', 'gsm_node_outputs', f'{index}.xtbout'), 'w') as f:
                f.write(text)
        with open(os.path.join(scratch, 'gsm', 'stringfile.xyz0000'), 'w') as f:
            f.write('string\n')
        record = {'gsm_log': 'gsm/stringfile.xyz0000', 'charge': charge, 'multiplicity': multiplicity}
        return scratch, record

    def _out(self, flags='--gfn 2 --chrg 0 --uhf 0', hamiltonian=None):
        return self.BANNER.format(flags=flags) + (hamiltonian or self.GFN2)

    def test_stated_when_every_archived_output_shows_gfn2_and_the_records_charge_and_spin(self):
        """Neutral singlet, anion and doublet records each match the charge and spin xtb was called with, and a
        call without ``--uhf`` is xtb's default of zero unpaired electrons"""
        for flags, charge, multiplicity in (('--gfn 2 --chrg 0 --uhf 0', 0, 1), ('--gfn 2 --chrg -1 --uhf 0', -1, 1),
                                            ('--gfn 2 --chrg 0 --uhf 1', 0, 2), ('--chrg 0', 0, 1)):
            with self.subTest(flags=flags):
                scratch, record = self._run(self._out(flags), self._out(flags), multiplicity=multiplicity,
                                            charge=charge)
                self.assertEqual(_gsm_level([record], scratch), {'method': 'gfn2', 'software': 'xtb'})

    def test_null_for_a_run_started_before_the_method_charge_and_spin_were_staged(self):
        """A pre-fix run (charge 0, no --uhf) is null for a doublet or an anion, whatever ograd is staged locally now"""
        for charge, multiplicity in ((0, 2), (-1, 1)):
            with self.subTest(charge=charge, multiplicity=multiplicity):
                scratch, record = self._run(self._out('--chrg 0'), charge=charge, multiplicity=multiplicity)
                self.assertIsNone(_gsm_level([record], scratch))

    def test_null_when_any_output_differs_or_nothing_was_archived(self):
        """One output with another Hamiltonian, charge or spin, or no archived output at all, states nothing"""
        good = self._out()
        for outputs in ((good, self._out(hamiltonian=self.GFN1)), (good, self._out('--gfn 2 --chrg 1 --uhf 0')),
                        (good, self._out('--gfn 2 --chrg 0 --uhf 1')), ()):
            with self.subTest(outputs=outputs):
                scratch, record = self._run(*outputs)
                self.assertIsNone(_gsm_level([record], scratch))

    def test_null_without_a_gsm_log_or_when_any_gsm_log_fails(self):
        """No GSM log gives null, and two GSM logs both need their outputs to match"""
        scratch, record = self._run(self._out())
        self.assertIsNone(_gsm_level([{'gsm_log': None}, {}], scratch))
        os.makedirs(os.path.join(scratch, 'other'))
        self.assertIsNone(_gsm_level([record, dict(record, gsm_log='other/stringfile.xyz0000')], scratch))

    def test_the_program_and_version_are_the_ones_the_outputs_state(self):
        """``_get_gsm_ess`` names xtb and the banner only when every output agrees"""
        scratch, record = self._run(self._out(), self._out())
        self.assertEqual(_get_gsm_ess(record['gsm_log'], scratch),
                         ('xtb', "xtb version 6.5.1 (b24c23e) compiled by 'conda@728c89f4b128' on 2022-07-12"))
        scratch, record = self._run(self._out(), self._out().replace('6.5.1', '6.6.0'))
        self.assertEqual(_get_gsm_ess(record['gsm_log'], scratch)[0], 'xtb')
        self.assertIsNone(_get_gsm_ess(record['gsm_log'], scratch)[1])
        scratch, record = self._run()
        self.assertEqual(_get_gsm_ess(record['gsm_log'], scratch), (None, None))

    def test_a_converged_ts_record_carries_the_gsm_program_keys(self):
        """The TS record's ess_software and ess_versions gain ``gsm``; without outputs they stay as they were"""
        scratch, record = self._run(self._out())
        spc = TestSpeciesLevelsAndIrcEndpoints.make_spc(self, 'TS0', is_ts=True)
        entry = {'TS0': {'convergence': True, 'job_types': {}, 'paths': {'gsm': os.path.join(scratch, 'gsm', 'stringfile.xyz0000')}}}
        result = _spc_to_dict(spc, entry, scratch)
        self.assertEqual(result['ess_software'], {'gsm': 'xtb'})
        self.assertIn('xtb version 6.5.1', result['ess_versions']['gsm'])
        shutil.rmtree(os.path.join(scratch, 'gsm', 'gsm_node_outputs'))
        self.assertIsNone(_spc_to_dict(spc, entry, scratch)['ess_software'])

    def test_the_writer_exports_the_header_key_as_null_without_a_gsm_run(self):
        """``gsm_level`` is always in the header, null when the run has no GSM log"""
        scratch = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, scratch, True)
        with patch('arc.output._compute_point_groups', return_value={}), \
                patch('arc.output._get_arkane_provenance', return_value=(None, None)), \
                patch('arc.output.get_git_commit', return_value=('', '')):
            write_output_yml(project='no_gsm', project_directory=scratch, species_dict=dict(), reactions=[],
                             output_dict=dict())
        document = read_yaml_file(os.path.join(scratch, 'output', 'output.yml'))
        self.assertIn('gsm_level', document)
        self.assertIsNone(document['gsm_level'])


class TestConformerProvenanceExport(unittest.TestCase):
    """Tests for ``conformer_levels``, ``conformer_energy_kind`` and ``conformer_energy_level`` of a record."""

    CONF_OPT = {'method': 'wb97xd', 'basis': 'def2-svp', 'software': 'gaussian', 'repr': 'wb97xd/def2-svp',
                'compatible_ess': ['gaussian']}
    CONF_SP = {'method': 'dlpno-ccsd(t)', 'basis': 'cc-pvtz', 'software': 'orca'}
    CONF_SP_EXPORTED = {'method': 'dlpno-ccsd(t)', 'basis': 'cc-pvtz'}
    CONF_OPT_EXPORTED = {'method': 'wb97xd', 'basis': 'def2-svp'}
    FF = {'kind': 'force_field_kcal_mol', 'level': None, 'force_field': 'MMFF94s (rdkit)'}

    def make_spc(self, energies, levels=None, sources=None, is_ts=False):
        spc = TestSpcToDict._make_spc_mock(self, is_ts=is_ts)
        xyz = spc.final_xyz
        spc.conformers = [xyz] * len(energies)
        spc.conformer_energies = list(energies)
        if levels is not None:
            spc.conformer_levels = levels
        else:
            del spc.conformer_levels
        if sources is not None:
            spc.conformer_energy_sources = sources
        else:
            del spc.conformer_energy_sources
        return spc

    def export(self, spc):
        result = _spc_to_dict(spc, {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}, '/abs')
        return result['conformer_levels'], result['conformer_energy_kind'], result['conformer_energy_level']

    def export_force_field(self, spc):
        result = _spc_to_dict(spc, {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}, '/abs')
        return result['conformer_force_field']

    def electronic(self, level):
        return {'kind': 'electronic_kj_mol', 'level': level}

    def test_optimized_conformers_carry_their_level_and_the_electronic_kind(self):
        """Test that all-optimized conformers export each level, kJ/mol as the kind and the shared energy level"""
        spc = self.make_spc([-100.0, -98.0], levels=[self.CONF_OPT, self.CONF_OPT],
                            sources=[self.electronic(self.CONF_OPT)] * 2)
        levels, kind, energy_level = self.export(spc)
        self.assertEqual(levels, [self.CONF_OPT_EXPORTED, self.CONF_OPT_EXPORTED])
        self.assertEqual(kind, 'electronic_kj_mol')
        self.assertEqual(energy_level, self.CONF_OPT_EXPORTED)

    def test_force_field_conformers_have_null_levels_and_no_energy_level(self):
        """Test that never-optimized conformers export null levels, the force-field kind and a null energy level"""
        spc = self.make_spc([1.2, 0.0], levels=[None, None], sources=[self.FF, self.FF])
        self.assertEqual(self.export(spc), ([None, None], 'force_field_kcal_mol', None))

    def test_force_field_energies_never_state_a_level(self):
        """Test that a level attached to a force-field energy source is not exported as the energy level"""
        spc = self.make_spc([1.2], levels=[None], sources=[{'kind': 'force_field_kcal_mol', 'level': self.CONF_OPT}])
        self.assertEqual(self.export(spc), ([None], 'force_field_kcal_mol', None))

    def test_a_conformer_single_point_sets_the_energy_level_but_not_the_geometry_level(self):
        """Test that conf_sp energies name the sp level while conformer_levels keep the conf_opt level"""
        spc = self.make_spc([-100.0, -98.0], levels=[self.CONF_OPT, self.CONF_OPT],
                            sources=[self.electronic(self.CONF_SP)] * 2)
        levels, kind, energy_level = self.export(spc)
        self.assertEqual(levels, [self.CONF_OPT_EXPORTED] * 2)
        self.assertEqual(kind, 'electronic_kj_mol')
        self.assertEqual(energy_level, self.CONF_SP_EXPORTED)

    def test_a_failed_conformer_optimization_makes_the_kind_null(self):
        """Test that some optimized conformers next to one keeping its force-field energy give a null kind and level"""
        spc = self.make_spc([-100.0, 1.2], levels=[self.CONF_OPT, None],
                            sources=[self.electronic(self.CONF_OPT), self.FF])
        levels, kind, energy_level = self.export(spc)
        self.assertEqual(levels, [self.CONF_OPT_EXPORTED, None])
        self.assertIsNone(kind)
        self.assertIsNone(energy_level)

    def test_energies_from_different_levels_make_the_energy_level_null(self):
        """Test that only some conformers getting a single point keeps the kind but drops the energy level"""
        spc = self.make_spc([-100.0, -98.0], levels=[self.CONF_OPT] * 2,
                            sources=[self.electronic(self.CONF_SP), self.electronic(self.CONF_OPT)])
        levels, kind, energy_level = self.export(spc)
        self.assertEqual(kind, 'electronic_kj_mol')
        self.assertIsNone(energy_level)

    def test_an_entry_without_an_energy_does_not_affect_the_kind(self):
        """Test that a conformer whose energy is null is ignored when stating the kind"""
        spc = self.make_spc([None, -98.0], levels=[None, self.CONF_OPT],
                            sources=[None, self.electronic(self.CONF_OPT)])
        self.assertEqual(self.export(spc)[1:], ('electronic_kj_mol', self.CONF_OPT_EXPORTED))

    def test_unrecorded_provenance_exports_nulls_of_the_right_length(self):
        """Test that a species restored without the provenance keys gets a null list of the right length and no kind"""
        spc = self.make_spc([-100.0, -98.0, None])
        self.assertEqual(self.export(spc), ([None, None, None], None, None))

    def test_a_short_or_long_provenance_list_is_fitted_to_the_conformers(self):
        """Test that stale lists of the wrong length never produce a mis-sized conformer_levels"""
        short = self.make_spc([-100.0, -98.0, -97.0], levels=[self.CONF_OPT], sources=[])
        long = self.make_spc([-100.0], levels=[self.CONF_OPT, self.CONF_OPT, self.CONF_OPT], sources=[None] * 4)
        self.assertEqual(self.export(short)[0], [self.CONF_OPT_EXPORTED, None, None])
        self.assertEqual(self.export(long)[0], [self.CONF_OPT_EXPORTED])

    def test_a_known_force_field_is_named_and_states_the_kcal_kind(self):
        """Test that energies of one force field and backend export that name and the kcal/mol kind"""
        spc = self.make_spc([1.2, 0.0], levels=[None, None], sources=[self.FF, self.FF])
        self.assertEqual(self.export_force_field(spc), 'MMFF94s (rdkit)')

    def test_a_force_field_whose_unit_is_not_kcal_is_named_without_a_kind(self):
        """Test that a kJ/mol force field gives a null kind but still names the force field"""
        source = {'kind': None, 'level': None, 'force_field': 'UFF (openbabel)'}
        spc = self.make_spc([1.2], levels=[None], sources=[source])
        self.assertEqual(self.export(spc), ([None], None, None))
        self.assertEqual(self.export_force_field(spc), 'UFF (openbabel)')

    def test_placeholder_energies_have_no_kind_and_no_force_field(self):
        """Test that an energy no force field computed (cheat sheet, monoatomic, diatomic) states no origin"""
        spc = self.make_spc([0.0], levels=[None], sources=[None])
        self.assertEqual(self.export(spc), ([None], None, None))
        self.assertIsNone(self.export_force_field(spc))

    def test_a_hand_edited_kind_or_force_field_that_is_not_a_string_is_unknown(self):
        """Test that an unhashable or non-string kind or force field exports null instead of raising"""
        for value in (['force_field_kcal_mol'], {'kind': 1}, 7):
            with self.subTest(value=value):
                kind = self.make_spc([1.2], levels=[None], sources=[{'kind': value, 'level': None,
                                                                     'force_field': 'MMFF94s (rdkit)'}])
                self.assertEqual(self.export(kind), ([None], None, None))
                self.assertEqual(self.export_force_field(kind), 'MMFF94s (rdkit)')
                force_field = self.make_spc([1.2], levels=[None], sources=[dict(self.FF, force_field=value)])
                self.assertEqual(self.export(force_field), ([None], 'force_field_kcal_mol', None))
                self.assertIsNone(self.export_force_field(force_field))

    def test_different_force_fields_or_electronic_energies_have_no_single_force_field(self):
        """Test that a mixture of force fields, or of a force field and an ESS energy, names none"""
        other = dict(self.FF, force_field='UFF (rdkit)')
        mixed = self.make_spc([1.2, 0.3], levels=[None, None], sources=[self.FF, other])
        self.assertIsNone(self.export_force_field(mixed))
        electronic = self.make_spc([1.2, -100.0], levels=[None, self.CONF_OPT],
                                   sources=[self.FF, self.electronic(self.CONF_OPT)])
        self.assertIsNone(self.export_force_field(electronic))
        optimized = self.make_spc([-100.0], levels=[self.CONF_OPT], sources=[self.electronic(self.CONF_OPT)])
        self.assertIsNone(self.export_force_field(optimized))

    def test_no_conformers_exports_all_null(self):
        """Test that a record with no conformers, including a TS, exports null for the three keys and no conformers"""
        for is_ts in (False, True):
            with self.subTest(is_ts=is_ts):
                spc = self.make_spc([], levels=[], sources=[], is_ts=is_ts)
                result = _spc_to_dict(spc, {'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}}, '/abs')
                self.assertNotIn('conformers', result)
                self.assertEqual((result['conformer_levels'], result['conformer_energy_kind'],
                                  result['conformer_energy_level'], result['conformer_force_field']),
                                 (None, None, None, None))

    def test_the_writer_puts_the_requested_levels_in_the_header(self):
        """Test that write_output_yml writes the five header levels, and null for each one not given"""
        scratch = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, scratch, True)
        levels = {'scan_level': Level(method='b3lyp', basis='6-31g'),
                  'irc_level': Level(method='wb97xd', basis='def2-svp'),
                  'conformer_opt_level': Level(method='wb97xd', basis='def2-svp'),
                  'conformer_sp_level': Level(method='dlpno-ccsd(t)', basis='cc-pvtz'),
                  'ts_guess_level': Level(method='pm7')}
        documents = dict()
        for name, kwargs in (('given', levels), ('absent', dict())):
            with patch('arc.output._compute_point_groups', return_value={}), \
                    patch('arc.output._get_arkane_provenance', return_value=('3.3.0', 'abc123')), \
                    patch('arc.output.get_git_commit', return_value=('def456', '2026-01-01')):
                write_output_yml(project=name, project_directory=scratch, species_dict=dict(), reactions=[],
                                 output_dict=dict(), **kwargs)
            documents[name] = read_yaml_file(os.path.join(scratch, 'output', 'output.yml'))
        for key, level in levels.items():
            with self.subTest(key=key):
                self.assertEqual(documents['given'][key]['method'], level.method)
                self.assertIn(key, documents['absent'])
                self.assertIsNone(documents['absent'][key])


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
        guess.index = 0
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
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        os.makedirs(os.path.join(self.tmp_dir, 'output'), exist_ok=True)

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
        spc.e0_atom_corrections_applied = None
        spc.e0_bond_corrections_applied = None
        spc.arkane_rotor_modes = None
        spc._is_linear = False
        spc.optical_isomers = 1
        spc.external_symmetry = 12
        spc.freqs = [1300.0, 1500.0, 3000.0]
        spc.rotors_dict = None
        spc.thermo = ThermoData(H298=-74.6, S298=186.3, Tmin=(300, 'K'), Tmax=(3000, 'K'))
        spc.number_of_radicals = None
        spc.derived_stability_verdict = None
        spc.scf_references = dict()
        return spc

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_provenance', return_value=('3.3.0', 'abc123'))
    @patch('arc.output.get_git_commit', return_value=('def456', '2026-01-01'))
    def test_writes_file_atomically(self, mock_arc_git, mock_arkane_provenance, mock_pg):
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
        self.assertEqual(doc['schema_version'], '1.3')
        self.assertEqual(doc['parser_evidence']['path'], 'parser_evidence.json')
        evidence_path = os.path.join(self.tmp_dir, 'output', 'parser_evidence.json')
        self.assertTrue(os.path.isfile(evidence_path))
        with open(evidence_path) as handle:
            evidence = json.load(handle)
        self.assertEqual(evidence['document_id'], doc['parser_evidence']['document_id'])
        self.assertEqual(doc['project'], 'test_project')
        self.assertEqual(doc['arc_git_commit'], 'def456')
        self.assertEqual(doc['arkane_git_commit'], 'abc123')
        self.assertIsInstance(doc['species'], list)
        self.assertEqual(len(doc['species']), 1)
        self.assertEqual(doc['species'][0]['label'], 'CH4')
        self.assertEqual(doc['reactions'], [])
        self.assertEqual(doc['transition_states'], [])

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_thermo_carries_the_arkane_correction_flags(self, mock_arc_git, mock_arkane_provenance, mock_pg):
        """output.yml states per species whether its thermo carries atom and bond corrections."""
        from arc.common import read_yaml_file
        uncorrected, corrected, unknown = (self._make_spc_mock(label) for label in ('CH4', 'C2H6', 'C3H8'))
        uncorrected.thermo.atom_corrections_applied = False
        uncorrected.thermo.bond_corrections_applied = False
        corrected.thermo.atom_corrections_applied = True
        corrected.thermo.bond_corrections_applied = True
        corrected.thermo.atom_corrections_level = Level(method='bmk', basis='cbsb7', software='gaussian')
        species_dict = {spc.label: spc for spc in (uncorrected, corrected, unknown)}
        output_dict = {label: {'convergence': True, 'paths': {}, 'job_types': {}} for label in species_dict}

        write_output_yml(
            project='test_correction_flags',
            project_directory=self.tmp_dir,
            species_dict=species_dict,
            reactions=[],
            output_dict=output_dict,
            sp_level=Level(method='apfd', basis='def2svp', software='gaussian'),
        )

        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        thermo = {entry['label']: entry['thermo'] for entry in doc['species']}
        self.assertIs(thermo['CH4']['atom_corrections_applied'], False)
        self.assertIs(thermo['CH4']['bond_corrections_applied'], False)
        self.assertIs(thermo['C2H6']['atom_corrections_applied'], True)
        self.assertIs(thermo['C2H6']['bond_corrections_applied'], True)
        self.assertIsNone(thermo['C3H8']['atom_corrections_applied'])
        self.assertIsNone(thermo['C3H8']['bond_corrections_applied'])
        self.assertIsNone(thermo['CH4']['atom_corrections_level'])
        self.assertIsNone(thermo['C3H8']['atom_corrections_level'])
        self.assertEqual(thermo['C2H6']['atom_corrections_level'],
                         _level_to_dict(Level(method='bmk', basis='cbsb7', software='gaussian')))
        self.assertNotEqual(thermo['C2H6']['atom_corrections_level'], doc['sp_level'])
        self.assertEqual(doc['sp_level']['method'], 'apfd')

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_ess_software_is_emitted_and_pairs_with_ess_versions(self, mock_arc_git, mock_arkane_provenance, mock_pg):
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
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_ess_software_is_null_for_a_non_converged_species(self, mock_arc_git, mock_arkane_provenance, mock_pg):
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
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_levels_of_theory(self, mock_arc_git, mock_arkane_provenance, mock_pg):
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
        self.assertIsNone(doc['freq_scale_factor_key'])
        self.assertIsNone(doc['freq_scale_factor_source'])  # user-provided
        self.assertEqual(doc['bac_type'], 'p')

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_a_rates_only_run_reports_the_requested_bac_but_applies_none(
            self, mock_arc_git, mock_arkane_provenance, mock_pg):
        """A run that computes rates without thermo never reaches a BAC-applying
        Arkane run, so no species carries a ``bond_additivity`` record even though
        the run requested ``bac_type: p``. The atom-energy record is unaffected --
        Arkane applies that in the kinetics run too."""
        from arc.common import read_yaml_file
        corrections = {'CH4': {'aec': {'value': -0.0234, 'value_unit': 'hartree', 'components': []},
                               'bac': {'value': -0.694, 'value_unit': 'kcal_mol',
                                       'bac_type': 'p', 'components': []}}}
        documents = {}
        for compute_thermo in (True, False):
            with patch('arc.output._compute_species_corrections', return_value=corrections):
                write_output_yml(
                    project='p',
                    project_directory=self.tmp_dir,
                    species_dict={'CH4': self._make_spc_mock()},
                    reactions=[],
                    output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
                    opt_level=Level(method='wb97xd', basis='def2tzvp', software='gaussian'),
                    bac_type='p',
                    compute_thermo=compute_thermo,
                    arkane_level_of_theory=Level(method='wb97xd', basis='def2tzvp',
                                                 software='gaussian'),
                )
            documents[compute_thermo] = read_yaml_file(
                os.path.join(self.tmp_dir, 'output', 'output.yml'))

        def correction_types(document):
            return [entry['correction_type']
                    for entry in document['species'][0]['energy_corrections']]

        self.assertEqual(documents[True]['bac_type'], 'p')
        self.assertEqual(documents[False]['bac_type'], 'p')
        self.assertIn('bond_additivity', correction_types(documents[True]))
        self.assertNotIn('bond_additivity', correction_types(documents[False]))
        self.assertIn('atom_energy', correction_types(documents[False]))

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_arkane_level_of_theory(self, mock_arc_git, mock_arkane_provenance, mock_pg):
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
    @patch('arc.output._get_arkane_provenance', return_value=('3.2.0', None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_arkane_version_identifies_the_software_without_a_git_commit(
            self, mock_arc_git, mock_arkane_provenance, mock_pg):
        """The corrections stay attributable on an install that has no RMG-Py git repo."""
        write_output_yml(
            project='test_arkane_version',
            project_directory=self.tmp_dir,
            species_dict={'CH4': self._make_spc_mock()},
            reactions=[],
            output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
            arkane_level_of_theory=Level(method='cbs-qb3', software='gaussian'),
            bac_type='p',
        )

        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        self.assertEqual(doc['arkane_version'], '3.2.0')
        self.assertIsNone(doc['arkane_git_commit'])
        self.assertEqual(mock_arkane_provenance.call_args[0][0], self.tmp_dir)

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_provenance', return_value=(None, 'abc123'))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_arkane_version_is_null_not_absent_when_unresolvable(
            self, mock_arc_git, mock_arkane_provenance, mock_pg):
        """An unresolvable version is emitted as null, the convention every optional key follows."""
        write_output_yml(
            project='test_arkane_version_null',
            project_directory=self.tmp_dir,
            species_dict={'CH4': self._make_spc_mock()},
            reactions=[],
            output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
        )

        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        self.assertIn('arkane_version', doc)
        self.assertIsNone(doc['arkane_version'])
        self.assertEqual(doc['arkane_git_commit'], 'abc123')

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_provenance', return_value=('3.3.0', 'abc123'))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_the_arkane_identity_is_emitted_without_any_corrections(
            self, mock_arc_git, mock_arkane_provenance, mock_pg):
        """The pair names the Arkane ARC invoked; it is not gated on corrections reaching the document."""
        write_output_yml(
            project='test_arkane_identity_without_corrections',
            project_directory=self.tmp_dir,
            species_dict={'CH4': self._make_spc_mock()},
            reactions=[],
            output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
        )

        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        self.assertIsNone(doc['atom_energy_corrections'])
        self.assertIsNone(doc['bond_additivity_corrections'])
        self.assertEqual(doc['arkane_version'], '3.3.0')
        self.assertEqual(doc['arkane_git_commit'], 'abc123')

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_arkane_level_of_theory_carries_its_year(
            self, mock_arc_git, mock_arkane_provenance, mock_pg):
        """The level's ``year`` reaches the emitted ``arkane_level_of_theory`` dict."""
        write_output_yml(
            project='test_arkane_year',
            project_directory=self.tmp_dir,
            species_dict={'CH4': self._make_spc_mock()},
            reactions=[],
            output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
            arkane_level_of_theory=Level(method='wb97xd', basis='def2tzvp',
                                         software='gaussian', year=2023),
        )

        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        self.assertEqual(doc['arkane_level_of_theory']['year'], 2023)

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_arkane_level_none_when_not_set(self, mock_arc_git, mock_arkane_provenance, mock_pg):
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
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_energy_correction_records_carry_the_matched_arkane_key(self, mock_arc_git, mock_arkane_provenance, mock_pg):
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
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_energy_correction_records_follow_the_recorded_arkane_switches(self, mock_arc_git,
                                                                          mock_arkane_provenance, mock_pg):
        """A correction kind the thermo run is recorded as not applying is dropped; an unrecorded one is kept."""
        aec_key = "LevelOfTheory(method='wb97xd',basis='def2tzvp',software='gaussian')"
        corrections = EnergyCorrections(aec={'C': -37.8, 'H': -0.5}, bac={'C-H': -0.17},
                                        aec_key=aec_key, bac_key=aec_key)
        species_corrections = {'CH4': {
            'aec': {'value': -0.02, 'value_unit': 'hartree', 'components': []},
            'bac': {'value': -0.7, 'value_unit': 'kcal_mol', 'components': []},
        }}
        cases = {(False, False): [],
                 (True, False): ['atom_energy'],
                 (True, True): ['atom_energy', 'bond_additivity'],
                 (None, None): ['atom_energy', 'bond_additivity']}
        for (aec, bac), expected in cases.items():
            with self.subTest(atom_corrections_applied=aec, bond_corrections_applied=bac):
                spc = self._make_spc_mock()
                if aec is not None:
                    spc.thermo.atom_corrections_applied = aec
                    spc.thermo.bond_corrections_applied = bac
                with patch('arc.output._get_energy_corrections', return_value=corrections), \
                        patch('arc.output._compute_species_corrections', return_value=species_corrections):
                    write_output_yml(
                        project='recorded_switches',
                        project_directory=self.tmp_dir,
                        species_dict={'CH4': spc},
                        reactions=[],
                        output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
                        arkane_level_of_theory=Level(method='wb97xd', basis='def2tzvp', software='gaussian'),
                        bac_type='p',
                    )
                doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
                self.assertEqual(sorted(r['correction_type'] for r in doc['species'][0]['energy_corrections']),
                                 expected)

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_no_correction_records_when_an_unmatched_frequency_level_disabled_aec(self, mock_arc_git,
                                                                                mock_arkane_provenance, mock_pg):
        """The reviewed contradiction, end to end: sp wb97xd/def2tzvp matches an Arkane AEC key, but with no
        frequency scale factor ARC asks Arkane for a composite model chemistry, the frequency level has no entry,
        and Arkane runs with ``useAtomCorrections = False``. output.yml must not then claim AEC/BAC records."""
        from arc.statmech.arkane import ArkaneAdapter
        from arc.common import save_yaml_file
        sp_level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        computed = ARCSpecies(label='CH4', smiles='C')
        arkane = ArkaneAdapter(output_directory=os.path.join(self.tmp_dir, 'output'),
                               calcs_directory=os.path.join(self.tmp_dir, 'calcs'),
                               output_dict=dict(), bac_type='p', species=[computed], sp_level=sp_level,
                               freq_level=Level(method='b3lyp', basis='sto-3g', software='gaussian'),
                               freq_scale_factor=None)
        inputs = list()

        def fake_run_arkane(statmech_dir):
            with open(os.path.join(statmech_dir, 'input.py'), 'r') as f:
                inputs.append(f.read())
            with open(os.path.join(statmech_dir, 'output.py'), 'w') as f:
                f.write('')
            save_yaml_file(path=os.path.join(statmech_dir, 'thermo.yaml'),
                           content={'CH4': {'H298': -105000.0, 'S298': 186.3, 'data': 'NASA()'}})
            return True

        with patch('arc.statmech.arkane.run_arkane', side_effect=fake_run_arkane), \
                patch.object(ArkaneAdapter, 'generate_species_files'), \
                patch('arc.statmech.arkane.execute_command', return_value=('', '')):
            arkane.compute_thermo()
        self.assertIn('useAtomCorrections = False\n', inputs[0])

        spc = self._make_spc_mock()
        spc.thermo = computed.thermo
        aec_key = "LevelOfTheory(method='wb97xd2023',basis='def2tzvp',software='gaussian')"
        corrections = EnergyCorrections(aec={'C': -37.8, 'H': -0.5}, bac={'C-H': -0.17},
                                        aec_key=aec_key, bac_key=aec_key)
        species_corrections = {'CH4': {
            'aec': {'value': -0.02, 'value_unit': 'hartree', 'components': []},
            'bac': {'value': -0.7, 'value_unit': 'kcal_mol', 'components': []},
        }}
        with patch('arc.output._get_energy_corrections', return_value=corrections), \
                patch('arc.output._compute_species_corrections', return_value=species_corrections):
            write_output_yml(
                project='unmatched_freq_level',
                project_directory=self.tmp_dir,
                species_dict={'CH4': spc},
                reactions=[],
                output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
                sp_level=sp_level,
                arkane_level_of_theory=sp_level,
                bac_type='p',
            )
        entry = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))['species'][0]
        self.assertIs(entry['thermo']['atom_corrections_applied'], False)
        self.assertIs(entry['thermo']['bond_corrections_applied'], False)
        self.assertIsNone(entry['thermo']['atom_corrections_level'])
        self.assertEqual(entry['energy_corrections'], [])

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_no_corrections_for_a_solvated_level(self, mock_arc_git, mock_arkane_provenance, mock_pg):
        """End to end: Arkane has no corrections for an SMD-solvated wb97xd/def2tzvp, so ARC runs it with
        ``useAtomCorrections = False`` and output.yml records no corrections, although gas-phase wb97xd/def2tzvp
        matches an Arkane key."""
        from arc.statmech.arkane import ArkaneAdapter
        from arc.common import save_yaml_file
        sp_level = Level(method='wb97xd', basis='def2tzvp', software='gaussian',
                         solvation_method='smd', solvent='water')
        computed = ARCSpecies(label='CH4', smiles='C')
        arkane = ArkaneAdapter(output_directory=os.path.join(self.tmp_dir, 'output'),
                               calcs_directory=os.path.join(self.tmp_dir, 'calcs'),
                               output_dict=dict(), bac_type='p', species=[computed], sp_level=sp_level,
                               freq_level=sp_level, freq_scale_factor=1.0)
        inputs = list()

        def fake_run_arkane(statmech_dir):
            with open(os.path.join(statmech_dir, 'input.py'), 'r') as f:
                inputs.append(f.read())
            with open(os.path.join(statmech_dir, 'output.py'), 'w') as f:
                f.write('')
            save_yaml_file(path=os.path.join(statmech_dir, 'thermo.yaml'),
                           content={'CH4': {'H298': -105000.0, 'S298': 186.3, 'data': 'NASA()'}})
            return True

        with patch('arc.statmech.arkane.run_arkane', side_effect=fake_run_arkane), \
                patch.object(ArkaneAdapter, 'generate_species_files'), \
                patch('arc.statmech.arkane.execute_command', return_value=('', '')):
            arkane.compute_thermo()
        self.assertIn('useAtomCorrections = False\n', inputs[0])
        self.assertIn('useBondCorrections = False\n', inputs[0])

        spc = self._make_spc_mock()
        spc.thermo = computed.thermo
        with patch('arc.output._compute_species_corrections', return_value={}):
            write_output_yml(
                project='solvated_level',
                project_directory=self.tmp_dir,
                species_dict={'CH4': spc},
                reactions=[],
                output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
                sp_level=sp_level,
                arkane_level_of_theory=sp_level,
                bac_type='p',
            )
        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        entry = doc['species'][0]
        self.assertIs(entry['thermo']['atom_corrections_applied'], False)
        self.assertIs(entry['thermo']['bond_corrections_applied'], False)
        self.assertIsNone(entry['thermo']['atom_corrections_level'])
        self.assertEqual(entry['energy_corrections'], [])
        self.assertIsNone(doc['atom_energy_corrections'])
        self.assertIsNone(doc['bond_additivity_corrections'])

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_each_correction_record_names_its_own_arkane_block(self, mock_arc_git, mock_arkane_provenance, mock_pg):
        """AEC and BAC blocks resolved under different keys each name the key they came from."""
        aec_key = "LevelOfTheory(method='wb97mv',basis='def2tzvpd',software='qchem')"
        bac_key = "LevelOfTheory(method='wb97mv2023',basis='def2tzvpd',software='qchem')"
        corrections = EnergyCorrections(aec={'C': -37.8, 'H': -0.5}, bac={'C-H': -0.17},
                                        aec_key=aec_key, bac_key=bac_key)
        species_corrections = {'CH4': {
            'aec': {'value': -0.02, 'value_unit': 'hartree', 'components': []},
            'bac': {'value': -0.7, 'value_unit': 'kcal_mol', 'components': []},
        }}
        with patch('arc.output._get_energy_corrections', return_value=corrections), \
             patch('arc.output._compute_species_corrections', return_value=species_corrections):
            write_output_yml(
                project='per_section_key_wiring',
                project_directory=self.tmp_dir,
                species_dict={'CH4': self._make_spc_mock()},
                reactions=[],
                output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
                sp_level=Level(method='wb97m-v', basis='def2tzvpd', software='qchem'),
                arkane_level_of_theory=Level(method='wb97m-v', basis='def2tzvpd', software='qchem'),
                bac_type='p',
            )

        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        records = doc['species'][0]['energy_corrections']
        keys = {r['correction_type']: r['matched_arkane_key'] for r in records}
        self.assertEqual(keys, {'atom_energy': aec_key, 'bond_additivity': bac_key})
        bac = next(r for r in records if r['correction_type'] == 'bond_additivity')
        self.assertNotIn('parameter_table', bac)

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_freq_scale_factor_key_names_the_block_the_factor_came_from(self, mock_arc_git, mock_arkane_provenance, mock_pg):
        """A level that is in ARC's table publishes that table's entry key and its citation."""
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        write_output_yml(
            project='freq_scale_key',
            project_directory=self.tmp_dir,
            species_dict={'CH4': self._make_spc_mock()},
            reactions=[],
            output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
            opt_level=level,
            freq_level=level,
            freq_scale_factor=0.988,
        )

        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        self.assertEqual(doc['freq_scale_factor_key'], 'wb97xd/def2tzvp, software: gaussian')
        self.assertIn('10.1021/ct100326h', doc['freq_scale_factor_source'])

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_a_level_absent_from_the_table_publishes_a_null_key(self, mock_arc_git, mock_arkane_provenance, mock_pg):
        """An unlisted level leaves both the key and the citation null rather than omitting them."""
        level = Level(method='totally_fake_method', basis='fake_basis')
        write_output_yml(
            project='freq_scale_key_absent',
            project_directory=self.tmp_dir,
            species_dict={'CH4': self._make_spc_mock()},
            reactions=[],
            output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
            opt_level=level,
            freq_level=level,
            freq_scale_factor=0.99,
        )

        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        self.assertIn('freq_scale_factor_key', doc)
        self.assertIsNone(doc['freq_scale_factor_key'])
        self.assertIsNone(doc['freq_scale_factor_source'])

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_no_tmp_files_left(self, mock_arc_git, mock_arkane_provenance, mock_pg):
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
    @patch('arc.output.build_parser_evidence', side_effect=RuntimeError('evidence failed'))
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
        self.assertEqual(doc['schema_version'], '1.3')
        self.assertNotIn('parser_evidence', doc)

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output.build_parser_evidence', side_effect=RuntimeError('evidence failed'))
    def test_evidence_failure_removes_the_previous_runs_sidecar(self, mock_build, mock_pg):
        """A run that cannot produce evidence must not leave the last run's beside its output.

        The sidecar is only interpretable against the ``output.yml`` naming its
        ``document_id``. A surviving sidecar from an earlier run sits beside a
        document with no descriptor at all, so nothing marks it stale and a
        consumer reads last week's Hessians as this run's.
        """
        out_dir = os.path.join(self.tmp_dir, 'output')
        os.makedirs(out_dir, exist_ok=True)
        stale_path = os.path.join(out_dir, 'parser_evidence.json')
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
        self.assertNotIn('parser_evidence', doc)
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
        self.assertEqual(destinations[-2:], ['parser_evidence.json', 'output.yml'])

    @patch('arc.output._compute_point_groups', return_value={})
    @patch('arc.output._get_arkane_provenance', return_value=(None, None))
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_cost_metrics_section(self, mock_arc_git, mock_arkane_provenance, mock_pg):
        """Cost metrics should aggregate per-ESS execution time and core-hours."""
        from arc.common import read_yaml_file
        spc = self._make_spc_mock()
        records = [
            {'job_name': 'opt_a1', 'label': 'CH4', 'job_type': 'opt', 'job_adapter': 'gaussian',
             'server': 'server1', 'cpu_cores': 8, 'run_time_sec': 3600.0, 'job_status': 'done'},
            {'job_name': 'freq_a2', 'label': 'CH4', 'job_type': 'freq', 'job_adapter': 'gaussian',
             'server': 'server1', 'cpu_cores': 8, 'run_time_sec': None, 'job_status': 'done'},
            {'job_name': 'sp_a3', 'label': 'CH4', 'job_type': 'sp', 'job_adapter': 'orca',
             'server': 'server2', 'cpu_cores': 16, 'run_time_sec': 900.0, 'job_status': 'done'},
        ]
        write_output_yml(
            project='test_cost',
            project_directory=self.tmp_dir,
            species_dict={'CH4': spc},
            reactions=[],
            output_dict={'CH4': {'convergence': True, 'paths': {}, 'job_types': {}}},
            t0=datetime.datetime.now().timestamp() - 7200.0,  # started 2 hrs ago
            completed_job_records=records,
        )
        doc = read_yaml_file(os.path.join(self.tmp_dir, 'output', 'output.yml'))
        cm = doc['cost_metrics']
        self.assertAlmostEqual(cm['wall_time_hrs'], 2.0, places=2)
        self.assertEqual(cm['total_job_count'], 3)
        self.assertEqual(cm['jobs_missing_time'], 1)
        self.assertEqual(cm['jobs_missing_cores'], 0)
        self.assertAlmostEqual(cm['total_execution_time_hrs'], 1.25)
        self.assertAlmostEqual(cm['total_core_hours'], 12.0)
        self.assertEqual(cm['per_ess']['gaussian']['job_count'], 2)
        self.assertEqual(cm['per_ess']['gaussian']['jobs_missing_time'], 1)
        self.assertAlmostEqual(cm['per_ess']['gaussian']['execution_time_hrs'], 1.0)
        self.assertAlmostEqual(cm['per_ess']['gaussian']['core_hours'], 8.0)
        self.assertEqual(cm['per_ess']['orca']['job_count'], 1)
        self.assertAlmostEqual(cm['per_ess']['orca']['execution_time_hrs'], 0.25)
        self.assertAlmostEqual(cm['per_ess']['orca']['core_hours'], 4.0)


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


class TestParseWavefunctionStabilityForOutput(unittest.TestCase):
    """
    Contains unit tests for the wavefunction stability entry written to output.yml.
    """

    def _write_log(self, body: str) -> str:
        """Write a Gaussian stability log to a temporary file and return its path."""
        with tempfile.NamedTemporaryFile(suffix='.log', mode='w', delete=False) as f:
            f.write(' Entering Gaussian System, Link 0=g16\n' + body)
            return f.name

    def test_freq_validity_reaches_the_output_record(self):
        """Test that output.yml carries the reference and the frequency-validity verdict"""
        body = ' SCF Done:  E(UwB97XD) =  -78.5936     A.U.\n' \
               ' Stability analysis using <AA,BB:AA,BB> singles matrix:\n' \
               ' Eigenvector   1:      Triplet-?Sym  Eigenvalue=-0.1434007  <S**2>=2.000\n' \
               ' The wavefunction has an RHF -> UHF instability.\n'
        path = self._write_log(body)
        try:
            result = _parse_wavefunction_stability(path, os.path.dirname(path))
            self.assertEqual(result['verdict'], 'external_instability')
            self.assertIn('restricted', result)
            self.assertIn('invalidates_analytic_freq', result)
            self.assertFalse(result['restricted'])
            self.assertTrue(result['invalidates_analytic_freq'])
            self.assertIsNotNone(result['log'])
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_restricted_external_instability_reaches_output_as_valid(self):
        """Test that a restricted external instability is recorded as not invalidating the freq"""
        body = ' SCF Done:  E(RwB97XD) =  -78.5936     A.U.\n' \
               ' Stability analysis using <AA,BB:AA,BB> singles matrix:\n' \
               ' The wavefunction has an RHF -> UHF instability.\n'
        path = self._write_log(body)
        try:
            result = _parse_wavefunction_stability(path, os.path.dirname(path))
            self.assertTrue(result['restricted'])
            self.assertFalse(result['invalidates_analytic_freq'])
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_no_stability_log_yields_nothing(self):
        """Test that a species with no stability analysis records no entry"""
        self.assertIsNone(_parse_wavefunction_stability(None, '/tmp'))
        self.assertIsNone(_parse_wavefunction_stability('/nonexistent/stability.log', '/tmp'))

class TestBacIsAppliedTo(unittest.TestCase):
    """Which species ARC actually routes through a BAC-applying Arkane run.

    ``arc/processor.py`` invokes the kinetics statmech adapter with
    ``bac_type=None`` and skips transition states when collecting species for
    the thermo adapter, so a rates-only project applies no bond additivity
    correction to anything and a TS never receives one from any run. Reporting
    one under ``energy_corrections`` would name a correction Arkane never made.
    """

    @staticmethod
    def _spc(is_ts=False, compute_thermo=True):
        spc = MagicMock()
        spc.is_ts = is_ts
        spc.compute_thermo = compute_thermo
        return spc

    def test_an_ordinary_species_on_a_thermo_run_gets_a_bac(self):
        self.assertTrue(_bac_is_applied_to(self._spc(), compute_thermo=True))

    def test_a_rates_only_run_applies_no_bac_at_all(self):
        self.assertFalse(_bac_is_applied_to(self._spc(), compute_thermo=False))

    def test_a_transition_state_never_gets_a_bac(self):
        self.assertFalse(_bac_is_applied_to(self._spc(is_ts=True), compute_thermo=True))

    def test_a_species_excluded_from_thermo_gets_no_bac(self):
        self.assertFalse(_bac_is_applied_to(self._spc(compute_thermo=False), compute_thermo=True))


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

    def test_pbac_keeps_the_applied_components_and_lists_the_skipped_bonds(self):
        """Arkane applies the bonds it has a parameter for and skips the rest, so the components sum to the total."""
        block = self._pbac_block()
        block['value'] = -0.694
        block['components'].append({'component_kind': 'bond', 'key': 'C#N', 'multiplicity': 2,
                                    'parameter_value': None, 'parameter_unit': 'kcal_mol',
                                    'contribution_value': None})
        sc = {'X': {'aec': self._aec_block(), 'bac': block}}
        out = _build_energy_corrections_for_species('X', sc, self._lot(), 'p')
        bac = next(e for e in out if e['correction_type'] == 'bond_additivity')
        self.assertEqual([component['key'] for component in bac['components']], ['C-H'])
        self.assertAlmostEqual(sum(component['contribution_value'] for component in bac['components']),
                               bac['total']['value'])
        self.assertEqual(bac['skipped_components'], [{'bond': 'C#N', 'count': 2}])

    def test_skipped_components_apply_to_a_petersson_record_only(self):
        """Empty when every Petersson bond has a parameter, and null for an atom-energy or a Melius record."""
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_energy_corrections_for_species('X', sc, self._lot(), 'p')
        self.assertEqual({record['correction_type']: record['skipped_components'] for record in out},
                         {'atom_energy': None, 'bond_additivity': []})
        sc = {'X': {'aec': self._aec_block(), 'bac': self._mbac_block()}}
        out = _build_energy_corrections_for_species('X', sc, self._lot(), 'm')
        self.assertEqual([record['skipped_components'] for record in out], [None, None])

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

    def test_each_record_names_the_key_matched_in_its_own_section(self):
        """The two sections are matched independently, so the two records may differ."""
        aec_key = "LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='qchem')"
        bac_key = "LevelOfTheory(method='wb97xd3',basis='def2tzvp')"
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_energy_corrections_for_species(
            'X', sc, self._lot(), 'p', aec_key=aec_key, bac_key=bac_key,
        )
        self.assertEqual([e['matched_arkane_key'] for e in out], [aec_key, bac_key])
        self.assertNotEqual(aec_key, bac_key)

    def test_a_refit_vintage_bac_key_is_reported_on_the_bac_record(self):
        """A BAC block refit under a later vintage is named by that block's own key."""
        aec_key = "LevelOfTheory(method='wb97mv',basis='def2tzvpd',software='qchem')"
        bac_key = "LevelOfTheory(method='wb97mv2023',basis='def2tzvpd',software='qchem')"
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_energy_corrections_for_species(
            'X', sc, self._lot(), 'p', aec_key=aec_key, bac_key=bac_key,
        )
        aec = next(e for e in out if e['correction_type'] == 'atom_energy')
        bac = next(e for e in out if e['correction_type'] == 'bond_additivity')
        self.assertEqual(aec['matched_arkane_key'], aec_key)
        self.assertEqual(bac['matched_arkane_key'], bac_key)

    def test_an_unmatched_bac_key_is_reported_as_null_not_as_the_aec_key(self):
        """An unmatched BAC section leaves the field present and null."""
        aec_key = "LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='qchem')"
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_energy_corrections_for_species(
            'X', sc, self._lot(), 'p', aec_key=aec_key, bac_key=None,
        )
        bac = next(e for e in out if e['correction_type'] == 'bond_additivity')
        self.assertIn('matched_arkane_key', bac)
        self.assertIsNone(bac['matched_arkane_key'])
        aec = next(e for e in out if e['correction_type'] == 'atom_energy')
        self.assertEqual(aec['matched_arkane_key'], aec_key)

    def test_an_unmatched_aec_key_leaves_the_bac_record_naming_its_own_key(self):
        """The AEC record's null key does not erase the BAC record's."""
        bac_key = "LevelOfTheory(method='wb97xd3',basis='def2tzvp')"
        sc = {'X': {'aec': self._aec_block(), 'bac': self._pbac_block()}}
        out = _build_energy_corrections_for_species(
            'X', sc, self._lot(), 'p', aec_key=None, bac_key=bac_key,
        )
        aec = next(e for e in out if e['correction_type'] == 'atom_energy')
        bac = next(e for e in out if e['correction_type'] == 'bond_additivity')
        self.assertIsNone(aec['matched_arkane_key'])
        self.assertEqual(bac['matched_arkane_key'], bac_key)

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
        self.assertEqual(bac['matched_arkane_key'], bac_key)
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

    def test_build_scan_result_on_an_unparseable_log_returns_none(self):
        """``parse_1d_scan_full_result`` honours its never-raises contract, so the
        builder carries no exception handling: a log that yields nothing — here an
        empty file, which used to raise ``TypeError`` out of the wrapper — still
        returns ``None`` rather than propagating."""
        project_directory = tempfile.mkdtemp(prefix='test_build_scan_result_')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        empty_log = os.path.join(project_directory, 'empty.out')
        with open(empty_log, 'w'):
            pass
        result = _build_scan_result_for_rotor(self._rotor(scan_path=empty_log), project_directory)
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

    def test_a_scan_record_states_the_program_of_its_own_log(self):
        """``ess_software`` and ``ess_version`` are identified from the scan log's banner, per rotor"""
        orca_log = os.path.join(ARC_TESTING_PATH, 'orca_example_opt.log')
        calcs = _build_rotor_scans(MagicMock(rotors_dict={0: self._rotor()}), '/tmp/project')
        self.assertEqual(calcs[0]['ess_software'], 'gaussian')
        self.assertIn('Gaussian', calcs[0]['ess_version'])
        with patch('arc.output._build_scan_result_for_rotor', return_value={'dimension': 1}):
            orca_calcs = _build_rotor_scans(MagicMock(rotors_dict={0: self._rotor(scan_path=orca_log)}), '/tmp/project')
        self.assertEqual(orca_calcs[0]['ess_software'], 'orca')
        self.assertIn('ORCA', orca_calcs[0]['ess_version'])

    def test_a_scan_record_without_a_readable_banner_states_null(self):
        """A missing log, or one that states no program, gives null and never raises"""
        scratch = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, scratch, True)
        bannerless = os.path.join(scratch, 'scan.log')
        with open(bannerless, 'w') as f:
            f.write('no program banner here\n')
        for scan_path in ('/nonexistent/scan.log', bannerless):
            with self.subTest(scan_path=scan_path), \
                    patch('arc.output._build_scan_result_for_rotor', return_value={'dimension': 1}):
                calcs = _build_rotor_scans(MagicMock(rotors_dict={0: self._rotor(scan_path=scan_path)}), '/tmp/project')
                self.assertIsNone(calcs[0]['ess_software'])
                self.assertIsNone(calcs[0]['ess_version'])

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

    def test_scan_points_state_the_isotopes_the_scan_log_states_for_their_geometry(self):
        """Every point carries the isotopes stated for its own geometry, beside ``geometry_xyz``."""
        parsed = self._stub_parsed(n_points=3)
        parsed['geometries'][0] = {key: value[:4] for key, value in parsed['geometries'][0].items()}
        with patch('arc.output.parse_1d_scan_full_result', return_value=parsed), \
                patch('arc.output._get_log_stated_isotopes',
                      side_effect=lambda path, symbols: [1 + (len(symbols) % 2)] * len(symbols)):
            result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        self.assertEqual([point['geometry_isotopes'] for point in result['samples']],
                         [[1] * 4, [2] * 5, [2] * 5])
        for point in result['samples']:
            self.assertEqual(len(point['geometry_isotopes']), len(point['geometry_xyz'].splitlines()))

    def test_scan_points_state_no_isotopes_a_scan_log_does_not_state(self):
        """A scan log that states no masses (the Gaussian opt-type scan) leaves every point's isotopes null."""
        with patch('arc.output.parse_1d_scan_full_result', return_value=self._stub_parsed(n_points=3)):
            result = _build_scan_result_for_rotor(self._rotor(), '/tmp/project')
        self.assertIsNotNone(result)
        for point in result['samples']:
            self.assertIsNone(point['geometry_isotopes'])

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
            self.assertNotIn('geometry_isotopes', point)
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


class TestTimedeltaToSeconds(unittest.TestCase):
    """Tests for _timedelta_to_seconds."""

    def test_none(self):
        self.assertIsNone(_timedelta_to_seconds(None))

    def test_timedelta(self):
        self.assertEqual(_timedelta_to_seconds(datetime.timedelta(minutes=1, seconds=30)), 90.0)

    def test_numeric(self):
        self.assertEqual(_timedelta_to_seconds(42), 42.0)
        self.assertEqual(_timedelta_to_seconds(12.345), 12.35)

    def test_str_hms(self):
        self.assertEqual(_timedelta_to_seconds('0:01:30'), 90.0)

    def test_str_hms_with_micro(self):
        self.assertEqual(_timedelta_to_seconds('1:00:00.500000'), 3600.5)

    def test_str_with_days(self):
        self.assertEqual(_timedelta_to_seconds('1 day, 2:00:00'), 93600.0)
        self.assertEqual(_timedelta_to_seconds('2 days, 0:00:01'), 172801.0)

    def test_unparseable_str(self):
        self.assertIsNone(_timedelta_to_seconds('bogus'))
        self.assertIsNone(_timedelta_to_seconds('12:30'))


class TestComputeCostMetrics(unittest.TestCase):
    """Tests for _compute_cost_metrics."""

    def test_empty_records(self):
        cm = _compute_cost_metrics(None, None)
        self.assertIsNone(cm['wall_time_hrs'])
        self.assertEqual(cm['total_job_count'], 0)
        self.assertEqual(cm['jobs_missing_time'], 0)
        self.assertEqual(cm['jobs_missing_cores'], 0)
        self.assertEqual(cm['total_execution_time_hrs'], 0.0)
        self.assertEqual(cm['total_core_hours'], 0.0)
        self.assertIsNone(cm['per_ess'])

    def test_two_ess_aggregation(self):
        records = [
            {'job_adapter': 'gaussian', 'cpu_cores': 8, 'run_time_sec': 7200.0},
            {'job_adapter': 'gaussian', 'cpu_cores': 4, 'run_time_sec': 1800.0},
            {'job_adapter': 'orca', 'cpu_cores': 16, 'run_time_sec': 3600.0},
        ]
        cm = _compute_cost_metrics(records, None)
        self.assertEqual(cm['total_job_count'], 3)
        self.assertEqual(cm['jobs_missing_time'], 0)
        self.assertAlmostEqual(cm['per_ess']['gaussian']['execution_time_hrs'], 2.5)
        self.assertAlmostEqual(cm['per_ess']['gaussian']['core_hours'], 18.0)
        self.assertAlmostEqual(cm['per_ess']['orca']['execution_time_hrs'], 1.0)
        self.assertAlmostEqual(cm['per_ess']['orca']['core_hours'], 16.0)
        self.assertAlmostEqual(cm['total_execution_time_hrs'], 3.5)
        self.assertAlmostEqual(cm['total_core_hours'], 34.0)

    def test_missing_time_and_cores_counted(self):
        records = [
            {'job_adapter': 'gaussian', 'cpu_cores': 8, 'run_time_sec': None},
            {'job_adapter': 'gaussian', 'cpu_cores': None, 'run_time_sec': 3600.0},
        ]
        cm = _compute_cost_metrics(records, None)
        self.assertEqual(cm['jobs_missing_time'], 1)
        self.assertEqual(cm['jobs_missing_cores'], 1)
        self.assertAlmostEqual(cm['per_ess']['gaussian']['execution_time_hrs'], 1.0)
        self.assertAlmostEqual(cm['per_ess']['gaussian']['core_hours'], 0.0)
        self.assertEqual(cm['per_ess']['gaussian']['jobs_missing_time'], 1)

    def test_run_time_as_timedelta_str(self):
        """Records restored from a restart file may carry str run times."""
        records = [{'job_adapter': 'xtb', 'cpu_cores': 1, 'run_time_sec': '0:30:00'}]
        cm = _compute_cost_metrics(records, None)
        self.assertAlmostEqual(cm['per_ess']['xtb']['execution_time_hrs'], 0.5)

    def test_unknown_adapter_bucket(self):
        records = [{'job_adapter': None, 'cpu_cores': 2, 'run_time_sec': 3600.0}]
        cm = _compute_cost_metrics(records, None)
        self.assertIn('unknown', cm['per_ess'])


class TestRmgDatabaseIdentity(unittest.TestCase):
    """Tests for ``_get_rmg_database_identity``, the identity of the quantum corrections Arkane loaded."""

    DATA = b"atom_energies = {}\npbac = {}\nmbac = {}\n"
    COMMIT = 'a1b2c3d4e5f60718293a4b5c6d7e8f9012345678'

    def make_database(self, prefix: str | None = None, data: bytes | None = None) -> str:
        """An RMG database directory holding ``input/quantum_corrections/data.py``, under a fresh temporary
        directory. Returns the path of that ``data.py``."""
        root = tempfile.mkdtemp(prefix='test_rmg_database_')
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        corrections = os.path.join(root, prefix or 'rmgdatabase', 'input', 'quantum_corrections')
        os.makedirs(corrections)
        data_path = os.path.join(corrections, 'data.py')
        with open(data_path, 'wb') as f:
            f.write(self.DATA if data is None else data)
        return data_path

    @staticmethod
    def database_root(data_path: str) -> str:
        return os.path.dirname(os.path.dirname(os.path.dirname(data_path)))

    @staticmethod
    def write(path: str, text: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(text)

    @staticmethod
    def identity(arkane_path, arc_root=None) -> dict:
        """The identity with Arkane reporting ``arkane_path`` and ARC reading the database at ``arc_root``."""
        with patch('arc.output._get_arkane_quantum_corrections_path', return_value=arkane_path):
            return _get_rmg_database_identity(arc_root)

    def test_a_git_checkout_states_its_head_and_the_hash_of_the_file_arkane_loaded(self):
        """A symbolic HEAD resolves through its loose ref, and a detached HEAD is the hash itself."""
        data_path = self.make_database()
        root = self.database_root(data_path)
        self.write(os.path.join(root, '.git', 'HEAD'), 'ref: refs/heads/main\n')
        self.write(os.path.join(root, '.git', 'refs', 'heads', 'main'), f'{self.COMMIT}\n')
        self.assertEqual(self.identity(data_path, root),
                         {'path_kind': 'git', 'git_commit': self.COMMIT, 'version': None,
                          'quantum_corrections_path': data_path,
                          'quantum_corrections_sha256': hashlib.sha256(self.DATA).hexdigest(),
                          'matches_arc_rmg_db_path': True})
        self.write(os.path.join(root, '.git', 'HEAD'), f'{self.COMMIT}\n')
        self.assertEqual(self.identity(data_path, root)['git_commit'], self.COMMIT)

    def test_an_unrelated_repository_further_up_is_not_the_database(self):
        """Only the database root, three directories above the file, can make the path a git checkout."""
        data_path = self.make_database()
        root = self.database_root(data_path)
        self.write(os.path.join(os.path.dirname(root), '.git', 'HEAD'), f'{self.COMMIT}\n')
        identity = self.identity(data_path, root)
        self.assertEqual((identity['path_kind'], identity['git_commit']), ('unknown', None))

    def test_a_packed_ref_and_a_worktree_gitfile_are_followed(self):
        data_path = self.make_database()
        root = self.database_root(data_path)
        common = os.path.join(os.path.dirname(root), 'main.git')
        self.write(os.path.join(common, 'packed-refs'), f'# pack-refs\n{self.COMMIT} refs/heads/main\n')
        worktree_git = os.path.join(common, 'worktrees', 'db')
        self.write(os.path.join(worktree_git, 'HEAD'), 'ref: refs/heads/main\n')
        self.write(os.path.join(worktree_git, 'commondir'), '../..\n')
        self.write(os.path.join(root, '.git'), f'gitdir: {worktree_git}\n')
        identity = self.identity(data_path, root)
        self.assertEqual((identity['path_kind'], identity['git_commit']), ('git', self.COMMIT))

    def test_an_unreadable_git_head_is_a_git_checkout_without_a_commit(self):
        data_path = self.make_database()
        root = self.database_root(data_path)
        self.write(os.path.join(root, '.git', 'HEAD'), 'garbage\n')
        identity = self.identity(data_path, root)
        self.assertEqual((identity['path_kind'], identity['git_commit']), ('git', None))
        self.assertIsNotNone(identity['quantum_corrections_sha256'])

    def test_a_conda_package_states_its_version_and_no_commit(self):
        data_path = self.make_database(prefix=os.path.join('envs', 'rmg_env', 'share', 'rmgdatabase'))
        prefix = os.path.dirname(os.path.dirname(os.path.dirname(self.database_root(data_path))))
        self.write(os.path.join(prefix, 'conda-meta', 'rmgdatabase-4.0.0.rc1-py39_0.json'),
                   json.dumps({'name': 'rmgdatabase', 'version': '4.0.0.rc1'}))
        self.write(os.path.join(prefix, 'conda-meta', 'rmgdatabase-extras-1.0-0.json'),
                   json.dumps({'name': 'rmgdatabase-extras', 'version': '1.0'}))
        identity = self.identity(data_path)
        self.assertEqual((identity['path_kind'], identity['git_commit'], identity['version']),
                         ('package', None, '4.0.0.rc1'))
        self.assertIsNotNone(identity['quantum_corrections_sha256'])

    def test_a_conda_record_that_is_not_a_mapping_is_ignored(self):
        data_path = self.make_database(prefix=os.path.join('envs', 'rmg_env', 'share', 'rmgdatabase'))
        prefix = os.path.dirname(os.path.dirname(os.path.dirname(self.database_root(data_path))))
        self.write(os.path.join(prefix, 'conda-meta', 'rmgdatabase-4.0.0-0.json'), json.dumps(['rmgdatabase']))
        identity = self.identity(data_path)
        self.assertEqual((identity['path_kind'], identity['version']), ('unknown', None))

    def test_the_hash_is_of_the_file_arkane_loaded_not_the_one_arc_reads(self):
        """When the two differ, the digest is Arkane's, ``matches_arc_rmg_db_path`` is false and a warning says so."""
        arkane_path = self.make_database(data=b'arkane = {}\n')
        arc_root = self.database_root(self.make_database(data=b'arc = {}\n'))
        with self.assertLogs('arc', level='WARNING') as captured:
            identity = self.identity(arkane_path, arc_root)
        self.assertEqual(identity['quantum_corrections_sha256'], hashlib.sha256(b'arkane = {}\n').hexdigest())
        self.assertIs(identity['matches_arc_rmg_db_path'], False)
        self.assertTrue(any('differ' in record.getMessage() for record in captured.records))

    def test_nothing_is_guessed_and_nothing_raises(self):
        """An Arkane path that is unavailable or unreadable, and an ARC path that is missing or unset, give null."""
        unknown = {'path_kind': 'unknown', 'git_commit': None, 'version': None, 'quantum_corrections_path': None,
                   'quantum_corrections_sha256': None, 'matches_arc_rmg_db_path': None}
        self.assertEqual(self.identity(None), unknown)
        missing = os.path.join(tempfile.gettempdir(), 'test_rmg_database_missing', 'data.py')
        self.assertEqual(self.identity(missing), dict(unknown, quantum_corrections_path=missing))
        data_path = self.make_database()
        with patch.dict('arc.output.settings', {'RMG_DB_PATH': None}):
            identity = self.identity(data_path)
        self.assertIsNone(identity['matches_arc_rmg_db_path'])
        self.assertIsNotNone(identity['quantum_corrections_sha256'])
        identity = self.identity(data_path, os.path.join(os.path.dirname(data_path), 'missing'))
        self.assertIsNone(identity['matches_arc_rmg_db_path'])

    def test_arkanes_path_is_read_from_the_rmg_environment_helper(self):
        """The path comes from the ``quantum_corrections_path`` the rmg_env script writes, and a failure is null."""
        with patch('arc.output.execute_command', return_value=('', '')), \
                patch('arc.output.read_yaml_file', return_value={'quantum_corrections_path': '/db/data.py'}):
            self.assertEqual(_get_arkane_quantum_corrections_path(), '/db/data.py')
        with patch('arc.output.execute_command', return_value=('', '')), \
                patch('arc.output.read_yaml_file', return_value={'aec': None}):
            self.assertIsNone(_get_arkane_quantum_corrections_path())
        with patch('arc.output.execute_command', side_effect=OSError('no rmg_env')):
            self.assertIsNone(_get_arkane_quantum_corrections_path())

    def test_the_header_of_the_document_carries_the_identity(self):
        identity = {'path_kind': 'git', 'git_commit': self.COMMIT, 'version': None, 'quantum_corrections_path': 'p',
                    'quantum_corrections_sha256': None, 'matches_arc_rmg_db_path': None}
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        with patch('arc.output._get_rmg_database_identity', return_value=identity), \
                patch('arc.output._get_arkane_provenance', return_value=(None, None)), \
                patch('arc.output.get_git_commit', return_value=('', '')):
            write_output_yml(project='p', project_directory=tmp_dir, species_dict=dict(), reactions=[],
                             output_dict=dict())
        document = read_yaml_file(os.path.join(tmp_dir, 'output', 'output.yml'))
        self.assertEqual(document['rmg_database'], identity)
        self.assertIsNone(document['arc_aec_yml_sha256'])


class TestStatmechArkaneKeys(unittest.TestCase):
    """The E0 correction switches and the Arkane treatment in ``_statmech_to_dict``."""

    def test_the_keys_come_from_the_species(self):
        spc = TestStatmechToDict()._make_spc(freqs=[100.0])
        spc.e0_atom_corrections_applied = True
        spc.e0_bond_corrections_applied = False
        spc.arkane_rotor_modes = ['HinderedRotor', 'HinderedRotorClassicalND']
        result = _statmech_to_dict(spc, '/tmp/project')
        self.assertIs(result['e0_atom_corrections_applied'], True)
        self.assertIs(result['e0_bond_corrections_applied'], False)
        self.assertEqual(result['arkane_rotors_applied'], 2)
        self.assertEqual(result['arkane_treatment'], 'rrho_1d_nd')

    def test_an_unparsed_arkane_output_is_null_and_an_empty_one_is_a_rigid_rotor(self):
        spc = TestStatmechToDict()._make_spc(freqs=[100.0])
        result = _statmech_to_dict(spc, '/tmp/project')
        self.assertIsNone(result['arkane_rotors_applied'])
        self.assertIsNone(result['arkane_treatment'])
        spc.arkane_rotor_modes = []
        result = _statmech_to_dict(spc, '/tmp/project')
        self.assertEqual(result['arkane_rotors_applied'], 0)
        self.assertEqual(result['arkane_treatment'], 'rrho')

    def test_the_kinetics_block_carries_the_switch_of_its_run(self):
        rxn = MagicMock()
        rxn.label, rxn.reactants, rxn.products, rxn.family, rxn.multiplicity, rxn.ts_label = 'r', [], [], 'f', 1, 'TS0'
        rxn.long_kinetic_description = None
        for switch in (True, False, None):
            rxn.kinetics = {'A': (1.0, 's^-1'), 'atom_corrections_applied': switch}
            self.assertIs(_rxn_to_dict(rxn)['kinetics']['atom_corrections_applied'], switch)
        rxn.kinetics = {'A': (1.0, 's^-1')}
        self.assertIsNone(_rxn_to_dict(rxn)['kinetics']['atom_corrections_applied'])


class TestQuantumCorrectionsPathHelper(unittest.TestCase):
    """Reading the path Arkane loaded must survive a full disk and be skipped when it is already known."""

    def test_a_failing_mkstemp_gives_none_and_leaves_no_file_behind(self):
        """Test that a failing mkstemp gives none and leaves no file behind"""
        with patch('arc.output.tempfile.mkstemp', side_effect=OSError(28, 'No space left on device')):
            self.assertIsNone(_get_arkane_quantum_corrections_path())
        fd, leftover = tempfile.mkstemp(suffix='.qm_input.yml')
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(leftover) and os.unlink(leftover))
        with patch('arc.output.tempfile.mkstemp', side_effect=[(os.open(leftover, os.O_RDONLY), leftover),
                                                              OSError(28, 'No space left on device')]):
            self.assertIsNone(_get_arkane_quantum_corrections_path())
        self.assertFalse(os.path.exists(leftover))

    def test_the_output_is_still_written_when_no_temporary_file_can_be_made(self):
        """Test that the output is still written when no temporary file can be made"""
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        real_mkstemp = tempfile.mkstemp

        def full_disk_for_arkane_scratch(*args, **kwargs):
            """Helper: full disk for arkane scratch"""
            if str(kwargs.get('suffix', '')).startswith('.qm_'):
                raise OSError(28, 'No space left on device')
            return real_mkstemp(*args, **kwargs)

        with patch('arc.output._get_arkane_quantum_corrections_path', _get_arkane_quantum_corrections_path), \
                patch('arc.output._get_arkane_provenance', return_value=(None, None)), \
                patch('arc.output.get_git_commit', return_value=('', '')), \
                patch('arc.output.tempfile.mkstemp', side_effect=full_disk_for_arkane_scratch):
            write_output_yml(project='p', project_directory=tmp_dir, species_dict=dict(), reactions=[],
                             output_dict=dict())
        document = read_yaml_file(os.path.join(tmp_dir, 'output', 'output.yml'))
        self.assertIsNone(document['rmg_database']['quantum_corrections_path'])
        self.assertEqual(document['rmg_database']['path_kind'], 'unknown')

    def test_the_corrections_lookup_survives_a_failing_second_mkstemp(self):
        """Test that the corrections lookup survives a failing second mkstemp"""
        fd, leftover = tempfile.mkstemp(suffix='.qm_input.yml')
        self.addCleanup(lambda: os.path.exists(leftover) and os.unlink(leftover))
        lot = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        with patch('arc.output._match_arkane_correction_keys', return_value=('aec-key', 'bac-key')), \
                patch('arc.output.tempfile.mkstemp', side_effect=[(fd, leftover),
                                                                  OSError(28, 'No space left on device')]), \
                patch('arc.output.execute_command') as execute:
            corrections = _get_energy_corrections(lot, 'p')
        execute.assert_not_called()
        self.assertEqual(corrections, EnergyCorrections(None, None, 'aec-key', 'bac-key', None))
        self.assertFalse(os.path.exists(leftover))

    def test_the_script_reports_its_path_with_the_tables(self):
        """Test that the script reports its path with the tables"""
        lot = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        with patch('arc.output._match_arkane_correction_keys', return_value=('aec', None)), \
                patch('arc.output.execute_command', return_value=('', '')), \
                patch('arc.output.read_yaml_file', return_value={'aec': {'H': -0.5}, 'bac': None,
                                                                 'quantum_corrections_path': '/db/data.py'}):
            corrections = _get_energy_corrections(lot, None)
        self.assertEqual(corrections.quantum_corrections_path, '/db/data.py')
        self.assertEqual(corrections.aec, {'H': -0.5})
        with patch('arc.output._match_arkane_correction_keys', return_value=(None, None)):
            self.assertIsNone(_get_energy_corrections(lot, None).quantum_corrections_path)

    def test_a_known_path_spares_the_dedicated_call(self):
        """Test that a known path spares the dedicated call"""
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        data_path = os.path.join(tmp_dir, 'db', 'input', 'quantum_corrections', 'data.py')
        os.makedirs(os.path.dirname(data_path))
        with open(data_path, 'w') as f:
            f.write('atom_energies = {}\n')
        with patch('arc.output._get_arkane_quantum_corrections_path') as helper:
            identity = _get_rmg_database_identity(arkane_path=data_path)
        helper.assert_not_called()
        self.assertEqual(identity['quantum_corrections_path'], data_path)

    def test_the_header_takes_the_path_from_the_corrections_and_asks_only_when_it_is_missing(self):
        """Test that the header takes the path from the corrections and asks only when it is missing"""
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        for reported, asked in (('/db/data.py', False), (None, True)):
            with self.subTest(reported=reported):
                corrections = EnergyCorrections(None, None, None, None, reported)
                with patch('arc.output._get_energy_corrections', return_value=corrections), \
                        patch('arc.output._get_rmg_database_identity', return_value={}) as identity, \
                        patch('arc.output._get_arkane_provenance', return_value=(None, None)), \
                        patch('arc.output.get_git_commit', return_value=('', '')):
                    write_output_yml(project='p', project_directory=tmp_dir, species_dict=dict(), reactions=[],
                                     output_dict=dict())
                identity.assert_called_once_with(arkane_path=reported)


class TestArcAecYmlDigestLevel(unittest.TestCase):
    """``arc_aec_yml_sha256`` is set when AEC.yml atomEnergies are rendered for the Arkane level of theory or for the
    composite or single-point level of the TS-check E0 run."""

    DIGEST = 'a' * 64

    def _digest(self, rendered_for: list, reactions=None, **levels):
        """Helper: digest"""
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)

        def digest_for(level):
            """Helper: digest for"""
            return self.DIGEST if any(level is candidate for candidate in rendered_for) else None

        with patch('arc.output.get_arc_aec_yml_sha256', side_effect=digest_for), \
                patch('arc.output._get_rmg_database_identity', return_value={}), \
                patch('arc.output._get_energy_corrections', return_value=EnergyCorrections(None, None, None, None)), \
                patch('arc.output._get_arkane_provenance', return_value=(None, None)), \
                patch('arc.output._rxn_to_dict', return_value={}), \
                patch('arc.output.get_git_commit', return_value=('', '')):
            write_output_yml(project='p', project_directory=tmp_dir, species_dict=dict(),
                             reactions=[MagicMock()] if reactions else [], output_dict=dict(), **levels)
        return read_yaml_file(os.path.join(tmp_dir, 'output', 'output.yml'))['arc_aec_yml_sha256']

    def test_a_composite_run_has_no_sp_level_and_still_uses_the_arkane_level(self):
        """Test that a composite run has no sp level and still uses the arkane level"""
        arkane_level = Level(method='cbs-qb3', software='gaussian')
        self.assertEqual(self._digest([arkane_level], sp_level=None, arkane_level_of_theory=arkane_level), self.DIGEST)

    def test_an_explicit_arkane_level_is_judged_even_when_the_sp_level_is_not_rendered(self):
        """Test that an explicit arkane level is judged even when the sp level is not rendered"""
        arkane_level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        sp_level = Level(method='dlpno-ccsd(t)', basis='cc-pvtz', software='orca')
        self.assertEqual(self._digest([arkane_level], sp_level=sp_level, arkane_level_of_theory=arkane_level),
                         self.DIGEST)

    def test_the_e0_run_levels_count_only_for_a_run_with_reactions(self):
        """Test that the e0 run levels count only for a run with reactions"""
        arkane_level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        sp_level = Level(method='dlpno-ccsd(t)', basis='cc-pvtz', software='orca')
        composite = Level(method='cbs-qb3', software='gaussian')
        self.assertEqual(self._digest([sp_level], reactions=True, sp_level=sp_level,
                                      arkane_level_of_theory=arkane_level), self.DIGEST)
        self.assertEqual(self._digest([composite], reactions=True, composite_method=composite,
                                      arkane_level_of_theory=arkane_level), self.DIGEST)
        self.assertIsNone(self._digest([], reactions=True, sp_level=sp_level, composite_method=composite,
                                       arkane_level_of_theory=arkane_level))
        self.assertIsNone(self._digest([sp_level], sp_level=sp_level, arkane_level_of_theory=arkane_level))
        self.assertIsNone(self._digest([composite], composite_method=composite,
                                       arkane_level_of_theory=arkane_level))
        self.assertEqual(self._digest([arkane_level], sp_level=sp_level, arkane_level_of_theory=arkane_level),
                         self.DIGEST)


class TestTsFrequenciesInEssOrder(unittest.TestCase):
    """The TS-only ``freq_frequencies_cm1_ess_order`` and ``reaction_coordinate_mode_index``."""

    make_spc = TestSpeciesLevelsAndIrcEndpoints.make_spc

    def _ts(self, freqs, nmd, converged=True, record=None):
        """Helper: ts"""
        spc = self.make_spc('TS0', is_ts=True)
        spc.freqs = freqs
        spc.ts_checks = {'NMD': nmd, 'warnings': ''}
        spc.nmd_record = {'frequency_cm1': min(freqs)} if record is None and freqs else (record or dict())
        return _spc_to_dict(spc, {'TS0': {'convergence': converged, 'job_types': {}, 'paths': {}}}, '/abs')

    def test_the_validated_mode_is_indexed_even_when_it_is_not_first(self):
        """With synthetic frequencies whose validated mode is listed second, the index counts from one"""
        freqs = [-18.2, -1235.4, 101.0, 350.5, 1500.0]
        result = self._ts(freqs, True)
        self.assertEqual(result['freq_frequencies_cm1_ess_order'], freqs)
        self.assertEqual(result['reaction_coordinate_mode_index'], 2)
        self.assertEqual(result['imaginary_frequencies_cm1'], [-1235.4, -18.2])

    def test_the_order_is_the_esss_not_a_sorted_one(self):
        """With a synthetic unsorted list, the frequencies are exported in the order given"""
        freqs = [300.0, -50.0, 1500.0, -900.0, 120.0]
        result = self._ts(freqs, True)
        self.assertEqual(result['freq_frequencies_cm1_ess_order'], freqs)
        self.assertEqual(result['reaction_coordinate_mode_index'], 4)

    def test_the_index_is_null_unless_the_displacement_check_passed(self):
        """Test that the index is null unless the displacement check passed"""
        freqs = [-18.2, -1235.4, 101.0]
        for nmd in (False, None):
            with self.subTest(nmd=nmd):
                result = self._ts(freqs, nmd)
                self.assertEqual(result['freq_frequencies_cm1_ess_order'], freqs)
                self.assertIsNone(result['reaction_coordinate_mode_index'])

    def test_a_check_forced_to_pass_by_skip_nmd_gives_no_index(self):
        """Test that a check forced to pass by skip nmd gives no index"""
        freqs = [-18.2, -1235.4, 101.0]
        result = self._ts(freqs, True, record={'frequency_cm1': -1235.4, 'forced': True})
        self.assertEqual(result['freq_frequencies_cm1_ess_order'], freqs)
        self.assertIsNone(result['reaction_coordinate_mode_index'])

    def test_the_index_needs_exactly_one_frequency_matching_the_validated_one(self):
        """Test that the index needs exactly one frequency matching the validated one"""
        freqs = [-18.2, -1235.4, 101.0]
        for validated, expected in ((-1235.405, 2), (-1235.4 + 0.02, None), (-900.0, None), (-18.2, 1)):
            with self.subTest(validated=validated):
                result = self._ts(freqs, True, record={'frequency_cm1': validated})
                self.assertEqual(result['reaction_coordinate_mode_index'], expected)
        result = self._ts([-500.0, -500.004, 100.0], True, record={'frequency_cm1': -500.0})
        self.assertIsNone(result['reaction_coordinate_mode_index'])

    def test_a_restart_without_a_recorded_frequency_gives_no_index(self):
        """Test that a restart without a recorded frequency gives no index"""
        freqs = [-18.2, -1235.4, 101.0]
        for record in (dict(), {'frequency_cm1': None}, {'frequency_cm1': True}):
            with self.subTest(record=record):
                spc = self.make_spc('TS0', is_ts=True)
                spc.freqs = freqs
                spc.ts_checks = {'NMD': True, 'warnings': ''}
                spc.nmd_record = record
                result = _spc_to_dict(spc, {'TS0': {'convergence': True, 'job_types': {}, 'paths': {}}}, '/abs')
                self.assertIsNone(result['reaction_coordinate_mode_index'])

    def test_both_are_null_without_recorded_frequencies_or_a_converged_run(self):
        """Test that both are null without recorded frequencies or a converged run"""
        for freqs, converged in ((None, True), ([], True), ([-1000.0, 100.0], False)):
            with self.subTest(freqs=freqs, converged=converged):
                result = self._ts(freqs, True, converged=converged)
                self.assertIsNone(result['freq_frequencies_cm1_ess_order'])
                self.assertIsNone(result['reaction_coordinate_mode_index'])

    def test_a_validated_ts_without_a_matching_negative_mode_has_no_index(self):
        """Test that a validated ts without a matching negative mode has no index"""
        result = self._ts([100.0, 200.0], True, record={'frequency_cm1': -1000.0})
        self.assertEqual(result['freq_frequencies_cm1_ess_order'], [100.0, 200.0])
        self.assertIsNone(result['reaction_coordinate_mode_index'])

    def test_a_species_record_has_neither_key(self):
        """Test that a species record has neither key"""
        result = _spc_to_dict(self.make_spc(), {}, '/abs')
        self.assertNotIn('freq_frequencies_cm1_ess_order', result)
        self.assertNotIn('reaction_coordinate_mode_index', result)


class TestReactionAtomMapExport(unittest.TestCase):
    """The reaction ``atom_map`` family of keys."""

    keys = ('atom_map', 'atom_map_reactant_labels', 'atom_map_product_labels', 'atom_map_source', 'atom_map_method')

    @staticmethod
    def _rxn():
        h2o_xyz_1 = """O      -0.00032832    0.39781490    0.00000000
                       H      -0.76330345   -0.19953755    0.00000000
                       H       0.76363177   -0.19827735    0.00000000"""
        h2o_xyz_2 = """H      -0.76330345   -0.19953755    0.00000000
                       H       0.76363177   -0.19827735    0.00000000
                       O      -0.00032832    0.39781490    0.00000000"""
        return ARCReaction(reactants=['H2O'], products=['H2O'],
                           r_species=[ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz_1)],
                           p_species=[ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz_2)])

    def test_a_declared_map_is_exported_as_declared(self):
        rxn = self._rxn()
        rxn.declare_atom_map([2, 0, 1])
        result = _rxn_to_dict(rxn)
        self.assertEqual(result['atom_map'], [2, 0, 1])
        self.assertEqual(result['atom_map_source'], 'declared')
        self.assertIsNone(result['atom_map_method'])

    def test_a_map_of_unrecorded_origin_has_no_source(self):
        rxn = self._rxn()
        rxn.atom_map = [2, 0, 1]
        result = _rxn_to_dict(rxn)
        self.assertEqual(result['atom_map'], [2, 0, 1])
        self.assertIsNone(result['atom_map_source'])
        self.assertIsNone(result['atom_map_method'])

    def test_an_inferred_map_is_exported_as_inferred_with_its_method_and_stays_so_after_a_restart(self):
        rxn = self._rxn()
        self.assertEqual(rxn.atom_map, [2, 0, 1])
        for exported in (rxn, ARCReaction(reaction_dict=rxn.as_dict())):
            result = _rxn_to_dict(exported)
            self.assertEqual((result['atom_map'], result['atom_map_source']), ([2, 0, 1], 'inferred'))
            self.assertTrue(result['atom_map_method'].startswith('arc.mapping.driver.map_reaction'))

    def test_exporting_never_computes_a_map(self):
        rxn = self._rxn()
        with patch('arc.reaction.reaction.map_reaction') as map_reaction:
            result = _rxn_to_dict(rxn)
            map_reaction.assert_not_called()
        for key in self.keys:
            self.assertIsNone(result[key])
        self.assertIsNone(rxn._atom_map)

    def test_a_product_without_a_geometry_exports_no_map_labels_source_or_method(self):
        """Test that a map whose product has no geometry is withheld whole, with its provenance."""
        rxn = self._rxn()
        rxn.declare_atom_map([2, 0, 1])
        product = rxn.p_species[0]
        product.initial_xyz = None
        product.final_xyz = None
        product.cheap_conformer = None
        product.most_stable_conformer = None
        product.conformers = list()
        self.assertIsNone(product.get_xyz(generate=False))
        result = _rxn_to_dict(rxn)
        for key in self.keys:
            self.assertIsNone(result[key], key)

    def test_the_labels_follow_the_species_order_the_map_counts_atoms_in_and_expand_repeats(self):
        """Test that labels follow r_species order, not the sorted reactant_labels, one per occurrence."""
        ch4 = ARCSpecies(label='CH4', smiles='C', xyz="""C 0.0 0.0 0.0
                                                      H 0.629 0.629 0.629
                                                      H -0.629 -0.629 0.629
                                                      H -0.629 0.629 -0.629
                                                      H 0.629 -0.629 -0.629""")
        oh = ARCSpecies(label='OH', smiles='[OH]', xyz="""O 0.0 0.0 0.0
                                                       H 0.97 0.0 0.0""")
        h2o = ARCSpecies(label='H2O', smiles='O', xyz="""O 0.0 0.0 0.0
                                                       H 0.96 0.0 0.0
                                                       H -0.24 0.93 0.0""")
        ch3 = ARCSpecies(label='CH3', smiles='[CH3]', xyz="""C 0.0 0.0 0.0
                                                           H 1.08 0.0 0.0
                                                           H -0.54 0.935 0.0
                                                           H -0.54 -0.935 0.0""")
        rxn = ARCReaction(r_species=[oh, ch4], p_species=[h2o, ch3])
        rxn.declare_atom_map([0, 1, 3, 2, 4, 5, 6])
        result = _rxn_to_dict(rxn)
        self.assertEqual(result['reactant_labels'], ['CH4', 'OH'])
        self.assertEqual(result['atom_map_reactant_labels'], ['OH', 'CH4'])
        self.assertEqual(result['atom_map_product_labels'], ['H2O', 'CH3'])
        self.assertEqual(result['atom_map'], [0, 1, 3, 2, 4, 5, 6])
        rxn = ARCReaction(label='CH3 + CH3 <=> C2H6',
                          r_species=[ch3],
                          p_species=[ARCSpecies(label='C2H6', smiles='CC', xyz="""C 0.0 0.0 0.0
                                                                              H -0.36 1.028 0.0
                                                                              H -0.36 -0.514 0.89
                                                                              H -0.36 -0.514 -0.89
                                                                              C 1.54 0.0 0.0
                                                                              H 1.9 1.028 0.0
                                                                              H 1.9 -0.514 0.89
                                                                              H 1.9 -0.514 -0.89""")])
        rxn.declare_atom_map([0, 1, 2, 3, 4, 5, 6, 7])
        result = _rxn_to_dict(rxn)
        self.assertEqual(result['reactant_labels'], ['CH3'])
        self.assertEqual(result['atom_map_reactant_labels'], ['CH3', 'CH3'])
        self.assertEqual(result['atom_map_product_labels'], ['C2H6'])

    def test_a_map_that_does_not_conserve_the_element_is_not_exported(self):
        rxn = self._rxn()
        rxn.declare_atom_map([2, 1, 0])
        self.assertEqual(_rxn_to_dict(rxn)['atom_map'], [2, 1, 0])
        rxn.declare_atom_map([0, 2, 1])
        result = _rxn_to_dict(rxn)
        for key in self.keys:
            self.assertIsNone(result[key])
        for bad in ([0, 1], [0, 1, 1], [0, 1, 3]):
            with self.subTest(bad=bad):
                rxn.declare_atom_map(bad)
                self.assertIsNone(_rxn_to_dict(rxn)['atom_map'])

    def test_a_malformed_map_is_not_exported_and_an_unknown_source_is_null(self):
        rxn = self._rxn()
        for bad in ('0,1,2', [0, 'a', 2], [True, 0, 1]):
            with self.subTest(bad=bad):
                rxn._atom_map = bad
                result = _rxn_to_dict(rxn)
                for key in self.keys:
                    self.assertIsNone(result[key])
        rxn._atom_map, rxn._atom_map_source = [2, 0, 1], 'guessed'
        result = _rxn_to_dict(rxn)
        self.assertEqual(result['atom_map'], [2, 0, 1])
        self.assertIsNone(result['atom_map_source'])
        self.assertIsNone(result['atom_map_method'])


class TestIrcParticipantMappingExport(unittest.TestCase):
    """The TS-only ``irc_participant_mapping``."""

    make_spc = TestSpeciesLevelsAndIrcEndpoints.make_spc
    mapping = {'sides_distinguishable': True, 'atom_order_matches_ts': None,
               'reactants': {'endpoint': 1, 'endpoint_label': 'IRC_TS0_1',
                             'participants': [{'label': 'CH3', 'position': 1, 'occurrence': 1,
                                               'atom_indices': [0, 1, 2, 3]},
                                              {'label': 'CH3', 'position': 2, 'occurrence': 2,
                                               'atom_indices': [4, 5, 6, 7]}]},
               'products': {'endpoint': 2, 'endpoint_label': None,
                            'participants': [{'label': 'C2H6', 'position': 1, 'occurrence': 1,
                                              'atom_indices': list(range(8))}]}}

    def _ts(self, mapping, irc=True):
        spc = self.make_spc('TS0', is_ts=True)
        spc.ts_checks = {'IRC': irc, 'warnings': ''}
        spc.irc_participant_mapping = copy.deepcopy(mapping)
        return _spc_to_dict(spc, {'TS0': {'convergence': True, 'job_types': {}, 'paths': {}}}, '/abs')

    def test_a_recorded_mapping_is_exported_as_recorded(self):
        self.assertEqual(self._ts(self.mapping)['irc_participant_mapping'], self.mapping)

    def test_it_is_null_when_not_recorded_or_when_the_irc_did_not_validate_the_ts(self):
        for mapping, irc in ((None, True), (self.mapping, False), (self.mapping, None)):
            with self.subTest(recorded=mapping is not None, irc=irc):
                self.assertIsNone(self._ts(mapping, irc=irc)['irc_participant_mapping'])

    def test_a_malformed_record_is_dropped_instead_of_exported(self):
        for path, value in ((('reactants', 'endpoint'), 3),
                            (('reactants', 'endpoint'), True),
                            (('products', 'endpoint_label'), 5),
                            (('products', 'participants'), 'x'),
                            (('reactants', 'participants', 0, 'atom_indices'), [0, 'a']),
                            (('reactants', 'participants', 0, 'label'), None),
                            (('reactants', 'participants', 0, 'position'), 0),
                            (('reactants', 'participants', 0, 'occurrence'), 0),
                            (('reactants', 'participants', 0, 'atom_indices'), [-1]),
                            (('reactants', 'participants', 0, 'atom_indices'), [1, 1]),
                            (('sides_distinguishable',), 'yes'),
                            (('atom_order_matches_ts',), 'yes')):
            with self.subTest(path=path):
                mapping = copy.deepcopy(self.mapping)
                target = mapping
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assertIsNone(self._ts(mapping)['irc_participant_mapping'])
        self.assertIsNone(self._ts({'reactants': self.mapping['reactants']})['irc_participant_mapping'])
        for missing in ('sides_distinguishable', 'atom_order_matches_ts'):
            mapping = copy.deepcopy(self.mapping)
            del mapping[missing]
            self.assertIsNone(self._ts(mapping)['irc_participant_mapping'])

    def test_a_species_record_has_no_such_key(self):
        self.assertNotIn('irc_participant_mapping', _spc_to_dict(self.make_spc(), {}, '/abs'))


def _xyz_dict(symbols, coords, isotopes=None):
    """Helper: xyz dict"""
    return xyz_from_data(coords=coords, symbols=symbols, isotopes=isotopes)


def _benzene_coords(stretch=1.0):
    """Helper: benzene coords"""
    coords, symbols = [], []
    for radius, symbol in ((1.397, 'C'), (2.481, 'H')):
        for k in range(6):
            angle = math.radians(60 * k)
            coords.append((stretch * radius * math.cos(angle), radius * math.sin(angle), 0.0))
            symbols.append(symbol)
    return symbols, coords


def _ammonia_coords():
    """Helper: ammonia coords"""
    bond, hnh = 1.012, math.radians(106.7)
    theta = math.asin(2 / math.sqrt(3) * math.sin(hnh / 2))
    rho, z = bond * math.sin(theta), -bond * math.cos(theta)
    hydrogens = [(rho * math.cos(math.radians(a)), rho * math.sin(math.radians(a)), z) for a in (0, 120, 240)]
    return ['N', 'H', 'H', 'H'], [(0.0, 0.0, 0.0)] + hydrogens


def _water_coords():
    """Helper: water coords"""
    bond, hoh = 0.9572, math.radians(104.5)
    return ['O', 'H', 'H'], [(0.0, 0.0, 0.0), (bond * math.sin(hoh / 2), bond * math.cos(hoh / 2), 0.0),
                             (-bond * math.sin(hoh / 2), bond * math.cos(hoh / 2), 0.0)]


def _allene_coords(stretch=1.0):
    """Helper: allene coords"""
    cc, ch, half = 1.31, 1.09, math.radians(59.2)
    coords, symbols = [(0.0, 0.0, 0.0), (0.0, 0.0, cc), (0.0, 0.0, -cc)], ['C', 'C', 'C']
    for sign, along_x in ((1, True), (-1, False)):
        for side in (1, -1):
            offset = side * ch * math.sin(half)
            coords.append((offset * stretch, 0.0, sign * (cc + ch * math.cos(half))) if along_x
                          else (0.0, offset, sign * (cc + ch * math.cos(half))))
            symbols.append('H')
    return symbols, coords


def _sf6_coords():
    """Helper: sf6 coords"""
    d = 1.56
    return ['S'] + ['F'] * 6, [(0.0, 0.0, 0.0), (d, 0, 0), (-d, 0, 0), (0, d, 0), (0, -d, 0), (0, 0, d), (0, 0, -d)]


def _methane_coords():
    """Helper: methane coords"""
    d = 0.629
    return ['C', 'H', 'H', 'H', 'H'], [(0.0, 0.0, 0.0), (d, d, d), (-d, -d, d), (-d, d, -d), (d, -d, -d)]


class TestRigidRotorKind(unittest.TestCase):
    """``statmech.rigid_rotor_kind`` from the principal moments of inertia of the exported geometry."""

    def _kind(self, symbols, coords, is_linear=False, isotopes=None, point_group=None):
        """Helper: kind"""
        spc = TestStatmechToDict()._make_spc(is_linear=is_linear, freqs=[100.0])
        spc.label = 'x'
        spc.final_xyz, spc.initial_xyz = _xyz_dict(symbols, coords, isotopes), None
        return _statmech_to_dict(spc, '/tmp/project', point_group=point_group)['rigid_rotor_kind']

    def test_symmetric_and_spherical_tops(self):
        """Test that symmetric and spherical tops"""
        self.assertEqual(self._kind(*_benzene_coords()), 'symmetric_top')
        self.assertEqual(self._kind(*_ammonia_coords()), 'symmetric_top')
        self.assertEqual(self._kind(*_methane_coords()), 'spherical_top')
        self.assertEqual(self._kind(*_sf6_coords()), 'spherical_top')

    def test_asymmetric_and_linear(self):
        """Test that asymmetric and linear"""
        self.assertEqual(self._kind(*_water_coords()), 'asymmetric_top')
        self.assertEqual(self._kind(['O', 'C', 'O'], [(-1.16, 0, 0), (0, 0, 0), (1.16, 0, 0)], is_linear=True),
                         'linear')

    def _unflagged(self, symbols, coords, is_linear=None):
        """Helper: unflagged"""
        spc = TestStatmechToDict()._make_spc(freqs=[100.0])
        spc._is_linear = is_linear
        spc.label = 'x'
        spc.final_xyz, spc.initial_xyz = _xyz_dict(symbols, coords), None
        return _statmech_to_dict(spc, '/tmp/project')['rigid_rotor_kind']

    def test_a_linear_species_that_does_not_say_so_is_not_a_symmetric_top(self):
        """Test that a linear species that does not say so is not a symmetric top"""
        self.assertEqual(self._unflagged(['O', 'C', 'O'], [(-1.16, 0, 0), (0, 0, 0), (1.16, 0, 0)]), 'linear')

    def test_a_bent_co2_is_not_linear(self):
        """At 174 degrees the smallest moment is about 7.5e-4 of the next, far above the collinearity cutoff"""
        half = math.radians(174 / 2)
        coords = [(-1.16 * math.sin(half), -1.16 * math.cos(half), 0.0), (0.0, 0.0, 0.0),
                  (1.16 * math.sin(half), -1.16 * math.cos(half), 0.0)]
        self.assertNotEqual(self._unflagged(['O', 'C', 'O'], coords), 'linear')
        self.assertNotEqual(self._unflagged(['O', 'C', 'O'], coords, is_linear=False), 'linear')

    def test_collinear_atoms_of_a_species_flagged_non_linear_give_no_kind(self):
        """Test that collinear atoms of a species flagged non linear give no kind"""
        with self.assertLogs('arc', level='WARNING'):
            kind = self._unflagged(['O', 'C', 'O'], [(-1.16, 0, 0), (0, 0, 0), (1.16, 0, 0)], is_linear=False)
        self.assertIsNone(kind)

    def test_a_distorted_geometry_is_judged_by_a_relative_tolerance_of_1e_3(self):
        """Test that a distorted geometry is judged by a relative tolerance of 1e 3"""
        self.assertEqual(self._kind(*_benzene_coords(stretch=1.0002)), 'symmetric_top')
        self.assertEqual(self._kind(*_benzene_coords(stretch=1.02)), 'asymmetric_top')

    def test_the_isotopes_the_geometry_carries_are_weighed(self):
        """Test that the isotopes the geometry carries are weighed"""
        symbols, coords = _methane_coords()
        self.assertEqual(self._kind(symbols, coords, isotopes=(12, 2, 1, 1, 1)), 'symmetric_top')
        self.assertEqual(self._kind(symbols, coords, isotopes=(12, 2, 2, 1, 1)), 'asymmetric_top')

    def test_a_missing_or_unweighable_geometry_gives_no_kind(self):
        """Test that a missing or unweighable geometry gives no kind"""
        spc = TestStatmechToDict()._make_spc(freqs=[100.0])
        spc.label = 'x'
        spc.final_xyz = None
        spc.initial_xyz = _xyz_dict(*_methane_coords())
        self.assertIsNone(_statmech_to_dict(spc, '/tmp/project')['rigid_rotor_kind'])
        spc.final_xyz = {'symbols': ('Xx', 'Xx', 'Xx'), 'isotopes': (1, 1, 1),
                         'coords': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))}
        self.assertIsNone(_statmech_to_dict(spc, '/tmp/project')['rigid_rotor_kind'])
        symbols, coords = _methane_coords()
        self.assertIsNone(self._kind(symbols, coords, isotopes=(12, 99, 1, 1, 1)))
        self.assertIsNone(self._kind(['H', 'H', 'H'], [(0, 0, 0)] * 3))

    def test_a_point_group_promotes_a_loosely_converged_symmetric_top(self):
        """Test that a point group promotes a loosely converged symmetric top"""
        for build, group in ((_ammonia_coords, 'C3v'), (_benzene_coords, 'D6h')):
            symbols, coords = build()
            loose = [(x * 1.004, y, z) for x, y, z in coords]
            with self.subTest(group=group):
                self.assertEqual(self._kind(symbols, loose), 'asymmetric_top')
                self.assertEqual(self._kind(symbols, loose, point_group=group), 'symmetric_top')
                self.assertEqual(self._kind(symbols, [(x * 1.2, y, z) for x, y, z in coords], point_group=group),
                                 'asymmetric_top')

    def test_a_cubic_point_group_promotes_a_loosely_converged_spherical_top(self):
        """Test that a cubic point group promotes a loosely converged spherical top"""
        symbols, coords = _methane_coords()
        loose = [(x, y * 1.004, z * 1.008) for x, y, z in coords]
        self.assertEqual(self._kind(symbols, loose), 'asymmetric_top')
        self.assertEqual(self._kind(symbols, loose, point_group='Td'), 'spherical_top')

    def test_a_low_symmetry_point_group_vetoes_an_accidental_degeneracy(self):
        """Test that a low symmetry point group vetoes an accidental degeneracy"""
        symbols, coords = _water_coords()
        self.assertEqual(self._kind(symbols, coords), 'asymmetric_top')
        symbols, coords = _ammonia_coords()
        for group in ('C1', 'Cs', 'C2v', 'D2h', 'S2', 'Ci'):
            with self.subTest(group=group):
                self.assertEqual(self._kind(symbols, coords), 'symmetric_top')
                self.assertEqual(self._kind(symbols, coords, point_group=group), 'asymmetric_top')
        symbols, coords = _methane_coords()
        self.assertEqual(self._kind(symbols, coords, point_group='C2v'), 'asymmetric_top')

    def test_an_s4_axis_makes_a_symmetric_top(self):
        """Allene (D2d, which contains S4) has equal moments, and neither D2d nor S4 may veto that"""
        symbols, coords = _allene_coords()
        for group in (None, 'D2d', 'S4', 'D3d', 'S6', 'D4h'):
            with self.subTest(group=group):
                self.assertEqual(self._kind(symbols, coords, point_group=group), 'symmetric_top')
        self.assertEqual(self._kind(symbols, coords, point_group='D2h'), 'asymmetric_top')

    def test_a_loosely_converged_d2d_geometry_is_promoted_by_its_point_group(self):
        """Test that a loosely converged d2d geometry is promoted by its point group"""
        symbols, coords = _allene_coords(stretch=1.03)
        self.assertEqual(self._kind(symbols, coords), 'asymmetric_top')
        self.assertEqual(self._kind(symbols, coords, point_group='D2d'), 'symmetric_top')
        self.assertEqual(self._kind(symbols, coords, point_group='S4'), 'symmetric_top')
        self.assertEqual(self._kind(symbols, coords, point_group='C2v'), 'asymmetric_top')

    def test_isotopes_that_cannot_be_compared_use_the_moments_alone(self):
        """Test that hand-edited isotopes or symbols neither raise nor enable the point group refinement"""
        for xyz in ({'symbols': ['C'], 'isotopes': 12}, {'symbols': None, 'isotopes': [12]},
                    {'symbols': ['Xx'], 'isotopes': [1]}, {'symbols': ['C'], 'isotopes': [[12]]}):
            with self.subTest(xyz=xyz):
                self.assertFalse(_has_default_isotopes(xyz))
        self.assertTrue(_has_default_isotopes({'symbols': ['C', 'H'], 'isotopes': [12, 1]}))

    def test_non_default_isotopes_use_the_moments_alone(self):
        """Test that non default isotopes use the moments alone"""
        symbols, coords = _methane_coords()
        self.assertEqual(self._kind(symbols, coords, isotopes=(12, 2, 1, 1, 1), point_group='C2v'), 'symmetric_top')
        self.assertEqual(self._kind(symbols, coords, isotopes=(12, 1, 1, 1, 1), point_group='Td'), 'spherical_top')

    def test_an_unusable_point_group_leaves_the_moments_alone(self):
        """Test that an unusable point group leaves the moments alone"""
        for group in (None, 'Dinfh', 'nonsense', 'Kh'):
            with self.subTest(group=group):
                self.assertEqual(self._kind(*_methane_coords(), point_group=group), 'spherical_top')


class TestLatentExports(unittest.TestCase):
    """The T1 diagnostic, kinetics comment and TS validation, reversibility, isotopes, ESS routes and properties."""

    gaussian_opt_log = os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')
    gaussian_freq_log = os.path.join(ARC_TESTING_PATH, 'freq', 'iC3H7.out')
    polarizability_log = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'Species',
                                      'NH2_freq.out')
    orca_log = os.path.join(ARC_TESTING_PATH, 'orca_example_opt.log')

    def _scratch(self) -> str:
        scratch = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, scratch, True)
        return scratch

    def _species_dict(self, spc=None, paths=None, converged=True, project_directory='/abs', sp_method=None) -> dict:
        spc = spc or TestSpcToDict._make_spc_mock(self)
        output_dict = {'CH4': {'convergence': converged, 'paths': paths or dict(), 'job_types': {'opt': True}}}
        if sp_method is not None:
            output_dict['CH4']['levels'] = {'sp': {'method': sp_method, 'basis': 'cc-pvtz'}}
        return _spc_to_dict(spc, output_dict, project_directory)

    def test_the_t1_diagnostic_is_the_recorded_number_and_null_otherwise(self):
        """Test that sp_t1_diagnostic is spc.t1 for a converged species at a coupled-cluster sp level only."""
        spc = TestSpcToDict._make_spc_mock(self)
        spc.t1 = 0.0123
        for method in ('ccsd(t)', 'CCSD(T)-F12', 'dlpno-ccsd(t)', 'qcisd(t)'):
            with self.subTest(method=method):
                self.assertEqual(self._species_dict(spc, sp_method=method)['sp_t1_diagnostic'], 0.0123)
        for method in ('wb97xd', 'b3lyp', 'mp2', None):
            with self.subTest(method=method):
                self.assertIsNone(self._species_dict(spc, sp_method=method)['sp_t1_diagnostic'])
        self.assertIsNone(self._species_dict(spc, converged=False, sp_method='ccsd(t)')['sp_t1_diagnostic'])
        for value in (None, float('nan'), float('inf'), True, '0.01', MagicMock()):
            with self.subTest(t1=value):
                spc.t1 = value
                self.assertIsNone(self._species_dict(spc, sp_method='ccsd(t)')['sp_t1_diagnostic'])
                self.assertIsNone(_get_t1_diagnostic(spc, {'method': 'ccsd(t)'}))

    def test_kinetics_comment_and_ts_validation_are_exported_or_null(self):
        """Test that the comment and the TS-validation marker of the kinetics dict reach the reaction record."""
        rxn = MagicMock()
        rxn.label = 'A <=> B'
        rxn.reactants, rxn.products, rxn.family, rxn.multiplicity, rxn.ts_label = ['A'], ['B'], None, 1, 'TS0'
        rxn.kinetics = {'A': (1.0, 's^-1'), 'n': 0.0, 'Ea': (1.0, 'kJ/mol'),
                        'comment': 'Fitted to 50 data points\nIRC failed', 'ts_validation': 'IRC failed'}
        kinetics = _rxn_to_dict(rxn)['kinetics']
        self.assertEqual(kinetics['comment'], 'Fitted to 50 data points\nIRC failed')
        self.assertEqual(kinetics['ts_validation'], 'IRC failed')
        rxn.kinetics = {'A': (1.0, 's^-1'), 'comment': '', 'ts_validation': 5}
        kinetics = _rxn_to_dict(rxn)['kinetics']
        self.assertIsNone(kinetics['comment'])
        self.assertIsNone(kinetics['ts_validation'])
        rxn.kinetics = {'A': (1.0, 's^-1')}
        kinetics = _rxn_to_dict(rxn)['kinetics']
        self.assertIsNone(kinetics['comment'])
        self.assertIsNone(kinetics['ts_validation'])

    def test_reversible_follows_the_reaction_arrow(self):
        """Test that '<=>' is reversible, '=>' is not, any other arrow states nothing, and ARC reactions use '<=>'."""
        rxn = ARCReaction(label='CH4 + OH <=> CH3 + H2O')
        self.assertEqual(rxn.arrow.strip(), '<=>')
        self.assertIs(_rxn_to_dict(rxn)['reversible'], True)
        for arrow, expected in ((' <=> ', True), ('=>', False), (' => ', False), ('->', None), (None, None),
                                (MagicMock(), None)):
            with self.subTest(arrow=arrow):
                self.assertIs(_get_reversible(SimpleNamespace(arrow=arrow)), expected)

    def test_xyz_isotopes_reads_the_list_the_dict_carries(self):
        """Test the isotope list of an xyz dict, and null for anything that does not state one per atom."""
        xyz = {'symbols': ('C', 'H'), 'isotopes': (13, 2), 'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 1.1))}
        self.assertEqual(_xyz_isotopes(xyz), [13, 2])
        for bad in (None, 'C 0 0 0', {'symbols': ('C',), 'coords': ((0.0, 0.0, 0.0),)},
                    {'symbols': ('C', 'H'), 'isotopes': (13,)}, {'symbols': ('C',), 'isotopes': (None,)},
                    {'symbols': ('C',), 'isotopes': (True,)}, {'symbols': ('C',), 'isotopes': (0,)},
                    {'symbols': ('C',), 'isotopes': ()}):
            with self.subTest(xyz=bad):
                self.assertIsNone(_xyz_isotopes(bad))

    def test_every_exported_geometry_states_its_isotopes(self):
        """Test the isotopes beside xyz, the conformers and the opt input, for a single-stage opt."""
        spc = TestSpcToDict._make_spc_mock(self)
        spc.final_xyz = dict(spc.final_xyz, isotopes=(13, 2, 1, 1, 1))
        spc.initial_xyz = dict(spc.final_xyz, isotopes=(12, 1, 1, 1, 1))
        spc.conformers = [dict(spc.final_xyz, isotopes=(12, 2, 2, 1, 1)), 'C 0.0 0.0 0.0']
        spc.conformer_energies = [0.0, 1.0]
        result = self._species_dict(spc)
        self.assertEqual(result['xyz_isotopes'], [13, 2, 1, 1, 1])
        self.assertEqual(result['conformers_isotopes'], [[12, 2, 2, 1, 1], None])
        self.assertEqual(len(result['conformers_isotopes']), len(result['conformers']))
        self.assertEqual(result['opt_input_xyz_isotopes'], [12, 1, 1, 1, 1])
        self.assertIsNone(result['coarse_opt_input_xyz_isotopes'])
        self.assertIsNone(result['coarse_opt_output_xyz_isotopes'])

    def test_a_geometry_without_isotopes_states_null(self):
        """Test that a geometry that carries no isotope list exports null isotopes, and no conformers no key."""
        spc = TestSpcToDict._make_spc_mock(self)
        spc.final_xyz = {key: value for key, value in spc.final_xyz.items() if key != 'isotopes'}
        spc.conformers = []
        result = self._species_dict(spc)
        self.assertIsNone(result['xyz_isotopes'])
        self.assertIsNone(result['opt_input_xyz_isotopes'])
        self.assertNotIn('conformers_isotopes', result)

    def test_a_coarse_stage_moves_the_isotopes_of_the_opt_input_with_the_geometry(self):
        """Test that, after a coarse opt, the fine opt's input isotopes are those the coarse log states."""
        spc = TestSpcToDict._make_spc_mock(self)
        spc.initial_xyz = dict(spc.final_xyz, isotopes=(12, 2, 1, 1, 1))
        parsed = {'symbols': ('C', 'H'), 'isotopes': (12, 1), 'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 1.1))}
        with patch('arc.output.parse_geometry', return_value=parsed), \
                patch('arc.output._get_log_stated_isotopes', return_value=[12, 2]):
            result = self._species_dict(spc, paths={'geo_coarse': self.gaussian_opt_log})
        self.assertEqual(result['coarse_opt_input_xyz_isotopes'], [12, 2, 1, 1, 1])
        self.assertEqual(result['coarse_opt_output_xyz_isotopes'], [12, 2])
        self.assertEqual(result['opt_input_xyz_isotopes'], [12, 2])

    def test_a_log_geometry_states_no_isotopes_the_log_does_not_state(self):
        """Test that symbol-derived isotopes of a parsed geometry are never exported for the coarse output."""
        spc = TestSpcToDict._make_spc_mock(self)
        parsed = {'symbols': ('C', 'H'), 'isotopes': (12, 1), 'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 1.1))}
        for log in (self.gaussian_opt_log, self.orca_log):
            with self.subTest(log=log), patch('arc.output.parse_geometry', return_value=parsed):
                result = self._species_dict(spc, paths={'geo_coarse': log})
                self.assertIsNotNone(result['coarse_opt_output_xyz'])
                self.assertIsNone(result['coarse_opt_output_xyz_isotopes'])
                self.assertIsNone(result['opt_input_xyz_isotopes'])

    def _mass_log(self, blocks) -> str:
        """Write a Gaussian-identified log holding the given blocks of ``(Z, mass text)`` lines and return its path."""
        scratch = self._scratch()
        log = os.path.join(scratch, 'output.out')
        with open(log, 'w') as handle:
            handle.write(' Entering Gaussian System, Link 0=g16\n')
            for block in blocks:
                for i, (atomic_number, mass) in enumerate(block, start=1):
                    handle.write(f' Atom {i:5d} has atomic number {atomic_number:2d} and mass {mass}\n')
        return log

    def test_the_isotopes_a_gaussian_log_states_are_read_from_its_mass_lines(self):
        """Test the isotopes of the mass lines, and null for other logs or atom lists."""
        symbols = ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H')
        self.assertEqual(_get_log_stated_isotopes(self.gaussian_freq_log, symbols),
                         [12, 12, 12, 1, 1, 1, 1, 1, 1, 1])
        self.assertIsNone(_get_log_stated_isotopes(self.gaussian_freq_log, symbols + ('H',)))
        self.assertIsNone(_get_log_stated_isotopes(self.gaussian_freq_log, ()))
        self.assertIsNone(_get_log_stated_isotopes(self.gaussian_opt_log, symbols))
        self.assertIsNone(_get_log_stated_isotopes(self.orca_log, symbols))
        self.assertIsNone(_get_log_stated_isotopes('/nonexistent.log', symbols))
        self.assertIsNone(_get_log_stated_isotopes(None, symbols))

    def test_the_isotopes_are_those_of_the_last_block_of_mass_lines(self):
        """Test that a log with two blocks states the last one (a readiso deuterium block)."""
        log = self._mass_log([[(6, '12.00000'), (1, '1.00783')], [(6, '12.00000'), (1, '2.01410')]])
        self.assertEqual(_get_log_stated_isotopes(log, ('C', 'H')), [12, 2])
        self.assertIsNone(_get_log_stated_isotopes(log, ('C', 'H', 'H')))

    def test_a_mass_block_that_does_not_match_the_symbols_states_nothing(self):
        """Test that a mismatched atomic number, a malformed mass and a ragged block give null and never raise."""
        log = self._mass_log([[(6, '12.00000'), (1, '1.00783')]])
        self.assertIsNone(_get_log_stated_isotopes(log, ('C', 'O')))
        self.assertIsNone(_get_log_stated_isotopes(log, ('H', 'C')))
        self.assertIsNone(_get_log_stated_isotopes(self._mass_log([[(6, '12.00000'), (1, '1.0.0')]]), ('C', 'H')))
        self.assertIsNone(_get_log_stated_isotopes(self._mass_log([[(6, '12.00000'), (1, '*****')]]), ('C', 'H')))
        self.assertIsNone(_get_log_stated_isotopes(self._mass_log([[(6, '12.00000'), (1, '1.00783')]]),
                                                   ('C', 'Xx')))

    def test_a_gaussian_route_is_read_from_the_log_in_preference_to_the_deck(self):
        """Test that the route is the log's echoed line, and the deck's only when the log holds none."""
        scratch = self._scratch()
        calc_dir = os.path.join(scratch, 'calcs', 'CH4', 'opt')
        os.makedirs(calc_dir)
        log = os.path.join(calc_dir, 'output.out')
        shutil.copyfile(self.gaussian_opt_log, log)
        expected = '#P opt=(calcfc) guess=mix uhf/3-21g IOp(2/9=2000) scf=xqc'
        self.assertEqual(_get_ess_route(log, scratch), expected)
        with open(os.path.join(calc_dir, 'input.gjf'), 'w') as handle:
            handle.write('%chk=check.chk\n#P wb97xd/def2svp opt=(tight) scf=xqc\n\nCH4\n\n0 1\nC 0.0 0.0 0.0\n')
        self.assertEqual(_get_ess_route(log, scratch), expected)
        self.assertEqual(_get_ess_route(os.path.relpath(log, scratch), scratch), expected)
        with open(log, 'w') as handle:
            handle.write(' Entering Gaussian System, Link 0=g16\n')
        self.assertEqual(_get_ess_route(log, scratch), '#P wb97xd/def2svp opt=(tight) scf=xqc')

    def test_a_gaussian_deck_route_continuation_is_joined_by_a_space(self):
        """Test that the lines of a deck route are stripped and joined by one space, unlike a log's wrapped route."""
        scratch = self._scratch()
        deck = os.path.join(scratch, 'input.gjf')
        with open(deck, 'w') as handle:
            handle.write('%chk=check.chk\n#P opt\nfreq\n\nCH4\n\n0 1\nC 0.0 0.0 0.0\n')
        self.assertEqual(_gaussian_route_text(deck), '#P opt freq')

    def test_an_orca_route_is_the_lowercase_keyword_line(self):
        """Test that an Orca job states its keyword lines as the input gave them, lowercased and without the '!'."""
        self.assertEqual(_get_ess_route(self.orca_log, '/dummy'), 'rks b3lyp/g sv(p) tightscf opt')

    def test_an_orca_route_drops_a_comment_and_keeps_keyword_lines_only(self):
        """Test that text from '#' onward on a '!' line is dropped and '%' blocks are not part of the route."""
        scratch = self._scratch()
        deck = os.path.join(scratch, 'input.in')
        with open(deck, 'w') as handle:
            handle.write('! B3LYP def2-SVP Opt # a comment\n%pal nprocs 4 end\n! TightSCF\n* xyz 0 1\n')
        self.assertEqual(_orca_route_text(deck), 'b3lyp def2-svp opt tightscf')
        with open(deck, 'w') as handle:
            handle.write('! B3LYP #c# def2-SVP\n')
        self.assertEqual(_orca_route_text(deck), 'b3lyp def2-svp')

    def test_no_route_is_stated_for_another_program_or_a_missing_log(self):
        """Test that nothing is reconstructed: other programs, missing and empty logs give null, never raising."""
        scratch = self._scratch()
        empty = os.path.join(scratch, 'empty.log')
        open(empty, 'w').close()
        molpro = os.path.join(ARC_TESTING_PATH, 'sp', 'mehylamine_CCSD(T).out')
        for log in (None, '', '/nonexistent.log', empty, molpro):
            with self.subTest(log=log):
                self.assertIsNone(_get_ess_route(log, scratch))
        with patch('arc.output._gaussian_route_text', side_effect=RuntimeError('boom')):
            self.assertIsNone(_get_ess_route(self.gaussian_opt_log, scratch))

    def test_the_species_record_carries_the_route_of_each_job(self):
        """Test the route keys of a species record: per job, from its own log, and null when unconverged."""
        paths = {'geo': self.gaussian_opt_log, 'freq': self.gaussian_freq_log, 'sp': self.orca_log,
                 'composite': '/nonexistent.log'}
        result = self._species_dict(paths=paths)
        self.assertEqual(result['opt_route'], '#P opt=(calcfc) guess=mix uhf/3-21g IOp(2/9=2000) scf=xqc')
        self.assertIn(' freq ', result['freq_route'])
        self.assertEqual(result['sp_route'], 'rks b3lyp/g sv(p) tightscf opt')
        self.assertIsNone(result['composite_route'])
        unconverged = self._species_dict(paths=paths, converged=False)
        for key in ('opt_route', 'freq_route', 'sp_route', 'composite_route'):
            self.assertIsNone(unconverged[key], key)

    def test_the_ts_record_carries_the_route_of_each_irc_job(self):
        """Test that irc_log_routes is index-aligned with irc_logs, null for a log without a route."""
        ts_spc = TestSpcToDict._make_spc_mock(self, label='TS0', is_ts=True)
        output_dict = {'TS0': {'convergence': True, 'job_types': {'opt': True},
                               'paths': {'irc': [self.gaussian_freq_log, '/nonexistent.log', self.orca_log]}}}
        result = _spc_to_dict(ts_spc, output_dict, '/abs')
        self.assertEqual(len(result['irc_log_routes']), len(result['irc_logs']))
        self.assertIn(' freq ', result['irc_log_routes'][0])
        self.assertIsNone(result['irc_log_routes'][1])
        self.assertEqual(result['irc_log_routes'][2], 'rks b3lyp/g sv(p) tightscf opt')

    def test_the_dipole_moment_its_density_and_the_polarizability_come_from_the_gaussian_logs(self):
        """Test the last Debye dipole and its density header of the opt log and the freq log's polarizability."""
        result = self._species_dict(paths={'geo': self.gaussian_opt_log, 'freq': self.polarizability_log})
        self.assertAlmostEqual(result['opt_dipole_moment_debye'], 0.2355)
        self.assertEqual(result['opt_dipole_moment_density'], 'SCF')
        expected = (6.926 + 11.044 + 9.431) / 3 * bohr_to_angstrom ** 3
        self.assertAlmostEqual(result['freq_polarizability_angstrom3'], expected)
        self.assertGreater(result['freq_polarizability_angstrom3'], 1.0)
        self.assertLess(result['freq_polarizability_angstrom3'], 2.0)

    def test_the_dipole_density_is_the_header_before_the_last_block(self):
        """Test that the density is the one named before the last dipole block, and null when no header precedes."""
        scratch = self._scratch()
        log = os.path.join(scratch, 'output.out')
        with open(log, 'w') as handle:
            handle.write(' Entering Gaussian System, Link 0=g16\n'
                         ' Population analysis using the SCF density.\n'
                         ' Dipole moment (field-independent basis, Debye):\n'
                         '    X=  0.0000    Y=  0.0000    Z=  1.0000  Tot=  1.0000\n'
                         ' Population analysis using the current density.\n'
                         ' Dipole moment (field-independent basis, Debye):\n'
                         '    X=  0.0000    Y=  0.0000    Z=  2.0000  Tot=  2.0000\n')
        self.assertEqual(_get_gaussian_dipole_moment(log, scratch), (2.0, 'current'))
        with open(log, 'w') as handle:
            handle.write(' Entering Gaussian System, Link 0=g16\n'
                         ' Dipole moment (field-independent basis, Debye):\n'
                         '    X=  0.0000    Y=  0.0000    Z=  1.0000  Tot=  1.0000\n')
        self.assertEqual(_get_gaussian_dipole_moment(log, scratch), (1.0, None))

    def test_the_dipole_density_ignores_a_header_after_the_last_block(self):
        """Test that a population-analysis header printed after the last dipole block is not its density."""
        scratch = self._scratch()
        log = os.path.join(scratch, 'output.out')
        with open(log, 'w') as handle:
            handle.write(' Entering Gaussian System, Link 0=g16\n'
                         ' Population analysis using the SCF density.\n'
                         ' Dipole moment (field-independent basis, Debye):\n'
                         '    X=  0.0000    Y=  0.0000    Z=  1.0000  Tot=  1.0000\n'
                         ' Population analysis using the current density.\n')
        self.assertEqual(_get_gaussian_dipole_moment(log, scratch), (1.0, 'SCF'))

    def test_an_unreadable_last_dipole_block_states_neither_value_nor_density(self):
        """Test that a last dipole block without a total leaves the value and its density null together."""
        scratch = self._scratch()
        log = os.path.join(scratch, 'output.out')
        with open(log, 'w') as handle:
            handle.write(' Entering Gaussian System, Link 0=g16\n'
                         ' Population analysis using the SCF density.\n'
                         ' Dipole moment (field-independent basis, Debye):\n'
                         '    X=  0.0000    Y=  0.0000    Z=  1.0000  Tot=  1.0000\n'
                         ' Population analysis using the current density.\n'
                         ' Dipole moment (field-independent basis, Debye):\n'
                         '    X=  ****    Y=  0.0000    Z=  1.0000  Tot=  ****\n')
        self.assertEqual(_get_gaussian_dipole_moment(log, scratch), (None, None))

    def _polarizability_log(self, *lines) -> tuple[str, str]:
        scratch = self._scratch()
        log = os.path.join(scratch, 'output.out')
        with open(log, 'w') as handle:
            handle.write(' Entering Gaussian System, Link 0=g16\n')
            for line in lines:
                handle.write(line + '\n')
        return log, scratch

    def test_the_polarizability_reads_fixed_width_fields_of_the_last_line_only(self):
        """Test run-together fields, an unreadable last line, an all-zero last line and the ordinary value."""
        valid = '  Exact polarizability:   6.926  -0.000  11.044  -0.000  -0.000   9.431'
        run_together = '  Exact polarizability: 566.951-100.058  42.556   0.000   0.000  10.000'
        expected = (566.951 + 42.556 + 10.0) / 3 * bohr_to_angstrom ** 3
        log, scratch = self._polarizability_log(valid, run_together)
        self.assertAlmostEqual(_get_gaussian_polarizability(log, scratch), expected)
        log, scratch = self._polarizability_log(valid, '  Exact polarizability:   6.926  abc')
        self.assertIsNone(_get_gaussian_polarizability(log, scratch))
        log, scratch = self._polarizability_log(
            valid, '  Exact polarizability:   0.000   0.000   0.000   0.000   0.000   0.000')
        self.assertIsNone(_get_gaussian_polarizability(log, scratch))
        log, scratch = self._polarizability_log(valid)
        self.assertAlmostEqual(_get_gaussian_polarizability(log, scratch),
                               (6.926 + 11.044 + 9.431) / 3 * bohr_to_angstrom ** 3)

    def test_no_dipole_moment_is_stated_for_an_ion_or_an_atom(self):
        """Test that a charged species and a monoatomic species export no dipole, and no polarizability for an atom."""
        paths = {'geo': self.gaussian_opt_log, 'freq': self.polarizability_log}
        ion = TestSpcToDict._make_spc_mock(self)
        ion.charge = 1
        result = self._species_dict(ion, paths=paths)
        self.assertIsNone(result['opt_dipole_moment_debye'])
        self.assertIsNone(result['opt_dipole_moment_density'])
        self.assertIsNotNone(result['freq_polarizability_angstrom3'])
        atom = TestSpcToDict._make_spc_mock(self, monoatomic=True)
        result = self._species_dict(atom, paths=paths)
        self.assertIsNone(result['opt_dipole_moment_debye'])
        self.assertIsNone(result['opt_dipole_moment_density'])
        self.assertIsNone(result['freq_polarizability_angstrom3'])

    def test_the_dipole_moment_and_polarizability_are_null_without_a_gaussian_value(self):
        """Test null for another program, a log that prints none, a missing log, an unconverged species."""
        result = self._species_dict(paths={'geo': self.orca_log, 'freq': self.gaussian_freq_log})
        self.assertIsNone(result['opt_dipole_moment_debye'])
        self.assertIsNone(result['opt_dipole_moment_density'])
        self.assertIsNone(result['freq_polarizability_angstrom3'])
        result = self._species_dict(paths={'geo': '/nonexistent.log', 'freq': '/nonexistent.log'})
        self.assertIsNone(result['opt_dipole_moment_debye'])
        self.assertIsNone(result['freq_polarizability_angstrom3'])
        result = self._species_dict(paths={'geo': self.gaussian_opt_log, 'freq': self.polarizability_log},
                                    converged=False)
        self.assertIsNone(result['opt_dipole_moment_debye'])
        self.assertIsNone(result['freq_polarizability_angstrom3'])
        with patch('arc.output.parse_dipole_moment', side_effect=RuntimeError('boom')):
            result = self._species_dict(paths={'geo': self.gaussian_opt_log})
        self.assertIsNone(result['opt_dipole_moment_debye'])


if __name__ == '__main__':
    unittest.main()
