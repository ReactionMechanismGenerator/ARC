"""
Tests for the arc.output module (consolidated output.yml writer).
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from arc.common import ARC_PATH
from arc.level import Level
from arc.common import ARC_TESTING_PATH
from arc.output import (
    _compute_point_groups,
    _get_arkane_git_commit,
    _get_rotor_barrier,
    _get_torsions,
    _get_ts_imag_freq,
    _level_to_dict,
    _make_rel_path,
    _parse_opt_log,
    _resolve_freq_scale_factor_source,
    _rxn_to_dict,
    _spc_to_dict,
    _statmech_to_dict,
    _thermo_to_dict,
    write_output_yml,
)
from arc.species.species import ARCSpecies, ThermoData


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


class TestGetArkaneGitCommit(unittest.TestCase):
    """Tests for _get_arkane_git_commit."""

    @patch('arc.imports.settings', {'RMG_PATH': '/fake/RMG-Py'})
    @patch('arc.output.get_git_commit', return_value=('abc1234', '2026-01-01'))
    def test_returns_hash(self, mock_git):
        result = _get_arkane_git_commit()
        self.assertEqual(result, 'abc1234')

    @patch('arc.imports.settings', {'RMG_PATH': '/fake/RMG-Py'})
    @patch('arc.output.get_git_commit', side_effect=Exception('no repo'))
    def test_returns_none_on_error(self, mock_git):
        self.assertIsNone(_get_arkane_git_commit())

    @patch('arc.imports.settings', {'RMG_PATH': '/fake/RMG-Py'})
    @patch('arc.output.get_git_commit', return_value=('', ''))
    def test_returns_none_for_empty(self, mock_git):
        self.assertIsNone(_get_arkane_git_commit())

    @patch('arc.imports.settings', {})
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
        self.assertIsNone(result['cp_data'])
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
        thermo = ThermoData(H298=-10.0, S298=200.0, Tmin=(300, 'K'), Tmax=(2000, 'K'), cp_data=cp)
        result = _thermo_to_dict(thermo)
        self.assertEqual(result['cp_data'], cp)

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

    @patch('arc.parser.parser.parse_1d_scan_energies', return_value=([0.0, 5.2, 10.1, 3.3], [0, 90, 180, 270]))
    def test_valid_barrier(self, mock_parse):
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            f.write(b'dummy scan data')
            tmp_path = f.name
        try:
            result = _get_rotor_barrier({'scan_path': tmp_path}, '/tmp')
            self.assertAlmostEqual(result, 10.1)
        finally:
            os.unlink(tmp_path)

    @patch('arc.parser.parser.parse_1d_scan_energies', side_effect=Exception('parse error'))
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
        n_steps, e_hartree = _parse_opt_log(opt_path, '/dummy')
        self.assertEqual(n_steps, 4)
        self.assertIsNotNone(e_hartree)
        self.assertAlmostEqual(e_hartree, -116.986089069, places=6)

    def test_missing_file(self):
        n_steps, e_hartree = _parse_opt_log('/nonexistent/file.log', '/tmp')
        self.assertIsNone(n_steps)
        self.assertIsNone(e_hartree)

    def test_none_path(self):
        n_steps, e_hartree = _parse_opt_log(None, '/tmp')
        self.assertIsNone(n_steps)
        self.assertIsNone(e_hartree)

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
        self.assertIsNotNone(result['zpe_hartree'])
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

    @patch('arc.job.local.execute_command')
    @patch('arc.imports.settings', {'RMG_ENV_NAME': 'rmg_env'})
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

    @patch('arc.job.local.execute_command')
    @patch('arc.imports.settings', {'RMG_ENV_NAME': 'rmg_env'})
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

    @patch('arc.job.local.execute_command', side_effect=Exception('conda not found'))
    @patch('arc.imports.settings', {'RMG_ENV_NAME': 'rmg_env'})
    def test_graceful_failure(self, mock_exec):
        species_dict = {
            'H2O': self._make_spc('H2O', ['O', 'H', 'H'], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]]),
        }
        result = _compute_point_groups(species_dict, self.tmp_dir)
        self.assertEqual(result, {})

    @patch('arc.job.local.execute_command')
    @patch('arc.imports.settings', {'RMG_ENV_NAME': 'rmg_env'})
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

    @patch('arc.job.local.execute_command')
    @patch('arc.imports.settings', {'RMG_ENV_NAME': 'rmg_env'})
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
        self.assertTrue(doc['energy_corrections_applied'])

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


if __name__ == '__main__':
    unittest.main()
