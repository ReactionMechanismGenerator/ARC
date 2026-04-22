#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for ARC's statmech.arkane module
"""

import os
import re
import shutil
import tempfile
import unittest

from arc.common import ARC_PATH, ARC_TESTING_PATH
from arc.constants import E_h_kJmol
from arc.exceptions import InputError
from arc.level import Level
from arc.reaction import ARCReaction
from arc.species import ARCSpecies
from arc.statmech.adapter import StatmechEnum
from arc.statmech.arkane import ArkaneAdapter
from arc.statmech.arkane import (
    _all_available_years,
    _available_years_for_level,
    _extract_section,
    find_best_across_files,
    _find_best_level_key_for_sp_level,
    get_qm_corrections_files,
    _level_to_str,
    _normalize_name,
    _parse_lot_params,
    _split_method_year,
    _warn_no_match,
    check_arkane_aec,
    check_arkane_bacs,
    get_arkane_model_chemistry,
)
from unittest.mock import patch


class TestEnumerationClasses(unittest.TestCase):
    """
    Contains unit tests for various enumeration classes.
    """

    def test_statmech_enum(self):
        """Test the StatmechEnum class"""
        self.assertEqual(StatmechEnum('arkane').value, 'arkane')
        with self.assertRaises(ValueError):
            StatmechEnum('wrong')


class TestArkaneAdapter(unittest.TestCase):
    """
    Contains unit tests for ArkaneAdapter.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.tmpdir = tempfile.mkdtemp(prefix='test_Arkane_')
        output_path_1 = os.path.join(cls.tmpdir, 'output_1')
        calcs_path_1 = os.path.join(cls.tmpdir, 'calcs_1')
        output_path_2 = os.path.join(cls.tmpdir, 'output_2')
        calcs_path_2 = os.path.join(cls.tmpdir, 'calcs_2')
        output_path_3 = os.path.join(cls.tmpdir, 'output_3')
        calcs_path_3 = os.path.join(cls.tmpdir, 'calcs_3')
        for path in [output_path_1, calcs_path_1, output_path_2, calcs_path_2, output_path_3, calcs_path_3]:
            if not os.path.isdir(path):
                os.makedirs(path)
        rxn_1 = ARCReaction(r_species=[ARCSpecies(label='CH3NH', smiles='C[NH]')],
                            p_species=[ARCSpecies(label='CH2NH2', smiles='[CH2]N')])
        rxn_1.ts_species = ARCSpecies(label='TS1', is_ts=True, xyz="""C      -0.68121000   -0.03232800    0.00786900
                                                                      H      -1.26057500    0.83953400   -0.27338400
                                                                      H      -1.14918300   -1.01052100   -0.06991000
                                                                      H       0.10325500    0.27373300    0.96444200
                                                                      N       0.74713700    0.12694800   -0.09842700
                                                                      H       1.16150600   -0.79827600    0.01973500""")
        cls.arkane_1 = ArkaneAdapter(output_directory=output_path_1,
                                     calcs_directory=calcs_path_1,
                                     output_dict=dict(),
                                     bac_type=None,
                                     sp_level=Level('gfn2'),
                                     freq_level=Level('gfn2'),
                                     freq_scale_factor=1.0,
                                     species=rxn_1.r_species + rxn_1.p_species + [rxn_1.ts_species],
                                     reactions=[rxn_1],
                                     )
        cls.arkane_2 = ArkaneAdapter(output_directory=output_path_2,
                                     calcs_directory=calcs_path_2,
                                     output_dict=dict(),
                                     bac_type=None,
                                     species=rxn_1.r_species[0])
        cls.ic3h7 = ARCSpecies(label='iC3H7', smiles='C[CH]C')
        cls.ic3h7.e_elect = 150.1
        opt_path = os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')
        freq_path = os.path.join(ARC_TESTING_PATH, 'freq', 'iC3H7.out')
        cls.arkane_3 = ArkaneAdapter(output_directory=output_path_3,
                                     calcs_directory=calcs_path_3,
                                     output_dict={'iC3H7': {'paths': {'freq': freq_path,
                                                                      'sp': opt_path,
                                                                      'opt': opt_path,
                                                                      'composite': '',
                                                                      }}},
                                     bac_type=None,
                                     species=[cls.ic3h7],
                                     sp_level=Level('gfn2'),
                                     )

    def test__str__(self):
        """Test the __str__ function"""
        for arkane in [self.arkane_1, self.arkane_2, self.arkane_3]:
            repr = arkane.__str__()
            self.assertIn('ArkaneAdapter(', repr)
            self.assertIn(f'output_directory={arkane.output_directory}, ', repr)
            self.assertIn(f'calcs_directory={arkane.calcs_directory}, ', repr)
            self.assertIn(f'bac_type={arkane.bac_type}, ', repr)
            self.assertIn(f'freq_scale_factor={arkane.freq_scale_factor}, ', repr)
            self.assertIn(f'species={[s.label for s in arkane.species]}, ', repr)
            if arkane.reactions is not None:
                self.assertIn(f'reactions={[r.label for r in arkane.reactions]}, ', repr)
            self.assertIn(f'T_min={arkane.T_min}, ', repr)
            self.assertIn(f'T_max={arkane.T_max}, ', repr)
            self.assertIn(f'T_count={arkane.T_count})', repr)
            if arkane.sp_level is not None:
                self.assertIn(f'sp_level={arkane.sp_level.simple()}', repr)

    def test_run_statmech_using_molecular_properties(self):
        """Test running statmech using molecular properties."""
        self.arkane_3.compute_thermo()
        plot_path = os.path.join(self.tmpdir, 'calcs_3', 'statmech', 'thermo', 'plots', 'iC3H7.pdf')
        if not os.path.isfile(plot_path):
            log_dir = os.path.dirname(os.path.dirname(plot_path))
            stdout_log = os.path.join(log_dir, 'stdout.log')
            stderr_log = os.path.join(log_dir, 'stderr.log')
            stdout_text = ''
            stderr_text = ''
            if os.path.isfile(stdout_log):
                with open(stdout_log, 'r') as f:
                    stdout_text = f.read()
            if os.path.isfile(stderr_log):
                with open(stderr_log, 'r') as f:
                    stderr_text = f.read()
            self.fail(f'Arkane did not generate {plot_path}.\nstdout.log:\n{stdout_text}\nstderr.log:\n{stderr_text}')
        self.assertTrue(os.path.isfile(plot_path))
        self.assertAlmostEqual(self.ic3h7.e0, 6.75565e+07)

    def test_level_to_str(self):
        """Test the _level_to_str function"""
        self.assertEqual(_level_to_str(Level('gfn2')),
                         "LevelOfTheory(method='gfn2',software='xtb')")
        self.assertEqual(_level_to_str(Level(method='b3lyp', basis='6-31g(d)')),
                         "LevelOfTheory(method='b3lyp',basis='631g(d)',software='gaussian')")
        self.assertEqual(_level_to_str(Level(method='CCSD(T)-F12', basis='cc-pVTZ-F12')),
                         "LevelOfTheory(method='ccsd(t)f12',basis='ccpvtzf12',software='molpro')")
        self.assertEqual(_level_to_str(Level(method='b97d3', basis='def2tzvp', software='gaussian', year=2023)),
                         "LevelOfTheory(method='b97d32023',basis='def2tzvp',software='gaussian')")

    def test_get_arkane_model_chemistry(self):
        """Test the get_arkane_model_chemistry function"""
        self.assertEqual(get_arkane_model_chemistry(sp_level=Level(method='CCSD(T)-F12', basis='cc-pVTZ-F12'),
                                                    freq_scale_factor=1.0),
                         "LevelOfTheory(method='ccsd(t)f12',basis='ccpvtzf12',software='molpro')")
        self.assertEqual(get_arkane_model_chemistry(sp_level=Level(method='CBS-QB3'),
                                                    freq_scale_factor=1.0),
                         "LevelOfTheory(method='cbsqb3',software='gaussian')")

    def test_get_arkane_model_chemistry_year_not_found(self):
        """Test warnings when a requested year is not found in the Arkane database."""
        level = Level(method='b97d3', basis='def2tzvp', software='gaussian', year=2099)
        with self.assertLogs('arc', level='WARNING') as cm:
            model_chemistry = get_arkane_model_chemistry(sp_level=level, freq_scale_factor=1.0)
        self.assertIsNone(model_chemistry)
        self.assertTrue(any('available years' in msg for msg in cm.output))

    def test_get_arkane_model_chemistry_latest_year(self):
        """Test selecting the latest available year when no year is specified."""
        model_chemistry = get_arkane_model_chemistry(sp_level=Level(method='CBS-QB3'),
                                                     freq_scale_factor=1.0)
        self.assertEqual(model_chemistry, "LevelOfTheory(method='cbsqb3',software='gaussian')")

    def test_level_helpers(self):
        """Test helper functions for method/basis/year parsing."""
        self.assertEqual(_normalize_name("DLPNO-CCSD(T)-F12"), "dlpnoccsd(t)f12")
        self.assertEqual(_normalize_name("dlpnoccsd(t)f122023"), "dlpnoccsd(t)f122023")

        base, year = _split_method_year("dlpnoccsd(t)f122023")
        self.assertEqual(base, "dlpnoccsd(t)f12")
        self.assertEqual(year, 2023)
        base, year = _split_method_year("dlpnoccsd(t)f12")
        self.assertEqual(base, "dlpnoccsd(t)f12")
        self.assertIsNone(year)

        self.assertEqual(_normalize_name("cc-pVTZ-F12"), "ccpvtzf12")
        self.assertEqual(_normalize_name("ccpvtz f12"), "ccpvtzf12")
        self.assertIsNone(_normalize_name(None))

        params = _parse_lot_params(
            "LevelOfTheory(method='dlpnoccsd(t)f122023',basis='ccpvtzf12',software='orca')"
        )
        self.assertEqual(params["method"], "dlpnoccsd(t)f122023")
        self.assertEqual(params["basis"], "ccpvtzf12")
        self.assertEqual(params["software"], "orca")

    def test_level_key_selection(self):
        """Test matching of LevelOfTheory keys by year and no-year preference."""
        section = '\n'.join([
            'atom_energies = {',
            "    \"LevelOfTheory(method='cbsqb3',software='gaussian')\": {},",
            "    \"LevelOfTheory(method='cbsqb32023',software='gaussian')\": {},",
            "}",
            "pbac = {",
        ])
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            f.write(section)
            path = f.name
        try:
            level = Level(method="CBS-QB3", software="gaussian")
            best = _find_best_level_key_for_sp_level(level, path, "atom_energies = {", "pbac = {")
            self.assertEqual(best, "LevelOfTheory(method='cbsqb3',software='gaussian')")

            level_year = Level(method="CBS-QB3", software="gaussian", year=2023)
            best_year = _find_best_level_key_for_sp_level(level_year, path, "atom_energies = {", "pbac = {")
            self.assertEqual(best_year, "LevelOfTheory(method='cbsqb32023',software='gaussian')")

            years = _available_years_for_level(level, path, "atom_energies = {", "pbac = {")
            self.assertEqual(years, [None, 2023])
        finally:
            os.remove(path)

    def test_conflicting_year_spec(self):
        """Test conflicting year in method suffix vs explicit year."""
        section = '\n'.join([
            'atom_energies = {',
            "    \"LevelOfTheory(method='b97d32023',software='gaussian')\": {},",
            "}",
            "pbac = {",
        ])
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            f.write(section)
            path = f.name
        try:
            level = Level(method="b97d32023", software="gaussian", year=2022)
            with self.assertRaises(InputError):
                _find_best_level_key_for_sp_level(level, path, "atom_energies = {", "pbac = {")
        finally:
            os.remove(path)

    def test_qm_corrections_file_path(self):
        """Test quantum corrections files are read from the RMG database path."""
        with tempfile.TemporaryDirectory() as rmg_root:
            rmg_qc = os.path.join(rmg_root, 'input', 'quantum_corrections', 'data.py')
            os.makedirs(os.path.dirname(rmg_qc), exist_ok=True)
            with open(rmg_qc, 'w') as f:
                f.write('# rmg qc\n')

            with patch('arc.statmech.arkane.RMG_DB_PATH', rmg_root):
                paths = get_qm_corrections_files()
                self.assertTrue(paths)
                self.assertEqual(paths[0], rmg_qc)

    def test_get_arkane_model_chemistry_from_qm_file(self):
        """Test reading LevelOfTheory keys from a quantum corrections file."""
        section = '\n'.join([
            'atom_energies = {',
            "    \"LevelOfTheory(method='cbsqb3',software='gaussian')\": {},",
            "}",
            "pbac = {",
        ])
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            f.write(section)
            path = f.name
        try:
            with patch('arc.statmech.arkane.get_qm_corrections_files', return_value=[path]):
                model_chemistry = get_arkane_model_chemistry(
                    sp_level=Level(method='CBS-QB3'),
                    freq_scale_factor=1.0,
                )
            self.assertEqual(model_chemistry, "LevelOfTheory(method='cbsqb3',software='gaussian')")
        finally:
            os.remove(path)

    def test_extract_section_eof(self):
        """Test _extract_section with section_end=None reads to EOF."""
        content = "header\nfreq_dict = {\n    key: val,\n}\ntrailer\n"
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".py") as f:
            f.write(content)
            path = f.name
        try:
            section = _extract_section(path, "freq_dict = {", None)
            self.assertIn("key: val", section)
            self.assertIn("trailer", section)
            # With an explicit end marker, trailer is excluded
            section_bounded = _extract_section(path, "freq_dict = {", "}")
            self.assertNotIn("trailer", section_bounded)
        finally:
            os.remove(path)

    def testfind_best_across_files(self):
        """Test multi-file search returns first match without overwriting."""
        file1_content = '\n'.join([
            'atom_energies = {',
            "    \"LevelOfTheory(method='b3lyp',basis='631g(d)',software='gaussian')\": {},",
            '}',
            'pbac = {',
        ])
        file2_content = '\n'.join([
            'atom_energies = {',
            "    \"LevelOfTheory(method='wb97xd',basis='def2tzvp',software='gaussian')\": {},",
            '}',
            'pbac = {',
        ])
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f1, \
             tempfile.NamedTemporaryFile(mode="w+", delete=False) as f2:
            f1.write(file1_content)
            f2.write(file2_content)
            path1, path2 = f1.name, f2.name
        try:
            # b3lyp is only in file1 — should be found
            level_b3 = Level(method='B3LYP', basis='6-31G(d)', software='gaussian')
            result = find_best_across_files(level_b3, [path1, path2], "atom_energies = {", "pbac = {")
            self.assertIn("b3lyp", result)
            # wb97xd is only in file2 — should still be found
            level_wb = Level(method='wB97X-D', basis='def2-TZVP', software='gaussian')
            result = find_best_across_files(level_wb, [path1, path2], "atom_energies = {", "pbac = {")
            self.assertIn("wb97xd", result)
            # imaginary method — not in either file
            level_fake = Level(method='fake', basis='fake')
            result = find_best_across_files(level_fake, [path1, path2], "atom_energies = {", "pbac = {")
            self.assertIsNone(result)
        finally:
            os.remove(path1)
            os.remove(path2)

    def test_all_available_years_aggregates(self):
        """Test _all_available_years aggregates across files."""
        file1 = '\n'.join([
            'atom_energies = {',
            "    \"LevelOfTheory(method='b97d3',basis='def2tzvp',software='gaussian')\": {},",
            '}',
            'pbac = {',
        ])
        file2 = '\n'.join([
            'atom_energies = {',
            "    \"LevelOfTheory(method='b97d32023',basis='def2tzvp',software='gaussian')\": {},",
            '}',
            'pbac = {',
        ])
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f1, \
             tempfile.NamedTemporaryFile(mode="w+", delete=False) as f2:
            f1.write(file1)
            f2.write(file2)
            path1, path2 = f1.name, f2.name
        try:
            level = Level(method='b97d3', basis='def2tzvp', software='gaussian')
            years = _all_available_years(level, [path1, path2], "atom_energies = {", "pbac = {")
            self.assertIn(None, years)
            self.assertIn(2023, years)
        finally:
            os.remove(path1)
            os.remove(path2)

    def test_warn_no_match_logs(self):
        """Test _warn_no_match emits a warning with available years."""
        file_content = '\n'.join([
            'atom_energies = {',
            "    \"LevelOfTheory(method='b97d32023',basis='def2tzvp',software='gaussian')\": {},",
            '}',
            'pbac = {',
        ])
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            f.write(file_content)
            path = f.name
        try:
            level = Level(method='b97d3', basis='def2tzvp', software='gaussian', year=2099)
            with self.assertLogs('arc', level='WARNING') as cm:
                _warn_no_match(level, [path], "atom_energies = {", "pbac = {", label="AEC")
            self.assertTrue(any('year 2099' in msg for msg in cm.output))
            self.assertTrue(any('2023' in msg for msg in cm.output))
        finally:
            os.remove(path)

    def test_generate_arkane_input(self):
        """Test generating Arkane input"""
        statmech_dir = os.path.join(ARC_TESTING_PATH, 'arkane_input_tests_delete')
        os.makedirs(statmech_dir, exist_ok=True)
        self.arkane_1.generate_arkane_input(statmech_dir=statmech_dir)
        input_path = os.path.join(statmech_dir, 'input.py')
        expected_lines = ["#!/usr/bin/env python",
                          "title = 'Arkane kinetics calculation'",
                          "        structure=SMILES('C[NH]'), spinMultiplicity=2)",
                          "        structure=SMILES('[CH2]N'), spinMultiplicity=2)",
                          "    label='CH3NH <=> CH2NH2',",
                          "    reactants=['CH3NH'],",
                          "    products=['CH2NH2'],",
                          "    transitionState='TS1',",
                          "    tunneling='Eckart',",
                          "kinetics(label='CH3NH <=> CH2NH2',",
                          "         Tmin=(300, 'K'), Tmax=(3000, 'K'), Tcount=50)",
                          ]
        with open(input_path, 'r') as f:
            lines = f.readlines()
        for expected_line in expected_lines:
            self.assertIn(expected_line + '\n', lines, f"Expected line '{expected_line}' not found in {input_path}")

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        """
        shutil.rmtree(cls.tmpdir, ignore_errors=True)
        shutil.rmtree(os.path.join(ARC_TESTING_PATH, 'arkane_input_tests_delete'), ignore_errors=True)


class TestArkaneOutputParsing(unittest.TestCase):
    """Tests for parsing functions that read Arkane output.py content."""

    def test_parse_e0(self):
        """Test parse_e0 extracts E0 from conformer blocks."""
        from arc.statmech.arkane import parse_e0
        content = """
conformer(
    label = 'CH4',
    E0 = (-88.8458, 'kJ/mol'),
    modes = [NonlinearRotor(symmetry=12)],
    spin_multiplicity = 1,
    optical_isomers = 1,
)
"""
        self.assertAlmostEqual(parse_e0('CH4', content), -88.8458)
        self.assertIsNone(parse_e0('missing_species', content))

    def test_parse_e0_positive(self):
        from arc.statmech.arkane import parse_e0
        content = "conformer(label='CHO', E0=(44.0971, 'kJ/mol'), modes=[], spin_multiplicity=2, optical_isomers=1)"
        self.assertAlmostEqual(parse_e0('CHO', content), 44.0971)

    def test_parse_conformer_statmech(self):
        """Test extraction of external_symmetry and optical_isomers."""
        from arc.statmech.arkane import _parse_conformer_statmech
        from unittest.mock import MagicMock
        content = """
conformer(
    label = 'H2O',
    E0 = (-200.0, 'kJ/mol'),
    modes = [
        NonlinearRotor(
            inertia = ([1.0, 2.0, 3.0], 'amu*angstrom^2'),
            symmetry = 2,
        ),
    ],
    spin_multiplicity = 1,
    optical_isomers = 1,
)
"""
        spc = MagicMock()
        spc.label = 'H2O'
        spc.optical_isomers = None
        spc.external_symmetry = None
        _parse_conformer_statmech(spc, content)
        self.assertEqual(spc.optical_isomers, 1)
        self.assertEqual(spc.external_symmetry, 2)

    def test_parse_conformer_statmech_linear(self):
        """Test with LinearRotor."""
        from arc.statmech.arkane import _parse_conformer_statmech
        from unittest.mock import MagicMock
        content = """
conformer(
    label = 'CO2',
    E0 = (-100.0, 'kJ/mol'),
    modes = [LinearRotor(inertia=(44.0, 'amu*angstrom^2'), symmetry=2)],
    spin_multiplicity = 1,
    optical_isomers = 1,
)
"""
        spc = MagicMock()
        spc.label = 'CO2'
        spc.optical_isomers = None
        spc.external_symmetry = None
        _parse_conformer_statmech(spc, content)
        self.assertEqual(spc.external_symmetry, 2)
        self.assertEqual(spc.optical_isomers, 1)

    def test_parse_reaction_kinetics_with_uncertainties(self):
        """Test that dA, dn, dEa, n_data_points are parsed from the comment."""
        from arc.statmech.arkane import parse_reaction_kinetics
        from unittest.mock import MagicMock
        content = """
conformer(label='TS0', E0=(50.0, 'kJ/mol'), modes=[], spin_multiplicity=2, optical_isomers=1)

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
        self.assertIsNotNone(rxn.kinetics)
        self.assertAlmostEqual(rxn.kinetics['A'][0], 1.2e10)
        self.assertAlmostEqual(rxn.kinetics['n'], 2.5)
        self.assertAlmostEqual(rxn.kinetics['Ea'][0], 45.6)
        self.assertAlmostEqual(rxn.kinetics['dA'], 1.48)
        self.assertAlmostEqual(rxn.kinetics['dn'], 0.05)
        self.assertAlmostEqual(rxn.kinetics['dEa'], 0.29)
        self.assertEqual(rxn.kinetics['dEa_units'], 'kJ/mol')
        self.assertEqual(rxn.kinetics['n_data_points'], 50)

    def test_parse_reaction_kinetics_no_comment(self):
        """Kinetics without a comment should still parse A, n, Ea."""
        from arc.statmech.arkane import parse_reaction_kinetics
        from unittest.mock import MagicMock
        content = """
conformer(label='TS0', E0=(50.0, 'kJ/mol'), modes=[], spin_multiplicity=2, optical_isomers=1)

kinetics(
    label = 'X <=> Y',
    kinetics = Arrhenius(
        A = (5.0, 's^-1'),
        n = 1.0,
        Ea = (20.0, 'kJ/mol'),
        T0 = (1, 'K'),
        Tmin = (300, 'K'),
        Tmax = (2000, 'K'),
    ),
)
"""
        rxn = MagicMock()
        rxn.label = 'X <=> Y'
        rxn.ts_species = MagicMock()
        rxn.ts_species.label = 'TS0'
        rxn.ts_species.e0 = None
        parse_reaction_kinetics(rxn, content)
        self.assertAlmostEqual(rxn.kinetics['A'][0], 5.0)
        self.assertAlmostEqual(rxn.kinetics['n'], 1.0)
        self.assertNotIn('dA', rxn.kinetics)

    def test_parse_thermo_data_block_scalars_are_float(self):
        """Verify Tmin, Tmax, H298, S298 are parsed as floats, not strings."""
        from arc.statmech.arkane import parse_thermo_data_block
        block = """
            H298 = (-108.9, 'kJ/mol'),
            S298 = (218.4, 'J/(mol*K)'),
            Tmin = (10.0, 'K'),
            Tmax = (3000.0, 'K'),
        """
        result = parse_thermo_data_block(block)
        self.assertIsInstance(result['Tmin'], float)
        self.assertIsInstance(result['Tmax'], float)
        self.assertAlmostEqual(result['Tmin'], 10.0)
        self.assertAlmostEqual(result['Tmax'], 3000.0)

    def test_find_scalar_word_boundary(self):
        """The ``n`` parameter must not match ``Tmin`` or substrings in the comment."""
        import re
        # Simulate find_scalar with word boundary
        arr_block = "A = (1.0, 's^-1'), n = 2.5, Ea = (30.0, 'kJ/mol'), Tmin = (300, 'K')"
        pat = rf"\bn\s*=\s*([-+]?[\d.eE+-]+)"
        m = re.search(pat, arr_block)
        self.assertIsNotNone(m)
        self.assertAlmostEqual(float(m.group(1)), 2.5)


class TestCheckArkaneCorrections(unittest.TestCase):
    """Tests for check_arkane_aec and check_arkane_bacs logging and matching."""

    def setUp(self):
        self._temp_files = []

    def tearDown(self):
        for path in self._temp_files:
            if os.path.exists(path):
                os.remove(path)

    def _make_data_file(self, aec_entries=None, pbac_entries=None, mbac_entries=None):
        """Create a temporary data.py with given section entries."""
        lines = ['atom_energies = {']
        for entry in (aec_entries or []):
            lines.append(f'    "{entry}": {{}},')
        lines.append('}')
        lines.append('pbac = {')
        for entry in (pbac_entries or []):
            lines.append(f'    "{entry}": {{}},')
        lines.append('}')
        lines.append('mbac = {')
        for entry in (mbac_entries or []):
            lines.append(f'    "{entry}": {{}},')
        lines.append('}')
        lines.append('freq_dict = {')
        lines.append('}')
        f = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py')
        f.write('\n'.join(lines))
        f.close()
        self._temp_files.append(f.name)
        return f.name

    def test_check_bacs_both_found_logs_info(self):
        """When both AEC and BAC match, check_arkane_bacs should log success and return True."""
        aec_key = "LevelOfTheory(method='b3lyp',basis='631g(d)',software='gaussian')"
        path = self._make_data_file(aec_entries=[aec_key], pbac_entries=[aec_key])
        level = Level(method='B3LYP', basis='6-31G(d)', software='gaussian')
        with patch('arc.statmech.arkane.get_qm_corrections_files', return_value=[path]):
            with self.assertLogs('arc', level='INFO') as cm:
                result = check_arkane_bacs(sp_level=level, bac_type='p')
        self.assertTrue(result)
        self.assertTrue(any('AEC and PBAC' in msg for msg in cm.output))

    def test_check_bacs_aec_only_logs_warning(self):
        """When AEC matches but BAC doesn't, should warn about missing BAC."""
        aec_key = "LevelOfTheory(method='dlpnoccsd(t)',basis='def2tzvp',software='orca')"
        path = self._make_data_file(aec_entries=[aec_key], pbac_entries=[])
        level = Level(method='DLPNO-CCSD(T)', basis='def2-TZVP', software='orca')
        with patch('arc.statmech.arkane.get_qm_corrections_files', return_value=[path]):
            with self.assertLogs('arc', level='WARNING') as cm:
                result = check_arkane_bacs(sp_level=level, bac_type='p')
        self.assertFalse(result)
        self.assertTrue(any('AEC' in msg and 'BAC' in msg for msg in cm.output))

    def test_check_bacs_neither_found_logs_warning(self):
        """When neither AEC nor BAC match, should warn about both missing."""
        path = self._make_data_file()
        level = Level(method='fake-method', basis='fake-basis')
        with patch('arc.statmech.arkane.get_qm_corrections_files', return_value=[path]):
            with self.assertLogs('arc', level='WARNING') as cm:
                result = check_arkane_bacs(sp_level=level, bac_type='p')
        self.assertFalse(result)
        self.assertTrue(any('AEC' in msg or 'BAC' in msg for msg in cm.output))

    def test_check_bacs_mbac_type(self):
        """When bac_type='m', should search the mbac section."""
        key = "LevelOfTheory(method='b3lyp',basis='631g(d)',software='gaussian')"
        path = self._make_data_file(aec_entries=[key], mbac_entries=[key])
        level = Level(method='B3LYP', basis='6-31G(d)', software='gaussian')
        with patch('arc.statmech.arkane.get_qm_corrections_files', return_value=[path]):
            with self.assertLogs('arc', level='INFO') as cm:
                result = check_arkane_bacs(sp_level=level, bac_type='m')
        self.assertTrue(result)
        self.assertTrue(any('MBAC' in msg for msg in cm.output))

    def test_check_aec_found_logs_info(self):
        """check_arkane_aec should log success when AEC matches."""
        aec_key = "LevelOfTheory(method='b3lyp',basis='631g(d)',software='gaussian')"
        path = self._make_data_file(aec_entries=[aec_key])
        level = Level(method='B3LYP', basis='6-31G(d)', software='gaussian')
        with patch('arc.statmech.arkane.get_qm_corrections_files', return_value=[path]):
            with self.assertLogs('arc', level='INFO') as cm:
                result = check_arkane_aec(sp_level=level)
        self.assertTrue(result)
        self.assertTrue(any('AEC' in msg and 'BAC disabled' in msg for msg in cm.output))

    def test_check_aec_not_found_logs_warning(self):
        """check_arkane_aec should warn when AEC doesn't match."""
        path = self._make_data_file()
        level = Level(method='fake-method', basis='fake-basis')
        with patch('arc.statmech.arkane.get_qm_corrections_files', return_value=[path]):
            with self.assertLogs('arc', level='WARNING') as cm:
                result = check_arkane_aec(sp_level=level)
        self.assertFalse(result)
        self.assertTrue(any('AEC' in msg for msg in cm.output))

    def test_check_bacs_different_aec_and_bac_keys(self):
        """AEC and BAC can have different LevelOfTheory keys and both should match independently."""
        aec_key = "LevelOfTheory(method='dlpnoccsd(t)2023',basis='def2tzvp',software='orca')"
        bac_key = "LevelOfTheory(method='dlpnoccsd(t)2023',basis='def2tzvp')"
        path = self._make_data_file(aec_entries=[aec_key], pbac_entries=[bac_key])
        level = Level(method='DLPNO-CCSD(T)', basis='def2-TZVP', software='orca')
        with patch('arc.statmech.arkane.get_qm_corrections_files', return_value=[path]):
            with self.assertLogs('arc', level='INFO') as cm:
                result = check_arkane_bacs(sp_level=level, bac_type='p')
        self.assertTrue(result)


class TestArkaneSpCompositeRendering(unittest.TestCase):
    """
    Phase 4: verify the Arkane species-file rendering branch for species whose
    ``e_elect_source == 'sp_composite'``. The composite total (kJ/mol) must be
    converted to Hartree and written as a bare ``energy = <float>`` assignment
    so Arkane consumes it directly (not via ``Log(...)``).
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="arkane_composite_")
        cls.opt_path = os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')
        cls.freq_path = os.path.join(ARC_TESTING_PATH, 'freq', 'iC3H7.out')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _make_adapter(self, species):
        output_dir = os.path.join(self.tmpdir, "output", species.label)
        calcs_dir = os.path.join(self.tmpdir, "calcs", species.label)
        for d in (output_dir, calcs_dir):
            os.makedirs(d, exist_ok=True)
        return ArkaneAdapter(
            output_directory=output_dir,
            calcs_directory=calcs_dir,
            output_dict={species.label: {'paths': {
                'freq': self.freq_path, 'sp': self.opt_path, 'opt': self.opt_path,
                'composite': '',
            }}},
            bac_type=None,
            species=[species],
            sp_level=Level('gfn2'),
            freq_level=Level('gfn2'),
            freq_scale_factor=1.0,
        )

    def test_composite_species_renders_explicit_numeric_energy(self):
        """``energy = <hartree_float>``, NOT ``energy = Log('...')``."""
        species = ARCSpecies(label='H2comp', smiles='[H][H]')
        species.e_elect = -200512.34  # kJ/mol (arbitrary realistic value)
        species.e_elect_source = 'sp_composite'
        adapter = self._make_adapter(species)
        species_dir = os.path.join(self.tmpdir, "species_" + species.label)
        os.makedirs(species_dir, exist_ok=True)
        adapter.generate_species_file(species, species_dir, skip_rotors=True)
        with open(species.arkane_file) as fh:
            content = fh.read()
        # Must NOT use Log() for energy; must contain a bare numeric assignment.
        self.assertNotIn("energy = Log(", content)
        expected_hartree = species.e_elect / E_h_kJmol
        # Arkane's file-format expects Hartree. Avoid an exact-string match
        # against ``str(expected_hartree)`` — Python's float repr and Mako's
        # formatting can differ in trailing digits / scientific-notation choice.
        # Extract the rendered value and compare numerically.
        match = re.search(r"^energy = ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
                          content, re.MULTILINE)
        self.assertIsNotNone(match, f"No bare numeric ``energy = …`` line:\n{content}")
        self.assertAlmostEqual(float(match.group(1)), expected_hartree, places=9)
        # Geometry / frequencies still use Log().
        self.assertIn("geometry = Log(", content)
        self.assertIn("frequencies = Log(", content)
        # The template's provenance comment mentions sp_composite.
        self.assertIn("sp_composite", content)
        self.assertIn("kJ/mol", content)

    def test_noncomposite_species_unchanged_energy_log_path(self):
        """Species without sp_composite must render the legacy ``energy = Log('sp_path')``."""
        species = ARCSpecies(label='iC3H7_legacy', smiles='C[CH]C')
        # e_elect_source stays None; e_elect is not set/relevant to legacy rendering.
        adapter = self._make_adapter(species)
        species_dir = os.path.join(self.tmpdir, "species_" + species.label)
        os.makedirs(species_dir, exist_ok=True)
        adapter.generate_species_file(species, species_dir, skip_rotors=True)
        with open(species.arkane_file) as fh:
            content = fh.read()
        self.assertIn(f"energy = Log('{self.opt_path}')", content)
        self.assertNotIn("sp_composite", content)

    def test_composite_species_with_missing_e_elect_raises(self):
        """e_elect_source='sp_composite' but no e_elect → clear error, not a silently broken file."""
        species = ARCSpecies(label='H2broken', smiles='[H][H]')
        species.e_elect = None
        species.e_elect_source = 'sp_composite'
        adapter = self._make_adapter(species)
        species_dir = os.path.join(self.tmpdir, "species_" + species.label)
        os.makedirs(species_dir, exist_ok=True)
        with self.assertRaises(ValueError) as ctx:
            adapter.generate_species_file(species, species_dir, skip_rotors=True)
        self.assertIn("sp_composite", str(ctx.exception))
        self.assertIn("e_elect is None", str(ctx.exception))

    def test_composite_rendered_energy_equals_kJmol_over_E_h_kJmol(self):
        """Round-trip: hartree written = (kJ/mol stored) / E_h_kJmol, to within fp precision."""
        species = ARCSpecies(label='H2precise', smiles='[H][H]')
        species.e_elect = -123456.789012
        species.e_elect_source = 'sp_composite'
        adapter = self._make_adapter(species)
        species_dir = os.path.join(self.tmpdir, "species_" + species.label)
        os.makedirs(species_dir, exist_ok=True)
        adapter.generate_species_file(species, species_dir, skip_rotors=True)
        with open(species.arkane_file) as fh:
            content = fh.read()
        match = re.search(r"^energy = (-?\d+\.\d+(?:e[+-]?\d+)?)$", content, re.MULTILINE)
        self.assertIsNotNone(match, f"No ``energy = <float>`` line in:\n{content}")
        rendered = float(match.group(1))
        self.assertAlmostEqual(rendered, species.e_elect / E_h_kJmol, places=9)


class TestReactionDhRxnConsumesKJmol(unittest.TestCase):
    """
    Phase 5: lock the invariant that ``set_reaction_dh_rxn`` consumes
    ``spc.e_elect`` in kJ/mol after an sp_composite finalization. The Hartree
    conversion happens *only* at the Arkane species-file rendering boundary
    (``generate_species_file``); everywhere else — including reaction ΔH and
    reaction energetics — ``spc.e_elect`` stays in kJ/mol.
    """

    def test_dh_rxn_uses_kJmol_e_elect_for_composite_species(self):
        tmpdir = tempfile.mkdtemp(prefix="arkane_dhrxn_")
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        # Two composite-finalized "species" standing in as reactant + product.
        reactant = ARCSpecies(label='R', smiles='[H][H]')
        reactant.e_elect = -200000.0            # kJ/mol
        reactant.e_elect_source = 'sp_composite'
        reactant.thermo = None                  # force the e_elect branch
        product = ARCSpecies(label='P', smiles='[H][H]')
        product.e_elect = -200100.0             # kJ/mol
        product.e_elect_source = 'sp_composite'
        product.thermo = None
        rxn = ARCReaction(r_species=[reactant], p_species=[product])
        rxn.thermo = None
        adapter = ArkaneAdapter(
            output_directory=tmpdir,
            calcs_directory=tmpdir,
            output_dict={},
            bac_type=None,
            species=[reactant, product],
            reactions=[rxn],
            sp_level=Level('gfn2'),
            freq_level=Level('gfn2'),
            freq_scale_factor=1.0,
        )
        adapter.set_reaction_dh_rxn(estimate_dh_rxn=True)
        # dh_rxn298 = (product.e_elect - reactant.e_elect) * 1e3 (J/mol conversion).
        expected_J = (product.e_elect - reactant.e_elect) * 1e3
        self.assertAlmostEqual(rxn.dh_rxn298, expected_J, places=6)
        # Sanity: the raw kJ/mol difference is -100; dh_rxn298 should be -1e5 J/mol.
        self.assertAlmostEqual(rxn.dh_rxn298, -1e5, places=6)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
