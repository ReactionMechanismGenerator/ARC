#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for ARC's statmech.arkane module
"""

import hashlib
import importlib
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from arc.checks.common import TS_IRC_FAILED_MARKER
from arc.common import ARC_PATH, ARC_TESTING_PATH, read_yaml_file, save_yaml_file
from arc.exceptions import InputError
from arc.level import Level
from arc.reaction import ARCReaction
from arc.species import ARCSpecies
from arc.statmech.adapter import StatmechEnum
from arc.statmech.arkane import ARKANE_STANDARD_STATE_PRESSURE_PA, ArkaneAdapter
from arc.statmech.arkane import (
    AEC_SECTION_END,
    AEC_SECTION_START,
    FREQ_SECTION_START,
    PBAC_SECTION_END,
    PBAC_SECTION_START,
    _all_available_years,
    _available_years_for_level,
    _effective_method,
    _extract_section,
    _match_aec_yml_key,
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
    get_arkane_treatment,
    get_file_sha256,
    normalized_method_and_basis,
    parse_e0,
    parse_reaction_kinetics,
    parse_thermo_data_block,
    run_arkane,
    _classify_arkane_stderr,
    _parse_conformer_statmech,
    _summarize_arkane_stderr,
)


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
        self.assertIsNotNone(self.ic3h7.thermo.H298)
        self.assertIs(self.ic3h7.thermo.atom_corrections_applied, True)
        self.assertIs(self.ic3h7.thermo.bond_corrections_applied, False)
        self.assertEqual(self.ic3h7.thermo.atom_corrections_level, Level('gfn2'))

    def test_parse_arkane_thermo_output_recovers_missing_thermo_container(self):
        """
        Test that ``parse_arkane_thermo_output`` does not crash when a species reaches the
        result-assignment loop with ``spc.thermo`` set to ``None`` (rather than the
        ``ARCSpecies.__init__`` default of a ``ThermoData`` instance), and that a well-formed
        sibling species (with the default ``ThermoData``) is populated correctly regardless.

        This is a regression test for a real campaign crash: ``arc/statmech/arkane.py`` assumed
        ``spc.thermo`` is never ``None`` at this point, even though other call sites in the
        codebase (``ArkaneAdapter.set_reaction_dh_rxn``, ``processor.process_arc_project``,
        ``output.get_species_output_dict``) already guard against exactly that.
        """
        tmpdir = tempfile.mkdtemp(prefix='test_Arkane_thermo_none_')
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        output_dir = os.path.join(tmpdir, 'output')
        calcs_dir = os.path.join(tmpdir, 'calcs')
        statmech_dir = os.path.join(calcs_dir, 'statmech', 'thermo')
        os.makedirs(statmech_dir)
        with open(os.path.join(statmech_dir, 'output.py'), 'w') as f:
            f.write('')
        content = {
            'none_thermo': {'H298': 1.0, 'S298': 2.0, 'data': 'NASA()'},
            'ok_thermo': {'H298': 3.0, 'S298': 4.0, 'data': 'NASA()'},
        }
        save_yaml_file(path=os.path.join(statmech_dir, 'thermo.yaml'), content=content)

        spc_none = ARCSpecies(label='none_thermo', smiles='O')
        spc_none.thermo = None
        spc_ok = ARCSpecies(label='ok_thermo', smiles='C')
        arkane = ArkaneAdapter(output_directory=output_dir,
                               calcs_directory=calcs_dir,
                               output_dict=dict(),
                               bac_type=None,
                               species=[spc_none, spc_ok])

        with patch('arc.statmech.arkane.execute_command', return_value=('', '')):
            arkane.parse_arkane_thermo_output(statmech_dir)

        self.assertIsNotNone(spc_none.thermo)
        self.assertEqual(spc_none.thermo.H298, 1.0)
        self.assertEqual(spc_none.thermo.S298, 2.0)
        self.assertIsNotNone(spc_ok.thermo)
        self.assertEqual(spc_ok.thermo.H298, 3.0)
        self.assertEqual(spc_ok.thermo.S298, 4.0)

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

    def test_lone_pair_species_uses_adjacency_list(self):
        """A lone-pair singlet (singlet carbene [CH2], mult 1) whose SMILES round-trips to a different
        multiplicity must be written to Arkane as an adjacency list, not SMILES + spinMultiplicity —
        the latter re-perceives [CH2] as a u2 biradical, violating Hund's rule when Arkane saves the
        thermo library. Normal species must still use SMILES."""
        ch2 = ARCSpecies(label='R2', smiles='[CH2]', multiplicity=1)
        h2o = ARCSpecies(label='P1', smiles='O', multiplicity=1)
        adapter = ArkaneAdapter(output_directory=self.tmpdir, calcs_directory=self.tmpdir,
                                output_dict={}, species=[ch2, h2o], sp_level=Level('gfn2'))
        content = adapter.render_arkane_input_template(statmech_dir=self.tmpdir)
        self.assertIn('adjacencyList(', content)
        self.assertIn('1 C u0 p1', content)              # correct singlet-carbene adjlist
        self.assertNotIn("SMILES('[CH2]')", content)     # must NOT use the lossy SMILES
        self.assertIn("structure=SMILES('O')", content)  # normal species unchanged

    def test_failed_kinetics_run_is_not_retried_without_tunneling(self):
        """An invalid Eckart barrier must not be accepted through an untunneled retry."""
        with tempfile.TemporaryDirectory() as statmech_dir:
            with open(os.path.join(statmech_dir, 'stderr.log'), 'w') as f:
                f.write('One or both of the barrier heights encountered in Eckart method are invalid')
            with patch('arc.statmech.arkane.create_statmech_dir', return_value=statmech_dir), \
                    patch.object(self.arkane_1, 'generate_arkane_input'), \
                    patch.object(self.arkane_1, 'generate_species_files'), \
                    patch.object(self.arkane_1, 'generate_ts_files'), \
                    patch('arc.statmech.arkane.run_arkane', return_value=True) as mock_run, \
                    patch.object(self.arkane_1, 'parse_arkane_kinetics_output') as mock_parse:
                self.arkane_1.compute_high_p_rate_coefficient(require_ts_convergence=False)

        mock_run.assert_called_once_with(statmech_dir)
        mock_parse.assert_called_once_with(statmech_dir)

    def test_kinetics_input_declares_reaction_species_without_compute_thermo(self):
        """A reactant/product carrying ``compute_thermo=False`` must still be declared as a
        ``species(...)`` line in a rendered kinetics input. Otherwise the ``reaction(...)`` block
        references a species Arkane never declared, and Arkane raises ``KeyError`` on the label
        (``arkane/input.py`` builds ``reactants`` from the declared-species dict). A species that no
        reaction names must still be excluded by ``compute_thermo=False``."""
        rxn = ARCReaction(r_species=[ARCSpecies(label='R', smiles='C[NH]', compute_thermo=False)],
                          p_species=[ARCSpecies(label='P', smiles='[CH2]N', compute_thermo=False)])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True)
        spc_x = ARCSpecies(label='X', smiles='O', compute_thermo=False)
        adapter = ArkaneAdapter(output_directory=self.tmpdir, calcs_directory=self.tmpdir,
                                output_dict=dict(), sp_level=Level('gfn2'),
                                species=rxn.r_species + rxn.p_species + [rxn.ts_species, spc_x],
                                reactions=[rxn])
        content = adapter.render_arkane_input_template(statmech_dir=self.tmpdir)
        self.assertIn("species('R',", content)
        self.assertIn("species('P',", content)
        self.assertNotIn("species('X',", content)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        """
        shutil.rmtree(cls.tmpdir, ignore_errors=True)
        shutil.rmtree(os.path.join(ARC_TESTING_PATH, 'arkane_input_tests_delete'), ignore_errors=True)


class TestDispersionAndSolvationMatching(unittest.TestCase):
    """
    Contains unit tests for matching a level's separate ``dispersion`` and ``solvation_method`` fields against
    Arkane's quantum corrections database and ARC's data/AEC.yml.
    """

    @staticmethod
    def _write_qm_file(keys):
        """Write a minimal quantum corrections data.py whose AEC section holds ``keys``, in order."""
        content = '\n'.join(['atom_energies = {']
                            + [f'    "{key}": {{}},' for key in keys]
                            + ['}', 'pbac = {'])
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py') as f:
            f.write(content)
        return f.name

    @staticmethod
    def _write_qm_sections(key):
        """Write a quantum corrections data.py whose atom energy, Petersson and Melius sections each hold ``key``."""
        sections = ('atom_energies = {', 'pbac = {', 'mbac = {')
        content = '\n'.join(line for start in sections for line in (start, f'    "{key}": {{}},', '}'))
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py') as f:
            f.write(content + '\nfreq_dict = {\n}\n')
        return f.name

    def _best(self, level, keys):
        """The AEC key matched for ``level`` among ``keys``."""
        path = self._write_qm_file(keys)
        self.addCleanup(os.remove, path)
        return _find_best_level_key_for_sp_level(level, path, 'atom_energies = {', 'pbac = {')

    def test_effective_method(self):
        """The dispersion is folded into the method in one spelling, wherever the level carries it."""
        self.assertEqual(_effective_method('b3lyp', 'gd3bj'), ('b3lypd3bj', None))
        self.assertEqual(_effective_method('B3LYP-D3(BJ)'), ('b3lypd3bj', None))
        self.assertEqual(_effective_method('b3lyp', 'EmpiricalDispersion=GD3BJ'), ('b3lypd3bj', None))
        self.assertEqual(_effective_method('b2plypd32023'), ('b2plypd3', 2023))
        self.assertEqual(_effective_method('wb97xd2023'), ('wb97xd', 2023))
        self.assertEqual(_effective_method('wb97xd3'), ('wb97xd3', None))
        self.assertEqual(_effective_method('dlpnoccsd(t)f122023'), ('dlpnoccsd(t)f12', 2023))
        self.assertEqual(_effective_method(None, 'gd3bj'), (None, None))
        self.assertEqual(_effective_method('b2plyp-gd3'), ('b2plypd3', None))
        self.assertEqual(_effective_method('b97', 'gd3'), ('b97 + d3', None))
        self.assertEqual(_effective_method('b97-gd3'), ('b97 + d3', None))
        self.assertEqual(_effective_method('wb97x', 'gd3'), ('wb97x + d3', None))
        self.assertEqual(_effective_method('wB97X-GD3'), ('wb97x + d3', None))
        self.assertEqual(_effective_method('b97-d3'), ('b97d3', None))
        self.assertEqual(_effective_method('b97d32023'), ('b97d3', 2023))
        self.assertEqual(_effective_method('wb97x-d3'), ('wb97xd3', None))

    def test_a_dispersion_field_does_not_match_the_plain_method(self):
        """b3lyp/def2tzvp with dispersion gd3bj and plain b3lyp/def2tzvp each match only their own key."""
        plain = Level(method='b3lyp', basis='def2tzvp', software='gaussian')
        d3bj = Level(method='b3lyp', basis='def2tzvp', software='gaussian', dispersion='gd3bj')
        plain_key = "LevelOfTheory(method='b3lyp2023',basis='def2tzvp',software='gaussian')"
        d3bj_key = "LevelOfTheory(method='b3lypd3bj2023',basis='def2tzvp',software='gaussian')"
        self.assertEqual(self._best(plain, [d3bj_key, plain_key]), plain_key)
        self.assertEqual(self._best(d3bj, [plain_key, d3bj_key]), d3bj_key)
        self.assertIsNone(self._best(plain, [d3bj_key]))
        self.assertIsNone(self._best(d3bj, [plain_key]))
        path = self._write_qm_sections(plain_key)
        self.addCleanup(os.remove, path)
        with patch('arc.statmech.arkane.get_qm_corrections_files', return_value=[path]):
            for start, end in ((AEC_SECTION_START, AEC_SECTION_END), (PBAC_SECTION_START, PBAC_SECTION_END)):
                with self.subTest(section=start):
                    self.assertEqual(find_best_across_files(plain, [path], start, end), plain_key)
                    self.assertIsNone(find_best_across_files(d3bj, [path], start, end))
            self.assertIsNone(get_arkane_model_chemistry(sp_level=d3bj, freq_scale_factor=1.0))
            self.assertFalse(check_arkane_bacs(sp_level=d3bj, bac_type='p'))
            with self.assertRaises(ValueError):
                check_arkane_aec(sp_level=d3bj, raise_error=True)

    def test_a_dispersion_field_matches_a_key_with_that_dispersion(self):
        """b2plyp with dispersion gd3 is Arkane's b2plypd3 for the AEC and the BAC entries. Frequency keys are
        matched as before, which reads only the method string."""
        qm_corr_files = get_qm_corrections_files()
        expected = "LevelOfTheory(method='b2plypd32023',basis='def2tzvp',software='gaussian')"
        freq_key = "LevelOfTheory(method='b2plypd3',basis='def2tzvp')"
        for level, expected_freq_key in (
                (Level(method='b2plyp', basis='def2tzvp', software='gaussian', dispersion='gd3'), None),
                (Level(method='b2plyp-gd3', basis='def2tzvp', software='gaussian'), None),
                (Level(method='b2plypd3', basis='def2tzvp', software='gaussian'), freq_key),
                (Level(method='b2plyp-d3', basis='def2-tzvp', software='gaussian'), freq_key)):
            with self.subTest(level=str(level)):
                self.assertEqual(find_best_across_files(level, qm_corr_files, AEC_SECTION_START, AEC_SECTION_END),
                                 expected)
                self.assertEqual(find_best_across_files(level, qm_corr_files, PBAC_SECTION_START,
                                                        PBAC_SECTION_END), expected)
                self.assertEqual(find_best_across_files(level, qm_corr_files, FREQ_SECTION_START, None),
                                 expected_freq_key)
                self.assertEqual(get_arkane_model_chemistry(sp_level=level, freq_scale_factor=1.0), expected)

    def test_a_dispersion_added_to_a_separately_parametrized_functional_matches_nothing(self):
        """B97 + D3 is not Grimme's B97-D3, and wB97X + D3 is not wB97X-D3: the separate field and Gaussian's
        g-prefixed spelling match none of Arkane's energy-correction keys, while the functionals' own names still
        match as they did before."""
        qm_corr_files = get_qm_corrections_files()
        for level in (Level(method='b97', basis='def2tzvp', software='gaussian', dispersion='gd3'),
                      Level(method='b97-gd3', basis='def2tzvp', software='gaussian'),
                      Level(method='wb97x', basis='def2tzvp', software='qchem', dispersion='gd3'),
                      Level(method='wb97x-gd3', basis='def2tzvp', software='qchem'),
                      Level(method='b97', basis='def2msvp', software='qchem', dispersion='d3')):
            for start, end in ((AEC_SECTION_START, AEC_SECTION_END), (PBAC_SECTION_START, PBAC_SECTION_END)):
                with self.subTest(level=str(level), section=start):
                    self.assertIsNone(find_best_across_files(level, qm_corr_files, start, end))
        for level, key in ((Level(method='b97-d3', basis='def2tzvp', software='gaussian'),
                            "LevelOfTheory(method='b97d32023',basis='def2tzvp',software='gaussian')"),
                           (Level(method='b97d3', basis='def2msvp', software='qchem'),
                            "LevelOfTheory(method='b97d3',basis='def2msvp',software='qchem')"),
                           (Level(method='wb97xd3', basis='def2tzvp', software='qchem'),
                            "LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='qchem')"),
                           (Level(method='wb97x-d3', basis='def2-tzvp', software='qchem'),
                            "LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='qchem')")):
            with self.subTest(level=str(level)):
                self.assertEqual(find_best_across_files(level, qm_corr_files, AEC_SECTION_START, AEC_SECTION_END),
                                 key)

    def test_frequency_keys_are_matched_as_before(self):
        """The dispersion field and the solvation method are not read for frequency keys: with no scale factor
        set, a missing frequency key would drop the whole model chemistry and the sp level's corrections."""
        qm_corr_files = get_qm_corrections_files()
        for level, key in ((Level(method='b3lyp', basis='6-31g(d,p)', software='gaussian', dispersion='gd3bj'),
                            "LevelOfTheory(method='b3lyp',basis='631g(d,p)')"),
                           (Level(method='wb97xd', basis='def2tzvp', software='gaussian',
                                  solvation_method='smd', solvent='water'),
                            "LevelOfTheory(method='wb97xd',basis='def2tzvp',software='gaussian')")):
            with self.subTest(level=str(level)):
                self.assertEqual(find_best_across_files(level, qm_corr_files, FREQ_SECTION_START, None), key)

    def test_a_gas_phase_sp_level_with_a_solvated_freq_level_keeps_its_corrections(self):
        """With no frequency scale factor, a gas-phase sp level and an SMD frequency level still get the composite
        model chemistry, and so the sp level's atom energy corrections, as before."""
        model_chemistry = get_arkane_model_chemistry(
            sp_level=Level(method='wb97xd', basis='def2tzvp', software='gaussian'),
            freq_level=Level(method='wb97xd', basis='def2tzvp', software='gaussian',
                             solvation_method='smd', solvent='water'),
            freq_scale_factor=None)
        self.assertEqual(model_chemistry,
                         "CompositeLevelOfTheory(\n"
                         "    freq=LevelOfTheory(method='wb97xd',basis='def2tzvp',software='gaussian'),\n"
                         "    energy=LevelOfTheory(method='wb97xd2023',basis='def2tzvp',software='gaussian')\n"
                         ")")

    def test_string_and_field_dispersion_forms_match_alike(self):
        """The dispersion in the method string or the separate field selects the same key, and only that key."""
        keys = ["LevelOfTheory(method='b3lypd3bj',basis='def2tzvp',software='gaussian')",
                "LevelOfTheory(method='b3lyp',basis='def2tzvp',software='gaussian')"]
        for level in (Level(method='b3lyp', basis='def2tzvp', software='gaussian', dispersion='gd3bj'),
                      Level(method='b3lyp-d3bj', basis='def2tzvp', software='gaussian'),
                      Level(method='b3lyp-d3(bj)', basis='def2tzvp', software='gaussian'),
                      Level(method='b3lyp', basis='def2tzvp', software='gaussian',
                            dispersion='empiricaldispersion=gd3bj')):
            with self.subTest(level=str(level)):
                self.assertEqual(self._best(level, keys), keys[0])
        self.assertEqual(self._best(Level(method='b3lyp', basis='def2tzvp', software='gaussian'), keys), keys[1])
        self.assertIsNone(self._best(Level(method='b3lyp', basis='def2tzvp', software='gaussian', dispersion='gd3'),
                                     keys))

    def test_wb97xd_and_wb97xd3_stay_distinct(self):
        """wB97X-D (built-in D2) and wB97X-D3 are two functionals."""
        keys = ["LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='qchem')",
                "LevelOfTheory(method='wb97xd',basis='def2tzvp',software='qchem')"]
        self.assertEqual(self._best(Level(method='wb97xd', basis='def2tzvp', software='qchem'), keys), keys[1])
        self.assertEqual(self._best(Level(method='wb97xd3', basis='def2tzvp', software='qchem'), keys), keys[0])

    def test_a_solvated_level_matches_no_key(self):
        """Arkane has only gas-phase corrections, so a solvated level has none, and says why."""
        qm_corr_files = get_qm_corrections_files()
        solvated = Level(method='wb97xd', basis='def2tzvp', software='gaussian',
                         solvation_method='smd', solvent='water')
        for start, end in ((AEC_SECTION_START, AEC_SECTION_END), (PBAC_SECTION_START, PBAC_SECTION_END)):
            with self.subTest(section=start):
                self.assertIsNone(find_best_across_files(solvated, qm_corr_files, start, end))
                self.assertEqual(_all_available_years(solvated, qm_corr_files, start, end), [])
        with self.assertLogs('arc', level='WARNING') as cm:
            self.assertIsNone(get_arkane_model_chemistry(sp_level=solvated, freq_scale_factor=1.0))
        self.assertTrue(any('solvation method smd' in msg for msg in cm.output))
        with self.assertRaises(ValueError) as error:
            check_arkane_aec(sp_level=solvated, raise_error=True)
        for remedy in ('solvated', 'compute_thermo', 'gas-phase arkane_level_of_theory', 'data/AEC.yml'):
            self.assertIn(remedy, str(error.exception))
        with self.assertRaises(ValueError):
            check_arkane_bacs(sp_level=solvated, bac_type='p', raise_error=True)
        self.assertFalse(check_arkane_bacs(sp_level=solvated, bac_type='p'))

    def test_common_levels_match_as_before(self):
        """Levels with no dispersion field and no solvation keep their keys."""
        qm_corr_files = get_qm_corrections_files()
        expected = [
            (Level(method='wb97xd', basis='def2tzvp', software='gaussian'),
             "LevelOfTheory(method='wb97xd2023',basis='def2tzvp',software='gaussian')"),
            (Level(method='CBS-QB3'), "LevelOfTheory(method='cbsqb3',software='gaussian')"),
            (Level(method='dlpno-ccsd(t)-f12', basis='cc-pvtz-f12', software='orca'),
             "LevelOfTheory(method='dlpnoccsd(t)f122023',basis='ccpvtzf12',software='orca')"),
            (Level(method='DLPNO-CCSD(T)', basis='def2-TZVP', software='orca'),
             "LevelOfTheory(method='dlpnoccsd(t)2023',basis='def2tzvp',software='orca')"),
            (Level(method='b97d3', basis='def2tzvp', software='gaussian'),
             "LevelOfTheory(method='b97d32023',basis='def2tzvp',software='gaussian')"),
            (Level(method='b3lyp', basis='6-31g(d,p)', software='gaussian'),
             "LevelOfTheory(method='b3lyp',basis='631g(d,p)',software='gaussian')"),
        ]
        for level, key in expected:
            with self.subTest(level=str(level)):
                self.assertEqual(find_best_across_files(level, qm_corr_files, AEC_SECTION_START, AEC_SECTION_END),
                                 key)
        self.assertEqual(find_best_across_files(Level(method='wb97xd', basis='def2tzvp', software='gaussian'),
                                                qm_corr_files, FREQ_SECTION_START, None),
                         "LevelOfTheory(method='wb97xd',basis='def2tzvp',software='gaussian')")

    def test_aec_yml_lookup(self):
        """ARC's data/AEC.yml is matched like Arkane's keys: dispersion folded in, and no solvated match."""
        aec_dict = read_yaml_file(os.path.join(ARC_PATH, 'data', 'AEC.yml'))
        self.assertEqual(_match_aec_yml_key(Level('gfn2'), aec_dict), 'gfn2')
        self.assertIsNone(_match_aec_yml_key(Level(method='gfn2', solvation_method='alpb', solvent='water'),
                                             aec_dict))
        self.assertIsNone(_match_aec_yml_key(Level(method='wb97xd', basis='def2tzvp'), aec_dict))
        aec_dict = {'b3lyp-d3bj/def2-tzvp': {'H': 1}, 'b3lyp/def2tzvp': {'H': 2}, 'b3lyp/def2tzvp (2023)': {'H': 3}}
        for level in (Level(method='b3lyp', basis='def2tzvp', dispersion='gd3bj'),
                      Level(method='b3lyp-d3bj', basis='def2tzvp'),
                      Level(method='B3LYP-D3(BJ)', basis='def2-TZVP')):
            with self.subTest(level=str(level)):
                self.assertEqual(_match_aec_yml_key(level, aec_dict), 'b3lyp-d3bj/def2-tzvp')
        self.assertEqual(_match_aec_yml_key(Level(method='b3lyp', basis='def2tzvp'), aec_dict), 'b3lyp/def2tzvp')
        self.assertEqual(_match_aec_yml_key(Level(method='b3lyp', basis='def2tzvp', year=2023), aec_dict),
                         'b3lyp/def2tzvp (2023)')
        self.assertIsNone(_match_aec_yml_key(Level(method='b3lyp', basis='def2tzvp', dispersion='gd3'), aec_dict))
        self.assertIsNone(_match_aec_yml_key(Level(method='b3lyp', basis='def2tzvp', solvation_method='smd',
                                                   solvent='water'), aec_dict))


class TestArkaneCorrectionFlags(unittest.TestCase):
    """
    Contains unit tests for the energy-correction switches ArkaneAdapter records on the thermo it parses.
    """

    def _compute_thermo_with_mocked_arkane(self, sp_level, bac_type, species=None,
                                           freq_level=None, freq_scale_factor=1.0,
                                           output_content='', e0_only=False):
        """
        Run ``compute_thermo`` with the Arkane and RMG subprocesses mocked out.

        The mocked Arkane run writes the ``thermo.yaml`` that ``save_arkane_thermo.py`` would, so the
        rendered ``input.py`` and the parsed thermo come from the same ``compute_thermo`` call.

        Args:
            species (list, optional): The species to run; a single CH4 by default.
            freq_level (Level, optional): Defaults to ``sp_level``.
            output_content (str, optional): The text of the ``output.py`` the mocked Arkane run writes.
            e0_only (bool, optional): Whether to run the E0-only mode.

        Returns:
            tuple: The first species and the text of the ``input.py`` Arkane was handed.
        """
        tmpdir = tempfile.mkdtemp(prefix='test_Arkane_corrections_')
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        species = species or [ARCSpecies(label='CH4', smiles='C')]
        arkane = ArkaneAdapter(output_directory=os.path.join(tmpdir, 'output'),
                               calcs_directory=os.path.join(tmpdir, 'calcs'),
                               output_dict=dict(),
                               bac_type=bac_type,
                               species=species,
                               sp_level=sp_level,
                               freq_level=freq_level or sp_level,
                               freq_scale_factor=freq_scale_factor)
        inputs = list()

        def fake_run_arkane(statmech_dir):
            with open(os.path.join(statmech_dir, 'input.py'), 'r') as f:
                inputs.append(f.read())
            with open(os.path.join(statmech_dir, 'output.py'), 'w') as f:
                f.write(output_content)
            save_yaml_file(path=os.path.join(statmech_dir, 'thermo.yaml'),
                           content={spc.label: {'H298': -74.6, 'S298': 186.3, 'data': 'NASA()'} for spc in species})
            return True

        with patch('arc.statmech.arkane.run_arkane', side_effect=fake_run_arkane), \
                patch.object(ArkaneAdapter, 'generate_species_files'), \
                patch('arc.statmech.arkane.execute_command', return_value=('', '')):
            arkane.compute_thermo(e0_only=e0_only)
        self.assertEqual(len(inputs), 1)
        return species[0], inputs[0]

    def test_thermo_records_arkane_native_atom_corrections(self):
        """A level Arkane's database has AEC for: the thermo records AEC, the requested BAC, and the
        sp level as the level whose atom energies were subtracted."""
        sp_level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        spc, input_py = self._compute_thermo_with_mocked_arkane(sp_level=sp_level, bac_type='p')
        self.assertIn('modelChemistry = ', input_py)
        self.assertNotIn('atomEnergies = ', input_py)
        self.assertIn('useAtomCorrections = True\n', input_py)
        self.assertIn('useBondCorrections = True\n', input_py)
        self.assertEqual(spc.thermo.H298, -74.6)
        self.assertIs(spc.thermo.atom_corrections_applied, True)
        self.assertIs(spc.thermo.bond_corrections_applied, True)
        self.assertEqual(spc.thermo.atom_corrections_level, sp_level)

    def test_thermo_records_a_dummy_arkane_level(self):
        """The examples/Stationary/bde setup: energies at apfd/def2svp, which Arkane has no AEC for, and a dummy
        arkane_level_of_theory of bmk/cbsb7, which the processor hands the adapter as its sp level. Arkane is
        told to apply AEC, so the flag is true, and the level recorded is the dummy one a consumer must compare
        with the apfd/def2svp energy level."""
        dummy = Level(method='bmk', basis='cbsb7', software='gaussian')
        spc, input_py = self._compute_thermo_with_mocked_arkane(
            sp_level=dummy, bac_type=None, freq_level=Level(method='apfd', basis='def2svp', software='gaussian'))
        self.assertIn("modelChemistry = LevelOfTheory(method='bmk'", input_py)
        self.assertIn('useAtomCorrections = True\n', input_py)
        self.assertIs(spc.thermo.atom_corrections_applied, True)
        self.assertIs(spc.thermo.bond_corrections_applied, False)
        self.assertEqual(spc.thermo.atom_corrections_level, dummy)
        self.assertNotEqual(spc.thermo.atom_corrections_level.method, 'apfd')

    def test_thermo_records_arc_aec_yml_atom_corrections(self):
        """A level only ARC's data/AEC.yml has atom energies for: AEC is applied although no
        Arkane key matched, which is the case the ``energy_corrections`` list cannot show."""
        spc, input_py = self._compute_thermo_with_mocked_arkane(sp_level=Level('gfn2'), bac_type=None)
        self.assertNotIn('modelChemistry = ', input_py)
        self.assertIn('atomEnergies = ', input_py)
        self.assertIn('useAtomCorrections = True\n', input_py)
        self.assertIn('useBondCorrections = False\n', input_py)
        self.assertIs(spc.thermo.atom_corrections_applied, True)
        self.assertIs(spc.thermo.bond_corrections_applied, False)
        self.assertEqual(spc.thermo.atom_corrections_level, Level('gfn2'))

    def test_thermo_records_the_no_atom_corrections_fallback(self):
        """A level neither Arkane nor ARC has atom energies for: Arkane runs without corrections,
        the requested BAC is dropped with them, and the thermo says so."""
        spc, input_py = self._compute_thermo_with_mocked_arkane(
            sp_level=Level(method='b3lyp', basis='sto-3g', software='gaussian'), bac_type='p')
        self.assertIn('useAtomCorrections = False\n', input_py)
        self.assertIn('useBondCorrections = False\n', input_py)
        self.assertEqual(spc.thermo.H298, -74.6)
        self.assertIs(spc.thermo.atom_corrections_applied, False)
        self.assertIs(spc.thermo.bond_corrections_applied, False)
        self.assertIsNone(spc.thermo.atom_corrections_level)

    def test_thermo_of_a_solvated_level_has_no_corrections(self):
        """Arkane has no corrections for a solvated level, so the run is the no-correction fallback, although
        the gas-phase wb97xd/def2tzvp has them."""
        for sp_level in (Level(method='wb97xd', basis='def2tzvp', software='gaussian',
                               solvation_method='smd', solvent='water'),
                         Level(method='gfn2', solvation_method='alpb', solvent='water')):
            with self.subTest(sp_level=str(sp_level)):
                spc, input_py = self._compute_thermo_with_mocked_arkane(sp_level=sp_level, bac_type='p')
                self.assertNotIn('modelChemistry = ', input_py)
                self.assertNotIn('atomEnergies = ', input_py)
                self.assertIn('useAtomCorrections = False\n', input_py)
                self.assertIn('useBondCorrections = False\n', input_py)
                self.assertIs(spc.thermo.atom_corrections_applied, False)
                self.assertIs(spc.thermo.bond_corrections_applied, False)
                self.assertIsNone(spc.thermo.atom_corrections_level)

    def test_thermo_of_a_dispersion_field_level_uses_only_that_dispersion(self):
        """b3lyp/def2tzvp + gd3bj gets no plain b3lyp corrections; b2plyp/def2tzvp + gd3 gets the b2plypd3 ones."""
        path = TestDispersionAndSolvationMatching._write_qm_sections(
            "LevelOfTheory(method='b3lyp2023',basis='def2tzvp',software='gaussian')")
        self.addCleanup(os.remove, path)
        with patch('arc.statmech.arkane.get_qm_corrections_files', return_value=[path]):
            spc, input_py = self._compute_thermo_with_mocked_arkane(
                sp_level=Level(method='b3lyp', basis='def2tzvp', software='gaussian', dispersion='gd3bj'),
                bac_type='p')
        self.assertNotIn('modelChemistry = ', input_py)
        self.assertIn('useAtomCorrections = False\n', input_py)
        self.assertIs(spc.thermo.atom_corrections_applied, False)
        self.assertIsNone(spc.thermo.atom_corrections_level)
        sp_level = Level(method='b2plyp', basis='def2tzvp', software='gaussian', dispersion='gd3')
        spc, input_py = self._compute_thermo_with_mocked_arkane(sp_level=sp_level, bac_type='p')
        self.assertIn("modelChemistry = LevelOfTheory(method='b2plypd32023'", input_py)
        self.assertIn('useAtomCorrections = True\n', input_py)
        self.assertIn('useBondCorrections = True\n', input_py)
        self.assertIs(spc.thermo.atom_corrections_applied, True)
        self.assertEqual(spc.thermo.atom_corrections_level, sp_level)

    def test_thermo_of_a_gas_phase_sp_level_with_a_solvated_freq_level_is_corrected(self):
        """With no frequency scale factor, an SMD frequency level still matches its gas-phase Arkane frequency key,
        as before, so the gas-phase sp level's corrections are applied."""
        sp_level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        spc, input_py = self._compute_thermo_with_mocked_arkane(
            sp_level=sp_level, bac_type='p', freq_scale_factor=None,
            freq_level=Level(method='wb97xd', basis='def2tzvp', software='gaussian',
                             solvation_method='smd', solvent='water'))
        self.assertIn('modelChemistry = CompositeLevelOfTheory(', input_py)
        self.assertIn('useAtomCorrections = True\n', input_py)
        self.assertIn('useBondCorrections = True\n', input_py)
        self.assertIs(spc.thermo.atom_corrections_applied, True)
        self.assertEqual(spc.thermo.atom_corrections_level, sp_level)

    def test_thermo_records_the_fallback_of_an_unmatched_frequency_level(self):
        """Without a frequency scale factor ARC asks Arkane for a composite model chemistry, which needs a
        frequency entry too. With none for the frequency level Arkane gets no model chemistry, and so no atom
        energy corrections, although the sp level alone has them."""
        spc, input_py = self._compute_thermo_with_mocked_arkane(
            sp_level=Level(method='wb97xd', basis='def2tzvp', software='gaussian'), bac_type='p',
            freq_level=Level(method='b3lyp', basis='sto-3g', software='gaussian'), freq_scale_factor=None)
        self.assertNotIn('modelChemistry = ', input_py)
        self.assertIn('useAtomCorrections = False\n', input_py)
        self.assertIs(spc.thermo.atom_corrections_applied, False)
        self.assertIs(spc.thermo.bond_corrections_applied, False)
        self.assertIsNone(spc.thermo.atom_corrections_level)

    def test_thermo_of_a_species_loaded_from_an_arkane_yaml_is_unknown(self):
        """Arkane loads a species declared by its own YAML file as-is and applies no correction to it, so this
        run's switches say nothing about its energy. A computed species in the same run is still stamped."""
        yml_spc = ARCSpecies(label='H2O', smiles='O')
        yml_spc.yml_path = os.path.join(ARC_TESTING_PATH, 'yml_testing', 'H2O.yml')
        computed = ARCSpecies(label='CH4', smiles='C')
        self._compute_thermo_with_mocked_arkane(
            sp_level=Level(method='wb97xd', basis='def2tzvp', software='gaussian'), bac_type='p',
            species=[yml_spc, computed])
        self.assertEqual(yml_spc.thermo.H298, -74.6)
        self.assertIsNone(yml_spc.thermo.atom_corrections_applied)
        self.assertIsNone(yml_spc.thermo.bond_corrections_applied)
        self.assertIsNone(yml_spc.thermo.atom_corrections_level)
        self.assertIs(computed.thermo.atom_corrections_applied, True)
        self.assertIs(computed.thermo.bond_corrections_applied, True)

    def test_thermo_correction_flags_stay_unknown_without_a_rendered_input(self):
        """Thermo parsed by an adapter that rendered no Arkane input is not stamped with a guess,
        and a species the run produced no thermo for is left untouched."""
        tmpdir = tempfile.mkdtemp(prefix='test_Arkane_corrections_unknown_')
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        statmech_dir = os.path.join(tmpdir, 'calcs', 'statmech', 'thermo')
        os.makedirs(statmech_dir)
        with open(os.path.join(statmech_dir, 'output.py'), 'w') as f:
            f.write('')
        save_yaml_file(path=os.path.join(statmech_dir, 'thermo.yaml'),
                       content={'CH4': {'H298': -74.6, 'S298': 186.3, 'data': 'NASA()'}})
        spc, other = ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='H2O', smiles='O')
        other.thermo.H298 = -241.8
        arkane = ArkaneAdapter(output_directory=os.path.join(tmpdir, 'output'),
                               calcs_directory=os.path.join(tmpdir, 'calcs'),
                               output_dict=dict(),
                               species=[spc, other])
        with patch('arc.statmech.arkane.execute_command', return_value=('', '')):
            arkane.parse_arkane_thermo_output(statmech_dir)
        self.assertEqual(spc.thermo.H298, -74.6)
        self.assertIsNone(spc.thermo.atom_corrections_applied)
        self.assertIsNone(spc.thermo.bond_corrections_applied)
        self.assertFalse(hasattr(other.thermo, 'atom_corrections_applied'))

        arkane.use_aec, arkane.use_bac = False, False
        with patch('arc.statmech.arkane.execute_command', return_value=('', '')):
            arkane.parse_arkane_thermo_output(statmech_dir)
        self.assertIs(spc.thermo.atom_corrections_applied, False)
        self.assertFalse(hasattr(other.thermo, 'atom_corrections_applied'))


THERMO_BLOCK = ("thermo(\n    label = '{label}',\n    thermo = ThermoData(\n"
                "        Tdata = ([300.0, 400.0], 'K'),\n        Cpdata = ([33.0, 36.0], 'J/(mol*K)'),\n"
                "        H298 = (-74.6, 'kJ/mol'),\n        S298 = (186.3, 'J/(mol*K)'),\n    )\n)\n")


class TestArkaneStandardStatePressure(unittest.TestCase):
    """Tests for the standard-state pressure recorded on the thermo of an Arkane run."""

    def _parse(self, species, output_content, thermo_yaml):
        """Run ``parse_arkane_thermo_output`` on a mocked run directory, with ``thermo_yaml`` (if any) in it."""
        tmpdir = tempfile.mkdtemp(prefix='test_Arkane_pressure_')
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        statmech_dir = os.path.join(tmpdir, 'calcs', 'statmech', 'thermo')
        os.makedirs(statmech_dir)
        with open(os.path.join(statmech_dir, 'output.py'), 'w') as f:
            f.write(output_content)
        if thermo_yaml is not None:
            save_yaml_file(path=os.path.join(statmech_dir, 'thermo.yaml'), content=thermo_yaml)
        arkane = ArkaneAdapter(output_directory=os.path.join(tmpdir, 'output'),
                               calcs_directory=os.path.join(tmpdir, 'calcs'),
                               output_dict=dict(),
                               species=species)
        with patch('arc.statmech.arkane.execute_command', return_value=('', '')):
            arkane.parse_arkane_thermo_output(statmech_dir)

    @unittest.skipUnless(importlib.util.find_spec('rmgpy') is not None, 'rmgpy is not importable')
    def test_the_constant_is_the_pressure_the_thermo_script_measures(self):
        """Test that the pressure recorded for thermo read from output.py is the one RMG's partition function applies"""
        scripts_path = os.path.join(ARC_PATH, 'arc', 'scripts')
        sys.path.insert(0, scripts_path)
        self.addCleanup(sys.path.remove, scripts_path)
        script = importlib.import_module('save_arkane_thermo')
        self.assertAlmostEqual(script.standard_state_pressure_pa(), ARKANE_STANDARD_STATE_PRESSURE_PA, places=3)

    def test_the_pressure_the_thermo_script_measured_is_recorded(self):
        """Test that a pressure carried by thermo.yaml is recorded as it is, whatever its value"""
        spc = ARCSpecies(label='CH4', smiles='C')
        self._parse([spc], '', {'CH4': {'H298': -74.6, 'S298': 186.3, 'data': 'NASA()',
                                         'standard_state_pressure_pa': 100000.0}})
        self.assertEqual(spc.thermo.standard_state_pressure_pa, 100000.0)

    def test_a_thermo_read_from_arkane_output_without_a_thermo_yaml_is_at_one_atmosphere(self):
        """Test that thermo parsed from output.py when thermo.yaml is missing or carries no pressure is at 1 atm"""
        for thermo_yaml in (None, {'CH4': {'H298': -74.6, 'S298': 186.3, 'data': 'NASA()'}}):
            with self.subTest(thermo_yaml=thermo_yaml):
                spc = ARCSpecies(label='CH4', smiles='C')
                self._parse([spc], THERMO_BLOCK.format(label='CH4'), thermo_yaml)
                self.assertEqual(spc.thermo.H298, -74.6)
                self.assertEqual(spc.thermo.standard_state_pressure_pa, 101325.0)

    def test_a_thermo_that_no_arkane_run_wrote_has_no_pressure(self):
        """Test that thermo set by a caller, with no block in the output and no thermo.yaml entry, stays null"""
        spc, other = ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='H2O', smiles='O')
        other.thermo.H298 = -241.8
        self._parse([spc, other], THERMO_BLOCK.format(label='CH4'), None)
        self.assertEqual(spc.thermo.standard_state_pressure_pa, 101325.0)
        self.assertIsNone(other.thermo.standard_state_pressure_pa)

    def test_a_species_without_thermo_in_the_output_has_none(self):
        """Test that the fallback needs a thermo block of the species in the output"""
        spc = ARCSpecies(label='CH4', smiles='C')
        self._parse([spc], '', None)
        self.assertIsNone(spc.thermo.H298)
        self.assertIsNone(getattr(spc.thermo, 'standard_state_pressure_pa', None))


def conformer_block(label: str, e0: float, modes: list[str]) -> str:
    """A ``conformer(...)`` block in the syntax Arkane writes to ``output.py``, with the given mode reprs."""
    mode_lines = ''.join(f'        {mode},\n' for mode in modes)
    return (f"conformer(\n    label = '{label}',\n    E0 = ({e0}, 'kJ/mol'),\n    modes = [\n"
            f"        IdealGasTranslation(mass=(16.0313, 'amu')),\n"
            f"        NonlinearRotor(inertia=([3.2, 3.2, 3.2], 'amu*angstrom^2'), symmetry=12),\n"
            f"        HarmonicOscillator(frequencies=([1300.0, 1500.0], 'cm^-1')),\n{mode_lines}    ],\n"
            f"    spin_multiplicity = 1,\n    optical_isomers = 1,\n)\n\n")


HINDERED_ROTOR = "HinderedRotor(inertia=(1.5, 'amu*angstrom^2'), symmetry=3, barrier=(12.0, 'kJ/mol'))"
FREE_ROTOR = "FreeRotor(inertia=(1.1, 'amu*angstrom^2'), symmetry=1)"
ND_ROTOR = "Mode(quantum=False)"


class TestArkaneE0CorrectionsAndTreatment(unittest.TestCase):
    """
    Tests for the correction switches stamped on every E0 an Arkane run writes, on the kinetics of a run, and for
    the rotor treatment read back from the conformer blocks of Arkane's own output. The Arkane and RMG subprocesses
    are mocked, and the rendered input and the parsed output come from the same adapter call.
    """
    wb97xd = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
    no_corrections = Level(method='b3lyp', basis='sto-3g', software='gaussian')

    def _compute_thermo(self, sp_level, bac_type, output_content, species=None, e0_only=False):
        return TestArkaneCorrectionFlags._compute_thermo_with_mocked_arkane(
            self, sp_level=sp_level, bac_type=bac_type, species=species, output_content=output_content,
            e0_only=e0_only)

    def test_the_thermo_and_e0_only_runs_stamp_their_own_switches_on_e0(self):
        """E0 is stamped with the atom and bond switches of the run that rendered the input: BAC on for a run
        with a bac_type, off without one, and both off at a level Arkane has no atom energies for."""
        content = conformer_block('CH4', -88.8, [])
        for e0_only in (False, True):
            for sp_level, bac_type, expected in ((self.wb97xd, 'p', (True, True)),
                                                 (self.wb97xd, None, (True, False)),
                                                 (self.no_corrections, 'p', (False, False))):
                with self.subTest(e0_only=e0_only, level=sp_level.simple(), bac_type=bac_type):
                    spc, _ = self._compute_thermo(sp_level, bac_type, content, e0_only=e0_only)
                    self.assertAlmostEqual(spc.e0, -88.8)
                    self.assertEqual((spc.e0_atom_corrections_applied, spc.e0_bond_corrections_applied), expected)

    def test_the_e0_is_stamped_with_the_aec_yml_digest_of_the_run_that_rendered_atom_energies_from_it(self):
        """E0 carries the digest of data/AEC.yml when the run rendered atom energies from it, in the thermo and the
        E0-only mode, and no digest when it did not or when the species is loaded from an Arkane YAML."""
        with open(os.path.join(ARC_PATH, 'data', 'AEC.yml'), 'rb') as f:
            expected = hashlib.sha256(f.read()).hexdigest()
        content = conformer_block('CH4', -88.8, [])
        for e0_only in (False, True):
            for sp_level, digest in ((Level('gfn2'), expected), (self.no_corrections, None)):
                with self.subTest(e0_only=e0_only, level=sp_level.simple()):
                    spc, _ = self._compute_thermo(sp_level, None, content, e0_only=e0_only)
                    self.assertEqual(spc.e0_aec_yml_sha256, digest)
        yml_spc = ARCSpecies(label='H2O', smiles='O')
        yml_spc.yml_path = os.path.join(ARC_TESTING_PATH, 'yml_testing', 'H2O.yml')
        computed = ARCSpecies(label='CH4', smiles='C')
        self._compute_thermo(Level('gfn2'), None, conformer_block('H2O', -240.0, []) + content,
                             species=[yml_spc, computed])
        self.assertIsNone(yml_spc.e0_aec_yml_sha256)
        self.assertEqual(computed.e0_aec_yml_sha256, expected)

    def test_an_e0_the_run_did_not_write_keeps_its_earlier_switches(self):
        """A species Arkane's output has no conformer block for keeps the E0 and the switches it had."""
        spc = ARCSpecies(label='CH4', smiles='C')
        spc.e0, spc.e0_atom_corrections_applied, spc.e0_bond_corrections_applied = -90.0, True, True
        self._compute_thermo(self.no_corrections, None, '', species=[spc])
        self.assertEqual((spc.e0, spc.e0_atom_corrections_applied, spc.e0_bond_corrections_applied),
                         (-90.0, True, True))

    def test_the_e0_of_a_species_loaded_from_an_arkane_yaml_has_unknown_switches(self):
        """Arkane loads a YAML species as-is, so the run's switches say nothing about its E0."""
        yml_spc = ARCSpecies(label='H2O', smiles='O')
        yml_spc.yml_path = os.path.join(ARC_TESTING_PATH, 'yml_testing', 'H2O.yml')
        computed = ARCSpecies(label='CH4', smiles='C')
        self._compute_thermo(self.wb97xd, 'p', conformer_block('H2O', -240.0, []) + conformer_block('CH4', -88.8, []),
                             species=[yml_spc, computed])
        self.assertAlmostEqual(yml_spc.e0, -240.0)
        self.assertIsNone(yml_spc.e0_atom_corrections_applied)
        self.assertIsNone(yml_spc.e0_bond_corrections_applied)
        self.assertIs(computed.e0_atom_corrections_applied, True)

    def test_the_rotor_modes_arkane_kept_are_read_from_its_own_output(self):
        """The hindered and free rotors in the modes list are recorded in order, and so is a ``Mode``, which is how
        Arkane writes a multi-dimensional rotor and whose treatment is unknown. An output with no rotor, which is
        what Arkane writes after dropping every rotor for lack of a force-constant matrix, is an empty list."""
        for modes, expected, treatment in (([HINDERED_ROTOR, FREE_ROTOR, HINDERED_ROTOR],
                                            ['HinderedRotor', 'FreeRotor', 'HinderedRotor'], 'rrho_1d'),
                                           ([], [], 'rrho'),
                                           ([ND_ROTOR], ['Mode'], None),
                                           ([HINDERED_ROTOR, ND_ROTOR], ['HinderedRotor', 'Mode'], None)):
            with self.subTest(expected=expected):
                spc, _ = self._compute_thermo(self.wb97xd, 'p', conformer_block('CH4', -88.8, modes))
                self.assertEqual(spc.arkane_rotor_modes, expected)
                self.assertEqual(get_arkane_treatment(spc.arkane_rotor_modes), treatment)

    def test_a_conformer_block_without_modes_leaves_the_rotor_modes_unparsed(self):
        spc, _ = self._compute_thermo(self.wb97xd, 'p', "conformer(label='CH4', E0=(-88.8, 'kJ/mol'))\n")
        self.assertIsNone(spc.arkane_rotor_modes)
        spc, _ = self._compute_thermo(self.wb97xd, 'p', '')
        self.assertIsNone(spc.arkane_rotor_modes)

    def test_the_treatment_names_follow_the_rotor_modes(self):
        self.assertIsNone(get_arkane_treatment(None))
        self.assertEqual(get_arkane_treatment([]), 'rrho')
        self.assertEqual(get_arkane_treatment(['FreeRotor']), 'rrho_1d')
        self.assertEqual(get_arkane_treatment(['HinderedRotor2D']), 'rrho_nd')
        self.assertEqual(get_arkane_treatment(['HinderedRotor2D', 'FreeRotor']), 'rrho_1d_nd')
        self.assertIsNone(get_arkane_treatment(['Mode']))
        self.assertIsNone(get_arkane_treatment(['HinderedRotor', 'Mode']))

    def test_a_yml_ts_or_well_leaves_the_kinetics_switch_unknown(self):
        """Arkane applies no correction to a species it loads from a YAML, so the kinetics run's switch is not
        stated for a reaction with such a TS or well, and the TS's own E0 switches are unknown too."""
        content = conformer_block('TS0', 50.0, []) + TestArkaneOutputParsing.kinetics_output_content.replace(
            "conformer(label='TS0', E0=(50.0, 'kJ/mol'), modes=[], spin_multiplicity=2, optical_isomers=1)", '')
        yml_path = os.path.join(ARC_TESTING_PATH, 'yml_testing', 'H2O.yml')
        for yml_holder in ('ts', 'well', None):
            with self.subTest(yml_holder=yml_holder):
                rxn = TestArkaneOutputParsing.isomerization_reaction()
                if yml_holder == 'ts':
                    rxn.ts_species.yml_path = yml_path
                elif yml_holder == 'well':
                    rxn.r_species[0].yml_path = yml_path
                parse_reaction_kinetics(rxn, content, use_aec=True, use_bac=False)
                self.assertEqual(rxn.kinetics['atom_corrections_applied'], None if yml_holder else True)
                self.assertEqual(rxn.ts_species.e0_atom_corrections_applied, None if yml_holder == 'ts' else True)

    def test_the_species_file_declares_the_bonds_the_corrections_are_computed_from(self):
        """``bonds`` is written from ``bond_corrections`` so Arkane's BAC and ARC's exported components use one
        dictionary, and is not written when ARC has none."""
        tmpdir = tempfile.mkdtemp(prefix='test_Arkane_bonds_')
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        spc = ARCSpecies(label='CH4', smiles='C')
        output_dict = {'CH4': {'paths': {'composite': '', 'sp': 'sp.log', 'freq': 'freq.log'}}}
        arkane = ArkaneAdapter(output_directory=tmpdir, calcs_directory=tmpdir, output_dict=output_dict,
                               species=[spc], sp_level=self.wb97xd)
        for bond_corrections in ({'C-H': 4}, dict()):
            with self.subTest(bond_corrections=bond_corrections):
                spc.bond_corrections = bond_corrections
                arkane.generate_species_file(spc, tmpdir, skip_rotors=True)
                with open(spc.arkane_file, 'r') as f:
                    content = f.read()
                if bond_corrections:
                    self.assertIn("bonds = {'C-H': 4}\n", content)
                else:
                    self.assertNotIn('bonds', content)
                local_context = dict()
                exec(content, {'Log': lambda path: path}, local_context)
                self.assertEqual(local_context.get('bonds'), bond_corrections or None)

    def test_the_adapter_records_the_digest_of_aec_yml_when_it_renders_atom_energies_from_it(self):
        """The digest of data/AEC.yml is recorded at render time, once however often an input is rendered, and not
        recorded for a level whose atom energies ARC does not render from it."""
        tmpdir = tempfile.mkdtemp(prefix='test_Arkane_aec_digest_')
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        with open(os.path.join(ARC_PATH, 'data', 'AEC.yml'), 'rb') as f:
            expected = hashlib.sha256(f.read()).hexdigest()
        for sp_level, recorded in ((Level('gfn2'), {expected}),
                                   (Level(method='b3lyp', basis='sto-3g', software='gaussian'), set())):
            with self.subTest(sp_level=sp_level):
                arkane = ArkaneAdapter(output_directory=tmpdir, calcs_directory=tmpdir, output_dict=dict(),
                                       species=[ARCSpecies(label='CH4', smiles='C')], sp_level=sp_level)
                self.assertEqual(arkane.aec_yml_sha256s, set())
                for _ in range(2):
                    arkane.render_arkane_input_template(statmech_dir=tmpdir, skip_rotors=True)
                self.assertEqual(arkane.aec_yml_sha256s, recorded)

    def test_get_file_sha256(self):
        """The digest is that of the file bytes, and ``None`` for a file that cannot be read."""
        tmpdir = tempfile.mkdtemp(prefix='test_Arkane_file_sha256_')
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        path = os.path.join(tmpdir, 'table.txt')
        with open(path, 'wb') as f:
            f.write(b'atom energies\n')
        self.assertEqual(get_file_sha256(path), hashlib.sha256(b'atom energies\n').hexdigest())
        self.assertIsNone(get_file_sha256(os.path.join(tmpdir, 'missing.txt')))

    def test_normalized_method_and_basis(self):
        """A level's method carries its dispersion, and its basis is normalized."""
        self.assertEqual(normalized_method_and_basis(Level(method='B3LYP-D3(BJ)', basis='def2-TZVP')),
                         ('b3lypd3bj', 'def2tzvp'))
        self.assertEqual(normalized_method_and_basis(Level(method='b3lyp', basis='def2tzvp', dispersion='gd3bj')),
                         ('b3lypd3bj', 'def2tzvp'))

    def _compute_kinetics(self, sp_level, output_content):
        """Run ``compute_high_p_rate_coefficient`` with Arkane mocked, for the nitroethane isomerization."""
        tmpdir = tempfile.mkdtemp(prefix='test_Arkane_kinetics_corrections_')
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        rxn = TestArkaneOutputParsing.isomerization_reaction()
        arkane = ArkaneAdapter(output_directory=os.path.join(tmpdir, 'output'),
                               calcs_directory=os.path.join(tmpdir, 'calcs'),
                               output_dict=dict(),
                               bac_type=None,
                               species=rxn.r_species + rxn.p_species,
                               reactions=[rxn],
                               sp_level=sp_level,
                               freq_level=sp_level,
                               freq_scale_factor=1.0)

        def fake_run_arkane(statmech_dir):
            with open(os.path.join(statmech_dir, 'output.py'), 'w') as f:
                f.write(output_content)
            return True

        with patch('arc.statmech.arkane.run_arkane', side_effect=fake_run_arkane), \
                patch.object(ArkaneAdapter, 'generate_species_files'), \
                patch.object(ArkaneAdapter, 'generate_ts_files'), \
                patch.object(ArkaneAdapter, 'filter_out_unconverged_reactions'), \
                patch('arc.statmech.arkane.plotter.log_kinetics'), \
                patch('arc.statmech.arkane.clean_output_directory'):
            arkane.compute_high_p_rate_coefficient()
        return rxn

    def test_the_kinetics_run_stamps_its_switches_on_the_ts_e0_and_on_the_kinetics(self):
        """The kinetics run applies no BAC. Its atom switch is stamped on the TS's E0 and recorded as
        ``atom_corrections_applied`` of the kinetics; the rotor modes of the TS and the wells are read as well."""
        content = (conformer_block('nitroethane', -10.0, [HINDERED_ROTOR])
                   + conformer_block('ethyl_nitrite', -5.0, [])
                   + TestArkaneOutputParsing.kinetics_output_content.replace(
                       "conformer(label='TS0', E0=(50.0, 'kJ/mol'), modes=[], spin_multiplicity=2, optical_isomers=1)",
                       conformer_block('TS0', 50.0, [FREE_ROTOR])))
        for sp_level, expected_atom in ((self.wb97xd, True), (self.no_corrections, False)):
            with self.subTest(level=sp_level.simple()):
                rxn = self._compute_kinetics(sp_level, content)
                self.assertIs(rxn.kinetics['atom_corrections_applied'], expected_atom)
                self.assertAlmostEqual(rxn.ts_species.e0, 50.0)
                self.assertIs(rxn.ts_species.e0_atom_corrections_applied, expected_atom)
                self.assertIs(rxn.ts_species.e0_bond_corrections_applied, False)
                self.assertEqual(rxn.ts_species.arkane_rotor_modes, ['FreeRotor'])
                self.assertEqual(rxn.r_species[0].arkane_rotor_modes, ['HinderedRotor'])
                self.assertEqual(rxn.p_species[0].arkane_rotor_modes, [])

    def test_the_kinetics_run_stamps_the_aec_yml_digest_on_the_ts_e0(self):
        """The TS E0 written by a kinetics run carries the digest of data/AEC.yml the run rendered atom energies from."""
        with open(os.path.join(ARC_PATH, 'data', 'AEC.yml'), 'rb') as f:
            expected = hashlib.sha256(f.read()).hexdigest()
        content = (conformer_block('nitroethane', -10.0, []) + conformer_block('ethyl_nitrite', -5.0, [])
                   + TestArkaneOutputParsing.kinetics_output_content.replace(
                       "conformer(label='TS0', E0=(50.0, 'kJ/mol'), modes=[], spin_multiplicity=2, optical_isomers=1)",
                       conformer_block('TS0', 50.0, [])))
        self.assertEqual(self._compute_kinetics(Level('gfn2'), content).ts_species.e0_aec_yml_sha256, expected)
        self.assertIsNone(self._compute_kinetics(self.no_corrections, content).ts_species.e0_aec_yml_sha256)

    def test_kinetics_parsed_without_a_rendered_input_state_no_switch(self):
        """Parsing with no record of the run's switches stamps null, never a default."""
        rxn = TestArkaneOutputParsing.isomerization_reaction()
        parse_reaction_kinetics(rxn, TestArkaneOutputParsing.kinetics_output_content)
        self.assertIsNone(rxn.kinetics['atom_corrections_applied'])
        self.assertIsNone(rxn.ts_species.e0_atom_corrections_applied)
        self.assertIsNone(rxn.ts_species.e0_bond_corrections_applied)

    def test_the_ts_check_e0_run_switches_travel_with_the_copied_e0(self):
        """``compute_rxn_e0`` runs an E0-only Arkane job without BAC on a copy of the reaction, and
        ``copy_e0_values`` brings each E0 and its switches back, so the TS carries the switches of that run while a
        well that already has an E0 keeps its own E0 and switches."""
        rxn = TestArkaneOutputParsing.isomerization_reaction()
        well = rxn.r_species[0]
        well.e0, well.e0_atom_corrections_applied, well.e0_bond_corrections_applied = -10.0, True, True
        rxn_copy = rxn.copy()
        content = (conformer_block('nitroethane', -11.0, []) + conformer_block('ethyl_nitrite', -5.0, [])
                   + conformer_block('TS0', 50.0, []))
        species = rxn_copy.r_species + rxn_copy.p_species + [rxn_copy.ts_species]
        self._compute_thermo(self.wb97xd, None, content, species=species, e0_only=True)
        rxn.copy_e0_values(rxn_copy)
        self.assertEqual((rxn.ts_species.e0, rxn.ts_species.e0_atom_corrections_applied,
                          rxn.ts_species.e0_bond_corrections_applied), (50.0, True, False))
        self.assertEqual((well.e0, well.e0_atom_corrections_applied, well.e0_bond_corrections_applied),
                         (-10.0, True, True))
        product = rxn.p_species[0]
        self.assertEqual((product.e0, product.e0_atom_corrections_applied, product.e0_bond_corrections_applied),
                         (-5.0, True, False))


class TestArkaneOutputParsing(unittest.TestCase):
    """Tests for parsing functions that read Arkane output.py content."""

    kinetics_output_content = """
conformer(label='TS0', E0=(50.0, 'kJ/mol'), modes=[], spin_multiplicity=2, optical_isomers=1)

kinetics(
    label = 'nitroethane <=> ethyl_nitrite',
    kinetics = Arrhenius(
        A = (5.0, 's^-1'),
        n = 1.0,
        Ea = (20.0, 'kJ/mol'),
        T0 = (1, 'K'),
        Tmin = (300, 'K'),
        Tmax = (2000, 'K'),
        comment = 'Fitted to 10 data points; dA = *|/ 1.1, dn = +|- 0.01, dEa = +|- 0.1 kJ/mol',
    ),
)
"""

    @staticmethod
    def isomerization_reaction() -> ARCReaction:
        """Build the real nitroethane <=> ethyl nitrite reaction with a TS species named 'TS0'."""
        rxn = ARCReaction(r_species=[ARCSpecies(label='nitroethane', smiles='CC[N+](=O)[O-]')],
                          p_species=[ARCSpecies(label='ethyl_nitrite', smiles='CCON=O')])
        rxn.ts_species = ARCSpecies(label='TS0', is_ts=True)
        return rxn

    def test_parse_e0(self):
        """Test parse_e0 extracts E0 from conformer blocks."""
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
        content = "conformer(label='CHO', E0=(44.0971, 'kJ/mol'), modes=[], spin_multiplicity=2, optical_isomers=1)"
        self.assertAlmostEqual(parse_e0('CHO', content), 44.0971)

    def test_parse_conformer_statmech(self):
        """Test extraction of external_symmetry and optical_isomers."""
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
        spc = ARCSpecies(label='H2O', smiles='O')
        _parse_conformer_statmech(spc, content)
        self.assertEqual(spc.optical_isomers, 1)
        self.assertEqual(spc.external_symmetry, 2)

    def test_parse_conformer_statmech_linear(self):
        """Test with LinearRotor."""
        content = """
conformer(
    label = 'CO2',
    E0 = (-100.0, 'kJ/mol'),
    modes = [LinearRotor(inertia=(44.0, 'amu*angstrom^2'), symmetry=2)],
    spin_multiplicity = 1,
    optical_isomers = 1,
)
"""
        spc = ARCSpecies(label='CO2', smiles='O=C=O')
        _parse_conformer_statmech(spc, content)
        self.assertEqual(spc.external_symmetry, 2)
        self.assertEqual(spc.optical_isomers, 1)

    def test_parse_reaction_kinetics_with_uncertainties(self):
        """Test that dA, dn, dEa, n_data_points are parsed from the comment."""
        content = """
conformer(label='TS0', E0=(50.0, 'kJ/mol'), modes=[], spin_multiplicity=2, optical_isomers=1)

kinetics(
    label = 'CH4 + OH <=> CH3 + H2O',
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
        rxn = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C'),
                                     ARCSpecies(label='OH', smiles='[OH]')],
                          p_species=[ARCSpecies(label='CH3', smiles='[CH3]'),
                                     ARCSpecies(label='H2O', smiles='O')])
        rxn.ts_species = ARCSpecies(label='TS0', is_ts=True)
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
        content = """
conformer(label='TS0', E0=(50.0, 'kJ/mol'), modes=[], spin_multiplicity=2, optical_isomers=1)

kinetics(
    label = 'nitroethane <=> ethyl_nitrite',
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
        rxn = self.isomerization_reaction()
        parse_reaction_kinetics(rxn, content)
        self.assertAlmostEqual(rxn.kinetics['A'][0], 5.0)
        self.assertAlmostEqual(rxn.kinetics['n'], 1.0)
        self.assertNotIn('dA', rxn.kinetics)

    def test_parse_reaction_kinetics_marks_an_irc_invalid_ts(self):
        """Kinetics of a TS that failed the IRC check are reported, and are labeled as invalid."""
        content = self.kinetics_output_content
        rxn = self.isomerization_reaction()
        rxn.ts_species.ts_checks['IRC'] = False
        rxn.ts_species.ts_checks['NMD'] = True
        parse_reaction_kinetics(rxn, content)
        self.assertAlmostEqual(rxn.kinetics['A'][0], 5.0)
        self.assertAlmostEqual(rxn.kinetics['Ea'][0], 20.0)
        self.assertIn(TS_IRC_FAILED_MARKER, rxn.kinetics['ts_validation'])
        self.assertIn(TS_IRC_FAILED_MARKER, rxn.kinetics['comment'])
        self.assertIn('Fitted to 10 data points', rxn.kinetics['comment'])

    def test_parse_reaction_kinetics_does_not_mark_an_unchecked_or_valid_ts(self):
        """Kinetics of a TS for which the IRC check was not performed or was passed are not labeled."""
        content = self.kinetics_output_content
        for irc_value in [None, True]:
            rxn = self.isomerization_reaction()
            rxn.ts_species.ts_checks['IRC'] = irc_value
            parse_reaction_kinetics(rxn, content)
            self.assertAlmostEqual(rxn.kinetics['A'][0], 5.0)
            self.assertNotIn('ts_validation', rxn.kinetics)
            self.assertNotIn(TS_IRC_FAILED_MARKER, rxn.kinetics['comment'])

    def test_parse_thermo_data_block_scalars_are_float(self):
        """Verify Tmin, Tmax, H298, S298 are parsed as floats, not strings."""
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


class TestRunArkaneOutputPySignal(unittest.TestCase):
    """``run_arkane``'s pass/fail signal is whether ``output.py`` was
    produced — NOT whether stderr is empty. The stderr classification
    is now advisory (logged) but doesn't gate the return value.

    Pre-fix bug: complete Arkane runs were discarded because cosmetic
    stderr noise (git rev-parse failure, OpenBabel warnings) tripped
    a false-failure gate. Both the thermo and kinetics callers now
    use the same authoritative output.py-existence signal.
    """

    def setUp(self):
        self._run_arkane = run_arkane
        self.tmp = tempfile.mkdtemp(prefix='arkane-stderr-test-')
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        # Pre-flight check in run_arkane requires input.py before the
        # subprocess fires.
        with open(os.path.join(self.tmp, 'input.py'), 'w') as f:
            f.write('# fake arkane input\n')

    def _create_output_py(self):
        with open(os.path.join(self.tmp, 'output.py'), 'w') as f:
            f.write('# fake arkane output\n')

    def _run_with_stderr(self, stderr_lines):
        with patch('arc.statmech.arkane.execute_command',
                   return_value=(['ok'], stderr_lines)):
            return self._run_arkane(self.tmp)

    # ---- output.py present: success regardless of stderr ----

    def test_empty_stderr_with_output_returns_true(self):
        self._create_output_py()
        self.assertTrue(self._run_with_stderr([]))

    def test_cosmetic_stderr_with_output_returns_true(self):
        """OpenBabel + git warnings are classified as cosmetic and don't gate."""
        self._create_output_py()
        self.assertTrue(self._run_with_stderr([
            'fatal: not a git repository (or any of the parent directories): .git',
            '*** Open Babel Warning  in InChI code',
            '  #1 :Accepted unusual valence(s): C(3)',
        ]))

    def test_real_error_with_output_returns_true_and_warns(self):
        """If output.py was produced, even a Python traceback in stderr
        doesn't fail the run — but it IS logged at WARNING so the
        operator sees it. This is the load-bearing change vs. the old
        behavior that would have returned False here.
        """
        self._create_output_py()
        with self.assertLogs('arc', level='WARNING') as logs:
            result = self._run_with_stderr([
                'Traceback (most recent call last):',
                'KeyError: "level_of_theory"',
            ])
        self.assertTrue(result, "output.py exists → success regardless of stderr")
        self.assertTrue(
            any('still produced output.py' in m for m in logs.output),
            f"expected the advisory warning to fire; got {logs.output}",
        )

    # ---- output.py missing: failure ----

    def test_missing_output_py_returns_false(self):
        """No output.py = Arkane never wrote its result. That's the
        only condition that should mean 'failure'."""
        self.assertFalse(self._run_with_stderr([]))

    def test_missing_output_py_with_real_error_logs_both_diagnostics(self):
        """When output is missing AND stderr has real errors, log both
        — the operator gets cause (stderr) and effect (no output)."""
        with self.assertLogs('arc', level='ERROR') as logs:
            result = self._run_with_stderr([
                'Traceback (most recent call last):',
                'ImportError: rmgpy not installed',
            ])
        self.assertFalse(result)
        joined = '\n'.join(logs.output)
        self.assertIn('Arkane run failed', joined)
        self.assertIn('ImportError', joined)
        self.assertIn('was not created', joined)

    # ---- pre-flight checks still gate ----

    def test_missing_input_py_returns_false_pre_flight(self):
        """If the pre-flight finds no input.py, we never even run the
        subprocess — and return False without checking stderr."""
        os.unlink(os.path.join(self.tmp, 'input.py'))
        self.assertFalse(self._run_with_stderr([]))

    def test_missing_statmech_dir_returns_false_pre_flight(self):
        self.assertFalse(run_arkane('/nonexistent/dir'))


class TestClassifyArkaneStderr(unittest.TestCase):
    """Direct tests of the stderr-noise filter, independent of run_arkane."""

    def setUp(self):
        self._classify = _classify_arkane_stderr

    def test_empty_input_returns_empty(self):
        self.assertEqual(self._classify(None), [])
        self.assertEqual(self._classify([]), [])

    def test_only_cosmetic_lines_return_empty(self):
        self.assertEqual(self._classify([
            '==============================',
            '*** Open Babel Warning  in InChI code',
            '  #1 :Accepted unusual valence(s): C(3)',
            'fatal: not a git repository (or any of the parent directories): .git',
            '',  # blank lines also dropped
        ]), [])

    def test_real_lines_returned_stripped(self):
        result = self._classify([
            'fatal: not a git repository',
            '  KeyError: "level_of_theory"  ',
            '*** Open Babel Warning in InChI code',
        ])
        self.assertEqual(result, ['KeyError: "level_of_theory"'])


class TestSummarizeArkaneStderr(unittest.TestCase):
    """The stderr summarizer condenses a traceback to the salient exception line for arc.log."""

    def setUp(self):
        self._summarize = _summarize_arkane_stderr

    def test_empty(self):
        self.assertEqual(self._summarize([]), '')

    def test_picks_exception_line_over_conda_wrapper(self):
        lines = [
            'Traceback (most recent call last):',
            '  File ".../kinetics.py", line 183, in generate_kinetics',
            'ValueError: One or both of the barrier heights of -20.7 and 28.2 kJ/mol are invalid.',
            'ERROR conda.cli.main_run:execute(148): `conda run python -m arkane input.py` failed.',
        ]
        summary = self._summarize(lines)
        self.assertTrue(summary.startswith('ValueError: One or both of the barrier heights'))
        self.assertIn('[+3 more stderr line(s)]', summary)
        self.assertNotIn('Traceback', summary)

    def test_single_line_no_extra_suffix(self):
        self.assertEqual(self._summarize(['RuntimeError: boom']), 'RuntimeError: boom')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
