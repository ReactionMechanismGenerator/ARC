#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for arc/scripts/save_arkane_thermo.py.

The output.py-fallback path reconstructs NASA thermo objects with the real rmgpy thermo
classes, so these tests require rmgpy (the rmg_env). Run them as a plain script (NOT via
pytest): rmg_env's Python 3.9 cannot import the ``arc`` package that pytest pulls in for a
file inside ``arc/scripts/`` (arc.common uses ``X | None`` PEP 604 annotations). As a script,
only the standalone module + ``common`` are imported:

    conda run -n rmg_env python arc/scripts/save_arkane_thermo_test.py

Under arc_env pytest the file collects and skips cleanly (no rmgpy), so CI is unaffected.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import rmgpy.thermo
    HAS_RMG = True
except ImportError:
    HAS_RMG = False


OUTPUT_PY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'testing', 'statmech', 'thermo_aa', 'output.py')

with open(OUTPUT_PY_PATH, 'r') as _f:
    OUTPUT_PY = _f.read()


@unittest.skipUnless(HAS_RMG, 'requires rmgpy (rmg_env)')
class TestOutputPyThermoFallback(unittest.TestCase):
    """The output.py fallback recovers thermo when RMG_libraries/thermo.py is absent."""

    def test_iter_thermo_calls_finds_each_block(self):
        """One node per species; the nested thermo= keyword is not mistaken for a call."""
        import ast
        import save_arkane_thermo as sat
        calls = sat._iter_thermo_calls(OUTPUT_PY)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(isinstance(c, ast.Call) for c in calls))
        self.assertTrue(all(c.func.id == 'thermo' for c in calls))

    def test_iter_thermo_calls_tolerates_parens_in_strings(self):
        """An unbalanced parenthesis inside a string literal does not drop the block."""
        import save_arkane_thermo as sat
        content = ("thermo(\n"
                   "    label = 'X',\n"
                   "    thermo = NASA(\n"
                   "        polynomials = [],\n"
                   "        comment = 'fitted using ( an unbalanced paren',\n"
                   "    ),\n"
                   ")\n")
        self.assertEqual(len(sat._iter_thermo_calls(content)), 1)

    def test_iter_thermo_calls_reports_unparseable_file(self):
        """A file that is not valid Python yields no blocks rather than raising."""
        import save_arkane_thermo as sat
        self.assertEqual(sat._iter_thermo_calls('thermo(label = ,,,)'), [])

    def test_one_bad_entry_does_not_lose_the_others(self):
        """A species whose thermo cannot be evaluated is skipped; the rest still reach thermo.yaml."""
        import save_arkane_thermo as sat
        from common import read_yaml_file
        broken = OUTPUT_PY + """
thermo(
    label = 'Broken',
    thermo = NASA(
        polynomials = [],
        Tmin = (10, 'K'), Tmax = (3000, 'K'),
    ),
)
"""
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'output.py'), 'w') as f:
                f.write(broken)
            try:
                os.chdir(d)
                sat.main()
                content = read_yaml_file(os.path.join(d, 'thermo.yaml'))
            finally:
                os.chdir(cwd)
        self.assertEqual(set(content), {'R1', 'R2', 'P1'})

    def test_load_thermo_entries_from_output_py(self):
        import save_arkane_thermo as sat
        from rmgpy.thermo import NASA, NASAPolynomial, ThermoData, Wilhoit
        local_context = {'ThermoData': ThermoData, 'Wilhoit': Wilhoit,
                         'NASAPolynomial': NASAPolynomial, 'NASA': NASA}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'output.py')
            with open(path, 'w') as f:
                f.write(OUTPUT_PY)
            entries = sat._load_thermo_entries_from_output_py(path, local_context)
        self.assertEqual(set(entries), {'R1', 'R2', 'P1'})
        for label in ('R1', 'R2', 'P1'):
            self.assertIsInstance(entries[label], NASA)
        self.assertAlmostEqual(entries['R1'].get_enthalpy(298.15), entries['R2'].get_enthalpy(298.15))

    def test_main_writes_thermo_yaml_from_output_py(self):
        """With output.py present but no RMG_libraries/thermo.py, main() still writes thermo.yaml
        for every species (including the duplicate R2), carrying the real thermochemistry.

        P1 is H2O and is checked against JANAF reference values rather than against whatever the
        code emits: dHf(298.15 K) = -241.8 kJ/mol, S(298.15 K) = 188.8 J/(mol*K), and, from the
        high-temperature polynomial, H(2000 K) - H(298.15 K) = 72.79 kJ/mol and
        S(2000 K) = 264.77 J/(mol*K). Tolerances are loose so the assertions test the
        reconstruction rather than the level of theory, and never rely on exact floats.

        R1/R2 are OH, whose S(298.15 K) is deliberately not compared to the accepted
        183.7 J/(mol*K) because the fixture's electronic partition function omits the 2-Pi
        spin-orbit structure. What is asserted for them is the A+A invariant: two blocks for the
        same species must reconstruct to the same thermo.
        """
        import save_arkane_thermo as sat
        from common import read_yaml_file
        from rmgpy.thermo import NASA, NASAPolynomial, ThermoData, Wilhoit
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            output_path = os.path.join(d, 'output.py')
            with open(output_path, 'w') as f:
                f.write(OUTPUT_PY)
            try:
                os.chdir(d)
                sat.main()
                yaml_path = os.path.join(d, 'thermo.yaml')
                self.assertTrue(os.path.isfile(yaml_path), 'thermo.yaml must be written from output.py')
                content = read_yaml_file(yaml_path)
            finally:
                os.chdir(cwd)
            entries = sat._load_thermo_entries_from_output_py(
                output_path, {'ThermoData': ThermoData, 'Wilhoit': Wilhoit,
                              'NASAPolynomial': NASAPolynomial, 'NASA': NASA})
        self.assertEqual(set(content), {'R1', 'R2', 'P1'})
        for label in ('R1', 'R2', 'P1'):
            entry = content[label]
            self.assertIsNotNone(entry['H298'])
            self.assertIsNotNone(entry['S298'])
            self.assertIsNotNone(entry['data'])
            self.assertIsNotNone(entry['nasa_low'])
            self.assertIsNotNone(entry['nasa_high'])
            self.assertEqual(len(entry['nasa_low']['coeffs']), 7)
        self.assertAlmostEqual(content['R1']['H298'], content['R2']['H298'])
        self.assertAlmostEqual(content['R1']['S298'], content['R2']['S298'])
        self.assertEqual(content['R1']['nasa_low'], content['R2']['nasa_low'])
        self.assertEqual(content['R1']['nasa_high'], content['R2']['nasa_high'])
        self.assertAlmostEqual(content['P1']['H298'], -241.8, delta=3.0)
        self.assertAlmostEqual(content['P1']['S298'], 188.8, delta=2.0)
        self.assertLess(content['P1']['H298'], content['R1']['H298'])
        self.assertTrue(all(cp['cp_j_mol_k'] > 0 for cp in content['P1']['cp_data']))
        h_increment = (entries['P1'].get_enthalpy(2000.0) - entries['P1'].get_enthalpy(298.15)) / 1000.0
        self.assertAlmostEqual(h_increment, 72.79, delta=3.0)
        self.assertAlmostEqual(entries['P1'].get_entropy(2000.0), 264.77, delta=3.0)
        self.assertGreater(content['P1']['nasa_high']['tmax_k'], 2000.0)

    def test_library_path_takes_precedence(self):
        """When RMG_libraries/thermo.py exists, main() uses it (happy path), not output.py."""
        import save_arkane_thermo as sat
        from common import read_yaml_file
        library_py = '''#!/usr/bin/env python
name = "test"
shortDesc = ""
longDesc = """"""
entry(
    index = 0,
    label = "OnlyFromLibrary",
    molecule = """
1 O u1 p2 c0 {2,S}
2 H u0 p0 c0 {1,S}
""",
    thermo = NASA(
        polynomials = [
            NASAPolynomial(coeffs=[3.49683, 0.000188285, -1.03135e-06, 1.63951e-09, -6.45157e-13, 2675.74, 1.48391],
                           Tmin=(10, 'K'), Tmax=(974.045, 'K')),
            NASAPolynomial(coeffs=[3.44056, -0.000267412, 7.28022e-07, -2.88523e-10, 3.54839e-14, 2719.28, 1.92114],
                           Tmin=(974.045, 'K'), Tmax=(3000, 'K')),
        ],
        Tmin=(10, 'K'), Tmax=(3000, 'K'),
    ),
)
'''
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'RMG_libraries'))
            with open(os.path.join(d, 'RMG_libraries', 'thermo.py'), 'w') as f:
                f.write(library_py)
            with open(os.path.join(d, 'output.py'), 'w') as f:
                f.write(OUTPUT_PY)
            try:
                os.chdir(d)
                sat.main()
                content = read_yaml_file(os.path.join(d, 'thermo.yaml'))
            finally:
                os.chdir(cwd)
        self.assertIn('OnlyFromLibrary', content)
        self.assertNotIn('R1', content)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
