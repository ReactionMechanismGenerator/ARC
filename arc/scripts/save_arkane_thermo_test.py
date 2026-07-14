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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for 'common' and the module under test

try:
    from rmgpy.thermo import NASA  # noqa: F401
    HAS_RMG = True
except ImportError:
    HAS_RMG = False


# A trimmed A+A Arkane thermo output.py: R1 and R2 are the two identical reactants (e.g. OH),
# P1 the distinct product. This is exactly the file Arkane leaves behind when its save_thermo_lib
# crashes on the R1==R2 duplicate, so RMG_libraries/thermo.py is never written.
OUTPUT_PY = """#!/usr/bin/env python
conformer(label='R1', E0=(22.2577, 'kJ/mol'), modes=[], spin_multiplicity=2, optical_isomers=1)
conformer(label='R2', E0=(22.2577, 'kJ/mol'), modes=[], spin_multiplicity=2, optical_isomers=1)
conformer(label='P1', E0=(-250.574, 'kJ/mol'), modes=[], spin_multiplicity=1, optical_isomers=1)

thermo(
    label = 'R1',
    thermo = NASA(
        polynomials = [
            NASAPolynomial(coeffs=[3.49683, 0.000188285, -1.03135e-06, 1.63951e-09, -6.45157e-13, 2675.74, 1.48391],
                           Tmin=(10, 'K'), Tmax=(974.045, 'K')),
            NASAPolynomial(coeffs=[3.44056, -0.000267412, 7.28022e-07, -2.88523e-10, 3.54839e-14, 2719.28, 1.92114],
                           Tmin=(974.045, 'K'), Tmax=(3000, 'K')),
        ],
        Tmin=(10, 'K'), Tmax=(3000, 'K'),
        E0=(22.2464, 'kJ/mol'), Cp0=(29.1007, 'J/(mol*K)'), CpInf=(37.4151, 'J/(mol*K)'),
    ),
)

thermo(
    label = 'R2',
    thermo = NASA(
        polynomials = [
            NASAPolynomial(coeffs=[3.49683, 0.000188285, -1.03135e-06, 1.63951e-09, -6.45157e-13, 2675.74, 1.48391],
                           Tmin=(10, 'K'), Tmax=(974.045, 'K')),
            NASAPolynomial(coeffs=[3.44056, -0.000267412, 7.28022e-07, -2.88523e-10, 3.54839e-14, 2719.28, 1.92114],
                           Tmin=(974.045, 'K'), Tmax=(3000, 'K')),
        ],
        Tmin=(10, 'K'), Tmax=(3000, 'K'),
        E0=(22.2464, 'kJ/mol'), Cp0=(29.1007, 'J/(mol*K)'), CpInf=(37.4151, 'J/(mol*K)'),
    ),
)

thermo(
    label = 'P1',
    thermo = NASA(
        polynomials = [
            NASAPolynomial(coeffs=[4.00485, -0.000245998, 8.95339e-07, 1.40307e-09, -1.18107e-12, -30136.7, -0.104547],
                           Tmin=(10, 'K'), Tmax=(772.675, 'K')),
            NASAPolynomial(coeffs=[3.50315, 0.00113242, 5.85407e-07, -3.70913e-10, 5.33992e-14, -30022.8, 2.42188],
                           Tmin=(772.675, 'K'), Tmax=(3000, 'K')),
        ],
        Tmin=(10, 'K'), Tmax=(3000, 'K'),
        E0=(-250.569, 'kJ/mol'), Cp0=(33.2579, 'J/(mol*K)'), CpInf=(58.2013, 'J/(mol*K)'),
    ),
)
"""


@unittest.skipUnless(HAS_RMG, 'requires rmgpy (rmg_env)')
class TestOutputPyThermoFallback(unittest.TestCase):
    """The output.py fallback recovers thermo when RMG_libraries/thermo.py is absent."""

    def test_iter_thermo_calls_finds_each_block(self):
        import save_arkane_thermo as sat
        calls = sat._iter_thermo_calls(OUTPUT_PY)
        self.assertEqual(len(calls), 3)  # R1, R2, P1 — not the inner thermo= kwarg
        self.assertTrue(all(c.startswith('thermo(') for c in calls))

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
        # The duplicate reactants carry identical thermo.
        self.assertAlmostEqual(entries['R1'].get_enthalpy(298.15), entries['R2'].get_enthalpy(298.15))

    def test_main_writes_thermo_yaml_from_output_py(self):
        """With output.py present but no RMG_libraries/thermo.py, main() still writes thermo.yaml
        for every species (including the duplicate R2)."""
        import save_arkane_thermo as sat
        from common import read_yaml_file
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'output.py'), 'w') as f:
                f.write(OUTPUT_PY)
            try:
                os.chdir(d)
                sat.main()
                yaml_path = os.path.join(d, 'thermo.yaml')
                self.assertTrue(os.path.isfile(yaml_path), 'thermo.yaml must be written from output.py')
                content = read_yaml_file(yaml_path)
            finally:
                os.chdir(cwd)
        self.assertEqual(set(content), {'R1', 'R2', 'P1'})
        for label in ('R1', 'R2', 'P1'):
            entry = content[label]
            self.assertIsNotNone(entry['H298'])
            self.assertIsNotNone(entry['S298'])
            self.assertIsNotNone(entry['data'])
            self.assertIsNotNone(entry['nasa_low'])
            self.assertIsNotNone(entry['nasa_high'])
            self.assertEqual(len(entry['nasa_low']['coeffs']), 7)
        # Duplicate reactants -> identical recovered thermo.
        self.assertAlmostEqual(content['R1']['H298'], content['R2']['H298'])
        self.assertAlmostEqual(content['R1']['S298'], content['R2']['S298'])

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
        self.assertIn('OnlyFromLibrary', content)   # library was used
        self.assertNotIn('R1', content)             # output.py was NOT used


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
