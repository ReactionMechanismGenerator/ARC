#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for ARC's statmech.arkane module
"""

import os
import shutil
import unittest

from arc.common import ARC_PATH
from arc.level import Level
from arc.reaction import ARCReaction
from arc.species import ARCSpecies
from arc.statmech.adapter import StatmechEnum
from arc.statmech.arkane import ArkaneAdapter
from arc.statmech.arkane import _level_to_str, _section_contains_key, get_arkane_model_chemistry


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
        output_path_1 = os.path.join(ARC_PATH, 'arc', 'testing', 'arkane_tests_delete', 'output_1')
        calcs_path_1 = os.path.join(ARC_PATH, 'arc', 'testing', 'arkane_tests_delete', 'calcs_1')
        output_path_2 = os.path.join(ARC_PATH, 'arc', 'testing', 'arkane_tests_delete', 'output_2')
        calcs_path_2 = os.path.join(ARC_PATH, 'arc', 'testing', 'arkane_tests_delete', 'calcs_2')
        output_path_3 = os.path.join(ARC_PATH, 'arc', 'testing', 'arkane_tests_delete', 'output_3')
        calcs_path_3 = os.path.join(ARC_PATH, 'arc', 'testing', 'arkane_tests_delete', 'calcs_3')
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
        opt_path = os.path.join(ARC_PATH, 'arc', 'testing', 'opt', 'iC3H7.out')
        freq_path = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'iC3H7.out')
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
        self.assertTrue(os.path.isfile(os.path.join(ARC_PATH, 'arc', 'testing', 'arkane_tests_delete', 'calcs_3',
                                                    'statmech', 'thermo', 'plots', 'iC3H7.pdf')))
        self.assertAlmostEqual(self.ic3h7.e0, 6.75565e+07)

    def test_level_to_str(self):
        """Test the _level_to_str function"""
        self.assertEqual(_level_to_str(Level('gfn2')),
                         "LevelOfTheory(method='gfn2',software='xtb')")
        self.assertEqual(_level_to_str(Level(method='b3lyp', basis='6-31g(d)')),
                         "LevelOfTheory(method='b3lyp',basis='631g(d)',software='gaussian')")
        self.assertEqual(_level_to_str(Level(method='CCSD(T)-F12', basis='cc-pVTZ-F12')),
                         "LevelOfTheory(method='ccsd(t)f12',basis='ccpvtzf12',software='molpro')")

    def test_section_contains_key(self):
        """Test the _section_contains_key function"""
        file_path = os.path.join(os.path.dirname(ARC_PATH), 'RMG-database', 'input', 'quantum_corrections', 'data.py')
        self.assertTrue(_section_contains_key(file_path=file_path,
                                              section_start="atom_energies = {",
                                              section_end="pbac = {",
                                              target="LevelOfTheory(method='b97d32023',basis='def2tzvp',software='gaussian')"))
        self.assertTrue(_section_contains_key(file_path=file_path,
                                              section_start="atom_energies = {",
                                              section_end="pbac = {",
                                              target="LevelOfTheory(method='ccsd(t)f12',basis='ccpvtzf12',software='molpro')"))
        self.assertFalse(_section_contains_key(file_path=file_path,
                                               section_start="atom_energies = {",
                                               section_end="pbac = {",
                                               target="LevelOfTheory(method=imaginary',basis='basis',software='ess')"))

    def test_get_arkane_model_chemistry(self):
        """Test the get_arkane_model_chemistry function"""
        self.assertEqual(get_arkane_model_chemistry(sp_level=Level(method='CCSD(T)-F12', basis='cc-pVTZ-F12'),
                                                    freq_scale_factor=1.0),
                         "LevelOfTheory(method='ccsd(t)f12',basis='ccpvtzf12',software='molpro')")
        self.assertEqual(get_arkane_model_chemistry(sp_level=Level(method='CBS-QB3'),
                                                    freq_scale_factor=1.0),
                         "LevelOfTheory(method='cbs-qb3',software='gaussian')")

    def test_generate_arkane_input(self):
        """Test generating Arkane input"""
        statmech_dir = os.path.join(ARC_PATH, 'arc', 'testing', 'arkane_input_tests_delete')
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
        for folder in ['arkane_tests_delete', 'arkane_input_tests_delete']:
            shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', folder), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
