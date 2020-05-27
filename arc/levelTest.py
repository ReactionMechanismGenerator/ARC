"""
This module contains unit tests for the arc.lot module
"""

import os
import unittest

from arkane.modelchem import LevelOfTheory

from arc.common import read_yaml_file
from arc.level import Level
from arc.settings import arc_path


class TestLevel(unittest.TestCase):
    """
    Contains unit tests for the ARC Level class
    """
    def test_level(self):
        """Test setting up Level"""
        level_1 = Level(method='b3lyp', basis='def2tzvp', auxiliary_basis='aug-def2-svp',
                        args='gd3bj', software='gaussian')
        self.assertEqual(level_1.method, 'b3lyp')
        self.assertEqual(level_1.basis, 'def2tzvp')
        self.assertEqual(level_1.auxiliary_basis, 'aug-def2-svp')
        self.assertEqual(level_1.args, ('gd3bj',))
        self.assertEqual(level_1.software, 'gaussian')
        self.assertEqual(str(level_1),
                         "b3lyp/def2tzvp, auxiliary_basis: aug-def2-svp, software: gaussian, args: ('gd3bj',) (dft)")

    def test_deduce_software(self):
        """Test deducing an ESS by the level"""
        self.assertEqual(Level(method='B3LYP', basis='6-311g+(d,f)').software, 'gaussian')
        level_1 = Level(method='B3LYP/6-311g+(d,f)')
        level_1.deduce_software(job_type='onedmin')
        self.assertEqual(level_1.software, 'onedmin')
        self.assertEqual(Level(method='b3lyp', basis='6-311g+(d,f)').software, 'gaussian')
        level_2 = Level(method='b3lyp', basis='6-311g+(d,f)')
        level_2.deduce_software(job_type='orbitals')
        self.assertEqual(level_2.software, 'qchem')
        self.assertEqual(Level(method='B3LYP', basis='6-311g+(d,f)').software, 'gaussian')
        level_3 = Level(method='B3LYP', basis='6-311g+(d,f)')
        level_3.deduce_software(job_type='irc')
        self.assertEqual(level_3.software, 'gaussian')
        self.assertEqual(Level(method='DLPNO-CCSD(T)', basis='def2-tzvp').software, 'orca')
        self.assertEqual(Level(method='PM6').software, 'gaussian')
        self.assertEqual(Level(method='HF').software, 'gaussian')
        self.assertEqual(Level(method='CCSD(T)-F12', basis='aug-cc-pVTZ').software, 'molpro')
        self.assertEqual(Level(method='CISD', basis='aug-cc-pVTZ').software, 'molpro')
        self.assertEqual(Level(method='b3lyp', basis='6-311g+(d,f)').software, 'gaussian')
        self.assertEqual(Level(method='wb97x-d', basis='def2-tzvp').software, 'qchem')
        self.assertEqual(Level(method='wb97xd', basis='def2-tzvp').software, 'gaussian')
        self.assertEqual(Level(method='b97', basis='def2-tzvp').software, 'gaussian')
        self.assertEqual(Level(method='m06-2x', basis='def2-tzvp').software, 'qchem')
        self.assertEqual(Level(method='m062x', basis='def2-tzvp').software, 'gaussian')
        self.assertEqual(Level(method='b3p86', basis='6-311g+(d,f)').software, 'terachem')

    def test_build(self):
        """Test bulding a Level object from a string or dict representation"""
        level_1 = Level(repr='wB97xd/def2-tzvp')
        self.assertEqual(level_1.method, 'wb97xd')
        self.assertEqual(level_1.basis, 'def2-tzvp')
        self.assertEqual(level_1.method_type, 'dft')
        self.assertEqual(str(level_1), "wb97xd/def2-tzvp, software: gaussian (dft)")
        level_2 = Level(repr='CBS-QB3')
        self.assertEqual(level_2.method, 'cbs-qb3')
        self.assertIsNone(level_2.basis)
        self.assertEqual(level_2.software, 'gaussian')
        self.assertEqual(level_2.method_type, 'composite')
        self.assertEqual(str(level_2), "cbs-qb3, software: gaussian (composite)")
        level_3 = Level(repr={'method': 'DLPNO-CCSD(T)',
                              'basis': 'def2-TZVp',
                              'auxiliary_basis': 'def2-tzvp/c',
                              'solvation_method': 'SMD',
                              'solvent': 'water',
                              'solvation_scheme_level': 'APFD/def2-TZVp'})
        self.assertEqual(level_3.method, 'dlpno-ccsd(t)')
        self.assertEqual(level_3.basis, 'def2-tzvp')
        self.assertEqual(level_3.auxiliary_basis, 'def2-tzvp/c')
        self.assertEqual(level_3.solvation_method, 'smd')
        self.assertEqual(level_3.solvent, 'water')
        self.assertIsInstance(level_3.solvation_scheme_level, Level)
        self.assertEqual(level_3.solvation_scheme_level.method, 'apfd')
        self.assertEqual(level_3.solvation_scheme_level.basis, 'def2-tzvp')
        self.assertEqual(str(level_3),
                         "dlpno-ccsd(t)/def2-tzvp, auxiliary_basis: def2-tzvp/c, solvation_method: smd, "
                         "solvent: water, solvation_scheme_level: 'apfd/def2-tzvp, software: gaussian (dft)', "
                         "software: orca (wavefunction)")

    def test_to_arkane(self):
        level_1 = Level(repr='wB97xd/def2-tzvp')
        self.assertEqual(level_1.to_arkane_level_of_theory(),
                         LevelOfTheory(method='wb97xd', basis='def2tzvp', software='gaussian'))
        level_2 = Level(repr='CBS-QB3')
        self.assertEqual(level_2.to_arkane_level_of_theory(),
                         LevelOfTheory(method='cbs-qb3', software='gaussian'))
        level_3 = Level(repr={'method': 'DLPNO-CCSD(T)',
                              'basis': 'def2-TZVp',
                              'auxiliary_basis': 'def2-tzvp/c',
                              'solvation_method': 'SMD',
                              'solvent': 'water',
                              'solvation_scheme_level': 'APFD/def2-TZVp'})
        self.assertEqual(level_3.to_arkane_level_of_theory(),
                         LevelOfTheory(method='dlpno-ccsd(t)', basis='def2tzvp', auxiliary_basis='def2tzvp/c',
                                       solvation_method='SMD', solvent='water', software='orca'))

    def test_ess_methods_yml(self):
        """Test reading the ess_methods.yml file"""
        ess_methods = read_yaml_file(path=os.path.join(arc_path, 'data', 'ess_methods.yml'))
        self.assertIsInstance(ess_methods, dict)
        for ess, methods in ess_methods.items():
            self.assertIsInstance(ess, str)
            for method in methods:
                self.assertIsInstance(method, str)
        self.assertIn('gaussian', list(ess_methods.keys()))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
