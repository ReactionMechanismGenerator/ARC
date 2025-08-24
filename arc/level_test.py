#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.lot module
"""

import os
import unittest

from arc.common import ARC_PATH, read_yaml_file
from arc.level import Level, assign_frequency_scale_factor


class TestLevel(unittest.TestCase):
    """
    Contains unit tests for the ARC Level class
    """

    def test_level(self):
        """Test setting up Level"""
        level_1 = Level(method='b3lyp', basis='def2tzvp', auxiliary_basis='aug-def2-svp',
                        dispersion='gd3bj', software='gaussian')
        self.assertEqual(level_1.method, 'b3lyp')
        self.assertEqual(level_1.basis, 'def2tzvp')
        self.assertEqual(level_1.auxiliary_basis, 'aug-def2-svp')
        self.assertEqual(level_1.dispersion, 'gd3bj')
        self.assertEqual(level_1.software, 'gaussian')
        self.assertEqual(str(level_1),
                         "b3lyp/def2tzvp, auxiliary_basis: aug-def2-svp, dispersion: gd3bj, software: gaussian")
        self.assertEqual(level_1.simple(), "b3lyp/def2tzvp")

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
        self.assertEqual(Level(method='new', basis='new', args={'keywords': {'general': 'iop(99/33=1)'}}).software,
                         'gaussian')

    def test_lower(self):
        """Test the Level.lower() method"""
        level = Level(method='B3LYP',
                      basis='6-311+G(3df,2p)',
                      software='Gaussian',
                      dispersion='empiricaldispersion=Gd3bj',
                      solvation_method='SMD',
                      solvent='Water',
                      args={'KeyWord': 'IOP(99/33=1)'})
        self.assertEqual(level.method, 'b3lyp')
        self.assertEqual(level.basis, '6-311+g(3df,2p)')
        self.assertEqual(level.software, 'gaussian')
        self.assertEqual(level.dispersion, 'empiricaldispersion=gd3bj')
        self.assertEqual(level.solvation_method, 'smd')
        self.assertEqual(level.solvent, 'water')
        self.assertEqual(level.args, {'keyword': {'general': 'iop(99/33=1)'}, 'block': {}})

    def test_equal(self):
        """Test identifying equal levels."""
        level_1 = Level(method='b3lyp', basis='def2tzvp', auxiliary_basis='aug-def2-svp',
                        dispersion='gd3bj', software='gaussian')
        level_2 = Level(method='b3lyp', basis='def2tzvp', auxiliary_basis='aug-def2-svp',
                        dispersion='gd3bj', software='gaussian')
        level_3 = Level(method='wb97xd', basis='def2tzvp', auxiliary_basis='aug-def2-svp',
                        dispersion='gd3bj', software='gaussian')
        self.assertEqual(level_1, level_2)
        self.assertNotEqual(level_1, level_3)

    def test_build(self):
        """Test building a Level object from a string or dict representation"""
        level_1 = Level(repr='wB97xd/def2-tzvp')
        self.assertEqual(level_1.method, 'wb97xd')
        self.assertEqual(level_1.basis, 'def2-tzvp')
        self.assertEqual(level_1.method_type, 'dft')
        self.assertEqual(str(level_1), 'wb97xd/def2-tzvp, software: gaussian')
        level_2 = Level(repr='CBS-QB3')
        self.assertEqual(level_2.method, 'cbs-qb3')
        self.assertIsNone(level_2.basis)
        self.assertEqual(level_2.software, 'gaussian')
        self.assertEqual(level_2.method_type, 'composite')
        self.assertEqual(str(level_2), 'cbs-qb3, software: gaussian')
        self.assertEqual(level_2.simple(), 'cbs-qb3')
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
                         "solvent: water, solvation_scheme_level: 'apfd/def2-tzvp, software: gaussian', software: orca")

    def test_ess_methods_yml(self):
        """Test reading the ess_methods.yml file"""
        ess_methods = read_yaml_file(path=os.path.join(ARC_PATH, 'data', 'ess_methods.yml'))
        self.assertIsInstance(ess_methods, dict)
        for ess, methods in ess_methods.items():
            self.assertIsInstance(ess, str)
            for method in methods:
                self.assertIsInstance(method, str)
        self.assertIn('gaussian', list(ess_methods.keys()))

    def test_copy(self):
        """Test copying the object"""
        level_1 = Level(repr='wB97xd/def2-tzvp')
        level_2 = level_1.copy()
        self.assertIsNot(level_1, level_2)
        self.assertEqual(level_1.as_dict(), level_2.as_dict())

    def test_determine_compatible_ess(self):
        """Test the determine_compatible_ess() method."""
        level_1 = Level(method='CCSD(T)', basis='cc-pvtz')
        self.assertIsNone(level_1.compatible_ess)
        level_1.determine_compatible_ess()
        self.assertEqual(sorted(level_1.compatible_ess), sorted(['cfour', 'gaussian', 'molpro']))

        level_2 = Level(method='B3LYP', basis='6-311(d,p)')
        self.assertIsNone(level_2.compatible_ess)
        level_2.determine_compatible_ess()
        self.assertEqual(sorted(level_2.compatible_ess), sorted(['gaussian', 'qchem', 'terachem']))

    def test_str(self):
        """Test the __str__() method."""
        self.assertEqual(str(Level(method='HF', basis='6-31g')), 'hf/6-31g, software: gaussian')
        self.assertEqual(str(Level(method='HF', basis='6-31+g(d,p)')), 'hf/6-31+g(d,p), software: gaussian')
        self.assertEqual(str(Level(method='PM6')), 'pm6, software: gaussian')
        self.assertEqual(str(Level(method='b3lyp', basis='6-311+g(d,p)')), 'b3lyp/6-311+g(d,p), software: gaussian')
        self.assertEqual(str(Level(method='M062x', basis='Aug-cc-pvdz')), 'm062x/aug-cc-pvdz, software: gaussian')
        self.assertEqual(str(Level(method='M062x', basis='Def2TZVP')), 'm062x/def2tzvp, software: gaussian')
        self.assertEqual(str(Level(method='M06-2x', basis='Def2-TZVP')), 'm06-2x/def2-tzvp, software: qchem')
        self.assertEqual(str(Level(method='wb97xd', basis='def2tzvp')), 'wb97xd/def2tzvp, software: gaussian')
        self.assertEqual(str(Level(method='wb97m-v', basis='def2-tzvpd')), 'wb97m-v/def2-tzvpd, software: qchem')
        self.assertEqual(str(Level(method='b2plypd3', basis='cc-pvtz')), 'b2plypd3/cc-pvtz, software: gaussian')
        self.assertEqual(str(Level(method='apfd', basis='def2tzvp')), 'apfd/def2tzvp, software: gaussian')
        self.assertEqual(str(Level(method='mp2', basis='cc-pvdz')), 'mp2/cc-pvdz, software: gaussian')
        self.assertEqual(str(Level(method='CBS-QB3')), 'cbs-qb3, software: gaussian')
        self.assertEqual(str(Level(method='CCSD(T)', basis='cc-pVTZ')), 'ccsd(t)/cc-pvtz, software: molpro')
        self.assertEqual(str(Level(method='CCSD(T)-F12', basis='cc-pVTZ-F12')), 'ccsd(t)-f12/cc-pvtz-f12, software: molpro')
        self.assertEqual(str(Level(method='wb97xd', basis='def2tzvp', solvation_method='smd', solvent='water')),
                         'wb97xd/def2tzvp, solvation_method: smd, solvent: water, software: gaussian')
        self.assertEqual(str(Level(basis='def2svp', compatible_ess=['gaussian', 'terachem'],method='wb97xd',
                                   method_type='dft', software='gaussian')), 'wb97xd/def2svp, software: gaussian')

    def test_assign_frequency_scale_factor(self):
        """Test the assign_frequency_scale_factor() method."""
        self.assertEqual(assign_frequency_scale_factor(Level(method='CCSD(T)', basis='cc-pvtz')), 0.975)
        self.assertEqual(assign_frequency_scale_factor(Level(method='CCSD(T)-F12', basis='cc-pvtz-F12')), 0.998)
        self.assertEqual(assign_frequency_scale_factor(Level(method='wb97xd', basis='Def2TZVP')), 0.988)
        self.assertEqual(assign_frequency_scale_factor(Level(method='CBS-QB3')), 1.004)
        self.assertEqual(assign_frequency_scale_factor(Level(method='PM6')), 1.093)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
