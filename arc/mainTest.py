#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the arc.main module
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest
import os
import shutil

from arc.main import ARC
from arc.species import ARCSpecies
from arc.settings import arc_path
from arc.exceptions import InputError

################################################################################


class TestARC(unittest.TestCase):
    """
    Contains unit tests for the ARC class
    """

    def test_as_dict(self):
        """Test the as_dict() method of ARC"""
        self.maxDiff = None
        ess_settings = {}
        spc1 = ARCSpecies(label='spc1', smiles=str('CC'), generate_thermo=False)
        arc0 = ARC(project='arc_test', ess_settings=ess_settings, scan_rotors=False, initial_trsh='scf=(NDump=30)',
                   arc_species_list=[spc1])
        restart_dict = arc0.as_dict()
        expected_dict = {'composite_method': '',
                         'conformer_level': 'b97-d3/6-311+g(d,p)',
                         'ts_guess_level': 'b3lyp/6-311+g(d,p)',
                         'ess_settings': {'ssh': True},
                         'fine': True,
                         'opt_level': 'wb97xd/6-311++g(d,p)',
                         'freq_level': 'wb97xd/6-311++g(d,p)',
                         'generate_conformers': True,
                         'initial_trsh': 'scf=(NDump=30)',
                         'model_chemistry': 'ccsd(t)-f12/cc-pvtz-f12',
                         'output': {},
                         'project': 'arc_test',
                         'running_jobs': {},
                         'reactions': [],
                         'scan_level': '',
                         'scan_rotors': False,
                         'sp_level': 'ccsd(t)-f12/cc-pvtz-f12',
                         't_min': None,
                         't_max': None,
                         't_count': None,
                         'use_bac': True,
                         'species': [{'bond_corrections': {'C-C': 1, 'C-H': 6},
                                      'arkane_file': None,
                                      'E0': None,
                                      'charge': 0,
                                      'external_symmetry': None,
                                      'optical_isomers': None,
                                      'final_xyz': '',
                                      'generate_thermo': False,
                                      'is_ts': False,
                                      'label': u'spc1',
                                      'long_thermo_description': "Bond corrections: {'C-C': 1, 'C-H': 6}\n",
                                      'mol': '1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}\n2 C u0 p0 c0 {1,S} {6,S} {7,S} {8,S}\n3 H u0 p0 c0 {1,S}\n4 H u0 p0 c0 {1,S}\n5 H u0 p0 c0 {1,S}\n6 H u0 p0 c0 {2,S}\n7 H u0 p0 c0 {2,S}\n8 H u0 p0 c0 {2,S}\n',
                                      'multiplicity': 1,
                                      'neg_freqs_trshed': [],
                                      'number_of_rotors': 0,
                                      'opt_level': '',
                                      'rotors_dict': {},
                                      't0': None,
                                      't1': None}],
                         }
        self.assertEqual(restart_dict, expected_dict)

    def test_from_dict(self):
        """Test the from_dict() method of ARC"""
        restart_dict = {'composite_method': '',
                         'conformer_level': 'b97-d3/6-311+g(d,p)',
                         'ess_settings': {'gaussian': 'pharos',
                                          'molpro': 'pharos',
                                          'qchem': u'pharos',
                                          'ssh': True},
                         'fine': True,
                         'freq_level': 'wb97x-d3/6-311+g(d,p)',
                         'generate_conformers': True,
                         'initial_trsh': 'scf=(NDump=30)',
                         'model_chemistry': 'ccsd(t)-f12/cc-pvtz-f12',
                         'opt_level': 'wb97x-d3/6-311+g(d,p)',
                         'output': {},
                         'project': 'arc_test',
                         'rxn_list': [],
                         'scan_level': '',
                         'scan_rotors': False,
                         'sp_level': 'ccsdt-f12/cc-pvqz-f12',
                         'species': [{'bond_corrections': {'C-C': 1, 'C-H': 6},
                                      'charge': 1,
                                      'conformer_energies': [],
                                      'conformers': [],
                                      'external_symmetry': 1,
                                      'final_xyz': '',
                                      'generate_thermo': False,
                                      'is_ts': False,
                                      'label': 'testing_spc1',
                                      'mol': '1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}\n2 C u0 p0 c0 {1,S} {6,S} {7,S} {8,S}\n3 H u0 p0 c0 {1,S}\n4 H u0 p0 c0 {1,S}\n5 H u0 p0 c0 {1,S}\n6 H u0 p0 c0 {2,S}\n7 H u0 p0 c0 {2,S}\n8 H u0 p0 c0 {2,S}\n',
                                      'multiplicity': 1,
                                      'neg_freqs_trshed': [],
                                      'number_of_rotors': 0,
                                      'opt_level': '',
                                      'optical_isomers': 1,
                                      'rotors_dict': {},
                                      'xyzs': []}],
                         'use_bac': True}
        arc1 = ARC(project='wrong', ess_settings=dict())
        project = 'arc_project_for_testing_delete_after_usage'
        project_directory = os.path.join(arc_path, 'Projects', project)
        arc1.from_dict(input_dict=restart_dict, project='testing_from_dict', project_directory=project_directory)
        self.assertEqual(arc1.project, 'testing_from_dict')
        self.assertTrue('arc_project_for_testing_delete_after_usage' in arc1.project_directory)
        self.assertTrue(arc1.fine)
        self.assertFalse(arc1.scan_rotors)
        self.assertEqual(arc1.sp_level, 'ccsdt-f12/cc-pvqz-f12')
        self.assertEqual(arc1.arc_species_list[0].label, 'testing_spc1')
        self.assertFalse(arc1.arc_species_list[0].is_ts)
        self.assertEqual(arc1.arc_species_list[0].charge, 1)

    def test_check_project_name(self):
        """Test project name validity"""
        ess_settings = {}
        with self.assertRaises(InputError):
            ARC(project='ar c', ess_settings=ess_settings)
        with self.assertRaises(InputError):
            ARC(project='ar:c', ess_settings=ess_settings)
        with self.assertRaises(InputError):
            ARC(project='ar<c', ess_settings=ess_settings)
        with self.assertRaises(InputError):
            ARC(project='ar%c', ess_settings=ess_settings)

    @classmethod
    def tearDownClass(cls):
        """A function that is run ONCE after all unit tests in this class."""
        project = 'arc_project_for_testing_delete_after_usage'
        project_directory = os.path.join(arc_path, 'Projects', project)
        shutil.rmtree(project_directory)
        projects = ['ar c', 'ar:c', 'ar<c', 'ar%c']
        for project in projects:
            project_directory = os.path.join(arc_path, 'Projects', project)
            shutil.rmtree(project_directory)

################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
