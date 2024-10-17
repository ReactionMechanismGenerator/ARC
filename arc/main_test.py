#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.main module
"""

import os
import shutil
import unittest

from arc.common import ARC_PATH
from arc.exceptions import InputError
from arc.imports import settings
from arc.level import Level
from arc.main import ARC, StatmechEnum, process_adaptive_levels
from arc.species.species import ARCSpecies

servers = settings['servers']


class TestEnumerationClasses(unittest.TestCase):
    """
    Contains unit tests for various enumeration classes.
    """

    def test_statmech_enum(self):
        """Test the StatmechEnum class"""
        self.assertEqual(StatmechEnum('arkane').value, 'arkane')
        with self.assertRaises(ValueError):
            StatmechEnum('wrong')


class TestARC(unittest.TestCase):
    """
    Contains unit tests for the ARC class
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.servers = servers.keys()
        cls.job_types1 = {'conf_opt': True,
                          'opt': True,
                          'fine_grid': False,
                          'freq': True,
                          'sp': True,
                          'conf_sp': False,
                          'rotors': False,
                          'orbitals': False,
                          'lennard_jones': False,
                          'bde': True,
                          }
        projects = ['arc_project_for_testing_delete_after_usage_test_from_dict',
                    'arc_model_chemistry_test', 'arc_test', 'test', 'unit_test_specific_job', 'wrong']
        for project in projects:
            project_directory = os.path.join(ARC_PATH, 'Projects', project)
            if os.path.isdir(project_directory):
                shutil.rmtree(project_directory, ignore_errors=True)

    def test_as_dict(self):
        """Test the as_dict() method of ARC"""
        spc1 = ARCSpecies(label='spc1',
                          smiles='CC',
                          compute_thermo=False,
                          )
        arc0 = ARC(project='arc_test',
                   job_types=self.job_types1,
                   species=[spc1],
                   level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)',
                   three_params=False,
                   ts_adapters=['heuristics', 'AutoTST', 'GCN', 'xtb_gsm'],
                   )
        arc0.freq_level.args['keyword']['general'] = 'scf=(NDamp=30)'
        restart_dict = arc0.as_dict()
        long_thermo_description = restart_dict['species'][0]['long_thermo_description']
        self.assertIn('Bond corrections:', long_thermo_description)
        self.assertIn("'C-C': 1", long_thermo_description)
        self.assertIn("'C-H': 6", long_thermo_description)
        # mol.atoms are not tested since all id's (including connectivity) changes depending on how the test is run.
        expected_dict = {'arkane_level_of_theory': {'basis': 'cc-pvdz-f12',
                                                    'method': 'ccsd(t)-f12',
                                                    'method_type': 'wavefunction',
                                                    'software': 'molpro'},
                         'conformer_opt_level': {'basis': 'def2svp',
                                             'compatible_ess': ['gaussian', 'terachem'],
                                             'method': 'wb97xd',
                                             'method_type': 'dft',
                                             'software': 'gaussian'},
                         'ess_settings': {'cfour': ['local'],
                                          'gaussian': ['local', 'server2'],
                                          'gcn': ['local'],
                                          'mockter': ['local'],
                                          'molpro': ['local', 'server2'],
                                          'onedmin': ['server1'],
                                          'openbabel': ['local'],
                                          'orca': ['local'],
                                          'qchem': ['server1'],
                                          'terachem': ['server1'],
                                          'torchani': ['local'],
                                          'xtb': ['local'],
                                          'xtb_gsm': ['local'],
                                          },
                         'freq_level': {'basis': '6-311+g(3df,2p)',
                                        'method': 'b3lyp',
                                        'method_type': 'dft',
                                        'software': 'gaussian'},
                         'freq_scale_factor': 0.967,
                         'irc_level': {'basis': '6-311+g(3df,2p)',
                                       'method': 'b3lyp',
                                       'method_type': 'dft',
                                       'software': 'gaussian'},
                         'job_memory': 14,
                         'job_types': {'bde': True,
                                       'conf_opt': True,
                                       'conf_sp': False,
                                       'fine': False,
                                       'freq': True,
                                       'irc': True,
                                       'onedmin': False,
                                       'opt': True,
                                       'orbitals': False,
                                       'rotors': False,
                                       'sp': True},
                         'max_job_time': 120,
                         'opt_level': {'basis': '6-311+g(3df,2p)',
                                       'method': 'b3lyp',
                                       'method_type': 'dft',
                                       'software': 'gaussian'},
                         'project': 'arc_test',
                         'sp_level': {'basis': 'cc-pvdz-f12',
                                      'method': 'ccsd(t)-f12',
                                      'method_type': 'wavefunction',
                                      'software': 'molpro'},
                         'species': [{'bond_corrections': {'C-C': 1, 'C-H': 6},
                                      'compute_thermo': False,
                                      'label': 'spc1',
                                      'long_thermo_description': long_thermo_description,
                                      'mol': {'atom_order': restart_dict['species'][0]['mol']['atom_order'],
                                              'atoms': restart_dict['species'][0]['mol']['atoms'],
                                              'multiplicity': 1,
                                              'props': {}},
                                      'multiplicity': 1,
                                      'number_of_rotors': 0}],
                         'three_params': False,
                         'ts_adapters': ['heuristics', 'AutoTST', 'GCN', 'xtb_gsm']}
        # import pprint  # left intentionally for debugging
        # print(pprint.pprint(restart_dict))
        self.assertEqual(restart_dict, expected_dict)

    def test_from_dict(self):
        """Test the from_dict() method of ARC"""
        restart_dict = {'composite_method': '',
                        'conformer_opt_level': 'b97-d3/6-311+g(d,p)',
                        'freq_level': 'wb97x-d3/6-311+g(d,p)',
                        'freq_scale_factor': 0.96,
                        'opt_level': 'wb97x-d3/6-311+g(d,p)',
                        'project': 'testing_from_dict',
                        'reactions': [],
                        'scan_level': '',
                        'sp_level': 'ccsd(t)-f12/cc-pvqz-f12',
                        'species': [{'bond_corrections': {'C-C': 1, 'C-H': 6},
                                     'charge': 1,
                                     'conformer_energies': [],
                                     'conformers': [],
                                     'external_symmetry': 1,
                                     'compute_thermo': False,
                                     'is_ts': False,
                                     'label': 'testing_spc1',
                                     'mol': '1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}\n2 C u0 p0 c0 {1,S} {6,S} {7,S} {8,S}'
                                            '\n3 H u0 p0 c0 {1,S}\n4 H u0 p0 c0 {1,S}\n5 H u0 p0 c0 {1,S}\n6 H u0 p0 '
                                            'c0 {2,S}\n7 H u0 p0 c0 {2,S}\n8 H u0 p0 c0 {2,S}\n',
                                     'multiplicity': 1,
                                     'neg_freqs_trshed': [],
                                     'number_of_rotors': 0,
                                     'opt_level': '',
                                     'optical_isomers': 1,
                                     'rotors_dict': {},
                                     'xyzs': []}],
                        'three_params': False,
                        'project_directory': os.path.join(ARC_PATH, 'Projects',
                                                          'arc_project_for_testing_delete_after_usage_test_from_dict'),
                        }
        arc1 = ARC(project='wrong', freq_scale_factor=0.95)
        self.assertEqual(arc1.freq_scale_factor, 0.95)  # user input
        arc2 = ARC(**restart_dict)
        self.assertEqual(arc2.freq_scale_factor, 0.96)  # loaded from the restart dict
        self.assertEqual(arc2.project, 'testing_from_dict')
        self.assertIn('arc_project_for_testing_delete_after_usage', arc2.project_directory)
        self.assertTrue(arc2.job_types['fine'])
        self.assertTrue(arc2.job_types['rotors'])
        self.assertEqual(arc2.sp_level.simple(), 'ccsd(t)-f12/cc-pvqz-f12')
        self.assertEqual(arc2.level_of_theory, '')
        self.assertEqual(arc2.species[0].label, 'testing_spc1')
        self.assertFalse(arc2.species[0].is_ts)
        self.assertEqual(arc2.species[0].charge, 1)
        self.assertFalse(arc2.three_params)

    def test_from_dict_specific_job(self):
        """Test the from_dict() method of ARC"""
        restart_dict = {'specific_job_type': 'bde',
                        'project': 'unit_test_specific_job',
                        'project_directory': os.path.join(ARC_PATH, 'Projects', 'unit_test_specific_job'),
                        }
        arc1 = ARC(**restart_dict)
        job_type_expected = {'conf_opt': False, 'conf_sp': False, 'opt': True, 'freq': True, 'sp': True, 'rotors': False,
                             'orbitals': False, 'bde': True, 'onedmin': False, 'fine': True, 'irc': False}
        self.assertEqual(arc1.job_types, job_type_expected)

    def test_check_project_name(self):
        """Test project name invalidity"""
        with self.assertRaises(InputError):
            ARC(project='ar c')
        with self.assertRaises(InputError):
            ARC(project='ar:c')
        with self.assertRaises(InputError):
            ARC(project='ar<c')
        with self.assertRaises(InputError):
            ARC(project='ar%c')

    def test_determine_model_chemistry_and_freq_scale_factor(self):
        """Test determining the model chemistry and the frequency scaling factor"""
        arc0 = ARC(project='arc_model_chemistry_test', level_of_theory='CBS-QB3')
        self.assertEqual(str(arc0.arkane_level_of_theory), "cbs-qb3, software: gaussian")
        self.assertEqual(arc0.freq_scale_factor, 1.004)

        arc1 = ARC(project='arc_model_chemistry_test', level_of_theory='cbs-qb3-paraskevas')
        self.assertEqual(str(arc1.arkane_level_of_theory), 'cbs-qb3-paraskevas, software: gaussian')
        self.assertEqual(arc1.freq_scale_factor, 1.004)
        self.assertEqual(arc1.bac_type, 'p')

        arc2 = ARC(project='arc_model_chemistry_test',
                   level_of_theory='ccsd(t)-f12/cc-pvtz-f12//m062x/cc-pvtz')
        self.assertEqual(str(arc2.arkane_level_of_theory), 'ccsd(t)-f12/cc-pvtz-f12, software: molpro')
        self.assertEqual(arc2.freq_scale_factor, 0.955)

        arc3 = ARC(project='arc_model_chemistry_test',
                   sp_level='ccsd(t)-f12/cc-pvtz-f12', opt_level='wb97xd/def2tzvp')
        self.assertEqual(str(arc3.arkane_level_of_theory), 'ccsd(t)-f12/cc-pvtz-f12, software: molpro')
        self.assertEqual(arc3.freq_scale_factor, 0.988)

    def test_determine_model_chemistry_for_job_types(self):
        """Test determining the model chemistry specification dictionary for job types"""
        # Test conflicted inputs: specify both level_of_theory and composite_method
        with self.assertRaises(InputError):
            ARC(project='test', level_of_theory='ccsd(t)-f12/cc-pvtz-f12//wb97x-d/aug-cc-pvtz',
                composite_method='cbs-qb3')

        # Test illegal level of theory specification (method contains multiple slashes)
        with self.assertRaises(ValueError):
            ARC(project='test', level_of_theory='dlpno-mp2-f12/D/cc-pVDZ(fi/sf/fw)//b3lyp/G/def2svp')

        # Test illegal job level specification (method contains multiple slashes)
        with self.assertRaises(ValueError):
            ARC(project='test', opt_level='b3lyp/d/def2tzvp/def2tzvp/c')

        # Test illegal job level specification (method contains empty space)
        with self.assertRaises(ValueError):
            ARC(project='test', opt_level='b3lyp/def2tzvp def2tzvp/c')

        # Test direct job level specification conflicts with level of theory specification
        with self.assertRaises(InputError):
            ARC(project='test', level_of_theory='b3lyp/sto-3g', opt_level='wb97xd/def2tzvp')

        # Test deduce levels from default method from settings.py
        arc1 = ARC(project='test')
        self.assertEqual(arc1.opt_level.simple(), 'wb97xd/def2tzvp')
        self.assertEqual(arc1.freq_level.simple(), 'wb97xd/def2tzvp')
        self.assertEqual(arc1.sp_level.simple(), 'ccsd(t)-f12/cc-pvtz-f12')

        # Test deduce levels from composite method specification
        arc2 = ARC(project='test', composite_method='cbs-qb3')
        self.assertIsNotNone(arc2.opt_level)
        self.assertIsNone(arc2.sp_level)
        self.assertIsNone(arc2.orbitals_level)
        self.assertEqual(arc2.freq_level.simple(), 'b3lyp/cbsb7')
        self.assertEqual(arc2.scan_level.simple(), 'b3lyp/cbsb7')
        self.assertEqual(arc2.composite_method.simple(), 'cbs-qb3')

        # Test deduce levels from level of theory specification
        arc3 = ARC(project='test', level_of_theory='ccsd(t)-f12/cc-pvtz-f12//wb97m-v/def2tzvpd', freq_scale_factor=1)
        self.assertEqual(arc3.opt_level.simple(), 'wb97m-v/def2tzvpd')
        self.assertEqual(arc3.freq_level.simple(), 'wb97m-v/def2tzvpd')
        self.assertEqual(arc3.sp_level.simple(), 'ccsd(t)-f12/cc-pvtz-f12')
        self.assertEqual(arc3.scan_level.simple(), 'wb97m-v/def2tzvpd')
        self.assertIsNone(arc3.orbitals_level)

        arc4 = ARC(project='test', opt_level='wb97x-d3/6-311++G(3df,3pd)', freq_level='m062x/def2-tzvpp',
                   sp_level='ccsd(t)f12/aug-cc-pvqz', calc_freq_factor=False)
        self.assertEqual(arc4.opt_level.simple(), 'wb97x-d3/6-311++g(3df,3pd)')
        self.assertEqual(arc4.freq_level.simple(), 'm062x/def2-tzvpp')
        self.assertEqual(arc4.sp_level.simple(), 'ccsd(t)f12/aug-cc-pvqz')

        # Test deduce freq level from opt level
        arc7 = ARC(project='test', opt_level='wb97xd/aug-cc-pvtz', calc_freq_factor=False)
        self.assertEqual(arc7.opt_level.simple(), 'wb97xd/aug-cc-pvtz')
        self.assertEqual(arc7.freq_level.simple(), 'wb97xd/aug-cc-pvtz')

        # Test a level not supported by Arkane does not raise error if compute_thermo is False
        arc8 = ARC(project='test', sp_level='method/unsupported', calc_freq_factor=False, compute_thermo=False)
        self.assertEqual(arc8.sp_level.simple(), 'method/unsupported')
        self.assertEqual(arc8.freq_level.simple(), 'wb97xd/def2tzvp')

        # Test that a level not supported by Arkane does raise an error if compute_thermo is True (default)
        with self.assertRaises(ValueError):
            ARC(project='test', sp_level='method/unsupported', calc_freq_factor=False)

        # Test dictionary format specification with auxiliary basis and DFT dispersion
        arc9 = ARC(project='test', opt_level={},
                   freq_level={'method': 'B3LYP/G', 'basis': 'cc-pVDZ(fi/sf/fw)', 'auxiliary_basis': 'def2-svp/C',
                               'dispersion': 'DEF2-tzvp/c'},
                   sp_level={'method': 'DLPNO-CCSD(T)-F12', 'basis': 'cc-pVTZ-F12',
                             'auxiliary_basis': 'aug-cc-pVTZ/C cc-pVTZ-F12-CABS'},
                   calc_freq_factor=False, compute_thermo=False)
        self.assertEqual(arc9.opt_level.simple(), 'wb97xd/def2tzvp')
        self.assertEqual(str(arc9.freq_level), 'b3lyp/g/cc-pvdz(fi/sf/fw), auxiliary_basis: def2-svp/c, '
                                               'dispersion: def2-tzvp/c, software: gaussian')
        self.assertEqual(str(arc9.sp_level),
                         'dlpno-ccsd(t)-f12/cc-pvtz-f12, auxiliary_basis: aug-cc-pvtz/c cc-pvtz-f12-cabs, '
                         'software: orca')

        # Test using default frequency and orbital level for composite job, also forbid rotors job
        arc10 = ARC(project='test', composite_method='cbs-qb3', calc_freq_factor=False,
                    job_types={'rotors': False, 'orbitals': True})
        self.assertEqual(arc10.freq_level.simple(), 'b3lyp/cbsb7')
        self.assertIsNone(arc10.scan_level)
        self.assertEqual(arc10.orbitals_level.simple(), 'b3lyp/cbsb7')

        # Test using specified frequency, scan, and orbital for composite job
        arc11 = ARC(project='test', composite_method='cbs-qb3', freq_level='wb97xd/6-311g', scan_level='apfd/def2svp',
                    orbitals_level='hf/sto-3g', job_types={'orbitals': True}, calc_freq_factor=False)
        self.assertEqual(arc11.scan_level.simple(), 'apfd/def2svp')
        self.assertEqual(arc11.freq_level.simple(), 'wb97xd/6-311g')
        self.assertEqual(arc11.orbitals_level.simple(), 'hf/sto-3g')

        # Test using default frequency and orbital level for job specified from level of theory, also forbid rotors job
        arc12 = ARC(project='test', level_of_theory='b3lyp/sto-3g', calc_freq_factor=False,
                    job_types={'rotors': False, 'orbitals': True}, compute_thermo=False)
        self.assertIsNone(arc12.scan_level)
        self.assertEqual(arc12.freq_level.simple(), 'b3lyp/sto-3g')
        self.assertEqual(arc12.orbitals_level.simple(), 'wb97x-d3/def2tzvp')

        # Test using specified scan level
        arc13 = ARC(project='test', level_of_theory='b3lyp/sto-3g', calc_freq_factor=False, scan_level='apfd/def2svp',
                    job_types={'rotors': True}, compute_thermo=False)
        self.assertEqual(arc13.scan_level.simple(), 'apfd/def2svp')

        # Test specifying semi-empirical and force-field methods using dictionary
        arc14 = ARC(project='test', opt_level={'method': 'AM1'}, freq_level={'method': 'PM6'},
                    sp_level={'method': 'AMBER'}, calc_freq_factor=False, compute_thermo=False)
        self.assertEqual(arc14.opt_level.simple(), 'am1')
        self.assertEqual(arc14.freq_level.simple(), 'pm6')
        self.assertEqual(arc14.sp_level.simple(), 'amber')

    def test_determine_unique_species_labels(self):
        """Test the determine_unique_species_labels method"""
        spc0 = ARCSpecies(label='spc0', smiles='CC', compute_thermo=False)
        spc1 = ARCSpecies(label='spc1', smiles='CC', compute_thermo=False)
        spc2 = ARCSpecies(label='spc2', smiles='CC', compute_thermo=False)
        arc0 = ARC(project='arc_test', job_types=self.job_types1, species=[spc0, spc1, spc2],
                   level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)')
        self.assertEqual(arc0.unique_species_labels, ['spc0', 'spc1', 'spc2'])
        spc3 = ARCSpecies(label='spc0', smiles='CC', compute_thermo=False)
        arc0.species.append(spc3)
        with self.assertRaises(ValueError):
            arc0.determine_unique_species_labels()

    def test_add_hydrogen_for_bde(self):
        """Test the add_hydrogen_for_bde method"""
        spc0 = ARCSpecies(label='spc0', smiles='CC', compute_thermo=False)
        arc0 = ARC(project='arc_test', job_types=self.job_types1, species=[spc0],
                   level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)')
        arc0.add_hydrogen_for_bde()
        self.assertEqual(len(arc0.species), 1)

        spc1 = ARCSpecies(label='spc1', smiles='CC', compute_thermo=False, bdes=['all_h'])
        arc1 = ARC(project='arc_test', job_types=self.job_types1, species=[spc1],
                   level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)')
        arc1.add_hydrogen_for_bde()
        self.assertEqual(len(arc1.species), 2)
        self.assertIn('H', [spc.label for spc in arc1.species])

    def test_process_adaptive_levels(self):
        """Test processing the adaptive levels"""
        adaptive_levels_1 = {(1, 5): {('opt', 'freq'): 'wb97xd/6-311+g(2d,2p)',
                                      ('sp',): 'ccsd(t)-f12/aug-cc-pvtz-f12'},
                             (6, 15): {('opt', 'freq'): 'b3lyp/cbsb7',
                                       'sp': 'dlpno-ccsd(t)/def2-tzvp'},
                             (16, 30): {('opt', 'freq'): 'b3lyp/6-31g(d,p)',
                                        'sp': {'method': 'wb97xd', 'basis': '6-311+g(2d,2p)'}},
                             (31, 'inf'): {('opt', 'freq'): 'b3lyp/6-31g(d,p)',
                                           'sp': 'b3lyp/6-311+g(d,p)'}}

        processed_1 = process_adaptive_levels(adaptive_levels_1)
        self.assertEqual(processed_1[(6, 15)][('sp',)].simple(), 'dlpno-ccsd(t)/def2-tzvp')
        self.assertEqual(processed_1[(16, 30)][('sp',)].simple(), 'wb97xd/6-311+g(2d,2p)')

        # test non dict
        with self.assertRaises(InputError):
            process_adaptive_levels(4)
        # wrong atom range
        with self.assertRaises(InputError):
            process_adaptive_levels({5: {('opt', 'freq'): 'wb97xd/6-311+g(2d,2p)',
                                         ('sp',): 'ccsd(t)-f12/aug-cc-pvtz-f12'},
                                     (6, 'inf'): {('opt', 'freq'): 'b3lyp/6-31g(d,p)',
                                                  'sp': 'b3lyp/6-311+g(d,p)'}})
        # no 'inf
        with self.assertRaises(InputError):
            process_adaptive_levels({(1, 5): {('opt', 'freq'): 'wb97xd/6-311+g(2d,2p)',
                                              ('sp',): 'ccsd(t)-f12/aug-cc-pvtz-f12'},
                                     (6, 75): {('opt', 'freq'): 'b3lyp/6-31g(d,p)',
                                               'sp': 'b3lyp/6-311+g(d,p)'}})
        # adaptive level not a dict
        with self.assertRaises(InputError):
            process_adaptive_levels({(1, 5): {('opt', 'freq'): 'wb97xd/6-311+g(2d,2p)',
                                              ('sp',): 'ccsd(t)-f12/aug-cc-pvtz-f12'},
                                     (6, 'inf'): 'b3lyp/6-31g(d,p)'})
        # non-consecutive atom ranges
        with self.assertRaises(InputError):
            process_adaptive_levels({(1, 5): {('opt', 'freq'): 'wb97xd/6-311+g(2d,2p)',
                                              ('sp',): 'ccsd(t)-f12/aug-cc-pvtz-f12'},
                                     (15, 'inf'): {('opt', 'freq'): 'b3lyp/6-31g(d,p)',
                                                   'sp': 'b3lyp/6-311+g(d,p)'}})

    def test_process_level_of_theory(self):
        """
        Tests the process_level_of_theory function.
        """
        arc0 = ARC(project='test_0', level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)',
                   bac_type=None, freq_scale_factor=1)
        arc1 = ARC(project='test_1', level_of_theory='wb97xd/6-311+g(2d,2p)',
                   arkane_level_of_theory="b3lyp/6-311+g(3df,2p)",
                   bac_type=None,
                   freq_scale_factor=1,
                   job_types={"freq": True,
                              "sp": True,
                              "opt": False})
        arc2 = ARC(project='test_2', sp_level='wb97xd/6-311+g(2d,2p)',
                   opt_level='wb97xd/6-311+g(2d,2p)',
                   arkane_level_of_theory="b3lyp/6-311+g(3df,2p)",
                   bac_type=None,
                   freq_scale_factor=1,
                   job_types={"freq": True,
                              "sp": False,
                              "opt": False})
        arc3 = ARC(project='test_3', sp_level='wb97xd/6-311+g(2d,2p)',
                   opt_level='wb97xd/6-311+g(2d,2p)',
                   arkane_level_of_theory="b3lyp/6-311+g(3df,2p)",
                   bac_type=None,
                   freq_scale_factor=1,
                   job_types={"opt": False})

        arc0.process_level_of_theory(), arc1.process_level_of_theory(), arc2.process_level_of_theory(), arc3.process_level_of_theory()
        for arc in [arc0, arc1, arc2, arc3]:
            self.assertIsInstance(arc.sp_level, Level)
            self.assertIsInstance(arc.opt_level, Level)
            self.assertIsInstance(arc.freq_level, Level)

    def test_unknown_ts_adapter(self):
        """
        Tests that ARC raises an error when unknown TS adapters are given.
        """
        spc1 = ARCSpecies(label='spc1',
                          smiles='CC',
                          compute_thermo=False,
                          )
        with self.assertRaises(InputError):
            arc0 = ARC(project='arc_test',
                       job_types=self.job_types1,
                       species=[spc1],
                       level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)',
                       three_params=False,
                       ts_adapters=['WRONG ADAPTER', 'AutoTST', 'GCN', 'xtb_gsm'],
                       )

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        projects = ['arc_project_for_testing_delete_after_usage_test_from_dict',
                    'arc_model_chemistry_test', 'arc_test', 'test', 'unit_test_specific_job', 'wrong']
        for project in projects:
            project_directory = os.path.join(ARC_PATH, 'Projects', project)
            if os.path.isdir(project_directory):
                shutil.rmtree(project_directory, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
