#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.main module
"""

import inspect
import os
import shutil
import unittest
from unittest.mock import patch

from arc.common import get_test_project_directory, get_test_project_name
from arc.exceptions import InputError
from arc.imports import settings
from arc.level import Level
from arc.main import ARC, process_adaptive_levels
from arc.scheduler import Scheduler
from arc.species.converter import str_to_xyz
from arc.species.species import ARCSpecies, TSGuess

servers = settings['servers']


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
            project_directory = get_test_project_directory(project)
            if os.path.isdir(project_directory):
                shutil.rmtree(project_directory, ignore_errors=True)

    def test_as_dict(self):
        """Test the as_dict() method of ARC"""
        spc1 = ARCSpecies(label='spc1',
                          smiles='CC',
                          compute_thermo=False,
                          )
        arc0 = ARC(project=get_test_project_name('arc_test'),
                   job_types=self.job_types1,
                   species=[spc1],
                   level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)',
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
                         'ess_settings': {'ase': ['local'],
                                          'cfour': ['local'],
                                          'gaussian': ['local', 'server2'],
                                          'gcn': ['local'],
                                          'mockter': ['local'],
                                          'molpro': ['local', 'server2'],
                                          'onedmin': ['server1'],
                                          'openbabel': ['local'],
                                          'orca': ['local'],
                                          'orca_neb': ['local'],
                                          'qchem': ['server1'],
                                          'qst2': ['local'],
                                          'terachem': ['server1'],
                                          'torchani': ['local'],
                                          'xtb': ['local'],
                                          'xtb_gsm': ['local'],
                                          },
                         'freq_level': {'args': {'block': {}, 'keyword': {'general': 'scf=(NDamp=30)'}},
                                        'basis': '6-311+g(3df,2p)',
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
                         'project': get_test_project_name('arc_test'),
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
                                     'multiplicity': 2,
                                     'neg_freqs_trshed': [],
                                     'number_of_rotors': 0,
                                     'opt_level': '',
                                     'optical_isomers': 1,
                                     'rotors_dict': {},
                                     'xyzs': []}],
                        'project_directory': get_test_project_directory(
                            'arc_project_for_testing_delete_after_usage_test_from_dict'),
                        }
        arc1 = ARC(project=get_test_project_name('wrong'), freq_scale_factor=0.95)
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

    def test_from_dict_specific_job(self):
        """Test the from_dict() method of ARC"""
        restart_dict = {'specific_job_type': 'bde',
                        'project': get_test_project_name('unit_test_specific_job'),
                        'project_directory': get_test_project_directory('unit_test_specific_job'),
                        }
        arc1 = ARC(**restart_dict)
        job_type_expected = {'conf_opt': False, 'conf_sp': False, 'opt': True, 'freq': True, 'sp': True, 'rotors': False,
                             'orbitals': False, 'bde': True, 'onedmin': False, 'fine': True, 'irc': False}
        self.assertEqual(arc1.job_types, job_type_expected)

    def test_save_project_info_file_skips_deleted_species(self):
        """Test that a species present in self.species but absent from self.output (e.g., an IRC
        endpoint species deleted mid-run) is omitted from the project info file and from the
        accompanying YAML file, instead of raising a KeyError."""
        arc0 = ARC(project='arc_info_test', species=[ARCSpecies(label='tst_spc', smiles='C')],
                   level_of_theory='b3lyp/6-31g', bac_type=None, compute_thermo=False,
                   freq_scale_factor=1.0, calc_freq_factor=False, job_types=self.job_types1,
                   ess_settings={'gaussian': ['local']})
        self.addCleanup(shutil.rmtree, arc0.project_directory, ignore_errors=True)
        arc0.species.append(ARCSpecies(label='IRC_TS0_1', smiles='O'))
        arc0.output = {'tst_spc': {'convergence': True}}
        arc0.save_project_info_file()
        with open(os.path.join(arc0.project_directory, f'{arc0.project}.info'), 'r') as f:
            content = f.read()
        self.assertIn('tst_spc', content)
        self.assertNotIn('IRC_TS0_1', content)
        with open(os.path.join(arc0.project_directory, f'{arc0.project}_info.yml'), 'r') as f:
            yml_content = f.read()
        self.assertIn('tst_spc', yml_content)
        self.assertNotIn('IRC_TS0_1', yml_content)

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_save_project_info_file_after_a_scheduler_deleted_an_irc_species(self, mock_run_opt):
        """Test that an ARC run whose Scheduler abandoned a TS guess, and thereby deleted the IRC
        species spawned for it, can still write its project info files. Wires a real ARC to a real
        Scheduler the way ARC.execute does, so it covers the shared species list rather than a
        stand-in for it."""
        ts_xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")
        ts_spc = ARCSpecies(label='TS0', is_ts=True, xyz=ts_xyz, multiplicity=1, charge=0,
                            compute_thermo=False)
        ts_spc.ts_guesses = [
            TSGuess(index=0, method='heuristics', success=True, energy=100.0, xyz=ts_xyz,
                    execution_time='0:00:01'),
            TSGuess(index=1, method='heuristics', success=True, energy=110.0, xyz=ts_xyz,
                    execution_time='0:00:01'),
        ]
        ts_spc.ts_guesses[0].opt_xyz = ts_xyz
        ts_spc.ts_guesses[0].imaginary_freqs = [-500.0]
        ts_spc.ts_guesses[1].opt_xyz = ts_xyz
        ts_spc.ts_guesses[1].imaginary_freqs = [-400.0]
        ts_spc.chosen_ts = 0
        ts_spc.chosen_ts_list = [0]
        ts_spc.ts_guesses_exhausted = False

        arc0 = ARC(project='arc_info_e2e_test', species=[ts_spc], level_of_theory='b3lyp/6-31g',
                   bac_type=None, compute_thermo=False, freq_scale_factor=1.0,
                   calc_freq_factor=False, job_types=self.job_types1,
                   ess_settings={'gaussian': ['local']})
        self.addCleanup(shutil.rmtree, arc0.project_directory, ignore_errors=True)
        sched = Scheduler(project=arc0.project, species_list=arc0.species,
                          ess_settings=arc0.ess_settings, opt_level=arc0.opt_level,
                          freq_level=arc0.freq_level, sp_level=arc0.sp_level,
                          ts_guess_level=arc0.ts_guess_level,
                          project_directory=arc0.project_directory, testing=True,
                          job_types=arc0.job_types)
        self.assertIs(arc0.species, sched.species_list)

        irc_label = 'IRC_TS0_1'
        sched.species_dict[irc_label] = ARCSpecies(label=irc_label, xyz=ts_xyz,
                                                   compute_thermo=False, irc_label='TS0')
        sched.species_list.append(sched.species_dict[irc_label])
        sched.unique_species_labels.append(irc_label)
        sched.initialize_output_dict(label=irc_label)
        ts_spc.irc_label = irc_label
        self.assertIn(irc_label, [spc.label for spc in arc0.species])

        sched.switch_ts('TS0')
        arc0.output = sched.output
        arc0.save_project_info_file()

        self.assertNotIn(irc_label, [spc.label for spc in arc0.species])
        with open(os.path.join(arc0.project_directory, f'{arc0.project}.info'), 'r') as f:
            content = f.read()
        self.assertIn('TS0', content)
        self.assertNotIn(irc_label, content)
        with open(os.path.join(arc0.project_directory, f'{arc0.project}_info.yml'), 'r') as f:
            yml_content = f.read()
        self.assertNotIn('IRC_TS0_1', yml_content)

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
        arc0 = ARC(project=get_test_project_name('arc_model_chemistry_test'), level_of_theory='CBS-QB3')
        self.assertEqual(str(arc0.arkane_level_of_theory), "cbs-qb3, software: gaussian")
        self.assertEqual(arc0.freq_scale_factor, 1.004)

        arc1 = ARC(project=get_test_project_name('arc_model_chemistry_test'), level_of_theory='cbs-qb3-paraskevas')
        self.assertEqual(str(arc1.arkane_level_of_theory), 'cbs-qb3-paraskevas, software: gaussian')
        self.assertEqual(arc1.freq_scale_factor, 1.004)
        self.assertEqual(arc1.bac_type, 'p')

        arc2 = ARC(project=get_test_project_name('arc_model_chemistry_test'),
                   level_of_theory='ccsd(t)-f12/cc-pvtz-f12//m062x/cc-pvtz')
        self.assertEqual(str(arc2.arkane_level_of_theory), 'ccsd(t)-f12/cc-pvtz-f12, software: molpro')
        self.assertEqual(arc2.freq_scale_factor, 0.955)

        arc3 = ARC(project=get_test_project_name('arc_model_chemistry_test'),
                   sp_level='ccsd(t)-f12/cc-pvtz-f12', opt_level='wb97xd/def2tzvp')
        self.assertEqual(str(arc3.arkane_level_of_theory), 'ccsd(t)-f12/cc-pvtz-f12, software: molpro')
        self.assertEqual(arc3.freq_scale_factor, 0.988)

    def test_determine_model_chemistry_for_job_types(self):
        """Test determining the model chemistry specification dictionary for job types"""
        # Test conflicted inputs: specify both level_of_theory and composite_method
        with self.assertRaises(InputError):
            ARC(project=get_test_project_name('test'), level_of_theory='ccsd(t)-f12/cc-pvtz-f12//wb97x-d/aug-cc-pvtz',
                composite_method='cbs-qb3')

        # Test illegal level of theory specification (method contains multiple slashes)
        with self.assertRaises(ValueError):
            ARC(project=get_test_project_name('test'), level_of_theory='dlpno-mp2-f12/D/cc-pVDZ(fi/sf/fw)//b3lyp/G/def2svp')

        # Test illegal job level specification (method contains multiple slashes)
        with self.assertRaises(ValueError):
            ARC(project=get_test_project_name('test'), opt_level='b3lyp/d/def2tzvp/def2tzvp/c')

        # Test illegal job level specification (method contains empty space)
        with self.assertRaises(ValueError):
            ARC(project=get_test_project_name('test'), opt_level='b3lyp/def2tzvp def2tzvp/c')

        # Test direct job level specification conflicts with level of theory specification
        with self.assertRaises(InputError):
            ARC(project=get_test_project_name('test'), level_of_theory='b3lyp/sto-3g', opt_level='wb97xd/def2tzvp')

        # Test deduce levels from default method from settings.py
        arc1 = ARC(project=get_test_project_name('test'))
        self.assertEqual(arc1.opt_level.simple(), 'wb97xd/def2tzvp')
        self.assertEqual(arc1.freq_level.simple(), 'wb97xd/def2tzvp')
        self.assertEqual(arc1.sp_level.simple(), 'ccsd(t)-f12/cc-pvtz-f12')

        # Test deduce levels from composite method specification
        arc2 = ARC(project=get_test_project_name('test'), composite_method='cbs-qb3')
        self.assertIsNotNone(arc2.opt_level)
        self.assertIsNone(arc2.sp_level)
        self.assertIsNone(arc2.orbitals_level)
        self.assertEqual(arc2.freq_level.simple(), 'b3lyp/cbsb7')
        self.assertEqual(arc2.scan_level.simple(), 'b3lyp/cbsb7')
        self.assertEqual(arc2.composite_method.simple(), 'cbs-qb3')

        # Test deduce levels from level of theory specification
        arc3 = ARC(project=get_test_project_name('test'), freq_scale_factor=1,
                   level_of_theory='ccsd(t)-f12/cc-pvtz-f12//wb97m-v/def2tzvpd')
        self.assertEqual(arc3.opt_level.simple(), 'wb97m-v/def2tzvpd')
        self.assertEqual(arc3.freq_level.simple(), 'wb97m-v/def2tzvpd')
        self.assertEqual(arc3.sp_level.simple(), 'ccsd(t)-f12/cc-pvtz-f12')
        self.assertEqual(arc3.scan_level.simple(), 'wb97m-v/def2tzvpd')
        self.assertIsNone(arc3.orbitals_level)

        arc4 = ARC(project=get_test_project_name('test'), opt_level='wb97x-d3/6-311++G(3df,3pd)', freq_level='m062x/def2-tzvpp',
                   sp_level='ccsd(t)f12/aug-cc-pvqz', calc_freq_factor=False, compute_thermo=False)
        self.assertEqual(arc4.opt_level.simple(), 'wb97x-d3/6-311++g(3df,3pd)')
        self.assertEqual(arc4.freq_level.simple(), 'm062x/def2-tzvpp')
        self.assertEqual(arc4.sp_level.simple(), 'ccsd(t)f12/aug-cc-pvqz')

        # Test deduce freq level from opt level
        arc7 = ARC(project=get_test_project_name('test'), opt_level='wb97xd/aug-cc-pvtz', calc_freq_factor=False)
        self.assertEqual(arc7.opt_level.simple(), 'wb97xd/aug-cc-pvtz')
        self.assertEqual(arc7.freq_level.simple(), 'wb97xd/aug-cc-pvtz')

        # Test a level not supported by Arkane does not raise error if compute_thermo is False
        arc8 = ARC(project=get_test_project_name('test'), sp_level='method/unsupported',
                   calc_freq_factor=False, compute_thermo=False)
        self.assertEqual(arc8.sp_level.simple(), 'method/unsupported')
        self.assertEqual(arc8.freq_level.simple(), 'wb97xd/def2tzvp')

        # Test that a level not supported by Arkane does raise an error if compute_thermo is True (default)
        with self.assertRaises(ValueError):
            ARC(project=get_test_project_name('test'), sp_level='method/unsupported', calc_freq_factor=False)

        # Test dictionary format specification with auxiliary basis and DFT dispersion
        arc9 = ARC(project=get_test_project_name('test'), opt_level={},
                   freq_level={'method': 'B3LYP/G', 'basis': 'cc-pVDZ(fi/sf/fw)', 'auxiliary_basis': 'def2-svp/C',
                               'dispersion': 'DEF2-tzvp/c'},
                   sp_level={'method': 'DLPNO-CCSD(T)-F12', 'basis': 'cc-pVTZ-F12',
                             'auxiliary_basis': 'aug-cc-pVTZ/C', 'cabs': 'cc-pVTZ-F12-CABS'},
                   calc_freq_factor=False, compute_thermo=False)
        self.assertEqual(arc9.opt_level.simple(), 'wb97xd/def2tzvp')
        self.assertEqual(str(arc9.freq_level), 'b3lyp/g/cc-pvdz(fi/sf/fw), auxiliary_basis: def2-svp/c, '
                                               'dispersion: def2-tzvp/c, software: gaussian')
        self.assertEqual(str(arc9.sp_level),
                         'dlpno-ccsd(t)-f12/cc-pvtz-f12, auxiliary_basis: aug-cc-pvtz/c, '
                         'cabs: cc-pvtz-f12-cabs, software: orca')

        # Test using default frequency and orbital level for composite job, also forbid rotors job
        arc10 = ARC(project=get_test_project_name('test'), composite_method='cbs-qb3', calc_freq_factor=False,
                    job_types={'rotors': False, 'orbitals': True})
        self.assertEqual(arc10.freq_level.simple(), 'b3lyp/cbsb7')
        self.assertIsNone(arc10.scan_level)
        self.assertEqual(arc10.orbitals_level.simple(), 'b3lyp/cbsb7')

        # Test using specified frequency, scan, and orbital for composite job
        arc11 = ARC(project=get_test_project_name('test'), composite_method='cbs-qb3',
                    freq_level='wb97xd/6-311g', scan_level='apfd/def2svp',
                    orbitals_level='hf/sto-3g', job_types={'orbitals': True}, calc_freq_factor=False)
        self.assertEqual(arc11.scan_level.simple(), 'apfd/def2svp')
        self.assertEqual(arc11.freq_level.simple(), 'wb97xd/6-311g')
        self.assertEqual(arc11.orbitals_level.simple(), 'hf/sto-3g')

        # Test using default frequency and orbital level for job specified from level of theory, also forbid rotors job
        arc12 = ARC(project=get_test_project_name('test'), level_of_theory='b3lyp/sto-3g', calc_freq_factor=False,
                    job_types={'rotors': False, 'orbitals': True}, compute_thermo=False)
        self.assertIsNone(arc12.scan_level)
        self.assertEqual(arc12.freq_level.simple(), 'b3lyp/sto-3g')
        self.assertEqual(arc12.orbitals_level.simple(), 'wb97x-d3/def2tzvp')

        # Test using specified scan level
        arc13 = ARC(project=get_test_project_name('test'), level_of_theory='b3lyp/sto-3g',
                    calc_freq_factor=False, scan_level='apfd/def2svp',
                    job_types={'rotors': True}, compute_thermo=False)
        self.assertEqual(arc13.scan_level.simple(), 'apfd/def2svp')

        # Test specifying semi-empirical and force-field methods using dictionary
        arc14 = ARC(project=get_test_project_name('test'), opt_level={'method': 'AM1'}, freq_level={'method': 'PM6'},
                    sp_level={'method': 'AMBER'}, calc_freq_factor=False, compute_thermo=False)
        self.assertEqual(arc14.opt_level.simple(), 'am1')
        self.assertEqual(arc14.freq_level.simple(), 'pm6')
        self.assertEqual(arc14.sp_level.simple(), 'amber')

        # Test explicit year in arkane_level_of_theory dictionary
        arc15 = ARC(project=get_test_project_name('test'),
                    sp_level='wb97xd/def2tzvp',
                    opt_level='wb97xd/def2tzvp',
                    arkane_level_of_theory={'method': 'wb97xd', 'basis': 'def2tzvp', 'year': 2023},
                    bac_type=None,
                    calc_freq_factor=False, compute_thermo=False)
        self.assertEqual(arc15.arkane_level_of_theory.year, 2023)

        # Test warning when year is specified on sp_level instead of arkane_level_of_theory
        arc16 = ARC(project=get_test_project_name('test'),
                    sp_level={'method': 'wb97xd', 'basis': 'def2tzvp', 'year': 2023},
                    opt_level='wb97xd/def2tzvp',
                    calc_freq_factor=False, compute_thermo=False)
        with open(os.path.join(arc16.project_directory, 'arc.log'), 'r') as f:
            log_content = f.read()
        self.assertIn('"year" attribute on sp_level', log_content)

    def test_determine_unique_species_labels(self):
        """Test the determine_unique_species_labels method"""
        spc0 = ARCSpecies(label='spc0', smiles='CC', compute_thermo=False)
        spc1 = ARCSpecies(label='spc1', smiles='CC', compute_thermo=False)
        spc2 = ARCSpecies(label='spc2', smiles='CC', compute_thermo=False)
        arc0 = ARC(project=get_test_project_name('arc_test'), job_types=self.job_types1, species=[spc0, spc1, spc2],
                   level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)')
        self.assertEqual(arc0.unique_species_labels, ['spc0', 'spc1', 'spc2'])
        spc3 = ARCSpecies(label='spc0', smiles='CC', compute_thermo=False)
        arc0.species.append(spc3)
        with self.assertRaises(ValueError):
            arc0.determine_unique_species_labels()

    def test_add_hydrogen_for_bde(self):
        """Test the add_hydrogen_for_bde method"""
        spc0 = ARCSpecies(label='spc0', smiles='CC', compute_thermo=False)
        arc0 = ARC(project=get_test_project_name('arc_test'), job_types=self.job_types1, species=[spc0],
                   level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)')
        arc0.add_hydrogen_for_bde()
        self.assertEqual(len(arc0.species), 1)

        spc1 = ARCSpecies(label='spc1', smiles='CC', compute_thermo=False, bdes=['all_h'])
        arc1 = ARC(project=get_test_project_name('arc_test'), job_types=self.job_types1, species=[spc1],
                   level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)')
        arc1.add_hydrogen_for_bde()
        self.assertEqual(len(arc1.species), 2)
        self.assertIn('H', [spc.label for spc in arc1.species])

    def test_process_adaptive_levels(self):
        """Test processing the adaptive levels (YAML-friendly list-of-entries schema)"""
        # None passes through.
        self.assertIsNone(process_adaptive_levels(None))

        # A normal, multi-range specification. Job types sharing a level are given as a
        # whitespace- or comma-separated key; a level may be a string or a Level dict.
        adaptive_levels_1 = [{'atom_range': [1, 5],
                              'levels': {'opt freq': 'wb97xd/6-311+g(2d,2p)',
                                         'sp': 'ccsd(t)-f12/aug-cc-pvtz-f12'}},
                             {'atom_range': [6, 15],
                              'levels': {'opt, freq': 'b3lyp/cbsb7',
                                         'sp': 'dlpno-ccsd(t)/def2-tzvp'}},
                             {'atom_range': [16, 30],
                              'levels': {'opt freq': 'b3lyp/6-31g(d,p)',
                                         'sp': {'method': 'wb97xd', 'basis': '6-311+g(2d,2p)'}}},
                             {'atom_range': [31, 'inf'],
                              'levels': {'opt freq': 'b3lyp/6-31g(d,p)',
                                         'sp': 'b3lyp/6-311+g(d,p)'}}]
        processed_1 = process_adaptive_levels(adaptive_levels_1)
        self.assertEqual(processed_1[(6, 15)][('sp',)].simple(), 'dlpno-ccsd(t)/def2-tzvp')
        self.assertEqual(processed_1[(16, 30)][('sp',)].simple(), 'wb97xd/6-311+g(2d,2p)')
        self.assertEqual(processed_1[(1, 5)][('opt', 'freq')].simple(), 'wb97xd/6-311+g(2d,2p)')

        # A single range covering everything, and a float 'inf' is accepted as the upper bound.
        processed_2 = process_adaptive_levels([{'atom_range': [1, float('inf')],
                                                'levels': {'opt freq': 'b3lyp/6-31g(d,p)',
                                                           'sp': 'b3lyp/6-311+g(d,p)'}}])
        self.assertEqual(processed_2[(1, 'inf')][('sp',)].simple(), 'b3lyp/6-311+g(d,p)')

        # Restart round-trip: as_dict() must emit the list form and reproduce the same structure.
        arc0 = ARC(project='adaptive_levels_test', adaptive_levels=adaptive_levels_1)
        restart_levels = arc0.as_dict()['adaptive_levels']
        self.assertIsInstance(restart_levels, list)
        reprocessed = process_adaptive_levels(restart_levels)
        self.assertEqual(reprocessed[(6, 15)][('sp',)].simple(), 'dlpno-ccsd(t)/def2-tzvp')
        self.assertEqual(set(reprocessed.keys()), set(processed_1.keys()))

        # Not a list (the legacy tuple-dict form is no longer accepted).
        with self.assertRaises(InputError):
            process_adaptive_levels(4)
        with self.assertRaises(InputError):
            process_adaptive_levels({(1, 5): {('opt', 'freq'): 'wb97xd/6-311+g(2d,2p)'},
                                     (6, 'inf'): {'sp': 'b3lyp/6-311+g(d,p)'}})
        # atom_range is not a 2-length list.
        with self.assertRaises(InputError):
            process_adaptive_levels([{'atom_range': [5], 'levels': {'sp': 'b3lyp/6-311+g(d,p)'}}])
        # 'inf' is only allowed as the upper bound.
        with self.assertRaises(InputError):
            process_adaptive_levels([{'atom_range': [float('inf'), 'inf'], 'levels': {'sp': 'b3lyp/6-311+g(d,p)'}}])
        with self.assertRaises(InputError):
            process_adaptive_levels([{'atom_range': ['inf', 10], 'levels': {'sp': 'b3lyp/6-311+g(d,p)'}}])
        # The last range does not end with 'inf'.
        with self.assertRaises(InputError):
            process_adaptive_levels([{'atom_range': [1, 5], 'levels': {'sp': 'wb97xd/def2tzvp'}},
                                     {'atom_range': [6, 75], 'levels': {'sp': 'b3lyp/6-311+g(d,p)'}}])
        # The first range does not start at 1.
        with self.assertRaises(InputError):
            process_adaptive_levels([{'atom_range': [2, 5], 'levels': {'sp': 'wb97xd/def2tzvp'}},
                                     {'atom_range': [6, 'inf'], 'levels': {'sp': 'b3lyp/6-311+g(d,p)'}}])
        # 'levels' is not a dict.
        with self.assertRaises(InputError):
            process_adaptive_levels([{'atom_range': [1, 5], 'levels': {'sp': 'wb97xd/def2tzvp'}},
                                     {'atom_range': [6, 'inf'], 'levels': 'b3lyp/6-31g(d,p)'}])
        # Non-consecutive atom ranges.
        with self.assertRaises(InputError):
            process_adaptive_levels([{'atom_range': [1, 5], 'levels': {'sp': 'wb97xd/def2tzvp'}},
                                     {'atom_range': [15, 'inf'], 'levels': {'sp': 'b3lyp/6-311+g(d,p)'}}])
        # An entry missing required keys.
        with self.assertRaises(InputError):
            process_adaptive_levels([{'levels': {'sp': 'wb97xd/def2tzvp'}}])

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
            ARC(project=get_test_project_name('arc_test'),
                job_types=self.job_types1,
                species=[spc1],
                level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)',
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
            project_directory = get_test_project_directory(project)
            if os.path.isdir(project_directory):
                shutil.rmtree(project_directory, ignore_errors=True)


class TestRestartRoundTrip(unittest.TestCase):
    """
    Test that a restart dictionary can be fed straight back into the ARC constructor.

    ``ARC.py`` restarts a project with ``ARC(**read_yaml_file('restart.yml'))``, and ``restart.yml``
    is the dictionary produced by ``ARC.as_dict()`` plus the keys the Scheduler adds to it. Any key
    written into that dictionary which ``ARC.__init__()`` does not accept makes every restart of an
    affected project fail with ``TypeError: got an unexpected keyword argument``, and nothing
    detects it until somebody actually restarts. ``Scheduler.save_restart_dict()`` writes into
    ARC's constructor namespace, so keys can be added there without touching ``main.py`` at all.
    """

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None

    def tearDown(self):
        for project in ('arc_restart_roundtrip',):
            project_directory = os.path.join(ARC_PATH, 'Projects', project)
            shutil.rmtree(project_directory, ignore_errors=True)

    def test_as_dict_output_is_accepted_by_the_constructor(self):
        """Every key ARC writes into a restart dictionary must be a constructor parameter."""
        arc0 = ARC(project='arc_restart_roundtrip',
                   species=[ARCSpecies(label='spc1', smiles='CC', compute_thermo=False)],
                   compute_thermo=False,
                   )
        restart_dict = arc0.as_dict()
        accepted = set(inspect.signature(ARC.__init__).parameters) - {'self'}
        unexpected = sorted(set(restart_dict) - accepted)
        self.assertEqual(unexpected, list(),
                         f'ARC.as_dict() emits key(s) that ARC.__init__() cannot accept: {unexpected}. '
                         f'Every key written into a restart dictionary must be a constructor parameter, '
                         f'or restarting any affected project raises TypeError.')

    def test_scheduler_restart_keys_are_accepted_by_the_constructor(self):
        """
        The keys ``Scheduler.save_restart_dict()`` adds must also be constructor parameters.

        The Scheduler writes into the same dictionary ARC is later reconstructed from, so a key
        added there is just as breaking as one added to ``as_dict()`` - and is easier to miss,
        because it does not touch ``main.py``.
        """
        accepted = set(inspect.signature(ARC.__init__).parameters) - {'self'}
        scheduler_written_keys = {'output', 'output_multi_spc', 'completed_job_records',
                                  'species', 'running_jobs'}
        unexpected = sorted(scheduler_written_keys - accepted)
        self.assertEqual(unexpected, list(),
                         f'Scheduler.save_restart_dict() writes key(s) ARC.__init__() cannot accept: '
                         f'{unexpected}.')

    def test_completed_job_records_survives_a_restart(self):
        """Cost records are persisted specifically so a restarted run keeps its history."""
        records = [{'job_name': 'opt_a1', 'job_type': 'opt', 'adapter': 'gaussian',
                    'server': 'local', 'cpu_cores': 8, 'run_time': 12.5, 'status': 'done'}]
        arc0 = ARC(project='arc_restart_roundtrip',
                   species=[ARCSpecies(label='spc1', smiles='CC', compute_thermo=False)],
                   compute_thermo=False,
                   completed_job_records=records,
                   )
        restart_dict = arc0.as_dict()
        self.assertEqual(restart_dict['completed_job_records'], records)
        arc1 = ARC(**restart_dict)
        self.assertEqual(arc1.completed_job_records, records)

    def test_absent_completed_job_records_defaults_to_empty(self):
        """A restart file written before the key existed must still construct."""
        arc0 = ARC(project='arc_restart_roundtrip',
                   species=[ARCSpecies(label='spc1', smiles='CC', compute_thermo=False)],
                   compute_thermo=False,
                   )
        self.assertEqual(arc0.completed_job_records, list())
        restart_dict = arc0.as_dict()
        self.assertNotIn('completed_job_records', restart_dict)
        self.assertEqual(ARC(**restart_dict).completed_job_records, list())


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
