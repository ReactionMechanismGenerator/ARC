#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.main module
"""

import logging
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from arc.common import ARC_PATH, get_logger, read_yaml_file, save_yaml_file
from arc.exceptions import InputError
from arc.imports import settings
from arc.job.adapters.gaussian import GaussianAdapter
from arc.job.ssh import SSHClient
from arc.level import Level
from arc.main import ARC, process_adaptive_levels, resolve_legacy_multiplicity_source
from arc.scheduler import Scheduler
from arc.settings.settings import default_levels_of_theory as repo_default_levels_of_theory
from arc.species.species import ARCSpecies

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
                   ts_adapters=['heuristics', 'linear', 'AutoTST', 'GCN', 'xtb_gsm', 'orca_neb'],
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
                                          'pyscf': ['local'],
                                          'qchem': ['server1'],
                                          'rits': ['local'],
                                          'terachem': ['server1'],
                                          'torchani': ['local'],
                                          'xtb': ['local'],
                                          'xtb_gsm': ['local'],
                                          },
                         'freq_level': {'args': {'block': {},
                                                 'keyword': {'general': 'scf=(NDamp=30)'}},
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
                                       'sp': True,
                                       'stability': False},
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
                         'ts_adapters': ['heuristics', 'linear', 'AutoTST', 'GCN', 'xtb_gsm', 'orca_neb']}
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

    def test_from_dict_specific_job(self):
        """Test the from_dict() method of ARC"""
        restart_dict = {'specific_job_type': 'bde',
                        'project': 'unit_test_specific_job',
                        'project_directory': os.path.join(ARC_PATH, 'Projects', 'unit_test_specific_job'),
                        }
        arc1 = ARC(**restart_dict)
        job_type_expected = {'conf_opt': False, 'conf_sp': False, 'opt': True, 'freq': True, 'sp': True, 'rotors': False,
                             'orbitals': False, 'bde': True, 'onedmin': False, 'fine': True, 'irc': False,
                             'stability': False}
        self.assertEqual(arc1.job_types, job_type_expected)

    def test_rotor_scan_resolution_input_key(self):
        """Test the rotor_scan_resolution input key is parsed, stored, and round-tripped."""
        arc0 = ARC(project='arc_test_scan_res', rotor_scan_resolution=4.0)
        self.assertEqual(arc0.rotor_scan_resolution, 4.0)
        self.assertEqual(arc0.as_dict()['rotor_scan_resolution'], 4.0)
        # Absent the key, the attribute is None and it is not written to the restart dict,
        # so an existing project's restart file is byte-identical to before this change.
        arc1 = ARC(project='arc_test_no_scan_res')
        self.assertIsNone(arc1.rotor_scan_resolution)
        self.assertNotIn('rotor_scan_resolution', arc1.as_dict())

    def test_rotor_scan_resolution_guard(self):
        """Test that a rotor scan resolution coarser than 20 degrees is refused."""
        with self.assertRaises(InputError):
            ARC(project='arc_test_coarse_scan_res', rotor_scan_resolution=30.0)
        with self.assertRaises(InputError):
            ARC(project='arc_test_nonpositive_scan_res', rotor_scan_resolution=0.0)
        # A non-numeric type (e.g. a quoted YAML value) is refused with a consistent InputError
        # rather than raising a raw TypeError from the numeric comparison.
        with self.assertRaises(InputError):
            ARC(project='arc_test_str_scan_res', rotor_scan_resolution='4.0')
        # A bool is an int subclass; it must be refused rather than silently read as 1 or 0.
        with self.assertRaises(InputError):
            ARC(project='arc_test_true_scan_res', rotor_scan_resolution=True)
        with self.assertRaises(InputError):
            ARC(project='arc_test_false_scan_res', rotor_scan_resolution=False)
        # A value that does not divide 360 evenly leaves a fractional final step and is refused.
        with self.assertRaises(InputError):
            ARC(project='arc_test_indivisible_scan_res', rotor_scan_resolution=7.0)
        # Exactly 20 degrees (18 points) is the coarsest still accepted.
        arc0 = ARC(project='arc_test_boundary_scan_res', rotor_scan_resolution=20.0)
        self.assertEqual(arc0.rotor_scan_resolution, 20.0)

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

    def test_g4_composite_takes_its_prescribed_freq_level_and_scale_factor(self):
        """Test that ARC pairs the G4 composite with the level the shipped settings prescribe.

        Patched with the repository's own mapping rather than read live: ``arc/imports.py`` overlays
        ``~/.arc/settings.py``, which may still carry the pre-mapping ``'B3LYP/CBSB7'`` string, and
        G4 would then silently take CBS-QB3's frequency level. Asserting against the repository
        default is what makes this a check on the shipped data instead of on local configuration.
        """
        with patch.dict('arc.level.default_levels_of_theory',
                        {'freq_for_composite': repo_default_levels_of_theory['freq_for_composite']}):
            arc_g4 = ARC(project='test', composite_method='g4')
        self.assertEqual(arc_g4.composite_method.simple(), 'g4')
        self.assertEqual(arc_g4.freq_level.simple(), 'b3lyp/6-31g(2df,p)')
        self.assertEqual(arc_g4.freq_scale_factor, 0.9992)
        self.assertIsNone(arc_g4.sp_level)

    _SP_ONLY_JOB_TYPES = {'conf_opt': False, 'conf_sp': False, 'opt': False, 'freq': False, 'sp': True,
                          'rotors': False, 'irc': False, 'orbitals': False, 'lennard_jones': False, 'bde': False}

    def _composite_arc(self, **kwargs):
        """Initialize a composite ARC against the repository's own freq_for_composite mapping.

        ``calc_freq_factor=False`` for the same reason the composite cases in
        ``test_determine_model_chemistry_for_job_types`` use it: G3 has no entry in
        ``data/freq_scale_factors.yml``, and ``check_freq_scaling_factor()`` would then spawn Truhlar
        scale-factor QM jobs for small species nobody asked for - an hour of them, and unrelated to
        what is under test here.

        Args:
            **kwargs: Passed to ``ARC``.

        Returns:
            ARC: The initialized object.
        """
        with patch.dict('arc.level.default_levels_of_theory',
                        {'freq_for_composite': repo_default_levels_of_theory['freq_for_composite']}):
            return ARC(project='test', calc_freq_factor=False, compute_thermo=False, compute_rates=False,
                       **kwargs)

    def test_a_composite_run_without_frequencies_needs_no_prescribed_freq_level(self):
        """A composite with no prescribed frequency level must not abort a run that asks for no frequencies.

        ARC refuses to guess a composite's frequency level, because the scale factor stored for a composite
        is that protocol's own ZPE factor. That refusal belongs where the level is actually used: the
        non-composite path leaves freq_level unset when 'freq' is off, while a single-point-only composite
        run used to raise at initialization, before any job existed to be mispaired.
        """
        arc_sp = self._composite_arc(composite_method='g3', job_types=self._SP_ONLY_JOB_TYPES)
        self.assertEqual(arc_sp.composite_method.simple(), 'g3')
        self.assertIsNone(arc_sp.freq_level)

    def test_a_composite_run_with_frequencies_still_refuses_an_unprescribed_freq_level(self):
        """Requesting frequencies for a composite ARC has no level for must still raise, not guess one"""
        with self.assertRaises(ValueError):
            self._composite_arc(composite_method='g3', job_types=dict(self._SP_ONLY_JOB_TYPES, freq=True))

    def test_a_composite_irc_run_still_needs_a_prescribed_freq_level(self):
        """An IRC run needs the composite's frequency level even with no frequency job requested.

        ``opt_level`` is None for a composite run, so ``spawn_post_irc_jobs()`` optimizes its endpoint
        species at ``freq_level``. Gating this resolution on job_types['freq'] alone would hand that call
        a None level instead of raising here, where the message says what to do about it.
        """
        with self.assertRaises(ValueError):
            self._composite_arc(composite_method='g3', job_types=dict(self._SP_ONLY_JOB_TYPES, irc=True))

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
                   sp_level='ccsd(t)f12/aug-cc-pvqz', calc_freq_factor=False, compute_thermo=False)
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
                             'auxiliary_basis': 'aug-cc-pVTZ/C', 'cabs': 'cc-pVTZ-F12-CABS'},
                   calc_freq_factor=False, compute_thermo=False)
        self.assertEqual(arc9.opt_level.simple(), 'wb97xd/def2tzvp')
        self.assertEqual(str(arc9.freq_level), 'b3lyp/g/cc-pvdz(fi/sf/fw), auxiliary_basis: def2-svp/c, '
                                               'dispersion: def2-tzvp/c, software: gaussian')
        self.assertEqual(str(arc9.sp_level),
                         'dlpno-ccsd(t)-f12/cc-pvtz-f12, auxiliary_basis: aug-cc-pvtz/c, '
                         'cabs: cc-pvtz-f12-cabs, software: orca')

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

        # Test explicit year in arkane_level_of_theory dictionary
        arc15 = ARC(project='test',
                    sp_level='wb97xd/def2tzvp',
                    opt_level='wb97xd/def2tzvp',
                    arkane_level_of_theory={'method': 'wb97xd', 'basis': 'def2tzvp', 'year': 2023},
                    bac_type=None,
                    calc_freq_factor=False, compute_thermo=False)
        self.assertEqual(arc15.arkane_level_of_theory.year, 2023)

        # Test warning when year is specified on sp_level instead of arkane_level_of_theory
        arc16 = ARC(project='test',
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
            arc0 = ARC(project='arc_test',
                       job_types=self.job_types1,
                       species=[spc1],
                       level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)',
                       ts_adapters=['WRONG ADAPTER', 'AutoTST', 'GCN', 'xtb_gsm'],
                       )

    def test_reaction_multiplicity_from_input_dict_reaches_ts(self):
        """Test that a reaction whose species are named by label in an input dict gets the right TS multiplicity"""
        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_test_rxn_multiplicity')
        input_dict = {'project': 'arc_test_rxn_multiplicity',
                      'project_directory': project_directory,
                      'calc_freq_factor': False,
                      'compute_thermo': False,
                      'species': [{'label': 'R', 'smiles': '[CH2]C(C)CC'}, {'label': 'P', 'smiles': 'CC(C)[CH]C'}],
                      'reactions': [{'label': 'R <=> P', 'reactants': ['R'], 'products': ['P']}]}
        arc0 = ARC(**input_dict)
        sched = Scheduler(project=arc0.project, species_list=arc0.species, rxn_list=arc0.reactions,
                          ess_settings={'gaussian': ['local']}, project_directory=project_directory, testing=True,
                          job_types={'conf_opt': False, 'opt': True, 'freq': True, 'sp': True, 'rotors': False,
                                     'fine_grid': False})
        self.assertEqual(arc0.reactions[0].multiplicity, 2)
        ts_species = [spc for spc in sched.species_list if spc.is_ts]
        self.assertEqual(len(ts_species), 1)
        self.assertEqual(ts_species[0].multiplicity, 2)

    def test_summary_reports_the_warnings_of_a_converged_species(self):
        """Test that the run summary prints the warnings of a species that converged"""
        arc0 = ARC(project='arc_test',
                   job_types=self.job_types1,
                   species=[ARCSpecies(label='spc1', smiles='CC', compute_thermo=False)],
                   level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)',
                   )
        arc0.output = {'spc1': {'convergence': True,
                                'job_types': {},
                                'info': '',
                                'warnings': 'the electronic energy and the ZPE were computed with different '
                                            'SCF references; ',
                                'errors': '',
                                'wavefunction_stability': 'external_instability (RHF-->UHF, -0.0312)',
                                }}
        with self.assertLogs(logger=get_logger(), level=logging.INFO) as captured:
            status_dict = arc0.summary()
        self.assertTrue(status_dict['spc1'])
        self.assertTrue(any('different SCF references' in record for record in captured.output))
        self.assertTrue(any('external_instability' in record for record in captured.output))

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        projects = ['arc_project_for_testing_delete_after_usage_test_from_dict',
                    'arc_model_chemistry_test', 'arc_test', 'arc_test_rxn_multiplicity', 'test',
                    'unit_test_specific_job', 'wrong']
        for project in projects:
            project_directory = os.path.join(ARC_PATH, 'Projects', project)
            if os.path.isdir(project_directory):
                shutil.rmtree(project_directory, ignore_errors=True)


class TestLegacyRestartMultiplicity(unittest.TestCase):
    """
    Restarting a project whose restart.yml was written before the reaction multiplicity fix (issue #1048).
    The fixture pins the doublet reaction R <=> P and its TS0 (which has a geometry) to a multiplicity of 1.
    """

    fixture_path = os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '6_legacy_doublet_rxn_multiplicity',
                                'restart.yml')

    def setUp(self):
        """Copy the legacy restart file into a fresh project directory"""
        self.project_directory = tempfile.mkdtemp(prefix='arc_legacy_restart_')
        with open(self.fixture_path, 'r') as f:
            content = f.read().replace('PROJECT_DIRECTORY', self.project_directory)
        self.restart_path = os.path.join(self.project_directory, 'restart.yml')
        with open(self.restart_path, 'w') as f:
            f.write(content)

    def tearDown(self):
        """Delete the project directory"""
        shutil.rmtree(self.project_directory, ignore_errors=True)

    def write_original_input(self, multiplicity: int | None, file_name: str = 'input.yml'):
        """
        Write an original input file for the project.

        Args:
            multiplicity (int | None): The reaction multiplicity to set in the input file, ``None`` to not set one.
            file_name (str, optional): The input file name.
        """
        rxn = {'label': 'R <=> P', 'reactants': ['R'], 'products': ['P']}
        if multiplicity is not None:
            rxn['multiplicity'] = multiplicity
        save_yaml_file(path=os.path.join(self.project_directory, file_name),
                       content={'project': 'legacy_doublet_restart',
                                'species': [{'label': 'R', 'smiles': '[CH2]C(C)CC'},
                                            {'label': 'P', 'smiles': 'CC(C)[CH]C'}],
                                'reactions': [rxn]})

    def load(self, extra_species: list | None = None) -> tuple[ARC, list[str]]:
        """
        Load the legacy restart file as ARC.py does.

        Args:
            extra_species (list, optional): Species dictionaries to add to the restart dictionary.

        Returns:
            tuple[ARC, list[str]]: The ARC object and the WARNING (or louder) lines of the project's arc.log.
        """
        restart_dict = read_yaml_file(path=self.restart_path, project_directory=self.project_directory)
        restart_dict['species'].extend(extra_species or list())
        restart_dict['project_directory'] = self.project_directory  # as ARC.py sets it from the file's location
        arc = ARC(**restart_dict)
        with open(os.path.join(self.project_directory, 'arc.log'), 'r') as f:
            lines = f.read().splitlines()
        return arc, [line for line in lines if line.startswith(('Warning: ', 'Error: ', 'Critical: '))]

    def test_the_fixture_is_a_legacy_restart_file(self):
        """Test that the fixture carries the pinned multiplicity on the reaction and on a TS with a geometry"""
        restart_dict = read_yaml_file(path=self.restart_path, project_directory=self.project_directory)
        self.assertEqual(restart_dict['reactions'][0]['multiplicity'], 1)
        self.assertNotIn('multiplicity_source', restart_dict['reactions'][0])
        ts_dict = [spc for spc in restart_dict['species'] if spc['label'] == 'TS0'][0]
        self.assertEqual(ts_dict['multiplicity'], 1)
        self.assertIn('final_xyz', ts_dict)
        self.assertTrue(restart_dict['output'])

    def test_a_legacy_doublet_restart_without_an_original_input_is_corrected(self):
        """Test that a legacy doublet restart loads, takes the derived multiplicity, and says why"""
        irc_dict = {'label': 'IRC_TS0_1', 'irc_label': 'TS0', 'multiplicity': 1, 'charge': 0, 'compute_thermo': False,
                    'final_xyz': ARCSpecies(label='R', smiles='[CH2]C(C)CC').get_xyz()}
        arc, warnings = self.load(extra_species=[irc_dict])
        self.assertEqual(arc.reactions[0].multiplicity, 2)
        self.assertEqual(arc.reactions[0].multiplicity_source, 'derived')
        mult_lines = [line for line in warnings if 'stored multiplicity 1' in line]
        self.assertEqual(len(mult_lines), 1)
        self.assertIn('derived multiplicity 2', mult_lines[0])
        self.assertIn('No original input file', mult_lines[0])
        self.assertIn('multiplicity_source: user', mult_lines[0])
        for label in ['TS0', 'IRC_TS0_1']:
            self.assertNotIn(label, [spc.label for spc in arc.species])
            self.assertNotIn(label, arc.output)
            self.assertNotIn(label, arc.running_jobs)
        self.assertEqual(arc.restart_dict['reactions'][0]['multiplicity'], 2)
        self.assertEqual(arc.restart_dict['reactions'][0]['multiplicity_source'], 'derived')

    def test_orphaned_server_jobs_are_listed_with_their_cancel_commands(self):
        """Test that the WARNING gives the server, job id, and cancel command of each dropped running TS job"""
        with patch.dict(servers, {'server2': {'cluster_soft': 'Slurm'}, 'server3': {'cluster_soft': 'PBS'}}):
            _, warnings = self.load()
        ts_lines = [line for line in warnings if 'TS0' in line and 'orphaned' in line]
        self.assertEqual(len(ts_lines), 1)
        delete_command = settings['delete_command']
        self.assertIn(f"'{delete_command['Slurm']} 12345' on server server2", ts_lines[0])
        self.assertIn(f"'{delete_command['PBS']} 67890' on server server3", ts_lines[0])
        self.assertIn('ARC does not cancel them', ts_lines[0])

    def test_a_multiplicity_set_in_the_original_input_is_kept(self):
        """Test that a legacy restart keeps a multiplicity its original input file set, and warns"""
        self.write_original_input(multiplicity=4)
        arc, warnings = self.load()
        self.assertEqual(arc.reactions[0].multiplicity, 4)
        self.assertEqual(arc.reactions[0].multiplicity_source, 'user')
        self.assertEqual(len([line for line in warnings if 'stored multiplicity 1' in line]), 0)
        user_lines = [line for line in warnings if 'user-set multiplicity 4' in line]
        self.assertEqual(len(user_lines), 1)
        self.assertIn('derived multiplicity 2', user_lines[0])

    def test_a_multiplicity_not_set_in_the_original_input_is_derived(self):
        """Test that a legacy restart whose original input file sets no multiplicity takes the derived value"""
        self.write_original_input(multiplicity=None, file_name='input.yaml')
        arc, warnings = self.load()
        self.assertEqual(arc.reactions[0].multiplicity, 2)
        self.assertEqual(arc.reactions[0].multiplicity_source, 'derived')
        mult_lines = [line for line in warnings if 'stored multiplicity 1' in line]
        self.assertEqual(len(mult_lines), 1)
        self.assertIn('does not set it', mult_lines[0])
        self.assertIn('input.yaml', mult_lines[0])

    def test_a_ts_given_in_an_input_file_is_never_dropped(self):
        """Test that a TS species of an input file (not a restart file) is kept even without a matching multiplicity"""
        arc = ARC(project='legacy_doublet_restart', project_directory=self.project_directory, calc_freq_factor=False,
                  compute_thermo=False,
                  species=[{'label': 'R', 'smiles': '[CH2]C(C)CC'}, {'label': 'P', 'smiles': 'CC(C)[CH]C'},
                           {'label': 'TS0', 'is_ts': True, 'rxn_label': 'R <=> P', 'charge': 0}],
                  reactions=[{'label': 'R <=> P', 'reactants': ['R'], 'products': ['P']}])
        self.assertIn('TS0', [spc.label for spc in arc.species])
        self.assertEqual(arc.reactions[0].multiplicity, 2)

    def test_resolve_legacy_multiplicity_source(self):
        """Test how the original input file decides the source of a legacy restart multiplicity"""
        rxn_dict = {'label': 'R <=> P', 'reactants': ['R'], 'products': ['P'], 'multiplicity': 1}
        same_dict, note = resolve_legacy_multiplicity_source(rxn_dict, (None, list()), 'proj')
        self.assertEqual(same_dict['multiplicity_source'], 'derived')
        self.assertNotIn('multiplicity_source', rxn_dict)
        self.assertIn('No original input file', note)
        sourced = dict(rxn_dict, multiplicity_source='derived')
        self.assertIs(resolve_legacy_multiplicity_source(sourced, (None, list()), 'proj')[0], sourced)
        inputs = [{'label': 'R <=> P_b', 'reactants': ['R'], 'products': ['P'], 'multiplicity': 3},
                  {'label': 'R <=> P', 'reactants': ['R'], 'products': ['P'], 'multiplicity': 4}]
        resolved, note = resolve_legacy_multiplicity_source(rxn_dict, ('input.yml', inputs), 'proj')
        self.assertEqual((resolved['multiplicity'], resolved['multiplicity_source']), (4, 'user'))
        self.assertIn('set to 4', note)
        ambiguous = [{'reactants': ['R'], 'products': ['P'], 'multiplicity': 3},
                     {'reactants': ['R'], 'products': ['P'], 'multiplicity': 4}]
        resolved, _ = resolve_legacy_multiplicity_source(rxn_dict, ('input.yml', ambiguous), 'proj')
        self.assertEqual((resolved['multiplicity'], resolved['multiplicity_source']), (1, 'derived'))
        resolved, _ = resolve_legacy_multiplicity_source(rxn_dict, ('input.yml', ambiguous[:1]), 'proj')
        self.assertEqual((resolved['multiplicity'], resolved['multiplicity_source']), (3, 'user'))

    def test_an_input_file_written_from_a_restart_is_not_an_original_input(self):
        """Test that an input.yml holding an output dictionary, even an empty one, is not the original input"""
        self.write_original_input(multiplicity=4)
        content = read_yaml_file(os.path.join(self.project_directory, 'input.yml'))
        content['output'] = dict()
        save_yaml_file(path=os.path.join(self.project_directory, 'input.yml'), content=content)
        arc, warnings = self.load()
        self.assertEqual(arc.reactions[0].multiplicity, 2)
        self.assertTrue(any('No original input file' in line for line in warnings))

    def test_the_reset_ts_is_rebuilt_and_searched_at_the_derived_multiplicity(self):
        """Test that the Scheduler builds a fresh, unverified TS0 and spawns its TS search at multiplicity 2"""
        arc, _ = self.load()
        sched = Scheduler(project=arc.project, species_list=arc.species, rxn_list=arc.reactions,
                          ess_settings={'gaussian': ['local']}, project_directory=self.project_directory,
                          testing=True, restart_dict=arc.restart_dict, ts_adapters=['linear'],
                          job_types={'conf_opt': False, 'opt': True, 'freq': True, 'sp': True, 'rotors': False,
                                     'fine_grid': False})
        ts = sched.species_dict['TS0']
        self.assertEqual(ts.multiplicity, 2)
        self.assertIsNone(ts.final_xyz)
        self.assertFalse(ts.tsg_spawned)
        self.assertFalse(any(ts.ts_checks.values()))
        self.assertFalse(sched.output['TS0']['convergence'])
        self.assertFalse(sched.output['TS0']['paths']['geo'])
        self.assertEqual(sched.running_jobs.get('TS0', list()), list())
        with patch('arc.job.adapter.JobAdapter.execute'):
            sched.spawn_ts_jobs()
        self.assertTrue(ts.tsg_spawned)
        tsg_jobs = list(sched.job_dict['TS0']['tsg'].values())
        self.assertEqual(len(tsg_jobs), 1)
        self.assertEqual(tsg_jobs[0].multiplicity, 2)
        self.assertEqual(tsg_jobs[0].species_label, 'TS0')


class TestExecuteReleasesPooledConnections(unittest.TestCase):
    """The SSH connections a run holds open must be released by the run, not by interpreter exit."""

    @staticmethod
    def _arc():
        """An ARC object without the project setup __init__ does, which this does not need."""
        return ARC.__new__(ARC)

    def test_the_pool_is_released_when_the_run_finishes(self):
        """A consumer that never goes through ARC.py must still release its connections."""
        with patch.object(ARC, '_execute', return_value={'spc': 'converged'}), \
                patch('arc.main.reset_default_pool') as released:
            status = self._arc().execute()
        self.assertEqual(status, {'spc': 'converged'})
        released.assert_called_once()

    def test_the_pool_is_released_when_the_run_raises(self):
        """An interrupted or failed run is exactly when connections would otherwise be left open."""
        with patch.object(ARC, '_execute', side_effect=ValueError('the run went wrong')), \
                patch('arc.main.reset_default_pool') as released:
            self.assertRaises(ValueError, self._arc().execute)
        released.assert_called_once()

    def test_the_pool_is_released_on_a_keyboard_interrupt(self):
        """Ctrl-C is how a long run usually ends, and it is not an Exception."""
        with patch.object(ARC, '_execute', side_effect=KeyboardInterrupt), \
                patch('arc.main.reset_default_pool') as released:
            self.assertRaises(KeyboardInterrupt, self._arc().execute)
        released.assert_called_once()


class TestServerMappingBorrowsItsConnection(unittest.TestCase):
    """The connection the ESS survey opens is the one the run's jobs then need."""

    REMOTE = {'zeus': {'cluster_soft': 'PBS', 'address': 'z.example.edu', 'un': 'u'}}

    def _map_servers(self, found):
        """Survey the remote servers with every find_package() answering ``found``."""
        arc_object = ARC.__new__(ARC)
        arc_object.ess_settings = dict()
        with patch('arc.main.servers', self.REMOTE), \
                patch('arc.main.borrow_ssh_client') as borrow:
            borrow.return_value.__enter__.return_value.find_package.return_value = found
            arc_object.determine_ess_settings()
        return arc_object, borrow

    def test_one_connection_is_borrowed_per_server(self):
        """The survey asks after five packages, and used to open one connection for all of them."""
        _, borrow = self._map_servers(found=[])
        borrow.assert_called_once_with('zeus')

    def test_the_borrowed_connection_is_released(self):
        _, borrow = self._map_servers(found=[])
        borrow.return_value.__exit__.assert_called_once()

    def test_what_the_survey_finds_is_unchanged(self):
        """Borrowing instead of opening must not change the answer the survey gives."""
        arc_object, _ = self._map_servers(found=['/usr/bin/g16'])
        self.assertEqual(arc_object.ess_settings['gaussian'], ['zeus'])
        self.assertEqual(arc_object.ess_settings['orca'], ['zeus'])


class ReachedTheCleanup(Exception):
    """Raised to stop a run right after its check file cleanup, so the rest of the run is not needed."""


class SchedulerStub(object):
    """Stands in for a Scheduler that has finished running a project's jobs on a server."""

    def __init__(self, remote_project_paths: dict):
        self.remote_project_paths = remote_project_paths
        self.output = dict()
        self.species_dict = dict()
        self.rxn_list = list()


class TestCheckFileCleanup(unittest.TestCase):
    """
    Contains unit tests for deleting ESS checkfiles when ARC terminates, both locally and on the servers.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.project = 'arc_check_file_cleanup_test'
        cls.other_project = 'an_unrelated_arc_project'
        cls.server = 'server2'
        cls.server_settings = {'cluster_soft': 'Slurm',
                               'address': 'server2.host.edu',
                               'un': 'test_user',
                               'key': 'path_to_rsa_key',
                               }

    def setUp(self):
        """
        A method that is run before each unit test in this class.
        Set up a fake remote server: a temporary directory in which the commands ARC would have sent
        to a server are actually executed, so that the real cleanup code path is exercised.
        The server definition is pinned so that neither a user settings file nor a missing 'server2'
        entry can change the remote path this test builds and cleans.
        """
        self.remote_root = tempfile.mkdtemp()
        self.project_directory = os.path.join(tempfile.mkdtemp(), self.project)
        for patcher in [patch.dict('arc.job.adapter.servers', {self.server: self.server_settings}),
                        patch.dict('arc.job.ssh.servers', {self.server: self.server_settings}),
                        patch.object(SSHClient, 'connect', lambda ssh_client: None),
                        patch.object(SSHClient, '_send_command_to_server',
                                     lambda ssh_client, command, remote_path='':
                                     self.send_command_to_fake_server(command, remote_path))]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def send_command_to_fake_server(self, command: str, remote_path: str = '') -> tuple:
        """
        Execute a command in the fake remote server directory instead of sending it to a server.

        Args:
            command (str): The command to execute.
            remote_path (str, optional): The directory path at which the command will be executed.

        Returns: tuple[list, list]
            The lines of the standard output and of the standard error streams.
        """
        result = subprocess.run(command, shell=True, capture_output=True, text=True,
                                cwd=os.path.join(self.remote_root, remote_path))
        return result.stdout.splitlines(True), result.stderr.splitlines(True)

    def get_remote_project_path(self, project: str) -> str:
        """
        Get the remote path of a project's directory, as spawning a job on the server determines it.

        Args:
            project (str): The ARC project name.

        Returns: str
            The remote path of the project's directory.
        """
        job = GaussianAdapter(execution_type='queue',
                              job_type='opt',
                              level=Level(method='b3lyp', basis='6-31g'),
                              project=project,
                              project_directory=self.project_directory,
                              species=[ARCSpecies(label='spc1', smiles='C')],
                              server=self.server,
                              testing=True,
                              )
        return job.remote_project_path

    def set_up_arc_and_check_files(self, keep_checks: bool) -> ARC:
        """
        Create an ARC object running Gaussian on the fake remote server, along with the check files
        it would have left behind locally and remotely.

        Args:
            keep_checks (bool): Whether to keep ESS checkfiles when ARC terminates.

        Returns: ARC
            The ARC object.
        """
        arc0 = ARC(project=self.project,
                   project_directory=self.project_directory,
                   species=[ARCSpecies(label='spc1', smiles='CC', compute_thermo=False)],
                   level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)',
                   ess_settings={'gaussian': ['local', self.server]},
                   keep_checks=keep_checks,
                   )
        self.remote_project_paths = {self.server: self.get_remote_project_path(self.project)}
        self.local_check_path = os.path.join(arc0.project_directory, 'calcs', 'Species', 'spc1',
                                             'opt_a1', 'check.chk')
        self.remote_check_path = os.path.join(self.remote_root, self.remote_project_paths[self.server],
                                              'spc1', 'opt_a1', 'check.chk')
        self.remote_output_path = os.path.join(self.remote_root, self.remote_project_paths[self.server],
                                               'spc1', 'opt_a1', 'input.log')
        self.other_project_check_path = os.path.join(self.remote_root,
                                                     self.get_remote_project_path(self.other_project),
                                                     'spc2', 'opt_a1', 'check.chk')
        for path in [self.local_check_path, self.remote_check_path,
                     self.remote_output_path, self.other_project_check_path]:
            if not os.path.isdir(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            with open(path, 'w') as f:
                f.write('dummy file content')
        return arc0

    def test_check_files_are_deleted_locally_and_remotely(self):
        """Test that check files are deleted on the server as well as locally when keep_checks is False,
        and that only check files, and only those under this project's own remote directory, are deleted"""
        arc0 = self.set_up_arc_and_check_files(keep_checks=False)
        arc0.clean_check_files(remote_project_paths=self.remote_project_paths)
        self.assertFalse(os.path.isfile(self.local_check_path))
        self.assertFalse(os.path.isfile(self.remote_check_path))
        self.assertTrue(os.path.isfile(self.remote_output_path))
        self.assertTrue(os.path.isfile(self.other_project_check_path))

    def test_check_files_are_kept_locally_and_remotely(self):
        """Test that check files are kept on the server as well as locally when keep_checks is True"""
        arc0 = self.set_up_arc_and_check_files(keep_checks=True)
        arc0.clean_check_files(remote_project_paths=self.remote_project_paths)
        self.assertTrue(os.path.isfile(self.local_check_path))
        self.assertTrue(os.path.isfile(self.remote_check_path))
        self.assertTrue(os.path.isfile(self.remote_output_path))
        self.assertTrue(os.path.isfile(self.other_project_check_path))

    def test_check_files_are_deleted_locally_when_no_server_was_used(self):
        """Test that a project which only ran locally still has its local check files deleted"""
        arc0 = self.set_up_arc_and_check_files(keep_checks=False)
        arc0.clean_check_files()
        self.assertFalse(os.path.isfile(self.local_check_path))
        self.assertTrue(os.path.isfile(self.remote_check_path))

    def test_a_run_reaches_the_remote_cleanup_with_the_scheduler_remote_paths(self):
        """Test that executing a project actually deletes the server's check files, the cleanup is wired"""
        arc0 = self.set_up_arc_and_check_files(keep_checks=False)
        scheduler = SchedulerStub(remote_project_paths=self.remote_project_paths)
        with patch('arc.main.Scheduler', return_value=scheduler), \
                patch.object(ARC, 'delete_leftovers', side_effect=ReachedTheCleanup):
            with self.assertRaises(ReachedTheCleanup):
                arc0.execute()
        self.assertFalse(os.path.isfile(self.local_check_path))
        self.assertFalse(os.path.isfile(self.remote_check_path))
        self.assertTrue(os.path.isfile(self.remote_output_path))
        self.assertTrue(os.path.isfile(self.other_project_check_path))

    def tearDown(self):
        """
        A method that is run after each unit test in this class.
        Detach ARC's log file handler before removing the directory it writes into,
        so that a later test logging through it does not hit a deleted file.
        """
        arc_logger = get_logger()
        for handler in arc_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                arc_logger.removeHandler(handler)
        shutil.rmtree(self.remote_root, ignore_errors=True)
        shutil.rmtree(os.path.dirname(self.project_directory), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
