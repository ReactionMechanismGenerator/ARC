#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.main module
"""

import os
import shutil
import unittest

from rmgpy import settings
from rmgpy.data.rmg import RMGDatabase
from rmgpy.molecule.molecule import Molecule
from rmgpy.species import Species

from arc.common import read_yaml_file
from arc.exceptions import InputError
from arc.main import ARC
from arc.settings import arc_path, servers
from arc.species.species import ARCSpecies


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
        cls.job_types1 = {'conformers': True,
                          'opt': True,
                          'fine_grid': False,
                          'freq': True,
                          'sp': True,
                          'rotors': False,
                          'orbitals': False,
                          'lennard_jones': False,
                          'bde': True,
                          }

    def test_as_dict(self):
        """Test the as_dict() method of ARC"""
        spc1 = ARCSpecies(label='spc1', smiles='CC', generate_thermo=False)
        arc0 = ARC(project='arc_test', job_types=self.job_types1, job_shortcut_keywords={'gaussian': 'scf=(NDump=30)'},
                   arc_species_list=[spc1], level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)')
        restart_dict = arc0.as_dict()
        long_thermo_description = restart_dict['species'][0]['long_thermo_description']
        self.assertIn('Bond corrections:', long_thermo_description)
        self.assertIn("'C-C': 1", long_thermo_description)
        self.assertIn("'C-H': 6", long_thermo_description)
        expected_dict = {'composite_method': '',
                         'conformer_level': {'auxiliary_basis': '',
                                             'basis': 'def2svp',
                                             'method': 'apfd',
                                             'dispersion': ''},
                         'ts_guess_level': {'auxiliary_basis': '',
                                            'basis': 'def2svp',
                                            'method': 'apfd',
                                            'dispersion': ''},
                         'opt_level': {'auxiliary_basis': '',
                                       'basis': '6-311+g(3df,2p)',
                                       'method': 'b3lyp',
                                       'dispersion': ''},
                         'freq_level': {'auxiliary_basis': '',
                                        'basis': '6-311+g(3df,2p)',
                                        'method': 'b3lyp',
                                        'dispersion': ''},
                         'scan_level': {'auxiliary_basis': '', 'basis': '', 'method': '', 'dispersion': ''},
                         'orbitals_level': {'auxiliary_basis': '', 'basis': '', 'method': '', 'dispersion': ''},
                         'sp_level': {'auxiliary_basis': '',
                                      'basis': 'cc-pvdz-f12',
                                      'method': 'ccsd(t)-f12',
                                      'dispersion': ''},
                         'freq_scale_factor': 0.967,
                         'job_shortcut_keywords': {'gaussian': 'scf=(NDump=30)'},
                         'max_job_time': 120,
                         'model_chemistry': 'ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)',
                         'output': {},
                         'project': 'arc_test',
                         'running_jobs': {},
                         'reactions': [],
                         'job_memory': 14,
                         'job_types': {'rotors': False,
                                       'conformers': True,
                                       'fine': False,
                                       'freq': True,
                                       'onedmin': False,
                                       'opt': True,
                                       'orbitals': False,
                                       'bde': True,
                                       'sp': True},
                         'level_of_theory': '',
                         't_min': None,
                         't_max': None,
                         't_count': None,
                         'use_bac': True,
                         'confs_to_dft': 5,
                         'allow_nonisomorphic_2d': False,
                         'calc_freq_factor': True,
                         'ess_settings': {'gaussian': ['local', 'server2'], 'onedmin': ['server1'],
                                          'molpro': ['server2'], 'qchem': ['server1'], 'orca': ['local'],
                                          'terachem': ['server1']},
                         'species': [{'bond_corrections': {'C-C': 1, 'C-H': 6},
                                      'arkane_file': None,
                                      'charge': 0,
                                      'consider_all_diastereomers': True,
                                      'external_symmetry': None,
                                      'optical_isomers': None,
                                      'generate_thermo': False,
                                      'is_ts': False,
                                      'label': 'spc1',
                                      'long_thermo_description': long_thermo_description,
                                      'mol': '1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}\n2 C u0 p0 c0 {1,S} {6,S} {7,S} '
                                             '{8,S}\n3 H u0 p0 c0 {1,S}\n4 H u0 p0 c0 {1,S}\n5 H u0 p0 c0 {1,S}\n6 H '
                                             'u0 p0 c0 {2,S}\n7 H u0 p0 c0 {2,S}\n8 H u0 p0 c0 {2,S}\n',
                                      'multiplicity': 1,
                                      'neg_freqs_trshed': [],
                                      'number_of_rotors': 0,
                                      'force_field': 'MMFF94s',
                                      't1': None}],
                         'specific_job_type': '',
                         }
        self.assertEqual(restart_dict, expected_dict)

    def test_from_dict(self):
        """Test the from_dict() method of ARC"""
        restart_dict = {'composite_method': '',
                        'conformer_level': 'b97-d3/6-311+g(d,p)',
                        'fine': True,
                        'freq_level': 'wb97x-d3/6-311+g(d,p)',
                        'freq_scale_factor': 0.96,
                        'generate_conformers': True,
                        'job_shortcut_keywords': {'gaussian': 'scf=(NDump=30)'},
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
                                     'generate_thermo': False,
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
                        'use_bac': True}
        arc1 = ARC(project='wrong', freq_scale_factor=0.95)
        self.assertEqual(arc1.freq_scale_factor, 0.95)  # user input
        project = 'arc_project_for_testing_delete_after_usage1'
        project_directory = os.path.join(arc_path, 'Projects', project)
        arc1.from_dict(input_dict=restart_dict, project='testing_from_dict', project_directory=project_directory)
        self.assertEqual(arc1.freq_scale_factor, 0.96)  # loaded from the restart dict
        self.assertEqual(arc1.project, 'testing_from_dict')
        self.assertTrue('arc_project_for_testing_delete_after_usage' in arc1.project_directory)
        self.assertTrue(arc1.job_types['fine'])
        self.assertTrue(arc1.job_types['rotors'])
        self.assertEqual(arc1.sp_level, 'ccsdt-f12/cc-pvqz-f12')
        self.assertEqual(arc1.level_of_theory, '')
        self.assertEqual(arc1.arc_species_list[0].label, 'testing_spc1')
        self.assertFalse(arc1.arc_species_list[0].is_ts)
        self.assertEqual(arc1.arc_species_list[0].charge, 1)

    def test_from_dict_specific_job(self):
        """Test the from_dict() method of ARC"""
        restart_dict = {'specific_job_type': 'bde'}
        project = 'unit_test_specific_job'
        project_directory = os.path.join(arc_path, 'Projects', project)
        arc1 = ARC(project=project)
        arc1.from_dict(input_dict=restart_dict, project=project, project_directory=project_directory)
        job_type_expected = {'conformers': False, 'opt': True, 'freq': True, 'sp': True, 'rotors': False,
                             'orbitals': False, 'bde': True, 'onedmin': False, 'fine': True}
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

    def test_restart(self):
        """
        Test restarting ARC through the ARC class in main.py via the input_dict argument of the API
        Rather than through ARC.py. Check that all files are in place and the log file content.
        """
        restart_path = os.path.join(arc_path, 'arc', 'testing', 'restart', 'restart(H,H2O2,N2H3,CH3CO2).yml')
        project = 'arc_project_for_testing_delete_after_usage2'
        project_directory = os.path.join(arc_path, 'Projects', project)
        arc1 = ARC(project=project, input_dict=restart_path, project_directory=project_directory)
        arc1.execute()
        self.assertEqual(arc1.freq_scale_factor, 0.988)

        self.assertTrue(os.path.isfile(os.path.join(project_directory, 'output', 'thermo.info')))
        with open(os.path.join(project_directory, 'output', 'thermo.info'), 'r') as f:
            thermo_dft_ccsdtf12_bac = False
            for line in f.readlines():
                if 'thermo_DFT_CCSDTF12_BAC' in line:
                    thermo_dft_ccsdtf12_bac = True
                    break
        self.assertTrue(thermo_dft_ccsdtf12_bac)

        with open(os.path.join(project_directory, 'arc_project_for_testing_delete_after_usage2.info'), 'r') as f:
            sts, n2h3, oet, lot, ap = False, False, False, False, False
            for line in f.readlines():
                if 'Considered the following species and TSs:' in line:
                    sts = True
                elif 'Species N2H3' in line:
                    n2h3 = True
                elif 'Overall time since project initiation:' in line:
                    oet = True
                elif 'Levels of theory used:' in line:
                    lot = True
                elif 'ARC project arc_project_for_testing_delete_after_usage2' in line:
                    ap = True
        self.assertTrue(sts)
        self.assertTrue(n2h3)
        self.assertTrue(oet)
        self.assertTrue(lot)
        self.assertTrue(ap)

        with open(os.path.join(project_directory, 'arc.log'), 'r') as f:
            aei, ver, git, spc, rtm, ldb, therm, src, ter =\
                False, False, False, False, False, False, False, False, False
            for line in f.readlines():
                if 'ARC execution initiated on' in line:
                    aei = True
                elif '#   Version:' in line:
                    ver = True
                elif 'The current git HEAD for ARC is:' in line:
                    git = True
                elif 'Considering species: CH3CO2_rad' in line:
                    spc = True
                elif 'All jobs for species N2H3 successfully converged. Run time' in line:
                    rtm = True
                elif 'Loading the RMG database...' in line:
                    ldb = True
                elif 'Thermodynamics for H2O2' in line:
                    therm = True
                elif 'Sources of thermoproperties determined by RMG for the parity plots:' in line:
                    src = True
                elif 'ARC execution terminated on' in line:
                    ter = True
        self.assertTrue(aei)
        self.assertTrue(ver)
        self.assertTrue(git)
        self.assertTrue(spc)
        self.assertTrue(rtm)
        self.assertTrue(ldb)
        self.assertTrue(therm)
        self.assertTrue(src)
        self.assertTrue(ter)

        self.assertTrue(os.path.isfile(os.path.join(project_directory, 'output', 'thermo_parity_plots.pdf')))

        status = read_yaml_file(os.path.join(project_directory, 'output', 'status.yml'))
        self.assertEqual(status['CH3CO2_rad']['isomorphism'],
                         'opt passed isomorphism check; '
                         'Conformers optimized and compared at b3lyp/6-31g(d,p) empiricaldispersion=gd3bj; ')
        self.assertTrue(status['CH3CO2_rad']['job_types']['sp'])

        with open(os.path.join(project_directory, 'output', 'Species', 'H2O2', 'arkane', 'species_dictionary.txt'),
                  'r') as f:
            lines = f.readlines()
        adj_list = ''
        for line in lines:
            if 'H2O2' not in line:
                adj_list += line
            if line == '\n':
                break
        mol1 = Molecule().from_adjacency_list(adj_list)
        self.assertEqual(mol1.to_smiles(), 'OO')

        thermo_library_path = os.path.join(project_directory, 'output', 'RMG libraries', 'thermo',
                                           'arc_project_for_testing_delete_after_usage2.py')
        new_thermo_library_path = os.path.join(settings['database.directory'], 'thermo', 'libraries',
                                               'arc_project_for_testing_delete_after_usage2.py')
        # copy the generated library to RMG-database
        shutil.copyfile(thermo_library_path, new_thermo_library_path)
        db = RMGDatabase()
        db.load(
            path=settings['database.directory'],
            thermo_libraries=['arc_project_for_testing_delete_after_usage2'],
            transport_libraries=[],
            reaction_libraries=[],
            seed_mechanisms=[],
            kinetics_families='none',
            kinetics_depositories=[],
            statmech_libraries=None,
            depository=False,
            solvation=False,
            testing=True,
        )

        spc2 = Species().from_smiles('CC([O])=O')
        spc2.generate_resonance_structures()
        spc2.thermo = db.thermo.get_thermo_data(spc2)
        self.assertAlmostEqual(spc2.get_enthalpy(298), -212439.26998495663, 1)
        self.assertAlmostEqual(spc2.get_entropy(298), 283.3972662956835, 1)
        self.assertAlmostEqual(spc2.get_heat_capacity(1000), 118.751379824224, 1)
        self.assertTrue('arc_project_for_testing_delete_after_usage2' in spc2.thermo.comment)

        # delete the generated library from RMG-database
        os.remove(new_thermo_library_path)

    def test_determine_model_chemistry_and_freq_scale_factor(self):
        """Test determining the model chemistry and the frequency scaling factor"""
        arc0 = ARC(project='arc_model_chemistry_test_0', level_of_theory='CBS-QB3')
        self.assertEqual(arc0.model_chemistry, 'cbs-qb3')
        self.assertEqual(arc0.freq_scale_factor, 1.00386)  # 0.99 * 1.014 = 1.00386

        arc1 = ARC(project='arc_model_chemistry_test_1', level_of_theory='CBS-QB3-Paraskevas')
        self.assertEqual(arc1.model_chemistry, 'cbs-qb3-paraskevas')
        self.assertEqual(arc1.freq_scale_factor, 1.00386)  # 0.99 * 1.014 = 1.00386
        self.assertEqual(arc1.use_bac, True)

        arc2 = ARC(project='arc_model_chemistry_test_2',
                   level_of_theory='ccsd(t)-f12/cc-pvtz-f12//m06-2x/cc-pvtz')
        self.assertEqual(arc2.model_chemistry, 'ccsd(t)-f12/cc-pvtz-f12//m06-2x/cc-pvtz')
        self.assertEqual(arc2.freq_scale_factor, 0.955)

        arc3 = ARC(project='arc_model_chemistry_test_3',
                   sp_level='ccsd(t)-f12/cc-pvtz-f12', opt_level='wb97x-d/aug-cc-pvtz')
        self.assertEqual(arc3.model_chemistry, 'ccsd(t)-f12/cc-pvtz-f12//wb97x-d/aug-cc-pvtz')
        self.assertEqual(arc3.freq_scale_factor, 0.988)

    def test_determine_model_chemistry_for_job_types(self):
        """Test determining the model chemistry specification dictionary for job types"""
        # Test conflicted inputs: specify both level_of_theory and composite_method
        with self.assertRaises(InputError):
            ARC(project='test', level_of_theory='ccsd(t)-f12/cc-pvtz-f12//wb97x-d/aug-cc-pvtz',
                composite_method='cbs-qb3')

        # Test illegal level of theory specification (method contains multiple slashes)
        with self.assertRaises(InputError):
            ARC(project='test', level_of_theory='dlpno-mp2-f12/D/cc-pVDZ(fi/sf/fw)//b3lyp/G/def2svp')

        # Test illegal job level specification (method contains multiple slashes)
        with self.assertRaises(InputError):
            ARC(project='test', opt_level='b3lyp/d/def2tzvp/def2tzvp/c')

        # Test illegal job level specification (method contains empty space)
        with self.assertRaises(InputError):
            ARC(project='test', opt_level='b3lyp/def2tzvp def2tzvp/c')

        # Test direct job level specification conflicts with level of theory specification
        with self.assertRaises(InputError):
            ARC(project='test', level_of_theory='b3lyp/sto-3g', opt_level='wb97xd/def2tzvp')

        # Test illegal level of theory specification (semi-empirical method)
        with self.assertRaises(InputError):
            ARC(project='test', level_of_theory='AM1')

        # Test deduce formatted levels from default method from settings.py
        arc1 = ARC(project='test')
        expected_opt_level = {'method': 'wb97xd', 'basis': 'def2tzvp', 'auxiliary_basis': '', 'dispersion': ''}
        expected_freq_level = {'method': 'wb97xd', 'basis': 'def2tzvp', 'auxiliary_basis': '', 'dispersion': ''}
        expected_sp_level = {'method': 'ccsd(t)-f12', 'basis': 'cc-pvtz-f12', 'auxiliary_basis': '', 'dispersion': ''}
        self.assertEqual(arc1.opt_level, expected_opt_level)
        self.assertEqual(arc1.freq_level, expected_freq_level)
        self.assertEqual(arc1.sp_level, expected_sp_level)

        # Test deduce formatted levels from composite method specification
        arc2 = ARC(project='test', composite_method='cbs-qb3')
        expected_opt_level = {'method': '', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}
        expected_freq_level = {'method': 'b3lyp', 'basis': 'cbsb7', 'auxiliary_basis': '', 'dispersion': ''}
        expected_sp_level = {'method': '', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}
        expected_scan_level = {'method': 'b3lyp', 'basis': 'cbsb7', 'auxiliary_basis': '', 'dispersion': ''}
        expected_orbitals_level = {'method': '', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}
        expected_composite_method = 'cbs-qb3'
        self.assertEqual(arc2.opt_level, expected_opt_level)
        self.assertEqual(arc2.freq_level, expected_freq_level)
        self.assertEqual(arc2.sp_level, expected_sp_level)
        self.assertEqual(arc2.scan_level, expected_scan_level)
        self.assertEqual(arc2.orbitals_level, expected_orbitals_level)
        self.assertEqual(arc2.composite_method, expected_composite_method)

        # Test deduce formatted levels from level of theory specification
        arc3 = ARC(project='test', level_of_theory='ccsd(t)-f12/cc-pvtz-f12//wb97x-d/aug-cc-pvtz')
        expected_opt_level = {'method': 'wb97x-d', 'basis': 'aug-cc-pvtz', 'auxiliary_basis': '', 'dispersion': ''}
        expected_freq_level = {'method': 'wb97x-d', 'basis': 'aug-cc-pvtz', 'auxiliary_basis': '', 'dispersion': ''}
        expected_sp_level = {'method': 'ccsd(t)-f12', 'basis': 'cc-pvtz-f12', 'auxiliary_basis': '', 'dispersion': ''}
        expected_scan_level = {'method': 'wb97x-d', 'basis': 'aug-cc-pvtz', 'auxiliary_basis': '', 'dispersion': ''}
        expected_orbitals_level = {'method': '', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}
        self.assertEqual(arc3.opt_level, expected_opt_level)
        self.assertEqual(arc3.freq_level, expected_freq_level)
        self.assertEqual(arc3.sp_level, expected_sp_level)
        self.assertEqual(arc3.scan_level, expected_scan_level)
        self.assertEqual(arc3.orbitals_level, expected_orbitals_level)

        # Test deduce formatted levels from job level specification with complex names
        arc4 = ARC(project='test', opt_level='wb97x-d3/6-311++G(3df,3pd)', freq_level='wb97M/6-311+G*-J',
                   sp_level='DLPNO-CCSD(T)-F12/cc-pVTZ-F12', calc_freq_factor=False)
        expected_opt_level = {'method': 'wb97x-d3', 'basis': '6-311++g(3df,3pd)', 'auxiliary_basis': '',
                              'dispersion': ''}
        expected_freq_level = {'method': 'wb97m', 'basis': '6-311+g*-j', 'auxiliary_basis': '', 'dispersion': ''}
        expected_sp_level = {'method': 'dlpno-ccsd(t)-f12', 'basis': 'cc-pvtz-f12', 'auxiliary_basis': '',
                             'dispersion': ''}
        self.assertEqual(arc4.opt_level, expected_opt_level)
        self.assertEqual(arc4.freq_level, expected_freq_level)
        self.assertEqual(arc4.sp_level, expected_sp_level)

        # Test deduce formatted levels from incomplete level of theory specification
        # e.g., if level_of_theory = b3lyp/sto-3g, assume user meant to run opt, freq, sp all at this level
        arc5 = ARC(project='test', level_of_theory='b3lyp/sto-3g', calc_freq_factor=False)
        expected_opt_level = {'method': 'b3lyp', 'basis': 'sto-3g', 'auxiliary_basis': '', 'dispersion': ''}
        expected_freq_level = {'method': 'b3lyp', 'basis': 'sto-3g', 'auxiliary_basis': '', 'dispersion': ''}
        expected_sp_level = {'method': 'b3lyp', 'basis': 'sto-3g', 'auxiliary_basis': '', 'dispersion': ''}
        self.assertEqual(arc5.opt_level, expected_opt_level)
        self.assertEqual(arc5.freq_level, expected_freq_level)
        self.assertEqual(arc5.sp_level, expected_sp_level)

        # Test direct job level specification does NOT conflict with level of theory specification
        arc6 = ARC(project='test', level_of_theory='b3lyp/sto-3g', opt_level='b3lyp/sto-3g', calc_freq_factor=False)
        expected_opt_level = {'method': 'b3lyp', 'basis': 'sto-3g', 'auxiliary_basis': '', 'dispersion': ''}
        expected_freq_level = {'method': 'b3lyp', 'basis': 'sto-3g', 'auxiliary_basis': '', 'dispersion': ''}
        expected_sp_level = {'method': 'b3lyp', 'basis': 'sto-3g', 'auxiliary_basis': '', 'dispersion': ''}
        self.assertEqual(arc6.opt_level, expected_opt_level)
        self.assertEqual(arc6.freq_level, expected_freq_level)
        self.assertEqual(arc6.sp_level, expected_sp_level)

        # Test deduce freq level from opt level
        arc7 = ARC(project='test', opt_level='b3lyp/sto-3g', calc_freq_factor=False)
        expected_opt_level = {'method': 'b3lyp', 'basis': 'sto-3g', 'auxiliary_basis': '', 'dispersion': ''}
        expected_freq_level = {'method': 'b3lyp', 'basis': 'sto-3g', 'auxiliary_basis': '', 'dispersion': ''}
        self.assertEqual(arc7.opt_level, expected_opt_level)
        self.assertEqual(arc7.freq_level, expected_freq_level)

        # Test dictionary format specification with auxiliary basis and DFT dispersion
        arc8 = ARC(project='test', opt_level={},
                   freq_level={'method': 'B3LYP/G', 'basis': 'cc-pVDZ(fi/sf/fw)', 'auxiliary_basis': 'def2-svp/C'},
                   sp_level={'method': 'DLPNO-CCSD(T)-F12', 'basis': 'cc-pVTZ-F12',
                             'auxiliary_basis': 'aug-cc-pVTZ/C cc-pVTZ-F12-CABS',
                             'dispersion': 'DEF2-tzvp/c'},
                   calc_freq_factor=False)
        expected_opt_level = {'method': 'wb97xd', 'basis': 'def2tzvp', 'auxiliary_basis': '', 'dispersion': ''}
        expected_freq_level = {'method': 'b3lyp/g', 'basis': 'cc-pvdz(fi/sf/fw)', 'auxiliary_basis': 'def2-svp/c',
                               'dispersion': ''}
        expected_sp_level = {'method': 'dlpno-ccsd(t)-f12', 'basis': 'cc-pvtz-f12',
                             'auxiliary_basis': 'aug-cc-pvtz/c cc-pvtz-f12-cabs', 'dispersion': 'def2-tzvp/c'}
        self.assertEqual(arc8.opt_level, expected_opt_level)
        self.assertEqual(arc8.freq_level, expected_freq_level)
        self.assertEqual(arc8.sp_level, expected_sp_level)

        # Test using default frequency and orbital level for composite job, also forbid rotors job
        arc9 = ARC(project='test', composite_method='cbs-qb3', calc_freq_factor=False,
                   job_types={'rotors': False, 'orbitals': True})
        expected_freq_level = {'method': 'b3lyp', 'basis': 'cbsb7', 'auxiliary_basis': '', 'dispersion': ''}
        expected_scan_level = {'method': '', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}
        expected_orbitals_level = {'method': 'b3lyp', 'basis': 'cbsb7', 'auxiliary_basis': '', 'dispersion': ''}
        self.assertEqual(arc9.freq_level, expected_freq_level)
        self.assertEqual(arc9.scan_level, expected_scan_level)
        self.assertEqual(arc9.orbitals_level, expected_orbitals_level)

        # Test using specified frequency, scan, and orbital for composite job
        arc10 = ARC(project='test', composite_method='cbs-qb3', freq_level='wb97xd/6-311g', scan_level='apfd/def2svp',
                    orbitals_level='hf/sto-3g', job_types={'orbitals': True}, calc_freq_factor=False)
        expected_scan_level = {'method': 'apfd', 'basis': 'def2svp', 'auxiliary_basis': '', 'dispersion': ''}
        expected_freq_level = {'method': 'wb97xd', 'basis': '6-311g', 'auxiliary_basis': '', 'dispersion': ''}
        expected_orbitals_level = {'method': 'hf', 'basis': 'sto-3g', 'auxiliary_basis': '', 'dispersion': ''}
        self.assertEqual(arc10.scan_level, expected_scan_level)
        self.assertEqual(arc10.freq_level, expected_freq_level)
        self.assertEqual(arc10.orbitals_level, expected_orbitals_level)

        # Test using default frequency and orbital level for job specified from level of theory, also forbid rotors job
        arc11 = ARC(project='test', level_of_theory='b3lyp/sto-3g', calc_freq_factor=False,
                    job_types={'rotors': False, 'orbitals': True})
        expected_scan_level = {'method': '', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}
        expected_orbitals_level = {'method': 'wb97x-d3', 'basis': '6-311++g(d,p)', 'auxiliary_basis': '',
                                   'dispersion': ''}
        self.assertEqual(arc11.scan_level, expected_scan_level)
        self.assertEqual(arc11.orbitals_level, expected_orbitals_level)

        # Test using specified scan level
        arc12 = ARC(project='test', level_of_theory='b3lyp/sto-3g', calc_freq_factor=False, scan_level='apfd/def2svp',
                    job_types={'rotors': True})
        expected_scan_level = {'method': 'apfd', 'basis': 'def2svp', 'auxiliary_basis': '', 'dispersion': ''}
        self.assertEqual(arc12.scan_level, expected_scan_level)

        # Test specifying semi-empirical and force-field methods using dictionary
        arc13 = ARC(project='test', opt_level={'method': 'AM1'}, freq_level={'method': 'PM6'},
                    sp_level={'method': 'AMBER'}, calc_freq_factor=False)
        expected_opt_level = {'method': 'am1', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}
        expected_freq_level = {'method': 'pm6', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}
        expected_sp_level = {'method': 'amber', 'basis': '', 'auxiliary_basis': '', 'dispersion': ''}
        self.assertEqual(arc13.opt_level, expected_opt_level)
        self.assertEqual(arc13.freq_level, expected_freq_level)
        self.assertEqual(arc13.sp_level, expected_sp_level)

    def test_determine_unique_species_labels(self):
        """Test the determine_unique_species_labels method"""
        spc0 = ARCSpecies(label='spc0', smiles='CC', generate_thermo=False)
        spc1 = ARCSpecies(label='spc1', smiles='CC', generate_thermo=False)
        spc2 = ARCSpecies(label='spc2', smiles='CC', generate_thermo=False)
        arc0 = ARC(project='arc_test', job_types=self.job_types1, arc_species_list=[spc0, spc1, spc2],
                   level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)')
        self.assertEqual(arc0.unique_species_labels, ['spc0', 'spc1', 'spc2'])
        spc3 = ARCSpecies(label='spc0', smiles='CC', generate_thermo=False)
        arc0.arc_species_list.append(spc3)
        with self.assertRaises(ValueError):
            arc0.determine_unique_species_labels()

    def test_add_hydrogen_for_bde(self):
        """Test the add_hydrogen_for_bde method"""
        spc0 = ARCSpecies(label='spc0', smiles='CC', generate_thermo=False)
        arc0 = ARC(project='arc_test', job_types=self.job_types1, arc_species_list=[spc0],
                   level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)')
        arc0.add_hydrogen_for_bde()
        self.assertEqual(len(arc0.arc_species_list), 1)

        spc1 = ARCSpecies(label='spc1', smiles='CC', generate_thermo=False, bdes=['all_h'])
        arc1 = ARC(project='arc_test', job_types=self.job_types1, arc_species_list=[spc1],
                   level_of_theory='ccsd(t)-f12/cc-pvdz-f12//b3lyp/6-311+g(3df,2p)')
        arc1.add_hydrogen_for_bde()
        self.assertEqual(len(arc1.arc_species_list), 2)
        self.assertIn('H', [spc.label for spc in arc1.arc_species_list])

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        # projects = ['arc_project_for_testing_delete_after_usage1', 'arc_project_for_testing_delete_after_usage2',
        #             'ar c', 'ar:c', 'ar<c', 'ar%c']
        projects = ['arc_project_for_testing_delete_after_usage1',
                    'ar c', 'ar:c', 'ar<c', 'ar%c']
        for project in projects:
            project_directory = os.path.join(arc_path, 'Projects', project)
            shutil.rmtree(project_directory)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
