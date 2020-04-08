#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.job module
"""

import datetime
import math
import os
import shutil
import unittest

from arc.exceptions import InputError
from arc.job.job import Job
from arc.settings import arc_path


class TestJob(unittest.TestCase):
    """
    Contains unit tests for the Job class
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.ess_settings = {'gaussian': ['server1', 'server2'], 'molpro': ['server2'],
                            'qchem': ['server1'], 'onedmin': ['server1'], 'orca': ['server1'], 'terachem': ['server1']}
        cls.xyz_c = {'symbols': ('C',), 'isotopes': (12,), 'coords': ((0.0, 0.0, 0.0),)}
        cls.job1 = Job(project='arc_project_for_testing_delete_after_usage3', ess_settings=cls.ess_settings,
                       species_name='tst_spc', xyz=cls.xyz_c, job_type='opt',
                       job_level_of_theory_dict={'method': 'b3lyp', 'basis': '6-31+g(d)'},
                       multiplicity=1, fine=True, job_num=100,
                       testing=True, project_directory=os.path.join(arc_path, 'Projects', 'project_test'))
        cls.job1.initial_time = datetime.datetime(2019, 3, 15, 19, 53, 7, 0)
        cls.job1.final_time = datetime.datetime(2019, 3, 15, 19, 53, 8, 0)
        cls.job1.determine_run_time()

    def test_as_dict(self):
        """Test Job.as_dict()"""
        job_dict = self.job1.as_dict()
        initial_time = job_dict['initial_time']
        final_time = job_dict['final_time']
        expected_dict = {'initial_time': initial_time,
                         'final_time': final_time,
                         'cpu_cores': 8,
                         'ess_settings': {'gaussian': ['server1', 'server2'], 'terachem': ['server1'],
                                          'molpro': [u'server2'], 'onedmin': [u'server1'], 'qchem': [u'server1'],
                                          'orca': ['server1']},
                         'species_name': 'tst_spc',
                         'is_ts': False,
                         'fine': True,
                         'job_id': 0,
                         'job_name': 'opt_a100',
                         'job_num': 100,
                         'job_server_name': 'a100',
                         'job_status': ['initializing',
                                        {'status': 'initializing', 'keywords': list(), 'error': '', 'line': ''}],
                         'job_type': 'opt',
                         'job_level_of_theory_dict': {'method': 'b3lyp', 'basis': '6-31+g(d)'},
                         'total_job_memory_gb': 14,
                         'multiplicity': 1,
                         'project': 'arc_project_for_testing_delete_after_usage3',
                         'project_directory': os.path.join(arc_path, 'Projects', 'project_test'),
                         'server': 'server1',
                         'max_job_time': 120,
                         'scan_res': 8.0,
                         'software': 'gaussian',
                         'xyz': 'C       0.00000000    0.00000000    0.00000000'}
        self.assertEqual(job_dict, expected_dict)

    def test_from_dict(self):
        """Test Job.from_dict()"""
        job_dict = self.job1.as_dict()
        job = Job(job_dict=job_dict)
        self.assertEqual(job.multiplicity, 1)
        self.assertEqual(job.charge, 0)
        self.assertEqual(job.species_name, 'tst_spc')
        self.assertEqual(job.server, 'server1')
        self.assertEqual(job.job_level_of_theory_dict, {'method': 'm062x', 'basis': '6-311g'})
        self.assertEqual(job.job_type, 'scan')
        self.assertEqual(job.project_directory.split('/')[-1], 'project_test')
        self.assertEqual(job.method, 'm062x')
        self.assertEqual(job.basis_set, '6-311g')
        self.assertFalse(job.is_ts)

    def test_automatic_ess_assignment(self):
        """Test that the Job module correctly assigns a software for specific methods and basis sets"""
        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz=self.xyz_c,
                   job_type='opt', job_level_of_theory_dict={'method': 'b3lyp', 'basis': '6-311++G(d,p)'},
                   multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'gaussian')

        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz=self.xyz_c,
                   job_type='opt', job_level_of_theory_dict={'method': 'ccsd(t)', 'basis': 'avtz'},
                   multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'molpro')

        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz=self.xyz_c,
                   job_type='opt', job_level_of_theory_dict={'method': 'wb97xd', 'basis': '6-311++g(d,p)'},
                   multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'terachem')

        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz=self.xyz_c,
                   job_type='opt', job_level_of_theory_dict={'method': 'wb97x-d3', 'basis': '6-311++g(d,p)'},
                   multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'qchem')

        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz=self.xyz_c,
                   job_type='opt', job_level_of_theory_dict={'method': 'b97', 'basis': '6-311++g(d,p)'},
                   multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'gaussian')

        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz=self.xyz_c,
                   job_type='opt', job_level_of_theory_dict={'method': 'm062x', 'basis': '6-311++g(d,p)'},
                   multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'gaussian')

        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz=self.xyz_c,
                   job_type='opt', job_level_of_theory_dict={'method': 'm06-2x', 'basis': '6-311++g(d,p)'},
                   multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'qchem')

        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz=self.xyz_c,
                   job_type='scan', job_level_of_theory_dict={'method': 'm062x', 'basis': '6-311++g(d,p)'},
                   multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'gaussian')

        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz=self.xyz_c,
                   job_type='scan', job_level_of_theory_dict={'method': 'm06-2x', 'basis': '6-311++g(d,p)'},
                   multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'qchem')
        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz=self.xyz_c,
                   job_type='sp', job_level_of_theory_dict={'method': 'dlpno-ccsd(t)', 'basis': 'aug-cc-pvtz',
                                                            'auxiliary_basis':'aug-cc-pvtz/c'}, multiplicity=1,
                   testing=True, project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True,
                   job_num=100)
        self.assertEqual(job0.software, 'orca')

        self.assertEqual(job0.total_job_memory_gb, 14)
        self.assertEqual(job0.max_job_time, 120)

    def test_bath_gas(self):
        """Test correctly assigning the bath_gas attribute"""
        self.assertIsNone(self.job1.bath_gas)

        job2 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc',
                   xyz=self.xyz_c, job_type='onedmin',
                   job_level_of_theory_dict={'method': 'b3lyp', 'basis': '6-31+g(d)'}, multiplicity=1,
                   testing=True, project_directory=os.path.join(arc_path, 'Projects', 'project_test'),
                   fine=True, job_num=100)
        self.assertEqual(job2.bath_gas, 'N2')

        job2 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc',
                   xyz=self.xyz_c, job_type='onedmin',
                   job_level_of_theory_dict={'method': 'b3lyp', 'basis': '6-31+g(d)'}, multiplicity=1,
                   testing=True, project_directory=os.path.join(arc_path, 'Projects', 'project_test'),
                   fine=True, job_num=100, bath_gas='Ar')
        self.assertEqual(job2.bath_gas, 'Ar')

    def test_deduce_software(self):
        """Test deducing the ESS software"""
        self.job1.job_type = 'onedmin'
        self.job1.software = None
        self.job1.deduce_software()
        self.assertEqual(self.job1.software, 'onedmin')

        self.job1.job_type = 'orbitals'
        self.job1.software = None
        self.job1.deduce_software()
        self.assertEqual(self.job1.software, 'qchem')

        self.job1.job_type = 'composite'
        self.job1.software = None
        self.job1.deduce_software()
        self.assertEqual(self.job1.software, 'gaussian')

        # test the levels_ess dict from settings
        self.job1.job_type = 'opt'
        self.job1.job_level_of_theory_dict = {'method': 'm06-2x', 'basis': '6-311g'}
        self.job1.software = None
        self.job1.deduce_software()
        self.assertEqual(self.job1.software, 'qchem')

        self.job1.job_type = 'opt'
        self.job1.job_level_of_theory_dict = {'method': 'ccsd(t)', 'basis': 'cc-pvtz'}
        self.job1.method = 'ccsd(t)'
        self.job1.software = None
        self.job1.deduce_software()
        self.assertEqual(self.job1.software, 'molpro')

        self.job1.job_type = 'opt'
        self.job1.job_level_of_theory_dict = {'method': 'wb97xd', 'basis': '6-311g'}
        self.job1.method = 'wb97xd'
        self.job1.software = None
        self.job1.deduce_software()
        self.assertEqual(self.job1.software, 'terachem')

        self.job1.job_type = 'scan'
        self.job1.job_level_of_theory_dict = {'method': 'm062x', 'basis': '6-311g'}
        self.job1.method = 'm062x'
        self.job1.software = None
        self.job1.deduce_software()
        self.assertEqual(self.job1.software, 'gaussian')

        # return to original value
        self.job1.level_of_theory = 'b3lyp/6-31+g(d)'
        self.job1.method = 'b3lyp'
        self.job1.software = None
        self.job1.deduce_software()
        self.assertEqual(self.job1.software, 'gaussian')

    def test_set_cpu_and_mem(self):
        """Test assigning number of cpu's and memory"""
        self.job1.cpu_cores = 8
        self.job1.total_job_memory_gb = 14
        self.job1.input_file_memory = None
        self.job1.submit_script_memory = None
        self.job1.server = 'server2'
        self.job1.software = 'molpro'
        self.job1.set_cpu_and_mem()
        self.assertEqual(self.job1.cpu_cores, 8)
        expected_memory = math.ceil(14 * 128 / 8)
        self.assertEqual(self.job1.input_file_memory, expected_memory)

        self.job1.server = 'server1'
        self.job1.cpu_cores = None
        self.job1.set_cpu_and_mem()
        self.assertEqual(self.job1.cpu_cores, 8)
        expected_memory = math.ceil(14 * 128 / 8)
        self.assertEqual(self.job1.input_file_memory, expected_memory)

        self.job1.cpu_cores = None
        self.job1.input_file_memory = None
        self.job1.submit_script_memory = None
        self.job1.server = 'server2'
        self.job1.software = 'terachem'
        self.job1.set_cpu_and_mem()
        self.assertEqual(self.job1.cpu_cores, 8)
        expected_memory = math.ceil(14 * 128 / 8)
        self.assertEqual(self.job1.input_file_memory, expected_memory)

        self.job1.cpu_cores = 8
        self.job1.input_file_memory = None
        self.job1.submit_script_memory = None
        self.job1.server = 'server2'
        self.job1.software = 'gaussian'
        self.job1.set_cpu_and_mem()
        self.assertEqual(self.job1.cpu_cores, 8)
        expected_memory = math.ceil(14 * 1024)
        self.assertEqual(self.job1.input_file_memory, expected_memory)

        self.job1.cpu_cores = 48
        self.job1.input_file_memory = None
        self.job1.submit_script_memory = None
        self.job1.server = 'server2'
        self.job1.software = 'orca'
        self.job1.set_cpu_and_mem()
        self.assertEqual(self.job1.cpu_cores, 48)
        expected_memory = math.ceil(14 * 1024 / 48)
        self.assertEqual(self.job1.input_file_memory, expected_memory)

        self.job1.cpu_cores = 48
        self.job1.input_file_memory = None
        self.job1.submit_script_memory = None
        self.job1.server = 'server2'
        self.job1.software = 'qchem'
        self.job1.set_cpu_and_mem()
        self.assertEqual(self.job1.cpu_cores, 48)
        self.assertEqual(self.job1.input_file_memory, 14)

    def test_set_file_paths(self):
        """Test setting file paths"""
        self.job1.job_type = 'onedmin'
        self.job1.set_file_paths()
        self.assertEqual(len(self.job1.additional_files_to_upload), 3)

        self.assertEqual(self.job1.additional_files_to_upload[0]['source'], 'path')
        self.assertEqual(self.job1.additional_files_to_upload[0]['name'], 'geo')
        self.assertFalse(self.job1.additional_files_to_upload[0]['make_x'])
        self.assertIn('geo.xyz', self.job1.additional_files_to_upload[0]['remote'])
        self.assertIn('geo.xyz', self.job1.additional_files_to_upload[0]['local'])

        self.assertEqual(self.job1.additional_files_to_upload[1]['source'], 'input_files')
        self.assertEqual(self.job1.additional_files_to_upload[1]['name'], 'm.x')
        self.assertTrue(self.job1.additional_files_to_upload[1]['make_x'])
        self.assertIn('m.x', self.job1.additional_files_to_upload[1]['remote'])
        self.assertEqual(self.job1.additional_files_to_upload[1]['local'], 'onedmin.molpro.x')

        self.assertEqual(self.job1.additional_files_to_upload[2]['source'], 'input_files')
        self.assertEqual(self.job1.additional_files_to_upload[2]['name'], 'qc.mol')
        self.assertFalse(self.job1.additional_files_to_upload[2]['make_x'])
        self.assertIn('qc.mol', self.job1.additional_files_to_upload[2]['remote'])
        self.assertEqual(self.job1.additional_files_to_upload[2]['local'], 'onedmin.qc.mol')

        self.job1.job_type = 'gromacs'
        self.job1.set_file_paths()
        self.assertEqual(len(self.job1.additional_files_to_upload), 6)

        self.assertEqual(self.job1.additional_files_to_upload[0]['source'], 'path')
        self.assertEqual(self.job1.additional_files_to_upload[0]['name'], 'gaussian.out')
        self.assertFalse(self.job1.additional_files_to_upload[0]['make_x'])
        self.assertIn('gaussian.out', self.job1.additional_files_to_upload[0]['remote'])
        self.assertIn('gaussian.out', self.job1.additional_files_to_upload[0]['local'])

        self.assertEqual(self.job1.additional_files_to_upload[1]['source'], 'path')
        self.assertEqual(self.job1.additional_files_to_upload[1]['name'], 'coords.yml')
        self.assertFalse(self.job1.additional_files_to_upload[1]['make_x'])
        self.assertIn('coords.yml', self.job1.additional_files_to_upload[1]['remote'])

        self.assertEqual(self.job1.additional_files_to_upload[2]['source'], 'path')
        self.assertEqual(self.job1.additional_files_to_upload[2]['name'], 'acpype.py')
        self.assertFalse(self.job1.additional_files_to_upload[2]['make_x'])
        self.assertIn('acpype.py', self.job1.additional_files_to_upload[2]['remote'])
        self.assertIn('acpype.py', self.job1.additional_files_to_upload[2]['local'])

        self.assertEqual(self.job1.additional_files_to_upload[3]['source'], 'path')
        self.assertEqual(self.job1.additional_files_to_upload[3]['name'], 'mdconf.py')
        self.assertFalse(self.job1.additional_files_to_upload[3]['make_x'])
        self.assertIn('mdconf.py', self.job1.additional_files_to_upload[3]['remote'])
        self.assertIn('mdconf.py', self.job1.additional_files_to_upload[3]['local'])

        self.assertEqual(self.job1.additional_files_to_upload[4]['source'], 'path')
        self.assertEqual(self.job1.additional_files_to_upload[4]['name'], 'M00.tleap')
        self.assertFalse(self.job1.additional_files_to_upload[4]['make_x'])
        self.assertIn('M00.tleap', self.job1.additional_files_to_upload[4]['remote'])
        self.assertIn('M00.tleap', self.job1.additional_files_to_upload[4]['local'])

        self.assertEqual(self.job1.additional_files_to_upload[5]['source'], 'path')
        self.assertEqual(self.job1.additional_files_to_upload[5]['name'], 'mdp.mdp')
        self.assertFalse(self.job1.additional_files_to_upload[5]['make_x'])
        self.assertIn('mdp.mdp', self.job1.additional_files_to_upload[5]['remote'])
        self.assertIn('mdp.mdp', self.job1.additional_files_to_upload[5]['local'])

    def test_format_max_job_time(self):
        """Test that the maximum job time can be formatted properly, including days, minutes, and seconds"""
        test_job = Job.__new__(Job)
        test_job.max_job_time = 59.888
        self.assertEqual(test_job.format_max_job_time('days'), '2-11:53:16')
        self.assertEqual(test_job.format_max_job_time('hours'), '59:53:16')

    def test_determine_model_chemistry(self):
        """Test that model chemistry method (e.g., b3lyp), basis set (e.g. def2-svp), and auxiliary basis set
        (e.g., def2-svp/c) can be correctly determined from level of theory (e.g. b3lyp/def2-svp def2-svp/c)."""
        test_job = Job.__new__(Job)

        # raise when encounter illegal inputs
        with self.assertRaises(InputError):
            test_job.job_level_of_theory_dict = 'b3lyp/6-31g'
            test_job.determine_model_chemistry()

        # raise when method is empty
        with self.assertRaises(InputError):
            test_job.job_level_of_theory_dict = {'basis': 'def2-tzvp'}
            test_job.determine_model_chemistry()

        # with auxiliary basis and DFT dispersion
        test_job.job_level_of_theory_dict = {'method': 'b3lyp', 'basis': 'def2-svp', 'auxiliary_basis': 'def2-svp/c',
                                             'dispersion': 'gd3bj'}
        method_expected, basis_expected, auxillary_expected, dispersion_expected = \
            'b3lyp', 'def2-svp', 'def2-svp/c', 'gd3bj'
        test_job.method, test_job.basis_set, test_job.auxiliary_basis_set, test_job.dispersion = '', '', '', ''
        test_job.determine_model_chemistry()
        self.assertEqual(test_job.method, method_expected)
        self.assertEqual(test_job.basis_set, basis_expected)
        self.assertEqual(test_job.auxiliary_basis_set, auxillary_expected)
        self.assertEqual(test_job.dispersion, dispersion_expected)

        # basic method and basis case
        test_job.job_level_of_theory_dict = {'method': 'wb97x-d3', 'basis': 'def2-tzvp'}
        method_expected, basis_expected, auxillary_expected, dispersion_expected = 'wb97x-d3', 'def2-tzvp', '', ''
        test_job.method, test_job.basis_set, test_job.auxiliary_basis_set, test_job.dispersion = '', '', '', ''
        test_job.determine_model_chemistry()
        self.assertEqual(test_job.method, method_expected)
        self.assertEqual(test_job.basis_set, basis_expected)
        self.assertEqual(test_job.auxiliary_basis_set, auxillary_expected)
        self.assertEqual(test_job.dispersion, dispersion_expected)

        # composite method
        test_job.job_level_of_theory_dict = {'method': 'cbs-qb3'}
        method_expected, basis_expected, auxillary_expected, dispersion_expected = 'cbs-qb3', '', '', ''
        test_job.method, test_job.basis_set, test_job.auxiliary_basis_set, test_job.dispersion = '', '', '', ''
        test_job.determine_model_chemistry()
        self.assertEqual(test_job.method, method_expected)
        self.assertEqual(test_job.basis_set, basis_expected)
        self.assertEqual(test_job.auxiliary_basis_set, auxillary_expected)
        self.assertEqual(test_job.dispersion, dispersion_expected)

        # method with speical characters (e.g., parentheses, slashes, stars)
        test_job.job_level_of_theory_dict = {'method': 'dlpno-CCSD(T)-F12a', 'basis': 'ma-DKH-def2-TZVP(-f)',
                                             'auxiliary_basis': '6-311G**'}
        method_expected, basis_expected, auxillary_expected = 'dlpno-CCSD(T)-F12a', 'ma-DKH-def2-TZVP(-f)', '6-311G**'
        test_job.method, test_job.basis_set, test_job.auxiliary_basis_set = '', '', ''
        test_job.determine_model_chemistry()
        self.assertEqual(test_job.method, method_expected)
        self.assertEqual(test_job.basis_set, basis_expected)
        self.assertEqual(test_job.auxiliary_basis_set, auxillary_expected)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        projects = ['arc_project_for_testing_delete_after_usage3']
        for project in projects:
            project_directory = os.path.join(arc_path, 'Projects', project)
            if os.path.isdir(project_directory):
                shutil.rmtree(project_directory)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
