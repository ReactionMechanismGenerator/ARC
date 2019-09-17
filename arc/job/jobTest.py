#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests of the arc.job.job module
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest
import os
import datetime
import shutil

from arc.job.job import Job
from arc.settings import arc_path

################################################################################


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
                            'qchem': ['server1'], 'onedmin': ['server1']}
        cls.job1 = Job(project='arc_project_for_testing_delete_after_usage3', ess_settings=cls.ess_settings,
                       species_name='tst_spc', xyz='C 0.0 0.0 0.0', job_type='opt', level_of_theory='b3lyp/6-31+g(d)',
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
                         'ess_settings': {'gaussian': ['server1', 'server2'],
                                          'molpro': [u'server2'], 'onedmin': [u'server1'], 'qchem': [u'server1']},
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
                         'level_of_theory': 'b3lyp/6-31+g(d)',
                         'memory': 14,
                         'multiplicity': 1,
                         'project': 'arc_project_for_testing_delete_after_usage3',
                         'project_directory': os.path.join(arc_path, 'Projects', 'project_test'),
                         'server': 'server1',
                         'max_job_time': 120,
                         'scan_res': 8.0,
                         'software': 'gaussian',
                         'xyz': 'C 0.0 0.0 0.0'}
        self.assertEqual(job_dict, expected_dict)

    def test_from_dict(self):
        """Test Job.from_dict()"""
        job_dict = self.job1.as_dict()
        job = Job(job_dict=job_dict)
        self.assertEqual(job.multiplicity, 1)
        self.assertEqual(job.charge, 0)
        self.assertEqual(job.species_name, 'tst_spc')
        self.assertEqual(job.server, 'server1')
        self.assertEqual(job.level_of_theory, 'm062x/6-311g')
        self.assertEqual(job.job_type, 'scan')
        self.assertEqual(job.project_directory.split('/')[-1], 'project_test')
        self.assertEqual(job.method, 'm062x')
        self.assertEqual(job.basis_set, '6-311g')
        self.assertFalse(job.is_ts)

    def test_automatic_ess_assignment(self):
        """Test that the Job module correctly assigns a software for specific methods and basis sets"""
        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz='C 0.0 0.0 0.0',
                   job_type='opt', level_of_theory='b3lyp/6-311++G(d,p)', multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'gaussian')

        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz='C 0.0 0.0 0.0',
                   job_type='opt', level_of_theory='ccsd(t)/avtz', multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'molpro')

        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz='C 0.0 0.0 0.0',
                   job_type='opt', level_of_theory='wb97xd/6-311++g(d,p)', multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'gaussian')

        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz='C 0.0 0.0 0.0',
                   job_type='opt', level_of_theory='wb97x-d3/6-311++g(d,p)', multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'qchem')

        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz='C 0.0 0.0 0.0',
                   job_type='opt', level_of_theory='b97/6-311++g(d,p)', multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'gaussian')

        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz='C 0.0 0.0 0.0',
                   job_type='opt', level_of_theory='m062x/6-311++g(d,p)', multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'gaussian')

        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz='C 0.0 0.0 0.0',
                   job_type='opt', level_of_theory='m06-2x/6-311++g(d,p)', multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'qchem')

        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz='C 0.0 0.0 0.0',
                   job_type='scan', level_of_theory='m062x/6-311++g(d,p)', multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'gaussian')

        job0 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc', xyz='C 0.0 0.0 0.0',
                   job_type='scan', level_of_theory='m06-2x/6-311++g(d,p)', multiplicity=1, testing=True,
                   project_directory=os.path.join(arc_path, 'Projects', 'project_test'), fine=True, job_num=100)
        self.assertEqual(job0.software, 'qchem')

        self.assertEqual(job0.memory_gb, 14)
        self.assertEqual(job0.max_job_time, 120)

    def test_bath_gas(self):
        """Test correctly assigning the bath_gas attribute"""
        self.assertIsNone(self.job1.bath_gas)

        job2 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc',
                   xyz='C 0.0 0.0 0.0', job_type='onedmin', level_of_theory='b3lyp/6-31+g(d)', multiplicity=1,
                   testing=True, project_directory=os.path.join(arc_path, 'Projects', 'project_test'),
                   fine=True, job_num=100)
        self.assertEqual(job2.bath_gas, 'N2')

        job2 = Job(project='project_test', ess_settings=self.ess_settings, species_name='tst_spc',
                   xyz='C 0.0 0.0 0.0', job_type='onedmin', level_of_theory='b3lyp/6-31+g(d)', multiplicity=1,
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

        self.job1.job_type = 'opt'
        self.job1.level_of_theory = 'm06-2x/6-311g'  # test the levels_ess dict from settings
        self.job1.software = None
        self.job1.deduce_software()
        self.assertEqual(self.job1.software, 'qchem')

        self.job1.job_type = 'opt'
        self.job1.level_of_theory = 'ccsd(t)/cc-pvtz'
        self.job1.method = 'ccsd(t)'
        self.job1.software = None
        self.job1.deduce_software()
        self.assertEqual(self.job1.software, 'molpro')

        self.job1.job_type = 'opt'
        self.job1.level_of_theory = 'wb97xd/6-311g'
        self.job1.method = 'wb97xd'
        self.job1.software = None
        self.job1.deduce_software()
        self.assertEqual(self.job1.software, 'gaussian')

        self.job1.job_type = 'scan'
        self.job1.level_of_theory = 'm062x/6-311g'
        self.job1.method = 'm062x'
        self.job1.software = None
        self.job1.deduce_software()
        self.assertEqual(self.job1.software, 'gaussian')

    def test_set_cpu_and_mem(self):
        """Test assigning number of cpu's and memory"""
        self.job1.cpus = None
        self.job1.memory = None
        self.job1.mem_per_cpu = None
        self.job1.server = 'server2'
        self.job1.software = 'molpro'
        self.job1.set_cpu_and_mem(memory=14)
        self.assertEqual(self.job1.cpus, 48)
        self.assertEqual(self.job1.memory, 14 * 128 / self.job1.cpus)
        self.job1.server = 'server1'
        self.job1.set_cpu_and_mem(memory=14)
        self.assertEqual(self.job1.cpus, 8)
        self.assertEqual(self.job1.memory, 14 * 128 / self.job1.cpus)

        self.job1.cpus = None
        self.job1.memory = None
        self.job1.mem_per_cpu = None
        self.job1.server = 'server2'
        self.job1.software = 'terachem'
        self.job1.set_cpu_and_mem(memory=14)
        self.assertEqual(self.job1.cpus, 48)
        self.assertEqual(self.job1.memory, 14 * 128 / self.job1.cpus)

        self.job1.cpus = None
        self.job1.memory = None
        self.job1.mem_per_cpu = None
        self.job1.server = 'server2'
        self.job1.software = 'gaussian'
        self.job1.set_cpu_and_mem(memory=14)
        self.assertEqual(self.job1.cpus, 48)
        self.assertEqual(self.job1.memory, 14 * 1000)

        self.job1.cpus = None
        self.job1.memory = None
        self.job1.mem_per_cpu = None
        self.job1.server = 'server2'
        self.job1.software = 'orca'
        self.job1.set_cpu_and_mem(memory=14)
        self.assertEqual(self.job1.cpus, 48)
        self.assertEqual(self.job1.memory, 14 * 1000 / self.job1.cpus)

        self.job1.cpus = None
        self.job1.memory = None
        self.job1.mem_per_cpu = None
        self.job1.server = 'server2'
        self.job1.software = 'qchem'
        self.job1.set_cpu_and_mem(memory=14)
        self.assertEqual(self.job1.cpus, 48)
        self.assertEqual(self.job1.memory, 14)

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


################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
