#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapter module
"""

import datetime
import math
import os
import time
import shutil
import unittest

import pandas as pd

from arc.common import ARC_PATH
from arc.imports import settings
from arc.job.adapter import DataPoint, JobEnum, JobTypeEnum, JobExecutionTypeEnum
from arc.job.adapters.gaussian import GaussianAdapter
from arc.level import Level
from arc.species import ARCSpecies


servers, submit_filenames = settings['servers'], settings['submit_filenames']


class TestEnumerationClasses(unittest.TestCase):
    """
    Contains unit tests for various enumeration classes.
    """

    def test_job_enum(self):
        """Test the JobEnum class"""
        self.assertEqual(JobEnum('gaussian').value, 'gaussian')
        self.assertEqual(JobEnum('molpro').value, 'molpro')
        self.assertEqual(JobEnum('orca').value, 'orca')
        self.assertEqual(JobEnum('psi4').value, 'psi4')
        self.assertEqual(JobEnum('qchem').value, 'qchem')
        self.assertEqual(JobEnum('terachem').value, 'terachem')
        self.assertEqual(JobEnum('autotst').value, 'autotst')
        self.assertEqual(JobEnum('heuristics').value, 'heuristics')
        self.assertEqual(JobEnum('kinbot').value, 'kinbot')
        self.assertEqual(JobEnum('gcn').value, 'gcn')
        self.assertEqual(JobEnum('user').value, 'user')
        with self.assertRaises(ValueError):
            JobEnum('wrong')

    def test_job_type_enum(self):
        """Test the JobTypeEnum class"""
        self.assertEqual(JobTypeEnum('composite').value, 'composite')
        self.assertEqual(JobTypeEnum('conformers').value, 'conformers')
        self.assertEqual(JobTypeEnum('freq').value, 'freq')
        self.assertEqual(JobTypeEnum('gen_confs').value, 'gen_confs')
        self.assertEqual(JobTypeEnum('irc').value, 'irc')
        self.assertEqual(JobTypeEnum('onedmin').value, 'onedmin')
        self.assertEqual(JobTypeEnum('opt').value, 'opt')
        self.assertEqual(JobTypeEnum('optfreq').value, 'optfreq')
        self.assertEqual(JobTypeEnum('orbitals').value, 'orbitals')
        self.assertEqual(JobTypeEnum('scan').value, 'scan')
        self.assertEqual(JobTypeEnum('sp').value, 'sp')
        self.assertEqual(JobTypeEnum('tsg').value, 'tsg')
        with self.assertRaises(ValueError):
            JobTypeEnum('wrong')

    def test_job_execution_type_enum(self):
        """Test the JobExecutionTypeEnum class"""
        self.assertEqual(JobExecutionTypeEnum('incore').value, 'incore')
        self.assertEqual(JobExecutionTypeEnum('queue').value, 'queue')
        self.assertEqual(JobExecutionTypeEnum('pipe').value, 'pipe')
        with self.assertRaises(ValueError):
            JobExecutionTypeEnum('wrong')


class TestDataPoint(unittest.TestCase):
    """
    Contains unit tests for the DataPoint class.
    """

    def test_as_dict(self):
        """Test the dictionary representation of a DataPoint instance"""
        xyz_1 = {'symbols': ('C', 'H', 'H', 'H', 'H'),
                 'isotopes': (12, 1, 1, 1, 1),
                 'coords': ((0.0, 0.0, 0.0),
                            (0.6300326, 0.6300326, 0.6300326),
                            (-0.6300326, -0.6300326, 0.6300326),
                            (-0.6300326, 0.6300326, -0.6300326),
                            (0.6300326, -0.6300326, -0.6300326))}
        data_point = DataPoint(charge=0,
                               job_types=['opt'],
                               label='spc1',
                               level={'method': 'cbs-qb3'},
                               multiplicity=1,
                               xyz_1=xyz_1,
                               )
        expected_dict = {'job_types': ['opt'],
                         'label': 'spc1',
                         'level': {'method': 'cbs-qb3'},
                         'xyz_1': xyz_1,
                         'status': 0,
                         'electronic_energy': None,
                         'error': None,
                         'frequencies': None,
                         'xyz_out': None}
        self.assertEqual(data_point.as_dict(), expected_dict)


class TestJobAdapter(unittest.TestCase):
    """
    Contains unit tests for the JobAdapter class.

    Here we use the GaussianAdapter class, but only test methods defined under the parent JobAdapter abstract class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.job_1 = GaussianAdapter(execution_type='incore',
                                    job_type='conformers',
                                    level=Level(method='cbs-qb3'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_JobAdapter'),
                                    species=[ARCSpecies(label='spc1',
                                                        xyz=['O 0 0 1',
                                                             'O 0 0 2',
                                                             'O 0 0 3',
                                                             'O 0 0 4',
                                                             'O 0 0 5',
                                                             'O 0 0 6']),
                                             ARCSpecies(label='spc2',
                                                        xyz=['N 0 0 1',
                                                             'N 0 0 2',
                                                             'N 0 0 3',
                                                             'N 0 0 4',
                                                             'N 0 0 5',
                                                             'N 0 0 6']),
                                             ARCSpecies(label='spc3',
                                                        xyz=['S 0 0 1',
                                                             'S 0 0 2',
                                                             'S 0 0 3',
                                                             'S 0 0 4',
                                                             'S 0 0 5',
                                                             'S 0 0 6']),
                                             ],
                                    testing=True,
                                    )
        cls.job_2 = GaussianAdapter(execution_type='incore',
                                    job_type='opt',
                                    level=Level(method='cbs-qb3'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_JobAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                    testing=True,
                                    )
        cls.spc_3a = ARCSpecies(label='methanol',
                                smiles='CO',
                                directed_rotors={'brute_force_sp': ['all']})
        cls.spc_3b = ARCSpecies(label='methanol',
                                smiles='CO',
                                directed_rotors={'brute_force_opt': ['all']})
        cls.spc_3c = ARCSpecies(label='methanol',
                                smiles='CO',
                                directed_rotors={'cont_opt': ['all']})
        cls.spc_3d = ARCSpecies(label='methanol',
                                smiles='CO',
                                directed_rotors={'brute_force_sp_diagonal': ['all']})
        cls.spc_3e = ARCSpecies(label='methanol',
                                smiles='CO',
                                directed_rotors={'brute_force_opt_diagonal': ['all']})
        cls.spc_3f = ARCSpecies(label='methanol',
                                smiles='CO',
                                directed_rotors={'cont_opt_diagonal': ['all']})
        cls.job_3 = GaussianAdapter(execution_type='incore',
                                    job_type='scan',
                                    torsions=[[1, 2, 3, 4]],
                                    level=Level(method='wb97xd', basis='def2-tzvp'),
                                    project='test_scans',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_JobAdapter_scan'),
                                    species=[cls.spc_3a, cls.spc_3b, cls.spc_3c, cls.spc_3d, cls.spc_3e, cls.spc_3f],
                                    testing=True,
                                    )

    def test_determine_job_array_parameters(self):
        """Test determining job array parameters"""
        self.assertEqual(self.job_1.iterate_by, ['species', 'conformers'])
        self.assertEqual(self.job_1.number_of_processes, 3 * 6)
        self.assertEqual(self.job_1.workers, 4)

    def test_determine_workers(self):
        """Test determining the number of workers"""
        self.job_2.number_of_processes, self.job_2.workers = 1, None
        self.job_2._determine_workers()
        self.assertEqual(self.job_2.workers, 1)

        self.job_2.number_of_processes, self.job_2.workers = 2, None
        self.job_2._determine_workers()
        self.assertEqual(self.job_2.workers, 1)

        self.job_2.number_of_processes, self.job_2.workers = 3, None
        self.job_2._determine_workers()
        self.assertEqual(self.job_2.workers, 1)

        self.job_2.number_of_processes, self.job_2.workers = 4, None
        self.job_2._determine_workers()
        self.assertEqual(self.job_2.workers, 2)

        self.job_2.number_of_processes, self.job_2.workers = 5, None
        self.job_2._determine_workers()
        self.assertEqual(self.job_2.workers, 2)

        self.job_2.number_of_processes, self.job_2.workers = 9, None
        self.job_2._determine_workers()
        self.assertEqual(self.job_2.workers, 2)

        self.job_2.number_of_processes, self.job_2.workers = 10, None
        self.job_2._determine_workers()
        self.assertEqual(self.job_2.workers, 4)

        self.job_2.number_of_processes, self.job_2.workers = 100, None
        self.job_2._determine_workers()
        self.assertEqual(self.job_2.workers, 6)

        self.job_2.number_of_processes, self.job_2.workers = 1000, None
        self.job_2._determine_workers()
        self.assertEqual(self.job_2.workers, 11)

        self.job_2.number_of_processes, self.job_2.workers = 1e4, None
        self.job_2._determine_workers()
        self.assertEqual(self.job_2.workers, 20)

        self.job_2.number_of_processes, self.job_2.workers = 1e5, None
        self.job_2._determine_workers()
        self.assertEqual(self.job_2.workers, 36)

        self.job_2.number_of_processes, self.job_2.workers = 1e6, None
        self.job_2._determine_workers()
        self.assertEqual(self.job_2.workers, 63)

        self.job_2.number_of_processes, self.job_2.workers = 1e7, None
        self.job_2._determine_workers()
        self.assertEqual(self.job_2.workers, 100)

        self.job_2.number_of_processes, self.job_2.workers = 1e8, None
        self.job_2._determine_workers()
        self.assertEqual(self.job_2.workers, 100)

    def test_write_hdf5(self):
        """Test writing the HDF5 file"""
        with pd.HDFStore(os.path.join(self.job_1.local_path, 'data.hdf5')) as store:
            data = store['df'].to_dict()
        self.assertEqual([key for key in data.keys()], ['spc1', 'spc2', 'spc3'])

    def test_write_hdf5_for_directed_scans(self):
        """Test writing the HDF5 file for directed scans"""
        with pd.HDFStore(os.path.join(self.job_1.local_path, 'data.hdf5')) as store:
            data = store['df'].to_dict()
        self.assertEqual([key for key in data.keys()], ['spc1', 'spc2', 'spc3'])

    def test_write_array_submit_script(self):
        """Test writing an array submit script"""
        self.job_1.write_submit_script()
        with open(os.path.join(self.job_1.local_path, submit_filenames[servers[self.job_1.server]['cluster_soft']]),
                  'r') as f:
            lines = f.readlines()
        array, hdf5 = False, False
        for line in lines:
            if '#SBATCH --array=1-4' in line:
                array = True
            if 'job/scripts/pipe.py' in line and 'data.hdf5' in line:
                hdf5 = True
        self.assertTrue(array)
        self.assertTrue(hdf5)

    def test_write_queue_submit_script(self):
        """Test writing a queue submit script"""
        self.job_2.number_of_processes, self.job_2.workers = 1, None
        self.job_2.write_submit_script()
        with open(os.path.join(self.job_2.local_path, submit_filenames[servers[self.job_2.server]['cluster_soft']]),
                  'r') as f:
            lines = f.readlines()
        array, hdf5, g16 = False, False, False
        for line in lines:
            if '#SBATCH --array=1-5' in line:
                array = True
            if 'job/scripts/pipe.py' in line and 'data.hdf5' in line:
                hdf5 = True
            if 'g16 < input.gjf > input.log' in line:
                g16 = True
        self.assertFalse(array)
        self.assertFalse(hdf5)
        self.assertTrue(g16)

    def test_determine_run_time(self):
        """Test determining the job run time"""
        self.job_2.initial_time = datetime.datetime.now()
        time.sleep(1)
        self.job_2.final_time = datetime.datetime.now()
        self.job_2.determine_run_time()
        self.assertEqual(self.job_2.run_time, datetime.timedelta(seconds=1))

    def test_set_cpu_and_mem(self):
        """Test assigning number of cpu's and memory"""
        # HTCondor
        self.job_1.cpu_cores = None
        self.job_1.set_cpu_and_mem()
        self.assertEqual(self.job_1.cpu_cores, 8)
        expected_memory = math.ceil((14 * 1024 * 1.1))
        self.assertEqual(self.job_1.submit_script_memory, expected_memory)

        # Slurm
        self.job_1.server = 'server2'
        self.job_1.cpu_cores = None
        self.job_1.set_cpu_and_mem()
        self.assertEqual(self.job_1.cpu_cores, 8)
        expected_memory = math.ceil((14 * 1024 * 1.1) / self.job_1.cpu_cores)
        self.assertEqual(self.job_1.submit_script_memory, expected_memory)
        self.job_1.server = 'local'

    def test_set_file_paths(self):
        """Test setting up the job's paths"""
        self.assertEqual(self.job_2.local_path, os.path.join(self.job_2.project_directory, 'calcs', 'Species',
                                                             self.job_2.species_label, self.job_2.job_name))
        self.assertEqual(self.job_2.remote_path, os.path.join('runs', 'ARC_Projects', self.job_2.project,
                                                              self.job_2.species_label, self.job_2.job_name))

    def test_format_max_job_time(self):
        """Test that the maximum job time can be formatted properly, including days, minutes, and seconds"""
        test_job = GaussianAdapter.__new__(GaussianAdapter)
        test_job.max_job_time = 59.888
        self.assertEqual(test_job.format_max_job_time('days'), '2-11:53:16')
        self.assertEqual(test_job.format_max_job_time('hours'), '59:53:16')

    def test_add_to_args(self):
        """Test adding parameters to self.args"""
        self.assertEqual(self.job_1.args, {'block': {}, 'keyword': {}, 'trsh': {}})
        self.job_1.add_to_args(val='val_tst_1')
        self.job_1.add_to_args(val='val_tst_2')
        self.job_1.add_to_args(val='val_tst_3', separator='     ')
        self.job_1.add_to_args(val="""val_tst_4\nval_tst_5""", key1='block', key2='specific_key_2')
        expected_args = {'keyword': {'general': 'val_tst_1 val_tst_2     val_tst_3'},
                         'block': {'specific_key_2': 'val_tst_4\nval_tst_5'},
                         'trsh': {}}
        self.assertEqual(self.job_1.args, expected_args)

    def test_get_file_property_dictionary(self):
        """Test getting the file property dictionary"""
        file_dict_1 = self.job_1.get_file_property_dictionary(file_name='file.name')
        self.assertEqual(file_dict_1, {'file_name': 'file.name',
                                       'local': os.path.join(self.job_1.local_path, 'file.name'),
                                       'remote': os.path.join(self.job_1.remote_path, 'file.name'),
                                       'source': 'path',
                                       'make_x': False})
        file_dict_2 = self.job_1.get_file_property_dictionary(file_name='m.x',
                                                              local='onedmin.molpro.x',
                                                              source='input_files',
                                                              make_x=True)
        self.assertEqual(file_dict_2, {'file_name': 'm.x',
                                       'local': 'onedmin.molpro.x',
                                       'remote': os.path.join(self.job_1.remote_path, 'm.x'),
                                       'source': 'input_files',
                                       'make_x': True})

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'test_JobAdapter'), ignore_errors=True)
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'test_JobAdapter_scan'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
