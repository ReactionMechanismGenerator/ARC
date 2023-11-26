#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.molpro module
"""

import math
import os
import shutil
import unittest

from arc.common import ARC_PATH
from arc.job.adapters.molpro import MolproAdapter
from arc.level import Level
from arc.settings.settings import input_filenames, output_filenames
from arc.species import ARCSpecies


class TestMolproAdapter(unittest.TestCase):
    """
    Contains unit tests for the MolproAdapter class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.job_1 = MolproAdapter(execution_type='queue',
                                  job_type='sp',
                                  level=Level(method='CCSD(T)-F12', basis='cc-pVTZ-f12'),
                                  project='test',
                                  project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_MolproAdapter_1'),
                                  species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                  testing=True,
                                  )
        cls.job_2 = MolproAdapter(execution_type='queue',
                                  job_type='opt',
                                  level=Level(method='CCSD(T)', basis='cc-pVQZ'),
                                  project='test',
                                  project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_MolproAdapter_2'),
                                  species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                  testing=True,
                                  )
        cls.job_3 = MolproAdapter(execution_type='queue',
                                  job_type='opt',
                                  level=Level(method='CCSD(T)', basis='cc-pVQZ'),
                                  project='test',
                                  project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_MolproAdapter_2'),
                                  species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                  testing=True,
                                  ess_trsh_methods=['memory','cpu', 'molpro_memory: 2800 '],
                                  job_memory_gb=64,
                                  )
        cls.job_4 = MolproAdapter(execution_type='queue',
                                  job_type='opt',
                                  level=Level(method='CCSD(T)', basis='cc-pVQZ'),
                                  project='test',
                                  project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_MolproAdapter_2'),
                                  species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                  testing=True,
                                  ess_trsh_methods=['memory','cpu', 'molpro_memory: 4300 '],
                                  job_memory_gb=64,
                                  )

    def test_set_cpu_and_mem(self):
        """Test assigning number of cpu's and memory"""
        self.job_1.cpu_cores = 48
        self.job_1.input_file_memory = None
        self.job_1.submit_script_memory = 14
        self.job_1.set_cpu_and_mem()
        self.assertEqual(self.job_1.cpu_cores, 48)

    def test_memory_change(self):
        """Test changing the memory

        1. Test that the required memory is set correctly and that the number of CPUs changes accordingly
        2. Test that the required memory requires 1 CPU and will therefore not attempt to the total memory
        """
        self.assertEqual(self.job_3.input_file_memory, 2864)
        self.assertEqual(self.job_3.cpu_cores, 3)

        self.assertEqual(self.job_4.input_file_memory, 4300)
        self.assertEqual(self.job_4.cpu_cores, 1)


    def test_set_input_file_memory(self):
        """Test setting the input_file_memory argument"""
        self.job_1.input_file_memory = None
        self.job_1.cpu_cores = 48
        self.job_1.set_input_file_memory()
        self.assertEqual(self.job_1.input_file_memory, 40)

        self.job_1.cpu_cores = 8
        self.job_1.set_input_file_memory()
        self.assertEqual(self.job_1.input_file_memory, 235)

        self.job_1.input_file_memory = None
        self.job_1.cpu_cores = 1
        self.job_1.set_input_file_memory()
        self.assertEqual(self.job_1.input_file_memory, 1880)

    def test_write_input_file(self):
        """Test writing Gaussian input files"""
        self.job_1.cpu_cores = 48
        self.job_1.set_input_file_memory()
        self.job_1.write_input_file()
        with open(os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]), 'r') as f:
            content_1 = f.read()
        job_1_expected_input_file = """***,spc1
memory,40,m;

geometry={angstrom;
O       0.00000000    0.00000000    1.00000000}

basis=cc-pvtz-f12



int;
{hf;
maxit,1000;
wf,spin=2,charge=0;}

uccsd(t)-f12;



---;

"""
        self.assertEqual(content_1, job_1_expected_input_file)

        self.job_2.cpu_cores = 48
        self.job_2.set_input_file_memory()
        self.job_2.write_input_file()
        with open(os.path.join(self.job_2.local_path, input_filenames[self.job_2.job_adapter]), 'r') as f:
            content_2 = f.read()
        job_2_expected_input_file = """***,spc1
memory,40,m;

geometry={angstrom;
O       0.00000000    0.00000000    1.00000000}

basis=cc-pvqz



int;
{hf;
maxit,1000;
wf,spin=2,charge=0;}

uccsd(t);

optg, savexyz='geometry.xyz'

---;

"""
        self.assertEqual(content_2, job_2_expected_input_file)

    def test_set_files(self):
        """Test setting files"""
        job_1_files_to_upload = [{'file_name': 'submit.sub',
                                  'local': os.path.join(self.job_1.local_path, 'submit.sub'),
                                  'remote': os.path.join(self.job_1.remote_path, 'submit.sub'),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'input.in',
                                  'local': os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]),
                                  'remote': os.path.join(self.job_1.remote_path, input_filenames[self.job_1.job_adapter]),
                                  'source': 'path',
                                  'make_x': False},
                                 ]
        job_1_files_to_download = [{'file_name':'output.out',
                                    'local': os.path.join(self.job_1.local_path, output_filenames[self.job_1.job_adapter]),
                                    'remote': os.path.join(self.job_1.remote_path, output_filenames[self.job_1.job_adapter]),
                                    'source': 'path',
                                    'make_x': False}]
        self.assertEqual(self.job_1.files_to_upload, job_1_files_to_upload)
        self.assertEqual(self.job_1.files_to_download, job_1_files_to_download)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        for folder in ['test_MolproAdapter_1', 'test_MolproAdapter_2']:
            shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', folder), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
