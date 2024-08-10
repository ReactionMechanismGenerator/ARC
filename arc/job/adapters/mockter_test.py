#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.mockter module
"""

import os
import shutil
import unittest

from arc.common import ARC_PATH, read_yaml_file
from arc.job.adapters.mockter import MockAdapter
from arc.level import Level
from arc.settings.settings import input_filenames, output_filenames
from arc.species import ARCSpecies


class TestMockAdapter(unittest.TestCase):
    """
    Contains unit tests for the MockAdapter class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.job_1 = MockAdapter(execution_type='incore',
                                job_type='sp',
                                level=Level(method='CCMockSD(T)', basis='cc-pVmockZ'),
                                project='test',
                                project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_MockAdapter_1'),
                                species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                testing=True,
                                )
        cls.job_2 = MockAdapter(job_type='opt',
                                level=Level(method='CCMockSD(T)', basis='cc-pVmockZ'),
                                project='test',
                                project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_MockAdapter_2'),
                                species=[ARCSpecies(label='spc2', xyz=['O 0 0 1'])],
                                testing=True,
                                )
        cls.job_3 = MockAdapter(job_type='freq',
                                level=Level(method='CCMockSD(T)', basis='cc-pVmockZ'),
                                project='test',
                                project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_MockAdapter_3'),
                                species=[ARCSpecies(label='spc3', xyz=['O 0 0 1\nH 0 0 0\nH 1 0 0'], is_ts=True)],
                                testing=True,
                                )

    def test_set_cpu_and_mem(self):
        """Test assigning number of cpu's and memory"""
        self.job_1.cpu_cores = 48
        self.job_1.input_file_memory = None
        self.job_1.submit_script_memory = 14
        self.job_1.set_cpu_and_mem()
        self.assertEqual(self.job_1.cpu_cores, 48)

    def test_set_input_file_memory(self):
        """Test setting the input_file_memory argument"""
        self.job_1.input_file_memory = None
        self.job_1.cpu_cores = 48
        self.job_1.set_input_file_memory()
        self.assertEqual(self.job_1.input_file_memory, 14)

    def test_write_input_file(self):
        """Test writing Gaussian input files"""
        self.job_1.cpu_cores = 48
        self.job_1.set_input_file_memory()
        self.job_1.write_input_file()
        content_1 = read_yaml_file(os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]))
        job_1_expected_input_file = {'basis': 'cc-pvmockz',
                                     'charge': 0,
                                     'job_type': 'sp',
                                     'label': 'spc1',
                                     'memory': 14.0,
                                     'method': 'ccmocksd(t)',
                                     'multiplicity': 3,
                                     'xyz': 'O       0.00000000    0.00000000    1.00000000'}
        self.assertEqual(content_1, job_1_expected_input_file)

        self.job_2.cpu_cores = 48
        self.job_2.set_input_file_memory()
        self.job_2.write_input_file()
        content_2 = read_yaml_file(os.path.join(self.job_2.local_path, input_filenames[self.job_2.job_adapter]))
        job_2_expected_input_file = {'basis': 'cc-pvmockz',
                                     'charge': 0,
                                     'job_type': 'opt',
                                     'label': 'spc2',
                                     'memory': 14.0,
                                     'method': 'ccmocksd(t)',
                                     'multiplicity': 3,
                                     'xyz': 'O       0.00000000    0.00000000    1.00000000'}
        self.assertEqual(content_2, job_2_expected_input_file)

        self.job_3.cpu_cores = 48
        self.job_3.set_input_file_memory()
        self.job_3.write_input_file()
        content_3 = read_yaml_file(os.path.join(self.job_3.local_path, input_filenames[self.job_3.job_adapter]))
        job_3_expected_input_file = {'basis': 'cc-pvmockz',
                                     'charge': 0,
                                     'job_type': 'freq',
                                     'label': 'spc3',
                                     'memory': 14.0,
                                     'method': 'ccmocksd(t)',
                                     'multiplicity': 1,
                                     'xyz': 'O       0.00000000    0.00000000    1.00000000\n'
                                            'H       0.00000000    0.00000000    0.00000000\n'
                                            'H       1.00000000    0.00000000    0.00000000'}
        self.assertEqual(content_3, job_3_expected_input_file)

    def test_executing_mockter(self):
        """Test executing mockter"""
        self.job_1.execute()
        output = read_yaml_file(os.path.join(self.job_1.local_path, output_filenames[self.job_1.job_adapter]))
        self.assertEqual(output['sp'], 0.0)
        self.assertEqual(output['T1'], 0.0002)

        self.job_2.execute()
        output = read_yaml_file(os.path.join(self.job_2.local_path, output_filenames[self.job_2.job_adapter]))
        self.assertEqual(output['xyz'], 'O       0.00000000    0.00000000    1.00000000')

        self.job_3.execute()
        output = read_yaml_file(os.path.join(self.job_3.local_path, output_filenames[self.job_3.job_adapter]))
        self.assertEqual(output['freqs'], [-500, 520, 540])
        self.assertEqual(output['adapter'], 'mockter')

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        for folder in ['test_MockAdapter_1', 'test_MockAdapter_2', 'test_MockAdapter_3']:
            shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', folder), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
