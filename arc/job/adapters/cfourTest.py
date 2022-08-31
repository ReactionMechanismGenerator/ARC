#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.cfour module
"""

import math
import os
import shutil
import unittest

from arc.common import ARC_PATH
from arc.job.adapters.cfour import CFourAdapter
from arc.level import Level
from arc.settings.settings import input_filenames, output_filenames
from arc.species import ARCSpecies


class TestCFourAdapter(unittest.TestCase):
    """
    Contains unit tests for the CFourAdapter class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        xyz = {'symbols': ('O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
               'isotopes': (16, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
               'coords': ((2.094965350070438, -0.6820312883655302, 0.41738812543556636),
                          (-0.17540789, 0.11818414, 0.5118097600000001),
                          (0.9251116997351838, -0.4681034307777509, -0.36086827472356287),
                          (-1.45486974, 0.34772573, -0.27221056),
                          (-0.37104416020111164, -0.5529098320998265, 1.3566465677436679),
                          (0.1653809158507713, 1.063511209038263, 0.9506975804596753),
                          (0.6185466335892884, -1.4314060698728557, -0.7802315547182584),
                          (1.176986440026325, 0.20190992061504034, -1.1892406328211544),
                          (-2.22790196, 0.76917222, 0.37791757),
                          (-1.8347639, -0.59150616, -0.68674647),
                          (-1.28838516, 1.04591267, -1.0987324),
                          (2.371381703321999, 0.17954058897321318, 0.7735703496393789))}
        cls.job_1 = CFourAdapter(execution_type='queue',
                                 job_type='sp',
                                 level=Level(method='CCSD(T)', basis='cc-pVTZ'),
                                 project='test',
                                 project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_CFourAdapter'),
                                 species=[ARCSpecies(label='spc1', xyz=xyz)],
                                 testing=True,
                                 )
        cls.job_1.cpu_cores = 48
        cls.job_1.submit_script_memory = None

    def test_set_cpu_and_mem(self):
        """Test assigning number of cpu's and memory"""
        self.job_1.set_cpu_and_mem()
        self.assertEqual(self.job_1.cpu_cores, 48)

    def test_set_input_file_memory(self):
        """Test setting the input_file_memory argument"""
        self.job_1.set_cpu_and_mem()
        expected_memory = math.ceil(14 * 128 * 1e6 / 48)
        self.assertEqual(self.job_1.input_file_memory, expected_memory)

    def test_write_input_file(self):
        """Test writing Gaussian input files"""
        self.job_1.write_input_file()
        with open(os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]), 'r') as f:
            content_1 = f.read()
        job_1_expected_input_file = """spc1
  O
  C       1      R1
  C       2      R2       1      A1
  C       3      R3       2      A2       1      D1
  H       3      R4       4      A3       2      D2
  H       3      R4       4      A4       5      D3
  H       2      R5       3      A5       4      D4
  H       2      R6       3      A6       4      D5
  H       4      R5       3      A7       8      D6
  H       4      R7       3      A8       9      D7
  H       4      R7       3      A9      10      D8
  H       1      R8       2     A10       3      D9

A10=107.5288
A1=109.8271
A2=111.8117
A3=109.8824
A4=109.7735
A5=110.8681
A6=111.3741
A7=110.2883
A8=110.9984
A9=110.9906
D1=180.7705
D2=238.0985
D3=243.8696
D4=299.9619
D5=60.5677
D6=204.2811
D7=240.2540
D8=239.4949
D9=299.9574
R1=1.4213
R2=1.5220
R3=1.5180
R4=1.0965
R5=1.0945
R6=1.0948
R7=1.0947
R8=0.9724

*CFOUR(CALC=ccsd(t),BASIS=cc-pvtz,CHARGE=0,MULTIPLICITY=1,MEMORY_SIZE=37333334
)


"""
        self.assertEqual(content_1, job_1_expected_input_file)

    def test_set_files(self):
        """Test setting files"""
        job_1_files_to_upload = [{'file_name': 'submit.sub',
                                  'local': os.path.join(self.job_1.local_path, 'submit.sub'),
                                  'remote': os.path.join(self.job_1.remote_path, 'submit.sub'),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'ZMAT',
                                  'local': os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]),
                                  'remote': os.path.join(self.job_1.remote_path, input_filenames[self.job_1.job_adapter]),
                                  'source': 'path',
                                  'make_x': False},
                                 ]
        job_1_files_to_download = [{'file_name': 'output.out',
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
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'test_CFourAdapter'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
