#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.orca module
"""

import math
import os
import shutil
import unittest

from arc.common import ARC_PATH
from arc.job.adapters.orca import OrcaAdapter
from arc.level import Level
from arc.settings.settings import input_filenames, output_filenames
from arc.species import ARCSpecies


class TestOrcaAdapter(unittest.TestCase):
    """
    Contains unit tests for the OrcaAdapter class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.job_1 = OrcaAdapter(execution_type='queue',
                                job_type='sp',
                                level=Level(method='DLPNO-CCSD(T)', basis='def2-tzvp', auxiliary_basis='def2-tzvp/c'),
                                project='test',
                                project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_OrcaAdapter'),
                                species=[ARCSpecies(label='CH3O',
                                                    xyz="""C       0.03807240    0.00035621   -0.00484242
                                                           O       1.35198769    0.01264937   -0.17195885
                                                           H      -0.33965241   -0.14992727    1.02079480
                                                           H      -0.51702680    0.90828035   -0.29592912
                                                           H      -0.53338088   -0.77135867   -0.54806440""")],
                                testing=True,
                                )

    def test_set_cpu_and_mem(self):
        """Test assigning number of cpu's and memory"""
        self.job_1.cpu_cores = 48
        self.job_1.input_file_memory = None
        self.job_1.submit_script_memory = None
        self.job_1.set_cpu_and_mem()
        self.assertEqual(self.job_1.cpu_cores, 48)

    def test_set_input_file_memory(self):
        """Test setting the input_file_memory argument"""
        expected_memory = math.ceil(14 * 1024 / 48)
        self.assertEqual(self.job_1.input_file_memory, expected_memory)

    def test_write_input_file(self):
        """Test writing Gaussian input files"""
        self.job_1.write_input_file()
        with open(os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]), 'r') as f:
            content_1 = f.read()
        job_1_expected_input_file = """!uHF dlpno-ccsd(t) def2-tzvp def2-tzvp/c tightscf normalpno 
! NRSCF # using Newton–Raphson SCF algorithm 
!sp 

%maxcore 299
%pal # job parallelization settings
nprocs 48
end
%scf # recommended SCF settings 
NRMaxIt 400
NRStart 0.00005
MaxIter 500
end


* xyz 0 2
C       0.03807240    0.00035621   -0.00484242
O       1.35198769    0.01264937   -0.17195885
H      -0.33965241   -0.14992727    1.02079480
H      -0.51702680    0.90828035   -0.29592912
H      -0.53338088   -0.77135867   -0.54806440
*
"""
        self.assertEqual(content_1, job_1_expected_input_file)

    def test_set_files(self):
        """Test setting files"""
        job_1_files_to_upload = [{'file_name': 'submit.sl',
                                  'local': os.path.join(self.job_1.local_path, 'submit.sl'),
                                  'remote': os.path.join(self.job_1.remote_path, 'submit.sl'),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'input.in',
                                  'local': os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]),
                                  'remote': os.path.join(self.job_1.remote_path, input_filenames[self.job_1.job_adapter]),
                                  'source': 'path',
                                  'make_x': False},
                                 ]
        job_1_files_to_download = [{'file_name': 'input.log',
                                    'local': os.path.join(self.job_1.local_path, output_filenames[self.job_1.job_adapter]),
                                    'remote': os.path.join(self.job_1.remote_path, output_filenames[self.job_1.job_adapter]),
                                    'source': 'path',
                                    'make_x': False}]
        self.assertEqual(self.job_1.files_to_upload, job_1_files_to_upload)
        self.assertEqual(self.job_1.files_to_download, job_1_files_to_download)

        # job_2_files_to_upload = [{'file_name': 'submit.sh',
        #                           'local': os.path.join(self.job_2.local_path, 'submit.sh'),
        #                           'remote': os.path.join(self.job_2.remote_path, 'submit.sh'),
        #                           'source': 'path',
        #                           'make_x': False},
        #                          {'file_name': 'data.hdf5',
        #                           'local': os.path.join(self.job_2.local_path, 'data.hdf5'),
        #                           'remote': os.path.join(self.job_2.remote_path, 'data.hdf5'),
        #                           'source': 'path',
        #                           'make_x': False}]
        # job_2_files_to_download = [{'file_name': 'data.hdf5',
        #                             'local': os.path.join(self.job_2.local_path, 'data.hdf5'),
        #                             'remote': os.path.join(self.job_2.remote_path, 'data.hdf5'),
        #                             'source': 'path',
        #                             'make_x': False}]
        # self.assertEqual(self.job_2.files_to_upload, job_2_files_to_upload)
        # self.assertEqual(self.job_2.files_to_download, job_2_files_to_download)


    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'test_OrcaAdapter'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
