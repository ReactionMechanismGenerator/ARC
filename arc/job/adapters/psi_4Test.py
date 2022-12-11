#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.psi4 module
"""

import math
import os
import shutil
import unittest

from arc.common import ARC_PATH
from arc.imports import settings
from arc.job.adapters.psi_4 import Psi4Adapter
from arc.level import Level
from arc.species import ARCSpecies
from arc.species.converter import str_to_xyz

input_filenames, output_filenames = settings['input_filenames'], settings['output_filenames']

class TestPSi4Adapter(unittest.TestCase):
    """
    Contains unit tests for the Psi_4Adapter class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.job_1 = Psi4Adapter(execution_type='queue',
                                    job_type='sp',
                                    level=Level(method='scf',
                                                basis='cc-pVDZ'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_Psi4Adapter'),
                                    species=[ARCSpecies(label='h2o', xyz=["""O          0.00000        0.00000        0.11779
                                                                             H          0.00000        0.75545       -0.47116
                                                                             H          0.00000       -0.75545       -0.47116"""])],
                                    testing=True)
        cls.job_2 = Psi4Adapter(execution_type='queue',
                                    job_type='freq',
                                    level=Level(method='b3lyp',
                                                basis='cc-pVDZ'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_Psi4Adapter'),
                                    species=[ARCSpecies(label='CCOC', xyz=["""C        1.06326         0.04025        -0.06563
                                                                              C        2.57853         0.07698        -0.05455
                                                                              O        3.06406        -1.21182        -0.42225
                                                                              C        4.48468        -1.25313        -0.43275
                                                                              H        0.64727         1.01281         0.21185
                                                                              H        0.69073        -0.71602         0.63322
                                                                              H        0.69139        -0.23302        -1.05868
                                                                              H        2.93736         0.33544         0.94781
                                                                              H        2.93809         0.82553        -0.76916
                                                                              H        4.79790        -2.26028        -0.72049
                                                                              H        4.88165        -1.03476         0.56332
                                                                              H        4.88329        -0.54132        -1.16204"""])],
                                    testing= True)
        cls.job_3 = Psi4Adapter(execution_type='queue',
                                    job_type='opt',
                                    level=Level(method='wb97x-d',
                                                basis='def2-TZVP'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_Psi4Adapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                    testing=True,
                                    )
        cls.job_5 = Psi4Adapter(execution_type='queue',
                                    job_type='sp',
                                    xyz = {'symbols': ('O', 'H', 'H'), 'isotopes': (16, 1, 1),
                                    'coords': ((0.0, 0.0, 0.11779), (0.0, 0.75545, -0.47116),
                                    (0.0, -0.75545, -0.47116))},
                                    level=Level(method='scf',
                                                basis='cc-pVDZ'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_Psi4Adapter'),
                                    species=[ARCSpecies(label='h2o', smiles = "O")],
                                    testing=True)

    def test_set_cpu_and_mem(self):
        """Test assigning number of cpu's and memory"""
        self.job_1.input_file_memory = None
        self.job_1.submit_script_memory = None
        self.job_1.server = 'server2'
        self.job_1.set_cpu_and_mem()
        self.assertEqual(self.job_1.cpu_cores, 8)

    def test_set_input_file_memory(self):
        """Test setting the input_file_memory argument"""
        expected_memory = math.ceil(14 * 1024)
        self.assertEqual(self.job_1.input_file_memory, expected_memory)
        self.assertEqual(self.job_2.input_file_memory, 14336)

    def test_write_input_file(self):
        """Test writing psi_4 input files"""
        self.job_1.write_input_file()
        with open(os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]), 'r') as f:
            content_1 = f.read()
        job_1_expected_input_file = """
memory 14.0 GB
molecule h2o {
0 1
O       0.00000000    0.00000000    0.11779000
H       0.00000000    0.75545000   -0.47116000
H       0.00000000   -0.75545000   -0.47116000
}
set {
scf_type df
dft_spherical_points 302
dft_radial_points 99
dft_basis_tolerance 1e-11
}


set basis cc-pvdz
set reference uhf
energy(name = 'scf', return_wfn='on')

"""
        self.assertEqual(content_1, job_1_expected_input_file)
        self.job_2.write_input_file()
        with open(os.path.join(self.job_2.local_path, input_filenames[self.job_2.job_adapter]), 'r') as f:
            content_2 = f.read()
        job_2_expected_input_file = """
memory 14.0 GB
molecule CCOC {
0 1
C       1.06326000    0.04025000   -0.06563000
C       2.57853000    0.07698000   -0.05455000
O       3.06406000   -1.21182000   -0.42225000
C       4.48468000   -1.25313000   -0.43275000
H       0.64727000    1.01281000    0.21185000
H       0.69073000   -0.71602000    0.63322000
H       0.69139000   -0.23302000   -1.05868000
H       2.93736000    0.33544000    0.94781000
H       2.93809000    0.82553000   -0.76916000
H       4.79790000   -2.26028000   -0.72049000
H       4.88165000   -1.03476000    0.56332000
H       4.88329000   -0.54132000   -1.16204000
}
set {
scf_type df
dft_spherical_points 302
dft_radial_points 99
dft_basis_tolerance 1e-11
}


set basis cc-pvdz
set reference uhf
frequency(name = 'b3lyp', return_wfn='on')

"""
        self.assertEqual(content_2, job_2_expected_input_file)
        self.job_3.write_input_file()
        with open(os.path.join(self.job_3.local_path, input_filenames[self.job_3.job_adapter]), 'r') as f:
            content_3 = f.read()
        job_3_expected_input_file = """
memory 14.0 GB
molecule spc1 {
0 3
O       0.00000000    0.00000000    1.00000000
}
set {
scf_type df
dft_spherical_points 302
dft_radial_points 99
dft_basis_tolerance 1e-11
}


set basis def2-tzvp
set reference uhf
optimize(name = 'wb97x-d', return_wfn='on', return_history='on', engine='optking', dertype='energy')

"""
        self.assertEqual(content_3, job_3_expected_input_file)
        
    def test_set_files(self):
        """Test setting files"""
        job_1_files_to_upload = [{'file_name': 'submit.sub',
                                  'local': os.path.join(self.job_1.local_path, 'submit.sub'),
                                  'remote': os.path.join(self.job_1.remote_path, 'submit.sub'),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'input.dat',
                                  'local': os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]),
                                  'remote': os.path.join(self.job_1.remote_path, input_filenames[self.job_1.job_adapter]),
                                  'source': 'path',
                                  'make_x': False},]
        job_1_files_to_download = [{'file_name': 'output.dat',
                                    'local': os.path.join(self.job_1.local_path, output_filenames[self.job_1.job_adapter]),
                                    'remote': os.path.join(self.job_1.remote_path, output_filenames[self.job_1.job_adapter]),
                                    'source': 'path',
                                    'make_x': False},
                                   {'file_name': 'check.chk',
                                    'local': os.path.join(self.job_1.local_path, 'check.chk'),
                                    'remote': os.path.join(self.job_1.remote_path, 'check.chk'),
                                    'source': 'path',
                                    'make_x': False}]
        self.assertEqual(self.job_1.files_to_upload, job_1_files_to_upload)
        self.assertEqual(self.job_1.files_to_download, job_1_files_to_download)

        job_2_files_to_upload = [{'file_name': 'submit.sub',
                                  'local': os.path.join(self.job_2.local_path, 'submit.sub'),
                                  'remote': os.path.join(self.job_2.remote_path, 'submit.sub'),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'input.dat',
                                  'local': os.path.join(self.job_2.local_path, 'input.dat'),
                                  'remote': os.path.join(self.job_2.remote_path, 'input.dat'),
                                  'source': 'path',
                                  'make_x': False}]
        job_2_files_to_download = [{'file_name': 'output.dat',
                                    'local': os.path.join(self.job_2.local_path, 'output.dat'),
                                    'remote': os.path.join(self.job_2.remote_path, 'output.dat'),
                                    'source': 'path',
                                    'make_x': False},
                                   {'file_name': 'check.chk',
                                    'local': os.path.join(self.job_2.local_path, 'check.chk'),
                                    'remote': os.path.join(self.job_2.remote_path, 'check.chk'),
                                    'source': 'path',
                                    'make_x': False}]
        self.assertEqual(self.job_2.files_to_upload, job_2_files_to_upload)
        self.assertEqual(self.job_2.files_to_download, job_2_files_to_download)

        job_3_files_to_upload = [{'file_name': 'submit.sub',
                                  'local': os.path.join(self.job_3.local_path, 'submit.sub'),
                                  'remote': os.path.join(self.job_3.remote_path, 'submit.sub'),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'input.dat',
                                  'local': os.path.join(self.job_3.local_path, 'input.dat'),
                                  'remote': os.path.join(self.job_3.remote_path, 'input.dat'),
                                  'source': 'path',
                                  'make_x': False}]
        job_3_files_to_download = [{'file_name': 'output.dat',
                                    'local': os.path.join(self.job_3.local_path, 'output.dat'),
                                    'remote': os.path.join(self.job_3.remote_path, 'output.dat'),
                                    'source': 'path',
                                    'make_x': False},
                                   {'file_name': 'check.chk',
                                    'local': os.path.join(self.job_3.local_path, 'check.chk'),
                                    'remote': os.path.join(self.job_3.remote_path, 'check.chk'),
                                    'source': 'path',
                                    'make_x': False}]
        self.assertEqual(self.job_3.files_to_upload, job_3_files_to_upload)
        self.assertEqual(self.job_3.files_to_download, job_3_files_to_download)

    def test_get_geometry(self):
        """Test get_geometry function"""
        # xyz in self.species
        xyz_1 = self.job_1.get_geometry()
        self.assertAlmostEqual(xyz_1, str_to_xyz("""O          0.00000        0.00000        0.11779
                                   H          0.00000        0.75545       -0.47116
                                   H          0.00000       -0.75545       -0.47116"""))
        # xyz in self.xyz
        xyz_5 = self.job_5.get_geometry()
        self.assertAlmostEqual(xyz_5, str_to_xyz("""O          0.00000        0.00000        0.11779
                                   H          0.00000        0.75545       -0.47116
                                   H          0.00000       -0.75545       -0.47116"""))

    def test_write_func_args(self):
        """test the write_func_args function"""
        func_args_1 = self.job_1.write_func_args("energy")
        self.assertEqual("name = 'scf', return_wfn='on'", func_args_1)
        func_args_2 = self.job_2.write_func_args("freq")
        self.assertEqual("name = 'b3lyp', return_wfn='on'", func_args_2)
        func_args_3 = self.job_3.write_func_args("optimize")
        self.assertEqual("name = 'wb97x-d', return_wfn='on', return_history='on', engine='optking', dertype='energy'", func_args_3)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'test_Psi4Adapter'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
