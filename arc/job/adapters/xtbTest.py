#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.xtb module
"""

import os
import shutil
import unittest

import numpy as np

from arc.common import ARC_PATH
from arc.job.adapters.xtb import xTBAdapter
from arc.level import Level
from arc.parser import parse_e_elect, parse_frequencies, parse_geometry
from arc.settings.settings import input_filenames, output_filenames
from arc.species import ARCSpecies


class TestxTBAdapter(unittest.TestCase):
    """
    Contains unit tests for the xTB class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.job_1 = xTBAdapter(execution_type='queue',
                               job_type='sp',
                               project='test_1',
                               project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_xTBAdapter_1'),
                               species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                               testing=True,
                               )
        cls.job_2 = xTBAdapter(execution_type='queue',
                               job_type='opt',
                               project='test_2',
                               fine=True,
                               project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_xTBAdapter_2'),
                               species=[ARCSpecies(label='spc2', smiles='CC[O]')],
                               testing=True,
                               )
        cls.job_3 = xTBAdapter(execution_type='queue',
                               job_type='scan',
                               project='test_3',
                               torsions=[[0, 1, 2, 3]],
                               project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_xTBAdapter_3'),
                               species=[ARCSpecies(label='spc1', xyz="""O       1.09412318   -0.31048292    0.52221323
N      -0.00341051    0.01666411    0.06406731
O      -0.88296129   -0.72127547   -0.38686359
H      -0.20775137    1.01509427    0.05730382""")],
                               testing=True,
                               )
        cls.job_4 = xTBAdapter(execution_type='incore',
                               job_type='scan',
                               constraints=[([1, 2], 1.0), ([1, 2, 3], 75.0), ([1, 2, 3, 4], 270.0)],
                               torsions=[[5, 6, 1, 4]],
                               project='test_4',
                               fine=True,
                               project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_xTBAdapter_4'),
                               species=[ARCSpecies(label='EtOH', smiles='CCO')],
                               testing=True,
                               )
        cls.job_5 = xTBAdapter(execution_type='incore',
                               job_type='sp',
                               project='test_5',
                               project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_xTBAdapter_5'),
                               species=[ARCSpecies(label='NCC', smiles='NCC')]
                               )
        cls.job_6 = xTBAdapter(execution_type='incore',
                               job_type='freq',
                               project='test_6',
                               project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_xTBAdapter_6'),
                               species=[ARCSpecies(label='HO2', smiles='O[O]')]
                               )
        cls.job_7 = xTBAdapter(execution_type='queue',
                               job_type='opt',
                               project='test_7',
                               fine=True,
                               project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_xTBAdapter_7'),
                               species=[ARCSpecies(label='spc7', xyz='O 0 0 1')],
                               testing=True,
                               )
        spc_8 = ARCSpecies(label='MeOH', smiles='CO')
        spc_8.determine_rotors()
        cls.job_8 = xTBAdapter(job_type='scan',
                               level=Level(method='xtb'),
                               project='test_8',
                               project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_xTBAdapter_8'),
                               species=[spc_8],
                               rotor_index=0,
                               )
        cls.job_9 = xTBAdapter(execution_type='queue',
                               job_type='opt',
                               project='test_9',
                               level=Level(method='xtb', solvent='water'),
                               project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_xTBAdapter_9'),
                               species=[ARCSpecies(label='spc2', smiles='CC[O]')],
                               )

    def test_write_input_file(self):
        """Test writing Gaussian input files"""
        self.job_1.write_input_file()
        with open(os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]), 'r') as f:
            content_1 = f.read()
        job_1_expected_input_file = 'xtb input.in > output.out\n'
        self.assertEqual(content_1, job_1_expected_input_file)
        with open(os.path.join(self.job_1.local_path, 'input.in'), 'r') as f:
            content_1_in = f.read()
        job_1_expected_input_file_in = """$coord angs
 0.00000000    0.00000000    1.00000000      o
$eht charge=0 unpaired=2
$end


"""
        self.assertEqual(content_1_in, job_1_expected_input_file_in)

        self.job_7.write_input_file()
        with open(os.path.join(self.job_2.local_path, input_filenames[self.job_2.job_adapter]), 'r') as f:
            content_7 = f.read()
        job_7_expected_input_file = 'xtb input.in --opt vtight > output.out\n'
        self.assertEqual(content_7, job_7_expected_input_file)
        with open(os.path.join(self.job_7.local_path, 'input.in'), 'r') as f:
            content_7_in = f.read()
        job_7_expected_input_file_in = """$coord angs
 0.00000000    0.00000000    1.00000000      o
$eht charge=0 unpaired=2
$end


"""
        self.assertEqual(content_7_in, job_7_expected_input_file_in)

        self.job_3.write_input_file()
        with open(os.path.join(self.job_3.local_path, input_filenames[self.job_3.job_adapter]), 'r') as f:
            content_3 = f.read()
        job_3_expected_input_file = 'xtb input.in --opt > output.out\n'
        self.assertEqual(content_3, job_3_expected_input_file)
        with open(os.path.join(self.job_3.local_path, 'input.in'), 'r') as f:
            content_3_in = f.read()
        job_3_expected_input_file_in = """$coord angs
 1.09412318   -0.31048292    0.52221323      o
-0.00341051    0.01666411    0.06406731      n
-0.88296129   -0.72127547   -0.38686359      o
-0.20775137    1.01509427    0.05730382      h
$eht charge=0 unpaired=2
$end

$constrain
   force constant=0.05
$scan
   dihedral: 1, 2, 3, 4, 179; 179, 539.0, 45
$end
"""
        self.assertEqual(content_3_in, job_3_expected_input_file_in)

        self.job_8.write_input_file()
        with open(os.path.join(self.job_8.local_path, input_filenames[self.job_8.job_adapter]), 'r') as f:
            content_8 = f.read()
        job_8_expected_input_file = 'xtb input.in --opt > output.out\n'
        self.assertEqual(content_8, job_8_expected_input_file)
        with open(os.path.join(self.job_8.local_path, 'input.in'), 'r') as f:
            content_8_in = f.read()
        job_8_expected_input_file_in = """$coord angs
-0.36862686   -0.00871354    0.04292587      c
 0.98182901   -0.04902010    0.46594709      o
-0.57257378    0.95163086   -0.43693396      h
-0.55632373   -0.82564527   -0.65815446      h
-1.01755588   -0.12311763    0.91437513      h
 1.53325126    0.05486569   -0.32815967      h
$eht charge=0 unpaired=0
$end

$constrain
   force constant=0.05
$scan
   dihedral: 3, 1, 2, 6, 299; 299, 659.0, 45
$end
"""
        self.assertEqual(content_8_in, job_8_expected_input_file_in)

        self.job_9.write_input_file()
        with open(os.path.join(self.job_9.local_path, input_filenames[self.job_9.job_adapter]), 'r') as f:
            content_9 = f.read()
        job_9_expected_input_file = 'xtb input.in --opt --alpb water > output.out\n'
        self.assertEqual(content_9, job_9_expected_input_file)

    def test_add_constraints_to_block(self):
        """Test the add_constraints_to_block() method."""
        block = self.job_4.add_constraints_to_block()
        self.assertEqual(block, """   distance: 1, 2, 1.0
   angle: 1, 2, 3, 75.0
   dihedral: 1, 2, 3, 4, 270.0
""")

    def test_set_files(self):
        """Test setting files"""
        job_1_files_to_upload = [{'file_name': 'submit.sub',
                                  'local': os.path.join(self.job_1.local_path, 'submit.sub'),
                                  'remote': os.path.join(self.job_1.remote_path, 'submit.sub'),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'input.sh',
                                  'local': os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]),
                                  'remote': os.path.join(self.job_1.remote_path, input_filenames[self.job_1.job_adapter]),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'input.in',
                                  'local': os.path.join(self.job_1.local_path, 'input.in'),
                                  'remote': os.path.join(self.job_1.remote_path, 'input.in'),
                                  'make_x': False,
                                  'source': 'path'}]
        job_1_files_to_download = [{'file_name': 'output.out',
                                    'local': os.path.join(self.job_1.local_path, output_filenames[self.job_1.job_adapter]),
                                    'remote': os.path.join(self.job_1.remote_path, output_filenames[self.job_1.job_adapter]),
                                    'source': 'path',
                                    'make_x': False},
                                   {'file_name': 'xtbscan.log',
                                    'local': os.path.join(self.job_1.local_path, 'xtbscan.log'),
                                    'remote': os.path.join(self.job_1.remote_path, 'xtbscan.log'),
                                    'make_x': False,
                                    'source': 'path'},
                                   {'file_name': 'g98.out',
                                    'local': os.path.join(self.job_1.local_path, 'g98.out'),
                                    'remote': os.path.join(self.job_1.remote_path, 'g98.out'),
                                    'make_x': False,
                                    'source': 'path'},
                                   {'file_name': 'hessian',
                                    'local': os.path.join(self.job_1.local_path, 'hessian'),
                                    'remote': os.path.join(self.job_1.remote_path, 'hessian'),
                                    'make_x': False,
                                    'source': 'path'}]
        self.assertEqual(self.job_1.files_to_upload, job_1_files_to_upload)
        self.assertEqual(self.job_1.files_to_download, job_1_files_to_download)

    def test_sp(self):
        """Test running a single-point energy calculation using xTB."""
        self.job_5.execute_incore()
        self.assertTrue(os.path.isfile(self.job_5.local_path_to_output_file))
        e_elect = parse_e_elect(self.job_5.local_path_to_output_file, software='xtb')
        self.assertAlmostEqual(e_elect, -28229.88078, places=2)

    def test_opt(self):
        """Test running a geometry optimization calculation using xTB."""
        self.job_2.execute_incore()
        self.assertEqual(parse_geometry(self.job_2.local_path_to_output_file)['symbols'],
                         ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H'))

    def test_freq(self):
        """Test running a vibrational frequency calculation using xTB."""
        self.job_6.execute_incore()
        np.testing.assert_almost_equal(parse_frequencies(self.job_6.local_path_to_output_file),
                                       np.array([1224.98, 1355.27, 3158.76], np.float64))

    def test_scan(self):
        """Test running a 1D scan calculation using xTB."""
        self.job_8.execute()
        self.assertTrue(os.path.isfile(os.path.join(self.job_8.local_path, 'xtbscan.log')))

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        for i in range(10):
            path = os.path.join(ARC_PATH, 'arc', 'testing', f'test_xTBAdapter_{i}')
            shutil.rmtree(path, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
