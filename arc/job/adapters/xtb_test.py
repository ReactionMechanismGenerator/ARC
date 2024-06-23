#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.xtb module
"""

import os
import shutil
import unittest

from arc.common import ARC_PATH, almost_equal_coords
from arc.job.adapters.xtb_adapter import xTBAdapter
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
                               level=Level(method='gfn1', args={'keyword': {'parallel': 'no'}}),
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
        cls.job_10 = xTBAdapter(execution_type='incore',
                                job_type='opt',
                                project='test_10',
                                level=Level(method='gfn2'),
                                project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_xTBAdapter_10'),
                                species=[ARCSpecies(label='TS', is_ts=True,
                                                    xyz="""C       -1.317288   -0.260819   -0.032638
                                                           C       -0.041481    0.531329    0.023478
                                                           C        1.279070   -0.197051   -0.023973
                                                           H       -2.209628    0.216416   -0.384249
                                                           H       -1.256224   -1.330754   -0.054542
                                                           H       -0.820484    0.291169    1.063154
                                                           H       -0.088157    1.550626   -0.307983
                                                           H        2.089909    0.431147    0.325728
                                                           H        1.509031   -0.512144   -1.040023
                                                           H        1.253745   -1.087218    0.596715""",
                                                    )],
                                )
        cls.ts_xyz = {'symbols': ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                      'isotopes': (12, 12, 12, 1, 1, 1, 1, 1, 1, 1),
                      'coords': ((-1.3345338960229, -0.2632523445749516, 0.03188606176448177),
                                 (-0.03589479579707156, 0.527632973819735, 0.04434529405302345),
                                 (1.2724590810435725, -0.19277067064013267, -0.025392450763076224),
                                 (-2.20104215102263, 0.18852992959406212, -0.4131838945600057),
                                 (-1.2676136848652564, -1.3347550453784343, 0.008605864856721757),
                                 (-0.8016438599302824, 0.3367173257930604, 1.0387122280763188),
                                 (-0.07474178000782485, 1.5210510840466809, -0.3750121755693191),
                                 (2.0764264425482923, 0.4046067573717955, 0.4055127476238458),
                                 (1.5460137858952223, -0.4172197395570762, -1.0619193027555185),
                                 (1.219063858158878, -1.1378392704747389, 0.512112627273528))}
        cls.job_11 = xTBAdapter(execution_type='incore',
                                job_type='freq',
                                project='test_11',
                                level=Level(method='gfn2'),
                                project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_xTBAdapter_11'),
                                species=[ARCSpecies(label='TS', is_ts=True, xyz=cls.ts_xyz)],
                                )
        cls.job_12 = xTBAdapter(execution_type='incore',
                                job_type='freq',
                                project='test_12',
                                level=Level(method='gfn2'),
                                project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_xTBAdapter_12'),
                                species=[ARCSpecies(label='H2', smiles='[H][H]')],
                                )

    def test_write_input_file(self):
        """Test writing Gaussian input files"""
        self.job_1.write_input_file()
        with open(os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]), 'r') as f:
            content_1 = f.read()
        job_1_expected_input_file = 'mol.sdf --gfn1 --uhf 2 --chrg 0 > output.out\n'
        self.assertEqual(content_1.split('xtb ')[1], job_1_expected_input_file)

        self.job_7.write_input_file()
        with open(os.path.join(self.job_2.local_path, input_filenames[self.job_2.job_adapter]), 'r') as f:
            content_7 = f.read()
        job_7_expected_input_file = 'mol.sdf --opt vtight --gfn2 --parallel --uhf 1 --chrg 0 > output.out\n'
        self.assertEqual(content_7.split('xtb ')[1], job_7_expected_input_file)

        self.job_3.write_input_file()
        with open(os.path.join(self.job_3.local_path, input_filenames[self.job_3.job_adapter]), 'r') as f:
            content_3 = f.read()
        job_3_expected_input_file = 'mol.sdf --opt tight --gfn2 --input input.in --parallel --uhf 2 --chrg 0 > output.out\n'
        self.assertEqual(content_3.split('xtb ')[1], job_3_expected_input_file)
        with open(os.path.join(self.job_3.local_path, 'input.in'), 'r') as f:
            content_3_in = f.read()
        job_3_expected_input_file_in = """$constrain
   force constant=0.05
$scan
   dihedral: 1, 2, 3, 4, 179; 179, 539.0, 45
$end"""
        self.assertEqual(content_3_in, job_3_expected_input_file_in)

        self.job_8.write_input_file()
        with open(os.path.join(self.job_8.local_path, input_filenames[self.job_8.job_adapter]), 'r') as f:
            content_8 = f.read()
        job_8_expected_input_file = 'mol.sdf --opt tight --gfn2 --input input.in --parallel --uhf 0 --chrg 0 > output.out\n'
        self.assertEqual(content_8.split('xtb ')[1], job_8_expected_input_file)
        with open(os.path.join(self.job_8.local_path, 'input.in'), 'r') as f:
            content_8_in = f.read()
        job_8_expected_input_file_in = """$constrain
   force constant=0.05
$scan
   dihedral: 3, 1, 2, 6, 299; 299, 659.0, 45
$end"""
        self.assertEqual(content_8_in, job_8_expected_input_file_in)

        self.job_9.write_input_file()
        with open(os.path.join(self.job_9.local_path, input_filenames[self.job_9.job_adapter]), 'r') as f:
            content_9 = f.read()
        job_9_expected_input_file = 'mol.sdf --opt tight --gfn2 --alpb water --parallel --uhf 1 --chrg 0 > output.out\n'
        self.assertEqual(content_9.split('xtb ')[1], job_9_expected_input_file)

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
                                  'source': 'path'},
                                 {'file_name': 'mol.sdf',
                                  'local': os.path.join(self.job_1.local_path, 'mol.sdf'),
                                  'remote': os.path.join(self.job_1.remote_path, 'mol.sdf'),
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
        self.assertAlmostEqual(e_elect, -28229.8803, places=2)

    def test_opt(self):
        """Test running a geometry optimization calculation using xTB."""
        self.job_2.execute_incore()
        self.assertEqual(parse_geometry(self.job_2.local_path_to_output_file)['symbols'],
                         ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H'))

    def test_opt_ts(self):
        """Test optimizing a TS using Sella."""
        self.assertIsNone(self.job_10.species[0].final_xyz)
        self.job_10.execute_incore()
        self.assertTrue(almost_equal_coords(self.job_10.species[0].final_xyz, self.ts_xyz))

    def test_freq(self):
        """Test running a vibrational frequency calculation using xTB."""
        self.job_6.execute_incore()
        expected_freqs = [1225, 1355, 3158]
        for freq, expected_freq in zip(parse_frequencies(self.job_6.local_path_to_output_file), expected_freqs):
            self.assertAlmostEqual(freq, expected_freq, places=0)

        self.job_11.execute_incore()
        expected_freqs = [-1825.7, 142.7, 337.6, 364.4, 679.1, 704.7]
        for freq, expected_freq in zip(parse_frequencies(self.job_11.local_path_to_output_file)[:5], expected_freqs):
            self.assertAlmostEqual(freq, expected_freq, places=0)

        self.job_12.execute_incore()
        expected_freqs = [4225.72]
        for freq, expected_freq in zip(parse_frequencies(self.job_12.local_path_to_output_file), expected_freqs):
            self.assertAlmostEqual(freq, expected_freq, places=0)

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
        for i in range(20):
            path = os.path.join(ARC_PATH, 'arc', 'testing', f'test_xTBAdapter_{i}')
            shutil.rmtree(path, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
