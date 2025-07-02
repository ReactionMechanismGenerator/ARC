#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.xtb_gsm module
"""

import os
import shutil
import unittest

from arc.common import ARC_PATH
from arc.job.adapters.ts.xtb_gsm import xTBGSMAdapter
from arc.level import Level
from arc.parser.parser import parse_trajectory
from arc.reaction import ARCReaction
from arc.species.species import ARCSpecies


class TestxTBGSMAdapter(unittest.TestCase):
    """
    Contains unit tests for the xTBGSMAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.job_1 = xTBGSMAdapter(project='test_1',
                                  job_type='tsg',
                                  project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_xTBAGSMdapter_1'),
                                  reactions=[ARCReaction(r_species=[ARCSpecies(label='HNO', smiles='N=O')],
                                                         p_species=[ARCSpecies(label='HON', smiles='[N-]=[OH+]')])],
                                  )
        cls.job_2 = xTBGSMAdapter(execution_type='incore',
                                  project='test_2',
                                  job_type='tsg',
                                  level=Level(method='xtb',  # These settings make the test converge in <1 min instead of ~25 min
                                              args={'keyword': {'max_opt_iters': 10,
                                                                'conv_tol': 0.5,
                                                                'add_node_tol': 0.5,
                                                                'final_opt': 10,
                                                                'nnodes': 9}}),
                                  project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_xTBAGSMdapter_2'),
                                  reactions=[ARCReaction(r_species=[ARCSpecies(label='HNO', smiles='N=O')],
                                                         p_species=[ARCSpecies(label='HON', smiles='[N-]=[OH+]')])],
                                  )
        cls.job_2.reactions[0].ts_species = ARCSpecies(label='TS2', is_ts=True)

    def test_set_files(self):
        """Test setting files"""
        job_1_files_to_upload = [{'file_name': 'submit.sub',
                                  'local': os.path.join(self.job_1.local_path, 'submit.sub'),
                                  'remote': os.path.join(self.job_1.remote_path, 'submit.sub'),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'initial0000.xyz',
                                  'local': os.path.join(self.job_1.local_path, 'scratch', 'initial0000.xyz'),
                                  'remote': os.path.join(self.job_1.remote_path, 'scratch', 'initial0000.xyz'),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'gsm.orca',
                                  'local': os.path.join(self.job_1.xtb_gsm_scripts_path, 'gsm.orca'),
                                  'remote': os.path.join(self.job_1.remote_path, 'gsm.orca'),
                                  'source': 'path',
                                  'make_x': True},
                                 {'file_name': 'inpfileq',
                                  'local': os.path.join(self.job_1.xtb_gsm_scripts_path, 'inpfileq'),
                                  'remote': os.path.join(self.job_1.remote_path, 'inpfileq'),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'ograd',
                                  'local': os.path.join(self.job_1.xtb_gsm_scripts_path, 'ograd'),
                                  'remote': os.path.join(self.job_1.remote_path, 'ograd'),
                                  'source': 'path', 'make_x': False},
                                 {'file_name': 'tm2orca.py',
                                  'local': os.path.join(self.job_1.xtb_gsm_scripts_path, 'tm2orca.py'),
                                  'remote': os.path.join(self.job_1.remote_path, 'tm2orca.py'),
                                  'source': 'path',
                                  'make_x': True}]
        job_1_files_to_download = [{'file_name': 'stringfile.xyz0000',
                                    'local': os.path.join(self.job_1.local_path, 'stringfile.xyz0000'),
                                    'remote': os.path.join(self.job_1.remote_path, 'stringfile.xyz0000'),
                                    'source': 'path', 'make_x': False}]
        self.assertEqual(self.job_1.files_to_upload, job_1_files_to_upload)
        self.assertEqual(self.job_1.files_to_download, job_1_files_to_download)

        self.assertEqual(self.job_2.files_to_upload, list())
        self.assertEqual(self.job_2.files_to_download, list())

    def test_write_input_file(self):
        """Test writing the initial0000.xyz file"""
        self.job_1.write_input_file()
        expected_string = """3

N      -0.51560854    0.35498613    0.00000000
O       0.53496586   -0.29425836    0.00000000
H      -1.32625080   -0.26220791    0.00000000
3

N       0.71282235   -0.34771438    0.00000000
O      -0.54253715    0.32431711    0.00000000
H      -1.29374518   -0.31588249    0.00000000
"""
        with open(self.job_1.scratch_initial0000_path, 'r') as f:
            actual_string = f.read()
        self.assertEqual(actual_string, expected_string)

    def test_set_inpfileq_keywords(self):
        """Test the set_inpfileq_keywords() method."""
        keywords = self.job_1.set_inpfileq_keywords()
        self.assertEqual(keywords['conv_tol'], '0.0005')
        self.assertEqual(keywords['growth_direction'], '0')
        self.assertEqual(keywords['initial_opt'], '0')
        self.assertEqual(keywords['final_opt'], '150')
        self.assertEqual(keywords['nnodes'], '15')

        keywords = self.job_2.set_inpfileq_keywords()
        self.assertEqual(keywords['conv_tol'], '0.5')
        self.assertEqual(keywords['max_opt_iters'], '10')
        self.assertEqual(keywords['add_node_tol'], '0.5')
        self.assertEqual(keywords['growth_direction'], '0')
        self.assertEqual(keywords['initial_opt'], '0')
        self.assertEqual(keywords['int_thresh'], '2.0')
        self.assertEqual(keywords['final_opt'], '10')
        self.assertEqual(keywords['nnodes'], '9')

    def test_execute_incore(self):
        """Test executing DE-GSM via xTB"""
        self.job_2.execute()
        traj = parse_trajectory(self.job_2.stringfile_path)
        self.assertEqual(len(traj), 9)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        for i in range(10):
            path = os.path.join(ARC_PATH, 'arc', 'testing', f'test_xTBAGSMdapter_{i}')
            shutil.rmtree(path, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
