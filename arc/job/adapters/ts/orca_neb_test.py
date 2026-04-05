#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.adapters.ts.orca_neb module.
"""

import os
import shutil
import datetime
import unittest
import unittest.mock
import pytest

from arc.common import ARC_TESTING_PATH
from arc.job.adapters.ts.orca_neb import OrcaNEBAdapter
from arc.level import Level
from arc.reaction import ARCReaction
from arc.species.species import ARCSpecies


class TestOrcaNEB(unittest.TestCase):
    """
    Contains unit tests for the OrcaNEBAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        
        cls.project_directory = os.path.join(ARC_TESTING_PATH, 'test_OrcaNEBAdapter')
        if os.path.exists(cls.project_directory):
            shutil.rmtree(cls.project_directory)
        os.makedirs(cls.project_directory)

        # Mock objects for both orca_neb and orca/adapter modules
        mock_input_filenames = {'orca_neb': 'input.in', 'orca': 'input.in'}
        mock_output_filenames = {'orca_neb': 'input.log', 'orca': 'input.log'}
        mock_servers = {'local': {'cluster_soft': 'local', 'un': 'user', 'queues': {}}}
        mock_submit_filenames = {'local': 'submit.sub'}
        mock_orca_neb_settings = {'keyword': {'interpolation': 'IDPP', 'nnodes': 15, 'preopt': 'true'}}
        mock_default_job_settings = {'job_total_memory_gb': 14, 'job_cpu_cores': 8}
        mock_t_max_format = {'local': 'hours'}
        mock_submit_scripts = {'local': {'orca': 'mock submit script content'}}

        # 1. Mock settings in orca_neb module
        cls.settings_patcher = unittest.mock.patch('arc.job.adapters.ts.orca_neb.settings', {
            'input_filenames': mock_input_filenames,
            'output_filenames': mock_output_filenames,
            'servers': mock_servers,
            'submit_filenames': mock_submit_filenames,
            'orca_neb_settings': mock_orca_neb_settings,
            'default_job_settings': mock_default_job_settings
        })
        cls.mock_settings = cls.settings_patcher.start()

        # 2. Mock settings in orca module (it's imported as OrcaAdapter)
        cls.orca_settings_patcher = unittest.mock.patch('arc.job.adapters.orca.settings', {
            'input_filenames': mock_input_filenames,
            'output_filenames': mock_output_filenames,
            'servers': mock_servers,
            'submit_filenames': mock_submit_filenames,
            'default_job_settings': mock_default_job_settings,
            'global_ess_settings': {}
        })
        cls.mock_orca_settings = cls.orca_settings_patcher.start()

        # 3. Mock module-level variables in arc.job.adapter and arc.job.adapters.ts.orca_neb's global scope
        cls.adapter_servers_patcher = unittest.mock.patch('arc.job.adapter.servers', mock_servers)
        cls.mock_adapter_servers = cls.adapter_servers_patcher.start()

        cls.adapter_submit_filenames_patcher = unittest.mock.patch('arc.job.adapter.submit_filenames', mock_submit_filenames)
        cls.mock_adapter_submit_filenames = cls.adapter_submit_filenames_patcher.start()

        cls.adapter_submit_scripts_patcher = unittest.mock.patch('arc.job.adapter.submit_scripts', mock_submit_scripts)
        cls.mock_adapter_submit_scripts = cls.adapter_submit_scripts_patcher.start()

        cls.adapter_t_max_format_patcher = unittest.mock.patch('arc.job.adapter.t_max_format', mock_t_max_format)
        cls.mock_adapter_t_max_format = cls.adapter_t_max_format_patcher.start()

        cls.adapter_output_filenames_patcher = unittest.mock.patch('arc.job.adapter.output_filenames', mock_output_filenames)
        cls.mock_adapter_output_filenames = cls.adapter_output_filenames_patcher.start()

        # Also need to mock output_filenames in orca_neb global scope if it was imported as such
        cls.orca_neb_output_filenames_patcher = unittest.mock.patch('arc.job.adapters.ts.orca_neb.output_filenames', mock_output_filenames)
        cls.mock_orca_neb_output_filenames = cls.orca_neb_output_filenames_patcher.start()

        # 4. Setup species and reaction
        cls.r_species = ARCSpecies(label='i-C3H7', smiles='C[CH]C')
        cls.p_species = ARCSpecies(label='n-C3H7', smiles='CC[CH2]')
        cls.reaction = ARCReaction(r_species=[cls.r_species],
                                   p_species=[cls.p_species])
        
        cls.level = Level(method='b3lyp', basis='sto-3g',
                          args={'keyword': {'scf_convergence': 'tightscf'}})
        
        # Initialize the adapter
        cls.job = OrcaNEBAdapter(project='test_orca_neb',
                                 job_type='tsg',
                                 project_directory=cls.project_directory,
                                 reactions=[cls.reaction],
                                 level=cls.level,
                                 server='local')
        # Ensure remote_path is set for tests
        cls.job.remote_path = os.path.join('/some/remote/path', cls.job.project, cls.job.job_type[0], cls.job.job_name)

    def test_task_1_preparation(self):
        """
        Task 1: Creating the input files and setting all the files.
        """
        # 1. Test set_files
        self.job.files_to_upload = list()
        self.job.files_to_download = list()
        self.job.set_files()
        
        # Check that files were added to upload/download lists
        upload_names = [f['file_name'] for f in self.job.files_to_upload]
        self.assertIn('input.in', upload_names)
        self.assertIn('reactant.xyz', upload_names)
        self.assertIn('product.xyz', upload_names)
        
        download_names = [f['file_name'] for f in self.job.files_to_download]
        self.assertIn('input.log', download_names)

        # 2. Verify files exist on disk
        self.assertTrue(os.path.exists(os.path.join(self.job.local_path, 'input.in')))
        self.assertTrue(os.path.exists(os.path.join(self.job.local_path, 'reactant.xyz')))
        self.assertTrue(os.path.exists(os.path.join(self.job.local_path, 'product.xyz')))

        # Verify input file content partially
        with open(os.path.join(self.job.local_path, 'input.in'), 'r') as f:
            content = f.read()
            self.assertIn('NEB-TS', content)
            self.assertIn('reactant.xyz', content)
            self.assertIn('product.xyz', content)

    def test_task_2_post_processing(self):
        """
        Task 2: Post processing parts.
        """
        # Path to the provided log file
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                                'testing', 'neb', 'neb_res.out')
        
        # Ensure the job's output path points to where we will copy the log
        output_path = self.job.local_path_to_output_file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy(log_path, output_path)
        
        self.job.reactions[0].ts_species.ts_guesses = list()
        self.job.initial_time = datetime.datetime(2026, 2, 16, 10, 0, 0)
        self.job.final_time = datetime.datetime(2026, 2, 16, 10, 15, 42)
        
        self.job.process_run()
        
        self.assertEqual(len(self.job.reactions[0].ts_species.ts_guesses), 1)
        tsg = self.job.reactions[0].ts_species.ts_guesses[0]
        self.assertTrue(tsg.success)
        self.assertEqual(tsg.method, 'orca_neb')
        self.assertIsNotNone(tsg.initial_xyz)
        self.assertEqual(len(tsg.initial_xyz['symbols']), 10)
        
        # Verify coordinates match the last one in the log
        # C -1.406738 -0.055989 0.104836
        self.assertAlmostEqual(tsg.initial_xyz['coords'][0][0], -1.406738, places=5)

    @classmethod
    def tearDownClass(cls):
        """
        A method that is run ONCE after all unit tests in this class.
        """
        cls.settings_patcher.stop()
        cls.orca_settings_patcher.stop()
        cls.adapter_servers_patcher.stop()
        cls.adapter_submit_filenames_patcher.stop()
        cls.adapter_submit_scripts_patcher.stop()
        cls.adapter_t_max_format_patcher.stop()
        cls.adapter_output_filenames_patcher.stop()
        cls.orca_neb_output_filenames_patcher.stop()
        shutil.rmtree(cls.project_directory, ignore_errors=True)


if __name__ == '__main__':
    pytest.main([__file__])
