#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.adapters.ts.orca_neb module.
"""

import os
import shutil
import tempfile
import datetime
import unittest
import unittest.mock
import pytest

from arc.exceptions import SettingsError
from arc.job import adapter as job_adapter_module
from arc.job.adapters.ts.orca_neb import OrcaNEBAdapter, insert_path_file_copy_back
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
        
        cls.project_directory = tempfile.mkdtemp(prefix='test_OrcaNEBAdapter_')
        cls.addClassCleanup(shutil.rmtree, cls.project_directory, ignore_errors=True)

        # Mock objects for both orca_neb and orca/adapter modules
        mock_input_filenames = {'orca_neb': 'input.in', 'orca': 'input.in'}
        mock_output_filenames = {'orca_neb': 'input.log', 'orca': 'input.log'}
        mock_servers = {'local': {'cluster_soft': 'local', 'un': 'user', 'queues': {}},
                        'remote_server': {'cluster_soft': 'PBS', 'un': 'user', 'path': '/home/user',
                                          'address': 'remote.host.edu', 'queues': {'q': '24:00:00'}},
                        'server_without_a_path': {'cluster_soft': 'PBS', 'un': 'user',
                                                  'address': 'remote.host.edu', 'queues': {'q': '24:00:00'}}}
        mock_submit_filenames = {'local': 'submit.sub', 'PBS': 'submit.sub'}
        mock_orca_neb_settings = {'keyword': {'interpolation': 'IDPP', 'nnodes': 15, 'preopt': 'true'}}
        mock_default_job_settings = {'job_total_memory_gb': 14, 'job_cpu_cores': 8}
        mock_t_max_format = {'local': 'hours', 'PBS': 'hours'}
        mock_submit_scripts = {'local': {'orca': 'mock submit script content'},
                               'remote_server': {'orca': 'mock submit script content'},
                               'server_without_a_path': {'orca': 'mock submit script content'}}

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

        cls.orca_neb_servers_patcher = unittest.mock.patch('arc.job.adapters.ts.orca_neb.servers', mock_servers)
        cls.mock_orca_neb_servers = cls.orca_neb_servers_patcher.start()

        cls.orca_neb_submit_filenames_patcher = unittest.mock.patch('arc.job.adapters.ts.orca_neb.submit_filenames',
                                                                    mock_submit_filenames)
        cls.mock_orca_neb_submit_filenames = cls.orca_neb_submit_filenames_patcher.start()

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

    def test_task_1b_a_remote_server_resolves_the_endpoint_files_on_the_server(self):
        """Orca reads reactant.xyz and product.xyz on the machine the job runs on."""
        job = OrcaNEBAdapter(project='test_orca_neb',
                             job_type='tsg',
                             project_directory=self.project_directory,
                             reactions=[self.reaction],
                             level=self.level,
                             server='remote_server')
        abs_path = job._get_abs_path()
        self.assertEqual(abs_path, job.remote_path)
        self.assertNotEqual(abs_path, job.local_path)
        self.assertTrue(os.path.isabs(abs_path))
        job.write_input_file()
        with open(os.path.join(job.local_path, 'input.in'), 'r') as f:
            content = f.read()
        self.assertIn(f'{job.remote_path}/reactant.xyz', content)
        self.assertIn(f'{job.remote_path}/product.xyz', content)
        self.assertNotIn(job.local_path, content)

    def test_task_1c_a_relative_remote_path_is_refused(self):
        """A server without a configured path yields a relative remote path Orca cannot follow."""
        job = OrcaNEBAdapter(project='test_orca_neb',
                             job_type='tsg',
                             project_directory=self.project_directory,
                             reactions=[self.reaction],
                             level=self.level,
                             server='remote_server')
        job.remote_path = os.path.join('runs', 'ARC_Projects', 'test_orca_neb')
        with self.assertRaises(SettingsError):
            job.write_input_file()

    def test_task_1e_a_server_without_a_path_is_refused_at_construction(self):
        """
        The backstop for a server that startup validation did not see, such as one a job was
        moved to by troubleshooting. It reports the same condition as the startup check and with
        the same error type, so the two read as one settings problem rather than two faults.
        """
        with self.assertRaises(SettingsError):
            OrcaNEBAdapter(project='test_orca_neb',
                           job_type='tsg',
                           project_directory=self.project_directory,
                           reactions=[self.reaction],
                           level=self.level,
                           server='server_without_a_path')

    def test_task_1f_the_refusal_names_the_server_and_the_setting_to_fix(self):
        """A refusal the reader cannot act on costs a support round."""
        job = OrcaNEBAdapter(project='test_orca_neb',
                             job_type='tsg',
                             project_directory=self.project_directory,
                             reactions=[self.reaction],
                             level=self.level,
                             server='remote_server')
        job.remote_path = os.path.join('runs', 'ARC_Projects', 'test_orca_neb')
        with self.assertRaises(SettingsError) as raised:
            job.write_input_file()
        self.assertIn('remote_server', str(raised.exception))
        self.assertIn('path', str(raised.exception))

    def test_task_1d_the_local_server_still_resolves_locally(self):
        """A local job reads the endpoint files from the directory ARC wrote them to."""
        self.assertTrue(os.path.isabs(self.job.remote_path))
        self.assertEqual(self.job._get_abs_path(), self.job.local_path)

    def _remote_job(self) -> OrcaNEBAdapter:
        """A NEB job on a remote PBS server, with its local files written to a fresh directory."""
        return OrcaNEBAdapter(project='test_orca_neb',
                              job_type='tsg',
                              project_directory=tempfile.mkdtemp(dir=self.project_directory),
                              reactions=[self.reaction],
                              level=self.level,
                              server='remote_server')

    def test_task_1g_the_submit_script_copies_the_path_files_to_the_remote_job_directory(self):
        """A submit script that runs Orca in scratch and removes it would otherwise lose the path."""
        submit_script = ('#!/bin/bash -l\n'
                         'cd /scratch/$PBS_JOBID\n'
                         '${{OrcaDir}}/orca input.in > $PBS_O_WORKDIR/input.log\n'
                         'cd $PBS_O_WORKDIR\n'
                         'rm -rf /scratch/$PBS_JOBID\n')
        job = self._remote_job()
        with unittest.mock.patch.dict(job_adapter_module.submit_scripts['remote_server'], {'orca': submit_script}):
            job.set_files()
        with open(os.path.join(job.local_path, 'submit.sub'), 'r') as f:
            lines = f.read().splitlines()
        orca_line = lines.index('${OrcaDir}/orca input.in > $PBS_O_WORKDIR/input.log')
        rm_line = lines.index('rm -rf /scratch/$PBS_JOBID')
        copy_line = next(i for i, line in enumerate(lines) if 'cp -f "$arc_neb_file"' in line)
        self.assertLess(orca_line, copy_line)
        self.assertLess(copy_line, rm_line)
        self.assertIn(f'{job.remote_path}/', lines[copy_line])
        for file_name in ['input_MEP_trj.xyz', 'input_initial_path_trj.xyz', 'input_MEP_ALL_trj.xyz']:
            self.assertIn(file_name, ' '.join(lines))

    def test_task_1h_htcondor_runs_the_neb_with_the_orca_job_shell_file(self):
        """HTCondor runs Orca from job.sh, which the NEB must upload and add the copy to."""
        job = self._remote_job()
        scripts = {'orca': 'Executable = job.sh\n', 'orca_job': '#!/bin/bash -l\n$OrcaDir/orca input.in > input.log\n'}
        with unittest.mock.patch.dict(job_adapter_module.submit_scripts['remote_server'], scripts):
            job.set_files()
        self.assertIn('job.sh', [f['file_name'] for f in job.files_to_upload])
        with open(os.path.join(job.local_path, 'job.sh'), 'r') as f:
            self.assertIn('cp -f "$arc_neb_file"', f.read())
        with open(os.path.join(job.local_path, 'submit.sub'), 'r') as f:
            self.assertEqual(f.read(), 'Executable = job.sh\n')
        self.assertEqual(job.job_adapter, 'orca_neb')

    def test_task_1i_a_submit_script_that_does_not_run_orca_is_reported(self):
        job = self._remote_job()
        with unittest.mock.patch('arc.job.adapters.ts.orca_neb.logger.warning') as warning:
            job.set_files()
        warning.assert_called_once()
        self.assertIn('remote_server', warning.call_args[0][0])

    def test_task_1j_only_the_path_files_orca_wrote_are_downloaded(self):
        """A NEB that stopped early writes some of the files, and a missing one is not an error."""
        job = self._remote_job()
        job.files_to_download = list()
        ssh = unittest.mock.MagicMock()
        ssh.list_file_names.return_value = ['input.log', 'input_initial_path_trj.xyz', 'input.interp', 'input.gbw']
        ssh_context = unittest.mock.MagicMock()
        ssh_context.__enter__.return_value = ssh
        with unittest.mock.patch.object(job, '_open_or_borrow_ssh', return_value=ssh_context):
            job.download_files()
        downloaded = [c.kwargs['remote_file_path'] for c in ssh.download_file.call_args_list]
        self.assertEqual(downloaded, [os.path.join(job.remote_path, 'input_initial_path_trj.xyz'),
                                      os.path.join(job.remote_path, 'input.interp')])
        ssh.list_file_names.assert_called_once_with(remote_path=job.remote_path)

    def test_task_1k_a_local_server_downloads_nothing(self):
        """The copy writes the path files to the local job directory directly."""
        with unittest.mock.patch.object(self.job, '_open_or_borrow_ssh') as open_ssh:
            self.job.download_files()
        open_ssh.assert_not_called()

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
        cls.orca_neb_servers_patcher.stop()
        cls.orca_neb_submit_filenames_patcher.stop()


class TestInsertPathFileCopyBack(unittest.TestCase):
    """
    Contains unit tests for inserting the NEB path file copy into a submit script.
    """

    names = ['input_MEP_trj.xyz', 'input_initial_path_trj.xyz']

    def insert(self, script: str, destination: str = '/home/user/job') -> str | None:
        return insert_path_file_copy_back(script=script, input_file_name='input.in',
                                          path_file_names=self.names, destination=destination)

    def test_the_copy_follows_the_line_that_runs_orca(self):
        script = 'cd $SCRATCH\n/opt/orca/orca input.in > input.log\ncp input.log $WORKDIR\nrm -rf $SCRATCH\n'
        lines = self.insert(script).splitlines()
        self.assertEqual(lines[:2], ['cd $SCRATCH', '/opt/orca/orca input.in > input.log'])
        self.assertEqual(lines[-2:], ['cp input.log $WORKDIR', 'rm -rf $SCRATCH'])
        self.assertEqual(lines[2:-2], [
            '# Copy the Orca NEB path files to the job directory (added by ARC).',
            'if ! [ . -ef /home/user/job ]; then',
            '    for arc_neb_file in input_MEP_trj.xyz input_initial_path_trj.xyz; do',
            '        if [ -f "$arc_neb_file" ]; then cp -f "$arc_neb_file" /home/user/job/ || true; fi',
            '    done',
            'fi'])

    def test_the_ways_submit_scripts_run_orca_are_recognized(self):
        for orca_line in ['orca input.in', '$orcadir/orca input.in > input.log', '${OrcaDir}/orca input.in',
                          '${ORCA_DIR}/orca input.in > input.log', 'srun "$ORCA/orca" "input.in" > out',
                          'time orca input.in > $PBS_O_WORKDIR/input.log || true']:
            with self.subTest(orca_line=orca_line):
                self.assertIsNotNone(self.insert(f'{orca_line}\n'))

    def test_lines_that_do_not_run_orca_on_the_input_file_are_left_alone(self):
        for line in ['cp "$PBS_O_WORKDIR/input.in" "$ORCA_SCRDIR"', 'cp input.in /scratch/orca/',
                     '# ${OrcaDir}/orca input.in > input.log', 'orca input.in.bak', 'orca_2mkl input.in',
                     'Executable = job.sh']:
            with self.subTest(line=line):
                self.assertIsNone(self.insert(f'{line}\n'))

    def test_the_copy_keeps_the_indentation_of_the_orca_line(self):
        script = 'if true; then\n    orca input.in > input.log\nfi\n'
        lines = self.insert(script).splitlines()
        self.assertTrue(all(line.startswith('    ') for line in lines[2:8]))
        self.assertEqual(lines[-1], 'fi')

    def test_the_copy_follows_a_continued_orca_command(self):
        script = 'orca input.in \\\n    > input.log\nrm -rf $SCRATCH'
        lines = self.insert(script).splitlines()
        self.assertEqual(lines[:2], ['orca input.in \\', '    > input.log'])
        self.assertEqual(lines[-1], 'rm -rf $SCRATCH')

    def test_a_destination_with_spaces_is_quoted(self):
        result = self.insert('orca input.in\n', destination='/home/user/my runs/job')
        self.assertIn("cp -f \"$arc_neb_file\" '/home/user/my runs/job'/ || true", result)


if __name__ == '__main__':
    pytest.main([__file__])
