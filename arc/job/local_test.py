#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.local module
"""

import datetime
import os
import shutil
import unittest
from unittest.mock import patch

import arc.job.local as local
import arc.settings.settings as default_settings
from arc.common import ARC_PATH


class TestLocal(unittest.TestCase):
    """
    Contains unit tests for the local module
    """

    def setUp(self) -> None:
        """Use repository defaults independently of the user's settings overlay."""
        for name in ('servers', 'check_status_command', 'delete_command', 'submit_command',
                     'submit_filenames', 'output_filenames'):
            patcher = patch.object(local, name, getattr(default_settings, name).copy())
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_execute_command(self):
        """Test executing a local command"""
        command1 = 'ls'
        out1 = local.execute_command(command1)
        self.assertIsInstance(out1, tuple)
        self.assertIsInstance(out1[0], list)
        self.assertIsInstance(out1[0][0], str)
        self.assertEqual(out1[1], [])
        if 'arc' in out1[0]:
            # Running from root
            self.assertIn('ARC.py', out1[0])
            self.assertIn('environment.yml', out1[0])
        else:
            # Running directly
            self.assertIn('adapter.py', out1[0])
            self.assertIn('ssh.py', out1[0])

    def test_determine_job_id(self):
        """Test determining a job ID from the stdout of a job submission command."""
        # Slurm
        stdout_1 = ['Submitted batch job 17670585']
        job_id = local._determine_job_id(stdout_1, cluster_soft='slurm')
        self.assertEqual(job_id, '17670585')

        # HTCondor
        stdout_2 = ['Submitting job(s).', '1 job(s) submitted to cluster 5263.']
        job_id = local._determine_job_id(stdout_2)
        self.assertEqual(job_id, '5263')

        # Wrong server name
        with self.assertRaises(ValueError):
            local._determine_job_id(stdout_2, cluster_soft='wrong')

    def test_get_last_modified_time(self):
        """Test the get_last_modified_time() function"""
        path = os.path.join(ARC_PATH, 'ARC.py')
        t = local.get_last_modified_time(path)
        self.assertIsInstance(t, datetime.datetime)
        t = local.get_last_modified_time('no file', path)
        self.assertIsInstance(t, datetime.datetime)

    def test_rename_output(self):
        """Test the rename_output() function"""
        path1 = os.path.join(ARC_PATH, 'scratch', 'input.log')
        path2 = os.path.join(ARC_PATH, 'scratch', 'output.out')
        if not os.path.exists(os.path.join(ARC_PATH, 'scratch')):
            os.makedirs(os.path.join(ARC_PATH, 'scratch'))
        with open(path1, 'w'):
            pass
        local.rename_output(local_file_path=path2, software='gaussian')
        self.assertFalse(os.path.isfile(path1))
        self.assertTrue(os.path.isfile(path2))
        shutil.rmtree(os.path.join(ARC_PATH, 'scratch'), ignore_errors=True)

    def test_parse_running_jobs_ids(self):
        """Test the parse_running_jobs_ids() function"""
        # Slurm:
        stdout = ['             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)',
                  '          10990729    normal     a207   alongd PD       0:00      1 (None)',
                  '          10990728   xeon-p8  xa1001d   alongd  R       0:05      1 d-19-14-2',
                  ]
        running_job_ids = local.parse_running_jobs_ids(stdout, cluster_soft='slurm')
        self.assertEqual(running_job_ids, ['10990729', '10990728'])

        # HTCondor:
        stdout = ['11224.0 R 8 6759 a2495 7',
                  '11225.0 R 8 6759 a2496 6',
                  '11226.0 R 8 6759 a2497 7',
                  '11227.0 R 8 6759 a2498 7',
                  '11228.0 R 8 6759 a2499 7',
                  '11229.0 P 8 6759 a2500 14',
                  '11230.0 P 8 6759 a2501 13',
                  '11231.0 P 8 6759 a2502 13',
                  ]
        running_job_ids = local.parse_running_jobs_ids(stdout, cluster_soft='htcondor')
        self.assertEqual(running_job_ids, ['11224', '11225', '11226', '11227', '11228', '11229', '11230', '11231'])

    def test_submit_job_pbs_compute_node_error(self):
        """Test submit_job() error handling for PBS compute node submissions."""
        stderr = ['qsub: Unauthorized Request: Please do NOT submit jobs on compute nodes!',
                  'Jobs should be submitted on login server, i.e. ZEUS.']
        with patch('arc.job.local.execute_command', side_effect=[([], stderr), ([], stderr)]):
            with patch('time.sleep', return_value=None):
                with self.assertRaises(ValueError) as cm:
                    local.submit_job(path='.', cluster_soft='pbs', submit_cmd='qsub', submit_filename='submit.sh')
        self.assertIn('compute node', str(cm.exception))

    def test_check_running_jobs_ids_without_a_queueing_system(self):
        """Test that a machine with no queueing system reports no queue job IDs"""
        for cluster_soft in ['local', 'Local']:
            with patch.dict(local.servers, {'local': {'cluster_soft': cluster_soft}}):
                with patch('arc.job.local.execute_command') as mock_execute:
                    self.assertEqual(local.check_running_jobs_ids(), list())
                mock_execute.assert_not_called()

    def test_check_running_jobs_ids_unsupported_cluster_software(self):
        """Test that an unrecognized cluster software is still rejected, naming the server"""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'no_such_scheduler'}}):
            with self.assertRaises(ValueError) as cm:
                local.check_running_jobs_ids()
        self.assertIn('local', str(cm.exception))
        self.assertIn('no_such_scheduler', str(cm.exception))

    def test_check_running_jobs_ids_rejects_non_string_scheduler(self) -> None:
        """Invalid values report the server and setting without executing a command."""
        for raw in (None, 42, True, [], {}):
            with self.subTest(raw=raw), \
                    patch.dict(local.servers, {'local': {'cluster_soft': raw}}), \
                    patch('arc.job.local.execute_command') as execute:
                with self.assertRaises(ValueError) as cm:
                    local.check_running_jobs_ids()
                self.assertIn("Server 'local'", str(cm.exception))
                self.assertIn(f"cluster software '{raw}'", str(cm.exception))
                execute.assert_not_called()


class TestClusterSoftNormalisation(unittest.TestCase):
    """``check_running_jobs_ids()`` and ``submit_job()`` must resolve a non-canonical
    ``cluster_soft`` spelling via their real ``get_canonical_cluster_soft('local')`` fallback call
    sites, alias SGE to OGE, refuse an unsupported value naming the server, and leave a canonical
    spelling working unchanged."""

    STATUS_STDOUT = {
        'slurm': ['header', '10990729 rest of line'],
        'oge': ['h0', 'h1', '540420 rest of line'],
        'pbs': ['h0', 'h1', 'h2', 'h3', 'h4', '2016614.zeldo.local rest of line'],
        'htcondor': ['3261.0 R 10 28161 a2719 56'],
    }
    STATUS_EXPECTED_FIRST_ID = {'slurm': '10990729', 'oge': '540420', 'pbs': '2016614', 'htcondor': '3261'}

    def setUp(self) -> None:
        """Use repository defaults independently of the user's settings overlay."""
        for name in ('servers', 'check_status_command', 'delete_command', 'submit_command',
                     'submit_filenames', 'output_filenames'):
            patcher = patch.object(local, name, getattr(default_settings, name).copy())
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_check_running_jobs_ids_resolves_non_canonical_spelling_per_software(self):
        """The status path resolves a non-canonical spelling for every supported cluster software."""
        non_canonical = {'slurm': 'SLURM', 'oge': 'Oge', 'pbs': 'Pbs', 'htcondor': 'HTCONDOR'}
        for canonical_lower, raw in non_canonical.items():
            with self.subTest(raw=raw):
                with patch.dict(local.servers, {'local': {'cluster_soft': raw}}):
                    with patch('arc.job.local.execute_command',
                               return_value=(self.STATUS_STDOUT[canonical_lower], [])):
                        job_ids = local.check_running_jobs_ids()
                self.assertEqual(job_ids[0], self.STATUS_EXPECTED_FIRST_ID[canonical_lower])

    def test_check_running_jobs_ids_aliases_sge_to_oge(self):
        """The status path treats SGE as OGE."""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'sge'}}):
            with patch('arc.job.local.execute_command', return_value=(self.STATUS_STDOUT['oge'], [])):
                job_ids = local.check_running_jobs_ids()
        self.assertEqual(job_ids[0], self.STATUS_EXPECTED_FIRST_ID['oge'])

    def test_check_running_jobs_ids_canonical_value_still_works(self):
        """A canonical spelling on the status path still resolves and parses as before."""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'PBS'}}):
            with patch('arc.job.local.execute_command', return_value=(self.STATUS_STDOUT['pbs'], [])):
                job_ids = local.check_running_jobs_ids()
        self.assertEqual(job_ids[0], self.STATUS_EXPECTED_FIRST_ID['pbs'])

    SUBMIT_STDOUT = {
        'slurm': (['Submitted batch job 17670585'], []),
        'oge': (['Your job 540420 ("job") has been submitted'], []),
        'pbs': (['2016614.zeldo.local'], []),
        'htcondor': (['Submitting job(s).', '1 job(s) submitted to cluster 5263.'], []),
    }
    SUBMIT_EXPECTED_ID = {'slurm': '17670585', 'oge': '540420', 'pbs': '2016614', 'htcondor': '5263'}

    def test_submit_job_resolves_non_canonical_spelling_per_software(self):
        """The submission path resolves a non-canonical spelling for every supported cluster software."""
        non_canonical = {'slurm': 'SLURM', 'oge': 'Oge', 'pbs': 'Pbs', 'htcondor': 'HTCONDOR'}
        for canonical_lower, raw in non_canonical.items():
            with self.subTest(raw=raw):
                with patch.dict(local.servers, {'local': {'cluster_soft': raw}}):
                    with patch('arc.job.local.execute_command',
                               return_value=self.SUBMIT_STDOUT[canonical_lower]):
                        status, job_id = local.submit_job(path='.')
                self.assertEqual(status, 'running')
                self.assertEqual(job_id, self.SUBMIT_EXPECTED_ID[canonical_lower])

    def test_submit_job_aliases_sge_to_oge(self):
        """The submission path treats SGE as OGE."""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'sge'}}):
            with patch('arc.job.local.execute_command', return_value=self.SUBMIT_STDOUT['oge']):
                status, job_id = local.submit_job(path='.')
        self.assertEqual(status, 'running')
        self.assertEqual(job_id, self.SUBMIT_EXPECTED_ID['oge'])

    def test_submit_job_rejects_unsupported_value_naming_the_server(self):
        """The submission path refuses an unrecognised cluster_soft, naming the actual server."""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'sun grid engine'}}):
            with self.assertRaises(ValueError) as cm:
                local.submit_job(path='.')
        self.assertIn('local', str(cm.exception))
        self.assertIn('sun grid engine', str(cm.exception))

    def test_submit_job_canonical_value_still_works(self):
        """A canonical spelling on the submission path still resolves and parses as before."""
        with patch.dict(local.servers, {'local': {'cluster_soft': 'Slurm'}}):
            with patch('arc.job.local.execute_command', return_value=self.SUBMIT_STDOUT['slurm']):
                status, job_id = local.submit_job(path='.')
        self.assertEqual(status, 'running')
        self.assertEqual(job_id, self.SUBMIT_EXPECTED_ID['slurm'])

    def test_submit_job_resolves_non_canonical_explicit_cluster_soft_argument(self):
        """An explicitly-passed, non-canonically-spelled cluster_soft argument still resolves."""
        with patch('arc.job.local.execute_command', return_value=self.SUBMIT_STDOUT['slurm']):
            status, job_id = local.submit_job(path='.', cluster_soft='SLURM')
        self.assertEqual(status, 'running')
        self.assertEqual(job_id, self.SUBMIT_EXPECTED_ID['slurm'])

    def test_submit_job_aliases_sge_to_oge_via_explicit_cluster_soft_argument(self):
        """An explicitly-passed 'sge' cluster_soft argument is treated as OGE."""
        with patch('arc.job.local.execute_command', return_value=self.SUBMIT_STDOUT['oge']):
            status, job_id = local.submit_job(path='.', cluster_soft='sge')
        self.assertEqual(status, 'running')
        self.assertEqual(job_id, self.SUBMIT_EXPECTED_ID['oge'])

    def test_status_and_delete_resolve_overlay_keys(self) -> None:
        """Status parsing and deletion handle whitespace and direct SGE overlays."""
        for raw, key, line, job_id in [(' PBS ', 'pbs', '2016614.zeldo user q a123 1 1 1 -- 01:00 R 00:20', 2016614),
                                     (' SGE ', 'SGE', '540420 0.4 a123 user r date queue', 540420),
                                     (' Slurm ', 'slurm', '10990729 normal a123 user R 0:00 1 node', 10990729)]:
            with self.subTest(raw=raw), \
                    patch.object(local, 'servers', {'local': {'cluster_soft': raw}}), \
                    patch.object(local, 'check_status_command', {key: 'custom status'}), \
                    patch.object(local, 'delete_command', {key: 'custom delete'}), \
                    patch.object(local, 'execute_command', return_value=([line], [])) as execute:
                self.assertEqual(local.check_job_status(job_id), 'running')
                execute.assert_called_with('custom status')
                local.delete_job(job_id)
                execute.assert_called_with(f'custom delete {job_id}', no_fail=True)

    def test_delete_all_jobs_uses_canonical_parser_for_overlay_keys(self) -> None:
        """The SGE alias reaches the deletion branch rather than silently deleting nothing."""
        cases = [(' SGE ', 'SGE', '540420 0.4 a123 user r date queue', 'a123'),
                 (' Slurm ', 'slurm', '123 normal a123 user R', '123'),
                 (' PBS ', 'pbs', '123.server user queue a123', '123.server'),
                 (' HTCondor ', 'htcondor', '123.0 R 10 42 a123 56', '123')]
        for raw, key, line, expected in cases:
            with self.subTest(raw=raw), \
                    patch.object(local, 'servers', {'local': {'cluster_soft': raw}}), \
                    patch.object(local, 'check_status_command', {key: 'custom status'}), \
                    patch.object(local, 'execute_command', return_value=([line], [])) as execute, \
                    patch.object(local, 'delete_job') as delete:
                local.delete_all_local_arc_jobs()
                execute.assert_called_once_with('custom status', no_fail=True)
                delete.assert_called_once_with(expected)

    def test_explicit_submission_resolves_direct_sge_overlay(self) -> None:
        with patch.object(local, 'submit_command', {'SGE': 'custom submit'}), \
                patch.object(local, 'submit_filenames', {'oge': 'custom.sh'}), \
                patch.object(local, 'execute_command', return_value=(['Your job 540420 ("job") has been submitted'], [])) as execute:
            self.assertEqual(local.submit_job('.', cluster_soft=' SGE '), ('running', '540420'))
        execute.assert_called_once_with('cd "."; custom submit custom.sh')

    def test_queue_less_operations_never_execute_a_command(self) -> None:
        with patch.object(local, 'servers', {'local': {'cluster_soft': ' Local '}}), \
                patch.object(local, 'execute_command') as execute:
            self.assertEqual(local.check_running_jobs_ids(), [])
            local.delete_all_local_arc_jobs()
            for operation in (lambda: local.check_job_status(123), lambda: local.delete_job(123),
                              lambda: local.submit_job('.', submit_cmd='custom', submit_filename='custom.sh')):
                with self.assertRaisesRegex(ValueError, 'local.*no queueing system.*not applicable'):
                    operation()
            execute.assert_not_called()



if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
