#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.pipe.pipe_coordinator module
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from arc.job.pipe.pipe_coordinator import PipeCoordinator
from arc.job.pipe.pipe_run import PipeRun
from arc.job.pipe.pipe_state import (
    PipeRunState,
    TaskState,
    TaskSpec,
    update_task_state,
)
from arc.species import ARCSpecies


_pipe_patches = []


def setUpModule():
    """Enable pipe mode for all tests in this module."""
    global _pipe_patches
    pipe_vals = {'enabled': True, 'min_tasks': 10, 'max_workers': 100,
                 'max_attempts': 3, 'lease_duration_hrs': 24}
    p = patch.dict('arc.job.pipe.pipe_coordinator.pipe_settings', pipe_vals)
    p.start()
    _pipe_patches.append(p)


def tearDownModule():
    """Restore pipe settings."""
    global _pipe_patches
    for p in _pipe_patches:
        p.stop()
    _pipe_patches.clear()


def _make_spec(task_id, task_family='conf_opt', engine='mockter', level=None,
               species_label='H2O', conformer_index=0, cores=4, mem=2048):
    """Helper to create a TaskSpec for testing."""
    spc = ARCSpecies(label=species_label, smiles='O')
    return TaskSpec(
        task_id=task_id,
        task_family=task_family,
        owner_type='species',
        owner_key=species_label,
        input_fingerprint=f'{task_id}_fp',
        engine=engine,
        level=level or {'method': 'mock', 'basis': 'mock'},
        required_cores=cores,
        required_memory_mb=mem,
        input_payload={'species_dicts': [spc.as_dict()]},
        ingestion_metadata={'conformer_index': conformer_index},
    )


def _make_mock_sched(project_directory):
    """Create a mock Scheduler with the attributes PipeCoordinator needs."""
    sched = MagicMock()
    sched.project = 'pipe_test_project'
    sched.project_directory = project_directory
    sched.ess_settings = {'orca': ['local'], 'mockter': ['local']}
    sched.testing = True
    sched.server_job_ids = list()
    spc = ARCSpecies(label='H2O', smiles='O')
    spc.conformers = [None] * 5
    spc.conformer_energies = [None] * 5
    sched.species_dict = {'H2O': spc}
    sched.output = {'H2O': {'paths': {}, 'job_types': {}}}
    return sched


def _complete_task(pipe_root, task_id):
    """Drive a task through the full lifecycle to COMPLETED."""
    now = time.time()
    update_task_state(pipe_root, task_id, new_status=TaskState.CLAIMED,
                      claimed_by='w', claim_token='tok',
                      claimed_at=now, lease_expires_at=now + 300)
    update_task_state(pipe_root, task_id, new_status=TaskState.RUNNING, started_at=now)
    update_task_state(pipe_root, task_id, new_status=TaskState.COMPLETED, ended_at=now)


class TestShouldUsePipe(unittest.TestCase):
    """Tests for PipeCoordinator.should_use_pipe()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_coord_test_')
        self.coord = PipeCoordinator(_make_mock_sched(self.tmpdir))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_true_for_homogeneous_batch(self):
        tasks = [_make_spec(f't_{i}') for i in range(15)]
        self.assertTrue(self.coord.should_use_pipe(tasks))

    def test_false_below_threshold(self):
        tasks = [_make_spec(f't_{i}') for i in range(5)]
        self.assertFalse(self.coord.should_use_pipe(tasks))

    def test_false_for_empty_list(self):
        self.assertFalse(self.coord.should_use_pipe([]))

    def test_false_for_heterogeneous_engine(self):
        tasks = [_make_spec(f't_{i}') for i in range(15)]
        tasks[0] = _make_spec('t_0', engine='gaussian')
        self.assertFalse(self.coord.should_use_pipe(tasks))

    def test_false_for_heterogeneous_level(self):
        tasks = [_make_spec(f't_{i}') for i in range(15)]
        tasks[3] = _make_spec('t_3', level={'method': 'b3lyp', 'basis': 'sto-3g'})
        self.assertFalse(self.coord.should_use_pipe(tasks))

    def test_false_for_heterogeneous_family(self):
        tasks = [_make_spec(f't_{i}') for i in range(15)]
        tasks[0] = _make_spec('t_0', task_family='conf_sp')
        self.assertFalse(self.coord.should_use_pipe(tasks))

    @patch('arc.job.pipe.pipe_coordinator.pipe_settings', {'enabled': False, 'min_tasks': 10})
    def test_false_when_disabled(self):
        tasks = [_make_spec(f't_{i}') for i in range(15)]
        self.assertFalse(self.coord.should_use_pipe(tasks))


class TestSubmitPipeRun(unittest.TestCase):
    """Tests for PipeCoordinator.submit_pipe_run()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_coord_submit_')
        self.coord = PipeCoordinator(_make_mock_sched(self.tmpdir))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_submit_returns_pipe_run(self):
        tasks = [_make_spec(f't_{i}') for i in range(3)]
        pipe = self.coord.submit_pipe_run('run_001', tasks)
        self.assertIsInstance(pipe, PipeRun)
        self.assertIn('run_001', self.coord.active_pipes)
        self.assertIs(self.coord.active_pipes['run_001'], pipe)

    def test_submit_stages_on_disk(self):
        tasks = [_make_spec(f't_{i}') for i in range(2)]
        pipe = self.coord.submit_pipe_run('run_disk', tasks)
        self.assertTrue(os.path.isdir(pipe.pipe_root))
        for t in tasks:
            self.assertTrue(os.path.isfile(
                os.path.join(pipe.pipe_root, 'tasks', t.task_id, 'spec.json')))

    def test_submit_uses_explicit_cluster_software(self):
        tasks = [_make_spec('t_0')]
        pipe = self.coord.submit_pipe_run('run_pbs', tasks, cluster_software='pbs')
        self.assertEqual(pipe.cluster_software, 'pbs')

    def test_submit_adds_job_id_to_server_job_ids(self):
        """Submitted pipe job ID is added to server_job_ids to prevent stale-snapshot race."""
        tasks = [_make_spec('t_0')]
        with patch.object(PipeRun, 'submit_to_scheduler', return_value=('submitted', '12345[]')):
            pipe = self.coord.submit_pipe_run('run_ids', tasks)
        self.assertIn('12345[]', self.coord.sched.server_job_ids)
        self.assertEqual(pipe.scheduler_job_id, '12345[]')


class TestRegisterFromDir(unittest.TestCase):
    """Tests for PipeCoordinator.register_pipe_run_from_dir()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_coord_register_')
        self.coord = PipeCoordinator(_make_mock_sched(self.tmpdir))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_register_reconstructs(self):
        tasks = [_make_spec(f't_{i}') for i in range(2)]
        original = self.coord.submit_pipe_run('run_restore', tasks, cluster_software='pbs')
        pipe_root = original.pipe_root
        del self.coord.active_pipes['run_restore']
        restored = self.coord.register_pipe_run_from_dir(pipe_root)
        self.assertIn('run_restore', self.coord.active_pipes)
        self.assertEqual(restored.run_id, 'run_restore')
        self.assertEqual(restored.cluster_software, 'pbs')


class TestPollPipes(unittest.TestCase):
    """Tests for PipeCoordinator.poll_pipes()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_coord_poll_')
        self.coord = PipeCoordinator(_make_mock_sched(self.tmpdir))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_poll_removes_completed_pipe(self):
        pipe = self.coord.submit_pipe_run('run_done', [_make_spec('t_done')])
        _complete_task(pipe.pipe_root, 't_done')
        self.coord.poll_pipes()
        self.assertNotIn('run_done', self.coord.active_pipes)

    def test_poll_keeps_pending_pipe(self):
        self.coord.submit_pipe_run('run_pending', [_make_spec('t_pending')])
        self.coord.poll_pipes()
        self.assertIn('run_pending', self.coord.active_pipes)

    def test_poll_removes_failed_pipe(self):
        pipe = self.coord.submit_pipe_run('run_fail', [_make_spec('t_fail')])
        pipe.status = PipeRunState.FAILED
        pipe._save_run_metadata()
        self.coord.poll_pipes()
        self.assertNotIn('run_fail', self.coord.active_pipes)

    def test_poll_removes_after_repeated_reconcile_failures(self):
        pipe = self.coord.submit_pipe_run('run_stuck', [_make_spec('t_stuck')])
        with patch.object(pipe, 'reconcile', side_effect=RuntimeError('corrupt')):
            for _ in range(3):
                self.coord.poll_pipes()
        self.assertNotIn('run_stuck', self.coord.active_pipes)

    def test_poll_resets_failure_count_on_success(self):
        pipe = self.coord.submit_pipe_run('run_flaky', [_make_spec('t_flaky')])
        with patch.object(pipe, 'reconcile', side_effect=RuntimeError('transient')):
            self.coord.poll_pipes()
        self.assertEqual(self.coord._pipe_poll_failures.get('run_flaky'), 1)
        self.coord.poll_pipes()  # succeeds this time
        self.assertNotIn('run_flaky', self.coord._pipe_poll_failures)

    def test_resubmission_adds_job_id_to_server_job_ids(self):
        """Resubmitted pipe job ID is added to server_job_ids."""
        pipe = self.coord.submit_pipe_run('run_resub', [_make_spec('t_resub')])

        def fake_reconcile():
            pipe._needs_resubmission = True
            return {TaskState.PENDING.value: 1}

        with patch.object(pipe, 'reconcile', side_effect=fake_reconcile), \
             patch.object(pipe, 'submit_to_scheduler', return_value=('submitted', '77777[]')):
            self.coord.poll_pipes()
        self.assertIn('77777[]', self.coord.sched.server_job_ids)


class TestIsSchedulerJobAlive(unittest.TestCase):
    """Tests for PipeCoordinator._is_scheduler_job_alive()."""

    def test_none_server_ids_returns_true(self):
        pipe = MagicMock()
        pipe.scheduler_job_id = '12345[]'
        self.assertTrue(PipeCoordinator._is_scheduler_job_alive(pipe, None))

    def test_none_scheduler_job_id_returns_true(self):
        pipe = MagicMock()
        pipe.scheduler_job_id = None
        self.assertTrue(PipeCoordinator._is_scheduler_job_alive(pipe, ['12345[0]']))

    def test_pbs_array_element_in_queue(self):
        pipe = MagicMock()
        pipe.scheduler_job_id = '4018898[]'
        self.assertTrue(PipeCoordinator._is_scheduler_job_alive(pipe, ['4018898[2]', '9999']))

    def test_pbs_array_not_in_queue(self):
        pipe = MagicMock()
        pipe.scheduler_job_id = '4018898[]'
        self.assertFalse(PipeCoordinator._is_scheduler_job_alive(pipe, ['9999', '5555']))

    def test_non_array_job_in_queue(self):
        pipe = MagicMock()
        pipe.scheduler_job_id = '12345'
        self.assertTrue(PipeCoordinator._is_scheduler_job_alive(pipe, ['12345', '9999']))

    def test_non_array_job_not_in_queue(self):
        pipe = MagicMock()
        pipe.scheduler_job_id = '12345'
        self.assertFalse(PipeCoordinator._is_scheduler_job_alive(pipe, ['9999']))

    def test_empty_queue(self):
        pipe = MagicMock()
        pipe.scheduler_job_id = '12345[]'
        self.assertFalse(PipeCoordinator._is_scheduler_job_alive(pipe, []))

    def test_slurm_array_element_in_queue(self):
        pipe = MagicMock()
        pipe.scheduler_job_id = '12345'
        self.assertTrue(PipeCoordinator._is_scheduler_job_alive(pipe, ['12345_7', '9999']))

    def test_slurm_array_not_in_queue(self):
        pipe = MagicMock()
        pipe.scheduler_job_id = '12345'
        self.assertFalse(PipeCoordinator._is_scheduler_job_alive(pipe, ['99999_7']))


class TestPollPipesJobGone(unittest.TestCase):
    """Test that poll_pipes passes server_job_ids to reconcile for orphan detection."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_coord_gone_')
        self.coord = PipeCoordinator(_make_mock_sched(self.tmpdir))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_poll_with_server_job_ids_cleans_stuck_pipe(self):
        """A pipe whose scheduler job left the queue gets cleaned up."""
        pipe = self.coord.submit_pipe_run('run_stuck', [_make_spec('t0')])
        pipe.scheduler_job_id = '99999[]'
        now = time.time()
        update_task_state(pipe.pipe_root, 't0', new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='tok',
                          claimed_at=now, lease_expires_at=now + 86400)
        update_task_state(pipe.pipe_root, 't0', new_status=TaskState.RUNNING, started_at=now)
        # Job 99999 is NOT in the queue — pass empty list.
        self.coord.poll_pipes(server_job_ids=[])
        # The pipe should have been cleaned up (orphaned → failed_terminal → ingested).
        self.assertNotIn('run_stuck', self.coord.active_pipes)


class TestIngestPipeResults(unittest.TestCase):
    """Tests for PipeCoordinator.ingest_pipe_results()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_coord_ingest_')
        self.sched = _make_mock_sched(self.tmpdir)
        self.coord = PipeCoordinator(self.sched)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_ingest_completed_task(self):
        task = _make_spec('t_ingest', conformer_index=2)
        pipe = self.coord.submit_pipe_run('run_ingest', [task])
        _complete_task(pipe.pipe_root, 't_ingest')
        with patch('arc.job.pipe.pipe_coordinator.ingest_completed_task') as mock_ingest:
            self.coord.ingest_pipe_results(pipe)
            mock_ingest.assert_called_once()

    def test_ingest_skips_unreadable_state(self):
        """Ingestion continues when a task's state.json is missing."""
        task = _make_spec('t_missing')
        pipe = PipeRun(project_directory=self.tmpdir, run_id='run_missing',
                       tasks=[task], cluster_software='slurm')
        pipe.stage()
        # Remove state.json to simulate corruption
        os.remove(os.path.join(pipe.pipe_root, 'tasks', 't_missing', 'state.json'))
        self.coord.ingest_pipe_results(pipe)  # should not raise

    @patch('arc.job.pipe.pipe_coordinator.job_factory')
    def test_failed_ess_ejects_to_scheduler_troubleshooting_with_parser_summary(self, mock_job_factory):
        """FAILED_ESS tasks with parser_summary should be troubleshooted immediately, not blindly rerun."""
        task = _make_spec('t_ess', task_family='species_sp', engine='orca', cores=12, mem=37888)
        pipe = self.coord.submit_pipe_run('run_ess', [task])
        now = time.time()
        update_task_state(pipe.pipe_root, 't_ess', new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='tok',
                          claimed_at=now, lease_expires_at=now + 300)
        update_task_state(pipe.pipe_root, 't_ess', new_status=TaskState.RUNNING, started_at=now)
        update_task_state(pipe.pipe_root, 't_ess', new_status=TaskState.FAILED_ESS,
                          ended_at=now + 1, failure_class='ess_error')
        result_path = os.path.join(pipe.pipe_root, 'tasks', 't_ess', 'attempts', '0', 'result.json')
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump({'parser_summary': {'status': 'errored',
                                          'keywords': ['MDCI', 'Memory', 'max_total_job_memory'],
                                          'error': 'Orca suggests to increase per cpu core memory to 10218 MB.',
                                          'line': 'Please increase MaxCore'}}, f)

        fake_job = MagicMock()
        mock_job_factory.return_value = fake_job

        self.coord.ingest_pipe_results(pipe)

        self.coord.sched.troubleshoot_ess.assert_called_once()
        self.coord.sched.run_job.assert_not_called()
        kwargs = self.coord.sched.troubleshoot_ess.call_args.kwargs
        self.assertEqual(kwargs['label'], 'H2O')
        self.assertIs(kwargs['job'], fake_job)
        self.assertEqual(kwargs['conformer'], 0)
        mock_job_factory.assert_called_once()
        factory_kwargs = mock_job_factory.call_args.kwargs
        self.assertEqual(factory_kwargs['cpu_cores'], 12)
        self.assertAlmostEqual(factory_kwargs['job_memory_gb'], 37.0)

    def test_failed_ess_without_parser_summary_preserves_resources_on_rerun(self):
        """FAILED_ESS tasks without parser_summary fall back to Scheduler.run_job with original resources."""
        task = _make_spec('t_ess_fallback', task_family='species_sp', engine='orca', cores=10, mem=20480)
        pipe = self.coord.submit_pipe_run('run_ess_fallback', [task])
        now = time.time()
        update_task_state(pipe.pipe_root, 't_ess_fallback', new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='tok',
                          claimed_at=now, lease_expires_at=now + 300)
        update_task_state(pipe.pipe_root, 't_ess_fallback', new_status=TaskState.RUNNING, started_at=now)
        update_task_state(pipe.pipe_root, 't_ess_fallback', new_status=TaskState.FAILED_ESS,
                          ended_at=now + 1, failure_class='ess_error')

        self.coord.ingest_pipe_results(pipe)

        self.coord.sched.troubleshoot_ess.assert_not_called()
        self.coord.sched.run_job.assert_called_once()
        kwargs = self.coord.sched.run_job.call_args.kwargs
        self.assertEqual(kwargs['cpu_cores'], 10)
        self.assertAlmostEqual(kwargs['memory'], 20.0)


class TestFailedTerminalSalvageAndEject(unittest.TestCase):
    """Tests for FAILED_TERMINAL salvage from on-disk result.json and eject fallback."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_coord_failed_term_')
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.sched = _make_mock_sched(self.tmpdir)
        self.coord = PipeCoordinator(self.sched)

    def _set_failed_terminal(self, pipe_root, task_id):
        now = time.time()
        update_task_state(pipe_root, task_id, new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='tok',
                          claimed_at=now, lease_expires_at=now + 300)
        update_task_state(pipe_root, task_id, new_status=TaskState.RUNNING,
                          started_at=now)
        update_task_state(pipe_root, task_id, new_status=TaskState.FAILED_TERMINAL,
                          ended_at=now + 1)

    def _write_attempt_result(self, pipe_root, task_id, attempt_index, status):
        attempt_dir = os.path.join(pipe_root, 'tasks', task_id, 'attempts', str(attempt_index))
        os.makedirs(attempt_dir, exist_ok=True)
        result_path = os.path.join(attempt_dir, 'result.json')
        with open(result_path, 'w') as f:
            json.dump({'status': status}, f)

    def test_failed_terminal_salvages_completed_result(self):
        """A FAILED_TERMINAL task with a COMPLETED result.json is salvaged, not ejected."""
        task = _make_spec('t_salvage', conformer_index=1)
        pipe = self.coord.submit_pipe_run('run_salvage', [task])
        self._set_failed_terminal(pipe.pipe_root, 't_salvage')
        self._write_attempt_result(pipe.pipe_root, 't_salvage', 0, 'COMPLETED')

        with patch('arc.job.pipe.pipe_coordinator.ingest_completed_task') as mock_ingest:
            self.coord.ingest_pipe_results(pipe)

        mock_ingest.assert_called_once()
        self.coord.sched.troubleshoot_ess.assert_not_called()
        self.coord.sched.run_job.assert_not_called()

    def test_failed_terminal_without_salvageable_result_ejects(self):
        """A FAILED_TERMINAL task with no salvageable result.json is ejected to the Scheduler."""
        task = _make_spec('t_no_salvage', conformer_index=0)
        pipe = self.coord.submit_pipe_run('run_no_salvage', [task])
        self._set_failed_terminal(pipe.pipe_root, 't_no_salvage')

        with patch.object(self.coord, '_eject_to_scheduler') as mock_eject, \
                patch('arc.job.pipe.pipe_coordinator.ingest_completed_task') as mock_ingest:
            self.coord.ingest_pipe_results(pipe)

        mock_eject.assert_called_once()
        mock_ingest.assert_not_called()

    def test_failed_terminal_with_only_running_result_does_not_salvage(self):
        """A result.json with status != COMPLETED must not be salvaged."""
        task = _make_spec('t_mixed', conformer_index=0)
        pipe = self.coord.submit_pipe_run('run_mixed', [task])
        self._set_failed_terminal(pipe.pipe_root, 't_mixed')
        self._write_attempt_result(pipe.pipe_root, 't_mixed', 0, 'RUNNING')

        with patch.object(self.coord, '_eject_to_scheduler') as mock_eject, \
                patch('arc.job.pipe.pipe_coordinator.ingest_completed_task') as mock_ingest:
            self.coord.ingest_pipe_results(pipe)

        mock_eject.assert_called_once()
        mock_ingest.assert_not_called()

    def test_failed_terminal_eject_marks_pipe_queue_as_attempted(self):
        """Ejecting a FAILED_TERMINAL task should pre-populate attempted_queues with the pipe queue."""
        task = _make_spec('t_q', task_family='rotor_scan_1d')
        # rotor_scan_1d requires rotor_index in ingestion_metadata.
        task.ingestion_metadata = {'rotor_index': 0}
        task.input_payload = {'torsions': [[0, 1, 2, 3]]}
        pipe = self.coord.submit_pipe_run('run_q', [task])
        pipe.submit_queue = 'short_q'
        self._set_failed_terminal(pipe.pipe_root, 't_q')

        self.coord.ingest_pipe_results(pipe)

        self.coord.sched.run_job.assert_called_once()
        kwargs = self.coord.sched.run_job.call_args.kwargs
        self.assertEqual(kwargs.get('attempted_queues'), ['short_q'])


class TestPostIngestRedispatch(unittest.TestCase):
    """Tests for the per-family _post_ingest_* re-dispatch hooks."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_coord_post_ingest_')
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.sched = _make_mock_sched(self.tmpdir)
        self.coord = PipeCoordinator(self.sched)

    def _make_rotor_scan_pipe(self):
        task1 = _make_spec('t_r0', task_family='rotor_scan_1d')
        task1.ingestion_metadata = {'rotor_index': 0}
        task1.input_payload = {'torsions': [[0, 1, 2, 3]]}
        task2 = _make_spec('t_r1', task_family='rotor_scan_1d')
        task2.ingestion_metadata = {'rotor_index': 1}
        task2.input_payload = {'torsions': [[1, 2, 3, 4]]}
        pipe = self.coord.submit_pipe_run('run_rs', [task1, task2])
        return pipe

    def test_post_ingest_rotor_scan_1d_redispatches_only_missing(self):
        """If some rotor scans landed but others didn't, re-dispatch only what's missing."""
        pipe = self._make_rotor_scan_pipe()
        self.sched.species_dict['H2O'].rotors_dict = {
            0: {'scan_path': '/tmp/scan0.log', 'success': True, 'torsion': [0, 1, 2, 3]},
            1: {'scan_path': '', 'success': None, 'torsion': [1, 2, 3, 4]},
        }
        self.coord._post_ingest_rotor_scan_1d(pipe, 'H2O')
        self.sched.run_scan_jobs.assert_called_once_with('H2O')

    def test_post_ingest_rotor_scan_1d_no_redispatch_when_all_present(self):
        pipe = self._make_rotor_scan_pipe()
        self.sched.species_dict['H2O'].rotors_dict = {
            0: {'scan_path': '/tmp/scan0.log', 'success': True, 'torsion': [0, 1, 2, 3]},
            1: {'scan_path': '/tmp/scan1.log', 'success': True, 'torsion': [1, 2, 3, 4]},
        }
        self.coord._post_ingest_rotor_scan_1d(pipe, 'H2O')
        self.sched.run_scan_jobs.assert_not_called()

    def test_post_ingest_pipe_run_dispatches_to_rotor_scan_handler(self):
        """_post_ingest_pipe_run should route rotor_scan_1d tasks to the rotor handler."""
        pipe = self._make_rotor_scan_pipe()
        self.sched.species_dict['H2O'].rotors_dict = {
            0: {'scan_path': '', 'success': None, 'torsion': [0, 1, 2, 3]},
        }
        with patch.object(self.coord, '_post_ingest_rotor_scan_1d') as mock_rs:
            self.coord._post_ingest_pipe_run(pipe)
        mock_rs.assert_called_once_with(pipe, 'H2O')

    def test_post_ingest_species_freq_redispatches_when_missing(self):
        task = _make_spec('t_freq', task_family='species_freq')
        task.ingestion_metadata = {}
        pipe = self.coord.submit_pipe_run('run_freq', [task])
        self.sched.output['H2O'] = {'paths': {}}
        self.coord._post_ingest_species_freq('H2O')
        self.sched.run_freq_job.assert_called_once_with('H2O')

    def test_post_ingest_species_sp_redispatches_when_missing(self):
        spc = self.sched.species_dict['H2O']
        spc.e_elect = None
        self.coord._post_ingest_species_sp('H2O')
        self.sched.run_sp_job.assert_called_once_with(label='H2O')


class TestComputePipeRoot(unittest.TestCase):
    """Tests for PipeCoordinator._compute_pipe_root()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_root_test_')
        self.sched = _make_mock_sched(self.tmpdir)
        # Add a TS species.
        ts_spc = MagicMock()
        ts_spc.is_ts = True
        self.sched.species_dict['TS0'] = ts_spc
        self.sched.species_dict['H2O'].is_ts = False
        self.coord = PipeCoordinator(self.sched)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_ts_species_path(self):
        tasks = [TaskSpec(task_id='t0', task_family='ts_opt', owner_type='species',
                          owner_key='TS0', input_fingerprint='fp', engine='gaussian',
                          level={'method': 'm'}, required_cores=1, required_memory_mb=1024,
                          input_payload={}, ingestion_metadata={})]
        result = self.coord._compute_pipe_root('TS0_ts_opt', tasks)
        self.assertIn(os.path.join('calcs', 'TSs', 'TS0', 'pipe_ts_opt_0'), result)

    def test_non_ts_species_path(self):
        tasks = [_make_spec('t0', task_family='conf_opt', species_label='H2O')]
        result = self.coord._compute_pipe_root('H2O_conf_opt', tasks)
        self.assertIn(os.path.join('calcs', 'Species', 'H2O', 'pipe_conf_opt_0'), result)

    def test_cross_species_batch(self):
        t1 = _make_spec('t0', species_label='H2O')
        t2 = TaskSpec(task_id='t1', task_family='conf_opt', owner_type='species',
                      owner_key='CH4', input_fingerprint='fp', engine='mockter',
                      level={'method': 'm'}, required_cores=1, required_memory_mb=1024,
                      input_payload={}, ingestion_metadata={})
        result = self.coord._compute_pipe_root('species_sp_batch', [t1, t2])
        self.assertIn(os.path.join('calcs', 'batches', 'pipe_species_sp_batch_0'), result)

    def test_auto_increment(self):
        tasks = [_make_spec('t0', task_family='conf_opt', species_label='H2O')]
        # Create existing pipe_conf_opt_0 directory.
        existing = os.path.join(self.tmpdir, 'calcs', 'Species', 'H2O', 'pipe_conf_opt_0')
        os.makedirs(existing)
        result = self.coord._compute_pipe_root('H2O_conf_opt', tasks)
        self.assertIn('pipe_conf_opt_1', result)


class TestNextIndexedDir(unittest.TestCase):
    """Tests for PipeCoordinator._next_indexed_dir()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_idx_test_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_nonexistent_parent(self):
        result = PipeCoordinator._next_indexed_dir('/nonexistent/path', 'pipe_opt')
        self.assertTrue(result.endswith('pipe_opt_0'))

    def test_empty_parent(self):
        result = PipeCoordinator._next_indexed_dir(self.tmpdir, 'pipe_opt')
        self.assertTrue(result.endswith('pipe_opt_0'))

    def test_increments_past_existing(self):
        os.makedirs(os.path.join(self.tmpdir, 'pipe_opt_0'))
        os.makedirs(os.path.join(self.tmpdir, 'pipe_opt_1'))
        result = PipeCoordinator._next_indexed_dir(self.tmpdir, 'pipe_opt')
        self.assertTrue(result.endswith('pipe_opt_2'))

    def test_ignores_non_matching(self):
        os.makedirs(os.path.join(self.tmpdir, 'pipe_opt_0'))
        os.makedirs(os.path.join(self.tmpdir, 'other_dir'))
        # Create a file (not a directory) with matching prefix.
        with open(os.path.join(self.tmpdir, 'pipe_opt_5'), 'w') as f:
            f.write('not a dir')
        result = PipeCoordinator._next_indexed_dir(self.tmpdir, 'pipe_opt')
        self.assertTrue(result.endswith('pipe_opt_1'))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
