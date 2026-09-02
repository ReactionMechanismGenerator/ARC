#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.pipe.pipe_coordinator module
"""

import functools
import json
import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from arc.job.pipe.pipe_coordinator import PipeCoordinator
from arc.job.pipe.pipe_run import PipeRun, get_task_attempt_dir
from arc.job.pipe.pipe_state import (
    PipeRunState,
    TaskState,
    TaskSpec,
    update_task_state,
)
from arc.level import Level
from arc.scheduler import Scheduler
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


def _make_mock_sched(project_directory, ess_settings=None):
    """Create a mock Scheduler with the attributes PipeCoordinator needs."""
    sched = MagicMock()
    sched.project = 'pipe_test_project'
    sched.project_directory = project_directory
    sched.ess_settings = {'orca': ['local'], 'mockter': ['local']}
    sched.testing = True
    sched.server_job_ids = list()
    sched.completed_job_records = list()
    sched.ess_settings = ess_settings if ess_settings is not None else {'mockter': ['local']}
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

    @patch('arc.job.pipe.pipe_coordinator.settings',
           {'servers': {'zeus': {'cluster_soft': 'PBS', 'address': 'z.example.edu', 'un': 'u'}}})
    def test_false_when_engine_resolves_to_remote_server(self):
        coord = PipeCoordinator(_make_mock_sched(self.tmpdir, ess_settings={'mockter': ['zeus']}))
        tasks = [_make_spec(f't_{i}') for i in range(15)]
        self.assertFalse(coord.should_use_pipe(tasks))

    def _should_use_pipe(self, ess_settings, tasks=None):
        """Return should_use_pipe() for a scheduler whose ESS settings are ``ess_settings``."""
        coord = PipeCoordinator(_make_mock_sched(self.tmpdir, ess_settings=ess_settings))
        return coord.should_use_pipe(tasks if tasks is not None
                                     else [_make_spec(f't_{i}') for i in range(15)])

    def test_true_when_the_engine_is_not_an_ess(self):
        """TS-guess methods run in this process and are given no server, which is this machine."""
        self.assertTrue(self._should_use_pipe({'gaussian': ['local']}))

    def test_false_when_the_engine_names_an_empty_server_list(self):
        """An ESS available nowhere resolves to nothing, and used to raise IndexError."""
        self.assertFalse(self._should_use_pipe({'mockter': []}))

    @patch('arc.job.pipe.pipe_coordinator.settings',
           {'servers': {'zeus': {'cluster_soft': 'PBS', 'address': 'z.example.edu', 'un': 'u'}}})
    def test_false_when_the_resolved_server_is_not_configured(self):
        """An unconfigured server is not known to be this machine, so the pipe is refused."""
        self.assertFalse(self._should_use_pipe({'mockter': ['not_a_configured_server']}))

    @patch('arc.job.pipe.pipe_coordinator.settings',
           {'servers': {'zeus': {'cluster_soft': 'PBS', 'address': 'z.example.edu', 'un': 'u'},
                        'local': {'cluster_soft': 'PBS', 'un': 'u'}}})
    def test_the_first_server_decides_even_when_a_later_one_is_local(self):
        """ARC submits to the first server named, so that is the one the pipe must be judged on."""
        self.assertFalse(self._should_use_pipe({'mockter': ['zeus', 'local']}))

    @patch('arc.job.pipe.pipe_coordinator.settings',
           {'servers': {'local': {'cluster_soft': 'PBS', 'un': 'u'}}})
    def test_true_when_the_server_is_local_in_another_case(self):
        """A server name is a settings key whose casing the user chose."""
        self.assertTrue(self._should_use_pipe({'mockter': ['LOCAL']}))

    @patch('arc.job.pipe.pipe_coordinator.settings',
           {'servers': {'local': {'cluster_soft': 'PBS', 'un': 'u'}}})
    def test_true_when_the_server_is_named_as_a_bare_string(self):
        """The ESS settings allow one server as a string rather than a one-item list."""
        self.assertTrue(self._should_use_pipe({'mockter': 'local'}))

    @patch('arc.job.pipe.pipe_coordinator.settings',
           {'servers': {'zeus': {'cluster_soft': 'PBS', 'address': 'z.example.edu', 'un': 'u'},
                        'local': {'cluster_soft': 'PBS', 'un': 'u'}}})
    def test_a_troubleshooting_server_override_is_honoured(self):
        """A job moved to another server by troubleshooting does not go through the pipe."""
        tasks = [_make_spec(f't_{i}') for i in range(15)]
        for task in tasks:
            task.args = {'trsh': {'server': 'zeus'}}
        self.assertFalse(self._should_use_pipe({'mockter': ['local']}, tasks=tasks))


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

        def fake_reconcile(scheduler_job_alive=True):
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
        pipe.submitted_at = None
        pipe.scheduler_job_id = '12345[]'
        self.assertTrue(PipeCoordinator._is_scheduler_job_alive(pipe, None))

    def test_none_scheduler_job_id_returns_true(self):
        pipe = MagicMock()
        pipe.submitted_at = None
        pipe.scheduler_job_id = None
        self.assertTrue(PipeCoordinator._is_scheduler_job_alive(pipe, ['12345[0]']))

    def test_pbs_array_element_in_queue(self):
        pipe = MagicMock()
        pipe.submitted_at = None
        pipe.scheduler_job_id = '4018898[]'
        self.assertTrue(PipeCoordinator._is_scheduler_job_alive(pipe, ['4018898[2]', '9999']))

    def test_pbs_array_not_in_queue(self):
        pipe = MagicMock()
        pipe.submitted_at = None
        pipe.scheduler_job_id = '4018898[]'
        self.assertFalse(PipeCoordinator._is_scheduler_job_alive(pipe, ['9999', '5555']))

    def test_non_array_job_in_queue(self):
        pipe = MagicMock()
        pipe.submitted_at = None
        pipe.scheduler_job_id = '12345'
        self.assertTrue(PipeCoordinator._is_scheduler_job_alive(pipe, ['12345', '9999']))

    def test_non_array_job_not_in_queue(self):
        pipe = MagicMock()
        pipe.submitted_at = None
        pipe.scheduler_job_id = '12345'
        self.assertFalse(PipeCoordinator._is_scheduler_job_alive(pipe, ['9999']))

    def test_empty_queue(self):
        pipe = MagicMock()
        pipe.submitted_at = None
        pipe.scheduler_job_id = '12345[]'
        self.assertFalse(PipeCoordinator._is_scheduler_job_alive(pipe, []))

    def test_slurm_array_element_in_queue(self):
        pipe = MagicMock()
        pipe.submitted_at = None
        pipe.scheduler_job_id = '12345'
        self.assertTrue(PipeCoordinator._is_scheduler_job_alive(pipe, ['12345_7', '9999']))

    def test_slurm_array_not_in_queue(self):
        pipe = MagicMock()
        pipe.submitted_at = None
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

    def test_ingest_records_task_cost(self):
        """A terminal task yields one cost record shaped like Scheduler job records."""
        task = _make_spec('t_cost', conformer_index=1)
        pipe = self.coord.submit_pipe_run('run_cost', [task])
        _complete_task(pipe.pipe_root, 't_cost')
        with patch('arc.job.pipe.pipe_coordinator.ingest_completed_task'):
            self.coord.ingest_pipe_results(pipe)
        self.assertEqual(len(self.sched.completed_job_records), 1)
        record = self.sched.completed_job_records[0]
        self.assertEqual(record['job_name'], 'pipe_run_cost/t_cost')
        self.assertEqual(record['label'], 'H2O')
        self.assertEqual(record['job_type'], 'conf_opt')
        self.assertEqual(record['job_adapter'], 'mockter')
        self.assertEqual(record['server'], 'pipe')
        self.assertEqual(record['cpu_cores'], 4)
        self.assertIsNotNone(record['run_time_sec'])
        self.assertGreaterEqual(record['run_time_sec'], 0.0)
        self.assertEqual(record['job_status'], TaskState.COMPLETED.value)

    def test_ingest_cost_dedup_on_reingest(self):
        """Re-ingesting the same pipe run (e.g., after a restart) must not double-count."""
        task = _make_spec('t_dedup')
        pipe = self.coord.submit_pipe_run('run_dedup', [task])
        _complete_task(pipe.pipe_root, 't_dedup')
        with patch('arc.job.pipe.pipe_coordinator.ingest_completed_task'):
            self.coord.ingest_pipe_results(pipe)
            self.coord.ingest_pipe_results(pipe)
        self.assertEqual(len(self.sched.completed_job_records), 1)

    def test_ingest_cost_skips_non_terminal_and_handles_missing_timing(self):
        """Non-terminal tasks are not recorded; cancelled tasks are, with a None run time."""
        now = time.time()
        pending_task = _make_spec('t_pending')
        cancelled_task = _make_spec('t_cancelled')
        pipe = self.coord.submit_pipe_run('run_mixed', [pending_task, cancelled_task])
        update_task_state(pipe.pipe_root, 't_cancelled', new_status=TaskState.CANCELLED,
                          ended_at=now)
        self.coord.ingest_pipe_results(pipe)
        self.assertEqual(len(self.sched.completed_job_records), 1)
        record = self.sched.completed_job_records[0]
        self.assertEqual(record['job_name'], 'pipe_run_mixed/t_cancelled')
        self.assertIsNone(record['run_time_sec'])
        self.assertEqual(record['job_status'], TaskState.CANCELLED.value)

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


class TestFinalizeSpeciesLeafTask(unittest.TestCase):
    """
    Tests that piped species_sp / species_freq tasks are routed through the Scheduler's own
    post-job checks, which own the ``output[label]['job_types']`` success flags. Without this,
    a piped job computes correctly and is ingested, yet the species is reported as failed.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_coord_finalize_')
        self.sched = _make_mock_sched(self.tmpdir)
        self.coord = PipeCoordinator(self.sched)
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        convergence_patch = patch('arc.job.pipe.pipe_coordinator.check_ess_convergence',
                                  return_value=True)
        self.mock_convergence = convergence_patch.start()
        self.addCleanup(convergence_patch.stop)

    def _stage_completed(self, task_id, task_family):
        """Stage a single-task pipe run, complete it, and give it a canonical output file."""
        task = _make_spec(task_id, task_family=task_family)
        pipe = self.coord.submit_pipe_run(f'run_{task_id}', [task])
        _complete_task(pipe.pipe_root, task_id)
        attempt_dir = get_task_attempt_dir(pipe.pipe_root, task_id, 0)
        os.makedirs(attempt_dir, exist_ok=True)
        output_file = os.path.join(attempt_dir, 'output.out')
        with open(output_file, 'w') as f:
            f.write('mock output\n')
        with open(os.path.join(attempt_dir, 'result.json'), 'w') as f:
            json.dump({'canonical_output_path': output_file}, f)
        return pipe, output_file

    def test_species_sp_is_finalized(self):
        """A completed species_sp task runs post_sp_actions, which sets job_types['sp']."""
        pipe, output_file = self._stage_completed('t_sp', 'species_sp')
        self.coord.ingest_pipe_results(pipe)
        self.sched.post_sp_actions.assert_called_once()
        kwargs = self.sched.post_sp_actions.call_args.kwargs
        self.assertEqual(kwargs['label'], 'H2O')
        self.assertEqual(kwargs['sp_path'], output_file)

    def test_species_freq_is_finalized(self):
        """A completed species_freq task runs post_freq_actions, which sets job_types['freq']."""
        pipe, output_file = self._stage_completed('t_freq', 'species_freq')
        with patch('arc.job.pipe.pipe_coordinator.parser.parse_frequencies',
                   return_value=[1500.0, 3000.0]):
            self.coord.ingest_pipe_results(pipe)
        self.sched.post_freq_actions.assert_called_once()
        kwargs = self.sched.post_freq_actions.call_args.kwargs
        self.assertEqual(kwargs['label'], 'H2O')
        self.assertEqual(kwargs['vibfreqs'], [1500.0, 3000.0])
        self.assertEqual(kwargs['job'].local_path_to_output_file, output_file)

    def test_other_families_are_not_finalized(self):
        """conf_opt has its own post-ingestion handler and must not be finalized here."""
        pipe, _ = self._stage_completed('t_conf', 'conf_opt')
        self.coord.ingest_pipe_results(pipe)
        self.sched.post_sp_actions.assert_not_called()
        self.sched.post_freq_actions.assert_not_called()

    def test_unknown_species_is_skipped(self):
        """An owner_key absent from species_dict is skipped rather than raising."""
        pipe, _ = self._stage_completed('t_unknown', 'species_sp')
        self.sched.species_dict.pop('H2O')
        self.coord.ingest_pipe_results(pipe)
        self.sched.post_sp_actions.assert_not_called()

    def test_non_converged_ess_is_not_finalized(self):
        """A task can be COMPLETED yet hold a non-converged output; it must not be flagged."""
        pipe, _ = self._stage_completed('t_unconverged', 'species_sp')
        self.mock_convergence.return_value = False
        self.coord.ingest_pipe_results(pipe)
        self.sched.post_sp_actions.assert_not_called()

    def test_missing_output_file_is_skipped(self):
        """A completed task with no locatable output does not reach the Scheduler checks."""
        task = _make_spec('t_no_out', task_family='species_sp')
        pipe = self.coord.submit_pipe_run('run_no_out', [task])
        _complete_task(pipe.pipe_root, 't_no_out')
        self.coord.ingest_pipe_results(pipe)
        self.sched.post_sp_actions.assert_not_called()


class _RealMethodSchedulerStub:
    """
    A Scheduler stand-in that runs the Scheduler's real post-job methods over stub state.

    Every attribute that is not stub state falls through to the Scheduler class, so whichever
    post-job method the coordinator reaches for is the production implementation, bound to this
    object. Tests can therefore observe what a piped task actually leaves behind, rather than
    only that some method was called on a mock.
    """

    def __init__(self, project_directory, species):
        self.project_directory = project_directory
        self.server_job_ids = list()
        self.species_dict = {species.label: species}
        self.output = {species.label: {'paths': {'sp': '', 'composite': '', 'freq': ''},
                                       'job_types': dict(),
                                       'warnings': '',
                                       'info': ''}}
        self.testing = True
        self.trsh_ess_jobs = False
        self.skip_nmd = False
        self.rxn_dict = dict()
        self.rxn_list = list()
        self.freq_level = Level(repr='wb97xd/def2tzvp')

    def __getattr__(self, name):
        return functools.partial(getattr(Scheduler, name), self)


class TestPipedFreqArtifacts(unittest.TestCase):
    """
    Tests that a piped species_freq task leaves behind everything a converged freq job is
    expected to leave behind, not merely the ``job_types['freq']`` success flag.

    Marking a species converged while ``species.freqs`` is unset and no ``freq.out`` sits in its
    output folder would trade a loud failure for a quiet one: ``compute_rxn_e0()`` abandons the
    reaction E0 check when that file is missing for any participant, and the frequencies would
    never reach the restart or output files.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_coord_freq_artifacts_')
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.species = ARCSpecies(label='H2O', smiles='O')
        self.sched = _RealMethodSchedulerStub(self.tmpdir, self.species)
        self.coord = PipeCoordinator(self.sched)
        self.freq_out = os.path.join(self.tmpdir, 'output', 'Species', 'H2O', 'geometry', 'freq.out')
        convergence_patch = patch('arc.job.pipe.pipe_coordinator.check_ess_convergence', return_value=True)
        convergence_patch.start()
        self.addCleanup(convergence_patch.stop)
        ingest_patch = patch('arc.job.pipe.pipe_coordinator.ingest_completed_task')
        ingest_patch.start()
        self.addCleanup(ingest_patch.stop)

    def _ingest_freq_task(self, task_id, vibfreqs):
        """Stage, complete and ingest a single species_freq task yielding ``vibfreqs``."""
        task = _make_spec(task_id, task_family='species_freq')
        pipe = self.coord.submit_pipe_run(f'run_{task_id}', [task])
        _complete_task(pipe.pipe_root, task_id)
        attempt_dir = get_task_attempt_dir(pipe.pipe_root, task_id, 0)
        os.makedirs(attempt_dir, exist_ok=True)
        output_file = os.path.join(attempt_dir, 'output.out')
        with open(output_file, 'w') as f:
            f.write('mock freq output\n')
        with open(os.path.join(attempt_dir, 'result.json'), 'w') as f:
            json.dump({'canonical_output_path': output_file}, f)
        with patch('arc.job.pipe.pipe_coordinator.parser.parse_frequencies', return_value=vibfreqs):
            self.coord.ingest_pipe_results(pipe)

    def test_converged_freq_sets_freqs_and_writes_freq_out(self):
        """A piped freq task that passes the check must populate species.freqs and freq.out."""
        self._ingest_freq_task('t_freq_ok', [1500.0, 3000.0, 3700.0])
        self.assertTrue(self.sched.output['H2O']['job_types']['freq'])
        self.assertEqual(self.species.freqs, [1500.0, 3000.0, 3700.0])
        self.assertTrue(os.path.isfile(self.freq_out),
                        f'Expected the freq output to be copied to {self.freq_out}')
        with open(self.freq_out, 'r') as f:
            self.assertIn('mock freq output', f.read())

    def test_imaginary_freq_leaves_no_artifacts(self):
        """
        A stable species carrying an imaginary frequency must fail the check, and must not be
        credited with any of the artifacts a converged freq job produces.
        """
        self._ingest_freq_task('t_freq_imag', [-800.0, 1500.0, 3000.0])
        self.assertFalse(self.sched.output['H2O']['job_types'].get('freq', False))
        self.assertIsNone(self.species.freqs)
        self.assertFalse(os.path.isfile(self.freq_out))


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
