#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.pipe.pipe_coordinator module
"""

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
    sched.project_directory = project_directory
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


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
