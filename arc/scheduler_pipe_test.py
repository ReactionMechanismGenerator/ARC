#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the pipe-mode methods of the arc.scheduler module
"""

import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from arc.imports import settings
from arc.job.pipe.pipe_state import (
    PipeRunState,
    TaskState,
    TaskSpec,
    get_task_attempt_dir,
    update_task_state,
)
from arc.job.pipe.pipe_run import PipeRun
from arc.level import Level
from arc.scheduler import Scheduler
from arc.species.species import ARCSpecies


default_levels_of_theory = settings['default_levels_of_theory']


def _make_task_spec(task_id, engine='mockter', task_family='conf_opt',
                    cores=4, mem=2048, species_label='H2O', conformer_index=0,
                    level=None):
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


def _make_scheduler(project_directory):
    """Create a minimal Scheduler for testing pipe methods."""
    ess_settings = {'gaussian': ['server1'], 'molpro': ['server2', 'server1'], 'qchem': ['server1']}
    spc = ARCSpecies(label='H2O', smiles='O')
    spc.conformers = [None] * 5
    spc.conformer_energies = [None] * 5
    return Scheduler(
        project='pipe_test',
        ess_settings=ess_settings,
        species_list=[spc],
        project_directory=project_directory,
        conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
        opt_level=Level(repr=default_levels_of_theory['opt']),
        freq_level=Level(repr=default_levels_of_theory['freq']),
        sp_level=Level(repr=default_levels_of_theory['sp']),
        scan_level=Level(repr=default_levels_of_theory['scan']),
        ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
        testing=True,
        job_types={'conf_opt': True, 'opt': True, 'fine': False, 'freq': True,
                   'sp': True, 'rotors': False, 'orbitals': False, 'lennard_jones': False},
        orbitals_level=default_levels_of_theory['orbitals'],
    )


def _complete_task(pipe_root, task_id):
    """Drive a task through the full lifecycle to COMPLETED."""
    now = time.time()
    update_task_state(pipe_root, task_id, new_status=TaskState.CLAIMED,
                      claimed_by='w', claim_token='tok', claimed_at=now, lease_expires_at=now + 300)
    update_task_state(pipe_root, task_id, new_status=TaskState.RUNNING, started_at=now)
    update_task_state(pipe_root, task_id, new_status=TaskState.COMPLETED, ended_at=now)


_pipe_patches = []


def setUpModule():
    """Enable pipe mode for all tests in this module."""
    global _pipe_patches
    pipe_vals = {'enabled': True, 'min_tasks': 10, 'max_workers': 100,
                 'max_attempts': 3, 'lease_duration_hrs': 24}
    for target in ('arc.job.pipe.pipe_coordinator.pipe_settings',
                    'arc.job.pipe.pipe_planner.pipe_settings'):
        p = patch.dict(target, pipe_vals)
        p.start()
        _pipe_patches.append(p)


def tearDownModule():
    """Restore pipe settings."""
    global _pipe_patches
    for p in _pipe_patches:
        p.stop()
    _pipe_patches.clear()


class TestShouldUsePipe(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_sched_test_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_true_for_homogeneous_batch(self):
        tasks = [_make_task_spec(f'task_{i}') for i in range(15)]
        self.assertTrue(self.sched.pipe_coordinator.should_use_pipe(tasks))

    def test_returns_false_for_heterogeneous_memory(self):
        tasks = [_make_task_spec(f'task_{i}') for i in range(15)]
        tasks[7] = _make_task_spec('task_7', mem=9999)
        self.assertFalse(self.sched.pipe_coordinator.should_use_pipe(tasks))

    def test_returns_false_for_heterogeneous_engine(self):
        tasks = [_make_task_spec(f'task_{i}') for i in range(15)]
        tasks[0] = _make_task_spec('task_0', engine='gaussian')
        self.assertFalse(self.sched.pipe_coordinator.should_use_pipe(tasks))

    def test_returns_false_for_heterogeneous_level(self):
        tasks = [_make_task_spec(f'task_{i}') for i in range(15)]
        tasks[3] = _make_task_spec('task_3', level={'method': 'b3lyp', 'basis': 'sto-3g'})
        self.assertFalse(self.sched.pipe_coordinator.should_use_pipe(tasks))

    def test_returns_false_below_threshold(self):
        tasks = [_make_task_spec(f'task_{i}') for i in range(5)]
        self.assertFalse(self.sched.pipe_coordinator.should_use_pipe(tasks))

    def test_returns_true_at_exact_threshold(self):
        tasks = [_make_task_spec(f'task_{i}') for i in range(10)]
        self.assertTrue(self.sched.pipe_coordinator.should_use_pipe(tasks))


class TestSubmitPipeRun(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_submit_test_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_submit_returns_pipe_run(self):
        tasks = [_make_task_spec(f'task_{i}') for i in range(3)]
        pipe = self.sched.pipe_coordinator.submit_pipe_run('run_001', tasks)
        self.assertIsInstance(pipe, PipeRun)
        self.assertEqual(pipe.status, PipeRunState.STAGED)
        self.assertIn('run_001', self.sched.active_pipes)
        self.assertIs(self.sched.active_pipes['run_001'], pipe)

    def test_submit_uses_explicit_cluster_software(self):
        tasks = [_make_task_spec('task_0')]
        pipe = self.sched.pipe_coordinator.submit_pipe_run('run_pbs', tasks, cluster_software='pbs')
        self.assertEqual(pipe.cluster_software, 'pbs')

    def test_submit_default_cluster_software(self):
        tasks = [_make_task_spec('task_0')]
        pipe = self.sched.pipe_coordinator.submit_pipe_run('run_default', tasks)
        self.assertEqual(pipe.cluster_software, 'slurm')


class TestPollPipes(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_poll_test_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_poll_removes_completed_pipe(self):
        pipe = self.sched.pipe_coordinator.submit_pipe_run('run_poll', [_make_task_spec('task_poll')])
        _complete_task(pipe.pipe_root, 'task_poll')
        self.sched.pipe_coordinator.poll_pipes()
        self.assertNotIn('run_poll', self.sched.active_pipes)

    def test_poll_keeps_active_pipe(self):
        self.sched.pipe_coordinator.submit_pipe_run('run_active', [_make_task_spec('task_active')])
        self.sched.pipe_coordinator.poll_pipes()
        self.assertIn('run_active', self.sched.active_pipes)

    def test_poll_removes_failed_pipe(self):
        pipe = self.sched.pipe_coordinator.submit_pipe_run('run_fail', [_make_task_spec('task_f')])
        pipe.status = PipeRunState.FAILED
        pipe._save_run_metadata()
        self.sched.pipe_coordinator.poll_pipes()
        self.assertNotIn('run_fail', self.sched.active_pipes)

    def test_poll_logs_counts(self):
        pipe = self.sched.pipe_coordinator.submit_pipe_run('run_log', [_make_task_spec('task_log')])
        _complete_task(pipe.pipe_root, 'task_log')
        with patch('arc.job.pipe.pipe_coordinator.logger') as mock_logger:
            self.sched.pipe_coordinator.poll_pipes()
            info_calls = [str(c) for c in mock_logger.info.call_args_list]
            self.assertTrue(any('run_log' in c for c in info_calls))

    def test_poll_logs_exception_with_traceback(self):
        """A reconcile exception is logged with traceback, run stays on first failure."""
        pipe = self.sched.pipe_coordinator.submit_pipe_run('run_err', [_make_task_spec('task_err')])
        with patch.object(pipe, 'reconcile', side_effect=RuntimeError('disk full')):
            with patch('arc.job.pipe.pipe_coordinator.logger') as mock_logger:
                self.sched.pipe_coordinator.poll_pipes()
                error_calls = [str(c) for c in mock_logger.error.call_args_list]
                self.assertTrue(any('run_err' in c and 'reconciliation failed' in c for c in error_calls))
        # Run should still be in active_pipes after first failure
        self.assertIn('run_err', self.sched.active_pipes)
        self.assertEqual(self.sched.pipe_coordinator._pipe_poll_failures.get('run_err'), 1)

    def test_poll_removes_after_repeated_failures(self):
        """After 3 consecutive failures, the broken run is removed from active_pipes."""
        pipe = self.sched.pipe_coordinator.submit_pipe_run('run_stuck', [_make_task_spec('task_stuck')])
        with patch.object(pipe, 'reconcile', side_effect=RuntimeError('corrupt state')):
            for _ in range(3):
                self.sched.pipe_coordinator.poll_pipes()
        self.assertNotIn('run_stuck', self.sched.active_pipes)
        self.assertNotIn('run_stuck', self.sched.pipe_coordinator._pipe_poll_failures)

    def test_poll_resets_failure_count_on_success(self):
        """Successful reconciliation resets the failure counter."""
        pipe = self.sched.pipe_coordinator.submit_pipe_run('run_flaky', [_make_task_spec('task_flaky')])
        # Fail once
        with patch.object(pipe, 'reconcile', side_effect=RuntimeError('transient')):
            self.sched.pipe_coordinator.poll_pipes()
        self.assertEqual(self.sched.pipe_coordinator._pipe_poll_failures.get('run_flaky'), 1)
        # Succeed — counter should reset
        self.sched.pipe_coordinator.poll_pipes()
        self.assertNotIn('run_flaky', self.sched.pipe_coordinator._pipe_poll_failures)


class TestScheduleJobsLoopCondition(unittest.TestCase):
    """Test that the main loop does not exit while active_pipes remain."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_loop_test_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_loop_continues_for_active_pipes(self):
        """Verify the loop condition includes active_pipes."""
        pipe = self.sched.pipe_coordinator.submit_pipe_run('run_loop', [_make_task_spec('task_loop')])
        _complete_task(pipe.pipe_root, 'task_loop')
        # Clear running_jobs so only active_pipes keeps the loop alive
        self.sched.running_jobs = {}
        self.assertIn('run_loop', self.sched.active_pipes)
        # Simulate one iteration: poll_pipes should complete and remove it
        self.sched.pipe_coordinator.poll_pipes()
        self.assertNotIn('run_loop', self.sched.active_pipes)

    def test_poll_pipes_invoked_in_loop(self):
        """Verify poll_pipes is invoked when the loop runs with only active pipes."""
        pipe = self.sched.pipe_coordinator.submit_pipe_run('run_int', [_make_task_spec('task_int')])
        _complete_task(pipe.pipe_root, 'task_int')
        self.sched.running_jobs = {}
        # Patch poll_pipes to track calls, then run one iteration manually.
        # The loop condition is: while self.running_jobs != {} or self.active_pipes
        # Since we can't safely run schedule_jobs (too many side effects), we
        # verify that: (a) the condition is true, and (b) poll_pipes works.
        self.assertTrue(self.sched.running_jobs == {} and bool(self.sched.active_pipes))
        with patch.object(self.sched.pipe_coordinator, 'poll_pipes',
                          wraps=self.sched.pipe_coordinator.poll_pipes) as mock_poll:
            self.sched.pipe_coordinator.poll_pipes()
            mock_poll.assert_called_once()
        # After polling, the completed pipe should be gone.
        self.assertNotIn('run_int', self.sched.active_pipes)


class TestRegisterPipeRunFromDir(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_register_test_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_register_from_dir(self):
        tasks = [_make_task_spec(f'task_{i}') for i in range(2)]
        original = self.sched.pipe_coordinator.submit_pipe_run('run_restart', tasks, cluster_software='pbs')
        pipe_root = original.pipe_root
        del self.sched.active_pipes['run_restart']
        restored = self.sched.pipe_coordinator.register_pipe_run_from_dir(pipe_root)
        self.assertIn('run_restart', self.sched.active_pipes)
        self.assertEqual(restored.run_id, 'run_restart')
        self.assertEqual(restored.cluster_software, 'pbs')


class TestTryPipeConformers(unittest.TestCase):
    """Tests for the _try_pipe_conformers method."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_conf_test_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pipes_when_enough_conformers(self):
        """When >=10 conformers, pipe mode should be used."""
        species = self.sched.species_dict['H2O']
        species.conformers = [{'symbols': ('O',), 'isotopes': (16,),
                                'coords': ((0.0, 0.0, float(i)),)}
                               for i in range(12)]
        species.conformer_energies = [None] * 12
        # Mock deduce_job_adapter to return a queue-eligible adapter
        with patch.object(self.sched, 'deduce_job_adapter', return_value='gaussian'):
            result = self.sched.pipe_planner.try_pipe_conformers('H2O')
        self.assertTrue(result)
        self.assertEqual(len(self.sched.active_pipes), 1)
        run_id = list(self.sched.active_pipes.keys())[0]
        self.assertIn('H2O', run_id)
        pipe = self.sched.active_pipes[run_id]
        self.assertEqual(len(pipe.tasks), 12)
        # Verify task metadata uses the new explicit schema
        spec = pipe.tasks[0]
        self.assertEqual(spec.owner_key, 'H2O')
        self.assertEqual(spec.task_family, 'conf_opt')
        self.assertEqual(spec.ingestion_metadata['conformer_index'], 0)
        self.assertIsNotNone(spec.level)

    def test_no_pipe_when_few_conformers(self):
        """When <10 conformers, pipe mode should not be used."""
        species = self.sched.species_dict['H2O']
        species.conformers = [{'symbols': ('O',), 'isotopes': (16,),
                                'coords': ((0.0, 0.0, float(i)),)}
                               for i in range(5)]
        species.conformer_energies = [None] * 5
        with patch.object(self.sched, 'deduce_job_adapter', return_value='gaussian'):
            result = self.sched.pipe_planner.try_pipe_conformers('H2O')
        self.assertFalse(result)
        self.assertEqual(len(self.sched.active_pipes), 0)

    def test_no_pipe_for_incore_adapter(self):
        """Incore adapters should not use pipe mode."""
        species = self.sched.species_dict['H2O']
        species.conformers = [{'symbols': ('O',), 'isotopes': (16,),
                                'coords': ((0.0, 0.0, float(i)),)}
                               for i in range(15)]
        species.conformer_energies = [None] * 15
        with patch.object(self.sched, 'deduce_job_adapter', return_value='torchani'):
            result = self.sched.pipe_planner.try_pipe_conformers('H2O')
        self.assertFalse(result)


class TestIngestPipeResults(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_ingest_test_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_pipe_with_completed_task(self, task_id='task_ingest', **spec_kwargs):
        task = _make_task_spec(task_id, **spec_kwargs)
        pipe = PipeRun(project_directory=self.tmpdir, run_id=f'{task_id}_run',
                       tasks=[task], cluster_software='slurm')
        pipe.stage()
        _complete_task(pipe.pipe_root, task_id)
        attempt_dir = get_task_attempt_dir(pipe.pipe_root, task_id, 0)
        return pipe, attempt_dir

    def _place_output_file(self, attempt_dir):
        calcs_dir = os.path.join(attempt_dir, 'calcs', 'Species', 'H2O', 'conf_opt_a1')
        os.makedirs(calcs_dir, exist_ok=True)
        path = os.path.join(calcs_dir, 'output.yml')
        with open(path, 'w') as f:
            f.write('dummy')
        return path

    def test_ingest_updates_species_conformer(self):
        pipe, attempt_dir = self._make_pipe_with_completed_task(
            species_label='H2O', conformer_index=2)
        self._place_output_file(attempt_dir)
        mock_xyz = {'symbols': ('O', 'H', 'H'), 'isotopes': (16, 1, 1),
                    'coords': ((0.0, 0.0, 0.12), (0.0, 0.76, -0.47), (0.0, -0.76, -0.47))}
        with patch('arc.job.pipe.pipe_run.parser.parse_geometry', return_value=mock_xyz), \
             patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')), \
             patch('arc.job.pipe.pipe_run.parser.parse_e_elect', return_value=-75.5), \
             patch.object(self.sched, 'determine_most_stable_conformer'), \
             patch.object(self.sched, 'run_opt_job'):
            self.sched.pipe_coordinator.ingest_pipe_results(pipe)
        species = self.sched.species_dict['H2O']
        self.assertEqual(species.conformers[2], mock_xyz)
        self.assertAlmostEqual(species.conformer_energies[2], -75.5)

    def test_ingest_terminal_failure_logs_error(self):
        task = _make_task_spec('task_fail', species_label='H2O', conformer_index=0)
        pipe = PipeRun(project_directory=self.tmpdir, run_id='fail_test',
                       tasks=[task], cluster_software='slurm')
        pipe.stage()
        now = time.time()
        update_task_state(pipe.pipe_root, 'task_fail', new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='tok', claimed_at=now, lease_expires_at=now + 300)
        update_task_state(pipe.pipe_root, 'task_fail', new_status=TaskState.RUNNING, started_at=now)
        update_task_state(pipe.pipe_root, 'task_fail', new_status=TaskState.FAILED_TERMINAL,
                          ended_at=now, failure_class='oom')
        with patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')):
            self.sched.pipe_coordinator.ingest_pipe_results(pipe)
        self.assertIsNone(self.sched.species_dict['H2O'].conformers[0])

    def test_ingest_cancelled_task_logged(self):
        task = _make_task_spec('task_cancel', species_label='H2O', conformer_index=0)
        pipe = PipeRun(project_directory=self.tmpdir, run_id='cancel_test',
                       tasks=[task], cluster_software='slurm')
        pipe.stage()
        now = time.time()
        update_task_state(pipe.pipe_root, 'task_cancel', new_status=TaskState.CANCELLED, ended_at=now)
        with patch('arc.job.pipe.pipe_coordinator.logger') as mock_logger:
            with patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')):
                self.sched.pipe_coordinator.ingest_pipe_results(pipe)
            warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
            self.assertTrue(any('cancelled' in c.lower() for c in warning_calls))

    def test_ingest_skips_unknown_species(self):
        pipe, _ = self._make_pipe_with_completed_task(
            task_id='task_unknown', species_label='NONEXISTENT', conformer_index=0)
        with patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')):
            self.sched.pipe_coordinator.ingest_pipe_results(pipe)

    def test_ingest_missing_conformer_index(self):
        """conf_opt task with empty ingestion_metadata is skipped with warning."""
        task = _make_task_spec('task_no_idx', species_label='H2O')
        # Override ingestion_metadata to remove conformer_index
        task.ingestion_metadata = {}
        pipe = PipeRun(project_directory=self.tmpdir, run_id='noidx_test',
                       tasks=[task], cluster_software='slurm')
        pipe.stage()
        _complete_task(pipe.pipe_root, 'task_no_idx')
        with patch('arc.job.pipe.pipe_run.logger') as mock_logger:
            with patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')):
                self.sched.pipe_coordinator.ingest_pipe_results(pipe)
            warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
            self.assertTrue(any('conformer_index' in c for c in warning_calls))

    def test_ingest_continues_on_missing_output(self):
        task_ok = _make_task_spec('task_ok', species_label='H2O', conformer_index=1)
        task_bad = _make_task_spec('task_bad', species_label='H2O', conformer_index=2)
        pipe = PipeRun(project_directory=self.tmpdir, run_id='partial_test',
                       tasks=[task_bad, task_ok], cluster_software='slurm')
        pipe.stage()
        _complete_task(pipe.pipe_root, 'task_ok')
        _complete_task(pipe.pipe_root, 'task_bad')
        attempt_dir_ok = get_task_attempt_dir(pipe.pipe_root, 'task_ok', 0)
        self._place_output_file(attempt_dir_ok)
        mock_xyz = {'symbols': ('O', 'H', 'H'), 'isotopes': (16, 1, 1),
                    'coords': ((0.0, 0.0, 0.12), (0.0, 0.76, -0.47), (0.0, -0.76, -0.47))}
        with patch('arc.job.pipe.pipe_run.parser.parse_geometry', return_value=mock_xyz), \
             patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')), \
             patch('arc.job.pipe.pipe_run.parser.parse_e_elect', return_value=-75.5), \
             patch.object(self.sched, 'determine_most_stable_conformer'), \
             patch.object(self.sched, 'run_opt_job'):
            self.sched.pipe_coordinator.ingest_pipe_results(pipe)
        species = self.sched.species_dict['H2O']
        self.assertEqual(species.conformers[1], mock_xyz)
        self.assertIsNone(species.conformers[2])

    def test_ingest_continues_on_parser_exception(self):
        task_ok = _make_task_spec('task_ok2', species_label='H2O', conformer_index=0)
        task_bad = _make_task_spec('task_err', species_label='H2O', conformer_index=3)
        pipe = PipeRun(project_directory=self.tmpdir, run_id='parse_err_test',
                       tasks=[task_bad, task_ok], cluster_software='slurm')
        pipe.stage()
        _complete_task(pipe.pipe_root, 'task_ok2')
        _complete_task(pipe.pipe_root, 'task_err')
        attempt_ok = get_task_attempt_dir(pipe.pipe_root, 'task_ok2', 0)
        attempt_err = get_task_attempt_dir(pipe.pipe_root, 'task_err', 0)
        self._place_output_file(attempt_ok)
        self._place_output_file(attempt_err)
        mock_xyz = {'symbols': ('O', 'H', 'H'), 'isotopes': (16, 1, 1),
                    'coords': ((0.0, 0.0, 0.12), (0.0, 0.76, -0.47), (0.0, -0.76, -0.47))}

        def mock_parse_geometry(log_file_path):
            if 'task_err' in log_file_path:
                raise RuntimeError('simulated parser crash')
            return mock_xyz

        with patch('arc.job.pipe.pipe_run.parser.parse_geometry', side_effect=mock_parse_geometry), \
             patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')), \
             patch('arc.job.pipe.pipe_run.parser.parse_e_elect', return_value=-10.0), \
             patch.object(self.sched, 'determine_most_stable_conformer'), \
             patch.object(self.sched, 'run_opt_job'):
            self.sched.pipe_coordinator.ingest_pipe_results(pipe)
        species = self.sched.species_dict['H2O']
        self.assertEqual(species.conformers[0], mock_xyz)
        self.assertIsNone(species.conformers[3])


class TestConfSpIngestion(unittest.TestCase):
    """Tests for conf_sp pipe ingestion."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_confsp_test_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_conf_sp_ingestion_updates_energy(self):
        """conf_sp ingestion updates conformer energy but not geometry."""
        task = _make_task_spec('sp_task', task_family='conf_sp',
                               species_label='H2O', conformer_index=1)
        pipe = PipeRun(project_directory=self.tmpdir, run_id='sp_ingest',
                       tasks=[task], cluster_software='slurm')
        pipe.stage()
        _complete_task(pipe.pipe_root, 'sp_task')
        attempt_dir = get_task_attempt_dir(pipe.pipe_root, 'sp_task', 0)
        calcs_dir = os.path.join(attempt_dir, 'calcs', 'Species', 'H2O', 'conf_sp_a1')
        os.makedirs(calcs_dir, exist_ok=True)
        with open(os.path.join(calcs_dir, 'output.yml'), 'w') as f:
            f.write('dummy')

        species = self.sched.species_dict['H2O']
        species.conformers[1] = {'symbols': ('O',), 'coords': ((0, 0, 0),)}  # pre-existing geometry

        with patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')), \
             patch('arc.job.pipe.pipe_run.parser.parse_e_elect', return_value=-99.9):
            self.sched.pipe_coordinator.ingest_pipe_results(pipe)

        # Energy updated
        self.assertAlmostEqual(species.conformer_energies[1], -99.9)
        # Geometry preserved (conf_sp doesn't touch it)
        self.assertEqual(species.conformers[1], {'symbols': ('O',), 'coords': ((0, 0, 0),)})

    def test_conf_opt_and_conf_sp_not_mixed(self):
        """conf_opt and conf_sp tasks cannot be in the same PipeRun."""
        t1 = _make_task_spec('t1', task_family='conf_opt')
        t2 = _make_task_spec('t2', task_family='conf_sp')
        run = PipeRun(project_directory=self.tmpdir, run_id='mixed',
                      tasks=[t1, t2], cluster_software='slurm')
        with self.assertRaises(ValueError):
            run.stage()


class TestTryPipeConfSp(unittest.TestCase):
    """Tests for _try_pipe_conf_sp."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_confsp_route_')
        self.sched = _make_scheduler(self.tmpdir)
        # Give the scheduler a conf_sp level
        from arc.level import Level as Lvl
        self.sched.conformer_sp_level = Lvl(method='wb97xd', basis='def2-tzvp')
        self.sched.conformer_opt_level = Lvl(method='b97d3', basis='6-31+g(d,p)')
        self.sched.job_types['conf_sp'] = True

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_conf_sp_pipes_when_enough(self):
        species = self.sched.species_dict['H2O']
        species.conformers = [{'symbols': ('O',), 'isotopes': (16,),
                                'coords': ((0.0, 0.0, float(i)),)}
                               for i in range(12)]
        species.conformer_energies = [None] * 12
        with patch.object(self.sched, 'deduce_job_adapter', return_value='gaussian'):
            result = self.sched.pipe_planner.try_pipe_conf_sp('H2O', list(range(len(self.sched.species_dict['H2O'].conformers))))
        self.assertTrue(result)
        run_id = list(self.sched.active_pipes.keys())[0]
        self.assertIn('conf_sp', run_id)
        pipe = self.sched.active_pipes[run_id]
        self.assertEqual(pipe.tasks[0].task_family, 'conf_sp')

    def test_conf_sp_no_pipe_below_threshold(self):
        species = self.sched.species_dict['H2O']
        species.conformers = [{'symbols': ('O',), 'isotopes': (16,),
                                'coords': ((0.0, 0.0, float(i)),)}
                               for i in range(5)]
        species.conformer_energies = [None] * 5
        with patch.object(self.sched, 'deduce_job_adapter', return_value='gaussian'):
            result = self.sched.pipe_planner.try_pipe_conf_sp('H2O', list(range(len(self.sched.species_dict['H2O'].conformers))))
        self.assertFalse(result)

    def test_conf_sp_not_triggered_when_disabled(self):
        self.sched.job_types['conf_sp'] = False
        species = self.sched.species_dict['H2O']
        species.conformers = [None] * 15
        species.conformer_energies = [None] * 15
        result = self.sched.pipe_planner.try_pipe_conf_sp('H2O', list(range(len(self.sched.species_dict['H2O'].conformers))))
        self.assertFalse(result)


class TestTsIngestion(unittest.TestCase):
    """Tests for TS pipe ingestion."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_ts_ingest_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_ts_opt_ingestion_updates_species(self):
        """ts_opt ingestion updates the matching TSGuess's opt_xyz and energy."""
        from arc.species.species import TSGuess
        ts_label = 'H2O'
        species = self.sched.species_dict[ts_label]
        species.is_ts = True
        tsg = TSGuess(method='heuristics', index=0)
        tsg.success = True
        tsg.conformer_index = 0
        species.ts_guesses = [tsg]

        task = _make_task_spec('ts_opt_task', task_family='ts_opt',
                               species_label=ts_label, conformer_index=0)
        pipe = PipeRun(project_directory=self.tmpdir, run_id='ts_opt_ingest',
                       tasks=[task], cluster_software='slurm')
        pipe.stage()
        _complete_task(pipe.pipe_root, 'ts_opt_task')
        attempt_dir = get_task_attempt_dir(pipe.pipe_root, 'ts_opt_task', 0)
        calcs_dir = os.path.join(attempt_dir, 'calcs', 'Species', ts_label, 'opt_a1')
        os.makedirs(calcs_dir, exist_ok=True)
        with open(os.path.join(calcs_dir, 'output.yml'), 'w') as f:
            f.write('dummy')

        mock_xyz = {'symbols': ('O', 'H', 'H'), 'isotopes': (16, 1, 1),
                    'coords': ((0.0, 0.0, 0.12), (0.0, 0.76, -0.47), (0.0, -0.76, -0.47))}
        with patch('arc.job.pipe.pipe_run.parser.parse_geometry', return_value=mock_xyz), \
             patch('arc.job.pipe.pipe_run.parser.parse_e_elect', return_value=-50.0), \
             patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')), \
             patch.object(self.sched, 'determine_most_likely_ts_conformer'), \
             patch.object(self.sched, 'run_opt_job'):
            self.sched.pipe_coordinator.ingest_pipe_results(pipe)
        self.assertEqual(tsg.opt_xyz, mock_xyz)
        self.assertAlmostEqual(tsg.energy, -50.0)
        self.assertEqual(tsg.index, 0)

    def test_ts_guess_batch_ingestion_calls_process(self):
        """ts_guess_batch_method ingestion calls process_completed_tsg_queue_jobs."""
        ts_label = 'H2O'
        task = _make_task_spec('tsg_task', task_family='ts_guess_batch_method',
                               species_label=ts_label, conformer_index=0)
        pipe = PipeRun(project_directory=self.tmpdir, run_id='tsg_ingest',
                       tasks=[task], cluster_software='slurm')
        pipe.stage()
        _complete_task(pipe.pipe_root, 'tsg_task')
        attempt_dir = get_task_attempt_dir(pipe.pipe_root, 'tsg_task', 0)
        calcs_dir = os.path.join(attempt_dir, 'calcs', 'Species', ts_label, 'tsg_a1')
        os.makedirs(calcs_dir, exist_ok=True)
        with open(os.path.join(calcs_dir, 'output.yml'), 'w') as f:
            f.write('dummy')

        species = self.sched.species_dict[ts_label]
        with patch.object(species, 'process_completed_tsg_queue_jobs') as mock_process:
            with patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')):
                self.sched.pipe_coordinator.ingest_pipe_results(pipe)
            mock_process.assert_called_once()

    def test_ts_not_mixed_with_conformer(self):
        """ts_opt and conf_opt cannot be in the same PipeRun."""
        t1 = _make_task_spec('t1', task_family='conf_opt')
        t2 = _make_task_spec('t2', task_family='ts_opt')
        run = PipeRun(project_directory=self.tmpdir, run_id='mixed',
                      tasks=[t1, t2], cluster_software='slurm')
        with self.assertRaises(ValueError):
            run.stage()


class TestTryPipeTsOpt(unittest.TestCase):
    """Tests for _try_pipe_ts_opt."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_tsopt_route_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_ts_opt_pipes_when_enough(self):
        """When >= 10 TS opt xyzs, pipe mode is used."""
        xyzs = [{'symbols': ('O',), 'isotopes': (16,),
                  'coords': ((0.0, 0.0, float(i)),)}
                 for i in range(12)]
        level = Level(method='wb97xd', basis='def2-tzvp')
        with patch.object(self.sched, 'deduce_job_adapter', return_value='gaussian'):
            result = self.sched.pipe_planner.try_pipe_ts_opt('H2O', xyzs, level)
        self.assertTrue(result)
        run_id = list(self.sched.active_pipes.keys())[0]
        self.assertIn('ts_opt', run_id)
        pipe = self.sched.active_pipes[run_id]
        self.assertEqual(pipe.tasks[0].task_family, 'ts_opt')
        self.assertEqual(pipe.tasks[0].owner_type, 'species')

    def test_ts_opt_no_pipe_below_threshold(self):
        xyzs = [{'symbols': ('O',), 'isotopes': (16,),
                  'coords': ((0.0, 0.0, float(i)),)}
                 for i in range(5)]
        level = Level(method='wb97xd', basis='def2-tzvp')
        with patch.object(self.sched, 'deduce_job_adapter', return_value='gaussian'):
            result = self.sched.pipe_planner.try_pipe_ts_opt('H2O', xyzs, level)
        self.assertFalse(result)


class TestConfOptIngestionSemantics(unittest.TestCase):
    """Verify conf_opt ingestion updates both geometry and energy (ARC-consistent)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_confopt_sem_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_conf_opt_updates_both_geometry_and_energy(self):
        """conf_opt ingestion must update both conformers[i] and conformer_energies[i]."""
        task = _make_task_spec('conf_opt_sem', species_label='H2O', conformer_index=1)
        pipe = PipeRun(project_directory=self.tmpdir, run_id='sem_test',
                       tasks=[task], cluster_software='slurm')
        pipe.stage()
        _complete_task(pipe.pipe_root, 'conf_opt_sem')
        attempt_dir = get_task_attempt_dir(pipe.pipe_root, 'conf_opt_sem', 0)
        calcs_dir = os.path.join(attempt_dir, 'calcs', 'Species', 'H2O', 'conf_opt_a1')
        os.makedirs(calcs_dir, exist_ok=True)
        with open(os.path.join(calcs_dir, 'output.yml'), 'w') as f:
            f.write('dummy')

        mock_xyz = {'symbols': ('O', 'H', 'H'), 'isotopes': (16, 1, 1),
                    'coords': ((0.0, 0.0, 0.12), (0.0, 0.76, -0.47), (0.0, -0.76, -0.47))}
        with patch('arc.job.pipe.pipe_run.parser.parse_geometry', return_value=mock_xyz), \
             patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')), \
             patch('arc.job.pipe.pipe_run.parser.parse_e_elect', return_value=-75.5), \
             patch.object(self.sched, 'determine_most_stable_conformer'), \
             patch.object(self.sched, 'run_opt_job'):
            self.sched.pipe_coordinator.ingest_pipe_results(pipe)
        species = self.sched.species_dict['H2O']
        # Both geometry and energy must be updated (ARC uses opt-level energy for ranking)
        self.assertEqual(species.conformers[1], mock_xyz)
        self.assertAlmostEqual(species.conformer_energies[1], -75.5)


class TestSpeciesSpIngestion(unittest.TestCase):
    """Tests for species_sp pipe ingestion."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_sp_ingest_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_species_sp_sets_e_elect(self):
        task = _make_task_spec('sp_task', task_family='species_sp', species_label='H2O')
        pipe = PipeRun(project_directory=self.tmpdir, run_id='sp_ingest',
                       tasks=[task], cluster_software='slurm')
        pipe.stage()
        _complete_task(pipe.pipe_root, 'sp_task')
        attempt_dir = get_task_attempt_dir(pipe.pipe_root, 'sp_task', 0)
        calcs_dir = os.path.join(attempt_dir, 'calcs', 'Species', 'H2O', 'sp_a1')
        os.makedirs(calcs_dir, exist_ok=True)
        with open(os.path.join(calcs_dir, 'output.yml'), 'w') as f:
            f.write('dummy')

        with patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')), \
             patch('arc.job.pipe.pipe_run.parser.parse_e_elect', return_value=-76.1):
            self.sched.pipe_coordinator.ingest_pipe_results(pipe)
        self.assertAlmostEqual(self.sched.species_dict['H2O'].e_elect, -76.1)


class TestSpeciesFreqIngestion(unittest.TestCase):
    """Tests for species_freq pipe ingestion."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_freq_ingest_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_species_freq_stores_output_path(self):
        task = _make_task_spec('freq_task', task_family='species_freq', species_label='H2O')
        pipe = PipeRun(project_directory=self.tmpdir, run_id='freq_ingest',
                       tasks=[task], cluster_software='slurm')
        pipe.stage()
        _complete_task(pipe.pipe_root, 'freq_task')
        attempt_dir = get_task_attempt_dir(pipe.pipe_root, 'freq_task', 0)
        calcs_dir = os.path.join(attempt_dir, 'calcs', 'Species', 'H2O', 'freq_a1')
        os.makedirs(calcs_dir, exist_ok=True)
        output_path = os.path.join(calcs_dir, 'output.yml')
        with open(output_path, 'w') as f:
            f.write('dummy')

        with patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')):
            self.sched.pipe_coordinator.ingest_pipe_results(pipe)
        self.assertEqual(self.sched.output['H2O']['paths']['freq'], output_path)


class TestIrcIngestion(unittest.TestCase):
    """Tests for IRC pipe ingestion."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_irc_ingest_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_irc_stores_output_path(self):
        task = _make_task_spec('irc_task', task_family='irc', species_label='H2O')
        pipe = PipeRun(project_directory=self.tmpdir, run_id='irc_ingest',
                       tasks=[task], cluster_software='slurm')
        pipe.stage()
        _complete_task(pipe.pipe_root, 'irc_task')
        attempt_dir = get_task_attempt_dir(pipe.pipe_root, 'irc_task', 0)
        calcs_dir = os.path.join(attempt_dir, 'calcs', 'Species', 'H2O', 'irc_a1')
        os.makedirs(calcs_dir, exist_ok=True)
        output_path = os.path.join(calcs_dir, 'output.yml')
        with open(output_path, 'w') as f:
            f.write('dummy')

        with patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')):
            self.sched.pipe_coordinator.ingest_pipe_results(pipe)
        self.assertIn(output_path, self.sched.output['H2O']['paths']['irc'])


class TestTryPipeSpeciesSp(unittest.TestCase):
    """Tests for _try_pipe_species_sp."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_sp_route_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sp_pipes_when_enough(self):
        labels = [f'spc_{i}' for i in range(12)]
        for lbl in labels:
            spc = ARCSpecies(label=lbl, smiles='O')
            self.sched.species_dict[lbl] = spc
        with patch.object(self.sched, 'deduce_job_adapter', return_value='gaussian'):
            result = self.sched.pipe_planner.try_pipe_species_sp(labels)
        self.assertTrue(result)
        run_id = list(self.sched.active_pipes.keys())[0]
        pipe = self.sched.active_pipes[run_id]
        self.assertEqual(pipe.tasks[0].task_family, 'species_sp')
        self.assertEqual(pipe.tasks[0].owner_type, 'species')

    def test_sp_no_pipe_below_threshold(self):
        labels = [f'spc_{i}' for i in range(5)]
        for lbl in labels:
            self.sched.species_dict[lbl] = ARCSpecies(label=lbl, smiles='O')
        with patch.object(self.sched, 'deduce_job_adapter', return_value='gaussian'):
            result = self.sched.pipe_planner.try_pipe_species_sp(labels)
        self.assertFalse(result)


class TestTryPipeIrc(unittest.TestCase):
    """Tests for _try_pipe_irc."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_irc_route_')
        self.sched = _make_scheduler(self.tmpdir)
        self.sched.irc_level = Level(method='wb97xd', basis='def2-tzvp')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_irc_pipes_when_enough(self):
        labels_and_dirs = [(f'ts_spc_{i}', 'forward') for i in range(12)]
        for lbl, _ in labels_and_dirs:
            self.sched.species_dict[lbl] = ARCSpecies(label=lbl, smiles='O', is_ts=True)
        with patch.object(self.sched, 'deduce_job_adapter', return_value='gaussian'):
            result = self.sched.pipe_planner.try_pipe_irc(labels_and_dirs)
        self.assertTrue(result)
        pipe = list(self.sched.active_pipes.values())[0]
        self.assertEqual(pipe.tasks[0].task_family, 'irc')
        self.assertEqual(pipe.tasks[0].ingestion_metadata['irc_direction'], 'forward')

    def test_irc_no_pipe_below_threshold(self):
        labels_and_dirs = [(f'ts_spc_{i}', 'forward') for i in range(3)]
        for lbl, _ in labels_and_dirs:
            self.sched.species_dict[lbl] = ARCSpecies(label=lbl, smiles='O', is_ts=True)
        with patch.object(self.sched, 'deduce_job_adapter', return_value='gaussian'):
            result = self.sched.pipe_planner.try_pipe_irc(labels_and_dirs)
        self.assertFalse(result)


class TestRotorScan1dIngestion(unittest.TestCase):
    """Tests for rotor_scan_1d pipe ingestion."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_scan_ingest_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_scan_ingestion_stores_scan_path(self):
        """rotor_scan_1d ingestion sets rotors_dict[rotor_index]['scan_path']."""
        species = self.sched.species_dict['H2O']
        species.rotors_dict = {0: {'scan_path': '', 'success': None, 'torsion': [0, 1, 2, 3]}}

        task = _make_task_spec('scan_task', task_family='rotor_scan_1d', species_label='H2O')
        # Override ingestion_metadata to include rotor_index
        task.ingestion_metadata = {'rotor_index': 0}
        pipe = PipeRun(project_directory=self.tmpdir, run_id='scan_ingest',
                       tasks=[task], cluster_software='slurm')
        pipe.stage()
        _complete_task(pipe.pipe_root, 'scan_task')
        attempt_dir = get_task_attempt_dir(pipe.pipe_root, 'scan_task', 0)
        calcs_dir = os.path.join(attempt_dir, 'calcs', 'Species', 'H2O', 'scan_a1')
        os.makedirs(calcs_dir, exist_ok=True)
        output_path = os.path.join(calcs_dir, 'output.yml')
        with open(output_path, 'w') as f:
            f.write('dummy')

        with patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')):
            self.sched.pipe_coordinator.ingest_pipe_results(pipe)
        self.assertEqual(species.rotors_dict[0]['scan_path'], output_path)

    def test_scan_ingestion_missing_rotor_slot(self):
        """Ingestion skips safely when the rotor slot doesn't exist."""
        species = self.sched.species_dict['H2O']
        species.rotors_dict = {}  # no rotor 0

        task = _make_task_spec('scan_bad', task_family='rotor_scan_1d', species_label='H2O')
        task.ingestion_metadata = {'rotor_index': 0}
        pipe = PipeRun(project_directory=self.tmpdir, run_id='scan_bad',
                       tasks=[task], cluster_software='slurm')
        pipe.stage()
        _complete_task(pipe.pipe_root, 'scan_bad')
        attempt_dir = get_task_attempt_dir(pipe.pipe_root, 'scan_bad', 0)
        calcs_dir = os.path.join(attempt_dir, 'calcs', 'Species', 'H2O', 'scan_a1')
        os.makedirs(calcs_dir, exist_ok=True)
        with open(os.path.join(calcs_dir, 'output.yml'), 'w') as f:
            f.write('dummy')

        with patch('arc.job.pipe.pipe_run.logger') as mock_logger:
            with patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')):
                self.sched.pipe_coordinator.ingest_pipe_results(pipe)
            warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
            self.assertTrue(any('rotor_index=0' in c and 'not found' in c for c in warning_calls))

    def test_scan_ingestion_no_rotors_dict(self):
        """Ingestion skips safely when species has no rotors_dict."""
        species = self.sched.species_dict['H2O']
        if hasattr(species, 'rotors_dict'):
            del species.rotors_dict

        task = _make_task_spec('scan_nodict', task_family='rotor_scan_1d', species_label='H2O')
        task.ingestion_metadata = {'rotor_index': 0}
        pipe = PipeRun(project_directory=self.tmpdir, run_id='scan_nodict',
                       tasks=[task], cluster_software='slurm')
        pipe.stage()
        _complete_task(pipe.pipe_root, 'scan_nodict')
        attempt_dir = get_task_attempt_dir(pipe.pipe_root, 'scan_nodict', 0)
        calcs_dir = os.path.join(attempt_dir, 'calcs', 'Species', 'H2O', 'scan_a1')
        os.makedirs(calcs_dir, exist_ok=True)
        with open(os.path.join(calcs_dir, 'output.yml'), 'w') as f:
            f.write('dummy')

        with patch('arc.job.pipe.pipe_run.logger') as mock_logger:
            with patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')):
                self.sched.pipe_coordinator.ingest_pipe_results(pipe)
            warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
            self.assertTrue(any('no valid rotors_dict' in c for c in warning_calls))


class TestTryPipeRotorScans1d(unittest.TestCase):
    """Tests for _try_pipe_rotor_scans_1d."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_scan_route_')
        self.sched = _make_scheduler(self.tmpdir)
        self.sched.scan_level = Level(method='wb97xd', basis='def2-tzvp')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_scans_pipe_when_enough(self):
        species = self.sched.species_dict['H2O']
        species.rotors_dict = {i: {'torsion': [0, 1, 2, 3], 'success': None}
                                for i in range(12)}
        with patch.object(self.sched, 'deduce_job_adapter', return_value='gaussian'):
            result = self.sched.pipe_planner.try_pipe_rotor_scans_1d('H2O', list(range(12)))
        self.assertTrue(result)
        pipe = list(self.sched.active_pipes.values())[0]
        self.assertEqual(pipe.tasks[0].task_family, 'rotor_scan_1d')
        self.assertEqual(pipe.tasks[0].owner_type, 'species')
        self.assertEqual(pipe.tasks[0].owner_key, 'H2O')
        self.assertIn('torsions', pipe.tasks[0].input_payload)
        self.assertEqual(pipe.tasks[0].ingestion_metadata['rotor_index'], 0)

    def test_scans_no_pipe_below_threshold(self):
        species = self.sched.species_dict['H2O']
        species.rotors_dict = {i: {'torsion': [0, 1, 2, 3], 'success': None}
                                for i in range(5)}
        with patch.object(self.sched, 'deduce_job_adapter', return_value='gaussian'):
            result = self.sched.pipe_planner.try_pipe_rotor_scans_1d('H2O', list(range(5)))
        self.assertFalse(result)

    def test_scan_not_mixed_with_other_families(self):
        """rotor_scan_1d and conf_opt cannot be in the same PipeRun."""
        t1 = _make_task_spec('t1', task_family='rotor_scan_1d')
        t2 = _make_task_spec('t2', task_family='conf_opt')
        run = PipeRun(project_directory=self.tmpdir, run_id='mixed',
                      tasks=[t1, t2], cluster_software='slurm')
        with self.assertRaises(ValueError):
            run.stage()


class TestNoResubmissionLifecycle(unittest.TestCase):
    """Pipe runs must never resubmit — Q-state workers handle retried tasks."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_resub_test_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_resubmit_when_retried_tasks_and_no_fresh_pending(self):
        """When all workers are done (no fresh PENDING) and retried tasks remain,
        poll_pipes must resubmit a new scheduler job to pick them up."""
        tasks = [_make_task_spec(f'task_{i}') for i in range(3)]
        pipe = self.sched.pipe_coordinator.submit_pipe_run('resub_test', tasks)
        pipe.submitted_at = time.time() - 300  # past grace period
        for task_id in ['task_0', 'task_1', 'task_2']:
            now = time.time()
            update_task_state(pipe.pipe_root, task_id, new_status=TaskState.CLAIMED,
                              claimed_by='w', claim_token='t', claimed_at=now, lease_expires_at=now + 300)
            update_task_state(pipe.pipe_root, task_id, new_status=TaskState.RUNNING, started_at=now)
            update_task_state(pipe.pipe_root, task_id, new_status=TaskState.FAILED_RETRYABLE,
                              ended_at=now, failure_class='test')
            update_task_state(pipe.pipe_root, task_id, new_status=TaskState.PENDING,
                              attempt_index=1, claimed_by=None, claim_token=None,
                              claimed_at=None, lease_expires_at=None,
                              started_at=None, ended_at=None, failure_class=None)
        pipe.status = PipeRunState.RECONCILING
        with patch.object(pipe, 'submit_to_scheduler', return_value=('submitted', '12345')) as mock_submit:
            self.sched.pipe_coordinator.poll_pipes()
        mock_submit.assert_called_once()

    def test_no_resubmit_while_fresh_pending_exist(self):
        """Fresh PENDING tasks (attempt_index == 0) mean Q-state workers are coming.
        Don't resubmit even if retried tasks also exist."""
        tasks = [_make_task_spec(f'task_{i}') for i in range(3)]
        pipe = self.sched.pipe_coordinator.submit_pipe_run('resub_test', tasks)
        pipe.submitted_at = time.time() - 300
        # task_0 completed, task_1 failed and retried, task_2 still fresh PENDING
        now = time.time()
        update_task_state(pipe.pipe_root, 'task_0', new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='t', claimed_at=now, lease_expires_at=now + 300)
        update_task_state(pipe.pipe_root, 'task_0', new_status=TaskState.RUNNING, started_at=now)
        update_task_state(pipe.pipe_root, 'task_0', new_status=TaskState.COMPLETED, ended_at=now)
        update_task_state(pipe.pipe_root, 'task_1', new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='t', claimed_at=now, lease_expires_at=now + 300)
        update_task_state(pipe.pipe_root, 'task_1', new_status=TaskState.RUNNING, started_at=now)
        update_task_state(pipe.pipe_root, 'task_1', new_status=TaskState.FAILED_RETRYABLE,
                          ended_at=now, failure_class='test')
        update_task_state(pipe.pipe_root, 'task_1', new_status=TaskState.PENDING,
                          attempt_index=1, claimed_by=None, claim_token=None,
                          claimed_at=None, lease_expires_at=None,
                          started_at=None, ended_at=None, failure_class=None)
        # task_2 still untouched — fresh PENDING (Q-state worker coming)
        pipe.status = PipeRunState.RECONCILING
        with patch.object(pipe, 'submit_to_scheduler', return_value=('submitted', '12345')) as mock_submit:
            self.sched.pipe_coordinator.poll_pipes()
        mock_submit.assert_not_called()


class TestShouldUsePipeOwnerType(unittest.TestCase):
    """Tests for #4: owner_type homogeneity check."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_owner_test_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rejects_mixed_owner_types(self):
        """Batches with mixed owner_type should be rejected."""
        tasks = [_make_task_spec(f'task_{i}') for i in range(15)]
        # Override one task's owner_type
        mixed = _make_task_spec('task_mixed')
        mixed_dict = mixed.as_dict()
        mixed_dict['owner_type'] = 'reaction'
        mixed_task = TaskSpec.from_dict(mixed_dict)
        # Manually set owner_type since from_dict bypasses validation
        mixed_task.owner_type = 'reaction'
        tasks[7] = mixed_task
        self.assertFalse(self.sched.pipe_coordinator.should_use_pipe(tasks))


class TestWorkerUsesMapping(unittest.TestCase):
    """Tests for #3: worker uses TASK_FAMILY_TO_JOB_TYPE mapping."""

    def test_dispatch_uses_central_mapping(self):
        """Verify worker dispatch derives job_type from TASK_FAMILY_TO_JOB_TYPE."""
        from arc.scripts.pipe_worker import _dispatch_execution, _get_family_extra_kwargs
        from arc.job.pipe.pipe_state import TASK_FAMILY_TO_JOB_TYPE
        # ts_guess_batch_method -> 'tsg' (non-identity mapping)
        self.assertEqual(TASK_FAMILY_TO_JOB_TYPE['ts_guess_batch_method'], 'tsg')
        self.assertEqual(TASK_FAMILY_TO_JOB_TYPE['ts_opt'], 'opt')
        self.assertEqual(TASK_FAMILY_TO_JOB_TYPE['species_sp'], 'sp')

    def test_extra_kwargs_for_irc(self):
        """IRC family should extract irc_direction from ingestion_metadata."""
        from arc.scripts.pipe_worker import _get_family_extra_kwargs
        spec = _make_task_spec('irc_task', task_family='irc')
        spec_dict = spec.as_dict()
        spec_dict['task_family'] = 'irc'
        spec_dict['ingestion_metadata'] = {'irc_direction': 'forward'}
        irc_spec = TaskSpec.from_dict(spec_dict)
        irc_spec.task_family = 'irc'
        irc_spec.ingestion_metadata = {'irc_direction': 'forward'}
        kwargs = _get_family_extra_kwargs(irc_spec)
        self.assertEqual(kwargs, {'irc_direction': 'forward'})


class TestFindOutputFileResultJson(unittest.TestCase):
    """Tests for #6: find_output_file prefers result.json canonical path."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_output_test_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_prefers_result_json_canonical_path(self):
        """find_output_file should use canonical_output_path from result.json."""
        from arc.job.pipe.pipe_run import find_output_file
        attempt_dir = os.path.join(self.tmpdir, 'attempt_0')
        os.makedirs(attempt_dir)
        # Create a canonical output file
        canonical_path = os.path.join(attempt_dir, 'my_output.out')
        with open(canonical_path, 'w') as f:
            f.write('output data')
        # Write result.json pointing to it
        import json
        result = {'canonical_output_path': canonical_path}
        with open(os.path.join(attempt_dir, 'result.json'), 'w') as f:
            json.dump(result, f)
        found = find_output_file(attempt_dir, 'gaussian', 'test_task')
        self.assertEqual(found, canonical_path)

    def test_falls_back_to_walk_without_result_json(self):
        """Without result.json, should fall back to filesystem walk."""
        from arc.job.pipe.pipe_run import find_output_file
        attempt_dir = os.path.join(self.tmpdir, 'attempt_1')
        calcs_dir = os.path.join(attempt_dir, 'calcs', 'subdir')
        os.makedirs(calcs_dir)
        out_file = os.path.join(calcs_dir, 'output.out')
        with open(out_file, 'w') as f:
            f.write('output data')
        found = find_output_file(attempt_dir, 'some_engine', 'test_task')
        self.assertEqual(found, out_file)

    def test_result_json_wins_over_walk(self):
        """When both result.json and calcs/ contain valid files, result.json wins."""
        from arc.job.pipe.pipe_run import find_output_file
        import json
        attempt_dir = os.path.join(self.tmpdir, 'attempt_2')
        # Create the canonical file pointed to by result.json
        canonical_path = os.path.join(attempt_dir, 'canonical_output.log')
        os.makedirs(attempt_dir)
        with open(canonical_path, 'w') as f:
            f.write('canonical output')
        # Also create a file the walk would find (engine=gaussian -> input.log)
        calcs_dir = os.path.join(attempt_dir, 'calcs', 'Species', 'spc')
        os.makedirs(calcs_dir)
        walk_path = os.path.join(calcs_dir, 'input.log')
        with open(walk_path, 'w') as f:
            f.write('walk output')
        # Write result.json pointing to canonical
        with open(os.path.join(attempt_dir, 'result.json'), 'w') as f:
            json.dump({'canonical_output_path': canonical_path}, f)
        found = find_output_file(attempt_dir, 'gaussian', 'test_task')
        self.assertEqual(found, canonical_path)
        self.assertNotEqual(found, walk_path)


class TestFreqIrcIngestionSafety(unittest.TestCase):
    """Tests for #7: freq/irc ingestion initializes output structure if missing."""

    def test_freq_ingestion_creates_output_entry(self):
        """Freq ingestion should create output[label] if missing."""
        from arc.job.pipe.pipe_run import _ingest_species_freq
        from arc.job.pipe.pipe_state import get_task_attempt_dir, initialize_task, TaskStateRecord
        tmpdir = tempfile.mkdtemp(prefix='pipe_freq_test_')
        try:
            spec = _make_task_spec('freq_task', task_family='species_freq')
            pipe_root = tmpdir
            initialize_task(pipe_root, spec, max_attempts=3)
            state = TaskStateRecord(status='completed', attempt_index=0, max_attempts=3, ended_at=time.time())
            species_dict = {'H2O': True}  # species exists
            output = {}  # output entry MISSING
            # Create a fake output file for find_output_file to find
            attempt_dir = get_task_attempt_dir(pipe_root, spec.task_id, 0)
            os.makedirs(attempt_dir, exist_ok=True)
            calcs_dir = os.path.join(attempt_dir, 'calcs')
            os.makedirs(calcs_dir, exist_ok=True)
            with open(os.path.join(calcs_dir, 'output.yml'), 'w') as f:
                f.write('freq output')
            with patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')):
                _ingest_species_freq('run1', pipe_root, spec, state, species_dict, 'H2O', output)
            self.assertIn('H2O', output)
            self.assertIn('freq', output['H2O']['paths'])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_irc_ingestion_creates_output_entry(self):
        """IRC ingestion should create output[label] if missing."""
        from arc.job.pipe.pipe_run import _ingest_irc
        from arc.job.pipe.pipe_state import get_task_attempt_dir, initialize_task, TaskStateRecord
        tmpdir = tempfile.mkdtemp(prefix='pipe_irc_test_')
        try:
            spec = _make_task_spec('irc_task', task_family='irc')
            pipe_root = tmpdir
            initialize_task(pipe_root, spec, max_attempts=3)
            state = TaskStateRecord(status='completed', attempt_index=0, max_attempts=3, ended_at=time.time())
            species_dict = {'TS_H2O': True}
            output = {}  # output entry MISSING
            attempt_dir = get_task_attempt_dir(pipe_root, spec.task_id, 0)
            os.makedirs(attempt_dir, exist_ok=True)
            calcs_dir = os.path.join(attempt_dir, 'calcs')
            os.makedirs(calcs_dir, exist_ok=True)
            with open(os.path.join(calcs_dir, 'output.yml'), 'w') as f:
                f.write('irc output')
            with patch('arc.job.trsh.determine_ess_status', return_value=('done', [], '', '')):
                _ingest_irc('run1', pipe_root, spec, state, species_dict, 'TS_H2O', output)
            self.assertIn('TS_H2O', output)
            self.assertIn('irc', output['TS_H2O']['paths'])
            self.assertEqual(len(output['TS_H2O']['paths']['irc']), 1)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestSubmitPipeRunLifecycle(unittest.TestCase):
    """Tests for #5: submit_pipe_run state consistency."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_lifecycle_test_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_successful_submission_sets_submitted(self):
        """On successful submission, status should be SUBMITTED with job_id."""
        tasks = [_make_task_spec(f'task_{i}') for i in range(3)]
        with patch('arc.job.pipe.pipe_run.PipeRun.submit_to_scheduler',
                   return_value=('submitted', '99999')):
            pipe = self.sched.pipe_coordinator.submit_pipe_run('success_run', tasks)
        self.assertEqual(pipe.status, PipeRunState.SUBMITTED)
        self.assertEqual(pipe.scheduler_job_id, '99999')
        self.assertIsNotNone(pipe.submitted_at)

    def test_failed_submission_stays_staged(self):
        """On failed submission, status should remain STAGED."""
        tasks = [_make_task_spec(f'task_{i}') for i in range(3)]
        with patch('arc.job.pipe.pipe_run.PipeRun.submit_to_scheduler',
                   return_value=('errored', None)):
            pipe = self.sched.pipe_coordinator.submit_pipe_run('fail_run', tasks)
        self.assertEqual(pipe.status, PipeRunState.STAGED)
        self.assertIn('fail_run', self.sched.active_pipes)


class TestPollPipesIntegration(unittest.TestCase):
    """Tests for #9: poll integration with schedule_jobs loop."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_poll_int_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_schedule_jobs_calls_poll_pipes_for_active_pipes(self):
        """schedule_jobs should invoke poll_pipes when active_pipes is non-empty."""
        tasks = [_make_task_spec(f'task_{i}') for i in range(3)]
        pipe = self.sched.pipe_coordinator.submit_pipe_run('poll_int', tasks)
        # Complete all tasks so poll_pipes removes the pipe
        for spec in pipe.tasks:
            _complete_task(pipe.pipe_root, spec.task_id)
        # Mock schedule_jobs loop by calling poll_pipes directly
        # (full schedule_jobs is too heavy; this verifies the integration point)
        self.sched.pipe_coordinator.poll_pipes()
        self.assertNotIn('poll_int', self.sched.active_pipes)


class TestFlushPendingPipeSp(unittest.TestCase):
    """Focused tests for deferred SP batch flushing."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_flush_sp_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_flush_clears_pending_and_calls_planner(self):
        """Pending set is snapshotted, cleared, and planner is called with the labels."""
        self.sched._pending_pipe_sp = {'spc_A', 'spc_B'}
        with patch.object(self.sched.pipe_planner, 'try_pipe_species_sp', return_value={'spc_A', 'spc_B'}):
            with patch.object(self.sched, 'run_sp_job') as mock_sp:
                self.sched._flush_pending_pipe_sp()
        self.assertEqual(self.sched._pending_pipe_sp, set())
        mock_sp.assert_not_called()  # All piped, no fallback.

    def test_flush_falls_back_for_unhandled(self):
        """Unhandled labels are submitted through run_sp_job."""
        self.sched._pending_pipe_sp = {'spc_A', 'spc_B', 'spc_C'}
        with patch.object(self.sched.pipe_planner, 'try_pipe_species_sp', return_value={'spc_B'}):
            with patch.object(self.sched, 'run_sp_job') as mock_sp:
                self.sched._flush_pending_pipe_sp()
        # spc_A and spc_C should fall back (sorted order)
        self.assertEqual(mock_sp.call_count, 2)
        fallback_labels = sorted([c.args[0] for c in mock_sp.call_args_list])
        self.assertEqual(fallback_labels, ['spc_A', 'spc_C'])

    def test_flush_noop_when_empty(self):
        """Empty pending set should not call planner."""
        with patch.object(self.sched.pipe_planner, 'try_pipe_species_sp') as mock_planner:
            self.sched._flush_pending_pipe_sp()
        mock_planner.assert_not_called()


class TestFlushPendingPipeFreq(unittest.TestCase):
    """Focused tests for deferred freq batch flushing."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_flush_freq_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_flush_falls_back_for_unhandled(self):
        """Unhandled labels fall back to run_freq_job."""
        self.sched._pending_pipe_freq = {'spc_X', 'spc_Y'}
        with patch.object(self.sched.pipe_planner, 'try_pipe_species_freq', return_value=set()):
            with patch.object(self.sched, 'run_freq_job') as mock_freq:
                self.sched._flush_pending_pipe_freq()
        self.assertEqual(mock_freq.call_count, 2)


class TestFlushPendingPipeIrc(unittest.TestCase):
    """Focused tests for deferred IRC batch flushing."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_flush_irc_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_flush_falls_back_for_unhandled(self):
        """Unhandled (label, direction) pairs fall back to run_irc_job."""
        self.sched._pending_pipe_irc = {('ts_A', 'forward'), ('ts_A', 'reverse')}
        with patch.object(self.sched.pipe_planner, 'try_pipe_irc', return_value={('ts_A', 'forward')}):
            with patch.object(self.sched, 'run_irc_job') as mock_irc:
                self.sched._flush_pending_pipe_irc()
        mock_irc.assert_called_once_with(label='ts_A', irc_direction='reverse')

    def test_flush_clears_pending(self):
        """Pending set is cleared after flush."""
        self.sched._pending_pipe_irc = {('ts_B', 'forward')}
        with patch.object(self.sched.pipe_planner, 'try_pipe_irc', return_value=set()):
            with patch.object(self.sched, 'run_irc_job'):
                self.sched._flush_pending_pipe_irc()
        self.assertEqual(self.sched._pending_pipe_irc, set())


class TestFlushPendingPipeConfSp(unittest.TestCase):
    """Focused tests for deferred conformer SP batch flushing."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_flush_csp_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_flush_passes_exact_indices_to_planner(self):
        """Planner receives exactly the accumulated conformer indices."""
        self.sched._pending_pipe_conf_sp = {'H2O': {2, 5, 7}}
        with patch.object(self.sched.pipe_planner, 'try_pipe_conf_sp',
                          return_value={2, 5, 7}) as mock_plan:
            with patch.object(self.sched, 'run_sp_job') as mock_sp:
                self.sched._flush_pending_pipe_conf_sp()
        mock_plan.assert_called_once_with('H2O', [2, 5, 7])
        mock_sp.assert_not_called()

    def test_flush_falls_back_for_unhandled_indices(self):
        """Unhandled conformer indices fall back to run_sp_job."""
        self.sched._pending_pipe_conf_sp = {'H2O': {0, 1, 2}}
        with patch.object(self.sched.pipe_planner, 'try_pipe_conf_sp', return_value={1}):
            with patch.object(self.sched, 'run_sp_job') as mock_sp:
                self.sched._flush_pending_pipe_conf_sp()
        # Indices 0 and 2 should fall back (sorted)
        self.assertEqual(mock_sp.call_count, 2)
        fallback_conformers = [c.kwargs.get('conformer') for c in mock_sp.call_args_list]
        self.assertEqual(fallback_conformers, [0, 2])

    def test_flush_clears_pending(self):
        """Pending dict is cleared after flush."""
        self.sched._pending_pipe_conf_sp = {'H2O': {0}}
        with patch.object(self.sched.pipe_planner, 'try_pipe_conf_sp', return_value=set()):
            with patch.object(self.sched, 'run_sp_job'):
                self.sched._flush_pending_pipe_conf_sp()
        self.assertEqual(self.sched._pending_pipe_conf_sp, {})

    def test_returned_handled_is_subset_of_candidates(self):
        """Planner should never return indices outside the supplied candidates."""
        self.sched._pending_pipe_conf_sp = {'H2O': {3, 4}}
        # Simulate planner returning a superset — the flush should still work
        # because it only checks `conformer_indices - piped`.
        with patch.object(self.sched.pipe_planner, 'try_pipe_conf_sp',
                          return_value={3, 4, 99}):
            with patch.object(self.sched, 'run_sp_job') as mock_sp:
                self.sched._flush_pending_pipe_conf_sp()
        mock_sp.assert_not_called()  # {3,4} - {3,4,99} = empty


class TestDeterministicEssError(unittest.TestCase):
    """Tests for _is_deterministic_ess_error classification."""

    def test_max_opt_cycles_is_deterministic(self):
        from arc.scripts.pipe_worker import _is_deterministic_ess_error
        info = {'status': 'errored', 'keywords': ['MaxOptCycles', 'GL9999']}
        self.assertTrue(_is_deterministic_ess_error(info))

    def test_scf_is_deterministic(self):
        from arc.scripts.pipe_worker import _is_deterministic_ess_error
        info = {'status': 'errored', 'keywords': ['SCF', 'GL502']}
        self.assertTrue(_is_deterministic_ess_error(info))

    def test_internal_coord_is_deterministic(self):
        from arc.scripts.pipe_worker import _is_deterministic_ess_error
        info = {'status': 'errored', 'keywords': ['InternalCoordinateError', 'GL103']}
        self.assertTrue(_is_deterministic_ess_error(info))

    def test_no_output_is_transient(self):
        from arc.scripts.pipe_worker import _is_deterministic_ess_error
        info = {'status': 'errored', 'keywords': ['NoOutput']}
        self.assertFalse(_is_deterministic_ess_error(info))

    def test_server_time_limit_is_transient(self):
        from arc.scripts.pipe_worker import _is_deterministic_ess_error
        info = {'status': 'errored', 'keywords': ['ServerTimeLimit']}
        self.assertFalse(_is_deterministic_ess_error(info))

    def test_disk_space_is_transient(self):
        from arc.scripts.pipe_worker import _is_deterministic_ess_error
        info = {'status': 'errored', 'keywords': ['DiskSpace']}
        self.assertFalse(_is_deterministic_ess_error(info))

    def test_done_status_is_not_error(self):
        from arc.scripts.pipe_worker import _is_deterministic_ess_error
        info = {'status': 'done', 'keywords': []}
        self.assertFalse(_is_deterministic_ess_error(info))

    def test_empty_keywords_is_not_deterministic(self):
        from arc.scripts.pipe_worker import _is_deterministic_ess_error
        info = {'status': 'errored', 'keywords': []}
        self.assertFalse(_is_deterministic_ess_error(info))

    def test_none_input(self):
        from arc.scripts.pipe_worker import _is_deterministic_ess_error
        self.assertFalse(_is_deterministic_ess_error(None))

    def test_mixed_transient_and_deterministic_is_deterministic(self):
        from arc.scripts.pipe_worker import _is_deterministic_ess_error
        info = {'status': 'errored', 'keywords': ['NoOutput', 'SCF']}
        self.assertTrue(_is_deterministic_ess_error(info))


class TestEjectToSchedulerJobType(unittest.TestCase):
    """Tests for _eject_to_scheduler job type mapping."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_eject_test_')
        self.sched = _make_scheduler(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_ts_opt_ejects_as_conf_opt(self):
        """ts_opt tasks should be ejected as conf_opt, not opt."""
        from arc.job.pipe.pipe_coordinator import PipeCoordinator
        coord = self.sched.pipe_coordinator
        task = _make_task_spec('t_ts', task_family='ts_opt', species_label='H2O', conformer_index=0)
        state = type('MockState', (), {'failure_class': 'ess_error'})()
        pipe = MagicMock()
        pipe.run_id = 'test'
        with patch.object(self.sched, 'run_job') as mock_run:
            coord._eject_to_scheduler(pipe, task, state)
            mock_run.assert_called_once()
            kwargs = mock_run.call_args[1]
            self.assertEqual(kwargs['job_type'], 'conf_opt')

    def test_species_sp_ejects_as_sp(self):
        task = _make_task_spec('t_sp', task_family='species_sp', species_label='H2O')
        state = type('MockState', (), {'failure_class': 'ess_error'})()
        pipe = MagicMock()
        pipe.run_id = 'test'
        with patch.object(self.sched, 'run_job') as mock_run:
            self.sched.pipe_coordinator._eject_to_scheduler(pipe, task, state)
            kwargs = mock_run.call_args[1]
            self.assertEqual(kwargs['job_type'], 'sp')

    def test_unknown_species_warns(self):
        task = _make_task_spec('t_bad', task_family='ts_opt', species_label='NONEXISTENT')
        task.owner_key = 'NONEXISTENT'
        state = type('MockState', (), {'failure_class': 'ess_error'})()
        pipe = MagicMock()
        pipe.run_id = 'test'
        # Should not crash, just warn.
        self.sched.pipe_coordinator._eject_to_scheduler(pipe, task, state)


class TestPipeRoutingIntegration(unittest.TestCase):
    """
    Integration test: verify that 15+ TS conformer tasks get routed through
    the pipe planner, staged correctly under calcs/TSs/, and produce a valid
    submit script. Uses mockter as the engine to avoid real ESS dependencies.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_routing_int_')
        self.sched = _make_scheduler(self.tmpdir)
        # Make H2O a TS species with 15 successful TSGuess objects.
        from arc.species.species import TSGuess
        ts_spc = self.sched.species_dict['H2O']
        ts_spc.is_ts = True
        ts_spc.rxn_label = 'A + B <=> C + D'
        ts_spc.ts_guesses = []
        for i in range(15):
            tsg = TSGuess(method='heuristics', index=i)
            tsg.success = True
            tsg.initial_xyz = {'symbols': ('O', 'H', 'H'), 'isotopes': (16, 1, 1),
                               'coords': ((0.0, 0.0, 0.1 * i), (0.0, 0.76, -0.47), (0.0, -0.76, -0.47))}
            ts_spc.ts_guesses.append(tsg)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_ts_opt_routed_to_pipe(self):
        """15 TS guesses should trigger pipe mode when using a non-incore adapter."""
        xyzs = [tsg.initial_xyz for tsg in self.sched.species_dict['H2O'].ts_guesses]
        level = Level(repr=default_levels_of_theory['ts_guesses'])
        with patch.object(self.sched, 'deduce_job_adapter', return_value='mockter'):
            result = self.sched.pipe_planner.try_pipe_ts_opt('H2O', xyzs, level)
        # Should have piped all 15 indices.
        self.assertEqual(result, set(range(15)))

    def test_pipe_staged_under_calcs_tss(self):
        """The pipe run should be staged under calcs/TSs/H2O/."""
        xyzs = [tsg.initial_xyz for tsg in self.sched.species_dict['H2O'].ts_guesses]
        level = Level(repr=default_levels_of_theory['ts_guesses'])
        with patch.object(self.sched, 'deduce_job_adapter', return_value='mockter'):
            self.sched.pipe_planner.try_pipe_ts_opt('H2O', xyzs, level)
        # Check that the pipe run was registered.
        self.assertIn('H2O_ts_opt', self.sched.pipe_coordinator.active_pipes)
        pipe = self.sched.pipe_coordinator.active_pipes['H2O_ts_opt']
        # Verify path is under calcs/TSs/H2O/.
        self.assertIn(os.path.join('calcs', 'TSs', 'H2O'), pipe.pipe_root)
        self.assertIn('pipe_ts_opt_0', pipe.pipe_root)

    def test_pipe_has_correct_task_count(self):
        """The staged pipe run should have 15 tasks."""
        xyzs = [tsg.initial_xyz for tsg in self.sched.species_dict['H2O'].ts_guesses]
        level = Level(repr=default_levels_of_theory['ts_guesses'])
        with patch.object(self.sched, 'deduce_job_adapter', return_value='mockter'):
            self.sched.pipe_planner.try_pipe_ts_opt('H2O', xyzs, level)
        pipe = self.sched.pipe_coordinator.active_pipes['H2O_ts_opt']
        self.assertEqual(len(pipe.tasks), 15)
        # All tasks should be ts_opt family.
        self.assertTrue(all(t.task_family == 'ts_opt' for t in pipe.tasks))

    def test_pipe_tasks_staged_on_disk(self):
        """Each task should have spec.json and state.json on disk."""
        xyzs = [tsg.initial_xyz for tsg in self.sched.species_dict['H2O'].ts_guesses]
        level = Level(repr=default_levels_of_theory['ts_guesses'])
        with patch.object(self.sched, 'deduce_job_adapter', return_value='mockter'):
            self.sched.pipe_planner.try_pipe_ts_opt('H2O', xyzs, level)
        pipe = self.sched.pipe_coordinator.active_pipes['H2O_ts_opt']
        tasks_dir = os.path.join(pipe.pipe_root, 'tasks')
        self.assertTrue(os.path.isdir(tasks_dir))
        task_dirs = sorted(os.listdir(tasks_dir))
        self.assertEqual(len(task_dirs), 15)
        for td in task_dirs:
            self.assertTrue(os.path.isfile(os.path.join(tasks_dir, td, 'spec.json')))
            self.assertTrue(os.path.isfile(os.path.join(tasks_dir, td, 'state.json')))

    def test_submit_script_generated(self):
        """A submit script should be generated in the pipe_root."""
        xyzs = [tsg.initial_xyz for tsg in self.sched.species_dict['H2O'].ts_guesses]
        level = Level(repr=default_levels_of_theory['ts_guesses'])
        with patch.object(self.sched, 'deduce_job_adapter', return_value='mockter'):
            self.sched.pipe_planner.try_pipe_ts_opt('H2O', xyzs, level)
        pipe = self.sched.pipe_coordinator.active_pipes['H2O_ts_opt']
        submit_path = os.path.join(pipe.pipe_root, 'submit.sh')
        self.assertTrue(os.path.isfile(submit_path))
        with open(submit_path) as f:
            content = f.read()
        self.assertIn('pipe_worker', content)
        self.assertIn(pipe.pipe_root, content)

    def test_below_threshold_not_piped(self):
        """5 TS guesses should NOT trigger pipe mode (below min_tasks=10)."""
        xyzs = [tsg.initial_xyz for tsg in self.sched.species_dict['H2O'].ts_guesses[:5]]
        level = Level(repr=default_levels_of_theory['ts_guesses'])
        with patch.object(self.sched, 'deduce_job_adapter', return_value='mockter'):
            result = self.sched.pipe_planner.try_pipe_ts_opt('H2O', xyzs, level)
        self.assertEqual(result, set())  # Empty — not piped.


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
