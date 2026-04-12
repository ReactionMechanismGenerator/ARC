#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.pipe_run module
"""

import json
import os
import shutil
import tempfile
import time
import unittest

from arc.job.adapters.mockter import MockAdapter
from arc.job.pipe.pipe_state import TaskState, PipeRunState, TaskSpec, read_task_state, update_task_state
from arc.job.pipe.pipe_run import PipeRun
from arc.level import Level
from arc.species import ARCSpecies


def _make_spec(task_id, label='H2O', smiles='O', task_family='conf_opt',
               engine='mockter', level=None):
    """Helper to create a TaskSpec for testing."""
    spc = ARCSpecies(label=label, smiles=smiles)
    return TaskSpec(
        task_id=task_id,
        task_family=task_family,
        owner_type='species',
        owner_key=label,
        input_fingerprint=f'{task_id}_fp',
        engine=engine,
        level=level or {'method': 'mock', 'basis': 'mock'},
        required_cores=1,
        required_memory_mb=512,
        input_payload={'species_dicts': [spc.as_dict()]},
        ingestion_metadata={'conformer_index': 0},
    )


class TestAdapterPipeRejection(unittest.TestCase):

    def test_execute_pipe_raises_value_error(self):
        job = MockAdapter(
            execution_type='incore', job_type='sp',
            level=Level(method='mock', basis='mock'),
            project='test',
            project_directory=os.path.join(tempfile.gettempdir(), 'pipe_reject_test'),
            species=[ARCSpecies(label='H2O', smiles='O')],
            testing=True)
        job.execution_type = 'pipe'
        with self.assertRaises(ValueError):
            job.execute()


class TestPipeRunStaging(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_run_stage_')
        self.tasks = [_make_spec(f'task_{i}') for i in range(3)]
        self.run = PipeRun(
            project_directory=self.tmpdir, run_id='test_001',
            tasks=self.tasks, cluster_software='slurm', max_attempts=3)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stage_creates_directory_tree(self):
        self.run.stage()
        for task in self.tasks:
            task_dir = os.path.join(self.run.pipe_root, 'tasks', task.task_id)
            self.assertTrue(os.path.isfile(os.path.join(task_dir, 'spec.json')))
            self.assertTrue(os.path.isfile(os.path.join(task_dir, 'state.json')))

    def test_stage_sets_status(self):
        self.run.stage()
        self.assertEqual(self.run.status, PipeRunState.STAGED)

    def test_run_json_written(self):
        self.run.stage()
        run_path = os.path.join(self.run.pipe_root, 'run.json')
        self.assertTrue(os.path.isfile(run_path))
        with open(run_path) as f:
            data = json.load(f)
        self.assertEqual(data['run_id'], 'test_001')
        self.assertEqual(data['status'], 'STAGED')

    def test_run_json_has_rich_metadata(self):
        """run.json includes homogeneous task_family, engine, level, and timestamps."""
        self.run.stage()
        with open(os.path.join(self.run.pipe_root, 'run.json')) as f:
            data = json.load(f)
        self.assertEqual(data['task_family'], 'conf_opt')
        self.assertEqual(data['engine'], 'mockter')
        self.assertEqual(data['level'], {'method': 'mock', 'basis': 'mock'})
        self.assertIsNotNone(data['created_at'])
        self.assertIsNone(data['submitted_at'])
        self.assertIsNone(data['scheduler_job_id'])


class TestPipeRunFromDir(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_run_fromdir_')
        self.tasks = [_make_spec(f'task_{i}') for i in range(2)]
        self.run = PipeRun(
            project_directory=self.tmpdir, run_id='restore_test',
            tasks=self.tasks, cluster_software='pbs',
            max_workers=50, max_attempts=5)
        self.run.stage()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_from_dir_reconstructs(self):
        restored = PipeRun.from_dir(self.run.pipe_root)
        self.assertEqual(restored.run_id, 'restore_test')
        self.assertEqual(restored.cluster_software, 'pbs')
        self.assertEqual(restored.max_workers, 50)
        self.assertEqual(restored.status, PipeRunState.STAGED)
        self.assertEqual(len(restored.tasks), 2)

    def test_from_dir_rich_metadata(self):
        restored = PipeRun.from_dir(self.run.pipe_root)
        self.assertIsNotNone(restored.created_at)
        self.assertIsNone(restored.scheduler_job_id)


class TestPipeRunWriteSubmitScript(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_submit_script_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_run(self, cluster_software, max_workers=10, n_tasks=None):
        n = n_tasks if n_tasks is not None else max_workers
        tasks = [_make_spec(f't_{i}') for i in range(n)]
        run = PipeRun(project_directory=self.tmpdir, run_id='sub_test',
                      tasks=tasks, cluster_software=cluster_software,
                      max_workers=max_workers)
        run.stage()
        return run

    def test_slurm_content(self):
        run = self._make_run('slurm', max_workers=25, n_tasks=25)
        path = run.write_submit_script()
        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            content = f.read()
        self.assertIn('#!/bin/bash -l', content)
        self.assertIn('#SBATCH --array=1-25', content)
        self.assertIn('WORKER_ID=$SLURM_ARRAY_TASK_ID', content)
        self.assertIn('-m arc.scripts.pipe_worker', content)

    def test_pbs_content(self):
        run = self._make_run('pbs', max_workers=8, n_tasks=8)
        path = run.write_submit_script()
        with open(path) as f:
            content = f.read()
        self.assertIn('#PBS -J 1-8', content)
        self.assertIn('WORKER_ID="$PBS_ARRAY_INDEX"', content)

    def test_htcondor_content(self):
        run = self._make_run('htcondor', max_workers=12, n_tasks=12)
        path = run.write_submit_script()
        self.assertEqual(os.path.basename(path), 'submit.sub')
        with open(path) as f:
            content = f.read()
        self.assertIn('queue 12', content)

    def test_overwrite_is_safe(self):
        run = self._make_run('slurm')
        p1 = run.write_submit_script()
        p2 = run.write_submit_script()
        self.assertEqual(p1, p2)

    def test_unsupported_raises(self):
        run = self._make_run('mystery')
        with self.assertRaises(NotImplementedError):
            run.write_submit_script()

    def test_shell_script_is_executable(self):
        """Shell submit scripts (slurm/pbs/sge) have executable permissions."""
        import stat
        run = self._make_run('slurm')
        path = run.write_submit_script()
        mode = os.stat(path).st_mode
        self.assertTrue(mode & stat.S_IXUSR, 'slurm script should be user-executable')

    def test_htcondor_sub_not_executable(self):
        """HTCondor .sub files should not have executable bit set."""
        import stat
        run = self._make_run('htcondor')
        path = run.write_submit_script()
        mode = os.stat(path).st_mode
        self.assertFalse(mode & stat.S_IXUSR, '.sub should not be executable')


class TestPipeRunReconcile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_run_reconcile_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _complete_task(self, pipe_root, task_id):
        now = time.time()
        update_task_state(pipe_root, task_id, new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='tok', claimed_at=now, lease_expires_at=now + 300)
        update_task_state(pipe_root, task_id, new_status=TaskState.RUNNING, started_at=now)
        update_task_state(pipe_root, task_id, new_status=TaskState.COMPLETED, ended_at=now)

    def test_orphan_retry_clears_claim_token(self):
        """Retry via reconcile clears claim_token."""
        run = PipeRun(project_directory=self.tmpdir, run_id='orphan',
                      tasks=[_make_spec('t')], cluster_software='slurm')
        run.stage()
        now = time.time()
        update_task_state(run.pipe_root, 't', new_status=TaskState.CLAIMED,
                          claimed_by='dead', claim_token='old_token',
                          claimed_at=now - 200, lease_expires_at=now - 10)
        run.reconcile()
        state = read_task_state(run.pipe_root, 't')
        self.assertEqual(state.status, 'PENDING')
        self.assertIsNone(state.claim_token)

    def test_all_completed(self):
        tasks = [_make_spec(f'task_{i}') for i in range(3)]
        run = PipeRun(project_directory=self.tmpdir, run_id='done',
                      tasks=tasks, cluster_software='slurm')
        run.stage()
        for t in tasks:
            self._complete_task(run.pipe_root, t.task_id)
        run.reconcile()
        self.assertEqual(run.status, PipeRunState.COMPLETED)
        self.assertIsNotNone(run.completed_at)
        with open(os.path.join(run.pipe_root, 'run.json')) as f:
            self.assertIsNotNone(json.load(f).get('completed_at'))

    def test_retryable_budget_exhausted(self):
        run = PipeRun(project_directory=self.tmpdir, run_id='exhausted',
                      tasks=[_make_spec('t')], cluster_software='slurm', max_attempts=1)
        run.stage()
        now = time.time()
        update_task_state(run.pipe_root, 't', new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='tok', claimed_at=now, lease_expires_at=now + 300)
        update_task_state(run.pipe_root, 't', new_status=TaskState.RUNNING, started_at=now)
        update_task_state(run.pipe_root, 't', new_status=TaskState.FAILED_RETRYABLE,
                          ended_at=now + 5, failure_class='timeout')
        run.reconcile()
        state = read_task_state(run.pipe_root, 't')
        self.assertEqual(state.status, 'FAILED_TERMINAL')

    def test_terminal_run_not_regressed(self):
        tasks = [_make_spec(f'task_{i}') for i in range(2)]
        run = PipeRun(project_directory=self.tmpdir, run_id='terminal',
                      tasks=tasks, cluster_software='slurm')
        run.stage()
        for t in tasks:
            self._complete_task(run.pipe_root, t.task_id)
        run.reconcile()
        self.assertEqual(run.status, PipeRunState.COMPLETED)
        run.reconcile()
        self.assertEqual(run.status, PipeRunState.COMPLETED)

    def test_scheduler_job_gone_orphans_running_task(self):
        """When the scheduler job is gone, RUNNING tasks are orphaned immediately."""
        tasks = [_make_spec('t0'), _make_spec('t1')]
        run = PipeRun(project_directory=self.tmpdir, run_id='gone',
                      tasks=tasks, cluster_software='pbs', max_attempts=1)
        run.stage()
        now = time.time()
        # t0 completed normally
        self._complete_task(run.pipe_root, 't0')
        # t1 is RUNNING with lease still valid (24h in the future)
        update_task_state(run.pipe_root, 't1', new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='tok', claimed_at=now,
                          lease_expires_at=now + 86400)
        update_task_state(run.pipe_root, 't1', new_status=TaskState.RUNNING, started_at=now)

        # With scheduler_job_alive=True (default), the task stays RUNNING.
        counts = run.reconcile(scheduler_job_alive=True)
        self.assertEqual(counts[TaskState.RUNNING.value], 1)

        # Reset run status so reconcile runs the main logic again.
        run.status = PipeRunState.SUBMITTED

        # With scheduler_job_alive=False, RUNNING is immediately orphaned → FAILED_TERMINAL.
        counts = run.reconcile(scheduler_job_alive=False)
        self.assertEqual(counts[TaskState.RUNNING.value], 0)
        state = read_task_state(run.pipe_root, 't1')
        self.assertEqual(state.status, 'FAILED_TERMINAL')
        self.assertEqual(run.status, PipeRunState.COMPLETED_PARTIAL)

    def test_scheduler_job_gone_stranded_pending_failed(self):
        """When the scheduler job is gone, unreachable PENDING tasks are FAILED_TERMINAL."""
        tasks = [_make_spec('t0'), _make_spec('t1')]
        run = PipeRun(project_directory=self.tmpdir, run_id='stranded',
                      tasks=tasks, cluster_software='slurm', max_attempts=3)
        run.stage()
        # t0 completed
        self._complete_task(run.pipe_root, 't0')
        # t1 was never claimed — still PENDING (e.g., worker crashed before claiming)

        run.reconcile(scheduler_job_alive=False)
        state = read_task_state(run.pipe_root, 't1')
        self.assertEqual(state.status, 'FAILED_TERMINAL')
        self.assertEqual(run.status, PipeRunState.COMPLETED_PARTIAL)


class TestPipeRunNoResubmission(unittest.TestCase):
    """Pipe runs must never flag resubmission — Q-state workers handle retried tasks."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_run_resub_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_never_resubmit_even_with_retried_tasks_and_no_workers(self):
        """Even when all workers are done and retried tasks remain,
        needs_resubmission must stay False — no automatic resubmission."""
        tasks = [_make_spec(f't{i}') for i in range(3)]
        run = PipeRun(project_directory=self.tmpdir, run_id='resub',
                      tasks=tasks, cluster_software='slurm', max_attempts=3)
        run.stage()
        run.submitted_at = time.time() - 300
        run.status = PipeRunState.SUBMITTED
        # All 3 workers started: t0 completed, t1 failed, t2 completed
        now = time.time()
        for tid in ('t0', 't2'):
            update_task_state(run.pipe_root, tid, new_status=TaskState.CLAIMED,
                              claimed_by='w', claim_token='tok', claimed_at=now,
                              lease_expires_at=now + 300)
            update_task_state(run.pipe_root, tid, new_status=TaskState.RUNNING, started_at=now)
            update_task_state(run.pipe_root, tid, new_status=TaskState.COMPLETED, ended_at=now)
        update_task_state(run.pipe_root, 't1', new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='tok', claimed_at=now,
                          lease_expires_at=now + 300)
        update_task_state(run.pipe_root, 't1', new_status=TaskState.RUNNING, started_at=now)
        update_task_state(run.pipe_root, 't1', new_status=TaskState.FAILED_RETRYABLE,
                          ended_at=now + 1, failure_class='timeout')

        run.reconcile()
        self.assertFalse(run.needs_resubmission,
                         'Should never resubmit — Q-state workers or manual intervention handle retries')


class TestPipeRunHomogeneity(unittest.TestCase):
    """Tests for PipeRun homogeneity validation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_homo_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_mixed_families_rejected(self):
        """Mixing conf_opt and conf_sp in one run is rejected."""
        tasks = [_make_spec('t1', task_family='conf_opt'),
                 _make_spec('t2', task_family='conf_sp')]
        run = PipeRun(project_directory=self.tmpdir, run_id='mixed',
                      tasks=tasks, cluster_software='slurm')
        with self.assertRaises(ValueError):
            run.stage()

    def test_mixed_engines_rejected(self):
        tasks = [_make_spec('t1', engine='mockter'),
                 _make_spec('t2', engine='gaussian')]
        run = PipeRun(project_directory=self.tmpdir, run_id='mixed_eng',
                      tasks=tasks, cluster_software='slurm')
        with self.assertRaises(ValueError):
            run.stage()

    def test_homogeneous_conf_sp_accepted(self):
        tasks = [_make_spec(f't_{i}', task_family='conf_sp') for i in range(3)]
        run = PipeRun(project_directory=self.tmpdir, run_id='sp_ok',
                      tasks=tasks, cluster_software='slurm')
        run.stage()
        self.assertEqual(run.status, PipeRunState.STAGED)
        with open(os.path.join(run.pipe_root, 'run.json')) as f:
            data = json.load(f)
        self.assertEqual(data['task_family'], 'conf_sp')

    def test_from_dir_reconstructs_conf_sp(self):
        """from_dir reconstructs conf_sp tasks correctly."""
        tasks = [_make_spec(f't_{i}', task_family='conf_sp') for i in range(2)]
        run = PipeRun(project_directory=self.tmpdir, run_id='sp_restore',
                      tasks=tasks, cluster_software='slurm')
        run.stage()
        restored = PipeRun.from_dir(run.pipe_root)
        self.assertEqual(len(restored.tasks), 2)
        self.assertEqual(restored.tasks[0].task_family, 'conf_sp')

    def test_mixed_ts_and_conformer_rejected(self):
        """Mixing ts_opt and conf_opt in one run is rejected."""
        tasks = [_make_spec('t1', task_family='conf_opt'),
                 _make_spec('t2', task_family='ts_opt')]
        run = PipeRun(project_directory=self.tmpdir, run_id='mixed_ts_conf',
                      tasks=tasks, cluster_software='slurm')
        with self.assertRaises(ValueError):
            run.stage()

    def test_mixed_ts_families_rejected(self):
        """Mixing ts_guess_batch_method and ts_opt in one run is rejected."""
        tasks = [_make_spec('t1', task_family='ts_guess_batch_method'),
                 _make_spec('t2', task_family='ts_opt')]
        run = PipeRun(project_directory=self.tmpdir, run_id='mixed_ts',
                      tasks=tasks, cluster_software='slurm')
        with self.assertRaises(ValueError):
            run.stage()

    def test_homogeneous_ts_opt_accepted(self):
        tasks = [_make_spec(f't_{i}', task_family='ts_opt') for i in range(3)]
        run = PipeRun(project_directory=self.tmpdir, run_id='ts_ok',
                      tasks=tasks, cluster_software='slurm')
        run.stage()
        self.assertEqual(run.status, PipeRunState.STAGED)
        with open(os.path.join(run.pipe_root, 'run.json')) as f:
            self.assertEqual(json.load(f)['task_family'], 'ts_opt')

    def test_from_dir_reconstructs_ts_opt(self):
        tasks = [_make_spec(f't_{i}', task_family='ts_opt') for i in range(2)]
        run = PipeRun(project_directory=self.tmpdir, run_id='ts_restore',
                      tasks=tasks, cluster_software='slurm')
        run.stage()
        restored = PipeRun.from_dir(run.pipe_root)
        self.assertEqual(len(restored.tasks), 2)
        self.assertEqual(restored.tasks[0].task_family, 'ts_opt')

    def test_homogeneous_species_sp_accepted(self):
        tasks = [_make_spec(f't_{i}', task_family='species_sp') for i in range(3)]
        run = PipeRun(project_directory=self.tmpdir, run_id='sp_ok',
                      tasks=tasks, cluster_software='slurm')
        run.stage()
        self.assertEqual(run.status, PipeRunState.STAGED)

    def test_homogeneous_species_freq_accepted(self):
        tasks = [_make_spec(f't_{i}', task_family='species_freq') for i in range(3)]
        run = PipeRun(project_directory=self.tmpdir, run_id='freq_ok',
                      tasks=tasks, cluster_software='slurm')
        run.stage()
        self.assertEqual(run.status, PipeRunState.STAGED)

    def test_homogeneous_irc_accepted(self):
        tasks = [_make_spec(f't_{i}', task_family='irc') for i in range(3)]
        run = PipeRun(project_directory=self.tmpdir, run_id='irc_ok',
                      tasks=tasks, cluster_software='slurm')
        run.stage()
        self.assertEqual(run.status, PipeRunState.STAGED)

    def test_mixed_sp_and_freq_rejected(self):
        tasks = [_make_spec('t1', task_family='species_sp'),
                 _make_spec('t2', task_family='species_freq')]
        run = PipeRun(project_directory=self.tmpdir, run_id='mixed_leaf',
                      tasks=tasks, cluster_software='slurm')
        with self.assertRaises(ValueError):
            run.stage()

    def test_from_dir_reconstructs_species_sp(self):
        tasks = [_make_spec(f't_{i}', task_family='species_sp') for i in range(2)]
        run = PipeRun(project_directory=self.tmpdir, run_id='sp_restore',
                      tasks=tasks, cluster_software='slurm')
        run.stage()
        restored = PipeRun.from_dir(run.pipe_root)
        self.assertEqual(len(restored.tasks), 2)
        self.assertEqual(restored.tasks[0].task_family, 'species_sp')

    def test_homogeneous_rotor_scan_1d_accepted(self):
        tasks = [_make_spec(f't_{i}', task_family='rotor_scan_1d') for i in range(3)]
        run = PipeRun(project_directory=self.tmpdir, run_id='scan_ok',
                      tasks=tasks, cluster_software='slurm')
        run.stage()
        self.assertEqual(run.status, PipeRunState.STAGED)

    def test_mixed_scan_and_conformer_rejected(self):
        tasks = [_make_spec('t1', task_family='rotor_scan_1d'),
                 _make_spec('t2', task_family='conf_opt')]
        run = PipeRun(project_directory=self.tmpdir, run_id='mixed_scan',
                      tasks=tasks, cluster_software='slurm')
        with self.assertRaises(ValueError):
            run.stage()

    def test_from_dir_reconstructs_rotor_scan_1d(self):
        tasks = [_make_spec(f't_{i}', task_family='rotor_scan_1d') for i in range(2)]
        run = PipeRun(project_directory=self.tmpdir, run_id='scan_restore',
                      tasks=tasks, cluster_software='slurm')
        run.stage()
        restored = PipeRun.from_dir(run.pipe_root)
        self.assertEqual(len(restored.tasks), 2)
        self.assertEqual(restored.tasks[0].task_family, 'rotor_scan_1d')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
