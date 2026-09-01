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
from unittest import mock

import arc.job.pipe.pipe_run as pipe_run_module
from arc.job.adapters.mockter import MockAdapter
from arc.job.pipe.pipe_state import (TaskState, TaskStateRecord, PipeRunState, TaskSpec, get_task_attempt_dir,
                                     read_task_state, update_task_state)
from arc.job.pipe.pipe_run import (PipeRun, build_rotor_scan_1d_tasks, local_cpu_budget,
                                   local_worker_limit, worker_cpu_cores)
import arc.parser.parser as parser
from arc.common import ARC_TESTING_PATH
from arc.level import Level
from arc.species import ARCSpecies
from arc.species.converter import str_to_xyz
from arc.species.species import TSGuess


def _make_spec(task_id, label='H2O', smiles='O', task_family='conf_opt',
               engine='mockter', level=None, required_cores=1, required_memory_mb=512):
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
        required_cores=required_cores,
        required_memory_mb=required_memory_mb,
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

    def test_local_content(self):
        """A local pipe run renders a plain background worker pool, with no queue directives."""
        run = self._make_run('local', max_workers=4, n_tasks=4)
        path = run.write_submit_script()
        self.assertEqual(os.path.basename(path), 'submit.sh')
        with open(path) as f:
            content = f.read()
        self.assertIn('-m arc.scripts.pipe_worker', content)
        self.assertIn('for WORKER_ID in $(seq 1', content)
        self.assertIn('wait', content)
        self.assertIn('export OMP_NUM_THREADS=', content)
        self.assertIn('export ARC_PIPE_LOCAL_CPUS=', content)  # carries the capped budget to the worker
        self.assertNotIn('#SBATCH', content)
        self.assertNotIn('#PBS', content)

    def test_queue_scripts_do_not_export_local_worker_cpus(self):
        """Only the local template sets ARC_PIPE_LOCAL_CPUS, so queued workers keep ARC's default cores."""
        for soft in ('slurm', 'pbs', 'sge', 'htcondor'):
            tasks = [_make_spec(f't_{i}') for i in range(4)]
            run = PipeRun(project_directory=self.tmpdir, run_id=f'sub_{soft}',
                          tasks=tasks, cluster_software=soft, max_workers=4)
            run.stage()
            with open(run.write_submit_script()) as f:
                self.assertNotIn('ARC_PIPE_LOCAL_CPUS', f.read())

    def test_local_worker_count_is_capped_by_machine(self):
        """The local pool never exceeds what the machine can run concurrently."""
        run = self._make_run('local', max_workers=1000, n_tasks=1000)
        _, _, array_size = run._submission_resources()
        self.assertLessEqual(array_size, local_worker_limit(run.tasks[0].required_cores,
                                                            run.tasks[0].required_memory_mb))
        self.assertGreaterEqual(array_size, 1)

    def test_local_per_worker_cores_capped_by_cpu_budget(self):
        """A worker asking for more cores than the local CPU budget is capped at the budget.

        Otherwise a single worker's OMP/MKL/OPENBLAS thread exports would exceed the budget.
        """
        tasks = [_make_spec(f't{i}', required_cores=8) for i in range(4)]
        run = PipeRun(project_directory=self.tmpdir, run_id='budget_cap',
                      tasks=tasks, cluster_software='local', max_workers=4)
        run.stage()
        servers = {'local': {'cluster_soft': 'local', 'cpus': 4}}
        pipe = {'local_max_workers': None}
        with mock.patch.object(pipe_run_module, 'servers_dict', servers), \
                mock.patch.dict(pipe_run_module.settings, {'pipe_settings': pipe, 'servers': servers}):
            cpus, _, array_size = run._submission_resources()
        self.assertEqual(cpus, 4)                  # capped from 8 down to the 4-core budget
        self.assertLessEqual(cpus * array_size, 4)  # workers x cores stays within the budget

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


class TestPipeRunEnvPreamble(unittest.TestCase):
    """Tests for _build_env_preamble and env injection into submit scripts."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_env_preamble_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_run(self, cluster_software='slurm', n_tasks=3):
        tasks = [_make_spec(f't_{i}') for i in range(n_tasks)]
        run = PipeRun(project_directory=self.tmpdir, run_id='env_test',
                      tasks=tasks, cluster_software=cluster_software,
                      max_workers=n_tasks)
        run.stage()
        return run

    def test_ld_library_path_in_slurm_script(self):
        """LD_LIBRARY_PATH export appears in generated slurm submit script."""
        run = self._make_run('slurm')
        path = run.write_submit_script()
        with open(path) as f:
            content = f.read()
        self.assertIn('export LD_LIBRARY_PATH=', content)
        self.assertIn('export CONDA_PREFIX=', content)

    def test_ld_library_path_in_pbs_script(self):
        """LD_LIBRARY_PATH export appears in generated PBS submit script."""
        run = self._make_run('pbs')
        path = run.write_submit_script()
        with open(path) as f:
            content = f.read()
        self.assertIn('export LD_LIBRARY_PATH=', content)

    def test_ld_library_path_before_python(self):
        """LD_LIBRARY_PATH export appears before the python command."""
        run = self._make_run('slurm')
        path = run.write_submit_script()
        with open(path) as f:
            content = f.read()
        ld_pos = content.index('export LD_LIBRARY_PATH=')
        py_pos = content.index('-m arc.scripts.pipe_worker')
        self.assertLess(ld_pos, py_pos)

    def test_pre_cmd_injected(self):
        """User-configured pre_cmd appears in the submit script."""
        from unittest.mock import patch
        run = self._make_run('slurm')
        patched = {'pre_cmd': 'module load openbabel/3.1'}
        with patch.dict('arc.job.pipe.pipe_run.pipe_settings', patched):
            path = run.write_submit_script()
        with open(path) as f:
            content = f.read()
        self.assertIn('module load openbabel/3.1', content)


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

    def test_lease_expiry_orphans_running_task(self):
        """A RUNNING task with an expired lease is detected as orphaned."""
        tasks = [_make_spec('t0'), _make_spec('t1')]
        run = PipeRun(project_directory=self.tmpdir, run_id='lease',
                      tasks=tasks, cluster_software='pbs', max_attempts=1)
        run.stage()
        now = time.time()
        self._complete_task(run.pipe_root, 't0')
        # t1 is RUNNING with an already-expired lease.
        update_task_state(run.pipe_root, 't1', new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='tok', claimed_at=now - 7200,
                          lease_expires_at=now - 10)
        update_task_state(run.pipe_root, 't1', new_status=TaskState.RUNNING,
                          started_at=now - 7200)
        run.reconcile()
        state = read_task_state(run.pipe_root, 't1')
        self.assertEqual(state.status, 'FAILED_TERMINAL')
        self.assertEqual(run.status, PipeRunState.COMPLETED_PARTIAL)


class TestPipeRunResubmission(unittest.TestCase):
    """Tests for the resubmission guard against PBS Q-state workers."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_run_resub_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_run(self, n_tasks=5):
        tasks = [_make_spec(f't{i}') for i in range(n_tasks)]
        run = PipeRun(project_directory=self.tmpdir, run_id='resub',
                      tasks=tasks, cluster_software='slurm', max_attempts=3)
        run.stage()
        run.submitted_at = time.time() - 300  # submitted 5 min ago (past grace period)
        run.status = PipeRunState.SUBMITTED
        return run

    def _fail_retryable(self, pipe_root, task_id):
        """Simulate a worker claiming, running, then failing a task."""
        now = time.time()
        update_task_state(pipe_root, task_id, new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='tok', claimed_at=now,
                          lease_expires_at=now + 300)
        update_task_state(pipe_root, task_id, new_status=TaskState.RUNNING, started_at=now)
        update_task_state(pipe_root, task_id, new_status=TaskState.FAILED_RETRYABLE,
                          ended_at=now + 1, failure_class='timeout')

    def _complete_task(self, pipe_root, task_id):
        now = time.time()
        update_task_state(pipe_root, task_id, new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='tok', claimed_at=now,
                          lease_expires_at=now + 300)
        update_task_state(pipe_root, task_id, new_status=TaskState.RUNNING, started_at=now)
        update_task_state(pipe_root, task_id, new_status=TaskState.COMPLETED, ended_at=now)

    def test_no_resubmit_while_fresh_pending_exist(self):
        """PBS Q-state workers: fresh PENDING tasks mean workers are still starting.
        Even with retried tasks, don't resubmit — those workers will claim retried tasks too."""
        run = self._make_run(n_tasks=5)
        # Workers 1-3 started: t0 completed, t1 failed, t2 completed
        # Workers 4-5 still in PBS Q state: t3, t4 are fresh PENDING
        self._complete_task(run.pipe_root, 't0')
        self._fail_retryable(run.pipe_root, 't1')
        self._complete_task(run.pipe_root, 't2')
        # t3, t4 untouched → fresh PENDING (attempt_index == 0)

        run.reconcile()
        self.assertFalse(run.needs_resubmission,
                         'Should NOT resubmit: Q-state workers will pick up retried tasks')

    def test_resubmit_when_all_workers_done_and_retried_tasks_remain(self):
        """All original workers finished but some tasks failed and were retried.
        No fresh PENDING → no more workers coming → must resubmit."""
        run = self._make_run(n_tasks=3)
        # All 3 workers started: t0 completed, t1 failed, t2 completed
        self._complete_task(run.pipe_root, 't0')
        self._fail_retryable(run.pipe_root, 't1')
        self._complete_task(run.pipe_root, 't2')

        run.reconcile()
        self.assertTrue(run.needs_resubmission,
                        'Should resubmit: no fresh pending, no active workers, retried tasks waiting')

    def test_no_resubmit_within_grace_period(self):
        """Even with retried tasks and no fresh pending, respect the grace period."""
        run = self._make_run(n_tasks=2)
        run.submitted_at = time.time() - 10  # only 10 seconds ago (within 120s grace)
        self._complete_task(run.pipe_root, 't0')
        self._fail_retryable(run.pipe_root, 't1')

        run.reconcile()
        self.assertFalse(run.needs_resubmission,
                         'Should NOT resubmit: within grace period')

    def test_no_resubmit_while_workers_still_active(self):
        """Active workers (CLAIMED/RUNNING) means work is in progress — no resubmit."""
        run = self._make_run(n_tasks=3)
        self._complete_task(run.pipe_root, 't0')
        self._fail_retryable(run.pipe_root, 't1')
        # t2 is currently running (worker still active)
        now = time.time()
        update_task_state(run.pipe_root, 't2', new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='tok', claimed_at=now,
                          lease_expires_at=now + 300)
        update_task_state(run.pipe_root, 't2', new_status=TaskState.RUNNING, started_at=now)

        run.reconcile()
        self.assertFalse(run.needs_resubmission,
                         'Should NOT resubmit: worker still active')


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


class TestLocalCpuBudget(unittest.TestCase):
    """Unit tests for the single global local CPU budget and how it sizes the local worker pool."""

    def test_local_cpu_budget_reads_local_server_cpus(self):
        """The budget is the 'cpus' of the server whose cluster_soft is 'local'."""
        servers = {'local': {'cluster_soft': 'local', 'cpus': 12, 'memory': 32}}
        with mock.patch.object(pipe_run_module, 'servers_dict', servers):
            self.assertEqual(local_cpu_budget(), 12)

    def test_local_cpu_budget_falls_back_to_machine_cores(self):
        """Without a local server 'cpus', the budget falls back to the physical core count."""
        servers = {'zeus': {'cluster_soft': 'pbs', 'cpus': 24}}
        with mock.patch.object(pipe_run_module, 'servers_dict', servers):
            self.assertEqual(local_cpu_budget(), max(1, os.cpu_count() or 1))

    def test_local_cpu_budget_prefers_server_named_local(self):
        """When several servers are 'local', the one named 'local' wins (matches the submit script)."""
        servers = {'workstation': {'cluster_soft': 'local', 'cpus': 64},
                   'local': {'cluster_soft': 'local', 'cpus': 12}}
        with mock.patch.object(pipe_run_module, 'servers_dict', servers):
            self.assertEqual(local_cpu_budget(), 12)

    def test_local_cpu_budget_uses_first_local_when_none_named_local(self):
        """If no server is named 'local', the first server with cluster_soft 'local' is used."""
        servers = {'workstation': {'cluster_soft': 'local', 'cpus': 16}}
        with mock.patch.object(pipe_run_module, 'servers_dict', servers):
            self.assertEqual(local_cpu_budget(), 16)

    def test_local_cpu_budget_tolerates_whitespace_and_case(self):
        """'cluster_soft' matching ignores surrounding whitespace and case."""
        servers = {'local': {'cluster_soft': '  Local ', 'cpus': 8}}
        with mock.patch.object(pipe_run_module, 'servers_dict', servers):
            self.assertEqual(local_cpu_budget(), 8)

    def test_local_cpu_budget_rejects_invalid_cpus(self):
        """A non-positive or non-numeric 'cpus' fails safe to the machine core count, not a crash."""
        fallback = max(1, os.cpu_count() or 1)
        for bad in (0, None, -8, 'local'):
            servers = {'local': {'cluster_soft': 'local', 'cpus': bad}}
            with mock.patch.object(pipe_run_module, 'servers_dict', servers):
                self.assertEqual(local_cpu_budget(), fallback)

    def test_local_worker_limit_bounded_by_cpu_budget(self):
        """The derived worker count is the CPU budget divided by the cores each worker needs."""
        servers = {'local': {'cluster_soft': 'local', 'cpus': 12}}
        pipe = {'local_max_workers': None}
        with mock.patch.object(pipe_run_module, 'servers_dict', servers), \
                mock.patch.dict(pipe_run_module.settings, {'pipe_settings': pipe, 'servers': servers}):
            self.assertEqual(local_worker_limit(cpus_per_worker=1, memory_mb_per_worker=0), 12)
            self.assertEqual(local_worker_limit(cpus_per_worker=4, memory_mb_per_worker=0), 3)

    def test_local_max_workers_overrides_cpu_budget(self):
        """An explicit local_max_workers wins over the derived CPU budget."""
        servers = {'local': {'cluster_soft': 'local', 'cpus': 12}}
        pipe = {'local_max_workers': 4}
        with mock.patch.object(pipe_run_module, 'servers_dict', servers), \
                mock.patch.dict(pipe_run_module.settings, {'pipe_settings': pipe, 'servers': servers}):
            self.assertEqual(local_worker_limit(cpus_per_worker=1, memory_mb_per_worker=0), 4)

    def test_worker_cpu_cores_reads_dedicated_env(self):
        """A local worker pins its job to the budget-capped cores exported as ARC_PIPE_LOCAL_CPUS."""
        self.assertEqual(worker_cpu_cores({'ARC_PIPE_LOCAL_CPUS': '4'}), 4)
        self.assertEqual(worker_cpu_cores({'ARC_PIPE_LOCAL_CPUS': '4,1'}), 4)

    def test_worker_cpu_cores_ignores_ambient_omp(self):
        """A queued worker (no ARC_PIPE_LOCAL_CPUS) never picks up an ambient OMP_NUM_THREADS."""
        self.assertIsNone(worker_cpu_cores({'OMP_NUM_THREADS': '64'}))

    def test_worker_cpu_cores_falls_back_to_none(self):
        """Unset or unparseable value yields None, so the caller uses ARC's default allocation."""
        for env in ({}, {'ARC_PIPE_LOCAL_CPUS': ''}, {'ARC_PIPE_LOCAL_CPUS': '0'}, {'ARC_PIPE_LOCAL_CPUS': 'abc'}):
            self.assertIsNone(worker_cpu_cores(env))


class TestIngestTsOpt(unittest.TestCase):
    """Test the ingestion of a completed pipe ts_opt task into the matching TSGuess."""

    def setUp(self):
        self.ts_xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")
        self.ts_species = ARCSpecies(label='TS_pipe', is_ts=True, xyz=self.ts_xyz, multiplicity=1, charge=0,
                                     compute_thermo=False)
        # The identity and the conformer job position deliberately diverge, and are not in the same order.
        self.tsg_a = TSGuess(index=7, method='heuristics', success=True, xyz=self.ts_xyz)
        self.tsg_a.conformer_index = 0
        self.tsg_b = TSGuess(index=3, method='gcn', success=True, xyz=self.ts_xyz)
        self.tsg_b.conformer_index = 1
        # Stored so that conformer_index disagrees with list position too: conformer 1 sits at
        # position 0. A lookup by position therefore cannot pass by coincidence.
        self.ts_species.ts_guesses = [self.tsg_b, self.tsg_a]
        self.pipe_root = tempfile.mkdtemp(prefix='pipe_ingest_ts_opt_')
        self.addCleanup(shutil.rmtree, self.pipe_root, ignore_errors=True)
        # Real converged TS conformer optimizations, discovered through the same result.json the
        # worker writes. Two different logs, so a cross-attributed result cannot pass unnoticed.
        self.logs = {0: os.path.join(ARC_TESTING_PATH, 'TS_confs', 'TS0_conf_0.out'),
                     1: os.path.join(ARC_TESTING_PATH, 'TS_confs', 'TS0_conf_1.out')}
        self.expected = {i: (parser.parse_geometry(log_file_path=path),
                             parser.parse_e_elect(log_file_path=path))
                         for i, path in self.logs.items()}

    def ingest(self, conformer_index, log_index=0):
        """Run _ingest_ts_opt() against a real attempt directory holding a real ESS log."""
        task_id = f't_ts_opt_{conformer_index}_{log_index}'
        spec = _make_spec(task_id, label='TS_pipe', task_family='ts_opt', engine='gaussian')
        spec.ingestion_metadata = {'conformer_index': conformer_index}
        state = TaskStateRecord(status=TaskState.COMPLETED.value, attempt_index=0)
        attempt_dir = get_task_attempt_dir(self.pipe_root, task_id, state.attempt_index)
        os.makedirs(attempt_dir, exist_ok=True)
        with open(os.path.join(attempt_dir, 'result.json'), 'w') as f:
            json.dump({'canonical_output_path': self.logs[log_index]}, f)
        pipe_run_module._ingest_ts_opt('run_0', self.pipe_root, spec, state,
                                       {'TS_pipe': self.ts_species}, 'TS_pipe')

    def test_ingestion_preserves_the_ts_guess_identity(self):
        """Test that ingesting a result does not overwrite TSGuess.index with the conformer index."""
        self.ingest(conformer_index=0, log_index=0)
        self.assertEqual(self.tsg_a.index, 7)
        self.assertEqual(self.tsg_a.conformer_index, 0)
        self.assertEqual(self.tsg_a.opt_xyz, self.expected[0][0])
        self.assertAlmostEqual(self.tsg_a.energy, self.expected[0][1], 5)

    def test_ingestion_updates_only_the_matching_ts_guess(self):
        """Test that a result is attributed by conformer_index, not by position in the ts_guesses list."""
        self.ingest(conformer_index=1, log_index=1)
        self.assertEqual(self.tsg_b.index, 3)
        self.assertEqual(self.tsg_b.opt_xyz, self.expected[1][0])
        self.assertAlmostEqual(self.tsg_b.energy, self.expected[1][1], 5)
        # The guess at list position 0, which holds the *other* conformer_index, is untouched.
        self.assertIsNone(self.tsg_a.opt_xyz)
        self.assertIsNone(self.tsg_a.energy)
        self.assertNotAlmostEqual(self.expected[0][1], self.expected[1][1], 3)

    def test_ingestion_of_an_unmatched_conformer_index_is_a_no_op(self):
        """Test that a result with no matching conformer_index does not touch any TSGuess."""
        with self.assertLogs('arc', level='WARNING') as context_manager:
            self.ingest(conformer_index=7, log_index=0)
        # Proves ingestion actually reached the attribution step rather than returning early.
        self.assertTrue(any('no TSGuess with conformer_index=7' in message
                            for message in context_manager.output))
        for tsg in (self.tsg_a, self.tsg_b):
            self.assertIsNone(tsg.opt_xyz)
            self.assertIsNone(tsg.energy)
        self.assertEqual(sorted(tsg.index for tsg in self.ts_species.ts_guesses), [3, 7])


class TestBuildRotorScan1dTasks(unittest.TestCase):
    """Unit tests for build_rotor_scan_1d_tasks: one task per rotor and scan_res propagation."""

    def setUp(self):
        self.spc = ARCSpecies(label='propane', smiles='CCC')
        # Two rotors, set explicitly to keep the test hermetic (no conformer generation).
        self.spc.rotors_dict = {0: {'torsion': [3, 0, 1, 2], 'pivots': [1, 2]},
                                1: {'torsion': [1, 2, 3, 8], 'pivots': [2, 3]}}
        self.level_dict = {'method': 'uma-s-1p2'}

    def test_one_task_per_rotor(self):
        tasks = build_rotor_scan_1d_tasks(self.spc, 'propane', [0, 1], self.level_dict, 'ase', 4096)
        self.assertEqual([t.task_id for t in tasks], ['propane_scan_r0', 'propane_scan_r1'])
        self.assertTrue(all(t.task_family == 'rotor_scan_1d' and t.engine == 'ase' for t in tasks))
        self.assertEqual([t.input_payload['rotor_index'] for t in tasks], [0, 1])

    def test_scan_res_carried_into_payload_when_given(self):
        tasks = build_rotor_scan_1d_tasks(self.spc, 'propane', [0, 1], self.level_dict, 'ase', 4096,
                                          scan_res=8.0)
        self.assertTrue(all(t.input_payload['scan_res'] == 8.0 for t in tasks))

    def test_scan_res_omitted_when_none(self):
        tasks = build_rotor_scan_1d_tasks(self.spc, 'propane', [0], self.level_dict, 'ase', 4096)
        self.assertNotIn('scan_res', tasks[0].input_payload)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
