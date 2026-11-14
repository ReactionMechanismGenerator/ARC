#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.scripts.pipe_worker module
"""

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from arc.job.pipe.pipe_run import PipeRun
from arc.job.pipe.pipe_state import (
    TaskState,
    TaskSpec,
    generate_claim_token,
    get_task_attempt_dir,
    initialize_task,
    read_task_state,
    update_task_state,
)
from arc.scripts.pipe_worker import (LeaseHeartbeat, claim_task, get_lease_duration_s, run_task, main,
                                     logger as worker_logger, _get_family_extra_kwargs)
from arc.species import ARCSpecies


def _make_h2o_spec(task_id='sp_h2o', task_family='conf_opt'):
    """Helper to create a TaskSpec for H2O using the mockter adapter."""
    spc = ARCSpecies(label='H2O', smiles='O')
    return TaskSpec(
        task_id=task_id,
        task_family=task_family,
        owner_type='species',
        owner_key='H2O',
        input_fingerprint=f'{task_id}_fp',
        engine='mockter',
        level={'method': 'mock', 'basis': 'mock'},
        required_cores=1,
        required_memory_mb=512,
        input_payload={'species_dicts': [spc.as_dict()]},
        ingestion_metadata={'conformer_index': 0},
    )


class TestLeaseHeartbeat(unittest.TestCase):

    LEASE = 0.9

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_heartbeat_test_')
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        patcher = patch('arc.scripts.pipe_worker._parse_ess_error', return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch('arc.scripts.pipe_worker.get_lease_duration_s', return_value=self.LEASE)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _stage(self, task_id='hb'):
        run = PipeRun(project_directory=self.tmpdir, run_id='hb_run',
                      tasks=[_make_h2o_spec(task_id)], cluster_software='slurm')
        run.stage()
        now = time.time()
        token = generate_claim_token()
        state = update_task_state(run.pipe_root, task_id, new_status=TaskState.CLAIMED,
                                  claimed_by='w', claim_token=token, claimed_at=now,
                                  lease_expires_at=now + self.LEASE)
        return run, state, token

    def _run_in_thread(self, run, state, token, dispatch):
        patcher = patch('arc.scripts.pipe_worker._dispatch_execution', side_effect=dispatch)
        patcher.start()
        self.addCleanup(patcher.stop)
        t = threading.Thread(target=run_task, args=(run.pipe_root, 'hb', state, 'w', token), daemon=True)
        t.start()
        self.addCleanup(t.join, 10)
        return t

    def _heartbeat_threads(self):
        return [t for t in threading.enumerate() if t.name.startswith('lease-heartbeat-')]

    def test_interval_capped(self):
        for lease, expected in ((0.9, 0.3), (3600, 300), (60, 20)):
            hb = LeaseHeartbeat(self.tmpdir, 'hb', 'w', 't', lease_duration=lease)
            self.assertAlmostEqual(hb.interval, expected)
            self.assertLess(hb.interval, lease)

    def test_relative_pipe_root_survives_chdir(self):
        run, state, token = self._stage()
        original = os.getcwd()
        os.chdir(os.path.dirname(run.pipe_root))
        self.addCleanup(os.chdir, original)
        hb = LeaseHeartbeat(os.path.basename(run.pipe_root), 'hb', 'w', token,
                            lease_duration=self.LEASE, interval=0.1)
        self.assertTrue(os.path.isabs(hb.pipe_root))
        other = tempfile.mkdtemp(prefix='pipe_heartbeat_cwd_')
        self.addCleanup(shutil.rmtree, other, ignore_errors=True)
        os.chdir(other)
        hb.start()
        time.sleep(0.5)
        hb.stop()
        self.assertGreater(read_task_state(run.pipe_root, 'hb').lease_expires_at, state.lease_expires_at)

    def test_transient_failure_retries_with_single_warning(self):
        run, state, token = self._stage()
        hb = LeaseHeartbeat(run.pipe_root, 'hb', 'w', token, lease_duration=self.LEASE, interval=0.1)
        real = update_task_state
        calls = []

        def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) <= 2:
                raise TimeoutError('slow lock')
            return real(*args, **kwargs)

        with patch('arc.scripts.pipe_worker.update_task_state', side_effect=flaky):
            with self.assertLogs(worker_logger, level='WARNING') as logs:
                hb.start()
                time.sleep(0.8)
                hb.stop()
        self.assertEqual(len(logs.output), 1)
        self.assertGreater(len(calls), 2)
        self.assertGreater(read_task_state(run.pipe_root, 'hb').lease_expires_at, state.lease_expires_at)

    def test_lease_renewed_while_task_runs_and_not_orphaned(self):
        run, state, token = self._stage()
        release = threading.Event()
        self.addCleanup(release.set)
        first_lease = state.lease_expires_at
        thread = self._run_in_thread(run, state, token, lambda spec, scratch: release.wait(10))
        time.sleep(self.LEASE * 2.5)
        mid = read_task_state(run.pipe_root, 'hb')
        self.assertEqual(mid.status, 'RUNNING')
        self.assertGreater(mid.lease_expires_at, first_lease + self.LEASE)
        self.assertGreater(mid.lease_expires_at, time.time())
        run.reconcile()
        after = read_task_state(run.pipe_root, 'hb')
        self.assertEqual(after.status, 'RUNNING')
        self.assertEqual(after.claim_token, token)
        release.set()
        thread.join(10)
        self.assertEqual(read_task_state(run.pipe_root, 'hb').status, 'COMPLETED')

    def test_heartbeat_stops_after_completion(self):
        run, state, token = self._stage()
        thread = self._run_in_thread(run, state, token, lambda spec, scratch: None)
        thread.join(10)
        self.assertEqual(self._heartbeat_threads(), [])
        lease = read_task_state(run.pipe_root, 'hb').lease_expires_at
        time.sleep(self.LEASE)
        self.assertEqual(read_task_state(run.pipe_root, 'hb').lease_expires_at, lease)

    def test_heartbeat_stops_after_exception(self):
        run, state, token = self._stage()

        def boom(spec, scratch):
            raise RuntimeError('boom')

        thread = self._run_in_thread(run, state, token, boom)
        thread.join(10)
        self.assertEqual(self._heartbeat_threads(), [])
        self.assertIn(read_task_state(run.pipe_root, 'hb').status, ('FAILED_RETRYABLE', 'FAILED_TERMINAL'))

    def test_heartbeat_stops_without_writing_when_reclaimed(self):
        run, state, token = self._stage()
        hb = LeaseHeartbeat(run.pipe_root, 'hb', 'w', token, lease_duration=self.LEASE, interval=0.1)
        update_task_state(run.pipe_root, 'hb', new_status=TaskState.ORPHANED, claimed_by=None, claim_token=None,
                          lease_expires_at=None)
        update_task_state(run.pipe_root, 'hb', new_status=TaskState.PENDING)
        now = time.time()
        other = update_task_state(run.pipe_root, 'hb', new_status=TaskState.CLAIMED, claimed_by='other',
                                  claim_token='other_token', claimed_at=now, lease_expires_at=now + 5)
        with self.assertLogs(worker_logger, level='WARNING') as logs:
            hb.start()
            time.sleep(0.5)
            hb.stop()
        self.assertEqual(len(logs.output), 1)
        self.assertEqual(self._heartbeat_threads(), [])
        final = read_task_state(run.pipe_root, 'hb')
        self.assertEqual(final.lease_expires_at, other.lease_expires_at)
        self.assertEqual(final.claim_token, 'other_token')

    def test_heartbeat_stops_without_writing_when_cancelled(self):
        run, state, token = self._stage()
        hb = LeaseHeartbeat(run.pipe_root, 'hb', 'w', token, lease_duration=self.LEASE, interval=0.1)
        cancelled = update_task_state(run.pipe_root, 'hb', new_status=TaskState.CANCELLED, ended_at=time.time())
        with self.assertLogs(worker_logger, level='WARNING'):
            hb.start()
            time.sleep(0.5)
            hb.stop()
        self.assertEqual(self._heartbeat_threads(), [])
        self.assertEqual(read_task_state(run.pipe_root, 'hb').lease_expires_at, cancelled.lease_expires_at)

    def test_dead_worker_is_orphaned_after_lease(self):
        run, state, token = self._stage()
        hb = LeaseHeartbeat(run.pipe_root, 'hb', 'w', token, lease_duration=self.LEASE, interval=0.1).start()
        time.sleep(0.5)
        hb.stop()
        time.sleep(self.LEASE + 0.3)
        run.reconcile()
        self.assertNotEqual(read_task_state(run.pipe_root, 'hb').claim_token, token)
        self.assertIn(read_task_state(run.pipe_root, 'hb').status, ('ORPHANED', 'PENDING'))


class TestClaimTask(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_claim_test_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_lease_duration(self):
        with patch.dict('arc.scripts.pipe_worker.pipe_settings', {}, clear=True):
            self.assertEqual(get_lease_duration_s(), 3600)

    def test_claim_sets_lease_from_setting(self):
        initialize_task(self.tmpdir, _make_h2o_spec('task_l'))
        with patch.dict('arc.scripts.pipe_worker.pipe_settings', {}, clear=True):
            _, state, _ = claim_task(self.tmpdir, 'worker-1')
        self.assertAlmostEqual(state.lease_expires_at - state.claimed_at, 3600, places=3)

    def test_claims_pending_task(self):
        initialize_task(self.tmpdir, _make_h2o_spec('task_a'))
        task_id, state, token = claim_task(self.tmpdir, 'worker-1')
        self.assertEqual(task_id, 'task_a')
        self.assertEqual(state.status, 'CLAIMED')
        self.assertEqual(state.claimed_by, 'worker-1')
        self.assertIsNotNone(token)
        self.assertEqual(state.claim_token, token)

    def test_skips_completed_and_running(self):
        initialize_task(self.tmpdir, _make_h2o_spec('task_01'))
        now = time.time()
        update_task_state(self.tmpdir, 'task_01', new_status=TaskState.CLAIMED,
                          claimed_by='w0', claim_token='t', claimed_at=now, lease_expires_at=now + 300)
        update_task_state(self.tmpdir, 'task_01', new_status=TaskState.RUNNING, started_at=now)
        update_task_state(self.tmpdir, 'task_01', new_status=TaskState.COMPLETED, ended_at=now)

        initialize_task(self.tmpdir, _make_h2o_spec('task_02'))
        update_task_state(self.tmpdir, 'task_02', new_status=TaskState.CLAIMED,
                          claimed_by='w0', claim_token='t', claimed_at=now, lease_expires_at=now + 300)
        update_task_state(self.tmpdir, 'task_02', new_status=TaskState.RUNNING, started_at=now)

        initialize_task(self.tmpdir, _make_h2o_spec('task_03'))
        task_id, state, token = claim_task(self.tmpdir, 'worker-5')
        self.assertEqual(task_id, 'task_03')

    def test_returns_none_when_no_tasks(self):
        task_id, state, token = claim_task(self.tmpdir, 'worker-1')
        self.assertIsNone(task_id)
        self.assertIsNone(token)

    def test_ignores_orphaned_tasks(self):
        initialize_task(self.tmpdir, _make_h2o_spec('task_orphan'))
        now = time.time()
        update_task_state(self.tmpdir, 'task_orphan', new_status=TaskState.CLAIMED,
                          claimed_by='dead', claim_token='t', claimed_at=now, lease_expires_at=now + 300)
        update_task_state(self.tmpdir, 'task_orphan', new_status=TaskState.ORPHANED)
        task_id, state, token = claim_task(self.tmpdir, 'worker-rescue')
        self.assertIsNone(task_id)


class TestRunTask(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_run_test_')
        # MockAdapter's output.yml is not a real ESS log, so _parse_ess_error
        # would falsely report convergence failure. Skip it for these tests.
        patcher = patch('arc.scripts.pipe_worker._parse_ess_error', return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _claim(self, task_id, worker_id='test-worker'):
        now = time.time()
        token = generate_claim_token()
        state = update_task_state(
            self.tmpdir, task_id, new_status=TaskState.CLAIMED,
            claimed_by=worker_id, claim_token=token,
            claimed_at=now, lease_expires_at=now + 86400)
        return state, token

    def test_successful_execution(self):
        spec = _make_h2o_spec('sp_h2o')
        initialize_task(self.tmpdir, spec)
        state, token = self._claim('sp_h2o')
        run_task(self.tmpdir, 'sp_h2o', state, 'test-worker', token)
        final = read_task_state(self.tmpdir, 'sp_h2o')
        self.assertEqual(final.status, 'COMPLETED')

    def test_result_json_written_on_success(self):
        spec = _make_h2o_spec('sp_result')
        initialize_task(self.tmpdir, spec)
        state, token = self._claim('sp_result')
        run_task(self.tmpdir, 'sp_result', state, 'test-worker', token)
        attempt_dir = get_task_attempt_dir(self.tmpdir, 'sp_result', 0)
        result_path = os.path.join(attempt_dir, 'result.json')
        self.assertTrue(os.path.isfile(result_path))
        with open(result_path) as f:
            result = json.load(f)
        self.assertEqual(result['task_id'], 'sp_result')
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertIsNotNone(result['started_at'])
        self.assertIsNotNone(result['ended_at'])
        for key in ('canonical_output_path', 'exit_code', 'failure_class',
                     'parser_summary', 'result_fields'):
            self.assertIn(key, result)

    def test_result_json_written_on_failure(self):
        """A failing task still produces result.json with status=FAILED."""
        # Create a valid spec, then corrupt the task_family on disk to trigger failure.
        spec = _make_h2o_spec('bad_job')
        initialize_task(self.tmpdir, spec)
        spec_path = os.path.join(self.tmpdir, 'tasks', 'bad_job', 'spec.json')
        with open(spec_path) as f:
            data = json.load(f)
        data['task_family'] = 'nonexistent_type'
        with open(spec_path, 'w') as f:
            json.dump(data, f)
        state, token = self._claim('bad_job')
        run_task(self.tmpdir, 'bad_job', state, 'test-worker', token)
        attempt_dir = get_task_attempt_dir(self.tmpdir, 'bad_job', 0)
        result_path = os.path.join(attempt_dir, 'result.json')
        self.assertTrue(os.path.isfile(result_path))
        with open(result_path) as f:
            result = json.load(f)
        self.assertEqual(result['status'], 'FAILED')
        self.assertIsNotNone(result['failure_class'])

    def test_output_preservation(self):
        spec = _make_h2o_spec('sp_h2o_out')
        initialize_task(self.tmpdir, spec)
        state, token = self._claim('sp_h2o_out')
        run_task(self.tmpdir, 'sp_h2o_out', state, 'test-worker', token)
        attempt_dir = get_task_attempt_dir(self.tmpdir, 'sp_h2o_out', 0)
        calcs_dir = os.path.join(attempt_dir, 'calcs')
        self.assertTrue(os.path.isdir(calcs_dir))

    def test_ownership_with_token(self):
        """If claim_token changes, worker does not overwrite terminal state."""
        spec = _make_h2o_spec('sp_stolen')
        initialize_task(self.tmpdir, spec)
        now = time.time()
        token_a = generate_claim_token()
        update_task_state(self.tmpdir, 'sp_stolen', new_status=TaskState.CLAIMED,
                          claimed_by='worker-A', claim_token=token_a,
                          claimed_at=now, lease_expires_at=now + 86400)
        # Simulate reassignment
        update_task_state(self.tmpdir, 'sp_stolen', new_status=TaskState.ORPHANED)
        update_task_state(self.tmpdir, 'sp_stolen', new_status=TaskState.PENDING,
                          attempt_index=1, claimed_by=None, claim_token=None,
                          claimed_at=None, lease_expires_at=None,
                          started_at=None, ended_at=None,
                          failure_class=None, retry_disposition=None)
        token_b = generate_claim_token()
        update_task_state(self.tmpdir, 'sp_stolen', new_status=TaskState.CLAIMED,
                          claimed_by='worker-B', claim_token=token_b,
                          claimed_at=now + 1, lease_expires_at=now + 86401)
        from arc.scripts.pipe_worker import _verify_ownership
        self.assertFalse(_verify_ownership(self.tmpdir, 'sp_stolen', 'worker-A', token_a))
        self.assertTrue(_verify_ownership(self.tmpdir, 'sp_stolen', 'worker-B', token_b))

    def test_scratch_cleanup(self):
        spec = _make_h2o_spec('sp_clean')
        initialize_task(self.tmpdir, spec)
        state, token = self._claim('sp_clean')
        run_task(self.tmpdir, 'sp_clean', state, 'test-worker', token)
        import glob
        leftover = glob.glob(os.path.join(tempfile.gettempdir(), 'pipe_sp_clean_*'))
        self.assertEqual(len(leftover), 0)

    def test_conf_sp_dispatch(self):
        """conf_sp task family dispatches correctly and produces result.json."""
        spec = _make_h2o_spec('conf_sp_task', task_family='conf_sp')
        initialize_task(self.tmpdir, spec)
        state, token = self._claim('conf_sp_task')
        run_task(self.tmpdir, 'conf_sp_task', state, 'test-worker', token)
        final = read_task_state(self.tmpdir, 'conf_sp_task')
        self.assertEqual(final.status, 'COMPLETED')
        attempt_dir = get_task_attempt_dir(self.tmpdir, 'conf_sp_task', 0)
        result_path = os.path.join(attempt_dir, 'result.json')
        self.assertTrue(os.path.isfile(result_path))
        with open(result_path) as f:
            result = json.load(f)
        self.assertEqual(result['status'], 'COMPLETED')

    def test_conf_opt_dispatch(self):
        """conf_opt task family dispatches correctly."""
        spec = _make_h2o_spec('conf_opt_task', task_family='conf_opt')
        initialize_task(self.tmpdir, spec)
        state, token = self._claim('conf_opt_task')
        run_task(self.tmpdir, 'conf_opt_task', state, 'test-worker', token)
        final = read_task_state(self.tmpdir, 'conf_opt_task')
        self.assertEqual(final.status, 'COMPLETED')

    def test_ts_opt_dispatch(self):
        """ts_opt task family dispatches via opt job_type and produces result.json."""
        spec = _make_h2o_spec('ts_opt_task', task_family='ts_opt')
        initialize_task(self.tmpdir, spec)
        state, token = self._claim('ts_opt_task')
        run_task(self.tmpdir, 'ts_opt_task', state, 'test-worker', token)
        final = read_task_state(self.tmpdir, 'ts_opt_task')
        self.assertEqual(final.status, 'COMPLETED')
        attempt_dir = get_task_attempt_dir(self.tmpdir, 'ts_opt_task', 0)
        self.assertTrue(os.path.isfile(os.path.join(attempt_dir, 'result.json')))

    def test_ts_guess_batch_dispatch(self):
        """ts_guess_batch_method dispatches via tsg job_type. May fail at adapter
        level (mockter doesn't natively support tsg without reactions), but the
        dispatch path itself should route correctly and write result.json."""
        spec = _make_h2o_spec('tsg_task', task_family='ts_guess_batch_method')
        initialize_task(self.tmpdir, spec)
        state, token = self._claim('tsg_task')
        run_task(self.tmpdir, 'tsg_task', state, 'test-worker', token)
        # The task should at least have written result.json (even on failure)
        attempt_dir = get_task_attempt_dir(self.tmpdir, 'tsg_task', 0)
        self.assertTrue(os.path.isfile(os.path.join(attempt_dir, 'result.json')))
        final = read_task_state(self.tmpdir, 'tsg_task')
        # Either COMPLETED (if mockter handled it) or FAILED_* (if adapter rejected tsg)
        self.assertIn(final.status, ('COMPLETED', 'FAILED_RETRYABLE', 'FAILED_TERMINAL'))

    def test_species_sp_dispatch(self):
        """species_sp task family dispatches via sp job_type."""
        spec = _make_h2o_spec('sp_task', task_family='species_sp')
        initialize_task(self.tmpdir, spec)
        state, token = self._claim('sp_task')
        run_task(self.tmpdir, 'sp_task', state, 'test-worker', token)
        final = read_task_state(self.tmpdir, 'sp_task')
        self.assertEqual(final.status, 'COMPLETED')
        attempt_dir = get_task_attempt_dir(self.tmpdir, 'sp_task', 0)
        self.assertTrue(os.path.isfile(os.path.join(attempt_dir, 'result.json')))

    def test_species_freq_dispatch(self):
        """species_freq task family dispatches via freq job_type."""
        spec = _make_h2o_spec('freq_task', task_family='species_freq')
        initialize_task(self.tmpdir, spec)
        state, token = self._claim('freq_task')
        run_task(self.tmpdir, 'freq_task', state, 'test-worker', token)
        final = read_task_state(self.tmpdir, 'freq_task')
        self.assertEqual(final.status, 'COMPLETED')

    def test_irc_dispatch(self):
        """irc task family dispatches via irc job_type."""
        spec = _make_h2o_spec('irc_task', task_family='irc')
        initialize_task(self.tmpdir, spec)
        state, token = self._claim('irc_task')
        run_task(self.tmpdir, 'irc_task', state, 'test-worker', token)
        # IRC may fail at adapter level (mockter may not handle irc natively),
        # but the dispatch route should work and result.json should be written.
        attempt_dir = get_task_attempt_dir(self.tmpdir, 'irc_task', 0)
        self.assertTrue(os.path.isfile(os.path.join(attempt_dir, 'result.json')))

    def test_rotor_scan_1d_dispatch(self):
        """rotor_scan_1d task family dispatches via scan job_type and writes result.json."""
        spec = _make_h2o_spec('scan_task', task_family='rotor_scan_1d')
        initialize_task(self.tmpdir, spec)
        state, token = self._claim('scan_task')
        run_task(self.tmpdir, 'scan_task', state, 'test-worker', token)
        attempt_dir = get_task_attempt_dir(self.tmpdir, 'scan_task', 0)
        self.assertTrue(os.path.isfile(os.path.join(attempt_dir, 'result.json')))

    def test_unsupported_family_fails(self):
        """An unsupported task_family causes FAILED_RETRYABLE."""
        spec = _make_h2o_spec('bad_family')
        initialize_task(self.tmpdir, spec)
        spec_path = os.path.join(self.tmpdir, 'tasks', 'bad_family', 'spec.json')
        with open(spec_path) as f:
            data = json.load(f)
        data['task_family'] = 'unsupported_scan'
        with open(spec_path, 'w') as f:
            json.dump(data, f)
        state, token = self._claim('bad_family')
        run_task(self.tmpdir, 'bad_family', state, 'test-worker', token)
        final = read_task_state(self.tmpdir, 'bad_family')
        self.assertIn(final.status, ('FAILED_RETRYABLE', 'FAILED_TERMINAL'))

    def test_scratch_creation_failure_marks_failed(self):
        """If tempfile.mkdtemp fails (e.g., I/O error), the task is properly
        marked FAILED_RETRYABLE instead of being left stuck in RUNNING."""
        spec = _make_h2o_spec('io_fail')
        initialize_task(self.tmpdir, spec)
        state, token = self._claim('io_fail')
        with patch('arc.scripts.pipe_worker.tempfile.mkdtemp',
                   side_effect=OSError(5, 'Input/output error')):
            run_task(self.tmpdir, 'io_fail', state, 'test-worker', token)
        final = read_task_state(self.tmpdir, 'io_fail')
        self.assertIn(final.status, ('FAILED_RETRYABLE', 'FAILED_TERMINAL'))


class TestWorkerLoop(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_loop_test_')
        patcher = patch('arc.scripts.pipe_worker._parse_ess_error', return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_main_processes_multiple_tasks(self):
        for i in range(3):
            initialize_task(self.tmpdir, _make_h2o_spec(f'task_{i}'))
        main(['--pipe_root', self.tmpdir, '--worker_id', 'worker-loop'])
        for i in range(3):
            state = read_task_state(self.tmpdir, f'task_{i}')
            self.assertEqual(state.status, 'COMPLETED')

    def test_main_no_tasks(self):
        main(['--pipe_root', self.tmpdir, '--worker_id', 'worker-1'])

    def test_no_duplicate_log_handlers(self):
        for i in range(3):
            initialize_task(self.tmpdir, _make_h2o_spec(f'task_log_{i}'))
        main(['--pipe_root', self.tmpdir, '--worker_id', 'worker-log'])
        self.assertLessEqual(len(worker_logger.handlers), 2)


class TestFamilyExtraKwargs(unittest.TestCase):
    """Unit tests for _get_family_extra_kwargs, which maps a task payload to adapter kwargs."""

    def _rotor_spec(self, payload):
        spc = ARCSpecies(label='ethane', smiles='CC')
        return TaskSpec(
            task_id='ethane_scan_r0', task_family='rotor_scan_1d', owner_type='species',
            owner_key='ethane', input_fingerprint='fp', engine='ase',
            level={'method': 'uma-s-1p2'}, required_cores=1, required_memory_mb=512,
            input_payload={'species_dicts': [spc.as_dict()], **payload},
            ingestion_metadata={'rotor_index': 0})

    def test_rotor_scan_forwards_torsions_and_rotor_index(self):
        kwargs = _get_family_extra_kwargs(self._rotor_spec({'torsions': [[2, 0, 1, 5]], 'rotor_index': 0}))
        self.assertEqual(kwargs['torsions'], [[2, 0, 1, 5]])
        self.assertEqual(kwargs['rotor_index'], 0)
        self.assertNotIn('args', kwargs)

    def test_rotor_scan_forwards_scan_res_as_trsh_args(self):
        kwargs = _get_family_extra_kwargs(self._rotor_spec(
            {'torsions': [[2, 0, 1, 5]], 'rotor_index': 0, 'scan_res': 8.0}))
        self.assertEqual(kwargs['args'], {'trsh': {'scan_res': 8.0}})

    def test_rotor_scan_omits_args_when_scan_res_absent(self):
        kwargs = _get_family_extra_kwargs(self._rotor_spec({'torsions': [[2, 0, 1, 5]], 'rotor_index': 0}))
        self.assertNotIn('args', kwargs)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
