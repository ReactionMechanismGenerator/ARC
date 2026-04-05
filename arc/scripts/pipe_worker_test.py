#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.scripts.pipe_worker module
"""

import json
import os
import shutil
import tempfile
import time
import unittest

from arc.job.pipe.pipe_state import (
    TaskState,
    TaskSpec,
    generate_claim_token,
    get_task_attempt_dir,
    initialize_task,
    read_task_state,
    update_task_state,
)
from arc.scripts.pipe_worker import claim_task, run_task, main, logger as worker_logger
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


class TestClaimTask(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_claim_test_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

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


class TestWorkerLoop(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_loop_test_')

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


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
