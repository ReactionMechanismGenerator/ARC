#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.pipe_state module
"""

import json
import os
import shutil
import tempfile
import threading
import time
import unittest

from arc.job.pipe.pipe_state import (
    TaskState,
    PipeRunState,
    TASK_TRANSITIONS,
    SUPPORTED_TASK_FAMILIES,
    TASK_FAMILY_TO_JOB_TYPE,
    check_valid_transition,
    TaskSpec,
    TaskStateRecord,
    generate_claim_token,
    initialize_task,
    read_task_state,
    update_task_state,
    write_result_json,
)


def _make_spec(task_id='t1', task_family='conf_opt', **overrides):
    defaults = dict(
        task_id=task_id,
        task_family=task_family,
        owner_type='species',
        owner_key='H2O',
        input_fingerprint='fp',
        engine='gaussian',
        level={'method': 'b3lyp', 'basis': '6-31g'},
        required_cores=4,
        required_memory_mb=2048,
        input_payload={'species_dicts': [{'label': 'H2O'}]},
        ingestion_metadata={'conformer_index': 0},
    )
    defaults.update(overrides)
    return TaskSpec(**defaults)


class TestTaskTransitions(unittest.TestCase):

    def test_all_valid_task_transitions(self):
        for src, targets in TASK_TRANSITIONS.items():
            if targets:  # terminal states have empty tuples
                for tgt in targets:
                    check_valid_transition(src, tgt)

    def test_no_self_transitions(self):
        for state in list(TaskState):
            with self.assertRaises(ValueError):
                check_valid_transition(state, state)

    def test_cross_type_raises(self):
        with self.assertRaises(TypeError):
            check_valid_transition(TaskState.PENDING, PipeRunState.CREATED)


class TestTaskSpec(unittest.TestCase):

    def test_conf_opt_roundtrip(self):
        spec = _make_spec(task_family='conf_opt')
        d = spec.as_dict()
        spec2 = TaskSpec.from_dict(json.loads(json.dumps(d)))
        self.assertEqual(spec2.task_family, 'conf_opt')
        self.assertEqual(spec2.owner_key, 'H2O')

    def test_conf_sp_roundtrip(self):
        spec = _make_spec(task_family='conf_sp')
        d = spec.as_dict()
        spec2 = TaskSpec.from_dict(json.loads(json.dumps(d)))
        self.assertEqual(spec2.task_family, 'conf_sp')

    def test_ts_guess_batch_method_roundtrip(self):
        spec = _make_spec(task_family='ts_guess_batch_method')
        d = spec.as_dict()
        spec2 = TaskSpec.from_dict(json.loads(json.dumps(d)))
        self.assertEqual(spec2.task_family, 'ts_guess_batch_method')
        self.assertEqual(spec2.owner_type, 'species')

    def test_ts_opt_roundtrip(self):
        spec = _make_spec(task_family='ts_opt')
        d = spec.as_dict()
        spec2 = TaskSpec.from_dict(json.loads(json.dumps(d)))
        self.assertEqual(spec2.task_family, 'ts_opt')

    def test_species_sp_roundtrip(self):
        spec = _make_spec(task_family='species_sp')
        d = spec.as_dict()
        spec2 = TaskSpec.from_dict(json.loads(json.dumps(d)))
        self.assertEqual(spec2.task_family, 'species_sp')

    def test_species_freq_roundtrip(self):
        spec = _make_spec(task_family='species_freq')
        d = spec.as_dict()
        spec2 = TaskSpec.from_dict(json.loads(json.dumps(d)))
        self.assertEqual(spec2.task_family, 'species_freq')

    def test_irc_roundtrip(self):
        spec = _make_spec(task_family='irc')
        d = spec.as_dict()
        spec2 = TaskSpec.from_dict(json.loads(json.dumps(d)))
        self.assertEqual(spec2.task_family, 'irc')

    def test_rotor_scan_1d_roundtrip(self):
        spec = _make_spec(task_family='rotor_scan_1d')
        d = spec.as_dict()
        spec2 = TaskSpec.from_dict(json.loads(json.dumps(d)))
        self.assertEqual(spec2.task_family, 'rotor_scan_1d')

    def test_supported_families(self):
        for fam in ('conf_opt', 'conf_sp', 'ts_guess_batch_method', 'ts_opt',
                     'species_sp', 'species_freq', 'irc', 'rotor_scan_1d'):
            self.assertIn(fam, SUPPORTED_TASK_FAMILIES)

    def test_family_to_job_type_mapping(self):
        self.assertEqual(TASK_FAMILY_TO_JOB_TYPE['conf_opt'], 'conf_opt')
        self.assertEqual(TASK_FAMILY_TO_JOB_TYPE['conf_sp'], 'conf_sp')
        self.assertEqual(TASK_FAMILY_TO_JOB_TYPE['ts_guess_batch_method'], 'tsg')
        self.assertEqual(TASK_FAMILY_TO_JOB_TYPE['ts_opt'], 'opt')
        self.assertEqual(TASK_FAMILY_TO_JOB_TYPE['species_sp'], 'sp')
        self.assertEqual(TASK_FAMILY_TO_JOB_TYPE['species_freq'], 'freq')
        self.assertEqual(TASK_FAMILY_TO_JOB_TYPE['irc'], 'irc')
        self.assertEqual(TASK_FAMILY_TO_JOB_TYPE['rotor_scan_1d'], 'scan')

    def test_validation_unsupported_family(self):
        with self.assertRaises(ValueError):
            _make_spec(task_family='scan')

    def test_validation_missing_task_family(self):
        with self.assertRaises(ValueError):
            _make_spec(task_family='')

    def test_validation_bad_owner_type(self):
        with self.assertRaises(ValueError):
            _make_spec(owner_type='molecule')

    def test_validation_missing_owner_key(self):
        with self.assertRaises(ValueError):
            _make_spec(owner_key='')

    def test_validation_missing_level(self):
        with self.assertRaises(ValueError):
            _make_spec(level=None)

    def test_validation_missing_input_payload(self):
        with self.assertRaises(ValueError):
            _make_spec(input_payload=None)

    def test_validation_missing_ingestion_metadata(self):
        with self.assertRaises(ValueError):
            _make_spec(ingestion_metadata=None)


class TestTaskStateRecord(unittest.TestCase):

    def test_claim_token_roundtrip(self):
        rec = TaskStateRecord(claim_token='abc123')
        d = rec.as_dict()
        rec2 = TaskStateRecord.from_dict(d)
        self.assertEqual(rec2.claim_token, 'abc123')


class TestGenerateClaimToken(unittest.TestCase):

    def test_tokens_are_unique(self):
        tokens = {generate_claim_token() for _ in range(100)}
        self.assertEqual(len(tokens), 100)


class TestInitializeTask(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_test_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_spec_and_state(self):
        spec = _make_spec(task_id='t1')
        task_dir = initialize_task(self.tmpdir, spec)
        self.assertTrue(os.path.isfile(os.path.join(task_dir, 'spec.json')))
        self.assertTrue(os.path.isfile(os.path.join(task_dir, 'state.json')))

    def test_duplicate_raises(self):
        spec = _make_spec(task_id='dup')
        initialize_task(self.tmpdir, spec)
        with self.assertRaises(FileExistsError):
            initialize_task(self.tmpdir, spec)

    def test_overwrite_allowed(self):
        spec = _make_spec(task_id='dup')
        initialize_task(self.tmpdir, spec, max_attempts=3)
        initialize_task(self.tmpdir, spec, max_attempts=5, overwrite=True)
        state = read_task_state(self.tmpdir, 'dup')
        self.assertEqual(state.max_attempts, 5)


class TestWriteResultJson(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_result_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_writes_and_reads(self):
        result = {'task_id': 't1', 'status': 'COMPLETED'}
        path = write_result_json(self.tmpdir, result)
        with open(path) as f:
            self.assertEqual(json.load(f)['task_id'], 't1')


class TestUpdateTaskState(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_test_')
        initialize_task(self.tmpdir, _make_spec(task_id='t'))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_lifecycle(self):
        now = time.time()
        update_task_state(self.tmpdir, 't', new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='tok',
                          claimed_at=now, lease_expires_at=now + 300)
        update_task_state(self.tmpdir, 't', new_status=TaskState.RUNNING, started_at=now)
        update_task_state(self.tmpdir, 't', new_status=TaskState.COMPLETED, ended_at=now + 10)
        self.assertEqual(read_task_state(self.tmpdir, 't').status, 'COMPLETED')

    def test_claimed_missing_fields(self):
        with self.assertRaises(ValueError):
            update_task_state(self.tmpdir, 't', new_status=TaskState.CLAIMED,
                              claimed_at=time.time(), lease_expires_at=time.time() + 300)

    def test_claimed_missing_claim_token(self):
        now = time.time()
        with self.assertRaises(ValueError):
            update_task_state(self.tmpdir, 't', new_status=TaskState.CLAIMED,
                              claimed_by='w', claimed_at=now, lease_expires_at=now + 300)

    def test_running_missing_started_at(self):
        now = time.time()
        update_task_state(self.tmpdir, 't', new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='tok',
                          claimed_at=now, lease_expires_at=now + 300)
        with self.assertRaises(ValueError):
            update_task_state(self.tmpdir, 't', new_status=TaskState.RUNNING)

    def test_completed_missing_ended_at(self):
        now = time.time()
        update_task_state(self.tmpdir, 't', new_status=TaskState.CLAIMED,
                          claimed_by='w', claim_token='tok',
                          claimed_at=now, lease_expires_at=now + 300)
        update_task_state(self.tmpdir, 't', new_status=TaskState.RUNNING, started_at=now)
        with self.assertRaises(ValueError):
            update_task_state(self.tmpdir, 't', new_status=TaskState.COMPLETED)

    def test_concurrent_claims(self):
        results, errors = [], []
        def claim(wid):
            try:
                update_task_state(self.tmpdir, 't', new_status=TaskState.CLAIMED,
                                  claimed_by=f'w-{wid}', claim_token=generate_claim_token(),
                                  claimed_at=time.time(), lease_expires_at=time.time() + 300)
                results.append(wid)
            except ValueError:
                errors.append(wid)
        threads = [threading.Thread(target=claim, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 4)


class TestFailedEssState(unittest.TestCase):
    """Tests for the FAILED_ESS task state."""

    def test_running_to_failed_ess_is_valid(self):
        self.assertIn(TaskState.FAILED_ESS, TASK_TRANSITIONS[TaskState.RUNNING])

    def test_failed_ess_is_terminal(self):
        self.assertEqual(TASK_TRANSITIONS[TaskState.FAILED_ESS], ())

    def test_transition_out_of_failed_ess_raises(self):
        with self.assertRaises(ValueError):
            check_valid_transition(TaskState.FAILED_ESS, TaskState.PENDING)

    def test_failed_ess_not_retried_by_reconcile(self):
        """FAILED_ESS tasks should not be reset to PENDING by reconcile."""
        from arc.job.pipe.pipe_run import PipeRun
        tmpdir = tempfile.mkdtemp(prefix='pipe_ess_noretry_')
        try:
            spec = TaskSpec(
                task_id='t_ess', task_family='conf_opt', owner_type='species',
                owner_key='spc', input_fingerprint='fp', engine='mockter',
                level={'method': 'm'}, required_cores=1, required_memory_mb=1024,
                input_payload={}, ingestion_metadata={})
            pipe = PipeRun(project_directory=tmpdir, run_id='noretry',
                           tasks=[spec], cluster_software='slurm',
                           pipe_root=os.path.join(tmpdir, 'calcs', 'pipe_test_0'))
            pipe.stage()
            now = time.time()
            update_task_state(pipe.pipe_root, 't_ess', new_status=TaskState.CLAIMED,
                              claimed_by='w', claim_token='t', claimed_at=now, lease_expires_at=now+300)
            update_task_state(pipe.pipe_root, 't_ess', new_status=TaskState.RUNNING, started_at=now)
            update_task_state(pipe.pipe_root, 't_ess', new_status=TaskState.FAILED_ESS,
                              ended_at=now, failure_class='ess_error')
            counts = pipe.reconcile()
            self.assertEqual(counts[TaskState.FAILED_ESS.value], 1)
            self.assertEqual(counts[TaskState.PENDING.value], 0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_failed_ess_counts_as_terminal(self):
        """FAILED_ESS should count toward terminal total so the pipe run completes."""
        from arc.job.pipe.pipe_run import PipeRun
        tmpdir = tempfile.mkdtemp(prefix='pipe_ess_terminal_')
        try:
            spec = TaskSpec(
                task_id='t_ess', task_family='conf_opt', owner_type='species',
                owner_key='spc', input_fingerprint='fp', engine='mockter',
                level={'method': 'm'}, required_cores=1, required_memory_mb=1024,
                input_payload={}, ingestion_metadata={})
            pipe = PipeRun(project_directory=tmpdir, run_id='term_test',
                           tasks=[spec], cluster_software='slurm',
                           pipe_root=os.path.join(tmpdir, 'calcs', 'pipe_test_0'))
            pipe.stage()
            now = time.time()
            update_task_state(pipe.pipe_root, 't_ess', new_status=TaskState.CLAIMED,
                              claimed_by='w', claim_token='t', claimed_at=now, lease_expires_at=now+300)
            update_task_state(pipe.pipe_root, 't_ess', new_status=TaskState.RUNNING, started_at=now)
            update_task_state(pipe.pipe_root, 't_ess', new_status=TaskState.FAILED_ESS,
                              ended_at=now, failure_class='ess_error')
            pipe.reconcile()
            from arc.job.pipe.pipe_state import PipeRunState
            self.assertIn(pipe.status, (PipeRunState.COMPLETED_PARTIAL, PipeRunState.COMPLETED))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
