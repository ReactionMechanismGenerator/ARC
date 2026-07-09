#!/usr/bin/env python3
# encoding: utf-8

"""Unit tests for arc.job.zombie — pure helpers and ESS classification."""

import datetime
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from arc.job import zombie


def _stub_job(job_adapter='molpro', job_type='sp', execution_type='queue',
              initial_offset_seconds=zombie.ZOMBIE_GRACE_SECONDS + 3600,
              job_name='sp_a3177', job_id=12345,
              server='server1', remote_path='/remote/no/such/path',
              local_path='/tmp/no/such/path',
              local_path_to_output_file='/tmp/no/such/output.out'):
    return SimpleNamespace(
        job_name=job_name, job_type=job_type, job_id=job_id,
        job_adapter=job_adapter, execution_type=execution_type,
        initial_time=datetime.datetime.now() - datetime.timedelta(seconds=initial_offset_seconds),
        server=server,
        local_path=local_path, local_path_to_output_file=local_path_to_output_file,
        remote_path=remote_path,
    )


class TestEssPeriodicWritersClassification(unittest.TestCase):
    def test_periodic_writers_set(self):
        self.assertEqual(
            zombie.ESS_PERIODIC_WRITERS,
            frozenset({'cfour', 'gaussian', 'molpro', 'orca', 'psi4', 'qchem', 'terachem'}),
        )

    def test_grace_period_default(self):
        self.assertEqual(zombie.ZOMBIE_GRACE_SECONDS, 21600)


class TestIsZombie(unittest.TestCase):
    def test_zombie_when_no_output_after_grace(self):
        job = _stub_job()
        with patch('arc.job.zombie.output_mtime', return_value=None):
            self.assertTrue(zombie.is_zombie(job, server_job_ids=[job.job_id]))

    def test_not_zombie_when_output_fresh(self):
        job = _stub_job()
        fresh = job.initial_time + datetime.timedelta(seconds=2000)
        with patch('arc.job.zombie.output_mtime', return_value=fresh):
            self.assertFalse(zombie.is_zombie(job, server_job_ids=[job.job_id]))

    def test_zombie_when_output_mtime_at_spawn_time(self):
        """An output file whose mtime equals spawn_time means ARC's own input
        write — no ESS progress. Treat as zombie."""
        job = _stub_job()
        with patch('arc.job.zombie.output_mtime', return_value=job.initial_time):
            self.assertTrue(zombie.is_zombie(job, server_job_ids=[job.job_id]))

    def test_grace_period_blocks(self):
        job = _stub_job(initial_offset_seconds=1800)  # 30 min
        with patch('arc.job.zombie.output_mtime', return_value=None):
            self.assertFalse(zombie.is_zombie(job, server_job_ids=[job.job_id]))

    def test_non_periodic_writer_skipped(self):
        job = _stub_job(job_adapter='xtb')
        with patch('arc.job.zombie.output_mtime', return_value=None):
            self.assertFalse(zombie.is_zombie(job, server_job_ids=[job.job_id]))

    def test_incore_skipped(self):
        job = _stub_job(execution_type='incore')
        with patch('arc.job.zombie.output_mtime', return_value=None):
            self.assertFalse(zombie.is_zombie(job, server_job_ids=[job.job_id]))

    def test_queue_done_skipped(self):
        job = _stub_job()
        with patch('arc.job.zombie.output_mtime', return_value=None):
            self.assertFalse(zombie.is_zombie(job, server_job_ids=[]))

    def test_no_initial_time_skipped(self):
        job = _stub_job()
        job.initial_time = None
        with patch('arc.job.zombie.output_mtime', return_value=None):
            self.assertFalse(zombie.is_zombie(job, server_job_ids=[job.job_id]))

    def test_now_argument_overrides_clock(self):
        """Pass an explicit ``now`` to remove wall-clock dependency in tests."""
        job = _stub_job(initial_offset_seconds=0)
        spawn = job.initial_time
        within_grace = spawn + datetime.timedelta(seconds=zombie.ZOMBIE_GRACE_SECONDS - 1)
        past_grace = spawn + datetime.timedelta(seconds=zombie.ZOMBIE_GRACE_SECONDS + 1)
        with patch('arc.job.zombie.output_mtime', return_value=None):
            self.assertFalse(zombie.is_zombie(job, [job.job_id], now=within_grace))
            self.assertTrue(zombie.is_zombie(job, [job.job_id], now=past_grace))

    def test_running_since_supersedes_initial_time(self):
        """A job PENDING for 8 h that only started RUNNING 1 h ago must not be flagged."""
        job = _stub_job(initial_offset_seconds=8 * 3600)
        recently = datetime.datetime.now() - datetime.timedelta(seconds=3600)
        with patch('arc.job.zombie.output_mtime', return_value=None):
            self.assertFalse(zombie.is_zombie(job, [job.job_id], running_since=recently))

    def test_running_since_past_grace_is_flagged(self):
        """No output past the grace measured from running_since → zombie, even for a young job."""
        job = _stub_job(initial_offset_seconds=1800)
        long_ago = datetime.datetime.now() - datetime.timedelta(seconds=zombie.ZOMBIE_GRACE_SECONDS + 3600)
        with patch('arc.job.zombie.output_mtime', return_value=None):
            self.assertTrue(zombie.is_zombie(job, [job.job_id], running_since=long_ago))

    def test_stalled_output_measured_from_running_since(self):
        """An output last touched before running_since means no traffic since the run began."""
        job = _stub_job()
        running_since = datetime.datetime.now() - datetime.timedelta(seconds=zombie.ZOMBIE_GRACE_SECONDS + 60)
        stale = running_since - datetime.timedelta(seconds=600)
        with patch('arc.job.zombie.output_mtime', return_value=stale):
            self.assertTrue(zombie.is_zombie(job, [job.job_id], running_since=running_since))


class TestGraceSettings(unittest.TestCase):
    def test_default_grace(self):
        self.assertEqual(zombie.get_zombie_grace_seconds(), 21600)
        self.assertEqual(zombie.get_zombie_grace_seconds('no_such_server'), 21600)

    def test_global_settings_value_used(self):
        with patch.dict(zombie.settings, {'zombie_grace_seconds': 50}):
            self.assertEqual(zombie.get_zombie_grace_seconds(), 50)
            self.assertEqual(zombie.get_zombie_grace_seconds('no_such_server'), 50)

    def test_per_server_override(self):
        with patch.dict(zombie.settings,
                        {'servers': {'fast': {'cluster_soft': 'Slurm', 'zombie_grace_seconds': 100}}}):
            self.assertEqual(zombie.get_zombie_grace_seconds('fast'), 100)
            self.assertEqual(zombie.get_zombie_grace_seconds('other'), 21600)

    def test_is_zombie_honors_per_server_override(self):
        job = _stub_job(server='fast', initial_offset_seconds=200)
        with patch.dict(zombie.settings, {'servers': {'fast': {'zombie_grace_seconds': 100}}}):
            with patch('arc.job.zombie.output_mtime', return_value=None):
                self.assertTrue(zombie.is_zombie(job, server_job_ids=[job.job_id]))
        with patch.dict(zombie.settings, {'servers': {'fast': {}}}):
            with patch('arc.job.zombie.output_mtime', return_value=None):
                self.assertFalse(zombie.is_zombie(job, server_job_ids=[job.job_id]))


class FakeSSHClientRaises:
    """A stand-in for ``SSHClient`` whose construction fails like a broken SSH connection."""

    def __init__(self, server: str) -> None:
        raise ConnectionError('ssh down')


class FakeSSHClientFileAbsent:
    """A stand-in for ``SSHClient`` where the connection succeeds but no output file exists."""

    def __init__(self, server: str) -> None:
        self.server = server

    def __enter__(self) -> 'FakeSSHClientFileAbsent':
        return self

    def __exit__(self, *args) -> bool:
        return False

    def get_last_modified_time(self, remote_file_path_1: str, remote_file_path_2: str = None):
        return None


class TestRemoteStatFailure(unittest.TestCase):
    def test_output_mtime_raises_on_ssh_failure(self):
        """A failed remote stat must be distinguishable from a missing output file."""
        job = _stub_job()
        with patch('arc.job.zombie.SSHClient', FakeSSHClientRaises):
            with self.assertRaises(zombie.RemoteStatError):
                zombie.output_mtime(job)

    def test_ssh_failure_skips_zombie_check(self):
        """A transient SSH failure must not flag a healthy job as a zombie."""
        job = _stub_job()
        with patch('arc.job.zombie.SSHClient', FakeSSHClientRaises):
            self.assertFalse(zombie.is_zombie(job, server_job_ids=[job.job_id]))

    def test_remote_file_absent_is_still_zombie(self):
        """A successful stat that finds no output file past the grace is a zombie."""
        job = _stub_job()
        with patch('arc.job.zombie.SSHClient', FakeSSHClientFileAbsent):
            self.assertIsNone(zombie.output_mtime(job))
            self.assertTrue(zombie.is_zombie(job, server_job_ids=[job.job_id]))


class TestOutputMtimeLocal(unittest.TestCase):
    def test_local_output_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, 'output.out')
            with open(out_path, 'w') as fh:
                fh.write('x')
            job = _stub_job(server='local', local_path=tmp, local_path_to_output_file=out_path)
            mtime = zombie.output_mtime(job)
            self.assertIsNotNone(mtime)
            self.assertIsInstance(mtime, datetime.datetime)

    def test_local_output_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = _stub_job(server='local', local_path=tmp,
                            local_path_to_output_file=os.path.join(tmp, 'nope.out'))
            self.assertIsNone(zombie.output_mtime(job))

    def test_local_server_none_treated_as_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, 'output.out')
            with open(out_path, 'w') as fh:
                fh.write('x')
            job = _stub_job(server=None, local_path=tmp, local_path_to_output_file=out_path)
            self.assertIsNotNone(zombie.output_mtime(job))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
