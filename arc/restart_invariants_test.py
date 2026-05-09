"""
Tests for the restart-state invariants checker.

Each named bug pattern from #632 / #624 has a test case here, plus the
clean-state, ambiguous, and tsg/standard-job pattern coverage.
"""

import unittest

from arc.restart_invariants import (
    InvariantViolation,
    check_restart_dict_consistency,
    repair_running_jobs,
    resolve_job_dict_path,
)


class _Job:
    """Lightweight stand-in for a job object — invariants only inspect dict membership."""


class TestResolveJobDictPath(unittest.TestCase):
    """Path-resolution helper: maps job name → (job_type_key, sub_key)."""

    def test_conf_opt_path(self):
        """``conf_opt_3`` resolves to ('conf_opt', 3)."""
        self.assertEqual(resolve_job_dict_path('conf_opt_3'), ('conf_opt', 3))

    def test_conf_sp_path(self):
        """``conf_sp_0`` resolves to ('conf_sp', 0)."""
        self.assertEqual(resolve_job_dict_path('conf_sp_0'), ('conf_sp', 0))

    def test_tsg_path(self):
        """``tsg2`` resolves to ('tsg', 2)."""
        self.assertEqual(resolve_job_dict_path('tsg2'), ('tsg', 2))

    def test_standard_job_path(self):
        """``opt_a4001`` resolves to ('opt', 'opt_a4001')."""
        self.assertEqual(resolve_job_dict_path('opt_a4001'), ('opt', 'opt_a4001'))

    def test_freq_job_path(self):
        """Standard freq jobs use the full job name as sub-key."""
        self.assertEqual(resolve_job_dict_path('freq_a4003'), ('freq', 'freq_a4003'))

    def test_malformed_returns_none(self):
        """A name with no underscore and not matching tsg returns None."""
        self.assertIsNone(resolve_job_dict_path('garbage'))


class TestConsistencyCleanCases(unittest.TestCase):
    """No violations expected on consistent snapshots."""

    def test_empty(self):
        """Empty inputs yield no violations."""
        self.assertEqual(check_restart_dict_consistency({}, {}), [])

    def test_single_conformer_consistent(self):
        """One conformer running with matching job_dict entry — clean."""
        job_dict = {'X': {'conf_opt': {0: _Job()}}}
        running = {'X': ['conf_opt_0']}
        self.assertEqual(check_restart_dict_consistency(running, job_dict), [])

    def test_mixed_conformer_and_standard_job(self):
        """Mixed conf_opt + standard opt + tsg — all backed by job_dict."""
        job_dict = {
            'X': {
                'conf_opt': {0: _Job(), 1: _Job()},
                'opt': {'opt_a4001': _Job()},
                'tsg': {2: _Job()},
            },
        }
        running = {'X': ['conf_opt_0', 'conf_opt_1', 'opt_a4001', 'tsg2']}
        self.assertEqual(check_restart_dict_consistency(running, job_dict), [])


class TestConsistencyBugReproductions(unittest.TestCase):
    """Each subtest reproduces a known #632/#624 failure shape."""

    def test_624_stale_conformer_after_restart(self):
        """
        #624: restart loads ``running_jobs`` with prior session names, then
        ``process_conformers`` wipes ``job_dict[label]['conf_opt']`` and
        repopulates incrementally. The first ``save_restart_dict`` after a
        single ``run_job`` sees a partial job_dict but the full old
        running_jobs list — exactly this configuration.
        """
        job_dict = {'r_177_[CH]=CC=C': {'conf_opt': {0: _Job()}}}
        running = {'r_177_[CH]=CC=C': ['conf_opt_1', 'conf_opt_0']}
        violations = check_restart_dict_consistency(running, job_dict)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].label, 'r_177_[CH]=CC=C')
        self.assertEqual(violations[0].job_name, 'conf_opt_1')
        self.assertIn('conf_opt', violations[0].reason)

    def test_632_multiple_stale_conformers(self):
        """
        #632 minimal shape: many conformer names linger after wipe-and-repopulate;
        only conformer 0 has been re-spawned at the moment of save.
        """
        job_dict = {'r_94_[CH2]CCC': {'conf_opt': {0: _Job()}}}
        running = {'r_94_[CH2]CCC': [
            'conf_opt_3', 'conf_opt_4', 'conf_opt_5', 'conf_opt_6', 'conf_opt_0',
        ]}
        violations = check_restart_dict_consistency(running, job_dict)
        self.assertEqual(len(violations), 4)
        self.assertEqual({v.job_name for v in violations},
                         {'conf_opt_3', 'conf_opt_4', 'conf_opt_5', 'conf_opt_6'})

    def test_label_missing_from_job_dict_entirely(self):
        """When job_dict has no entry for the label, every running name is unresolved."""
        running = {'X': ['conf_opt_0', 'opt_a4001']}
        violations = check_restart_dict_consistency(running, {})
        self.assertEqual(len(violations), 2)
        for v in violations:
            self.assertIn('no entry for label', v.reason)

    def test_job_type_bucket_missing(self):
        """Job_dict has the label but lacks the job_type bucket the running name needs."""
        job_dict = {'X': {'opt': {'opt_a4001': _Job()}}}
        running = {'X': ['conf_opt_0']}
        violations = check_restart_dict_consistency(running, job_dict)
        self.assertEqual(len(violations), 1)
        self.assertIn("missing job_type key", violations[0].reason)

    def test_standard_job_with_stale_name(self):
        """A non-conformer job with a name that doesn't match any sub-key."""
        job_dict = {'X': {'freq': {'freq_a4001': _Job()}}}
        running = {'X': ['freq_a9999']}  # stale from prior session
        violations = check_restart_dict_consistency(running, job_dict)
        self.assertEqual(len(violations), 1)
        self.assertIn('missing sub-key', violations[0].reason)

    def test_tsg_index_out_of_range(self):
        """A tsg name pointing past the populated index range."""
        job_dict = {'TS0': {'tsg': {0: _Job()}}}
        running = {'TS0': ['tsg5']}
        violations = check_restart_dict_consistency(running, job_dict)
        self.assertEqual(len(violations), 1)
        self.assertIn('missing sub-key', violations[0].reason)


class TestRepair(unittest.TestCase):
    """The repair drops bad entries and reports them, leaving the snapshot saveable."""

    def test_repair_clean_is_identity(self):
        """Already-consistent input is returned unchanged with no violations."""
        job_dict = {'X': {'conf_opt': {0: _Job()}}}
        running = {'X': ['conf_opt_0']}
        repaired, violations = repair_running_jobs(running, job_dict)
        self.assertEqual(repaired, running)
        self.assertEqual(violations, [])

    def test_repair_drops_unresolved_names(self):
        """Stale conformer names disappear; resolvable ones survive."""
        job_dict = {'X': {'conf_opt': {0: _Job()}}}
        running = {'X': ['conf_opt_0', 'conf_opt_4', 'conf_opt_7']}
        repaired, violations = repair_running_jobs(running, job_dict)
        self.assertEqual(repaired, {'X': ['conf_opt_0']})
        self.assertEqual(len(violations), 2)

    def test_repair_drops_label_when_no_resolvable_names(self):
        """A label with all names unresolved is removed from the output mapping."""
        job_dict = {'X': {'conf_opt': {0: _Job()}}}
        running = {'X': ['conf_opt_0'], 'Y': ['conf_opt_3']}
        repaired, violations = repair_running_jobs(running, job_dict)
        self.assertEqual(repaired, {'X': ['conf_opt_0']})
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].label, 'Y')

    def test_repair_input_unmutated(self):
        """The input running_jobs dict is not modified in-place."""
        job_dict = {'X': {'conf_opt': {0: _Job()}}}
        running = {'X': ['conf_opt_0', 'conf_opt_4']}
        snapshot = dict(running)
        repair_running_jobs(running, job_dict)
        self.assertEqual(running, snapshot)


if __name__ == '__main__':
    unittest.main()
