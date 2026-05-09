"""
Regression tests for ``Scheduler.troubleshoot_conformer_isomorphism`` (#622).

#622 surfaces as a ``KeyError`` raised from inside
``troubleshoot_conformer_isomorphism`` when ARC restarts after all conformer
opt jobs have completed but before isomorphism troubleshooting was performed.
On restart, ``job_dict[label]`` is reconstructed only from *running* jobs;
already-completed conformers leave no entries behind, so
``self.job_dict[label]['conf_opt'][0]`` raises ``KeyError``.

Pre-fix the line ``job = self.job_dict[label]['conf_opt'][0]`` (and the
parallel access in the per-conformer respawn loop) crash with no recovery.
Post-fix the method derives ``software`` from ``conformer_opt_level`` and
respawns each conformer at the troubleshoot level without requiring a live
``Job`` object, so the path is robust to a post-restart empty bucket.
"""

import os
import shutil
import tempfile
import unittest

from arc.level import Level
from arc.scheduler import Scheduler


class _StubSpecies:
    """
    Minimal species stand-in for trsh_conformer_isomorphism — only needs
    ``.conformers``, ``.conformers_before_opt``, ``.is_ts``, ``.mol``,
    ``.label``, ``.charge``.
    """

    class _StubMol:
        def copy(self, deep=False):
            return self

        def to_smiles(self) -> str:
            return 'CCCC'

    def __init__(self, label: str, n_conformers: int):
        self.label = label
        self.is_ts = False
        self.charge = 0
        self.mol = self._StubMol()
        placeholder = {
            'symbols': ('C', 'C', 'C', 'C'),
            'isotopes': (12, 12, 12, 12),
            'coords': ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0),
                       (3.0, 0.0, 0.0), (4.5, 0.0, 0.0)),
        }
        self.conformers = [placeholder] * n_conformers
        self.conformers_before_opt = tuple(self.conformers)


def _make_minimal_scheduler() -> Scheduler:
    """
    Construct a Scheduler with just enough state for
    ``troubleshoot_conformer_isomorphism`` to run, bypassing ``__init__``.

    Returns:
        Scheduler: A bare-bones instance — no project directory, no servers,
                   no real jobs. Use only with in-memory call paths.
    """
    sched = Scheduler.__new__(Scheduler)
    sched.trsh_ess_jobs = True
    sched.allow_nonisomorphic_2d = False
    sched.species_dict = {}
    sched.job_dict = {}
    sched.running_jobs = {}
    sched.output = {}
    sched.restart_dict = {}
    sched.restart_path = os.path.join(tempfile.mkdtemp(prefix='arc_restart_'), 'restart.yml')
    sched.save_restart = False
    sched.run_job_calls = []

    def _record_run_job(**kwargs):
        sched.run_job_calls.append(kwargs)

    sched.run_job = _record_run_job
    sched.conformer_opt_level = Level(method='wb97xd', basis='def2tzvp')
    sched.conformer_opt_level.software = 'gaussian'
    return sched


class TestTroubleshootConformerIsomorphismRegression(unittest.TestCase):
    """#622 reproductions: trsh path must not crash when conf_opt bucket is empty."""

    def setUp(self):
        """Fresh minimal Scheduler per test."""
        self.sched = _make_minimal_scheduler()
        self.tempdir = os.path.dirname(self.sched.restart_path)

    def tearDown(self):
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def test_622_empty_conf_opt_bucket_does_not_crash(self):
        """
        Post-restart shape: ``job_dict[label]`` exists but has no ``conf_opt``
        bucket because no conformer jobs are running. Pre-fix this raises
        ``KeyError: 'conf_opt'`` at the index access.
        """
        label = 'n_butane'
        self.sched.species_dict = {label: _StubSpecies(label, 3)}
        self.sched.job_dict = {label: {}}
        self.sched.output = {label: {'conformers': '', 'job_types': {}}}

        self.sched.troubleshoot_conformer_isomorphism(label=label)

        self.assertEqual(len(self.sched.run_job_calls), 3)
        for call in self.sched.run_job_calls:
            self.assertEqual(call['label'], label)
            self.assertEqual(call['job_type'], 'conf_opt')
            self.assertEqual(call['job_adapter'], 'gaussian')

    def test_622_empty_conf_opt_dict_does_not_crash(self):
        """
        Adjacent shape: ``conf_opt`` bucket is present but empty (e.g. just
        wiped by ``process_conformers``). Pre-fix: ``KeyError: 0``.
        """
        label = 'n_butane'
        self.sched.species_dict = {label: _StubSpecies(label, 3)}
        self.sched.job_dict = {label: {'conf_opt': {}}}
        self.sched.output = {label: {'conformers': '', 'job_types': {}}}

        self.sched.troubleshoot_conformer_isomorphism(label=label)

        self.assertEqual(len(self.sched.run_job_calls), 3)

    def test_622_partial_conf_opt_bucket_handled(self):
        """
        Mixed shape: only conformer 0 survived in ``job_dict``; conformers 1
        and 2 are absent. Trsh must respawn all three from the species's
        ``conformers_before_opt`` rather than KeyErroring on conformer 1.
        """
        label = 'n_butane'
        self.sched.species_dict = {label: _StubSpecies(label, 3)}

        class _StubJob:
            job_adapter = 'gaussian'
            ess_trsh_methods = []

        self.sched.job_dict = {label: {'conf_opt': {0: _StubJob()}}}
        self.sched.output = {label: {'conformers': '', 'job_types': {}}}

        self.sched.troubleshoot_conformer_isomorphism(label=label)

        self.assertEqual(len(self.sched.run_job_calls), 3)

    def test_622_trsh_disabled_short_circuits(self):
        """When ``trsh_ess_jobs`` is False the method returns without touching job_dict."""
        label = 'n_butane'
        self.sched.trsh_ess_jobs = False
        self.sched.species_dict = {label: _StubSpecies(label, 2)}
        self.sched.job_dict = {label: {}}
        self.sched.output = {label: {'conformers': '', 'job_types': {}}}

        self.sched.troubleshoot_conformer_isomorphism(label=label)

        self.assertEqual(self.sched.run_job_calls, [])


if __name__ == '__main__':
    unittest.main()
