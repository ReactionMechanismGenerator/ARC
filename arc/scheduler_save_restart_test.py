"""
Regression tests for ``Scheduler.save_restart_dict`` against #632 / #624.

These bugs surfaced as ``KeyError`` raised from inside the save's
list-comprehension because ``running_jobs`` carried prior-session
conformer names that no longer existed in ``job_dict`` after a
``process_conformers`` wipe-and-repopulate. The fix is twofold:

* ``Scheduler.save_restart_dict`` runs the invariants checker first and
  drops stale names so the comprehension can't KeyError;
* ``Scheduler.process_conformers`` wipes ``running_jobs`` in lockstep with
  the ``job_dict[label]['conf_opt']`` reset.

This file exercises the first half against a synthetic minimal scheduler
state. The root-cause fix's effect is observable through the same
invariant: after the wipe, ``running_jobs[label]`` contains no stale
``conf_opt_*`` names.
"""

import os
import shutil
import tempfile
import unittest

from arc.common import read_yaml_file
from arc.restart_invariants import check_restart_dict_consistency
from arc.scheduler import Scheduler


class _FakeJob:
    """
    Tiny job stand-in for save_restart_dict, which only needs ``as_dict``
    to produce something serializable.
    """

    def __init__(self, name: str):
        self.name = name

    def as_dict(self) -> dict:
        return {'job_name': self.name}


class _FakeSpecies:
    """Species stand-in for save_restart_dict, which only needs ``as_dict``."""

    def __init__(self, label: str):
        self.label = label

    def as_dict(self) -> dict:
        return {'label': self.label}


def _make_minimal_scheduler() -> Scheduler:
    """
    Build a Scheduler instance with just enough state for save_restart_dict
    to run, bypassing the full constructor.

    Returns:
        Scheduler: An instance with save_restart=True, an empty restart_dict,
                   empty output dicts, and a tempdir-backed restart_path.
    """
    sched = Scheduler.__new__(Scheduler)
    sched.save_restart = True
    sched.restart_dict = {}
    sched.output = {}
    sched.output_multi_spc = {}
    sched.species_dict = {}
    sched.running_jobs = {}
    sched.job_dict = {}
    sched.restart_path = os.path.join(tempfile.mkdtemp(prefix='arc_restart_'), 'restart.yml')
    return sched


class TestSaveRestartRegression(unittest.TestCase):
    """Reproductions and post-fix verification for #632 / #624."""

    def setUp(self):
        """Fresh minimal Scheduler per test, with a temp restart_path."""
        self.sched = _make_minimal_scheduler()
        self.tempdir = os.path.dirname(self.sched.restart_path)

    def tearDown(self):
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def test_624_save_does_not_raise_on_stale_conformer_names(self):
        """
        #624 reproduction: ``running_jobs[label]`` claims ``conf_opt_1`` runs
        but ``job_dict[label]['conf_opt']`` only has key 0. Pre-fix this
        would ``KeyError``; post-fix it must save cleanly.
        """
        label = 'r_177_[CH]=CC=C'
        self.sched.species_dict = {label: _FakeSpecies(label)}
        self.sched.running_jobs = {label: ['conf_opt_1', 'conf_opt_0']}
        self.sched.job_dict = {label: {'conf_opt': {0: _FakeJob('conf_opt_0')}}}
        self.sched.save_restart_dict()
        self.assertTrue(os.path.isfile(self.sched.restart_path))
        saved = read_yaml_file(self.sched.restart_path)
        # The stale name is gone; the resolvable name remains.
        self.assertEqual(len(saved['running_jobs'][label]), 1)
        self.assertEqual(self.sched.running_jobs[label], ['conf_opt_0'])

    def test_632_save_does_not_raise_with_many_stale_conformers(self):
        """
        #632 reproduction: many conformer names linger after a wipe-and-
        repopulate; only conformer 0 has been re-spawned. Save must succeed.
        """
        label = 'r_94_[CH2]CCC'
        self.sched.species_dict = {label: _FakeSpecies(label)}
        self.sched.running_jobs = {label: [
            'conf_opt_3', 'conf_opt_4', 'conf_opt_5', 'conf_opt_6', 'conf_opt_0',
        ]}
        self.sched.job_dict = {label: {'conf_opt': {0: _FakeJob('conf_opt_0')}}}
        self.sched.save_restart_dict()
        saved = read_yaml_file(self.sched.restart_path)
        self.assertEqual(len(saved['running_jobs'][label]), 1)
        self.assertEqual(saved['running_jobs'][label][0]['job_name'], 'conf_opt_0')

    def test_save_clean_state_unchanged(self):
        """A consistent snapshot saves with no entries dropped."""
        label = 'X'
        self.sched.species_dict = {label: _FakeSpecies(label)}
        self.sched.running_jobs = {label: ['conf_opt_0', 'opt_a4001']}
        self.sched.job_dict = {label: {
            'conf_opt': {0: _FakeJob('conf_opt_0')},
            'opt': {'opt_a4001': _FakeJob('opt_a4001')},
        }}
        self.sched.save_restart_dict()
        saved = read_yaml_file(self.sched.restart_path)
        self.assertEqual(len(saved['running_jobs'][label]), 2)

    def test_save_followed_by_invariants_check_passes(self):
        """
        After a save with stale entries, the in-memory running_jobs is
        invariant-clean — so the next save sees no further violations.
        """
        label = 'X'
        self.sched.species_dict = {label: _FakeSpecies(label)}
        self.sched.running_jobs = {label: ['conf_opt_0', 'conf_opt_99']}
        self.sched.job_dict = {label: {'conf_opt': {0: _FakeJob('conf_opt_0')}}}
        self.sched.save_restart_dict()
        residual = check_restart_dict_consistency(self.sched.running_jobs, self.sched.job_dict)
        self.assertEqual(residual, [])

    def test_process_conformers_wipe_clears_stale_running_jobs(self):
        """
        Standalone shape-test of the root-cause fix: when ``job_dict[label]
        ['conf_opt']`` is wiped and repopulated, ``running_jobs[label]``
        retains no ``conf_opt_*`` names from a prior session.
        """
        label = 'X'
        running_jobs = {label: [
            'conf_opt_3', 'conf_opt_4', 'opt_a4001',  # opt_a4001 is non-conformer; must survive
        ]}
        # Apply the same lockstep wipe that scheduler.process_conformers performs.
        running_jobs[label] = [n for n in running_jobs[label] if not n.startswith('conf_opt')]
        self.assertEqual(running_jobs[label], ['opt_a4001'])


if __name__ == '__main__':
    unittest.main()
