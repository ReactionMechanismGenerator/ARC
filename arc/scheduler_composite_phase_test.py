"""
Regression tests for ``Scheduler._restart_spawn_for_composite_species`` (#358).

#358 surfaces when ARC restarts on a composite-method run before composite
ever started. The pre-fix elif branch matched any state where
``'composite' not in job_dict[label]`` — including "composite never ran" —
and spawned freq under the comment "composite is done". Freq then ran on a
species without a converged composite geometry.

The fix gates the post-composite spawn on real completion: either
``output[label]['job_types']['composite']`` is True or
``output[label]['paths']['composite']`` is populated. These markers are
both set atomically when composite finishes (see
``parse_composite_geo`` /``check_composite_job``), so they are the
authoritative completion signal.
"""

import os
import shutil
import tempfile
import unittest

from arc.scheduler import Scheduler


class _StubSpecies:
    """
    Minimal species stand-in for the composite-restart spawn path. Carries
    only the attributes the method reads.
    """

    def __init__(self, label: str, is_ts: bool = False, number_of_atoms: int = 6):
        self.label = label
        self.is_ts = is_ts
        self.number_of_atoms = number_of_atoms
        self.irc_label = None


def _make_scheduler() -> Scheduler:
    """
    Build a Scheduler with just enough state to drive the composite-restart
    spawn path without running real jobs.

    Returns:
        Scheduler: An instance whose ``run_composite_job``, ``run_freq_job``,
                   and ``run_scan_jobs`` are recorded into a list rather
                   than executed.
    """
    sched = Scheduler.__new__(Scheduler)
    sched.composite_method = 'cbs-qb3'
    sched.job_types = {'rotors': False, 'freq': True}
    sched.species_dict = {}
    sched.job_dict = {}
    sched.running_jobs = {}
    sched.output = {}
    sched.spawn_calls = []

    def _record(name):
        def _f(label, *args, **kwargs):
            sched.spawn_calls.append((name, label))
        return _f

    sched.run_composite_job = _record('composite')
    sched.run_freq_job = _record('freq')
    sched.run_scan_jobs = _record('scan')
    return sched


class TestRestartSpawnForCompositeSpecies(unittest.TestCase):
    """#358: freq must not spawn before composite has actually completed."""

    def setUp(self):
        """Fresh scheduler with one species per test, in a tempdir for the geo path."""
        self.sched = _make_scheduler()
        self.tempdir = tempfile.mkdtemp(prefix='arc_composite_test_')
        self.label = 'CH3OH'
        self.sched.species_dict[self.label] = _StubSpecies(self.label, number_of_atoms=6)
        self.sched.job_dict[self.label] = {}
        self.sched.output[self.label] = {
            'job_types': {'composite': False, 'freq': False, 'opt': False, 'sp': False, 'fine': False},
            'paths': {'geo': '', 'composite': '', 'freq': '', 'sp': ''},
        }

    def tearDown(self):
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def test_composite_never_ran_spawns_composite(self):
        """No composite output and no in-flight composite job — must spawn composite, not freq."""
        self.sched._restart_spawn_for_composite_species(self.sched.species_dict[self.label])
        self.assertEqual(self.sched.spawn_calls, [('composite', self.label)])

    def test_358_composite_never_ran_must_not_spawn_freq(self):
        """
        Pre-fix this case spawned freq because the elif only checked that
        composite wasn't running, not that it had ever finished. The geo
        path is set to a real file so the first ``if`` doesn't match (its
        guard checks ``not os.path.isfile(...)``); without the completion
        gate the elif matched and freq leaked through.
        """
        geo_path = os.path.join(self.tempdir, 'geometry.xyz')
        with open(geo_path, 'w') as fh:
            fh.write('placeholder\n')
        self.sched.output[self.label]['paths']['geo'] = geo_path

        self.sched._restart_spawn_for_composite_species(self.sched.species_dict[self.label])

        self.assertNotIn(('freq', self.label), self.sched.spawn_calls)

    def test_composite_done_via_output_flag_spawns_freq(self):
        """When ``output['job_types']['composite']`` is True, spawn freq."""
        self.sched.output[self.label]['job_types']['composite'] = True
        self.sched.output[self.label]['paths']['composite'] = '/tmp/composite.log'

        self.sched._restart_spawn_for_composite_species(self.sched.species_dict[self.label])

        self.assertEqual(self.sched.spawn_calls, [('freq', self.label)])

    def test_composite_running_does_nothing(self):
        """A composite job in flight (present in job_dict) blocks every spawn."""
        self.sched.job_dict[self.label]['composite'] = {'composite_a4001': object()}

        self.sched._restart_spawn_for_composite_species(self.sched.species_dict[self.label])

        self.assertEqual(self.sched.spawn_calls, [])

    def test_freq_already_running_skipped(self):
        """Composite done, but freq is in flight — freq must not spawn again."""
        self.sched.output[self.label]['job_types']['composite'] = True
        self.sched.output[self.label]['paths']['composite'] = '/tmp/composite.log'
        self.sched.job_dict[self.label]['freq'] = {'freq_a4002': object()}

        self.sched._restart_spawn_for_composite_species(self.sched.species_dict[self.label])

        self.assertEqual(self.sched.spawn_calls, [])

    def test_monoatomic_skips_freq(self):
        """Composite done on a monoatomic species — freq is not applicable."""
        self.sched.species_dict[self.label].number_of_atoms = 1
        self.sched.output[self.label]['job_types']['composite'] = True
        self.sched.output[self.label]['paths']['composite'] = '/tmp/composite.log'

        self.sched._restart_spawn_for_composite_species(self.sched.species_dict[self.label])

        self.assertEqual(self.sched.spawn_calls, [])

    def test_irc_label_skips_post_composite_jobs(self):
        """When the species is an IRC twin, no freq/scan spawn even after composite is done."""
        self.sched.species_dict[self.label].irc_label = 'CH3OH_irc_forward'
        self.sched.output[self.label]['job_types']['composite'] = True
        self.sched.output[self.label]['paths']['composite'] = '/tmp/composite.log'

        self.sched._restart_spawn_for_composite_species(self.sched.species_dict[self.label])

        self.assertEqual(self.sched.spawn_calls, [])

    def test_rotors_enabled_spawns_scan_after_composite(self):
        """When rotors are requested and composite is done, run_scan_jobs is invoked."""
        self.sched.job_types['rotors'] = True
        self.sched.output[self.label]['job_types']['composite'] = True
        self.sched.output[self.label]['paths']['composite'] = '/tmp/composite.log'

        self.sched._restart_spawn_for_composite_species(self.sched.species_dict[self.label])

        self.assertIn(('freq', self.label), self.sched.spawn_calls)
        self.assertIn(('scan', self.label), self.sched.spawn_calls)


if __name__ == '__main__':
    unittest.main()
