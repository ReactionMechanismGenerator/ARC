#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.pipe.pipe_planner module
"""

import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from arc.job.pipe.pipe_coordinator import PipeCoordinator
from arc.job.pipe.pipe_planner import PipePlanner
from arc.level import Level
from arc.species import ARCSpecies


_pipe_patches = []


def setUpModule():
    """Enable pipe mode for all tests in this module."""
    global _pipe_patches
    pipe_vals = {'enabled': True, 'min_tasks': 10, 'max_workers': 100,
                 'max_attempts': 3, 'lease_duration_s': 86400}
    for target in ('arc.job.pipe.pipe_coordinator.pipe_settings',
                    'arc.job.pipe.pipe_planner.pipe_settings'):
        p = patch.dict(target, pipe_vals)
        p.start()
        _pipe_patches.append(p)


def tearDownModule():
    """Restore pipe settings."""
    global _pipe_patches
    for p in _pipe_patches:
        p.stop()
    _pipe_patches.clear()


def _make_mock_sched(project_directory):
    """Create a mock Scheduler with attributes the planner needs."""
    sched = MagicMock()
    sched.project_directory = project_directory
    sched.memory = 14.0
    sched.conformer_opt_level = Level(method='b97d3', basis='6-31+g(d,p)')
    sched.conformer_sp_level = Level(method='wb97xd', basis='def2-tzvp')
    sched.sp_level = Level(method='wb97xd', basis='def2-tzvp')
    sched.freq_level = Level(method='wb97xd', basis='def2-tzvp')
    sched.scan_level = Level(method='wb97xd', basis='def2-tzvp')
    sched.irc_level = Level(method='wb97xd', basis='def2-tzvp')
    sched.ess_settings = {'gaussian': ['server1']}
    sched.job_types = {'conf_opt': True, 'conf_sp': True, 'opt': True,
                       'freq': True, 'sp': True, 'rotors': True}
    spc = ARCSpecies(label='H2O', smiles='O')
    spc.conformers = [{'symbols': ('O',), 'isotopes': (16,),
                       'coords': ((0.0, 0.0, float(i)),)}
                      for i in range(12)]
    spc.conformer_energies = [None] * 12
    spc.rotors_dict = {i: {'torsion': [0, 1, 2, 3], 'success': None}
                       for i in range(12)}
    sched.species_dict = {'H2O': spc}
    sched.output = {'H2O': {'paths': {}, 'job_types': {}}}
    sched.deduce_job_adapter = MagicMock(return_value='gaussian')
    return sched


class TestTryPipeConformers(unittest.TestCase):
    """Tests for PipePlanner.try_pipe_conformers()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_planner_test_')
        self.sched = _make_mock_sched(self.tmpdir)
        self.coord = PipeCoordinator(self.sched)
        self.planner = PipePlanner(self.sched, self.coord)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pipes_when_enough_conformers(self):
        """12 conformers exceeds threshold, all indices should be piped."""
        handled = self.planner.try_pipe_conformers('H2O')
        self.assertEqual(handled, set(range(12)))
        self.assertEqual(len(self.coord.active_pipes), 1)
        run_id = list(self.coord.active_pipes.keys())[0]
        self.assertIn('H2O', run_id)
        self.assertIn('conf_opt', run_id)

    def test_no_pipe_for_few_conformers(self):
        """5 conformers is below threshold."""
        self.sched.species_dict['H2O'].conformers = [None] * 5
        handled = self.planner.try_pipe_conformers('H2O')
        self.assertEqual(handled, set())
        self.assertEqual(len(self.coord.active_pipes), 0)

    def test_no_pipe_for_incore_adapter(self):
        """Incore adapters should not use pipe."""
        self.sched.deduce_job_adapter.return_value = 'torchani'
        handled = self.planner.try_pipe_conformers('H2O')
        self.assertEqual(handled, set())

    def test_task_specs_have_correct_metadata(self):
        """Verify built TaskSpecs have the expected fields."""
        self.planner.try_pipe_conformers('H2O')
        pipe = list(self.coord.active_pipes.values())[0]
        spec = pipe.tasks[0]
        self.assertEqual(spec.task_family, 'conf_opt')
        self.assertEqual(spec.owner_type, 'species')
        self.assertEqual(spec.owner_key, 'H2O')
        self.assertIn('conformer_index', spec.ingestion_metadata)
        self.assertIsNotNone(spec.level)
        self.assertIn('species_dicts', spec.input_payload)


class TestTryPipeConfSp(unittest.TestCase):
    """Tests for PipePlanner.try_pipe_conf_sp()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_planner_confsp_')
        self.sched = _make_mock_sched(self.tmpdir)
        self.coord = PipeCoordinator(self.sched)
        self.planner = PipePlanner(self.sched, self.coord)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pipes_conf_sp(self):
        handled = self.planner.try_pipe_conf_sp('H2O', list(range(12)))
        self.assertEqual(handled, set(range(12)))

    def test_no_pipe_when_disabled(self):
        self.sched.job_types['conf_sp'] = False
        handled = self.planner.try_pipe_conf_sp('H2O', list(range(12)))
        self.assertEqual(handled, set())

    def test_no_pipe_when_same_level(self):
        self.sched.conformer_sp_level = self.sched.conformer_opt_level
        handled = self.planner.try_pipe_conf_sp('H2O', list(range(12)))
        self.assertEqual(handled, set())

    def test_no_pipe_for_empty_indices(self):
        handled = self.planner.try_pipe_conf_sp('H2O', [])
        self.assertEqual(handled, set())


class TestTryPipeTsOpt(unittest.TestCase):
    """Tests for PipePlanner.try_pipe_ts_opt()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_planner_tsopt_')
        self.sched = _make_mock_sched(self.tmpdir)
        self.coord = PipeCoordinator(self.sched)
        self.planner = PipePlanner(self.sched, self.coord)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pipes_ts_opt(self):
        xyzs = [{'symbols': ('O',), 'isotopes': (16,),
                  'coords': ((0.0, 0.0, float(i)),)}
                 for i in range(12)]
        level = Level(method='wb97xd', basis='def2-tzvp')
        handled = self.planner.try_pipe_ts_opt('H2O', xyzs, level)
        self.assertEqual(handled, set(range(12)))
        pipe = list(self.coord.active_pipes.values())[0]
        self.assertEqual(pipe.tasks[0].task_family, 'ts_opt')

    def test_no_pipe_below_threshold(self):
        xyzs = [{'symbols': ('O',), 'isotopes': (16,), 'coords': ((0, 0, 0),)}] * 5
        level = Level(method='wb97xd', basis='def2-tzvp')
        handled = self.planner.try_pipe_ts_opt('H2O', xyzs, level)
        self.assertEqual(handled, set())


class TestTryPipeSpeciesSp(unittest.TestCase):
    """Tests for PipePlanner.try_pipe_species_sp()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_planner_sp_')
        self.sched = _make_mock_sched(self.tmpdir)
        # Add more species to exceed threshold
        for i in range(12):
            lbl = f'spc_{i}'
            self.sched.species_dict[lbl] = ARCSpecies(label=lbl, smiles='O')
            self.sched.output[lbl] = {'paths': {}, 'job_types': {}}
        self.coord = PipeCoordinator(self.sched)
        self.planner = PipePlanner(self.sched, self.coord)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pipes_species_sp(self):
        labels = [f'spc_{i}' for i in range(12)]
        handled = self.planner.try_pipe_species_sp(labels)
        self.assertEqual(handled, set(labels))
        pipe = list(self.coord.active_pipes.values())[0]
        self.assertEqual(pipe.tasks[0].task_family, 'species_sp')

    def test_no_pipe_below_threshold(self):
        handled = self.planner.try_pipe_species_sp(['spc_0', 'spc_1'])
        self.assertEqual(handled, set())


class TestTryPipeIrc(unittest.TestCase):
    """Tests for PipePlanner.try_pipe_irc()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_planner_irc_')
        self.sched = _make_mock_sched(self.tmpdir)
        for i in range(12):
            lbl = f'ts_{i}'
            self.sched.species_dict[lbl] = ARCSpecies(label=lbl, smiles='O', is_ts=True)
        self.coord = PipeCoordinator(self.sched)
        self.planner = PipePlanner(self.sched, self.coord)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pipes_irc(self):
        pairs = [(f'ts_{i}', 'forward') for i in range(12)]
        handled = self.planner.try_pipe_irc(pairs)
        self.assertEqual(handled, set(pairs))
        pipe = list(self.coord.active_pipes.values())[0]
        self.assertEqual(pipe.tasks[0].task_family, 'irc')
        self.assertEqual(pipe.tasks[0].ingestion_metadata['irc_direction'], 'forward')

    def test_no_pipe_when_no_irc_level(self):
        self.sched.irc_level = None
        handled = self.planner.try_pipe_irc([(f'ts_{i}', 'forward') for i in range(12)])
        self.assertEqual(handled, set())


class TestTryPipeRotorScans(unittest.TestCase):
    """Tests for PipePlanner.try_pipe_rotor_scans_1d()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_planner_scan_')
        self.sched = _make_mock_sched(self.tmpdir)
        self.coord = PipeCoordinator(self.sched)
        self.planner = PipePlanner(self.sched, self.coord)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pipes_scans(self):
        handled = self.planner.try_pipe_rotor_scans_1d('H2O', list(range(12)))
        self.assertEqual(handled, set(range(12)))
        pipe = list(self.coord.active_pipes.values())[0]
        self.assertEqual(pipe.tasks[0].task_family, 'rotor_scan_1d')
        self.assertIn('torsions', pipe.tasks[0].input_payload)
        self.assertIn('rotor_index', pipe.tasks[0].ingestion_metadata)

    def test_no_pipe_below_threshold(self):
        handled = self.planner.try_pipe_rotor_scans_1d('H2O', [0, 1, 2])
        self.assertEqual(handled, set())

    def test_no_pipe_when_no_scan_level(self):
        self.sched.scan_level = None
        handled = self.planner.try_pipe_rotor_scans_1d('H2O', list(range(12)))
        self.assertEqual(handled, set())


class TestTryPipeTsg(unittest.TestCase):
    """Tests for PipePlanner.try_pipe_tsg()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pipe_planner_tsg_')
        self.sched = _make_mock_sched(self.tmpdir)
        self.coord = PipeCoordinator(self.sched)
        self.planner = PipePlanner(self.sched, self.coord)
        self.rxn = MagicMock()
        self.rxn.ts_label = 'TS0'
        self.rxn.as_dict.return_value = {'label': 'rxn_1'}
        self.sched.species_dict['TS0'] = ARCSpecies(label='TS0', smiles='O', is_ts=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pipes_tsg_when_enough_same_method(self):
        """10+ instances of the same method triggers pipe."""
        methods = ['heuristics'] * 12
        handled = self.planner.try_pipe_tsg(self.rxn, methods)
        self.assertEqual(handled, {'heuristics'})

    def test_no_pipe_for_few_methods(self):
        """Typical 3-method list stays below threshold."""
        methods = ['heuristics', 'kinbot', 'autotst']
        handled = self.planner.try_pipe_tsg(self.rxn, methods)
        self.assertEqual(handled, set())

    def test_mixed_methods_only_pipe_large_groups(self):
        """Only the method with 12 instances gets piped."""
        methods = ['heuristics'] * 12 + ['kinbot'] * 3
        handled = self.planner.try_pipe_tsg(self.rxn, methods)
        self.assertEqual(handled, {'heuristics'})
        self.assertNotIn('kinbot', handled)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
