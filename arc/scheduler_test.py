#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.scheduler module
"""

import copy
import csv
import logging
import shlex
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import sys
from types import SimpleNamespace


import arc.job.adapters.ts.autotst_ts as autotst_ts
import arc.job.adapters.ts.gcn_ts as gcn_ts
import arc.job.adapters.ts.goflow_ts as goflow_ts
import arc.job.adapters.ts.kinbot_ts as kinbot_ts
import arc.job.adapters.ts.rits_ts as rits_ts
import arc.job.adapters.ts.xtb_gsm as xtb_gsm
import arc.parser.parser as parser
import arc.settings.settings as repo_settings
from arc.checks.common import (IRC_CAUSE_COMPARISON_INCONCLUSIVE, IRC_CAUSE_ENDPOINT_OPT_NOT_CONVERGED,
                               IRC_CAUSE_JOB_FAILED, IRC_JOB_FAILED_WARNING, NMD_ACCEPTED_ON_IRC_WARNING,
                               NMD_NEAR_MISS_KEY, TS_IRC_NOT_CONFIRMED_MARKER, get_ts_validation_comment)
from arc.checks.nmd_test import make_c2h5cl_rxn
from arc.checks.ts import check_ts
from arc.common import ARC_PATH, ARC_TESTING_PATH, almost_equal_coords_lists, initialize_job_types, read_yaml_file, \
    save_yaml_file
from arc.exceptions import (DependencyError, InputError, QueueQueryError, ReactionError, ServerError,
                            SettingsError)
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import (adopted_reference_is_unrestricted, default_incore_adapters,
                                      derived_instability_breaks_spin_symmetry, is_restricted,
                                      REFERENCE_AGNOSTIC_METHOD_TYPES, REFERENCE_CHANGE_AVAILABLE_KEY,
                                      ts_adapters_by_rmg_family, ts_adapters_for_unknown_unimolecular)
from arc.job.factory import job_factory
from arc.job.pipe.pipe_coordinator import PipeCoordinator
from arc.job.pipe.pipe_run import build_rotor_scan_1d_tasks
from arc.job.pipe.pipe_state import TaskState, initialize_task, update_task_state
from arc.job.ssh import SSHClient
from arc.job.trsh import determine_ess_status
from arc.level import Level, get_freq_level_for_composite_method
from arc.output import _statmech_to_dict, _ts_checks_to_dict, write_output_yml
from arc.plotter import save_conformers_file
from arc.processor import not_verified_causes
from arc.scheduler import (COLLAPSED_REFERENCE_MESSAGE, CONFORMER_ISOMORPHISM_TRSH_EXHAUSTED,
                           CONFORMER_ISOMORPHISM_TRSH_NOTE,
                           CONFORMER_SET_REPLACED_NOTE, DEFER, INVALID_ANALYTIC_FREQ_MESSAGE, MAX_S_SQUARED_DEVIATION,
                           MIXED_SCF_REFERENCE_MESSAGE, QUEUE_QUERY_COOLDOWN, REFUSE,
                           REFUSED_CONFORMER_TRSH_RECOVERY_MESSAGE, SUBMIT,
                           SPIN_CONTAMINATION_MESSAGE, STABILITY_ANALYSIS_ADAPTERS,
                           SYMMETRY_BREAKING_ADAPTERS, UNREACHABLE_REFERENCE_MESSAGE,
                           UNREADABLE_CONFORMER_TRSH_LEVEL_MESSAGE, Scheduler, SchedulerError,
                           exception_summary_line, get_conformer_isomorphism_trsh_history, get_conformer_trsh_levels,
                           get_respawned_conformer_trsh_methods, irc_ended_usably, is_same_conformer,
                           job_level_signature, outage_record, parse_job_e_elect, species_has_freq,
                           species_has_geo, species_has_sp, species_has_sp_and_freq, tsg_method_matches_adapter)
from arc.imports import settings
from arc.reaction import ARCReaction
from arc.species.conformers_test import (C2H5_XYZ, C2H6_XYZ, CH3_XYZ, CH4_XYZ, NC3H7OO_XYZ, QOOH_XYZ, TS0_XYZ,
                                         TS1_XYZ)
from arc.species.conformers import find_internal_rotors
from arc.species.converter import str_to_xyz
from arc.species.species import REACTIVE_RING_INVALIDATION_REASON, ARCSpecies, TSGuess
from arc.species.vectors import calculate_dihedral_angle
from arc.statmech.arkane import ROTOR_FALLBACK_SCAN_FAILED


default_levels_of_theory = settings['default_levels_of_theory']


class TestHasPendingPipeWork(unittest.TestCase):
    """Tests for Scheduler.has_pending_pipe_work()."""

    def test_has_pending_pipe_work(self):
        """
        A species routed to pipe mode holds no running_jobs entries, so it must be reported as
        still busy until every pending batch and active pipe run has released it. Otherwise the
        main loop drops its label and check_all_done() never runs for it.
        """
        label = 'spc_under_test'
        sched = MagicMock()
        sched._pending_pipe_sp = set()
        sched._pending_pipe_freq = set()
        sched._pending_pipe_irc = list()
        sched._pending_pipe_conf_sp = dict()
        sched.active_pipes = dict()
        self.assertFalse(Scheduler.has_pending_pipe_work(sched, label))

        sched._pending_pipe_sp.add(label)
        self.assertTrue(Scheduler.has_pending_pipe_work(sched, label))
        sched._pending_pipe_sp.clear()

        sched._pending_pipe_freq.add(label)
        self.assertTrue(Scheduler.has_pending_pipe_work(sched, label))
        sched._pending_pipe_freq.clear()

        sched._pending_pipe_irc.append((label, 'forward'))
        self.assertTrue(Scheduler.has_pending_pipe_work(sched, label))
        sched._pending_pipe_irc.clear()

        sched._pending_pipe_conf_sp[label] = {0, 1}
        self.assertTrue(Scheduler.has_pending_pipe_work(sched, label))
        sched._pending_pipe_conf_sp.clear()

        pipe = MagicMock()
        pipe.tasks = [MagicMock(owner_key=label)]
        sched.active_pipes['run_0'] = pipe
        self.assertTrue(Scheduler.has_pending_pipe_work(sched, label))
        self.assertFalse(Scheduler.has_pending_pipe_work(sched, 'a_label_no_pipe_holds'))


class TestScheduler(unittest.TestCase):
    """
    Contains unit tests for the Scheduler class
    """
    def make_project_directory(self, name):
        """
        Create a unique project directory for the running test and schedule its removal.

        Args:
            name (str): A descriptive name, used as the directory name prefix.

        Returns:
            str: The path of the created directory.
        """
        projects_path = os.path.join(ARC_PATH, 'Projects')
        os.makedirs(projects_path, exist_ok=True)
        project_directory = tempfile.mkdtemp(prefix=f'{name}_', dir=projects_path)
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        return project_directory

    def setUp(self):
        """
        A method that is run before each unit test in this class.

        The fixtures below are rebuilt per test: many of these tests mutate the schedulers,
        the species and the jobs they are handed, so a fixture shared across tests (or across
        xdist workers, each of which runs its own class-level setup) would make the outcome of
        one test depend on which other tests ran before it.
        """
        self.maxDiff = None
        self.ess_settings = {'gaussian': ['server1'], 'molpro': ['server2', 'server1'], 'qchem': ['server1']}
        self.project_directory = self.make_project_directory('arc_project_for_testing_delete_after_usage3')
        xyz1 = str_to_xyz("""C      -0.57422867   -0.01669771    0.01229213
N       0.82084044    0.08279104   -0.37769346
H      -1.05737005   -0.84067772   -0.52007494
H      -1.10211468    0.90879867   -0.23383011
H      -0.66133128   -0.19490562    1.08785111
H       0.88047852    0.26966160   -1.37780789
H       1.27889520   -0.81548721   -0.22940984""")
        self.spc1 = ARCSpecies(label='methylamine', smiles='CN', xyz=xyz1)
        self.spc2 = ARCSpecies(label='C2H6', smiles='CC')
        xyz3 = """C       1.11424367   -0.01231165   -0.11493630
C      -0.07257945   -0.17830906   -0.16010022
O      -1.38500471   -0.36381519   -0.20928090
H       2.16904830    0.12689206   -0.07152274
H      -1.82570782    0.42754384   -0.56130718"""
        self.spc3 = ARCSpecies(label='CtripCO', smiles='C#CO', xyz=xyz3)
        self.job1 = job_factory(job_adapter='gaussian', project='project_test', ess_settings=self.ess_settings,
                                species=[self.spc1], xyz=xyz1, job_type='conf_opt',
                                conformer=0, level=Level(repr={'method': 'b97-d3', 'basis': '6-311+g(d,p)'}),
                                project_directory=self.project_directory, job_num=101)
        self.job2 = job_factory(job_adapter='gaussian', project='project_test', ess_settings=self.ess_settings,
                                species=[self.spc1], xyz=xyz1, job_type='conf_opt',
                                conformer=1, level=Level(repr={'method': 'b97-d3', 'basis': '6-311+g(d,p)'}),
                                project_directory=self.project_directory, job_num=102)
        self.job3 = job_factory(job_adapter='qchem', project='project_test', ess_settings=self.ess_settings,
                                species=[self.spc2], job_type='freq',
                                level=Level(repr={'method': 'wb97x-d3', 'basis': '6-311+g(d,p)'}),
                                project_directory=self.project_directory, job_num=103)
        self.job4 = job_factory(job_adapter='gaussian', project='project_test_4', ess_settings=self.ess_settings,
                                species=[self.spc1], xyz=xyz1, job_type='scan', torsions=[[3, 1, 2, 6]], rotor_index=0,
                                level=Level(repr={'method': 'b3lyp', 'basis': 'cbsb7'}),
                                project_directory=self.project_directory, job_num=104)
        self.job_types1 = {'conf_opt': True,
                           'conf_sp': False,
                           'opt': True,
                           'fine': False,
                           'freq': True,
                           'sp': True,
                           'rotors': False,
                           'orbitals': False,
                           'lennard_jones': False,
                           }
        self.job_types2 = {'conf_opt': True,
                           'conf_sp': False,
                           'opt': True,
                           'fine': False,
                           'freq': True,
                           'sp': True,
                           'rotors': True,
                           }
        self.sched1 = Scheduler(project='project_test_1', ess_settings=self.ess_settings,
                                species_list=[self.spc1, self.spc2, self.spc3],
                                composite_method=None,
                                conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                                opt_level=Level(repr=default_levels_of_theory['opt']),
                                freq_level=Level(repr=default_levels_of_theory['freq']),
                                sp_level=Level(repr=default_levels_of_theory['sp']),
                                scan_level=Level(repr=default_levels_of_theory['scan']),
                                ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                                project_directory=self.project_directory,
                                testing=True,
                                job_types=self.job_types1,
                                orbitals_level=default_levels_of_theory['orbitals'],
                                adaptive_levels=None,
                                )
        self.sched2 = Scheduler(project='project_test_2', ess_settings=self.ess_settings,
                                species_list=[self.spc1, self.spc2, self.spc3],
                                composite_method=None,
                                conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                                opt_level=Level(repr=default_levels_of_theory['opt']),
                                freq_level=Level(repr=default_levels_of_theory['freq']),
                                sp_level=Level(repr=default_levels_of_theory['sp']),
                                scan_level=Level(repr=default_levels_of_theory['scan']),
                                ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                                project_directory=self.project_directory,
                                testing=True,
                                job_types=self.job_types1,
                                orbitals_level=default_levels_of_theory['orbitals'],
                                adaptive_levels=None,
                                )
        self.sched3 = Scheduler(project='project_test_4', ess_settings=self.ess_settings,
                                species_list=[self.spc1],
                                composite_method=Level(repr='CBS-QB3'),
                                conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                                opt_level=Level(repr=get_freq_level_for_composite_method('cbs-qb3')),
                                freq_level=Level(repr=get_freq_level_for_composite_method('cbs-qb3')),
                                scan_level=Level(repr=default_levels_of_theory['scan_for_composite']),
                                ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                                project_directory=self.project_directory,
                                testing=True,
                                job_types=self.job_types2,
                                )

    def test_conformers(self):
        """Test the parse_conformer_energy() and determine_most_stable_conformer() methods"""
        label = 'methylamine'
        self.job1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'methylamine_conformer_0.out')
        self.job1.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]
        self.job2.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'methylamine_conformer_1.out')
        self.job2.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]
        self.sched1.job_dict[label] = dict()
        self.sched1.job_dict[label]['conf_opt'] = dict()
        self.sched1.job_dict[label]['conf_opt'][0] = self.job1
        self.sched1.job_dict[label]['conf_opt'][1] = self.job2
        self.sched1.species_dict[label].conformer_energies = [None, None]
        self.sched1.species_dict[label].conformers = [None, None]
        self.sched1.parse_conformer(job=self.job1, label=label, i=0)
        self.sched1.parse_conformer(job=self.job2, label=label, i=1)
        expecting = [-251596.4435088726, -254221.9433698632]
        self.assertAlmostEqual(self.sched1.species_dict[label].conformer_energies[0], expecting[0], 5)
        self.assertAlmostEqual(self.sched1.species_dict[label].conformer_energies[1], expecting[1], 5)
        self.sched1.species_dict[label].conformers[0] = parser.parse_geometry(log_file_path=self.job1.local_path_to_output_file)
        self.sched1.species_dict[label].conformers[1] = parser.parse_geometry(log_file_path=self.job2.local_path_to_output_file)

        self.sched1.determine_most_stable_conformer(label=label)
        expecting = {'symbols': ('N', 'C', 'H', 'H', 'H', 'H', 'H'), 'isotopes': (14, 12, 1, 1, 1, 1, 1),
                     'coords': ((-0.7419989889, -0.1327547549, 0.0), (0.7023470134, 0.0158023979, 0.0),
                                (0.9803673385, 1.0735720944, 0.0), (1.1309109832, -0.4595567954, 0.886650896),
                                (1.1309109832, -0.4595567954, -0.886650896), (-1.131139079, 0.3400036467, 0.8147241874),
                                (-1.131139079, 0.3400036467, -0.8147241874))}
        self.assertTrue(almost_equal_coords_lists(self.sched1.species_dict[label].initial_xyz, expecting))
        methylamine_conf_path = os.path.join(self.sched1.project_directory, 'output', 'Species', 'methylamine',
                                             'geometry', 'conformers', 'conformers_after_optimization.txt')
        self.assertTrue(os.path.isfile(methylamine_conf_path))
        with open(methylamine_conf_path, 'r') as f:
            lines = f.readlines()
        self.assertTrue('Conformers for methylamine, optimized at the wb97xd/def2svp level' in lines[0])
        self.assertEqual(lines[11], 'SMILES: CN\n')
        self.assertTrue('Relative Energy:' in lines[12])
        self.assertEqual(lines[16][0], 'N')

        self.sched1.output['C2H6'] = {'info': '',
                                      'paths': {'composite': '', 'freq': '', 'geo': ''},
                                      'isomorphism': '',
                                      'warnings': '',
                                      'errors': '',
                                      'job_types': {'opt': False, 'composite': False, 'sp': False, 'fine': False,
                                                    'freq': False, 'conf_opt': False, 'conf_sp': False},
                                      'convergence': False, 'conformers': '', 'restart': ''}
        self.sched1.run_conformer_jobs()
        save_conformers_file(project_directory=self.sched1.project_directory,
                             label='C2H6',
                             xyzs=self.sched1.species_dict['C2H6'].conformers,
                             level_of_theory=Level(method='CBS-QB3'),
                             multiplicity=1,
                             charge=0,
                             before_optimization=True,)
        c2h6_conf_path = os.path.join(self.sched1.project_directory, 'output', 'Species', 'C2H6', 'geometry',
                                      'conformers', 'conformers_before_optimization.txt')
        self.assertTrue(os.path.isfile(c2h6_conf_path))
        with open(c2h6_conf_path, 'r') as f:
            lines = f.readlines()
        self.assertEqual(lines[0], 'Conformers for C2H6, computed using a force field:\n')
        self.assertEqual(lines[2], 'conformer 0:\n')
        self.assertEqual(lines[3][0], 'C')
        self.assertEqual(lines[11], '\n')
        self.assertEqual(lines[12], 'SMILES: CC\n')

    def test_restore_running_jobs_conformer_reconnects(self):
        """Restarting with a live conformer job must reconnect to it, not crash.

        Regression for the restore-path job-name contract. During normal operation a running
        conformer job is stored in ``running_jobs`` as ``'{job_type}_{i}'`` (e.g. ``'conf_opt_0'``),
        and every consumer of ``running_jobs`` parses that format. ``restore_running_jobs`` used to
        emit the fossil ``'conformer{i}'`` instead, which ``get_i_from_job_name`` returns ``None``
        for; the first scheduling sweep after a restart (``get_completed_incore_jobs``) then fell
        into its fallback branch, derived an empty job-type from the underscore-less name, and died
        with ``KeyError: ''``. This drives a restart payload carrying a live conformer job through
        the real restore + sweep path and asserts the reconnection instead of the crash.
        """
        label = 'methylamine'
        xyz = """C      -0.57422867   -0.01669771    0.01229213
N       0.82084044    0.08279104   -0.37769346
H      -1.05737005   -0.84067772   -0.52007494
H      -1.10211468    0.90879867   -0.23383011
H      -0.66133128   -0.19490562    1.08785111
H       0.88047852    0.26966160   -1.37780789
H       1.27889520   -0.81548721   -0.22940984"""
        spc = ARCSpecies(label=label, smiles='CN', xyz=xyz)
        sched = Scheduler(project='project_test_restore_conf', ess_settings=self.ess_settings,
                          species_list=[spc], composite_method=None,
                          conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          scan_level=Level(repr=default_levels_of_theory['scan']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=self.project_directory, testing=True,
                          job_types=self.job_types1,
                          orbitals_level=default_levels_of_theory['orbitals'], adaptive_levels=None)
        # Two live conformer jobs -- a conf_opt and a conf_sp -- serialized exactly as ARC writes
        # them into the restart file. conf_sp jobs can equally be in flight during a restart, and are
        # routed differently on read-back (get_completed_incore_jobs reads job_dict[label]['conf_sp']).
        conf_opt_job = job_factory(job_adapter='gaussian', project='project_test_restore_conf',
                                   ess_settings=self.ess_settings, species=[spc], xyz=xyz,
                                   job_type='conf_opt', conformer=0,
                                   level=Level(repr={'method': 'wb97xd', 'basis': 'def2svp'}),
                                   project_directory=self.project_directory, job_num=901)
        conf_sp_job = job_factory(job_adapter='gaussian', project='project_test_restore_conf',
                                  ess_settings=self.ess_settings, species=[spc], xyz=xyz,
                                  job_type='conf_sp', conformer=0,
                                  level=Level(repr={'method': 'wb97xd', 'basis': 'def2svp'}),
                                  project_directory=self.project_directory, job_num=902)
        sched.restart_dict = {'running_jobs': {label: [conf_opt_job.as_dict(), conf_sp_job.as_dict()]}}
        sched.running_jobs = dict()
        sched.job_dict = dict()

        sched.restore_running_jobs()
        # Each conformer job is filed under its own job_type keyed by its integer index -- a conf_sp
        # job under 'conf_sp', not 'conf_opt'. Filing conf_sp under 'conf_opt' would crash the sweep
        # below with KeyError: 'conf_sp'.
        self.assertIn('conf_opt', sched.job_dict[label])
        self.assertIn(0, sched.job_dict[label]['conf_opt'])
        self.assertIn('conf_sp', sched.job_dict[label])
        self.assertIn(0, sched.job_dict[label]['conf_sp'])
        # And no container is created for a job type that was not restored: a 'tsg' key here would be
        # a phantom TS-guess entry for a species that has no TS guess job at all.
        self.assertEqual(set(sched.job_dict[label].keys()), {'conf_opt', 'conf_sp'})

        # The first scheduling sweep after a restart reproduces the production crash on the unfixed
        # code: get_i_from_job_name('conformer0') is None, the fallback derives an empty job-type
        # from the underscore-less name, and self.job_dict[label][''] raises KeyError: ''.
        sched.get_completed_incore_jobs()
        self.assertEqual(sched.completed_incore_jobs, list())

        # And the restored names are the live '{job_type}_{i}' format, not the fossil 'conformer{i}'.
        self.assertEqual(sched.running_jobs[label], ['conf_opt_0', 'conf_sp_0'])

    _METHYLAMINE_XYZ = """C      -0.57422867   -0.01669771    0.01229213
N       0.82084044    0.08279104   -0.37769346
H      -1.05737005   -0.84067772   -0.52007494
H      -1.10211468    0.90879867   -0.23383011
H      -0.66133128   -0.19490562    1.08785111
H       0.88047852    0.26966160   -1.37780789
H       1.27889520   -0.81548721   -0.22940984"""

    def _species_with_rotor(self, label, success=None, scan_path=None, invalidation_reason='', irc_label=None):
        """Build a species with a single (CN) torsion in its rotors_dict."""
        spc = ARCSpecies(label=label, smiles='CN', xyz=self._METHYLAMINE_XYZ, irc_label=irc_label)
        spc.final_xyz = spc.get_xyz()  # a restarted species whose optimization has finished
        spc.number_of_rotors = 1
        spc.rotors_dict = {0: {'pivots': [1, 2], 'top': [3, 4, 5], 'scan': [3, 1, 2, 6], 'torsion': [2, 0, 1, 5],
                               'success': success, 'scan_path': scan_path,
                               'invalidation_reason': invalidation_reason, 'directed_scan_type': 'ess'}}
        return spc

    def _stage_rotor_scan_pipe_task(self, sched, label, rotor_index, status=TaskState.PENDING, lease_expired=False):
        """
        Stage a 1D rotor scan pipe task on disk as a previous session's pipe run would have left it.

        This goes through the real ``PipeCoordinator`` and ``pipe_state`` writers, so the directory layout
        the restart loop looks for is the one the pipe system actually writes rather than a replica of it:
        a change to either side of that convention turns these tests red instead of silently detaching the
        restart guard from the pipe runs it exists to find.

        Args:
            sched (Scheduler): A scheduler supplying the project directory and the species to stage under.
            label (str): The species label.
            rotor_index (int): The rotor's index in the species' ``rotors_dict``.
            status (TaskState, optional): The state to walk the staged task to.
            lease_expired (bool, optional): Whether the worker's lease on the task has already expired.

        Returns:
            str: The ``pipe_root`` the task was staged under.
        """
        tasks = build_rotor_scan_1d_tasks(sched.species_dict[label], label, [rotor_index],
                                          {'method': 'wb97xd', 'basis': 'def2tzvp'}, 'gaussian', 14000)
        pipe_root = PipeCoordinator(sched)._compute_pipe_root(f'{label}_scan_1d', tasks)
        initialize_task(pipe_root, tasks[0])
        now = time.time()
        claimed_at, lease_expires_at = (now - 7200, now - 3600) if lease_expired else (now, now + 3600)
        lifecycle = [(TaskState.CLAIMED, {'claimed_by': 'worker-0', 'claim_token': 'token',
                                          'claimed_at': claimed_at, 'lease_expires_at': lease_expires_at}),
                     (TaskState.RUNNING, {'started_at': now}),
                     (TaskState.COMPLETED, {'ended_at': now})]
        if status != TaskState.PENDING:
            statuses = [task_state for task_state, _ in lifecycle]
            if status not in statuses:
                raise ValueError(f'Cannot stage a task in {status}, only PENDING or one of {statuses}')
            for next_status, fields in lifecycle[:statuses.index(status) + 1]:
                update_task_state(pipe_root, tasks[0].task_id, next_status, **fields)
        return pipe_root

    def _restart_scheduler(self, species_list, converged=(), stage=None, **patched):
        """
        Construct a non-testing Scheduler from a restart dict in which every species already has a geometry
        (so the restart loop in ``__init__`` runs), with ``schedule_jobs`` and the given methods patched out.

        Args:
            species_list (list): The species to restart.
            converged (tuple, optional): Labels whose saved ``output[label]['convergence']`` is ``True``,
                                         i.e. species the previous session had already finished.
            stage (callable, optional): Called with a ``testing=True`` Scheduler before the restarted one is
                                        constructed, to lay down on-disk state that a previous session would
                                        have left in the project directory (e.g. a staged pipe run).
            **patched: Scheduler method names to patch out.

        Returns:
            dict: The mocks, keyed by method name (``schedule_jobs`` included).
        """
        job_types = dict(self.job_types1, rotors=True)
        kwargs = dict(project='project_test_restart_loop', ess_settings=self.ess_settings,
                      species_list=species_list, composite_method=None,
                      conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                      opt_level=Level(repr=default_levels_of_theory['opt']),
                      freq_level=Level(repr=default_levels_of_theory['freq']),
                      sp_level=Level(repr=default_levels_of_theory['sp']),
                      scan_level=Level(repr=default_levels_of_theory['scan']),
                      ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                      project_directory=tempfile.mkdtemp(prefix='arc_test_restart_loop_'),
                      job_types=job_types, orbitals_level=default_levels_of_theory['orbitals'],
                      adaptive_levels=None)
        self.addCleanup(shutil.rmtree, kwargs['project_directory'], ignore_errors=True)
        testing_sched = Scheduler(testing=True, **kwargs)
        output = testing_sched.output
        for spc_output in output.values():
            spc_output['paths'].update({'geo': 'geo.out', 'freq': 'freq.out', 'sp': 'sp.out'})
        for label in converged:
            output[label]['convergence'] = True
        if stage is not None:
            stage(testing_sched)
        mocks = dict()
        with patch.object(Scheduler, 'schedule_jobs') as schedule_jobs:
            mocks['schedule_jobs'] = schedule_jobs
            patchers = {name: patch.object(Scheduler, name) for name in patched}
            for name, patcher in patchers.items():
                mocks[name] = patcher.start()
                self.addCleanup(patcher.stop)
            mocks['sched'] = Scheduler(testing=False, restart_dict={'output': output}, **kwargs)
        return mocks

    def test_restart_does_not_scan_irc_endpoint_rotors(self):
        """IRC endpoint species, identified by irc_label, get no restart rotor scan"""
        regular = self._species_with_rotor('regular', success=True, scan_path=__file__)
        irc = self._species_with_rotor('regular_endpoint_1', success=False, irc_label='regular')
        mocks = self._restart_scheduler([regular, irc], run_scan_jobs=None)
        self.assertEqual(mocks['run_scan_jobs'].call_args_list, list())

    def test_restart_still_scans_a_ts_whose_irc_species_were_spawned(self):
        """A TS's irc_label lists the IRC species it spawned; it does not make the TS an IRC endpoint"""
        regular = self._species_with_rotor('regular', success=True, scan_path=__file__)
        mocks = self._restart_scheduler([regular], run_scan_jobs=None)
        ts = self._species_with_rotor('ts_a', success=None, scan_path=None, irc_label='IRC_ts_a_1 IRC_ts_a_2')
        ts.is_ts = True
        sched = mocks['sched']
        sched.output['ts_a'] = {'paths': {'geo': 'geo.out'}}
        self.assertTrue(sched.rotor_scans_pending_on_restart(ts))

    def test_restart_still_scans_a_failed_regular_rotor(self):
        """The IRC guard does not suppress the restart scan of an ordinary species whose rotor failed"""
        regular = self._species_with_rotor('regular', success=False)
        irc = self._species_with_rotor('regular_endpoint_1', success=False, irc_label='regular')
        mocks = self._restart_scheduler([regular, irc], run_scan_jobs=None)
        self.assertEqual([c.args for c in mocks['run_scan_jobs'].call_args_list], [('regular',)])

    def test_restart_scans_a_never_scanned_rotor(self):
        """A rotor with no result and no scan_path (e.g., added by hand to restart.yml) is spawned on restart"""
        regular = self._species_with_rotor('regular', success=None, scan_path=None)
        mocks = self._restart_scheduler([regular], run_scan_jobs=None)
        self.assertEqual([c.args for c in mocks['run_scan_jobs'].call_args_list], [('regular',)])

    def test_restart_scans_a_never_scanned_rotor_of_a_converged_species(self):
        """A never-scanned rotor is spawned even when the species was saved as converged"""
        regular = self._species_with_rotor('regular', success=None, scan_path=None)
        mocks = self._restart_scheduler([regular], converged=('regular',), run_scan_jobs=None)
        self.assertEqual([c.args for c in mocks['run_scan_jobs'].call_args_list], [('regular',)])

    def test_restart_does_not_rescan_a_failed_rotor_of_a_converged_species(self):
        """A converged species is not re-scanned on every restart for a rotor whose scan already failed"""
        regular = self._species_with_rotor('regular', success=False)
        mocks = self._restart_scheduler([regular], converged=('regular',), run_scan_jobs=None)
        self.assertEqual(mocks['run_scan_jobs'].call_args_list, list())

    def test_restart_does_not_rescan_a_scanned_or_invalidated_rotor(self):
        """Rotors that already have a result, or are invalidated, are not spawned again on restart"""
        done = self._species_with_rotor('done', success=True, scan_path=__file__)
        invalidated = self._species_with_rotor('invalidated', success=False, invalidation_reason='not a torsion')
        mocks = self._restart_scheduler([done, invalidated], run_scan_jobs=None)
        self.assertEqual(mocks['run_scan_jobs'].call_args_list, list())

    def test_restart_does_not_duplicate_a_rotor_scan_running_in_a_pipe(self):
        """A rotor a pipe worker holds under a live lease is not spawned again as a standalone ESS job"""
        regular = self._species_with_rotor('regular', success=None, scan_path=None)
        mocks = self._restart_scheduler(
            [regular], run_scan_jobs=None,
            stage=lambda sched: self._stage_rotor_scan_pipe_task(sched, 'regular', 0, TaskState.RUNNING))
        self.assertEqual(mocks['run_scan_jobs'].call_args_list, list())

    def test_restart_scans_a_rotor_whose_pipe_worker_lease_expired(self):
        """A RUNNING task whose worker lease expired (e.g., a walltime-killed worker) no longer blocks the scan"""
        regular = self._species_with_rotor('regular', success=None, scan_path=None)
        mocks = self._restart_scheduler(
            [regular], run_scan_jobs=None,
            stage=lambda sched: self._stage_rotor_scan_pipe_task(sched, 'regular', 0, TaskState.RUNNING,
                                                                 lease_expired=True))
        self.assertEqual([c.args for c in mocks['run_scan_jobs'].call_args_list], [('regular',)])

    def test_restart_scans_a_rotor_whose_pipe_task_is_still_pending(self):
        """A PENDING task carries no lease, and no session reloads its pipe run, so it does not block the scan"""
        regular = self._species_with_rotor('regular', success=None, scan_path=None)
        mocks = self._restart_scheduler(
            [regular], run_scan_jobs=None,
            stage=lambda sched: self._stage_rotor_scan_pipe_task(sched, 'regular', 0, TaskState.PENDING))
        self.assertEqual([c.args for c in mocks['run_scan_jobs'].call_args_list], [('regular',)])

    def test_restart_scans_a_rotor_whose_pipe_task_already_terminated(self):
        """A terminated pipe task does not block the scan: nothing ingests its results after a restart"""
        regular = self._species_with_rotor('regular', success=None, scan_path=None)
        mocks = self._restart_scheduler(
            [regular], run_scan_jobs=None,
            stage=lambda sched: self._stage_rotor_scan_pipe_task(sched, 'regular', 0, TaskState.COMPLETED))
        self.assertEqual([c.args for c in mocks['run_scan_jobs'].call_args_list], [('regular',)])

    def test_restart_scans_a_rotor_whose_pipe_task_state_is_unreadable(self):
        """A pipe task state that cannot be read shows no live lease, so it does not block the scan"""
        def stage(sched):
            pipe_root = self._stage_rotor_scan_pipe_task(sched, 'regular', 0, TaskState.RUNNING)
            with open(os.path.join(pipe_root, 'tasks', 'regular_scan_r0', 'state.json'), 'w') as f:
                f.write('{"status": ')
        regular = self._species_with_rotor('regular', success=None, scan_path=None)
        mocks = self._restart_scheduler([regular], run_scan_jobs=None, stage=stage)
        self.assertEqual([c.args for c in mocks['run_scan_jobs'].call_args_list], [('regular',)])

    def test_restart_pipe_guard_is_per_rotor_not_per_species(self):
        """A pipe task staged for one rotor does not suppress the restart scan of the species' other rotor"""
        regular = self._species_with_rotor('regular', success=None, scan_path=None)
        regular.rotors_dict[1] = dict(regular.rotors_dict[0], pivots=[1, 3], torsion=[2, 0, 1, 6])
        regular.number_of_rotors = 2
        mocks = self._restart_scheduler(
            [regular], run_scan_jobs=None,
            stage=lambda sched: self._stage_rotor_scan_pipe_task(sched, 'regular', 1, TaskState.RUNNING))
        self.assertEqual([c.args for c in mocks['run_scan_jobs'].call_args_list], [('regular',)])

    def test_restart_scans_only_the_rotor_no_pipe_still_owes(self):
        """The rotor in flight in a pipe is not resubmitted when the species' other rotor triggers the spawn"""
        regular = self._species_with_rotor('regular', success=None, scan_path=None)
        regular.rotors_dict[1] = dict(regular.rotors_dict[0], pivots=[1, 3], torsion=[2, 0, 1, 6])
        regular.number_of_rotors = 2
        mocks = self._restart_scheduler(
            [regular], run_job=None,
            stage=lambda sched: self._stage_rotor_scan_pipe_task(sched, 'regular', 1, TaskState.RUNNING))
        self.assertEqual([c.kwargs['rotor_index'] for c in mocks['run_job'].call_args_list], [0])

    def test_restart_spawns_a_rotor_scan_before_the_pipe_planner_exists(self):
        """
        The restart loop in __init__ calls run_scan_jobs() before the pipe planner is constructed, so a restart
        with a rotor to spawn used to die with AttributeError. Without a planner each rotor is scanned alone.
        """
        regular = self._species_with_rotor('regular', success=None, scan_path=None)
        mocks = self._restart_scheduler([regular], run_job=None)
        self.assertEqual(mocks['run_job'].call_count, 1)
        self.assertEqual(mocks['run_job'].call_args.kwargs['job_type'], 'scan')
        self.assertEqual(mocks['run_job'].call_args.kwargs['rotor_index'], 0)

    def test_restore_running_jobs_skips_a_job_of_an_unknown_species(self):
        """A restored job referencing a species label that is not in this project is skipped, not a KeyError"""
        label = 'methylamine'
        spc = ARCSpecies(label=label, smiles='CN', xyz=self._METHYLAMINE_XYZ)
        project_directory = tempfile.mkdtemp(prefix='arc_test_restore_unknown_')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='project_test_restore_unknown', ess_settings=self.ess_settings,
                          species_list=[spc], composite_method=None,
                          conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          scan_level=Level(repr=default_levels_of_theory['scan']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory, testing=True,
                          job_types=self.job_types1,
                          orbitals_level=default_levels_of_theory['orbitals'], adaptive_levels=None)
        opt_job = job_factory(job_adapter='gaussian', project='project_test_restore_unknown',
                              ess_settings=self.ess_settings, species=[spc], xyz=self._METHYLAMINE_XYZ,
                              job_type='opt', level=Level(repr={'method': 'wb97xd', 'basis': 'def2svp'}),
                              project_directory=project_directory, job_num=903)
        stale = dict(opt_job.as_dict(), job_name='tsg9', species_labels=['rxn_label_TS'])
        sched.restart_dict = {'running_jobs': {label: [stale, opt_job.as_dict()],
                                               'removed_species': [opt_job.as_dict()]}}
        sched.running_jobs = dict()
        sched.job_dict = dict()
        sched.restore_running_jobs()
        self.assertEqual(sched.running_jobs[label], [opt_job.job_name])
        self.assertEqual(list(sched.job_dict[label]['opt'].keys()), [opt_job.job_name])
        # No running_jobs key for a species that is not in the project; schedule_jobs() would never clear it.
        self.assertNotIn('removed_species', sched.running_jobs)

    def test_save_restart_dict_skips_untracked_conformer_jobs(self):
        """A conf_opt/conf_sp/tsg name in running_jobs with no job_dict entry is skipped with a warning"""
        label = 'methylamine'
        spc = ARCSpecies(label=label, smiles='CN', xyz=self._METHYLAMINE_XYZ)
        project_directory = tempfile.mkdtemp(prefix='arc_test_save_restart_')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='project_test_save_restart', ess_settings=self.ess_settings,
                          species_list=[spc], composite_method=None,
                          conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          scan_level=Level(repr=default_levels_of_theory['scan']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory, testing=True,
                          job_types=self.job_types1,
                          orbitals_level=default_levels_of_theory['orbitals'], adaptive_levels=None)
        tracked = MagicMock()
        tracked.as_dict.return_value = {'job_name': 'conf_opt_0'}
        sched.job_dict[label] = {'conf_opt': {0: tracked}}
        sched.running_jobs[label] = ['conf_opt_0', 'conf_opt_5', 'conf_sp_2', 'tsg3']
        sched.restart_dict = dict()
        sched.save_restart = True
        with self.assertLogs('arc', level='WARNING') as logs:
            sched.save_restart_dict()
        self.assertEqual(sched.restart_dict['running_jobs'][label], [{'job_name': 'conf_opt_0'}])
        self.assertEqual(len(logs.output), 3)
        # The untracked names leave running_jobs too, so the next poll does not KeyError on them.
        self.assertEqual(sched.running_jobs[label], ['conf_opt_0'])
        sched.get_completed_incore_jobs()

    _ISSUE622_BROKEN_XYZ = """C      -0.57422867   -0.01669771    0.01229213
N       3.82084044    0.08279104   -0.37769346
H      -1.05737005   -0.84067772   -0.52007494
H      -1.10211468    0.90879867   -0.23383011
H      -0.66133128   -0.19490562    1.08785111
H       3.88047852    0.26966160   -1.37780789
H       4.27889520   -0.81548721   -0.22940984"""

    def _issue622_kwargs(self, species_list, **overrides):
        """
        Scheduler kwargs shared by the ARC issue #622 restart tests.

        Args:
            species_list (list): The species.
            **overrides: Scheduler kwargs that replace the shared ones.

        Returns:
            dict: The Scheduler kwargs.
        """
        project_directory = tempfile.mkdtemp(prefix='arc_test_issue622_')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        return dict(project='project_test_issue622', ess_settings=self.ess_settings,
                    species_list=species_list, composite_method=None,
                    conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                    opt_level=Level(repr=default_levels_of_theory['opt']),
                    freq_level=Level(repr=default_levels_of_theory['freq']),
                    sp_level=Level(repr=default_levels_of_theory['sp']),
                    scan_level=Level(repr=default_levels_of_theory['scan']),
                    ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                    project_directory=project_directory,
                    job_types=self.job_types1, orbitals_level=default_levels_of_theory['orbitals'],
                    adaptive_levels=None) | overrides

    def _issue622_restarted_scheduler(self, conformers_note=''):
        """
        Rebuild a Scheduler from a restart dict holding a species with two optimized, non-isomorphic conformers,
        no initial_xyz, and no running conformer jobs.

        Args:
            conformers_note (str, optional): Text appended to the species' output 'conformers' entry before the restart.

        Returns:
            Scheduler: The restarted scheduler, with ``schedule_jobs`` not yet called.
        """
        label = 'methylamine'
        broken_xyz = str_to_xyz(self._ISSUE622_BROKEN_XYZ)
        spc = ARCSpecies(label=label, smiles='CN')
        spc.conformers = [broken_xyz, broken_xyz]
        spc.conformers_before_opt = [str_to_xyz(self._METHYLAMINE_XYZ), str_to_xyz(self._METHYLAMINE_XYZ)]
        spc.conformer_energies = [0.0, 5.0]
        restored_spc = ARCSpecies(species_dict=spc.as_dict())
        self.assertIsNone(restored_spc.initial_xyz)
        kwargs = self._issue622_kwargs([restored_spc])
        output = Scheduler(testing=True, **kwargs).output
        output[label]['conformers'] += conformers_note
        restart_dict = {'output': output, 'running_jobs': {}}
        with patch.object(Scheduler, 'schedule_jobs'):
            sched = Scheduler(testing=False, restart_dict=restart_dict, **kwargs)
        self.assertEqual(sched.job_dict[label], dict())
        sched.running_jobs = dict()
        return sched

    def _issue622_run_schedule_jobs(self, sched):
        """
        Run ``schedule_jobs`` up to its first main-loop poll, with job spawning and restart saving mocked: each
        submitted conformer job is listed as running, as the real ``run_job`` does.

        Args:
            sched (Scheduler): The scheduler.

        Returns:
            MagicMock: The ``run_job`` mock.
        """
        class _ReachedMainLoop(Exception):
            """Raised to stop the run at the first main-loop poll."""

        def run_job(**kwargs):
            """List the submitted conformer job as running."""
            label = kwargs['label']
            sched.job_dict[label].setdefault('conf_opt', dict())[kwargs['conformer']] = MagicMock()
            sched.running_jobs.setdefault(label, list()).append(f"conf_opt_{kwargs['conformer']}")

        with patch.object(Scheduler, 'spawn_ts_jobs'), patch.object(Scheduler, 'run_job', side_effect=run_job) as rj, \
                patch.object(Scheduler, 'save_restart_dict'), \
                patch.object(Scheduler, 'get_server_job_ids', side_effect=_ReachedMainLoop):
            try:
                sched.schedule_jobs()
            except _ReachedMainLoop:
                pass
        return rj

    def test_issue622_restart_troubleshoots_non_isomorphic_conformers_with_no_conf_opt_jobs(self):
        """
        A restart whose species holds optimized, non-isomorphic conformers but no running conformer jobs
        reaches troubleshoot_conformer_isomorphism with an empty job_dict entry (ARC issue #622).
        It respawns both conformers at the first troubleshooting level, in the conformer level's adapter.
        """
        label = 'methylamine'
        sched = self._issue622_restarted_scheduler()
        with self.assertLogs('arc', level='WARNING') as logs:
            run_job = self._issue622_run_schedule_jobs(sched)
        self.assertTrue(any('no conformer isomorphism troubleshooting level is recorded' in line
                            for line in logs.output))
        self.assertEqual(run_job.call_count, 2)
        for conformer, call in enumerate(run_job.call_args_list):
            self.assertEqual(call.kwargs['job_type'], 'conf_opt')
            self.assertEqual(call.kwargs['conformer'], conformer)
            self.assertEqual(call.kwargs['level_of_theory'], 'wb97xd/def2TZVP')
            self.assertEqual(call.kwargs['job_adapter'], 'gaussian')
            self.assertIn('conformer wb97xd/def2TZVP', call.kwargs['ess_trsh_methods'])
        self.assertIsNot(run_job.call_args_list[0].kwargs['ess_trsh_methods'],
                         run_job.call_args_list[1].kwargs['ess_trsh_methods'])
        self.assertIn('Troubleshooting conformer isomorphism at wb97xd/def2TZVP; ',
                      sched.output[label]['conformers'])

    def test_issue622_restart_recovers_the_troubleshooting_history_from_output(self):
        """
        The note troubleshoot_conformer_isomorphism writes to output survives a restart, so the restarted run
        moves on to the next troubleshooting level, and reaches the exhausted path once all levels were tried.
        """
        label = 'methylamine'
        sched = self._issue622_restarted_scheduler()
        self._issue622_run_schedule_jobs(sched)
        first_note = sched.output[label]['conformers']

        sched = self._issue622_restarted_scheduler(conformers_note=first_note)
        run_job = self._issue622_run_schedule_jobs(sched)
        self.assertEqual(run_job.call_count, 2)
        self.assertEqual({c.kwargs['level_of_theory'] for c in run_job.call_args_list}, {'apfd/def2TZVP'})
        self.assertIn('conformer wb97xd/def2TZVP', run_job.call_args_list[0].kwargs['ess_trsh_methods'])
        second_note = sched.output[label]['conformers']

        sched = self._issue622_restarted_scheduler(conformers_note=second_note)
        run_job = self._issue622_run_schedule_jobs(sched)
        run_job.assert_not_called()
        self.assertIn('Error: No conformer was found to be isomorphic with the 2D graph representation!',
                      sched.output[label]['conformers'])

    def _issue622_scheduler_with_running_conformers(self, energies, running, job_dict):
        """
        Build a Scheduler whose species has conformers, no initial_xyz, and conformer jobs listed as running,
        the state ``restore_running_jobs`` leaves behind when a restart catches the conformer phase mid-run.

        Args:
            energies (list): The species' conformer energies.
            running (list): The job names listed in ``running_jobs`` for the species.
            job_dict (dict): The species' ``job_dict`` entry.

        Returns:
            Scheduler: The scheduler.
        """
        label = 'methylamine'
        spc = ARCSpecies(label=label, smiles='CN')
        spc.conformers = [str_to_xyz(self._METHYLAMINE_XYZ), str_to_xyz(self._ISSUE622_BROKEN_XYZ)]
        spc.conformers_before_opt = [str_to_xyz(self._METHYLAMINE_XYZ), str_to_xyz(self._METHYLAMINE_XYZ)]
        spc.conformer_energies = energies
        sched = Scheduler(testing=True, **self._issue622_kwargs([spc]))
        self.assertIsNone(sched.species_dict[label].initial_xyz)
        sched.testing = False
        sched.running_jobs[label] = running
        sched.job_dict[label] = job_dict
        sched.dont_gen_confs.append(label)
        return sched

    def _issue622_assert_top_block_does_not_decide(self, sched, resubmitted=()):
        """
        Run ``schedule_jobs`` up to its first main-loop poll, with ``run_conformer_jobs`` not mocked, and assert
        no conformer decision or opt was made, only the given conformers were submitted, and the restored
        conformer jobs are kept.

        Args:
            sched (Scheduler): The scheduler.
            resubmitted (tuple, optional): The conformers expected to be submitted at the conformer level.
        """
        label = 'methylamine'
        running_before = list(sched.running_jobs[label])
        job_dict_before = {job_type: dict(jobs) for job_type, jobs in sched.job_dict[label].items()}

        class _ReachedMainLoop(Exception):
            pass

        with patch.object(Scheduler, 'determine_most_stable_conformer',
                          wraps=sched.determine_most_stable_conformer) as determine, \
                patch.object(Scheduler, 'run_opt_job') as run_opt_job, \
                patch.object(Scheduler, 'run_composite_job') as run_composite_job, \
                patch.object(Scheduler, 'run_job') as run_job, \
                patch.object(sched.pipe_planner, 'try_pipe_conformers', return_value=set()), \
                patch.object(Scheduler, 'spawn_ts_jobs'), \
                patch.object(Scheduler, 'get_server_job_ids', side_effect=_ReachedMainLoop):
            with self.assertRaises(_ReachedMainLoop):
                sched.schedule_jobs()
        determine.assert_not_called()
        run_opt_job.assert_not_called()
        run_composite_job.assert_not_called()
        self.assertEqual([c.kwargs['conformer'] for c in run_job.call_args_list], list(resubmitted))
        for call in run_job.call_args_list:
            self.assertEqual(call.kwargs['job_type'], 'conf_opt')
            self.assertEqual(call.kwargs['level_of_theory'], sched.conformer_opt_level)
        self.assertIsNone(sched.species_dict[label].initial_xyz)
        self.assertEqual(sched.running_jobs[label], running_before)
        self.assertEqual(sched.job_dict[label], job_dict_before)

    def test_get_conformer_trsh_levels(self):
        """Test reading the conformer isomorphism troubleshooting levels from conf_opt jobs' ess_trsh_methods"""
        self.assertEqual(get_conformer_trsh_levels([]), list())
        self.assertEqual(get_conformer_trsh_levels([MagicMock(ess_trsh_methods=None)]), list())
        jobs = [MagicMock(ess_trsh_methods=['memory', 'conformer wb97xd/def2TZVP', 'conf_opt: wb97xd/def2TZVP',
                                            'conformer apfd/def2TZVP', 'conf_opt: apfd/def2TZVP']),
                MagicMock(ess_trsh_methods=['conf_opt: wb97xd/def2TZVP', 'scf=(qc)'])]
        self.assertEqual(get_conformer_trsh_levels(jobs), ['wb97xd/def2TZVP', 'apfd/def2TZVP'])

    def test_get_conformer_isomorphism_trsh_history(self):
        """Test reading the tried conformer isomorphism troubleshooting levels from output[label]['conformers']"""
        self.assertEqual(get_conformer_isomorphism_trsh_history(None), list())
        self.assertEqual(get_conformer_isomorphism_trsh_history(''), list())
        note = 'most stable conformer (0) did not pass isomorphism check; ' \
               'Troubleshooting conformer isomorphism at wb97xd/def2TZVP; ' \
               'Unconverged' \
               'Troubleshooting conformer isomorphism at apfd/def2TZVP; '
        self.assertEqual(get_conformer_isomorphism_trsh_history(note),
                         ['conformer wb97xd/def2TZVP', 'conformer apfd/def2TZVP'])

    def test_issue622_fresh_run_without_running_conformer_jobs_decides_as_before(self):
        """
        In a run without a restart and with no conformer job running, the guard changes nothing: the top of
        schedule_jobs decides the species' conformer and spawns opt, as on main.
        """
        label = 'methylamine'
        spc = ARCSpecies(label=label, smiles='CN')
        spc.conformers = [str_to_xyz(self._METHYLAMINE_XYZ), str_to_xyz(self._ISSUE622_BROKEN_XYZ)]
        spc.conformers_before_opt = [str_to_xyz(self._METHYLAMINE_XYZ), str_to_xyz(self._METHYLAMINE_XYZ)]
        spc.conformer_energies = [0.0, 5.0]
        sched = Scheduler(testing=True, **self._issue622_kwargs([spc]))
        sched.testing = False
        self.assertFalse(sched.has_running_conformer_jobs(label))
        self.assertNotIn(label, sched.dont_gen_confs)

        class _ReachedMainLoop(Exception):
            pass

        with patch.object(Scheduler, 'determine_most_stable_conformer',
                          wraps=sched.determine_most_stable_conformer) as determine, \
                patch.object(Scheduler, 'run_opt_job') as run_opt_job, \
                patch.object(Scheduler, 'run_job') as run_job, \
                patch.object(Scheduler, 'spawn_ts_jobs'), \
                patch.object(Scheduler, 'get_server_job_ids', side_effect=_ReachedMainLoop):
            with self.assertRaises(_ReachedMainLoop):
                sched.schedule_jobs()
        determine.assert_called_once_with(label)
        self.assertEqual(sched.species_dict[label].initial_xyz, str_to_xyz(self._METHYLAMINE_XYZ))
        run_opt_job.assert_called_once_with(label, fine=sched.fine_only)
        run_job.assert_not_called()

    def test_issue622_restart_does_not_resubmit_running_conformers(self):
        """
        A restart with conformer 0 finished and conformer 1 still running must not resubmit the running conformer,
        nor go through process_conformers, which would reset job_dict[label]['conf_opt'] and leave the restored
        job running untracked. Only conformer 0, which is not running, is submitted again.
        """
        label = 'methylamine'
        job_1 = MagicMock(job_adapter='gaussian', ess_trsh_methods=list())
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[0.0, None], running=['conf_opt_1'], job_dict={'conf_opt': {1: job_1}})
        self._issue622_assert_top_block_does_not_decide(sched, resubmitted=(0,))
        self.assertIs(sched.job_dict[label]['conf_opt'][1], job_1)
        self.assertEqual(sched.running_jobs[label], ['conf_opt_1'])

    def test_issue622_restart_before_any_conformer_finished_does_not_resubmit(self):
        """
        A restart before any conformer finished (all energies None) must not resubmit the running conformers either.
        """
        label = 'methylamine'
        job_0, job_1 = MagicMock(), MagicMock()
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[None, None], running=['conf_opt_0', 'conf_opt_1'],
            job_dict={'conf_opt': {0: job_0, 1: job_1}})
        self._issue622_assert_top_block_does_not_decide(sched)
        self.assertIs(sched.job_dict[label]['conf_opt'][0], job_0)
        self.assertIs(sched.job_dict[label]['conf_opt'][1], job_1)
        self.assertEqual(sched.running_jobs[label], ['conf_opt_0', 'conf_opt_1'])

    def test_issue622_restart_mid_run_waits_for_running_conformers(self):
        """
        Conformer 0 finished before the restart and conformer 1 is still running: the restart must not decide
        on the partial set, and troubleshooting must not need a job for conformer 0 (was KeyError: 0).
        """
        label = 'methylamine'
        job_1 = MagicMock(job_adapter='qchem', ess_trsh_methods=['memory'])
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[0.0, None], running=['conf_opt_1'], job_dict={'conf_opt': {1: job_1}})
        self._issue622_assert_top_block_does_not_decide(sched, resubmitted=(0,))

        with patch.object(Scheduler, 'run_job') as run_job:
            sched.troubleshoot_conformer_isomorphism(label=label)
        self.assertEqual(run_job.call_count, 2)
        self.assertEqual({c.kwargs['job_adapter'] for c in run_job.call_args_list}, {'qchem'})
        self.assertEqual({c.kwargs['level_of_theory'] for c in run_job.call_args_list}, {'wb97x-d3/def2-TZVP'})
        self.assertIs(run_job.call_args_list[1].kwargs['ess_trsh_methods'], job_1.ess_trsh_methods)
        self.assertIsNot(run_job.call_args_list[0].kwargs['ess_trsh_methods'], job_1.ess_trsh_methods)
        self.assertIn('conformer wb97x-d3/def2-TZVP', run_job.call_args_list[0].kwargs['ess_trsh_methods'])

    def test_issue622_restart_mid_troubleshooting_does_not_repeat_a_level(self):
        """
        A restart during a troubleshooting round restores a conformer job whose own list does not name the level
        (only the first conformer's job did); the level recorded in output must still count as tried.
        """
        label = 'methylamine'
        job_1 = MagicMock(job_adapter='gaussian', ess_trsh_methods=['conf_opt: wb97xd/def2TZVP'])
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[0.0, None], running=['conf_opt_1'], job_dict={'conf_opt': {1: job_1}})
        sched.output[label]['conformers'] += 'Troubleshooting conformer isomorphism at wb97xd/def2TZVP; '
        with patch.object(Scheduler, 'run_job') as run_job:
            sched.troubleshoot_conformer_isomorphism(label=label)
        self.assertEqual({c.kwargs['level_of_theory'] for c in run_job.call_args_list}, {'apfd/def2TZVP'})

    def test_issue622_restart_with_only_conf_sp_running(self):
        """
        Only conf_sp jobs run at restart, so restore_running_jobs leaves an empty 'conf_opt' dict: the restart
        must not decide early, and troubleshooting falls back to the conformer level's adapter.
        """
        label = 'methylamine'
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[0.0, 5.0], running=['conf_sp_0'], job_dict={'conf_opt': dict(), 'conf_sp': {0: MagicMock()}})
        self._issue622_assert_top_block_does_not_decide(sched)

        with patch.object(Scheduler, 'run_job') as run_job:
            sched.troubleshoot_conformer_isomorphism(label=label)
        self.assertEqual(run_job.call_count, 2)
        self.assertEqual({c.kwargs['job_adapter'] for c in run_job.call_args_list}, {'gaussian'})
        self.assertEqual({c.kwargs['level_of_theory'] for c in run_job.call_args_list}, {'wb97xd/def2TZVP'})

    def test_issue622_restart_does_not_spawn_opt_on_a_partial_conformer_set(self):
        """
        Finished conformers that mix a non-isomorphic most stable one and an isomorphic one, with a third conformer
        still running: the restart must not pick the isomorphic one and spawn opt before the running one ends.
        """
        label = 'methylamine'
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[0.0, 3.0, None], running=['conf_opt_2'], job_dict={'conf_opt': {2: MagicMock()}})
        spc = sched.species_dict[label]
        spc.conformers = [str_to_xyz(self._ISSUE622_BROKEN_XYZ), str_to_xyz(self._METHYLAMINE_XYZ),
                          str_to_xyz(self._METHYLAMINE_XYZ)]
        spc.conformers_before_opt = [str_to_xyz(self._METHYLAMINE_XYZ)] * 3
        self._issue622_assert_top_block_does_not_decide(sched, resubmitted=(0, 1))

    def _held_conformer_scheduler(self, held_in: str):
        """
        Build a restarted Scheduler whose conformer 1 job is out of ``running_jobs`` but still unfinished, held in
        ``stranded_jobs`` or ``deferred_jobs``, with its ``job_dict`` entry kept and conformer 0 finished.

        Args:
            held_in (str): ``'stranded_jobs'`` or ``'deferred_jobs'``.

        Returns:
            tuple[Scheduler, MagicMock]: The scheduler and the held conformer 1 job.
        """
        job_1 = MagicMock(job_adapter='gaussian', ess_trsh_methods=list())
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[0.0, None], running=list(), job_dict={'conf_opt': {1: job_1}})
        getattr(sched, held_in)['methylamine'] = ['conf_opt_1']
        return sched, job_1

    def _run_conformer_jobs_with_mocked_submission(self, sched):
        """
        Run ``run_conformer_jobs`` with every job submission mocked.

        Args:
            sched (Scheduler): The scheduler.

        Returns:
            dict: The mocks, keyed by the mocked method name.
        """
        with patch.object(Scheduler, 'run_job') as run_job, \
                patch.object(Scheduler, 'run_opt_job') as run_opt_job, \
                patch.object(Scheduler, 'run_composite_job') as run_composite_job, \
                patch.object(Scheduler, 'process_conformers', wraps=sched.process_conformers) as process_conformers, \
                patch.object(Scheduler, 'determine_most_stable_conformer') as determine, \
                patch.object(sched.pipe_planner, 'try_pipe_conformers', return_value=set()):
            sched.run_conformer_jobs()
        return {'run_job': run_job, 'run_opt_job': run_opt_job, 'run_composite_job': run_composite_job,
                'process_conformers': process_conformers, 'determine_most_stable_conformer': determine}

    def test_a_stranded_or_deferred_conformer_job_keeps_the_species_from_advancing(self):
        """A conformer job left on an unreadable queue, or held back from one, counts as unfinished"""
        label = 'methylamine'
        for held_in in ('stranded_jobs', 'deferred_jobs'):
            with self.subTest(held_in=held_in):
                sched, job_1 = self._held_conformer_scheduler(held_in)
                self.assertNotIn('conf_opt_1', sched.running_jobs[label])
                self.assertTrue(sched.has_running_conformer_jobs(label))
                mocks = self._run_conformer_jobs_with_mocked_submission(sched)
                mocks['process_conformers'].assert_not_called()
                mocks['determine_most_stable_conformer'].assert_not_called()
                mocks['run_opt_job'].assert_not_called()
                mocks['run_composite_job'].assert_not_called()
                self.assertIsNone(sched.species_dict[label].initial_xyz)
                self.assertIs(sched.job_dict[label]['conf_opt'][1], job_1)
                self.assertEqual(getattr(sched, held_in)[label], ['conf_opt_1'])

    def test_a_stranded_or_deferred_conf_opt_job_is_not_submitted_again(self):
        """The restored-conformer block submits no new job for a conformer whose conf_opt job is held"""
        for held_in in ('stranded_jobs', 'deferred_jobs'):
            with self.subTest(held_in=held_in):
                sched, _ = self._held_conformer_scheduler(held_in)
                run_job = self._run_conformer_jobs_with_mocked_submission(sched)['run_job']
                submitted = [call.kwargs['conformer'] for call in run_job.call_args_list]
                self.assertEqual(submitted.count(1), 0)
                self.assertEqual(submitted.count(0), 1)
                for call in run_job.call_args_list:
                    self.assertEqual(call.kwargs['job_type'], 'conf_opt')

    def _issue622_fresh_scheduler(self, conformers_before_opt=True):
        """
        Build a Scheduler (no restart) whose species has two optimized, non-isomorphic conformers with energies,
        no initial_xyz, and no conf_opt entry in job_dict.

        Args:
            conformers_before_opt (bool, optional): Whether to set the species' ``conformers_before_opt``.

        Returns:
            Scheduler: The scheduler.
        """
        label = 'methylamine'
        spc = ARCSpecies(label=label, smiles='CN')
        spc.conformers = [str_to_xyz(self._ISSUE622_BROKEN_XYZ), str_to_xyz(self._ISSUE622_BROKEN_XYZ)]
        if conformers_before_opt:
            spc.conformers_before_opt = [str_to_xyz(self._METHYLAMINE_XYZ), str_to_xyz(self._METHYLAMINE_XYZ)]
        spc.conformer_energies = [0.0, 5.0]
        sched = Scheduler(testing=True, **self._issue622_kwargs([spc]))
        self.assertNotIn('conf_opt', sched.job_dict[label])
        return sched

    def test_issue622_troubleshoot_without_conformers_before_opt(self):
        """
        A species whose conformers did not go through process_conformers (e.g., read from a
        conformers_after_optimization file) has no conformers_before_opt; troubleshooting respawns its conformers.
        """
        label = 'methylamine'
        sched = self._issue622_fresh_scheduler(conformers_before_opt=False)
        self.assertIsNone(sched.species_dict[label].conformers_before_opt)
        with patch.object(Scheduler, 'run_job') as run_job:
            sched.troubleshoot_conformer_isomorphism(label=label)
        self.assertEqual(run_job.call_count, 2)
        for conformer, call in enumerate(run_job.call_args_list):
            self.assertEqual(call.kwargs['xyz'], sched.species_dict[label].conformers[conformer])

    def test_issue622_trsh_note_written_only_after_every_conformer_was_respawned(self):
        """
        The tried-level note is written only once every conformer was submitted at that level, and the restart
        file is saved after it; a run killed mid-respawn leaves no note for the level.
        """
        label = 'methylamine'
        note = 'Troubleshooting conformer isomorphism at wb97xd/def2TZVP; '
        sched = self._issue622_fresh_scheduler()
        with patch.object(Scheduler, 'run_job', side_effect=[None, RuntimeError('killed')]):
            with self.assertRaises(RuntimeError):
                sched.troubleshoot_conformer_isomorphism(label=label)
        self.assertNotIn(note, sched.output[label]['conformers'])

        sched = self._issue622_fresh_scheduler()
        notes_at_save = list()
        with patch.object(Scheduler, 'run_job') as run_job, \
                patch.object(Scheduler, 'save_restart_dict',
                             side_effect=lambda: notes_at_save.append(sched.output[label]['conformers'])):
            sched.troubleshoot_conformer_isomorphism(label=label)
        self.assertEqual(run_job.call_count, 2)
        self.assertEqual(len(notes_at_save), 1)
        self.assertIn(note, notes_at_save[0])

    def test_issue622_restart_mid_respawn_repeats_the_unfinished_level(self):
        """
        A run killed mid-respawn restores a conformer job whose own list names the level, with no note in output:
        the level was not applied to every conformer, so the restart reruns it rather than moving on.
        """
        label = 'methylamine'
        job_1 = MagicMock(job_adapter='gaussian',
                          ess_trsh_methods=['conformer wb97xd/def2TZVP', 'conf_opt: wb97xd/def2TZVP'])
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[0.0, None], running=['conf_opt_1'], job_dict={'conf_opt': {1: job_1}})
        with patch.object(Scheduler, 'run_job') as run_job:
            sched.troubleshoot_conformer_isomorphism(label=label)
        self.assertEqual({c.kwargs['level_of_theory'] for c in run_job.call_args_list}, {'wb97xd/def2TZVP'})

    def test_issue622_piped_conformers_without_restart_are_troubleshot(self):
        """
        In a run with no restart, every conformer piped leaves no conf_opt entry in job_dict; once the pipe
        ingests them and none is isomorphic, the species is troubleshot instead of silently getting no opt.
        """
        label = 'methylamine'
        sched = self._issue622_fresh_scheduler()
        with patch.object(Scheduler, 'run_job') as run_job, \
                self.assertLogs('arc', level='WARNING') as logs:
            sched.pipe_coordinator._post_ingest_conf_opt(label)
        self.assertEqual(run_job.call_count, 2)
        self.assertFalse(any('Failed to determine most stable conformer' in line for line in logs.output))
        warnings = [line for line in logs.output if 'no conformer isomorphism troubleshooting level' in line]
        self.assertEqual(len(warnings), 1)
        self.assertNotIn('restart', warnings[0])

    def test_issue622_restart_does_not_rerun_an_exhausted_species(self):
        """
        A restart file written before troubleshooting notes existed marks an exhausted species only with the
        exhausted error; such a species is not troubleshot again. The per-round error alone does not stop it.
        """
        label = 'methylamine'
        round_error = 'Error: No conformer was found to be isomorphic with the 2D graph representation at ' \
                      'wb97xd/def2svp; '
        exhausted = 'Error: No conformer was found to be isomorphic with the 2D graph representation!; '
        self.assertEqual(exhausted, CONFORMER_ISOMORPHISM_TRSH_EXHAUSTED)
        sched = self._issue622_fresh_scheduler()
        sched.output[label]['conformers'] += round_error * 3 + exhausted
        with patch.object(Scheduler, 'run_job') as run_job:
            sched.troubleshoot_conformer_isomorphism(label=label)
        run_job.assert_not_called()
        self.assertEqual(sched.output[label]['conformers'].count(exhausted), 1)

        sched = self._issue622_fresh_scheduler()
        sched.output[label]['conformers'] += round_error
        with patch.object(Scheduler, 'run_job') as run_job:
            sched.troubleshoot_conformer_isomorphism(label=label)
        self.assertEqual(run_job.call_count, 2)

    def test_issue622_restored_conformer_finishing_triggers_one_decision(self):
        """
        The guards leave a species restored during a troubleshooting respawn to the main loop: nothing is
        resubmitted, and once the restored conf_opt_1 finishes, the main loop decides the species' conformer
        exactly once and spawns its opt job.
        """
        label = 'methylamine'
        job_1 = MagicMock(job_adapter='gaussian', job_id=1,
                          ess_trsh_methods=['conformer wb97xd/def2TZVP', 'conf_opt: wb97xd/def2TZVP'])
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[0.0, None], running=['conf_opt_1'], job_dict={'conf_opt': {1: job_1}})
        sched.output[label]['conformers'] += 'Troubleshooting conformer isomorphism at wb97xd/def2TZVP; '
        spc = sched.species_dict[label]
        spc.conformers = [str_to_xyz(self._METHYLAMINE_XYZ), str_to_xyz(self._METHYLAMINE_XYZ)]

        def end_job(job, label, job_name):
            sched.running_jobs[label].remove(job_name)
            return True

        def parse_conformer(job, label, i):
            spc.conformer_energies[i] = 2.0
            return False

        class _OptSpawned(Exception):
            pass

        def server_job_ids(specific_server=None):
            sched.server_job_ids = list()

        with patch.object(Scheduler, 'determine_most_stable_conformer',
                          wraps=sched.determine_most_stable_conformer) as determine, \
                patch.object(Scheduler, 'end_job', side_effect=end_job), \
                patch.object(Scheduler, 'parse_conformer', side_effect=parse_conformer), \
                patch.object(Scheduler, 'run_opt_job', side_effect=_OptSpawned) as run_opt_job, \
                patch.object(Scheduler, 'run_job') as run_job, \
                patch.object(sched.pipe_planner, 'try_pipe_conformers', return_value=set()), \
                patch.object(Scheduler, 'spawn_ts_jobs'), \
                patch.object(Scheduler, 'get_server_job_ids', side_effect=server_job_ids):
            with self.assertRaises(_OptSpawned):
                sched.schedule_jobs()
        determine.assert_called_once_with(label, sp_flag=False)
        run_opt_job.assert_called_once()
        run_job.assert_not_called()
        self.assertEqual(spc.initial_xyz, str_to_xyz(self._METHYLAMINE_XYZ))

    def test_issue622_restart_submits_conformers_that_were_never_submitted(self):
        """
        A restart that caught the initial conformer submission midway restores only conf_opt_0, while the species
        still holds the force-field energies of its generated conformers. The conformers that are not running are
        submitted, the running one is not resubmitted, and the force-field energies are cleared.
        """
        label = 'methylamine'
        job_0 = MagicMock(job_adapter='gaussian', ess_trsh_methods=list())
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[-50.0, -60.0, -70.0], running=['conf_opt_0'], job_dict={'conf_opt': {0: job_0}})
        spc = sched.species_dict[label]
        spc.conformers = [str_to_xyz(self._METHYLAMINE_XYZ)] * 3
        spc.conformers_before_opt = [str_to_xyz(self._METHYLAMINE_XYZ)] * 3

        class _ReachedMainLoop(Exception):
            pass

        with patch.object(Scheduler, 'determine_most_stable_conformer') as determine, \
                patch.object(Scheduler, 'run_job') as run_job, \
                patch.object(sched.pipe_planner, 'try_pipe_conformers', return_value=set()), \
                patch.object(Scheduler, 'spawn_ts_jobs'), \
                patch.object(Scheduler, 'get_server_job_ids', side_effect=_ReachedMainLoop):
            with self.assertRaises(_ReachedMainLoop):
                sched.schedule_jobs()
        determine.assert_not_called()
        self.assertEqual([c.kwargs['conformer'] for c in run_job.call_args_list], [1, 2])
        for call in run_job.call_args_list:
            self.assertEqual(call.kwargs['job_type'], 'conf_opt')
            self.assertEqual(call.kwargs['level_of_theory'], sched.conformer_opt_level)
        self.assertIs(sched.job_dict[label]['conf_opt'][0], job_0)
        self.assertEqual(spc.conformer_energies, [None, None, None])

    def test_issue622_restart_decides_on_job_energies_only(self):
        """
        A restart during the first conformer round, with force-field energies on the species, conf_opt_0 running,
        and conformers 1 and 2 never submitted, submits conformers 1 and 2 and selects the conformer once every
        job ended, from the job energies only.
        """
        label = 'methylamine'
        job_0 = MagicMock(job_adapter='gaussian', ess_trsh_methods=list(), job_id=100)
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[-50.0, -60.0, -70.0], running=['conf_opt_0'], job_dict={'conf_opt': {0: job_0}})
        spc = sched.species_dict[label]
        spc.conformers = [str_to_xyz(self._METHYLAMINE_XYZ)] * 3
        spc.conformers_before_opt = [str_to_xyz(self._METHYLAMINE_XYZ)] * 3
        job_energies = {0: 3.0, 1: 1.0, 2: 2.0}
        energies_at_decision = list()

        def run_job(**kwargs):
            conformer = kwargs['conformer']
            sched.job_dict[label]['conf_opt'][conformer] = MagicMock(job_id=100 + conformer)
            sched.running_jobs[label].append(f'conf_opt_{conformer}')

        def end_job(job, label, job_name):
            sched.running_jobs[label].remove(job_name)
            return True

        def parse_conformer(job, label, i):
            spc.conformer_energies[i] = job_energies[i]
            return False

        class _Decided(Exception):
            pass

        def determine(label, sp_flag=False):
            energies_at_decision.append(list(spc.conformer_energies))
            raise _Decided

        def server_job_ids(specific_server=None):
            sched.server_job_ids = list()

        with patch.object(Scheduler, 'determine_most_stable_conformer', side_effect=determine), \
                patch.object(Scheduler, 'run_job', side_effect=run_job) as run_job_mock, \
                patch.object(Scheduler, 'end_job', side_effect=end_job), \
                patch.object(Scheduler, 'parse_conformer', side_effect=parse_conformer), \
                patch.object(sched.pipe_planner, 'try_pipe_conformers', return_value=set()), \
                patch.object(Scheduler, 'spawn_ts_jobs'), \
                patch.object(Scheduler, 'get_server_job_ids', side_effect=server_job_ids):
            with self.assertRaises(_Decided):
                sched.schedule_jobs()
        self.assertEqual([c.kwargs['conformer'] for c in run_job_mock.call_args_list], [1, 2])
        self.assertEqual(energies_at_decision, [[3.0, 1.0, 2.0]])

    def test_issue622_restart_after_a_troubleshooting_note_keeps_the_energies(self):
        """
        A species whose output records a troubleshooting level or the exhausted marker is not in its first
        conformer round, even when the restored job carries no troubleshooting tag; its energies are kept.
        """
        label = 'methylamine'
        for note in ('Troubleshooting conformer isomorphism at wb97xd/def2TZVP; ',
                     CONFORMER_ISOMORPHISM_TRSH_EXHAUSTED):
            with self.subTest(note=note):
                job_1 = MagicMock(job_adapter='gaussian', ess_trsh_methods=list())
                sched = self._issue622_scheduler_with_running_conformers(
                    energies=[0.0, None], running=['conf_opt_1'], job_dict={'conf_opt': {1: job_1}})
                sched.output[label]['conformers'] += note
                self._issue622_assert_top_block_does_not_decide(sched)
                self.assertEqual(sched.species_dict[label].conformer_energies, [0.0, None])

    def test_issue622_restart_mid_respawn_resubmits_at_the_trsh_level(self):
        """
        A restart during a troubleshooting respawn that was killed before its note was written restores a job
        tagged with the troubleshooting level; the conformer that was not respawned has its earlier-level energy
        cleared and is resubmitted at that level with the restored job's adapter and tags, and the note is written.
        """
        label = 'methylamine'
        job_1 = MagicMock(job_adapter='gaussian',
                          ess_trsh_methods=['conformer wb97xd/def2TZVP', 'conf_opt: wb97xd/def2TZVP'])
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[0.0, None], running=['conf_opt_1'], job_dict={'conf_opt': {1: job_1}})
        with patch.object(Scheduler, 'run_job') as run_job, patch.object(Scheduler, 'save_restart_dict') as save:
            sched.run_conformer_jobs(labels=[label])
        self.assertEqual(len(run_job.call_args_list), 1)
        call = run_job.call_args_list[0].kwargs
        self.assertEqual((call['conformer'], call['level_of_theory'], call['job_adapter']),
                         (0, 'wb97xd/def2TZVP', 'gaussian'))
        self.assertEqual(call['ess_trsh_methods'], job_1.ess_trsh_methods)
        self.assertIsNot(call['ess_trsh_methods'], job_1.ess_trsh_methods)
        self.assertEqual(call['xyz'], sched.species_dict[label].conformers_before_opt[0])
        self.assertEqual(sched.species_dict[label].conformer_energies, [None, None])
        self.assertEqual(get_conformer_isomorphism_trsh_history(sched.output[label]['conformers']),
                         ['conformer wb97xd/def2TZVP'])
        save.assert_called_once()

    def test_i109_restart_mid_second_level_respawn_resubmits_at_that_level(self):
        """
        A restart that caught the respawn at the second troubleshooting level, after the first level's note was
        written, resubmits the conformer that was not respawned at the second level, not the first.
        """
        label = 'methylamine'
        job_1 = MagicMock(job_adapter='gaussian',
                          ess_trsh_methods=['conformer wb97xd/def2TZVP', 'conf_opt: wb97xd/def2TZVP',
                                            'conformer apfd/def2TZVP', 'conf_opt: apfd/def2TZVP'])
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[0.0, None], running=['conf_opt_1'], job_dict={'conf_opt': {1: job_1}})
        sched.output[label]['conformers'] += 'Troubleshooting conformer isomorphism at wb97xd/def2TZVP; '
        with patch.object(Scheduler, 'run_job') as run_job, patch.object(Scheduler, 'save_restart_dict'):
            sched.run_conformer_jobs(labels=[label])
        self.assertEqual([(c.kwargs['conformer'], c.kwargs['level_of_theory']) for c in run_job.call_args_list],
                         [(0, 'apfd/def2TZVP')])
        self.assertEqual(sched.species_dict[label].conformer_energies, [None, None])
        self.assertEqual(get_conformer_isomorphism_trsh_history(sched.output[label]['conformers']),
                         ['conformer wb97xd/def2TZVP', 'conformer apfd/def2TZVP'])

    def test_i109_restart_mid_respawn_reads_either_level_tag(self):
        """
        A restored job tagged with the troubleshooting level by its ``'conformer <level>'`` tag only, beside an
        untagged one, still leads the restart to resubmit the conformer that was not respawned at that level.
        """
        label = 'methylamine'
        job_0 = MagicMock(job_adapter='gaussian', ess_trsh_methods=None)
        job_1 = MagicMock(job_adapter='gaussian', ess_trsh_methods=['conformer wb97xd/def2TZVP'])
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[0.0, None, 5.0], running=['conf_opt_0', 'conf_opt_1'],
            job_dict={'conf_opt': {0: job_0, 1: job_1}})
        sched.species_dict[label].conformers_before_opt.append(str_to_xyz(self._METHYLAMINE_XYZ))
        sched.species_dict[label].conformers.append(str_to_xyz(self._ISSUE622_BROKEN_XYZ))
        with patch.object(Scheduler, 'run_job') as run_job, patch.object(Scheduler, 'save_restart_dict'):
            sched.run_conformer_jobs(labels=[label])
        self.assertEqual([(c.kwargs['conformer'], c.kwargs['level_of_theory'], c.kwargs['ess_trsh_methods'])
                          for c in run_job.call_args_list],
                         [(2, 'wb97xd/def2TZVP', ['conformer wb97xd/def2TZVP', 'conf_opt: wb97xd/def2TZVP'])])
        self.assertEqual(sched.species_dict[label].conformer_energies, [0.0, None, None])

    def test_i109_restart_without_a_readable_level_warns(self):
        """
        A restart whose species is past its first round only by the exhausted marker, with an untagged running
        job, keeps its energies, resubmits nothing at a troubleshooting level, warns naming the species, and
        records the warning once in the species' output warnings, which the restart file keeps.
        """
        label = 'methylamine'
        job_1 = MagicMock(job_adapter='gaussian', ess_trsh_methods=list())
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[0.0, None], running=['conf_opt_1'], job_dict={'conf_opt': {1: job_1}})
        sched.output[label]['conformers'] += CONFORMER_ISOMORPHISM_TRSH_EXHAUSTED
        with patch.object(Scheduler, 'run_job') as run_job, self.assertLogs('arc', level='WARNING') as logs:
            sched.run_conformer_jobs(labels=[label])
        run_job.assert_not_called()
        self.assertEqual(sched.species_dict[label].conformer_energies, [0.0, None])
        warnings = [line for line in logs.output if 'No conformer isomorphism troubleshooting level' in line]
        self.assertEqual(len(warnings), 1)
        self.assertIn(label, warnings[0])
        self.assertEqual(sched.output[label]['warnings'].count(UNREADABLE_CONFORMER_TRSH_LEVEL_MESSAGE), 1)
        sched.restart_dict, sched.save_restart, sched.running_jobs[label] = dict(), True, list()
        sched.save_restart_dict()
        restarted = self._i109_restart(sched)
        self.assertEqual(restarted.output[label]['warnings'].count(UNREADABLE_CONFORMER_TRSH_LEVEL_MESSAGE), 1)
        restarted.running_jobs[label] = ['conf_opt_1']
        restarted.job_dict[label] = {'conf_opt': {1: job_1}}
        with patch.object(Scheduler, 'run_job'):
            restarted.run_conformer_jobs(labels=[label])
        self.assertEqual(restarted.output[label]['warnings'].count(UNREADABLE_CONFORMER_TRSH_LEVEL_MESSAGE), 1)

    @staticmethod
    def _i109_level(level_of_theory) -> str:
        """
        Normalize a ``run_job`` level_of_theory argument (a Level or a string) for comparison.

        Args:
            level_of_theory (Level, str): The level passed to ``run_job``.

        Returns:
            str: The level in its simple lowercase form.
        """
        level = level_of_theory if isinstance(level_of_theory, Level) else Level(repr=level_of_theory)
        return level.simple().lower()

    def _i109_live_scheduler(self, **overrides):
        """
        Build a live-run Scheduler that saves a real restart file, whose species has three conformers, and run its
        first conformer round at the conformer level with the real ``run_job``, submission patched out.

        Args:
            **overrides: Scheduler kwargs that replace the shared ones.

        Returns:
            Scheduler: The scheduler, with the three round-1 conf_opt jobs listed as running.
        """
        label = 'methylamine'
        spc = ARCSpecies(label=label, smiles='CN')
        spc.conformers = [str_to_xyz(self._METHYLAMINE_XYZ)] * 3
        spc.conformer_energies = [None] * 3
        sched = Scheduler(testing=True, restart_dict=dict(), **self._issue622_kwargs([spc], **overrides))
        sched.trsh_ess_jobs = True
        sched.species_dict[label].conformers_before_opt = tuple(spc.conformers)
        with patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            for i, xyz in enumerate(spc.conformers):
                sched.run_job(label=label, xyz=xyz, job_type='conf_opt', level_of_theory=sched.conformer_opt_level,
                              conformer=i)
        return sched

    def _i109_end_conformer_jobs(self, sched, energies):
        """
        End every running conf_opt job of the species as ``end_job`` and ``parse_conformer`` do: the name leaves
        running_jobs, the restart file is saved, then the energy and the optimized (non-isomorphic) xyz are set.

        Args:
            sched (Scheduler): The scheduler.
            energies (list): The energy each conformer's job yields, by conformer index.
        """
        label = 'methylamine'
        spc = sched.species_dict[label]
        for job_name in list(sched.running_jobs[label]):
            i = int(job_name.rsplit('_', 1)[1])
            sched.running_jobs[label].remove(job_name)
            sched.save_restart_dict()
            spc.conformer_energies[i] = energies[i]
            spc.conformers[i] = str_to_xyz(self._ISSUE622_BROKEN_XYZ)

    def _i109_restart(self, sched, **overrides):
        """
        Restart from the restart file the given scheduler wrote, as ``ARC.py restart.yml`` would.

        Args:
            sched (Scheduler): The scheduler of the interrupted run.
            **overrides: Scheduler kwargs that replace the shared ones.

        Returns:
            Scheduler: The restarted scheduler, with ``schedule_jobs`` not yet called.
        """
        restart_dict = read_yaml_file(sched.restart_path)
        spc = ARCSpecies(species_dict=restart_dict['species'][0])
        with patch.object(Scheduler, 'schedule_jobs'):
            return Scheduler(testing=False, restart_dict=restart_dict, **self._issue622_kwargs([spc], **overrides))

    def _i109_interrupted_respawn_restart(self, round_1_energies, kill_after=1):
        """
        Run a first conformer round, let none of the conformers pass isomorphism so the real troubleshooting
        respawns them all at the first troubleshooting level, kill the run after ``kill_after`` respawn submissions,
        and restart from the restart file that run left behind.

        Args:
            round_1_energies (list): The conformer-level energies of the three round-1 conformers.
            kill_after (int, optional): The number of respawn submissions that went through before the kill.

        Returns:
            Scheduler: The restarted scheduler, with ``schedule_jobs`` not yet called.
        """
        label = 'methylamine'
        sched = self._i109_live_scheduler()
        self._i109_end_conformer_jobs(sched, round_1_energies)
        self._i109_kill_mid_respawn(sched, kill_after=kill_after)
        self.assertEqual(get_conformer_isomorphism_trsh_history(sched.output[label]['conformers']), list())
        return self._i109_restart(sched)

    def _i109_kill_mid_respawn(self, sched, kill_after=1):
        """
        Let none of the species' conformers pass isomorphism so the real troubleshooting respawns them, and kill
        the run after ``kill_after`` respawn submissions went through the real ``run_job``.

        Args:
            sched (Scheduler): A live-run scheduler whose conf_opt jobs ended.
            kill_after (int, optional): The number of respawn submissions that went through before the kill.
        """
        real_run_job, respawned = sched.run_job, list()

        def run_job_until_killed(**kwargs):
            """Run the real run_job until ``kill_after`` submissions went through, then simulate a kill."""
            if len(respawned) == kill_after:
                raise KeyboardInterrupt('ARC was killed mid-respawn')
            respawned.append(kwargs['conformer'])
            real_run_job(**kwargs)

        with patch.object(sched, 'run_job', side_effect=run_job_until_killed), \
                patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            with self.assertRaises(KeyboardInterrupt):
                sched.determine_most_stable_conformer('methylamine')
        self.assertEqual(respawned, list(range(kill_after)))

    def _i109_drive_restart(self, sched, job_energies):
        """
        Run ``schedule_jobs`` on a restarted scheduler: every conf_opt job ends with the given energy, and the
        conformer decision records the energies it sees and stops the run.

        Args:
            sched (Scheduler): The restarted scheduler.
            job_energies (dict): The energy each conformer's job yields, keyed by conformer index.

        Returns:
            tuple: The ``run_job`` calls' kwargs, and the list of energy lists seen at each decision.
        """
        label = 'methylamine'
        spc = sched.species_dict[label]
        submitted, energies_at_decision = list(), list()

        def run_job(**kwargs):
            """Record the submission and list a stub conf_opt job as running."""
            submitted.append(kwargs)
            sched.job_dict[label]['conf_opt'][kwargs['conformer']] = MagicMock(job_id=200 + kwargs['conformer'])
            sched.running_jobs[label].append(f"conf_opt_{kwargs['conformer']}")

        def end_job(job, label, job_name):
            """End the job successfully by removing it from running_jobs."""
            sched.running_jobs[label].remove(job_name)
            return True

        def parse_conformer(job, label, i):
            """Set the conformer's energy to the one its job yields."""
            spc.conformer_energies[i] = job_energies[i]
            return False

        class _Decided(Exception):
            """Raised to stop the run at the conformer decision."""

        def determine(label, sp_flag=False):
            """Record the energies the decision sees and stop the run."""
            energies_at_decision.append(list(spc.conformer_energies))
            raise _Decided

        def server_job_ids(specific_server=None):
            """Report no server jobs so the stub jobs are polled as terminated."""
            sched.server_job_ids = list()

        with patch.object(Scheduler, 'determine_most_stable_conformer', side_effect=determine), \
                patch.object(Scheduler, 'run_job', side_effect=run_job), \
                patch.object(Scheduler, 'save_restart_dict'), \
                patch.object(Scheduler, 'end_job', side_effect=end_job), \
                patch.object(Scheduler, 'parse_conformer', side_effect=parse_conformer), \
                patch.object(sched.pipe_planner, 'try_pipe_conformers', return_value=set()), \
                patch.object(Scheduler, 'spawn_ts_jobs'), \
                patch.object(Scheduler, 'get_server_job_ids', side_effect=server_job_ids), \
                patch.object(JobAdapter, 'execute'):
            with self.assertRaises(_Decided):
                sched.schedule_jobs()
        return submitted, energies_at_decision

    def test_i109_restart_mid_respawn_does_not_select_across_levels(self):
        """
        A run killed after the troubleshooting respawn resubmitted conformer 0 only must not, after a restart,
        select a conformer from conformer-level energies of conformers 1 and 2 mixed with the troubleshooting-level
        energy of conformer 0: conformers 1 and 2 are resubmitted at the troubleshooting level.
        """
        label = 'methylamine'
        sched = self._i109_interrupted_respawn_restart(round_1_energies=[0.0, 10.0, 20.0])
        restored = sched.job_dict[label]['conf_opt']
        self.assertEqual(list(restored), [0])
        self.assertEqual(self._i109_level(restored[0].level), 'wb97xd/def2tzvp')
        self.assertIn('conf_opt: wb97xd/def2TZVP', restored[0].ess_trsh_methods)
        self.assertEqual(sched.species_dict[label].conformer_energies, [None, 10.0, 20.0])

        trsh_level_energies = {0: 30.0, 1: 20.0, 2: 10.0}
        submitted, energies_at_decision = self._i109_drive_restart(sched, trsh_level_energies)
        self.assertEqual(energies_at_decision, [[30.0, 20.0, 10.0]],
                         'the conformer was selected from energies of two levels')
        self.assertEqual([(c['conformer'], self._i109_level(c['level_of_theory'])) for c in submitted],
                         [(1, 'wb97xd/def2tzvp'), (2, 'wb97xd/def2tzvp')])

    def test_i109_restart_mid_respawn_resubmits_a_none_energy_conformer_at_the_trsh_level(self):
        """
        A conformer whose round-1 job gave no energy, not yet respawned when the run was killed mid-respawn, is
        resubmitted on restart at the troubleshooting level, not at the conformer level that already failed.
        """
        sched = self._i109_interrupted_respawn_restart(round_1_energies=[0.0, 10.0, None])
        submitted, _ = self._i109_drive_restart(sched, {0: 30.0, 1: 20.0, 2: 10.0})
        levels = {c['conformer']: self._i109_level(c['level_of_theory']) for c in submitted}
        self.assertIn(2, levels)
        self.assertEqual(levels[2], 'wb97xd/def2tzvp',
                         'the None-energy conformer was resubmitted at the base conformer level')

    def test_i109_restart_mid_respawn_records_the_level_in_the_restart_file(self):
        """
        Once a restart completed an interrupted respawn, its restart file records the level as tried: a second
        restart from that file whose conformers again fail isomorphism moves on to the next level.
        """
        label = 'methylamine'
        sched = self._i109_interrupted_respawn_restart(round_1_energies=[0.0, 10.0, 20.0])
        with patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            sched.run_conformer_jobs(labels=[label])
        self.assertEqual(sorted(sched.job_dict[label]['conf_opt']), [0, 1, 2])
        for i in (1, 2):
            self.assertEqual(self._i109_level(sched.job_dict[label]['conf_opt'][i].level), 'wb97xd/def2tzvp')
        restart_dict = read_yaml_file(sched.restart_path)
        self.assertEqual(get_conformer_isomorphism_trsh_history(restart_dict['output'][label]['conformers']),
                         ['conformer wb97xd/def2TZVP'])

        restarted = self._i109_restart(sched)
        self.assertEqual(sorted(restarted.job_dict[label]['conf_opt']), [0, 1, 2])
        with patch.object(Scheduler, 'run_job') as run_job:
            restarted.troubleshoot_conformer_isomorphism(label=label)
        self.assertEqual({c.kwargs['level_of_theory'] for c in run_job.call_args_list}, {'apfd/def2TZVP'})

    def test_get_respawned_conformer_trsh_methods(self):
        """Test building the tags of a conformer that a troubleshooting respawn submits with no job of its own"""
        history = ['conformer wb97xd/def2TZVP']
        self.assertEqual(get_respawned_conformer_trsh_methods(history, 'apfd/def2TZVP'),
                         ['conformer wb97xd/def2TZVP', 'conformer apfd/def2TZVP', 'conf_opt: apfd/def2TZVP'])
        self.assertEqual(history, ['conformer wb97xd/def2TZVP'])
        self.assertEqual(get_respawned_conformer_trsh_methods(['conformer apfd/def2TZVP'], 'apfd/def2TZVP'),
                         ['conformer apfd/def2TZVP', 'conf_opt: apfd/def2TZVP'])

    def test_i109_recovery_does_not_copy_the_reference_jobs_ess_trsh_history(self):
        """
        The conformers a restart resubmits to complete an interrupted respawn start from the species' isomorphism
        history and the level tags only, not from the ESS troubleshooting attempts of the restored job.
        """
        label = 'methylamine'
        job_1 = MagicMock(job_adapter='gaussian',
                          ess_trsh_methods=['trsh_attempt', 'memory', 'trsh_attempt',
                                            'conformer wb97xd/def2TZVP', 'conf_opt: wb97xd/def2TZVP'])
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[0.0, None], running=['conf_opt_1'], job_dict={'conf_opt': {1: job_1}})
        with patch.object(Scheduler, 'run_job') as run_job, patch.object(Scheduler, 'save_restart_dict'):
            sched.run_conformer_jobs(labels=[label])
        self.assertEqual(len(run_job.call_args_list), 1)
        tags = run_job.call_args_list[0].kwargs['ess_trsh_methods']
        self.assertNotIn('trsh_attempt', tags)
        self.assertNotIn('memory', tags)
        self.assertEqual(tags, ['conformer wb97xd/def2TZVP', 'conf_opt: wb97xd/def2TZVP'])

    def _i109_recovery_state(self, branch, energies=None):
        """
        Build a restarted scheduler in the state that one conformer isomorphism troubleshooting recovery branch
        handles: a running conformer tagged with a level the note does not record (``'unnoted'``), a running
        conformer at the last noted level (``'noted'``), or no running conformer and no energy (``'idle'``).

        Args:
            branch (str): ``'unnoted'``, ``'noted'`` or ``'idle'``.
            energies (list, optional): The conformer energies. Defaults to ``[None, None]``.

        Returns:
            Scheduler: The scheduler.
        """
        label = 'methylamine'
        energies = energies if energies is not None else [None, None]
        job_1 = MagicMock(job_adapter='gaussian',
                          ess_trsh_methods=['conformer wb97xd/def2TZVP', 'conf_opt: wb97xd/def2TZVP'])
        if branch == 'idle':
            sched = self._issue622_scheduler_with_running_conformers(energies=energies, running=[], job_dict=dict())
        else:
            sched = self._issue622_scheduler_with_running_conformers(
                energies=energies, running=['conf_opt_1'], job_dict={'conf_opt': {1: job_1}})
        if branch in ('noted', 'idle'):
            sched.output[label]['conformers'] += 'Troubleshooting conformer isomorphism at wb97xd/def2TZVP; '
        return sched

    def test_i109_refused_recovery_submits_nothing(self):
        """
        With ``trsh_ess_jobs`` off or the exhausted marker in the conformers note, no recovery branch submits a
        job, at a troubleshooting level or at the conformer level, nor generates conformers; each warns once naming
        the species and records the warning once in the species' output warnings.
        """
        label = 'methylamine'
        for branch in ('unnoted', 'noted', 'idle'):
            for condition in ('trsh_ess_jobs off', 'exhausted'):
                with self.subTest(branch=branch, condition=condition):
                    sched = self._i109_recovery_state(branch)
                    if condition == 'exhausted':
                        sched.output[label]['conformers'] += CONFORMER_ISOMORPHISM_TRSH_EXHAUSTED
                    else:
                        sched.trsh_ess_jobs = False
                    with patch.object(Scheduler, 'run_job') as run_job, \
                            patch.object(Scheduler, 'save_restart_dict'), \
                            patch.object(Scheduler, 'process_conformers') as process_conformers, \
                            patch.object(ARCSpecies, 'generate_conformers') as generate_conformers, \
                            self.assertLogs('arc', level='WARNING') as logs:
                        sched.run_conformer_jobs(labels=[label])
                        sched.run_conformer_jobs(labels=[label])
                    run_job.assert_not_called()
                    process_conformers.assert_not_called()
                    generate_conformers.assert_not_called()
                    self.assertEqual(sched.species_dict[label].conformer_energies, [None, None])
                    warnings = [line for line in logs.output if REFUSED_CONFORMER_TRSH_RECOVERY_MESSAGE in line]
                    self.assertEqual(len(warnings), 2)
                    self.assertIn(label, warnings[0])
                    self.assertEqual(
                        sched.output[label]['warnings'].count(REFUSED_CONFORMER_TRSH_RECOVERY_MESSAGE), 1)

    def test_i109_refused_recovery_decides_on_the_running_jobs_energies_only(self):
        """
        A refused recovery of a respawn caught midway clears the earlier-level energy of every conformer with no
        running job, so the decision sees only the energies of the restored troubleshooting-level jobs.
        """
        label = 'methylamine'
        for condition in ('trsh_ess_jobs off', 'exhausted'):
            with self.subTest(condition=condition):
                sched = self._i109_recovery_state('unnoted', energies=[0.0, None])
                if condition == 'exhausted':
                    sched.output[label]['conformers'] += CONFORMER_ISOMORPHISM_TRSH_EXHAUSTED
                else:
                    sched.trsh_ess_jobs = False
                with patch.object(Scheduler, 'run_job') as run_job, patch.object(Scheduler, 'save_restart_dict'):
                    sched.run_conformer_jobs(labels=[label])
                run_job.assert_not_called()
                self.assertEqual(sched.species_dict[label].conformer_energies, [None, None])
                self.assertEqual(get_conformer_isomorphism_trsh_history(sched.output[label]['conformers']), list())

    def test_i109_restart_of_an_invalidated_species_does_not_reach_idle_recovery(self):
        """
        A species troubleshot at a conformer isomorphism level, selected, then invalidated by the real
        ``troubleshoot_negative_freq`` has ``convergence`` False; its restart file keeps its initial_xyz and
        energies, so a restart neither resubmits its conformers nor generates new ones.
        """
        label = 'methylamine'
        sched = self._i109_live_scheduler()
        self._i109_end_conformer_jobs(sched, [0.0, 10.0, 20.0])
        with patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            sched.determine_most_stable_conformer(label)
        spc = sched.species_dict[label]
        for job_name in list(sched.running_jobs[label]):
            i = int(job_name.rsplit('_', 1)[1])
            sched.running_jobs[label].remove(job_name)
            spc.conformer_energies[i] = 10.0 * i
            spc.conformers[i] = str_to_xyz(self._METHYLAMINE_XYZ)
        with patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            sched.determine_most_stable_conformer(label)
        self.assertIsNotNone(spc.initial_xyz)
        invalidation = (list(), list(), ['Error: Encountered negative frequencies too many times; '
                                         'Invalidating species; '], list())
        with patch('arc.scheduler.trsh_negative_freq', return_value=invalidation), patch.object(JobAdapter, 'delete'):
            sched.troubleshoot_negative_freq(label=label, job=MagicMock())
        self.assertIs(sched.output[label]['convergence'], False)
        sched.save_restart_dict()

        with patch.object(Scheduler, 'run_job'):
            restarted = self._i109_restart(sched)
        self.assertIs(restarted.output[label]['convergence'], False)
        self.assertIsNotNone(restarted.species_dict[label].initial_xyz)
        with patch.object(Scheduler, 'run_job') as run_job, \
                patch.object(Scheduler, 'process_conformers') as process_conformers, \
                patch.object(ARCSpecies, 'generate_conformers') as generate_conformers:
            restarted.run_conformer_jobs(labels=[label])
        run_job.assert_not_called()
        process_conformers.assert_not_called()
        generate_conformers.assert_not_called()

    def test_i109_restart_after_a_trsh_level_job_ended_unparsed_does_not_select_across_levels(self):
        """
        A respawn at the troubleshooting level completes and writes its note; conformer 1's job then ends and the
        restart file is saved without it, but the run is killed before its energy is parsed. After a restart,
        the conformer is not selected from conformer 1's earlier-level energy: conformer 1 is resubmitted at the
        troubleshooting level.
        """
        label = 'methylamine'
        sched = self._i109_live_scheduler()
        self._i109_end_conformer_jobs(sched, [0.0, 10.0, 20.0])
        with patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            sched.determine_most_stable_conformer(label)
        self.assertEqual(get_conformer_isomorphism_trsh_history(sched.output[label]['conformers']),
                         ['conformer wb97xd/def2TZVP'])
        self.assertEqual(sorted(sched.running_jobs[label]), ['conf_opt_0', 'conf_opt_1', 'conf_opt_2'])
        sched.running_jobs[label].remove('conf_opt_1')
        sched.save_restart_dict()

        restarted = self._i109_restart(sched)
        self.assertEqual(sorted(restarted.job_dict[label]['conf_opt']), [0, 2])
        submitted, energies_at_decision = self._i109_drive_restart(restarted, {0: 30.0, 1: 20.0, 2: 10.0})
        self.assertEqual(energies_at_decision, [[30.0, 20.0, 10.0]],
                         'the conformer was selected from energies of two levels')
        self.assertEqual([(c['conformer'], self._i109_level(c['level_of_theory'])) for c in submitted],
                         [(1, 'wb97xd/def2tzvp')])

    def test_i109_restart_resubmits_a_none_energy_conformer_at_the_noted_level(self):
        """
        A restart whose running conf_opt job is at the last noted troubleshooting level resubmits a conformer that
        has no energy and no job at that level, with the species' history tags, not at the conformer level.
        """
        label = 'methylamine'
        job_1 = MagicMock(job_adapter='gaussian',
                          ess_trsh_methods=['trsh_attempt', 'conformer wb97xd/def2TZVP', 'conf_opt: wb97xd/def2TZVP'])
        sched = self._issue622_scheduler_with_running_conformers(
            energies=[None, None], running=['conf_opt_1'], job_dict={'conf_opt': {1: job_1}})
        sched.output[label]['conformers'] += 'Troubleshooting conformer isomorphism at wb97xd/def2TZVP; '
        with patch.object(Scheduler, 'run_job') as run_job, patch.object(Scheduler, 'save_restart_dict'):
            sched.run_conformer_jobs(labels=[label])
        self.assertEqual(len(run_job.call_args_list), 1)
        call = run_job.call_args_list[0].kwargs
        self.assertEqual((call['conformer'], call['level_of_theory'], call['job_adapter']),
                         (0, 'wb97xd/def2TZVP', 'gaussian'))
        self.assertEqual(call['ess_trsh_methods'], ['conformer wb97xd/def2TZVP', 'conf_opt: wb97xd/def2TZVP'])
        self.assertEqual(call['xyz'], sched.species_dict[label].conformers_before_opt[0])
        self.assertEqual(get_conformer_isomorphism_trsh_history(sched.output[label]['conformers']),
                         ['conformer wb97xd/def2TZVP'])

    def test_i109_restart_with_no_energy_and_no_running_job_resubmits_at_the_noted_level(self):
        """
        A respawn at the troubleshooting level completes and writes its note; every respawned job then ends and
        the restart file is saved, with no energy parsed (two jobs failed, the third ended as the run was
        killed). After a restart, all conformers are resubmitted at the troubleshooting level and no new
        conformers are generated.
        """
        label = 'methylamine'
        sched = self._i109_live_scheduler()
        self._i109_end_conformer_jobs(sched, [0.0, 10.0, 20.0])
        with patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            sched.determine_most_stable_conformer(label)
        for job_name in list(sched.running_jobs[label]):
            sched.running_jobs[label].remove(job_name)
            sched.save_restart_dict()

        restarted = self._i109_restart(sched)
        self.assertEqual(restarted.species_dict[label].conformer_energies, [None, None, None])
        self.assertFalse(restarted.running_jobs.get(label))
        with patch.object(Scheduler, 'run_job') as run_job, \
                patch.object(Scheduler, 'process_conformers') as process_conformers, \
                patch.object(ARCSpecies, 'generate_conformers') as generate_conformers:
            restarted.run_conformer_jobs(labels=[label])
        process_conformers.assert_not_called()
        generate_conformers.assert_not_called()
        self.assertEqual([(c.kwargs['conformer'], c.kwargs['level_of_theory'], c.kwargs['ess_trsh_methods'])
                          for c in run_job.call_args_list],
                         [(i, 'wb97xd/def2TZVP', ['conformer wb97xd/def2TZVP', 'conf_opt: wb97xd/def2TZVP'])
                          for i in range(3)])

    def test_i109_respawn_clears_the_energies_it_resubmits(self):
        """The troubleshooting respawn clears the earlier-level energy of every conformer it resubmits."""
        label = 'methylamine'
        sched = self._i109_live_scheduler()
        self._i109_end_conformer_jobs(sched, [0.0, 10.0, 20.0])
        with patch.object(Scheduler, 'run_job') as run_job:
            sched.troubleshoot_conformer_isomorphism(label=label)
        self.assertEqual([c.kwargs['conformer'] for c in run_job.call_args_list], [0, 1, 2])
        self.assertEqual(sched.species_dict[label].conformer_energies, [None, None, None])

    def _i112_respawned_scheduler(self, conformers_note=''):
        """
        Run a first conformer round that fails isomorphism, and let the real troubleshooting respawn every
        conformer at the next troubleshooting level, submission patched out.

        Args:
            conformers_note (str, optional): Text appended to the species' conformers note before the respawn.

        Returns:
            Scheduler: The scheduler, with the three respawned conf_opt jobs listed as running.
        """
        sched = self._i109_live_scheduler()
        self._i109_end_conformer_jobs(sched, [0.0, 10.0, 20.0])
        sched.output['methylamine']['conformers'] += conformers_note
        with patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            sched.determine_most_stable_conformer('methylamine')
        return sched

    @staticmethod
    def _i112_fail_conformer_jobs(sched, keywords):
        """
        End every running conf_opt job of the species as ``errored`` and handle it as ``schedule_jobs`` does: the
        real ``parse_conformer`` (hence ``troubleshoot_ess``) per job, then the real
        ``determine_most_stable_conformer`` once no conformer job is running. Submission is patched out.

        Args:
            sched (Scheduler): The scheduler.
            keywords (dict): The ESS error keywords of each conformer's job, keyed by conformer index.

        Returns:
            list: The ``(conformer, level)`` of every conf_opt job submitted meanwhile.
        """
        label = 'methylamine'
        submitted, real_run_job = list(), sched.run_job

        def run_job(**kwargs):
            """Record the submission and run the real run_job."""
            submitted.append((kwargs['conformer'], TestScheduler._i109_level(kwargs['level_of_theory'])))
            real_run_job(**kwargs)

        with patch.object(sched, 'run_job', side_effect=run_job), \
                patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            for job_name in [name for name in sched.running_jobs[label] if name.startswith('conf_opt')]:
                i = int(job_name.rsplit('_', 1)[1])
                job = sched.job_dict[label]['conf_opt'][i]
                job.job_status = ['done', {'status': 'errored', 'keywords': keywords[i],
                                           'error': 'conformer job failed', 'line': ''}]
                sched.running_jobs[label].remove(job_name)
                sched.parse_conformer(job=job, label=label, i=i)
            if not sched.has_running_conformer_jobs(label):
                sched.determine_most_stable_conformer(label)
        return submitted

    def test_i112_all_none_after_a_trsh_level_moves_to_the_next_level(self):
        """
        When every conformer job at a conformer isomorphism troubleshooting level fails and its ESS
        troubleshooting cannot resubmit it, the species moves on to the next troubleshooting level.
        """
        label = 'methylamine'
        sched = self._i112_respawned_scheduler()
        self.assertEqual(get_conformer_isomorphism_trsh_history(sched.output[label]['conformers']),
                         ['conformer wb97xd/def2TZVP'])
        submitted = self._i112_fail_conformer_jobs(sched, {i: ['DiskSpace'] for i in range(3)})
        self.assertEqual(submitted, [(i, 'apfd/def2tzvp') for i in range(3)])
        self.assertEqual(get_conformer_isomorphism_trsh_history(sched.output[label]['conformers']),
                         ['conformer wb97xd/def2TZVP', 'conformer apfd/def2TZVP'])
        self.assertEqual(sched.species_dict[label].conformer_energies, [None, None, None])

    def test_i112_all_none_at_the_last_trsh_level_marks_it_exhausted(self):
        """When every conformer job at the last troubleshooting level fails, troubleshooting ends as exhausted."""
        label = 'methylamine'
        sched = self._i112_respawned_scheduler()
        self._i112_fail_conformer_jobs(sched, {i: ['DiskSpace'] for i in range(3)})
        submitted = self._i112_fail_conformer_jobs(sched, {i: ['DiskSpace'] for i in range(3)})
        self.assertEqual(submitted, list())
        self.assertIn(CONFORMER_ISOMORPHISM_TRSH_EXHAUSTED, sched.output[label]['conformers'])
        self.assertFalse(sched.has_running_conformer_jobs(label))

    def test_i112_all_none_at_the_conformer_level_does_not_troubleshoot_isomorphism(self):
        """When every first-round conformer job fails, no conformer isomorphism troubleshooting level is tried."""
        label = 'methylamine'
        sched = self._i109_live_scheduler()
        submitted = self._i112_fail_conformer_jobs(sched, {i: ['DiskSpace'] for i in range(3)})
        self.assertEqual(submitted, list())
        self.assertEqual(get_conformer_isomorphism_trsh_history(sched.output[label]['conformers']), list())

    def test_i112_all_none_moves_to_the_next_level_once_ess_troubleshooting_ends(self):
        """
        While ESS troubleshooting still reruns a failed troubleshooting-level conformer job, the species stays at
        that level; once the rerun fails too, it moves to the next level exactly once.
        """
        label = 'methylamine'
        sched = self._i112_respawned_scheduler()
        submitted = self._i112_fail_conformer_jobs(sched, {0: ['MaxOptCycles', 'GL9999'], 1: ['DiskSpace'],
                                                           2: ['DiskSpace']})
        self.assertEqual(submitted, [(0, 'wb97xd/def2tzvp')])
        self.assertEqual(sched.running_jobs[label], [sched.job_dict[label]['conf_opt'][0].job_name])
        submitted = self._i112_fail_conformer_jobs(sched, {0: ['DiskSpace']})
        self.assertEqual(submitted, [(i, 'apfd/def2tzvp') for i in range(3)])
        self.assertEqual(sched.output[label]['conformers'].count(
            CONFORMER_ISOMORPHISM_TRSH_NOTE.format(level='apfd/def2TZVP')), 1)

    def test_i109_restart_with_only_conf_sp_running_does_not_warn(self):
        """
        A first-round restart that finds only conf_sp jobs running, their conf_opt jobs done, is not a
        troubleshooting state: it resubmits nothing and records no unreadable-level warning.
        """
        label = 'methylamine'
        sched = self._i109_live_scheduler()
        self._i109_end_conformer_jobs(sched, [0.0, 10.0, 20.0])
        with patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            for i, xyz in enumerate(sched.species_dict[label].conformers):
                sched.run_job(label=label, xyz=xyz, job_type='conf_sp', level_of_theory=sched.sp_level,
                              conformer=i)
        restarted = self._i109_restart(sched)
        self.assertEqual(sorted(restarted.job_dict[label]['conf_sp']), [0, 1, 2])
        self.assertFalse(restarted.job_dict[label].get('conf_opt'))
        with patch.object(Scheduler, 'run_job') as run_job, patch('arc.scheduler.logger') as logger:
            restarted.run_conformer_jobs(labels=[label])
        run_job.assert_not_called()
        self.assertFalse([c for c in logger.warning.call_args_list
                          if 'No conformer isomorphism troubleshooting level' in str(c)])
        self.assertNotIn(UNREADABLE_CONFORMER_TRSH_LEVEL_MESSAGE, restarted.output[label]['warnings'])

    def test_i112_idle_recovery_submits_with_the_adapter_of_the_adaptive_level(self):
        """
        A species whose conf_opt jobs run at an adaptive level in another software than the conformer level, after
        a noted troubleshooting round whose jobs all ended with no energy, is resubmitted on restart with the
        adapter of the level its jobs actually run at.
        """
        label = 'methylamine'
        adaptive_level = Level(method='b3lyp', basis='def2-svp', software='qchem')
        overrides = dict(adaptive_levels={(1, 'inf'): {('conf_opt',): adaptive_level}})
        self.assertEqual(Level(repr=default_levels_of_theory['conformer']).software, 'gaussian')
        sched = self._i109_live_scheduler(**overrides)
        self.assertEqual({job.job_adapter for job in sched.job_dict[label]['conf_opt'].values()}, {'qchem'})
        self._i109_end_conformer_jobs(sched, [0.0, 10.0, 20.0])
        with patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            sched.determine_most_stable_conformer(label)
        self.assertEqual(len(get_conformer_isomorphism_trsh_history(sched.output[label]['conformers'])), 1)
        for job_name in list(sched.running_jobs[label]):
            sched.running_jobs[label].remove(job_name)
            sched.save_restart_dict()

        restarted = self._i109_restart(sched, **overrides)
        self.assertFalse(restarted.running_jobs.get(label))
        with patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            restarted.run_conformer_jobs(labels=[label])
        jobs = restarted.job_dict[label]['conf_opt']
        self.assertEqual(sorted(jobs), [0, 1, 2])
        for job in jobs.values():
            self.assertEqual(job.level.software, 'qchem')
            self.assertEqual(job.job_adapter, job.level.software,
                             'the job runs one software\'s level with another software\'s adapter')

    def _i112_conf_sp_restart(self, trsh_ess_jobs=True):
        """
        Run a conformer round with conformer single points that fails isomorphism, let the real troubleshooting
        respawn every conformer at its first level, then: conformer 0's conf_opt ends with an energy and its conf_sp
        is submitted, conformer 1's conf_opt fails, conformer 2's conf_opt and conf_sp end with an energy. Restart
        from the restart file that run left behind, with only conformer 0's conf_sp running.

        Args:
            trsh_ess_jobs (bool, optional): Whether the restarted run troubleshoots ESS jobs.

        Returns:
            Scheduler: The restarted scheduler, with ``schedule_jobs`` not yet called.
        """
        label = 'methylamine'
        overrides = dict(conformer_sp_level=Level(repr=default_levels_of_theory['sp']),
                         job_types=dict(self.job_types1, conf_sp=True))
        sched = self._i109_live_scheduler(**overrides)
        self._i109_end_conformer_jobs(sched, [0.0, 10.0, 20.0])
        with patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            sched.determine_most_stable_conformer(label)
        self.assertEqual(get_conformer_isomorphism_trsh_history(sched.output[label]['conformers']),
                         ['conformer wb97xd/def2TZVP'])
        self._i109_end_conformer_jobs(sched, [30.0, None, 25.0])
        with patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            sched.run_sp_job(label=label, conformer=0)
        self.assertEqual(sched.running_jobs[label], ['conf_sp_0'])
        sched.save_restart_dict()
        restarted = self._i109_restart(sched, **overrides)
        restarted.trsh_ess_jobs = trsh_ess_jobs
        self.assertEqual(sorted(restarted.job_dict[label]['conf_sp']), [0])
        self.assertFalse(restarted.job_dict[label].get('conf_opt'))
        self.assertEqual(restarted.species_dict[label].conformer_energies, [30.0, None, 25.0])
        return restarted

    def test_i112_restart_with_only_conf_sp_running_resubmits_at_the_noted_level(self):
        """
        A restart that finds only a conf_sp job running after a noted troubleshooting round resubmits the sibling
        whose conf_opt failed at the noted level, and nothing at the conformer level.
        """
        label = 'methylamine'
        restarted = self._i112_conf_sp_restart()
        with patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            restarted.run_conformer_jobs(labels=[label])
        jobs = restarted.job_dict[label].get('conf_opt', dict())
        self.assertEqual([(i, self._i109_level(job.level)) for i, job in sorted(jobs.items())],
                         [(1, 'wb97xd/def2tzvp')],
                         'a conformer was resubmitted at the conformer level after a troubleshooting round')

    def test_i112_restart_with_only_conf_sp_running_and_trsh_off_submits_nothing(self):
        """
        With ESS troubleshooting off, a restart that finds only a conf_sp job running after a noted troubleshooting
        round submits nothing, and warns once naming the species.
        """
        label = 'methylamine'
        restarted = self._i112_conf_sp_restart(trsh_ess_jobs=False)
        with patch.object(Scheduler, 'run_job') as run_job, patch('arc.scheduler.logger') as logger:
            restarted.run_conformer_jobs(labels=[label])
        run_job.assert_not_called()
        warnings = [str(c) for c in logger.warning.call_args_list
                    if REFUSED_CONFORMER_TRSH_RECOVERY_MESSAGE in str(c)]
        self.assertEqual(len(warnings), 1)
        self.assertIn(label, warnings[0])

    def _i109_negative_freq_replacement(self, sched):
        """
        Replace the species' conformers with the real ``troubleshoot_negative_freq``, which submits the new set at
        the conformer level through the real ``run_job``, and save the restart file.

        Args:
            sched (Scheduler): A live-run scheduler.
        """
        label = 'methylamine'
        new_confs = [str_to_xyz(self._METHYLAMINE_XYZ)] * 3
        with patch('arc.scheduler.trsh_negative_freq', return_value=([], new_confs, [], [])), \
                patch.object(JobAdapter, 'delete'), patch.object(JobAdapter, 'execute'), \
                patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'):
            sched.troubleshoot_negative_freq(label=label, job=MagicMock())
        sched.save_restart_dict()

    def test_i109_negative_freq_replacement_ends_the_trsh_history_in_the_restart_file(self):
        """
        ``troubleshoot_negative_freq`` marks the end of the replaced conformer set's troubleshooting history in the
        conformers note; the restart file keeps the marker, so the reloaded history is empty and later notes count.
        """
        label = 'methylamine'
        sched = self._i109_live_scheduler()
        self._i109_end_conformer_jobs(sched, [0.0, 10.0, 20.0])
        with patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            sched.determine_most_stable_conformer(label)
        self._i109_negative_freq_replacement(sched)
        restart_dict = read_yaml_file(sched.restart_path)
        note = restart_dict['output'][label]['conformers']
        self.assertTrue(note.endswith(CONFORMER_SET_REPLACED_NOTE))
        self.assertIn('Troubleshooting conformer isomorphism at wb97xd/def2TZVP; ', note)
        self.assertEqual(get_conformer_isomorphism_trsh_history(note), list())
        self.assertEqual(get_conformer_isomorphism_trsh_history(
            note + 'Troubleshooting conformer isomorphism at apfd/def2TZVP; '), ['conformer apfd/def2TZVP'])

        restarted = self._i109_restart(sched)
        self.assertEqual(get_conformer_isomorphism_trsh_history(restarted.output[label]['conformers']), list())
        self.assertEqual(sorted(restarted.job_dict[label]['conf_opt']), [0, 1, 2])
        self.assertEqual(get_conformer_trsh_levels(restarted.job_dict[label]['conf_opt'].values()), list())

    def test_i109_restart_mid_respawn_after_a_negative_freq_replacement_does_not_select_across_levels(self):
        """
        A species troubleshot at the first troubleshooting level has its conformers replaced by
        ``troubleshoot_negative_freq``; the new set fails isomorphism and the respawn, again at the first level,
        is killed after conformer 0. After a restart, conformers 1 and 2 are resubmitted at that level, not kept
        with their conformer-level energies, although the earlier set's note names the level as tried.
        """
        label = 'methylamine'
        sched = self._i109_live_scheduler()
        self._i109_end_conformer_jobs(sched, [0.0, 10.0, 20.0])
        with patch.object(Scheduler, 'check_max_simultaneous_jobs_limit'), patch.object(JobAdapter, 'execute'):
            sched.determine_most_stable_conformer(label)
        self.assertEqual(get_conformer_isomorphism_trsh_history(sched.output[label]['conformers']),
                         ['conformer wb97xd/def2TZVP'])
        self._i109_negative_freq_replacement(sched)
        self.assertEqual(sorted(sched.running_jobs[label]), ['conf_opt_0', 'conf_opt_1', 'conf_opt_2'])
        self._i109_end_conformer_jobs(sched, [0.0, 10.0, 20.0])
        self._i109_kill_mid_respawn(sched, kill_after=1)
        self.assertEqual(self._i109_level(sched.job_dict[label]['conf_opt'][0].level), 'wb97xd/def2tzvp')

        restarted = self._i109_restart(sched)
        submitted, energies_at_decision = self._i109_drive_restart(restarted, {0: 30.0, 1: 20.0, 2: 10.0})
        self.assertEqual(energies_at_decision, [[30.0, 20.0, 10.0]],
                         'the conformer was selected from energies of two levels')
        self.assertEqual([(c['conformer'], self._i109_level(c['level_of_theory'])) for c in submitted],
                         [(1, 'wb97xd/def2tzvp'), (2, 'wb97xd/def2tzvp')])

    def _issue622_scheduler_after_negative_freq_trsh(self):
        """
        Build a Scheduler whose species went through one conformer isomorphism troubleshooting round, then had
        troubleshoot_negative_freq replace its conformers with a perturbed set that optimized non-isomorphic.

        Returns:
            tuple: The scheduler and the perturbed conformer geometries.
        """
        label = 'methylamine'
        sched = self._issue622_fresh_scheduler()
        sched.output[label]['conformers'] += 'Troubleshooting conformer isomorphism at wb97xd/def2TZVP; '
        sched.trsh_ess_jobs = True
        perturbed = [str_to_xyz(self._ISSUE622_BROKEN_XYZ), str_to_xyz(self._METHYLAMINE_XYZ)]

        def run_job(**kwargs):
            sched.job_dict[label]['conf_opt'][kwargs['conformer']] = \
                MagicMock(job_adapter='gaussian', ess_trsh_methods=list(kwargs.get('ess_trsh_methods') or list()))

        with patch('arc.scheduler.trsh_negative_freq', return_value=([], perturbed, [], [])), \
                patch.object(Scheduler, 'delete_all_species_jobs'), \
                patch.object(Scheduler, 'run_job', side_effect=run_job):
            sched.troubleshoot_negative_freq(label=label, job=MagicMock())
        sched.species_dict[label].conformers = [str_to_xyz(self._ISSUE622_BROKEN_XYZ)] * 2
        sched.species_dict[label].conformer_energies = [0.0, 5.0]
        return sched, perturbed

    def test_issue622_negative_freq_trsh_restarts_the_conformer_isomorphism_ladder(self):
        """
        A perturbed conformer set from troubleshoot_negative_freq that optimizes non-isomorphic is troubleshot
        from the first level, although the notes name that level as tried on the previous set.
        """
        label = 'methylamine'
        sched, _ = self._issue622_scheduler_after_negative_freq_trsh()
        with patch.object(Scheduler, 'run_job') as run_job:
            sched.troubleshoot_conformer_isomorphism(label=label)
        self.assertEqual(run_job.call_count, 2)
        self.assertEqual({c.kwargs['level_of_theory'] for c in run_job.call_args_list}, {'wb97xd/def2TZVP'})

    def test_issue622_negative_freq_trsh_respawns_the_perturbed_geometries(self):
        """
        Conformer isomorphism troubleshooting after troubleshoot_negative_freq respawns the perturbed geometries,
        not the conformers the species had before the perturbation.
        """
        label = 'methylamine'
        sched, perturbed = self._issue622_scheduler_after_negative_freq_trsh()
        with patch.object(Scheduler, 'run_job') as run_job:
            sched.troubleshoot_conformer_isomorphism(label=label)
        self.assertEqual([c.kwargs['xyz'] for c in run_job.call_args_list], perturbed)

    def test_check_negative_freq(self):
        """Test the check_negative_freq() method"""
        label = 'C2H6'
        self.job3.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'C2H6_freq_QChem.out')
        self.job3.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]
        vibfreqs = parser.parse_frequencies(log_file_path=self.job3.local_path_to_output_file)
        self.assertTrue(self.sched1.check_negative_freq(label=label, job=self.job3, vibfreqs=vibfreqs)[0])
        # Unparsed frequencies (``None``) must be treated as an unsuccessful freq job, not crash.
        self.assertFalse(self.sched1.check_negative_freq(label=label, job=self.job3, vibfreqs=None)[0])
        # An empty freqs list for a polyatomic non-TS species must not sneak through as successful.
        self.assertFalse(self.sched1.check_negative_freq(label=label, job=self.job3, vibfreqs=list())[0])

    def test_post_freq_actions(self):
        """
        Test the post_freq_actions() method.

        This is the whole success path of a freq job, shared by check_freq_job() and by pipe mode.
        Passing the imaginary frequency check is not on its own enough to call a freq job done:
        species.freqs and the freq.out copy under the species output folder must exist too, or
        consumers such as compute_rxn_e0() have nothing to read.
        """
        label = 'C2H6'
        self.job3.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'C2H6_freq_QChem.out')
        self.job3.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]
        freq_path = os.path.join(self.project_directory, 'output', 'Species', label, 'geometry', 'freq.out')
        if os.path.isfile(freq_path):
            os.remove(freq_path)
        self.sched1.species_dict[label].freqs = None

        vibfreqs = parser.parse_frequencies(log_file_path=self.job3.local_path_to_output_file)
        freq_ok, switch_ts = self.sched1.post_freq_actions(label=label, job=self.job3, vibfreqs=vibfreqs)
        self.assertTrue(freq_ok)
        self.assertFalse(switch_ts)
        self.assertTrue(self.sched1.output[label]['job_types']['freq'])
        self.assertEqual(self.sched1.species_dict[label].freqs, [float(f) for f in vibfreqs])
        self.assertTrue(os.path.isfile(freq_path))

        # A failed check must leave none of the above behind.
        os.remove(freq_path)
        self.sched1.species_dict[label].freqs = None
        self.sched1.output[label]['job_types']['freq'] = False
        freq_ok, switch_ts = self.sched1.post_freq_actions(label=label, job=self.job3,
                                                           vibfreqs=[-500.0] + list(vibfreqs))
        self.assertFalse(freq_ok)
        self.assertFalse(switch_ts)
        self.assertFalse(self.sched1.output[label]['job_types']['freq'])
        self.assertIsNone(self.sched1.species_dict[label].freqs)
        self.assertFalse(os.path.isfile(freq_path))

    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_check_freq_job_does_not_switch_ts_when_nmd_cannot_be_run(self, mock_switch_ts):
        """
        Test that a TS is not discarded when the normal mode displacement check cannot be performed.

        The forming and breaking bond indices are indices into the concatenated reactant atoms, so a
        TS geometry spanning a different number of atoms cannot be analyzed at all. Reporting that as
        a failed check calls switch_ts() and throws away a TS that was never examined, and switching
        to another guess cannot change the atom count that made the analysis impossible.
        """
        oh_xyz = str_to_xyz("""O       0.48890387    0.00000000    0.00000000
H      -0.48890387    0.00000000    0.00000000""")
        h2o_xyz = str_to_xyz("""O      -0.00032832    0.39781490    0.00000000
H      -0.76330345   -0.19953755    0.00000000
H       0.76363177   -0.19827735    0.00000000""")
        ch4_xyz = str_to_xyz("""C      -0.00000000    0.00000000    0.00000000
H      -0.65055201   -0.77428020   -0.41251879
H      -0.34927558    0.98159583   -0.32768232
H      -0.02233792   -0.04887375    1.09087665
H       1.02216551   -0.15844188   -0.35067554""")
        ts_xyz = ch4_xyz
        rxn = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C', xyz=ch4_xyz),
                                     ARCSpecies(label='OH', smiles='[OH]', xyz=oh_xyz)],
                          p_species=[ARCSpecies(label='CH3', smiles='[CH3]', xyz=str_to_xyz(
                              """C       0.00000000    0.00000001   -0.00000000
H       1.06690511   -0.17519582    0.05416493
H      -0.68531716   -0.83753536   -0.02808565
H      -0.38158795    1.01273118   -0.02607927""")),
                                     ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz)])
        ts_label = 'TS_short'
        ts_spc = ARCSpecies(label=ts_label, is_ts=True, xyz=ts_xyz, multiplicity=3, charge=0, compute_thermo=False)
        ts_spc.ts_guesses = [TSGuess(index=0, method='heuristics', success=True, energy=0.0, xyz=ts_xyz,
                                     execution_time='0:00:01')]
        ts_spc.ts_guesses[0].opt_xyz = ts_xyz
        ts_spc.chosen_ts = 0
        ts_spc.rxn_index = 0
        rxn.ts_species = ts_spc
        rxn.ts_label = ts_label

        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_nmd_no_switch_ts')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='test_nmd_no_switch', ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )
        sched.rxn_dict[0] = rxn

        job = job_factory(job_adapter='gaussian', project='test_nmd_no_switch', ess_settings=self.ess_settings,
                          species=[ts_spc], job_type='freq',
                          level=Level(repr=default_levels_of_theory['freq']),
                          project_directory=project_directory, job_num=105)
        job.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'CHO_neg_freq.out')
        job.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]
        sched.check_freq_job(label=ts_label, job=job)
        self.assertIsNone(sched.species_dict[ts_label].ts_checks['NMD'])
        mock_switch_ts.assert_not_called()

    def test_determine_adaptive_level(self):
        """Test the determine_adaptive_level() method"""
        # adaptive_levels get converted to ``Level`` objects in main, but here we skip main and test Scheduler directly
        adaptive_levels = {(1, 5):      {('opt', 'freq'): Level(repr='wb97xd/6-311+g(2d,2p)'),
                                         ('sp',): Level(repr='ccsd(t)-f12/aug-cc-pvtz-f12')},
                           (6, 15):     {('opt', 'freq'): Level(repr='b3lyp/cbsb7'),
                                         ('sp',): Level(repr='dlpno-ccsd(t)/def2-tzvp')},
                           (16, 30):    {('opt', 'freq'): Level(repr='b3lyp/6-31g(d,p)'),
                                         ('sp',): Level(repr='wb97xd/6-311+g(2d,2p)')},
                           (31, 'inf'): {('opt', 'freq'): Level(repr='b3lyp/6-31g(d,p)'),
                                         ('sp',): Level(repr='b3lyp/6-311+g(d,p)')}}

        sched2 = Scheduler(project='project_test',
                           ess_settings=self.ess_settings,
                           species_list=[self.spc1, self.spc2],
                           composite_method=None,
                           conformer_opt_level=default_levels_of_theory['conformer'],
                           opt_level=default_levels_of_theory['opt'],
                           freq_level=default_levels_of_theory['freq'],
                           sp_level=default_levels_of_theory['sp'],
                           scan_level=default_levels_of_theory['scan'],
                           ts_guess_level=default_levels_of_theory['ts_guesses'],
                           project_directory=self.project_directory,
                           testing=True, job_types=self.job_types1,
                           orbitals_level=default_levels_of_theory['orbitals'],
                           adaptive_levels=adaptive_levels)
        original_level = Level(method='CBS-QB3')
        level1 = sched2.determine_adaptive_level(original_level_of_theory=original_level,
                                                 job_type='opt',
                                                 heavy_atoms=5)
        level2 = sched2.determine_adaptive_level(original_level_of_theory=original_level,
                                                 job_type='freq',
                                                 heavy_atoms=5)
        level3 = sched2.determine_adaptive_level(original_level_of_theory=original_level,
                                                 job_type='opt',
                                                 heavy_atoms=20)
        level4 = sched2.determine_adaptive_level(original_level_of_theory=original_level,
                                                 job_type='composite',
                                                 heavy_atoms=50)
        level5 = sched2.determine_adaptive_level(original_level_of_theory=original_level,
                                                 job_type='orbitals',
                                                 heavy_atoms=5)
        level6 = sched2.determine_adaptive_level(original_level_of_theory=original_level,
                                                 job_type='sp',
                                                 heavy_atoms=7)
        level7 = sched2.determine_adaptive_level(original_level_of_theory=original_level,
                                                 job_type='sp',
                                                 heavy_atoms=25)
        self.assertEqual(level1.simple(), 'wb97xd/6-311+g(2d,2p)')
        self.assertEqual(level2.simple(), 'wb97xd/6-311+g(2d,2p)')
        self.assertEqual(level3.simple(), 'b3lyp/6-31g(d,p)')
        self.assertEqual(level4.simple(), 'cbs-qb3')
        self.assertEqual(level5.simple(), 'cbs-qb3')
        self.assertEqual(level6.simple(), 'dlpno-ccsd(t)/def2-tzvp')
        self.assertEqual(level7.simple(), 'wb97xd/6-311+g(2d,2p)')

    def test_initialize_output_dict(self):
        """Test Scheduler.initialize_output_dict"""
        self.sched1.output['C2H6']['info'] = 'some text'
        self.assertTrue(self.sched1._does_output_dict_contain_info())
        self.sched1.output = dict()
        self.assertEqual(self.sched1.output, dict())
        self.sched1.initialize_output_dict()
        self.assertFalse(self.sched1._does_output_dict_contain_info())
        empty_species_dict = {'conformers': '',
                              'convergence': None,
                              'errors': '',
                              'info': '',
                              'isomorphism': '',
                              'job_types': {'rotors': True,
                                            'composite': False,
                                            'conf_opt': False,
                                            'conf_sp': False,
                                            'fine': False,
                                            'freq': False,
                                            'lennard_jones': False,
                                            'onedmin': False,
                                            'opt': False,
                                            'orbitals': False,
                                            'sp': False},
                              'paths': {'composite': '', 'freq': '', 'geo': '', 'geo_coarse': '', 'sp': ''},
                              'restart': '', 'warnings': '',
                              'conformer_verified': None, 'conformer_verified_cause': ''}
        initialized_output_dict = {'C2H6': empty_species_dict,
                                   'CtripCO': empty_species_dict,
                                   'methylamine': empty_species_dict,
                                   }
        self.assertEqual(self.sched1.output, initialized_output_dict)

    def test_stability_does_not_gate_convergence(self):
        """Test that the stability diagnostic never holds a species back from converging"""
        original_output = self.sched1.output
        original_job_types = self.sched1.job_types
        self.addCleanup(setattr, self.sched1, 'output', original_output)
        self.addCleanup(setattr, self.sched1, 'job_types', original_job_types)
        self.sched1.output = dict()
        self.sched1.initialize_output_dict()
        self.sched1.job_types = dict(original_job_types)
        self.sched1.job_types['stability'] = True
        label = 'C2H6'

        self.sched1.output[label]['job_types'] = {job_type: True for job_type in self.sched1.job_types}
        self.sched1.output[label]['job_types']['stability'] = False
        self.sched1.output[label]['convergence'] = None
        self.sched1.check_all_done(label=label)
        self.assertTrue(self.sched1.output[label]['convergence'],
                        msg='an unrun stability diagnostic held the species back from converging')

        self.sched1.output[label]['job_types'] = {job_type: True for job_type in self.sched1.job_types}
        self.sched1.output[label]['job_types']['sp'] = False
        self.sched1.output[label]['convergence'] = None
        self.sched1.check_all_done(label=label)
        self.assertNotEqual(self.sched1.output[label]['convergence'], True,
                            msg='the stability exemption is over-broad: a missing sp job still converged')

    def test_stability_lookup_survives_a_restart_predating_the_job_type(self):
        """Test that enabling the diagnostic cannot raise on a restart.yml written without it"""
        original_output = self.sched1.output
        original_job_types = self.sched1.job_types
        self.addCleanup(setattr, self.sched1, 'output', original_output)
        self.addCleanup(setattr, self.sched1, 'job_types', original_job_types)
        label = 'C2H6'
        restart_shaped = {'conf_opt': True, 'opt': True, 'fine': False, 'freq': True, 'sp': True,
                          'rotors': True, 'orbitals': False, 'lennard_jones': False, 'conf_sp': False,
                          'composite': False, 'onedmin': False}
        self.sched1.output = {label: {'job_types': dict(restart_shaped), 'paths': {}, 'convergence': None,
                                      'conformers': '', 'isomorphism': '', 'restart': '', 'errors': '',
                                      'warnings': '', 'info': ''}}
        self.sched1.job_types = dict(restart_shaped)
        self.sched1.job_types['stability'] = True
        self.assertNotIn('stability', self.sched1.output[label]['job_types'])
        self.sched1.check_all_done(label=label)
        self.assertTrue(self.sched1.output[label]['convergence'])

    def test_errored_orbitals_job_is_still_rerun_on_memory_error(self):
        """Test that the stability diagnostic did not change how an errored orbitals job is handled"""
        for job_type, expected_rerun in [('orbitals', True), ('stability', False)]:
            job = MagicMock()
            job.job_type = job_type
            job.job_name = f'{job_type}_a1'
            job.job_id = 1
            job.job_memory_gb = 14
            job.job_status = ['done', {'status': 'errored', 'keywords': ['memory'],
                                       'error': 'Insufficient job memory'}]
            self.sched1.running_jobs['C2H6'] = [job.job_name]
            with patch.object(self.sched1, '_run_a_job') as run_a_job:
                self.sched1.end_job(job=job, label='C2H6', job_name=job.job_name)
            self.assertEqual(run_a_job.called, expected_rerun,
                             msg=f'{job_type} job re-run was {run_a_job.called}, expected {expected_rerun}')

    def _completed_geometry_job(self, job_adapter, job_type, check_file_name, orbitals=b'orbitals'):
        """Build a stand-in for a completed geometry job holding a downloaded orbitals file."""
        local_path = tempfile.mkdtemp(prefix='arc_test_scheduler_end_job_')
        self.addCleanup(shutil.rmtree, local_path, ignore_errors=True)
        with open(os.path.join(local_path, 'output.out'), 'w') as f:
            f.write('output')
        if orbitals is not None:
            with open(os.path.join(local_path, check_file_name), 'wb') as f:
                f.write(orbitals)
        job = MagicMock()
        job.job_adapter = job_adapter
        job.job_type = job_type
        job.job_name = f'{job_type}_a1'
        job.job_id = 1
        job.check_file_name = check_file_name
        job.local_path = local_path
        job.local_path_to_output_file = os.path.join(local_path, 'output.out')
        job.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]
        job.directed_scan_type = None
        job.execution_type = 'queue'
        return job

    def _end_a_completed_job(self, job, label='C2H6'):
        """Run end_job for a completed job and return the checkfile the species came away with."""
        original_checkfile = self.sched1.species_dict[label].checkfile
        self.addCleanup(setattr, self.sched1.species_dict[label], 'checkfile', original_checkfile)
        self.sched1.species_dict[label].checkfile = None
        self.sched1.running_jobs[label] = [job.job_name]
        with patch.object(self.sched1, 'save_restart_dict'):
            terminated = self.sched1.end_job(job=job, label=label, job_name=job.job_name)
        self.assertTrue(terminated)
        return self.sched1.species_dict[label].checkfile

    def test_end_job_adopts_the_orbitals_file_its_ess_names(self):
        """Test that an ORCA geometry job hands the species its input.gbw, as Gaussian does its check.chk"""
        for job_adapter, check_file_name in [('orca', 'input.gbw'), ('gaussian', 'check.chk')]:
            for job_type in ['opt', 'optfreq', 'composite']:
                job = self._completed_geometry_job(job_adapter=job_adapter, job_type=job_type,
                                                   check_file_name=check_file_name)
                self.assertEqual(self._end_a_completed_job(job),
                                 os.path.join(job.local_path, check_file_name),
                                 msg=f'a {job_adapter} {job_type} job did not hand over its {check_file_name}')

    def test_end_job_adopts_no_orbitals_from_a_job_that_is_not_a_geometry_job(self):
        """Test that only the job types the guess chain reads from hand over their orbitals"""
        for job_type in ['sp', 'freq', 'scan']:
            job = self._completed_geometry_job(job_adapter='orca', job_type=job_type,
                                               check_file_name='input.gbw')
            self.assertIsNone(self._end_a_completed_job(job), msg=f'a {job_type} job handed over orbitals')

    def test_end_job_refuses_an_empty_orbitals_file(self):
        """Test that the zero-byte file a failed download leaves behind is not adopted"""
        job = self._completed_geometry_job(job_adapter='orca', job_type='opt',
                                           check_file_name='input.gbw', orbitals=b'')
        self.assertTrue(os.path.isfile(os.path.join(job.local_path, 'input.gbw')))
        self.assertEqual(os.path.getsize(os.path.join(job.local_path, 'input.gbw')), 0)
        with self.assertLogs('arc', level='INFO') as captured:
            checkfile = self._end_a_completed_job(job)
        self.assertIsNone(checkfile)
        self.assertIn('input.gbw', '\n'.join(captured.output))

    def test_end_job_adopts_no_orbitals_when_the_download_left_nothing(self):
        """Test that a job whose orbitals never came back leaves the species without a checkfile"""
        job = self._completed_geometry_job(job_adapter='orca', job_type='opt',
                                           check_file_name='input.gbw', orbitals=None)
        self.assertFalse(os.path.isfile(os.path.join(job.local_path, 'input.gbw')))
        self.assertIsNone(self._end_a_completed_job(job))

    def _stability_opt_job(self, checkfile, method='wb97xd', adapter='gaussian',
                           basis='def2-TZVP', restricted_used=None, fine=False):
        """Build a minimal stand-in for a converged Gaussian opt job."""
        job = MagicMock()
        job.job_adapter = adapter
        job.job_name = 'opt_a1'
        job.job_type = 'opt'
        job.fine = fine
        job.restricted_used = restricted_used
        job.level = Level(method=method, basis=basis) if basis is not None else Level(method=method)
        job.local_path_to_check_file = checkfile
        job.local_path_to_output_file = '/nonexistent/opt.out'
        return job

    def _prepare_stability_ts(self, label='C2H6', checkfile=None, is_ts=True, enabled=True):
        """Point a scheduler species at a checkfile and enable the stability diagnostic."""
        species = self.sched1.species_dict[label]
        original_job_types = self.sched1.job_types
        original_checkfile = species.checkfile
        original_is_ts = species.is_ts
        original_final_xyz = species.final_xyz
        original_jobs = self.sched1.job_dict.get(label)
        self.addCleanup(setattr, self.sched1, 'job_types', original_job_types)
        self.addCleanup(setattr, species, 'checkfile', original_checkfile)
        self.addCleanup(setattr, species, 'is_ts', original_is_ts)
        self.addCleanup(setattr, species, 'final_xyz', original_final_xyz)
        self.addCleanup(setattr, species, 'stability_analysis_ran', False)
        self.addCleanup(setattr, species, 'stability_pending_opt_job', None)
        self.addCleanup(setattr, species, 'stability_reoptimized', False)
        self.addCleanup(setattr, species, 'derived_stability_verdict', None)

        def _restore_jobs():
            if original_jobs is None:
                self.sched1.job_dict.pop(label, None)
            else:
                self.sched1.job_dict[label] = original_jobs
        self.addCleanup(_restore_jobs)

        job_types = initialize_job_types(dict())
        job_types.update(original_job_types)
        job_types['stability'] = enabled
        self.sched1.job_types = job_types
        species.is_ts = is_ts
        species.checkfile = checkfile
        species.stability_analysis_ran = False
        species.stability_pending_opt_job = None
        species.stability_reoptimized = False
        species.derived_stability_verdict = None
        species.final_xyz = {'symbols': ('O', 'H'), 'isotopes': (16, 1),
                             'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))}
        self.sched1.job_dict[label] = dict()
        return label

    def _spawn_post_opt(self, label, job, job_name='opt_a1'):
        """Drive spawn_post_opt_jobs for an opt job and return the run_job mock it spawned through."""
        self.sched1.job_dict[label]['opt'] = {job_name: job}
        self.sched1.output[label]['paths']['geo'] = ''
        with patch.object(self.sched1, 'run_scan_jobs'), \
                patch.object(self.sched1, 'spawn_ts_jobs'), \
                patch.object(self.sched1, 'run_job') as run_job:
            self.sched1.spawn_post_opt_jobs(label=label, job_name=job_name)
        return run_job

    def test_stability_job_spawned_from_the_opt_job_state(self):
        """Test that the stability job takes its level and orbitals from the opt job and the converged geometry"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        label = self._prepare_stability_ts(checkfile=checkfile)
        job = self._stability_opt_job(checkfile=checkfile)
        with patch.object(self.sched1, 'run_job') as run_job:
            self.assertTrue(self.sched1.run_stability_job(label=label, opt_job=job))
        self.assertTrue(run_job.called)
        kwargs = run_job.call_args.kwargs
        self.assertEqual(kwargs['job_type'], 'stability')
        self.assertEqual(kwargs['job_adapter'], 'gaussian')
        self.assertIs(kwargs['xyz'], self.sched1.species_dict[label].final_xyz)
        self.assertIs(kwargs['level_of_theory'], job.level)
        self.assertTrue(self.sched1.species_dict[label].stability_analysis_ran)

    def test_stability_job_not_spawned_at_a_level_with_no_broken_symmetry_reference(self):
        """Test that the analysis runs at the levels a broken-symmetry reference describes and no other"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        label = self._prepare_stability_ts(checkfile=checkfile)
        species = self.sched1.species_dict[label]
        for method, spawned in [('wb97xd', True), ('hf', True), ('ccsd(t)', False), ('mp2', False)]:
            species.stability_analysis_ran = False
            job = self._stability_opt_job(checkfile=checkfile, method=method, restricted_used=True)
            with patch.object(self.sched1, 'run_job') as run_job:
                self.assertEqual(self.sched1.run_stability_job(label=label, opt_job=job), spawned,
                                 msg=f'an optimization at {method} was not handled as {spawned}')
            self.assertEqual(run_job.called, spawned)

    def test_stability_job_not_spawned_when_checkfile_superseded(self):
        """Test that a species holding a different checkfile than its opt job wrote is skipped"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f_old, \
                tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f_new:
            old_checkfile, new_checkfile = f_old.name, f_new.name
        for path in (old_checkfile, new_checkfile):
            self.addCleanup(lambda p=path: os.path.isfile(p) and os.remove(p))
        label = self._prepare_stability_ts(checkfile=new_checkfile)
        job = self._stability_opt_job(checkfile=old_checkfile)
        with patch.object(self.sched1, 'run_job') as run_job:
            self.assertFalse(self.sched1.run_stability_job(label=label, opt_job=job))
        self.assertFalse(run_job.called)

    def test_stability_job_not_spawned_without_a_checkfile(self):
        """Test that an opt job with no checkfile is skipped rather than run with guess=mix"""
        label = self._prepare_stability_ts(checkfile=None)
        job = self._stability_opt_job(checkfile=None)
        with patch.object(self.sched1, 'run_job') as run_job:
            self.assertFalse(self.sched1.run_stability_job(label=label, opt_job=job))
        self.assertFalse(run_job.called)

    def test_stability_job_not_spawned_for_a_job_that_is_not_a_submitted_ess_job(self):
        """Test that a job carrying no ESS state is skipped"""
        label = self._prepare_stability_ts(checkfile=None)
        piped = SimpleNamespace(local_path_to_output_file='/nonexistent/opt.out',
                                level=Level(method='wb97xd', basis='def2-TZVP'),
                                job_status=['done', {'status': 'done'}])
        with patch.object(self.sched1, 'run_job') as run_job:
            self.assertFalse(self.sched1.run_stability_job(label=label, opt_job=piped))
        self.assertFalse(run_job.called)

    def test_stability_job_not_spawned_twice(self):
        """Test that a TS gets at most one stability job, and that the guard is species state"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        label = self._prepare_stability_ts(checkfile=checkfile)
        job = self._stability_opt_job(checkfile=checkfile)
        self.sched1.species_dict[label].stability_analysis_ran = True
        with patch.object(self.sched1, 'run_job') as run_job:
            self.assertFalse(self.sched1.run_stability_job(label=label, opt_job=job))
        self.assertFalse(run_job.called)

    def test_stability_job_reports_an_ess_it_is_not_implemented_for(self):
        """Test that an opt job of an unsupported ESS is refused with a warning naming it, once per ESS"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        self.addCleanup(self.sched1.stability_unimplemented_ess.clear)
        self.sched1.stability_unimplemented_ess.clear()
        label = self._prepare_stability_ts(checkfile=checkfile)
        job = self._stability_opt_job(checkfile=checkfile, adapter='qchem')
        with patch.object(self.sched1, 'run_job') as run_job:
            with self.assertLogs('arc', level='WARNING') as captured:
                self.sched1.run_stability_job(label=label, opt_job=job)
        self.assertFalse(run_job.called)
        message = '\n'.join(captured.output)
        self.assertIn(label, message)
        self.assertIn('qchem', message)
        for ess in STABILITY_ANALYSIS_ADAPTERS:
            self.assertIn(ess, message)
        self.assertEqual(self.sched1.stability_unimplemented_ess, {'qchem'})

        with patch.object(self.sched1, 'run_job') as run_job, \
                patch('arc.scheduler.logger') as mocked_logger:
            self.sched1.run_stability_job(label=label, opt_job=job)
        self.assertFalse(run_job.called)
        self.assertFalse(mocked_logger.warning.called)

        molpro_job = self._stability_opt_job(checkfile=checkfile, adapter='molpro')
        with patch.object(self.sched1, 'run_job') as run_job, \
                patch('arc.scheduler.logger') as mocked_logger:
            self.sched1.run_stability_job(label=label, opt_job=molpro_job)
        self.assertFalse(run_job.called)
        self.assertTrue(mocked_logger.warning.called)
        self.assertIn('molpro', mocked_logger.warning.call_args.args[0])
        self.assertEqual(self.sched1.stability_unimplemented_ess, {'qchem', 'molpro'})

    def _adopted_reference_ts(self, label='C2H6'):
        """Point a scheduler species at an adopted external instability and return its label."""
        label = self._prepare_stability_ts(label=label, checkfile=None, is_ts=True)
        species = self.sched1.species_dict[label]
        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertTrue(adopted_reference_is_unrestricted(species))
        return label

    def test_an_unbreakable_unrestricted_reference_is_reported_once_per_ess(self):
        """Test that an ESS offered neither symmetry-breaking mechanism is warned about, once per ESS"""
        self.addCleanup(self.sched1.unbreakable_reference_ess.clear)
        self.sched1.unbreakable_reference_ess.clear()
        label = self._adopted_reference_ts()
        job = self._stability_opt_job(checkfile=None, adapter='molpro', restricted_used=False)
        with self.assertLogs('arc', level='WARNING') as captured:
            self.sched1.warn_on_collapsible_unrestricted_reference(label=label, job=job)
        message = '\n'.join(captured.output)
        self.assertIn(label, message)
        self.assertIn('molpro', message)
        for ess in SYMMETRY_BREAKING_ADAPTERS:
            self.assertIn(ess, message)
        self.assertEqual(self.sched1.unbreakable_reference_ess, {'molpro'})

        with patch('arc.scheduler.logger') as mocked_logger:
            self.sched1.warn_on_collapsible_unrestricted_reference(label=label, job=job)
        self.assertFalse(mocked_logger.warning.called)

        qchem_job = self._stability_opt_job(checkfile=None, adapter='qchem', restricted_used=False)
        with patch('arc.scheduler.logger') as mocked_logger:
            self.sched1.warn_on_collapsible_unrestricted_reference(label=label, job=qchem_job)
        self.assertTrue(mocked_logger.warning.called)
        self.assertIn('qchem', mocked_logger.warning.call_args.args[0])
        self.assertEqual(self.sched1.unbreakable_reference_ess, {'molpro', 'qchem'})

    def test_a_collapsible_reference_is_recorded_on_every_species_it_is_reached_for(self):
        """Test that the species output warnings name each species, while the log names each ESS once"""
        self.addCleanup(self.sched1.unbreakable_reference_ess.clear)
        self.sched1.unbreakable_reference_ess.clear()
        label = self._adopted_reference_ts()
        original_warnings = self.sched1.output[label]['warnings']
        self.addCleanup(self.sched1.output[label].__setitem__, 'warnings', original_warnings)
        job = self._stability_opt_job(checkfile=None, adapter='molpro', restricted_used=False)
        self.sched1.warn_on_collapsible_unrestricted_reference(label=label, job=job)
        self.assertIn(COLLAPSED_REFERENCE_MESSAGE, self.sched1.output[label]['warnings'])

        with patch('arc.scheduler.logger') as mocked_logger:
            self.sched1.warn_on_collapsible_unrestricted_reference(label=label, job=job)
        self.assertFalse(mocked_logger.warning.called)
        self.assertEqual(self.sched1.output[label]['warnings'].count(COLLAPSED_REFERENCE_MESSAGE), 1)

    def test_the_collapsible_reference_report_tolerates_a_job_carrying_no_name(self):
        """Test that a job object with no job_name is reported rather than raising"""
        self.addCleanup(self.sched1.unbreakable_reference_ess.clear)
        self.sched1.unbreakable_reference_ess.clear()
        label = self._adopted_reference_ts()
        original_warnings = self.sched1.output[label]['warnings']
        self.addCleanup(self.sched1.output[label].__setitem__, 'warnings', original_warnings)
        job = SimpleNamespace(job_adapter='molpro',
                              restricted_used=False,
                              level=Level(method='wb97xd', basis='def2-TZVP'),
                              )
        with self.assertLogs('arc', level='WARNING') as captured:
            self.sched1.warn_on_collapsible_unrestricted_reference(label=label, job=job)
        self.assertIn('molpro', '\n'.join(captured.output))
        self.assertEqual(self.sched1.unbreakable_reference_ess, {'molpro'})

    def test_no_report_for_an_ess_arc_breaks_the_spin_symmetry_for(self):
        """Test that the ESSs ARC writes a guess or a directive for are not reported"""
        self.addCleanup(self.sched1.unbreakable_reference_ess.clear)
        self.sched1.unbreakable_reference_ess.clear()
        label = self._adopted_reference_ts()
        for adapter in sorted(SYMMETRY_BREAKING_ADAPTERS):
            job = self._stability_opt_job(checkfile=None, adapter=adapter, restricted_used=False)
            with patch('arc.scheduler.logger') as mocked_logger:
                self.sched1.warn_on_collapsible_unrestricted_reference(label=label, job=job)
            self.assertFalse(mocked_logger.warning.called, msg=f'{adapter} was reported')
        self.assertEqual(self.sched1.unbreakable_reference_ess, set())

    def test_no_report_without_an_adopted_verdict(self):
        """Test that a species whose reference the analysis did not decide is not reported"""
        self.addCleanup(self.sched1.unbreakable_reference_ess.clear)
        self.sched1.unbreakable_reference_ess.clear()
        label = self._prepare_stability_ts(checkfile=None, is_ts=True)
        for verdict in [None,
                        {'verdict': 'stable', 'restricted': True},
                        {'verdict': 'external_instability', 'restricted': False},
                        ]:
            self.sched1.species_dict[label].derived_stability_verdict = verdict
            job = self._stability_opt_job(checkfile=None, adapter='molpro', restricted_used=False)
            with patch('arc.scheduler.logger') as mocked_logger:
                self.sched1.warn_on_collapsible_unrestricted_reference(label=label, job=job)
            self.assertFalse(mocked_logger.warning.called, msg=f'the verdict {verdict} was reported')
        self.assertEqual(self.sched1.unbreakable_reference_ess, set())

    def test_no_report_for_a_job_that_declared_no_unrestricted_reference(self):
        """Test that the report follows the reference the job's input declared"""
        self.addCleanup(self.sched1.unbreakable_reference_ess.clear)
        self.sched1.unbreakable_reference_ess.clear()
        label = self._adopted_reference_ts()
        for restricted_used in [None, True, [False]]:
            job = self._stability_opt_job(checkfile=None, adapter='molpro', restricted_used=restricted_used)
            with patch('arc.scheduler.logger') as mocked_logger:
                self.sched1.warn_on_collapsible_unrestricted_reference(label=label, job=job)
            self.assertFalse(mocked_logger.warning.called,
                             msg=f'a job whose reference memo is {restricted_used} was reported')
        self.assertEqual(self.sched1.unbreakable_reference_ess, set())

    def test_stability_job_spawned_for_every_capable_ess(self):
        """Test that each ESS ARC implements the analysis for spawns the job in that ESS"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        self.addCleanup(self.sched1.stability_unimplemented_ess.clear)
        self.sched1.stability_unimplemented_ess.clear()
        label = self._prepare_stability_ts(checkfile=checkfile)
        for adapter in sorted(STABILITY_ANALYSIS_ADAPTERS):
            job = self._stability_opt_job(checkfile=checkfile, adapter=adapter)
            self.sched1.species_dict[label].stability_analysis_ran = False
            with patch.object(self.sched1, 'run_job') as run_job:
                self.sched1.run_stability_job(label=label, opt_job=job)
            self.assertTrue(run_job.called, msg=f'no stability job was spawned for {adapter}')
            self.assertEqual(run_job.call_args.kwargs['job_adapter'], adapter)
            self.assertEqual(run_job.call_args.kwargs['job_type'], 'stability')
        self.assertEqual(self.sched1.stability_unimplemented_ess, set())

    def test_stability_job_spawned_for_an_orca_gbw_checkfile(self):
        """Test that the checkfile identity gate reads an ORCA .gbw as it does a Gaussian .chk"""
        with tempfile.NamedTemporaryFile(suffix='.gbw', delete=False) as f_new, \
                tempfile.NamedTemporaryFile(suffix='.gbw', delete=False) as f_old:
            checkfile, old_checkfile = f_new.name, f_old.name
        for path in (checkfile, old_checkfile):
            self.addCleanup(lambda p=path: os.path.isfile(p) and os.remove(p))
        label = self._prepare_stability_ts(checkfile=checkfile)
        job = self._stability_opt_job(checkfile=checkfile, adapter='orca')
        with patch.object(self.sched1, 'run_job') as run_job:
            self.sched1.run_stability_job(label=label, opt_job=job)
        self.assertTrue(run_job.called)
        self.assertEqual(run_job.call_args.kwargs['job_adapter'], 'orca')

        self.sched1.species_dict[label].stability_analysis_ran = False
        self.assertTrue(os.path.isfile(old_checkfile),
                        msg='the superseded .gbw must exist, or the identity gate is never reached')
        superseded = self._stability_opt_job(checkfile=old_checkfile, adapter='orca')
        with patch.object(self.sched1, 'run_job') as run_job:
            self.sched1.run_stability_job(label=label, opt_job=superseded)
        self.assertFalse(run_job.called)

    def test_stability_job_not_spawned_for_a_non_dft_level(self):
        """Test that a level Gaussian offers no stability analysis for is skipped"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        label = self._prepare_stability_ts(checkfile=checkfile)
        for method, expected in [('ccsd(t)', False), ('cbs-qb3', False), ('hf', True), ('wb97xd', True)]:
            job = self._stability_opt_job(checkfile=checkfile, method=method)
            self.sched1.species_dict[label].stability_analysis_ran = False
            with patch.object(self.sched1, 'run_job') as run_job:
                self.sched1.run_stability_job(label=label, opt_job=job)
            self.assertEqual(run_job.called, expected, msg=f'{method} spawned={run_job.called}')

    def test_the_stability_gate_admits_a_restricted_species_that_is_not_a_ts(self):
        """Test that a well whose opt job declared a restricted reference is tested"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        label = self._prepare_stability_ts(checkfile=checkfile, is_ts=False)
        job = self._stability_opt_job(checkfile=checkfile, restricted_used=True)
        with patch.object(self.sched1, 'run_job') as run_job:
            self.assertTrue(self.sched1.run_stability_job(label=label, opt_job=job))
        self.assertEqual(run_job.call_args.kwargs['job_type'], 'stability')

    def test_the_stability_gate_refuses_an_unrestricted_species_that_is_not_a_ts(self):
        """Test that a well whose opt job already ran unrestricted is not tested"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        label = self._prepare_stability_ts(checkfile=checkfile, is_ts=False)
        job = self._stability_opt_job(checkfile=checkfile, restricted_used=False)
        with patch.object(self.sched1, 'run_job') as run_job:
            self.assertFalse(self.sched1.run_stability_job(label=label, opt_job=job))
        self.assertFalse(run_job.called)

    def test_the_stability_gate_refuses_a_reference_agnostic_species_that_is_not_a_ts(self):
        """Test that a method ARC writes no r/u prefix for is not read as a restricted reference"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        label = self._prepare_stability_ts(checkfile=checkfile, is_ts=False)
        for method, basis in [('cbs-qb3', None), ('am1', None), ('mmff94s', None)]:
            job = self._stability_opt_job(checkfile=checkfile, method=method, basis=basis, restricted_used=True)
            self.assertIn(job.level.method_type, ['force_field', 'composite', 'semiempirical'],
                          msg=f'{method} is not a reference-agnostic method type')
            with patch.object(self.sched1, 'run_job') as run_job:
                self.assertFalse(self.sched1.run_stability_job(label=label, opt_job=job),
                                 msg=f'{method} was admitted to the stability diagnostic')
            self.assertFalse(run_job.called)

    def test_the_stability_gate_refuses_a_species_carrying_no_reference_memo(self):
        """Test that a well whose opt job never wrote an ESS input is not tested"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        label = self._prepare_stability_ts(checkfile=checkfile, is_ts=False)
        job = self._stability_opt_job(checkfile=checkfile, restricted_used=None)
        with patch.object(self.sched1, 'run_job') as run_job:
            self.assertFalse(self.sched1.run_stability_job(label=label, opt_job=job))
        self.assertFalse(run_job.called)

    def test_the_stability_gate_still_admits_a_ts_whatever_reference_it_ran(self):
        """Test that a TS does not depend on the reference its opt job declared"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        label = self._prepare_stability_ts(checkfile=checkfile, is_ts=True)
        for restricted_used in [True, False, None]:
            job = self._stability_opt_job(checkfile=checkfile, restricted_used=restricted_used)
            self.sched1.species_dict[label].stability_analysis_ran = False
            with patch.object(self.sched1, 'run_job'):
                self.assertTrue(self.sched1.run_stability_job(label=label, opt_job=job),
                                msg=f'a TS whose opt job declared {restricted_used} was refused')

    def test_post_opt_jobs_hold_the_freq_the_sp_and_the_irc_until_the_verdict_is_in(self):
        """Test that a spawned stability analysis is the only job the post-opt path enqueues"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        label = self._prepare_stability_ts(checkfile=checkfile, is_ts=True)
        self.sched1._pending_pipe_freq.discard(label)
        self.sched1._pending_pipe_sp.discard(label)
        self.sched1._pending_pipe_irc.discard((label, 'forward'))
        self.sched1._pending_pipe_irc.discard((label, 'reverse'))
        job = self._stability_opt_job(checkfile=checkfile)
        run_job = self._spawn_post_opt(label=label, job=job)
        self.assertEqual(run_job.call_args.kwargs['job_type'], 'stability')
        self.assertNotIn(label, self.sched1._pending_pipe_freq)
        self.assertNotIn(label, self.sched1._pending_pipe_sp)
        self.assertNotIn((label, 'forward'), self.sched1._pending_pipe_irc)
        self.assertEqual(self.sched1.species_dict[label].stability_pending_opt_job, 'opt_a1')

    def test_the_re_optimization_releases_the_irc_onto_the_adopted_reference(self):
        """Test that the IRC an analysis held is enqueued once the re-optimized species comes back"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        label = self._prepare_stability_ts(checkfile=checkfile, is_ts=True)
        for pending in [self.sched1._pending_pipe_freq, self.sched1._pending_pipe_sp]:
            pending.discard(label)
            self.addCleanup(pending.discard, label)
        for direction in ['forward', 'reverse']:
            self.sched1._pending_pipe_irc.discard((label, direction))
            self.addCleanup(self.sched1._pending_pipe_irc.discard, (label, direction))
        species = self.sched1.species_dict[label]
        job = self._stability_opt_job(checkfile=checkfile)
        run_job = self._spawn_post_opt(label=label, job=job)
        self.assertEqual(run_job.call_args.kwargs['job_type'], 'stability')
        self.assertNotIn((label, 'forward'), self.sched1._pending_pipe_irc)

        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        with patch.object(self.sched1, 'run_job') as run_job:
            self.sched1.spawn_post_stability_jobs(label=label)
        self.assertEqual(run_job.call_args.kwargs['job_type'], 'opt')
        self.assertTrue(species.stability_reoptimized)
        self.assertNotIn((label, 'forward'), self.sched1._pending_pipe_irc)

        run_job = self._spawn_post_opt(label=label, job=job, job_name='opt_a2')
        self.assertFalse(run_job.called)
        self.assertIn((label, 'forward'), self.sched1._pending_pipe_irc)
        self.assertIn((label, 'reverse'), self.sched1._pending_pipe_irc)
        self.assertIn(label, self.sched1._pending_pipe_freq)
        self.assertIn(label, self.sched1._pending_pipe_sp)
        self.assertIsNone(species.stability_pending_opt_job)

    def test_an_irc_rejection_reduces_the_verdict_and_releases_the_held_analysis_state(self):
        """Test that a TS the IRC check rejects switches guess with its adopted verdict carried"""
        label = self._prepare_stability_ts(checkfile=None, is_ts=True)
        species = self.sched1.species_dict[label]
        original_convergence = self.sched1.output[label]['convergence']
        self.addCleanup(self.sched1.output[label].__setitem__, 'convergence', original_convergence)
        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        species.stability_pending_opt_job = 'opt_a1'
        species.populate_ts_checks()
        species.ts_checks['IRC'] = False
        with patch.object(self.sched1, 'determine_most_likely_ts_conformer'), \
                patch.object(self.sched1, 'delete_all_species_jobs'), \
                patch.object(self.sched1, 'run_opt_job'), \
                patch.object(self.sched1, 'run_composite_job'):
            self.sched1.process_irc_verdict(ts_label=label, rxn=None)
        self.assertIsNone(species.stability_pending_opt_job)
        self.assertEqual(species.derived_stability_verdict['verdict'], 'external_instability')
        self.assertTrue(adopted_reference_is_unrestricted(species))

    def _irc_scheduler(self, irc_file_name='rxn_1_irc_1.out', multiplicity=2):
        """
        Build a testing Scheduler holding a TS and one regular species, plus a forward IRC job of that TS.

        Args:
            irc_file_name (str, optional): The IRC output in arc/testing/irc whose geometry the TS takes.
            multiplicity (int, optional): The TS multiplicity, matching that IRC output.

        Returns:
            tuple: The scheduler, the TS label, and the IRC job.
        """
        ts_xyz = parser.parse_geometry(log_file_path=os.path.join(ARC_TESTING_PATH, 'irc', irc_file_name))
        ts = ARCSpecies(label='TS0', is_ts=True, xyz=ts_xyz, multiplicity=multiplicity, charge=0)
        sched = Scheduler(project='project_test_irc', ess_settings=self.ess_settings,
                          species_list=[ts, self.spc1], composite_method=None,
                          conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          scan_level=Level(repr=default_levels_of_theory['scan']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=self.project_directory, testing=True,
                          job_types=self.job_types1, orbitals_level=default_levels_of_theory['orbitals'],
                          adaptive_levels=None)
        job = job_factory(job_adapter='gaussian', project='project_test_irc', ess_settings=self.ess_settings,
                          species=[ts], xyz=ts_xyz, job_type='irc', irc_direction='forward',
                          level=Level(repr={'method': 'wb97xd', 'basis': 'def2svp'}),
                          project_directory=self.project_directory, job_num=1587)
        return sched, ts.label, job

    @staticmethod
    def _write_irc_output(job, content=''):
        """
        Write the output file an IRC job left behind.

        Args:
            job (JobAdapter): The IRC job.
            content (str, optional): The file content; empty by default, as a node fault leaves it.
        """
        os.makedirs(os.path.dirname(job.local_path_to_output_file), exist_ok=True)
        with open(job.local_path_to_output_file, 'w') as f:
            f.write(content)

    @staticmethod
    def _set_ess_status_from_output(job):
        """
        Set an IRC job's status to what ARC determines from the Gaussian output it left behind.

        Args:
            job (JobAdapter): The IRC job, whose output file already exists.
        """
        status, keywords, error, line = determine_ess_status(output_path=job.local_path_to_output_file,
                                                             species_label=job.species_label,
                                                             job_type='irc',
                                                             software='gaussian')
        job.job_status = ['done', {'status': status, 'keywords': keywords, 'error': error, 'line': line}]

    @staticmethod
    def _irc_state(sched, label):
        """
        Snapshot the state a failed IRC must not change.

        Args:
            sched (Scheduler): The scheduler.
            label (str): The TS label.

        Returns:
            dict: Deep copies of the relevant state.
        """
        return copy.deepcopy({'irc_paths': sched.output[label]['paths']['irc'],
                              'irc_directions': sched.output[label]['paths'].get('irc_directions', list()),
                              'irc_label': sched.species_dict[label].irc_label,
                              'unique_species_labels': sched.unique_species_labels,
                              'species_labels': sorted(sched.species_dict.keys()),
                              'other_output': sched.output['methylamine']})

    def _assert_failed_irc_left_no_trace(self, sched, label, job, before, cause='NoOutput'):
        """
        Assert that a failed IRC was recorded as an error and changed nothing else.

        Args:
            sched (Scheduler): The scheduler.
            label (str): The TS label.
            job (JobAdapter): The failed IRC job.
            before (dict): Deep copies of the state taken before the IRC was handled.
            cause (str, optional): Text the recorded error must contain, e.g., an ESS status keyword.
        """
        self.assertEqual(sched.output[label]['paths']['irc'], before['irc_paths'])
        self.assertEqual(sched.output[label]['paths'].get('irc_directions', list()), before['irc_directions'])
        self.assertFalse(sched.output[label]['job_types'].get('irc'))
        self.assertEqual(sched.species_dict[label].irc_label, before['irc_label'])
        self.assertEqual(sched.unique_species_labels, before['unique_species_labels'])
        self.assertEqual(sorted(sched.species_dict.keys()), before['species_labels'])
        self.assertIsNone(sched.species_dict[label].ts_checks['IRC'])
        self.assertIn(job.job_name, sched.output[label]['errors'])
        self.assertIn(cause, sched.output[label]['errors'])
        self.assertEqual(sched.output['methylamine'], before['other_output'])

    def test_an_irc_with_an_empty_output_is_recorded_as_failed_and_spawns_nothing(self):
        """Test that an IRC whose output is empty, too short, or cut mid-geometry is a failed IRC"""
        with open(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out'), 'r') as f:
            lines = f.readlines()
        orientation_line = next(i for i, line in enumerate(lines) if 'orientation:' in line)
        contents = {'empty': ('', 'NoOutput'),
                    'too short': (' Entering Gaussian System, Link 0=g16\n Initial command:\n', 'NoOutput'),
                    'cut mid-geometry': (''.join(lines[:orientation_line + 8]), 'Unknown')}
        for case, (content, keyword) in contents.items():
            with self.subTest(case=case):
                sched, label, job = self._irc_scheduler()
                self._write_irc_output(job, content=content)
                self._set_ess_status_from_output(job)
                self.assertEqual(job.job_status[1]['keywords'], [keyword])
                before = self._irc_state(sched, label)
                with patch.object(sched, 'run_job') as run_job:
                    sched.spawn_post_irc_jobs(label=label, job=job)
                self.assertFalse(run_job.called)
                self._assert_failed_irc_left_no_trace(sched, label, job, before, cause=keyword)

    def test_an_irc_cut_after_its_first_points_is_recorded_as_failed(self):
        """Test that an IRC log cut by a node fault after some points is a failed IRC, not a near-TS endpoint"""
        with open(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out'), 'r') as f:
            lines = f.readlines()
        point_lines = [i for i, line in enumerate(lines) if 'Point Number:' in line and 'Path Number' in line]
        cuts = {'after point 1': point_lines[1] + 1,
                'inside the last geometry table':
                    max(i for i, line in enumerate(lines) if 'orientation:' in line) + 7}
        for case, cut in cuts.items():
            with self.subTest(case=case):
                sched, label, job = self._irc_scheduler()
                self._write_irc_output(job, content=''.join(lines[:cut]))
                self.assertIsNotNone(parser.parse_geometry(log_file_path=job.local_path_to_output_file))
                self._set_ess_status_from_output(job)
                self.assertEqual(job.job_status[1]['keywords'], ['Unknown'])
                before = self._irc_state(sched, label)
                with patch.object(sched, 'run_job') as run_job:
                    sched.spawn_post_irc_jobs(label=label, job=job)
                self.assertFalse(run_job.called)
                self._assert_failed_irc_left_no_trace(sched, label, job, before, cause='Unknown')

    def test_an_irc_killed_by_the_server_or_for_memory_is_recorded_as_failed(self):
        """Test that an IRC cut by a server time limit, a memory kill, or a GS2 failure is a failed IRC"""
        with open(os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out'), 'r') as f:
            lines = f.readlines()
        point_lines = [i for i, line in enumerate(lines) if 'Point Number:' in line and 'Path Number' in line]
        statuses = {'ServerTimeLimit': 'Job killed by the server since it reached the maximal time limit.',
                    'Memory': 'Insufficient job memory.',
                    'GS2Opt': 'GS2 optimization failed.'}
        for keyword, error in statuses.items():
            with self.subTest(keyword=keyword):
                sched, label, job = self._irc_scheduler()
                self._write_irc_output(job, content=''.join(lines[:point_lines[1] + 1]))
                self.assertIsNotNone(parser.parse_geometry(log_file_path=job.local_path_to_output_file))
                keywords = [keyword, 'GL123'] if keyword == 'GS2Opt' else [keyword]
                job.job_status = ['done', {'status': 'errored', 'keywords': keywords, 'error': error, 'line': ''}]
                before = self._irc_state(sched, label)
                with patch.object(sched, 'run_job') as run_job:
                    sched.spawn_post_irc_jobs(label=label, job=job)
                self.assertFalse(run_job.called)
                self._assert_failed_irc_left_no_trace(sched, label, job, before, cause=keyword)
                self.assertIn(error, sched.output[label]['errors'])
                self.assertIn(error, sched.species_dict[label].ts_checks['warnings'])

    def test_an_irc_with_a_server_error_is_recorded_as_failed(self):
        """Test that an IRC job the server reports as errored is a failed IRC, though end_job() does not end it"""
        sched, label, job = self._irc_scheduler()
        self._write_irc_output(job)

        def server_error(**kwargs):
            """Set the server status of the IRC job to errored, as end_job() leaves it, and return False."""
            job.job_status = ['errored', {'status': 'initializing', 'keywords': list(), 'error': '', 'line': ''}]
            return False

        before = self._irc_state(sched, label)
        with patch.object(sched, 'end_job', side_effect=server_error), patch.object(sched, 'run_job') as run_job:
            sched.end_irc_job(label=label, job=job)
        self.assertFalse(run_job.called)
        self._assert_failed_irc_left_no_trace(sched, label, job, before, cause='server status: errored')

    def test_an_irc_rerun_by_end_job_is_not_recorded_as_failed(self):
        """Test that an IRC job end_job() reran is neither spawned from nor recorded as a failed IRC"""
        sched, label, job = self._irc_scheduler()
        job.job_status = ['done', {'status': 'errored', 'keywords': ['ServerTimeLimit'], 'error': '', 'line': ''}]
        with patch.object(sched, 'end_job', return_value=False), \
                patch.object(sched, 'spawn_post_irc_jobs') as spawn_post_irc_jobs:
            sched.end_irc_job(label=label, job=job)
        self.assertFalse(spawn_post_irc_jobs.called)
        with patch.object(sched, 'end_job', return_value=True), \
                patch.object(sched, 'spawn_post_irc_jobs') as spawn_post_irc_jobs:
            sched.end_irc_job(label=label, job=job)
        spawn_post_irc_jobs.assert_called_once_with(label=label, job=job)

    def _end_errored_irc_without_output(self, sched, label, job):
        """
        End an IRC job the server reports as errored and whose output file is missing, through the real end_job().

        Args:
            sched (Scheduler): The scheduler.
            label (str): The TS label.
            job (JobAdapter): The IRC job, whose output file does not exist.

        Returns:
            MagicMock: The patched ``run_job``, which marks the job it submits as running, as ``run_job`` does.
        """
        sched.running_jobs[label] = [job.job_name]

        def server_error():
            """Set the server status of the IRC job to errored, as determine_job_status() does."""
            job.job_status = ['errored', {'status': 'initializing', 'keywords': list(), 'error': '', 'line': ''}]
            return True

        def submit(**kwargs):
            """Mark a new IRC job of the TS as running."""
            sched.running_jobs[label].append('irc_a1588')

        with patch.object(job, 'determine_job_status', side_effect=server_error), \
                patch.object(job, 'rename_output_file'), patch('arc.scheduler.time.sleep'), \
                patch.object(sched, 'run_job', side_effect=submit) as run_job:
            sched.end_irc_job(label=label, job=job)
        return run_job

    def test_an_errored_irc_rerun_for_a_missing_output_is_not_recorded_as_failed(self):
        """Test that an errored IRC end_job() reruns for a missing output on the first look leaves no failed IRC"""
        sched, label, job = self._irc_scheduler()
        self.assertFalse(os.path.isfile(job.local_path_to_output_file))
        run_job = self._end_errored_irc_without_output(sched, label, job)
        self.assertTrue(run_job.called)
        self.assertIn('restart_due_to_file_not_found', job.ess_trsh_methods)
        self.assertEqual(sched.running_jobs[label], ['irc_a1588'])
        self.assertNotIn(job.job_name, sched.output[label]['errors'])
        self.assertNotIn(IRC_JOB_FAILED_WARNING, sched.species_dict[label].ts_checks.get('warnings') or '')

    def test_an_errored_irc_missing_its_output_twice_is_recorded_as_failed(self):
        """Test that an errored IRC whose output is missing on the second look is a failed IRC"""
        sched, label, job = self._irc_scheduler()
        job.ess_trsh_methods.append('restart_due_to_file_not_found')
        before = self._irc_state(sched, label)
        run_job = self._end_errored_irc_without_output(sched, label, job)
        self.assertFalse(run_job.called)
        self._assert_failed_irc_left_no_trace(sched, label, job, before, cause='server status: errored')
        self.assertIn(IRC_JOB_FAILED_WARNING, sched.species_dict[label].ts_checks['warnings'])

    def test_irc_ended_usably(self):
        """Test which server and ESS statuses of an IRC job allow its last geometry to be used"""
        cases = [('done', {'status': 'done', 'keywords': list()}, True),
                 ('done', {'status': 'errored', 'keywords': ['DeltaX', 'GL123']}, True),
                 ('done', {'status': 'errored', 'keywords': ['DeltaX']}, True),
                 ('done', {'status': 'errored', 'keywords': ['DeltaX', 'Memory']}, False),
                 ('done', {'status': 'errored', 'keywords': list()}, False),
                 ('done', {'status': 'errored', 'keywords': None}, False),
                 ('done', {'status': 'errored', 'keywords': ['ServerTimeLimit']}, False),
                 ('done', {'status': 'unconverged', 'keywords': list()}, False),
                 ('done', dict(), False),
                 ('errored', {'status': 'done', 'keywords': list()}, False),
                 (None, {'status': 'done', 'keywords': list()}, False)]
        for server_status, ess_status, expected in cases:
            with self.subTest(server_status=server_status, ess_status=ess_status):
                self.assertEqual(irc_ended_usably(server_status=server_status, ess_status=ess_status), expected)

    def test_a_completed_irc_output_without_a_geometry_is_recorded_as_failed(self):
        """Test that an IRC whose status is not a failure but whose output holds no geometry is a failed IRC"""
        sched, label, job = self._irc_scheduler()
        self._write_irc_output(job, content=' Entering Gaussian System, Link 0=g16\n Initial command:\n')
        job.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]
        before = self._irc_state(sched, label)
        with patch.object(sched, 'run_job') as run_job:
            sched.spawn_post_irc_jobs(label=label, job=job)
        self.assertFalse(run_job.called)
        self._assert_failed_irc_left_no_trace(sched, label, job, before, cause='no geometry could be parsed')

    def test_an_irc_output_the_parser_cannot_read_is_recorded_as_failed(self):
        """Test that an IRC whose geometry parsing raises IndexError or ValueError is a failed IRC"""
        for error in [IndexError, ValueError]:
            with self.subTest(error=error.__name__):
                sched, label, job = self._irc_scheduler()
                self._write_irc_output(job)
                job.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]
                before = self._irc_state(sched, label)
                with patch.object(sched, 'run_job') as run_job, \
                        patch('arc.scheduler.parser.parse_geometry', side_effect=error('truncated table')):
                    sched.spawn_post_irc_jobs(label=label, job=job)
                self.assertFalse(run_job.called)
                self._assert_failed_irc_left_no_trace(sched, label, job, before, cause=error.__name__)
                self.assertIn(f'parsing raised {error.__name__}: truncated table', sched.output[label]['errors'])

    def test_an_unexpected_parser_error_in_an_irc_output_propagates(self):
        """Test that a geometry parser error other than IndexError or ValueError is raised, not recorded"""
        sched, label, job = self._irc_scheduler()
        self._write_irc_output(job)
        job.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]
        with patch.object(sched, 'run_job'), \
                patch('arc.scheduler.parser.parse_geometry', side_effect=RuntimeError('parser bug')):
            with self.assertRaises(RuntimeError):
                sched.spawn_post_irc_jobs(label=label, job=job)
        self.assertEqual(sched.output[label]['errors'], '')
        self.assertEqual(sched.species_dict[label].ts_checks['warnings'], '')

    def test_a_failed_irc_cause_is_written_without_list_syntax_or_a_missing_direction(self):
        """Test that the failed-IRC message lists keywords plainly and omits an unknown direction"""
        sched, label, job = self._irc_scheduler()
        self._write_irc_output(job)
        job.irc_direction = None
        job.job_status = ['done', {'status': 'errored', 'keywords': ['NoOutput', 'Unknown'],
                                   'error': 'Log file could not be read', 'line': ''}]
        with patch.object(sched, 'run_job'):
            sched.spawn_post_irc_jobs(label=label, job=job)
        messages = {'errors': sched.output[label]['errors'],
                    'warnings': sched.species_dict[label].ts_checks['warnings']}
        for entry, message in messages.items():
            with self.subTest(entry=entry):
                self.assertIn(f'IRC job {job.job_name} failed: Log file could not be read '
                              f'(keywords: NoOutput, Unknown)', message)
                self.assertNotIn('None', message)
                self.assertNotIn('[', message)
                self.assertNotIn("'", message)
        job.irc_direction = 'reverse'
        job.job_status[1]['keywords'] = ['Unknown']
        with patch.object(sched, 'run_job'):
            sched.spawn_post_irc_jobs(label=label, job=job)
        self.assertIn(f'IRC job {job.job_name} (reverse) failed', sched.output[label]['errors'])

    def test_a_failed_irc_writes_its_cause_to_the_ts_entry_of_output_yml(self):
        """Test that the cause of a failed IRC reaches the TS's ts_checks warnings in output.yml"""
        sched, label, job = self._irc_scheduler()
        self._write_irc_output(job)
        job.job_status = ['done', {'status': 'errored', 'keywords': ['NoOutput'],
                                   'error': 'Log file could not be read', 'line': ''}]
        with patch.object(sched, 'run_job'):
            sched.spawn_post_irc_jobs(label=label, job=job)
        self.assertIn(job.job_name, sched.species_dict[label].ts_checks['warnings'])
        write_output_yml(project='project_test_irc', project_directory=self.project_directory,
                         species_dict=sched.species_dict, reactions=list(), output_dict=sched.output,
                         compute_thermo=False)
        document = read_yaml_file(os.path.join(self.project_directory, 'output', 'output.yml'))
        ts_entry = next(ts for ts in document['transition_states'] if ts['label'] == label)
        self.assertIsNone(ts_entry['ts_checks']['IRC'])
        self.assertFalse(ts_entry['irc_converged'])
        self.assertEqual(ts_entry['irc_logs'], list())
        self.assertIn(job.job_name, ts_entry['ts_checks']['warnings'])
        self.assertIn('Log file could not be read', ts_entry['ts_checks']['warnings'])

    def test_a_rate_from_a_ts_whose_irc_failed_carries_a_caution_in_output_yml(self):
        """Test that the reaction of a TS with a failed IRC job carries a machine-readable caution in output.yml"""
        sched, label, job = self._irc_scheduler()
        self._write_irc_output(job)
        self._set_ess_status_from_output(job)
        with patch.object(sched, 'run_job'):
            sched.spawn_post_irc_jobs(label=label, job=job)
        ts_species = sched.species_dict[label]
        self.assertIsNone(get_ts_validation_comment(ts_species))
        rxn = ARCReaction(label='CH4 + OH <=> CH3 + H2O', ts_label=label,
                          kinetics={'A': (1.2e10, 'cm^3/(mol*s)'), 'n': 0.5, 'Ea': (45.6, 'kJ/mol')})
        rxn.ts_species = ts_species
        write_output_yml(project='project_test_irc', project_directory=self.project_directory,
                         species_dict=sched.species_dict, reactions=[rxn], output_dict=sched.output,
                         compute_thermo=False)
        document = read_yaml_file(os.path.join(self.project_directory, 'output', 'output.yml'))
        self.assertTrue(document['reactions'][0]['ts_validation_caution'].startswith(
            f'{TS_IRC_NOT_CONFIRMED_MARKER} ({IRC_CAUSE_JOB_FAILED})'))

    def _spawn_both_irc_endpoints(self):
        """
        Build a testing Scheduler whose TS ran a forward and a reverse IRC, each spawning an endpoint species.

        Returns:
            tuple: The scheduler, the TS label, and the labels of the two endpoint species.
        """
        sched, label, job = self._irc_scheduler()
        for direction, file_name in [('forward', 'rxn_1_irc_1.out'), ('reverse', 'rxn_1_irc_2.out')]:
            job.irc_direction = direction
            with open(os.path.join(ARC_TESTING_PATH, 'irc', file_name), 'r') as f:
                self._write_irc_output(job, content=f.read())
            self._set_ess_status_from_output(job)
            with patch.object(sched, 'run_job'), patch('arc.scheduler.plotter.save_irc_traj_animation'):
                sched.spawn_post_irc_jobs(label=label, job=job)
        endpoint_labels = sched.species_dict[label].irc_label.split()
        self.assertEqual(len(endpoint_labels), 2)
        return sched, label, endpoint_labels

    def _caution_in_output_yml(self, sched, label):
        """
        Write output.yml for one reaction of a TS, with IRC requested, and read back the reaction's caution.

        Args:
            sched (Scheduler): The scheduler.
            label (str): The TS label.

        Returns:
            str | None: The ``ts_validation_caution`` of the reaction, or ``None`` if it is absent.
        """
        rxn = ARCReaction(label='CH4 + OH <=> CH3 + H2O', ts_label=label,
                          kinetics={'A': (1.2e10, 'cm^3/(mol*s)'), 'n': 0.5, 'Ea': (45.6, 'kJ/mol')})
        rxn.ts_species = sched.species_dict[label]
        write_output_yml(project='project_test_irc', project_directory=self.project_directory,
                         species_dict={label: sched.species_dict[label]}, reactions=[rxn], output_dict=sched.output,
                         compute_thermo=False, irc_requested=True)
        document = read_yaml_file(os.path.join(self.project_directory, 'output', 'output.yml'))
        return document['reactions'][0].get('ts_validation_caution')

    def test_a_rate_whose_irc_endpoint_opt_did_not_converge_carries_a_caution(self):
        """Test that a rate whose IRC endpoint opt never converged, so no comparison ran, carries a caution"""
        sched, label, endpoint_labels = self._spawn_both_irc_endpoints()
        sched.output[endpoint_labels[0]]['paths']['geo'] = os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_1.out')
        self.assertIsNone(sched.species_dict[label].ts_checks['IRC'])
        caution = self._caution_in_output_yml(sched, label)
        self.assertTrue(caution.startswith(f'{TS_IRC_NOT_CONFIRMED_MARKER} ({IRC_CAUSE_ENDPOINT_OPT_NOT_CONVERGED})'))

    def test_a_rate_whose_irc_comparison_is_inconclusive_carries_a_caution(self):
        """Test that a rate whose optimized IRC endpoints could not be compared with the wells carries a caution"""
        sched, label, endpoint_labels = self._spawn_both_irc_endpoints()
        for endpoint_label, file_name in zip(endpoint_labels, ['rxn_1_irc_1.out', 'rxn_1_irc_2.out']):
            sched.output[endpoint_label]['paths']['geo'] = os.path.join(ARC_TESTING_PATH, 'irc', file_name)
        with patch('arc.scheduler.check_irc_species_and_rxn', return_value=None) as check_irc_species_and_rxn:
            sched.check_irc_species(label=endpoint_labels[1])
        check_irc_species_and_rxn.assert_called_once()
        self.assertIsNone(sched.species_dict[label].ts_checks['IRC'])
        caution = self._caution_in_output_yml(sched, label)
        self.assertTrue(caution.startswith(f'{TS_IRC_NOT_CONFIRMED_MARKER} ({IRC_CAUSE_COMPARISON_INCONCLUSIVE})'))

    def test_an_irc_with_a_parsable_output_spawns_its_species_and_opt(self):
        """
        Test that a converged IRC spawns its species and opt, and that a Delta-x IRC still does.

        ``irc_failed.out`` stopped with 'Delta-x Convergence NOT Met' (keywords ``DeltaX``, ``GL123``) and its last
        geometry is about 0.007 Angstrom from the TS. Spawning it as an endpoint is pre-existing behaviour,
        kept unchanged here: an ESS status of ``errored`` with only ``DeltaX`` and ``GL123`` still spawns.
        """
        cases = [('rxn_1_irc_1.out', 2, list()), ('irc_failed.out', 1, ['DeltaX', 'GL123'])]
        for file_name, multiplicity, keywords in cases:
            with self.subTest(file_name=file_name):
                sched, label, job = self._irc_scheduler(irc_file_name=file_name, multiplicity=multiplicity)
                with open(os.path.join(ARC_TESTING_PATH, 'irc', file_name), 'r') as f:
                    self._write_irc_output(job, content=f.read())
                self._set_ess_status_from_output(job)
                self.assertEqual(job.job_status[1]['keywords'], keywords)
                expected_xyz = parser.parse_geometry(log_file_path=job.local_path_to_output_file)
                with patch.object(sched, 'run_job') as run_job:
                    sched.spawn_post_irc_jobs(label=label, job=job)
                self.assertEqual(sched.output[label]['paths']['irc'], [job.local_path_to_output_file])
                self.assertEqual(sched.output[label]['paths']['irc_directions'], ['forward'])
                self.assertEqual(sched.species_dict[label].irc_label, 'IRC_TS0_1')
                irc_spc = sched.species_dict['IRC_TS0_1']
                self.assertEqual(irc_spc.irc_label, label)
                self.assertTrue(almost_equal_coords_lists(irc_spc.get_xyz(), expected_xyz))
                run_job.assert_called_once()
                self.assertEqual(run_job.call_args.kwargs['label'], 'IRC_TS0_1')
                self.assertEqual(run_job.call_args.kwargs['job_type'], 'opt')
                self.assertEqual(sched.output[label]['errors'], '')

    def test_a_restored_irc_job_with_an_empty_output_does_not_crash(self):
        """Test that an IRC job restored from a restart file and found with an empty output does not crash"""
        sched, label, job = self._irc_scheduler()
        sched.output[label]['paths']['irc'] = [os.path.join(ARC_TESTING_PATH, 'irc', 'rxn_1_irc_2.out')]
        sched.output[label]['paths']['irc_directions'] = ['reverse']
        sched.restart_dict = {'running_jobs': {label: [job.as_dict()]}}
        sched.running_jobs, sched.job_dict = dict(), dict()
        sched.restore_running_jobs()
        restored_job = sched.job_dict[label]['irc'][job.job_name]
        self.assertEqual(restored_job.irc_direction, 'forward')
        self._write_irc_output(restored_job)
        restored_job.job_status = ['done', {'status': 'errored', 'keywords': ['NoOutput'],
                                            'error': 'Log file could not be read', 'line': ''}]
        before = self._irc_state(sched, label)
        with patch.object(sched, 'run_job') as run_job, \
                patch('arc.scheduler.plotter.save_irc_traj_animation') as save_irc_traj_animation:
            sched.spawn_post_irc_jobs(label=label, job=restored_job)
        self.assertFalse(run_job.called)
        self.assertFalse(save_irc_traj_animation.called)
        self._assert_failed_irc_left_no_trace(sched, label, restored_job, before)

    def test_post_opt_jobs_proceed_where_no_stability_analysis_is_spawned(self):
        """Test that a species the analysis does not admit enqueues its freq and sp as usual"""
        label = self._prepare_stability_ts(checkfile=None, is_ts=False, enabled=False)
        self.sched1._pending_pipe_freq.discard(label)
        self.sched1._pending_pipe_sp.discard(label)
        self.addCleanup(self.sched1._pending_pipe_freq.discard, label)
        self.addCleanup(self.sched1._pending_pipe_sp.discard, label)
        job = self._stability_opt_job(checkfile=None, restricted_used=True)
        run_job = self._spawn_post_opt(label=label, job=job)
        self.assertFalse(run_job.called)
        self.assertIn(label, self.sched1._pending_pipe_freq)
        self.assertIn(label, self.sched1._pending_pipe_sp)
        self.assertIsNone(self.sched1.species_dict[label].stability_pending_opt_job)

    def test_post_opt_jobs_proceed_where_the_opt_job_carries_no_ess_state(self):
        """Test that an opt job the analysis cannot read leaves the freq and the sp enqueued"""
        label = self._prepare_stability_ts(checkfile=None, is_ts=True, enabled=True)
        self.sched1._pending_pipe_freq.discard(label)
        self.sched1._pending_pipe_sp.discard(label)
        self.addCleanup(self.sched1._pending_pipe_freq.discard, label)
        self.addCleanup(self.sched1._pending_pipe_sp.discard, label)
        self.addCleanup(self.sched1._pending_pipe_irc.discard, (label, 'forward'))
        self.addCleanup(self.sched1._pending_pipe_irc.discard, (label, 'reverse'))
        piped = SimpleNamespace(local_path_to_output_file='/nonexistent/opt.out',
                                level=Level(method='wb97xd', basis='def2-TZVP'),
                                job_status=['done', {'status': 'done'}])
        run_job = self._spawn_post_opt(label=label, job=piped)
        self.assertFalse(run_job.called)
        self.assertIn(label, self.sched1._pending_pipe_freq)
        self.assertIn(label, self.sched1._pending_pipe_sp)
        self.assertIsNone(self.sched1.species_dict[label].stability_pending_opt_job)

    def test_the_re_optimization_reads_the_orbitals_the_analysis_relaxed_into(self):
        """Test that orbitals are carried over only from an analysis that followed the instability"""
        with tempfile.NamedTemporaryFile(suffix='.gbw', delete=False) as f:
            relaxed = f.name
        self.addCleanup(lambda: os.path.isfile(relaxed) and os.remove(relaxed))
        label = self._prepare_stability_ts(checkfile=relaxed, is_ts=True)
        species = self.sched1.species_dict[label]
        stability_job = MagicMock()
        stability_job.local_path_to_check_file = relaxed
        self.sched1.job_dict[label]['stability'] = {'stability_a2': stability_job}

        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True,
                                             'followed_to_stable': True}
        self.sched1.adopt_stability_orbitals(label=label)
        self.assertEqual(species.checkfile, relaxed)

        species.checkfile = relaxed
        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True,
                                             'followed_to_stable': False}
        self.sched1.adopt_stability_orbitals(label=label)
        self.assertIsNone(species.checkfile)

    def test_a_stable_verdict_releases_the_held_jobs_unchanged(self):
        """Test that a verdict ARC does not act on lets the freq and the sp proceed"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        label = self._prepare_stability_ts(checkfile=checkfile, is_ts=True)
        self.addCleanup(self.sched1._pending_pipe_freq.discard, label)
        self.addCleanup(self.sched1._pending_pipe_sp.discard, label)
        self.addCleanup(self.sched1._pending_pipe_irc.discard, (label, 'forward'))
        self.addCleanup(self.sched1._pending_pipe_irc.discard, (label, 'reverse'))
        job = self._stability_opt_job(checkfile=checkfile)
        self._spawn_post_opt(label=label, job=job)
        self.sched1.species_dict[label].derived_stability_verdict = {'verdict': 'stable', 'restricted': True}
        with patch.object(self.sched1, 'run_scan_jobs'), \
                patch.object(self.sched1, 'spawn_ts_jobs'), \
                patch.object(self.sched1, 'run_job') as run_job:
            self.sched1.spawn_post_stability_jobs(label=label)
        self.assertFalse(run_job.called)
        self.assertIn(label, self.sched1._pending_pipe_freq)
        self.assertIn(label, self.sched1._pending_pipe_sp)
        self.assertIsNone(self.sched1.species_dict[label].stability_pending_opt_job)
        self.assertFalse(self.sched1.species_dict[label].stability_reoptimized)

    def test_an_adoptable_verdict_re_optimizes_the_species_exactly_once(self):
        """Test that an adopted external instability re-runs the opt and that the guard holds after it"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        label = self._prepare_stability_ts(checkfile=checkfile, is_ts=True)
        self.addCleanup(self.sched1._pending_pipe_freq.discard, label)
        self.addCleanup(self.sched1._pending_pipe_sp.discard, label)
        self.sched1._pending_pipe_freq.discard(label)
        self.sched1._pending_pipe_sp.discard(label)
        job = self._stability_opt_job(checkfile=checkfile, fine=True)
        self._spawn_post_opt(label=label, job=job)
        species = self.sched1.species_dict[label]
        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertTrue(adopted_reference_is_unrestricted(species))
        with patch.object(self.sched1, 'run_job') as run_job:
            self.sched1.spawn_post_stability_jobs(label=label)
        self.assertTrue(run_job.called)
        kwargs = run_job.call_args.kwargs
        self.assertEqual(kwargs['job_type'], 'opt')
        self.assertTrue(kwargs['fine'])
        self.assertIs(kwargs['xyz'], species.final_xyz)
        self.assertIs(species.initial_xyz, species.final_xyz)
        self.assertTrue(species.stability_reoptimized)
        self.assertNotIn(label, self.sched1._pending_pipe_freq)

        species.stability_pending_opt_job = 'opt_a1'
        with patch.object(self.sched1, 'run_scan_jobs'), \
                patch.object(self.sched1, 'spawn_ts_jobs'), \
                patch.object(self.sched1, 'run_job') as run_job:
            self.sched1.spawn_post_stability_jobs(label=label)
        self.assertFalse(run_job.called)
        self.assertIn(label, self.sched1._pending_pipe_freq)

    def test_the_re_optimization_guard_survives_a_restart(self):
        """Test that the one-re-optimization guard is written to and read back from the restart dictionary"""
        species = ARCSpecies(label='spc_under_test', smiles='CC')
        species.stability_analysis_ran = True
        species.stability_pending_opt_job = 'opt_a3'
        species.stability_reoptimized = True
        restored = ARCSpecies(species_dict=species.as_dict())
        self.assertTrue(restored.stability_analysis_ran)
        self.assertEqual(restored.stability_pending_opt_job, 'opt_a3')
        self.assertTrue(restored.stability_reoptimized)

        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        label = self._prepare_stability_ts(checkfile=checkfile, is_ts=True)
        restored_species = self.sched1.species_dict[label]
        restored_species.stability_analysis_ran = True
        restored_species.stability_pending_opt_job = 'opt_a1'
        restored_species.stability_reoptimized = True
        restored_species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.addCleanup(self.sched1._pending_pipe_freq.discard, label)
        self.addCleanup(self.sched1._pending_pipe_sp.discard, label)
        self.sched1.job_dict[label]['opt'] = {'opt_a1': self._stability_opt_job(checkfile=checkfile)}
        with patch.object(self.sched1, 'run_scan_jobs'), \
                patch.object(self.sched1, 'spawn_ts_jobs'), \
                patch.object(self.sched1, 'run_job') as run_job:
            self.sched1.spawn_post_stability_jobs(label=label)
        self.assertFalse(run_job.called)

    def test_a_restart_releases_work_held_by_an_analysis_that_is_no_longer_running(self):
        """Test that a resumed run does not leave a species holding its freq and sp forever"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        label = self._prepare_stability_ts(checkfile=checkfile, is_ts=True)
        self.addCleanup(self.sched1._pending_pipe_freq.discard, label)
        self.addCleanup(self.sched1._pending_pipe_sp.discard, label)
        self.addCleanup(self.sched1._pending_pipe_irc.discard, (label, 'forward'))
        self.addCleanup(self.sched1._pending_pipe_irc.discard, (label, 'reverse'))
        self.sched1._pending_pipe_freq.discard(label)
        original_running = self.sched1.running_jobs.get(label)
        self.addCleanup(self.sched1.running_jobs.__setitem__, label, original_running or list())
        species = self.sched1.species_dict[label]
        species.stability_analysis_ran = True
        species.stability_pending_opt_job = 'opt_a1'
        self.sched1.job_dict[label]['opt'] = {'opt_a1': self._stability_opt_job(checkfile=checkfile)}

        self.sched1.running_jobs[label] = ['stability_a2']
        with patch.object(self.sched1, 'run_scan_jobs'), \
                patch.object(self.sched1, 'spawn_ts_jobs'):
            self.sched1.release_held_stability_work(label=label)
        self.assertEqual(species.stability_pending_opt_job, 'opt_a1')
        self.assertNotIn(label, self.sched1._pending_pipe_freq)

        self.sched1.running_jobs[label] = list()
        with patch.object(self.sched1, 'run_scan_jobs'), \
                patch.object(self.sched1, 'spawn_ts_jobs'):
            self.sched1.release_held_stability_work(label=label)
        self.assertIsNone(species.stability_pending_opt_job)
        self.assertIn(label, self.sched1._pending_pipe_freq)

    def test_a_ts_switch_releases_the_held_optimization_and_carries_the_verdict(self):
        """Test that abandoning a TS guess drops the pending opt job while keeping an adopted verdict"""
        label = self._prepare_stability_ts(checkfile=None, is_ts=True)
        species = self.sched1.species_dict[label]
        species.stability_pending_opt_job = 'opt_a1'
        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.sched1.carry_stability_verdict_across_ts_switch(label=label)
        self.assertIsNone(species.stability_pending_opt_job)
        self.assertEqual(species.derived_stability_verdict['verdict'], 'external_instability')
        self.assertTrue(adopted_reference_is_unrestricted(species),
                        msg='the next TS guess must start unrestricted from its first optimization')

    def test_the_stability_diagnostic_stays_inert_while_the_job_type_is_off(self):
        """Test that a species the gate admits still runs nothing unless the user asked for it"""
        with tempfile.NamedTemporaryFile(suffix='.chk', delete=False) as f:
            checkfile = f.name
        self.addCleanup(lambda: os.path.isfile(checkfile) and os.remove(checkfile))
        label = self._prepare_stability_ts(checkfile=checkfile, is_ts=True, enabled=False)
        job = self._stability_opt_job(checkfile=checkfile, restricted_used=True)
        with patch.object(self.sched1, 'run_job') as run_job:
            self.assertFalse(self.sched1.run_stability_job(label=label, opt_job=job))
        self.assertFalse(run_job.called)
        self.assertFalse(self.sched1.species_dict[label].stability_analysis_ran)
        self.assertIsNone(self.sched1.species_dict[label].stability_pending_opt_job)

    def test_post_freq_actions_spawns_no_stability_analysis(self):
        """Test that the frequency path holds no stability trigger of its own"""
        label = self._prepare_stability_ts(checkfile=None, is_ts=True)
        self.sched1.species_dict[label].ts_checks = {'NMD': True}
        job = MagicMock()
        job.job_adapter = 'gaussian'
        job.job_name = 'freq_a1'
        job.job_type = 'freq'
        job.restricted_used = True
        job.level = Level(method='wb97xd', basis='def2-TZVP')
        job.local_path_to_output_file = '/nonexistent/freq.out'
        with patch.object(self.sched1, 'check_negative_freq', return_value=(True, False)), \
                patch.object(self.sched1, 'check_rxn_e0_by_spc'), \
                patch.object(self.sched1, 'run_job') as run_job, \
                patch('arc.scheduler.safe_copy_file'), \
                patch('arc.scheduler.parser.parse_polarizability', return_value=None):
            freq_ok, switched = self.sched1.post_freq_actions(label=label, job=job, vibfreqs=[-1000.0])
        self.assertTrue(freq_ok)
        self.assertFalse(switched)
        self.assertFalse(run_job.called)
        self.assertFalse(self.sched1.species_dict[label].stability_analysis_ran)

    def test_check_stability_job_survives_an_unreadable_log(self):
        """Test that a corrupt stability log is reported and does not propagate"""
        label = 'C2H6'
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            f.write(b'\xff\xfe\x00binary garbage\n')
            log_path = f.name
        self.addCleanup(lambda: os.path.isfile(log_path) and os.remove(log_path))
        job = MagicMock()
        job.job_status = ['done', {'status': 'done'}]
        job.local_path_to_output_file = log_path
        self.sched1.output[label]['paths'].pop('stability', None)
        self.sched1.check_stability_job(label=label, job=job)
        self.assertNotIn('stability', self.sched1.output[label]['paths'])

    def _honour_the_reference_change(self):
        """Point every level of the scheduler at an ESS ARC breaks the spin symmetry for."""
        for attribute in ['opt_level', 'freq_level', 'sp_level']:
            self.addCleanup(setattr, self.sched1, attribute, getattr(self.sched1, attribute))
            setattr(self.sched1, attribute, Level(method='wb97xd', basis='def2tzvp', software='gaussian'))

    def _run_check_stability(self, fixture_name: str, label: str = 'C2H6', status: str = 'done'):
        """Run check_stability_job against a real stability fixture and capture its log records."""
        job = MagicMock()
        job.job_status = [status, {'status': status}]
        job.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'stability', fixture_name)
        self.sched1.output[label]['paths'].pop('stability', None)
        self.sched1.output[label].pop('wavefunction_stability', None)
        self.addCleanup(self.sched1.output[label].__setitem__, 'warnings',
                        self.sched1.output[label]['warnings'])
        self.addCleanup(setattr, self.sched1.species_dict[label], 'derived_stability_verdict',
                        self.sched1.species_dict[label].derived_stability_verdict)
        with self.assertLogs('arc', level='DEBUG') as captured:
            self.sched1.check_stability_job(label=label, job=job)
        return captured.records

    def test_unstable_ts_is_warned_with_its_eigenvalue(self):
        """Test that an instability is a warning naming the negative root and its eigenvalue"""
        records = self._run_check_stability('rhf_uhf_instability_singlet_ts.out')
        stability = [r for r in records if 'stability' in r.getMessage().lower()
                     or 'wavefunction' in r.getMessage().lower()]
        self.assertTrue(stability, msg='no stability log record emitted')
        self.assertTrue(any(r.levelno == logging.WARNING for r in stability),
                        msg=f'no warning for an unstable TS: {[r.levelno for r in stability]}')
        message = ' '.join(r.getMessage() for r in stability)
        self.assertIn('Triplet-A', message)
        self.assertIn('-0.0642', message)
        self.assertIn('RHF -> UHF', message)
        self.assertIn('Triplet-A', self.sched1.output['C2H6']['wavefunction_stability'])

    def test_stable_ts_is_not_warned(self):
        """Test that a stable wavefunction does not raise a warning"""
        records = self._run_check_stability('stable_unrestricted_doublet_ts.out')
        self.assertFalse([r for r in records if r.levelno >= logging.WARNING],
                         msg=f'a stable TS produced {[r.getMessage() for r in records]}')
        self.assertEqual(self.sched1.output['C2H6']['wavefunction_stability'], 'stable')

    def test_check_stability_job_records_the_structured_verdict_on_the_species(self):
        """Test that the parsed verdict reaches the species object, not only the output summary string"""
        self._run_check_stability('rhf_uhf_instability_singlet_ts.out')
        verdict = self.sched1.species_dict['C2H6'].derived_stability_verdict
        self.assertIsInstance(verdict, dict)
        self.assertEqual(verdict['verdict'], 'external_instability')
        self.assertIs(verdict['restricted'], True)

    def test_a_declared_number_of_radicals_survives_a_contradicting_verdict(self):
        """Test that the check runs, disagrees, warns, and leaves the declared value in place"""
        species = self.sched1.species_dict['C2H6']
        self.addCleanup(setattr, species, 'number_of_radicals', species.number_of_radicals)
        species.number_of_radicals = 1
        records = self._run_check_stability('rhf_uhf_instability_singlet_ts.out')
        self.assertEqual(species.number_of_radicals, 1)
        self.assertIsInstance(species.derived_stability_verdict, dict)
        warnings = ' '.join(r.getMessage() for r in records if r.levelno == logging.WARNING)
        self.assertIn('C2H6', warnings)
        self.assertIn('number_of_radicals = 1', warnings)
        self.assertIn('external instability', warnings)
        self.assertIn('The declared value is the one ARC uses', warnings)

    def test_a_declared_biradical_singlet_contradicted_by_a_stable_verdict_warns(self):
        """Test that a declared broken-symmetry character the calculation does not support is warned about"""
        species = self.sched1.species_dict['C2H6']
        self.addCleanup(setattr, species, 'number_of_radicals', species.number_of_radicals)
        species.number_of_radicals = 2
        records = self._run_check_stability('stable_restricted_singlet_ts.out')
        self.assertEqual(species.number_of_radicals, 2)
        warnings = ' '.join(r.getMessage() for r in records if r.levelno == logging.WARNING)
        self.assertIn('C2H6', warnings)
        self.assertIn('number_of_radicals = 2', warnings)
        self.assertIn('stable under the perturbations considered', warnings)
        self.assertIn('not supported by the calculation', warnings)

    def test_adopting_a_measured_verdict_is_logged_as_such(self):
        """Test that ARC says so when it adopts a verdict the user declared nothing against"""
        species = self.sched1.species_dict['C2H6']
        self.addCleanup(setattr, species, 'number_of_radicals', species.number_of_radicals)
        species.number_of_radicals = None
        self.addCleanup(setattr, species, 'is_ts', species.is_ts)
        species.is_ts = True
        self._honour_the_reference_change()
        records = self._run_check_stability('rhf_uhf_instability_singlet_ts.out')
        self.assertIsNone(species.number_of_radicals)
        warnings = ' '.join(r.getMessage() for r in records if r.levelno == logging.WARNING)
        self.assertIn('No number_of_radicals was declared for C2H6', warnings)
        self.assertIn('adopting that verdict', warnings)
        self.assertIn('run unrestricted', warnings)
        self.assertTrue(adopted_reference_is_unrestricted(species))
        self.assertNotIn(UNREACHABLE_REFERENCE_MESSAGE, self.sched1.output['C2H6']['warnings'])

    def test_a_verdict_no_ess_of_the_run_can_reach_is_reported_and_not_adopted(self):
        """Test that an instability the run's ESSs cannot break the spin symmetry for decides nothing"""
        species = self.sched1.species_dict['C2H6']
        self.addCleanup(setattr, species, 'number_of_radicals', species.number_of_radicals)
        self.addCleanup(setattr, species, 'is_ts', species.is_ts)
        original_warnings = self.sched1.output['C2H6']['warnings']
        self.addCleanup(self.sched1.output['C2H6'].__setitem__, 'warnings', original_warnings)
        species.number_of_radicals = None
        species.is_ts = True
        self.addCleanup(setattr, self.sched1, 'sp_level', self.sched1.sp_level)
        self.sched1.sp_level = Level(method='wb97xd', basis='def2tzvp', software='molpro')
        records = self._run_check_stability('rhf_uhf_instability_singlet_ts.out')
        verdict = species.derived_stability_verdict
        self.assertEqual(verdict['verdict'], 'external_instability')
        self.assertFalse(verdict['reference_change_available'])
        self.assertFalse(adopted_reference_is_unrestricted(species))
        self.assertIn(UNREACHABLE_REFERENCE_MESSAGE, self.sched1.output['C2H6']['warnings'])
        warnings = ' '.join(r.getMessage() for r in records if r.levelno == logging.WARNING)
        self.assertIn('does not act on it', warnings)
        self.assertNotIn('adopting that verdict', warnings)

    def test_a_correlated_single_point_does_not_block_the_adoption_of_a_verdict(self):
        """Test that an sp the verdict decides no reference for is not tested against the adapters"""
        species = self.sched1.species_dict['C2H6']
        self.addCleanup(setattr, species, 'number_of_radicals', species.number_of_radicals)
        self.addCleanup(setattr, species, 'is_ts', species.is_ts)
        original_warnings = self.sched1.output['C2H6']['warnings']
        self.addCleanup(self.sched1.output['C2H6'].__setitem__, 'warnings', original_warnings)
        species.number_of_radicals = None
        species.is_ts = True
        self._honour_the_reference_change()
        for method, basis in [('ccsd(t)-f12', 'cc-pvtz-f12'), ('dlpno-ccsd(t)', 'def2-tzvp')]:
            self.sched1.sp_level = Level(method=method, basis=basis, software='molpro')
            self.assertTrue(self.sched1.stability_verdict_can_be_honoured(label='C2H6'),
                            msg=f'an sp at {method} in an ESS ARC breaks no symmetry for blocked the adoption')
        records = self._run_check_stability('rhf_uhf_instability_singlet_ts.out')
        self.assertTrue(adopted_reference_is_unrestricted(species))
        self.assertNotIn(UNREACHABLE_REFERENCE_MESSAGE, self.sched1.output['C2H6']['warnings'])
        warnings = ' '.join(r.getMessage() for r in records if r.levelno == logging.WARNING)
        self.assertIn('adopting that verdict', warnings)

    def test_stability_verdict_can_be_honoured_reads_the_levels_the_e0_comes_from(self):
        """Test that the geometry, the ZPE and the electronic energy each have to be reachable"""
        for attribute in ['opt_level', 'freq_level', 'sp_level']:
            self.addCleanup(setattr, self.sched1, attribute, getattr(self.sched1, attribute))
        for adapter in sorted(SYMMETRY_BREAKING_ADAPTERS):
            for attribute in ['opt_level', 'freq_level', 'sp_level']:
                setattr(self.sched1, attribute, Level(method='wb97xd', basis='def2tzvp', software=adapter))
            self.assertTrue(self.sched1.stability_verdict_can_be_honoured(label='C2H6'),
                            msg=f'an all-{adapter} run was refused')
        for attribute in ['opt_level', 'freq_level', 'sp_level']:
            for other in ['opt_level', 'freq_level', 'sp_level']:
                setattr(self.sched1, other, Level(method='wb97xd', basis='def2tzvp', software='gaussian'))
            setattr(self.sched1, attribute, Level(method='wb97xd', basis='def2tzvp', software='qchem'))
            self.assertFalse(self.sched1.stability_verdict_can_be_honoured(label='C2H6'),
                             msg=f'a run whose {attribute} is qchem was accepted')

    def test_a_reference_agnostic_level_neither_honours_a_verdict_nor_blocks_it(self):
        """Test that a level ARC writes no reference prefix for is not tested against the adapters"""
        for attribute in ['opt_level', 'freq_level', 'sp_level']:
            self.addCleanup(setattr, self.sched1, attribute, getattr(self.sched1, attribute))
        for other in ['opt_level', 'freq_level', 'sp_level']:
            setattr(self.sched1, other, Level(method='wb97xd', basis='def2tzvp', software='gaussian'))
        for method in ['cbs-qb3', 'am1']:
            self.sched1.sp_level = Level(method=method, software='molpro')
            self.assertIn(self.sched1.sp_level.method_type, REFERENCE_AGNOSTIC_METHOD_TYPES)
            self.assertTrue(self.sched1.stability_verdict_can_be_honoured(label='C2H6'),
                            msg=f'an sp at {method} blocked the adoption')

    def test_a_stability_job_that_died_after_printing_its_verdict_is_read(self):
        """Test that a log holding a complete analysis is read whatever the job status says"""
        species = self.sched1.species_dict['C2H6']
        self.addCleanup(setattr, species, 'derived_stability_verdict', species.derived_stability_verdict)
        species.derived_stability_verdict = None
        self._run_check_stability('orca_rhf_uhf_instability_no_restart_crash.out', status='errored')
        verdict = species.derived_stability_verdict
        self.assertIsInstance(verdict, dict)
        self.assertEqual(verdict['verdict'], 'unattributed_instability')
        self.assertEqual(verdict['n_analyses'], 1)

    def test_a_well_verdict_is_reported_and_not_adopted(self):
        """Test that an instability measured for a species that is not a TS is said to change nothing"""
        species = self.sched1.species_dict['C2H6']
        self.addCleanup(setattr, species, 'number_of_radicals', species.number_of_radicals)
        self.addCleanup(setattr, species, 'is_ts', species.is_ts)
        species.number_of_radicals = None
        species.is_ts = False
        records = self._run_check_stability('rhf_uhf_instability_singlet_ts.out')
        warnings = ' '.join(r.getMessage() for r in records if r.levelno == logging.WARNING)
        self.assertIn('external instability', warnings)
        self.assertIn('does not act on it', warnings)
        self.assertIn('number_of_radicals = 2', warnings)
        self.assertNotIn('adopting that verdict', warnings)
        self.assertEqual(species.derived_stability_verdict['verdict'], 'external_instability')
        self.assertFalse(adopted_reference_is_unrestricted(species))

    def test_a_stable_verdict_on_a_silent_species_adopts_nothing(self):
        """Test that a stable verdict leaves the reference decision exactly where it was"""
        species = self.sched1.species_dict['C2H6']
        self.addCleanup(setattr, species, 'number_of_radicals', species.number_of_radicals)
        species.number_of_radicals = None
        records = self._run_check_stability('stable_restricted_singlet_ts.out')
        self.assertFalse([r for r in records if r.levelno >= logging.WARNING],
                         msg=f'a stable verdict produced {[r.getMessage() for r in records]}')

    def _reference_job(self, job_type, restricted, method='wb97xd'):
        """Build a stand-in for a completed ESS job that memoized the reference its input declared."""
        return SimpleNamespace(job_type=job_type,
                               restricted_used=restricted,
                               level=Level(method=method, basis='def2-TZVP'),
                               )

    def _reset_reference_records(self, label='C2H6'):
        """Clear the SCF reference records and output warnings of a species, restoring them afterwards."""
        species = self.sched1.species_dict[label]
        self.addCleanup(setattr, species, 'scf_references', species.scf_references)
        original_warnings = self.sched1.output[label]['warnings']
        self.addCleanup(self.sched1.output[label].__setitem__, 'warnings', original_warnings)
        species.scf_references = dict()
        self.sched1.output[label]['warnings'] = ''
        return species

    def test_a_mixed_scf_reference_between_sp_and_freq_is_warned_about(self):
        """Test that an E0 summing an unrestricted energy and a restricted ZPE is reported"""
        species = self._reset_reference_records()
        self.sched1.record_scf_reference(label='C2H6', job=self._reference_job('freq', True))
        with self.assertLogs('arc', level='WARNING') as captured:
            self.sched1.record_scf_reference(label='C2H6', job=self._reference_job('sp', False))
        message = ' '.join(r.getMessage() for r in captured.records)
        self.assertIn('C2H6', message)
        self.assertIn('E_elect(unrestricted)', message)
        self.assertIn('ZPE(restricted)', message)
        self.assertEqual(species.scf_references, {'freq': 'restricted', 'sp': 'unrestricted'})
        self.assertIn('different SCF references', self.sched1.output['C2H6']['warnings'])

    def test_one_scf_reference_for_both_jobs_is_not_warned_about(self):
        """Test that the common case, both jobs on one reference, stays silent"""
        species = self._reset_reference_records()
        self.sched1.record_scf_reference(label='C2H6', job=self._reference_job('freq', False))
        with self.assertNoLogs('arc', level='WARNING'):
            self.sched1.record_scf_reference(label='C2H6', job=self._reference_job('sp', False))
        self.assertEqual(species.scf_references, {'freq': 'unrestricted', 'sp': 'unrestricted'})
        self.assertEqual(self.sched1.output['C2H6']['warnings'], '')

    def test_a_composite_sp_records_no_scf_reference(self):
        """Test that a level ARC writes no r/u prefix for is not compared against a DFT job's reference"""
        species = self._reset_reference_records()
        self.sched1.record_scf_reference(label='C2H6', job=self._reference_job('freq', False))
        with self.assertNoLogs('arc', level='WARNING'):
            self.sched1.record_scf_reference(label='C2H6', job=self._reference_job('sp', True, method='cbs-qb3'))
        self.assertEqual(species.scf_references, {'freq': 'unrestricted'})

    def test_the_reference_recorded_is_the_memo_the_job_adapter_left(self):
        """Test that the scheduler reads the reference is_restricted recorded, under that name"""
        species = self._reset_reference_records()
        probe = SimpleNamespace(run_multi_species=False,
                                job_type='freq',
                                level=Level(method='wb97xd', basis='def2-TZVP'),
                                multiplicity=3,
                                species=[self.sched1.species_dict['C2H6']],
                                )
        self.assertFalse(is_restricted(probe))
        self.sched1.record_scf_reference(label='C2H6', job=probe)
        self.assertEqual(species.scf_references, {'freq': 'unrestricted'})

    def test_a_corrupt_reference_record_is_read_as_holding_nothing(self):
        """Test that the consistency check treats a scf_references that is not a mapping as empty"""
        species = self._reset_reference_records()
        for references in [None, [], 'restricted', ['freq', 'restricted']]:
            species.scf_references = references
            with self.assertNoLogs('arc', level='WARNING'):
                self.sched1.check_scf_reference_consistency(label='C2H6')
            self.assertEqual(self.sched1.output['C2H6']['warnings'], '')

    def test_a_job_carrying_no_reference_memo_records_nothing(self):
        """Test that a pipe task, which never wrote an ESS input, is skipped"""
        species = self._reset_reference_records()
        piped = SimpleNamespace(job_type='freq', level=Level(method='wb97xd', basis='def2-TZVP'))
        self.sched1.record_scf_reference(label='C2H6', job=piped)
        self.assertEqual(species.scf_references, dict())

    def test_an_optfreq_job_records_the_reference_its_zpe_came_from(self):
        """Test that a combined opt+freq job is recorded as the source of the ZPE's reference"""
        species = self._reset_reference_records()
        self.sched1.record_scf_reference(label='C2H6', job=self._reference_job('optfreq', True))
        self.assertEqual(species.scf_references, {'freq': 'restricted'})
        with self.assertLogs('arc', level='WARNING') as captured:
            self.sched1.record_scf_reference(label='C2H6', job=self._reference_job('sp', False))
        self.assertIn('ZPE(restricted)', ' '.join(r.getMessage() for r in captured.records))

    def test_a_job_type_that_decides_neither_energy_nor_zpe_records_nothing(self):
        """Test that only the jobs an E0 is built from are compared against each other"""
        species = self._reset_reference_records()
        for job_type in ['opt', 'scan', 'irc', 'orbitals', 'composite', 'conf_opt', 'stability']:
            self.sched1.record_scf_reference(label='C2H6', job=self._reference_job(job_type, True))
            self.assertEqual(species.scf_references, dict(), msg=f'{job_type} recorded a reference')

    def test_an_adopted_verdict_with_a_correlated_sp_is_reported_as_a_mixed_reference(self):
        """Test that an adopted species whose electronic energy stays restricted is reported as mixing"""
        label = 'C2H6'
        species = self._reset_reference_records(label)
        self.addCleanup(setattr, species, 'is_ts', species.is_ts)
        self.addCleanup(setattr, species, 'number_of_radicals', species.number_of_radicals)
        self.addCleanup(setattr, species, 'derived_stability_verdict', species.derived_stability_verdict)
        species.is_ts = True
        species.number_of_radicals = None
        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertTrue(adopted_reference_is_unrestricted(species))
        freq_job = job_factory(job_adapter='gaussian', project='project_test', ess_settings=self.ess_settings,
                               species=[species], job_type='freq',
                               level=Level(method='wb97xd', basis='def2-TZVP'),
                               project_directory=self.project_directory, job_num=911)
        sp_job = job_factory(job_adapter='molpro', project='project_test', ess_settings=self.ess_settings,
                             species=[species], job_type='sp',
                             level=Level(method='ccsd(t)-f12', basis='cc-pvtz-f12'),
                             project_directory=self.project_directory, job_num=912)
        self.assertFalse(is_restricted(freq_job))
        self.assertTrue(is_restricted(sp_job))
        self.sched1.record_scf_reference(label=label, job=freq_job)
        with self.assertLogs('arc', level='WARNING') as captured:
            self.sched1.record_scf_reference(label=label, job=sp_job)
        self.assertEqual(species.scf_references, {'freq': 'unrestricted', 'sp': 'restricted'})
        self.assertIn(MIXED_SCF_REFERENCE_MESSAGE, self.sched1.output[label]['warnings'])
        message = ' '.join(r.getMessage() for r in captured.records)
        self.assertIn('mixes two potential energy surfaces', message)

    def test_a_queued_jobs_scf_reference_survives_a_restart(self):
        """Test that a job restored from a restart reports the reference it ran with, not today's"""
        label = 'C2H6'
        species = self._reset_reference_records(label)
        self.addCleanup(setattr, species, 'is_ts', species.is_ts)
        self.addCleanup(setattr, species, 'derived_stability_verdict', species.derived_stability_verdict)
        self.addCleanup(setattr, self.sched1, 'restart_dict', self.sched1.restart_dict)
        self.addCleanup(self.sched1.running_jobs.pop, label, None)
        job = job_factory(job_adapter='gaussian', project='project_test', ess_settings=self.ess_settings,
                          species=[species], job_type='sp', level=Level(method='wb97xd', basis='def2-TZVP'),
                          project_directory=self.project_directory, job_num=901)
        self.addCleanup(self.sched1.job_dict.get(label, dict()).pop, 'sp', None)
        self.assertTrue(is_restricted(job))
        job_description = job.as_dict()
        self.assertIs(job_description['restricted_used'], True)

        species.is_ts = True
        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.sched1.restart_dict = {'running_jobs': {label: [job_description]}}
        self.sched1.restore_running_jobs()
        restored = self.sched1.job_dict[label]['sp'][job.job_name]
        self.assertIs(restored.restricted_used, True,
                      msg='the restored job reported the reference it would be given today')
        self.sched1.record_scf_reference(label=label, job=restored)
        self.assertEqual(species.scf_references, {'sp': 'restricted'})

    def test_a_restored_job_that_persisted_no_reference_recomputes_one(self):
        """Test that a restart written before the memo existed still restores its jobs"""
        label = 'C2H6'
        species = self._reset_reference_records(label)
        self.addCleanup(setattr, self.sched1, 'restart_dict', self.sched1.restart_dict)
        self.addCleanup(self.sched1.running_jobs.pop, label, None)
        job = job_factory(job_adapter='gaussian', project='project_test', ess_settings=self.ess_settings,
                          species=[species], job_type='sp', level=Level(method='wb97xd', basis='def2-TZVP'),
                          project_directory=self.project_directory, job_num=902)
        self.addCleanup(self.sched1.job_dict.get(label, dict()).pop, 'sp', None)
        job_description = job.as_dict()
        del job_description['restricted_used']
        self.sched1.restart_dict = {'running_jobs': {label: [job_description]}}
        self.sched1.restore_running_jobs()
        restored = self.sched1.job_dict[label]['sp'][job.job_name]
        self.assertIs(restored.restricted_used, True)

    def _abandoned_ts_freq_job(self, label='C2H6'):
        """Put a species into the state a freq job that fails the NMD check leaves it in."""
        species = self._reset_reference_records(label)
        self.addCleanup(setattr, species, 'is_ts', species.is_ts)
        self.addCleanup(setattr, species, 'ts_guesses_exhausted', species.ts_guesses_exhausted)
        self.addCleanup(setattr, species, 'derived_stability_verdict', species.derived_stability_verdict)
        species.is_ts = True
        species.ts_guesses_exhausted = True
        species.ts_checks = {'NMD': False}
        species.scf_references = {'freq': 'restricted', 'sp': 'unrestricted'}
        self.sched1.output[label]['warnings'] = MIXED_SCF_REFERENCE_MESSAGE
        job = MagicMock()
        job.job_adapter = 'gaussian'
        job.job_name = 'freq_a1'
        job.level = Level(method='wb97xd', basis='def2-TZVP')
        job.job_type = 'freq'
        job.restricted_used = True
        job.job_status = ['done', {'status': 'done'}]
        job.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate',
                                                     'calcs', 'Species', 'NH2_freq.out')
        return species, job

    def test_a_switched_away_ts_guess_does_not_write_its_reference_back(self):
        """Test that the freq job of an abandoned TS guess records nothing after the switch cleared it"""
        label = 'C2H6'
        species, job = self._abandoned_ts_freq_job(label)
        with patch.object(self.sched1, 'check_negative_freq', return_value=(True, False)), \
                patch.object(self.sched1, 'determine_most_likely_ts_conformer'), \
                patch.object(self.sched1, 'delete_all_species_jobs'), \
                patch('arc.scheduler.parser.parse_frequencies', return_value=[-1000.0]), \
                patch('arc.scheduler.safe_copy_file'), \
                patch('arc.scheduler.parser.parse_polarizability', return_value=None):
            self.sched1.check_freq_job(label=label, job=job)
        self.assertEqual(species.scf_references, dict())
        self.assertNotIn('different SCF references', self.sched1.output[label]['warnings'])

    def test_a_ts_guess_that_is_kept_still_records_its_freq_reference(self):
        """Test that suppressing the record at a switch did not suppress it for a TS that passed"""
        label = 'C2H6'
        species, job = self._abandoned_ts_freq_job(label)
        species.ts_checks = {'NMD': True}
        species.scf_references = dict()
        self.sched1.output[label]['warnings'] = ''
        with patch.object(self.sched1, 'check_negative_freq', return_value=(True, False)), \
                patch.object(self.sched1, 'check_rxn_e0_by_spc'), \
                patch('arc.scheduler.parser.parse_frequencies', return_value=[-1000.0]), \
                patch('arc.scheduler.safe_copy_file'), \
                patch('arc.scheduler.parser.parse_polarizability', return_value=None):
            self.sched1.check_freq_job(label=label, job=job)
        self.assertEqual(species.scf_references, {'freq': 'restricted'})

    def test_post_sp_actions_records_the_reference_of_the_job_the_energy_came_from(self):
        """Test that the job supplying the electronic energy is recorded under the energy's key"""
        label = 'C2H6'
        species = self._reset_reference_records(label)
        self.addCleanup(setattr, species, 'e_elect', species.e_elect)
        self.addCleanup(self.sched1.output[label]['paths'].__setitem__, 'sp',
                        self.sched1.output[label]['paths'].get('sp'))
        sp_path = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'Species', 'NH2_freq.out')
        self.sched1.post_sp_actions(label=label, sp_path=sp_path,
                                    level=Level(method='wb97xd', basis='def2-TZVP'),
                                    job=self._reference_job('opt', True))
        self.assertEqual(species.scf_references, {'sp': 'restricted'})

    def test_post_sp_actions_records_nothing_where_no_job_is_named(self):
        """Test that a caller with no job to name, a restored species among them, records nothing"""
        label = 'C2H6'
        species = self._reset_reference_records(label)
        self.addCleanup(setattr, species, 'e_elect', species.e_elect)
        self.addCleanup(self.sched1.output[label]['paths'].__setitem__, 'sp',
                        self.sched1.output[label]['paths'].get('sp'))
        sp_path = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'Species', 'NH2_freq.out')
        self.sched1.post_sp_actions(label=label, sp_path=sp_path,
                                    level=Level(method='wb97xd', basis='def2-TZVP'))
        self.assertEqual(species.scf_references, dict())

    def test_an_sp_at_the_opt_level_hands_the_optimization_job_over(self):
        """Test that the branch submitting no sp job still names the job the energy is read from"""
        label = 'C2H6'
        self._reset_reference_records(label)
        self.addCleanup(self.sched1.job_dict[label].pop, 'opt', None)
        self.addCleanup(self.sched1.output[label]['paths'].__setitem__, 'geo',
                        self.sched1.output[label]['paths'].get('geo'))
        opt_job = self._reference_job('opt', True)
        opt_job.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate',
                                                          'calcs', 'Species', 'NH2_freq.out')
        opt_job.rename_output_file = MagicMock()
        self.sched1.job_dict[label]['opt'] = {'opt_a1': opt_job}
        self.sched1.output[label]['paths']['geo'] = opt_job.local_path_to_output_file
        with patch.object(self.sched1, 'post_sp_actions') as post_sp_actions:
            self.sched1.run_sp_job(label=label, level=self.sched1.opt_level)
        self.assertIs(post_sp_actions.call_args.kwargs['job'], opt_job)

    def test_a_single_level_run_can_report_a_mixed_reference(self):
        """Test that a species whose sp level equals its opt level is not blind to a mixed reference"""
        label = 'C2H6'
        species = self._reset_reference_records(label)
        self.addCleanup(setattr, species, 'e_elect', species.e_elect)
        self.addCleanup(self.sched1.output[label]['paths'].__setitem__, 'sp',
                        self.sched1.output[label]['paths'].get('sp'))
        sp_path = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'Species', 'NH2_freq.out')
        self.sched1.record_scf_reference(label=label, job=self._reference_job('freq', False))
        with self.assertLogs('arc', level='WARNING') as captured:
            self.sched1.post_sp_actions(label=label, sp_path=sp_path,
                                        level=Level(method='wb97xd', basis='def2-TZVP'),
                                        job=self._reference_job('opt', True))
        self.assertEqual(species.scf_references, {'freq': 'unrestricted', 'sp': 'restricted'})
        self.assertIn('E_elect(restricted)', ' '.join(r.getMessage() for r in captured.records))
        self.assertIn(MIXED_SCF_REFERENCE_MESSAGE, self.sched1.output[label]['warnings'])

    def _stability_verdict_job(self, label='C2H6'):
        """Build a completed stability job pointing at a real analysis log."""
        species = self.sched1.species_dict[label]
        self.addCleanup(setattr, species, 'derived_stability_verdict', species.derived_stability_verdict)
        self.addCleanup(self.sched1.output[label]['paths'].pop, 'stability', None)
        self.addCleanup(self.sched1.output[label].pop, 'wavefunction_stability', None)
        self.addCleanup(self.sched1.output[label].__setitem__, 'info', self.sched1.output[label]['info'])
        job = MagicMock()
        job.job_status = ['done', {'status': 'done'}]
        job.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'stability',
                                                     'stable_restricted_singlet_ts.out')
        return job

    def test_an_invalidated_analytic_hessian_reaches_the_output_warnings(self):
        """Test that a verdict putting the analytic frequencies out of range is reported in output.yml"""
        label = 'C2H6'
        self._reset_reference_records(label)
        job = self._stability_verdict_job(label)
        verdict = {'verdict': 'internal_instability', 'internal_instability': True,
                   'external_instability': None, 'relaxations': [], 'negative_eigenvectors': [],
                   'lowest_eigenvalue': -0.0731, 'restricted': True, 'invalidates_analytic_freq': True}
        with patch('arc.scheduler.parser.parse_wavefunction_stability', return_value=verdict):
            self.sched1.check_stability_job(label=label, job=job)
        self.assertIn(INVALID_ANALYTIC_FREQ_MESSAGE, self.sched1.output[label]['warnings'])

    def test_a_verdict_leaving_the_analytic_hessian_defined_adds_no_warning(self):
        """Test that the warning is raised by the invalidating verdicts alone"""
        label = 'C2H6'
        self._reset_reference_records(label)
        job = self._stability_verdict_job(label)
        self.sched1.check_stability_job(label=label, job=job)
        self.assertNotIn(INVALID_ANALYTIC_FREQ_MESSAGE, self.sched1.output[label]['warnings'])

    def test_a_stability_verdict_carries_the_log_it_was_read_from(self):
        """Test that the verdict on the species names the analysis that produced it"""
        label = 'C2H6'
        job = self._stability_verdict_job(label)
        self.sched1.check_stability_job(label=label, job=job)
        self.assertEqual(self.sched1.species_dict[label].derived_stability_verdict['log'],
                         job.local_path_to_output_file)

    def _spin_contamination_species(self, label='C2H6', multiplicity=2):
        """Put a species at a given multiplicity with empty output warnings, restoring both afterwards."""
        species = self._reset_reference_records(label)
        self.addCleanup(setattr, species, 'multiplicity', species.multiplicity)
        species.multiplicity = multiplicity
        return species

    def test_a_spin_contaminated_energy_is_warned_about(self):
        """Test that an energy taken from a badly contaminated wavefunction is reported"""
        label = 'C2H6'
        self._spin_contamination_species(label)
        path = os.path.join(ARC_TESTING_PATH, 'stability', 'stable_spin_contaminated_doublet_ts.out')
        with self.assertLogs('arc', level='WARNING') as captured:
            self.sched1.check_spin_contamination(label=label, sp_path=path)
        message = ' '.join(r.getMessage() for r in captured.records)
        self.assertIn('1.7488', message)
        self.assertIn('0.75', message)
        self.assertIn(SPIN_CONTAMINATION_MESSAGE, self.sched1.output[label]['warnings'])

    def test_a_clean_open_shell_energy_is_not_warned_about(self):
        """Test that the ordinary contamination of a converged doublet stays off the warning channel"""
        label = 'C2H6'
        self._spin_contamination_species(label)
        path = os.path.join(ARC_TESTING_PATH, 'stability', 'stable_unrestricted_doublet_ts.out')
        with self.assertNoLogs('arc', level='WARNING'):
            self.sched1.check_spin_contamination(label=label, sp_path=path)
        self.assertEqual(self.sched1.output[label]['warnings'], '')

    def test_a_restricted_energy_has_no_spin_diagnostic_to_check(self):
        """Test that a closed-shell log, which prints no <S**2>, is passed over rather than reported clean"""
        label = 'C2H6'
        self._spin_contamination_species(label, multiplicity=1)
        path = os.path.join(ARC_TESTING_PATH, 'stability', 'stable_restricted_singlet_ts.out')
        with self.assertNoLogs('arc', level='WARNING'):
            self.sched1.check_spin_contamination(label=label, sp_path=path)
        self.assertEqual(self.sched1.output[label]['warnings'], '')

    def test_a_missing_energy_log_is_not_a_spin_contamination_verdict(self):
        """Test that an absent or unnamed log yields neither a warning nor a raise"""
        label = 'C2H6'
        self._spin_contamination_species(label)
        for path in [None, '', os.path.join(ARC_TESTING_PATH, 'stability', 'no_such_log.out')]:
            with self.assertNoLogs('arc', level='WARNING'):
                self.sched1.check_spin_contamination(label=label, sp_path=path)
        self.assertEqual(self.sched1.output[label]['warnings'], '')

    def test_the_spin_contamination_threshold_is_the_one_the_module_documents(self):
        """Test the threshold itself, so a change to it is a deliberate one"""
        self.assertEqual(MAX_S_SQUARED_DEVIATION, 0.1)

    def _prepare_verdict_for_switch(self, verdict, chosen_ts=3, label='C2H6', number_of_radicals=None):
        """Put a species into the state a TS switch would find it in, restoring it afterwards."""
        species = self.sched1.species_dict[label]
        self.addCleanup(setattr, species, 'derived_stability_verdict', species.derived_stability_verdict)
        self.addCleanup(setattr, species, 'scf_references', species.scf_references)
        self.addCleanup(setattr, species, 'chosen_ts', species.chosen_ts)
        self.addCleanup(setattr, species, 'is_ts', species.is_ts)
        self.addCleanup(setattr, species, 'number_of_radicals', species.number_of_radicals)
        original_warnings = self.sched1.output[label]['warnings']
        self.addCleanup(self.sched1.output[label].__setitem__, 'warnings', original_warnings)
        self.sched1.output[label]['warnings'] = MIXED_SCF_REFERENCE_MESSAGE
        species.is_ts = True
        species.number_of_radicals = number_of_radicals
        species.chosen_ts = chosen_ts
        species.scf_references = {'freq': 'restricted', 'sp': 'unrestricted'}
        species.derived_stability_verdict = verdict
        return species

    def test_an_adopted_instability_is_carried_across_a_ts_switch_without_geometry_detail(self):
        """Test that the reference decision survives a TS switch while the abandoned numbers do not"""
        species = self._prepare_verdict_for_switch(
            {'verdict': 'external_instability', 'restricted': True, 'relaxations': ['RHF -> UHF'],
             'negative_eigenvectors': [{'label': 'Triplet-A', 'eigenvalue': -0.0642}],
             'lowest_eigenvalue': -0.0642, 'invalidates_analytic_freq': False,
             'log': '/calcs/TSs/TS0/stability_a5/output.out'})
        self.sched1.carry_stability_verdict_across_ts_switch(label='C2H6')
        self.assertEqual(species.derived_stability_verdict,
                         {'verdict': 'external_instability', 'restricted': True,
                          'relaxations': ['RHF -> UHF'], 'measured_on_ts_guess': 3,
                          'log': '/calcs/TSs/TS0/stability_a5/output.out'})
        self.assertTrue(derived_instability_breaks_spin_symmetry(species))
        self.assertEqual(species.scf_references, dict())
        self.assertNotIn('different SCF references', self.sched1.output['C2H6']['warnings'])

    def test_a_ts_switch_leaves_no_summary_of_the_abandoned_geometry_behind(self):
        """Test that the run summary and output.yml are reduced together, not one of the two"""
        label = 'C2H6'
        summary = 'external_instability (Triplet-A, -0.0642)'
        species = self._prepare_verdict_for_switch(
            {'verdict': 'external_instability', 'restricted': True, 'relaxations': ['RHF -> UHF'],
             'negative_eigenvectors': [{'label': 'Triplet-A', 'eigenvalue': -0.0642}],
             'lowest_eigenvalue': -0.0642, 'invalidates_analytic_freq': True}, label=label)
        self.addCleanup(self.sched1.output[label].__setitem__, 'info', self.sched1.output[label]['info'])
        self.addCleanup(self.sched1.output[label].pop, 'wavefunction_stability', None)
        self.sched1.output[label]['wavefunction_stability'] = summary
        self.sched1.output[label]['info'] = f'T1 = 0.011; Wavefunction stability: {summary}; '
        self.sched1.output[label]['warnings'] += INVALID_ANALYTIC_FREQ_MESSAGE + SPIN_CONTAMINATION_MESSAGE
        self.sched1.carry_stability_verdict_across_ts_switch(label=label)
        self.assertIsNone(self.sched1.output[label]['wavefunction_stability'])
        self.assertEqual(self.sched1.output[label]['info'], 'T1 = 0.011; ')
        self.assertNotIn('Triplet-A', self.sched1.output[label]['info'])
        self.assertNotIn(INVALID_ANALYTIC_FREQ_MESSAGE, self.sched1.output[label]['warnings'])
        self.assertNotIn(SPIN_CONTAMINATION_MESSAGE, self.sched1.output[label]['warnings'])
        self.assertEqual(species.derived_stability_verdict['measured_on_ts_guess'], 3)

    def test_a_ts_switch_strips_the_unreachable_reference_warning(self):
        """Test that the warning goes with the verdict it describes, which is always dropped"""
        label = 'C2H6'
        species = self._prepare_verdict_for_switch(
            {'verdict': 'external_instability', 'restricted': True, 'relaxations': ['RHF -> UHF'],
             'negative_eigenvectors': [], 'lowest_eigenvalue': -0.0642, 'invalidates_analytic_freq': False,
             REFERENCE_CHANGE_AVAILABLE_KEY: False}, label=label)
        self.sched1.output[label]['warnings'] += UNREACHABLE_REFERENCE_MESSAGE
        self.sched1.carry_stability_verdict_across_ts_switch(label=label)
        self.assertIsNone(species.derived_stability_verdict)
        self.assertNotIn(UNREACHABLE_REFERENCE_MESSAGE, self.sched1.output[label]['warnings'])
        self.assertNotIn(MIXED_SCF_REFERENCE_MESSAGE, self.sched1.output[label]['warnings'])

    def test_a_verdict_a_declaration_blocks_is_not_carried_across_a_ts_switch(self):
        """Test that a verdict ARC will never adopt is not promised to the next TS guess"""
        for number_of_radicals in [0, 1, 2]:
            species = self._prepare_verdict_for_switch(
                {'verdict': 'external_instability', 'restricted': True, 'relaxations': ['RHF -> UHF'],
                 'negative_eigenvectors': [], 'lowest_eigenvalue': -0.0642, 'invalidates_analytic_freq': False},
                number_of_radicals=number_of_radicals)
            self.sched1.carry_stability_verdict_across_ts_switch(label='C2H6')
            self.assertIsNone(species.derived_stability_verdict,
                              msg=f'a verdict was carried for a declared number_of_radicals '
                                  f'of {number_of_radicals}')

    def test_every_verdict_that_decides_nothing_is_dropped_at_a_ts_switch(self):
        """Test that a verdict with no consumer is not attributed to the next TS guess"""
        for verdict in [{'verdict': 'stable', 'restricted': True},
                        {'verdict': 'internal_instability', 'restricted': True},
                        {'verdict': 'unknown', 'restricted': None},
                        {'verdict': 'external_instability', 'restricted': False},
                        ]:
            species = self._prepare_verdict_for_switch(dict(verdict))
            self.sched1.carry_stability_verdict_across_ts_switch(label='C2H6')
            self.assertIsNone(species.derived_stability_verdict, msg=f'{verdict} was carried over')
            self.assertEqual(species.scf_references, dict())

    def test_a_dropped_verdict_leaves_the_next_ts_guess_to_be_measured(self):
        """Test that abandoning a guess whose verdict decides nothing re-opens the analysis"""
        for verdict in [{'verdict': 'stable', 'restricted': True},
                        {'verdict': 'internal_instability', 'restricted': True},
                        {'verdict': 'external_instability', 'restricted': False},
                        None,
                        ]:
            species = self._prepare_verdict_for_switch(dict(verdict) if verdict is not None else None)
            species.stability_analysis_ran = True
            self.sched1.carry_stability_verdict_across_ts_switch(label='C2H6')
            self.assertIsNone(species.derived_stability_verdict, msg=f'{verdict} was carried over')
            self.assertFalse(species.stability_analysis_ran,
                             msg=f'the next guess is not measured after {verdict} was dropped')

    def test_a_carried_verdict_leaves_the_next_ts_guess_unmeasured(self):
        """Test that a verdict that decides the next guess' reference is not measured against"""
        species = self._prepare_verdict_for_switch(
            {'verdict': 'external_instability', 'restricted': True, 'relaxations': ['RHF -> UHF'],
             'negative_eigenvectors': [], 'lowest_eigenvalue': -0.0642, 'invalidates_analytic_freq': False})
        species.stability_analysis_ran = True
        self.sched1.carry_stability_verdict_across_ts_switch(label='C2H6')
        self.assertIsNotNone(species.derived_stability_verdict)
        self.assertTrue(species.stability_analysis_ran)

    def test_switch_ts_reduces_the_stability_verdict(self):
        """Test that the TS switch path is what reduces the verdict, not only the helper"""
        species = self.sched1.species_dict['C2H6']
        self.addCleanup(setattr, species, 'ts_guesses_exhausted', species.ts_guesses_exhausted)
        species.ts_guesses_exhausted = True
        with patch.object(self.sched1, 'determine_most_likely_ts_conformer'), \
                patch.object(self.sched1, 'delete_all_species_jobs'), \
                patch.object(self.sched1, 'carry_stability_verdict_across_ts_switch') as carry:
            self.sched1.switch_ts(label='C2H6')
        self.assertTrue(carry.called)

    def test_switch_ts_carries_the_verdict_before_the_next_guess_is_chosen(self):
        """Test that the carried verdict names the guess it was measured on, not the one replacing it"""
        label, abandoned_guess, next_guess = 'C2H6', 3, 7
        species = self._prepare_verdict_for_switch(
            {'verdict': 'external_instability', 'restricted': True, 'relaxations': ['RHF -> UHF'],
             'negative_eigenvectors': [], 'lowest_eigenvalue': -0.0642,
             'invalidates_analytic_freq': False},
            chosen_ts=abandoned_guess, label=label)
        self.addCleanup(setattr, species, 'ts_guesses_exhausted', species.ts_guesses_exhausted)
        self.addCleanup(self.sched1.output[label].pop, 'wavefunction_stability', None)
        species.ts_guesses_exhausted = True

        def choose_the_next_guess(label):
            """Stand in for the TS guess selection, which picks a different guess."""
            self.sched1.species_dict[label].chosen_ts = next_guess

        with patch.object(self.sched1, 'determine_most_likely_ts_conformer',
                          side_effect=choose_the_next_guess), \
                patch.object(self.sched1, 'delete_all_species_jobs'):
            self.sched1.switch_ts(label=label)
        self.assertEqual(species.chosen_ts, next_guess)
        self.assertEqual(species.derived_stability_verdict['measured_on_ts_guess'], abandoned_guess)

    def test_does_output_dict_contain_info(self):
        """Test Scheduler.does_output_dict_contain_info"""
        self.sched1.output = dict()
        self.sched1.initialize_output_dict()
        self.assertFalse(self.sched1._does_output_dict_contain_info())

        self.sched1.output['C2H6']['info'] = 'some text'
        self.sched1.output['C2H6']['job_types']['freq'] = True
        self.sched1.output['C2H6']['paths']['sp'] = 'some/path/out.out'
        self.assertTrue(self.sched1._does_output_dict_contain_info())

    def test_spawn_post_opt_jobs_skips_reactive_ring_rotors(self):
        """Test that no scan job is spawned for a torsion in a TS reactive ring, while a non-ring TS rotor is scanned"""
        cases = [(ARCReaction(r_species=[ARCSpecies(label='nC3H7OO', smiles='CCCO[O]', xyz=NC3H7OO_XYZ)],
                              p_species=[ARCSpecies(label='QOOH', smiles='[CH2]CCOO', xyz=QOOH_XYZ)]), TS0_XYZ, []),
                 (ARCReaction(r_species=[ARCSpecies(label='CH3', smiles='[CH3]', xyz=CH3_XYZ),
                                         ARCSpecies(label='C2H6', smiles='CC', xyz=C2H6_XYZ)],
                              p_species=[ARCSpecies(label='CH4', smiles='C', xyz=CH4_XYZ),
                                         ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=C2H5_XYZ)]), TS1_XYZ,
                  [[6, 4, 5, 9]])]
        job_types = initialize_job_types({'conf_opt': False, 'conf_sp': False, 'opt': True, 'fine': False,
                                          'freq': False, 'sp': False, 'rotors': True, 'irc': False,
                                          'orbitals': False, 'bde': False, 'lennard_jones': False})
        project_directory = tempfile.mkdtemp(prefix='test_reactive_ring_')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        for i, (rxn, ts_xyz, expected_torsions) in enumerate(cases):
            sched = Scheduler(project=f'project_test_reactive_ring_{i}', ess_settings=self.ess_settings,
                              species_list=rxn.r_species + rxn.p_species, rxn_list=[rxn],
                              opt_level=Level(repr=default_levels_of_theory['opt']),
                              scan_level=Level(repr=default_levels_of_theory['scan']),
                              project_directory=project_directory, testing=True, job_types=job_types)
            ts = sched.species_dict[rxn.ts_label]
            self.assertIsNone(ts.mol)
            ts.final_xyz = str_to_xyz(ts_xyz)
            sched.job_dict[ts.label] = dict()
            with patch.object(sched, 'run_stability_job', return_value=False), \
                    patch.object(sched.pipe_planner, 'try_pipe_rotor_scans_1d', return_value=set()), \
                    patch.object(sched, 'run_job') as run_job:
                sched.spawn_post_opt_jobs(label=ts.label, job_name='opt_a1')
            scan_torsions = [call.kwargs['torsions'] for call in run_job.call_args_list
                             if call.kwargs.get('job_type') == 'scan']
            self.assertEqual(scan_torsions, expected_torsions, rxn.label)
            if not expected_torsions:
                self.assertEqual(len(ts.rotors_dict), 3)
                self.assertTrue(all('(code: reactive ring)' in rotor['invalidation_reason']
                                    for rotor in ts.rotors_dict.values()))

    def test_restart_rechecks_restored_ts_rotors(self):
        """Test that the unchecked rotors of a TS restored from a restart file are checked for its reactive ring"""
        rxn = ARCReaction(r_species=[ARCSpecies(label='nC3H7OO', smiles='CCCO[O]', xyz=NC3H7OO_XYZ)],
                          p_species=[ARCSpecies(label='QOOH', smiles='[CH2]CCOO', xyz=QOOH_XYZ)])
        rxn.ts_label = 'TS0'
        ts = ARCSpecies(label='TS0', is_ts=True, xyz=TS0_XYZ)
        ts.final_xyz = str_to_xyz(TS0_XYZ)
        for i, rotor in enumerate(find_internal_rotors(ts.mol)):  # As written by a run before the check existed.
            ts.rotors_dict[i] = rotor
        ts.number_of_rotors = len(ts.rotors_dict)
        self.assertEqual(ts.number_of_rotors, 3)
        job_types = initialize_job_types({'conf_opt': False, 'conf_sp': False, 'opt': True, 'fine': False,
                                          'freq': False, 'sp': False, 'rotors': True, 'irc': False,
                                          'orbitals': False, 'bde': False, 'lennard_jones': False})
        project_directory = tempfile.mkdtemp(prefix='test_reactive_ring_restart_')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        Scheduler(project='project_test_reactive_ring_restart', ess_settings=self.ess_settings,
                  species_list=rxn.r_species + rxn.p_species + [ts], rxn_list=[rxn],
                  opt_level=Level(repr=default_levels_of_theory['opt']),
                  scan_level=Level(repr=default_levels_of_theory['scan']),
                  project_directory=project_directory, testing=True, job_types=job_types)
        for rotor in ts.rotors_dict.values():
            self.assertIs(rotor['success'], False)
            self.assertIn('(code: reactive ring)', rotor['invalidation_reason'])

    def test_restart_cancels_the_scan_jobs_of_reactive_ring_rotors(self):
        """Test that a restart cancels the running scan jobs of TS rotors the reactive-ring check rejects"""
        rxn = ARCReaction(r_species=[ARCSpecies(label='nC3H7OO', smiles='CCCO[O]', xyz=NC3H7OO_XYZ)],
                          p_species=[ARCSpecies(label='QOOH', smiles='[CH2]CCOO', xyz=QOOH_XYZ)])
        rxn.ts_label = 'TS0'
        ts = ARCSpecies(label='TS0', is_ts=True, xyz=TS0_XYZ)
        ts.final_xyz = str_to_xyz(TS0_XYZ)
        for i, rotor in enumerate(find_internal_rotors(ts.mol)):
            ts.rotors_dict[i] = rotor
        ts.number_of_rotors = len(ts.rotors_dict)
        job_types = initialize_job_types({'conf_opt': False, 'conf_sp': False, 'opt': True, 'fine': False,
                                          'freq': False, 'sp': False, 'rotors': True, 'irc': False,
                                          'orbitals': False, 'bde': False, 'lennard_jones': False})
        project_directory = tempfile.mkdtemp(prefix='test_reactive_ring_cancel_')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        with patch.object(Scheduler, 'cancel_reactive_ring_scan_jobs') as cancel:
            Scheduler(project='project_test_reactive_ring_cancel', ess_settings=self.ess_settings,
                      species_list=rxn.r_species + rxn.p_species + [ts], rxn_list=[rxn],
                      opt_level=Level(repr=default_levels_of_theory['opt']),
                      scan_level=Level(repr=default_levels_of_theory['scan']),
                      project_directory=project_directory, testing=True, job_types=job_types)
        cancel.assert_called_once_with(label='TS0')

    def test_cancel_reactive_ring_scan_jobs(self):
        """Test that only the running scan jobs of reactive-ring rejections are cancelled, incore jobs without a delete"""
        rejected = {'pivots': [2, 3], 'success': False, 'invalidation_reason': REACTIVE_RING_INVALIDATION_REASON}
        kept = {'pivots': [3, 4], 'success': None, 'invalidation_reason': ''}
        jobs = {'scan_a1': MagicMock(rotor_index=0, execution_type='queue'),
                'scan_a2': MagicMock(rotor_index=1, execution_type='queue')}
        directed_jobs = {'directed_scan_a3': MagicMock(rotor_index=0, execution_type='incore')}
        sched = MagicMock()
        sched.species_dict = {'TS0': SimpleNamespace(rotors_dict={0: rejected, 1: kept})}
        sched.job_dict = {'TS0': {'scan': dict(jobs), 'directed_scan': dict(directed_jobs)}}
        sched.running_jobs = {'TS0': ['scan_a1', 'scan_a2', 'directed_scan_a3']}

        Scheduler.cancel_reactive_ring_scan_jobs(sched, label='TS0')

        jobs['scan_a1'].delete.assert_called_once()
        jobs['scan_a2'].delete.assert_not_called()
        directed_jobs['directed_scan_a3'].delete.assert_not_called()
        self.assertEqual(sched.running_jobs['TS0'], ['scan_a2'])
        self.assertEqual(list(sched.job_dict['TS0']['scan']), ['scan_a2'])
        self.assertEqual(sched.job_dict['TS0']['directed_scan'], dict())

    def test_non_rotor(self):
        """Test that a 180 degree angle on either side of a torsion is not considered as a rotor."""
        self.sched1.species_dict['CtripCO'].rotors_dict = {
            0: {'torsion': [1, 2, 3, 4], 'top': [3, 5], 'scan': [1, 2, 3, 5], 'number_of_running_jobs': 0,
                'success': None, 'invalidation_reason': '', 'times_dihedral_set': 0, 'trsh_methods': [], 'scan_path': '',
                'directed_scan_type': '', 'directed_scan': {}, 'dimensions': 1, 'original_dihedrals': [],
                'cont_indices': []}}
        self.sched1.species_dict['CtripCO'].number_of_rotors = 1
        self.sched1.job_types['rotors'] = True
        self.sched1.run_scan_jobs(label='CtripCO')
        self.assertEqual(self.sched1.species_dict['CtripCO'].rotors_dict[0]['invalidation_reason'],
                         'not a torsional mode (angles = 0.20, 13.03 degrees)')
        self.assertFalse(self.sched1.species_dict['CtripCO'].rotors_dict[0]['success'])

    def test_set_scan_resolution(self):
        """Test that set_scan_resolution() routes the run-level value only for scan jobs."""
        # Absent a run-level value (default None), args are untouched: the settings default applies.
        args = self.sched1.set_scan_resolution(args={'keyword': {}, 'block': {}}, job_type='scan')
        self.assertNotIn('trsh', args)
        sched = Scheduler(project='project_test_scan_res', ess_settings=self.ess_settings,
                          species_list=[self.spc1], scan_level=Level(repr=default_levels_of_theory['scan']),
                          project_directory=self.project_directory, testing=True, job_types=self.job_types1,
                          rotor_scan_resolution=6.0)
        self.assertEqual(sched.rotor_scan_resolution, 6.0)
        # With a run-level value, a scan job receives it through args['trsh']['scan_res'].
        args = sched.set_scan_resolution(args={'keyword': {}, 'block': {}}, job_type='scan')
        self.assertEqual(args['trsh']['scan_res'], 6.0)
        # Non-scan jobs are never affected.
        args = sched.set_scan_resolution(args={'keyword': {}, 'block': {}}, job_type='opt')
        self.assertNotIn('trsh', args)
        # An explicit troubleshooting scan_res is never overridden.
        args = sched.set_scan_resolution(args={'keyword': {}, 'block': {}, 'trsh': {'scan_res': 2.0}},
                                         job_type='scan')
        self.assertEqual(args['trsh']['scan_res'], 2.0)

    def test_rotor_scan_resolution_reaches_scan_job(self):
        """Test that a run-level rotor_scan_resolution reaches a real scan job's scan_res attribute."""
        sched = Scheduler(project='project_test_scan_res_job', ess_settings=self.ess_settings,
                          species_list=[self.spc1], scan_level=Level(repr=default_levels_of_theory['scan']),
                          project_directory=self.project_directory, testing=True, job_types=self.job_types1,
                          rotor_scan_resolution=6.0)
        args = sched.set_scan_resolution(args={'keyword': {}, 'block': {}}, job_type='scan')
        job = job_factory(job_adapter='gaussian', project='project_test_scan_res_job',
                          ess_settings=self.ess_settings, species=[self.spc1], xyz=self.spc1.get_xyz(),
                          job_type='scan', torsions=[[3, 1, 2, 6]], rotor_index=0, args=args,
                          level=Level(repr={'method': 'b3lyp', 'basis': 'cbsb7'}),
                          project_directory=self.project_directory, job_num=201)
        self.assertEqual(job.scan_res, 6.0)
        # Absent the run-level value, the job falls back to the settings resolution (byte-identical).
        args_default = self.sched1.set_scan_resolution(args={'keyword': {}, 'block': {}}, job_type='scan')
        job_default = job_factory(job_adapter='gaussian', project='project_test_scan_res_job',
                                  ess_settings=self.ess_settings, species=[self.spc1], xyz=self.spc1.get_xyz(),
                                  job_type='scan', torsions=[[3, 1, 2, 6]], rotor_index=0, args=args_default,
                                  level=Level(repr={'method': 'b3lyp', 'basis': 'cbsb7'}),
                                  project_directory=self.project_directory, job_num=202)
        self.assertEqual(job_default.scan_res, settings['rotor_scan_resolution'])

    def test_deduce_job_adapter(self):
        """Test the deduce_job_adapter() method."""
        level_1 = Level(method='CBS-QB3')
        job_type_1 = 'composite'
        job_adapter_1 = self.sched1.deduce_job_adapter(level=level_1, job_type=job_type_1)
        self.assertEqual(job_adapter_1, 'gaussian')

        level_2 = Level(repr='dlpno-ccsd(t)/def2-svp')
        job_type_2 = 'sp'
        job_adapter_2 = self.sched1.deduce_job_adapter(level=level_2, job_type=job_type_2)
        self.assertEqual(job_adapter_2, 'orca')

        level_3 = Level(repr='ccsd(t)/cc-pvtz')
        job_type_3 = 'sp'
        job_adapter_3 = self.sched1.deduce_job_adapter(level=level_3, job_type=job_type_3)
        self.assertEqual(job_adapter_3, 'molpro')

        level_4 = Level(repr='m06-2x/def2-svp')
        job_type_4 = 'freq'
        job_adapter_4 = self.sched1.deduce_job_adapter(level=level_4, job_type=job_type_4)
        self.assertEqual(job_adapter_4, 'qchem')

        level_5 = Level(repr='pbe/def2-svp')
        job_type_5 = 'freq'
        job_adapter_5 = self.sched1.deduce_job_adapter(level=level_5, job_type=job_type_5)
        self.assertEqual(job_adapter_5, 'terachem')

    def test_check_scan_job(self):
        """Test the check_scan_job() method."""
        self.job4.job_status[1]['status'] = 'done'
        self.job4.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'rotor_scans', 'N2O3.out')
        self.sched3.check_scan_job(label='methylamine', job=self.job4)
        self.assertTrue(self.sched3.species_dict['methylamine'].rotors_dict[self.job4.rotor_index]['success'])
        # arc/output.py's _parse_scan_constraints dispatches the constraint
        # parser on this field; a completed scan job must stamp it with the
        # adapter that actually produced the log (see TestScanSoftwareStamp
        # below for isolated coverage of both producer call sites).
        self.assertEqual(self.sched3.species_dict['methylamine'].rotors_dict[self.job4.rotor_index]['scan_software'],
                         self.job4.job_adapter)

        self.job4.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'rotor_scans', 'l103_err.out')
        self.job4.job_status[1]['status'] = 'errored'
        self.job4.job_status[1]['error'] = 'Internal coordinate error'
        self.sched3.check_scan_job(label='methylamine', job=self.job4)
        self.assertFalse(self.sched3.species_dict['methylamine'].rotors_dict[self.job4.rotor_index]['success'])
        self.assertIn('Internal coordinate error', self.sched3.species_dict['methylamine'].rotors_dict[self.job4.rotor_index]['invalidation_reason'])

    def test_check_rxn_e0_by_spc(self):
        """Test the check_rxn_e0_by_spc() method."""
        rxn_dict = \
            {'label': 'nC3H7 <=> iC3H7', 'index': 0, 'multiplicity': 2, 'charge': 0, 'reactants': ['nC3H7'], 'products': ['iC3H7'],
             'r_species': [{'force_field': 'MMFF94s', 'is_ts': False, 'label': 'nC3H7',
                            'long_thermo_description': "Bond corrections: {'C-C': 2, 'C-H': 7}\n", 'multiplicity': 2,
                            'charge': 0, 'compute_thermo': True, 'number_of_rotors': 0, 'arkane_file': None,
                            'consider_all_diastereomers': True, 'e_elect': -311073.1524474179, 'run_time': 64.0,
                            'opt_level': 'b3lyp/6-31g(d,p)', 'conf_is_isomorphic': True,
                            'bond_corrections': {'C-C': 2, 'C-H': 7}, 'mol': {'atoms': [
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 1, 'charge': 0, 'label': '*1',
                      'lone_pairs': 0, 'id': -26782, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-26781: 1.0, -26779: 1.0, -26778: 1.0}},
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '*2',
                      'lone_pairs': 0, 'id': -26781, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-26782: 1.0, -26780: 1.0, -26777: 1.0, -26776: 1.0}},
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26780, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-26781: 1.0, -26775: 1.0, -26774: 1.0, -26773: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26779, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26782: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26778, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26782: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26777, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26781: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '*3',
                      'lone_pairs': 0, 'id': -26776, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26781: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26775, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26780: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26774, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26780: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26773, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26780: 1.0}}], 'multiplicity': 2, 'props': {}, 'atom_order': [-26782, -26781, -26780, -26779, -26778, -26777, -26776, -26775, -26774, -26773]},
                            'initial_xyz': 'C       1.37804355    0.27791700   -0.19511840\nC       0.17557158   -0.34036318    0.43265003\nC      -0.83187173    0.70418067    0.88324591\nH       2.32472110   -0.25029805   -0.17789388\nH       1.28332450    1.14667614   -0.83695597\nH      -0.29365298   -1.02042821   -0.28596734\nH       0.48922284   -0.93756983    1.29560539\nH      -1.19281782    1.29832390    0.03681748\nH      -1.69636720    0.21982441    1.34850246\nH      -0.39178710    1.38838724    1.61666119',
                            'final_xyz': 'C       1.39393700    0.26537900   -0.20838600\nC       0.19342400   -0.33106400    0.44496000\nC      -0.84902400    0.70694400    0.88292100\nH       2.32817400   -0.28300300   -0.27001000\nH       1.31393200    1.18780400   -0.77595800\nH      -0.29199000   -1.04919300   -0.24230300\nH       0.50265500   -0.93665200    1.30845500\nH      -1.19276500    1.30046200    0.02844400\nH      -1.72540600    0.22697800    1.32987900\nH      -0.42854900    1.39899700    1.61954600',
                            'checkfile': '/storage/ce_dana/alongd/runs/ARC/debug13/calcs/Species/nC3H7/opt_a23998/check.chk',
                            'cheap_conformer': 'C       1.33903242    0.28849749    0.51672185\nC       0.18657092   -0.40960576   -0.12107655\nC      -1.14634812    0.08737020    0.41314525\nH       2.30155135   -0.20706406    0.57574068\nH       1.28914995    1.34778675    0.74274705\nH       0.23056488   -0.25673734   -1.20448178\nH       0.27283709   -1.48573243    0.06394873\nH      -1.22296487   -0.06954352    1.49444959\nH      -1.97028260   -0.45101088   -0.06611972\nH      -1.28011103    1.15603956    0.21368858',
                            'conformers': [
                                'C       1.37804355    0.27791700   -0.19511840\nC       0.17557158   -0.34036318    0.43265003\nC      -0.83187173    0.70418067    0.88324591\nH       2.32472110   -0.25029805   -0.17789388\nH       1.28332450    1.14667614   -0.83695597\nH      -0.29365298   -1.02042821   -0.28596734\nH       0.48922284   -0.93756983    1.29560539\nH      -1.19281782    1.29832390    0.03681748\nH      -1.69636720    0.21982441    1.34850246\nH      -0.39178710    1.38838724    1.61666119'],
                            'conformer_energies': [None], 'conformers_before_opt': [
                     'C       1.37804355    0.27791700   -0.19511840\nC       0.17557158   -0.34036318    0.43265003\nC      -0.83187173    0.70418067    0.88324591\nH       2.32472110   -0.25029805   -0.17789388\nH       1.28332450    1.14667614   -0.83695597\nH      -0.29365298   -1.02042821   -0.28596734\nH       0.48922284   -0.93756983    1.29560539\nH      -1.19281782    1.29832390    0.03681748\nH      -1.69636720    0.21982441    1.34850246\nH      -0.39178710    1.38838724    1.61666119']}],
             'p_species': [{'force_field': 'MMFF94s', 'is_ts': False, 'label': 'iC3H7',
                            'long_thermo_description': "Bond corrections: {'C-C': 2, 'C-H': 7}\n", 'multiplicity': 2,
                            'charge': 0, 'compute_thermo': True, 'number_of_rotors': 0, 'arkane_file': None,
                            'consider_all_diastereomers': True, 'e_elect': -311090.81145707075, 'run_time': 61.0,
                            'opt_level': 'b3lyp/6-31g(d,p)', 'conf_is_isomorphic': True,
                            'bond_corrections': {'C-C': 2, 'C-H': 7}, 'mol': {'atoms': [
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32758, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-32757: 1.0, -32755: 1.0, -32754: 1.0, -32753: 1.0}},
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 1, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32757, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-32758: 1.0, -32756: 1.0, -32752: 1.0}},
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32756, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-32757: 1.0, -32751: 1.0, -32750: 1.0, -32749: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32755, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-32758: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32754, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-32758: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32753, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-32758: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32752, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-32757: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32751, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-32756: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32750, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-32756: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -32749, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-32756: 1.0}}], 'multiplicity': 2, 'props': {}, 'atom_order': [-32758, -32757, -32756, -32755, -32754, -32753, -32752, -32751, -32750, -32749]},
                            'initial_xyz': 'C       1.29196387    0.15815210    0.32047503\nC      -0.03887789   -0.17543467    0.89494533\nC      -1.26222918    0.47039644    0.34836510\nH       1.40933232    1.23955428    0.20511486\nH       2.08593721   -0.19903577    0.98301313\nH       1.41699441   -0.31973461   -0.65525752\nH      -0.13933823   -1.05339936    1.52398873\nH      -1.51964710    0.03926484   -0.62319221\nH      -2.10441807    0.31322346    1.02876738\nH      -1.11812298    1.54852996    0.23271515',
                            'final_xyz': 'C       1.30445700    0.14930600    0.33486100\nC      -0.03852500   -0.17292600    0.89922000\nC      -1.27617300    0.46478200    0.36303900\nH       1.43314300    1.23075500    0.19226600\nH       2.11562900   -0.20650200    0.97825700\nH       1.46118100   -0.30946900   -0.65803600\nH      -0.13925700   -1.05320700    1.52930100\nH      -1.56011100    0.05987600   -0.62504600\nH      -2.13512900    0.31314300    1.02467200\nH      -1.14362100    1.54575800    0.22040200',
                            'checkfile': '/storage/ce_dana/alongd/runs/ARC/debug13/calcs/Species/iC3H7/opt_a23999/check.chk',
                            'cheap_conformer': 'C      -1.28873024    0.06292844    0.10889819\nC       0.01096161   -0.45756396   -0.39342150\nC       1.28410310    0.11324608    0.12206177\nH      -1.49844465    1.04581965   -0.32238736\nH      -1.28247249    0.14649430    1.19953628\nH      -2.09838469   -0.61664655   -0.17318515\nH       0.02736023   -1.06013834   -1.29522253\nH       2.12255117   -0.53409831   -0.15158596\nH       1.26342625    0.19628892    1.21256167\nH       1.45962973    1.10366979   -0.30725541',
                            'conformers': [
                                'C       1.29196387    0.15815210    0.32047503\nC      -0.03887789   -0.17543467    0.89494533\nC      -1.26222918    0.47039644    0.34836510\nH       1.40933232    1.23955428    0.20511486\nH       2.08593721   -0.19903577    0.98301313\nH       1.41699441   -0.31973461   -0.65525752\nH      -0.13933823   -1.05339936    1.52398873\nH      -1.51964710    0.03926484   -0.62319221\nH      -2.10441807    0.31322346    1.02876738\nH      -1.11812298    1.54852996    0.23271515'],
                            'conformer_energies': [None], 'conformers_before_opt': [
                     'C       1.29196387    0.15815210    0.32047503\nC      -0.03887789   -0.17543467    0.89494533\nC      -1.26222918    0.47039644    0.34836510\nH       1.40933232    1.23955428    0.20511486\nH       2.08593721   -0.19903577    0.98301313\nH       1.41699441   -0.31973461   -0.65525752\nH      -0.13933823   -1.05339936    1.52398873\nH      -1.51964710    0.03926484   -0.62319221\nH      -2.10441807    0.31322346    1.02876738\nH      -1.11812298    1.54852996    0.23271515']}],
             'ts_species': {'force_field': 'MMFF94s', 'is_ts': True, 'label': 'TS0', 'long_thermo_description': '',
                            'multiplicity': 2, 'charge': 0, 'compute_thermo': False, 'number_of_rotors': 0,
                            'arkane_file': None, 'consider_all_diastereomers': True, 'ts_guesses': [
                     {'t0': '2022-05-26T23:41:03.211794', 'method': 'autotst', 'method_index': 0,
                      'method_direction': 'F', 'success': True, 'energy': 457.09273842390394, 'index': 0,
                      'imaginary_freqs': None, 'conformer_index': 0, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:17.672822',
                      'initial_xyz': 'C       0.06870000   -0.52310000   -0.65000000\nC       1.32690000   -0.17800000    0.12310000\nC      -1.61580000    0.23640000    0.43190000\nH      -0.94590000   -0.88470000    0.05810000\nH       0.00080000   -0.05630000   -1.65490000\nH       1.36660000   -0.77570000    1.05730000\nH       2.21840000   -0.40920000   -0.49590000\nH       1.32650000    0.90240000    0.37660000\nH      -1.31000000    1.13360000    1.01690000\nH      -2.43630000    0.55470000   -0.26310000',
                      'opt_xyz': 'C       0.61418600   -0.57125700   -1.11315700\nC       1.52769300   -0.13581800   -0.03895800\nC      -1.76088500    0.17894600    0.38021300\nH      -1.71341100   -0.89798100    0.54776400\nH      -0.18837200    0.14692200   -1.38061400\nH       1.10728900   -0.50824200    0.91631600\nH       2.52104100   -0.60562200   -0.12803100\nH       1.64035400    0.95840500    0.07522100\nH      -1.26555200    0.85783800    1.07567800\nH      -2.48244300    0.57690900   -0.33443300'},
                     {'t0': '2022-05-26T23:41:03.211794', 'method': 'autotst', 'method_index': 1,
                      'method_direction': 'F', 'success': True, 'energy': None, 'index': 1, 'imaginary_freqs': None,
                      'conformer_index': 1, 'successful_irc': None, 'successful_normal_mode': None,
                      'execution_time': '0:00:17.672822',
                      'initial_xyz': 'C       1.27140000   -0.19880000    0.27060000\nC      -1.52300000    0.74650000   -0.35470000\nC       0.02370000   -0.73290000   -0.41700000\nH      -0.96230000   -0.12240000   -1.12430000\nH       1.61070000    0.72270000   -0.24570000\nH       1.04290000    0.03790000    1.33060000\nH       2.07780000   -0.96010000    0.22660000\nH      -1.03890000    1.73950000   -0.25260000\nH      -2.20860000    0.44390000    0.46320000\nH      -0.29360000   -1.67630000    0.10330000'},
                     {'t0': '2022-05-26T23:41:20.911151', 'method': 'autotst', 'method_index': 2,
                      'method_direction': 'R', 'success': True, 'energy': None, 'index': 2, 'imaginary_freqs': None,
                      'conformer_index': 2, 'successful_irc': None, 'successful_normal_mode': None,
                      'execution_time': '0:00:15.846009',
                      'initial_xyz': 'C       1.71270000   -0.29390000    0.04290000\nC      -1.29740000   -0.16230000   -0.10640000\nC      -0.10790000    0.76300000   -0.32970000\nH       1.08610000    0.51490000   -0.81510000\nH       1.48670000   -1.37900000   -0.00500000\nH       2.09490000    0.08260000    1.01380000\nH      -2.24090000    0.38520000   -0.31090000\nH      -1.29900000   -0.52400000    0.94300000\nH      -1.22320000   -1.03060000   -0.79300000\nH      -0.21210000    1.64410000    0.36030000'},
                     {'t0': '2022-05-26T23:41:20.911151', 'method': 'autotst', 'method_index': 3,
                      'method_direction': 'R', 'success': True, 'energy': None, 'index': 3, 'imaginary_freqs': None,
                      'conformer_index': 3, 'successful_irc': None, 'successful_normal_mode': None,
                      'execution_time': '0:00:15.846009',
                      'initial_xyz': 'C       0.14170000    0.77420000   -0.31580000\nC       1.33000000   -0.02950000    0.17580000\nC      -1.64460000   -0.24370000    0.27000000\nH      -0.90890000    0.81700000    0.50710000\nH       0.07300000    0.86010000   -1.42040000\nH       1.38540000    0.02680000    1.28260000\nH       2.26370000    0.38470000   -0.25780000\nH       1.22000000   -1.08930000   -0.13430000\nH      -2.43100000   -0.17220000   -0.52760000\nH      -1.42930000   -1.32810000    0.42060000'},
                     {'t0': '2022-05-26T23:41:37.414768', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'F', 'success': True, 'energy': 180.8585423058248, 'index': 4,
                      'imaginary_freqs': [-1962.3757], 'conformer_index': 4, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:02.875351',
                      'initial_xyz': 'C      -1.06700206   -0.98290563    0.09666771\nC      -0.01429677    0.14576414   -0.26970389\nC       1.07411957    0.53558373    0.44180292\nH      -0.90097213   -1.99955773   -0.00804090\nH      -1.92791498   -0.83538461    0.61150122\nH      -0.72827876   -0.72893214   -0.99451387\nH      -0.19050112    0.61306989   -1.05721593\nH       0.65847605    1.37783504    0.86201239\nH       1.92377913    0.72413480   -0.06474017\nH       1.29896247   -0.04988281    1.22807050',
                      'opt_xyz': 'C      -0.94073000   -1.08641400   -0.13521400\nC      -0.33120900    0.27400500   -0.10738200\nC       1.03297600    0.48550500    0.47374300\nH      -0.34195200   -1.93903700    0.19367700\nH      -2.02636000   -1.19201700   -0.17452300\nH      -0.43853800   -0.46541200   -1.16876000\nH      -1.02644700    1.11801000   -0.14324700\nH       1.47599100    1.43717700    0.14587900\nH       1.72184900   -0.32388500    0.18160800\nH       1.00079200    0.49179200    1.58006000'},
                     {'t0': '2022-05-26T23:41:40.296647', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': 28.47611632832559, 'index': 5,
                      'imaginary_freqs': [-247.9856], 'conformer_index': 5, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.891656',
                      'initial_xyz': 'C      -1.17498302    0.20234825    0.61815524\nC      -0.14451513   -0.22229944   -0.36327118\nC       1.37613642   -0.20638606   -0.29276791\nH      -1.91179907    0.93071961    0.36481556\nH      -1.65566945   -0.34783489    1.24894297\nH      -0.69110036   -0.96965396    0.09287225\nH      -0.52582997   -0.29021457   -1.37846231\nH       1.47940874    0.65462989   -0.58598584\nH       1.83152962   -0.71277982   -1.11679804\nH       1.74064636    0.29004735    0.63243920',
                      'opt_xyz': 'C      -1.20591200    0.47424500    0.36946100\nC      -0.29394000   -0.44521000   -0.36471500\nC       1.20715500   -0.13952500   -0.20416700\nH      -2.28344900    0.44537500    0.18865600\nH      -0.84180200    1.09602600    1.19197200\nH      -0.49608300   -1.48771500   -0.04130900\nH      -0.57224000   -0.44812600   -1.43454100\nH       1.37268200    0.73806100    0.43870000\nH       1.68512500    0.07671500   -1.17052800\nH       1.75228700   -0.98126900    0.24641000'},
                     {'t0': '2022-05-26T23:41:42.194417', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'F', 'success': True, 'energy': 180.85896501125535, 'index': 6,
                      'imaginary_freqs': None, 'conformer_index': 6, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.913523',
                      'initial_xyz': 'C      -1.20873415    0.71530855   -0.52397704\nC       0.31399596    0.81271130   -0.77977943\nC       1.24493992    0.12856162    0.05752292\nH      -1.95132613    0.96989143    0.11211920\nH      -1.76489258    0.25215778   -1.29827285\nH      -0.50961399    1.76985371   -0.06161229\nH       0.62333548    1.23630238   -1.62280464\nH       1.41772580    0.03376916    0.95344961\nH       2.18181777    0.58830410   -0.08182961\nH       0.75694025   -0.81274462    0.46448511',
                      'opt_xyz': 'C      -1.22745100    0.78774200   -0.50893200\nC       0.23979800    0.78871300   -0.77355000\nC       1.19005200    0.16594100    0.20252100\nH      -1.59024000    0.40579300    0.44822900\nH      -1.92750900    0.86269900   -1.34296200\nH      -0.44232200    1.82484300   -0.39144600\nH       0.54372500    0.88104400   -1.82053600\nH       0.90894700    0.40388700    1.24163400\nH       2.22267500    0.51098300    0.04740600\nH       1.18651400   -0.93753000    0.11693700'},
                     {'t0': '2022-05-26T23:41:44.113571', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': 28.47585902933497, 'index': 7,
                      'imaginary_freqs': [-247.9818], 'conformer_index': 7, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.908796',
                      'initial_xyz': 'C       1.02964175    0.68634045   -0.24918917\nC       0.06535231   -0.13109583    0.30977562\nC      -1.42600191   -0.11903072    0.61585003\nH       1.62296820    1.51068902    0.02148751\nH       1.33659041    0.45585936   -1.11817455\nH       0.58185875    0.60811055    0.84054208\nH       0.52493042   -1.22503757    0.64394557\nH      -1.53707004    0.36833912    1.51066422\nH      -1.94065011   -0.88619411    0.56052530\nH      -1.86324120    0.63528162   -0.11346494',
                      'opt_xyz': 'C       0.95062100    0.66915600   -0.44795600\nC       0.21199900   -0.19479100    0.51349900\nC      -1.28747400    0.12620900    0.65892300\nH       0.54583300    1.63209800   -0.77168200\nH       1.97180500    0.41800500   -0.74581000\nH       0.70253200   -0.12992100    1.50713500\nH       0.35123500   -1.25292900    0.22542900\nH      -1.54934400    0.40978500    1.68835900\nH      -1.91697900   -0.73384400    0.38894600\nH      -1.58584900    0.95949400    0.00511800'},
                     {'t0': '2022-05-26T23:41:46.029238', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'F', 'success': True, 'energy': 180.8588232342736, 'index': 8,
                      'imaginary_freqs': None, 'conformer_index': 8, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.874294',
                      'initial_xyz': 'C      -1.48518515   -0.23374367   -0.23082997\nC      -0.24483547    0.51183116    0.41401744\nC       1.12809086    0.09508162    0.25930798\nH      -2.24081373    0.35100281   -0.63667858\nH      -1.89986777   -1.13838661   -0.37233800\nH      -1.16789770   -0.34802154    0.82655811\nH      -0.53055906    1.36634922    0.83735430\nH       1.39702737    0.53136593   -0.54898417\nH       1.63798177    0.33128169    1.15132225\nH       1.27656150   -0.84899682   -0.05066128',
                      'opt_xyz': 'C      -1.50785500   -0.17356000   -0.23397900\nC      -0.37774200    0.49776600    0.46979900\nC       1.03237900    0.04062300    0.25540200\nH      -2.43960000    0.36960000   -0.40168300\nH      -1.30039300   -1.05924100   -0.83912300\nH      -1.21188200   -0.32485400    1.02906500\nH      -0.55506800    1.52491200    0.80243500\nH       1.43111700    0.40933600   -0.70936600\nH       1.70382800    0.39261800    1.05206300\nH       1.09571900   -1.05943700    0.22445500'},
                     {'t0': '2022-05-26T23:41:47.908778', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': 180.86181630415376, 'index': 9,
                      'imaginary_freqs': None, 'conformer_index': 9, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.874496',
                      'initial_xyz': 'C       1.76142025   -0.32615915   -0.31378934\nC       0.26012418   -0.50720125   -0.57989764\nC      -0.83998406    0.51821351   -0.61445594\nH       2.41904354   -0.49318668   -1.15896177\nH       2.07220912   -0.43000120    0.63903868\nH       0.91324747   -0.90038979   -0.88085961\nH      -0.25748059   -1.59517562   -0.60023719\nH      -0.67072332    1.01107574    0.27245176\nH      -1.69623446    0.36441237   -1.05993485\nH      -0.47145543    1.33631945   -1.17536044',
                      'opt_xyz': 'C       1.65863100   -0.47380300   -0.35467500\nC       0.17468800   -0.61083700   -0.40103100\nC      -0.68874000    0.56815100   -0.72949700\nH       2.11011800    0.47530200   -0.65302000\nH       2.24747600   -1.16958900    0.24557700\nH       0.98476000   -1.05412600   -1.31295700\nH      -0.25460900   -1.43358700    0.17871800\nH      -0.78230000    1.24962700    0.13753100\nH      -1.70193000    0.26482300   -1.03037000\nH      -0.25792800    1.16194800   -1.55228100'},
                     {'t0': '2022-05-26T23:41:51.692281', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': 180.85910678829532, 'index': 10,
                      'imaginary_freqs': None, 'conformer_index': 10, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.857693',
                      'initial_xyz': 'C       1.45932961   -0.40568867   -0.48573068\nC       0.07347362    0.17359543   -0.56615287\nC      -1.32016015   -0.44909847   -0.37993887\nH       2.03022957   -1.01502204   -0.05810530\nH       2.13141823    0.20080915   -1.03013027\nH       0.64928532   -0.32624221   -1.27710116\nH      -0.01041280    1.17878938   -0.71969837\nH      -1.24753046   -1.07629967    0.34866881\nH      -2.06527519    0.17695132   -0.33415791\nH      -1.42948318   -1.16617382   -1.21987271',
                      'opt_xyz': 'C       1.39402000   -0.37546100   -0.68083900\nC       0.05132900    0.27274800   -0.69128200\nC      -1.17344800   -0.50633300   -0.32188400\nH       1.45964400   -1.44843800   -0.48606700\nH       2.28898600    0.23638300   -0.55441700\nH       0.70827900   -0.08260100   -1.75281000\nH       0.03733000    1.36144800   -0.58346900\nH      -1.25420900   -0.63011100    0.77497600\nH      -2.09322700   -0.01591200   -0.67250300\nH      -1.14782900   -1.52010300   -0.75392300'},
                     {'t0': '2022-05-26T23:41:53.557121', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'F', 'success': True, 'energy': 180.85898338980041, 'index': 11,
                      'imaginary_freqs': None, 'conformer_index': 11, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.856500',
                      'initial_xyz': 'C      -1.51046324   -0.05654161   -0.17449787\nC      -0.19565231   -0.49037039    0.45278808\nC       0.89220119    0.38680810    0.88834977\nH      -2.35916400   -0.09211956    0.43497366\nH      -1.94329524    0.31062192   -0.99955177\nH      -0.51100838   -0.25559592   -0.80780500\nH      -0.27175251   -1.40853167    0.85534734\nH       1.03489912    1.27390730    0.29284352\nH       1.72903490   -0.00521671    0.87521076\nH       0.39971116    0.88248992    1.71725774',
                      'opt_xyz': 'C      -1.55102600   -0.01764800   -0.15063200\nC      -0.27961500   -0.49037900    0.46836100\nC       0.81488200    0.47842700    0.79522200\nH      -2.46365400   -0.60315800   -0.02583100\nH      -1.63676300    1.03145600   -0.44337600\nH      -0.60079400   -0.65705600   -0.77833000\nH      -0.32952200   -1.43561000    1.01706800\nH       0.94016900    1.22520500   -0.00580600\nH       1.78194900   -0.02655600    0.93690600\nH       0.58888500    1.04077100    1.72133500'},
                     {'t0': '2022-05-26T23:41:55.421723', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': 180.8587628477835, 'index': 12,
                      'imaginary_freqs': None, 'conformer_index': 12, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.853634',
                      'initial_xyz': 'C      -0.58550024    1.08561718   -0.15663983\nC       0.42889541    0.45084977    0.59194607\nC       1.37000704   -0.60319155    0.74652404\nH      -1.47753155    1.46616268    0.40266326\nH      -0.57472807    1.33417463   -1.12147713\nH      -0.71895623   -0.17481887    0.10822438\nH       0.81476825    1.35811055    1.05770040\nH       1.98313403   -0.47356719   -0.09324066\nH       1.39554203   -1.56005931    0.87285095\nH       1.42157912   -0.48325348    1.76832342',
                      'opt_xyz': 'C      -0.63997100    0.88380900   -0.20145300\nC       0.34594100    0.56011900    0.86922900\nC       1.40902800   -0.46981300    0.64022500\nH      -1.13365300    1.85720800   -0.20589100\nH      -0.63149900    0.29502200   -1.12181100\nH      -0.89396500    0.17556700    0.86618200\nH       0.51614700    1.33529500    1.62234300\nH       2.23406900   -0.06376700    0.02428000\nH       1.00940700   -1.34313100    0.09886900\nH       1.84170500   -0.83028400    1.58490200'},
                     {'t0': '2022-05-26T23:41:59.196027', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': None, 'index': 13, 'imaginary_freqs': None,
                      'conformer_index': 13, 'successful_irc': None, 'successful_normal_mode': None,
                      'execution_time': '0:00:01.878886',
                      'initial_xyz': 'C       0.96748012   -0.81678355   -0.31229910\nC       0.37854999    0.36768591    0.51071781\nC      -1.07459605    0.66234928    0.62251425\nH       1.73308313   -0.68185002   -0.92448866\nH       1.04368222   -1.71272838    0.00532236\nH       1.10471821    0.25733542    0.11611196\nH       1.05092859    1.07043862    1.24027312\nH      -1.39371753    0.75815308   -0.36373425\nH      -1.41272283    1.44293737    1.03271341\nH      -1.48376942   -0.15894234    1.14175749'},
                     {'t0': '2022-05-26T23:42:01.079910', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'F', 'success': True, 'energy': 180.85898601531517, 'index': 14,
                      'imaginary_freqs': None, 'conformer_index': 14, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.857904',
                      'initial_xyz': 'C       0.76262105   -0.76400435    0.93583715\nC      -0.65082604   -0.60366839    0.43407044\nC      -1.19700181    0.60047364   -0.20144393\nH       1.00494814   -1.72844446    1.08477473\nH       1.24139726   -0.43856725    1.77211285\nH       0.63145959   -0.97310996   -0.35562319\nH      -1.31025076   -1.13134599    0.93603009\nH      -0.49987090    1.15099120   -0.75667942\nH      -1.96762645    0.30284613   -0.80094039\nH      -1.43324780    1.20295048    0.58301079',
                      'opt_xyz': 'C       0.72155900   -0.86117000    0.97318200\nC      -0.61848700   -0.72093700    0.33453400\nC      -1.12815500    0.62572200   -0.07815300\nH       0.92710900   -1.71883100    1.61623600\nH       1.36170600    0.02052200    1.05393400\nH       0.43775100   -1.18715300   -0.25927600\nH      -1.33931300   -1.51820200    0.53930300\nH      -0.33104900    1.23030900   -0.54085700\nH      -1.95289300    0.54866000   -0.80154200\nH      -1.49662600    1.19920100    0.79378800'},
                     {'t0': '2022-05-26T23:42:02.943259', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': 0.000761395029257983, 'index': 15,
                      'imaginary_freqs': None, 'conformer_index': 15, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.972401',
                      'initial_xyz': 'C      -0.88464719   -0.15515247   -0.44346783\nC       0.51426482    0.36782986    0.08452535\nC       1.67532301   -0.16636546    0.76834404\nH      -1.73940670   -0.05415071    0.08939913\nH      -0.95850170   -0.52282810   -1.42392790\nH      -0.21972930   -0.35693800   -0.12551412\nH       0.56886059    1.42613912   -0.16336975\nH       1.98881960   -1.04647338    0.11560042\nH       2.46397924    0.41142815    0.83529609\nH       1.35643530   -0.47905838    1.72272897',
                      'opt_xyz': 'C      -0.59745500   -0.27793600   -0.57857500\nC       0.48467300    0.52323400    0.05729400\nC       1.53324200   -0.09835600    0.91430400\nH      -1.12677200   -0.90979500    0.15898600\nH      -1.34483800    0.35715900   -1.07491300\nH      -0.19950600   -0.97597300   -1.34239600\nH       0.54166300    1.59877300   -0.13597200\nH       1.41852300   -1.19214200    0.96826900\nH       2.55256900    0.11174200    0.54188200\nH       1.50329900    0.28772400    1.95073600'},
                     {'t0': '2022-05-26T23:42:04.922705', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'F', 'success': True, 'energy': 180.8585974413436, 'index': 16,
                      'imaginary_freqs': None, 'conformer_index': 16, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.981361',
                      'initial_xyz': 'C       1.45752501   -0.34142691   -1.12588692\nC       0.28476441    0.55947429   -0.89261472\nC      -0.58748847    0.63957649    0.22596583\nH       2.07569289   -1.14329469   -0.79948431\nH       2.20992613    0.01069019   -1.45111012\nH       0.37588233   -0.65742445   -1.53954506\nH       0.08357757    1.29397428   -1.73200774\nH      -0.80598485   -0.35379100    0.63491660\nH      -1.45885408    0.91999155    0.06410865\nH       0.05302338    0.88666624    1.01776028',
                      'opt_xyz': 'C       1.50202500   -0.33601100   -1.14219900\nC       0.28744200    0.51648300   -0.99711700\nC      -0.52374300    0.47896400    0.26147300\nH       1.73964600   -1.05373100   -0.35347400\nH       2.29035900   -0.04080700   -1.83732000\nH       0.35207100   -0.55979200   -1.71961800\nH       0.24456900    1.41057000   -1.62633000\nH      -0.63348600   -0.55198700    0.63573000\nH      -1.53267100    0.89056300    0.11241000\nH      -0.03814700    1.06018300    1.06854700'},
                     {'t0': '2022-05-26T23:42:06.912255', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': 180.8586131943157, 'index': 17,
                      'imaginary_freqs': None, 'conformer_index': 17, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.880731',
                      'initial_xyz': 'C      -0.74663436   -0.55736071    0.14056355\nC       0.56897283    0.02896419    0.40835238\nC       1.86595774   -0.83959478    0.71856529\nH      -1.21906948   -0.27343133   -0.69941294\nH      -1.41548634   -1.06072807    0.73646659\nH      -0.18981230    0.43102172    1.16182208\nH       0.84268308    0.82189310   -0.03522815\nH       1.67278290   -1.53900778    1.42446065\nH       2.71231127   -0.30402660    0.73884761\nH       1.88601792   -1.52979362   -0.16393194',
                      'opt_xyz': 'C      -0.75934400   -0.43268700    0.22653800\nC       0.65504000    0.03833200    0.25318700\nC       1.75696600   -0.88370100    0.67631100\nH      -1.48104000    0.05952800   -0.42780900\nH      -0.98946500   -1.43166100    0.60438900\nH      -0.24707700    0.36024700    1.12928800\nH       0.89598200    0.88661000   -0.39432800\nH       1.45676900   -1.49377900    1.54388300\nH       2.66878100   -0.33436700    0.95240100\nH       2.02111100   -1.59058500   -0.13335500'},
                     {'t0': '2022-05-26T23:42:08.797819', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'F', 'success': True, 'energy': 0.0, 'index': 18, 'imaginary_freqs': None,
                      'conformer_index': 18, 'successful_irc': None, 'successful_normal_mode': None,
                      'execution_time': '0:00:01.927177',
                      'initial_xyz': 'C       1.25889516   -0.08437126    0.41225553\nC      -0.00654572   -0.50146621   -0.40311623\nC      -1.15623581    0.17680863   -0.62774318\nH       1.95039654    0.60666728    0.25720629\nH       1.42539263   -0.28843823    1.43446612\nH       1.29356039   -0.91786581   -0.35060552\nH       0.01991384   -1.51818204   -0.68589437\nH      -1.04161429    0.91045481   -1.20867670\nH      -1.89031172   -0.47944239   -0.92619979\nH      -1.32774293    0.66987532    0.21154730',
                      'opt_xyz': 'C       1.26777600   -0.27870900    0.29276500\nC       0.07463000   -0.62022400   -0.53010300\nC      -1.16029300    0.21360700   -0.52169200\nH       1.59040700    0.76699400    0.13219700\nH       1.05951400   -0.36638400    1.37791700\nH       2.12334900   -0.93227100    0.07125200\nH       0.08167100   -1.53252500   -1.13428800\nH      -1.37845000    0.64026200   -1.51890000\nH      -2.05479500   -0.37150500   -0.23997200\nH      -1.07810100    1.05479500    0.18406400'},
                     {'t0': '2022-05-26T23:42:10.729805', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': None, 'index': 19, 'imaginary_freqs': None,
                      'conformer_index': 19, 'successful_irc': None, 'successful_normal_mode': None,
                      'execution_time': '0:00:01.908651',
                      'initial_xyz': 'C      -1.39099145   -0.51042563    0.51720619\nC      -0.23563087    0.22043005   -0.48801798\nC       1.02234614    0.22202557   -0.15813138\nH      -1.66820598   -1.36889911    0.67907137\nH      -2.04905176    0.37884614    0.97399098\nH      -0.64659941    0.19653773    0.31738496\nH      -0.70008379    0.71301955   -1.30898690\nH       1.47499561   -0.65745252   -0.29022256\nH       1.59206700    0.78563404   -0.95010835\nH       1.42435002    0.75069726    0.67192549'},
                     {'t0': '2022-05-26T23:42:12.647837', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'F', 'success': True, 'energy': 180.8588179833023, 'index': 20,
                      'imaginary_freqs': None, 'conformer_index': 20, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.894893',
                      'initial_xyz': 'C       1.00167012    0.51517284   -0.06591963\nC      -0.18289231    0.17261295   -0.82903284\nC      -1.55816829   -0.17413202   -0.13559744\nH       1.52568972    1.36995649    0.10040943\nH       1.80391240   -0.14884450    0.13658467\nH       0.23577479    1.34846580   -0.45981669\nH      -0.07382769   -0.08149067   -1.80809546\nH      -1.44916737   -1.09953022    0.21490166\nH      -2.30217791    0.14279181   -0.75846672\nH      -1.60130811    0.19376379    0.86936384',
                      'opt_xyz': 'C       1.07037200    0.51932700   -0.08930100\nC      -0.17470700    0.14173400   -0.81755900\nC      -1.45969100   -0.06121300   -0.07563200\nH       1.01161600    0.73188100    0.98080900\nH       2.04355700    0.28507500   -0.52448500\nH       0.35222500    1.32806600   -0.82117000\nH      -0.03975400   -0.35171200   -1.78471400\nH      -1.47658900   -1.04259500    0.43586800\nH      -2.33157300   -0.01684200   -0.74442900\nH      -1.59595100    0.70504400    0.70494300'},
                     {'t0': '2022-05-26T23:42:14.548687', 'method': 'gcn', 'method_index': None,
                      'method_direction': 'R', 'success': True, 'energy': 180.85891512676608, 'index': 21,
                      'imaginary_freqs': None, 'conformer_index': 21, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:01.856999',
                      'initial_xyz': 'C      -1.29100406   -0.56441748   -0.56283945\nC      -0.69772559    0.45237696    0.12930320\nC       0.48780710    0.53718090    1.09389055\nH      -2.13487411   -1.01646090   -0.37586549\nH      -0.74074459   -1.03662610   -1.36385643\nH      -0.88092804    0.78589129   -0.95387864\nH      -1.42911565    1.25710464    0.56285071\nH       1.13139296    1.01143909    0.52348405\nH       1.22643602    0.13242149    1.36027193\nH       0.07100621    0.09331925    1.99217510',
                      'opt_xyz': 'C      -1.28361900   -0.58490300   -0.52884500\nC      -0.77840100    0.59669300    0.22753600\nC       0.50959100    0.51909800    0.98833400\nH      -2.35094600   -0.67802000   -0.73703700\nH      -0.65628400   -1.47706100   -0.59368100\nH      -0.73787900    0.47431800   -1.06384700\nH      -1.52697700    1.33501600    0.52991200\nH       0.91155600    1.51658700    1.21846100\nH       1.27761800   -0.02903900    0.41908400\nH       0.37758900   -0.02045900    1.94561900'},
                     {'t0': '2022-05-26T23:42:16.557742', 'method': 'kinbot', 'method_index': 0,
                      'method_direction': 'F', 'success': True, 'energy': 17.085965428326745, 'index': 22,
                      'imaginary_freqs': [-90.065], 'conformer_index': 22, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:00.006560',
                      'initial_xyz': 'C       1.39393700    0.26537900   -0.20838600\nC       0.19342400   -0.33106400    0.44496000\nC      -0.84902400    0.70694400    0.88292100\nH       2.32817400   -0.28300300   -0.27001000\nH       1.31393200    1.18780400   -0.77595800\nH      -0.29199000   -1.04919300   -0.24230300\nH       0.50265500   -0.93665200    1.30845500\nH      -1.19276500    1.30046200    0.02844400\nH      -1.72540600    0.22697800    1.32987900\nH      -0.42854900    1.39899700    1.61954600',
                      'opt_xyz': 'C       1.37026500    0.21375700   -0.27043800\nC       0.15134000   -0.35818800    0.36799100\nC      -0.80019700    0.70757700    0.93794300\nH       2.33530800    0.22065000    0.24218900\nH       1.29190400    0.76724000   -1.21086100\nH      -0.40898300   -0.95723900   -0.37328200\nH       0.43817300   -1.05326200    1.17318700\nH      -1.13112000    1.40316500    0.15146200\nH      -1.69586400    0.24180000    1.37707100\nH      -0.30643800    1.30115200    1.72228500'},
                     {'t0': '2022-05-26T23:42:16.606748', 'method': 'kinbot', 'method_index': 1,
                      'method_direction': 'R', 'success': True, 'energy': 6.82630343362689e-05, 'index': 23,
                      'imaginary_freqs': None, 'conformer_index': 23, 'successful_irc': None,
                      'successful_normal_mode': None, 'execution_time': '0:00:00.005740',
                      'initial_xyz': 'C       1.30445700    0.14930600    0.33486100\nC      -0.03852500   -0.17292600    0.89922000\nC      -1.27617300    0.46478200    0.36303900\nH       1.43314300    1.23075500    0.19226600\nH       2.11562900   -0.20650200    0.97825700\nH       1.46118100   -0.30946900   -0.65803600\nH      -0.13925700   -1.05320700    1.52930100\nH      -1.56011100    0.05987600   -0.62504600\nH      -2.13512900    0.31314300    1.02467200\nH      -1.14362100    1.54575800    0.22040200',
                      'opt_xyz': 'C       1.31569700    0.21321300    0.38752400\nC      -0.03847200   -0.25392000    0.79404500\nC      -1.28436600    0.38955400    0.29002800\nH       1.53200500    1.23187800    0.76714600\nH       2.11024900   -0.44892600    0.75976300\nH       1.41460600    0.27511900   -0.71237600\nH      -0.13099100   -1.07201200    1.51484000\nH      -1.90661200   -0.31435900   -0.29433400\nH      -1.92744900    0.75124200    1.11310300\nH      -1.06307300    1.24972500   -0.36080500'}],
                            'ts_conf_spawned': True, 'ts_guesses_exhausted': False, 'ts_number': 0, 'ts_report': '',
                            'rxn_label': 'nC3H7 <=> iC3H7', 'rxn_index': 0,
                            'successful_methods': ['autotst', 'autotst', 'autotst', 'autotst', 'gcn', 'gcn', 'gcn',
                                                   'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'gcn',
                                                   'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'kinbot', 'kinbot'],
                            'unsuccessful_methods': [], 'chosen_ts_method': 'gcn', 'chosen_ts': 4,
                            'rxn_zone_atom_indices': None, 'chosen_ts_list': [18, 23, 15, 22, 7, 5, 4],
                            'ts_checks': {'E0': None, 'e_elect': True, 'IRC': None, 'freq': True,
                                          'NMD': None, 'warnings': ''},
                            'e_elect': -310902.61556421133, 'tsg_spawned': True, 'opt_level': 'b3lyp/6-31g(d,p)',
                            'bond_corrections': {}, 'mol': {'atoms': [
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26582, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-26581: 1.0, -26579: 1.0, -26578: 1.0, -26577: 1.0}},
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 1, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26581, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-26582: 1.0, -26580: 1.0, -26576: 1.0}},
                     {'element': {'number': 6, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26580, 'props': {'inRing': False}, 'atomtype': 'Cs',
                      'edges': {-26581: 1.0, -26575: 1.0, -26574: 1.0, -26573: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26579, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26582: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26578, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26582: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26577, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26582: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26576, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26581: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26575, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26580: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26574, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26580: 1.0}},
                     {'element': {'number': 1, 'isotope': -1}, 'radical_electrons': 0, 'charge': 0, 'label': '',
                      'lone_pairs': 0, 'id': -26573, 'props': {'inRing': False}, 'atomtype': 'H',
                      'edges': {-26580: 1.0}}], 'multiplicity': 2, 'props': {},
                     'atom_order': [-26582, -26581, -26580, -26579, -26578, -26577, -26576, -26575, -26574, -26573]},
                            'initial_xyz': 'C      -0.94073000   -1.08641400   -0.13521400\nC      -0.33120900    0.27400500   -0.10738200\nC       1.03297600    0.48550500    0.47374300\nH      -0.34195200   -1.93903700    0.19367700\nH      -2.02636000   -1.19201700   -0.17452300\nH      -0.43853800   -0.46541200   -1.16876000\nH      -1.02644700    1.11801000   -0.14324700\nH       1.47599100    1.43717700    0.14587900\nH       1.72184900   -0.32388500    0.18160800\nH       1.00079200    0.49179200    1.58006000',
                            'final_xyz': 'C      -0.94403900   -1.08919600   -0.13528100\nC      -0.33161500    0.27504900   -0.10775400\nC       1.03465100    0.48772700    0.47568400\nH      -0.35007600   -1.93768200    0.18473700\nH      -2.02160800   -1.19512500   -0.17500700\nH      -0.44246400   -0.46665500   -1.16369500\nH      -1.02252600    1.11305500   -0.14369600\nH       1.47717100    1.43311000    0.14664900\nH       1.72023500   -0.31816900    0.18700200\nH       1.00664300    0.49760900    1.57720100',
                            'checkfile': '/storage/ce_dana/alongd/runs/ARC/debug13/calcs/TSs/TS0/opt_a24061/check.chk'},
             'done_opt_r_n_p': True, 'family': 'intra_H_migration', 'family_own_reverse': True, 'ts_label': 'TS0'}
        rxn = ARCReaction(reaction_dict=rxn_dict)
        output = {'nC3H7': {'paths': {'geo': os.path.join(ARC_TESTING_PATH, 'opt', 'nC3H7.out'),
                                      'freq': os.path.join(ARC_TESTING_PATH, 'freq', 'nC3H7.out'),
                                      'sp': os.path.join(ARC_TESTING_PATH, 'opt', 'nC3H7.out'),
                                      'composite': ''},
                            'restart': '', 'convergence': True,
                            'job_types': {'conf_opt': True, 'conf_sp': False, 'opt': True, 'freq': True, 'sp': True, 'rotors': True, 'irc': True, 'fine': True},
                            },
                  'iC3H7': {'paths': {'geo': os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out'),
                                      'freq': os.path.join(ARC_TESTING_PATH, 'freq', 'iC3H7.out'),
                                      'sp': os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out'),
                                      'composite': ''},
                            'restart': '', 'convergence': True,
                            'job_types': {'conf_opt': True, 'conf_sp': False, 'opt': True, 'freq': True, 'sp': True, 'rotors': True, 'irc': True, 'fine': True},
                            },
                  'TS0': {'paths': {'geo': os.path.join(ARC_TESTING_PATH, 'opt', 'TS_nC3H7-iC3H7.out'),
                                    'freq': os.path.join(ARC_TESTING_PATH, 'freq', 'TS_nC3H7-iC3H7.out'),
                                    'sp': os.path.join(ARC_TESTING_PATH, 'opt', 'TS_nC3H7-iC3H7.out'),
                                    'composite': ''},
                          'restart': '', 'convergence': True,
                          'job_types': {'conf_opt': True, 'conf_sp': False, 'opt': True, 'freq': True, 'sp': True, 'rotors': True, 'irc': True, 'fine': True},
                            },
                  }
        project_directory = self.make_project_directory('arc_project_for_testing_delete_after_usage6')
        os.makedirs(os.path.join(project_directory, 'output', 'Species', 'nC3H7', 'geometry'), exist_ok=True)
        os.makedirs(os.path.join(project_directory, 'output', 'Species', 'iC3H7', 'geometry'), exist_ok=True)
        os.makedirs(os.path.join(project_directory, 'output', 'rxns', 'TS0', 'geometry'), exist_ok=True)
        shutil.copy(src=os.path.join(ARC_TESTING_PATH, 'freq', 'nC3H7.out'),
                    dst=os.path.join(project_directory, 'output', 'Species', 'nC3H7', 'geometry', 'freq.out'))
        shutil.copy(src=os.path.join(ARC_TESTING_PATH, 'freq', 'iC3H7.out'),
                    dst=os.path.join(project_directory, 'output', 'Species', 'iC3H7', 'geometry', 'freq.out'))
        shutil.copy(src=os.path.join(ARC_TESTING_PATH, 'freq', 'TS_nC3H7-iC3H7.out'),
                    dst=os.path.join(project_directory, 'output', 'rxns', 'TS0', 'geometry', 'freq.out'))
        sched = Scheduler(project='test_rxn_e0_check',
                          ess_settings=self.ess_settings,
                          project_directory=project_directory,
                          rxn_list=[rxn],
                          species_list=rxn.r_species + rxn.p_species + [rxn.ts_species],
                          kinetics_adapter='arkane',
                          freq_scale_factor=1.0,
                          sp_level=Level(repr='B3LYP/6-31G(d,p)'),
                          job_types=initialize_job_types(),
                          restart_dict={'output': output},
                          )
        self.assertEqual(rxn.ts_species.ts_checks,
                         {'E0': None, 'e_elect': True, 'IRC': None, 'freq': True, 'NMD': None, 'warnings': ''})

        job_1 = job_factory(job_adapter='gaussian',
                            species=[ARCSpecies(label='SPC', smiles='C')],
                            job_type='freq',
                            level=Level(repr='B3LYP/6-31G(d,p)'),
                            project='test_project',
                            project_directory=project_directory,
                            )
        job_1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'TS_nC3H7-iC3H7.out')
        check_ts(reaction=rxn, verbose=True, job=job_1, checks=['NMD'])
        self.assertEqual(rxn.ts_species.ts_checks, {'E0': None, 'e_elect': True, 'IRC': None, 'freq': True, 'NMD': True, 'warnings': ''})

    def test_save_e_elect(self):
        """Test the save_e_elect() method."""
        project_directory = self.make_project_directory('save_e_elect')
        e_elect_summary_path = os.path.join(project_directory, 'output', 'e_elect_summary.yml')
        sched = Scheduler(project='test_save_e_elect',
                          ess_settings=self.ess_settings,
                          project_directory=project_directory,
                          species_list=[ARCSpecies(label='formaldehyde', smiles='C=O'),
                                        ARCSpecies(label='mehylamine', smiles='CN')],
                          freq_scale_factor=1.0,
                          opt_level=Level(method='B3LYP', basis='6-31G(d,p)', software='gaussian'),
                          sp_level=Level(method='B3LYP', basis='6-31G(d,p)', software='gaussian'),
                          job_types={'opt': True, 'fine_grid': False, 'freq': False, 'sp': True, 'rotors': False,
                                     'conf_opt': False, 'conf_sp': False, 'irc': False},
                          report_e_elect=True,
                          testing=True,
                          )
        sched.post_sp_actions(label='formaldehyde',
                              sp_path=os.path.join(ARC_TESTING_PATH, 'sp', 'formaldehyde_sp_terachem_output.out'))
        self.assertTrue(os.path.isfile(e_elect_summary_path))
        content = read_yaml_file(e_elect_summary_path)
        self.assertEqual(content, {'formaldehyde': -300621.95378630824})

        sched.post_sp_actions(label='mehylamine',
                              sp_path=os.path.join(ARC_TESTING_PATH, 'sp', 'mehylamine_CCSD(T).out'))
        content = read_yaml_file(e_elect_summary_path)
        self.assertEqual(content, {'formaldehyde': -300621.95378630824, 'mehylamine': -251360.00924747565})

    def test_species_has_geo_sp_freq(self):
        """Test the species_has_geo() / species_has_sp() / species_has_freq() functions."""
        for property_, species_has_property in zip(['geo', 'sp', 'freq'], [species_has_geo, species_has_sp, species_has_freq]):
            species_output_dict = {'paths': {property_: False, 'composite': False}}
            self.assertFalse(species_has_property((species_output_dict)))
            species_output_dict = {'paths': {property_: True, 'composite': False}}
            self.assertTrue(species_has_property((species_output_dict)))
            species_output_dict = {'paths': {property_: False, 'composite': True}}
            self.assertTrue(species_has_property((species_output_dict)))
            species_output_dict = {'paths': {property_: True, 'composite': True}}
            self.assertTrue(species_has_property((species_output_dict)))
        yml_path=os.path.join(ARC_TESTING_PATH, 'yml_testing', 'N4H6.yml')
        species_output_dict = {'paths': {'geo': False, 'sp': False, 'freq': False, 'composite': False}}
        self.assertTrue(species_has_freq(species_output_dict=species_output_dict, yml_path=yml_path))
        self.assertTrue(species_has_geo(species_output_dict=species_output_dict, yml_path=yml_path))
        self.assertTrue(species_has_sp(species_output_dict=species_output_dict, yml_path=yml_path))
        self.assertTrue(species_has_sp_and_freq(species_output_dict=species_output_dict, yml_path=yml_path))

    def test_add_label_to_unique_species_labels(self):
        """Test the add_label_to_unique_species_labels() method."""
        self.assertEqual(self.sched2.unique_species_labels, ['methylamine', 'C2H6', 'CtripCO'])
        unique_label = self.sched2.add_label_to_unique_species_labels(label='new_species_15')
        self.assertEqual(unique_label, 'new_species_15')
        self.assertEqual(self.sched2.unique_species_labels, ['methylamine', 'C2H6', 'CtripCO', 'new_species_15'])
        unique_label = self.sched2.add_label_to_unique_species_labels(label='new_species_15')
        self.assertEqual(unique_label, 'new_species_15_0')
        self.assertEqual(self.sched2.unique_species_labels, ['methylamine', 'C2H6', 'CtripCO', 'new_species_15', 'new_species_15_0'])
        unique_label = self.sched2.add_label_to_unique_species_labels(label='new_species_15')
        self.assertEqual(unique_label, 'new_species_15_1')
        self.assertEqual(self.sched2.unique_species_labels, ['methylamine', 'C2H6', 'CtripCO', 'new_species_15', 'new_species_15_0', 'new_species_15_1'])

    def test_troubleshoot_ess_max_attempts(self):
        """Test that troubleshoot_ess respects the max_ess_trsh limit."""
        label = 'methylamine'
        self.sched1.output = dict()
        self.sched1.initialize_output_dict()
        self.assertEqual(self.sched1.output[label]['errors'], '')

        job = job_factory(job_adapter='gaussian', project='project_test', ess_settings=self.ess_settings,
                          species=[self.spc1], xyz=self.spc1.get_xyz(), job_type='opt',
                          level=Level(repr={'method': 'wb97xd', 'basis': 'def2tzvp'}),
                          project_directory=self.project_directory, job_num=200)
        job.ess_trsh_methods = ['trsh_attempt'] * 25

        self.sched1.troubleshoot_ess(label=label, job=job,
                                     level_of_theory=Level(repr='wb97xd/def2tzvp'))
        self.assertIn('ESS troubleshooting attempts exhausted', self.sched1.output[label]['errors'])

    def test_troubleshoot_ess_under_max_attempts(self):
        """Test that troubleshoot_ess does not block when under the max_ess_trsh limit."""
        label = 'methylamine'
        self.sched1.output = dict()
        self.sched1.initialize_output_dict()

        job = job_factory(job_adapter='gaussian', project='project_test', ess_settings=self.ess_settings,
                          species=[self.spc1], xyz=self.spc1.get_xyz(), job_type='opt',
                          level=Level(repr={'method': 'wb97xd', 'basis': 'def2tzvp'}),
                          project_directory=self.project_directory, job_num=201)
        job.ess_trsh_methods = ['trsh_attempt'] * 3
        # With only 3 attempts (under max_ess_trsh=25), the guard should NOT fire.
        # Verify the error message is NOT set (i.e., the guard did not block).
        # We use max_attempts - 1 to test just below the threshold.
        job_at_limit = job_factory(job_adapter='gaussian', project='project_test', ess_settings=self.ess_settings,
                                   species=[self.spc1], xyz=self.spc1.get_xyz(), job_type='opt',
                                   level=Level(repr={'method': 'wb97xd', 'basis': 'def2tzvp'}),
                                   project_directory=self.project_directory, job_num=202)
        job_at_limit.ess_trsh_methods = ['trsh_attempt'] * 24
        self.assertNotIn('ESS troubleshooting attempts exhausted', self.sched1.output[label]['errors'])

    def test_tsg_method_matches_adapter(self):
        """Test matching a TSGuess method string to the TS-search adapter that produced it."""
        self.assertTrue(tsg_method_matches_adapter('xTB-GSM', 'xtb_gsm'))
        self.assertTrue(tsg_method_matches_adapter('KinBot-UMA', 'kinbot'))
        self.assertTrue(tsg_method_matches_adapter('orca_neb', 'orca_neb'))
        self.assertTrue(tsg_method_matches_adapter('qst2', 'qst2'))
        self.assertTrue(tsg_method_matches_adapter('Linear (w=0.50, 0)', 'linear'))
        self.assertFalse(tsg_method_matches_adapter('GCN', 'qst2'))
        self.assertFalse(tsg_method_matches_adapter(None, 'qst2'))
        self.assertFalse(tsg_method_matches_adapter('GCN', None))

    def test_record_tsg_job_error(self):
        """A tsg job number is an adapter number, not a position in ts_guesses.

        Recording the error must never index ts_guesses by the job number: it must annotate the
        guesses that adapter produced, and must not raise when it produced none.
        """
        ts_spc = ARCSpecies(label='TS_tsg_err', is_ts=True, multiplicity=1, charge=0, compute_thermo=False)
        ts_spc.ts_guesses = [TSGuess(index=0, method='GCN', success=False),
                             TSGuess(index=1, method='xTB-GSM', success=False),
                             ]
        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_for_testing_delete_after_usage_tsg_err')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='project_test_tsg_err', ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )
        job = MagicMock()
        job.job_name, job.job_adapter = 'tsg4', 'qst2'
        sched.record_tsg_job_error(label='TS_tsg_err', job=job, output_error='Could not troubleshoot; ')
        self.assertEqual([tsg.errors for tsg in ts_spc.ts_guesses], ['', ''])

        job.job_name, job.job_adapter = 'tsg2', 'xtb_gsm'
        sched.record_tsg_job_error(label='TS_tsg_err', job=job, output_error='Could not troubleshoot; ')
        self.assertEqual(ts_spc.ts_guesses[0].errors, '')
        self.assertIn('Could not troubleshoot', ts_spc.ts_guesses[1].errors)

        # A guess that did produce a geometry is not the casualty of another adapter's failure, even
        # when adapters have merged their names into it (e.g. 'heuristics and gcn').
        merged = TSGuess(index=2, method='heuristics and gcn', success=True)
        ts_spc.ts_guesses.append(merged)
        job.job_name, job.job_adapter = 'tsg3', 'gcn'
        sched.record_tsg_job_error(label='TS_tsg_err', job=job, output_error='Could not troubleshoot; ')
        self.assertEqual(merged.errors, '')
        self.assertIn('Could not troubleshoot', ts_spc.ts_guesses[0].errors)

    @patch('arc.scheduler.Scheduler.run_job')
    def test_troubleshoot_ess_tsg_job_number_is_not_a_guess_position(self, mock_run_job):
        """A tsg job number exceeding the number of guesses must not raise from troubleshoot_ess().

        The adapter dispatch number in a ``tsg<i>`` job name is not a position in ``ts_guesses``:
        an adapter may contribute several guesses or none, so a high dispatch number with few
        guesses generated so far used to raise IndexError and abort the whole run.
        """
        ts_spc = ARCSpecies(label='TS_trsh_tsg', is_ts=True, multiplicity=1, charge=0, compute_thermo=False)
        ts_spc.ts_guesses = [TSGuess(index=0, method='GCN', success=False),
                             TSGuess(index=1, method='xTB-GSM', success=False),
                             ]
        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_for_testing_delete_after_usage_trsh_tsg')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='project_test_trsh_tsg', ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )
        sched.job_dict['TS_trsh_tsg'] = {'tsg': dict()}
        sched.running_jobs['TS_trsh_tsg'] = list()
        job = MagicMock()
        # Dispatch number 9 with only two guesses: the pre-fix code subscripted ts_guesses[9].
        job.job_name, job.job_adapter, job.job_type = 'tsg9', 'gcn', 'tsg'
        job.job_status = ['done', {'status': 'errored', 'keywords': ['Unknown'], 'error': 'unknown', 'line': ''}]
        job.ess_trsh_methods, job.fine, job.job_memory_gb, job.cpu_cores = list(), False, 8, 4
        job.level = Level(repr=default_levels_of_theory['ts_guesses'])
        job.args = {'keyword': dict(), 'block': dict(), 'trsh': dict()}
        job.times_rerun, job.job_id, job.server = 0, 1, 'server1'
        job.conformer = None
        sched.troubleshoot_ess(label='TS_trsh_tsg', job=job, level_of_theory=job.level)  # must not raise
        # The GCN guess, not a position, carries the error.
        self.assertIn('gcn', ts_spc.ts_guesses[0].method.lower())

    def _make_ts_with_an_exhausted_job(self, label: str, job_type: str):
        """
        Build a TS with a chosen guess and one rotor, and a Gaussian job of ``job_type`` for that TS whose
        job-level troubleshooting is exhausted. The job status is parsed from a real Gaussian TS rotor scan
        that ended with "Number of steps exceeded" after every method trsh_ess_job offers for it was already used.

        Returns:
            tuple: The scheduler and the job.
        """
        ts_xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")
        ts_spc = ARCSpecies(label=label, is_ts=True, xyz=ts_xyz, multiplicity=1, charge=0, compute_thermo=False)
        ts_spc.ts_guesses = [TSGuess(index=0, method='heuristics', success=True, energy=100.0, xyz=ts_xyz,
                                     execution_time='0:00:01'),
                             TSGuess(index=1, method='heuristics', success=True, energy=110.0, xyz=ts_xyz,
                                     execution_time='0:00:01'),
                             ]
        for tsg in ts_spc.ts_guesses:
            tsg.opt_xyz = ts_xyz
        ts_spc.chosen_ts, ts_spc.chosen_ts_list, ts_spc.ts_guesses_exhausted = 0, [0], False
        ts_spc.final_xyz = ts_xyz
        ts_spc.rotors_dict = {0: {'pivots': [1, 2], 'top': [2, 3], 'scan': [3, 1, 2, 4], 'torsion': [2, 0, 1, 3],
                                  'number_of_running_jobs': 0, 'success': None, 'invalidation_reason': '',
                                  'times_dihedral_set': 0, 'scan_path': '', 'trsh_counter': 0, 'trsh_methods': list(),
                                  'symmetry': None, 'dimensions': 1, 'original_dihedrals': list(),
                                  'cont_indices': list(), 'directed_scan_type': 'ess', 'directed_scan': dict()}}
        ts_spc.number_of_rotors = 1
        project_directory = os.path.join(ARC_PATH, 'Projects', f'arc_project_for_testing_delete_after_usage_{label}')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project=f'project_test_{label}', ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types2,  # rotors=True
                          )
        sched.job_dict[label] = {'opt': dict(), 'freq': dict(), 'scan': dict()}
        sched.running_jobs[label] = list()
        status, keywords, error, line = determine_ess_status(
            output_path=os.path.join(ARC_TESTING_PATH, 'trsh', 'gaussian', 'ts_scan_maxsteps.out'),
            species_label=label, job_type=job_type, software='gaussian')
        job = MagicMock()
        job.job_name, job.job_adapter, job.job_type = f'{job_type}_a100', 'gaussian', job_type
        job.job_status = ['done', {'status': status, 'keywords': keywords, 'error': error, 'line': line}]
        job.ess_trsh_methods = ['opt=(maxcycle=200)', 'opt=(RFO)', 'opt=(GDIIS)', 'opt=(GEDIIS)', 'int=(Acc2E=14)']
        job.fine, job.job_memory_gb, job.cpu_cores, job.checkfile = False, 8, 4, None
        job.level = Level(repr=default_levels_of_theory['scan' if job_type == 'scan' else 'opt'])
        job.times_rerun, job.job_id, job.server, job.conformer = 0, 1, 'server1', None
        job.rotor_index = 0 if job_type == 'scan' else None
        job.torsions = [ts_spc.rotors_dict[0]['torsion']] if job_type == 'scan' else None
        job.dihedrals, job.directed_scan_type = None, 'ess' if job_type == 'scan' else None
        return sched, job

    @patch('arc.scheduler.Scheduler.run_job')
    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_troubleshoot_ess_ts_scan_exhausted_leaves_the_rotor_harmonic(self, mock_switch_ts, mock_run_job):
        """A TS rotor scan whose job-level troubleshooting ran out fails that rotor and keeps the TS guess.

        A crashed scan says nothing about whether the saddle point is right. Switching the guess discards a TS
        that may have passed freq, IRC and NMD, and empties rotors_dict so every torsion is scanned again.
        """
        label = 'TS_scan_crash'
        sched, job = self._make_ts_with_an_exhausted_job(label=label, job_type='scan')
        self.assertEqual(job.job_status[1]['line'], 'Number of steps exceeded')
        sched.troubleshoot_ess(label=label, job=job, level_of_theory=job.level)
        mock_switch_ts.assert_not_called()
        mock_run_job.assert_not_called()
        rotors_dict = sched.species_dict[label].rotors_dict
        self.assertEqual(list(rotors_dict.keys()), [0])
        self.assertIs(rotors_dict[0]['success'], False)
        reason = rotors_dict[0]['invalidation_reason']
        self.assertIn('scan job', reason.lower())
        self.assertIn(job.job_name, reason)
        self.assertIn('Maximum optimization cycles reached', reason)
        self.assertEqual(sched.species_dict[label].chosen_ts, 0)

    @patch('arc.scheduler.Scheduler.run_job')
    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_troubleshoot_ess_ts_opt_exhausted_still_switches_the_guess(self, mock_switch_ts, mock_run_job):
        """A TS opt job whose job-level troubleshooting ran out still switches the TS guess."""
        label = 'TS_opt_crash'
        sched, job = self._make_ts_with_an_exhausted_job(label=label, job_type='opt')
        sched.troubleshoot_ess(label=label, job=job, level_of_theory=job.level)
        mock_switch_ts.assert_called_once_with(label=label)
        mock_run_job.assert_not_called()
        self.assertIsNone(sched.species_dict[label].rotors_dict[0]['success'])
        self.assertEqual(sched.species_dict[label].rotors_dict[0]['invalidation_reason'], '')

    @patch('arc.scheduler.Scheduler.run_job')
    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_ts_scan_exhausted_rotor_reaches_the_output_harmonic_fallback(self, mock_switch_ts, mock_run_job):
        """The rotor of a TS scan that could not be troubleshot is listed in output.yml as a failed scan."""
        label = 'TS_scan_crash_out'
        sched, job = self._make_ts_with_an_exhausted_job(label=label, job_type='scan')
        sched.troubleshoot_ess(label=label, job=job, level_of_theory=job.level)
        spc = sched.species_dict[label]
        fallback = _statmech_to_dict(spc, sched.project_directory)['harmonic_fallback_torsions']
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0]['pivot_atoms'], [1, 2])
        self.assertEqual(fallback[0]['cause'], ROTOR_FALLBACK_SCAN_FAILED)
        self.assertIn(job.job_name, fallback[0]['reason'])
        self.assertIn('Maximum optimization cycles reached', fallback[0]['reason'])

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_run_ts_conformer_jobs_single_success_provenance(self, mock_run_opt):
        """The provenance of a lone successful guess must describe that guess, not ts_guesses[0]."""
        ts_xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")
        ts_spc = ARCSpecies(label='TS_single', is_ts=True, multiplicity=1, charge=0, compute_thermo=False)
        failed = TSGuess(index=0, method='qst2', success=False)
        good = TSGuess(index=1, method='xTB-GSM', success=True, xyz=ts_xyz)
        ts_spc.ts_guesses = [failed, good]
        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_for_testing_delete_after_usage_tsg_single')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        good.log_path = os.path.join(project_directory, 'stringfile.xyz0000')
        sched = Scheduler(project='project_test_tsg_single', ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )
        sched.job_dict['TS_single'] = dict()
        sched.run_ts_conformer_jobs(label='TS_single')
        self.assertEqual(ts_spc.chosen_ts_method, 'xtb-gsm')
        self.assertEqual(ts_spc.successful_methods, ['xtb-gsm'])
        self.assertEqual(good.energy, 0.0)
        self.assertIsNone(failed.energy)
        self.assertEqual(sched.output['TS_single']['paths']['gsm'], good.log_path)
        self.assertEqual(sched.output['TS_single']['paths']['neb'], '')

    @patch('arc.scheduler.Scheduler.run_job')
    def test_run_ts_conformer_jobs_drops_orphaned_conf_opt_running_jobs(self, mock_run_job):
        """Respawning the TS conformers cancels the superseded conf_opt jobs and stops tracking them, piped or not."""
        xyz_a = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")
        xyz_b = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       2.51402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")
        label = 'TS_respawn'
        ts_spc = ARCSpecies(label=label, is_ts=True, multiplicity=1, charge=0, compute_thermo=False)
        ts_spc.ts_guesses = [TSGuess(index=0, method='heuristics', success=True, xyz=xyz_a),
                             TSGuess(index=1, method='heuristics', success=True, xyz=xyz_b)]
        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_for_testing_delete_after_usage_ts_respawn')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='project_test_ts_respawn', ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )
        for piped_indices in [set(), {1}]:
            with self.subTest(piped_indices=piped_indices):
                sched.pipe_planner = MagicMock()
                sched.pipe_planner.try_pipe_ts_opt.return_value = piped_indices
                superseded = {0: MagicMock(), 1: MagicMock()}
                sched.job_dict[label] = {'conf_opt': dict(superseded)}
                sched.running_jobs[label] = ['conf_opt_0', 'conf_opt_1', 'freq_a7']
                mock_run_job.reset_mock()
                sched.run_ts_conformer_jobs(label=label)
                self.assertEqual(sched.running_jobs[label], ['freq_a7'])
                self.assertEqual(mock_run_job.call_count, 2 - len(piped_indices))
                for job in superseded.values():
                    job.delete.assert_called_once()
                self.assertFalse(any(job in superseded.values()
                                     for job in sched.job_dict[label].get('conf_opt', dict()).values()))

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_switch_ts_cleanup(self, mock_run_opt):
        """Test that switch_ts resets job_types, convergence, cleans up IRC species, and clears pending pipes."""
        ts_xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")

        ts_spc = ARCSpecies(label='TS_test', is_ts=True, xyz=ts_xyz, multiplicity=1, charge=0,
                            compute_thermo=False)
        # Create two TSGuess objects so determine_most_likely_ts_conformer can pick the 2nd after the 1st fails.
        ts_spc.ts_guesses = [
            TSGuess(index=0, method='heuristics', success=True, energy=100.0, xyz=ts_xyz,
                    execution_time='0:00:01'),
            TSGuess(index=1, method='heuristics', success=True, energy=110.0, xyz=ts_xyz,
                    execution_time='0:00:01'),
        ]
        ts_spc.ts_guesses[0].opt_xyz = ts_xyz
        ts_spc.ts_guesses[0].imaginary_freqs = [-500.0]
        ts_spc.ts_guesses[1].opt_xyz = ts_xyz
        ts_spc.ts_guesses[1].imaginary_freqs = [-400.0]
        # Simulate guess 0 already tried.
        ts_spc.chosen_ts = 0
        ts_spc.chosen_ts_list = [0]
        ts_spc.ts_guesses_exhausted = False

        project_directory = os.path.join(ARC_PATH, 'Projects',
                                         'arc_project_for_testing_delete_after_usage4')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='test_switch_ts', ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )

        ts_label = 'TS_test'
        # Simulate state after guess 0 completed: freq/sp/opt marked done.
        sched.output[ts_label]['job_types']['opt'] = True
        sched.output[ts_label]['job_types']['freq'] = True
        sched.output[ts_label]['job_types']['sp'] = True
        sched.output[ts_label]['convergence'] = True
        sched.job_dict[ts_label] = {'opt': {}, 'freq': {}, 'sp': {}}
        sched.running_jobs[ts_label] = []

        # Simulate IRC species spawned from guess 0.
        irc_label_1 = 'IRC_TS_test_1'
        irc_label_2 = 'IRC_TS_test_2'
        irc_spc_1 = ARCSpecies(label=irc_label_1, xyz=ts_xyz, compute_thermo=False,
                                irc_label=ts_label)
        irc_spc_2 = ARCSpecies(label=irc_label_2, xyz=ts_xyz, compute_thermo=False,
                                irc_label=ts_label)
        ts_spc.irc_label = f'{irc_label_1} {irc_label_2}'
        sched.species_dict[irc_label_1] = irc_spc_1
        sched.species_dict[irc_label_2] = irc_spc_2
        sched.species_list.extend([irc_spc_1, irc_spc_2])
        sched.unique_species_labels.extend([irc_label_1, irc_label_2])
        sched.running_jobs[irc_label_1] = ['opt_a100']
        sched.running_jobs[irc_label_2] = ['opt_a101']
        sched.job_dict[irc_label_1] = {'opt': {}}
        sched.job_dict[irc_label_2] = {'opt': {}}
        sched.initialize_output_dict(label=irc_label_1)
        sched.initialize_output_dict(label=irc_label_2)

        # Simulate pending pipe entries from the old guess.
        sched._pending_pipe_sp.add(ts_label)
        sched._pending_pipe_freq.add(ts_label)
        sched._pending_pipe_irc.add((ts_label, 'forward'))
        sched._pending_pipe_irc.add((ts_label, 'reverse'))

        # Call switch_ts, should pick guess 1 and clean up all state from guess 0.
        sched.switch_ts(ts_label)

        # Verify guess 1 was selected.
        self.assertEqual(sched.species_dict[ts_label].chosen_ts, 1)
        self.assertIn(1, sched.species_dict[ts_label].chosen_ts_list)

        # Verify IRC species from guess 0 fully removed.
        self.assertNotIn(irc_label_1, sched.species_dict)
        self.assertNotIn(irc_label_2, sched.species_dict)
        self.assertNotIn(irc_label_1, sched.running_jobs)
        self.assertNotIn(irc_label_2, sched.running_jobs)
        self.assertNotIn(irc_label_1, sched.job_dict)
        self.assertNotIn(irc_label_2, sched.job_dict)
        self.assertNotIn(irc_label_1, sched.output)
        self.assertNotIn(irc_label_2, sched.output)
        self.assertNotIn(irc_label_1, sched.unique_species_labels)
        self.assertNotIn(irc_label_2, sched.unique_species_labels)
        self.assertIsNone(sched.species_dict[ts_label].irc_label)

        # Verify job_types reset and convergence cleared.
        self.assertFalse(sched.output[ts_label]['job_types']['opt'])
        self.assertFalse(sched.output[ts_label]['job_types']['freq'])
        self.assertFalse(sched.output[ts_label]['job_types']['sp'])
        self.assertIsNone(sched.output[ts_label]['convergence'])

        # Verify pending pipe entries cleared.
        self.assertNotIn(ts_label, sched._pending_pipe_sp)
        self.assertNotIn(ts_label, sched._pending_pipe_freq)
        self.assertNotIn((ts_label, 'forward'), sched._pending_pipe_irc)
        self.assertNotIn((ts_label, 'reverse'), sched._pending_pipe_irc)

        # Verify ts_checks were reset.
        self.assertIsNone(sched.species_dict[ts_label].ts_checks['freq'])
        self.assertIsNone(sched.species_dict[ts_label].ts_checks['NMD'])
        self.assertIsNone(sched.species_dict[ts_label].ts_checks['E0'])

        # Verify rotors convergence flag preserved as True (not blanket-reset to False).
        self.assertTrue(sched.output[ts_label]['job_types']['rotors'])

    def test_deleting_a_ts_irc_species_keeps_the_shared_species_list(self):
        """Test that cleaning up a TS's IRC species edits the species list ARC shares, rather than replacing it"""
        ts_xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")
        spc = ARCSpecies(label='NH3', smiles='N', compute_thermo=False)
        ts_spc = ARCSpecies(label='TS_shared', is_ts=True, xyz=ts_xyz, multiplicity=1, charge=0, compute_thermo=False)
        species = [spc, ts_spc]
        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_for_testing_delete_after_usage_shared_list')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='test_shared_species_list', ess_settings=self.ess_settings,
                          species_list=species,
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )
        self.assertIs(sched.species_list, species)
        irc_spc = ARCSpecies(label='IRC_TS_shared_1', xyz=ts_xyz, compute_thermo=False, irc_label='TS_shared')
        sched.species_list.append(irc_spc)
        sched.species_dict[irc_spc.label] = irc_spc
        sched.unique_species_labels.append(irc_spc.label)
        sched.initialize_output_dict(label=irc_spc.label)
        ts_spc.irc_label = irc_spc.label
        sched.job_dict['TS_shared'] = dict()

        sched.delete_all_species_jobs('TS_shared')

        self.assertIs(sched.species_list, species)
        self.assertEqual([s.label for s in species], ['NH3', 'TS_shared'])
        self.assertTrue(all(s.label in sched.output for s in species))

        later_spc = ARCSpecies(label='IRC_TS_shared_2', xyz=ts_xyz, compute_thermo=False, irc_label='TS_shared')
        sched.species_list.append(later_spc)
        self.assertIn(later_spc, species)

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_switch_ts_rotors_reset(self, mock_run_opt):
        """Test that switch_ts resets rotors_dict when rotors are enabled, and preserves the None sentinel."""
        ts_xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")

        ts_spc = ARCSpecies(label='TS_rot', is_ts=True, xyz=ts_xyz, multiplicity=1, charge=0,
                            compute_thermo=False)
        ts_spc.ts_guesses = [
            TSGuess(index=0, method='heuristics', success=True, energy=100.0, xyz=ts_xyz,
                    execution_time='0:00:01'),
            TSGuess(index=1, method='heuristics', success=True, energy=110.0, xyz=ts_xyz,
                    execution_time='0:00:01'),
        ]
        ts_spc.ts_guesses[0].opt_xyz = ts_xyz
        ts_spc.ts_guesses[0].imaginary_freqs = [-500.0]
        ts_spc.ts_guesses[1].opt_xyz = ts_xyz
        ts_spc.ts_guesses[1].imaginary_freqs = [-400.0]
        ts_spc.chosen_ts = 0
        ts_spc.chosen_ts_list = [0]
        ts_spc.ts_guesses_exhausted = False
        # Simulate stale rotors from previous guess.
        ts_spc.rotors_dict = {0: {'pivots': [1, 2], 'scan_path': '', 'success': True}}
        ts_spc.number_of_rotors = 1

        project_directory = os.path.join(ARC_PATH, 'Projects',
                                         'arc_project_for_testing_delete_after_usage5')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='test_switch_ts_rot', ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types2,  # rotors=True
                          )

        ts_label = 'TS_rot'
        sched.output[ts_label]['job_types']['opt'] = True
        sched.output[ts_label]['job_types']['freq'] = True
        sched.job_dict[ts_label] = {'opt': {}, 'freq': {}, 'sp': {}}
        sched.running_jobs[ts_label] = []

        sched.switch_ts(ts_label)

        # rotors_dict should be reset so determine_rotors re-runs for the new geometry.
        self.assertEqual(sched.species_dict[ts_label].rotors_dict, {})
        self.assertEqual(sched.species_dict[ts_label].number_of_rotors, 0)

        # Now test that rotors_dict=None sentinel is preserved (species marked to skip rotors).
        ts_spc2 = ARCSpecies(label='TS_norot', is_ts=True, xyz=ts_xyz, multiplicity=1, charge=0,
                             compute_thermo=False)
        ts_spc2.ts_guesses = [
            TSGuess(index=0, method='heuristics', success=True, energy=100.0, xyz=ts_xyz,
                    execution_time='0:00:01'),
            TSGuess(index=1, method='heuristics', success=True, energy=110.0, xyz=ts_xyz,
                    execution_time='0:00:01'),
        ]
        ts_spc2.ts_guesses[0].opt_xyz = ts_xyz
        ts_spc2.ts_guesses[0].imaginary_freqs = [-500.0]
        ts_spc2.ts_guesses[1].opt_xyz = ts_xyz
        ts_spc2.ts_guesses[1].imaginary_freqs = [-400.0]
        ts_spc2.chosen_ts = 0
        ts_spc2.chosen_ts_list = [0]
        ts_spc2.ts_guesses_exhausted = False
        ts_spc2.rotors_dict = None  # Sentinel: skip rotor scans.

        project_directory2 = os.path.join(ARC_PATH, 'Projects',
                                          'arc_project_for_testing_delete_after_usage6')
        self.addCleanup(shutil.rmtree, project_directory2, ignore_errors=True)
        sched2 = Scheduler(project='test_switch_ts_norot', ess_settings=self.ess_settings,
                           species_list=[ts_spc2],
                           opt_level=Level(repr=default_levels_of_theory['opt']),
                           freq_level=Level(repr=default_levels_of_theory['freq']),
                           sp_level=Level(repr=default_levels_of_theory['sp']),
                           ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                           project_directory=project_directory2,
                           testing=True,
                           job_types=self.job_types2,  # rotors=True
                           )

        ts_label2 = 'TS_norot'
        sched2.output[ts_label2]['job_types']['opt'] = True
        sched2.job_dict[ts_label2] = {'opt': {}, 'freq': {}, 'sp': {}}
        sched2.running_jobs[ts_label2] = []

        sched2.switch_ts(ts_label2)

        # rotors_dict=None must be preserved — do not re-enable rotor scans.
        self.assertIsNone(sched2.species_dict[ts_label2].rotors_dict)

    def setup_ts_scheduler_for_freq_check(self, project, chosen_ts, chosen_ts_list=None):
        """
        Set up a Scheduler with a single TS species whose TSGuess ``index`` (identity) and
        ``conformer_index`` (position among the successful guesses) deliberately diverge.

        Args:
            project (str): The project name.
            chosen_ts (int): The value to assign to the TS species ``chosen_ts`` attribute.
            chosen_ts_list (list, optional): The value to assign to the ``chosen_ts_list`` attribute.

        Returns:
            tuple: The Scheduler instance, the TS species label, and the list of TSGuess objects.
        """
        ts_xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")
        ts_label = 'TS_freq_identity'
        ts_spc = ARCSpecies(label=ts_label, is_ts=True, xyz=ts_xyz, multiplicity=1, charge=0, compute_thermo=False)
        tsg_failed = TSGuess(index=0, method='autotst', success=False, xyz=ts_xyz, execution_time='0:00:01')
        tsg_a = TSGuess(index=1, method='heuristics', success=True, energy=100.0, xyz=ts_xyz,
                        execution_time='0:00:01')
        tsg_b = TSGuess(index=2, method='gcn', success=True, energy=90.0, xyz=ts_xyz, execution_time='0:00:01')
        for tsg in [tsg_failed, tsg_a, tsg_b]:
            tsg.opt_xyz = ts_xyz
        # Only successful guesses get conformer optimization jobs, so the positions and the identities diverge.
        tsg_a.conformer_index = 0
        tsg_b.conformer_index = 1
        # Deliberately stored so that identity, list position and conformer_index all disagree:
        # index 2 sits at position 0, index 0 at position 1, index 1 at position 2. A lookup that
        # subscripts ts_guesses, or that matches on conformer_index, therefore finds the wrong guess.
        ts_spc.ts_guesses = [tsg_b, tsg_failed, tsg_a]
        ts_spc.chosen_ts = chosen_ts
        ts_spc.chosen_ts_list = chosen_ts_list if chosen_ts_list is not None else list()
        ts_spc.ts_guesses_exhausted = False
        project_directory = os.path.join(ARC_PATH, 'Projects', project)
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project=project, ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )
        sched.job_dict[ts_label] = {'opt': dict(), 'freq': dict(), 'sp': dict()}
        sched.running_jobs[ts_label] = list()
        return sched, ts_label, {0: tsg_failed, 1: tsg_a, 2: tsg_b}

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_check_negative_freq_assigns_freqs_to_the_chosen_ts_guess(self, mock_run_opt):
        """Test that the imaginary frequencies are assigned to the TSGuess identified by ``chosen_ts``."""
        sched, ts_label, tsgs = self.setup_ts_scheduler_for_freq_check(
            project='test_ts_freq_identity_pass', chosen_ts=2)
        job = MagicMock()
        job.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'C2H6_freq_QChem.out')
        freq_ok, switched_ts = sched.check_negative_freq(label=ts_label, job=job, vibfreqs=[-1000.0, 500.0, 1500.0])
        self.assertTrue(freq_ok)
        self.assertFalse(switched_ts)
        self.assertEqual(tsgs[2].imaginary_freqs, [-1000.0])
        self.assertIsNone(tsgs[0].imaginary_freqs)
        self.assertIsNone(tsgs[1].imaginary_freqs)
        self.assertTrue(sched.species_dict[ts_label].ts_checks['freq'])
        self.assertTrue(sched.output[ts_label]['job_types']['freq'])

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_check_negative_freq_rejects_two_imaginary_freqs(self, mock_run_opt):
        """Test that a TS with two imaginary frequencies is rejected when index and conformer_index diverge."""
        sched, ts_label, tsgs = self.setup_ts_scheduler_for_freq_check(
            project='test_ts_freq_identity_reject', chosen_ts=2, chosen_ts_list=[2])
        job = MagicMock()
        job.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'C2H6_freq_QChem.out')
        freq_ok, switched_ts = sched.check_negative_freq(label=ts_label, job=job,
                                                         vibfreqs=[-1200.0, -800.0, 500.0, 1500.0])
        self.assertFalse(freq_ok)
        self.assertTrue(switched_ts)
        self.assertEqual(tsgs[2].imaginary_freqs, [-1200.0, -800.0])
        self.assertNotEqual(sched.species_dict[ts_label].ts_checks['freq'], True)
        self.assertFalse(sched.output[ts_label]['job_types']['freq'])
        # A different TS guess was selected instead of silently accepting the wrong one.
        self.assertEqual(sched.species_dict[ts_label].chosen_ts, 1)

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_check_negative_freq_no_matching_ts_guess(self, mock_run_opt):
        """Test that a ``chosen_ts`` value matching no TSGuess is logged and is not silently accepted."""
        sched, ts_label, tsgs = self.setup_ts_scheduler_for_freq_check(
            project='test_ts_freq_identity_no_match', chosen_ts=99)
        job = MagicMock()
        job.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'C2H6_freq_QChem.out')
        with self.assertLogs('arc', level='WARNING') as context_manager:
            freq_ok, switched_ts = sched.check_negative_freq(label=ts_label, job=job,
                                                             vibfreqs=[-1000.0, 500.0, 1500.0])
        self.assertFalse(freq_ok)
        self.assertTrue(switched_ts)
        self.assertTrue(any('99' in message and ts_label in message for message in context_manager.output))
        self.assertTrue(all(tsg.imaginary_freqs is None for tsg in tsgs.values()))
        self.assertNotEqual(sched.species_dict[ts_label].ts_checks['freq'], True)
        self.assertFalse(sched.output[ts_label]['job_types']['freq'])

    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_check_negative_freq_no_chosen_ts_judges_the_freqs_directly(self, mock_switch_ts):
        """Test that a good freq result arriving while no TS guess is selected still converges the TS.

        A repaired restart file resets ``chosen_ts``, and a standalone TS given several xyz guesses
        never sets it, so a freq job can land with no selection. The frequencies cannot be attributed
        to a guess, but they still describe the optimized geometry, so the check is decided on them
        directly. Leaving it unverified would strand the species: for a TS, ``check_freq_job`` neither
        troubleshoots nor respawns, so nothing would ever set ``chosen_ts`` again.
        """
        sched, ts_label, tsgs = self.setup_ts_scheduler_for_freq_check(
            project='test_ts_freq_identity_no_chosen', chosen_ts=None)
        job = MagicMock()
        job.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'C2H6_freq_QChem.out')
        with self.assertLogs('arc', level='WARNING') as context_manager:
            freq_ok, switched_ts = sched.check_negative_freq(label=ts_label, job=job,
                                                             vibfreqs=[-1000.0, 500.0, 1500.0])
        self.assertTrue(freq_ok)
        self.assertFalse(switched_ts)
        self.assertTrue(any(ts_label in message for message in context_manager.output))
        mock_switch_ts.assert_not_called()
        # The verdict is recorded on the species, not attributed to any one guess.
        self.assertTrue(sched.species_dict[ts_label].ts_checks['freq'])
        self.assertTrue(sched.output[ts_label]['job_types']['freq'])
        self.assertTrue(all(tsg.imaginary_freqs is None for tsg in tsgs.values()))

    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_check_negative_freq_no_chosen_ts_does_not_switch_on_a_bad_result(self, mock_switch_ts):
        """Test that a bad freq result with no selected guess fails the check without switching guesses.

        Switching here would discard a geometry that may already have passed every other check,
        without knowing which guess replaces it.
        """
        sched, ts_label, tsgs = self.setup_ts_scheduler_for_freq_check(
            project='test_ts_freq_identity_no_chosen_bad', chosen_ts=None)
        job = MagicMock()
        job.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'C2H6_freq_QChem.out')
        with self.assertLogs('arc', level='WARNING') as context_manager:
            freq_ok, switched_ts = sched.check_negative_freq(label=ts_label, job=job,
                                                             vibfreqs=[-1200.0, -800.0, 500.0, 1500.0])
        self.assertFalse(freq_ok)
        self.assertFalse(switched_ts)
        self.assertTrue(any(ts_label in message for message in context_manager.output))
        mock_switch_ts.assert_not_called()
        self.assertNotEqual(sched.species_dict[ts_label].ts_checks['freq'], True)
        self.assertFalse(sched.output[ts_label]['job_types']['freq'])

    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_check_negative_freq_uses_the_sole_successful_guess_when_none_was_chosen(self, mock_switch_ts):
        """Test the ``chosen_ts is None`` fallback: one successful guess is the chosen guess.

        A TS whose single successful guess went straight to a geometry optimization never sets
        ``chosen_ts``, so the frequencies must still be attributed to that guess.
        """
        sched, ts_label, tsgs = self.setup_ts_scheduler_for_freq_check(
            project='test_ts_freq_identity_sole', chosen_ts=None)
        # Leave exactly one successful guess, as when only one TS-search method produced a geometry.
        sched.species_dict[ts_label].ts_guesses = [tsgs[0], tsgs[2]]
        job = MagicMock()
        job.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'C2H6_freq_QChem.out')
        freq_ok, switched_ts = sched.check_negative_freq(label=ts_label, job=job,
                                                        vibfreqs=[-1000.0, 500.0, 1500.0])
        self.assertTrue(freq_ok)
        self.assertFalse(switched_ts)
        mock_switch_ts.assert_not_called()
        # Attributed to the sole successful guess (identity 2), not to the failed one (identity 0).
        self.assertEqual(tsgs[2].imaginary_freqs, [-1000.0])
        self.assertIsNone(tsgs[0].imaginary_freqs)

    def setup_ts_scheduler_for_conf_opt(self, project):
        """
        Set up a Scheduler with a TS species whose TSGuess objects are ordered so that a guess's
        position in the ``ts_guesses`` list, its ``index`` identity and its ``conformer_index``
        all differ from one another.

        Args:
            project (str): The project name.

        Returns:
            tuple: The Scheduler instance, the TS species label, and the list of TSGuess objects.
        """
        ts_xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")
        ts_label = 'TS_conf_opt_identity'
        ts_spc = ARCSpecies(label=ts_label, is_ts=True, xyz=ts_xyz, multiplicity=1, charge=0, compute_thermo=False)
        # Clustering removed the guess that held index 1, and the surviving guesses were reordered,
        # so neither the list position nor the identity equals the conformer job index.
        tsg_a = TSGuess(index=5, method='gcn', success=True, xyz=ts_xyz, execution_time='0:00:01')
        tsg_a.conformer_index = 1
        tsg_b = TSGuess(index=2, method='heuristics', success=True, xyz=ts_xyz, execution_time='0:00:01')
        tsg_b.conformer_index = 0
        ts_spc.ts_guesses = [tsg_a, tsg_b]
        project_directory = os.path.join(ARC_PATH, 'Projects', project)
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project=project, ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )
        return sched, ts_label, [tsg_a, tsg_b]

    def test_parse_conformer_attributes_ts_results_by_conformer_index(self):
        """Test that a TS conf_opt result is written to the TSGuess whose conformer_index matches the job."""
        sched, ts_label, (tsg_a, tsg_b) = self.setup_ts_scheduler_for_conf_opt(
            project='test_ts_conf_opt_identity')
        job_0, job_1 = MagicMock(), MagicMock()
        job_0.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'methylamine_conformer_0.out')
        job_1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'methylamine_conformer_1.out')
        for job in (job_0, job_1):
            job.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]

        sched.parse_conformer(job=job_0, label=ts_label, i=0)
        self.assertAlmostEqual(tsg_b.energy, -251596.4435088726, 5)
        self.assertEqual(tsg_b.opt_xyz, parser.parse_geometry(log_file_path=job_0.local_path_to_output_file))
        self.assertIsNone(tsg_a.energy)
        self.assertIsNone(tsg_a.opt_xyz)

        sched.parse_conformer(job=job_1, label=ts_label, i=1)
        self.assertAlmostEqual(tsg_a.energy, -254221.9433698632, 5)
        self.assertEqual(tsg_a.opt_xyz, parser.parse_geometry(log_file_path=job_1.local_path_to_output_file))
        # The identities must survive the ingestion, they are what chosen_ts refers to.
        self.assertEqual([tsg.index for tsg in sched.species_dict[ts_label].ts_guesses], [5, 2])

    def test_parse_conformer_warns_when_no_ts_guess_matches(self):
        """Test that a TS conf_opt result with no matching conformer_index is logged and not attributed."""
        sched, ts_label, (tsg_a, tsg_b) = self.setup_ts_scheduler_for_conf_opt(
            project='test_ts_conf_opt_identity_no_match')
        job = MagicMock()
        job.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'methylamine_conformer_0.out')
        job.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]
        with self.assertLogs('arc', level='WARNING') as context_manager:
            self.assertFalse(sched.parse_conformer(job=job, label=ts_label, i=5))
        self.assertTrue(any(ts_label in message for message in context_manager.output))
        for tsg in (tsg_a, tsg_b):
            self.assertIsNone(tsg.energy)
            self.assertIsNone(tsg.opt_xyz)
        self.assertEqual([tsg.index for tsg in sched.species_dict[ts_label].ts_guesses], [5, 2])

    def make_irc_scheduler(self,
                           ts_label: str,
                           project_directory_name: str,
                           num_guesses: int = 2,
                           chosen_ts_list: list | None = None,
                           project_directory: str | None = None,
                           ) -> Scheduler:
        """
        A helper for generating a Scheduler instance with a single TS species that has several TS guesses,
        simulating the state right after the first chosen guess completed its opt/freq/sp jobs.

        Args:
            ts_label (str): The TS species label.
            project_directory_name (str): The name of the testing project directory.
            num_guesses (int, optional): The number of TS guesses to generate.
            chosen_ts_list (list, optional): The indices of the TS guesses that were already tried.
            project_directory (str, optional): The project directory. Defaults to ``project_directory_name``
                                               under ``ARC_PATH/Projects``.

        Returns:
            Scheduler: The Scheduler instance.
        """
        ts_xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")
        chosen_ts_list = chosen_ts_list if chosen_ts_list is not None else [0]
        ts_spc = ARCSpecies(label=ts_label, is_ts=True, xyz=ts_xyz, multiplicity=1, charge=0, compute_thermo=False)
        ts_spc.ts_guesses = [TSGuess(index=i, method='heuristics', success=True, energy=100.0 + 10 * i,
                                     xyz=ts_xyz, execution_time='0:00:01')
                             for i in range(num_guesses)]
        for tsg in ts_spc.ts_guesses:
            tsg.opt_xyz = ts_xyz
            tsg.imaginary_freqs = [-500.0]
        ts_spc.chosen_ts = chosen_ts_list[-1]
        ts_spc.chosen_ts_list = list(chosen_ts_list)
        ts_spc.ts_guesses_exhausted = False
        project_directory = project_directory or os.path.join(ARC_PATH, 'Projects', project_directory_name)
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project=project_directory_name, ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )
        sched.output[ts_label]['job_types']['opt'] = True
        sched.output[ts_label]['job_types']['freq'] = True
        sched.output[ts_label]['job_types']['sp'] = True
        sched.output[ts_label]['convergence'] = True
        sched.job_dict[ts_label] = {'opt': {}, 'freq': {}, 'sp': {}}
        sched.running_jobs[ts_label] = list()
        return sched

    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_process_irc_verdict_false_switches_ts(self, mock_switch_ts):
        """Test that a positively failed IRC check rejects the TS and searches for a different TS guess."""
        ts_label = 'TS_irc_false'
        sched = self.make_irc_scheduler(ts_label=ts_label,
                                        project_directory_name='arc_project_for_testing_delete_after_usage_irc_1')
        sched.species_dict[ts_label].ts_checks['IRC'] = False
        with self.assertLogs('arc', level='ERROR') as log_records:
            sched.process_irc_verdict(ts_label=ts_label, rxn=None)
        mock_switch_ts.assert_called_once_with(ts_label)
        self.assertTrue(any('do NOT correspond' in record for record in log_records.output))

    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_process_irc_verdict_none_does_not_switch_ts(self, mock_switch_ts):
        """Test that an IRC check which was not performed does not reject the TS."""
        ts_label = 'TS_irc_none'
        sched = self.make_irc_scheduler(ts_label=ts_label,
                                        project_directory_name='arc_project_for_testing_delete_after_usage_irc_2')
        self.assertIsNone(sched.species_dict[ts_label].ts_checks['IRC'])
        with self.assertNoLogs('arc', level='ERROR'):
            sched.process_irc_verdict(ts_label=ts_label, rxn=None)
        mock_switch_ts.assert_not_called()
        self.assertTrue(sched.output[ts_label]['convergence'])

    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_process_irc_verdict_true_does_not_switch_ts(self, mock_switch_ts):
        """Test that a passed IRC check does not reject the TS."""
        ts_label = 'TS_irc_true'
        sched = self.make_irc_scheduler(ts_label=ts_label,
                                        project_directory_name='arc_project_for_testing_delete_after_usage_irc_3')
        sched.species_dict[ts_label].ts_checks['IRC'] = True
        with self.assertNoLogs('arc', level='ERROR'):
            sched.process_irc_verdict(ts_label=ts_label, rxn=None)
        mock_switch_ts.assert_not_called()
        self.assertTrue(sched.output[ts_label]['convergence'])

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_process_irc_verdict_false_terminates_when_guesses_are_exhausted(self, mock_run_opt):
        """Test that rejecting a TS by the IRC check terminates once all TS guesses were tried."""
        ts_label = 'TS_irc_exhausted'
        sched = self.make_irc_scheduler(ts_label=ts_label,
                                        project_directory_name='arc_project_for_testing_delete_after_usage_irc_4',
                                        num_guesses=1,
                                        chosen_ts_list=[0],
                                        )
        sched.species_dict[ts_label].ts_checks['IRC'] = False
        sched.process_irc_verdict(ts_label=ts_label, rxn=None)
        mock_run_opt.assert_not_called()
        self.assertTrue(sched.species_dict[ts_label].ts_guesses_exhausted
                        or sched.species_dict[ts_label].chosen_ts is None)
        self.assertFalse(sched.output[ts_label]['convergence'])
        self.assertIs(sched.species_dict[ts_label].ts_checks['IRC'], False)
        for job_type in sched.output[ts_label]['job_types']:
            sched.output[ts_label]['job_types'][job_type] = True
        sched.check_all_done(ts_label)
        self.assertFalse(sched.output[ts_label]['convergence'])

    @patch('arc.scheduler.check_irc_species_and_rxn')
    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_check_irc_species_rejects_a_ts_with_a_failed_irc(self, mock_run_opt, mock_check_irc_species_and_rxn):
        """Test that check_irc_species rejects a TS whose IRC endpoints do not match the requested wells."""
        ts_label = 'TS_irc_reject'
        sched = self.make_irc_scheduler(ts_label=ts_label,
                                        project_directory_name='arc_project_for_testing_delete_after_usage_irc_5',
                                        num_guesses=2,
                                        )
        ts_spc = sched.species_dict[ts_label]

        def fail_irc(**kwargs):
            """Simulate an IRC check the TS did not pass."""
            ts_spc.ts_checks['IRC'] = False

        mock_check_irc_species_and_rxn.side_effect = fail_irc
        irc_label_1, irc_label_2 = f'IRC_{ts_label}_1', f'IRC_{ts_label}_2'
        for irc_label in [irc_label_1, irc_label_2]:
            irc_spc = ARCSpecies(label=irc_label, xyz=ts_spc.get_xyz(), compute_thermo=False, irc_label=ts_label)
            sched.species_dict[irc_label] = irc_spc
            sched.species_list.append(irc_spc)
            sched.unique_species_labels.append(irc_label)
            sched.job_dict[irc_label] = {'opt': {}}
            sched.running_jobs[irc_label] = list()
            sched.initialize_output_dict(label=irc_label)
            sched.output[irc_label]['paths']['geo'] = f'{irc_label}_geo.out'
        ts_spc.irc_label = f'{irc_label_1} {irc_label_2}'
        sched.output[ts_label]['paths']['irc'] = ['irc_f.out', 'irc_r.out']

        sched.check_irc_species(label=irc_label_1)

        mock_check_irc_species_and_rxn.assert_called_once()
        self.assertEqual(sched.species_dict[ts_label].chosen_ts, 1)
        self.assertIn(1, sched.species_dict[ts_label].chosen_ts_list)
        self.assertNotIn(irc_label_1, sched.species_dict)
        self.assertNotIn(irc_label_2, sched.species_dict)
        self.assertNotIn(irc_label_1, sched.running_jobs)
        self.assertNotIn(irc_label_2, sched.output)
        self.assertIsNone(sched.species_dict[ts_label].irc_label)
        self.assertIsNone(sched.species_dict[ts_label].ts_checks['IRC'])
        mock_run_opt.assert_called_once()

    NEAR_MISS_RECORD = {'amplitude': 0.9,
                        'sub_gate_bonds': [{'atoms': [1, 2], 'symbols': 'C-Cl', 'percent': 4.55}]}

    def make_deferred_scheduler(self,
                                ts_label: str,
                                project_directory_name: str,
                                irc: bool = True,
                                irc_jobs_running: bool = True,
                                num_guesses: int = 2,
                                ) -> Scheduler:
        """
        A helper for generating a Scheduler instance whose TS failed NMD as a near miss and waits for its IRC.

        Args:
            ts_label (str): The TS species label.
            project_directory_name (str): The name of the testing project directory.
            irc (bool, optional): Whether IRC jobs are enabled for the run.
            irc_jobs_running (bool, optional): Whether the two IRC jobs of the TS are still running.
            num_guesses (int, optional): The number of TS guesses to generate.

        Returns:
            Scheduler: The Scheduler instance.
        """
        sched = self.make_irc_scheduler(ts_label=ts_label, project_directory_name=project_directory_name,
                                        num_guesses=num_guesses,
                                        project_directory=tempfile.mkdtemp(prefix=f'{project_directory_name}_'))
        sched.job_types['irc'] = irc
        sched.output[ts_label]['job_types'].setdefault('irc', False)
        sched.output[ts_label]['convergence'] = None
        ts_checks = sched.species_dict[ts_label].ts_checks
        ts_checks['NMD'] = False
        ts_checks['freq'] = True
        ts_checks[NMD_NEAR_MISS_KEY] = {'amplitude': self.NEAR_MISS_RECORD['amplitude'],
                                        'sub_gate_bonds': [dict(bond) for bond in
                                                           self.NEAR_MISS_RECORD['sub_gate_bonds']]}
        sched.job_dict[ts_label]['irc'] = dict()
        if irc_jobs_running:
            for job_name in ['irc_a1', 'irc_a2']:
                sched.job_dict[ts_label]['irc'][job_name] = MagicMock(job_name=job_name, execution_type='queue')
                sched.running_jobs[ts_label].append(job_name)
        return sched

    def make_irc_species(self, sched: Scheduler, ts_label: str, geo: bool = True) -> list[str]:
        """
        A helper that attaches two optimized IRC species to a TS in a Scheduler, as spawn_post_irc_jobs() does.

        Args:
            sched (Scheduler): The Scheduler instance.
            ts_label (str): The TS species label.
            geo (bool, optional): Whether the IRC species already have an optimized geometry.

        Returns:
            list[str]: The labels of the two IRC species.
        """
        ts_spc = sched.species_dict[ts_label]
        irc_labels = [f'IRC_{ts_label}_1', f'IRC_{ts_label}_2']
        for irc_label in irc_labels:
            irc_spc = ARCSpecies(label=irc_label, xyz=ts_spc.get_xyz(), compute_thermo=False, irc_label=ts_label)
            sched.species_dict[irc_label] = irc_spc
            sched.species_list.append(irc_spc)
            sched.unique_species_labels.append(irc_label)
            sched.job_dict[irc_label] = {'opt': {}}
            sched.running_jobs[irc_label] = list()
            sched.initialize_output_dict(label=irc_label)
            if geo:
                sched.output[irc_label]['paths']['geo'] = f'{irc_label}_geo.out'
        ts_spc.irc_label = ' '.join(irc_labels)
        sched.output[ts_label]['paths']['irc'] = ['irc_f.out', 'irc_r.out']
        return irc_labels

    def make_near_miss_freq_scheduler(self,
                                      project_directory_name: str,
                                      irc: bool,
                                      log_name: str = 'TS_C2H5Cl_guess7_near_miss.log',
                                      ):
        """
        A helper for a Scheduler whose TS has just finished the freq job of a shipped near-miss C2H5Cl saddle.

        Args:
            project_directory_name (str): The name of the testing project directory.
            irc (bool): Whether IRC jobs are enabled for the run.
            log_name (str, optional): The freq log of the saddle, under ``ARC_TESTING_PATH/freq``.

        Returns:
            tuple[Scheduler, str, JobAdapter, list]: The Scheduler, the TS label, the freq job, and the two
                                                     running IRC job mocks.
        """
        log_path = os.path.join(ARC_TESTING_PATH, 'freq', log_name)
        ts_label = 'TS_near_miss'
        ts_xyz = parser.parse_geometry(log_file_path=log_path)
        rxn = make_c2h5cl_rxn()
        ts_spc = ARCSpecies(label=ts_label, is_ts=True, xyz=ts_xyz, multiplicity=1, charge=0, compute_thermo=False)
        ts_spc.ts_guesses = [TSGuess(index=i, method='heuristics', success=True, energy=float(i), xyz=ts_xyz,
                                     execution_time='0:00:01') for i in range(2)]
        for tsg in ts_spc.ts_guesses:
            tsg.opt_xyz = ts_xyz
        ts_spc.chosen_ts = 0
        ts_spc.chosen_ts_list = [0]
        ts_spc.rxn_index = 0
        rxn.ts_species = ts_spc
        rxn.ts_label = ts_label
        project_directory = tempfile.mkdtemp(prefix=f'{project_directory_name}_')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        job_types = dict(self.job_types1, irc=irc)
        sched = Scheduler(project=project_directory_name, ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=job_types,
                          )
        sched.rxn_dict[0] = rxn
        sched.job_dict[ts_label] = {'opt': {}, 'freq': {}, 'irc': {}}
        sched.running_jobs[ts_label] = list()
        irc_jobs = list()
        for job_name in ['irc_a1', 'irc_a2']:
            irc_job = MagicMock(job_name=job_name, execution_type='queue')
            sched.job_dict[ts_label]['irc'][job_name] = irc_job
            sched.running_jobs[ts_label].append(job_name)
            irc_jobs.append(irc_job)
        job = job_factory(job_adapter='gaussian', project=project_directory_name, ess_settings=self.ess_settings,
                          species=[ts_spc], job_type='freq', level=Level(repr=default_levels_of_theory['freq']),
                          project_directory=project_directory, job_num=106)
        job.local_path_to_output_file = log_path
        job.job_status = ['done', {'status': 'done', 'keywords': list(), 'error': '', 'line': ''}]
        return sched, ts_label, job, irc_jobs

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_a_near_miss_nmd_failure_defers_to_the_irc_and_keeps_its_irc_jobs(self, mock_run_opt):
        """Test that a TS failing NMD only on the 5% gate is not switched, and its IRC jobs keep running."""
        sched, ts_label, job, irc_jobs = self.make_near_miss_freq_scheduler(
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_1', irc=True)
        sched.check_freq_job(label=ts_label, job=job)
        ts_checks = sched.species_dict[ts_label].ts_checks
        self.assertIs(ts_checks['NMD'], False)
        self.assertEqual([bond['symbols'] for bond in ts_checks[NMD_NEAR_MISS_KEY]['sub_gate_bonds']], ['C-Cl'])
        self.assertEqual(sched.running_jobs[ts_label], ['irc_a1', 'irc_a2'])
        for irc_job in irc_jobs:
            irc_job.delete.assert_not_called()
        self.assertEqual(sched.species_dict[ts_label].chosen_ts, 0)
        mock_run_opt.assert_not_called()
        self.assertTrue(sched.is_nmd_deferred(ts_label))

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_a_deferred_ts_has_its_reaction_zone_rotors_invalidated_as_a_passing_ts_does(self, mock_run_opt):
        """Test that a deferred TS loses its rotors with both pivots in the reaction zone."""
        sched, ts_label, job, _ = self.make_near_miss_freq_scheduler(
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_rotors', irc=True,
            log_name='TS_C2H5Cl_four_centre_near_miss.log')
        sched.species_dict[ts_label].rotors_dict = {
            0: {'pivots': [1, 2], 'top': [1, 4, 5, 6], 'scan': [3, 1, 2, 7], 'success': None,
                'invalidation_reason': '', 'times_dihedral_set': 0, 'scan_path': '', 'max_e': None,
                'symmetry': None, 'dimensions': 1, 'original_dihedrals': [], 'cont_indices': [],
                'directed_scan_type': 'ess', 'directed_scan': {}}}
        with patch('arc.checks.ts.DEFAULT_AMPLITUDE', [0.9]):
            sched.check_freq_job(label=ts_label, job=job)
        self.assertTrue(sched.is_nmd_deferred(ts_label))
        rotor = sched.species_dict[ts_label].rotors_dict[0]
        self.assertIs(rotor['success'], False)
        self.assertIn('pivTS', rotor['invalidation_reason'])

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_a_freq_recheck_drops_the_previous_irc_acceptance_warning_from_the_output(self, mock_run_opt):
        """Test that a new NMD evaluation of a TS drops the IRC acceptance warning of the previous one from the
        Scheduler output and keeps the other warnings."""
        sched, ts_label, job, _ = self.make_near_miss_freq_scheduler(
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_recheck', irc=True)
        sched.output[ts_label]['warnings'] = f'Other; {NMD_ACCEPTED_ON_IRC_WARNING}C-Cl (1, 2) 4.55%; '
        sched.check_freq_job(label=ts_label, job=job)
        self.assertTrue(sched.is_nmd_deferred(ts_label))
        self.assertEqual(sched.output[ts_label]['warnings'], 'Other; ')

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_a_freq_recheck_that_defers_an_accepted_ts_withholds_its_convergence(self, mock_run_opt):
        """Test that an IRC-accepted TS deferred again by a new NMD evaluation, and a deferred TS restored as
        converged, are not reported as converged while their IRC runs, and stay in the main loop."""
        sched, ts_label, job, _ = self.make_near_miss_freq_scheduler(
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_reenter', irc=True)
        ts_spc = sched.species_dict[ts_label]
        for job_type in sched.output[ts_label]['job_types']:
            sched.output[ts_label]['job_types'][job_type] = True
        sched.check_freq_job(label=ts_label, job=job)
        ts_spc.ts_checks['IRC'] = True
        sched.process_irc_verdict(ts_label=ts_label, rxn=None)
        self.assertIs(sched.output[ts_label]['convergence'], True)
        sched.check_freq_job(label=ts_label, job=job)
        self.assertTrue(sched.is_nmd_deferred(ts_label))
        self.assertNotIn('accepted_on_irc', ts_spc.ts_checks[NMD_NEAR_MISS_KEY])
        self.assertIsNone(sched.output[ts_label]['convergence'])
        sched.settle_nmd_deferrals()
        self.assertIsNone(sched.output[ts_label]['convergence'])
        self.assertEqual(ts_spc.chosen_ts, 0)
        mock_run_opt.assert_not_called()
        sched.output[ts_label]['convergence'] = True
        sched.settle_nmd_deferrals()
        self.assertTrue(sched.is_nmd_deferred(ts_label))
        self.assertIsNone(sched.output[ts_label]['convergence'])

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_a_freq_recheck_with_irc_jobs_running_drops_the_irc_verdict_of_the_previous_geometry(self, mock_run_opt):
        """Test that an IRC-accepted TS deferred again by the freq job of a new geometry, while its new IRC jobs
        run, is not accepted on the IRC verdict of its previous geometry."""
        sched, ts_label, job, _ = self.make_near_miss_freq_scheduler(
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_stale_irc', irc=True)
        ts_spc = sched.species_dict[ts_label]
        for job_type in sched.output[ts_label]['job_types']:
            sched.output[ts_label]['job_types'][job_type] = True
        sched.check_freq_job(label=ts_label, job=job)
        ts_spc.ts_checks['IRC'] = True
        sched.process_irc_verdict(ts_label=ts_label, rxn=None)
        self.assertIs(sched.output[ts_label]['convergence'], True)
        sched.check_freq_job(label=ts_label, job=job)
        self.assertTrue(sched.is_irc_pending(ts_label))
        self.assertTrue(sched.is_nmd_deferred(ts_label))
        self.assertIsNone(ts_spc.ts_checks['IRC'])
        sched.settle_nmd_deferrals()
        self.assertNotIn('accepted_on_irc', ts_spc.ts_checks[NMD_NEAR_MISS_KEY])
        self.assertIsNone(sched.output[ts_label]['convergence'])
        self.assertEqual(ts_spc.chosen_ts, 0)
        mock_run_opt.assert_not_called()

    def test_a_deferred_ts_the_run_gave_up_on_is_not_switched(self):
        """Test that a deferred TS whose convergence is False is not switched to its next guess once its IRC
        gives no verdict, or when IRC jobs are not run."""
        for irc in [True, False]:
            with self.subTest(irc=irc):
                ts_label = f'TS_defer_given_up_switch_{int(irc)}'
                sched = self.make_deferred_scheduler(
                    ts_label=ts_label, irc=irc, irc_jobs_running=False,
                    project_directory_name=f'arc_project_for_testing_delete_after_usage_nmd_defer_given_up_{int(irc)}')
                sched.output[ts_label]['convergence'] = False
                with patch.object(sched, 'switch_ts') as mock_switch_ts:
                    sched.settle_nmd_deferrals()
                    sched.process_irc_verdict(ts_label=ts_label, rxn=None)
                mock_switch_ts.assert_not_called()
                self.assertIs(sched.output[ts_label]['convergence'], False)

    def test_a_deferred_ts_the_run_gave_up_on_is_not_accepted_on_a_true_irc_verdict(self):
        """Test that a deferred TS whose convergence is False stays unconverged when its IRC verdict is True."""
        ts_label = 'TS_defer_given_up_accept'
        sched = self.make_deferred_scheduler(
            ts_label=ts_label, irc_jobs_running=False,
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_given_up_accept')
        irc_labels = self.make_irc_species(sched, ts_label)
        sched.running_jobs[irc_labels[0]].append('opt_a3')
        sched.output[ts_label]['convergence'] = False
        ts_spc = sched.species_dict[ts_label]
        ts_spc.ts_checks['IRC'] = True
        with patch.object(sched, 'switch_ts') as mock_switch_ts:
            sched.process_irc_verdict(ts_label=ts_label, rxn=None)
            sched.settle_nmd_deferrals()
        mock_switch_ts.assert_not_called()
        self.assertNotIn('accepted_on_irc', ts_spc.ts_checks[NMD_NEAR_MISS_KEY])
        self.assertIs(sched.output[ts_label]['convergence'], False)

    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_a_near_miss_nmd_failure_with_irc_off_switches_at_once(self, mock_switch_ts):
        """Test that a near-miss NMD failure is switched at once when IRC jobs are not run."""
        sched, ts_label, job, _ = self.make_near_miss_freq_scheduler(
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_2', irc=False)
        sched.check_freq_job(label=ts_label, job=job)
        self.assertIsNotNone(sched.species_dict[ts_label].ts_checks.get(NMD_NEAR_MISS_KEY))
        mock_switch_ts.assert_called_once_with(ts_label)

    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_an_nmd_failure_that_is_not_a_near_miss_switches_at_once(self, mock_switch_ts):
        """Test that an NMD failure whose evaluation finds no near miss drops the near-miss record of a previous
        evaluation and is switched at once."""
        sched, ts_label, job, _ = self.make_near_miss_freq_scheduler(
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_3', irc=True)
        ts_checks = sched.species_dict[ts_label].ts_checks
        ts_checks[NMD_NEAR_MISS_KEY] = {'amplitude': self.NEAR_MISS_RECORD['amplitude'],
                                        'sub_gate_bonds': [dict(bond) for bond in
                                                           self.NEAR_MISS_RECORD['sub_gate_bonds']]}
        with patch('arc.checks.nmd.get_near_miss_for_any_mapping', return_value=None):
            sched.check_freq_job(label=ts_label, job=job)
        self.assertIs(ts_checks['NMD'], False)
        self.assertNotIn(NMD_NEAR_MISS_KEY, ts_checks)
        mock_switch_ts.assert_called_once_with(ts_label)

    def test_a_deferred_ts_with_a_true_irc_verdict_is_accepted_and_records_both_facts(self):
        """Test that IRC True accepts a deferred TS, which keeps NMD False and records the IRC acceptance."""
        ts_label = 'TS_defer_true'
        sched = self.make_deferred_scheduler(
            ts_label=ts_label, irc_jobs_running=False,
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_4')
        self.make_irc_species(sched, ts_label)
        ts_spc = sched.species_dict[ts_label]
        ts_spc.ts_checks['IRC'] = True
        with patch.object(sched, 'switch_ts') as mock_switch_ts:
            sched.process_irc_verdict(ts_label=ts_label, rxn=None)
        mock_switch_ts.assert_not_called()
        self.assertFalse(sched.is_nmd_deferred(ts_label))
        self.assertIs(ts_spc.ts_checks['NMD'], False)
        self.assertIs(ts_spc.ts_checks[NMD_NEAR_MISS_KEY]['accepted_on_irc'], True)
        expected = f'{NMD_ACCEPTED_ON_IRC_WARNING}C-Cl (1, 2) 4.55% at amplitude 0.9; '
        self.assertIn(expected, ts_spc.ts_checks['warnings'])
        self.assertIn(expected, sched.output[ts_label]['warnings'])
        published = _ts_checks_to_dict(ts_spc)
        self.assertIs(published['NMD'], False)
        self.assertIs(published['IRC'], True)
        self.assertIn(expected, published['warnings'])
        self.assertTrue(sched.output[ts_label]['convergence'])
        sched.process_irc_verdict(ts_label=ts_label, rxn=None)
        self.assertEqual(ts_spc.ts_checks['warnings'].count(NMD_ACCEPTED_ON_IRC_WARNING), 1)

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_switching_an_irc_accepted_ts_drops_its_acceptance_warning_and_keeps_the_others(self, mock_run_opt):
        """Test that a TS accepted on IRC and later switched (e.g., by the E0 check) does not hand its
        acceptance warning to the next guess, while unrelated output warnings stay."""
        ts_label = 'TS_defer_true_then_switched'
        sched = self.make_deferred_scheduler(
            ts_label=ts_label, irc_jobs_running=False,
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_14')
        self.make_irc_species(sched, ts_label)
        sched.output[ts_label]['warnings'] = 'Unrelated warning; '
        sched.species_dict[ts_label].ts_checks['IRC'] = True
        sched.process_irc_verdict(ts_label=ts_label, rxn=None)
        self.assertIn(NMD_ACCEPTED_ON_IRC_WARNING, sched.output[ts_label]['warnings'])
        sched.switch_ts(ts_label)
        self.assertNotIn(NMD_ACCEPTED_ON_IRC_WARNING, sched.output[ts_label]['warnings'])
        self.assertIn('Unrelated warning; ', sched.output[ts_label]['warnings'])

    def test_is_irc_pending_ignores_terminal_irc_tasks_in_an_active_batch(self):
        """Test that only an IRC task that can still yield a verdict keeps a TS's IRC work pending."""
        ts_label = 'TS_defer_pipe'
        sched = self.make_deferred_scheduler(
            ts_label=ts_label, irc_jobs_running=False,
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_15')
        pipe = MagicMock(pipe_root='pipe_root')
        pipe.tasks = [MagicMock(owner_key=ts_label, task_family='irc', task_id='irc_task'),
                      MagicMock(owner_key='other_species', task_family='species_sp', task_id='sp_task')]
        sched.active_pipes = {'pipe_root': pipe}
        expected = {TaskState.FAILED_TERMINAL: False, TaskState.CANCELLED: False, TaskState.PENDING: True,
                    TaskState.RUNNING: True, TaskState.COMPLETED: True, TaskState.FAILED_ESS: True}
        for status, pending in expected.items():
            with self.subTest(status=status.value), \
                    patch('arc.scheduler.read_task_state', return_value=MagicMock(status=status.value)):
                self.assertIs(sched.is_irc_pending(ts_label), pending)
        with patch('arc.scheduler.read_task_state', side_effect=FileNotFoundError):
            self.assertTrue(sched.is_irc_pending(ts_label))

    def test_an_irc_job_held_back_by_a_queue_guard_keeps_a_deferred_ts_waiting(self):
        """Test that an IRC job of a deferred TS, or a job of its IRC species, that is unsubmitted, put off, or
        stranded keeps the IRC pending and the TS on its current guess."""
        ts_label = 'TS_defer_held'
        sched = self.make_deferred_scheduler(
            ts_label=ts_label, irc_jobs_running=False,
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_held')
        irc_label = self.make_irc_species(sched, ts_label)[0]
        held_jobs = {(ts_label, 'irc_a1'): True, (irc_label, 'opt_a5'): True, (ts_label, 'sp_a3'): False}
        for (label, job_name), pending in held_jobs.items():
            for holder in ['unsubmitted', 'deferred', 'stranded']:
                with self.subTest(label=label, job_name=job_name, holder=holder):
                    sched.unsubmitted_jobs, sched.deferred_jobs, sched.stranded_jobs = list(), dict(), dict()
                    if holder == 'unsubmitted':
                        sched.unsubmitted_jobs.append((label, MagicMock(job_name=job_name,
                                                                        job_type=job_name.split('_')[0])))
                    else:
                        getattr(sched, f'{holder}_jobs')[label] = [job_name]
                    self.assertIs(sched.is_irc_pending(ts_label), pending)
                    with patch.object(sched, 'switch_ts') as mock_switch_ts:
                        sched.settle_nmd_deferrals()
                    self.assertEqual(mock_switch_ts.call_count, 0 if pending else 1)
                    self.assertTrue(sched.is_nmd_deferred(ts_label))

    def test_a_stale_irc_acceptance_is_revoked_unless_irc_is_true(self):
        """Test that an IRC-accepted near miss stays accepted only while its IRC verdict is True."""
        for i, (irc, through_settle) in enumerate([(None, False), (None, True), (False, True), (True, True)]):
            with self.subTest(irc=irc, through_settle=through_settle):
                ts_label = f'TS_defer_stale_{i}'
                sched = self.make_deferred_scheduler(
                    ts_label=ts_label, irc_jobs_running=False,
                    project_directory_name=f'arc_project_for_testing_delete_after_usage_nmd_defer_16_{i}')
                self.make_irc_species(sched, ts_label)
                ts_spc = sched.species_dict[ts_label]
                ts_spc.ts_checks['IRC'] = True
                sched.process_irc_verdict(ts_label=ts_label, rxn=None)
                self.assertTrue(sched.output[ts_label]['convergence'])
                ts_spc.ts_checks['IRC'] = irc
                with patch.object(sched, 'switch_ts') as mock_switch_ts, \
                        patch.object(sched, 'is_irc_pending', return_value=False):
                    if through_settle:
                        sched.settle_nmd_deferrals()
                    else:
                        sched.process_irc_verdict(ts_label=ts_label, rxn=None)
                if irc is True:
                    mock_switch_ts.assert_not_called()
                    self.assertFalse(sched.is_nmd_deferred(ts_label))
                    self.assertIs(ts_spc.ts_checks[NMD_NEAR_MISS_KEY]['accepted_on_irc'], True)
                    self.assertEqual(ts_spc.ts_checks['warnings'].count(NMD_ACCEPTED_ON_IRC_WARNING), 1)
                    self.assertTrue(sched.output[ts_label]['convergence'])
                else:
                    mock_switch_ts.assert_called_once_with(ts_label)
                    self.assertNotIn('accepted_on_irc', ts_spc.ts_checks[NMD_NEAR_MISS_KEY])
                    self.assertNotIn(NMD_ACCEPTED_ON_IRC_WARNING, ts_spc.ts_checks['warnings'])
                    self.assertNotIn(NMD_ACCEPTED_ON_IRC_WARNING, sched.output[ts_label]['warnings'])
                    self.assertFalse(sched.output[ts_label]['convergence'])

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_a_deferred_ts_with_a_false_irc_verdict_is_switched(self, mock_run_opt):
        """Test that IRC False rejects a deferred TS through switch_ts, which clears the deferred state."""
        ts_label = 'TS_defer_false'
        sched = self.make_deferred_scheduler(
            ts_label=ts_label, irc_jobs_running=False,
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_5')
        self.make_irc_species(sched, ts_label)
        sched.species_dict[ts_label].ts_checks['IRC'] = False
        with self.assertLogs('arc', level='ERROR'):
            sched.process_irc_verdict(ts_label=ts_label, rxn=None)
        self.assertEqual(sched.species_dict[ts_label].chosen_ts, 1)
        self.assertNotIn(NMD_NEAR_MISS_KEY, sched.species_dict[ts_label].ts_checks)
        self.assertFalse(sched.is_nmd_deferred(ts_label))
        mock_run_opt.assert_called_once()

    def test_a_deferred_ts_with_an_undetermined_irc_verdict_is_switched(self):
        """Test that an IRC verdict of None, once both IRC ends were checked, switches a deferred TS."""
        ts_label = 'TS_defer_none'
        sched = self.make_deferred_scheduler(
            ts_label=ts_label, irc_jobs_running=False,
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_6')
        irc_labels = self.make_irc_species(sched, ts_label)
        with patch('arc.scheduler.check_irc_species_and_rxn') as mock_check, \
                patch.object(sched, 'switch_ts') as mock_switch_ts:
            sched.check_irc_species(label=irc_labels[0])
        mock_check.assert_called_once()
        self.assertIsNone(sched.species_dict[ts_label].ts_checks['IRC'])
        mock_switch_ts.assert_called_once_with(ts_label)

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_a_deferred_ts_on_its_last_guess_stays_rejected_when_its_irc_gives_no_verdict(self, mock_run_opt):
        """Test that a deferred TS whose guesses run out keeps NMD False, ends unconverged, and is not retried."""
        for irc, name in [(True, 'nmd_defer_last_1'), (False, 'nmd_defer_last_2')]:
            with self.subTest(irc=irc):
                ts_label = f'TS_defer_last_{int(irc)}'
                sched = self.make_deferred_scheduler(
                    ts_label=ts_label, irc=irc, irc_jobs_running=False, num_guesses=1,
                    project_directory_name=f'arc_project_for_testing_delete_after_usage_{name}')
                if irc:
                    sched.resolve_nmd_deferral(ts_label)
                else:
                    sched.settle_nmd_deferrals()
                mock_run_opt.assert_not_called()
                self.assertIs(sched.species_dict[ts_label].ts_checks['NMD'], False)
                self.assertIs(sched.output[ts_label]['convergence'], False)
                self.assertFalse(sched.is_nmd_deferred(ts_label))
                with patch.object(sched, 'switch_ts') as mock_switch_ts:
                    sched.settle_nmd_deferrals()
                mock_switch_ts.assert_not_called()

    def test_a_deferred_ts_whose_irc_jobs_ended_without_output_is_switched_once_nothing_is_pending(self):
        """Test that a deferred TS waits while IRC work is pending, and is switched once none is left."""
        ts_label = 'TS_defer_crash'
        sched = self.make_deferred_scheduler(
            ts_label=ts_label,
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_7')
        with patch.object(sched, 'switch_ts') as mock_switch_ts:
            sched.settle_nmd_deferrals()
            mock_switch_ts.assert_not_called()
            sched.running_jobs[ts_label] = ['irc_a2']
            sched.settle_nmd_deferrals()
            mock_switch_ts.assert_not_called()
            sched.running_jobs[ts_label] = list()
            sched._pending_pipe_irc.add((ts_label, 'reverse'))
            sched.settle_nmd_deferrals()
            mock_switch_ts.assert_not_called()
            sched._pending_pipe_irc.clear()
            sched.settle_nmd_deferrals()
        mock_switch_ts.assert_called_once_with(ts_label)

    def test_a_deferred_ts_waits_while_its_irc_species_are_optimized(self):
        """Test that a deferred TS waits for the optimization of its IRC species, and is switched when it fails."""
        ts_label = 'TS_defer_irc_opt'
        sched = self.make_deferred_scheduler(
            ts_label=ts_label, irc_jobs_running=False,
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_8')
        irc_labels = self.make_irc_species(sched, ts_label, geo=False)
        sched.running_jobs[irc_labels[1]] = ['opt_a3']
        with patch.object(sched, 'switch_ts') as mock_switch_ts:
            sched.settle_nmd_deferrals()
            mock_switch_ts.assert_not_called()
            sched.running_jobs[irc_labels[1]] = list()
            sched.settle_nmd_deferrals()
        mock_switch_ts.assert_called_once_with(ts_label)

    @patch('arc.scheduler.Scheduler.run_job')
    def test_a_deferred_ts_whose_irc_stopped_on_delta_x_spawns_its_end_optimization(self, mock_run_job):
        """Test that an IRC that ended on the Gaussian DeltaX error is used, and the deferred TS waits for it."""
        ts_label = 'TS_defer_deltax'
        sched = self.make_deferred_scheduler(
            ts_label=ts_label, irc_jobs_running=False,
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_9')
        log_path = os.path.join(ARC_TESTING_PATH, 'irc', 'TS_C2H5Cl_irc_deltax.log')
        status, keywords, error, line = determine_ess_status(output_path=log_path, species_label=ts_label,
                                                             job_type='irc', software='gaussian')
        self.assertEqual((status, keywords), ('errored', ['DeltaX', 'GL123']))
        irc_job = MagicMock(job_name='irc_a1', job_type='irc', job_adapter='gaussian', execution_type='queue',
                            local_path_to_output_file=log_path, irc_direction='forward', species=[
                                sched.species_dict[ts_label]], directed_scan_type=None, ess_trsh_methods=list(),
                            job_status=['done', {'status': status, 'keywords': keywords, 'error': error,
                                                 'line': line}])
        sched.job_dict[ts_label]['irc']['irc_a1'] = irc_job
        sched.running_jobs[ts_label].append('irc_a1')

        def queue_the_job(label, **kwargs):
            """Record the queued end optimization as a running job, as run_job() does."""
            sched.running_jobs.setdefault(label, list()).append('opt_a2')

        mock_run_job.side_effect = queue_the_job
        self.assertTrue(sched.end_job(job=irc_job, label=ts_label, job_name='irc_a1'))
        sched.spawn_post_irc_jobs(label=ts_label, job=irc_job)
        irc_label = f'IRC_{ts_label}_1'
        self.assertIn(irc_label, sched.species_dict)
        self.assertEqual(mock_run_job.call_args.kwargs['label'], irc_label)
        self.assertEqual(sched.running_jobs[irc_label], ['opt_a2'])
        with patch.object(sched, 'switch_ts') as mock_switch_ts:
            sched.settle_nmd_deferrals()
        mock_switch_ts.assert_not_called()
        self.assertTrue(sched.is_nmd_deferred(ts_label))

    def test_a_deferred_ts_is_not_marked_converged_before_its_irc_verdict(self):
        """Test that check_all_done() does not converge a TS that waits for its IRC verdict."""
        ts_label = 'TS_defer_converge'
        sched = self.make_deferred_scheduler(
            ts_label=ts_label,
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_10')
        for job_type in sched.output[ts_label]['job_types']:
            sched.output[ts_label]['job_types'][job_type] = True
        sched.check_all_done(ts_label)
        self.assertIsNone(sched.output[ts_label]['convergence'])
        sched.species_dict[ts_label].ts_checks.pop(NMD_NEAR_MISS_KEY)
        sched.species_dict[ts_label].ts_checks['NMD'] = True
        sched.check_all_done(ts_label)
        self.assertTrue(sched.output[ts_label]['convergence'])

    @patch('arc.scheduler.Scheduler.generate_final_ts_guess_report')
    @patch('arc.scheduler.Scheduler.spawn_ts_jobs')
    @patch('arc.scheduler.Scheduler.run_conformer_jobs')
    def test_a_restart_in_the_deferred_state_resumes_waiting_or_exits(self, *mocks):
        """Test that a restarted deferred TS waits while its IRC jobs run, and is switched when none is left."""
        ts_label = 'TS_defer_restart'
        sched = self.make_deferred_scheduler(
            ts_label=ts_label, irc_jobs_running=False,
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_11')
        restored = ARCSpecies(species_dict=sched.species_dict[ts_label].as_dict())
        self.assertEqual(restored.ts_checks[NMD_NEAR_MISS_KEY], self.NEAR_MISS_RECORD)
        sched.species_dict[ts_label] = restored
        sched.species_list = [restored]
        with patch.object(sched, 'switch_ts') as mock_switch_ts:
            sched.running_jobs[ts_label] = ['irc_a1']
            sched.settle_nmd_deferrals()
            mock_switch_ts.assert_not_called()
            del sched.running_jobs[ts_label]
            sched.schedule_jobs()
        mock_switch_ts.assert_called_once_with(ts_label)

        sched.job_types['irc'] = False
        sched.running_jobs[ts_label] = ['irc_a1']
        with patch.object(sched, 'switch_ts') as mock_switch_ts:
            sched.settle_nmd_deferrals()
        mock_switch_ts.assert_called_once_with(ts_label)

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_a_restarted_deferred_ts_with_a_false_irc_verdict_is_rejected_as_process_irc_verdict_does(
            self, mock_run_opt):
        """Test that a deferred TS restored with IRC False is rejected, and is unconverged once guesses run out."""
        ts_label = 'TS_defer_restart_false'
        sched = self.make_deferred_scheduler(
            ts_label=ts_label, irc_jobs_running=False,
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_13')
        ts_spc = sched.species_dict[ts_label]
        ts_spc.ts_guesses = ts_spc.ts_guesses[:1]
        ts_spc.chosen_ts, ts_spc.chosen_ts_list = 0, [0]
        ts_spc.ts_checks['IRC'] = False
        with self.assertLogs('arc', level='ERROR'):
            sched.settle_nmd_deferrals()
        mock_run_opt.assert_not_called()
        self.assertFalse(sched.output[ts_label]['convergence'])
        self.assertIs(sched.species_dict[ts_label].ts_checks['IRC'], False)
        self.assertNotIn(NMD_NEAR_MISS_KEY, sched.species_dict[ts_label].ts_checks)

    def test_a_restart_file_without_a_near_miss_record_behaves_as_before(self):
        """Test that a TS restored without the near-miss key, or with a malformed one, is never deferred."""
        ts_label = 'TS_defer_old_restart'
        sched = self.make_deferred_scheduler(
            ts_label=ts_label,
            project_directory_name='arc_project_for_testing_delete_after_usage_nmd_defer_12')
        ts_checks = sched.species_dict[ts_label].ts_checks
        for bad in [None, 'C-Cl 4.55%', {}, {'sub_gate_bonds': []}]:
            ts_checks[NMD_NEAR_MISS_KEY] = bad
            self.assertFalse(sched.is_nmd_deferred(ts_label), msg=f'record {bad}')
        ts_checks.pop(NMD_NEAR_MISS_KEY)
        self.assertFalse(sched.is_nmd_deferred(ts_label))
        with patch.object(sched, 'switch_ts') as mock_switch_ts:
            sched.settle_nmd_deferrals()
        mock_switch_ts.assert_not_called()

    @patch('arc.scheduler.Scheduler.generate_final_ts_guess_report')
    @patch('arc.scheduler.Scheduler.spawn_ts_jobs')
    @patch('arc.scheduler.Scheduler.run_conformer_jobs')
    def test_schedule_jobs_tolerates_a_label_deleted_while_it_is_iterated(self,
                                                                         mock_run_conformer_jobs,
                                                                         mock_spawn_ts_jobs,
                                                                         mock_generate_report):
        """
        Test that the main loop survives a species label being removed from running_jobs mid-iteration.

        Rejecting a TS by its IRC verdict calls switch_ts(), which calls delete_all_species_jobs()
        on the TS, which in turn deletes the running_jobs entries of the very IRC species labels the
        main loop is iterating over. The main loop must therefore not assume that a label it started
        the iteration with is still a key of running_jobs by the time it reaches the cleanup below.
        """
        ts_label = 'TS_irc_deleted_label'
        sched = self.make_irc_scheduler(ts_label=ts_label,
                                        project_directory_name='arc_project_for_testing_delete_after_usage_irc_6')
        irc_label = f'IRC_{ts_label}_1'
        irc_spc = ARCSpecies(label=irc_label, xyz=sched.species_dict[ts_label].get_xyz(),
                             compute_thermo=False, irc_label=ts_label)
        sched.species_dict[irc_label] = irc_spc
        sched.species_list.append(irc_spc)
        sched.job_dict[irc_label] = {'opt': {}}
        sched.initialize_output_dict(label=irc_label)
        del sched.running_jobs[ts_label]
        sched.running_jobs[irc_label] = list()
        sched.unique_species_labels = [irc_label]

        def delete_the_iterated_label(label):
            """Delete the label being iterated, as delete_all_species_jobs() does for an IRC species."""
            del sched.running_jobs[label]

        with patch.object(sched, 'check_all_done', side_effect=delete_the_iterated_label) as mock_check_all_done:
            sched.schedule_jobs()
        mock_check_all_done.assert_called_once_with(irc_label)
        self.assertEqual(sched.running_jobs, dict())

    @patch('arc.scheduler.Scheduler.run_job')
    def test_run_sp_monoatomic_dlpno(self, mock_run_job):
        """Monoatomic H falls back to HF; heavier atoms (O) keep DLPNO intact."""
        dlpno_level = Level(method='DLPNO-CCSD(T)-F12', basis='cc-pVTZ-F12',
                            auxiliary_basis='aug-cc-pVTZ/C', cabs='cc-pVTZ-F12-CABS',
                            software='orca')

        for label, smiles in [('H_atom', '[H]'), ('O_atom', '[O]')]:
            self.sched1.species_dict[label] = ARCSpecies(label=label, smiles=smiles)
            self.sched1.job_dict[label] = {}
            self.sched1.output[label] = {'paths': {}, 'job_types': {},
                                         'errors': '', 'warnings': '', 'conformers': ''}

        # Single-electron atom → HF fallback, aux/cabs preserved.
        self.sched1.run_sp_job(label='H_atom', level=dlpno_level)
        h_level = mock_run_job.call_args.kwargs['level_of_theory']
        self.assertEqual(h_level.method, 'hf')
        self.assertEqual(h_level.basis, 'cc-pvtz-f12')
        self.assertEqual(h_level.auxiliary_basis, 'aug-cc-pvtz/c')
        self.assertEqual(h_level.cabs, 'cc-pvtz-f12-cabs')

        # Heavier monoatomic → DLPNO level unchanged.
        mock_run_job.reset_mock()
        self.sched1.run_sp_job(label='O_atom', level=dlpno_level)
        o_level = mock_run_job.call_args.kwargs['level_of_theory']
        self.assertEqual(o_level.method, 'dlpno-ccsd(t)-f12')
        self.assertEqual(o_level.cabs, 'cc-pvtz-f12-cabs')

    def test_check_directed_scan_job_skips_isomorphism_for_ts(self):
        """check_directed_scan_job must not call check_xyz_isomorphism for a TS; is_isomorphic is recorded as True."""
        ts_xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")
        ts_spc = ARCSpecies(label='TS_dirscan', is_ts=True, xyz=ts_xyz, multiplicity=1, charge=0,
                            compute_thermo=False)
        ts_spc.rotors_dict = {0: {'pivots': [1, 2], 'directed_scan': {}}}

        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_ts_iso_dirscan')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='test_ts_iso_dirscan', ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )

        job_mock = MagicMock()
        job_mock.job_status = [None, {'status': 'done'}]
        job_mock.local_path_to_output_file = '/fake/path.log'
        job_mock.pivots = [1, 2]
        job_mock.dihedrals = [45.0]
        job_mock.ess_trsh_methods = []

        with patch('arc.species.species.ARCSpecies.check_xyz_isomorphism') as mock_iso, \
                patch('arc.scheduler.parser.parse_geometry', return_value=ts_xyz), \
                patch('arc.scheduler.parser.parse_e_elect', return_value=-123.45):
            sched.check_directed_scan_job(label='TS_dirscan', job=job_mock)

        mock_iso.assert_not_called()
        recorded = sched.species_dict['TS_dirscan'].rotors_dict[0]['directed_scan'][('45.00',)]
        self.assertTrue(recorded['is_isomorphic'])

    def test_check_directed_scan_job_parses_energy_from_a_real_log(self):
        """check_directed_scan_job must record a float energy parsed from the real ESS output file.

        The parsers made by parser.make_parser() take a ``log_file_path`` argument; calling one with
        ``path=`` raises a TypeError that leaves every rotor unsuccessful, so no HinderedRotor block
        reaches Arkane and the thermo silently degrades to RRHO. No parser is mocked here.
        """
        h2o2_xyz = str_to_xyz("""O       0.68416100    0.00000000    0.02026600
        O      -0.68416100    0.00000000    0.02026600
        H       0.87768200    0.75828100   -0.44584400
        H      -0.87768200   -0.75828100   -0.44584400""")
        ts_spc = ARCSpecies(label='TS_dirscan_e', is_ts=True, xyz=h2o2_xyz, multiplicity=1, charge=0,
                            compute_thermo=False)
        ts_spc.rotors_dict = {0: {'pivots': [1, 2], 'directed_scan': {}}}

        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_dirscan_e_elect')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='test_dirscan_e_elect', ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )

        job_mock = MagicMock()
        job_mock.job_status = [None, {'status': 'done'}]
        job_mock.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'rotor_scans', 'H2O2.out')
        job_mock.pivots = [1, 2]
        job_mock.dihedrals = [45.0]
        job_mock.ess_trsh_methods = []

        sched.check_directed_scan_job(label='TS_dirscan_e', job=job_mock)

        recorded = sched.species_dict['TS_dirscan_e'].rotors_dict[0]['directed_scan'][('45.00',)]
        self.assertIsInstance(recorded['energy'], float)
        self.assertAlmostEqual(recorded['energy'], -398031.18523281615, places=3)

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_troubleshoot_scan_job_skips_isomorphism_for_ts(self, mock_run_opt):
        """troubleshoot_scan_job must not call check_xyz_isomorphism for a TS when applying 'change conformer'."""
        ts_xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")
        new_xyz = str_to_xyz("""N       0.91000000    0.52000000    0.00000000
        H       1.81000000    1.04000000    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91000000    1.23000000    0.72000000""")
        ts_spc = ARCSpecies(label='TS_trsh', is_ts=True, xyz=ts_xyz, multiplicity=1, charge=0,
                            compute_thermo=False)
        ts_spc.rotors_dict = {0: {'pivots': [1, 2], 'scan': [3, 1, 2, 4], 'scan_path': '',
                                  'invalidation_reason': '', 'success': None, 'symmetry': None,
                                  'times_dihedral_set': 0, 'trsh_methods': [], 'trsh_counter': 0}}

        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_ts_iso_trsh')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='test_ts_iso_trsh', ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )
        sched.trsh_ess_jobs = True
        sched.trsh_rotors = True

        job_mock = MagicMock()
        job_mock.species_label = 'TS_trsh'
        job_mock.rotor_index = 0
        job_mock.torsions = [[3, 1, 2, 4]]
        job_mock.job_name = 'scan_a200'

        with patch('arc.species.species.ARCSpecies.check_xyz_isomorphism') as mock_iso, \
                patch('arc.scheduler.Scheduler.delete_all_species_jobs'):
            sched.troubleshoot_scan_job(job=job_mock, methods={'change conformer': new_xyz})

        mock_iso.assert_not_called()
        self.assertEqual(sched.species_dict['TS_trsh'].final_xyz, new_xyz)
        mock_run_opt.assert_called_once()

    @patch('arc.scheduler.Scheduler.run_job')
    def test_troubleshoot_scan_job_skips_isomorphism_for_ts_non_conformer(self, mock_run_job):
        """troubleshoot_scan_job must not call check_xyz_isomorphism for a TS in the non-'change conformer' branch."""
        ts_xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")
        ts_spc = ARCSpecies(label='TS_trsh_nc', is_ts=True, xyz=ts_xyz, multiplicity=1, charge=0,
                            compute_thermo=False)
        ts_spc.rotors_dict = {0: {'pivots': [1, 2], 'scan': [3, 1, 2, 4], 'scan_path': '',
                                  'invalidation_reason': '', 'success': None, 'symmetry': None,
                                  'times_dihedral_set': 0, 'trsh_methods': [], 'trsh_counter': 0}}

        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_ts_iso_trsh_nc')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='test_ts_iso_trsh_nc', ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )
        sched.trsh_ess_jobs = True
        sched.trsh_rotors = True

        job_mock = MagicMock()
        job_mock.species_label = 'TS_trsh_nc'
        job_mock.rotor_index = 0
        job_mock.torsions = [[2, 0, 1, 3]]  # 0-indexed; torsions_to_scans yields the 1-indexed scan [3, 1, 2, 4]
        job_mock.job_name = 'scan_a201'
        job_mock.scan_res = 8.0
        job_mock.xyz = ts_xyz
        job_mock.level = Level(repr=default_levels_of_theory['scan'])
        job_mock.local_path_to_output_file = '/fake/path.log'

        with patch('arc.species.species.ARCSpecies.check_xyz_isomorphism') as mock_iso, \
                patch('arc.scheduler.trsh_scan_job') as mock_trsh_scan:
            mock_trsh_scan.return_value = [], 4.0

            sched.troubleshoot_scan_job(job=job_mock, methods={'inc_res': None})

        mock_iso.assert_not_called()
        mock_trsh_scan.assert_called_once_with(
            label='TS_trsh_nc',
            scan_res=8.0,
            scan=[3, 1, 2, 4],
            scan_list=[[3, 1, 2, 4]],
            methods={'inc_res': None},
            log_file='/fake/path.log',
        )

        mock_run_job.assert_called_once()
        _, kwargs = mock_run_job.call_args
        self.assertEqual(kwargs['label'], 'TS_trsh_nc')
        self.assertEqual(kwargs['xyz'], ts_xyz)
        self.assertEqual(kwargs['level_of_theory'], job_mock.level)
        self.assertEqual(kwargs['job_type'], 'scan')
        self.assertEqual(kwargs['torsions'], [[2, 0, 1, 3]])
        self.assertEqual(kwargs['scan_trsh'], [])
        self.assertEqual(kwargs['trsh'], {'scan_res': 4.0})
        self.assertEqual(kwargs['rotor_index'], 0)

    def test_report_running_jobs_snapshot(self):
        """The snapshot file is overwritten on change and left untouched (heartbeat only) on no change."""
        path = self.sched1.running_jobs_snapshot_path
        if os.path.isfile(path):
            os.remove(path)
        self.sched1._last_status_payload = None
        self.sched1.running_jobs = {'spcA': ['opt_a1234']}
        self.sched1.active_pipes = {}

        # First call: file doesn't exist, snapshot must be written. An unresolvable job
        # (no matching Job object in job_dict) falls back to just its name.
        self.sched1.report_running_jobs_snapshot()
        snapshot = read_yaml_file(path)
        self.assertIn('timestamp', snapshot)
        self.assertEqual(snapshot['running_jobs'], {'spcA': [{'name': 'opt_a1234'}]})

        # Second call with identical payload: file must be left untouched.
        first_mtime = os.path.getmtime(path)
        self.sched1.report_running_jobs_snapshot()
        self.assertEqual(os.path.getmtime(path), first_mtime)

        # Mutating the live running_jobs lists must not alias the stored payload.
        self.sched1.running_jobs['spcA'].append('sp_b5678')
        self.assertNotEqual(self.sched1._last_status_payload['running_jobs'], self.sched1.running_jobs)

        # Third call after a change: the file must be overwritten with the new snapshot only.
        self.sched1.report_running_jobs_snapshot()
        snapshot = read_yaml_file(path)
        self.assertEqual(snapshot['running_jobs'],
                         {'spcA': [{'name': 'opt_a1234'}, {'name': 'sp_b5678'}]})

        # A resolvable job carries its server identifiers (server_name/job_id/adapter/server)
        # into the snapshot, so the file alone tells you what to look for in qstat.
        job = SimpleNamespace(job_name='tsg3', job_server_name='a3129', job_id='4438988',
                              job_adapter='orca_neb', server='server1')
        self.sched1.job_dict['TS0'] = {'tsg': {3: job}}
        self.sched1.running_jobs = {'TS0': ['tsg3']}
        self.sched1._last_status_payload = None
        self.sched1.report_running_jobs_snapshot()
        snapshot = read_yaml_file(path)
        self.assertEqual(snapshot['running_jobs']['TS0'][0],
                         {'name': 'tsg3', 'server_name': 'a3129', 'job_id': '4438988',
                          'adapter': 'orca_neb', 'server': 'server1'})

        # A label that has jobs in job_dict, none of which match the requested job_name, also
        # falls back to just the name rather than borrowing another job's identifiers.
        self.sched1.running_jobs = {'TS0': ['sp_b5678']}
        self.sched1._last_status_payload = None
        self.sched1.report_running_jobs_snapshot()
        snapshot = read_yaml_file(path)
        self.assertEqual(snapshot['running_jobs'], {'TS0': [{'name': 'sp_b5678'}]})

        os.remove(path)

    @patch('arc.scheduler.job_factory')
    def test_run_job_reports_a_collapsible_reference_for_the_job_it_spawned(self, mock_job_factory):
        """Test that every job run_job() spawns is offered to the collapsible-reference report"""
        job_mock = MagicMock()
        job_mock.job_name = 'sp_a0000'
        job_mock.server = None
        mock_job_factory.return_value = job_mock
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian')
        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_run_job_collapsible_reference')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='test_run_job_collapsible_reference', ess_settings=self.ess_settings,
                          species_list=[ARCSpecies(label='C2H6', smiles='CC')],
                          opt_level=level,
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )
        with patch.object(sched, 'warn_on_collapsible_unrestricted_reference') as reported:
            sched.run_job(label='C2H6', job_type='sp', level_of_theory=level, job_adapter='molpro')
        self.assertTrue(reported.called)
        self.assertEqual(reported.call_args.kwargs['label'], 'C2H6')
        self.assertIs(reported.call_args.kwargs['job'], job_mock)

    @patch('arc.scheduler.job_factory')
    def test_run_job_does_not_alias_level_args(self, mock_job_factory):
        """Test that run_job() passes a detached copy of the level args to the job."""
        job_mock = MagicMock()
        job_mock.job_name = 'opt_a0000'
        job_mock.server = None
        mock_job_factory.return_value = job_mock
        level = Level(method='wb97xd', basis='def2tzvp', software='gaussian',
                      args={'keyword': {'opt': 'opt=(verytight)'}})

        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_run_job_level_args')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='test_run_job_level_args', ess_settings=self.ess_settings,
                          species_list=[ARCSpecies(label='C2H6', smiles='CC')],
                          opt_level=level,
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )
        sched.run_job(label='C2H6', job_type='opt', level_of_theory=level, job_adapter='gaussian')

        args = mock_job_factory.call_args.kwargs['args']
        self.assertEqual(args['keyword'], {'opt': 'opt=(verytight)'})
        self.assertIsNot(args['keyword'], level.args['keyword'])
        args['keyword']['dft_grid'] = 'defgrid2'
        self.assertEqual(level.args, {'keyword': {'opt': 'opt=(verytight)'}, 'block': dict()})

    @patch('arc.scheduler.Scheduler.check_max_simultaneous_jobs_limit')
    @patch('arc.scheduler.job_factory')
    def test_run_job_records_the_remote_project_path_of_each_server(self, mock_job_factory, mock_limit):
        """Test that run_job() records where the project lives on every server it spawns a job on."""
        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_run_job_remote_paths')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='test_run_job_remote_paths', ess_settings=self.ess_settings,
                          species_list=[ARCSpecies(label='C2H6', smiles='CC')],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types1,
                          )
        self.assertEqual(sched.remote_project_paths, dict())

        for job_name, server, remote_project_path in [('opt_a0000', 'server1', 'runs/ARC_Projects/a_project'),
                                                      ('opt_a0001', 'server1', 'a_later_job_does_not_overwrite'),
                                                      ('opt_a0002', 'server2', 'runs/ARC_Projects/a_project'),
                                                      ('opt_a0003', 'local', None),
                                                      ('opt_a0004', None, 'no_server_is_not_recorded')]:
            job_mock = MagicMock()
            job_mock.job_name, job_mock.server = job_name, server
            job_mock.remote_project_path = remote_project_path
            mock_job_factory.return_value = job_mock
            sched.run_job(label='C2H6', job_type='opt',
                          level_of_theory=Level(repr=default_levels_of_theory['opt']), job_adapter='gaussian')

        self.assertEqual(sched.remote_project_paths, {'server1': 'runs/ARC_Projects/a_project',
                                                      'server2': 'runs/ARC_Projects/a_project'})


class StubJob(object):
    """
    A stand-in for a job object, so that Scheduler.run_job() can be driven without executing anything.
    A job of an unavailable adapter raises DependencyError from execute(), as the KinBot and
    AutoTST adapters do when their backend is not installed.
    """

    def __init__(self,
                 job_type: str,
                 job_adapter: str | None,
                 unavailable: bool = False,
                 raises: BaseException | None = None,
                 submitted_job_id: int | None = None,
                 partial_guess: tuple | None = None,
                 ):
        self.job_type = job_type
        self.job_adapter = job_adapter
        self.unavailable = unavailable
        self.raises = raises
        self.job_name = f'{job_type}_{job_adapter}'
        self.job_id = None
        self.submitted_job_id = submitted_job_id
        self.partial_guess = partial_guess
        self.execution_type = 'incore' if submitted_job_id is None else 'queue'
        self.server = None

    def as_dict(self) -> dict:
        """Return a dictionary representation of the job, used when saving the restart file."""
        return {'job_type': self.job_type, 'job_adapter': self.job_adapter}

    def execute(self) -> None:
        """Execute the job, raising if its external backend is not installed or if it was told to raise."""
        if self.unavailable:
            raise DependencyError(f'The {self.job_adapter} python executable was not found.')
        if self.submitted_job_id is not None:
            self.job_id = self.submitted_job_id  # The job reached the queue before raising.
        if self.partial_guess is not None:
            ts_species, guess = self.partial_guess
            ts_species.ts_guesses.append(guess)  # The adapter saved a guess before raising.
        if self.raises is not None:
            raise self.raises


class TestSpawnTsJobsAdmission(unittest.TestCase):
    """
    Contains unit tests for the TS adapter admission logic of Scheduler.spawn_ts_jobs().
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.ess_settings = {'gaussian': ['server1']}
        cls.projects = [f'arc_project_for_testing_delete_after_usage_tsg_{i}' for i in range(1, 25)] \
            + [f'arc_project_for_testing_delete_after_usage_tsg_save_{i}' for i in range(1, 3)]

    @classmethod
    def tearDownClass(cls):
        """
        A method that is run after all unit tests in this class, deleting the project directories.
        """
        for project in cls.projects:
            shutil.rmtree(os.path.join(ARC_PATH, 'Projects', project), ignore_errors=True)

    def test_spawn_ts_jobs_unknown_family_admission_predicate(self):
        """Test the admission predicate for TS adapters of reactions with an unknown family."""
        self.assertIn('linear', ts_adapters_for_unknown_unimolecular)
        self.assertIn('linear', default_incore_adapters)
        reactant_xyz = """C  -1.3087    0.0068    0.0318
                          C   0.1715   -0.0344    0.0210
                          N   0.9054   -0.9001    0.6395
                          O   2.1683   -0.5483    0.3437
                          N   2.1499    0.5449   -0.4631
                          N   0.9613    0.8655   -0.6660
                          H  -1.6558    0.9505    0.4530
                          H  -1.6934   -0.0680   -0.9854
                          H  -1.6986   -0.8169    0.6255"""
        reactant = ARCSpecies(label='azide_r', smiles='C([C]1=[N]O[N]=[N]1)', xyz=reactant_xyz)
        product_xyz = """C  -1.0108   -0.0114   -0.0610
                         C   0.4780    0.0191    0.0139
                         N   1.2974   -0.9930    0.4693
                         O   0.6928   -1.9845    0.8337
                         N   1.7456    1.9701   -0.6976
                         N   1.1642    1.0763   -0.3716
                         H  -1.4020    0.9134   -0.4821
                         H  -1.3327   -0.8499   -0.6803
                         H  -1.4329   -0.1554    0.9349"""
        product = ARCSpecies(label='azide_p', smiles='[N-]=[N+]=C(N=O)C', xyz=product_xyz)
        rxn_unimolecular = ARCReaction(r_species=[reactant], p_species=[product])
        self.assertIsNone(rxn_unimolecular.family)
        rxn_bimolecular = ARCReaction(r_species=[ARCSpecies(label='H', smiles='[H]'),
                                                 ARCSpecies(label='CH4', smiles='C')],
                                      p_species=[ARCSpecies(label='H2', smiles='[H][H]'),
                                                 ARCSpecies(label='CH3', smiles='[CH3]')])
        self.assertEqual(rxn_bimolecular.family, 'H_Abstraction')
        # Replicates the three-clause admission predicate from Scheduler.spawn_ts_jobs() (scheduler.py).
        for rxn, expected_admission in [(rxn_unimolecular, True), (rxn_bimolecular, False)]:
            family_known = rxn.family is not None and rxn.family in ts_adapters_by_rmg_family
            admit_unknown_family = (not family_known
                                    and 'linear' in ts_adapters_for_unknown_unimolecular
                                    and rxn.is_unimolecular())
            self.assertEqual(admit_unknown_family, expected_admission)

    def setup_hocho_scheduler(self,
                              ts_adapters: list,
                              project: str,
                              unavailable_adapters: list | None = None,
                              ) -> tuple:
        """
        Set up a Scheduler holding HOCHO <=> H2O + CO (family 1,2_Insertion_CO) with optimized wells.

        Only ``job_factory()`` is replaced, by one handing back ``StubJob``, so that
        ``spawn_ts_jobs()`` can be driven end to end without executing anything while ``run_job()``
        still does its real bookkeeping. A job for an adapter named in ``unavailable_adapters``
        raises ``DependencyError`` from ``execute()``, exactly as the KinBot and AutoTST adapters
        do when their backend is not installed -- that is, after ``run_job()`` has already
        registered the job in ``running_jobs`` and in ``job_dict``.

        Args:
            ts_adapters (list): The TS adapters to configure the Scheduler with.
            project (str): The project name, also the project directory name.
            unavailable_adapters (list, optional): Adapters whose job raises ``DependencyError``.

        Returns:
            tuple: The Scheduler, its ARCReaction, and the list accumulating the
                   (job type, adapter) pairs jobs were created for.
        """
        unavailable_adapters = unavailable_adapters or list()
        r_species = [ARCSpecies(label='HOCHO', smiles='OC=O')]
        p_species = [ARCSpecies(label='H2O', smiles='O'), ARCSpecies(label='CO', smiles='[C-]#[O+]')]
        for spc in r_species + p_species:
            spc.final_xyz = spc.get_xyz()  # Marks the well as optimized, gating spawn_ts_jobs().
        rxn = ARCReaction(r_species=r_species, p_species=p_species)
        self.assertEqual(rxn.family, '1,2_Insertion_CO')
        sched = Scheduler(project=project,
                          ess_settings=self.ess_settings,
                          species_list=r_species + p_species,
                          rxn_list=[rxn],
                          project_directory=os.path.join(ARC_PATH, 'Projects', project),
                          ts_adapters=ts_adapters,
                          job_types=initialize_job_types(),
                          conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          testing=True,
                          )
        spawned = list()

        def stub_job_factory(job_type, job_adapter=None, **kwargs):
            """Record the job being created and hand back a StubJob instead of a real job."""
            spawned.append((job_type, job_adapter))
            return StubJob(job_type=job_type, job_adapter=job_adapter,
                           unavailable=job_adapter in unavailable_adapters)

        # These tests drive the DependencyError backstop, so the engine check before it must pass on any host.
        for patcher in [patch('arc.scheduler.job_factory', new=stub_job_factory),
                        patch.object(kinbot_ts, 'KINBOT_PYTHON', __file__)]:
            patcher.start()
            self.addCleanup(patcher.stop)
        return sched, rxn, spawned

    def test_spawn_ts_jobs_warns_when_no_adapter_is_eligible(self):
        """Test that a reaction with no eligible configured adapter is reported, not skipped silently.

        Regression test for a real campaign failure: ``~/.arc/settings.py`` pinned
        ``ts_adapters = ['heuristics', 'AutoTST', 'GCN', 'xtb_gsm', 'orca_neb']`` (predating 'linear'
        joining ARC's own default list), while the reaction's family, ``1,2_Insertion_CO``, admits
        only ``['kinbot', 'goflow', 'rits', 'linear']``. The intersection was empty, so zero TS guess
        jobs were spawned -- silently. ``tsg_spawned`` latches True regardless, so nothing retried,
        and the TS guess report came back with successful_methods AND unsuccessful_methods both
        empty. The only user-visible symptom was 'TS did not converge', hours later.
        """
        configured = ['heuristics', 'autotst', 'gcn', 'xtb_gsm', 'orca_neb']
        sched, rxn, spawned = self.setup_hocho_scheduler(ts_adapters=configured,
                                                         project='arc_project_for_testing_delete_after_usage_tsg_1')
        eligible = [a.lower() for a in ts_adapters_by_rmg_family[rxn.family]]
        self.assertEqual([m for m in configured if m in eligible], list())
        self.assertIn('linear', eligible)  # 'linear' bridges them, a correctly-configured run is unaffected.

        with self.assertLogs('arc', level='WARNING') as captured:
            sched.spawn_ts_jobs()
        self.assertEqual([adapter for job_type, adapter in spawned if job_type == 'tsg'], list())
        self.assertTrue(any('Not spawning any TS search job' in record.getMessage()
                            and rxn.family in record.getMessage() for record in captured.records))
        self.assertTrue(rxn.ts_species.tsg_spawned)  # Latched, hence the single chance to report this.

    def test_spawn_ts_jobs_warns_when_ts_adapters_is_empty(self):
        """Test that an empty ts_adapters list warns instead of raising."""
        sched, rxn, spawned = self.setup_hocho_scheduler(ts_adapters=list(),
                                                         project='arc_project_for_testing_delete_after_usage_tsg_2')
        with self.assertLogs('arc', level='WARNING') as captured:
            sched.spawn_ts_jobs()
        self.assertEqual([adapter for job_type, adapter in spawned if job_type == 'tsg'], list())
        self.assertTrue(any('Not spawning any TS search job' in record.getMessage()
                            for record in captured.records))

    def test_spawn_ts_jobs_does_not_warn_when_a_user_guess_exists(self):
        """Test that a user-supplied TS guess suppresses the 'no TS guess will be generated' warning."""
        sched, rxn, spawned = self.setup_hocho_scheduler(ts_adapters=['heuristics'],
                                                         project='arc_project_for_testing_delete_after_usage_tsg_3')
        rxn.ts_species.ts_guesses.append(TSGuess(method='user guess 0',
                                                 xyz=rxn.r_species[0].get_xyz(),
                                                 index=0))
        with self.assertNoLogs('arc', level='WARNING'):
            sched.spawn_ts_jobs()
        self.assertEqual([adapter for job_type, adapter in spawned if job_type == 'tsg'], list())
        self.assertTrue(len(spawned))  # The user guess itself does proceed, so nothing is lost here.
        self.assertEqual(len(rxn.ts_species.ts_guesses), 1)

    def test_spawn_ts_jobs_survives_an_uninstalled_optional_adapter(self):
        """Test that an adapter whose backend is missing is recorded and skipped, not fatal.

        Regression test for a real campaign failure. ``ts_adapters`` listed 'kinbot', whose backend
        is optional and not installed on that host, so ``KinBotAdapter.execute_incore`` raised
        ``DependencyError('The KinBot python executable was not found...')``. That propagated
        through ``run_job`` -> ``spawn_ts_jobs`` -> ``spawn_post_opt_jobs`` and terminated the whole
        ARC run -- discarding the 7 TS guesses 'linear' had successfully produced for the same
        reaction moments earlier. A missing optional dependency must cost that adapter, not the run.
        """
        sched, rxn, spawned = self.setup_hocho_scheduler(ts_adapters=['kinbot', 'linear'],
                                                         project='arc_project_for_testing_delete_after_usage_tsg_4',
                                                         unavailable_adapters=['kinbot'])
        sched.spawn_ts_jobs()  # Must not raise: that is the whole point.
        # 'linear' still ran, the run was not lost.
        self.assertEqual([adapter for job_type, adapter in spawned if job_type == 'tsg'], ['kinbot', 'linear'])
        self.assertEqual(list(rxn.ts_species.skipped_methods), ['kinbot'])
        self.assertEqual(rxn.ts_species.unsuccessful_methods, list())
        # run_job() registers a job before executing it, so the failed adapter left bookkeeping behind
        # that must be rolled back: exactly one 'tsg0' entry, holding the job that actually ran.
        self.assertEqual(sched.running_jobs[rxn.ts_label], ['tsg0'])
        self.assertEqual(list(sched.job_dict[rxn.ts_label]['tsg'].keys()), [0])
        self.assertEqual(sched.job_dict[rxn.ts_label]['tsg'][0].job_adapter, 'linear')

    def test_spawn_ts_jobs_warns_when_every_eligible_adapter_is_unavailable(self):
        """Test that a reaction whose every eligible adapter is uninstalled is reported, and leaves no state."""
        sched, rxn, spawned = self.setup_hocho_scheduler(ts_adapters=['kinbot', 'linear'],
                                                         project='arc_project_for_testing_delete_after_usage_tsg_5',
                                                         unavailable_adapters=['kinbot', 'linear'])
        with self.assertLogs('arc', level='WARNING') as captured:
            sched.spawn_ts_jobs()
        self.assertEqual(list(rxn.ts_species.skipped_methods), ['kinbot', 'linear'])
        self.assertEqual(rxn.ts_species.unsuccessful_methods, list())
        self.assertTrue(any("all of its eligible adapters ['kinbot', 'linear'] are unavailable" in record.getMessage()
                            for record in captured.records))
        self.assertEqual(sched.running_jobs.get(rxn.ts_label, list()), list())
        self.assertEqual(sched.job_dict[rxn.ts_label].get('tsg', dict()), dict())

    def saved_ts_dict(self, sched: Scheduler, ts_label: str) -> dict:
        """
        Read the dictionary of a TS species from the restart file on disk.

        Args:
            sched (Scheduler): The Scheduler that saved the restart file.
            ts_label (str): The TS species label.

        Returns:
            dict: The saved dictionary of the TS species.
        """
        content = read_yaml_file(sched.restart_path)
        return next(spc for spc in content['species'] if spc['label'] == ts_label)

    def test_a_trailing_engine_absent_skip_is_saved(self):
        """Test that a skip recorded after the last dispatched adapter reaches the restart file."""
        sched, rxn, spawned = self.setup_hocho_scheduler(ts_adapters=['linear', 'kinbot'],
                                                         project='arc_project_for_testing_delete_after_usage_tsg_save_1')
        sched.restart_dict = dict()
        with patch.object(kinbot_ts, 'KINBOT_PYTHON', None):
            sched.spawn_ts_jobs()
        self.assertEqual(spawned, [('tsg', 'linear')])
        self.assertIn('kinbot', self.saved_ts_dict(sched, rxn.ts_label).get('skipped_methods', dict()))

    def test_a_trailing_dependency_error_skip_is_saved(self):
        """Test that a skip recorded by the DependencyError rollback after the last dispatched adapter is saved."""
        sched, rxn, spawned = self.setup_hocho_scheduler(ts_adapters=['linear', 'kinbot'],
                                                         project='arc_project_for_testing_delete_after_usage_tsg_save_2',
                                                         unavailable_adapters=['kinbot'])
        sched.restart_dict = dict()
        sched.spawn_ts_jobs()
        self.assertEqual(spawned, [('tsg', 'linear'), ('tsg', 'kinbot')])
        self.assertEqual(sched.running_jobs[rxn.ts_label], ['tsg0'])
        self.assertIn('kinbot', self.saved_ts_dict(sched, rxn.ts_label).get('skipped_methods', dict()))


    def setup_absent_engine_scheduler(self,
                                      rxns: list,
                                      ts_adapters: list,
                                      project: str,
                                      absent_at_start: bool = False,
                                      ) -> tuple:
        """
        Set up a Scheduler whose engine-backed TS adapters all have their engine absent.
        With ``absent_at_start`` the engines are already absent when the Scheduler starts; otherwise
        every adapter is reported available at start, and the engines go missing afterwards.

        Engine-free adapters ('heuristics', 'linear') get a ``StubJob``; any other adapter that reaches
        ``job_factory()`` gets the real adapter, so a missing engine meets the adapter's real code.
        The engine of every engine-backed adapter is made absent by pointing its executable setting to
        ``None``, and Orca NEB has no server in the ESS settings. xTB-GSM's command runner fails the
        test if called, since an absent engine must never be executed.

        Args:
            rxns (list): The ARCReaction objects; their wells are marked as optimized.
            ts_adapters (list): The TS adapters to configure the Scheduler with.
            project (str): The project name, also the project directory name.
            absent_at_start (bool, optional): Whether the engines are already absent when the Scheduler starts.

        Returns:
            tuple: The Scheduler and the list accumulating the (job type, adapter) pairs jobs were created for.
        """
        species = [spc for rxn in rxns for spc in rxn.r_species + rxn.p_species]
        for spc in species:
            spc.final_xyz = spc.get_xyz()
        for index, rxn in enumerate(rxns):
            rxn.index = index  # As ARC does, so that each reaction gets its own TS label.
        spawned = list()

        def hybrid_job_factory(job_type, job_adapter=None, **kwargs):
            """Record the job, stub the engine-free adapters, and build the real adapter otherwise."""
            spawned.append((job_type, job_adapter))
            if job_adapter in ['heuristics', 'linear']:
                return StubJob(job_type=job_type, job_adapter=job_adapter)
            return job_factory(job_type=job_type, job_adapter=job_adapter, **kwargs)

        def no_execution(*args, **kwargs):
            """Fail if an absent engine is executed."""
            raise AssertionError(f'An absent engine was executed: {args} {kwargs}')

        patchers = [patch('arc.scheduler.job_factory', new=hybrid_job_factory),
                    patch.object(autotst_ts, 'AUTOTST_PYTHON', None),
                    patch.object(kinbot_ts, 'KINBOT_PYTHON', None),
                    patch.object(gcn_ts, 'TS_GCN_PYTHON', None),
                    patch.object(goflow_ts, 'GOFLOW_PYTHON', None),
                    patch.object(rits_ts, 'RITS_PYTHON', None),
                    patch.object(xtb_gsm, 'XTB', None, create=True),
                    patch.object(xtb_gsm, 'execute_command', new=no_execution),
                    ]
        if absent_at_start:
            for patcher in patchers:
                patcher.start()
                self.addCleanup(patcher.stop)
            sched = self.make_ts_scheduler(project=project, species=species, rxns=rxns, ts_adapters=ts_adapters)
        else:
            with patch('arc.scheduler.ts_adapter_unavailable_reason', return_value=None):
                sched = self.make_ts_scheduler(project=project, species=species, rxns=rxns, ts_adapters=ts_adapters)
            for patcher in patchers:
                patcher.start()
                self.addCleanup(patcher.stop)
        return sched, spawned

    def make_ts_scheduler(self, project: str, species: list, rxns: list, ts_adapters: list | None) -> Scheduler:
        """
        Make a testing Scheduler for TS search tests.

        Args:
            project (str): The project name, also the project directory name.
            species (list): The ARCSpecies objects.
            rxns (list): The ARCReaction objects.
            ts_adapters (list): The TS adapters to configure the Scheduler with.

        Returns:
            Scheduler: The Scheduler.
        """
        return Scheduler(project=project,
                         ess_settings=self.ess_settings,
                         species_list=species,
                         rxn_list=rxns,
                         project_directory=os.path.join(ARC_PATH, 'Projects', project),
                         ts_adapters=ts_adapters,
                         job_types=initialize_job_types(),
                         conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                         opt_level=Level(repr=default_levels_of_theory['opt']),
                         testing=True,
                         )

    def test_spawn_ts_jobs_drops_each_adapter_whose_engine_is_absent(self):
        """Test that an adapter whose engine went missing after the run started is dropped with one named warning.

        Covers every engine-backed adapter in the family map, and an adapter the map lists that has
        no module in this ARC build (as 'crest' had). The latter is injected into the map here, so the
        test does not depend on which unregistered names the map happens to carry.
        """
        unregistered = 'absent_unregistered_ts_adapter'
        cases = [('autotst', 'AUTOTST_PYTHON', 'intra_H_migration'),
                 ('kinbot', 'KINBOT_PYTHON', 'intra_H_migration'),
                 ('gcn', 'TS_GCN_PYTHON', 'intra_H_migration'),
                 ('goflow', 'GOFLOW_PYTHON', 'intra_H_migration'),
                 ('rits', 'RITS_PYTHON', 'intra_H_migration'),
                 ('xtb_gsm', 'XTB', 'intra_H_migration'),
                 ('orca_neb', 'ESS settings', 'intra_H_migration'),
                 ('autotst', 'AUTOTST_PYTHON', 'H_Abstraction'),
                 (unregistered, 'registered', 'H_Abstraction'),
                 ]
        family_map = {'H_Abstraction': ts_adapters_by_rmg_family['H_Abstraction'] + [unregistered]}
        for i, (adapter, cause, family) in enumerate(cases):
            with self.subTest(adapter=adapter, family=family), patch.dict(ts_adapters_by_rmg_family, family_map):
                if family == 'H_Abstraction':
                    rxn = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='OH', smiles='[OH]')],
                                      p_species=[ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='H2O', smiles='O')])
                    engine_free = 'heuristics'
                else:
                    rxn = ARCReaction(r_species=[ARCSpecies(label='CH3CH2NH', smiles='CC[NH]')],
                                      p_species=[ARCSpecies(label='CH2CH2NH2', smiles='[CH2]CN')])
                    engine_free = 'linear'
                self.assertEqual(rxn.family, family)
                self.assertIn(adapter, ts_adapters_by_rmg_family[family])
                sched, spawned = self.setup_absent_engine_scheduler(
                    rxns=[rxn], ts_adapters=[adapter, engine_free],
                    project=f'arc_project_for_testing_delete_after_usage_tsg_6_{i}')
                self.projects.append(f'arc_project_for_testing_delete_after_usage_tsg_6_{i}')
                with self.assertLogs('arc', level='WARNING') as captured:
                    sched.spawn_ts_jobs()
                self.assertEqual(len(captured.records), 1, msg=[r.getMessage() for r in captured.records])
                record = captured.records[0]
                self.assertEqual(record.levelno, logging.WARNING)
                for token in [repr(adapter), rxn.label, cause]:
                    self.assertIn(token, record.getMessage())
                self.assertNotIn(('tsg', adapter), spawned)
                self.assertEqual(sched.running_jobs[rxn.ts_label], ['tsg0'])
                self.assertEqual({k: v.job_adapter for k, v in sched.job_dict[rxn.ts_label]['tsg'].items()},
                                 {0: engine_free})
                self.assertEqual(list(rxn.ts_species.skipped_methods), [adapter])
                self.assertIn(cause, rxn.ts_species.skipped_methods[adapter])
                self.assertEqual(rxn.ts_species.unsuccessful_methods, list())
                self.assertFalse(sched.output[rxn.ts_label]['errors'])

    def test_spawn_ts_jobs_with_no_available_adapter_ends_only_that_reaction(self):
        """Test that a reaction whose every eligible adapter is absent ends with a named error, while another goes on."""
        unregistered = 'absent_unregistered_ts_adapter'
        family_map = {'H_Abstraction': ts_adapters_by_rmg_family['H_Abstraction'] + [unregistered]}
        with patch.dict(ts_adapters_by_rmg_family, family_map):
            rxn_a = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='OH', smiles='[OH]')],
                                p_species=[ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='H2O', smiles='O')])
            rxn_b = ARCReaction(r_species=[ARCSpecies(label='CH3CH2NH', smiles='CC[NH]')],
                                p_species=[ARCSpecies(label='CH2CH2NH2', smiles='[CH2]CN')])
            sched, spawned = self.setup_absent_engine_scheduler(
                rxns=[rxn_a, rxn_b], ts_adapters=['autotst', unregistered, 'linear'],
                project='arc_project_for_testing_delete_after_usage_tsg_7')
            with self.assertLogs('arc', level='WARNING') as captured:
                sched.spawn_ts_jobs()
        errors = [r.getMessage() for r in captured.records if r.levelno == logging.ERROR]
        self.assertEqual(len(errors), 1, msg=errors)
        for token in [rxn_a.label, "'autotst'", repr(unregistered), 'AUTOTST_PYTHON', 'registered']:
            self.assertIn(token, errors[0])
            self.assertIn(token, sched.output[rxn_a.ts_label]['errors'])
        self.assertFalse(sched.output[rxn_a.ts_label]['convergence'])
        self.assertTrue(sched.species_dict[rxn_a.ts_label].ts_guesses_exhausted)
        self.assertEqual(sched.job_dict[rxn_a.ts_label].get('tsg', dict()), dict())
        # The other reaction goes on: its only available adapter ran, and it carries no error.
        self.assertEqual({k: v.job_adapter for k, v in sched.job_dict[rxn_b.ts_label]['tsg'].items()}, {0: 'linear'})
        self.assertEqual(list(rxn_b.ts_species.skipped_methods), ['autotst'])
        self.assertEqual(rxn_b.ts_species.unsuccessful_methods, list())
        self.assertFalse(sched.output[rxn_b.ts_label]['errors'])
        self.assertNotIn(('tsg', 'autotst'), spawned)

    def test_spawn_ts_jobs_with_no_available_adapter_saves_the_restart_file(self):
        """Test that a TS search ended for lack of an available adapter is saved to the restart file."""
        rxn = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='OH', smiles='[OH]')],
                          p_species=[ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='H2O', smiles='O')])
        sched, spawned = self.setup_absent_engine_scheduler(
            rxns=[rxn], ts_adapters=['autotst'], project='arc_project_for_testing_delete_after_usage_tsg_13')
        with patch.object(sched, 'save_restart_dict') as save_restart_dict:
            sched.spawn_ts_jobs()
        self.assertFalse(sched.output[rxn.ts_label]['convergence'])
        self.assertTrue(sched.species_dict[rxn.ts_label].ts_guesses_exhausted)
        self.assertEqual(spawned, list())
        save_restart_dict.assert_called()

    def test_spawn_ts_jobs_with_no_available_adapter_keeps_a_yaml_ts_converged(self):
        """Test that a TS loaded from an Arkane YAML file stays converged when no TS search adapter is available."""
        rxn = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='OH', smiles='[OH]')],
                          p_species=[ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='H2O', smiles='O')])
        sched, spawned = self.setup_absent_engine_scheduler(
            rxns=[rxn], ts_adapters=['autotst'], project='arc_project_for_testing_delete_after_usage_tsg_12')
        sched.species_dict[rxn.ts_label].yml_path = os.path.join(ARC_PATH, 'TS.yml')
        sched.output[rxn.ts_label]['convergence'] = True
        sched.spawn_ts_jobs()
        self.assertTrue(sched.output[rxn.ts_label]['convergence'])
        self.assertFalse(sched.output[rxn.ts_label]['errors'])
        self.assertFalse(sched.species_dict[rxn.ts_label].ts_guesses_exhausted)

    def test_engines_absent_at_start_are_reported_once_in_one_block(self):
        """Test that absent engines are listed once at run start with their install command, and not per reaction."""
        rxn = ARCReaction(r_species=[ARCSpecies(label='CH3CH2NH', smiles='CC[NH]')],
                          p_species=[ARCSpecies(label='CH2CH2NH2', smiles='[CH2]CN')])
        with self.assertLogs('arc', level='WARNING') as start_logs:
            sched, spawned = self.setup_absent_engine_scheduler(
                rxns=[rxn], ts_adapters=['autotst', 'kinbot', 'orca_neb', 'linear'],
                project='arc_project_for_testing_delete_after_usage_tsg_8', absent_at_start=True)
        blocks = [r.getMessage() for r in start_logs.records if 'not available on this machine' in r.getMessage()]
        self.assertEqual(len(blocks), 1, msg=[r.getMessage() for r in start_logs.records])
        for token in ["'autotst'", 'AUTOTST_PYTHON', 'make install-autotst', "'kinbot'", 'make install-kinbot',
                      "'orca_neb'", 'No make target']:
            self.assertIn(token, blocks[0])
        self.assertNotIn("'linear'", blocks[0])
        self.assertEqual(set(sched.absent_ts_adapters), {'autotst', 'kinbot', 'orca_neb'})
        with self.assertLogs('arc', level='INFO') as captured:
            sched.spawn_ts_jobs()
        warnings = [r.getMessage() for r in captured.records if r.levelno >= logging.WARNING]
        self.assertEqual(warnings, list())
        lines = [r.getMessage() for r in captured.records if rxn.label in r.getMessage() and 'running' in r.getMessage()]
        self.assertEqual(len(lines), 1, msg=[r.getMessage() for r in captured.records])
        self.assertIn("running ['linear']", lines[0])
        self.assertIn("skipped (engine absent): ['autotst', 'kinbot', 'orca_neb']", lines[0])
        self.assertEqual(set(rxn.ts_species.skipped_methods), {'autotst', 'kinbot', 'orca_neb'})
        self.assertIn('AUTOTST_PYTHON', rxn.ts_species.skipped_methods['autotst'])
        self.assertEqual(rxn.ts_species.unsuccessful_methods, list())
        self.assertEqual(spawned, [('tsg', 'linear')])

    def test_the_start_block_lists_only_adapters_this_run_could_dispatch(self):
        """Test that an absent engine no reaction in the run could use is left out of the run-start block."""
        rxn = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='OH', smiles='[OH]')],
                          p_species=[ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='H2O', smiles='O')])
        self.assertEqual(rxn.family, 'H_Abstraction')
        with self.assertLogs('arc', level='WARNING') as start_logs:
            sched, _ = self.setup_absent_engine_scheduler(
                rxns=[rxn], ts_adapters=['heuristics', 'autotst', 'gcn', 'xtb_gsm', 'kinbot', 'goflow', 'rits'],
                project='arc_project_for_testing_delete_after_usage_tsg_14', absent_at_start=True)
        self.assertEqual(set(sched.absent_ts_adapters), {'autotst'})
        blocks = [r.getMessage() for r in start_logs.records if 'not available on this machine' in r.getMessage()]
        self.assertEqual(len(blocks), 1, msg=[r.getMessage() for r in start_logs.records])
        self.assertIn("'autotst'", blocks[0])
        for adapter in ['gcn', 'xtb_gsm', 'kinbot', 'goflow', 'rits']:
            self.assertNotIn(repr(adapter), blocks[0])

    def test_resolving_again_drops_stale_absence_causes(self):
        """Test that a second resolve_ts_adapters() call does not keep the causes of the first."""
        rxn = ARCReaction(r_species=[ARCSpecies(label='CH3CH2NH', smiles='CC[NH]')],
                          p_species=[ARCSpecies(label='CH2CH2NH2', smiles='[CH2]CN')])
        sched, _ = self.setup_absent_engine_scheduler(rxns=[rxn], ts_adapters=['autotst', 'linear'],
                                                      project='arc_project_for_testing_delete_after_usage_tsg_15',
                                                      absent_at_start=True)
        self.assertEqual(set(sched.absent_ts_adapters), {'autotst'})
        sched.resolve_ts_adapters(['linear'])
        self.assertEqual(sched.absent_ts_adapters, dict())

    def test_a_changed_absence_cause_is_loud_for_the_reaction(self):
        """Test that an engine absent at run start for another cause is warned about for the reaction."""
        rxn = ARCReaction(r_species=[ARCSpecies(label='CH3CH2NH', smiles='CC[NH]')],
                          p_species=[ARCSpecies(label='CH2CH2NH2', smiles='[CH2]CN')])
        sched, _ = self.setup_absent_engine_scheduler(rxns=[rxn], ts_adapters=['autotst', 'linear'],
                                                      project='arc_project_for_testing_delete_after_usage_tsg_11',
                                                      absent_at_start=True)
        sched.absent_ts_adapters['autotst'] = 'an earlier cause'
        with self.assertLogs('arc', level='WARNING') as captured:
            sched.spawn_ts_jobs()
        self.assertEqual(len(captured.records), 1, msg=[r.getMessage() for r in captured.records])
        for token in ["'autotst'", rxn.label, 'AUTOTST_PYTHON']:
            self.assertIn(token, captured.records[0].getMessage())

    def test_no_absence_block_when_every_engine_is_present(self):
        """Test that no absence block is logged when every configured TS engine is present."""
        rxn = ARCReaction(r_species=[ARCSpecies(label='CH3CH2NH', smiles='CC[NH]')],
                          p_species=[ARCSpecies(label='CH2CH2NH2', smiles='[CH2]CN')])
        sched, _ = self.setup_absent_engine_scheduler(rxns=[rxn], ts_adapters=['linear', 'heuristics'],
                                                      project='arc_project_for_testing_delete_after_usage_tsg_9',
                                                      absent_at_start=True)
        self.assertEqual(sched.absent_ts_adapters, dict())

    def test_dependency_error_backstop_records_a_skip_and_stays_loud(self):
        """Test that an engine that passes the check but raises DependencyError is a loud skip, not a failure."""
        sched, rxn, spawned = self.setup_hocho_scheduler(
            ts_adapters=['kinbot', 'linear'], unavailable_adapters=['kinbot'],
            project='arc_project_for_testing_delete_after_usage_tsg_10')
        with self.assertLogs('arc', level='ERROR') as captured:
            sched.spawn_ts_jobs()
        self.assertTrue(any("'kinbot'" in r.getMessage() and rxn.label in r.getMessage() for r in captured.records))
        self.assertIn('kinbot', rxn.ts_species.skipped_methods)
        self.assertEqual(rxn.ts_species.unsuccessful_methods, list())

    def setup_two_reaction_scheduler(self,
                                     project: str,
                                     raising: dict,
                                     submitted: dict | None = None,
                                     partial_guesses: dict | None = None,
                                     ts_adapters: list | None = None,
                                     ) -> tuple:
        """
        Set up a Scheduler holding two 1,2_Insertion_CO reactions with optimized wells, whose TS adapter
        jobs raise the exceptions given in ``raising``.

        Args:
            project (str): The project name, also the project directory name.
            raising (dict): Keys are (adapter, reactant label) pairs, values are the exception that adapter's
                            job raises from ``execute()`` for the reaction of that reactant.
            submitted (dict, optional): Keys are (adapter, reactant label) pairs, values are the queue job ID
                                        that adapter's job got before it raised.
            partial_guesses (dict, optional): Keys are (adapter, reactant label) pairs, values are a TSGuess that
                                              adapter's job saves on its reaction's TS species before it raises.
            ts_adapters (list, optional): The TS adapters, ['kinbot', 'linear'] by default.

        Returns:
            tuple: The Scheduler, the HOCHO reaction, the CH3CHO reaction, and the list accumulating the
                   (job type, adapter, reactant label) triples jobs were created for.
        """
        hocho, ch3cho = ARCSpecies(label='HOCHO', smiles='OC=O'), ARCSpecies(label='CH3CHO', smiles='CC=O')
        h2o, ch4, co = (ARCSpecies(label='H2O', smiles='O'), ARCSpecies(label='CH4', smiles='C'),
                        ARCSpecies(label='CO', smiles='[C-]#[O+]'))
        species = [hocho, ch3cho, h2o, ch4, co]
        for spc in species:
            spc.final_xyz = spc.get_xyz()  # Marks the well as optimized, gating spawn_ts_jobs().
        rxn_1 = ARCReaction(r_species=[hocho], p_species=[h2o, co])
        rxn_2 = ARCReaction(r_species=[ch3cho], p_species=[ch4, co])
        rxn_1.index, rxn_2.index = 0, 1  # ARC assigns these; they make the two TS labels distinct.
        self.assertEqual(rxn_1.family, '1,2_Insertion_CO')
        self.assertEqual(rxn_2.family, '1,2_Insertion_CO')
        sched = Scheduler(project=project,
                          ess_settings=self.ess_settings,
                          species_list=species,
                          rxn_list=[rxn_1, rxn_2],
                          project_directory=os.path.join(ARC_PATH, 'Projects', project),
                          ts_adapters=ts_adapters or ['kinbot', 'linear'],
                          job_types=initialize_job_types(),
                          conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          testing=True,
                          )
        spawned = list()

        def stub_job_factory(job_type, job_adapter=None, reactions=None, **kwargs):
            """Record the job being created and hand back a StubJob that raises as instructed."""
            reactant = reactions[0].r_species[0].label if reactions else None
            spawned.append((job_type, job_adapter, reactant))
            guess = (partial_guesses or dict()).get((job_adapter, reactant))
            return StubJob(job_type=job_type, job_adapter=job_adapter, raises=raising.get((job_adapter, reactant)),
                           submitted_job_id=(submitted or dict()).get((job_adapter, reactant)),
                           partial_guess=(reactions[0].ts_species, guess) if guess is not None else None)

        for patcher in [patch('arc.scheduler.job_factory', new=stub_job_factory),
                        patch.object(kinbot_ts, 'KINBOT_PYTHON', __file__)]:
            patcher.start()
            self.addCleanup(patcher.stop)
        return sched, rxn_1, rxn_2, spawned

    def test_spawn_ts_jobs_isolates_an_adapter_exception_to_its_reaction(self):
        """Test that an adapter raising for one reaction costs that adapter for that reaction, not the run.

        Regression test for a real pilot failure: the heuristics adapter raised ``KeyError: None`` deep in
        its nitrile hydrolysis geometry builder for HCN + H2O. Nothing caught it, so the whole ARC process
        died for every reaction in the project, and a restart repeated it.
        """
        sched, rxn_1, rxn_2, spawned = self.setup_two_reaction_scheduler(
            project='arc_project_for_testing_delete_after_usage_tsg_16',
            raising={('kinbot', 'HOCHO'): KeyError(None)})
        with self.assertLogs('arc', level='ERROR') as captured:
            sched.spawn_ts_jobs()  # Must not raise.

        self.assertEqual([(adapter, reactant) for job_type, adapter, reactant in spawned if job_type == 'tsg'],
                         [('kinbot', 'HOCHO'), ('linear', 'HOCHO'), ('kinbot', 'CH3CHO'), ('linear', 'CH3CHO')])
        # The other reaction's TS guess stage is untouched.
        self.assertEqual(rxn_2.ts_species.ts_adapter_failures, list())
        self.assertEqual(rxn_2.ts_species.unsuccessful_methods, list())
        self.assertEqual(sched.running_jobs[rxn_2.ts_label], ['tsg0', 'tsg1'])
        self.assertEqual([job.job_adapter for job in sched.job_dict[rxn_2.ts_label]['tsg'].values()],
                         ['kinbot', 'linear'])
        # The failing reaction recorded the failure, rolled the job back, and went on with 'linear'.
        failure = {'adapter': 'kinbot', 'exception_type': 'KeyError', 'message': 'None'}
        self.assertEqual(rxn_1.ts_species.ts_adapter_failures, [failure])
        self.assertEqual(rxn_1.ts_species.unsuccessful_methods, ['kinbot'])
        self.assertEqual(sched.running_jobs[rxn_1.ts_label], ['tsg0'])
        self.assertEqual(sched.job_dict[rxn_1.ts_label]['tsg'][0].job_adapter, 'linear')
        # The ERROR record names the reaction and the adapter, and carries the traceback.
        errors = [record.getMessage() for record in captured.records if record.levelno == logging.ERROR]
        self.assertTrue(any(rxn_1.label in message and "'kinbot'" in message and 'Traceback' in message
                            and 'KeyError' in message for message in errors), errors)

    def test_spawn_ts_jobs_keeps_tracking_a_submitted_job_that_raised(self):
        """Test that a queue job that was submitted before its adapter raised is recorded but stays tracked."""
        sched, rxn_1, rxn_2, spawned = self.setup_two_reaction_scheduler(
            project='arc_project_for_testing_delete_after_usage_tsg_17',
            raising={('kinbot', 'HOCHO'): RuntimeError('lost the connection after submitting')},
            submitted={('kinbot', 'HOCHO'): 4321})
        with self.assertLogs('arc', level='ERROR') as captured:
            sched.spawn_ts_jobs()
        self.assertTrue(any('its submitted job stays tracked' in record.getMessage() for record in captured.records))
        self.assertEqual(rxn_1.ts_species.ts_adapter_failures,
                         [{'adapter': 'kinbot', 'exception_type': 'RuntimeError',
                           'message': 'lost the connection after submitting'}])
        self.assertEqual(rxn_1.ts_species.unsuccessful_methods, list())
        self.assertEqual(sched.running_jobs[rxn_1.ts_label], ['tsg0', 'tsg1'])
        self.assertEqual(sched.job_dict[rxn_1.ts_label]['tsg'][0].job_id, 4321)
        self.assertEqual(sched.job_dict[rxn_1.ts_label]['tsg'][1].job_adapter, 'linear')

    def test_spawn_ts_jobs_starts_conformers_for_guesses_an_adapter_saved_before_raising(self):
        """Test that a guess saved by an adapter that then raised is still optimized when no tracked job remains.

        Rolling the failed job back removes the one job whose completion would have started the TS conformer
        jobs, so without this the saved guess would never be tried.
        """
        guess = TSGuess(method='kinbot', xyz=ARCSpecies(label='HOCHO', smiles='OC=O').get_xyz(), index=0,
                        success=True)
        sched, rxn_1, rxn_2, spawned = self.setup_two_reaction_scheduler(
            project='arc_project_for_testing_delete_after_usage_tsg_18',
            raising={('kinbot', 'HOCHO'): RuntimeError('the reverse run failed')},
            partial_guesses={('kinbot', 'HOCHO'): guess},
            ts_adapters=['kinbot'])
        with self.assertLogs('arc', level='ERROR'):
            sched.spawn_ts_jobs()
        self.assertEqual(rxn_1.ts_species.ts_guesses, [guess])
        self.assertTrue(rxn_1.ts_species.ts_conf_spawned)
        self.assertEqual(sched.running_jobs[rxn_1.ts_label], ['opt_gaussian'])
        self.assertEqual(list(sched.job_dict[rxn_1.ts_label]['opt'].keys()), ['opt_gaussian'])

    def test_spawn_ts_jobs_lets_keyboard_interrupt_and_system_exit_propagate(self):
        """Test that the adapter exception boundary never swallows KeyboardInterrupt or SystemExit."""
        for i, exception in ((19, KeyboardInterrupt()), (20, SystemExit(1))):
            with self.subTest(exception=type(exception).__name__):
                sched, rxn_1, rxn_2, spawned = self.setup_two_reaction_scheduler(
                    project=f'arc_project_for_testing_delete_after_usage_tsg_{i}',
                    raising={('kinbot', 'HOCHO'): exception})
                with self.assertRaises(type(exception)):
                    sched.spawn_ts_jobs()
                self.assertEqual(rxn_1.ts_species.ts_adapter_failures, list())

    def test_spawn_ts_jobs_reports_when_every_adapter_raised(self):
        """Test that a reaction whose every adapter raised ends through the no-guess path and leaves no state."""
        sched, rxn_1, rxn_2, spawned = self.setup_two_reaction_scheduler(
            project='arc_project_for_testing_delete_after_usage_tsg_21',
            raising={('kinbot', 'HOCHO'): ValueError('first line\nsecond line'),
                     ('linear', 'HOCHO'): RuntimeError()})
        with self.assertLogs('arc', level='WARNING') as captured:
            sched.spawn_ts_jobs()
        self.assertEqual(rxn_1.ts_species.ts_adapter_failures,
                         [{'adapter': 'kinbot', 'exception_type': 'ValueError', 'message': 'first line'},
                          {'adapter': 'linear', 'exception_type': 'RuntimeError', 'message': 'RuntimeError'}])
        self.assertEqual(sched.running_jobs.get(rxn_1.ts_label, list()), list())
        self.assertEqual(sched.job_dict[rxn_1.ts_label].get('tsg', dict()), dict())
        self.assertTrue(any('Not spawning any TS search job' in record.getMessage()
                            and "['kinbot', 'linear'] raised an exception" in record.getMessage()
                            for record in captured.records))
        self.assertEqual(sched.running_jobs[rxn_2.ts_label], ['tsg0', 'tsg1'])

    def test_spawn_ts_jobs_ends_only_the_reaction_whose_every_adapter_raised(self):
        """Test that ending the TS search of a reaction whose every adapter raised leaves the other reaction going."""
        sched, rxn_1, rxn_2, spawned = self.setup_two_reaction_scheduler(
            project='arc_project_for_testing_delete_after_usage_tsg_22',
            raising={('kinbot', 'HOCHO'): ValueError('kinbot failed'),
                     ('linear', 'HOCHO'): RuntimeError('linear failed')})
        with patch.object(sched, 'end_ts_search_without_guesses',
                          wraps=sched.end_ts_search_without_guesses) as end_search, \
                self.assertLogs('arc', level='ERROR'):
            sched.spawn_ts_jobs()
        self.assertEqual([call.kwargs['label'] for call in end_search.call_args_list], [rxn_1.ts_label])
        self.assertIs(sched.output[rxn_1.ts_label]['convergence'], False)
        self.assertTrue(sched.species_dict[rxn_1.ts_label].ts_guesses_exhausted)
        self.assertIn('Not spawning any TS search job', sched.output[rxn_1.ts_label]['errors'])
        self.assertIn(rxn_1.label, sched.output[rxn_1.ts_label]['errors'])
        self.assertEqual(sched.running_jobs[rxn_2.ts_label], ['tsg0', 'tsg1'])
        self.assertEqual([job.job_adapter for job in sched.job_dict[rxn_2.ts_label]['tsg'].values()],
                         ['kinbot', 'linear'])
        self.assertIsNot(sched.output[rxn_2.ts_label]['convergence'], False)
        self.assertFalse(sched.output[rxn_2.ts_label]['errors'])
        self.assertFalse(sched.species_dict[rxn_2.ts_label].ts_guesses_exhausted)

    def test_spawn_ts_jobs_does_not_end_a_search_whose_raising_adapter_saved_a_guess(self):
        """Test that a guess saved by an adapter that then raised starts the TS conformer jobs even if every
        other eligible adapter raised, and that the TS search of that reaction is not ended."""
        guess = TSGuess(method='kinbot', xyz=ARCSpecies(label='HOCHO', smiles='OC=O').get_xyz(), index=0,
                        success=True)
        sched, rxn_1, rxn_2, spawned = self.setup_two_reaction_scheduler(
            project='arc_project_for_testing_delete_after_usage_tsg_23',
            raising={('kinbot', 'HOCHO'): RuntimeError('the reverse run failed'),
                     ('linear', 'HOCHO'): RuntimeError('linear failed')},
            partial_guesses={('kinbot', 'HOCHO'): guess})
        with patch.object(sched, 'end_ts_search_without_guesses',
                          wraps=sched.end_ts_search_without_guesses) as end_search, \
                self.assertLogs('arc', level='ERROR'):
            sched.spawn_ts_jobs()
        self.assertEqual([failure['adapter'] for failure in rxn_1.ts_species.ts_adapter_failures], ['kinbot', 'linear'])
        end_search.assert_not_called()
        self.assertEqual(rxn_1.ts_species.ts_guesses, [guess])
        self.assertTrue(rxn_1.ts_species.ts_conf_spawned)
        self.assertEqual(sched.running_jobs[rxn_1.ts_label], ['opt_gaussian'])
        self.assertIsNot(sched.output[rxn_1.ts_label]['convergence'], False)
        self.assertFalse(sched.output[rxn_1.ts_label]['errors'])
        self.assertFalse(sched.species_dict[rxn_1.ts_label].ts_guesses_exhausted)

    def test_spawn_ts_jobs_does_not_end_a_search_while_an_adapter_job_is_tracked(self):
        """Test that an adapter that ran without raising keeps its job tracked, so the no-search message never
        fires for it; whether it found a guess is settled when its job completes."""
        sched, rxn_1, rxn_2, spawned = self.setup_two_reaction_scheduler(
            project='arc_project_for_testing_delete_after_usage_tsg_24',
            raising={('kinbot', 'HOCHO'): RuntimeError('kinbot failed')})
        with patch.object(sched, 'end_ts_search_without_guesses',
                          wraps=sched.end_ts_search_without_guesses) as end_search, \
                self.assertLogs('arc', level='ERROR'):
            sched.spawn_ts_jobs()
        end_search.assert_not_called()
        self.assertEqual(rxn_1.ts_species.ts_guesses, list())
        self.assertEqual(sched.running_jobs[rxn_1.ts_label], ['tsg0'])
        self.assertEqual(sched.job_dict[rxn_1.ts_label]['tsg'][0].job_adapter, 'linear')
        self.assertFalse(sched.output[rxn_1.ts_label]['errors'])


class TestSchedulerTsAdaptersSource(unittest.TestCase):
    """
    Contains unit tests for how the Scheduler resolves, logs and records its effective ts_adapters.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.projects = [f'arc_project_for_testing_delete_after_usage_ts_source_{i}' for i in range(4)]

    @classmethod
    def tearDownClass(cls):
        """
        A method that is run after all unit tests in this class, deleting the project directories.
        """
        for project in cls.projects:
            shutil.rmtree(os.path.join(ARC_PATH, 'Projects', project), ignore_errors=True)

    def make_scheduler(self, index: int, ts_adapters: list | None) -> Scheduler:
        """
        Make a testing Scheduler with one species and no reaction.

        Args:
            index (int): The project index.
            ts_adapters (list): The ts_adapters argument, as main.ARC passes it from the input file.

        Returns:
            Scheduler: The Scheduler.
        """
        return Scheduler(project=self.projects[index],
                         ess_settings={'gaussian': ['server1']},
                         species_list=[ARCSpecies(label='H2O', smiles='O')],
                         project_directory=os.path.join(ARC_PATH, 'Projects', self.projects[index]),
                         ts_adapters=ts_adapters,
                         job_types=initialize_job_types(),
                         conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                         opt_level=Level(repr=default_levels_of_theory['opt']),
                         testing=True,
                         )

    def test_repo_default_is_named_and_logged(self):
        """Test that a run without an override uses and names the repo default."""
        with patch('arc.scheduler.local_settings_keys', frozenset()), \
                patch('arc.scheduler.default_ts_adapters', list(repo_settings.ts_adapters)), \
                self.assertLogs('arc', level='INFO') as captured:
            sched = self.make_scheduler(0, None)
        self.assertEqual(sched.ts_adapters, repo_settings.ts_adapters)
        self.assertEqual(sched.ts_adapters_source, 'repo default')
        lines = [r.getMessage() for r in captured.records if 'TS adapters for this run' in r.getMessage()]
        self.assertEqual(len(lines), 1)
        self.assertIn('repo default', lines[0])
        self.assertIn(str(repo_settings.ts_adapters), lines[0])
        self.assertFalse([r for r in captured.records if r.levelno >= logging.WARNING and 'leaves out' in r.getMessage()])

    def test_local_settings_that_drop_linear_get_one_warning(self):
        """Test that ~/.arc/settings.py dropping 'linear' is named as the source and warned about once."""
        local = ['heuristics', 'AutoTST', 'GCN', 'xtb_gsm', 'orca_neb']
        with patch('arc.scheduler.local_settings_keys', frozenset({'ts_adapters'})), \
                patch('arc.scheduler.default_ts_adapters', local), \
                self.assertLogs('arc', level='INFO') as captured:
            sched = self.make_scheduler(1, None)
        self.assertEqual(sched.ts_adapters, ['heuristics', 'autotst', 'gcn', 'xtb_gsm', 'orca_neb'])
        self.assertEqual(sched.ts_adapters_source, '~/.arc/settings.py')
        warnings = [r.getMessage() for r in captured.records if 'leaves out' in r.getMessage()]
        self.assertEqual(len(warnings), 1)
        self.assertEqual([r.levelno for r in captured.records if 'leaves out' in r.getMessage()], [logging.WARNING])
        self.assertNotIn("'rits'", warnings[0])
        for token in ['~/.arc/settings.py', "'linear'", "'kinbot'", "'goflow'"]:
            self.assertIn(token, warnings[0])

    def test_input_file_override_is_named(self):
        """Test that an input-file ts_adapters list is named as the source and its omissions warned about once."""
        with patch('arc.scheduler.local_settings_keys', frozenset({'ts_adapters'})), \
                self.assertLogs('arc', level='INFO') as captured:
            sched = self.make_scheduler(2, ['Heuristics'])
        self.assertEqual(sched.ts_adapters, ['heuristics'])
        self.assertEqual(sched.ts_adapters_source, 'input file')
        warnings = [r.getMessage() for r in captured.records if 'leaves out' in r.getMessage()]
        self.assertEqual(len(warnings), 1)
        self.assertIn('input file', warnings[0])
        self.assertIn("'linear'", warnings[0])

    def test_a_wrong_type_is_refused(self):
        """Test that a ts_adapters value that is not a list of strings stops the Scheduler, naming its source."""
        with self.assertRaisesRegex(InputError, 'input file'):
            self.make_scheduler(3, 'linear')


class TestExceptionSummaryLine(unittest.TestCase):
    """Contains unit tests for exception_summary_line()."""

    def test_exception_summary_line(self):
        """The summary is the first non-empty message line, the type name for an empty message, and is capped."""
        self.assertEqual(exception_summary_line(KeyError(None)), 'None')
        self.assertEqual(exception_summary_line(ValueError('\n  first  \nsecond')), 'first')
        self.assertEqual(exception_summary_line(RuntimeError()), 'RuntimeError')
        line = exception_summary_line(ValueError('x' * 500), max_length=20)
        self.assertEqual(line, 'x' * 17 + '...')
        self.assertEqual(len(line), 20)


class TestSchedulerAdaptiveReactionLevels(unittest.TestCase):
    """
    Contains unit tests for the reaction-wide adaptive levels of theory logic
    (Scheduler._apply_adaptive_reaction_levels and the spawn_job override).
    """
    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.ess_settings = {'gaussian': ['server1'], 'molpro': ['server2', 'server1'], 'qchem': ['server1']}
        # A grain at exactly 1 heavy atom, and another for everything larger, so single-heavy-atom wells
        # (e.g. CH4, OH) land on a different grain than a 2-heavy-atom reaction.
        cls.adaptive_levels = {(1, 1): {('opt', 'freq'): Level(repr='wb97xd/def2tzvp'),
                                        ('sp',): Level(repr='ccsd(t)-f12/cc-pvtz-f12')},
                               (2, 'inf'): {('opt', 'freq'): Level(repr='b3lyp/6-31g(d,p)'),
                                            ('sp',): Level(repr='b3lyp/6-311+g(d,p)')}}

    def build_scheduler(self, rxn, species_list, name):
        """
        Build a testing Scheduler for a single reaction under the class adaptive levels.

        Args:
            rxn (ARCReaction): The reaction.
            species_list (list): The species list.
            name (str): A unique project name.

        Returns:
            Scheduler: The constructed (testing) scheduler.
        """
        project_directory = os.path.join(ARC_PATH, 'Projects', f'{name}_delete')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        return Scheduler(project=name,
                         ess_settings=self.ess_settings,
                         species_list=species_list,
                         rxn_list=[rxn],
                         opt_level=Level(repr='b3lyp/6-31g(d,p)'),
                         sp_level=Level(repr='b3lyp/6-311+g(d,p)'),
                         freq_level=Level(repr='b3lyp/6-31g(d,p)'),
                         adaptive_levels=self.adaptive_levels,
                         project_directory=project_directory,
                         job_types=initialize_job_types(),
                         testing=True)

    def test_bimolecular_creates_copies(self):
        """Test that with thermo_at_own_level=True, wells on a different grain get relabeled copies"""
        r = [ARCSpecies(label='CH4', smiles='C', thermo_at_own_level=True),
             ARCSpecies(label='OH', smiles='[OH]', thermo_at_own_level=True)]
        p = [ARCSpecies(label='CH3', smiles='[CH3]', thermo_at_own_level=True),
             ARCSpecies(label='H2O', smiles='O', thermo_at_own_level=True)]
        rxn = ARCReaction(label='CH4 + OH <=> CH3 + H2O', r_species=r, p_species=p)
        sched = self.build_scheduler(rxn, r + p, 'adaptive_bimol')

        # The reaction is now defined with the copies, and stays self-consistent.
        self.assertEqual(rxn.reactants, ['CH4_TS0', 'OH_TS0'])
        self.assertEqual(rxn.products, ['CH3_TS0', 'H2O_TS0'])
        self.assertEqual([s.label for s in rxn.r_species], ['CH4_TS0', 'OH_TS0'])
        self.assertEqual(rxn.label, 'CH4_TS0 + OH_TS0 <=> CH3_TS0 + H2O_TS0')
        rxn.check_attributes()  # Must not raise.

        # The copies are autonomous, evaluated at the reaction-wide level, and not part of the thermo library.
        for copy_label in ['CH4_TS0', 'OH_TS0', 'CH3_TS0', 'H2O_TS0']:
            self.assertIn(copy_label, sched.species_dict)
            self.assertIn(copy_label, sched.output)
            copy_spc = sched.species_dict[copy_label]
            self.assertEqual(copy_spc.adaptive_lot_n_heavy, 2)
            self.assertFalse(copy_spc.compute_thermo)
            self.assertFalse(copy_spc.include_in_thermo_lib)
            # The copy must not itself opt into copy behavior, else a restart would relabel it again.
            self.assertFalse(copy_spc.thermo_at_own_level)

        # The originals are untouched - they keep their own granular level and their own thermo.
        for original_label in ['CH4', 'OH', 'CH3', 'H2O']:
            original = sched.species_dict[original_label]
            self.assertIsNone(original.adaptive_lot_n_heavy)
            self.assertTrue(original.compute_thermo)

    def test_duplicate_reaction_labels_are_rejected_before_adaptive_relabel(self):
        """Test that duplicate reactions are refused even though the adaptive relabel would make their labels distinct"""
        species = [ARCSpecies(label=label, smiles=smiles, thermo_at_own_level=True)
                   for label, smiles in [('CH4', 'C'), ('OH', '[OH]'), ('CH3', '[CH3]'), ('H2O', 'O')]]
        rxn_1 = ARCReaction(label='CH4 + OH <=> CH3 + H2O', r_species=species[:2], p_species=species[2:])
        rxn_2 = ARCReaction(label='CH4 + OH <=> CH3 + H2O', r_species=species[:2], p_species=species[2:])
        project_directory = os.path.join(ARC_PATH, 'Projects', 'adaptive_duplicate_labels_delete')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        with self.assertRaises(InputError) as cm:
            Scheduler(project='adaptive_duplicate_labels',
                      ess_settings=self.ess_settings,
                      species_list=species,
                      rxn_list=[rxn_1, rxn_2],
                      opt_level=Level(repr='b3lyp/6-31g(d,p)'),
                      sp_level=Level(repr='b3lyp/6-311+g(d,p)'),
                      freq_level=Level(repr='b3lyp/6-31g(d,p)'),
                      adaptive_levels=self.adaptive_levels,
                      project_directory=project_directory,
                      job_types=initialize_job_types(),
                      testing=True)
        self.assertIn("'CH4 + OH <=> CH3 + H2O'", str(cm.exception))

    def test_thermo_at_own_level_default_no_copy(self):
        """Test that by default (thermo_at_own_level=False) the species itself takes the reaction-wide level, no copy"""
        r = [ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='OH', smiles='[OH]')]
        p = [ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='H2O', smiles='O')]
        rxn = ARCReaction(label='CH4 + OH <=> CH3 + H2O', r_species=r, p_species=p)
        sched = self.build_scheduler(rxn, r + p, 'adaptive_default')

        self.assertEqual(rxn.label, 'CH4 + OH <=> CH3 + H2O')
        self.assertFalse(any('_TS' in label for label in sched.species_dict))
        self.assertEqual(sched.species_dict['CH4'].adaptive_lot_n_heavy, 2)

    def test_shared_species_across_grains_gets_copy(self):
        """Test that a no-copy species shared by reactions on different grains gets a copy for the second reaction"""
        oh = ARCSpecies(label='OH', smiles='[OH]')
        h2o = ARCSpecies(label='H2O', smiles='O')
        rxn1 = ARCReaction(label='CH4 + OH <=> CH3 + H2O',
                           r_species=[ARCSpecies(label='CH4', smiles='C'), oh],
                           p_species=[ARCSpecies(label='CH3', smiles='[CH3]'), h2o])
        rxn2 = ARCReaction(label='C3H8 + OH <=> nC3H7 + H2O',
                           r_species=[ARCSpecies(label='C3H8', smiles='CCC'), oh],
                           p_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC'), h2o])
        project_directory = os.path.join(ARC_PATH, 'Projects', 'adaptive_shared_delete')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        species_list = rxn1.r_species + rxn1.p_species + [rxn2.r_species[0], rxn2.p_species[0]]
        sched = Scheduler(project='adaptive_shared',
                          ess_settings=self.ess_settings,
                          species_list=species_list,
                          rxn_list=[rxn1, rxn2],
                          opt_level=Level(repr='b3lyp/6-31g(d,p)'),
                          sp_level=Level(repr='b3lyp/6-311+g(d,p)'),
                          freq_level=Level(repr='b3lyp/6-31g(d,p)'),
                          adaptive_levels={(1, 1): {('sp',): Level(repr='ccsd(t)-f12/cc-pvtz-f12')},
                                           (2, 3): {('sp',): Level(repr='dlpno-ccsd(t)/def2-tzvp')},
                                           (4, 'inf'): {('sp',): Level(repr='b3lyp/6-311+g(d,p)')}},
                          project_directory=project_directory,
                          job_types=initialize_job_types(),
                          testing=True)

        # rxn1 (2 heavy atoms) set the shared wells' overrides; rxn1 itself is unchanged.
        self.assertEqual(rxn1.label, 'CH4 + OH <=> CH3 + H2O')
        self.assertEqual(sched.species_dict['OH'].adaptive_lot_n_heavy, 2)
        # rxn2 (4 heavy atoms) lands on a different grain, so the shared wells got dedicated copies.
        self.assertEqual(set(rxn2.reactants), {'C3H8', 'OH_TS1'})
        self.assertEqual(set(rxn2.products), {'nC3H7', 'H2O_TS1'})
        self.assertEqual(rxn2.label,
                         rxn2.arrow.join([rxn2.plus.join(rxn2.reactants), rxn2.plus.join(rxn2.products)]))
        for copy_label in ['OH_TS1', 'H2O_TS1']:
            self.assertEqual(sched.species_dict[copy_label].adaptive_lot_n_heavy, 4)
            self.assertFalse(sched.species_dict[copy_label].compute_thermo)
        # Unshared rxn2 wells just took the rxn2 override, no copies.
        self.assertEqual(sched.species_dict['C3H8'].adaptive_lot_n_heavy, 4)
        self.assertEqual(sched.species_dict['nC3H7'].adaptive_lot_n_heavy, 4)

    def test_unimolecular_no_copy(self):
        """Test that a reaction whose well shares the reaction's grain gets no copies"""
        r = [ARCSpecies(label='nC3H7', smiles='[CH2]CC')]
        p = [ARCSpecies(label='iC3H7', smiles='C[CH]C')]
        rxn = ARCReaction(label='nC3H7 <=> iC3H7', r_species=r, p_species=p)
        sched = self.build_scheduler(rxn, r + p, 'adaptive_unimol')

        self.assertEqual(rxn.label, 'nC3H7 <=> iC3H7')
        self.assertFalse(any('_TS' in label for label in sched.species_dict))
        self.assertIsNone(sched.species_dict['nC3H7'].adaptive_lot_n_heavy)

    def test_spawn_job_uses_override(self):
        """Test that determine_adaptive_level is driven by adaptive_lot_n_heavy when set"""
        spc = ARCSpecies(label='CH4', smiles='C')
        sched = Scheduler(project='adaptive_override',
                          ess_settings=self.ess_settings,
                          species_list=[spc],
                          opt_level=Level(repr='b3lyp/6-31g(d,p)'),
                          sp_level=Level(repr='b3lyp/6-311+g(d,p)'),
                          freq_level=Level(repr='b3lyp/6-31g(d,p)'),
                          adaptive_levels=self.adaptive_levels,
                          project_directory=os.path.join(ARC_PATH, 'Projects', 'adaptive_override_delete'),
                          job_types=initialize_job_types(),
                          testing=True)
        self.addCleanup(shutil.rmtree, os.path.join(ARC_PATH, 'Projects', 'adaptive_override_delete'),
                        ignore_errors=True)
        original = Level(method='CBS-QB3')
        # 1 heavy atom (own count) -> the (1, 1) grain.
        self.assertEqual(sched.determine_adaptive_level(original, 'sp', spc.number_of_heavy_atoms).simple(),
                         'ccsd(t)-f12/cc-pvtz-f12')
        # With the override set to a 2-heavy-atom reaction, the (2, 'inf') grain is used instead.
        spc.adaptive_lot_n_heavy = 2
        heavy = spc.adaptive_lot_n_heavy if spc.adaptive_lot_n_heavy is not None else spc.number_of_heavy_atoms
        self.assertEqual(sched.determine_adaptive_level(original, 'sp', heavy).simple(), 'b3lyp/6-311+g(d,p)')

    def test_apply_adaptive_reaction_levels_idempotent(self):
        """Test that re-applying the reaction-wide logic (as on restart) does not duplicate the copies"""
        r = [ARCSpecies(label='CH4', smiles='C', thermo_at_own_level=True),
             ARCSpecies(label='OH', smiles='[OH]', thermo_at_own_level=True)]
        p = [ARCSpecies(label='CH3', smiles='[CH3]', thermo_at_own_level=True),
             ARCSpecies(label='H2O', smiles='O', thermo_at_own_level=True)]
        rxn = ARCReaction(label='CH4 + OH <=> CH3 + H2O', r_species=r, p_species=p)
        sched = self.build_scheduler(rxn, r + p, 'adaptive_idempotent')
        labels_after_first = set(sched.species_dict.keys())

        # Re-applying (mimicking a restart that already carries the copies) must be a no-op, not re-copy.
        sched._apply_adaptive_reaction_levels()
        self.assertEqual(set(sched.species_dict.keys()), labels_after_first)
        self.assertEqual(rxn.reactants, ['CH4_TS0', 'OH_TS0'])
        self.assertEqual(rxn.products, ['CH3_TS0', 'H2O_TS0'])
        self.assertFalse(any(label.endswith('_TS0_TS0') for label in sched.species_dict))

    def test_apply_adaptive_reaction_levels_label_collision(self):
        """Test that an unrelated species already occupying the copy label raises rather than being silently used"""
        r = [ARCSpecies(label='CH4', smiles='C', thermo_at_own_level=True),
             ARCSpecies(label='OH', smiles='[OH]', thermo_at_own_level=True)]
        p = [ARCSpecies(label='CH3', smiles='[CH3]', thermo_at_own_level=True),
             ARCSpecies(label='H2O', smiles='O', thermo_at_own_level=True)]
        rxn = ARCReaction(label='CH4 + OH <=> CH3 + H2O', r_species=r, p_species=p)
        # An unrelated user species that happens to occupy the copy label ARC would generate for CH4.
        collider = ARCSpecies(label='CH4_TS0', smiles='O')
        with self.assertRaises(SchedulerError):
            self.build_scheduler(rxn, r + p + [collider], 'adaptive_collision')


class TestGetServerJobIds(unittest.TestCase):
    """The status poll runs every cycle for every job, so it is the hottest SSH caller there is."""

    @staticmethod
    def _sched(servers):
        """A stand-in carrying only what get_server_job_ids() reads, with one tracked queue job on each server."""
        jobs = {f'opt_{server}': SimpleNamespace(job_name=f'opt_{server}', job_id=900 + i, server=server,
                                                 execution_type='queue') for i, server in enumerate(servers)}
        return SimpleNamespace(servers=servers, server_job_ids=None, job_dict={'spc': {'opt': jobs}},
                               running_jobs={'spc': list(jobs)},
                               queue_query_failures=dict(), unreadable_queue_servers=dict(),
                               answered_queue_servers=set(),
                               clear_job_query_failures_seen_running=lambda server, server_job_ids: None,
                               job_ids_in_query_cooldown=lambda: list(),
                               queue_clock=time.monotonic, queue_outage_limit=4 * 3600)

    def test_a_remote_server_is_polled_through_a_pooled_client(self):
        """Opening a connection per poll is what the pool exists to stop."""
        sched = self._sched(['zeus'])
        client = MagicMock()
        client.check_running_jobs_ids.return_value = ['101', '102']
        with patch('arc.scheduler.borrow_ssh_client') as borrow:
            borrow.return_value.__enter__.return_value = client
            Scheduler.get_server_job_ids(sched)
        borrow.assert_called_once_with('zeus')
        self.assertEqual(sched.server_job_ids, ['101', '102'])

    def test_the_borrowed_client_is_released(self):
        """A borrow that is not exited would hold the pool's client for the rest of the run."""
        sched = self._sched(['zeus'])
        with patch('arc.scheduler.borrow_ssh_client') as borrow:
            borrow.return_value.__enter__.return_value = MagicMock()
            Scheduler.get_server_job_ids(sched)
        borrow.return_value.__exit__.assert_called_once()

    def test_every_poll_of_one_server_goes_through_one_borrow(self):
        """Each cycle borrows once per server, which is one pooled client for the whole run."""
        sched = self._sched(['zeus'])
        with patch('arc.scheduler.borrow_ssh_client') as borrow:
            borrow.return_value.__enter__.return_value = MagicMock()
            for _ in range(50):
                Scheduler.get_server_job_ids(sched)
        self.assertEqual(borrow.call_count, 50)

    def test_a_local_server_is_not_polled_over_ssh(self):
        """The local queue is read with a local command, and must not touch the pool."""
        sched = self._sched(['local'])
        with patch('arc.scheduler.borrow_ssh_client') as borrow, \
                patch('arc.scheduler.check_running_jobs_ids', return_value=['7']):
            Scheduler.get_server_job_ids(sched)
        borrow.assert_not_called()
        self.assertEqual(sched.server_job_ids, ['7'])

    def test_a_specific_server_limits_the_poll_to_it(self):
        sched = self._sched(['zeus', 'atlas'])
        with patch('arc.scheduler.borrow_ssh_client') as borrow:
            borrow.return_value.__enter__.return_value = MagicMock()
            Scheduler.get_server_job_ids(sched, specific_server='atlas')
        borrow.assert_called_once_with('atlas')


class TestGetServerJobIdsWhenTheQueueCannotBeRead(unittest.TestCase):
    """A queue that does not answer keeps every tracked job running instead of ending them all (#394)."""

    SOCKET_TIMEOUT = ['squeue: error: slurm_receive_msg: Socket timed out on send/recv operation\n',
                      'slurm_load_jobs error: Socket timed out on send/recv operation\n']
    SLURM_LISTING = ['             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)\n',
                     '                21     short      a21     user  R       0:01      1 node01\n']
    SERVERS = {'zeus': {'address': 'zeus.example.edu', 'un': 'user', 'cluster_soft': 'Slurm'},
               'atlas': {'address': 'atlas.example.edu', 'un': 'user', 'cluster_soft': 'Slurm'}}

    def setUp(self):
        servers = patch('arc.job.ssh.servers', self.SERVERS)
        servers.start()
        self.addCleanup(servers.stop)
        sleep = patch('time.sleep')
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        self.client = SSHClient('zeus')

    @staticmethod
    def _job(job_name, job_id, server='zeus', execution_type='queue'):
        return SimpleNamespace(job_name=job_name, job_id=job_id, server=server, execution_type=execution_type)

    def _sched(self):
        """A stand-in tracking queue jobs on two servers, an incore job, and a job that already ended."""
        job_dict = {'spc': {'opt': {'opt_a11': self._job('opt_a11', 11),
                                    'opt_a14': self._job('opt_a14', 14)},
                            'sp': {'sp_a12': self._job('sp_a12', 12, execution_type='incore')},
                            'freq': {'freq_a13': self._job('freq_a13', 13, server='atlas')},
                            'conf_opt': {0: self._job('conf_opt_0', 15)}}}
        running_jobs = {'spc': ['opt_a11', 'sp_a12', 'freq_a13', 'conf_opt_0']}
        return SimpleNamespace(servers=['zeus'], server_job_ids=None, job_dict=job_dict, running_jobs=running_jobs,
                               queue_query_failures=dict(), unreadable_queue_servers=dict(),
                               answered_queue_servers=set(),
                               clear_job_query_failures_seen_running=lambda server, server_job_ids: None,
                               job_ids_in_query_cooldown=lambda: list(),
                               queue_clock=time.monotonic, queue_outage_limit=4 * 3600)

    def _poll(self, sched, *replies):
        """Run one get_server_job_ids() with the zeus server boundary answering ``replies``."""
        with patch('arc.scheduler.borrow_ssh_client') as borrow, \
                patch.object(self.client, '_send_command_to_server', side_effect=list(replies)) as send:
            borrow.return_value.__enter__.return_value = self.client
            Scheduler.get_server_job_ids(sched)
        return send

    def test_a_failed_poll_keeps_the_tracked_jobs_running(self):
        sched = self._sched()
        with self.assertLogs('arc', level='WARNING') as logged:
            self._poll(sched, *[([], self.SOCKET_TIMEOUT, 1)] * 3)
        self.assertEqual(sorted(sched.server_job_ids), [11, 15])
        self.assertEqual(len(logged.output), 1)
        self.assertIn('zeus', logged.output[0])

    def test_a_missing_local_queue_command_is_an_outage_while_local_queue_jobs_are_tracked(self):
        """A missing local command reads as no jobs while no local queue job is tracked, and is an outage after."""
        sched = self._sched()
        sched.servers = ['local']
        missing = ([], ['/bin/sh: 1: /usr/bin/squeue: not found'], 127)
        with patch.dict('arc.job.local.servers', {'local': {'cluster_soft': 'Slurm'}}), \
                patch('arc.job.local.execute_command', return_value=missing):
            Scheduler.get_server_job_ids(sched, specific_server='local')
            self.assertNotIn('local', sched.queue_query_failures)
            self.assertNotIn('local', sched.answered_queue_servers)
            sched.answered_queue_servers.add('local')
            sched.job_dict['spc']['opt']['opt_a11'].server = 'local'
            with self.assertLogs('arc', level='WARNING'):
                Scheduler.get_server_job_ids(sched)
        self.assertEqual(sched.queue_query_failures['local']['exit_status'], 127)
        self.assertEqual(sched.server_job_ids, [11])

    def test_a_real_local_answer_counts_while_a_missing_command_would_read_as_no_jobs(self):
        """A real answer from an idle local queue counts as an answer, so a later missing command is an outage."""
        sched = self._sched()
        sched.servers = ['local']
        with patch.dict('arc.job.local.servers', {'local': {'cluster_soft': 'Slurm'}}), \
                patch('arc.job.local.execute_command', return_value=([], [], 0)):
            Scheduler.get_server_job_ids(sched, specific_server='local')
        self.assertIn('local', sched.answered_queue_servers)
        sched.job_dict['spc']['opt']['opt_a11'].server = 'local'
        missing = ([], ['/bin/sh: 1: /usr/bin/squeue: not found'], 127)
        with patch.dict('arc.job.local.servers', {'local': {'cluster_soft': 'Slurm'}}), \
                patch('arc.job.local.execute_command', return_value=missing), \
                self.assertLogs('arc', level='WARNING'):
            Scheduler.get_server_job_ids(sched)
        self.assertEqual(sched.queue_query_failures['local']['exit_status'], 127)

    def test_a_missing_queue_command_on_a_server_that_never_answered_stops_arc(self):
        sched = self._sched()
        missing = ([], ['bash: /usr/bin/squeue: No such file or directory'], 127)
        with patch.dict('arc.scheduler.servers_dict', {'zeus': {'cluster_soft': 'Slurm'}}), \
                self.assertRaises(SettingsError) as cm:
            self._poll(sched, *[missing] * 3)
        self.assertIn('zeus', str(cm.exception))
        self.assertIn('check_status_command', str(cm.exception))
        self.assertNotIn('zeus', sched.queue_query_failures)

    def test_a_missing_queue_command_after_an_answer_is_an_outage(self):
        sched = self._sched()
        self._poll(sched, (self.SLURM_LISTING, [], 0))
        self.assertIn('zeus', sched.answered_queue_servers)
        with self.assertLogs('arc', level='WARNING'):
            self._poll(sched, *[([], ['bash: squeue: command not found'], 127)] * 3)
        self.assertEqual(sched.queue_query_failures['zeus']['exit_status'], 127)

    def test_a_missing_local_queue_command_stops_arc_when_the_local_server_caps_its_jobs(self):
        """The job limit check queries an idle local server that caps its jobs, and a missing command stops ARC."""
        sched = self._sched()
        sched.servers = ['local']
        missing = ([], ['/bin/sh: 1: /usr/bin/squeue: not found'], 127)
        with patch.dict('arc.job.local.servers', {'local': {'cluster_soft': 'Slurm'}}), \
                patch.dict('arc.scheduler.servers_dict', {'local': {'cluster_soft': 'Slurm',
                                                                    'max_simultaneous_jobs': 10}}), \
                patch('arc.job.local.execute_command', return_value=missing), \
                self.assertRaises(SettingsError):
            Scheduler.get_server_job_ids(sched, specific_server='local')

    def test_a_good_poll_reads_the_queue(self):
        sched = self._sched()
        self._poll(sched, (self.SLURM_LISTING, [], 0))
        self.assertEqual(sched.server_job_ids, ['21'])

    def test_polls_inside_the_cooldown_do_not_query_the_server_again(self):
        sched = self._sched()
        with self.assertLogs('arc', level='WARNING'):
            first = self._poll(sched, *[([], self.SOCKET_TIMEOUT, 1)] * 3)
        with self.assertNoLogs('arc', level='WARNING'):
            second = self._poll(sched)
        self.assertEqual(first.call_count, 3)
        self.assertEqual(second.call_count, 0)
        self.assertEqual(self.sleep.call_count, 2)
        self.assertEqual(sorted(sched.server_job_ids), [11, 15])

    def test_the_first_good_poll_after_the_cooldown_clears_it(self):
        sched = self._sched()
        with self.assertLogs('arc', level='WARNING'):
            self._poll(sched, *[([], self.SOCKET_TIMEOUT, 1)] * 3)
        sched.queue_query_failures['zeus']['last_attempt'] -= 3600
        self._poll(sched, (self.SLURM_LISTING, [], 0))
        self.assertEqual(sched.server_job_ids, ['21'])
        self.assertNotIn('zeus', sched.queue_query_failures)

    def test_the_cooldown_of_one_server_does_not_hold_another(self):
        sched = self._sched()
        sched.queue_query_failures['atlas'] = outage_record(time.monotonic())
        self._poll(sched, (self.SLURM_LISTING, [], 0))
        self.assertEqual(sched.server_job_ids, ['21'])


class TestJobLimitWhenTheQueueCannotBeRead(unittest.TestCase):
    """The tracked-jobs fallback undercounts the user's queue, so it must not open the job limit."""

    def test_submission_is_put_off_without_waiting_while_the_queue_cannot_be_read(self):
        sched = SimpleNamespace(queue_query_failures=dict(), server_job_ids=list(), unreadable_queue_servers=dict(),
                                queue_clock=time.monotonic, queue_outage_limit=4 * 3600)
        polls = list()

        def get_server_job_ids(specific_server=None):
            polls.append(specific_server)
            sched.queue_query_failures['zeus'] = outage_record(time.monotonic())
            sched.server_job_ids = ['11']
        sched.get_server_job_ids = get_server_job_ids
        with patch.dict('arc.scheduler.servers_dict', {'zeus': {'max_simultaneous_jobs': 10}}), \
                patch('time.sleep', side_effect=AssertionError('waited for an unreadable queue')):
            self.assertEqual(Scheduler.check_max_simultaneous_jobs_limit(sched, 'zeus'), DEFER)
        self.assertEqual(polls, ['zeus', None])

    def test_a_full_queue_that_answers_is_waited_for(self):
        sched = SimpleNamespace(queue_query_failures=dict(), server_job_ids=list(), unreadable_queue_servers=dict(),
                                queue_clock=time.monotonic, queue_outage_limit=4 * 3600)
        answers = iter([['11', '12'], ['11'], ['11']])

        def get_server_job_ids(specific_server=None):
            sched.server_job_ids = next(answers)
        sched.get_server_job_ids = get_server_job_ids
        with patch.dict('arc.scheduler.servers_dict', {'zeus': {'max_simultaneous_jobs': 2}}), \
                patch('time.sleep') as sleep:
            self.assertEqual(Scheduler.check_max_simultaneous_jobs_limit(sched, 'zeus'), SUBMIT)
        self.assertEqual(sleep.call_count, 1)

    def test_the_wait_ends_at_the_outage_limit(self):
        sched = SimpleNamespace(queue_query_failures=dict(), server_job_ids=list(), unreadable_queue_servers=dict(),
                                queue_clock=time.monotonic, queue_outage_limit=4 * 3600)

        def get_server_job_ids(specific_server=None):
            sched.queue_query_failures['zeus'] = outage_record(time.monotonic() - 5 * 3600)
            sched.server_job_ids = ['11']
        sched.get_server_job_ids = get_server_job_ids
        with patch.dict('arc.scheduler.servers_dict', {'zeus': {'max_simultaneous_jobs': 10}}), \
                patch('time.sleep', side_effect=AssertionError('waited past the outage limit')):
            self.assertEqual(Scheduler.check_max_simultaneous_jobs_limit(sched, 'zeus'), REFUSE)

    def test_an_answered_queue_under_the_limit_does_not_wait(self):
        sched = SimpleNamespace(queue_query_failures=dict(), server_job_ids=list(), unreadable_queue_servers=dict(),
                                queue_clock=time.monotonic, queue_outage_limit=4 * 3600)
        sched.get_server_job_ids = lambda specific_server=None: None
        with patch.dict('arc.scheduler.servers_dict', {'zeus': {'max_simultaneous_jobs': 10}}), \
                patch('time.sleep') as sleep:
            Scheduler.check_max_simultaneous_jobs_limit(sched, 'zeus')
        sleep.assert_not_called()


class TestEndJobWhenTheQueueCannotBeRead(unittest.TestCase):

    def test_the_job_stays_running_and_is_not_rerun(self):
        sched = MagicMock()
        sched.running_jobs = {'spc': ['opt_a11']}
        job = MagicMock()
        job.job_status = ['running', {'status': 'running'}]
        job.job_type = 'opt'
        job.execution_type = 'queue'
        job.local_path_to_output_file = '/no/such/output.out'
        job.determine_job_status.return_value = False
        with patch('time.sleep'):
            result = Scheduler.end_job(sched, job=job, label='spc', job_name='opt_a11')
        self.assertFalse(result)
        sched._run_a_job.assert_not_called()
        self.assertEqual(sched.running_jobs, {'spc': ['opt_a11']})
        self.assertEqual(job.job_status, ['running', {'status': 'running'}])


class _QueueOutageServer(object):
    """A stand-in for a pooled SSH client whose queue query fails while ``answer`` is ``None``."""

    def __init__(self):
        self.answer, self.queries = None, 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def check_running_jobs_ids(self) -> list:
        self.queries += 1
        if self.answer is None:
            raise QueueQueryError('The queue status command on server1 failed 3 times',
                                  server='server1', output='slurm_load_jobs error: Socket timed out', exit_status=1)
        return list(self.answer)


class TestQueueOutageBound(unittest.TestCase):
    """An unreadable queue is waited for up to a time bound, then its jobs are left for a restart (#394)."""

    HOUR = 3600.0
    JOB_TYPES = {'conf_opt': False, 'conf_sp': False, 'opt': True, 'fine': False, 'freq': True, 'sp': True,
                 'rotors': False, 'orbitals': False, 'lennard_jones': False, 'irc': False, 'bde': False}
    XYZ = """C      -0.57422867   -0.01669771    0.01229213
N       0.82084044    0.08279104   -0.37769346
H      -1.05737005   -0.84067772   -0.52007494
H      -1.10211468    0.90879867   -0.23383011
H      -0.66133128   -0.19490562    1.08785111
H       0.88047852    0.26966160   -1.37780789
H       1.27889520   -0.81548721   -0.22940984"""

    def setUp(self):
        self.project_directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.project_directory, ignore_errors=True)
        self.ess_settings = {'gaussian': ['server1']}
        self.now = 0.0
        self.server = _QueueOutageServer()
        borrow = patch('arc.scheduler.borrow_ssh_client', side_effect=lambda server: self.server)
        borrow.start()
        self.addCleanup(borrow.stop)
        ledger_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, ledger_dir, ignore_errors=True)
        ledger = patch('arc.job.adapter.local_arc_path', ledger_dir)
        ledger.start()
        self.addCleanup(ledger.stop)

    def _scheduler(self, restart_dict: dict | None = None, testing: bool = True) -> Scheduler:
        sched = Scheduler(project='queue_outage', ess_settings=self.ess_settings,
                          species_list=[ARCSpecies(label='methylamine', smiles='CN', xyz=self.XYZ)],
                          composite_method=None,
                          conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          scan_level=Level(repr=default_levels_of_theory['scan']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=self.project_directory, testing=testing,
                          job_types=dict(self.JOB_TYPES), restart_dict=restart_dict,
                          orbitals_level=default_levels_of_theory['orbitals'], adaptive_levels=None)
        sched.queue_clock = lambda: self.now
        if sched.restart_dict is None:
            sched.restart_dict = dict()
        return sched

    def _running(self, sched: Scheduler, job_id: int = 4242):
        """Track one submitted opt job on server1, as run_job() leaves it."""
        job = job_factory(job_adapter='gaussian', project='queue_outage', ess_settings=self.ess_settings,
                          species=[sched.species_dict['methylamine']], xyz=self.XYZ, job_type='opt',
                          level=Level(repr={'method': 'wb97xd', 'basis': 'def2tzvp'}),
                          project_directory=self.project_directory, job_num=7, server='server1', testing=True)
        job.job_id, job.execution_type = job_id, 'queue'
        sched.job_dict['methylamine']['opt'] = {job.job_name: job}
        sched.running_jobs['methylamine'] = [job.job_name]
        sched.servers = ['server1']
        return job

    def _second_job(self, sched: Scheduler, server: str, job_id: int):
        """
        Track a second submitted queue job (a freq job) for methylamine.

        Args:
            sched (Scheduler): The scheduler.
            server (str): The server the job runs on.
            job_id (int): The job's queue ID.

        Returns:
            JobAdapter: The job.
        """
        job = job_factory(job_adapter='gaussian', project='queue_outage', ess_settings=self.ess_settings,
                          species=[sched.species_dict['methylamine']], xyz=self.XYZ, job_type='freq',
                          level=Level(repr={'method': 'wb97xd', 'basis': 'def2tzvp'}),
                          project_directory=self.project_directory, job_num=8, server=server, testing=True)
        job.job_id, job.execution_type = job_id, 'queue'
        sched.job_dict['methylamine']['freq'] = {job.job_name: job}
        sched.running_jobs['methylamine'].append(job.job_name)
        return job

    def _pass(self, sched: Scheduler, hours: float):
        """One main-loop pass at ``hours`` into the outage."""
        self.now = hours * self.HOUR
        sched.strand_unreadable_queue_jobs()
        sched.get_server_job_ids()

    def _outage(self, sched: Scheduler, until_hours: float, step_hours: float = 0.25, from_hours: float = 0.0):
        hours = from_hours
        while hours < until_hours:
            self._pass(sched, hours)
            hours += step_hours
        self._pass(sched, until_hours)

    def test_a_command_that_can_hide_a_failure_is_warned_about_once_per_server_at_start(self):
        """
        A pipe, an OR-list (``||``) or any ``;`` in ``check_status_command`` gets one warning per server at start.

        Each of these can make the command exit with another command's status, so a failed queue query can read as
        an empty queue. A bare command and an ``&&``-list get no warning.
        """
        self.ess_settings = {'gaussian': ['server1'], 'molpro': ['server1']}
        for command, form in (('qstat -u $USER | grep queue_outage', 'a pipe'),
                              ('qstat -u $USER | tail -n +3', 'a pipe'),
                              ('qstat -u $USER || true', 'an OR-list (||)'),
                              ('squeue -u $USER || echo', 'an OR-list (||)'),
                              ('qstat -u $USER; true', 'a ";"'),
                              ('cd /tmp; squeue -u $USER', 'a ";"'),
                              ('qstat -u $USER | grep queue_outage || true', 'a pipe and an OR-list (||)'),
                              ('qstat -u $USER', None),
                              ('cd /tmp && squeue -u $USER', None)):
            with self.subTest(command=command), \
                    patch.dict(settings['check_status_command'], {'OGE': command}), \
                    self.assertLogs('arc', level='WARNING') as logs:
                self._scheduler()
                logging.getLogger('arc').warning('sentinel')
            warned = [line for line in logs.output if 'check_status_command' in line]
            self.assertEqual(len(warned), 0 if form is None else 1, warned)
            if form is not None:
                self.assertIn(f'server server1 contains {form}: {command}', warned[0])
                self.assertIn('bare queue command', warned[0])
                self.assertIn('exits non-zero on an empty queue', warned[0])
                self.assertIn('exits 0 when the queue command fails', warned[0])

    def test_a_semicolon_after_variable_assignments_only_is_not_warned_about(self):
        """
        A ``;`` that only ends variable assignments leaves the queue command last, as in ARC's default OGE command.

        A ``;`` followed by another command after the queue command, or preceded by a command that is not an
        assignment, is still warned about.
        """
        for command, warns in ((settings['check_status_command']['OGE'], False),
                               ('export SGE_ROOT=/opt/sge; /opt/sge/bin/lx24-amd64/qstat -u $USER', False),
                               ('SGE_ROOT=/opt/sge; export SGE_CELL="default"; qstat -u $USER', False),
                               ('export SGE_ROOT=/opt/sge; qstat -u $USER; true', True),
                               ('qstat -u $USER; true', True),
                               ('cd dir; squeue', True)):
            with self.subTest(command=command), \
                    patch.dict(settings['check_status_command'], {'OGE': command}), \
                    self.assertLogs('arc', level='WARNING') as logs:
                self._scheduler()
                logging.getLogger('arc').warning('sentinel')
            warned = [line for line in logs.output if 'check_status_command' in line]
            self.assertEqual(len(warned), int(warns), warned)

    def test_before_the_bound_the_jobs_are_held(self):
        sched = self._scheduler()
        job = self._running(sched)
        with self.assertLogs('arc', level='WARNING') as logged:
            self._outage(sched, 3.9)
        self.assertFalse(any('ERROR' in line for line in logged.output))
        self.assertEqual(sched.running_jobs['methylamine'], [job.job_name])
        self.assertIn(4242, sched.server_job_ids)
        self.assertEqual(sched.stranded_jobs, dict())

    def test_the_outage_warning_gives_elapsed_and_remaining_time_at_most_every_ten_minutes(self):
        """Over 21 minutes of failed polls, the warning is logged at 0, 10.5 and 21 minutes only."""
        sched = self._scheduler()
        self._running(sched)
        with self.assertLogs('arc', level='WARNING') as logged:
            for minutes in (0, 2, 5, 9, 10.5, 12, 19, 21):
                self.now = minutes * 60.0
                sched.get_server_job_ids()
        warnings = [record.getMessage() for record in logged.records if record.levelno == logging.WARNING]
        self.assertEqual(len(warnings), 3, warnings)
        self.assertIn('0 min', warnings[0])
        self.assertIn('4 h 0 min', warnings[0])
        self.assertIn('10 min', warnings[1])
        self.assertIn('3 h 49 min', warnings[1])
        self.assertIn('21 min', warnings[2])
        self.assertIn('3 h 39 min', warnings[2])
        self.assertTrue(all('server1' in line for line in warnings))

    def test_at_the_bound_one_error_names_server_jobs_stderr_and_the_restart_command(self):
        """At the outage limit one ERROR is logged, and the server is not polled again afterwards."""
        sched = self._scheduler()
        job = self._running(sched)
        status = [job.job_status[0], dict(job.job_status[1])]
        with patch.object(job, 'delete', side_effect=AssertionError('deleted')), \
                patch.object(sched, 'end_job', side_effect=AssertionError('ended')), \
                self.assertLogs('arc', level='WARNING') as logged:
            self._outage(sched, 4.0)
            queries = self.server.queries
            self._outage(sched, 6.0)
        errors = [record.getMessage() for record in logged.records if record.levelno == logging.ERROR]
        self.assertEqual(len(errors), 1, errors)
        self.assertIn('server1', errors[0])
        self.assertIn('4242', errors[0])
        self.assertIn('slurm_load_jobs error: Socket timed out', errors[0])
        restart_command = shlex.join([sys.executable, os.path.join(ARC_PATH, 'ARC.py'),
                                      os.path.join(self.project_directory, 'restart.yml')])
        self.assertIn(restart_command, errors[0])
        self.assertIn('No job was ended or deleted', errors[0])
        self.assertEqual(self.server.queries, queries)
        self.assertNotIn('methylamine', {k: v for k, v in sched.running_jobs.items() if v})
        self.assertEqual(sched.stranded_jobs, {'methylamine': [job.job_name]})
        self.assertEqual(job.job_status, status)
        record = sched.output['methylamine']['unreadable_queue']['server1']
        self.assertEqual(record['server'], 'server1')
        self.assertEqual(record['job_ids'], [4242])
        self.assertIn('Socket timed out', record['last_output'])
        self.assertEqual(record['exit_status'], 1)

    def test_jobs_stranded_on_two_servers_keep_both_records(self):
        sched = self._scheduler()
        job = self._running(sched)
        other = job_factory(job_adapter='gaussian', project='queue_outage', ess_settings=self.ess_settings,
                            species=[sched.species_dict['methylamine']], xyz=self.XYZ, job_type='freq',
                            level=Level(repr={'method': 'wb97xd', 'basis': 'def2tzvp'}),
                            project_directory=self.project_directory, job_num=8, server='server2', testing=True)
        other.job_id, other.execution_type = 5151, 'queue'
        sched.job_dict['methylamine']['freq'] = {other.job_name: other}
        sched.running_jobs['methylamine'].append(other.job_name)
        sched.servers = ['server1', 'server2']
        with self.assertLogs('arc', level='WARNING'):
            self._outage(sched, 4.0)
        records = sched.output['methylamine']['unreadable_queue']
        self.assertEqual({server: record['job_ids'] for server, record in records.items()},
                         {'server1': [4242], 'server2': [5151]})
        self.assertEqual(sorted(sched.stranded_jobs['methylamine']), sorted([job.job_name, other.job_name]))

    def test_the_restart_command_survives_a_path_with_spaces(self):
        sched = self._scheduler()
        sched.restart_path = '/home/user/ARC projects/run 1/restart.yml'
        self.assertEqual(shlex.split(sched.restart_command()),
                         [sys.executable, os.path.join(ARC_PATH, 'ARC.py'), '/home/user/ARC projects/run 1/restart.yml'])

    def test_the_check_files_of_stranded_jobs_are_kept(self):
        sched = self._scheduler()
        self._running(sched)
        sched.remote_project_paths = {'server1': '/home/user/runs/ARC_Projects/queue_outage',
                                      'server2': '/home/user/runs/ARC_Projects/queue_outage'}
        self.assertEqual(sched.cleanable_remote_project_paths(), sched.remote_project_paths)
        with self.assertLogs('arc', level='WARNING'):
            self._outage(sched, 4.0)
        self.assertEqual(sched.cleanable_remote_project_paths(),
                         {'server2': '/home/user/runs/ARC_Projects/queue_outage'})

    def test_the_bound_is_recorded_in_status_and_restart_files(self):
        sched = self._scheduler()
        job = self._running(sched)
        with self.assertLogs('arc', level='WARNING'):
            self._outage(sched, 4.0)
        restart = read_yaml_file(os.path.join(self.project_directory, 'restart.yml'))
        self.assertEqual([j['job_id'] for j in restart['running_jobs']['methylamine']], [4242])
        self.assertEqual(restart['output']['methylamine']['unreadable_queue']['server1']['job_ids'], [4242])
        self.assertIn(job.job_name, restart['output']['methylamine']['errors'])

    def test_a_restart_after_the_bound_picks_the_jobs_up_again(self):
        sched = self._scheduler()
        job = self._running(sched)
        with self.assertLogs('arc', level='WARNING'):
            self._outage(sched, 4.0)
        restart = read_yaml_file(os.path.join(self.project_directory, 'restart.yml'))
        restarted = self._scheduler(restart_dict={'output': restart['output'],
                                                  'running_jobs': restart['running_jobs']})
        self.assertEqual(restarted.running_jobs['methylamine'], [job.job_name])
        self.assertNotIn('unreadable_queue', restarted.output['methylamine'])
        self.assertEqual(restarted.stranded_jobs, dict())
        self.assertEqual(restarted.servers, ['server1'])
        self.server.answer = ['4242']
        queries = self.server.queries
        restarted.get_server_job_ids()
        self.assertEqual(self.server.queries, queries + 1)
        self.assertEqual(restarted.server_job_ids, ['4242'])

    def test_a_good_answer_ends_the_outage_and_resets_the_clock(self):
        """A good answer at 3.5 h resets the clock, so a new outage from 3.75 h is not stranded before 7.75 h."""
        sched = self._scheduler()
        self._running(sched)
        with self.assertLogs('arc', level='WARNING'):
            self._outage(sched, 3.0)
        self.server.answer = ['4242']
        self._pass(sched, 3.5)
        self.assertNotIn('server1', sched.queue_query_failures)
        self.server.answer = None
        with self.assertLogs('arc', level='WARNING') as logged:
            self._outage(sched, 7.25, from_hours=3.75)
        self.assertFalse(any(record.levelno == logging.ERROR for record in logged.records))
        self.assertEqual(sched.stranded_jobs, dict())

    def test_submission_past_the_bound_is_refused_and_named(self):
        sched = self._scheduler()
        sched.servers = ['server1']
        sched.unreadable_queue_servers.setdefault('server1', dict())
        server1 = dict(settings['servers']['server1'], max_simultaneous_jobs=10)
        with patch.dict('arc.scheduler.servers_dict', {'server1': server1}), \
                patch('time.sleep', side_effect=[None] * 3 + [AssertionError('held for ever')]), \
                patch('arc.job.adapter.JobAdapter.execute', side_effect=AssertionError('submitted')), \
                self.assertLogs('arc', level='ERROR') as logged:
            sched.run_opt_job('methylamine')
        self.assertEqual(sched.running_jobs.get('methylamine', list()), list())
        self.assertEqual(len(logged.output), 1)
        self.assertIn('opt_', logged.output[0])
        self.assertIn('server1', logged.output[0])
        self.assertIn('restart submits it', logged.output[0])

    def test_an_incore_job_does_not_wait_for_an_unreadable_queue(self):
        sched = self._scheduler()
        job = MagicMock()
        job.job_name, job.server, job.execution_type, job.remote_project_path = 'opt_a9', 'server1', 'incore', None
        with patch('arc.scheduler.job_factory', return_value=job), \
                patch.object(Scheduler, 'check_max_simultaneous_jobs_limit',
                             side_effect=AssertionError('an incore job waited for the queue')), \
                patch.object(Scheduler, 'warn_on_collapsible_unrestricted_reference'), \
                patch.object(Scheduler, 'save_restart_dict'):
            sched.run_job(label='methylamine', job_type='opt',
                          level_of_theory=Level(repr=default_levels_of_theory['opt']), job_adapter='gaussian')
        job.execute.assert_called_once()

    def _refuse_an_opt_job(self, sched: Scheduler):
        """Run an opt job on server1 after its queue stayed unreadable past the bound, which left it unpolled."""
        sched.servers = ['server1']
        sched.unreadable_queue_servers.setdefault('server1', dict())
        server1 = dict(settings['servers']['server1'], max_simultaneous_jobs=10)
        with patch.dict('arc.scheduler.servers_dict', {'server1': server1}), \
                patch('time.sleep', side_effect=[None] * 3 + [AssertionError('held for ever')]), \
                patch('arc.job.adapter.JobAdapter.execute', side_effect=AssertionError('submitted')), \
                self.assertLogs('arc', level='ERROR') as logged:
            sched.run_opt_job('methylamine')
        return logged

    def test_a_refused_job_stays_in_the_restart_file_unsubmitted(self):
        sched = self._scheduler()
        self._refuse_an_opt_job(sched)
        job_name = list(sched.job_dict['methylamine']['opt'])[0]
        self.assertEqual(sched.stranded_jobs['methylamine'], [job_name])
        restart = read_yaml_file(os.path.join(self.project_directory, 'restart.yml'))
        refused = restart['running_jobs']['methylamine']
        self.assertEqual([(j['job_name'], j['job_id'], j['job_status'][0]) for j in refused],
                         [(job_name, None, 'initializing')])

    def _refuse_jobs(self, sched: Scheduler, csv_dir: str) -> list:
        """Run two directed scans and two opt jobs on server1 after its queue stayed unreadable past the bound."""
        sched.servers = ['server1']
        sched.unreadable_queue_servers.setdefault('server1', dict())
        server1 = dict(settings['servers']['server1'], max_simultaneous_jobs=10)
        scan = dict(label='methylamine', xyz=self.XYZ, level_of_theory=Level(repr=default_levels_of_theory['scan']),
                    job_type='directed_scan', directed_scan_type='brute_force_opt', torsions=[[1, 2, 3, 4]],
                    rotor_index=0, job_adapter='gaussian')
        opt = dict(label='methylamine', job_type='opt', level_of_theory=Level(repr=default_levels_of_theory['opt']),
                   job_adapter='gaussian')
        with patch.dict('arc.scheduler.servers_dict', {'server1': server1}), \
                patch('arc.job.adapter.local_arc_path', csv_dir), \
                patch('arc.job.adapter.JobAdapter.execute', side_effect=AssertionError('submitted')), \
                self.assertLogs('arc', level='ERROR'):
            sched.run_job(dihedrals=[30.0], **scan)
            sched.run_job(dihedrals=[60.0], **scan)
            sched.run_job(**opt)
            sched.run_job(**opt)
        return sched.stranded_jobs['methylamine']

    def test_refused_jobs_get_distinct_names_and_all_stay_in_the_restart_file(self):
        csv_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, csv_dir, ignore_errors=True)
        sched = self._scheduler()
        names = self._refuse_jobs(sched, csv_dir)
        self.assertEqual(len(names), 4)
        self.assertEqual(len(set(names)), 4, names)
        self.assertEqual(sorted(sched.job_dict['methylamine']['directed_scan']), sorted(names[:2]))
        self.assertEqual(sorted(sched.job_dict['methylamine']['opt']), sorted(names[2:]))
        self.assertEqual(len({job.job_server_name for jobs in sched.job_dict['methylamine'].values()
                              for job in jobs.values()}), 4)
        restart = read_yaml_file(os.path.join(self.project_directory, 'restart.yml'))
        self.assertEqual(sorted(j['job_name'] for j in restart['running_jobs']['methylamine']), sorted(names))
        with open(os.path.join(csv_dir, 'initiated_jobs.csv')) as f:
            rows = list(csv.reader(f))
        self.assertEqual(sorted(row[7] for row in rows[1:]), sorted(names))
        self.assertEqual({row[8] for row in rows[1:]}, {''})

    def test_a_refused_job_gets_a_row_with_its_job_id_once_submitted(self):
        csv_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, csv_dir, ignore_errors=True)
        sched = self._scheduler()
        self._refuse_jobs(sched, csv_dir)
        csv_path = os.path.join(csv_dir, 'initiated_jobs.csv')
        with open(csv_path) as f:
            rows = f.readlines()
        restart = read_yaml_file(os.path.join(self.project_directory, 'restart.yml'))
        with patch.object(Scheduler, 'schedule_jobs'), patch.object(Scheduler, 'run_job'):
            restarted = self._scheduler(restart_dict={'output': restart['output'],
                                                      'running_jobs': restart['running_jobs']},
                                        testing=False)
        self.assertTrue(all(job.restarted for _, job in restarted.unsubmitted_jobs))
        restarted.unreadable_queue_servers['server1'] = dict()
        with patch('arc.job.adapter.local_arc_path', csv_dir), \
                patch('arc.job.adapter.JobAdapter.execute', side_effect=AssertionError('submitted')), \
                self.assertLogs('arc', level='ERROR'):
            restarted.submit_unsubmitted_jobs()
        restarted.unsubmitted_jobs = [(label, job) for label, names in restarted.stranded_jobs.items()
                                      for jobs in restarted.job_dict[label].values()
                                      for job in jobs.values() if job.job_name in names]
        restarted.unreadable_queue_servers = dict()
        self.server.answer = list()
        server1 = dict(settings['servers']['server1'], max_simultaneous_jobs=10)
        job_ids = iter(range(9001, 9005))

        def dispatch(job, execution_type):
            job.job_id = next(job_ids)
        with patch('arc.job.adapter.local_arc_path', csv_dir), \
                patch('arc.job.adapter.JobAdapter._dispatch_execution', autospec=True,
                      side_effect=dispatch) as dispatched, \
                patch.dict('arc.scheduler.servers_dict', {'server1': server1}), \
                patch('arc.job.adapter.get_default_pool'), \
                patch('time.sleep', side_effect=AssertionError('held')), \
                patch.object(Scheduler, 'warn_on_collapsible_unrestricted_reference'), \
                patch.object(Scheduler, 'save_restart_dict'):
            restarted.submit_unsubmitted_jobs()
        self.assertEqual(dispatched.call_count, 4)
        with open(csv_path) as f:
            lines = f.readlines()
        self.assertEqual(lines[:len(rows)], rows)
        reserved, submitted = list(csv.reader(rows[1:])), list(csv.reader(lines[len(rows):]))
        self.assertEqual(sorted((row[1], row[7]) for row in submitted), sorted((row[1], row[7]) for row in reserved))
        self.assertEqual(sorted(row[8] for row in submitted), ['9001', '9002', '9003', '9004'])

    def test_a_restart_submits_a_refused_job_once_and_does_not_re_derive_it(self):
        """
        A restart submits the restored job itself instead of spawning it again.

        A caller may latch a flag once its job is spawned (e.g., ``ts_conf_spawned``), so re-deriving the job is not
        enough. Once the queue answers again under the limit, the restored job is submitted once.
        """
        sched = self._scheduler()
        self._refuse_an_opt_job(sched)
        job_name = list(sched.job_dict['methylamine']['opt'])[0]
        restart = read_yaml_file(os.path.join(self.project_directory, 'restart.yml'))
        spawned = list()
        with patch.object(Scheduler, 'schedule_jobs'), \
                patch.object(Scheduler, 'run_job',
                             lambda _, **kwargs: spawned.append((kwargs['label'], kwargs['job_type']))):
            restarted = self._scheduler(restart_dict={'output': restart['output'],
                                                      'running_jobs': restart['running_jobs']},
                                        testing=False)
        self.assertNotIn(('methylamine', 'opt'), spawned)
        self.assertEqual(restarted.running_jobs['methylamine'], [job_name])
        self.assertEqual([job.job_name for _, job in restarted.unsubmitted_jobs], [job_name])
        self.assertNotIn(None, restarted.server_job_ids)
        self.server.answer = list()
        server1 = dict(settings['servers']['server1'], max_simultaneous_jobs=10)
        with patch('arc.job.adapter.JobAdapter.execute', autospec=True) as execute, \
                patch.dict('arc.scheduler.servers_dict', {'server1': server1}), \
                patch('time.sleep', side_effect=AssertionError('held')), \
                patch.object(Scheduler, 'warn_on_collapsible_unrestricted_reference') as warn, \
                patch.object(Scheduler, 'save_restart_dict'):
            restarted.submit_unsubmitted_jobs()
            restarted.submit_unsubmitted_jobs()
        self.assertEqual([call.args[0].job_name for call in execute.call_args_list], [job_name])
        self.assertEqual([call.kwargs['job'].job_name for call in warn.call_args_list], [job_name])
        self.assertIn('server1', restarted.servers)
        self.assertEqual(restarted.unsubmitted_jobs, list())

    def test_an_outage_puts_a_capped_submission_off_while_incore_work_and_restart_saves_go_on(self):
        sched = self._scheduler()
        sched.servers = ['server1']
        server1 = dict(settings['servers']['server1'], max_simultaneous_jobs=10)
        with patch.dict('arc.scheduler.servers_dict', {'server1': server1}), \
                patch('time.sleep', side_effect=AssertionError('waited for an unreadable queue')), \
                patch('arc.job.adapter.JobAdapter.execute', side_effect=AssertionError('submitted')):
            sched.run_job(label='methylamine', job_type='opt',
                          level_of_theory=Level(repr=default_levels_of_theory['opt']), job_adapter='gaussian')
        deferred = [name for name in sched.job_dict['methylamine']['opt']
                    if name not in sched.running_jobs.get('methylamine', list())]
        self.assertEqual(len(deferred), 1)
        self.assertEqual(sched.deferred_jobs['methylamine'], deferred)
        self.assertEqual([job.job_name for _, job in sched.unsubmitted_jobs], deferred)
        self.assertIn(deferred[0], sched.unfinished_job_names('methylamine'))
        incore = MagicMock()
        incore.job_name, incore.server, incore.execution_type, incore.remote_project_path = 'sp_a9', 'local', 'incore', None
        with patch('arc.scheduler.job_factory', return_value=incore), \
                patch('time.sleep', side_effect=AssertionError('waited for an unreadable queue')), \
                patch.object(Scheduler, 'warn_on_collapsible_unrestricted_reference'), \
                patch.object(Scheduler, 'save_restart_dict'):
            sched.run_job(label='methylamine', job_type='sp',
                          level_of_theory=Level(repr=default_levels_of_theory['sp']), job_adapter='gaussian')
        incore.execute.assert_called_once()
        restart = read_yaml_file(os.path.join(self.project_directory, 'restart.yml'))
        saved = {j['job_name']: j for j in restart['running_jobs']['methylamine']}
        self.assertIsNone(saved[deferred[0]]['job_id'])
        self.assertEqual(saved[deferred[0]]['job_status'][0], 'initializing')
        self.server.answer = list()
        self.now = 1.5 * self.HOUR
        with patch.dict('arc.scheduler.servers_dict', {'server1': server1}), \
                patch('arc.job.adapter.JobAdapter.execute', autospec=True) as execute, \
                patch('time.sleep', side_effect=AssertionError('held')), \
                patch.object(Scheduler, 'warn_on_collapsible_unrestricted_reference'), \
                patch.object(Scheduler, 'save_restart_dict'):
            sched.submit_unsubmitted_jobs()
        self.assertEqual([call.args[0].job_name for call in execute.call_args_list], deferred)
        self.assertIn(deferred[0], sched.running_jobs['methylamine'])
        self.assertEqual(sched.deferred_jobs, dict())
        self.assertEqual(sched.unsubmitted_jobs, list())

    def test_the_main_loop_retries_put_off_jobs_on_every_pass(self):
        sched = self._scheduler()
        sched.unsubmitted_jobs = [('methylamine', MagicMock())]
        calls, passes = list(), list()

        def submit():
            calls.append(len(calls))
            if len(calls) == 2:
                sched.unsubmitted_jobs = list()

        def strand():
            passes.append(None)
            if len(passes) > 3:
                raise _StopLoop
        with patch.object(sched, 'submit_unsubmitted_jobs', side_effect=submit), \
                patch.object(sched, 'strand_unreadable_queue_jobs', side_effect=strand), \
                patch.object(Scheduler, 'run_conformer_jobs'), patch.object(Scheduler, 'spawn_ts_jobs'), \
                patch.object(Scheduler, 'get_server_job_ids'), patch.object(Scheduler, 'get_completed_incore_jobs'), \
                patch.object(Scheduler, 'generate_final_ts_guess_report'), patch('time.sleep'):
            sched.schedule_jobs()
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(passes), 1)

    def test_per_job_query_failures_are_bounded_though_the_bulk_query_answers(self):
        sched = self._scheduler()
        job = self._running(sched)
        self.server.answer = list()
        with patch.object(job, 'determine_job_status', return_value=False), \
                self.assertLogs('arc', level='ERROR') as logged:
            hours = 0.0
            while hours <= 4.0:
                self.now = hours * self.HOUR
                sched.strand_unreadable_queue_jobs()
                if 'server1' in sched.unreadable_queue_servers:
                    break
                sched.get_server_job_ids()
                self.assertNotIn(4242, sched.server_job_ids)
                self.assertFalse(sched.end_job(job=job, label='methylamine', job_name=job.job_name))
                hours += 0.5
        self.assertEqual(hours, 4.0)
        self.assertEqual(sched.stranded_jobs['methylamine'], [job.job_name])
        self.assertEqual(sched.running_jobs['methylamine'], list())
        self.assertTrue(any('per-job status query' in line for line in logged.output))

    def test_a_job_that_answers_does_not_restart_the_clock_of_one_that_keeps_failing(self):
        sched = self._scheduler()
        failing = self._running(sched)
        answering = SimpleNamespace(job_name=failing.job_name, job_id=4243, server='server1', execution_type='queue')
        with patch.object(failing, 'determine_job_status', return_value=False):
            sched.end_job(job=failing, label='methylamine', job_name=failing.job_name)
        since = sched.job_query_failures['server1']['since']
        self.now = 2 * self.HOUR
        sched.clear_job_query_failure(answering)
        self.assertEqual(sched.job_query_failures['server1']['since'], since)
        sched.clear_job_query_failure(failing)
        self.assertNotIn('server1', sched.job_query_failures)

    def test_a_job_the_bulk_query_lists_again_stops_its_per_job_clock(self):
        sched = self._scheduler()
        job = self._running(sched)
        with patch.object(job, 'determine_job_status', return_value=False):
            sched.end_job(job=job, label='methylamine', job_name=job.job_name)
        self.server.answer = ['4242']
        self.now = self.HOUR
        sched.get_server_job_ids()
        self.assertNotIn('server1', sched.job_query_failures)
        self.now = 5 * self.HOUR
        sched.strand_unreadable_queue_jobs()
        self.assertEqual(sched.unreadable_queue_servers, dict())

    def test_a_server_past_both_outage_limits_is_stranded_once(self):
        sched = self._scheduler()
        job = self._running(sched)
        with patch.object(job, 'determine_job_status', return_value=False):
            sched.end_job(job=job, label='methylamine', job_name=job.job_name)
        with self.assertLogs('arc', level='WARNING'):
            sched.get_server_job_ids()
        self.now = 5 * self.HOUR
        with self.assertLogs('arc', level='ERROR') as logged:
            sched.strand_unreadable_queue_jobs()
        self.assertEqual(len([r for r in logged.records if r.levelno == logging.ERROR]), 1)
        self.assertEqual(sched.stranded_jobs['methylamine'], [job.job_name])
        self.assertEqual((sched.queue_query_failures, sched.job_query_failures), (dict(), dict()))

    def test_a_cancelled_job_leaves_no_per_job_clock_behind(self):
        sched = self._scheduler()
        cancelled = self._running(sched, job_id=4242)
        kept = self._running(sched, job_id=4243)
        sched.job_dict['methylamine']['opt'] = {cancelled.job_name: cancelled, kept.job_name: kept}
        sched.running_jobs['methylamine'] = [cancelled.job_name, kept.job_name]
        with patch.object(cancelled, 'determine_job_status', return_value=False):
            sched.end_job(job=cancelled, label='methylamine', job_name=cancelled.job_name)
        sched.running_jobs['methylamine'].remove(cancelled.job_name)
        self.now = 5 * self.HOUR
        with self.assertNoLogs('arc', level='ERROR'):
            sched.strand_unreadable_queue_jobs()
        self.assertEqual(sched.unreadable_queue_servers, dict())
        self.assertEqual(sched.job_query_failures, dict())

    def test_a_cancelled_job_is_dropped_from_the_per_job_clock_by_a_good_bulk_answer(self):
        """A good bulk answer, asked for another job ARC still tracks on server1, drops the cancelled job's clock."""
        sched = self._scheduler()
        cancelled = self._running(sched, job_id=4242)
        with patch.object(cancelled, 'determine_job_status', return_value=False):
            sched.end_job(job=cancelled, label='methylamine', job_name=cancelled.job_name)
        other = SimpleNamespace(job_name='freq_a2', job_id=4243, server='server1', execution_type='queue')
        sched.job_dict['methylamine']['freq'] = {other.job_name: other}
        sched.running_jobs['methylamine'] = [other.job_name]
        self.server.answer = ['4243']
        sched.get_server_job_ids()
        self.assertEqual(sched.job_query_failures, dict())

    def test_the_per_job_outage_record_keeps_the_real_error(self):
        sched = self._scheduler()
        job = self._running(sched)
        error = QueueQueryError('failed 3 times', server='server1', output='qstat: Unknown Job Id', exit_status=153)
        with patch.object(job, '_check_job_server_status', side_effect=error), \
                self.assertLogs('arc', level='WARNING'):
            self.assertFalse(sched.end_job(job=job, label='methylamine', job_name=job.job_name))
        record = sched.job_query_failures['server1']
        self.assertEqual((record['exit_status'], record['output']), (153, 'qstat: Unknown Job Id'))

    def test_a_bulk_query_that_cannot_reach_its_server_is_an_outage(self):
        """A failure to connect keeps the tracked jobs and starts the outage clock, as a failed query does"""
        sched = self._scheduler()
        self._running(sched)
        for error in (ServerError('Could not connect to server server1'), OSError('Network is unreachable')):
            sched.queue_query_failures = dict()
            with patch('arc.scheduler.borrow_ssh_client', side_effect=error), \
                    self.assertLogs('arc', level='WARNING'):
                sched.get_server_job_ids()
            self.assertEqual(sched.server_job_ids, [4242])
            record = sched.queue_query_failures['server1']
            self.assertIsNone(record['exit_status'])
            self.assertIn(str(error), record['output'])
        self.now = 4 * self.HOUR
        with self.assertLogs('arc', level='ERROR'):
            sched.strand_unreadable_queue_jobs()
        self.assertIn('server1', sched.unreadable_queue_servers)

    def _fail_per_job_query(self, sched: Scheduler, job, hours: float):
        """Fail the per-job status query of ``job`` at ``hours``."""
        self.now = hours * self.HOUR
        with patch.object(job, 'determine_job_status', return_value=False):
            sched.end_job(job=job, label='methylamine', job_name=job.job_name)

    def test_the_outage_start_moves_to_the_earliest_job_still_failing(self):
        """Job A fails at 0 h, job B first fails at 3.99 h; once A is resolved, B's own failure starts the outage"""
        for resolve in ('answer', 'bulk answer', 'cancel'):
            sched = self._scheduler()
            first = self._running(sched, job_id=4242)
            second = self._second_job(sched, 'server1', job_id=4243)
            self._fail_per_job_query(sched, first, 0.0)
            self._fail_per_job_query(sched, second, 3.99)
            if resolve == 'answer':
                sched.clear_job_query_failure(first)
            elif resolve == 'bulk answer':
                self.server.answer = ['4242']
                sched.get_server_job_ids()
            else:
                sched.running_jobs['methylamine'].remove(first.job_name)
            self.now = 4.5 * self.HOUR
            with self.assertNoLogs('arc', level='ERROR'):
                sched.strand_unreadable_queue_jobs()
            self.assertEqual(sched.unreadable_queue_servers, dict(), resolve)
            self.assertEqual(sched.job_query_failures['server1']['since'], 3.99 * self.HOUR, resolve)
            self.now = 7.99 * self.HOUR
            with self.assertLogs('arc', level='ERROR'):
                sched.strand_unreadable_queue_jobs()
            self.assertIn(second.job_name, sched.stranded_jobs['methylamine'], resolve)

    def test_a_running_per_job_answer_after_an_empty_bulk_answer_keeps_the_job(self):
        """The per-job answer says the job still runs, so it is neither rerun nor ended though its output is absent"""
        sched = self._scheduler()
        job = self._running(sched)
        self.server.answer = list()
        sched.get_server_job_ids()
        self.assertNotIn(job.job_id, sched.server_job_ids)
        with patch.object(job, '_check_job_server_status', return_value='running'), \
                patch.object(sched, '_run_a_job') as rerun, patch('time.sleep'):
            self.assertFalse(sched.end_job(job=job, label='methylamine', job_name=job.job_name))
        rerun.assert_not_called()
        self.assertEqual(sched.running_jobs['methylamine'], [job.job_name])
        self.assertEqual(job.job_status[0], 'running')

    def test_a_put_off_job_of_an_unconverged_species_is_not_submitted(self):
        sched = self._scheduler()
        job = self._running(sched)
        job.job_id = None
        sched.defer_job_on_unreadable_queue(label='methylamine', job=job)
        sched.output['methylamine']['convergence'] = False
        self.server.answer = list()
        with patch('arc.job.adapter.JobAdapter.execute', side_effect=AssertionError('submitted')), \
                patch.object(Scheduler, 'save_restart_dict') as save:
            sched.submit_unsubmitted_jobs()
        self.assertEqual(sched.unsubmitted_jobs, list())
        self.assertEqual(sched.deferred_jobs, dict())
        self.assertNotIn(job.job_name, sched.unfinished_job_names('methylamine'))
        save.assert_called()

    def test_a_job_without_an_id_does_not_poll_a_failing_queue_forever(self):
        """An unacknowledged submission is checked by its output, not by a queue query that cannot find it"""
        sched = self._scheduler()
        job = self._running(sched)
        job.job_id = None
        error = QueueQueryError('failed 3 times', server='server1', output='Socket timed out', exit_status=1)
        with patch('arc.job.adapter.check_job_status', side_effect=error) as query, \
                patch.object(job, '_check_job_ess_status'), patch.object(job, '_get_additional_job_info'), \
                patch.object(sched, '_run_a_job') as rerun, patch('time.sleep'), \
                self.assertLogs('arc', level='WARNING'):
            self.assertFalse(sched.end_job(job=job, label='methylamine', job_name=job.job_name))
        query.assert_not_called()
        rerun.assert_called_once()
        self.assertEqual(sched.job_query_failures, dict())

    def test_a_job_put_off_on_many_passes_is_logged_and_saved_once(self):
        """Putting off an already put-off job changes nothing, so it logs no line and saves no restart file."""
        sched = self._scheduler()
        sched.servers = ['server1']
        server1 = dict(settings['servers']['server1'], max_simultaneous_jobs=10)
        with patch.dict('arc.scheduler.servers_dict', {'server1': server1}), \
                patch('time.sleep', side_effect=AssertionError('waited for an unreadable queue')), \
                patch('arc.job.adapter.JobAdapter.execute', side_effect=AssertionError('submitted')), \
                patch.object(Scheduler, 'save_restart_dict') as save, \
                self.assertLogs('arc', level='INFO') as logged:
            sched.run_job(label='methylamine', job_type='opt',
                          level_of_theory=Level(repr=default_levels_of_theory['opt']), job_adapter='gaussian')
            for minutes in range(2, 42, 2):
                self.now = minutes * 60.0
                sched.submit_unsubmitted_jobs()
        put_off = [line for line in logged.output if 'Put off job' in line]
        self.assertEqual(len(put_off), 1, put_off)
        self.assertEqual(save.call_count, 1)
        self.assertEqual(len(sched.unsubmitted_jobs), 1)
        label, job = sched.unsubmitted_jobs[0]
        self.assertIn(job.job_name, sched.deferred_jobs[label])
        self.server.answer = list()
        self.now = 50 * 60.0
        with patch.dict('arc.scheduler.servers_dict', {'server1': server1}), \
                patch('arc.job.adapter.JobAdapter.execute') as execute:
            sched.submit_unsubmitted_jobs()
        execute.assert_called_once()
        self.assertEqual((sched.unsubmitted_jobs, sched.deferred_jobs), (list(), dict()))
        self.assertIn(job.job_name, sched.running_jobs[label])

    def test_deleting_a_species_jobs_drops_its_put_off_jobs(self):
        sched = self._scheduler()
        sched.servers = ['server1']
        server1 = dict(settings['servers']['server1'], max_simultaneous_jobs=10)
        with patch.dict('arc.scheduler.servers_dict', {'server1': server1}), \
                patch('time.sleep', side_effect=AssertionError('waited for an unreadable queue')), \
                patch('arc.job.adapter.JobAdapter.execute', side_effect=AssertionError('submitted')):
            sched.run_job(label='methylamine', job_type='opt',
                          level_of_theory=Level(repr=default_levels_of_theory['opt']), job_adapter='gaussian')
        self.assertTrue(sched.unsubmitted_jobs)
        sched.delete_all_species_jobs('methylamine')
        self.assertEqual((sched.deferred_jobs, sched.unsubmitted_jobs), (dict(), list()))
        self.server.answer = list()
        with patch('arc.job.adapter.JobAdapter.execute', side_effect=AssertionError('submitted')):
            sched.submit_unsubmitted_jobs()

    def test_a_per_job_clock_with_no_tracked_job_left_strands_nothing(self):
        sched = self._scheduler()
        job = self._running(sched)
        with patch.object(job, 'determine_job_status', return_value=False):
            sched.end_job(job=job, label='methylamine', job_name=job.job_name)
        self.assertIn('server1', sched.job_query_failures)
        del sched.running_jobs['methylamine']
        self.now = 5 * self.HOUR
        with self.assertNoLogs('arc', level='ERROR'):
            sched.strand_unreadable_queue_jobs()
        self.assertEqual(sched.unreadable_queue_servers, dict())
        self.assertEqual(sched.job_query_failures, dict())

    def test_a_failed_per_job_query_waits_out_the_cooldown(self):
        """A job whose own status query failed reads as running, and is not queried, for ``QUEUE_QUERY_COOLDOWN``."""
        sched = self._scheduler()
        job = self._running(sched)
        self.server.answer = list()
        with patch.object(job, 'determine_job_status', return_value=False) as query:
            sched.get_server_job_ids()
            self.assertNotIn(4242, sched.server_job_ids)
            self.assertFalse(sched.end_job(job=job, label='methylamine', job_name=job.job_name))
            self.now = QUEUE_QUERY_COOLDOWN - 1.0
            sched.get_server_job_ids()
            self.assertIn(4242, sched.server_job_ids)
            self.now = QUEUE_QUERY_COOLDOWN + 1.0
            sched.get_server_job_ids()
            self.assertNotIn(4242, sched.server_job_ids)
        self.assertEqual(query.call_count, 1)

    def test_each_failed_job_waits_out_its_own_cooldown(self):
        """A later failure of another job on the same server does not extend the cooldown of the first."""
        sched = self._scheduler()
        first = self._running(sched)
        second = self._second_job(sched, server='server1', job_id=5151)
        self.server.answer = list()
        with patch.object(first, 'determine_job_status', return_value=False), \
                patch.object(second, 'determine_job_status', return_value=False):
            sched.end_job(job=first, label='methylamine', job_name=first.job_name)
            self.now = QUEUE_QUERY_COOLDOWN - 10.0
            sched.end_job(job=second, label='methylamine', job_name=second.job_name)
        self.now = QUEUE_QUERY_COOLDOWN + 1.0
        sched.get_server_job_ids()
        self.assertNotIn(4242, sched.server_job_ids)
        self.assertIn(5151, sched.server_job_ids)

    def test_a_job_in_cooldown_does_not_hide_a_job_with_its_id_on_another_server(self):
        """A job ID in cooldown on one server is not listed as running when another server's job holds that ID."""
        sched = self._scheduler()
        first = self._running(sched)
        self._second_job(sched, server='server2', job_id=4242)
        sched.servers = ['server1', 'server2']
        self.server.answer = list()
        with patch.object(first, 'determine_job_status', return_value=False):
            sched.end_job(job=first, label='methylamine', job_name=first.job_name)
        self.now = 1.0
        sched.get_server_job_ids()
        self.assertNotIn(4242, sched.server_job_ids)

    def test_the_main_loop_sleeps_and_spaces_per_job_queries_while_they_fail(self):
        """
        Over 40 main-loop passes with the bulk query answering and the per-job query failing, every pass sleeps 30 s,
        and the job is queried once per ``QUEUE_QUERY_COOLDOWN``, not on every pass.
        """
        sched = self._scheduler()
        job = self._running(sched)
        self.server.answer = list()
        passes, sleeps = list(), list()

        def strand():
            passes.append(self.now)
            if len(passes) > 40:
                raise _StopLoop

        def sleep(seconds):
            sleeps.append(seconds)
            self.now += seconds
        with patch.object(job, 'determine_job_status', return_value=False) as query, \
                patch.object(sched, 'strand_unreadable_queue_jobs', side_effect=strand), \
                patch.object(Scheduler, 'run_conformer_jobs'), patch.object(Scheduler, 'spawn_ts_jobs'), \
                patch.object(Scheduler, 'get_completed_incore_jobs'), \
                patch.object(Scheduler, 'generate_final_ts_guess_report'), \
                patch('time.sleep', side_effect=sleep), self.assertRaises(_StopLoop):
            sched.schedule_jobs()
        self.assertEqual(sleeps, [30] * 40)
        self.assertEqual(query.call_count, 20)
        self.assertEqual(sched.running_jobs['methylamine'], [job.job_name])

    def test_a_server_with_no_tracked_jobs_is_not_queried(self):
        """A check of all servers skips a server where ARC tracks no queue job, so its outage never starts."""
        sched = self._scheduler()
        sched.servers = ['server1']
        with self.assertNoLogs('arc', level='WARNING'):
            sched.get_server_job_ids()
        self.assertEqual(self.server.queries, 0)
        self.assertEqual(sched.queue_query_failures, dict())

    def test_an_outage_with_no_job_left_on_the_server_strands_nothing(self):
        """A job queue outage on a server whose tracked jobs are all gone is dropped, not stranded at the bound."""
        sched = self._scheduler()
        self._running(sched)
        with self.assertLogs('arc', level='WARNING'):
            self._outage(sched, 1.0)
        self.assertIn('server1', sched.queue_query_failures)
        sched.running_jobs['methylamine'] = list()
        self.now = 5 * self.HOUR
        with self.assertNoLogs('arc', level='ERROR'):
            sched.strand_unreadable_queue_jobs()
        self.assertEqual(sched.queue_query_failures, dict())
        self.assertEqual(sched.unreadable_queue_servers, dict())
        self.assertFalse(sched.keeps_local_check_files())

    def test_a_put_off_job_keeps_its_queue_outage_running_until_it_is_held_back(self):
        """
        A job put off on an idle server whose queue cannot be read keeps that outage's clock running, so the job is
        held back for a restart at the bound, with one ERROR naming it, and no ERROR strands the idle server.
        """
        sched = self._scheduler()
        sched.servers = ['server1']
        server1 = dict(settings['servers']['server1'], max_simultaneous_jobs=10)
        with patch.dict('arc.scheduler.servers_dict', {'server1': server1}), \
                patch('time.sleep', side_effect=AssertionError('waited for an unreadable queue')), \
                patch('arc.job.adapter.JobAdapter.execute', side_effect=AssertionError('submitted')), \
                self.assertLogs('arc', level='WARNING') as logged:
            sched.run_opt_job('methylamine')
            job_name = list(sched.job_dict['methylamine']['opt'])[0]
            hours = 0.25
            while hours <= 4.0:
                self.now = hours * self.HOUR
                sched.strand_unreadable_queue_jobs()
                if hours < 4.0:
                    self.assertEqual(sched.deferred_jobs['methylamine'], [job_name])
                sched.submit_unsubmitted_jobs()
                hours += 0.25
        errors = [record.getMessage() for record in logged.records if record.levelno == logging.ERROR]
        self.assertEqual(len(errors), 1, errors)
        self.assertIn(f'Did not submit job {job_name}', errors[0])
        self.assertEqual(sched.stranded_jobs['methylamine'], [job_name])

    def test_schedule_jobs_submits_a_restored_held_back_job(self):
        sched = self._scheduler()
        self._refuse_an_opt_job(sched)
        job_name = list(sched.job_dict['methylamine']['opt'])[0]
        restart = read_yaml_file(os.path.join(self.project_directory, 'restart.yml'))
        with patch.object(Scheduler, 'schedule_jobs'), patch.object(Scheduler, 'run_job'):
            restarted = self._scheduler(restart_dict={'output': restart['output'],
                                                      'running_jobs': restart['running_jobs']},
                                        testing=False)
        self.server.answer = list()
        server1 = dict(settings['servers']['server1'], max_simultaneous_jobs=10)
        with patch('arc.job.adapter.JobAdapter.execute', autospec=True) as execute, \
                patch.dict('arc.scheduler.servers_dict', {'server1': server1}), \
                patch('time.sleep', side_effect=AssertionError('held')), \
                patch.object(Scheduler, 'run_conformer_jobs'), patch.object(Scheduler, 'spawn_ts_jobs'), \
                patch.object(Scheduler, 'warn_on_collapsible_unrestricted_reference'), \
                patch.object(Scheduler, 'save_restart_dict'), \
                patch.object(Scheduler, 'strand_unreadable_queue_jobs', side_effect=_StopLoop):
            with self.assertRaises(_StopLoop):
                restarted.schedule_jobs()
        self.assertEqual([call.args[0].job_name for call in execute.call_args_list], [job_name])

    def test_a_restart_drops_the_errors_an_unreadable_queue_wrote(self):
        sched = self._scheduler()
        self._running(sched)
        sched.output['methylamine']['errors'] = 'An earlier error; '
        with self.assertLogs('arc', level='WARNING'):
            self._outage(sched, 4.0)
        self._refuse_an_opt_job(sched)
        restart = read_yaml_file(os.path.join(self.project_directory, 'restart.yml'))
        errors = restart['output']['methylamine']['errors']
        self.assertIn('was unreadable past the outage limit', errors)
        self.assertIn('was not submitted to server1', errors)
        restarted = self._scheduler(restart_dict={'output': restart['output'],
                                                  'running_jobs': restart['running_jobs']})
        self.assertEqual(restarted.output['methylamine']['errors'], 'An earlier error; ')

    def test_conformers_are_not_spawned_again_while_a_conformer_job_is_unfinished(self):
        for tracked in ('running_jobs', 'stranded_jobs', None):
            with self.subTest(tracked=tracked):
                sched = self._scheduler()
                sched.dont_gen_confs = ['methylamine']
                if tracked is not None:
                    getattr(sched, tracked)['methylamine'] = ['conf_opt_0']
                with patch.object(Scheduler, 'process_conformers') as process:
                    sched.run_conformer_jobs()
                self.assertEqual(process.call_count, 0 if tracked else 1)

    def test_a_restart_refuses_an_unsubmitted_job_again_while_its_queue_stays_unreadable(self):
        sched = self._scheduler()
        self._refuse_an_opt_job(sched)
        job_name = list(sched.job_dict['methylamine']['opt'])[0]
        restart = read_yaml_file(os.path.join(self.project_directory, 'restart.yml'))
        with patch.object(Scheduler, 'schedule_jobs'), patch.object(Scheduler, 'run_job'):
            restarted = self._scheduler(restart_dict={'output': restart['output'],
                                                      'running_jobs': restart['running_jobs']},
                                        testing=False)
        restarted.unreadable_queue_servers['server1'] = dict()
        with patch('arc.job.adapter.JobAdapter.execute', side_effect=AssertionError('submitted')), \
                self.assertLogs('arc', level='ERROR'):
            restarted.submit_unsubmitted_jobs()
        self.assertEqual(restarted.running_jobs['methylamine'], list())
        self.assertEqual(restarted.stranded_jobs['methylamine'], [job_name])

    def test_cancelling_a_species_drops_its_unsubmitted_stranded_job(self):
        sched = self._scheduler()
        self._refuse_an_opt_job(sched)
        with patch('arc.job.adapter.JobAdapter.delete') as delete:
            sched.delete_all_species_jobs('methylamine')
        delete.assert_not_called()
        self.assertNotIn('methylamine', sched.stranded_jobs)

    def test_a_cancelled_species_submits_nothing_after_a_restart(self):
        sched = self._scheduler()
        self._refuse_an_opt_job(sched)
        with patch('arc.job.adapter.JobAdapter.delete') as delete:
            sched.delete_all_species_jobs('methylamine')
        delete.assert_not_called()
        sched.save_restart_dict()
        restart = read_yaml_file(os.path.join(self.project_directory, 'restart.yml'))
        with patch.object(Scheduler, 'schedule_jobs'), patch.object(Scheduler, 'run_job'):
            restarted = self._scheduler(restart_dict={'output': restart['output'],
                                                      'running_jobs': restart['running_jobs']},
                                        testing=False)
        self.server.answer = list()
        server1 = dict(settings['servers']['server1'], max_simultaneous_jobs=10)
        with patch('arc.job.adapter.JobAdapter.execute', autospec=True) as execute, \
                patch.dict('arc.scheduler.servers_dict', {'server1': server1}), \
                patch('time.sleep', side_effect=AssertionError('held')), \
                patch.object(Scheduler, 'warn_on_collapsible_unrestricted_reference'), \
                patch.object(Scheduler, 'save_restart_dict'):
            restarted.submit_unsubmitted_jobs()
        execute.assert_not_called()
        self.assertNotIn('methylamine', [label for label, _ in restarted.unsubmitted_jobs])
        self.assertFalse(restarted.stranded_jobs.get('methylamine'))
        self.assertFalse(restart['running_jobs'].get('methylamine'))

    def test_cancelling_a_species_cancels_its_submitted_stranded_job(self):
        sched = self._scheduler()
        job = self._running(sched)
        with self.assertLogs('arc', level='WARNING'):
            self._outage(sched, 4.0)
        self.assertEqual(sched.stranded_jobs, {'methylamine': [job.job_name]})
        with patch.object(job, 'delete') as delete, self.assertLogs('arc', level='WARNING') as logged:
            sched.delete_all_species_jobs('methylamine')
        delete.assert_called_once_with()
        self.assertNotIn('methylamine', sched.stranded_jobs)
        warnings = [record.getMessage() for record in logged.records if record.levelno == logging.WARNING]
        self.assertEqual(len(warnings), 1, warnings)
        for expected in (job.job_name, '4242', 'server1', 'cannot confirm', 'may still be running'):
            self.assertIn(expected, warnings[0])

    def test_a_stranded_job_that_cannot_be_cancelled_is_dropped_with_a_warning(self):
        sched = self._scheduler()
        job = self._running(sched)
        with self.assertLogs('arc', level='WARNING'):
            self._outage(sched, 4.0)
        with patch.object(job, 'delete', side_effect=OSError('connection refused')), \
                self.assertLogs('arc', level='WARNING') as logged:
            sched.delete_all_species_jobs('methylamine')
        warnings = [record.getMessage() for record in logged.records if record.levelno == logging.WARNING]
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn(job.job_name, warnings[0])
        self.assertIn('4242', warnings[0])
        self.assertIn('server1', warnings[0])
        self.assertIn('connection refused', warnings[0])
        self.assertIn('may still be running', warnings[0])
        self.assertNotIn('methylamine', sched.stranded_jobs)

    def test_an_absent_or_malformed_setting_bounds_at_four_hours(self):
        with patch.dict('arc.scheduler.settings', {'queue_outage_limit_hrs': None}):
            self.assertEqual(self._scheduler().queue_outage_limit, 4 * self.HOUR)
        for value in ('four', -2, True):
            with patch.dict('arc.scheduler.settings', {'queue_outage_limit_hrs': value}), \
                    self.assertLogs('arc', level='WARNING') as logged:
                sched = self._scheduler()
            self.assertEqual(sched.queue_outage_limit, 4 * self.HOUR, value)
            self.assertTrue(any('queue_outage_limit_hrs' in line for line in logged.output))
        with patch.dict('arc.scheduler.settings', {'queue_outage_limit_hrs': 1}):
            self.assertEqual(self._scheduler().queue_outage_limit, self.HOUR)


class _StopLoop(Exception):
    """Ends a test's pass through the scheduler's main loop."""


class TestTrackedWorkInTheMainLoop(unittest.TestCase):
    """A job whose status is unknown, or that was left on an unreadable queue, is not treated as finished (#394)."""

    JOB_TYPES = {'conf_opt': False, 'conf_sp': False, 'opt': True, 'fine': False, 'freq': True, 'sp': True,
                 'rotors': False, 'orbitals': False, 'lennard_jones': False, 'irc': False, 'bde': False}

    def setUp(self):
        project_directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        self.label = 'methylamine'
        self.sched = Scheduler(project='tracked_work', ess_settings={'gaussian': ['server1']},
                               species_list=[ARCSpecies(label=self.label, smiles='CN')],
                               composite_method=None,
                               conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                               opt_level=Level(repr=default_levels_of_theory['opt']),
                               freq_level=Level(repr=default_levels_of_theory['freq']),
                               sp_level=Level(repr=default_levels_of_theory['sp']),
                               scan_level=Level(repr=default_levels_of_theory['scan']),
                               ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                               project_directory=project_directory, testing=True,
                               job_types=dict(self.JOB_TYPES), restart_dict=dict(),
                               orbitals_level=default_levels_of_theory['orbitals'], adaptive_levels=None)
        for name in ('run_conformer_jobs', 'spawn_ts_jobs', 'get_completed_incore_jobs',
                     'generate_final_ts_guess_report', 'determine_most_stable_conformer', 'run_opt_job',
                     'process_directed_scans', 'parse_conformer'):
            patcher = patch.object(self.sched, name, return_value=False)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        patcher = patch.object(self.sched, 'get_server_job_ids',
                               side_effect=lambda specific_server=None: setattr(self.sched, 'server_job_ids', list()))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _job(self, job_type: str, job_id: int, **kwargs) -> MagicMock:
        job = MagicMock(job_type=job_type, job_id=job_id, server='server1', execution_type='queue', **kwargs)
        job.as_dict.return_value = {'job_type': job_type, 'job_id': job_id}
        return job

    def _ending(self, *removed: str):
        """An end_job() that ends the named jobs: it drops them from running_jobs and reports success."""
        def end_job(job, label, job_name):
            if job_name in removed:
                self.sched.running_jobs[label].remove(job_name)
                return True
            raise _StopLoop(job_name)
        return end_job

    def _directed_scan(self) -> tuple[MagicMock, str]:
        local_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, local_path, ignore_errors=True)
        job = self._job('directed_scan', 51, job_name='directed_scan_a51', directed_scan_type='brute_force_sp',
                        rotor_index=0, pivots=[1, 2], local_path=local_path)
        self.sched.species_dict[self.label].rotors_dict = {0: {'number_of_running_jobs': 1, 'pivots': [1, 2]}}
        self.sched.job_dict[self.label]['directed_scan'] = {job.job_name: job}
        self.sched.running_jobs[self.label] = [job.job_name]
        return job, local_path

    def test_a_directed_scan_of_unknown_status_keeps_its_count_and_its_files(self):
        """A failed per-job query (``end_job()`` returns ``False`` and keeps the job running) leaves the scan alone."""
        job, local_path = self._directed_scan()
        calls = list()

        def end_job(job, label, job_name):
            calls.append(job_name)
            if len(calls) > 1:
                raise _StopLoop(job_name)
            return False
        with patch.object(self.sched, 'end_job', side_effect=end_job), self.assertRaises(_StopLoop):
            self.sched.schedule_jobs()
        self.assertEqual(self.sched.species_dict[self.label].rotors_dict[0]['number_of_running_jobs'], 1)
        self.assertTrue(os.path.isdir(local_path))
        self.assertEqual(self.sched.running_jobs[self.label], [job.job_name])
        self.process_directed_scans.assert_not_called()

    def test_a_directed_scan_that_failed_for_good_is_still_counted_and_cleaned(self):
        """A terminal failure, after which ``end_job()`` no longer tracks the job, still counts and cleans the scan."""
        job, local_path = self._directed_scan()

        def end_job(job, label, job_name):
            self.sched.running_jobs[label].remove(job_name)
            return False
        with patch.object(self.sched, 'end_job', side_effect=end_job), \
                patch.object(self.sched, 'check_all_done'):
            self.sched.schedule_jobs()
        self.assertEqual(self.sched.species_dict[self.label].rotors_dict[0]['number_of_running_jobs'], 0)
        self.assertFalse(os.path.isdir(local_path))
        self.process_directed_scans.assert_called_once()

    def test_a_stranded_conformer_job_holds_back_the_choice_of_conformer(self):
        """A stranded conformer job is not polled, and keeps the conformer choice and the opt job from starting."""
        done, stranded = self._job('conf_opt', 60), self._job('conf_opt', 61)
        self.sched.job_dict[self.label]['conf_opt'] = {0: done, 1: stranded}
        self.sched.running_jobs[self.label] = ['conf_opt_0']
        self.sched.stranded_jobs[self.label] = ['conf_opt_1']
        with patch.object(self.sched, 'end_job', side_effect=self._ending('conf_opt_0')) as end_job:
            self.sched.schedule_jobs()
        self.assertEqual([c.kwargs['job_name'] for c in end_job.call_args_list], ['conf_opt_0'])
        self.determine_most_stable_conformer.assert_not_called()
        self.run_opt_job.assert_not_called()
        self.assertNotEqual(self.sched.output[self.label]['convergence'], True)

    def test_cancelling_a_species_deletes_its_stranded_conformer_and_ts_guess_jobs(self):
        conformer = self._job('conf_opt', 61, job_name='conf_opt_1')
        guess = self._job('tsg', 62, job_name='tsg3')
        self.sched.job_dict[self.label]['conf_opt'] = {1: conformer}
        self.sched.job_dict[self.label]['tsg'] = {3: guess}
        self.sched.stranded_jobs[self.label] = ['conf_opt_1', 'tsg3']
        self.sched.delete_all_species_jobs(self.label)
        conformer.delete.assert_called_once_with()
        guess.delete.assert_called_once_with()
        self.assertNotIn(self.label, self.sched.stranded_jobs)

    def test_a_stranded_ts_guess_job_holds_back_the_conformers(self):
        done, stranded = self._job('tsg', 70, local_path_to_output_file='out.txt'), self._job('tsg', 71)
        self.sched.job_dict[self.label]['tsg'] = {0: done, 1: stranded}
        self.sched.running_jobs[self.label] = ['tsg0']
        self.sched.stranded_jobs[self.label] = ['tsg1']
        with patch.object(self.sched, 'end_job', side_effect=self._ending('tsg0')) as end_job:
            self.sched.schedule_jobs()
        self.assertEqual([c.kwargs['job_name'] for c in end_job.call_args_list], ['tsg0'])
        self.assertNotIn(([], {'labels': [self.label]}), [(list(c.args), c.kwargs)
                                                           for c in self.run_conformer_jobs.call_args_list])

    def test_check_all_done_does_not_converge_a_species_with_a_stranded_job(self):
        for job_type, spawned in self.JOB_TYPES.items():
            if spawned:
                self.sched.output[self.label]['job_types'][job_type] = True
        self.sched.job_dict[self.label]['conf_opt'] = {1: self._job('conf_opt', 61)}
        self.sched.stranded_jobs[self.label] = ['conf_opt_1']
        with self.assertNoLogs('arc', level='ERROR'):
            self.sched.check_all_done(self.label)
        self.assertNotEqual(self.sched.output[self.label]['convergence'], True)


class TestTsGuessPathProvenance(unittest.TestCase):
    """Tests method-specific routing of TS path-search artifacts."""

    def test_method_specific_path_keys(self):
        from arc.scheduler import _ts_guess_paths_key

        self.assertEqual(_ts_guess_paths_key('orca_neb'), 'neb')
        self.assertEqual(_ts_guess_paths_key(' xTB-GSM '), 'gsm')
        self.assertIsNone(_ts_guess_paths_key('gcn'))
        self.assertIsNone(_ts_guess_paths_key(None))

    def test_merged_guess_path_provenance(self):
        from arc.scheduler import _ts_guess_path_provenance

        guess = TSGuess(index=0, method='gcn', success=True, xyz='C 0 0 0')
        guess.method_sources = ['gcn', 'xtb-gsm']
        guess.method_source_paths = {'xtb-gsm': '/run/stringfile.xyz0000'}
        self.assertEqual(
            _ts_guess_path_provenance(guess),
            ('gsm', '/run/stringfile.xyz0000'),
        )


class TestScanSoftwareStamp(unittest.TestCase):
    """Tests that a completed scan job stamps ``rotors_dict[...]['scan_software']``.

    ``arc/output.py``'s ``_scan_constraint_parser_for`` dispatches the
    rotor-scan constraint parser on this field. Nothing sets it unless the
    scheduler stamps it here (``Scheduler.end_job`` and
    ``Scheduler.check_scan_job``); without the stamp, dispatch silently
    falls back to the Gaussian parser for every scan regardless of which
    software actually produced the log. These tests drive the two
    producer call sites directly with a ``MagicMock`` ``self`` (as
    ``TestHasPendingPipeWork`` above does) so they don't depend on the
    ``~/.arc`` server configuration that ``TestScheduler``'s job fixtures
    require.
    """

    def _rotor(self):
        return {
            'pivots': [2, 3],
            'dimensions': 1,
            'scan': [1, 2, 3, 4],
            'torsion': [0, 1, 2, 3],
            'trsh_methods': [],
            'invalidation_reason': '',
            'success': None,
            'scan_path': '',
            # deliberately no 'scan_software' key, mirroring
            # find_internal_rotors()'s output.
        }

    def test_end_job_stamps_scan_software_for_a_finished_scan(self):
        from arc.scheduler import Scheduler

        rotor = self._rotor()
        rotor['pivots'] = [2, 3]
        spc = MagicMock()
        spc.rotors_dict = {0: rotor}

        sched = MagicMock()
        sched.species_dict = {'spc': spc}
        sched.project_directory = '/fake/project'
        sched.output = {}
        sched.running_jobs = {'spc': ['scan_job']}

        # A real (empty) file: end_job() retries with real time.sleep() calls
        # until os.path.isfile() succeeds, so a nonexistent path would hang
        # the test for ~50s before falling into an unrelated branch.
        output_file = tempfile.NamedTemporaryFile(suffix='.out', delete=False)
        output_file.close()
        self.addCleanup(os.remove, output_file.name)

        job = MagicMock()
        job.job_status = ['done', {'status': 'done'}]
        job.job_adapter = 'orca'
        job.job_type = 'scan'
        job.directed_scan_type = 'ess'
        job.pivots = [2, 3]
        job.local_path_to_output_file = output_file.name
        job.local_path = '/fake/local_path'
        job.species_label = 'spc'
        job.job_name = 'scan_job'
        job.execution_type = 'incore'

        result = Scheduler.end_job(sched, job=job, label='spc', job_name='scan_job')

        self.assertTrue(result)
        self.assertEqual(rotor['scan_software'], 'orca')
        self.assertEqual(rotor['scan_path'], output_file.name)

    def test_check_scan_job_stamps_scan_software(self):
        from arc.scheduler import Scheduler

        rotor = self._rotor()
        spc = MagicMock()
        spc.rotors_dict = {0: rotor}
        spc.get_xyz.return_value = {'coords': (), 'symbols': ()}
        spc.final_xyz = None
        spc.preserve_param_in_scan = None
        spc.is_ts = False

        sched = MagicMock()
        sched.species_dict = {'spc': spc}
        sched.project_directory = '/fake/project'
        sched.trsh_ess_jobs = False
        sched.trsh_rotors = False

        job = MagicMock()
        job.job_status = [None, {'status': 'done', 'error': None}]
        job.rotor_index = 0
        job.local_path_to_output_file = '/fake/scan.out'
        job.job_adapter = 'orca'
        job.torsions = [[0, 1, 2, 3]]
        job.xyz = {'coords': (), 'symbols': ()}
        job.is_ts = False
        job.species_label = 'spc'

        # Drive the "energies could not be read" branch: it reaches the
        # scan_software stamp without needing determine_rotor_symmetry(),
        # scan_quality_check(), or real log-file parsing.
        with patch('arc.scheduler.parser.parse_1d_scan_energies_from_specific_angle',
                   return_value=(None, None)), \
             patch('arc.scheduler.calculate_dihedral_angle', return_value=0.0):
            Scheduler.check_scan_job(sched, label='spc', job=job)

        self.assertEqual(rotor['scan_software'], 'orca')
        self.assertEqual(rotor['scan_path'], '/fake/scan.out')
        self.assertFalse(rotor['success'])

    def test_check_scan_job_keeps_a_reactive_ring_rejection(self):
        """A good scan that finishes after the reactive-ring check rejected its rotor (e.g., restored on a restart)
        does not make the rotor a hindered rotor again."""
        from arc.scheduler import Scheduler

        rotor = self._rotor()
        rotor['success'], rotor['invalidation_reason'] = False, REACTIVE_RING_INVALIDATION_REASON
        spc = MagicMock()
        spc.rotors_dict = {0: rotor}
        spc.get_xyz.return_value = {'coords': (), 'symbols': ()}
        spc.final_xyz = None
        spc.preserve_param_in_scan = None
        spc.is_ts = True

        sched = MagicMock()
        sched.species_dict = {'TS0': spc}
        sched.project_directory = '/fake/project'
        sched.trsh_ess_jobs = False
        sched.trsh_rotors = False

        job = MagicMock()
        job.job_status = [None, {'status': 'done', 'error': None}]
        job.rotor_index = 0
        job.local_path_to_output_file = '/fake/scan.out'
        job.job_adapter = 'orca'
        job.torsions = [[0, 1, 2, 3]]
        job.xyz = {'coords': (), 'symbols': ()}
        job.is_ts = True
        job.species_label = 'TS0'

        with patch('arc.scheduler.parser.parse_1d_scan_energies_from_specific_angle',
                   return_value=([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0], list(range(7)))), \
             patch('arc.scheduler.parser.parse_1d_scan_coords', return_value=list()), \
             patch('arc.scheduler.calculate_dihedral_angle', return_value=0.0), \
             patch('arc.scheduler.scan_quality_check', return_value=(False, '', '', dict())), \
             patch('arc.scheduler.determine_rotor_symmetry', return_value=(1, 3.0, 0.0)), \
             patch('arc.scheduler.plotter.plot_1d_rotor_scan'):
            Scheduler.check_scan_job(sched, label='TS0', job=job)

        self.assertIs(rotor['success'], False)
        self.assertEqual(rotor['invalidation_reason'], REACTIVE_RING_INVALIDATION_REASON)

    def test_run_scan_jobs_does_not_rescan_a_reactive_ring_rejection(self):
        """A rejected reactive-ring rotor whose previous scan file is missing (e.g., on a restart) is not scanned again."""
        from arc.scheduler import Scheduler

        rotor = self._rotor()
        rotor.update({'success': False, 'invalidation_reason': REACTIVE_RING_INVALIDATION_REASON,
                      'scan_path': '/missing/scan.out', 'torsion': [[0, 1, 2, 3]], 'directed_scan_type': 'ess'})
        spc = MagicMock()
        spc.rotors_dict = {0: rotor}
        spc.multi_species = None
        sched = MagicMock()
        sched.job_types = {'rotors': True}
        sched.species_dict = {'TS0': spc}
        sched.job_dict = {'TS0': dict()}
        sched.running_jobs = {'TS0': list()}
        sched.pipe_planner.try_pipe_rotor_scans_1d.return_value = set()

        Scheduler.run_scan_jobs(sched, label='TS0')

        sched.run_job.assert_not_called()
        sched.spawn_directed_scan_jobs.assert_not_called()
        self.assertIs(rotor['success'], False)
        self.assertEqual(rotor['scan_path'], '/missing/scan.out')

    def test_continuous_scan_is_not_continued_for_a_reactive_ring_rejection(self):
        """A restored continuous directed scan does not spawn its next point once its rotor is a reactive-ring rejection."""
        from arc.scheduler import Scheduler

        rotor = self._rotor()
        rotor.update({'success': False, 'invalidation_reason': REACTIVE_RING_INVALIDATION_REASON,
                      'directed_scan_type': 'cont_opt', 'directed_scan': dict(), 'original_dihedrals': [60.0],
                      'cont_indices': list()})
        spc = MagicMock()
        spc.rotors_dict = {0: rotor}
        sched = MagicMock()
        sched.species_dict = {'TS0': spc}
        sched.job_dict = {'TS0': {'directed_scan': dict()}}

        Scheduler.spawn_directed_scan_jobs(sched, label='TS0', rotor_index=0,
                                           xyz={'symbols': ('C', 'C', 'C', 'C'), 'isotopes': (12, 12, 12, 12),
                                                'coords': ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0),
                                                           (2.0, 1.4, 0.0), (3.5, 1.4, 0.5))})

        sched.run_job.assert_not_called()
        self.assertIs(rotor['success'], False)

    def test_end_job_does_not_resubmit_a_reactive_ring_rejection_on_insufficient_memory(self):
        """A restored scan of a rejected reactive-ring rotor that fails for memory is not resubmitted by end_job(),
        neither as a scan nor as a directed scan, and the job still leaves running_jobs."""
        from arc.scheduler import Scheduler

        output_file = tempfile.NamedTemporaryFile(suffix='.out', delete=False)
        output_file.close()
        self.addCleanup(os.remove, output_file.name)
        for job_type, directed_scan_type in [('scan', 'ess'), ('directed_scan', 'brute_force_opt')]:
            with self.subTest(job_type=job_type):
                rotor = self._rotor()
                rotor.update({'success': False, 'invalidation_reason': REACTIVE_RING_INVALIDATION_REASON,
                              'directed_scan_type': directed_scan_type})
                spc = MagicMock()
                spc.rotors_dict = {0: rotor}
                sched = MagicMock()
                sched.species_dict = {'TS0': spc}
                sched.output = {'TS0': dict()}
                sched.running_jobs = {'TS0': ['scan_job']}
                sched.adaptive_levels = None
                sched.max_job_time = 120
                sched.memory = 14
                sched._run_a_job = lambda job, label, rerun=False: Scheduler._run_a_job(sched, job, label, rerun)
                sched.run_job = lambda **kwargs: Scheduler.run_job(sched, **kwargs)

                job = MagicMock()
                job.job_status = ['errored', {'status': 'errored', 'keywords': ['memory'],
                                              'error': 'Insufficient job memory.'}]
                job.job_type = job_type
                job.directed_scan_type = directed_scan_type
                job.job_adapter = 'gaussian'
                job.job_memory_gb = 14
                job.rotor_index = 0
                job.torsions = [[0, 1, 2, 3]]
                job.level = Level(method='wb97xd', basis='def2tzvp')
                job.args = dict()
                job.local_path_to_output_file = output_file.name
                job.execution_type = 'incore'
                job.job_name = 'scan_job'

                with patch('arc.scheduler.job_factory') as factory:
                    result = Scheduler.end_job(sched, job=job, label='TS0', job_name='scan_job')

                factory.assert_not_called()
                self.assertFalse(result)
                self.assertEqual(sched.running_jobs['TS0'], list())
                self.assertIs(rotor['success'], False)
                self.assertEqual(rotor['invalidation_reason'], REACTIVE_RING_INVALIDATION_REASON)


class TestSchedulerReactionChargeBalance(unittest.TestCase):
    """
    Contains the charge balance check of reactions whose species the Scheduler attaches.
    """

    def build_scheduler(self, rxn, species_list):
        """
        Build a testing Scheduler for a single reaction.

        Args:
            rxn (ARCReaction): The reaction.
            species_list (list): The species list.

        Returns:
            Scheduler: The constructed (testing) scheduler.
        """
        project_directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        return Scheduler(project='charge_balance',
                         ess_settings={'gaussian': ['server1']},
                         species_list=species_list,
                         rxn_list=[rxn],
                         opt_level=Level(repr='b3lyp/6-31g(d,p)'),
                         sp_level=Level(repr='b3lyp/6-311+g(d,p)'),
                         freq_level=Level(repr='b3lyp/6-31g(d,p)'),
                         project_directory=project_directory,
                         job_types=initialize_job_types(),
                         testing=True)

    def test_a_reaction_built_from_labels_is_charge_checked_when_its_species_are_attached(self):
        """Test that a label-only reaction whose attached species are not charge balanced is refused"""
        species = [ARCSpecies(label='HO-', smiles='[OH-]', charge=-1), ARCSpecies(label='CH3OH', smiles='CO'),
                   ARCSpecies(label='H2O', smiles='O'), ARCSpecies(label='CH3O', smiles='C[O]')]
        rxn = ARCReaction(label='HO- + CH3OH <=> H2O + CH3O')
        with self.assertRaises(ReactionError):
            self.build_scheduler(rxn, species)

    def test_a_balanced_reaction_built_from_labels_gets_its_charge(self):
        """Test that a label-only reaction whose attached species are charge balanced gets their net charge"""
        species = [ARCSpecies(label='HO-', smiles='[OH-]', charge=-1), ARCSpecies(label='CH3OH', smiles='CO'),
                   ARCSpecies(label='H2O', smiles='O'), ARCSpecies(label='CH3O-', smiles='C[O-]', charge=-1)]
        rxn = ARCReaction(label='HO- + CH3OH <=> H2O + CH3O-')
        self.build_scheduler(rxn, species)
        self.assertEqual(rxn.charge, -1)

    def test_duplicate_reaction_labels_are_rejected_by_the_scheduler(self):
        """Test that a Scheduler built directly with two reactions sharing a label refuses them"""
        species = [ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='OH', smiles='[OH]'),
                   ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='H2O', smiles='O')]
        rxn_1 = ARCReaction(label='CH4 + OH <=> CH3 + H2O')
        rxn_2 = ARCReaction(label='CH4 + OH <=> CH3 + H2O')
        project_directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        with self.assertRaises(InputError) as cm:
            Scheduler(project='duplicate_labels',
                      ess_settings={'gaussian': ['server1']},
                      species_list=species,
                      rxn_list=[rxn_1, rxn_2],
                      opt_level=Level(repr='b3lyp/6-31g(d,p)'),
                      sp_level=Level(repr='b3lyp/6-311+g(d,p)'),
                      freq_level=Level(repr='b3lyp/6-31g(d,p)'),
                      project_directory=project_directory,
                      job_types=initialize_job_types(),
                      testing=True)
        self.assertIn("'CH4 + OH <=> CH3 + H2O'", str(cm.exception))


    def test_a_label_only_reaction_keeps_its_family_and_ts_adapters(self):
        """Test that a label-only reaction gets its RMG family and family TS adapters once its species attach"""
        species = [ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='OH', smiles='[OH]'),
                   ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='H2O', smiles='O')]
        rxn = ARCReaction(label='CH4 + OH <=> CH3 + H2O')
        self.build_scheduler(rxn, species)
        self.assertEqual(rxn.family, 'H_Abstraction')
        for adapter in ['heuristics', 'autotst']:
            self.assertIsNotNone(Scheduler.ts_adapter_admission(adapter, rxn), adapter)

class TestChangeConformerReopt(unittest.TestCase):
    """
    Contains unit tests for the 'change conformer' rotor troubleshooting path and its conformer_verified record.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.ess_settings = {'gaussian': ['server1'], 'molpro': ['server2', 'server1'], 'qchem': ['server1']}
        cls.job_types = initialize_job_types({'conf_opt': False, 'opt': True, 'freq': True, 'sp': True,
                                              'rotors': True, 'fine': False})
        cls.scan = [3, 1, 2, 4]
        cls.nc3h7_log = os.path.join(ARC_TESTING_PATH, 'opt', 'nC3H7.out')

    def make_sched(self, name: str, composite: bool = False) -> Scheduler:
        """
        Make a Scheduler with an H2O2 species whose old geometry has a 120 deg H-O-O-H dihedral.

        Args:
            name (str): A unique name for the project and the species label.
            composite (bool): Whether to run a composite method.

        Returns:
            Scheduler: The Scheduler object.
        """
        spc = ARCSpecies(label=name, smiles='OO')
        spc.rotors_dict = {0: {'pivots': [1, 2], 'scan': self.scan, 'torsion': [2, 0, 1, 3], 'scan_path': '',
                               'invalidation_reason': '', 'success': None, 'symmetry': None,
                               'times_dihedral_set': 0, 'trsh_methods': [], 'trsh_counter': 0}}
        spc.final_xyz = spc.get_xyz()
        spc.set_dihedral(scan=self.scan, index=1, deg_abs=120.0, count=False)
        spc.final_xyz = spc.initial_xyz
        project_directory = os.path.join(ARC_PATH, 'Projects', f'arc_project_{name}')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        kwargs = {'composite_method': Level(repr='cbs-qb3')} if composite else dict()
        sched = Scheduler(project=name, ess_settings=self.ess_settings, species_list=[spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          scan_level=Level(repr=default_levels_of_theory['scan']),
                          project_directory=project_directory, testing=True, job_types=self.job_types, **kwargs)
        sched.trsh_ess_jobs = True
        sched.trsh_rotors = True
        return sched

    def new_xyz(self, sched: Scheduler, label: str, dihedral: float = 60.0) -> dict:
        """
        Get a copy of the species geometry with the scanned dihedral set to ``dihedral``.

        Args:
            sched (Scheduler): The Scheduler object.
            label (str): The species label.
            dihedral (float): The dihedral angle in degrees.

        Returns:
            dict: The new xyz.
        """
        spc = sched.species_dict[label]
        old_initial = spc.initial_xyz
        spc.set_dihedral(scan=self.scan, index=1, deg_abs=dihedral, count=False)
        xyz, spc.initial_xyz = spc.initial_xyz, old_initial
        return xyz

    def scan_job(self, label: str, energy_gap: float | None = 3.2) -> tuple:
        """
        Make the scan job stand-in and the 'change conformer' methods that troubleshoot_scan_job receives.

        Args:
            label (str): The species label.
            energy_gap (float, optional): The energy gap reported by the scan in kJ/mol.

        Returns:
            tuple: The job stand-in and the energy gap.
        """
        return SimpleNamespace(species_label=label, rotor_index=0, torsions=[[2, 0, 1, 3]], job_name='scan_a300'), \
            energy_gap

    def dihedral(self, xyz: dict) -> float:
        """
        The H-O-O-H dihedral of an H2O2 geometry.

        Args:
            xyz (dict): The geometry.

        Returns:
            float: The dihedral in degrees.
        """
        return calculate_dihedral_angle(coords=xyz, torsion=self.scan, index=1)

    def change_conformer(self, sched: Scheduler, label: str, new_xyz: dict) -> None:
        """
        Run troubleshoot_scan_job with a 'change conformer' action, with job execution stubbed.

        Args:
            sched (Scheduler): The Scheduler object.
            label (str): The species label.
            new_xyz (dict): The lower conformer.
        """
        job, gap = self.scan_job(label)
        with patch('arc.job.adapter.JobAdapter.execute'), \
                patch('arc.scheduler.Scheduler.check_max_simultaneous_jobs_limit'):
            success, actions = sched.troubleshoot_scan_job(job=job, methods={'change conformer': new_xyz,
                                                                             'energy_gap': gap})
        self.assertTrue(success)
        self.assertIs(sched.species_dict[label].rotors_dict[0]['trsh_methods'][-1], actions)

    def test_change_conformer_reopt_starts_from_new_xyz(self):
        """The opt job built after a 'change conformer' gets the new conformer as its input geometry."""
        sched = self.make_sched('cc_opt')
        new_xyz = self.new_xyz(sched, 'cc_opt')
        self.change_conformer(sched, 'cc_opt', new_xyz)
        opt_job = list(sched.job_dict['cc_opt']['opt'].values())[-1]
        self.assertAlmostEqual(abs(self.dihedral(opt_job.xyz)), 60.0, delta=0.5)
        self.assertAlmostEqual(abs(self.dihedral(sched.species_dict['cc_opt'].initial_xyz)), 60.0, delta=0.5)

    def test_change_conformer_composite_starts_from_new_xyz(self):
        """The composite job built after a 'change conformer' gets the new conformer as its input geometry."""
        sched = self.make_sched('cc_comp', composite=True)
        new_xyz = self.new_xyz(sched, 'cc_comp')
        self.change_conformer(sched, 'cc_comp', new_xyz)
        comp_job = list(sched.job_dict['cc_comp']['composite'].values())[-1]
        self.assertAlmostEqual(abs(self.dihedral(comp_job.xyz)), 60.0, delta=0.5)

    def test_change_conformer_restart_keeps_new_xyz(self):
        """A restart after the 'change conformer' re-builds the opt job from the new conformer."""
        sched = self.make_sched('cc_restart')
        new_xyz = self.new_xyz(sched, 'cc_restart')
        self.change_conformer(sched, 'cc_restart', new_xyz)
        restored = ARCSpecies(species_dict=sched.species_dict['cc_restart'].as_dict())
        self.assertAlmostEqual(abs(self.dihedral(restored.initial_xyz)), 60.0, delta=0.5)
        record = restored.rotors_dict[0]['trsh_methods'][-1]
        self.assertIsNone(record['verified'])
        self.assertAlmostEqual(abs(self.dihedral(record['pre_xyz'])), 120.0, delta=0.5)
        sched_2 = self.make_sched('cc_restart_2')
        sched_2.species_dict['cc_restart_2'].initial_xyz = restored.initial_xyz
        sched_2.species_dict['cc_restart_2'].final_xyz = restored.final_xyz
        with patch('arc.job.adapter.JobAdapter.execute'), \
                patch('arc.scheduler.Scheduler.check_max_simultaneous_jobs_limit'):
            sched_2.run_opt_job('cc_restart_2')
        opt_job = list(sched_2.job_dict['cc_restart_2']['opt'].values())[-1]
        self.assertAlmostEqual(abs(self.dihedral(opt_job.xyz)), 60.0, delta=0.5)

    def test_change_conformer_records_pending_and_resets_flag(self):
        """A 'change conformer' records the old geometry and the gap, and resets conformer_verified to pending."""
        sched = self.make_sched('cc_record')
        sched.output['cc_record']['conformer_verified'] = False
        sched.output['cc_record']['conformer_verified_cause'] = 'old cause'
        self.change_conformer(sched, 'cc_record', self.new_xyz(sched, 'cc_record'))
        record = sched.species_dict['cc_record'].rotors_dict[0]['trsh_methods'][-1]
        self.assertEqual(record['torsion'], self.scan)
        self.assertEqual(record['energy_gap'], 3.2)
        self.assertIsNone(record['pre_e_elect'])
        self.assertIsNone(record['pre_level'])
        self.assertIsNone(record['verified'])
        self.assertIsNone(sched.output['cc_record']['conformer_verified'])
        self.assertEqual(sched.output['cc_record']['conformer_verified_cause'], '')

    def test_check_change_conformer_reopt_same_energy_and_torsion_is_error(self):
        """A re-opt that returns the old energy and the old torsion flags the species as not verified."""
        sched = self.make_sched('cc_same')
        label = 'cc_same'
        log = self.nc3h7_log
        xyz = parser.parse_geometry(log_file_path=log)
        torsion = [1, 2, 3, 4]
        sched.species_dict[label].final_xyz = xyz
        sched.species_dict[label].rotors_dict[0]['trsh_methods'] = [
            {'change conformer': xyz, 'pre_xyz': xyz, 'pre_e_elect': parser.parse_e_elect(log_file_path=log),
             'torsion': torsion, 'energy_gap': 3.2, 'verified': None}]
        with self.assertLogs('arc', level='WARNING') as logs:
            sched.check_change_conformer_reopt(label=label, job=SimpleNamespace(job_type='opt', local_path_to_xyz=None,
                                                                                local_path_to_output_file=log))
        self.assertFalse(sched.output[label]['conformer_verified'])
        cause = sched.output[label]['conformer_verified_cause']
        self.assertIn('returned to the previous geometry', cause)
        self.assertIn('3.20 kJ/mol lower', cause)
        self.assertIn(cause, sched.output[label]['warnings'])
        self.assertTrue(any('returned to the previous geometry' in line for line in logs.output))
        self.assertFalse(sched.species_dict[label].rotors_dict[0]['trsh_methods'][-1]['verified'])
        self.assertIsNone(sched.output[label]['convergence'])

    def test_check_change_conformer_reopt_new_minimum_passes(self):
        """A re-opt that reaches a different torsion and a different energy marks the record verified."""
        sched = self.make_sched('cc_pass')
        label = 'cc_pass'
        log = self.nc3h7_log
        xyz = parser.parse_geometry(log_file_path=log)
        old_spc = ARCSpecies(label='nC3H7_old', smiles='[CH2]CC', xyz=xyz)
        old_spc.determine_rotors()
        torsion = old_spc.rotors_dict[0]['scan']
        old_spc.final_xyz = xyz
        old_spc.set_dihedral(scan=torsion, index=1, deg_increment=60.0, count=False)
        sched.species_dict[label].final_xyz = xyz
        sched.species_dict[label].rotors_dict[0]['trsh_methods'] = [
            {'change conformer': xyz, 'pre_xyz': old_spc.initial_xyz,
             'pre_e_elect': parser.parse_e_elect(log_file_path=log) + 3.0,
             'torsion': torsion, 'energy_gap': 3.2, 'verified': None}]
        sched.check_change_conformer_reopt(label=label, job=SimpleNamespace(job_type='opt', local_path_to_xyz=None,
                                                                            local_path_to_output_file=log))
        self.assertTrue(sched.species_dict[label].rotors_dict[0]['trsh_methods'][-1]['verified'])
        self.assertIsNone(sched.output[label]['conformer_verified'])
        self.assertEqual(sched.output[label]['warnings'], '')

    def test_check_change_conformer_reopt_energy_up_is_not_verified(self):
        """
        A re-opt that keeps the scanned dihedral but ends at a higher energy than before the conformer change
        did not reach the lower conformer, so its record must not be verified and the species must not pass
        the RMG library gate. The change is recorded with the re-opt geometry as its baseline (dihedral change
        0) and a baseline energy 3 kJ/mol below the re-opt energy; ``is_same_conformer`` is spied on to show
        the judge is reached with these inputs.
        """
        sched = self.make_sched('cc_up')
        label = 'cc_up'
        log = self.nc3h7_log
        xyz = parser.parse_geometry(log_file_path=log)
        old_spc = ARCSpecies(label='nC3H7_old', smiles='[CH2]CC', xyz=xyz)
        old_spc.determine_rotors()
        torsion = old_spc.rotors_dict[0]['scan']
        sched.species_dict[label].final_xyz = xyz
        sched.species_dict[label].rotors_dict[0]['trsh_methods'] = [
            {'change conformer': xyz, 'pre_xyz': xyz,
             'pre_e_elect': parser.parse_e_elect(log_file_path=log) - 3.0,
             'torsion': torsion, 'energy_gap': 3.2, 'verified': None}]
        with patch('arc.scheduler.is_same_conformer', wraps=is_same_conformer) as spy:
            sched.check_change_conformer_reopt(label=label, job=SimpleNamespace(
                job_type='opt', local_path_to_xyz=None, local_path_to_output_file=log))
        spy.assert_called_once()
        self.assertLess(spy.call_args.kwargs['d_dihedral'], 2.0)
        self.assertAlmostEqual(spy.call_args.kwargs['d_energy'], 3.0, places=3)
        sched.record_pending_conformer_changes()
        self.assertIsNot(sched.species_dict[label].rotors_dict[0]['trsh_methods'][-1]['verified'], True)
        self.assertTrue(not_verified_causes(label, sched.output))

    def check_reopt_without_dihedral(self, label: str, energy_shift: float) -> tuple[Scheduler, dict]:
        """
        Judge a re-opt whose scanned dihedral cannot be compared (no geometry recorded before the change).

        Args:
            label (str): The species label.
            energy_shift (float): The re-opt energy minus the energy recorded before the change, in kJ/mol.

        Returns:
            tuple[Scheduler, dict]: The scheduler and the judged 'change conformer' record.
        """
        sched = self.make_sched(label)
        method = {'change conformer': None, 'pre_xyz': None,
                  'pre_e_elect': parser.parse_e_elect(log_file_path=self.nc3h7_log) - energy_shift,
                  'torsion': self.scan, 'energy_gap': 3.2, 'verified': None}
        sched.species_dict[label].rotors_dict[0]['trsh_methods'] = [method]
        sched.check_change_conformer_reopt(label=label, job=SimpleNamespace(
            job_type='opt', local_path_to_xyz=None, local_path_to_output_file=self.nc3h7_log))
        sched.record_pending_conformer_changes()
        return sched, method

    def test_check_change_conformer_reopt_energy_up_without_dihedral_is_not_verified(self):
        """A re-opt that ends at a higher energy is not verified also when its dihedral cannot be compared."""
        label = 'cc_up_nodih'
        sched, method = self.check_reopt_without_dihedral(label=label, energy_shift=3.0)
        output = sched.output
        self.assertIs(method['verified'], False)
        self.assertIs(output[label]['conformer_verified'], False)
        self.assertIn('3.00 kJ/mol above', output[label]['conformer_verified_cause'])
        self.assertEqual(not_verified_causes(label, output), [output[label]['conformer_verified_cause']])

    def test_check_change_conformer_reopt_energy_down_without_dihedral_is_verified(self):
        """A re-opt that ends at a lower energy is verified when its dihedral cannot be compared."""
        label = 'cc_down_nodih'
        sched, method = self.check_reopt_without_dihedral(label=label, energy_shift=-3.0)
        self.assertIs(method['verified'], True)
        self.assertIsNone(sched.output[label]['conformer_verified'])

    def test_check_change_conformer_reopt_symmetry_equivalent_copy_is_not_verified(self):
        """A re-opt that lands on the old minimum with its methyl rotated by 120 deg did not reach the lower conformer."""
        sched = self.make_sched('cc_methyl')
        label = 'cc_methyl'
        log = self.nc3h7_log
        xyz = parser.parse_geometry(log_file_path=log)
        old_spc = ARCSpecies(label='nC3H7_old', smiles='[CH2]CC', xyz=xyz)
        old_spc.determine_rotors()
        methyl_torsion = old_spc.rotors_dict[1]['scan']  # C-C-C-H, referenced to a methyl H
        self.assertEqual(len(old_spc.rotors_dict[1]['top']), 4)
        old_spc.final_xyz = xyz
        old_spc.set_dihedral(scan=methyl_torsion, index=1, deg_increment=120.0, count=False)
        sched.species_dict[label].final_xyz = xyz
        record = {'change conformer': xyz, 'pre_xyz': old_spc.initial_xyz,
                  'pre_e_elect': parser.parse_e_elect(log_file_path=log), 'torsion': methyl_torsion,
                  'energy_gap': 3.2, 'verified': None}
        sched.species_dict[label].rotors_dict[0]['trsh_methods'] = [record]
        sched.check_change_conformer_reopt(label=label, job=SimpleNamespace(job_type='opt', local_path_to_xyz=None,
                                                                            local_path_to_output_file=log))
        self.assertIs(record['verified'], False)
        self.assertIs(sched.output[label]['conformer_verified'], False)
        self.assertIn('dihedral change 120.0 deg', sched.output[label]['conformer_verified_cause'])

    def test_check_change_conformer_reopt_ignores_legacy_records(self):
        """A 'change conformer' record written before the check existed (no 'verified' key) is not re-judged."""
        sched = self.make_sched('cc_legacy')
        xyz = sched.species_dict['cc_legacy'].final_xyz
        sched.species_dict['cc_legacy'].rotors_dict[0]['trsh_methods'] = [{'change conformer': xyz}]
        sched.check_change_conformer_reopt(label='cc_legacy', job=SimpleNamespace(
            job_type='opt', local_path_to_xyz=None, local_path_to_output_file=self.nc3h7_log))
        self.assertNotIn('verified', sched.species_dict['cc_legacy'].rotors_dict[0]['trsh_methods'][0])
        self.assertIsNone(sched.output['cc_legacy']['conformer_verified'])

    def test_change_conformer_at_bound_records_not_verified(self):
        """At max_rotor_trsh, an unresolved lower conformer flags the species as not verified."""
        sched = self.make_sched('cc_bound')
        sched.species_dict['cc_bound'].rotors_dict[0]['trsh_counter'] = 4
        job, gap = self.scan_job('cc_bound')
        with self.assertLogs('arc', level='WARNING') as logs:
            success, actions = sched.troubleshoot_scan_job(
                job=job, methods={'change conformer': self.new_xyz(sched, 'cc_bound'), 'energy_gap': gap})
        self.assertFalse(success)
        self.assertFalse(sched.output['cc_bound']['conformer_verified'])
        cause = sched.output['cc_bound']['conformer_verified_cause']
        self.assertIn('still finds a conformer 3.20 kJ/mol lower', cause)
        self.assertIn(cause, sched.output['cc_bound']['warnings'])
        self.assertTrue(any('still finds a conformer' in line for line in logs.output))
        self.assertNotIn('opt', sched.job_dict['cc_bound'])

    def test_change_conformer_same_conformer_twice_records_cause(self):
        """Proposing the same lower conformer twice flags the species with that cause."""
        sched = self.make_sched('cc_twice')
        new_xyz = self.new_xyz(sched, 'cc_twice')
        sched.species_dict['cc_twice'].rotors_dict[0]['trsh_methods'] = [{'change conformer': new_xyz}]
        job, gap = self.scan_job('cc_twice')
        success, _ = sched.troubleshoot_scan_job(job=job, methods={'change conformer': new_xyz, 'energy_gap': gap})
        self.assertFalse(success)
        self.assertFalse(sched.output['cc_twice']['conformer_verified'])
        self.assertIn('same lower conformer', sched.output['cc_twice']['conformer_verified_cause'])

    def test_flag_conformer_not_verified_keeps_every_cause(self):
        """Two unresolved causes are both kept in conformer_verified_cause, and a repeated one only once."""
        sched = self.make_sched('cc_causes')
        sched.flag_conformer_not_verified(label='cc_causes', cause='cause A.')
        sched.flag_conformer_not_verified(label='cc_causes', cause='cause B.')
        sched.flag_conformer_not_verified(label='cc_causes', cause='cause A.')
        self.assertEqual(sched.output['cc_causes']['conformer_verified_cause'], 'cause A. cause B.')

    def test_check_change_conformer_reopt_without_comparison_stays_pending(self):
        """A re-opt that can be compared neither by energy nor by geometry is left pending, not verified."""
        sched = self.make_sched('cc_nocomp')
        method = {'change conformer': None, 'pre_xyz': None, 'pre_e_elect': None, 'torsion': self.scan,
                  'energy_gap': 3.2, 'verified': None}
        sched.species_dict['cc_nocomp'].rotors_dict[0]['trsh_methods'] = [method]
        sched.check_change_conformer_reopt(label='cc_nocomp', job=SimpleNamespace(
            job_type='opt', local_path_to_xyz='/no/such/file.xyz', local_path_to_output_file=self.nc3h7_log))
        self.assertIsNone(method['verified'])
        self.assertAlmostEqual(parse_job_e_elect(SimpleNamespace(local_path_to_xyz=self.nc3h7_log,
                                                                 local_path_to_output_file='/no/such/file.log')),
                               parser.parse_e_elect(log_file_path=self.nc3h7_log))
        self.assertIsNone(parse_job_e_elect(SimpleNamespace(local_path_to_xyz=None, local_path_to_output_file=None)))
        self.assertIsNone(sched.output['cc_nocomp']['conformer_verified'])
        method['pre_e_elect'] = parser.parse_e_elect(log_file_path=self.nc3h7_log)
        sched.check_change_conformer_reopt(label='cc_nocomp', job=SimpleNamespace(
            job_type='opt', local_path_to_xyz=None, local_path_to_output_file=self.nc3h7_log))
        self.assertIsNone(method['verified'])
        self.assertIsNone(sched.output['cc_nocomp']['conformer_verified'])

    def test_is_same_conformer(self):
        """A known, unchanged energy is the same conformer at any dihedral; without it, the dihedral decides."""
        self.assertFalse(is_same_conformer(d_dihedral=None, d_energy=None))
        self.assertTrue(is_same_conformer(d_dihedral=0.5, d_energy=0.01))
        self.assertTrue(is_same_conformer(d_dihedral=0.5, d_energy=None))
        self.assertTrue(is_same_conformer(d_dihedral=None, d_energy=0.0))
        self.assertTrue(is_same_conformer(d_dihedral=120.0, d_energy=0.0))
        self.assertTrue(is_same_conformer(d_dihedral=120.0, d_energy=0.0, energy_gap=3.2))
        self.assertFalse(is_same_conformer(d_dihedral=0.5, d_energy=3.2))
        self.assertFalse(is_same_conformer(d_dihedral=0.0, d_energy=0.1))
        self.assertFalse(is_same_conformer(d_dihedral=2.0, d_energy=None))
        # A gap below the energy tolerance (a nearly free rotor) cannot be told apart by energy; the dihedral decides.
        self.assertFalse(is_same_conformer(d_dihedral=120.0, d_energy=0.0, energy_gap=0.05))
        self.assertTrue(is_same_conformer(d_dihedral=0.5, d_energy=0.0, energy_gap=0.05))
        self.assertFalse(is_same_conformer(d_dihedral=2.0, d_energy=0.0, energy_gap=0.05))
        self.assertFalse(is_same_conformer(d_dihedral=None, d_energy=0.0, energy_gap=0.05))

    def test_unresolved_lower_conformer_without_trsh_records_not_verified(self):
        """A lower conformer that troubleshooting is not allowed to resolve flags the species."""
        sched = self.make_sched('cc_notrsh')
        sched.trsh_rotors = False
        sched.flag_unresolved_lower_conformer(label='cc_notrsh', rotor_index=0, energy_gap=3.2,
                                              reason='rotor troubleshooting is disabled')
        self.assertFalse(sched.output['cc_notrsh']['conformer_verified'])
        self.assertIn('rotor troubleshooting is disabled', sched.output['cc_notrsh']['conformer_verified_cause'])

    def make_coccoo_scan(self, name: str, invalidation_reason: str = '') -> tuple:
        """
        Make a Scheduler with a COCCOO species and a done scan job whose log finds a conformer 4.60 kJ/mol lower.

        Args:
            name (str): A unique name for the project and the species label.
            invalidation_reason (str): The rotor's invalidation reason before the check.

        Returns:
            tuple: The Scheduler and the scan job.
        """
        log = os.path.join(ARC_TESTING_PATH, 'rotor_scans', 'COCCOO.out')
        xyz = parser.parse_1d_scan_coords(log)[0]
        spc = ARCSpecies(label=name, smiles='COCCOO', xyz=xyz)
        spc.final_xyz = xyz
        spc.determine_rotors()
        spc.rotors_dict[3]['invalidation_reason'] = invalidation_reason
        project_directory = os.path.join(ARC_PATH, 'Projects', f'arc_project_{name}')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project=name, ess_settings=self.ess_settings, species_list=[spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          scan_level=Level(repr=default_levels_of_theory['scan']),
                          project_directory=project_directory, testing=True, job_types=self.job_types)
        sched.trsh_ess_jobs, sched.trsh_rotors = True, False
        job = job_factory(job_adapter='gaussian', project=name, ess_settings=self.ess_settings,
                          species=[sched.species_dict[name]], xyz=xyz, job_type='scan', torsions=[[2, 1, 4, 3]],
                          rotor_index=3, level=Level(repr=default_levels_of_theory['scan']),
                          project_directory=project_directory, job_num=301)
        job.job_status[1]['status'] = 'done'
        job.local_path_to_output_file = log
        return sched, job

    def test_check_scan_job_flags_lower_conformer_without_trsh(self):
        """check_scan_job flags a lower conformer that rotor troubleshooting is disabled for."""
        sched, job = self.make_coccoo_scan('cc_scan_notrsh')
        with patch('arc.plotter.plot_1d_rotor_scan'):
            sched.check_scan_job(label='cc_scan_notrsh', job=job)
        self.assertFalse(sched.output['cc_scan_notrsh']['conformer_verified'])
        self.assertIn('4.60 kJ/mol lower', sched.output['cc_scan_notrsh']['conformer_verified_cause'])
        self.assertIn('troubleshooting was not applied', sched.output['cc_scan_notrsh']['conformer_verified_cause'])

    def test_check_scan_job_does_not_flag_a_pivts_rotor(self):
        """A TS reaction-zone rotor ('pivTS') is never troubleshot, so it is not flagged as a lower conformer."""
        sched, job = self.make_coccoo_scan('cc_scan_pivts', invalidation_reason='pivTS')
        sched.trsh_rotors = True
        with patch('arc.plotter.plot_1d_rotor_scan'):
            sched.check_scan_job(label='cc_scan_pivts', job=job)
        self.assertIsNone(sched.output['cc_scan_pivts']['conformer_verified'])

    def test_set_conformer_verified(self):
        """conformer_verified becomes True only once every rotor was scanned and checked, else stays None."""
        sched = self.make_sched('cc_set')
        self.assertIsNone(sched.output['cc_set']['conformer_verified'])
        self.assertEqual(sched.output['cc_set']['conformer_verified_cause'], '')
        rotor = sched.species_dict['cc_set'].rotors_dict[0]
        sched.set_conformer_verified('cc_set')
        self.assertIsNone(sched.output['cc_set']['conformer_verified'])
        rotor['success'], rotor['scan_path'] = True, self.nc3h7_log
        rotor['trsh_methods'] = [{'change conformer': None, 'verified': None}]
        sched.set_conformer_verified('cc_set')
        self.assertIsNone(sched.output['cc_set']['conformer_verified'])
        rotor['trsh_methods'][0]['verified'] = True
        rotor['trsh_methods'].insert(0, {'change conformer': None, 'verified': False})
        sched.set_conformer_verified('cc_set')
        self.assertTrue(sched.output['cc_set']['conformer_verified'])
        sched.output['cc_set']['conformer_verified'] = False
        sched.set_conformer_verified('cc_set')
        self.assertFalse(sched.output['cc_set']['conformer_verified'])
        sched.species_dict['cc_set'].rotors_dict = dict()
        sched.output['cc_set']['conformer_verified'] = None
        sched.set_conformer_verified('cc_set')
        self.assertIsNone(sched.output['cc_set']['conformer_verified'])
        sched.output['cc_set'].pop('conformer_verified')
        sched.set_conformer_verified('cc_set')
        self.assertIsNone(sched.output['cc_set']['conformer_verified'])
        sched.species_dict['cc_set'].rotors_dict = {0: dict(rotor), 1: dict(rotor, success=False, scan_path='')}
        sched.set_conformer_verified('cc_set')
        self.assertIsNone(sched.output['cc_set']['conformer_verified'])
        sched.species_dict['cc_set'].rotors_dict = {0: dict(rotor), 1: dict(rotor, success=False)}
        sched.set_conformer_verified('cc_set')
        self.assertIsNone(sched.output['cc_set']['conformer_verified'])

    def test_record_pending_conformer_changes(self):
        """The end of a run counts the unjudged conformer changes of each species, also after a restart."""
        sched = self.make_sched('cc_pending')
        label = 'cc_pending'
        sched.record_pending_conformer_changes()
        self.assertEqual(sched.output[label]['conformer_change_pending'], 0)
        self.assertEqual(sched.output[label]['conformer_change_judged'], 0)
        rotor = sched.species_dict[label].rotors_dict[0]
        rotor['trsh_methods'] = [{'change conformer': None, 'verified': True},
                                 {'change conformer': None, 'verified': False},
                                 {'change conformer': None}]
        sched.record_pending_conformer_changes()
        self.assertEqual(sched.output[label]['conformer_change_pending'], 0)
        self.assertEqual(sched.output[label]['conformer_change_judged'], 2)
        rotor['trsh_methods'].append({'change conformer': None, 'verified': None})
        sched.record_pending_conformer_changes()
        self.assertEqual(sched.output[label]['conformer_change_pending'], 1)
        self.assertEqual(sched.output[label]['conformer_change_judged'], 2)
        rotor['trsh_methods'][-1]['verified'] = True
        sched.record_pending_conformer_changes()
        self.assertEqual(sched.output[label]['conformer_change_pending'], 0)
        self.assertEqual(sched.output[label]['conformer_change_judged'], 3)

        rotor['trsh_methods'][-1]['verified'] = None
        restarted_spc = ARCSpecies(species_dict=sched.species_dict[label].as_dict())
        output = {label: {key: val for key, val in sched.output[label].items() if key != 'conformer_change_pending'}}
        restarted = Scheduler(project=label, ess_settings=self.ess_settings, species_list=[restarted_spc],
                              opt_level=Level(repr=default_levels_of_theory['opt']),
                              freq_level=Level(repr=default_levels_of_theory['freq']),
                              sp_level=Level(repr=default_levels_of_theory['sp']),
                              scan_level=Level(repr=default_levels_of_theory['scan']),
                              project_directory=sched.project_directory, testing=True, job_types=self.job_types,
                              restart_dict={'output': output})
        self.assertNotIn('conformer_change_pending', restarted.output[label])
        restarted.record_pending_conformer_changes()
        self.assertEqual(restarted.output[label]['conformer_change_pending'], 1)

    def test_judged_conformer_change_with_no_rescan_verdict_is_not_verified(self):
        """
        A species whose conformer change was judged but whose rotor rescans never set ``conformer_verified`` does not
        pass the RMG library gate. A species that never changed its conformer still passes with ``None``.
        """
        sched = self.make_sched('cc_judged')
        label = 'cc_judged'
        rotor = sched.species_dict[label].rotors_dict[0]
        sched.record_pending_conformer_changes()
        self.assertIsNone(sched.output[label]['conformer_verified'])
        self.assertEqual(not_verified_causes(label, sched.output), list())
        rotor['trsh_methods'] = [{'change conformer': None}]
        sched.record_pending_conformer_changes()
        self.assertEqual(not_verified_causes(label, sched.output), list())
        rotor['trsh_methods'].append({'change conformer': None, 'verified': True})
        rotor['success'], rotor['scan_path'] = False, ''
        sched.set_conformer_verified(label)
        sched.record_pending_conformer_changes()
        self.assertIsNone(sched.output[label]['conformer_verified'])
        self.assertEqual(not_verified_causes(label, sched.output),
                         [f'{label}: conformer change not confirmed by a rotor rescan'])
        rotor['success'], rotor['scan_path'] = True, self.nc3h7_log
        sched.set_conformer_verified(label)
        self.assertTrue(sched.output[label]['conformer_verified'])
        self.assertEqual(not_verified_causes(label, sched.output), list())

    def test_change_conformer_record_is_in_the_restart_saved_by_the_reopt(self):
        """The pending record is attached before the re-opt is spawned, so the restart file it saves holds it."""
        sched = self.make_sched('cc_ckpt')
        saved = list()
        sched.save_restart_dict = lambda: saved.append(
            [dict(method) for method in sched.species_dict['cc_ckpt'].rotors_dict[0]['trsh_methods']])
        self.change_conformer(sched, 'cc_ckpt', self.new_xyz(sched, 'cc_ckpt'))
        self.assertTrue(saved)
        self.assertTrue(all(len(checkpoint) == 1 and checkpoint[0]['verified'] is None
                            and 'pre_xyz' in checkpoint[0] for checkpoint in saved))

    def test_restart_between_the_opt_and_the_scan_keeps_the_energy_before_the_change(self):
        """A restart after the opt finished but before the scan found a lower conformer keeps the opt's energy and level."""
        label = 'cc_restart_e'
        sched = self.make_sched(label)
        level = Level(repr=default_levels_of_theory['opt'])
        opt_job = self.geo_job('opt', 'opt_a1', level)
        sched.job_dict[label]['opt'] = {'opt_a1': opt_job}
        old_xyz = sched.species_dict[label].final_xyz
        e_elect = parser.parse_e_elect(log_file_path=self.nc3h7_log)
        with patch('arc.scheduler.parser.parse_geometry', return_value=old_xyz), \
                patch('arc.scheduler.plotter'), \
                patch('arc.species.species.ARCSpecies.check_xyz_isomorphism', return_value=True):
            self.assertTrue(sched.parse_opt_geo(label=label, job=opt_job))
        self.assertEqual(sched.get_latest_geo_job_e_elect(label), (e_elect, job_level_signature(opt_job)))

        # restart.yml holds the output dict and the species, and only the running jobs (here, the rotor scans).
        restart_path = os.path.join(sched.project_directory, 'restart_before_scan.yml')
        save_yaml_file(path=restart_path,
                       content={'output': sched.output,
                                'species': [spc.as_dict() for spc in sched.species_dict.values()],
                                'running_jobs': dict()})
        restart_dict = read_yaml_file(path=restart_path)
        restarted = Scheduler(project=label, ess_settings=self.ess_settings,
                              species_list=[ARCSpecies(species_dict=restart_dict['species'][0])],
                              opt_level=Level(repr=default_levels_of_theory['opt']),
                              freq_level=Level(repr=default_levels_of_theory['freq']),
                              sp_level=Level(repr=default_levels_of_theory['sp']),
                              scan_level=Level(repr=default_levels_of_theory['scan']),
                              project_directory=sched.project_directory, testing=True, job_types=self.job_types,
                              restart_dict={'output': restart_dict['output']})
        restarted.trsh_ess_jobs, restarted.trsh_rotors = True, True
        self.assertFalse(restarted.job_dict[label].get('opt'))

        new_xyz = self.new_xyz(restarted, label)
        self.change_conformer(restarted, label, new_xyz)
        record = restarted.species_dict[label].rotors_dict[0]['trsh_methods'][-1]
        self.assertIsNotNone(record['pre_e_elect'], 'the restart lost the energy before the conformer change')
        self.assertAlmostEqual(record['pre_e_elect'], e_elect)
        self.assertEqual(record['pre_level'], job_level_signature(opt_job))

        # The re-opt moved the dihedral 60 deg but kept the energy: judged by energy, the change was not reached.
        restarted.species_dict[label].final_xyz = new_xyz
        restarted.check_change_conformer_reopt(label=label, job=self.geo_job('opt', 'opt_a2', level))
        self.assertIs(record['verified'], False)
        self.assertIs(restarted.output[label]['conformer_verified'], False)

    def test_stored_geo_energy_follows_the_geometry(self):
        """The stored energy is dropped when the geometry moves to one without a comparable energy."""
        label = 'cc_geo_e'
        sched = self.make_sched(label, composite=True)
        composite_job = self.geo_job('composite', 'composite_a1', Level(repr='cbs-qb3'))
        sched.record_geo_job_e_elect(label=label, job=composite_job)
        self.assertAlmostEqual(sched.get_latest_geo_job_e_elect(label)[0],
                               parser.parse_e_elect(log_file_path=self.nc3h7_log))
        # An opt troubleshooting the composite run sets final_xyz; the composite energy no longer belongs to it.
        sched.record_geo_job_e_elect(label=label, job=self.geo_job('opt', 'opt_a2',
                                                                   Level(repr=default_levels_of_theory['opt'])))
        self.assertEqual(sched.get_latest_geo_job_e_elect(label), (None, None))
        # A conformer change moves final_xyz to the scan's geometry, which no job computed yet.
        sched.record_geo_job_e_elect(label=label, job=composite_job)
        self.change_conformer(sched, label, self.new_xyz(sched, label))
        record = sched.species_dict[label].rotors_dict[0]['trsh_methods'][-1]
        self.assertIsNotNone(record['pre_e_elect'])
        self.assertEqual(sched.get_latest_geo_job_e_elect(label), (None, None))

    def test_check_change_conformer_reopt_waits_for_the_baseline_job_type(self):
        """An opt that troubleshoots a composite run (or the reverse) is not compared with the other level's energy."""
        log = self.nc3h7_log
        xyz = parser.parse_geometry(log_file_path=log)
        for composite, other_type, own_type in ((True, 'opt', 'composite'), (False, 'composite', 'opt')):
            label = f'cc_type_{own_type}'
            sched = self.make_sched(label, composite=composite)
            sched.species_dict[label].final_xyz = xyz
            method = {'change conformer': xyz, 'pre_xyz': xyz, 'torsion': [1, 2, 3, 4], 'energy_gap': 3.2,
                      'pre_e_elect': parser.parse_e_elect(log_file_path=log) + 500.0, 'verified': None}
            sched.species_dict[label].rotors_dict[0]['trsh_methods'] = [method]
            sched.check_change_conformer_reopt(label=label, job=SimpleNamespace(
                job_type=other_type, local_path_to_xyz=None, local_path_to_output_file=log))
            self.assertIsNone(method['verified'])
            method['pre_e_elect'] -= 500.0
            sched.check_change_conformer_reopt(label=label, job=SimpleNamespace(
                job_type=own_type, local_path_to_xyz=None, local_path_to_output_file=log))
            self.assertFalse(method['verified'])
            self.assertFalse(sched.output[label]['conformer_verified'])

    def test_job_level_signature(self):
        """Two jobs at the same level differ in signature only when an ESS troubleshooting argument differs."""
        level = Level(repr=default_levels_of_theory['opt'])
        plain = SimpleNamespace(level=level, args={'keyword': {}, 'block': {}, 'trsh': {}})
        self.assertEqual(job_level_signature(plain), job_level_signature(SimpleNamespace(level=level, args=dict())))
        self.assertNotEqual(job_level_signature(plain),
                            job_level_signature(SimpleNamespace(level=level, args={'trsh': {'trsh': ['vdz']}})))
        self.assertNotEqual(job_level_signature(plain),
                            job_level_signature(SimpleNamespace(level=Level(repr='b3lyp/6-31g(d)'), args=dict())))

    def test_check_change_conformer_reopt_ignores_the_energy_of_a_different_level(self):
        """A baseline from a troubleshot level (e.g. Molpro 'vdz') is not compared by energy; the dihedral decides."""
        sched = self.make_sched('cc_level')
        label = 'cc_level'
        log = self.nc3h7_log
        xyz = parser.parse_geometry(log_file_path=log)
        level = Level(repr=default_levels_of_theory['opt'])
        sched.species_dict[label].final_xyz = xyz
        method = {'change conformer': xyz, 'pre_xyz': xyz, 'torsion': [1, 2, 3, 4], 'energy_gap': 3.2,
                  'pre_e_elect': parser.parse_e_elect(log_file_path=log) + 500.0, 'verified': None,
                  'pre_level': job_level_signature(SimpleNamespace(level=level, args={'trsh': {'trsh': ['vdz']}}))}
        sched.species_dict[label].rotors_dict[0]['trsh_methods'] = [method]
        sched.check_change_conformer_reopt(label=label, job=SimpleNamespace(
            job_type='opt', level=level, args=dict(), local_path_to_xyz=None, local_path_to_output_file=log))
        self.assertFalse(method['verified'])
        self.assertFalse(sched.output[label]['conformer_verified'])
        self.assertIn('energy not compared', sched.output[label]['conformer_verified_cause'])

    def pending_change(self, sched: Scheduler, label: str) -> tuple[dict, dict, dict]:
        """
        Record a pending 'change conformer' from the 120 deg geometry to a 60 deg one.

        Args:
            sched (Scheduler): The Scheduler object.
            label (str): The species label.

        Returns:
            tuple[dict, dict, dict]: The record, the old geometry, and the changed geometry.
        """
        old_xyz = sched.species_dict[label].final_xyz
        changed_xyz = self.new_xyz(sched, label, dihedral=60.0)
        record = {'change conformer': changed_xyz, 'pre_xyz': old_xyz, 'pre_e_elect': None, 'torsion': self.scan,
                  'energy_gap': 3.2, 'verified': None}
        sched.species_dict[label].rotors_dict[0]['trsh_methods'] = [record]
        return record, old_xyz, changed_xyz

    def geo_job(self, job_type: str, job_name: str, level: Level) -> SimpleNamespace:
        """
        A converged geometry job stand-in.

        Args:
            job_type (str): The job type.
            job_name (str): The job name.
            level (Level): The job level.

        Returns:
            SimpleNamespace: The job stand-in.
        """
        return SimpleNamespace(job_status=['done', {'status': 'done'}], job_type=job_type, job_name=job_name,
                               job_adapter='gaussian', is_ts=False, fine=False, level=level, args=dict(),
                               local_path_to_output_file=self.nc3h7_log, local_path_to_xyz=None)

    def test_rejected_composite_result_leaves_the_conformer_change_pending(self):
        """A composite result rejected for an imaginary frequency is not judged; the accepted retry is."""
        label = 'cc_comp_rej'
        sched = self.make_sched(label, composite=True)
        record, old_xyz, changed_xyz = self.pending_change(sched, label)
        job = self.geo_job('composite', 'composite_a1', Level(repr='cbs-qb3'))
        with patch('arc.scheduler.parser.parse_geometry', side_effect=[changed_xyz, old_xyz]), \
                patch('arc.scheduler.parser.parse_frequencies', side_effect=[[-150.0, 400.0], [150.0, 400.0]]), \
                patch('arc.scheduler.parse_job_e_elect', return_value=None), \
                patch('arc.scheduler.plotter'), \
                patch('arc.scheduler.Scheduler.troubleshoot_negative_freq'), \
                patch('arc.scheduler.Scheduler.troubleshoot_ess'), \
                patch('arc.species.species.ARCSpecies.check_xyz_isomorphism', return_value=True):
            self.assertFalse(sched.parse_composite_geo(label=label, job=job))
            self.assertIsNone(record['verified'])
            self.assertTrue(sched.parse_composite_geo(label=label, job=job))
        self.assertIs(record['verified'], False)
        self.assertIs(sched.output[label]['conformer_verified'], False)

    def test_rejected_freq_of_an_opt_leaves_the_conformer_change_pending(self):
        """With a separate freq job, the change is judged when the freq passes, also across a restart."""
        label = 'cc_opt_rej'
        sched = self.make_sched(label)
        record, old_xyz, changed_xyz = self.pending_change(sched, label)
        level = Level(repr=default_levels_of_theory['opt'])
        freq_job = SimpleNamespace(job_type='freq', job_name='freq_a2', level=level,
                                   job_status=['done', {'status': 'done'}], local_path_to_output_file=self.nc3h7_log)
        sched.job_dict[label]['opt'] = dict()
        with patch('arc.scheduler.parser.parse_geometry', side_effect=[changed_xyz, old_xyz]), \
                patch('arc.scheduler.parse_job_e_elect', return_value=None), \
                patch('arc.scheduler.plotter'), \
                patch('arc.scheduler.Scheduler.post_opt_geo_work'), \
                patch('arc.species.species.ARCSpecies.check_xyz_isomorphism', return_value=True):
            for job_name, xyz in (('opt_a1', changed_xyz), ('opt_a3', old_xyz)):
                sched.job_dict[label]['opt'][job_name] = self.geo_job('opt', job_name, level)
                self.assertTrue(sched.parse_opt_geo(label=label, job=sched.job_dict[label]['opt'][job_name]))
                self.assertIsNone(record['verified'])
                if job_name == 'opt_a1':
                    self.assertFalse(sched.post_freq_actions(label=label, job=freq_job, vibfreqs=[-150.0, 400.0])[0])
                    self.assertIsNone(record['verified'])
            sched.job_dict[label]['opt'] = dict()  # a restart restores no completed job
            sched.species_dict[label] = ARCSpecies(species_dict=sched.species_dict[label].as_dict())
            record = sched.species_dict[label].rotors_dict[0]['trsh_methods'][-1]
            self.assertTrue(sched.post_freq_actions(label=label, job=freq_job, vibfreqs=[150.0, 400.0])[0])
        self.assertIs(record['verified'], False)
        self.assertIs(sched.output[label]['conformer_verified'], False)

    def test_change_conformer_records_a_chronological_index(self):
        """Each 'change conformer' record gets the next index over the records of every rotor of the species."""
        sched = self.make_sched('cc_index')
        rotors = sched.species_dict['cc_index'].rotors_dict
        rotors[1] = dict(rotors[0], trsh_methods=[{'change conformer': None}])
        self.change_conformer(sched, 'cc_index', self.new_xyz(sched, 'cc_index', dihedral=60.0))
        self.assertEqual(rotors[0]['trsh_methods'][-1]['change_index'], 2)
        sched.running_jobs['cc_index'] = list()
        self.change_conformer(sched, 'cc_index', self.new_xyz(sched, 'cc_index', dihedral=180.0))
        self.assertEqual(rotors[0]['trsh_methods'][-1]['change_index'], 3)

    def test_check_change_conformer_reopt_judges_the_latest_change(self):
        """The latest change is judged against its own baseline, not an older pending change of a later rotor."""
        sched = self.make_sched('cc_latest')
        label = 'cc_latest'
        log = self.nc3h7_log
        xyz = parser.parse_geometry(log_file_path=log)
        sched.species_dict[label].final_xyz = xyz
        rotors = sched.species_dict[label].rotors_dict
        latest = {'change conformer': xyz, 'pre_xyz': xyz, 'pre_e_elect': parser.parse_e_elect(log_file_path=log),
                  'torsion': [1, 2, 3, 4], 'energy_gap': 3.2, 'verified': None, 'change_index': 2}
        stale = {'change conformer': None, 'pre_xyz': None, 'pre_e_elect': None, 'torsion': [1, 2, 3, 4],
                 'energy_gap': 1.0, 'verified': None, 'change_index': 1}
        rotors[0]['trsh_methods'] = [latest]
        rotors[1] = dict(rotors[0], trsh_methods=[stale])
        sched.check_change_conformer_reopt(label=label, job=SimpleNamespace(job_type='opt', local_path_to_xyz=None,
                                                                            local_path_to_output_file=log))
        self.assertFalse(latest['verified'])
        self.assertFalse(stale['verified'])
        self.assertFalse(sched.output[label]['conformer_verified'])
        self.assertIn('3.20 kJ/mol lower', sched.output[label]['conformer_verified_cause'])

    def test_check_change_conformer_reopt_collinear_dihedral_stays_pending(self):
        """A dihedral over collinear atoms (NaN) cannot be compared, so an unchanged energy keeps the record pending."""
        sched = self.make_sched('cc_nan')
        label = 'cc_nan'
        xyz = str_to_xyz("""C 0.0 0.0 0.0
                            C 0.0 0.0 1.2
                            N 0.0 0.0 2.4
                            H 1.0 0.0 3.0""")
        sched.species_dict[label].final_xyz = xyz
        method = {'change conformer': xyz, 'pre_xyz': xyz, 'torsion': [1, 2, 3, 4], 'energy_gap': 3.2,
                  'pre_e_elect': parser.parse_e_elect(log_file_path=self.nc3h7_log), 'verified': None}
        sched.species_dict[label].rotors_dict[0]['trsh_methods'] = [method]
        sched.check_change_conformer_reopt(label=label, job=SimpleNamespace(
            job_type='opt', local_path_to_xyz=None, local_path_to_output_file=self.nc3h7_log))
        self.assertIsNone(method['verified'])
        self.assertIsNone(sched.output[label]['conformer_verified'])

    def test_check_scan_job_change_conformer_rescans_the_rotor(self):
        """After a successful conformer change the triggering rotor keeps no scan path, so it is scanned again."""
        sched, job = self.make_coccoo_scan('cc_rescan')
        sched.trsh_rotors = True
        with patch('arc.plotter.plot_1d_rotor_scan'), patch('arc.job.adapter.JobAdapter.execute'), \
                patch('arc.scheduler.Scheduler.check_max_simultaneous_jobs_limit'):
            sched.check_scan_job(label='cc_rescan', job=job)
        rotor = sched.species_dict['cc_rescan'].rotors_dict[3]
        self.assertEqual(sum('change conformer' in method for method in rotor['trsh_methods']), 1)
        self.assertEqual(rotor['scan_path'], '')
        self.assertEqual(rotor['invalidation_reason'], '')
        self.assertIsNone(rotor['success'])

    def test_change_conformer_same_conformer_twice_keeps_the_results(self):
        """Proposing an already tried conformer deletes no job and does not block thermo; it only flags the species."""
        sched = self.make_sched('cc_keep')
        new_xyz = self.new_xyz(sched, 'cc_keep')
        sched.species_dict['cc_keep'].rotors_dict[0]['trsh_methods'] = [{'change conformer': new_xyz}]
        sched.output['cc_keep']['job_types']['opt'] = True
        sched.output['cc_keep']['paths']['geo'] = self.nc3h7_log
        job, gap = self.scan_job('cc_keep')
        success, actions = sched.troubleshoot_scan_job(job=job, methods={'change conformer': new_xyz,
                                                                         'energy_gap': gap})
        self.assertFalse(success)
        self.assertEqual(actions, dict())
        self.assertFalse(sched.output['cc_keep']['conformer_verified'])
        self.assertIsNot(sched.output['cc_keep']['convergence'], False)
        self.assertEqual(sched.output['cc_keep']['errors'], '')
        self.assertTrue(sched.output['cc_keep']['job_types']['opt'])
        self.assertEqual(sched.output['cc_keep']['paths']['geo'], self.nc3h7_log)

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_switch_ts_resets_the_conformer_verdict(self, mock_run_opt):
        """A replacement TS guess is a new geometry, so the previous guess's not-verified verdict is dropped."""
        ts_xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")
        ts_spc = ARCSpecies(label='TS_cc', is_ts=True, xyz=ts_xyz, multiplicity=1, charge=0, compute_thermo=False)
        ts_spc.ts_guesses = [TSGuess(index=i, method='heuristics', success=True, energy=100.0 + 10 * i, xyz=ts_xyz,
                                     execution_time='0:00:01') for i in range(2)]
        for i, tsg in enumerate(ts_spc.ts_guesses):
            tsg.opt_xyz, tsg.imaginary_freqs = ts_xyz, [-500.0 + 100 * i]
        ts_spc.chosen_ts, ts_spc.chosen_ts_list, ts_spc.ts_guesses_exhausted = 0, [0], False
        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_cc_switch_ts')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project='cc_switch_ts', ess_settings=self.ess_settings, species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory, testing=True, job_types=self.job_types)
        sched.job_dict['TS_cc'] = {'opt': {}, 'freq': {}, 'sp': {}}
        sched.running_jobs['TS_cc'] = []
        sched.flag_conformer_not_verified(label='TS_cc', cause='old guess cause.')
        sched.switch_ts('TS_cc')
        self.assertTrue(mock_run_opt.called)
        self.assertIsNone(sched.output['TS_cc']['conformer_verified'])
        self.assertEqual(sched.output['TS_cc']['conformer_verified_cause'], '')
        self.assertIn('old guess cause.', sched.output['TS_cc']['warnings'])


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
