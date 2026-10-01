#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.scheduler module
"""

import datetime
import logging
import math
from functools import partial
import os
import shutil
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch


import arc.job.ssh as ssh
import arc.parser.parser as parser
from arc.checks.ts import check_ts
from arc.common import (ARC_PATH,
                        ARC_TESTING_PATH,
                        almost_equal_coords_lists,
                        get_test_project_directory,
                        initialize_job_types,
                        read_yaml_file,
                        save_yaml_file,
                        )
from arc.exceptions import DependencyError
from arc.job.adapters.common import (adopted_reference_is_unrestricted, default_incore_adapters,
                                      derived_instability_breaks_spin_symmetry, is_restricted,
                                      REFERENCE_AGNOSTIC_METHOD_TYPES, REFERENCE_CHANGE_AVAILABLE_KEY,
                                      ts_adapters_by_rmg_family, ts_adapters_for_unknown_unimolecular)
from arc.job.factory import job_factory
from arc.level import Level
from arc.plotter import save_conformers_file
from arc.scheduler import (COLLAPSED_REFERENCE_MESSAGE, INVALID_ANALYTIC_FREQ_MESSAGE, MAX_S_SQUARED_DEVIATION,
                           MIXED_SCF_REFERENCE_MESSAGE, SPIN_CONTAMINATION_MESSAGE, STABILITY_ANALYSIS_ADAPTERS,
                           SYMMETRY_BREAKING_ADAPTERS, UNREACHABLE_REFERENCE_MESSAGE,
                           Scheduler, SchedulerError, species_has_freq, species_has_geo, species_has_sp,
                           species_has_sp_and_freq, species_is_ready_for_e0, tsg_method_matches_adapter)
from arc.imports import settings
from arc.settings.settings import input_filenames
from arc.reaction import ARCReaction
from arc.species.converter import str_to_xyz
from arc.species.species import ARCSpecies, TSGuess


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
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.ess_settings = {'gaussian': ['server1'], 'molpro': ['server2', 'server1'], 'qchem': ['server1']}
        cls.project_directory = get_test_project_directory('arc_project_for_testing_delete_after_usage3')
        xyz1 = str_to_xyz("""C      -0.57422867   -0.01669771    0.01229213
N       0.82084044    0.08279104   -0.37769346
H      -1.05737005   -0.84067772   -0.52007494
H      -1.10211468    0.90879867   -0.23383011
H      -0.66133128   -0.19490562    1.08785111
H       0.88047852    0.26966160   -1.37780789
H       1.27889520   -0.81548721   -0.22940984""")
        cls.spc1 = ARCSpecies(label='methylamine', smiles='CN', xyz=xyz1)
        cls.spc2 = ARCSpecies(label='C2H6', smiles='CC')
        xyz3 = """C       1.11424367   -0.01231165   -0.11493630
C      -0.07257945   -0.17830906   -0.16010022
O      -1.38500471   -0.36381519   -0.20928090
H       2.16904830    0.12689206   -0.07152274
H      -1.82570782    0.42754384   -0.56130718"""
        cls.spc3 = ARCSpecies(label='CtripCO', smiles='C#CO', xyz=xyz3)
        cls.job1 = job_factory(job_adapter='gaussian', project='project_test', ess_settings=cls.ess_settings,
                               species=[cls.spc1], xyz=xyz1, job_type='conf_opt',
                               conformer=0, level=Level(repr={'method': 'b97-d3', 'basis': '6-311+g(d,p)'}),
                               project_directory=cls.project_directory, job_num=101)
        cls.job2 = job_factory(job_adapter='gaussian', project='project_test', ess_settings=cls.ess_settings,
                               species=[cls.spc1], xyz=xyz1, job_type='conf_opt',
                               conformer=1, level=Level(repr={'method': 'b97-d3', 'basis': '6-311+g(d,p)'}),
                               project_directory=cls.project_directory, job_num=102)
        cls.job3 = job_factory(job_adapter='qchem', project='project_test', ess_settings=cls.ess_settings,
                               species=[cls.spc2], job_type='freq',
                               level=Level(repr={'method': 'wb97x-d3', 'basis': '6-311+g(d,p)'}),
                               project_directory=cls.project_directory, job_num=103)
        cls.job4 = job_factory(job_adapter='gaussian', project='project_test_4', ess_settings=cls.ess_settings,
                               species=[cls.spc1], xyz=xyz1, job_type='scan', torsions=[[3, 1, 2, 6]], rotor_index=0,
                               level=Level(repr={'method': 'b3lyp', 'basis': 'cbsb7'}),
                               project_directory=cls.project_directory, job_num=104)
        cls.job_types1 = {'conf_opt': True,
                          'conf_sp': False,
                          'opt': True,
                          'fine': False,
                          'freq': True,
                          'sp': True,
                          'rotors': False,
                          'orbitals': False,
                          'lennard_jones': False,
                          }
        cls.job_types2 = {'conf_opt': True,
                          'conf_sp': False,
                          'opt': True,
                          'fine': False,
                          'freq': True,
                          'sp': True,
                          'rotors': True,
                          }
        cls.sched1 = Scheduler(project='project_test_1', ess_settings=cls.ess_settings,
                               species_list=[cls.spc1, cls.spc2, cls.spc3],
                               composite_method=None,
                               conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                               opt_level=Level(repr=default_levels_of_theory['opt']),
                               freq_level=Level(repr=default_levels_of_theory['freq']),
                               sp_level=Level(repr=default_levels_of_theory['sp']),
                               scan_level=Level(repr=default_levels_of_theory['scan']),
                               ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                               project_directory=cls.project_directory,
                               testing=True,
                               job_types=cls.job_types1,
                               orbitals_level=default_levels_of_theory['orbitals'],
                               adaptive_levels=None,
                               )
        cls.sched2 = Scheduler(project='project_test_2', ess_settings=cls.ess_settings,
                               species_list=[cls.spc1, cls.spc2, cls.spc3],
                               composite_method=None,
                               conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                               opt_level=Level(repr=default_levels_of_theory['opt']),
                               freq_level=Level(repr=default_levels_of_theory['freq']),
                               sp_level=Level(repr=default_levels_of_theory['sp']),
                               scan_level=Level(repr=default_levels_of_theory['scan']),
                               ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                               project_directory=cls.project_directory,
                               testing=True,
                               job_types=cls.job_types1,
                               orbitals_level=default_levels_of_theory['orbitals'],
                               adaptive_levels=None,
                               )
        cls.sched3 = Scheduler(project='project_test_4', ess_settings=cls.ess_settings,
                               species_list=[cls.spc1],
                               composite_method=Level(repr='CBS-QB3'),
                               conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                               opt_level=Level(repr=default_levels_of_theory['freq_for_composite']),
                               freq_level=Level(repr=default_levels_of_theory['freq_for_composite']),
                               scan_level=Level(repr=default_levels_of_theory['scan_for_composite']),
                               ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                               project_directory=cls.project_directory,
                               testing=True,
                               job_types=cls.job_types2,
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
                              'levels': dict(),
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
                              'restart': '', 'warnings': ''}
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

    def _restore_sp_state_after_test(self, label):
        """Restore the sp-related fields of a shared species and its output record after the test."""
        species = self.sched1.species_dict[label]
        output = self.sched1.output[label]
        for attribute in ('t1', 'e_elect', 'active', 'sp_level'):
            owner = self.sched1 if attribute == 'sp_level' else species
            self.addCleanup(setattr, owner, attribute, getattr(owner, attribute))
        self.addCleanup(output.__setitem__, 'info', output['info'])
        for container, key in ((output['paths'], 'sp'), (output['paths'], 'sp_sol'), (output['paths'], 'sp_no_sol'),
                               (output['job_types'], 'sp')):
            if key in container:
                self.addCleanup(container.__setitem__, key, container[key])
            else:
                self.addCleanup(container.pop, key, None)

    def test_post_sp_actions_t1_follows_the_recorded_sp_path(self):
        """Test that t1 is re-derived for the recorded sp path and not left stale or taken from a scheme job"""
        label = 'C2H6'
        species = self._reset_reference_records(label)
        self._restore_sp_state_after_test(label)
        sp_path = os.path.join(ARC_TESTING_PATH, 'sp', 'TS_x118_sp_CCSD(T).out')
        ccsd_level = Level(method='ccsd(t)', basis='cc-pVTZ', software='molpro')
        other_path = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'Species', 'NH2_freq.out')
        self.sched1.sp_level = ccsd_level
        self.sched1.output[label]['job_types']['sp'] = False
        with patch.object(self.sched1, 'check_spin_contamination'):
            self.sched1.post_sp_actions(label=label, sp_path=sp_path, level=ccsd_level)
            self.assertAlmostEqual(species.t1, 0.04469461, places=7)
            self.sched1.sp_level = Level(method='wb97xd', basis='def2-TZVP')
            self.sched1.output[label]['job_types']['sp'] = False
            self.sched1.post_sp_actions(label=label, sp_path=other_path, level=self.sched1.sp_level)
        self.assertIsNone(species.t1)

    def test_post_sp_actions_solvation_scheme_jobs_do_not_overwrite_the_recorded_sp(self):
        """Test that t1, e_elect and the T1 report come only from the job whose sp path is kept"""
        label = 'C2H6'
        species = self._reset_reference_records(label)
        self._restore_sp_state_after_test(label)
        ccsd_path = os.path.join(ARC_TESTING_PATH, 'sp', 'TS_x118_sp_CCSD(T).out')
        dft_path = os.path.join(ARC_TESTING_PATH, 'sp', 'formaldehyde_sp_terachem_output.out')
        scheme_level = Level(method='wb97xd', basis='def2-TZVP')
        self.sched1.sp_level = Level(method='ccsd(t)', basis='cc-pVTZ', software='molpro',
                                     solvation_scheme_level=scheme_level)
        self.sched1.output[label]['job_types']['sp'] = False
        self.sched1.output[label]['info'] = ''
        with patch.object(self.sched1, 'run_sp_job'), \
                patch.object(self.sched1, 'check_spin_contamination'), \
                patch.object(self.sched1, 'check_rxn_e0_by_spc'):
            self.sched1.post_sp_actions(label=label, sp_path=ccsd_path, level=self.sched1.sp_level)
            expected_e_elect = species.e_elect
            sol_level = Level(method='wb97xd', basis='def2-TZVP', solvation_method='smd', solvent='water')
            self.sched1.post_sp_actions(label=label, sp_path=dft_path, level=sol_level)
            self.sched1.post_sp_actions(label=label, sp_path=dft_path, level=scheme_level)
        self.assertAlmostEqual(species.t1, 0.04469461, places=7)
        self.assertAlmostEqual(species.e_elect, expected_e_elect, places=3)
        self.assertAlmostEqual(species.e_elect, parser.parse_e_elect(ccsd_path), places=3)
        self.assertEqual(self.sched1.output[label]['paths']['sp'], ccsd_path)
        self.assertEqual(self.sched1.output[label]['paths']['sp_sol'], dft_path)
        self.assertEqual(self.sched1.output[label]['paths']['sp_no_sol'], dft_path)
        self.assertEqual(self.sched1.output[label]['info'].count('T1 ='), 1)

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

    def test_switch_ts_forgets_the_e0_and_the_arkane_record_of_the_abandoned_guess(self):
        """Test that the next TS guess does not inherit the E0, its correction switches or the Arkane rotor modes"""
        species = self.sched1.species_dict['C2H6']
        for attribute in ('e0', 'e0_atom_corrections_applied', 'e0_bond_corrections_applied', 'e0_aec_yml_sha256',
                          'arkane_rotor_modes'):
            self.addCleanup(setattr, species, attribute, getattr(species, attribute))
        self.addCleanup(setattr, species, 'ts_guesses_exhausted', species.ts_guesses_exhausted)
        species.ts_guesses_exhausted = True
        species.e0, species.e0_atom_corrections_applied, species.e0_bond_corrections_applied = 50.0, True, False
        species.e0_aec_yml_sha256 = 'a' * 64
        species.arkane_rotor_modes = ['HinderedRotor']
        with patch.object(self.sched1, 'determine_most_likely_ts_conformer'), \
                patch.object(self.sched1, 'delete_all_species_jobs'):
            self.sched1.switch_ts(label='C2H6')
        self.assertIsNone(species.e0_aec_yml_sha256)
        self.assertIsNone(species.e0)
        self.assertIsNone(species.e0_atom_corrections_applied)
        self.assertIsNone(species.e0_bond_corrections_applied)
        self.assertIsNone(species.arkane_rotor_modes)

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
        project_directory = get_test_project_directory('arc_project_for_testing_delete_after_usage6')
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
                          project_directory=get_test_project_directory('arc_project_for_testing_delete_after_usage6'),
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
                            project_directory=get_test_project_directory('arc_project_for_testing_delete_after_usage6'),
                            )
        job_1.local_path_to_output_file = os.path.join(ARC_TESTING_PATH, 'freq', 'TS_nC3H7-iC3H7.out')
        check_ts(reaction=rxn, verbose=True, job=job_1, checks=['NMD'])
        self.assertEqual(rxn.ts_species.ts_checks, {'E0': None, 'e_elect': True, 'IRC': None, 'freq': True, 'NMD': True, 'warnings': ''})

    def test_save_e_elect(self):
        """Test the save_e_elect() method."""
        project_directory = os.path.join(ARC_PATH, 'Projects', 'save_e_elect')
        e_elect_summary_path = os.path.join(project_directory, 'output', 'e_elect_summary.yml')
        if os.path.isfile(os.path.join(project_directory, 'output', 'e_elect_summary.yml')):
            os.remove(os.path.join(project_directory, 'output', 'e_elect_summary.yml'))
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
        shutil.rmtree(project_directory, ignore_errors=True)

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

    def test_species_is_ready_for_e0(self):
        """Monoatomic species require an SP energy but legitimately have no frequencies."""
        output = {'paths': {'sp': 'sp.out', 'freq': '', 'composite': ''}}
        monoatomic = ARCSpecies(label='O', smiles='[O]')
        molecular = ARCSpecies(label='OH', smiles='[OH]')
        self.assertTrue(species_is_ready_for_e0(output, monoatomic))
        self.assertFalse(species_is_ready_for_e0(output, molecular))
        output['paths']['sp'] = ''
        self.assertFalse(species_is_ready_for_e0(output, monoatomic))

    @patch('arc.scheduler.check_ts')
    def test_monoatomic_participant_reaches_e0_check_and_switches_ts(self, mock_check_ts):
        """A failed E0 check switches guesses even when one reaction participant is monoatomic."""
        scheduler = object.__new__(Scheduler)
        species = {
            'R': ARCSpecies(label='R', smiles='OO'),
            'O': ARCSpecies(label='O', smiles='[O]'),
            'P': ARCSpecies(label='P', smiles='O=O'),
            'TS0': ARCSpecies(label='TS0', is_ts=True),
        }
        species['TS0'].ts_guesses_exhausted = False
        species['TS0'].chosen_ts = 1
        rxn = MagicMock()
        rxn.reactants = ['R', 'O']
        rxn.products = ['P']
        rxn.ts_label = 'TS0'
        rxn.ts_species = species['TS0']
        rxn.label = 'R + O <=> P'
        scheduler.rxn_list = [rxn]
        scheduler.species_dict = species
        scheduler.output = {
            label: {'paths': {'sp': 'sp.out', 'freq': '' if label == 'O' else 'freq.out', 'composite': ''},
                    'convergence': True}
            for label in species
        }
        scheduler.project_directory = '/tmp'
        scheduler.kinetics_adapter = 'arkane'
        scheduler.sp_level = Level('gfn2')
        scheduler.composite_method = None
        scheduler.freq_scale_factor = 1.0
        scheduler.switch_ts = MagicMock()

        def fail_e0(**kwargs):
            kwargs['reaction'].ts_species.ts_checks['E0'] = False

        mock_check_ts.side_effect = fail_e0
        scheduler.check_rxn_e0_by_spc('O')

        mock_check_ts.assert_called_once()
        scheduler.switch_ts.assert_called_once_with('TS0')

    @patch('arc.scheduler.check_ts')
    @patch('arc.scheduler.parser.parse_e_elect', return_value=-75.0)
    def test_atomic_sp_completion_triggers_e0_check_and_switches_ts(self, mock_parse_e_elect, mock_check_ts):
        """Completing the last atomic SP triggers the E0 check and switches a failed TS."""
        scheduler = object.__new__(Scheduler)
        species = {
            'R': ARCSpecies(label='R', smiles='OO'),
            'O': ARCSpecies(label='O', smiles='[O]'),
            'P': ARCSpecies(label='P', smiles='O=O'),
            'TS0': ARCSpecies(label='TS0', is_ts=True),
        }
        species['TS0'].ts_guesses_exhausted = False
        species['TS0'].chosen_ts = 1
        rxn = MagicMock()
        rxn.reactants = ['R', 'O']
        rxn.products = ['P']
        rxn.ts_label = 'TS0'
        rxn.ts_species = species['TS0']
        rxn.label = 'R + O <=> P'
        scheduler.rxn_list = [rxn]
        scheduler.species_dict = species
        scheduler.output = {
            label: {'paths': {'sp': '' if label == 'O' else 'sp.out',
                              'freq': '' if label == 'O' else 'freq.out',
                              'composite': ''},
                    'job_types': {'sp': False},
                    'info': '',
                    'convergence': True}
            for label in species
        }
        scheduler.project_directory = '/tmp'
        scheduler.kinetics_adapter = 'arkane'
        scheduler.sp_level = Level('gfn2')
        scheduler.composite_method = None
        scheduler.freq_scale_factor = 1.0
        scheduler.report_e_elect = False
        scheduler.switch_ts = MagicMock()

        def fail_e0(**kwargs):
            kwargs['reaction'].ts_species.ts_checks['E0'] = False

        mock_check_ts.side_effect = fail_e0
        scheduler.post_sp_actions('O', 'atomic-sp.out')

        mock_parse_e_elect.assert_called_once_with('atomic-sp.out')
        mock_check_ts.assert_called_once()
        scheduler.switch_ts.assert_called_once_with('TS0')

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
        job.job_status[1] = {'status': 'errored', 'keywords': ['SCF'], 'error': 'some error', 'line': 'line'}
        with patch('arc.scheduler.trsh_ess_job', return_value=([], ['trsh_attempt', 'mock'], False,
                                                               Level(repr='wb97xd/def2tzvp'), 'gaussian', 'opt',
                                                               False, '', 14, '', 8, False)) as mock_trsh, \
                patch.object(self.sched1, 'run_job') as mock_run_job, \
                patch.object(self.sched1, 'save_restart_dict'):
            self.sched1.troubleshoot_ess(label=label, job=job,
                                         level_of_theory=Level(repr='wb97xd/def2tzvp'))
        mock_trsh.assert_called_once()
        mock_run_job.assert_called_once()
        self.assertNotIn('ESS troubleshooting attempts exhausted', self.sched1.output[label]['errors'])

    def test_troubleshoot_ess_orca_reduces_cpu_when_memory_is_capped(self):
        """Test that ORCA troubleshooting preserves capped total memory and reduces cpu cores."""
        label = 'methylamine'
        self.sched1.output = dict()
        self.sched1.initialize_output_dict()

        job = MagicMock()
        job.job_name = 'sp_a203'
        job.job_type = 'sp'
        job.job_adapter = 'orca'
        job.level = Level(repr={'method': 'dlpno-ccsd(T)'})
        job.server = 'server1'
        job.fine = True
        job.cpu_cores = 32
        job.job_memory_gb = 250
        job.ess_trsh_methods = list()
        job.torsions = None
        job.dihedrals = None
        job.directed_scan_type = None
        job.rotor_index = None
        job.job_status = ['done', {'status': 'errored',
                                   'keywords': ['MDCI', 'Memory', 'max_total_job_memory'],
                                   'error': 'Orca suggests to increase per cpu core memory to 10218 MB.',
                                   'line': 'Please increase MaxCore'}]

        with patch.object(self.sched1, 'run_job') as mock_run_job, \
                patch.object(self.sched1, 'save_restart_dict'):
            self.sched1.troubleshoot_ess(label=label, job=job, level_of_theory=job.level)

        kwargs = mock_run_job.call_args.kwargs
        self.assertEqual(kwargs['cpu_cores'], 24)
        self.assertEqual(kwargs['memory'], 250)
        self.assertIn('cpu', kwargs['ess_trsh_methods'])

    def test_troubleshoot_ess_orca_increases_total_memory_when_not_capped(self):
        """Test that ORCA troubleshooting increases total memory when the node cap was not hit."""
        label = 'methylamine'
        self.sched1.output = dict()
        self.sched1.initialize_output_dict()

        job = MagicMock()
        job.job_name = 'sp_a204'
        job.job_type = 'sp'
        job.job_adapter = 'orca'
        job.level = Level(repr={'method': 'dlpno-ccsd(T)'})
        job.server = 'server1'
        job.fine = True
        job.cpu_cores = 32
        job.job_memory_gb = 250
        job.ess_trsh_methods = list()
        job.torsions = None
        job.dihedrals = None
        job.directed_scan_type = None
        job.rotor_index = None
        job.job_status = ['done', {'status': 'errored',
                                   'keywords': ['MDCI', 'Memory'],
                                   'error': 'Orca suggests to increase per cpu core memory to 10218 MB.',
                                   'line': 'Please increase MaxCore'}]

        with patch.object(self.sched1, 'run_job') as mock_run_job, \
                patch.object(self.sched1, 'save_restart_dict'):
            self.sched1.troubleshoot_ess(label=label, job=job, level_of_theory=job.level)

        kwargs = mock_run_job.call_args.kwargs
        self.assertEqual(kwargs['cpu_cores'], 24)
        self.assertEqual(kwargs['memory'], 250)
        self.assertIn('memory', kwargs['ess_trsh_methods'])

    def test_troubleshoot_ess_orca_rewrites_input_with_reduced_cores_and_higher_maxcore(self):
        """Test ORCA troubleshooting end-to-end from failure to rewritten input file."""
        label = 'methylamine'
        self.sched1.output = dict()
        self.sched1.initialize_output_dict()

        job = MagicMock()
        job.job_name = 'sp_a205'
        job.job_type = 'sp'
        job.job_adapter = 'orca'
        job.level = Level(repr={'method': 'dlpno-ccsd(T)'})
        job.server = 'server1'
        job.fine = True
        job.cpu_cores = 32
        job.job_memory_gb = 250
        job.ess_trsh_methods = list()
        job.torsions = None
        job.dihedrals = None
        job.directed_scan_type = None
        job.rotor_index = None
        job.job_status = ['done', {'status': 'errored',
                                   'keywords': ['MDCI', 'Memory', 'max_total_job_memory'],
                                   'error': 'Orca suggests to increase per cpu core memory to 10218 MB.',
                                   'line': 'Please increase MaxCore'}]

        with patch.object(self.sched1, 'run_job') as mock_run_job, \
                patch.object(self.sched1, 'save_restart_dict'):
            self.sched1.troubleshoot_ess(label=label, job=job, level_of_theory=job.level)

        kwargs = mock_run_job.call_args.kwargs
        temp_project_dir = os.path.join(ARC_TESTING_PATH, 'test_scheduler_orca_trsh_input')
        try:
            rerun_job = job_factory(job_adapter=kwargs['job_adapter'],
                                    project='project_test_scheduler_orca_trsh_input',
                                    ess_settings=self.ess_settings,
                                    species=[self.spc1],
                                    xyz=self.spc1.get_xyz(),
                                    job_type=kwargs['job_type'],
                                    level=kwargs['level_of_theory'],
                                    project_directory=temp_project_dir,
                                    cpu_cores=kwargs['cpu_cores'],
                                    job_memory_gb=kwargs['memory'],
                                    ess_trsh_methods=kwargs['ess_trsh_methods'],
                                    execution_type='incore',
                                    fine=kwargs['fine'],
                                    server=job.server,
                                    testing=True)
            rerun_job.write_input_file()
            with open(os.path.join(rerun_job.local_path, input_filenames[rerun_job.job_adapter]), 'r') as f:
                content = f.read()
            original_maxcore = math.ceil(rerun_job.job_memory_gb * 1024 / job.cpu_cores)
            self.assertIn('%pal nprocs 24 end', content)
            self.assertIn(f'%maxcore {rerun_job.input_file_memory}', content)
            self.assertGreater(rerun_job.input_file_memory, original_maxcore)
        finally:
            shutil.rmtree(temp_project_dir, ignore_errors=True)

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
        project_directory = get_test_project_directory('arc_project_for_testing_delete_after_usage_tsg_err')
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
        project_directory = get_test_project_directory('arc_project_for_testing_delete_after_usage_trsh_tsg')
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
        project_directory = get_test_project_directory('arc_project_for_testing_delete_after_usage_tsg_single')
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

    @patch('arc.scheduler.Scheduler.run_opt_job')
    def test_switch_ts_cleanup(self, mock_run_opt):
        """Test that switch_ts resets job_types, convergence, clears pending pipes, and cleans up IRC species
        both from the Scheduler's own state and from the caller's species list object."""
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

        project_directory = get_test_project_directory('arc_project_for_testing_delete_after_usage4')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        caller_species_list = [ts_spc]
        sched = Scheduler(project='test_switch_ts', ess_settings=self.ess_settings,
                          species_list=caller_species_list,
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

        self.assertNotIn(irc_label_1, [spc.label for spc in caller_species_list])
        self.assertNotIn(irc_label_2, [spc.label for spc in caller_species_list])

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

        project_directory = get_test_project_directory('arc_project_for_testing_delete_after_usage5')
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

        project_directory2 = get_test_project_directory('arc_project_for_testing_delete_after_usage6')
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

    def make_irc_scheduler(self,
                           ts_label: str,
                           project_directory_name: str,
                           num_guesses: int = 2,
                           chosen_ts_list: list | None = None,
                           ) -> Scheduler:
        """
        A helper for generating a Scheduler instance with a single TS species that has several TS guesses,
        simulating the state right after the first chosen guess completed its opt/freq/sp jobs.

        Args:
            ts_label (str): The TS species label.
            project_directory_name (str): The name of the testing project directory.
            num_guesses (int, optional): The number of TS guesses to generate.
            chosen_ts_list (list, optional): The indices of the TS guesses that were already tried.

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
        project_directory = os.path.join(ARC_PATH, 'Projects', project_directory_name)
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
        self.assertFalse(sched.species_dict[ts_label].ts_checks['IRC'])
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
                           ) -> Scheduler:
        """
        A helper for generating a Scheduler instance with a single TS species that has several TS guesses,
        simulating the state right after the first chosen guess completed its opt/freq/sp jobs.

        Args:
            ts_label (str): The TS species label.
            project_directory_name (str): The name of the testing project directory.
            num_guesses (int, optional): The number of TS guesses to generate.
            chosen_ts_list (list, optional): The indices of the TS guesses that were already tried.

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
        project_directory = os.path.join(ARC_PATH, 'Projects', project_directory_name)
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
        self.assertEqual(mock_check_irc_species_and_rxn.call_args.kwargs['endpoint_labels'],
                         (irc_label_1, irc_label_2))
        self.assertEqual(mock_check_irc_species_and_rxn.call_args.kwargs['irc_log_paths'],
                         ['irc_f.out', 'irc_r.out'])
        self.assertEqual(mock_check_irc_species_and_rxn.call_args.kwargs['endpoint_log_paths'],
                         [f'{irc_label_1}_geo.out', f'{irc_label_2}_geo.out'])
        self.assertEqual(sched.species_dict[ts_label].chosen_ts, 1)
        self.assertIn(1, sched.species_dict[ts_label].chosen_ts_list)
        self.assertNotIn(irc_label_1, sched.species_dict)
        self.assertNotIn(irc_label_2, sched.species_dict)
        self.assertNotIn(irc_label_1, sched.running_jobs)
        self.assertNotIn(irc_label_2, sched.output)
        self.assertIsNone(sched.species_dict[ts_label].irc_label)
        self.assertIsNone(sched.species_dict[ts_label].ts_checks['IRC'])
        mock_run_opt.assert_called_once()

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

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        projects = ['arc_project_for_testing_delete_after_usage3', 'arc_project_for_testing_delete_after_usage6']
        for project in projects:
            shutil.rmtree(get_test_project_directory(project), ignore_errors=True)


class TestPathsTemplateInitialization(unittest.TestCase):
    """``initialize_output_dict`` must seed both ``neb`` and ``gsm``
    slots on TS species so the per-method routing in
    ``run_ts_conformer_jobs`` / ``determine_most_likely_ts_conformer``
    can write into pre-existing keys (and the post-restart reset path
    in ``restart_species`` preserves them).
    """

    def test_ts_species_paths_template_includes_gsm(self):
        # Light test: assert the source of truth at the literal call
        # site; a full Scheduler-instance test is heavy and adds no
        # signal beyond the static template check.
        with open(os.path.join(ARC_PATH, 'arc', 'scheduler.py')) as f:
            sched_src = f.read()
        self.assertIn("self.output[species.label]['paths']['gsm'] = ''", sched_src)


class _StopScheduling(Exception):
    """An exception raised by a mocked Scheduler.end_job() to terminate the scheduling loop."""


class TestSchedulerStaleServerJobIds(unittest.TestCase):
    """
    Contains unit tests for the Scheduler's handling of a server whose queue could not be queried.
    """

    @classmethod
    def setUpClass(cls):
        """A function that is run ONCE before all unit tests in this class"""
        cls.ess_settings = {'gaussian': ['server1']}
        cls.project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_for_testing_delete_after_usage_stale')

    def setUp(self):
        """A function that is run before every unit test in this class"""
        self.spc = ARCSpecies(label='C2H6', smiles='CC')
        self.sched = self.make_scheduler()
        job = self.make_job(job_num=101, job_id='4556708', server='local')
        self.sched.job_dict['C2H6'] = {'opt': {'opt_a101': job}}
        self.sched.running_jobs = {'C2H6': ['opt_a101']}
        self.sched.servers = ['local']
        self.addCleanup(ssh.reset_queue_query_history)
        self.addCleanup(shutil.rmtree, self.project_directory, ignore_errors=True)

    def make_scheduler(self) -> Scheduler:
        """
        Make a Scheduler tracking a single species and no job types other than 'opt'.

        Returns:
            Scheduler: The Scheduler.
        """
        return Scheduler(project='project_test_stale_queue', ess_settings=self.ess_settings,
                         species_list=[self.spc],
                         composite_method=None,
                         conformer_opt_level=Level(repr=default_levels_of_theory['conformer']),
                         opt_level=Level(repr=default_levels_of_theory['opt']),
                         freq_level=Level(repr=default_levels_of_theory['freq']),
                         sp_level=Level(repr=default_levels_of_theory['sp']),
                         scan_level=Level(repr=default_levels_of_theory['scan']),
                         ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                         project_directory=self.project_directory,
                         testing=True,
                         job_types={'conf_opt': False, 'conf_sp': False, 'opt': True, 'fine': False,
                                    'freq': False, 'sp': False, 'rotors': False},
                         )

    def make_job(self, job_num: int, job_id: str, server: str):
        """
        Make an opt job which reports the given job ID and server.

        Args:
            job_num (int): The job number.
            job_id (str): The job ID on the server.
            server (str): The server the job runs on.

        Returns:
            JobAdapter: The job.
        """
        job = job_factory(job_adapter='gaussian', project='project_test_stale_queue', ess_settings=self.ess_settings,
                          species=[self.spc], job_type='opt',
                          level=Level(repr={'method': 'b3lyp', 'basis': '6-31g'}),
                          project_directory=self.project_directory, job_num=job_num)
        job.job_id, job.server = job_id, server
        return job

    def _run_scheduling_cycle(self, queue_query_result):
        """
        Run a single scheduling cycle with the local queue status command returning ``queue_query_result``.

        Args:
            queue_query_result (tuple): The stdout, stderr and exit status of the queue status command.

        Returns:
            MagicMock: The mocked ``Scheduler.end_job``.
        """
        def stop_the_loop(*args, **kwargs):
            self.sched.running_jobs = dict()

        end_job = MagicMock(side_effect=_StopScheduling)
        with patch('arc.job.local.execute_command', return_value=queue_query_result), \
                patch.object(Scheduler, 'end_job', end_job), \
                patch.object(Scheduler, 'run_conformer_jobs'), \
                patch.object(Scheduler, 'spawn_ts_jobs'), \
                patch('arc.scheduler.time.sleep', side_effect=stop_the_loop):
            try:
                self.sched.schedule_jobs()
            except _StopScheduling:
                pass
        return end_job

    def test_get_server_job_ids_of_a_failed_query(self):
        """Test that a queue which could not be queried marks its server as stale"""
        with patch('arc.scheduler.check_running_jobs_ids', return_value=['4556708']):
            self.sched.get_server_job_ids()
        self.assertEqual(self.sched.server_job_ids, ['4556708'])
        self.assertEqual(self.sched.stale_servers, set())
        with patch('arc.scheduler.check_running_jobs_ids', return_value=None):
            self.sched.get_server_job_ids()
        self.assertEqual(self.sched.server_job_ids, list())
        self.assertEqual(self.sched.stale_servers, {'local'})
        with patch('arc.scheduler.check_running_jobs_ids', return_value=['4556708']):
            self.sched.get_server_job_ids()
        self.assertEqual(self.sched.stale_servers, set())

    def test_a_failed_query_does_not_end_jobs(self):
        """Test that a queue which could not be queried does not cause a running job to be ended"""
        end_job = self._run_scheduling_cycle(([], ['qstat: cannot connect to server'], 1))
        end_job.assert_not_called()
        self.assertEqual(self.sched.running_jobs, dict())

    def test_a_failed_query_is_not_offered_to_the_pipe_coordinator(self):
        """Test that a queue which could not be queried does not make a pipe run's job look gone"""
        def stop_the_loop(*args, **kwargs):
            self.sched.running_jobs = dict()
            self.sched.active_pipes.clear()

        self.sched.active_pipes['run_1'] = MagicMock(tasks=list())
        poll_pipes = MagicMock()
        with patch('arc.job.local.execute_command',
                   return_value=([], ['qstat: cannot connect to server'], 1)), \
                patch.object(Scheduler, 'end_job', MagicMock(side_effect=_StopScheduling)), \
                patch.object(Scheduler, 'run_conformer_jobs'), \
                patch.object(Scheduler, 'spawn_ts_jobs'), \
                patch.object(self.sched.pipe_coordinator, 'poll_pipes', poll_pipes), \
                patch('arc.scheduler.time.sleep', side_effect=stop_the_loop):
            try:
                self.sched.schedule_jobs()
            except _StopScheduling:
                pass
        poll_pipes.assert_called_with(server_job_ids=None)

    def test_a_silently_failed_query_does_not_end_jobs(self):
        """Test that a queue status command which failed without a diagnostic does not end a running job"""
        for return_code in (124, -9, 2):
            ssh.reset_queue_query_history()
            self.setUp()
            end_job = self._run_scheduling_cycle(([], [], return_code))
            end_job.assert_not_called()

    def test_an_empty_queue_does_end_jobs(self):
        """Test that a queue which answered that it is empty does cause a running job to be ended"""
        end_job = self._run_scheduling_cycle(([], [], 0))
        end_job.assert_called_once()

    def test_a_stale_server_does_not_hold_up_a_healthy_one(self):
        """Test that a job on a healthy server is ended while another server cannot be queried"""
        remote_job = self.make_job(job_num=102, job_id='4556709', server='server1')
        self.sched.job_dict['C2H6']['opt']['opt_a102'] = remote_job
        self.sched.running_jobs = {'C2H6': ['opt_a101', 'opt_a102']}
        self.sched.servers = ['local', 'server1']
        borrow = MagicMock()
        borrow.return_value.__enter__.return_value.check_running_jobs_ids.return_value = list()
        with patch('arc.scheduler.borrow_ssh_client', borrow):
            end_job = self._run_scheduling_cycle(([], ['qstat: cannot connect to server'], 1))
        self.assertEqual(self.sched.stale_servers, {'local'})
        end_job.assert_called_once()
        self.assertIs(end_job.call_args.kwargs['job'], remote_job)

    def test_a_failed_query_does_not_free_the_max_simultaneous_jobs_limit(self):
        """Test that a queue which could not be queried is not treated as having room for more jobs"""
        with patch('arc.scheduler.servers_dict', {'local': {'max_simultaneous_jobs': 10}}), \
                patch('arc.job.local.execute_command',
                      return_value=([], ['qstat: cannot connect to server'], 1)), \
                patch('arc.scheduler.time.sleep', side_effect=_StopScheduling):
            with self.assertRaises(_StopScheduling):
                self.sched.check_max_simultaneous_jobs_limit(server='local')

    def test_restore_running_jobs_populates_the_servers(self):
        """Test that jobs restored from a restart file make their servers queryable"""
        self.sched.servers, self.sched.server_job_ids = list(), list()
        self.sched.running_jobs, self.sched.job_dict = dict(), dict()
        self.sched.restart_dict = {'running_jobs': {'C2H6': [{'job_name': 'opt_a101',
                                                              'job_type': 'opt',
                                                              'species_labels': ['C2H6']}]}}
        restored_job = MagicMock(job_id='4556708', server='server1')
        with patch('arc.scheduler.job_factory', return_value=restored_job):
            self.sched.restore_running_jobs()
        self.assertEqual(self.sched.servers, ['server1'])
        self.assertEqual(self.sched.server_job_ids, ['4556708'])

    def test_a_new_scheduler_forgets_the_queue_query_history_of_a_previous_run(self):
        """Test that a queue outage recorded by a previous run does not immediately stop a new one"""
        ssh._queue_query_history['local'] = {
            'failing_since': datetime.datetime.now() - ssh.QUEUE_QUERY_TOLERANCE - datetime.timedelta(minutes=1),
            'consecutive_failures': 100, 'ever_answered': True, 'last_warned': None}
        sched = self.make_scheduler()
        sched.servers = ['local']
        with patch('arc.job.local.execute_command', return_value=([], ['qstat: cannot connect to server'], 1)):
            sched.get_server_job_ids()
        self.assertEqual(sched.stale_servers, {'local'})


class StubJob(object):
    """
    A stand-in for a job object, so that Scheduler.run_job() can be driven without executing anything.
    A job of an unavailable adapter raises DependencyError from execute(), as the KinBot and
    AutoTST adapters do when their backend is not installed.
    """

    def __init__(self, job_type: str, job_adapter: str | None, unavailable: bool = False):
        self.job_type = job_type
        self.job_adapter = job_adapter
        self.unavailable = unavailable
        self.job_name = f'{job_type}_{job_adapter}'
        self.job_id = None
        self.server = None
        self.execution_type = 'queue'

    def as_dict(self) -> dict:
        """Return a dictionary representation of the job, used when saving the restart file."""
        return {'job_type': self.job_type, 'job_adapter': self.job_adapter}

    def execute(self) -> None:
        """Execute the job, raising if its external backend is not installed."""
        if self.unavailable:
            raise DependencyError(f'The {self.job_adapter} python executable was not found.')


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
        cls.projects = [f'arc_project_for_testing_delete_after_usage_tsg_{i}' for i in range(1, 8)]

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

        patcher = patch('arc.scheduler.job_factory', new=stub_job_factory)
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
        self.assertEqual(rxn.ts_species.unsuccessful_methods, ['kinbot'])
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
        self.assertEqual(rxn.ts_species.unsuccessful_methods, ['kinbot', 'linear'])
        self.assertTrue(any("all of its eligible adapters ['kinbot', 'linear'] are unavailable" in record.getMessage()
                            for record in captured.records))
        self.assertEqual(sched.running_jobs.get(rxn.ts_label, list()), list())
        self.assertEqual(sched.job_dict[rxn.ts_label].get('tsg', dict()), dict())

    def test_spawn_ts_jobs_skips_a_reaction_with_an_unknown_charge_or_multiplicity(self):
        """Test that no TS search job is spawned while the charge or the multiplicity is unknown."""
        for index, attribute in ((6, 'charge'), (7, 'multiplicity')):
            with self.subTest(attribute=attribute):
                sched, rxn, spawned = self.setup_hocho_scheduler(
                    ts_adapters=['linear'],
                    project=f'arc_project_for_testing_delete_after_usage_tsg_{index}')
                with patch.object(ARCReaction, attribute, new_callable=PropertyMock, return_value=None), \
                        self.assertLogs('arc', level='INFO') as captured:
                    sched.spawn_ts_jobs()
                self.assertEqual(spawned, list())
                self.assertTrue(any('Not spawning TS search jobs' in record.getMessage()
                                    for record in captured.records))


class TestSchedulerTSReporting(unittest.TestCase):
    """
    Contains unit tests for the TS validation reporting of the Scheduler
    (Scheduler.report_omitted_ts_guesses).
    """

    def test_report_omitted_ts_guesses(self):
        """The TS guesses that are omitted from the reported guess list are named along with the reason."""
        ts = ARCSpecies(label='TS0', is_ts=True)
        ts.ts_guesses = list()
        for index, (success, energy) in enumerate([(True, 0.0), (False, None), (True, None), (True, 5.0),
                                                   (False, None)]):
            tsg = TSGuess(index=index, method='autotst')
            tsg.success, tsg.energy = success, energy
            ts.ts_guesses.append(tsg)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.species_dict = {'TS0': ts}
        with self.assertLogs('arc', level='INFO') as cm:
            scheduler.report_omitted_ts_guesses(label='TS0')
        self.assertEqual(len(cm.records), 1)
        message = cm.records[0].getMessage()
        self.assertIn('TS guesses not listed above for TS0: 1, 4 (the guess method or its optimization did not '
                      'succeed); 2 (no energy was obtained).', message)

    def test_report_omitted_ts_guesses_none_omitted(self):
        """Nothing is reported when every TS guess is listed."""
        ts = ARCSpecies(label='TS0', is_ts=True)
        tsg = TSGuess(index=0, method='autotst')
        tsg.success, tsg.energy = True, 0.0
        ts.ts_guesses = [tsg]
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.species_dict = {'TS0': ts}
        with self.assertNoLogs('arc', level='INFO'):
            scheduler.report_omitted_ts_guesses(label='TS0')


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


class TestTroubleshootEssJobTypeGuard(unittest.TestCase):
    """
    Test that only a geometry-determining job type may reject a TS guess when ESS troubleshooting
    is exhausted, and that an exhausted rotor scan invalidates just that rotor.
    """

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        cls.ess_settings = {'gaussian': ['server1'], 'molpro': ['server2', 'server1'], 'qchem': ['server1']}
        cls.job_types = {'conf_opt': False, 'opt': True, 'fine': False, 'freq': True, 'sp': True,
                         'rotors': True, 'irc': False}
        cls.ts_xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")

    def build_sched(self, name: str):
        """Build a minimal Scheduler holding a single TS with two guesses and one rotor."""
        ts_spc = ARCSpecies(label='TS_test', is_ts=True, xyz=self.ts_xyz, multiplicity=1, charge=0,
                            compute_thermo=False)
        ts_spc.ts_guesses = [
            TSGuess(index=0, method='heuristics', success=True, energy=100.0, xyz=self.ts_xyz,
                    execution_time='0:00:01'),
            TSGuess(index=1, method='heuristics', success=True, energy=110.0, xyz=self.ts_xyz,
                    execution_time='0:00:01'),
        ]
        for tsg in ts_spc.ts_guesses:
            tsg.opt_xyz = self.ts_xyz
            tsg.imaginary_freqs = [-500.0]
        ts_spc.chosen_ts = 0
        ts_spc.chosen_ts_list = [0]
        ts_spc.ts_guesses_exhausted = False
        ts_spc.rotors_dict = {0: {'pivots': [1, 2], 'top': [2, 3], 'scan': [3, 1, 2, 4], 'torsion': [2, 0, 1, 3],
                                  'number_of_running_jobs': 0, 'success': None, 'invalidation_reason': '',
                                  'times_dihedral_set': 0, 'trsh_counter': 0, 'trsh_methods': list(),
                                  'scan_path': '', 'directed_scan_type': '', 'directed_scan': dict(),
                                  'dimensions': 1, 'original_dihedrals': list(), 'cont_indices': list()}}
        project_directory = os.path.join(ARC_PATH, 'Projects', f'arc_test_trsh_guard_{name}')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project=f'test_trsh_guard_{name}', ess_settings=self.ess_settings,
                          species_list=[ts_spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types=self.job_types,
                          )
        sched.trsh_ess_jobs, sched.trsh_rotors = True, True
        return sched

    def make_failed_job(self, sched, job_type: str, job_num: int, rotor_index=None):
        """Build a job that has failed at the ESS level."""
        kwargs = dict()
        if rotor_index is not None:
            kwargs['rotor_index'] = rotor_index
            kwargs['torsions'] = [sched.species_dict['TS_test'].rotors_dict[rotor_index]['torsion']]
        job = job_factory(job_adapter='gaussian', project='project_test', ess_settings=self.ess_settings,
                          species=[sched.species_dict['TS_test']], xyz=self.ts_xyz, job_type=job_type,
                          level=Level(repr={'method': 'wb97xd', 'basis': 'def2tzvp'}),
                          project_directory=sched.project_directory, job_num=job_num, **kwargs)
        job.job_status = ['done', {'status': 'errored', 'keywords': ['MaxOptCycles'],
                                   'error': 'Maximum optimization cycles reached', 'line': 'Number of steps exceeded'}]
        return job

    @patch('arc.scheduler.trsh_ess_job')
    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_exhausted_scan_invalidates_rotor_and_keeps_ts(self, mock_switch_ts, mock_trsh):
        """An unconvergeable rotor scan must invalidate only that rotor, never discard the TS."""
        mock_trsh.return_value = (list(), list(), False, Level(repr='wb97xd/def2tzvp'),
                                  'gaussian', 'scan', False, '', 14, '', 8, True)
        sched = self.build_sched('scan')
        job = self.make_failed_job(sched, 'scan', 300, rotor_index=0)
        sched.troubleshoot_ess(label='TS_test', job=job, level_of_theory=Level(repr='wb97xd/def2tzvp'))
        mock_switch_ts.assert_not_called()
        rotor = sched.species_dict['TS_test'].rotors_dict[0]
        self.assertIs(rotor['success'], False)
        self.assertIn('exhausted', rotor['invalidation_reason'])

    @patch('arc.scheduler.trsh_ess_job')
    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_exhausted_directed_scan_invalidates_rotor(self, mock_switch_ts, mock_trsh):
        """A directed scan is gated by the same 'success is not None' check, so it must also be invalidated."""
        mock_trsh.return_value = (list(), list(), False, Level(repr='wb97xd/def2tzvp'),
                                  'gaussian', 'directed_scan', False, '', 14, '', 8, True)
        sched = self.build_sched('directed')
        job = self.make_failed_job(sched, 'directed_scan', 301, rotor_index=0)
        sched.troubleshoot_ess(label='TS_test', job=job, level_of_theory=Level(repr='wb97xd/def2tzvp'))
        mock_switch_ts.assert_not_called()
        self.assertIs(sched.species_dict['TS_test'].rotors_dict[0]['success'], False)

    @patch('arc.scheduler.trsh_ess_job')
    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_exhausted_opt_still_switches_ts(self, mock_switch_ts, mock_trsh):
        """A failed geometry-determining job must still reject the TS guess, as before."""
        mock_trsh.return_value = (list(), list(), False, Level(repr='wb97xd/def2tzvp'),
                                  'gaussian', 'opt', False, '', 14, '', 8, True)
        sched = self.build_sched('opt')
        job = self.make_failed_job(sched, 'opt', 302)
        sched.troubleshoot_ess(label='TS_test', job=job, level_of_theory=Level(repr='wb97xd/def2tzvp'))
        mock_switch_ts.assert_called_once_with(label='TS_test')

    @patch('arc.scheduler.trsh_ess_job')
    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_exhausted_sp_does_not_switch_ts(self, mock_switch_ts, mock_trsh):
        """A refinement job that cannot invalidate a geometry must not discard the TS."""
        mock_trsh.return_value = (list(), list(), False, Level(repr='wb97xd/def2tzvp'),
                                  'gaussian', 'sp', False, '', 14, '', 8, True)
        sched = self.build_sched('sp')
        job = self.make_failed_job(sched, 'sp', 303)
        sched.troubleshoot_ess(label='TS_test', job=job, level_of_theory=Level(repr='wb97xd/def2tzvp'))
        mock_switch_ts.assert_not_called()

    @patch('arc.scheduler.trsh_ess_job')
    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_exhausted_scan_on_a_rotor_without_an_invalidation_reason(self, mock_switch_ts, mock_trsh):
        """
        A rotor restored from older restart data can lack the 'invalidation_reason' key.

        ``ARCSpecies.as_dict()`` serializes a rotor by iterating the keys it happens to have, so a
        key absent when the project was written stays absent after the round trip. Appending to it
        must therefore not assume it exists.
        """
        mock_trsh.return_value = (list(), list(), False, Level(repr='wb97xd/def2tzvp'),
                                  'gaussian', 'scan', False, '', 14, '', 8, True)
        sched = self.build_sched('legacy_rotor')
        del sched.species_dict['TS_test'].rotors_dict[0]['invalidation_reason']
        job = self.make_failed_job(sched, 'scan', 305, rotor_index=0)
        sched.troubleshoot_ess(label='TS_test', job=job, level_of_theory=Level(repr='wb97xd/def2tzvp'))
        mock_switch_ts.assert_not_called()
        rotor = sched.species_dict['TS_test'].rotors_dict[0]
        self.assertIs(rotor['success'], False)
        self.assertIn('exhausted', rotor['invalidation_reason'])

    @patch('arc.scheduler.trsh_ess_job')
    @patch('arc.scheduler.Scheduler.switch_ts')
    def test_scan_rotor_not_left_pending(self, mock_switch_ts, mock_trsh):
        """Leaving 'success' as None would make run_scan_jobs() re-spawn the exhausted scan."""
        mock_trsh.return_value = (list(), list(), False, Level(repr='wb97xd/def2tzvp'),
                                  'gaussian', 'scan', False, '', 14, '', 8, True)
        sched = self.build_sched('pending')
        job = self.make_failed_job(sched, 'scan', 304, rotor_index=0)
        sched.troubleshoot_ess(label='TS_test', job=job, level_of_theory=Level(repr='wb97xd/def2tzvp'))
        self.assertIsNotNone(sched.species_dict['TS_test'].rotors_dict[0]['success'])


class TestTroubleshootOptJobIdentity(unittest.TestCase):
    """
    Test that opt troubleshooting acts on the job that actually failed, rather than re-deriving one
    from the highest job number. Job numbers are only monotonic within a single execution, so a
    restarted project holds several numbering eras at once.
    """

    @classmethod
    def setUpClass(cls):
        cls.ess_settings = {'gaussian': ['server1'], 'molpro': ['server2', 'server1'], 'qchem': ['server1']}
        cls.xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
        H       1.81402049    1.03819414    0.00000000
        H       0.00000000    0.00000000    0.00000000
        H       0.91779059    1.22790192    0.72426890""")

    def build_sched(self, name: str):
        """Build a minimal Scheduler holding one species."""
        spc = ARCSpecies(label='spc_test', xyz=self.xyz, multiplicity=1, charge=0, compute_thermo=False)
        project_directory = os.path.join(ARC_PATH, 'Projects', f'arc_test_optjob_{name}')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        sched = Scheduler(project=f'test_optjob_{name}', ess_settings=self.ess_settings,
                          species_list=[spc],
                          opt_level=Level(repr=default_levels_of_theory['opt']),
                          freq_level=Level(repr=default_levels_of_theory['freq']),
                          sp_level=Level(repr=default_levels_of_theory['sp']),
                          ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                          project_directory=project_directory,
                          testing=True,
                          job_types={'conf_opt': False, 'opt': True, 'fine': True, 'freq': True,
                                     'sp': True, 'rotors': False, 'irc': False},
                          )
        sched.trsh_ess_jobs = True
        return sched

    @staticmethod
    def make_job(job_name: str, fine: bool, ess_done: bool):
        """Build a stand-in opt job. Only the attributes troubleshoot_opt_jobs reads are set."""
        job = MagicMock()
        job.job_name = job_name
        job.fine = fine
        job.ess_trsh_methods = list()
        job.job_status = ['done', {'status': 'done' if ess_done else 'errored',
                                   'keywords': [] if ess_done else ['SCF'],
                                   'error': '' if ess_done else 'SCF failed', 'line': ''}]
        return job

    def register(self, sched, jobs):
        """Register opt jobs on the scheduler in the given (insertion) order."""
        sched.job_dict['spc_test'] = {'opt': {j.job_name: j for j in jobs}}

    def test_restarted_project_does_not_troubleshoot_a_stale_higher_numbered_job(self):
        """The reported crash: a stale high-numbered done+fine job must not be picked over the real failure."""
        sched = self.build_sched('stale')
        stale = self.make_job('opt_a9755', fine=True, ess_done=True)     # old era, higher number, looks healthy
        failed = self.make_job('opt_a2098', fine=True, ess_done=False)   # actually latest, actually failed
        self.register(sched, [stale, failed])
        with patch.object(Scheduler, 'troubleshoot_ess') as mock_trsh_ess:
            sched.troubleshoot_opt_jobs(label='spc_test', job=failed)
        mock_trsh_ess.assert_called_once()
        self.assertIs(mock_trsh_ess.call_args.kwargs['job'], failed)

    def test_get_latest_opt_job_uses_insertion_order_not_job_number(self):
        """Recency must come from insertion order; the highest number can belong to an old era."""
        sched = self.build_sched('latest')
        stale = self.make_job('opt_a9755', fine=True, ess_done=True)
        newest = self.make_job('opt_a2098', fine=True, ess_done=True)
        self.register(sched, [stale, newest])
        self.assertIs(sched.get_latest_opt_job('spc_test'), newest)

    def test_get_preceding_opt_job_matches_by_identity(self):
        """The predecessor is resolved by the job's identity, not by decrementing its number."""
        sched = self.build_sched('prev')
        first = self.make_job('opt_a9755', fine=False, ess_done=True)
        second = self.make_job('opt_a2095', fine=False, ess_done=True)
        third = self.make_job('opt_a2098', fine=True, ess_done=False)
        self.register(sched, [first, second, third])
        self.assertIs(sched.get_preceding_opt_job('spc_test', third), second)
        self.assertIsNone(sched.get_preceding_opt_job('spc_test', first))

    def test_get_preceding_opt_job_returns_none_for_unregistered_job(self):
        """A job that is not registered for this species has no predecessor."""
        sched = self.build_sched('unreg')
        registered = self.make_job('opt_a2095', fine=False, ess_done=True)
        self.register(sched, [registered])
        stranger = self.make_job('opt_a3000', fine=True, ess_done=False)
        self.assertIsNone(sched.get_preceding_opt_job('spc_test', stranger))

    def test_stale_high_numbered_job_is_not_selected_when_no_job_is_passed(self):
        """
        Reproduces the reported crash using the original call signature.

        A restarted project holds a stale ``opt_a9755`` (done, fine) alongside the real, newer
        ``opt_a2098`` that failed. Selecting by highest job number picks the stale one, finds it
        healthy, and raises ``SchedulerError: opt job ... seems right, yet "run_opt_job" was
        called``, which aborts the whole run.
        """
        sched = self.build_sched('fallback')
        stale = self.make_job('opt_a9755', fine=True, ess_done=True)
        failed = self.make_job('opt_a2098', fine=True, ess_done=False)
        self.register(sched, [stale, failed])
        with patch.object(Scheduler, 'troubleshoot_ess') as mock_trsh_ess:
            sched.troubleshoot_opt_jobs(label='spc_test')
        mock_trsh_ess.assert_called_once()
        self.assertIs(mock_trsh_ess.call_args.kwargs['job'], failed)

    def test_no_opt_jobs_does_not_raise(self):
        """With no opt job to troubleshoot, log and return rather than raising."""
        sched = self.build_sched('none')
        sched.job_dict['spc_test'] = {'opt': dict()}
        self.assertIsNone(sched.troubleshoot_opt_jobs(label='spc_test'))


class TestSchedulerTSGuessReportAlignment(unittest.TestCase):
    """
    Contains unit tests for the column alignment of the successful TS guess block reported by
    Scheduler.determine_most_likely_ts_conformer().
    """

    TITLES = ['TS Guess', 'Method', 'Rel. Energy', 'Guess Time', 'Img Freq']

    @staticmethod
    def make_ts_guess(index, method, method_sources, energy, execution_time,
                      success=True, errors='', imaginary_freqs=None):
        """Return a TSGuess with the given reporting attributes and a geometry unique to its index."""
        tsg = TSGuess(index=index,
                      method=method,
                      energy=energy,
                      execution_time=execution_time,
                      xyz={'symbols': ('H', 'H'), 'isotopes': (1, 1),
                           'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.74 + 0.01 * index))},
                      )
        tsg.method_sources = method_sources
        tsg.success = success
        tsg.errors = errors
        tsg.imaginary_freqs = imaginary_freqs if imaginary_freqs is not None else [-500.0 - index]
        tsg.opt_xyz = tsg.initial_xyz
        return tsg

    def build_scheduler(self, ts_guesses):
        """Return a Scheduler holding a TS species carrying the given guesses."""
        ts = ARCSpecies(label='TS0', is_ts=True)
        ts.ts_guesses = ts_guesses
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.species_dict = {'TS0': ts}
        scheduler.output = {'TS0': {'paths': dict()}}
        scheduler.project_directory = ''
        scheduler.ts_guess_level = None
        return scheduler

    def invoke(self, scheduler, level='INFO', draw=False):
        """Run the TS guess selection once and return the lines emitted after the guess block header."""
        with nullcontext() if draw else patch('arc.scheduler.plotter.draw_structure'), \
                patch('arc.scheduler.plotter.save_conformers_file'):
            with self.assertLogs('arc', level=level) as cm:
                scheduler.determine_most_likely_ts_conformer(label='TS0')
        messages = [record.getMessage() for record in cm.records]
        start = next(i for i, message in enumerate(messages) if 'Geometry *guesses*' in message)
        return [message for message in messages[start + 1:]
                if message.strip() and not message.startswith('TS guesses not listed above')]

    def report(self, ts_guesses):
        """Run the TS guess selection on the given guesses and return the emitted table lines."""
        return self.invoke(self.build_scheduler(ts_guesses))

    def column_spans(self, lines):
        """Return the (start, stop) offset of each column, read off the rule line of the given table."""
        rules = [line for line in lines if line and set(line) <= set('- ')]
        self.assertTrue(rules, msg='the reported block has no rule line, so it is not a table:\n' + '\n'.join(lines))
        rule = rules[0]
        spans, start = list(), None
        for i, char in enumerate(rule + ' '):
            if char == '-' and start is None:
                start = i
            elif char != '-' and start is not None:
                spans.append((start, i))
                start = None
        return spans

    def assert_tabulated(self, lines, n_rows, titles=None):
        """Assert that the given lines form a table whose cells stay inside their own columns."""
        titles = titles if titles is not None else self.TITLES
        spans = self.column_spans(lines)
        rendered = '\n'.join(lines)
        self.assertEqual(len(spans), len(titles), msg=f'expected {len(titles)} columns in:\n{rendered}')
        title_line = lines[0]
        for title, (start, stop) in zip(titles, spans):
            self.assertIn(title, title_line[start:stop], msg=f'{title!r} is not in its own column in:\n{rendered}')
        for line in lines:
            padded = line.ljust(spans[-1][1])
            for start, stop in spans:
                if start:
                    self.assertEqual(padded[start - 1], ' ',
                                     msg=f'a cell bleeds into the column at offset {start} in:\n{rendered}')
            self.assertEqual(''.join(padded[start:stop] for start, stop in spans).replace(' ', ''),
                             line.replace(' ', ''),
                             msg=f'a character falls outside every column in:\n{rendered}')
        self.assertEqual(len(self.column_cells(lines, 0)), n_rows, msg=f'expected {n_rows} rows in:\n{rendered}')

    def column_cells(self, lines, column):
        """Return the stripped cells of the given column, for the data rows only."""
        spans = self.column_spans(lines)
        rule_index = next(i for i, line in enumerate(lines) if line and set(line) <= set('- '))
        start, stop = spans[column]
        return [line.ljust(stop)[start:stop].strip() for line in lines[rule_index + 1:]]

    def test_columns_align_across_mixed_methods_and_indices(self):
        """Test that the table holds given mixed method name lengths and 1-, 2- and 3-digit indices."""
        lines = self.report([
            self.make_ts_guess(0, 'heuristics', ['heuristics', 'crest'], -50.0, datetime.timedelta(seconds=3.42)),
            self.make_ts_guess(36, 'autotst', ['autotst'], -48.27, datetime.timedelta(seconds=18.15)),
            self.make_ts_guess(136, 'gcn', ['gcn'], 1234.5, datetime.timedelta(hours=13, minutes=7, seconds=6.5)),
        ])
        self.assert_tabulated(lines, n_rows=3)
        self.assertEqual(self.column_cells(lines, 0), ['0', '36', '136'])
        self.assertEqual(self.column_cells(lines, 1), ['heuristics (also: crest)', 'autotst', 'gcn'])
        self.assertEqual(self.column_cells(lines, 2), ['0.00', '1.73', '1284.50'])

    def test_a_status_column_is_added_only_when_a_reported_guess_has_an_error(self):
        """Test that the status column appears given an error on a reported guess, and is omitted otherwise."""
        lines = self.report([
            self.make_ts_guess(1, 'kinbot', ['kinbot'], 0.0, datetime.timedelta(seconds=1.5), errors='some error'),
            self.make_ts_guess(2, 'xtb_gsm', ['xtb_gsm', 'gcn', 'autotst'], 7.5, datetime.timedelta(days=2, hours=3)),
        ])
        self.assert_tabulated(lines, n_rows=2, titles=self.TITLES + ['Status'])
        self.assertEqual(self.column_cells(lines, 5), ['some error', ''])
        lines = self.report([
            self.make_ts_guess(1, 'kinbot', ['kinbot'], 0.0, datetime.timedelta(seconds=1.5)),
            self.make_ts_guess(2, 'xtb_gsm', ['xtb_gsm', 'gcn', 'autotst'], 7.5, datetime.timedelta(days=2, hours=3)),
        ])
        self.assert_tabulated(lines, n_rows=2)
        self.assertNotIn('Status', '\n'.join(lines))

    def test_the_guess_time_column_spans_seconds_to_days(self):
        """Test that the guess time column reports each duration in the largest unit it fills."""
        lines = self.report([
            self.make_ts_guess(1, 'heuristics', ['heuristics'], 0.0, datetime.timedelta(seconds=3.42)),
            self.make_ts_guess(2, 'autotst', ['autotst'], 1.0, datetime.timedelta(hours=13, minutes=7, seconds=6.5)),
            self.make_ts_guess(3, 'kinbot', ['kinbot'], 2.0, datetime.timedelta(days=2, hours=3)),
        ])
        self.assert_tabulated(lines, n_rows=3)
        self.assertEqual(self.column_cells(lines, 3), ['3.4 s', '13.1 h', '2.1 d'])

    def test_sub_minute_guess_times_stay_distinct(self):
        """Test that the sub-minute guess timings of a real run do not collapse into one cell."""
        lines = self.report([
            self.make_ts_guess(1, 'heuristics', ['heuristics'], 0.0, datetime.timedelta(seconds=3.4)),
            self.make_ts_guess(2, 'gcn', ['gcn'], 1.0, datetime.timedelta(seconds=16.8)),
            self.make_ts_guess(3, 'autotst', ['autotst'], 2.0, datetime.timedelta(seconds=18.1)),
        ])
        cells = self.column_cells(lines, 3)
        self.assertEqual(len(set(cells)), 3, msg=f'the guess times collapsed into {set(cells)} in:\n'
                                                 + '\n'.join(lines))
        self.assertEqual(cells, ['3.4 s', '16.8 s', '18.1 s'])

    def test_the_imaginary_frequency_column(self):
        """Test that the imaginary frequency column lists the frequencies and is blank when there are none."""
        lines = self.report([
            self.make_ts_guess(1, 'heuristics', ['heuristics'], 0.0, datetime.timedelta(seconds=1),
                               imaginary_freqs=[-1204.53]),
            self.make_ts_guess(2, 'autotst', ['autotst'], 1.0, datetime.timedelta(seconds=1),
                               imaginary_freqs=[-209.4, -109.9]),
        ])
        self.assert_tabulated(lines, n_rows=2)
        self.assertEqual(self.column_cells(lines, 4), ['-1204.5', '-209.4, -109.9'])

    def test_column_widths_ignore_unreported_guesses(self):
        """Test that a guess which is not reported does not widen the columns of the guesses that are."""
        def reported_guesses():
            return [self.make_ts_guess(3, 'heuristics', ['heuristics'], 0.0, datetime.timedelta(seconds=2.5)),
                    self.make_ts_guess(4, 'autotst', ['autotst'], 3.5, datetime.timedelta(seconds=4.5))]
        omitted = self.make_ts_guess(1234, 'user guess', ['user guess', 'heuristics', 'autotst', 'gcn'],
                                     9.5, datetime.timedelta(days=5), success=False, errors='a very long error')
        narrow = self.report(reported_guesses())
        wide = self.report(reported_guesses() + [omitted])
        self.assert_tabulated(narrow, n_rows=2)
        self.assert_tabulated(wide, n_rows=2)
        self.assertEqual(narrow, wide)
        self.assertNotIn('a very long error', '\n'.join(wide))

    def test_no_line_interrupts_the_table_at_debug_verbosity(self):
        """Test that the per-guess draw does not emit a line between the rule and the last table row"""
        scheduler = self.build_scheduler([
            self.make_ts_guess(0, 'heuristics', ['heuristics'], -50.0, datetime.timedelta(seconds=3.42)),
            self.make_ts_guess(1, 'autotst', ['autotst'], -48.0, datetime.timedelta(seconds=18.15)),
        ])
        lines = self.invoke(scheduler, level='DEBUG', draw=True)
        rule_index = next(i for i, line in enumerate(lines) if line and set(line) <= set('- '))
        table = lines[:rule_index + 3]
        self.assert_tabulated(table, n_rows=2)
        self.assertEqual(self.column_cells(table, 0), ['0', '1'],
                         msg='a non-table line was emitted between the table rows:\n' + '\n'.join(lines))
        self.assertIn('not drawing 3D!', lines,
                      msg='the structures were not drawn, so this test cannot detect an interruption')
        self.assertGreater(lines.index('not drawing 3D!'), rule_index + 2)

    def test_every_reported_guess_is_drawn_once_in_row_order(self):
        """Test that the structures are drawn once per reported row, in the order of the rows"""
        guesses = [self.make_ts_guess(0, 'heuristics', ['heuristics'], -50.0, datetime.timedelta(seconds=3.42)),
                   self.make_ts_guess(1, 'autotst', ['autotst'], -48.0, datetime.timedelta(seconds=18.15)),
                   self.make_ts_guess(2, 'gcn', ['gcn'], None, datetime.timedelta(seconds=1), success=False)]
        scheduler = self.build_scheduler(guesses)
        with patch('arc.scheduler.plotter.draw_structure') as draw_structure, \
                patch('arc.scheduler.plotter.save_conformers_file'):
            scheduler.determine_most_likely_ts_conformer(label='TS0')
        self.assertEqual([call.kwargs['xyz'] for call in draw_structure.call_args_list],
                         [guesses[0].initial_xyz, guesses[1].initial_xyz])

    def test_the_reported_energies_are_relative_to_the_lowest_energy_of_any_guess(self):
        """Test that an unsuccessful guess holding the lowest energy is still the reference point"""
        lines = self.report([
            self.make_ts_guess(1, 'heuristics', ['heuristics'], 10.0, datetime.timedelta(seconds=1)),
            self.make_ts_guess(2, 'autotst', ['autotst'], 20.0, datetime.timedelta(seconds=2)),
            self.make_ts_guess(3, 'kinbot', ['kinbot'], -100.0, datetime.timedelta(seconds=3), success=False),
        ])
        self.assertEqual(self.column_cells(lines, 2), ['110.00', '120.00'])

    def test_a_repeat_invocation_reports_the_same_energies(self):
        """Test that invoking the selection twice for one label does not shift the reported energies"""
        scheduler = self.build_scheduler([
            self.make_ts_guess(1, 'heuristics', ['heuristics'], 10.0, datetime.timedelta(seconds=1)),
            self.make_ts_guess(2, 'autotst', ['autotst'], 20.0, datetime.timedelta(seconds=2)),
            self.make_ts_guess(3, 'kinbot', ['kinbot'], -100.0, datetime.timedelta(seconds=3), success=False),
        ])
        first = self.invoke(scheduler)
        second = self.invoke(scheduler)
        self.assertEqual(self.column_cells(first, 2), ['110.00', '120.00'])
        self.assertEqual(self.column_cells(second, 2), self.column_cells(first, 2))
        self.assertEqual([tsg.energy for tsg in scheduler.species_dict['TS0'].ts_guesses], [110.0, 120.0, 0.0])


class TestQueueSnapshotPerPass(unittest.TestCase):
    """
    The queue snapshot (``server_job_ids``) is taken once per pass of the scheduling loop, so a job submitted
    while one label is processed must already be in it when a later label of the same pass is checked.
    """

    labels = [f'snap_{i}' for i in range(8)]

    def setUp(self):
        """Build a Scheduler over several species, with scratch outside the fixtures directory."""
        project_directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        species = [ARCSpecies(label=label, smiles='C', compute_thermo=False) for label in self.labels]
        self.sched = Scheduler(project='snapshot_test',
                               ess_settings={'gaussian': ['server1']},
                               species_list=species,
                               opt_level=Level(repr='b3lyp/6-31g(d,p)'),
                               freq_level=Level(repr='b3lyp/6-31g(d,p)'),
                               sp_level=Level(repr='b3lyp/6-311+g(d,p)'),
                               project_directory=project_directory,
                               testing=True,
                               job_types=initialize_job_types(),
                               )
        self.sched.running_jobs = {label: list() for label in self.labels}
        self.sched.unique_species_labels = list(self.labels)
        self.sched.active_pipes = dict()
        for name in ('generate_final_ts_guess_report', 'spawn_ts_jobs', 'run_conformer_jobs',
                     'release_held_stability_work', 'save_restart_dict'):
            patcher = patch.object(self.sched, name)
            patcher.start()
            self.addCleanup(patcher.stop)

    def one_pass_only(self):
        """Make the scheduling loop end after its first pass."""
        def end_the_pass():
            self.sched.running_jobs.clear()
        patcher = patch.object(self.sched, 'flush_pending_pipe_batches', side_effect=end_the_pass)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_queue_is_polled_once_per_pass_not_once_per_label(self):
        """N labels in one pass cost one queue query and one incore scan, not N."""
        self.one_pass_only()
        with patch.object(self.sched, 'get_server_job_ids') as mock_ids, \
                patch.object(self.sched, 'get_completed_incore_jobs') as mock_incore, \
                patch.object(self.sched, 'check_all_done'):
            self.sched.schedule_jobs()
        self.assertEqual(mock_ids.call_count, 1)
        self.assertEqual(mock_incore.call_count, 1)

    def test_job_submitted_earlier_in_the_pass_is_not_treated_as_finished(self):
        """A job submitted while processing the first label is still running when a later label is checked."""
        self.one_pass_only()
        first, last = self.labels[0], self.labels[-1]

        def stale_queue_query(*args, **kwargs):
            """A queue query that cannot yet see a job submitted a moment ago."""
            self.sched.server_job_ids = list()

        def submit_a_job_for_the_last_label(label):
            """Submit the last label's first job while the first label is being processed."""
            if label == first:
                with patch('arc.scheduler.job_factory', side_effect=self.make_job):
                    self.sched.run_job(job_type='opt', label=last, level_of_theory='b3lyp/6-31g(d,p)',
                                       job_adapter='gaussian')

        with patch.object(self.sched, 'get_server_job_ids', side_effect=stale_queue_query), \
                patch.object(self.sched, 'get_completed_incore_jobs'), \
                patch.object(self.sched, 'check_all_done', side_effect=submit_a_job_for_the_last_label), \
                patch.object(self.sched, 'end_job', return_value=False) as mock_end_job:
            self.sched.schedule_jobs()
        mock_end_job.assert_not_called()

    def make_job(self, job_type, job_adapter=None, execution_type='queue', **kwargs):
        """Stand in for job_factory, returning a job that the queue accepted under a fixed ID."""
        job = StubJob(job_type=job_type, job_adapter=job_adapter)
        job.execution_type = execution_type
        job.job_id = '4242' if execution_type == 'queue' else 7
        return job

    def test_run_job_adds_a_submitted_queue_job_to_the_snapshot(self):
        """A queue job's ID is in ``server_job_ids`` as soon as ``run_job`` returns."""
        self.sched.server_job_ids = ['1']
        with patch('arc.scheduler.job_factory', side_effect=self.make_job):
            self.sched.run_job(job_type='opt', label=self.labels[0], level_of_theory='b3lyp/6-31g(d,p)',
                               job_adapter='gaussian')
        self.assertEqual(self.sched.server_job_ids, ['1', '4242'])

    def test_run_job_leaves_the_snapshot_alone_for_a_failed_submission(self):
        """A submission that failed returns an empty job ID ('' locally, 0 over SSH), which must not enter the snapshot."""
        for failed_id in ('', 0, None):
            with self.subTest(job_id=failed_id):
                self.sched.server_job_ids = ['1']

                def failed_job(job_type, job_adapter=None, **kwargs):
                    job = StubJob(job_type=job_type, job_adapter=job_adapter)
                    job.execution_type = 'queue'
                    job.job_id = failed_id
                    return job

                with patch('arc.scheduler.job_factory', side_effect=failed_job):
                    self.sched.run_job(job_type='opt', label=self.labels[0], level_of_theory='b3lyp/6-31g(d,p)',
                                       job_adapter='gaussian')
                self.assertEqual(self.sched.server_job_ids, ['1'])

    def test_run_job_leaves_the_snapshot_alone_for_an_incore_job(self):
        """An incore job has no queue ID, so it must not enter the queue snapshot."""
        with patch('arc.scheduler.job_factory', side_effect=self.make_job):
            self.sched.run_job(job_type='opt', label=self.labels[0], level_of_theory='b3lyp/6-31g(d,p)',
                               job_adapter=sorted(default_incore_adapters)[0])
        self.assertEqual(self.sched.server_job_ids, list())


class TestGetServerJobIds(unittest.TestCase):
    """The status poll runs every cycle for every job, so it is the hottest SSH caller there is."""

    @staticmethod
    def _sched(servers):
        """A stand-in carrying only what get_server_job_ids() reads."""
        return SimpleNamespace(servers=servers, server_job_ids=None, stale_servers=set())

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


class TestJobLevelRecording(unittest.TestCase):
    """Tests that the scheduler records the level of each job whose log path it stores."""

    def setUp(self):
        self.project_directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.project_directory, ignore_errors=True)
        self.run_opt = Level(method='b3lyp', basis='6-31g(d,p)')
        self.job_opt = Level(method='wb97xd', basis='def2-tzvp', software='gaussian')
        self.job_sp = Level(method='dlpno-ccsd(t)', basis='cc-pvtz', software='orca')
        self.job_levels = {'opt': self.job_opt, 'freq': self.job_opt, 'sp': self.job_sp}
        self.ts_xyz = str_to_xyz("""N       0.91779059    0.51946178    0.00000000
H       1.81402049    1.03819414    0.00000000
H       0.00000000    0.00000000    0.00000000
H       0.91779059    1.22790192    0.72426890""")
        self.spc = ARCSpecies(label='C2H6', smiles='CC')
        self.ts = ARCSpecies(label='TS0', is_ts=True, xyz=self.ts_xyz, multiplicity=1, charge=0,
                             compute_thermo=False)
        self.sched = Scheduler(project='levels_test', ess_settings={'gaussian': ['server1']},
                               species_list=[self.spc, self.ts],
                               opt_level=self.run_opt,
                               freq_level=self.run_opt,
                               sp_level=Level(method='b3lyp', basis='6-311+g(d,p)'),
                               ts_guess_level=Level(repr=default_levels_of_theory['ts_guesses']),
                               project_directory=self.project_directory,
                               testing=True,
                               job_types={'conf_opt': True, 'conf_sp': False, 'opt': True, 'fine': False,
                                          'freq': True, 'sp': True, 'rotors': False, 'irc': True},
                               )
        self.log = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'calcs', 'Species', 'NH2_freq.out')

    def make_job(self, level, job_type='opt', **kwargs):
        """Build a stand-in for a completed ESS job."""
        return SimpleNamespace(job_type=job_type, level=level, local_path_to_output_file=self.log,
                               local_path_to_xyz=None, job_name=f'{job_type}_a1', job_adapter='gaussian',
                               is_ts=False, job_status=['done', {'status': 'done'}], restricted_used=True,
                               **kwargs)

    def test_a_gaussian_job_is_recorded_at_the_requested_level(self):
        """Test that the level recorded for a Gaussian job keeps the requested spelling"""
        requested = Level(method='wb97xd', basis='def2-tzvp', software='gaussian')
        with patch('arc.scheduler.job_factory', side_effect=partial(job_factory, testing=True)), \
                patch('arc.job.adapters.gaussian.GaussianAdapter.execute'), \
                patch.object(self.sched, 'check_max_simultaneous_jobs_limit'), \
                patch.object(self.sched, 'save_restart_dict'):
            self.sched.run_job(label='C2H6', job_type='opt', level_of_theory=requested, job_adapter='gaussian')
        job = next(iter(self.sched.job_dict['C2H6']['opt'].values()))
        self.assertEqual(job.level.basis, 'def2-tzvp')
        self.sched.record_job_level(label='C2H6', job_key='opt', level=job.level)
        self.assertEqual(self.sched.output['C2H6']['levels']['opt']['basis'], 'def2-tzvp')

    def test_the_irc_level_survives_a_troubleshot_direction(self):
        """Test that a forward IRC and a reverse IRC respawned from the first job's level record one level"""
        requested = Level(method='wb97xd', basis='def2-tzvp', software='gaussian')
        jobs = list()
        with patch('arc.scheduler.job_factory', side_effect=partial(job_factory, testing=True)), \
                patch('arc.job.adapters.gaussian.GaussianAdapter.execute'), \
                patch.object(self.sched, 'check_max_simultaneous_jobs_limit'), \
                patch.object(self.sched, 'save_restart_dict'):
            self.sched.run_job(label='TS0', job_type='irc', level_of_theory=requested, job_adapter='gaussian',
                               irc_direction='forward')
            jobs.append(next(iter(self.sched.job_dict['TS0']['irc'].values())))
            self.sched.run_job(label='TS0', job_type='irc', level_of_theory=jobs[0].level, job_adapter='gaussian',
                               irc_direction='reverse')
            jobs.append(list(self.sched.job_dict['TS0']['irc'].values())[-1])
        self.sched.output['TS0']['paths']['irc'] = ['forward.out']
        self.sched.record_irc_level(label='TS0', level=jobs[0].level)
        self.sched.output['TS0']['paths']['irc'].append('reverse.out')
        self.sched.record_irc_level(label='TS0', level=jobs[1].level)
        self.assertEqual(self.sched.output['TS0']['levels']['irc']['basis'], 'def2-tzvp')

    def test_a_new_output_dict_has_an_empty_levels_record(self):
        """Test that the output dictionary of every species starts with a levels record"""
        self.assertEqual(self.sched.output['C2H6']['levels'], dict())

    def test_the_opt_job_level_is_recorded_and_stamped_on_the_species(self):
        """Test that opt records the level of the job (adaptive level), not the run's opt level, and stamps opt_level"""
        with patch('arc.scheduler.plotter.save_geo'):
            self.spc.final_xyz = self.ts_xyz
            self.sched.post_opt_geo_work('C2H6', self.make_job(self.job_opt))
        self.assertEqual(self.sched.output['C2H6']['levels']['opt'], self.job_opt.as_dict())
        self.assertNotEqual(self.sched.output['C2H6']['levels']['opt'], self.run_opt.as_dict())
        self.assertEqual(self.spc.opt_level, self.job_opt.simple())
        self.assertNotEqual(self.spc.opt_level, self.run_opt.simple())

    def test_a_rerun_opt_at_a_different_level_replaces_the_recorded_level(self):
        """Test that the recorded level is that of the job whose log is exported, after troubleshooting changed it"""
        retry_level = Level(method='b3lyp', basis='6-31g(d,p)', software='gaussian')
        with patch('arc.scheduler.plotter.save_geo'):
            self.spc.final_xyz = self.ts_xyz
            self.sched.post_opt_geo_work('C2H6', self.make_job(self.job_opt))
            self.sched.post_opt_geo_work('C2H6', self.make_job(retry_level))
        self.assertEqual(self.sched.output['C2H6']['levels']['opt'], retry_level.as_dict())

    def test_the_freq_job_level_is_recorded(self):
        """Test that a converged stable-species freq job records its own level next to its log path"""
        freq_level = Level(method='wb97xd', basis='def2-svp', software='gaussian')
        self.sched.post_freq_actions(label='C2H6', job=self.make_job(freq_level, 'freq'),
                                     vibfreqs=[100.0, 200.0, 300.0])
        self.assertEqual(self.sched.output['C2H6']['paths']['freq'], self.log)
        self.assertEqual(self.sched.output['C2H6']['levels']['freq'], freq_level.as_dict())

    def test_a_freq_job_with_an_unexpected_imaginary_mode_records_no_level(self):
        """Test that no level is recorded for a freq job whose log path is not recorded"""
        self.sched.post_freq_actions(label='C2H6', job=self.make_job(self.job_opt, 'freq'),
                                     vibfreqs=[-100.0, 200.0, 300.0])
        self.assertNotIn('freq', self.sched.output['C2H6']['levels'])

    def test_the_sp_job_level_is_recorded(self):
        """Test that the sp job records its own level"""
        with patch('arc.scheduler.parser.parse_e_elect', return_value=-100.0), \
                patch.object(self.sched, 'check_spin_contamination'), \
                patch.object(self.sched, 'save_e_elect'):
            self.sched.post_sp_actions('C2H6', sp_path=self.log, level=self.job_sp,
                                       job=self.make_job(self.job_sp, 'sp'))
        self.assertEqual(self.sched.output['C2H6']['levels']['sp'], self.job_sp.as_dict())

    def sp_actions(self, **kwargs):
        """Run post_sp_actions on C2H6 with the parsers stubbed out."""
        with patch('arc.scheduler.parser.parse_e_elect', return_value=-100.0), \
                patch.object(self.sched, 'check_spin_contamination'), \
                patch.object(self.sched, 'save_e_elect'):
            self.sched.post_sp_actions('C2H6', **kwargs)

    def test_an_sp_read_from_the_opt_log_records_the_level_of_the_opt_job(self):
        """Test that a job-less sp taken from the geometry log takes the recorded opt level, not the run default"""
        self.sched.record_job_level(label='C2H6', job_key='opt', level=self.job_opt)
        self.sched.output['C2H6']['paths']['geo'] = self.log
        self.sp_actions(sp_path=self.log, level=self.run_opt)
        self.assertEqual(self.sched.output['C2H6']['levels']['sp'], self.job_opt.as_dict())

    def test_a_jobless_sp_from_another_log_records_the_level_argument(self):
        """Test that a pipe-finalized sp, whose log is not the geometry log, records the level it ran at"""
        self.sched.record_job_level(label='C2H6', job_key='opt', level=self.job_opt)
        self.sched.output['C2H6']['paths']['geo'] = os.path.join(self.project_directory, 'opt.out')
        self.sp_actions(sp_path=self.log, level=self.job_sp)
        self.assertEqual(self.sched.output['C2H6']['levels']['sp'], self.job_sp.as_dict())

    def test_a_composite_sp_records_the_composite_level(self):
        """Test that the sp read from a composite log records the composite job's level"""
        composite = Level(method='cbs-qb3', software='gaussian')
        self.sched.record_job_level(label='C2H6', job_key='opt', level=self.job_opt)
        self.sp_actions(sp_path=self.log, level=composite)
        self.assertEqual(self.sched.output['C2H6']['levels']['sp'], composite.as_dict())

    def test_the_sp_level_is_restored_after_the_solvation_scheme_jobs(self):
        """Test that the extra solvation-scheme sp jobs do not replace the level of the exported sp log"""
        scheme = Level(method='b3lyp', basis='6-31g')
        self.sched.sp_level = Level(method='wb97xd', basis='def2-tzvp', solvation_method='smd', solvent='water',
                                    solvation_scheme_level=scheme)
        self.sched.output['C2H6']['job_types']['sp'] = True
        self.sched.record_job_level(label='C2H6', job_key='sp', level=self.job_sp)
        original_log = os.path.join(ARC_TESTING_PATH, 'methylamine_conformer_0.out')
        scheme_log = os.path.join(ARC_TESTING_PATH, 'methylamine_conformer_1.out')
        original_energy = parser.parse_e_elect(original_log)
        scheme_energy = parser.parse_e_elect(scheme_log)
        self.assertNotEqual(original_energy, scheme_energy)
        self.sched.output['C2H6']['paths']['sp'] = original_log
        solvated = Level(method='wb97xd', basis='def2-tzvp', solvation_method='smd', solvent='water')
        with patch.object(self.sched, 'check_spin_contamination'), patch.object(self.sched, 'save_e_elect'):
            self.sched.post_sp_actions('C2H6', sp_path=scheme_log, level=solvated,
                                       job=self.make_job(solvated, 'sp'))
        self.assertEqual(self.sched.output['C2H6']['paths']['sp'], original_log)
        self.assertEqual(self.sched.output['C2H6']['levels']['sp'], self.job_sp.as_dict())
        self.assertEqual(self.sched.output['C2H6']['paths']['sp_sol'], scheme_log)
        self.assertEqual(self.sched.species_dict['C2H6'].e_elect, original_energy)

    def test_the_t1_diagnostic_is_restored_after_the_solvation_scheme_jobs(self):
        """Test that the T1 diagnostic is the original sp job's, not the last scheme job's"""
        scheme = Level(method='ccsd', basis='6-31g')
        self.sched.sp_level = Level(method='ccsd(t)', basis='cc-pvtz', solvation_method='smd', solvent='water',
                                    solvation_scheme_level=scheme)
        self.sched.output['C2H6']['job_types']['sp'] = True
        original_log = os.path.join(ARC_TESTING_PATH, 'methylamine_conformer_0.out')
        scheme_log = os.path.join(ARC_TESTING_PATH, 'methylamine_conformer_1.out')
        self.sched.output['C2H6']['paths']['sp'] = original_log
        solvated = Level(method='ccsd(t)', basis='cc-pvtz', solvation_method='smd', solvent='water')
        t1_by_path = {original_log: 0.01, scheme_log: 0.03}
        with patch('arc.scheduler.parser.parse_t1', side_effect=lambda path: t1_by_path[path]), \
                patch.object(self.sched, 'check_spin_contamination'), patch.object(self.sched, 'save_e_elect'):
            self.sched.post_sp_actions('C2H6', sp_path=scheme_log, level=solvated, job=self.make_job(solvated, 'sp'))
        self.assertEqual(self.sched.species_dict['C2H6'].t1, 0.01)

    def test_the_mrci_pre_sp_records_its_level(self):
        """Test that the CCSD job run before an MRCI sp records its own level next to its log path"""
        self.sched.sp_level = Level(method='mrci', basis='cc-pvtz')
        ccsd = Level(method='ccsd', basis='cc-pvdz', software='molpro')
        with patch.object(self.sched, 'run_sp_job'):
            self.sched.check_sp_job('C2H6', self.make_job(ccsd, 'sp'))
        self.assertEqual(self.sched.output['C2H6']['levels']['sp'], ccsd.as_dict())

    def test_a_monoatomic_sp_log_that_becomes_the_geometry_log_records_no_opt_level(self):
        """Test that no opt level is recorded for a species whose sp log stands in for the opt log"""
        job = self.make_job(self.job_sp, 'sp')
        with patch.object(ARCSpecies, 'number_of_atoms', new_callable=PropertyMock, return_value=1), \
                patch.object(self.sched, 'post_sp_actions'), \
                patch.object(self.sched, 'save_restart_dict'):
            self.sched.check_sp_job('C2H6', job)
        self.assertEqual(self.sched.output['C2H6']['paths']['geo'], self.log)
        self.assertNotIn('opt', self.sched.output['C2H6']['levels'])

    def test_the_composite_job_level_is_recorded_and_stamped_on_the_species(self):
        """Test that a composite job records its level and stamps it on the species instead of the run's method"""
        composite = Level(method='cbs-qb3', software='gaussian')
        self.sched.composite_method = Level(method='g4', software='gaussian')
        self.sched.trsh_ess_jobs = False
        with patch('arc.scheduler.parser.parse_geometry', return_value=self.ts_xyz), \
                patch('arc.scheduler.plotter'), \
                patch.object(self.sched, 'check_negative_freq', return_value=(False, False)), \
                patch('arc.scheduler.parser.parse_frequencies', return_value=[100.0]):
            self.sched.parse_composite_geo('C2H6', self.make_job(composite, 'composite'))
        self.assertEqual(self.sched.output['C2H6']['levels']['composite'], composite.as_dict())
        self.assertEqual(self.spc.opt_level, composite.simple())

    def test_a_composite_job_records_no_freq_level(self):
        """Test that the freq log that is the composite log records no freq level, and the composite one stays"""
        composite = Level(method='cbs-qb3', software='gaussian')
        job = self.make_job(composite, 'composite')
        self.sched.post_freq_actions(label='C2H6', job=job, vibfreqs=[100.0, 200.0, 300.0])
        self.sched.record_job_level(label='C2H6', job_key='composite', level=job.level)
        self.assertEqual(self.sched.output['C2H6']['paths']['freq'], self.log)
        self.assertNotIn('freq', self.sched.output['C2H6']['levels'])
        self.assertEqual(self.sched.output['C2H6']['levels']['composite'], composite.as_dict())

    def test_the_ts_freq_job_level_is_recorded(self):
        """Test that the TS branch of the imaginary frequency check records the level of the freq job"""
        freq_level = Level(method='wb97xd', basis='def2-svp', software='gaussian')
        job = self.make_job(freq_level, 'freq')
        self.assertIsNone(self.ts.chosen_ts)
        with patch.object(self.sched, 'switch_ts') as switch_ts:
            freq_ok, switched = self.sched.check_negative_freq(label='TS0', job=job, vibfreqs=[-1000.0, 100.0, 200.0])
        switch_ts.assert_not_called()
        self.assertTrue(freq_ok)
        self.assertFalse(switched)
        self.assertEqual(self.sched.output['TS0']['paths']['freq'], self.log)
        self.assertEqual(self.sched.output['TS0']['levels']['freq'], freq_level.as_dict())

    def test_an_irc_job_level_is_recorded_and_the_endpoint_carries_its_direction(self):
        """Test that IRC records the job level, and each endpoint species records the direction of its own job"""
        irc_level = Level(method='wb97xd', basis='def2-svp', software='gaussian')
        with patch('arc.scheduler.parser.parse_geometry', return_value=self.ts_xyz), \
                patch('arc.scheduler.plotter'), \
                patch.object(self.sched, 'run_job'):
            for direction in ('reverse', 'forward'):
                job = self.make_job(irc_level, 'irc', irc_direction=direction)
                job.species = [self.ts]
                self.sched.spawn_post_irc_jobs('TS0', job)
        self.assertEqual(self.sched.output['TS0']['levels']['irc'], irc_level.as_dict())
        self.assertEqual(self.sched.output['IRC_TS0_1']['irc_direction'], 'reverse')
        self.assertEqual(self.sched.output['IRC_TS0_2']['irc_direction'], 'forward')
        self.assertEqual(self.sched.species_dict['IRC_TS0_1'].irc_label, 'TS0')

    def test_an_irc_pair_at_different_levels_records_no_irc_level(self):
        """Test that forward and reverse IRC jobs at different levels leave the level of the pair unrecorded"""
        with patch('arc.scheduler.parser.parse_geometry', return_value=self.ts_xyz), \
                patch('arc.scheduler.plotter'), \
                patch.object(self.sched, 'run_job'):
            for direction, level in (('forward', self.job_opt), ('reverse', self.job_sp)):
                job = self.make_job(level, 'irc', irc_direction=direction)
                job.species = [self.ts]
                self.sched.spawn_post_irc_jobs('TS0', job)
        self.assertNotIn('irc', self.sched.output['TS0']['levels'])

    def test_each_irc_log_records_its_own_level_in_lockstep(self):
        """Test that logs at different levels each keep their own level, although the pair's level is dropped"""
        with patch('arc.scheduler.parser.parse_geometry', return_value=self.ts_xyz), \
                patch('arc.scheduler.plotter'), \
                patch.object(self.sched, 'run_job'):
            for direction, level in (('forward', self.job_opt), ('reverse', self.job_sp)):
                job = self.make_job(level, 'irc', irc_direction=direction)
                job.species = [self.ts]
                self.sched.spawn_post_irc_jobs('TS0', job)
        paths = self.sched.output['TS0']['paths']
        self.assertEqual(len(paths['irc']), 2)
        self.assertEqual(paths['irc_levels'], [self.job_opt.as_dict(), self.job_sp.as_dict()])
        self.assertEqual(paths['irc_directions'], ['forward', 'reverse'])
        self.assertNotIn('irc', self.sched.output['TS0']['levels'])

    def test_a_restart_output_without_levels_is_recorded_into_without_raising(self):
        """Test that an output dictionary restored from an older restart file, with no levels key, still records"""
        del self.sched.output['C2H6']['levels']
        self.sched.record_job_level(label='C2H6', job_key='opt', level=self.job_opt)
        self.assertEqual(self.sched.output['C2H6']['levels'], {'opt': self.job_opt.as_dict()})

    def test_recorded_levels_survive_a_restart_file_round_trip_as_independent_dicts(self):
        """Test that the recorded levels round-trip through YAML, hold no aliases, and share nothing with the level"""
        level = Level(method='wb97xd', basis='def2-tzvp', solvation_method='smd', solvent='water',
                      args={'keyword': {'opt': 'tight'}}, solvation_scheme_level=Level(method='b3lyp'))
        self.sched.record_job_level(label='C2H6', job_key='opt', level=level)
        self.sched.record_job_level(label='C2H6', job_key='freq', level=level)
        path = os.path.join(self.project_directory, 'levels_restart.yml')
        save_yaml_file(path=path, content={'output': self.sched.output})
        with open(path) as handle:
            self.assertNotIn('&id', handle.read())
        restored = read_yaml_file(path)['output']['C2H6']['levels']
        self.assertEqual(restored['opt']['solvation_method'], 'smd')
        self.assertNotIn('solvation_scheme_level', restored['opt'])
        self.sched.output['C2H6']['levels']['opt']['args']['keyword']['opt'] = 'loose'
        self.assertEqual(level.args['keyword']['opt'], 'tight')
        self.assertEqual(self.sched.output['C2H6']['levels']['freq']['args']['keyword']['opt'], 'tight')

    def test_a_none_level_removes_the_record(self):
        """Test that a job without a level does not leave a stale level next to a new log path"""
        self.sched.record_job_level(label='C2H6', job_key='opt', level=self.job_opt)
        self.sched.record_job_level(label='C2H6', job_key='opt', level=None)
        self.assertNotIn('opt', self.sched.output['C2H6']['levels'])

    def test_deleting_the_jobs_of_a_species_clears_its_recorded_levels(self):
        """Test that levels recorded for the logs of a discarded TS guess are not carried to the next guess"""
        self.sched.record_job_level(label='TS0', job_key='opt', level=self.job_opt)
        self.sched.record_job_level(label='TS0', job_key='irc', level=self.job_opt)
        self.sched.delete_all_species_jobs('TS0')
        self.assertEqual(self.sched.output['TS0']['levels'], dict())

    def test_deleting_the_jobs_of_a_ts_clears_its_irc_logs_and_their_levels_together(self):
        """Test that the per-log IRC levels are reset with the IRC logs, so a rerun stays in lockstep"""
        def spawn():
            with patch('arc.scheduler.parser.parse_geometry', return_value=self.ts_xyz), \
                    patch('arc.scheduler.plotter'), \
                    patch.object(self.sched, 'run_job'):
                job = self.make_job(self.job_opt, 'irc', irc_direction='forward')
                job.species = [self.ts]
                self.sched.spawn_post_irc_jobs('TS0', job)
        spawn()
        paths = self.sched.output['TS0']['paths']
        self.assertEqual((len(paths['irc']), len(paths['irc_levels'])), (1, 1))
        self.sched.delete_all_species_jobs('TS0')
        self.assertEqual(self.sched.output['TS0']['paths']['irc_levels'], list())
        spawn()
        paths = self.sched.output['TS0']['paths']
        self.assertEqual((len(paths['irc']), len(paths['irc_levels'])), (1, 1))

    def test_a_restart_that_holds_information_keeps_its_levels_and_gains_the_key_for_new_species(self):
        """Test that initializing the output of a new species does not disturb the levels of a restored one"""
        self.sched.record_job_level(label='C2H6', job_key='opt', level=self.job_opt)
        new_spc = ARCSpecies(label='CH4', smiles='C')
        self.sched.species_list.append(new_spc)
        self.sched.species_dict['CH4'] = new_spc
        self.sched.initialize_output_dict(label='CH4')
        self.sched.initialize_output_dict()
        self.assertEqual(self.sched.output['C2H6']['levels'], {'opt': self.job_opt.as_dict()})
        self.assertEqual(self.sched.output['CH4']['levels'], dict())


class TestConformerProvenanceRecording(unittest.TestCase):
    """Tests that the scheduler records the level and the energy source of every conformer result."""

    LOG_0 = os.path.join(ARC_TESTING_PATH, 'methylamine_conformer_0.out')
    LOG_1 = os.path.join(ARC_TESTING_PATH, 'methylamine_conformer_1.out')

    def setUp(self):
        self.spc = ARCSpecies(label='methylamine', smiles='CN')
        xyz = self.spc.get_xyz()
        self.spc.conformers = [xyz, xyz]
        self.spc.conformer_energies = [1.5, 0.0]
        self.spc.conformer_levels = [None, None]
        self.spc.conformer_logs = [None, None]
        self.spc.conformer_energy_sources = [{'kind': 'force_field_kcal_mol', 'level': None}] * 2
        self.sched = MagicMock()
        self.sched.species_dict = {'methylamine': self.spc}
        self.opt_level = Level(repr={'method': 'b97-d3', 'basis': '6-311+g(d,p)'})
        self.sp_level = Level(repr={'method': 'wb97x-d3', 'basis': 'def2-tzvp'})

    def make_job(self, log, level, job_type='conf_opt', status='done'):
        job = MagicMock()
        job.local_path_to_output_file = log
        job.level = level
        job.job_type = job_type
        job.times_rerun = 1
        job.job_status = [status, {'status': status, 'keywords': list(), 'error': '', 'line': ''}]
        return job

    def parse(self, job, i):
        return Scheduler.parse_conformer(self.sched, job=job, label='methylamine', i=i)

    def test_a_converged_conf_opt_records_its_level_and_the_electronic_energy_source(self):
        """Test that the conf_opt job's level is stored as the geometry level and as the energy level"""
        self.assertFalse(self.parse(self.make_job(self.LOG_0, self.opt_level), 0))
        self.assertEqual(self.spc.conformer_levels, [self.opt_level.as_dict(), None])
        self.assertEqual(self.spc.conformer_energy_sources,
                         [{'kind': 'electronic_kj_mol', 'level': self.opt_level.as_dict()},
                          {'kind': 'force_field_kcal_mol', 'level': None}])
        self.assertAlmostEqual(self.spc.conformer_energies[0], -251596.4435088726, 5)

    def test_a_converged_conf_opt_records_the_log_the_geometry_came_from(self):
        """Test that the optimization log is recorded for the conformer it produced, and nothing else gets one"""
        self.parse(self.make_job(self.LOG_0, self.opt_level), 0)
        self.assertEqual(self.spc.conformer_logs, [self.LOG_0, None])
        self.parse(self.make_job(self.LOG_0, self.sp_level, job_type='conf_sp'), 0)
        self.assertEqual(self.spc.conformer_logs, [self.LOG_0, None])

    def test_a_failed_or_unparsed_conf_opt_records_no_log(self):
        """Test that a conformer that keeps its force-field geometry has no log"""
        self.parse(self.make_job(self.LOG_0, self.opt_level, status='errored'), 0)
        with patch('arc.scheduler.parser.parse_geometry', return_value=None):
            self.parse(self.make_job(self.LOG_0, self.opt_level), 1)
        self.assertEqual(self.spc.conformer_logs, [None, None])

    def test_a_conf_opt_whose_geometry_cannot_be_parsed_records_no_geometry_level(self):
        """Test that no geometry level is recorded for a conformer whose optimized geometry was not read"""
        with patch('arc.scheduler.parser.parse_geometry', return_value=None):
            self.parse(self.make_job(self.LOG_0, self.opt_level), 0)
        self.assertEqual(self.spc.conformer_levels, [None, None])

    def test_a_conf_opt_whose_geometry_cannot_be_parsed_keeps_the_force_field_conformer(self):
        """Test that the force-field geometry, its energy and its source are untouched when no geometry is read"""
        original = self.spc.conformers[0]
        with patch('arc.scheduler.parser.parse_geometry', return_value=None):
            self.parse(self.make_job(self.LOG_0, self.opt_level), 0)
        self.assertIs(self.spc.conformers[0], original)
        self.assertEqual(self.spc.conformer_energies, [1.5, 0.0])
        self.assertEqual(self.spc.conformer_energy_sources[0], {'kind': 'force_field_kcal_mol', 'level': None})

    def test_a_failed_conf_opt_keeps_the_force_field_record(self):
        """Test that a conformer whose optimization failed stays a null level with its force-field energy source"""
        self.assertFalse(self.parse(self.make_job(self.LOG_0, self.opt_level, status='errored'), 0))
        self.assertEqual(self.spc.conformer_levels, [None, None])
        self.assertEqual(self.spc.conformer_energies, [1.5, 0.0])
        self.assertEqual(self.spc.conformer_energy_sources[0], {'kind': 'force_field_kcal_mol', 'level': None})

    def test_a_conf_sp_records_the_energy_level_and_leaves_the_geometry_level(self):
        """Test that a conformer single point does not change the geometry's level but names the energy's level"""
        self.parse(self.make_job(self.LOG_0, self.opt_level), 0)
        self.parse(self.make_job(self.LOG_0, self.sp_level, job_type='conf_sp'), 0)
        self.assertEqual(self.spc.conformer_levels[0], self.opt_level.as_dict())
        self.assertEqual(self.spc.conformer_energy_sources[0],
                         {'kind': 'electronic_kj_mol', 'level': self.sp_level.as_dict()})

    def test_a_troubleshooting_rerun_at_another_level_replaces_the_recorded_level(self):
        """Test that the level of the last conformer job that produced the geometry is the one recorded"""
        self.parse(self.make_job(self.LOG_0, self.opt_level), 0)
        self.parse(self.make_job(self.LOG_0, self.sp_level), 0)
        self.assertEqual(self.spc.conformer_levels[0], self.sp_level.as_dict())

    def test_conformers_set_without_provenance_lists_are_padded(self):
        """Test that assigning conformers directly (as a restart or a test does) never mis-lengths the records"""
        self.spc.conformer_levels = list()
        self.spc.conformer_energy_sources = list()
        self.parse(self.make_job(self.LOG_1, self.opt_level), 1)
        self.assertEqual(self.spc.conformer_levels, [None, self.opt_level.as_dict()])
        self.assertEqual(len(self.spc.conformer_energy_sources), 2)
        self.assertIsNone(self.spc.conformer_energy_sources[0])

    def test_perturbed_conformers_for_a_negative_frequency_start_with_no_provenance(self):
        """Test that replacing the conformers wholesale resets the levels and the energy sources to nulls"""
        self.parse(self.make_job(self.LOG_0, self.opt_level), 0)
        xyz = self.spc.get_xyz()
        self.sched.trsh_ess_jobs = True
        self.sched.job_types = dict()
        self.sched.job_dict = {'methylamine': dict()}
        self.sched.conformer_opt_level = self.opt_level
        self.sched.output = {'methylamine': {'errors': '', 'warnings': ''}}
        with patch('arc.scheduler.trsh_negative_freq', return_value=([], [xyz, xyz, xyz], [], [])):
            Scheduler.troubleshoot_negative_freq(self.sched, label='methylamine', job=self.make_job(self.LOG_0, None))
        self.assertEqual(self.spc.conformer_energies, [None] * 3)
        self.assertEqual(self.spc.conformer_levels, [None] * 3)
        self.assertEqual(self.spc.conformer_energy_sources, [None] * 3)

    def test_the_records_survive_a_restart_and_older_restarts_load_as_nulls(self):
        """Test the as_dict/from_dict round trip, and a restart dictionary that predates the new keys"""
        self.parse(self.make_job(self.LOG_0, self.opt_level), 0)
        restored = ARCSpecies(species_dict=self.spc.as_dict())
        self.assertEqual(restored.conformer_levels, self.spc.conformer_levels)
        self.assertEqual(restored.conformer_energy_sources, self.spc.conformer_energy_sources)
        old = self.spc.as_dict()
        del old['conformer_levels']
        del old['conformer_energy_sources']
        old_restored = ARCSpecies(species_dict=old)
        self.assertEqual(old_restored.conformer_levels, [None, None])
        self.assertEqual(old_restored.conformer_energy_sources, [None, None])


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
