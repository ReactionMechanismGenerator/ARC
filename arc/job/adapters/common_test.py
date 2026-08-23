#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.common module
"""

import shutil
import tempfile
import unittest

from types import SimpleNamespace

import arc.job.adapters.common as common
from arc.job.adapters.gaussian import GaussianAdapter
from arc.job.adapters.molpro import MolproAdapter
from arc.level import Level
from arc.species import ARCSpecies


class TestJobCommon(unittest.TestCase):
    """
    Contains unit tests for the job.adapters.common module.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.project_directory = tempfile.mkdtemp(prefix='test_JobAdaptersCommon_')
        cls.addClassCleanup(shutil.rmtree, cls.project_directory, ignore_errors=True)
        cls.job_1 = GaussianAdapter(execution_type='incore',
                                    job_type='composite',
                                    level=Level(method='cbs-qb3-paraskevas'),
                                    project='test',
                                    project_directory=cls.project_directory,
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=1)],
                                    testing=True,
                                    args={'keyword': {'general': 'IOp(1/12=5,3/44=0)'}},
                                    )
        cls.job_2 = GaussianAdapter(execution_type='incore',
                                    job_type='scan',
                                    torsions=[[1, 2, 3, 4]],
                                    level=Level(method='wb97xd', basis='def2tzvp'),
                                    project='test',
                                    project_directory=cls.project_directory,
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=3)],
                                    testing=True,
                                    args={'keyword': {'general': 'IOp(1/12=5,3/44=0)'}},
                                    )
        cls.job_3 = GaussianAdapter(execution_type='incore',
                                    job_type='scan',
                                    torsions=[[1, 2, 3, 4]],
                                    level=Level(method='wb97xd', basis='def2tzvp'),
                                    project='test',
                                    project_directory=cls.project_directory,
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=1, number_of_radicals=2)],
                                    testing=True,
                                    args={'keyword': {'general': 'IOp(1/12=5,3/44=0)'}},
                                    )
        cls.job_multi = GaussianAdapter(execution_type='incore',
                                    job_type='scan',
                                    torsions=[[1, 2, 3, 4]],
                                    level=Level(method='wb97xd', basis='def2tzvp'),
                                    project='test',
                                    project_directory=cls.project_directory,
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=1, number_of_radicals=2, multi_species='mltspc1'),
                                            ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=1, number_of_radicals=1, multi_species='mltspc1')],
                                    testing=True,
                                    args={'keyword': {'general': 'IOp(1/12=5,3/44=0)'}},
                                    run_multi_species = True,
                                    )

    def test_is_restricted(self):
        """Test the is_restricted() function"""
        self.assertTrue(common.is_restricted(self.job_1))
        self.assertFalse(common.is_restricted(self.job_2))
        self.assertFalse(common.is_restricted(self.job_3))
        benchmark_list = [False, True]
        self.assertEqual(common.is_restricted(self.job_multi),benchmark_list)

    WATER_XYZ = """O      0.00000000    0.00000000    0.11815400
H      0.00000000    0.76336400   -0.47261500
H      0.00000000   -0.76336400   -0.47261500"""

    def _singlet_job(self, number_of_radicals=None, method='wb97xd', multiplicity=1, is_ts=False):
        """Build a Gaussian adapter whose species is restricted unless something else says otherwise."""
        species = ARCSpecies(label='spc1', xyz=self.WATER_XYZ, multiplicity=multiplicity,
                             number_of_radicals=number_of_radicals)
        species.is_ts = is_ts
        project_directory = tempfile.mkdtemp(prefix='arc_test_common_')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        return GaussianAdapter(execution_type='incore',
                               job_type='sp',
                               level=Level(method=method, basis='def2tzvp'),
                               project='test',
                               project_directory=project_directory,
                               species=[species],
                               testing=True,
                               )

    def test_a_derived_external_instability_makes_a_silent_ts_unrestricted(self):
        """Test that a measured external instability flips the reference when the user declared nothing"""
        job = self._singlet_job(is_ts=True)
        self.assertTrue(common.is_species_restricted(job))
        job.species[0].derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertFalse(common.is_species_restricted(job))

    def test_a_derived_external_instability_does_not_flip_a_well(self):
        """Test that a species that is not a TS keeps its reference under a measured external instability"""
        job = self._singlet_job()
        self.assertFalse(job.species[0].is_ts)
        job.species[0].derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertTrue(common.is_species_restricted(job))
        self.assertTrue(common.derived_reference_is_unrestricted(job.species[0]))
        self.assertFalse(common.adopted_reference_is_unrestricted(job.species[0]))

    def test_an_internal_instability_does_not_flip_the_reference(self):
        """Test that only an external instability of a restricted reference makes a species unrestricted"""
        job = self._singlet_job(is_ts=True)
        for verdict in [{'verdict': 'internal_instability', 'restricted': True},
                        {'verdict': 'stable', 'restricted': True},
                        {'verdict': 'unknown', 'restricted': None},
                        {'verdict': 'external_instability', 'restricted': None},
                        {'verdict': 'external_instability', 'restricted': False},
                        None,
                        ]:
            job.species[0].derived_stability_verdict = verdict
            self.assertTrue(common.is_species_restricted(job),
                            msg=f'{verdict} should not have made the species unrestricted')

    def test_a_spin_relaxation_is_told_from_another_external_one(self):
        """Test that only a relaxation to an unrestricted reference reports broken spin symmetry"""
        species = ARCSpecies(label='spc', xyz=self.WATER_XYZ, multiplicity=1)
        for relaxations, expected in [(['RHF -> UHF'], True),
                                      (['RKS -> UKS'], True),
                                      (['rhf -> uhf'], True),
                                      (['RHF->UHF'], True),
                                      (['RHF -> CRHF'], False),
                                      (['RHF -> CRHF', 'RHF -> UHF'], True),
                                      (['RHF -> CUHF'], False),
                                      (list(), None),
                                      (None, None),
                                      ]:
            species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True,
                                                 'relaxations': relaxations}
            self.assertEqual(common.derived_instability_breaks_spin_symmetry(species), expected,
                             msg=f'the relaxations {relaxations} were not read as {expected}')

    def test_a_verdict_naming_no_relaxation_reports_nothing(self):
        """Test that a verdict without relaxations is unknown rather than negative"""
        species = ARCSpecies(label='spc', xyz=self.WATER_XYZ, multiplicity=1)
        for verdict in [None, dict(), {'verdict': 'stable', 'restricted': True}, 'external_instability']:
            species.derived_stability_verdict = verdict
            self.assertIsNone(common.derived_instability_breaks_spin_symmetry(species),
                              msg=f'the verdict {verdict} reported a relaxation')
        self.assertIsNone(common.derived_instability_breaks_spin_symmetry(None))

    def test_a_declared_number_of_radicals_wins_over_a_contradicting_verdict(self):
        """Test that a declared closed-shell character is not overridden by a measured instability"""
        job = self._singlet_job(number_of_radicals=1, is_ts=True)
        job.species[0].derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertTrue(common.is_species_restricted(job))

    def test_a_declared_biradical_stays_unrestricted_under_a_stable_verdict(self):
        """Test that a declared biradical singlet is not made restricted by a stable verdict"""
        job = self._singlet_job(number_of_radicals=2)
        job.species[0].derived_stability_verdict = {'verdict': 'stable', 'restricted': True}
        self.assertFalse(common.is_species_restricted(job))

    def test_a_derived_verdict_is_read_off_the_species_that_was_passed(self):
        """Test that the per-species entry point consults the species it was given, not the job's first"""
        job = self._singlet_job()
        other = ARCSpecies(label='spc2', xyz=self.WATER_XYZ, multiplicity=1)
        other.is_ts = True
        other.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertTrue(common.is_species_restricted(job))
        self.assertFalse(common.is_species_restricted(job, other))

    def test_a_composite_level_ignores_a_derived_verdict(self):
        """Test that the composite early return still short-circuits every other consideration"""
        job = self._singlet_job(method='cbs-qb3', is_ts=True)
        job.species[0].derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertEqual(job.level.method_type, 'composite')
        self.assertTrue(common.is_species_restricted(job))

    def test_a_correlated_single_point_is_not_flipped_by_an_adopted_verdict(self):
        """Test that an adopted verdict decides no reference for a correlated wavefunction level"""
        for method in ['dlpno-ccsd(t)', 'ccsd(t)-f12', 'ccsd(t)', 'mp2']:
            job = self._singlet_job(method=method, is_ts=True)
            job.species[0].derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
            self.assertTrue(common.adopted_reference_is_unrestricted(job.species[0]))
            self.assertEqual(job.level.method_type, 'wavefunction')
            self.assertTrue(common.is_species_restricted(job),
                            msg=f'an adopted verdict flipped the reference of a {method} single point')

    def test_an_adopted_verdict_flips_the_levels_a_broken_symmetry_reference_describes(self):
        """Test that a DFT and a Hartree-Fock job of an adopted species run unrestricted"""
        for method in ['wb97xd', 'b3lyp', 'hf']:
            job = self._singlet_job(method=method, is_ts=True)
            job.species[0].derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
            self.assertFalse(common.is_species_restricted(job),
                             msg=f'an adopted verdict did not flip a {method} job')

    def test_a_correlated_level_keeping_its_reference_is_reported(self):
        """Test that a job keeping its reference under an adopted verdict says so"""
        job = self._singlet_job(method='dlpno-ccsd(t)', is_ts=True)
        job.species[0].derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        with self.assertLogs(logger='arc', level='INFO') as captured:
            restricted = common.is_species_restricted(job)
        self.assertTrue(restricted)
        message = ' '.join(captured.output)
        self.assertIn('spc1', message)
        self.assertIn('dlpno-ccsd(t)', message)
        self.assertIn('restricted reference', message)

    def test_the_level_decides_only_whether_a_measured_verdict_is_acted_on(self):
        """Test that a multiplicity and a declaration make a correlated level unrestricted as before"""
        triplet = self._singlet_job(method='dlpno-ccsd(t)', multiplicity=3)
        self.assertFalse(common.is_species_restricted(triplet))
        biradical = self._singlet_job(method='dlpno-ccsd(t)', number_of_radicals=2)
        self.assertFalse(common.is_species_restricted(biradical))

    def test_level_admits_a_broken_symmetry_reference(self):
        """Test which levels a measured wavefunction-stability verdict decides the reference of"""
        for method, expected in [('wb97xd', True), ('b3lyp', True), ('m06-hf', True), ('hf', True),
                                 ('rhf', True), ('uhf', True), ('rohf', True), ('ccsd(t)', False),
                                 ('dlpno-ccsd(t)', False), ('ccsd(t)-f12', False), ('mp2', False),
                                 ('casscf', False), ('cbs-qb3', False), ('am1', False), ('gfn2-xtb', False)]:
            level = Level(method=method, software='gaussian')
            self.assertEqual(common.level_admits_a_broken_symmetry_reference(level), expected,
                             msg=f'{method} was not read as {expected}')
        self.assertFalse(common.level_admits_a_broken_symmetry_reference(None))
        for method_type in common.REFERENCE_AGNOSTIC_METHOD_TYPES:
            self.assertNotIn(method_type, common.BROKEN_SYMMETRY_METHOD_TYPES)

    def test_a_double_hybrid_admits_no_broken_symmetry_reference(self):
        """Test that a double hybrid is refused although ARC types it as DFT"""
        for method in ['b2plyp', 'b2plypd3', 'b2plyp-d3', 'mpw2plyp', 'pbe0dh', 'pbeqidh', 'dsdpbep86',
                       'DSD-PBEP86', 'DSD-PBEPBE-D3', 'wb97x-2', 'xyg3', 'pwpb95']:
            level = Level(method=method, software='gaussian')
            self.assertEqual(level.method_type, 'dft', msg=f'{method} was not typed as DFT')
            self.assertFalse(common.level_admits_a_broken_symmetry_reference(level),
                             msg=f'the double hybrid {method} was admitted')
        for method in ['b3lyp', 'wb97xd', 'wb97x-d', 'm062x', 'wb97m-v', 'b97-d3', 'pbe0']:
            level = Level(method=method, software='gaussian')
            self.assertTrue(common.level_admits_a_broken_symmetry_reference(level),
                            msg=f'the functional {method} was refused')

    def test_hf_3c_admits_a_broken_symmetry_reference(self):
        """Test that HF-3c is read as Hartree-Fock, its corrections not entering the wavefunction"""
        for method in ['hf-3c', 'HF-3c', 'hf3c']:
            level = Level(method=method, software='orca')
            self.assertTrue(common.level_admits_a_broken_symmetry_reference(level),
                            msg=f'{method} was refused')

    def test_derived_reference_is_unrestricted(self):
        """Test that only an external instability of a restricted reference reads as unrestricted"""
        species = ARCSpecies(label='spc1', xyz=self.WATER_XYZ, multiplicity=1)
        self.assertFalse(common.derived_reference_is_unrestricted(species))
        self.assertFalse(common.derived_reference_is_unrestricted(None))
        species.derived_stability_verdict = 'external_instability'
        self.assertFalse(common.derived_reference_is_unrestricted(species))
        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertTrue(common.derived_reference_is_unrestricted(species))
        species.derived_stability_verdict = {'verdict': 'internal_instability', 'restricted': True}
        self.assertFalse(common.derived_reference_is_unrestricted(species))

    def test_adopted_reference_is_unrestricted(self):
        """Test that a verdict is acted on for a transition state and reported only for anything else"""
        species = ARCSpecies(label='spc1', xyz=self.WATER_XYZ, multiplicity=1)
        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertFalse(common.adopted_reference_is_unrestricted(species))
        species.is_ts = True
        self.assertTrue(common.adopted_reference_is_unrestricted(species))
        species.derived_stability_verdict = {'verdict': 'internal_instability', 'restricted': True}
        self.assertFalse(common.adopted_reference_is_unrestricted(species))
        self.assertFalse(common.adopted_reference_is_unrestricted(None))

    def test_a_verdict_no_ess_can_reach_is_not_one_arc_acts_on(self):
        """Test that a verdict stamped as unreachable is reported and decides nothing"""
        species = ARCSpecies(label='spc1', xyz=self.WATER_XYZ, multiplicity=1)
        species.is_ts = True
        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertTrue(common.adopted_reference_is_unrestricted(species))
        species.derived_stability_verdict[common.REFERENCE_CHANGE_AVAILABLE_KEY] = True
        self.assertTrue(common.adopted_reference_is_unrestricted(species))
        species.derived_stability_verdict[common.REFERENCE_CHANGE_AVAILABLE_KEY] = False
        self.assertFalse(common.adopted_reference_is_unrestricted(species))
        self.assertTrue(common.derived_reference_is_unrestricted(species))
        self.assertIsNone(common.open_shell_character_source(species))

    def test_a_relaxation_that_is_not_the_spin_one_is_not_acted_on(self):
        """Test that an instability relaxing the reality of the orbitals decides no reference"""
        species = ARCSpecies(label='spc1', xyz=self.WATER_XYZ, multiplicity=1)
        species.is_ts = True
        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True,
                                             'relaxations': ['RHF -> CRHF']}
        self.assertIs(common.derived_instability_breaks_spin_symmetry(species), False)
        self.assertFalse(common.adopted_reference_is_unrestricted(species))
        self.assertTrue(common.derived_reference_is_unrestricted(species))
        species.derived_stability_verdict['relaxations'] = ['RHF -> CRHF', 'RHF -> UHF']
        self.assertTrue(common.adopted_reference_is_unrestricted(species))
        species.derived_stability_verdict['relaxations'] = []
        self.assertIsNone(common.derived_instability_breaks_spin_symmetry(species))
        self.assertTrue(common.adopted_reference_is_unrestricted(species))

    def test_species_may_read_previous_orbitals(self):
        """Test that a species holding no orbitals of the reference it adopted reads none"""
        species = ARCSpecies(label='spc1', xyz=self.WATER_XYZ, multiplicity=1)
        self.assertTrue(common.species_may_read_previous_orbitals(species))
        species.is_ts = True
        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertFalse(common.species_may_read_previous_orbitals(species))
        species.checkfile = 'a path the species holds'
        self.assertTrue(common.species_may_read_previous_orbitals(species))
        species.checkfile = None
        species.derived_stability_verdict = {'verdict': 'stable', 'restricted': True}
        self.assertTrue(common.species_may_read_previous_orbitals(species))

    def test_a_declared_number_of_radicals_blocks_the_adoption_of_a_ts_verdict(self):
        """Test that a declaration of any value stops a TS verdict from being one ARC acts on"""
        species = ARCSpecies(label='spc1', xyz=self.WATER_XYZ, multiplicity=1)
        species.is_ts = True
        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertTrue(common.adopted_reference_is_unrestricted(species))
        for number_of_radicals in [0, 1, 2, 3]:
            species.number_of_radicals = number_of_radicals
            self.assertFalse(common.adopted_reference_is_unrestricted(species),
                             msg=f'number_of_radicals = {number_of_radicals} did not block the adoption')
        species.number_of_radicals = None
        self.assertTrue(common.adopted_reference_is_unrestricted(species))

    def test_the_reference_memo_round_trips_through_a_restart(self):
        """Test that a job's SCF reference is persisted rather than recomputed on restore"""
        job = self._singlet_job(is_ts=True)
        self.assertTrue(common.is_restricted(job))
        job_dict = job.as_dict()
        self.assertIs(job_dict['restricted_used'], True)
        job.species[0].derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertFalse(common.is_restricted(job))
        self.assertIs(job.as_dict()['restricted_used'], False)
        piped = SimpleNamespace(level=Level(method='wb97xd', basis='def2tzvp'))
        self.assertIsNone(common.job_scf_reference_is_restricted(piped))

    def test_an_unadopted_well_verdict_is_credited_to_no_source(self):
        """Test that a verdict ARC reports without acting on it is not named as the deciding source"""
        species = ARCSpecies(label='spc1', xyz=self.WATER_XYZ, multiplicity=1)
        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertIsNone(common.open_shell_character_source(species))
        species.is_ts = True
        self.assertEqual(common.open_shell_character_source(species), 'derived')

    def test_job_scf_reference_is_restricted_reads_the_memo(self):
        """Test that the reference reported is the one the job's own memo holds"""
        job = self._singlet_job()
        self.assertTrue(common.is_restricted(job))
        self.assertIs(common.job_scf_reference_is_restricted(job), True)
        triplet = self._singlet_job(multiplicity=3)
        common.is_restricted(triplet)
        self.assertIs(common.job_scf_reference_is_restricted(triplet), False)
        piped = SimpleNamespace(level=Level(method='wb97xd', basis='def2tzvp'))
        self.assertIsNone(common.job_scf_reference_is_restricted(piped))

    def test_job_scf_reference_is_restricted_declines_a_reference_agnostic_level(self):
        """Test that a level ARC writes no r/u prefix for reports no reference"""
        for method in ['cbs-qb3', 'am1', 'mmff94s']:
            job = self._singlet_job(method=method)
            common.is_restricted(job)
            self.assertIn(job.level.method_type, ['force_field', 'composite', 'semiempirical'])
            self.assertIsNone(common.job_scf_reference_is_restricted(job),
                              msg=f'{method} reported a reference')

    def test_job_scf_reference_is_restricted_declines_a_multi_species_memo(self):
        """Test that a per-species memo is not read as a single decision"""
        self.assertEqual(common.is_restricted(self.job_multi), [False, True])
        self.assertIsNone(common.job_scf_reference_is_restricted(self.job_multi))

    def test_open_shell_character_source(self):
        """Test that only a declaration that attributes open-shell character is named as the source"""
        species = ARCSpecies(label='spc1', xyz=self.WATER_XYZ, multiplicity=1)
        species.is_ts = True
        self.assertIsNone(common.open_shell_character_source(species))
        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertEqual(common.open_shell_character_source(species), 'derived')
        species.number_of_radicals = 2
        self.assertEqual(common.open_shell_character_source(species), 'declared')
        species.derived_stability_verdict = None
        self.assertEqual(common.open_shell_character_source(species), 'declared')

    def test_a_declaration_that_attributes_no_open_shell_character_is_not_a_source(self):
        """Test that a declared 0 or 1 blocks the verdict without being credited with the decision"""
        species = ARCSpecies(label='spc1', xyz=self.WATER_XYZ, multiplicity=1)
        species.is_ts = True
        species.derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        for number_of_radicals in [0, 1]:
            species.number_of_radicals = number_of_radicals
            self.assertIsNone(common.open_shell_character_source(species),
                              msg=f'number_of_radicals = {number_of_radicals} was named as the source')
            self.assertFalse(common.adopted_reference_is_unrestricted(species),
                             msg=f'number_of_radicals = {number_of_radicals} did not block the adoption')
            self.assertTrue(common.derived_reference_is_unrestricted(species))

    def test_is_restricted_memoizes_the_decision_it_made(self):
        """Test that the reference a job's input declared stays readable off the job afterwards"""
        job = self._singlet_job(is_ts=True)
        self.assertTrue(common.is_restricted(job))
        self.assertIs(job.restricted_used, True)
        job.species[0].derived_stability_verdict = {'verdict': 'external_instability', 'restricted': True}
        self.assertIs(job.restricted_used, True)
        self.assertFalse(common.is_restricted(job))
        self.assertIs(job.restricted_used, False)
        self.assertEqual(common.is_restricted(self.job_multi), [False, True])
        self.assertEqual(self.job_multi.restricted_used, [False, True])

    def test_reference_agnostic_method_types(self):
        """Test the method types whose reference ARC does not prefix"""
        self.assertEqual(common.REFERENCE_AGNOSTIC_METHOD_TYPES, ['force_field', 'composite', 'semiempirical'])

    def test_check_argument_consistency(self):
        """Test the check_argument_consistency() function"""
        common.check_argument_consistency(self.job_1)
        common.check_argument_consistency(self.job_2)
        with self.assertRaises(NotImplementedError):
            MolproAdapter(execution_type='incore',
                          job_type='irc',
                          level=Level(method='ccsd(t)', basis='cc-pvtz'),
                          project='test',
                          project_directory=self.project_directory,
                          species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=1)],
                          testing=True,
                          )
        with self.assertRaises(ValueError):
            GaussianAdapter(execution_type='incore',
                            job_type='irc',
                            level=Level(method='b3lyp', basis='def2svp'),
                            project='test',
                            project_directory=self.project_directory,
                            species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=1)],
                            testing=True,
                            args={'keyword': {'general': 'IOp(1/12=5,3/44=0)'}},
                            )
        with self.assertRaises(NotImplementedError):
            spc = ARCSpecies(label='ethane', smiles='CC')
            spc.determine_rotors()
            spc.rotors_dict['directed_scan_type'] = 'ess'
            MolproAdapter(execution_type='incore',
                          job_type='scan',
                          torsions=[[1, 2, 3, 4]],
                          level=Level(method='ccsd(t)', basis='cc-pvtz'),
                          project='test',
                          project_directory=self.project_directory,
                          species=[spc],
                          testing=True,
                          )
        self.job_2.scan_res = 55.6
        with self.assertRaises(ValueError):
            common.check_argument_consistency(self.job_2)
        # A resolution coarser than 20 degrees that still divides 360 (e.g. from a poor
        # ~/.arc/settings.py rotor_scan_resolution) is caught at the job-build chokepoint.
        self.job_2.scan_res = 30.0
        with self.assertRaises(ValueError):
            common.check_argument_consistency(self.job_2)

    def test_update_input_dict_with_args(self):
        """Test the update_input_dict_with_args() function"""
        input_dict = common.update_input_dict_with_args(args={}, input_dict={})
        self.assertEqual(input_dict, dict())

        input_dict = common.update_input_dict_with_args(args={'block': {'1': """a\nb"""}},
                                                        input_dict={'block': ''})
        self.assertEqual(input_dict, {'block': """\n\na\nb"""})

        input_dict = common.update_input_dict_with_args(args={'block': {'1': """a\nb"""}},
                                                        input_dict={'block': """x\ny\n"""})
        self.assertEqual(input_dict, {'block': """x\ny\na\nb"""})

        input_dict = common.update_input_dict_with_args(args={'keyword': {'scan_trsh': 'keyword 1'}},
                                                        input_dict={})
        self.assertEqual(input_dict, {'scan_trsh': 'keyword 1'})

        input_dict = common.update_input_dict_with_args(args={'keyword': {'opt': 'keyword 2'}},
                                                        input_dict={})
        self.assertEqual(input_dict, {'keywords': 'keyword 2'})

    def test_update_input_dict_with_multiple_args(self):
        """Test the update_input_dict_with_args() function with multiple arguments in args."""
        args = {
            'block': {'1': 'block text 1'},
            'keyword': {'opt': 'keyword opt'},
            'trsh': ['trsh value 1']
        }
        input_dict = {'block': 'existing block', 'keywords': 'existing keywords', 'trsh': 'existing trsh'}
        expected_dict = {
            'block': 'existing block\nblock text 1',
            'keywords': 'existing keywords keyword opt',
            'trsh': 'existing trsh trsh value 1'
        }

        result_dict = common.update_input_dict_with_args(args=args, input_dict=input_dict)
        self.assertEqual(result_dict, expected_dict)

    def test_set_job_args(self):
        """Test the set_job_args() function"""
        args = common.set_job_args(args=None, level=Level(repr='CBS-QB3'), job_name='j1')
        self.assertEqual(args, {'keyword': dict(), 'block': dict(), 'trsh': dict()})

        args = common.set_job_args(args=dict(), level=Level(repr='CBS-QB3'), job_name='j1')
        self.assertEqual(args, {'keyword': dict(), 'block': dict(), 'trsh': dict()})

        args = common.set_job_args(args={'keyword': 'k1'}, level=Level(repr='CBS-QB3'), job_name='j1')
        self.assertEqual(args, {'keyword':'k1', 'block': dict(), 'trsh': dict()})

    def test_get_dropped_level_args(self):
        """Test the get_dropped_level_args() function"""
        level_args = {'keyword': {'opt': 'opt=(verytight)', 'general': 'scf=(qc)'}, 'block': dict()}

        self.assertEqual(common.get_dropped_level_args(args=dict(), level_args=dict()), dict())
        self.assertEqual(common.get_dropped_level_args(args=dict(), level_args=level_args),
                         {'keyword': {'opt': 'opt=(verytight)', 'general': 'scf=(qc)'}})
        self.assertEqual(common.get_dropped_level_args(args={'keyword': {'opt': 'opt=(verytight)'}},
                                                       level_args=level_args),
                         {'keyword': {'general': 'scf=(qc)'}})
        self.assertEqual(common.get_dropped_level_args(args={'keyword': {'opt': 'opt=(verytight)',
                                                                        'general': 'scf=(qc)'}},
                                                       level_args=level_args),
                         dict())
        self.assertEqual(common.get_dropped_level_args(args={'keyword': {'opt': 'opt=(tight)'}},
                                                       level_args=level_args),
                         {'keyword': {'opt': 'opt=(verytight)', 'general': 'scf=(qc)'}})
        self.assertEqual(common.get_dropped_level_args(args={'keyword': 'k1'}, level_args=level_args),
                         {'keyword': {'opt': 'opt=(verytight)', 'general': 'scf=(qc)'}})
        self.assertEqual(common.get_dropped_level_args(args={'keyword': None}, level_args=level_args),
                         {'keyword': {'opt': 'opt=(verytight)', 'general': 'scf=(qc)'}})

    def test_set_job_args_warns_only_for_dropped_options(self):
        """Test that set_job_args() warns only for level options which the job args do not carry."""
        level = Level(method='wb97xd', basis='def2tzvp', args={'keyword': {'opt': 'opt=(verytight)'}})

        applied_args = {'keyword': {'opt': 'opt=(verytight)'}, 'block': dict()}
        with self.assertNoLogs(logger='arc', level='WARNING'):
            args = common.set_job_args(args=applied_args, level=level, job_name='j1')
        self.assertEqual(args['keyword']['opt'], 'opt=(verytight)')

        dropping_args = {'keyword': dict(), 'block': dict(), 'trsh': {'trsh': 'scf=(qc)'}}
        with self.assertLogs(logger='arc', level='WARNING') as captured:
            common.set_job_args(args=dropping_args, level=level, job_name='j1')
        message = ''.join(captured.output)
        self.assertIn('opt=(verytight)', message)
        self.assertIn('j1', message)
        self.assertNotIn('troubleshooting', message)

    def test_set_job_args_with_a_non_dict_args_entry(self):
        """Test that set_job_args() tolerates a job args entry which is not a dictionary."""
        level = Level(method='wb97xd', basis='def2tzvp', args={'keyword': {'opt': 'opt=(verytight)'}})
        with self.assertLogs(logger='arc', level='WARNING') as captured:
            args = common.set_job_args(args={'keyword': 'k1'}, level=level, job_name='j1')
        self.assertEqual(args, {'keyword': 'k1', 'block': dict(), 'trsh': dict()})
        self.assertIn('opt=(verytight)', ''.join(captured.output))

    def test_set_job_args_adopts_level_args_when_job_args_carry_no_options(self):
        """Test that set_job_args() adopts the level args when the job args carry no options."""
        level = Level(method='wb97xd', basis='def2tzvp', args={'keyword': {'opt': 'opt=(verytight)'}})
        args = common.set_job_args(args={'keyword': dict(), 'block': dict()}, level=level, job_name='j1')
        self.assertEqual(args, {'keyword': {'opt': 'opt=(verytight)'}, 'block': dict(), 'trsh': dict()})
        args['keyword']['dft_grid'] = 'defgrid2'
        self.assertNotIn('dft_grid', level.args['keyword'])

    def test_set_job_args_does_not_alias_level_args(self):
        """Test that the returned job args do not alias the level's args dictionaries."""
        level = Level(method='wb97xd', basis='def2tzvp', args={'keyword': {'opt': 'opt=(verytight)'}})
        args = common.set_job_args(args=None, level=level, job_name='j1')
        args['keyword']['dft_grid'] = 'defgrid2'
        self.assertNotIn('dft_grid', level.args['keyword'])
        self.assertEqual(level.args['keyword'], {'opt': 'opt=(verytight)'})

    def test_which(self):
        """Test the which() function"""
        ans = common.which(command='python', return_bool=True, raise_error=False)
        self.assertTrue(ans)

        ans = common.which(command='python', return_bool=False, raise_error=False)
        self.assertIn('arc_env/bin/python', ans)

        ans = common.which(command='ls', return_bool=True, raise_error=False)
        self.assertTrue(ans)

        ans = common.which(command='fake_command_1', return_bool=True, raise_error=False)
        self.assertFalse(ans)

        ans = common.which(command=['python'], return_bool=False, raise_error=False)
        self.assertIn('bin/python', ans)
    
    def test_combine_parameters(self):
        """Test the combine_parameters function for normal input."""
        input_dict = {'param1': 'value1 term1 value2', 'param2': 'another value term2'}
        terms = ['term1', 'term2']
        expected_dict = {'param1': 'value1  value2', 'param2': 'another value '}
        expected_parameters = ['term1', 'term2']

        modified_dict, parameters = common.combine_parameters(input_dict, terms)
        
        self.assertEqual(modified_dict, expected_dict)
        self.assertEqual(parameters, expected_parameters)

    def test_combine_parameters_empty_input(self):
        """Test the combine_parameters function with empty input."""
        input_dict = {}
        terms = ['term1', 'term2']
        expected_dict = {}
        expected_parameters = []

        modified_dict, parameters = common.combine_parameters(input_dict, terms)
        
        self.assertEqual(modified_dict, expected_dict)
        self.assertEqual(parameters, expected_parameters)

    def test_combine_parameters_no_match(self):
        """Test the combine_parameters function with no matching terms."""
        input_dict = {'param1': 'value1 value2', 'param2': 'another value'}
        terms = ['nonexistent']
        expected_dict = {'param1': 'value1 value2', 'param2': 'another value'}
        expected_parameters = []

        modified_dict, parameters = common.combine_parameters(input_dict, terms)
        
        self.assertEqual(modified_dict, expected_dict)
        self.assertEqual(parameters, expected_parameters)

    def test_combine_parameters_multiple_occurrences(self):
        """Test the combine_parameters function with multiple occurrences of the same term."""
        input_dict = {'param1': 'value1 term1 value2 term1', 'param2': 'another term2 value term2'}
        terms = ['term1', 'term2']
        expected_dict = {'param1': 'value1  value2 ', 'param2': 'another  value '}
        expected_parameters = ['term1', 'term2']

        modified_dict, parameters = common.combine_parameters(input_dict, terms)

        self.assertEqual(modified_dict, expected_dict)
        self.assertEqual(sorted(parameters), expected_parameters)

    def test_combine_parameters_overlapping_terms(self):
        """Test the combine_parameters function with overlapping terms."""
        input_dict = {'param1': 'value term1 term123', 'param2': 'another term2 value'}
        terms = ['term1', 'term123', 'term2']
        expected_dict = {'param1': 'value  ', 'param2': 'another  value'}
        expected_parameters = ['term1', 'term123', 'term2']

        modified_dict, parameters = common.combine_parameters(input_dict, terms)

        self.assertEqual(modified_dict, expected_dict)
        self.assertEqual(sorted(parameters), expected_parameters)

    def test_input_dict_strip(self):
        """Test the input_dict_strip() function"""
        input_dict = {
            'key1': '  value1  ',
            'key2': ' value2  ',
            'key3': '\nvalue3\n',
            'key4': ' value4\n',
            'key5': None,
            'key6': 42,  # Not a string, should remain unchanged
            'key7': '\nvalue7 '
        }

        expected_stripped_dict = {
            'key1': 'value1',
            'key2': 'value2',
            'key3': '\nvalue3\n',
            'key4': 'value4\n',
            'key6': 42,  # Should not be stripped
            'key7': '\nvalue7'
        }

        stripped_dict = common.input_dict_strip(input_dict)
        self.assertEqual(stripped_dict, expected_stripped_dict)


class TestResolveJobServer(unittest.TestCase):
    """The one answer to "which server will this job go to", shared by the adapters and the pipe."""

    def test_the_first_server_listed_is_the_one_used(self):
        """ARC submits to the first entry, so anything else would describe a different job."""
        self.assertEqual(common.resolve_job_server({'gaussian': ['zeus', 'atlas']}, 'gaussian'),
                         'zeus')

    def test_a_bare_string_is_read_as_a_single_server(self):
        self.assertEqual(common.resolve_job_server({'gaussian': 'zeus'}, 'gaussian'), 'zeus')

    def test_an_unlisted_adapter_resolves_to_nothing(self):
        self.assertIsNone(common.resolve_job_server({'gaussian': ['zeus']}, 'orca'))

    def test_an_empty_server_list_resolves_to_nothing(self):
        """The open-coded version indexed [0] here and raised IndexError."""
        self.assertIsNone(common.resolve_job_server({'gaussian': []}, 'gaussian'))

    def test_empty_ess_settings_resolve_to_nothing(self):
        self.assertIsNone(common.resolve_job_server(dict(), 'gaussian'))
        self.assertIsNone(common.resolve_job_server(None, 'gaussian'))

    def test_a_troubleshooting_override_wins(self):
        """The override exists to move a job off the server that failed it."""
        self.assertEqual(common.resolve_job_server({'gaussian': ['zeus']}, 'gaussian',
                                                   args={'trsh': {'server': 'atlas'}}),
                         'atlas')

    def test_an_override_is_honoured_even_when_it_is_empty(self):
        """The caller must see the same server the job itself would be given."""
        self.assertIsNone(common.resolve_job_server({'gaussian': ['zeus']}, 'gaussian',
                                                    args={'trsh': {'server': None}}))

    def test_trsh_args_without_a_server_fall_through(self):
        self.assertEqual(common.resolve_job_server({'gaussian': ['zeus']}, 'gaussian',
                                                   args={'trsh': {'scan_res': 8}}),
                         'zeus')

    def test_args_without_a_trsh_entry_fall_through(self):
        self.assertEqual(common.resolve_job_server({'gaussian': ['zeus']}, 'gaussian',
                                                   args={'keyword': {}}),
                         'zeus')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
