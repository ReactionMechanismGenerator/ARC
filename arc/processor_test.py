#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.processor module
"""

import os
import shutil
import tempfile
import unittest

import arc.processor as processor
from arc.checks.common import TS_IRC_FAILED_MARKER
from arc.common import read_yaml_file
from arc.reaction import ARCReaction
from arc.species import ARCSpecies


class TestProcessor(unittest.TestCase):
    """
    Contains unit tests for the Processor class
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.scratch_dir = tempfile.mkdtemp(prefix='arc_test_processor_')
        cls.addClassCleanup(shutil.rmtree, cls.scratch_dir, ignore_errors=True)
        cls.ch4 = ARCSpecies(label='CH4', smiles='C')
        cls.nh3 = ARCSpecies(label='NH3', smiles='N')
        cls.h = ARCSpecies(label='H', smiles='[H]')
        cls.ch4_bde_1_2_a = ARCSpecies(label='CH4_BDE_1_2_A', smiles='[CH3]')
        cls.ch4.e0, cls.h.e0, cls.ch4_bde_1_2_a.e0 = 10, 25, 35
        cls.ch4.bdes = [(1, 2)]

    def test_process_bdes(self):
        """Test the process_bdes() method"""
        bde_report = processor.process_bdes(label='CH4',
                                            species_dict={'CH4': self.ch4,
                                                          'NH3': self.nh3,
                                                          'H': self.h,
                                                          'CH4_BDE_1_2_A': self.ch4_bde_1_2_a})
        self.assertEqual(bde_report, {(1, 2): 50})

    def test_compare_rates(self):
        """Test the compare_rates() method"""
        rxn_1 = ARCReaction(r_species=[self.ch4, self.h],
                            p_species=[ARCSpecies(label='CH3', smiles='[CH3]'),
                                       ARCSpecies(label='H2', smiles='[H][H]')],
                            kinetics={'A': 4.79e+05, 'n': 2.5, 'Ea': 40.12},
                            )
        rxn_1.ts_species = ARCSpecies(label='TS1', is_ts=True)
        rxn_1.ts_species.ts_checks['IRC'] = False
        rxn_2 = ARCReaction(r_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC')],
                            p_species=[ARCSpecies(label='iC3H7', smiles='C[CH]C')],
                            kinetics={'A': 7.18e5, 'n': 2.05, 'Ea': 151.88},
                            )
        rxn_2.ts_species = ARCSpecies(label='TS2', is_ts=True)
        output_directory = os.path.join(self.scratch_dir, 'process_kinetics')
        reactions_to_compare = processor.compare_rates(rxns_for_kinetics_lib=[rxn_1, rxn_2],
                                                       output_directory=output_directory,
                                                       )
        self.assertEqual(len(reactions_to_compare), 2)
        # Library matches plus every family estimate are reported: both the training-set
        # depository hits and the rate-rule estimate (the old 'training' comment filter used
        # to drop them, and the rule tree that produces the rate-rule estimate was not built).
        self.assertEqual(len(reactions_to_compare[0].rmg_kinetics), 5)
        self.assertEqual(len(reactions_to_compare[1].rmg_kinetics), 3)
        comments_0 = [entry['comment'] for entry in reactions_to_compare[0].rmg_kinetics]
        self.assertTrue(any(comment.startswith('Library:') for comment in comments_0))
        self.assertTrue(any('source: rate rules' in comment for comment in comments_0))
        self.assertTrue(any('training' in comment for comment in comments_0))
        self.assertTrue(os.path.isfile(os.path.join(output_directory, 'rate_plots.pdf')))
        kinetics_yml_path = os.path.join(output_directory, 'RMG_kinetics.yml')
        self.assertTrue(os.path.isfile(kinetics_yml_path))
        content = read_yaml_file(path=kinetics_yml_path)
        self.assertIn(TS_IRC_FAILED_MARKER, content[0]['ts_validation'])
        self.assertNotIn('ts_validation', content[1])

    def test_thermo_lib_has_isomorph_keeps_atom_order(self):
        """Test that _thermo_lib_has_isomorph() does not re-sort the atoms of the compared mols"""
        first, second = ARCSpecies(label='first', smiles='CC=O'), ARCSpecies(label='second', smiles='C=CO')
        before = [list(spc.mol.atoms) for spc in (first, second)]
        self.assertFalse(processor._thermo_lib_has_isomorph(second, [first]))
        self.assertEqual([list(spc.mol.atoms) for spc in (first, second)], before)

    def test_thermo_lib_has_isomorph_after_plain_isomorphism_check(self):
        """Test that _thermo_lib_has_isomorph() works after a plain is_isomorphic() call on the same mols"""
        first, second = ARCSpecies(label='first', smiles='CC=O'), ARCSpecies(label='second', smiles='C=CO')
        self.assertFalse(first.mol.is_isomorphic(second.mol))
        self.assertFalse(processor._thermo_lib_has_isomorph(second, [first]))

    def test_process_arc_project_removes_a_stale_kinetics_library(self):
        """A kinetics library left by a previous session is removed when no rate coefficient survives"""
        project_directory = tempfile.mkdtemp(prefix='arc_test_stale_kinetics_lib_')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        lib_path = os.path.join(project_directory, 'output', 'RMG libraries', 'kinetics')
        os.makedirs(lib_path)
        with open(os.path.join(lib_path, 'reactions.py'), 'w') as f:
            f.write('# a rate computed by a previous session\n')
        ch3, h2 = ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='H2', smiles='[H][H]')
        rxn = ARCReaction(r_species=[self.ch4, self.h], p_species=[ch3, h2])
        rxn.ts_label = 'TS0'
        output_dict = {label: {'convergence': label != 'TS0'} for label in ['CH4', 'H', 'CH3', 'H2', 'TS0']}
        processor.process_arc_project(thermo_adapter='arkane', kinetics_adapter='arkane', project='stale',
                                      project_directory=project_directory,
                                      species_dict={spc.label: spc for spc in [self.ch4, self.h, ch3, h2]},
                                      reactions=[rxn], output_dict=output_dict, compute_thermo=False,
                                      compare_to_rmg=False)
        self.assertFalse(os.path.isdir(lib_path))

    def test_process_arc_project_logs_a_stale_kinetics_library_it_cannot_remove(self):
        """A stale kinetics library that cannot be removed is reported, not dropped silently"""
        project_directory = tempfile.mkdtemp(prefix='arc_test_stuck_kinetics_lib_')
        lib_path = os.path.join(project_directory, 'output', 'RMG libraries', 'kinetics')
        locked = os.path.join(lib_path, 'locked')
        os.makedirs(locked)
        with open(os.path.join(locked, 'reactions.py'), 'w') as f:
            f.write('# a rate computed by a previous session\n')
        os.chmod(locked, 0o500)
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        self.addCleanup(os.chmod, locked, 0o700)
        ch3, h2 = ARCSpecies(label='CH3', smiles='[CH3]'), ARCSpecies(label='H2', smiles='[H][H]')
        rxn = ARCReaction(r_species=[self.ch4, self.h], p_species=[ch3, h2])
        rxn.ts_label = 'TS0'
        output_dict = {label: {'convergence': label != 'TS0'} for label in ['CH4', 'H', 'CH3', 'H2', 'TS0']}
        with self.assertLogs('arc', level='ERROR') as logs:
            processor.process_arc_project(thermo_adapter='arkane', kinetics_adapter='arkane', project='stuck',
                                          project_directory=project_directory,
                                          species_dict={spc.label: spc for spc in [self.ch4, self.h, ch3, h2]},
                                          reactions=[rxn], output_dict=output_dict, compute_thermo=False,
                                          compare_to_rmg=False)
        self.assertTrue(any('Could not remove the stale kinetics library' in line for line in logs.output))


class TestNotVerifiedGate(unittest.TestCase):
    """
    Contains unit tests for the not-verified gate in front of the RMG libraries.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.oh = ARCSpecies(label='OH', smiles='[OH]')
        cls.h2 = ARCSpecies(label='H2', smiles='[H][H]')
        cls.h2o = ARCSpecies(label='H2O', smiles='O')
        cls.h = ARCSpecies(label='H', smiles='[H]')
        cls.rxn = ARCReaction(r_species=[cls.oh, cls.h2], p_species=[cls.h2o, cls.h])
        cls.rxn.ts_label = 'TS0'
        cls.cause = 'Rotor scan of torsion [1, 2, 3, 4] of OH still finds a conformer 3.20 kJ/mol lower.'

    def output_dict(self, flagged: str | None = None) -> dict:
        """
        Make an output dictionary in which one label, if any, is not verified.

        Args:
            flagged (str, optional): The label whose conformer_verified is False.

        Returns:
            dict: The output dictionary.
        """
        output = {label: {'conformer_verified': True, 'conformer_verified_cause': ''}
                  for label in ['OH', 'H2', 'H2O', 'H', 'TS0']}
        if flagged is not None:
            output[flagged] = {'conformer_verified': False, 'conformer_verified_cause': self.cause}
        return output

    def test_not_verified_causes(self):
        """not_verified_causes() returns a cause only for a False conformer_verified."""
        self.assertEqual(processor.not_verified_causes('X', dict()), list())
        self.assertEqual(processor.not_verified_causes('OH', self.output_dict()), list())
        self.assertEqual(processor.not_verified_causes('OH', {'OH': {'conformer_verified': None}}), list())
        self.assertEqual(processor.not_verified_causes('OH', self.output_dict('OH')), [self.cause])
        self.assertEqual(processor.not_verified_causes('OH', {'OH': {'conformer_verified': False}}),
                         ['OH is not verified to be at the lowest conformer found'])

    def test_gate_leaves_out_a_flagged_species_by_default(self):
        """A not verified species is left out of the thermo library by default."""
        kept, comments = processor.gate_not_verified_library_entries(items=[self.oh, self.h2o],
                                                                     output_dict=self.output_dict('OH'))
        self.assertEqual(kept, [self.h2o])
        self.assertEqual(comments, dict())

    def test_gate_includes_a_flagged_species_with_opt_in(self):
        """With the opt-in, a not verified species is kept and its comment states the cause and gap."""
        kept, comments = processor.gate_not_verified_library_entries(items=[self.oh, self.h2o],
                                                                     output_dict=self.output_dict('OH'),
                                                                     allow_not_verified=True)
        self.assertEqual(kept, [self.oh, self.h2o])
        self.assertEqual(list(comments.keys()), ['OH'])
        self.assertTrue(comments['OH'].startswith('NOT VERIFIED: '))
        self.assertIn('3.20 kJ/mol lower', comments['OH'])

    def test_gate_leaves_out_a_reaction_with_a_flagged_ts(self):
        """A reaction whose TS is not verified is left out of the kinetics library."""
        kept, _ = processor.gate_not_verified_library_entries(items=[self.rxn], output_dict=self.output_dict('TS0'))
        self.assertEqual(kept, list())
        kept, _ = processor.gate_not_verified_library_entries(items=[self.rxn], output_dict=self.output_dict())
        self.assertEqual(kept, [self.rxn])

    def test_gate_leaves_out_a_reaction_with_a_flagged_product(self):
        """A reaction whose reactant or product is not verified is left out, and kept with the opt-in."""
        for label in ['OH', 'H']:
            kept, _ = processor.gate_not_verified_library_entries(items=[self.rxn],
                                                                  output_dict=self.output_dict(label))
            self.assertEqual(kept, list())
        kept, comments = processor.gate_not_verified_library_entries(items=[self.rxn],
                                                                     output_dict=self.output_dict('H'),
                                                                     allow_not_verified=True)
        self.assertEqual(kept, [self.rxn])
        self.assertIn('3.20 kJ/mol lower', comments[self.rxn.label])

    def test_gate_leaves_out_an_unjudged_conformer_change(self):
        """A species that ends the run with an unjudged conformer change is left out, and so are its reactions."""
        output = self.output_dict()
        output['OH'] = {'conformer_verified': None, 'conformer_verified_cause': '', 'conformer_change_pending': 1}
        self.assertEqual(processor.not_verified_causes('OH', output), ['OH: conformer change not judged'])
        oh = ARCSpecies(label='OH', smiles='[OH]')
        oh.thermo.data = 'NASA()'
        selected, comments = processor.select_thermo_lib_species(species=[oh], output_dict=output)
        self.assertEqual(selected, list())
        self.assertEqual(comments, dict())
        kept, _ = processor.gate_not_verified_library_entries(items=[self.rxn], output_dict=output)
        self.assertEqual(kept, list())

    def test_gate_includes_an_unjudged_conformer_change_with_opt_in(self):
        """With the opt-in, a species with an unjudged conformer change is kept, and its comment states why."""
        output = self.output_dict()
        output['OH'] = {'conformer_verified': None, 'conformer_verified_cause': '', 'conformer_change_pending': 2}
        oh = ARCSpecies(label='OH', smiles='[OH]')
        oh.thermo.data = 'NASA()'
        selected, comments = processor.select_thermo_lib_species(species=[oh], output_dict=output,
                                                                 allow_not_verified=True)
        self.assertEqual(selected, [oh])
        self.assertEqual(comments['OH'], 'NOT VERIFIED: OH: conformer change not judged')
        kept, comments = processor.gate_not_verified_library_entries(items=[self.rxn], output_dict=output,
                                                                     allow_not_verified=True)
        self.assertEqual(kept, [self.rxn])
        self.assertIn('conformer change not judged', comments[self.rxn.label])

    def test_gate_passes_an_unverified_species_with_no_pending_change(self):
        """conformer_verified None with no pending conformer change still passes the gate."""
        output = self.output_dict()
        for pending in [dict(), {'conformer_change_pending': 0}]:
            output['OH'] = {'conformer_verified': None, 'conformer_verified_cause': '', **pending}
            self.assertEqual(processor.not_verified_causes('OH', output), list())
            oh = ARCSpecies(label='OH', smiles='[OH]')
            oh.thermo.data = 'NASA()'
            selected, _ = processor.select_thermo_lib_species(species=[oh], output_dict=output)
            self.assertEqual(selected, [oh])

    def test_select_thermo_lib_species_prefers_the_verified_isomorph(self):
        """A not verified species never displaces an isomorphic verified one from the thermo library."""
        oh_1, oh_2 = ARCSpecies(label='OH_1', smiles='[OH]'), ARCSpecies(label='OH_2', smiles='[OH]')
        for spc in [oh_1, oh_2, self.h2o]:
            spc.thermo.data = 'NASA()'
        output = {'OH_1': {'conformer_verified': False, 'conformer_verified_cause': self.cause},
                  'OH_2': {'conformer_verified': True}}
        selected, comments = processor.select_thermo_lib_species(species=[oh_1, oh_2], output_dict=output)
        self.assertEqual(selected, [oh_2])
        selected, comments = processor.select_thermo_lib_species(species=[oh_1, oh_2], output_dict=output,
                                                                 allow_not_verified=True)
        self.assertEqual(selected, [oh_2])
        self.assertEqual(comments, dict())
        selected, comments = processor.select_thermo_lib_species(species=[oh_1, self.h2o], output_dict=output,
                                                                 allow_not_verified=True)
        self.assertEqual(selected, [self.h2o, oh_1])
        self.assertIn('OH_1', comments)
        oh_2.include_in_thermo_lib = False
        selected, _ = processor.select_thermo_lib_species(species=[oh_1, oh_2], output_dict=output,
                                                          allow_not_verified=True)
        self.assertEqual(selected, [oh_1])


    def test_remove_gated_out_library(self):
        """A library left from an earlier run is removed only when this run's entries were all gated out."""
        libraries_path = tempfile.mkdtemp(prefix='arc_test_gated_out_libraries_')
        lib_path = os.path.join(libraries_path, 'thermo')
        self.addCleanup(shutil.rmtree, libraries_path, ignore_errors=True)
        processor.remove_gated_out_library(libraries_path=libraries_path, library='thermo', gated_out=True)
        self.assertFalse(os.path.isdir(lib_path))
        os.makedirs(lib_path)
        processor.remove_gated_out_library(libraries_path=libraries_path, library='thermo', gated_out=False)
        self.assertTrue(os.path.isdir(lib_path))
        processor.remove_gated_out_library(libraries_path=libraries_path, library='kinetics', gated_out=True)
        self.assertTrue(os.path.isdir(lib_path))
        processor.remove_gated_out_library(libraries_path=libraries_path, library='thermo', gated_out=True)
        self.assertFalse(os.path.isdir(lib_path))

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
