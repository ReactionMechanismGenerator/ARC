#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.processor module
"""

import os
import shutil
import unittest

import arc.processor as processor
from arc.checks.common import TS_IRC_FAILED_MARKER
from arc.common import ARC_TESTING_PATH, read_yaml_file
from arc.reaction import ARCReaction
from arc.species import ARCSpecies
from arc.species.conformers import embed_rdkit, read_rdkit_embedded_conformer_i
from arc.species.species import are_coords_compliant_with_graph


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
        output_directory = os.path.join(ARC_TESTING_PATH, 'process_kinetics')
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


    @staticmethod
    def _spc_with_xyz(label, smiles):
        """Build an ARCSpecies whose xyz follows the atom order of its SMILES-built mol."""
        spc = ARCSpecies(label=label, smiles=smiles)
        rd_mol = embed_rdkit(label, spc.mol, num_confs=1)
        spc.final_xyz = read_rdkit_embedded_conformer_i(rd_mol, 0)
        return spc

    @staticmethod
    def _mol_state(spc):
        """The atom order and bonds of a species' mol, as indices."""
        atoms = spc.mol.atoms
        return ([atom.element.symbol for atom in atoms],
                sorted((atoms.index(a1), atoms.index(a2))
                       for a1 in atoms for a2 in a1.edges if atoms.index(a1) < atoms.index(a2)))

    def test_thermo_lib_has_isomorph_keeps_atom_order(self):
        """Test that _thermo_lib_has_isomorph() does not re-sort the atoms of the compared species' mols"""
        acetaldehyde = self._spc_with_xyz('acetaldehyde', 'CC=O')
        vinyl_alcohol = self._spc_with_xyz('vinyl_alcohol', 'C=CO')
        states = [self._mol_state(acetaldehyde), self._mol_state(vinyl_alcohol)]
        self.assertFalse(processor._thermo_lib_has_isomorph(vinyl_alcohol, [acetaldehyde]))
        self.assertEqual([self._mol_state(acetaldehyde), self._mol_state(vinyl_alcohol)], states)
        self.assertTrue(are_coords_compliant_with_graph(acetaldehyde.final_xyz, acetaldehyde.mol))
        self.assertTrue(are_coords_compliant_with_graph(vinyl_alcohol.final_xyz, vinyl_alcohol.mol))

    def test_thermo_lib_has_isomorph_isomorphic_keeps_atom_order(self):
        """Test that an isomorphic match is found without re-sorting the atoms of either mol"""
        first = self._spc_with_xyz('first', 'CC=O')
        second = self._spc_with_xyz('second', 'O=CC')
        states = [self._mol_state(first), self._mol_state(second)]
        self.assertTrue(processor._thermo_lib_has_isomorph(second, [first]))
        self.assertEqual([self._mol_state(first), self._mol_state(second)], states)
        self.assertTrue(are_coords_compliant_with_graph(first.final_xyz, first.mol))
        self.assertTrue(are_coords_compliant_with_graph(second.final_xyz, second.mol))

    def test_thermo_lib_has_isomorph_after_plain_isomorphism_check(self):
        """Test that a preceding plain is_isomorphic() call on the same mols does not break or reorder the check"""
        first = self._spc_with_xyz('first', 'CC=O')
        second = self._spc_with_xyz('second', 'O=CC')
        self.assertTrue(first.mol.is_isomorphic(second.mol))
        states = [self._mol_state(first), self._mol_state(second)]
        self.assertTrue(processor._thermo_lib_has_isomorph(second, [first]))
        self.assertEqual([self._mol_state(first), self._mol_state(second)], states)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        """
        directories = [os.path.join(ARC_TESTING_PATH, 'process_kinetics'),
                      ]
        for dir_path in directories:
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
