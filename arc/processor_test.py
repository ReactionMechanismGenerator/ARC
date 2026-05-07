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

    def test_classify_species_for_thermo_skips_irc_endpoints(self):
        """IRC endpoint species and TSs must be skipped, not flagged as unconverged."""
        ch4 = ARCSpecies(label='CH4', smiles='C')
        nh3 = ARCSpecies(label='NH3', smiles='N')  # converged but compute_thermo unset → unconverged bucket
        nh3.compute_thermo = False
        ts1 = ARCSpecies(label='TS1', smiles='[CH3]', is_ts=True)
        irc_fwd = ARCSpecies(label='IRC_TS1_1', smiles='C', compute_thermo=False, irc_label='TS1')
        irc_rev = ARCSpecies(label='IRC_TS1_2', smiles='C', compute_thermo=False, irc_label='TS1')
        species_dict = {'CH4': ch4, 'NH3': nh3, 'TS1': ts1,
                        'IRC_TS1_1': irc_fwd, 'IRC_TS1_2': irc_rev}
        # IRC labels intentionally absent from output_dict — the helper must not look them up.
        output_dict = {'CH4': {'convergence': True}, 'NH3': {'convergence': True},
                       'TS1': {'convergence': True}}
        converged, e0_only, unconverged = processor.classify_species_for_thermo(
            species_dict=species_dict, output_dict=output_dict)
        self.assertEqual([s.label for s in converged], ['CH4'])
        self.assertEqual(e0_only, [])
        self.assertEqual([s.label for s in unconverged], ['NH3'])

    def test_classify_species_for_thermo_e0_only_and_unconverged(self):
        """e0_only species route to e0 bucket; species with compute_thermo but no convergence go unconverged."""
        e0_spc = ARCSpecies(label='E0_only_spc', smiles='C')
        e0_spc.e0_only = True
        unconv = ARCSpecies(label='Unconv', smiles='N')
        species_dict = {'E0_only_spc': e0_spc, 'Unconv': unconv}
        output_dict = {'E0_only_spc': {'convergence': True}, 'Unconv': {'convergence': False}}
        converged, e0_only, unconverged = processor.classify_species_for_thermo(
            species_dict=species_dict, output_dict=output_dict)
        self.assertEqual(converged, [])
        self.assertEqual([s.label for s in e0_only], ['E0_only_spc'])
        self.assertEqual([s.label for s in unconverged], ['Unconv'])

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
