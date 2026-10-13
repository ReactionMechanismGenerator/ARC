#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.processor module
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

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

    def test_process_arc_project_returns_the_aec_digests_the_adapters_recorded(self):
        """The digests of data/AEC.yml that the statmech adapters recorded are returned, distinct and sorted."""
        project_directory = tempfile.mkdtemp(prefix='test_processor_aec_digest_')
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        spc = ARCSpecies(label='CH4', smiles='C')
        e0_only = ARCSpecies(label='H', smiles='[H]')
        e0_only.e0_only = True
        output_dict = {'CH4': {'convergence': True, 'job_types': dict()},
                       'H': {'convergence': True, 'job_types': dict()}}
        reaction = MagicMock(label='CH4 <=> CH4', ts_label='TS0', r_species=[spc], p_species=[spc], kinetics=None)
        output_dict['TS0'] = {'convergence': True, 'job_types': dict()}
        for compute_rates, expected in ((False, ['a' * 64, 'b' * 64]), (True, ['a' * 64, 'b' * 64, 'c' * 64])):
            with self.subTest(compute_rates=compute_rates):
                adapters = [MagicMock(aec_yml_sha256s={'b' * 64}), MagicMock(aec_yml_sha256s={'a' * 64})]
                if compute_rates:
                    adapters.insert(0, MagicMock(aec_yml_sha256s={'c' * 64}))
                with patch('arc.processor.statmech_factory', side_effect=adapters), \
                        patch('arc.processor.plotter'), patch('arc.processor.write_unconverged_log'), \
                        patch('arc.processor.clean_output_directory'):
                    recorded = processor.process_arc_project(
                        thermo_adapter='arkane', kinetics_adapter='arkane', project='p',
                        project_directory=project_directory, species_dict={'CH4': spc, 'H': e0_only},
                        reactions=[reaction], output_dict=output_dict, compute_thermo=True,
                        compute_rates=compute_rates, compare_to_rmg=False)
                self.assertEqual(recorded, expected)
        with patch('arc.processor.statmech_factory', side_effect=[MagicMock(spec=['compute_thermo'])]), \
                patch('arc.processor.plotter'), patch('arc.processor.write_unconverged_log'), \
                patch('arc.processor.clean_output_directory'):
            self.assertEqual(processor.process_arc_project(
                thermo_adapter='other', kinetics_adapter='other', project='p', project_directory=project_directory,
                species_dict={'CH4': spc}, reactions=[], output_dict=output_dict,
                compute_thermo=True, compute_rates=False, compare_to_rmg=False), list())

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
