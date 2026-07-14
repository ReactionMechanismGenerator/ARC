#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.processor module
"""

import os
import shutil
import unittest
from unittest import mock

import arc.processor as processor
from arc.common import ARC_TESTING_PATH
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

    def test_compare_rates(self):
        """Test the compare_rates() method"""
        rxn_1 = ARCReaction(r_species=[self.ch4, self.h],
                            p_species=[ARCSpecies(label='CH3', smiles='[CH3]'),
                                       ARCSpecies(label='H2', smiles='[H][H]')],
                            kinetics={'A': 4.79e+05, 'n': 2.5, 'Ea': 40.12},
                            )
        rxn_2 = ARCReaction(r_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC')],
                            p_species=[ARCSpecies(label='iC3H7', smiles='C[CH]C')],
                            kinetics={'A': 7.18e5, 'n': 2.05, 'Ea': 151.88},
                            )
        output_directory = os.path.join(ARC_TESTING_PATH, 'process_kinetics')
        reactions_to_compare = processor.compare_rates(rxns_for_kinetics_lib=[rxn_1, rxn_2],
                                                       output_directory=output_directory,
                                                       )
        self.assertEqual(len(reactions_to_compare), 2)
        self.assertEqual(len(reactions_to_compare[0].rmg_kinetics), 3)
        self.assertEqual(len(reactions_to_compare[1].rmg_kinetics), 1)
        self.assertTrue(os.path.isfile(os.path.join(output_directory, 'rate_plots.pdf')))
        self.assertTrue(os.path.isfile(os.path.join(output_directory, 'RMG_kinetics.yml')))


    def test_compare_thermo_ignores_benign_rmg_stderr(self):
        """Benign RMG INFO/WARNING stderr chatter must not be logged as an error when thermo was computed."""
        benign_stderr = ['INFO:root:Loading thermodynamics library from primaryThermoLibrary.py ...',
                         'WARNING:root:Setting a default value for tolerance.',
                         '']
        computed_species = [{'label': 'CH4', 'adjlist': 'x', 'h298': -17.9, 's298': 44.5, 'comment': 'GAV'}]
        with mock.patch.object(processor, 'execute_command', return_value=([], benign_stderr)), \
                mock.patch.object(processor, 'save_yaml_file'), \
                mock.patch.object(processor, 'read_yaml_file', return_value=computed_species), \
                mock.patch.object(processor.plotter, 'draw_thermo_parity_plots'):
            with self.assertLogs(processor.logger, level='DEBUG') as cm:
                processor.compare_thermo(species_for_thermo_lib=[ARCSpecies(label='CH4', smiles='C')],
                                         output_directory=os.path.join(ARC_TESTING_PATH, 'process_thermo'))
        self.assertFalse(any('Error while running RMG thermo script' in msg for msg in cm.output))

    def test_compare_thermo_reports_real_error(self):
        """A genuine traceback on stderr (or a missing deliverable) must still be logged as an error."""
        real_stderr = ['INFO:root:Loading thermodynamics library ...',
                       'Traceback (most recent call last):',
                       'RuntimeError: RMG database failed to load']
        # deliverable was seeded but never populated with h298/s298 -> genuine failure
        seeded_species = [{'label': 'CH4', 'adjlist': 'x'}]
        with mock.patch.object(processor, 'execute_command', return_value=([], real_stderr)), \
                mock.patch.object(processor, 'save_yaml_file'), \
                mock.patch.object(processor, 'read_yaml_file', return_value=seeded_species), \
                mock.patch.object(processor.plotter, 'draw_thermo_parity_plots'):
            with self.assertLogs(processor.logger, level='ERROR') as cm:
                processor.compare_thermo(species_for_thermo_lib=[ARCSpecies(label='CH4', smiles='C')],
                                         output_directory=os.path.join(ARC_TESTING_PATH, 'process_thermo'))
        self.assertTrue(any('Error while running RMG thermo script' in msg for msg in cm.output))

    def test_compare_thermo_reports_missing_deliverable(self):
        """Benign-only stderr but an unpopulated deliverable (no h298/s298) must still log an error."""
        benign_stderr = ['INFO:root:Loading thermodynamics library ...',
                         'WARNING:root:Setting a default value for tolerance.']
        seeded_species = [{'label': 'CH4', 'adjlist': 'x'}]  # pre-seeded, never populated -> script failed
        with mock.patch.object(processor, 'execute_command', return_value=([], benign_stderr)), \
                mock.patch.object(processor, 'save_yaml_file'), \
                mock.patch.object(processor, 'read_yaml_file', return_value=seeded_species), \
                mock.patch.object(processor.plotter, 'draw_thermo_parity_plots'):
            with self.assertLogs(processor.logger, level='ERROR') as cm:
                processor.compare_thermo(species_for_thermo_lib=[ARCSpecies(label='CH4', smiles='C')],
                                         output_directory=os.path.join(ARC_TESTING_PATH, 'process_thermo'))
        self.assertTrue(any('Error while running RMG thermo script' in msg for msg in cm.output))

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        """
        directories = [os.path.join(ARC_TESTING_PATH, 'process_kinetics'),
                       os.path.join(ARC_TESTING_PATH, 'process_thermo'),
                      ]
        for dir_path in directories:
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
