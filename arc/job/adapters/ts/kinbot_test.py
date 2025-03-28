#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.autotst module
"""

import os
import shutil
import unittest

from rmgpy.species import Species

from arc.common import ARC_PATH
from arc.job.adapters.ts.kinbot_ts import KinBotAdapter, HAS_KINBOT
from arc.reaction import ARCReaction
from arc.species import ARCSpecies


class TestKinBotAdapter(unittest.TestCase):
    """
    Contains unit tests for the AutoTSTAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None

    # @pytest.mark.skip(reason="KinBot support in ARC has been deprecated")
    def test_intra_h_migration(self):
        """Test KinBot for intra H migration reactions"""
        if HAS_KINBOT:
            rxn1 = ARCReaction(reactants=['CC[O]'], products=['[CH2]CO'],
                               r_species=[ARCSpecies(label='CC[O]', smiles='CC[O]')],
                               p_species=[ARCSpecies(label='[CH2]CO', smiles='[CH2]CO')])
            self.assertEqual(rxn1.family, 'intra_H_migration')
            kinbot1 = KinBotAdapter(job_type='tsg',
                                    reactions=[rxn1],
                                    testing=True,
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_KinBot', 'tst1'),
                                    )
            kinbot1.execute_incore()
            self.assertTrue(rxn1.ts_species.is_ts)
            self.assertEqual(rxn1.ts_species.charge, 0)
            self.assertEqual(rxn1.ts_species.multiplicity, 2)
            self.assertEqual(len(rxn1.ts_species.ts_guesses), 2)
            self.assertEqual(rxn1.ts_species.ts_guesses[0].initial_xyz['symbols'],
                             ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H'))
            self.assertEqual(rxn1.ts_species.ts_guesses[1].initial_xyz['symbols'],
                             ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H'))
            self.assertEqual(len(rxn1.ts_species.ts_guesses[1].initial_xyz['coords']), 8)
            self.assertEqual(rxn1.ts_species.ts_guesses[0].method, 'kinbot')
            self.assertEqual(rxn1.ts_species.ts_guesses[1].method, 'kinbot')
            self.assertEqual(rxn1.ts_species.ts_guesses[0].method_index, 0)
            self.assertEqual(rxn1.ts_species.ts_guesses[1].method_index, 1)
            self.assertEqual(rxn1.ts_species.ts_guesses[0].method_direction, 'F')
            self.assertEqual(rxn1.ts_species.ts_guesses[1].method_direction, 'R')
            self.assertLess(rxn1.ts_species.ts_guesses[0].execution_time.seconds, 300)  # 0:00:00.003082
            self.assertTrue(rxn1.ts_species.ts_guesses[0].success)
            self.assertTrue(rxn1.ts_species.ts_guesses[1].success)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'test_KinBot'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
