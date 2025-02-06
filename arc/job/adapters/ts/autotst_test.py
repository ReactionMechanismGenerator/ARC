#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.autotst module
"""

import os
import shutil
import unittest

from arc.common import ARC_PATH
from arc.job.adapters.ts.autotst_ts import AutoTSTAdapter, HAS_AUTOTST
from arc.reaction import ARCReaction
from arc.species import ARCSpecies


class TestAutoTSTAdapter(unittest.TestCase):
    """
    Contains unit tests for the AutoTSTAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None

    def test_has_autotst(self):
        """Test that AutoTST was successfully imported"""
        self.assertTrue(HAS_AUTOTST)

    @unittest.skip("This test is deprecated and will be removed in future versions.")
    def test_autotst_h_abstraction(self):
        """Test AutoTST for H Abstraction reactions"""
        rxn1 = ARCReaction(reactants=['CCC', 'HO2'], products=['C3H7', 'H2O2'],
                           r_species=[ARCSpecies(label='CCC', smiles='CCC'),
                                      ARCSpecies(label='HO2', smiles='O[O]')],
                           p_species=[ARCSpecies(label='C3H7', smiles='[CH2]CC'),
                                      ARCSpecies(label='H2O2', smiles='OO')])
        self.assertEqual(rxn1.family, 'H_Abstraction')
        atst1 = AutoTSTAdapter(job_type='tsg',
                               reactions=[rxn1],
                               testing=True,
                               project='test',
                               project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_AutoTST', 'tst1'),
                               )
        atst1.execute_incore()
        self.assertTrue(rxn1.ts_species.is_ts)
        self.assertEqual(rxn1.ts_species.charge, 0)
        self.assertEqual(rxn1.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn1.ts_species.ts_guesses), 4)
        self.assertEqual(rxn1.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('O', 'O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'))
        self.assertEqual(rxn1.ts_species.ts_guesses[1].initial_xyz['symbols'],
                         ('O', 'O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'))
        self.assertEqual(len(rxn1.ts_species.ts_guesses[1].initial_xyz['coords']), 14)
        self.assertTrue(rxn1.ts_species.ts_guesses[0].success)
        self.assertTrue(rxn1.ts_species.ts_guesses[1].success)
        self.assertTrue(rxn1.ts_species.ts_guesses[2].success)
        self.assertTrue(rxn1.ts_species.ts_guesses[3].success)

        rxn2 = ARCReaction(reactants=['CCCOH', 'OH'], products=['CCCO', 'H2O'],
                           r_species=[ARCSpecies(label='CCCO', smiles='CCCO'),
                                      ARCSpecies(label='OH', smiles='[OH]')],
                           p_species=[ARCSpecies(label='CCC[O]', smiles='CCC[O]'),
                                      ARCSpecies(label='H2O', smiles='O')])
        self.assertEqual(rxn2.family, 'H_Abstraction')
        atst2 = AutoTSTAdapter(job_type='tsg',
                               reactions=[rxn2],
                               testing=True,
                               project='test',
                               project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_AutoTST', 'ts2'),
                               )
        atst2.execute_incore()
        self.assertTrue(rxn2.ts_species.is_ts)
        self.assertEqual(rxn2.ts_species.charge, 0)
        self.assertEqual(rxn2.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn2.ts_species.ts_guesses), 4)
        self.assertEqual(rxn2.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('O', 'O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'))
        self.assertEqual(len(rxn2.ts_species.ts_guesses[1].initial_xyz['coords']), 14)

        rxn3 = ARCReaction(reactants=['C=COH', 'H'], products=['C=CO', 'H2'],
                           r_species=[ARCSpecies(label='C=CO', smiles='C=CO'),
                                      ARCSpecies(label='H', smiles='[H]')],
                           p_species=[ARCSpecies(label='C=C[O]', smiles='C=C[O]'),
                                      ARCSpecies(label='H2', smiles='[H][H]')])
        self.assertEqual(rxn3.family, 'H_Abstraction')
        atst3 = AutoTSTAdapter(job_type='tsg',
                               reactions=[rxn3],
                               testing=True,
                               project='test',
                               project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_AutoTST', 'tst3'),
                               )
        atst3.execute_incore()
        self.assertTrue(rxn3.ts_species.is_ts)
        self.assertEqual(rxn3.ts_species.charge, 0)
        self.assertEqual(rxn3.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn3.ts_species.ts_guesses), 4)
        self.assertEqual(rxn3.ts_species.ts_guesses[0].initial_xyz['symbols'], ('O', 'C', 'C', 'H', 'H', 'H', 'H', 'H'))
        self.assertEqual(len(rxn3.ts_species.ts_guesses[0].initial_xyz['coords']), 8)
        self.assertEqual(len(rxn3.ts_species.ts_guesses[1].initial_xyz['coords']), 8)
        self.assertEqual(len(rxn3.ts_species.ts_guesses[2].initial_xyz['coords']), 8)
        self.assertEqual(len(rxn3.ts_species.ts_guesses[3].initial_xyz['coords']), 8)
    @unittest.skip("This test is deprecated and will be removed in future versions.")
    def test_autotst_intra_h_migration(self):
        """Test AutoTST for intra-H migration reactions"""
        rxn1 = ARCReaction(reactants=['[CH2]CO'], products=['CC[O]'],
                           r_species=[ARCSpecies(label='[CH2]CO', smiles='[CH2]CO')],
                           p_species=[ARCSpecies(label='CC[O]', smiles='CC[O]')])
        self.assertEqual(rxn1.family, 'intra_H_migration')
        atst1 = AutoTSTAdapter(job_type='tsg',
                               reactions=[rxn1],
                               testing=True,
                               project='test',
                               project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_AutoTST', 'tst4'),
                               )
        atst1.execute_incore()
        self.assertEqual(rxn1.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn1.ts_species.ts_guesses), 4)
        self.assertEqual(rxn1.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('O', 'C', 'C', 'H', 'H', 'H', 'H', 'H'))
        self.assertEqual(rxn1.ts_species.ts_guesses[1].initial_xyz['symbols'],
                         ('O', 'C', 'C', 'H', 'H', 'H', 'H', 'H'))
        self.assertEqual(len(rxn1.ts_species.ts_guesses[1].initial_xyz['coords']), 8)
        self.assertEqual(rxn1.ts_species.ts_guesses[0].method, 'autotst')
        self.assertEqual(rxn1.ts_species.ts_guesses[1].method, 'autotst')
        self.assertEqual(rxn1.ts_species.ts_guesses[2].method, 'autotst')
        self.assertEqual(rxn1.ts_species.ts_guesses[3].method, 'autotst')
        self.assertEqual(rxn1.ts_species.ts_guesses[0].method_index, 0)
        self.assertEqual(rxn1.ts_species.ts_guesses[1].method_index, 1)
        self.assertEqual(rxn1.ts_species.ts_guesses[2].method_index, 2)
        self.assertEqual(rxn1.ts_species.ts_guesses[3].method_index, 3)
        self.assertEqual(rxn1.ts_species.ts_guesses[0].method_direction, 'F')
        self.assertEqual(rxn1.ts_species.ts_guesses[1].method_direction, 'F')
        self.assertEqual(rxn1.ts_species.ts_guesses[2].method_direction, 'R')
        self.assertEqual(rxn1.ts_species.ts_guesses[3].method_direction, 'R')
        self.assertLess(rxn1.ts_species.ts_guesses[3].execution_time.seconds, 300)  # 0:00:13.143187
        self.assertTrue(rxn1.ts_species.ts_guesses[0].success)
        self.assertTrue(rxn1.ts_species.ts_guesses[1].success)
        self.assertTrue(rxn1.ts_species.ts_guesses[2].success)
        self.assertTrue(rxn1.ts_species.ts_guesses[3].success)
    @unittest.skip("This test is deprecated and will be removed in future versions.")
    def test_autotst_r_addition_multiple_bond(self):
        """Test AutoTST for R addition multiple bond reactions"""
        rxn1 = ARCReaction(reactants=['C#C', '[OH]'], products=['[CH]=CO'],
                           r_species=[ARCSpecies(label='C#C', smiles='C#C'),
                                      ARCSpecies(label='[OH]', smiles='[OH]')],
                           p_species=[ARCSpecies(label='[CH]=CO', smiles='[CH]=CO')])
        self.assertEqual(rxn1.family, 'R_Addition_MultipleBond')
        atst1 = AutoTSTAdapter(job_type='tsg',
                               reactions=[rxn1],
                               testing=True,
                               project='test',
                               project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_AutoTST', 'tst4'),
                               )
        atst1.execute_incore()
        self.assertEqual(rxn1.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn1.ts_species.ts_guesses), 2)
        self.assertEqual(rxn1.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('O', 'C', 'C', 'H', 'H', 'H'))
        self.assertEqual(rxn1.ts_species.ts_guesses[1].initial_xyz['symbols'],
                         ('O', 'C', 'C', 'H', 'H', 'H'))
        self.assertEqual(len(rxn1.ts_species.ts_guesses[1].initial_xyz['coords']), 6)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'test_AutoTST'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
