#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for ARC's statmech.arkane modele
"""

import os
import shutil
import unittest

import arc.common as common
from arc.level import Level
from arc.reaction import ARCReaction, ARCSpecies
from arc.statmech.arkane import ArkaneAdapter


class TestArkaneAdapter(unittest.TestCase):
    """
    Contains unit tests for ArkaneAdapter.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        path_1 = os.path.join(common.ARC_PATH, 'arc', 'testing', 'statmech', 'arkane_1')
        os.makedirs(path_1)
        rxn_1 = ARCReaction(r_species=[ARCSpecies(label='CH3NH', smiles='C[NH]')],
                            p_species=[ARCSpecies(label='CH2NH2', smiles='[CH2]N')])
        rxn_1.ts_species = ARCSpecies(label='TS1', is_ts=True, xyz="""C      -0.68121000   -0.03232800    0.00786900
                                                                      H      -1.26057500    0.83953400   -0.27338400
                                                                      H      -1.14918300   -1.01052100   -0.06991000
                                                                      H       0.10325500    0.27373300    0.96444200
                                                                      N       0.74713700    0.12694800   -0.09842700
                                                                      H       1.16150600   -0.79827600    0.01973500""")
        cls.arkane_1 = ArkaneAdapter(output_directory=path_1, output_dict=dict(), bac_type=None, reaction=rxn_1)
        cls.arkane_2 = ArkaneAdapter(output_directory=path_1, output_dict=dict(), bac_type=None, species=rxn_1.r_species[0])

    def test_arkane_has_en_corr(self):
        """Test the arkane_has_en_corr() function"""
        self.arkane_1.sp_level = Level('CBS-QB3')
        self.assertTrue(self.arkane_1.arkane_has_en_corr())

        # B3LYP/6-31G(d,p) does not support nitrogen AECs
        self.arkane_1.sp_level = Level(method='B3LYP', basis='6-31G(d,p)')
        self.assertFalse(self.arkane_1.arkane_has_en_corr())

        self.arkane_1.sp_level = Level(method='fake', basis='basis')
        self.assertFalse(self.arkane_1.arkane_has_en_corr())

        self.arkane_2.sp_level = Level('CBS-QB3')
        self.assertTrue(self.arkane_2.arkane_has_en_corr())

        self.arkane_2.sp_level = Level(method='fake', basis='basis')
        self.assertFalse(self.arkane_2.arkane_has_en_corr())

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        """
        shutil.rmtree(os.path.join(common.ARC_PATH, 'arc', 'testing', 'statmech'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
