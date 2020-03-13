#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for TS guess generation methods
"""

import unittest

from rmgpy.reaction import Reaction
from rmgpy.species import Species


class TestAutoTST(unittest.TestCase):
    """
    Contains unit tests for AutoTST
    """

    @classmethod
    def setUpClass(cls):
        """
        A function run ONCE before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.spc1 = Species().from_smiles('C')
        cls.spc2 = Species().from_smiles('[OH]')
        cls.spc3 = Species().from_smiles('[CH3]')
        cls.spc4 = Species().from_smiles('O')
        cls.spc5 = Species().from_smiles('CCCC')
        cls.spc6 = Species().from_smiles('O[O]')
        cls.spc7 = Species().from_smiles('[CH2]CCC')
        cls.spc8 = Species().from_smiles('OO')
        cls.spc9 = Species().from_smiles('CC[O]')
        cls.spc10 = Species().from_smiles('[NH2]')
        cls.spc11 = Species().from_smiles('[CH2]C[O]')
        cls.spc12 = Species().from_smiles('N')
        cls.reaction1 = Reaction(reactants=[cls.spc1, cls.spc2], products=[cls.spc3, cls.spc4])
        cls.reaction2 = Reaction(reactants=[cls.spc5, cls.spc6], products=[cls.spc7, cls.spc8])
        cls.reaction3 = Reaction(reactants=[cls.spc9, cls.spc10], products=[cls.spc11, cls.spc12])

    def test_generate_reaction_string(self):
        """Test the generate_reaction_string() function"""
        reaction_label1 = atst.get_reaction_label(rmg_reaction=self.reaction1)
        reaction_label2 = atst.get_reaction_label(rmg_reaction=self.reaction2)
        reaction_label3 = atst.get_reaction_label(rmg_reaction=self.reaction3)
        self.assertEqual(reaction_label1, 'C+[OH]_[CH3]+O')
        self.assertEqual(reaction_label2, 'CCCC+[O]O_[CH2]CCC+OO')
        self.assertEqual(reaction_label3, 'CC[O]+[NH2]_[CH2]C[O]+N')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
