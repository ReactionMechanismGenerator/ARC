#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests for TS guess generation methods
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest

from rmgpy.species import Species
from rmgpy.reaction import Reaction

from arc.ts import atst

################################################################################


class TestAutoTST(unittest.TestCase):
    """
    Contains unit tests for AutoTST
    """

    @classmethod
    def setUpClass(cls):
        """
        A function run ONCE before all unit tests in this class.
        """
        cls.spc1 = Species().fromSMILES(str('C'))
        cls.spc2 = Species().fromSMILES(str('[OH]'))
        cls.spc3 = Species().fromSMILES(str('[CH3]'))
        cls.spc4 = Species().fromSMILES(str('O'))
        cls.spc5 = Species().fromSMILES(str('CCCC'))
        cls.spc6 = Species().fromSMILES(str('O[O]'))
        cls.spc7 = Species().fromSMILES(str('[CH2]CCC'))
        cls.spc8 = Species().fromSMILES(str('OO'))
        cls.spc9 = Species().fromSMILES(str('CC[O]'))
        cls.spc10 = Species().fromSMILES(str('[NH2]'))
        cls.spc11 = Species().fromSMILES(str('[CH2]C[O]'))
        cls.spc12 = Species().fromSMILES(str('N'))
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

    def test_run_autotsts(self):
        """Test running AutoTST"""
        self.maxDiff = None
        reaction_label1 = atst.get_reaction_label(rmg_reaction=self.reaction1)
        xyz1a = atst.autotst(reaction_label=reaction_label1, reaction_family='H_Abstraction')
        xyz1b = atst.autotst(rmg_reaction=self.reaction1, reaction_family='H_Abstraction')
        xyz2 = atst.autotst(rmg_reaction=self.reaction2, reaction_family='H_Abstraction')
        self.assertEqual(xyz1a, xyz1b)
        expected_xyz1 = """O       1.66720000   -0.47960000   -0.01490000
C      -0.82600000    0.05910000    0.00060000
H       0.30830000   -0.36560000   -0.03850000
H      -0.95970000    0.82970000   -0.78620000
H      -1.01810000    0.50970000    0.99600000
H      -1.53870000   -0.77340000   -0.17220000
H       2.36710000    0.22020000    0.01520000
"""
        expected_xyz2 = """O       3.52880000    1.08880000    0.26840000
O       2.57010000    0.20370000    0.43990000
C      -0.55710000   -0.79590000    0.46080000
C      -0.83740000    0.39160000   -0.46560000
C       0.74180000   -1.49460000    0.07550000
C      -2.13610000    1.09460000   -0.08170000
H      -1.39260000   -1.52710000    0.39950000
H      -0.47360000   -0.44040000    1.51080000
H      -0.00140000    1.12190000   -0.40310000
H      -0.91770000    0.03690000   -1.51640000
H       1.78900000   -0.61750000    0.16280000
H       0.92810000   -2.34310000    0.76710000
H       0.67640000   -1.88780000   -0.96150000
H      -2.31340000    1.94900000   -0.76840000
H      -2.99310000    0.39200000   -0.16030000
H      -2.07290000    1.48160000    0.95770000
H       3.46120000    1.34610000   -0.68540000
"""
        self.assertEqual(xyz1a, expected_xyz1)
        self.assertEqual(xyz2, expected_xyz2)

    def test_ts_error(self):
        """Test that AutoTST raises a TSError for a non-H-Abstraction reaction
        This test actually does use an H abstraction reaction that raises this error in AutoTST
        Once this is solved, add reaction 3 to the test above, and replace it with a disprop reaction for this test
        """
        xyz = atst.autotst(rmg_reaction=self.reaction3, reaction_family='H_Abstraction')
        self.assertTrue(xyz is None)
        # with self.assertRaises(TSError):
        #     atst.autotst(rmg_reaction=self.reaction3, reaction_family='H_Abstraction')


################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
