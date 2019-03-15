#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests of the arc.species.species module
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest
import os

from rmgpy.molecule.molecule import Molecule
from rmgpy.species import Species
from rmgpy.reaction import Reaction

from arc.species.zmat import ZMatrixAtom, ZMatrix
from arc.species.converter import get_xyz_string, get_xyz_matrix, molecules_from_xyz
from arc.settings import arc_path

################################################################################


class TestZMatrixAtom(unittest.TestCase):
    """
    Contains unit tests for the TSGuess class
    """
    # @classmethod
    # def setUpClass(cls):
    #     """
    #     A method that is run before all unit tests in this class.
    #     """
    #     cls.maxDiff = None
    #     spc1 = Species().fromSMILES(str('CON=O'))
    #     spc1.label = str('CONO')
    #     spc2 = Species().fromSMILES(str('C[N+](=O)[O-]'))
    #     spc2.label = str('CNO2')
    #     rmg_reaction = Reaction(reactants=[spc1], products=[spc2])
    #     cls.tsg1 = TSGuess(rmg_reaction=rmg_reaction, method='AutoTST', family='H_Abstraction')
    #     xyz = """N       0.9177905887     0.5194617797     0.0000000000
    #              H       1.8140204898     1.0381941417     0.0000000000
    #              H      -0.4763167868     0.7509348722     0.0000000000
    #              N       0.9992350860    -0.7048575683     0.0000000000
    #              N      -1.4430010939     0.0274543367     0.0000000000
    #              H      -0.6371484821    -0.7497769134     0.0000000000
    #              H      -2.0093636431     0.0331190314    -0.8327683174
    #              H      -2.0093636431     0.0331190314     0.8327683174"""
    #     cls.tsg2 = TSGuess(xyz=xyz)

    def test_to_str(self):
        """Test TSGuess.as_dict()"""
        test_atom = ZMatrixAtom(symbol='C', indices_list=[1, 2, 3, 4], distance=1, angle=2, dihedral=3)
        test_atom_str = test_atom.to_str()
        expected_str = ""
        self.assertEqual(test_atom_str, expected_str)


################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
