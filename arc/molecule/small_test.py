
import unittest

import arc.molecule.element as elements
from arc.molecule import Molecule
from arc.molecule.atomtype import ATOMTYPES
from arc.molecule.group import ActionError, GroupAtom, GroupBond, Group

class TestGroupBond(unittest.TestCase):    
    def test_apply_action_decrement_bond(self):
        """
        Test the GroupBond.apply_action() method for a CHANGE_BOND action.
        """
        action = ['CHANGE_BOND', '*1', -1, '*2']
        for order0 in self.orderList:
            bond0 = GroupBond(None, None, order=order0)
            bond = bond0.copy()
            try:
                bond.apply_action(action)
            except ActionError:
                self.assertTrue(1 in order0 or 1.5 in order0)