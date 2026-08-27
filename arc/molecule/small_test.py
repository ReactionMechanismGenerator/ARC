
import unittest

import arc.molecule.element as elements
from arc.molecule import Molecule
from arc.molecule.atomtype import ATOMTYPES
from arc.molecule.group import ActionError, GroupAtom, GroupBond, Group

class TestGroupBond(unittest.TestCase):
    def setUp(self):
        """
        A method called before each unit test in this class.
        """
        self.bond = GroupBond(None, None, order=[2])
        self.orderList = [[1], [2], [3], [1.5], [1, 2], [2, 1], [2, 3], [1, 2, 3]]  # todo : unit tests for vdw

    def test_apply_action_decrement_bond(self):
        """
        Test the GroupBond.apply_action() method for a CHANGE_BOND action.
        """
        action = ['CHANGE_BOND', '*1', -1, '*2']
        for order0 in self.orderList:
            print(order0)
            bond0 = GroupBond(None, None, order=order0)
            bond = bond0.copy()
            try:
                bond.apply_action(action)
                print(f'order0: {order0}, new order: {bond.order}')
            except ActionError:
                self.assertTrue(1 in order0 or 1.5 in order0)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))