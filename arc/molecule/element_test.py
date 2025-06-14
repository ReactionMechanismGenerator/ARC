#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the molecule.element module.
"""

import unittest

import arc.molecule.element as element


class TestElement(unittest.TestCase):
    """
    Contains unit tests of the Element class.
    """

    def setUp(self):
        """
        A function run before each unit test in this class.
        """
        self.element = element.C
        self.element_x = element.X

    def test_pickle(self):
        """
        Test that an Element object can be successfully pickled and
        unpickled with no loss of information.
        """
        import pickle
        element = pickle.loads(pickle.dumps(self.element))
        self.assertEqual(self.element.number, element.number)
        self.assertEqual(self.element.symbol, element.symbol)
        self.assertEqual(self.element.name, element.name)
        self.assertEqual(self.element.mass, element.mass)

    def test_get_element(self):
        """
        Test the molecule.elements.get_element() method.
        """
        self.assertIs(element.get_element(6), self.element)
        self.assertIs(element.get_element('C'), self.element)
        self.assertIs(element.get_element(0), self.element_x)
        self.assertIs(element.get_element('X'), self.element_x)

    def test_get_element_isotope(self):
        """
        Test that the molecule.elements.get_element() method works for isotopes.
        """
        self.assertTrue(isinstance(element.get_element('C', isotope=13), element.Element))
        self.assertTrue(isinstance(element.get_element(6, isotope=13), element.Element))

    def test_chemkin_name(self):
        """
        Test that retrieving the chemkin name of an element works.
        """
        d = element.get_element('H', isotope=2)
        self.assertEqual(d.chemkin_name, 'D')

        c13 = element.get_element('C', isotope=13)
        self.assertEqual(c13.chemkin_name, 'CI')

        o18 = element.get_element('O', isotope=18)
        self.assertEqual(o18.chemkin_name, 'OI')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
