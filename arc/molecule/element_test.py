#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the molecule.element module.
"""

import unittest

import arc.molecule.element
from arc.molecule.element import Element


class TestElement(unittest.TestCase):
    """
    Contains unit tests of the Element class.
    """

    def setUp(self):
        """
        A function run before each unit test in this class.
        """
        self.element = molecule.molecule.element.C
        self.element_x = molecule.molecule.element.X

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

    def test_output(self):
        """
        Test that we can reconstruct an Element object from its repr()
        output with no loss of information.
        """
        namespace = {}
        exec('element = {0!r}'.format(self.element), globals(), namespace)
        self.assertIn('element', namespace)
        element = namespace['element']
        self.assertEqual(self.element.number, element.number)
        self.assertEqual(self.element.symbol, element.symbol)
        self.assertEqual(self.element.name, element.name)
        self.assertEqual(self.element.mass, element.mass)

    def test_get_element(self):
        """
        Test the molecule.elements.get_element() method.
        """
        self.assertTrue(molecule.molecule.element.get_element(6) is self.element)
        self.assertTrue(molecule.molecule.element.get_element('C') is self.element)
        self.assertTrue(molecule.molecule.element.get_element(0) is self.element_x)
        self.assertTrue(molecule.molecule.element.get_element('X') is self.element_x)

    def test_get_element_isotope(self):
        """
        Test that the molecule.elements.get_element() method works for isotopes.
        """
        self.assertTrue(isinstance(molecule.molecule.element.get_element('C', isotope=13), Element))
        self.assertTrue(isinstance(molecule.molecule.element.get_element(6, isotope=13), Element))

    def test_chemkin_name(self):
        """
        Test that retrieving the chemkin name of an element works.
        """
        d = molecule.molecule.element.get_element('H', isotope=2)
        self.assertEqual(d.chemkin_name, 'D')

        c13 = molecule.molecule.element.get_element('C', isotope=13)
        self.assertEqual(c13.chemkin_name, 'CI')

        o18 = molecule.molecule.element.get_element('O', isotope=18)
        self.assertEqual(o18.chemkin_name, 'OI')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
