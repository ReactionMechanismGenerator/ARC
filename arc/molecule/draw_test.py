#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.molecule.atomtype module.
"""

import os
import os.path
import unittest

from arc.molecule import Molecule
from arc.molecule.draw import MoleculeDrawer
from arc.species import ARCSpecies


class TestMoleculeDrawer(unittest.TestCase):
    """
    Contains unit tests of the MoleculeDrawer class.
    """

    def setUp(self):
        """
        A function run before each unit test in this class.
        """
        self.drawer = MoleculeDrawer()
        self.molecule = Molecule(smiles='CC(=O)CC')

    def test_draw_png(self):
        """
        Test we can create PNG files from molecules.
        """
        try:
            from cairocffi import ImageSurface
        except ImportError:
            from cairo import ImageSurface
        path = 'test_molecule.png'
        if os.path.exists(path):
            os.unlink(path)
        surface, _cr, (_xoff, _yoff, width, height) = self.drawer.draw(self.molecule, file_format='png', target=path)
        self.assertTrue(os.path.exists(path), "File doesn't exist")
        self.assertGreater(width, height)
        os.unlink(path)
        self.assertIsInstance(surface, ImageSurface)

    def test_draw_pdf(self):
        """
        Test we can create PDF files from molecules.
        """
        try:
            from cairocffi import PDFSurface
        except ImportError:
            from cairo import PDFSurface
        path = 'test_molecule.pdf'
        if os.path.exists(path):
            os.unlink(path)
        surface, _cr, (_xoff, _yoff, width, height) = self.drawer.draw(self.molecule, file_format='pdf', target=path)
        self.assertIsInstance(surface, PDFSurface)
        self.assertGreater(width, height)
        os.unlink(path)

    def test_draw_polycycle(self):
        """
        Test we can draw a polycyclic molecule
        """
        try:
            from cairocffi import PDFSurface
        except ImportError:
            from cairo import PDFSurface
        path = 'test_molecule.pdf'
        if os.path.exists(path):
            os.unlink(path)
        polycycle = Molecule(smiles="C123CC4CC1COCC2CCC34")
        surface, _cr, (_xoff, _yoff, width, height) = self.drawer.draw(polycycle, file_format='pdf', target=path)
        self.assertIsInstance(surface, PDFSurface)
        self.assertGreater(width, height)
        os.unlink(path)

    def test_draw_pdf_without_file(self):
        """
        Test we can create PDF surface without a temporary file (newer versions of PyCairo?)
        """
        try:
            from cairocffi import PDFSurface
        except ImportError:
            from cairo import PDFSurface
        surface, _cr, (_xoff, _yoff, width, height) = self.drawer.draw(self.molecule, file_format='pdf')
        self.assertIsInstance(surface, PDFSurface)
        self.assertGreater(width, height)

    def test_draw_non_standard_bonds(self):
        """
        Test we can draw a molecule with non-standard bonds (e.g., triple bond).
        """
        spec = ARCSpecies(label='S1', smiles='[CH2]C=C[CH2]')
        hybrid = spec.get_resonance_hybrid()
        try:
            from cairocffi import PDFSurface
        except ImportError:
            from cairo import PDFSurface
        surface, _cr, (_xoff, _yoff, width, height) = self.drawer.draw(hybrid, file_format='pdf')
        self.assertGreater(width, height)
        self.assertIsInstance(surface, PDFSurface)

    def test_draw_hydrogen_bond_adsorbate(self):
        """
        Test we can draw a molecule with a hydrogen bond adsorbate.
        """
        molecule = Molecule().from_adjacency_list("""
1  O u0 p3 c-1 {2,S} {10,H}
2  N u0 p0 c+1 {1,S} {3,D} {4,S}
3  O u0 p2 c0 {2,D}
4  O u0 p2 c0 {2,S} {7,S}
5  N u0 p1 c0 {6,S} {8,S} {9,S} {7,H}
6  O u0 p2 c0 {5,S} {10,S}
7  H u0 p0 c0 {4,S} {5,H}
8  H u0 p0 c0 {5,S}
9  H u0 p0 c0 {5,S}
10 H u0 p0 c0 {6,S} {1,H}
11 X u0 p0 c0
        """
        )
        try:
            from cairocffi import PDFSurface
        except ImportError:
            from cairo import PDFSurface
        surface, _cr, (_xoff, _yoff, _width, _height) = self.drawer.draw(molecule, file_format='pdf')
        self.assertIsInstance(surface, PDFSurface)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
