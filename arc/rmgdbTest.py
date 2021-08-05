#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.rmgdb module
"""

import unittest

from rmgpy.data.kinetics.family import KineticsFamily
from rmgpy.data.kinetics.library import LibraryReaction
from rmgpy.reaction import Reaction
from rmgpy.species import Species

import arc.rmgdb as rmgdb


class TestRMGDB(unittest.TestCase):
    """
    Contains unit tests for the rmgdb module
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.rmgdb = rmgdb.make_rmg_database_object()
        rmgdb.load_rmg_database(rmgdb=cls.rmgdb)

    def test_load_rmg_database(self):
        """Test loading the full RMG database"""
        self.assertTrue(any([fam == 'H_Abstraction' for fam in self.rmgdb.kinetics.families]))
        self.assertTrue(any([lib == 'BurkeH2O2inN2' for lib in self.rmgdb.kinetics.libraries]))
        self.assertTrue(any([lib == 'thermo_DFT_CCSDTF12_BAC' for lib in self.rmgdb.thermo.libraries]))
        self.assertTrue(any([lib == 'NOx2018' for lib in self.rmgdb.transport.libraries]))

    def test_thermo(self):
        """Test that thermodata is loaded correctly from RMG's database"""
        spc = Species().from_smiles('O=C=O')
        spc.thermo = self.rmgdb.thermo.get_thermo_data(spc)
        self.assertAlmostEqual(spc.get_enthalpy(298), -393547.040000, 5)
        self.assertAlmostEqual(spc.get_entropy(298), 213.71872, 5)
        self.assertAlmostEqual(spc.get_heat_capacity(1000), 54.35016, 5)

    def test_determining_rmg_kinetics(self):
        """Test the determine_rmg_kinetics() function"""
        r1 = Species().from_smiles('C')
        r2 = Species().from_smiles('O[O]')
        p1 = Species().from_smiles('[CH3]')
        p2 = Species().from_smiles('OO')
        r1.thermo = self.rmgdb.thermo.get_thermo_data(r1)
        r2.thermo = self.rmgdb.thermo.get_thermo_data(r2)
        p1.thermo = self.rmgdb.thermo.get_thermo_data(p1)
        p2.thermo = self.rmgdb.thermo.get_thermo_data(p2)
        rxn = Reaction(reactants=[r1, r2], products=[p1, p2])
        dh_rxn298 = sum([product.get_enthalpy(298) for product in rxn.products])\
            - sum([reactant.get_enthalpy(298) for reactant in rxn.reactants])
        rmg_reactions = rmgdb.determine_rmg_kinetics(rmgdb=self.rmgdb, reaction=rxn, dh_rxn298=dh_rxn298)
        self.assertFalse(rmg_reactions[0].kinetics.is_pressure_dependent())
        found_rxn = False
        for rxn in rmg_reactions:
            if isinstance(rxn, LibraryReaction) and rxn.library == 'Klippenstein_Glarborg2016':
                self.assertAlmostEqual(rxn.kinetics.get_rate_coefficient(1000, 1e5), 38.2514795642, 7)
                found_rxn = True
        self.assertTrue(found_rxn)

    def test_get_family(self):
        """Test the get_family() function"""
        family_1 = rmgdb.get_family(rmgdb=self.rmgdb, label='ketoenol')
        self.assertIsInstance(family_1, KineticsFamily)
        self.assertEqual(family_1.label, 'ketoenol')
        self.assertTrue(family_1.save_order)

        family_2 = rmgdb.get_family(rmgdb=self.rmgdb, label='H_Abstraction')
        self.assertIsInstance(family_2, KineticsFamily)
        self.assertEqual(family_2.label, 'H_Abstraction')
        self.assertTrue(family_2.save_order)

        family_3 = rmgdb.get_family(rmgdb=self.rmgdb, label='intra_H_migration')
        self.assertIsInstance(family_3, KineticsFamily)
        self.assertEqual(family_3.label, 'intra_H_migration')
        self.assertTrue(family_3.save_order)

        family_4 = rmgdb.get_family(rmgdb=self.rmgdb, label='Intra_H_Migration')  # test wrong capitalization still works
        self.assertIsInstance(family_4, KineticsFamily)
        self.assertEqual(family_4.label, 'intra_H_migration')  # returns the correct capitalization
        self.assertTrue(family_4.save_order)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
