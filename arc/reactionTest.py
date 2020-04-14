#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.reaction module
"""

import unittest

from rmgpy.reaction import Reaction
from rmgpy.species import Species

import arc.rmgdb as rmgdb
from arc.exceptions import ReactionError
from arc.reaction import ARCReaction
from arc.settings import default_ts_methods
from arc.species import ARCSpecies


class TestARCReaction(unittest.TestCase):
    """
    Contains unit tests for the ARCSpecies class
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.rmgdb = rmgdb.make_rmg_database_object()
        rmgdb.load_families_only(cls.rmgdb)
        cls.rxn1 = ARCReaction(reactants=['CH4', 'OH'], products=['CH3', 'H2O'])
        cls.rxn1.rmg_reaction = Reaction(reactants=[Species().from_smiles('C'), Species().from_smiles('[OH]')],
                                         products=[Species().from_smiles('[CH3]'), Species().from_smiles('O')])
        cls.rxn2 = ARCReaction(reactants=['C2H5', 'OH'], products=['C2H4', 'H2O'])
        cls.rxn2.rmg_reaction = Reaction(reactants=[Species().from_smiles('C[CH2]'),
                                                    Species().from_smiles('[OH]')],
                                         products=[Species().from_smiles('C=C'), Species().from_smiles('O')])
        cls.rxn3 = ARCReaction(reactants=['CH3CH2NH'], products=['CH2CH2NH2'])
        cls.rxn3.rmg_reaction = Reaction(reactants=[Species().from_smiles('CC[NH]')],
                                         products=[Species().from_smiles('[CH2]CN')])
        cls.rxn4 = ARCReaction(reactants=['[NH2]', 'N[NH]'], products=['N', 'N[N]'])
        cls.rxn4.rmg_reaction = Reaction(reactants=[Species().from_smiles('[NH2]'), Species().from_smiles('N[NH]')],
                                         products=[Species().from_smiles('N'), Species().from_smiles('N[N]')])

    def test_str(self):
        """Test the string representation of the object"""
        str_representation = str(self.rxn1)
        expected_representation = 'ARCReaction(label="CH4 + OH <=> CH3 + H2O", ' \
                                  'rmg_reaction="C + [OH] <=> [CH3] + O", ' \
                                  'multiplicity=2, charge=0)'
        self.assertEqual(str_representation, expected_representation)

    def test_as_dict(self):
        """Test Species.as_dict()"""
        rxn_dict = self.rxn1.as_dict()
        expected_dict = {'charge': 0,
                         'multiplicity': None,
                         'family': None,
                         'family_own_reverse': 0,
                         'label': 'CH4 + OH <=> CH3 + H2O',
                         'long_kinetic_description': u'',
                         'index': None,
                         'p_species': [],
                         'products': ['CH3', 'H2O'],
                         'r_species': [],
                         'reactants': ['CH4', 'OH'],
                         'ts_label': None,
                         'ts_xyz_guess': [],
                         'ts_methods': [tsm.lower() for tsm in default_ts_methods]}
        self.assertEqual(rxn_dict, expected_dict)

    def test_from_dict(self):
        """Test Species.from_dict()"""
        rxn_dict = self.rxn1.as_dict()
        rxn = ARCReaction(reaction_dict=rxn_dict)
        self.assertEqual(rxn.label, 'CH4 + OH <=> CH3 + H2O')
        self.assertEqual(rxn.ts_methods, [tsm.lower() for tsm in default_ts_methods])

    def test_rmg_reaction_to_str(self):
        """Test the rmg_reaction_to_str() method and the reaction label generated"""
        spc1 = Species().from_smiles('CON=O')
        spc1.label = 'CONO'
        spc2 = Species().from_smiles('C[N+](=O)[O-]')
        spc2.label = 'CNO2'
        rmg_reaction = Reaction(reactants=[spc1], products=[spc2])
        rxn = ARCReaction(rmg_reaction=rmg_reaction)
        rxn_str = rxn.rmg_reaction_to_str()
        self.assertEqual(rxn_str, 'CON=O <=> [O-][N+](=O)C')
        self.assertEqual(rxn.label, 'CONO <=> CNO2')

    def test_rxn_family(self):
        """Test that ARC gets the correct RMG family for different reactions"""
        self.rxn1.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(self.rxn1.family.label, 'H_Abstraction')
        self.assertTrue(self.rxn1.family_own_reverse)
        self.rxn2.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(self.rxn2.family.label, 'Disproportionation')
        self.assertFalse(self.rxn2.family_own_reverse)
        self.rxn3.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(self.rxn3.family.label, 'intra_H_migration')
        self.assertTrue(self.rxn3.family_own_reverse)

    def test_determine_charge(self):
        """Test determine charge"""
        self.rxn1.determine_rxn_charge()
        self.assertEqual(self.rxn1.charge, 0)

    def test_determine_multiplicity(self):
        """Test determine multiplicity"""
        self.rxn1.determine_rxn_multiplicity()
        self.assertEqual(self.rxn1.multiplicity, 2)

        self.rxn2.arc_species_from_rmg_reaction()
        self.rxn2.determine_rxn_multiplicity()
        self.assertEqual(self.rxn2.multiplicity, 1)

        self.rxn3.determine_rxn_multiplicity()
        self.assertEqual(self.rxn3.multiplicity, 2)

        self.rxn4.determine_rxn_multiplicity()
        self.assertEqual(self.rxn4.multiplicity, 3)

    def test_check_atom_balance(self):
        """Test the Reaction check_atom_balance method"""

        # A normal reaction
        rxn1 = ARCReaction(reactants=['CH4', 'OH'], products=['CH3', 'H2O'])
        rxn1.r_species = [ARCSpecies(label='CH4', smiles='C'),
                          ARCSpecies(label='OH', smiles='[OH]')]
        rxn1.p_species = [ARCSpecies(label='CH3', smiles='[CH3]'),
                          ARCSpecies(label='H2O', smiles='O')]
        self.assertTrue(rxn1.check_atom_balance())

        # A non-balanced reaction
        rxn2 = ARCReaction(reactants=['CH4', 'OH'], products=['CH4', 'H2O'])
        rxn2.r_species = [ARCSpecies(label='CH4', smiles='C'),
                          ARCSpecies(label='OH', smiles='[OH]')]
        rxn2.p_species = [ARCSpecies(label='CH4', smiles='C'),
                          ARCSpecies(label='H2O', smiles='O')]
        self.assertFalse(rxn2.check_atom_balance(raise_error=False))
        with self.assertRaises(ReactionError):
            rxn2.check_atom_balance()

        # A reaction with the same species twice on one side
        rxn3 = ARCReaction(reactants=['CH4', 'OH', 'H2O'], products=['CH3', 'H2O', 'H2O'])
        rxn3.r_species = [ARCSpecies(label='CH4', smiles='C'),
                          ARCSpecies(label='OH', smiles='[OH]'),
                          ARCSpecies(label='H2O', smiles='O')]
        rxn3.p_species = [ARCSpecies(label='CH3', smiles='[CH3]'),
                          ARCSpecies(label='H2O', smiles='O')]
        self.assertTrue(rxn3.check_atom_balance())

    def test_get_species_count(self):
        """Test the get_species_count() method"""
        rxn1 = ARCReaction(reactants=['CH4', 'OH', 'H2O'], products=['CH3', 'H2O', 'H2O'])
        spc1 = ARCSpecies(label='OH', smiles='[OH]')
        spc2 = ARCSpecies(label='H2O', smiles='O')
        self.assertEqual(rxn1.get_species_count(species=spc1, well=0), 1)
        self.assertEqual(rxn1.get_species_count(species=spc1, well=1), 0)
        self.assertEqual(rxn1.get_species_count(species=spc2, well=0), 1)
        self.assertEqual(rxn1.get_species_count(species=spc2, well=1), 2)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
