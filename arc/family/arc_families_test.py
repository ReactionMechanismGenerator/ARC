#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the kinetic families defined under arc.data.families.
"""

import unittest
import os

from arc.family.family import ReactionFamily, get_reaction_family_products, get_recipe_actions
from arc.imports import settings
from arc.reaction.reaction import ARCReaction
from arc.species.species import ARCSpecies

ARC_FAMILIES_PATH = settings['ARC_FAMILIES_PATH']


class TestCarbonylBasedHydrolysisReactionFamily(unittest.TestCase):
    """
    Contains unit tests for the carbonyl-based hydrolysis reaction family.
    """

    @classmethod
    def setUpClass(cls):
        """Set up the test by defining the carbonyl-based hydrolysis reaction family."""
        cls.family = ReactionFamily('carbonyl_based_hydrolysis')

    def test_carbonyl_based_hydrolysis_reaction(self):
        """Test if carbonyl_based hydrolysis products are correctly generated."""
        carbonyl = ARCSpecies(label='carbonyl', smiles='CC(=O)OC')
        water = ARCSpecies(label='H2O', smiles='O')
        acid = ARCSpecies(label='acid', smiles='CC(=O)O')
        alcohol = ARCSpecies(label='alcohol', smiles='CO')
        rxn = ARCReaction(r_species=[carbonyl, water], p_species=[acid, alcohol])
        products = get_reaction_family_products(rxn)
        product_smiles = [p.to_smiles() for p in products[0]['products']]
        expected_product_smiles = ['CC(=O)O', 'CO']
        self.assertEqual(product_smiles, expected_product_smiles)

    def test_recipe_actions(self):
        """Test if the reaction recipe is applied correctly."""
        groups_file_path = os.path.join(ARC_FAMILIES_PATH, 'carbonyl_based_hydrolysis.py')
        with open(groups_file_path, 'r') as f:
            groups_as_lines = f.readlines()
        actions = get_recipe_actions(groups_as_lines)
        expected_actions = [
            ['BREAK_BOND', '*1', 1, '*2'],
            ['BREAK_BOND', '*3', 1, '*4'],
            ['FORM_BOND', '*1', 1, '*4'],
            ['FORM_BOND', '*2', 1, '*3'],
        ]
        self.assertEqual(actions, expected_actions)

    def test_carbonyl_based_hydrolysis_withP(self):
        """Test if carbonyl-based hydrolysis products are correctly generated."""
        carbonyl= ARCSpecies(label='carbonyl', smiles='CP(=O)(OC)O')
        water = ARCSpecies(label='H2O', smiles='O')
        acid = ARCSpecies(label='acid', smiles='CP(=O)(O)O')
        alcohol = ARCSpecies(label='alcohol', smiles='CO')
        rxn = ARCReaction(r_species=[carbonyl, water], p_species=[acid, alcohol])
        products = get_reaction_family_products(rxn)
        product_smiles = [p.to_smiles() for p in products[0]['products']]
        expected_product_smiles = ['CP(=O)(O)O', 'CO']
        self.assertEqual(product_smiles, expected_product_smiles)


class TestNitrileHydrolysisReactionFamily(unittest.TestCase):
    """
    Contains unit tests for the nitrile hydrolysis reaction family.
    """

    @classmethod
    def setUpClass(cls):
        """Set up the test by defining the nitrile hydrolysis reaction family."""
        cls.family = ReactionFamily('nitrile_hydrolysis')

    def test_nitrile_hydrolysis_reaction(self):
        """Test if nitrile hydrolysis products are correctly generated."""
        nitrile = ARCSpecies(label='nitrile', smiles='CC#N')
        water = ARCSpecies(label='H2O', smiles='O')
        acid = ARCSpecies(label='acid', smiles='CC(=N)O')
        rxn = ARCReaction(r_species=[nitrile, water], p_species=[acid])
        products = get_reaction_family_products(rxn)
        product_smiles = [p.to_smiles() for p in products[0]['products']]
        expected_product_smiles = ['CC(=N)O']
        self.assertEqual(product_smiles, expected_product_smiles)

    def test_recipe_actions(self):
        """Test if the reaction recipe is applied correctly for nitrile hydrolysis."""
        groups_file_path = os.path.join(ARC_FAMILIES_PATH, 'nitrile_hydrolysis.py')
        with open(groups_file_path, 'r') as f:
            groups_as_lines = f.readlines()
        actions = get_recipe_actions(groups_as_lines)
        expected_actions =[
        ['CHANGE_BOND', '*1', -1, '*2'],
        ['BREAK_BOND', '*3', 1, '*4'],
        ['FORM_BOND', '*1', 1, '*4'],
        ['FORM_BOND', '*2', 1, '*3'],
        ]
        self.assertEqual(actions, expected_actions)


class TestEtherHydrolysisReactionFamily(unittest.TestCase):
    """
    Contains unit tests for the ether hydrolysis reaction family.
    """

    @classmethod
    def setUpClass(cls):
        """Set up the test by defining the ether hydrolysis reaction family."""
        cls.family = ReactionFamily('ether_hydrolysis')

    def test_ether_hydrolysis_reaction(self):
        """Test if ether hydrolysis products are correctly generated."""
        ether = ARCSpecies(label='ether', smiles='CCOC')
        water = ARCSpecies(label='H2O', smiles='O')
        alcohol1 = ARCSpecies(label='alcohol1', smiles='CCO')
        alcohol2 = ARCSpecies(label='alcohol2', smiles='CO')
        rxn = ARCReaction(r_species=[ether, water], p_species=[alcohol1, alcohol2])
        products = get_reaction_family_products(rxn)
        product_smiles = [p.to_smiles() for p in products[0]['products']]
        expected_product_smiles = ['CCO', 'CO']
        self.assertEqual(product_smiles, expected_product_smiles)

    def test_recipe_actions(self):
        """Test if the reaction recipe is applied correctly."""
        groups_file_path = os.path.join(ARC_FAMILIES_PATH, 'ether_hydrolysis.py')
        with open(groups_file_path, 'r') as f:
            groups_as_lines = f.readlines()
        actions = get_recipe_actions(groups_as_lines)
        expected_actions = [
        ['BREAK_BOND', '*1', 1, '*2'],
        ['BREAK_BOND', '*3', 1, '*4'],
        ['FORM_BOND', '*1', 1, '*4'],
        ['FORM_BOND', '*2', 1, '*3'],
        ]
        self.assertEqual(actions, expected_actions)


class TestQASN2ReactionFamily(unittest.TestCase):
    """
    Contains unit tests for the qa-sn2 reaction family.
    """

    @classmethod
    def setUpClass(cls):
        """Set up the test by defining the qa_sn2 reaction family."""
        cls.family = ReactionFamily('qa_sn2')

    def test_qa_sn2_reaction(self):
        """Test if qa-sn2 products are correctly generated."""
        qa = ARCSpecies(label='QA', smiles='C[N+](C)(C)C')
        oh_d1 = ARCSpecies(label='OH_d1', smiles='[OH-].O')

        alcohol_water = ARCSpecies(label='alcohol_water', smiles='CO.O')
        amine = ARCSpecies(label='amine', smiles='CN(C)C')

        rxn = ARCReaction(r_species=[qa, oh_d1], p_species=[alcohol_water, amine])

        products = get_reaction_family_products(rxn)
        product_smiles = [p.to_smiles() for p in products[0]['products']]

        expected_product_smiles = ['CO.O', 'CN(C)C']
        self.assertEqual(sorted(product_smiles), sorted(expected_product_smiles))

        # The OH_d1 species (a disconnected hydroxide-water group) may be listed first.
        rxn = ARCReaction(r_species=[oh_d1, qa], p_species=[alcohol_water, amine])
        self.assertEqual(rxn.family, 'qa_sn2')
        products = get_reaction_family_products(rxn)
        self.assertEqual(sorted(p.to_smiles() for p in products[0]['products']), sorted(expected_product_smiles))

    def test_recipe_actions(self):
        """Test if the reaction recipe is applied correctly."""
        groups_file_path = os.path.join(ARC_FAMILIES_PATH, 'qa_sn2.py')
        with open(groups_file_path, 'r') as f:
            groups_as_lines = f.readlines()

        actions = get_recipe_actions(groups_as_lines)

        expected_actions = [
            ['BREAK_BOND', '*1', 1, '*2'],
            ['FORM_BOND', '*1', 1, '*3'],
            ['LOSE_PAIR', '*3', '1'],
            ['GAIN_PAIR', '*2', '1'],
        ]

        self.assertEqual(actions, expected_actions)

    def test_qa_sn2_with_a_heteroatom_on_the_reacting_carbon(self):
        """Test that a heteroatom bonded to the reacting carbon (a methoxymethyl group) matches the family."""
        qa = ARCSpecies(label='QA', smiles='COC[N+](C)(C)C')
        oh_d1 = ARCSpecies(label='OH_d1', smiles='[OH-].O')
        rxn = ARCReaction(r_species=[qa, oh_d1], p_species=[ARCSpecies(label='alcohol_water', smiles='COCO.O'),
                                                             ARCSpecies(label='amine', smiles='CN(C)C')])
        self.assertEqual(rxn.family, 'qa_sn2')
        products = get_reaction_family_products(rxn)
        self.assertEqual(sorted(p.to_smiles() for p in products[0]['products']), sorted(['COCO.O', 'CN(C)C']))

    def test_qa_tsn2_with_longer_alkyl_group(self):
        """Test if qa-sn2 products are correctly generated for a longer alkyl substituent."""
        qa = ARCSpecies(label='QA', smiles='CC[N+](C)(C)C')
        oh_d1 = ARCSpecies(label='OH_d1', smiles='[OH-].O')

        alcohol_water = ARCSpecies(label='alcohol_water', smiles='CCO.O')
        amine = ARCSpecies(label='amine', smiles='CN(C)C')

        rxn = ARCReaction(r_species=[qa, oh_d1], p_species=[alcohol_water, amine])

        products = get_reaction_family_products(rxn)
        product_smiles = [p.to_smiles() for p in products[0]['products']]

        expected_product_smiles = ['CCO.O', 'CN(C)C']
        self.assertEqual(sorted(product_smiles), sorted(expected_product_smiles))


class TestQAHofmannEliminationReactionFamily(unittest.TestCase):
    """
    Contains unit tests for the qa_hofmann reaction family.
    """

    @classmethod
    def setUpClass(cls):
        """Set up the test by defining the qa_hofmann reaction family."""
        cls.family = ReactionFamily('qa_hofmann')

    def test_qa_hofmann_elimination_reaction(self):
        """Test if qa_hofmann products are correctly generated."""
        qa = ARCSpecies(label='QA', smiles='CC[N+](C)(C)C')
        oh_d1 = ARCSpecies(label='OH_d1', smiles='[OH-].O')

        alkene = ARCSpecies(label='alkene', smiles='C=C')
        amine = ARCSpecies(
            label='amine', smiles='CN(C)C')
        water_dimer = ARCSpecies(label='water_dimer', smiles='O.O')

        rxn = ARCReaction(r_species=[qa, oh_d1], p_species=[alkene, amine, water_dimer])

        products = get_reaction_family_products(rxn)
        product_smiles = [p.to_smiles() for p in products[0]['products']]

        expected_product_smiles = ['C=C', 'CN(C)C', 'O.O']
        self.assertEqual(sorted(product_smiles), sorted(expected_product_smiles))

        # The OH_d1 species (a disconnected hydroxide-water group) may be listed first.
        rxn = ARCReaction(r_species=[oh_d1, qa], p_species=[alkene, amine, water_dimer])
        self.assertEqual(rxn.family, 'qa_hofmann')
        products = get_reaction_family_products(rxn)
        self.assertEqual(sorted(p.to_smiles() for p in products[0]['products']), sorted(expected_product_smiles))

    def test_recipe_actions(self):
        """Test if the reaction recipe is applied correctly."""
        groups_file_path = os.path.join(ARC_FAMILIES_PATH, 'qa_hofmann.py')
        with open(groups_file_path, 'r') as f:
            groups_as_lines = f.readlines()

        actions = get_recipe_actions(groups_as_lines)

        expected_actions = [
            ['BREAK_BOND', '*1', 1, '*2'],
            ['BREAK_BOND', '*3', 1, '*4'],
            ['CHANGE_BOND', '*1', 1, '*3'],
            ['FORM_BOND', '*5', 1, '*4'],
            ['LOSE_PAIR', '*5', '1'],
            ['GAIN_PAIR', '*2', '1'],
        ]

        self.assertEqual(actions, expected_actions)

    def test_qa_hofmann_elimination_of_choline(self):
        """Test that choline, with a hydroxyl group on the beta carbon, matches the family."""
        qa = ARCSpecies(label='choline', smiles='OCC[N+](C)(C)C')
        oh_d1 = ARCSpecies(label='OH_d1', smiles='[OH-].O')
        rxn = ARCReaction(r_species=[qa, oh_d1], p_species=[ARCSpecies(label='enol', smiles='C=CO'),
                                                             ARCSpecies(label='amine', smiles='CN(C)C'),
                                                             ARCSpecies(label='water_dimer', smiles='O.O')])
        self.assertEqual(rxn.family, 'qa_hofmann')
        products = get_reaction_family_products(rxn)
        self.assertEqual(sorted(p.to_smiles() for p in products[0]['products']), sorted(['C=CO', 'CN(C)C', 'O.O']))

    def test_qa_hofmann_elimination_does_not_open_a_ring(self):
        """
        Test that a Hofmann elimination whose C_alpha-N bond is in a ring (a ring-opening elimination) does not match,
        while a Hofmann elimination of a QA with a ring that is kept (N outside the ring) does.
        """
        oh_d1 = ARCSpecies(label='OH_d1', smiles='[OH-].O')
        dmp = ARCSpecies(label='DMP', smiles='C[N+]1(C)CCCCC1')
        rxn = ARCReaction(r_species=[dmp, oh_d1], p_species=[ARCSpecies(label='aminoalkene', smiles='C=CCCCN(C)C'),
                                                              ARCSpecies(label='H2O_1', smiles='O'),
                                                              ARCSpecies(label='H2O_2', smiles='O')])
        self.assertIsNone(rxn.family)
        self.assertEqual(self.family.non_ring_bonds, [('*1', '*2')])
        cyclopentyl_tma = ARCSpecies(label='QA', smiles='C[N+](C)(C)C1CCCC1')
        rxn = ARCReaction(r_species=[cyclopentyl_tma, oh_d1],
                          p_species=[ARCSpecies(label='cyclopentene', smiles='C1=CCCC1'),
                                     ARCSpecies(label='amine', smiles='CN(C)C'),
                                     ARCSpecies(label='water_dimer', smiles='O.O')])
        self.assertEqual(rxn.family, 'qa_hofmann')

    def test_qa_hofmann_merges_the_water_fragments(self):
        """Test that the two water fragments of the recipe are merged into the water dimer product."""
        self.assertTrue(self.family.has_a_disconnected_reactant)
        rxn = ARCReaction(r_species=[ARCSpecies(label='QA', smiles='CC[N+](C)(C)C'),
                                     ARCSpecies(label='OH_d1', smiles='[OH-].O')],
                          p_species=[ARCSpecies(label='alkene', smiles='C=C'),
                                     ARCSpecies(label='amine', smiles='CN(C)C'),
                                     ARCSpecies(label='water_dimer', smiles='O.O')])
        products = get_reaction_family_products(rxn)
        self.assertEqual(sorted(p.to_smiles() for p in products[0]['products']), sorted(['C=C', 'CN(C)C', 'O.O']))

    def test_qa_hofmann_elimination_with_longer_alkyl_group(self):
        """Test if qa_hofmann products are correctly generated for a longer alkyl substituent."""
        qa = ARCSpecies(label='QA', smiles='CCC[N+](C)(C)C')
        oh_d1 = ARCSpecies(label='OH_d1', smiles='[OH-].O')

        alkene = ARCSpecies(label='alkene', smiles='C=CC')
        amine = ARCSpecies(label='amine', smiles='CN(C)C')
        water_dimer = ARCSpecies(label='water_dimer', smiles='O.O')

        rxn = ARCReaction(r_species=[qa, oh_d1], p_species=[alkene, amine, water_dimer])

        products = get_reaction_family_products(rxn)
        product_smiles = [p.to_smiles() for p in products[0]['products']]

        expected_product_smiles = ['C=CC', 'CN(C)C', 'O.O']
        self.assertEqual(sorted(product_smiles), sorted(expected_product_smiles))


if __name__ == '__main__':
    unittest.main()
