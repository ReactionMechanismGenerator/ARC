import unittest
import sys
import os
from arc.imports import settings
ARC_FAMILIES_PATH = settings['ARC_FAMILIES_PATH']
sys.path.append(ARC_FAMILIES_PATH)

from arc.family.family import ReactionFamily, get_reaction_family_products, get_recipe_actions
from arc.reaction.reaction import ARCReaction
from arc.species.species import ARCSpecies

class TestEsterHydrolysisReactionFamily(unittest.TestCase):
    """
    Contains unit tests for the ester hydrolysis reaction family.
    """
    @classmethod
    def setUpClass(cls):
        """Set up the test by defining the ester hydrolysis reaction family."""
        cls.family = ReactionFamily('ester_hydrolysis')

    def test_ester_hydrolysis_reaction(self):
        """Test if ester hydrolysis products are correctly generated."""
        ester = ARCSpecies(label='ester', smiles='CC(=O)OC')
        water = ARCSpecies(label='H2O', smiles='O')
        acid = ARCSpecies(label='acid', smiles='CC(=O)O')
        alcohol = ARCSpecies(label='alcohol', smiles='CO')
        rxn = ARCReaction(r_species=[ester, water], p_species=[acid, alcohol])
        products = get_reaction_family_products(rxn)
        product_smiles = [p.to_smiles() for p in products[0]['products']]
        expected_product_smiles = ['CO', 'CC(=O)O']
        self.assertEqual(product_smiles, expected_product_smiles)

    def test_recipe_actions(self):
        """Test if the reaction recipe is applied correctly."""
        groups_file_path = os.path.join(ARC_FAMILIES_PATH, 'ester_hydrolysis.py')
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

    def test_ester_hydrolysis_withP(self):
        """Test if ester hydrolysis products are correctly generated."""
        ester = ARCSpecies(label='ester', smiles='CP(=O)(OC)O')
        water = ARCSpecies(label='H2O', smiles='O')
        acid = ARCSpecies(label='acid', smiles='CP(=O)(O)O')
        alcohol = ARCSpecies(label='alcohol', smiles='CO')
        rxn = ARCReaction(r_species=[ester, water], p_species=[acid, alcohol])
        products = get_reaction_family_products(rxn)
        product_smiles = [p.to_smiles() for p in products[0]['products']]
        expected_product_smiles = ['CO', 'CP(=O)(O)O']
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

class TestImineHydrolysisReactionFamily(unittest.TestCase):
    """
    Contains unit tests for the imine hydrolysis reaction family.
    """
    @classmethod
    def setUpClass(cls):
        """Set up the test by defining the imine hydrolysis reaction family."""
        cls.family = ReactionFamily('imine_hydrolysis')
    def test_imine_hydrolysis_reaction(self):
        """Test if imine hydrolysis products are correctly generated."""
        imine = ARCSpecies(label='imine', smiles='CC(=N)C')
        water = ARCSpecies(label='H2O', smiles='O')
        amine = ARCSpecies(label='amine', smiles='CC(O)(N)C')
        rxn = ARCReaction(r_species=[imine, water], p_species=[amine])
        products = get_reaction_family_products(rxn)
        expected_product_smiles = ['CC(O)(N)C']
        product_smiles = [p.to_smiles() for p in products[0]['products']]
        self.assertEqual(product_smiles, expected_product_smiles)

    def test_recipe_actions(self):
        """Test if the reaction recipe is applied correctly for imine hydrolysis."""
        groups_file_path = os.path.join(ARC_FAMILIES_PATH, 'imine_hydrolysis.py')
        with open(groups_file_path, 'r') as f:
            groups_as_lines = f.readlines()
        actions = get_recipe_actions(groups_as_lines)
        expected_actions = [
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
        """Set up the test by defining the ester hydrolysis reaction family."""
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


if __name__ == '__main__':
    unittest.main()
