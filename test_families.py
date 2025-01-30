import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Code/ARC/data/families'))

from arc.reaction.family import ReactionFamily, get_reaction_family_products, get_recipe_actions
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
        rxn = ARCReaction(r_species=[ester, water], p_species=[acid, alcohol], rmg_family_set=['ester_hydrolysis'])
        products = get_reaction_family_products(rxn)
        product_smiles = [p.to_smiles() for p in products[0]['products']]
        expected_product_smiles = ['CC(=O)O', 'CO']
        self.assertEqual(product_smiles, expected_product_smiles)

    def test_recipe_actions(self):
        """Test if the reaction recipe is applied correctly."""
        groups_file_path = os.path.join(os.path.dirname(__file__), '../../Code/ARC/data/families/ester_hydrolysis/groups.py')
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
        rxn = ARCReaction(r_species=[ester, water], p_species=[acid, alcohol], rmg_family_set=['ester_hydrolysis'])
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
        rxn = ARCReaction(r_species=[nitrile, water], p_species=[acid], rmg_family_set=['nitrile_hydrolysis'])
        products = get_reaction_family_products(rxn)
        product_smiles = [p.to_smiles() for p in products[0]['products']]
        expected_product_smiles = ['CC(=N)O']
        self.assertEqual(set(product_smiles), expected_product_smiles)

    def test_recipe_actions(self):
        """Test if the reaction recipe is applied correctly for nitrile hydrolysis."""
        groups_file_path = os.path.join(os.path.dirname(__file__),
                                        '../../Code/ARC/data/families/nitrile_hydrolysis/groups.py')
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
        rxn = ARCReaction(r_species=[imine, water], p_species=[amine], rmg_family_set=['imine_hydrolysis'])
        products = get_reaction_family_products(rxn)
        product_smiles = [p.to_smiles() for p in products[0]['products']]
        self.assertEqual(set(product_smiles), amine)

    def test_recipe_actions(self):
        """Test if the reaction recipe is applied correctly for imine hydrolysis."""
        groups_file_path = os.path.join(os.path.dirname(__file__),
                                        '../../Code/ARC/data/families/imine_hydrolysis/groups.py')
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
        rxn = ARCReaction(r_species=[ether, water], p_species=[alcohol1, alcohol2], rmg_family_set=['ether_hydrolysis'])
        products = get_reaction_family_products(rxn)
        product_smiles = [p.to_smiles() for p in products[0]['products']]
        expected_product_smiles = ['CCO', 'CO']
        self.assertEqual(set(product_smiles), expected_product_smiles)

    def test_recipe_actions(self):
        """Test if the reaction recipe is applied correctly."""
        groups_file_path = os.path.join(os.path.dirname(__file__), '../../Code/ARC/data/families/ether_hydrolysis/groups.py')
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
