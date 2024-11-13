import unittest
import sys
import os

# Add the path to the directory where ester_hydrolysis is located
sys.path.append('/home/leen/Code/ARC/data/families')


from arc.reaction.family import ReactionFamily, get_reaction_family_products, get_recipe_actions
from arc.reaction.reaction import ARCReaction
from arc.species.species import ARCSpecies

# Print the current working directory to confirm the script's execution location
print("Current working directory:", os.getcwd())

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
        # Define the ester and water as reactants
        ester_smiles = 'CC(=O)OC'  # Methyl acetate
        water_smiles = 'O'

        # Create ARC species for the reactants
        ester = ARCSpecies(label='ester', smiles=ester_smiles)
        water = ARCSpecies(label='H2O', smiles=water_smiles)

        # Define the expected products (acetic acid and methanol)
        acid_smiles = 'CC(=O)O'
        alcohol_smiles = 'CO'

        acid = ARCSpecies(label='acid', smiles=acid_smiles)
        alcohol = ARCSpecies(label='alcohol', smiles=alcohol_smiles)

        # Create a reaction instance
        rxn = ARCReaction(r_species=[ester, water], p_species=[acid, alcohol], rmg_family_set=['ester_hydrolysis'])

        # Get reaction products
        products = get_reaction_family_products(rxn)

        # Check if the generated products match expected ones
        product_smiles = [p.to_smiles() for p in products[0]['products']]
        expected_product_smiles = {acid_smiles, alcohol_smiles}

        self.assertEqual(set(product_smiles), expected_product_smiles)

    def test_recipe_actions(self):
        """Test if the reaction recipe is applied correctly."""
        # Load the groups.py file content as lines
        groups_file_path = '/home/leen/Code/ARC/data/families/ester_hydrolysis/groups.py'
        with open(groups_file_path, 'r') as f:
            groups_as_lines = f.readlines()

        # Get the recipe actions from the groups file content
        actions = get_recipe_actions(groups_as_lines)

        # Define the expected actions
        expected_actions = [
            ['BREAK_BOND', '*1', 1, '*2'],
            ['BREAK_BOND', '*3', 1, '*4'],
            ['FORM_BOND', '*1', 1, '*4'],
            ['FORM_BOND', '*2', 1, '*3'],
        ]

        # Assert that the actions match the expected actions
        self.assertEqual(actions, expected_actions)

    def test_ester_hydrolysis_withP(self):
        """Test if ester hydrolysis products are correctly generated."""
        # Define the ester and water as reactants
        ester_smiles = 'CP(=O)(OC)O'  # Phosphoester
        water_smiles = 'O'

        # Create ARC species for the reactants
        ester = ARCSpecies(label='ester', smiles=ester_smiles)
        water = ARCSpecies(label='H2O', smiles=water_smiles)

        # Define the expected products (acetic acid and methanol)
        acid_smiles = 'CP(=O)(O)O'
        alcohol_smiles = 'CO'

        acid = ARCSpecies(label='acid', smiles=acid_smiles)
        alcohol = ARCSpecies(label='alcohol', smiles=alcohol_smiles)

        # Create a reaction instance
        rxn = ARCReaction(r_species=[ester, water], p_species=[acid, alcohol], rmg_family_set=['ester_hydrolysis'])

        # Get reaction products
        products = get_reaction_family_products(rxn)

        # Check if the generated products match expected ones
        product_smiles = [p.to_smiles() for p in products[0]['products']]
        expected_product_smiles = {acid_smiles, alcohol_smiles}

        self.assertEqual(set(product_smiles), expected_product_smiles)


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
        # Define the nitrile and water as reactants
        nitrile_smiles = 'CC#N'  # Acetonitrile
        water_smiles = 'O'

        # Create ARC species for the reactants
        nitrile = ARCSpecies(label='nitrile', smiles=nitrile_smiles)
        water = ARCSpecies(label='H2O', smiles=water_smiles)

        # Define the expected products (acetamide as an intermediate and methanol)
        acid_smiles = 'CC(=N)O'  # Acetamide
        acid = ARCSpecies(label='acid', smiles=acid_smiles)

        # Create a reaction instance
        rxn = ARCReaction(r_species=[nitrile, water], p_species=[acid], rmg_family_set=['nitrile_hydrolysis'])

        # Get reaction products
        products = get_reaction_family_products(rxn)

        # Check if the generated products match expected ones
        product_smiles = [p.to_smiles() for p in products[0]['products']]
        expected_product_smiles = {acid_smiles}

        self.assertEqual(set(product_smiles), expected_product_smiles)

    def test_recipe_actions(self):
        """Test if the reaction recipe is applied correctly for nitrile hydrolysis."""
        # Load the groups.py file content as lines
        groups_file_path = '/home/leen/Code/ARC/data/families/nitrile_hydrolysis/groups.py'
        with open(groups_file_path, 'r') as f:
            groups_as_lines = f.readlines()

        # Get the recipe actions from the groups file content
        actions = get_recipe_actions(groups_as_lines)

        # Define the expected actions for nitrile hydrolysis
        expected_actions =[
        ['CHANGE_BOND', '*1', -1, '*2'],
        ['BREAK_BOND', '*3', 1, '*4'],
        ['FORM_BOND', '*1', 1, '*4'],
        ['FORM_BOND', '*2', 1, '*3'],
        ]

        # Assert that the actions match the expected actions
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
        # Define the imine and water as reactants
        imine_smiles = 'CC(=N)C'  
        water_smiles = 'O'

        # Create ARC species for the reactants
        imine = ARCSpecies(label='imine', smiles=imine_smiles)
        water = ARCSpecies(label='H2O', smiles=water_smiles)

        # Define the expected product (assuming a hydrolyzed imine product)
        acid_smiles = 'CC(O)(N)C'  # Example hydrolyzed product
        acid = ARCSpecies(label='acid', smiles=acid_smiles)

        # Create a reaction instance
        rxn = ARCReaction(r_species=[imine, water], p_species=[acid], rmg_family_set=['imine_hydrolysis'])

        # Get reaction products
        products = get_reaction_family_products(rxn)

        # Check if the generated products match expected ones
        product_smiles = [p.to_smiles() for p in products[0]['products']]
        expected_product_smiles = {acid_smiles}

        self.assertEqual(set(product_smiles), expected_product_smiles)

    def test_recipe_actions(self):
        """Test if the reaction recipe is applied correctly for imine hydrolysis."""
        # Load the groups.py file content as lines
        groups_file_path = '/home/leen/Code/ARC/data/families/imine_hydrolysis/groups.py'
        with open(groups_file_path, 'r') as f:
            groups_as_lines = f.readlines()

        # Get the recipe actions from the groups file content
        actions = get_recipe_actions(groups_as_lines)

        # Define the expected actions for imine hydrolysis
        expected_actions = [
            ['CHANGE_BOND', '*1', -1, '*2'],
            ['BREAK_BOND', '*3', 1, '*4'],
            ['FORM_BOND', '*1', 1, '*4'],
            ['FORM_BOND', '*2', 1, '*3'],
        ]

        # Assert that the actions match the expected actions
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
        # Define the ether and water as reactants
        ether_smiles = 'CCOC'  # Dimethyl ether
        water_smiles = 'O'

        # Create ARC species for the reactants
        ether = ARCSpecies(label='ether', smiles=ether_smiles)
        water = ARCSpecies(label='H2O', smiles=water_smiles)

        # Define the expected products 
        acid_smiles = 'CCO'
        alcohol_smiles = 'CO'

        acid = ARCSpecies(label='acid', smiles=acid_smiles)
        alcohol = ARCSpecies(label='alcohol', smiles=alcohol_smiles)

        # Create a reaction instance
        rxn = ARCReaction(r_species=[ether, water], p_species=[acid, alcohol], rmg_family_set=['ether_hydrolysis'])

        # Get reaction products
        products = get_reaction_family_products(rxn)

        # Check if the generated products match expected ones
        product_smiles = [p.to_smiles() for p in products[0]['products']]
        expected_product_smiles = {acid_smiles, alcohol_smiles}

        self.assertEqual(set(product_smiles), expected_product_smiles)

    def test_recipe_actions(self):
        """Test if the reaction recipe is applied correctly."""
        # Load the groups.py file content as lines
        groups_file_path = '/home/leen/Code/ARC/data/families/ether_hydrolysis/groups.py'
        with open(groups_file_path, 'r') as f:
            groups_as_lines = f.readlines()

        # Get the recipe actions from the groups file content
        actions = get_recipe_actions(groups_as_lines)

        # Define the expected actions
        expected_actions = [
        ['BREAK_BOND', '*1', 1, '*2'],
        ['BREAK_BOND', '*3', 1, '*4'],
        ['FORM_BOND', '*1', 1, '*4'],
        ['FORM_BOND', '*2', 1, '*3'],
        ]

        # Assert that the actions match the expected actions
        self.assertEqual(actions, expected_actions)


if __name__ == '__main__':
    unittest.main()
