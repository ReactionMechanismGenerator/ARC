import unittest
from rmgpy.molecule import Molecule
from arc.reaction.family import ReactionFamily, get_reaction_family_products
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
        product_smiles = [p.molecule[0].to_smiles() for p in products[0]['products']]
        expected_product_smiles = {acid_smiles, alcohol_smiles}

        self.assertEqual(set(product_smiles), expected_product_smiles)
    """
    def test_reverse_reaction(self):
        # Define the acid and alcohol as reactants for the reverse reaction
        acid_smiles = 'CC(=O)O'
        alcohol_smiles = 'CO'

        acid = ARCSpecies(label='acid', smiles=acid_smiles)
        alcohol = ARCSpecies(label='alcohol', smiles=alcohol_smiles)

        # Define expected ester and water as products
        ester_smiles = 'CC(=O)OC'
        water_smiles = 'O'

        ester = ARCSpecies(label='ester', smiles=ester_smiles)
        water = ARCSpecies(label='H2O', smiles=water_smiles)

        # Create a reaction instance for the reverse reaction
        rxn_reverse = ARCReaction(r_species=[acid, alcohol], p_species=[ester, water], rmg_family_set=['condensation'])

        # Get reaction products
        products_reverse = get_reaction_family_products(rxn_reverse)

        # Check if the generated products match expected ones
        reverse_product_smiles = [p.molecule[0].to_smiles() for p in products_reverse[0]['products']]
        expected_reverse_product_smiles = {ester_smiles, water_smiles}

        self.assertEqual(set(reverse_product_smiles), expected_reverse_product_smiles)
    """

    def test_recipe_actions(self):
        """Test if the reaction recipe is applied correctly."""
        expected_actions = [
            ['BREAK_BOND', '*1', 1, '*2'],
            ['BREAK_BOND', '*3', 1, '*4'],
            ['FORM_BOND', '*1', 1, '*4'],
            ['FORM_BOND', '*2', 1, '*3'],
        ]
        actions = self.family.recipe.actions  
        self.assertEqual(actions, expected_actions)

    """def test_reversibility(self):
        self.assertTrue(self.family.reversible)
        self.assertEqual(self.family.reverse, "condensation")
    """
    

    def test_recognize_hydrolysis_family(self):
        """Test if the system recognizes a hydrolysis reaction as part of the ester hydrolysis family."""
        # Define the ester and water as reactants
        ester_smiles = 'CC(=O)OC'  # Methyl acetate
        water_smiles = 'O'

        # Define expected products for hydrolysis (acetic acid and methanol)
        acid_smiles = 'CC(=O)O'
        alcohol_smiles = 'CO'

        # Create ARC species for reactants and expected products
        ester = ARCSpecies(label='ester', smiles=ester_smiles)
        water = ARCSpecies(label='H2O', smiles=water_smiles)
        acid = ARCSpecies(label='acid', smiles=acid_smiles)
        alcohol = ARCSpecies(label='alcohol', smiles=alcohol_smiles)

        # Define a reaction with reactants and products
        rxn = ARCReaction(r_species=[ester, water], p_species=[acid, alcohol])

        # Determine if the system identifies it as a hydrolysis reaction
        family_detected = rxn.determine_family()  # Should return 'ester_hydrolysis' or similar identifier

        # Assert that the detected family matches ester hydrolysis
        self.assertEqual(family_detected, 'ester_hydrolysis')

if __name__ == '__main__':
    unittest.main()
