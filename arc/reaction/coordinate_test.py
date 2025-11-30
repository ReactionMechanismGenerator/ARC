"""
Unit tests for the coordinate module
"""

import unittest
from typing import Dict, List, Optional, Tuple, Union

from arc.reaction.coordinate import (
    identify_reaction_coordinate,
    identify_h_abstraction_coordinate,
    create_constraint_scan_input,
)

# Mock classes to simulate ARCReaction, ARCSpecies, and Molecule for testing
class MockMolecule:
    def __init__(self, atoms=None, bonds=None):
        self.atoms = atoms if atoms is not None else []
        self._bonds = bonds if bonds is not None else {}

    def get_bond(self, atom1_idx, atom2_idx):
        return self._bonds.get(tuple(sorted((atom1_idx, atom2_idx))))

    def has_bond(self, atom1_idx, atom2_idx):
        return tuple(sorted((atom1_idx, atom2_idx))) in self._bonds


class MockARCSpecies:
    def __init__(self, label="mock_species", mol=None):
        self.label = label
        self.mol = mol


class MockReactionFamily:
    def __init__(self, label="H_Abstraction"):
        self.label = label


class MockARCReaction:
    def __init__(self, label="mock_reaction", family=None, atom_map=None, r_species=None, p_species=None, formed_bonds=None, broken_bonds=None):
        self.label = label
        self.family = family  # Directly assign, allows None
        self.atom_map = atom_map
        self.r_species = r_species if r_species is not None else []
        self.p_species = p_species if p_species is not None else []
        self._formed_bonds = formed_bonds
        self._broken_bonds = broken_bonds

    def get_formed_and_broken_bonds(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        return self._formed_bonds, self._broken_bonds


class TestCoordinate(unittest.TestCase):
    """
    Contains unit tests for the coordinate module.
    """

    def test_create_constraint_scan_input(self):
        """Test creating constraint scan input string."""
        coord_info = {
            'donor_idx': 0,
            'hydrogen_idx': 1,
            'acceptor_idx': 2,
            'type': 'H_Abstraction'
        }
        
        result = create_constraint_scan_input(
            coord_info=coord_info,
            nsteps=10,
            stepsize=0.1,
            scan_type='distance'
        )
        
        # Check that result contains expected Gaussian ModRedundant commands
        self.assertIn('B 3 2 F', result)  # Freeze acceptor-hydrogen (1-indexed)
        self.assertIn('B 1 2 S 10 0.100', result)  # Scan donor-hydrogen (1-indexed)
        self.assertIn('\n\n', result)  # Should start with blank line

    def test_create_constraint_scan_input_missing_indices(self):
        """Test that missing indices raise ValueError."""
        coord_info = {
            'donor_idx': 0,
            'hydrogen_idx': None,  # Missing!
            'acceptor_idx': 2,
        }
        
        with self.assertRaises(ValueError):
            create_constraint_scan_input(
                coord_info=coord_info,
                nsteps=10,
                stepsize=0.1,
            )

    def test_identify_h_abstraction_coordinate_valid(self):
        """Test identifying coordinates for a valid H-abstraction."""
        # Simulate R-H + X. -> R. + H-X
        # Let's say H is atom 1, R is atom 0, X is atom 2 (0-indexed)
        # Breaking bond: (0, 1) -> R-H
        # Forming bond: (1, 2) -> H-X
        mock_rxn = MockARCReaction(
            family=MockReactionFamily(label="H_Abstraction"),
            formed_bonds=[(1, 2)],
            broken_bonds=[(0, 1)]
        )
        
        coord = identify_h_abstraction_coordinate(mock_rxn)
        
        self.assertIsNotNone(coord)
        self.assertEqual(coord['donor_idx'], 0)
        self.assertEqual(coord['hydrogen_idx'], 1)
        self.assertEqual(coord['acceptor_idx'], 2)

    def test_identify_h_abstraction_coordinate_non_h_abstraction_family(self):
        """Test with a non-H-abstraction family."""
        mock_rxn = MockARCReaction(
            family=MockReactionFamily(label="Other_Reaction"),
            formed_bonds=[(1, 2)],
            broken_bonds=[(0, 1)]
        )
        
        coord = identify_h_abstraction_coordinate(mock_rxn)
        self.assertIsNone(coord)

    def test_identify_h_abstraction_coordinate_no_family(self):
        """Test with no family defined."""
        mock_rxn = MockARCReaction(
            family=None,
            formed_bonds=[(1, 2)],
            broken_bonds=[(0, 1)]
        )
        
        coord = identify_h_abstraction_coordinate(mock_rxn)
        self.assertIsNone(coord)

    def test_identify_h_abstraction_coordinate_incorrect_bond_count(self):
        """Test with incorrect number of formed/broken bonds."""
        # Too many formed bonds
        mock_rxn_1 = MockARCReaction(
            family=MockReactionFamily(label="H_Abstraction"),
            formed_bonds=[(1, 2), (3, 4)],
            broken_bonds=[(0, 1)]
        )
        coord_1 = identify_h_abstraction_coordinate(mock_rxn_1)
        self.assertIsNone(coord_1)

        # Too few broken bonds
        mock_rxn_2 = MockARCReaction(
            family=MockReactionFamily(label="H_Abstraction"),
            formed_bonds=[(1, 2)],
            broken_bonds=[]
        )
        coord_2 = identify_h_abstraction_coordinate(mock_rxn_2)
        self.assertIsNone(coord_2)

    def test_identify_h_abstraction_coordinate_hydrogen_not_common(self):
        """Test case where transferring H is not common in bonds (shouldn't happen with correct mapping)."""
        mock_rxn = MockARCReaction(
            family=MockReactionFamily(label="H_Abstraction"),
            formed_bonds=[(3, 4)],  # H not in breaking bond
            broken_bonds=[(0, 1)]
        )
        
        coord = identify_h_abstraction_coordinate(mock_rxn)
        self.assertIsNone(coord)

    def test_identify_reaction_coordinate_h_abstraction(self):
        """Test identify_reaction_coordinate for H_Abstraction."""
        mock_rxn = MockARCReaction(
            family=MockReactionFamily(label="H_Abstraction"),
            formed_bonds=[(1, 2)],
            broken_bonds=[(0, 1)]
        )
        
        coord = identify_reaction_coordinate(mock_rxn)
        
        self.assertIsNotNone(coord)
        self.assertEqual(coord['donor_idx'], 0)
        self.assertEqual(coord['hydrogen_idx'], 1)
        self.assertEqual(coord['acceptor_idx'], 2)
        self.assertEqual(coord['type'], 'H_Abstraction')

    def test_identify_reaction_coordinate_no_family(self):
        """Test identify_reaction_coordinate with no family."""
        mock_rxn = MockARCReaction(
            family=None,
            formed_bonds=[(1, 2)],
            broken_bonds=[(0, 1)]
        )
        
        coord = identify_reaction_coordinate(mock_rxn)
        self.assertIsNone(coord)

    def test_identify_reaction_coordinate_unsupported_family(self):
        """Test identify_reaction_coordinate with an unsupported family."""
        mock_rxn = MockARCReaction(
            family=MockReactionFamily(label="Unsupported_Family"),
            formed_bonds=[(1, 2)],
            broken_bonds=[(0, 1)]
        )
        
        coord = identify_reaction_coordinate(mock_rxn)
        self.assertIsNone(coord)


if __name__ == '__main__':
    unittest.main()
