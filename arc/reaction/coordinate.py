"""
Module for identifying and handling reaction coordinates for constraint scans.
"""

from typing import Optional, Dict, TYPE_CHECKING

from arc.common import get_logger

if TYPE_CHECKING:
    from arc.reaction import ARCReaction

logger = get_logger()


def identify_h_abstraction_coordinate(rxn: 'ARCReaction') -> Optional[Dict]:
    """
    Identify the reaction coordinate for an H-abstraction reaction.
    
    For H-abstraction: R-H + X· → R· + H-X
    Need to identify:
    - Donor atom (R)
    - Hydrogen atom (H)
    - Acceptor atom (X)
    
    Args:
        rxn: ARCReaction object
        
    Returns:
        Dict with keys: 'donor_idx', 'hydrogen_idx', 'acceptor_idx'
        None if coordinate cannot be identified
    """
    if rxn.family is None:
        return None
    
    # Check if this is H-abstraction family
    if 'H_Abstraction' not in rxn.family:
        return None

    formed_bonds, broken_bonds = rxn.get_formed_and_broken_bonds()

    if len(formed_bonds) != 1 or len(broken_bonds) != 1:
        logger.warning(f"Expected 1 formed and 1 broken bond for H_Abstraction, but got {len(formed_bonds)} formed and {len(broken_bonds)} broken bonds for reaction {rxn.label}.")
        return None

    breaking_bond = broken_bonds[0]  # (R_idx, H_idx)
    forming_bond = formed_bonds[0]  # (H_idx, X_idx)

    # Find the common atom (hydrogen)
    hydrogen_idx = None
    if breaking_bond[0] in forming_bond:
        hydrogen_idx = breaking_bond[0]
    elif breaking_bond[1] in forming_bond:
        hydrogen_idx = breaking_bond[1]

    if hydrogen_idx is None:
        logger.warning(f"Could not identify the transferring hydrogen atom for H_Abstraction in reaction {rxn.label}.")
        return None

    # Identify donor (R) and acceptor (X)
    donor_idx = breaking_bond[0] if breaking_bond[1] == hydrogen_idx else breaking_bond[1]
    acceptor_idx = forming_bond[0] if forming_bond[1] == hydrogen_idx else forming_bond[1]

    # Verify donor is bonded to H in reactants
    # Iterate through reactant species to find the one containing the hydrogen atom
    r_species_with_h = None
    for r_spc in rxn.r_species:
        # This is a heuristic check: look for an atom with the same index if indices are preserved,
        # or we rely on the atom map.
        # Since we are working with mapped indices, we should use the reaction's reactant/product checks.
        # A more robust way is to check if the bond (donor_idx, hydrogen_idx) exists in the reactant bonds list.
        pass

    r_bonds, _ = rxn.get_bonds()
    # r_bonds contains tuples of sorted atom indices (0-indexed) present in reactants.
    
    bond_dh = tuple(sorted((donor_idx, hydrogen_idx)))
    bond_ah = tuple(sorted((acceptor_idx, hydrogen_idx)))
    
    donor_bonded = bond_dh in r_bonds
    acceptor_bonded = bond_ah in r_bonds
    
    if acceptor_bonded and not donor_bonded:
        logger.warning(f"Identified Donor {donor_idx} and Acceptor {acceptor_idx} seem swapped based on reactant connectivity. Swapping them.")
        donor_idx, acceptor_idx = acceptor_idx, donor_idx
    elif not donor_bonded and not acceptor_bonded:
        logger.warning(f"Neither identified Donor {donor_idx} nor Acceptor {acceptor_idx} seem bonded to Hydrogen {hydrogen_idx} in reactants. Check atom mapping.")

    logger.info(f"H-Abstraction Coordinate Identification for {rxn.label}:")
    logger.info(f"  Hydrogen Atom Index: {hydrogen_idx + 1} (0-indexed: {hydrogen_idx})")
    logger.info(f"  Donor Atom Index (Breaking Bond): {donor_idx + 1} (0-indexed: {donor_idx})")
    logger.info(f"  Acceptor Atom Index (Forming Bond): {acceptor_idx + 1} (0-indexed: {acceptor_idx})")

    return {
        'donor_idx': donor_idx,
        'hydrogen_idx': hydrogen_idx,
        'acceptor_idx': acceptor_idx,
    }


def identify_reaction_coordinate(rxn: 'ARCReaction') -> Optional[Dict]:
    """
    Identify the reaction coordinate for constraint scan.
    Dispatches to family-specific methods.
    
    Args:
        rxn: ARCReaction object
        
    Returns:
        Dict with keys depending on reaction type:
        - For H-abstraction: 'donor_idx', 'hydrogen_idx', 'acceptor_idx', 'type'
        - For other reactions: TBD
        None if coordinate cannot be identified
    """
    if rxn.family is None:
        logger.debug('Reaction has no family assigned, cannot identify coordinate')
        return None
    
    family_name = rxn.family
    
    if 'H_Abstraction' in family_name:
        coord = identify_h_abstraction_coordinate(rxn)
        if coord:
            coord['type'] = 'H_Abstraction'
        return coord
    
    # TODO: Add support for other reaction families:
    # - Addition reactions
    # - Elimination reactions
    # - Substitution reactions
    # etc.
    
    logger.warning(f'Reaction coordinate identification not implemented for family: {family_name}')
    return None


def create_constraint_scan_input(coord_info: Dict, 
                                 nsteps: int = 25, 
                                 stepsize: float = 0.05,
                                 scan_type: str = 'distance') -> str:
    """
    Create the ModRedundant constraint block for Gaussian input.
    
    Args:
        coord_info: Dictionary with atom indices (donor_idx, hydrogen_idx, acceptor_idx)
        nsteps: Number of scan steps
        stepsize: Step size in Angstroms
        scan_type: 'distance' to scan D-H distance, 'difference' for coordinate difference
        
    Returns:
        String with ModRedundant constraint commands
    """
    donor = coord_info.get('donor_idx')
    hydrogen = coord_info.get('hydrogen_idx')
    acceptor = coord_info.get('acceptor_idx')
    
    if donor is None or hydrogen is None or acceptor is None:
        raise ValueError('Missing required atom indices in coord_info')
    
    # Gaussian uses 1-based indexing
    donor += 1
    hydrogen += 1
    acceptor += 1
    
    constraints = []
    
    if scan_type == 'distance':
        # Option A: Freeze A-H, scan D-H
        constraints.append(f'B {acceptor} {hydrogen} F')  # Freeze forming bond
        constraints.append(f'B {donor} {hydrogen} S {nsteps} {stepsize:.3f}')  # Scan breaking bond
        
    elif scan_type == 'difference':
        # Option B: Scan the difference coordinate D = B1 - B2
        # This is more complex and may require custom coordinate definition
        # For now, not fully implemented
        logger.warning('Difference coordinate scan not yet fully implemented, using distance scan')
        constraints.append(f'B {acceptor} {hydrogen} F')
        constraints.append(f'B {donor} {hydrogen} S {nsteps} {stepsize:.3f}')
    
    return '\n\n' + '\n'.join(constraints) + '\n'
