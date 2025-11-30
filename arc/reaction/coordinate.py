"""
Module for identifying and handling reaction coordinates for constraint scans.
"""

from typing import Optional, Dict, TYPE_CHECKING, Union

import numpy as np
from arc.common import get_logger
from arc.species.converter import check_xyz_dict

if TYPE_CHECKING:
    from arc.reaction import ARCReaction

logger = get_logger()


def identify_h_abstraction_from_xyz(xyz: dict) -> Optional[Dict]:
    """
    Identify H-abstraction atoms (H, A, B) purely from geometry.
    Finds a Hydrogen atom that has exactly two heavy atom neighbors within bonding distance.
    
    Args:
        xyz: ARC xyz dictionary
        
    Returns:
        Dict with keys 'hydrogen_idx', 'heavy1_idx', 'heavy2_idx' or None.
        Note: Does NOT distinguish Donor vs Acceptor.
    """
    if not xyz or 'coords' not in xyz or 'symbols' not in xyz:
        return None
        
    coords = np.array(xyz['coords'])
    symbols = xyz['symbols']
    num_atoms = len(symbols)
    
    if num_atoms < 3:
        return None
        
    # Simple distance matrix
    # (N, 3) - (N, 1, 3) -> (N, N, 3)
    delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(delta**2, axis=-1))
    
    # Threshold for "bonding" or "close interaction" in a TS
    # Normal CH bond is ~1.1, TS bond might be ~1.3-1.5.
    # Heavy-Heavy bond is > 1.4.
    # Let's check for H atoms having 2 heavy neighbors within ~1.6 Angstroms?
    # Or simply find the H with the *closest* two heavy atoms.
    
    candidates = []
    
    for i, sym in enumerate(symbols):
        if sym == 'H':
            # Find distances to heavy atoms
            heavy_indices = [j for j, s in enumerate(symbols) if s != 'H']
            if len(heavy_indices) < 2:
                continue
                
            # Get distances to all heavy atoms
            h_dists = [(dists[i, j], j) for j in heavy_indices]
            h_dists.sort(key=lambda x: x[0])
            
            # Check the two closest heavy atoms
            d1, idx1 = h_dists[0]
            d2, idx2 = h_dists[1]
            
            # If both are reasonably close (e.g. < 2.0 Angstroms for a TS), this is a candidate
            if d1 < 2.0 and d2 < 2.0:
                candidates.append({
                    'hydrogen_idx': i,
                    'heavy1_idx': idx1,
                    'heavy2_idx': idx2,
                    'sum_dist': d1 + d2
                })
    
    if not candidates:
        return None
        
    # Return candidate with smallest sum of bond lengths (likely the active site)
    best = min(candidates, key=lambda x: x['sum_dist'])
    return best


def identify_h_abstraction_coordinate(rxn: 'ARCReaction', xyz: Optional[dict] = None) -> Optional[Dict]:
    """
    Identify the reaction coordinate for an H-abstraction reaction.
    
    Args:
        rxn: ARCReaction object
        xyz: Optional xyz dictionary of the TS guess. If provided, geometric identification 
             will be attempted first.
    """
    donor_idx = None
    hydrogen_idx = None
    acceptor_idx = None
    
    # 1. Try Geometry-based identification if xyz is available
    if xyz:
        geo_info = identify_h_abstraction_from_xyz(xyz)
        if geo_info:
            logger.info(f"Geometry-based identification found H atom {geo_info['hydrogen_idx']} "
                        f"between heavy atoms {geo_info['heavy1_idx']} and {geo_info['heavy2_idx']}")
            hydrogen_idx = geo_info['hydrogen_idx']
            # We don't know which is donor/acceptor yet, assign tentatively
            donor_idx = geo_info['heavy1_idx']
            acceptor_idx = geo_info['heavy2_idx']
    
    # 2. If no geometry match (or no xyz), fall back to Graph-based identification
    if hydrogen_idx is None:
        if rxn.family is None or 'H_Abstraction' not in rxn.family:
            return None

        formed_bonds, broken_bonds = rxn.get_formed_and_broken_bonds()

        if len(formed_bonds) != 1 or len(broken_bonds) != 1:
            logger.warning(f"Expected 1 formed and 1 broken bond for H_Abstraction...")
            return None

        breaking_bond = broken_bonds[0]
        forming_bond = formed_bonds[0]

        if breaking_bond[0] in forming_bond:
            hydrogen_idx = breaking_bond[0]
        elif breaking_bond[1] in forming_bond:
            hydrogen_idx = breaking_bond[1]

        if hydrogen_idx is None:
            return None

        donor_idx = breaking_bond[0] if breaking_bond[1] == hydrogen_idx else breaking_bond[1]
        acceptor_idx = forming_bond[0] if forming_bond[1] == hydrogen_idx else forming_bond[1]

    # 3. Robust verification & Assignment using Reactant/Product connectivity
    # This step is CRITICAL to assign/correct Donor vs Acceptor roles
    r_bonds, p_bonds = rxn.get_bonds()
    
    bond_dh = tuple(sorted((donor_idx, hydrogen_idx)))
    bond_ah = tuple(sorted((acceptor_idx, hydrogen_idx)))
    
    dh_in_r = bond_dh in r_bonds
    dh_in_p = bond_dh in p_bonds
    ah_in_r = bond_ah in r_bonds
    ah_in_p = bond_ah in p_bonds
    
    logger.info(f"Connectivity Check:"
                f"\n  Donor-H ({donor_idx}-{hydrogen_idx}): In Reactants? {dh_in_r}, In Products? {dh_in_p}"
                f"\n  Acceptor-H ({acceptor_idx}-{hydrogen_idx}): In Reactants? {ah_in_r}, In Products? {ah_in_p}")

    # Logic: True Donor is Bonded in R, Not in P. True Acceptor is Not in R, Bonded in P.
    
    # Case 1: Swapped (Current Donor is actually Acceptor)
    if (not dh_in_r and dh_in_p) and (ah_in_r and not ah_in_p):
        logger.warning(f"Swapping Donor/Acceptor based on R/P connectivity logic.")
        donor_idx, acceptor_idx = acceptor_idx, donor_idx
    
    # Case 2: Both bonded in R? (Ambiguous input, e.g. Water reactant)
    elif dh_in_r and ah_in_r:
        # The one that is NOT bonded in products is the true Donor (bond broken).
        if dh_in_p and not ah_in_p:
             logger.warning(f"Swapping Donor/Acceptor: Donor bond persists in products, Acceptor bond lost.")
             donor_idx, acceptor_idx = acceptor_idx, donor_idx
             
    # Case 3: Standard correction if we just guessed wrong from geometry
    elif (not dh_in_r) and (ah_in_r):
         logger.warning(f"Swapping Donor/Acceptor: Identified Donor not bonded in Reactants.")
         donor_idx, acceptor_idx = acceptor_idx, donor_idx

    logger.info(f"H-Abstraction Coordinate Identification for {rxn.label}:")
    logger.info(f"  Hydrogen Atom Index: {hydrogen_idx + 1} (0-indexed: {hydrogen_idx})")
    logger.info(f"  Donor Atom Index (Breaking Bond): {donor_idx + 1} (0-indexed: {donor_idx})")
    logger.info(f"  Acceptor Atom Index (Forming Bond): {acceptor_idx + 1} (0-indexed: {acceptor_idx})")

    return {
        'donor_idx': donor_idx,
        'hydrogen_idx': hydrogen_idx,
        'acceptor_idx': acceptor_idx,
    }


def identify_reaction_coordinate(rxn: 'ARCReaction', xyz: Optional[dict] = None) -> Optional[Dict]:
    """
    Identify the reaction coordinate for constraint scan.
    Dispatches to family-specific methods.
    
    Args:
        rxn: ARCReaction object
        xyz: Optional xyz dictionary (e.g. from TS guess)
        
    Returns:
        Dict with keys depending on reaction type:
        - For H-abstraction: 'donor_idx', 'hydrogen_idx', 'acceptor_idx', 'type'
        None if coordinate cannot be identified
    """
    if rxn.family is None:
        logger.debug('Reaction has no family assigned, cannot identify coordinate')
        return None
    
    family_name = rxn.family
    
    if 'H_Abstraction' in family_name:
        coord = identify_h_abstraction_coordinate(rxn, xyz)
        if coord:
            coord['type'] = 'H_Abstraction'
        return coord
    
    # TODO: Add support for other reaction families
    
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
