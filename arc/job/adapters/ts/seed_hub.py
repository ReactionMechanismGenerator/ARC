"""
Shared TS-seed and wrapper-constraint hub.

This module centralizes:
1. How TS seeds are requested from a base TS-search adapter.
2. How wrapper adapters (e.g., CREST) request family-specific constraints for a seed.
"""

from typing import Dict, List, Optional

from arc.common import get_logger
from arc.species.converter import xyz_to_dmat

logger = get_logger()


def get_ts_seeds(reaction: 'ARCReaction',
                 base_adapter: str = 'heuristics',
                 dihedral_increment: Optional[int] = None,
                 ) -> List[dict]:
    """
    Return TS seed entries from a base TS-search adapter.

    Seed schema:
        - ``xyz`` (dict): Cartesian coordinates.
        - ``family`` (str): The family associated with this seed.
        - ``method`` (str): Human-readable generator label.
        - ``source_adapter`` (str): Adapter id that generated the seed.
        - ``metadata`` (dict, optional): Adapter-specific auxiliary fields.

    Args:
        reaction: The ARC reaction object.
        base_adapter: The underlying TS-search adapter providing seeds.
        dihedral_increment: Optional scan increment used by adapters that support it.
    """
    adapter = (base_adapter or '').lower()
    if adapter != 'heuristics':
        raise ValueError(f'Unsupported TS seed base adapter: {base_adapter}')

    # Lazily import to avoid circular imports with heuristics.py.
    from arc.job.adapters.ts.heuristics import FAMILY_SETS, h_abstraction, hydrolysis

    xyz_entries = list()
    if reaction.family == 'H_Abstraction':
        xyzs = h_abstraction(reaction=reaction, dihedral_increment=dihedral_increment)
        for entry in xyzs:
            xyz = entry.get('xyz') if isinstance(entry, dict) else entry
            method = entry.get('method', 'Heuristics') if isinstance(entry, dict) else 'Heuristics'
            if xyz is not None:
                xyz_entries.append({
                    'xyz': xyz,
                    'method': method,
                    'family': reaction.family,
                    'source_adapter': 'heuristics',
                    'metadata': {},
                })
    elif reaction.family in FAMILY_SETS['hydrolysis_set_1'] or reaction.family in FAMILY_SETS['hydrolysis_set_2']:
        try:
            xyzs_raw, families, indices = hydrolysis(reaction=reaction)
            xyz_entries = [{
                'xyz': xyz,
                'method': 'Heuristics',
                'family': family,
                'source_adapter': 'heuristics',
                'metadata': {'indices': idx},
            } for xyz, family, idx in zip(xyzs_raw, families, indices)]
        except ValueError:
            xyz_entries = list()
    return xyz_entries


def get_wrapper_constraints(wrapper: str,
                            reaction: 'ARCReaction',
                            seed: dict,
                            ) -> Optional[dict]:
    """
    Return wrapper-specific constraints for a TS seed.

    Args:
        wrapper: Wrapper adapter id (e.g., ``crest``).
        reaction: The ARC reaction object.
        seed: A seed entry returned by :func:`get_ts_seeds`.
    """
    wrapper_name = (wrapper or '').lower()
    if wrapper_name != 'crest':
        raise ValueError(f'Unsupported wrapper adapter: {wrapper}')
    return _get_crest_constraints(reaction=reaction, seed=seed)


def _get_crest_constraints(reaction: 'ARCReaction', seed: dict) -> Optional[Dict[str, int]]:
    """
    Return CREST constraints for a seed.

    Currently, only H_Abstraction is supported.
    """
    family = seed.get('family') or reaction.family
    xyz = seed.get('xyz')
    if family != 'H_Abstraction' or xyz is None:
        return None
    return _get_h_abs_atoms_from_xyz(xyz)


def _get_h_abs_atoms_from_xyz(xyz: dict) -> Optional[Dict[str, int]]:
    """
    Determine H-abstraction atoms from a TS guess.

    Returns:
        Optional[Dict[str, int]]: ``{'H': int, 'A': int, 'B': int}``, or ``None``.
    """
    symbols = xyz.get('symbols') if isinstance(xyz, dict) else None
    if not symbols:
        return None
    dmat = xyz_to_dmat(xyz)
    if dmat is None:
        return None

    closest_atoms = dict()
    for i in range(len(symbols)):
        nearest = sorted(
            ((dmat[i][j], j) for j in range(len(symbols)) if j != i),
            key=lambda x: x[0],
        )[:2]
        closest_atoms[i] = [idx for _, idx in nearest]

    hydrogen_indices = [i for i, symbol in enumerate(symbols) if symbol.startswith('H')]
    condition_occurrences = list()

    for hydrogen_index in hydrogen_indices:
        atom_neighbors = closest_atoms[hydrogen_index]
        is_heavy_present = any(not symbols[atom].startswith('H') for atom in atom_neighbors)
        if_hydrogen_present = any(symbols[atom].startswith('H') and atom != hydrogen_index for atom in atom_neighbors)

        if is_heavy_present and if_hydrogen_present:
            condition_occurrences.append({'H': hydrogen_index, 'A': atom_neighbors[0], 'B': atom_neighbors[1]})

    if condition_occurrences:
        if len(condition_occurrences) > 1:
            occurrence_distances = list()
            for occurrence in condition_occurrences:
                h_atom = occurrence['H']
                a_atom = occurrence['A']
                b_atom = occurrence['B']
                occurrence_distances.append((occurrence, dmat[h_atom][a_atom] + dmat[h_atom][b_atom]))
            best_occurrence = min(occurrence_distances, key=lambda x: x[1])[0]
            return {'H': best_occurrence['H'], 'A': best_occurrence['A'], 'B': best_occurrence['B']}
        single_occurrence = condition_occurrences[0]
        return {'H': single_occurrence['H'], 'A': single_occurrence['A'], 'B': single_occurrence['B']}

    min_distance = float('inf')
    selected_hydrogen = None
    selected_heavy_atoms = None
    for hydrogen_index in hydrogen_indices:
        atom_neighbors = closest_atoms[hydrogen_index]
        heavy_atoms = [atom for atom in atom_neighbors if not symbols[atom].startswith('H')]
        if len(heavy_atoms) < 2:
            continue
        distances = dmat[hydrogen_index][heavy_atoms[0]] + dmat[hydrogen_index][heavy_atoms[1]]
        if distances < min_distance:
            min_distance = distances
            selected_hydrogen = hydrogen_index
            selected_heavy_atoms = heavy_atoms

    if selected_hydrogen is not None and selected_heavy_atoms is not None:
        return {'H': selected_hydrogen, 'A': selected_heavy_atoms[0], 'B': selected_heavy_atoms[1]}

    logger.warning('No valid hydrogen atom found for CREST H-abstraction atoms.')
    return None

