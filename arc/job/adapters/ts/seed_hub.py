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
                entry_metadata = entry.get('metadata') if isinstance(entry, dict) else None
                metadata = entry_metadata.copy() if isinstance(entry_metadata, dict) else {}
                if 'reactive_atoms' not in metadata:
                    reactive_atoms = _get_h_abs_atoms_from_xyz(xyz)
                    if reactive_atoms is not None:
                        metadata['reactive_atoms'] = reactive_atoms
                xyz_entries.append({
                    'xyz': xyz,
                    'method': method,
                    'family': reaction.family,
                    'source_adapter': 'heuristics',
                    'metadata': metadata,
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
    metadata = seed.get('metadata')
    if isinstance(metadata, dict) and 'reactive_atoms' in metadata:
        reactive_atoms = metadata['reactive_atoms']
        if _is_valid_h_abs_atom_assignment(xyz=xyz, atoms=reactive_atoms):
            return reactive_atoms
        logger.warning(f'Invalid explicit CREST H-abstraction atom assignment: {reactive_atoms}')
        return None
    return _get_h_abs_atoms_from_xyz(xyz)


def _is_valid_h_abs_atom_assignment(xyz: dict, atoms: Optional[Dict[str, int]]) -> bool:
    """Return whether ``atoms`` identifies a heavy-atom--H--heavy-atom triad in ``xyz``."""
    symbols = xyz.get('symbols') if isinstance(xyz, dict) else None
    if not symbols or not isinstance(atoms, dict) or set(atoms) != {'A', 'H', 'B'}:
        return False
    if any(not isinstance(atoms[key], int) or not 0 <= atoms[key] < len(symbols) for key in atoms):
        return False
    return (symbols[atoms['H']].startswith('H')
            and not symbols[atoms['A']].startswith('H')
            and not symbols[atoms['B']].startswith('H')
            and atoms['A'] != atoms['B'])


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

    hydrogen_indices = [i for i, symbol in enumerate(symbols) if symbol.startswith('H')]
    min_distance = float('inf')
    selected_hydrogen = None
    selected_heavy_atoms = None
    for hydrogen_index in hydrogen_indices:
        heavy_atoms = sorted(
            (atom for atom, symbol in enumerate(symbols)
             if atom != hydrogen_index and not symbol.startswith('H')),
            key=lambda atom: dmat[hydrogen_index][atom],
        )[:2]
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
