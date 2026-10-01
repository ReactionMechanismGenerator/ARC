"""
Shared TS-seed and wrapper-constraint hub.

This module centralizes:
1. How TS seeds are requested from a base TS-search adapter.
2. How wrapper adapters (e.g., CREST) request family-specific constraints for a seed.
"""

from typing import Dict, List, Optional

from arc.common import almost_equal_coords, get_logger
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
    elif reaction.family == 'XY_Addition_MultipleBond':
        # Lazily import to keep the family-specific builder decoupled from this hub.
        from arc.job.adapters.ts.xy_addition import xy_addition
        for entry in xy_addition(reaction=reaction):
            xyz_entries.append({
                'xyz': entry['xyz'],
                'method': entry.get('method', 'Heuristics-XY'),
                'family': reaction.family,
                'source_adapter': 'heuristics',
                'metadata': entry.get('metadata', {}).copy(),
            })
    return xyz_entries


def get_backup_ts_seeds(reaction: 'ARCReaction',
                        exclude_method: str = 'crest',
                        ) -> List[dict]:
    """
    Build CREST seed entries from TS guesses that OTHER adapters already produced.

    This is a fallback for when CREST's own heuristic seed construction
    (:func:`get_ts_seeds`) yields nothing -- e.g. a linear/cumulene reactive center
    (such as HCCO in H_Abstraction) that the heuristic Z-matrix builder cannot
    assemble. Any successful non-CREST TS guess already present on
    ``reaction.ts_species.ts_guesses`` is a valid CREST seed: CREST only needs a seed
    geometry plus the family reactive-atom constraints, and the constraints are
    re-derived from the seed geometry by :func:`get_wrapper_constraints` -- they do not
    depend on how the seed geometry was originally built.

    Seeds are returned with empty ``metadata`` so the wrapper-constraint derivation
    re-infers the reactive atoms from the geometry itself (robust to the source
    adapter's atom ordering). Guesses whose method contains ``exclude_method`` are
    skipped so CREST is never seeded from a prior CREST result (feedback-loop guard).

    Args:
        reaction: The ARC reaction object.
        exclude_method: A method substring to exclude (default ``'crest'``).

    Returns:
        List[dict]: Seed entries in the same schema as :func:`get_ts_seeds`.
    """
    ts_species = getattr(reaction, 'ts_species', None)
    ts_guesses = getattr(ts_species, 'ts_guesses', None) or list()
    exclude = (exclude_method or '').lower()
    seeds = list()
    seen_xyzs = list()
    for tsg in ts_guesses:
        method = (getattr(tsg, 'method', '') or '').lower()
        if not getattr(tsg, 'success', False):
            continue
        if exclude and exclude in method:
            continue
        xyz = getattr(tsg, 'opt_xyz', None) or getattr(tsg, 'initial_xyz', None)
        if not isinstance(xyz, dict) or not xyz.get('symbols'):
            continue
        if any(almost_equal_coords(xyz, seen) for seen in seen_xyzs):
            continue
        seen_xyzs.append(xyz)
        seeds.append({
            'xyz': xyz,
            'method': getattr(tsg, 'method', None) or 'external',
            'family': reaction.family,
            'source_adapter': method or 'external',
            'metadata': {},
        })
    return seeds


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


def _get_crest_constraints(reaction: 'ARCReaction', seed: dict) -> Optional[dict]:
    """
    Return a generic CREST constraint specification for a seed.

    The specification contains zero-based participating ``atoms`` and ``distance_pairs``.
    H-abstraction and the intramolecular NO2 -> ONO conversion additionally supply
    ``angle_atoms`` so completed geometries retain the seed's three-center orientation
    (heavy-atom--H--heavy-atom, or C--N--O for the migrating nitro oxygen).
    """
    family = seed.get('family') or reaction.family
    xyz = seed.get('xyz')
    if xyz is None:
        return None
    metadata = seed.get('metadata')
    explicit_atoms = metadata.get('reactive_atoms') if isinstance(metadata, dict) else None
    if family == 'H_Abstraction':
        reactive_atoms = explicit_atoms if explicit_atoms is not None else _get_h_abs_atoms_from_xyz(xyz)
        if _is_valid_h_abs_atom_assignment(xyz=xyz, atoms=reactive_atoms):
            return {
                'A': reactive_atoms['A'],
                'H': reactive_atoms['H'],
                'B': reactive_atoms['B'],
                'atoms': tuple(reactive_atoms[key] for key in ('A', 'H', 'B')),
                'distance_pairs': (
                    (reactive_atoms['A'], reactive_atoms['H']),
                    (reactive_atoms['H'], reactive_atoms['B']),
                ),
                'angle_atoms': tuple(reactive_atoms[key] for key in ('A', 'H', 'B')),
            }
        if explicit_atoms is not None:
            logger.warning(f'Invalid explicit CREST H-abstraction atom assignment: {explicit_atoms}')
        return None
    if family == 'XY_Addition_MultipleBond':
        if _is_valid_xy_atom_assignment(xyz=xyz, atoms=explicit_atoms):
            return {
                'atoms': tuple(explicit_atoms[label] for label in ('*1', '*2', '*3', '*4')),
                'distance_pairs': (
                    (explicit_atoms['*1'], explicit_atoms['*3']),
                    (explicit_atoms['*2'], explicit_atoms['*4']),
                    (explicit_atoms['*3'], explicit_atoms['*4']),
                ),
            }
        logger.warning(f'Invalid explicit CREST XY-addition atom assignment: {explicit_atoms}')
    if family == 'intra_NO2_ONO_conversion':
        reactive_atoms = _normalize_no2_ono_atoms(explicit_atoms)
        if reactive_atoms is None:
            reactive_atoms = _get_no2_ono_atoms_from_reaction(reaction)
        if _is_valid_no2_ono_atom_assignment(xyz=xyz, atoms=reactive_atoms):
            return {
                'C': reactive_atoms['C'],
                'N': reactive_atoms['N'],
                'O': reactive_atoms['O'],
                'atoms': tuple(reactive_atoms[role] for role in ('C', 'N', 'O')),
                'distance_pairs': (
                    (reactive_atoms['C'], reactive_atoms['N']),
                    (reactive_atoms['C'], reactive_atoms['O']),
                    (reactive_atoms['N'], reactive_atoms['O']),
                ),
                'angle_atoms': tuple(reactive_atoms[role] for role in ('C', 'N', 'O')),
            }
        logger.warning(f'Invalid CREST intra_NO2_ONO_conversion atom assignment: {reactive_atoms}')
    return None


def _get_no2_ono_atoms_from_reaction(reaction: 'ARCReaction') -> Optional[Dict[str, int]]:
    """
    Determine the ``intra_NO2_ONO_conversion`` recipe atoms from the RMG family label map.

    The backup-seed route (:func:`get_backup_ts_seeds`) reuses TS guesses generated by other
    adapters and therefore carries no generator metadata, so the recipe atoms are taken from
    the reaction's family label map instead. The family recipe is
    ``BREAK_BOND *1-*2`` / ``FORM_BOND *1-*3``, i.e. ``*1`` is the carbon losing the nitro
    group, ``*2`` is the nitrogen, and ``*3`` is the migrating oxygen. The label-map indices
    refer to the reaction's reactant atom order, which is also the TS-guess atom order.

    Args:
        reaction: The ARC reaction object.

    Returns:
        Optional[Dict[str, int]]: ``{'C': int, 'N': int, 'O': int}``, or ``None``.
    """
    try:
        product_dicts = getattr(reaction, 'product_dicts', None) or list()
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
        logger.debug(f'Could not obtain family product dicts for CREST intra_NO2_ONO_conversion constraints: {e}')
        return None
    for product_dict in product_dicts:
        if not isinstance(product_dict, dict):
            continue
        reactive_atoms = _normalize_no2_ono_atoms(product_dict.get('r_label_map'))
        if reactive_atoms is not None:
            return reactive_atoms
    logger.debug('No intra_NO2_ONO_conversion recipe label map was found for the CREST seed.')
    return None


def _normalize_no2_ono_atoms(atoms: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
    """
    Translate an ``intra_NO2_ONO_conversion`` atom mapping into ``C``/``N``/``O`` roles.

    Both the role keys themselves and the RMG recipe labels ``*1`` (carbon), ``*2``
    (nitrogen) and ``*3`` (migrating oxygen) are accepted.

    Args:
        atoms: A candidate atom mapping.

    Returns:
        Optional[Dict[str, int]]: ``{'C': int, 'N': int, 'O': int}``, or ``None``.
    """
    if not isinstance(atoms, dict):
        return None
    if {'C', 'N', 'O'}.issubset(atoms):
        return {role: atoms[role] for role in ('C', 'N', 'O')}
    if {'*1', '*2', '*3'}.issubset(atoms):
        return {'C': atoms['*1'], 'N': atoms['*2'], 'O': atoms['*3']}
    return None


def _is_valid_no2_ono_atom_assignment(xyz: dict, atoms: Optional[Dict[str, int]]) -> bool:
    """Return whether ``atoms`` identifies a distinct, in-range C--N--O recipe triad in ``xyz``."""
    symbols = xyz.get('symbols') if isinstance(xyz, dict) else None
    if not symbols or not isinstance(atoms, dict) or set(atoms) != {'C', 'N', 'O'}:
        return False
    indices = tuple(atoms[role] for role in ('C', 'N', 'O'))
    if any(not isinstance(index, int) or not 0 <= index < len(symbols) for index in indices):
        return False
    if len(set(indices)) != 3:
        return False
    return all(symbols[atoms[role]] == role for role in ('C', 'N', 'O'))


def _is_valid_xy_atom_assignment(xyz: dict, atoms: Optional[Dict[str, int]]) -> bool:
    """Return whether ``atoms`` identifies four distinct, in-range XY recipe atoms."""
    symbols = xyz.get('symbols') if isinstance(xyz, dict) else None
    if not symbols or not isinstance(atoms, dict) or set(atoms) != {'*1', '*2', '*3', '*4'}:
        return False
    indices = tuple(atoms[label] for label in ('*1', '*2', '*3', '*4'))
    return (all(isinstance(index, int) and 0 <= index < len(symbols) for index in indices)
            and len(set(indices)) == 4)


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
