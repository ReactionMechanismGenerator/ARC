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

# Longest distance at which a hydrogen is taken to be covalently bound to a neighbour, used to
# tell a free H radical (a valid donor/acceptor) from a bound spectator hydrogen.
H_ATOM_BOND_CUTOFF = 1.5

# The reaction families handled by the heuristics hydrolysis seed builder, mirroring
# ``heuristics.FAMILY_SETS``. Held here so that resolving them does not import heuristics.
HYDROLYSIS_FAMILIES = ('carbonyl_based_hydrolysis',
                       'ether_hydrolysis',
                       'nitrile_hydrolysis',
                       )

# The hydrolysis mechanism roles whose mutual geometry CREST has to hold fixed: ``a`` is the
# electrophilic centre, ``b`` the leaving group, ``o`` the water oxygen and ``h1`` the water
# hydrogen that transfers from ``o`` to ``b``.
HYDROLYSIS_REACTIVE_ROLES = ('a', 'b', 'o', 'h1')

# The order in which ``heuristics.hydrolysis()`` emits its per-guess index sequence.
HYDROLYSIS_INDEX_ORDER = ('a', 'b', 'e', 'o', 'h1', 'd')

# All six pairwise separations among the four reactive atoms. Four atoms have six internal
# degrees of freedom, so the six pairwise distances determine the reactive core rigidly, up to
# a reflection that the reference geometry resolves.
HYDROLYSIS_DISTANCE_ROLE_PAIRS = (('a', 'b'),
                                  ('a', 'o'),
                                  ('a', 'h1'),
                                  ('b', 'o'),
                                  ('b', 'h1'),
                                  ('o', 'h1'),
                                  )


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
    from arc.job.adapters.ts.heuristics import h_abstraction, hydrolysis

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
    elif reaction.family in get_hydrolysis_families():
        try:
            xyzs_raw, families, indices = hydrolysis(reaction=reaction)
        except ValueError:
            xyz_entries = list()
        else:
            for xyz, family, idx in zip(xyzs_raw, families, indices):
                metadata = {'indices': idx}
                reactive_atoms = get_hydrolysis_reactive_atoms(idx)
                if reactive_atoms is not None:
                    metadata['reactive_atoms'] = reactive_atoms
                xyz_entries.append({
                    'xyz': xyz,
                    'method': 'Heuristics',
                    'family': family,
                    'source_adapter': 'heuristics',
                    'metadata': metadata,
                })
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


def get_hydrolysis_families() -> List[str]:
    """
    Return the reaction family names handled by the heuristics hydrolysis seed builder.

    These mirror ``heuristics.FAMILY_SETS`` and are held here so that resolving them does not
    import ``arc.job.adapters.ts.heuristics``. ``test_hydrolysis_families_match_family_sets``
    fails if the two ever disagree.

    Returns:
        List[str]: The families of both hydrolysis parameter sets.
    """
    return list(HYDROLYSIS_FAMILIES)


def get_hydrolysis_reactive_atoms(indices) -> Optional[Dict[str, int]]:
    """
    Return the named hydrolysis reactive atoms of a seed's raw index metadata.

    Two shapes are accepted: a mapping already keyed by the mechanism roles, of which only
    ``a``, ``b``, ``o`` and ``h1`` are kept, and the positional sequence
    ``(a, b, e, o, h1, d)`` that :func:`arc.job.adapters.ts.heuristics.hydrolysis` emits.

    Args:
        indices: The ``indices`` entry of a hydrolysis seed's metadata.

    Returns:
        Optional[Dict[str, int]]: The ``a``, ``b``, ``o`` and ``h1`` atom indices, or ``None``
                                  if the roles cannot be resolved from ``indices``.
    """
    positional = None
    if isinstance(indices, dict):
        positional = indices
    elif isinstance(indices, (list, tuple)) and len(indices) == len(HYDROLYSIS_INDEX_ORDER):
        positional = dict(zip(HYDROLYSIS_INDEX_ORDER, indices))
    if positional is None or any(role not in positional for role in HYDROLYSIS_REACTIVE_ROLES):
        return None
    return {role: positional[role] for role in HYDROLYSIS_REACTIVE_ROLES}


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
    H-abstraction additionally supplies ``angle_atoms`` so completed geometries retain
    the seed's heavy-atom--H--heavy-atom orientation.

    The hydrolysis families pin all six pairwise distances among the four reactive atoms
    ``a``, ``b``, ``o`` and ``h1``, which are read from the seed metadata: either from an
    explicit ``reactive_atoms`` mapping or, failing that, from the raw ``indices`` entry.
    ``None`` is returned when neither resolves to four distinct, in-range atoms whose ``h1``
    is a hydrogen.
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
        return None
    if family in get_hydrolysis_families():
        raw_indices = metadata.get('indices') if isinstance(metadata, dict) else None
        reactive_atoms = explicit_atoms if explicit_atoms is not None \
            else get_hydrolysis_reactive_atoms(raw_indices)
        if _is_valid_hydrolysis_atom_assignment(xyz=xyz, atoms=reactive_atoms):
            return {
                'atoms': tuple(reactive_atoms[role] for role in HYDROLYSIS_REACTIVE_ROLES),
                'distance_pairs': tuple((reactive_atoms[role_1], reactive_atoms[role_2])
                                        for role_1, role_2 in HYDROLYSIS_DISTANCE_ROLE_PAIRS),
            }
        logger.warning(f'Could not determine the CREST hydrolysis reactive atoms of a {family} seed '
                       f'from the metadata {metadata!r}, skipping this seed.')
    return None


def _is_valid_hydrolysis_atom_assignment(xyz: dict, atoms: Optional[Dict[str, int]]) -> bool:
    """
    Return whether ``atoms`` identifies four distinct, in-range hydrolysis reactive atoms.

    The transferred atom ``h1`` has to be a hydrogen, which is what distinguishes a correctly
    ordered role assignment from one whose roles were permuted.

    Args:
        xyz (dict): The seed geometry.
        atoms (Optional[Dict[str, int]]): The candidate role-to-index mapping.

    Returns:
        bool: Whether ``atoms`` is a usable hydrolysis reactive-atom assignment.
    """
    symbols = xyz.get('symbols') if isinstance(xyz, dict) else None
    if not symbols or not isinstance(atoms, dict) or set(atoms) != set(HYDROLYSIS_REACTIVE_ROLES):
        return False
    indices = tuple(atoms[role] for role in HYDROLYSIS_REACTIVE_ROLES)
    if not all(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(symbols)
               for index in indices):
        return False
    if len(set(indices)) != len(HYDROLYSIS_REACTIVE_ROLES):
        return False
    return symbols[atoms['h1']] == 'H'


def _is_valid_xy_atom_assignment(xyz: dict, atoms: Optional[Dict[str, int]]) -> bool:
    """Return whether ``atoms`` identifies four distinct, in-range XY recipe atoms."""
    symbols = xyz.get('symbols') if isinstance(xyz, dict) else None
    if not symbols or not isinstance(atoms, dict) or set(atoms) != {'*1', '*2', '*3', '*4'}:
        return False
    indices = tuple(atoms[label] for label in ('*1', '*2', '*3', '*4'))
    return (all(isinstance(index, int) and 0 <= index < len(symbols) for index in indices)
            and len(set(indices)) == 4)


def _is_valid_h_abs_atom_assignment(xyz: dict, atoms: Optional[Dict[str, int]]) -> bool:
    """
    Return whether ``atoms`` identifies a donor--H--acceptor triad in ``xyz``.

    Only the transferred atom has to be a hydrogen. The donor and the acceptor may be hydrogens
    themselves, as in ``R-H + H <=> R + H2``: the CREST constraints are index-based, so the element
    of the donor and acceptor is irrelevant to them. Requiring both to be heavy atoms silently
    excluded every abstraction by an H atom.
    """
    symbols = xyz.get('symbols') if isinstance(xyz, dict) else None
    if not symbols or not isinstance(atoms, dict) or set(atoms) != {'A', 'H', 'B'}:
        return False
    if any(not isinstance(atoms[key], int) or not 0 <= atoms[key] < len(symbols) for key in atoms):
        return False
    if len({atoms['A'], atoms['H'], atoms['B']}) != 3:
        return False
    if symbols[atoms['H']] != 'H':
        return False
    if all(symbols[atoms[key]] != 'H' for key in ('A', 'B')):
        return True
    # A hydrogen donor or acceptor is only meaningful when it is a free H atom, as in
    # ``R-H + H <=> R + H2``. A hydrogen that is covalently bound to something else is a spectator
    # that a faulty generator mapping named by mistake, which is what the heavy-atom-only rule was
    # really guarding against.
    dmat = xyz_to_dmat(xyz)
    if dmat is None:
        return False
    for key in ('A', 'B'):
        index = atoms[key]
        if symbols[index] != 'H':
            continue
        neighbours = [dmat[index][i] for i in range(len(symbols)) if i not in (index, atoms['H'])]
        if neighbours and min(neighbours) < H_ATOM_BOND_CUTOFF:
            return False
    return True


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

    hydrogen_indices = [i for i, symbol in enumerate(symbols) if symbol == 'H']
    min_distance = float('inf')
    selected_hydrogen = None
    selected_heavy_atoms = None
    for hydrogen_index in hydrogen_indices:
        heavy_atoms = sorted(
            (atom for atom, symbol in enumerate(symbols)
             if atom != hydrogen_index and symbol != 'H'),
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
