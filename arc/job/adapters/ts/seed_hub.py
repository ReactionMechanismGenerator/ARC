"""
Shared TS-seed and wrapper-constraint hub.

This module centralizes:
1. How TS seeds are requested from a base TS-search adapter.
2. How wrapper adapters (e.g., CREST) request family-specific constraints for a seed.

``H_ATOM_BOND_CUTOFF`` is the shortest distance (in Angstrom) at which a hydrogen is taken to be free of a
neighbour. It is the floor of every element-specific cutoff, and the cutoff of an element whose bond to hydrogen
cannot be estimated at all. ``H_ATOM_BOND_CUTOFF_SLACK`` is the factor applied to the estimated single bond length
of an element with hydrogen to obtain that element's cutoff where that exceeds the floor.

The hydrolysis constants describe the four-center hydrolysis core as :func:`arc.job.adapters.ts.heuristics.hydrolysis`
emits it: ``HYDROLYSIS_REACTIVE_ROLES`` are the mechanism roles whose mutual geometry CREST holds fixed (``a`` the
electrophilic centre, ``b`` the leaving group, ``o`` the water oxygen and ``h1`` the water hydrogen that transfers from
``o`` to ``b``), ``HYDROLYSIS_INDEX_ORDER`` is the order of the per-guess index sequence, and
``HYDROLYSIS_DISTANCE_ROLE_PAIRS`` are all six pairwise separations among the four reactive atoms, which determine
the reactive core rigidly up to a reflection that the reference geometry resolves.
"""

from functools import lru_cache
from typing import Dict, List, Optional, Sequence

from arc.common import (COVALENT_RADII,
                        SINGLE_BOND_LENGTH,
                        almost_equal_coords,
                        get_atom_radius,
                        get_logger,
                        get_single_bond_length,
                        )
from arc.species.converter import xyz_to_dmat

logger = get_logger()

H_ATOM_BOND_CUTOFF = 1.5

H_ATOM_BOND_CUTOFF_SLACK = 1.15

HYDROLYSIS_REACTIVE_ROLES = ('a', 'b', 'o', 'h1')

HYDROLYSIS_INDEX_ORDER = ('a', 'b', 'e', 'o', 'h1', 'd')

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

    The family-specific seed builders are imported on the call rather than at module load.

    Args:
        reaction: The ARC reaction object.
        base_adapter: The underlying TS-search adapter providing seeds.
        dihedral_increment: Optional scan increment used by adapters that support it.
    """
    adapter = (base_adapter or '').lower()
    if adapter != 'heuristics':
        raise ValueError(f'Unsupported TS seed base adapter: {base_adapter}')

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
                    reactive_atoms = get_h_abs_atoms(reaction=reaction, xyz=xyz)
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
    Return the reaction family names handled by the heuristics hydrolysis seed builder,
    i.e., the families of every set in ``heuristics.FAMILY_SETS``. The heuristics module is
    imported on the call rather than at module load.

    Returns:
        List[str]: The families of both hydrolysis parameter sets.
    """
    from arc.job.adapters.ts.heuristics import FAMILY_SETS
    return [family for families in FAMILY_SETS.values() for family in families]


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
    re-infers the reactive atoms, from the reaction's atom labels where they are
    available and from the geometry itself otherwise. Guesses whose method contains
    ``exclude_method`` are skipped so CREST is never seeded from a prior CREST result
    (feedback-loop guard).

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
    the seed's heavy-atom--H--heavy-atom orientation. Its A/H/B triad is taken from the seed
    metadata when given there, and is otherwise resolved with :func:`get_h_abs_atoms`.

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
        reactive_atoms = explicit_atoms if explicit_atoms is not None \
            else get_h_abs_atoms(reaction=reaction, xyz=xyz)
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


@lru_cache(maxsize=None)
def _h_atom_bond_cutoff(symbol: str) -> float:
    """
    Return the distance below which ``symbol`` is taken to be covalently bound to a hydrogen.

    The cutoff is ``H_ATOM_BOND_CUTOFF_SLACK`` times an estimate of the ``symbol``--hydrogen single
    bond length, floored at ``H_ATOM_BOND_CUTOFF``. The estimate is ARC's tabulated single bond
    length where the pair has one, and the covalent radii sum otherwise, falling back to the largest
    tabulated radius of an element that ``COVALENT_RADII`` keys only by hybridisation or spin state.
    ``H_ATOM_BOND_CUTOFF`` is returned for a ``symbol`` with neither.

    Args:
        symbol (str): The element symbol of the neighbour.

    Returns:
        float: The bonding cutoff in Angstrom.
    """
    if not isinstance(symbol, str):
        return H_ATOM_BOND_CUTOFF
    if f'{symbol}_H' in SINGLE_BOND_LENGTH or f'H_{symbol}' in SINGLE_BOND_LENGTH:
        return max(H_ATOM_BOND_CUTOFF, H_ATOM_BOND_CUTOFF_SLACK * get_single_bond_length(symbol, 'H'))
    radius = get_atom_radius(symbol)
    if radius is None:
        radius = max((value for key, value in COVALENT_RADII.items() if key.split('_')[0] == symbol),
                     default=None)
    if radius is None:
        return H_ATOM_BOND_CUTOFF
    return max(H_ATOM_BOND_CUTOFF, H_ATOM_BOND_CUTOFF_SLACK * (radius + get_atom_radius('H')))


def _is_free_h_atom(symbols: Sequence[str],
                    dmat,
                    index: int,
                    transferred_h_index: int,
                    ) -> bool:
    """
    Return whether atom ``index`` is a hydrogen that is not covalently bound to any atom other than
    the transferred hydrogen ``transferred_h_index``.

    A free H atom is a valid H-abstraction reactant, as in ``R-H + H <=> R + H2``, while a hydrogen
    that is bound to something else is a spectator. Each neighbour is compared against the cutoff of
    its own element, :func:`_h_atom_bond_cutoff`, so that a long bond such as Sn--H or I--H is
    recognised as one.

    The neighbour scan skips both ``index`` itself and ``transferred_h_index``, and nothing else: an
    ``index`` whose only close contact is the hypothesised transferred hydrogen is free, while every
    other contact, including a TS-stretched one, keeps it bound.

    Args:
        symbols (Sequence[str]): The element symbols of the geometry.
        dmat: The interatomic distance matrix of the geometry.
        index (int): The index of the atom to check.
        transferred_h_index (int): The index of the transferred hydrogen.

    Returns:
        bool: Whether ``index`` is a free hydrogen atom.
    """
    if symbols[index] != 'H':
        return False
    return all(dmat[index][i] >= _h_atom_bond_cutoff(symbols[i])
               for i in range(len(symbols)) if i not in (index, transferred_h_index))


def _is_valid_h_abs_atom_assignment(xyz: dict, atoms: Optional[Dict[str, int]]) -> bool:
    """
    Return whether ``atoms`` identifies an ``A``--``H``--``B`` triad in ``xyz``, where ``H`` is the
    transferred hydrogen and ``A`` and ``B`` are its two partner atoms.

    Only ``H`` has to be a hydrogen. ``A`` and ``B`` may be hydrogens themselves, as in
    ``R-H + H <=> R + H2``, provided that such a hydrogen is a free H atom by
    :func:`_is_free_h_atom`; a hydrogen that is covalently bound to something else is a spectator
    and invalidates the assignment.
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
    dmat = xyz_to_dmat(xyz)
    if dmat is None:
        return False
    for key in ('A', 'B'):
        index = atoms[key]
        if symbols[index] == 'H' and not _is_free_h_atom(symbols=symbols,
                                                         dmat=dmat,
                                                         index=index,
                                                         transferred_h_index=atoms['H'],
                                                         ):
            return False
    return True


def get_h_abs_atoms(reaction: 'ARCReaction', xyz: dict) -> Optional[Dict[str, int]]:
    """
    Determine the A/H/B triad of an H_Abstraction TS seed.

    The reaction's atom labels are used when they are available, see
    :func:`get_h_abs_atoms_from_reaction`; the triad is inferred from the geometry alone,
    see :func:`_get_h_abs_atoms_from_xyz`, only when they are not.

    Args:
        reaction (ARCReaction): The reaction the seed belongs to.
        xyz (dict): The seed geometry, in the reaction's reactant atom order.

    Returns:
        Optional[Dict[str, int]]: ``{'H': int, 'A': int, 'B': int}``, or ``None``.
    """
    reactive_atoms = get_h_abs_atoms_from_reaction(reaction=reaction, xyz=xyz)
    if reactive_atoms is not None:
        return reactive_atoms
    return _get_h_abs_atoms_from_xyz(xyz)


def get_h_abs_atoms_from_reaction(reaction: 'ARCReaction', xyz: dict) -> Optional[Dict[str, int]]:
    """
    Determine the A/H/B triad of an H_Abstraction TS seed from the reaction's atom labels.

    Every product dictionary of the reaction's own family that was discovered in the forward
    direction contributes the candidate ``{'A': *1, 'H': *2, 'B': *3}`` of its ``r_label_map``; a
    dictionary of another family or one marked ``discovered_in_reverse``, whose ``r_label_map``
    indexes the products, is skipped. A candidate that is not a valid triad in ``xyz`` by
    :func:`_is_valid_h_abs_atom_assignment` is discarded, and of the remaining candidates the one
    whose ``H`` is closest to ``A`` and ``B`` in sum is returned, the first one on a tie.

    Args:
        reaction (ARCReaction): The reaction the seed belongs to.
        xyz (dict): The seed geometry, in the reaction's reactant atom order.

    Returns:
        Optional[Dict[str, int]]: ``{'H': int, 'A': int, 'B': int}``, or ``None`` if the reaction
                                  carries no product dictionaries or none of them yields a valid triad.
    """
    product_dicts = getattr(reaction, 'product_dicts', None) or list()
    candidates = list()
    for product_dict in product_dicts:
        if not isinstance(product_dict, dict) or product_dict.get('family') != reaction.family \
                or product_dict.get('discovered_in_reverse', False):
            continue
        r_label_map = product_dict.get('r_label_map')
        if not isinstance(r_label_map, dict) or any(label not in r_label_map for label in ('*1', '*2', '*3')):
            continue
        candidate = {'A': r_label_map['*1'], 'H': r_label_map['*2'], 'B': r_label_map['*3']}
        if _is_valid_h_abs_atom_assignment(xyz=xyz, atoms=candidate) and candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        return None
    dmat = xyz_to_dmat(xyz)
    if dmat is None:
        return None
    return min(candidates, key=lambda atoms: dmat[atoms['H']][atoms['A']] + dmat[atoms['H']][atoms['B']])


def _get_h_abs_atoms_from_xyz(xyz: dict) -> Optional[Dict[str, int]]:
    """
    Determine H-abstraction atoms from a TS guess.

    ``H`` is the transferred hydrogen, ``A`` the atom nearest to it and ``B`` the second nearest,
    both chosen among the heavy atoms and the free hydrogens of the geometry. Free hydrogens are
    identified with :func:`_is_free_h_atom`, the same check :func:`_is_valid_h_abs_atom_assignment`
    applies, so that an abstraction by or from a free H atom (``R-H + H <=> R + H2``) is resolvable
    while a hydrogen bound to something else stays excluded as a spectator.

    ``H`` is the hydrogen whose two partners are closest to it in sum. Every tie, both between two
    candidate hydrogens and between two equidistant partners of one hydrogen, is resolved in favour
    of the lowest index.

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
    selected_partners = None
    for hydrogen_index in hydrogen_indices:
        partners = sorted(
            (atom for atom in range(len(symbols))
             if atom != hydrogen_index
             and (symbols[atom] != 'H'
                  or _is_free_h_atom(symbols=symbols,
                                     dmat=dmat,
                                     index=atom,
                                     transferred_h_index=hydrogen_index,
                                     ))),
            key=lambda atom: dmat[hydrogen_index][atom],
        )[:2]
        if len(partners) < 2:
            continue
        distances = dmat[hydrogen_index][partners[0]] + dmat[hydrogen_index][partners[1]]
        if distances < min_distance:
            min_distance = distances
            selected_hydrogen = hydrogen_index
            selected_partners = partners

    if selected_hydrogen is not None and selected_partners is not None:
        return {'H': selected_hydrogen, 'A': selected_partners[0], 'B': selected_partners[1]}

    logger.warning('No valid hydrogen atom found for CREST H-abstraction atoms.')
    return None
