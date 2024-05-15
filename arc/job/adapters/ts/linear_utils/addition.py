"""
Fragment-based addition / dissociation TS geometry builders
"""

from collections import deque
from functools import partial
from itertools import permutations, product
from typing import TYPE_CHECKING
from collections.abc import Sequence

import numpy as np

from arc.common import get_logger, get_single_bond_length
from arc.job.adapters.ts.linear_utils.geom_utils import bfs_path, mol_to_adjacency
from arc.job.adapters.ts.linear_utils.isomerization import ring_closure_xyz
from arc.job.adapters.ts.linear_utils.path_spec import ReactionPathSpec, insertion_ring_extra_stretch, validate_addition_guess
from arc.job.adapters.ts.linear_utils.postprocess import PAULING_DELTA, has_detached_hydrogen, has_too_many_fragments
from arc.species.species import ARCSpecies, colliding_atoms

if TYPE_CHECKING:
    from arc.molecule import Molecule


logger = get_logger()


# ---------------------------------------------------------------------------
# Module-level helpers shared by fragmentation builders
# ---------------------------------------------------------------------------

def _connected_components(adj: dict[int, set[int]],
                          n_atoms: int,
                          removed_bonds: list[tuple[int, int]],
                          ) -> list[set[int]]:
    """
    Return the connected components after removing a set of bonds.

    Builds a copy of ``adj`` with each bond in ``removed_bonds`` deleted
    in both directions, then BFS-walks every atom index in
    ``[0, n_atoms)`` to find the connected components. Atoms missing
    from ``adj`` are treated as isolated.

    Args:
        adj (dict[int, set[int]]): Original adjacency map (atom index →
            set of bonded atom indices). Not modified.
        n_atoms (int): Total number of atoms; the BFS sweep covers
            ``range(n_atoms)``.
        removed_bonds (list[tuple[int, int]]): Bonds to delete from the
            adjacency map before computing components.

    Returns:
        list[set[int]]: One ``set[int]`` per component, partitioning
            ``range(n_atoms)``.
    """
    local_adj = {k: set(v) for k, v in adj.items()}
    for a, b in removed_bonds:
        local_adj.setdefault(a, set()).discard(b)
        local_adj.setdefault(b, set()).discard(a)
    visited: set[int] = set()
    components: list[set[int]] = []
    for start in range(n_atoms):
        if start in visited:
            continue
        component: set[int] = set()
        queue: deque = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            queue.extend(local_adj.get(node, set()) - visited)
        components.append(component)
    return components


def _fragment_element_formulas(fragments: list[set[int]],
                               symbols: Sequence[str],
                               ) -> list[dict[str, int]]:
    """
    Build per-fragment element-count formulas from atom indices.

    Walks each fragment and counts element occurrences in ``symbols``.
    The result list has the same length and order as ``fragments``.

    Args:
        fragments (list[set[int]]): Fragment partition; each set is a
            collection of atom indices into ``symbols``.
        symbols (Sequence[str]): Per-atom element symbols.

    Returns:
        list[dict[str, int]]: One ``{element: count}`` dict per fragment,
            preserving fragment order.
    """
    frag_formulas: list[dict[str, int]] = []
    for frag in fragments:
        formula: dict[str, int] = {}
        for idx in frag:
            sym = symbols[idx]
            formula[sym] = formula.get(sym, 0) + 1
        frag_formulas.append(formula)
    return frag_formulas


def _fragments_match_target_formulas(fragments: list[set[int]],
                                     symbols: Sequence[str],
                                     target_sorted: list[dict[str, int]],
                                     target_heavy: list[dict[str, int]],
                                     target_total_h: int,
                                     n_products: int,
                                     strict: bool,
                                     ) -> bool:
    """
    Check whether a fragment partition matches the target product formulas.

    When ``strict`` is ``True``, the per-fragment element counts must
    exactly equal the target product formulas (including hydrogen).
    When ``strict`` is ``False``, the heavy-atom-only formulas must
    match and the total hydrogen count across all fragments must equal
    the total target hydrogen count, but hydrogens may redistribute
    between fragments. Both checks are order-independent because the
    per-fragment formulas are sorted before comparison.

    Args:
        fragments (list[set[int]]): Fragment partition produced by
            :func:`_connected_components`.
        symbols (Sequence[str]): Per-atom element symbols.
        target_sorted (list[dict[str, int]]): Target product formulas
            (with H counts), sorted by element-count items, used in the
            strict comparison.
        target_heavy (list[dict[str, int]]): Target heavy-only formulas,
            sorted, used in the relaxed comparison.
        target_total_h (int): Total hydrogen count across all target
            products, used in the relaxed comparison.
        n_products (int): Expected number of fragments.
        strict (bool): ``True`` for exact element match, ``False`` for
            heavy-only + total-H match.

    Returns:
        bool: ``True`` when the fragment partition matches the target
            formulas under the chosen comparison mode.
    """
    if len(fragments) != n_products:
        return False
    frag_formulas = _fragment_element_formulas(fragments, symbols)
    if strict:
        frag_sorted = sorted(frag_formulas, key=lambda d: sorted(d.items()))
        return frag_sorted == target_sorted
    frag_heavy = _heavy_only_formulas(frag_formulas)
    frag_total_h = sum(f.get('H', 0) for f in frag_formulas)
    return frag_heavy == target_heavy and frag_total_h == target_total_h


def _min_distance_to_heavy_atoms_in_set(pos: np.ndarray,
                                        atom_indices: set[int],
                                        ts_coords: np.ndarray,
                                        symbols: Sequence[str],
                                        exclude_indices: set[int],
                                        ) -> float:
    """
    Minimum distance from a point to the heavy atoms in a fragment.

    Iterates over ``atom_indices`` and returns the smallest distance
    from ``pos`` to any non-hydrogen atom whose index is not in
    ``exclude_indices``. Hydrogens are always skipped — only heavy
    fragment atoms count toward the clearance.

    Args:
        pos (np.ndarray): The candidate point, shape ``(3,)``.
        atom_indices (set[int]): Atom indices belonging to the fragment
            of interest (typically a source fragment whose heavies the
            placement should clear).
        ts_coords (np.ndarray): The full TS coordinate array, shape
            ``(N, 3)``.
        symbols (Sequence[str]): Per-atom element symbols.
        exclude_indices (set[int]): Atom indices to skip in addition to
            all hydrogens (typically the donor and the migrating H).

    Returns:
        float: The smallest clearance, or ``float('inf')`` when no
            eligible heavy atom remains.
    """
    md = float('inf')
    for idx in atom_indices:
        if idx in exclude_indices or symbols[idx] == 'H':
            continue
        md = min(md, float(np.linalg.norm(pos - ts_coords[idx])))
    return md


def _negative_bond_distance(coords: np.ndarray,
                            item: tuple[tuple[int, int], int],
                            ) -> float:
    """
    Negative current distance of a candidate bond, for descending sort.

    The sort-key is the *negative* bond distance so that
    ``sorted(..., key=partial(_negative_bond_distance, coords))`` yields
    candidates ordered from the longest current bond distance down to
    the shortest. ``item`` is shaped as the candidate tuples produced
    by :func:`detect_intra_frag_ring_bonds`: ``((a, b), path_length)``;
    only the bond endpoints ``(a, b)`` participate in the score.

    Args:
        coords (np.ndarray): Atomic coordinate array, shape ``(N, 3)``.
        item (tuple[tuple[int, int], int]): Candidate of the form
            ``((a, b), path_length)``.

    Returns:
        float: ``-||coords[a] - coords[b]||`` (negative for descending sort).
    """
    a, b = item[0]
    return -float(np.linalg.norm(coords[a] - coords[b]))


def _heavy_only_formula(formula: dict[str, int]) -> dict[str, int]:
    """
    Strip the hydrogen count from a single element-count formula.

    Args:
        formula (dict[str, int]): A formula mapping element symbol to
            count (e.g. ``{'C': 2, 'H': 4, 'O': 1}``).

    Returns:
        dict[str, int]: A new dict with the ``'H'`` key removed; all
            other entries are passed through unchanged.
    """
    return {k: v for k, v in formula.items() if k != 'H'}


def _heavy_only_formulas(formulas: list[dict[str, int]]) -> list[dict[str, int]]:
    """
    Strip hydrogen counts from a list of element-count formulas.

    Returns a deterministic, sorted copy in which each formula keeps
    only non-hydrogen elements. Used to compare fragmentations modulo
    H redistribution: two fragment sets agree on heavy composition when
    their heavy-only formulas are equal as sorted lists.

    Args:
        formulas (list[dict[str, int]]): One formula per fragment, each
            mapping element symbol to count (e.g. ``{'C': 2, 'H': 4}``).

    Returns:
        list[dict[str, int]]: A list of heavy-only formulas, sorted by
            element-count items so the result is order-independent.
    """
    return sorted(
        ({k: v for k, v in f.items() if k != 'H'} for f in formulas),
        key=lambda d: sorted(d.items()),
    )


def _strip_to_connectivity(mol: Molecule) -> Molecule:
    """
    Return a deep copy of ``mol`` with bond orders, radicals, and charges flattened.

    Sets every bond order to ``1.0``, every atom's radical electrons,
    formal charge, and lone pairs to ``0``, and the molecule's
    multiplicity to ``1``. The atom and bond identities (graph
    structure) are preserved. The intent is to compare two molecules
    purely by connectivity — RMG's ``find_isomorphism(strict=False)``
    still rejects normalized graphs when the input retains stale bond
    orders or charges, so callers normalize both sides before comparing.

    Args:
        mol (Molecule): RMG molecule to normalize. Not mutated; the
            input is deep-copied first.

    Returns:
        Molecule: The deep-copied, connectivity-only normalized molecule.
    """
    m = mol.copy(deep=True)
    for bond in m.get_all_edges():
        bond.order = 1.0
    for atom in m.atoms:
        atom.radical_electrons = 0
        atom.charge = 0
        atom.lone_pairs = 0
    m.multiplicity = 1
    return m


def _multi_atom_idx_to_species(m_idx: int,
                               cum_sizes: list[int],
                               ) -> int | None:
    """
    Map a flat multi-species atom index to its species index.

    The multi-species side is laid out as a concatenation of per-species
    atom blocks. ``cum_sizes`` is the prefix-sum of per-species atom
    counts, so for ``k`` species ``cum_sizes`` has length ``k + 1`` and
    ``cum_sizes[0] == 0``. An atom index ``m_idx`` belongs to species
    ``k`` when ``cum_sizes[k] <= m_idx < cum_sizes[k + 1]``.

    Args:
        m_idx (int): The flat atom index in the concatenated multi-species
            atom layout.
        cum_sizes (list[int]): Prefix-sum array of per-species atom
            counts; length is ``num_species + 1``.

    Returns:
        int | None: The species index ``k`` containing ``m_idx``, or
            ``None`` when ``m_idx`` falls outside the layout.
    """
    for k in range(len(cum_sizes) - 1):
        if cum_sizes[k] <= m_idx < cum_sizes[k + 1]:
            return k
    return None


def _labels_map_to_consistent_species(candidate: dict[int, tuple[int, int]],
                                      product_dict: dict | None,
                                      uni_is_reactant: bool,
                                      multi_species: list[ARCSpecies],
                                      ) -> bool:
    """
    Verify that RMG label groups all map to a single multi-species side.

    A reaction family supplies labelled atoms (``r_label_map`` and
    ``p_label_map``). Atoms sharing a label across the unimolecular and
    multi-species sides must belong to the same product species in the
    candidate fragment-to-species mapping. This rules out fragmentations
    that would split a labelled group across two products.

    The check is a no-op (returns ``True``) when ``product_dict`` is
    ``None`` or when either label map is missing.

    Args:
        candidate (dict[int, tuple[int, int]]): Candidate map from each
            unimolecular atom index to ``(species_idx, atom_idx_in_species)``.
        product_dict (dict | None): The reaction-family product dict
            with ``r_label_map``, ``p_label_map``, and (when
            ``uni_is_reactant`` is ``False``) ``products`` keys.
        uni_is_reactant (bool): ``True`` when the unimolecular side is
            the reactant.
        multi_species (list[ARCSpecies]): The multi-species side
            (products when ``uni_is_reactant`` is ``True``).

    Returns:
        bool: ``True`` when the candidate is consistent (or when no
            label data is available); ``False`` when a labelled group
            spans more than one species.
    """
    if product_dict is None:
        return True
    if uni_is_reactant:
        uni_lmap = product_dict.get('r_label_map')
        multi_lmap = product_dict.get('p_label_map')
    else:
        uni_lmap = product_dict.get('p_label_map')
        multi_lmap = product_dict.get('r_label_map')
    if not uni_lmap or not multi_lmap:
        return True

    if uni_is_reactant:
        multi_mols = product_dict.get('products', [])
    else:
        multi_mols = [sp.mol for sp in multi_species]
    cum_sizes = [0]
    for m in multi_mols:
        cum_sizes.append(cum_sizes[-1] + len(m.atoms))

    label_groups: dict[int, set[int]] = {}
    for label, uni_idx in uni_lmap.items():
        m_idx = multi_lmap.get(label)
        if m_idx is None:
            continue
        sp_id = _multi_atom_idx_to_species(m_idx, cum_sizes)
        if sp_id is None:
            continue
        label_groups.setdefault(sp_id, set()).add(uni_idx)

    for group_indices in label_groups.values():
        species_set: set[int] = set()
        for uni_idx in group_indices:
            mapping = candidate.get(uni_idx)
            if mapping is not None:
                species_set.add(mapping[0])
        if len(species_set) > 1:
            return False
    return True


# ---- Fragment helpers ----

def find_split_bonds_by_fragmentation(uni_mol: Molecule,
                                      product_species: list[ARCSpecies],
                                      ) -> list[list[tuple[int, int]]]:
    """
    Find bonds in the unimolecular species that, when removed, fragment it
    into components matching the product species by element composition.

    This is a fallback for cases where ``product_dicts`` are unavailable or
    give unreliable reactive-bond information. It works by enumerating
    candidate bond cuts (1-bond or 2-bond) and checking whether the resulting
    fragments match the products' element counts.

    Args:
        uni_mol (Molecule): RMG Molecule of the unimolecular species.
        product_species (list[ARCSpecies]): The product species on the multi-species side.

    Returns:
        list[list[tuple[int, int]]]: Each entry is a list of (i, j) bond tuples that, when removed,
            yield a valid fragmentation. Returns an empty list if no valid cut is found.
    """
    n_products = len(product_species)
    if n_products < 2:
        return []

    target_formulas: list[dict[str, int]] = []
    for sp in product_species:
        formula: dict[str, int] = {}
        for sym in sp.get_xyz()['symbols']:
            formula[sym] = formula.get(sym, 0) + 1
        target_formulas.append(formula)
    target_sorted = sorted(target_formulas, key=lambda d: sorted(d.items()))

    n_atoms = len(uni_mol.atoms)
    adj = mol_to_adjacency(uni_mol)
    bonds: list[tuple[int, int]] = sorted({(min(a, b), max(a, b)) for a, nbrs in adj.items() for b in nbrs})

    symbols = tuple(atom.element.symbol for atom in uni_mol.atoms)

    target_heavy = _heavy_only_formulas(target_formulas)
    target_total_h = sum(f.get('H', 0) for f in target_formulas)

    results: list[list[tuple[int, int]]] = []

    # ---- Single-bond cuts, exact match ----
    for bond in bonds:
        frags = _connected_components(adj, n_atoms, [bond])
        if _fragments_match_target_formulas(frags, symbols, target_sorted, target_heavy,
                                            target_total_h, n_products, strict=True):
            results.append([bond])
    if results:
        return results

    # ---- Single-bond cuts, relaxed H match ----
    for bond in bonds:
        frags = _connected_components(adj, n_atoms, [bond])
        if _fragments_match_target_formulas(frags, symbols, target_sorted, target_heavy,
                                            target_total_h, n_products, strict=False):
            results.append([bond])
    if results:
        return results

    # ---- Two-bond cuts (exact then relaxed) ----
    if n_products <= 3:
        for strict in (True, False):
            for i in range(len(bonds)):
                for j in range(i + 1, len(bonds)):
                    frags = _connected_components(adj, n_atoms, [bonds[i], bonds[j]])
                    if _fragments_match_target_formulas(frags, symbols, target_sorted, target_heavy,
                                                        target_total_h, n_products, strict=strict):
                        results.append([bonds[i], bonds[j]])
                if len(results) >= 10:
                    break
            if results:
                return results
    return results


def map_and_verify_fragments(uni_mol: Molecule,
                             split_bonds: list[tuple[int, int]],
                             multi_species: list[ARCSpecies],
                             cross_bonds: list[tuple[int, int]] | None = None,
                             product_dict: dict | None = None,
                             uni_is_reactant: bool = True,
                             ) -> dict[int, tuple[int, int]] | None:
    """
    Validate that severing *split_bonds* (and reconnecting *cross_bonds*) in
    *uni_mol* produces fragments that topologically match *multi_species*.

    The function works by:

    1. Deep-copying *uni_mol* and removing every bond in *split_bonds* (adjusting
       radical electrons on the severed endpoints).
    2. Re-adding every bond in *cross_bonds* (these are forming bonds that exist in
       the products but not the reactant). This merges isolated migrating atoms back
       into the fragment they belong to in the product.
    3. Calling ``split()`` on the modified graph and checking that the number of
       connected components equals the number of product species.
    4. Using connectivity-normalized ``is_isomorphic`` (bond orders and radical
       electrons removed) with ``strict=False`` to match each fragment to exactly
       one product species.
    5. Validating against the RMG reaction-family label maps (``r_label_map`` and
       ``p_label_map`` from *product_dict*): all labeled atoms that belong to the
       same product must end up in the same fragment.

    Args:
        uni_mol: RMG Molecule of the unimolecular species.
        split_bonds: Bonds to sever (breaking bonds present in *uni_mol*'s graph).
        multi_species: Product (or reactant) species on the multi-species side.
        cross_bonds: Forming bonds absent from *uni_mol* to reconnect after severing.
        product_dict: Optional dict with ``r_label_map``, ``p_label_map``, and
            ``products`` keys for label verification.
        uni_is_reactant: True when *uni_mol* is the reactant.

    Returns:
        Dict mapping *uni_mol* atom index → ``(species_list_index, atom_index_in_species)`` if validation succeeds,
            else ``None``.
    """
    n_atoms = len(uni_mol.atoms)

    # ---- 1. Deep copy and sever split bonds ----
    mol_copy = uni_mol.copy(deep=True)
    copy_atoms = list(mol_copy.atoms)

    for a_idx, b_idx in split_bonds:
        try:
            bond = mol_copy.get_bond(copy_atoms[a_idx], copy_atoms[b_idx])
        except (ValueError, KeyError):
            return None
        order = max(1, int(bond.order))
        copy_atoms[a_idx].radical_electrons += order
        copy_atoms[b_idx].radical_electrons += order
        mol_copy.remove_bond(bond)

    # ---- 2. Reconnect cross bonds (forming bonds) ----
    BondClass = None
    if cross_bonds:
        existing_edges = list(mol_copy.get_all_edges())
        if existing_edges:
            BondClass = type(existing_edges[0])

    for a_idx, b_idx in (cross_bonds or []):
        if BondClass is None:
            break
        try:
            mol_copy.get_bond(copy_atoms[a_idx], copy_atoms[b_idx])
            continue
        except (ValueError, KeyError):
            pass
        new_bond = BondClass(copy_atoms[a_idx], copy_atoms[b_idx], order=1)
        mol_copy.add_bond(new_bond)
        copy_atoms[a_idx].radical_electrons = max(
            0, copy_atoms[a_idx].radical_electrons - 1)
        copy_atoms[b_idx].radical_electrons = max(
            0, copy_atoms[b_idx].radical_electrons - 1)

    # ---- 3. Split into fragments ----
    try:
        fragments = mol_copy.split()
    except Exception:
        return None

    if len(fragments) != len(multi_species):
        return None

    # Fragments inherit stale connectivity values from the parent, which causes
    # ``is_isomorphic`` to reject valid matches.
    for frag in fragments:
        frag.update_connectivity_values()

    # ---- 4. Build fragment → original-index mapping ----
    frag_data: list[tuple[Molecule, list[int]]] = []
    for frag in fragments:
        orig_indices: list[int] = []
        for frag_atom in frag.atoms:
            found = next((i for i, ca in enumerate(copy_atoms)
                          if ca is frag_atom), None)
            if found is None:
                return None
            orig_indices.append(found)
        frag_data.append((frag, orig_indices))

    # ---- 5. Connectivity-normalized isomorphism ----
    n_species = len(multi_species)

    norm_prods = [_strip_to_connectivity(sp.mol) for sp in multi_species]
    norm_prod_atoms = [list(np_mol.atoms) for np_mol in norm_prods]

    # ---- 6. Try all fragment ↔ product permutations ----
    for perm in permutations(range(n_species)):
        all_ok = True
        iso_per_frag: list[Tuple] = []

        for frag_idx, species_idx in enumerate(perm):
            frag, orig_indices = frag_data[frag_idx]

            prod_mol = multi_species[species_idx].mol
            if len(frag.atoms) != len(prod_mol.atoms):
                all_ok = False
                break

            n_frag = _strip_to_connectivity(frag)
            n_prod = norm_prods[species_idx]

            iso_maps = n_frag.find_isomorphism(n_prod, strict=False)
            if not iso_maps:
                all_ok = False
                break

            n_frag_atoms = list(n_frag.atoms)
            np_atoms = norm_prod_atoms[species_idx]
            iso_per_frag.append(
                (iso_maps, n_frag_atoms, np_atoms, orig_indices, species_idx))

        if not all_ok or len(iso_per_frag) != len(frag_data):
            continue

        for combo in product(*(maps for maps, _, _, _, _ in iso_per_frag)):
            candidate_map: dict[int, tuple[int, int]] = {}
            for iso_map, (_, nf_atoms, np_atoms_i, orig_idx, sp_idx) in zip(combo, iso_per_frag):
                for nf_atom, np_atom in iso_map.items():
                    fi = nf_atoms.index(nf_atom)
                    pi = np_atoms_i.index(np_atom)
                    candidate_map[orig_idx[fi]] = (sp_idx, pi)
            if len(candidate_map) == n_atoms and _labels_map_to_consistent_species(
                    candidate_map, product_dict, uni_is_reactant, multi_species):
                return candidate_map

    return None


def build_concerted_ts(uni_xyz: dict,
                       uni_mol: Molecule,
                       split_bonds: list[tuple[int, int]],
                       cross_bonds: list[tuple[int, int]],
                       ) -> dict | None:
    """
    Build a TS guess for concerted multi-bond reactions by simultaneously
    stretching breaking bonds and contracting forming bonds.

    For each **split bond** (breaking), both endpoints are pushed apart
    symmetrically toward the Pauling TS estimate. For each **cross bond**
    (forming), both endpoints are pulled together toward the Pauling TS
    estimate. The displacements are scaled by *weight* and applied
    iteratively (3 rounds) to allow coupled adjustments to converge.

    This handles concerted eliminations (e.g. XY_elimination_hydroxyl)
    and retro-cycloadditions where multiple bonds change simultaneously.

    Args:
        uni_xyz: Unimolecular-species XYZ coordinates.
        uni_mol: RMG Molecule of the unimolecular species.
        split_bonds: Bonds to break (stretch).
        cross_bonds: Bonds to form (contract).

    Returns:
        TS guess XYZ dictionary, or ``None`` if validation fails.
    """
    if not split_bonds and not cross_bonds:
        return None
    symbols = uni_xyz['symbols']
    coords = np.array(uni_xyz['coords'], dtype=float)

    split_targets: dict[tuple[int, int], float] = {}
    for a, b in split_bonds:
        sbl = get_single_bond_length(symbols[a], symbols[b]) or 1.5
        both_heavy = symbols[a] != 'H' and symbols[b] != 'H'
        if both_heavy:
            split_targets[(a, b)] = sbl + 2.0 * PAULING_DELTA
        else:
            split_targets[(a, b)] = sbl + PAULING_DELTA

    cross_targets: dict[tuple[int, int], float] = {}
    for a, b in cross_bonds:
        sbl = get_single_bond_length(symbols[a], symbols[b]) or 1.5
        both_h = symbols[a] == 'H' and symbols[b] == 'H'
        if both_h:
            # H-H forms quickly — target close to equilibrium.
            cross_targets[(a, b)] = sbl * 1.12
        else:
            cross_targets[(a, b)] = sbl + PAULING_DELTA

    # Ring bonds adjacent to the reactive core may strengthen (single→double); shorten by 5%.
    ring_atoms = set()
    for a, b in split_bonds + cross_bonds:
        ring_atoms.update((a, b))
    strengthen_targets: dict[tuple[int, int], float] = {}
    atom_to_idx_conc = {atom: idx for idx, atom in enumerate(uni_mol.atoms)}
    for atom in uni_mol.atoms:
        ia = atom_to_idx_conc[atom]
        if ia not in ring_atoms:
            continue
        for nbr, bond in atom.bonds.items():
            ib = atom_to_idx_conc[nbr]
            if ib not in ring_atoms or ib <= ia:
                continue
            key = (ia, ib)
            if key in split_targets or key in cross_targets:
                continue
            sbl = get_single_bond_length(symbols[ia], symbols[ib]) or 1.5
            strengthen_targets[key] = sbl * 0.95

    for _ in range(10):  # converge coupled adjustments
        for (a, b), d_target in split_targets.items():
            d_cur = float(np.linalg.norm(coords[a] - coords[b]))
            if d_cur < 1e-6:
                continue
            if d_cur < d_target:
                vec = coords[b] - coords[a]
                direction = vec / d_cur
                half_push = (d_target - d_cur) * 0.5
                coords[a] -= direction * half_push
                coords[b] += direction * half_push

        for (a, b), d_target in cross_targets.items():
            d_cur = float(np.linalg.norm(coords[a] - coords[b]))
            if d_cur < 1e-6:
                continue
            if d_cur > d_target:
                vec = coords[b] - coords[a]
                direction = vec / d_cur
                half_pull = (d_cur - d_target) * 0.5
                coords[a] += direction * half_pull
                coords[b] -= direction * half_pull

        for (a, b), d_target in strengthen_targets.items():
            d_cur = float(np.linalg.norm(coords[a] - coords[b]))
            if d_cur < 1e-6 or d_cur < d_target:
                continue
            vec = coords[b] - coords[a]
            direction = vec / d_cur
            half_pull = (d_cur - d_target) * 0.3  # gentle
            coords[a] += direction * half_pull
            coords[b] -= direction * half_pull

    # ---- Collision resolution: push apart any pair closer than 0.7× SBL ----
    reactive_indices = set()
    for a, b in split_bonds + cross_bonds:
        reactive_indices.update((a, b))
    for _ in range(5):
        any_collision = False
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                d = float(np.linalg.norm(coords[i] - coords[j]))
                sbl_ij = get_single_bond_length(symbols[i], symbols[j]) or 1.5
                min_d = sbl_ij * 0.7
                if 1e-6 < d < min_d:
                    any_collision = True
                    vec = coords[j] - coords[i]
                    direction = vec / d
                    push = (min_d - d) * 0.5
                    coords[i] -= direction * push
                    coords[j] += direction * push
        if not any_collision:
            break

    ts_xyz = {'symbols': symbols,
              'isotopes': uni_xyz.get('isotopes', tuple(0 for _ in range(len(symbols)))),
              'coords': tuple(tuple(float(x) for x in row) for row in coords)}
    if colliding_atoms(ts_xyz):
        return None
    return ts_xyz


def stretch_bond(uni_xyz: dict,
                  uni_mol: Molecule,
                  split_bonds: list[tuple[int, int]],
                  cross_bonds: list[tuple[int, int]] | None = None,
                  weight: float = 0.5,
                  label: str = '',
                  path_spec: ReactionPathSpec | None = None,
                  family: str | None = None,
                  ) -> dict | None:
    """
    Create a TS guess by stretching specified bonds in the unimolecular species.

    For 2-fragment splits (simple dissociation), the smaller fragment is rigidly
    translated away along the inter-fragment axis so that each split bond
    reaches its Pauling TS estimate (single-bond length + ``PAULING_DELTA``).

    For 3+ fragment splits with a cross bond (insertion/elimination pattern),
    the insertion-ring geometry builder is used instead: the mobile and migrating
    fragments are repositioned to form a 3-membered TS ring.

    Args:
        uni_xyz (dict): XYZ coordinates of the unimolecular species.
        uni_mol (Molecule): RMG Molecule of the unimolecular species.
        split_bonds (list[tuple[int, int]]): Bonds to remove to fragment the molecule.
        cross_bonds (list[tuple[int, int]], optional): Bonds that connect atoms
            across different fragments (used for insertion-ring detection).
        weight (float): Interpolation weight (0=reactant-like, 1=product-like).
        label (str): Logging label.
        path_spec (ReactionPathSpec, optional): path-spec forwarded to
            :func:`validate_addition_guess`, which routes validation through
            the path-spec-aware checker when ``path_spec`` is provided, and
            falls back to the generic TS validator otherwise.
        family (str, optional): Reaction family label for validation context.

    Returns:
        dict | None: TS guess XYZ, or None if validation fails.
    """
    n_atoms = len(uni_xyz['symbols'])
    cross_bonds = cross_bonds or []

    adj = mol_to_adjacency(uni_mol)
    for a, b in split_bonds:
        adj[a].discard(b)
        adj[b].discard(a)

    visited: set[int] = set()
    fragments: list[set[int]] = []
    for start in range(n_atoms):
        if start in visited:
            continue
        component: set[int] = set()
        queue: deque = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            queue.extend(adj[node] - visited)
        fragments.append(component)

    if len(fragments) < 2:
        logger.debug(f'Linear addition ({label}): split bonds did not fragment molecule.')
        return None

    # ---- 3-membered insertion-ring pattern ----
    if len(fragments) >= 3 and cross_bonds:
        result = try_insertion_ring(uni_xyz, uni_mol, fragments, split_bonds,
                                    cross_bonds, weight, n_atoms,
                                    path_spec=path_spec,
                                    family=family)
        if result is not None:
            return result

    # ---- Simple stretch: translate smallest fragment away ----
    fragments.sort(key=len)
    small_frag = fragments[0]
    large_frag: set[int] = set()
    for f in fragments[1:]:
        large_frag.update(f)

    small_anchors: list[int] = []
    large_anchors: list[int] = []
    for a, b in split_bonds:
        if a in small_frag:
            small_anchors.append(a)
            large_anchors.append(b)
        elif b in small_frag:
            small_anchors.append(b)
            large_anchors.append(a)

    if not small_anchors:
        logger.debug(f'Linear addition ({label}): no anchors in small fragment.')
        return None

    coords = np.array(uni_xyz['coords'], dtype=float)
    small_centroid = np.mean(coords[small_anchors], axis=0)
    large_centroid = np.mean(coords[large_anchors], axis=0)

    direction = small_centroid - large_centroid
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    direction /= norm

    target_dists, current_dists= [], []
    for a, b in split_bonds:
        if a in small_frag or b in small_frag:
            sym_a = uni_xyz['symbols'][a]
            sym_b = uni_xyz['symbols'][b]
            sbl = get_single_bond_length(sym_a, sym_b)
            target_dists.append(sbl + PAULING_DELTA)
            current_dists.append(float(np.linalg.norm(coords[a] - coords[b])))

    if not target_dists:
        return None

    avg_target = float(np.mean(target_dists))
    avg_current = float(np.mean(current_dists))
    delta = (avg_target - avg_current) * 2.0 * (1.0 - weight)
    if delta < 0:
        delta = 0.0

    ts_coords = coords.copy()
    displacement = direction * delta
    for idx in small_frag:
        ts_coords[idx] += displacement

    ts_xyz: dict = {'symbols': uni_xyz['symbols'],
                    'isotopes': uni_xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
                    'coords': tuple(tuple(float(x) for x in row) for row in ts_coords)}

    # split_bonds are the forming bonds from the TS perspective (addition direction).
    is_valid, reason = validate_addition_guess(
        xyz=ts_xyz, uni_mol=uni_mol, forming_bonds=split_bonds,
        label=label, path_spec=path_spec)
    if not is_valid:
        logger.debug(f'Linear addition ({label}): rejected — {reason}.')
        return None

    return ts_xyz


def try_insertion_ring(uni_xyz: dict,
                       uni_mol: Molecule,
                       fragments: list[set[int]],
                       split_bonds: list[tuple[int, int]],
                       cross_bonds: list[tuple[int, int]],
                       weight: float,
                       n_atoms: int,
                       path_spec: ReactionPathSpec | None = None,
                       family: str | None = None,
                       ) -> dict | None:
    """
    Attempt to build a 3-membered insertion-ring TS geometry.

    When fragmenting the unimolecular species yields 3+ fragments and there
    is a cross bond connecting two of the split-bond partners, we have an
    insertion/elimination pattern (e.g. 1,2_Insertion_CO: A-B → A + :C: + B
    where C inserts between A and B).

    Args:
        uni_xyz (dict): XYZ of the unimolecular species.
        uni_mol (Molecule): RMG Molecule of the unimolecular species.
        fragments (list[set[int]]): Connected components after removing split bonds.
        split_bonds (list[tuple[int, int]]): Removed bonds.
        cross_bonds (list[tuple[int, int]]): Bonds connecting atoms across fragments.
        weight (float): Interpolation weight.
        n_atoms (int): Total number of atoms.
        path_spec (ReactionPathSpec, optional): path-spec forwarded to
            :func:`validate_addition_guess`, which routes validation through
            the path-spec-aware checker when provided.
        family (str, optional): Reaction family label for validation context.

    Returns:
        dict | None: TS guess XYZ, or None if the pattern doesn't apply.
    """
    # ---- Identify central atom (appears in 2+ split bonds) ----
    sb_atom_count: dict[int, int] = {}
    for a, b in split_bonds:
        sb_atom_count[a] = sb_atom_count.get(a, 0) + 1
        sb_atom_count[b] = sb_atom_count.get(b, 0) + 1
    central_atom: int | None = None
    for atom, count in sb_atom_count.items():
        if count >= 2:
            central_atom = atom
            break

    if central_atom is None:
        return None

    partners: list[int] = []
    for a, b in split_bonds:
        if a == central_atom:
            partners.append(b)
        elif b == central_atom:
            partners.append(a)

    # ---- Substrate / migrating atoms: partners joined by a cross bond ----
    sub_atom: int | None = None
    mig_atom: int | None = None
    for pi_idx in range(len(partners)):
        for pj_idx in range(pi_idx + 1, len(partners)):
            pi, pj = partners[pi_idx], partners[pj_idx]
            if any((a == pi and b == pj) or (a == pj and b == pi) for a, b in cross_bonds):
                frag_pi = next(f for f in fragments if pi in f)
                frag_pj = next(f for f in fragments if pj in f)
                if len(frag_pi) >= len(frag_pj):
                    sub_atom, mig_atom = pi, pj
                else:
                    sub_atom, mig_atom = pj, pi
                break
        if sub_atom is not None:
            break

    if sub_atom is None or mig_atom is None:
        return None

    central_frag = next(f for f in fragments if central_atom in f)
    sub_frag = next(f for f in fragments if sub_atom in f)
    mig_frag = next(f for f in fragments if mig_atom in f)

    if len(central_frag) <= len(sub_frag):
        mobile_frag, mobile_atom, anchor_atom = central_frag, central_atom, sub_atom
    else:
        mobile_frag, mobile_atom, anchor_atom = sub_frag, sub_atom, central_atom

    coords = np.array(uni_xyz['coords'], dtype=float)
    ts_coords = coords.copy()

    family_for_calibration = family if family is not None else (
        path_spec.family if path_spec is not None else None)
    extra_stretch = insertion_ring_extra_stretch(family_for_calibration)

    sym_mob = uni_xyz['symbols'][mobile_atom]
    sym_anch = uni_xyz['symbols'][anchor_atom]
    target_ma = get_single_bond_length(sym_mob, sym_anch) + PAULING_DELTA + extra_stretch
    vec_ma = coords[mobile_atom] - coords[anchor_atom]
    dist_ma = float(np.linalg.norm(vec_ma))
    if dist_ma < 1e-6:
        return None
    dir_ma = vec_ma / dist_ma
    delta_ma = (target_ma - dist_ma) * 2.0 * (1.0 - weight)
    for idx in mobile_frag:
        ts_coords[idx] += dir_ma * delta_ma

    sym_mig = uni_xyz['symbols'][mig_atom]
    target_mob_mig = get_single_bond_length(sym_mob, sym_mig) + PAULING_DELTA + extra_stretch
    target_anch_mig = get_single_bond_length(sym_anch, sym_mig) + PAULING_DELTA + extra_stretch

    pos_mob = ts_coords[mobile_atom].copy()
    pos_anch = ts_coords[anchor_atom].copy()
    pos_mig = ts_coords[mig_atom].copy()
    d12 = float(np.linalg.norm(pos_mob - pos_anch))

    if d12 < 1e-6:
        return None
    axis = (pos_mob - pos_anch) / d12
    x_along = (d12 ** 2 + target_anch_mig ** 2 - target_mob_mig ** 2) / (2.0 * d12)
    h = float(np.sqrt(max(target_anch_mig ** 2 - x_along ** 2, 0.0)))

    proj_on_axis = pos_anch + axis * x_along
    mig_proj_t = float(np.dot(pos_mig - pos_anch, axis))
    mig_foot = pos_anch + axis * mig_proj_t
    perp = pos_mig - mig_foot
    perp_norm = float(np.linalg.norm(perp))
    if perp_norm < 1e-6:
        perp = np.cross(axis, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(axis, np.array([0.0, 1.0, 0.0]))
        perp /= np.linalg.norm(perp)
    else:
        perp /= perp_norm

    new_mig_pos = proj_on_axis + perp * h
    interp = 2.0 * (1.0 - weight)
    new_mig_pos = pos_mig + (new_mig_pos - pos_mig) * interp
    mig_disp = new_mig_pos - ts_coords[mig_atom]
    for idx in mig_frag:
        ts_coords[idx] += mig_disp

    ts_xyz: dict = {'symbols': uni_xyz['symbols'],
                    'isotopes': uni_xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
                    'coords': tuple(tuple(float(c) for c in row) for row in ts_coords)}
    is_valid, reason = validate_addition_guess(xyz=ts_xyz, uni_mol=uni_mol, forming_bonds=split_bonds,
                                               label='insertion-ring', path_spec=path_spec)
    if not is_valid:
        if extra_stretch > 0.0 and 'too many fragments' in (reason or ''):
            relaxed_max_heavy = 2.0 + extra_stretch + 0.10
            if not has_too_many_fragments(
                    ts_xyz, max_heavy_heavy=relaxed_max_heavy):
                if (not colliding_atoms(ts_xyz)
                        and not has_detached_hydrogen(ts_xyz, max_h_heavy_dist=3.0)):
                    logger.debug(f'Linear (insertion-ring): calibration ({family_for_calibration},'
                                 f' +{extra_stretch:.2f} Å): accepting via relaxed '
                                 f'heavy-heavy threshold {relaxed_max_heavy:.2f} Å '
                                 f'after generic-validator fragments rejection.')
                    return ts_xyz
        logger.debug(f'Linear (insertion-ring): rejected (family={family_for_calibration}, '
                     f'extra_stretch={extra_stretch}) — {reason}.')
        return None
    return ts_xyz


def stretch_core_from_large(ts_xyz: dict,
                             split_bonds: list[tuple[int, int]],
                             core: set[int],
                             large_prod_atoms: set[int],
                             weight: float = 0.5,
                             ) -> dict:
    """
    Stretch the core of the small product away from the large product.

    When fragmenting the reactant produces 3+ fragments (e.g. in elimination
    reactions), ``stretch_bond`` only moves the smallest fragment. This
    function handles the remaining split bonds: those that connect the *core*
    of the small product to the large product.

    The core atoms (and any migrating atoms riding on the same fragment) are
    rigidly translated so that the relevant split bonds reach TS-like distances.

    Args:
        ts_xyz: Current TS guess XYZ (after ``stretch_bond``).
        split_bonds: All breaking bonds.
        core: Atom indices forming the connected core of the small product.
        large_prod_atoms: Atom indices belonging to the large product.
        weight: Interpolation weight (0 = reactant-like, 1 = product-like).

    Returns:
        dict: Modified XYZ with core atoms translated.
    """
    symbols = ts_xyz['symbols']
    coords = np.array(ts_xyz['coords'], dtype=float)

    core_anchors: list[int] = []
    large_anchors: list[int] = []
    for a, b in split_bonds:
        if a in core and b in large_prod_atoms:
            core_anchors.append(a)
            large_anchors.append(b)
        elif b in core and a in large_prod_atoms:
            core_anchors.append(b)
            large_anchors.append(a)

    if not core_anchors:
        return ts_xyz

    core_centroid = np.mean(coords[core_anchors], axis=0)
    large_centroid = np.mean(coords[large_anchors], axis=0)
    direction = core_centroid - large_centroid
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return ts_xyz
    direction /= norm

    target_dists, current_dists = [], []
    for ca, la in zip(core_anchors, large_anchors):
        sbl = get_single_bond_length(symbols[ca], symbols[la])
        target_dists.append(sbl + PAULING_DELTA)
        current_dists.append(float(np.linalg.norm(coords[ca] - coords[la])))

    avg_target = float(np.mean(target_dists))
    avg_current = float(np.mean(current_dists))
    delta = (avg_target - avg_current) * 2.0 * (1.0 - weight)
    if delta < 0:
        delta = 0.0

    # Move the entire core; migrating atoms are handled separately.
    for idx in core:
        coords[idx] += direction * delta

    return {'symbols': ts_xyz['symbols'],
            'isotopes': ts_xyz.get('isotopes', tuple(0 for _ in range(len(symbols)))),
            'coords': tuple(tuple(float(x) for x in row) for row in coords)}


def migrate_verified_atoms(ts_xyz: dict,
                           uni_mol: Molecule,
                           migrating_atoms: set[int],
                           core: set[int],
                           large_prod_atoms: set[int],
                           cross_bonds: list[tuple[int, int]] | None = None,
                           ) -> dict:
    """
    Migrate specific atoms identified by ``map_and_verify_fragments``.

    Unlike ``migrate_h_between_fragments`` (which guesses which H to move by
    composition matching), this function moves exactly the atoms in
    *migrating_atoms*: the set of atom indices that belong to one product but
    are bonded only to atoms in the other product in the reactant graph.

    Each migrating atom is placed at a TS-like position between its current
    heavy-atom donor (in *large_prod_atoms*) and its acceptor in *core*,
    using triangulation when the spheres overlap. The acceptor is determined
    from *cross_bonds* (forming bonds) when available, falling back to the
    nearest heavy atom in *core*.

    Args:
        ts_xyz: TS guess XYZ (already stretched by ``stretch_bond``).
        uni_mol: RMG Molecule of the unimolecular species.
        migrating_atoms: Atom indices that need to move between product groups.
        core: Atom indices forming the connected core of the small product.
        large_prod_atoms: Atom indices belonging to the large product.
        cross_bonds: Forming bonds absent from uni_mol (used to identify the exact acceptor atom for each migrating atom).

    Returns:
        dict: Modified XYZ with migrating atoms partially displaced.
    """
    symbols = ts_xyz['symbols']
    coords = np.array(ts_xyz['coords'], dtype=float)
    ts_coords = coords.copy()
    atom_to_idx = {atom: idx for idx, atom in enumerate(uni_mol.atoms)}
    core_heavy = [idx for idx in core if symbols[idx] != 'H']

    # Acceptor may live in any product fragment, not only ``core`` (e.g.
    # Korcek_step2 has intra-large H migrations); the cross-bond heavy partner
    # is authoritative.
    cross_acceptor: dict[int, int] = {}
    for a, b in (cross_bonds or []):
        if a in migrating_atoms and symbols[b] != 'H':
            cross_acceptor[a] = b
        elif b in migrating_atoms and symbols[a] != 'H':
            cross_acceptor[b] = a

    for h_idx in migrating_atoms:
        # Prefer donor in ``large_prod_atoms``, fall back to any heavy reactant
        # neighbor to cover intra-large H migrations (e.g. Korcek_step2).
        donor = None
        for nbr in uni_mol.atoms[h_idx].bonds.keys():
            nbr_idx = atom_to_idx[nbr]
            if symbols[nbr_idx] != 'H' and nbr_idx in large_prod_atoms:
                donor = nbr_idx
                break
        if donor is None:
            for nbr in uni_mol.atoms[h_idx].bonds.keys():
                nbr_idx = atom_to_idx[nbr]
                if symbols[nbr_idx] != 'H':
                    donor = nbr_idx
                    break
        if donor is None:
            continue

        acceptor = cross_acceptor.get(h_idx)
        if acceptor is None:
            if not core_heavy:
                continue
            core_coords = ts_coords[core_heavy]
            dists = np.linalg.norm(core_coords - ts_coords[h_idx], axis=1)
            acceptor = core_heavy[int(dists.argmin())]

        d_pos = ts_coords[donor]
        a_pos = ts_coords[acceptor]
        h_pos = ts_coords[h_idx]
        da_vec = a_pos - d_pos
        da_dist = float(np.linalg.norm(da_vec))
        if da_dist < 1e-6:
            continue
        da_hat = da_vec / da_dist

        d_DH = get_single_bond_length(symbols[donor], symbols[h_idx]) + PAULING_DELTA
        d_AH = get_single_bond_length(symbols[acceptor], symbols[h_idx]) + PAULING_DELTA

        if da_dist <= d_DH + d_AH:
            # Spheres overlap — triangulate.
            x = (da_dist ** 2 + d_DH ** 2 - d_AH ** 2) / (2.0 * da_dist)
            h_sq = d_DH ** 2 - x ** 2
            h_perp = np.sqrt(max(h_sq, 0.0))
            proj = d_pos + np.dot(h_pos - d_pos, da_hat) * da_hat
            perp = h_pos - proj
            perp_norm = float(np.linalg.norm(perp))
            if perp_norm > 1e-8:
                n_perp = perp / perp_norm
            else:
                arb = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(da_hat, arb)) > 0.9:
                    arb = np.array([0.0, 1.0, 0.0])
                n_perp = np.cross(da_hat, arb)
                n_perp /= np.linalg.norm(n_perp)
            ideal = d_pos + da_hat * x + n_perp * h_perp
        else:
            # Spheres don't overlap — place on the D-A axis at d_DH from donor.
            ideal = d_pos + da_hat * d_DH

        # Direct placement: Cartesian interpolation from the reactant would pass
        # through the donor when the H swings around (e.g. face-to-face on O).
        ts_coords[h_idx] = ideal

    return {'symbols': ts_xyz['symbols'],
            'isotopes': ts_xyz.get('isotopes', tuple(0 for _ in range(len(symbols)))),
            'coords': tuple(tuple(float(x) for x in row) for row in ts_coords)}


def migrate_h_between_fragments(ts_xyz: dict,
                                 uni_mol: Molecule,
                                 split_bonds: list[tuple[int, int]],
                                 product_species: list[ARCSpecies],
                                 ) -> dict:
    """
    Partially displace H atoms that need to migrate between fragments
    to match the product species' compositions.

    After ``stretch_bond`` rigidly translates fragments apart, H atoms
    remain on their original fragment. In reactions where H redistribution
    occurs (e.g. 1,3_Insertion_CO2: R-C(=O)OH → R-H + O=C=O), the TS
    should show the migrating H partially displaced toward its destination.

    This function:

    1. Identifies fragments from the split bonds.
    2. Computes element compositions for each fragment.
    3. Compares with product compositions to find H surplus/deficit.
    4. For each surplus fragment, finds the H atom closest to the deficit
       fragment and places it on the donor→acceptor axis at a TS-like
       distance (interpolated by ``weight``). Using the donor→acceptor
       axis instead of a direct H→acceptor line avoids near-collisions
       with other atoms in the source fragment (e.g. the C in a CO₂ group).

    Args:
        ts_xyz: TS guess XYZ from ``stretch_bond`` (already stretched).
        uni_mol: RMG Molecule of the unimolecular species.
        split_bonds: Bonds that were cut to create fragments.
        product_species: Product species for composition matching.

    Returns:
        dict: Modified XYZ with H atoms partially migrated, or the original XYZ unchanged if no migration is needed.
    """
    n_atoms = len(ts_xyz['symbols'])
    n_products = len(product_species)

    # ---- 1. Build adjacency and fragment ----
    atom_to_idx = {atom: idx for idx, atom in enumerate(uni_mol.atoms)}
    adj = mol_to_adjacency(uni_mol)
    for a, b in split_bonds:
        adj[a].discard(b)
        adj[b].discard(a)

    visited: set[int] = set()
    fragments: list[set[int]] = []
    for start in range(n_atoms):
        if start in visited:
            continue
        component: set[int] = set()
        queue: deque = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            queue.extend(adj[node] - visited)
        fragments.append(component)

    if len(fragments) != n_products:
        return ts_xyz

    # ---- 2. Fragment and target element formulas ----
    symbols = ts_xyz['symbols']
    frag_formulas: list[dict[str, int]] = []
    for frag in fragments:
        formula: dict[str, int] = {}
        for idx in frag:
            sym = symbols[idx]
            formula[sym] = formula.get(sym, 0) + 1
        frag_formulas.append(formula)

    target_formulas: list[dict[str, int]] = []
    for sp in product_species:
        formula: dict[str, int] = {}
        for sym in sp.get_xyz()['symbols']:
            formula[sym] = formula.get(sym, 0) + 1
        target_formulas.append(formula)

    # ---- 3. Match fragments to targets by heavy-atom composition ----
    frag_to_target: dict[int, int] = {}
    used_targets: set[int] = set()
    for fi, ff in enumerate(frag_formulas):
        hf = _heavy_only_formula(ff)
        for ti, tf in enumerate(target_formulas):
            if ti not in used_targets and _heavy_only_formula(tf) == hf:
                frag_to_target[fi] = ti
                used_targets.add(ti)
                break

    if len(frag_to_target) != n_products:
        return ts_xyz

    # ---- 4. Per-fragment H surplus/deficit ----
    h_surplus: dict[int, int] = {}
    for fi in range(n_products):
        ti = frag_to_target[fi]
        frag_h = frag_formulas[fi].get('H', 0)
        target_h = target_formulas[ti].get('H', 0)
        diff = frag_h - target_h
        if diff != 0:
            h_surplus[fi] = diff  # positive = surplus, negative = deficit

    if not h_surplus:
        return ts_xyz

    coords = np.array(ts_xyz['coords'], dtype=float)
    ts_coords = coords.copy()

    # ---- 5. Migrate H atoms across surplus/deficit pairs ----
    surplus_frags = [fi for fi, d in h_surplus.items() if d > 0]
    deficit_frags = [fi for fi, d in h_surplus.items() if d < 0]

    for s_fi in surplus_frags:
        n_to_move = h_surplus[s_fi]
        h_indices = [idx for idx in fragments[s_fi] if symbols[idx] == 'H']
        if not h_indices or not deficit_frags:
            continue

        deficit_heavy: list[int] = []
        for d_fi in deficit_frags:
            deficit_heavy.extend(idx for idx in fragments[d_fi] if symbols[idx] != 'H')
        if not deficit_heavy:
            continue

        deficit_heavy_coords = ts_coords[deficit_heavy]

        # Migrating H should come from a non-anchor heavy atom to form a proper
        # TS ring (e.g. O on one C of ethylene, H migrating from the other C).
        split_anchors_in_frag: set[int] = set()
        for a, b in split_bonds:
            if a in fragments[s_fi]:
                split_anchors_in_frag.add(a)
            if b in fragments[s_fi]:
                split_anchors_in_frag.add(b)

        # Sort: prefer H not bonded to a split-bond anchor, then by distance
        # to nearest deficit-fragment heavy atom.
        h_dists: list[tuple[int, float, int, bool]] = []
        for h_idx in h_indices:
            dists = np.linalg.norm(deficit_heavy_coords - ts_coords[h_idx], axis=1)
            min_dist = float(dists.min())
            nearest_heavy = deficit_heavy[int(dists.argmin())]
            on_anchor = any(atom_to_idx[nbr] in split_anchors_in_frag for nbr in uni_mol.atoms[h_idx].bonds.keys())
            h_dists.append((h_idx, min_dist, nearest_heavy, on_anchor))
        h_dists.sort(key=lambda x: (x[3], x[1]))

        for h_idx, _, nearest_heavy, _ in h_dists[:n_to_move]:
            donor_heavy = None
            for nbr in uni_mol.atoms[h_idx].bonds.keys():
                nbr_idx = atom_to_idx[nbr]
                if symbols[nbr_idx] != 'H' and nbr_idx in fragments[s_fi]:
                    donor_heavy = nbr_idx
                    break

            if donor_heavy is None:
                continue

            # Triangulate a non-collinear D-H-A to avoid passing through atoms
            # between donor and acceptor (e.g. the C in a CO₂ group).
            d_pos = ts_coords[donor_heavy]
            a_pos = ts_coords[nearest_heavy]
            h_pos = ts_coords[h_idx]
            da_vec = a_pos - d_pos
            da_dist = float(np.linalg.norm(da_vec))
            if da_dist < 1e-6:
                continue
            da_hat = da_vec / da_dist

            d_DH = get_single_bond_length(symbols[donor_heavy], 'H') + PAULING_DELTA
            d_AH = get_single_bond_length(symbols[nearest_heavy], 'H') + PAULING_DELTA

            if da_dist <= d_DH + d_AH:
                # Spheres overlap → triangulate.
                x = (da_dist ** 2 + d_DH ** 2 - d_AH ** 2) / (2.0 * da_dist)
                h_sq = d_DH ** 2 - x ** 2
                h_perp = np.sqrt(max(h_sq, 0.0))

                proj = d_pos + np.dot(h_pos - d_pos, da_hat) * da_hat
                perp = h_pos - proj
                perp_norm = float(np.linalg.norm(perp))
                if perp_norm > 1e-8:
                    n_perp = perp / perp_norm
                else:
                    ref = np.array([1.0, 0.0, 0.0]) if abs(da_hat[0]) < 0.9 \
                        else np.array([0.0, 1.0, 0.0])
                    n_perp = np.cross(da_hat, ref)
                    n_perp /= np.linalg.norm(n_perp)

                cand_plus = d_pos + x * da_hat + h_perp * n_perp
                cand_minus = d_pos + x * da_hat - h_perp * n_perp

                # Pick candidate with greater clearance from source-fragment heavies.
                exclude_indices = {h_idx, donor_heavy}
                clearance_plus = _min_distance_to_heavy_atoms_in_set(
                    cand_plus, fragments[s_fi], ts_coords, symbols, exclude_indices)
                clearance_minus = _min_distance_to_heavy_atoms_in_set(
                    cand_minus, fragments[s_fi], ts_coords, symbols, exclude_indices)
                new_h = cand_plus if clearance_plus >= clearance_minus else cand_minus
            else:
                # Spheres don't overlap → collinear placement at d_DH from donor.
                new_h = d_pos + d_DH * da_hat

            ts_coords[h_idx] = new_h

    result = {'symbols': ts_xyz['symbols'],
              'isotopes': ts_xyz['isotopes'],
              'coords': tuple(tuple(float(x) for x in row) for row in ts_coords)}

    if colliding_atoms(result):
        return ts_xyz
    return result


def _reposition_leaving_groups(xyz: dict,
                               pre_xyz: dict,
                               split_bonds: list[tuple[int, int]],
                               frag_id: list[int],
                               n_atoms: int,
                               extra_stretch: float = 0.0,
                               ) -> dict:
    """
    Reposition leaving-group fragments after ring closure.

    Ring closure moves ring-member atoms but leaves disconnected fragments
    (the leaving groups) at their original positions. This can make the
    split-bond distance arbitrarily large or small.

    This function:

    1. Translates the leaving fragment to follow its ring anchor's
       displacement (so relative geometry is preserved).
    2. Stretches the split bond to a TS-appropriate distance along the
       **pre-ring-closure** bond direction (which reliably points away from
       the ring, unlike the post-closure direction which may cross other
       ring atoms).

    Args:
        xyz: Post-ring-closure XYZ geometry.
        pre_xyz: Pre-ring-closure XYZ geometry (for original bond directions).
        split_bonds: Bonds that were cut to create fragments.
        frag_id: Fragment assignment per atom.
        n_atoms: Number of atoms.
        extra_stretch: Additional stretch (Å) beyond the Pauling TS estimate,
            applied when a leaving-group departure coincides with ring closure.

    Returns:
        Modified XYZ dict with leaving groups repositioned.
    """
    coords = np.array(xyz['coords'], dtype=float)
    pre_coords = np.array(pre_xyz['coords'], dtype=float)
    for sb_a, sb_b in split_bonds:
        if frag_id[sb_a] == frag_id[sb_b]:
            continue
        frag_a = [i for i in range(n_atoms) if frag_id[i] == frag_id[sb_a]]
        frag_b = [i for i in range(n_atoms) if frag_id[i] == frag_id[sb_b]]
        if len(frag_a) <= len(frag_b):
            leaving_frag, leaving_anchor, ring_anchor = frag_a, sb_a, sb_b
        else:
            leaving_frag, leaving_anchor, ring_anchor = frag_b, sb_b, sb_a

        current_dist = float(np.linalg.norm(coords[leaving_anchor] - coords[ring_anchor]))
        sym_a = xyz['symbols'][ring_anchor]
        sym_b = xyz['symbols'][leaving_anchor]
        ts_target = get_single_bond_length(sym_a, sym_b) + PAULING_DELTA + extra_stretch

        if current_dist <= ts_target + 0.5:
            continue

        # ---- Step 1: follow ring anchor's displacement from ring closure ----
        anchor_displacement = coords[ring_anchor] - pre_coords[ring_anchor]
        for idx in leaving_frag:
            coords[idx] += anchor_displacement

        # ---- Step 2: stretch along the pre-closure direction (points away from ring) ----
        pre_vec = pre_coords[leaving_anchor] - pre_coords[ring_anchor]
        pre_dist = float(np.linalg.norm(pre_vec))
        if pre_dist < 1e-6:
            continue
        direction = pre_vec / pre_dist
        ideal_pos = coords[ring_anchor] + direction * ts_target
        shift = ideal_pos - coords[leaving_anchor]
        for idx in leaving_frag:
            coords[idx] += shift

    return {'symbols': xyz['symbols'],
            'isotopes': xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
            'coords': tuple(tuple(float(x) for x in row) for row in coords)}


def apply_intra_frag_contraction(xyz: dict,
                                 mol: Molecule,
                                 split_bonds: list[tuple[int, int]],
                                 multi_species: list[ARCSpecies],
                                 weight: float = 0.5,
                                 label: str = '',
                                 ) -> list[dict]:
    """
    Apply angular ring contraction for intra-fragment forming bonds.

    After ``stretch_bond()`` separates fragments by stretching the split bonds,
    any forming bond whose two atoms remain in the same fragment requires
    angular contraction to bring them closer together. This function
    identifies such bonds and applies ``ring_closure_xyz()`` for each one,
    returning a separate TS guess per candidate forming bond.

    Forming bonds are detected from product ring topology via
    ``detect_intra_frag_ring_bonds()``. Inter-fragment forming bonds (the
    ``cross_bonds`` used elsewhere in the addition pipeline) are orthogonal
    to this function's concern and are handled by ``stretch_bond()``.

    When multiple candidates exist (e.g. due to resonance-equivalent atom
    assignments), each produces an independent TS guess so that the best
    one can be selected downstream.

    Args:
        xyz: Post-stretch XYZ geometry.
        mol: RMG Molecule of the unimolecular species.
        split_bonds: Bonds severed by ``stretch_bond()``.
        multi_species: Product species (multi-species side).
        weight: Interpolation weight (0 = no contraction, 0.5 = TS-like).
        label: Debug label for logging.

    Returns:
        List of modified XYZ dicts with ring contraction applied.
        Returns ``[xyz]`` (the original geometry) if no contraction was
        needed or applicable.
    """
    forming = detect_intra_frag_ring_bonds(mol, split_bonds, multi_species, xyz)
    if not forming:
        return [xyz]

    # ---- Adjacency excluding split bonds; label fragments ----
    n_atoms = len(mol.atoms)
    adj = mol_to_adjacency(mol)
    for a, b in split_bonds:
        adj[a].discard(b)
        adj[b].discard(a)

    frag_id: list[int] = [-1] * n_atoms
    fid = 0
    for start in range(n_atoms):
        if frag_id[start] >= 0:
            continue
        queue: deque = deque([start])
        while queue:
            node = queue.popleft()
            if frag_id[node] >= 0:
                continue
            frag_id[node] = fid
            queue.extend(n for n in adj[node] if frag_id[n] < 0)
        fid += 1

    split_endpoints: set[int] = set()
    for sb in split_bonds:
        split_endpoints.update(sb)

    # A fragment with ≤ _MAX_LG_HEAVY heavies across a split bond is a leaving
    # group (e.g. CH3 in ExoTetCyclic).
    _MAX_LG_HEAVY = 1
    has_small_leaving_frag = False
    for sb_a, sb_b in split_bonds:
        if frag_id[sb_a] == frag_id[sb_b]:
            continue
        heavy_a = sum(1 for i in range(n_atoms)
                      if frag_id[i] == frag_id[sb_a] and mol.atoms[i].symbol != 'H')
        heavy_b = sum(1 for i in range(n_atoms)
                      if frag_id[i] == frag_id[sb_b] and mol.atoms[i].symbol != 'H')
        if min(heavy_a, heavy_b) <= _MAX_LG_HEAVY:
            has_small_leaving_frag = True
            break

    results: list[dict] = []
    for bond, ring_size in forming:
        a, b = bond
        if a >= n_atoms or b >= n_atoms:
            continue
        if frag_id[a] != frag_id[b]:
            continue
        if b in adj[a]:
            continue

        coords = np.array(xyz['coords'], dtype=float)
        current_dist = float(np.linalg.norm(coords[a] - coords[b]))
        sym_a = xyz['symbols'][a]
        sym_b = xyz['symbols'][b]
        sbl = get_single_bond_length(sym_a, sym_b)
        # Strained (3-/4-membered) rings have an earlier TS → longer forming bond.
        ring_correction = 0.15 if ring_size == 3 else (0.08 if ring_size == 4 else 0.0)
        # Ring-closure endpoint doubling as a split-bond endpoint → earlier TS.
        has_leaving_group = bool(has_small_leaving_frag and ({a, b} & split_endpoints) and ring_correction == 0.0)
        leaving_group_correction = 0.25 if has_leaving_group else 0.0
        ts_target = sbl + PAULING_DELTA + ring_correction + leaving_group_correction
        target = current_dist + (ts_target - current_dist) * min(2.0 * weight, 1.0)
        if target >= current_dist - 0.01:
            continue

        contracted = ring_closure_xyz(
            xyz, mol, forming_bond=bond,
            target_distance=target, exclude_bonds=split_bonds)
        if contracted is not None:
            logger.debug(f'Linear addition ({label}): ring contraction applied for '
                         f'bond ({a},{b}), {current_dist:.2f} → {target:.2f} Å.')
            # Only reposition stranded (too-far) leaving groups; a too-close
            # one is handled by downstream ``stretch_bond()``.
            if has_leaving_group:
                contracted = _reposition_leaving_groups(contracted, xyz, split_bonds, frag_id, n_atoms,
                                                        extra_stretch=leaving_group_correction)
            results.append(contracted)
    return results if results else [xyz]


def detect_intra_frag_ring_bonds(mol: Molecule,
                                 split_bonds: list[tuple[int, int]],
                                 multi_species: list[ARCSpecies],
                                 xyz: dict,
                                 ) -> list[tuple[tuple[int, int], int]]:
    """
    Detect bonds that should form within a fragment (ring closure) to match product ring topology.

    After severing ``split_bonds``, each remaining connected component is a
    fragment. If any product species is cyclic, this function searches for
    pairs of non-bonded heavy atoms in each fragment that are connected by a
    short path whose length matches a product ring size. Such pairs are
    likely forming bonds (ring closures that occur simultaneously with the
    split-bond scission).

    Candidates are sorted by descending current distance so that the
    longest (least-advanced) forming bond is attempted first.

    Args:
        mol (Molecule): RMG Molecule of the unimolecular species.
        split_bonds (list[tuple[int, int]]): Bonds already severed by ``stretch_bond()``.
        multi_species (list[ARCSpecies]): Product species on the multi-species side.
        xyz (dict): Current XYZ coordinates (used for distance-based sorting).

    Returns:
        list[tuple[tuple[int, int], int]]: Intra-fragment forming-bond candidates as ``((i, j), ring_size)`` pairs,
            where ``i < j`` and ``ring_size`` is the BFS path length (number of ring atoms).
    """
    product_ring_sizes: set[int] = set()
    for sp in multi_species:
        if sp.mol.is_cyclic():
            try:
                for ring in sp.mol.get_smallest_set_of_smallest_rings():
                    product_ring_sizes.add(len(ring))
            except Exception as e:
                logger.debug(f"Could not determine product ring sizes for species "
                             f"{getattr(sp, 'label', 'unknown')!r}; proceeding without "
                             f"ring-size constraints. Error: {type(e).__name__}: {e}", exc_info=True)
    if not product_ring_sizes:
        return []

    n_atoms = len(mol.atoms)
    adj = mol_to_adjacency(mol)
    for a, b in split_bonds:
        adj[a].discard(b)
        adj[b].discard(a)

    visited: set[int] = set()
    fragments: list[set[int]] = []
    for start in range(n_atoms):
        if start in visited:
            continue
        comp: set[int] = set()
        queue: deque = deque([start])
        while queue:
            node = queue.popleft()
            if node in comp:
                continue
            comp.add(node)
            queue.extend(n for n in adj[node] if n not in comp)
        visited |= comp
        fragments.append(comp)

    # A forming bond must have at least one endpoint adjacent to a severed edge.
    split_endpoints: set[int] = set()
    for sb in split_bonds:
        for atom_idx in sb:
            split_endpoints.add(atom_idx)

    candidates: list[tuple[tuple[int, int], list[int]]] = []
    for frag in fragments:
        heavy = sorted(i for i in frag if mol.atoms[i].symbol != 'H')
        for ai, a in enumerate(heavy):
            for b in heavy[ai + 1:]:
                if b in adj[a]:
                    continue
                if a not in split_endpoints and b not in split_endpoints:
                    continue
                path = bfs_path(adj, a, b)
                if path is not None and len(path) in product_ring_sizes:
                    candidates.append(((min(a, b), max(a, b)), path))

    if not candidates:
        return []

    # Require path element multiset to match a product ring (reject e.g.
    # all-carbon paths when the product ring contains O/S).
    product_ring_elements: list[tuple[str, ...]] = []
    for sp in multi_species:
        if sp.mol.is_cyclic():
            try:
                for ring in sp.mol.get_smallest_set_of_smallest_rings():
                    elems = tuple(sorted(a.symbol for a in ring))
                    product_ring_elements.append(elems)
            except Exception as e:
                logger.debug(
                    f"Could not extract smallest rings for species "
                    f"{getattr(sp, 'label', '<unknown>')}: {e}"
                )

    coords = np.array(xyz['coords'], dtype=float)
    sort_key = partial(_negative_bond_distance, coords)

    if product_ring_elements:
        filtered = []
        for bond, path in candidates:
            path_elems = tuple(sorted(mol.atoms[i].symbol for i in path))
            if path_elems in product_ring_elements:
                filtered.append((bond, len(path)))
        if filtered:
            filtered.sort(key=sort_key)
            return filtered

    result = [(bond, len(path)) for bond, path in candidates]
    result.sort(key=sort_key)
    return result
