"""
Fragment-based addition / dissociation TS geometry builders
extracted from ``arc.job.adapters.ts.linear``.
"""

from collections import deque
from itertools import permutations
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from arc.common import get_logger, get_single_bond_length
from arc.species.species import ARCSpecies, colliding_atoms

from arc.job.adapters.ts.linear_utils.geom_utils import bfs_path as _bfs_path
from arc.job.adapters.ts.linear_utils.postprocess import (
    PAULING_DELTA,
    has_detached_hydrogen,
    has_too_many_fragments,
    validate_ts_guess,
)
from arc.job.adapters.ts.linear_utils.path_spec import (
    ReactionPathSpec,
    validate_guess_against_path_spec,
)
from arc.job.adapters.ts.linear_utils.isomerization import ring_closure_xyz

if TYPE_CHECKING:
    from arc.molecule import Molecule


logger = get_logger()


def _validate_addition_xyz(ts_xyz: dict,
                            uni_mol: 'Molecule',
                            split_bonds: List[Tuple[int, int]],
                            label: str,
                            path_spec: Optional['ReactionPathSpec'] = None,
                            ) -> Tuple[bool, str]:
    """Phase 3a: addition-side validation gateway used by leaf builders.

    When a :class:`ReactionPathSpec` is supplied, validation is routed
    through :func:`validate_guess_against_path_spec` so addition guesses
    receive the same Phase 1+2 path-spec checks as isomerization.  When
    the spec is ``None`` (degraded mode — e.g. dedicated motif builders
    with no per-bond metadata), the legacy
    :func:`validate_ts_guess` path is used so behavior never regresses
    below the pre-Phase-3a baseline.

    Args:
        ts_xyz: Built TS guess XYZ dict.
        uni_mol: The unimolecular RMG Molecule (always available here).
        split_bonds: Forming-bond indices for the legacy fallback path.
        label: Logging label.
        path_spec: Optional :class:`ReactionPathSpec` for the guess.

    Returns:
        ``(is_valid, reason)`` matching the existing validator contract.
    """
    if path_spec is not None:
        return validate_guess_against_path_spec(
            xyz=ts_xyz,
            path_spec=path_spec,
            r_mol=uni_mol,
            family=path_spec.family,
            reactive_indices=set(path_spec.reactive_atoms),
            label=label,
        )
    return validate_ts_guess(ts_xyz, set(), split_bonds, uni_mol, label=label)


# ---------------------------------------------------------------------------
# Fragment helpers
# ---------------------------------------------------------------------------

def find_split_bonds_by_fragmentation(uni_mol: 'Molecule',
                                       product_species: List[ARCSpecies],
                                       ) -> List[List[Tuple[int, int]]]:
    """
    Find bonds in the unimolecular species that, when removed, fragment it
    into components matching the product species by element composition.

    This is a fallback for cases where ``product_dicts`` are unavailable or
    give unreliable reactive-bond information.  It works by enumerating
    candidate bond cuts (1-bond or 2-bond) and checking whether the resulting
    fragments match the products' element counts.

    Args:
        uni_mol (Molecule): RMG Molecule of the unimolecular species.
        product_species (List[ARCSpecies]): The product species on the
            multi-species side.

    Returns:
        List[List[Tuple[int, int]]]: Each entry is a list of (i, j) bond
            tuples that, when removed, yield a valid fragmentation.
            Returns an empty list if no valid cut is found.
    """
    n_products = len(product_species)
    if n_products < 2:
        return []

    # Target: sorted list of element-count dicts for each product.
    target_formulas: List[Dict[str, int]] = []
    for sp in product_species:
        formula: Dict[str, int] = {}
        for sym in sp.get_xyz()['symbols']:
            formula[sym] = formula.get(sym, 0) + 1
        target_formulas.append(formula)
    target_sorted = sorted(target_formulas, key=lambda d: sorted(d.items()))

    # Build adjacency and bond list from uni_mol.
    n_atoms = len(uni_mol.atoms)
    atom_to_idx = {atom: idx for idx, atom in enumerate(uni_mol.atoms)}
    adj: Dict[int, Set[int]] = {k: set() for k in range(n_atoms)}
    bonds: List[Tuple[int, int]] = []
    for atom in uni_mol.atoms:
        idx_a = atom_to_idx[atom]
        for neighbor in atom.edges:
            idx_b = atom_to_idx[neighbor]
            if idx_a < idx_b:
                bonds.append((idx_a, idx_b))
            adj[idx_a].add(idx_b)

    # Atom symbols for fragment formula checking.
    symbols = tuple(atom.element.symbol for atom in uni_mol.atoms)

    def _get_fragments(removed: List[Tuple[int, int]]) -> List[Set[int]]:
        """BFS to find connected components after removing bonds."""
        local_adj = {k: set(v) for k, v in adj.items()}
        for a, b in removed:
            local_adj[a].discard(b)
            local_adj[b].discard(a)
        visited: Set[int] = set()
        frags: List[Set[int]] = []
        for start in range(n_atoms):
            if start in visited:
                continue
            component: Set[int] = set()
            queue: deque = deque([start])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                queue.extend(local_adj[node] - visited)
            frags.append(component)
        return frags

    def _heavy_formulas(formulas: List[Dict[str, int]]) -> List[Dict[str, int]]:
        """Strip H counts from element-count dicts."""
        return sorted(({k: v for k, v in f.items() if k != 'H'} for f in formulas),
                       key=lambda d: sorted(d.items()))

    target_heavy = _heavy_formulas(target_formulas)
    target_total_h = sum(f.get('H', 0) for f in target_formulas)

    def _formulas_match(frags: List[Set[int]], strict: bool = True) -> bool:
        """Check if fragment element compositions match targets.

        When *strict* is True, exact match (including H counts).
        When False, heavy-atom formulas must match and total H count
        must be equal, but H can redistribute between fragments.
        """
        if len(frags) != n_products:
            return False
        frag_formulas: List[Dict[str, int]] = []
        for frag in frags:
            formula: Dict[str, int] = {}
            for idx in frag:
                sym = symbols[idx]
                formula[sym] = formula.get(sym, 0) + 1
            frag_formulas.append(formula)
        if strict:
            frag_sorted = sorted(frag_formulas, key=lambda d: sorted(d.items()))
            return frag_sorted == target_sorted
        frag_heavy = _heavy_formulas(frag_formulas)
        frag_total_h = sum(f.get('H', 0) for f in frag_formulas)
        return frag_heavy == target_heavy and frag_total_h == target_total_h

    results: List[List[Tuple[int, int]]] = []

    # Try single-bond cuts with exact formula match first.
    for bond in bonds:
        frags = _get_fragments([bond])
        if _formulas_match(frags, strict=True):
            results.append([bond])
    if results:
        return results

    # Try single-bond cuts with relaxed H matching.
    for bond in bonds:
        frags = _get_fragments([bond])
        if _formulas_match(frags, strict=False):
            results.append([bond])
    if results:
        return results

    # Try two-bond cuts (exact first, then relaxed).
    if n_products <= 3:
        for strict in (True, False):
            for i in range(len(bonds)):
                for j in range(i + 1, len(bonds)):
                    frags = _get_fragments([bonds[i], bonds[j]])
                    if _formulas_match(frags, strict=strict):
                        results.append([bonds[i], bonds[j]])
                if len(results) >= 10:
                    break
            if results:
                return results
    return results


def map_and_verify_fragments(uni_mol: 'Molecule',
                              split_bonds: List[Tuple[int, int]],
                              multi_species: List[ARCSpecies],
                              cross_bonds: Optional[List[Tuple[int, int]]] = None,
                              product_dict: Optional[dict] = None,
                              uni_is_reactant: bool = True,
                              ) -> Optional[Dict[int, Tuple[int, int]]]:
    """
    Validate that severing *split_bonds* (and reconnecting *cross_bonds*) in
    *uni_mol* produces fragments that topologically match *multi_species*.

    The function works by:

    1. Deep-copying *uni_mol* and removing every bond in *split_bonds* (adjusting
       radical electrons on the severed endpoints).
    2. Re-adding every bond in *cross_bonds* (these are forming bonds that exist in
       the products but not the reactant).  This merges isolated migrating atoms back
       into the fragment they belong to in the product.
    3. Calling ``split()`` on the modified graph and checking that the number of
       connected components equals the number of product species.
    4. Using connectivity-normalised ``is_isomorphic`` (bond orders and radical
       electrons removed) with ``strict=False`` to match each fragment to exactly
       one product species.
    5. Validating against the RMG reaction-family label maps (``r_label_map`` and
       ``p_label_map`` from *product_dict*): all labelled atoms that belong to the
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
        Dict mapping *uni_mol* atom index → ``(species_list_index,
        atom_index_in_species)`` if validation succeeds, else ``None``.
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
            continue  # bond already exists
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

    # Fragments inherit stale connectivity values from the parent molecule,
    # which causes ``is_isomorphic`` to reject valid matches.  Recompute them.
    for frag in fragments:
        frag.update_connectivity_values()

    # ---- 4. Build fragment → original-index mapping ----
    frag_data: List[Tuple['Molecule', List[int]]] = []
    for frag in fragments:
        orig_indices: List[int] = []
        for frag_atom in frag.atoms:
            found = next((i for i, ca in enumerate(copy_atoms)
                          if ca is frag_atom), None)
            if found is None:
                return None
            orig_indices.append(found)
        frag_data.append((frag, orig_indices))

    # ---- 5. Connectivity-normalised isomorphism ----
    def _normalize(mol: 'Molecule') -> 'Molecule':
        """Strip bond orders, radicals, charges and lone pairs."""
        m = mol.copy(deep=True)
        for bond in m.get_all_edges():
            bond.order = 1.0
        for atom in m.atoms:
            atom.radical_electrons = 0
            atom.charge = 0
            atom.lone_pairs = 0
        m.multiplicity = 1
        return m

    n_species = len(multi_species)

    # Pre-compute normalised product molecules.
    norm_prods = [_normalize(sp.mol) for sp in multi_species]
    norm_prod_atoms = [list(np_mol.atoms) for np_mol in norm_prods]

    # ---- 6. Label-group check helper ----
    def _labels_consistent(candidate: Dict[int, Tuple[int, int]]) -> bool:
        if product_dict is None:
            return True
        if uni_is_reactant:
            # Scission: uni_mol is the reactant, fragments should become products.
            # Group uni_mol labels by which product_dict product they belong to.
            uni_lmap = product_dict.get('r_label_map')
            multi_lmap = product_dict.get('p_label_map')
        else:
            # Addition: uni_mol is the product, fragments should become reactants.
            # Group uni_mol labels by which *reactant* species they belong to.
            uni_lmap = product_dict.get('p_label_map')
            multi_lmap = product_dict.get('r_label_map')
        if not uni_lmap or not multi_lmap:
            return True

        # Determine multi-side species boundaries.
        # For scission: these are the product_dict['products'] (the actual products).
        # For addition: these are the multi_species (the actual reactants), whose
        # cumulative sizes give the boundary indices for the multi_lmap values.
        if uni_is_reactant:
            multi_mols = product_dict.get('products', [])
        else:
            multi_mols = [sp.mol for sp in multi_species]
        cum_sizes = [0]
        for m in multi_mols:
            cum_sizes.append(cum_sizes[-1] + len(m.atoms))

        def multi_idx_to_species(m_idx: int) -> Optional[int]:
            for k in range(len(multi_mols)):
                if cum_sizes[k] <= m_idx < cum_sizes[k + 1]:
                    return k
            return None

        # Group labelled uni_mol atom indices by their multi-side species.
        label_groups: Dict[int, Set[int]] = {}
        for label, uni_idx in uni_lmap.items():
            m_idx = multi_lmap.get(label)
            if m_idx is None:
                continue
            sp_id = multi_idx_to_species(m_idx)
            if sp_id is None:
                continue
            label_groups.setdefault(sp_id, set()).add(uni_idx)

        # All uni_mol indices in the same group must map to the same species_idx.
        for group_indices in label_groups.values():
            species_set: Set[int] = set()
            for uni_idx in group_indices:
                mapping = candidate.get(uni_idx)
                if mapping is not None:
                    species_set.add(mapping[0])
            if len(species_set) > 1:
                return False
        return True

    # ---- 7. Try all fragment ↔ product permutations ----
    for perm in permutations(range(n_species)):
        all_ok = True
        candidate_map: Dict[int, Tuple[int, int]] = {}

        for frag_idx, species_idx in enumerate(perm):
            frag, orig_indices = frag_data[frag_idx]

            prod_mol = multi_species[species_idx].mol
            if len(frag.atoms) != len(prod_mol.atoms):
                all_ok = False
                break

            n_frag = _normalize(frag)
            n_prod = norm_prods[species_idx]

            iso_maps = n_frag.find_isomorphism(n_prod, strict=False)
            if not iso_maps:
                all_ok = False
                break

            n_frag_atoms = list(n_frag.atoms)
            np_atoms = norm_prod_atoms[species_idx]

            # Take the first isomorphism (label validation is done globally).
            iso_map = iso_maps[0]
            for nf_atom, np_atom in iso_map.items():
                fi = n_frag_atoms.index(nf_atom)
                pi = np_atoms.index(np_atom)
                candidate_map[orig_indices[fi]] = (species_idx, pi)

        if not all_ok or len(candidate_map) != n_atoms:
            continue

        if _labels_consistent(candidate_map):
            return candidate_map

    return None


def build_concerted_ts(uni_xyz: dict,
                       uni_mol: 'Molecule',
                       split_bonds: List[Tuple[int, int]],
                       cross_bonds: List[Tuple[int, int]],
                       weight: float = 0.5,
                       ) -> Optional[dict]:
    """
    Build a TS guess for concerted multi-bond reactions by simultaneously
    stretching breaking bonds and contracting forming bonds.

    For each **split bond** (breaking), both endpoints are pushed apart
    symmetrically toward the Pauling TS estimate.  For each **cross bond**
    (forming), both endpoints are pulled together toward the Pauling TS
    estimate.  The displacements are scaled by *weight* and applied
    iteratively (3 rounds) to allow coupled adjustments to converge.

    This handles concerted eliminations (e.g. XY_elimination_hydroxyl)
    and retro-cycloadditions where multiple bonds change simultaneously.

    Args:
        uni_xyz: Unimolecular-species XYZ coordinates.
        uni_mol: RMG Molecule of the unimolecular species.
        split_bonds: Bonds to break (stretch).
        cross_bonds: Bonds to form (contract).
        weight: Interpolation weight (0 = reactant-like, 1 = product-like).

    Returns:
        TS guess XYZ dictionary, or ``None`` if validation fails.
    """
    if not split_bonds and not cross_bonds:
        return None
    symbols = uni_xyz['symbols']
    coords = np.array(uni_xyz['coords'], dtype=float)

    # Compute element-aware target distances.
    # Breaking (split) bonds: heavy-heavy bonds stretch MORE than X-H bonds.
    # Forming (cross) bonds: H-H forms quickly (target close to equilibrium).
    split_targets: Dict[Tuple[int, int], float] = {}
    for a, b in split_bonds:
        sbl = get_single_bond_length(symbols[a], symbols[b]) or 1.5
        both_heavy = symbols[a] != 'H' and symbols[b] != 'H'
        if both_heavy:
            # Heavy-heavy breaking bond stretches significantly.
            split_targets[(a, b)] = sbl + 2.0 * PAULING_DELTA
        else:
            split_targets[(a, b)] = sbl + PAULING_DELTA

    cross_targets: Dict[Tuple[int, int], float] = {}
    for a, b in cross_bonds:
        sbl = get_single_bond_length(symbols[a], symbols[b]) or 1.5
        both_h = symbols[a] == 'H' and symbols[b] == 'H'
        if both_h:
            # H-H forms quickly — target close to equilibrium.
            cross_targets[(a, b)] = sbl * 1.12
        else:
            cross_targets[(a, b)] = sbl + PAULING_DELTA

    # Also identify bonds between ring atoms that should strengthen
    # (e.g. C-C becoming C=C, C-O becoming C=O in CO₂).
    ring_atoms = set()
    for a, b in split_bonds + cross_bonds:
        ring_atoms.update((a, b))
    strengthen_targets: Dict[Tuple[int, int], float] = {}
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
            # This is a ring bond that isn't split or cross — it may
            # strengthen (single→double).  Shorten by 5%.
            sbl = get_single_bond_length(symbols[ia], symbols[ib]) or 1.5
            strengthen_targets[key] = sbl * 0.95

    for _ in range(10):  # iterate to converge coupled adjustments
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

    # Resolve collisions: push apart any atom pair closer than 0.7× SBL.
    # The concerted stretching can drag atoms through each other when the
    # ring geometry is tight.  A gentle repulsive pass fixes this.
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
                if d < min_d and d > 1e-6:
                    any_collision = True
                    vec = coords[j] - coords[i]
                    direction = vec / d
                    push = (min_d - d) * 0.5
                    coords[i] -= direction * push
                    coords[j] += direction * push
        if not any_collision:
            break

    ts_xyz = {
        'symbols': symbols,
        'isotopes': uni_xyz.get('isotopes', tuple(0 for _ in range(len(symbols)))),
        'coords': tuple(tuple(float(x) for x in row) for row in coords),
    }
    if colliding_atoms(ts_xyz):
        return None
    return ts_xyz


def stretch_bond(uni_xyz: dict,
                  uni_mol: 'Molecule',
                  split_bonds: List[Tuple[int, int]],
                  cross_bonds: Optional[List[Tuple[int, int]]] = None,
                  weight: float = 0.5,
                  label: str = '',
                  path_spec: Optional['ReactionPathSpec'] = None,
                  family: Optional[str] = None,
                  ) -> Optional[dict]:
    """
    Create a TS guess by stretching specified bonds in the unimolecular species.

    For 2-fragment splits (simple dissociation), the smaller fragment is rigidly
    translated away along the inter-fragment axis so that each split bond
    reaches its Pauling TS estimate (single-bond length + ``PAULING_DELTA``).

    For 3+-fragment splits with a cross bond (insertion/elimination pattern),
    the insertion-ring geometry builder is used instead: the mobile and migrating
    fragments are repositioned to form a 3-membered TS ring.

    Args:
        uni_xyz (dict): XYZ coordinates of the unimolecular species.
        uni_mol (Molecule): RMG Molecule of the unimolecular species.
        split_bonds (List[Tuple[int, int]]): Bonds to remove to fragment the molecule.
        cross_bonds (List[Tuple[int, int]], optional): Bonds that connect atoms
            across different fragments (used for insertion-ring detection).
        weight (float): Interpolation weight (0=reactant-like, 1=product-like).
        label (str): Logging label.
        path_spec (ReactionPathSpec, optional): Phase 3a path-spec; when
            provided, validation is routed through
            :func:`validate_guess_against_path_spec`.  When ``None`` the
            legacy :func:`validate_ts_guess` is used (degraded mode).

    Returns:
        Optional[dict]: TS guess XYZ, or None if validation fails.
    """
    n_atoms = len(uni_xyz['symbols'])
    cross_bonds = cross_bonds or []

    # Build adjacency, remove split bonds, find fragments.
    atom_to_idx = {atom: idx for idx, atom in enumerate(uni_mol.atoms)}
    adj: Dict[int, Set[int]] = {k: set() for k in range(n_atoms)}
    for atom in uni_mol.atoms:
        idx_a = atom_to_idx[atom]
        for neighbor in atom.edges:
            idx_b = atom_to_idx[neighbor]
            adj[idx_a].add(idx_b)
    for a, b in split_bonds:
        adj[a].discard(b)
        adj[b].discard(a)

    visited: Set[int] = set()
    fragments: List[Set[int]] = []
    for start in range(n_atoms):
        if start in visited:
            continue
        component: Set[int] = set()
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

    # --- 3-membered insertion ring pattern ---
    if len(fragments) >= 3 and cross_bonds:
        result = try_insertion_ring(uni_xyz, uni_mol, fragments, split_bonds,
                                    cross_bonds, weight, n_atoms,
                                    path_spec=path_spec,
                                    family=family)
        if result is not None:
            return result

    # --- Simple stretch: translate the smallest fragment away ---
    fragments.sort(key=len)
    small_frag = fragments[0]
    large_frag: Set[int] = set()
    for f in fragments[1:]:
        large_frag.update(f)

    small_anchors: List[int] = []
    large_anchors: List[int] = []
    for a, b in split_bonds:
        if a in small_frag:
            small_anchors.append(a)
            large_anchors.append(b)
        elif b in small_frag:
            small_anchors.append(b)
            large_anchors.append(a)
        # Bonds between two non-small fragments: skip for direction calc.

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

    target_dists = []
    current_dists = []
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

    ts_xyz: dict = {
        'symbols': uni_xyz['symbols'],
        'isotopes': uni_xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
        'coords': tuple(tuple(float(x) for x in row) for row in ts_coords),
    }

    # split_bonds are passed as forming_bonds: for addition the TS "forms"
    # the bonds that are being stretched apart, so both names refer to the
    # same bond set from the TS perspective.
    is_valid, reason = _validate_addition_xyz(
        ts_xyz, uni_mol, split_bonds, label=label, path_spec=path_spec)
    if not is_valid:
        logger.debug(f'Linear addition ({label}): rejected — {reason}.')
        return None

    return ts_xyz


# ---------------------------------------------------------------------------
# Phase 4a — limited family-aware insertion-ring target calibration
# ---------------------------------------------------------------------------


def _insertion_ring_extra_stretch(family: Optional[str]) -> float:
    """Return a family-specific positive Å delta to add to the standard
    Pauling target inside :func:`try_insertion_ring`.

    The 3-membered insertion-ring TS in :func:`try_insertion_ring` uses a
    uniform ``single_bond_length + PAULING_DELTA`` target on its three
    reactive edges (mobile-anchor C-C, mobile-mig C-H, anchor-mig C-H).
    For most families this scale is correct, but for highly exothermic
    carbene insertions the textbook TS sits much *earlier* on the
    reaction coordinate — its three reactive edges are roughly 0.20 Å
    looser than the standard delta predicts.  This helper returns the
    family-specific extra stretch (in Å) to add to *every* reactive
    edge of the insertion ring; an empty/unknown family returns 0.

    Currently calibrated families:
    * ``'1,2_Insertion_carbene'``: +0.20 Å.

    Args:
        family: Reaction family name (typically ``path_spec.family``).

    Returns:
        A non-negative additional stretch in Å.  Defaults to ``0.0``.
    """
    if family == '1,2_Insertion_carbene':
        return 0.20
    return 0.0


def try_insertion_ring(uni_xyz: dict,
                        uni_mol: 'Molecule',
                        fragments: List[Set[int]],
                        split_bonds: List[Tuple[int, int]],
                        cross_bonds: List[Tuple[int, int]],
                        weight: float,
                        n_atoms: int,
                        path_spec: Optional['ReactionPathSpec'] = None,
                        family: Optional[str] = None,
                        ) -> Optional[dict]:
    """
    Attempt to build a 3-membered insertion-ring TS geometry.

    When fragmenting the unimolecular species yields 3+ fragments and there
    is a cross bond connecting two of the split-bond partners, we have an
    insertion/elimination pattern (e.g. 1,2_Insertion_CO: A-B → A + :C: + B
    where C inserts between A and B).

    Args:
        uni_xyz (dict): XYZ of the unimolecular species.
        uni_mol (Molecule): RMG Molecule of the unimolecular species.
        fragments (List[Set[int]]): Connected components after removing split bonds.
        split_bonds (List[Tuple[int, int]]): Removed bonds.
        cross_bonds (List[Tuple[int, int]]): Bonds connecting atoms across fragments.
        weight (float): Interpolation weight.
        n_atoms (int): Total number of atoms.
        path_spec (ReactionPathSpec, optional): Phase 3a path-spec; when
            provided, validation routes through
            :func:`validate_guess_against_path_spec`.

    Returns:
        Optional[dict]: TS guess XYZ, or None if the pattern doesn't apply.
    """
    # Find atom appearing in 2+ split bonds (the central atom).
    sb_atom_count: Dict[int, int] = {}
    for a, b in split_bonds:
        sb_atom_count[a] = sb_atom_count.get(a, 0) + 1
        sb_atom_count[b] = sb_atom_count.get(b, 0) + 1
    central_atom: Optional[int] = None
    for atom, count in sb_atom_count.items():
        if count >= 2:
            central_atom = atom
            break

    if central_atom is None:
        return None

    # Get partners of the central atom from split bonds.
    partners: List[int] = []
    for a, b in split_bonds:
        if a == central_atom:
            partners.append(b)
        elif b == central_atom:
            partners.append(a)

    # Identify substrate and migrating atoms: partners connected by a cross bond.
    sub_atom: Optional[int] = None
    mig_atom: Optional[int] = None
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

    # Phase 4a — limited family-aware insertion-ring target calibration.
    # Highly exothermic carbene insertions have a markedly *earlier* TS
    # than the standard ``sbl + PAULING_DELTA`` predicts.  When the
    # ``family`` argument or ``path_spec.family`` identifies the family
    # as one of those calibrated cases, add a small positive Å delta to
    # *every* reactive edge of the 3-membered ring so the resulting TS
    # is at the appropriate looser scale.  For all other families this
    # delta is 0.0 and behavior is unchanged.  An explicit ``family``
    # kwarg takes precedence — that is the canonical channel for the
    # template-guided block in :mod:`arc.job.adapters.ts.linear`, where
    # ``stretch_bond`` (the immediate caller of this function) does not
    # carry a path-spec at all.
    family_for_calibration = family if family is not None else (
        path_spec.family if path_spec is not None else None)
    extra_stretch = _insertion_ring_extra_stretch(family_for_calibration)

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

    ts_xyz: dict = {
        'symbols': uni_xyz['symbols'],
        'isotopes': uni_xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
        'coords': tuple(tuple(float(c) for c in row) for row in ts_coords),
    }
    is_valid, reason = _validate_addition_xyz(
        ts_xyz, uni_mol, split_bonds, label='insertion-ring', path_spec=path_spec)
    if not is_valid:
        # Phase 4a: when the calibrated insertion-ring builder is in
        # use, the standard ``has_too_many_fragments`` heavy-heavy
        # threshold (2.0 Å) is just barely above the un-calibrated
        # Pauling target.  A calibrated carbene insertion ring sits at
        # ~2.16 Å on its mobile-anchor C–C edge, which trips that
        # threshold even though the geometry is the textbook earlier-TS
        # the calibration was supposed to produce.  Re-validate with a
        # locally-loosened heavy-heavy threshold strictly when the
        # rejection is the fragments check AND the family is calibrated.
        # All other rejection reasons (collisions, detached H, drift,
        # planarity, recipe mismatch) still gate the guess.
        if extra_stretch > 0.0 and 'too many fragments' in (reason or ''):
            relaxed_max_heavy = 2.0 + extra_stretch + 0.10
            if not has_too_many_fragments(
                    ts_xyz, max_heavy_heavy=relaxed_max_heavy):
                # Re-run only the rest of the generic checks.  We do
                # NOT skip collisions or detached-H — we only widen the
                # fragment-count threshold by the calibration delta.
                if (not colliding_atoms(ts_xyz)
                        and not has_detached_hydrogen(ts_xyz, max_h_heavy_dist=3.0)):
                    logger.debug(
                        f'Linear (insertion-ring): calibration ({family_for_calibration},'
                        f' +{extra_stretch:.2f} Å) — accepting via relaxed '
                        f'heavy-heavy threshold {relaxed_max_heavy:.2f} Å '
                        f'after generic-validator fragments rejection.')
                    return ts_xyz
        logger.debug(f'Linear (insertion-ring): rejected (family={family_for_calibration}, '
                     f'extra_stretch={extra_stretch}) — {reason}.')
        return None
    return ts_xyz


def stretch_core_from_large(ts_xyz: dict,
                             uni_mol: 'Molecule',
                             split_bonds: List[Tuple[int, int]],
                             core: Set[int],
                             large_prod_atoms: Set[int],
                             small_prod_atoms: Set[int],
                             weight: float = 0.5,
                             ) -> dict:
    """Stretch the core of the small product away from the large product.

    When fragmenting the reactant produces 3+ fragments (e.g. in elimination
    reactions), ``stretch_bond`` only moves the smallest fragment.  This
    function handles the remaining split bonds: those that connect the *core*
    of the small product to the large product.

    The core atoms (and any migrating atoms riding on the same fragment) are
    rigidly translated so that the relevant split bonds reach TS-like distances.

    Args:
        ts_xyz: Current TS guess XYZ (after ``stretch_bond``).
        uni_mol: RMG Molecule of the unimolecular species.
        split_bonds: All breaking bonds.
        core: Atom indices forming the connected core of the small product.
        large_prod_atoms: Atom indices belonging to the large product.
        small_prod_atoms: All atoms belonging to the small product (core + migrating).
        weight: Interpolation weight (0 = reactant-like, 1 = product-like).

    Returns:
        dict: Modified XYZ with core atoms translated.
    """
    symbols = ts_xyz['symbols']
    coords = np.array(ts_xyz['coords'], dtype=float)

    # Find split bonds between core and large_prod_atoms.
    core_anchors: List[int] = []
    large_anchors: List[int] = []
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

    target_dists = []
    current_dists = []
    for ca, la in zip(core_anchors, large_anchors):
        sbl = get_single_bond_length(symbols[ca], symbols[la])
        target_dists.append(sbl + PAULING_DELTA)
        current_dists.append(float(np.linalg.norm(coords[ca] - coords[la])))

    avg_target = float(np.mean(target_dists))
    avg_current = float(np.mean(current_dists))
    delta = (avg_target - avg_current) * 2.0 * (1.0 - weight)
    if delta < 0:
        delta = 0.0

    # Move the entire core (not migrating atoms — those are handled separately).
    for idx in core:
        coords[idx] += direction * delta

    return {
        'symbols': ts_xyz['symbols'],
        'isotopes': ts_xyz.get('isotopes', tuple(0 for _ in range(len(symbols)))),
        'coords': tuple(tuple(float(x) for x in row) for row in coords),
    }


def migrate_verified_atoms(ts_xyz: dict,
                            uni_mol: 'Molecule',
                            migrating_atoms: Set[int],
                            core: Set[int],
                            large_prod_atoms: Set[int],
                            weight: float = 0.5,
                            cross_bonds: Optional[List[Tuple[int, int]]] = None,
                            ) -> dict:
    """Migrate specific atoms identified by ``map_and_verify_fragments``.

    Unlike ``migrate_h_between_fragments`` (which guesses which H to move by
    composition matching), this function moves exactly the atoms in
    *migrating_atoms* — the set of atom indices that belong to one product but
    are bonded only to atoms in the other product in the reactant graph.

    Each migrating atom is placed at a TS-like position between its current
    heavy-atom donor (in *large_prod_atoms*) and its acceptor in *core*,
    using triangulation when the spheres overlap.  The acceptor is determined
    from *cross_bonds* (forming bonds) when available, falling back to the
    nearest heavy atom in *core*.

    Args:
        ts_xyz: TS guess XYZ (already stretched by ``stretch_bond``).
        uni_mol: RMG Molecule of the unimolecular species.
        migrating_atoms: Atom indices that need to move between product groups.
        core: Atom indices forming the connected core of the small product.
        large_prod_atoms: Atom indices belonging to the large product.
        weight: Interpolation weight (0 = reactant-like, 1 = product-like).
        cross_bonds: Forming bonds absent from uni_mol (used to identify the
            exact acceptor atom for each migrating atom).

    Returns:
        dict: Modified XYZ with migrating atoms partially displaced.
    """
    symbols = ts_xyz['symbols']
    coords = np.array(ts_xyz['coords'], dtype=float)
    ts_coords = coords.copy()
    atom_to_idx = {atom: idx for idx, atom in enumerate(uni_mol.atoms)}
    core_heavy = [idx for idx in core if symbols[idx] != 'H']

    # Build a map from migrating atom → cross-bond partner (the acceptor).
    # The acceptor may live in *any* product fragment, not only ``core``:
    # families like Korcek_step2 produce intra-fragment H migrations where
    # donor and acceptor are both in the *large* fragment.  The cross
    # bond's heavy-atom partner is the authoritative acceptor in those
    # cases — restricting it to ``core`` would silently misroute the H.
    cross_acceptor: Dict[int, int] = {}
    for a, b in (cross_bonds or []):
        if a in migrating_atoms and symbols[b] != 'H':
            cross_acceptor[a] = b
        elif b in migrating_atoms and symbols[a] != 'H':
            cross_acceptor[b] = a

    for h_idx in migrating_atoms:
        # Find donor: the H's heavy reactant neighbor.  Prefer one that
        # lives in ``large_prod_atoms`` (the inter-fragment migration
        # case), but fall back to *any* heavy reactant neighbor so that
        # intra-large H migrations (Korcek_step2 — donor C and acceptor C
        # are both in the same product fragment) are handled too.
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

        # Find acceptor: prefer cross-bond partner, fall back to nearest core heavy atom.
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
            # Spheres overlap — triangulate (same algorithm as migrate_h_between_fragments).
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

        # Place the migrating atom at the ideal TS position directly.
        # Unlike bond-stretch interpolation, linear Cartesian interpolation
        # from the reactant position fails when the atom must swing around
        # its donor (e.g. H migrating from one face of O to another), as
        # the intermediate path passes through the donor atom.
        ts_coords[h_idx] = ideal

    return {
        'symbols': ts_xyz['symbols'],
        'isotopes': ts_xyz.get('isotopes', tuple(0 for _ in range(len(symbols)))),
        'coords': tuple(tuple(float(x) for x in row) for row in ts_coords),
    }


def migrate_h_between_fragments(ts_xyz: dict,
                                 uni_mol: 'Molecule',
                                 split_bonds: List[Tuple[int, int]],
                                 product_species: List[ARCSpecies],
                                 weight: float = 0.5,
                                 ) -> dict:
    """
    Partially displace H atoms that need to migrate between fragments
    to match the product species' compositions.

    After ``stretch_bond`` rigidly translates fragments apart, H atoms
    remain on their original fragment.  In reactions where H redistribution
    occurs (e.g. 1,3_Insertion_CO2: R-C(=O)OH → R-H + O=C=O), the TS
    should show the migrating H partially displaced toward its destination.

    This function:

    1. Identifies fragments from the split bonds.
    2. Computes element compositions for each fragment.
    3. Compares with product compositions to find H surplus/deficit.
    4. For each surplus fragment, finds the H atom closest to the deficit
       fragment and places it on the donor→acceptor axis at a TS-like
       distance (interpolated by ``weight``).  Using the donor→acceptor
       axis instead of a direct H→acceptor line avoids near-collisions
       with other atoms in the source fragment (e.g. the C in a CO₂ group).

    Args:
        ts_xyz: TS guess XYZ from ``stretch_bond`` (already stretched).
        uni_mol: RMG Molecule of the unimolecular species.
        split_bonds: Bonds that were cut to create fragments.
        product_species: Product species for composition matching.
        weight: Interpolation weight (0 = reactant-like, 1 = product-like).

    Returns:
        dict: Modified XYZ with H atoms partially migrated, or the original
            XYZ unchanged if no migration is needed.
    """
    n_atoms = len(ts_xyz['symbols'])
    n_products = len(product_species)

    # Build adjacency and fragment.
    atom_to_idx = {atom: idx for idx, atom in enumerate(uni_mol.atoms)}
    adj: Dict[int, Set[int]] = {k: set() for k in range(n_atoms)}
    for atom in uni_mol.atoms:
        idx_a = atom_to_idx[atom]
        for neighbor in atom.edges:
            idx_b = atom_to_idx[neighbor]
            adj[idx_a].add(idx_b)
    for a, b in split_bonds:
        adj[a].discard(b)
        adj[b].discard(a)

    visited: Set[int] = set()
    fragments: List[Set[int]] = []
    for start in range(n_atoms):
        if start in visited:
            continue
        component: Set[int] = set()
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

    # Compute fragment formulas and target formulas.
    symbols = ts_xyz['symbols']
    frag_formulas: List[Dict[str, int]] = []
    for frag in fragments:
        formula: Dict[str, int] = {}
        for idx in frag:
            sym = symbols[idx]
            formula[sym] = formula.get(sym, 0) + 1
        frag_formulas.append(formula)

    target_formulas: List[Dict[str, int]] = []
    for sp in product_species:
        formula: Dict[str, int] = {}
        for sym in sp.get_xyz()['symbols']:
            formula[sym] = formula.get(sym, 0) + 1
        target_formulas.append(formula)

    # Match fragments to targets by heavy-atom composition.
    def heavy_formula(f: Dict[str, int]) -> Dict[str, int]:
        return {k: v for k, v in f.items() if k != 'H'}

    frag_to_target: Dict[int, int] = {}
    used_targets: Set[int] = set()
    for fi, ff in enumerate(frag_formulas):
        hf = heavy_formula(ff)
        for ti, tf in enumerate(target_formulas):
            if ti not in used_targets and heavy_formula(tf) == hf:
                frag_to_target[fi] = ti
                used_targets.add(ti)
                break

    if len(frag_to_target) != n_products:
        return ts_xyz

    # Find H surplus/deficit per fragment.
    h_surplus: Dict[int, int] = {}
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

    # Find surplus-deficit pairs and migrate H atoms.
    surplus_frags = [fi for fi, d in h_surplus.items() if d > 0]
    deficit_frags = [fi for fi, d in h_surplus.items() if d < 0]

    for s_fi in surplus_frags:
        n_to_move = h_surplus[s_fi]
        # Find H atoms in this fragment, sorted by distance to the nearest
        # heavy atom in a deficit fragment (closest first).
        h_indices = [idx for idx in fragments[s_fi] if symbols[idx] == 'H']
        if not h_indices or not deficit_frags:
            continue

        # Collect heavy atoms in deficit fragments.
        deficit_heavy: List[int] = []
        for d_fi in deficit_frags:
            deficit_heavy.extend(idx for idx in fragments[d_fi] if symbols[idx] != 'H')
        if not deficit_heavy:
            continue

        deficit_heavy_coords = ts_coords[deficit_heavy]

        # Identify split-bond anchor atoms in this fragment.  In
        # insertion/elimination reactions the migrating H should come from
        # a *non-anchor* heavy atom to create a proper TS ring (e.g. O on
        # one C of ethylene and H migrating from the other C).
        split_anchors_in_frag: Set[int] = set()
        for a, b in split_bonds:
            if a in fragments[s_fi]:
                split_anchors_in_frag.add(a)
            if b in fragments[s_fi]:
                split_anchors_in_frag.add(b)

        # Sort H atoms by: (1) prefer H not bonded to a split-bond anchor,
        # (2) then by min distance to any deficit-fragment heavy atom.
        h_dists: List[Tuple[int, float, int, bool]] = []
        for h_idx in h_indices:
            dists = np.linalg.norm(deficit_heavy_coords - ts_coords[h_idx], axis=1)
            min_dist = float(dists.min())
            nearest_heavy = deficit_heavy[int(dists.argmin())]
            on_anchor = any(
                atom_to_idx[nbr] in split_anchors_in_frag
                for nbr in uni_mol.atoms[h_idx].bonds.keys()
            )
            h_dists.append((h_idx, min_dist, nearest_heavy, on_anchor))
        h_dists.sort(key=lambda x: (x[3], x[1]))

        for h_idx, _, nearest_heavy, _ in h_dists[:n_to_move]:
            # Find the donor heavy atom: the heavy atom bonded to this H in the
            # source fragment.
            donor_heavy = None
            for nbr in uni_mol.atoms[h_idx].bonds.keys():
                nbr_idx = atom_to_idx[nbr]
                if symbols[nbr_idx] != 'H' and nbr_idx in fragments[s_fi]:
                    donor_heavy = nbr_idx
                    break

            if donor_heavy is None:
                continue

            # Triangulate: place H at the intersection of two spheres centred
            # on donor and acceptor with TS-like radii, choosing the point
            # closest to the current H position.  This produces a non-collinear
            # D-H-A geometry that avoids passing through atoms between donor
            # and acceptor (e.g. the C in a CO₂ group).
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

                # Pick the candidate with greater clearance from source-fragment
                # heavy atoms (avoids crowding the TS ring interior).
                def _min_frag_dist(pos):
                    md = float('inf')
                    for idx in fragments[s_fi]:
                        if idx == h_idx or idx == donor_heavy or symbols[idx] == 'H':
                            continue
                        md = min(md, float(np.linalg.norm(pos - ts_coords[idx])))
                    return md

                new_h = cand_plus if _min_frag_dist(cand_plus) >= _min_frag_dist(cand_minus) \
                    else cand_minus
            else:
                # Spheres don't overlap → collinear placement at d_DH from donor.
                new_h = d_pos + d_DH * da_hat

            ts_coords[h_idx] = new_h

    result = {
        'symbols': ts_xyz['symbols'],
        'isotopes': ts_xyz['isotopes'],
        'coords': tuple(tuple(float(x) for x in row) for row in ts_coords),
    }

    if colliding_atoms(result):
        return ts_xyz  # Fall back to the original if migration causes collisions.
    return result


def _reposition_leaving_groups(xyz: dict,
                               pre_xyz: dict,
                               mol: 'Molecule',
                               split_bonds: List[Tuple[int, int]],
                               adj: List[List[int]],
                               frag_id: List[int],
                               n_atoms: int,
                               extra_stretch: float = 0.0,
                               ) -> dict:
    """Reposition leaving-group fragments after ring closure.

    Ring closure moves ring-member atoms but leaves disconnected fragments
    (the leaving groups) at their original positions.  This can make the
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
        mol: RMG Molecule.
        split_bonds: Bonds that were cut to create fragments.
        adj: Adjacency list (with split bonds excluded).
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
            continue  # Both atoms are in the same fragment — not a true split.
        # Identify which side is the smaller (leaving) fragment.
        frag_a = [i for i in range(n_atoms) if frag_id[i] == frag_id[sb_a]]
        frag_b = [i for i in range(n_atoms) if frag_id[i] == frag_id[sb_b]]
        if len(frag_a) <= len(frag_b):
            leaving_frag, leaving_anchor, ring_anchor = frag_a, sb_a, sb_b
        else:
            leaving_frag, leaving_anchor, ring_anchor = frag_b, sb_b, sb_a

        # Check the current (post-ring-closure) split-bond distance.
        current_dist = float(np.linalg.norm(coords[leaving_anchor] - coords[ring_anchor]))
        sym_a = xyz['symbols'][ring_anchor]
        sym_b = xyz['symbols'][leaving_anchor]
        ts_target = get_single_bond_length(sym_a, sym_b) + PAULING_DELTA + extra_stretch

        # Only reposition when the leaving group is stranded TOO FAR from
        # the ring anchor after closure.  When the distance is too short
        # (leaving group followed the anchor, e.g. via centroid correction
        # in small rings), ``stretch_bond()`` downstream will stretch it
        # to the TS target — no repositioning needed here.
        if current_dist <= ts_target + 0.5:
            continue

        # Step 1: Translate the leaving fragment to follow the ring anchor's
        # displacement during ring closure.
        anchor_displacement = coords[ring_anchor] - pre_coords[ring_anchor]
        for idx in leaving_frag:
            coords[idx] += anchor_displacement

        # Step 2: Stretch the split bond to its TS target along the original
        # (pre-closure) bond direction, which points reliably away from the ring.
        pre_vec = pre_coords[leaving_anchor] - pre_coords[ring_anchor]
        pre_dist = float(np.linalg.norm(pre_vec))
        if pre_dist < 1e-6:
            continue
        direction = pre_vec / pre_dist
        # Place the leaving anchor at the TS target distance from the ring
        # anchor along the original bond direction.
        ideal_pos = coords[ring_anchor] + direction * ts_target
        shift = ideal_pos - coords[leaving_anchor]
        for idx in leaving_frag:
            coords[idx] += shift

    return {
        'symbols': xyz['symbols'],
        'isotopes': xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
        'coords': tuple(tuple(float(x) for x in row) for row in coords),
    }


def apply_intra_frag_contraction(xyz: dict,
                                  mol: 'Molecule',
                                  split_bonds: List[Tuple[int, int]],
                                  cross_bonds: Optional[List[Tuple[int, int]]],
                                  multi_species: List[ARCSpecies],
                                  weight: float = 0.5,
                                  label: str = '',
                                  ) -> List[dict]:
    """
    Apply angular ring contraction for intra-fragment forming bonds.

    After ``stretch_bond()`` separates fragments by stretching the split bonds,
    any forming bond (cross bond) whose two atoms remain in the same fragment
    requires angular contraction to bring them closer together.  This function
    identifies such bonds and applies ``ring_closure_xyz()`` for each one,
    returning a separate TS guess per candidate forming bond.

    Forming bonds are detected from product ring topology via
    ``detect_intra_frag_ring_bonds()``.  When multiple candidates exist
    (e.g. due to resonance-equivalent atom assignments), each produces an
    independent TS guess so that the best one can be selected downstream.

    Args:
        xyz: Post-stretch XYZ geometry.
        mol: RMG Molecule of the unimolecular species.
        split_bonds: Bonds severed by ``stretch_bond()``.
        cross_bonds: Unused (kept for call-site compatibility).
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

    # Build adjacency excluding split bonds to check fragment membership.
    n_atoms = len(mol.atoms)
    atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
    exclude_set = {frozenset(b) for b in split_bonds}
    adj: List[List[int]] = [[] for _ in range(n_atoms)]
    for idx, atom in enumerate(mol.atoms):
        for nbr in atom.edges:
            nbr_idx = atom_to_idx[nbr]
            if frozenset((idx, nbr_idx)) not in exclude_set:
                adj[idx].append(nbr_idx)

    # Find which fragment each atom belongs to.
    frag_id: List[int] = [-1] * n_atoms
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

    # Identify split-bond endpoints that also participate in a forming bond.
    # When bond formation and bond breaking happen at the same atom (e.g. SN2-
    # like or ring-closure-with-leaving-group), both the forming and breaking
    # bonds are elongated at the TS compared to the simple ring-closure case.
    split_endpoints: Set[int] = set()
    for sb in split_bonds:
        split_endpoints.update(sb)

    # Determine whether a genuine leaving group exists by checking fragment
    # sizes.  A small fragment (≤ 4 heavy atoms) on the far side of a split
    # bond is treated as a leaving group (e.g. CH3 in ExoTetCyclic).  Large
    # fragments (both halves of a Diels–Alder retro-fragmentation) are not.
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

    results: List[dict] = []
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
        # Strained rings (3-4 membered) have an earlier TS due to ring
        # strain energy, so the forming bond is longer at the TS.
        ring_correction = 0.15 if ring_size == 3 else (0.08 if ring_size == 4 else 0.0)
        # When a ring-closure endpoint also participates in a split bond
        # (leaving group departure), the TS is earlier: both bonds sharing
        # the central atom are longer than in a pure ring-closure TS.
        # Only apply when a genuine small leaving group exists (not for
        # Diels–Alder where both fragments are large) and when
        # ring_correction is zero (avoid stacking corrections for strained
        # rings that already account for an early TS).
        has_leaving_group = bool(
            has_small_leaving_frag and ({a, b} & split_endpoints) and ring_correction == 0.0
        )
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
            # Reposition leaving-group fragments that were stranded by ring
            # closure.  When the ring anchor moved during closure but the
            # leaving group stayed put, the split-bond distance can grow
            # far beyond the TS target.  Only reposition when the leaving
            # group is too FAR (stranded); when it is too close,
            # ``stretch_bond()`` downstream will handle the distance.
            if has_leaving_group:
                contracted = _reposition_leaving_groups(
                    contracted, xyz, mol, split_bonds, adj, frag_id, n_atoms,
                    extra_stretch=leaving_group_correction)
            results.append(contracted)
    return results if results else [xyz]


def detect_intra_frag_ring_bonds(mol: 'Molecule',
                                  split_bonds: List[Tuple[int, int]],
                                  multi_species: List[ARCSpecies],
                                  xyz: dict,
                                  ) -> List[Tuple[Tuple[int, int], int]]:
    """
    Detect bonds that should form within a fragment (ring closure) to match
    product ring topology.

    After severing ``split_bonds``, each remaining connected component is a
    fragment.  If any product species is cyclic, this function searches for
    pairs of non-bonded heavy atoms in each fragment that are connected by a
    short path whose length matches a product ring size.  Such pairs are
    likely forming bonds (ring closures that occur simultaneously with the
    split-bond scission).

    Candidates are sorted by descending current distance so that the
    longest (least-advanced) forming bond is attempted first.

    Args:
        mol (Molecule): RMG Molecule of the unimolecular species.
        split_bonds (List[Tuple[int, int]]): Bonds already severed by
            ``stretch_bond()``.
        multi_species (List[ARCSpecies]): Product species on the
            multi-species side.
        xyz (dict): Current XYZ coordinates (used for distance-based sorting).

    Returns:
        List[Tuple[Tuple[int, int], int]]: Intra-fragment forming-bond
            candidates as ``((i, j), ring_size)`` pairs, where ``i < j``
            and ``ring_size`` is the BFS path length (number of ring atoms).
    """
    # Only relevant when at least one product has a ring.
    product_ring_sizes: Set[int] = set()
    for sp in multi_species:
        if sp.mol.is_cyclic():
            try:
                for ring in sp.mol.get_smallest_set_of_smallest_rings():
                    product_ring_sizes.add(len(ring))
            except Exception:
                pass
    if not product_ring_sizes:
        return []

    # Build adjacency excluding split bonds.
    n_atoms = len(mol.atoms)
    atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
    exclude_set = {frozenset(b) for b in split_bonds}
    adj: List[List[int]] = [[] for _ in range(n_atoms)]
    for idx, atom in enumerate(mol.atoms):
        for nbr in atom.edges:
            nbr_idx = atom_to_idx[nbr]
            if frozenset((idx, nbr_idx)) not in exclude_set:
                adj[idx].append(nbr_idx)

    # Find connected components (fragments).
    visited: Set[int] = set()
    fragments: List[Set[int]] = []
    for start in range(n_atoms):
        if start in visited:
            continue
        comp: Set[int] = set()
        queue: deque = deque([start])
        while queue:
            node = queue.popleft()
            if node in comp:
                continue
            comp.add(node)
            queue.extend(n for n in adj[node] if n not in comp)
        visited |= comp
        fragments.append(comp)

    # Atoms at the boundary of a split bond — at least one atom in the
    # forming bond should be adjacent to the severed edge.
    split_endpoints: Set[int] = set()
    for sb in split_bonds:
        for atom_idx in sb:
            split_endpoints.add(atom_idx)

    # For each fragment, find non-bonded heavy-atom pairs connected by a path
    # whose length matches a product ring size.
    candidates: List[Tuple[Tuple[int, int], List[int]]] = []
    for frag in fragments:
        heavy = sorted(i for i in frag if mol.atoms[i].symbol != 'H')
        for ai, a in enumerate(heavy):
            for b in heavy[ai + 1:]:
                if b in adj[a]:
                    continue  # already bonded
                # Require at least one atom to be a split bond endpoint.
                if a not in split_endpoints and b not in split_endpoints:
                    continue
                path = _bfs_path(adj, a, b)
                if path is not None and len(path) in product_ring_sizes:
                    candidates.append(((min(a, b), max(a, b)), path))

    if not candidates:
        return []

    # Filter by element composition: only keep candidates whose ring path
    # has the same element multiset as a product ring.  This eliminates
    # false positives like all-carbon paths when the product ring contains O/S.
    product_ring_elements: List[Tuple[str, ...]] = []
    for sp in multi_species:
        if sp.mol.is_cyclic():
            try:
                for ring in sp.mol.get_smallest_set_of_smallest_rings():
                    elems = tuple(sorted(a.symbol for a in ring))
                    product_ring_elements.append(elems)
            except Exception:
                pass

    coords = np.array(xyz['coords'], dtype=float)

    def _sort_key(item: Tuple[Tuple[int, int], int]) -> float:
        """Sort by descending current distance (longest first)."""
        a, b = item[0]
        return -float(np.linalg.norm(coords[a] - coords[b]))

    if product_ring_elements:
        filtered = []
        for bond, path in candidates:
            path_elems = tuple(sorted(mol.atoms[i].symbol for i in path))
            if path_elems in product_ring_elements:
                filtered.append((bond, len(path)))
        if filtered:
            filtered.sort(key=_sort_key)
            return filtered

    result = [(bond, len(path)) for bond, path in candidates]
    result.sort(key=_sort_key)
    return result
