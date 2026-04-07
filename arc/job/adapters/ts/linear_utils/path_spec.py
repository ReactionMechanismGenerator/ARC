"""
Path-local data model and validation plumbing for the linear TS-guess adapter.

This module is the Phase 1 foundation layer.  It defines:

* :class:`ReactionPathSpec` — a deterministic, path-local description of one
  reaction channel: which bonds break, which form, which change order, which
  unchanged near-core bonds must be preserved, plus per-bond reactant- and
  product-side reference distances and bond orders.

* :func:`get_ts_target_distance` — a role-aware target distance helper used
  by downstream geometry stages.  Roles are ``breaking``, ``forming``,
  ``changed``, and ``unchanged_near_core``.

* :func:`has_recipe_channel_mismatch` — a precise, conservative recipe-fidelity
  check used by the orchestration validation wrapper.

* :func:`validate_guess_against_path_spec` — the single orchestration-level
  validation entry point.  It delegates to :func:`validate_ts_guess` for the
  generic geometry/family checks and adds the recipe-channel mismatch guard.

The module is intentionally small and explicit.  It does NOT add new chemistry
heuristics or change generation policies — it only formalizes path-local
metadata and centralizes orchestration validation.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from arc.common import get_single_bond_length
from arc.job.adapters.ts.linear_utils.postprocess import (
    PAULING_DELTA,
    validate_ts_guess,
)


HETERO_HEAVY_ATOMS: Set[str] = {'N', 'O', 'S', 'P'}

if TYPE_CHECKING:
    from rmgpy.molecule.molecule import Molecule


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CanonicalBond = Tuple[int, int]


def _canon(i: int, j: int) -> CanonicalBond:
    """Return a canonical (min, max) bond key."""
    return (i, j) if i <= j else (j, i)


def _canon_list(bonds) -> List[CanonicalBond]:
    """Canonicalize and deterministically sort a list of bonds."""
    return sorted({_canon(int(a), int(b)) for a, b in bonds or []})


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _build_adjacency(mol: 'Molecule') -> Dict[int, Set[int]]:
    """Build a {atom_index -> {neighbor_index, ...}} adjacency dict from a molecule."""
    atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
    adj: Dict[int, Set[int]] = {i: set() for i in range(len(mol.atoms))}
    for atom in mol.atoms:
        ia = atom_to_idx[atom]
        for nbr in atom.bonds:
            adj[ia].add(atom_to_idx[nbr])
    return adj


def _bond_order_map(mol: 'Molecule') -> Dict[CanonicalBond, float]:
    """Return a {(min,max) -> bond_order} dict for every bond in *mol*."""
    atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
    out: Dict[CanonicalBond, float] = {}
    for atom in mol.atoms:
        ia = atom_to_idx[atom]
        for nbr, bond in atom.bonds.items():
            ib = atom_to_idx[nbr]
            key = _canon(ia, ib)
            if key not in out:
                out[key] = float(bond.order)
    return out


def _all_bonds(mol: 'Molecule') -> List[CanonicalBond]:
    """Return all bonds in *mol* as a sorted list of canonical (min,max) tuples."""
    return sorted(_bond_order_map(mol).keys())


def _multi_source_bfs(adj: Dict[int, Set[int]], sources: Set[int]) -> Dict[int, int]:
    """Return graph distances from the multi-source set *sources* to every atom.

    Atoms unreachable from *sources* are absent from the returned dict.
    """
    dist: Dict[int, int] = {s: 0 for s in sources}
    q: deque = deque(sources)
    while q:
        u = q.popleft()
        d_next = dist[u] + 1
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = d_next
                q.append(v)
    return dist


# ---------------------------------------------------------------------------
# Changed-bond extraction (exact algorithm)
# ---------------------------------------------------------------------------

def _compute_changed_bonds(
    r_mol: 'Molecule',
    p_mol: 'Molecule',
    atom_map: Optional[List[int]] = None,
) -> List[CanonicalBond]:
    """Return bonds present on both sides whose order changes by more than 1e-6.

    Preferred form: when ``r_mol`` and ``p_mol`` are in the same atom ordering
    (i.e. ``p_mol`` is the path-local mapped product), iterate over reactant
    bonds and check whether the same (i, j) pair carries a bond in ``p_mol``;
    if both have a bond, compare bond orders.

    Fallback form: when ``atom_map`` is supplied, map reactant indices into
    product indices first, then compare bond existence and order.

    The result is in **reactant atom ordering** in either case.
    """
    r_orders = _bond_order_map(r_mol)
    if atom_map is None:
        p_orders = _bond_order_map(p_mol)
        out: List[CanonicalBond] = []
        for (i, j), r_order in r_orders.items():
            key = _canon(i, j)
            if key in p_orders and abs(r_order - p_orders[key]) > 1e-6:
                out.append(key)
        return sorted(out)
    # Fallback: use atom_map to remap reactant indices into product indices.
    p_orders = _bond_order_map(p_mol)
    out_fb: List[CanonicalBond] = []
    for (i, j), r_order in r_orders.items():
        try:
            pi, pj = atom_map[i], atom_map[j]
        except (IndexError, TypeError):
            continue
        p_key = _canon(int(pi), int(pj))
        if p_key in p_orders and abs(r_order - p_orders[p_key]) > 1e-6:
            out_fb.append(_canon(i, j))  # store in reactant ordering
    return sorted(out_fb)


# ---------------------------------------------------------------------------
# Unchanged near-core bonds (exact BFS-shell algorithm)
# ---------------------------------------------------------------------------

def _compute_unchanged_near_core(
    r_mol: 'Molecule',
    breaking_bonds: List[CanonicalBond],
    forming_bonds: List[CanonicalBond],
    changed_bonds: List[CanonicalBond],
) -> List[CanonicalBond]:
    """Return unchanged bonds in the first shell around the reactive core.

    Algorithm:

    1. **Reactive atom set**: union of all atom indices appearing in
       ``breaking_bonds``, ``forming_bonds``, or ``changed_bonds``.
    2. **Multi-source BFS** on the reactant graph from the reactive set,
       producing ``dist[a]`` for every atom.
    3. For every reactant bond ``(u, v)`` that is NOT already a reactive
       bond, include it iff
       ``min(dist[u], dist[v]) <= 1`` **and** ``max(dist[u], dist[v]) <= 2``.

    This selects unchanged bonds directly attached to the reactive core
    plus the next backbone shell, while excluding far spectator bonds.
    Far-away bonds (e.g. atoms 3+ hops from the core) are never returned.
    Atoms unreachable from the reactive set are treated as having
    distance ``+inf``.
    """
    reactive_atoms: Set[int] = set()
    for a, b in breaking_bonds:
        reactive_atoms.update((a, b))
    for a, b in forming_bonds:
        reactive_atoms.update((a, b))
    for a, b in changed_bonds:
        reactive_atoms.update((a, b))

    if not reactive_atoms:
        return []

    adj = _build_adjacency(r_mol)
    dist = _multi_source_bfs(adj, reactive_atoms)

    reactive_bond_set: Set[CanonicalBond] = set()
    for a, b in breaking_bonds:
        reactive_bond_set.add(_canon(a, b))
    for a, b in forming_bonds:
        reactive_bond_set.add(_canon(a, b))
    for a, b in changed_bonds:
        reactive_bond_set.add(_canon(a, b))

    INF = 10 ** 9
    out: List[CanonicalBond] = []
    for u, v in _all_bonds(r_mol):
        key = _canon(u, v)
        if key in reactive_bond_set:
            continue
        du = dist.get(u, INF)
        dv = dist.get(v, INF)
        if min(du, dv) <= 1 and max(du, dv) <= 2:
            out.append(key)
    return sorted(out)


# ---------------------------------------------------------------------------
# Reference-distance helpers
# ---------------------------------------------------------------------------

def _xyz_distance(xyz: dict, i: int, j: int) -> Optional[float]:
    """Return the i-j distance from an ARC xyz dict, or ``None`` if unavailable."""
    if xyz is None:
        return None
    coords = xyz.get('coords')
    if coords is None or i >= len(coords) or j >= len(coords):
        return None
    a = np.asarray(coords[i], dtype=float)
    b = np.asarray(coords[j], dtype=float)
    return float(np.linalg.norm(a - b))


def _bond_present(bond_orders: Dict[CanonicalBond, float], i: int, j: int) -> bool:
    return _canon(i, j) in bond_orders


def _safe_order(bond_orders: Dict[CanonicalBond, float], i: int, j: int) -> Optional[float]:
    return bond_orders.get(_canon(i, j))


# ---------------------------------------------------------------------------
# ReactionPathSpec dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReactionPathSpec:
    """Path-local description of a single reaction channel.

    This is the Phase 1 data model.  It is intentionally minimal: it captures
    *what* changes between reactant and product for one specific path, in a
    deterministic, canonical-keyed form, without any chemistry policy.

    All bond collections are stored as deterministic sorted lists of
    canonical ``(min, max)`` tuples in **reactant atom ordering**.
    Per-bond metadata is keyed by canonical bonds.
    """

    breaking_bonds: List[CanonicalBond] = field(default_factory=list)
    forming_bonds: List[CanonicalBond] = field(default_factory=list)
    changed_bonds: List[CanonicalBond] = field(default_factory=list)
    unchanged_near_core_bonds: List[CanonicalBond] = field(default_factory=list)

    reactive_atoms: Set[int] = field(default_factory=set)

    weight: float = 0.5
    family: Optional[str] = None

    # Per-bond bond orders (None on a side where the bond does not exist).
    bond_order_r: Dict[CanonicalBond, Optional[float]] = field(default_factory=dict)
    bond_order_p: Dict[CanonicalBond, Optional[float]] = field(default_factory=dict)

    # Per-bond reference distances (None if not available).
    ref_dist_r: Dict[CanonicalBond, Optional[float]] = field(default_factory=dict)
    ref_dist_p: Dict[CanonicalBond, Optional[float]] = field(default_factory=dict)

    @classmethod
    def build(cls,
              r_mol: 'Molecule',
              mapped_p_mol: Optional['Molecule'],
              breaking_bonds,
              forming_bonds,
              r_xyz: Optional[dict] = None,
              op_xyz: Optional[dict] = None,
              weight: float = 0.5,
              family: Optional[str] = None,
              atom_map: Optional[List[int]] = None,
              ) -> 'ReactionPathSpec':
        """Construct a deterministic ``ReactionPathSpec`` from path-local objects.

        Args:
            r_mol: Reactant RMG Molecule (defines reactant atom ordering).
            mapped_p_mol: Product RMG Molecule already in reactant atom
                ordering.  When supplied, ``changed_bonds`` is computed by
                direct comparison.  When ``None``, ``atom_map`` is used as a
                fallback.
            breaking_bonds: Iterable of (i, j) reactant-side bond tuples.
            forming_bonds: Iterable of (i, j) reactant-side bond tuples.
            r_xyz: Reactant XYZ dict (used for reference distances).
            op_xyz: Product XYZ already in reactant atom ordering (used for
                product-side reference distances).
            weight: Interpolation weight (0=R-like, 1=P-like).
            family: RMG reaction family name.
            atom_map: Fallback atom mapping for ``changed_bonds`` when
                ``mapped_p_mol`` is unavailable.

        Returns:
            A fully populated, deterministic ``ReactionPathSpec``.
        """
        bb_canon = _canon_list(breaking_bonds)
        fb_canon = _canon_list(forming_bonds)

        if mapped_p_mol is not None:
            changed = _compute_changed_bonds(r_mol, mapped_p_mol, atom_map=None)
        elif atom_map is not None:
            changed = _compute_changed_bonds(r_mol, mapped_p_mol or r_mol,
                                             atom_map=atom_map)
        else:
            changed = []

        # Drop any "changed" bond that is also flagged as breaking/forming —
        # those are reactive bonds, not pure bond-order changes.
        bb_set = set(bb_canon)
        fb_set = set(fb_canon)
        changed = [k for k in changed if k not in bb_set and k not in fb_set]

        unchanged_near_core = _compute_unchanged_near_core(
            r_mol, bb_canon, fb_canon, changed)

        reactive_atoms: Set[int] = set()
        for a, b in bb_canon + fb_canon + changed:
            reactive_atoms.update((a, b))

        # Per-bond bond orders.
        r_orders = _bond_order_map(r_mol)
        p_orders = _bond_order_map(mapped_p_mol) if mapped_p_mol is not None else {}

        bond_order_r: Dict[CanonicalBond, Optional[float]] = {}
        bond_order_p: Dict[CanonicalBond, Optional[float]] = {}
        ref_dist_r: Dict[CanonicalBond, Optional[float]] = {}
        ref_dist_p: Dict[CanonicalBond, Optional[float]] = {}

        for key in bb_canon + fb_canon + changed + unchanged_near_core:
            i, j = key
            bond_order_r[key] = _safe_order(r_orders, i, j)
            bond_order_p[key] = _safe_order(p_orders, i, j) if p_orders else None
            ref_dist_r[key] = _xyz_distance(r_xyz, i, j) if r_xyz is not None else None
            ref_dist_p[key] = _xyz_distance(op_xyz, i, j) if op_xyz is not None else None

        return cls(
            breaking_bonds=bb_canon,
            forming_bonds=fb_canon,
            changed_bonds=changed,
            unchanged_near_core_bonds=unchanged_near_core,
            reactive_atoms=reactive_atoms,
            weight=float(weight),
            family=family,
            bond_order_r=bond_order_r,
            bond_order_p=bond_order_p,
            ref_dist_r=ref_dist_r,
            ref_dist_p=ref_dist_p,
        )


# ---------------------------------------------------------------------------
# Role-aware target distance helper
# ---------------------------------------------------------------------------

VALID_ROLES = ('breaking', 'forming', 'changed', 'unchanged_near_core')


def get_ts_target_distance(
    bond: Tuple[int, int],
    role: str,
    symbols: Tuple[str, ...],
    d_r: Optional[float] = None,
    d_p: Optional[float] = None,
    bo_r: Optional[float] = None,
    bo_p: Optional[float] = None,
    weight: float = 0.5,
    family: Optional[str] = None,
) -> float:
    """Return a TS target distance for *bond* given its role.

    The function is foundation plumbing.  It does NOT change chemistry policy.
    For ``breaking`` and ``forming`` it reuses the same single-bond-length +
    Pauling-delta semantics already used in the codebase.

    Args:
        bond: ``(i, j)`` atom-index pair (used to look up symbols).
        role: One of ``'breaking'``, ``'forming'``, ``'changed'``, or
            ``'unchanged_near_core'``.
        symbols: Tuple of element symbols, indexed compatibly with ``bond``.
        d_r: Reactant reference distance (Å), if available.
        d_p: Product reference distance (Å), if available.
        bo_r: Reactant bond order, if available.
        bo_p: Product bond order, if available.
        weight: Interpolation weight; 0 = reactant-like, 1 = product-like.
        family: Reaction family name (currently unused but reserved).

    Returns:
        Target distance in Å.

    Raises:
        ValueError: If ``role`` is not a recognized role.
    """
    if role not in VALID_ROLES:
        raise ValueError(f'Unknown role {role!r}; expected one of {VALID_ROLES}.')

    i, j = int(bond[0]), int(bond[1])
    sym_i = symbols[i]
    sym_j = symbols[j]
    sbl = get_single_bond_length(sym_i, sym_j) or 1.5

    if role == 'breaking':
        # Same semantics already used by adjust_reactive_bond_distances:
        # stretch breaking bonds toward sbl + Pauling delta.
        return float(sbl + PAULING_DELTA)

    if role == 'forming':
        # Same semantics already used by adjust_reactive_bond_distances:
        # forming bonds also approach sbl + Pauling delta from above/below.
        return float(sbl + PAULING_DELTA)

    if role == 'changed':
        # Bond exists on both sides but its order changes.  Linear interpolation
        # is the conservative, well-defined choice.
        if d_r is not None and d_p is not None:
            w = max(0.0, min(1.0, float(weight)))
            return float((1.0 - w) * d_r + w * d_p)
        if d_r is not None:
            return float(d_r)
        if d_p is not None:
            return float(d_p)
        # Conservative fallback: a single-bond length.  Documented and tested.
        return float(sbl)

    # role == 'unchanged_near_core'
    if d_r is not None:
        return float(d_r)
    return float(sbl)


# ---------------------------------------------------------------------------
# Recipe-channel mismatch
# ---------------------------------------------------------------------------

def has_recipe_channel_mismatch(
    path_spec: ReactionPathSpec,
    xyz: dict,
    r_mol: 'Molecule',
) -> Tuple[bool, str]:
    """Return ``(True, reason)`` if *xyz* violates the recipe channel.

    Phase 1 implements the exact, conservative checks specified in the
    handoff:

    1. **Failed to break** — any bond in ``path_spec.breaking_bonds`` whose
       distance is below 1.30 Å.
    2. **Failed to form** — any bond in ``path_spec.forming_bonds`` whose
       distance exceeds 3.00 Å.
    3. **Snapped spectator** — any bond in
       ``path_spec.unchanged_near_core_bonds`` whose distance exceeds
       ``1.25 × get_single_bond_length(symbols[i], symbols[j])``.

    Args:
        path_spec: The path-local spec to compare against.
        xyz: TS guess XYZ dict.
        r_mol: Reactant molecule (used only for the symbol fallback).

    Returns:
        ``(True, reason)`` if a mismatch is detected, otherwise
        ``(False, '')``.
    """
    if xyz is None:
        return False, ''
    coords = np.asarray(xyz['coords'], dtype=float)
    symbols = xyz['symbols']

    # 1. Failed to break.
    for i, j in path_spec.breaking_bonds:
        d = float(np.linalg.norm(coords[i] - coords[j]))
        if d < 1.30:
            return True, f'failed-to-break: {symbols[i]}{i}-{symbols[j]}{j}={d:.3f}'

    # 2. Failed to form.
    for i, j in path_spec.forming_bonds:
        d = float(np.linalg.norm(coords[i] - coords[j]))
        if d > 3.00:
            return True, f'failed-to-form: {symbols[i]}{i}-{symbols[j]}{j}={d:.3f}'

    # 3. Snapped spectator (unchanged near-core bond pulled apart).
    for i, j in path_spec.unchanged_near_core_bonds:
        sbl = get_single_bond_length(symbols[i], symbols[j]) or 1.5
        limit = 1.25 * sbl
        d = float(np.linalg.norm(coords[i] - coords[j]))
        if d > limit:
            return True, (f'snapped-spectator: {symbols[i]}{i}-{symbols[j]}{j}='
                          f'{d:.3f} > {limit:.3f}')

    return False, ''


# ---------------------------------------------------------------------------
# Path chemistry classification
# ---------------------------------------------------------------------------


class PathChemistry(Enum):
    """Coarse-grained chemistry buckets used for validator and scorer routing.

    The classification is intentionally derived from the path-local
    :class:`ReactionPathSpec` (and reactant atom symbols) only — it does NOT
    use family strings.  This makes the buckets robust to RMG family naming
    drift and lets the orchestration apply consistent validation/triage
    behavior across families that share the same underlying chemistry.
    """

    SUBSTITUTION_LIKE = 'substitution_like'
    H_TRANSFER = 'h_transfer'
    NON_H_GROUP_SHIFT = 'non_h_group_shift'
    CONCERTED_HETERO_REARRANGEMENT = 'concerted_hetero_rearrangement'
    CYCLOADDITION_OR_RING_CLOSURE = 'cycloaddition_or_ring_closure'
    GENERIC = 'generic'


def _heavy_forming_bonds(
    forming_bonds: List[CanonicalBond],
    symbols: Tuple[str, ...],
) -> List[CanonicalBond]:
    """Return forming bonds whose endpoints are both heavy (non-H) atoms."""
    heavy: List[CanonicalBond] = []
    for i, j in forming_bonds:
        if 0 <= i < len(symbols) and 0 <= j < len(symbols):
            if symbols[i] != 'H' and symbols[j] != 'H':
                heavy.append((i, j))
    return heavy


def _shared_atoms_between_bb_and_fb(
    breaking_bonds: List[CanonicalBond],
    forming_bonds: List[CanonicalBond],
) -> Set[int]:
    """Return atom indices that participate in both a breaking and a forming bond."""
    bb_atoms: Set[int] = set()
    for a, b in breaking_bonds:
        bb_atoms.update((a, b))
    fb_atoms: Set[int] = set()
    for a, b in forming_bonds:
        fb_atoms.update((a, b))
    return bb_atoms & fb_atoms


def classify_path_chemistry(
    path_spec: ReactionPathSpec,
    r_mol: Optional['Molecule'],
    symbols: Tuple[str, ...],
) -> PathChemistry:
    """Classify *path_spec* into a coarse-grained chemistry bucket.

    The rules are evaluated in this exact order; the first matching rule wins:

    1. **SUBSTITUTION_LIKE** — exactly one breaking bond, exactly one forming
       bond, and exactly one atom shared between the two bonds, and that
       shared atom is not a hydrogen.  This captures classical SN2-like
       substitutions where a single non-H atom is the pivot.
    2. **H_TRANSFER** — any atom shared between a breaking and a forming
       bond is a hydrogen.  This captures every reaction where an H atom
       walks between two heavy atoms (intra/inter H-migration, ene, RH-add).
    3. **NON_H_GROUP_SHIFT** — any atom shared between a breaking and a
       forming bond is a heavy atom (e.g. 1,2-shift of C/S/halogen).
    4. **CONCERTED_HETERO_REARRANGEMENT** — at least two breaking bonds,
       at least two forming bonds, and at least two heavy reactive atoms
       are heteroatoms (N/O/S/P).  This captures Korcek-like and many
       elimination/cycloreversion patterns built around heteroatom cores.
    5. **CYCLOADDITION_OR_RING_CLOSURE** — at least one heavy-heavy
       forming bond.  This catches Diels–Alder, exo/endo cyclisations,
       and most ring-forming additions.
    6. **GENERIC** — anything else.

    Args:
        path_spec: The :class:`ReactionPathSpec` describing the channel.
        r_mol: Reactant molecule (currently unused; reserved for future
            adjacency-based refinements).
        symbols: Tuple of atom symbols indexed compatibly with the bonds
            in *path_spec*.

    Returns:
        The matching :class:`PathChemistry` bucket.
    """
    bb = list(path_spec.breaking_bonds)
    fb = list(path_spec.forming_bonds)
    shared = _shared_atoms_between_bb_and_fb(bb, fb)

    # Rule 1: SUBSTITUTION_LIKE
    if len(bb) == 1 and len(fb) == 1 and len(shared) == 1:
        shared_atom = next(iter(shared))
        if 0 <= shared_atom < len(symbols) and symbols[shared_atom] != 'H':
            return PathChemistry.SUBSTITUTION_LIKE

    # Rule 2: H_TRANSFER (any shared atom is H)
    if shared and any(0 <= a < len(symbols) and symbols[a] == 'H' for a in shared):
        return PathChemistry.H_TRANSFER

    # Rule 3: NON_H_GROUP_SHIFT (any shared atom is non-H)
    if shared and any(0 <= a < len(symbols) and symbols[a] != 'H' for a in shared):
        return PathChemistry.NON_H_GROUP_SHIFT

    # Rule 4: CONCERTED_HETERO_REARRANGEMENT
    if len(bb) >= 2 and len(fb) >= 2:
        reactive_heavy_hetero: Set[int] = set()
        for a, b in bb + fb:
            for idx in (a, b):
                if 0 <= idx < len(symbols) and symbols[idx] in HETERO_HEAVY_ATOMS:
                    reactive_heavy_hetero.add(idx)
        if len(reactive_heavy_hetero) >= 2:
            return PathChemistry.CONCERTED_HETERO_REARRANGEMENT

    # Rule 5: CYCLOADDITION_OR_RING_CLOSURE
    heavy_fb = _heavy_forming_bonds(fb, symbols)
    if len(heavy_fb) >= 1:
        return PathChemistry.CYCLOADDITION_OR_RING_CLOSURE

    # Rule 6: GENERIC fallback
    return PathChemistry.GENERIC


# ---------------------------------------------------------------------------
# Phase 2 validation sub-checks
# ---------------------------------------------------------------------------


def has_bad_changed_bond_length(
    path_spec: ReactionPathSpec,
    xyz: dict,
    symbols: Tuple[str, ...],
) -> Tuple[bool, str]:
    """Reject guesses with bond-order-changed bonds far from their target distance.

    For each bond in :attr:`ReactionPathSpec.changed_bonds`, the target
    distance is obtained from :func:`get_ts_target_distance` (role
    ``'changed'``).  Bonds with no resolvable target are skipped.  The
    rejection thresholds are:

    * 0.20 Å when at least one endpoint is hydrogen
    * 0.25 Å when both endpoints are heavy atoms

    Args:
        path_spec: Path-local spec.
        xyz: TS guess XYZ dict.
        symbols: Atom symbols indexed compatibly with the bonds.

    Returns:
        ``(True, reason)`` if any changed bond is out of tolerance,
        otherwise ``(False, '')``.
    """
    if xyz is None or not path_spec.changed_bonds:
        return False, ''
    coords = np.asarray(xyz['coords'], dtype=float)

    # Phase 2b — strict frontier exemption.
    #
    # A "changed" bond is exempt from the strict distance check ONLY when
    # BOTH conditions hold:
    #
    #   (1) the absolute bond-order shift is at least 0.5
    #       (the bond is undergoing real electronic restructuring), AND
    #   (2) the bond is *directly physically adjacent* to a breaking or
    #       forming bond — it shares at least one atom with at least one
    #       bond in path_spec.breaking_bonds ∪ path_spec.forming_bonds.
    #
    # If a changed bond has a large order shift but is geometrically
    # isolated from the reactive core (no shared atom with any
    # breaking/forming bond), it MUST still be validated against its
    # target distance — those are the impostor channels we want to catch.
    frontier_atoms: Set[int] = set()
    for a, b in list(path_spec.breaking_bonds) + list(path_spec.forming_bonds):
        frontier_atoms.update((int(a), int(b)))

    def _is_frontier_exempt(bi: int, bj: int,
                            bo_r_local: Optional[float],
                            bo_p_local: Optional[float]) -> bool:
        # Condition (1): non-trivial bond-order shift.
        if bo_r_local is None or bo_p_local is None:
            return False
        if abs(float(bo_r_local) - float(bo_p_local)) < 0.5:
            return False
        # Condition (2): direct topological adjacency to a breaking/forming bond.
        if (bi not in frontier_atoms) and (bj not in frontier_atoms):
            return False
        return True

    for i, j in path_spec.changed_bonds:
        if i >= len(coords) or j >= len(coords):
            continue
        d_r = path_spec.ref_dist_r.get((i, j))
        d_p = path_spec.ref_dist_p.get((i, j))
        bo_r = path_spec.bond_order_r.get((i, j))
        bo_p = path_spec.bond_order_p.get((i, j))
        # Strict 2-condition frontier exemption (Phase 2b).
        if _is_frontier_exempt(i, j, bo_r, bo_p):
            continue
        try:
            target = get_ts_target_distance(
                bond=(i, j),
                role='changed',
                symbols=symbols,
                d_r=d_r,
                d_p=d_p,
                bo_r=bo_r,
                bo_p=bo_p,
                weight=path_spec.weight,
                family=path_spec.family,
            )
        except (ValueError, IndexError):
            continue
        if target is None:
            continue
        d = float(np.linalg.norm(coords[i] - coords[j]))
        h_involved = (symbols[i] == 'H' or symbols[j] == 'H')
        tol = 0.20 if h_involved else 0.25
        if abs(d - target) > tol:
            return True, (f'bad-changed-bond: {symbols[i]}{i}-{symbols[j]}{j}'
                          f'={d:.3f} target={target:.3f} tol={tol:.2f}')
    return False, ''


def has_bad_unchanged_near_core_bond(
    path_spec: ReactionPathSpec,
    xyz: dict,
    symbols: Tuple[str, ...],
) -> Tuple[bool, str]:
    """Reject guesses where an unchanged near-core bond drifts too far.

    For each bond in :attr:`ReactionPathSpec.unchanged_near_core_bonds`,
    use the stored reactant reference distance if available, otherwise
    fall back to :func:`get_ts_target_distance` (role
    ``'unchanged_near_core'``).  Reject the guess if the actual distance
    is below ``0.82 × target`` or above ``1.25 × target``.

    Args:
        path_spec: Path-local spec.
        xyz: TS guess XYZ dict.
        symbols: Atom symbols indexed compatibly with the bonds.

    Returns:
        ``(True, reason)`` if any unchanged near-core bond is out of
        tolerance, otherwise ``(False, '')``.
    """
    if xyz is None or not path_spec.unchanged_near_core_bonds:
        return False, ''
    coords = np.asarray(xyz['coords'], dtype=float)
    for i, j in path_spec.unchanged_near_core_bonds:
        if i >= len(coords) or j >= len(coords):
            continue
        target = path_spec.ref_dist_r.get((i, j))
        if target is None:
            try:
                target = get_ts_target_distance(
                    bond=(i, j),
                    role='unchanged_near_core',
                    symbols=symbols,
                    d_r=None,
                    weight=path_spec.weight,
                    family=path_spec.family,
                )
            except (ValueError, IndexError):
                continue
        if target is None or target <= 0.0:
            continue
        d = float(np.linalg.norm(coords[i] - coords[j]))
        if d < 0.82 * target or d > 1.25 * target:
            return True, (f'bad-unchanged-near-core: {symbols[i]}{i}-{symbols[j]}{j}'
                          f'={d:.3f} ref={target:.3f}')
    return False, ''


def has_inward_blocking_h_on_forming_axis(
    path_spec: ReactionPathSpec,
    xyz: dict,
    r_mol: Optional['Molecule'],
    symbols: Tuple[str, ...],
    chemistry: PathChemistry,
) -> Tuple[bool, str]:
    """Reject guesses where a non-reactive H blocks a heavy-heavy forming axis.

    Only active when *chemistry* is :attr:`PathChemistry.CYCLOADDITION_OR_RING_CLOSURE`
    or :attr:`PathChemistry.SUBSTITUTION_LIKE`.

    For each heavy-heavy forming bond ``(a, b)``, scan non-reactive
    hydrogens attached (graph-bonded) to either endpoint.  An H is
    considered "inward-blocking" if both:

    * The angle ``∠(H, endpoint, opposite_endpoint) < 85°`` (it points
      across the forming axis)
    * The distance ``d(H, opposite_endpoint) < 1.60 Å`` (it intrudes
      into the forming pocket)

    Args:
        path_spec: Path-local spec.
        xyz: TS guess XYZ dict.
        r_mol: Reactant molecule (used for graph adjacency).
        symbols: Atom symbols indexed compatibly with the bonds.
        chemistry: Pre-computed :class:`PathChemistry` bucket.

    Returns:
        ``(True, reason)`` if a blocking H is found, otherwise
        ``(False, '')``.
    """
    if chemistry not in (PathChemistry.CYCLOADDITION_OR_RING_CLOSURE,
                         PathChemistry.SUBSTITUTION_LIKE):
        return False, ''
    if xyz is None or r_mol is None:
        return False, ''
    heavy_fb = _heavy_forming_bonds(list(path_spec.forming_bonds), symbols)
    if not heavy_fb:
        return False, ''

    coords = np.asarray(xyz['coords'], dtype=float)
    adj = _build_adjacency(r_mol)
    reactive = set(path_spec.reactive_atoms)

    for a, b in heavy_fb:
        for ep, opp in ((a, b), (b, a)):
            for nbr in adj.get(ep, ()):
                if nbr == opp:
                    continue
                if nbr >= len(symbols) or symbols[nbr] != 'H':
                    continue
                if nbr in reactive:
                    continue
                vec_h = coords[nbr] - coords[ep]
                vec_op = coords[opp] - coords[ep]
                norm_h = float(np.linalg.norm(vec_h))
                norm_op = float(np.linalg.norm(vec_op))
                if norm_h < 1e-6 or norm_op < 1e-6:
                    continue
                cos_theta = float(np.dot(vec_h, vec_op) / (norm_h * norm_op))
                cos_theta = max(-1.0, min(1.0, cos_theta))
                angle_deg = float(np.degrees(np.arccos(cos_theta)))
                d_h_opp = float(np.linalg.norm(coords[nbr] - coords[opp]))
                if angle_deg < 85.0 and d_h_opp < 1.60:
                    return True, (f'inward-blocking-H: H{nbr} on {symbols[ep]}{ep} '
                                  f'angle={angle_deg:.1f} d(H,{symbols[opp]}{opp})'
                                  f'={d_h_opp:.3f}')
    return False, ''


def has_bad_reactive_core_planarity(
    path_spec: ReactionPathSpec,
    xyz: dict,
    symbols: Tuple[str, ...],
    chemistry: PathChemistry,
) -> Tuple[bool, str]:
    """Reject concerted hetero rearrangements where the reactive core is non-planar.

    Only active when *chemistry* is
    :attr:`PathChemistry.CONCERTED_HETERO_REARRANGEMENT`.

    Collects all heavy reactive-core atoms (those participating in any
    breaking/forming/changed bond, excluding H), then SVD-fits the best
    plane to their coordinates.  Rejects the guess if the RMS distance
    from that plane exceeds 0.35 Å.  When the SVD itself fails (e.g.
    due to a degenerate geometry), the guess is rejected.

    Args:
        path_spec: Path-local spec.
        xyz: TS guess XYZ dict.
        symbols: Atom symbols indexed compatibly with the bonds.
        chemistry: Pre-computed :class:`PathChemistry` bucket.

    Returns:
        ``(True, reason)`` if the reactive core deviates from planarity
        too strongly, otherwise ``(False, '')``.
    """
    if chemistry != PathChemistry.CONCERTED_HETERO_REARRANGEMENT:
        return False, ''
    if xyz is None:
        return False, ''
    core_atoms: Set[int] = set()
    for a, b in (list(path_spec.breaking_bonds)
                 + list(path_spec.forming_bonds)
                 + list(path_spec.changed_bonds)):
        for idx in (a, b):
            if 0 <= idx < len(symbols) and symbols[idx] != 'H':
                core_atoms.add(idx)
    if len(core_atoms) < 4:
        return False, ''
    coords = np.asarray(xyz['coords'], dtype=float)
    try:
        pts = np.array([coords[i] for i in sorted(core_atoms)], dtype=float)
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        proj = np.dot(centered, normal)
        rms = float(np.sqrt(np.mean(proj * proj)))
    except Exception:
        # Treat SVD failure as a rejection — the geometry is degenerate
        # enough that the planarity test cannot be applied.
        return True, 'reactive-core-planarity: SVD failed'
    if rms > 0.35:
        return True, f'reactive-core-non-planar: rms={rms:.3f}'
    return False, ''


# ---------------------------------------------------------------------------
# Phase 4a — narrow recipe-consistency / wrong-channel screening
# ---------------------------------------------------------------------------


def has_wrong_h_migration_committed(
    path_spec: ReactionPathSpec,
    xyz: dict,
    r_mol: Optional['Molecule'],
    symbols: Tuple[str, ...],
    chemistry: PathChemistry,
) -> Tuple[bool, str]:
    """Reject H-transfer guesses where a *spectator* H sits much closer
    to the intended acceptor than the *intended* migrating H.

    This is a narrow Phase 4a screening rule.  It runs only when ALL
    of the following hold:

    * the path chemistry is :attr:`PathChemistry.H_TRANSFER`, AND
    * the path-spec contains at least one forming bond with exactly
      one H endpoint (the intended migrating H), AND
    * the migrating H still sits close to its donor in the guess
      (i.e. it has not yet started moving toward the acceptor).

    Inside the rule, the migrating H is the H endpoint of the forming
    bond.  The acceptor is the heavy endpoint of that same forming bond.
    The donor is the H's *reactant* heavy neighbor.  A spectator H is
    any other H atom in the molecule that is also bonded to a heavy
    atom in the reactant graph (i.e. a normal C–H or X–H), excluding
    the migrating H itself.

    The guess is rejected only when **both** conditions hold for at
    least one spectator H:

    1. The spectator H's distance to the acceptor is at most
       ``0.70 ×`` the migrating H's distance to the same acceptor —
       a meaningful margin, not "nearest H wins".
    2. The migrating H's distance to its donor is still close to a
       normal bond length (``≤ 1.20 × sbl(donor, H)``), meaning the
       intended channel has not actually engaged.

    Both conditions are required so the rule does not over-fire on
    legitimate TS geometries where the migrating H has begun moving
    toward the acceptor and momentarily passes near a spectator H.

    Args:
        path_spec: Path-local spec.
        xyz: TS guess XYZ dict.
        r_mol: Reactant RMG :class:`Molecule` (used for the donor /
            spectator graph adjacency).
        symbols: Tuple of atom symbols.
        chemistry: Pre-computed :class:`PathChemistry` bucket.

    Returns:
        ``(True, reason)`` if a clearly committed spectator H is
        detected, otherwise ``(False, '')``.
    """
    if chemistry != PathChemistry.H_TRANSFER:
        return False, ''
    if xyz is None or r_mol is None:
        return False, ''

    # Identify intended (h, donor, acceptor) triples from the path spec.
    triples: List[Tuple[int, int, int]] = []
    adj = _build_adjacency(r_mol)
    for a, b in path_spec.forming_bonds:
        if 0 <= a < len(symbols) and 0 <= b < len(symbols):
            if symbols[a] == 'H' and symbols[b] != 'H':
                h_idx, acceptor = int(a), int(b)
            elif symbols[b] == 'H' and symbols[a] != 'H':
                h_idx, acceptor = int(b), int(a)
            else:
                continue
            # Donor is the H's *reactant* heavy neighbor.
            donor: Optional[int] = None
            for nbr in adj.get(h_idx, ()):
                if 0 <= nbr < len(symbols) and symbols[nbr] != 'H':
                    donor = int(nbr)
                    break
            if donor is None or donor == acceptor:
                continue
            triples.append((h_idx, donor, acceptor))
    if not triples:
        return False, ''

    coords = np.asarray(xyz['coords'], dtype=float)

    for h_idx, donor, acceptor in triples:
        if h_idx >= len(coords) or donor >= len(coords) or acceptor >= len(coords):
            continue
        # Condition 2 first (cheaper): the migrating H must still be
        # near its donor.  If it has already started moving, the rule
        # does not apply because the intended channel has engaged.
        sbl_dh = get_single_bond_length(symbols[donor], 'H')
        if sbl_dh is None or sbl_dh <= 0.0:
            continue
        d_donor_h = float(np.linalg.norm(coords[h_idx] - coords[donor]))
        if d_donor_h > 1.20 * float(sbl_dh):
            continue
        d_intended_acceptor = float(np.linalg.norm(coords[h_idx] - coords[acceptor]))
        if d_intended_acceptor < 1e-6:
            continue
        # Now check condition 1: any spectator H much closer to the
        # acceptor.
        for h_other in range(len(symbols)):
            if h_other == h_idx:
                continue
            if symbols[h_other] != 'H':
                continue
            # Spectator H must itself be bonded to a heavy atom in the
            # reactant graph (skip stray H, lone H, or H atoms already
            # part of another reactive forming bond).
            heavy_parents = [n for n in adj.get(h_other, ())
                             if 0 <= n < len(symbols) and symbols[n] != 'H']
            if not heavy_parents:
                continue
            d_spec_acceptor = float(np.linalg.norm(coords[h_other] - coords[acceptor]))
            if d_spec_acceptor > 0.70 * d_intended_acceptor:
                continue
            return True, (
                f'wrong-h-migration: spectator H{h_other} d(H,acceptor)='
                f'{d_spec_acceptor:.3f} much closer than intended H{h_idx} '
                f'd(H,acceptor)={d_intended_acceptor:.3f}; intended H still '
                f'near donor d(D,H)={d_donor_h:.3f}'
            )
    return False, ''


def has_committed_spectator_group(
    path_spec: ReactionPathSpec,
    xyz: dict,
    r_mol: Optional['Molecule'],
    symbols: Tuple[str, ...],
) -> Tuple[bool, str]:
    """Reject guesses where a spectator heavy atom is committed to a
    reactive site it does not appear in the path-spec for.

    This is a narrow Phase 4a screening rule.  It is **opt-in by
    family** — the check only runs for an explicit allowlist of
    families where the failure pattern is already known to occur:

    * ``1,3_Insertion_ROR``: the wrong CH₃ commits to the wrong
      carbonyl C, generating an impostor with a near-bond-length
      contact between a spectator heavy atom and a reactive site.
    * Reverse-template ``R_Addition_MultipleBond``: a wrong radical
      addition target attaches in the reverse direction.

    The rule rejects only when a heavy non-reactive atom has formed an
    obviously committed local contact (distance below ``0.85 × sbl``)
    with a heavy atom that is already an endpoint of one of the
    reactive bonds in the path-spec but is not its expected partner
    according to the spec.

    Args:
        path_spec: Path-local spec.
        xyz: TS guess XYZ dict.
        r_mol: Reactant RMG :class:`Molecule`.
        symbols: Tuple of atom symbols.

    Returns:
        ``(True, reason)`` if a clearly committed spectator heavy
        atom is detected, otherwise ``(False, '')``.
    """
    family = path_spec.family or ''
    # Strict, opt-in family allowlist.
    if family not in ('1,3_Insertion_ROR', 'R_Addition_MultipleBond'):
        return False, ''
    if xyz is None or r_mol is None:
        return False, ''

    coords = np.asarray(xyz['coords'], dtype=float)
    reactive = set(int(i) for i in path_spec.reactive_atoms)
    if not reactive:
        return False, ''

    # Heavy reactive endpoints we care about — the union over all
    # forming-bond and breaking-bond endpoints, restricted to heavy
    # atoms.
    heavy_reactive_endpoints: Set[int] = set()
    for a, b in (list(path_spec.forming_bonds)
                 + list(path_spec.breaking_bonds)):
        for idx in (a, b):
            if 0 <= idx < len(symbols) and symbols[idx] != 'H':
                heavy_reactive_endpoints.add(int(idx))
    if not heavy_reactive_endpoints:
        return False, ''

    # The set of "approved" heavy partners for each reactive endpoint —
    # i.e. the heavy atoms that the path-spec actually says it should
    # form/break a bond with.
    approved_partners: Dict[int, Set[int]] = {ep: set()
                                              for ep in heavy_reactive_endpoints}
    for a, b in (list(path_spec.forming_bonds)
                 + list(path_spec.breaking_bonds)):
        if a in heavy_reactive_endpoints and 0 <= b < len(symbols) and symbols[b] != 'H':
            approved_partners[int(a)].add(int(b))
        if b in heavy_reactive_endpoints and 0 <= a < len(symbols) and symbols[a] != 'H':
            approved_partners[int(b)].add(int(a))

    n_atoms = len(symbols)
    for ep in sorted(heavy_reactive_endpoints):
        if ep >= len(coords):
            continue
        approved = approved_partners[ep]
        for k in range(n_atoms):
            if k == ep or k in reactive or k in approved:
                continue
            if symbols[k] == 'H':
                continue
            sbl = get_single_bond_length(symbols[ep], symbols[k])
            if sbl is None or sbl <= 0.0:
                continue
            d = float(np.linalg.norm(coords[ep] - coords[k]))
            if d < 0.85 * float(sbl):
                return True, (
                    f'committed-spectator: {symbols[k]}{k} '
                    f'd(spectator,{symbols[ep]}{ep})={d:.3f} < 0.85*sbl='
                    f'{0.85 * float(sbl):.3f}; family={family!r}'
                )
    return False, ''


# ---------------------------------------------------------------------------
# Phase 2 path-spec scoring
# ---------------------------------------------------------------------------


def score_guess_against_path_spec(
    path_spec: ReactionPathSpec,
    xyz: dict,
    r_mol: Optional['Molecule'],
    symbols: Tuple[str, ...],
    chemistry: Optional[PathChemistry] = None,
) -> float:
    """Compute a path-spec fidelity score for *xyz* (lower = better).

    The score sums per-bond deviations from role-aware target distances
    and adds fixed penalties for the planarity and inward-blocking-H
    sub-checks.  No family-specific weighting is applied.

    Per-bond contributions::

        breaking            : abs(d - target) / 0.35
        forming             : abs(d - target) / 0.35
        changed (H-involved): abs(d - target) / 0.20
        changed (heavy)     : abs(d - target) / 0.25
        unchanged_near_core : abs(d - target) / 0.20

    Penalty contributions::

        +5.0 if has_inward_blocking_h_on_forming_axis returns True
        +5.0 if has_bad_reactive_core_planarity returns True

    Args:
        path_spec: Path-local spec to score against.
        xyz: TS guess XYZ dict.
        r_mol: Reactant molecule (needed for the inward-blocking-H check).
        symbols: Atom symbols indexed compatibly with the bonds.
        chemistry: Pre-computed :class:`PathChemistry` bucket.  If ``None``,
            the function classifies internally.

    Returns:
        A non-negative float; lower is better.
    """
    if xyz is None:
        return float('inf')
    coords = np.asarray(xyz['coords'], dtype=float)
    if chemistry is None:
        chemistry = classify_path_chemistry(path_spec, r_mol, symbols)

    score = 0.0

    def _bond_dist(i: int, j: int) -> Optional[float]:
        if i >= len(coords) or j >= len(coords):
            return None
        return float(np.linalg.norm(coords[i] - coords[j]))

    # Breaking bonds
    for i, j in path_spec.breaking_bonds:
        d = _bond_dist(i, j)
        if d is None:
            continue
        try:
            target = get_ts_target_distance(
                bond=(i, j), role='breaking', symbols=symbols,
                d_r=path_spec.ref_dist_r.get((i, j)),
                d_p=path_spec.ref_dist_p.get((i, j)),
                bo_r=path_spec.bond_order_r.get((i, j)),
                bo_p=path_spec.bond_order_p.get((i, j)),
                weight=path_spec.weight, family=path_spec.family)
        except (ValueError, IndexError):
            continue
        score += abs(d - target) / 0.35

    # Forming bonds
    for i, j in path_spec.forming_bonds:
        d = _bond_dist(i, j)
        if d is None:
            continue
        try:
            target = get_ts_target_distance(
                bond=(i, j), role='forming', symbols=symbols,
                d_r=path_spec.ref_dist_r.get((i, j)),
                d_p=path_spec.ref_dist_p.get((i, j)),
                bo_r=path_spec.bond_order_r.get((i, j)),
                bo_p=path_spec.bond_order_p.get((i, j)),
                weight=path_spec.weight, family=path_spec.family)
        except (ValueError, IndexError):
            continue
        score += abs(d - target) / 0.35

    # Changed bonds
    for i, j in path_spec.changed_bonds:
        d = _bond_dist(i, j)
        if d is None:
            continue
        try:
            target = get_ts_target_distance(
                bond=(i, j), role='changed', symbols=symbols,
                d_r=path_spec.ref_dist_r.get((i, j)),
                d_p=path_spec.ref_dist_p.get((i, j)),
                bo_r=path_spec.bond_order_r.get((i, j)),
                bo_p=path_spec.bond_order_p.get((i, j)),
                weight=path_spec.weight, family=path_spec.family)
        except (ValueError, IndexError):
            continue
        if target is None:
            continue
        h_involved = (symbols[i] == 'H' or symbols[j] == 'H')
        denom = 0.20 if h_involved else 0.25
        score += abs(d - target) / denom

    # Unchanged near-core bonds
    for i, j in path_spec.unchanged_near_core_bonds:
        d = _bond_dist(i, j)
        if d is None:
            continue
        target = path_spec.ref_dist_r.get((i, j))
        if target is None:
            try:
                target = get_ts_target_distance(
                    bond=(i, j), role='unchanged_near_core', symbols=symbols,
                    d_r=None, weight=path_spec.weight, family=path_spec.family)
            except (ValueError, IndexError):
                continue
        if target is None or target <= 0.0:
            continue
        score += abs(d - target) / 0.20

    # Inward-blocking H penalty
    blocked, _ = has_inward_blocking_h_on_forming_axis(
        path_spec, xyz, r_mol, symbols, chemistry)
    if blocked:
        score += 5.0

    # Reactive-core planarity penalty
    nonplanar, _ = has_bad_reactive_core_planarity(
        path_spec, xyz, symbols, chemistry)
    if nonplanar:
        score += 5.0

    return float(score)


# ---------------------------------------------------------------------------
# Centralized validation wrapper
# ---------------------------------------------------------------------------

def validate_guess_against_path_spec(
    xyz: dict,
    path_spec: ReactionPathSpec,
    r_mol: 'Molecule',
    family: Optional[str] = None,
    anchor_xyz: Optional[dict] = None,
    reactive_indices: Optional[Set[int]] = None,
    migrating_hs: Optional[Set[int]] = None,
    label: str = '',
    strict_generic: bool = False,
) -> Tuple[bool, str]:
    """Run generic validation followed by the recipe-channel mismatch guard.

    This is the orchestration-level validation entry point.  It composes
    the existing :func:`validate_ts_guess` (collisions, detached atoms,
    fragment count, drift, family motif checks) with the new path-local
    :func:`has_recipe_channel_mismatch` guard.  Both checks are gating —
    failing either rejects the guess.

    Phase 1 behavior:
        By default (``strict_generic=False``) the wrapper does NOT pass
        ``anchor_xyz`` / ``reactive_indices`` into the generic validator.
        That keeps the orchestration's prior behavior intact and ensures
        the only new gate added in Phase 1 is
        :func:`has_recipe_channel_mismatch`.  Setting ``strict_generic=True``
        opts in to the strongest sanity checks (backbone-drift screening
        relative to ``anchor_xyz``, etc.) — that capability is intentionally
        deferred to Phase 2.

    Args:
        xyz: TS guess XYZ dict.
        path_spec: Path-local spec for the reaction channel.
        r_mol: Reactant molecule.
        family: Reaction family (overrides ``path_spec.family`` for the
            generic validator if explicitly supplied).
        anchor_xyz: Optional anchor geometry for backbone-drift checking
            (only honored when ``strict_generic=True``).
        reactive_indices: Atom indices to exempt from generic detached/drift
            checks (only honored when ``strict_generic=True``).
        migrating_hs: Migrating-hydrogen indices for the family validator.
        label: Logging label.
        strict_generic: If True, pass ``anchor_xyz``/``reactive_indices``
            into the generic validator (activating drift checks).  Defaults
            to False to preserve current orchestration behavior.

    Returns:
        ``(True, '')`` if the guess passes both layers; otherwise
        ``(False, reason)`` with an explicit rejection reason.
    """
    if migrating_hs is None:
        migrating_hs = set()
    effective_family = family if family is not None else path_spec.family

    # Pre-compute the path chemistry once — it is used by both the generic
    # validator dispatch (H_TRANSFER routing) and the new sub-checks.
    if xyz is not None and 'symbols' in xyz:
        symbols = tuple(xyz['symbols'])
    elif r_mol is not None:
        symbols = tuple(a.element.symbol for a in r_mol.atoms)
    else:
        symbols = tuple()

    try:
        chemistry = classify_path_chemistry(path_spec, r_mol, symbols)
    except Exception:
        chemistry = PathChemistry.GENERIC

    # 1. Generic validation (collisions, detached atoms, fragments,
    #    family-specific motif filters).  By default we DO NOT pass anchor
    #    arguments — that preserves Phase 0 orchestration behavior.
    if strict_generic:
        anchor_kwarg = anchor_xyz
        reactive_kwarg = (reactive_indices if reactive_indices is not None
                          else set(path_spec.reactive_atoms))
    else:
        anchor_kwarg = None
        reactive_kwarg = None

    ok_generic, reason_generic = validate_ts_guess(
        xyz=xyz,
        migrating_hs=migrating_hs,
        forming_bonds=list(path_spec.forming_bonds),
        r_mol=r_mol,
        label=label,
        family=effective_family,
        anchor_xyz=anchor_kwarg,
        reactive_indices=reactive_kwarg,
        chemistry=chemistry.value if chemistry is not None else None,
    )
    if not ok_generic:
        return False, reason_generic

    # 2. Path-local recipe channel mismatch (Phase 1 guard, retained).
    mismatch, reason_mismatch = has_recipe_channel_mismatch(path_spec, xyz, r_mol)
    if mismatch:
        return False, f'recipe-mismatch:{reason_mismatch}'

    # 3. Phase 2 sub-checks: changed-bond length, unchanged-near-core,
    #    inward-blocking H, and reactive-core planarity.
    bad, reason = has_bad_changed_bond_length(path_spec, xyz, symbols)
    if bad:
        return False, f'phase2:{reason}'
    bad, reason = has_bad_unchanged_near_core_bond(path_spec, xyz, symbols)
    if bad:
        return False, f'phase2:{reason}'
    bad, reason = has_inward_blocking_h_on_forming_axis(
        path_spec, xyz, r_mol, symbols, chemistry)
    if bad:
        return False, f'phase2:{reason}'
    bad, reason = has_bad_reactive_core_planarity(
        path_spec, xyz, symbols, chemistry)
    if bad:
        return False, f'phase2:{reason}'

    # 4. Phase 4a narrow recipe-consistency / wrong-channel screening.
    #    These are strictly additive and only fire on opt-in cases:
    #    * has_wrong_h_migration_committed: only when chemistry is
    #      H_TRANSFER and a path-spec migrating H is named.
    #    * has_committed_spectator_group: only for an explicit family
    #      allowlist (1,3_Insertion_ROR, R_Addition_MultipleBond).
    bad, reason = has_wrong_h_migration_committed(
        path_spec, xyz, r_mol, symbols, chemistry)
    if bad:
        return False, f'phase4a:{reason}'
    bad, reason = has_committed_spectator_group(
        path_spec, xyz, r_mol, symbols)
    if bad:
        return False, f'phase4a:{reason}'

    return True, ''
