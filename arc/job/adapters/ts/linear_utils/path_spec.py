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
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from arc.common import get_single_bond_length
from arc.job.adapters.ts.linear_utils.postprocess import (
    PAULING_DELTA,
    validate_ts_guess,
)

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
    )
    if not ok_generic:
        return False, reason_generic

    # 2. Path-local recipe channel mismatch.
    mismatch, reason_mismatch = has_recipe_channel_mismatch(path_spec, xyz, r_mol)
    if mismatch:
        return False, f'recipe-mismatch:{reason_mismatch}'

    return True, ''
