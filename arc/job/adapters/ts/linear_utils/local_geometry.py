"""Phase 3b — local reactive-center geometry helpers.

This module provides small, deterministic, *local* utilities for cleaning
up post-migration addition TS geometries before they are validated by the
final-stage path-spec wrapper.

Design rules (from the Phase 3b directive):

* Each helper only moves the immediate terminal group or an explicitly
  supplied atom subset.
* No global torsion scans, no whole-molecule rotation, no broad
  "fix-everything" passes.
* Bond lengths are preserved when nothing else compels a change.
* Helpers are deterministic — calling them twice on the same input
  produces identical output.
* Helpers preserve symmetry when symmetry is meaningful (CH₂/CH₃).
* Helpers do not introduce H crowding or new collisions.

These helpers are intentionally side-effect free: each one accepts an XYZ
dict and returns a *new* XYZ dict.
"""

from collections import deque
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from arc.common import get_logger, get_single_bond_length
from arc.job.adapters.ts.linear_utils.postprocess import PAULING_DELTA

if TYPE_CHECKING:
    from arc.molecule import Molecule
    from arc.species.species import ARCSpecies


logger = get_logger()


# ---------------------------------------------------------------------------
# Small XYZ helpers
# ---------------------------------------------------------------------------


def _xyz_with_coords(xyz: dict, coords: np.ndarray) -> dict:
    """Return a new XYZ dict that copies *xyz* but replaces ``coords``."""
    return {
        'symbols': xyz['symbols'],
        'isotopes': xyz.get('isotopes',
                            tuple(0 for _ in range(len(xyz['symbols'])))),
        'coords': tuple(tuple(float(x) for x in row) for row in coords),
    }


def _heavy_neighbors(mol: 'Molecule', atom_idx: int,
                      symbols: Tuple[str, ...]) -> List[int]:
    """Return graph-bonded non-H neighbor indices of ``atom_idx``."""
    atom_to_idx = {atom: i for i, atom in enumerate(mol.atoms)}
    neighbors: List[int] = []
    for nbr in mol.atoms[atom_idx].bonds.keys():
        ni = atom_to_idx[nbr]
        if symbols[ni] != 'H':
            neighbors.append(ni)
    return neighbors


def _h_neighbors(mol: 'Molecule', atom_idx: int,
                  symbols: Tuple[str, ...]) -> List[int]:
    """Return graph-bonded H neighbor indices of ``atom_idx``."""
    atom_to_idx = {atom: i for i, atom in enumerate(mol.atoms)}
    h_idxs: List[int] = []
    for nbr in mol.atoms[atom_idx].bonds.keys():
        ni = atom_to_idx[nbr]
        if symbols[ni] == 'H':
            h_idxs.append(ni)
    return h_idxs


# ---------------------------------------------------------------------------
# Helper 1 — terminal CH2/CH3 H bond-length regularization
# ---------------------------------------------------------------------------


def regularize_terminal_h_geometry(xyz: dict,
                                    mol: 'Molecule',
                                    center: int,
                                    exclude_atoms: Optional[Set[int]] = None,
                                    ) -> dict:
    """Snap terminal H bond lengths around ``center`` back to ~sbl.

    For each H atom bonded to ``center`` in the molecular graph (skipping
    any in ``exclude_atoms``), if its current distance from ``center``
    is more than ~30 % away from ``get_single_bond_length(C, H)``, the H
    is moved along the existing C–H direction so that the new distance
    equals the standard single-bond length.  Direction is preserved.
    Atoms not in the immediate first shell of ``center`` are never
    touched.

    This is a *bond length* regularizer.  It does not change angles or
    rotate H atoms.

    Args:
        xyz: TS guess XYZ dict.
        mol: RMG Molecule defining the bond graph.
        center: Heavy-atom index whose H neighbors should be regularized.
        exclude_atoms: Optional H indices to skip (e.g. the migrating H).

    Returns:
        New XYZ dict with the regularized H positions.  When no H atom
        needs regularization the original input is returned unchanged.
    """
    if xyz is None:
        return xyz
    coords = np.asarray(xyz['coords'], dtype=float).copy()
    symbols = xyz['symbols']
    if center >= len(coords) or symbols[center] == 'H':
        return xyz
    exclude = exclude_atoms or set()
    h_idxs = [h for h in _h_neighbors(mol, center, symbols) if h not in exclude]
    if not h_idxs:
        return xyz

    sbl = get_single_bond_length(symbols[center], 'H')
    if sbl is None or sbl <= 0.0:
        return xyz
    low = sbl * 0.70
    high = sbl * 1.30

    changed = False
    center_pos = coords[center]
    for h in h_idxs:
        vec = coords[h] - center_pos
        d = float(np.linalg.norm(vec))
        if d < 1e-6:
            continue
        if low <= d <= high:
            continue
        coords[h] = center_pos + (vec / d) * sbl
        changed = True
    if not changed:
        return xyz
    return _xyz_with_coords(xyz, coords)


# ---------------------------------------------------------------------------
# Helper 2 — orient non-migrating H atoms away from a donor–acceptor axis
# ---------------------------------------------------------------------------


def orient_h_away_from_axis(xyz: dict,
                              mol: 'Molecule',
                              donor: int,
                              acceptor: int,
                              exclude_h: Optional[Set[int]] = None,
                              ) -> dict:
    """Rotate non-migrating H atoms around their parent C–H bond
    until they no longer block the donor–acceptor reaction axis.

    For each non-migrating H atom whose parent (graph neighbor) is
    ``donor`` or ``acceptor`` and whose current geometry satisfies
    ``∠(H, parent, opposite) < 85°`` AND ``d(H, opposite) < 1.60 Å``,
    the H is reflected through the (parent, opposite) axis so it points
    outward.  This is the same blocking condition the Phase 2 sub-check
    :func:`has_inward_blocking_h_on_forming_axis` rejects on.

    The reflection moves only one H atom per call iteration; nothing
    else.  No molecule-wide rotation.

    Args:
        xyz: TS guess XYZ dict.
        mol: RMG Molecule.
        donor: Donor heavy-atom index.
        acceptor: Acceptor heavy-atom index.
        exclude_h: Optional set of H indices to skip (the migrating H).

    Returns:
        A new XYZ dict; if no H needed re-orientation, the original is
        returned unchanged.
    """
    if xyz is None:
        return xyz
    coords = np.asarray(xyz['coords'], dtype=float).copy()
    symbols = xyz['symbols']
    if donor >= len(coords) or acceptor >= len(coords):
        return xyz
    exclude = exclude_h or set()

    changed = False
    for parent, opposite in ((donor, acceptor), (acceptor, donor)):
        parent_pos = coords[parent]
        opp_pos = coords[opposite]
        axis = opp_pos - parent_pos
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-6:
            continue
        axis_hat = axis / axis_norm
        for h in _h_neighbors(mol, parent, symbols):
            if h in exclude:
                continue
            h_vec = coords[h] - parent_pos
            h_norm = float(np.linalg.norm(h_vec))
            if h_norm < 1e-6:
                continue
            cos_theta = float(np.dot(h_vec, axis_hat) / h_norm)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            angle_deg = float(np.degrees(np.arccos(cos_theta)))
            d_h_opp = float(np.linalg.norm(coords[h] - opp_pos))
            if angle_deg < 85.0 and d_h_opp < 1.60:
                # Reflect H through the axis: new = parent + (h_vec - 2*<h_vec,axis>·axis_hat)
                # That negates the axis-parallel component, flipping the H to the
                # opposite side of the parent–opposite line.
                proj = float(np.dot(h_vec, axis_hat))
                reflected = h_vec - 2.0 * proj * axis_hat
                coords[h] = parent_pos + reflected
                changed = True
    if not changed:
        return xyz
    return _xyz_with_coords(xyz, coords)


# ---------------------------------------------------------------------------
# Helper 3 — migrating-H local cleanup (donor–acceptor triangulation)
# ---------------------------------------------------------------------------


def clean_migrating_h(xyz: dict,
                       mol: 'Molecule',
                       donor: int,
                       acceptor: int,
                       h_idx: int,
                       weight: float = 0.5,
                       ) -> dict:
    """Re-place a migrating H at the triangulated TS position between
    ``donor`` and ``acceptor``.

    This is the standalone version of the triangulation logic that
    :func:`migrate_verified_atoms` uses internally.  It is *idempotent*:
    calling it twice on the same input produces the same result, because
    the new H position depends only on (donor_pos, acceptor_pos,
    sbl(D,H), sbl(A,H)), not on the H's own current location.

    Args:
        xyz: TS guess XYZ dict.
        mol: RMG Molecule (used only for symbol lookup).
        donor: Donor heavy-atom index.
        acceptor: Acceptor heavy-atom index.
        h_idx: Migrating H atom index.
        weight: Currently unused; reserved for future weight-aware
            blending.  Phase 3b places the H at the triangulated point.

    Returns:
        A new XYZ dict with the migrating H re-placed.
    """
    if xyz is None:
        return xyz
    coords = np.asarray(xyz['coords'], dtype=float).copy()
    symbols = xyz['symbols']
    if h_idx >= len(coords) or symbols[h_idx] != 'H':
        return xyz
    if donor >= len(coords) or acceptor >= len(coords):
        return xyz

    d_pos = coords[donor]
    a_pos = coords[acceptor]
    da_vec = a_pos - d_pos
    da_dist = float(np.linalg.norm(da_vec))
    if da_dist < 1e-6:
        return xyz
    da_hat = da_vec / da_dist

    sbl_dh = get_single_bond_length(symbols[donor], 'H')
    sbl_ah = get_single_bond_length(symbols[acceptor], 'H')
    if sbl_dh is None or sbl_ah is None:
        return xyz
    d_DH = float(sbl_dh) + PAULING_DELTA
    d_AH = float(sbl_ah) + PAULING_DELTA

    h_pos = coords[h_idx]
    if da_dist <= d_DH + d_AH:
        # Triangulate.
        x = (da_dist ** 2 + d_DH ** 2 - d_AH ** 2) / (2.0 * da_dist)
        h_sq = d_DH ** 2 - x ** 2
        h_perp = float(np.sqrt(max(h_sq, 0.0)))
        proj = d_pos + np.dot(h_pos - d_pos, da_hat) * da_hat
        perp = h_pos - proj
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm > 1e-8:
            n_perp = perp / perp_norm
        else:
            arb = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(da_hat, arb))) > 0.9:
                arb = np.array([0.0, 1.0, 0.0])
            n_perp = np.cross(da_hat, arb)
            nrm = float(np.linalg.norm(n_perp))
            if nrm < 1e-8:
                return xyz
            n_perp = n_perp / nrm
        ideal = d_pos + da_hat * x + n_perp * h_perp
    else:
        # Spheres do not overlap — place H on the D–A axis at d_DH from donor.
        ideal = d_pos + da_hat * d_DH

    coords[h_idx] = ideal
    return _xyz_with_coords(xyz, coords)


# ---------------------------------------------------------------------------
# Helper 4 — local CH2/CH3 H symmetry restoration
# ---------------------------------------------------------------------------


def restore_terminal_h_symmetry(xyz: dict,
                                  mol: 'Molecule',
                                  center: int,
                                  exclude_atoms: Optional[Set[int]] = None,
                                  ) -> dict:
    """For a clearly-CH₂ or CH₃ heavy ``center``, average each H about
    the (center, parent) axis so equivalent H atoms land symmetrically.

    The "clear" condition is: ``center`` has exactly one heavy neighbor
    in the bond graph and exactly two or three H neighbors in the bond
    graph (after removing ``exclude_atoms``).  When that holds, the H
    atoms are rotated about the (parent → center) axis so that they sit
    at evenly spaced azimuth around the axis while keeping their current
    radial distances and along-axis offsets.

    The helper is conservative:

    * if the center is not a clear CH₂/CH₃ → returns ``xyz`` unchanged
    * if any per-H bond length would change by more than 0.05 Å → bails
      and returns ``xyz`` unchanged (to avoid introducing crowding)
    * never moves any non-H atom

    Args:
        xyz: TS guess XYZ dict.
        mol: RMG Molecule.
        center: Heavy-atom index whose terminal CH₂/CH₃ to symmetrize.
        exclude_atoms: Optional H indices to skip (e.g. migrating H).

    Returns:
        New XYZ dict; on bail-out the original input is returned.
    """
    if xyz is None:
        return xyz
    coords = np.asarray(xyz['coords'], dtype=float).copy()
    symbols = xyz['symbols']
    if center >= len(coords) or symbols[center] == 'H':
        return xyz
    exclude = exclude_atoms or set()
    heavy_nbrs = _heavy_neighbors(mol, center, symbols)
    h_nbrs = [h for h in _h_neighbors(mol, center, symbols) if h not in exclude]
    if len(heavy_nbrs) != 1:
        return xyz
    if len(h_nbrs) not in (2, 3):
        return xyz

    parent = heavy_nbrs[0]
    parent_pos = coords[parent]
    center_pos = coords[center]
    axis = center_pos - parent_pos
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        return xyz
    axis_hat = axis / axis_norm

    # Compute each H's along-axis offset and radial distance / azimuth.
    along: List[float] = []
    radial: List[float] = []
    for h in h_nbrs:
        rel = coords[h] - center_pos
        a = float(np.dot(rel, axis_hat))
        perp = rel - a * axis_hat
        r = float(np.linalg.norm(perp))
        along.append(a)
        radial.append(r)

    # Build an orthonormal basis (u, v) perpendicular to axis_hat.
    arb = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis_hat, arb))) > 0.9:
        arb = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis_hat, arb)
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1e-8:
        return xyz
    u = u / u_norm
    v = np.cross(axis_hat, u)

    avg_along = float(np.mean(along))
    avg_radial = float(np.mean(radial))
    n_h = len(h_nbrs)

    new_positions: List[np.ndarray] = []
    for k in range(n_h):
        theta = 2.0 * np.pi * k / n_h
        new_h = center_pos + avg_along * axis_hat \
            + avg_radial * (np.cos(theta) * u + np.sin(theta) * v)
        new_positions.append(new_h)

    # Bail out if any H would move more than 0.05 Å in bond length from
    # its previous distance to the center (avoid introducing crowding).
    for h, new_h in zip(h_nbrs, new_positions):
        old_d = float(np.linalg.norm(coords[h] - center_pos))
        new_d = float(np.linalg.norm(new_h - center_pos))
        if abs(new_d - old_d) > 0.05:
            return xyz

    for h, new_h in zip(h_nbrs, new_positions):
        coords[h] = new_h

    return _xyz_with_coords(xyz, coords)


# ---------------------------------------------------------------------------
# Helper 5 — identify per-H migration pairs
# ---------------------------------------------------------------------------


def identify_h_migration_pairs(xyz: dict,
                                 mol: 'Molecule',
                                 migrating_atoms: Set[int],
                                 core: Set[int],
                                 large_prod_atoms: Set[int],
                                 cross_bonds: Optional[List[Tuple[int, int]]] = None,
                                 ) -> List[Dict]:
    """Determine the (donor, acceptor) heavy atoms for each migrating H.

    This mirrors the donor/acceptor logic inside
    :func:`arc.job.adapters.ts.linear_utils.addition.migrate_verified_atoms`,
    but exposes it as a standalone function so the orchestrator
    (``interpolate_addition``) can recover the migration topology
    *without* re-running the migration.  The output is a deterministic
    list of dicts so callers can both feed local-cleanup helpers and
    drive Phase 3b topology gates from the same source.

    Args:
        xyz: TS guess XYZ dict (used for the nearest-core fallback only).
        mol: RMG Molecule of the unimolecular species.
        migrating_atoms: Atom indices that are migrating.
        core: Atom indices in the small product's core.
        large_prod_atoms: Atom indices in the large product.
        cross_bonds: Forming bonds (verified, from the template).

    Returns:
        A deterministic list ordered by ``h_idx``.  Each entry is::

            {
                'h_idx': int,
                'donor': int,
                'acceptor': int,
                'source': 'cross_bond' | 'nearest_core',
            }

        Migrating atoms with no identifiable donor are dropped.
    """
    if xyz is None or not migrating_atoms:
        return []
    coords = np.asarray(xyz['coords'], dtype=float)
    symbols = xyz['symbols']
    atom_to_idx = {atom: i for i, atom in enumerate(mol.atoms)}

    cross_acceptor: Dict[int, int] = {}
    for a, b in (cross_bonds or []):
        if a in migrating_atoms and b in core:
            cross_acceptor[a] = b
        elif b in migrating_atoms and a in core:
            cross_acceptor[b] = a

    core_heavy = sorted(idx for idx in core if symbols[idx] != 'H')

    out: List[Dict] = []
    for h_idx in sorted(migrating_atoms):
        if h_idx >= len(coords):
            continue
        # Donor: heavy neighbor of h_idx in large_prod_atoms.
        donor: Optional[int] = None
        for nbr in mol.atoms[h_idx].bonds.keys():
            ni = atom_to_idx[nbr]
            if symbols[ni] != 'H' and ni in large_prod_atoms:
                donor = ni
                break
        if donor is None:
            continue
        # Acceptor: cross-bond partner if available, else nearest core heavy atom.
        acceptor = cross_acceptor.get(h_idx)
        source = 'cross_bond'
        if acceptor is None:
            if not core_heavy:
                continue
            dists = np.linalg.norm(
                coords[core_heavy] - coords[h_idx], axis=1)
            acceptor = core_heavy[int(dists.argmin())]
            source = 'nearest_core'
        out.append({
            'h_idx': int(h_idx),
            'donor': int(donor),
            'acceptor': int(acceptor),
            'source': source,
        })
    return out


# ---------------------------------------------------------------------------
# Phase 3c — fragmentation-fallback single-H migration inference
# ---------------------------------------------------------------------------


def _split_into_fragments(uni_mol: 'Molecule',
                            split_bonds: Sequence[Tuple[int, int]],
                            ) -> List[Set[int]]:
    """Return the connected components of ``uni_mol`` after removing
    every bond in ``split_bonds``.

    Used by :func:`infer_frag_fallback_h_migration` to assign donor and
    acceptor atoms to opposite fragments.  Operates purely on the
    reactant-side molecular graph; never touches geometry.
    """
    n = len(uni_mol.atoms)
    atom_to_idx = {atom: i for i, atom in enumerate(uni_mol.atoms)}
    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for atom in uni_mol.atoms:
        ia = atom_to_idx[atom]
        for nbr in atom.bonds.keys():
            adj[ia].add(atom_to_idx[nbr])
    cuts = {(min(int(a), int(b)), max(int(a), int(b)))
            for (a, b) in (split_bonds or [])}
    for a, b in cuts:
        adj[a].discard(b)
        adj[b].discard(a)

    visited: Set[int] = set()
    fragments: List[Set[int]] = []
    for start in range(n):
        if start in visited:
            continue
        comp: Set[int] = set()
        q: deque = deque([start])
        while q:
            node = q.popleft()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            for nbr in adj[node]:
                if nbr not in visited:
                    q.append(nbr)
        fragments.append(comp)
    return fragments


def _heavy_formula_of_fragment(symbols: Sequence[str],
                                 fragment: Set[int]) -> Tuple[Tuple[str, int], ...]:
    """Return a hashable heavy-atom composition for *fragment*."""
    counts: Dict[str, int] = {}
    for idx in fragment:
        sym = symbols[idx]
        if sym == 'H':
            continue
        counts[sym] = counts.get(sym, 0) + 1
    return tuple(sorted(counts.items()))


def _h_count_of_fragment(symbols: Sequence[str], fragment: Set[int]) -> int:
    return sum(1 for idx in fragment if symbols[idx] == 'H')


def infer_frag_fallback_h_migration(pre_xyz: dict,
                                     post_xyz: dict,
                                     uni_mol: 'Molecule',
                                     split_bonds: Sequence[Tuple[int, int]],
                                     multi_species: Optional[Sequence['ARCSpecies']],
                                     label: str = '',
                                     displacement_threshold: float = 0.05,
                                     ) -> Optional[Dict]:
    """Phase 3c: deterministically infer the single-H migration triple
    for a fragmentation-fallback addition guess.

    The helper combines five strict signals (S1–S5) and returns either
    one trustworthy ``(h_idx, donor, acceptor, source)`` record or
    ``None`` whenever any ambiguity remains.  It NEVER returns multiple
    candidates and NEVER partially enriches.

    The contract:

    * **S1 — pre/post displacement**: exactly one H atom must have moved
      by more than ``displacement_threshold`` Å between ``pre_xyz`` and
      ``post_xyz``.  Multi-H or zero-H displacement → ``None``.
    * **S2 — reactant-graph adjacency**: the displaced H must have
      *exactly one* heavy neighbor in the reactant graph (the donor).
      Two heavy neighbors of the same H → ``None``.
    * **S3 — split-bond fragment membership**: after cutting
      ``split_bonds``, the donor and the acceptor must lie in
      *different* connected components.  Same-fragment placement is
      chemically incompatible with fragment-to-fragment H transfer →
      ``None``.
    * **S4 — product-composition consistency** (when ``multi_species``
      is supplied): the donor's fragment must show an H surplus
      relative to its matched product, and the acceptor's fragment an
      H deficit.  This mirrors the same logic
      :func:`migrate_h_between_fragments` itself uses.  Skipped when
      ``multi_species`` is ``None`` or fragment-to-product matching is
      ambiguous.
    * **S5 — local donor/acceptor geometry consistency**: in the
      post-migration geometry, the migrated H must satisfy
      ``d(D,H) ≤ 1.6 × sbl(D,H)`` AND ``d(A,H) ≤ 1.6 × sbl(A,H)``
      AND no other heavy atom in the acceptor's fragment can be closer
      to the H than ``0.95 × sbl(rival,H)``.

    Args:
        pre_xyz: Pre-migration TS XYZ dict (output of ``stretch_bond``).
        post_xyz: Post-migration TS XYZ dict (output of
            ``migrate_h_between_fragments``).
        uni_mol: Unimolecular reactant Molecule (defines the bond graph).
        split_bonds: The fragmentation cut.
        multi_species: Sequence of product :class:`ARCSpecies` for the
            optional composition consistency check (S4).  May be
            ``None`` to skip S4.
        label: Optional logging label.
        displacement_threshold: Å threshold for "moved" in S1.

    Returns:
        A single migration record dict
        ``{'h_idx': int, 'donor': int, 'acceptor': int, 'source': 'frag_inferred'}``
        on full success, otherwise ``None``.
    """
    if pre_xyz is None or post_xyz is None or uni_mol is None:
        return None
    pre_arr = np.asarray(pre_xyz['coords'], dtype=float)
    post_arr = np.asarray(post_xyz['coords'], dtype=float)
    if pre_arr.shape != post_arr.shape:
        return None
    symbols = post_xyz['symbols']
    n_atoms = len(symbols)

    # ---- S1: exactly one H moved ----
    moved_h: List[int] = []
    for h in range(n_atoms):
        if symbols[h] != 'H':
            continue
        if float(np.linalg.norm(post_arr[h] - pre_arr[h])) > displacement_threshold:
            moved_h.append(h)
    if len(moved_h) != 1:
        if logger is not None:
            logger.debug(f'Linear addition ({label}): frag-fallback inference '
                         f'skipped (S1: {len(moved_h)} H atoms moved).')
        return None
    h_idx = moved_h[0]

    # ---- S2: exactly one heavy neighbor of the migrated H ----
    atom_to_idx = {atom: i for i, atom in enumerate(uni_mol.atoms)}
    heavy_nbrs: List[int] = []
    for nbr in uni_mol.atoms[h_idx].bonds.keys():
        ni = atom_to_idx[nbr]
        if symbols[ni] != 'H':
            heavy_nbrs.append(ni)
    if len(heavy_nbrs) != 1:
        if logger is not None:
            logger.debug(f'Linear addition ({label}): frag-fallback inference '
                         f'skipped (S2: H{h_idx} has {len(heavy_nbrs)} '
                         f'heavy neighbors).')
        return None
    donor = heavy_nbrs[0]

    # ---- S3: donor and acceptor on different fragments after the cut ----
    fragments = _split_into_fragments(uni_mol, split_bonds)
    if len(fragments) < 2:
        if logger is not None:
            logger.debug(f'Linear addition ({label}): frag-fallback inference '
                         f'skipped (S3: only {len(fragments)} fragment(s) '
                         f'after cut).')
        return None

    donor_frag_idx: Optional[int] = None
    for fi, frag in enumerate(fragments):
        if donor in frag:
            donor_frag_idx = fi
            break
    if donor_frag_idx is None:
        if logger is not None:
            logger.debug(f'Linear addition ({label}): frag-fallback inference '
                         f'skipped (S3: donor {donor} not in any fragment).')
        return None
    other_frag_atoms: Set[int] = set()
    for fi, frag in enumerate(fragments):
        if fi != donor_frag_idx:
            other_frag_atoms.update(frag)
    if not other_frag_atoms:
        return None

    # The acceptor candidate set: heavy atoms in the *other* fragments.
    acceptor_candidates = [i for i in other_frag_atoms if symbols[i] != 'H']
    if not acceptor_candidates:
        if logger is not None:
            logger.debug(f'Linear addition ({label}): frag-fallback inference '
                         f'skipped (S3: no heavy atom in any non-donor fragment).')
        return None

    # ---- S4: composition consistency (donor fragment has H surplus,
    #          acceptor fragment has H deficit), when products available.
    acceptor_frag_idx: Optional[int] = None
    if multi_species:
        try:
            target_h: List[int] = []
            target_heavy: List[Tuple[Tuple[str, int], ...]] = []
            for sp in multi_species:
                sp_symbols = sp.get_xyz()['symbols']
                target_h.append(sum(1 for s in sp_symbols if s == 'H'))
                heavy_counts: Dict[str, int] = {}
                for s in sp_symbols:
                    if s == 'H':
                        continue
                    heavy_counts[s] = heavy_counts.get(s, 0) + 1
                target_heavy.append(tuple(sorted(heavy_counts.items())))
        except Exception:  # pragma: no cover - defensive
            target_h = []
            target_heavy = []

        if target_heavy and len(target_heavy) == len(fragments):
            # Match each fragment to a target by unique heavy formula.
            frag_heavy = [_heavy_formula_of_fragment(symbols, f) for f in fragments]
            # Build a deterministic 1-to-1 matching: each target consumed once.
            frag_to_target: Dict[int, int] = {}
            used_targets: Set[int] = set()
            ok_match = True
            for fi, fh in enumerate(frag_heavy):
                match = None
                for ti, th in enumerate(target_heavy):
                    if ti in used_targets:
                        continue
                    if th == fh:
                        match = ti
                        break
                if match is None:
                    ok_match = False
                    break
                frag_to_target[fi] = match
                used_targets.add(match)

            if ok_match:
                # Compute H surplus/deficit per fragment.
                surplus: Dict[int, int] = {}
                for fi, ti in frag_to_target.items():
                    diff = _h_count_of_fragment(symbols, fragments[fi]) - target_h[ti]
                    surplus[fi] = diff
                # Donor's fragment must have surplus > 0; acceptor's must have deficit < 0.
                if surplus.get(donor_frag_idx, 0) <= 0:
                    if logger is not None:
                        logger.debug(
                            f'Linear addition ({label}): frag-fallback '
                            f'inference skipped (S4: donor fragment surplus '
                            f'{surplus.get(donor_frag_idx, 0)} ≤ 0).')
                    return None
                deficit_frags = [fi for fi, d in surplus.items() if d < 0]
                if len(deficit_frags) != 1:
                    if logger is not None:
                        logger.debug(
                            f'Linear addition ({label}): frag-fallback '
                            f'inference skipped (S4: {len(deficit_frags)} '
                            f'deficit fragments).')
                    return None
                acceptor_frag_idx = deficit_frags[0]

    # If S4 narrowed the acceptor fragment, restrict candidates to it.
    if acceptor_frag_idx is not None:
        acceptor_candidates = [i for i in fragments[acceptor_frag_idx]
                                if symbols[i] != 'H']
        if not acceptor_candidates:
            if logger is not None:
                logger.debug(f'Linear addition ({label}): frag-fallback '
                             f'inference skipped (S4: deficit fragment has '
                             f'no heavy atoms).')
            return None

    # ---- S5: local donor/acceptor geometry consistency ----
    sbl_dh = get_single_bond_length(symbols[donor], 'H')
    if not sbl_dh:
        return None
    sbl_dh = float(sbl_dh)
    d_dh = float(np.linalg.norm(post_arr[donor] - post_arr[h_idx]))
    if d_dh > 1.60 * sbl_dh:
        if logger is not None:
            logger.debug(f'Linear addition ({label}): frag-fallback inference '
                         f'skipped (S5: d(D,H)={d_dh:.3f} > {1.60 * sbl_dh:.3f}).')
        return None

    # The acceptor is the heavy atom in the acceptor candidate set that
    # is closest to the migrated H — and uniquely so within its fragment.
    cand_arr = np.asarray([post_arr[i] for i in acceptor_candidates], dtype=float)
    h_pos = post_arr[h_idx]
    cand_dists = np.linalg.norm(cand_arr - h_pos, axis=1)
    nearest_idx = int(cand_dists.argmin())
    acceptor = int(acceptor_candidates[nearest_idx])
    nearest_d = float(cand_dists[nearest_idx])

    sbl_ah = get_single_bond_length(symbols[acceptor], 'H')
    if not sbl_ah:
        return None
    sbl_ah = float(sbl_ah)
    if nearest_d > 1.60 * sbl_ah:
        if logger is not None:
            logger.debug(f'Linear addition ({label}): frag-fallback inference '
                         f'skipped (S5: d(A,H)={nearest_d:.3f} > '
                         f'{1.60 * sbl_ah:.3f}).')
        return None

    # No rival heavy atom inside the acceptor's fragment may be too
    # close to the migrated H either.  This is the "uniquely within
    # its fragment" half of S5.
    if acceptor_frag_idx is not None:
        rival_atoms = [i for i in fragments[acceptor_frag_idx]
                       if symbols[i] != 'H' and i != acceptor]
    else:
        # No S4 narrowing — restrict the rival check to the same
        # fragment the chosen acceptor lives in.
        chosen_frag: Optional[Set[int]] = None
        for frag in fragments:
            if acceptor in frag:
                chosen_frag = frag
                break
        rival_atoms = ([i for i in chosen_frag
                        if symbols[i] != 'H' and i != acceptor]
                       if chosen_frag is not None else [])
    for rival in rival_atoms:
        sbl_rh = get_single_bond_length(symbols[rival], 'H')
        if not sbl_rh:
            continue
        d_rh = float(np.linalg.norm(post_arr[rival] - h_pos))
        if d_rh < 0.95 * float(sbl_rh):
            if logger is not None:
                logger.debug(f'Linear addition ({label}): frag-fallback '
                             f'inference skipped (S5: rival '
                             f'{symbols[rival]}{rival} at d={d_rh:.3f} < '
                             f'{0.95 * float(sbl_rh):.3f}).')
            return None

    return {
        'h_idx': int(h_idx),
        'donor': int(donor),
        'acceptor': int(acceptor),
        'source': 'frag_inferred',
    }
