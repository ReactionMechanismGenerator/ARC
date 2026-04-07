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
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from arc.common import get_single_bond_length
from arc.job.adapters.ts.linear_utils.postprocess import PAULING_DELTA

if TYPE_CHECKING:
    from arc.molecule import Molecule


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
