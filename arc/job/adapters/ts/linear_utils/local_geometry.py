"""
Local reactive-center geometry helpers for the linear TS-guess adapter.

This module owns the small, deterministic, *immediate-shell* utilities
that the addition and isomerization pipelines use to clean up post-
generation TS geometries before they reach the final-stage path-spec
wrapper. The helpers operate on a single named heavy center and its
direct H/heavy neighbors; they never scan the whole molecule.

Helper categories:

* **Bond-length regularizers**: :func:`regularize_terminal_h_geometry`
  snaps clearly over-stretched or compressed terminal H atoms back to
  ``sbl(C, H)``.
* **Spectator-H reorientation**: :func:`orient_h_away_from_axis`
  reflects a non-migrating H atom away from a donor → acceptor reaction
  axis when it currently sits in the blocking pocket.
* **Migrating-H triangulation**: :func:`clean_migrating_h` re-places a
  single migrating H at the donor-acceptor Pauling-triangulated TS
  position.
* **Terminal CH₂/CH₃ asymmetry detection and repair**:
  :func:`is_terminal_group_asymmetric` and
  :func:`restore_terminal_h_symmetry`. Detect umbrella inversion and
  azimuthal distortion, then repair while preserving each H's original
  bond length to the parent.
* **Internal reactive CH₂ detection and repair**:
  :func:`is_internal_reactive_ch2_misoriented` and
  :func:`repair_internal_reactive_ch2`. Detect heavy-corridor
  crowding, squeezing and sp³ plane violations on internal CH₂ shells,
  then repair while preserving each H's bond length.
* **Cleanup orchestrator** — :func:`apply_reactive_center_cleanup`
  composes all of the above around an explicit reactive shell
  (donors, acceptors, named reactive centers). It is the single
  composition entry point for the live cleanup sites in
  :func:`arc.job.adapters.ts.linear.interpolate_addition` and
  :func:`arc.job.adapters.ts.linear.interpolate_isomerization`.

Design invariants (every helper in this file follows them):

* Each helper only moves the immediate terminal group or an explicitly
  supplied atom subset.
* No global torsion scans, no whole-molecule rotation, no broad
  "fix-everything" passes.
* Bond lengths are preserved when nothing else compels a change; the
  terminal-group and internal-CH₂ repair primitives explicitly
  preserve each H's per-atom bond length to its parent.
* Helpers are deterministic — calling them twice on the same input
  produces identical output.
* Helpers preserve symmetry when symmetry is meaningful (CH₂/CH₃).
* Helpers do not introduce H crowding or new collisions.

The helpers are side-effect free: each one accepts an XYZ dict and
returns a *new* XYZ dict (never an in-place mutation).

The graph-aware migration *inference* helpers
(:func:`identify_h_migration_pairs`,
:func:`infer_frag_fallback_h_migration`) live in the sibling module
:mod:`arc.job.adapters.ts.linear_utils.migration_inference`.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from arc.common import get_logger, get_single_bond_length
from arc.job.adapters.ts.linear_utils.postprocess import PAULING_DELTA

if TYPE_CHECKING:
    from arc.molecule import Molecule


logger = get_logger()


# ---------------------------------------------------------------------------
# Small XYZ helpers
# ---------------------------------------------------------------------------

def _xyz_with_coords(xyz: dict, coords: np.ndarray) -> dict:
    """Return a new XYZ dict that copies *xyz* but replaces ``coords``."""
    return {'symbols': xyz['symbols'],
            'isotopes': xyz.get('isotopes', tuple(0 for _ in range(len(xyz['symbols'])))),
            'coords': tuple(tuple(float(x) for x in row) for row in coords)}


def _heavy_neighbors(mol: 'Molecule', atom_idx: int, symbols: Tuple[str, ...]) -> List[int]:
    """Return graph-bonded non-H neighbor indices of ``atom_idx``."""
    atom_to_idx = {atom: i for i, atom in enumerate(mol.atoms)}
    neighbors: List[int] = []
    for nbr in mol.atoms[atom_idx].bonds.keys():
        ni = atom_to_idx[nbr]
        if symbols[ni] != 'H':
            neighbors.append(ni)
    return neighbors


def _h_neighbors(mol: 'Molecule', atom_idx: int, symbols: Tuple[str, ...]) -> List[int]:
    """Return graph-bonded H neighbor indices of ``atom_idx``."""
    atom_to_idx = {atom: i for i, atom in enumerate(mol.atoms)}
    h_idxs: List[int] = []
    for nbr in mol.atoms[atom_idx].bonds.keys():
        ni = atom_to_idx[nbr]
        if symbols[ni] == 'H':
            h_idxs.append(ni)
    return h_idxs


def _sticky_pairing_arc_cost(src_dirs: Sequence[np.ndarray],
                             target_dirs: Sequence[np.ndarray],
                             pairing: Sequence[int],
                             ) -> float:
    """
    Compute the angular displacement cost for a source-to-target pairing.

    Each source unit vector is mapped to a target unit vector by
    ``pairing[src_k] = tgt_k``. The total cost is the sum of
    ``1 - cos(θ)`` over all pairs, with ``θ`` the angle between the
    source and target direction. Equivalent to the squared chord length
    on the unit sphere (up to a factor of two), so minimizing this cost
    yields the assignment that moves each H by the smallest arc.

    Args:
        src_dirs (Sequence[np.ndarray]): Source unit vectors.
        target_dirs (Sequence[np.ndarray]): Target unit vectors.
        pairing (Sequence[int]): For each source index ``src_k``, the index
            ``tgt_k = pairing[src_k]`` of the target it is mapped to.

    Returns:
        float: Total angular displacement cost (non-negative).
    """
    total = 0.0
    for src_k, tgt_k in enumerate(pairing):
        cos_t = float(np.dot(src_dirs[src_k], target_dirs[tgt_k]))
        cos_t = max(-1.0, min(1.0, cos_t))
        total += 1.0 - cos_t
    return total


# ---------------------------------------------------------------------------
# Helper 1 — terminal CH2/CH3 H bond-length regularization
# ---------------------------------------------------------------------------

def regularize_terminal_h_geometry(xyz: dict,
                                   mol: 'Molecule',
                                   center: int,
                                   exclude_atoms: Optional[Set[int]] = None,
                                   ) -> dict:
    """
    Snap terminal H bond lengths around ``center`` back to ~sbl.

    For each H atom bonded to ``center`` in the molecular graph (skipping
    any in ``exclude_atoms``), if its current distance from ``center``
    is more than ~30 % away from ``get_single_bond_length(C, H)``, the H
    is moved along the existing C–H direction so that the new distance
    equals the standard single-bond length. Direction is preserved.
    Atoms not in the immediate first shell of ``center`` are never
    touched.

    This is a *bond length* regularizer. It does not change angles or
    rotate H atoms.

    Args:
        xyz: TS guess XYZ dict.
        mol: RMG Molecule defining the bond graph.
        center: Heavy-atom index whose H neighbors should be regularized.
        exclude_atoms: Optional H indices to skip (e.g. the migrating H).

    Returns:
        New XYZ dict with the regularized H positions. When no H atom
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
    """
    Rotate non-migrating H atoms around their parent C–H bond
    until they no longer block the donor–acceptor reaction axis.

    For each non-migrating H atom whose parent (graph neighbor) is
    ``donor`` or ``acceptor`` and whose current geometry satisfies
    ``∠(H, parent, opposite) < 85°`` AND ``d(H, opposite) < 1.60 Å``,
    the H is reflected through the (parent, opposite) axis so it points
    outward. This is the same blocking condition the sub-check
    :func:`has_inward_blocking_h_on_forming_axis` rejects on.

    The reflection moves only one H atom per call iteration; nothing
    else. No molecule-wide rotation.

    Args:
        xyz: TS guess XYZ dict.
        mol: RMG Molecule.
        donor: Donor heavy-atom index.
        acceptor: Acceptor heavy-atom index.
        exclude_h: Optional set of H indices to skip (the migrating H).

    Returns:
        A new XYZ dict; if no H needed re-orientation, the original is returned unchanged.
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
    """
    Re-place a migrating H at the triangulated TS position between ``donor`` and ``acceptor``.

    This is the standalone version of the triangulation logic that
    :func:`migrate_verified_atoms` uses internally. It is *idempotent*:
    calling it twice on the same input produces the same result, because
    the new H position depends only on (donor_pos, acceptor_pos,
    sbl(D,H), sbl(A,H)), not on the H's own current location.

    Args:
        xyz: TS guess XYZ dict.
        mol: RMG Molecule (used only for symbol lookup).
        donor: Donor heavy-atom index.
        acceptor: Acceptor heavy-atom index.
        h_idx: Migrating H atom index.
        weight: Currently unused; reserved for future weight-aware blending. Places the H at the triangulated point.

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
        # Spheres do not overlap: place H on the D–A axis at d_DH from donor.
        ideal = d_pos + da_hat * d_DH

    coords[h_idx] = ideal
    return _xyz_with_coords(xyz, coords)


# ---------------------------------------------------------------------------
# Helper 4 — local CH2/CH3 H symmetry restoration
# ---------------------------------------------------------------------------

def is_terminal_group_asymmetric(xyz: dict,
                                 center_idx: int,
                                 parent_idx: int,
                                 h_indices: Sequence[int],
                                 threshold_deg: float = 20.0,
                                 ) -> bool:
    """
    Pure detector for unphysically distorted terminal CH₂/CH₃ groups.

    This is the safety gate for symmetry restoration: it answers
    "should the symmetrizer touch this terminal group?" without ever
    moving an atom. The detector returns ``True`` only when the H
    geometry around ``center_idx`` is *clearly* distorted — either an
    inverted umbrella (one or more H atoms pointing back toward the
    parent atom) or strongly non-uniform azimuthal spacing around the
    parent → center axis. All other cases — including normal,
    near-tetrahedral terminal groups generated by force-field cleanup —
    return ``False`` so the symmetrizer leaves them byte-for-byte unchanged.

    Terminal eligibility:
        Only evaluates the math when the caller has supplied a
        plausible terminal CH₂/CH₃ context. Specifically, the function
        returns ``False`` (i.e. "do nothing") if any of:

        * ``len(h_indices)`` is not 2 or 3,
        * ``center_idx`` or ``parent_idx`` is out of range,
        * ``parent_idx == center_idx``,
        * the parent → center axis vanishes (degenerate geometry).

        The orchestrator wiring also enforces these conditions before
        calling this helper, so the in-helper checks are a defensive
        second line.

    Algorithm:

    1. **Umbrella inversion check.** Define
       ``axis = normalize(coords[center] - coords[parent])`` (parent →
       center, pointing outward). For each H, define
       ``h_dir = normalize(coords[H] - coords[center])`` and compute
       ``angle(axis, h_dir)``. A normal outward CH₃ H sits at
       ~70° from this axis (180° − 109.5°, the supplement of the
       tetrahedral H–C–C bond angle). When *any* H angle exceeds
       100°, the H is pointing back into the parent's hemisphere — the
       "inverted umbrella" pathology — and the function returns
       ``True`` immediately.

    2. **Azimuthal distortion check.** Build a deterministic
       orthonormal basis ``(u, v)`` in the plane orthogonal to ``axis``.
       The basis is built from a fixed reference vector, so two
       evaluations of the same input always agree.

       Project each H vector onto the (u, v) plane. If any projection
       has near-zero magnitude (the H is exactly on the axis), the
       group is collapsed and the function returns ``True``.

       Otherwise compute signed azimuth angles ``theta_i ∈ [0, 2π)`` for
       the projections, sort them, and compute the cyclic spacings
       around the axis.

       For **CH₃** (3 H atoms): the ideal cyclic spacing is exactly
       120°. When the maximum deviation from 120° exceeds
       ``threshold_deg`` (default 20°), the group is significantly
       non-uniform and the function returns ``True``.

       For **CH₂** (2 H atoms): compute the smaller of the two cyclic
       separations between the projections. When that smaller
       separation drops below 80°, the two H atoms are squeezed
       together and the function returns ``True``.

    3. Otherwise return ``False``.

    Args:
        xyz: TS guess XYZ dict — used as a *read-only* coordinate source.
        center_idx: Heavy-atom index of the terminal CH₂/CH₃ carbon (or
            other heavy atom in the same role).
        parent_idx: Index of the single heavy neighbor of ``center_idx``.
            The asymmetry math is referenced against the
            (parent → center) axis.
        h_indices: Sequence of 2 or 3 H atom indices that are bonded to
            ``center_idx`` in the molecular graph. Pass only the H
            atoms that should be considered (the orchestrator filters
            out migrating Hs before calling this).
        threshold_deg: Maximum allowed CH₃ azimuthal deviation from the
            ideal 120° spacing. Default 20°. Tighter thresholds make
            the detector more eager; looser thresholds make it more
            permissive.

    Returns:
        ``True`` if the terminal group is unphysically distorted, ``False`` otherwise (including all
        "not eligible" / degenerate cases — the orchestrator interprets ``False`` as "leave the group alone").
    """
    if xyz is None:
        return False
    n_h = len(h_indices)
    if n_h not in (2, 3):
        return False
    coords = np.asarray(xyz['coords'], dtype=float)
    n_atoms = len(coords)
    if not (0 <= center_idx < n_atoms and 0 <= parent_idx < n_atoms):
        return False
    if center_idx == parent_idx:
        return False
    for h in h_indices:
        if not (0 <= int(h) < n_atoms):
            return False

    center_pos = coords[center_idx]
    parent_pos = coords[parent_idx]
    axis_vec = center_pos - parent_pos
    axis_norm = float(np.linalg.norm(axis_vec))
    if axis_norm < 1e-8:
        return False
    axis = axis_vec / axis_norm

    # ---- 1. Umbrella inversion check ----
    for h in h_indices:
        h_vec = coords[int(h)] - center_pos
        h_norm = float(np.linalg.norm(h_vec))
        if h_norm < 1e-8:
            # H sits exactly on the center: degenerate; treat as asymmetric.
            return True
        cos_theta = float(np.dot(h_vec, axis) / h_norm)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_deg = float(np.degrees(np.arccos(cos_theta)))
        if angle_deg > 100.0:
            return True

    # ---- 2. Azimuthal distortion check ----
    arb = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, arb))) > 0.9:
        arb = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, arb)
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1e-8:
        return False
    u = u / u_norm
    v = np.cross(axis, u)

    azimuths: List[float] = []
    for h in h_indices:
        rel = coords[int(h)] - center_pos
        proj = rel - float(np.dot(rel, axis)) * axis
        proj_norm = float(np.linalg.norm(proj))
        if proj_norm < 1e-8:
            # H collinear with axis: group collapsed in the perpendicular plane.
            return True
        theta = float(np.arctan2(float(np.dot(proj, v)),
                                 float(np.dot(proj, u))))
        if theta < 0.0:
            theta += 2.0 * np.pi
        azimuths.append(theta)

    azimuths.sort()
    two_pi = 2.0 * np.pi
    spacings: List[float] = []
    for k in range(n_h):
        nxt = azimuths[(k + 1) % n_h]
        cur = azimuths[k]
        diff = nxt - cur if (k + 1) < n_h else (two_pi - cur + azimuths[0])
        spacings.append(float(diff))

    if n_h == 3:
        ideal = two_pi / 3.0  # 120° in radians
        max_dev_deg = float(np.degrees(max(abs(s - ideal) for s in spacings)))
        if max_dev_deg > threshold_deg:
            return True
    else:
        # n_h == 2: smaller cyclic separation is always min(spacings).
        smaller_deg = float(np.degrees(min(spacings)))
        if smaller_deg < 80.0:
            return True

    return False


def restore_terminal_h_symmetry(xyz: dict,
                                mol: 'Molecule',
                                center: int,
                                exclude_atoms: Optional[Set[int]] = None,
                                ) -> dict:
    """
    For a clearly-CH₂ or CH₃ heavy ``center``, re-seat the H atoms
    around the (parent → center) axis at evenly spaced azimuth while
    **preserving each H's original distance to the center**.

    Handles true terminal CH₃ umbrella-inversion cases by placing each
    H *individually* on the outward tetrahedral cone:

    1. The terminal eligibility check is unchanged: the center must
       have exactly one heavy neighbor (the parent) and exactly 2 or 3
       H neighbors after exempting ``exclude_atoms``.
    2. For each H atom, compute three per-H quantities from the
       *current* geometry:
         * ``r_i``       = current ``|H_i - center|`` — the bond
           length to preserve.
         * ``θ_i``       = current azimuth in the (u, v) plane via
           ``atan2(proj·v, proj·u)``, normalized to ``[0, 2π)`` — used
           for slot assignment.
         * ``fold_α_i`` = ``min(α_i, π − α_i)`` where ``α_i`` is the
           angle between the outward axis and ``H_i − center`` — this
           "folds" inverted H atoms onto their would-be outward cone
           angle.
    3. ``target_cone = mean(fold_α_i)``. This preserves the original
       group's cone chemistry (sp³ → ~70.5°, sp² → ~90°) while
       ignoring inversion sign.
    4. Sort H atoms by ``θ_i`` (deterministic, basis-fixed).
    5. Pin ``offset = θ_0`` (the first sorted H's current azimuth)
       so that the minimum-displacement target arrangement keeps the
       first H at its current azimuth — already-azimuth-symmetric
       groups end up at the same coordinates as the input.
    6. For each H k in sorted order, place it at::

           target_θ_k = offset + 2π·k / n_h
           target_dir = sin(target_cone) · (cos(target_θ_k)·u
                                            + sin(target_θ_k)·v)
                        + cos(target_cone) · axis
           new_pos    = center + r_i · target_dir

       Because ``target_dir`` is a unit vector, ``|new_pos − center| = r_i``
       exactly — every H's bond length is preserved by construction.

    The repair stays strictly local: the parent atom and any atoms
    not in the H neighbor list are read-only. Non-H atoms are never
    moved. When the eligibility check fails (non-terminal center,
    degenerate axis, missing basis, ``r_i = 0``) the function returns
    the original ``xyz`` unchanged.

    Args:
        xyz: TS guess XYZ dict.
        mol: RMG Molecule.
        center: Heavy-atom index whose terminal CH₂/CH₃ to symmetrize.
        exclude_atoms: Optional H indices to skip (e.g. migrating H).

    Returns:
        A new XYZ dict with the H atoms at their target azimuth slots
        and preserved bond lengths. On any eligibility failure the
        original input is returned unchanged.
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
    axis_vec = center_pos - parent_pos
    axis_norm = float(np.linalg.norm(axis_vec))
    if axis_norm < 1e-6:
        return xyz
    axis_hat = axis_vec / axis_norm

    arb = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis_hat, arb))) > 0.9:
        arb = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis_hat, arb)
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1e-8:
        return xyz
    u = u / u_norm
    v = np.cross(axis_hat, u)

    h_data: List[Tuple[int, float, float, float]] = []
    for h in h_nbrs:
        rel = coords[h] - center_pos
        r_i = float(np.linalg.norm(rel))
        if r_i < 1e-8:
            return xyz
        cos_alpha = float(np.dot(rel, axis_hat) / r_i)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        alpha_i = float(np.arccos(cos_alpha))
        # Fold inverted H angles back to their outward-cone equivalent
        # so target_cone ignores inversion sign.
        fold_alpha = min(alpha_i, float(np.pi) - alpha_i)
        proj_u = float(np.dot(rel, u))
        proj_v = float(np.dot(rel, v))
        proj_norm = float(np.sqrt(proj_u * proj_u + proj_v * proj_v))
        if proj_norm < 1e-8:
            # H collinear with axis: deterministic fallback azimuth.
            theta_i = 0.0
        else:
            theta_i = float(np.arctan2(proj_v, proj_u))
            if theta_i < 0.0:
                theta_i += 2.0 * float(np.pi)
        h_data.append((int(h), r_i, float(theta_i), float(fold_alpha)))

    target_cone = float(np.mean([d[3] for d in h_data]))
    sin_c = float(np.sin(target_cone))
    cos_c = float(np.cos(target_cone))

    h_data.sort(key=lambda d: d[2])
    n_h = len(h_data)

    # Pin first sorted H at its current azimuth (minimum-displacement choice).
    offset = h_data[0][2]

    new_positions: List[Tuple[int, np.ndarray]] = []
    for k, (h_idx, r_i, _theta, _fold) in enumerate(h_data):
        target_azimuth = offset + 2.0 * float(np.pi) * k / n_h
        cos_az = float(np.cos(target_azimuth))
        sin_az = float(np.sin(target_azimuth))
        target_dir = sin_c * (cos_az * u + sin_az * v) + cos_c * axis_hat
        new_pos = center_pos + r_i * target_dir
        new_positions.append((h_idx, new_pos))

    for h_idx, new_pos in new_positions:
        coords[h_idx] = new_pos

    return _xyz_with_coords(xyz, coords)


# ---------------------------------------------------------------------------
# internal reactive CH₂ misorientation detection and repair
# ---------------------------------------------------------------------------

def is_internal_reactive_ch2_misoriented(xyz: dict,
                                         center_idx: int,
                                         heavy_nbr_indices: Sequence[int],
                                         h_indices: Sequence[int],
                                         ) -> bool:
    """
    Pure detector for misoriented internal reactive CH₂ centers.

    This is the safety gate for the internal-CH₂ repair primitive.
    It answers "should the internal-CH₂ repair touch this center?" without
    ever moving an atom. The detector is intentionally local: it only
    inspects the heavy-neighbor frame around ``center_idx`` and the two
    bonded H atoms, never global hybridization or distant atoms.

    Eligibility:
        Returns ``False`` immediately unless

        * ``center_idx`` is in range,
        * ``heavy_nbr_indices`` has exactly two distinct entries (this is
          an *internal* center, not a terminal one — terminal CH₂/CH₃ go
          through the terminal-group orchestrator pathway instead),
        * ``h_indices`` has exactly two distinct entries (a CH₂ shell),
        * all referenced indices are in range and refer to distinct atoms,
        * the local heavy-neighbor frame is non-degenerate
          (``|v_a|`` and ``|v_b|`` are non-zero, the bisector
          ``v̂_a + v̂_b`` is non-zero, and the cross-product
          ``v_a × v_b`` is non-zero — i.e. the two heavy neighbors are
          not collinear with the center).

    Local heavy-neighbor frame:
        With ``c = coords[center_idx]``, ``a = coords[heavy_nbr_a]``,
        ``b = coords[heavy_nbr_b]``, define

            v_a = a − c
            v_b = b − c
            v̂_a = v_a / |v_a|
            v̂_b = v_b / |v_b|
            ŵ   = (v̂_a + v̂_b) / |v̂_a + v̂_b|     (heavy-corridor bisector,
                                                   pointing INTO the heavy
                                                   side of the center)
            n̂   = (v_a × v_b) / |v_a × v_b|       (out-of-plane normal,
                                                   perpendicular to both
                                                   heavy bonds)

        For a healthy sp³ CH₂, the two H atoms sit roughly opposite each
        other across the heavy plane. Each H should have

        * ``proj_w_i = (h_i − c) · ŵ`` clearly *negative* (the H sits
          *behind* the heavy corridor, not crowding it), and
        * ``proj_n_i = (h_i − c) · n̂`` non-trivially non-zero with
          *opposite signs* between the two H atoms (one above, one below
          the heavy plane).

         flags violations of these expectations.

    Detection rules — return ``True`` as soon as any rule fires:

    1. **Squeezed H–C–H angle.** If the angle between ``(h_0 − c)`` and
       ``(h_1 − c)`` falls below ``80°`` (well below the ideal sp³
       ``109.47°``), the two H atoms are squeezed together and the CH₂
       shell is unphysical.

    2. **Heavy-corridor crowding.** If both H atoms have
       ``proj_w_i / |h_i − c| > 0.30``, both H atoms are pointing
       substantially into the same forbidden side (the heavy corridor)
       instead of forming the back-side pair the local frame demands.

    3. **sp³ plane violation.** If both H atoms have
       ``|proj_n_i / |h_i − c|| > 0.15`` *and* the projections share the
       same sign, both H atoms lie on the same side of the heavy plane —
       inconsistent with a real sp³ CH₂ shell, where the two H atoms
       should straddle the plane on opposite sides.

    All ratios use per-H bond length normalization so the thresholds are
    distance-independent. The three rules use the explicit
    ``80°`` / ``0.30`` / ``0.15`` thresholds called out above; they are
    intentionally hard-coded because the calibration is empirical and
    should not be tuned per call site.

    Args:
        xyz: TS guess XYZ dict — used as a *read-only* coordinate source.
        center_idx: Heavy-atom index of the internal reactive CH₂ carbon.
        heavy_nbr_indices: Sequence of exactly two heavy neighbor indices
            of ``center_idx`` in the molecular graph. The orchestrator
            filters/derives these from the bond graph before calling.
        h_indices: Sequence of exactly two H atom indices bonded to
            ``center_idx``. The orchestrator filters out exempt H atoms
            (e.g. migrating H) before calling.

    Returns:
        ``True`` if the internal CH₂ shell is unphysically misoriented,
        ``False`` otherwise (including all "not eligible" / degenerate
        cases — the orchestrator interprets ``False`` as "leave the
        shell alone").
    """
    if xyz is None:
        return False
    if heavy_nbr_indices is None or h_indices is None:
        return False
    heavy_list = [int(i) for i in heavy_nbr_indices]
    h_list = [int(i) for i in h_indices]
    if len(heavy_list) != 2 or len(h_list) != 2:
        return False
    if heavy_list[0] == heavy_list[1] or h_list[0] == h_list[1]:
        return False
    coords = np.asarray(xyz['coords'], dtype=float)
    n_atoms = len(coords)
    if not (0 <= center_idx < n_atoms):
        return False
    for idx in heavy_list + h_list:
        if not (0 <= idx < n_atoms):
            return False
        if idx == center_idx:
            return False
    if set(heavy_list) & set(h_list):
        return False

    c = coords[center_idx]
    a = coords[heavy_list[0]]
    b = coords[heavy_list[1]]
    v_a = a - c
    v_b = b - c
    n_a = float(np.linalg.norm(v_a))
    n_b = float(np.linalg.norm(v_b))
    if n_a < 1e-8 or n_b < 1e-8:
        return False
    va_hat = v_a / n_a
    vb_hat = v_b / n_b
    w_raw = va_hat + vb_hat
    w_norm = float(np.linalg.norm(w_raw))
    if w_norm < 1e-6:
        # Heavy neighbors anti-parallel: no well-defined corridor.
        return False
    w_hat = w_raw / w_norm
    n_raw = np.cross(v_a, v_b)
    n_norm = float(np.linalg.norm(n_raw))
    if n_norm < 1e-6:
        # Heavy neighbors collinear with center: no well-defined heavy plane.
        return False
    n_hat = n_raw / n_norm

    h0_vec = coords[h_list[0]] - c
    h1_vec = coords[h_list[1]] - c
    r0 = float(np.linalg.norm(h0_vec))
    r1 = float(np.linalg.norm(h1_vec))
    if r0 < 1e-8 or r1 < 1e-8:
        # H on center: degenerate; signal misorientation so repair rebuilds.
        return True

    # ---- Rule A: squeezed H–C–H angle ----
    cos_hch = float(np.dot(h0_vec, h1_vec) / (r0 * r1))
    cos_hch = max(-1.0, min(1.0, cos_hch))
    angle_hch_deg = float(np.degrees(np.arccos(cos_hch)))
    if angle_hch_deg < 80.0:
        return True

    # ---- Rule B: heavy-corridor crowding ----
    proj_w_0 = float(np.dot(h0_vec, w_hat))
    proj_w_1 = float(np.dot(h1_vec, w_hat))
    if (proj_w_0 / r0) > 0.30 and (proj_w_1 / r1) > 0.30:
        return True

    # ---- Rule C: sp³ plane violation ----
    proj_n_0 = float(np.dot(h0_vec, n_hat))
    proj_n_1 = float(np.dot(h1_vec, n_hat))
    ratio_n_0 = proj_n_0 / r0
    ratio_n_1 = proj_n_1 / r1
    if (abs(ratio_n_0) > 0.15
            and abs(ratio_n_1) > 0.15
            and (ratio_n_0 * ratio_n_1) > 0.0):
        return True

    return False


def repair_internal_reactive_ch2(xyz: dict,
                                 center_idx: int,
                                 heavy_nbr_indices: Sequence[int],
                                 h_indices: Sequence[int],
                                 ) -> dict:
    """
    Local repair primitive for an internal reactive CH₂ shell.

    Re-seats the two H atoms of an internal CH₂ center onto the canonical
    sp³-tetrahedral back-side cone of the local heavy-neighbor frame
    while *preserving each H's original center–H bond length exactly*.
    Heavy neighbors and all atoms outside the immediate ``{center, h0,
    h1}`` shell are read-only.

    Local frame (same as :func:`is_internal_reactive_ch2_misoriented`):

        v_a = a − c,                v_b = b − c
        ŵ   = normalize(v̂_a + v̂_b)    (heavy-corridor bisector, IN side)
        n̂   = normalize(v_a × v_b)    (heavy-plane normal)

    Target H directions:

        target_0_dir = α · ŵ + β · n̂
        target_1_dir = α · ŵ − β · n̂

        with α = −1/√3 ≈ −0.5774 (back of corridor — opposite side from the heavy neighbors)
        and β = √(2/3) ≈ 0.8165 (out of the heavy plane, on opposite sides for the two H atoms)

    Both ``target_*_dir`` are unit vectors with cone angle ``arccos(α)
    ≈ 109.47°`` to ``ŵ``. Together they form an inter-H angle of
    ``2·arccos(α/(α² + β²)¹ᐟ²)`` which collapses to the canonical sp³
    tetrahedral ``109.47°`` between the two H atoms — the chemically
    correct CH₂ shell.

    Sticky pairing:
        The two input H atoms are mapped onto the two target slots by
        choosing the assignment that minimizes the total angular
        displacement (sum of arc lengths between the input H direction
        and its assigned target direction). This keeps each H in the
        slot it was already closest to, so already-good shells move the
        least and the repair is deterministic.

    Per-H bond length preservation:
        For each input H ``h_i`` with current ``r_i = |coords[h_i] − c|``,
        the new position is ``c + r_i · target_dir`` where ``target_dir``
        is a unit vector. The norm ``|new_pos − c| = r_i`` is therefore
        preserved exactly (to floating-point tolerance) with no
        additional bookkeeping.

    Eligibility:
        On any of the failure cases that
        :func:`is_internal_reactive_ch2_misoriented` would also reject
        (degenerate frame, missing indices, non-CH₂ shell, etc.) the
        function returns the original ``xyz`` unchanged.

    Args:
        xyz: TS guess XYZ dict.
        center_idx: Heavy-atom index of the internal CH₂ carbon.
        heavy_nbr_indices: Sequence of exactly two heavy neighbor indices.
        h_indices: Sequence of exactly two H atom indices bonded to
            ``center_idx``.

    Returns:
        A new XYZ dict with the two H atoms reseated onto the canonical
        sp³ back-side cone, with each H's bond length preserved. When
        the eligibility check fails the original input is returned
        unchanged.
    """
    if xyz is None:
        return xyz
    if heavy_nbr_indices is None or h_indices is None:
        return xyz
    heavy_list = [int(i) for i in heavy_nbr_indices]
    h_list = [int(i) for i in h_indices]
    if len(heavy_list) != 2 or len(h_list) != 2:
        return xyz
    if heavy_list[0] == heavy_list[1] or h_list[0] == h_list[1]:
        return xyz
    coords = np.asarray(xyz['coords'], dtype=float).copy()
    n_atoms = len(coords)
    if not (0 <= center_idx < n_atoms):
        return xyz
    for idx in heavy_list + h_list:
        if not (0 <= idx < n_atoms):
            return xyz
        if idx == center_idx:
            return xyz
    if set(heavy_list) & set(h_list):
        return xyz

    c = coords[center_idx]
    a = coords[heavy_list[0]]
    b = coords[heavy_list[1]]
    v_a = a - c
    v_b = b - c
    n_a = float(np.linalg.norm(v_a))
    n_b = float(np.linalg.norm(v_b))
    if n_a < 1e-8 or n_b < 1e-8:
        return xyz
    va_hat = v_a / n_a
    vb_hat = v_b / n_b
    w_raw = va_hat + vb_hat
    w_norm = float(np.linalg.norm(w_raw))
    if w_norm < 1e-6:
        return xyz
    w_hat = w_raw / w_norm
    n_raw = np.cross(v_a, v_b)
    n_norm = float(np.linalg.norm(n_raw))
    if n_norm < 1e-6:
        return xyz
    n_hat = n_raw / n_norm

    # Tetrahedral back-of-corridor target directions.
    # α² + β² = 1/3 + 2/3 = 1, so both are unit vectors by construction.
    alpha = -1.0 / float(np.sqrt(3.0))
    beta = float(np.sqrt(2.0 / 3.0))
    target_dirs = [alpha * w_hat + beta * n_hat, alpha * w_hat - beta * n_hat]
    # Re-normalize defensively against floating-point drift.
    for k, td in enumerate(target_dirs):
        td_norm = float(np.linalg.norm(td))
        if td_norm < 1e-8:
            return xyz
        target_dirs[k] = td / td_norm

    h_data: List[Tuple[int, float, np.ndarray]] = []
    for h in h_list:
        rel = coords[h] - c
        r_i = float(np.linalg.norm(rel))
        if r_i < 1e-8:
            return xyz
        h_data.append((h, r_i, rel / r_i))

    # Sticky pairing: minimize sum of (1 - cos θ) on the unit sphere.
    src_dirs = [hd[2] for hd in h_data]
    pairing_a = (0, 1)
    pairing_b = (1, 0)
    cost_a = _sticky_pairing_arc_cost(src_dirs, target_dirs, pairing_a)
    cost_b = _sticky_pairing_arc_cost(src_dirs, target_dirs, pairing_b)
    chosen = pairing_a if cost_a <= cost_b else pairing_b

    new_positions: List[Tuple[int, np.ndarray]] = []
    for src_k, tgt_k in enumerate(chosen):
        h_idx, r_i, _src_dir = h_data[src_k]
        new_pos = c + r_i * target_dirs[tgt_k]
        new_positions.append((h_idx, new_pos))

    for h_idx, new_pos in new_positions:
        coords[h_idx] = new_pos

    return _xyz_with_coords(xyz, coords)


# ---------------------------------------------------------------------------
# reactive-center local-geometry orchestrator
# ---------------------------------------------------------------------------

def apply_reactive_center_cleanup(xyz: dict,
                                  mol: 'Molecule',
                                  migrations: Optional[List[Dict]] = None,
                                  reactive_centers: Optional[Set[int]] = None,
                                  exempt_h_indices: Optional[Set[int]] = None,
                                  weight: float = 0.5,
                                  restore_symmetry: bool = True,
                                  ) -> dict:
    """
    Thin orchestrator over the existing local-geometry helpers.

    This composes the local reactive-center helpers around an explicit
    reactive shell. It is **not** a global "fix everything" pass — it
    touches only:

    * the donor and acceptor heavy atoms named in *migrations* (if any),
    * the heavy atoms in *reactive_centers* (if any), and
    * the H atoms directly bonded to those heavy centers.

    The orchestrator's *only* job is to compose existing helpers in a
    deterministic order so that callers don't repeat the same boilerplate
    in multiple sites. No new geometry primitives are introduced — every
    atomic move is delegated to one of:

    * :func:`clean_migrating_h` — re-place each migrating H at its
      donor/acceptor triangulated position (called only when the migration
      tuple has a trustworthy donor/acceptor pair, e.g. via
      :func:`identify_h_migration_pairs` or
      :func:`infer_frag_fallback_h_migration`).
    * :func:`orient_h_away_from_axis` — reflect non-migrating H atoms
      that are blocking the donor → acceptor corridor.
    * :func:`regularize_terminal_h_geometry` — snap any over-stretched
      or compressed terminal H bond lengths around the donor / acceptor /
      reactive-center heavy atoms back to ~sbl.
    * :func:`restore_terminal_h_symmetry` — when the heavy center is a
      *clear* terminal CH₂ or CH₃ (one heavy neighbor + 2 or 3 H
      neighbors), reseat the H atoms at evenly spaced azimuth around the
      (parent → center) axis. This corrects pathologies like the
      Korcek_step1 terminal CH₃ inversion.

    The migrating H itself (when present) is always exempted from the
    orientation, regularization, and symmetry passes so that the
    triangulated TS placement from :func:`clean_migrating_h` is preserved.

    Args:
        xyz: Post-generation TS guess XYZ dict. May be ``None``, in which
            case the function returns ``None`` unchanged.
        mol: Reactant RMG :class:`Molecule` defining the bond graph.
        migrations: Optional iterable of dicts as produced by
            :func:`identify_h_migration_pairs` or
            :func:`infer_frag_fallback_h_migration`. Each dict must
            contain ``'h_idx'``, ``'donor'``, and ``'acceptor'`` integer
            indices. Other keys are ignored.
        reactive_centers: Optional set of heavy-atom indices that are
            reactive but are not the donor/acceptor of an H migration —
            e.g. a CH₃ that participates in a C–C forming bond and whose
            H geometry should be cleaned up. These centers receive the
            terminal-H regularization and symmetry-restoration passes
            but are never given to :func:`clean_migrating_h` or
            :func:`orient_h_away_from_axis` (those need explicit donor /
            acceptor information).
        exempt_h_indices: Optional H atom indices to exempt from the
            terminal-H regularization and symmetry-restoration passes.
            This is the canonical way for callers using the
            ``reactive_centers`` branch (i.e. those that already
            triangulated their migrating H elsewhere) to keep the
            migrating H pinned at its TS position rather than snapping
            it back to a normal C–H bond length.
        weight: Interpolation weight forwarded to
            :func:`clean_migrating_h`. Currently unused inside that
            helper but kept for forward-compatibility.
        restore_symmetry: When ``False``, the symmetry-restoration pass
            is skipped (the orchestrator becomes a strict superset of the
            current inline cleanup). Defaults to ``True``.

    Returns:
        A new XYZ dict with the local cleanups applied. When the input
        does not name any reactive shell, the original XYZ is returned
        unchanged (no copies made).
    """
    if xyz is None:
        return xyz
    migration_list: List[Dict] = list(migrations or [])
    centers_set: Set[int] = set(reactive_centers or set())
    if not migration_list and not centers_set:
        return xyz

    n_atoms = len(xyz['symbols'])
    migrating_h_indices: Set[int] = set(int(h) for h in (exempt_h_indices or set()))
    cleanup_centers: Set[int] = set()

    # ---- 1. Per-migration triangulation and axis clearing.
    for rec in migration_list:
        try:
            h_idx = int(rec['h_idx'])
            donor = int(rec['donor'])
            acceptor = int(rec['acceptor'])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= h_idx < n_atoms and 0 <= donor < n_atoms and 0 <= acceptor < n_atoms):
            continue
        migrating_h_indices.add(h_idx)
        cleanup_centers.add(donor)
        cleanup_centers.add(acceptor)
        xyz = clean_migrating_h(xyz, mol, donor, acceptor, h_idx, weight=weight)
        # Exempt migrating H so triangulation is not undone.
        xyz = orient_h_away_from_axis(
            xyz, mol, donor, acceptor, exclude_h={h_idx})

    # ---- 2. Terminal-H length regularization on every cleanup center.
    centers_set |= cleanup_centers
    for center in sorted(centers_set):
        if not (0 <= center < n_atoms):
            continue
        xyz = regularize_terminal_h_geometry(
            xyz, mol, center, exclude_atoms=migrating_h_indices)

    # ---- 3. Optional terminal CH₂/CH₃ symmetry restoration.
    #         Gated by is_terminal_group_asymmetric so already-symmetric
    #         groups are left byte-for-byte unchanged.
    if restore_symmetry:
        symbols = xyz['symbols']
        for center in sorted(centers_set):
            if not (0 <= center < n_atoms):
                continue
            if symbols[center] == 'H':
                continue
            heavy_nbrs = _heavy_neighbors(mol, center, symbols)
            if len(heavy_nbrs) != 1:
                continue
            h_nbrs = [h for h in _h_neighbors(mol, center, symbols)
                      if h not in migrating_h_indices]
            if len(h_nbrs) not in (2, 3):
                continue
            parent = heavy_nbrs[0]
            if not is_terminal_group_asymmetric(
                    xyz,
                    center_idx=center,
                    parent_idx=parent,
                    h_indices=h_nbrs):
                continue
            xyz = restore_terminal_h_symmetry(
                xyz, mol, center, exclude_atoms=migrating_h_indices)

    # ---- 4. internal reactive CH₂ misorientation repair.
    #
    #         Sibling pass to the terminal-symmetry restoration above.
    #         The terminal pass (the terminal-group orchestrator) targets centers with one
    #         heavy neighbor and 2/3 H atoms; this pass targets centers
    #         with **two** heavy neighbors and **two** H atoms — i.e.
    #         internal CH₂ shells, the failure cluster was
    #         scoped to repair (1,2_shiftC reactive CH₂,
    #         Intra_Diels_alder_monocyclic attacking CH₂,
    #         Intra_R_Add_Endocyclic reactive CH₂, intra_NO2_ONO_conversion,
    #         Intra_OH_migration internal-CH₂ misorientations).
    #
    #         The two pathways stay deliberately separate: this is a
    #         second eligibility check + repair, NOT a generalization of
    #         the terminal-group logic. An internal CH₂ that already
    #         passes :func:`is_internal_reactive_ch2_misoriented` is
    #         left byte-for-byte unchanged.
    if restore_symmetry:
        symbols = xyz['symbols']
        for center in sorted(centers_set):
            if not (0 <= center < n_atoms):
                continue
            if symbols[center] == 'H':
                continue
            heavy_nbrs = _heavy_neighbors(mol, center, symbols)
            if len(heavy_nbrs) != 2:
                continue  # not internal — terminal centers handled above
            h_nbrs = [h for h in _h_neighbors(mol, center, symbols)
                      if h not in migrating_h_indices]
            if len(h_nbrs) != 2:
                continue  # not a CH₂ shell
            if not is_internal_reactive_ch2_misoriented(
                    xyz,
                    center_idx=center,
                    heavy_nbr_indices=heavy_nbrs,
                    h_indices=h_nbrs):
                continue  # internal CH₂ already well-oriented — leave it alone
            xyz = repair_internal_reactive_ch2(
                xyz,
                center_idx=center,
                heavy_nbr_indices=heavy_nbrs,
                h_indices=h_nbrs,
            )

    return xyz
