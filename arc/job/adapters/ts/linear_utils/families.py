"""
Dedicated TS builders for specific reaction families.

These functions handle reaction families whose TS geometry requires
large-scale conformational changes (dihedral rotations + bond stretching)
that cannot be captured by generic Z-matrix interpolation or fragment
stretching.

Each builder takes the unimolecular species geometry and molecular graph,
identifies the reactive substructure, folds the molecule into the TS
conformation via dihedral rotations, and adjusts bond distances.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from arc.common import get_single_bond_length
from arc.job.adapters.ts.linear_utils.geom_utils import (
    dihedral_deg,
    rotate_atoms,
)

if TYPE_CHECKING:
    from rmgpy.molecule.molecule import Molecule
    from arc.job.adapters.ts.linear_utils.path_spec import ReactionPathSpec

logger = logging.getLogger(__name__)

PAULING_DELTA: float = 0.42


# ---------------------------------------------------------------------------
# Geometry helpers (thin wrappers over geom_utils where possible)
# ---------------------------------------------------------------------------

def _dihedral_angle(coords: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    """Return the dihedral angle A-B-C-D in degrees (-180, 180].

    Delegates to :func:`geom_utils.dihedral_deg`.
    """
    return dihedral_deg(coords[a], coords[b], coords[c], coords[d])


def _rotate_fragment(coords: np.ndarray,
                     axis_origin: int,
                     axis_end: int,
                     angle_deg: float,
                     moving_atoms: Set[int],
                     ) -> np.ndarray:
    """Rotate *moving_atoms* by *angle_deg* around the axis origin→end.

    Delegates to :func:`geom_utils.rotate_atoms`.
    """
    axis = coords[axis_end] - coords[axis_origin]
    axis_len = float(np.linalg.norm(axis))
    if axis_len < 1e-10:
        return coords
    new_coords = coords.copy()
    rotate_atoms(new_coords, origin=coords[axis_origin],
                 axis=axis / axis_len, indices=moving_atoms,
                 angle=np.radians(angle_deg))
    return new_coords


def _set_bond_distance(coords: np.ndarray,
                       fixed: int,
                       mobile: int,
                       target_dist: float,
                       mobile_frag: Set[int],
                       ) -> np.ndarray:
    """Translate *mobile_frag* so that the fixed–mobile distance is *target_dist*."""
    vec = coords[mobile] - coords[fixed]
    d_cur = float(np.linalg.norm(vec))
    if d_cur < 1e-10:
        return coords
    direction = vec / d_cur
    displacement = direction * (target_dist - d_cur)
    new_coords = coords.copy()
    for idx in mobile_frag:
        new_coords[idx] += displacement
    return new_coords


def _bfs_fragment(adj: Dict[int, Set[int]], start: int, block: Set[int]) -> Set[int]:
    """BFS from *start*, not crossing any atom in *block*."""
    visited: Set[int] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited or node in block:
            continue
        visited.add(node)
        stack.extend(adj[node] - visited)
    return visited


# ---------------------------------------------------------------------------
# XY_elimination_hydroxyl builder
# ---------------------------------------------------------------------------

def build_xy_elimination_ts(uni_xyz: dict,
                            uni_mol: 'Molecule',
                            ) -> Optional[dict]:
    """
    Build a TS for XY_elimination_hydroxyl: R-CH₂-CH₂-C(=O)OH → R-CH=CH₂ + H₂ + CO₂.

    The TS is a planar 6-membered ring formed by folding the molecule:

    .. code-block:: text

        Cα ---- Hα
        |         |       Hα detaches from Cα
        Cβ      Hoh       Hoh detaches from Ooh
        |         |       Hα and Hoh form H₂
        Ccarb -- Ooh      Cα=Cβ double bond forms, Cβ-Ccarb breaks

    Algorithm:

    1. **Identify ring atoms** from the molecular graph:
       Ccarb (has =O and -OH), Ooh (-OH oxygen), Hoh (H on Ooh),
       Cβ (C bonded to Ccarb), Cα (C bonded to Cβ), Hα (H on Cα
       whose dihedral Hα-Cα-Cβ-Ccarb is closest to 0°, i.e., most
       eclipsed with the carboxyl — this is the H in the ring plane).
    2. **Rotate dihedrals** to fold the chain into a ring:
       - Cα-Cβ-Ccarb-Ooh → 0° (bring hydroxyl syn to Cα)
       - Cβ-Ccarb-Ooh-Hoh → 0° (fold Hoh toward Cα)
       - Hα-Cα-Cβ-Ccarb → 0° (bring Hα into the ring plane)
    3. **Set TS bond distances** using Pauling estimates:
       - Cα-Hα: stretch to 1.51 Å (C-H Pauling)
       - Ooh-Hoh: stretch to 1.39 Å (O-H Pauling)
       - Cβ-Ccarb: stretch to SBL + 2×δ ≈ 2.38 Å (heavy-heavy breaking)
       - Cα-Cβ: shorten to ~1.42 Å (C=C forming)

    Args:
        uni_xyz: Reactant XYZ coordinates.
        uni_mol: Reactant RMG Molecule.

    Returns:
        TS guess XYZ dictionary, or ``None`` if the pattern is not detected.
    """
    symbols = uni_xyz['symbols']
    n_atoms = len(symbols)
    atom_to_idx = {atom: idx for idx, atom in enumerate(uni_mol.atoms)}

    # Build adjacency.
    adj: Dict[int, Set[int]] = {k: set() for k in range(n_atoms)}
    bond_orders: Dict[Tuple[int, int], float] = {}
    for atom in uni_mol.atoms:
        ia = atom_to_idx[atom]
        for nbr, bond in atom.bonds.items():
            ib = atom_to_idx[nbr]
            adj[ia].add(ib)
            key = (min(ia, ib), max(ia, ib))
            bond_orders[key] = bond.order

    # --- Step 1: Identify the 6-membered ring atoms ---
    # Find C_carb: a carbon with one =O neighbour and one -OH neighbour.
    c_carb = o_double = o_hydroxyl = h_hydroxyl = c_beta = c_alpha = h_alpha = None

    for i, sym in enumerate(symbols):
        if sym != 'C':
            continue
        o_neighbors = [j for j in adj[i] if symbols[j] == 'O']
        if len(o_neighbors) < 2:
            continue
        # Check for one C=O and one C-O(H).
        o_dbl = o_single = None
        for oj in o_neighbors:
            key = (min(i, oj), max(i, oj))
            order = bond_orders.get(key, 1.0)
            h_on_o = [k for k in adj[oj] if symbols[k] == 'H']
            if order >= 1.5 and not h_on_o:
                o_dbl = oj
            elif h_on_o:
                o_single = oj
        if o_dbl is not None and o_single is not None:
            c_carb = i
            o_double = o_dbl
            o_hydroxyl = o_single
            h_hydroxyl = [k for k in adj[o_hydroxyl] if symbols[k] == 'H'][0]
            break

    if c_carb is None:
        return None

    # Find C_beta: a C neighbour of C_carb.
    c_beta_candidates = [j for j in adj[c_carb] if symbols[j] == 'C']
    if not c_beta_candidates:
        return None
    c_beta = c_beta_candidates[0]

    # Find C_alpha: a C neighbour of C_beta that is not C_carb.
    c_alpha_candidates = [j for j in adj[c_beta] if symbols[j] == 'C' and j != c_carb]
    if not c_alpha_candidates:
        return None
    c_alpha = c_alpha_candidates[0]

    # Find H_alpha: pick the H on C_alpha whose dihedral Hα-Cα-Cβ-Ccarb is
    # closest to 0° (eclipsed with Ccarb).  This is the H that will end up
    # in the 6-membered ring plane after the folding rotations.
    h_alpha_candidates = [j for j in adj[c_alpha] if symbols[j] == 'H']
    if not h_alpha_candidates:
        return None

    coords = np.array(uni_xyz['coords'], dtype=float)

    # Pick the H_alpha with the smallest |dihedral| (closest to eclipsed).
    best_h = None
    best_abs_dih = 999.0
    for h_cand in h_alpha_candidates:
        dih = abs(_dihedral_angle(coords, h_cand, c_alpha, c_beta, c_carb))
        if dih < best_abs_dih:
            best_abs_dih = dih
            best_h = h_cand
    h_alpha = best_h

    logger.debug(f'XY_elimination: c_carb={c_carb}, o_dbl={o_double}, o_oh={o_hydroxyl}, '
                 f'h_oh={h_hydroxyl}, c_beta={c_beta}, c_alpha={c_alpha}, h_alpha={h_alpha}')

    # --- Step 2: Rotate dihedrals to form the ring ---

    # 2a. Rotate Cα-Cβ-Ccarb-Ooh → 0° (bring hydroxyl syn to Cα).
    dih_1 = _dihedral_angle(coords, c_alpha, c_beta, c_carb, o_hydroxyl)
    frag_1 = _bfs_fragment(adj, c_carb, block={c_beta})
    coords = _rotate_fragment(coords, axis_origin=c_beta, axis_end=c_carb,
                              angle_deg=-dih_1, moving_atoms=frag_1)

    # 2b. Rotate Cβ-Ccarb-Ooh-Hoh → 0° (fold Hoh toward Cα).
    dih_2 = _dihedral_angle(coords, c_beta, c_carb, o_hydroxyl, h_hydroxyl)
    frag_2 = _bfs_fragment(adj, o_hydroxyl, block={c_carb})
    coords = _rotate_fragment(coords, axis_origin=c_carb, axis_end=o_hydroxyl,
                              angle_deg=-dih_2, moving_atoms=frag_2)

    # 2c. Rotate Hα around Cα-Cβ to MINIMIZE Hα-Hoh distance.
    #     Scan 360° in 5° steps and pick the angle that brings Hα closest to Hoh.
    frag_3 = _bfs_fragment(adj, c_alpha, block={c_beta})
    best_angle = 0.0
    best_dist = float('inf')
    for angle_step in range(72):  # 0°, 5°, 10°, ..., 355°
        angle = angle_step * 5.0
        trial = _rotate_fragment(coords, axis_origin=c_beta, axis_end=c_alpha,
                                 angle_deg=angle, moving_atoms=frag_3)
        d = float(np.linalg.norm(trial[h_alpha] - trial[h_hydroxyl]))
        if d < best_dist:
            best_dist = d
            best_angle = angle
    coords = _rotate_fragment(coords, axis_origin=c_beta, axis_end=c_alpha,
                              angle_deg=best_angle, moving_atoms=frag_3)

    # --- Step 3: Set TS bond distances ---

    # 3a. Stretch C_beta - C_carb (breaking bond) to heavy-heavy Pauling.
    #     Do this FIRST since it moves the carboxyl fragment away, giving
    #     space for the H atoms to be positioned.
    sbl_cc = get_single_bond_length('C', 'C') or 1.54
    d_cc_break_target = sbl_cc + 2.0 * PAULING_DELTA  # ~2.38
    frag_carboxyl = _bfs_fragment(adj, c_carb, block={c_beta})
    coords = _set_bond_distance(coords, fixed=c_beta, mobile=c_carb,
                                target_dist=d_cc_break_target, mobile_frag=frag_carboxyl)

    # 3b. Shorten C_alpha - C_beta (forming C=C).
    d_cc_form_target = sbl_cc * 0.93  # ~1.43
    frag_alpha = _bfs_fragment(adj, c_alpha, block={c_beta})
    coords = _set_bond_distance(coords, fixed=c_beta, mobile=c_alpha,
                                target_dist=d_cc_form_target, mobile_frag=frag_alpha)

    # 3c. Place Hα and Hoh at TS positions using triangulation.
    #     Hα is placed at Pauling C-H distance from Cα, directed toward Hoh.
    #     Hoh is placed at Pauling O-H distance from Ooh, directed toward Hα.
    #     Then both are adjusted to achieve the target H-H distance.
    sbl_ch = get_single_bond_length('C', 'H') or 1.09
    sbl_oh = get_single_bond_length('O', 'H') or 0.97
    sbl_hh = get_single_bond_length('H', 'H') or 0.74
    d_ch_target = sbl_ch + PAULING_DELTA    # ~1.51
    d_oh_target = sbl_oh + PAULING_DELTA    # ~1.39
    d_hh_target = sbl_hh * 1.12             # ~0.83

    # Direction from Cα toward the midpoint of (current Hα, current Hoh).
    mid_hh = 0.5 * (coords[h_alpha] + coords[h_hydroxyl])
    dir_ca_to_mid = mid_hh - coords[c_alpha]
    d_ca_mid = float(np.linalg.norm(dir_ca_to_mid))
    if d_ca_mid > 1e-6:
        dir_ca_to_mid /= d_ca_mid
    else:
        dir_ca_to_mid = coords[h_alpha] - coords[c_alpha]
        dir_ca_to_mid /= max(float(np.linalg.norm(dir_ca_to_mid)), 1e-10)

    # Place Hα at d_ch_target from Cα, toward the midpoint.
    coords[h_alpha] = coords[c_alpha] + dir_ca_to_mid * d_ch_target

    # Direction from Ooh toward Hα (now placed).
    dir_o_to_h = coords[h_alpha] - coords[o_hydroxyl]
    d_o_h = float(np.linalg.norm(dir_o_to_h))
    if d_o_h > 1e-6:
        dir_o_to_h /= d_o_h
    # Place Hoh at d_oh_target from Ooh, toward Hα.
    coords[h_hydroxyl] = coords[o_hydroxyl] + dir_o_to_h * d_oh_target

    # Fine-tune: place both H atoms to achieve the target H-H distance.
    # Move them along the Hα-Hoh axis to get d_hh_target.
    hh_vec = coords[h_hydroxyl] - coords[h_alpha]
    hh_dist = float(np.linalg.norm(hh_vec))
    if hh_dist > 1e-6:
        hh_mid = 0.5 * (coords[h_alpha] + coords[h_hydroxyl])
        hh_dir = hh_vec / hh_dist
        coords[h_alpha] = hh_mid - hh_dir * d_hh_target * 0.5
        coords[h_hydroxyl] = hh_mid + hh_dir * d_hh_target * 0.5

    ts_xyz = {
        'symbols': symbols,
        'isotopes': uni_xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
        'coords': tuple(tuple(float(x) for x in row) for row in coords),
    }
    return ts_xyz


# ---------------------------------------------------------------------------
# 1,3_sigmatropic_rearrangement bespoke builder
# ---------------------------------------------------------------------------


def build_1_3_sigmatropic_rearrangement_ts(
        r_xyz: dict,
        r_mol: 'Molecule',
        breaking_bonds: List[Tuple[int, int]],
        forming_bonds: List[Tuple[int, int]],
) -> Optional[dict]:
    """Build a compact TS for a 1,3-sigmatropic rearrangement.

    In a [1,3]-sigmatropic shift one atom (the *migrating atom*) moves
    from the *origin* to the *target* via the allylic/heteroallylic
    bridge.  The TS has the migrating atom positioned at calibrated
    distances from both origin and target, with unchanged near-core
    heavy-heavy bonds preserved in a chemically sane window.

    Motif identification (from ``breaking_bonds`` / ``forming_bonds``):

    * Exactly one breaking bond ``(origin, migrating)`` and exactly one
      forming bond ``(target, migrating)`` must share a common atom (the
      migrating atom).
    * The function returns ``None`` if these conditions are not met or
      if any index is out of range.

    Geometry construction:

    1. The migrating atom is placed at the intersection of two spheres
       centred on the origin (radius = calibrated breaking-side target)
       and target (radius = calibrated forming-side target).  The
       perpendicular direction is taken from the migrating atom's
       current position to keep the guess in the same hemisphere as
       the reactant geometry.
    2. H-only substituents of the migrating atom are translated rigidly
       with it.
    3. All other atoms remain at their reactant positions.
    4. If any heavy-heavy near-core bond (within the reactive core)
       ends up shorter than 0.9 Å after migration, the guess is
       considered unphysical and ``None`` is returned.

    Calibrated distances (specific to 1,3-sigmatropic shifts):

    * Breaking side: ``sbl + 0.77 Å`` (calibrated to the DFT-curated TS
      at ~2.24 Å for C–N; the [1,3]-sigmatropic TS is orbital-symmetry-
      forbidden and sits at a late, stretched geometry).
    * Forming side: ``sbl + 0.38 Å`` (calibrated to ~1.85 Å for N–C;
      the forming bond is already mostly formed in the TS).

    Args:
        r_xyz: Reactant XYZ dict.
        r_mol: Reactant RMG Molecule.
        breaking_bonds: List of (i, j) bonds that break.
        forming_bonds: List of (i, j) bonds that form.

    Returns:
        TS XYZ dict, or ``None`` if the motif cannot be identified
        unambiguously or if the guess is unphysical.
    """
    bb = list(breaking_bonds or [])
    fb = list(forming_bonds or [])
    if len(bb) != 1 or len(fb) != 1:
        return None

    # Identify migrating atom (common to both the breaking and forming bond).
    bb_set = set(bb[0])
    fb_set = set(fb[0])
    common = bb_set & fb_set
    if len(common) != 1:
        return None
    migrating = common.pop()
    origin = (bb_set - {migrating}).pop()
    target = (fb_set - {migrating}).pop()

    symbols = r_xyz['symbols']
    n_atoms = len(symbols)
    if not all(0 <= idx < n_atoms for idx in (migrating, origin, target)):
        return None

    coords = np.array(r_xyz['coords'], dtype=float).copy()
    atom_to_idx = {atom: idx for idx, atom in enumerate(r_mol.atoms)}

    # Calibrated TS target distances for [1,3]-sigmatropic shifts.
    # These are narrowly scoped to this family and calibrated from
    # DFT-curated TS reference geometries.  The TS is orbital-symmetry-
    # forbidden (Woodward-Hoffmann), so the breaking bond is stretched
    # much more than generic Pauling delta predicts.
    sbl_break = get_single_bond_length(symbols[origin], symbols[migrating]) or 1.5
    sbl_form = get_single_bond_length(symbols[target], symbols[migrating]) or 1.5
    _SIGMA_BREAK_STRETCH = 0.77  # Å above sbl (calibrated: ~2.24 for C-N)
    _SIGMA_FORM_STRETCH = 0.38   # Å above sbl (calibrated: ~1.85 for N-C)
    d_break_target = sbl_break + _SIGMA_BREAK_STRETCH
    d_form_target = sbl_form + _SIGMA_FORM_STRETCH

    # Place the migrating atom at the two-sphere intersection of
    # origin (radius d_break_target) and target (radius d_form_target).
    origin_pos = coords[origin]
    target_pos = coords[target]
    ot_vec = target_pos - origin_pos
    ot_dist = float(np.linalg.norm(ot_vec))
    if ot_dist < 1e-6:
        return None
    ot_hat = ot_vec / ot_dist

    if ot_dist <= d_break_target + d_form_target:
        # Spheres overlap — triangulate.
        x = (ot_dist ** 2 + d_break_target ** 2 - d_form_target ** 2) / (2.0 * ot_dist)
        h_sq = d_break_target ** 2 - x ** 2
        h_perp = float(np.sqrt(max(h_sq, 0.0)))
    else:
        # Spheres don't overlap — place at d_break_target along the axis.
        x = d_break_target
        h_perp = 0.0

    # Perpendicular direction: use the out-of-plane normal of the
    # (origin, target, bridge_neighbor) triangle.  In a cyclic
    # [1,3]-shift the migrating atom has a bridge neighbor (bonded
    # to it in both R and P, e.g. C2 in the imidazole case) that
    # lies IN the ring plane.  Using the ring-plane normal as the
    # perpendicular direction ensures the migrating atom is placed
    # OUT of the plane, avoiding collision with the bridge neighbor.
    mig_pos = coords[migrating]
    # Find the bridge neighbor: a heavy neighbor of the migrating atom
    # that is neither the origin nor the target.
    bridge = None
    for nbr in r_mol.atoms[migrating].bonds.keys():
        ni = atom_to_idx[nbr]
        if symbols[ni] != 'H' and ni != origin and ni != target:
            bridge = ni
            break
    if bridge is not None:
        aux_vec = coords[bridge] - origin_pos
        perp = np.cross(ot_hat, aux_vec)
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm > 1e-8:
            perp_hat = perp / perp_norm
            # Choose the sign that puts the migrating atom on the same
            # side as its current position.
            proj_on_axis = origin_pos + ot_hat * np.dot(mig_pos - origin_pos, ot_hat)
            mig_offset = mig_pos - proj_on_axis
            if float(np.dot(mig_offset, perp_hat)) < 0:
                perp_hat = -perp_hat
        else:
            perp_hat = None
    else:
        perp_hat = None

    if perp_hat is None:
        # Fallback: use the migrating atom's current offset.
        proj_on_axis = origin_pos + ot_hat * np.dot(mig_pos - origin_pos, ot_hat)
        perp = mig_pos - proj_on_axis
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm > 1e-8:
            perp_hat = perp / perp_norm
        else:
            arb = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(ot_hat, arb))) > 0.9:
                arb = np.array([0.0, 1.0, 0.0])
            perp_hat = np.cross(ot_hat, arb)
            perp_hat /= max(float(np.linalg.norm(perp_hat)), 1e-10)

    new_mig_pos = origin_pos + ot_hat * x + perp_hat * h_perp

    # Identify H-only substituents on the migrating atom (move them
    # rigidly with the migrating atom).
    displacement = new_mig_pos - coords[migrating]
    moving_set = {migrating}
    for nbr in r_mol.atoms[migrating].bonds.keys():
        ni = atom_to_idx[nbr]
        if symbols[ni] == 'H':
            moving_set.add(ni)
    for idx in moving_set:
        coords[idx] += displacement

    # Local core-shaping step: adjust the bridge atom(s) toward the
    # migrating atom's new position so that the bridge–migrating
    # distance approaches its curated TS value (~1.66 Å for C–C in
    # the imidazole case) rather than being over-stretched.
    #
    # The bridge atom is the heavy neighbor of the migrating atom that
    # is neither the origin nor the target — it persists in both
    # reactant and product.  Moving it (and its H-only substituents)
    # partially toward the migrating atom's new position closes the
    # gap without disturbing the calibrated motif edges (which are
    # anchored to origin and target, not the bridge).
    _BRIDGE_TARGET_FACTOR = 0.40  # fraction of the gap to close
    if bridge is not None:
        d_bridge_mig = float(np.linalg.norm(coords[bridge] - coords[migrating]))
        # Calibrated bridge–migrating distance: midpoint between the
        # reactant value (aromatic ~1.37) and a stretched single bond
        # (~1.66).  Use a family-calibrated target.
        sbl_bridge = get_single_bond_length(
            symbols[bridge], symbols[migrating]) or 1.5
        d_bridge_target = sbl_bridge + 0.12  # ~1.66 for C–C
        if d_bridge_mig > d_bridge_target:
            bridge_dir = coords[migrating] - coords[bridge]
            bridge_dir_norm = float(np.linalg.norm(bridge_dir))
            if bridge_dir_norm > 1e-6:
                bridge_hat = bridge_dir / bridge_dir_norm
                # Move bridge partially toward the calibrated target.
                gap = d_bridge_mig - d_bridge_target
                shift = bridge_hat * gap * _BRIDGE_TARGET_FACTOR
                bridge_moving = {bridge}
                for nbr in r_mol.atoms[bridge].bonds.keys():
                    ni = atom_to_idx[nbr]
                    if symbols[ni] == 'H':
                        bridge_moving.add(ni)
                for idx in bridge_moving:
                    coords[idx] += shift

    # Ring planarity enforcement: project all ring heavy atoms (and
    # their directly bonded H atoms) onto the best-fit plane of the
    # ring.  The [1,3]-sigmatropic TS for cyclic substrates (e.g. the
    # imidazole → azirine rearrangement) should be approximately
    # planar, but the out-of-plane displacement from the triangulation
    # leaves the migrating atom far from the ring plane.
    ring_atoms = {migrating, origin, target}
    if bridge is not None:
        ring_atoms.add(bridge)
    # Add any other heavy atoms bonded to two ring members (ring-closing).
    for idx in list(ring_atoms):
        for nbr in r_mol.atoms[idx].bonds.keys():
            ni = atom_to_idx[nbr]
            if symbols[ni] != 'H' and ni not in ring_atoms:
                # Check if this neighbor is bonded to another ring member.
                nbr_ring_count = sum(
                    1 for nbr2 in r_mol.atoms[ni].bonds.keys()
                    if atom_to_idx[nbr2] in ring_atoms)
                if nbr_ring_count >= 2:
                    ring_atoms.add(ni)
    _PLANARITY_FRACTION = 0.20  # fraction of out-of-plane offset to remove
    if len(ring_atoms) >= 3:
        ring_list = sorted(ring_atoms)
        ring_pts = coords[ring_list]
        centroid = ring_pts.mean(axis=0)
        centered = ring_pts - centroid
        _, _, vt = np.linalg.svd(centered)
        plane_normal = vt[-1]
        # Partially project ring atoms toward the plane.  Full
        # projection can collapse bonds between atoms that are far
        # apart in-plane but close in 3D (e.g. C0-N4 in the imidazole
        # case).  Moving only a fraction preserves inter-atom distances
        # while reducing the out-of-plane deviation.
        for ri in ring_list:
            offset = float(np.dot(coords[ri] - centroid, plane_normal))
            coords[ri] -= offset * _PLANARITY_FRACTION * plane_normal
            for nbr in r_mol.atoms[ri].bonds.keys():
                hi = atom_to_idx[nbr]
                if symbols[hi] == 'H':
                    h_offset = float(np.dot(coords[hi] - centroid, plane_normal))
                    coords[hi] -= h_offset * _PLANARITY_FRACTION * plane_normal

    # Sanity guard: check that no heavy-heavy near-core bond collapsed
    # to less than 0.9 Å (would indicate an unphysical geometry).
    for atom in r_mol.atoms:
        ia = atom_to_idx[atom]
        if symbols[ia] == 'H':
            continue
        for nbr in atom.bonds.keys():
            ib = atom_to_idx[nbr]
            if symbols[ib] == 'H':
                continue
            d = float(np.linalg.norm(coords[ia] - coords[ib]))
            if d < 0.9:
                logger.debug(f'1,3_sigmatropic builder: near-core bond '
                             f'{symbols[ia]}{ia}-{symbols[ib]}{ib}={d:.3f} collapsed; '
                             f'returning None.')
                return None

    ts_xyz = {
        'symbols': symbols,
        'isotopes': r_xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
        'coords': tuple(tuple(float(c) for c in row) for row in coords),
    }
    return ts_xyz


# ---------------------------------------------------------------------------
# Baeyer-Villiger_step2 bespoke builder
# ---------------------------------------------------------------------------


def build_baeyer_villiger_step2_ts(
        uni_xyz: dict,
        uni_mol: 'Molecule',
        split_bonds: List[Tuple[int, int]],
) -> Optional[dict]:
    """Build a concerted TS for the Baeyer-Villiger step 2 rearrangement.

    In BV step 2, a Criegee intermediate rearranges concertedly:
    the O-O peroxide bond breaks, an alkyl group on the *other*
    (non-carbonyl) side of the peroxide migrates from its parent
    carbon toward the peroxide oxygen, forming a new C-O bond while
    the parent carbon loses the migrating group.

    Motif identification:

    1. ``split_bonds`` must contain at least one O-O peroxide cut.
    2. One peroxide O (*carbonyl-side O*) is bonded to a C with a C=O.
    3. The other peroxide O (*other-side O*) is bonded to a *quaternary-
       like C* (the origin of the migrating group).
    4. The migrating group root is a non-O, non-H neighbor of the
       quaternary C.  When there are multiple candidates (e.g. two
       CH₃ groups), each is tried and the first non-colliding guess
       is returned.
    5. If any of these conditions are not met, return ``None``.

    Geometry construction (calibrated to DFT-curated TS):

    1. Stretch O-O to ~2.02 Å (``sbl + 2 × 0.27``; calibrated from
       the curated TS).
    2. Stretch C_parent-C_migrating to ~2.30 Å (``sbl + 0.76``;
       calibrated — the migrating group is leaving its parent).
    3. Contract C_migrating toward O_other to ~2.16 Å
       (``sbl(C,O) + 0.73``; calibrated — the migrating group is
       approaching the peroxide O through which it migrates).
    4. If the guess has atom collisions, try the next candidate.

    Args:
        uni_xyz: Criegee intermediate XYZ dict.
        uni_mol: Criegee intermediate RMG Molecule.
        split_bonds: Fragmentation bond cuts (from
            :func:`find_split_bonds_by_fragmentation`).

    Returns:
        TS XYZ dict, or ``None`` if the BV motif cannot be identified.
    """
    from arc.species.species import colliding_atoms as _colliding

    symbols = uni_xyz['symbols']
    n_atoms = len(symbols)
    atom_to_idx = {atom: idx for idx, atom in enumerate(uni_mol.atoms)}

    # Build adjacency and bond orders.
    adj: Dict[int, Set[int]] = {k: set() for k in range(n_atoms)}
    bond_orders: Dict[Tuple[int, int], float] = {}
    for atom in uni_mol.atoms:
        ia = atom_to_idx[atom]
        for nbr, bond in atom.bonds.items():
            ib = atom_to_idx[nbr]
            adj[ia].add(ib)
            key = (min(ia, ib), max(ia, ib))
            bond_orders[key] = bond.order

    # Step 1: Find the O-O peroxide bond.
    oo_bond = None
    for a, b in split_bonds:
        if symbols[a] == 'O' and symbols[b] == 'O':
            oo_bond = (a, b)
            break
    if oo_bond is None:
        return None

    # Step 2: Identify the carbonyl-side O and the other-side O.
    o_carb_side = None
    o_other_side = None
    c_carbonyl = None
    for o_candidate in oo_bond:
        for nbr_idx in adj[o_candidate]:
            if symbols[nbr_idx] != 'C':
                continue
            for nbr2_idx in adj[nbr_idx]:
                if nbr2_idx == o_candidate:
                    continue
                if symbols[nbr2_idx] != 'O':
                    continue
                key = (min(nbr_idx, nbr2_idx), max(nbr_idx, nbr2_idx))
                if bond_orders.get(key, 1.0) >= 1.5:
                    o_carb_side = o_candidate
                    c_carbonyl = nbr_idx
                    break
            if c_carbonyl is not None:
                break
        if c_carbonyl is not None:
            break
    if c_carbonyl is None:
        return None
    o_other_side = oo_bond[0] if oo_bond[1] == o_carb_side else oo_bond[1]

    # Step 3: Identify the quaternary-like C on the other side of the
    # peroxide (the parent of the migrating group).
    c_parent = None
    for nbr_idx in adj[o_other_side]:
        if symbols[nbr_idx] == 'C':
            c_parent = nbr_idx
            break
    if c_parent is None:
        return None

    # Step 4: Identify candidate migrating groups: non-O, non-H
    # neighbors of c_parent (excluding o_other_side itself).
    mig_candidates = []
    for nbr_idx in adj[c_parent]:
        if nbr_idx == o_other_side:
            continue
        if symbols[nbr_idx] in ('O', 'H'):
            continue
        mig_candidates.append(nbr_idx)
    if not mig_candidates:
        return None

    # Calibrated TS target distances (specific to BV step 2,
    # calibrated from DFT-curated TS reference).
    sbl_oo = get_single_bond_length('O', 'O') or 1.48
    sbl_cc = get_single_bond_length('C', 'C') or 1.54
    sbl_co = get_single_bond_length('C', 'O') or 1.43

    _BV_OO_STRETCH = 0.27    # Å per side (total stretch ≈ 2 × 0.27 above sbl)
    _BV_CC_STRETCH = 0.76    # Å above sbl for migrating-group departure
    _BV_CO_STRETCH = 0.73    # Å above sbl(C,O) for migrating-group approach

    d_oo_target = sbl_oo + 2.0 * _BV_OO_STRETCH
    d_cc_target = sbl_cc + _BV_CC_STRETCH
    d_approach_target = sbl_co + _BV_CO_STRETCH

    # Step 5: Identify the OH oxygen on c_parent (if present) — its
    # bond to c_parent shortens in the TS as it forms a new C=O.
    o_hydroxyl = None
    for nbr_idx in adj[c_parent]:
        if nbr_idx == o_other_side:
            continue
        if symbols[nbr_idx] != 'O':
            continue
        # Check if this O has an H neighbor (hydroxyl).
        h_on_o = [k for k in adj[nbr_idx] if symbols[k] == 'H']
        if h_on_o:
            o_hydroxyl = nbr_idx
            break

    # Calibrated C_parent-OH target distance (from DFT TS: 1.279 Å).
    # In the concerted mechanism, C_parent forms a new C=O with the
    # hydroxyl oxygen as the migrating group departs.
    _BV_CO_SHORTEN_TARGET = 1.28  # Å (calibrated)

    # Try each candidate migrating group; collect all non-colliding
    # guesses and return the best-fit one.
    best_ts: Optional[dict] = None
    best_score = float('inf')

    for c_migrating in mig_candidates:
        coords = np.array(uni_xyz['coords'], dtype=float).copy()

        # 4a. Stretch the O-O peroxide bond.
        frag_other = _bfs_fragment(adj, o_other_side, block={o_carb_side})
        coords = _set_bond_distance(coords, fixed=o_carb_side,
                                    mobile=o_other_side,
                                    target_dist=d_oo_target,
                                    mobile_frag=frag_other)

        # 4b. Place C_migrating via two-sphere triangulation between
        # C_parent (at d_cc_target) and O_other (at d_approach_target).
        # This avoids the sequential stretch-then-contract problem
        # where the contraction undoes the stretch.
        frag_mig = _bfs_fragment(adj, c_migrating, block={c_parent})
        p1 = coords[c_parent]
        p2 = coords[o_other_side]
        axis = p2 - p1
        axis_dist = float(np.linalg.norm(axis))
        if axis_dist < 1e-6:
            continue
        axis_hat = axis / axis_dist

        r1 = d_cc_target
        r2 = d_approach_target
        if axis_dist <= r1 + r2:
            x = (axis_dist ** 2 + r1 ** 2 - r2 ** 2) / (2.0 * axis_dist)
            h_sq = r1 ** 2 - x ** 2
            h_perp = float(np.sqrt(max(h_sq, 0.0)))
        else:
            x = r1
            h_perp = 0.0

        # Perpendicular direction: use C_migrating's current offset
        # from the C_parent → O_other axis.
        mig_pos = coords[c_migrating]
        proj = p1 + axis_hat * np.dot(mig_pos - p1, axis_hat)
        perp = mig_pos - proj
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm > 1e-8:
            perp_hat = perp / perp_norm
        else:
            arb = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(axis_hat, arb))) > 0.9:
                arb = np.array([0.0, 1.0, 0.0])
            perp_hat = np.cross(axis_hat, arb)
            perp_hat /= max(float(np.linalg.norm(perp_hat)), 1e-10)

        new_mig_pos = p1 + axis_hat * x + perp_hat * h_perp
        displacement = new_mig_pos - coords[c_migrating]
        for idx in frag_mig:
            coords[idx] += displacement

        # 4c. Shorten C_parent-O_hydroxyl bond toward calibrated TS
        # target (~1.28 Å).  In the concerted mechanism, C_parent forms
        # a new C=O with the hydroxyl oxygen as the migrating group
        # departs.  Only the O and its H(s) are moved.
        if o_hydroxyl is not None:
            frag_oh = _bfs_fragment(adj, o_hydroxyl, block={c_parent})
            coords = _set_bond_distance(
                coords, fixed=c_parent, mobile=o_hydroxyl,
                target_dist=_BV_CO_SHORTEN_TARGET,
                mobile_frag=frag_oh)

        # 4d. Stretch the hydroxyl O-H bond toward the calibrated TS
        # target (~1.40 Å).  In the concerted BV mechanism, the
        # hydroxyl H loosens from O_hydroxyl as the C=O forms.  The
        # curated TS has d(O-H) ≈ 1.40 Å (stretched from ~0.97 Å).
        # The H is moved along the O→H direction without moving any
        # other atom.
        _BV_OH_STRETCH_TARGET = 1.40  # Å (calibrated from DFT TS)
        if o_hydroxyl is not None:
            h_on_oh = [k for k in adj[o_hydroxyl] if symbols[k] == 'H']
            for h_idx in h_on_oh:
                oh_vec = coords[h_idx] - coords[o_hydroxyl]
                oh_dist = float(np.linalg.norm(oh_vec))
                if oh_dist > 1e-6 and oh_dist < _BV_OH_STRETCH_TARGET:
                    oh_hat = oh_vec / oh_dist
                    coords[h_idx] = coords[o_hydroxyl] + oh_hat * _BV_OH_STRETCH_TARGET

        ts_xyz = {
            'symbols': symbols,
            'isotopes': uni_xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
            'coords': tuple(tuple(float(c) for c in row) for row in coords),
        }
        if _colliding(ts_xyz):
            continue

        # Score: sum of squared deviations from calibrated targets.
        d_oo = float(np.linalg.norm(coords[oo_bond[0]] - coords[oo_bond[1]]))
        d_cc = float(np.linalg.norm(coords[c_parent] - coords[c_migrating]))
        d_co = float(np.linalg.norm(coords[c_migrating] - coords[o_other_side]))
        score = ((d_oo - d_oo_target) ** 2
                 + (d_cc - d_cc_target) ** 2
                 + (d_co - d_approach_target) ** 2)
        if score < best_score:
            best_score = score
            best_ts = ts_xyz

    return best_ts
