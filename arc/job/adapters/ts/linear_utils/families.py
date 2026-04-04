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

if TYPE_CHECKING:
    from rmgpy.molecule.molecule import Molecule

logger = logging.getLogger(__name__)

PAULING_DELTA: float = 0.42


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dihedral_angle(coords: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    """Return the dihedral angle A-B-C-D in degrees (-180, 180]."""
    b1 = coords[b] - coords[a]
    b2 = coords[c] - coords[b]
    b3 = coords[d] - coords[c]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0
    n1 /= n1_norm
    n2 /= n2_norm
    b2_unit = b2 / max(np.linalg.norm(b2), 1e-10)
    m1 = np.cross(n1, b2_unit)
    return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))


def _rotate_fragment(coords: np.ndarray,
                     axis_origin: int,
                     axis_end: int,
                     angle_deg: float,
                     moving_atoms: Set[int],
                     ) -> np.ndarray:
    """Rotate *moving_atoms* by *angle_deg* around the axis origin→end."""
    axis = coords[axis_end] - coords[axis_origin]
    axis_len = float(np.linalg.norm(axis))
    if axis_len < 1e-10:
        return coords
    axis /= axis_len
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    # Rodrigues' rotation
    new_coords = coords.copy()
    for idx in moving_atoms:
        v = coords[idx] - coords[axis_origin]
        v_rot = (v * cos_a
                 + np.cross(axis, v) * sin_a
                 + axis * np.dot(axis, v) * (1.0 - cos_a))
        new_coords[idx] = coords[axis_origin] + v_rot
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
       Cβ (C bonded to Ccarb), Cα (C bonded to Cβ), Hα (H on Cα).
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

    # Find H_alpha: pick the H on C_alpha whose dihedral Hα-Cα-Cβ-Ccarb is closest to ±60°.
    h_alpha_candidates = [j for j in adj[c_alpha] if symbols[j] == 'H']
    if not h_alpha_candidates:
        return None

    coords = np.array(uni_xyz['coords'], dtype=float)

    # Pick the H_alpha with the smallest |dihedral| (closest to gauche).
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
