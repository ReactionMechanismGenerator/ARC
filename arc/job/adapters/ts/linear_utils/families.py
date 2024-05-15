"""
Dedicated TS builders for specific reaction families.

Each builder takes the reactant (or unimolecular intermediate) geometry
and molecular graph, identifies the reactive substructure from
breaking/forming bond indices, and constructs a TS guess using
family-specific geometric rules (two-sphere triangulation, ring
closure, fragment rotation, etc.).

Supported families:

* ``XY_elimination``: 4-center concerted elimination.
* ``1,3_sigmatropic_rearrangement``: ring-atom reconstruction.
* ``Baeyer-Villiger_step2``: concerted peroxide rearrangement.
* ``Singlet_Carbene_Intra_Disproportionation``: carbene H-shift.
* ``Korcek_step1``: keto-peroxide ring closure + H transfer.
* ``Intra_OH_migration``: early-TS ring closure for OH migration.
* ``intra_substitutionS_isomerization``: concerted S bond-swap.
* ``Retroene``: 6-membered ring fragmentation.

All builders return ``None`` when the motif cannot be identified
unambiguously or the resulting geometry collides.
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from arc.common import get_single_bond_length
from arc.job.adapters.ts.linear_utils.geom_utils import bfs_fragment, dihedral_deg, mol_to_adjacency, rotate_atoms
from arc.job.adapters.ts.linear_utils.isomerization import ring_closure_xyz
from arc.species.species import colliding_atoms

if TYPE_CHECKING:
    from arc.molecule.molecule import Molecule


logger = logging.getLogger(__name__)

PAULING_DELTA: float = 0.42


# ---------------------------------------------------------------------------
# Geometry helpers (thin wrappers over geom_utils where possible)
# ---------------------------------------------------------------------------

def _dihedral_angle(coords: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    """
    Return the dihedral angle A-B-C-D in degrees (-180, 180].
    Delegates to :func:`geom_utils.dihedral_deg`.
    """
    return dihedral_deg(coords[a], coords[b], coords[c], coords[d])


def _rotate_fragment(coords: np.ndarray,
                     axis_origin: int,
                     axis_end: int,
                     angle_deg: float,
                     moving_atoms: Set[int],
                     ) -> np.ndarray:
    """
    Rotate *moving_atoms* by *angle_deg* around the axis origin→end.
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

        Cα  ----  Hα
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

    adj = mol_to_adjacency(uni_mol)
    atom_to_idx = {atom: idx for idx, atom in enumerate(uni_mol.atoms)}
    bond_orders: Dict[Tuple[int, int], float] = {}
    for atom in uni_mol.atoms:
        ia = atom_to_idx[atom]
        for nbr, bond in atom.bonds.items():
            ib = atom_to_idx[nbr]
            key = (min(ia, ib), max(ia, ib))
            bond_orders[key] = bond.order

    # --- Step 1: Identify the 6-membered ring atoms ---
    c_carb = o_double = o_hydroxyl = h_hydroxyl = None

    for i, sym in enumerate(symbols):
        if sym != 'C':
            continue
        o_neighbors = [j for j in adj[i] if symbols[j] == 'O']
        if len(o_neighbors) < 2:
            continue
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

    c_beta_candidates = [j for j in adj[c_carb] if symbols[j] == 'C']
    if not c_beta_candidates:
        return None
    c_beta = c_beta_candidates[0]

    c_alpha_candidates = [j for j in adj[c_beta] if symbols[j] == 'C' and j != c_carb]
    if not c_alpha_candidates:
        return None
    c_alpha = c_alpha_candidates[0]

    h_alpha_candidates = [j for j in adj[c_alpha] if symbols[j] == 'H']
    if not h_alpha_candidates:
        return None

    coords = np.array(uni_xyz['coords'], dtype=float)

    # Pick H_alpha closest to eclipsed (smallest |dihedral Hα-Cα-Cβ-Ccarb|);
    # this is the H that lands in the 6-membered ring plane.
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

    # 2a. Cα-Cβ-Ccarb-Ooh → 0° (bring hydroxyl syn to Cα).
    dih_1 = _dihedral_angle(coords, c_alpha, c_beta, c_carb, o_hydroxyl)
    frag_1 = bfs_fragment(adj, c_carb, block={c_beta})
    coords = _rotate_fragment(coords, axis_origin=c_beta, axis_end=c_carb, angle_deg=-dih_1, moving_atoms=frag_1)

    # 2b. Cβ-Ccarb-Ooh-Hoh → 0° (fold Hoh toward Cα).
    dih_2 = _dihedral_angle(coords, c_beta, c_carb, o_hydroxyl, h_hydroxyl)
    frag_2 = bfs_fragment(adj, o_hydroxyl, block={c_carb})
    coords = _rotate_fragment(coords, axis_origin=c_carb, axis_end=o_hydroxyl, angle_deg=-dih_2, moving_atoms=frag_2)

    # 2c. Scan Hα around Cα-Cβ in 5° steps; pick the angle minimizing Hα-Hoh.
    frag_3 = bfs_fragment(adj, c_alpha, block={c_beta})
    best_angle = 0.0
    best_dist = float('inf')
    for angle_step in range(72):
        angle = angle_step * 5.0
        trial = _rotate_fragment(coords, axis_origin=c_beta, axis_end=c_alpha, angle_deg=angle, moving_atoms=frag_3)
        d = float(np.linalg.norm(trial[h_alpha] - trial[h_hydroxyl]))
        if d < best_dist:
            best_dist = d
            best_angle = angle
    coords = _rotate_fragment(coords, axis_origin=c_beta, axis_end=c_alpha, angle_deg=best_angle, moving_atoms=frag_3)

    # --- Step 3: Set TS bond distances ---

    # 3a. Break Cβ-Ccarb first so the carboxyl fragment clears room for the H atoms.
    sbl_cc = get_single_bond_length('C', 'C') or 1.54
    d_cc_break_target = sbl_cc + 2.0 * PAULING_DELTA  # ~2.38 Å
    frag_carboxyl = bfs_fragment(adj, c_carb, block={c_beta})
    coords = _set_bond_distance(coords, fixed=c_beta, mobile=c_carb, target_dist=d_cc_break_target, mobile_frag=frag_carboxyl)

    # 3b. Shorten Cα-Cβ (forming C=C).
    d_cc_form_target = sbl_cc * 0.93  # ~1.43 Å
    frag_alpha = bfs_fragment(adj, c_alpha, block={c_beta})
    coords = _set_bond_distance(coords, fixed=c_beta, mobile=c_alpha, target_dist=d_cc_form_target, mobile_frag=frag_alpha)

    # 3c. Triangulate Hα and Hoh into the TS H-H bridge.
    sbl_ch = get_single_bond_length('C', 'H') or 1.09
    sbl_oh = get_single_bond_length('O', 'H') or 0.97
    sbl_hh = get_single_bond_length('H', 'H') or 0.74
    d_ch_target = sbl_ch + PAULING_DELTA    # ~1.51 Å
    d_oh_target = sbl_oh + PAULING_DELTA    # ~1.39 Å
    d_hh_target = sbl_hh * 1.12             # ~0.83 Å

    mid_hh = 0.5 * (coords[h_alpha] + coords[h_hydroxyl])
    dir_ca_to_mid = mid_hh - coords[c_alpha]
    d_ca_mid = float(np.linalg.norm(dir_ca_to_mid))
    if d_ca_mid > 1e-6:
        dir_ca_to_mid /= d_ca_mid
    else:
        dir_ca_to_mid = coords[h_alpha] - coords[c_alpha]
        dir_ca_to_mid /= max(float(np.linalg.norm(dir_ca_to_mid)), 1e-10)

    coords[h_alpha] = coords[c_alpha] + dir_ca_to_mid * d_ch_target

    dir_o_to_h = coords[h_alpha] - coords[o_hydroxyl]
    d_o_h = float(np.linalg.norm(dir_o_to_h))
    if d_o_h > 1e-6:
        dir_o_to_h /= d_o_h
    coords[h_hydroxyl] = coords[o_hydroxyl] + dir_o_to_h * d_oh_target

    # Fine-tune both H positions along the Hα-Hoh axis to hit d_hh_target.
    hh_vec = coords[h_hydroxyl] - coords[h_alpha]
    hh_dist = float(np.linalg.norm(hh_vec))
    if hh_dist > 1e-6:
        hh_mid = 0.5 * (coords[h_alpha] + coords[h_hydroxyl])
        hh_dir = hh_vec / hh_dist
        coords[h_alpha] = hh_mid - hh_dir * d_hh_target * 0.5
        coords[h_hydroxyl] = hh_mid + hh_dir * d_hh_target * 0.5

    ts_xyz = {'symbols': symbols,
              'isotopes': uni_xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
              'coords': tuple(tuple(float(x) for x in row) for row in coords)}
    return ts_xyz


# ---------------------------------------------------------------------------
# 1,3_sigmatropic_rearrangement bespoke builder
# ---------------------------------------------------------------------------

def build_1_3_sigmatropic_rearrangement_ts(r_xyz: dict,
                                           r_mol: 'Molecule',
                                           breaking_bonds: List[Tuple[int, int]],
                                           forming_bonds: List[Tuple[int, int]],
                                           ) -> Optional[dict]:
    """
    Build a compact TS for a 1,3-sigmatropic rearrangement.

    In a [1,3]-sigmatropic shift one atom (the *migrating atom*) moves
    from the *origin* to the *target* via the allylic/heteroallylic
    bridge. The TS has the migrating atom positioned at calibrated
    distances from both origin and target.

    All ring atoms are reconstructed via two-sphere placement to
    maintain a physically sane ring geometry rather than only moving
    the migrating atom and leaving others at reactant positions.

    Motif identification (from ``breaking_bonds`` / ``forming_bonds``):

    * Exactly one breaking bond ``(origin, migrating)`` and exactly one
      forming bond ``(target, migrating)`` must share a common atom (the
      migrating atom).
    * The function returns ``None`` if these conditions are not met or
      if any index is out of range.

    Calibrated distances (specific to 1,3-sigmatropic shifts):

    * Breaking side: ``sbl + 0.77 Å`` (calibrated to ~2.24 Å for C–N).
    * Forming side: ``sbl + 0.38 Å`` (calibrated to ~1.85 Å for N–C).
    * Bridge: ``sbl + 0.12 Å`` (calibrated to ~1.66 Å for C–C).

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

    # The migrating atom is common to both the breaking and forming bond.
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
    sbl_break = get_single_bond_length(symbols[origin], symbols[migrating]) or 1.5
    sbl_form = get_single_bond_length(symbols[target], symbols[migrating]) or 1.5
    _SIGMA_BREAK_STRETCH = 0.77
    _SIGMA_FORM_STRETCH = 0.38
    d_break_target = sbl_break + _SIGMA_BREAK_STRETCH
    d_form_target = sbl_form + _SIGMA_FORM_STRETCH

    # Bridge = heavy neighbor of the migrating atom that is neither origin nor target.
    bridge = None
    for nbr in r_mol.atoms[migrating].bonds.keys():
        ni = atom_to_idx[nbr]
        if symbols[ni] != 'H' and ni != origin and ni != target:
            bridge = ni
            break

    ring_atoms = {migrating, origin, target}
    if bridge is not None:
        ring_atoms.add(bridge)
    for idx in list(ring_atoms):
        for nbr in r_mol.atoms[idx].bonds.keys():
            ni = atom_to_idx[nbr]
            if symbols[ni] != 'H' and ni not in ring_atoms:
                nbr_ring_count = sum(
                    1 for nbr2 in r_mol.atoms[ni].bonds.keys()
                    if atom_to_idx[nbr2] in ring_atoms)
                if nbr_ring_count >= 2:
                    ring_atoms.add(ni)

    # Reactant ring-plane normal is reused as the perpendicular for two-sphere placements.
    ring_list = sorted(ring_atoms)
    if len(ring_list) >= 3:
        ring_pts = coords[ring_list]
        centroid = ring_pts.mean(axis=0)
        centered = ring_pts - centroid
        _, _, vt = np.linalg.svd(centered)
        ring_normal = vt[-1]
    else:
        ring_normal = np.array([0.0, 0.0, 1.0])

    def _two_sphere_place(center1, r1, center2, r2, ref_pos):
        """Place a point at the two-sphere intersection, on the same
        side as ``ref_pos`` relative to the center1→center2 axis."""
        axis = center2 - center1
        axis_d = float(np.linalg.norm(axis))
        if axis_d < 1e-6:
            return None
        axis_h = axis / axis_d
        if axis_d <= r1 + r2:
            x = (axis_d ** 2 + r1 ** 2 - r2 ** 2) / (2.0 * axis_d)
            h_sq = r1 ** 2 - x ** 2
            h_p = float(np.sqrt(max(h_sq, 0.0)))
        else:
            x = r1
            h_p = 0.0
        proj = center1 + axis_h * np.dot(ref_pos - center1, axis_h)
        ref_offset = ref_pos - proj
        ref_norm = float(np.linalg.norm(ref_offset))
        if ref_norm > 1e-8:
            p_hat = ref_offset / ref_norm
        else:
            p_hat = ring_normal.copy()
            p_hat -= axis_h * np.dot(p_hat, axis_h)
            pn = float(np.linalg.norm(p_hat))
            if pn > 1e-8:
                p_hat /= pn
            else:
                arb = np.array([1.0, 0.0, 0.0])
                if abs(float(np.dot(axis_h, arb))) > 0.9:
                    arb = np.array([0.0, 1.0, 0.0])
                p_hat = np.cross(axis_h, arb)
                p_hat /= max(float(np.linalg.norm(p_hat)), 1e-10)
        return center1 + axis_h * x + p_hat * h_p

    def _move_atom_with_h(idx, new_pos):
        """Move atom ``idx`` to ``new_pos`` and translate its H-only
        neighbors rigidly."""
        disp = new_pos - coords[idx]
        moving = {idx}
        for nbr in r_mol.atoms[idx].bonds.keys():
            ni = atom_to_idx[nbr]
            if symbols[ni] == 'H':
                moving.add(ni)
        for k in moving:
            coords[k] += disp

    # Step 1: Place migrating atom at calibrated breaking/forming distances.
    new_mig = _two_sphere_place(
        coords[origin], d_break_target,
        coords[target], d_form_target,
        coords[migrating])
    if new_mig is None:
        return None
    _move_atom_with_h(migrating, new_mig)

    # Step 2: Place bridge at calibrated distance from migrating, reactant distance from target.
    if bridge is not None:
        sbl_bridge = get_single_bond_length(
            symbols[bridge], symbols[migrating]) or 1.5
        d_bridge_target = sbl_bridge + 0.12  # ~1.66 Å for C-C
        d_bridge_target_side = float(np.linalg.norm(
            np.array(r_xyz['coords'][bridge]) - np.array(r_xyz['coords'][target])))
        new_bridge = _two_sphere_place(
            coords[migrating], d_bridge_target,
            coords[target], d_bridge_target_side,
            coords[bridge])
        if new_bridge is not None:
            _move_atom_with_h(bridge, new_bridge)

    # Step 3: Place remaining ring atoms at reactant distances from their two ring neighbors.
    remaining = ring_atoms - {migrating, origin, target}
    if bridge is not None:
        remaining -= {bridge}
    for rem_idx in remaining:
        ring_nbrs = []
        for nbr in r_mol.atoms[rem_idx].bonds.keys():
            ni = atom_to_idx[nbr]
            if ni in ring_atoms and symbols[ni] != 'H':
                ring_nbrs.append(ni)
        if len(ring_nbrs) >= 2:
            n1, n2 = ring_nbrs[0], ring_nbrs[1]
            d1 = float(np.linalg.norm(
                np.array(r_xyz['coords'][rem_idx]) - np.array(r_xyz['coords'][n1])))
            d2 = float(np.linalg.norm(
                np.array(r_xyz['coords'][rem_idx]) - np.array(r_xyz['coords'][n2])))
            new_rem = _two_sphere_place(
                coords[n1], d1, coords[n2], d2, coords[rem_idx])
            if new_rem is not None:
                _move_atom_with_h(rem_idx, new_rem)

    # Step 4: Light planarity enforcement using the ORIGINAL ring-plane normal
    # (computed before any atoms moved).
    _PLANARITY_FRACTION = 0.50
    if len(ring_list) >= 3:
        current_ring_pts = coords[ring_list]
        current_centroid = current_ring_pts.mean(axis=0)
        for ri in ring_list:
            offset = float(np.dot(coords[ri] - current_centroid, ring_normal))
            coords[ri] -= offset * _PLANARITY_FRACTION * ring_normal
            for nbr in r_mol.atoms[ri].bonds.keys():
                hi = atom_to_idx[nbr]
                if symbols[hi] == 'H':
                    h_offset = float(np.dot(
                        coords[hi] - current_centroid, ring_normal))
                    coords[hi] -= h_offset * _PLANARITY_FRACTION * ring_normal

    # Guard: reject if any heavy-heavy bond collapsed below 0.9 Å.
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

    ts_xyz = {'symbols': symbols,
              'isotopes': r_xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
              'coords': tuple(tuple(float(c) for c in row) for row in coords)}
    return ts_xyz


# ---------------------------------------------------------------------------
# Baeyer-Villiger_step2 bespoke builder
# ---------------------------------------------------------------------------

def build_baeyer_villiger_step2_ts(uni_xyz: dict,
                                   uni_mol: 'Molecule',
                                   split_bonds: List[Tuple[int, int]],
                                   ) -> Optional[dict]:
    """
    Build a concerted TS for the Baeyer-Villiger step 2 rearrangement.

    In BV step 2, a Criegee intermediate rearranges concertedly:
    the O-O peroxide bond breaks, an alkyl group on the *other*
    (non-carbonyl) side of the peroxide migrates from its parent
    carbon toward the peroxide oxygen, forming a new C-O bond while
    the parent carbon loses the migrating group.

    Motif identification:

    1. ``split_bonds`` must contain at least one O-O peroxide cut.
    2. One peroxide O (*carbonyl-side O*) is bonded to a C with a C=O.
    3. The other peroxide O (*other-side O*) is bonded to a *quaternary-like C*
       (the origin of the migrating group).
    4. The migrating group root is a non-O, non-H neighbor of the
       quaternary C. When there are multiple candidates (e.g. two
       CH₃ groups), each is tried and the first non-colliding guess is returned.
    5. If any of these conditions are not met, return ``None``.

    Geometry construction (calibrated to DFT-curated TS):

    1. Stretch O-O to ~2.02 Å (``sbl + 2 × 0.27``; calibrated from the curated TS).
    2. Stretch C_parent-C_migrating to ~2.30 Å (``sbl + 0.76``;
       calibrated — the migrating group is leaving its parent).
    3. Contract C_migrating toward O_other to ~2.16 Å
       (``sbl(C,O) + 0.73``; calibrated — the migrating group is
       approaching the peroxide O through which it migrates).
    4. If the guess has atom collisions, try the next candidate.

    Args:
        uni_xyz: Criegee intermediate XYZ dict.
        uni_mol: Criegee intermediate RMG Molecule.
        split_bonds: Fragmentation bond cuts (from :func:`find_split_bonds_by_fragmentation`).

    Returns:
        TS XYZ dict, or ``None`` if the BV motif cannot be identified.
    """
    symbols = uni_xyz['symbols']
    n_atoms = len(symbols)

    adj = mol_to_adjacency(uni_mol)
    atom_to_idx = {atom: idx for idx, atom in enumerate(uni_mol.atoms)}
    bond_orders: Dict[Tuple[int, int], float] = {}
    for atom in uni_mol.atoms:
        ia = atom_to_idx[atom]
        for nbr, bond in atom.bonds.items():
            ib = atom_to_idx[nbr]
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

    # Step 2: Identify carbonyl-side O vs. other-side O.
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

    # Step 3: Identify the quaternary-like C on the other side of the peroxide
    # (parent of the migrating group).
    c_parent = None
    for nbr_idx in adj[o_other_side]:
        if symbols[nbr_idx] == 'C':
            c_parent = nbr_idx
            break
    if c_parent is None:
        return None

    # Step 4: Candidate migrating groups = non-O, non-H neighbors of c_parent.
    mig_candidates = []
    for nbr_idx in adj[c_parent]:
        if nbr_idx == o_other_side:
            continue
        if symbols[nbr_idx] in ('O', 'H'):
            continue
        mig_candidates.append(nbr_idx)
    if not mig_candidates:
        return None

    # Calibrated TS target distances (from DFT-curated BV step-2 reference).
    sbl_oo = get_single_bond_length('O', 'O') or 1.48
    sbl_cc = get_single_bond_length('C', 'C') or 1.54
    sbl_co = get_single_bond_length('C', 'O') or 1.43

    _BV_OO_STRETCH = 0.27    # Å per side (total ≈ 2 × 0.27 above sbl)
    _BV_CC_STRETCH = 0.76    # Å above sbl for migrating-group departure
    _BV_CO_STRETCH = 0.73    # Å above sbl(C,O) for migrating-group approach

    d_oo_target = sbl_oo + 2.0 * _BV_OO_STRETCH
    d_cc_target = sbl_cc + _BV_CC_STRETCH
    d_approach_target = sbl_co + _BV_CO_STRETCH

    # Step 5: Identify the OH oxygen on c_parent — it contracts into a new C=O in the TS.
    o_hydroxyl = None
    for nbr_idx in adj[c_parent]:
        if nbr_idx == o_other_side:
            continue
        if symbols[nbr_idx] != 'O':
            continue
        h_on_o = [k for k in adj[nbr_idx] if symbols[k] == 'H']
        if h_on_o:
            o_hydroxyl = nbr_idx
            break

    _BV_CO_SHORTEN_TARGET = 1.28  # Å, calibrated from DFT TS (1.279)

    best_ts: Optional[dict] = None
    best_score = float('inf')

    for c_migrating in mig_candidates:
        coords = np.array(uni_xyz['coords'], dtype=float).copy()

        # 4a. Stretch the O-O peroxide bond.
        frag_other = bfs_fragment(adj, o_other_side, block={o_carb_side})
        coords = _set_bond_distance(coords, fixed=o_carb_side,
                                    mobile=o_other_side,
                                    target_dist=d_oo_target,
                                    mobile_frag=frag_other)

        # 4b. Triangulate C_migrating between C_parent and O_other simultaneously
        # (a sequential stretch-then-contract would undo itself).
        frag_mig = bfs_fragment(adj, c_migrating, block={c_parent})
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

        # 4c. Shorten C_parent-O_hydroxyl toward calibrated TS target (~1.28 Å);
        # only the O and its H(s) move.
        if o_hydroxyl is not None:
            frag_oh = bfs_fragment(adj, o_hydroxyl, block={c_parent})
            coords = _set_bond_distance(
                coords, fixed=c_parent, mobile=o_hydroxyl,
                target_dist=_BV_CO_SHORTEN_TARGET,
                mobile_frag=frag_oh)

        # 4d. Concerted H migration from O_hydroxyl to the carbonyl O
        # (~1.40 Å from each, calibrated).
        _BV_H_TRANSFER_TARGET = 1.40
        if o_hydroxyl is not None:
            o_carbonyl_dbl = None
            for nbr_idx in adj[c_carbonyl]:
                if nbr_idx == o_carb_side:
                    continue
                if symbols[nbr_idx] != 'O':
                    continue
                key = (min(c_carbonyl, nbr_idx), max(c_carbonyl, nbr_idx))
                if bond_orders.get(key, 1.0) >= 1.5:
                    o_carbonyl_dbl = nbr_idx
                    break
            h_on_oh = [k for k in adj[o_hydroxyl] if symbols[k] == 'H']
            if o_carbonyl_dbl is not None and h_on_oh:
                h_idx = h_on_oh[0]
                p1 = coords[o_hydroxyl]
                p2 = coords[o_carbonyl_dbl]
                ax = p2 - p1
                ax_d = float(np.linalg.norm(ax))
                if ax_d > 1e-6:
                    ax_h = ax / ax_d
                    r1 = _BV_H_TRANSFER_TARGET
                    r2 = _BV_H_TRANSFER_TARGET
                    if ax_d <= r1 + r2:
                        xp = ax_d / 2.0
                        hp_sq = r1 ** 2 - xp ** 2
                        hp = float(np.sqrt(max(hp_sq, 0.0)))
                    else:
                        xp = r1
                        hp = 0.0
                    h_pos = coords[h_idx]
                    proj = p1 + ax_h * np.dot(h_pos - p1, ax_h)
                    perp = h_pos - proj
                    pn = float(np.linalg.norm(perp))
                    if pn > 1e-8:
                        ph = perp / pn
                    else:
                        arb = np.array([1.0, 0.0, 0.0])
                        if abs(float(np.dot(ax_h, arb))) > 0.9:
                            arb = np.array([0.0, 1.0, 0.0])
                        ph = np.cross(ax_h, arb)
                        ph /= max(float(np.linalg.norm(ph)), 1e-10)
                    coords[h_idx] = p1 + ax_h * xp + ph * hp
            elif h_on_oh:
                # Fallback: stretch O-H along existing direction.
                for h_idx in h_on_oh:
                    oh_vec = coords[h_idx] - coords[o_hydroxyl]
                    oh_dist = float(np.linalg.norm(oh_vec))
                    if oh_dist > 1e-6 and oh_dist < _BV_H_TRANSFER_TARGET:
                        oh_hat = oh_vec / oh_dist
                        coords[h_idx] = (coords[o_hydroxyl]
                                         + oh_hat * _BV_H_TRANSFER_TARGET)

        ts_xyz = {'symbols': symbols,
                  'isotopes': uni_xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
                  'coords': tuple(tuple(float(c) for c in row) for row in coords)}
        if colliding_atoms(ts_xyz):
            continue

        # Score = sum of squared deviations from calibrated targets.
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


# ---------------------------------------------------------------------------
# Singlet_Carbene_Intra_Disproportionation bespoke builder
# ---------------------------------------------------------------------------

def build_singlet_carbene_intra_disproportionation_ts(r_xyz: dict,
                                                      r_mol: 'Molecule',
                                                      ) -> Optional[dict]:
    """
    Build a TS for singlet carbene intramolecular H-shift.

    In this reaction, a singlet carbene C (divalent, lone pair) on a
    ring accepts an H from an adjacent CH₂ center. The TS has the
    migrating H between the donor C and the carbene C at
    Pauling-triangulated distances.

    Motif identification (from molecular graph):

    * Find the carbene C: a carbon with exactly 2 heavy neighbors
      and 0 H neighbors (consistent with a singlet carbene).
    * Find the donor: a neighbor of the carbene C that has at least
      1 bonded H.
    * Select the migrating H: the H on the donor that is closest to
      the carbene C.

    Returns ``None`` if the motif cannot be identified.
    """
    symbols = r_xyz['symbols']
    n_atoms = len(symbols)
    atom_to_idx = {atom: idx for idx, atom in enumerate(r_mol.atoms)}

    # Carbene C: 2 heavy neighbours, 0 H neighbours.
    carbene_c = None
    for atom in r_mol.atoms:
        idx = atom_to_idx[atom]
        if symbols[idx] != 'C':
            continue
        heavy = [n for n in atom.bonds.keys() if n.element.symbol != 'H']
        h_nbrs = [n for n in atom.bonds.keys() if n.element.symbol == 'H']
        if len(heavy) == 2 and len(h_nbrs) == 0:
            carbene_c = idx
            break
    if carbene_c is None:
        return None

    # Prefer the donor with most H atoms (CH₂ over CH); singlet carbene
    # disproportionation typically draws from CH₂.
    donor_c = None
    best_h_count = 0
    for nbr in r_mol.atoms[carbene_c].bonds.keys():
        ni = atom_to_idx[nbr]
        if symbols[ni] == 'H':
            continue
        h_on_donor = [n for n in nbr.bonds.keys() if n.element.symbol == 'H']
        if len(h_on_donor) > best_h_count:
            best_h_count = len(h_on_donor)
            donor_c = ni
    if donor_c is None:
        return None

    coords = np.array(r_xyz['coords'], dtype=float).copy()
    donor_h_list = []
    for nbr in r_mol.atoms[donor_c].bonds.keys():
        ni = atom_to_idx[nbr]
        if symbols[ni] == 'H':
            donor_h_list.append(ni)
    if not donor_h_list:
        return None

    sbl_ch = get_single_bond_length('C', 'H') or 1.09
    d_target = sbl_ch + PAULING_DELTA  # ~1.51 Å

    donor_pos = coords[donor_c]
    carbene_pos = coords[carbene_c]
    dc_vec = carbene_pos - donor_pos
    dc_dist = float(np.linalg.norm(dc_vec))
    if dc_dist < 1e-6:
        return None
    dc_hat = dc_vec / dc_dist

    if dc_dist <= 2.0 * d_target:
        x = dc_dist / 2.0
        h_sq = d_target ** 2 - x ** 2
        h_perp = float(np.sqrt(max(h_sq, 0.0)))
    else:
        x = d_target
        h_perp = 0.0

    # For a CH₂ donor the migrating H must sit on the OPPOSITE face from
    # the retained H — the two H's end up on opposite faces of the ring,
    # which is the chemically correct TS.
    if len(donor_h_list) == 2:
        d0 = float(np.linalg.norm(coords[donor_h_list[0]] - carbene_pos))
        d1 = float(np.linalg.norm(coords[donor_h_list[1]] - carbene_pos))
        if d0 <= d1:
            migrating_h = donor_h_list[0]
            bonded_h = donor_h_list[1]
        else:
            migrating_h = donor_h_list[1]
            bonded_h = donor_h_list[0]
        # Negate the bonded-H offset to put the migrating H on the opposite face.
        bonded_pos = coords[bonded_h]
        proj = donor_pos + dc_hat * np.dot(bonded_pos - donor_pos, dc_hat)
        perp = bonded_pos - proj
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm > 1e-8:
            perp_hat = -(perp / perp_norm)
        else:
            arb = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(dc_hat, arb))) > 0.9:
                arb = np.array([0.0, 1.0, 0.0])
            perp_hat = np.cross(dc_hat, arb)
            perp_hat /= max(float(np.linalg.norm(perp_hat)), 1e-10)
    else:
        migrating_h = donor_h_list[0]
        h_pos = coords[migrating_h]
        proj = donor_pos + dc_hat * np.dot(h_pos - donor_pos, dc_hat)
        perp = h_pos - proj
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm > 1e-8:
            perp_hat = perp / perp_norm
        else:
            arb = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(dc_hat, arb))) > 0.9:
                arb = np.array([0.0, 1.0, 0.0])
            perp_hat = np.cross(dc_hat, arb)
            perp_hat /= max(float(np.linalg.norm(perp_hat)), 1e-10)

    coords[migrating_h] = donor_pos + dc_hat * x + perp_hat * h_perp

    ts_xyz = {'symbols': symbols,
              'isotopes': r_xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
              'coords': tuple(tuple(float(c) for c in row) for row in coords)}
    return ts_xyz


# ---------------------------------------------------------------------------
# Korcek_step1 (Intra_RH_Add_Exocyclic) bespoke builder
# ---------------------------------------------------------------------------


def build_korcek_step1_ts(r_xyz: dict,
                          r_mol: 'Molecule') -> Optional[dict]:
    """
    Build a TS for Korcek step 1 ring closure with H transfer.

    In Korcek step 1, an open-chain keto-peroxide cyclizes:
    the carbonyl C forms a bond to the terminal peroxide O (closing a
    6-membered ring containing the O-O peroxide), while the peroxide H
    transfers to the carbonyl O.

    Motif identification:

    * Find the aldehyde / ketone: a C bonded to an O via a double bond.
    * Find the peroxide: an O-O pair where at least one O has a bonded H.
    * The terminal peroxide O (the one with H) must be reachable from
      the carbonyl C through a short chain (4-6 bonds), forming a
      5-7 membered ring.

    Returns ``None`` if the motif cannot be identified.
    """
    symbols = r_xyz['symbols']
    n_atoms = len(symbols)
    atom_to_idx = {atom: idx for idx, atom in enumerate(r_mol.atoms)}

    adj = mol_to_adjacency(r_mol)
    bond_orders: Dict[Tuple[int, int], float] = {}
    for atom in r_mol.atoms:
        ia = atom_to_idx[atom]
        for nbr, bond in atom.bonds.items():
            ib = atom_to_idx[nbr]
            key = (min(ia, ib), max(ia, ib))
            bond_orders[key] = bond.order

    # Step 1: Collect all carbonyl C=O pairs.
    carbonyl_pairs = []
    for atom in r_mol.atoms:
        c_idx = atom_to_idx[atom]
        if symbols[c_idx] != 'C':
            continue
        for nbr, bond in atom.bonds.items():
            o_idx = atom_to_idx[nbr]
            if symbols[o_idx] == 'O' and bond.order >= 1.5:
                carbonyl_pairs.append((c_idx, o_idx))

    # Step 2: Find the peroxide O-O bond.
    peroxide_pair = None
    for atom in r_mol.atoms:
        ia = atom_to_idx[atom]
        if symbols[ia] != 'O':
            continue
        for nbr in atom.bonds.keys():
            ib = atom_to_idx[nbr]
            if symbols[ib] == 'O' and ib > ia:
                peroxide_pair = (ia, ib)
                break
        if peroxide_pair is not None:
            break
    if peroxide_pair is None:
        return None

    # Terminal peroxide O = the one with H; the other is the internal O.
    o_term = None
    h_peroxide = None
    for o_idx in peroxide_pair:
        for nbr_idx in adj[o_idx]:
            if symbols[nbr_idx] == 'H':
                o_term = o_idx
                h_peroxide = nbr_idx
                break
        if o_term is not None:
            break
    if o_term is None:
        return None

    # Step 3: BFS from o_term along the heavy-atom chain to find the nearest carbonyl C.
    best_path, best_carb_c, best_o_dbl = None, None, None
    for c_idx, o_dbl_idx in carbonyl_pairs:
        visited = {o_term}
        queue = deque([(o_term, [o_term])])
        while queue:
            curr, path = queue.popleft()
            if curr == c_idx:
                if best_path is None or len(path) < len(best_path):
                    best_path = path
                    best_carb_c = c_idx
                    best_o_dbl = o_dbl_idx
                break
            for nxt in adj[curr]:
                if nxt in visited:
                    continue
                if symbols[nxt] == 'H':
                    continue
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
    if best_path is None or len(best_path) < 4 or len(best_path) > 8:
        return None

    # Step 4: Fold the chain via ring_closure_xyz.
    forming_bond = (best_carb_c, o_term)
    sbl_co = get_single_bond_length('C', 'O') or 1.43
    d_ring_target = sbl_co + PAULING_DELTA  # ~1.85 Å
    rc_xyz = ring_closure_xyz(
        r_xyz, r_mol, forming_bond=forming_bond,
        target_distance=d_ring_target)
    if rc_xyz is None:
        return None
    if colliding_atoms(rc_xyz):
        return None

    # Step 5: Place the peroxide H equidistant (~1.39 Å) between o_term and o_dbl
    # via two-sphere intersection (it is transferring from o_term to o_dbl).
    coords = np.array(rc_xyz['coords'], dtype=float)
    sbl_oh = get_single_bond_length('O', 'H') or 0.97
    d_h_target = sbl_oh + PAULING_DELTA  # ~1.39 Å
    o_term_pos = coords[o_term]
    o_dbl_pos = coords[best_o_dbl]
    h_pos = coords[h_peroxide]

    ax = o_dbl_pos - o_term_pos
    ax_d = float(np.linalg.norm(ax))
    if ax_d > 1e-6:
        ax_h = ax / ax_d
        r1 = d_h_target
        r2 = d_h_target
        if ax_d <= r1 + r2:
            xp = ax_d / 2.0
            hp_sq = r1 ** 2 - xp ** 2
            hp = float(np.sqrt(max(hp_sq, 0.0)))
        else:
            xp = r1
            hp = 0.0
        proj = o_term_pos + ax_h * np.dot(h_pos - o_term_pos, ax_h)
        perp = h_pos - proj
        pn = float(np.linalg.norm(perp))
        if pn > 1e-8:
            ph = perp / pn
        else:
            arb = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(ax_h, arb))) > 0.9:
                arb = np.array([0.0, 1.0, 0.0])
            ph = np.cross(ax_h, arb)
            ph /= max(float(np.linalg.norm(ph)), 1e-10)
        coords[h_peroxide] = o_term_pos + ax_h * xp + ph * hp

    ts_xyz = {'symbols': rc_xyz['symbols'],
              'isotopes': rc_xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
              'coords': tuple(tuple(float(c) for c in row) for row in coords)}
    if colliding_atoms(ts_xyz):
        return None
    return ts_xyz


# ---------------------------------------------------------------------------
# Intra_OH_migration bespoke builder
# ---------------------------------------------------------------------------

def build_intra_oh_migration_ts(r_xyz: dict,
                                r_mol: 'Molecule',
                                breaking_bonds: List[Tuple[int, int]],
                                forming_bonds: List[Tuple[int, int]]) -> Optional[dict]:
    """
    Build an early TS for intra-OH migration.

    In ``Intra_OH_migration``, an OH migrates from one heavy atom to
    another through a small ring (typically 3–5 membered). The TS is
    early: the migrating H is barely displaced from the donor O, and
    the forming C–O bond is still long (~2.1 Å). Generic Z-matrix
    interpolation at weight 0.5 over-contracts the ring and places
    spectator H atoms incorrectly.

    This builder uses ``ring_closure_xyz`` to fold the chain at a
    calibrated early-TS forming-bond distance, then adjusts the
    breaking O–O bond and orients spectator H atoms on the radical
    center perpendicular to the forming-bond axis.

    Args:
        r_xyz: Reactant XYZ dict.
        r_mol: Reactant RMG Molecule.
        breaking_bonds: Bonds that break (should contain the O–O).
        forming_bonds: Bonds that form (should contain the C–O).

    Returns:
        TS XYZ dict, or ``None`` if the motif cannot be identified.
    """
    bb = list(breaking_bonds or [])
    fb = list(forming_bonds or [])
    if not bb or not fb:
        return None

    symbols = r_xyz['symbols']
    n_atoms = len(symbols)
    atom_to_idx = {atom: idx for idx, atom in enumerate(r_mol.atoms)}

    forming_co = None
    for a, b in fb:
        sa, sb = symbols[a], symbols[b]
        if (sa == 'C' and sb == 'O') or (sa == 'O' and sb == 'C'):
            forming_co = (a, b) if sa == 'C' else (b, a)  # (C, O)
            break
    if forming_co is None:
        return None
    c_radical, o_migrating = forming_co

    breaking_oo = None
    for a, b in bb:
        if symbols[a] == 'O' and symbols[b] == 'O':
            breaking_oo = (a, b)
            break
    if breaking_oo is None:
        return None

    # The migrating O must participate in both bonds; swap if needed.
    if o_migrating not in breaking_oo:
        if forming_co[1] in breaking_oo:
            o_migrating = forming_co[1]
        else:
            return None
    o_other = breaking_oo[0] if breaking_oo[1] == o_migrating else breaking_oo[1]

    # Step 1: Ring closure at the calibrated early-TS C-O distance.
    # ~2.08 Å is larger than Pauling sbl+delta (1.85 Å) because this is an early TS.
    _OH_MIG_CO_TARGET = 2.08

    rc_xyz = ring_closure_xyz(
        r_xyz, r_mol, forming_bond=(c_radical, o_migrating),
        target_distance=_OH_MIG_CO_TARGET)
    if rc_xyz is None:
        return None

    coords = np.array(rc_xyz['coords'], dtype=float)

    # Step 2: Stretch the breaking O-O toward ~1.90 Å (sbl + PAULING_DELTA).
    sbl_oo = get_single_bond_length('O', 'O') or 1.48
    d_oo_target = sbl_oo + PAULING_DELTA
    oo_vec = coords[o_migrating] - coords[o_other]
    oo_dist = float(np.linalg.norm(oo_vec))
    if oo_dist > 1e-6 and oo_dist < d_oo_target:
        oo_hat = oo_vec / oo_dist
        stretch = (d_oo_target - oo_dist)
        frag = {o_migrating}
        for nbr in r_mol.atoms[o_migrating].bonds.keys():
            ni = atom_to_idx[nbr]
            if symbols[ni] == 'H':
                frag.add(ni)
        for idx in frag:
            coords[idx] += oo_hat * stretch

    # Step 3: Orient spectator H atoms on c_radical perpendicular to the
    # forming-bond axis so they don't lean toward the approaching OH.
    co_vec = coords[o_migrating] - coords[c_radical]
    co_dist = float(np.linalg.norm(co_vec))
    if co_dist > 1e-6:
        co_hat = co_vec / co_dist
        h_on_c = []
        for nbr in r_mol.atoms[c_radical].bonds.keys():
            ni = atom_to_idx[nbr]
            if symbols[ni] == 'H':
                h_on_c.append(ni)
        if len(h_on_c) >= 2:
            backbone_c = None
            for nbr in r_mol.atoms[c_radical].bonds.keys():
                ni = atom_to_idx[nbr]
                if symbols[ni] != 'H' and ni != o_migrating:
                    backbone_c = ni
                    break
            if backbone_c is not None:
                sbl_ch = get_single_bond_length('C', 'H') or 1.09
                for hi in h_on_c:
                    h_vec = coords[hi] - coords[c_radical]
                    h_dist = float(np.linalg.norm(h_vec))
                    if h_dist < 1e-6:
                        continue
                    h_perp = h_vec - co_hat * np.dot(h_vec, co_hat)
                    h_perp_norm = float(np.linalg.norm(h_perp))
                    if h_perp_norm > 1e-8:
                        h_perp_hat = h_perp / h_perp_norm
                    else:
                        arb = np.array([1.0, 0.0, 0.0])
                        if abs(float(np.dot(co_hat, arb))) > 0.9:
                            arb = np.array([0.0, 1.0, 0.0])
                        h_perp_hat = np.cross(co_hat, arb)
                        h_perp_hat /= max(float(np.linalg.norm(h_perp_hat)), 1e-10)
                    # Slight inward tilt (away from the forming bond); cos(109.5°) ≈ -0.33.
                    tilt = -0.10
                    coords[hi] = (coords[c_radical]
                                  + co_hat * sbl_ch * tilt
                                  + h_perp_hat * sbl_ch * np.sqrt(1.0 - tilt ** 2))

    ts_xyz = {'symbols': symbols,
              'isotopes': r_xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
              'coords': tuple(tuple(float(c) for c in row) for row in coords)}
    if colliding_atoms(ts_xyz):
        return None
    return ts_xyz


# ---------------------------------------------------------------------------
# intra_substitutionS_isomerization bespoke builder
# ---------------------------------------------------------------------------

def build_intra_substitution_s_isomerization_ts(r_xyz: dict,
                                                r_mol: 'Molecule',
                                                breaking_bonds: List[Tuple[int, int]],
                                                forming_bonds: List[Tuple[int, int]]) -> Optional[dict]:
    """
    Build a TS for concerted sulfur bond-swap (intra_substitutionS).

    In ``intra_substitutionS_isomerization``, a sulfur atom migrates
    from one partner to another in a concerted bond-swap: the S-S (or
    S-X) bond breaks while a new C-S bond forms. The migrating S
    carries its bonded fragment (e.g. CH₃) as a rigid unit.

    Generic Z-mat interpolation fails because the migrating S changes
    neighbors drastically, causing atom collisions or spectator-bond
    compression. This builder instead:

    1. Identifies the migrating atom (shared between bb and fb).
    2. Identifies the breaking partner and forming partner.
    3. Computes the rigid fragment bonded to the migrating atom
       (excluding both partners).
    4. Places the migrating atom at a weighted position between its
       reactant location and a Pauling-like TS target, moving the
       rigid fragment along.

    Args:
        r_xyz: Reactant XYZ dict.
        r_mol: Reactant RMG Molecule.
        breaking_bonds: Bonds that break (e.g., S-S).
        forming_bonds: Bonds that form (e.g., C-S).

    Returns:
        TS XYZ dict, or ``None`` if the motif cannot be identified.
    """
    bb = list(breaking_bonds or [])
    fb = list(forming_bonds or [])
    if len(bb) != 1 or len(fb) != 1:
        return None

    # The migrating atom is shared between the breaking and forming bonds.
    bb_set = set(bb[0])
    fb_set = set(fb[0])
    shared = bb_set & fb_set
    if len(shared) != 1:
        return None
    migrating = shared.pop()
    breaking_partner = (bb_set - {migrating}).pop()
    forming_partner = (fb_set - {migrating}).pop()

    symbols = r_xyz['symbols']
    n_atoms = len(symbols)
    coords = np.array(r_xyz['coords'], dtype=float).copy()
    atom_to_idx = {atom: idx for idx, atom in enumerate(r_mol.atoms)}

    # Rigid fragment bonded to the migrating atom, excluding both partners.
    block = {breaking_partner, forming_partner}
    frag: Set[int] = set()
    q: deque = deque([migrating])
    while q:
        node = q.popleft()
        if node in frag or node in block:
            continue
        frag.add(node)
        for nbr in r_mol.atoms[node].bonds:
            ni = atom_to_idx[nbr]
            if ni not in frag and ni not in block:
                q.append(ni)

    sbl_break = get_single_bond_length(
        symbols[migrating], symbols[breaking_partner]) or 2.05
    sbl_form = get_single_bond_length(
        symbols[migrating], symbols[forming_partner]) or 1.81
    d_break_target = sbl_break + PAULING_DELTA  # ~2.47 Å for S-S
    d_form_target = sbl_form + PAULING_DELTA    # ~2.23 Å for C-S

    d_break = float(np.linalg.norm(
        coords[migrating] - coords[breaking_partner]))
    d_form = float(np.linalg.norm(
        coords[migrating] - coords[forming_partner]))

    # Push migrating away from breaking_partner and pull it toward forming_partner.
    displacement = np.zeros(3)

    if d_break > 1e-6 and d_break < d_break_target:
        dir_away = coords[migrating] - coords[breaking_partner]
        dir_away /= d_break
        displacement += dir_away * (d_break_target - d_break)

    if d_form > 1e-6 and d_form > d_form_target:
        dir_toward = coords[forming_partner] - coords[migrating]
        dir_toward /= d_form
        displacement += dir_toward * (d_form - d_form_target)

    for k in frag:
        coords[k] += displacement

    ts_xyz_sub = {
        'symbols': symbols,
        'isotopes': r_xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
        'coords': tuple(tuple(float(c) for c in row) for row in coords),
    }
    if colliding_atoms(ts_xyz_sub):
        return None
    return ts_xyz_sub


# ---------------------------------------------------------------------------
# Retroene bespoke builder
# ---------------------------------------------------------------------------

def build_retroene_ts(r_xyz: dict,
                      r_mol: 'Molecule',
                      breaking_bonds: List[Tuple[int, int]],
                      forming_bonds: List[Tuple[int, int]]) -> Optional[dict]:
    """
    Build a 6-membered-ring TS for retro-ene fragmentation.

    In a Retroene reaction, an ester fragments through a concerted 6-membered ring TS::

         H(mig)
        /      \\
      C(donor)  O(ester)
      |          |
      C(leaving) C(carbonyl)
        \\      /
         O(C=O)

    The migrating H leaves the donor C and arrives at the ester O,
    while the C=O oxygen develops a contact with the leaving C.

    Returns ``None`` if the motif cannot be identified.
    """
    bb = list(breaking_bonds or [])
    fb = list(forming_bonds or [])
    if len(bb) != 2 or len(fb) != 1:
        return None

    symbols = r_xyz['symbols']
    n_atoms = len(symbols)
    coords = np.array(r_xyz['coords'], dtype=float).copy()
    atom_to_idx = {atom: idx for idx, atom in enumerate(r_mol.atoms)}

    # Migrating H is present in one bb entry AND in fb.
    mig_h = None
    donor = None
    other_bb_idx = None
    for ba, bb_bond in enumerate(bb):
        for atom_idx in bb_bond:
            if symbols[atom_idx] == 'H' and atom_idx in fb[0]:
                mig_h = atom_idx
                donor = bb_bond[0] if bb_bond[1] == atom_idx else bb_bond[1]
                other_bb_idx = 1 - ba
                break
        if mig_h is not None:
            break
    if mig_h is None or other_bb_idx is None:
        return None

    # The other breaking bond is the sigma bond (e.g. O3-C4).
    sigma_bb = bb[other_bb_idx]

    leaving_c = None
    ester_o = None
    for candidate in sigma_bb:
        for nbr in r_mol.atoms[candidate].bonds:
            if atom_to_idx[nbr] == donor:
                leaving_c = candidate
                ester_o = sigma_bb[0] if sigma_bb[1] == candidate else sigma_bb[1]
                break
        if leaving_c is not None:
            break
    if leaving_c is None or ester_o is None:
        return None

    # carbonyl_c = non-H, non-leaving-C neighbour of ester_o.
    carbonyl_c = None
    for nbr in r_mol.atoms[ester_o].bonds:
        ni = atom_to_idx[nbr]
        if ni != leaving_c and symbols[ni] != 'H':
            carbonyl_c = ni
            break
    if carbonyl_c is None:
        return None

    # carbonyl_o = double-bonded O on carbonyl_c (not ester_o).
    carbonyl_o = None
    for nbr, bond in r_mol.atoms[carbonyl_c].bonds.items():
        ni = atom_to_idx[nbr]
        if symbols[ni] == 'O' and ni != ester_o and bond.order >= 1.5:
            carbonyl_o = ni
            break
    if carbonyl_o is None:
        return None

    # The acid fragment and the donor stay at reactant positions; only C4 is
    # repositioned via two-sphere triangulation so d(O3,C4) stretches and
    # d(C5,C4) contracts simultaneously. Then ring_closure_xyz folds O2
    # toward C4, and finally H13 is placed between C5 and O3.

    # Step 1: Place C4 at the two-sphere intersection of O3 and C5.
    d_break = 2.5   # O3-C4 (breaking sigma)
    d_pi = 1.40     # C5-C4 (forming pi bond)

    o3_pos = coords[ester_o]
    c5_pos = coords[donor]
    axis = c5_pos - o3_pos
    axis_dist = float(np.linalg.norm(axis))
    if axis_dist < 1e-6:
        return None
    axis_hat = axis / axis_dist

    if axis_dist <= d_break + d_pi:
        x_c4 = (axis_dist ** 2 + d_break ** 2 - d_pi ** 2) / (2.0 * axis_dist)
        h_sq = d_break ** 2 - x_c4 ** 2
        h_c4 = float(np.sqrt(max(h_sq, 0.0)))
    else:
        x_c4 = d_break
        h_c4 = 0.0

    c4_pos = coords[leaving_c]
    proj_c4 = o3_pos + axis_hat * np.dot(c4_pos - o3_pos, axis_hat)
    perp_c4 = c4_pos - proj_c4
    pn_c4 = float(np.linalg.norm(perp_c4))
    if pn_c4 > 1e-8:
        perp_c4_hat = perp_c4 / pn_c4
    else:
        arb = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(axis_hat, arb))) > 0.9:
            arb = np.array([0.0, 1.0, 0.0])
        perp_c4_hat = np.cross(axis_hat, arb)
        perp_c4_hat /= max(float(np.linalg.norm(perp_c4_hat)), 1e-10)

    new_c4 = o3_pos + axis_hat * x_c4 + perp_c4_hat * h_c4
    c4_displacement = new_c4 - coords[leaving_c]

    # Move C4 and its direct H substituents — NOT C5.
    c4_moving = {leaving_c}
    for nbr in r_mol.atoms[leaving_c].bonds:
        ni = atom_to_idx[nbr]
        if symbols[ni] == 'H':
            c4_moving.add(ni)
    for k in c4_moving:
        coords[k] += c4_displacement

    # Step 2: Fold O2 toward C4's new position.
    intermediate_xyz = {'symbols': symbols,
                        'isotopes': r_xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
                        'coords': tuple(tuple(float(c) for c in row) for row in coords)}
    d_co_target = 2.2
    rc_xyz = ring_closure_xyz(
        intermediate_xyz, r_mol,
        forming_bond=(carbonyl_o, leaving_c),
        target_distance=d_co_target)
    if rc_xyz is not None:
        coords = np.array(rc_xyz['coords'], dtype=float)

    # Step 3: Place migrating H between C5 and O3.
    sbl_ch = get_single_bond_length('C', 'H') or 1.09
    sbl_oh = get_single_bond_length('O', 'H') or 0.97
    d_ch = sbl_ch + PAULING_DELTA  # ~1.51 Å
    d_oh = sbl_oh + PAULING_DELTA  # ~1.39 Å
    d_pos = coords[donor]
    a_pos = coords[ester_o]
    da = a_pos - d_pos
    da_dist = float(np.linalg.norm(da))
    if da_dist < 1e-6:
        return None
    da_hat = da / da_dist
    if da_dist <= d_ch + d_oh:
        x_proj = (da_dist ** 2 + d_ch ** 2 - d_oh ** 2) / (2.0 * da_dist)
        h_sq = d_ch ** 2 - x_proj ** 2
        h_perp = float(np.sqrt(max(h_sq, 0.0)))
    else:
        x_proj = d_ch
        h_perp = 0.0
    h_pos = coords[mig_h]
    proj = d_pos + da_hat * np.dot(h_pos - d_pos, da_hat)
    perp = h_pos - proj
    pn = float(np.linalg.norm(perp))
    if pn > 1e-8:
        perp_hat = perp / pn
    else:
        arb = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(da_hat, arb))) > 0.9:
            arb = np.array([0.0, 1.0, 0.0])
        perp_hat = np.cross(da_hat, arb)
        perp_hat /= max(float(np.linalg.norm(perp_hat)), 1e-10)
    coords[mig_h] = d_pos + da_hat * x_proj + perp_hat * h_perp

    ts_xyz = {'symbols': symbols,
              'isotopes': r_xyz.get('isotopes', tuple(0 for _ in range(n_atoms))),
              'coords': tuple(tuple(float(c) for c in row) for row in coords)}
    if colliding_atoms(ts_xyz):
        return None
    return ts_xyz
