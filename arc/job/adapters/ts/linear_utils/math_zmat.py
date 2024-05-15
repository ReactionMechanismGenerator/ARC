"""
Z-matrix utilities, weight/grid helpers, and mathematical interpolation functions
extracted from ``arc.job.adapters.ts.linear``.
"""

import copy
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

from arc.common import get_angle_in_180_range, get_logger
from arc.species.zmat import check_ordered_zmats

if TYPE_CHECKING:
    from arc.reaction import ARCReaction


logger = get_logger()

BOND_LENGTH_MIN: float = 0.4    # Angstroms; shortest physically valid bond
ANGLE_MIN: float = 1.0          # degrees; away from the 0 deg singularity
ANGLE_MAX: float = 179.0        # degrees; away from the 180 deg singularity
BASE_WEIGHT_GRID = (0.35, 0.50, 0.65)
HAMMOND_DELTA = 0.10
WEIGHT_ROUND = 3


# ---------------------------------------------------------------------------
# Z-matrix helpers
# ---------------------------------------------------------------------------

def get_all_zmat_rows(var_name: str) -> List[int]:
    """
    Return every Z-matrix row index encoded in a variable name.

    Consolidated variables use ``|`` to pack several equivalent rows into a single
    variable.  This helper returns all of them, not just the first.

    Examples::

        'R_1_0'       → [1]
        'R_2|4_0|0'   → [2, 4]
        'A_3_1_0'     → [3]
        'D_4_1_0_2'   → [4]
        'DX_5_1_0_2'  → [5]

    Args:
        var_name (str): Z-matrix variable name.

    Returns:
        List[int]: All row indices encoded in the name, or an empty list if the name
        cannot be parsed.
    """
    parts = var_name.split('_')
    if len(parts) < 2:
        return []
    try:
        return [int(idx) for idx in parts[1].split('|')]
    except (ValueError, IndexError):
        return []


def get_all_referenced_atoms(var_name: str) -> List[int]:
    """
    Return every atom index referenced anywhere in a Z-matrix variable name.

    Unlike :func:`get_all_zmat_rows`, which returns only the *defined* row(s),
    this helper returns **all** atom indices encoded in the variable name,
    including the reference atoms used for distance, angle, and dihedral
    definitions.

    Examples::

        'R_1_0'       → [1, 0]
        'R_2|4_0|0'   → [2, 4, 0, 0]
        'A_3_1_0'     → [3, 1, 0]
        'D_4_3_0_2'   → [4, 3, 0, 2]
        'DX_5_1_0_2'  → [5, 1, 0, 2]

    Args:
        var_name (str): Z-matrix variable name.

    Returns:
        List[int]: All atom indices encoded in the name, or an empty list if
        the name cannot be parsed.
    """
    parts = var_name.split('_')
    if len(parts) < 2:
        return []
    atoms: List[int] = []
    for part in parts[1:]:
        for idx_str in part.split('|'):
            try:
                atoms.append(int(idx_str))
            except ValueError:
                pass
    return atoms


def get_r_constraints(expected_breaking_bonds: List[Tuple[int, int]],
                      expected_forming_bonds: List[Tuple[int, int]],
                      ) -> Dict[str, list]:
    """
    Get the "R_atom" constraints for the reactant ZMat.

    Atoms are sorted by their participation frequency in changing bonds (most frequent
    first), with atom index as a deterministic tiebreaker so results are canonical
    regardless of input ordering.

    Args:
        expected_breaking_bonds (List[Tuple[int, int]]): Expected breaking bonds.
        expected_forming_bonds (List[Tuple[int, int]]): Expected forming bonds.

    Returns:
        Dict[str, list]: The constraints.
    """
    constraints = list()
    atom_occurrences: Dict[int, int] = dict()
    for bond in expected_breaking_bonds + expected_forming_bonds:
        for atom in bond:
            atom_occurrences[atom] = atom_occurrences.get(atom, 0) + 1
    atoms_sorted_by_frequency = [k for k, _ in sorted(atom_occurrences.items(),
                                                       key=lambda item: (-item[1], item[0]))]
    for i, atom in enumerate(atoms_sorted_by_frequency):
        for bond in expected_breaking_bonds + expected_forming_bonds:
            if atom in bond and all(a not in atoms_sorted_by_frequency[:i] for a in bond):
                constraints.append(bond if atom == bond[0] else (bond[1], bond[0]))
                break
    return {'R_atom': constraints}


def average_zmat_params(zmat_1: dict,
                        zmat_2: dict,
                        weight: float = 0.5,
                        reactive_xyz_indices: Optional[Set[int]] = None,
                        ) -> Optional[dict]:
    """
    Interpolate internal coordinates using a weight with type-aware, singularity-safe math.

    Variable types are determined **definitively from the coords matrix column position**
    (column 0 = bond length R, column 1 = valence angle A, column 2 = dihedral D/DX),
    not purely from the variable-name prefix.  This avoids mis-classification of
    dummy-atom variables (AX_, DX_) that carry the same physical meaning as their
    non-dummy counterparts.

    Interpolation rules:

    * **Dihedrals**: circular shortest-path via :func:`interp_dihedral_deg`.
      The wrap-around discontinuity at ±180° is handled automatically.
    * **Angles**: linear, then clamped to ``[1.0°, 179.0°]`` to keep the Z-matrix
      frame away from the mathematical singularities at 0° and 180°.
    * **Bond lengths** — linear, then clamped to ≥ 0.4 Å (shortest valid chemical
      bond).  A ``WARNING`` is emitted if clamping occurs.
    * **All values** are cast to ``float`` before arithmetic to prevent silent integer
      division or NumPy scalar type issues.

    If ``reactive_xyz_indices`` is provided, only variables that reference at least
    one reactive XYZ atom (extracted by :func:`get_all_referenced_atoms`) are
    interpolated.  All other (spectator) variables are preserved from ``zmat_1``
    (the reactant anchor).  This prevents remote torsions and unrelated bond lengths
    from being averaged, which can wash out good TS geometry or introduce noise.

    Args:
        zmat_1 (dict): ZMat 1 (anchor / reactant side).
        zmat_2 (dict): ZMat 2 (product side, updated by product XYZ).
        weight (float, optional): The weight to use on a scale of 0 (the reactant) to 1 (the product).
                                  A value of 0.5 means exactly in the middle.
        reactive_xyz_indices (Optional[Set[int]]): XYZ atom indices of atoms that participate
            in forming or breaking bonds.  When provided, only Z-matrix variables that
            reference a reactive XYZ atom are interpolated; spectator variables
            are copied from ``zmat_1``.  When ``None``, all variables are interpolated (legacy behavior).

    Returns:
        Optional[dict]: The weighted average ZMat, or ``None`` if the inputs are
        incompatible (mismatched schemas, out-of-range weight, or missing keys).
    """
    if not (0.0 <= weight <= 1.0):
        logger.debug(f'average_zmat_params: weight {weight} is out of [0, 1] range; returning None.')
        return None
    if 'vars' not in zmat_1 or 'vars' not in zmat_2 or 'coords' not in zmat_1 or 'coords' not in zmat_2:
        logger.debug("average_zmat_params: one of the zmats is missing 'vars' or 'coords'; returning None.")
        return None
    if not check_ordered_zmats(zmat_1, zmat_2):
        logger.debug('average_zmat_params: zmats have mismatched symbol order or variable keys; returning None.')
        return None

    bond_vars: Set[str] = set()
    angle_vars: Set[str] = set()
    dihedral_vars: Set[str] = set()
    for row in zmat_1['coords']:
        if not isinstance(row, (tuple, list)) or len(row) != 3:
            continue
        r_var, a_var, d_var = row
        if isinstance(r_var, str):
            bond_vars.add(r_var)
        if isinstance(a_var, str):
            angle_vars.add(a_var)
        if isinstance(d_var, str):
            dihedral_vars.add(d_var)

    zmat_map: Dict[int, int] = zmat_1.get('map', {})
    if not zmat_map and reactive_xyz_indices is not None:
        logger.debug('average_zmat_params: zmat_1 has no row→atom map; '
                     'reactive filtering will be skipped and all variables interpolated.')

    ts_zmat = copy.deepcopy(zmat_1)
    ts_zmat['vars'] = dict()

    for key, a_raw in zmat_1['vars'].items():
        if key not in zmat_2['vars']:
            logger.debug(f"average_zmat_params: zmat_2 is missing variable {key!r}; returning None.")
            return None
        b_raw = zmat_2['vars'][key]
        a = float(a_raw)
        b = float(b_raw)

        if reactive_xyz_indices is not None:
            all_refs = get_all_referenced_atoms(key)
            if all_refs:
                covered_xyz = {zmat_map[r] for r in all_refs if r in zmat_map}
                if covered_xyz and not (covered_xyz & reactive_xyz_indices):
                    ts_zmat['vars'][key] = a
                    continue

        if key in dihedral_vars:
            ts_zmat['vars'][key] = interp_dihedral_deg(a, b, w=weight)

        elif key in angle_vars:
            result = a + weight * (b - a)
            if result < ANGLE_MIN:
                logger.debug(f'average_zmat_params: angle {key!r} clamped from '
                             f'{result:.4f}° to {ANGLE_MIN}° (singularity floor).')
                result = ANGLE_MIN
            elif result > ANGLE_MAX:
                logger.debug(f'average_zmat_params: angle {key!r} clamped from '
                             f'{result:.4f}° to {ANGLE_MAX}° (singularity ceiling).')
                result = ANGLE_MAX
            ts_zmat['vars'][key] = result

        elif key in bond_vars:
            result = a + weight * (b - a)
            if result < BOND_LENGTH_MIN:
                logger.warning(
                    f'average_zmat_params: interpolated bond length for {key!r} '
                    f'({result:.4f} Å) is below the physical minimum '
                    f'({BOND_LENGTH_MIN} Å); clamping.'
                )
                result = BOND_LENGTH_MIN
            ts_zmat['vars'][key] = result

        else:
            ts_zmat['vars'][key] = a + weight * (b - a)

    return ts_zmat


# ---------------------------------------------------------------------------
# Weight / grid helpers
# ---------------------------------------------------------------------------

def get_rxn_weight(rxn: 'ARCReaction',
                   w_min: float = 0.30,
                   w_max: float = 0.70,
                   delta_e_sat: float = 150.0,
                   reorg_energy: Optional[Union[float, Tuple[float, float]]] = None,
                   ) -> Optional[float]:
    """
    Estimate an interpolation weight w (0=reactant-like, 1=product-like) using reaction thermochemistry only.

    Chemically motivated model:
        Use a Hammond/Leffler parameter (alpha) via a Marcus-like relation:
            alpha ≈ 0.5 + ΔE / (2*λ)
        where λ is an effective "reorganization energy" scale (same units as ΔE).
        We then clamp alpha into [w_min, w_max].

    Defaults:
        By default we choose λ so that |ΔE| = delta_e_sat maps to the extrema w_min / w_max.

    Units assumption: all energies (e0 or e_elect) must be in the same units (kJ/mol by ARC convention).

    Args:
        rxn: The reaction to process.
        w_min: Minimum allowed weight (reactant-like limit), in [0, 0.5].
        w_max: Maximum allowed weight (product-like limit), in [0.5, 1].
        delta_e_sat: Magnitude of reaction energy for considering the TS fully shifted to the extrema (kJ/mol).
        reorg_energy:
            Either:
              - None (derive λ from delta_e_sat and (w_min,w_max)),
              - a single float λ used for both signs,
              - a tuple (λ_exo, λ_endo) to allow asymmetry.

    Returns:
        The estimated weight, or None if energies are unavailable.
    """
    if w_min > w_max:
        w_min, w_max = w_max, w_min
    if not (0.0 <= w_min <= 0.5 <= w_max <= 1.0):
        raise ValueError(f"Invalid bounds: w_min={w_min}, w_max={w_max}. Require w_min in [0,0.5] and w_max in [0.5,1].")
    if delta_e_sat <= 0.0:
        raise ValueError(f"delta_e_sat must be > 0, got {delta_e_sat}")

    reactants, products = rxn.get_reactants_and_products(return_copies=False)
    r_e0 = [spc.e0 for spc in reactants]
    p_e0 = [spc.e0 for spc in products]
    if all(e is not None for e in (r_e0 + p_e0)):
        r_e = r_e0
        p_e = p_e0
    else:
        r_ee = [spc.e_elect for spc in reactants]
        p_ee = [spc.e_elect for spc in products]
        if not all(e is not None for e in (r_ee + p_ee)):
            return None
        r_e = r_ee
        p_e = p_ee

    delta_e = sum(p_e) - sum(r_e)
    if abs(delta_e) < 1e-3:
        return 0.5

    if reorg_energy is None:
        lam_endo = delta_e_sat / (2.0 * (w_max - 0.5)) if (w_max - 0.5) > 0 else float('inf')
        lam_exo  = delta_e_sat / (2.0 * (0.5 - w_min)) if (0.5 - w_min) > 0 else float('inf')
    elif isinstance(reorg_energy, tuple):
        if len(reorg_energy) != 2:
            raise ValueError(f"reorg_energy tuple must be (lambda_exo, lambda_endo), got {reorg_energy}")
        lam_exo, lam_endo = float(reorg_energy[0]), float(reorg_energy[1])
    else:
        lam_exo = lam_endo = float(reorg_energy)

    if lam_exo <= 0.0 or lam_endo <= 0.0:
        raise ValueError(f"Reorganization energies must be > 0. Got lambda_exo={lam_exo}, lambda_endo={lam_endo}")

    lam = lam_endo if delta_e > 0.0 else lam_exo

    w = 0.5 + delta_e / (2.0 * lam)
    if w < w_min:
        return w_min
    if w > w_max:
        return w_max
    return w


def get_weight_grid(rxn: 'ARCReaction',
                    include_hammond: bool = True,
                    base_grid: Tuple[float, ...] = BASE_WEIGHT_GRID,
                    hammond_delta: float = HAMMOND_DELTA,
                    ) -> List[float]:
    """
    Generate a small set of interpolation weights to try.
    Always includes a symmetric grid around 0.5, and optionally also tries
    a Hammond/Marcus-biased guess ± delta.

    Returns:
        List[float]: Sorted unique weights in [0, 1].
    """
    weights: List[float] = list(base_grid)
    if include_hammond:
        w0 = get_rxn_weight(rxn)
        if w0 is not None:
            weights.extend([w0 - hammond_delta, w0, w0 + hammond_delta])
    uniq = {round(clip01(w), WEIGHT_ROUND) for w in weights}
    return sorted(uniq)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def interp_dihedral_deg(a: float, b: float, w: float = 0.5) -> float:
    """
    Interpolate dihedral angles in degrees along the shortest signed difference in (-180, 180].
    E.g., the distance between -179 and 179 is 2 degrees, not 358.

    Args:
        a (float): The first angle in degrees.
        b (float): The second angle in degrees.
        w (float, optional): The weight between 0 and 1.

    Returns:
        float: The interpolated angle in degrees.
    """
    a = get_angle_in_180_range(a, round_to=None)
    b = get_angle_in_180_range(b, round_to=None)
    d = get_angle_in_180_range(b - a, round_to=None)
    return get_angle_in_180_range(a + w * d, round_to=None)


def clip01(x: float) -> float:
    """Clip a float to the [0, 1] range."""
    return max(0.0, min(1.0, x))
