"""
A module for N-dimensional (ND) rotor scan utilities.

Contains pure scan-surface helpers:
grid generation, point iteration, continuous-scan state management,
energy normalization, and adaptive sparse 2D scan logic.
Does **not** own job orchestration or species metadata --
those remain in Scheduler.
"""

import itertools
import math
from typing import Dict, Iterator, List, Optional, Tuple, Union

from arc.common import extremum_list, get_angle_in_180_range, get_logger
from arc.exceptions import InputError, SchedulerError
from arc.species.vectors import calculate_dihedral_angle

logger = get_logger()


ADAPTIVE_DEFAULT_BATCH_SIZE = 10
ADAPTIVE_DEFAULT_MAX_POINTS = 200
ADAPTIVE_DEFAULT_MIN_POINTS = 20

VALIDATION_ENERGY_JUMP_THRESHOLD = 30.0      # kJ/mol between adjacent grid points
VALIDATION_GEOMETRY_RMSD_THRESHOLD = 1.5     # Angstrom, distance-matrix RMSD
VALIDATION_PERIODIC_ENERGY_THRESHOLD = 5.0   # kJ/mol mismatch across wraparound
VALIDATION_PERIODIC_RMSD_THRESHOLD = 1.0     # Angstrom across wraparound
VALIDATION_BRANCH_JUMP_EDGE_COUNT = 2        # min suspicious edges to flag a point


def validate_scan_resolution(increment: float) -> None:
    """
    Validate that the scan resolution divides 360 evenly.

    Args:
        increment (float): The scan resolution in degrees.

    Raises:
        SchedulerError: If the increment is not positive or does not divide 360 evenly.
    """
    if increment <= 0:
        raise SchedulerError(f'The directed scan got a non-positive scan resolution of {increment}')
    quotient = 360.0 / increment
    if not math.isclose(quotient, round(quotient), abs_tol=1e-9):
        raise SchedulerError(f'The directed scan got an illegal scan resolution of {increment}')


def get_torsion_dihedral_grid(xyz: dict,
                              torsions: list,
                              increment: float,
                              ) -> Dict[Tuple[int, ...], List[float]]:
    """
    Build the per-torsion list of dihedral angles for a brute-force scan.

    For each torsion in ``torsions``, computes the current dihedral from ``xyz``
    and generates a list of ``int(360 / increment) + 1`` evenly spaced angles
    starting from that dihedral, each wrapped into the -180..+180 range.

    Args:
        xyz (dict): The 3D coordinates (ARC xyz dict with 'coords' key).
        torsions (list): List of torsion definitions (each a list of 4 atom indices, 0-indexed).
        increment (float): The scan resolution in degrees.

    Returns:
        dict: Keys are torsion tuples, values are lists of dihedral angles.
    """
    dihedrals = dict()
    for torsion in torsions:
        original_dihedral = get_angle_in_180_range(
            calculate_dihedral_angle(coords=xyz['coords'], torsion=torsion, index=0))
        dihedrals[tuple(torsion)] = [
            get_angle_in_180_range(original_dihedral + i * increment)
            for i in range(int(360 / increment) + 1)
        ]
    return dihedrals


def iter_brute_force_scan_points(dihedrals_by_torsion: Dict[Tuple[int, ...], List[float]],
                                 torsions: list,
                                 diagonal: bool = False,
                                 ) -> Iterator[Tuple[float, ...]]:
    """
    Yield dihedral-angle tuples for every point in a brute-force scan.

    Args:
        dihedrals_by_torsion (dict): Mapping ``{torsion_tuple: [angle_0, angle_1, ...]}``
            as returned by :func:`get_torsion_dihedral_grid`.
        torsions (list): Ordered list of torsion definitions (each a list of 4 ints).
        diagonal (bool, optional): If ``True``, all torsions are incremented
            simultaneously (1-D diagonal path through ND space).
            If ``False`` (default), the full cartesian product is generated.

    Yields:
        tuple: A tuple of dihedral angles, one per torsion, in the order of ``torsions``.
    """
    if not diagonal:
        for combo in itertools.product(*[dihedrals_by_torsion[tuple(t)] for t in torsions]):
            yield combo
    else:
        n_points = len(dihedrals_by_torsion[tuple(torsions[0])])
        for i in range(n_points):
            yield tuple(dihedrals_by_torsion[tuple(t)][i] for t in torsions)


def initialize_continuous_scan_state(rotor_dict: dict,
                                     xyz: dict,
                                     ) -> None:
    """
    Initialize the continuous-scan bookkeeping fields on a rotor dict
    (``cont_indices`` and ``original_dihedrals``) if they have not been set yet.

    Modifies ``rotor_dict`` **in place**.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        xyz (dict): The 3D coordinates (ARC xyz dict).
    """
    torsions = rotor_dict['torsion']
    if not len(rotor_dict['cont_indices']):
        rotor_dict['cont_indices'] = [0] * len(torsions)
    if not len(rotor_dict['original_dihedrals']):
        rotor_dict['original_dihedrals'] = [
            f'{calculate_dihedral_angle(coords=xyz["coords"], torsion=scan, index=1):.2f}'
            for scan in rotor_dict['scan']
        ]  # stored as str for YAML compatibility


def get_continuous_scan_dihedrals(rotor_dict: dict,
                                  increment: float,
                                  ) -> List[float]:
    """
    Compute the dihedral angles for the *current* continuous-scan step,
    based on ``cont_indices`` and ``original_dihedrals`` stored in the rotor dict.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.

    Returns:
        list: A list of dihedral angles (one per torsion) for this step.
    """
    dihedrals = []
    for index, original_dihedral_str in enumerate(rotor_dict['original_dihedrals']):
        original_dihedral = get_angle_in_180_range(float(original_dihedral_str))
        dihedral = original_dihedral + rotor_dict['cont_indices'][index] * increment
        dihedral = get_angle_in_180_range(dihedral)
        dihedrals.append(dihedral)
    return dihedrals


def is_continuous_scan_complete(rotor_dict: dict,
                                increment: float,
                                ) -> bool:
    """
    Check whether a continuous directed scan has visited every grid point.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.

    Returns:
        bool: ``True`` if the scan is complete (all counters exhausted).
    """
    max_num = 360 / increment + 1  # dihedral angles per dimension
    return rotor_dict['cont_indices'][-1] == max_num - 1  # 0-indexed


def increment_continuous_scan_indices(rotor_dict: dict,
                                      increment: float,
                                      diagonal: bool = False,
                                      ) -> None:
    """
    Advance the continuous-scan counters by one step.

    For a diagonal scan every counter is incremented together.
    For a non-diagonal scan the counters are incremented like an odometer
    (innermost dimension first).

    Modifies ``rotor_dict['cont_indices']`` **in place**.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.
        diagonal (bool, optional): Whether this is a diagonal scan.
    """
    torsions = rotor_dict['torsion']
    max_num = 360 / increment + 1

    if diagonal:
        rotor_dict['cont_indices'] = [rotor_dict['cont_indices'][0] + 1] * len(torsions)
    else:
        for index in range(len(torsions)):
            if rotor_dict['cont_indices'][index] < max_num - 1:
                rotor_dict['cont_indices'][index] += 1
                break
            elif rotor_dict['cont_indices'][index] == max_num - 1 and index < len(torsions) - 1:
                rotor_dict['cont_indices'][index] = 0


def normalize_directed_scan_energies(rotor_dict: dict) -> Tuple[dict, int]:
    """
    Build a ``results`` dict for a non-ESS directed scan and normalize energies
    so that the minimum is zero.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
            Must contain ``'directed_scan'``, ``'directed_scan_type'``, and ``'scan'`` keys.

    Returns:
        tuple: A two-element tuple:
            - results (dict): ``{'directed_scan_type': ..., 'scans': ..., 'directed_scan': ...}``
              with energies shifted so the minimum is 0.
            - trshed_points (int): Number of scan points that required troubleshooting.
    """
    dihedrals = [[float(d) for d in key] for key in rotor_dict['directed_scan'].keys()]
    sorted_dihedrals = sorted(dihedrals)
    min_energy = extremum_list(
        [entry['energy'] for entry in rotor_dict['directed_scan'].values()],
        return_min=True,
    )
    results = {
        'directed_scan_type': rotor_dict['directed_scan_type'],
        'scans': rotor_dict['scan'],
        'directed_scan': rotor_dict['directed_scan'],
    }
    trshed_points = 0
    for dihedral_list in sorted_dihedrals:
        key = tuple(f'{d:.2f}' for d in dihedral_list)
        dihedral_dict = results['directed_scan'][key]
        if dihedral_dict['trsh']:
            trshed_points += 1
        if dihedral_dict['energy'] is not None and min_energy is not None:
            dihedral_dict['energy'] -= min_energy
    return results, trshed_points


def format_dihedral_key(dihedrals: list) -> Tuple[str, ...]:
    """
    Build the legacy string-tuple key used to index ``rotor_dict['directed_scan']``.

    Args:
        dihedrals (list): A list of dihedral angles (floats).

    Returns:
        tuple: A tuple of ``'{angle:.2f}'`` strings, one per dihedral.
    """
    return tuple(f'{dihedral:.2f}' for dihedral in dihedrals)


def record_directed_scan_point(rotor_dict: dict,
                               dihedrals: list,
                               energy: Optional[float],
                               xyz: Optional[dict],
                               is_isomorphic: bool,
                               trsh: list,
                               ) -> None:
    """
    Record a single completed directed-scan point into the legacy
    ``rotor_dict['directed_scan']`` structure.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        dihedrals (list): The dihedral angles that define this scan point.
        energy (float or None): The electronic energy (absolute, un-normalized).
        xyz (dict or None): The optimized geometry for this point.
        is_isomorphic (bool): Whether the optimized geometry is isomorphic with the species graph.
        trsh (list): Troubleshooting methods applied to this point.
    """
    key = format_dihedral_key(dihedrals)
    rotor_dict['directed_scan'][key] = {
        'energy': energy,
        'xyz': xyz,
        'is_isomorphic': is_isomorphic,
        'trsh': trsh,
    }


def get_rotor_dict_by_pivots(rotors_dict: dict,
                             pivots: Union[List[int], List[List[int]]],
                             ) -> Optional[Tuple[int, dict]]:
    """
    Look up a rotor dict entry by its pivots.

    Args:
        rotors_dict (dict): The full ``species.rotors_dict`` mapping.
        pivots: The pivot(s) to match against.

    Returns:
        tuple or None: ``(rotor_index, rotor_dict)`` if found, else ``None``.
    """
    for rotor_index, rotor_dict in rotors_dict.items():
        if rotor_dict['pivots'] == pivots:
            return rotor_index, rotor_dict
    return None


def finalize_directed_scan_results(rotor_dict: dict,
                                   parse_nd_scan_energies_func=None,
                                   increment: Optional[float] = None,
                                   ) -> Tuple[dict, int]:
    """
    Produce the final results payload for a completed directed scan.

    For ESS-controlled scans (``directed_scan_type == 'ess'``), delegates to the
    parser via ``parse_nd_scan_energies_func``.  For brute-force and continuous scans,
    normalizes energies so the minimum is zero and counts troubleshot points.

    For adaptive 2D scans, also runs surface validation if ``increment`` is provided.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        parse_nd_scan_energies_func (callable, optional): A callable that takes
            ``log_file_path`` and returns a list whose first element is the
            results dict.  Only needed for ESS scans.  Pass
            ``parser.parse_nd_scan_energies`` from the caller to avoid importing
            the parser here (which would create a circular import through
            ``arc.__init__``).
        increment (float, optional): The scan resolution in degrees.  If provided
            and the scan is adaptive, surface validation is run.

    Returns:
        tuple: ``(results, trshed_points)`` where *results* has the structure
            consumed by ``plotter.save_nd_rotor_yaml``, ``plotter.plot_1d_rotor_scan``,
            and ``plotter.plot_2d_rotor_scan``.
    """
    if rotor_dict['directed_scan_type'] == 'ess':
        if parse_nd_scan_energies_func is None:
            raise ValueError('parse_nd_scan_energies_func must be provided for ESS directed scans')
        results = parse_nd_scan_energies_func(log_file_path=rotor_dict['scan_path'])[0]
        return results, 0
    results, trshed_points = normalize_directed_scan_energies(rotor_dict)
    # Attach optional sparse metadata for adaptive scans (non-breaking addition)
    if is_adaptive_enabled(rotor_dict):
        state = rotor_dict.get('adaptive_scan', {})
        results['sampling_policy'] = 'adaptive'
        results['adaptive_scan_summary'] = {
            'completed_count': len(state.get('completed_points', [])),
            'failed_count': len(state.get('failed_points', [])),
            'invalid_count': len(state.get('invalid_points', [])),
            'stopping_reason': state.get('stopping_reason'),
            'failed_points': [list(p) for p in state.get('failed_points', [])],
            'invalid_points': [list(p) for p in state.get('invalid_points', [])],
        }
        # Run surface validation, coupling metrics, and classification if increment is available
        if increment is not None:
            update_adaptive_validation_state(rotor_dict, increment)
            update_nd_classification(rotor_dict, increment)
            validation = state.get('validation', {})
            results['validation_summary'] = {
                'discontinuous_edges': len(validation.get('discontinuous_edges', [])),
                'periodic_inconsistencies': len(validation.get('periodic_inconsistencies', [])),
                'branch_jump_points': len(validation.get('branch_jump_points', [])),
                'status': validation.get('status', 'not_run'),
                'thresholds': validation.get('thresholds', {}),
            }
            coupling = state.get('coupling_metrics', {})
            results['coupling_summary'] = {
                'nonseparability_score': coupling.get('nonseparability_score'),
                'cross_term_strength': coupling.get('cross_term_strength'),
                'status': coupling.get('status', 'not_run'),
            }
            quality = state.get('surface_quality', {})
            results['surface_quality_summary'] = {
                'quality_score': quality.get('quality_score'),
                'coverage_fraction': quality.get('coverage_fraction'),
                'status': quality.get('status', 'not_run'),
            }
            nd_cls = state.get('nd_classification', {})
            results['classification_summary'] = {
                'classification': nd_cls.get('classification'),
                'confidence': nd_cls.get('confidence'),
                'recommended_action': nd_cls.get('recommended_action'),
                'reason': nd_cls.get('reason'),
            }
    return results, trshed_points


def decrement_running_jobs(rotor_dict: dict) -> bool:
    """
    Decrement the brute-force running-jobs counter and return whether all jobs
    for this rotor have finished.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.

    Returns:
        bool: ``True`` if all brute-force jobs for this rotor have terminated
            (counter reached 0).
    """
    rotor_dict['number_of_running_jobs'] -= 1
    if rotor_dict['number_of_running_jobs'] < 0:
        logger.warning(f'Running jobs counter went below zero '
                       f'({rotor_dict["number_of_running_jobs"]}), clamping to 0.')
        rotor_dict['number_of_running_jobs'] = 0
    return rotor_dict['number_of_running_jobs'] == 0


# ===========================================================================
# Adaptive sparse 2D scan helpers
# ===========================================================================

def _angular_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute the Euclidean distance between two 2D angle points
    using periodic-aware differences on each dimension.

    Args:
        p1 (tuple): First point ``(phi0, phi1)`` in degrees.
        p2 (tuple): Second point ``(phi0, phi1)`` in degrees.

    Returns:
        float: The distance in degrees.
    """
    d0 = abs(p1[0] - p2[0]) % 360.0
    d0 = min(d0, 360.0 - d0)
    d1 = abs(p1[1] - p2[1]) % 360.0
    d1 = min(d1, 360.0 - d1)
    return math.hypot(d0, d1)


def _normalize_angle_key(phi: float) -> float:
    """Wrap an angle into -180..+180 and round to 2 decimals."""
    return round(get_angle_in_180_range(phi), 2)


def point_to_key(point: Tuple[float, float]) -> Tuple[str, ...]:
    """Convert a 2D angle tuple to the normalized legacy string key."""
    return tuple(f'{_normalize_angle_key(a):.2f}' for a in point)


# ---------------------------------------------------------------------------
# Policy / eligibility
# ---------------------------------------------------------------------------

def is_adaptive_eligible(rotor_dict: dict) -> bool:
    """
    Check whether a rotor dict is eligible for adaptive sparse scanning.

    Eligibility requires:
        * ``directed_scan_type`` is ``'brute_force_sp'`` or ``'brute_force_opt'``
        * ``dimensions`` == 2
        * not a diagonal scan type

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.

    Returns:
        bool: ``True`` if the rotor is eligible for adaptive scanning.
    """
    dst = rotor_dict.get('directed_scan_type', '')
    if dst not in ('brute_force_sp', 'brute_force_opt'):
        return False
    if rotor_dict.get('dimensions', 1) != 2:
        return False
    if 'diagonal' in dst:
        return False
    return True


def is_adaptive_enabled(rotor_dict: dict) -> bool:
    """
    Check whether adaptive scanning is both eligible and enabled for a rotor.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.

    Returns:
        bool: ``True`` if the rotor should use adaptive sparse scanning.
    """
    if not is_adaptive_eligible(rotor_dict):
        return False
    policy = rotor_dict.get('sampling_policy', 'dense')
    return policy == 'adaptive'


# ---------------------------------------------------------------------------
# Adaptive state initialization
# ---------------------------------------------------------------------------

def _make_empty_adaptive_state(batch_size: int = ADAPTIVE_DEFAULT_BATCH_SIZE,
                               max_points: Optional[int] = ADAPTIVE_DEFAULT_MAX_POINTS,
                               min_points: int = ADAPTIVE_DEFAULT_MIN_POINTS,
                               ) -> dict:
    """Return a fresh, YAML-serializable adaptive_scan state dict."""
    return {
        'enabled': True,
        'phase': 'seed',
        'batch_size': batch_size,
        'candidate_points': list(),
        'pending_points': list(),
        'completed_points': list(),
        'failed_points': list(),
        'invalid_points': list(),
        'seed_points': list(),
        'selected_points_history': list(),
        'stopping_reason': None,
        'max_points': max_points,
        'min_points': min_points,
        'fit_metadata': dict(),
        'surface_model': dict(),
    }


def initialize_adaptive_scan_state(rotor_dict: dict,
                                   xyz: dict,
                                   increment: float,
                                   batch_size: int = ADAPTIVE_DEFAULT_BATCH_SIZE,
                                   max_points: Optional[int] = ADAPTIVE_DEFAULT_MAX_POINTS,
                                   min_points: int = ADAPTIVE_DEFAULT_MIN_POINTS,
                                   ) -> None:
    """
    Initialize adaptive scan state on a rotor dict if it does not already exist.

    Also generates the deterministic seed points and stores them.
    Modifies ``rotor_dict`` **in place**.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        xyz (dict): The 3D coordinates (ARC xyz dict).
        increment (float): The scan resolution in degrees.
        batch_size (int): Number of points to submit per adaptive batch.
        max_points (int or None): Maximum total points before stopping.
        min_points (int): Minimum points before stopping is allowed.
    """
    if 'adaptive_scan' in rotor_dict and rotor_dict['adaptive_scan'].get('enabled', False):
        return  # already initialized
    state = _make_empty_adaptive_state(batch_size=batch_size, max_points=max_points, min_points=min_points)
    rotor_dict['adaptive_scan'] = state
    rotor_dict['sampling_policy'] = 'adaptive'
    # Populate original_dihedrals from current geometry so grid origin is consistent
    # between seed generation and later candidate generation.
    torsions = rotor_dict['torsion']
    if not rotor_dict.get('original_dihedrals'):
        rotor_dict['original_dihedrals'] = [
            f'{_normalize_angle_key(calculate_dihedral_angle(coords=xyz["coords"], torsion=t, index=0)):.2f}'
            for t in torsions
        ]
    seeds = generate_adaptive_seed_points(rotor_dict, xyz, increment)
    state['seed_points'] = [list(s) for s in seeds]


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------

def generate_adaptive_seed_points(rotor_dict: dict,
                                  xyz: dict,
                                  increment: float,
                                  ) -> List[Tuple[float, float]]:
    """
    Generate a deterministic set of seed points for an adaptive 2D scan.

    The seed includes:
      1. The current-geometry point.
      2. A coarse grid at 3x the base increment (covering the full 2D surface sparsely).
      3. Two 1D cuts along each dimension through the current-geometry point.

    All angles are normalized to -180..+180.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        xyz (dict): The 3D coordinates (ARC xyz dict).
        increment (float): The base scan resolution in degrees.

    Returns:
        list: A list of ``(phi0, phi1)`` tuples (deduplicated).
    """
    torsions = rotor_dict['torsion']
    if len(torsions) != 2:
        raise InputError(f'Adaptive seed generation requires exactly 2 torsions, got {len(torsions)}')

    # Current geometry dihedral values
    orig_0 = _normalize_angle_key(
        calculate_dihedral_angle(coords=xyz['coords'], torsion=torsions[0], index=0))
    orig_1 = _normalize_angle_key(
        calculate_dihedral_angle(coords=xyz['coords'], torsion=torsions[1], index=0))

    n_fine = int(360 / increment) + 1
    fine_angles_0 = [_normalize_angle_key(orig_0 + i * increment) for i in range(n_fine)]
    fine_angles_1 = [_normalize_angle_key(orig_1 + i * increment) for i in range(n_fine)]

    # Coarse grid: every 3rd step of the fine grid (ensures manageable seed count)
    coarse_step = 3
    coarse_0 = fine_angles_0[::coarse_step]
    coarse_1 = fine_angles_1[::coarse_step]

    seen = set()
    seeds = []

    def _add(p):
        key = point_to_key(p)
        if key not in seen:
            seen.add(key)
            seeds.append((float(key[0]), float(key[1])))

    # 1. Current conformation
    _add((orig_0, orig_1))

    # 2. Coarse grid
    for a0 in coarse_0:
        for a1 in coarse_1:
            _add((a0, a1))

    # 3. 1D cuts through origin along each dimension
    for a0 in fine_angles_0:
        _add((a0, orig_1))
    for a1 in fine_angles_1:
        _add((orig_0, a1))

    return seeds


# ---------------------------------------------------------------------------
# Bookkeeping helpers
# ---------------------------------------------------------------------------

def mark_scan_points_pending(rotor_dict: dict, points: List[list]) -> None:
    """
    Add points to the pending list in the adaptive state.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        points (list): Points to mark pending (each a 2-element list of floats).
    """
    state = rotor_dict['adaptive_scan']
    pending_keys = {point_to_key(tuple(p)) for p in state['pending_points']}
    for p in points:
        key = point_to_key(tuple(p))
        if key not in pending_keys:
            state['pending_points'].append(list(p))
            pending_keys.add(key)


def mark_scan_point_completed(rotor_dict: dict,
                              point: list,
                              energy: Optional[float],
                              xyz: Optional[dict],
                              is_isomorphic: bool,
                              trsh: list,
                              ) -> None:
    """
    Record a completed adaptive scan point.

    Moves the point from pending to completed and writes
    into the legacy ``directed_scan`` structure.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        point (list): The 2D dihedral angles.
        energy: The electronic energy.
        xyz: The optimized geometry.
        is_isomorphic (bool): Isomorphism check result.
        trsh (list): Troubleshooting methods applied.
    """
    state = rotor_dict['adaptive_scan']
    key = point_to_key(tuple(point))
    # Remove from pending
    state['pending_points'] = [p for p in state['pending_points']
                               if point_to_key(tuple(p)) != key]
    # Add to completed (if not already there)
    if not any(point_to_key(tuple(c)) == key for c in state['completed_points']):
        state['completed_points'].append(list(point))
    # Also write into legacy directed_scan
    record_directed_scan_point(rotor_dict, point, energy, xyz, is_isomorphic, trsh)


def mark_scan_point_failed(rotor_dict: dict,
                           point: list,
                           reason: Optional[str] = None,
                           ) -> None:
    """
    Record a failed adaptive scan point.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        point (list): The 2D dihedral angles.
        reason (str, optional): Reason for failure.
    """
    state = rotor_dict['adaptive_scan']
    key = point_to_key(tuple(point))
    state['pending_points'] = [p for p in state['pending_points']
                               if point_to_key(tuple(p)) != key]
    if not any(point_to_key(tuple(f)) == key for f in state['failed_points']):
        state['failed_points'].append(list(point))


def mark_scan_point_invalid(rotor_dict: dict,
                            point: list,
                            reason: Optional[str] = None,
                            ) -> None:
    """
    Record an invalid adaptive scan point (e.g. non-isomorphic).

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        point (list): The 2D dihedral angles.
        reason (str, optional): Reason for invalidation.
    """
    state = rotor_dict['adaptive_scan']
    key = point_to_key(tuple(point))
    state['pending_points'] = [p for p in state['pending_points']
                               if point_to_key(tuple(p)) != key]
    if not any(point_to_key(tuple(inv)) == key for inv in state['invalid_points']):
        state['invalid_points'].append(list(point))


def get_completed_adaptive_points(rotor_dict: dict) -> List[list]:
    """Return the completed points list from adaptive state."""
    return rotor_dict.get('adaptive_scan', {}).get('completed_points', [])


def get_pending_adaptive_points(rotor_dict: dict) -> List[list]:
    """Return the pending points list from adaptive state."""
    return rotor_dict.get('adaptive_scan', {}).get('pending_points', [])


def _all_visited_keys(rotor_dict: dict) -> set:
    """Return the set of string-tuple keys for all visited/submitted points."""
    state = rotor_dict.get('adaptive_scan', {})
    keys = set()
    for lst_name in ('completed_points', 'pending_points', 'failed_points', 'invalid_points'):
        for p in state.get(lst_name, []):
            keys.add(point_to_key(tuple(p)))
    return keys


# ---------------------------------------------------------------------------
# Surrogate / surface model
# ---------------------------------------------------------------------------

def fit_adaptive_surface_model(rotor_dict: dict) -> dict:
    """
    Fit a lightweight RBF-like interpolation model from completed scan points.

    The model is an inverse-distance-weighted (IDW) interpolation on
    periodic 2D angle space.  The returned dict is YAML-serializable and
    contains only the data needed to evaluate predictions.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.

    Returns:
        dict: A model dict with keys ``'centers'``, ``'values'``, ``'length_scale'``.
    """
    state = rotor_dict['adaptive_scan']
    directed = rotor_dict['directed_scan']
    centers = []
    values = []
    for pt in state['completed_points']:
        key = point_to_key(tuple(pt))
        entry = directed.get(key, None)
        if entry is not None and entry.get('energy') is not None:
            centers.append([float(pt[0]), float(pt[1])])
            values.append(float(entry['energy']))
    model = {
        'type': 'idw',
        'centers': centers,
        'values': values,
        'length_scale': 30.0,  # degrees; controls smoothing
    }
    rotor_dict['adaptive_scan']['surface_model'] = model
    rotor_dict['adaptive_scan']['fit_metadata'] = {
        'n_points': len(centers),
    }
    return model


def predict_surface_values(model_dict: dict, query_points: List[list]) -> List[Optional[float]]:
    """
    Predict energy values at query points using the fitted model.

    Uses inverse-distance weighting with periodic angular distance.

    Args:
        model_dict (dict): A model dict as returned by :func:`fit_adaptive_surface_model`.
        query_points (list): List of ``[phi0, phi1]`` query points.

    Returns:
        list: Predicted energy values (``None`` if the model has no data).
    """
    centers = model_dict.get('centers', [])
    values = model_dict.get('values', [])
    length_scale = model_dict.get('length_scale', 30.0)

    if not centers:
        return [None] * len(query_points)

    predictions = []
    for qp in query_points:
        weights = []
        for c in centers:
            d = _angular_distance(tuple(qp), tuple(c))
            if d < 1e-8:
                weights.append(1e12)  # essentially exact match
            else:
                weights.append(1.0 / (d / length_scale) ** 2)
        total_w = sum(weights)
        if total_w < 1e-30:
            predictions.append(None)
        else:
            pred = sum(w * v for w, v in zip(weights, values)) / total_w
            predictions.append(pred)
    return predictions


def score_candidate_points(rotor_dict: dict, candidate_points: List[list]) -> List[float]:
    """
    Score candidate points for adaptive acquisition.

    The score is a combination of:
      1. **Distance score**: How far the candidate is from the nearest sampled point
         (prefer points in under-sampled regions).
      2. **Energy score**: Lower predicted energy is mildly preferred
         (explore low-energy regions more).

    Higher score means higher priority for selection.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        candidate_points (list): Candidate ``[phi0, phi1]`` points.

    Returns:
        list: A score for each candidate (higher = more desirable).
    """
    state = rotor_dict['adaptive_scan']
    model = state.get('surface_model', {})

    # Collect all sampled centers
    sampled = []
    for pt in state.get('completed_points', []):
        sampled.append(tuple(pt))
    for pt in state.get('failed_points', []):
        sampled.append(tuple(pt))

    predictions = predict_surface_values(model, candidate_points)

    scores = []
    for i, cp in enumerate(candidate_points):
        # Distance to nearest sampled point
        if sampled:
            min_dist = min(_angular_distance(tuple(cp), s) for s in sampled)
        else:
            min_dist = 360.0  # max possible

        dist_score = min_dist / 360.0  # normalize to [0, 1]-ish

        # Energy preference: lower predicted energy -> mild bonus
        energy_score = 0.0
        pred = predictions[i]
        if pred is not None and model.get('values'):
            e_range = max(model['values']) - min(model['values']) if len(model['values']) > 1 else 1.0
            if e_range > 1e-10:
                energy_score = 0.2 * (1.0 - (pred - min(model['values'])) / e_range)

        scores.append(dist_score + energy_score)
    return scores


# ---------------------------------------------------------------------------
# Candidate generation & selection
# ---------------------------------------------------------------------------

def generate_adaptive_candidate_points(rotor_dict: dict, increment: float) -> List[list]:
    """
    Generate the full set of candidate grid points that have not been visited.

    Returns all points on the full dense grid that are not yet in any
    visited set (completed, pending, failed, invalid).

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.

    Returns:
        list: Unvisited ``[phi0, phi1]`` points on the full grid.
    """
    visited = _all_visited_keys(rotor_dict)

    # Reconstruct the full grid angles from the seed data or from first principles.
    # Use original_dihedrals if available, else start from 0.
    orig_dihedrals = rotor_dict.get('original_dihedrals', [])
    if orig_dihedrals and len(orig_dihedrals) == 2:
        start_0, start_1 = float(orig_dihedrals[0]), float(orig_dihedrals[1])
    else:
        start_0, start_1 = 0.0, 0.0

    n = int(360 / increment) + 1
    angles_0 = [_normalize_angle_key(start_0 + i * increment) for i in range(n)]
    angles_1 = [_normalize_angle_key(start_1 + i * increment) for i in range(n)]

    candidates = []
    seen = set()
    for a0 in angles_0:
        for a1 in angles_1:
            key = point_to_key((a0, a1))
            if key not in visited and key not in seen:
                seen.add(key)
                candidates.append([float(key[0]), float(key[1])])
    return candidates


def select_next_adaptive_points(rotor_dict: dict,
                                increment: float,
                                batch_size: Optional[int] = None,
                                ) -> List[list]:
    """
    Select the next batch of points to submit for an adaptive scan.

    If the scan is in the ``'seed'`` phase, returns the unsubmitted seed points.
    Otherwise fits a surrogate, scores candidates, and returns the top-scoring batch.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.
        batch_size (int, optional): Override the batch size from adaptive state.

    Returns:
        list: Selected ``[phi0, phi1]`` points.
    """
    state = rotor_dict['adaptive_scan']
    bs = batch_size if batch_size is not None else state.get('batch_size', ADAPTIVE_DEFAULT_BATCH_SIZE)
    visited = _all_visited_keys(rotor_dict)

    if state['phase'] == 'seed':
        # Return seed points that haven't been submitted yet
        unsubmitted = []
        for s in state['seed_points']:
            key = point_to_key(tuple(s))
            if key not in visited:
                unsubmitted.append(s)
        # Transition to adaptive phase once all seeds are dispatched
        if len(unsubmitted) <= bs:
            state['phase'] = 'adaptive'
        return unsubmitted[:bs]

    # Adaptive phase: fit model, generate candidates, score & select
    candidates = generate_adaptive_candidate_points(rotor_dict, increment)
    if not candidates:
        return []

    fit_adaptive_surface_model(rotor_dict)
    scores = score_candidate_points(rotor_dict, candidates)

    # Sort by score descending, take top batch_size
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    selected = [candidates[i] for i, _ in indexed[:bs]]

    state['selected_points_history'].append([list(p) for p in selected])
    return selected


# ---------------------------------------------------------------------------
# Stopping logic
# ---------------------------------------------------------------------------

def should_continue_adaptive_scan(rotor_dict: dict, increment: float) -> bool:
    """
    Determine whether the adaptive scan should continue submitting new batches.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.

    Returns:
        bool: ``True`` if more batches should be submitted.
    """
    reason = get_adaptive_stopping_reason(rotor_dict, increment)
    if reason is not None:
        rotor_dict['adaptive_scan']['stopping_reason'] = reason
        return False
    return True


def get_adaptive_stopping_reason(rotor_dict: dict, increment: float) -> Optional[str]:
    """
    Return the stopping reason, or ``None`` if the scan should continue.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.

    Returns:
        str or None: Reason for stopping, or ``None``.
    """
    state = rotor_dict['adaptive_scan']
    n_completed = len(state['completed_points'])
    n_pending = len(state['pending_points'])

    # Already stopped
    if state.get('stopping_reason') is not None:
        return state['stopping_reason']

    # Max points reached
    max_pts = state.get('max_points')
    if max_pts is not None and (n_completed + n_pending) >= max_pts:
        return 'max_points_reached'

    # No more candidates on the grid
    candidates = generate_adaptive_candidate_points(rotor_dict, increment)
    if not candidates and n_pending == 0:
        return 'grid_exhausted'

    # All grid points have been visited (full coverage).
    # Use int(360/increment) per dimension (not +1) because angle normalization
    # maps +180 to -180, so the endpoint duplicates the start.
    n_grid = int(360 / increment) ** 2
    n_visited = len(_all_visited_keys(rotor_dict))
    if n_visited >= n_grid:
        return 'full_coverage'

    # Min points check: don't stop before reaching min_points
    min_pts = state.get('min_points', ADAPTIVE_DEFAULT_MIN_POINTS)
    if n_completed < min_pts:
        return None

    return None


def is_adaptive_scan_complete(rotor_dict: dict, increment: float) -> bool:
    """
    Check if an adaptive scan is fully complete (stopped and no pending jobs).

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.

    Returns:
        bool: ``True`` if the adaptive scan has stopped and has no pending points.
    """
    state = rotor_dict.get('adaptive_scan', {})
    if not state.get('enabled', False):
        return False
    n_pending = len(state.get('pending_points', []))
    if n_pending > 0:
        return False
    if state.get('stopping_reason') is not None:
        return True
    return not should_continue_adaptive_scan(rotor_dict, increment)


# ===========================================================================
# Surface validation for adaptive 2D scans
# ===========================================================================


def _make_empty_validation_state() -> dict:
    """Return a fresh, YAML-serializable validation state dict."""
    return {
        'enabled': True,
        'status': 'not_run',
        'neighbor_edges_checked': 0,
        'discontinuous_edges': [],
        'periodic_edges_checked': 0,
        'periodic_inconsistencies': [],
        'branch_jump_points': [],
        'energy_jump_summary': {},
        'geometry_rmsd_summary': {},
        'thresholds': {
            'energy_jump': VALIDATION_ENERGY_JUMP_THRESHOLD,
            'geometry_rmsd': VALIDATION_GEOMETRY_RMSD_THRESHOLD,
            'periodic_energy': VALIDATION_PERIODIC_ENERGY_THRESHOLD,
            'periodic_rmsd': VALIDATION_PERIODIC_RMSD_THRESHOLD,
            'branch_jump_edge_count': VALIDATION_BRANCH_JUMP_EDGE_COUNT,
        },
        'notes': [],
    }


def _periodic_neighbor_offsets(increment: float) -> List[Tuple[float, float]]:
    """Return the 4 cardinal neighbor offsets for a 2D grid."""
    return [(increment, 0.0), (-increment, 0.0), (0.0, increment), (0.0, -increment)]


def get_sampled_point_neighbors(rotor_dict: dict,
                                point: list,
                                increment: float,
                                ) -> List[list]:
    """
    Return sampled neighboring points of ``point`` on the 2D scan grid.

    Neighbors are the 4 cardinal grid-adjacent points (±increment on each axis)
    that exist in the completed scan data.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        point (list): ``[phi0, phi1]`` in degrees.
        increment (float): The scan resolution in degrees.

    Returns:
        list: Neighboring points that have completed scan data.
    """
    directed = rotor_dict.get('directed_scan', {})
    neighbors = []
    for d0, d1 in _periodic_neighbor_offsets(increment):
        nb = [_normalize_angle_key(point[0] + d0), _normalize_angle_key(point[1] + d1)]
        key = point_to_key(tuple(nb))
        if key in directed:
            neighbors.append(nb)
    return neighbors


def iter_sampled_neighbor_edges(rotor_dict: dict,
                                increment: float,
                                ) -> Iterator[Tuple[list, list]]:
    """
    Yield unique pairs of neighboring sampled points for validation.

    Each edge ``(point_a, point_b)`` is yielded exactly once, where both
    points have completed scan data.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.

    Yields:
        tuple: ``(point_a, point_b)`` where each is ``[phi0, phi1]``.
    """
    directed = rotor_dict.get('directed_scan', {})
    seen_edges = set()
    for key_tuple in directed.keys():
        pt = [float(key_tuple[0]), float(key_tuple[1])]
        for d0, d1 in _periodic_neighbor_offsets(increment):
            nb = [_normalize_angle_key(pt[0] + d0), _normalize_angle_key(pt[1] + d1)]
            nb_key = point_to_key(tuple(nb))
            if nb_key in directed:
                edge = tuple(sorted([point_to_key(tuple(pt)), nb_key]))
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    yield pt, nb


def calculate_neighbor_energy_jump(rotor_dict: dict,
                                   point_a: list,
                                   point_b: list,
                                   ) -> Optional[float]:
    """
    Compute the absolute energy difference between two neighboring scan points.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        point_a (list): ``[phi0, phi1]`` first point.
        point_b (list): ``[phi0, phi1]`` second point.

    Returns:
        float or None: Absolute energy difference in kJ/mol, or ``None`` if
            either point lacks energy data.
    """
    directed = rotor_dict.get('directed_scan', {})
    key_a = point_to_key(tuple(point_a))
    key_b = point_to_key(tuple(point_b))
    entry_a = directed.get(key_a)
    entry_b = directed.get(key_b)
    if entry_a is None or entry_b is None:
        return None
    e_a = entry_a.get('energy')
    e_b = entry_b.get('energy')
    if e_a is None or e_b is None:
        return None
    return abs(float(e_a) - float(e_b))


def calculate_neighbor_geometry_rmsd(rotor_dict: dict,
                                     point_a: list,
                                     point_b: list,
                                     ) -> Optional[float]:
    """
    Compute the distance-matrix RMSD between the optimized geometries of two
    neighboring scan points.

    Uses the full molecular geometry (all atoms). This is a lightweight proxy
    for detecting branch jumps where non-rotor atoms rearrange significantly.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        point_a (list): ``[phi0, phi1]`` first point.
        point_b (list): ``[phi0, phi1]`` second point.

    Returns:
        float or None: The RMSD of the two distance matrices (Angstrom), or
            ``None`` if either point lacks geometry data.
    """
    from arc.species.converter import compare_confs
    directed = rotor_dict.get('directed_scan', {})
    key_a = point_to_key(tuple(point_a))
    key_b = point_to_key(tuple(point_b))
    entry_a = directed.get(key_a)
    entry_b = directed.get(key_b)
    if entry_a is None or entry_b is None:
        return None
    xyz_a = entry_a.get('xyz')
    xyz_b = entry_b.get('xyz')
    if not isinstance(xyz_a, dict) or not isinstance(xyz_b, dict):
        return None
    if 'coords' not in xyz_a or 'coords' not in xyz_b:
        return None
    try:
        return compare_confs(xyz_a, xyz_b, rmsd_score=True)
    except Exception:
        return None


def classify_neighbor_edge_continuity(rotor_dict: dict,
                                      point_a: list,
                                      point_b: list,
                                      energy_threshold: float = VALIDATION_ENERGY_JUMP_THRESHOLD,
                                      rmsd_threshold: float = VALIDATION_GEOMETRY_RMSD_THRESHOLD,
                                      ) -> dict:
    """
    Classify a neighbor edge as continuous or suspicious.

    An edge is flagged as discontinuous if:
      - energy jump exceeds ``energy_threshold``, OR
      - geometry RMSD exceeds ``rmsd_threshold``

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        point_a (list): ``[phi0, phi1]`` first point.
        point_b (list): ``[phi0, phi1]`` second point.
        energy_threshold (float): Max acceptable energy jump (kJ/mol).
        rmsd_threshold (float): Max acceptable distance-matrix RMSD (Angstrom).

    Returns:
        dict: Classification with keys ``'continuous'`` (bool), ``'energy_jump'``,
              ``'geometry_rmsd'``, ``'reasons'`` (list of str).
    """
    e_jump = calculate_neighbor_energy_jump(rotor_dict, point_a, point_b)
    g_rmsd = calculate_neighbor_geometry_rmsd(rotor_dict, point_a, point_b)
    reasons = []
    if e_jump is not None and e_jump > energy_threshold:
        reasons.append(f'energy_jump={e_jump:.2f}')
    if g_rmsd is not None and g_rmsd > rmsd_threshold:
        reasons.append(f'geometry_rmsd={g_rmsd:.4f}')
    return {
        'continuous': len(reasons) == 0,
        'energy_jump': round(e_jump, 4) if e_jump is not None else None,
        'geometry_rmsd': round(g_rmsd, 6) if g_rmsd is not None else None,
        'reasons': reasons,
    }


def _is_periodic_edge(point_a: list, point_b: list, increment: float) -> bool:
    """Check whether an edge between two points wraps across the -180/+180 boundary."""
    for i in range(2):
        diff = abs(point_a[i] - point_b[i])
        if diff > 360.0 - 1.5 * increment:
            return True
    return False


def check_periodic_edge_consistency(rotor_dict: dict,
                                    point_a: list,
                                    point_b: list,
                                    energy_threshold: float = VALIDATION_PERIODIC_ENERGY_THRESHOLD,
                                    rmsd_threshold: float = VALIDATION_PERIODIC_RMSD_THRESHOLD,
                                    ) -> dict:
    """
    Check consistency of an edge that wraps across the periodic boundary.

    Periodic edges should have similar energies/geometries if the surface
    is well-behaved across the -180/+180 wrap.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        point_a (list): ``[phi0, phi1]`` first point.
        point_b (list): ``[phi0, phi1]`` second point (the wrap partner).
        energy_threshold (float): Max acceptable energy mismatch (kJ/mol).
        rmsd_threshold (float): Max acceptable geometry RMSD (Angstrom).

    Returns:
        dict: With keys ``'consistent'`` (bool), ``'energy_mismatch'``,
              ``'geometry_rmsd'``, ``'reasons'`` (list of str).
    """
    e_jump = calculate_neighbor_energy_jump(rotor_dict, point_a, point_b)
    g_rmsd = calculate_neighbor_geometry_rmsd(rotor_dict, point_a, point_b)
    reasons = []
    if e_jump is not None and e_jump > energy_threshold:
        reasons.append(f'periodic_energy_mismatch={e_jump:.2f}')
    if g_rmsd is not None and g_rmsd > rmsd_threshold:
        reasons.append(f'periodic_geometry_mismatch={g_rmsd:.4f}')
    return {
        'consistent': len(reasons) == 0,
        'energy_mismatch': round(e_jump, 4) if e_jump is not None else None,
        'geometry_rmsd': round(g_rmsd, 6) if g_rmsd is not None else None,
        'reasons': reasons,
    }


def detect_branch_jump_points(rotor_dict: dict,
                              increment: float,
                              energy_threshold: float = VALIDATION_ENERGY_JUMP_THRESHOLD,
                              rmsd_threshold: float = VALIDATION_GEOMETRY_RMSD_THRESHOLD,
                              min_suspicious_edges: int = VALIDATION_BRANCH_JUMP_EDGE_COUNT,
                              ) -> List[list]:
    """
    Detect points suspected of being on a different PES branch.

    A point is flagged if it is connected to ``>= min_suspicious_edges``
    discontinuous neighbor edges.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.
        energy_threshold (float): Energy jump threshold for edge classification.
        rmsd_threshold (float): Geometry RMSD threshold for edge classification.
        min_suspicious_edges (int): Minimum suspicious edges to flag a point.

    Returns:
        list: Flagged points ``[[phi0, phi1], ...]``.
    """
    suspicious_count = {}  # key_str -> count
    for pt_a, pt_b in iter_sampled_neighbor_edges(rotor_dict, increment):
        classification = classify_neighbor_edge_continuity(
            rotor_dict, pt_a, pt_b, energy_threshold, rmsd_threshold)
        if not classification['continuous']:
            for pt in [pt_a, pt_b]:
                k = point_to_key(tuple(pt))
                suspicious_count[k] = suspicious_count.get(k, 0) + 1
    flagged = []
    for k, count in suspicious_count.items():
        if count >= min_suspicious_edges:
            flagged.append([float(k[0]), float(k[1])])
    return flagged


def run_adaptive_surface_validation(rotor_dict: dict,
                                    increment: float,
                                    ) -> dict:
    """
    Compute a full surface-validation summary for an adaptive 2D scan.

    Checks all sampled neighbor edges for energy and geometry continuity,
    identifies periodic wraparound inconsistencies, and flags suspected
    branch-jump points.  Results are stored in a YAML-serializable dict.

    This function does **not** modify stored scan data or energies.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.

    Returns:
        dict: The validation state dict (also stored in
              ``rotor_dict['adaptive_scan']['validation']``).
    """
    validation = _make_empty_validation_state()
    thresholds = validation['thresholds']

    # --- Neighbor edge continuity ---
    discontinuous = []
    energy_jumps = []
    geometry_rmsds = []
    n_edges = 0

    for pt_a, pt_b in iter_sampled_neighbor_edges(rotor_dict, increment):
        n_edges += 1
        cl = classify_neighbor_edge_continuity(
            rotor_dict, pt_a, pt_b,
            thresholds['energy_jump'], thresholds['geometry_rmsd'])
        if cl['energy_jump'] is not None:
            energy_jumps.append(cl['energy_jump'])
        if cl['geometry_rmsd'] is not None:
            geometry_rmsds.append(cl['geometry_rmsd'])
        if not cl['continuous']:
            discontinuous.append({
                'point_a': [round(x, 2) for x in pt_a],
                'point_b': [round(x, 2) for x in pt_b],
                'energy_jump': cl['energy_jump'],
                'geometry_rmsd': cl['geometry_rmsd'],
                'reasons': cl['reasons'],
            })

    validation['neighbor_edges_checked'] = n_edges
    validation['discontinuous_edges'] = discontinuous

    if energy_jumps:
        validation['energy_jump_summary'] = {
            'min': round(min(energy_jumps), 4),
            'max': round(max(energy_jumps), 4),
            'mean': round(sum(energy_jumps) / len(energy_jumps), 4),
            'count': len(energy_jumps),
        }
    if geometry_rmsds:
        validation['geometry_rmsd_summary'] = {
            'min': round(min(geometry_rmsds), 6),
            'max': round(max(geometry_rmsds), 6),
            'mean': round(sum(geometry_rmsds) / len(geometry_rmsds), 6),
            'count': len(geometry_rmsds),
        }

    # --- Periodic edge consistency ---
    periodic_issues = []
    n_periodic = 0
    directed = rotor_dict.get('directed_scan', {})
    for key_tuple in directed.keys():
        pt = [float(key_tuple[0]), float(key_tuple[1])]
        for d0, d1 in _periodic_neighbor_offsets(increment):
            nb = [_normalize_angle_key(pt[0] + d0), _normalize_angle_key(pt[1] + d1)]
            if _is_periodic_edge(pt, nb, increment):
                nb_key = point_to_key(tuple(nb))
                if nb_key in directed:
                    n_periodic += 1
                    pc = check_periodic_edge_consistency(
                        rotor_dict, pt, nb,
                        thresholds['periodic_energy'], thresholds['periodic_rmsd'])
                    if not pc['consistent']:
                        periodic_issues.append({
                            'point_a': [round(x, 2) for x in pt],
                            'point_b': [round(x, 2) for x in nb],
                            'energy_mismatch': pc['energy_mismatch'],
                            'geometry_rmsd': pc['geometry_rmsd'],
                            'reasons': pc['reasons'],
                        })

    validation['periodic_edges_checked'] = n_periodic
    validation['periodic_inconsistencies'] = periodic_issues

    # --- Branch-jump detection ---
    flagged = detect_branch_jump_points(
        rotor_dict, increment,
        thresholds['energy_jump'], thresholds['geometry_rmsd'],
        thresholds['branch_jump_edge_count'])
    validation['branch_jump_points'] = flagged

    # --- Status ---
    if n_edges == 0:
        validation['status'] = 'no_edges'
        validation['notes'].append('No neighbor edges found; too few sampled points for validation.')
    else:
        validation['status'] = 'complete'
        if discontinuous:
            validation['notes'].append(
                f'{len(discontinuous)} of {n_edges} neighbor edges are discontinuous.')
        if periodic_issues:
            validation['notes'].append(
                f'{len(periodic_issues)} periodic boundary inconsistencies found.')
        if flagged:
            validation['notes'].append(
                f'{len(flagged)} points suspected of branch jumps.')
        if not discontinuous and not periodic_issues and not flagged:
            validation['notes'].append('Surface passed all continuity checks.')

    return validation


def update_adaptive_validation_state(rotor_dict: dict,
                                     increment: float,
                                     ) -> None:
    """
    Run surface validation and store results in the rotor's adaptive state.

    Only runs for adaptive 2D brute-force scans.  Does nothing for dense or
    other scan types.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.
    """
    if not is_adaptive_enabled(rotor_dict):
        return
    state = rotor_dict.get('adaptive_scan', {})
    if not state.get('enabled', False):
        return
    validation = run_adaptive_surface_validation(rotor_dict, increment)
    state['validation'] = validation
    # Log summary
    n_disc = len(validation.get('discontinuous_edges', []))
    n_periodic = len(validation.get('periodic_inconsistencies', []))
    n_branch = len(validation.get('branch_jump_points', []))
    if n_disc or n_periodic or n_branch:
        logger.warning(f'Adaptive scan surface validation: '
                       f'{n_disc} discontinuous edges, '
                       f'{n_periodic} periodic inconsistencies, '
                       f'{n_branch} branch-jump suspects.')


# ===========================================================================
# Coupling metrics, surface quality, and ND classification
# ===========================================================================

# Thresholds for coupling classification (V1 defaults)
COUPLING_NONSEP_THRESHOLD = 0.15        # Relative separable-fit error above this → coupled
COUPLING_CROSS_TERM_THRESHOLD = 0.10    # Cross-term fraction above this → coupled
QUALITY_MIN_POINTS = 9                  # Minimum completed points for any analysis
QUALITY_FAILED_FRACTION_LIMIT = 0.20    # Above this → unreliable
QUALITY_INVALID_FRACTION_LIMIT = 0.15   # Above this → unreliable
QUALITY_DISC_EDGE_FRACTION_LIMIT = 0.25 # Above this → unreliable


def extract_adaptive_2d_surface_arrays(rotor_dict: dict) -> dict:
    """
    Extract sampled 2D coordinates and energies into numpy arrays.

    Only includes completed points that have non-None energy.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.

    Returns:
        dict: ``{'phi0': np.array, 'phi1': np.array, 'energy': np.array, 'n_points': int}``
    """
    import numpy as np
    directed = rotor_dict.get('directed_scan', {})
    phi0_list, phi1_list, energy_list = [], [], []
    for key, entry in directed.items():
        e = entry.get('energy')
        if e is not None and len(key) == 2:
            phi0_list.append(float(key[0]))
            phi1_list.append(float(key[1]))
            energy_list.append(float(e))
    return {
        'phi0': np.array(phi0_list, dtype=np.float64),
        'phi1': np.array(phi1_list, dtype=np.float64),
        'energy': np.array(energy_list, dtype=np.float64),
        'n_points': len(phi0_list),
    }


def fit_separable_surface_proxy(surface_data: dict) -> dict:
    """
    Build a simple separable approximation E(phi0, phi1) ≈ f(phi0) + g(phi1) + c.

    The separable components are estimated by discrete averaging:
        f(phi0_i) = mean_over_phi1 { E(phi0_i, phi1_j) } - c
        g(phi1_j) = mean_over_phi0 { E(phi0_i, phi1_j) } - c
        c          = overall mean of all sampled energies

    This works on irregularly sampled data by grouping points by their
    phi0 and phi1 keys.

    Args:
        surface_data (dict): As returned by :func:`extract_adaptive_2d_surface_arrays`.

    Returns:
        dict: ``{'c': float, 'f_values': dict, 'g_values': dict, 'separable_predictions': np.array}``
            where ``f_values[phi0_key] = f(phi0)`` and ``g_values[phi1_key] = g(phi1)``.
    """
    import numpy as np
    phi0 = surface_data['phi0']
    phi1 = surface_data['phi1']
    energy = surface_data['energy']
    n = surface_data['n_points']

    if n == 0:
        return {'c': 0.0, 'f_values': {}, 'g_values': {}, 'separable_predictions': np.array([])}

    c = float(np.mean(energy))

    # Group energies by phi0 key and phi1 key
    phi0_groups = {}  # phi0_str -> list of energies
    phi1_groups = {}  # phi1_str -> list of energies
    for i in range(n):
        k0 = f'{phi0[i]:.2f}'
        k1 = f'{phi1[i]:.2f}'
        phi0_groups.setdefault(k0, []).append(energy[i])
        phi1_groups.setdefault(k1, []).append(energy[i])

    f_values = {k: float(np.mean(v)) - c for k, v in phi0_groups.items()}
    g_values = {k: float(np.mean(v)) - c for k, v in phi1_groups.items()}

    # Build separable predictions at each sampled point
    sep_pred = np.zeros(n, dtype=np.float64)
    for i in range(n):
        k0 = f'{phi0[i]:.2f}'
        k1 = f'{phi1[i]:.2f}'
        sep_pred[i] = f_values.get(k0, 0.0) + g_values.get(k1, 0.0) + c

    return {
        'c': c,
        'f_values': f_values,
        'g_values': g_values,
        'separable_predictions': sep_pred,
    }


def calculate_separable_fit_error(surface_data: dict, separable_fit: dict) -> float:
    """
    Calculate the RMS error of the separable fit relative to the energy range.

    Returns the RMSE of (E_actual - E_separable) normalized by the range of E_actual.
    A value near 0 means the surface is well-described by a separable model.

    Args:
        surface_data (dict): As returned by :func:`extract_adaptive_2d_surface_arrays`.
        separable_fit (dict): As returned by :func:`fit_separable_surface_proxy`.

    Returns:
        float: Normalized RMSE (dimensionless). Returns 0.0 if insufficient data.
    """
    import numpy as np
    energy = surface_data['energy']
    sep_pred = separable_fit['separable_predictions']
    if len(energy) < 2 or len(sep_pred) < 2:
        return 0.0
    residuals = energy - sep_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    e_range = float(np.max(energy) - np.min(energy))
    if e_range < 1e-10:
        return 0.0
    return rmse / e_range


def calculate_nonseparability_score(surface_data: dict, separable_fit: dict) -> float:
    """
    Compute a nonseparability score: the fraction of total variance NOT explained
    by the separable model.

    Score near 0 → separable; score near 1 → strongly coupled.

    Formula: 1 - R² where R² = 1 - SS_res / SS_tot.

    Args:
        surface_data (dict): As returned by :func:`extract_adaptive_2d_surface_arrays`.
        separable_fit (dict): As returned by :func:`fit_separable_surface_proxy`.

    Returns:
        float: Nonseparability score in [0, 1]. Returns 0.0 if insufficient data.
    """
    import numpy as np
    energy = surface_data['energy']
    sep_pred = separable_fit['separable_predictions']
    if len(energy) < 3:
        return 0.0
    ss_tot = float(np.sum((energy - np.mean(energy)) ** 2))
    if ss_tot < 1e-10:
        return 0.0
    ss_res = float(np.sum((energy - sep_pred) ** 2))
    r_squared = 1.0 - ss_res / ss_tot
    return max(0.0, min(1.0, 1.0 - r_squared))


def calculate_cross_term_strength(surface_data: dict, separable_fit: dict) -> float:
    """
    Estimate the strength of cross-term coupling as the fraction of total energy
    variance attributable to the non-separable residual.

    This is essentially the same as the nonseparability score but expressed as
    the ratio of residual variance to total variance.

    Args:
        surface_data (dict): As returned by :func:`extract_adaptive_2d_surface_arrays`.
        separable_fit (dict): As returned by :func:`fit_separable_surface_proxy`.

    Returns:
        float: Cross-term strength fraction in [0, 1].
    """
    return calculate_nonseparability_score(surface_data, separable_fit)


def calculate_low_energy_path_coupling(surface_data: dict) -> float:
    """
    Heuristic for low-energy-path coupling: measures whether the minimum-energy
    path through the 2D surface is axis-aligned (separable) or diagonal (coupled).

    Method: among the lowest 25% of energy points, compute the correlation
    coefficient between phi0 and phi1.  High |correlation| suggests diagonal
    low-energy valleys → coupling.

    Args:
        surface_data (dict): As returned by :func:`extract_adaptive_2d_surface_arrays`.

    Returns:
        float: Absolute correlation of phi0 and phi1 among low-energy points, in [0, 1].
               Returns 0.0 if insufficient data.
    """
    import numpy as np
    n = surface_data['n_points']
    if n < 4:
        return 0.0
    energy = surface_data['energy']
    phi0 = surface_data['phi0']
    phi1 = surface_data['phi1']

    # Select the lowest 25% of points
    threshold = np.percentile(energy, 25)
    mask = energy <= threshold
    if mask.sum() < 3:
        return 0.0

    # Compute correlation between phi0 and phi1 in the low-energy subset
    # Use sin/cos to handle periodicity
    sin0 = np.sin(np.radians(phi0[mask]))
    cos0 = np.cos(np.radians(phi0[mask]))
    sin1 = np.sin(np.radians(phi1[mask]))
    cos1 = np.cos(np.radians(phi1[mask]))

    # Cross-correlation: max of |corr(sin0,sin1)|, |corr(sin0,cos1)|, etc.
    max_corr = 0.0
    for a in [sin0, cos0]:
        for b in [sin1, cos1]:
            if np.std(a) > 1e-10 and np.std(b) > 1e-10:
                corr = abs(float(np.corrcoef(a, b)[0, 1]))
                if not np.isnan(corr):
                    max_corr = max(max_corr, corr)
    return max_corr


def compute_coupling_metrics(rotor_dict: dict) -> dict:
    """
    Compute all coupling metrics for an adaptive 2D scan.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.

    Returns:
        dict: The coupling_metrics dict (also stored in ``rotor_dict['adaptive_scan']``).
    """
    metrics = {
        'enabled': True,
        'status': 'not_run',
        'nonseparability_score': None,
        'cross_term_strength': None,
        'low_energy_path_coupling': None,
        'separable_fit_error': None,
        'coupled_fit_proxy': None,
        'thresholds': {
            'nonseparability': COUPLING_NONSEP_THRESHOLD,
            'cross_term': COUPLING_CROSS_TERM_THRESHOLD,
        },
        'notes': [],
    }

    surface = extract_adaptive_2d_surface_arrays(rotor_dict)
    if surface['n_points'] < QUALITY_MIN_POINTS:
        metrics['status'] = 'insufficient_data'
        metrics['notes'].append(f'Only {surface["n_points"]} points; need >= {QUALITY_MIN_POINTS}.')
        return metrics

    sep_fit = fit_separable_surface_proxy(surface)
    metrics['nonseparability_score'] = round(calculate_nonseparability_score(surface, sep_fit), 6)
    metrics['cross_term_strength'] = round(calculate_cross_term_strength(surface, sep_fit), 6)
    metrics['separable_fit_error'] = round(calculate_separable_fit_error(surface, sep_fit), 6)
    metrics['low_energy_path_coupling'] = round(calculate_low_energy_path_coupling(surface), 6)
    metrics['coupled_fit_proxy'] = metrics['nonseparability_score']
    metrics['status'] = 'complete'
    return metrics


def update_coupling_metrics(rotor_dict: dict) -> None:
    """
    Compute and store coupling metrics on the rotor's adaptive state.

    Only runs for adaptive 2D brute-force scans.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
    """
    if not is_adaptive_enabled(rotor_dict):
        return
    metrics = compute_coupling_metrics(rotor_dict)
    rotor_dict.setdefault('adaptive_scan', {})['coupling_metrics'] = metrics


# ---------------------------------------------------------------------------
# Surface quality metrics
# ---------------------------------------------------------------------------

def calculate_coverage_fraction(rotor_dict: dict, increment: float) -> float:
    """
    Fraction of the full dense grid that has been visited (completed/failed/invalid).

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.

    Returns:
        float: Coverage fraction in [0, 1].
    """
    # Use int(360/increment) per dimension (not +1) because angle normalization
    # maps +180 to -180, so the endpoint duplicates the start.
    n_grid = int(360 / increment) ** 2
    if n_grid == 0:
        return 0.0
    n_visited = len(_all_visited_keys(rotor_dict))
    return min(1.0, n_visited / n_grid)


def calculate_failed_fraction(rotor_dict: dict) -> float:
    """Fraction of submitted points that failed."""
    state = rotor_dict.get('adaptive_scan', {})
    total = (len(state.get('completed_points', []))
             + len(state.get('failed_points', []))
             + len(state.get('invalid_points', [])))
    if total == 0:
        return 0.0
    return len(state.get('failed_points', [])) / total


def calculate_invalid_fraction(rotor_dict: dict) -> float:
    """Fraction of submitted points that are invalid (non-isomorphic)."""
    state = rotor_dict.get('adaptive_scan', {})
    total = (len(state.get('completed_points', []))
             + len(state.get('failed_points', []))
             + len(state.get('invalid_points', [])))
    if total == 0:
        return 0.0
    return len(state.get('invalid_points', [])) / total


def calculate_validation_warning_fraction(rotor_dict: dict) -> float:
    """
    Fraction of checked neighbor edges that were flagged as discontinuous.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.

    Returns:
        float: Warning fraction in [0, 1].
    """
    validation = rotor_dict.get('adaptive_scan', {}).get('validation', {})
    n_edges = validation.get('neighbor_edges_checked', 0)
    if n_edges == 0:
        return 0.0
    n_disc = len(validation.get('discontinuous_edges', []))
    return n_disc / n_edges


def calculate_periodic_consistency_score(rotor_dict: dict) -> float:
    """
    Periodic consistency as (1 - fraction of inconsistent periodic edges).

    Returns 1.0 if all periodic edges are consistent or if no edges were checked.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.

    Returns:
        float: Score in [0, 1]. Higher is better.
    """
    validation = rotor_dict.get('adaptive_scan', {}).get('validation', {})
    n_periodic = validation.get('periodic_edges_checked', 0)
    if n_periodic == 0:
        return 1.0
    n_issues = len(validation.get('periodic_inconsistencies', []))
    return 1.0 - n_issues / n_periodic


def calculate_overall_quality_score(rotor_dict: dict, increment: float) -> float:
    """
    Compute a composite surface quality score in [0, 1].

    Weighted combination:
      - 30% coverage fraction
      - 25% (1 - failed fraction)
      - 20% (1 - invalid fraction)
      - 15% (1 - validation warning fraction)
      - 10% periodic consistency score

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.

    Returns:
        float: Quality score in [0, 1]. Higher is better.
    """
    cov = calculate_coverage_fraction(rotor_dict, increment)
    fail = calculate_failed_fraction(rotor_dict)
    inv = calculate_invalid_fraction(rotor_dict)
    warn = calculate_validation_warning_fraction(rotor_dict)
    per = calculate_periodic_consistency_score(rotor_dict)
    return 0.30 * cov + 0.25 * (1.0 - fail) + 0.20 * (1.0 - inv) + 0.15 * (1.0 - warn) + 0.10 * per


def compute_surface_quality_metrics(rotor_dict: dict, increment: float) -> dict:
    """
    Compute all surface quality metrics for an adaptive 2D scan.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.

    Returns:
        dict: The surface_quality dict.
    """
    state = rotor_dict.get('adaptive_scan', {})
    n_completed = len(state.get('completed_points', []))
    metrics = {
        'enabled': True,
        'status': 'complete',
        'coverage_fraction': round(calculate_coverage_fraction(rotor_dict, increment), 4),
        'completed_fraction': None,
        'failed_fraction': round(calculate_failed_fraction(rotor_dict), 4),
        'invalid_fraction': round(calculate_invalid_fraction(rotor_dict), 4),
        'validation_warning_fraction': round(calculate_validation_warning_fraction(rotor_dict), 4),
        'periodic_consistency_score': round(calculate_periodic_consistency_score(rotor_dict), 4),
        'quality_score': round(calculate_overall_quality_score(rotor_dict, increment), 4),
        'thresholds': {
            'min_points': QUALITY_MIN_POINTS,
            'failed_fraction_limit': QUALITY_FAILED_FRACTION_LIMIT,
            'invalid_fraction_limit': QUALITY_INVALID_FRACTION_LIMIT,
            'disc_edge_fraction_limit': QUALITY_DISC_EDGE_FRACTION_LIMIT,
        },
        'notes': [],
    }
    total = (n_completed + len(state.get('failed_points', []))
             + len(state.get('invalid_points', [])))
    if total > 0:
        metrics['completed_fraction'] = round(n_completed / total, 4)
    if n_completed < QUALITY_MIN_POINTS:
        metrics['status'] = 'insufficient_data'
        metrics['notes'].append(f'Only {n_completed} completed points; need >= {QUALITY_MIN_POINTS}.')
    return metrics


def update_surface_quality_metrics(rotor_dict: dict, increment: float) -> None:
    """
    Compute and store surface quality metrics on the rotor's adaptive state.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.
    """
    if not is_adaptive_enabled(rotor_dict):
        return
    metrics = compute_surface_quality_metrics(rotor_dict, increment)
    rotor_dict.setdefault('adaptive_scan', {})['surface_quality'] = metrics


# ---------------------------------------------------------------------------
# ND rotor classification
# ---------------------------------------------------------------------------

def classify_adaptive_nd_rotor(rotor_dict: dict, increment: float) -> dict:
    """
    Classify an adaptive 2D ND rotor as separable, coupled, or unreliable.

    Logic:
      1. If surface quality is insufficient or too many failures → ``"unreliable"``
      2. If quality is acceptable and nonseparability is below threshold → ``"separable"``
      3. If quality is acceptable and nonseparability is above threshold → ``"coupled"``

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.

    Returns:
        dict: The nd_classification dict.
    """
    result = {
        'enabled': True,
        'status': 'not_run',
        'classification': None,
        'confidence': None,
        'reason': None,
        'recommended_action': None,
        'notes': [],
    }

    # Ensure we have quality and coupling metrics
    state = rotor_dict.get('adaptive_scan', {})
    quality = state.get('surface_quality', {})
    coupling = state.get('coupling_metrics', {})

    # Check if we have enough data
    if quality.get('status') == 'insufficient_data' or coupling.get('status') == 'insufficient_data':
        result['status'] = 'insufficient_data'
        result['classification'] = 'unreliable'
        result['reason'] = 'Insufficient completed points for reliable analysis.'
        result['recommended_action'] = 'fallback_due_to_surface_quality'
        result['confidence'] = 0.0
        return result

    # Check quality thresholds for unreliable
    failed_frac = quality.get('failed_fraction', 0.0) or 0.0
    invalid_frac = quality.get('invalid_fraction', 0.0) or 0.0
    warn_frac = quality.get('validation_warning_fraction', 0.0) or 0.0
    quality_score = quality.get('quality_score', 0.0) or 0.0

    unreliable_reasons = []
    if failed_frac > QUALITY_FAILED_FRACTION_LIMIT:
        unreliable_reasons.append(f'failed_fraction={failed_frac:.2f} > {QUALITY_FAILED_FRACTION_LIMIT}')
    if invalid_frac > QUALITY_INVALID_FRACTION_LIMIT:
        unreliable_reasons.append(f'invalid_fraction={invalid_frac:.2f} > {QUALITY_INVALID_FRACTION_LIMIT}')
    if warn_frac > QUALITY_DISC_EDGE_FRACTION_LIMIT:
        unreliable_reasons.append(f'disc_edge_fraction={warn_frac:.2f} > {QUALITY_DISC_EDGE_FRACTION_LIMIT}')

    if unreliable_reasons:
        result['status'] = 'complete'
        result['classification'] = 'unreliable'
        result['reason'] = '; '.join(unreliable_reasons)
        result['recommended_action'] = 'fallback_due_to_surface_quality'
        result['confidence'] = round(max(0.0, 1.0 - quality_score), 2)
        result['notes'].append('Surface quality issues prevent reliable coupling analysis.')
        return result

    # Classify based on coupling
    nonsep = coupling.get('nonseparability_score', 0.0) or 0.0
    cross_term = coupling.get('cross_term_strength', 0.0) or 0.0

    is_coupled = (nonsep > COUPLING_NONSEP_THRESHOLD or cross_term > COUPLING_CROSS_TERM_THRESHOLD)

    result['status'] = 'complete'
    if is_coupled:
        result['classification'] = 'coupled'
        result['reason'] = (f'nonseparability={nonsep:.4f} (threshold={COUPLING_NONSEP_THRESHOLD}), '
                            f'cross_term={cross_term:.4f} (threshold={COUPLING_CROSS_TERM_THRESHOLD})')
        result['recommended_action'] = 'retain_as_coupled_2d_surface'
        result['confidence'] = round(min(1.0, nonsep / COUPLING_NONSEP_THRESHOLD), 2)
    else:
        result['classification'] = 'separable'
        result['reason'] = (f'nonseparability={nonsep:.4f} (threshold={COUPLING_NONSEP_THRESHOLD}), '
                            f'cross_term={cross_term:.4f} (threshold={COUPLING_CROSS_TERM_THRESHOLD})')
        result['recommended_action'] = 'treat_as_separable_1d_like'
        result['confidence'] = round(min(1.0, (COUPLING_NONSEP_THRESHOLD - nonsep) / COUPLING_NONSEP_THRESHOLD), 2)

    return result


def update_nd_classification(rotor_dict: dict, increment: float) -> None:
    """
    Run coupling metrics, surface quality, and classification, and store all
    results on the rotor's adaptive state.

    Only runs for adaptive 2D brute-force scans.

    Args:
        rotor_dict (dict): A single entry from ``species.rotors_dict``.
        increment (float): The scan resolution in degrees.
    """
    if not is_adaptive_enabled(rotor_dict):
        return
    state = rotor_dict.setdefault('adaptive_scan', {})

    # Compute coupling metrics
    update_coupling_metrics(rotor_dict)

    # Compute surface quality
    update_surface_quality_metrics(rotor_dict, increment)

    # Classify
    classification = classify_adaptive_nd_rotor(rotor_dict, increment)
    state['nd_classification'] = classification

    # Log
    cls = classification.get('classification', 'unknown')
    reason = classification.get('reason', '')
    action = classification.get('recommended_action', '')
    logger.info(f'Adaptive 2D rotor classified as "{cls}": {reason}. '
                f'Recommended: {action}.')
