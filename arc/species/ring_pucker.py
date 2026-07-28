#!/usr/bin/env python3
# encoding: utf-8

"""
Ring puckering conformer generation for ARC.

Provides Cremer-Pople puckering coordinates and canonical pucker-state
enumeration for 5- and 6-membered rings (and fused/bridged bicyclic systems).
These are used to both seed ring-conformer sampling and to score ring-pucker
coverage of a conformer ensemble in arc.species.conformers.

Important: every function in this module that takes ``ring_coords`` assumes the atoms are
supplied in ring-connectivity order (i.e., consecutive entries are bonded, and the last entry
is bonded back to the first). Establishing and verifying that ordering is the caller's
responsibility, not this module's; passing atoms in an arbitrary or scrambled order will
silently produce a meaningless Cremer-Pople decomposition unless ``validate_ring_order`` is
used to catch it first.
"""

import collections
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


class RingPuckerError(Exception):
    """Raised when ring puckering analysis or generation fails."""


#: Minimum acceptable norm of the fitted ring-plane normal vector before it is treated as
#: numerically degenerate (e.g., collinear or otherwise ill-defined ring points).
_NORMAL_EPS = 1e-8

#: Total puckering amplitude Q (Angstrom) below which a ring is classified as 'planar' rather
#: than assigned a definite pucker label.
PLANARITY_AMPLITUDE_THRESHOLD = 0.1

#: Small numerical tolerance (degrees) used when binning phi angles into half-open intervals,
#: so that points that fall exactly on a bin boundary (subject to floating-point noise) bin
#: deterministically instead of via round-half-to-even.
_BIN_TOL_DEG = 1e-9


@dataclass
class CremerPopleParams:
    """Cremer-Pople puckering coordinates of a single ring.

    Attributes:
        ring_size (int): Number of atoms in the ring.
        amplitude (float): Total puckering amplitude, Q (Angstrom).
        terms (List[Tuple[int, float, Optional[float]]]): The individual (m, q_m, phi_m_deg)
            puckering terms for 2 <= m <= floor((ring_size - 1) / 2). phi_m_deg is None for the
            single (phase-less) q_(N/2) term present when ring_size is even.
        q2 (Optional[float]): The m=2 puckering amplitude, if present.
        phi2_deg (Optional[float]): The m=2 puckering phase angle in degrees, if present.
        q3 (Optional[float]): The signed q_(N/2) puckering coordinate for a 6-ring.
        theta_deg (Optional[float]): Spherical polar angle (0-180 deg), defined for 6-rings.
        phi_deg (Optional[float]): Spherical azimuthal angle (0-360 deg), defined for 6-rings.
    """

    ring_size: int
    amplitude: float
    terms: List[Tuple[int, float, Optional[float]]] = field(default_factory=list)
    q2: Optional[float] = None
    phi2_deg: Optional[float] = None
    q3: Optional[float] = None
    theta_deg: Optional[float] = None
    phi_deg: Optional[float] = None


def ring_mean_plane(ring_coords: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Cremer-Pople mean plane of a ring.

    Args:
        ring_coords (Sequence[Sequence[float]]): Cartesian coordinates of the ring atoms, in
            ring-connectivity order.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The mean-plane centroid r0 (length-3 array),
        the unit normal vector n-hat (length-3 array), and the per-atom out-of-plane
        displacements z_j (length-N array), all in the input coordinate frame.

    Raises:
        RingPuckerError: If ``ring_coords`` is not an (N x 3) array with N >= 3, or if the
            fitted ring-plane normal is numerically degenerate (the ring points are collinear,
            degenerate, or not supplied in ring-connectivity order).
    """
    coords = np.asarray(ring_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] < 3:
        raise RingPuckerError(
            f'ring_coords must be an (N x 3) array with N >= 3, got shape {coords.shape}.')
    n = coords.shape[0]
    r0 = np.mean(coords, axis=0)
    shifted = coords - r0
    idx = np.arange(n)
    angles = 2.0 * np.pi * idx / n
    r1 = np.sum(shifted * np.sin(angles)[:, None], axis=0)
    r2 = np.sum(shifted * np.cos(angles)[:, None], axis=0)
    normal = np.cross(r1, r2)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < _NORMAL_EPS:
        raise RingPuckerError(
            'Could not determine a well-defined ring-plane normal (the ring points are '
            'collinear, degenerate, or not supplied in ring-connectivity order).')
    normal = normal / normal_norm
    z = shifted @ normal
    return r0, normal, z


def _ring_z_displacements(ring_coords: Sequence[Sequence[float]]) -> Tuple[np.ndarray, int]:
    """Compute the Cremer-Pople out-of-plane displacements z_j of a ring.

    Args:
        ring_coords (Sequence[Sequence[float]]): Cartesian coordinates of the ring atoms, in
            ring-connectivity order.

    Returns:
        Tuple[np.ndarray, int]: The z_j displacements (length N array) and the ring size N.
    """
    coords = np.asarray(ring_coords, dtype=float)
    _, _, z = ring_mean_plane(ring_coords)
    return z, coords.shape[0]


def validate_ring_order(ring_coords: Sequence[Sequence[float]], max_bond: float = 2.2) -> None:
    """Check that ring atoms are supplied in ring-connectivity order.

    The Cremer-Pople decomposition computed by this module assumes that consecutive entries of
    ``ring_coords`` are bonded to each other, and that the last entry is bonded back to the
    first. This function does not know the true bonding pattern; it only heuristically checks
    that every consecutive pair (including the closure bond) is within a plausible bonding
    distance of each other. Establishing and verifying the true connectivity order is the
    caller's responsibility, not this module's.

    Args:
        ring_coords (Sequence[Sequence[float]]): Cartesian coordinates of the ring atoms, in
            ring-connectivity order.
        max_bond (float): The maximum allowed distance (Angstrom) between consecutive ring atoms.

    Returns:
        None

    Raises:
        RingPuckerError: If any consecutive pair of ring atoms (including the closure bond)
            exceeds ``max_bond``.
    """
    coords = np.asarray(ring_coords, dtype=float)
    n = coords.shape[0]
    for i in range(n):
        j = (i + 1) % n
        distance = float(np.linalg.norm(coords[i] - coords[j]))
        if distance > max_bond:
            raise RingPuckerError(
                f'Ring atoms at indices {i} and {j} are {distance:.3f} Angstrom apart, exceeding '
                f'max_bond={max_bond}; ring_coords do not appear to be in ring-connectivity order.')


def puckering_amplitude(ring_coords: Sequence[Sequence[float]]) -> float:
    """Compute the total Cremer-Pople puckering amplitude of a single ring.

    Important: ``ring_coords`` must be supplied in ring-connectivity order (consecutive entries
    bonded, last bonded back to first). Establishing and verifying that order is the caller's
    responsibility, not this module's.

    Args:
        ring_coords (Sequence[Sequence[float]]): Cartesian coordinates of the ring atoms, in
            ring-connectivity order.

    Returns:
        float: The total puckering amplitude Q (Angstrom), >= 0, 0 for a planar ring.
    """
    validate_ring_order(ring_coords)
    z, _ = _ring_z_displacements(ring_coords)
    return float(np.sqrt(np.sum(z ** 2)))


def cremer_pople_params(ring_coords: Sequence[Sequence[float]]) -> CremerPopleParams:
    """Compute the full set of Cremer-Pople puckering coordinates of a ring.

    Important: ``ring_coords`` must be supplied in ring-connectivity order (consecutive entries
    bonded, last bonded back to first). Establishing and verifying that order is the caller's
    responsibility, not this module's.

    Args:
        ring_coords (Sequence[Sequence[float]]): Cartesian coordinates of the ring atoms, in
            ring-connectivity order.

    Returns:
        CremerPopleParams: The Cremer-Pople puckering coordinates of the ring.
    """
    validate_ring_order(ring_coords)
    z, n = _ring_z_displacements(ring_coords)
    idx = np.arange(n)
    max_m = (n - 1) // 2
    terms: List[Tuple[int, float, Optional[float]]] = []
    q2 = phi2_deg = None
    for m in range(2, max_m + 1):
        angles = 2.0 * np.pi * m * idx / n
        c = np.sqrt(2.0 / n) * np.sum(z * np.cos(angles))
        s = -np.sqrt(2.0 / n) * np.sum(z * np.sin(angles))
        q_m = float(np.sqrt(c ** 2 + s ** 2))
        phi_m_deg = float(np.degrees(np.arctan2(s, c)) % 360.0)
        terms.append((m, q_m, phi_m_deg))
        if m == 2:
            q2, phi2_deg = q_m, phi_m_deg

    q_half = None
    if n % 2 == 0:
        signs = (-1.0) ** idx
        q_half = float(np.sum(signs * z) / np.sqrt(n))
        terms.append((n // 2, q_half, None))

    amplitude = puckering_amplitude(ring_coords)

    q3 = theta_deg = phi_deg = None
    if n == 6:
        q3 = q_half
        theta_deg = float(np.degrees(np.arctan2(q2, q3)))
        phi_deg = phi2_deg

    return CremerPopleParams(
        ring_size=n,
        amplitude=amplitude,
        terms=terms,
        q2=q2,
        phi2_deg=phi2_deg,
        q3=q3,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
    )


def _half_open_bin_index(value_deg: float, bin_width_deg: float, num_bins: int) -> int:
    """Map an angle into a half-open bin index, with a small tolerance for boundary values.

    Bin k covers the half-open interval [k * bin_width - bin_width / 2, k * bin_width +
    bin_width / 2), wrapping around the full circle. A small numerical tolerance is applied so
    that a value that falls exactly on a bin boundary (subject to floating-point noise) bins
    deterministically, instead of via round-half-to-even.

    Args:
        value_deg (float): The angle in degrees.
        bin_width_deg (float): The width of each bin in degrees.
        num_bins (int): The number of bins spanning the full circle.

    Returns:
        int: The bin index in [0, num_bins).
    """
    shifted = value_deg + bin_width_deg / 2.0 - _BIN_TOL_DEG
    return int(np.floor(shifted / bin_width_deg)) % num_bins


def classify_pucker(ring_coords: Sequence[Sequence[float]]) -> str:
    """Classify a ring's Cremer-Pople puckering coordinates into a canonical pucker state.

    Args:
        ring_coords (Sequence[Sequence[float]]): Cartesian coordinates of the ring atoms, in
            ring-connectivity order.

    Returns:
        str: For a 6-ring: 'chair', 'boat', 'twist-boat', or 'half-chair' (theta outside the
            chair and boat/twist-boat windows). For a 5-ring: 'envelope' or 'twist'. For any
            ring size, 'planar' if the total puckering amplitude Q is below
            PLANARITY_AMPLITUDE_THRESHOLD, regardless of ring size.
    """
    n = len(ring_coords)
    params = cremer_pople_params(ring_coords)

    if params.amplitude < PLANARITY_AMPLITUDE_THRESHOLD:
        return 'planar'

    if n == 6:
        if params.theta_deg <= 30.0 or params.theta_deg >= 150.0:
            return 'chair'
        if abs(params.theta_deg - 90.0) <= 30.0:
            phi = params.phi_deg % 360.0
            k = _half_open_bin_index(phi, 30.0, 12)
            return 'boat' if k % 2 == 0 else 'twist-boat'
        return 'half-chair'

    if n == 5:
        phi2 = params.phi2_deg % 360.0
        k = _half_open_bin_index(phi2, 18.0, 20)
        return 'envelope' if k % 2 == 0 else 'twist'

    raise RingPuckerError(f'classify_pucker only supports 5- and 6-membered rings, got a {n}-membered ring.')


def canonical_pucker_states(ring_size: int) -> List[str]:
    """Enumerate the discrete canonical pucker states of a given ring size.

    Args:
        ring_size (int): The number of atoms in the ring, 5 or 6.

    Returns:
        List[str]: The canonical pucker state labels.
    """
    if ring_size == 6:
        return ['chair', 'boat', 'twist-boat']
    if ring_size == 5:
        return ['envelope', 'twist']
    raise RingPuckerError(f'canonical_pucker_states only supports 5- and 6-membered rings, got ring_size={ring_size}.')


#: Default radius (Angstrom) used to place atoms on the regular polygon in ``ideal_pucker_geometry``.
DEFAULT_RING_RADIUS = 1.4

#: Default total Cremer-Pople puckering amplitude (Angstrom) per canonical pucker label.
DEFAULT_PUCKER_AMPLITUDES = {
    'chair': 0.63,
    'boat': 0.65,
    'twist-boat': 0.65,
    'envelope': 0.45,
    'twist': 0.45,
}


def ideal_pucker_geometry(ring_size: int, label: str, amplitude: Optional[float] = None) -> np.ndarray:
    """Synthesize an idealized ring geometry for a canonical Cremer-Pople pucker state.

    The geometry is generated by placing ring_size atoms on a regular planar polygon and then
    displacing them out of plane via the inverse Cremer-Pople transform, using (q_m, phi_m)
    values chosen to sit at the canonical geometric point of the requested pucker label.

    Args:
        ring_size (int): The number of atoms in the ring, 5 or 6.
        label (str): A canonical pucker label, one of the strings returned by
            ``canonical_pucker_states(ring_size)``.
        amplitude (Optional[float]): The total puckering amplitude Q (Angstrom) to use. If None,
            a sensible default is used per label.

    Returns:
        np.ndarray: An (N x 3) array of idealized ring atom coordinates.
    """
    valid_labels = canonical_pucker_states(ring_size)
    if label not in valid_labels:
        raise RingPuckerError(f"Unknown pucker label '{label}' for ring_size={ring_size}; expected one of {valid_labels}.")

    q_amplitude = amplitude if amplitude is not None else DEFAULT_PUCKER_AMPLITUDES[label]

    q2 = phi2_deg = 0.0
    q_half = 0.0
    if ring_size == 6:
        if label == 'chair':
            q2, phi2_deg, q_half = 0.0, 0.0, q_amplitude
        elif label == 'boat':
            q2, phi2_deg, q_half = q_amplitude, 0.0, 0.0
        elif label == 'twist-boat':
            q2, phi2_deg, q_half = q_amplitude, 30.0, 0.0
    else:  # ring_size == 5
        if label == 'envelope':
            q2, phi2_deg = q_amplitude, 0.0
        elif label == 'twist':
            q2, phi2_deg = q_amplitude, 18.0

    idx = np.arange(ring_size)
    xy_angles = 2.0 * np.pi * idx / ring_size
    x = DEFAULT_RING_RADIUS * np.cos(xy_angles)
    y = DEFAULT_RING_RADIUS * np.sin(xy_angles)

    phi2_rad = np.radians(phi2_deg)
    z = np.sqrt(2.0 / ring_size) * q2 * np.cos(phi2_rad + 2.0 * np.pi * 2 * idx / ring_size)
    if ring_size % 2 == 0:
        z = z + (q_half / np.sqrt(ring_size)) * ((-1.0) ** idx)

    return np.column_stack([x, y, z])


def pucker_state_id(ring_coords: Sequence[Sequence[float]]) -> str:
    """Compute a fine-grained (phase-resolved) pucker state identifier.

    This is a TERTIARY diagnostic: unlike ``classify_pucker``, which collapses conformers into
    a handful of coarse Cremer-Pople labels, this function additionally bins the puckering
    phase angle so that, e.g., the ~6 distinct boats or ~6 distinct twist-boats of a 6-ring are
    distinguished from one another. Planar and chair/half-chair conformers, for which the phase
    angle is meaningless or degenerate, are returned unbinned.

    Args:
        ring_coords (Sequence[Sequence[float]]): Cartesian coordinates of the ring atoms, in
                                                  ring-connectivity order.

    Returns:
        str: A pucker state id, either a bare coarse label ('planar', 'chair', 'half-chair') or
             a phase-binned label of the form ``f'{label}@{bin_deg}'``.

    Note:
        The phase bin is computed from ``ring_coords`` in the given ring-atom order and is NOT
        canonicalized under ring automorphisms: e.g. ``'twist-boat@30'`` and ``'twist-boat@210'``
        may in fact be symmetry-equivalent states for a symmetric ring, but this function will
        report them as distinct ids. This is a diagnostic label, not a canonical conformer
        identity.
    """
    label = classify_pucker(ring_coords)
    if label in ('planar', 'chair', 'half-chair'):
        return label
    params = cremer_pople_params(ring_coords)
    n = len(ring_coords)
    if n == 6:
        phi = params.phi_deg % 360.0
        bin_width_deg = 30.0
        num_bins = 12
    else:
        phi = params.phi2_deg % 360.0
        bin_width_deg = 18.0
        num_bins = 20
    k = _half_open_bin_index(phi, bin_width_deg, num_bins)
    bin_deg = int(round(k * bin_width_deg))
    return f'{label}@{bin_deg}'


def pucker_label_counts(ring_coords_iterable) -> collections.Counter:
    """Count the coarse Cremer-Pople pucker labels of an iterable of ring geometries.

    This is a SECONDARY diagnostic that annotates a conformer set with its distribution of
    coarse pucker labels; it is not a conformer-identity or coverage metric on its own.

    Args:
        ring_coords_iterable: An iterable of ring-coordinate arrays, each in ring-connectivity
                              order.

    Returns:
        collections.Counter: A counter of ``classify_pucker(...)`` results.
    """
    return collections.Counter(classify_pucker(ring_coords) for ring_coords in ring_coords_iterable)
