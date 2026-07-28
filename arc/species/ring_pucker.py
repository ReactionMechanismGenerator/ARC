#!/usr/bin/env python3
# encoding: utf-8

"""
Ring puckering conformer generation for ARC.

Provides Cremer-Pople puckering coordinates and canonical pucker-state
enumeration for 5- and 6-membered rings (and fused/bridged bicyclic systems).
These are used to both seed ring-conformer sampling and to score ring-pucker
coverage of a conformer ensemble in arc.species.conformers.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


class RingPuckerError(Exception):
    """Raised when ring puckering analysis or generation fails."""


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


def _ring_z_displacements(ring_coords: Sequence[Sequence[float]]) -> Tuple[np.ndarray, int]:
    """Compute the Cremer-Pople out-of-plane displacements z_j of a ring.

    Args:
        ring_coords (Sequence[Sequence[float]]): Cartesian coordinates of the ring atoms, in
            ring-connectivity order.

    Returns:
        Tuple[np.ndarray, int]: The z_j displacements (length N array) and the ring size N.
    """
    coords = np.asarray(ring_coords, dtype=float)
    n = coords.shape[0]
    r0 = np.mean(coords, axis=0)
    shifted = coords - r0
    idx = np.arange(n)
    angles = 2.0 * np.pi * idx / n
    r1 = np.sum(shifted * np.sin(angles)[:, None], axis=0)
    r2 = np.sum(shifted * np.cos(angles)[:, None], axis=0)
    normal = np.cross(r1, r2)
    normal = normal / np.linalg.norm(normal)
    z = shifted @ normal
    return z, n


def puckering_amplitude(ring_coords: Sequence[Sequence[float]]) -> float:
    """Compute the total Cremer-Pople puckering amplitude of a single ring.

    Args:
        ring_coords (Sequence[Sequence[float]]): Cartesian coordinates of the ring atoms, in
            ring-connectivity order.

    Returns:
        float: The total puckering amplitude Q (Angstrom), >= 0, 0 for a planar ring.
    """
    z, _ = _ring_z_displacements(ring_coords)
    return float(np.sqrt(np.sum(z ** 2)))


def cremer_pople_params(ring_coords: Sequence[Sequence[float]]) -> CremerPopleParams:
    """Compute the full set of Cremer-Pople puckering coordinates of a ring.

    Args:
        ring_coords (Sequence[Sequence[float]]): Cartesian coordinates of the ring atoms, in
            ring-connectivity order.

    Returns:
        CremerPopleParams: The Cremer-Pople puckering coordinates of the ring.
    """
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


def classify_pucker(ring_coords: Sequence[Sequence[float]]) -> str:
    """Classify a ring's Cremer-Pople puckering coordinates into a canonical pucker state.

    Args:
        ring_coords (Sequence[Sequence[float]]): Cartesian coordinates of the ring atoms, in
            ring-connectivity order.

    Returns:
        str: 'chair', 'boat', or 'twist-boat' for a 6-ring; 'envelope' or 'twist' for a 5-ring.
    """
    n = len(ring_coords)
    params = cremer_pople_params(ring_coords)

    if n == 6:
        if params.theta_deg <= 45.0 or params.theta_deg >= 135.0:
            return 'chair'
        phi = params.phi_deg % 360.0
        k = int(round(phi / 30.0)) % 12
        return 'boat' if k % 2 == 0 else 'twist-boat'

    if n == 5:
        phi2 = params.phi2_deg % 360.0
        k = int(round(phi2 / 18.0)) % 20
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
