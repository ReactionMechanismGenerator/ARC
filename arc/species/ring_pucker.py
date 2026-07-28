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
