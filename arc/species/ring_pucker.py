#!/usr/bin/env python3
# encoding: utf-8

"""
Ring puckering conformer generation for ARC.

Provides Cremer-Pople puckering coordinates and canonical pucker-state
enumeration for 5- and 6-membered rings (and fused/bridged bicyclic systems).
These are used to both seed ring-conformer sampling and to score ring-pucker
coverage of a conformer ensemble in arc.species.conformers.
"""

from typing import Sequence

import numpy as np


class RingPuckerError(Exception):
    """Raised when ring puckering analysis or generation fails."""


def puckering_amplitude(ring_coords: Sequence[Sequence[float]]) -> float:
    """Compute the total Cremer-Pople puckering amplitude of a single ring.

    The total puckering amplitude Q is a non-negative scalar measuring how far
    the ring departs from planarity; it is exactly zero for a perfectly planar
    ring.

    Args:
        ring_coords: Ordered Cartesian coordinates (N x 3) of the ring atoms,
            given in ring-connectivity order.

    Returns:
        The total Cremer-Pople puckering amplitude Q, in the same length units
        as the input coordinates.
    """
    raise NotImplementedError('puckering_amplitude is not yet implemented')
