"""
Shared low-level geometry helpers used by the linear TS-guess builder
modules (``isomerization``, ``postprocess``, ``addition``).

All functions are pure (no side effects beyond in-place coordinate mutation
where documented) and free of closure captures, so they can be imported
anywhere without circular-dependency issues.
"""

from collections import deque
from typing import Dict, List, Optional, Set

import numpy as np


def bfs_path(adj: List[List[int]], src: int, dst: int) -> Optional[List[int]]:
    """Return the shortest path from *src* to *dst* using the adjacency list *adj*.

    Args:
        adj: Atom adjacency list (``adj[i]`` lists neighbours of atom *i*).
        src: Source atom index.
        dst: Destination atom index.

    Returns:
        List of atom indices from *src* to *dst*, or ``None`` if unreachable.
    """
    if src == dst:
        return [src]
    prev: Dict[int, Optional[int]] = {src: None}
    bfs_q: deque = deque([src])
    while bfs_q:
        node = bfs_q.popleft()
        for nbr in adj[node]:
            if nbr not in prev:
                prev[nbr] = node
                if nbr == dst:
                    path: List[int] = []
                    cur: Optional[int] = dst
                    while cur is not None:
                        path.append(cur)
                        cur = prev[cur]
                    return path[::-1]
                bfs_q.append(nbr)
    return None


def downstream(adj: List[List[int]], cut_a: int, cut_b: int) -> Set[int]:
    """Return all atom indices reachable from *cut_b* without crossing *cut_a*.

    This identifies the fragment on the *cut_b* side of the *cut_a*-*cut_b*
    bond, useful for rotating a molecular fragment around a bond axis.

    Args:
        adj: Atom adjacency list.
        cut_a: Atom index on the "fixed" side of the cut.
        cut_b: Atom index on the "moving" side of the cut.

    Returns:
        Set of atom indices reachable from *cut_b* (including *cut_b* itself).
    """
    visited = {cut_a}
    stack = [cut_b]
    result: Set[int] = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        result.add(node)
        for nbr in adj[node]:
            if nbr not in visited:
                stack.append(nbr)
    return result


def rotate_atoms(c: np.ndarray, origin: np.ndarray, axis: np.ndarray,
                 indices: Set[int], angle: float) -> None:
    """Rotate atoms at *indices* around *origin*+*axis* by *angle* radians, in-place.

    Uses Rodrigues' rotation formula.

    Args:
        c: Coordinate array (N x 3), modified in-place.
        origin: Point on the rotation axis.
        axis: Unit vector along the rotation axis.
        indices: Set of atom indices to rotate.
        angle: Rotation angle in radians.
    """
    cos_a = np.cos(angle); sin_a = np.sin(angle)
    for di in indices:
        v = c[di] - origin
        c[di] = (origin
                 + v * cos_a
                 + np.cross(axis, v) * sin_a
                 + axis * np.dot(axis, v) * (1.0 - cos_a))


def dihedral_deg(p1: np.ndarray, p2: np.ndarray,
                 p3: np.ndarray, p4: np.ndarray) -> float:
    """Dihedral angle in degrees for points p1-p2-p3-p4 (IUPAC sign convention).

    Args:
        p1: Position of atom 1.
        p2: Position of atom 2.
        p3: Position of atom 3.
        p4: Position of atom 4.

    Returns:
        Dihedral angle in degrees, range [-180, 180].
    """
    b1 = p2 - p1; b2 = p3 - p2; b3 = p4 - p3
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    n1_n = np.linalg.norm(n1); n2_n = np.linalg.norm(n2)
    if n1_n < 1e-8 or n2_n < 1e-8:
        return 0.0
    n1 /= n1_n; n2 /= n2_n
    cos_d = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    angle = np.degrees(np.arccos(cos_d))
    if np.dot(np.cross(n1, n2), b2) < 0:
        angle = -angle
    return angle
