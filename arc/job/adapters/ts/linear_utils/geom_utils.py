"""
Shared low-level geometry and graph helpers used by the linear TS-guess
builder modules (``isomerization``, ``postprocess``, ``addition``,
``families``).

All functions are pure (no side effects beyond in-place coordinate mutation
where documented) and free of closure captures, so they can be imported
anywhere without circular-dependency issues.

Graph/fragment utilities:

* :func:`mol_to_adjacency`: build an ``{int: Set[int]}`` adjacency dict from an RMG Molecule.
* :func:`bfs_fragment`: BFS from a start atom, blocking specified atoms, returning the reachable fragment.
* :func:`split_mol_at_bonds`: split a molecule into connected components after removing specified bonds.
"""

from collections import deque
from typing import Dict, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple

import numpy as np

from arc.common import get_logger

if TYPE_CHECKING:
    from rmgpy.molecule.molecule import Molecule


logger = get_logger()


# ---------------------------------------------------------------------------
# Graph / fragment utilities
# ---------------------------------------------------------------------------


def mol_to_adjacency(mol: 'Molecule') -> Dict[int, Set[int]]:
    """Build an ``{atom_index: {neighbor_indices}}`` adjacency dict from
    the molecular graph.

    This is the canonical way to obtain connectivity for BFS/fragment
    operations — it uses the bond topology, NOT distance thresholds.

    Args:
        mol: RMG Molecule.

    Returns:
        Adjacency dict keyed by atom index.
    """
    atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
    adj: Dict[int, Set[int]] = {i: set() for i in range(len(mol.atoms))}
    for atom in mol.atoms:
        ia = atom_to_idx[atom]
        for nbr in atom.bonds:
            ib = atom_to_idx[nbr]
            adj[ia].add(ib)
    return adj


def bfs_fragment(adj: Dict[int, Set[int]],
                 start: int,
                 block: Optional[Set[int]] = None,
                 ) -> Set[int]:
    """Return the set of atom indices reachable from ``start`` via BFS,
    not crossing any atom in ``block``.

    Args:
        adj: Adjacency dict (e.g. from :func:`mol_to_adjacency`).
        start: Starting atom index.
        block: Atom indices that act as barriers (not traversed,
            not included in the returned set).

    Returns:
        Set of reachable atom indices (always includes ``start``
        unless it is in ``block``).
    """
    if block is None:
        block = set()
    visited: Set[int] = set()
    queue: deque = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited or node in block:
            continue
        visited.add(node)
        for nbr in adj.get(node, ()):
            if nbr not in visited and nbr not in block:
                queue.append(nbr)
    return visited


def split_mol_at_bonds(mol: 'Molecule',
                       cut_bonds: Sequence[Tuple[int, int]],
                       ) -> List[Set[int]]:
    """Split a molecule into connected components after removing
    specified bonds.

    Args:
        mol: RMG Molecule.
        cut_bonds: Bonds to remove (pairs of atom indices).

    Returns:
        List of sets, each containing atom indices in one fragment.
    """
    adj = mol_to_adjacency(mol)
    # Remove cut bonds from adjacency.
    for a, b in cut_bonds:
        adj.get(a, set()).discard(b)
        adj.get(b, set()).discard(a)
    # Find connected components via BFS.
    visited: Set[int] = set()
    components: List[Set[int]] = []
    for start in range(len(mol.atoms)):
        if start in visited:
            continue
        comp = bfs_fragment(adj, start)
        visited |= comp
        components.append(comp)
    return components


def bfs_path(adj: List[List[int]], src: int, dst: int) -> Optional[List[int]]:
    """Return the shortest path from *src* to *dst* using the adjacency list *adj*.

    Args:
        adj: Atom adjacency list (``adj[i]`` lists neighbors of atom *i*).
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
        logger.debug('dihedral_deg: degenerate geometry (p1-p2-p3 or p2-p3-p4 collinear); returning 0.0.')
        return 0.0
    n1 /= n1_n; n2 /= n2_n
    cos_d = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    angle = np.degrees(np.arccos(cos_d))
    if np.dot(np.cross(n1, n2), b2) < 0:
        angle = -angle
    return angle
