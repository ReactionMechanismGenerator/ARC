"""
Shared low-level geometry and graph helpers used by the linear TS-guess
builder modules (``isomerization``, ``postprocess``, ``addition``, ``families``).

All functions are pure (no side effects beyond in-place coordinate mutation
where documented) and free of closure captures, so they can be imported
anywhere without circular-dependency issues.

Graph/fragment utilities:

* :func:`mol_to_adjacency`: build an ``{int: set[int]}`` adjacency dict from an RMG Molecule.
* :func:`bfs_fragment`: BFS from a start atom, blocking specified atoms, returning the reachable fragment.
* :func:`split_mol_at_bonds`: split a molecule into connected components after removing specified bonds.
"""

from collections import deque
from typing import TYPE_CHECKING
from collections.abc import Sequence

import numpy as np

from arc.common import get_logger

if TYPE_CHECKING:
    from arc.molecule.molecule import Molecule


logger = get_logger()


# ---------------------------------------------------------------------------
# Graph / fragment utilities
# ---------------------------------------------------------------------------

def canonical_bond(a: int, b: int) -> tuple[int, int]:
    """Return ``(min(a, b), max(a, b))`` — the canonical key for an undirected bond."""
    return (a, b) if a <= b else (b, a)


def atom_index_map(mol: Molecule) -> dict:
    """Return a mapping from each ``mol.atoms`` element to its integer index."""
    return {atom: idx for idx, atom in enumerate(mol.atoms)}


def heavy_neighbors_of(mol: Molecule,
                       atom_idx: int,
                       symbols: Sequence[str],
                       ) -> list[int]:
    """Return graph-bonded non-H neighbor indices of ``atom_idx``."""
    a2i = atom_index_map(mol)
    return [a2i[nbr] for nbr in mol.atoms[atom_idx].bonds
            if symbols[a2i[nbr]] != 'H']


def h_neighbors_of(mol: Molecule,
                   atom_idx: int,
                   symbols: Sequence[str],
                   ) -> list[int]:
    """Return graph-bonded H neighbor indices of ``atom_idx``."""
    a2i = atom_index_map(mol)
    return [a2i[nbr] for nbr in mol.atoms[atom_idx].bonds
            if symbols[a2i[nbr]] == 'H']


def xyz_with_coords(xyz: dict, coords: np.ndarray) -> dict:
    """
    Return a new XYZ dict copying ``xyz`` with ``coords`` replaced.

    All keys of ``xyz`` are preserved (e.g. ``symbols``, ``isotopes``,
    plus any extra keys). The ``coords`` entry is replaced with a
    tuple-of-tuples of floats. If ``isotopes`` is absent it is filled in
    with all-zeros of length ``len(symbols)``.
    """
    out = dict(xyz)
    if 'isotopes' not in out:
        out['isotopes'] = tuple(0 for _ in range(len(xyz['symbols'])))
    out['coords'] = tuple(tuple(float(x) for x in row) for row in coords)
    return out


def mol_to_adjacency(mol: Molecule) -> dict[int, set[int]]:
    """
    Build an ``{atom_index: {neighbor_indices}}`` adjacency dict from the molecular graph.

    This is the canonical way to obtain connectivity for BFS/fragment
    operations — it uses the bond topology, NOT distance thresholds.

    Args:
        mol: RMG Molecule.

    Returns:
        Adjacency dict keyed by atom index.
    """
    a2i = atom_index_map(mol)
    adj: dict[int, set[int]] = {i: set() for i in range(len(mol.atoms))}
    for atom in mol.atoms:
        ia = a2i[atom]
        for nbr in atom.bonds:
            adj[ia].add(a2i[nbr])
    return adj


def bfs_fragment(adj: dict[int, set[int]],
                 start: int,
                 block: set[int] | None = None,
                 ) -> set[int]:
    """
    Return the set of atom indices reachable from ``start`` via BFS, not crossing any atom in ``block``.

    Args:
        adj: Adjacency dict (e.g. from :func:`mol_to_adjacency`).
        start: Starting atom index.
        block: Atom indices that act as barriers (not traversed,
            not included in the returned set).

    Returns:
        Set of reachable atom indices (always includes ``start`` unless it is in ``block``).
    """
    if block is None:
        block = set()
    visited: set[int] = set()
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


def split_mol_at_bonds(mol: Molecule,
                       cut_bonds: Sequence[tuple[int, int]],
                       ) -> list[set[int]]:
    """
    Split a molecule into connected components after removing specified bonds.

    Args:
        mol: RMG Molecule.
        cut_bonds: Bonds to remove (pairs of atom indices).

    Returns:
        List of sets, each containing atom indices in one fragment.
    """
    adj = mol_to_adjacency(mol)
    for a, b in cut_bonds:
        adj.get(a, set()).discard(b)
        adj.get(b, set()).discard(a)
    visited: set[int] = set()
    components: list[set[int]] = []
    for start in range(len(mol.atoms)):
        if start in visited:
            continue
        comp = bfs_fragment(adj, start)
        visited |= comp
        components.append(comp)
    return components


def bfs_path(adj: list[list[int]], src: int, dst: int) -> list[int] | None:
    """
    Return the shortest path from *src* to *dst* using the adjacency list *adj*.

    Args:
        adj: Atom adjacency list (``adj[i]`` lists neighbors of atom *i*).
        src: Source atom index.
        dst: Destination atom index.

    Returns:
        List of atom indices from *src* to *dst*, or ``None`` if unreachable.
    """
    if src == dst:
        return [src]
    prev: dict[int, int | None] = {src: None}
    bfs_q: deque = deque([src])
    while bfs_q:
        node = bfs_q.popleft()
        for nbr in adj[node]:
            if nbr not in prev:
                prev[nbr] = node
                if nbr == dst:
                    path: list[int] = []
                    cur: int | None = dst
                    while cur is not None:
                        path.append(cur)
                        cur = prev[cur]
                    return path[::-1]
                bfs_q.append(nbr)
    return None


def downstream(adj: list[list[int]], cut_a: int, cut_b: int) -> set[int]:
    """
    Return all atom indices reachable from *cut_b* without crossing *cut_a*.

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
    result: set[int] = set()
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
                 indices: set[int], angle: float) -> None:
    """
    Rotate atoms at *indices* around *origin*+*axis* by *angle* radians, in-place.

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
    """
    Dihedral angle in degrees for points p1-p2-p3-p4 (IUPAC sign convention).

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


def two_sphere_intersection(center1: np.ndarray,
                            r1: float,
                            center2: np.ndarray,
                            r2: float,
                            ref_pos: np.ndarray,
                            fallback_perp: np.ndarray | None = None,
                            ) -> np.ndarray | None:
    """
    Place a point at the intersection of two spheres around two centers.

    The two spheres' surfaces intersect iff
    ``|r1 - r2| <= d <= r1 + r2`` where ``d`` is the inter-center distance.
    The returned point lies on the same side of the inter-center axis as
    ``ref_pos``.

    When the surfaces do not intersect — either because the spheres are
    disjoint (``d > r1 + r2``) or because one sphere is fully inside the
    other (``d < |r1 - r2|``) — the function falls back to a collinear
    placement at distance ``r1`` from ``center1`` along the inter-center
    axis. The ``r2`` constraint is then not satisfied; this is the best
    approximation when no exact solution exists.

    Args:
        center1 (np.ndarray): The first sphere center, shape ``(3,)``.
        r1 (float): Radius of the first sphere.
        center2 (np.ndarray): The second sphere center, shape ``(3,)``.
        r2 (float): Radius of the second sphere.
        ref_pos (np.ndarray): Reference point used to pick the side of the
            inter-center axis.
        fallback_perp (np.ndarray, optional): Perpendicular direction to
            fall back on when ``ref_pos`` is collinear with the axis.
            Need not be unit-length or exactly perpendicular — the
            function projects it onto the plane perpendicular to the
            axis. When ``None`` (default) or also collinear with the axis,
            an arbitrary perpendicular is constructed.

    Returns:
        np.ndarray | None: The placed point in 3-space, or ``None`` if
            the inter-center distance is degenerate (< 1e-6).
    """
    axis = center2 - center1
    axis_d = float(np.linalg.norm(axis))
    if axis_d < 1e-6:
        return None
    axis_h = axis / axis_d
    if abs(r1 - r2) <= axis_d <= r1 + r2:
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
        if fallback_perp is None:
            p_hat = np.zeros(3)
        else:
            p_hat = fallback_perp.copy()
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
