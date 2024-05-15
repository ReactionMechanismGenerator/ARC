"""
Isomerization-specific 3D geometry builders: near-attack conformations,
ring-closure algorithm, 4-center interchange, and Z-matrix branch generation,
extracted from ``arc.job.adapters.ts.linear``.
"""

import copy
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from arc.common import get_logger, get_single_bond_length
from arc.species.converter import zmat_to_xyz
from arc.species.zmat import update_zmat_by_xyz

from arc.job.adapters.ts.linear_utils.geom_utils import (
    bfs_path,
    downstream,
    rotate_atoms,
    dihedral_deg,
    mol_to_adjacency,
)
from arc.job.adapters.ts.linear_utils.math_zmat import average_zmat_params
from arc.job.adapters.ts.linear_utils.postprocess import (
    PAULING_DELTA,
    build_zmat_with_retry,
    postprocess_ts_guess,
    validate_ts_guess,
    fix_crowded_h_atoms,
)

if TYPE_CHECKING:
    from arc.molecule import Molecule


logger = get_logger()

RING_CLOSURE_THRESHOLD: float = 4.5  # Angstroms; forming-bond distance above which ring-closure algorithm is used

# Per-position backbone dihedral magnitudes (degrees) for cyclic TS of a given ring size.
# Each list has (N-3) entries, one per rotatable interior bond (outer-to-inner order).
# Sum rule: sum of entries ≈ (N-4) × 95°.
# Values for N=6 are derived from CCCOO TS data: HCCC=50°, CCCO=60°, CCOO=80° (sum=190=2×95).
_TS_RING_DIHEDRALS: dict[int, list[float]] = {
    4: [0.0],                            # 1 interior bond; nearly planar (highly strained); sum=0=0×95
    5: [45.0, 50.0],                     # 2 interior bonds; puckered envelope; sum=95=1×95
    6: [50.0, 60.0, 80.0],               # 3 interior bonds; chair-like; sum=190=2×95
    7: [55.0, 65.0, 75.0, 90.0],         # 4 interior bonds; twist-chair; sum=285=3×95
    8: [55.0, 65.0, 75.0, 85.0, 100.0],  # 5 interior bonds; crown/boat-chair; sum=380=4×95
}
_TS_RING_DIHEDRAL_DEFAULT: float = 60.0


def get_path_length(mol: Molecule, src: int, dst: int) -> int | None:
    """Return the shortest-path length (number of bonds) between *src* and *dst*, or None if disconnected."""
    if src == dst:
        return 0
    atom_to_idx = {atom: idx for idx, atom in enumerate(mol.atoms)}
    queue = deque([(src, 0)])
    visited = {src}
    while queue:
        current, depth = queue.popleft()
        for nbr in mol.atoms[current].bonds:
            nbr_idx = atom_to_idx[nbr]
            if nbr_idx == dst:
                return depth + 1
            if nbr_idx not in visited:
                visited.add(nbr_idx)
                queue.append((nbr_idx, depth + 1))
    return None


# ---------------------------------------------------------------------------
# Backbone atom mapping (fallback when RMG template matching fails)
# ---------------------------------------------------------------------------

def backbone_atom_map(r_mol: Molecule,
                      p_mol: Molecule,
                      ) -> list[int] | None:
    """
    Find an atom map between R and P by matching heavy-atom backbones.

    For isomerization reactions the heavy-atom connectivity is usually
    preserved (only bond orders change). This function builds undirected
    heavy-atom-only graphs, matches them via VF2 graph isomorphism, then
    assigns hydrogen atoms by their bonded heavy atom. Migrating H atoms
    (bonded to different heavy atoms in R vs P) are left over and paired
    by element.

    Args:
        r_mol: Reactant RMG Molecule.
        p_mol: Product RMG Molecule.

    Returns:
        atom_map or ``None`` if the backbones are not isomorphic.
        ``atom_map[r_index] = p_index``.
    """
    n_r, n_p = len(r_mol.atoms), len(p_mol.atoms)
    if n_r != n_p:
        return None

    # --- 1. Build heavy-atom-only undirected graphs
    r_heavy = {i for i, a in enumerate(r_mol.atoms) if a.symbol != 'H'}
    p_heavy = {i for i, a in enumerate(p_mol.atoms) if a.symbol != 'H'}
    if len(r_heavy) != len(p_heavy):
        return None

    r_graph = nx.Graph()
    p_graph = nx.Graph()
    for i in r_heavy:
        r_graph.add_node(i, elem=r_mol.atoms[i].symbol)
    for i in p_heavy:
        p_graph.add_node(i, elem=p_mol.atoms[i].symbol)

    for bond in r_mol.get_all_edges():
        i, j = r_mol.atoms.index(bond.atom1), r_mol.atoms.index(bond.atom2)
        if i in r_heavy and j in r_heavy:
            r_graph.add_edge(i, j)
    for bond in p_mol.get_all_edges():
        i, j = p_mol.atoms.index(bond.atom1), p_mol.atoms.index(bond.atom2)
        if i in p_heavy and j in p_heavy:
            p_graph.add_edge(i, j)

    # --- 2. Find heavy-atom isomorphism (ignoring bond orders)
    gm = nx.isomorphism.GraphMatcher(
        r_graph, p_graph,
        node_match=lambda n1, n2: n1['elem'] == n2['elem'])
    if not gm.is_isomorphic():
        found = False
        # Ring-forming: P has one extra edge; try removing each.
        if p_graph.number_of_edges() == r_graph.number_of_edges() + 1:
            for u, v in list(p_graph.edges()):
                p_trial = p_graph.copy()
                p_trial.remove_edge(u, v)
                gm2 = nx.isomorphism.GraphMatcher(
                    r_graph, p_trial,
                    node_match=lambda n1, n2: n1['elem'] == n2['elem'])
                if gm2.is_isomorphic():
                    gm = gm2
                    found = True
                    break
        # Atom-migration (e.g. halogen): equal edge count, one breaks and one forms.
        elif p_graph.number_of_edges() == r_graph.number_of_edges():
            for ru, rv in list(r_graph.edges()):
                r_trial = r_graph.copy()
                r_trial.remove_edge(ru, rv)
                for pu, pv in list(p_graph.edges()):
                    p_trial = p_graph.copy()
                    p_trial.remove_edge(pu, pv)
                    gm2 = nx.isomorphism.GraphMatcher(
                        r_trial, p_trial,
                        node_match=lambda n1, n2: n1['elem'] == n2['elem'])
                    if gm2.is_isomorphic():
                        gm = gm2
                        found = True
                        break
                if found:
                    break
        if not found:
            return None
    heavy_map: dict[int, int] = gm.mapping  # {r_idx: p_idx}

    # --- 3. Assign H atoms via bonded heavy atom
    atom_map: list[int | None] = [None] * n_r
    for r_idx, p_idx in heavy_map.items():
        atom_map[r_idx] = p_idx

    p_h_used: set[int] = set()
    for r_heavy_idx, p_heavy_idx in heavy_map.items():
        r_h = [i for i in range(n_r)
               if r_mol.atoms[i].symbol == 'H'
               and r_mol.has_bond(r_mol.atoms[r_heavy_idx], r_mol.atoms[i])]
        p_h = [j for j in range(n_p)
               if p_mol.atoms[j].symbol == 'H'
               and p_mol.has_bond(p_mol.atoms[p_heavy_idx], p_mol.atoms[j])
               and j not in p_h_used]
        for rh, ph in zip(r_h, p_h):
            atom_map[rh] = ph
            p_h_used.add(ph)

    # --- 4. Pair any remaining unmapped atoms (migrating H's)
    mapped_p = {v for v in atom_map if v is not None}
    unmapped_r = [i for i in range(n_r) if atom_map[i] is None]
    unmapped_p = [j for j in range(n_p) if j not in mapped_p]
    if len(unmapped_r) != len(unmapped_p):
        return None
    ur_by_elem: dict[str, list[int]] = defaultdict(list)
    up_by_elem: dict[str, list[int]] = defaultdict(list)
    for i in unmapped_r:
        ur_by_elem[r_mol.atoms[i].symbol].append(i)
    for j in unmapped_p:
        up_by_elem[p_mol.atoms[j].symbol].append(j)
    if set(ur_by_elem.keys()) != set(up_by_elem.keys()):
        return None
    for elem in ur_by_elem:
        if len(ur_by_elem[elem]) != len(up_by_elem[elem]):
            return None
        for ri, pj in zip(ur_by_elem[elem], up_by_elem[elem]):
            atom_map[ri] = pj

    if any(v is None for v in atom_map):
        return None
    return atom_map


# ---------------------------------------------------------------------------
# Cumulated-bond (allene / cumulene) detection
# ---------------------------------------------------------------------------

def path_has_cumulated_bonds(mol: Molecule,
                             forming_bond: tuple[int, int],
                             ) -> bool:
    """
    Check whether the BFS path between forming-bond atoms passes through
    a cumulated double-bond segment (e.g. C=C=C).

    Cumulated double bonds are geometrically linear (sp-hybridized center),
    which makes Z-matrix dihedral interpolation ill-defined. When this
    function returns ``True``, the ring-closure algorithm should be used
    instead of Z-matrix interpolation.

    Args:
        mol: RMG Molecule whose bond topology is used for the BFS.
        forming_bond: ``(i, j)`` atom-index pair of the bond being formed.

    Returns:
        ``True`` if at least one interior vertex of the shortest path has
        both flanking bonds with order >= 2 (cumulated).
    """
    i_atom, j_atom = forming_bond

    adj = mol_to_adjacency(mol)

    prev: dict[int, int | None] = {i_atom: None}
    queue: deque = deque([i_atom])
    path: list[int] | None = None
    while queue:
        node = queue.popleft()
        for nbr in adj[node]:
            if nbr not in prev:
                prev[nbr] = node
                if nbr == j_atom:
                    p: list[int] = []
                    cur: int | None = j_atom
                    while cur is not None:
                        p.append(cur)
                        cur = prev[cur]
                    path = p[::-1]
                    break
                queue.append(nbr)
        if path is not None:
            break

    if path is None or len(path) < 3:
        return False

    for k in range(1, len(path) - 1):
        prev_atom = mol.atoms[path[k - 1]]
        curr_atom = mol.atoms[path[k]]
        next_atom = mol.atoms[path[k + 1]]
        bo_prev = float(curr_atom.bonds[prev_atom].order) if prev_atom in curr_atom.bonds else 1.0
        bo_next = float(curr_atom.bonds[next_atom].order) if next_atom in curr_atom.bonds else 1.0
        if bo_prev >= 2.0 and bo_next >= 2.0:
            return True
    return False


# ---------------------------------------------------------------------------
# Ring-closure geometry helpers
# ---------------------------------------------------------------------------

def _flatten_path_dihedrals(coords: np.ndarray,
                            path: list[int],
                            adj: dict[int, set[int]],
                            fraction: float = 1.0,
                            ) -> None:
    """
    Rotate every consecutive 4-atom dihedral along ``path`` toward 0°.

    For each tuple ``(p1, p2, p3, p4)`` of consecutive atoms in ``path``,
    compute the current dihedral and rotate the downstream subgraph
    around the ``p2-p3`` axis by ``-current * fraction`` degrees, where
    "downstream" is the connected component reachable from ``p3`` once
    the ``p2-p3`` edge is removed. With ``fraction = 1.0`` the path is
    fully flattened; smaller fractions partially flatten and are used
    by the bisection in :func:`ring_closure_xyz` to dial in the
    forming-bond distance.

    Args:
        coords (np.ndarray): Coordinate array, shape ``(N, 3)``. Mutated
            in place.
        path (list[int]): Ordered list of atom indices defining the
            backbone path that closes into the forming ring.
        adj (dict[int, set[int]]): Molecular adjacency map (atom index →
            set of bonded atom indices), used to determine the
            downstream subgraph for each rotation.
        fraction (float): Fraction of each current dihedral to remove.
            Defaults to ``1.0`` (full flattening).

    Returns:
        None: ``coords`` is mutated in place.
    """
    for k in range(len(path) - 3):
        p1, p2, p3, p4 = path[k], path[k + 1], path[k + 2], path[k + 3]
        current_dih = dihedral_deg(coords[p1], coords[p2], coords[p3], coords[p4])
        if abs(current_dih) < 0.1:
            continue
        origin = coords[p2].copy()
        axis_raw = coords[p3] - origin
        axis_norm = np.linalg.norm(axis_raw)
        if axis_norm < 1e-8:
            continue
        axis = axis_raw / axis_norm
        down = downstream(adj, p2, p3)
        if not down:
            continue
        rotate_atoms(coords, origin, axis, down, np.radians(-current_dih * fraction))


def _compress_path_angles(coords: np.ndarray,
                          path: list[int],
                          adj: dict[int, set[int]],
                          target_ang: float,
                          angle_flex: list[float] | None = None,
                          ) -> None:
    """
    Compress each interior bond angle along ``path`` toward ``target_ang``.

    For every interior atom ``b`` along ``path`` (with ring neighbors
    ``a`` and ``c``), this rotates the downstream subgraph anchored at
    ``c`` so the ``a–b–c`` angle moves toward ``target_ang``. When
    ``a`` and ``c`` are collinear with ``b`` (degenerate cross
    product), an arbitrary perpendicular axis is constructed.

    The optional ``angle_flex`` per-position weights scale how much each
    angle is allowed to bend. Positions corresponding to cumulated
    double bonds (e.g. allene sp centers) are typically given low flex
    (≈0.1) so the linear geometry is preserved while remaining angles
    absorb the bending; non-cumulated positions absorb the leftover
    bend in proportion to ``1 / mean_bond_order``.

    Args:
        coords (np.ndarray): Coordinate array, shape ``(N, 3)``. Mutated
            in place.
        path (list[int]): Ordered list of atom indices defining the
            backbone path.
        adj (dict[int, set[int]]): Molecular adjacency map used for the
            downstream rotation.
        target_ang (float): Target interior bond angle in degrees.
        angle_flex (list[float] | None): Per-position flex weights of
            length ``len(path) - 2`` (one per interior atom). When
            ``None`` (default), all positions get a uniform weight of
            ``1.0``.

    Returns:
        None: ``coords`` is mutated in place.
    """
    for k in range(1, len(path) - 1):
        a = path[k - 1]
        b = path[k]
        cc = path[k + 1]

        v_ba = coords[a] - coords[b]
        v_bc = coords[cc] - coords[b]
        norm_ba = np.linalg.norm(v_ba)
        norm_bc = np.linalg.norm(v_bc)
        if norm_ba < 1e-8 or norm_bc < 1e-8:
            continue

        cos_cur = np.clip(np.dot(v_ba, v_bc) / (norm_ba * norm_bc), -1.0, 1.0)
        cur_ang = np.degrees(np.arccos(cos_cur))
        delta = target_ang - cur_ang
        if angle_flex is not None:
            delta *= angle_flex[k - 1]
        if abs(delta) < 0.01:
            continue

        rot_axis = np.cross(v_ba, v_bc)
        rot_norm = np.linalg.norm(rot_axis)
        if rot_norm < 1e-8:
            arb = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(v_ba / norm_ba, arb)) > 0.9:
                arb = np.array([0.0, 1.0, 0.0])
            rot_axis = np.cross(v_ba, arb)
            rot_norm = np.linalg.norm(rot_axis)
            if rot_norm < 1e-8:
                continue
        rot_axis = rot_axis / rot_norm

        down = downstream(adj, b, cc)
        if not down:
            continue
        rotate_atoms(coords, coords[b].copy(), rot_axis, down, np.radians(delta))


def _build_ring_geometry(coords: np.ndarray,
                         path: list[int],
                         adj: dict[int, set[int]],
                         target_ang: float,
                         angle_flex: list[float] | None = None,
                         dih_fraction: float = 1.0,
                         ) -> float:
    """
    Apply the two ring-closure passes and return the new endpoint distance.

    The function first flattens path dihedrals (by ``dih_fraction``) and
    then compresses interior angles toward ``target_ang``, both in place
    on ``coords``. The returned value is the distance between the first
    and last atom of ``path`` after both passes — used by the bisection
    in :func:`ring_closure_xyz` to home in on the requested forming-bond
    distance.

    Args:
        coords (np.ndarray): Coordinate array, shape ``(N, 3)``. Mutated
            in place by both passes.
        path (list[int]): Ordered list of atom indices defining the ring
            path.
        adj (dict[int, set[int]]): Molecular adjacency map shared by
            both passes.
        target_ang (float): Target interior bond angle in degrees.
        angle_flex (list[float] | None): Per-position flex weights
            forwarded to the angle-compression pass.
        dih_fraction (float): Fraction of each dihedral to remove in the
            flattening pass; ``1.0`` flattens fully.

    Returns:
        float: Euclidean distance between ``coords[path[-1]]`` and
            ``coords[path[0]]`` after the two passes.
    """
    _flatten_path_dihedrals(coords, path, adj, fraction=dih_fraction)
    _compress_path_angles(coords, path, adj, target_ang, angle_flex=angle_flex)
    return float(np.linalg.norm(coords[path[-1]] - coords[path[0]]))


# ---------------------------------------------------------------------------
# 3D builders
# ---------------------------------------------------------------------------

def get_near_attack_xyz(xyz: dict,
                        mol: Molecule,
                        bonds: list[tuple[int, int]],
                        ) -> dict:
    """
    Return a near-attack conformation of ``xyz`` by setting each backbone dihedral to
    the ring-size-appropriate ideal angle for the cyclic transition state.

    For each forming bond (i, j):

    1. Find the BFS path from ``i`` to ``j``; the ring size is ``len(path)``.
    2. Look up the ideal backbone dihedral magnitude ``T`` for that ring size
       (see ``_TS_RING_DIHEDRALS``; 0° for 4-membered, 60° for 6-membered, etc.).
    3. For each rotatable interior bond at position ``k`` in the path, using atoms
       ``path[k-1], path[k], path[k+1], path[k+2]`` to define the dihedral:
       a. Compute the rotation needed to set the dihedral to ``+T`` vs ``-T``.
       b. Apply the rotation that most reduces the ``i``–``j`` distance.
    4. Bonds are processed in outer-to-inner order (towards ``j``), so each rotation
       uses updated coordinates from the previous one.

    Bonds that are not rotatable (double, triple, aromatic, in-ring) are skipped.
    Paths shorter than 4 atoms (no interior bond) are skipped.

    Args:
        xyz (dict): XYZ coordinate dictionary (``{'symbols': ..., 'coords': ...}``).
        mol (Molecule): The RMG Molecule whose bond topology matches ``xyz``.
        bonds (list[tuple[int, int]]): Atom-index pairs ``(i, j)`` — forming bonds —
            in reactant atom ordering.

    Returns:
        dict: A new XYZ dictionary with the near-attack geometry.
    """
    coords = np.array(xyz['coords'], dtype=float)

    adj = mol_to_adjacency(mol)

    for (i_atom, j_atom) in bonds:
        path = bfs_path(adj, i_atom, j_atom)
        if path is None or len(path) < 4:
            continue

        ring_size = len(path)
        dihedrals_list = _TS_RING_DIHEDRALS.get(ring_size)

        for k in range(1, len(path) - 2):
            a_idx, b_idx = path[k], path[k + 1]
            bond = mol.atoms[a_idx].bonds.get(mol.atoms[b_idx])
            if bond is None or not bond.is_rotatable():
                continue
            origin = coords[a_idx].copy()
            axis_raw = coords[b_idx] - origin
            axis_norm = np.linalg.norm(axis_raw)
            if axis_norm < 1e-8:
                continue
            axis = axis_raw / axis_norm
            down = downstream(adj, a_idx, b_idx)
            if not down:
                continue

            if dihedrals_list is not None and k - 1 < len(dihedrals_list):
                target_mag = dihedrals_list[k - 1]
            else:
                target_mag = _TS_RING_DIHEDRAL_DEFAULT

            current_dih = dihedral_deg(coords[path[k - 1]], coords[a_idx],
                                        coords[b_idx], coords[path[k + 2]])

            delta_plus = (target_mag - current_dih) * np.pi / 180.0
            delta_minus = (-target_mag - current_dih) * np.pi / 180.0

            trial_plus = coords.copy()
            rotate_atoms(trial_plus, origin, axis, down, delta_plus)
            dist_plus = float(np.linalg.norm(trial_plus[j_atom] - trial_plus[i_atom]))

            trial_minus = coords.copy()
            rotate_atoms(trial_minus, origin, axis, down, delta_minus)
            dist_minus = float(np.linalg.norm(trial_minus[j_atom] - trial_minus[i_atom]))

            rotate_atoms(coords, origin, axis, down,
                    delta_plus if dist_plus <= dist_minus else delta_minus)

    for (i_atom, j_atom) in bonds:
        if xyz['symbols'][i_atom] == 'H':
            h_atom, acceptor = i_atom, j_atom
        elif xyz['symbols'][j_atom] == 'H':
            h_atom, acceptor = j_atom, i_atom
        else:
            continue
        path = bfs_path(adj, h_atom, acceptor)
        if path is None or len(path) < 3:
            continue
        pivot_a, pivot_b = path[1], path[2]
        t_origin = coords[pivot_a].copy()
        t_axis_raw = coords[pivot_b] - t_origin
        t_axis_norm = np.linalg.norm(t_axis_raw)
        if t_axis_norm < 1e-8:
            continue
        t_axis = t_axis_raw / t_axis_norm
        t_group = {nbr for nbr in adj[pivot_a] if nbr != pivot_b}
        if h_atom not in t_group:
            continue
        best_dist = float(np.linalg.norm(coords[h_atom] - coords[acceptor]))
        best_delta = 0.0
        for deg in np.arange(-60.0, 61.0, 5.0):
            if abs(deg) < 1e-6:
                continue
            ang = deg * np.pi / 180.0
            trial = coords.copy()
            rotate_atoms(trial, t_origin, t_axis, t_group, ang)
            dist = float(np.linalg.norm(trial[h_atom] - trial[acceptor]))
            if dist < best_dist:
                best_dist = dist
                best_delta = ang
        if abs(best_delta) > 1e-6:
            rotate_atoms(coords, t_origin, t_axis, t_group, best_delta)

    for (i_atom, j_atom) in bonds:
        if xyz['symbols'][i_atom] == 'H':
            h_atom, acceptor = i_atom, j_atom
        elif xyz['symbols'][j_atom] == 'H':
            h_atom, acceptor = j_atom, i_atom
        else:
            continue
        path = bfs_path(adj, h_atom, acceptor)
        if path is None or len(path) < 4:
            continue
        donor_c, next_b, ref_a = path[1], path[2], path[3]

        s_axis_raw = coords[next_b] - coords[donor_c]
        s_axis_norm = np.linalg.norm(s_axis_raw)
        if s_axis_norm < 1e-8:
            continue
        s_axis = s_axis_raw / s_axis_norm
        s_origin = coords[donor_c].copy()

        dih_mig = dihedral_deg(coords[h_atom], coords[donor_c], coords[next_b], coords[ref_a])
        other_hs = [nbr for nbr in adj[donor_c]
                    if nbr != next_b and nbr != h_atom and xyz['symbols'][nbr] == 'H']
        for other_h in other_hs:
            dih_h = dihedral_deg(coords[other_h], coords[donor_c], coords[next_b], coords[ref_a])
            t_plus = dih_mig + 120.0
            t_minus = dih_mig - 120.0
            d_plus = (t_plus - dih_h) * np.pi / 180.0
            d_minus = (t_minus - dih_h) * np.pi / 180.0
            delta = d_plus if abs(d_plus) <= abs(d_minus) else d_minus
            if abs(delta) > 1e-6:
                rotate_atoms(coords, s_origin, s_axis, {other_h}, delta)

    new_xyz = copy.deepcopy(xyz)
    new_xyz['coords'] = tuple(tuple(float(v) for v in row) for row in coords)
    return new_xyz


def ring_closure_xyz(xyz: dict,
                      mol: Molecule,
                      forming_bond: tuple[int, int],
                      target_distance: float = 2.3,
                      exclude_bonds: list[tuple[int, int]] | None = None,
                      ) -> dict | None:
    """
    Generate a TS guess for a ring-closing reaction by uniformly compressing
    bond angles along the shortest atom chain connecting the two reactive sites.

    This function is designed for cases where the two reactive sites are far
    apart (> 4.5 Å) in the reactant geometry, making standard Z-matrix
    interpolation unreliable.

    Also used for intra-fragment ring contraction after ``stretch_bond()``
    when a forming bond connects two atoms within the same fragment — e.g.,
    cyclic ether/thioether formation where the O–O or S–O bond breaks and a
    C–O or C–S ring forms simultaneously.

    Algorithm:
        1. Identify the shortest atom chain connecting the two reactive sites.
        2. Uniformly compress all bond angles along that chain to the ideal
           regular-polygon interior angle for the ring size, bringing the
           reactive sites together into a ring geometry.
        3. Set the forming bond length to ``target_distance`` (≈ 1.5× the C–C
           single-bond length of 1.54 Å) — this bond remains longer than all
           other bonds in the structure.
        4. Detect any H atom entrapped inside the forming ring and rotate its
           dihedral angle by 180° relative to the ring plane.

    Args:
        xyz (dict): XYZ coordinate dictionary.
        mol (Molecule): The RMG Molecule whose bond topology matches ``xyz``.
        forming_bond (tuple[int, int]): Atom-index pair ``(i, j)`` of the bond
            being formed.
        target_distance (float): Desired distance (Å) between the two reactive
            sites in the TS guess. Default 2.3 Å (≈ 1.5 × 1.54 Å).
        exclude_bonds (list[tuple[int, int]], optional): Bonds to exclude from
            the adjacency graph (e.g. severed split bonds). The BFS path will
            not traverse these edges, ensuring it finds the correct ring path
            through the remaining fragment.

    Returns:
        dict | None: A new XYZ dictionary with the ring-closure TS geometry,
            or ``None`` if the chain could not be found or the geometry could
            not be constructed.
    """
    i_atom, j_atom = forming_bond
    coords = np.array(xyz['coords'], dtype=float)
    n_atoms = len(coords)

    adj = mol_to_adjacency(mol)
    for a, b in (exclude_bonds or []):
        adj[a].discard(b)
        adj[b].discard(a)

    path = bfs_path(adj, i_atom, j_atom)
    if path is None or len(path) < 3:
        return None

    ring_size = len(path)
    ideal_angle = (ring_size - 2) * 180.0 / ring_size

    # Cumulated double bonds (e.g. C=C=C) are geometrically linear (sp center) and
    # must stay nearly linear during ring closure; remaining angles absorb the
    # bending. Cumulated angles get 0.1 flex; others get 1/mean_bo. Paths without
    # cumulated bonds use uniform compression.
    path_bond_orders: list[float] = []
    for k in range(len(path) - 1):
        a_atom_obj = mol.atoms[path[k]]
        b_atom_obj = mol.atoms[path[k + 1]]
        if b_atom_obj in a_atom_obj.bonds:
            path_bond_orders.append(float(a_atom_obj.bonds[b_atom_obj].order))
        else:
            path_bond_orders.append(1.0)

    _angle_flex: list[float] | None = None
    if len(path_bond_orders) >= 2:
        has_cumulated = any(
            path_bond_orders[k] >= 2.0 and path_bond_orders[k + 1] >= 2.0
            for k in range(len(path_bond_orders) - 1)
        )
        if has_cumulated:
            raw_flex = []
            for k in range(len(path_bond_orders) - 1):
                bo_left = path_bond_orders[k]
                bo_right = path_bond_orders[k + 1]
                if bo_left >= 2.0 and bo_right >= 2.0:
                    raw_flex.append(0.1)
                else:
                    mean_bo = (bo_left + bo_right) / 2.0
                    raw_flex.append(1.0 / max(mean_bo, 0.5))
            total_flex = sum(raw_flex)
            if total_flex > 1e-8:
                n_flex = len(raw_flex)
                _angle_flex = [f * n_flex / total_flex for f in raw_flex]

    # --- Step 0: scale path bond lengths for small rings ---
    if ring_size == 3:
        strain_corr = 0.14
        pivot = path[1]
        path_centroid_before = coords[path].mean(axis=0)

        path_bonds_info: list[tuple[int, int, float, float, float]] = []
        for k in range(len(path) - 1):
            ak, bk = path[k], path[k + 1]
            sym_a = xyz['symbols'][ak]
            sym_b = xyz['symbols'][bk]
            sbl = get_single_bond_length(sym_a, sym_b)
            cur = float(np.linalg.norm(coords[ak] - coords[bk]))
            gap = max(sbl - cur, 0.0)
            path_bonds_info.append((ak, bk, sbl, cur, gap))

        max_gap = max((info[4] for info in path_bonds_info), default=0.0)

        if max_gap > 0.01:
            for ak, bk, sbl, cur, gap in path_bonds_info:
                target_len = sbl + strain_corr * (gap / max_gap)
                if target_len <= cur + 0.001:
                    continue
                stretch = target_len - cur
                if ak == pivot:
                    endpoint, anchor = bk, ak
                else:
                    endpoint, anchor = ak, bk
                direction = coords[endpoint] - coords[anchor]
                d_norm = np.linalg.norm(direction)
                if d_norm < 1e-8:
                    continue
                direction /= d_norm
                down = downstream(adj, anchor, endpoint)
                for di in down:
                    coords[di] += direction * stretch

            for nbr in adj[pivot]:
                if nbr in set(path):
                    continue
                sym_p = xyz['symbols'][pivot]
                sym_n = xyz['symbols'][nbr]
                sbl_adj = get_single_bond_length(sym_p, sym_n)
                cur_adj = float(np.linalg.norm(coords[nbr] - coords[pivot]))
                if cur_adj >= sbl_adj:
                    continue
                dbl_len = sbl_adj - 0.24
                if cur_adj <= dbl_len:
                    continue
                frac = (cur_adj - dbl_len) / (sbl_adj - dbl_len) if sbl_adj > dbl_len else 0.0
                shortening = strain_corr * frac * max(path_bonds_info[0][4], path_bonds_info[-1][4]) / max_gap
                new_adj = cur_adj - shortening
                if new_adj < dbl_len:
                    new_adj = dbl_len
                delta_adj = cur_adj - new_adj
                if delta_adj < 0.001:
                    continue
                dir_adj = coords[nbr] - coords[pivot]
                dir_adj_norm = np.linalg.norm(dir_adj)
                if dir_adj_norm < 1e-8:
                    continue
                dir_adj /= dir_adj_norm
                down_adj = downstream(adj, pivot, nbr)
                for di in down_adj:
                    coords[di] -= dir_adj * delta_adj

            centroid_shift = path_centroid_before - coords[path].mean(axis=0)
            coords += centroid_shift

    # --- Steps 1–3: bisect to achieve target_distance ---
    original_coords = coords.copy()

    start_angles: list[float] = []
    for k in range(1, len(path) - 1):
        a, b, c = path[k - 1], path[k], path[k + 1]
        v_ba = coords[a] - coords[b]
        v_bc = coords[c] - coords[b]
        n_ba = np.linalg.norm(v_ba)
        n_bc = np.linalg.norm(v_bc)
        if n_ba < 1e-8 or n_bc < 1e-8:
            start_angles.append(ideal_angle)
        else:
            cos_a = np.clip(np.dot(v_ba, v_bc) / (n_ba * n_bc), -1.0, 1.0)
            start_angles.append(np.degrees(np.arccos(cos_a)))
    avg_start = float(np.mean(start_angles)) if start_angles else ideal_angle

    lo, hi = ideal_angle, avg_start
    best_coords = coords.copy()
    found = False
    for _ in range(30):
        mid = (lo + hi) / 2.0
        trial = original_coords.copy()
        dist = _build_ring_geometry(trial, path, adj, mid, angle_flex=_angle_flex)
        if abs(dist - target_distance) < 0.005:
            best_coords = trial
            found = True
            break
        if dist > target_distance:
            hi = mid
        else:
            lo = mid
        best_coords = trial

    if not found:
        trial_full = original_coords.copy()
        dist_full = _build_ring_geometry(trial_full, path, adj, avg_start, angle_flex=_angle_flex)
        if dist_full < target_distance:
            dih_lo, dih_hi = 0.0, 1.0
            for _ in range(30):
                dih_mid = (dih_lo + dih_hi) / 2.0
                trial = original_coords.copy()
                dist = _build_ring_geometry(trial, path, adj, avg_start,
                                            angle_flex=_angle_flex, dih_fraction=dih_mid)
                if abs(dist - target_distance) < 0.005:
                    best_coords = trial
                    found = True
                    break
                if dist > target_distance:
                    dih_lo = dih_mid
                else:
                    dih_hi = dih_mid
                best_coords = trial

    coords[:] = best_coords

    # --- Step 4: detect and flip H atoms entrapped inside the forming ring ---
    ring_center = coords[path].mean(axis=0)
    ring_coords = coords[path] - ring_center
    _, _, vh = np.linalg.svd(ring_coords)
    ring_normal = vh[-1]
    rn_norm = np.linalg.norm(ring_normal)
    if rn_norm > 1e-8:
        ring_normal = ring_normal / rn_norm
    ring_atoms_set = set(path)
    endpoint_set = {i_atom, j_atom}
    for k in range(len(path)):
        atom_idx = path[k]
        if atom_idx in endpoint_set:
            continue
        h_indices = [h for h in adj[atom_idx]
                     if h not in ring_atoms_set and xyz['symbols'][h] == 'H']
        if not h_indices:
            continue
        entrapped: list[int] = []
        for h_idx in h_indices:
            to_h = coords[h_idx] - coords[atom_idx]
            in_plane = to_h - ring_normal * np.dot(to_h, ring_normal)
            in_plane_len = np.linalg.norm(in_plane)
            out_of_plane = abs(np.dot(to_h, ring_normal))
            if out_of_plane > in_plane_len * 0.5:
                continue
            to_center = ring_center - coords[atom_idx]
            in_plane_center = to_center - ring_normal * np.dot(to_center, ring_normal)
            if np.dot(in_plane, in_plane_center) > 0:
                entrapped.append(h_idx)
        if not entrapped:
            continue
        path_nbrs = [p for p in [path[k - 1] if k > 0 else None,
                                 path[k + 1] if k < len(path) - 1 else None]
                     if p is not None]
        if not path_nbrs:
            continue
        pivot = path_nbrs[0]
        rot_origin = coords[atom_idx].copy()
        rot_ax = coords[pivot] - rot_origin
        rot_ax_norm = np.linalg.norm(rot_ax)
        if rot_ax_norm < 1e-8:
            continue
        rot_ax = rot_ax / rot_ax_norm
        rotate_atoms(coords, rot_origin, rot_ax, set(h_indices), np.pi)

    # --- Step 5: rotate substituent groups that occlude the forming bond ---
    for endpoint, opposite in [(i_atom, j_atom), (j_atom, i_atom)]:
        subs = [s for s in adj[endpoint] if s not in ring_atoms_set]
        if not subs:
            continue
        forming_dist = float(np.linalg.norm(coords[endpoint] - coords[opposite]))
        closest_sub_dist = min(float(np.linalg.norm(coords[s] - coords[opposite]))
                               for s in subs)
        if ring_size > 4 and closest_sub_dist >= forming_dist:
            continue
        ring_nbr = [p for p in adj[endpoint] if p in ring_atoms_set]
        if not ring_nbr:
            continue
        pivot = ring_nbr[0]
        rot_origin = coords[endpoint].copy()
        rot_ax_raw = coords[pivot] - rot_origin
        rot_ax_norm = np.linalg.norm(rot_ax_raw)
        if rot_ax_norm < 1e-8:
            continue
        rot_ax = rot_ax_raw / rot_ax_norm
        sub_set: set[int] = set()
        for s in subs:
            sub_set.update(downstream(adj, endpoint, s))

        best_min_dist = closest_sub_dist
        best_angle = 0.0
        for deg in range(5, 360, 5):
            trial = coords.copy()
            rotate_atoms(trial, rot_origin, rot_ax, sub_set, np.radians(deg))
            min_d = min(float(np.linalg.norm(trial[s] - trial[opposite]))
                        for s in subs)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_angle = float(deg)
        if best_angle > 0:
            rotate_atoms(coords, rot_origin, rot_ax, sub_set,
                    np.radians(best_angle))

        # Collinear substituents (e.g. allene endpoint C=C=C) lie on the rotation
        # axis and cannot be moved by rotating around it; bend them 60° away from
        # the opposite endpoint so the exocyclic group points out of the ring.
        for s in subs:
            if xyz['symbols'][s] == 'H':
                continue
            s_dist = float(np.linalg.norm(coords[s] - coords[opposite]))
            if s_dist >= forming_dist:
                continue
            v_sub = coords[s] - coords[endpoint]
            v_sub_len = np.linalg.norm(v_sub)
            if v_sub_len < 1e-8:
                continue
            cos_col = abs(np.dot(v_sub / v_sub_len, rot_ax))
            if cos_col < 0.95:
                continue
            v_to_opp = coords[opposite] - coords[endpoint]
            bend_ax = np.cross(rot_ax, v_to_opp)
            bn = np.linalg.norm(bend_ax)
            if bn < 1e-8:
                bend_ax = np.cross(rot_ax, ring_center - coords[endpoint])
                bn = np.linalg.norm(bend_ax)
            if bn < 1e-8:
                continue
            bend_ax /= bn
            s_down = downstream(adj, endpoint, s)
            best_bend = 0.0
            best_d = s_dist
            for sign in [1.0, -1.0]:
                trial = coords.copy()
                rotate_atoms(trial, coords[endpoint].copy(), bend_ax, s_down,
                        sign * np.radians(60))
                d = float(np.linalg.norm(trial[s] - trial[opposite]))
                if d > best_d:
                    best_d = d
                    best_bend = sign * 60.0
            if abs(best_bend) > 0.01:
                rotate_atoms(coords, coords[endpoint].copy(), bend_ax, s_down,
                        np.radians(best_bend))

    # --- Step 5b: pyramidalise endpoint CH₂ groups for small rings ---
    if ring_size <= 3:
        for ep in [i_atom, j_atom]:
            h_subs = [s for s in adj[ep]
                      if s not in ring_atoms_set and xyz['symbols'][s] == 'H']
            if len(h_subs) < 2:
                continue
            rn_list = [p for p in adj[ep] if p in ring_atoms_set]
            if not rn_list:
                continue
            opp = j_atom if ep == i_atom else i_atom
            v_rn = coords[rn_list[0]] - coords[ep]
            v_op = coords[opp] - coords[ep]
            n_rn = float(np.linalg.norm(v_rn))
            n_op = float(np.linalg.norm(v_op))
            if n_rn < 1e-8 or n_op < 1e-8:
                continue
            cos_ring = float(np.clip(np.dot(v_rn, v_op) / (n_rn * n_op), -1.0, 1.0))
            ring_ang = np.degrees(np.arccos(cos_ring))
            target_hch = max(180.0 - 2.0 * ring_ang - 5.0, 60.0)
            v1 = coords[h_subs[0]] - coords[ep]
            v2 = coords[h_subs[1]] - coords[ep]
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-8 or n2 < 1e-8:
                continue
            cos_hch = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            cur_hch = np.degrees(np.arccos(cos_hch))
            if cur_hch <= target_hch + 0.5:
                continue
            int_bis = v_rn / n_rn + v_op / n_op
            ib_norm = float(np.linalg.norm(int_bis))
            if ib_norm < 1e-8:
                continue
            bis_target = -int_bis / ib_norm
            cur_bis = v1 / n1 + v2 / n2
            cb_norm = float(np.linalg.norm(cur_bis))
            if cb_norm < 1e-8:
                continue
            cur_bis /= cb_norm
            n_path = np.cross(coords[path[1]] - coords[path[0]],
                              coords[path[2]] - coords[path[1]])
            np_norm = float(np.linalg.norm(n_path))
            if np_norm < 1e-8:
                continue
            n_path /= np_norm
            cos_tilt = float(np.clip(np.dot(cur_bis, bis_target), -1.0, 1.0))
            cross_tilt = np.cross(cur_bis, bis_target)
            sin_tilt = float(np.dot(cross_tilt, n_path))
            tilt_angle = float(np.arctan2(sin_tilt, cos_tilt))
            if abs(tilt_angle) > 1e-6:
                sub_set_tilt = set(h_subs)
                rotate_atoms(coords, coords[ep].copy(), n_path, sub_set_tilt,
                        tilt_angle)
            v1 = coords[h_subs[0]] - coords[ep]
            v2 = coords[h_subs[1]] - coords[ep]
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-8 or n2 < 1e-8:
                continue
            cur_bis = v1 / n1 + v2 / n2
            cb_norm = float(np.linalg.norm(cur_bis))
            if cb_norm < 1e-8:
                continue
            cur_bis /= cb_norm
            cos_hch2 = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            cur_hch2 = np.degrees(np.arccos(cos_hch2))
            compress_deg = (cur_hch2 - target_hch) / 2.0
            if compress_deg <= 0:
                continue
            compress_rad = np.radians(compress_deg)
            for h in h_subs:
                vec = coords[h] - coords[ep]
                vec_len = float(np.linalg.norm(vec))
                if vec_len < 1e-8:
                    continue
                vec_u = vec / vec_len
                rax = np.cross(vec_u, cur_bis)
                rax_n = float(np.linalg.norm(rax))
                if rax_n < 1e-8:
                    continue
                rax /= rax_n
                rotate_atoms(coords, coords[ep].copy(), rax, {h}, compress_rad)

    # --- Step 6: normalise X-H bond lengths ---
    for idx in range(n_atoms):
        if xyz['symbols'][idx] == 'H':
            continue
        sym_heavy = xyz['symbols'][idx]
        for h_idx in adj[idx]:
            if xyz['symbols'][h_idx] != 'H':
                continue
            vec = coords[h_idx] - coords[idx]
            cur_len = np.linalg.norm(vec)
            if cur_len < 1e-8:
                continue
            target_len = get_single_bond_length(sym_heavy, 'H')
            coords[h_idx] = coords[idx] + vec * (target_len / cur_len)

    new_xyz = copy.deepcopy(xyz)
    new_xyz['coords'] = tuple(tuple(float(v) for v in row) for row in coords)
    return new_xyz


def build_4center_interchange_ts(r_xyz: dict,
                                 r_mol: Molecule,
                                 bb: list[tuple[int, int]],
                                 fb: list[tuple[int, int]],
                                 weight: float = 0.5,
                                 label: str = '',
                                 ) -> dict | None:
    """
    Build a TS guess for a 4-center interchange reaction (e.g. 1,2_XY_interchange).

    In a 4-center interchange two substituents swap positions on adjacent centers::

            X           X ---- C1
           / \\             |          |
      C1 - C2   TS:   C2 ---- Y
           \\ /
            Y

    The reactant has X-C1 and Y-C2; the product has Y-C1 and X-C2. In the TS
    all four bonds (X-C1, X-C2, Y-C1, Y-C2) are partial.

    Args:
        r_xyz: Reactant XYZ coordinate dictionary.
        r_mol: Reactant RMG Molecule.
        bb: List of breaking-bond (i, j) pairs.
        fb: List of forming-bond (i, j) pairs.
        weight: Interpolation weight (0 = reactant, 1 = product).
        label: Logging label for debug messages.

    Returns:
        A validated XYZ dict, or ``None`` if pattern detection or validation fails.
    """
    if len(bb) != 2 or len(fb) != 2:
        return None
    reactive_atoms: set[int] = set()
    for bond in bb + fb:
        reactive_atoms.update(bond)
    if len(reactive_atoms) != 4:
        return None

    centers: tuple[int, int] | None = None
    for a in reactive_atoms:
        for b in reactive_atoms:
            if a < b and r_mol.has_bond(r_mol.atoms[a], r_mol.atoms[b]):
                if (a, b) not in bb and (b, a) not in bb:
                    centers = (a, b)
                    break
        if centers is not None:
            break
    if centers is None:
        logger.debug(f'Linear ({label}): 4-center interchange pattern not detected (no center pair bonded in R).')
        return None

    migrants = sorted(reactive_atoms - set(centers))
    if len(migrants) != 2:
        return None

    c1, c2 = centers
    m1, m2 = migrants
    symbols = r_xyz['symbols']
    coords = np.array(r_xyz['coords'], dtype=float)
    atom_to_idx = {atom: idx for idx, atom in enumerate(r_mol.atoms)}

    scale = 2.0 * (1.0 - weight)

    pos_c1, pos_c2 = coords[c1], coords[c2]
    axis = pos_c2 - pos_c1
    axis_len = float(np.linalg.norm(axis))
    if axis_len < 1e-6:
        return None
    axis_hat = axis / axis_len

    ts_coords = coords.copy()

    for migrant in [m1, m2]:
        sym_m = symbols[migrant]
        target_mc1 = get_single_bond_length(sym_m, symbols[c1]) + PAULING_DELTA * scale
        target_mc2 = get_single_bond_length(sym_m, symbols[c2]) + PAULING_DELTA * scale

        x_along = (axis_len ** 2 + target_mc1 ** 2 - target_mc2 ** 2) / (2.0 * axis_len)
        h_sq = target_mc1 ** 2 - x_along ** 2
        if h_sq < 0:
            logger.debug(f'Linear ({label}): 4-center triangle infeasible for migrant '
                         f'{sym_m}{migrant} (h²={h_sq:.4f}).')
            return None
        h = float(np.sqrt(h_sq))

        m_vec = coords[migrant] - pos_c1
        m_proj = float(np.dot(m_vec, axis_hat))
        m_foot = pos_c1 + axis_hat * m_proj
        perp = coords[migrant] - m_foot
        perp_len = float(np.linalg.norm(perp))
        if perp_len > 1e-6:
            perp_hat = perp / perp_len
        else:
            perp_hat = np.cross(axis_hat, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(perp_hat) < 1e-6:
                perp_hat = np.cross(axis_hat, np.array([0.0, 1.0, 0.0]))
            perp_hat = perp_hat / np.linalg.norm(perp_hat)

        new_pos = pos_c1 + axis_hat * x_along + perp_hat * h
        displacement = new_pos - coords[migrant]
        ts_coords[migrant] = new_pos

        rmg_atom = r_mol.atoms[migrant]
        for neighbor in rmg_atom.edges.keys():
            n_idx = atom_to_idx[neighbor]
            if neighbor.is_hydrogen():
                ts_coords[n_idx] = coords[n_idx] + displacement

    m1_vec = ts_coords[m1] - pos_c1
    m2_vec = ts_coords[m2] - pos_c1
    m1_perp = m1_vec - axis_hat * float(np.dot(m1_vec, axis_hat))
    m2_perp = m2_vec - axis_hat * float(np.dot(m2_vec, axis_hat))
    if float(np.dot(m1_perp, m2_perp)) > 0:
        m2_foot = pos_c1 + axis_hat * float(np.dot(ts_coords[m2] - pos_c1, axis_hat))
        displacement_flip = 2.0 * (m2_foot - ts_coords[m2])
        ts_coords[m2] += displacement_flip
        rmg_atom = r_mol.atoms[m2]
        for neighbor in rmg_atom.edges.keys():
            n_idx = atom_to_idx[neighbor]
            if neighbor.is_hydrogen():
                ts_coords[n_idx] += displacement_flip

    ts_xyz = {'symbols': r_xyz['symbols'], 'isotopes': r_xyz['isotopes'],
              'coords': tuple(tuple(c) for c in ts_coords)}

    is_valid, reason = validate_ts_guess(
        ts_xyz, set(), fb, r_mol, label=f'{label}, 4center', family=None)
    if not is_valid:
        logger.debug(f'Linear ({label}): 4-center TS rejected — {reason}.')
        return None
    return ts_xyz


def generate_zmat_branch(anchor_xyz: dict,
                         anchor_mol: Molecule,
                         target_xyz: dict,
                         weight: float,
                         reactive_xyz_indices: set[int],
                         anchors: list | None,
                         constraints: dict | None,
                         r_mol: Molecule,
                         forming_bonds: list[tuple[int, int]],
                         breaking_bonds: list[tuple[int, int]],
                         label: str = '',
                         skip_postprocess: bool = False,
                         family: str | None = None,
                         r_label_map: dict[str, int] | None = None,
                         redistribute_ch2: bool = False,
                         ) -> dict | None:
    """
    Generate a single TS guess from one Z-matrix branch (Type R or Type P).

    Builds a Z-matrix from ``anchor_xyz`` / ``anchor_mol``, projects
    ``target_xyz`` onto that topology, blends at ``weight``, converts back
    to Cartesian, optionally applies the postprocessing pipeline, and validates.

    Args:
        anchor_xyz: XYZ of the anchor side (near-attack conformation).
        anchor_mol: Molecule topology for the anchor side.
        target_xyz: XYZ of the opposite side (near-attack conformation).
        weight: Interpolation weight (0 = anchor, 1 = target).
        reactive_xyz_indices: Atom indices participating in reactive bonds.
        anchors: Preferred anchor atoms for the Z-matrix.
        constraints: ``R_atom`` constraints for reactive bonds.
        r_mol: Reactant Molecule (used for postprocessing bond topology).
        forming_bonds: Forming-bond index pairs.
        breaking_bonds: Breaking-bond index pairs.
        label: Logging label for debug messages.
        skip_postprocess: If ``True``, skip the family-specific postprocessing
            pipeline. Useful for fallback paths where nearly all atoms are
            reactive and H repairs would perturb the geometry.
        family: RMG reaction family name, used to dispatch to the appropriate
            post-processing and validation handlers.
        r_label_map: Family-labeled atom map from the product_dict.
        redistribute_ch2: If ``True``, run an additional CH₂ redistribution
            pass after postprocessing to fix collapsed CH₂ groups adjacent
            to reactive atoms.

    Returns:
        A validated XYZ dict, or ``None`` if any step fails or the guess is rejected by validation.
    """
    zmat = build_zmat_with_retry(anchor_xyz, anchor_mol, anchors, constraints, label=label)
    if zmat is None:
        return None
    target_zmat = update_zmat_by_xyz(zmat=zmat, xyz=target_xyz)
    if target_zmat is None:
        logger.debug(f'Linear ({label}): update_zmat_by_xyz returned None.')
        return None
    ts_zmat = average_zmat_params(zmat_1=zmat, zmat_2=target_zmat,
                                  weight=weight, reactive_xyz_indices=reactive_xyz_indices)
    if ts_zmat is None:
        logger.debug(f'Linear ({label}): average_zmat_params returned None.')
        return None
    ts_xyz = zmat_to_xyz(ts_zmat)
    if ts_xyz is None:
        logger.debug(f'Linear ({label}): zmat_to_xyz returned None.')
        return None
    if skip_postprocess:
        migrating_hs: set[int] = set()
    else:
        ts_xyz, migrating_hs = postprocess_ts_guess(ts_xyz, r_mol, forming_bonds, breaking_bonds,
                                                    family=family, r_label_map=r_label_map)
    if redistribute_ch2:
        ts_xyz = fix_crowded_h_atoms(ts_xyz, r_mol, skip_h_indices=migrating_hs, redistribute_ch2=True)
    is_valid, _ = validate_ts_guess(ts_xyz, migrating_hs, forming_bonds, r_mol, label=label, family=family)
    return ts_xyz if is_valid else None
