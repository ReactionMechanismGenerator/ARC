"""
TS Split
"""

from typing import Dict, List, Tuple, Union
import numpy as np
from collections import deque

from arc.common import SINGLE_BOND_LENGTH
from arc.species.converter import xyz_to_dmat, translate_to_center_of_mass


MAX_LENGTH = 3.0


def get_group_xyzs_and_key_indices_from_ts(xyz: dict,
                                           a: int,
                                           b: int,
                                           h: int,
                                           ) -> Tuple[dict, dict, dict]:
    """
    Get the two corresponding XYZs of groups in an H abstraction TS based on the H atom being abstracted.

    Args:
        xyz (dict): The TS xyz.
        a (int): The index of one of the heavy atoms H is connected to.
        b (int): The index of the other heavy atoms H is connected to.
        h (int): The index of the H atom being abstract

    Returns:
        Tuple[dict, dict, dict]:
          - xyz of group 1
          - xyz of group 2
          - Keys are 'g1_a', 'g1_h', 'g2_a', 'g2_h', values are atom indices i the returned group xyzs.
    """
    g1, g2 = divide_h_abs_ts_int_groups(xyz, a, b, h)
    g1_xyz, g1_map = split_xyz_by_indices(xyz=xyz, indices=g1)
    g2_xyz, g2_map = split_xyz_by_indices(xyz=xyz, indices=g2)
    index_dict = {'g1_a': g1_map[a], 'g1_h': g1_map[h], 'g2_a': g2_map[b], 'g2_h': g2_map[h]}
    return g1_xyz, g2_xyz, index_dict


def split_xyz_by_indices(xyz: dict,
                         indices: List[int],
                         ) -> Tuple[dict, Dict[int, int]]:
    """
    Split an XYZ dictionary by indices.
    Also, map the indices in to_map_indices to the new indices in the returned XYZ.

    Args:
        xyz (dict): The XYZ dictionary.
        indices (List[int]): The indices to split by.

    Returns:
        Tuple[dict, dict[int, int]]:
          - The split XYZ dictionary.
          - The new indices of the atoms in to_map_indices in the returned XYZ. Keys are the original indices.
    """
    new_xyz = dict()
    new_xyz['symbols'] = tuple(symbol for i, symbol in enumerate(xyz['symbols']) if i in indices)
    new_xyz['isotopes'] = tuple(isotope for i, isotope in enumerate(xyz['isotopes']) if i in indices)
    new_xyz['coords'] = tuple(coord for i, coord in enumerate(xyz['coords']) if i in indices)
    mapped_index = 0
    map_ = dict()
    for i in range(len(xyz['symbols'])):
        if i in indices:
            map_[i] = mapped_index
            mapped_index += 1
    new_xyz = translate_to_center_of_mass(new_xyz)
    return new_xyz, map_


def divide_h_abs_ts_int_groups(xyz: dict,
                               a: int,
                               b: int,
                               h: int,
                               ) -> Tuple[List[int], List[int]]:
    """
    Divide the atoms in the TS into two groups based on the H atom being abstracted.
    Get the indices of the atoms in the two groups, each includes the abstracted H (so R1H, R2H).
    """
    dmat = xyz_to_dmat(xyz)
    symbols = xyz['symbols']
    adjlist = get_adjlist_from_dmat(dmat, symbols, h, a, b)
    g1 = iterative_dfs(adjlist, start=a, border=h)
    g2 = iterative_dfs(adjlist, start=b, border=h)
    return g1, g2


def get_adjlist_from_dmat(dmat: Union[np.ndarray, list],
                          symbols: Tuple[str, ...],
                          h: int,
                          a: int,
                          b: int,
                          ) -> Dict[int, List[int]]:
    """
    Get an adjacency list from a DMat.

    Args:
        dmat (np.ndarray): The distance matrix.
        symbols (Tuple[str]): THe chemical elements.
        h (int): The index of the H atom being abstracted (all indices are 0-indexed)
        a (int): The index of one of the heavy atoms H is connected to.
        b (int): The index of the other heavy atoms H is connected to.

    Returns:
        Dict[int, List[int]]: The adjlist.
    """
    adjlist = dict()
    for atom_1 in range(len(symbols)):
        if atom_1 == h:
            continue
        for atom_2 in range(len(symbols)):
            if atom_2 in [h, atom_1]:
                continue
            if dmat[atom_1][atom_2] <= MAX_LENGTH:

                if bonded(dmat[atom_1][atom_2], symbols[atom_1], symbols[atom_2]):
                    if atom_1 not in adjlist:
                        adjlist[atom_1] = list()
                    adjlist[atom_1].append(atom_2)
    adjlist[h] = [a, b]
    adjlist[a].append(h)
    adjlist[b].append(h)
    return adjlist


def bonded(distance: float, s1: str, s2: str) -> bool:
    """
    Determine whether two atoms are bonded based on their distance and chemical symbols.

    Args:
        distance (float): The distance between the atoms.
        s1 (str): The chemical symbol of the first atom.
        s2 (str): The chemical symbol of the second atom.

    Returns:
        bool: Whether the atoms are bonded.
    """
    bond_key = f'{s1}_{s2}'
    ref_dist = SINGLE_BOND_LENGTH.get(bond_key, None) or SINGLE_BOND_LENGTH.get(f'{s2}_{s1}', None)
    if ref_dist is None:
        return False
    if distance <= ref_dist * 1.3:  # todo: test & magic number, make CONSTANT
        return True
    return False


def iterative_dfs(adjlist: Dict[int, List[int]],
                  start: int,
                  border: int,
                  ) -> List[int]:
    """
    A depth first search (DFS) graph traversal algorithm to determine indices that belong to a subgroup of the graph.
    The subgroup is being explored from the key atom and will not pass the border atom.
    This is an iterative and not a recursive algorithm since Python doesn't have a great support for recursion
    since it lacks Tail Recursion Elimination and because there is a limit of recursion stack depth (by default is 1000).

    Args:
        adjlist (Dict[int, List[int]]): The adjacency list.
        start (int): The index of the atom to start the DFS from.
        border (int): The index of the atom that is the border of the subgroup.

    Returns:
        List[int]: The indices of atoms in the subgroup including the border atom.
    """
    visited = [border]
    stack = deque()
    stack.append(start)
    while stack:
        key = stack.pop()
        if key in visited:
            continue
        visited.append(key)
        neighbors = adjlist[key]
        for neighbor in neighbors:
            if neighbor not in visited:
                stack.append(neighbor)
    return visited
