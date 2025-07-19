"""
This is the engine part of the atom mapping module.
Here, the core function for calculation the atom map are located.
Strategy:
    1) The wrapper function map_rxn is called by the driver.
    2) The wrapper calls the relevant functions, in order. The algorithm is specified on each of the functions.
    3) The atom map is returned to the driver.
"""

import numpy as np
from collections import deque
from itertools import product
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from arc.common import convert_list_index_0_to_1, extremum_list, get_angle_in_180_range, logger, signed_angular_diff
from arc.exceptions import AtomTypeError, ConformerError, InputError, SpeciesError
from arc.family import ReactionFamily
from arc.molecule import Molecule
from arc.molecule.resonance import generate_resonance_structures_safely
from arc.species import ARCSpecies
from arc.species.conformers import determine_chirality
from arc.species.converter import compare_confs, sort_xyz_using_indices, xyz_from_data
from arc.species.vectors import calculate_dihedral_angle, get_delta_angle

if TYPE_CHECKING:
    from arc.molecule.molecule import Atom
    from arc.reaction import ARCReaction


RESERVED_FINGERPRINT_KEYS = ['self', 'chirality', 'label']


def map_two_species(spc_1: Union[ARCSpecies, Molecule],
                    spc_2: Union[ARCSpecies, Molecule],
                    map_type: str = 'list',
                    backend: str = 'ARC',
                    consider_chirality: bool = True,
                    inc_vals: Optional[int] = None,
                    verbose: bool = False,
                    ) -> Optional[Union[List[int], Dict[int, int]]]:
    """
    Map the atoms in spc_1 to the atoms in spc_2.
    All indices are 0-indexed.
    If a dict type atom map is returned, it could conveniently be used to map ``spc_2`` -> ``spc_1`` by doing::
        ordered_spc1.atoms = [spc_2.atoms[atom_map[i]] for i in range(len(spc_2.atoms))]

    Args:
        spc_1 (Union[ARCSpecies, Molecule]): Species 1.
        spc_2 (Union[ARCSpecies, Molecule]): Species 2.
        map_type (str, optional): Whether to return a 'list' or a 'dict' map type.
        backend (str, optional): Currently only ``ARC``'s method is implemented as the backend.
        consider_chirality (bool, optional): Whether to consider chirality when fingerprinting.
        inc_vals (int, optional): An optional integer by which all values in the atom map list will be incremented.
        verbose (bool, optional): Whether to use logging.

    Returns:
        Optional[Union[List[int], Dict[int, int]]]:
            The atom map. By default, a list is returned.
            If the map is of ``list`` type, entry indices are atom indices of ``spc_1``, entry values are atom indices of ``spc_2``.
            If the map is of ``dict`` type, keys are atom indices of ``spc_1``, values are atom indices of ``spc_2``.
    """
    spc_1, spc_2 = get_arc_species(spc_1), get_arc_species(spc_2)
    if not check_species_before_mapping(spc_1, spc_2, verbose=verbose):
        if verbose:
            logger.warning(f'Could not map species {spc_1} and {spc_2}.')
        return None

    # A shortcut for mono-atomic species.
    if spc_1.number_of_atoms == spc_2.number_of_atoms == 1:
        if map_type == 'dict':
            return {0: 0}
        return [0]

    # A shortcut for homonuclear diatomic species.
    if spc_1.number_of_atoms == spc_2.number_of_atoms == 2 \
            and len(set([atom.element.symbol for atom in spc_1.mol.atoms])) == 1:
        if map_type == 'dict':
            return {0: 0, 1: 1}
        return [0, 1]

    # A shortcut for species with all different elements:
    if len(set([atom.element.symbol for atom in spc_1.mol.atoms])) == spc_1.number_of_atoms:
        atom_map = {}
        for i, atom_1 in enumerate(spc_1.mol.atoms):
            for j, atom_2 in enumerate(spc_2.mol.atoms):
                if atom_1.element.symbol == atom_2.element.symbol:
                    atom_map[i] = j
                    break
        if map_type == 'list':
            atom_map = [v for k, v in sorted(atom_map.items(), key=lambda item: item[0])]
        return atom_map

    if backend.lower() not in ['arc']:
        raise ValueError(f'The backend {backend} is not supported for 3DAM.')
    atom_map = None

    if backend.lower() == 'arc':
        fingerprint_1 = fingerprint(spc_1, consider_chirality=consider_chirality)
        fingerprint_2 = fingerprint(spc_2, consider_chirality=consider_chirality)
        candidates = identify_superimposable_candidates(fingerprint_1, fingerprint_2)
        if candidates is None or len(candidates) == 0:
            consider_chirality = not consider_chirality
            fingerprint_1 = fingerprint(spc_1, consider_chirality=consider_chirality)
            fingerprint_2 = fingerprint(spc_2, consider_chirality=consider_chirality)
            candidates = identify_superimposable_candidates(fingerprint_1, fingerprint_2)
            if candidates is None or len(candidates) == 0:
                logger.warning(f'Could not identify superimposable candidates {spc_1} and {spc_2}.')
                return None
        if not len(candidates):
                return None
        else:
            rmsds, fixed_spcs = list(), list()
            candidate = None
            for candidate in candidates:
                fixed_spc_1, fixed_spc_2 = fix_dihedrals_by_backbone_mapping(spc_1, spc_2, backbone_map=candidate)
                fixed_spcs.append((fixed_spc_1, fixed_spc_2))
                backbone_1, backbone_2 = set(list(candidate.keys())), set(list(candidate.values()))
                xyz1, xyz2 = fixed_spc_1.get_xyz(), fixed_spc_2.get_xyz()
                xyz1 = xyz_from_data(coords=[xyz1['coords'][i] for i in range(fixed_spc_1.number_of_atoms) if i in backbone_1],
                                     symbols=[xyz1['symbols'][i] for i in range(fixed_spc_1.number_of_atoms) if i in backbone_1],
                                     isotopes=[xyz1['isotopes'][i] for i in range(fixed_spc_1.number_of_atoms) if i in backbone_1])
                xyz2 = xyz_from_data(coords=[xyz2['coords'][i] for i in range(fixed_spc_2.number_of_atoms) if i in backbone_2],
                                     symbols=[xyz2['symbols'][i] for i in range(fixed_spc_2.number_of_atoms) if i in backbone_2],
                                     isotopes=[xyz2['isotopes'][i] for i in range(fixed_spc_2.number_of_atoms) if i in backbone_2])
                no_gap_candidate = remove_gaps_from_values(candidate)
                xyz2 = sort_xyz_using_indices(xyz2, indices=[v for k, v in sorted(no_gap_candidate.items(),
                                                                                  key=lambda item: item[0])])
                rmsds.append(compare_confs(xyz1=xyz1, xyz2=xyz2, rmsd_score=True))
            chosen_candidate_index = rmsds.index(min(rmsds))
            fixed_spc_1, fixed_spc_2 = fixed_spcs[chosen_candidate_index]
            if candidate is not None:
                atom_map = map_hydrogens(fixed_spc_1, fixed_spc_2, candidate)
                if map_type == 'list':
                    atom_map = [v for k, v in sorted(atom_map.items(), key=lambda item: item[0])]

    if inc_vals is not None:
        atom_map = [value + inc_vals for value in atom_map]
    return atom_map


def get_arc_species(spc: Union[ARCSpecies, Molecule]) -> ARCSpecies:
    """
    Convert an object to an ARCSpecies object.

    Args:
        spc (Union[ARCSpecies, Molecule]): An input object.

    Returns:
        ARCSpecies: The corresponding ARCSpecies object.
    """
    if isinstance(spc, ARCSpecies):
        return spc
    if isinstance(spc, Molecule):
        return ARCSpecies(label='S', mol=spc)
    raise ValueError(f'Species entries may only be ARCSpecies or RMG Molecule.\nGot {spc} which is a {type(spc)}.')


def check_species_before_mapping(spc_1: ARCSpecies,
                                 spc_2: ARCSpecies,
                                 verbose: bool = False,
                                 ) -> bool:
    """
    Perform general checks before mapping two species.

    Args:
        spc_1 (ARCSpecies): Species 1.
        spc_2 (ARCSpecies): Species 2.
        verbose (bool, optional): Whether to use logging.

    Returns:
        bool: ``True`` if all checks passed, ``False`` otherwise.
    """
    if spc_1.mol is None or spc_2.mol is None:
        return False
    if spc_1.mol.fingerprint != spc_2.mol.fingerprint:
        raise ValueError(f'The two species sent for mapping have different molecular formula. Got:\n{spc_1.mol.to_smiles()}\n{spc_2.mol.to_smiles()}')
    # Check number of atoms > 0.
    if spc_1.number_of_atoms == 0 or spc_2.number_of_atoms == 0:
        if verbose:
            logger.warning(f'The number of atoms must be larger than 0, '
                           f'got {spc_1.number_of_atoms} and {spc_2.number_of_atoms}.')
        return False
    # Check the same number of atoms.
    if spc_1.number_of_atoms != spc_2.number_of_atoms:
        if verbose:
            logger.warning(f'The number of atoms must be identical between the two species, '
                           f'got {spc_1.number_of_atoms} and {spc_2.number_of_atoms}.')
        return False
    # Check the same number of each element.
    element_dict_1, element_dict_2 = dict(), dict()
    for atom in spc_1.mol.vertices:
        element_dict_1[atom.element.symbol] = element_dict_1.get(atom.element.symbol, 0) + 1
    for atom in spc_2.mol.vertices:
        element_dict_2[atom.element.symbol] = element_dict_2.get(atom.element.symbol, 0) + 1
    for key, val in element_dict_1.items():
        if val != element_dict_2[key]:
            if verbose:
                logger.warning(f'The chemical formula of the two species is not identical, got the following elements:\n'
                               f'{element_dict_1}\n{element_dict_2}')
            return False
    # Check the same number of bonds between similar elements (ignore bond order).
    bonds_dict_1, bonds_dict_2 = get_bonds_dict(spc_1), get_bonds_dict(spc_2)
    for key, val in bonds_dict_1.items():
        if key not in bonds_dict_2.keys() or val != bonds_dict_2[key]:
            if verbose:
                logger.warning(f'The chemical bonds in the two species are not identical, got the following bonds:\n'
                               f'{bonds_dict_1}\n{bonds_dict_2}')
            return False
    # Check whether both species are cyclic.
    if spc_1.mol.is_cyclic() != spc_2.mol.is_cyclic():
        if verbose:
            logger.warning(f'Both species should be either linear or non-linear, got:\n'
                           f'linear = {spc_1.mol.is_linear()} and linear = {spc_2.mol.is_linear()}.')
        return False
    return True


def get_bonds_dict(spc: ARCSpecies) -> Dict[str, int]:
    """
    Get a dictionary of bonds by elements in the species ignoring bond orders.

    Args:
        spc (ARCSpecies): The species to examine.

    Returns:
        Dict[str, int]: Keys are 'A-B' strings of element symbols sorted alphabetically (e.g., 'C-H' or 'H-O'),
                        values are the number of such bonds (ignoring bond order).
    """
    bond_dict = dict()
    bonds = spc.mol.get_all_edges()
    for bond in bonds:
        key = '-'.join(sorted([bond.atom1.element.symbol, bond.atom2.element.symbol]))
        bond_dict[key] = bond_dict.get(key, 0) + 1
    return bond_dict


def fingerprint(spc: ARCSpecies,
                consider_chirality: bool = True
                ) -> Dict[int, Dict[str, Union[str, List[int]]]]:
    """
    Determine the species fingerprint.
    For any heavy atom in the ``spc`` its element (``self``) will be determined,
    the element types and numbers of adjacent atoms are determined,
    and any chirality information will be determined if relevant.

    Args:
        spc (ARCSpecies): The input species.
        consider_chirality (bool, optional): Whether to consider chirality when fingerprinting.

    Returns:
        Dict[int, Dict[str, List[int]]]: Keys are indices of heavy atoms, values are dicts. keys are element symbols,
                                         values are indices of adjacent atoms corresponding to this element.
    """
    fingerprint_dict = dict()
    chirality_dict = determine_chirality(conformers=[{'xyz': spc.get_xyz()}],
                                         label=spc.label,
                                         mol=spc.mol)[0]['chirality'] if consider_chirality else {}
    for i, atom_1 in enumerate(spc.mol.atoms):
        if atom_1.is_non_hydrogen():
            atom_fingerprint = {'self': atom_1.element.symbol}
            if consider_chirality:
                for key, val in chirality_dict.items():
                    if i in key:
                        atom_fingerprint['chirality'] = val
            for atom_2 in atom_1.edges.keys():
                if atom_2.element.symbol not in atom_fingerprint .keys():
                    atom_fingerprint[atom_2.element.symbol] = list()
                atom_fingerprint[atom_2.element.symbol] = sorted(atom_fingerprint[atom_2.element.symbol]
                                                                 + [spc.mol.atoms.index(atom_2)])
            fingerprint_dict[i] = atom_fingerprint
    return fingerprint_dict


def identify_superimposable_candidates(fingerprint_1: Dict[int, Dict[str, Union[str, List[int]]]],
                                       fingerprint_2: Dict[int, Dict[str, Union[str, List[int]]]],
                                       ) -> List[Dict[int, int]]:
    """
    Identify candidate ordering of heavy atoms (only) that could potentially be superimposed.

    Args:
        fingerprint_1 (Dict[int, Dict[str, Union[str, List[int]]]]): Adjacent element dict for species 1.
        fingerprint_2 (Dict[int, Dict[str, Union[str, List[int]]]]): Adjacent element dict for species 2.

    Returns:
        List[Dict[int, int]]: Entries are superimposable candidate dicts. Keys are atom indices of heavy atoms
                              of species 1, values are potentially mapped atom indices of species 2.
    """
    candidates = list()
    for key_1 in fingerprint_1.keys():
        for key_2 in fingerprint_2.keys():
            # Try all combinations of heavy atoms.
            result = iterative_dfs(fingerprint_1, fingerprint_2, key_1, key_2)
            if result is not None:
                candidates.append(result)
    return prune_identical_dicts(candidates)


def are_adj_elements_in_agreement(fingerprint_1: Dict[str, Union[str, List[int]]],
                                  fingerprint_2: Dict[str, Union[str, List[int]]],
                                  ) -> bool:
    """
    Check whether two dictionaries representing adjacent elements are in agreement
    w.r.t. the type and number of adjacent elements.
    Also checks the identity of the parent ("self") element.

    Args:
          fingerprint_1 (Dict[str, List[int]]): Adjacent elements dictionary 1.
          fingerprint_2 (Dict[str, List[int]]): Adjacent elements dictionary 2.

    Returns:
        bool: ``True`` if the two dicts represent identical adjacent elements, ``False`` otherwise.
    """
    if len(fingerprint_1) != len(fingerprint_2):
        return False
    for token in RESERVED_FINGERPRINT_KEYS:
        if token in fingerprint_1 and token in fingerprint_2 and fingerprint_1[token] != fingerprint_2[token]:
            return False
    for key, val in fingerprint_1.items():
        if key not in RESERVED_FINGERPRINT_KEYS and (key not in fingerprint_2 or len(val) != len(fingerprint_2[key])):
            return False
    return True



def iterative_dfs(fingerprint_1: Dict[int, Dict[str, List[int]]],
                  fingerprint_2: Dict[int, Dict[str, List[int]]],
                  key_1: int,
                  key_2: int,
                  allow_first_key_pair_to_disagree: bool = False,
                  ) -> Optional[Dict[int, int]]:
    """
    A depth first search (DFS) graph traversal algorithm to determine possible superimposable ordering of heavy atoms.
    This is an iterative and not a recursive algorithm since Python doesn't have a great support for recursion
    since it lacks Tail Recursion Elimination and because there is a limit of recursion stack depth (by default is 1000).

    Args:
        fingerprint_1 (Dict[int, Dict[str, List[int]]]): Adjacent elements dictionary 1 (graph 1).
        fingerprint_2 (Dict[int, Dict[str, List[int]]]): Adjacent elements dictionary 2 (graph 2).
        key_1 (int): The starting index for graph 1.
        key_2 (int): The starting index for graph 2.
        allow_first_key_pair_to_disagree (bool, optional): ``True`` to not enforce agreement between the fingerprint
                                                           of ``key_1`` and ``key_2``.

    Returns:
        Optional[Dict[int, int]]: ``None`` if this is an invalid superimposable candidate. Keys are atom indices of
                                  heavy atoms of species 1, values are potentially mapped atom indices of species 2.
    """
    visited_1, visited_2 = list(), list()
    stack_1, stack_2 = deque(), deque()
    stack_1.append(key_1)
    stack_2.append(key_2)
    result: Dict[int, int] = dict()
    while stack_1 and stack_2:
        current_key_1 = stack_1.pop()
        current_key_2 = stack_2.pop()
        if current_key_1 in visited_1 or current_key_2 in visited_2:
            continue
        if not are_adj_elements_in_agreement(fingerprint_1[current_key_1], fingerprint_2[current_key_2]) \
                and not (allow_first_key_pair_to_disagree and len(result) == 0):
            continue
        visited_1.append(current_key_1)
        visited_2.append(current_key_2)
        result[current_key_1] = current_key_2
        for symbol in fingerprint_1[current_key_1].keys():
            if symbol not in RESERVED_FINGERPRINT_KEYS + ['H']:
                for combination_tuple in product(fingerprint_1[current_key_1][symbol], fingerprint_2[current_key_2][symbol]):
                    if combination_tuple[0] not in visited_1 and combination_tuple[1] not in visited_2:
                        stack_1.append(combination_tuple[0])
                        stack_2.append(combination_tuple[1])
    if len(result) != len(fingerprint_1):
        return None
    return result


def prune_identical_dicts(dicts_list: List[dict]) -> List[dict]:
    """
    Return a list of unique dictionaries.

    Args:
        dicts_list (List[dict]): A list of dicts to prune.

    Returns:
        List[dict]: A list of unique dicts.
    """
    new_dicts_list = list()
    for new_dict in dicts_list:
        unique_ = True
        for existing_dict in new_dicts_list:
            if unique_:
                for new_key, new_val in new_dict.items():
                    if new_key not in existing_dict.keys() or new_val == existing_dict[new_key]:
                        unique_ = False
                        break
        if unique_:
            new_dicts_list.append(new_dict)
    return new_dicts_list


def remove_gaps_from_values(data: Dict[int, int]) -> Dict[int, int]:
    """
    Return a dictionary of integer keys and values with consecutive values starting at 0.

    Args:
        data (Dict[int, int]): A dictionary of integers as keys and values.

    Returns:
        Dict[int, int]: A dictionary of integers as keys and values with consecutive values starting at 0.
    """
    new_data = dict()
    val = 0
    for key, _ in sorted(data.items(), key=lambda item: item[1]):
        new_data[key] = val
        val += 1
    return new_data


def fix_dihedrals_by_backbone_mapping(spc_1: ARCSpecies,
                                      spc_2: ARCSpecies,
                                      backbone_map: Dict[int, int],
                                      ) -> Tuple[ARCSpecies, ARCSpecies]:
    """
    Fix the dihedral angles of two mapped species to align them.

    Args:
        spc_1 (ARCSpecies): Species 1.
        spc_2 (ARCSpecies): Species 2.
        backbone_map (Dict[int, int]): The backbone map.

    Returns:
        Tuple[ARCSpecies, ARCSpecies]: The corresponding species with aligned dihedral angles.
    """
    if not spc_1.rotors_dict or not spc_2.rotors_dict:
        spc_1.determine_rotors()
        spc_2.determine_rotors()
    spc_1_copy, spc_2_copy = spc_1.copy(), spc_2.copy()
    torsions = get_backbone_dihedral_angles(spc_1, spc_2, backbone_map)
    deviations = [get_backbone_dihedral_deviation_score(spc_1, spc_2, backbone_map, torsions=torsions)]
    # Loop while the deviation improves by more than 1 degree:
    while len(torsions) and (len(deviations) < 2 or deviations[-2] - deviations[-1] > 1):
        for torsion_dict in torsions:
            angle = 0.5 * sum([torsion_dict['angle 1'], torsion_dict['angle 2']])
            spc_1_copy.set_dihedral(scan=convert_list_index_0_to_1(torsion_dict['torsion 1']),
                                    deg_abs=angle, count=False, chk_rotor_list=False, xyz=spc_1_copy.get_xyz())
            spc_2_copy.set_dihedral(scan=convert_list_index_0_to_1(torsion_dict['torsion 2']),
                                    deg_abs=angle, count=False, chk_rotor_list=False, xyz=spc_2_copy.get_xyz())
            spc_1_copy.final_xyz, spc_2_copy.final_xyz = spc_1_copy.initial_xyz, spc_2_copy.initial_xyz
        torsions = get_backbone_dihedral_angles(spc_1_copy, spc_2_copy, backbone_map)
        deviations.append(get_backbone_dihedral_deviation_score(spc_1_copy, spc_2_copy, backbone_map, torsions=torsions))
    return spc_1_copy, spc_2_copy


def get_backbone_dihedral_deviation_score(spc_1: ARCSpecies,
                                          spc_2: ARCSpecies,
                                          backbone_map: Dict[int, int],
                                          torsions: Optional[List[Dict[str, Union[float, List[int]]]]] = None
                                          ) -> float:
    """
    Determine a deviation score for dihedral angles of torsions.
    We don't consider here "terminal" torsions, just pure backbone torsions.

    Args:
        spc_1 (ARCSpecies): Species 1.
        spc_2 (ARCSpecies): Species 2.
        backbone_map (Dict[int, int]): The backbone map.
        torsions (Optional[List[Dict[str, Union[float, List[int]]]]], optional): The backbone dihedral angles.

    Returns:
        Float: The dihedral deviation score.
    """
    torsions = torsions or get_backbone_dihedral_angles(spc_1, spc_2, backbone_map)
    return sum([abs(torsion_dict['angle 1'] - torsion_dict['angle 2']) for torsion_dict in torsions])


def get_backbone_dihedral_angles(spc_1: ARCSpecies,
                                 spc_2: ARCSpecies,
                                 backbone_map: Dict[int, int],
                                 ) -> List[Dict[str, Union[float, List[int]]]]:
    """
    Determine the dihedral angles of the backbone torsions of two backbone mapped species.
    The output has the following format::

        torsions = [{'torsion 1': [0, 1, 2, 3],  # The first torsion in terms of species 1's indices.
                     'torsion 2': [5, 7, 2, 4],  # The first torsion in terms of species 2's indices.
                     'angle 1': 60.0,  # The corresponding dihedral angle to 'torsion 1'.
                     'angle 2': 125.1,  # The corresponding dihedral angle to 'torsion 2'.
                    },
                    {}, ...  # The second torsion, and so on.
                   ]

    Args:
        spc_1 (ARCSpecies): Species 1.
        spc_2 (ARCSpecies): Species 2.
        backbone_map (Dict[int, int]): The backbone map.

    Returns:
        List[Dict[str, Union[float, List[int]]]]: The corresponding species with aligned dihedral angles.
    """
    torsions = list()
    if not spc_1.rotors_dict or not spc_2.rotors_dict:
        spc_1.determine_rotors()
        spc_2.determine_rotors()
    if spc_1.rotors_dict is not None and spc_2.rotors_dict is not None:
        for rotor_dict_1 in spc_1.rotors_dict.values():
            torsion_1 = rotor_dict_1['torsion']
            if spc_1.mol.atoms[torsion_1[0]].is_non_hydrogen() \
                    and spc_1.mol.atoms[torsion_1[3]].is_non_hydrogen():
                # This is not a "terminal" torsion.
                for rotor_dict_2 in spc_2.rotors_dict.values():
                    torsion_2 = [backbone_map[t_1] for t_1 in torsion_1]
                    if all(pivot_2 in [torsion_2[1], torsion_2[2]]
                           for pivot_2 in [rotor_dict_2['torsion'][1], rotor_dict_2['torsion'][2]]):
                        torsions.append({'torsion 1': torsion_1,
                                         'torsion 2': torsion_2,
                                         'angle 1': calculate_dihedral_angle(coords=spc_1.get_xyz(), torsion=torsion_1),
                                         'angle 2': calculate_dihedral_angle(coords=spc_2.get_xyz(), torsion=torsion_2)})
    return torsions


def map_lists(list_1: List[float],
              list_2: List[float],
              ) -> Dict[int, int]:
    """
    Map two lists with equal lengths of floats by proximity of entries.
    Assuming no two entries in a list are identical.
    Note: values are treated as cyclic in a 0-360 range (i.e., 1 is closer to 259 than to 5)!

    Args:
        list_1 (List[float]): List 1.
        list_2 (List[float]): List 2.

    Returns:
        Dict[int, int]: The lists map. Keys correspond to entry indices in ``list_1``,
                        Values correspond to respective entry indices in ``list_2``.
    """
    if len(list_1) != len(list_2):
        raise ValueError(f'Lists must be of the same length, got:\n{list_1}\n{list_2}\n'
                         f'with lengths {len(list_1)} and {len(list_2)}.')
    list_map = dict()
    dict_1, dict_2 = {k: True for k in list_1}, {k: True for k in list_2}
    while any(dict_1.values()):
        min_1_index = list_1.index(min([k for k, v in dict_1.items() if v]))
        angle_deltas = [get_delta_angle(list_1[min_1_index], angle_2) if dict_2[angle_2] else None for angle_2 in list_2]
        min_2_index = angle_deltas.index(extremum_list(angle_deltas, return_min=True))
        dict_1[list_1[min_1_index]], dict_2[list_2[min_2_index]] = False, False
        list_map[min_1_index] = min_2_index
    return list_map


def _find_hydrogen_neighbors(heavy_index: int, atoms: List['Atom']) -> List[int]:
    """
    Return the indices of all hydrogen atoms bonded to the specified heavy atom.

    Args:
        heavy_index (int): Index of the heavy atom in the atoms list.
        atoms (List[Atom]): List of Atom objects comprising the molecule.

    Returns:
        List[int]: Sorted list of indices of hydrogen atoms bonded to the heavy atom.

    Raises:
        IndexError: If heavy_index is out of range.
    """
    if heavy_index < 0 or heavy_index >= len(atoms):
        raise IndexError(f"heavy_index {heavy_index} is out of range for atoms list of length {len(atoms)}")
    heavy = atoms[heavy_index]
    neighbors = [i for i, atom in enumerate(atoms) if atom.is_hydrogen() and heavy in atom.bonds]
    return sorted(neighbors)


def _find_preferably_heavy_neighbor(heavy_index: int,
                                    spc: ARCSpecies,
                                    partial_atom_map: Optional[Dict[int, int]] = None,
                                    exclude_index: Optional[int] = None,
                                    ) -> Optional[int]:
    """
    Return the indices of a non-hydrogen atoms bonded to the specified heavy atom.
    If no heavy atom neighbors are found, it will return a hydrogen neighbor.
    Anyway, the returned atom is verified to be in the partial_atom_map.

    Args:
        heavy_index (int): Index of the heavy atom in the atoms list.
        spc (ARCSpecies): ARCSpecies object containing the molecule.
        exclude_index (Optional[int]): Index of an atom to exclude from the result.
        partial_atom_map (Optional[Dict[int, int]]): A mapping of atom indices to indices in the product species.

    Returns:
        Optional[int]: An integer atom index, or None if no neighbor qualifies.

    Raises:
        IndexError: If heavy_index is out of range.
    """
    atoms = spc.mol.atoms
    if heavy_index < 0 or heavy_index >= len(atoms):
        raise IndexError(f"heavy_index {heavy_index} is out of range for atoms list of length {len(atoms)}")
    partial_atom_map = partial_atom_map or {}

    # 1) Try rotors_dict first
    if spc.rotors_dict:
        for rotor_dict in spc.rotors_dict.values():
            if heavy_index == rotor_dict['torsion'][1] and exclude_index != rotor_dict['torsion'][2]:
                return rotor_dict['torsion'][2]
            elif heavy_index == rotor_dict['torsion'][2] and exclude_index != rotor_dict['torsion'][1]:
                return rotor_dict['torsion'][1]

    # 2) Fall back to any non-H neighbor
    neighbors = [i for i, atom in enumerate(atoms) if not atom.is_hydrogen()
                 and atoms[heavy_index] in atom.bonds and i != exclude_index]
    if not neighbors:
        # 3) If no non-H neighbors, try to find a hydrogen neighbor
        neighbors = [i for i, atom in enumerate(atoms) if atom.is_hydrogen()
                     and atoms[heavy_index] in atom.bonds and i != exclude_index and i in partial_atom_map]
    if len(neighbors):
        for neighbor in neighbors:
            if neighbor in partial_atom_map:
                return neighbor
    return None


def _select_ch3_anchors(heavy_index: int,
                        spc: ARCSpecies,
                        backbone_map: Dict[int, int],
                        ) -> Tuple[Optional[int], Optional[int]]:
    """
    Select two anchors A and B for a XH3 center X (usually X is C) so that:
      - A is the mapped heavy neighbor (C → A)
      - B is a second anchor forming a non-linear angle B-A-X

    Strategy:
      1) Identify the one heavy neighbor A of X.
      2) B should preferably be a heavy atom, but can also be an H.
      3) Fallback to any other heavy atom B ≠ X, A that gives a non‐linear angle.
      4) Fallback to any hydrogen already present in backbone_map.
      5) Fallback to any hydrogen.

    Args:
        heavy_index: index of CH3 carbon in spc.mol.atoms
        spc: ARCSpecies with rotors_dict and xyz
        backbone_map: heavy-atom map C→C' in product

    Returns:
        Tuple(A_index, B_index) or None, None if no valid anchors.
    """
    A_index = _find_preferably_heavy_neighbor(heavy_index=heavy_index, spc=spc, partial_atom_map=backbone_map)
    if A_index is None:
        return None, None
    B_index = _find_preferably_heavy_neighbor(heavy_index=A_index, spc=spc, partial_atom_map=backbone_map, exclude_index=heavy_index)
    if B_index is None:
        for atom in spc.mol.atoms[A_index].edges:
            atom_index = spc.mol.atoms.index(atom)
            if atom_index != heavy_index:
                B_index = atom_index
                break
    if B_index is None:
        return None, None
    return A_index, B_index


def _construct_local_axes(center_idx: int,
                          anchor1_idx: int,
                          anchor2_idx: int,
                          coords: List[Tuple[float, float, float]]
                          ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Construct a right-handed local coordinate frame at a central atom C.

    The x-axis is defined along the vector from C to anchor1 (A).
    The y-axis lies in the plane defined by atoms (B, C, A), orthogonal to x.
    The z-axis is orthonormal to the plane (x × y).

    Args:
        center_idx (int): Index of the central atom C.
        anchor1_idx (int): Index of the primary anchor A.
        anchor2_idx (int): Index of the secondary anchor B.
        coords (List[Tuple[float, float, float]]): List of 3D coordinates for all atoms.

    Returns: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Tuple of three numpy arrays (e_x, e_y, e_z), each shape (3,), forming an orthonormal basis.

    Raises:
        ValueError: if any two anchor vectors are co-linear or zero-length.
    """
    if center_idx is None or anchor1_idx is None or anchor2_idx is None:
        return None, None, None
    # Convert to numpy arrays
    c = np.array(coords[center_idx], dtype=float)
    a = np.array(coords[anchor1_idx], dtype=float)
    b = np.array(coords[anchor2_idx], dtype=float)

    # Primary axis: C->A
    v_ca = a - c
    norm_ca = np.linalg.norm(v_ca)
    if norm_ca < 1e-8:
        raise ValueError(f"Anchor1 at index {anchor1_idx} coincides with center {center_idx}.")
    e_x = v_ca / norm_ca

    # Secondary vector: C->B
    v_cb = b - c
    norm_cb = np.linalg.norm(v_cb)
    if norm_cb < 1e-8:
        raise ValueError(f"Anchor2 at index {anchor2_idx} coincides with center {center_idx}.")

    # z-axis: perpendicular to plane CA-CB
    e_z = np.cross(e_x, v_cb)
    norm_z = np.linalg.norm(e_z)
    if norm_z < 1e-8:
        raise ValueError("Anchors are colinear; cannot define unique plane.")
    e_z /= norm_z

    # y-axis: complete right-handed system
    e_y = np.cross(e_z, e_x)

    return e_x, e_y, e_z


def _compute_azimuthal_angles(center_idx: int,
                              hydrogen_indices: List[int],
                              coords: List[Tuple[float, float, float]],
                              e_y: np.ndarray,
                              e_z: np.ndarray
                              ) -> Dict[int, float]:
    """
    Compute the azimuthal angles (in degrees) of hydrogen atoms around a principal axis.

    For each hydrogen, project vector C->H onto the plane spanned by e_y and e_z,
    then calculate φ = atan2(dot(v, e_z), dot(v, e_y)) mod 360.

    Args:
        center_idx (int): Index of the central atom C.
        hydrogen_indices (List[int]): List of hydrogen atom indices.
        coords (List[Tuple[float, float, float]]): 3D coordinates for all atoms.
        e_y (np.ndarray): Local unit y-axis.
        e_z (np.ndarray): Local unit z-axis.

    Returns:
        Dict mapping each hydrogen index to its azimuthal angle in [0, 360).
    """
    c = np.array(coords[center_idx], dtype=float)
    angles: Dict[int, float] = dict()
    for h_idx in hydrogen_indices:
        v = np.array(coords[h_idx], dtype=float) - c
        proj_y = float(np.dot(v, e_y))
        proj_z = float(np.dot(v, e_z))
        angle = float(np.degrees(np.arctan2(proj_z, proj_y)) % 360)
        angles[h_idx] = angle
    return angles


def _determine_cyclic_shift(angles_1: Dict[int, float],
                            angles_2: Dict[int, float]
                            ) -> Dict[int, int]:
    """
    Determine the optimal cyclic permutation mapping between two sets of azimuthal angles.

    Given two dictionaries of hydrogen indices to angles (in degrees),
    this finds the cyclic shift of the second angle-list that minimizes
    the total periodic angular difference to the first.

    Args:
        angles_1 (Dict[int, float]): Mapping H-index → angle for species1.
        angles_2 (Dict[int, float]): Mapping H-index → angle for species2.

    Returns:
        Dict[int, int]: Mapping from each H-index in angles_1 to the best-matched H-index in angles_2.
    """
    # If no hydrogens, return empty mapping
    if not angles_1 or not angles_2:
        return {}
    # Sort by angle
    items1 = sorted(angles_1.items(), key=lambda x: x[1])  # List of (h_idx, angle)
    items2 = sorted(angles_2.items(), key=lambda x: x[1])
    idx1, ang1 = zip(*items1)
    idx2, ang2 = zip(*items2)
    n = len(idx1)
    best_cost = float('inf')
    best_shift = 0
    # Try all cyclic shifts
    for shift in range(n):
        cost = 0.0
        for i in range(n):
            a1 = ang1[i]
            a2 = ang2[(i + shift) % n]
            diff = abs(a1 - a2) % 360
            diff = min(diff, 360 - diff)
            cost += diff
        if cost < best_cost:
            best_cost = cost
            best_shift = shift
    # Build mapping using best shift
    mapping: Dict[int, int] = dict()
    for i in range(n):
        mapping[idx1[i]] = idx2[(i + best_shift) % n]
    return mapping


def _map_xh3_group(heavy_idx_1: int,
                   heavy_index_2: int,
                   spc_1: ARCSpecies,
                   spc_2: ARCSpecies,
                   backbone_map: Dict[int, int]
                   ) -> Optional[Dict[int, int]]:
    """
    Map hydrogens of a CH3 group between two species using local azimuthal angles.

    Args:
        heavy_idx_1 (int): Index of CH3 carbon atom in species 1.
        heavy_index_2 (int): Index of CH3 carbon atom in species 2.
        spc_1 (ARCSpecies): Species 1.
        spc_2 (ARCSpecies): Species 2.
        backbone_map (Dict[int, int]): Existing backbone atom mapping.

    Returns:
        Optional[Dict[int, int]]: Mapping from hydrogen indices in spc_1 to hydrogen indices in spc_2.
    """
    anchors_1 = _select_ch3_anchors(heavy_idx_1, spc_1, backbone_map)
    anchors_2 = _select_ch3_anchors(heavy_index_2, spc_2, {val: key for key, val in backbone_map.items()})  # reverse map

    if not anchors_1 or not anchors_2:
        return None

    coords_1, coords_2 = spc_1.get_xyz()['coords'], spc_2.get_xyz()['coords']

    e_x1, e_y1, e_z1 = _construct_local_axes(heavy_idx_1, anchors_1[0], anchors_1[1], coords_1)
    e_x2, e_y2, e_z2 = _construct_local_axes(heavy_index_2, anchors_2[0], anchors_2[1], coords_2)

    hydrogens_1 = _find_hydrogen_neighbors(heavy_idx_1, spc_1.mol.atoms)
    hydrogens_2 = _find_hydrogen_neighbors(heavy_index_2, spc_2.mol.atoms)

    angles_1 = _compute_azimuthal_angles(heavy_idx_1, hydrogens_1, coords_1, e_y1, e_z1)
    angles_2 = _compute_azimuthal_angles(heavy_index_2, hydrogens_2, coords_2, e_y2, e_z2)

    return _determine_cyclic_shift(angles_1, angles_2)


def _compute_ch2_pair_dihedrals(coords_1, coords_2,
                                heavy_index_1, heavy_index_2,
                                neighbor_A1, neighbor_A2, neighbor_B1, neighbor_B2,
                                h_a_1, h_b_1, h_a_2, h_b_2,
                                ) -> Tuple[float, float, float, float]:
    """
    A helper function for _map_xh2_group.
    Return phi_1_a, phi_1_b, phi_2_a, phi_2_b all mapped into [-180,180).
    """
    phi_1_a = get_angle_in_180_range(
        calculate_dihedral_angle(coords=coords_1, torsion=[neighbor_B1, neighbor_A1, heavy_index_1, h_a_1]))
    phi_1_b = get_angle_in_180_range(
        calculate_dihedral_angle(coords=coords_1, torsion=[neighbor_B1, neighbor_A1, heavy_index_1, h_b_1]))
    phi_2_a = get_angle_in_180_range(
        calculate_dihedral_angle(coords=coords_2, torsion=[neighbor_B2, neighbor_A2, heavy_index_2, h_a_2]))
    phi_2_b = get_angle_in_180_range(
        calculate_dihedral_angle(coords=coords_2, torsion=[neighbor_B2, neighbor_A2, heavy_index_2, h_b_2]))
    return phi_1_a, phi_1_b, phi_2_a, phi_2_b


def _map_xh2_group(heavy_index_1: int,
                   heavy_index_2: int,
                   spc_1: ARCSpecies,
                   spc_2: ARCSpecies,
                   backbone_map: Dict[int, int],
                   ) -> Dict[int, int]:
    """
    Map the two hydrogens of a XH2 group (X is, e.g., C) by comparing signed dihedral angles.

    Strategy:
      1. Identify the two H's on X1 and X2.
      2. Find a non-H neighbor A1 of X1 and its mapped counterpart A2 in X2.
      3. Compute dihedral φ1 = dihedral(A1–X1–Ha1–Hb1) in species 1.
      4. Compute φ2_fwd = dihedral(A2–X2–Ha2–Hb2) and φ2_rev = dihedral(A2–X2–Hb2–Ha2).
      5. Assign Ha1→Ha2, Hb1→Hb2 if |φ1–φ2_fwd| < |φ1–φ2_rev|, else swap.

    Args:
        heavy_index_1: Index of the CH2 carbon in species 1.
        heavy_index_2: Index of the corresponding CH2 carbon in species 2.
        spc_1: ARCSpecies for species 1.
        spc_2: ARCSpecies for species 2.
        backbone_map: Existing heavy‐atom mapping X1→X2, etc.

    Returns:
        Dict[int,int]: Mapping of the two H indices from spc_1 to spc_2, or {} if unable to apply dihedral strategy.
    """
    h_list_1 = _find_hydrogen_neighbors(heavy_index_1, spc_1.mol.atoms)
    h_list_2 = _find_hydrogen_neighbors(heavy_index_2, spc_2.mol.atoms)
    if len(h_list_1) != 2 or len(h_list_2) != 2:
        return {}

    neighbor_A1 = _find_preferably_heavy_neighbor(heavy_index_1, spc_1, partial_atom_map=backbone_map)
    neighbor_A2 = backbone_map.get(neighbor_A1, None)
    if neighbor_A2 is None:
        return {}

    coords_1, coords_2 = spc_1.get_xyz()['coords'], spc_2.get_xyz()['coords']
    h_a_1, h_b_1 = h_list_1
    h_a_2, h_b_2 = h_list_2

    heavy_atom_terminal = len([atom for atom in spc_1.mol.atoms[heavy_index_1].edges if not atom.is_hydrogen()
                           and spc_1.mol.atoms.index(atom) != neighbor_A1]) == 0
    if heavy_atom_terminal:
        neighbor_B1 = _find_preferably_heavy_neighbor(neighbor_A1, spc_1, partial_atom_map=backbone_map, exclude_index=heavy_index_1)
        neighbor_B2 = backbone_map.get(neighbor_B1, None)
        if neighbor_B2 is not None:
            phi_h_a_1, phi_h_b_1, phi_h_a_2, phi_h_b_2 = _compute_ch2_pair_dihedrals(
                coords_1, coords_2, heavy_index_1, heavy_index_2,
                neighbor_A1, neighbor_A2, neighbor_B1, neighbor_B2, h_a_1, h_b_1, h_a_2, h_b_2)
            diff_ha1_ha2, diff_hb1_hb2 = signed_angular_diff(phi_h_a_1, phi_h_a_2), signed_angular_diff(phi_h_b_1, phi_h_b_2)
            diff_ha1_hb2, diff_hb1_ha2 = signed_angular_diff(phi_h_a_1, phi_h_b_2), signed_angular_diff(phi_h_b_1, phi_h_a_2)
            fwd_diff, rev_diff = abs(diff_ha1_ha2) + abs(diff_hb1_hb2), abs(diff_ha1_hb2) + abs(diff_hb1_ha2)
            if fwd_diff < rev_diff:
                # prefer Ha1-Ha2, Hb1-Hb2 mapping
                return {h_a_1: h_a_2, h_b_1: h_b_2}
            # prefer Ha1-Hb2, Hb1-Ha2 mapping
            return {h_a_1: h_b_2, h_b_1: h_a_2}

    # compute reference dihedral in species 1
    phi_1 = calculate_dihedral_angle(coords=coords_1, torsion=[neighbor_A1, heavy_index_1, h_a_1, h_b_1])

    # compute two candidate dihedrals in species 2
    phi_2_fwd = calculate_dihedral_angle(coords=coords_2, torsion=[neighbor_A2, heavy_index_2, h_a_2, h_b_2])
    phi_2_rev = calculate_dihedral_angle(coords=coords_2, torsion=[neighbor_A2, heavy_index_2, h_b_2, h_a_2])

    # choose pairing minimizing angular separation
    diff_fwd = signed_angular_diff(phi_1, phi_2_fwd)
    diff_rev = signed_angular_diff(phi_1, phi_2_rev)

    forward_mapping = {h_a_1: h_a_2, h_b_1: h_b_2}
    reverse_mapping = {h_a_1: h_b_2, h_b_1: h_a_2}

    chosen_mapping = forward_mapping if abs(diff_fwd) < abs(diff_rev) else reverse_mapping

    if spc_1.mol.atoms[heavy_index_1].is_nitrogen():
        neighbor_B1 = _find_preferably_heavy_neighbor(neighbor_A1, spc_1, partial_atom_map=backbone_map, exclude_index=heavy_index_1)
        neighbor_B2 = backbone_map.get(neighbor_B1, None)
        if neighbor_B2 is not None:
            phi_h_a_1, phi_h_b_1, phi_h_a_2, phi_h_b_2 = _compute_ch2_pair_dihedrals(
                coords_1, coords_2, heavy_index_1, heavy_index_2,
                neighbor_A1, neighbor_A2, neighbor_B1, neighbor_B2, h_a_1, h_b_1, h_a_2, h_b_2)
            avg_abs_1, avg_abs_2 = (abs(phi_h_a_1) + abs(phi_h_b_1)) / 2, (abs(phi_h_a_2) + abs(phi_h_b_2)) / 2
            n_inversion_mode_1, n_inversion_mode_2 = avg_abs_1 < 90.0, avg_abs_2 < 90.0
            if n_inversion_mode_1 != n_inversion_mode_2:
                # identified an N inversion mode, swap the mapping
                chosen_mapping = reverse_mapping if chosen_mapping is forward_mapping else forward_mapping

    return chosen_mapping


def map_hydrogens(spc_1: ARCSpecies,
                  spc_2: ARCSpecies,
                  backbone_map: Dict[int, int],
                  ) -> Dict[int, int]:
    """
    Atom map hydrogen atoms between two species with a known mapped heavy atom backbone.
    If only a single hydrogen atom is bonded to a given heavy atom, it is straight-forwardly mapped.

    Args:
        spc_1 (ARCSpecies): Species 1.
        spc_2 (ARCSpecies): Species 2.
        backbone_map (Dict[int, int]): The backbone map.

    Returns:
        Dict[int, int]: The atom map. Keys are 0-indices in ``spc_1``, values are corresponding 0-indices in ``spc_2``.
    """
    atom_map = backbone_map.copy()
    atoms_1, atoms_2 = spc_1.mol.atoms, spc_2.mol.atoms
    for heavy_index_1, heavy_index_2 in backbone_map.items():
        h_indices_1 = [h for h in _find_hydrogen_neighbors(heavy_index_1, atoms_1) if h not in atom_map]
        if not h_indices_1:
            continue
        h_indices_2 = _find_hydrogen_neighbors(heavy_index_2, atoms_2)
        if len(h_indices_1) != len(h_indices_2):
            raise ValueError(f"Mismatch H‐counts at {heavy_index_1}: {len(h_indices_2)} vs {len(h_indices_2)}.\n"
                             f"Cannot atom map species {spc_1.mol.smiles} and {spc_2.mol.smiles}.")

        if all(atom.is_hydrogen() for atom in atoms_1[heavy_index_1].edges.keys()):
            # This is a heavy atom with only H atoms adjacent to it (e.g., CH4, NH3, H2).
            for atom_1, atom_2 in zip([atom for atom in atoms_1 if atom.is_hydrogen()],
                                      [atom for atom in atoms_2 if atom.is_hydrogen()]):
                atom_map[atoms_1.index(atom_1)] = atoms_2.index(atom_2)
        elif len(h_indices_1) == 1:
            atom_map[h_indices_1[0]] = h_indices_2[0]
            continue
        elif len(h_indices_1) == 2:
            mapped = _map_xh2_group(heavy_index_1, heavy_index_2, spc_1, spc_2, atom_map)
            if mapped:
                atom_map.update(mapped)
                continue
        elif len(h_indices_1) == 3:
            if len(spc_1.mol.atoms) <= 5:
                # A-XH3 should be a trivial assignment:
                atom_map[h_indices_1[0]] = h_indices_2[0]
                atom_map[h_indices_1[1]] = h_indices_2[1]
                atom_map[h_indices_1[2]] = h_indices_2[2]
            else:
                mapped = _map_xh3_group(heavy_index_1, heavy_index_2, spc_1, spc_2, atom_map)
                if mapped:
                    atom_map.update(mapped)
    return atom_map


def check_atom_map(rxn: 'ARCReaction') -> bool:
    """
    A helper function for testing a reaction atom map.
    Tests that element symbols are ordered correctly.
    Tests that the elements in the atom map are unique, so that the function is one to one.
    Note: These are necessary but not sufficient conditions.

    Args:
        rxn (ARCReaction): The reaction to examine.

    Returns: bool
        Whether the atom mapping makes sense.
    """
    if len(rxn.atom_map) != sum([spc.number_of_atoms for spc in rxn.r_species]):
        return False
    r_elements, p_elements = list(), list()
    for r_species in rxn.r_species:
        r_elements.extend(list(r_species.get_xyz()['symbols']))
    for p_species in rxn.p_species:
        p_elements.extend(list(p_species.get_xyz()['symbols']))
    for i, map_i in enumerate(rxn.atom_map):
        if r_elements[i] != p_elements[map_i]:
            break
    for i in range(len(np.unique(rxn.atom_map))):
        if np.unique(rxn.atom_map)[i] != i:
            return False
    return True


def add_adjacent_hydrogen_atoms_to_map_based_on_a_specific_torsion(spc_1: ARCSpecies,
                                                                   spc_2: ARCSpecies,
                                                                   heavy_atom_1: 'Atom',
                                                                   heavy_atom_2: 'Atom',
                                                                   torsion: List[int],
                                                                   atom_map: Dict[int, int],
                                                                   find_torsion_end_to_map: bool = True,
                                                                   ) -> Dict[int, int]:
    """
    Map hydrogen atoms around one end of a specific torsion (or pseudo-torsion) by matching dihedral angles.

    Args:
          spc_1 (ARCSpecies): Species 1.
          spc_2 (ARCSpecies): Species 2.
          heavy_atom_1 (Atom): The heavy atom from ``spc_1`` around which hydrogen atoms will be mapped.
          heavy_atom_2 (Atom): The heavy atom from ``spc_2`` around which hydrogen atoms will be mapped.
          torsion (List[int]): 0-indices of 4 consecutive atoms in ``spc_1``.
          atom_map (Dict[int, int]): A partial atom map between ``spc_1`` and ``spc_2``.
          find_torsion_end_to_map (bool, optional): Whether to automatically identify the side of the torsion that
                                                    requires mapping. If ``False``, the last atom, ``torsion[-1]``
                                                    is assumed to be on the side that requires mapping.

    Returns:
        Dict[int, int]: The updated atom map.
    """
    atoms_1, atoms_2 = spc_1.mol.atoms, spc_2.mol.atoms
    hydrogen_indices_1 = [atoms_1.index(atom)
                          for atom in heavy_atom_1.edges.keys() if atom.is_hydrogen()]
    hydrogen_indices_2 = [atoms_2.index(atom)
                          for atom in heavy_atom_2.edges.keys() if atom.is_hydrogen()]
    torsion_1_base = torsion[::-1][:-1] if torsion[-1] not in hydrogen_indices_1 and find_torsion_end_to_map else torsion[:-1]
    torsions_1 = [torsion_1_base + [h_index] for h_index in hydrogen_indices_1]
    if torsion_1_base[0] not in atom_map.keys():
        # There's an unmapped hydrogen atom in torsion_1_base. Randomly map it to a respective hydrogen.
        atom_map[torsion_1_base[0]] = [atoms_2.index(atom)
                                       for atom in atoms_2[atom_map[torsion_1_base[1]]].edges.keys()
                                       if atom.is_hydrogen()][0]
    torsions_2 = [[atom_map[t] for t in torsion_1_base] + [h_index] for h_index in hydrogen_indices_2]
    dihedrals_1 = [calculate_dihedral_angle(coords=spc_1.get_xyz(), torsion=torsion)
                   for torsion in torsions_1]
    dihedrals_2 = [calculate_dihedral_angle(coords=spc_2.get_xyz(), torsion=torsion)
                   for torsion in torsions_2]
    dihedral_map_dict = map_lists(dihedrals_1, dihedrals_2)
    for key, val in dihedral_map_dict.items():
        atom_map[hydrogen_indices_1[key]] = hydrogen_indices_2[val]
    return atom_map


def flip_map(atom_map: Optional[List[int]]) -> Optional[List[int]]:
    """
    Flip an atom map so that the products map to the reactants.

    Args:
        atom_map (List[int]): The atom map in a list format.

    Returns:
        Optional[List[int]]: The flipped atom map.
    """
    if atom_map is None:
        return None
    flipped_map = [-1] * len(atom_map)
    for index, entry in enumerate(atom_map):
        flipped_map[entry] = index
    if any(entry < 0 for entry in flipped_map):
        raise ValueError(f'Cannot flip the atom map {atom_map}')
    return flipped_map


def make_bond_changes(rxn: 'ARCReaction', r_cuts: List[ARCSpecies], r_label_dict: dict) -> None:
    """
    Makes bond changes before matching the reactants and products.

    Args:
        rxn ('ARCReaction'): An ARCReaction object
        r_cuts (List[ARCSpecies]): The cut products
        r_label_dict (dict): The dictionary of reactant atom labels for the family recipe.
    """
    family = ReactionFamily(label=rxn.family)
    for action in family.actions:
        if action[0].lower() == "change_bond":
            indices = r_label_dict[action[1]], r_label_dict[action[3]]
            for r_cut in r_cuts:
                r_cut_mol_copy = r_cut.mol.copy(deep=True)
                if indices[0] in [int(atom.label) for atom in r_cut.mol.atoms] and indices[1] in [int(atom.label) for atom in r_cut.mol.atoms]:
                    atom1, atom2 = 0, 0
                    for atom in r_cut.mol.atoms:
                        if int(atom.label) == indices[0]:
                            atom1 = atom
                        elif int(atom.label) == indices[1]:
                            atom2 = atom
                    if atom1.radical_electrons == 0 and atom2.radical_electrons == 0: # Both atoms do not have any radicals, but their bond is changing. There probably is resonance, so this will not affect the isomorphism check.
                        return
                    elif atom1.radical_electrons == 0 and atom2.radical_electrons != 0:
                        atom1.lone_pairs -= 1
                        atom2.lone_pairs += 1
                        atom1.charge += 1
                        atom2.charge -= 1
                        atom2.radical_electrons -= 2
                    elif atom2.radical_electrons == 0 and atom1.radical_electrons != 0:
                        atom2.lone_pairs -= 1
                        atom1.lone_pairs += 1
                        atom2.charge += 1
                        atom1.charge -= 1
                        atom1.radical_electrons -= 2
                    else:
                        atom1.decrement_radical()
                        atom2.decrement_radical()
                    r_cut.mol.get_bond(atom1, atom2).order += action[2]
                    try:
                        r_cut.mol.update(sort_atoms=False)
                    except AtomTypeError:
                        r_cut.mol = r_cut_mol_copy


def update_xyz(species: List[ARCSpecies]) -> List[ARCSpecies]:
    """
    A helper function, updates the xyz values of each species after cutting. This is important, since the
    scission sometimes scrambles the Molecule object, and updating the xyz makes up for that.

    Args:
        species (List[ARCSpecies]): the scission products that needs to be updated.

    Returns:
        List[ARCSpecies]: A newly generated copies of the ARCSpecies, with updated xyz.
    """
    new = list()
    for spc in species:
        new_spc = ARCSpecies(label="copy", mol=spc.mol.copy(deep=True))
        xyz_1, xyz_2 = None, None
        try:
            xyz_1 = new_spc.get_xyz()
        except (ConformerError, InputError, TypeError, ValueError):
            pass
        try:
            xyz_2 = spc.get_xyz()
        except (ConformerError, InputError, TypeError, ValueError):
            pass
        new_spc.final_xyz = xyz_1 or xyz_2
        new.append(new_spc)
    return new


def r_cut_p_cut_isomorphic(reactant: ARCSpecies, product_: ARCSpecies) -> bool:
    """
    A function for checking if the reactant and product are the same molecule.

    Args:
        reactant (ARCSpecies): An ARCSpecies. might be as a result of scissors()
        product_ (ARCSpecies): an ARCSpecies. might be as a result of scissors()

    Returns:
        bool: ``True`` if they are isomorphic, ``False`` otherwise.
    """
    res1 = generate_resonance_structures_safely(reactant.mol, save_order=True)
    for res in res1:
        if res.fingerprint == product_.mol.fingerprint or product_.mol.is_isomorphic(res, save_order=True):
            return True
    return False


def pairing_reactants_and_products_for_mapping(r_cuts: List[ARCSpecies],
                                               p_cuts: List[ARCSpecies]
                                               )-> List[Tuple[ARCSpecies,ARCSpecies]]:
    """
    A function for matching reactants and products in scissored products.

    Args:
        r_cuts (List[ARCSpecies]): A list of the scissored species in the reactants
        p_cuts (List[ARCSpecies]): A list of the scissored species in the reactants

    Returns:
        List[Tuple[ARCSpecies,ARCSpecies]]: A list of paired reactant and products, to be sent to map_two_species.
    """
    pairs = list()
    for reactant_cut in r_cuts:
        for product_cut in p_cuts:
            if r_cut_p_cut_isomorphic(reactant_cut, product_cut):
                pairs.append((reactant_cut, product_cut))
                p_cuts.remove(product_cut) # Just in case there are two of the same species in the list, matching them by order.
                break
    return pairs


def map_pairs(pairs: List[Tuple[ARCSpecies, ARCSpecies]]) -> List[List[int]]:
    """
    A function that maps the matched species together

    Args:
         (List[Tuple[ARCSpecies, ARCSpecies]]): A list of the pairs of reactants and species.

    Returns:
        List[List[int]]: A list of the mapped species
    """
    maps = list()
    for pair in pairs:
        maps.append(map_two_species(pair[0], pair[1]))
    return maps


def label_species_atoms(species: List['ARCSpecies']) -> None:
    """
    Adds the labels to the ``.mol.atoms`` properties of the species object.

    Args:
        species (List['ARCSpecies']): ARCSpecies object to be labeled.
    """
    index = 0
    for spc in species:
        for atom in spc.mol.atoms:
            atom.label = str(index)
            index += 1


def glue_maps(maps: List[List[int]],
              pairs: List[Tuple[ARCSpecies, ARCSpecies]],
              r_label_map: Dict[str, int],
              p_label_map: Dict[str, int],
              total_atoms: int,
              ) -> List[int]:
    """
    Join the maps from the parts of a bimolecular reaction.

    Args:
        maps (List[List[int]]): The list of all maps of the isomorphic cuts.
        pairs (List[Tuple[ARCSpecies, ARCSpecies]]): The pairs of the reactants and products.
        r_label_map (Dict[str, int]): A dictionary mapping the reactant labels to their indices.
        p_label_map (Dict[str, int]): A dictionary mapping the product labels to their indices.
        total_atoms (int): The total number of atoms across all reactants.

    Returns:
        List[int]: An Atom Map of the complete reaction.
    """
    # 1) Build base map
    am_dict: Dict[int,int] = {}
    for map_list, (r_cut, p_cut) in zip(maps, pairs):
        for local_i, r_atom in enumerate(r_cut.mol.atoms):
            r_glob = int(r_atom.label)
            p_glob = int(p_cut.mol.atoms[map_list[local_i]].label)
            am_dict[r_glob] = p_glob

    # 2) Override via swap: any time we impose r_glob→target,
    #    if some other r2 already points at target, swap their values.
    for tag, r_glob in r_label_map.items():
        if tag not in p_label_map:
            continue
        target = p_label_map[tag]
        original = am_dict[r_glob]
        if original == target:
            continue
        # find any other reactant already pointing at `target`
        rival = next((r for r, p in am_dict.items() if p == target and r != r_glob), None)
        if rival is not None:
            # swap: rival takes `original`
            am_dict[rival] = original
        # now assign r_glob → target
        am_dict[r_glob] = target

    # 3) Build final list, error if missing any index
    final = []
    for i in range(total_atoms):
        if i not in am_dict:
            raise ValueError(f"Invalid atom_map: missing reactant index {i}")
        final.append(am_dict[i])
    return final


def determine_bdes_on_spc_based_on_atom_labels(spc: "ARCSpecies", bde: Tuple[int, int]) -> bool:
    """
    A function for determining whether the species in question contains the bond specified by the bond dissociation indices.
    Also, assigns the correct BDE to the species.

    Args:
        spc (ARCSpecies): The species in question, with labels atom indices.
        bde (Tuple[int, int]): The bde in question.

    Returns:
        bool: Whether the bde is based on the atom labels.
    """
    index1, index2 = bde[0], bde[1]
    new_bde, atoms = list(), list()
    for index, atom in enumerate(spc.mol.atoms):
        if atom.label == str(index1) or atom.label == str(index2):
            new_bde.append(index+1)
            atoms.append(atom)
        if len(new_bde) == 2:
            break
    if len(new_bde) == 2 and atoms[1] in atoms[0].bonds.keys():
        spc.bdes = [tuple(new_bde)]
        return True
    return False


def cut_species_based_on_atom_indices(species: List["ARCSpecies"],
                                      bdes: List[Tuple[int, int]],
                                      ref_species: Optional[List["ARCSpecies"]] = None,
                                      ) -> Optional[List["ARCSpecies"]]:
    """
    A function for scissoring species based on their atom indices.

    Args:
        species (List[ARCSpecies]): The species list that requires scission.
        bdes (List[Tuple[int, int]]): A list of the atoms between which the bond should be scissored.
                                      The atoms are described using the atom labels, and not the actual atom positions.
        ref_species (Optional[List[ARCSpecies]]): A reference species list for which BDE indices are given.

    Returns:
        Optional[List["ARCSpecies"]]: The species list input after the scission.
    """
    if ref_species is not None:
        bdes = translate_bdes_based_on_ref_species(species, ref_species, bdes)
    if not bdes:
        return species
    for bde in bdes:
        for index, spc in enumerate(species):
            if determine_bdes_on_spc_based_on_atom_labels(spc, bde):
                candidate = species.pop(index)
                candidate.final_xyz = candidate.get_xyz()
                if candidate.mol.copy(deep=True).smiles == "[H][H]":
                    labels = [atom.label for atom in candidate.mol.copy(deep=True).atoms]
                    try:
                        h1 = candidate.scissors()[0]
                    except SpeciesError:
                        return None
                    h2 = h1.copy()
                    h2.mol.atoms[0].label = labels[0] if h1.mol.atoms[0].label != labels[0] else labels[1]
                    species += [h1, h2]
                else:
                    try:
                        species += candidate.scissors()
                    except SpeciesError:
                        return None
                break
    return species


def translate_bdes_based_on_ref_species(species: List["ARCSpecies"],
                                        ref_species: List["ARCSpecies"],
                                        bdes: List[Tuple[int, int]],
                                        ) -> Optional[List[Tuple[int, int]]]:
    """
    Translate a list of BDE indices based on a reference species list.
    The given BDE indices refer to `ref_species`, and they'll be translated to refer to `species`.

    Args:
        species (List[ARCSpecies]): The species list for which the indices should be translated.
        ref_species (List[ARCSpecies]): The reference species list.
        bdes (List[Tuple[int, int]]): The BDE indices to be translated.

    Returns:
        Optional[List[Tuple[int, int]]]: The translated BDE indices, or None if translation fails.
    """
    if not bdes:
        return list()
    all_indices = set()
    for bde in bdes:
        all_indices.update(bde)
    sorted_indices = sorted(all_indices)
    translated_indices = translate_indices_based_on_ref_species(species=species,
                                                                ref_species=ref_species,
                                                                indices=sorted_indices)
    if translated_indices is None:
        return None
    index_translation_map = dict(zip(sorted_indices, translated_indices))
    new_bdes = list()
    for bde in bdes:
        a, b = bde
        translated_a = index_translation_map.get(a)
        translated_b = index_translation_map.get(b)
        if translated_a is not None and translated_b is not None:
            new_bdes.append(tuple(sorted([translated_a, translated_b])))
        else:
            return None
    return new_bdes


def translate_indices_based_on_ref_species(species: List["ARCSpecies"],
                                           ref_species: List["ARCSpecies"],
                                           indices: List[int],
                                           ) -> Optional[List[int]]:
    """
    Translate a list of atom indices based on a reference species list.
    The given indices refer to `ref_species`, and they'll be translated to refer to `species`.

    Args:
        species (List[ARCSpecies]): The species list for which the indices should be translated.
        ref_species (List[ARCSpecies]): The reference species list.
        indices (List[int]): The list of atom indices to be translated.

    Returns:
        Optional[List[int]]: The translated atom indices if all translations are successful; otherwise, None.
    """
    visited_ref_species = list()
    species_map = dict()  # maps ref species j to species i
    index_map = dict()  # keys are ref species j indices, values are atom maps between ref species j and species i
    for i, spc in enumerate(species):
        for j, ref_spc in enumerate(ref_species):
            if j not in visited_ref_species and spc.is_isomorphic(ref_spc):
                visited_ref_species.append(j)
                species_map[j] = i
                index_map[j] = map_two_species(ref_spc, spc)
                break
    ref_spcs_lengths = [ref_spc.number_of_atoms for ref_spc in ref_species]
    accum_sum_ref_spcs_lengths = [sum(ref_spcs_lengths[:i+1]) for i in range(len(ref_spcs_lengths))]
    spcs_lengths = [spc.number_of_atoms for spc in species]
    accum_sum_spcs_lengths = [sum(spcs_lengths[:i+1]) for i in range(len(spcs_lengths))]

    def translate_single_index(index: int) -> Optional[int]:
        for ref_idx, ref_len in enumerate(accum_sum_ref_spcs_lengths):
            if index < ref_len:
                atom_map = index_map.get(ref_idx)
                species_i = species_map.get(ref_idx)
                if atom_map is None or species_i is None:
                    return None
                increment = accum_sum_spcs_lengths[species_i - 1] if species_i > 0 else 0
                ref_start = accum_sum_ref_spcs_lengths[ref_idx - 1] if ref_idx > 0 else 0
                translated_atom = atom_map[index - ref_start] + increment
                return translated_atom
        return None

    new_indices = list()
    for idx in indices:
        translated_idx = translate_single_index(idx)
        if translated_idx is not None:
            new_indices.append(translated_idx)
        else:
            return None
    return new_indices


def copy_species_list_for_mapping(species: List["ARCSpecies"]) -> List["ARCSpecies"]:
    """
    A helper function for copying the species list for mapping. Also keeps the atom indices when copying.

    Args:
        species (List[ARCSpecies]): The species list to be copied.

    Returns:
        List[ARCSpecies]: The copied species list.
    """
    copies = [spc.copy() for spc in species]
    for copy, spc in zip(copies, species):
        for atom1, atom2 in zip(copy.mol.atoms, spc.mol.atoms):
            atom1.label = atom2.label
    return copies


def find_all_breaking_bonds(rxn: "ARCReaction",
                            r_direction: bool,
                            pdi: int = 0,
                            ) -> Optional[List[Tuple[int, int]]]:
    """
    A function for finding all the broken (or formed of the direction to consider starts with the products)
    bonds during a chemical reaction, based on marked atom labels.

    Args:
        rxn (ARCReaction): The reaction in question.
        r_direction (bool): Whether to consider the reactants direction (``True``) or the products direction (``False``).
        pdi (int): The product dictionary index to consider. Defaults to 0.

    Returns:
        List[Tuple[int, int]]: Entries are tuples of the form (atom_index1, atom_index2) for each broken bond (1-indexed),
                               representing the atom indices to be cut.
    """
    family = ReactionFamily(label=rxn.family)
    if not len(rxn.product_dicts):
        return None
    label_dict = rxn.product_dicts[pdi][f'{"r" if r_direction else "p"}_label_map']
    breaking_bonds = list()
    for action in family.actions:
        if action[0].lower() == ("break_bond" if r_direction else "form_bond"):
            breaking_bonds.append(tuple(sorted((label_dict[action[1]], label_dict[action[3]]))))
    if not r_direction:
        breaking_bonds = translate_bdes_based_on_ref_species(
            species=rxn.get_reactants_and_products()[1],
            ref_species=[ARCSpecies(label=f'S{i}', mol=mol) for i, mol in enumerate(rxn.product_dicts[pdi]['products'])],
            bdes=breaking_bonds)
    return breaking_bonds


def get_template_product_order(rxn: 'ARCReaction',
                               template_products: List[Molecule]
                               ) -> List[int]:
    """
    Determine the order of the template products based on the reaction products.

    Args:
        rxn (ARCReaction): the reaction whose .p_species defines the desired order.
        template_products (List[Molecule]): the RMG‐template Molecule list (typically comes from product_dicts[0]['products']).

    Returns:
        List[int]: A list of indices such that for each template molecule index.
                   The corresponding position in the order list yields the index on the reaction product.
    """
    order: List[int] = list()
    used = set()
    for template_mol in template_products:
        for i, prod in enumerate(rxn.p_species):
            if i not in used and prod.is_isomorphic(template_mol):
                order.append(i)
                used.add(i)
                break
        else:
            raise ValueError(f"No template‐product match. Got the following templates:\n"
                             f"{[template_mol.to_smiles() for template_mol in template_products]}.")
    return order


def reorder_p_label_map(p_label_map: Dict[str, int],
                        template_order: List[int],
                        template_products: List['Molecule'],
                        actual_products: List[ARCSpecies],
                        ) -> Dict[str, int]:
    """
    Re‐compute the global indices in p_label_map_tpl to account for a new ordering (template_order) of the template products.

    Args:
        p_label_map (Dict[str,int]): Original template‐product‐label → global‐atom‐index map (in the order of template_products).
        template_order (List[int]): A permutation of range(len(template_products)) mapping new positions → old indices.
        template_products (List[Molecule]): List of template Molecule/ARCSpecies in RMG-generated order.
        actual_products (List[ARCSpecies]): List of ARCSpecies in the reaction’s true product order.

    Returns:
        Dict[str,int]: A new product‐label → global‐atom‐index map consistent with the reordered product list.
    """
    arc_lengths = [len(spc.mol.atoms) for spc in actual_products]
    template_lengths = [len(mol.atoms) for mol in template_products]
    template_to_arc_maps = dict()
    for templet_i, template_product in enumerate(template_products):
        arc_mol_num = template_order[templet_i]
        atom_map_1 = map_two_species(template_product, actual_products[arc_mol_num])  # index is template, value is rxn
        arc_offset = sum(arc_lengths[:arc_mol_num])
        template_offset = sum(template_lengths[:templet_i])
        atom_map_1 = {i + template_offset: v + arc_offset for i, v in enumerate(atom_map_1)}
        template_to_arc_maps.update(atom_map_1)
    updated_p_label_map = dict()
    for label, template_index in p_label_map.items():
        updated_p_label_map[label] = template_to_arc_maps[template_index]
    return updated_p_label_map
