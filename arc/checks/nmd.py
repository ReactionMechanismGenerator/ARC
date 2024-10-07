"""
A module for checking the normal mode displacement of a TS.
"""

import numpy as np
from collections import Counter
from itertools import product
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import qcelemental as qcel

import arc.rmgdb as rmgdb
from arc import parser
from arc.common import get_logger
from arc.species.converter import get_most_common_isotope_for_element, xyz_from_data, xyz_to_np_array
from arc.species.vectors import calculate_distance, VectorsError

if TYPE_CHECKING:
    from arc.job.adapter import JobAdapter
    from arc.reaction import ARCReaction
    from rmgpy.molecule.molecule import Molecule

logger = get_logger()


def analyze_ts_normal_mode_displacement(reaction: 'ARCReaction',
                                        job: Optional['JobAdapter'],
                                        amplitude: Union[float, list] = 0.25,
                                        weights: Union[bool, np.array] = True,
                                        ) -> Optional[bool]:
    """
    Analyze the normal mode displacement by identifying bonds that break and form
    and comparing them to the expected given reaction.
    Note that the TS geometry must be in the standard orientation for the normal mode displacement to be relevant.

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        job (JobAdapter): The frequency job object instance that points to the respective log file.
        amplitude (Union[float, list]): The amplitude of the normal mode displacement motion to check.
                                        If a list, all possible results are returned.
        weights (Union[bool, np.array]): Whether to use weights for the displacement.
                                         If ``False``, use ones as weights. If ``True``, use sqrt of atom masses.
                                         If an array, use the array values it as individual weights per atom.

    Returns:
        Optional[bool]: Whether the TS normal mode displacement is consistent with the desired reaction.
    """
    if job is None:
        return
    if reaction.family is None:
        rmgdb.determine_family(reaction)

    ts_xyz = reaction.ts_species.get_xyz()
    try:
        freqs, normal_mode_disp = parser.parse_normal_mode_displacement(path=job.local_path_to_output_file)
    except NotImplementedError:
        logger.warning(f'Could not parse frequencies (and cannot compute NMD) for TS {reaction.ts_species.label}.')
        return

    amplitude = [amplitude] if isinstance(amplitude, float) else amplitude
    weights = get_weights_from_xyz(xyz=ts_xyz, weights=weights)
    r_eq_atoms, _ = find_equivalent_atoms(reaction=reaction, reactant_only=True)
    formed_bonds, broken_bonds = reaction.get_formed_and_broken_bonds()
    for amp in amplitude:
        if not amp:
            continue
        xyzs = get_displaced_xyzs(xyz=ts_xyz, amplitude=amp, normal_mode_disp=normal_mode_disp[0], weights=weights)
        if is_nmd_correct_for_any_mapping(reaction=reaction,
                                          xyzs=xyzs,
                                          formed_bonds=formed_bonds,
                                          broken_bonds=broken_bonds,
                                          r_eq_atoms=r_eq_atoms,
                                          ):
            return True
    return False


def is_nmd_correct_for_any_mapping(reaction: 'ARCReaction',
                                   xyzs: Tuple[dict, dict],
                                   formed_bonds: List[Tuple[int, int]],
                                   broken_bonds: List[Tuple[int, int]],
                                   r_eq_atoms: List[List[int]],
                                   ) -> bool:
    """
    Check if the normal mode displacement is consistent with the desired reaction for any atom mapping.

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        xyzs (Tuple[dict, dict]): The Cartesian coordinates of the TS displaced along the normal mode.
        formed_bonds (List[Tuple[int, int]]): The bonds that are formed in the reaction.
        broken_bonds (List[Tuple[int, int]]): The bonds that are broken in the reaction.
        r_eq_atoms (List[List[int]]): A list of equivalent atoms in the reactants.

    Returns:
        bool: Whether the TS normal mode displacement is consistent with the desired reaction for any atom mapping.
    """
    modified_bond_grand_list = get_eq_formed_and_broken_bonds(formed_bonds=formed_bonds,
                                                              broken_bonds=broken_bonds,
                                                              r_eq_atoms=r_eq_atoms,
                                                              )
    for eq_formed_bonds, eq_broken_bonds in modified_bond_grand_list:
        reactive_bonds_diffs = get_bond_length_changes(bonds=eq_formed_bonds + eq_broken_bonds, xyzs=xyzs)
        r_bonds, _ = reaction.get_bonds(r_bonds_only=True)
        non_reactive_bonds = list()
        for bond in r_bonds:
            if bond not in eq_formed_bonds and bond not in eq_broken_bonds:
                non_reactive_bonds.append(bond)
        baseline, std = get_bond_length_changes_baseline_and_std(non_reactive_bonds=non_reactive_bonds, xyzs=xyzs)
        sigma = (np.min(reactive_bonds_diffs) - baseline) / std
        if sigma > 3:
            return True


def get_eq_formed_and_broken_bonds(formed_bonds: List[Tuple[int, int]],
                                   broken_bonds: List[Tuple[int, int]],
                                   r_eq_atoms: List[List[int]],
                                   ) -> List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """
    Get the equivalent formed and broken bonds.

    Args:
         formed_bonds (List[Tuple[int, int]]): The bonds that are formed in the reaction.
         broken_bonds (List[Tuple[int, int]]): The bonds that are broken in the reaction.
         r_eq_atoms (List[List[int]]): A list of equivalent atoms in the reactants.

    Returns:
         List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]: The equivalent formed and broken bonds.
    """
    all_changing_indices = list()
    for bond in formed_bonds + broken_bonds:
        all_changing_indices.extend([bond[0], bond[1]])
    all_changing_indices = list(set(all_changing_indices))
    r_eq_atoms = [eq for eq in r_eq_atoms if any(i in eq for i in all_changing_indices)]
    modified_bond_grand_list = translate_all_tuples_simultaneously(list_1=formed_bonds,
                                                                   list_2=broken_bonds,
                                                                   equivalences=r_eq_atoms)
    return modified_bond_grand_list


def translate_all_tuples_simultaneously(list_1: List[Tuple[int, int]],
                                        list_2: List[Tuple[int, int]],
                                        equivalences: List[List[int]],
                                        ) -> List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """
    Translate all tuples simultaneously using a mapping.

    Args:
        list_1 (List[Tuple[int, int]]): The first list of tuples.
        list_2 (List[Tuple[int, int]]): The second list of tuples.
        equivalences (List[List[int]]): A list of equivalent atoms.

    Returns:
        List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]: The translated tuples.
    """
    mapping = create_equivalence_mapping(equivalences)
    all_indices = {i for tup in list_1 + list_2 for i in tup}
    translated_options = {i: mapping.get(i, [i]) for i in all_indices}
    translated_combinations = list(product(*(translated_options[i] for i in all_indices)))
    index_to_pos = {val: idx for idx, val in enumerate(all_indices)}
    all_translated_tuples = list()
    for combo in translated_combinations:
        combo_dict = {i: combo[index_to_pos[i]] for i in all_indices}
        translated_list_1 = [(combo_dict[a], combo_dict[b]) for (a, b) in list_1]
        translated_list_2 = [(combo_dict[a], combo_dict[b]) for (a, b) in list_2]
        if len(translated_list_1) != len(set(translated_list_1)):
            continue
        if len(translated_list_2) != len(set(translated_list_2)):
            continue
        all_translated_tuples.append((translated_list_1, translated_list_2))
    return all_translated_tuples


def create_equivalence_mapping(equivalences: List[List[int]]) -> Dict[int, List[int]]:
    """
    Create a mapping of atom indices to equivalence groups.

    Args:
        equivalences (List[List[int]]): A list of equivalent atoms.

    Returns:
        Dict[int, List[int]]: The mapping of atom indices to equivalence groups
    """
    mapping = dict()
    for group in equivalences:
        for item in group:
            mapping[item] = group
    return mapping


def get_weights_from_xyz(xyz: dict,
                         weights: Union[bool, np.array] = True,
                         ) -> np.array:
    """
    Get weights for atoms in a molecule.

    Args:
        xyz (dict): The Cartesian coordinates.
        weights (Union[bool, np.array]): Whether to use weights for the displacement.
                                         If ``False``, use ones as weights. If ``True``, use sqrt of atom masses.
                                         If an array, use it as weights.

    Returns:
        np.array: The weights for the atoms.
    """
    symbols = list()
    for i, symbol in enumerate(xyz['symbols']):
        if 'isotopes' in xyz.keys() and xyz['isotopes'][i] != get_most_common_isotope_for_element(symbol):
            symbols.append(symbol + str(xyz['isotopes'][i]))
        else:
            symbols.append(symbol)

    if isinstance(weights, bool):
        if weights:
            atom_masses = np.array([qcel.periodictable.to_mass(symbol) for symbol in symbols]).reshape(-1, 1)
            weights = np.sqrt(atom_masses)
        else:
            weights = np.ones((len(xyz['symbols']), 1))
    return weights


def get_displaced_xyzs(xyz: dict,
                       amplitude: float,
                       normal_mode_disp: np.array,
                       weights: np.array,
                       ) -> Tuple[dict, dict]:
    """
    Get the Cartesian coordinates of the TS displaced along a normal mode.

    Args:
        xyz (dict): The Cartesian coordinates.
        amplitude (float): The amplitude of the displacement.
        normal_mode_disp (np.array): The normal mode displacement matrix corresponding to the imaginary frequency.
        weights (np.array): The weights for the atoms.

    Returns:
        Tuple[dict, dict]: The Cartesian coordinates of the TS displaced along the normal mode.
    """
    np_coords = xyz_to_np_array(xyz)
    xyz_1 = xyz_from_data(coords=np_coords - amplitude * normal_mode_disp * weights,
                          symbols=xyz['symbols'], isotopes=xyz['isotopes'])
    xyz_2 = xyz_from_data(coords=np_coords + amplitude * normal_mode_disp * weights,
                          symbols=xyz['symbols'], isotopes=xyz['isotopes'])
    return xyz_1, xyz_2


def get_bond_length_changes_baseline_and_std(non_reactive_bonds: List[Tuple[int, int]],
                                             xyzs: Tuple[dict, dict],
                                             ) -> Tuple[float, float]:
    """
    Get the baseline for bond length change of all non-reactive bonds.

    Todo:
        When we have a comprehensive list of atom maps, we can pass the reaction and the atom map number to use, and do:
            non_reactive_bonds = set(r_bonds) & set(p_bonds)

    Args:
        non_reactive_bonds (Set[Tuple[int, int]]): The non-reactive bonds.
        xyzs (Tuple[dict, dict]): The Cartesian coordinates of the TS displaced along the normal mode.

    Returns:
        Tuple[float, float]:
            - The max baseline of bond length differences for non-reactive bonds.
            - The standard deviation of bond length differences for non-reactive bonds.
    """
    diffs = get_bond_length_changes(bonds=non_reactive_bonds, xyzs=xyzs)
    baseline_max = np.max(diffs)
    std = np.std(diffs)
    return baseline_max, std


def get_bond_length_changes(bonds: Union[List[Tuple[int, int]], Set[Tuple[int, int]]],
                            xyzs: Tuple[dict, dict],
                            ) -> np.array:
    """
    Get the bond length changes of specific bonds.

    Args:
        bonds (Union[list, tuple]): The bonds to check.
        xyzs (Tuple[dict, dict]): The Cartesian coordinates of the TS displaced along the normal mode.

    Returns:
        np.array: The bond length changes of the specified bonds.
    """
    diffs = list()
    for bond in bonds:
        r_bond_length = get_bond_length_in_reaction(bond=bond, xyz=xyzs[0])
        p_bond_length = get_bond_length_in_reaction(bond=bond, xyz=xyzs[1])
        if r_bond_length is None or p_bond_length is None:
            continue
        diffs.append(abs(r_bond_length - p_bond_length))
    diffs = np.array(diffs)
    return diffs


def get_bond_length_in_reaction(bond: Union[Tuple[int, int], List[int]],
                                xyz: dict,
                                ) -> Optional[float]:
    """
    Get the length of a bond in either the reactants or the products of a reaction.

    Args:
        bond (Tuple[int, int]): The bond to check.
        xyz (dict): The Cartesian coordinates mapped to the indices of the reactants.

    Returns:
        float: The bond length.
    """
    try:
        distance = calculate_distance(coords=xyz, atoms=bond, index=0)
    except (VectorsError, TypeError, IndexError):
        return None
    return distance


def find_equivalent_atoms(reaction: 'ARCReaction',
                          reactant_only: bool = True,
                          ) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Find equivalent atoms in the reactants and products of a reaction.
    This is a tentative function that should be replaced when atom mapping returns a list.
    It is meant to suggest additional atoms that can move instead of the ones selected by the atom map.

    Args:
        reaction (ARCReaction): The reaction for which equivalent atoms are searched.
        reactant_only (bool): Whether to search for equivalent atoms in the reactants only.

    Returns:
        Tuple[List[List[int]], List[List[int]]]:
            - A list of equivalent atoms in the reactants.
            - A list of equivalent atoms in the products, indices are atom mapped to the reactants.
    """
    reactants, products = reaction.get_reactants_and_products(arc=False, return_copies=True)
    r_eq_atoms, p_eq_atoms = list(), list()
    for i, reactant in enumerate(reactants):
        r_eq_atoms.extend(identify_equivalent_atoms_in_molecule(molecule=reactant.molecule[0],
                                                                inc=sum([len(r.molecule[0].atoms) for r in reactants[:i]]),
                                                                atom_map=None,
                                                                ))
    if not reactant_only:
        for i, product in enumerate(products):
            p_eq_atoms.extend(identify_equivalent_atoms_in_molecule(molecule=product.molecule[0],
                                                                    inc=sum([len(p.molecule[0].atoms) for p in products[:i]]),
                                                                    atom_map=reaction.atom_map,
                                                                    ))
    return r_eq_atoms, p_eq_atoms


def identify_equivalent_atoms_in_molecule(molecule: 'Molecule',
                                          inc: int = 0,
                                          atom_map: Optional[List[int]] = None,
                                          ) -> List[List[int]]:
    """
    Identify equivalent atoms in a molecule.

    Args:
        molecule (Molecule): The molecule to check.
        inc (int): The increment to be added.
        atom_map (Optional[List[int]]): The atom map.

    Returns:
        List[List[int]]: A list of equivalent atoms.
    """
    element_counts = Counter([atom.element.number for atom in molecule.atoms])
    repeated_elements = [element for element, count in element_counts.items() if count > 1]
    eq_atoms = list()
    for element_number in repeated_elements:
        fingerprint_dict = dict()
        for i, atom in enumerate(molecule.atoms):
            if atom.element.number != element_number:
                continue
            index = i + inc if atom_map is None else atom_map.index(i + inc)
            fingerprint_dict[index] = fingerprint_atom(atom_index=i, molecule=molecule)
        for index, fp in fingerprint_dict.items():
            for i in range(len(eq_atoms)):
                if index in eq_atoms[i]:
                    break
            else:
                eq_atoms.append([index])
            for other_index, other_fp in fingerprint_dict.items():
                if index == other_index:
                    continue
                if fp == other_fp:
                    for i in range(len(eq_atoms)):
                        if index in eq_atoms[i] and other_index not in eq_atoms[i]:
                            eq_atoms[i].append(other_index)
    eq_atoms = [eq for eq in eq_atoms if len(eq) > 1]
    return eq_atoms


def fingerprint_atom(atom_index: int,
                     molecule: 'Molecule',
                     excluded_atom_indices: Optional[List[int]] = None,
                     degree: int = 3,
                     ) -> list:
    """
    Fingerprint an atom in a molecule.

    Args:
        atom_index (int): The index of the atom to map.
        molecule (Molecule): The molecule to which the atom belongs.
        excluded_atom_indices (Optional[List[int]]): Atom indices to exclude from the mapping.
        degree (int): The degree of the atom map.

    Returns:
        List[int]: The atom map.
    """
    atom_0 = molecule.atoms[atom_index]
    fingerprint = [atom_0.element.number]
    if degree == 0:
        return fingerprint
    if len(molecule.atoms) > 1:
        fingerprint.append([])
    excluded_atom_indices = excluded_atom_indices or list()
    excluded_atom_indices.append(atom_index)
    for atom_1 in atom_0.edges.keys():
        atom_1_index = molecule.atoms.index(atom_1)
        if atom_1_index in excluded_atom_indices:
            continue
        excluded_atom_indices.append(atom_1_index)
        if degree == 1 or len(atom_1.edges) == 1:
            fingerprint[-1].append(atom_1.element.number)
        else:
            fingerprint.append(fingerprint_atom(atom_index=molecule.atoms.index(atom_1),
                                                    molecule=molecule,
                                                    excluded_atom_indices=excluded_atom_indices,
                                                    degree=degree - 1))
    for i in range(len(fingerprint)):
        if isinstance(fingerprint[i], list) and not any(isinstance(j, list) for j in fingerprint[i]):
            fingerprint[i].sort()
    pops = list()
    for i, fp in enumerate(fingerprint):
        if not fp:
            pops.append(i)
    for i in sorted(pops, reverse=True):
        fingerprint.pop(i)
    return fingerprint
