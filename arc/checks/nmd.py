"""
A module for checking the normal mode displacement of a TS.
"""

import numpy as np
from collections import Counter
from itertools import product
from typing import TYPE_CHECKING

from arc.parser import parser
from arc.common import get_element_mass, get_logger
from arc.species.converter import get_most_common_isotope_for_element, xyz_from_data, xyz_to_np_array
from arc.species.vectors import calculate_distance, VectorsError

if TYPE_CHECKING:
    from arc.job.adapter import JobAdapter
    from arc.reaction import ARCReaction
    from arc.molecule.molecule import Molecule

logger = get_logger()

# Module-level constants for NMD validation
SIGMA_THRESHOLD = 3.0
STD_FLOOR = 1e-4
DIRECTIONALITY_MIN_DELTA = 0.005

def analyze_ts_normal_mode_displacement(reaction: 'ARCReaction',
                                        job: 'JobAdapter' | None,
                                        amplitude: float | list = 0.25,
                                        weights: bool | np.array = True,
                                        ) -> bool | None:
    """
    Analyze the normal mode displacement by identifying bonds that break and form
    and comparing them to the expected given reaction.
    Note that the TS geometry must be in the standard orientation for the normal mode displacement to be relevant.

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        job (JobAdapter): The frequency job object instance that points to the respective log file.
        amplitude (float | list): The amplitude of the normal mode displacement motion to check.
                                        If a list, all possible results are returned.
        weights (bool | np.array): Whether to use weights for the displacement.
                                         If ``False``, use ones as weights. If ``True``, use sqrt of atom masses.
                                         If an array, use the array values it as individual weights per atom.

    Returns:
        bool | None: Whether the TS normal mode displacement is consistent with the desired reaction.
    """
    if job is None:
        return None
    ts_xyz = reaction.ts_species.get_xyz()
    n_ts = len(ts_xyz['symbols'])
    n_expected = sum(spc.number_of_atoms for spc in reaction.r_species)
    if n_ts != n_expected:
        return False
    try:
        freqs, normal_mode_disp = parser.parse_normal_mode_displacement(log_file_path=job.local_path_to_output_file)
    except NotImplementedError:
        logger.warning(f'Could not parse frequencies for TS {reaction.ts_species.label}.')
        return None

    amplitude_list = [amplitude] if isinstance(amplitude, (float, int)) else amplitude
    weights_array = get_weights_from_xyz(xyz=ts_xyz, weights=weights)
    r_eq_atoms, _ = find_equivalent_atoms(reaction=reaction, reactant_only=True)
    formed_bonds, broken_bonds = reaction.get_formed_and_broken_bonds()
    changed_bonds = reaction.get_changed_bonds()

    for amp in amplitude_list:
        if not amp:
            continue
        xyzs = get_displaced_xyzs(xyz=ts_xyz, amplitude=amp, normal_mode_disp=normal_mode_disp[0], weights=weights_array)

        nmd_correct = is_nmd_correct_for_any_mapping(
            reaction=reaction,
            xyzs=xyzs,
            formed_bonds=formed_bonds,
            broken_bonds=broken_bonds,
            changed_bonds=changed_bonds,
            r_eq_atoms=r_eq_atoms,
            weights=weights_array,
            amplitude=amp)
        if nmd_correct:
            return True
    return False

def check_bond_directionality(formed_bonds: list[tuple[int, int]],
                              broken_bonds: list[tuple[int, int]],
                              xyzs: tuple[dict, dict],
                              min_delta: float = DIRECTIONALITY_MIN_DELTA,
                              ) -> bool:
    """
    Check that formed and broken bonds move in opposite directions along the normal mode.

    For a valid TS mode, formed bonds should shorten in one displacement direction
    while broken bonds should lengthen in the same direction (and vice versa).
    Only bonds with ``|delta| > min_delta`` are checked to avoid false failures from numerical noise.

    Args:
        formed_bonds (list[tuple[int, int]]): The bonds that are formed in the reaction.
        broken_bonds (list[tuple[int, int]]): The bonds that are broken in the reaction.
        xyzs (tuple[dict, dict]): The Cartesian coordinates of the TS displaced along the normal mode.
        min_delta (float): Minimum absolute signed difference for a bond to participate in the check.

    Returns:
        bool: Whether the bond directionality is consistent with a reactive mode.
    """
    if not formed_bonds and not broken_bonds:
        return True

    def _get_signed_deltas(bonds):
        deltas = []
        for bond in bonds:
            # Use unweighted distances for directionality to use the min_delta threshold.
            d1 = get_bond_length_in_reaction(bond=bond, xyz=xyzs[0], weights=None)
            d2 = get_bond_length_in_reaction(bond=bond, xyz=xyzs[1], weights=None)
            if d1 is None or d2 is None:
                continue
            delta = d1 - d2
            if abs(delta) > min_delta:
                deltas.append(delta)
        return deltas

    deltas_formed = _get_signed_deltas(formed_bonds)
    deltas_broken = _get_signed_deltas(broken_bonds)

    if not deltas_formed and not deltas_broken:
        logger.debug('check_bond_directionality: no bonds exceeded min_delta threshold; '
                     'returning True (vacuously consistent).')

    # Check internal consistency: all formed bonds must move in the same direction
    if deltas_formed and not all(np.sign(d) == np.sign(deltas_formed[0]) for d in deltas_formed):
        return False

    # Check internal consistency: all broken bonds must move in the same direction
    if deltas_broken and not all(np.sign(d) == np.sign(deltas_broken[0]) for d in deltas_broken):
        return False

    # Check that formed and broken bonds move in opposite directions
    if deltas_formed and deltas_broken:
        if np.sign(deltas_formed[0]) == np.sign(deltas_broken[0]):
            return False

    return True

def is_nmd_correct_for_any_mapping(reaction: 'ARCReaction',
                                   xyzs: tuple[dict, dict],
                                   formed_bonds: list[tuple[int, int]],
                                   broken_bonds: list[tuple[int, int]],
                                   changed_bonds: list[tuple[int, int]],
                                   r_eq_atoms: list[list[int]],
                                   weights: np.ndarray | None,
                                   amplitude: float,
                                   ) -> bool:
    """
    Check if the normal mode displacement is consistent with the desired reaction for any atom mapping.

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        xyzs (tuple[dict, dict]): The Cartesian coordinates of the TS displaced along the normal mode.
        formed_bonds (list[tuple[int, int]]): The bonds that are formed in the reaction.
        broken_bonds (list[tuple[int, int]]): The bonds that are broken in the reaction.
        changed_bonds (list[tuple[int, int]]): The bonds that are changed in the reaction.
        r_eq_atoms (list[list[int]]): A list of equivalent atoms in the reactants.
        weights (np.array): The weights for the atoms.
        amplitude (float): The motion amplitude.

    Returns:
        bool: Whether the TS normal mode displacement is consistent with the desired reaction for any atom mapping.
    """
    modified_bond_grand_list = get_eq_formed_and_broken_bonds(formed_bonds=formed_bonds,
                                                              broken_bonds=broken_bonds,
                                                              changed_bonds=changed_bonds,
                                                              r_eq_atoms=r_eq_atoms,
                                                              )
    for eq_formed_bonds, eq_broken_bonds, eq_changed_bonds in modified_bond_grand_list:
        if not check_bond_directionality(formed_bonds=eq_formed_bonds,
                                         broken_bonds=eq_broken_bonds,
                                         xyzs=xyzs):
            continue

        primary_bonds = eq_formed_bonds + eq_broken_bonds
        is_isomerization = not primary_bonds and bool(eq_changed_bonds)

        if is_isomerization:
            check_bonds = eq_changed_bonds
        elif primary_bonds:
            check_bonds = primary_bonds
        else:
            continue

        check_diffs, report = get_bond_length_changes(bonds=check_bonds,
                                                      xyzs=xyzs,
                                                      weights=weights,
                                                      amplitude=amplitude,
                                                      return_none_if_change_is_insignificant=True,
                                                      considered_reactive=True,
                                                      )
        if check_diffs is None or len(check_diffs) == 0:
            continue

        r_bonds, _ = reaction.get_bonds(r_bonds_only=True)
        reactive_bonds = set(eq_formed_bonds) | set(eq_broken_bonds) | set(eq_changed_bonds)
        non_reactive_bonds = [bond for bond in r_bonds if bond not in reactive_bonds]
        baseline, std = get_bond_length_changes_baseline_and_std(non_reactive_bonds=non_reactive_bonds,
                                                                 xyzs=xyzs,
                                                                 weights=weights,
                                                                 )

        min_check_diff = float(np.min(check_diffs))

        if baseline is None:
            # Small molecule case: Get UNWEIGHTED changes to compare against a physical distance
            unweighted_diffs, _ = get_bond_length_changes(
                bonds=check_bonds,
                xyzs=xyzs,
                weights=None,
                amplitude=amplitude,
                return_none_if_change_is_insignificant=False
            )
            if unweighted_diffs is not None and len(unweighted_diffs) > 0:
                # For small molecules without a baseline, check that the minimum reactive
                # bond-length change exceeds 10% of the displacement amplitude.
                if np.min(unweighted_diffs) > 0.1 * amplitude:
                    return True
            continue

        std = max(std, STD_FLOOR)
        sigma = (min_check_diff - baseline) / std
        if sigma > SIGMA_THRESHOLD:
            return True
    return False

def get_eq_formed_and_broken_bonds(formed_bonds: list[tuple[int, int]],
                                   broken_bonds: list[tuple[int, int]],
                                   changed_bonds: list[tuple[int, int]],
                                   r_eq_atoms: list[list[int]],
                                   ) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]]:
    """
    Get the equivalent formed and broken bonds.

    Args:
         formed_bonds (list[tuple[int, int]]): The bonds that are formed in the reaction.
         broken_bonds (list[tuple[int, int]]): The bonds that are broken in the reaction.
         changed_bonds (list[tuple[int, int]]): The bonds that are changed in the reaction.
         r_eq_atoms (list[list[int]]): A list of equivalent atoms in the reactants.

    Returns:
         list[tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]]: The equivalent formed and broken bonds.
    """
    all_changing_indices = list()
    for bond in formed_bonds + broken_bonds:
        all_changing_indices.extend([bond[0], bond[1]])
    all_changing_indices = list(set(all_changing_indices))
    r_eq_atoms = [eq for eq in r_eq_atoms if any(i in eq for i in all_changing_indices)]
    modified_bond_grand_list = translate_all_tuples_simultaneously(list_1=formed_bonds,
                                                                   list_2=broken_bonds,
                                                                   list_3=changed_bonds,
                                                                   equivalences=r_eq_atoms)
    return modified_bond_grand_list

def translate_all_tuples_simultaneously(list_1: list[tuple[int, int]],
                                        list_2: list[tuple[int, int]],
                                        list_3: list[tuple[int, int]],
                                        equivalences: list[list[int]],
                                        ) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]]:
    """
    Translate all tuples simultaneously using a mapping.

    Args:
        list_1 (list[tuple[int, int]]): The first list of tuples.
        list_2 (list[tuple[int, int]]): The second list of tuples.
        list_3 (list[tuple[int, int]]): The third list of tuples.
        equivalences (list[list[int]]): A list of equivalent atoms.

    Returns:
        list[tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]]: The translated tuples.
    """
    mapping = create_equivalence_mapping(equivalences)
    all_indices = {i for tup in list_1 + list_2 + list_3 for i in tup}
    translated_options = {i: mapping.get(i, [i]) for i in all_indices}
    translated_combinations = list(product(*(translated_options[i] for i in all_indices)))
    index_to_pos = {val: idx for idx, val in enumerate(all_indices)}
    all_translated_tuples = list()
    for combo in translated_combinations:
        combo_dict = {i: combo[index_to_pos[i]] for i in all_indices}
        translated_list_1 = [(combo_dict[a], combo_dict[b]) for (a, b) in list_1]
        translated_list_2 = [(combo_dict[a], combo_dict[b]) for (a, b) in list_2]
        translated_list_3 = [(combo_dict[a], combo_dict[b]) for (a, b) in list_3]
        if len(translated_list_1) != len(set(translated_list_1)):
            continue
        if len(translated_list_2) != len(set(translated_list_2)):
            continue
        if len(translated_list_3) != len(set(translated_list_3)):
            continue
        all_translated_tuples.append((translated_list_1, translated_list_2, translated_list_3))
    return all_translated_tuples

def create_equivalence_mapping(equivalences: list[list[int]]) -> dict[int, list[int]]:
    """
    Create a mapping of atom indices to equivalence groups.

    Args:
        equivalences (list[list[int]]): A list of equivalent atoms.

    Returns:
        dict[int, list[int]]: The mapping of atom indices to equivalence groups
    """
    mapping = dict()
    for group in equivalences:
        for item in group:
            mapping[item] = group
    return mapping

def get_weights_from_xyz(xyz: dict,
                         weights: bool | np.array = True,
                         ) -> np.array:
    """
    Get weights for atoms in a molecule.

    Args:
        xyz (dict): The Cartesian coordinates.
        weights (bool | np.array): Whether to use weights for the displacement.
                                         If ``False``, use ones as weights. If ``True``, use sqrt of atom masses.
                                         If an array, use it as weights.

    Returns:
        np.array: The weights for the atoms.
    """
    if isinstance(weights, bool):
        if weights:
            symbols = list()
            isotopes = [None] * len(xyz['symbols'])
            for i, symbol in enumerate(xyz['symbols']):
                symbols.append(symbol)
                if 'isotopes' in xyz.keys() and xyz['isotopes'][i] != get_most_common_isotope_for_element(symbol):
                    isotopes[i] = xyz['isotopes'][i]
            atom_masses = np.array([get_element_mass(symbol)[0] for symbol in symbols]).reshape(-1, 1)
            weights = np.sqrt(atom_masses)
        else:
            weights = np.ones((len(xyz['symbols']), 1))
    return weights

def get_displaced_xyzs(xyz: dict,
                       amplitude: float,
                       normal_mode_disp: np.array,
                       weights: np.array,
                       ) -> tuple[dict, dict]:
    """
    Get the Cartesian coordinates of the TS displaced along a normal mode.

    Args:
        xyz (dict): The Cartesian coordinates.
        amplitude (float): The amplitude of the displacement.
        normal_mode_disp (np.array): The normal mode displacement matrix corresponding to the imaginary frequency.
        weights (np.array): The weights for the atoms.

    Returns:
        tuple[dict, dict]: The Cartesian coordinates of the TS displaced along the normal mode.
    """
    np_coords = xyz_to_np_array(xyz)
    xyz_1 = xyz_from_data(coords=np_coords - amplitude * normal_mode_disp * weights,
                          symbols=xyz['symbols'], isotopes=xyz['isotopes'])
    xyz_2 = xyz_from_data(coords=np_coords + amplitude * normal_mode_disp * weights,
                          symbols=xyz['symbols'], isotopes=xyz['isotopes'])
    return xyz_1, xyz_2

def get_bond_length_changes_baseline_and_std(non_reactive_bonds: list[tuple[int, int]],
                                             xyzs: tuple[dict, dict],
                                             weights: np.array | None = None,
                                             ) -> tuple[float | None, float | None]:
    """
    Get the baseline and spread of bond length changes for non-reactive bonds using robust statistics.

    Uses the median as the baseline and MAD (median absolute deviation) scaled by 1.4826
    as the spread estimator, which is robust against outliers from floppy rotors.

    Todo:
        When we have a comprehensive list of atom maps, we can pass the reaction and the atom map number to use, and do:
            non_reactive_bonds = set(r_bonds) & set(p_bonds)

    Args:
        non_reactive_bonds (list[tuple[int, int]]): The non-reactive bonds.
        xyzs (tuple[dict, dict]): The Cartesian coordinates of the TS displaced along the normal mode.
        weights (np.array): The weights for the atoms.

    Returns:
        tuple[float | None, float | None]:
            - The median baseline of bond length differences for non-reactive bonds.
            - The MAD-based spread estimate of bond length differences for non-reactive bonds.
    """
    diffs, _ = get_bond_length_changes(bonds=non_reactive_bonds, xyzs=xyzs, weights=weights)
    if diffs is None or len(diffs) == 0:
        return None, None
    baseline = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - baseline)))
    std = mad * 1.4826  # scale factor for consistency with normal distribution
    return baseline, std

def get_bond_length_changes(bonds: list[tuple[int, int]] | set[tuple[int, int]],
                            xyzs: tuple[dict, dict],
                            weights: np.array | None = None,
                            amplitude: float | None = None,
                            return_none_if_change_is_insignificant: bool = False,
                            considered_reactive: bool = False,
                            ) -> np.array | None:
    """
    Get the bond length changes of specific bonds.

    Args:
        bonds (list | tuple): The bonds to check.
        xyzs (tuple[dict, dict]): The Cartesian coordinates of the TS displaced along the normal mode.
        weights (np.array, optional): The weights for the atoms.
        amplitude (float | None): The motion amplitude.
        return_none_if_change_is_insignificant (bool, optional): Whether to check for significant motion
                                                                 and return None if motion is insignificant.
                                                                 Relevant for bonds that change during a reaction,
                                                                 not for the background.
        considered_reactive (bool): Whether the bonds are considered reactive in the reaction.

    Returns:
        np.array | None: The bond length changes of the specified bonds.
    """
    diffs = list()
    report = None
    amplitude = amplitude or 1.0

    for bond in bonds:
        r_bond_length = get_bond_length_in_reaction(bond=bond, xyz=xyzs[0], weights=weights)
        p_bond_length = get_bond_length_in_reaction(bond=bond, xyz=xyzs[1], weights=weights)
        if r_bond_length is None or p_bond_length is None:
            continue
        diff = abs(r_bond_length - p_bond_length)
        if return_none_if_change_is_insignificant \
                and abs(diff / r_bond_length) < 0.05 \
                and abs(diff / p_bond_length) < 0.05:
            return None, None
        diffs.append(diff)
        if considered_reactive:
            report = f'{float(r_bond_length):.2f} {float(p_bond_length):.2f} {float(diff):.2f} {amplitude}'
    diffs = np.array(diffs)
    return diffs, report

def get_bond_length_in_reaction(bond: tuple[int, int] | list[int],
                                xyz: dict,
                                weights: np.array | None = None,
                                ) -> float | None:
    """
    Get the length of a bond in either the reactants or the products of a reaction.

    Args:
        bond (tuple[int, int]): The bond to check.
        xyz (dict): The Cartesian coordinates mapped to the indices of the reactants.
        weights (np.array): The weights for the atoms.

    Returns:
        float: The bond length.
    """
    try:
        distance = calculate_distance(coords=xyz, atoms=bond, index=0)
    except (VectorsError, TypeError, IndexError):
        return None
    if weights is not None:
        distance *= np.sqrt(weights[bond[0]] * weights[bond[1]])
    if isinstance(distance, np.ndarray):
        return distance.item()
    return float(distance)

def find_equivalent_atoms(reaction: 'ARCReaction',
                          reactant_only: bool = True,
                          ) -> tuple[list[list[int]], list[list[int]]]:
    """
    Find equivalent atoms in the reactants and products of a reaction.
    This is a tentative function that should be replaced when atom mapping returns a list.
    It is meant to suggest additional atoms that can move instead of the ones selected by the atom map.

    Args:
        reaction (ARCReaction): The reaction for which equivalent atoms are searched.
        reactant_only (bool): Whether to search for equivalent atoms in the reactants only.

    Returns:
        tuple[list[list[int]], list[list[int]]]:
            - A list of equivalent atoms in the reactants.
            - A list of equivalent atoms in the products, indices are atom mapped to the reactants.
    """
    reactants, products = reaction.get_reactants_and_products(return_copies=True)
    r_eq_atoms, p_eq_atoms = list(), list()
    for i, reactant in enumerate(reactants):
        r_eq_atoms.extend(identify_equivalent_atoms_in_molecule(molecule=reactant.mol,
                                                                inc=sum([len(r.mol.atoms) for r in reactants[:i]]),
                                                                atom_map=None,
                                                                ))
    if not reactant_only:
        for i, product in enumerate(products):
            p_eq_atoms.extend(identify_equivalent_atoms_in_molecule(molecule=product.mol,
                                                                    inc=sum([len(p.mol.atoms) for p in products[:i]]),
                                                                    atom_map=reaction.atom_map,
                                                                    ))
    return r_eq_atoms, p_eq_atoms

def identify_equivalent_atoms_in_molecule(molecule: 'Molecule',
                                          inc: int = 0,
                                          atom_map: list[int] | None = None,
                                          ) -> list[list[int]]:
    """
    Identify equivalent atoms in a molecule.

    Args:
        molecule (Molecule): The molecule to check.
        inc (int): The increment to be added.
        atom_map (list[int] | None): The atom map.

    Returns:
        list[list[int]]: A list of equivalent atoms.
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
                     excluded_atom_indices: list[int] | None = None,
                     depth: int = 3,
                     ) -> list:
    """
    Fingerprint an atom in a molecule.

    Args:
        atom_index (int): The index of the atom to map.
        molecule (Molecule): The molecule to which the atom belongs.
        excluded_atom_indices (list[int] | None): Atom indices to exclude from the mapping.
        depth (int): The depth of the atom map.

    Returns:
        list[int]: The atom map.
    """
    atom_0 = molecule.atoms[atom_index]
    fingerprint = [atom_0.element.number]
    if depth == 0:
        return fingerprint
    excluded_atom_indices = excluded_atom_indices or list()
    excluded_atom_indices = excluded_atom_indices + [atom_index]
    neighbors = list()
    for atom_1 in sorted(atom_0.edges.keys(), key=lambda x: x.element.number):
        atom_1_index = molecule.atoms.index(atom_1)
        if atom_1_index in excluded_atom_indices:
            continue
        if depth == 1 or len(atom_1.edges) == 1:
            neighbors.append(atom_1.element.number)
        else:
            sub_fingerprint = fingerprint_atom(atom_index=atom_1_index,
                                               molecule=molecule,
                                               excluded_atom_indices=excluded_atom_indices,
                                               depth=depth - 1)
            neighbors.append(sub_fingerprint)
    sorted_neighbors = sorted([([n] if isinstance(n, int) else n) for n in neighbors],
                              key=lambda x: x[0] if isinstance(x, list) else x)
    fingerprint.extend(sorted_neighbors)
    return fingerprint
