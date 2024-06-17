"""
A module for checking the normal mode displacement of a TS.
"""

import numpy as np
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

import qcelemental as qcel

import arc.rmgdb as rmgdb
from arc import parser
from arc.common import get_logger
from arc.imports import settings
from arc.species.converter import get_most_common_isotope_for_element, xyz_from_data, xyz_to_np_array
from arc.species.vectors import calculate_distance, VectorsError

if TYPE_CHECKING:
    from arc.job.adapter import JobAdapter
    from arc.reaction import ARCReaction

logger = get_logger()

LOWEST_MAJOR_TS_FREQ, HIGHEST_MAJOR_TS_FREQ = settings['LOWEST_MAJOR_TS_FREQ'], settings['HIGHEST_MAJOR_TS_FREQ']


def analyze_ts_normal_mode_displacement(reaction: 'ARCReaction',
                                        job: Optional['JobAdapter'],
                                        amplitude: Union[float, list] = 0.25,
                                        weights: Union[bool, np.array] = True,
                                        ) -> Optional[bool]:
    """
    Analyze the normal mode displacement by identifying bonds that break and form
    and comparing them to the expected RMG template, if available.

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        job (JobAdapter): The frequency job object instance that points to the respective log file.
        amplitude (Union[float, list]): The amplitude of the normal mode displacement motion to check.
                                         If a list, all possible results are returned.
        weights (Union[bool, np.array]): Whether to use weights for the displacement.
                                         If ``False``, use ones as weights. If ``True``, use sqrt of atom masses.
                                         If an array, use it as weights.

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
    for amp in amplitude:
        if not amp:
            continue
        xyzs = get_displaced_xyzs(xyz=ts_xyz, amplitude=amp, normal_mode_disp=normal_mode_disp[0], weights=weights)
        reacting_bonds_diffs = get_bond_length_changes_of_reacting_bonds(reaction=reaction, xyzs=xyzs)  # need weights??
        baseline, std = get_bond_length_changes_baseline_and_std(reaction=reaction, xyzs=xyzs)
        print(f'\n\n\nmin: {np.min(reacting_bonds_diffs)}\nbaseline: {baseline}\nstd: {std}\n\n\n')
        sigma = (np.min(reacting_bonds_diffs) - baseline) / std
        if sigma > 3:
            return True
    return False


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

    if isinstance(weights, bool) and weights:
        atom_masses = np.array([qcel.periodictable.to_mass(symbol) for symbol in symbols]).reshape(-1, 1)
        weights = np.sqrt(atom_masses)
    elif isinstance(weights, bool) and not weights:
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


def get_bond_length_changes_baseline_and_std(reaction: 'ARCReaction',
                                             xyzs: Tuple[dict, dict],
                                             ) -> Tuple[float, float]:
    """
    Get the baseline for bond length change of all non-reacting bonds.

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        xyzs (Tuple[dict, dict]): The Cartesian coordinates of the TS displaced along the normal mode.

    Returns:
        Tuple[float, float]:
            - The max baseline of bond length differences for non-reactive bonds.
            - The standard deviation of bond length differences for non-reactive bonds.
    """
    r_bonds, p_bonds = reaction.get_bonds()
    non_reactive_bonds = set(r_bonds) & set(p_bonds)
    diffs = get_bond_length_changes(bonds=non_reactive_bonds, xyzs=xyzs)
    baseline_max = np.max(diffs)
    std = np.std(diffs)
    return baseline_max, std


def get_bond_length_changes_of_reacting_bonds(reaction: 'ARCReaction',
                                              xyzs: Tuple[dict, dict],
                                              ) -> np.array:
    """
    Get the bond length changes of all reacting bonds.

    Args:
        reaction (ARCReaction): The reaction for which the TS is checked.
        xyzs (Tuple[dict, dict]): The Cartesian coordinates of the TS displaced along the normal mode.

    Returns:
        np.array: The bond length changes of all reacting bonds.
    """
    formed_bonds, broken_bonds = reaction.get_formed_and_broken_bonds()
    return get_bond_length_changes(bonds=formed_bonds + broken_bonds, xyzs=xyzs)


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
        xyz (dict): The Cartesian coordinates.

    Returns:
        float: The bond length.
    """
    try:
        distance = calculate_distance(coords=xyz, atoms=bond, index=0)
    except (VectorsError, TypeError, IndexError):
        return None
    return distance
