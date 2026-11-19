"""
A module for creating hydrogen bond structures of water around an ion, currently implemented for OH-.
Creating a comprehensive list of structures/conformations, and running ARC to get the Gibbs Free Energy of main ones.
"""

from typing import Dict, List, Optional

import copy

from arc.species.converter import check_xyz_dict, check_zmat_dict, zmat_from_xyz


# The electronegative ion to which water molecules will H-bond should be atom 0 in the xyz dictionary
KNOWN_IONS = {'OH-': {'name': 'hydroxide',
                      'charge': -1,
                      'multiplicity': 1,
                      'SMILES': '[OH-]',
                      'comment': 'wb97xd/jul-cc-pvtz SMD DMSO',
                      'xyz': """O                  0.00000000    0.00000000    0.10663100
                                H                  0.00000000    0.00000000   -0.85305000"""},
              }
R_H_BOND = 1.5  # ???
WATER_R_H_O = 0.97
WATER_A_H_O_H = 109.5


def create_hb_structures(hydration_level: int,
                         ion: str = 'OH-',
                         e_threshold: float = 100,
                         min_hb_per_ion: Optional[int] = None,
                         t_list: Optional[List[int]] = None,
                         ) -> List[Dict]:
    """
    Create hydrogen bonded structures of water surrounding an ion.

    Args:
        hydration_level (int): The number of water molecules per ion molecule.
        ion (str, optional): The ion to create the network around. Default is 'OH-'.
        e_threshold (float, optional): The energy threshold for the optimization. Default is 100. Units: kJ/mol
        min_hb_per_ion (int, optional): The minimum number of hydrogen bonds per ion molecule. Default is the min(hydration_level, 2).
        t_list (list, optional): A list of temperatures to run ARC at. Default is [298].

    Returns:
        List[dict]: A list of dictionaries, each containing the following keys:
                    - 'xyz' (dict): The xyz of the structure.
                    - 'n_ions' (int): The number of ion molecules in the network.
                    - 'hb_per_ion' (Tuple[int]): The number of hydrogen bonds per ion molecule.
                    - 'Gs': The Gibbs Free Energy of the structure for each temperature.
    """
    check_args(hydration_level, ion, e_threshold, min_hb_per_ion, t_list)

    ion_zmat = get_ion_zmat(ion)
    all_structures = list()
    for hb_num_per_ion in range(min_hb_per_ion, hydration_level, 1):
        xyz = get_origin_structure(ion_zmat, hydration_level, hb_num_per_ion)
        structures = generate_combinations(xyz)
        structures = optimize_structures(structures, e_threshold)
        all_structures.extend(structures)

    run_arc(all_structures)

    post_process


def check_args(hydration_level: int,
               ion: str,
               e_threshold: float,
               min_hb_per_ion: Optional[int],
               t_list: Optional[List[int]],
               ) -> None:
    """
    Check the input arguments for the create_hb_network function.

    Args:
        hydration_level (int): The number of water molecules per ion molecule.
        ion (str): The ion to create the network around.
        e_threshold (float): The energy threshold for the optimization.
        min_hb_per_ion (int, optional): The minimum number of hydrogen bonds per ion molecule.
        t_list (list, optional): A list of temperatures to run ARC at.

    Raises:
        ValueError: If any of the input arguments are invalid.
    """
    if hydration_level < 1:
        raise ValueError(f'Invalid hydration_level: {hydration_level}. Must be at least 1.')
    if hydration_level > 8:
        raise ValueError(f'Invalid hydration_level: {hydration_level}. Must be at most 8.')
    if ion not in KNOWN_IONS:
        raise ValueError(f'Invalid ion: {ion}. Must be one of {KNOWN_IONS.keys()}.')
    if e_threshold < 1:
        raise ValueError(f'Invalid e_threshold: {e_threshold}. Must be at least 1 kJ/mol.')
    if min_hb_per_ion > hydration_level:
        raise ValueError(f'Invalid min_hb_per_ion: {min_hb_per_ion}. Must be at most the hydration_level ({hydration_level}).')
    if min_hb_per_ion < 1:
        raise ValueError(f'Invalid min_hb_per_ion: {min_hb_per_ion}. Must be at least 1.')
    if not isinstance(t_list, list):
        raise ValueError(f'Invalid t_list: {t_list}. Must be a list of floats.')
    if len(t_list) == 0:
        raise ValueError(f'Invalid t_list: {t_list}. Must contain at least one temperature.')
    for t in t_list:
        if not isinstance(t, (int, float)):
            raise ValueError(f'Invalid temperature: {t}. Must be a float.')
        if t < 0:
            raise ValueError(f'Invalid temperature: {t}. Must be at least 0 K.')


def get_ion_zmat(ion: str) -> dict:
    """
    Get the zmat of the ion.

    Args:
        ion (str): The ion to get the zmat of.

    Returns:
        dict: The zmat of the ion.
    """
    ion_xyz = check_xyz_dict(KNOWN_IONS[ion]['xyz'])
    ion_zmat = zmat_from_xyz(ion_xyz)
    return ion_zmat


def get_origin_structure(ion_zmat: dict,
                         hydration_level: int,
                         hb_num_per_ion: int,
                         ) -> dict:
    """
    Get the xyz of the origin structure as a basis for further structures.

    Args:
        ion_zmat (dict): The zmat of the ion.
        hydration_level (int): The number of water molecules per ion molecule.
        hb_num_per_ion (int): The number of hydrogen bonds per ion molecule.

    Returns:
        dict: The xyz of the origin structure.
    """
    xyz = get_skeletal_structure(ion_zmat=ion_zmat, num_w_mols=hb_num_per_ion)
    if hydration_level > hb_num_per_ion:
        xyz = add_non_hb_water_molecules(xyz=xyz, num=hydration_level - hb_num_per_ion)
    return xyz


def get_skeletal_structure(ion_zmat: dict,
                           num_w_mols: int,
                           ) -> dict:
    """
    Get the skeletal structure of the network.

    Args:
        ion_zmat (dict): The zmat of the ion.
        num_w_mols (int): The number of water molecules to add.

    Returns:
        dict: The zmat of the skeletal structure.
    """
    beta = 110  # The angle between the water molecule's H atom and the ion
    zmat = copy.deepcopy(ion_zmat)
    for i in range(num_w_mols):
        dihedarl = i * 360.0 / num_w_mols
        zmat = add_water_molecule(zmat, ion_zmat, dihedarl, beta)
    return zmat


def add_water_molecule(zmat: dict,
                       ion_zmat: dict,
                       dihedarl: float,
                       beta: float,
                       ) -> dict:
    """
    Add a water molecule to the network.

    Args:
        zmat (dict): The zmat of the network.
        ion_zmat (dict): The zmat of the ion.
        dihedarl (float): The dihedral angle between the water molecule's O atom and the ion.
        beta (float): The angle between the water molecule's H atom and the ion.

    Returns:
        dict: The zmat of the structure with the added water molecule.
    """
    zmat = check_zmat_dict(zmat)
    oh_o_index = 0
    oh_h_index = 1
    last_added_o_index = get_last_added_water_o_index(zmat, ion_zmat)
    num_atoms = len(zmat['symbols'])
    h2o_h1_index = num_atoms
    h2o_o_index = num_atoms + 1
    h2o_h2_index = num_atoms + 2
    zmat['symbols'] = zmat['symbols'] + ('H', 'O', 'H')
    zmat['coords'] = zmat['coords'] + ((f'R_{h2o_h1_index}_{oh_o_index}',
                                        f'A_{h2o_h1_index}_{oh_o_index}_{oh_h_index}',
                                        f'D_{h2o_h1_index}_{oh_o_index}_{oh_h_index}_{last_added_o_index}' if last_added_o_index is not None else None),
                                       (f'R_{h2o_o_index}_{oh_o_index}',
                                        f'A_{h2o_o_index}_{oh_o_index}_{oh_h_index}',
                                        f'D_{h2o_o_index}_{oh_o_index}_{oh_h_index}_{last_added_o_index or h2o_h1_index}'),
                                       (f'R_{h2o_h2_index}_{h2o_o_index}',
                                        f'A_{h2o_h2_index}_{h2o_o_index}_{h2o_h1_index}',
                                        f'D_{h2o_h2_index}_{h2o_o_index}_{h2o_h1_index}_{oh_h_index}'))
    zmat['vars'][f'R_{h2o_h1_index}_{oh_o_index}'] = R_H_BOND
    zmat['vars'][f'A_{h2o_h1_index}_{oh_o_index}_{oh_h_index}'] = beta
    zmat['vars'][f'R_{h2o_o_index}_{oh_o_index}'] = R_H_BOND + WATER_R_H_O
    zmat['vars'][f'A_{h2o_o_index}_{oh_o_index}_{oh_h_index}'] = beta
    zmat['vars'][f'D_{h2o_o_index}_{oh_o_index}_{oh_h_index}_{last_added_o_index or h2o_h1_index}'] = dihedarl
    zmat['vars'][f'R_{h2o_h2_index}_{h2o_o_index}'] = WATER_R_H_O
    zmat['vars'][f'A_{h2o_h2_index}_{h2o_o_index}_{h2o_h1_index}'] = WATER_A_H_O_H
    zmat['vars'][f'D_{h2o_h2_index}_{h2o_o_index}_{h2o_h1_index}_{oh_h_index}'] = 90
    if last_added_o_index is not None:
        zmat['vars'][f'D_{h2o_h1_index}_{oh_o_index}_{oh_h_index}_{last_added_o_index}'] = dihedarl
    for entry in [h2o_h1_index, h2o_o_index, h2o_h2_index]:
        zmat['map'][entry] = entry
    return zmat


def get_last_added_water_o_index(zmat: dict,
                                 ion_xyz: dict,
                                 ) -> Optional[int]:
    """
    Get the index of the last added water oxygen atom.

    Args:
        zmat (dict): The xyz of the network.
        ion_xyz (dict): The xyz of the ion.

    Returns:
        Optional[int]: The index of the last added water oxygen atom, or None if no water molecules were added yet.
    """
    num_ion_atoms = len(ion_xyz['symbols'])
    for i, symbol in enumerate(zmat['symbols']):
        if symbol == 'O' and i > num_ion_atoms:
            return i
    return None
