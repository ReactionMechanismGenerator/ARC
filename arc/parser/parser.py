"""
A module for parsing information from various files.
"""

import os
from typing import Dict, List, Optional, Tuple


from arc.common import get_logger, read_yaml_file
from arc.exceptions import InputError, ParserError
from arc.parser.factory import ess_factory
from arc.species.converter import str_to_xyz


logger = get_logger()


def determine_ess(log_file_path: str,
                  raise_error: bool = True,
                  ) -> Optional[str]:
    """
    Determine the ESS that generated a specific output file.

    Args:
        log_file_path (str): The disk location of the output file of interest.
        raise_error (bool): Whether to raise an error if the ESS cannot be determined.

    Returns: Optional[str]
        The ESS name, e.g., 'gaussian', 'molpro', 'orca', 'qchem', 'terachem', or 'psi4'. ``None`` if unknown.
    """
    ess_name = None
    if log_file_path.endswith('.yml') or log_file_path.endswith('.yaml'):
        content = read_yaml_file(log_file_path)
        ess_name = content.get('adapter', 'yaml')
        return ess_name
    if os.path.splitext(log_file_path)[-1] in ['.xyz', '.dat', '.geometry']:
        ess_name = 'terachem'
        return ess_name
    with open(log_file_path, 'r') as f:
        line = f.readline().lower()
        while ess_name is None and line != '':
            if 'gaussian' in line:
                ess_name = 'gaussian'
                break
            elif 'molpro' in line:
                ess_name = 'molpro'
                break
            elif 'o   r   c   a' in line or 'orca' in line:
                ess_name = 'orca'
                break
            elif 'psi4' in line or 'rob parrish' in line:
                ess_name = 'psi4'
                break
            elif 'qchem' in line:
                ess_name = 'qchem'
                break
            elif 'terachem' in line:
                ess_name = 'terachem'
                break
            elif 'x T B' in line:
                ess_name = 'xtb'
                break
            line = f.readline().lower()
    if ess_name is None:
        if raise_error:
            raise InputError(f'The ESS that generated the file at {log_file_path} could not be identified.')
        else:
            return None
    return ess_name


def parse_xyz_from_file(path: str) -> Optional[Dict[str, tuple]]:
    """
    Parse xyz coordinated from:
    - .xyz: XYZ file
    - .gjf: Gaussian input file
    - .out or .log: ESS output file (Gaussian, Molpro, Orca, QChem, TeraChem) - calls parse_geometry()
    - .yml or .yaml files
    - other: Molpro or QChem input file

    Args:
        path (str): The file path.

    Raises:
        ParserError: If the coordinates could not be parsed.

    Returns: Optional[Dict[str, tuple]]
        The parsed cartesian coordinates.
    """
    if not os.path.isfile(path):
        raise InputError(f'Could not find file {path}')
    if path.endswith('.yml'):
        content = read_yaml_file(path)
        if isinstance(content, dict) and 'xyz' in content.keys():
            return content['xyz']

    lines = _get_lines_from_file(path)
    file_extension = os.path.splitext(path)[1]

    xyz = None
    relevant_lines = list()

    if file_extension == '.xyz':
        for i, line in enumerate(reversed(lines)):
            splits = line.strip().split()
            if len(splits) == 1 and all([c.isdigit() for c in splits[0]]):
                # this is the last number of atoms line (important when parsing trajectories)
                num_of_atoms = int(splits[0])
                break
        else:
            raise ParserError(f'Could not identify the number of atoms line in the xyz file {path}')
        index = len(lines) - i - 1
        relevant_lines = lines[index + 2: index + 2 + num_of_atoms]
    elif file_extension == '.gjf':
        start_parsing = False
        for line in lines:
            if start_parsing and line and line != '\n' and line != '\r\n':
                relevant_lines.append(line)
            elif start_parsing:
                break
            else:
                splits = line.split()
                if len(splits) == 2 and all([s.isdigit() for s in splits]):
                    start_parsing = True
    elif 'out' in file_extension or 'log' in file_extension:
        xyz = ess_factory(path).parse_geometry()
    elif "yml" in file_extension or 'yaml' in file_extension:
        content = read_yaml_file(path=path)
        if "xyz" in content.keys():
            xyz = content["xyz"]
        elif "opt_xyz" in content.keys():
            xyz = content["opt_xyz"]
        if isinstance(xyz, str):
            xyz = str_to_xyz(xyz)
    else:
        record = False
        for line in lines:
            if '$end' in line or '}' in line:
                break
            if record and len(line.split()) == 4:
                relevant_lines.append(line)
            elif '$molecule' in line:
                record = True
            elif 'geometry={' in line:
                record = True
        if not relevant_lines:
            raise ParserError(f'Could not parse xyz coordinates from file {path}')
    if xyz is None and relevant_lines:
        xyz = str_to_xyz(''.join([line for line in relevant_lines if line]))
    return xyz


def 1parse_trajectory(path: str) -> Optional[List[Dict[str, tuple]]]:
    """
    Parse all geometries from an xyz trajectory file or an ESS output file.

    Args:
        path (str): The file path.

    Raises:
        ParserError: If the trajectory could not be read.

    Returns: Optional[List[Dict[str, tuple]]]
        Entries are xyz's on the trajectory.
    """
    ess_file = False
    if path.split('.')[-1] != 'xyz':
        ess_file = determine_ess(log_file_path=path, raise_error=False)
        ess_file = bool(ess_file)
    if ess_file:
        traj = ess_factory(log_file_path=path).parse_1d_scan_coords()
    else:
        # this is not an ESS output file, probably an XYZ format file with several Cartesian coordinates
        skip_line = False
        num_of_atoms = 0
        traj, xyz_lines = list(), list()
        lines = _get_lines_from_file(path)
        for line in lines:
            splits = line.strip().split()
            if len(splits) == 1 and all([c.isdigit() for c in splits[0]]):
                if len(xyz_lines):
                    if len(xyz_lines) != num_of_atoms:
                        raise ParserError(f'Could not parse trajectory, expected {num_of_atoms} atoms, '
                                          f'but got {len(xyz_lines)} for point {len(traj) + 1} in the trajectory.')
                    traj.append(str_to_xyz(''.join([xyz_line for xyz_line in xyz_lines])))
                num_of_atoms = int(splits[0])
                skip_line = True
                xyz_lines = list()
            elif skip_line:
                # skip the comment line
                skip_line = False
                continue
            else:
                xyz_lines.append(line)
        if len(xyz_lines):
            # add the last point in the trajectory
            if len(xyz_lines) != num_of_atoms:
                raise ParserError(f'Could not parse trajectory, expected {num_of_atoms} atoms, '
                                  f'but got {len(xyz_lines)} for point {len(traj) + 1} in the trajectory.')
            traj.append(str_to_xyz(''.join([xyz_line for xyz_line in xyz_lines])))
    if not len(traj):
        logger.error(f'Could not parse trajectory from {path}')
        return None
    return traj


def parse_dipole_moment(path: str,
                        raise_error: bool = True,
                        ) -> Optional[float]:
    """
    Parse the dipole moment in Debye from an opt job output file.

    Args:
        path: The ESS log file.
        raise_error (bool): Whether to raise an error if the dipole moment could not be parsed.

    Returns: Optional[float]
        The dipole moment in Debye.
    """
    dipole_moment = ess_factory(log_file_path=path).parse_dipole_moment()
    if dipole_moment is None and raise_error:
        raise ParserError('Could not parse the dipole moment')
    return dipole_moment


def _get_lines_from_file(path: str) -> List[str]:
    """
    A helper function for getting a list of lines from a file.

    Args:
        path (str): The file path.

    Raises:
        InputError: If the file could not be read.

    Returns: List[str]
        Entries are lines from the file.
    """
    if os.path.isfile(path):
        with open(path, 'r') as f:
            lines = f.readlines()
    else:
        raise InputError(f'Could not find file {path}')
    return lines


def process_conformers_file(conformers_path: str) -> Tuple[List[Dict[str, tuple]], List[float]]:
    """
    Parse coordinates and energies from an ARC conformers file of either species or TSs.

    Args:
        conformers_path (str): The path to an ARC conformers file
                               (either a "conformers_before_optimization" or
                               a "conformers_after_optimization" file).

    Raises:
        InputError: If the file could not be found.

    Returns: Tuple[List[Dict[str, tuple]], List[float]]
        Conformer coordinates in a dict format, the respective energies in kJ/mol.
    """
    if not os.path.isfile(conformers_path):
        raise InputError(f'Conformers file {conformers_path} could not be found')
    with open(conformers_path, 'r') as f:
        lines = f.readlines()
    xyzs, energies = list(), list()
    line_index = 0
    while line_index < len(lines):
        if 'conformer' in lines[line_index] and ':' in lines[line_index] and lines[line_index].strip()[-2].isdigit():
            xyz, energy = '', None
            line_index += 1
            while len(lines) and line_index < len(lines) and lines[line_index].strip() \
                    and 'SMILES' not in lines[line_index] \
                    and 'energy' not in lines[line_index].lower() \
                    and 'guess method' not in lines[line_index].lower():
                xyz += lines[line_index]
                line_index += 1
            while len(lines) and line_index < len(lines) and 'conformer' not in lines[line_index]:
                if 'relative energy:' in lines[line_index].lower():
                    energy = float(lines[line_index].split()[2])
                line_index += 1
            xyzs.append(str_to_xyz(xyz))
            energies.append(energy)
        else:
            line_index += 1
    return xyzs, energies
