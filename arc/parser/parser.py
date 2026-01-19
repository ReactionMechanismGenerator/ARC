"""
A module for parsing information from various files.
"""

import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Match


from arc.common import get_logger
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
        ess_name = 'yaml'
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
            elif 'x t b' in line.lower() or '$vibrational spectrum' in line:
                ess_name = 'xtb'
                break
            line = f.readline().lower()
    if ess_name is None and os.path.splitext(log_file_path)[-1] in ['.xyz', '.dat', '.geometry']:
        ess_name = 'terachem'
    if ess_name is None:
        if raise_error:
            raise InputError(f'The ESS that generated the file at {log_file_path} could not be identified.')
        else:
            logger.warning(f'The ESS that generated the file at {log_file_path} could not be identified.')
            return None
    return ess_name


def parse_xyz_from_file(log_file_path: str) -> Optional[Dict[str, tuple]]:
    """
    Fallback for parse_geometry()

    Parse xyz coordinated from:
    - .xyz: XYZ file
    - .gjf: Gaussian input file
    - .out or .log: ESS output file (Gaussian, Molpro, Orca, QChem, TeraChem) - calls parse_geometry()
    - .yml or .yaml files
    - other: Molpro or QChem input file

    Args:
        log_file_path (str): The file path.

    Returns: Optional[Dict[str, tuple]]
        The parsed cartesian coordinates.
    """
    if not os.path.isfile(log_file_path):
        raise InputError(f'Could not find file {log_file_path}')
    lines = _get_lines_from_file(log_file_path)
    file_extension = os.path.splitext(log_file_path)[1]
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
            raise ParserError(f'Could not identify the number of atoms line in the xyz file {log_file_path}')
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
            raise ParserError(f'Could not parse xyz coordinates from file {log_file_path}')
    if xyz is None and relevant_lines:
        xyz = str_to_xyz(''.join([line for line in relevant_lines if line]))
    return xyz


def make_parser(parse_method: str,
                return_type: Type,
                error_message: str,
                fallback: Optional[Callable] = None,
                ) -> Callable[[str, bool], Optional[Any]]:
    """
    Create a parser function for a specific ESS property.

    Args:
        parse_method (str): Name of the parse method in ESSAdapter
        return_type (Type): Expected return type (for type hinting)
        error_message (str): Template for error message (uses {path} placeholder)
        fallback (callable, optional): Fallback function to call if the ESS adapter method fails.

    Returns:
        Callable: Configured parser function
    """
    def parser(log_file_path: str, raise_error: bool = False) -> return_type:
        ess_name = determine_ess(log_file_path=log_file_path, raise_error=False)
        result = None
        if ess_name is not None:
            adapter = ess_factory(log_file_path=log_file_path, ess_adapter=ess_name)
            method = getattr(adapter, parse_method, None)
            if callable(method):
                result = method()
        if result is None and fallback is not None:
            try:
                result = fallback(log_file_path)
            except Exception as e:
                logger.error(f'Fallback parsing failed for {log_file_path}: {e}')
        if result is None and raise_error:
            raise ParserError(error_message.format(path=log_file_path))
        return result
    return parser


parse_geometry = make_parser(
    parse_method='parse_geometry',
    return_type=Optional[Dict[str, tuple]],
    error_message='Could not parse the geometry from {path} using either an ESS adapter or XYZ parser.',
    fallback=parse_xyz_from_file,
)

parse_frequencies = make_parser(
    parse_method='parse_frequencies',
    return_type=Optional['np.ndarray'],
    error_message='Could not parse frequencies from {path}',
)

parse_normal_mode_displacement = make_parser(
    parse_method='parse_normal_mode_displacement',
    return_type=Tuple[Optional['np.ndarray'], Optional['np.ndarray']],
    error_message='Could not parse normal mode displacement from {path}',
)

parse_t1 = make_parser(
    parse_method='parse_t1',
    return_type=Optional[float],
    error_message='Could not parse T1 from {path}',
)

parse_e_elect = make_parser(
    parse_method='parse_e_elect',
    return_type=Optional[float],
    error_message='Could not parse e_elect from {path}',
)

parse_zpe_correction = make_parser(
    parse_method='parse_zpe_correction',
    return_type=Optional[float],
    error_message='Could not parse zpe correction from {path}',
)

parse_1d_scan_energies = make_parser(
    parse_method='parse_1d_scan_energies',
    return_type=Tuple[Optional[List[float]], Optional[List[float]]],
    error_message='Could not parse 1d scan energies from {path}',
)

parse_1d_scan_coords = make_parser(
    parse_method='parse_1d_scan_coords',
    return_type=Optional[List[Dict[str, tuple]]],
    error_message='Could not parse 1d scan coords from {path}',
)

parse_irc_traj = make_parser(
    parse_method='parse_irc_traj',
    return_type=Optional[List[Dict[str, tuple]]],
    error_message='Could not parse IRC trajectory from {path}',
)

parse_scan_conformers = make_parser(
    parse_method='parse_scan_conformers',
    return_type=Optional['pd.DataFrame'],
    error_message='Could not parse scan conformers from {path}',
)

parse_nd_scan_energies = make_parser(
    parse_method='parse_nd_scan_energies',
    return_type=Optional[Dict],
    error_message='Could not parse nd scan energies from {path}',
)

parse_dipole_moment = make_parser(
    parse_method='parse_dipole_moment',
    return_type=Optional[float],
    error_message='Could not parse dipole moment from {path}',
)

parse_polarizability = make_parser(
    parse_method='parse_polarizability',
    return_type=Optional[float],
    error_message='Could not parse polarizability from {path}',
)


def parse_1d_scan_energies_from_specific_angle(log_file_path: str,
                                               initial_angle: float,
                                               ) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Parse the energies of a 1D scan from a specific angle.

    Args:
        log_file_path (str): The path to the ESS output file.
        initial_angle (float): The initial angle of the scan in degrees.

    Returns: Tuple[Optional[List[float]], Optional[List[float]]]
        The electronic energy in kJ/mol and the dihedral scan angle in degrees.
    """
    energies, angles = parse_1d_scan_energies(log_file_path=log_file_path)
    if angles is not None and len(angles) > 0:
        angles = [float(angle) + initial_angle for angle in angles]
    else:
        angles = None
    return energies, angles


def parse_scan_args(file_path: str) -> dict:
    """
    Get the scan arguments, including which internal coordinates (IC) are being scanned, which are frozen,
    what is the step size and the number of atoms, etc.

    Args:
        file_path (str): The path to a readable output file.

    Raises:
        NotImplementedError: If files other than Gaussian log is input

    Returns: dict
        A dictionary that contains the scan arguments as well as step number, step size, number of atom::

              {'scan': <list, atom indexes of the torsion to be scanned>,
               'freeze': <list, list of internal coordinates identified by atom indexes>,
               'step': <int, number of steps to scan>,
               'step_size': <float, the size of each step>,
               'n_atom': <int, the number of atoms of the molecule>,
               }
    """
    ess_name = determine_ess(log_file_path=file_path, raise_error=False)
    scan_args = {'scan': None, 'freeze': [],
                 'step': 0, 'step_size': 0, 'n_atom': 0}
    if ess_name == 'gaussian':
        try:
            # g09, g16
            scan_blk = parse_str_blocks(file_path, 'The following ModRedundant input section has been read:',
                                        'Isotopes and Nuclear Properties', regex=False)[0][1:-1]
        except IndexError:  # Cannot find any block
            # g03
            scan_blk_1 = parse_str_blocks(file_path, 'The following ModRedundant input section has been read:',
                                          'GradGradGradGrad', regex=False)[0][1:-2]
            scan_blk_2 = parse_str_blocks(file_path, 'NAtoms=',
                                          'One-electron integrals computed', regex=False)[0][:1]
            scan_blk = scan_blk_1 + scan_blk_2
        scan_pat = r'[DBA]?(\s+\d+){2,4}\s+S\s+\d+[\s\d.]+'
        frz_pat = r'[DBA]?(\s+\d+){2,4}\s+F'
        value_pat = r'[\d.]+'
        for line in scan_blk:
            if re.search(scan_pat, line.strip()):
                values = re.findall(value_pat, line)
                scan_len = len(values) - 2  # atom indexes + step + stepsize
                scan_args['scan'] = [int(values[i]) for i in range(scan_len)]
                scan_args['step'] = int(values[-2])
                scan_args['step_size'] = float(values[-1])
            if re.search(frz_pat, line.strip()):
                values = re.findall(value_pat, line)
                scan_args['freeze'].append([int(values[i]) for i in range(len(values))])
            if 'NAtoms' in line:
                scan_args['n_atom'] = int(line.split()[1])
    else:
        raise NotImplementedError(f'parse_scan_args() can currently only parse Gaussian output files, got {ess_name}')
    return scan_args


def parse_str_blocks(file_path: str,
                     head_pat: Union[Match, str],
                     tail_pat: Union[Match, str],
                     regex: bool = True,
                     tail_count: int = 1,
                     block_count: int = 1,
                     ) -> List[str]:
    """
    Return a list of blocks defined by the head pattern and the tail pattern.

    Args:
        file_path (str): The path to the readable file.
        head_pat (str/regex): Str pattern or regular expression of the head of the block.
        tail_pat (str/regex): Str pattern or regular expresion of the tail of the block.
        regex (bool, optional): Use regex (True) or str pattern (False) to search.
        tail_count (int, optional): The number of times that the tail repeats.
        block_count (int, optional): The max number of blocks to search. -1 for any number.

    Raises:
        InputError: If the file could not be found.

    Returns: List[str]
        List of str blocks.
    """
    if not os.path.isfile(file_path):
        raise InputError('Could not find file {0}'.format(file_path))
    with open(file_path, 'r') as f:
        blks = []
        # Different search mode
        if regex:
            def search(x, y):
                return re.search(x, y)
        else:
            def search(x, y):
                return x in y
        # 'search' for the head or 'read' until the tail
        mode = 'search'
        line = f.readline()
        while line != '':
            if mode == 'search':
                # Stop searching if enough blocks were found.
                if (len(blks)) == block_count:
                    break
                # Check if matching the head pattern.
                else:
                    match = search(head_pat, line)
                    # Switch to 'read' mode.
                    if match:
                        tail_repeat = 0
                        mode = 'read'
                        blks.append([])
                        blks[-1].append(line)
            elif mode == 'read':
                blks[-1].append(line)
                match = search(tail_pat, line)
                if match:
                    tail_repeat += 1
                    # If there are enough tail patterns, switch to 'search' mode.
                    if tail_repeat == tail_count:
                        mode = 'search'
            line = f.readline()
        # Remove the last incomplete search
        if len(blks) > 0 and (tail_repeat != tail_count):
            blks.pop()
        return blks


def parse_trajectory(path: str) -> Optional[List[Dict[str, tuple]]]:
    """
    Parse all geometries from an xyz trajectory file or an ESS output file.

    Args:
        path (str): The file path.

    Raises:
        ParserError: If the trajectory could not be read.

    Returns: Optional[List[Dict[str, tuple]]]
        Entries are xyz's on the trajectory.
    """
    ess_file, traj = False, None
    if path.split('.')[-1] != 'xyz':
        ess_file = bool(determine_ess(log_file_path=path, raise_error=False))
    if ess_file:
        try:
            traj = parse_1d_scan_coords(log_file_path=path)
        except (ValueError, TypeError):
            pass
        if not traj:
            try:
                traj = parse_irc_traj(log_file_path=path)
            except (ValueError, TypeError):
                return None
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
    if not traj:
        logger.error(f'Could not parse trajectory from {path}')
        return None
    return traj


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


def parse_active_space(sp_path: str, species: 'ARCSpecies') -> Optional[Dict[str, Union[List[int], Tuple[int, int]]]]:
    """
    Parse the active space (electrons and orbitals) from a Molpro CCSD output file.

    Args:
        sp_path (str): The path to the sp job output file.
        species ('ARCSpecies'): The species to consider.

    Returns:
        Optional[Dict[str, Union[List[int], Tuple[int, int]]]]:
            The active orbitals. Possible keys are:
                 'occ' (List[int]): The occupied orbitals.
                 'closed' (List[int]): The closed-shell orbitals.
                 'frozen' (List[int]): The frozen orbitals.
                 'core' (List[int]): The core orbitals.
                 'e_o' (Tuple[int, int]): The number of active electrons, determined by the total number
                 of electrons minus the core electrons (2 e's per heavy atom), and the number of active
                 orbitals, determined by the number of closed-shell orbitals and active orbitals
                 (w/o core orbitals).
    """
    if not os.path.isfile(sp_path):
        raise InputError(f'Could not find file {sp_path}')
    if not determine_ess(sp_path) == 'molpro':
        raise InputError(f'File {sp_path} is not a Molpro output file, cannot parse active space.')
    active = dict()
    num_heavy_atoms = sum([1 for symbol in species.get_xyz()['symbols'] if symbol not in ['H', 'D', 'T']])
    nuclear_charge, total_closed_shell_orbitals, total_active_orbitals = None, None, None
    core_orbitals, closed_shell_orbitals, active_orbitals = None, None, None
    with open(sp_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'NUCLEAR CHARGE:' in line:
            #  NUCLEAR CHARGE:                   32
            nuclear_charge = int(line.split()[-1])
        if 'Number of core orbitals:' in line:
            # Number of core orbitals:           1 (   1   0   0   0   0   0   0   0 )
            # Number of core orbitals:           3 (   3 )
            core_orbitals = line.split('(')[-1].split(')')[0].split()
        if 'Number of closed-shell orbitals:' in line:
            #  Number of closed-shell orbitals:  11 (   9   2 )
            #  Number of closed-shell orbitals:   8 (   8 )
            closed_shell_orbitals = line.split('(')[-1].split(')')[0].split()
            total_closed_shell_orbitals = int(line.split('(')[0].split()[-1])
        if 'Number of active' in line and 'orbitals' in line:
            #  Number of active  orbitals:        2 (   1   1 )
            #  Number of active  orbitals:        2 (   2 )
            active_orbitals = line.split('(')[-1].split(')')[0].split()
            total_active_orbitals = int(line.split('(')[0].split()[-1])
        if None not in [nuclear_charge, total_closed_shell_orbitals, total_active_orbitals,
                        core_orbitals, closed_shell_orbitals, active_orbitals]:
            break
    if nuclear_charge is None:
        return None
    active_space_electrons = nuclear_charge - species.charge - 2 * num_heavy_atoms
    if (total_closed_shell_orbitals is None) ^ (total_active_orbitals is None):
        num_active_space_orbitals = total_closed_shell_orbitals or total_active_orbitals
    else:
        num_active_space_orbitals = total_closed_shell_orbitals + total_active_orbitals
    active['e_o'] = (active_space_electrons, num_active_space_orbitals)
    if core_orbitals is not None:
        active['occ'] = [int(c) for c in core_orbitals]
        if closed_shell_orbitals is not None:
            active['closed'] = [int(c) for c in closed_shell_orbitals]
            for i in range(len(closed_shell_orbitals)):
                active['occ'][i] += int(closed_shell_orbitals[i])
                if active_orbitals is not None:
                    active['occ'][i] += int(active_orbitals[i])
        if active_orbitals is not None and int(active_orbitals[0]) == 0:
            # No 1st irrep is suggested to be active.
            # We should therefore add another active orbital to the 1st irrep.
            active['occ'][0] += 1
    return active
