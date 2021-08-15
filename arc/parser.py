"""
A module for parsing information from various files.
"""

import os
import re
from typing import Dict, List, Match, Optional, Tuple, Union

import numpy as np
import pandas as pd
import qcelemental as qcel

from arkane.exceptions import LogError
from arkane.ess import ess_factory, GaussianLog, MolproLog, OrcaLog, QChemLog, TeraChemLog

from arc.common import determine_ess, get_close_tuple, get_logger, is_same_pivot
from arc.exceptions import InputError, ParserError
from arc.species.converter import str_to_xyz, xyz_from_data


logger = get_logger()


def parse_frequencies(path: str,
                      software: str,
                      ) -> np.ndarray:
    """
    Parse the frequencies from a freq job output file.

    Args:
        path (str): The log file path.
        software (str): The ESS.

    Returns: np.ndarray
        The parsed frequencies (in cm^-1).
    """
    lines = _get_lines_from_file(path)
    freqs = np.array([], np.float64)
    if software.lower() == 'qchem':
        for line in lines:
            if ' Frequency:' in line:
                items = line.split()
                for i, item in enumerate(items):
                    if i:
                        freqs = np.append(freqs, [(float(item))])
    elif software.lower() == 'gaussian':
        with open(path, 'r') as f:
            line = f.readline()
            while line != '':
                # this line intends to only capture the last occurrence of the frequencies
                if 'and normal coordinates' in line:
                    freqs = np.array([], np.float64)
                if 'Frequencies --' in line:
                    freqs = np.append(freqs, [float(frq) for frq in line.split()[2:]])
                line = f.readline()
    elif software.lower() == 'molpro':
        read = False
        for line in lines:
            if 'Nr' in line and '[1/cm]' in line:
                continue
            if read:
                if line == os.linesep:
                    read = False
                    continue
                freqs = np.append(freqs, [float(line.split()[-1])])
            if 'Low' not in line and 'Vibration' in line and 'Wavenumber' in line:
                read = True
    elif software.lower() == 'orca':
        with open(path, 'r') as f:
            line = f.readline()
            read = True
            while line:
                if 'VIBRATIONAL FREQUENCIES' in line:
                    while read:
                        if not line.strip():
                            line = f.readline()
                        elif not line.split()[0] == '0:':
                            line = f.readline()
                        else:
                            read = False
                    while line.strip():
                        if float(line.split()[1]) != 0.0:
                            freqs = np.append(freqs, [float(line.split()[1])])
                        line = f.readline()
                    break
                else:
                    line = f.readline()
    elif software.lower() == 'terachem':
        read_output = False
        for line in lines:
            if '=== Mode' in line:
                # example: '=== Mode 1: 1198.526 cm^-1 ==='
                freqs = np.append(freqs, [float(line.split()[3])])
            elif 'Vibrational Frequencies/Thermochemical Analysis After Removing Rotation and Translation' in line:
                read_output = True
                continue
            elif read_output:
                if 'Temperature (Kelvin):' in line or 'Frequency(cm-1)' in line:
                    continue
                if not line.strip():
                    break
                # example:
                # 'Mode  Eigenvalue(AU)  Frequency(cm-1)  Intensity(km/mol)   Vib.Temp(K)      ZPE(AU) ...'
                # '  1     0.0331810528   170.5666870932      52.2294230772  245.3982965841   0.0003885795 ...'
                freqs = np.append(freqs, [float(line.split()[2])])

    else:
        raise ParserError(f'parse_frequencies() can currently only parse Gaussian, Molpro, Orca, QChem and TeraChem '
                          f'files, got {software}')
    logger.debug(f'Using parser.parse_frequencies(). Determined frequencies are: {freqs}')
    return freqs


def parse_normal_mode_displacement(path: str,
                                   software: Optional[str] = None,
                                   raise_error: bool = True,
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse frequencies and normal modes displacement.

    Args:
        path (str): The path to the log file.
        software (str, optional): The software to used to generate the log file.
        raise_error (bool, optional): Whether to raise an error if the ESS isn't implemented.

    Raises:
        NotImplementedError: If the parser is not implemented for the ESS this log file belongs to.

    Returns: Tuple[np.ndarray, np.ndarray]
        The frequencies (in cm^-1) and The normal displacement modes.
    """
    software = software or determine_ess(path)
    freqs, normal_mode_disp, normal_mode_disp_entries = list(), list(), list()
    num_of_freqs_per_line = 3
    with open(path, 'r') as f:
        lines = f.readlines()
    if software == 'gaussian':
        parse, parse_normal_mode_disp = False, False
        for line in lines:
            if 'Harmonic frequencies (cm**-1)' in line:
                # e.g.:  Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering
                parse = True
            if parse and len(line.split()) in [0, 1, 3]:
                parse_normal_mode_disp = False
                normal_mode_disp.extend(normal_mode_disp_entries)
                normal_mode_disp_entries = list()
            if parse and 'Frequencies --' in line:
                # e.g.:  Frequencies --    -18.0696               127.6948               174.9499
                splits = line.split()
                freqs.extend(float(freq) for freq in splits[2:])
                num_of_freqs_per_line = len(splits) - 2
                normal_mode_disp_entries = list()
            elif parse_normal_mode_disp:
                # parsing, e.g.:
                #   Atom  AN      X      Y      Z        X      Y      Z        X      Y      Z
                #      1   6    -0.00   0.00  -0.09    -0.00   0.00  -0.18     0.00  -0.00  -0.16
                #      2   7    -0.00   0.00  -0.10     0.00  -0.00   0.02     0.00  -0.00   0.26
                splits = line.split()[2:]
                for i in range(num_of_freqs_per_line):
                    if len(normal_mode_disp_entries) < i + 1:
                        normal_mode_disp_entries.append(list())
                    normal_mode_disp_entries[i].append(splits[3 * i: 3 * i + 3])
            elif parse and 'Atom  AN      X      Y      Z' in line:
                parse_normal_mode_disp = True
            elif parse and not line or '-------------------' in line:
                parse = False
    elif raise_error:
        raise NotImplementedError(f'parse_normal_mode_displacement() is currently not implemented for {software}.')
    freqs = np.array(freqs, np.float64)
    normal_mode_disp = np.array(normal_mode_disp, np.float64)
    return freqs, normal_mode_disp


def parse_geometry(path: str) -> Optional[Dict[str, tuple]]:
    """
    Parse the xyz geometry from an ESS log file.

    Args:
        path (str): The ESS log file to parse from.

    Returns: Optional[Dict[str, tuple]]
        The cartesian geometry.
    """
    log = ess_factory(fullpath=path, check_for_errors=False)
    try:
        coords, number, _ = log.load_geometry()
    except LogError:
        logger.debug(f'Could not parse xyz from {path}')

        # try parsing Gaussian standard orientation instead of the input orientation parsed by Arkane
        lines = _get_lines_from_file(path)
        xyz_str = ''
        for i in range(len(lines)):
            if 'Standard orientation:' in lines[i]:
                xyz_str = ''
                j = i
                while len(lines) and not lines[j].split()[0].isdigit():
                    j += 1
                while len(lines) and '-------------------' not in lines[j]:
                    splits = lines[j].split()
                    xyz_str += f'{qcel.periodictable.to_E(int(splits[1]))}  {splits[3]}  {splits[4]}  {splits[5]}\n'
                    j += 1
                break

        if xyz_str:
            return str_to_xyz(xyz_str)

        return None

    return xyz_from_data(coords=coords, numbers=number)


def parse_t1(path: str) -> Optional[float]:
    """
    Parse the T1 parameter from a Molpro or Orca coupled cluster calculation.

    Args:
        path (str): The ess log file path.

    Returns: Optional[float]
        The T1 parameter.
    """
    if not os.path.isfile(path):
        raise InputError(f'Could not find file {path}')
    log = ess_factory(fullpath=path, check_for_errors=False)
    try:
        t1 = log.get_T1_diagnostic()
    except (LogError, NotImplementedError):
        logger.warning('Could not read t1 from {0}'.format(path))
        t1 = None
    return t1


def parse_e_elect(path: str,
                  zpe_scale_factor: float = 1.,
                  ) -> Optional[float]:
    """
    Parse the electronic energy from an sp job output file.

    Args:
        path (str): The ESS log file to parse from.
        zpe_scale_factor (float): The ZPE scaling factor, used only for composite methods in Gaussian via Arkane.

    Returns: Optional[float]
        The electronic energy in kJ/mol.
    """
    if not os.path.isfile(path):
        raise InputError(f'Could not find file {path}')
    log = ess_factory(fullpath=path, check_for_errors=False)
    try:
        e_elect = log.load_energy(zpe_scale_factor) * 0.001  # convert to kJ/mol
    except (LogError, NotImplementedError):
        logger.warning(f'Could not read e_elect from {path}')
        e_elect = None
    return e_elect


def parse_zpe(path: str) -> Optional[float]:
    """
    Determine the calculated ZPE from a frequency output file

    Args:
        path (str): The path to a frequency calculation output file.

    Returns: Optional[float]
        The calculated zero point energy in kJ/mol.
    """
    if not os.path.isfile(path):
        raise InputError(f'Could not find file {path}')
    log = ess_factory(fullpath=path, check_for_errors=False)
    try:
        zpe = log.load_zero_point_energy() * 0.001  # convert to kJ/mol
    except (LogError, NotImplementedError):
        logger.warning('Could not read zpe from {0}'.format(path))
        zpe = None
    return zpe


def parse_1d_scan_energies(path: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Parse the 1D torsion scan energies from an ESS log file.

    Args:
        path (str): The ESS log file to parse from.

    Raises:
        InputError: If ``path`` is invalid.

    Returns: Tuple[Optional[List[float]], Optional[List[float]]]
        The electronic energy in kJ/mol and the dihedral scan angle in degrees.
    """
    if not os.path.isfile(path):
        raise InputError(f'Could not find file {path}')
    log = ess_factory(fullpath=path, check_for_errors=False)
    try:
        energies, angles = log.load_scan_energies()
        energies *= 0.001  # convert to kJ/mol
        angles *= 180 / np.pi  # convert to degrees
    except (LogError, NotImplementedError, ZeroDivisionError):
        logger.warning(f'Could not read energies from {path}')
        energies, angles = None, None
    return energies, angles


def parse_1d_scan_coords(path: str) -> List[Dict[str, tuple]]:
    """
    Parse the 1D torsion scan coordinates from an ESS log file.

    Args:
        path (str): The ESS log file to parse from.

    Returns: list
        The Cartesian coordinates.
    """
    lines = _get_lines_from_file(path)
    log = ess_factory(fullpath=path, check_for_errors=False)
    if not isinstance(log, GaussianLog):
        raise NotImplementedError(f'Currently parse_1d_scan_coords only supports Gaussian files, got {type(log)}')
    traj = list()
    done = False
    i = 0
    while not done:
        if i >= len(lines) or 'Normal termination of Gaussian' in lines[i] or 'Error termination via' in lines[i]:
            done = True
        elif 'Optimization completed' in lines[i]:
            while len(lines) and 'Input orientation:' not in lines[i]:
                i += 1
                if 'Error termination via' in lines[i]:
                    return traj
            i += 5
            xyz_str = ''
            while len(lines) and '--------------------------------------------' not in lines[i]:
                splits = lines[i].split()
                xyz_str += f'{qcel.periodictable.to_E(int(splits[1]))}  {splits[3]}  {splits[4]}  {splits[5]}\n'
                i += 1
            traj.append(str_to_xyz(xyz_str))
        i += 1
    return traj


def parse_nd_scan_energies(path: str,
                           software: Optional[str] = None,
                           return_original_dihedrals: bool = False,
                           ) -> Tuple[dict, Optional[List[float]]]:
    """
    Parse the ND torsion scan energies from an ESS log file.

    Args:
        path (str): The ESS log file to parse from.
        software (str, optional): The software used to run this scan, default is 'gaussian'.
        return_original_dihedrals (bool, optional): Whether to return the dihedral angles of the original conformer.
                                                    ``True`` to return, default is ``False``.

    Raises:
        InputError: If ``path`` is invalid.

    Returns: Tuple[dict, Optional[List[float]]]
        The "results" dictionary, which has the following structure::

              results = {'directed_scan_type': <str, used for the fig name>,
                         'scans': <list, entries are lists of torsion indices>,
                         'directed_scan': <dict, keys are tuples of '{0:.2f}' formatted dihedrals,
                                           values are dictionaries with the following keys and values:
                                           {'energy': <float, energy in kJ/mol>,  * only this is used here
                                            'xyz': <dict>,
                                            'is_isomorphic': <bool>,
                                            'trsh': <list, job.ess_trsh_methods>}>
                         },

        The dihedrals angles of the original conformer
    """
    software = software or determine_ess(path)
    results = {'directed_scan_type': f'ess_{software}',
               'scans': list(),
               'directed_scan': dict(),
               }
    if software == 'gaussian':
        # internal variables:
        # - scan_d_dict (dict): keys are scanning dihedral names (e.g., 'D2', or 'D4'), values are the corresponding
        #                       torsion indices tuples (e.g., (4, 1, 2, 5), or (4, 1, 3, 6)).
        # - dihedrals_dict (dict): keys are torsion tuples (e.g., (4, 1, 2, 5), or (4, 1, 3, 6)),
        #                          values are lists of dihedral angles in degrees corresponding to the torsion
        #                          (e.g., [-159.99700, -149.99690, -139.99694, -129.99691, -119.99693]).
        # - torsions (list): entries are torsion indices that are scanned, e.g.: [(4, 1, 2, 5), (4, 1, 3, 6)]
        with open(path, 'r', buffering=8192) as f:
            line = f.readline()
            symbols, torsions, shape, resolution, original_dihedrals = list(), list(), list(), list(), list()
            scan_d_dict = dict()
            min_e = None
            while line:
                line = f.readline()
                if 'The following ModRedundant input section has been read:' in line:
                    # ' The following ModRedundant input section has been read:'
                    # ' D       4       1       2       5 S  36 10.000'
                    # ' D       4       1       3       6 S  36 10.000'
                    line = f.readline()
                    while True:
                        splits = line.split()
                        if len(splits) == 8:
                            torsions.append(tuple([int(index) for index in splits[1:5]]))
                            shape.append(int(splits[6]) + 1)  # the last point is repeated
                            resolution.append(float(splits[7]))
                        else:
                            break
                        line = f.readline()
                    results['scans'] = torsions
                    if 'Symbolic Z-matrix:' in line:
                        #  ---------------------
                        #  HIR calculation by AI
                        #  ---------------------
                        #  Symbolic Z-matrix:
                        #  Charge =  0 Multiplicity = 1
                        #  c
                        #  o                    1    oc2
                        #  o                    1    oc3      2    oco3
                        #  o                    1    oc4      2    oco4     3    dih4     0
                        #  h                    2    ho5      1    hoc5     3    dih5     0
                        #  h                    3    ho6      1    hoc6     4    dih6     0
                        #        Variables:
                        #   oc2                   1.36119
                        #   oc3                   1.36119
                        #   oco3                114.896
                        #   oc4                   1.18581
                        #   oco4                122.552
                        #   dih4                180.
                        #   ho5                   0.9637
                        #   hoc5                111.746
                        #   dih5                 20.003
                        #   ho6                   0.9637
                        #   hoc6                111.746
                        #   dih6               -160.
                        for i in range(2):
                            f.readline()
                        while 'Variables' not in line:
                            symbols.append(line.split()[0].upper())
                            line = f.readline()
                if 'Initial Parameters' in line:
                    #                            ----------------------------
                    #                            !    Initial Parameters    !
                    #                            ! (Angstroms and Degrees)  !
                    #  --------------------------                            --------------------------
                    #  ! Name  Definition              Value          Derivative Info.                !
                    #  --------------------------------------------------------------------------------
                    #  ! R1    R(1,2)                  1.3612         calculate D2E/DX2 analytically  !
                    #  ! R2    R(1,3)                  1.3612         calculate D2E/DX2 analytically  !
                    #  ! R3    R(1,4)                  1.1858         calculate D2E/DX2 analytically  !
                    #  ! R4    R(2,5)                  0.9637         calculate D2E/DX2 analytically  !
                    #  ! R5    R(3,6)                  0.9637         calculate D2E/DX2 analytically  !
                    #  ! A1    A(2,1,3)              114.896          calculate D2E/DX2 analytically  !
                    #  ! A2    A(2,1,4)              122.552          calculate D2E/DX2 analytically  !
                    #  ! A3    A(3,1,4)              122.552          calculate D2E/DX2 analytically  !
                    #  ! A4    A(1,2,5)              111.746          calculate D2E/DX2 analytically  !
                    #  ! A5    A(1,3,6)              111.746          calculate D2E/DX2 analytically  !
                    #  ! D1    D(3,1,2,5)             20.003          calculate D2E/DX2 analytically  !
                    #  ! D2    D(4,1,2,5)           -159.997          Scan                            !
                    #  ! D3    D(2,1,3,6)             20.0            calculate D2E/DX2 analytically  !
                    #  ! D4    D(4,1,3,6)           -160.0            Scan                            !
                    #  --------------------------------------------------------------------------------
                    for i in range(5):
                        line = f.readline()
                    # original_zmat = {'symbols': list(), 'coords': list(), 'vars': dict()}
                    while '--------------------------' not in line:
                        splits = line.split()
                        # key = splits[2][:-1].replace('(', '_').replace(',', '_')
                        # val = float(splits[3])
                        # original_zmat['symbols'].append(symbols[len(original_zmat['symbols'])])
                        # original_zmat['vars'][key] = val
                        if 'Scan' in line:
                            scan_d_dict[splits[1]] = \
                                tuple([int(index) for index in splits[2][2:].replace(')', '').split(',')])
                            original_dihedrals.append(float(splits[3]))
                        line = f.readline()

                elif 'Summary of Optimized Potential Surface Scan' in line:
                    # ' Summary of Optimized Potential Surface Scan (add -264.0 to energies):'
                    base_e = float(line.split('(add ')[1].split()[0])
                    energies, dihedrals_dict = list(), dict()
                    dihedral_num = 0
                    while 'Grad' not in line:
                        line = f.readline()
                        splits = line.split()
                        if 'Eigenvalues --' in line:
                            # convert Hartree energy to kJ/mol
                            energies = [(base_e + float(e)) * 4.3597447222071e-18 * 6.02214179e23 * 1e-3
                                        for e in splits[2:]]
                            min_es = min(energies)
                            min_e = min_es if min_e is None else min(min_e, min_es)
                            dihedral_num = 0
                        if splits[0] in list(scan_d_dict.keys()) \
                                and scan_d_dict[splits[0]] not in list(dihedrals_dict.keys()):
                            # parse the dihedral information
                            # '           D1          20.00308  30.00361  40.05829  50.36777  61.07341'
                            # '           D2        -159.99700-149.99690-139.99694-129.99691-119.99693'
                            # '           D3          19.99992  19.99959  19.94509  19.63805  18.93967'
                            # '           D4        -160.00000-159.99990-159.99994-159.99991-159.99993'
                            dihedrals = [float(dihedral) for dihedral in line.replace('-', ' -').split()[1:]]
                            for i in range(len(dihedrals)):
                                if 0 > dihedrals[i] >= -0.0049999:
                                    dihedrals[i] = 0.0
                            dihedrals_dict[scan_d_dict[splits[0]]] = dihedrals
                            dihedral_num += 1
                        if len(list(dihedrals_dict.keys())) == len(list(scan_d_dict.keys())):
                            # we have all the data for this block, pass to ``results`` and initialize ``dihedrals_dict``
                            for i, energy in enumerate(energies):
                                dihedral_list = [dihedrals_dict[torsion][i] for torsion in torsions]  # ordered
                                key = tuple(f'{dihedral:.2f}' for dihedral in dihedral_list)
                                # overwrite previous values for a close key if found:
                                key = get_close_tuple(key, results['directed_scan'].keys()) or key
                                results['directed_scan'][key] = {'energy': energy}
                            dihedrals_dict = dict()  # keys are torsion tuples, values are dihedral angles
                    break
            line = f.readline()
    else:
        raise NotImplementedError(f'parse_nd_scan_energies is currently only implemented for Gaussian, got {software}.')
    for key in results['directed_scan'].keys():
        results['directed_scan'][key] = {'energy': results['directed_scan'][key]['energy'] - min_e}
    if return_original_dihedrals:
        return results, original_dihedrals
    else:
        return results, None


def parse_xyz_from_file(path: str) -> Optional[Dict[str, tuple]]:
    """
    Parse xyz coordinated from:
    - .xyz: XYZ file
    - .gjf: Gaussian input file
    - .out or .log: ESS output file (Gaussian, Molpro, Orca, QChem, TeraChem) - calls parse_geometry()
    - other: Molpro or QChem input file

    Args:
        path (str): The file path.

    Raises:
        ParserError: If the coordinates could not be parsed.

    Returns: Optional[Dict[str, tuple]]
        The parsed cartesian coordinates.
    """
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
        xyz = parse_geometry(path)
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
    lines = _get_lines_from_file(path)

    ess_file = False
    if path.split('.')[-1] != 'xyz':
        try:
            log = ess_factory(fullpath=path, check_for_errors=False)
            ess_file = True
        except InputError:
            ess_file = False

    if ess_file:
        if not isinstance(log, GaussianLog):
            raise NotImplementedError(f'Currently parse_trajectory only supports Gaussian files, got {type(log)}')
        traj = list()
        done = False
        i = 0
        while not done:
            if i >= len(lines) or 'Normal termination of Gaussian' in lines[i] or 'Error termination via' in lines[i]:
                done = True
            elif 'Input orientation:' in lines[i]:
                i += 5
                xyz_str = ''
                while len(lines) and '--------------------------------------------' not in lines[i]:
                    splits = lines[i].split()
                    xyz_str += f'{qcel.periodictable.to_E(int(splits[1]))}  {splits[3]}  {splits[4]}  {splits[5]}\n'
                    i += 1
                traj.append(str_to_xyz(xyz_str))
            i += 1

    else:
        # this is not an ESS output file, probably an XYZ format file with several Cartesian coordinates
        skip_line = False
        num_of_atoms = 0
        traj, xyz_lines = list(), list()
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


def parse_dipole_moment(path: str) -> Optional[float]:
    """
    Parse the dipole moment in Debye from an opt job output file.

    Args:
        path: The ESS log file.

    Returns: Optional[float]
        The dipole moment in Debye.
    """
    lines = _get_lines_from_file(path)
    log = ess_factory(path, check_for_errors=False)
    dipole_moment = None
    if isinstance(log, GaussianLog):
        # example:
        # Dipole moment (field-independent basis, Debye):
        # X=             -0.0000    Y=             -0.0000    Z=             -1.8320  Tot=              1.8320
        read = False
        for line in lines:
            if 'dipole moment' in line.lower() and 'debye' in line.lower():
                read = True
            elif read:
                dipole_moment = float(line.split()[-1])
                read = False
    elif isinstance(log, MolproLog):
        # example: ' Dipole moment /Debye                   2.96069859     0.00000000     0.00000000'
        for line in lines:
            if 'dipole moment' in line.lower() and '/debye' in line.lower():
                splits = line.split()
                dm_x, dm_y, dm_z = float(splits[-3]), float(splits[-2]), float(splits[-1])
                dipole_moment = (dm_x ** 2 + dm_y ** 2 + dm_z ** 2) ** 0.5
    elif isinstance(log, OrcaLog):
        # example: 'Magnitude (Debye)      :      2.11328'
        for line in lines:
            if 'Magnitude (Debye)' in line:
                dipole_moment = float(line.split()[-1])
    elif isinstance(log, QChemLog):
        # example:
        #     Dipole Moment (Debye)
        #          X       0.0000      Y       0.0000      Z       2.0726
        #        Tot       2.0726
        skip = False
        read = False
        for line in lines:
            if 'dipole moment' in line.lower() and 'debye' in line.lower():
                skip = True
            elif skip:
                skip = False
                read = True
            elif read:
                dipole_moment = float(line.split()[-1])
                read = False
    elif isinstance(log, TeraChemLog):
        # example: 'DIPOLE MOMENT: {-0.000178, -0.000003, -0.000019} (|D| = 0.000179) DEBYE'
        for line in lines:
            if 'dipole moment' in line.lower() and 'debye' in line.lower():
                splits = line.split('{')[1].split('}')[0].replace(',', '').split()
                dm_x, dm_y, dm_z = float(splits[0]), float(splits[1]), float(splits[2])
                dipole_moment = (dm_x ** 2 + dm_y ** 2 + dm_z ** 2) ** 0.5
    else:
        raise ParserError('Currently dipole moments can only be parsed from either Gaussian, Molpro, Orca, QChem, '
                          'or TeraChem optimization output files')
    if dipole_moment is None:
        raise ParserError('Could not parse the dipole moment')
    return dipole_moment


def parse_polarizability(path: str) -> Optional[float]:
    """
    Parse the polarizability from a freq job output file, returns the value in Angstrom^3.

    Args:
        path: The ESS log file.

    Returns: Optional[float]
        The polarizability in Angstrom^3.
    """
    lines = _get_lines_from_file(path)
    polarizability = None
    for line in lines:
        if 'Isotropic polarizability for W' in line:
            # example:  Isotropic polarizability for W=    0.000000       11.49 Bohr**3.
            # 1 Bohr = 0.529177 Angstrom
            polarizability = float(line.split()[-2]) * 0.529177 ** 3
    return polarizability


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
        raise InputError('Conformers file {0} could not be found'.format(conformers_path))
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
                # Stop searching if found enough blocks
                if (len(blks)) == block_count:
                    break
                # Check if matching the head pattern
                else:
                    match = search(head_pat, line)
                    # Switch to 'read' mode
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
                    # If see enough tail patterns, switch to 'search' mode
                    if tail_repeat == tail_count:
                        mode = 'search'
            line = f.readline()
        # Remove the last incomplete search
        if len(blks) > 0 and (tail_repeat != tail_count):
            blks.pop()
        return blks


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
    log = ess_factory(fullpath=file_path, check_for_errors=False)
    scan_args = {'scan': None, 'freeze': [],
                 'step': 0, 'step_size': 0, 'n_atom': 0}
    if isinstance(log, GaussianLog):
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
        raise NotImplementedError(f'parse_scan_args() can currently only parse Gaussian output '
                                  f'files, got {log}')
    return scan_args


def parse_ic_info(file_path: str) -> pd.DataFrame:
    """
    Get the information of internal coordinates (ic) of an intermediate scan conformer.

    Args:
        file_path (str): The path to a readable output file.

    Raises:
        NotImplementedError: If files other than Gaussian log is input

    Returns: pd.DataFrame
        A DataFrame containing the information of the internal coordinates
    """
    log = ess_factory(fullpath=file_path, check_for_errors=False)
    ic_dict = {item: []
               for item in ['label', 'type', 'atoms', 'redundant', 'scan']}
    scan_args = parse_scan_args(file_path)
    max_atom_ind = scan_args['n_atom']
    if isinstance(log, GaussianLog):
        ic_info_block = parse_str_blocks(file_path, 'Initial Parameters', '-----------', regex=False,
                                         tail_count=3)[0][5:-1]
        for line in ic_info_block:
            # Line example with split() indices:
            # 0 1     2                        3              4         5       6            7
            # ! R1    R(1, 2)                  1.3581         calculate D2E/DX2 analytically !
            terms = line.split()
            ic_dict['label'].append(terms[1])
            ic_dict['type'].append(terms[1][0])  # 'R: bond, A: angle, D: dihedral
            atom_inds = re.split(r'[(),]', terms[2])[1:-1]
            ic_dict['atoms'].append([int(atom_ind) for atom_ind in atom_inds])

            # Identify redundant, cases like 5 atom angles or redundant atoms
            if (ic_dict['type'][-1] == 'A' and len(atom_inds) > 3) \
                    or (ic_dict['type'][-1] == 'R' and len(atom_inds) > 2) \
                    or (ic_dict['type'][-1] == 'D' and len(atom_inds) > 4):
                ic_dict['redundant'].append(True)
            else:
                # Sometimes, redundant atoms with weird indices are added.
                # Reason unclear. Maybe to better define the molecule, or to
                # solve equations more easily.
                weird_indices = [index for index in ic_dict['atoms'][-1]
                                 if index <= 0 or index > max_atom_ind]
                if weird_indices:
                    ic_dict['redundant'].append(True)
                else:
                    ic_dict['redundant'].append(False)

            # Identify ics being scanned
            if len(scan_args['scan']) == len(atom_inds) == 4 \
                    and is_same_pivot(scan_args['scan'], ic_dict['atoms'][-1]):
                ic_dict['scan'].append(True)
            elif len(scan_args['scan']) == len(atom_inds) == 2 \
                    and set(scan_args['scan']) == set(ic_dict['atoms'][-1]):
                ic_dict['scan'].append(True)
            else:
                # Currently doesn't support scan of angles
                ic_dict['scan'].append(False)
    else:
        raise NotImplementedError(f'parse_ic_info() can currently only parse Gaussian output '
                                  f'files, got {log}')
    ic_info = pd.DataFrame.from_dict(ic_dict)
    ic_info = ic_info.set_index('label')
    return ic_info


def parse_ic_values(ic_block: List[str],
                    software: Optional[str] = None,
                    ) -> pd.DataFrame:
    """
    Get the internal coordinates (ic) for an intermediate scan conformer

    Args:
        ic_block (list): A list of strings containing the optimized internal coordinates of
        an intermediate scan conformer
        software(str, optional): The software to used to generate the log file.

    Raises:
        NotImplementedError: If the software is not supported

    Returns:
        pd.DataFrame: A DataFrame containing the values of the internal coordinates
    """
    ic_dict = {item: [] for item in ['label', 'value']}
    if software == 'gaussian':
        for line in ic_block:
            # Line example with split() indices:
            # 0 1     2                       3              4      5    6                   7
            # ! R1    R(1,2)                  1.3602         -DE/DX =    0.0                 !
            terms = line.split()
            ic_dict['label'].append(terms[1])
            ic_dict['value'].append(float(terms[3]))
    else:
        raise NotImplementedError(f'parse_ics() can currently only parse Gaussian output '
                                  f'files, got {software}')
    ics = pd.DataFrame.from_dict(ic_dict)
    ics = ics.set_index('label')
    return ics


def parse_scan_conformers(file_path: str) -> pd.DataFrame:
    """
    Parse all the internal coordinates of all the scan intermediate conformers and tabulate
    all the info into a single DataFrame object. Any redundant internal coordinates
    will be removed during the process.

    Args:
        file_path (str): The path to a readable output file.

    Raises:
        NotImplementedError: If files other than Gaussian log is input

    Returns:
        pd.DataFrame: a list of conformers containing the all the internal
                       coordinates information in pd.DataFrame
    """
    log = ess_factory(fullpath=file_path, check_for_errors=False)
    scan_args = parse_scan_args(file_path)
    scan_ic_info = parse_ic_info(file_path)
    if isinstance(log, GaussianLog):
        software = 'gaussian'
        ic_blks = parse_str_blocks(file_path, 'Optimized Parameters', '-----------', regex=False,
                                   tail_count=3, block_count=(scan_args['step'] + 1))
    else:
        raise NotImplementedError(f'parse_scan_conformers() can currently only parse Gaussian output '
                                  f'files, got {log}')
    # Extract IC values for each conformer
    conformers = []
    for ind, ic_blk in enumerate(ic_blks):
        ics = parse_ic_values(ic_blk[5:-1], software)
        ics.rename(columns={'value': ind}, inplace=True)
        conformers.append(ics)
    # Concatenate ICs of conformers to a single table and remove redundant ICs
    scan_conformers = pd.concat([scan_ic_info] + conformers, axis=1)
    red_ind = scan_conformers[scan_conformers.redundant == True].index
    if not red_ind.empty:
        scan_conformers.drop(red_ind, inplace=True)
    return scan_conformers
