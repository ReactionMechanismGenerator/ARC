#!/usr/bin/env python3
# encoding: utf-8

"""
A module for parsing information from various files.
"""

import numpy as np
import os

from arkane.exceptions import LogError
from arkane.logs.gaussian import GaussianLog
from arkane.logs.molpro import MolproLog
from arkane.logs.qchem import QChemLog
from arkane.util import determine_qm_software

from arc.common import get_logger
from arc.exceptions import InputError, ParserError
from arc.species.converter import xyz_from_data, str_to_xyz


logger = get_logger()


def parse_frequencies(path, software):
    """
    Parse the frequencies from a freq job output file.
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
    else:
        raise ParserError('parse_frequencies() can currently only parse Molpro, QChem and Gaussian files,'
                          ' got {0}'.format(software))
    logger.debug('Using parser.parse_frequencies. Determined frequencies are: {0}'.format(freqs))
    return freqs


def parse_t1(path):
    """
    Parse the T1 parameter from a Molpro coupled cluster calculation.
    """
    if not os.path.isfile(path):
        raise InputError('Could not find file {0}'.format(path))
    log = determine_qm_software(fullpath=path)
    try:
        t1 = log.get_T1_diagnostic()
    except (LogError, NotImplementedError):
        logger.warning('Could not read t1 from {0}'.format(path))
        t1 = None
    return t1


def parse_e_elect(path, zpe_scale_factor=1.):
    """
    Parse the electronic energy from an sp job output file.

    Args:
        path (str): The ESS log file to parse from.
        zpe_scale_factor: The ZPE scaling factor, used only for composite methods in Gaussian via Arkane.

    Returns:
        e_elect (float): The electronic energy in kJ/mol
    """
    if not os.path.isfile(path):
        raise InputError('Could not find file {0}'.format(path))
    log = determine_qm_software(fullpath=path)
    try:
        e_elect = log.load_energy(zpe_scale_factor) * 0.001  # convert to kJ/mol
    except (LogError, NotImplementedError):
        logger.warning('Could not read e_elect from {0}'.format(path))
        e_elect = None
    return e_elect


def parse_zpe(path):
    """
    Determine the calculated ZPE from a frequency output file

    Args:
        path (str): The path to a frequency calculation output file.

    Returns:
        float: The calculated zero point energy in kJ/mol.
    """
    if not os.path.isfile(path):
        raise InputError('Could not find file {0}'.format(path))
    log = determine_qm_software(fullpath=path)
    try:
        zpe = log.load_zero_point_energy() * 0.001  # convert to kJ/mol
    except (LogError, NotImplementedError):
        logger.warning('Could not read zpe from {0}'.format(path))
        zpe = None
    return zpe


def parse_scan_energies(path):
    """
    Parse the torsion scan energies from an ESS log file.

    Args:
        path (str): The ESS log file to parse from.

    Returns:
        energies (list): The electronic energy in kJ/mol.
        angles (list): The scan angles in degrees.
    """
    if not os.path.isfile(path):
        raise InputError('Could not find file {0}'.format(path))
    log = determine_qm_software(fullpath=path)
    try:
        energies, angles = log.load_scan_energies()
        energies *= 0.001  # convert to kJ/mol
        angles *= 180 / np.pi  # convert to degrees
    except (LogError, NotImplementedError):
        logger.warning('Could not read energies from {0}'.format(path))
        energies, angles = None, None
    return energies, angles


def parse_xyz_from_file(path):
    """
    Parse xyz coordinated from:
    - .xyz: XYZ file
    - .gjf: Gaussian input file
    - .out or .log: ESS output file (Gaussian, QChem, Molpro)
    - other: Molpro or QChem input file

    Args:
        path (str): The file path.

    Returns:
        xyz (dict): The parsed coordinates.

    Raises:
        ParserError: If the coordinates could not be parsed.
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
        log = determine_qm_software(fullpath=path)
        try:
            coords, number, _ = log.load_geometry()
            xyz = xyz_from_data(coords=coords, numbers=number)
        except LogError:
            xyz = None
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


def parse_dipole_moment(path):
    """
    Parse the dipole moment in Debye from an opt job output file.
    """
    lines = _get_lines_from_file(path)
    log = determine_qm_software(path)
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
    elif isinstance(log, MolproLog):
        # example:
        #  Dipole moment /Debye                   2.96069859     0.00000000     0.00000000
        for line in lines:
            if 'dipole moment' in line.lower() and '/debye' in line.lower():
                splits = line.split()
                dm_x, dm_y, dm_z = float(splits[-3]), float(splits[-2]), float(splits[-1])
                dipole_moment = (dm_x ** 2 + dm_y ** 2 + dm_z ** 2) ** 0.5
    else:
        raise ParserError('Currently dipole moments can only be parsed from either Gaussian, Molpro, or QChem '
                          'optimization output files')
    if dipole_moment is None:
        raise ParserError('Could not parse the dipole moment')
    return dipole_moment


def parse_polarizability(path):
    """
    Parse the polarizability from a freq job output file, returns the value in Angstrom^3.
    """
    lines = _get_lines_from_file(path)
    polarizability = None
    for line in lines:
        if 'Isotropic polarizability for W' in line:
            # example:  Isotropic polarizability for W=    0.000000       11.49 Bohr**3.
            # 1 Bohr = 0.529177 Angstrom
            polarizability = float(line.split()[-2]) * 0.529177 ** 3
    return polarizability


def _get_lines_from_file(path):
    """
    A helper function for getting a list of lines from the file at `path`.
    """
    if os.path.isfile(path):
        with open(path, 'r') as f:
            lines = f.readlines()
    else:
        raise InputError('Could not find file {0}'.format(path))
    return lines


def process_conformers_file(conformers_path):
    """
    Parse coordinates and energies from an ARC conformers file of either species or TSs.

    Args:
        conformers_path (str): The path to an ARC conformers file
                               (either a "conformers_before_optimization" or
                               a "conformers_after_optimization" file).

    Returns:
        xyz (list): Entries are conformer coordinates in a dict format.
    Returns:
        energies (list): Entries float numbers representing the energies in kJ/mol.

    Raises:
        InputError: If the file could not be found.
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
            while line_index < len(lines) and lines[line_index].strip() and 'SMILES' not in lines[line_index]\
                    and 'energy' not in lines[line_index].lower() and 'guess method' not in lines[line_index].lower():
                xyz += lines[line_index]
                line_index += 1
            while line_index < len(lines) and 'conformer' not in lines[line_index]:
                if 'relative energy:' in lines[line_index].lower():
                    energy = float(lines[line_index].split()[2])
                line_index += 1
            xyzs.append(str_to_xyz(xyz))
            energies.append(energy)
        else:
            line_index += 1
    return xyzs, energies
