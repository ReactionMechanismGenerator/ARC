#!/usr/bin/env python
# encoding: utf-8


"""
A module for parsing information from files
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import os

from arkane.statmech import determine_qm_software
from arkane.qchem import QChemLog
from arkane.gaussian import GaussianLog
from arkane.molpro import MolproLog

from arc.common import get_logger
from arc.species.converter import get_xyz_string, standardize_xyz_string
from arc.arc_exceptions import InputError, ParserError

"""
Various ESS parsing tools
"""

##################################################################

logger = get_logger()


def parse_frequencies(path, software):
    """
    Parse the frequencies from a freq job output file
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
    Parse the T1 parameter from a Molpro coupled cluster calculation
    """
    lines = _get_lines_from_file(path)
    t1 = None
    for line in lines:
        if 'T1 diagnostic:' in line:
            t1 = float(line.split()[-1])
    return t1


def parse_e_elect(path, zpe_scale_factor=1.):
    """
    Parse the zero K energy, E0, from an sp job output file
    """
    if not os.path.isfile(path):
        raise InputError('Could not find file {0}'.format(path))
    log = determine_qm_software(fullpath=path)
    try:
        e_elect = log.loadEnergy(zpe_scale_factor) * 0.001  # convert to kJ/mol
    except Exception:
        logger.warning('Could not read e_elect from {0}'.format(path))
        e_elect = None
    return e_elect


def parse_xyz_from_file(path):
    """
    Parse xyz coordinated from:
    .xyz - XYZ file
    .gjf - Gaussian input file
    .out or .log - ESS output file (Gaussian, QChem, Molpro)
    other - Molpro or QChem input file
    """
    lines = _get_lines_from_file(path)
    file_extension = os.path.splitext(path)[1]

    xyz = None
    relevant_lines = list()

    if file_extension == '.xyz':
        relevant_lines = lines[2:]
    elif file_extension == '.gjf':
        for line in lines[5:]:
            if line and line != '\n' and line != '\r\n':
                relevant_lines.append(line)
            else:
                break
    elif 'out' in file_extension or 'log' in file_extension:
        log = determine_qm_software(fullpath=path)
        coord, number, _ = log.loadGeometry()
        xyz = get_xyz_string(coord=coord, number=number)
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
            raise ParserError('Could not parse xyz coordinates from file {0}'.format(path))
    if xyz is None and relevant_lines:
        xyz = ''.join([line for line in relevant_lines if line])
    return standardize_xyz_string(xyz)


def parse_dipole_moment(path):
    """
    Parse the dipole moment in Debye from an opt job output file
    """
    if path is None:
        logger.warning('Could not parse dipole moment, opt path is None')
        return None
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
    Parse the polarizability from a freq job output file, returns the value in Angstrom^3
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
    """A helper function for getting a list of lines from the file at `path`"""
    if os.path.isfile(path):
        with open(path, 'r') as f:
            lines = f.readlines()
    else:
        raise InputError('Could not find file {0}'.format(path))
    return lines
