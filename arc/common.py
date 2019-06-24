#!/usr/bin/env python
# encoding: utf-8

"""
ARC's common module
This module should not import any other module that has logging to avoid circular imports.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import logging
import warnings
import os
import sys
import time
import datetime
import shutil
import subprocess
import yaml
import numpy as np

from arkane.common import get_element_mass

from arc.settings import arc_path, servers
from arc.arc_exceptions import InputError, SettingsError

##################################################################

logger = logging.getLogger('arc')

VERSION = '1.0.0'


def read_file(path):
    """
    Read the ARC YAML input file and return the parameters in a dictionary.

    Args:
        (str, unicode): The input file path.

    Returns:
        dict: The input dictionary read from the file.
    """
    if not os.path.isfile(path):
        raise InputError('Could not find the input file {0}'.format(path))
    with open(path, 'r') as f:
        input_dict = yaml.load(stream=f, Loader=yaml.FullLoader)
    return input_dict


def time_lapse(t0):
    """
    A helper function returning the elapsed time since t0.

    Args:
        t0 (time.pyi): The initial time the count starts from.

    Returns:
        str, unicode: A "D HH:MM:SS" formatted time difference between now and t0.
    """
    t = time.time() - t0
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        d = str(d) + ' days, '
    else:
        d = ''
    return '{0}{1:02.0f}:{2:02.0f}:{3:02.0f}'.format(d, h, m, s)


def check_ess_settings(ess_settings=None):
    """
    A helper function to convert servers in the ess_settings dict to lists
    Assists in troubleshooting job and trying a different server
    Also check ESS and servers.

    Args:
        ess_settings (dict, optional): ARC's ESS settings dictionary.

    Returns:
        dict: An updated ARC ESS dictionary.
    """
    if ess_settings is None or not ess_settings:
        return dict()
    settings = dict()
    for software, server_list in ess_settings.items():
        if isinstance(server_list, (str, unicode)):
            settings[software] = [server_list]
        elif isinstance(server_list, list):
            for server in server_list:
                if not isinstance(server, (str, unicode)):
                    raise SettingsError('Server name could only be a string. '
                                        'Got {0} which is {1}'.format(server, type(server)))
                settings[software.lower()] = server_list
        else:
            raise SettingsError('Servers in the ess_settings dictionary could either be a string or a list of '
                                'strings. Got: {0} which is a {1}'.format(server_list, type(server_list)))
    # run checks:
    for ess, server_list in settings.items():
        if ess.lower() not in ['gaussian', 'qchem', 'molpro', 'onedmin', 'orca']:
            raise SettingsError('Recognized ESS software are Gaussian, QChem, Molpro, Orca or OneDMin. '
                                'Got: {0}'.format(ess))
        for server in server_list:
            if not isinstance(server, bool) and server.lower() not in servers.keys():
                server_names = [name for name in servers.keys()]
                raise SettingsError('Recognized servers are {0}. Got: {1}'.format(server_names, server))
    logger.info('\nUsing the following ESS settings:\n{0}\n'.format(settings))
    return settings


def initialize_log(log_file, project, project_directory=None, verbose=logging.INFO):
    """
    Set up a logger for ARC.

    Args:
        log_file (str, unicode): The log file name.
        project (str, unicode): A name for the project.
        project_directory (str, unicode, optional): The path to the project directory.
        verbose (int, optional): Specify the amount of log text seen.
    """
    # backup and delete an existing log file if needed
    if project_directory is not None and os.path.isfile(log_file):
        if not os.path.isdir(os.path.join(project_directory, 'log_and_restart_archive')):
            os.mkdir(os.path.join(project_directory, 'log_and_restart_archive'))
        local_time = datetime.datetime.now().strftime("%H%M%S_%b%d_%Y")
        log_backup_name = 'arc.old.' + local_time + '.log'
        shutil.copy(log_file, os.path.join(project_directory, 'log_and_restart_archive', log_backup_name))
        os.remove(log_file)

    logger.setLevel(verbose)
    logger.propagate = False

    # Use custom level names for cleaner log output
    logging.addLevelName(logging.CRITICAL, 'Critical: ')
    logging.addLevelName(logging.ERROR, 'Error: ')
    logging.addLevelName(logging.WARNING, 'Warning: ')
    logging.addLevelName(logging.INFO, '')
    logging.addLevelName(logging.DEBUG, '')
    logging.addLevelName(0, '')

    # Create formatter and add to handlers
    formatter = logging.Formatter('%(levelname)s%(message)s')

    # Remove old handlers before adding ours
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

    # Create console handler; send everything to stdout rather than stderr
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(verbose)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Create file handler
    fh = logging.FileHandler(filename=log_file)
    fh.setLevel(verbose)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    log_header(project=project)

    # ignore warnings from external libraries:
    warnings.filterwarnings(action='ignore', module='.*paramiko.*')
    warnings.filterwarnings(action='ignore', module='.*cclib.*')
    warnings.filterwarnings(action='ignore', module='.*rmg.*')
    logging.captureWarnings(capture=False)


def get_logger():
    """
    Get the ARC logger (avoid having multiple entries of the logger).
    """
    return logger


def log_header(project, level=logging.INFO):
    """
    Output a header containing identifying information about ARC to the log.

    Args:
        project (str, unicode): The ARC project name to be logged in the header.
        level: The desired logging level.
    """
    logger.log(level, 'ARC execution initiated on {0}'.format(time.asctime()))
    logger.log(level, '')
    logger.log(level, '###############################################################')
    logger.log(level, '#                                                             #')
    logger.log(level, '#                 Automatic Rate Calculator                   #')
    logger.log(level, '#                            ARC                              #')
    logger.log(level, '#                                                             #')
    logger.log(level, '#   Version: {0}{1}                                       #'.format(
        VERSION, ' ' * (10 - len(VERSION))))
    logger.log(level, '#                                                             #')
    logger.log(level, '###############################################################')
    logger.log(level, '')

    # Extract HEAD git commit from ARC
    head, date = get_git_commit()
    branch_name = get_git_branch()
    if head != '' and date != '':
        logger.log(level, 'The current git HEAD for ARC is:')
        logger.log(level, '    {0}\n    {1}'.format(head, date))
    if branch_name and branch_name != 'master':
        logger.log(level, '    (running on the {0} branch)\n'.format(branch_name))
    else:
        logger.log(level, '\n')
    logger.info('Starting project {0}'.format(project))


def get_git_commit():
    """
    Get the recent git commit to be logged.

    Note:
        Returns empty strings if hash and date cannot be determined.

    Returns:
        str, unicode: The git HEAD commit hash.
        str, unicode: The git HEAD commit date.
    """
    if os.path.exists(os.path.join(arc_path, '.git')):
        try:
            return subprocess.check_output(['git', 'log', '--format=%H%n%cd', '-1'], cwd=arc_path).splitlines()
        except (subprocess.CalledProcessError, OSError):
            return '', ''
    else:
        return '', ''


def get_git_branch():
    """
    Get the git branch to be logged.

    Returns:
        str, unicode: The git branch name.
    """
    if os.path.exists(os.path.join(arc_path, '.git')):
        try:
            branch_list = subprocess.check_output(['git', 'branch'], cwd=arc_path).splitlines()
        except (subprocess.CalledProcessError, OSError):
            return ''
        for branch_name in branch_list:
            if '*' in branch_name:
                return branch_name[2:]
    else:
        return ''


def log_footer(execution_time, level=logging.INFO):
    """
    Output a footer for the log.

    Args:
        execution_time (str, unicode): The overall execution time for ARC.
        level: The desired logging level.
    """
    logger.log(level, '')
    logger.log(level, 'Total execution time: {0}'.format(execution_time))
    logger.log(level, 'ARC execution terminated on {0}'.format(time.asctime()))


def save_dict_file(path, restart_dict):
    """
    Save an input / restart YAML file
    """
    yaml.add_representer(str, string_representer)
    yaml.add_representer(unicode, unicode_representer)
    logger.debug('Creating a restart file...')
    content = yaml.dump(data=restart_dict, encoding='utf-8', allow_unicode=True)
    with open(path, 'w') as f:
        f.write(content)


def string_representer(dumper, data):
    """Add a custom string representer to use block literals for multiline strings"""
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data, style='|')
    return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data)


def unicode_representer(dumper, data):
    """Add a custom unicode representer to use block literals for multiline strings"""
    if len(data.splitlines()) > 1:
        return yaml.ScalarNode(tag='tag:yaml.org,2002:str', value=data, style='|')
    return yaml.ScalarNode(tag='tag:yaml.org,2002:str', value=data)


ATOM_RADII = {'H': 0.31, 'He': 0.28,
              'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
              'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
              'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.39, 'Fe': 1.32,
              'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19,
              'Se': 1.20, 'Br': 1.20, 'Kr': 1.16, 'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75,
              'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45,
              'Cd': 1.44, 'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40,
              'Cs': 2.44, 'Ba': 2.15, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46,
              'Bi': 1.48, 'Po': 1.40, 'At': 1.50, 'Rn': 1.50, 'Fr': 2.60, 'Ra': 2.21, 'U': 1.96}


def get_atom_radius(symbol):
    """
    Get the atom covalent radius in Angstroms, data in the ATOM_RADII dict taken from DOI: 10.1039/b801115j.
    Change to QCElemental after transitioning to Py3.

    Args:
        symbol (str, unicode): The atomic symbol.

    Returns:
        float: The atomic covalent radius (None if not found).
    """
    if not isinstance(symbol, (str, unicode)):
        raise InputError('the symbol argument must be string, got {0} which is a {1}'.format(symbol, type(symbol)))

    if symbol in ATOM_RADII:
        return ATOM_RADII[symbol]
    else:
        return None


def get_center_of_mass(xyz):
    """
    Get the center of mass of xyz coordinates.
    Assumes arc.converter.standardize_xyz_string() was already called for xyz.
    Note that xyz from ESS output is usually already centered at the center of mass (to some precision).

    Args:
        xyz (list, string, unicode): The xyz coordinates in a string.

    Returns:
        tuple: The center of mass coordinates.
    """
    masses, coords = list(), list()
    for line in xyz.splitlines():
        if line.strip():
            splits = line.split()
            masses.append(get_element_mass(str(splits[0]))[0])
            coords.append([float(splits[1]), float(splits[2]), float(splits[3])])
    cm_x, cm_y, cm_z = 0, 0, 0
    for coord, mass in zip(coords, masses):
        cm_x += coord[0] * mass
        cm_y += coord[1] * mass
        cm_z += coord[2] * mass
    cm_x /= sum(masses)
    cm_y /= sum(masses)
    cm_z /= sum(masses)
    return cm_x, cm_y, cm_z
