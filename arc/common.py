#!/usr/bin/env python
# encoding: utf-8

"""
This module contains functions which are shared across multiple ARC modules.
As such, it should not import any other ARC module (specifically ones that use logging) to avoid circular imports.

VERSION is the full ARC version, using `semantic versioning <https://semver.org/>`_.

ATOM_RADII data taken from `DOI 10.1039/b801115j <http://dx.doi.org/10.1039/b801115j>`_.
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

from rmgpy.molecule.element import getElement
from rmgpy.qm.qmdata import QMData
from rmgpy.qm.symmetry import PointGroupCalculator

from arc.settings import arc_path, servers
from arc.arc_exceptions import InputError, SettingsError

##################################################################

logger = logging.getLogger('arc')

VERSION = '1.1.0'


def time_lapse(t0):
    """
    A helper function returning the elapsed time since t0.

    Args:
        t0 (time.pyi): The initial time the count starts from.

    Returns:
        str: A "D HH:MM:SS" formatted time difference between now and t0.
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
        if ess.lower() not in ['gaussian', 'qchem', 'molpro', 'orca', 'terachem', 'onedmin', 'gromacs']:
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
        log_file (str): The log file name.
        project (str): A name for the project.
        project_directory (str, optional): The path to the project directory.
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

    # ignore Paramiko and cclib warnings:
    warnings.filterwarnings(action='ignore', module='.*paramiko.*')
    warnings.filterwarnings(action='ignore', module='.*cclib.*')
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
        project (str): The ARC project name to be logged in the header.
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
        tuple: The git HEAD commit hash and the git HEAD commit date, each as a string.
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
        str: The git branch name.
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
        execution_time (str): The overall execution time for ARC.
        level: The desired logging level.
    """
    logger.log(level, '')
    logger.log(level, 'Total execution time: {0}'.format(execution_time))
    logger.log(level, 'ARC execution terminated on {0}'.format(time.asctime()))


def read_yaml_file(path):
    """
    Read a YAML file (usually an input / restart file, but also conformers file)
    and return the parameters as python variables.

    Args:
        path (str): The YAML file path to read.

    Returns:
        dict or list: The content read from the file.
    """
    if not os.path.isfile(path):
        raise InputError('Could not find the YAML file {0}'.format(path))
    with open(path, 'r') as f:
        content = yaml.load(stream=f, Loader=yaml.FullLoader)
    return content


def save_yaml_file(path, content):
    """
    Save a YAML file (usually an input / restart file, but also conformers file)

    Args:
        path (str): The YAML file path to save.
        content (list, dict): The content to save.
    """
    yaml.add_representer(str, string_representer)
    yaml.add_representer(unicode, unicode_representer)
    logger.debug('Creating a restart file...')
    content = yaml.dump(data=content, encoding='utf-8', allow_unicode=True)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


def string_representer(dumper, data):
    """
    Add a custom string representer to use block literals for multiline strings.
    """
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data, style='|')
    return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data)


def unicode_representer(dumper, data):
    """
    Add a custom unicode representer to use block literals for multiline strings in unicode.
    """
    if len(data.splitlines()) > 1:
        return yaml.ScalarNode(tag='tag:yaml.org,2002:str', value=data, style='|')
    return yaml.ScalarNode(tag='tag:yaml.org,2002:str', value=data)


def get_ordinal_indicator(number):
    """
    Returns the ordinal indicator for an integer.

    Args:
        number (int): An integer for which the ordinal indicator will be determined.

    Returns:
        str: The integer's ordinal indicator.
    """
    ordinal_dict = {1: 'st', 2: 'nd', 3: 'rd'}
    if number > 13:
        number %= 10
    if number in ordinal_dict.keys():
        return ordinal_dict[number]
    return 'th'


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
    Get the atom covalent radius in Angstroms, data in the ATOM_RADII dict.
    (Change to QCElemental after transitioning to Py3)

    Args:
        symbol (str): The atomic symbol.

    Returns:
        float: The atomic covalent radius (None if not found).
    """
    if not isinstance(symbol, (str, unicode)):
        raise InputError('the symbol argument must be string, got {0} which is a {1}'.format(symbol, type(symbol)))

    if symbol in ATOM_RADII:
        return ATOM_RADII[symbol]
    else:
        return None


def determine_symmetry(coords, symbols):
    """
    Determine external symmetry and chirality (optical isomers) of the species.

    Args:
        coords (list): The 3D coordinates of a molecule in array-format.
        symbols (list): Entries are atomic symbols correcponding to coords.

    Returns:
        int: The external symmetry number.
    Returns:
        int: 1 if no chiral centers are present, 2 if chiral centers are present.
    """
    atom_numbers = list()  # List of atomic numbers
    for symbol in symbols:
        atom_numbers.append(getElement(str(symbol)).number)
    coords = np.array(coords, np.float64)  # N x 3 numpy.ndarray of atomic coordinates
    #  in the same order as `atom_numbers`
    unique_id = '0'  # Just some name that the SYMMETRY code gives to one of its jobs
    scr_dir = os.path.join(arc_path, str('scratch'))  # Scratch directory that the SYMMETRY code writes its files in
    if not os.path.exists(scr_dir):
        os.makedirs(scr_dir)
    symmetry = optical_isomers = 1
    qmdata = QMData(
        groundStateDegeneracy=1,  # Only needed to check if valid QMData
        numberOfAtoms=len(atom_numbers),
        atomicNumbers=atom_numbers,
        atomCoords=(coords, str('angstrom')),
        energy=(0.0, str('kcal/mol'))  # Only needed to avoid error
    )
    settings = type(str(''), (), dict(symmetryPath=str('symmetry'), scratchDirectory=scr_dir))()
    pgc = PointGroupCalculator(settings, unique_id, qmdata)
    pg = pgc.calculate()
    if pg is not None:
        symmetry = pg.symmetryNumber
        optical_isomers = 2 if pg.chiral else optical_isomers
    return symmetry, optical_isomers


def min_list(lst):
    """
    A helper function for finding the minimum of a list of integers where some of the entries might be None.
    """
    if len(lst) == 0:
        return None
    elif len(lst) == 1:
        return lst[0]
    elif all([entry is None for entry in lst]):
        return None
    return min([entry for entry in lst if entry is not None])
