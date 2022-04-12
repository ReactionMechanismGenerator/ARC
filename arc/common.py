"""
This module contains functions which are shared across multiple ARC modules.
As such, it should not import any other ARC module (specifically ones that use the logger defined here)
to avoid circular imports.

VERSION is the full ARC version, using `semantic versioning <https://semver.org/>`_.
"""

import ast
import datetime
import logging
import os
import pprint
import re
import shutil
import subprocess
import sys
import time
import warnings
import yaml
from itertools import chain
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import qcelemental as qcel

from arkane.ess import ess_factory, GaussianLog, MolproLog, OrcaLog, QChemLog, TeraChemLog
import rmgpy
from rmgpy.molecule.atomtype import ATOMTYPES
from rmgpy.molecule.element import get_element
from rmgpy.molecule.molecule import Atom, Bond, Molecule
from rmgpy.qm.qmdata import QMData
from rmgpy.qm.symmetry import PointGroupCalculator

from arc.exceptions import InputError, SettingsError
from arc.imports import settings


if TYPE_CHECKING:
    from rmgpy.reaction import Reaction
    from arc.reaction import ARCReaction


logger = logging.getLogger('arc')

# Absolute path to the ARC folder.
ARC_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# Absolute path to RMG-Py folder.
RMG_PATH = os.path.abspath(os.path.dirname(os.path.dirname(rmgpy.__file__)))
# Absolute path to RMG-database folder.
RMG_DATABASE_PATH = os.path.abspath(os.path.dirname(rmgpy.settings['database.directory']))

VERSION = '1.1.0'


default_job_types, servers = settings['default_job_types'], settings['servers']


def initialize_job_types(job_types: dict,
                         specific_job_type: str = '',
                         ) -> dict:
    """
    A helper function for initializing job_types.
    Returns the comprehensive (default values for missing job types) job types for ARC.

    Args:
        job_types (dict): Keys are job types, values are booleans of whether or not to consider this job type.
        specific_job_type (str): Specific job type to execute. Legal strings are job types (keys of job_types dict).

    Returns: dict
        An updated (comprehensive) job type dictionary.
    """
    if specific_job_type:
        logger.info(f'Specific_job_type {specific_job_type} was requested by the user.')
        if job_types:
            logger.warning('Both job_types and specific_job_type were given, use only specific_job_type to '
                           'populate the job_types dictionary.')
        job_types = {job_type: False for job_type in default_job_types.keys()}
        try:
            job_types[specific_job_type] = True
        except KeyError:
            raise InputError(f'Specified job type {specific_job_type} is not supported.')

    if specific_job_type == 'bde':
        bde_default = {'opt': True, 'fine_grid': True, 'freq': True, 'sp': True}
        job_types.update(bde_default)

    defaults_to_true = ['conformers', 'fine', 'freq', 'irc', 'opt', 'rotors', 'sp']
    defaults_to_false = ['bde', 'onedmin', 'orbitals']
    if job_types is None:
        job_types = default_job_types
        logger.info("Job types were not specified, using ARC's defaults")
    else:
        logger.debug(f'the following job types were specified: {job_types}.')
    if 'lennard_jones' in job_types:
        job_types['onedmin'] = job_types['lennard_jones']
        del job_types['lennard_jones']
    if 'fine_grid' in job_types:
        job_types['fine'] = job_types['fine_grid']
        del job_types['fine_grid']
    for job_type in defaults_to_true:
        if job_type not in job_types:
            # set default value to True if this job type key is missing
            job_types[job_type] = True
    for job_type in defaults_to_false:
        if job_type not in job_types:
            # set default value to False if this job type key is missing
            job_types[job_type] = False
    for job_type in job_types.keys():
        if job_type not in defaults_to_true and job_type not in defaults_to_false:
            if job_type == '1d_rotors':
                logging.error("Note: The `1d_rotors` job type was renamed to simply `rotors`. "
                              "Please modify your input accordingly (see ARC's documentation for examples).")
            raise InputError(f"Job type '{job_type}' is not supported. Check the job types dictionary "
                             "(either in ARC's input or in default_job_types under settings).")
    job_types_report = [job_type for job_type, val in job_types.items() if val]
    logger.info(f'\nConsidering the following job types: {job_types_report}\n')
    return job_types


def determine_ess(log_file: str) -> str:
    """
    Determine the ESS to which the log file belongs.

    Args:
        log_file (str): The ESS log file path.

    Returns: str
        The ESS log class from Arkane.
    """
    log = ess_factory(log_file, check_for_errors=False)
    if isinstance(log, GaussianLog):
        return 'gaussian'
    if isinstance(log, MolproLog):
        return 'molpro'
    if isinstance(log, OrcaLog):
        return 'orca'
    if isinstance(log, QChemLog):
        return 'qchem'
    if isinstance(log, TeraChemLog):
        return 'terachem'
    raise InputError(f'Could not identify the log file in {log_file} as belonging to '
                     f'Gaussian, Molpro, Orca, QChem, or TeraChem.')


def check_ess_settings(ess_settings: Optional[dict] = None) -> dict:
    """
    A helper function to convert servers in the ess_settings dict to lists
    Assists in troubleshooting job and trying a different server
    Also check ESS and servers.

    Args:
        ess_settings (dict, optional): ARC's ESS settings dictionary.

    Returns: dict
        An updated ARC ESS dictionary.
    """
    if ess_settings is None or not ess_settings:
        return dict()
    settings_dict = dict()
    for software, server_list in ess_settings.items():
        if isinstance(server_list, str):
            settings_dict[software] = [server_list]
        elif isinstance(server_list, list):
            for server in server_list:
                if not isinstance(server, str):
                    raise SettingsError(f'Server name could only be a string. Got {server} which is {type(server)}')
                settings_dict[software.lower()] = server_list
        else:
            raise SettingsError(f'Servers in the ess_settings dictionary could either be a string or a list of '
                                f'strings. Got: {server_list} which is a {type(server_list)}')
    # run checks:
    for ess, server_list in settings_dict.items():
        if ess.lower() not in ['gaussian', 'qchem', 'molpro', 'orca', 'terachem', 'onedmin', 'psi4']:
            raise SettingsError(f'Recognized ESS software are Gaussian, QChem, Molpro, Orca, TeraChem, Psi4, '
                                f'or OneDMin. Got: {ess}')
        for server in server_list:
            if not isinstance(server, bool) and server.lower() not in [s.lower() for s in servers.keys()]:
                server_names = [name for name in servers.keys()]
                raise SettingsError(f'Recognized servers are {server_names}. Got: {server}')
    logger.info(f'\nUsing the following ESS settings:\n{pprint.pformat(settings_dict)}\n')
    return settings_dict


def initialize_log(log_file: str,
                   project: str,
                   project_directory: Optional[str] = None,
                   verbose: int = logging.INFO,
                   ) -> None:
    """
    Set up a logger for ARC.

    Args:
        log_file (str): The log file name.
        project (str): A name for the project.
        project_directory (str, optional): The path to the project directory.
        verbose (int, optional): Specify the amount of log text seen.
    """
    # Backup and delete an existing log file if needed.
    if project_directory is not None and os.path.isfile(log_file):
        if not os.path.isdir(os.path.join(project_directory, 'log_and_restart_archive')):
            os.mkdir(os.path.join(project_directory, 'log_and_restart_archive'))
        local_time = datetime.datetime.now().strftime("%H%M%S_%b%d_%Y")
        log_backup_name = 'arc.old.' + local_time + '.log'
        shutil.copy(log_file, os.path.join(project_directory, 'log_and_restart_archive', log_backup_name))
        os.remove(log_file)

    logger.setLevel(verbose)
    logger.propagate = False

    # Use custom level names for cleaner log output.
    logging.addLevelName(logging.CRITICAL, 'Critical: ')
    logging.addLevelName(logging.ERROR, 'Error: ')
    logging.addLevelName(logging.WARNING, 'Warning: ')
    logging.addLevelName(logging.INFO, '')
    logging.addLevelName(logging.DEBUG, '')
    logging.addLevelName(0, '')

    # Create formatter and add to handlers.
    formatter = logging.Formatter('%(levelname)s%(message)s')

    # Remove old handlers before adding ours.
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

    # Create console handler; send everything to stdout rather than stderr.
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(verbose)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Create file handler.
    fh = logging.FileHandler(filename=log_file)
    fh.setLevel(verbose)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    log_header(project=project)

    # Ignore Paramiko, cclib, and matplotlib warnings:
    warnings.filterwarnings(action='ignore', module='.*paramiko.*')
    warnings.filterwarnings(action='ignore', module='.*cclib.*')
    warnings.filterwarnings(action='ignore', module='.*matplotlib.*')
    logging.captureWarnings(capture=False)


def get_logger():
    """
    Get the ARC logger (avoid having multiple entries of the logger).
    """
    return logger


def log_header(project: str,
               level: int = logging.INFO,
               ) -> None:
    """
    Output a header containing identifying information about ARC to the log.

    Args:
        project (str): The ARC project name to be logged in the header.
        level (int, optional): The desired logging level.
    """
    logger.log(level, f'ARC execution initiated on {time.asctime()}')
    logger.log(level, '')
    logger.log(level, '###############################################################')
    logger.log(level, '#                                                             #')
    logger.log(level, '#                 Automatic Rate Calculator                   #')
    logger.log(level, '#                            ARC                              #')
    logger.log(level, '#                                                             #')
    logger.log(level, f'#   Version: {VERSION}{" " * (10 - len(VERSION))}                                       #')
    logger.log(level, '#                                                             #')
    logger.log(level, '###############################################################')
    logger.log(level, '')

    paths_dict = {'ARC': ARC_PATH, 'RMG-Py': RMG_PATH, 'RMG-database': RMG_DATABASE_PATH}
    for repo, path in paths_dict.items():
        # Extract HEAD git commit.
        head, date = get_git_commit(path)
        branch_name = get_git_branch(path)
        if head != '' and date != '':
            logger.log(level, f'The current git HEAD for {repo} is:')
            logger.log(level, f'    {head}\n    {date}')
        if branch_name and branch_name != 'main':
            logger.log(level, f'    (running on the {branch_name} branch)\n')
        else:
            logger.log(level, '\n')

    logger.info(f'Starting project {project}\n\n')


def log_footer(execution_time: str,
               level: int = logging.INFO,
               ) -> None:
    """
    Output a footer for the log.

    Args:
        execution_time (str): The overall execution time for ARC.
        level (int, optional): The desired logging level.
    """
    logger.log(level, '')
    logger.log(level, f'Total execution time: {execution_time}')
    logger.log(level, f'ARC execution terminated on {time.asctime()}')


def get_git_commit(path: Optional[str] = None) -> Tuple[str, str]:
    """
    Get the recent git commit to be logged.

    Note:
        Returns empty strings if hash and date cannot be determined.

    Args:
        path (str, optional): The path to check.

    Returns: tuple
        The git HEAD commit hash and the git HEAD commit date, each as a string.
    """
    path = path or ARC_PATH
    head, date = '', ''
    if os.path.exists(os.path.join(path, '.git')):
        try:
            head, date = subprocess.check_output(['git', 'log', '--format=%H%n%cd', '-1'], cwd=path).splitlines()
            head, date = head.decode(), date.decode()
        except (subprocess.CalledProcessError, OSError):
            return head, date
    return head, date


def get_git_branch(path: Optional[str] = None) -> str:
    """
    Get the git branch to be logged.

    Args:
        path (str, optional): The path to check.

    Returns: str
        The git branch name.
    """
    path = path or ARC_PATH
    if os.path.exists(os.path.join(path, '.git')):
        try:
            branch_list = subprocess.check_output(['git', 'branch'], cwd=path).splitlines()
        except (subprocess.CalledProcessError, OSError):
            return ''
        for branch_name in branch_list:
            if '*' in branch_name.decode():
                return branch_name.decode()[2:]
    else:
        return ''


def read_yaml_file(path: str,
                   project_directory: Optional[str] = None,
                   ) -> Union[dict, list]:
    """
    Read a YAML file (usually an input / restart file, but also conformers file)
    and return the parameters as python variables.

    Args:
        path (str): The YAML file path to read.
        project_directory (str, optional): The current project directory to rebase upon.

    Returns: Union[dict, list]
        The content read from the file.
    """
    if project_directory is not None:
        path = globalize_paths(path, project_directory)
    if not isinstance(path, str):
        raise InputError(f'path must be a string, got {path} which is a {type(path)}')
    if not os.path.isfile(path):
        raise InputError(f'Could not find the YAML file {path}')
    with open(path, 'r') as f:
        content = yaml.load(stream=f, Loader=yaml.FullLoader)
    return content


def save_yaml_file(path: str,
                   content: Union[list, dict],
                   ) -> None:
    """
    Save a YAML file (usually an input / restart file, but also conformers file).

    Args:
        path (str): The YAML file path to save.
        content (list, dict): The content to save.
    """
    if not isinstance(path, str):
        raise InputError(f'path must be a string, got {path} which is a {type(path)}')
    yaml_str = to_yaml(py_content=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(yaml_str)


def from_yaml(yaml_str: str) -> Union[dict, list]:
    """
    Read a YAML string and decode to the respective Python object.
    Args:
        yaml_str (str): The YAML string content.
    Returns: Union[dict, list]
        The respective Python object.
    """
    return yaml.load(stream=yaml_str, Loader=yaml.FullLoader)


def to_yaml(py_content: Union[list, dict]) -> str:
    """
    Convert a Python list or dictionary to a YAML string format.

    Args:
        py_content (list, dict): The Python content to save.

    Returns: str
        The corresponding YAML representation.
    """
    yaml.add_representer(str, string_representer)
    yaml_str = yaml.dump(data=py_content)
    return yaml_str


def globalize_paths(file_path: str,
                    project_directory: str,
                    ) -> str:
    """
    Rebase all file paths in the contents of the given file on the current project path.
    Useful when restarting an ARC project in a different folder or on a different machine.

    Args:
        file_path (str): A path to the file to check.
                         The contents of this file will be changed and saved as a different file.
        project_directory (str): The current project directory to rebase upon.

    Returns: str
        A path to the respective file with rebased absolute file paths.
    """
    modified = False
    new_lines = list()
    if project_directory[-1] != '/':
        project_directory += '/'
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        new_line = globalize_path(line, project_directory)
        modified = modified or new_line != line
        new_lines.append(new_line)
    if modified:
        base_name, file_name = os.path.split(file_path)
        file_name_splits = file_name.split('.')
        new_file_name = '.'.join(file_name_splits[:-1]) + '_globalized.' + str(file_name_splits[-1])
        new_path = os.path.join(base_name, new_file_name)
        with open(new_path, 'w') as f:
            f.writelines(new_lines)
        return new_path
    else:
        return file_path


def globalize_path(string: str,
                   project_directory: str,
                   ) -> str:
    """
    Rebase an absolute file path on the current project path.
    Useful when restarting an ARC project in a different folder or on a different machine.

    Args:
        string (str): A string containing a path to rebase.
        project_directory (str): The current project directory to rebase upon.

    Returns: str
        A string with the rebased path.
    """
    if '/calcs/Species/' in string or '/calcs/TSs/' in string and project_directory not in string:
        splits = string.split('/calcs/')
        prefix = splits[0].split('/')[0]
        new_string = prefix + project_directory
        new_string += '/' if new_string[-1] != '/' else ''
        new_string += 'calcs/' + splits[-1]
        return new_string
    return string


def string_representer(dumper, data):
    """
    Add a custom string representer to use block literals for multiline strings.
    """
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data, style='|')
    return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data)


def get_ordinal_indicator(number: int) -> str:
    """
    Returns the ordinal indicator for an integer.

    Args:
        number (int): An integer for which the ordinal indicator will be determined.

    Returns: str
        The integer's ordinal indicator.
    """
    ordinal_dict = {1: 'st', 2: 'nd', 3: 'rd'}
    if number > 13:
        number %= 10
    if number in list(ordinal_dict.keys()):
        return ordinal_dict[number]
    return 'th'


def get_number_with_ordinal_indicator(number: int) -> str:
    """
    Returns the number as a string with the ordinal indicator.

    Args:
        number (int): An integer for which the ordinal indicator will be determined.

    Returns: str
        The number with the respective ordinal indicator.
    """
    return f'{number}{get_ordinal_indicator(number)}'


def get_atom_radius(symbol: str) -> float:
    """
    Get the atom covalent radius of an atom in Angstroms.

    Args:
        symbol (str): The atomic symbol.

    Raises:
        TypeError: If ``symbol`` is of wrong type.

    Returns: float
        The atomic covalent radius (None if not found).
    """
    if not isinstance(symbol, str):
        raise TypeError(f'The symbol argument must be string, got {symbol} which is a {type(symbol)}')
    try:
        r = qcel.covalentradii.get(symbol, units='angstrom')
    except qcel.exceptions.NotAnElementError:
        r = None
    return r


# A bond length dictionary of single bonds in Angstrom.
# https://sites.google.com/site/chempendix/bond-lengths
# https://courses.lumenlearning.com/suny-potsdam-organicchemistry/chapter/1-3-basics-of-bonding/
# 'N-O' is taken from NH2OH, 'N+_N+' and 'N+_O-' are taken from N2O4.
# 'H_H' was artificially modified from 0.74 to 1.0 since it collides quickly at 0.55.
SINGLE_BOND_LENGTH = {'Br_Br': 2.29, 'Br_Cr': 1.94, 'Br_H': 1.41,
                      'C_C': 1.54, 'C_Cl': 1.77, 'C_F': 1.35, 'C_H': 1.09, 'C_I': 2.13,
                      'C_N': 1.47, 'C_O': 1.43, 'C_P': 1.87, 'C_S': 1.81, 'C_Si': 1.86,
                      'Cl_Cl': 1.99, 'Cl_H': 1.27, 'Cl_N': 1.75, 'Cl_Si': 2.03, 'Cl_P': 2.03, 'Cl_S': 2.07,
                      'F_F': 1.42, 'F_H': 0.92, 'F_P': 1.57, 'F_S': 1.56, 'F_Si': 1.56, 'F_Xe': 1.90,
                      'H_H': 1.0, 'H_I': 1.61, 'H_N': 1.04, 'H_O': 0.96, 'H_P': 1.42, 'H_S': 1.34, 'H_Si': 1.48,
                      'I_I': 2.66,
                      'N_N': 1.45, 'N+1_N+1': 1.81, 'N_O': 1.44, 'N+1_O-1': 1.2,
                      'O_O': 1.48, 'O_P': 1.63, 'O_S': 1.58, 'O_Si': 1.66,
                      'P_P': 2.21,
                      'S_S': 2.05,
                      'Si_Si': 2.35,
                      }


def get_single_bond_length(symbol_1: str,
                           symbol_2: str,
                           charge_1: int = 0,
                           charge_2: int = 0,
                           ) -> float:
    """
    Get an approximate for a single bond length between two elements.

    Args:
        symbol_1 (str): Symbol 1.
        symbol_2 (str): Symbol 2.
        charge_1 (int, optional): The partial charge of the atom represented by ``symbol_1``.
        charge_2 (int, optional): The partial charge of the atom represented by ``symbol_2``.

    Returns: float
        The estimated single bond length in Angstrom.
    """
    if charge_1 and charge_2:
        symbol_1 = f"{symbol_1}{'+' if charge_1 > 0 else ''}{charge_1}"
        symbol_2 = f"{symbol_2}{'+' if charge_2 > 0 else ''}{charge_2}"
    bond1, bond2 = '_'.join([symbol_1, symbol_2]), '_'.join([symbol_2, symbol_1])
    if bond1 in SINGLE_BOND_LENGTH.keys():
        return SINGLE_BOND_LENGTH[bond1]
    if bond2 in SINGLE_BOND_LENGTH.keys():
        return SINGLE_BOND_LENGTH[bond2]
    return 2.5


def get_bonds_from_dmat(dmat: np.ndarray,
                        elements: Union[Tuple[str, ...], List[str]],
                        charges: Optional[List[int]] = None,
                        tolerance: float = 1.2,
                        bond_lone_hydrogens: bool = True,
                        ) -> List[Tuple[int, int]]:
    """
    Guess the connectivity of a molecule from its distance matrix representation.

    Args:
        dmat (np.ndarray): An NxN matrix of atom distances in Angstrom.
        elements (List[str]): The corresponding element list in the same atomic order.
        charges (List[int], optional): A corresponding list of formal atomic charges.
        tolerance (float, optional): A factor by which the single bond length threshold is multiplied for the check.
        bond_lone_hydrogens (bool, optional): Whether to assign a bond to hydrogen atoms which were not identified
                                              as bonded. If so, the closest atom will be considered.

    Returns:
        List[Tuple[int, int]]:
            A list of tuple entries, each represents a bond and contains sorted atom indices.
    """
    if len(elements) != dmat.shape[0] or len(elements) != dmat.shape[1] or len(dmat.shape) != 2:
        raise ValueError(f'The dimensions of the DMat {dmat.shape} must be equal to the number of elements {len(elements)}')
    bonds, bonded_hydrogens = list(), list()
    charges = charges or [0] * len(elements)
    # Heavy atoms
    for i, e_1 in enumerate(elements):
        for j, e_2 in enumerate(elements):
            if i > j and e_1 != 'H' and e_2 != 'H' and dmat[i, j] < tolerance * \
                    get_single_bond_length(symbol_1=e_1,
                                           symbol_2=e_2,
                                           charge_1=charges[i],
                                           charge_2=charges[j]):
                bonds.append(tuple(sorted([i, j])))
    # Hydrogen atoms except for H--H bonds, make sure each H has only one bond (the closest heavy atom).
    for i, e_1 in enumerate(elements):
        if e_1 == 'H':
            j = get_extremum_index(lst=dmat[i], return_min=True, skip_values=[0])
            if i != j and (elements[j] != 'H'):
                bonds.append(tuple(sorted([i, j])))
                bonded_hydrogens.append(i)
    # Lone hydrogens, also important for the H2 molecule.
    if bond_lone_hydrogens:
        for i, e_1 in enumerate(elements):
            j = get_extremum_index(lst=dmat[i], return_min=True, skip_values=[0])
            bond = tuple(sorted([i, j]))
            if i != j and e_1 == 'H' and i not in bonded_hydrogens and j not in bonded_hydrogens and bond not in bonds:
                bonds.append(bond)
    return bonds


def determine_symmetry(xyz: dict) -> Tuple[int, int]:
    """
    Determine external symmetry and chirality (optical isomers) of the species.

    Args:
        xyz (dict): The 3D coordinates.

    Returns: Tuple[int, int]
        - The external symmetry number.
        - ``1`` if no chiral centers are present, ``2`` if chiral centers are present.
    """
    atom_numbers = list()
    for symbol in xyz['symbols']:
        atom_numbers.append(get_element(symbol).number)
    # Coords is an N x 3 numpy.ndarray of atomic coordinates in the same order as `atom_numbers`.
    coords = np.array(xyz['coords'], np.float64)
    unique_id = '0'  # Just some name that the SYMMETRY code gives to one of its jobs.
    scr_dir = os.path.join('/tmp', 'symmetry_scratch')  # Scratch directory that the SYMMETRY code writes its files in.
    if not os.path.exists(scr_dir):
        os.makedirs(scr_dir)
    symmetry = optical_isomers = 1
    qmdata = QMData(
        groundStateDegeneracy=1,  # Only needed to check if valid QMData.
        numberOfAtoms=len(atom_numbers),
        atomicNumbers=atom_numbers,
        atomCoords=(coords, 'angstrom'),
        energy=(0.0, 'kcal/mol')  # Dummy
    )
    symmetry_settings = type('', (), dict(symmetryPath='symmetry', scratchDirectory=scr_dir))()
    pgc = PointGroupCalculator(symmetry_settings, unique_id, qmdata)
    pg = pgc.calculate()
    if pg is not None:
        symmetry = pg.symmetry_number
        optical_isomers = 2 if pg.chiral else optical_isomers
    return symmetry, optical_isomers


def determine_top_group_indices(mol, atom1, atom2, index=1) -> Tuple[list, bool]:
    """
    Determine the indices of a "top group" in a molecule.
    The top is defined as all atoms connected to atom2, including atom2, excluding the direction of atom1.
    Two ``atom_list_to_explore`` are used so the list the loop iterates through isn't changed within the loop.

    Args:
        mol (Molecule): The Molecule object to explore.
        atom1 (Atom): The pivotal atom in mol.
        atom2 (Atom): The beginning of the top relative to atom1 in mol.
        index (bool, optional): Whether to return 1-index or 0-index conventions. 1 for 1-index.

    Returns: Tuple[list, bool]
        - The indices of the atoms in the top (either 0-index or 1-index, as requested).
        - Whether the top has heavy atoms (is not just a hydrogen atom). ``True`` if it has heavy atoms.
    """
    top = list()
    explored_atom_list, atom_list_to_explore1, atom_list_to_explore2 = [atom1], [atom2], []
    while len(atom_list_to_explore1 + atom_list_to_explore2):
        for atom3 in atom_list_to_explore1:
            top.append(mol.vertices.index(atom3) + index)
            for atom4 in atom3.edges.keys():
                if atom4 not in explored_atom_list and atom4 not in atom_list_to_explore2:
                    if atom4.is_hydrogen():
                        # append H w/o further exploring
                        top.append(mol.vertices.index(atom4) + index)
                    else:
                        atom_list_to_explore2.append(atom4)  # explore it further
            explored_atom_list.append(atom3)  # mark as explored
        atom_list_to_explore1, atom_list_to_explore2 = atom_list_to_explore2, []
    return top, not atom2.is_hydrogen()


def extremum_list(lst: list,
                  return_min: bool = True,
                  ) -> Optional[Union[int, None]]:
    """
    A helper function for finding the extremum (either minimum or maximum) of a list of numbers (int/float)
    where some of the entries might be ``None``.

    Args:
        lst (list): The list.
        return_min (bool, optional): Whether to return the minimum or the maximum.
                                    ``True`` for minimum, ``False`` for maximum, ``True`` by default.

    Returns: int
        The entry with the minimal/maximal value.
    """
    if len(lst) == 0:
        return None
    elif len(lst) == 1:
        return lst[0]
    elif all([entry is None for entry in lst]):
        return None
    if return_min:
        return min([entry for entry in lst if entry is not None])
    else:
        return max([entry for entry in lst if entry is not None])


def sort_two_lists_by_the_first(list1: List[Union[float, int, None]],
                                list2: List[Union[float, int, None]],
                                ) -> Tuple[List[Union[float, int]], List[Union[float, int]]]:
    """
    Sort two lists in increasing order by the values of the first list.
    Ignoring None entries from list1 and their respective entries in list2.
    The function was written in this format rather the more pytonic ``zip(*sorted(zip(list1, list2)))`` style
    to accommodate for dictionaries as entries of list2, otherwise a
    ``TypeError: '<' not supported between instances of 'dict' and 'dict'`` error is raised.

    Args:
        list1 (list, tuple): Entries are floats or ints (could also be None).
        list2 (list, tuple): Entries could be anything.

    Raises:
        InputError: If types are wrong, or lists are not the same length.

    Returns: Tuple[list, list]
        - Sorted values from list1, ignoring None entries.
        - Respective entries from list2.
    """
    if not isinstance(list1, (list, tuple)) or not isinstance(list2, (list, tuple)):
        raise InputError(f'Arguments must be lists, got: {type(list1)} and {type(list2)}')
    for entry in list1:
        if not isinstance(entry, (float, int)) and entry is not None:
            raise InputError(f'Entries of list1 must be either floats or integers, got: {type(entry)}.')
    if len(list1) != len(list2):
        raise InputError(f'Both lists must be the same length, got {len(list1)} and {len(list2)}')

    # remove None entries from list1 and their respective entries from list2:
    new_list1, new_list2 = list(), list()
    for entry1, entry2 in zip(list1, list2):
        if entry1 is not None:
            new_list1.append(entry1)
            new_list2.append(entry2)
    indices = list(range(len(new_list1)))

    zipped_lists = zip(new_list1, indices)
    sorted_lists = sorted(zipped_lists)
    sorted_list1 = [x for x, _ in sorted_lists]
    sorted_indices = [x for _, x in sorted_lists]
    sorted_list2 = [0] * len(new_list2)
    for counter, index in enumerate(sorted_indices):
        sorted_list2[counter] = new_list2[index]
    return sorted_list1, sorted_list2


def key_by_val(dictionary: dict,
               value: Any,
               ) -> Any:
    """
    A helper function for getting a key from a dictionary corresponding to a certain value.
    Does not check for value unicity.

    Args:
        dictionary (dict): The dictionary.
        value: The value.

    Raises:
        ValueError: If the value could not be found in the dictionary.

    Returns: Any
        The key.
    """
    for key, val in dictionary.items():
        if val == value or (isinstance(value, int) and val == f'X{value}'):
            return key
    raise ValueError(f'Could not find value {value} in the dictionary\n{dictionary}')


def almost_equal_lists(iter1: Union[list, tuple, np.ndarray],
                       iter2: Union[list, tuple, np.ndarray],
                       rtol: float = 1e-05,
                       atol: float = 1e-08,
                       ) -> bool:
    """
    A helper function for checking whether two iterables are almost equal.

    Args:
        iter1 (list, tuple, np.array): An iterable.
        iter2 (list, tuple, np.array): An iterable.
        rtol (float, optional): The relative tolerance parameter.
        atol (float, optional): The absolute tolerance parameter.

    Returns: bool
        ``True`` if they are almost equal, ``False`` otherwise.
    """
    if len(iter1) != len(iter2):
        return False
    for entry1, entry2 in zip(iter1, iter2):
        if isinstance(entry1, (list, tuple, np.ndarray)) and isinstance(entry2, (list, tuple, np.ndarray)):
            return almost_equal_lists(iter1=entry1, iter2=entry2, rtol=rtol, atol=atol)
        else:
            if isinstance(entry1, (int, float)) and isinstance(entry2, (int, float)):
                if not np.isclose([entry1], [entry2], rtol=rtol, atol=atol):
                    return False
            else:
                if entry1 != entry2:
                    return False
    return True


def almost_equal_coords(xyz1: dict,
                        xyz2: dict,
                        rtol: float = 1e-05,
                        atol: float = 1e-08,
                        ) -> bool:
    """
    A helper function for checking whether two xyz's are almost equal. Also checks equal symbols.

    Args:
        xyz1 (dict): Cartesian coordinates.
        xyz2 (dict): Cartesian coordinates.
        rtol (float, optional): The relative tolerance parameter.
        atol (float, optional): The absolute tolerance parameter.

    Returns: bool
        ``True`` if they are almost equal, ``False`` otherwise.
    """
    if not isinstance(xyz1, dict) or not isinstance(xyz2, dict):
        raise TypeError(f'xyz1 and xyz2 must be dictionaries, got {type(xyz1)} and {type(xyz2)}:\n{xyz1}\n{xyz2}')
    for symbol_1, symbol_2 in zip(xyz1['symbols'], xyz2['symbols']):
        if symbol_1 != symbol_2:
            logger.warning(f"Cannot compare coords, xyz1 and xyz2 have different symbols:"
                           f"\n{xyz1['symbols']}\nand:\n{xyz2['symbols']}")
    for xyz_coord1, xyz_coord2 in zip(xyz1['coords'], xyz2['coords']):
        for xyz1_c, xyz2_c in zip(xyz_coord1, xyz_coord2):
            if not np.isclose([xyz1_c], [xyz2_c], rtol=rtol, atol=atol):
                return False
    return True


def almost_equal_coords_lists(xyz1: Union[List[dict], dict],
                              xyz2: Union[List[dict], dict],
                              rtol: float = 1e-05,
                              atol: float = 1e-08,
                              ) -> bool:
    """
    A helper function for checking two lists of xyzs has at least one entry in each that is almost equal.
    Useful for comparing xyzs in unit tests.

    Args:
        xyz1 (Union[List[dict], dict]): Either a dict-format xyz, or a list of them.
        xyz2 (Union[List[dict], dict]): Either a dict-format xyz, or a list of them.
        rtol (float, optional): The relative tolerance parameter.
        atol (float, optional): The absolute tolerance parameter.

    Returns: bool
        Whether at least one entry in each input xyzs is almost equal to an entry in the other xyz.
    """
    if not isinstance(xyz1, list):
        xyz1 = [xyz1]
    if not isinstance(xyz2, list):
        xyz2 = [xyz2]
    for xyz1_entry in xyz1:
        for xyz2_entry in xyz2:
            if xyz1_entry['symbols'] != xyz2_entry['symbols']:
                continue
            if almost_equal_coords(xyz1_entry, xyz2_entry, rtol=rtol, atol=atol):
                return True
    return False


def is_notebook() -> bool:
    """
    Check whether ARC was called from an IPython notebook.

    Returns: bool
        ``True`` if ARC was called from a notebook, ``False`` otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole.
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython.
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter.


def is_str_float(value: Optional[str]) -> bool:
    """
    Check whether a string can be converted to a floating number.

    Args:
        value (str): The string to check.

    Returns: bool
        ``True`` if it can, ``False`` otherwise.
    """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def is_str_int(value: Optional[str]) -> bool:
    """
    Check whether a string can be converted to an integer.

    Args:
        value (str): The string to check.

    Returns: bool
        ``True`` if it can, ``False`` otherwise.
    """
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False


def time_lapse(t0) -> str:
    """
    A helper function returning the elapsed time since t0.

    Args:
        t0 (time.pyi): The initial time the count starts from.

    Returns: str
        A "D HH:MM:SS" formatted time difference between now and t0.
    """
    t = time.time() - t0
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        d = str(d) + ' days, '
    else:
        d = ''
    return f'{d}{h:02.0f}:{m:02.0f}:{s:02.0f}'


def estimate_orca_mem_cpu_requirement(num_heavy_atoms: int,
                                      server: str = '',
                                      consider_server_limits: bool = False,
                                      ) -> Tuple[int, float]:
    """
    Estimates memory and cpu requirements for an Orca job.

    Args:
        num_heavy_atoms (int): The number of heavy atoms in the species.
        server (str): The name of the server where Orca runs.
        consider_server_limits (bool):  Try to give realistic estimations.

    Returns: Tuple[int, float]:
        - The amount of total memory (MB)
        - The number of cpu cores required for the Orca job for a given species.
    """
    max_server_mem_gb = None
    max_server_cpus = None

    if consider_server_limits:
        if server in servers:
            max_server_mem_gb = servers[server].get('memory', None)
            max_server_cpus = servers[server].get('cpus', None)
        else:
            logger.debug(f'Cannot find server {server} in settings.py')
            consider_server_limits = False

        max_server_mem_gb = max_server_mem_gb if is_str_int(max_server_mem_gb) else None
        max_server_cpus = max_server_cpus if is_str_int(max_server_cpus) else None

    est_cpu = 2 + num_heavy_atoms * 4
    if consider_server_limits and max_server_cpus and est_cpu > max_server_cpus:
        est_cpu = max_server_cpus

    est_memory = 2000.0 * est_cpu
    if consider_server_limits and max_server_mem_gb and est_memory > max_server_mem_gb * 1024:
        est_memory = max_server_mem_gb * 1024

    return est_cpu, est_memory


def check_torsion_change(torsions: pd.DataFrame,
                         index_1: Union[int, str],
                         index_2: Union[int, str],
                         threshold: Union[float, int] = 20.0,
                         delta: Union[float, int] = 0.0,
                         ) -> pd.DataFrame:
    """
    Compare two sets of torsions (in DataFrame) and check if any entry has a
    difference larger than threshold. The output is a DataFrame consisting of
    ``True``/``False``, indicating which torsions changed significantly.

    Args:
        torsions (pd.DataFrame): A DataFrame consisting of multiple sets of torsions.
        index_1 (Union[int, str]): The index of the first conformer.
        index_2 (Union[int, str]): The index of the second conformer.
        threshold (Union[float, int]): The threshold used to determine the difference significance.
        delta (Union[float, int]): A known difference between torsion pairs,
                                   delta = tor[index_1] - tor[index_2].
                                   E.g.,for the torsions to be scanned, the
                                   differences are equal to the scan resolution.

    Returns: pd.DataFrame
        A DataFrame consisting of ``True``/``False``, indicating
        which torsions changed significantly. ``True`` for significant change.
    """
    # First iteration without 180/-180 adjustment
    change = (torsions[index_1] - torsions[index_2] - delta).abs() > threshold
    # Apply 180/-180 adjustment to those shown significance
    for label in change[change == True].index:
        # a -180 / 180 flip causes different sign
        if torsions.loc[label, index_1] * torsions.loc[label, index_2] < 0:
            if torsions.loc[label, index_1] < 0 \
              and abs(torsions.loc[label, index_1] + 360 - torsions.loc[label, index_2] - delta) < threshold:
                change[label] = False
            elif torsions.loc[label, index_2] < 0 \
                    and abs(torsions.loc[label, index_1] - 360 - torsions.loc[label, index_2] - delta) < threshold:
                change[label] = False
    return change


def is_same_pivot(torsion1: Union[list, str],
                  torsion2: Union[list, str],
                  ) -> Optional[bool]:
    """
    Check if two torsions have the same pivots.

    Args:
        torsion1 (Union[list, str]): The four atom indices representing the first torsion.
        torsion2 (Union: [list, str]): The four atom indices representing the second torsion.

    Returns: Optional[bool]
        ``True`` if two torsions share the same pivots.
    """
    torsion1 = ast.literal_eval(torsion1) if isinstance(torsion1, str) else torsion1
    torsion2 = ast.literal_eval(torsion2) if isinstance(torsion2, str) else torsion2
    if not (len(torsion1) == len(torsion2) == 4):
        return False
    if torsion1[1:3] == torsion2[1:3] or torsion1[1:3] == torsion2[1:3][::-1]:
        return True


def is_same_sequence_sublist(child_list: list, parent_list: list) -> bool:
    """
    Check if the parent list has a sublist which is identical to the child list including the sequence.
    Examples:
        - child_list = [1,2,3], parent_list=[5,1,2,3,9] -> ``True``
        - child_list = [1,2,3], parent_list=[5,6,1,3,9] -> ``False``

    Args:
        child_list (list): The child list (the pattern to search in the parent list).
        parent_list (list): The parent list.

    Returns: bool
        ``True`` if the sublist is in the parent list.
    """
    if len(parent_list) < len(child_list):
        return False
    if any([item not in parent_list for item in child_list]):
        return False
    for index in range(len(parent_list) - len(child_list) + 1):
        if child_list == parent_list[index:index + len(child_list)]:
            return True
    return False


def get_ordered_intersection_of_two_lists(l1: list,
                                          l2: list,
                                          order_by_first_list: Optional[bool] = True,
                                          return_unique: Optional[bool] = True,
                                          ) -> list:
    """
    Find the intersection of two lists by order.

    Examples:
        - l1 = [1, 2, 3, 3, 5, 6], l2 = [6, 3, 5, 5, 1], order_by_first_list = ``True``, return_unique = ``True``
          -> [1, 3, 5, 6] unique values in the intersection of l1 and l2, order following value's first appearance in l1

        - l1 = [1, 2, 3, 3, 5, 6], l2 = [6, 3, 5, 5, 1], order_by_first_list = ``True``, return_unique = ``False``
          -> [1, 3, 3, 5, 6] unique values in the intersection of l1 and l2, order following value's first appearance in l1

        - l1 = [1, 2, 3, 3, 5, 6], l2 = [6, 3, 5, 5, 1], order_by_first_list = ``False``, return_unique = ``True``
          -> [6, 3, 5, 1] unique values in the intersection of l1 and l2, order following value's first appearance in l2

        - l1 = [1, 2, 3, 3, 5, 6], l2 = [6, 3, 5, 5, 1], order_by_first_list = ``False``, return_unique = ``False``
          -> [6, 3, 5, 5, 1] unique values in the intersection of l1 and l2, order following value's first appearance in l2

    Args:
        l1 (list): The first list.
        l2 (list): The second list.
        order_by_first_list (bool, optional): Whether to order the output list using the order of the values in the first list.
        return_unique (bool, optional): Whether to return only unique values in the intersection of two lists.

    Returns: list
        An ordered list of the intersection of two input lists.
    """
    if order_by_first_list:
        l3 = [v for v in l1 if v in l2]
    else:
        l3 = [v for v in l2 if v in l1]

    lookup = set()
    if return_unique:
        l3 = [v for v in l3 if v not in lookup and lookup.add(v) is None]

    return l3


def get_angle_in_180_range(angle: float,
                           round_to: Optional[int] = 2,
                           ) -> float:
    """
    Get the corresponding angle in the -180 to +180 degree range.

    Args:
        angle (float): An angle in degrees.
        round_to (int, optional): The number of decimal figures to round the result to.
                                  ``None`` to not round. Default: 2.

    Returns:
        float: The corresponding angle in the -180 to +180 degree range.
    """
    angle = float(angle)
    while not (-180 <= angle < 180):
        factor = 360 if angle < -180 else -360
        angle += factor
    if round_to is not None:
        return round(angle, round_to)
    return angle


def get_close_tuple(key_1: Tuple[Union[float, str], ...],
                    keys: List[Tuple[Union[float, str], ...]],
                    tolerance: float = 0.05,
                    raise_error: bool = False,
                    ) -> Optional[Tuple[Union[float, str], ...]]:
    """
    Get a key from a list of keys close in value to the given key.
    Even if just one of the items in the key has a close match, use the close value.

    Args:
        key_1 (Tuple[Union[float, str], Union[float, str]]): The key used for the search.
        keys (List[Tuple[Union[float, str], Union[float, str]]]): The list of keys to search within.
        tolerance (float, optional): The tolerance within which keys are determined to be close.
        raise_error (bool, optional): Whether to raise a ValueError if a close key wasn't found.

    Raises:
        ValueError: If a key in ``keys`` has a different length than ``key_1``.
        ValueError: If a close key was not found and ``raise_error`` is ``True``.

    Returns:
        Optional[Tuple[Union[float, str], ...]]: A key from the keys list close in value to the given key.
    """
    key_1_floats = tuple(float(item) for item in key_1)
    for key_2 in keys:
        if len(key_1) != len(key_2):
            raise ValueError(f'Length of key_1, {key_1}, ({len(key_1)}) must be equal to the lengths of all keys '
                             f'(got a second key, {key_2}, with length {len(key_2)}).')
        key_2_floats = tuple(float(item) for item in key_2)
        if all(abs(item_1 - item_2) <= tolerance for item_1, item_2 in zip(key_1_floats, key_2_floats)):
            return key_2

    updated_key_1 = [None] * len(key_1)
    for key_2 in keys:
        key_2_floats = tuple(float(item) for item in key_2)
        for i, item in enumerate(key_2_floats):
            if abs(item - key_1_floats[i]) <= tolerance:
                updated_key_1[i] = key_2[i]

    if any(key is not None for key in updated_key_1):
        return tuple(updated_key_1[i] or key_1[i] for i in range(len(key_1)))
    elif not raise_error:
        # couldn't find a close key
        return None
    raise ValueError(f'Could not locate a key close to {key_1} within the tolerance {tolerance} in the given keys list.')


def timedelta_from_str(time_str: str):
    """
    Get a datetime.timedelta object from its str() representation

    Args:
        time_str (str): The string representation of a datetime.timedelta object.

    Returns:
        datetime.timedelta: The corresponding timedelta object.
    """
    regex = re.compile(r'((?P<hours>\d+?)hr)?((?P<minutes>\d+?)m)?((?P<seconds>\d+?)s)?')

    parts = regex.match(time_str)
    if not parts:
        return
    parts = parts.groupdict()
    time_params = {}
    for (name, param) in parts.items():
        if param:
            time_params[name] = int(param)
    return datetime.timedelta(**time_params)


def torsions_to_scans(descriptor: Optional[List[List[int]]],
                      direction: int = 1,
                      ) -> Optional[List[List[int]]]:
    """
    Convert torsions to scans or vice versa.
    In ARC we define a torsion as a list of four atoms with 0-indices.
    We define a scan  as a list of four atoms with 1-indices.
    This function converts one format to the other.

    Args:
        descriptor (list): The torsions or scans list.
        direction (int, optional): 1: Convert torsions to scans; -1: Convert scans to torsions.

    Returns:
        Optional[List[List[int]]]: The converted indices.
    """
    if descriptor is None:
        return None
    if not isinstance(descriptor, (list, tuple)):
        raise TypeError(f'Expected a list, got {descriptor} which is a {type(descriptor)}')
    if not isinstance(descriptor[0], (list, tuple)):
        descriptor = [descriptor]
    direction = direction if direction == 1 else -1  # Anything other than 1 is translated to -1.
    new_descriptor = [convert_list_index_0_to_1(entry, direction) for entry in descriptor]
    if any(any(item < 0 for item in entry) for entry in new_descriptor):
        raise ValueError(f'Got an illegal value when converting:\n{descriptor}\ninto:\n{new_descriptor}')
    return new_descriptor


def convert_list_index_0_to_1(_list: Union[list, tuple], direction: int = 1) -> Union[list, tuple]:
    """
    Convert a list from 0-indexed to 1-indexed, or vice versa.
    Ensures positive values in the resulting list.

    Args:
        _list (list): The list to be converted.
        direction (int, optional): Either 1 or -1 to convert 0-indexed to 1-indexed or vice versa, respectively.

    Raises:
        ValueError: If the new list contains negative values.

    Returns:
        Union[list, tuple]: The converted indices.
    """
    new_list = [item + direction for item in _list]
    if any(val < 0 for val in new_list):
        raise ValueError(f'The resulting list from converting {_list} has negative values:\n{new_list}')
    if isinstance(_list, tuple):
        new_list = tuple(new_list)
    return new_list


def rmg_mol_to_dict_repr(mol: Molecule,
                         reset_atom_ids: bool = False,
                         testing: bool = False,
                         ) -> dict:
    """
    Generate a dict representation of an RMG ``Molecule`` object instance.

    Args:
        mol (Molecule): The RMG ``Molecule`` object instance.
        reset_atom_ids (bool, optional): Whether to reset the atom IDs in the .mol Molecule attribute.
                                         Useful when copying the object to avoid duplicate atom IDs between
                                         different object instances.
        testing (bool, optional): Whether this is called during a test, in which case atom IDs should be deterministic.

    Returns:
        dict: The corresponding dict representation.
    """
    mol = mol.copy(deep=True)
    if testing:
        counter = 0
        for atom in mol.atoms:
            atom.id = counter
            counter += 1
    elif len(mol.atoms) > 1 and mol.atoms[0].id == mol.atoms[1].id or reset_atom_ids:
        mol.assign_atom_ids()
    return {'atoms': [{'element': {'number': atom.element.number,
                                   'isotope': atom.element.isotope,
                                   },
                       'radical_electrons': atom.radical_electrons,
                       'charge': atom.charge,
                       'label': atom.label,
                       'lone_pairs': atom.lone_pairs,
                       'id': atom.id,
                       'props': atom.props,
                       'atomtype': atom.atomtype.label,
                       'edges': {atom_2.id: bond.order
                                 for atom_2, bond in atom.edges.items()},
                       } for atom in mol.atoms],
            'multiplicity': mol.multiplicity,
            'props': mol.props,
            'atom_order': [atom.id for atom in mol.atoms]
            }


def rmg_mol_from_dict_repr(representation: dict,
                           is_ts: bool = False,
                           ) -> Optional[Molecule]:
    """
    Generate a dict representation of an RMG ``Molecule`` object instance.

    Args:
        representation (dict): A dict representation of an RMG ``Molecule`` object instance.
        is_ts (bool, optional): Whether the ``Molecule`` represents a TS.

    Returns:
        ``Molecule``: The corresponding RMG ``Molecule`` object instance.

    """
    mol = Molecule(multiplicity=representation['multiplicity'],
                   props=representation['props'])
    atoms = {atom_dict['id']: Atom(element=get_element(value=atom_dict['element']['number'],
                                                       isotope=atom_dict['element']['isotope']),
                                   radical_electrons=atom_dict['radical_electrons'],
                                   charge=atom_dict['charge'],
                                   lone_pairs=atom_dict['lone_pairs'],
                                   id=atom_dict['id'],
                                   props=atom_dict['props'],
                                   ) for atom_dict in representation['atoms']}
    for atom_dict in representation['atoms']:
        atoms[atom_dict['id']].atomtype = ATOMTYPES[atom_dict['atomtype']]
    mol.atoms = list(atoms[atom_id] for atom_id in representation['atom_order'])
    for i, atom_1 in enumerate(atoms.values()):
        for atom_2_id, bond_order in representation['atoms'][i]['edges'].items():
            bond = Bond(atom_1, atoms[atom_2_id], bond_order)
            mol.add_bond(bond)
    mol.update_atomtypes(raise_exception=False)
    mol.update_multiplicity()
    if not is_ts:
        mol.identify_ring_membership()
        mol.update_connectivity_values()
    return mol


def calc_rmsd(x: Union[list, np.array],
              y: Union[list, np.array],
              ) -> float:
    """
    Compute the root-mean-square deviation between two matrices.

    Args:
        x (np.array): Matrix 1.
        y (np.array): Matrix 2.

    Returns:
        float: The RMSD score of two matrices.
    """
    x = np.array(x) if isinstance(x, list) else x
    y = np.array(y) if isinstance(y, list) else y
    d = x - y
    n = x.shape[0]
    sqr_sum = (d ** 2).sum()
    rmsd = np.sqrt(sqr_sum / n)
    return float(rmsd)


def _check_r_n_p_symbols_between_rmg_and_arc_rxns(arc_reaction: 'ARCReaction',
                                                  rmg_reactions: List['Reaction'],
                                                  ) -> bool:
    """
    A helper function to check that atom symbols are in the correct order between an ARC reaction
    and its corresponding RMG reactions generated by the get_rmg_reactions_from_arc_reaction() function.
    Used internally for testing.

    Args:
        arc_reaction (ARCReaction): The ARCReaction object to inspect.
        rmg_reactions (List['Reaction']): Entries are RMG Reaction objects to inspect.
                                          Could contain either Species or Molecule object as reactants/products.

    Returns:
        bool: Whether atom symbols are in the same respective order.
    """
    result = True
    num_rs, num_ps = len(arc_reaction.r_species), len(arc_reaction.p_species)
    arc_r_symbols = [atom.element.symbol for atom in chain(*tuple(arc_reaction.r_species[i].mol.atoms for i in range(num_rs)))]
    arc_p_symbols = [atom.element.symbol for atom in chain(*tuple(arc_reaction.p_species[i].mol.atoms for i in range(num_ps)))]
    for rmg_reaction in rmg_reactions:
        rmg_r_symbols = [atom.element.symbol
                         for atom in chain(*tuple(rmg_reaction.reactants[i].atoms
                                                  if isinstance(rmg_reaction.reactants[i], Molecule)
                                                  else rmg_reaction.reactants[i].molecule[0].atoms
                                                  for i in range(num_rs)))]
        rmg_p_symbols = [atom.element.symbol
                         for atom in chain(*tuple(rmg_reaction.products[i].atoms
                                                  if isinstance(rmg_reaction.products[i], Molecule)
                                                  else rmg_reaction.products[i].molecule[0].atoms
                                                  for i in range(num_ps)))]
        if any(symbol_1 != symbol_2 for symbol_1, symbol_2 in zip(arc_r_symbols, rmg_r_symbols)):
            print('\nDifferent element order in reactants between ARC and RMG:')  # Don't modify to logging.
            print(arc_r_symbols)
            print(rmg_r_symbols)
            result = False
        if any(symbol_1 != symbol_2 for symbol_1, symbol_2 in zip(arc_p_symbols, rmg_p_symbols)):
            print('\nDifferent element order in products between ARC and RMG:')
            print(arc_p_symbols)
            print(rmg_p_symbols)
            result = False
    return result
