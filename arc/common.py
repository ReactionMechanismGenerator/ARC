#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains functions which are shared across multiple ARC modules.
As such, it should not import any other ARC module (specifically ones that use the logger defined here)
  to avoid circular imports.

VERSION is the full ARC version, using `semantic versioning <https://semver.org/>`_.

ATOM_RADII data taken from `DOI 10.1039/b801115j <http://dx.doi.org/10.1039/b801115j>`_.
"""

import datetime
import logging
import os
import shutil
import subprocess
import sys
import time
import warnings
import yaml

import numpy as np

from arkane.ess import GaussianLog, MolproLog, QChemLog, OrcaLog
from arkane.util import determine_qm_software
from rmgpy.molecule.element import get_element
from rmgpy.qm.qmdata import QMData
from rmgpy.qm.symmetry import PointGroupCalculator

from arc.exceptions import InputError, SettingsError
from arc.settings import arc_path, servers, default_job_types


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
        if isinstance(server_list, str):
            settings[software] = [server_list]
        elif isinstance(server_list, list):
            for server in server_list:
                if not isinstance(server, str):
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
            if not isinstance(server, bool) and server.lower() not in list(servers.keys()):
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
        logger.log(level, f'    {head}\n    {date}')
    if branch_name and branch_name != 'master':
        logger.log(level, f'    (running on the {branch_name} branch)\n')
    else:
        logger.log(level, '\n')
    logger.info('Starting project {0}'.format(project))


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


def get_git_commit():
    """
    Get the recent git commit to be logged.

    Note:
        Returns empty strings if hash and date cannot be determined.

    Returns:
        tuple: The git HEAD commit hash and the git HEAD commit date, each as a string.
    """
    head, date = '', ''
    if os.path.exists(os.path.join(arc_path, '.git')):
        try:
            head, date = subprocess.check_output(['git', 'log', '--format=%H%n%cd', '-1'], cwd=arc_path).splitlines()
            head, date = head.decode(), date.decode()
        except (subprocess.CalledProcessError, OSError):
            return head, date
    return head, date


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
            if '*' in branch_name.decode():
                return branch_name.decode()[2:]
    else:
        return ''


def read_yaml_file(path):
    """
    Read a YAML file (usually an input / restart file, but also conformers file)
    and return the parameters as python variables.

    Args:
        path (str): The YAML file path to read.

    Returns:
        dict or list: The content read from the file.
    """
    if not isinstance(path, str):
        raise InputError(f'path must be a string, got {path} which is a {type(path)}')
    if not os.path.isfile(path):
        raise InputError(f'Could not find the YAML file {path}')
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
    if not isinstance(path, str):
        raise InputError(f'path must be a string, got {path} which is a {type(path)}')
    yaml.add_representer(str, string_representer)
    logger.debug('Creating a restart file...')
    content = yaml.dump(data=content)
    if os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
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
    if number in list(ordinal_dict.keys()):
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
    if not isinstance(symbol, str):
        raise InputError('the symbol argument must be string, got {0} which is a {1}'.format(symbol, type(symbol)))

    if symbol in ATOM_RADII:
        return ATOM_RADII[symbol]
    else:
        return None


def determine_symmetry(xyz):
    """
    Determine external symmetry and chirality (optical isomers) of the species.

    Args:
        xyz (dict): The 3D coordinates.

    Returns:
        int: The external symmetry number.
    Returns:
        int: 1 if no chiral centers are present, 2 if chiral centers are present.
    """
    atom_numbers = list()  # List of atomic numbers
    for symbol in xyz['symbols']:
        atom_numbers.append(get_element(symbol).number)
    # coords is an N x 3 numpy.ndarray of atomic coordinates in the same order as `atom_numbers`
    coords = np.array(xyz['coords'], np.float64)
    unique_id = '0'  # Just some name that the SYMMETRY code gives to one of its jobs
    scr_dir = os.path.join(arc_path, 'scratch')  # Scratch directory that the SYMMETRY code writes its files in
    if not os.path.exists(scr_dir):
        os.makedirs(scr_dir)
    symmetry = optical_isomers = 1
    qmdata = QMData(
        groundStateDegeneracy=1,  # Only needed to check if valid QMData
        numberOfAtoms=len(atom_numbers),
        atomicNumbers=atom_numbers,
        atomCoords=(coords, 'angstrom'),
        energy=(0.0, 'kcal/mol')  # Only needed to avoid error
    )
    settings = type('', (), dict(symmetryPath='symmetry', scratchDirectory=scr_dir))()
    pgc = PointGroupCalculator(settings, unique_id, qmdata)
    pg = pgc.calculate()
    if pg is not None:
        symmetry = pg.symmetry_number
        optical_isomers = 2 if pg.chiral else optical_isomers
    return symmetry, optical_isomers


def min_list(lst):
    """
    A helper function for finding the minimum of a list of integers where some of the entries might be None.

    Args:
        lst (list): The list.

    Returns:
        int: The entry with the minimal value.
    """
    if len(lst) == 0:
        return None
    elif len(lst) == 1:
        return lst[0]
    elif all([entry is None for entry in lst]):
        return None
    return min([entry for entry in lst if entry is not None])


def key_by_val(dictionary, value):
    """
    A helper function for getting a key from a dictionary corresponding to a certain value.
    Does not check for value unicity.

    Args:
        dictionary (dict): The dictionary.
        value: The value.

    Returns:
        The key.
    """
    for key, val in dictionary.items():
        if val == value:
            return key
    raise ValueError(f'Could not find value {value} in the dictionary\n{dictionary}')


def initialize_job_types(job_types, specific_job_type=''):
    """
    A helper function for initializing job_types.
    Returns the comprehensive (default values for missing job types) job types for ARC.

    Args:
        job_types (dict): Keys are job types, values are booleans of whether or not to consider this job type.
        specific_job_type (str): Specific job type to execute. Legal strings are job types (keys of job_types dict).

    Returns:
        job_types (dict): An updated (comprehensive) job type dictionary.
    """

    if specific_job_type:
        logger.info(f'Specific_job_type {specific_job_type} is given by user.')
        if job_types:
            logger.warning('Both job_types and specific_job_type are given, ARC will only use specific_job_type to '
                           'populate the job_types dictionary.')
        job_types = {job_type: False for job_type in default_job_types.keys()}
        try:
            job_types[specific_job_type] = True
        except KeyError:
            raise InputError(f'Specified job type {specific_job_type} is not supported.')

    if specific_job_type == 'bde':
        bde_default = {'opt': True, 'fine_grid': True, 'freq': True, 'sp': True}
        job_types.update(bde_default)

    defaults_to_true = ['conformers', 'opt', 'fine', 'freq', 'sp', 'rotors']
    defaults_to_false = ['onedmin', 'orbitals', 'bde']
    if job_types is None:
        job_types = default_job_types
        logger.info('Job types were not specified, using default values from ARC settings')
    else:
        logger.info(f'the following job types were specified: {job_types}.')
    if 'lennard_jones' in job_types:
        # rename lennard_jones to OneDMin
        job_types['onedmin'] = job_types['lennard_jones']
        del job_types['lennard_jones']
    if 'fine_grid' in job_types:
        # rename fine_grid to fine
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


def determine_ess(log_file):
    """
    Determine the ESS to which the log file belongs.

    Args:
        log_file (str): The ESS log file path.

    Returns:
        str: The ESS (either 'gaussian', 'qchem', or 'molpro').

    Raises:
        InputError: If the log file could not be identified.
    """
    log = determine_qm_software(log_file)
    if isinstance(log, GaussianLog):
        return 'gaussian'
    if isinstance(log, QChemLog):
        return 'qchem'
    if isinstance(log, MolproLog):
        return 'molpro'
    if isinstance(log, OrcaLog):
        return 'orca'
    raise InputError(f'Could not identify the log file in {log_file} as belonging to Gaussian, QChem, Molpro, or Orca.')


def calculate_dihedral_angle(coords, torsion):
    """
    Calculate a dihedral angle. Inspired by ASE Atoms.get_dihedral().

    Args:
        coords (list, tuple): The array-format or tuple-format coordinates.
        torsion (list): The 4 atoms defining the dihedral angle, 1-indexed.

    Returns:
        float: The dihedral angle.
    """
    torsion = [t - 1 for t in torsion]  # convert 1-index to 0-index
    coords = np.asarray(coords, dtype=np.float32)
    a = coords[torsion[1]] - coords[torsion[0]]
    b = coords[torsion[2]] - coords[torsion[1]]
    c = coords[torsion[3]] - coords[torsion[2]]
    bxa = np.cross(b, a)
    bxa /= np.linalg.norm(bxa)
    cxb = np.cross(c, b)
    cxb /= np.linalg.norm(cxb)
    angle = np.vdot(bxa, cxb)
    # check for numerical trouble due to finite precision:
    if angle < -1:
        angle = -1
    elif angle > 1:
        angle = 1
    angle = np.arccos(angle)
    if np.vdot(bxa, c) > 0:
        angle = 2 * np.pi - angle
    return angle * 180 / np.pi


def almost_equal_coords(xyz1, xyz2):
    """
    A helper function for checking whether two xyz's are almost equal.

    Args:
        xyz1 (dict): Coordinates.
        xyz2 (dict): Coordinates.
    """
    for xyz_coord1, xyz_coord2 in zip(xyz1['coords'], xyz2['coords']):
        for xyz1_c, xyz2_c in zip(xyz_coord1, xyz_coord2):
            if not np.isclose([xyz1_c], [xyz2_c]):
                return False
    return True


def almost_equal_coords_lists(xyz1, xyz2):
    """
    A helper function for checking two lists of xyz's has at least one entry in each that is almost equal.
    Useful for comparing xyz's in unit tests.

    Args:
        xyz1 (list, dict): Either a dict-format xyz, or a list of them.
        xyz2 (list, dict): Either a dict-format xyz, or a list of them.

    Returns:
        bool: Whether at least one entry in each input xyz's is almost equal to an entry in the other xyz.
    """
    if not isinstance(xyz1, list):
        xyz1 = [xyz1]
    if not isinstance(xyz2, list):
        xyz2 = [xyz2]
    for xyz1_entry in xyz1:
        for xyz2_entry in xyz2:
            if xyz1_entry['symbols'] != xyz2_entry['symbols']:
                return False
            if almost_equal_coords(xyz1_entry, xyz2_entry):
                break
        else:
            return False
    return True


def sort_two_lists_by_the_first(list1, list2):
    """
    Sort two lists in increasing order by the values of the first list.
    Ignoring None entries from list1 and their respective entries in list2.
    The function was written in this format rather the more pytonic ``zip(*sorted(zip(list1, list2)))`` style
    to accommodate for dictionaries as entries of list2, otherwise a
    ``TypeError: '<' not supported between instances of 'dict' and 'dict'`` error is raised.

    Args:
        list1 (list, tuple): Entries are floats or ints (could also be None).
        list2 (list, tuple): Entries could be anything.

    Returns:
        list: Sorted values from list1, ignoring None entries.
    Returns:
        list: Respective entries from list2.

    Raises:
        InputError: If types are wrong, or lists are not the same length.
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


def determine_model_chemistry_type(method):
    """
    Determine the type of a model chemistry (e.g., DFT, wavefunction, force field, semi-empirical, composite).

    Args:
        method (str): method in a model chemistry. e.g., b3lyp, cbs-qb3, am1, dlpno-ccsd(T)

    Returns:
        model_chemistry_type (str): class of model chemistry.
    """
    given_method = method.lower()
    wave_function_methods = ['hf', 'cc', 'ci', 'mp2', 'mp3', 'cp', 'cep', 'nevpt', 'dmrg', 'ri', 'cas', 'ic', 'mr',
                             'bd', 'mbpt']
    semiempirical_methods = ['am', 'pm', 'zindo', 'mndo', 'xtb', 'nddo']
    force_field_methods = ['amber', 'mmff', 'dreiding', 'uff', 'qmdff', 'gfn', 'gaff', 'ghemical', 'charmm', 'ani']
    # all composite methods supported by Gaussian
    composite_methods = ['cbs-4m', 'cbs-qb3', 'rocbs-qb3', 'cbs-apno', 'w1u', 'w1ro', 'w1bd', 'g1', 'g2', 'g3', 'g4',
                         'g2mp2', 'g3mp2', 'g3b3', 'g3mp2b3', 'g4mp2', 'cbs-qb3-paraskevas']

    # Composite methods
    if given_method in composite_methods:
        model_chemistry_class = 'composite'
        return model_chemistry_class

    # Special cases
    if given_method in ['m06hf', 'm06-hf']:
        model_chemistry_class = 'dft'
        return model_chemistry_class

    # General cases
    if any(wf_method in given_method for wf_method in wave_function_methods):
        model_chemistry_class = 'wavefunction'
    elif any(sm_method in given_method for sm_method in semiempirical_methods):
        model_chemistry_class = 'semiempirical'
    elif any(ff_method in given_method for ff_method in force_field_methods):
        model_chemistry_class = 'force_field'   # a.k.a molecular dynamics
    else:
        logger.debug(f'Assuming {given_method} is a DFT method.')
        model_chemistry_class = 'dft'
    return model_chemistry_class


def format_level_of_theory_for_logging(level_of_theory_dict):
    """
    Format level of theory dictionary to string for logging purposes.

    Args:
        level_of_theory_dict (dict): level of theory dictionary.

    Returns:
        level_of_theory_log_str (str): level of theory string for logging.
    """
    method = level_of_theory_dict.get('method', '')
    basis = level_of_theory_dict.get('basis', '')
    auxiliary_basis = level_of_theory_dict.get('auxiliary_basis', '')
    dispersion = level_of_theory_dict.get('dispersion', '')
    level_of_theory_log_str = '/'.join([method, basis]) if basis else method
    level_of_theory_log_str = '/'.join([level_of_theory_log_str, auxiliary_basis]) if auxiliary_basis\
        else level_of_theory_log_str
    level_of_theory_log_str = ' '.join([level_of_theory_log_str, dispersion]) if dispersion\
        else level_of_theory_log_str
    return level_of_theory_log_str


def is_notebook():
    """
    Check whether ARC was called from an IPython notebook.

    Returns:
        bool: ``True`` if ARC was called from a notebook, ``False`` otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
