"""
This module contains functions which are shared across multiple Job modules.
As such, it should not import any other ARC modules to avoid circular imports.
"""

import datetime
import os
import shutil
import sys
import re

from pprint import pformat
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from arc.common import get_logger
from arc.imports import settings
from arc.level import Level

if TYPE_CHECKING:
    from arc.job.adapter import JobAdapter
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies

logger = get_logger()

default_job_settings, global_ess_settings, rotor_scan_resolution = \
    settings['default_job_settings'], settings['global_ess_settings'], settings['rotor_scan_resolution']


ts_adapters_by_rmg_family = {'1+2_Cycloaddition': ['kinbot'],
                             '1,2_Insertion_CO': ['kinbot'],
                             '1,2_Insertion_carbene': ['kinbot'],
                             '1,2_shiftC': ['gcn', 'xtb_gsm'],
                             '1,2_shiftS': ['gcn', 'kinbot', 'xtb_gsm'],
                             '1,3_Insertion_CO2': ['kinbot'],
                             '1,3_Insertion_ROR': ['kinbot'],
                             '1,3_Insertion_RSR': ['kinbot'],
                             '1,4_Cyclic_birad_scission': ['gcn', 'xtb_gsm'],
                             '2+2_cycloaddition': ['kinbot'],
                             '6_membered_central_C-C_shift': ['gcn', 'xtb_gsm'],
                             'Concerted_Intra_Diels_alder_monocyclic_1,2_shiftH': ['gcn', 'xtb_gsm'],
                             'Cyclic_Ether_Formation': ['kinbot'],
                             'Cyclopentadiene_scission': ['gcn', 'xtb_gsm'],
                             'Diels_alder_addition': ['kinbot'],
                             'H_Abstraction': ['heuristics', 'autotst'],
                             'HO2_Elimination_from_PeroxyRadical': ['kinbot'],
                             'Intra_2+2_cycloaddition_Cd': ['gcn', 'xtb_gsm'],
                             'Intra_5_membered_conjugated_C=C_C=C_addition': ['gcn', 'xtb_gsm'],
                             'Intra_Diels_alder_monocyclic': ['gcn', 'kinbot', 'xtb_gsm'],
                             'Intra_Disproportionation': ['gcn', 'xtb_gsm'],
                             'Intra_ene_reaction': ['gcn', 'kinbot', 'xtb_gsm'],
                             'intra_H_migration': ['autotst', 'gcn', 'kinbot', 'xtb_gsm'],
                             'intra_NO2_ONO_conversion': ['gcn', 'xtb_gsm'],
                             'intra_OH_migration': ['gcn', 'kinbot', 'xtb_gsm'],
                             'Intra_RH_Add_Endocyclic': ['gcn', 'kinbot', 'xtb_gsm'],
                             'Intra_RH_Add_Exocyclic': ['gcn', 'kinbot', 'xtb_gsm'],
                             'Intra_R_Add_Endocyclic': ['gcn', 'kinbot', 'xtb_gsm'],
                             'Intra_R_Add_Exo_scission': ['gcn', 'xtb_gsm'],
                             'Intra_R_Add_Exocyclic': ['gcn', 'kinbot', 'xtb_gsm'],
                             'Intra_R_Add_ExoTetCyclic': ['kinbot'],
                             'Intra_Retro_Diels_alder_bicyclic': ['kinbot'],
                             'intra_substitutionCS_isomerization': ['gcn', 'xtb_gsm'],
                             'intra_substitutionS_isomerization': ['gcn', 'xtb_gsm'],
                             'Ketoenol': ['gcn', 'kinbot', 'xtb_gsm'],
                             'Korcek_step1': ['gcn', 'xtb_gsm'],
                             'Korcek_step2': ['kinbot'],
                             'R_Addition_COm': ['kinbot'],
                             'R_Addition_CSm': ['kinbot'],
                             'R_Addition_MultipleBond': ['autotst', 'kinbot'],
                             'Retroene': ['kinbot'],
                             'Singlet_Carbene_Intra_Disproportionation': ['gcn', 'xtb_gsm'],
                             }

all_families_ts_adapters = []
adapters_that_do_not_require_a_level_arg = ['xtb', 'torchani']

# Default is "queue", "pipe" will be called whenever needed. So just list 'incore'.
default_incore_adapters = ['autotst', 'gcn', 'heuristics', 'kinbot', 'psi4', 'xtb', 'xtb_gsm', 'torchani', 'openbabel']


def _initialize_adapter(obj: 'JobAdapter',
                        is_ts: bool,
                        project: str,
                        project_directory: str,
                        job_type: Union[List[str], str],
                        args: Optional[dict] = None,
                        bath_gas: Optional[str] = None,
                        checkfile: Optional[str] = None,
                        conformer: Optional[int] = None,
                        constraints: Optional[List[Tuple[List[int], float]]] = None,
                        cpu_cores: Optional[str] = None,
                        dihedral_increment: Optional[float] = None,
                        dihedrals: Optional[List[float]] = None,
                        directed_scan_type: Optional[str] = None,
                        ess_settings: Optional[dict] = None,
                        ess_trsh_methods: Optional[List[str]] = None,
                        fine: bool = False,
                        initial_time: Optional[Union['datetime.datetime', str]] = None,
                        irc_direction: Optional[str] = None,
                        job_id: Optional[int] = None,
                        job_memory_gb: float = 14.0,
                        job_name: Optional[str] = None,
                        job_num: Optional[int] = None,
                        job_server_name: Optional[str] = None,
                        job_status: Optional[List[Union[dict, str]]] = None,
                        level: Optional[Level] = None,
                        max_job_time: Optional[float] = None,
                        reactions: Optional[List['ARCReaction']] = None,
                        rotor_index: Optional[int] = None,
                        server: Optional[str] = None,
                        server_nodes: Optional[list] = None,
                        species: Optional[List['ARCSpecies']] = None,
                        testing: bool = False,
                        times_rerun: int = 0,
                        torsions: Optional[List[List[int]]] = None,
                        tsg: Optional[int] = None,
                        xyz: Optional[dict] = None,
                        ):
    """
    A common Job adapter initializer function.
    """
    if not is_ts and any(arg is None for arg in [job_type, project, project_directory]):
        raise ValueError(f'All of the following arguments must be given:\n'
                         f'job_type, level, project, project_directory\n'
                         f'Got: {job_type}, {level}, {project}, {project_directory}, respectively')
    if not is_ts and obj.job_adapter not in adapters_that_do_not_require_a_level_arg and level is None:
        raise ValueError(f'A `level` argument must be given')

    obj.project = project
    obj.project_directory = project_directory
    if obj.project_directory and not os.path.isdir(obj.project_directory):
        os.makedirs(obj.project_directory)

    obj.additional_job_info = None
    obj.args = args or dict()
    obj.bath_gas = bath_gas
    obj.checkfile = checkfile
    obj.conformer = conformer
    obj.constraints = constraints or list()
    obj.cpu_cores = cpu_cores
    obj.dihedral_increment = dihedral_increment
    obj.dihedrals = dihedrals
    obj.directed_scan_type = directed_scan_type
    obj.ess_settings = ess_settings or global_ess_settings
    obj.ess_trsh_methods = ess_trsh_methods or list()
    obj.files_to_download = list()
    obj.files_to_upload = list()
    obj.final_time = None
    obj.fine = fine
    obj.initial_time = datetime.datetime.strptime(initial_time.split('.')[0], '%Y-%m-%d %H:%M:%S') \
        if isinstance(initial_time, str) else initial_time
    obj.input_file_memory = None
    obj.irc_direction = irc_direction
    obj.iterate_by = list()
    obj.job_id = job_id
    obj.job_memory_gb = job_memory_gb
    obj.job_name = job_name
    obj.job_num = job_num
    obj.job_server_name = job_server_name
    obj.job_status = job_status \
        or ['initializing', {'status': 'initializing', 'keywords': list(), 'error': '', 'line': ''}]
    obj.job_type = job_type if isinstance(job_type, str) else job_type[0]  # always a string
    obj.job_types = job_type if isinstance(job_type, list) else [job_type]  # always a list
    # When restarting ARC and re-setting the jobs, ``level`` is a string, convert it to a Level object instance
    obj.level = Level(repr=level) if not isinstance(level, Level) and level is not None else level
    obj.max_job_time = max_job_time or default_job_settings.get('job_time_limit_hrs', 120)
    obj.number_of_processes = 0
    obj.reactions = [reactions] if reactions is not None and not isinstance(reactions, list) else reactions
    obj.remote_path = None
    obj.restarted = bool(job_num)  # If job_num was given, this is a restarted job, don't save as initiated jobs.
    obj.rotor_index = rotor_index
    obj.run_time = None
    obj.server = server
    obj.server_nodes = server_nodes or list()
    obj.species = [species] if species is not None and not isinstance(species, list) else species
    obj.submit_script_memory = None
    obj.testing = testing
    obj.times_rerun = times_rerun
    obj.torsions = [torsions] if torsions is not None and not isinstance(torsions[0], list) else torsions
    obj.pivots = [[tor[1] + 1, tor[2] + 1] for tor in obj.torsions] if obj.torsions is not None else None
    obj.tsg = tsg
    obj.workers = None
    obj.xyz = obj.species[0].get_xyz() if obj.species is not None and xyz is None else xyz

    if obj.job_num is None or obj.job_name is None or obj.job_server_name:
        obj._set_job_number()

    if obj.species is not None:
        obj.charge = obj.species[0].charge
        obj.multiplicity = obj.species[0].multiplicity
        obj.is_ts = obj.species[0].is_ts
        obj.species_label = obj.species[0].label
        if len(obj.species) > 1:
            obj.species_label += f'_and_{len(obj.species) - 1}_others'
    elif obj.reactions is not None:
        obj.charge = obj.reactions[0].charge
        obj.multiplicity = obj.reactions[0].multiplicity
        obj.is_ts = True
        obj.species_label = obj.reactions[0].ts_species.label if obj.reactions[0].ts_species is not None \
            else f'TS_{obj.job_num}'
        if len(obj.reactions) > 1:
            obj.species_label += f'_and_{len(obj.reactions) - 1}_others'
    else:
        obj.charge = None
        obj.multiplicity = None
        obj.is_ts = None
        obj.species_label = None

    obj.args = set_job_args(args=obj.args, level=obj.level, job_name=obj.job_name)
    if obj.execution_type != 'incore' and obj.job_adapter in obj.ess_settings.keys():
        if 'server' in obj.args['trsh']:
            obj.server = obj.args['trsh']['server']
        elif obj.job_adapter in obj.ess_settings.keys():
            if isinstance(obj.ess_settings[obj.job_adapter], list):
                obj.server = obj.ess_settings[obj.job_adapter][0]
            else:
                obj.server = obj.ess_settings[obj.job_adapter]

    obj.set_file_paths()
    obj.set_cpu_and_mem()
    if obj.execution_type != 'incore' and obj.job_adapter in obj.ess_settings.keys():
        obj.determine_job_array_parameters()

    # Set scan_res if required by trsh
    if obj.args and 'trsh' in obj.args.keys() and 'scan_res' in obj.args['trsh'].keys():
        obj.scan_res = obj.args['trsh']['scan_res']
        # Remove it from the args dict
        obj.args['trsh'].pop('scan_res')
    else:
        obj.scan_res = rotor_scan_resolution

    obj.set_files()
    check_argument_consistency(obj)


def is_restricted(obj) -> bool:
    """
    Check whether a Job Adapter should be executed as restricted or unrestricted.

    Args:
        obj: The job adapter object.

    Returns:
        bool: Whether to run as restricted (``True``) or not (``False``).
    """
    if (obj.multiplicity > 1 and obj.level.method_type != 'composite') \
            or (obj.species[0].number_of_radicals is not None and obj.species[0].number_of_radicals > 1):
        # run an unrestricted electronic structure calculation if the spin multiplicity is greater than one,
        # or if it is one but the number of radicals is greater than one (e.g., bi-rad singlet)
        # don't run unrestricted for composite methods such as CBS-QB3, it'll be done automatically if the
        # multiplicity is greater than one, but do specify uCBS-QB3 for example for bi-rad singlets.
        if obj.species[0].number_of_radicals is not None and obj.species[0].number_of_radicals > 1:
            logger.info(f'Using an unrestricted method for species {obj.species_label} which has '
                        f'{obj.species[0].number_of_radicals} radicals and multiplicity {obj.multiplicity}.')
        return False
    return True


def check_argument_consistency(obj: 'JobAdapter'):
    """
    Check that general arguments of a job adapter are consistent.

    Args:
        obj (JobAdapter): The specific (not abstract) job adapter object instance.
    """
    if obj.job_type == 'irc' and obj.job_adapter in ['molpro']:
        raise NotImplementedError(f'IRC is not implemented for the {obj.job_adapter} job adapter.')
    if obj.job_type == 'irc' and (obj.irc_direction is None or obj.irc_direction not in ['forward', 'reverse']):
        raise ValueError(f'Missing the irc_direction argument for job type irc. '
                         f'It must be either "forward" or "reverse".\nGot: {obj.irc_direction}')
    if obj.job_type == 'scan' and obj.job_adapter in ['molpro']:
        for species in obj.species:
            if any(rotor_dict['directed_scan_type'] == 'ess' for rotor_dict in species.rotors_dict.values()):
                raise NotImplementedError(f'The {obj.job_adapter} job adapter does not support ESS scans.')
    if obj.job_type == 'scan' and divmod(360, obj.scan_res)[1]:
        raise ValueError(f'Got an illegal rotor scan resolution of {obj.scan_res}.')
    if obj.job_type == 'scan' and ((not obj.species[0].rotors_dict or obj.rotor_index is None) and obj.torsions is None):
        # If this is a scan job type and species.rotors_dict is empty (e.g., via pipe), then torsions must be set up.
        raise ValueError('Either torsions or a species rotors_dict along with a rotor_index argument '
                         'must be specified for an ESS scan job.')


def update_input_dict_with_args(args: dict,
                                input_dict: dict,
                                ) -> dict:
    """
    Update the job input_dict attribute with keywords and blocks from the args attribute.

    The function iterates over each key in the 'args' dictionary. Depending on the key type 
    ('block', 'keyword', 'trsh'), it updates the 'input_dict' accordingly. For 'block' and 
    'keyword' types, it appends the corresponding values to the 'input_dict'. For 'trsh', 
    it handles both list and non-list values.
    
    Args:
        args (dict): The job arguments with the following structure:
            args = {
                'block': {key1: value1, key2: value2, ...},
                'keyword': {key1: value1, key2: value2, ...},
                'trsh': {key1: value1, key2: value2, ... or key: [value1, value2, ...]},
                ...
            }

        input_dict (dict): The job input dict with a structure that may be updated:
            input_dict = {
                'block': <block string>,
                'keywords': <keywords string>,
                'scan_trsh': <scan_trsh string>,
                'trsh': <trsh string>,
                ...
            }

    Returns:
        dict: The updated input_dict with appended or modified values based on the args.

    Example:
        Input args structure:
            args = {
                'block': {'1': 'block text 1', '2': 'block text 2'},
                'keyword': {'opt': 'keyword opt', 'freq': 'keyword freq'},
                'trsh': ['trsh value 1', 'trsh value 2']
            }

        Input input_dict structure:
            input_dict = {
                'block': 'existing block text',
                'keywords': 'existing keywords',
                'scan_trsh': 'existing scan_trsh',
                'trsh': 'existing trsh'
            }

        Output input_dict structure:
            input_dict = {
                'block': 'existing block text\n\nblock text 1\nblock text 2\n',
                'keywords': 'existing keywords keyword opt keyword freq ',
                'scan_trsh': 'existing scan_trsh',
                'trsh': 'existing trsh trsh value 1 trsh value 2 '
            }

    """
    for arg_type, arg_dict in args.items():
        if arg_type == 'block' and arg_dict:
            input_dict['block'] = '\n\n' if not input_dict['block'] else input_dict['block']
            for block in arg_dict.values():
                # Chek if input_dict['block'] already contains a value and that it ends with a newline
                if input_dict['block'] and not input_dict['block'].endswith('\n'):
                    input_dict['block'] += '\n'
                input_dict['block'] += f'{block}'
        elif arg_type == 'keyword' and arg_dict:
            for key, value in arg_dict.items():
                if key == 'scan_trsh':
                    if 'scan_trsh' not in input_dict.keys():
                        input_dict['scan_trsh'] = ''
                    # Check if input_dict['scan_trsh'] already contains a value
                    if input_dict['scan_trsh']:
                        input_dict['scan_trsh'] += f' {value}'
                    else:
                        input_dict['scan_trsh'] += f'{value}'
                else:
                    if 'keywords' not in input_dict.keys():
                        input_dict['keywords'] = ''
                # Check if input_dict['keywords'] already contains a value
                    if input_dict['keywords']:
                        input_dict['keywords'] += f' {value}'
                    else:
                        input_dict['keywords'] += f'{value}'
        elif arg_type == 'trsh':
            if isinstance(arg_dict, list):
                # arg_dict is a list, iterate through its elements
                for val in arg_dict:
                    if 'trsh' not in input_dict.keys():
                        input_dict['trsh'] = ''
                    # Check if input_dict['trsh'] already contains a value
                    if input_dict['trsh']:
                        input_dict['trsh'] += f' {val}'
                    else:
                        input_dict['trsh'] += f'{val}'
            else:
                # arg_dict is a dictionary, proceed as before
                for key, value in arg_dict.items():
                    # Check if value is a list
                    if isinstance(value, list):
                        for val in value:
                            if key not in input_dict.keys():
                                input_dict[key] = ''
                            if input_dict[key]:
                                input_dict[key] += f' {val}'
                            else:
                                input_dict[key] += f'{val}'
                    else:
                        if key not in input_dict.keys():
                            input_dict[key] = ''
                        if input_dict[key]:
                            input_dict[key] += f' {value}'
                        else:
                            input_dict[key] += f'{value}'

    return input_dict

def input_dict_strip(input_dict: dict) -> dict:
    """
    Strip all values in the input dict of leading and trailing whitespace.
    """
    stripped_dict = {
        key: (
            val.rstrip() if isinstance(val, str) and val.startswith('\n') and not val.endswith('\n') else
            val.lstrip() if isinstance(val, str) and not val.startswith('\n') and val.endswith('\n') else
            val.strip() if isinstance(val, str) and not val.startswith('\n') and not val.endswith('\n') else val
        )
        for key, val in input_dict.items() if val is not None
    }

    return stripped_dict


def set_job_args(args: Optional[dict],
                 level: Optional['Level'],
                 job_name: str,
                 ) -> dict:
    """
    Set the job args considering args from ``level`` and from ``trsh``.

    Args:
        args (dict): The job specific arguments.
        level (Level): The level of theory.
        job_name (str): The job name.

    Returns:
        dict: The initialized job specific arguments.
    """
    # Ignore user-specified additional job arguments when troubleshooting.
    if args is not None and args and any(val for val in args.values()) \
            and level is not None and level.args and any(val for val in level.args.values()):
        logger.warning(f'When troubleshooting {job_name}, ARC ignores the following user-specified options:\n'
                       f'{pformat(level.args)}')
    elif not args and level is not None:
        args = level.args
    for key in ['keyword', 'block', 'trsh']:
        if key not in args.keys():
            args[key] = dict()
    return args


def which(command: Union[str, list],
          return_bool: bool = True,
          raise_error: bool = False,
          raise_msg: Optional[str] = None,
          env: Optional[str] = None,
          ) -> Optional[Union[bool, str]]:
    """
    Test to see if a command (or a software package via its executable command) is available.

    Args:
        command (Union[str, list]): The command(s) to check (checking whether at least one is available).
        return_bool (bool, optional): Whether to return a Boolean answer.
        raise_error (bool, optional): Whether to raise an error is the command is not found.
        raise_msg (str, optional): An error message to print in addition to a generic message if the command isn't found.
        env (str, optional): A string representation of all environment variables separated by the os.pathsep symbol.

    Raises:
        ModuleNotFoundError: If ``raise_error`` is True and the command wasn't found.

    Returns:
        Optional[Union[bool, str]]:
            The command path or ``None``, returns ``True`` or ``False`` if ``return_bool`` is set to ``True``.
    """
    if env is None:
        lenv = {"PATH": os.pathsep + os.environ.get("PATH", "") + os.path.dirname(sys.executable),
                "PYTHONPATH": os.pathsep + os.environ.get("PYTHONPATH", ""),
                }
    else:
        lenv = {"PATH": os.pathsep.join([os.path.abspath(x) for x in env.split(os.pathsep) if x != ""])}
    lenv = {k: v for k, v in lenv.items() if v is not None}

    command = [command] if isinstance(command, str) else command
    ans = None
    for comm in command:
        ans = shutil.which(comm, mode=os.F_OK | os.X_OK, path=lenv["PATH"] + lenv["PYTHONPATH"])
        if ans:
            break

    if raise_error and ans is None:
        raise_msg = raise_msg if raise_msg is not None else ''
        raise ModuleNotFoundError(f"The command {command}"
                                  f"was not found in envvar PATH nor in PYTHONPATH.\n{raise_msg}")

    if return_bool:
        return bool(ans)
    else:
        return ans

def combine_parameters(input_dict: dict, terms: list) -> Tuple[dict, List]:
    """
    Extract and combine specific parameters from a dictionary's string values based on a list of terms.

    This function iterates over each key-value pair in the input dictionary. If a value is a string,
    it searches for occurrences of each term in the 'terms' list within the string. Each found term is 
    appended to a 'parameters' list. The terms found are then removed from the string value in the dictionary.
    The function returns a modified copy of the input dictionary with the terms removed from the string values,
    and a list of extracted parameters.

    Args:
        input_dict (dict): A dictionary with string values from which terms are to be extracted.
        terms (list): A list of string terms to be searched for within the dictionary's values.
        
    Returns:
        tuple: A tuple containing two elements:
            - A dictionary with the same structure as 'input_dict', but with the terms removed from the string values.
            - A list of extracted parameters (terms found in the string values of the input dictionary).
        
    """

    input_dict_copy = input_dict.copy()
    parameters = []

    for key, value in input_dict_copy.items():
        if isinstance(value, str):
            for term in terms:
                matches = re.findall(term, value)
                for match in matches:
                    if match:
                        parameters.append(match)
                        value = re.sub(term, '', value)
            input_dict_copy[key] = value

    return input_dict_copy, parameters
