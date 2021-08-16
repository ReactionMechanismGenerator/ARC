"""
This module contains functions which are shared across multiple Job modules.
As such, it should not import any other ARC modules to avoid circular imports.
"""

import os
import shutil
import sys

from pprint import pformat
from typing import TYPE_CHECKING, Optional, Union

from arc.common import get_logger

if TYPE_CHECKING:
    from arc.level import Level

logger = get_logger()


ts_adapters_by_rmg_family = {'1+2_Cycloaddition': ['kinbot'],
                             '1,2_Insertion_CO': ['kinbot'],
                             '1,2_Insertion_carbene': ['kinbot'],
                             '1,2_shiftC': ['gcn'],
                             '1,2_shiftS': ['gcn', 'kinbot'],
                             '1,3_Insertion_CO2': ['kinbot'],
                             '1,3_Insertion_ROR': ['kinbot'],
                             '1,3_Insertion_RSR': ['kinbot'],
                             '1,4_Cyclic_birad_scission': ['gcn'],
                             '2+2_cycloaddition_CCO': ['kinbot'],
                             '2+2_cycloaddition_CO': ['kinbot'],
                             '2+2_cycloaddition_CS': ['kinbot'],
                             '2+2_cycloaddition_Cd': ['kinbot'],
                             '6_membered_central_C-C_shift': ['gcn'],
                             'Concerted_Intra_Diels_alder_monocyclic_1,2_shiftH': ['gcn'],
                             'Cyclic_Ether_Formation': ['kinbot'],
                             'Cyclopentadiene_scission': ['gcn'],
                             'Diels_alder_addition': ['kinbot'],
                             'H_Abstraction': ['heuristics', 'autotst'],
                             'HO2_Elimination_from_PeroxyRadical': ['kinbot'],
                             'Intra_2+2_cycloaddition_Cd': ['gcn'],
                             'Intra_5_membered_conjugated_C=C_C=C_addition': ['gcn'],
                             'Intra_Diels_alder_monocyclic': ['gcn', 'kinbot'],
                             'Intra_Disproportionation': ['gcn'],
                             'Intra_ene_reaction': ['gcn', 'kinbot'],
                             'intra_H_migration': ['autotst', 'gcn', 'kinbot'],
                             'intra_NO2_ONO_conversion': ['gcn'],
                             'intra_OH_migration': ['gcn', 'kinbot'],
                             'Intra_RH_Add_Endocyclic': ['gcn', 'kinbot'],
                             'Intra_RH_Add_Exocyclic': ['gcn', 'kinbot'],
                             'Intra_R_Add_Endocyclic': ['gcn', 'kinbot'],
                             'Intra_R_Add_Exo_scission': ['gcn'],
                             'Intra_R_Add_Exocyclic': ['gcn', 'kinbot'],
                             'Intra_R_Add_ExoTetCyclic': ['kinbot'],
                             'Intra_Retro_Diels_alder_bicyclic': ['kinbot'],
                             'intra_substitutionCS_isomerization': ['gcn'],
                             'intra_substitutionS_isomerization': ['gcn'],
                             'ketoenol': ['gcn', 'kinbot'],
                             'Korcek_step1': ['gcn'],
                             'Korcek_step2': ['kinbot'],
                             'R_Addition_COm': ['kinbot'],
                             'R_Addition_CSm': ['kinbot'],
                             'R_Addition_MultipleBond': ['autotst', 'kinbot'],
                             'Retroene': ['kinbot'],
                             'Singlet_Carbene_Intra_Disproportionation': ['gcn'],
                             }

all_families_ts_adapters = []


# Default is "queue", "pipe" will be called whenever needed. So just list 'incore'.
default_incore_adapters = ['autotst', 'gcn', 'heuristics', 'kinbot', 'psi4']


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


def check_argument_consistency(obj):
    """
    Check than general arguments of a job adapter are consistent.

    Args:
        obj: The job adapter object.
    """
    if obj.job_type == 'irc' and obj.job_adapter in ['molpro']:
        raise NotImplementedError(f'IRC is not implemented for the {obj.job_adapter} job adapter.')
    if obj.job_type == 'irc' and (obj.irc_direction is None or obj.irc_direction not in ['forward', 'reverse']):
        raise ValueError(f'Missing the irc_direction argument for job type irc. '
                         f'It must be either "forward" or "reverse".\nGot: {obj.irc_direction}')
    if obj.job_type == 'scan' and obj.job_adapter in ['molpro'] \
            and any(species.rotors_dict['directed_scan_type'] == 'ess' for species in obj.species):
        raise NotImplementedError(f'The {obj.job_adapter} job adapter does not support ESS scans.')
    if obj.job_type == 'scan' and divmod(360, obj.scan_res)[1]:
        raise ValueError(f'Got an illegal rotor scan resolution of {obj.scan_res}.')
    if obj.job_type == 'scan' and ((not obj.species[0].rotors_dict or obj.rotor_index is None) and obj.torsions is None):
        # If this is a scan job type and species.rotors_dict is empty (e.g., via pipe), then torsions must be set up
        raise ValueError(f'Either torsions or a species rotors_dict along with a rotor_index argument '
                         f'must be specified for an ESS scan job.')


def update_input_dict_with_args(args: dict,
                                input_dict: dict,
                                ) -> dict:
    """
    Update the job input_dict attribute with keywords and blocks from the args attribute.

    Args:
        args (dict): The job arguments.
        input_dict (dict): The job input dict.

    Returns:
        dict: The updated input_dict.
    """
    for arg_type, arg_dict in args.items():
        if arg_type == 'block' and arg_dict:
            input_dict['block'] = '\n\n' if not input_dict['block'] else input_dict['block']
            for block in arg_dict.values():
                input_dict['block'] += f'{block}\n'
        elif arg_type == 'keyword' and arg_dict:
            for key, value in arg_dict.items():
                if key == 'scan_trsh':
                    input_dict['scan_trsh'] += f'{value} '
                else:
                    input_dict['keywords'] += f'{value} '
    return input_dict


def set_job_args(args: Optional[dict],
                 level: 'Level',
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
    if args and any(val for val in args.values()) \
            and level.args and any(val for val in level.args.values()):
        logger.warning(f'When troubleshooting {job_name}, ARC ignores the following user-specified options:\n'
                       f'{pformat(level.args)}')
    elif not args:
        args = level.args
    # Add missing keywords to args.
    for key in ['keyword', 'block', 'trsh']:
        if key not in args:
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
            The command path or None, returns ``True`` or ``False`` if ``return_bool`` is ``True``.
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
