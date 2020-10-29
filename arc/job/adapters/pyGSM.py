"""
A TS adapter for using the double ended growing string method (DE GSM)
to predict TS guesses for any reaction. This adapter uses pyGSM, a Python wrapper
on the original GSM code from the Zimmerman research group.
https://github.com/ZimmermanGroup/pyGSM
"""

import datetime
import os
from pprint import pformat
from typing import TYPE_CHECKING, List, Optional, Tuple, Union


from arc.common import get_logger
from arc.imports import incore_commands, settings
from arc.job.adapter import JobAdapter
from arc.job.factory import register_job_adapter, constraint_type_dict
from arc.parser import _get_lines_from_file
from arc.species.converter import calc_rmsd_wrapper, str_to_xyz, xyz_file_format_to_xyz

if TYPE_CHECKING:
    from arc.reaction import ARCReaction

logger = get_logger()

default_job_settings, global_ess_settings, input_filenames, output_filenames, rotor_scan_resolution, servers, \
    submit_filenames = settings['default_job_settings'], settings['global_ess_settings'], settings['input_filenames'], \
                       settings['output_filenames'], settings['rotor_scan_resolution'], settings['servers'], \
                       settings['submit_filenames']

def read_xyzs(filename):
    """
    Read the xyz file written for DE GSM
    Args:
        filename (str): name of xyz file to read
    Returns:
        reactant_dict (dict): ARC xyz dictionary for the reactant
        product_dict (dict): ARC xyz dictionary for the product
    """
    lines = open(filename).readlines()
    num_atoms = int(lines[0])
    start_atom = 2
    end_atom = start_atom + num_atoms
    reactant_str = ""
    for line in lines[start_atom:end_atom]:
        reactant_str += line
    reactant_dict = str_to_xyz(reactant_str)

    start_atom = end_atom + 2
    end_atom = start_atom + num_atoms
    product_str = ""
    for line in lines[start_atom:end_atom]:
        product_str += line
    product_dict = str_to_xyz(product_str)

    return reactant_dict, product_dict


class PyGSM(JobAdapter):
    """
    A class for executing pyGSM jobs.

    Args:
        execution_type (str): The execution type, validated against ``JobExecutionTypeEnum``.
        job_type (str): The job's type, validated against ``JobTypeEnum``.
                        If it's a list, pipe.py will be called.
        level (Level): The level of theory to use.
        project (str): The project's name. Used for setting the remote path.
        project_directory (str): The path to the local project directory.
        args (dict, optional): Methods (including troubleshooting) to be used in input files.
                               Keys are either 'keyword', 'block', or 'trsh', values are dictionaries with values
                               to be used either as keywords or as blocks in the respective software input file.
                               If 'trsh' is specified, an action will be taken instead of appending a keyword or a block
                               to the input file (e.g., change server or change scan resolution).
        bath_gas (str, optional): A bath gas. Currently only used in OneDMin to calculate L-J parameters. Allowed values
                                  are: ``'He'``, ``'Ne'``, ``'Ar'``, ``'Kr'``, ``'H2'``, ``'N2'``, or ``'O2'``.
        checkfile (str, optional): The path to a previous Gaussian checkfile to be used in the current job.
        constraints (list, optional): A list of constraints to use during an optimization or scan.
        cpu_cores (int, optional): The total number of cpu cores requested for a job.
        dihedrals (List[float], optional): The dihedral angels corresponding to self.torsions.
        ess_settings (dict, optional): A dictionary of available ESS and a corresponding server list.
        ess_trsh_methods (List[str], optional): A list of troubleshooting methods already tried out.
        fine (bool, optional): Whether to use fine geometry optimization parameters. Default: ``False``
        initial_time (datetime.datetime, optional): The time at which this job was initiated.
        irc_direction (str, optional): The direction of the IRC job (`forward` or `reverse`).
        job_id (int, optional): The job's ID determined by the server.
        job_memory_gb (int, optional): The total job allocated memory in GB (14 by default).
        job_name (str, optional): The job's name (e.g., 'opt_a103').
        job_num (int, optional): Used as the entry number in the database, as well as in ``job_name``.
        job_status (int, optional): The job's server and ESS statuses.
        max_job_time (float, optional): The maximal allowed job time on the server in hours (can be fractional).
        reactions (List[ARCReaction], optional): Entries are ARCReaction instances, used for TS search methods.
        rotor_index (int, optional): The 0-indexed rotor number (key) in the species.rotors_dict dictionary.
        server_nodes (list, optional): The nodes this job was previously submitted to.
        species (List[ARCSpecies], optional): Entries are ARCSpecies instances.
                                              Either ``reactions`` or ``species`` must be given.
        tasks (int, optional): The number of tasks to use in a job array (each task has several threads).
        testing (bool, optional): Whether the object is generated for testing purposes, ``True`` if it is.
        torsions (List[List[int]], optional): The 0-indexed atom indices of the torsions identifying this scan point.
        """

        def __init__(self,
                     execution_type: str,
                     job_type: Union[List[str], str],
                     level: 'Level',
                     project: str,
                     project_directory: str,
                     args: Optional[dict] = None,
                     bath_gas: Optional[str] = None,
                     checkfile: Optional[str] = None,
                     constraints: Optional[List[Tuple[List[int], float]]] = None,
                     cpu_cores: Optional[str] = None,
                     dihedrals: Optional[List[float]] = None,
                     ess_settings: Optional[dict] = None,
                     ess_trsh_methods: Optional[List[str]] = None,
                     fine: bool = False,
                     initial_time: Optional['datetime.datetime'] = None,
                     irc_direction: Optional[str] = None,
                     job_id: Optional[int] = None,
                     job_memory_gb: float = 14.0,
                     job_name: Optional[str] = None,
                     job_num: Optional[int] = None,
                     job_status: Optional[List[Union[dict, str]]] = None,
                     max_job_time: Optional[float] = None,
                     reactions: Optional[List['ARCReaction']] = None,
                     rotor_index: Optional[int] = None,
                     server_nodes: Optional[list] = None,
                     species: Optional[List['ARCSpecies']] = None,
                     tasks: Optional[int] = None,
                     testing: bool = False,
                     torsions: List[List[int]] = None,
                     ):

            self.job_adapter = 'pyGSM'

            # add an incore execution of cont opt scan directed by ARC
            self.execution_type = execution_type
            self.job_types = job_type if isinstance(job_type, list) else [job_type]  # always a list
            self.job_type = job_type if isinstance(job_type, str) else job_type[0]  # always a string
            self.level = level
            self.project = project
            self.project_directory = project_directory
            self.args = args or dict()
            self.bath_gas = bath_gas
            self.checkfile = checkfile
            self.constraints = constraints or list()
            self.cpu_cores = cpu_cores
            self.dihedrals = dihedrals
            self.ess_settings = ess_settings or global_ess_settings
            self.ess_trsh_methods = ess_trsh_methods or list()
            self.fine = fine
            self.initial_time = datetime.datetime.strptime(initial_time, '%Y-%m-%d %H:%M:%S') \
                if isinstance(initial_time, str) else initial_time
            self.irc_direction = irc_direction
            self.job_id = job_id
            self.job_memory_gb = job_memory_gb
            self.job_name = job_name
            self.job_num = job_num
            self.job_status = job_status \
                              or ['initializing',
                                  {'status': 'initializing', 'keywords': list(), 'error': '', 'line': ''}]
            self.max_job_time = max_job_time or default_job_settings.get('job_time_limit_hrs', 120)
            self.reactions = reactions
            self.rotor_index = rotor_index
            self.server_nodes = server_nodes or list()
            self.species = species
            self.tasks = tasks
            self.testing = testing
            self.torsions = torsions

            if self.species is None:
                raise ValueError('Cannot execute Gaussian without an ARCSpecies object.')

            # Ignore user-specified additional job arguments when troubleshoot.
            if self.args and any(val for val in self.args.values()) \
                    and self.level.args and any(val for val in self.level.args.values()):
                logger.warning(
                    f'When troubleshooting {self.job_name}, ARC ignores the following user-specified options:\n'
                    f'{pformat(self.level.args)}')
            elif not self.args:
                self.args = self.level.args
            # Add missing keywords to args.
            for key in ['keyword', 'block', 'trsh']:
                if key not in self.args:
                    self.args[key] = dict()

            if self.job_num is None:
                self._set_job_number()
                self.job_name = f'{self.job_type}_a{self.job_num}'

            if self.job_type == 'irc' and (
                    self.irc_direction is None or self.irc_direction not in ['forward', 'reverse']):
                raise ValueError(f'Missing the irc_direction argument for job type irc. '
                                 f'It must be either "forward" or "reverse".\nGot: {self.irc_direction}')

            self.final_time = None
            self.charge = self.species[0].charge
            self.multiplicity = self.species[0].multiplicity
            self.is_ts = self.species[0].is_ts
            self.run_time = None
            self.scan_res = self.args['trsh']['scan_res'] if 'scan_res' in self.args['trsh'] else rotor_scan_resolution
            if self.job_type == 'scan' and divmod(360, self.scan_res)[1]:
                raise ValueError(f'Got an illegal rotor scan resolution of {self.scan_res}.')
            if self.job_type == 'scan' and not self.species[0].rotors_dict and self.torsions is None:
                # If this is a scan job type and species.rotors_dict is empty (e.g., via pipe), then torsions must be set up
                raise ValueError(f'Either a species rotors_dict or torsions must be specified for an ESS scan job.')

            self.server = self.args['trsh']['server'] if 'server' in self.args['trsh'] \
                else self.ess_settings[self.job_adapter][0] if isinstance(self.ess_settings[self.job_adapter], list) \
                else self.ess_settings[self.job_adapter]
            # self.species_label = self.species.label if self.species is not None else self.reaction.ts_label
            self.species_label = self.species[0].label

            self.cpu_cores, self.input_file_memory, self.submit_script_memory = None, None, None
            self.set_cpu_and_mem()
            self.set_file_paths()  # Set file paths.

            self.tasks = None
            self.iterate_by = list()
            self.number_of_processes = 0
            self.determine_job_array_parameters()  # writes the local HDF5 file if needed

            self.set_files()  # Set the actual files (and write them if relevant).

            if self.checkfile is None and os.path.isfile(os.path.join(self.local_path, 'check.chk')):
                self.checkfile = os.path.join(self.local_path, 'check.chk')

            if job_num is None:
                # This checks job_num and not self.job_num on purpose.
                # If job_num was given, then don't save as initiated jobs, this is a restarted job.
                self._write_initiated_job_to_csv_file()

    def execute(self):
        """
        Uses DE GSM to determine the TS structure for the given ARC Reaction.
        """
        # self.reactions is a list containing one ARCReaction object
        for arc_reaction in self.reactions:
            # extract dictionary for reactant/s
            reactant1 = arc_reaction.r_species[0]
            if len(arc_reaction.r_species) == 2:
                reactant2 = arc_reaction.r_species[1]
            else:
                reactant2 = None

            # extract dictionary for product/s
            product1 = arc_reaction.p_species[0]
            if len(arc_reaction.p_species) == 2:
                product2 = arc_reaction.p_species[1]
            else:
                product2 = None
            # create directory to store TS calculations for this reaction
            # https://reactionmechanismgenerator.github.io/ARC/output.html
            ts_dir = os.path.join(self.project_dir, 'TSs', arc_reaction.label)
            if os.path.isdir(ts_dir):
                os.mkdir(ts_dir)
            filepath = os.path.join(ts_dir, 'reaction.xyz')
            reactant1_xyz, reactant2_xyz, product1_xyz, product2_xyz, rmsd = calc_rmsd_wrapper(reactant1_xyz=reactant1,
                                                                                               reactant2_xyz=reactant2,
                                                                                               product1_xyz=product1,
                                                                                               product2_xyz=product2,
                                                                                               xyz_file_name=filepath)
            # update the product xyz dictionaries with the atoms sorted to match the products
            arc_reaction.r_species[0].final_xyz = product1_xyz
            if len(arc_reaction.p_species) == 2:
                arc_reaction.r_species[1].final_xyz = product2_xyz

            lines = _get_lines_from_file(filepath)
            read = False
            for line in lines:
                if line == 'Finished GSM!':
                    read = True
                    logger.info(f'pyGSM successfully converged for reaction: {arc_reaction.label}')
                    break
            if read:
                ts_dictionary = xyz_file_format_to_xyz('TSnode.xyz')
                return [ts_dictionary]
            else:
                logger.error(f'pyGSM did not converge for reaction: {arc_reaction.label}')
                return list()


register_job_adapter('pyGSM', PyGSM)
