"""
An adapter for executing Psi4 jobs

https://www.psicode.org/
"""

import datetime
import math
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

#import psi4
from mako.template import Template

from arc.common import get_logger, torsions_to_scans
from arc.imports import incore_commands, settings
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import (check_argument_consistency,
                                     set_job_args,
                                     update_input_dict_with_args,
                                     which)
from arc.job.factory import register_job_adapter
from arc.job.local import execute_command, submit_job
from arc.level import Level
from arc.job.ssh import SSHClient
from arc.species.converter import xyz_to_str, zmat_from_xyz, zmat_to_str

if TYPE_CHECKING:
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies

logger = get_logger()

constraint_type_dict_optking = {2: 'frozen_distance', 3: 'frozen_angle', 4: 'frozen_dihedral'}  # https://psicode.org/psi4manual/master/optking.html
constraint_type_dict_geometric = {2: 'distance', 3: 'angle', 4: 'dihedral'}  # https://psicode.org/psi4manual/master/optking.html#interface-to-geometric

default_job_settings, global_ess_settings, input_filenames, output_filenames, rotor_scan_resolution, servers, \
    submit_filenames = settings['default_job_settings'], settings['global_ess_settings'], settings['input_filenames'], \
                       settings['output_filenames'], settings['rotor_scan_resolution'], settings['servers'], \
                       settings['submit_filenames']

input_template = """memory ${memory} GB
molecule ${label} {
${charge} ${multiplicity}
${geometry}
}
${scan}
${indent}set basis ${basis}
${indent}set reference uhf
${indent}${function}(${function_args})
${epilogue}

"""


class Psi4Adapter(JobAdapter):
    """
    A class for executing Psi4 jobs.

    Args:
        project (str): The project's name. Used for setting the remote path.
        project_directory (str): The path to the local project directory.
        job_type (list, str): The job's type, validated against ``JobTypeEnum``. If it's a list, pipe.py will be called.
        args (dict, optional): Methods (including troubleshooting) to be used in input files.
                               Keys are either 'keyword', 'block', or 'trsh', values are dictionaries with values
                               to be used either as keywords or as blocks in the respective software input file.
                               If 'trsh' is specified, an action might be taken instead of appending a keyword or a
                               block to the input file (e.g., change server or change scan resolution).
        bath_gas (str, optional): A bath gas. Currently only used in OneDMin to calculate L-J parameters.
        checkfile (str, optional): The path to a previous Gaussian checkfile to be used in the current job.
        conformer (int, optional): Conformer number if optimizing conformers.
        constraints (list, optional): A list of constraints to use during an optimization or scan.
        cpu_cores (int, optional): The total number of cpu cores requested for a job.
        dihedrals (List[float], optional): The dihedral angels corresponding to self.torsions.
        ess_settings (dict, optional): A dictionary of available ESS and a corresponding server list.
        ess_trsh_methods (List[str], optional): A list of troubleshooting methods already tried out.
        execution_type (str, optional): The execution type, 'incore', 'queue', or 'pipe'.
        fine (bool, optional): Whether to use fine geometry optimization parameters. Default: ``False``.
        initial_time (datetime.datetime or str, optional): The time at which this job was initiated.
        irc_direction (str, optional): The direction of the IRC job (`forward` or `reverse`).
        job_id (int, optional): The job's ID determined by the server.
        job_memory_gb (int, optional): The total job allocated memory in GB (14 by default).
        job_name (str, optional): The job's name (e.g., 'opt_a103').
        job_num (int, optional): Used as the entry number in the database, as well as in ``job_name``.
        job_server_name (str, optional): Job's name on the server (e.g., 'a103').
        job_status (list, optional): The job's server and ESS statuses.
        level (Level, optional): The level of theory to use.
        max_job_time (float, optional): The maximal allowed job time on the server in hours (can be fractional).
        reactions (List[ARCReaction], optional): Entries are ARCReaction instances, used for TS search methods.
        rotor_index (int, optional): The 0-indexed rotor number (key) in the species.rotors_dict dictionary.
        server (str): The server to run on.
        server_nodes (list, optional): The nodes this job was previously submitted to.
        species (List[ARCSpecies], optional): Entries are ARCSpecies instances.
                                              Either ``reactions`` or ``species`` must be given.
        testing (bool, optional): Whether the object is generated for testing purposes, ``True`` if it is.
        times_rerun (int, optional): Number of times this job was re-run with the same arguments (no trsh methods).
        torsions (List[List[int]], optional): The 0-indexed atom indices of the torsion(s).
        tsg (int, optional): TSGuess number if optimizing TS guesses.
        xyz (dict, optional): The 3D coordinates to use. If not give, species.get_xyz() will be used.
        dihedral_increment (float, optional): The degrees increment to use when scanning dihedrals of TS guesses.
    """

    def __init__(self,
                 project: str,
                 project_directory: str,
                 job_type: Union[List[str], str],
                 args: Optional[dict] = None,
                 bath_gas: Optional[str] = None,
                 checkfile: Optional[str] = None,
                 conformer: Optional[int] = None,
                 constraints: Optional[List[Tuple[List[int], float]]] = None,
                 cpu_cores: Optional[str] = None,
                 dihedrals: Optional[List[float]] = None,
                 ess_settings: Optional[dict] = None,
                 ess_trsh_methods: Optional[List[str]] = None,
                 execution_type: Optional[str] = None,
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
                 dihedral_increment: Optional[float] = None,
                 ):

        self.job_adapter = 'psi4'
        self.execution_type = execution_type or 'incore'
        self.command = 'psi4.py'
        self.url = 'https://www.psicode.org/'

        if species is None:
            raise ValueError('Cannot execute Psi4 without an ARCSpecies object.')

        if any(arg is None for arg in [job_type, level]):
            raise ValueError(f'All of the following arguments must be given:\n'
                             f'job_type, level, project, project_directory\n'
                             f'Got: {job_type}, {level}, {project}, {project_directory}, respectively')

        self.project = project
        self.project_directory = project_directory
        if self.project_directory and not os.path.isdir(self.project_directory):
            os.makedirs(self.project_directory)
        self.job_types = job_type if isinstance(job_type, list) else [job_type]  # always a list
        self.job_type = job_type if isinstance(job_type, str) else job_type[0]  # always a string
        self.args = args or dict()
        self.bath_gas = bath_gas
        self.checkfile = checkfile
        self.conformer = conformer
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
        self.job_server_name = job_server_name
        self.job_status = job_status \
            or ['initializing', {'status': 'initializing', 'keywords': list(), 'error': '', 'line': ''}]
        # When restarting ARC and re-setting the jobs, ``level`` is a string, convert it to a Level object instance
        self.level = Level(repr=level) if not isinstance(level, Level) else level
        self.max_job_time = max_job_time or default_job_settings.get('job_time_limit_hrs', 120)
        self.reactions = [reactions] if reactions is not None and not isinstance(reactions, list) else reactions
        self.rotor_index = rotor_index
        self.server = server
        self.server_nodes = server_nodes or list()
        self.species = [species] if species is not None and not isinstance(species, list) else species
        self.testing = testing
        self.torsions = torsions
        self.tsg = tsg
        self.xyz = xyz or self.species[0].get_xyz()
        self.times_rerun = times_rerun

        if self.job_num is None or self.job_name is None or self.job_server_name:
            self._set_job_number()

        self.args = set_job_args(args=self.args, level=self.level, job_name=self.job_name)

        self.final_time = None
        self.run_time = None
        self.charge = self.species[0].charge
        self.multiplicity = self.species[0].multiplicity
        self.is_ts = self.species[0].is_ts
        self.scan_res = self.args['trsh']['scan_res'] if 'scan_res' in self.args['trsh'] else rotor_scan_resolution

        self.server = self.args['trsh']['server'] if 'server' in self.args['trsh'] \
            else self.ess_settings[self.job_adapter][0] if isinstance(self.ess_settings[self.job_adapter], list) \
            else self.ess_settings[self.job_adapter]
        self.species_label = self.species[0].label
        if len(self.species) > 1:
            self.species_label += f'_and_{len(self.species) - 1}_others'

        self.cpu_cores, self.input_file_memory, self.submit_script_memory = None, None, None
        self.set_cpu_and_mem()
        self.set_file_paths()

        self.workers = None
        self.iterate_by = list()
        self.number_of_processes = 0
        self.incore_capacity = 5
        self.determine_job_array_parameters()  # Writes the local HDF5 file if needed.

        self.files_to_upload = list()
        self.files_to_download = list()
        self.set_files()  # Set the actual files (and write them if relevant).

        self.restrarted = bool(job_num)  # If job_num was given, this is a restarted job, don't save as initiated jobs.
        self.additional_job_info = None

        check_argument_consistency(self)

    def write_input_file(self) -> None:
        """
        Write the input file to execute the job on the server.
        """
        scan, indent, epilogue = '', '', ''
        geometry = xyz_to_str(self.get_geometry())
        if self.job_type in ['conformers', 'opt']:
            func = 'optimize'
        elif self.job_type == 'sp':
            func = 'energy'
        elif self.job_type == 'freq':
            func = 'frequency'
        elif self.job_type == 'scan':
            # todo: parameter in zmat, or not? zmat or xyz? zero PES in ND
            func = 'E = optimize'
            scan += '\nPES = list()\n'
            for i, torsion in enumerate(self.torsions):
                scan += f'{indent}for t{i} in range({int((360.0 / self.scan_res) + 1)}):\n'
                indent += '  '
            fixed_dihedrals, new_indent = '', ''
            scans = torsions_to_scans(self.torsions)
            for i, scan_ in enumerate(scans):
                fixed_dihedrals += f'{scan_[0]} {scan_[1]} {scan_[2]} {scan_[3]} t{i} * {self.scan_res} '
            scan += f'{indent}set optking fixed_dihedral = {fixed_dihedrals}'
            epilogue = f'{indent}PES{"".join(["[t" + str(i) +"]" for i in range(len(self.torsions))])} = E\n\n' \
                       f'print("\\nPES energy as a function of phis:\\n")\n'
            for i, torsion in enumerate(self.torsions):
                epilogue += f'for t{i} in range({(360.0 / self.scan_res) + 1}):\n'
                new_indent += '  '
            epilogue += f'{new_indent}print(PES{"".join(["[t" + str(i) +"]" for i in range(len(self.torsions))])})'
            geometry = zmat_to_str(zmat=zmat_from_xyz(xyz=self.get_geometry(),
                                                      mol=self.species[0].mol if self.species is not None else None,
                                                      constraints={'D_groups': self.torsions},
                                                      consolidate=False,
                                                      is_ts=self.species[0].is_ts if self.species is not None else False,
                                                      ),
                                   zmat_format='psi4',
                                   consolidate=False,
                                   )
        else:
            raise NotImplementedError(f'Psi4 job type {self.job_type} is not implemented')

        input_dict = {'memory': self.job_memory_gb,
                      'label': self.species_label,
                      'charge': self.species[0].charge,
                      'multiplicity': self.species[0].multiplicity,
                      'geometry': geometry,
                      'basis': self.level.basis,
                      'scan': scan,
                      'indent': indent,
                      'function': func,
                      'function_args': self.write_func_args(func),
                      'epilogue': epilogue,
                      }

        with open(os.path.join(self.local_path, input_filenames[self.job_adapter]), 'w') as f:
            f.write(Template(input_template).render(**input_dict))

    def set_files(self) -> None:
        """
        Set files to be uploaded and downloaded. Writes the files if needed.
        Modifies the self.files_to_upload and self.files_to_download attributes.

        self.files_to_download is a list of remote paths.

        self.files_to_upload is a list of dictionaries, each with the following keys:
        ``'name'``, ``'source'``, ``'make_x'``, ``'local'``, and ``'remote'``.
        If ``'source'`` = ``'path'``, then the value in ``'local'`` is treated as a file path.
        Else if ``'source'`` = ``'input_files'``, then the value in ``'local'`` will be taken
        from the respective entry in inputs.py
        If ``'make_x'`` is ``True``, the file will be made executable.
        """
        # Todo: check check file in psi4
        # 1. ** Upload **
        # 1.1. submit file
        if self.execution_type != 'incore':
            # we need a submit file for single or array jobs (either submitted to local or via SSH)
            self.write_submit_script()
            self.files_to_upload.append(self.get_file_property_dictionary(
                file_name=submit_filenames[servers[self.server]['cluster_soft']]))
        # 1.2. input file
        if not self.iterate_by:
            # if this is not a job array, we need the ESS input file
            self.write_input_file()
            self.files_to_upload.append(self.get_file_property_dictionary(file_name=input_filenames[self.job_adapter]))
        # 1.3. checkfile
        if self.checkfile is not None and os.path.isfile(self.checkfile):
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='check.chk',
                                                                          local=self.checkfile))
        # 1.4. HDF5 file
        if self.iterate_by and os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='data.hdf5'))
        # 1.5 job.sh
        job_sh_dict = self.set_job_shell_file_to_upload()  # Set optional job.sh files if relevant.
        if job_sh_dict is not None:
            self.files_to_upload.append(job_sh_dict)
        # 2. ** Download **
        # 2.1. HDF5 file
        if self.iterate_by and os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
            self.files_to_download.append(self.get_file_property_dictionary(file_name='data.hdf5'))
        else:
            # 2.2. log file
            self.files_to_download.append(self.get_file_property_dictionary(
                file_name=output_filenames[self.job_adapter]))
            # 2.3. checkfile
            self.files_to_download.append(self.get_file_property_dictionary(file_name='check.chk'))

    def set_additional_file_paths(self) -> None:
        """
        Set additional file paths specific for the adapter.
        Called from set_file_paths() and extends it.
        """
        pass

    def set_input_file_memory(self) -> None:
        """
        Set the input_file_memory attribute.
        """
        # Gaussian's memory is in MB, total for all cpu cores
        self.input_file_memory = math.ceil(self.job_memory_gb * 1024)

    def execute_incore(self):
        """
        Execute a job incore.
        """
        which(self.command,
              return_bool=True,
              raise_error=True,
              raise_msg=f'Please install {self.job_adapter}, see {self.url} for more information.',
              )
        self._log_job_execution()
        execute_command(incore_commands[self.server][self.job_adapter])

    def execute_queue(self):
        """
        Execute a job to the server's queue.
        """
        self._log_job_execution()
        # Submit to queue, differentiate between local (same machine using its queue) and remote servers.
        if self.server != 'local':
            with SSHClient(self.server) as ssh:
                self.job_status[0], self.job_id = ssh.submit_job(remote_path=self.remote_path)
        else:
            # submit to the local queue
            self.job_status[0], self.job_id = submit_job(path=self.local_path)

    def get_geometry(self):
        if self.xyz is not None:
            return self.xyz
        elif self.species[0] is not None:
            return self.species[0].get_xyz()
        else:
            raise ValueError("Geometry is required to preform the calculations.")

    def write_func_args(self, func) -> str:
        if func in ['optimize', 'conformers', 'scan', 'E = optimize']:
            func_args = f"name = '{self.level.method}', return_wfn='on', return_history='on', engine='optking', dertype='energy'"
        elif func == 'energy':
            func_args = f"name = '{self.level.method}', return_wfn='on'"
        else:
            func_args = f"name = '{self.level.method}', return_wfn='on'"
        return func_args


register_job_adapter('psi4', Psi4Adapter)
