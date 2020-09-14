"""
An adapter for executing Gaussian jobs

https://gaussian.com/
"""

import datetime
import math
import os
from pprint import pformat
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from mako.template import Template

from arc.common import get_logger
from arc.exceptions import JobError
from arc.imports import incore_commands, settings
from arc.job.adapter import JobAdapter, constraint_type_dict
from arc.job.factory import register_job_adapter
from arc.job.local import execute_command, submit_job
from arc.job.ssh import SSHClient
from arc.species.converter import xyz_to_str

if TYPE_CHECKING:
    import datetime
    from arc.level import Level
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies

logger = get_logger()

default_job_settings, global_ess_settings, input_filenames, output_filenames, rotor_scan_resolution, servers, \
    submit_filenames = settings['default_job_settings'], settings['global_ess_settings'], settings['input_filenames'], \
                       settings['output_filenames'], settings['rotor_scan_resolution'], settings['servers'], \
                       settings['submit_filenames']


# job_type_1: '' for sp, irc, or composite methods, 'opt=calcfc', 'opt=(calcfc,ts,noeigen)',
# job_type_2: '' or 'freq iop(7/33=1)' (cannot be combined with CBS-QB3)
#             'scf=(tight,direct) int=finegrid irc=(rcfc,forward,maxpoints=100,stepsize=10) geom=check' for irc f
#             'scf=(tight,direct) int=finegrid irc=(rcfc,reverse,maxpoints=100,stepsize=10) geom=check' for irc r
# scan: '\nD 3 1 5 8 S 36 10.000000' (with the line break)
# restricted: '' or 'u' for restricted / unrestricted
# `iop(2/9=2000)` makes Gaussian print the geometry in the input orientation even for molecules with more
#   than 50 atoms (important so it matches the hessian, and so that Arkane can parse the geometry)
input_template = """${checkfile}
%%mem=${memory}mb
%%NProcShared=${cpus}

#P ${job_type_1} ${restricted}${method}${slash_1}${basis}${slash_2}${auxiliary_basis} ${job_type_2} ${fine} IOp(2/9=2000) ${keywords} ${dispersion}

${label}

${charge} ${multiplicity}
${xyz}${scan}${scan_trsh}${block}


"""


class GaussianAdapter(JobAdapter):
    """
    A class for executing Gaussian jobs.

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

        self.job_adapter = 'gaussian'

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
            or ['initializing', {'status': 'initializing', 'keywords': list(), 'error': '', 'line': ''}]
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
            logger.warning(f'When troubleshooting {self.job_name}, ARC ignores the following user-specified options:\n'
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

        if self.job_type == 'irc' and (self.irc_direction is None or self.irc_direction not in ['forward', 'reverse']):
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

    def write_input_file(self) -> None:
        """
        Write the input file to execute the job on the server.
        """
        input_dict = dict()
        for key in ['block',
                    'dispersion',
                    'fine',
                    'job_type_1',
                    'job_type_2',
                    'keywords',
                    'restricted',
                    'scan',
                    'slash_1',
                    'slash_2',
                    ]:
            input_dict[key] = ''
        input_dict['auxiliary_basis'] = self.level.auxiliary_basis or ''
        input_dict['basis'] = self.level.basis or ''
        input_dict['charge'] = self.charge
        input_dict['checkfile'] = '%chk=check.chk'
        input_dict['cpus'] = self.cpu_cores
        input_dict['label'] = self.species_label
        input_dict['memory'] = self.input_file_memory
        input_dict['method'] = self.level.method
        input_dict['multiplicity'] = self.multiplicity
        input_dict['scan_trsh'] = self.args['trsh']['scan_trsh'] if 'scan_trsh' in self.args['trsh'] else ''
        input_dict['xyz'] = xyz_to_str(self.species[0].get_xyz())

        for arg_type, arg_dict in self.args.items():
            if arg_type == 'block' and arg_dict:
                input_dict['block'] = '\n\n' if not input_dict['block'] else input_dict['block']
                for block in arg_dict.values():
                    input_dict['block'] += f'{block}\n'
            elif arg_type == 'keyword' and arg_dict:
                for key, keyword in arg_dict.items():
                    input_dict['keywords'] += f'{keyword} '

        if self.level.basis is not None:
            input_dict['slash_1'] = '/'
            if self.level.auxiliary_basis is not None:
                input_dict['slash_2'] = '/'

        if self.level.method_type in ['semiempirical', 'force_field']:
            self.checkfile = None

        # Determine HF/DFT restriction type
        if (self.multiplicity > 1 and self.level.method_type != 'composite') \
                or (self.species[0].number_of_radicals is not None and self.species[0].number_of_radicals > 1):
            # run an unrestricted electronic structure calculation if the spin multiplicity is greater than one,
            # or if it is one but the number of radicals is greater than one (e.g., bi-rad singlet)
            # don't run unrestricted for composite methods such as CBS-QB3, it'll be done automatically if the
            # multiplicity is greater than one, but do specify uCBS-QB3 for example for bi-rad singlets.
            if self.species[0].number_of_radicals is not None and self.species[0].number_of_radicals > 1:
                logger.info(f'Using an unrestricted method for species {self.species_label} which has '
                            f'{self.species[0].number_of_radicals} radicals and multiplicity {self.multiplicity}.')
            input_dict['restricted'] = 'u'

        if self.level.dispersion is not None:
            input_dict['dispersion'] = self.level.dispersion

        if self.level.method[:2] == 'ro':
            self.add_to_args(val='use=L506')
        else:
            # xqc will do qc (quadratic convergence) if the job fails w/o it, so use by default
            self.add_to_args(val='scf=xqc')

        if self.level.method == 'cbs-qb3-paraskevas':
            # convert cbs-qb3-paraskevas to cbs-qb3
            self.level.method = 'cbs-qb3'

        # Job type specific options
        if self.job_type in ['opt', 'conformers', 'optfreq', 'composite']:
            keywords = ['ts', 'calcfc', 'noeigentest', 'maxcycles=100'] if self.is_ts else ['calcfc']
            if self.level.method in ['rocbs-qb3']:
                # There're no analytical 2nd derivatives (FC) for this method.
                keywords = ['ts', 'noeigentest', 'maxcycles=100'] if self.is_ts else []
            if self.fine:
                if self.level.method_type in ['dft', 'composite']:
                    # Note that the Acc2E argument is not available in Gaussian03
                    input_dict['fine'] = 'scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)'
                if self.is_ts:
                    keywords.extend(['tight', 'maxstep=5'])
                else:
                    keywords.extend(['tight', 'maxstep=5'])
            input_dict['job_type_1'] = f"opt=({', '.join(key for key in keywords)})"

        elif self.job_type == 'freq':
            input_dict['job_type_2'] = 'freq IOp(7/33=1) scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)'

        elif self.job_type == 'optfreq':
            input_dict['job_type_2'] = 'freq IOp(7/33=1)'

        elif self.job_type == 'sp':
            input_dict['job_type_1'] = 'scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)'

        elif self.job_type == 'scan' and (not self.species[0].rotors_dict or self.species[0].rotors_dict
                and self.species[0].rotors_dict[self.rotor_index]['directed_scan_type'] == 'ess'):
            # In a pipe run, the species object is initialized with species.rotors_dict as an empty dict.
            scans = list()
            if self.species[0].rotors_dict:
                for scan_indices in self.species[0].rotors_dict[self.rotor_index]['scan']:
                    scans.append(' '.join([str(atom_index) for atom_index in scan_indices]))
            else:
                for torsion in self.torsions:
                    scans.append(' '.join([str(atom_index + 1) for atom_index in torsion]))
            ts = 'ts, ' if self.is_ts else ''
            input_dict['job_type_1'] = f'opt=({ts}modredundant, calcfc, noeigentest, maxStep=5) scf=(tight, direct) ' \
                                       f'integral=(grid=ultrafine, Acc2E=12)'
            input_dict['scan'] = '\n\n' if not input_dict['scan'] else input_dict['scan']
            for scan in scans:
                input_dict['scan'] += f'D {scan} S {int(360 / self.scan_res)} {self.scan_res:.1f}\n'

        elif self.job_type == 'irc':
            if self.fine:
                # Note that the Acc2E argument is not available in Gaussian03
                input_dict['fine'] = 'scf=(direct) integral=(grid=ultrafine, Acc2E=12)'
            input_dict['job_type_1'] = f'irc=(CalcAll, {self.irc_direction}, maxpoints=50, stepsize=7)'

        for constraint_tuple in self.constraints:
            constraint_type = constraint_type_dict[len(constraint_tuple[0])]
            constraint_atom_indices = ' '.join([str(atom_index) for atom_index in constraint_tuple[0]])
            input_dict['scan'] = '\n\n' if not input_dict['scan'] else input_dict['scan']
            input_dict['scan'] += f"{constraint_type} {constraint_atom_indices} ={constraint_tuple[1]:.2f} B\n" \
                                  f"{constraint_type} {constraint_atom_indices} F"

        if self.level.solvation_method is not None:
            input_dict['job_type_1'] += f' SCRF=({self.level.solvation_method}, Solvent={self.level.solvent})'

        input_dict['job_type_1'] += ' guess=read' if self.checkfile is not None else ' guess=mix'

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
        self.files_to_upload, self.files_to_download = list(), list()

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

    def execute(self) -> Optional[dict]:
        """
        Execute a job.
        The execution type could be incore, single, or array.

        Returns: Optional[dict]
            May return a value only if running incore through an API.
        """
        self._log_job_execution()

        self.write_input_file()

        if self.execution_type == 'incore':
            if self.tasks is not None and os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
                # Todo: execute pipe incore, set self.job_status[0]
                pass
            else:
                execute_command(incore_commands[self.server][self.job_adapter])
        else:
            # submit to queue
            if self.tasks is not None and os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
                # Todo: submit pipe either locally or via ssh, set self.job_status[0]
                pass
            if self.server != 'local':
                # submit_job returns job server status and job server id
                with SSHClient(self.server) as ssh:
                    self.job_status[0], self.job_id = ssh.submit_job(remote_path=self.remote_path)
            else:
                # submit to queue locally
                self.job_status[0], self.job_id = submit_job(path=self.local_path)
        return None


register_job_adapter('gaussian', GaussianAdapter)
