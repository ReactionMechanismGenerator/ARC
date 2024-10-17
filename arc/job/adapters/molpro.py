"""
An adapter for executing molpro jobs

https://www.molpro.net/
"""

import datetime
import math
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import socket

from mako.template import Template

from arc.common import get_logger
from arc.exceptions import JobError
from arc.imports import incore_commands, settings
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import (_initialize_adapter,
                                     is_restricted,
                                     update_input_dict_with_args,
                                     which,
                                     )
from arc.job.factory import register_job_adapter
from arc.job.local import execute_command
from arc.level import Level
from arc.species.converter import xyz_to_str

if TYPE_CHECKING:
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies

logger = get_logger()

default_job_settings, global_ess_settings, input_filenames, output_filenames, servers, submit_filenames = \
    settings['default_job_settings'], settings['global_ess_settings'], settings['input_filenames'], \
    settings['output_filenames'], settings['servers'], settings['submit_filenames']

input_template = """***,${label}
memory,${memory},m;

geometry={angstrom;
${xyz}}
${orbitals}
basis=${basis}
${keywords}
${auxiliary_basis}
${cabs}
int;
{hf;${shift}
maxit,1000;
wf,spin=${spin},charge=${charge};}

${restricted}${method};

${job_type_1}
${job_type_2}${block}
---;

"""


class MolproAdapter(JobAdapter):
    """
    A class for executing Molpro jobs.

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
        dihedral_increment (float, optional): The degrees increment to use when scanning dihedrals of TS guesses.
        dihedrals (List[float], optional): The dihedral angels corresponding to self.torsions.
        directed_scan_type (str, optional): The type of the directed scan.
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
        run_multi_species (bool, optional): Whether to run a job for multiple species in the same input file.
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
                 dihedral_increment: Optional[float] = None,
                 dihedrals: Optional[List[float]] = None,
                 directed_scan_type: Optional[str] = None,
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
                 run_multi_species: bool = False,
                 reactions: Optional[List['ARCReaction']] = None,
                 rotor_index: Optional[int] = None,
                 server: Optional[str] = None,
                 server_nodes: Optional[list] = None,
                 queue: Optional[str] = None,
                 attempted_queues: Optional[List[str]] = None,
                 species: Optional[List['ARCSpecies']] = None,
                 testing: bool = False,
                 times_rerun: int = 0,
                 torsions: Optional[List[List[int]]] = None,
                 tsg: Optional[int] = None,
                 xyz: Optional[dict] = None,
                 ):

        self.incore_capacity = 1
        self.job_adapter = 'molpro'
        self.execution_type = execution_type or 'queue'
        self.command = 'molpro'
        self.url = 'https://www.molpro.net/'

        if species is None:
            raise ValueError('Cannot execute Molpro without an ARCSpecies object.')

        _initialize_adapter(obj=self,
                            is_ts=False,
                            project=project,
                            project_directory=project_directory,
                            job_type=job_type,
                            args=args,
                            bath_gas=bath_gas,
                            checkfile=checkfile,
                            conformer=conformer,
                            constraints=constraints,
                            cpu_cores=cpu_cores,
                            dihedral_increment=dihedral_increment,
                            dihedrals=dihedrals,
                            directed_scan_type=directed_scan_type,
                            ess_settings=ess_settings,
                            ess_trsh_methods=ess_trsh_methods,
                            fine=fine,
                            initial_time=initial_time,
                            irc_direction=irc_direction,
                            job_id=job_id,
                            job_memory_gb=job_memory_gb,
                            job_name=job_name,
                            job_num=job_num,
                            job_server_name=job_server_name,
                            job_status=job_status,
                            level=level,
                            max_job_time=max_job_time,
                            run_multi_species=run_multi_species,
                            reactions=reactions,
                            rotor_index=rotor_index,
                            server=server,
                            server_nodes=server_nodes,
                            queue=queue,
                            attempted_queues=attempted_queues,
                            species=species,
                            testing=testing,
                            times_rerun=times_rerun,
                            torsions=torsions,
                            tsg=tsg,
                            xyz=xyz,
                            )

    def write_input_file(self) -> None:
        """
        Write the input file to execute the job on the server.
        """
        input_dict = dict()
        for key in ['block',
                    'job_type_1',
                    'job_type_2',
                    'keywords',
                    'memory',
                    'method',
                    'orbitals',
                    'restricted',
                    ]:
            input_dict[key] = ''
        input_dict['auxiliary_basis'] = self.level.auxiliary_basis or ''
        input_dict['basis'] = 'cc-pVDZ' if 'trsh' in self.args and 'vdz' in self.args['trsh'] \
            else self.level.basis or ''
        input_dict['cabs'] = self.level.cabs or ''
        input_dict['charge'] = self.charge
        input_dict['label'] = self.species_label
        input_dict['memory'] = self.input_file_memory
        input_dict['method'] = self.level.method
        input_dict['shift'] = self.args['trsh']['shift'] if 'shift' in self.args['trsh'].keys() else ''
        input_dict['spin'] = self.multiplicity - 1
        input_dict['xyz'] = xyz_to_str(self.xyz)

        if not is_restricted(self):
            input_dict['restricted'] = 'u'

        # Job type specific options
        if self.job_type in ['opt', 'optfreq', 'conf_opt']:
            keywords = ['optg', 'root=2', 'method=qsd', 'readhess', "savexyz='geometry.xyz'"] if self.is_ts \
                else ['optg', "savexyz='geometry.xyz'"]
            input_dict['job_type_1'] = ', '.join(key for key in keywords)

        elif self.job_type in ['freq', 'optfreq']:
            input_dict['job_type_2'] = '{frequencies;\nthermo;\nprint,HESSIAN,thermo;}'

        elif self.job_type in ['sp', 'conf_sp']:
            pass

        elif self.job_type == 'scan':
            # Todo: Automatically go with directed scans instead of ess scan
            pass

        if 'IGNORE_ERROR in the ORBITAL directive' in self.args['trsh'].keys():
            keywords.append('ORBITAL,IGNORE_ERROR')

        if 'mrci' in self.level.method:
            if self.species[0].occ > 16:
                raise JobError(f'Will not execute an MRCI calculation with more than 16 occupied orbitals '
                               f'(got {self.species[0].occ}).\n'
                               f'Selective occ, closed, core, frozen keyword still not implemented.')
            input_dict['orbitals'] = '\ngprint,orbitals;\n'
            input_dict['block'] += '\n\nE_mrci=energy;\nE_mrci_Davidson=energd;\n\ntable,E_mrci,E_mrci_Davidson;'
            input_dict['method'] = input_dict['rerstricted'] = ''
            input_dict['shift'] = 'shift,-1.0,-0.5;'
            input_dict['job_type_1'] = f"""{{multi;
{self.species[0].occ}noextra,failsafe,config,csf;
wf,spin={input_dict['spin']},charge={input_dict['charge']};
natorb,print,ci;}}"""
            input_dict['job_type_2'] = f"""{{mrci;
${self.species[0].occ}wf,spin=${input_dict['spin']},charge=${input_dict['charge']};}}"""

        input_dict = update_input_dict_with_args(args=self.args, input_dict=input_dict)

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
        # 1.3. HDF5 file
        if self.iterate_by and os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='data.hdf5'))
        # 1.4 job.sh
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
        # Molpro's memory is per cpu core and in MW (mega word; 1000 MW = 7.45 GB on a 64-bit machine)
        # The conversion from mW to GB was done using this (https://deviceanalytics.com/words-to-bytes-converter/)
        # specifying a 64-bit architecture.
        #
        # See also:
        # https://www.molpro.net/pipermail/molpro-user/2010-April/003723.html
        # In the link, they describe the conversion of 100,000,000 Words (100Mword) is equivalent to
        # 800,000,000 bytes (800 mb).
        # Formula - (100,000,000 [Words]/( 800,000,000 [Bytes] / (job mem in gb * 1000,000,000 [Bytes])))/ 1000,000 [Words -> MegaWords]
        # The division by 1E6 is for converting into MWords
                # Due to Zeus's configuration, there is only 1 nproc so the memory should not be divided by cpu_cores. 
        self.input_file_memory = math.ceil(self.job_memory_gb / (7.45e-3 * self.cpu_cores)) if 'zeus' not in socket.gethostname() else math.ceil(self.job_memory_gb / (7.45e-3))
        
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
        execute_command(incore_commands[self.job_adapter])

    def execute_queue(self):
        """
        Execute a job to the server's queue.
        """
        self.legacy_queue_execution()


register_job_adapter('molpro', MolproAdapter)
