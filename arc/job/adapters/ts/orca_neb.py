"""
An adapter for executing Orca's Nudged Elastic Band (NEB) jobs.

Orca NEB:
https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/neb.html
"""

import datetime
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from mako.template import Template

from arc.common import get_logger
from arc.imports import incore_commands, settings
from arc.job.adapters.common import is_restricted, which
from arc.job.adapters.orca import _format_orca_method, _format_orca_basis
from arc.job.factory import register_job_adapter
from arc.job.local import execute_command
from arc.level import Level
from arc.parser.parser import parse_geometry
from arc.species import TSGuess
from arc.species.converter import xyz_to_xyz_file_format

from arc.species import ARCSpecies

if TYPE_CHECKING:
    from arc.reaction import ARCReaction

logger = get_logger()

input_filenames, output_filenames, servers, submit_filenames, orca_neb_settings = \
    settings['input_filenames'], settings['output_filenames'], settings['servers'], settings['submit_filenames'], \
    settings['orca_neb_settings']


input_template = """
!${restricted}HF ${method} ${basis} NEB-TS
%%maxcore ${memory}
%%pal nprocs ${cpus} end

%%neb 
   Interpolation    ${interpolation}
   NImages  ${nnodes}
   PrintLevel   3
   PreOpt           ${preopt}
   NEB_END_XYZFILE "${abs_path}/product.xyz"
END

* XYZFILE ${charge} ${multiplicity} ${abs_path}/reactant.xyz
"""


from arc.job.adapters.orca import OrcaAdapter


class OrcaNEBAdapter(OrcaAdapter):
    """
    A class for executing Orca NEB jobs.

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

        if reactions is None:
            raise ValueError('Cannot execute Orca NEB without an ARCReaction object.')

        if reactions and reactions[0].ts_species is None:
            # Create a dummy TS species from the reactants (or products).
            # This is a temporary placeholder to satisfy OrcaAdapter's species requirement.
            # The actual TS geometry will be obtained from the NEB calculation.
            if reactions[0].r_species:
                reactions[0].ts_species = ARCSpecies(label=f'{reactions[0].label}_TS',
                                                      is_ts=True,
                                                      charge=reactions[0].charge,
                                                      multiplicity=reactions[0].multiplicity,
                                                      xyz=reactions[0].r_species[0].get_xyz())
            else:
                raise ValueError('ARCReaction object must contain reactant species to initialize Orca NEB.')

        level = level or orca_neb_settings.get('level', '')
        if not level:
            raise ValueError('A level of theory must be specified for Orca NEB jobs, either in the job arguments or in the settings file.')
        species_for_super = [reactions[0].ts_species]
        super().__init__(project=project,
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
                         execution_type=execution_type,
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
                         species=species_for_super,
                         testing=testing,
                         times_rerun=times_rerun,
                         torsions=torsions,
                         tsg=tsg,
                         xyz=xyz,
                         )

        self.job_adapter = 'orca_neb'
        self.url = 'https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/neb.html'
        self.command = 'orca'
        self.local_path_to_output_file = os.path.join(self.local_path, output_filenames[self.job_adapter])
        self.execution_type = execution_type or 'queue'

    def write_input_file(self) -> None:
        """
        Write the input file to execute the job on the server.
        """
        input_dict = dict()

        input_dict['restricted'] = 'r' if is_restricted(self) else 'u'
        input_dict['method'] = _format_orca_method(self.level.method)
        input_dict['basis'] = _format_orca_basis(self.level.basis)
        input_dict['memory'] = self.input_file_memory
        input_dict['cpus'] = self.cpu_cores
        input_dict['charge'] = self.charge
        input_dict['multiplicity'] = self.multiplicity
        input_dict['abs_path'] = self.local_path

        # NEB specific parameters
        neb_settings = orca_neb_settings.get('keyword', {})
        input_dict['interpolation'] = neb_settings.get('interpolation', 'IDPP')
        input_dict['nnodes'] = neb_settings.get('nnodes', 15)
        input_dict['preopt'] = neb_settings.get('preopt', 'true')

        # Write reactant.xyz and product.xyz
        atom_map = self.reactions[0].atom_map
        if atom_map is None:
            raise ValueError('Cannot write Orca NEB input file without an atom map in the reaction.')

        reactant_xyz = self.reactions[0].get_reactants_xyz(return_format=dict)
        product_xyz = self.reactions[0].get_products_xyz(return_format=dict) # This implicitly uses the atom map.

        with open(os.path.join(self.local_path, 'reactant.xyz'), 'w') as f:
            f.write(xyz_to_xyz_file_format(reactant_xyz))
        with open(os.path.join(self.local_path, 'product.xyz'), 'w') as f:
            f.write(xyz_to_xyz_file_format(product_xyz))

        # Render and write the NEB input file
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
            # 1.2.1 reactant.xyz and product.xyz
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='reactant.xyz'))
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='product.xyz'))

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
            # 2.3. Hessian file generated by frequency calculations
            # The Hessian file is useful when the user would like to project out the rotors
            if self.job_type in ['freq', 'optfreq']:
                self.files_to_download.append(self.get_file_property_dictionary(file_name='input.hess'))

    def set_additional_file_paths(self) -> None:
        """
        Set additional file paths specific for the adapter.
        Called from set_file_paths() and extends it.
        """
        self.reactant_path = os.path.join(self.local_path, 'reactant.xyz')
        self.product_path = os.path.join(self.local_path, 'product.xyz')

    def cleanup_files(self):
        """Remove unneeded files after run."""
        files_to_remove = ['reactant.xyz', 'product.xyz', input_filenames[self.job_adapter]]
        for file_name in files_to_remove:
            file_path = os.path.join(self.local_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

    def process_run(self):
        """
        Process a completed orca-NEB run.
        """
        tsg = TSGuess(method='orca_neb',
                      index=len(self.reactions[0].ts_species.ts_guesses),
                      success=False,
                      t0=self.initial_time,
                      )
        if os.path.isfile(self.local_path_to_output_file):
            tsg.initial_xyz = parse_geometry(self.local_path_to_output_file)
            tsg.execution_time = self.final_time - self.initial_time
            tsg.success = True
        self.reactions[0].ts_species.ts_guesses.append(tsg)

    def write_submit_script(self) -> None:
        """
        Write a submit script to execute the job.
        """
        original_job_adapter = self.job_adapter
        self.job_adapter = 'orca' # Temporarily change to 'orca' for submit script lookup
        try:
            super().write_submit_script()
        finally:
            self.job_adapter = original_job_adapter # Revert job_adapter

    def set_input_file_memory(self) -> None:
        """
        Set the input_file_memory attribute.
        """
        super().set_input_file_memory()

    def execute_incore(self):
        """
        Execute a job incore.
        """
        self.initial_time = self.initial_time if self.initial_time else datetime.datetime.now()
        which(self.command,
              return_bool=True,
              raise_error=True,
              raise_msg=f'Please install {self.job_adapter}, see {self.url} for more information.',
              )
        self._log_job_execution()
        execute_command([f'cd {self.local_path}'] + incore_commands[self.job_adapter], executable='/bin/bash')
        self.cleanup_files()
        self.final_time = datetime.datetime.now()
        self.process_run()

    def execute_queue(self):
        """
        Execute a job to the server's queue.
        """
        self.legacy_queue_execution()


register_job_adapter('orca_neb', OrcaNEBAdapter)
