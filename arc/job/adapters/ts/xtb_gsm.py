"""
An adapter for executing DE-GSM (Double Ended Growing String Method) jobs via xTB (Semiempirical Extended Tight-Binding Program Package)

xTB:
https://github.com/grimme-lab/xtb
https://xtb-docs.readthedocs.io/en/latest/contents.html

GSM:
https://zimmermangroup.github.io/molecularGSM/index.html
https://github.com/ZimmermanGroup/molecularGSM/wiki
https://doi.org/10.1021/ct400319w
Advanced GSM settings: https://github.com/ZimmermanGroup/molecularGSM/wiki#items-of-note-and-advanced-options
"""

import datetime
import os
import shutil
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from mako.template import Template

from arc.common import ARC_PATH, get_logger, safe_copy_file
from arc.imports import incore_commands, settings
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter
from arc.job.factory import register_job_adapter
from arc.job.local import change_mode, execute_command
from arc.level import Level
from arc.parser import parse_trajectory
from arc.species import TSGuess
from arc.species.converter import xyz_to_xyz_file_format

if TYPE_CHECKING:
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies

logger = get_logger()

input_filenames, output_filenames, servers, submit_filenames, xtb_gsm_settings = \
    settings['input_filenames'], settings['output_filenames'], settings['servers'], settings['submit_filenames'], \
    settings['xtb_gsm_settings']


input_template = """# FSM/GSM/SSM inpfileq

------------- QCHEM Scratch Info ------------------------
$QCSCRATCH/    # path for scratch dir. end with "/" 
GSM_go1q       # name of run
---------------------------------------------------------

------------ String Info --------------------------------
SM_TYPE                 ${sm_type}
RESTART                 ${restart}
MAX_OPT_ITERS           ${max_opt_iters}
STEP_OPT_ITERS          ${step_opt_iters}
CONV_TOL                ${conv_tol}
ADD_NODE_TOL	        ${add_node_tol}
SCALING                 ${scaling}
SSM_DQMAX               ${ssm_dqmax}
GROWTH_DIRECTION        ${growth_direction}
INT_THRESH              ${int_thresh}
MIN_SPACING             ${min_spacing}
BOND_FRAGMENTS          ${bond_fragments}
INITIAL_OPT             ${initial_opt}
FINAL_OPT               ${final_opt}
PRODUCT_LIMIT           ${product_limit}
TS_FINAL_TYPE           ${ts_final_type}
NNODES                  ${nnodes}
---------------------------------------------------------
"""


class xTBGSMAdapter(JobAdapter):
    """
    A class for executing DE-GSM jobs via xTB.

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
                 species: Optional[List['ARCSpecies']] = None,
                 testing: bool = False,
                 times_rerun: int = 0,
                 torsions: Optional[List[List[int]]] = None,
                 tsg: Optional[int] = None,
                 xyz: Optional[dict] = None,
                 ):

        self.incore_capacity = 1
        self.job_adapter = 'xtb_gsm'
        self.execution_type = execution_type or 'queue'
        self.command = 'xtb'
        self.url = 'https://github.com/grimme-lab/xtb'

        if reactions is None:
            raise ValueError('Cannot execute DE-GSM via xTB without an ARCReaction object.')

        _initialize_adapter(obj=self,
                            is_ts=True,
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
        r_xyz = xyz_to_xyz_file_format(self.reactions[0].get_reactants_xyz(return_format=dict))
        p_xyz = xyz_to_xyz_file_format(self.reactions[0].get_products_xyz(return_format=dict))

        scratch_path = os.path.join(self.local_path, 'scratch')
        if not os.path.isdir(scratch_path):
            os.makedirs(scratch_path)

        with open(self.scratch_initial0000_path, 'w') as f:
            f.write(f'{r_xyz}{p_xyz}')

        inpfileq_args = self.set_inpfileq_keywords()
        with open(self.inpfileq_path, 'w') as f:
            f.write(Template(input_template).render(**inpfileq_args))

        if self.execution_type == 'incore':
            safe_copy_file(source=os.path.join(self.xtb_gsm_scripts_path, 'gsm.orca'), destination=self.gsm_orca_path)
            safe_copy_file(source=os.path.join(self.xtb_gsm_scripts_path, 'ograd'), destination=self.ograd_path)
            safe_copy_file(source=os.path.join(self.xtb_gsm_scripts_path, 'tm2orca.py'), destination=self.tm2orca_path)
            change_mode(mode='+x', file_name=self.gsm_orca_path)
            change_mode(mode='+x', file_name=self.tm2orca_path)

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
        if not self.iterate_by:
            self.write_input_file()
        # 1. ** Upload **
        # 1.1. submit file
        if self.execution_type != 'incore':
            # we need a submit file for single or array jobs (either submitted to local or via SSH)
            self.write_submit_script()
            self.files_to_upload.append(self.get_file_property_dictionary(
                file_name=submit_filenames[servers[self.server]['cluster_soft']]))
            # 1.1 initial0000.xyz
            self.files_to_upload.append(self.get_file_property_dictionary(
                file_name='initial0000.xyz',
                local=os.path.join(self.local_path, 'scratch', 'initial0000.xyz'),
                remote=os.path.join(self.remote_path, 'scratch', 'initial0000.xyz')))
            # 1.2 gsm.orca
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='gsm.orca',
                                                                          make_x=True,
                                                                          local=os.path.join(self.xtb_gsm_scripts_path, 'gsm.orca')))
            # 1.3 inpfileq
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='inpfileq',
                                                                          local=os.path.join(self.xtb_gsm_scripts_path, 'inpfileq')))
            # 1.4 ograd
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='ograd',
                                                                          local=os.path.join(self.xtb_gsm_scripts_path, 'ograd')))
            # 1.5 tm2orca.py
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='tm2orca.py',
                                                                          make_x=True,
                                                                          local=os.path.join(self.xtb_gsm_scripts_path, 'tm2orca.py')))
        # 1.6 job.sh
        job_sh_dict = self.set_job_shell_file_to_upload()  # Set optional job.sh files if relevant.
        if job_sh_dict is not None:
            self.files_to_upload.append(job_sh_dict)
        # 1.7. HDF5 file
        if self.iterate_by and os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='data.hdf5'))
        # 2. ** Download **
        # 2.1. HDF5 file
        if self.iterate_by and os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
            self.files_to_download.append(self.get_file_property_dictionary(file_name='data.hdf5'))
        elif self.execution_type != 'incore':
            # 2.2. stringfile
            self.files_to_download.append(self.get_file_property_dictionary(file_name='stringfile.xyz0000'))

    def set_additional_file_paths(self) -> None:
        """
        Set additional file paths specific for the adapter.
        Called from set_file_paths() and extends it.
        """
        self.xtb_gsm_scripts_path = os.path.join(ARC_PATH, 'arc', 'job', 'adapters', 'scripts', 'xtb_gsm')
        self.gsm_orca_path = os.path.join(self.local_path, 'gsm.orca')
        self.inpfileq_path = os.path.join(self.local_path, 'inpfileq')
        self.ograd_path = os.path.join(self.local_path, 'ograd')
        self.tm2orca_path = os.path.join(self.local_path, 'tm2orca.py')
        self.scratch_initial0000_path = os.path.join(self.local_path, 'scratch', 'initial0000.xyz')
        self.stringfile_path = os.path.join(self.local_path, 'stringfile.xyz0000')

    def set_inpfileq_keywords(self) -> dict:
        """
        Set keywords for the input file ``inpfileq``.
        See: https://github.com/ZimmermanGroup/molecularGSM/wiki#items-of-note-and-advanced-options

        Keywords::
            SM_TYPE: Toggles between GSM and SSM. Use the indicated 3 letter combinations to utilize each of the chosen features.
            RESTART: Read restart.xyz.
            MAX_OPT_ITERS: Maximum iterations for each step of the GSM.
            STEP_OPT_ITERS: Maximum optimizing iterations for each step of the SSM. If starting structure is not
                            properly optimized jobs can fail if the max opt set by this variable is exceeded!
                            For FSM/SSM.
            CONV_TOL: Controls the optimization threshold for nodes. Smaller threshold increases the run time,
                      but can improve TS finding in difficult systems.
            ADD_NODE_TOL: Tolerance variable for the addition of next node in GSM. Higher numbers will afford fast growth,
                          but decrease accuracy of reaction path identification. For GSM
            SCALING: For opt steps. This feature controls how the step size is adjusted based on the topography of the
                     previous optimization step. Since the step size is the product of dqmax and the scaling variable,
                     increasing this variable will increase the step size. For more rapid adjustment based on the
                     topography of the reaction path, increase this variable. For less automatic adjustment of the step size,
                     decrease this variable. For the most part, the default setting is fine.
            SSM_DQMAX: Controls the spacing between the nodes in SSM. In cases where the RP struggles to converge or the
                       optimization of RP fails, decreasing this value might help. Add step size.
            GROWTH_DIRECTION: GSM specific toggle which enables user control of growth direction.
                              Typically the default (0) is preferred. However, this is a good toggle for debugging more
                              difficult cases. For new users think of the location of the TS; if the TS favors the product
                              then growth from the product with low tolerance convergence could provide better TS identification.
                              normal/react/prod: 0/1/2.
            INT_THRESH: Detection threshold for intermediate during string growth.
                        GSM will not consider structures that have a energy below this threshold as a TS along the RP.
            MIN_SPACING: Node spacing SSM.
            BOND_FRAGMENTS: Make IC's for fragments.
            INITIAL_OPT: Starting structure optimization performed on the first structure of the input file.
                         Opt steps first node.
            FINAL_OPT: Max number of optimization steps performed on the last node of an SSM run. Opt steps last SSM node.
            PRODUCT_LIMIT: Max energy for product exploration in kcal/mol.
            TS_FINAL_TYPE: Determines whether rotations are considered causation for termination of the GSM or SSM run.
                           Typically, we are searching for a change in bond connectivity so this value is set to 1 for delta bond.
                           0=no bond breaking, 1=breaking of bond.
            NNODES: Max number of nodes used for GSM, including endpoints. Set this number high (30) for SE-GSM as the convergence criterion
                    typically results in less nodes needed and too small a number for this setting will result in job failure.
                    For DE-GSM the typical is an odd number ranging from 9-15 with higher values being used for identifying
                    multiple TSs along the path.

        Returns:
            dict: THe updated keywords.
        """
        parameters = {'sm_type': 'GSM',
                      'restart': 0,
                      'max_opt_iters': 160,
                      'step_opt_iters': 30,
                      'conv_tol': 0.0005,
                      'add_node_tol': 0.1,
                      'scaling': 1.0,
                      'ssm_dqmax': 0.8,
                      'growth_direction': 0,
                      'int_thresh': 2.0,
                      'min_spacing': 5.0,
                      'bond_fragments': 1,
                      'initial_opt': 0,
                      'final_opt': 150,
                      'product_limit': 100.0,
                      'ts_final_type': 1,
                      'nnodes': 15,
                      }
        parameters.update(xtb_gsm_settings)
        updated_args = dict()
        for key, val in parameters.items():
            updated_args[key] = self.level.args['keyword'][key] \
                if self.level is not None and key in self.level.args['keyword'] else str(val)
        return updated_args

    def process_run(self):
        """
        Process a completed xTB-GSM run.
        """
        tsg = TSGuess(method='xTB-GSM',
                      index=len(self.reactions[0].ts_species.ts_guesses),
                      success=False,
                      t0=self.initial_time,
                      )
        if os.path.isfile(self.stringfile_path):
            traj = parse_trajectory(self.stringfile_path)
            tsg.initial_xyz = traj[int((len(traj) - 1) / 2) + 1]
            tsg.execution_time = self.final_time - self.initial_time
            tsg.success = True
        self.reactions[0].ts_species.ts_guesses.append(tsg)

    def cleanup_files(self):
        """Remove unneeded files after run."""
        for path in [self.gsm_orca_path]:
            shutil.rmtree(path, ignore_errors=True)

    def set_input_file_memory(self) -> None:
        """
        Set the input_file_memory attribute.
        """
        pass

    def execute_incore(self):
        """
        Execute a job incore.
        """
        self.initial_time = self.initial_time if self.initial_time else datetime.datetime.now()
        self._log_job_execution()
        execute_command([f'cd {self.local_path}'] + incore_commands['xtb_gsm'], executable='/bin/bash')
        self.cleanup_files()
        self.final_time = datetime.datetime.now()
        self.process_run()

    def execute_queue(self):
        """
        Execute a job to the server's queue.
        """
        self.legacy_queue_execution()


register_job_adapter('xtb_gsm', xTBGSMAdapter)
