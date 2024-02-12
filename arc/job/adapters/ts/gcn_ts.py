"""
An adapter for executing GCN (graph convolutional network) jobs

Code source:
https://github.com/ReactionMechanismGenerator/TS-GCN
Literature source:
https://chemrxiv.org/articles/Genereting_Transition_States_of_Isomerization_Reactions_with_Deep_Learning/12302084
"""

import datetime
import os
import subprocess
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from rdkit import Chem

from arc.common import ARC_PATH, almost_equal_coords, get_logger, save_yaml_file
from arc.imports import settings
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter
from arc.job.factory import register_job_adapter
from arc.plotter import save_geo
from arc.species.converter import rdkit_conf_from_mol, str_to_xyz
from arc.species.species import ARCSpecies, TSGuess, colliding_atoms

HAS_GCN = True
try:
    from inference import inference
except (ImportError, ModuleNotFoundError):
    HAS_GCN = False

if TYPE_CHECKING:
    from arc.level import Level
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies

servers, submit_filenames, TS_GCN_PYTHON = settings['servers'], settings['submit_filenames'], settings['TS_GCN_PYTHON']

DIHEDRAL_INCREMENT = 10
GCN_SCRIPT_PATH = os.path.join(ARC_PATH, 'arc', 'job', 'adapters', 'scripts', 'gcn_script.py')

logger = get_logger()


class GCNAdapter(JobAdapter):
    """
    A class for executing GCN (graph convolutional network) jobs.

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
        dihedral_increment (float, optional): Used by this adapter to determine the number of times to execute GCN
                                              in each direction (forward and reverse). Default: 10.
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
        level (Level, optionnal): The level of theory to use.
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
                 level: Optional['Level'] = None,
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

        self.incore_capacity = 100
        self.job_adapter = 'gcn'
        self.execution_type = execution_type or 'incore'
        self.command = 'inference.py'
        self.url = 'https://github.com/ReactionMechanismGenerator/TS-GCN'

        self.repetitions = int(dihedral_increment or DIHEDRAL_INCREMENT)

        if reactions is None:
            raise ValueError('Cannot execute GCN without ARCReaction object(s).')

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
        pass

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
        # 1.3. HDF5 file
        elif os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='data.hdf5'))
        # 1.4 job.sh
        job_sh_dict = self.set_job_shell_file_to_upload()  # Set optional job.sh files if relevant.
        if job_sh_dict is not None:
            self.files_to_upload.append(job_sh_dict)
        # 1.5 YAML input file
        self.files_to_upload.append(self.get_file_property_dictionary(file_name='input.yml'))
        # 2. ** Download **
        # 2.1. HDF5 file
        if self.iterate_by and os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
            self.files_to_download.append(self.get_file_property_dictionary(file_name='data.hdf5'))
        else:
            # 2.2. Results
            self.files_to_download.append(self.get_file_property_dictionary(file_name='TS_fwd.xyz'))
            self.files_to_download.append(self.get_file_property_dictionary(file_name='TS_rev.xyz'))

    def set_additional_file_paths(self) -> None:
        """
        Set additional file paths specific for the adapter.
        Called from set_file_paths() and extends it.
        """
        self.reactant_path = os.path.join(self.local_path, "reactant.sdf")
        self.product_path = os.path.join(self.local_path, "product.sdf")
        self.ts_fwd_path = os.path.join(self.local_path, "TS_fwd.xyz")
        self.ts_rev_path = os.path.join(self.local_path, "TS_rev.xyz")
        self.yml_in_path = os.path.join(self.local_path, "input.yml")
        self.yml_out_path = os.path.join(self.local_path, "output.yml")

    def set_input_file_memory(self) -> None:
        """
        Set the input_file_memory attribute.
        """
        self.cpu_cores, self.job_memory_gb = 1, 1

    def execute_incore(self):
        """
        Execute a job incore.
        """
        self.initial_time = self.initial_time if self.initial_time else datetime.datetime.now()
        self.execute_gcn(exe_type='incore')
        self.final_time = datetime.datetime.now()

    def execute_queue(self):
        """
        Execute a job to the server's queue.
        """
        self.execute_gcn(exe_type='queue')

    def execute_gcn(self, exe_type: str = 'incore'):
        """
        Execute a job either incore or to the queue.

        Args:
            exe_type (str, optional): Whether to execute 'incore' or 'queue'.
        """
        self._log_job_execution()
        rxn = self.reactions[0]
        if not rxn.is_isomerization():
            return
        if rxn.ts_species is None:
            rxn.ts_species = ARCSpecies(label=self.species_label,
                                        is_ts=True,
                                        charge=rxn.charge,
                                        multiplicity=rxn.multiplicity,
                                        )
        write_sdf_files(rxn=rxn,
                        reactant_path=self.reactant_path,
                        product_path=self.product_path,
                        )
        if exe_type == 'queue':
            input_dict = {'reactant_path': self.reactant_path,
                          'product_path': self.product_path,
                          'local_path': self.local_path,
                          'yml_out_path': self.yml_out_path,
                          'repetitions': self.repetitions,
                          }
            save_yaml_file(path=self.yml_in_path, content=input_dict)
            self.legacy_queue_execution()
        elif exe_type == 'incore':
            for _ in range(self.repetitions):
                run_subprocess_locally(direction='F',
                                       reactant_path=self.reactant_path,
                                       product_path=self.product_path,
                                       ts_path=self.ts_fwd_path,
                                       local_path=self.local_path,
                                       ts_species=rxn.ts_species,
                                       )
                run_subprocess_locally(direction='R',
                                       reactant_path=self.product_path,
                                       product_path=self.reactant_path,
                                       ts_path=self.ts_rev_path,
                                       local_path=self.local_path,
                                       ts_species=rxn.ts_species,
                                       )
            if len(self.reactions) < 5:
                successes = len([tsg for tsg in rxn.ts_species.ts_guesses if tsg.success and 'gcn' in tsg.method])
                if successes:
                    logger.info(f'GCN successfully found {successes} TS guesses for {rxn.label}.')
                else:
                    logger.info(f'GCN did not find any successful TS guesses for {rxn.label}.')


def write_sdf_files(rxn: 'ARCReaction',
                    reactant_path: str,
                    product_path: str,
                    ):
    """
    Write reactant and product SDF files using RDKit.

    Args:
        rxn (ARCReaction): The relevant reaction.
        reactant_path (str): The path to the reactant SDF file.
        product_path (str): The path to the product SDF file.
    """
    reactant_rdkit_mol = rdkit_conf_from_mol(rxn.r_species[0].mol, rxn.r_species[0].get_xyz())[1]
    mapped_product = rxn.get_single_mapped_product_xyz()
    product_rdkit_mol = rdkit_conf_from_mol(mapped_product.mol, mapped_product.get_xyz())[1]
    w = Chem.SDWriter(reactant_path)
    w.write(reactant_rdkit_mol)
    w.close()
    w = Chem.SDWriter(product_path)
    w.write(product_rdkit_mol)
    w.close()


def run_subprocess_locally(direction: str,
                           reactant_path: str,
                           product_path: str,
                           ts_path: str,
                           local_path: str,
                           ts_species: ARCSpecies,
                           ):
    """
    Run GCN incore using a subprocess.

    Args:
        direction (str): Either 'F' or 'R' for forward ort reverse directions, respectively.
        reactant_path (str): The path to the reactant SDF file.
        product_path (str): The path to the product SDF file.
        ts_path (str): The path to the resulting TS guess file.
        local_path (str): The local path to the job folder.
        ts_species (ARCSpecies): The TS ``ARCSpecies`` object instance.
    """
    ts_xyz = None
    tsg = TSGuess(method='GCN',
                  method_direction=direction,
                  index=len(ts_species.ts_guesses),
                  )
    tsg.tic()
    commands = ['source ~/.bashrc',
                f'{TS_GCN_PYTHON} {GCN_SCRIPT_PATH} '
                f'--r_sdf_path {product_path} '
                f'--p_sdf_path {reactant_path} '
                f'--ts_xyz_path {ts_path}']
    command = '; '.join(commands)
    output = subprocess.run(command, shell=True, executable='/bin/bash')
    if output.returncode:
        logger.warning(f'GCN subprocess ran in the reverse direction did not '
                       f'give a successful return code for {ts_species}.\n'
                       f'Got return code: {output.returncode}\n'
                       f'stdout: {output.stdout}\n'
                       f'stderr: {output.stderr}')
    elif os.path.isfile(ts_path):
        ts_xyz = str_to_xyz(ts_path)
    tsg.tok()
    process_tsg(direction=direction,
                ts_xyz=ts_xyz,
                local_path=local_path,
                ts_species=ts_species,
                tsg=tsg,
                )


def process_tsg(direction: str,
                ts_xyz: Optional[dict],
                local_path: str,
                ts_species: ARCSpecies,
                tsg: TSGuess,
                ):
    """
    Process a single TS guess created by GCN.

    Args:
        direction (str): Either 'F' or 'R' for forward ort reverse directions, respectively.
        ts_xyz (dict): The TS coordinates.
        local_path (str): The local path to the job folder.
        ts_species (ARCSpecies): The TS ``ARCSpecies`` object instance.
        tsg (TSGuess): The relevant ``TSGuess`` object instance.
    """
    unique = True
    if ts_xyz is not None and not colliding_atoms(ts_xyz):
        for other_tsg in ts_species.ts_guesses:
            if other_tsg.success and almost_equal_coords(ts_xyz, other_tsg.initial_xyz):
                if 'gcn' not in other_tsg.method.lower():
                    other_tsg.method += ' and GCN'
                unique = False
                break
        if unique:
            tsg.success = True
            tsg.process_xyz(ts_xyz)
            save_geo(xyz=ts_xyz,
                     path=local_path,
                     filename=f'GCN {direction} {tsg.index}',
                     format_='xyz',
                     comment=f'GCN {direction}',
                     )
    else:
        tsg.success = False
    if unique and tsg.success:
        ts_species.ts_guesses.append(tsg)


register_job_adapter('gcn', GCNAdapter)
