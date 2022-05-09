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

from arc.common import ARC_PATH, almost_equal_coords, get_logger
from arc.imports import settings
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import check_argument_consistency
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


TS_GCN_PYTHON = settings['TS_GCN_PYTHON']

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
        level (Level, optionnal): The level of theory to use.
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
        dihedral_increment (float, optional): Used by this adapter to determine the number of times to execute GCN
                                              in each direction (forward and reverse). Default: 10.
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
                 level: Optional['Level'] = None,
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
                 dihedral_increment: Optional[float] = 10,
                 ):

        self.job_adapter = 'gcn'
        self.execution_type = execution_type or 'incore'
        self.command = 'inference.py'
        self.url = 'https://github.com/ReactionMechanismGenerator/TS-GCN'

        if reactions is None:
            raise ValueError('Cannot execute AutoTST without ARCReaction object(s).')

        self.job_types = job_type if isinstance(job_type, list) else [job_type]  # always a list
        self.job_type = job_type
        self.project = project
        self.project_directory = project_directory
        if self.project_directory and not os.path.isdir(self.project_directory):
            os.makedirs(self.project_directory)
        self.args = args or dict()
        self.bath_gas = bath_gas
        self.checkfile = checkfile
        self.conformer = conformer
        self.constraints = constraints or list()
        self.cpu_cores = cpu_cores
        self.dihedrals = dihedrals
        self.ess_settings = ess_settings
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
        self.level = level
        self.max_job_time = max_job_time
        self.reactions = [reactions] if reactions is not None and not isinstance(reactions, list) else reactions
        self.rotor_index = rotor_index
        self.server = server
        self.server_nodes = server_nodes or list()
        self.species = [species] if species is not None and not isinstance(species, list) else species
        self.testing = testing
        self.torsions = torsions
        self.tsg = tsg
        self.xyz = xyz
        self.times_rerun = times_rerun
        self.repetitions = int(dihedral_increment)

        self.species_label = self.reactions[0].ts_species.label if self.reactions[0].ts_species is not None \
            else f'TS_{self.job_num}'  # The ts_species attribute should be initialized in a normal ARC run
        if len(self.reactions) > 1:
            self.species_label += f'_and_{len(self.reactions) - 1}_others'

        if self.job_num is None or self.job_name is None or self.job_server_name:
            self._set_job_number()

        self.args = dict()

        self.final_time = None
        self.run_time = None
        self.charge = None
        self.multiplicity = None
        self.is_ts = True
        self.scan_res = None
        self.set_file_paths()

        self.workers = None
        self.iterate_by = list()
        self.number_of_processes = 0
        self.incore_capacity = 100
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
        pass

    def set_additional_file_paths(self) -> None:
        """
        Set additional file paths specific for the adapter.
        Called from set_file_paths() and extends it.
        """
        self.reactant_path = os.path.join(self.local_path, "reactant.sdf")
        self.product_path = os.path.join(self.local_path, "product.sdf")
        self.ts_fwd_path = os.path.join(self.local_path, "TS_fwd.xyz")
        self.ts_rev_path = os.path.join(self.local_path, "TS_rev.xyz")

    def set_input_file_memory(self) -> None:
        """
        Set the input_file_memory attribute.
        """
        pass

    def parse_tsg(self):
        """
        Parse the coordinates from a completed tsg job.
        Store in rxn.ts_species.ts_guesses as a TSGuess object instance.
        """
        pass

    def execute_incore(self):
        """
        Execute a job incore.
        """
        if not HAS_GCN:
            raise ModuleNotFoundError(f'Could not import GCN, make sure it is properly installed.\n'
                                      f'See {self.url} for more information, or use the Makefile provided with ARC.')
        self._log_job_execution()
        self.initial_time = self.initial_time if self.initial_time else datetime.datetime.now()
        self.reactions = [self.reactions] if not isinstance(self.reactions, list) else self.reactions
        for rxn in self.reactions:

            if rxn.ts_species is None:
                # Mainly used while testing, in an ARC run the TS species should already exist at this point.
                rxn.ts_species = ARCSpecies(label=self.species_label,
                                            is_ts=True,
                                            charge=rxn.charge,
                                            multiplicity=rxn.multiplicity,
                                            )

            # Check that this is indeed an isomerization reaction, i.e., only one reactant and one product.
            num_reactants = len(rxn.r_species)
            num_products = len(rxn.p_species)

            if num_reactants > 1:
                logger.error(f'Error while using GCN with reactants: {rxn.r_species}.\n'
                             f'Isomerization reactions must have only 1 reactant.')
            if num_products > 1:
                logger.error(f'Error while using GCN with products: {rxn.p_species}.\n'
                             f'Isomerization reactions must have only 1 product.')
            if num_reactants > 1 or num_products > 1:
                return None

            # prepare run
            reactant = rxn.r_species[0]
            reactant_rdkit_mol = rdkit_conf_from_mol(reactant.mol, reactant.get_xyz())[1]

            # GCN requires an atom-mapped reaction so map the product atoms onto the reactant atoms
            mapped_product = rxn.get_single_mapped_product_xyz()
            product_rdkit_mol = rdkit_conf_from_mol(mapped_product.mol, mapped_product.get_xyz())[1]

            # write input files for GCN to the TS project folder
            w = Chem.SDWriter(self.reactant_path)
            w.write(reactant_rdkit_mol)
            w.close()

            w = Chem.SDWriter(self.product_path)
            w.write(product_rdkit_mol)
            w.close()
            script_path = os.path.join(ARC_PATH, 'arc', 'job', 'adapters', 'ts', 'scripts', 'gcn_script.py')
            command_0 = 'source ~/.bashrc'

            for _ in range(self.repetitions):
                # GCN is highly non-deterministic.

                ts_xyz_fwd, ts_xyz_rev = None, None

                # Run the GCN as a subprocess in the forward directions.
                ts_guess_f = TSGuess(method=f'GCN',
                                     method_direction='F',
                                     index=len(rxn.ts_species.ts_guesses),
                                     )
                ts_guess_f.tic()

                commands = [command_0]
                commands.append(f'{TS_GCN_PYTHON} {script_path} '
                                f'--r_sdf_path {self.reactant_path} '
                                f'--p_sdf_path {self.product_path} '
                                f'--ts_xyz_path {self.ts_fwd_path}')
                command = '; '.join(commands)
                output = subprocess.run(command, shell=True, executable='/bin/bash')
                if output.returncode:
                    logger.warning(f'GCN subprocess ran in the forward direction did not '
                                   f'give a successful return code for {rxn}.\n'
                                   f'Got return code: {output.returncode}\n'
                                   f'stdout: {output.stdout}\n'
                                   f'stderr: {output.stderr}')
                elif os.path.isfile(self.ts_fwd_path):
                    ts_xyz_fwd = str_to_xyz(self.ts_fwd_path)

                ts_guess_f.tok()

                unique = True
                if ts_xyz_fwd is not None and not colliding_atoms(ts_xyz_fwd):
                    for other_tsg in rxn.ts_species.ts_guesses:
                        if other_tsg.success and almost_equal_coords(ts_xyz_fwd, other_tsg.initial_xyz):
                            if 'gcn' not in other_tsg.method.lower():
                                other_tsg.method += ' and GCN'
                            unique = False
                            break
                    if unique:
                        ts_guess_f.success = True
                        ts_guess_f.process_xyz(ts_xyz_fwd)
                        save_geo(xyz=ts_xyz_fwd,
                                 path=self.local_path,
                                 filename=f'GCN F {ts_guess_f.index}',
                                 format_='xyz',
                                 comment='GCN F',
                                 )
                else:
                    ts_guess_f.success = False
                if unique and ts_guess_f.success:
                    rxn.ts_species.ts_guesses.append(ts_guess_f)

                # run the GCN as a subprocess in the reverse directions
                ts_guess_r = TSGuess(method=f'GCN',
                                     method_direction='R',
                                     index=len(rxn.ts_species.ts_guesses),
                                     )
                ts_guess_r.tic()

                commands = [command_0]
                commands.append(f'{TS_GCN_PYTHON} {script_path} '
                                f'--r_sdf_path {self.product_path} '
                                f'--p_sdf_path {self.reactant_path} '
                                f'--ts_xyz_path {self.ts_rev_path}')
                command = '; '.join(commands)
                output = subprocess.run(command, shell=True, executable='/bin/bash')
                if output.returncode:
                    logger.warning(f'GCN subprocess ran in the reverse direction did not '
                                   f'give a successful return code for {rxn}.\n'
                                   f'Got return code: {output.returncode}\n'
                                   f'stdout: {output.stdout}\n'
                                   f'stderr: {output.stderr}')
                elif os.path.isfile(self.ts_rev_path):
                    ts_xyz_rev = str_to_xyz(self.ts_rev_path)

                ts_guess_r.tok()

                unique = True
                if ts_xyz_rev is not None and not colliding_atoms(ts_xyz_rev):
                    for other_tsg in rxn.ts_species.ts_guesses:
                        if other_tsg.success and almost_equal_coords(ts_xyz_rev, other_tsg.initial_xyz):
                            if 'gcn' not in other_tsg.method.lower():
                                other_tsg.method += ' and GCN'
                            unique = False
                            break
                    if unique:
                        ts_guess_r.success = True
                        ts_guess_r.process_xyz(ts_xyz_rev)
                        save_geo(xyz=ts_xyz_rev,
                                 path=self.local_path,
                                 filename=f'GCN R {ts_guess_f.index}',
                                 format_='xyz',
                                 comment='GCN R',
                                 )
                else:
                    ts_guess_r.success = False
                if unique and ts_guess_r.success:
                    rxn.ts_species.ts_guesses.append(ts_guess_r)

            if len(self.reactions) < 5:
                successes = len([tsg for tsg in rxn.ts_species.ts_guesses if tsg.success and 'gcn' in tsg.method])
                if successes:
                    logger.info(f'GCN successfully found {successes} TS guesses for {rxn.label}.')
                else:
                    logger.info(f'GCN did not find any successful TS guesses for {rxn.label}.')

        self.final_time = datetime.datetime.now()

    def execute_queue(self):
        """
        (Execute a job to the server's queue.)
        A single GCN job will always be executed incore.
        """
        self.execute_incore()


register_job_adapter('gcn', GCNAdapter)
