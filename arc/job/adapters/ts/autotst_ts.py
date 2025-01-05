"""
An adapter for executing AutoTST jobs

https://github.com/ReactionMechanismGenerator/AutoTST
"""

import datetime
import os
import subprocess
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from arc.common import almost_equal_coords, ARC_PATH, get_logger, read_yaml_file
from arc.imports import settings
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter
from arc.job.factory import register_job_adapter
from arc.plotter import save_geo
from arc.species.converter import xyz_from_data
from arc.species.species import ARCSpecies, TSGuess, colliding_atoms

HAS_AUTOTST = True
try:
    # new format
    from autotst.reaction import Reaction as AutoTST_Reaction
except (ImportError, ModuleNotFoundError):
    try:
        # old format
        from autotst.reaction import AutoTST_Reaction
    except (ImportError, ModuleNotFoundError):
        HAS_AUTOTST = False

if TYPE_CHECKING:
    from arc.level import Level
    from arc.reaction import ARCReaction


AUTOTST_PYTHON = settings['AUTOTST_PYTHON']

logger = get_logger()


class AutoTSTAdapter(JobAdapter):
    """
    A class for executing AutoTST jobs.

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
                 level: Optional['Level'] = None,
                 max_job_time: Optional[float] = None,
                 run_multi_species: bool = False,
                 reactions: Optional[List['ARCReaction']] = None,
                 rotor_index: Optional[int] = None,
                 server: Optional[str] = None,
                 server_nodes: Optional[list] = None,
                 queue: Optional[str] = None,
                 attempted_queues: Optional[List[str]] = None,
                 species: Optional[List[ARCSpecies]] = None,
                 testing: bool = False,
                 times_rerun: int = 0,
                 torsions: Optional[List[List[int]]] = None,
                 tsg: Optional[int] = None,
                 xyz: Optional[dict] = None,
                 ):

        self.incore_capacity = 20
        self.job_adapter = 'autotst'
        self.execution_type = execution_type or 'incore'
        self.command = None  # AutoTST does not have an executable file, just an API.
        self.url = 'https://github.com/ReactionMechanismGenerator/AutoTST'
        self.supported_families = ['intra_H_migration',
                                   'H_Abstraction',
                                   'R_Addition_MultipleBond']

        if reactions is None:
            raise ValueError('Cannot execute AutoTST without ARCReaction object(s).')

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
        pass

    def set_additional_file_paths(self) -> None:
        """
        Set additional file paths specific for the adapter.
        Called from set_file_paths() and extends it.
        """
        self.output_path = os.path.join(self.local_path, 'ts_results.yml')

    def set_input_file_memory(self) -> None:
        """
        Set the input_file_memory attribute.
        """
        pass

    def execute_incore(self):
        """
        Execute a job incore.
        """
        if not HAS_AUTOTST:
            raise ModuleNotFoundError(f'Could not import AutoTST, make sure it is properly installed.\n'
                                      f'See {self.url} for more information, or use the Makefile provided with ARC.')
        self._log_job_execution()
        self.initial_time = self.initial_time if self.initial_time else datetime.datetime.now()

        self.reactions = [self.reactions] if not isinstance(self.reactions, list) else self.reactions
        for rxn in self.reactions:
            if rxn.family in self.supported_families:
                if rxn.ts_species is None:
                    # Mainly used for testing, in an ARC run the TS species should already exist.
                    rxn.ts_species = ARCSpecies(label='TS',
                                                is_ts=True,
                                                charge=rxn.charge,
                                                multiplicity=rxn.multiplicity,
                                                )
                reaction_label_fwd = get_autotst_reaction_string(rxn)
                reaction_label_rev = get_autotst_reaction_string(ARCReaction(r_species=rxn.p_species,
                                                                             p_species=rxn.r_species,
                                                                             reactants=rxn.products,
                                                                             products=rxn.reactants))

                i = 0
                for reaction_label, direction in zip([reaction_label_fwd, reaction_label_rev], ['F', 'R']):
                    # run AutoTST as a subprocess in the desired direction
                    script_path = os.path.join(ARC_PATH, 'arc', 'job', 'adapters', 'scripts', 'autotst_script.py')
                    commands = ['source ~/.bashrc', f'"{AUTOTST_PYTHON}" "{script_path}" "{reaction_label}" "{self.output_path}"']
                    command = '; '.join(commands)

                    tic = datetime.datetime.now()

                    output = subprocess.run(command, shell=True, executable='/bin/bash')

                    tok = datetime.datetime.now() - tic

                    if output.returncode:
                        direction_str = 'forward' if direction == 'F' else 'reverse'
                        logger.warning(f'AutoTST subprocess did not give a successful return code for {rxn} '
                                       f'in the {direction_str} direction.\n'
                                       f'Got return code: {output.returncode}\n'
                                       f'stdout: {output.stdout}\n'
                                       f'stderr: {output.stderr}')
                    if os.path.isfile(self.output_path):
                        results = read_yaml_file(path=self.output_path)
                        if results:
                            for result in results:
                                xyz = xyz_from_data(coords=result['coords'], numbers=result['numbers'])
                                unique = True
                                for other_tsg in rxn.ts_species.ts_guesses:
                                    if other_tsg.success and almost_equal_coords(xyz, other_tsg.initial_xyz):
                                        if 'autotst' not in other_tsg.method.lower():
                                            other_tsg.method += ' and AutoTST'
                                        unique = False
                                        break
                                if unique and not colliding_atoms(xyz):
                                    ts_guess = TSGuess(method='AutoTST',
                                                       method_direction=direction,
                                                       method_index=i,
                                                       t0=tic,
                                                       execution_time=tok,
                                                       xyz=xyz,
                                                       success=True,
                                                       index=len(rxn.ts_species.ts_guesses),
                                                       )
                                    rxn.ts_species.ts_guesses.append(ts_guess)
                                    save_geo(xyz=xyz,
                                             path=self.local_path,
                                             filename=f'AutoTST {direction}',
                                             format_='xyz',
                                             comment=f'AutoTST {direction}',
                                             )
                                    i += 1
                        else:
                            ts_guess = TSGuess(method=f'AutoTST',
                                               method_direction=direction,
                                               method_index=i,
                                               t0=tic,
                                               execution_time=tok,
                                               success=False,
                                               index=len(rxn.ts_species.ts_guesses),
                                               )
                            rxn.ts_species.ts_guesses.append(ts_guess)
                            i += 1

            if len(self.reactions) < 5:
                successes = len([tsg for tsg in rxn.ts_species.ts_guesses if tsg.success and 'autotst' in tsg.method])
                if successes:
                    logger.info(f'AutoTST successfully found {successes} TS guesses for {rxn.label}.')
                else:
                    logger.info(f'AutoTST did not find any successful TS guesses for {rxn.label}.')

        self.final_time = datetime.datetime.now()

    def execute_queue(self):
        """
        (Execute a job to the server's queue.)
        A single AutoTST job will always be executed incore.
        """
        self.execute_incore()


def get_autotst_reaction_string(rxn: ARCReaction) -> str:
    """
    Returns the AutoTST reaction string in the form of r1+r2_p1+p2 (e.g., `CCC+[O]O_[CH2]CC+OO`).

    Args:
        rxn (ARCReaction): The ARC reaction instance.

    Returns:
        str: The AutoTST reaction string.
    """
    reactants, products = rxn.get_reactants_and_products()
    if len(reactants) > 1:
        reactants_string = '+'.join([reactant.mol.copy(deep=True).to_smiles() for reactant in reactants])
    else:
        reactants_string = reactants[0].mol.copy(deep=True).to_smiles()
    if len(products) > 1:
        products_string = '+'.join([product.mol.copy(deep=True).to_smiles() for product in products])
    else:
        products_string = products[0].mol.copy(deep=True).to_smiles()
    reaction_label = '_'.join([reactants_string, products_string])
    return reaction_label


register_job_adapter('autotst', AutoTSTAdapter)
