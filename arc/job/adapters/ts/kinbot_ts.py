"""
An adapter for executing KinBot jobs

https://github.com/zadorlab/KinBot
"""

import datetime
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from arc.common import almost_equal_coords, get_logger
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import check_argument_consistency
from arc.job.factory import register_job_adapter
from arc.plotter import save_geo
from arc.species.converter import xyz_from_data, xyz_to_kinbot_list
from arc.species.species import ARCSpecies, TSGuess, colliding_atoms

HAS_KINBOT = True
try:
    from kinbot.modify_geom import modify_coordinates
    from kinbot.reaction_finder import ReactionFinder
    from kinbot.reaction_generator import ReactionGenerator
    from kinbot.parameters import Parameters
    from kinbot.qc import QuantumChemistry
    from kinbot.stationary_pt import StationaryPoint
except (ImportError, ModuleNotFoundError):
    HAS_KINBOT = False

if TYPE_CHECKING:
    from rmgpy.molecule import Molecule
    from arc.level import Level
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies

logger = get_logger()


if not HAS_KINBOT:
    # Create a dummy class to properly compile this module if KinBot is missing.
    class ReactionGenerator(object):
        def __init__(self, species, par, qc, input_file):
            self.species, self.par, self.qc, self.input_file = species, par, qc, input_file

class KinBotAdapter(JobAdapter):
    """
    A class for executing KinBot jobs.

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
                 dihedral_increment: Optional[float] = None,
                 ):

        self.job_adapter = 'kinbot'
        self.execution_type = execution_type or 'incore'
        self.command = None  # KinBot does not have an executable file, just an API.
        self.url = 'https://github.com/zadorlab/KinBot'
        self.family_map = {'1+2_Cycloaddition': ['r12_cycloaddition'],
                           '1,2_Insertion_CO': ['r12_insertion_R'],
                           '1,2_Insertion_carbene': ['r12_insertion_R'],
                           '1,2_shiftS': ['12_shift_S_F', '12_shift_S_R'],
                           '1,3_Insertion_CO2': ['r13_insertion_CO2'],
                           '1,3_Insertion_ROR': ['r13_insertion_ROR'],
                           '1,3_Insertion_RSR': ['r13_insertion_RSR'],
                           '2+2_cycloaddition_CCO': ['r22_cycloaddition'],
                           '2+2_cycloaddition_CO': ['r22_cycloaddition'],
                           '2+2_cycloaddition_CS': ['r22_cycloaddition'],
                           '2+2_cycloaddition_Cd': ['r22_cycloaddition'],
                           'Cyclic_Ether_Formation': ['Cyclic_Ether_Formation'],
                           'Diels_alder_addition': ['Diels_alder_addition'],
                           'HO2_Elimination_from_PeroxyRadical': ['HO2_Elimination_from_PeroxyRadical'],
                           'Intra_Diels_alder_monocyclic': ['Intra_Diels_alder_R'],
                           'Intra_ene_reaction': ['cpd_H_migration'],
                           'intra_H_migration': ['intra_H_migration', 'intra_H_migration_suprafacial'],
                           'intra_OH_migration': ['intra_OH_migration'],
                           'Intra_R_Add_Endocyclic': ['Intra_R_Add_Endocyclic_F'],
                           'Intra_R_Add_Exocyclic': ['Intra_R_Add_Exocyclic_F'],
                           'Intra_R_Add_ExoTetCyclic': ['Intra_R_Add_ExoTetCyclic_F'],
                           'Intra_Retro_Diels_alder_bicyclic': ['Intra_Diels_alder_R'],  # not sure if these fit together
                           'Intra_RH_Add_Endocyclic': ['Intra_RH_Add_Endocyclic_F', 'Intra_RH_Add_Endocyclic_R'],
                           'Intra_RH_Add_Exocyclic': ['Intra_RH_Add_Exocyclic_F', 'Intra_RH_Add_Exocyclic_R'],
                           'ketoenol': ['ketoenol'],
                           'Korcek_step2': ['Korcek_step2'],
                           'R_Addition_COm': ['R_Addition_COm3_R'],
                           'R_Addition_CSm': ['R_Addition_CSm_R'],
                           'R_Addition_MultipleBond': ['R_Addition_MultipleBond'],
                           'Retroene': ['Retro_Ene'],
                           # '?': ['intra_R_migration'],  # unknown
                           }
        self.supported_families = list(self.family_map.keys())

        if reactions is None:
            raise ValueError('Cannot execute KinBot without ARCReaction object(s).')

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
        self.incore_capacity = 20
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
        pass

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
        if not HAS_KINBOT:
            raise ModuleNotFoundError(f'Could not import KinBot, make sure it is properly installed.\n'
                                      f'See {self.url} for more information, or use the Makefile provided with ARC.')

        self._log_job_execution()
        self.initial_time = self.initial_time if self.initial_time else datetime.datetime.now()

        self.reactions = [self.reactions] if not isinstance(self.reactions, list) else self.reactions
        for rxn in self.reactions:
            if rxn.family.label in self.supported_families:
                if rxn.ts_species is None:
                    # Mainly used for testing, in an ARC run the TS species should already exist.
                    rxn.ts_species = ARCSpecies(label='TS',
                                                is_ts=True,
                                                charge=rxn.charge,
                                                multiplicity=rxn.multiplicity,
                                                )
                species_to_explore = dict()
                if len(rxn.r_species) == 1:
                    species_to_explore['F'] = rxn.r_species[0]
                if len(rxn.p_species) == 1:
                    species_to_explore['R'] = rxn.p_species[0]

                if not species_to_explore:
                    logger.error(f'Cannot execute KinBot for a non-unimolecular reaction.\n'
                                 f'Got {len(rxn.r_species)} reactants and {rxn.p_species} products in\n{rxn}.')
                    continue

                method_index = 0
                for method_direction, spc in species_to_explore.items():
                    symbols = spc.get_xyz()['symbols']
                    for m, mol in enumerate(spc.mol_list):
                        reaction_generator = set_up_kinbot(mol=mol,
                                                           families=self.family_map[rxn.family.label],
                                                           kinbot_xyz=xyz_to_kinbot_list(spc.get_xyz()),
                                                           multiplicity=rxn.multiplicity,
                                                           charge=rxn.charge,
                                                           )
                        for r, kinbot_rxn in enumerate(reaction_generator.species.reac_obj):
                            step, fix, change, release = kinbot_rxn.get_constraints(step=20,
                                                                                    geom=kinbot_rxn.species.geom)
                            ts_guess = TSGuess(method=f'KinBot',
                                               method_direction=method_direction,
                                               method_index=method_index,
                                               index=len(rxn.ts_species.ts_guesses),
                                               )
                            ts_guess.tic()

                            change_starting_zero = list()
                            for c in change:
                                c_new = [ci - 1 for ci in c[:-1]]
                                c_new.append(c[-1])
                                change_starting_zero.append(c_new)

                            success, coords = modify_coordinates(species=kinbot_rxn.species,
                                                                 name=kinbot_rxn.instance_name,
                                                                 geom=kinbot_rxn.species.geom,
                                                                 changes=change_starting_zero,
                                                                 bond=kinbot_rxn.species.bond,
                                                                 )

                            ts_guess.tok()
                            unique = True

                            if success:
                                ts_guess.success = True
                                xyz = xyz_from_data(coords=coords, symbols=symbols)
                                if xyz is None or colliding_atoms(xyz):
                                    success = False
                                else:
                                    for other_tsg in rxn.ts_species.ts_guesses:
                                        if other_tsg.success and almost_equal_coords(xyz, other_tsg.initial_xyz):
                                            if 'kinbot' not in other_tsg.method.lower():
                                                other_tsg.method += ' and KinBot'
                                            unique = False
                                            break
                                    if unique:
                                        ts_guess.process_xyz(xyz)
                                        save_geo(xyz=xyz,
                                                 path=self.local_path,
                                                 filename=f'KinBot {method_direction} {method_index}',
                                                 format_='xyz',
                                                 comment=f'KinBot {method_direction} {method_index}'
                                                 )
                            if not success:
                                ts_guess.success = False
                            if unique:
                                rxn.ts_species.ts_guesses.append(ts_guess)
                                method_index += 1

            if len(self.reactions) < 5:
                successes = len([tsg for tsg in rxn.ts_species.ts_guesses if tsg.success and 'kinbot' in tsg.method])
                if successes:
                    logger.info(f'KinBot successfully found {successes} TS guesses for {rxn.label}.')
                else:
                    logger.info(f'KinBot did not find any successful TS guesses for {rxn.label}.')

        self.final_time = datetime.datetime.now()

    def execute_queue(self):
        """
        (Execute a job to the server's queue.)
        A single KinBot job will always be executed incore.
        """
        self.execute_incore()


def set_up_kinbot(mol: 'Molecule',
                  families: List[str],
                  kinbot_xyz: List[Union[str, float]],
                  multiplicity: int,
                  charge: int,
                  ) -> ReactionGenerator:
    """
    This will set up KinBot to run for a unimolecular reaction starting from the single reactant side.

    Args:
        mol (Molecule): The RMG Molecule instance representing the unimolecular well to react.
        families (List[str]): The specific KinBot families to try.
        kinbot_xyz (list): The cartesian coordinates of the well in the KinBot list format.
        multiplicity (int): The well/reaction multiplicity.
        charge (int): The well/reaction charge.

    Returns:
        ReactionGenerator: The KinBot ReactionGenerator instance.
    """
    params = Parameters()
    params.par['title'] = 'ARC'
    # molecule information
    params.par['smiles'] = mol.to_smiles()
    params.par['structure'] = kinbot_xyz
    params.par['charge'] = charge
    params.par['mult'] = multiplicity
    params.par['dimer'] = 0
    # steps
    params.par['reaction_search'] = 1
    params.par['families'] = families
    params.par['homolytic_scissions'] = 0
    params.par['pes'] = 0
    params.par['high_level'] = 0
    params.par['conformer_search'] = 0
    params.par['me'] = 0
    #     params.par['one_reaction_fam'] = 1
    params.par['ringrange'] = [3, 9]

    well = StationaryPoint(name='well0',
                           charge=charge,
                           mult=multiplicity,
                           structure=kinbot_xyz,
                           )

    well.calc_chemid()
    well.bond_mx()
    well.find_cycle()
    well.find_atom_eqv()
    well.find_conf_dihedral()

    qc = QuantumChemistry(params.par)
    rxn_finder = ReactionFinder(well, params.par, qc)
    rxn_finder.find_reactions()

    reaction_generator = ReactionGenerator(species=well,
                                           par=params.par,
                                           qc=qc,
                                           input_file=None,
                                           )

    return reaction_generator


register_job_adapter('kinbot', KinBotAdapter)
