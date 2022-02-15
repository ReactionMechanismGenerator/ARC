"""
An adapter for executing TS guess jobs based on heuristics

Todo:
    - eventually this module needs to be applied for all H's connected to the same heavy atom,
      summing up the rates appropriately
    - test H2O2 as the RH abstractor, see that both TS chiralities are attained
    - Add tests
    - add database and train from database, see https://github.com/ReactionMechanismGenerator/ARC/commit/081df5bf8e53987e9ff48eef481c17997f9cff2a, https://github.com/ReactionMechanismGenerator/ARC/commit/9a569ee80331494dcca26490fd66accc69697380, https://github.com/ReactionMechanismGenerator/ARC/commit/10d255467334f7821547e7dc5d98bb50bbfab7c0
    - use FF or TorchANI to only retain guesses with reasonable energies
    - Think: two H sites on a CH2 element, one being abstracted. On which one in the reactant do we put the abstractor? Can/should we try both?
"""

import datetime
import itertools
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

from rmgpy.exceptions import ActionError
from rmgpy.molecule.molecule import Molecule
from rmgpy.reaction import Reaction
from rmgpy.species import Species
from arkane.statmech import is_linear

from arc.common import almost_equal_coords, get_logger, key_by_val
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import check_argument_consistency, ts_adapters_by_rmg_family
from arc.job.factory import register_job_adapter
from arc.plotter import save_geo
from arc.species.converter import compare_zmats, zmat_from_xyz, zmat_to_xyz
from arc.species.mapping import map_arc_rmg_species, map_two_species
from arc.species.species import ARCSpecies, TSGuess, colliding_atoms
from arc.species.zmat import get_parameter_from_atom_indices, is_angle_linear, remove_1st_atom, up_param

if TYPE_CHECKING:
    from rmgpy.data.kinetics.family import KineticsFamily
    from arc.level import Level
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies


DIHEDRAL_INCREMENT = 30

logger = get_logger()


class HeuristicsAdapter(JobAdapter):
    """
    A class for executing TS guess jobs based on heuristics.

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

        self.job_adapter = 'heuristics'
        self.execution_type = execution_type or 'incore'

        if reactions is None:
            raise ValueError('Cannot execute TS Heuristics without ARCReaction object(s).')

        if dihedral_increment is not None and (dihedral_increment < 0 or dihedral_increment > 360):
            raise ValueError(f'dihedral_increment should be between 0 to 360, got: {dihedral_increment}')

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
        self.dihedral_increment = dihedral_increment or DIHEDRAL_INCREMENT
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

        self.tasks = None
        self.iterate_by = list()
        self.number_of_processes = 0
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

    def execute_incore(self):
        """
        Execute a job incore.
        """
        self._log_job_execution()
        self.initial_time = self.initial_time if self.initial_time else datetime.datetime.now()

        supported_families = [key for key, val in ts_adapters_by_rmg_family.items() if 'heuristics' in val]

        self.reactions = [self.reactions] if not isinstance(self.reactions, list) else self.reactions
        for rxn in self.reactions:
            family_label = rxn.family.label
            if family_label not in supported_families:
                logger.warning(f'The heuristics TS search adapter does not support the {family_label} reaction family.')
                continue
            if any(spc.get_xyz() is None for spc in rxn.r_species + rxn.p_species):
                logger.warning(f'The heuristics TS search adapter cannot process a reaction if 3D coordinates of '
                               f'some/all of its reactants/products are missing.\nNot processing {rxn}.')
                continue

            if rxn.ts_species is None:
                # Mainly used for testing, in an ARC run the TS species should already exist.
                rxn.ts_species = ARCSpecies(label='TS',
                                            is_ts=True,
                                            charge=rxn.charge,
                                            multiplicity=rxn.multiplicity,
                                            )
            rxn.arc_species_from_rmg_reaction()
            reactants, products = rxn.get_reactants_and_products(arc=True)
            reactant_mol_combinations = list(itertools.product(*list(reactant.mol_list for reactant in reactants)))
            product_mol_combinations = list(itertools.product(*list(product.mol_list for product in products)))
            reaction_list = list()
            for reactants in list(reactant_mol_combinations):
                for products in list(product_mol_combinations):
                    rxns = react(reactants=list(reactants),
                                 products=list(products),
                                 family=rxn.family,
                                 arc_reaction=rxn,
                                 )
                    if rxns is not None:
                        reaction_list.extend(rxns)

            xyzs = list()
            tsg = None
            if family_label == 'H_Abstraction':
                # Todo: train guess params
                # r1_stretch_, r2_stretch_, a2_ = get_training_params(
                #     family='H_Abstraction',
                #     atom_type_key=tuple(sorted([atom_a.atomtype.label, atom_b.atomtype.label])),
                #     atom_symbol_key=tuple(sorted([atom_a.element.symbol, atom_b.element.symbol])),
                # )
                # r1_stretch_, r2_stretch_, a2_ = 1.2, 1.2, 170  # general guesses
                tsg = TSGuess(method=f'Heuristics')
                tsg.tic()
                xyzs = h_abstraction(arc_reaction=rxn,
                                     rmg_reactions=reaction_list,
                                     dihedral_increment=self.dihedral_increment,
                                     )
                tsg.tok()

            for method_index, xyz in enumerate(xyzs):
                unique = True
                for other_tsg in rxn.ts_species.ts_guesses:
                    if almost_equal_coords(xyz, other_tsg.initial_xyz):
                        if 'heuristics' not in other_tsg.method.lower():
                            other_tsg.method += ' and Heuristics'
                        unique = False
                        break
                if unique:
                    ts_guess = TSGuess(method=f'Heuristics',
                                       index=len(rxn.ts_species.ts_guesses),
                                       method_index=method_index,
                                       t0=tsg.t0,
                                       execution_time=tsg.execution_time,
                                       success=True,
                                       family=family_label,
                                       xyz=xyz,
                                       )
                    rxn.ts_species.ts_guesses.append(ts_guess)
                    save_geo(xyz=xyz,
                             path=self.local_path,
                             filename=f'Heuristics {method_index}',
                             format_='xyz',
                             comment=f'Heuristics {method_index}, family: {family_label}',
                             )

            if len(self.reactions) < 5:
                successes = len([tsg for tsg in rxn.ts_species.ts_guesses if tsg.success and 'heuristics' in tsg.method])
                if successes:
                    logger.info(f'Heuristics successfully found {successes} TS guesses for {rxn.label}.')
                else:
                    logger.info(f'Heuristics did not find any successful TS guesses for {rxn.label}.')

        self.final_time = datetime.datetime.now()

    def execute_queue(self):
        """
        (Execute a job to the server's queue.)
        A single Heuristics job will always be executed incore.
        """
        self.execute_incore()


def combine_coordinates_with_redundant_atoms(xyz_1: Union[dict, str],
                                             xyz_2: Union[dict, str],
                                             mol_1: 'Molecule',
                                             mol_2: 'Molecule',
                                             reactant_2: ARCSpecies,
                                             h1: int,
                                             h2: int,
                                             c: Optional[int] = None,
                                             d: Optional[int] = None,
                                             r1_stretch: float = 1.2,
                                             r2_stretch: float = 1.2,
                                             a2: float = 180,
                                             d2: Optional[float] = None,
                                             d3: Optional[float] = None,
                                             keep_dummy: bool = False,
                                             reactants_reversed: bool = False,
                                             ) -> dict:
    """
    Combine two coordinates that share an atom.
    For this redundant atom case, only three additional degrees of freedom (here ``a2``, ``d2``, and ``d3``)
    are required.

    Atom scheme (dummy atom X will be added if the A-H-B angle is close to 180 degrees)::

                    X           D
                    |         /
            A -- H1 - H2 -- B
          /
        C
        |--- mol1 --|-- mol2 ---|

    zmats will be constructed in the following way::

        zmat1 = {'symbols': ('C', 'H', 'H', 'H', 'H'),
                 'coords': ((None, None, None),  # 0, atom A
                            ('R_1_0', None, None),  # 1, atom C
                            ('R_2_1', 'A_2_1_0', None),  # 2
                            ('R_3_2', 'A_3_2_0', 'D_3_2_0_1'),  # 3
                            ('R_4_3' * r1_stretch, 'A_4_3_2', 'D_4_3_2_1')),  # 4, H1
                 'vars': {...},
                 'map': {...}}

        zmat2 = {'symbols': ('H', 'H', 'H', 'H', 'C'),
                 'coords': ((None, None, None),  # H2, redundant H atom, will be united with H1
                            ('R_5_4' * r2_stretch, a2 (B-H-A) = 'A_5_4_0', d2 (B-H-A-C) = 'D_5_4_0_1'),  # 5, atom B
                            ('R_6_5', 'A_6_5_4', d3 (D-B-H-A) = 'D_6_5_4_0'),  # 6, atom D
                            ('R_7_6', 'A_7_6_4', 'D_7_6_4_5'),  # 7
                            ('R_8_7', 'A_8_7_6', 'D_8_7_6_5')),  # 8
                 'vars': {...},
                 'map': {...}}

    Args:
        xyz_1 (Union[dict, str]): The Cartesian coordinates of ``mol_1`` (including the redundant atom).
        xyz_2 (Union[dict, str]): The Cartesian coordinates of ``mol_2`` (including the redundant atom).
        mol_1 (Molecule): The RMG Molecule instance corresponding to ``xyz1``.
        mol_2 (Molecule): The RMG Molecule instance corresponding to ``xyz2``.
        reactant_2 (ARCSpecies): The other reactant, R(*3) for H_Abstraction, that is not use for generating
                                 the TS coordinates using heuristics. Will be used for atom mapping.
        h1 (int): The 0-index of a terminal redundant H atom in ``xyz1`` (atom H1).
        h2 (int): The 0-index of a terminal redundant H atom in ``xyz2`` (atom H2).
        c (Optional[int]): The 0-index of an atom in ``xyz1`` connected to either A or H1 which is neither A nor H1
                           (atom C).
        d (Optional[int]): The 0-index of an atom in ``xyz2`` connected to either B or H2 which is neither B nor H2
                           (atom D).
        r1_stretch (float, optional): The factor by which to multiply (stretch/shrink) the bond length to the terminal
                                      atom ``h1`` in ``xyz1`` (bond A-H1).
        r2_stretch (float, optional): The factor by which to multiply (stretch/shrink) the bond length to the terminal
                                      atom ``h2`` in ``xyz2`` (bond B-H2).
        a2 (float, optional): The angle (in degrees) in the combined structure between atoms B-H-A (angle B-H-A).
        d2 (Optional[float]): The dihedral angle (in degrees) between atoms B-H-A-C (dihedral B-H-A-C).
                              This argument must be given only if the a2 angle is not linear,
                              and mol2 has 3 or more atoms, otherwise it is meaningless.
        d3 (Optional[float]): The dihedral angel (in degrees) between atoms D-B-H-A (dihedral D-B-H-A).
                              This parameter is mandatory only if atom D exists (i.e., if ``mol2`` has 3 or more atoms)
                              and angle a2 is not linear.
        keep_dummy (bool, optional): Whether to keep a dummy atom if added, ``True`` to keep, ``False`` by default.
        reactants_reversed (bool, optional): Whether the reactants were reversed relative to the RMG template.

    Returns:
        dict: The combined Cartesian coordinates.

    Todo:
        - Accept xyzs of the radicals as well as E0's of all species, average xyz of atoms by energy similarity
          before returning the final cartesian coordinates
    """
    is_a2_linear = is_angle_linear(a2)
    is_mol_1_linear = is_linear(np.array(xyz_1['coords']))
    d2 = d2 if not is_a2_linear else None
    num_atoms_mol_1, num_atoms_mol_2 = len(mol_1.atoms), len(mol_2.atoms)

    if num_atoms_mol_1 == 1 or num_atoms_mol_2 == 1:
        raise ValueError(
            f'The molecule arguments to combine_coordinates_with_redundant_atoms must each have more than 1 '
            f'atom (including the abstracted hydrogen atom in each), got {len(mol_1.atoms)} atoms in mol_1 '
            f'and {len(mol_2.atoms)} atoms in mol_2.')
    if d2 is None and not is_a2_linear and num_atoms_mol_1 > 2:
        raise ValueError('The d2 parameter (the B-H-A-C dihedral) must be given if the a2 angle (B-H-A) is not close '
                         'to 180 degrees, got None.')
    if d3 is None and not is_a2_linear and num_atoms_mol_2 > 2:
        raise ValueError('The d3 parameter (the A-H-B-D dihedral) must be given if the a2 angle (B-H-A) is not linear '
                         'and mol_2 has 3 or more atoms, got None.')
    if c is None and num_atoms_mol_1 > 2:
        raise ValueError('The c parameter (the index of atom C in xyz1) must be given if mol_1 has 3 or more atoms, '
                         'got None.')
    if d is None and num_atoms_mol_2 > 2:
        raise ValueError('The d parameter (the index of atom D in xyz2) must be given if mol_2 has 3 or more atoms, '
                         'got None.')
    if d3 is None and d is not None and num_atoms_mol_1 > 2 and not is_mol_1_linear:
        raise ValueError('The d3 parameter (dihedral D-B-H-A) must be given if mol_1 has 3 or more atoms, got None.')

    a = mol_1.atoms.index(list(mol_1.atoms[h1].edges.keys())[0])
    b = mol_2.atoms.index(list(mol_2.atoms[h2].edges.keys())[0])
    if c == a:
        raise ValueError(f'The value for c ({c}) is invalid (it represents atom A, not atom C)')
    if d == b:
        raise ValueError(f'The value for d ({d}) is invalid (it represents atom B, not atom D)')

    zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_1, xyz_2, mol_1, mol_2, h1, h2, a, b, c, d)

    # Stretch the A--H1 and B--H2 bonds.
    stretch_zmat_bond(zmat=zmat_1, indices=(h1, a), stretch=r1_stretch)
    stretch_zmat_bond(zmat=zmat_2, indices=(b, h2), stretch=r2_stretch)

    add_dummy = is_a2_linear and len(zmat_1['symbols']) > 2 and not is_mol_1_linear
    glue_params = determine_glue_params(zmat=zmat_1,
                                        add_dummy=add_dummy,
                                        a=a,
                                        h1=h1,
                                        c=c,
                                        d=d,
                                        )
    new_symbols, new_coords, new_vars, new_map = get_modified_params_from_zmat_2(zmat_1=zmat_1,
                                                                                 zmat_2=zmat_2,
                                                                                 reactant_2=reactant_2,
                                                                                 add_dummy=add_dummy,
                                                                                 glue_params=glue_params,
                                                                                 a=a,
                                                                                 h1=h1,
                                                                                 c=c,
                                                                                 a2=a2,
                                                                                 d2=d2,
                                                                                 d3=d3,
                                                                                 reactants_reversed=reactants_reversed,
                                                                                 )
    combined_zmat = {'symbols': new_symbols, 'coords': new_coords, 'vars': new_vars, 'map': new_map}
    for i, coords in enumerate(combined_zmat['coords']):
        if i > 2 and None in coords:
            raise ValueError(f'Could not combine zmats, got a None parameter beyond the 3rd row:\n'
                             f'{coords} in:\n{combined_zmat}')

    ts_xyz = zmat_to_xyz(zmat=combined_zmat, keep_dummy=keep_dummy)
    return ts_xyz


def generate_the_two_constrained_zmats(xyz_1: dict,
                                       xyz_2: dict,
                                       mol_1: 'Molecule',
                                       mol_2: 'Molecule',
                                       h1: int,
                                       h2: int,
                                       a: int,
                                       b: int,
                                       c: Optional[int],
                                       d: Optional[int],
                                       ) -> Tuple[dict, dict]:
    """
    Generate the two constrained zmats required for combining coordinates with a redundant atom.

    Args:
        xyz_1 (dict): The Cartesian coordinates of ``mol_1`` (including the redundant atom).
        xyz_2 (dict): The Cartesian coordinates of ``mol_2`` (including the redundant atom).
        mol_1 (Molecule): The RMG Molecule instance corresponding to ``xyz_1``.
        mol_2 (Molecule): The RMG Molecule instance corresponding to ``xyz_2``.
        h1 (int): The 0-index of a terminal redundant H atom in ``xyz_1`` (atom H1).
        h2 (int): The 0-index of a terminal redundant H atom in ``xyz_2`` (atom H2).
        a (int): The 0-index of an atom in ``xyz_1`` connected to H1 (atom A).
        b (int): The 0-index of an atom in ``xyz_2`` connected to H2 (atom B).
        c (Optional[int]): The 0-index of an atom in ``xyz_1`` connected to either A or H1 which is neither A nor H1 (atom C).
        d (Optional[int]): The 0-index of an atom in ``xyz_2`` connected to either B or H2 which is neither B nor H2 (atom D).

    Returns:
        Tuple[dict, dict]: The two zmats.
    """
    zmat1 = zmat_from_xyz(xyz=xyz_1,
                          mol=mol_1,
                          is_ts=True,
                          constraints={'A_group': [(h1, a, c)]} if c is not None else {'R_atom': [(h1, a)]},
                          consolidate=False,
                          )
    zmat2 = zmat_from_xyz(xyz=xyz_2,
                          mol=mol_2,
                          is_ts=True,
                          constraints={'A_group': [(d, b, h2)]} if d is not None else {'R_group': [(b, h2)]},
                          consolidate=False,
                          )
    return zmat1, zmat2


def stretch_zmat_bond(zmat: dict,
                      indices: Tuple[int, int],
                      stretch: float):
    """
    Stretch a bond in a zmat.

    Args:
        zmat: The zmat to process.
        indices (tuple): A length 2 tuple with the 0-indices of the xyz (not zmat) atoms representing the bond to stretch.
        stretch (float): The factor by which to multiply (stretch/shrink) the bond length.
    """
    param = get_parameter_from_atom_indices(zmat=zmat, indices=indices, xyz_indexed=True)
    zmat['vars'][param] *= stretch


def determine_glue_params(zmat: dict,
                          add_dummy: bool,
                          h1: int,
                          a: int,
                          c: Optional[int],
                          d: Optional[int],
                          ) -> Tuple[str, str, str]:
    """
    Determine glue parameters ``a2``, ``d2``, and ``d3`` for combining two zmats.
    Modifies the ``zmat`` argument if a dummy atom needs to be added.

    Args:
        zmat (dict): The first zmat onto which the second zmat will be glued.
        add_dummy (bool): Whether to add a dummy atom.
        h1 (int): The 0-index of atom H1 (in mol_1).
        a (int): The 0-index of atom A (in mol_1).
        c (Optional[int]): The 0-index of atom C (in mol_1).
        d (Optional[int]): The 0-index of atom D (in mol_2).

    Returns:
        Tuple[str, str, str]: The a2, d2, and d3 zmat glue parameters.
    """
    num_atoms_1 = len(zmat['symbols'])  # The number of atoms in zmat1, used to increment the atom indices in zmat2.
    # zh = num_atoms_1 - 1  # The atom index of H in the combined zmat.
    za = key_by_val(zmat['map'], a)  # The atom index of A in the combined zmat.
    h1 = key_by_val(zmat['map'], h1)  # The atom index of H1 in the combined zmat.
    zb = num_atoms_1 + int(add_dummy)  # The atom index of B in the combined zmat, if a2=180, consider the dummy atom.
    zc = key_by_val(zmat['map'], c) if c is not None else None
    zd = num_atoms_1 + 1 + int(add_dummy) if d is not None else None  # The atom index of D in the combined zmat.
    if add_dummy:
        # Add a dummy atom.
        zmat['map'][len(zmat['symbols'])] = f"X{len(zmat['symbols'])}"
        zx = num_atoms_1
        zmat['symbols'] = tuple(list(zmat['symbols']) + ['X'])
        r_str = f'RX_{zx}_{h1}'
        a_str = f'AX_{zx}_{h1}_{za}'
        d_str = f'DX_{zx}_{h1}_{za}_{zc}' if zc is not None else None  # X-H-A-C
        zmat['coords'] = tuple(list(zmat['coords']) + [(r_str, a_str, d_str)])  # The coords of the dummy atom.
        zmat['vars'][r_str] = 1.0
        zmat['vars'][a_str] = 90.0
        if d_str is not None:
            zmat['vars'][d_str] = 0
        param_a2 = f'A_{zb}_{h1}_{za}'  # B-H-A
        param_d2 = f'D_{zb}_{h1}_{zx}_{za}'  # B-H-X-A
        param_d3 = f'D_{zd}_{zb}_{za}_{zc if c is not None else zx}' if d is not None else None  # D-B-A-C/X
    else:
        param_a2 = f'A_{zb}_{h1}_{za}'  # B-H-A
        param_d2 = f'D_{zb}_{h1}_{za}_{zc}' if zc is not None else None  # B-H-A-C
        param_d3 = f'D_{zd}_{zb}_{h1}_{za}' if d is not None else None  # D-B-H-A
    return param_a2, param_d2, param_d3


def get_modified_params_from_zmat_2(zmat_1: dict,
                                    zmat_2: dict,
                                    reactant_2: ARCSpecies,
                                    add_dummy: bool,
                                    glue_params: Tuple[str, str, str],
                                    h1: int,
                                    a: int,
                                    c: Optional[int],
                                    a2: float,
                                    d2: Optional[float],
                                    d3: Optional[float],
                                    reactants_reversed: bool = False,
                                    ) -> Tuple[tuple, tuple, dict, dict]:
    """
    Generate a modified zmat2 (in parts):
    Remove the first atom, change the indices of all existing parameter, and add "glue" parameters.

    Args:
        zmat_1 (dict): The zmat describing R1H.
        zmat_2 (dict): The zmat describing R2H.
        reactant_2 (ARCSpecies): The other reactant, R(*3) for H_Abstraction, that is not use for generating
                                 the TS using heuristics. Will be used for atom mapping.
        add_dummy (bool): Whether to add a dummy atom.
        glue_params (Tuple[str, str, str]): param_a2, param_d2, param_d3.
        h1 (int): The 0-index of atom H1 (in mol_1).
        a (int): The 0-index of atom A (in mol_1).
        c (int): The 0-index of an atom in ``xyz1`` connected to either A or H1 which is neither A nor H1 (atom C).
        a2 (float): The angle (in degrees) in the combined structure between atoms B-H-A (angle B-H-A).
        d2 (float): The dihedral angle (in degrees) between atoms B-H-A-C (dihedral B-H-A-C).
        d3 (float): The dihedral angel (in degrees) between atoms D-B-H-A (dihedral D-B-H-A).
        reactants_reversed (bool, optional): Whether the reactants were reversed relative to the RMG template.

    Returns:
        Tuple[tuple, tuple, dict, dict]: new_symbols, new_coords, new_vars, new_map.
    """
    # Remove the redundant H from zmat_2, it's the first atom. No need for further sorting, the zmat map will do that.
    new_symbols = tuple(zmat_1['symbols'] + zmat_2['symbols'][1:])

    new_coords, new_vars = list(), dict()
    param_a2, param_d2, param_d3 = glue_params
    num_atoms_1 = len(zmat_1['symbols'])
    for i, coords in enumerate(zmat_2['coords'][1:]):
        new_coord = list()
        for j, param in enumerate(coords):
            if param is not None:
                if i == 0:
                    # Atom B should refer to H1.
                    new_param = f'R_{num_atoms_1}_{h1}'
                elif i == 1 and j == 0:
                    # Atom D should refer to atom B.
                    new_param = f'R_{num_atoms_1 + 1}_{num_atoms_1}'
                elif i == 1 and j == 1:
                    new_param = f'A_{num_atoms_1 + 1}_{num_atoms_1}_{a}'
                else:
                    new_param = up_param(param=param, increment=num_atoms_1 - 1)
                new_coord.append(new_param)
                new_vars[new_param] = zmat_2['vars'][param]  # Keep the original parameter R/A/D values.
            else:
                if i == 0 and j == 1:
                    # This is a2.
                    new_coord.append(param_a2)
                    new_vars[param_a2] = a2 + 90 if add_dummy else a2
                elif i == 0 and j == 2 and c is not None:
                    # This is d2.
                    new_coord.append(param_d2)
                    new_vars[param_d2] = d2 or 0 if not add_dummy else 0
                elif i == 1 and j == 2 and param_d3 is not None:
                    # This is d3.
                    if param_d3 is not None:
                        new_coord.append(param_d3)
                        new_vars[param_d3] = d3 or 0
                    else:
                        new_coord.append(None)
                else:
                    new_coord.append(None)
        new_coords.append(tuple(new_coord))
    new_map = get_new_zmat_2_map(zmat_1=zmat_1,
                                 zmat_2=zmat_2,
                                 reactant_2=reactant_2,
                                 reactants_reversed=reactants_reversed,
                                 )
    new_coords = tuple(list(zmat_1['coords']) + new_coords)
    new_vars = {**zmat_1['vars'], **new_vars}
    return new_symbols, new_coords, new_vars, new_map


def get_new_zmat_2_map(zmat_1: dict,
                       zmat_2: dict,
                       reactant_2: Optional[ARCSpecies],
                       reactants_reversed: bool = False,
                       ) -> Dict[int, Union[int, str]]:
    """
    Get the map of the combined zmat.

    Args:
        zmat_1 (dict): The zmat describing R1H. Contains a dummy atom at the end if a2 is linear.
        zmat_2 (dict): The zmat describing R2H.
        reactant_2 (ARCSpecies): The other reactant, R(*3) for H_Abstraction, that is not use for generating
                                 the TS using heuristics. Will be used for atom mapping.
        reactants_reversed (bool, optional): Whether the reactants were reversed relative to the RMG template.

    Returns:
        Dict[int, Union[int, str]]: The combined zmat map element.
    """
    new_map = dict()
    num_atoms_1, num_atoms_2 = len(zmat_1['symbols']), len(zmat_2['symbols']) - 1  # Redundant H in zmat_2.

    # 1. Add zmat_1's map.
    val_inc = num_atoms_2 if reactants_reversed else 0
    for key, val in zmat_1['map'].items():
        if isinstance(val, str) and 'X' in val:
            new_map[key] = f'X{int(val[1:]) + val_inc}'
        else:
            new_map[key] = val + val_inc

    # 2. Add zmat_2's map. Use the reaction atom_map, remember that dummy atoms are not considered in the atom_map.
    zmat_2_mod = remove_1st_atom(zmat_2)
    spc_from_zmat_2 = ARCSpecies(label='spc_from_zmat_2', xyz=zmat_2_mod)
    atom_map = map_two_species(spc_1=spc_from_zmat_2, spc_2=reactant_2, consider_chirality=False)
    dummy_2_indices = [int(val[1:]) for val in zmat_2['map'].values() if isinstance(val, str) and 'X' in val]
    key_inc = num_atoms_1
    val_inc = 0 if reactants_reversed else num_atoms_1
    for i, (key, val) in enumerate(zmat_2_mod['map'].items()):
        # Atoms in zmat_2 always come after atoms in zmat_1 in the new zmat, regardless of the reactants/products
        # order on each side of the given reaction.
        new_key = key + key_inc
        # Use the atom_map to map atoms in zmat_2 (i.e., values in zmat_2's 'map') to atoms in the R(*3)
        # reactant (at least for H-Abstraction reactions), since zmat_2 was built based on atoms in the R(*3)-H(*2)
        # **product** (at least for H-Abstraction reactions).
        if isinstance(val, str) and 'X' in val:
            # A dummy atom is not in the atom_map, look for the preceding atom and add 1.
            dummy_index = int(val[1:])
            new_val = atom_map[dummy_index - 1] + 1 + val_inc \
                      + len([dummy_2_index for dummy_2_index in dummy_2_indices if dummy_index > dummy_2_index])
            new_val = f'X{new_val}'
        else:
            new_val = atom_map[val] + val_inc \
                      + len([dummy_2_index for dummy_2_index in dummy_2_indices if val > dummy_2_index])
        new_map[new_key] = new_val
    if len(list(new_map.values())) != len(set(new_map.values())):
        raise ValueError(f'Could not generate a combined zmat map with no repeating values.\n{new_map}')
    return new_map


def react(reactants: List[Union[Molecule, Species]],
          products: List[Union[Molecule, Species]],
          family: 'KineticsFamily',
          arc_reaction: 'ARCReaction',
          ) -> Optional[List[Reaction]]:
    """
    React molecules to give the requested products via an RMG family,
    resulting in a reaction with RMG's atom labels for the reactants and products.

    Args:
        reactants (List['Molecule']): Entries are Molecule instances of the reaction reactants.
        products (List['Molecule']): Entries are Molecule instances of the reaction products.
        family (KineticsFamily): The RMG reaction family instance.
        arc_reaction (ARCReaction): The corresponding ARCReaction object instance.

    Returns:
        Optional[Reaction]: An RMG Reaction instance with atom-labeled reactants and products.
    """
    # Assure Molecule object instances:
    reactant_mols, product_mols = list(), list()
    for reactant in reactants:
        reactant_mols.append(reactant.copy(deep=True) if isinstance(reactant, Molecule) else reactant.molecule[0].copy(deep=True))
    for product in products:
        product_mols.append(product.copy(deep=True) if isinstance(product, Molecule) else product.molecule[0].copy(deep=True))
    reactants_copy, products_copy = [r.copy(deep=True) for r in reactant_mols], [p.copy(deep=True) for p in product_mols]

    try:
        reactions = family.generate_reactions(reactants=reactants_copy,
                                              products=products_copy,
                                              prod_resonance=False,
                                              delete_labels=False,
                                              relabel_atoms=False,
                                              )
    except (ActionError, ValueError):
        return None
    for reaction in reactions:
        try:
            family.add_atom_labels_for_reaction(reaction=reaction,
                                                output_with_resonance=False,
                                                save_order=True,
                                                )
        except (ActionError, ValueError):
            continue

    for i in range(len(reactions)):
        r_map, p_map = map_arc_rmg_species(arc_reaction=arc_reaction, rmg_reaction=reactions[i], concatenate=False)
        ordered_rmg_reactants = [reactions[i].reactants[r_map[j]] for j in range(len(reactions[i].reactants))]
        ordered_rmg_products = [reactions[i].products[p_map[j]] for j in range(len(reactions[i].products))]
        reactions[i].reactants = ordered_rmg_reactants
        reactions[i].products = ordered_rmg_products

    # Re-map all molecules since atoms in the RMG products were shuffled (products re-created from the reactants).
    output, label_dicts = list(), list()
    for reaction in reactions:
        for index in range(len(reaction.reactants)):
            # The RMG molecule will get a random 3D conformer, don't consider chirality when mapping.
            atom_map = map_two_species(spc_1=reactant_mols[index],
                                       spc_2=reaction.reactants[index],
                                       consider_chirality=False,
                                       )
            if atom_map is None:
                break
            new_atoms_list = list()
            for i in range(len(reactant_mols[index].atoms)):
                reactant_mols[index].atoms[i].id = reaction.reactants[index].molecule[0].atoms[atom_map[i]].id
                reactant_mols[index].atoms[i].label = reaction.reactants[index].molecule[0].atoms[atom_map[i]].label
                new_atoms_list.append(reactant_mols[index].atoms[i])
            reaction.reactants[index].molecule[0].atoms = new_atoms_list
        else:
            for index in range(len(reaction.products)):
                # The RMG molecule will get a random 3D conformer, don't consider chirality when mapping.
                atom_map = map_two_species(spc_1=product_mols[index],
                                           spc_2=reaction.products[index],
                                           consider_chirality=False,
                                           )
                if atom_map is None:
                    break
                new_atoms_list = list()
                for i in range(len(product_mols[index].atoms)):
                    product_mols[index].atoms[i].id = reaction.products[index].molecule[0].atoms[atom_map[i]].id
                    product_mols[index].atoms[i].label = reaction.products[index].molecule[0].atoms[atom_map[i]].label
                    new_atoms_list.append(product_mols[index].atoms[i])
                reaction.products[index].molecule[0].atoms = new_atoms_list
            else:
                if reaction is not None:
                    # Check that the RMG reaction atom labels are unique.
                    label_dict = dict()
                    for reactant, product in zip(reaction.reactants, reaction.products):
                        for s, spc in enumerate([reactant, product]):
                            for i, atom in enumerate(spc.molecule[0].atoms):
                                if atom.label:
                                    label_dict[f'{"P" if s else "R"}_{atom.label}'] = i
                    if label_dict not in label_dicts:
                        label_dicts.append(label_dict)
                        output.append(reaction)
    return output


def find_distant_neighbor(rmg_mol: 'Molecule',
                          start: int,
                          ) -> Optional[int]:
    """
    Find the 0-index of a distant neighbor (2 steps away) if possible from the starting atom.
    Preferably, a heavy atom will be returned.

    Args:
        rmg_mol ('Molecule'): The RMG molecule object instance to explore.
        start (int): The 0-index of the start atom.

    Returns:
        Optional[int]: The 0-index of the distant neighbor.
    """
    if len(rmg_mol.atoms) <= 2:
        return None
    distant_neighbor_h_index = None
    for neighbor in rmg_mol.atoms[start].edges.keys():
        for distant_neighbor in neighbor.edges.keys():
            distant_neighbor_index = rmg_mol.atoms.index(distant_neighbor)
            if distant_neighbor_index != start:
                if distant_neighbor.is_hydrogen():
                    distant_neighbor_h_index = distant_neighbor_index
                else:
                    return distant_neighbor_index
    return distant_neighbor_h_index


# Family-specific heuristics functions:


def h_abstraction(arc_reaction: 'ARCReaction',
                  rmg_reactions: List['Reaction'],
                  r1_stretch: float = 1.2,
                  r2_stretch: float = 1.2,
                  a2: float = 180,
                  dihedral_increment: int = 20,
                  ) -> List[dict]:
    """
    Generate TS guesses for reactions of the RMG ``H_Abstraction`` family.

    Args:
        arc_reaction: An ARCReaction instance.
        rmg_reactions: Entries are RMGReaction instances. The reactants and products attributes should not contain
                       resonance structures as only the first molecule is consider - pass several Reaction entries
                       instead. Atoms must be labeled according to the RMG reaction family.
        r1_stretch (float, optional): The factor by which to multiply (stretch/shrink) the bond length to the terminal
                                      atom ``h1`` in ``xyz1`` (bond A-H1) relative to the respective well.
        r2_stretch (float, optional): The factor by which to multiply (stretch/shrink) the bond length to the terminal
                                      atom ``h2`` in ``xyz2`` (bond B-H2) relative to the respective well.
        a2 (float, optional): The angle (in degrees) in the combined structure between atoms B-H-A (angle B-H-A).
        dihedral_increment (int, optional): The dihedral increment to use for B-H-A-C and D-B-H-C dihedral scans.

    Returns: List[dict]
        Entries are Cartesian coordinates of TS guesses for all reactions.
    """
    if not len(rmg_reactions):
        raise ValueError('Cannot generate TS guesses without an RMG Reaction object instance.')

    xyz_guesses = list()

    # Identify R1H and R2H in the "R1H + R2 <=> R1 + R2H" or "R2 + R1H <=> R2H + R1" reaction
    # using the first RMG reaction; all other RMG reactions and the ARC reaction should have the same order.
    # The expected RMG atom labels are: R(*1)-H(*2) + R(*3)j <=> R(*1)j + R(*3)-H(*2).
    reactants_reversed, products_reversed = False, False
    for atom in rmg_reactions[0].reactants[1].molecule[0].atoms:
        if atom.label == '*2':
            reactants_reversed = True
            break
    for atom in rmg_reactions[0].products[0].molecule[0].atoms:
        if atom.label == '*2':
            products_reversed = True
            break

    arc_reactant = arc_reaction.r_species[int(reactants_reversed)]  # Get R(*1)-H(*2).
    arc_product = arc_reaction.p_species[int(not products_reversed)]  # Get R(*3)-H(*2).

    if any([is_linear(coordinates=np.array(arc_reactant.get_xyz()['coords'])),
            is_linear(coordinates=np.array(arc_product.get_xyz()['coords']))]) and is_angle_linear(a2):
        # Don't modify dihedrals for an attacking H (or other linear radical) at a linear angle, C ~ A -- H1 - H2 -- H.
        dihedral_increment = 360

    for rmg_reaction in rmg_reactions:
        rmg_reactant_mol = rmg_reaction.reactants[int(reactants_reversed)].molecule[0]
        rmg_product_mol = rmg_reaction.products[int(not products_reversed)].molecule[0]
        h1 = rmg_reactant_mol.atoms.index([atom for atom in rmg_reactant_mol.atoms if atom.label == '*2'][0])
        h2 = rmg_product_mol.atoms.index([atom for atom in rmg_product_mol.atoms if atom.label == '*2'][0])

        c = find_distant_neighbor(rmg_mol=rmg_reactant_mol, start=h1)
        d = find_distant_neighbor(rmg_mol=rmg_product_mol, start=h2)

        # d2 describes the B-H-A-C dihedral, populate d2_values if C exists and the B-H-A angle (a2) is not linear.
        d2_values = list(range(0, 360, dihedral_increment)) if len(rmg_reactant_mol.atoms) > 2 \
            and not is_angle_linear(a2) else list()

        # d3 describes the D-B-H-A dihedral, populate d3_values if D exists.
        d3_values = list(range(0, 360, dihedral_increment)) if len(rmg_product_mol.atoms) > 2 else list()

        if len(d2_values) and len(d3_values):
            d2_d3_product = list(itertools.product(d2_values, d3_values))
        elif len(d2_values):
            d2_d3_product = [(d2, None) for d2 in d2_values]
        elif len(d3_values):
            d2_d3_product = [(None, d3) for d3 in d3_values]
        else:
            d2_d3_product = [(None, None)]

        zmats = list()
        for d2, d3 in d2_d3_product:
            xyz_guess = None
            try:
                xyz_guess = combine_coordinates_with_redundant_atoms(
                    xyz_1=arc_reactant.get_xyz(),
                    xyz_2=arc_product.get_xyz(),
                    mol_1=rmg_reactant_mol,
                    mol_2=rmg_product_mol,
                    reactant_2=arc_reaction.r_species[int(not reactants_reversed)],
                    h1=h1,
                    h2=h2,
                    c=c,
                    d=d,
                    r1_stretch=r1_stretch,
                    r2_stretch=r2_stretch,
                    a2=a2,
                    d2=d2,
                    d3=d3,
                    reactants_reversed=reactants_reversed,
                )
            except ValueError as e:
                logger.error(f'Could not generate a guess using Heuristics for H abstraction reaction, got:\n{e}')

            if xyz_guess is not None and not colliding_atoms(xyz_guess):
                zmat_guess = zmat_from_xyz(xyz_guess, is_ts=True)
                for existing_zmat_guess in zmats:
                    if compare_zmats(existing_zmat_guess, zmat_guess):
                        break
                else:
                    # This TS is unique, and has no atom collisions.
                    zmats.append(zmat_guess)
                    xyz_guesses.append(xyz_guess)

    return xyz_guesses


register_job_adapter('heuristics', HeuristicsAdapter)
