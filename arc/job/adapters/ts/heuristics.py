"""
An adapter for executing TS guess jobs based on heuristics

Todo:
    - eventually this module needs to be applied for all H's connected to the same heavy atom,
      summing up the rates appropriately
    - test H2O2 as the RH abstractor, see that both TS chiralities are attained
    - Add tests
    - add database and train from database, see
      https://github.com/ReactionMechanismGenerator/ARC/commit/081df5bf8e53987e9ff48eef481c17997f9cff2a,
      https://github.com/ReactionMechanismGenerator/ARC/commit/9a569ee80331494dcca26490fd66accc69697380,
      https://github.com/ReactionMechanismGenerator/ARC/commit/10d255467334f7821547e7dc5d98bb50bbfab7c0
    - use FF or TorchANI to only retain guesses with reasonable energies
    - Think: two H sites on a CH2 element, one being abstracted. On which one in the reactant do we put the abstractor?
      Can/should we try both?
"""

import datetime
import itertools
import copy
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

from rmgpy.molecule.molecule import Molecule
from arkane.statmech import is_linear

from arc.common import ARC_PATH, almost_equal_coords, get_logger, is_angle_linear, key_by_val, read_yaml_file
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter, ts_adapters_by_rmg_family
from arc.job.factory import register_job_adapter
from arc.plotter import save_geo
from arc.species.converter import compare_zmats, relocate_zmat_dummy_atoms_to_the_end, zmat_from_xyz, zmat_to_xyz , add_atom_to_xyz_using_internal_coords
from arc.mapping.engine import map_two_species
from arc.species.species import ARCSpecies, TSGuess, SpeciesError, colliding_atoms
from arc.species.zmat import get_parameter_from_atom_indices, remove_1st_atom, up_param, xyz_to_zmat
from arc.family.family import  get_reaction_family_products

if TYPE_CHECKING:
    from arc.level import Level
    from arc.reaction import ARCReaction


FAMILY_SETS = {'set_1': ['ester_hydrolysis', 'ether_hydrolysis', 'imine_hydrolysis'],
               'set_2': ['nitrile_hydrolysis']}

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
                 species: Optional[List[ARCSpecies]] = None,
                 testing: bool = False,
                 times_rerun: int = 0,
                 torsions: Optional[List[List[int]]] = None,
                 tsg: Optional[int] = None,
                 xyz: Optional[dict] = None,
                 ):

        self.incore_capacity = 50
        self.job_adapter = 'heuristics'
        self.command = None
        self.execution_type = execution_type or 'incore'

        if reactions is None:
            raise ValueError('Cannot execute TS Heuristics without ARCReaction object(s).')

        if dihedral_increment is not None and (dihedral_increment < 0 or dihedral_increment > 360):
            raise ValueError(f'dihedral_increment should be between 0 to 360, got: {dihedral_increment}')
        dihedral_increment = dihedral_increment or DIHEDRAL_INCREMENT

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
            if rxn.family not in supported_families:
                logger.warning(f'The heuristics TS search adapter does not support the {rxn.family} reaction family.')
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

            xyzs = list()
            tsg = None
            if rxn.family == 'H_Abstraction':
                tsg = TSGuess(method='Heuristics')
                tsg.tic()
                xyzs = h_abstraction(reaction=rxn, dihedral_increment=self.dihedral_increment)
                tsg.tok()

            if rxn.family in FAMILY_SETS['set_1'] or rxn.family in FAMILY_SETS['set_2']:
                tsg = TSGuess(method='Heuristics')
                tsg.tic()
                xyzs = hydrolysis(reaction=rxn)
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
                    ts_guess = TSGuess(method='Heuristics',
                                       index=len(rxn.ts_species.ts_guesses),
                                       method_index=method_index,
                                       t0=tsg.t0,
                                       execution_time=tsg.execution_time,
                                       success=True,
                                       family=rxn.family,
                                       xyz=xyz,
                                       )
                    rxn.ts_species.ts_guesses.append(ts_guess)
                    save_geo(xyz=xyz,
                             path=self.local_path,
                             filename=f'Heuristics_{method_index}',
                             format_='xyz',
                             comment=f'Heuristics {method_index}, family: {rxn.family}',
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
                    new_coord.append(param_d3)
                    new_vars[param_d3] = d3 or 0
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
    Get the map of the combined zmat ignoring the redundant H in ``zmat_2``.

    R1H           +           R2           <=>           P1           +           P2H
    zmat_1                                                                        zmat_2
    remains                  mapped                                               (H is redundant)
    (bumped                  to
    indices                  P2H
    if reversed)

    Args:
        zmat_1 (dict): The zmat describing R1H. Contains a dummy atom at the end if a2 is linear.
        zmat_2 (dict): The zmat describing R2H.
        reactant_2 (ARCSpecies): The other reactant, R(*3) for H_Abstraction, that is not used for generating
                                 the TS using heuristics. Will be used for atom mapping.
        reactants_reversed (bool, optional): Whether the reactants were reversed relative to the RMG template.

    Returns:
        Dict[int, Union[int, str]]: The combined zmat map element.
    """
    new_map = get_new_map_based_on_zmat_1(zmat_1=zmat_1, zmat_2=zmat_2, reactants_reversed=reactants_reversed)
    zmat_2_mod = remove_1st_atom(zmat_2)
    zmat_2_mod['map'] = relocate_zmat_dummy_atoms_to_the_end(zmat_2_mod['map'])
    spc_from_zmat_2 = ARCSpecies(label='spc_from_zmat_2', xyz=zmat_2_mod)
    atom_map = map_two_species(spc_1=spc_from_zmat_2, spc_2=reactant_2, consider_chirality=False)
    new_map = update_new_map_based_on_zmat_2(new_map=new_map,
                                             zmat_2=zmat_2_mod,
                                             num_atoms_1=len(zmat_1['symbols']),
                                             atom_map=atom_map,
                                             reactants_reversed=reactants_reversed,
                                             )
    if len(list(new_map.values())) != len(set(new_map.values())):
        raise ValueError(f'Could not generate a combined zmat map with no repeating values.\n{new_map}')
    return new_map


def get_new_map_based_on_zmat_1(zmat_1: dict,
                                zmat_2: dict,
                                reactants_reversed: bool = False,
                                ) -> dict:
    """
    Generate an initial map for the combined zmats, here only consider ``zmat_1``.

    Args:
        zmat_1 (dict): The zmat describing R1H. Contains a dummy atom at the end if a2 is linear.
        zmat_2 (dict): The zmat describing R2H.
        reactants_reversed (bool, optional): Whether the reactants were reversed relative to the RMG template.

    Returns:
        dict: The initial map for the combined zmats.
    """
    new_map = dict()
    val_inc = len(zmat_2['symbols']) - 1 if reactants_reversed else 0
    for key, val in zmat_1['map'].items():
        if isinstance(val, str) and 'X' in val:
            new_map[key] = f'X{int(val[1:]) + val_inc}'
        else:
            new_map[key] = val + val_inc
    return new_map


def update_new_map_based_on_zmat_2(new_map: dict,
                                   zmat_2: dict,
                                   num_atoms_1,
                                   atom_map: dict,
                                   reactants_reversed: bool = False,
                                   ):
    """
    Update the map for the combined zmats, here only consider the modified version of ``zmat_2``.
    This function assumes that all dummy atoms are located at the end of the respective Cartesian coordinates
    for ``zmat_2`` (i.e., that relocate_zmat_dummy_atoms_to_the_end() was called).

    Args:
        new_map (dict): The initial map for the combined zmats based on ``zmat_1``.
        zmat_2 (dict): The modified ``zmat_2`` (with the 1st atom removed and all dummy atoms relocated to the end).
        num_atoms_1 (int): The number of atoms in ``zmat_1``.
        atom_map (dict): The atom-map relating the product that corresponds with ``zmat_2`` to the reactant molecule.
        reactants_reversed (bool, optional): Whether the reactants were reversed relative to the RMG template.

    Returns:
        dict: The updated map for the combined zmats.
    """
    if atom_map is None:
        raise ValueError('Could not generate a combined zmat map without an atom_map.')
    key_inc = num_atoms_1
    val_inc = 0 if reactants_reversed else num_atoms_1
    dummy_atom_counter = 0
    num_of_non_x_atoms_in_zmat_2 = len([val for val in zmat_2['map'].values() if isinstance(val, int)])
    for key, val in zmat_2['map'].items():
        # Atoms in zmat_2 always come after atoms in zmat_1 in the new zmat, regardless of the reactants/products
        # order on each side of the given reaction.
        new_key = key + key_inc
        # Use the atom_map to map atoms in zmat_2 (i.e., values in zmat_2's 'map') to atoms in the R(*3)
        # **reactant** (at least for H-Abstraction reactions), since zmat_2 was built based on atoms in the R(*3)-H(*2)
        # **product** (at least for H-Abstraction reactions).
        if isinstance(val, str) and 'X' in val:
            # A dummy atom is not in the atom_map.
            new_val = num_of_non_x_atoms_in_zmat_2 + val_inc + dummy_atom_counter
            new_val = f'X{new_val}'
            dummy_atom_counter += 1
        else:
            new_val = atom_map[val] + val_inc
        new_map[new_key] = new_val
    return new_map


def find_distant_neighbor(mol: 'Molecule',
                          start: int,
                          ) -> Optional[int]:
    """
    Find the 0-index of a distant neighbor (2 steps away) if possible from the starting atom.
    Preferably, a heavy atom will be returned.

    Args:
        mol ('Molecule'): The RMG molecule object instance to explore.
        start (int): The 0-index of the start atom.

    Returns:
        Optional[int]: The 0-index of the distant neighbor.
    """
    if len(mol.atoms) <= 2:
        return None
    distant_neighbor_h_index = None
    for neighbor in mol.atoms[start].edges.keys():
        for distant_neighbor in neighbor.edges.keys():
            distant_neighbor_index = mol.atoms.index(distant_neighbor)
            if distant_neighbor_index != start:
                if distant_neighbor.is_hydrogen():
                    distant_neighbor_h_index = distant_neighbor_index
                else:
                    return distant_neighbor_index
    return distant_neighbor_h_index


# Family-specific heuristics functions:


def are_h_abs_wells_reversed(rxn: 'ARCReaction',
                             product_dict: dict,
                             ) -> Tuple[bool, bool]:
    """
    Determine whether the reactants or the products in an H_Abstraction reaction are reversed
    relative to the RMG template: R(*1)-H(*2) + R(*3)j <=> R(*1)j + R(*3)-H(*2)

    Args:
        rxn (ARCReaction): The ARCReaction object.
        product_dict (dict): The product dictionary.

    Returns:
        Tuple[bool, bool]: reactants_reversed, products_reversed.
    """
    r_star_2 = product_dict['r_label_map']['*2']
    p_star_2 = product_dict['p_label_map']['*2']
    r_species, p_species = rxn.get_reactants_and_products(arc=True, return_copies=True)
    reactants_reversed = len(r_species[0].mol.atoms) < r_star_2
    products_reversed = len(product_dict['products'][0].atoms) >= p_star_2
    same_order_between_rxn_prods_and_dict_prods = p_species[0].is_isomorphic(product_dict['products'][0])
    products_reversed = products_reversed == same_order_between_rxn_prods_and_dict_prods
    return reactants_reversed, products_reversed


def h_abstraction(reaction: 'ARCReaction',
                  r1_stretch: float = 1.2,
                  r2_stretch: float = 1.2,
                  a2: float = 180,
                  dihedral_increment: Optional[int] = None,
                  ) -> List[dict]:
    """
    Generate TS guesses for reactions of the RMG ``H_Abstraction`` family.

    Args:
        reaction: An ARCReaction instance.
        r1_stretch (float, optional): The factor by which to multiply (stretch/shrink) the bond length to the terminal
                                      atom ``h1`` in ``xyz1`` (bond A-H1) relative to the respective well.
        r2_stretch (float, optional): The factor by which to multiply (stretch/shrink) the bond length to the terminal
                                      atom ``h2`` in ``xyz2`` (bond B-H2) relative to the respective well.
        a2 (float, optional): The angle (in degrees) in the combined structure between atoms B-H-A (angle B-H-A).
        dihedral_increment (int, optional): The dihedral increment to use for B-H-A-C and D-B-H-C dihedral scans.

    Returns: List[dict]
        Entries are Cartesian coordinates of TS guesses for all reactions.
    """
    xyz_guesses = list()
    dihedral_increment = dihedral_increment or DIHEDRAL_INCREMENT
    product_dicts = get_reaction_family_products(rxn=reaction,
                                                 rmg_family_set=[reaction.family],
                                                 consider_rmg_families=True,
                                                 consider_arc_families=False,
                                                 discover_own_reverse_rxns_in_reverse=False,
                                                 )

    reactants_reversed, products_reversed = are_h_abs_wells_reversed(rxn=reaction, product_dict=product_dicts[0])
    for product_dict in product_dicts:
        # Identify R1H and R2H in the "R1H + R2 <=> R1 + R2H" or "R2 + R1H <=> R2H + R1" reaction
        # The expected RMG atom labels are: R(*1)-H(*2) + R(*3)j <=> R(*1)j + R(*3)-H(*2).
        # They appear in each product_dict under the 'r_label_map' key.
        reactants, products = reaction.get_reactants_and_products(arc=True, return_copies=False)
        reactant = reactants[int(reactants_reversed)]  # Get R(*1)-H(*2).
        reactant_2 = reactants[int(not reactants_reversed)]  # Get R(*3)j.
        product = products[int(not products_reversed)]  # Get R(*3)-H(*2).
        r_mol, p_mol = reactant.mol.copy(deep=True), product.mol.copy(deep=True)
        if any([is_linear(coordinates=np.array(reactant.get_xyz()['coords'])),
                is_linear(coordinates=np.array(product.get_xyz()['coords']))]) and is_angle_linear(a2):
            # Don't modify dihedrals for an attacking H (or other linear radical) at a linear angle, C ~ A -- H1 - H2 -- H.
            dihedral_increment = 360

        h1 = product_dict['r_label_map']['*2']
        if reactants_reversed:
            h1 -= len(reactants[0].mol.atoms)
        h2 = product_dict['p_label_map']['*2']
        dict_prods_in_same_order_as_rxn_prods = products[0].is_isomorphic(product_dict['products'][0])
        if products_reversed != dict_prods_in_same_order_as_rxn_prods:
            h2 -= len(product_dict['products'][0].atoms)
        dict_prod_index = 1 if dict_prods_in_same_order_as_rxn_prods != products_reversed else 0  # index of R(*3)-H(*2) in product_dict['products']
        product_atom_map = map_two_species(spc_1=product_dict['products'][dict_prod_index], spc_2=product)
        h2 = product_atom_map[h2]

        c = find_distant_neighbor(mol=r_mol, start=h1)
        d = find_distant_neighbor(mol=p_mol, start=h2)

        # d2 describes the B-H-A-C dihedral, populate d2_values if C exists and the B-H-A angle (a2) is not linear.
        d2_values = list(range(0, 360, dihedral_increment)) if len(r_mol.atoms) > 2 \
            and not is_angle_linear(a2) else list()

        # d3 describes the D-B-H-A dihedral, populate d3_values if D exists.
        d3_values = list(range(0, 360, dihedral_increment)) if len(p_mol.atoms) > 2 else list()

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
                    xyz_1=reactant.get_xyz(),
                    xyz_2=product.get_xyz(),
                    mol_1=r_mol,
                    mol_2=p_mol,
                    reactant_2=reactant_2,
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
            except (ValueError, SpeciesError) as e:
                logger.debug(f'Could not generate a guess using Heuristics for H abstraction reaction, got:\n{e}')

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


def hydrolysis(reaction: 'ARCReaction') -> Tuple[List[dict], List[dict]]:
    """
    Generate TS guesses for reactions of the ARC "hydrolysis" families.

    Args:
        reaction: An ARCReaction instance.

    Returns:
        Tuple containing:
            - List[dict]: Cartesian coordinates of TS guesses
            - List[dict]: Z-matrices of TS guesses
    """
    xyz_guesses_total, zmats_total = [], []
    product_dicts, ester_and_ether_families = get_products_and_check_families(reaction)
    hydrolysis_parameters = load_hydrolysis_parameters()
    dihedrals_to_change_num = 0

    while not xyz_guesses_total or (ester_and_ether_families and not has_ester_hydrolysis(xyz_guesses_total)):
        dihedrals_to_change_num += 1
        for product_dict in product_dicts:
            reaction_family = product_dict["family"]
            is_set_1 = reaction_family in hydrolysis_parameters["family_sets"]["set_1"]
            is_set_2 = reaction_family in hydrolysis_parameters["family_sets"]["set_2"]

            main_reactant, water, initial_xyz, xyz_indices = extract_reactant_and_indices(reaction,
                                                                                          product_dict,
                                                                                          is_set_1)
            base_xyz_indices = {
                "a": xyz_indices["a"],
                "b": xyz_indices["b"],
                "f": xyz_indices["f"],
                "o": xyz_indices["o"],
                "h1": xyz_indices["h1"],
            }
            chosen_xyz_indices, xyz_guesses, zmats_total= process_chosen_d_indices(
                initial_xyz, base_xyz_indices, xyz_indices, hydrolysis_parameters,
                reaction_family, water, zmats_total, is_set_1, is_set_2,dihedrals_to_change_num)

            if xyz_guesses:
                xyz_guesses_total.append({
                    "family": reaction_family,
                    "indices": list(chosen_xyz_indices),
                    "xyz_guesses": xyz_guesses
                })

    return xyz_guesses_total, zmats_total

def get_products_and_check_families(reaction: 'ARCReaction') -> Tuple[List[dict], bool]:
    """
    Get all reaction products and determine if both ester and ether hydrolysis families are present.

    Args:
        reaction: An ARCReaction instance.

    Returns:
        Tuple containing:
            - List[dict]: Product dictionaries with reaction family information
            - bool: True if both ester and ether hydrolysis families are present
    """
    product_dicts = get_reaction_family_products(
        rxn=reaction,
        rmg_family_set="default",
        consider_rmg_families=False,
        consider_arc_families=True,
    )
    ester_present = any(
        "ester_hydrolysis" in (d.get("family", []) if isinstance(d.get("family"), list) else [d.get("family")])
        for d in product_dicts
    )
    ether_present = any(
        "ether_hydrolysis" in (d.get("family", []) if isinstance(d.get("family"), list) else [d.get("family")])
        for d in product_dicts
    )

    return product_dicts, (ester_present and ether_present)

def load_hydrolysis_parameters() -> dict:
    """
    Load parameters for hydrolysis reactions from the YAML configuration file.

    Returns:
        dict: Hydrolysis parameters loaded from the configuration file containing
              family sets and specific parameters for different reaction types.
    """
    return read_yaml_file(os.path.join(ARC_PATH, "data", "hydrolysis_families_parameters.yml"))

def has_ester_hydrolysis(xyz_guesses_total: List[dict]) -> bool:
    """
    Check if ester hydrolysis is present in the generated transition state guesses.

    Args:
        xyz_guesses_total: List of dictionaries containing transition state guesses
                          and their family information.

    Returns:
        bool: True if ester hydrolysis is present in any of the transition state guesses.
    """
    return any(item["family"] == "ester_hydrolysis" for item in xyz_guesses_total)

def extract_reactant_and_indices(reaction: 'ARCReaction',
                                 product_dict: dict,
                                 is_set_1: bool) -> Tuple[ARCSpecies, ARCSpecies, dict, dict]:
    """
    Extract the reactant molecules and relevant atomic indices (a,b,f,d,o,h1) for the hydrolysis reaction.

    Args:
        reaction: An ARCReaction instance.
        product_dict: Dictionary containing reaction product information and atom mappings.
        is_set_1: Whether the reaction is in the first set of hydrolysis families.

    Returns:
        Tuple containing:
            - main_reactant(ARCSpecies): The main reactant molecule
            - water(ARCSpecies): The water molecule
            - initial_xyz(dict): Initial XYZ coordinates of the main reactant
            - xyz_indices(dict): Dictionary mapping a, b, f, d , o, h1 atoms to their indices in the xyz dictionary
    """
    main_reactant, water = process_hydrolysis_reaction(reaction)
    a_xyz_index = product_dict["r_label_map"]["*1"]
    b_xyz_index = product_dict["r_label_map"]["*2"]
    two_neighbors = is_set_1
    f_xyz_index, d_xyz_indices = get_neighbors_by_electronegativity(
        main_reactant,
        a_xyz_index,
        b_xyz_index,
        two_neighbors
    )
    o_index = len(main_reactant.mol.atoms)
    h1_index = o_index + 1

    initial_xyz = main_reactant.get_xyz()
    xyz_indices = {
        "a": a_xyz_index,
        "b": b_xyz_index,
        "f": f_xyz_index,
        "d": d_xyz_indices,
        "o": o_index,
        "h1": h1_index
    }

    return main_reactant, water, initial_xyz, xyz_indices

def process_chosen_d_indices(initial_xyz: dict,
                             base_xyz_indices: dict,
                             xyz_indices: dict,
                             hydrolysis_parameters: dict,
                             reaction_family: str,
                             water: 'ARCSpecies',
                             zmats_total: List[dict],
                             is_set_1: bool,
                             is_set_2: bool,
                             dihedrals_to_change_num: int
                             ) -> Tuple[dict[str, int], List[dict[str, Any]], List[dict[str, Any]]]:
    """
    Iterates over the 'd' indices to process TS guess generation.

    Args:
        initial_xyz (dict): Initial Cartesian coordinates.
        base_xyz_indices (Dict[str, int]): Base indices for TS generation.
        xyz_indices (Dict[str, List[int]]): All relevant indices including 'd'.
        hydrolysis_parameters (Dict[str, Any]): Hydrolysis-specific parameters.
        reaction_family (str): The reaction family.
        water ('ARCSpecies'): Water molecule info.
        zmats_total (List[Dict[str, Any]]): List to accumulate Z-matrices.
        is_set_1 (bool): Flag indicating if reaction_family is in set 1.
        is_set_2 (bool): Flag indicating if reaction_family is in set 2.
        dihedrals_to_change_num (int): The current iteration for adjusting dihedrals.

    Returns:
        Tuple[Dict[str, int], List[Dict[str, Any]], List[Dict[str, Any]]]:
            - Chosen indices for TS generation.
            - List of generated transition state (TS) guesses.
            - Updated list of Z-matrices.
    """
    for d_index in xyz_indices.get("d", []):
        chosen_xyz_indices = {**base_xyz_indices, "d": d_index}
        current_zmat, zmat_indices = setup_zmat_indices(initial_xyz, chosen_xyz_indices)

        stretch_ab_bond(current_zmat, chosen_xyz_indices, zmat_indices,hydrolysis_parameters, reaction_family)
        if not adjust_dihedral_angles(current_zmat, zmat_indices, dihedrals_to_change_num):
            continue

        ts_guesses, zmats_total = process_family_specific_adjustments(is_set_1, is_set_2, reaction_family, hydrolysis_parameters,
                                                                      current_zmat, water, chosen_xyz_indices, zmats_total
                                                                      )

        if ts_guesses:
            return chosen_xyz_indices, ts_guesses, zmats_total
    return {}, [], zmats_total

def process_hydrolysis_reaction(reaction: 'ARCReaction') -> Tuple[ARCSpecies, ARCSpecies]:
    """
    Process a hydrolysis reaction by recognizing main reactant and water species.

    Args:
        reaction: An ARCReaction instance.

    Returns:
        Tuple containing:
            - ARCSpecies: The main reactant
            - ARCSpecies: The water molecule

    Raises:
        ValueError: If reactants don't include both water and a non-water molecule.
    """
    arc_reactants, _ = reaction.get_reactants_and_products(arc=True, return_copies=True)
    arc_reactant, water = None, None

    for spc in arc_reactants:
        if not is_water(spc):
            arc_reactant = spc
        else:
            water = spc

    if arc_reactant is None or water is None:
        raise ValueError("Reactants must include a non-water molecule and water.")

    return arc_reactant, water


def is_water(spc: 'ARCSpecies') -> bool:
    """
    Check if a given species is a water molecule.

    Args:
        spc: The species to check.

    Returns:
        bool: True if the species is water (H2O), False otherwise.
    """
    if len(spc.mol.atoms) != 3:
        return False
    O_counter, H_counter = 0, 0
    for atom in spc.mol.atoms:
        if atom.is_oxygen():
            O_counter += 1
        if atom.is_hydrogen():
            H_counter += 1
    return O_counter == 1 and H_counter == 2

def get_neighbors_by_electronegativity(spc: 'ARCSpecies',
                                       atom_index: int,
                                       exclude_index: int,
                                       two_neighbors: bool = True) -> Tuple[int, List[int]]:
    """
    Retrieve the top two neighbors of a given atom in a species, sorted by their effective electronegativity,
    excluding a specified neighbor.
    Effective electronegativity is calculated as:
        Effective Electronegativity = Electronegativity of the neighbor * Bond order.
    Sorting rules:
    1. Neighbors are sorted in descending order of their effective electronegativity.
    2. If two neighbors have the same effective electronegativity, the tie is broken by comparing the
       sum of the effective electronegativities of their own bonded neighbors.
       The neighbor with the higher sum will be ranked first.

    Args:
        spc (ARCSpecies): The species containing the atoms.
        atom_index (int): Index of the central atom.
        exclude_index (int): Index of atom to exclude from neighbors list.
        two_neighbors (bool): Whether to return two neighbors (True) or one (False).

    Returns:
        Tuple[int, List[int]]: A tuple where:
            - The first element is the index of the most electronegative neighbor.
            - The second element is a list of the remaining ranked neighbors (empty if only one neighbor is requested).

    Raises:
        ValueError: If the atom has no valid neighbors.
    """
    neighbors = [neighbor for neighbor in spc.mol.atoms[atom_index].edges.keys()
                 if spc.mol.atoms.index(neighbor) != exclude_index]

    if not neighbors:
        raise ValueError(f"Atom at index {atom_index} has no valid neighbors.")
    electronegativities = read_yaml_file(os.path.join(ARC_PATH, 'data', 'electronegativity.yml'))
    def get_neighbor_total_electronegativity(neighbor: 'Atom') -> float:
        """
        Calculate the total electronegativity of a neighbor based on its bonded neighbors.
        Args:
            neighbor (Atom): The atom to calculate the total electronegativity for.
        Returns:
            float: The total electronegativity of the neighbor
        """
        return sum(
            electronegativities[n.symbol] * neighbor.edges[n].order
            for n in neighbor.edges.keys()
        )
    effective_electronegativities = [(electronegativities[n.symbol] * spc.mol.atoms[atom_index].edges[n].order,
            get_neighbor_total_electronegativity(n), n ) for n in neighbors]
    effective_electronegativities.sort(reverse=True, key=lambda x: (x[0], x[1]))
    sorted_neighbors = [spc.mol.atoms.index(n[2]) for n in effective_electronegativities]
    most_electronegative = sorted_neighbors[0]
    remaining_neighbors = sorted_neighbors[1:] if two_neighbors else []
    return most_electronegative, remaining_neighbors

def setup_zmat_indices(initial_xyz: dict,
                       xyz_indices: dict) -> Tuple[dict, dict]:
    """
    Convert XYZ coordinates to Z-matrix format and set up corresponding indices.

    Args:
        initial_xyz (dict): XYZ coordinates of the molecule.
        xyz_indices (dict): Dictionary mapping atom types to their XYZ indices.

    Returns:
        tuple: A tuple containing:
            - dict: Z-matrix representation of the molecule
            - dict: Dictionary mapping atom types to their Z-matrix indices
    """
    initial_zmat = zmat_from_xyz(initial_xyz, consolidate=False)
    zmat_indices = {
        'a': key_by_val(initial_zmat.get('map', {}), xyz_indices['a']),
        'b': key_by_val(initial_zmat.get('map', {}), xyz_indices['b']),
        'f': key_by_val(initial_zmat.get('map', {}), xyz_indices['f']),
        'd': key_by_val(initial_zmat.get('map', {}), xyz_indices['d']) if xyz_indices['d'] is not None else None
    }
    return initial_zmat, zmat_indices

def adjust_dihedral_angles(initial_zmat: dict,
                           zmat_indices: dict,
                           dihedrals_to_change_num: int) -> bool:
    """
    Adjust dihedral angles in the Z-matrix based on given indices and number of dihedrals to change.

    Args:
        initial_zmat (dict): The Z-matrix to modify.
        zmat_indices (dict): Dictionary containing Z-matrix indices for atoms 'a', 'b', 'f', 'd'.
        dihedrals_to_change_num (int): Number of dihedral angles to adjust.

    Returns:
        bool: True if adjustments were made successfully, False if no more adjustments possible.
    """
    matches = get_matching_dihedrals(initial_zmat, zmat_indices['a'], zmat_indices['b'],
                                    zmat_indices['f'], zmat_indices['d'])
    total_dihedrals_num = len(matches)
    if (dihedrals_to_change_num > total_dihedrals_num) and (total_dihedrals_num != 0):
        return False
    indices_list = matches[:dihedrals_to_change_num] if matches else None

    if indices_list:
        for indices in indices_list:
            push_up_dihedral(zmat=initial_zmat, indices=indices, base_adjustment_factor=0.3)

    return True

def push_up_dihedral(zmat: dict,
                     indices: List[int],
                     base_adjustment_factor: float) -> None:
    """
    Adjust the value of a dihedral angle in the Z-matrix based on its current value.

    Args:
        zmat (Dict): The initial Z-matrix.
        indices (List[int]): The indices defining the dihedral angle.
        base_adjustment_factor (float): Base factor for adjustment.

    Returns:
        None
    """
    parameter_name = get_parameter_from_atom_indices(zmat=zmat, indices=indices, xyz_indexed=False)
    current_value = zmat['vars'].get(parameter_name, 0)
    print(f"Current dihedral value for {parameter_name}: {current_value}")

    if abs(current_value) < 10:
        adjustment_factor = base_adjustment_factor + 1
    elif 170 <= abs(current_value) <= 190:
        adjustment_factor = 1 - base_adjustment_factor
    else:
        print(f"No adjustment needed for {parameter_name} (not close to 0 or ±180 degrees)")
        return
    # apply adjustment and normalize the angle
    new_value = (current_value + 360) * adjustment_factor - 360 if abs(
        current_value) < 1e-3 else current_value * adjustment_factor
    normalized_value = (new_value + 180) % 360 - 180
    zmat['vars'][parameter_name] = normalized_value
    print(f"Updated dihedral value for {parameter_name}: {normalized_value}")

def get_matching_dihedrals(zmat: dict,
                          a: int,
                          b: int,
                          f: int,
                          d: Optional[int]) -> List[List[int]]:
    """
    Retrieve all dihedral angles in the Z-matrix that match the given atom indices.
    This function scans the Z-matrix for dihedral parameters (keys starting with 'D_' or 'DX_')
    and collects those whose indices match the specified atoms.

    Args:
        zmat (dict): The Z-matrix containing atomic coordinates and parameters.
        a (int): The first atom index to match.
        b (int): The second atom index to match.
        f (int): The third atom index to match.
        d (Optional[int]): The fourth atom index to match (optional).

    Returns:
        List[List[int]]: A list of lists, where each sublist contains the indices of a matching dihedral.
                         Returns an empty list if no matches are found.
    """
    matches = []
    for key in zmat['vars']:
        if key.startswith('D_') or key.startswith('DX_'):
            indices = [int(idx) for idx in key.split('_')[1:]]
            if d is not None:
                if a in indices and b in indices and (f in indices or d in indices):
                    matches.append(indices)
            else:
                if a in indices and b in indices and f in indices:
                    matches.append(indices)
    return matches

def stretch_ab_bond(initial_zmat: 'dict',
                    xyz_indices: 'dict',
                    zmat_indices: 'dict',
                    hydrolysis_parameters: 'dict',
                    reaction_family: str) -> None:
    """
    Stretch the bond between atoms a and b in the Z-matrix based on the reaction family parameters.

    Args:
        initial_zmat (dict): The Z-matrix to modify.
        xyz_indices (dict): Dictionary containing atom indices in XYZ coordinates.
        zmat_indices (dict): Dictionary containing atom indices in Z-matrix coordinates.
        hydrolysis_parameters (dict): Parameters for hydrolysis reactions.
        reaction_family (str): The type of reaction family, used to get specific stretch parameters.

    Returns:
        None: Modifies the Z-matrix in place.

    Note:
        The stretching direction is determined by comparing atom positions in both
        coordinate systems to ensure consistent bond modifications. The stretch degree
        is obtained from the hydrolysis parameters for the specific reaction family.
    """
    a_before_b_xyz = xyz_indices['a'] < xyz_indices['b']
    a_before_b_zmat = zmat_indices['a'] < zmat_indices['b']
    stretch_degree = hydrolysis_parameters['family_parameters'][str(reaction_family)]['stretch']

    if int(a_before_b_zmat) + int(a_before_b_xyz) == 1:
        indices = (min(xyz_indices['a'], xyz_indices['b']), max(xyz_indices['a'], xyz_indices['b']))
    else:
        indices = (max(xyz_indices['a'], xyz_indices['b']), min(xyz_indices['a'], xyz_indices['b']))

    stretch_zmat_bond(zmat=initial_zmat, indices=indices, stretch=stretch_degree)

def process_family_specific_adjustments(is_set_1: bool,
                                        is_set_2: bool,
                                        reaction_family: str,
                                        hydrolysis_parameters: dict,
                                        initial_zmat: dict,
                                        water: 'ARCSpecies',
                                        xyz_indices: dict,
                                        zmats_total: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Process specific adjustments for different hydrolysis reaction families if needed, then generate TS guesses .

    Args:
        is_set_1 (bool): Whether the reaction belongs to parameter set 1.
        is_set_2 (bool): Whether the reaction belongs to parameter set 2.
        reaction_family (str): Type of hydrolysis reaction ('ether_hydrolysis' or others).
        hydrolysis_parameters (dict): Parameters for different hydrolysis families.
        initial_zmat (dict): Initial Z-matrix of the molecule.
        water (ARCSpecies): Water molecule for the reaction.
        xyz_indices (dict): Dictionary of atom indices in XYZ coordinates.
        zmats_total (List[dict]): List of existing Z-matrices.

    Returns:
        Tuple[List[dict], List[dict]]: Generated XYZ guesses and updated Z-matrices list.

    Raises:
        ValueError: If the reaction family is not supported.
    """
    a_xyz, b_xyz, f_xyz, o_xyz, h1_xyz, d_xyz= xyz_indices.values()
    r_atoms = [a_xyz, o_xyz, o_xyz]
    a_atoms = [[b_xyz, a_xyz], [a_xyz, o_xyz], [h1_xyz, o_xyz]]
    d_atoms = ([[f_xyz, d_xyz, a_xyz], [b_xyz, a_xyz, o_xyz], [a_xyz, h1_xyz, o_xyz]]
               if d_xyz is not None else
               [[f_xyz, b_xyz, a_xyz], [b_xyz, a_xyz, o_xyz], [a_xyz, h1_xyz, o_xyz]])
    r_value = hydrolysis_parameters['default_parameters']['r_value']
    a_value = hydrolysis_parameters['family_parameters'][str(reaction_family)]['a_value']
    d_values = hydrolysis_parameters['family_parameters'][str(reaction_family)]['d_values']

    if is_set_1:
        if reaction_family == 'ether_hydrolysis':
            r_value[0] = hydrolysis_parameters['family_parameters'][str(reaction_family)]['r_value_adjustment']
        initial_xyz = zmat_to_xyz(initial_zmat)
        return generate_hydrolysis_ts_guess(initial_xyz, water, r_atoms, a_atoms, d_atoms,
                                            r_value, a_value, d_values, zmats_total, is_set_1)
    elif is_set_2:
        initial_xyz = zmat_to_xyz(initial_zmat)
        return generate_hydrolysis_ts_guess(initial_xyz, water, r_atoms, a_atoms, d_atoms,
                                            r_value, a_value, d_values, zmats_total, is_set_1, threshold=0.6)
    else:
        raise ValueError(f"Family {reaction_family} not supported for hydrolysis TS guess generation.")

def generate_hydrolysis_ts_guess(initial_xyz: dict,
                                 water: 'ARCSpecies',
                                 r_atoms: List[int],
                                 a_atoms: List[List[int]],
                                 d_atoms: List[List[int]],
                                 r_value: List[float],
                                 a_value: List[float],
                                 d_values: List[List[float]],
                                 zmats_total: List[dict],
                                 is_set_1: bool,
                                 threshold: float = 0.8
                                 ) -> Tuple[List[dict], List[dict]]:
    """
    Generate Z-matrices and Cartesian coordinates for transition state (TS) guesses.

    Args:
        initial_xyz (dict): The initial coordinates of the reactant.
        water (ARCSpecies): The water molecule involved in the reaction.
        r_atoms (List[int]): Atom pairs for defining bond distances.
        a_atoms (List[List[int]]): Atom triplets for defining bond angles.
        d_atoms (List[List[int]]): Atom quartets for defining dihedral angles.
        r_value (List[float]): Bond distances for each atom pair.
        a_value (List[float]): Bond angles for each atom triplet.
        d_values (List[List[float]]): Sets of dihedral angles for TS guesses.
        zmats_total (List[dict]): Existing Z-matrices to avoid duplicates.
        is_set_1 (bool): Whether the reaction belongs to parameter set 1.
        threshold (float): Threshold for atom collision checking.

    Returns:
        Tuple[List[dict], List[dict]]: Unique TS guesses (XYZ coords and Z-matrices).
    """
    xyz_guesses = []

    for index, d_value in enumerate(d_values):
        xyz_guess = copy.deepcopy(initial_xyz)

        for i in range(3):
            xyz_guess = add_atom_to_xyz_using_internal_coords(
                xyz=xyz_guess,
                element=water.mol.atoms[i].element.symbol,
                r_index=r_atoms[i],
                a_indices=a_atoms[i],
                d_indices=d_atoms[i],
                r_value=r_value[i],
                a_value=a_value[i],
                d_value=d_value[i]
            )
        if is_set_1:
            if check_dao_angle(d_atoms[0] , xyz_guess):
                continue
        zmat_guess = xyz_to_zmat(xyz_guess)
        duplicate = any(compare_zmats(existing, zmat_guess) for existing in zmats_total)

        if xyz_guess is not None and not colliding_atoms(xyz_guess, threshold=threshold) and not duplicate:
            xyz_guesses.append(xyz_guess)
            zmats_total.append(zmat_guess)
        else:
            print(f"Colliding atoms or existing guess: {xyz_guess}")

    return xyz_guesses, zmats_total

def hydrolysis(reaction: 'ARCReaction'):
    """
    Generate TS guesses for reactions of the ARC "hydrolysis" families.

    Args:
        reaction: An ARCReaction instance.

    Returns:
        List[dict]: Cartesian coordinates of TS guesses for all reactions.
    """
    xyz_guesses_total, zmats_total= [], []
    product_dicts = get_reaction_family_products(
        rxn=reaction,
        rmg_family_set = 'default',
        consider_rmg_families = False,
        consider_arc_families = True,
        )
    ester_and_ether_families = False
    if any('ester_hydrolysis' in d.get('family', []) for d in product_dicts) and \
            any('ether_hydrolysis' in d.get('family', []) for d in product_dicts):
        ester_and_ether_families = True
    counter = 0
    while not xyz_guesses_total \
            or (ester_and_ether_families and not any(item['family'] == 'ester_hydrolysis' for item in xyz_guesses_total)):
        counter += 1
        for product_dict in product_dicts:
            arc_reactants, _ = reaction.get_reactants_and_products(arc=True, return_copies=True)
            arc_reactant, water = None, None
            for spc in arc_reactants:
                if not is_water(spc):
                    arc_reactant = spc
                else:
                    water = spc
            if arc_reactant is None or water is None:
                raise ValueError("Reactants must include a non-water molecule and water.")
            initial_xyz = arc_reactant.get_xyz()
            initial_zmat = zmat_from_xyz(initial_xyz, consolidate=False)
            print(initial_zmat)
            map_dict = initial_zmat.get('map', {})
            a_xyz_index, b_xyz_index = product_dict['r_label_map']['*1'], product_dict['r_label_map']['*2']
            a_zmat_index, b_zmat_index = key_by_val(map_dict, a_xyz_index), key_by_val(map_dict, b_xyz_index)
            a_before_b_xyz, a_before_b_zmat = a_xyz_index < b_xyz_index, a_zmat_index < b_zmat_index
            o_index = len(arc_reactant.mol.atoms)
            h1_index = o_index + 1
            r_atoms = [a_xyz_index, o_index, o_index]
            r_value = [1.8, 1.21, 0.97]  # todo: take all hard-coded numbers from this function into a designated YAML file
            a_atoms = [[b_xyz_index, a_xyz_index], [a_xyz_index, o_index], [h1_index, o_index]]
            family = product_dict['family']
            if family in FAMILY_SETS['set_1']:
                f_xyz_index, d_xyz_index = get_neighbors_by_electronegativity(arc_reactant, a_xyz_index, b_xyz_index, True)
                print(a_xyz_index, b_xyz_index, f_xyz_index, d_xyz_index)
                f_zmat_index, d_zmat_index = key_by_val(map_dict, f_xyz_index), key_by_val(map_dict, d_xyz_index)
                total_dihedrals = count_all_possible_dihedrals(initial_zmat, a_zmat_index, b_zmat_index, f_zmat_index, d_zmat_index)
                if (counter > total_dihedrals) and (total_dihedrals != 0):
                    print("All possible dihedral adjustments have been tried.")
                    return xyz_guesses_total, zmats_total
                d_atoms = [[f_xyz_index, d_xyz_index, a_xyz_index], [b_xyz_index, a_xyz_index, o_index], [a_xyz_index, h1_index, o_index]]
                indices_list = find_matching_dihedral(initial_zmat, a_zmat_index, b_zmat_index, f_zmat_index, d_zmat_index,counter)
                for indices in indices_list:
                    if indices is not None:
                        push_up_dihedral(zmat=initial_zmat, indices=indices, base_adjustment_factor=0.3)

                if family == 'ether_hydrolysis':
                    if int(a_before_b_zmat) + int(a_before_b_xyz) == 1:
                        stretch_zmat_bond(zmat=initial_zmat, indices=(min(a_xyz_index, b_xyz_index), max(a_xyz_index, b_xyz_index)), stretch=1.5)
                    else:
                        stretch_zmat_bond(zmat=initial_zmat, indices=(max(a_xyz_index, b_xyz_index), min(a_xyz_index, b_xyz_index)), stretch=1.5)
                    r_value[0] = 2.1
                    a_value = [65, 72, 106]
                    d_values = [[98.25, -0.72, 103], [-98.25, -0.72, 103], [98.25, -0.72, -103], [-98.25, -0.72, -103]]
                elif family == 'imine_hydrolysis':
                    if int(a_before_b_zmat) + int(a_before_b_xyz) == 1:
                        stretch_zmat_bond(zmat=initial_zmat, indices=(min(a_xyz_index, b_xyz_index), max(a_xyz_index, b_xyz_index)), stretch=1.3)
                    else:
                        stretch_zmat_bond(zmat=initial_zmat, indices=(max(a_xyz_index, b_xyz_index), min(a_xyz_index, b_xyz_index)), stretch=1.3)
                    a_value = [78, 70, 111]
                    d_values = [[108, 12, 113], [-108, 12, 113], [108, 12, -113], [-108, 12, -113]]
                else:
                    if int(a_before_b_zmat) + int(a_before_b_xyz) == 1:
                        stretch_zmat_bond(zmat=initial_zmat, indices=(min(a_xyz_index, b_xyz_index), max(a_xyz_index, b_xyz_index)), stretch=1.3)
                    else:
                        stretch_zmat_bond(zmat=initial_zmat, indices=(max(a_xyz_index, b_xyz_index), min(a_xyz_index, b_xyz_index)), stretch=1.3)
                    a_value = [77, 71, 111]
                    d_values = [[140, 1.64, 103], [-140, 1.64, 103], [140, 1.64, -103], [-140, 1.64, -103]]
                initial_xyz = zmat_to_xyz(initial_zmat)
                xyz_guesses, zmats_total = generate_hydrolysis_ts_guess(
                    initial_xyz, water, r_atoms, a_atoms, d_atoms, r_value, a_value, d_values, zmats_total
                )

            elif family in FAMILY_SETS['set_2']:
                f_xyz_index, d_xyz_index= get_neighbors_by_electronegativity(arc_reactant, a_xyz_index, b_xyz_index, False), None
                f_zmat_index = key_by_val(map_dict, f_xyz_index)
                total_dihedrals = count_all_possible_dihedrals(initial_zmat, a_zmat_index, b_zmat_index, f_zmat_index, None)
                if (counter > total_dihedrals) and (total_dihedrals != 0):
                    print("All possible dihedral adjustments have been tried.")
                    return xyz_guesses_total, zmats_total
                indices_list = find_matching_dihedral(initial_zmat, a_zmat_index, b_zmat_index, f_zmat_index,None, counter)
                for indices in indices_list:
                    if indices is not None:
                        push_up_dihedral(zmat=initial_zmat, indices=indices, base_adjustment_factor=0.3)
                if int(a_before_b_zmat) + int(a_before_b_xyz) == 1:
                    stretch_zmat_bond(zmat=initial_zmat, indices=(min(a_xyz_index, b_xyz_index), max(a_xyz_index, b_xyz_index)), stretch=1.1)
                else:
                    stretch_zmat_bond(zmat=initial_zmat, indices=(max(a_xyz_index, b_xyz_index), min(a_xyz_index, b_xyz_index)), stretch=1.1)
                a_atoms = [[b_xyz_index, a_xyz_index], [a_xyz_index, o_index], [h1_index, o_index]]
                a_value = [97, 58, 111]
                d_atoms = [[f_xyz_index, b_xyz_index, a_xyz_index], [b_xyz_index, a_xyz_index, o_index], [a_xyz_index, h1_index, o_index]]
                d_values = [[174, -0.0154, 104], [-174, -0.0154, 104], [174, -0.0154, -104], [-174, -0.0154, -104]]
                initial_xyz = zmat_to_xyz(initial_zmat)
                xyz_guesses, zmats_total = generate_hydrolysis_ts_guess(
                    initial_xyz, water, r_atoms, a_atoms, d_atoms, r_value, a_value, d_values, zmats_total,threshold=0.6
                )

            else:
                raise ValueError(f_xyz_index, "Family {product_dict['family']} not supported for hydrolysis TS guess generation")

            if xyz_guesses:
                xyz_guesses_total.append({'family': product_dict['family'],
                                          'indices': [a_xyz_index, b_xyz_index, f_xyz_index, d_xyz_index, o_index, h1_index],
                                          'xyz_guesses': xyz_guesses},
                                         )

    return xyz_guesses_total, zmats_total


register_job_adapter('heuristics', HeuristicsAdapter)
        


