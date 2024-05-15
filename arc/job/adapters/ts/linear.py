"""
An adapter for executing TS guess jobs based on linear interpolation of internal coordinate values.
"""

import copy
import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from arc.common import almost_equal_coords, get_logger
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter, ts_adapters_by_rmg_family
from arc.job.factory import register_job_adapter
from arc.mapping.engine import (get_atom_indices_of_labeled_atoms_in_an_rmg_reaction,
                                get_rmg_reactions_from_arc_reaction,
                                )
from arc.plotter import save_geo
from arc.species.converter import order_xyz_by_atom_map, zmat_to_xyz
from arc.species.species import ARCSpecies, TSGuess, colliding_atoms
from arc.species.zmat import check_ordered_zmats, get_atom_order, update_zmat_by_xyz, xyz_to_zmat

if TYPE_CHECKING:
    from arc.level import Level
    from arc.reaction import ARCReaction


DIHEDRAL_INCREMENT = 30

logger = get_logger()


class LinearAdapter(JobAdapter):
    """
    A class for executing TS guess jobs based on linear interpolation of internal coordinate values.

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
                 reactions: Optional[List['ARCReaction']] = None,
                 rotor_index: Optional[int] = None,
                 server: Optional[str] = None,
                 server_nodes: Optional[list] = None,
                 species: Optional[List[ARCSpecies]] = None,
                 testing: bool = False,
                 times_rerun: int = 0,
                 torsions: Optional[List[List[int]]] = None,
                 tsg: Optional[int] = None,
                 xyz: Optional[dict] = None,
                 ):

        self.incore_capacity = 50
        self.job_adapter = 'linear'
        self.command = None
        self.execution_type = execution_type or 'incore'

        if reactions is None:
            raise ValueError('Cannot execute TS Linear without ARCReaction object(s).')

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

        supported_families = [key for key, val in ts_adapters_by_rmg_family.items() if 'linear' in val]

        self.reactions = [self.reactions] if not isinstance(self.reactions, list) else self.reactions
        for rxn in self.reactions:
            family_label = rxn.family.label
            if family_label not in supported_families or not rxn.is_unimolecular():
                logger.warning(f'The heuristics TS search adapter does not support the {family_label} reaction family.')
                continue
            if any(spc.get_xyz() is None for spc in rxn.r_species + rxn.p_species):
                logger.warning(f'The linear TS search adapter cannot process a reaction if 3D coordinates of '
                               f'some/all of its reactants/products are missing.\nNot processing {rxn}.')
                continue

            rxn.ts_species = rxn.ts_species or ARCSpecies(label='TS',
                                                          is_ts=True,
                                                          charge=rxn.charge,
                                                          multiplicity=rxn.multiplicity,
                                                          )

            t0_0 = datetime.datetime.now()
            xyzs_0 = interpolate(rxn=rxn, use_weights=False)
            t_ex_0 = datetime.datetime.now() - t0_0

            t0_1 = datetime.datetime.now()
            xyzs_1 = interpolate(rxn=rxn, use_weights=True)
            t_ex_1 = datetime.datetime.now() - t0_1

            index = 0
            for xyzs, t0, t_ex in zip([xyzs_0, xyzs_1], [t0_0, t0_1], [t_ex_0, t_ex_1]):
                if xyzs is None or not len(xyzs):
                    continue
                for xyz in xyzs:
                    if colliding_atoms(xyz):
                        continue
                    unique = True
                    for other_tsg in rxn.ts_species.ts_guesses:
                        if almost_equal_coords(xyz, other_tsg.initial_xyz):
                            if 'linear' not in other_tsg.method.lower():
                                other_tsg.method += f' and Linear {index}'
                            unique = False
                            break
                    if unique:
                        ts_guess = TSGuess(method=f'linear {index}',
                                           index=len(rxn.ts_species.ts_guesses),
                                           method_index=index,
                                           t0=t0,
                                           execution_time=t_ex,
                                           success=True,
                                           family=family_label,
                                           xyz=xyz,
                                           )
                        rxn.ts_species.ts_guesses.append(ts_guess)
                        save_geo(xyz=xyz,
                                 path=self.local_path,
                                 filename=f'Linear {index}',
                                 format_='xyz',
                                 comment=f'Linear {index}, family: {family_label}',
                                 )
                    index += 1

            if len(self.reactions) < 5:
                successes = len([tsg for tsg in rxn.ts_species.ts_guesses if tsg.success and 'linear' in tsg.method])
                if successes:
                    logger.info(f'Linear successfully found {successes} TS guesses for {rxn.label}.')
                else:
                    logger.info(f'Linear did not find any successful TS guesses for {rxn.label}.')

        self.final_time = datetime.datetime.now()

    def execute_queue(self):
        """
        (Execute a job to the server's queue.)
        A single Heuristics job will always be executed incore.
        """
        self.execute_incore()


def interpolate(rxn: 'ARCReaction',
                use_weights: bool = False,
                ) -> Optional[dict]:
    """
    Search for a TS by interpolating internal coords.

    Args:
        rxn (ARCReaction): The reaction to process.
        use_weights (bool, optional): Whether to use the well energies to determine relative interpolation weights.

    Returns:
        Optional[dict]: The XYZ coordinates guess.
    """
    if rxn.is_isomerization():
        return interpolate_isomerization(rxn=rxn, use_weights=use_weights)
    return None


def interpolate_isomerization(rxn: 'ARCReaction',
                              use_weights: bool = False,
                              ) -> Optional[List[dict]]:
    """
    Search for a TS of an isomerization reaction by interpolating internal coords.

    Args:
        rxn (ARCReaction): The reaction to process.
        use_weights (bool, optional): Whether to use the well energies to determine relative interpolation weights.

    Returns:
        Optional[List[dict]]: The XYZ coordinates guesses.
    """
    weight = get_rxn_weight(rxn) if use_weights else 0.5
    if weight is None:
        return None
    rmg_reactions = get_rmg_reactions_from_arc_reaction(arc_reaction=rxn) or list()
    ts_xyzs = list()
    for rmg_reaction in rmg_reactions:
        r_label_dict = get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn,
                                                                            rmg_reaction=rmg_reaction)[0]
        if r_label_dict is None:
            continue
        expected_breaking_bonds, expected_forming_bonds = rxn.get_expected_changing_bonds(r_label_dict=r_label_dict)
        if expected_breaking_bonds is None or expected_forming_bonds is None:
            continue
        r_zmat = xyz_to_zmat(xyz=rxn.r_species[0].get_xyz(),
                             mol=rxn.r_species[0].mol,
                             consolidate=False,
                             constraints=get_r_constraints(expected_breaking_bonds=expected_breaking_bonds,
                                                           expected_forming_bonds=expected_forming_bonds),
                             )
        ordered_p_xyz = order_xyz_by_atom_map(xyz=rxn.p_species[0].get_xyz(), atom_map=rxn.atom_map)
        p_zmat = update_zmat_by_xyz(zmat=r_zmat, xyz=ordered_p_xyz)
        ts_zmat = average_zmat_params(zmat_1=r_zmat, zmat_2=p_zmat, weight=weight)

        if ts_zmat is not None:
            ts_xyzs.append(zmat_to_xyz(ts_zmat))
    return ts_xyzs


def get_r_constraints(expected_breaking_bonds: List[Tuple[int, int]],
                      expected_forming_bonds: List[Tuple[int, int]],
                      ) -> Dict[str, list]:
    """
    Get the "R_atom" constraints for the reactant ZMat.

    Args:
        expected_breaking_bonds (List[Tuple[int, int]]): Expected breaking bonds.
        expected_forming_bonds (List[Tuple[int, int]]): Expected forming bonds.

    Returns:
        Dict[str, list]: The constraints.
    """
    constraints = list()
    atom_occurrences = dict()
    for bond in expected_breaking_bonds + expected_forming_bonds:
        for atom in bond:
            if atom not in atom_occurrences:
                atom_occurrences[atom] = 0
            atom_occurrences[atom] += 1
    atoms_sorted_by_frequency = [k for k, _ in sorted(atom_occurrences.items(), key=lambda item: item[1], reverse=True)]
    for i, atom in enumerate(atoms_sorted_by_frequency):
        for bond in expected_breaking_bonds + expected_forming_bonds:
            if atom in bond and all(a not in atoms_sorted_by_frequency[:i] for a in bond):
                constraints.append(bond if atom == bond[0] else (bond[1], bond[0]))
                break
    return {'R_atom': constraints}


def average_zmat_params(zmat_1: dict,
                        zmat_2: dict,
                        weight: float = 0.5,
                        ) -> Optional[dict]:
    """
    Average internal coordinates using a weight.

    Args:
        zmat_1 (dict): ZMat 1.
        zmat_2 (dict): ZMat 2.
        weight (float, optional): The weight to use on a scale of 0 (the reactant) to 1 (the product).
                                  A value of 0.5 means exactly in the middle.

    Returns:
        Optional[dict]: The weighted average ZMat.
    """
    if not check_ordered_zmats(zmat_1, zmat_2) or weight < 0 or weight > 1:
        return None
    ts_zmat = copy.deepcopy(zmat_1)
    ts_zmat['vars'] = dict()
    for key in zmat_1['vars'].keys():
        ts_zmat['vars'][key] = zmat_1['vars'][key] + weight * (zmat_2['vars'][key] - zmat_1['vars'][key])
    return ts_zmat


def get_rxn_weight(rxn: 'ARCReaction') -> Optional[float]:
    """
    Get ratio between the activation energy (reactants to TS) to the overall energy path (reactants to TS to products).

    Args:
        rxn (ARCReaction): The reaction to process.

    Returns:
        float: The reaction weight.
    """
    reactants, products = rxn.get_reactants_and_products(arc=True, return_copies=False)
    r_e0 = [r.e0 for r in reactants]
    p_e0 = [p.e0 for p in products]
    ts_e0 = rxn.ts_species.e0
    if any(entry is None for entry in r_e0 + p_e0 + [ts_e0]):
        r_ee = [r.e_elect for r in reactants]
        p_ee = [p.e_elect for p in products]
        ts_ee = rxn.ts_species.e_elect
        if any(entry is None for entry in r_e0 + p_e0 + [ts_e0]):
            return None
        return get_weight(r_ee, p_ee, ts_ee)
    return get_weight(r_e0, p_e0, ts_e0)


def get_weight(r_e: List[Optional[float]],
               p_e: List[Optional[float]],
               ts_e: Optional[float],
               ) -> Optional[float]:
    """
    Get the path ratio of reactants-TS to reactants-TS-products.

    Args:
        r_e (List[float]): Reactant energies.
        p_e (List[float]): Product energies.
        ts_e: TS energy.

    Returns:
        Optional[float]: The reaction path ratio.
    """
    if any(entry is None for entry in r_e + p_e + [ts_e]):
        return None
    r_to_ts = ts_e - sum(r_e)
    p_to_ts = ts_e - sum(p_e)
    return r_to_ts / (r_to_ts + p_to_ts)


register_job_adapter('linear', LinearAdapter)
