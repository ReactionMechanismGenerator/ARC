"""
An adapter for executing TS guess jobs based on linear interpolation of internal coordinate values.
"""

import copy
import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from arc.common import almost_equal_coords, get_angle_in_180_range, get_logger
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter, ts_adapters_by_rmg_family
from arc.job.factory import register_job_adapter
from arc.mapping.driver import map_rxn
from arc.plotter import save_geo
from arc.species.converter import order_xyz_by_atom_map, zmat_to_xyz
from arc.species.species import ARCSpecies, TSGuess, colliding_atoms
from arc.species.zmat import check_ordered_zmats, update_zmat_by_xyz, xyz_to_zmat

if TYPE_CHECKING:
    from arc.level import Level
    from arc.reaction import ARCReaction


BASE_WEIGHT_GRID = (0.35, 0.50, 0.65)
HAMMOND_DELTA = 0.10
WEIGHT_ROUND = 3

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
            if rxn.family not in supported_families or not rxn.is_unimolecular():
                logger.warning(f'The linear TS search adapter does not support the {rxn.family} reaction family.')
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
            weights = get_weight_grid(rxn)
            for w_i, w in enumerate(weights):
                t0 = datetime.datetime.now()
                xyzs = interpolate(rxn=rxn, weight=w)
                t_ex = datetime.datetime.now() - t0
                if not xyzs:
                    continue

                for xyz in xyzs:
                    if colliding_atoms(xyz):
                        continue
                    unique = True
                    for other_tsg in rxn.ts_species.ts_guesses:
                        if almost_equal_coords(xyz, other_tsg.initial_xyz):
                            if 'linear' not in other_tsg.method.lower():
                                other_tsg.method += f' and Linear (w={w:.2f})'
                            unique = False
                            break

                    if unique:
                        method = f'linear w={w:.2f}'
                        ts_guess = TSGuess(method=method,
                                           index=len(rxn.ts_species.ts_guesses),
                                           method_index=w_i,
                                           t0=t0,
                                           execution_time=t_ex,
                                           success=True,
                                           family=rxn.family,
                                           xyz=xyz,
                                           )
                        rxn.ts_species.ts_guesses.append(ts_guess)

                        save_geo(xyz=xyz,
                                 path=self.local_path,
                                 filename=f'Linear w={w:.2f}',
                                 format_='xyz',
                                 comment=f'Linear w={w:.2f}, family: {rxn.family}',
                                 )

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
                weight: float = 0.5,
                ) -> Optional[List[dict]]:
    """
    Search for a TS by interpolating internal coords.

    Args:
        rxn (ARCReaction): The reaction to process.
        weight (float): Interpolation weight (0=reactant-like, 1=product-like).

    Returns:
        Optional[List[dict]]: The XYZ coordinates guess.
    """
    if rxn.is_isomerization():
        return interpolate_isomerization(rxn=rxn, weight=weight)
    return None


def interpolate_isomerization(rxn: 'ARCReaction',
                              weight: float = 0.5,
                              ) -> Optional[List[dict]]:
    """
    Search for a TS of an isomerization reaction by interpolating internal coords.

    Args:
        rxn (ARCReaction): The reaction to process.
        weight (float): Interpolation weight (0=reactant-like, 1=product-like).

    Returns:
        Optional[List[dict]]: The XYZ coordinates guesses.
    """
    if not (0.0 <= weight <= 1.0):
        return None
    ts_xyzs: List[dict] = list()
    for i, product_dict in enumerate(rxn.product_dicts):
        r_label_dict = product_dict['r_label_map']
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
        atom_map = map_rxn(rxn=rxn, product_dict_index_to_try=i)
        print(f'i: {i},\natom_map: {[i+1 for i in atom_map]}\nproduct_dict: {product_dict}\n\n')
        if atom_map is None:
            continue
        ordered_p_xyz = order_xyz_by_atom_map(xyz=rxn.p_species[0].get_xyz(), atom_map=atom_map)
        # print(f'r_zmat: {r_zmat}')
        # print(f'ordered_p_xyz: {ordered_p_xyz}')
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
    Interpolate internal coordinates using a weight.
    Bond lengths/angles are interpolated linearly.
    Dihedrals (D* / DX* in the coords template) are interpolated on a circle
    using the shortest signed difference in (-180, 180].
    Dihedrals are typically named like 'D_...' (real atoms) or 'DX_...' (dummy X dihedral).

    Args:
        zmat_1 (dict): ZMat 1.
        zmat_2 (dict): ZMat 2.
        weight (float, optional): The weight to use on a scale of 0 (the reactant) to 1 (the product).
                                  A value of 0.5 means exactly in the middle.

    Returns:
        Optional[dict]: The weighted average ZMat.
    """
    if not check_ordered_zmats(zmat_1, zmat_2) or not (0.0 <= weight <= 1.0):
        return None
    if 'vars' not in zmat_1 or 'vars' not in zmat_2 or 'coords' not in zmat_1 or 'coords' not in zmat_2:
        return None

    dihedral_vars = set()
    for row in zmat_1['coords']:
        # row is a tuple like (Rvar, Avar, Dvar) or (None,None,None) for first atom
        if isinstance(row, (tuple, list)) and len(row) == 3:
            dvar = row[2]
            if isinstance(dvar, str) and (dvar.startswith('D_') or dvar.startswith('DX_')):
                dihedral_vars.add(dvar)

    ts_zmat = copy.deepcopy(zmat_1)
    ts_zmat['vars'] = dict()

    for key, a in zmat_1['vars'].items():
        if key not in zmat_2['vars']:
            return None
        b = zmat_2['vars'][key]
        if key in dihedral_vars:
            ts_zmat['vars'][key] = interp_dihedral_deg(a, b, w=weight)
        else:
            ts_zmat['vars'][key] = a + weight * (b - a)

    return ts_zmat


def get_rxn_weight(rxn: 'ARCReaction',
                   w_min: float = 0.30,
                   w_max: float = 0.70,
                   delta_e_sat: float = 150.0,
                   reorg_energy: Optional[Union[float, Tuple[float, float]]] = None,
                   ) -> Optional[float]:
    """
    Estimate an interpolation weight w (0=reactant-like, 1=product-like) using reaction thermochemistry only.

    Chemically motivated model:
        Use a Hammond/Leffler parameter (alpha) via a Marcus-like relation:
            alpha ≈ 0.5 + ΔE / (2*λ)
        where λ is an effective "reorganization energy" scale (same units as ΔE).
        We then clamp alpha into [w_min, w_max].

    Defaults:
        By default we choose λ so that |ΔE| = delta_e_sat maps to the extrema w_min / w_max.

    Args:
        rxn: The reaction to process.
        w_min: Minimum allowed weight (reactant-like limit), in [0, 0.5].
        w_max: Maximum allowed weight (product-like limit), in [0.5, 1].
        delta_e_sat: Magnitude of reaction energy for considering the TS fully shifted to the extrema (kJ/mol).
        reorg_energy:
            Either:
              - None (derive λ from delta_e_sat and (w_min,w_max)),
              - a single float λ used for both signs,
              - a tuple (λ_exo, λ_endo) to allow asymmetry.

    Returns:
        The estimated weight, or None if energies are unavailable.
    """
    if w_min > w_max:
        w_min, w_max = w_max, w_min
    if not (0.0 <= w_min <= 0.5 <= w_max <= 1.0):
        raise ValueError(f"Invalid bounds: w_min={w_min}, w_max={w_max}. "
                         f"Require w_min in [0,0.5] and w_max in [0.5,1].")
    if delta_e_sat <= 0.0:
        raise ValueError(f"delta_e_sat must be > 0, got {delta_e_sat}")

    reactants, products = rxn.get_reactants_and_products(return_copies=False)
    r_e0 = [spc.e0 for spc in reactants]
    p_e0 = [spc.e0 for spc in products]
    if all(e is not None for e in (r_e0 + p_e0)):
        r_e = r_e0
        p_e = p_e0
    else:
        r_ee = [spc.e_elect for spc in reactants]
        p_ee = [spc.e_elect for spc in products]
        if not all(e is not None for e in (r_ee + p_ee)):
            return None
        r_e = r_ee
        p_e = p_ee

    delta_e = sum(p_e) - sum(r_e)
    if abs(delta_e) < 1e-3:
        return 0.5

    if reorg_energy is None:
        lam_endo = delta_e_sat / (2.0 * (w_max - 0.5)) if (w_max - 0.5) > 0 else float('inf')
        lam_exo  = delta_e_sat / (2.0 * (0.5 - w_min)) if (0.5 - w_min) > 0 else float('inf')
    elif isinstance(reorg_energy, tuple):
        if len(reorg_energy) != 2:
            raise ValueError(f"reorg_energy tuple must be (lambda_exo, lambda_endo), got {reorg_energy}")
        lam_exo, lam_endo = float(reorg_energy[0]), float(reorg_energy[1])
    else:
        lam_exo = lam_endo = float(reorg_energy)

    if lam_exo <= 0.0 or lam_endo <= 0.0:
        raise ValueError(f"Reorganization energies must be > 0. Got lambda_exo={lam_exo}, lambda_endo={lam_endo}")

    lam = lam_endo if delta_e > 0.0 else lam_exo

    w = 0.5 + delta_e / (2.0 * lam)
    if w < w_min:
        return w_min
    if w > w_max:
        return w_max
    return w


def get_weight_grid(rxn: 'ARCReaction',
                    include_hammond: bool = True,
                    base_grid: Tuple[float, ...] = BASE_WEIGHT_GRID,
                    hammond_delta: float = HAMMOND_DELTA,
                    ) -> List[float]:
    """
    Generate a small set of interpolation weights to try.
    Always includes a symmetric grid around 0.5, and optionally also tries
    a Hammond/Marcus-biased guess ± delta.

    Returns:
        List[float]: Sorted unique weights in [0, 1].
    """
    weights: List[float] = list(base_grid)

    if include_hammond:
        w0 = get_rxn_weight(rxn)
        if w0 is not None:
            weights.extend([w0 - hammond_delta, w0, w0 + hammond_delta])
    uniq = {round(_clip01(w), WEIGHT_ROUND) for w in weights}
    return sorted(uniq)


def interp_dihedral_deg(a: float, b: float, w: float = 0.5) -> float:
    """
    Interpolate dihedral angles in degrees along the shortest signed difference in (-180, 180].
    E.g., the distance between -179 and 179 is 2 degrees, not 358.

    Args:
        a (float): The first angle in degrees.
        b (float): The second angle in degrees.
        w (float, optional): The weight between 0 and 1.

    Returns:
        float: The interpolated angle in degrees.
    """
    a = get_angle_in_180_range(a, round_to=None)
    b = get_angle_in_180_range(b, round_to=None)
    d = get_angle_in_180_range(b - a, round_to=None)
    return get_angle_in_180_range(a + w * d, round_to=2)


def _clip01(x: float) -> float:
    """Clip a float to the [0, 1] range."""
    return max(0.0, min(1.0, x))


register_job_adapter('linear', LinearAdapter)
