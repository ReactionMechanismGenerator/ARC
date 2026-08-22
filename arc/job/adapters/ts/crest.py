"""
Utilities for running CREST within ARC.

Separated from heuristics so CREST can be conditionally imported and reused.

``TERMINAL_JOB_STATUSES`` lists every status after which a job is no longer polled.
:func:`arc.job.local.check_job_status` reports ``'done'``, ``'running'`` or ``'errored'`` and never
reports ``'failed'``, so ``'errored'`` has to be part of the terminal set for an errored job to stop
being polled. ``'failed'`` is retained because :func:`process_completed_jobs` reports it separately.

``MAX_CREST_SEEDS`` bounds how many CREST jobs a single reaction spawns. Each seed becomes one
multi-core metadynamics job, and the adapter executes in core, so the seed list is capped to keep
both the number of concurrent jobs and the time ARC's main loop spends blocked on them bounded.
"""

import datetime
import math
import os
import time

import numpy as np
from typing import TYPE_CHECKING, List, Optional, Union

from arc.common import almost_equal_coords, calc_rmsd, get_logger
from arc.exceptions import ConverterError
from arc.imports import settings, submit_scripts
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter, ts_adapters_by_rmg_family
from arc.job.adapters.ts.heuristics import DIHEDRAL_INCREMENT
from arc.job.adapters.ts.seed_hub import get_backup_ts_seeds, get_ts_seeds, get_wrapper_constraints
from arc.job.factory import register_job_adapter
from arc.job.local import check_job_status, delete_job, submit_job
from arc.plotter import save_geo
from arc.species.converter import reorder_xyz_string, xyz_file_format_to_xyz, xyz_to_str
from arc.species.species import ARCSpecies, TSGuess, colliding_atoms

if TYPE_CHECKING:
    from arc.level import Level
    from arc.reaction import ARCReaction

logger = get_logger()

MAX_CHECK_INTERVAL_SECONDS = 100
TERMINAL_JOB_STATUSES = ('done', 'errored', 'failed')
DEFAULT_MAX_CREST_WALL_TIME = 24 * 60 * 60
MAX_CREST_SEEDS = 8

# Number of atoms CREST holds rigid for each family it supports: the A--H--B triad of an
# H-abstraction, and the a/b/o/h1 tetrahedron of a hydrolysis. A family that is absent is never
# gated off by _crest_reactive_core_covers_molecule().
CREST_REACTIVE_CORE_SIZES = {'H_Abstraction': 3,
                             'carbonyl_based_hydrolysis': 4,
                             'ether_hydrolysis': 4,
                             'nitrile_hydrolysis': 4,
                             }

CREST_PATH = settings.get("CREST_PATH", None)
CREST_ENV_PATH = settings.get("CREST_ENV_PATH", None)
SERVERS = settings.get("servers", {})


def crest_available() -> bool:
    """
    Return whether CREST is configured for use.

    CREST needs a configured local server and at least one of ``CREST_PATH`` (a standalone
    executable) or ``CREST_ENV_PATH`` (an environment activation line).

    Returns:
        bool: Whether CREST can be used.
    """
    return bool(SERVERS.get("local")) and bool(CREST_PATH or CREST_ENV_PATH)


class CrestAdapter(JobAdapter):
    """
    A class for executing CREST TS conformer searches based on heuristics-generated guesses.
    """

    def __init__(self,
                 project: str,
                 project_directory: str,
                 job_type: Union[List[str], str],
                 args: Optional[dict] = None,
                 bath_gas: Optional[str] = None,
                 checkfile: Optional[str] = None,
                 conformer: Optional[int] = None,
                 constraints: Optional[List] = None,
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
        self.job_adapter = 'crest'
        self.command = None
        self.execution_type = execution_type or 'incore'

        if reactions is None:
            raise ValueError('Cannot execute TS CREST without ARCReaction object(s).')

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
        pass

    def set_files(self) -> None:
        pass

    def set_additional_file_paths(self) -> None:
        pass

    def set_input_file_memory(self) -> None:
        pass

    def execute_incore(self):
        """
        Run the CREST TS conformer search for every supported reaction of this job.

        For each reaction a seed list is built from the heuristics adapter, or, when that yields
        nothing (e.g. a linear/cumulene reactive center such as HCCO that the heuristic Z-matrix
        builder cannot assemble), from a successful TS guess that another adapter already appended
        to ``rxn.ts_species.ts_guesses``. An external guess is a valid seed because CREST only needs
        a seed geometry plus the family reactive-atom constraints, and those constraints are
        re-derived from the seed geometry. Guesses whose method contains ``crest`` are excluded so
        CREST is never seeded from a prior CREST result.

        The seed list is truncated to ``MAX_CREST_SEEDS`` entries, keeping the most geometrically
        distinct ones, and the number of dropped seeds is logged. Each surviving seed is submitted
        as one CREST job; the resulting geometries are appended as TS guesses unless they duplicate
        an existing successful guess, in which case that guess records ``'crest'`` as an additional
        method source. CREST is run without an explicit method flag, i.e., at its GFN2-xTB default.
        """
        self._log_job_execution()
        self.initial_time = self.initial_time if self.initial_time else datetime.datetime.now()

        supported_families = [key for key, val in ts_adapters_by_rmg_family.items() if 'crest' in val]

        self.reactions = [self.reactions] if not isinstance(self.reactions, list) else self.reactions
        for rxn_index, rxn in enumerate(self.reactions):
            if rxn.family not in supported_families:
                logger.warning(f'The CREST TS search adapter does not support the {rxn.family} reaction family.')
                continue
            if any(spc.get_xyz() is None for spc in rxn.r_species + rxn.p_species):
                logger.warning(f'The CREST TS search adapter cannot process a reaction if 3D coordinates of '
                               f'some/all of its reactants/products are missing.\nNot processing {rxn}.')
                continue
            if not crest_available():
                logger.warning('CREST is not available. Skipping CREST TS search.')
                break

            if _crest_reactive_core_covers_molecule(rxn):
                logger.info(
                    f'Skipping CREST TS search for {rxn.label}: the reactive core spans essentially '
                    f'the entire molecule (<=1 spectator atom), so CREST has no conformational degrees '
                    f'of freedom to sample. Deferring to the other TS-search methods.'
                )
                continue

            if rxn.ts_species is None:
                rxn.ts_species = ARCSpecies(label='TS',
                                            is_ts=True,
                                            charge=rxn.charge,
                                            multiplicity=rxn.multiplicity,
                                            )

            tsg = TSGuess(method='CREST')
            tsg.tic()

            crest_job_dirs = []
            crest_references = {}
            xyz_guesses = get_ts_seeds(
                reaction=rxn,
                base_adapter='heuristics',
                dihedral_increment=self.dihedral_increment,
            )
            if not xyz_guesses:
                xyz_guesses = get_backup_ts_seeds(rxn, exclude_method='crest')
                if xyz_guesses:
                    logger.info(
                        f'CREST heuristic seed construction failed for {rxn.label}; falling back to '
                        f'{len(xyz_guesses)} external TS guess(es) as CREST seed(s).'
                    )
            if not xyz_guesses:
                logger.warning(f'CREST TS search failed to generate any seed guesses for {rxn.label}.')
                tsg.tok()
                continue

            if len(xyz_guesses) > MAX_CREST_SEEDS:
                logger.info(
                    f'The CREST seed generation produced {len(xyz_guesses)} seeds for {rxn.label}. '
                    f'Submitting only the {MAX_CREST_SEEDS} most geometrically distinct of them and '
                    f'dropping the other {len(xyz_guesses) - MAX_CREST_SEEDS}, so that a single reaction '
                    f'cannot occupy the queue with an unbounded number of concurrent CREST jobs. Raise '
                    f'MAX_CREST_SEEDS to sample more of them.'
                )
                xyz_guesses = _select_diverse_seeds(seeds=xyz_guesses, max_seeds=MAX_CREST_SEEDS)

            for iteration, xyz_entry in enumerate(xyz_guesses):
                xyz_guess = xyz_entry.get("xyz")
                family = xyz_entry.get("family", rxn.family)
                if xyz_guess is None:
                    continue

                crest_constraints = get_wrapper_constraints(
                    wrapper='crest',
                    reaction=rxn,
                    seed=xyz_entry,
                )
                if not crest_constraints:
                    logger.warning(
                        f"Could not determine CREST constraint atoms for {rxn.label} crest seed {iteration} "
                        f"(family: {family}). Skipping this CREST seed."
                    )
                    continue

                crest_job_dir = crest_ts_conformer_search(
                    xyz_guess,
                    constraints=crest_constraints,
                    path=self.local_path,
                    xyz_crest_int=f'{rxn_index}_{iteration}',
                    charge=rxn.charge if rxn.charge is not None else 0,
                    multiplicity=rxn.multiplicity if rxn.multiplicity is not None else 1,
                )
                if crest_job_dir is None:
                    break
                crest_job_dirs.append(crest_job_dir)
                crest_references[crest_job_dir] = {
                    'xyz': xyz_guess,
                    'constraints': crest_constraints,
                }

            if not crest_job_dirs:
                logger.warning(f'CREST TS search failed to prepare any jobs for {rxn.label}.')
                tsg.tok()
                continue

            crest_jobs = submit_crest_jobs(crest_job_dirs)
            monitor_crest_jobs(crest_jobs)
            xyz_guesses_crest = process_completed_jobs(crest_jobs, crest_references=crest_references)
            tsg.tok()

            for method_index, xyz in enumerate(xyz_guesses_crest):
                if xyz is None:
                    continue
                unique = True
                for other_tsg in rxn.ts_species.ts_guesses:
                    if not other_tsg.success or other_tsg.initial_xyz is None:
                        continue
                    if almost_equal_coords(xyz, other_tsg.initial_xyz):
                        if hasattr(other_tsg, "method_sources"):
                            other_tsg.method_sources = other_tsg._normalize_method_sources(
                                (other_tsg.method_sources or []) + ["crest"]
                            )
                        unique = False
                        break
                if unique:
                    ts_guess = TSGuess(method='CREST',
                                       method_index=method_index,
                                       t0=tsg.t0,
                                       execution_time=tsg.execution_time,
                                       success=True,
                                       family=rxn.family,
                                       xyz=xyz,
                                       )
                    rxn.ts_species.append_ts_guess(ts_guess)
                    save_geo(xyz=xyz,
                             path=self.local_path,
                             filename=f'CREST_{method_index}',
                             format_='xyz',
                             comment=f'CREST {method_index}, family: {rxn.family}',
                             )

            if len(self.reactions) < 5:
                successes = [tsg for tsg in rxn.ts_species.ts_guesses if tsg.success and 'crest' in tsg.method.lower()]
                if successes:
                    logger.info(f'CREST successfully found {len(successes)} TS guesses for {rxn.label}.')
                else:
                    logger.info(f'CREST did not find any successful TS guesses for {rxn.label}.')

        self.final_time = datetime.datetime.now()

    def execute_queue(self):
        self.execute_incore()


def _crest_reactive_core_covers_molecule(rxn: 'ARCReaction') -> bool:
    """
    Return whether the CREST reactive core spans essentially the whole molecule.

    The reactive core is the set of atoms CREST constrains: the three-center A--H--B triad of an
    H_Abstraction, and the four-center a/b/o/h1 tetrahedron of a hydrolysis. When the TS has at
    most one atom outside that core there are no meaningful spectator conformational degrees of
    freedom for CREST metadynamics to sample, so CREST cannot improve on the heuristic seed and
    should be skipped (the other TS-search methods still cover the case).

    This is intentionally conservative: it only returns ``True`` for a family listed in
    ``CREST_REACTIVE_CORE_SIZES`` that has <=1 spectator atom, never for a system that retains
    real spectator degrees of freedom, and never for a family whose core size is unknown.

    Args:
        rxn (ARCReaction): The reaction to check.

    Returns:
        bool: Whether CREST should be skipped for this reaction.
    """
    reactive_core_size = CREST_REACTIVE_CORE_SIZES.get(getattr(rxn, 'family', None))
    if reactive_core_size is None:
        return False
    reactant_species = getattr(rxn, 'r_species', None) or []
    atom_counts = [getattr(spc, 'number_of_atoms', None) for spc in reactant_species]
    if not atom_counts or any(count is None for count in atom_counts):
        return False
    return (sum(atom_counts) - reactive_core_size) <= 1


def _select_diverse_seeds(seeds: List[dict], max_seeds: int) -> List[dict]:
    """
    Return at most ``max_seeds`` seed entries, preferring geometrically distinct ones.

    The first entry is always kept, and every further entry is the one whose smallest RMSD to the
    already selected entries is the largest, so that near-duplicate seeds are dropped first. The
    selected entries are returned in their original order. If the seed geometries cannot be compared
    (a missing or differently shaped coordinate array), the first ``max_seeds`` entries are returned
    instead.

    Args:
        seeds (List[dict]): Seed entries in the schema returned by ``get_ts_seeds()``.
        max_seeds (int): The maximum number of seed entries to return.

    Returns:
        List[dict]: The selected seed entries.
    """
    if max_seeds is None or max_seeds <= 0 or len(seeds) <= max_seeds:
        return list(seeds)
    coords = list()
    for seed in seeds:
        xyz = seed.get('xyz') if isinstance(seed, dict) else None
        entry = None
        if isinstance(xyz, dict) and xyz.get('coords'):
            entry = np.array(xyz['coords'], dtype=float)
        coords.append(entry)
    if any(entry is None or entry.shape != coords[0].shape for entry in coords):
        logger.debug('Could not compare the CREST seed geometries, keeping the first '
                     f'{max_seeds} seeds.')
        return list(seeds[:max_seeds])
    selected = [0]
    remaining = list(range(1, len(seeds)))
    while len(selected) < max_seeds and remaining:
        best_index, best_distance = remaining[0], -1.0
        for index in remaining:
            distance = min(calc_rmsd(coords[index], coords[chosen]) for chosen in selected)
            if distance > best_distance:
                best_index, best_distance = index, distance
        selected.append(best_index)
        remaining.remove(best_index)
    return [seeds[index] for index in sorted(selected)]


def crest_ts_conformer_search(
    xyz_guess: dict,
    a_atom: Optional[int] = None,
    h_atom: Optional[int] = None,
    b_atom: Optional[int] = None,
    path: str = "",
    xyz_crest_int: Union[int, str] = 0,
    constraints: Optional[dict] = None,
    charge: int = 0,
    multiplicity: int = 1,
) -> Optional[str]:
    """
    Prepare a CREST TS conformer search job.

    Writes ``coords.ref`` (TURBOMOLE format, Bohr, ``x y z SYMBOL`` per line), ``constraints.inp``
    and a PBS/HTCondor submit script based on ``submit_scripts['local']['crest']`` into a
    ``crest_<xyz_crest_int>`` sub-directory of ``path``, and returns that directory. A
    ``crest_best.xyz`` left behind by an earlier run in the same directory is removed first, so a
    run that dies on launch cannot have the previous run's geometry ingested as its own result.

    ``constraints`` holds zero-based ``atoms`` and ``distance_pairs``, and, for the three-center
    H-abstraction case, ``angle_atoms``. Every pair in ``distance_pairs`` is written as a
    ``distance`` restraint; for the H-abstraction case the heavy--heavy A--B separation is written
    as well. Those three distances rigidly determine the A--H--B triad, so no angle restraint is
    emitted. The four-center hydrolysis case supplies all six pairwise distances among its reactive
    atoms in ``distance_pairs`` and no ``angle_atoms``, which likewise determines its reactive core
    without an angle restraint. Atoms outside the reactive zone form the ``$metadyn`` atom list;
    when the reactive zone covers every atom no ``$metadyn`` block is written, since an atom-less
    block only restates the CREST default.

    The restraint stiffness is written as ``force constant=0.5``: xtb parses ``key=value`` in this
    block and silently ignores the colon form, falling back to its 0.05 default.

    ``--chrg`` and ``--uhf`` are appended to the CREST command for a charged or open-shell system;
    without them CREST samples the neutral closed-shell species, which is the wrong electronic state
    for the doublet TS of an H-abstraction.

    Args:
        xyz_guess (dict): The seed geometry.
        a_atom (int, optional): Legacy zero-based donor index, used only when ``constraints`` is None.
        h_atom (int, optional): Legacy zero-based transferred-H index, used only when ``constraints`` is None.
        b_atom (int, optional): Legacy zero-based acceptor index, used only when ``constraints`` is None.
        path (str, optional): The parent directory in which the CREST job directory is created.
        xyz_crest_int (Union[int, str], optional): The CREST job directory suffix, unique per seed.
        constraints (dict, optional): The CREST constraint specification.
        charge (int, optional): The net charge of the system.
        multiplicity (int, optional): The spin multiplicity of the system.

    Returns:
        Optional[str]: The CREST job directory path, or ``None`` if the local cluster software is
                       not supported.

    Raises:
        ValueError: If neither ``constraints`` nor a complete set of legacy A, H and B indices is given.
    """
    if constraints is None:
        if not all(isinstance(atom, int) for atom in (a_atom, h_atom, b_atom)):
            raise ValueError('CREST requires either a constraint specification or legacy A, H, and B atom indices.')
        constraints = {
            'atoms': (a_atom, h_atom, b_atom),
            'distance_pairs': ((a_atom, h_atom), (h_atom, b_atom)),
            'angle_atoms': (a_atom, h_atom, b_atom),
        }

    path = os.path.join(path, f"crest_{xyz_crest_int}")
    os.makedirs(path, exist_ok=True)

    stale_best_path = os.path.join(path, "crest_best.xyz")
    if os.path.isfile(stale_best_path):
        try:
            os.remove(stale_best_path)
        except OSError as e:
            logger.error(f"Could not remove the stale CREST geometry at {stale_best_path}, a failed CREST run "
                         f"in this directory could have it ingested as its own result: "
                         f"{e.__class__.__name__}: {e}")

    symbols = xyz_guess["symbols"]
    converted_coords = reorder_xyz_string(
        xyz_str=xyz_to_str(xyz_guess),
        reverse_atoms=True,
        convert_to="bohr",
    )
    coords_ref_content = f"$coord\n{converted_coords}\n$end\n"
    coords_ref_path = os.path.join(path, "coords.ref")
    with open(coords_ref_path, "w") as f:
        f.write(coords_ref_content)

    num_atoms = len(symbols)
    participating_atoms = tuple(atom + 1 for atom in constraints['atoms'])
    distance_pairs = tuple((atom_1 + 1, atom_2 + 1)
                           for atom_1, atom_2 in constraints['distance_pairs'])

    angle_atoms = constraints.get('angle_atoms')
    heavy_heavy_pair = None
    if angle_atoms is not None:
        heavy_heavy_pair = (angle_atoms[0] + 1, angle_atoms[2] + 1)

    list_of_atoms_numbers_not_participating_in_reaction = [
        i for i in range(1, num_atoms + 1) if i not in participating_atoms
    ]
    if not list_of_atoms_numbers_not_participating_in_reaction:
        logger.warning(f"All {num_atoms} atoms are in the CREST reactive zone, so no $metadyn atom list "
                       f"is written and CREST will bias every atom by default.")

    constraints_path = os.path.join(path, "constraints.inp")
    with open(constraints_path, "w") as f:
        f.write("$constrain\n")
        f.write(f"  atoms: {', '.join(map(str, participating_atoms))}\n")
        f.write("  force constant=0.5\n")
        f.write("  reference=coords.ref\n")
        for atom_1, atom_2 in distance_pairs:
            f.write(f"  distance: {atom_1}, {atom_2}, auto\n")
        if heavy_heavy_pair is not None:
            f.write(f"  distance: {heavy_heavy_pair[0]}, {heavy_heavy_pair[1]}, auto\n")
        if list_of_atoms_numbers_not_participating_in_reaction:
            f.write("$metadyn\n")
            f.write(
                f'  atoms: {", ".join(map(str, list_of_atoms_numbers_not_participating_in_reaction))}\n'
            )
        f.write("$end\n")

    local_server = SERVERS.get("local", {})
    cpus = int(local_server.get("cpus", 8))
    if CREST_ENV_PATH:
        crest_exe = "crest"
    else:
        crest_exe = CREST_PATH if CREST_PATH is not None else "crest"

    commands = [
        crest_exe,
        "coords.ref",
        "--cinp constraints.inp",
        "--noreftopo",
        f"-T {cpus}",
    ]
    if charge:
        commands.append(f"--chrg {charge}")
    if multiplicity and multiplicity > 1:
        commands.append(f"--uhf {multiplicity - 1}")
    command = " ".join(commands)

    activation_line = CREST_ENV_PATH or ""

    if SERVERS.get("local") is not None:
        cluster_soft = SERVERS["local"]["cluster_soft"].lower()
        local_templates = submit_scripts.get("local", {})
        crest_template = local_templates.get("crest")
        crest_job_template = local_templates.get("crest_job")

        if cluster_soft in ["condor", "htcondor"]:
            if crest_template is None:
                crest_template = (
                    "universe = vanilla\n"
                    "executable = job.sh\n"
                    "output = out.txt\n"
                    "error = err.txt\n"
                    "log = log.txt\n"
                    "request_cpus = {cpus}\n"
                    "request_memory = {memory}\n"
                    "JobBatchName = {name}\n"
                    "queue\n"
                )
            if crest_job_template is None:
                crest_job_template = (
                    "#!/bin/bash -l\n"
                    "{activation_line}\n"
                    'cd "{path}"\n'
                    "{commands}\n"
                )
            sub_job = crest_template
            format_params = {
                "name": f"crest_{xyz_crest_int}",
                "cpus": cpus,
                "memory": int(SERVERS["local"].get("memory", 32.0) * 1024),
            }
            sub_job = sub_job.format(**format_params)

            with open(
                os.path.join(path, settings["submit_filenames"]["HTCondor"]), "w"
            ) as f:
                f.write(sub_job)

            crest_job = crest_job_template.format(
                path=path,
                activation_line=activation_line,
                commands=command,
            )

            with open(os.path.join(path, "job.sh"), "w") as f:
                f.write(crest_job)
            os.chmod(os.path.join(path, "job.sh"), 0o700)

            for fname in ("out.txt", "err.txt"):
                fpath = os.path.join(path, fname)
                if not os.path.exists(fpath):
                    with open(fpath, "w") as f:
                        f.write("")
                    os.chmod(fpath, 0o600)

        elif cluster_soft == "pbs":
            if crest_template is None:
                crest_template = (
                    "#!/bin/bash -l\n"
                    "#PBS -q {queue}\n"
                    "#PBS -N {name}\n"
                    "#PBS -l select=1:ncpus={cpus}:mem={memory}gb\n"
                    "#PBS -o out.txt\n"
                    "#PBS -e err.txt\n\n"
                    "{activation_line}\n"
                    'cd "{path}"\n'
                    "{commands}\n"
                )
            sub_job = crest_template
            format_params = {
                "queue": SERVERS["local"].get("queue", "alon_q"),
                "name": f"crest_{xyz_crest_int}",
                "cpus": cpus,
                "memory": int(
                    SERVERS["local"].get("memory", 32)
                    if SERVERS["local"].get("memory", 32) < 60
                    else 40
                ),
                "activation_line": activation_line,
                "path": path,
                "commands": command,
            }
            sub_job = sub_job.format(**format_params)

            submit_filename = settings["submit_filenames"]["PBS"]
            submit_path = os.path.join(path, submit_filename)
            with open(submit_path, "w") as f:
                f.write(sub_job)
            os.chmod(submit_path, 0o700)

        else:
            logger.warning(f"CREST does not support the {cluster_soft!r} cluster software of the local server, "
                           f"skipping the CREST TS search.")
            return None

    return path


def submit_crest_jobs(crest_paths: List[str]) -> dict:
    """
    Submit CREST jobs to the server.

    A submission that yields no job id is skipped. ``submit_job()`` signals that either as
    ``(None, None)`` or as ``('errored', '')``, the latter whenever the submit command wrote to
    stderr or its stdout had an unrecognized shape, so the guard has to be on falsiness rather than
    on ``None``. Keying the returned dict on such a value would collapse every failed submission
    onto a single entry and lose the paths of all but the last one, and an empty-string key would in
    addition adopt the first line of the queue listing as its status, because
    ``check_job_status_in_stdout()`` matches a job id as a substring of the status line.

    Args:
        crest_paths (List[str]): List of paths to the CREST directories.

    Returns:
        dict: A dictionary keyed by job ID, holding the job path and status.
    """
    crest_jobs = {}
    for crest_path in crest_paths:
        job_status, job_id = submit_job(path=crest_path)
        if not job_id:
            logger.error(f"Could not submit a CREST job for {crest_path} (got job ID {job_id!r} and status "
                         f"{job_status!r}), skipping it.")
            continue
        logger.debug(f"CREST job {job_id} submitted for {crest_path}")
        crest_jobs[job_id] = {"path": crest_path, "status": job_status}
    if crest_jobs:
        job_ids = list(crest_jobs.keys())
        parent = os.path.dirname(crest_paths[0])
        logger.info(f"Submitted {len(job_ids)} CREST jobs ({job_ids[0]}-{job_ids[-1]}) for {parent}")
    return crest_jobs


def monitor_crest_jobs(crest_jobs: dict,
                       check_interval: int = 300,
                       max_time_seconds: float = DEFAULT_MAX_CREST_WALL_TIME,
                       ) -> None:
    """
    Monitor CREST jobs until they are complete.

    Polls every non-terminal job until all jobs reach a terminal status or ``max_time_seconds``
    elapses. On expiry each still-running job is cancelled with :func:`arc.job.local.delete_job` so
    that it does not keep occupying the queue unowned; a cancellation failure is logged and does not
    stop the remaining jobs from being cancelled. A cancelled job that had already written
    ``crest_best.xyz`` is marked ``'done'`` so that geometry is still ingested downstream, and one
    that wrote nothing is marked ``'errored'``.

    Args:
        crest_jobs (dict): Dictionary containing job information (job ID, path, and status).
        check_interval (int): Time interval (in seconds) to wait between status checks.
        max_time_seconds (float): The wall time after which the remaining jobs are given up on.
    """
    deadline = time.time() + max_time_seconds
    while True:
        all_done = True
        for job_id, job_info in crest_jobs.items():
            if job_info["status"] not in TERMINAL_JOB_STATUSES:
                try:
                    job_info["status"] = check_job_status(job_id)
                except Exception as e:
                    logger.error(f"Error checking job status for job {job_id}: {e}")
                    job_info["status"] = "errored"
                if job_info["status"] not in TERMINAL_JOB_STATUSES:
                    all_done = False
        if all_done:
            break
        if time.time() > deadline:
            unfinished = {job_id: info["status"] for job_id, info in crest_jobs.items()
                          if info["status"] not in TERMINAL_JOB_STATUSES}
            logger.error(f"CREST jobs did not finish within {max_time_seconds} s, cancelling them: {unfinished}.")
            for job_id, info in crest_jobs.items():
                if info["status"] in TERMINAL_JOB_STATUSES:
                    continue
                try:
                    delete_job(job_id)
                except Exception as e:
                    logger.error(f"Could not cancel the timed-out CREST job {job_id}, it may still be occupying "
                                 f"the queue: {e.__class__.__name__}: {e}")
                crest_path = info.get("path")
                crest_best_path = os.path.join(crest_path, "crest_best.xyz") if crest_path else None
                if crest_best_path is not None and os.path.isfile(crest_best_path):
                    info["status"] = "done"
                    logger.info(f"The timed-out CREST job {job_id} had already written {crest_best_path}, "
                                f"ingesting that geometry.")
                else:
                    info["status"] = "errored"
                    logger.info(f"The timed-out CREST job {job_id} wrote no geometry, dropping it.")
            break
        time.sleep(min(check_interval, MAX_CHECK_INTERVAL_SECONDS))


def process_completed_jobs(crest_jobs: dict, crest_references: dict) -> list:
    """
    Process the completed CREST jobs and update XYZ guesses.

    Only jobs with a ``'done'`` status are read. A geometry is rejected if it cannot be parsed
    (CREST killed mid-write leaves a truncated file, which must not abort ARC), is empty, holds a
    non-finite coordinate, has colliding atoms, has no reference seed recorded for its path, or no
    longer preserves the seed's constrained reactive zone.

    Args:
        crest_jobs (dict): Dictionary containing job information.
        crest_references (dict): Reference seed geometry and constraint specification, keyed by CREST path.

    Returns:
        list: The accepted CREST geometries.
    """
    xyz_guesses = []
    for job_id, job_info in crest_jobs.items():
        crest_path = job_info["path"]
        if job_info["status"] == "done":
            crest_best_path = os.path.join(crest_path, "crest_best.xyz")
            if os.path.exists(crest_best_path):
                with open(crest_best_path, "r") as f:
                    content = f.read()
                try:
                    xyz_guess = xyz_file_format_to_xyz(content)
                except (ConverterError, IndexError, ValueError) as e:
                    logger.warning(f"Could not parse the CREST geometry at {crest_best_path}, skipping it: "
                                   f"{e.__class__.__name__}: {e}")
                    continue
                if xyz_guess is None or not xyz_guess.get('symbols'):
                    logger.warning(f"The CREST geometry at {crest_best_path} is empty, skipping it.")
                    continue
                if any(not np.isfinite(coord) for xyz_row in xyz_guess['coords'] for coord in xyz_row):
                    logger.warning(f"The CREST geometry at {crest_best_path} contains non-finite coordinates, "
                                   f"skipping it.")
                    continue
                if colliding_atoms(xyz_guess):
                    logger.warning(f"Rejecting the CREST geometry from {crest_path}: it has colliding atoms.")
                    continue
                reference = crest_references.get(crest_path)
                if reference is None:
                    logger.warning(f"Rejecting unvalidated CREST geometry from {crest_path}: reference data is missing.")
                    continue
                if not _preserves_reactive_constraints(
                    xyz=xyz_guess,
                    reference_xyz=reference['xyz'],
                    constraints=reference['constraints'],
                ):
                    logger.warning(
                        f"Rejecting CREST geometry from {crest_path}: it does not preserve the reactive constraints."
                    )
                    continue
                xyz_guesses.append(xyz_guess)
            else:
                logger.error(f"crest_best.xyz not found in {crest_path}")
        elif job_info["status"] == "failed":
            logger.error(f"CREST job failed for {crest_path}")

    return xyz_guesses


def _preserves_reactive_constraints(xyz: dict, reference_xyz: dict, constraints: dict) -> bool:
    """Return whether a CREST geometry preserves the seed's constrained reactive zone."""
    symbols = xyz.get('symbols') if isinstance(xyz, dict) else None
    reference_symbols = reference_xyz.get('symbols') if isinstance(reference_xyz, dict) else None
    if not symbols or symbols != reference_symbols or not isinstance(constraints, dict):
        return False
    try:
        atoms = tuple(constraints['atoms'])
        distance_pairs = tuple(tuple(pair) for pair in constraints['distance_pairs'])
        if (not atoms or len(set(atoms)) != len(atoms)
                or any(not isinstance(atom, int) or not 0 <= atom < len(symbols) for atom in atoms)
                or any(len(pair) != 2 or pair[0] not in atoms or pair[1] not in atoms
                       for pair in distance_pairs)):
            return False
        for atom_1, atom_2 in distance_pairs:
            distance = math.dist(xyz['coords'][atom_1], xyz['coords'][atom_2])
            reference_distance = math.dist(reference_xyz['coords'][atom_1], reference_xyz['coords'][atom_2])
            if abs(distance - reference_distance) > max(0.5, reference_distance * 0.5):
                return False
        angle_atoms = constraints.get('angle_atoms')
        if angle_atoms is not None:
            atom_1, vertex, atom_2 = angle_atoms
            angle = _get_angle(xyz['coords'][atom_1], xyz['coords'][vertex], xyz['coords'][atom_2])
            reference_angle = _get_angle(
                reference_xyz['coords'][atom_1],
                reference_xyz['coords'][vertex],
                reference_xyz['coords'][atom_2],
            )
            if abs(angle - reference_angle) > 30.0:
                return False
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    return True


def _get_angle(point_1: tuple, vertex: tuple, point_2: tuple) -> float:
    """Return the angle in degrees for ``point_1``--``vertex``--``point_2``."""
    vector_1 = tuple(coord - center for coord, center in zip(point_1, vertex))
    vector_2 = tuple(coord - center for coord, center in zip(point_2, vertex))
    norm_product = math.sqrt(sum(coord ** 2 for coord in vector_1) * sum(coord ** 2 for coord in vector_2))
    if not norm_product:
        raise ValueError('Cannot determine an angle from a zero-length vector.')
    cosine = sum(coord_1 * coord_2 for coord_1, coord_2 in zip(vector_1, vector_2)) / norm_product
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

register_job_adapter('crest', CrestAdapter)
