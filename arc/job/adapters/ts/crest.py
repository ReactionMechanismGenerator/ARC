"""
Utilities for running CREST within ARC.

Separated from heuristics so CREST can be conditionally imported and reused.

``TERMINAL_JOB_STATUSES`` lists every status after which a job is no longer polled, and
``FAILED_JOB_STATUSES`` the terminal statuses :func:`process_completed_jobs` logs as failures.
``UNKNOWN_JOB_STATUS`` is what :func:`check_crest_job_status` reports when the queue could not be listed, and
``UNCONFIRMED_DONE_STATUS`` what it reports for a job absent from an empty listing. A job is taken to be
``'done'`` from empty listings only once ``EMPTY_LISTING_CONFIRMATIONS`` of them were seen without a non-empty
listing in between. After a cancellation the queue is listed up to ``CANCEL_CONFIRMATION_POLLS`` times,
``CANCEL_CONFIRMATION_INTERVAL_SECONDS`` apart, until the cancelled job is absent from it, since a cancelled
job that was running stays listed while it exits.

``MAX_CREST_SEEDS`` is the largest number of CREST jobs a single reaction spawns; each seed becomes one
multi-core metadynamics job, and the adapter waits for the jobs of a reaction in core.
``MAX_CREST_WALL_TIME_SECONDS`` caps, together with the adapter's ``max_job_time``, the wall time a
CREST job may run, counted from the moment it is first observed to have started, before it is
cancelled. A job starts by writing ``CREST_STARTED_MARKER`` into its directory.
``MAX_CREST_QUEUE_WAIT_SECONDS`` is the time a submitted job may remain unstarted (e.g., waiting in
the queue) before it is cancelled and marked ``'errored'``. Every non-terminal job is subject to
one of the two bounds, so the wait for a reaction's jobs always ends.

``CREST_REACTIVE_CORE_SIZES`` is the number of atoms CREST holds rigid for each family it supports: the
A--H--B triad of an H-abstraction, the *1/*2/*3/*4 ring of a four-centre XY addition, and the a/b/o/h1
tetrahedron of a hydrolysis. A family that is absent is never gated off by
:func:`_crest_reactive_core_covers_molecule`.

``MAX_RESTRAINED_DISTANCE_DEVIATION`` (Angstrom) is the largest deviation from the seed geometry a
completed CREST geometry may show in a restrained distance before it is rejected.

``SUPPORTED_CLUSTER_SOFTWARE`` maps the cluster software names of a local server CREST can submit to (PBS and
HTCondor) to their canonical names. ``CREST_JOB_ID_FILE`` is the file in a CREST job directory that holds the id of
the job submitted from it, from submission until the job is no longer polled or, for a cancelled job, until a queue
listing confirms it left the queue.

``CREST_PATH``, ``CREST_ENV_PATH`` and ``CREST_STANDALONE_DIR`` are read from the settings; when neither
of the first two is configured the executable is located on first use, see :func:`resolve_crest_paths`.
"""

import datetime
import glob
import math
import os
import time

import numpy as np
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from arc.common import almost_equal_coords, calc_rmsd, get_logger
from arc.exceptions import ConverterError, SettingsError, VectorsError
from arc.imports import settings, submit_scripts
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter, ts_adapters_by_rmg_family
from arc.job.adapters.ts.heuristics import DIHEDRAL_INCREMENT
from arc.job.adapters.ts.seed_hub import get_backup_ts_seeds, get_ts_seeds, get_wrapper_constraints
from arc.job.factory import register_job_adapter
from arc.job.local import delete_job, execute_command, submit_job
from arc.job.ssh import check_job_status_in_stdout
from arc.plotter import save_geo
from arc.settings.crest import find_xtb_executable, get_crest_paths, get_path_activation_line
from arc.species.converter import xyz_file_format_to_xyz, xyz_to_dmat, xyz_to_turbomol_format
from arc.species.species import ARCSpecies, TSGuess, colliding_atoms
from arc.species.vectors import calculate_distance

if TYPE_CHECKING:
    from arc.level import Level
    from arc.reaction import ARCReaction

logger = get_logger()

MAX_CHECK_INTERVAL_SECONDS = 100
TERMINAL_JOB_STATUSES = ('done', 'errored', 'failed')
FAILED_JOB_STATUSES = ('errored', 'failed')
UNKNOWN_JOB_STATUS = 'unknown'
UNCONFIRMED_DONE_STATUS = 'unconfirmed_done'
EMPTY_LISTING_CONFIRMATIONS = 3
CANCEL_CONFIRMATION_POLLS = 6
CANCEL_CONFIRMATION_INTERVAL_SECONDS = 10
MAX_CREST_WALL_TIME_SECONDS = 2 * 60 * 60
MAX_CREST_QUEUE_WAIT_SECONDS = 12 * 60 * 60
MAX_CREST_SEEDS = 8
MAX_RESTRAINED_DISTANCE_DEVIATION = 0.15
DEFAULT_CREST_MEMORY_GB = 14.0
CREST_STARTED_MARKER = 'crest.started'
SUPPORTED_CLUSTER_SOFTWARE = {'condor': 'HTCondor', 'htcondor': 'HTCondor', 'pbs': 'PBS'}
CREST_JOB_ID_FILE = 'crest.job_id'

CREST_REACTIVE_CORE_SIZES = {'H_Abstraction': 3,
                             'XY_Addition_MultipleBond': 4,
                             'carbonyl_based_hydrolysis': 4,
                             'ether_hydrolysis': 4,
                             'nitrile_hydrolysis': 4,
                             }

CREST_PATH = settings.get("CREST_PATH", None)
CREST_ENV_PATH = settings.get("CREST_ENV_PATH", None)
CREST_STANDALONE_DIR = settings.get("CREST_STANDALONE_DIR", None)
SERVERS = settings.get("servers", {})


def resolve_crest_paths() -> Tuple[Optional[str], Optional[str]]:
    """
    Return the CREST executable path and the shell line activating its environment.

    The configured ``CREST_PATH`` and ``CREST_ENV_PATH`` are returned as they are when either is set.
    Otherwise the filesystem is searched on the first call, under ``CREST_STANDALONE_DIR``, the common
    conda/mamba ``crest_env`` locations and ``PATH``, and the result is cached by
    :func:`arc.settings.crest.get_crest_paths`.

    Returns:
        Tuple[Optional[str], Optional[str]]: The executable path and the activation line,
                                             ``(None, None)`` if CREST could not be located.
    """
    if CREST_PATH or CREST_ENV_PATH:
        return CREST_PATH, CREST_ENV_PATH
    return get_crest_paths(standalone_dir=CREST_STANDALONE_DIR)


def get_cluster_soft_setting(setting_name: str, cluster_soft: str) -> Optional[str]:
    """
    Return the entry a per-cluster-software setting, e.g. ``settings['submit_filenames']``, holds for a
    cluster software.

    ``cluster_soft`` is matched case-insensitively, and ``'condor'`` is an alias of ``'HTCondor'``.

    Args:
        setting_name (str): The name of the setting, a dictionary keyed by cluster software.
        cluster_soft (str): The cluster software of the local server.

    Returns:
        Optional[str]: The entry, ``None`` if the setting holds none for this cluster software.
    """
    canonical = SUPPORTED_CLUSTER_SOFTWARE.get(cluster_soft.lower(), cluster_soft).lower()
    entries = settings.get(setting_name) or {}
    for key, value in entries.items():
        if isinstance(key, str) and key.lower() == canonical:
            return value
    return None


def get_submit_filename(cluster_soft: str) -> Optional[str]:
    """
    Return the submit file name configured for a cluster software in ``settings['submit_filenames']``.

    Args:
        cluster_soft (str): The cluster software of the local server, matched as by :func:`get_cluster_soft_setting`.

    Returns:
        Optional[str]: The submit file name, ``None`` if the settings hold none for this cluster software.
    """
    return get_cluster_soft_setting('submit_filenames', cluster_soft)


def crest_job_started(crest_path: Optional[str]) -> bool:
    """
    Return whether a CREST job has begun executing, i.e., whether it wrote ``CREST_STARTED_MARKER``.

    Args:
        crest_path (str, optional): The CREST job directory.

    Returns:
        bool: Whether the job has started.
    """
    return bool(crest_path) and os.path.isfile(os.path.join(crest_path, CREST_STARTED_MARKER))


def crest_available() -> bool:
    """
    Return whether CREST is configured for use.

    CREST needs a local server whose ``cluster_soft`` is spelled ``'PBS'`` or ``'HTCondor'``, the keys of the
    submission settings, and an executable
    path or an environment activation line from :func:`resolve_crest_paths`. A crest executable that is run without
    an environment activation line also needs an xtb executable, next to it or on ``PATH``, see
    :func:`arc.settings.crest.find_xtb_executable`. The reason CREST is not usable is logged, except for a missing
    local server or executable. When both ``CREST_PATH`` and ``CREST_ENV_PATH`` are set, that ``CREST_ENV_PATH``
    is used is logged.

    Returns:
        bool: Whether CREST can be used.
    """
    local_server = SERVERS.get("local")
    if not local_server:
        return False
    cluster_soft = local_server.get("cluster_soft")
    if not isinstance(cluster_soft, str) or SUPPORTED_CLUSTER_SOFTWARE.get(cluster_soft.lower()) != cluster_soft:
        logger.warning(f"CREST is not used: it is only supported on a local server whose cluster_soft is 'PBS' or "
                       f"'HTCondor', and the local server's cluster_soft is {cluster_soft!r}.")
        return False
    crest_path, crest_env_path = resolve_crest_paths()
    if not crest_path and not crest_env_path:
        return False
    if CREST_PATH and CREST_ENV_PATH:
        logger.info(f"Both CREST_PATH ({CREST_PATH}) and CREST_ENV_PATH ({CREST_ENV_PATH}) are set. CREST jobs "
                    f"activate CREST_ENV_PATH and run the crest it provides; CREST_PATH is not used.")
    if not crest_env_path and find_xtb_executable(crest_path) is None:
        logger.warning(f"CREST is not used: no xtb executable was found next to {crest_path} or on PATH, "
                       f"and CREST calls xtb.")
        return False
    return True


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

        CREST jobs left in the queue by an earlier execution for the same TS are cancelled first, see
        :func:`cancel_orphaned_crest_jobs`.

        For each reaction a seed list is built from the heuristics adapter, or, when that yields
        nothing, from the successful TS guesses other adapters already appended to
        ``rxn.ts_species.ts_guesses``, see :func:`arc.job.adapters.ts.seed_hub.get_backup_ts_seeds`;
        guesses whose method contains ``crest`` are excluded.

        The seed list is truncated to ``MAX_CREST_SEEDS`` entries, keeping the most geometrically
        distinct ones, and the number of dropped seeds is logged. Each surviving seed is submitted
        as one CREST job, and the jobs of a reaction are waited for synchronously, for at most
        :meth:`get_crest_wall_time_seconds`, before the results are ingested. The resulting
        geometries are appended as TS guesses unless they duplicate an existing successful guess,
        in which case that guess records ``'crest'`` as an additional method source. CREST is run
        without an explicit method flag, i.e., at its GFN2-xTB default.
        """
        self._log_job_execution()
        self.initial_time = self.initial_time if self.initial_time else datetime.datetime.now()

        supported_families = [key for key, val in ts_adapters_by_rmg_family.items() if 'crest' in val]

        self.reactions = [self.reactions] if not isinstance(self.reactions, list) else self.reactions
        if not crest_available():
            logger.warning('CREST is not available. Skipping CREST TS search.')
            self.final_time = datetime.datetime.now()
            return
        cancel_orphaned_crest_jobs(os.path.dirname(self.local_path))

        for rxn_index, rxn in enumerate(self.reactions):
            if rxn.family not in supported_families:
                logger.warning(f'The CREST TS search adapter does not support the {rxn.family} reaction family.')
                continue
            if any(spc.get_xyz() is None for spc in rxn.r_species + rxn.p_species):
                logger.warning(f'The CREST TS search adapter cannot process a reaction if 3D coordinates of '
                               f'some/all of its reactants/products are missing.\nNot processing {rxn}.')
                continue

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
                    queue=self.queue,
                    memory_gb=self.job_memory_gb,
                )
                if crest_job_dir is None:
                    continue
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
            monitor_crest_jobs(crest_jobs,
                               max_time_seconds=self.get_crest_wall_time_seconds(),
                               max_queue_wait_seconds=MAX_CREST_QUEUE_WAIT_SECONDS,
                               )
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

    def get_crest_wall_time_seconds(self) -> float:
        """
        Return the wall time, in seconds, after which the CREST jobs of a reaction are cancelled.

        This is the smaller of ``MAX_CREST_WALL_TIME_SECONDS`` and the adapter's ``max_job_time``
        (which is given in hours).

        Returns:
            float: The wall time in seconds.
        """
        max_job_time_seconds = self.max_job_time * 3600 if self.max_job_time else MAX_CREST_WALL_TIME_SECONDS
        return min(MAX_CREST_WALL_TIME_SECONDS, max_job_time_seconds)


def _crest_reactive_core_covers_molecule(rxn: 'ARCReaction') -> bool:
    """
    Return whether the CREST reactive core spans essentially the whole molecule.

    The reactive core is the set of atoms CREST constrains: the three-center A--H--B triad of an
    H_Abstraction, the four-center *1/*2/*3/*4 ring of an XY_Addition_MultipleBond, and the
    four-center a/b/o/h1 tetrahedron of a hydrolysis. When the TS has at most one atom outside that
    core there are no meaningful spectator conformational degrees of freedom for CREST metadynamics
    to sample, so CREST cannot improve on the heuristic seed and should be skipped (the other
    TS-search methods still cover the case).

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

    Seeds are compared through their interatomic distance matrices, so that a rigid translation or
    rotation of a geometry does not count as a distinct seed. The first entry is always kept, and
    every further entry is the one whose smallest distance-matrix RMSD to the already selected
    entries is the largest, so that near-duplicate seeds are dropped first. The selected entries are
    returned in their original order. If the seed geometries cannot be compared (a missing
    coordinate array or a different number of atoms), the first ``max_seeds`` entries are returned
    instead.

    Args:
        seeds (List[dict]): Seed entries in the schema returned by ``get_ts_seeds()``.
        max_seeds (int): The maximum number of seed entries to return.

    Returns:
        List[dict]: The selected seed entries.
    """
    if max_seeds is None or max_seeds <= 0 or len(seeds) <= max_seeds:
        return list(seeds)
    dmats = list()
    for seed in seeds:
        xyz = seed.get('xyz') if isinstance(seed, dict) else None
        dmat = xyz_to_dmat(xyz) if isinstance(xyz, dict) and xyz.get('coords') else None
        dmats.append(dmat)
    if any(dmat is None or dmat.shape != dmats[0].shape for dmat in dmats):
        logger.debug('Could not compare the CREST seed geometries, keeping the first '
                     f'{max_seeds} seeds.')
        return list(seeds[:max_seeds])
    selected = [0]
    remaining = list(range(1, len(seeds)))
    while len(selected) < max_seeds and remaining:
        best_index, best_distance = remaining[0], -1.0
        for index in remaining:
            distance = min(calc_rmsd(dmats[index], dmats[chosen]) for chosen in selected)
            if distance > best_distance:
                best_index, best_distance = index, distance
        selected.append(best_index)
        remaining.remove(best_index)
    return [seeds[index] for index in sorted(selected)]


def crest_ts_conformer_search(
    xyz_guess: dict,
    constraints: dict,
    path: str = "",
    xyz_crest_int: Union[int, str] = 0,
    charge: int = 0,
    multiplicity: int = 1,
    queue: Optional[str] = None,
    memory_gb: Optional[float] = None,
) -> Optional[str]:
    """
    Prepare a CREST TS conformer search job.

    Writes ``coords.ref`` (TURBOMOLE ``$coord`` format in Bohr, see
    :func:`arc.species.converter.xyz_to_turbomol_format`), ``constraints.inp``
    and a PBS/HTCondor submit script based on ``submit_scripts['local']['crest']`` into a
    ``crest_<xyz_crest_int>`` sub-directory of ``path``, and returns that directory. A
    ``crest_best.xyz`` or ``CREST_STARTED_MARKER`` left behind by an earlier run in the same
    directory is removed first, so a run that dies on launch cannot have the previous run's
    geometry ingested as its own result. If such a file exists and cannot be removed, ``None`` is
    returned and no job is prepared: :func:`process_completed_jobs` only tests whether
    ``crest_best.xyz`` exists, so leaving the stale file in place would let it be reported as this
    run's result. The job command first writes ``CREST_STARTED_MARKER`` and then runs CREST.

    The local server's ``cluster_soft`` is matched case-insensitively; ``None`` is returned, with
    a warning, if it is not one of ``SUPPORTED_CLUSTER_SOFTWARE`` or if
    ``settings['submit_filenames']`` holds no submit file name for it.

    ``constraints`` holds zero-based ``atoms`` and ``distance_pairs``, and, for the three-center
    H-abstraction case, ``angle_atoms``. Every pair in ``distance_pairs`` is written as a
    ``distance`` restraint; for the H-abstraction case the A--B separation of the two outer
    ``angle_atoms`` is written as well, and no angle restraint is emitted. These three distances do
    not hold a near-linear A--H--B triad rigidly: at 180 degrees the A--B distance is stationary in
    the angle, so a small change of it allows a large bend. Within the
    ``MAX_RESTRAINED_DISTANCE_DEVIATION`` acceptance tolerance of :func:`_preserves_reactive_constraints`,
    a seed with a linear C/N/O triad admits a bend to about 140 degrees with A--H and H--B at their seed
    lengths, and to about 115 degrees when they also stretch within the tolerance. The four-center
    hydrolysis and XY-addition cases supply all six pairwise distances among their reactive atoms in
    ``distance_pairs`` and no ``angle_atoms``. Atoms outside the reactive zone form the ``$metadyn``
    atom list; when the reactive zone covers every atom no ``$metadyn`` block is written.

    The restraint stiffness is written as ``force constant=0.5``, in the ``key=value`` form xtb reads in
    this block.

    ``--chrg`` with the net charge is appended to the CREST command for a charged system, and ``--uhf``
    with the number of unpaired electrons (``multiplicity - 1``) for an open-shell one.

    When :func:`resolve_crest_paths` gives an environment activation line, the job runs it and calls
    ``crest`` from that environment. Otherwise the job puts the directory of the crest executable first
    on ``PATH``, see :func:`arc.settings.crest.get_path_activation_line`, so that the xtb next to it is
    the one CREST calls, and runs the executable by its path.

    The submit script requests ``memory_gb`` of memory, capped at the local server's ``memory``
    when that is configured, and the first of the local server's ``queues`` unless ``queue`` is
    given. A PBS submit script without a known queue carries no ``#PBS -q`` directive.

    Args:
        xyz_guess (dict): The seed geometry.
        constraints (dict): The CREST constraint specification.
        path (str, optional): The parent directory in which the CREST job directory is created.
        xyz_crest_int (Union[int, str], optional): The CREST job directory suffix, unique per seed.
        charge (int, optional): The net charge of the system.
        multiplicity (int, optional): The spin multiplicity of the system.
        queue (str, optional): The queue to submit the job to.
        memory_gb (float, optional): The memory to request for the job in GB,
                                     ``DEFAULT_CREST_MEMORY_GB`` by default.

    Returns:
        Optional[str]: The CREST job directory path, or ``None`` if the directory holds a
                       ``CREST_JOB_ID_FILE`` naming an earlier job not confirmed to have left the queue,
                       a stale file could not be removed, the local cluster software is not supported,
                       or it has no submit file name.
    """
    path = os.path.join(path, f"crest_{xyz_crest_int}")
    os.makedirs(path, exist_ok=True)

    job_id_path = os.path.join(path, CREST_JOB_ID_FILE)
    if os.path.isfile(job_id_path):
        logger.error(f"The CREST job directory {path} holds {CREST_JOB_ID_FILE} of an earlier job that was not "
                     f"confirmed to have left the queue and may still write into it, skipping this CREST seed. "
                     f"A later run retries cancelling that job.")
        return None

    for stale_name in ("crest_best.xyz", CREST_STARTED_MARKER):
        stale_path = os.path.join(path, stale_name)
        if os.path.isfile(stale_path):
            try:
                os.remove(stale_path)
            except OSError as e:
                logger.error(f"Could not remove the stale CREST file at {stale_path}, skipping this CREST "
                             f"seed so that the previous run's geometry cannot be ingested as its own result: "
                             f"{e.__class__.__name__}: {e}")
                return None

    symbols = xyz_guess["symbols"]
    coords_ref_path = os.path.join(path, "coords.ref")
    with open(coords_ref_path, "w") as f:
        f.write(xyz_to_turbomol_format(xyz_guess, units="bohr"))

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
    memory = memory_gb if memory_gb else DEFAULT_CREST_MEMORY_GB
    server_memory = local_server.get("memory")
    if server_memory:
        memory = min(memory, server_memory)
    queue = queue or next(iter(local_server.get("queues", {})), None)
    crest_path, env_activation_line = resolve_crest_paths()
    activation_line = env_activation_line or get_path_activation_line(crest_path)
    crest_exe = "crest" if env_activation_line or crest_path is None else crest_path

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
    command = f"touch {CREST_STARTED_MARKER}\n" + " ".join(commands)

    if SERVERS.get("local") is not None:
        cluster_soft = SERVERS["local"]["cluster_soft"]
        if cluster_soft.lower() not in SUPPORTED_CLUSTER_SOFTWARE:
            logger.warning(f"CREST does not support the {cluster_soft!r} cluster software of the local server, "
                           f"skipping the CREST TS search.")
            return None
        submit_filename = get_submit_filename(cluster_soft)
        if submit_filename is None:
            logger.warning(f"The submit_filenames setting has no entry for the {cluster_soft!r} cluster software "
                           f"of the local server, skipping the CREST TS search.")
            return None
        cluster_soft = cluster_soft.lower()
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
                "memory": int(memory * 1024),
            }
            sub_job = sub_job.format(**format_params)

            with open(os.path.join(path, submit_filename), "w") as f:
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
                    + ("#PBS -q {queue}\n" if queue else "")
                    + "#PBS -N {name}\n"
                    "#PBS -l select=1:ncpus={cpus}:mem={memory}gb\n"
                    "#PBS -o out.txt\n"
                    "#PBS -e err.txt\n\n"
                    "{activation_line}\n"
                    'cd "{path}"\n'
                    "{commands}\n"
                )
            sub_job = crest_template
            format_params = {
                "queue": queue,
                "name": f"crest_{xyz_crest_int}",
                "cpus": cpus,
                "memory": int(math.ceil(memory)),
                "activation_line": activation_line,
                "path": path,
                "commands": command,
            }
            sub_job = sub_job.format(**format_params)

            submit_path = os.path.join(path, submit_filename)
            with open(submit_path, "w") as f:
                f.write(sub_job)
            os.chmod(submit_path, 0o700)

    return path


def submit_crest_jobs(crest_paths: List[str]) -> dict:
    """
    Submit CREST jobs to the server.

    A submission that raises a ``ValueError`` (the scheduler rejected it) or yields a falsy job id
    (``submit_job()`` returns ``(None, None)`` or ``('errored', '')`` for a failed submission) is
    logged and skipped. The id of every submitted job is written to ``CREST_JOB_ID_FILE`` in its
    directory.

    Args:
        crest_paths (List[str]): List of paths to the CREST directories.

    Returns:
        dict: A dictionary keyed by job ID, holding the job path, status and submission time
              (``'submitted'``, in seconds since the epoch).
    """
    crest_jobs = {}
    for crest_path in crest_paths:
        try:
            job_status, job_id = submit_job(path=crest_path)
        except ValueError as e:
            logger.error(f"The scheduler rejected the CREST job for {crest_path}, skipping it: {e}")
            continue
        if not job_id:
            logger.error(f"Could not submit a CREST job for {crest_path} (got job ID {job_id!r} and status "
                         f"{job_status!r}), skipping it.")
            continue
        logger.debug(f"CREST job {job_id} submitted for {crest_path}")
        with open(os.path.join(crest_path, CREST_JOB_ID_FILE), "w") as f:
            f.write(f"{job_id}\n")
        crest_jobs[job_id] = {"path": crest_path, "status": job_status, "submitted": time.time()}
    if crest_jobs:
        job_ids = list(crest_jobs.keys())
        parent = os.path.dirname(crest_paths[0])
        logger.info(f"Submitted {len(job_ids)} CREST jobs ({job_ids[0]}-{job_ids[-1]}) for {parent}")
    return crest_jobs


def cancel_orphaned_crest_jobs(species_path: str) -> None:
    """
    Cancel the CREST jobs an earlier execution submitted for a TS and stopped polling.

    Every ``CREST_JOB_ID_FILE`` in a CREST job directory under a job directory of ``species_path``
    (i.e., ``<species_path>/<job>/crest_<suffix>/``) names a job that was submitted and never reached a
    terminal status while polled. Each such job is cancelled with :func:`_cancel_crest_job`. The file is
    removed once the job is confirmed to have left the queue, including a job that finished before the
    cancellation, and when it names no job. Otherwise the file is kept, so a later call retries the
    cancellation, and :func:`crest_ts_conformer_search` does not reuse its directory.

    Args:
        species_path (str): The calcs directory of the TS species.
    """
    for job_id_path in sorted(glob.glob(os.path.join(species_path, "*", "crest_*", CREST_JOB_ID_FILE))):
        with open(job_id_path, "r") as f:
            job_id = f.read().strip()
        if job_id:
            logger.info(f"Cancelling the CREST job {job_id} that an earlier ARC run submitted from "
                        f"{os.path.dirname(job_id_path)} and stopped polling.")
            if not _cancel_crest_job(job_id):
                logger.warning(f"Keeping {job_id_path} of the orphaned CREST job {job_id} so a later run retries "
                               f"its cancellation.")
                continue
        os.remove(job_id_path)


def check_crest_job_status(job_id: str) -> str:
    """
    Return the status of a CREST job from a listing of the local server's queue.

    The queue is listed with the ``check_status_command`` of the local server's cluster software. A
    job found in the listing has the status :func:`arc.job.ssh.check_job_status_in_stdout` reads from
    its line, ``'running'`` or ``'errored'``. A job absent from a listing that holds any line is
    ``'done'``. ``UNCONFIRMED_DONE_STATUS`` is returned for a job absent from an empty listing, which
    an empty queue and a listing that silently returned nothing both produce. ``UNKNOWN_JOB_STATUS``
    is returned when the queue could not be listed or read: the command could not be run, it wrote
    to stderr and nothing to stdout, or its line for the job holds a status that is not classified.

    Args:
        job_id (str): The job id.

    Returns:
        str: The job status.
    """
    cluster_soft = SERVERS.get("local", {}).get("cluster_soft") or ""
    command = get_cluster_soft_setting("check_status_command", cluster_soft)
    if command is None:
        logger.error(f"The check_status_command setting has no entry for the {cluster_soft!r} cluster software.")
        return UNKNOWN_JOB_STATUS
    try:
        stdout, stderr = execute_command(command, no_fail=True)
    except (OSError, SettingsError) as e:
        logger.warning(f"Could not run the queue listing command to check the CREST job {job_id}: "
                       f"{e.__class__.__name__}: {e}")
        return UNKNOWN_JOB_STATUS
    if stdout is None or (not stdout and stderr):
        logger.warning(f"Could not list the queue to check the CREST job {job_id}: {stderr}")
        return UNKNOWN_JOB_STATUS
    if not stdout:
        return UNCONFIRMED_DONE_STATUS
    try:
        return check_job_status_in_stdout(job_id=job_id, stdout=stdout, server="local")
    except (IndexError, ValueError) as e:
        logger.warning(f"Could not read the status of the CREST job {job_id} from the queue listing: "
                       f"{e.__class__.__name__}: {e}")
        return UNKNOWN_JOB_STATUS


def _cancel_crest_job(job_id: str) -> bool:
    """
    Cancel a CREST job with :func:`arc.job.local.delete_job` and confirm that it left the queue.

    After a cancellation that raised no error, the queue is listed with :func:`check_crest_job_status`
    up to ``CANCEL_CONFIRMATION_POLLS`` times, ``CANCEL_CONFIRMATION_INTERVAL_SECONDS`` apart. The job is
    confirmed to have left the queue on the first listing that was read and does not hold it, i.e. whose
    status is ``'done'`` or ``UNCONFIRMED_DONE_STATUS``; a listing that could not be read
    (``UNKNOWN_JOB_STATUS``) or that still holds the job is followed by the next one. The job is left
    unconfirmed when the cancellation raised, which is logged as an error and not followed by any
    listing, or when every listing was used up: an error is logged when the last listing could not be
    read, and a warning when it still held the job.

    Args:
        job_id (str): The job id.

    Returns:
        bool: Whether the job was confirmed to have left the queue.
    """
    try:
        delete_job(job_id)
    except Exception as e:
        logger.error(f"Could not cancel the CREST job {job_id}, it may still be occupying "
                     f"the queue: {e.__class__.__name__}: {e}")
        return False
    status = UNKNOWN_JOB_STATUS
    for poll in range(CANCEL_CONFIRMATION_POLLS):
        if poll:
            time.sleep(CANCEL_CONFIRMATION_INTERVAL_SECONDS)
        status = check_crest_job_status(job_id)
        if status in ("done", UNCONFIRMED_DONE_STATUS):
            return True
    if status == UNKNOWN_JOB_STATUS:
        logger.error(f"Could not list the queue to confirm that the cancelled CREST job {job_id} left it, "
                     f"it may still be occupying the queue.")
    else:
        logger.warning(f"The cancelled CREST job {job_id} is still listed in the queue with the status {status!r} "
                       f"after {CANCEL_CONFIRMATION_POLLS} listings {CANCEL_CONFIRMATION_INTERVAL_SECONDS} s apart.")
    return False


def monitor_crest_jobs(crest_jobs: dict,
                       check_interval: int = MAX_CHECK_INTERVAL_SECONDS,
                       max_time_seconds: float = MAX_CREST_WALL_TIME_SECONDS,
                       max_queue_wait_seconds: Optional[float] = None,
                       ) -> None:
    """
    Monitor CREST jobs until they are complete.

    Polls every non-terminal job with :func:`check_crest_job_status`, every ``check_interval`` seconds
    and at most every ``MAX_CHECK_INTERVAL_SECONDS``, until all jobs reach a terminal status. A poll
    that reports ``UNKNOWN_JOB_STATUS`` leaves the job's status untouched. A job reported as
    ``UNCONFIRMED_DONE_STATUS`` keeps its status until it has been reported so on
    ``EMPTY_LISTING_CONFIRMATIONS`` polls without a non-empty listing in between, and is then
    ``'done'``. A job the listing reports as ``'errored'``, e.g. a held or suspended PBS job, is still
    in the queue and is cancelled.

    Two bounds apply to every non-terminal job. A job's run-time clock starts at the poll on which
    it is first observed to have started (see :func:`crest_job_started`) and is recorded under its
    ``'started'`` key; a job that has been running for more than ``max_time_seconds`` is given up
    on. A job that has not started is given up on once more than ``max_queue_wait_seconds``
    (``MAX_CREST_QUEUE_WAIT_SECONDS`` by default) have passed since its ``'submitted'`` time, which
    is recorded by :func:`submit_crest_jobs` and, for a job lacking it, set at the first poll.

    A job given up on is cancelled with :func:`_cancel_crest_job`; a job not confirmed to have left the
    queue is recorded as ``'cancel_failed'`` and does not stop the remaining jobs from being cancelled.
    A cancelled job that had already written ``crest_best.xyz`` is marked ``'done'`` so that geometry is
    still ingested downstream, and one that wrote nothing, including every job that never started, is
    marked ``'errored'``. Once all jobs are terminal, the ``CREST_JOB_ID_FILE`` files of those not
    recorded as ``'cancel_failed'`` are removed, so :func:`cancel_orphaned_crest_jobs` retries the others.

    Args:
        crest_jobs (dict): Dictionary containing job information (job ID, path, status and submission time).
        check_interval (int): Time interval (in seconds) to wait between status checks.
        max_time_seconds (float): The run time after which a started job is given up on.
        max_queue_wait_seconds (float, optional): The time after submission after which an unstarted
                                                  job is given up on.
    """
    if max_queue_wait_seconds is None:
        max_queue_wait_seconds = MAX_CREST_QUEUE_WAIT_SECONDS
    while True:
        all_done = True
        for job_id, job_info in crest_jobs.items():
            if job_info["status"] not in TERMINAL_JOB_STATUSES:
                _update_crest_job_status(job_id=job_id, job_info=job_info)
                if job_info.get("submitted") is None:
                    job_info["submitted"] = time.time()
                if job_info.get("started") is None and crest_job_started(job_info.get("path")):
                    job_info["started"] = time.time()
                if job_info["status"] not in TERMINAL_JOB_STATUSES:
                    all_done = False
        if all_done:
            break
        now = time.time()
        expired = {job_id: info for job_id, info in crest_jobs.items()
                   if info["status"] not in TERMINAL_JOB_STATUSES
                   and info.get("started") is not None
                   and now - info["started"] > max_time_seconds}
        never_started = {job_id: info for job_id, info in crest_jobs.items()
                         if info["status"] not in TERMINAL_JOB_STATUSES
                         and info.get("started") is None
                         and now - info["submitted"] > max_queue_wait_seconds}
        if expired:
            expired_statuses = {job_id: info["status"] for job_id, info in expired.items()}
            logger.error(f"CREST jobs did not finish within {max_time_seconds} s of starting, cancelling them: "
                         f"{expired_statuses}.")
        if never_started:
            logger.error(f"CREST jobs {list(never_started)} never started within {max_queue_wait_seconds} s of "
                         f"their submission, cancelling them.")
        for job_id, info in {**expired, **never_started}.items():
            if not _cancel_crest_job(job_id):
                info["cancel_failed"] = True
            crest_path = info.get("path")
            crest_best_path = os.path.join(crest_path, "crest_best.xyz") if crest_path else None
            if job_id in expired and crest_best_path is not None and os.path.isfile(crest_best_path):
                info["status"] = "done"
                logger.info(f"The timed-out CREST job {job_id} had already written {crest_best_path}, "
                            f"ingesting that geometry.")
            elif job_id in expired:
                info["status"] = "errored"
                logger.info(f"The timed-out CREST job {job_id} wrote no geometry, dropping it.")
            else:
                info["status"] = "errored"
                logger.info(f"The CREST job {job_id} never started, dropping it.")
        if (expired or never_started) and all(info["status"] in TERMINAL_JOB_STATUSES
                                              for info in crest_jobs.values()):
            break
        time.sleep(min(check_interval, MAX_CHECK_INTERVAL_SECONDS))
    for job_info in crest_jobs.values():
        job_id_path = os.path.join(job_info["path"], CREST_JOB_ID_FILE) if job_info.get("path") else None
        if job_id_path is not None and os.path.isfile(job_id_path) and not job_info.get("cancel_failed"):
            os.remove(job_id_path)


def _update_crest_job_status(job_id: str, job_info: dict) -> None:
    """
    Update the status of a non-terminal CREST job from one poll of :func:`check_crest_job_status`.

    ``UNKNOWN_JOB_STATUS`` leaves the status untouched. ``UNCONFIRMED_DONE_STATUS`` increments the
    job's ``'unconfirmed_polls'`` count, and sets the status to ``'done'`` once that count reaches
    ``EMPTY_LISTING_CONFIRMATIONS``; a status read from a non-empty listing resets the count. An
    ``'errored'`` job is cancelled with :func:`_cancel_crest_job` before its status is set, and is recorded
    as ``'cancel_failed'`` when it is not confirmed to have left the queue.

    Args:
        job_id (str): The job id.
        job_info (dict): The job's entry in the monitored jobs, updated in place.
    """
    status = check_crest_job_status(job_id)
    if status == UNKNOWN_JOB_STATUS:
        logger.warning(f"Keeping the status {job_info['status']!r} of the CREST job {job_id} and retrying on "
                       f"the next poll.")
        return
    if status == UNCONFIRMED_DONE_STATUS:
        job_info["unconfirmed_polls"] = job_info.get("unconfirmed_polls", 0) + 1
        if job_info["unconfirmed_polls"] >= EMPTY_LISTING_CONFIRMATIONS:
            job_info["status"] = "done"
        return
    job_info["unconfirmed_polls"] = 0
    if status == "errored":
        logger.error(f"The CREST job {job_id} is listed in an errored, held or suspended state, cancelling it.")
        if not _cancel_crest_job(job_id):
            job_info["cancel_failed"] = True
    job_info["status"] = status


def process_completed_jobs(crest_jobs: dict, crest_references: dict) -> list:
    """
    Process the completed CREST jobs and update XYZ guesses.

    Only jobs with a ``'done'`` status are read. A geometry is rejected if it cannot be parsed
    (CREST killed mid-write leaves a truncated file, which must not abort ARC), is empty, holds a
    non-finite coordinate, has colliding atoms, has no reference seed recorded for its path, or no
    longer preserves the seed's constrained reactive zone. A job whose status is one of
    ``FAILED_JOB_STATUSES`` is logged as an error and contributes no geometry.

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
                    logger.warning(f"Rejecting unvalidated CREST geometry from {crest_path}: "
                                   f"reference data is missing.")
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
        elif job_info["status"] in FAILED_JOB_STATUSES:
            logger.error(f"The CREST job {job_id} for {crest_path} ended with status "
                         f"{job_info['status']!r}, no geometry is taken from it.")

    return xyz_guesses


def _preserves_reactive_constraints(xyz: dict, reference_xyz: dict, constraints: dict) -> bool:
    """
    Return whether a CREST geometry preserves the seed's constrained reactive zone.

    Every restrained distance has to stay within ``MAX_RESTRAINED_DISTANCE_DEVIATION`` of its value
    in the seed. The restrained distances are the ``distance_pairs`` and, when ``angle_atoms`` is
    given, the separation of its two outer atoms, which :func:`crest_ts_conformer_search` restrains
    as well. The angle itself is not restrained and is not checked.

    Args:
        xyz (dict): The completed CREST geometry.
        reference_xyz (dict): The seed geometry.
        constraints (dict): The constraint specification the job was run with.

    Returns:
        bool: Whether the reactive zone is preserved.
    """
    symbols = xyz.get('symbols') if isinstance(xyz, dict) else None
    reference_symbols = reference_xyz.get('symbols') if isinstance(reference_xyz, dict) else None
    if not symbols or symbols != reference_symbols or not isinstance(constraints, dict):
        return False
    try:
        atoms = tuple(constraints['atoms'])
        distance_pairs = tuple(tuple(pair) for pair in constraints['distance_pairs'])
        angle_atoms = constraints.get('angle_atoms')
        if angle_atoms is not None:
            distance_pairs += ((angle_atoms[0], angle_atoms[2]),)
        if (not atoms or len(set(atoms)) != len(atoms)
                or any(not isinstance(atom, int) or not 0 <= atom < len(symbols) for atom in atoms)
                or any(len(pair) != 2 or pair[0] not in atoms or pair[1] not in atoms
                       for pair in distance_pairs)):
            return False
        for atom_1, atom_2 in distance_pairs:
            distance = calculate_distance(coords=xyz, atoms=[atom_1, atom_2])
            reference_distance = calculate_distance(coords=reference_xyz, atoms=[atom_1, atom_2])
            if abs(distance - reference_distance) > MAX_RESTRAINED_DISTANCE_DEVIATION:
                return False
    except (IndexError, KeyError, TypeError, ValueError, VectorsError):
        return False
    return True


register_job_adapter('crest', CrestAdapter)
