"""
This module contains functions which are shared across multiple Job modules.
As such, it should not import any other ARC modules to avoid circular imports.
"""

import datetime
import os
import shutil
import sys
import re

from pprint import pformat
from typing import TYPE_CHECKING

from arc.common import get_logger
from arc.imports import settings
from arc.level import Level

if TYPE_CHECKING:
    from arc.job.adapter import JobAdapter
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies


logger = get_logger()

default_job_settings, global_ess_settings, rotor_scan_resolution = \
    settings['default_job_settings'], settings['global_ess_settings'], settings['rotor_scan_resolution']

REFERENCE_AGNOSTIC_METHOD_TYPES = ['force_field', 'composite', 'semiempirical']

BROKEN_SYMMETRY_METHOD_TYPES = ['dft']
BROKEN_SYMMETRY_METHODS = ['hf', 'hf3c', 'rhf', 'uhf', 'rohf']
DOUBLE_HYBRID_METHODS = ['b2plyp', 'b2plypd', 'b2plypd3', 'b2plypd3bj', 'b2gpplyp', 'b2kplyp', 'b2tplyp',
                         'mpw2plyp', 'mpw2plypd', 'wb2plyp', 'wb2gpplyp',
                         'pbe0dh', 'pbe02', 'pbeqidh', 'pwpb95', 'ripwpb95', 'ptpss',
                         'dsdblyp', 'dsdblypd3', 'dsdpbep86', 'dsdpbep86d3', 'dsdpbeb95', 'dsdpbeb95d3',
                         'dsdpbepbe', 'dsdpbepbed3', 'revdsdpbep86', 'revdsdpbep86d3', 'revdsdpbeb95',
                         'dodblyp', 'dodpbep86', 'dodpbeb95',
                         'xyg3', 'xygjos', 'wb97x2', 'wb97m2']

DERIVED_UNRESTRICTED_VERDICT = 'external_instability'
SPIN_RELAXED_REFERENCE_PREFIX = 'U'
REFERENCE_CHANGE_AVAILABLE_KEY = 'reference_change_available'

ts_adapters_by_rmg_family = {'1+2_Cycloaddition': ['kinbot', 'goflow', 'rits', 'linear'],
                             '1,2_Insertion_CO': ['kinbot', 'goflow', 'rits', 'linear'],
                             '1,2_Insertion_carbene': ['kinbot', 'goflow', 'rits', 'linear'],
                             '1,2_NH3_elimination': ['goflow', 'rits', 'linear'],
                             '1,2_XY_interchange': ['orca_neb', 'goflow', 'rits', 'linear'],
                             '1,2_shiftC': ['gcn', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             '1,2_shiftS': ['gcn', 'kinbot', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             '1,3_Insertion_CO2': ['kinbot', 'goflow', 'rits', 'linear'],
                             '1,3_Insertion_ROR': ['kinbot', 'goflow', 'rits', 'linear'],
                             '1,3_Insertion_RSR': ['kinbot', 'goflow', 'rits', 'linear'],
                             '1,3_NH3_elimination': ['goflow', 'rits', 'linear'],
                             '1,3_sigmatropic_rearrangement': ['goflow', 'rits', 'linear'],
                             '1,4_Cyclic_birad_scission': ['gcn', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             '1,4_Linear_birad_scission': ['goflow', 'rits', 'linear'],
                             '2+2_cycloaddition': ['kinbot', 'goflow', 'rits', 'linear'],
                             '6_membered_central_C-C_shift': ['gcn', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'Baeyer-Villiger_step2': ['goflow', 'rits', 'linear'],
                             'Birad_recombination': ['goflow', 'rits', 'linear'],
                             'Concerted_Intra_Diels_alder_monocyclic_1,2_shiftH': ['gcn', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'Cyclic_Ether_Formation': ['kinbot', 'goflow', 'rits', 'linear'],
                             'Cyclic_Thioether_Formation': ['goflow', 'rits', 'linear'],
                             'Cyclopentadiene_scission': ['gcn', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'Diels_alder_addition': ['kinbot', 'goflow', 'rits', 'linear'],
                             'Diels_alder_addition_Aromatic': ['goflow', 'rits', 'linear'],
                             'HO2_Elimination_from_PeroxyRadical': ['kinbot', 'goflow', 'rits', 'linear'],
                             'H_Abstraction': ['heuristics', 'autotst', 'crest'],
                             'Intra_2+2_cycloaddition_Cd': ['gcn', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'Intra_5_membered_conjugated_C=C_C=C_addition': ['gcn', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'Intra_Diels_alder_monocyclic': ['gcn', 'kinbot', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'Intra_Disproportionation': ['gcn', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'Intra_RH_Add_Endocyclic': ['gcn', 'kinbot', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'Intra_RH_Add_Exocyclic': ['gcn', 'kinbot', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'Intra_R_Add_Endocyclic': ['gcn', 'kinbot', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'Intra_R_Add_ExoTetCyclic': ['kinbot', 'goflow', 'rits', 'linear'],
                             'Intra_R_Add_Exo_scission': ['gcn', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'Intra_R_Add_Exocyclic': ['gcn', 'kinbot', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'Intra_Retro_Diels_alder_bicyclic': ['kinbot', 'goflow', 'rits', 'linear'],
                             'Intra_ene_reaction': ['gcn', 'kinbot', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'Ketoenol': ['gcn', 'kinbot', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'Korcek_step1': ['gcn', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'Korcek_step2': ['kinbot', 'goflow', 'rits', 'linear'],
                             'R_Addition_COm': ['kinbot', 'goflow', 'rits', 'linear'],
                             'R_Addition_CSm': ['kinbot', 'goflow', 'rits', 'linear'],
                             'R_Addition_MultipleBond': ['autotst', 'kinbot', 'goflow', 'rits', 'linear'],
                             'Retroene': ['kinbot', 'goflow', 'rits', 'linear'],
                             'Singlet_Carbene_Intra_Disproportionation': ['gcn', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'XY_Addition_MultipleBond': ['goflow', 'rits', 'linear'],
                             'XY_elimination_hydroxyl': ['goflow', 'rits', 'linear'],
                             'carbonyl_based_hydrolysis': ['heuristics'],
                             'ether_hydrolysis': ['heuristics'],
                             'halocarbene_recombination': ['goflow', 'rits', 'linear'],
                             'halocarbene_recombination_double': ['goflow', 'rits', 'linear'],
                             'intra_H_migration': ['autotst', 'gcn', 'kinbot', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'intra_NO2_ONO_conversion': ['gcn', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'intra_OH_migration': ['gcn', 'kinbot', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'intra_halogen_migration': ['goflow', 'rits', 'linear'],
                             'intra_substitutionCS_cyclization': ['goflow', 'rits', 'linear'],
                             'intra_substitutionCS_isomerization': ['gcn', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'intra_substitutionS_cyclization': ['goflow', 'rits', 'linear'],
                             'intra_substitutionS_isomerization': ['gcn', 'xtb_gsm', 'orca_neb', 'goflow', 'rits', 'linear'],
                             'lone_electron_pair_bond': ['goflow', 'rits', 'linear'],
                             'nitrile_hydrolysis': ['heuristics']
                             }

all_families_ts_adapters = []

# Adapters that may run on any unimolecular reaction when RMG fails to assign a
# family. These adapters must tolerate rxn.family being None or unknown.
ts_adapters_for_unknown_unimolecular = ['goflow', 'rits', 'linear']

adapters_that_do_not_require_a_level_arg = ['xtb', 'torchani', 'ase']

# Default is "queue", "pipe" will be called whenever needed. So just list 'incore'.
default_incore_adapters = ['autotst', 'crest', 'gcn', 'goflow', 'heuristics', 'kinbot', 'linear', 'openbabel',
                           'psi4', 'rits', 'torchani', 'xtb', 'xtb_gsm']


def _initialize_adapter(obj: JobAdapter,
                        is_ts: bool,
                        project: str,
                        project_directory: str,
                        job_type: list[str] | str,
                        args: dict | None = None,
                        bath_gas: str | None = None,
                        checkfile: str | None = None,
                        conformer: int | None = None,
                        constraints: list[tuple[list[int], float]] | None = None,
                        cpu_cores: str | None = None,
                        dihedral_increment: float | None = None,
                        dihedrals: list[float] | None = None,
                        directed_scan_type: str | None = None,
                        ess_settings: dict | None = None,
                        ess_trsh_methods: list[str] | None = None,
                        fine: bool = False,
                        initial_time: datetime.datetime | str | None = None,
                        irc_direction: str | None = None,
                        job_id: int | None = None,
                        job_memory_gb: float = 14.0,
                        job_name: str | None = None,
                        job_num: int | None = None,
                        job_server_name: str | None = None,
                        job_status: list[dict | str] | None = None,
                        level: Level | None = None,
                        max_job_time: float | None = None,
                        run_multi_species: bool = False,
                        reactions: list[ARCReaction] | None = None,
                        rotor_index: int | None = None,
                        server: str | None = None,
                        server_nodes: list | None = None,
                        queue: str | None = None,
                        attempted_queues: list[str] | None = None,
                        species: list[ARCSpecies] | None = None,
                        testing: bool = False,
                        times_rerun: int = 0,
                        torsions: list[list[int]] | None = None,
                        tsg: int | None = None,
                        xyz: dict | list[dict] | None = None,
                        ):
    """
    A common Job adapter initializer function.
    """
    if not is_ts and any(arg is None for arg in [job_type, project, project_directory]):
        raise ValueError(f'All of the following arguments must be given:\n'
                         f'job_type, level, project, project_directory\n'
                         f'Got: {job_type}, {level}, {project}, {project_directory}, respectively')
    if not is_ts and obj.job_adapter not in adapters_that_do_not_require_a_level_arg and level is None:
        raise ValueError(f'A `level` argument must be given')

    obj.project = project
    obj.project_directory = project_directory
    if obj.project_directory:
        # exist_ok rather than a prior isdir() check: another adapter being initialized concurrently can
        # create this directory between the check and the call, which raised FileExistsError.
        os.makedirs(obj.project_directory, exist_ok=True)

    obj.additional_job_info = None
    obj.args = args or dict()
    obj.bath_gas = bath_gas
    obj.checkfile = obj.readable_checkfile(checkfile)
    obj.conformer = conformer
    obj.constraints = constraints or list()
    obj.cpu_cores = cpu_cores
    obj.dihedral_increment = dihedral_increment
    obj.dihedrals = dihedrals
    obj.directed_scan_type = directed_scan_type
    obj.ess_settings = ess_settings or global_ess_settings
    obj.ess_trsh_methods = ess_trsh_methods or list()
    obj.files_to_download = list()
    obj.files_to_upload = list()
    obj.final_time = None
    obj.fine = fine
    obj.initial_time = datetime.datetime.strptime(initial_time.split('.')[0], '%Y-%m-%d %H:%M:%S') \
        if isinstance(initial_time, str) else initial_time
    obj.input_file_memory = None
    obj.irc_direction = irc_direction
    obj.iterate_by = list()
    obj.job_id = job_id
    obj.job_memory_gb = job_memory_gb
    obj.job_name = job_name
    obj.job_num = job_num
    obj.job_server_name = job_server_name
    obj.job_status = job_status \
                     or ['initializing', {'status': 'initializing', 'keywords': list(), 'error': '', 'line': ''}]
    obj.job_type = job_type if isinstance(job_type, str) else job_type[0]  # always a string
    obj.job_types = job_type if isinstance(job_type, list) else [job_type]  # always a list
    # When restarting ARC and re-setting the jobs, ``level`` is a string, convert it to a Level object instance
    obj.level = Level(repr=level) if not isinstance(level, Level) and level is not None else level
    obj.max_job_time = max_job_time or default_job_settings.get('job_time_limit_hrs', 120)
    obj.run_multi_species = run_multi_species
    obj.number_of_processes = 0
    obj.reactions = [reactions] if reactions is not None and not isinstance(reactions, list) else reactions
    obj.remote_path = None
    obj.remote_project_path = None
    obj.restarted = bool(job_num)  # If job_num was given, this is a restarted job, don't save as initiated jobs.
    obj.rotor_index = rotor_index
    obj.run_time = None
    obj.server = server
    obj.queue = queue
    obj.attempted_queues = attempted_queues or list()
    obj.server_nodes = server_nodes or list()
    obj.species = [species] if species is not None and not isinstance(species, list) else species
    obj.submit_script_memory = None
    obj.submit_script_memory_mib = None
    obj.testing = testing
    obj.times_rerun = times_rerun
    obj.torsions = [torsions] if torsions is not None and not isinstance(torsions[0], list) else torsions
    obj.pivots = [[tor[1] + 1, tor[2] + 1] for tor in obj.torsions] if obj.torsions is not None else None
    obj.tsg = tsg
    obj.workers = None
    if not obj.run_multi_species:
        obj.xyz = obj.species[0].get_xyz() if obj.species is not None and xyz is None else xyz
    else:
        obj.xyz = list()
        if obj.species is not None:
            for spc in obj.species:
                obj.xyz.append(spc.get_xyz() if xyz is None else xyz)

    if obj.job_num is None or obj.job_name is None or obj.job_server_name:
        obj._set_job_number()

    if obj.species is not None:
        if not obj.run_multi_species:
            obj.charge = obj.species[0].charge
            obj.multiplicity = obj.species[0].multiplicity
            obj.is_ts = obj.species[0].is_ts
            obj.species_label = obj.species[0].label
            if len(obj.species) > 1:
                obj.species_label += f'_and_{len(obj.species) - 1}_others'
        else:
            obj.charge = list()
            obj.multiplicity = list()
            obj.is_ts = obj.species[0].is_ts
            obj.species_label = list()
            for spc in obj.species:
                obj.charge.append(spc.charge)
                obj.multiplicity.append(spc.multiplicity)
                obj.species_label.append(spc.label)
    elif obj.reactions is not None:
        obj.charge = obj.reactions[0].charge
        obj.multiplicity = obj.reactions[0].multiplicity
        obj.is_ts = True
        obj.species_label = obj.reactions[0].ts_species.label if obj.reactions[0].ts_species is not None \
            else f'TS_{obj.job_num}'
        if len(obj.reactions) > 1:
            obj.species_label += f'_and_{len(obj.reactions) - 1}_others'
    else:
        obj.charge = None
        obj.multiplicity = None
        obj.is_ts = None
        obj.species_label = None

    obj.args = set_job_args(args=obj.args, level=obj.level, job_name=obj.job_name)
    if obj.execution_type != 'incore' and obj.job_adapter in obj.ess_settings.keys() and obj.server is None:
        obj.server = resolve_job_server(ess_settings=obj.ess_settings,
                                        job_adapter=obj.job_adapter,
                                        args=obj.args)

    obj.set_file_paths()
    obj.set_cpu_and_mem()

    # Set scan_res if required by trsh
    if obj.args and 'trsh' in obj.args.keys() and 'scan_res' in obj.args['trsh'].keys():
        obj.scan_res = obj.args['trsh']['scan_res']
        # Remove it from the args dict
        obj.args['trsh'].pop('scan_res')
    else:
        obj.scan_res = rotor_scan_resolution

    obj.set_files()
    check_argument_consistency(obj)


def resolve_job_server(ess_settings: dict,
                       job_adapter: str,
                       args: dict | None = None,
                       ) -> str | None:
    """
    Return the server that a ``job_adapter`` job will be submitted to.

    A troubleshooting override in ``args['trsh']['server']`` wins, since it is set precisely to
    move a job off the server that failed it, and it is honoured even when it is empty so that
    the caller sees the same server the job itself would be given. Otherwise the server is the
    first one the ESS settings name for the adapter, which is the entry ARC submits to; a bare
    string there is read as a single server.

    Args:
        ess_settings (dict): The ESS settings, mapping an adapter to the server or the list of
                             servers it is available on.
        job_adapter (str): The job adapter to resolve a server for.
        args (dict, optional): The job's arguments, whose ``'trsh'`` entry may carry a
                               ``'server'`` override.

    Returns: str | None
        The server name, or ``None`` when neither an override nor the ESS settings name one.
    """
    trsh_args = (args or dict()).get('trsh') or dict()
    if isinstance(trsh_args, dict) and 'server' in trsh_args:
        return trsh_args['server']
    servers_for_adapter = (ess_settings or dict()).get(job_adapter)
    if isinstance(servers_for_adapter, str):
        return servers_for_adapter or None
    if isinstance(servers_for_adapter, (list, tuple)) and len(servers_for_adapter):
        return servers_for_adapter[0]
    return None


def is_restricted(obj: JobAdapter) -> bool | list[bool]:
    """
    Check whether a Job Adapter should be executed as restricted or unrestricted.
    If the job adapter contains a list of species, return True or False per species.

    The decision is also memoized on the job adapter as ``obj.restricted_used``, in the
    same shape it is returned in. Adapters call this while writing their input file, so
    the memo is the reference that job's input actually declared, and it is rewritten only
    when that input is rewritten. A consumer that recomputes the decision instead reports
    the reference the species would get today, which for a job that has already run is not
    the same question.

    The memo is written to the restart file by ``JobAdapter.as_dict()`` and restored by
    ``Scheduler.restore_running_jobs()`` after the adapter is rebuilt, because rebuilding
    it re-composes the input file and so calls this function again: without the restore, a
    job that was queued before a reference decision changed would come back from a restart
    carrying the reference it would be given now rather than the one it is running with.

    Args:
        obj: The job adapter object.

    Returns:
        bool | list[bool]: Whether to run as restricted (``True``) or not (``False``).
    """
    if not obj.run_multi_species:
        restricted = is_species_restricted(obj)
    else:
        restricted = [is_species_restricted(obj, species) for species in obj.species]
    obj.restricted_used = restricted
    return restricted


def job_scf_reference_is_restricted(obj: JobAdapter) -> bool | None:
    """
    Report the SCF reference a job declared in the input it ran, or ``None`` where it declared none.

    The value is read off the job adapter's ``restricted_used`` memo, which ``is_restricted()``
    writes while the input is being composed, so it is the reference that job actually ran with
    rather than the one the species would be given today. ``None`` is returned for a job carrying
    no memo, a pipe task among them, for a multi-species job, whose memo is a decision per species
    rather than a single one, and for the force field, composite and semiempirical method types,
    for which ARC writes no reference prefix and whose flag is therefore not a reference choice
    ARC made.

    Args:
        obj: The job adapter object.

    Returns:
        bool | None: Whether the job declared a restricted reference, or ``None`` if it declared none.
    """
    restricted = getattr(obj, 'restricted_used', None)
    if not isinstance(restricted, bool):
        return None
    level = getattr(obj, 'level', None)
    if level is None or level.method_type in REFERENCE_AGNOSTIC_METHOD_TYPES:
        return None
    return restricted


def level_admits_a_broken_symmetry_reference(level: Level | None) -> bool:
    """
    Check whether a broken-symmetry SCF reference describes what a level computes.

    A level whose energy IS the energy of its SCF determinant admits one. The determinant is the
    whole description there, so relaxing its spin symmetry onto the lower solution lowers the
    number the level reports, and a broken-symmetry determinant is the standard single-reference
    description of a species whose restricted determinant is not the ground state. Density
    functional theory and Hartree-Fock are those levels, which
    ``BROKEN_SYMMETRY_METHOD_TYPES`` and ``BROKEN_SYMMETRY_METHODS`` name between them: the
    Hartree-Fock methods carry the ``'wavefunction'`` method type they share with the correlated
    methods, so the method type alone does not separate them and the method name is read as well.
    ``HF-3c`` is one of them. Its corrections - a geometrical counterpoise term, a dispersion term
    and a short-range basis term - are additive functions of the nuclear coordinates rather than
    of the wavefunction, so the level's energy is still the energy of its determinant plus a
    number the reference does not enter.

    A CORRELATED WAVEFUNCTION METHOD DOES NOT ADMIT ONE. Its SCF determinant is the zeroth-order
    reference a correlation expansion is built about rather than the answer, and the expansion is
    parameterized about a spin-adapted reference. Breaking the symmetry of that reference lowers
    the SCF energy and RAISES the correlated one, because the symmetry-broken orbitals absorb
    into themselves the static correlation the expansion would otherwise recover, leaving less of
    it for the expansion to find. It also suppresses the ``T1`` diagnostic ARC reads off a coupled
    cluster single point, whose purpose is to report a reference the expansion is a poor
    description about: on a broken-symmetry reference ``T1`` falls below the threshold at which
    ARC reports multireference character, so the character the stability analysis measured is left
    both uncorrected and unreported.

    A DOUBLE HYBRID DOES NOT ADMIT ONE EITHER, and ``DOUBLE_HYBRID_METHODS`` is read before the
    method type because ARC types a double hybrid as density functional theory. Its energy is not
    the energy of its Kohn-Sham determinant: a perturbative second-order correlation term is added
    to it, expanded about that determinant, which is the construction the correlated methods are
    excluded for. The names are matched with their hyphens and underscores dropped, so a level
    written either way is recognized, and the list is a deny-list rather than a classification of
    every functional, so a double hybrid it does not name is admitted as ordinary density
    functional theory.

    A reference-agnostic level, one ARC writes no reference prefix for at all, admits nothing to
    change and is reported here as admitting no broken-symmetry reference.

    Args:
        level (Level, optional): The level of theory to check.

    Returns:
        bool: Whether a broken-symmetry reference describes what the level computes.
    """
    if level is None:
        return False
    method = (level.method or '').lower().replace('-', '').replace('_', '')
    if method in DOUBLE_HYBRID_METHODS:
        return False
    return level.method_type in BROKEN_SYMMETRY_METHOD_TYPES \
        or method in BROKEN_SYMMETRY_METHODS


def derived_reference_is_unrestricted(species: ARCSpecies | None) -> bool:
    """
    Check whether a species' measured wavefunction-stability verdict calls for an unrestricted reference.

    Only an external instability of a restricted reference does. An external instability is
    a relaxation of a constraint the reference imposes, Gaussian's RHF -> UHF class, so a
    lower solution exists outside the spin symmetry the restricted reference holds the
    wavefunction in and that reference is not the ground state. An internal instability lies
    within the reference's own spin symmetry, so it is not evidence of broken-symmetry
    character and does not call for a different reference. A ``'stable'`` verdict, an
    ``'unknown'`` one, an absent verdict, and a verdict whose reference could not be read all
    return ``False``.

    Args:
        species (ARCSpecies, optional): The species to check.

    Returns:
        bool: Whether the measured verdict calls for an unrestricted reference.
    """
    verdict = getattr(species, 'derived_stability_verdict', None)
    if not isinstance(verdict, dict):
        return False
    return verdict.get('verdict') == DERIVED_UNRESTRICTED_VERDICT and verdict.get('restricted') is True


def derived_instability_breaks_spin_symmetry(species: ARCSpecies | None) -> bool | None:
    """
    Report whether a measured instability relaxed the SPIN constraint, or ``None`` where it says nothing.

    An external instability names the constraint it relaxed, which an ESS reports as a pair of
    reference labels and which the parsers store on the verdict as ``relaxations``. Only a
    relaxation whose target reference is an unrestricted one, the RHF -> UHF class and its
    RKS -> UKS equivalent, is evidence of broken-symmetry character: the two electrons of a pair
    occupy different spatial orbitals in the lower solution, which is what a symmetry-broken real
    determinant describes. A relaxation to a COMPLEX reference, which Gaussian reports as
    RHF -> CRHF, relaxes the reality of the orbitals rather than the pairing of the spins, and the
    lower solution it points to is a complex one that no real determinant reaches, symmetry-broken
    or otherwise. Forcing an unpaired real determinant onto such a species describes neither the
    restricted solution nor the complex one it is being compared against.

    ``None`` is returned for a verdict naming no relaxation, which is any verdict that is not an
    external instability and any verdict reduced to the reference decision it carries, so a caller
    acting on the relaxation can tell "relaxed something other than spin" from "does not say".

    Args:
        species (ARCSpecies, optional): The species to check.

    Returns: bool | None
        Whether the relaxations the verdict names include a spin relaxation.
    """
    verdict = getattr(species, 'derived_stability_verdict', None)
    relaxations = verdict.get('relaxations') if isinstance(verdict, dict) else None
    if not relaxations:
        return None
    return any(str(relaxation).split('->')[-1].strip().upper().startswith(SPIN_RELAXED_REFERENCE_PREFIX)
               for relaxation in relaxations)


def adopted_reference_is_unrestricted(species: ARCSpecies | None) -> bool:
    """
    Check whether a species' measured stability verdict is one ARC acts on, and not only reports.

    ARC acts on a verdict for a transition state only. The analysis is run for any species whose
    tested reference was restricted, and acting on it means re-optimizing the species on the lower
    solution and running every job that follows there. The energy that produces is a
    broken-symmetry one, spin-contaminated and unprojected, so adopting a verdict for a well would
    write a contaminated energy into that species' thermo and into every reaction the species
    appears in, on the strength of a measurement of the reference alone. A transition state has no
    thermo of its own, and the reference of its remaining jobs is the decision the analysis
    informs. A well whose verdict is not adopted is reported instead, and declaring
    ``number_of_radicals`` for it runs its optimization, frequency and single point unrestricted
    together.

    A declared ``number_of_radicals`` of any value blocks adoption, since ``is_species_restricted``
    decides from the declared value alone whenever there is one, so a verdict measured alongside a
    declaration is reported and never acted on.

    A verdict naming the constraints it relaxed, none of which is the spin constraint, is reported
    and never acted on. Gaussian's ``RHF -> CRHF`` is such a verdict: it relaxes the reality of the
    orbitals rather than the pairing of the spins, and the lower solution it points at is a complex
    one that no real determinant reaches, symmetry-broken or otherwise, so running the species
    unrestricted describes neither the restricted solution nor the one it is being compared
    against. ``derived_instability_breaks_spin_symmetry`` reports that, and its ``None``, a verdict
    naming no relaxation at all, does not block adoption.

    A verdict carrying ``REFERENCE_CHANGE_AVAILABLE_KEY`` set to ``False`` is reported and never
    acted on either. That key records whether the ESSs that run this species' geometry, its
    Hessian and its electronic energy can each be given a symmetry-breaking reference, which
    ``Scheduler.stability_verdict_can_be_honoured`` decides when the verdict is recorded. An
    unrestricted reference an ESS cannot break the spin symmetry of collapses back to the
    restricted solution the verdict rejected, so acting on the verdict there would move the
    geometry onto the broken-symmetry surface while leaving the energy on the restricted one, and
    the number the run publishes would belong to neither. A verdict carrying the key set to
    ``True``, and one carrying no such key at all, is adopted on the strength of the measurement
    alone.

    The energy an adopted verdict produces, where the single point runs at one of the levels the
    verdict decides and which ``level_admits_a_broken_symmetry_reference`` defines, is a
    broken-symmetry one: it is spin-contaminated and it
    is not projected here. A broken-symmetry determinant mixes in the higher multiplicity, so its
    energy lies ABOVE the spin-pure low-spin energy, and the restricted energy it replaces lies
    above the broken-symmetry one in turn: E_projected < E_BS < E_restricted. Adoption therefore
    moves the energy toward the spin-pure value without reaching it, and what remains is a
    residual of the same sign rather than an overshoot. ``arc/checks/spin.py`` holds the Yamaguchi
    approximate spin-projection arithmetic that estimates E_projected from the broken-symmetry and
    high-spin energies and their ``S**2`` values; the residual error after adoption is the
    contamination, not the reference.

    WHERE THAT ERROR LANDS. Adoption acts for a transition state only, so a TS whose restricted
    reference was unstable runs unrestricted while the reactants and products it is compared
    against stay restricted. The adopted TS energy still sits above the spin-pure one while the
    wells, whose restricted references are stable and carry no such contamination, do not, so the
    barrier the run reports is systematically OVERestimated, by the residual contamination of the
    TS alone. Adoption shrinks that overestimate without removing it: the restricted TS energy it
    replaces sat higher still. The bias is one-sided because the asymmetry is: nothing projects it
    out and nothing raises the wells to match.

    Args:
        species (ARCSpecies, optional): The species to check.

    Returns:
        bool: Whether the measured verdict decides this species' reference.
    """
    verdict = getattr(species, 'derived_stability_verdict', None)
    if isinstance(verdict, dict) and verdict.get(REFERENCE_CHANGE_AVAILABLE_KEY) is False:
        return False
    if derived_instability_breaks_spin_symmetry(species) is False:
        return False
    return (getattr(species, 'number_of_radicals', None) is None
            and derived_reference_is_unrestricted(species)
            and bool(getattr(species, 'is_ts', False)))


def species_may_read_previous_orbitals(species: ARCSpecies | None) -> bool:
    """
    Check whether a job of this species may start from orbitals the species does not hold.

    Every adapter that reads an orbital guess takes it from the species, and falls back to
    whatever orbitals file sits in its own job directory where the species holds none. A
    species carrying an adopted wavefunction-stability verdict and no checkfile is holding
    none deliberately: the orbitals it dropped describe the restricted reference the verdict
    rejected, and an unrestricted SCF seeded from them returns to that solution, since a
    restricted solution is a stationary point of the unrestricted equations too. The job
    directory of a job whose name a previous job of the same species already carried holds
    exactly such a file, so the fallback is refused for as long as the species holds no
    orbitals of the reference it adopted, and the job composes the symmetry-breaking
    directive that reaches the lower solution instead.

    Args:
        species (ARCSpecies, optional): The species to check.

    Returns:
        bool: Whether a job of this species may adopt an orbitals file the species does not hold.
    """
    return not (adopted_reference_is_unrestricted(species) and getattr(species, 'checkfile', None) is None)


def open_shell_character_source(species: ARCSpecies | None) -> str | None:
    """
    Report which source attributed open-shell character to a species beyond its spin multiplicity.

    Returns ``'declared'`` when the user declared a ``number_of_radicals`` greater than one,
    which is the only declaration that attributes open-shell character beyond the multiplicity
    and which always wins over a measured verdict; ``'derived'`` when the user declared nothing
    and a measured wavefunction-stability verdict ARC acts on calls for an unrestricted
    reference; and ``None`` when neither applies, in which case the spin multiplicity alone
    decides the reference.

    A declared ``number_of_radicals`` of zero or one is not a source: ``is_species_restricted``
    turns a declaration into an unrestricted reference only above one, so such a declaration
    attributes no open-shell character. It still blocks a measured verdict from being adopted,
    which is why it does not fall through to ``'derived'`` either. A verdict ARC reports without
    acting on it, which is any verdict measured for a species that is not a transition state,
    likewise decides nothing and is not credited as the source.

    Args:
        species (ARCSpecies, optional): The species to check.

    Returns: str | None
        ``'declared'``, ``'derived'``, or ``None``.
    """
    number_of_radicals = getattr(species, 'number_of_radicals', None)
    if number_of_radicals is not None:
        return 'declared' if number_of_radicals > 1 else None
    if adopted_reference_is_unrestricted(species):
        return 'derived'
    return None


def is_species_restricted(obj: JobAdapter,
                          species: ARCSpecies | None = None,
                          ) -> bool:
    """
    Check whether a species should be executed as restricted or unrestricted.

    A user-declared ``number_of_radicals`` always decides. Only when the user declared
    nothing does a measured wavefunction-stability verdict enter, and then only an external
    instability of a restricted reference measured for a transition state, which makes the
    species unrestricted. That precedence is written once, in
    ``adopted_reference_is_unrestricted``, and is not restated here.

    An adopted verdict decides the reference of the levels a broken-symmetry reference describes,
    which ``level_admits_a_broken_symmetry_reference`` defines: the geometry and the Hessian of an
    adopted species come from the lower solution, and its correlated single point keeps the
    spin-adapted reference its correlation expansion is built about. The spin multiplicity and a
    declared ``number_of_radicals`` are not gated on the level and decide every level alike, so an
    open-shell species runs unrestricted at a correlated level as it always has; what the level
    decides is only whether a MEASURED verdict is what breaks the symmetry.

    Args:
        obj: The job adapter object.
        species (ARCSpecies, optional): The species to check.

    Returns:
        bool: Whether to run as restricted (``True``) or not (``False``).
    """

    if obj.level.method_type in REFERENCE_AGNOSTIC_METHOD_TYPES:
        return True

    multiplicity = obj.multiplicity if species is None else species.multiplicity
    species_obj = obj.species[0] if species is None else species
    number_of_radicals = species_obj.number_of_radicals
    species_label = species_obj.label
    if multiplicity > 1 or (number_of_radicals is not None and number_of_radicals > 1):
        # run an unrestricted electronic structure calculation if the spin multiplicity is greater than one,
        # or if it is one but the number of radicals is greater than one (e.g., bi-rad singlet)
        # don't run unrestricted for composite methods such as CBS-QB3, it'll be done automatically if the
        # multiplicity is greater than one, but do specify uCBS-QB3 for example for bi-rad singlets.
        if number_of_radicals is not None and number_of_radicals > 1:
            logger.info(f'Using an unrestricted method for species {species_label} which has '
                        f'{number_of_radicals} radicals and multiplicity {multiplicity}.')
        return False
    if adopted_reference_is_unrestricted(species_obj):
        if not level_admits_a_broken_symmetry_reference(obj.level):
            logger.info(f'Composing a restricted reference for the {obj.job_type} job of species {species_label} '
                        f'at {obj.level}, whose wavefunction stability analysis was adopted: that level reports a '
                        f'correlation energy expanded about its SCF determinant rather than the energy of the '
                        f'determinant itself, so a broken-symmetry reference does not describe what it computes.')
            return True
        logger.info(f'Using an unrestricted method for species {species_label}, whose wavefunction stability '
                    f'analysis reported an external instability of its restricted reference and for which no '
                    f'number_of_radicals was declared.')
        return False
    return True


def check_argument_consistency(obj: JobAdapter):
    """
    Check that general arguments of a job adapter are consistent.

    Args:
        obj (JobAdapter): The specific (not abstract) job adapter object instance.
    """
    if obj.job_type == 'irc' and obj.job_adapter in ['molpro']:
        raise NotImplementedError(f'IRC is not implemented for the {obj.job_adapter} job adapter.')
    if obj.job_type == 'irc' and (obj.irc_direction is None or obj.irc_direction not in ['forward', 'reverse']):
        raise ValueError(f'Missing the irc_direction argument for job type irc. '
                         f'It must be either "forward" or "reverse".\nGot: {obj.irc_direction}')
    if obj.job_type == 'scan' and obj.job_adapter in ['molpro']:
        for species in obj.species:
            if any(rotor_dict['directed_scan_type'] == 'ess' for rotor_dict in species.rotors_dict.values()):
                raise NotImplementedError(f'The {obj.job_adapter} job adapter does not support ESS scans.')
    if obj.job_type == 'scan' and (obj.scan_res <= 0 or obj.scan_res > 20 or divmod(360, obj.scan_res)[1]):
        raise ValueError(f'Got an illegal rotor scan resolution of {obj.scan_res} degrees. It must be a '
                         f'positive value no coarser than 20 degrees that divides 360 evenly '
                         f'(check the rotor_scan_resolution in your input file and in ~/.arc/settings.py).')
    if obj.job_type == 'scan' and (
            (not obj.species[0].rotors_dict or obj.rotor_index is None) and obj.torsions is None):
        # If this is a scan job type and species.rotors_dict is empty (e.g., via pipe), then torsions must be set up.
        raise ValueError('Either torsions or a species rotors_dict along with a rotor_index argument '
                         'must be specified for an ESS scan job.')


def update_input_dict_with_args(args: dict,
                                input_dict: dict,
                                ) -> dict:
    """
    Update the job input_dict attribute with keywords and blocks from the args attribute.

    The function iterates over each key in the 'args' dictionary. Depending on the key type 
    ('block', 'keyword', 'trsh'), it updates the 'input_dict' accordingly. For 'block' and 
    'keyword' types, it appends the corresponding values to the 'input_dict'. For 'trsh', 
    it handles both list and non-list values.
    
    Args:
        args (dict): The job arguments with the following structure:
            args = {
                'block': {key1: value1, key2: value2, ...},
                'keyword': {key1: value1, key2: value2, ...},
                'trsh': {key1: value1, key2: value2, ... or key: [value1, value2, ...]},
                ...
            }

        input_dict (dict): The job input dict with a structure that may be updated:
            input_dict = {
                'block': <block string>,
                'keywords': <keywords string>,
                'scan_trsh': <scan_trsh string>,
                'trsh': <trsh string>,
                ...
            }

    Returns:
        dict: The updated input_dict with appended or modified values based on the args.

    Example:
        Input args structure:
            args = {
                'block': {'1': 'block text 1', '2': 'block text 2'},
                'keyword': {'opt': 'keyword opt', 'freq': 'keyword freq'},
                'trsh': ['trsh value 1', 'trsh value 2']
            }

        Input input_dict structure:
            input_dict = {
                'block': 'existing block text',
                'keywords': 'existing keywords',
                'scan_trsh': 'existing scan_trsh',
                'trsh': 'existing trsh'
            }

        Output input_dict structure:
            input_dict = {
                'block': 'existing block text\n\nblock text 1\nblock text 2\n',
                'keywords': 'existing keywords keyword opt keyword freq ',
                'scan_trsh': 'existing scan_trsh',
                'trsh': 'existing trsh trsh value 1 trsh value 2 '
            }

    """
    for arg_type, arg_dict in args.items():
        if arg_type == 'block' and arg_dict:
            input_dict['block'] = '\n\n' if not input_dict['block'] else input_dict['block']
            for block in arg_dict.values():
                # Chek if input_dict['block'] already contains a value and that it ends with a newline
                if input_dict['block'] and not input_dict['block'].endswith('\n'):
                    input_dict['block'] += '\n'
                input_dict['block'] += f'{block}'
        elif arg_type == 'keyword' and arg_dict:
            for key, value in arg_dict.items():
                if key == 'scan_trsh':
                    if 'scan_trsh' not in input_dict.keys():
                        input_dict['scan_trsh'] = ''
                    # Check if input_dict['scan_trsh'] already contains a value
                    if input_dict['scan_trsh']:
                        input_dict['scan_trsh'] += f' {value}'
                    else:
                        input_dict['scan_trsh'] += f'{value}'
                else:
                    if 'keywords' not in input_dict.keys():
                        input_dict['keywords'] = ''
                    # Check if input_dict['keywords'] already contains a value
                    if input_dict['keywords']:
                        input_dict['keywords'] += f' {value}'
                    else:
                        input_dict['keywords'] += f'{value}'
        elif arg_type == 'trsh':
            if isinstance(arg_dict, list):
                # arg_dict is a list, iterate through its elements
                for val in arg_dict:
                    if 'trsh' not in input_dict.keys():
                        input_dict['trsh'] = ''
                    # Check if input_dict['trsh'] already contains a value
                    if input_dict['trsh']:
                        input_dict['trsh'] += f' {val}'
                    else:
                        input_dict['trsh'] += f'{val}'
            else:
                # arg_dict is a dictionary, proceed as before
                for key, value in arg_dict.items():
                    # Check if value is a list
                    if isinstance(value, list):
                        for val in value:
                            if key not in input_dict.keys():
                                input_dict[key] = ''
                            if input_dict[key]:
                                input_dict[key] += f' {val}'
                            else:
                                input_dict[key] += f'{val}'
                    else:
                        if key not in input_dict.keys():
                            input_dict[key] = ''
                        if input_dict[key]:
                            input_dict[key] += f' {value}'
                        else:
                            input_dict[key] += f'{value}'

    return input_dict


def input_dict_strip(input_dict: dict) -> dict:
    """
    Strip all values in the input dict of leading and trailing whitespace.
    """
    stripped_dict = {
        key: (
            val.rstrip() if isinstance(val, str) and val.startswith('\n') and not val.endswith('\n') else
            val.lstrip() if isinstance(val, str) and not val.startswith('\n') and val.endswith('\n') else
            val.strip() if isinstance(val, str) and not val.startswith('\n') and not val.endswith('\n') else val
        )
        for key, val in input_dict.items() if val is not None
    }

    return stripped_dict


def get_dropped_level_args(args: dict,
                           level_args: dict,
                           ) -> dict:
    """
    Get the options of a level of theory which the job args do not carry.

    An option of ``level_args`` is dropped if ``args`` holds no entry under the same two keys or
    holds a different value for it. An entry of ``args`` which is not a dictionary is treated as
    carrying no options at all.

    Args:
        args (dict): The job specific arguments.
        level_args (dict): The args of the level of theory, a dictionary of dictionaries.

    Returns:
        dict: The dropped options, keyed as ``level_args`` is, without the empty entries.
    """
    dropped_args = dict()
    for key, val in level_args.items():
        job_val = args.get(key, None)
        job_val = job_val if isinstance(job_val, dict) else dict()
        dropped = {key_2: val_2 for key_2, val_2 in val.items() if job_val.get(key_2, None) != val_2}
        if dropped:
            dropped_args[key] = dropped
    return dropped_args


def set_job_args(args: dict | None,
                 level: Level | None,
                 job_name: str,
                 ) -> dict:
    """
    Set the job args considering args from ``level`` and from ``trsh``.

    The args of ``level`` are adopted when ``args`` carries no options. When ``args`` does carry
    options it is used as given, and a warning reports the options of ``level`` which it does not
    carry, i.e., only the options which are actually being dropped.

    Args:
        args (dict): The job specific arguments.
        level (Level): The level of theory.
        job_name (str): The job name.

    Returns:
        dict: The initialized job specific arguments.
    """
    args_carry_options = args is not None and any(val for val in args.values())
    if args_carry_options and level is not None and level.args and any(val for val in level.args.values()):
        dropped_args = get_dropped_level_args(args=args, level_args=level.args)
        if dropped_args:
            logger.warning(f'ARC ignores the following user-specified level of theory options '
                           f'in job {job_name}:\n{pformat(dropped_args)}')
    elif not args_carry_options and level is not None:
        args = {**(args or dict()), **level.get_args()}
    for key in ['keyword', 'block', 'trsh']:
        if key not in args.keys():
            args[key] = dict()
    return args


def which(command: str | list,
          return_bool: bool = True,
          raise_error: bool = False,
          raise_msg: str | None = None,
          env: str | None = None,
          ) -> bool | str | None:
    """
    Test to see if a command (or a software package via its executable command) is available.

    Args:
        command (str | list): The command(s) to check (checking whether at least one is available).
        return_bool (bool, optional): Whether to return a Boolean answer.
        raise_error (bool, optional): Whether to raise an error is the command is not found.
        raise_msg (str, optional): An error message to print in addition to a generic message if the command isn't found.
        env (str, optional): A string representation of all environment variables separated by the os.pathsep symbol.

    Raises:
        ModuleNotFoundError: If ``raise_error`` is True and the command wasn't found.

    Returns:
        bool | str | None:
            The command path or ``None``, returns ``True`` or ``False`` if ``return_bool`` is set to ``True``.
    """
    if env is None:
        lenv = {"PATH": os.pathsep + os.environ.get("PATH", "") + os.pathsep + os.path.dirname(sys.executable),
                "PYTHONPATH": os.pathsep + os.environ.get("PYTHONPATH", ""),
                }
    else:
        lenv = {"PATH": os.pathsep.join([os.path.abspath(x) for x in env.split(os.pathsep) if x != ""])}
    lenv = {k: v for k, v in lenv.items() if v is not None}

    command = [command] if isinstance(command, str) else command
    ans = None
    for comm in command:
        ans = shutil.which(comm, mode=os.F_OK | os.X_OK, path=lenv["PATH"] + os.pathsep + lenv["PYTHONPATH"])
        if ans:
            break

    if raise_error and ans is None:
        raise_msg = raise_msg if raise_msg is not None else ''
        raise ModuleNotFoundError(f"The command {command}"
                                  f"was not found in envvar PATH nor in PYTHONPATH.\n{raise_msg}")

    if return_bool:
        return bool(ans)
    else:
        return ans


def combine_parameters(input_dict: dict, terms: list) -> tuple[dict, list]:
    """
    Extract and combine specific parameters from a dictionary's string values based on a list of terms.

    This function iterates over each key-value pair in the input dictionary. If a value is a string,
    it searches for occurrences of each term in the 'terms' list within the string. Each found term is 
    appended to a 'parameters' list. The terms found are then removed from the string value in the dictionary.
    The function returns a modified copy of the input dictionary with the terms removed from the string values,
    and a list of extracted parameters.

    Args:
        input_dict (dict): A dictionary with string values from which terms are to be extracted.
        terms (list): A list of string terms to be searched for within the dictionary's values.
        
    Returns:
        tuple: A tuple containing two elements:
            - A dictionary with the same structure as 'input_dict', but with the terms removed from the string values.
            - A list of extracted parameters (terms found in the string values of the input dictionary).
        
    """

    input_dict_copy = input_dict.copy()
    parameters = []

    for key, value in input_dict_copy.items():
        if isinstance(value, str):
            # Sort terms by length in descending order to handle overlapping terms
            for term in sorted(terms, key=len, reverse=True):
                matches = re.findall(term, value)
                for match in matches:
                    if match:
                        parameters.append(match)
                        value = re.sub(term, '', value)
            input_dict_copy[key] = value

    # Parameters may appear as one word, so need to split them via comma
    parameters = [param.split(',') for param in parameters]
    # Flatten the list of lists
    parameters = [item for sublist in parameters for item in sublist]
    # Remove empty spaces from the beginning and end of each parameter
    parameters = [param.strip() for param in parameters]
    # Parameters may appear multiple times in the input_dict, so remove duplicates
    parameters = list(set(parameters))
    # Sort the list
    parameters.sort()

    return input_dict_copy, parameters
