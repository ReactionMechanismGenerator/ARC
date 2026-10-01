"""
Module for writing the consolidated output.yml at the end of an ARC run.

output.yml supersedes output/status.yml and <project>.info: it consolidates
all result data into a single file with run-relative paths so downstream
consumers (TCKDB, analysis scripts) need only one file.

Written atomically at the very end of a run. If the run is interrupted, the
file will not exist rather than be partially written.
"""

import datetime
import glob
import json
import math
import os
import re
import tempfile
from typing import Any, NamedTuple
import uuid

import yaml

from arc.common import (
    ARC_PATH,
    NUMBER_BY_SYMBOL,
    VERSION,
    get_git_commit,
    get_logger,
    is_str_float,
    is_str_int,
    read_yaml_file,
    save_yaml_file,
)
from arc.constants import E_h_kJmol, bohr_to_angstrom
from arc.exceptions import InputError
from arc.imports import settings
from arc.job.adapters.common import open_shell_character_source
from arc.job.env_run import rmg_env_command
from arc.job.local import execute_command
from arc.level import RECORDED_LEVEL_STRIPPED_KEYS, adaptive_levels_as_list, level_as_plain_dict
from arc.parser.adapters.gaussian import parse_gaussian_constraints
from arc.parser.adapters.orca import parse_orca_constraints
from arc.parser.parser import (
    determine_ess,
    parse_1d_scan_energies,
    parse_dipole_moment,
    parse_e_elect,
    parse_ess_version,
    parse_opt_steps,
    parse_s_squared,
    parse_t1,
    parse_wavefunction_stability,
    parse_zpe_correction,
    s_squared_expected_from_multiplicity,
    parse_1d_scan_full_result,
    parse_geometry,
    parse_scan_args,
)
from arc.checks.common import TS_ATOM_MAP_METHOD, TS_ATOM_MAP_UNAVAILABLE_REASONS
from arc.reaction.reaction import ATOM_MAP_SOURCES
from arc.species.converter import get_most_common_isotope_for_element, xyz_to_str
from arc.species.species import are_coords_compliant_with_graph
from arc.species.vectors import calculate_dihedral_angle
from arc.statmech.arkane import (
    AEC_SECTION_START, AEC_SECTION_END,
    MBAC_SECTION_START, MBAC_SECTION_END,
    PBAC_SECTION_START, PBAC_SECTION_END,
    find_best_across_files, get_arkane_treatment, get_file_sha256,
    get_qm_corrections_files,
)
from arc.parser_evidence import (
    EVIDENCE_FILENAME,
    EVIDENCE_SCHEMA_NAME,
    EVIDENCE_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
    build_parser_evidence,
    fsync_directory,
    write_parser_evidence_atomic,
)


logger = get_logger()

LEVEL_JOB_KEYS = ('opt', 'freq', 'sp', 'composite', 'irc')
LEVEL_DICT_STRIPPED_KEYS = ('repr', 'compatible_ess')

HESSIAN_METHOD_ANALYTIC = 'analytic'
HESSIAN_METHOD_FD_GRADIENT = 'finite_difference_gradient'
HESSIAN_METHOD_FD_ENERGY = 'finite_difference_energy'

GAUSSIAN_ANALYTIC_METHODS = frozenset({'hf', 'mp2', 'cis', 'casscf'})
GAUSSIAN_FD_GRADIENT_METHODS = frozenset({'mp3', 'mp4(sdq)', 'mp4sdq', 'mp4(dq)', 'mp4dq',
                                          'qcisd', 'ccsd', 'ccd', 'cid', 'cisd', 'bd'})
GAUSSIAN_FD_ENERGY_METHODS = frozenset({'mp4', 'mp4(sdtq)', 'mp4sdtq', 'mp5',
                                        'ccsd(t)', 'ccsd-t', 'qcisd(t)', 'qcisd(tq)',
                                        'bd(t)', 'bd(tq)'})
GAUSSIAN_RO_FD_ENERGY_METHODS = frozenset({'mp2', 'mp3', 'mp4', 'mp4(sdq)', 'mp4sdq',
                                           'mp4(sdtq)', 'mp4sdtq',
                                           'ccd', 'ccsd', 'ccsd(t)', 'ccsd-t'})
GAUSSIAN_NO_ANALYTIC_HESSIAN_DFT_MARKERS = ('dsd', 'pwpb95', 'qidh', 'pbe0dh', 'pbe0-dh',
                                            'b2gpplyp', 'wb97x-2')
GAUSSIAN_CORRELATED_METHOD_REGEX = re.compile(r'^(?:ro|r|u)?(?:mp[3-9]|cc|qci|ci|bd|cas|hf)(?:\b|[a-z]*\()')
GAUSSIAN_FREQ_KEYWORD_REGEX = re.compile(
    r'\bfreq(?:uency)?(?![a-z])(?:\s*=?\s*\(([^)]*)\)|\s*=\s*([^\s,]+))?',
    re.IGNORECASE)

GAUSSIAN_ATOM_MASS_REGEX = re.compile(r'^\s*Atom\s+\d+\s+has atomic number\s+(\d+)\s+and mass\s+(\d+\.\d+)\s*$')
GAUSSIAN_DIPOLE_TOTAL_REGEX = re.compile(r'Tot=\s*-?\d+\.\d+')
GAUSSIAN_POPULATION_DENSITY_REGEX = re.compile(r'Population analysis using the\s+(\S+)\s+density', re.IGNORECASE)

ORCA_ANALYTIC_HESSIAN_HF_METHODS = frozenset({'hf', 'rhf', 'uhf', 'rohf'})
ORCA_DOUBLE_HYBRID_MARKERS = ('2plyp', 'dsd', 'pwpb95', 'qidh', 'pbe0dh', 'pbe0-dh', 'wb97x-2')
ORCA_NO_ANALYTIC_HESSIAN_MARKERS = ('ri-jk', 'rijk')
ORCA_ANALYTIC_HESSIAN_MARKER = 'ORCA SCF HESSIAN'
ORCA_INPUT_ECHO_END_MARKER = '****END OF INPUT****'
ORCA_ROUTE_SCAN_LINE_LIMIT = 5000
ORCA_HESSIAN_MARKER_SCAN_LINE_LIMIT = 50000
ORCA_FREQ_KEYWORD_REGEX = re.compile(r'(?<![a-z])freq(?![a-z])')

ARKANE_STATMECH_SUBDIRS = ('thermo', 'kinetics')
ARKANE_LOG_FILENAMES = ('arkane.log', 'stdout.log')
ARKANE_BANNER_SCAN_LINE_LIMIT = 40
ARKANE_VERSION_BANNER_REGEX = re.compile(r'^#\s+Version:(.*)$')
ARKANE_GIT_HEAD_MARKER = 'The current git HEAD for RMG-Py is:'
ARKANE_GIT_HASH_REGEX = re.compile(r'^[0-9a-f]{7,40}$')
ARKANE_TORSION_TREATMENT_BY_MODE = {'HinderedRotor': 'hindered_rotor', 'FreeRotor': 'free_rotor'}
RMG_DATABASE_CONDA_PACKAGE = 'rmgdatabase'
RMG_DATABASE_CONDA_SEARCH_DEPTH = 6
GIT_COMMIT_HASH_REGEX = re.compile(r'^[0-9a-f]{40}$')
RMG_PY_VERSION_REGEX = re.compile(r'''^__version__\s*=\s*(['"])([^'"]+)\1''', re.MULTILINE)


def write_output_yml(
    project: str,
    project_directory: str,
    species_dict: dict,
    reactions: list,
    output_dict: dict,
    opt_level=None,
    freq_level=None,
    sp_level=None,
    neb_level=None,
    scan_level=None,
    irc_level=None,
    conformer_opt_level=None,
    conformer_sp_level=None,
    ts_guess_level=None,
    composite_method=None,
    freq_scale_factor: float | None = None,
    freq_scale_factor_user_provided: bool = False,
    bac_type: str | None = None,
    compute_thermo: bool = True,
    arkane_level_of_theory=None,
    irc_requested: bool = True,
    t0: float | None = None,
    completed_job_records: list | None = None,
    adaptive_levels: dict | None = None,
    arc_aec_yml_sha256s: list[str] | None = None,
) -> None:
    """
    Write the consolidated output.yml to <project_directory>/output/output.yml.

    Non-converged species appear with ``converged: false`` and null result fields.
    Monoatomic species have null for all freq/statmech fields (not absent).

    Args:
        project (str): ARC project name.
        project_directory (str): Root directory of this ARC project.
        species_dict (dict): {label: ARCSpecies} for all species and TSs.
        reactions (list): list of ARCReaction objects.
        output_dict (dict): {label: {convergence, paths, job_types, ...}}.
        opt_level (Level, optional): Level of theory for geometry optimization.
        freq_level (Level, optional): Level of theory for frequency calculations.
        sp_level (Level, optional): Level of theory for single-point energies.
        neb_level (Level, optional): Level of theory for NEB TS search (from orca_neb_settings).
        scan_level (Level, optional): The requested level of the rotor scans, ``None`` if scans were not requested.
        irc_level (Level, optional): The requested level of the IRC jobs, ``None`` if IRC could not run.
        conformer_opt_level (Level, optional): The requested level of the conformer optimizations,
            ``None`` if none could run.
        conformer_sp_level (Level, optional): The requested level of the conformer single points that follow the
            conformer optimizations, ``None`` if none could run.
        ts_guess_level (Level, optional): The requested level at which TS guesses are optimized and compared,
            ``None`` if the run has no TS.
        composite_method (Level, optional): Composite method (e.g., CBS-QB3, G4).
        freq_scale_factor (float, optional): The harmonic frequencies scaling factor used.
        freq_scale_factor_user_provided (bool): Whether the user explicitly set the scale factor.
        bac_type (str, optional): The BAC type ('p', 'm', or None) the run requested.
        compute_thermo (bool): Whether the run computes thermo, which is the only
            Arkane run that applies a bond additivity correction.
        arkane_level_of_theory (Level, optional): The composite LOT Arkane uses for energy corrections.
            Recorded only when it differs from sp_level.
        irc_requested (bool): Whether IRC jobs were requested for this run.
        t0 (float, optional): The epoch timestamp when the ARC run started.
        completed_job_records (list, optional): Lightweight per-job cost records accumulated by the Scheduler
            (see Scheduler._record_completed_job), used for the run-level cost metrics.
        adaptive_levels (dict, optional): The processed adaptive levels of the run, keyed by
            ``(min_heavy_atoms, max_heavy_atoms)`` tuples, each mapping job-type tuples to ``Level`` objects.
            ``None`` when the run does not use adaptive levels.
        arc_aec_yml_sha256s (list[str], optional): The SHA-256 digests of ARC's ``data/AEC.yml`` that the Arkane
            adapters of the final processing recorded when they rendered atom energies from it. The digest recorded
            with the E0 of every exported species and TS (``e0_aec_yml_sha256``, which covers the E0 computed by the TS
            check) is collected with them. ``None`` or empty when none was recorded (and for a run restarted from
            before the digest was recorded).
    """
    doc: dict[str, Any] = {}

    # ---- header ----------------------------------------------------------------
    doc['schema_version'] = OUTPUT_SCHEMA_VERSION
    arc_git, _ = get_git_commit(ARC_PATH)
    doc['project'] = project
    doc['arc_version'] = VERSION
    doc['arc_git_commit'] = arc_git or None
    arkane_version, arkane_git_commit = _get_arkane_provenance(project_directory, t0)
    doc['arkane_version'] = arkane_version
    doc['arkane_git_commit'] = arkane_git_commit
    corrections = _get_energy_corrections(arkane_level_of_theory, bac_type)
    doc['rmg_database'] = _get_rmg_database_identity(arkane_path=corrections.quantum_corrections_path)
    doc['arc_aec_yml_sha256'] = None
    doc['datetime_started'] = (
        datetime.datetime.fromtimestamp(t0).strftime('%Y-%m-%d %H:%M') if t0 is not None else None
    )
    doc['datetime_completed'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    # ---- run cost metrics -------------------------------------------------------
    doc['cost_metrics'] = _compute_cost_metrics(completed_job_records, t0)

    # ---- levels of theory -------------------------------------------------------
    doc['composite_method'] = _level_to_dict(composite_method)
    doc['opt_level'] = _level_to_dict(opt_level)
    doc['freq_level'] = _level_to_dict(freq_level)
    doc['sp_level'] = _level_to_dict(sp_level)
    if neb_level is not None:
        doc['neb_level'] = _level_to_dict(neb_level)
    doc['scan_level'] = _level_to_dict(scan_level)
    doc['irc_level'] = _level_to_dict(irc_level)
    doc['conformer_opt_level'] = _level_to_dict(conformer_opt_level)
    doc['conformer_sp_level'] = _level_to_dict(conformer_sp_level)
    doc['ts_guess_level'] = _level_to_dict(ts_guess_level)
    doc['gsm_level'] = None
    doc['arkane_level_of_theory'] = _level_to_dict(arkane_level_of_theory)
    doc['adaptive_levels'] = adaptive_levels_as_list(adaptive_levels, strip=LEVEL_DICT_STRIPPED_KEYS)
    doc['freq_scale_factor'] = freq_scale_factor
    freq_scale_factor_key, freq_scale_factor_source = (
        (None, None) if freq_scale_factor_user_provided
        else _resolve_freq_scale_factor_entry(freq_level)
    )
    doc['freq_scale_factor_key'] = freq_scale_factor_key
    doc['freq_scale_factor_source'] = freq_scale_factor_source
    doc['bac_type'] = bac_type
    doc['atom_energy_corrections'] = corrections.aec
    doc['bond_additivity_corrections'] = corrections.bac

    # ---- per-job software (used for input-deck filename lookup) ----------------
    # freq/sp fall back to opt's software because the runtime falls back
    # to opt_level when freq_level/sp_level aren't explicitly set, and
    # the same level → same software → same deck filename.
    opt_software = getattr(opt_level, 'software', None)
    software_by_job = {
        'opt': opt_software,
        'freq': getattr(freq_level, 'software', None) or opt_software,
        'sp': getattr(sp_level, 'software', None) or opt_software,
    }

    # ---- species and TSs --------------------------------------------------------
    point_groups = _compute_point_groups(species_dict, project_directory)
    species_corrections = _compute_species_corrections(
        species_dict, corrections.aec_key, bac_type if compute_thermo else None, project_directory,
    )
    doc['species'] = []
    doc['transition_states'] = []
    aec_yml_sha256s = list(arc_aec_yml_sha256s or list())
    for spc in species_dict.values():
        if getattr(spc, 'e0', None) is not None:
            aec_yml_sha256s.append(getattr(spc, 'e0_aec_yml_sha256', None))
        d = _spc_to_dict(spc, output_dict, project_directory, point_groups,
                         irc_requested=irc_requested, software_by_job=software_by_job,
                         freq_level=freq_level or opt_level)
        d['energy_corrections'] = _build_energy_corrections_for_species(
            spc.label, species_corrections, arkane_level_of_theory,
            bac_type if _bac_is_applied_to(spc, compute_thermo) else None,
            aec_table=corrections.aec, bac_table=corrections.bac,
            aec_key=corrections.aec_key, bac_key=corrections.bac_key,
        )
        d['energy_corrections'] = _drop_unapplied_corrections(d['energy_corrections'], *_correction_switches(spc))
        if spc.is_ts:
            doc['transition_states'].append(d)
        else:
            doc['species'].append(d)

    doc['arc_aec_yml_sha256'] = _get_recorded_aec_yml_sha256(aec_yml_sha256s)
    doc['gsm_level'] = _state_gsm_provenance(doc['transition_states'], project_directory)

    # ---- reactions --------------------------------------------------------------
    doc['reactions'] = [_rxn_to_dict(rxn) for rxn in reactions]

    # ---- atomic write -----------------------------------------------------------
    out_dir = os.path.join(project_directory, 'output')
    os.makedirs(out_dir, exist_ok=True)
    document_id = uuid.uuid4().hex
    try:
        evidence_doc = build_parser_evidence(
            output_doc=doc,
            project_directory=project_directory,
            document_id=document_id,
        )
        evidence_path = write_parser_evidence_atomic(
            evidence_doc=evidence_doc,
            output_directory=out_dir,
        )
        doc['parser_evidence'] = {
            'path': EVIDENCE_FILENAME,
            'schema_name': EVIDENCE_SCHEMA_NAME,
            'schema_version': EVIDENCE_SCHEMA_VERSION,
            'document_id': document_id,
        }
        evidence_counts = _evidence_status_counts(evidence_doc)
        logger.info(
            'Wrote parser evidence to %s '
            '(freq_hessian: available=%d unavailable=%d; '
            'irc: available=%d unavailable=%d; '
            'gsm: available=%d unavailable=%d)',
            evidence_path,
            evidence_counts['freq_hessian']['available'],
            evidence_counts['freq_hessian']['unavailable'],
            evidence_counts['irc']['available'],
            evidence_counts['irc']['unavailable'],
            evidence_counts['gsm']['available'],
            evidence_counts['gsm']['unavailable'],
        )
    except Exception as exc:
        logger.warning('Could not build/write optional parser evidence: %s', exc)
        _discard_stale_evidence(out_dir)
    out_path = os.path.join(out_dir, 'output.yml')
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix='.output.yml.tmp')
    try:
        os.close(fd)
        save_yaml_file(path=tmp_path, content=doc)
        with open(tmp_path, 'rb') as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_path, out_path)
        fsync_directory(out_dir)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.debug(f'Failed to remove temporary output file {tmp_path}', exc_info=True)
        raise
    logger.info(f'Wrote consolidated results to {out_path}')

# ── helpers ──────────────────────────────────────────────────────────────────

GSM_XTB_HAMILTONIAN_REGEX = re.compile(r'Hamiltonian\s+GFN2-xTB\b')
GSM_XTB_PROGRAM_CALL_REGEX = re.compile(r'program call\s*:(.*)')
GSM_XTB_CHARGE_REGEX = re.compile(r'(?:^|\s)--chrg\s+(-?\d+)(?:\s|$)')
GSM_XTB_UHF_REGEX = re.compile(r'(?:^|\s)--uhf\s+(\d+)(?:\s|$)')
GSM_XTB_VERSION_REGEX = re.compile(r'^\s*\*?\s*(xtb version\s+\S.*)$')
GSM_XTB_SCAN_LINE_LIMIT = 800


def _gsm_xtbout_paths(gsm_log: str | None, project_directory: str) -> list[str]:
    """The archived per-node xtb outputs (``gsm_node_outputs/*.xtbout``) beside a GSM stringfile, sorted."""
    if not gsm_log or not isinstance(gsm_log, str):
        return list()
    gsm_log = gsm_log if os.path.isabs(gsm_log) else os.path.join(project_directory, gsm_log)
    return sorted(glob.glob(os.path.join(os.path.dirname(gsm_log), 'gsm_node_outputs', '*.xtbout')))


def _parse_gsm_xtbout(path: str) -> dict[str, Any]:
    """
    Read what one xtb output states about the calculation: the ``version`` banner, whether it ran the GFN2-xTB
    Hamiltonian (``gfn2``), and the ``charge`` and number of unpaired electrons (``uhf``) of its program call,
    xtb's defaults (0 and 0) where the call omits a flag. A value of an output with no program call is ``None``
    (``gfn2`` is ``False`` without the Hamiltonian line). Never raises.
    """
    observed: dict[str, Any] = {'version': None, 'gfn2': False, 'charge': None, 'uhf': None}
    try:
        with open(path, 'r', errors='ignore') as f:
            for index, line in enumerate(f):
                if index >= GSM_XTB_SCAN_LINE_LIMIT:
                    break
                version_match = GSM_XTB_VERSION_REGEX.match(line)
                if version_match and observed['version'] is None:
                    observed['version'] = version_match.group(1).strip()
                if GSM_XTB_HAMILTONIAN_REGEX.search(line):
                    observed['gfn2'] = True
                call_match = GSM_XTB_PROGRAM_CALL_REGEX.search(line)
                if call_match:
                    charge_match = GSM_XTB_CHARGE_REGEX.search(call_match.group(1))
                    uhf_match = GSM_XTB_UHF_REGEX.search(call_match.group(1))
                    observed['charge'] = int(charge_match.group(1)) if charge_match else 0
                    observed['uhf'] = int(uhf_match.group(1)) if uhf_match else 0
    except OSError:
        logger.debug(f"Could not read the xtb output '{path}'", exc_info=True)
    return observed


def _state_gsm_provenance(ts_records: list[dict], project_directory: str) -> dict | None:
    """
    Read the archived per-node xtb outputs of every GSM log once, state the program and version they report on the
    converged records, and return the level of the xtb_gsm path searches of the run.

    A converged record with a GSM log gains ``gsm`` in its ``ess_software`` and ``ess_versions``, each only when
    there is at least one archived output and every output states the same program, or the same version banner.

    The level is ``{'method': 'gfn2', 'software': 'xtb'}`` only when the run has at least one GSM log and, beside
    every GSM log, every archived per-node xtb output shows the GFN2-xTB Hamiltonian and a program call with the
    charge and the number of unpaired electrons (multiplicity minus one) of the TS record; otherwise it is ``None``.

    Args:
        ts_records (list[dict]): The transition-state records of the document, updated in place.
        project_directory (str): Root directory of the project, against which relative log paths resolve.

    Returns:
        dict | None: The level, or ``None``.
    """
    stated_by_every_gsm_log = True
    with_gsm = [record for record in ts_records if record.get('gsm_log')]
    for record in with_gsm:
        xtbouts = _gsm_xtbout_paths(record['gsm_log'], project_directory)
        observations = [_parse_gsm_xtbout(path) for path in xtbouts]
        if record.get('converged') is True:
            softwares = {_identify_log_ess(path, project_directory) for path in xtbouts}
            versions = {observed['version'] for observed in observations}
            if len(softwares) == 1 and None not in softwares:
                record['ess_software'] = {**(record.get('ess_software') or dict()), 'gsm': softwares.pop()}
            if len(versions) == 1 and None not in versions:
                record['ess_versions'] = {**(record.get('ess_versions') or dict()), 'gsm': versions.pop()}
        multiplicity = record.get('multiplicity')
        if not observations or not isinstance(multiplicity, int) or any(
                not observed['gfn2'] or observed['charge'] != record.get('charge')
                or observed['uhf'] != multiplicity - 1 for observed in observations):
            stated_by_every_gsm_log = False
    return {'method': 'gfn2', 'software': 'xtb'} if with_gsm and stated_by_every_gsm_log else None


def _discard_stale_evidence(output_directory: str) -> None:
    """Remove a ``parser_evidence.json`` left behind by an earlier run.

    The sidecar is only trustworthy alongside the ``output.yml`` that names its
    ``document_id``; that pairing is how a consumer detects a stale or
    interrupted write. When this run could not produce evidence, the document
    omits the descriptor, so a sidecar surviving from a previous run would sit
    beside the new ``output.yml`` with nothing to compare it against and would
    read as current. Deleting it is what keeps "no descriptor" and "no sidecar"
    the same statement.

    Args:
        output_directory (str): The project's ``output`` directory.
    """
    stale_path = os.path.join(output_directory, EVIDENCE_FILENAME)
    if not os.path.isfile(stale_path):
        return
    try:
        os.unlink(stale_path)
        fsync_directory(output_directory)
    except OSError:
        logger.error(
            'Could not remove the stale parser evidence sidecar %s. It belongs to an earlier '
            'run and does not match the output.yml just written; delete it before consuming '
            'either file.', stale_path, exc_info=True,
        )
    else:
        logger.info('Removed the previous run\'s parser evidence sidecar %s.', stale_path)


def _evidence_status_counts(evidence_doc: dict) -> dict[str, dict[str, int]]:
    """Count available and unavailable envelopes separately by evidence kind."""
    counts = {
        evidence_kind: {'available': 0, 'unavailable': 0}
        for evidence_kind in ('freq_hessian', 'irc', 'gsm')
    }
    for record in evidence_doc.get('records') or []:
        if not isinstance(record, dict):
            continue
        for evidence_kind in ('freq_hessian', 'irc', 'gsm'):
            envelope = record.get(evidence_kind)
            if not isinstance(envelope, dict):
                continue
            status = envelope.get('status')
            if status in counts[evidence_kind]:
                counts[evidence_kind][status] += 1
    return counts


def _read_git_head_commit(repository: str) -> str | None:
    """
    Return the commit ``HEAD`` of the git checkout at ``repository``, read from its ``.git`` files without
    running git, or ``None`` when it cannot be read. Follows a ``.git`` file (worktree or submodule), a
    symbolic ``HEAD``, and ``packed-refs``. Never raises.
    """
    try:
        git_dir = os.path.join(repository, '.git')
        if os.path.isfile(git_dir):
            with open(git_dir, 'r') as f:
                pointer = f.read().strip()
            if not pointer.startswith('gitdir:'):
                return None
            git_dir = os.path.normpath(os.path.join(repository, pointer[len('gitdir:'):].strip()))
        common_dir = git_dir
        if os.path.isfile(os.path.join(git_dir, 'commondir')):
            with open(os.path.join(git_dir, 'commondir'), 'r') as f:
                common_dir = os.path.normpath(os.path.join(git_dir, f.read().strip()))
        with open(os.path.join(git_dir, 'HEAD'), 'r') as f:
            head = f.read().strip()
        if GIT_COMMIT_HASH_REGEX.match(head):
            return head
        if not head.startswith('ref:'):
            return None
        ref = head[len('ref:'):].strip()
        for ref_dir in (git_dir, common_dir):
            ref_path = os.path.join(ref_dir, ref)
            if os.path.isfile(ref_path):
                with open(ref_path, 'r') as f:
                    commit = f.read().strip()
                return commit if GIT_COMMIT_HASH_REGEX.match(commit) else None
        packed_refs = os.path.join(common_dir, 'packed-refs')
        if os.path.isfile(packed_refs):
            with open(packed_refs, 'r') as f:
                for line in f:
                    fields = line.split()
                    if len(fields) == 2 and fields[1] == ref and GIT_COMMIT_HASH_REGEX.match(fields[0]):
                        return fields[0]
    except (OSError, UnicodeDecodeError):
        logger.debug(f"Could not read the git HEAD of '{repository}'", exc_info=True)
    return None


def _read_conda_package_version(path: str, package: str) -> str | None:
    """
    Return the version of the conda ``package`` that installed ``path``, read from the ``conda-meta`` record of
    the nearest enclosing conda prefix, or ``None`` when ``path`` is not inside a prefix that lists the package.
    Never raises.
    """
    directory = os.path.abspath(path)
    try:
        for _ in range(RMG_DATABASE_CONDA_SEARCH_DEPTH):
            directory = os.path.dirname(directory)
            meta_dir = os.path.join(directory, 'conda-meta')
            if os.path.isdir(meta_dir):
                for record_path in sorted(glob.glob(os.path.join(glob.escape(meta_dir), f'{package}-*.json'))):
                    with open(record_path, 'r') as f:
                        record = json.load(f)
                    if isinstance(record, dict) and record.get('name') == package \
                            and isinstance(record.get('version'), str):
                        return record['version']
                return None
            if directory == os.path.dirname(directory):
                break
    except (OSError, ValueError):
        logger.debug(f"Could not read the conda record of '{package}' for '{path}'", exc_info=True)
    return None


def _run_qm_corrections_script(aec_key: str | None = None,
                               bac_key: str | None = None,
                               bac_type: str | None = None,
                               ) -> dict | None:
    """
    Run ``arc/scripts/get_qm_corrections.py`` in the RMG conda environment for the given keys and return its output
    (``aec``, ``bac`` and ``quantum_corrections_path``), or ``None`` when it could not be run or read. The temporary
    files are created inside the guarded block and removed whichever of them exists. Never raises.
    """
    script_path = os.path.join(ARC_PATH, 'arc', 'scripts', 'get_qm_corrections.py')
    tmp_in = tmp_out = None
    try:
        fd_in, tmp_in = tempfile.mkstemp(suffix='.qm_input.yml')
        os.close(fd_in)
        fd_out, tmp_out = tempfile.mkstemp(suffix='.qm_output.yml')
        os.close(fd_out)
        save_yaml_file(path=tmp_in, content={'aec_key': aec_key, 'bac_key': bac_key, 'bac_type': bac_type})
        command = rmg_env_command(py_args=[script_path, tmp_in, tmp_out])
        _, stderr = execute_command(command=command, shell=True, executable='/bin/bash')
        if stderr:
            logger.warning(f'get_qm_corrections.py stderr: {stderr}')
        result = read_yaml_file(tmp_out)
        return result if isinstance(result, dict) else {}
    except Exception:
        logger.debug('Could not run get_qm_corrections.py', exc_info=True)
        return None
    finally:
        for temporary_path in (tmp_in, tmp_out):
            if temporary_path is None:
                continue
            try:
                os.unlink(temporary_path)
            except OSError:
                logger.debug(f'Failed to remove temporary file {temporary_path!r}', exc_info=True)


def _get_recorded_aec_yml_sha256(digests: list[str] | None) -> str | None:
    """
    The SHA-256 of ARC's ``data/AEC.yml`` that the run recorded when it rendered atom energies from it.

    Args:
        digests (list[str], optional): The digests recorded during the run.

    Returns:
        str | None: The digest when exactly one distinct digest was recorded; ``None`` when none was, and, with a
                    warning, when several differing ones were.
    """
    distinct = sorted({digest for digest in digests or list() if isinstance(digest, str) and digest})
    if len(distinct) > 1:
        logger.warning(f'ARC rendered atom energies from {len(distinct)} different versions of data/AEC.yml during '
                       f'this run ({", ".join(distinct)}); arc_aec_yml_sha256 is not exported.')
        return None
    return distinct[0] if distinct else None


def _get_arkane_quantum_corrections_path() -> str | None:
    """
    The ``quantum_corrections/data.py`` that Arkane loaded, as ``arkane.encorr.data.quantum_corrections_path`` in the
    RMG conda environment, or ``None`` when it cannot be determined. Never raises.
    """
    path = (_run_qm_corrections_script() or {}).get('quantum_corrections_path')
    return path if isinstance(path, str) and path else None


def _get_rmg_database_identity(rmg_db_path: str | None = None,
                               arkane_path: str | None = None) -> dict[str, Any]:
    """
    The identity of the quantum corrections tables Arkane used, which are the ``quantum_corrections/data.py`` its own
    ``rmgpy`` settings point at (``arkane.encorr.data.quantum_corrections_path``), not necessarily the file under
    ARC's ``RMG_DB_PATH``.

    Returns ``{path_kind, git_commit, version, quantum_corrections_sha256}``. ``quantum_corrections_sha256`` is the
    SHA-256 of Arkane's file. ``path_kind`` is
    ``'git'`` when that file lies in a git checkout (``git_commit`` is then its ``HEAD``), ``'package'`` when it lies
    inside a conda prefix that lists the ``rmgdatabase`` package (``version`` is then that package's version), and
    ``'unknown'`` otherwise. A git checkout is recognised only at the database root, three directories above the
    file (``<root>/input/quantum_corrections/data.py``), so an unrelated repository further up is not mistaken for it.
    A warning is logged when the first file :func:`get_qm_corrections_files` lists for ``rmg_db_path`` has another
    digest than Arkane's file. Every value that cannot be determined is ``None``. Never raises.

    Args:
        rmg_db_path (str, optional): ARC's RMG database path. Defaults to ``settings['RMG_DB_PATH']``.
        arkane_path (str, optional): The ``quantum_corrections/data.py`` Arkane loaded, when already known. Defaults to
                                     asking the RMG environment.
    """
    identity: dict[str, Any] = {'path_kind': 'unknown', 'git_commit': None, 'version': None,
                                'quantum_corrections_sha256': None}
    if not isinstance(arkane_path, str) or not arkane_path:
        arkane_path = _get_arkane_quantum_corrections_path()
    if arkane_path is None:
        return identity
    identity['quantum_corrections_sha256'] = get_file_sha256(arkane_path)
    database_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(arkane_path))))
    if os.path.exists(os.path.join(database_root, '.git')):
        identity['path_kind'] = 'git'
        identity['git_commit'] = _read_git_head_commit(database_root)
    if identity['path_kind'] == 'unknown':
        version = _read_conda_package_version(arkane_path, RMG_DATABASE_CONDA_PACKAGE)
        if version is not None:
            identity['path_kind'] = 'package'
            identity['version'] = version
    rmg_db_path = rmg_db_path or settings.get('RMG_DB_PATH')
    if isinstance(rmg_db_path, str) and rmg_db_path and identity['quantum_corrections_sha256'] is not None:
        try:
            arc_sha256 = get_file_sha256(get_qm_corrections_files(rmg_db_path)[0])
        except (InputError, OSError):
            logger.debug(f"Could not locate the quantum corrections under '{rmg_db_path}'", exc_info=True)
            arc_sha256 = None
        if arc_sha256 is not None and arc_sha256 != identity['quantum_corrections_sha256']:
            logger.warning(f"The quantum corrections Arkane loaded ({arkane_path}) differ from the ones ARC reads "
                           f"under RMG_DB_PATH ({rmg_db_path}). ARC matches the correction keys against the "
                           f"latter, so a matched key may be missing or different in the file Arkane loaded.")
    return identity


class ArkaneProvenance(NamedTuple):
    """The version and git HEAD commit hash of one Arkane (RMG-Py) install.

    Both fields are read from the same install, so they never describe different
    software. Either is ``None`` when that install does not expose it.
    """
    version: str | None
    git_commit: str | None


def _get_arkane_provenance(project_directory: str | None, t0: float | None = None) -> ArkaneProvenance:
    """Return the version and git HEAD commit of the Arkane (RMG-Py) that ARC invoked.

    Read from the header Arkane prints at the top of its own log under
    ``<project_directory>/calcs/statmech/<subdir>/``, skipping a log last modified
    before ``t0``. When no such log yields a version, both fields come instead from
    ``RMG_PATH``: ``__version__`` in ``rmgpy/version.py`` and that repository's HEAD.

    Arkane prints its header when it starts, so a run that started and then failed is
    reported here the same as one that completed.
    """
    if isinstance(project_directory, str):
        for subdir in ARKANE_STATMECH_SUBDIRS:
            for file_name in ARKANE_LOG_FILENAMES:
                provenance = _parse_arkane_log_provenance(
                    os.path.join(project_directory, 'calcs', 'statmech', subdir, file_name), t0)
                if provenance.version is not None:
                    return provenance
    rmg_path = settings.get('RMG_PATH')
    return ArkaneProvenance(_parse_rmg_py_version(rmg_path), _get_rmg_py_git_commit(rmg_path))


def _parse_arkane_log_provenance(log_path: str, t0: float | None = None) -> ArkaneProvenance:
    """Return the version and RMG-Py HEAD commit Arkane's header reports in ``log_path``.

    Only the first ``ARKANE_BANNER_SCAN_LINE_LIMIT`` lines are read. Both fields are
    ``None`` when the file is absent, unreadable, last modified before ``t0``, or
    carries no header; the commit alone is ``None`` when Arkane found no git repository
    and printed none. When the scanned lines hold more than one header, both fields come
    from the first one, so an appended log never pairs one run's version with another
    run's commit.
    """
    version, git_commit = None, None
    try:
        if not os.path.isfile(log_path) or (t0 is not None and os.path.getmtime(log_path) < t0):
            return ArkaneProvenance(None, None)
        with open(log_path, 'r', errors='ignore') as f:
            hash_line_follows = False
            for line_number, line in enumerate(f):
                if line_number >= ARKANE_BANNER_SCAN_LINE_LIMIT:
                    break
                if hash_line_follows:
                    candidate = line.strip()
                    if git_commit is None and ARKANE_GIT_HASH_REGEX.match(candidate) is not None:
                        git_commit = candidate
                    hash_line_follows = False
                    continue
                if line.startswith(ARKANE_GIT_HEAD_MARKER):
                    hash_line_follows = True
                    continue
                if version is None:
                    match = ARKANE_VERSION_BANNER_REGEX.match(line)
                    if match is not None:
                        version = match.group(1).rstrip().rstrip('#').strip() or None
    except OSError as exc:
        logger.debug("Could not read the Arkane log '%s': %s", log_path, exc)
        return ArkaneProvenance(None, None)
    return ArkaneProvenance(version, git_commit)


def _get_rmg_py_git_commit(rmg_path: str | None) -> str | None:
    """Return the HEAD commit hash of the RMG-Py repository at ``rmg_path``, or None.

    ``None`` when ``rmg_path`` is unset or is not a path to a directory, when that
    directory holds no git repository, and when git's output cannot be read as a hash
    and a date. A ``rmg_path`` that is set but does not name a directory is logged as a
    warning.
    """
    if not rmg_path:
        return None
    if not isinstance(rmg_path, str) or not os.path.isdir(rmg_path):
        logger.warning("RMG_PATH is set to '%s', which is not a directory. The RMG-Py git commit "
                       "cannot be recorded in output.yml.", rmg_path)
        return None
    try:
        head, _ = get_git_commit(rmg_path)
    except ValueError:
        logger.debug("Could not read a git HEAD commit hash for the RMG-Py repository at '%s'",
                     rmg_path, exc_info=True)
        return None
    return head or None


def _parse_rmg_py_version(rmg_path: str | None) -> str | None:
    """Return ``__version__`` from ``rmgpy/version.py`` under ``rmg_path``, or None.

    ``None`` when ``rmg_path`` is unset or is not a path string, and when the version
    file is absent, unreadable, or holds no quoted ``__version__`` assignment.
    """
    if not rmg_path or not isinstance(rmg_path, str):
        return None
    version_path = os.path.join(rmg_path, 'rmgpy', 'version.py')
    if not os.path.isfile(version_path):
        return None
    try:
        with open(version_path, 'r', errors='ignore') as f:
            content = f.read()
    except OSError as exc:
        logger.debug("Could not read the RMG-Py version file '%s': %s", version_path, exc)
        return None
    match = RMG_PY_VERSION_REGEX.search(content)
    return match.group(2) if match is not None else None


def _level_to_dict(level) -> dict | None:
    """Convert a Level object or a stored level dict to a dict with all non-None fields, or None.

    ``repr`` and ``compatible_ess`` are removed and ``solvation_scheme_level`` is converted recursively,
    so the returned dict holds only plain types that ``yaml.safe_load`` accepts.
    """
    if level is None:
        return None
    if isinstance(level, dict) or hasattr(level, 'as_dict'):
        return level_as_plain_dict(level, strip=LEVEL_DICT_STRIPPED_KEYS)
    return {
        'method': getattr(level, 'method', None),
        'basis': getattr(level, 'basis', None),
        'software': getattr(level, 'software', None),
    }


def _recorded_level_to_dict(level) -> dict | None:
    """Convert a stored per-job level to a dict without the fields ARC only deduced, or None.

    ``repr``, ``compatible_ess``, ``software`` and ``solvation_scheme_level`` are removed: the program that ran a
    job is stated by ``ess_software``, not by the level.
    """
    return level_as_plain_dict(level, strip=RECORDED_LEVEL_STRIPPED_KEYS) \
        if isinstance(level, dict) or hasattr(level, 'as_dict') else None


def _conformer_provenance_to_fields(spc, n_conformers: int) -> tuple[list | None, str | None, dict | None,
                                                                     str | None]:
    """
    Build ``conformer_levels``, ``conformer_energy_kind``, ``conformer_energy_level`` and
    ``conformer_force_field`` of a record.

    ``conformer_levels`` has one entry per exported conformer: the level of the conformer optimization job that
    produced the geometry, or ``None`` for a geometry that was not optimized or whose level was not recorded.
    It is ``None`` when the record exports no conformers.

    The kind, level and force field describe ``conformer_energies`` as a whole, taken over the entries that hold an
    energy. ``conformer_energy_kind`` is set only when every such entry has the same recorded kind
    (``'force_field_kcal_mol'`` or ``'electronic_kj_mol'``); an energy of unknown origin, or a mixture of kinds
    (some conformers optimized, others still holding force-field energies), makes it ``None``.
    ``conformer_energy_level`` is the shared level of those entries when the kind is electronic and all of them
    were computed at the same level, and ``None`` otherwise (force-field energies have no level).
    ``conformer_force_field`` is the force field and backend that produced those entries when every one of them
    names the same, and ``None`` otherwise. A kind or force field that is not a non-empty string counts as unknown.

    Args:
        spc: The species (an ``ARCSpecies`` or anything with the same conformer attributes).
        n_conformers (int): The number of exported conformers.

    Returns:
        tuple: ``(conformer_levels, conformer_energy_kind, conformer_energy_level, conformer_force_field)``.
    """
    if not n_conformers:
        return None, None, None, None

    def _padded(attr: str) -> list:
        values = getattr(spc, attr, None)
        values = list(values) if isinstance(values, (list, tuple)) else list()
        return (values + [None] * n_conformers)[:n_conformers]

    levels = [_recorded_level_to_dict(level) for level in _padded('conformer_levels')]
    energies = _padded('conformer_energies')
    sources = [source if isinstance(source, dict) else None for source in _padded('conformer_energy_sources')]
    held = [source for energy, source in zip(energies, sources) if energy is not None]
    if not held:
        return levels, None, None, None
    force_fields = {_str_or_none(source.get('force_field')) if source is not None else None for source in held}
    force_field = force_fields.pop() if len(force_fields) == 1 else None
    if any(source is None or _str_or_none(source.get('kind')) is None for source in held):
        return levels, None, None, force_field
    kinds = {source['kind'] for source in held}
    if len(kinds) != 1:
        return levels, None, None, force_field
    kind = kinds.pop()
    if kind != 'electronic_kj_mol':
        return levels, kind, None, force_field
    energy_levels = [_recorded_level_to_dict(source.get('level')) for source in held]
    energy_level = energy_levels[0] if all(level == energy_levels[0] for level in energy_levels) else None
    return levels, kind, energy_level, None


def _get_conformer_ess(spc, n_conformers: int, project_directory: str) -> tuple[list[str | None], list[str | None]]:
    """
    Build ``conformer_ess_software`` and ``conformer_ess_version``: the program, and the version banner, stated by the
    optimization log each exported conformer geometry was parsed from (``conformer_logs``).

    An entry is ``None`` for a conformer with no recorded log (a force-field geometry, a user-supplied one, an older
    restart), for a log that is missing or states no program, and, for the version, wherever the program is ``None``
    or the log states no banner.

    Args:
        spc: The species (an ``ARCSpecies`` or anything with the same conformer attributes).
        n_conformers (int): The number of exported conformers.
        project_directory (str): The directory relative log paths resolve against.

    Returns:
        tuple: ``(conformer_ess_software, conformer_ess_version)``, each with ``n_conformers`` entries.
    """
    logs = getattr(spc, 'conformer_logs', None)
    logs = list(logs) if isinstance(logs, (list, tuple)) else list()
    logs = (logs + [None] * n_conformers)[:n_conformers]
    observed: dict[str, tuple[str | None, str | None]] = dict()
    software, version = list(), list()
    for log_path in logs:
        if not isinstance(log_path, str) or not log_path:
            software.append(None)
            version.append(None)
            continue
        if log_path not in observed:
            observed[log_path] = _observe_log_ess(log_path, project_directory)
        program, banner = observed[log_path]
        software.append(program)
        version.append(banner if program is not None else None)
    return software, version


def _get_conformers_isotopes(spc, conformers: list, project_directory: str) -> list[list[int] | None]:
    """
    Build ``conformers_isotopes``: for each exported conformer, the isotope mass numbers the optimization log it was
    parsed from (``conformer_logs``) states, one per atom, or ``None`` when that log is missing, not recorded or
    states no masses.

    Args:
        spc: The species (an ``ARCSpecies`` or anything with the same conformer attributes).
        conformers (list): The exported conformer geometries, as dictionaries or strings.
        project_directory (str): The directory relative log paths resolve against.

    Returns:
        list: One entry per conformer.
    """
    logs = getattr(spc, 'conformer_logs', None)
    logs = list(logs) if isinstance(logs, (list, tuple)) else list()
    logs = (logs + [None] * len(conformers))[:len(conformers)]
    result = list()
    for conformer, log_path in zip(conformers, logs):
        symbols = conformer.get('symbols') if isinstance(conformer, dict) else None
        result.append(_get_log_stated_isotopes(_abs_existing_path(log_path, project_directory), symbols))
    return result


def _get_exported_xyz(spc) -> dict | None:
    """The geometry a species record exports as ``xyz``: its final geometry, else its initial one, else ``None``."""
    return spc.final_xyz if spc.final_xyz is not None else spc.initial_xyz


def _species_levels_to_dict(entry: dict, is_ts: bool) -> dict:
    """
    Build the per-record ``levels`` object from the levels the scheduler recorded for the jobs
    whose logs are exported (``output_dict[label]['levels']``).

    Every key of ``LEVEL_JOB_KEYS`` is always present. A value is ``None`` where no such job ran, where the
    level was not recorded (an older run or restart), and, for ``irc``, on every record that is not a TS.

    Args:
        entry (dict): The scheduler's output dictionary of one species (may be empty).
        is_ts (bool): Whether the record is a transition state.

    Returns:
        dict: ``{'opt', 'freq', 'sp', 'composite', 'irc'}`` mapped to a level dictionary or ``None``.
    """
    recorded = entry.get('levels') if isinstance(entry, dict) else None
    recorded = recorded if isinstance(recorded, dict) else dict()
    return {key: _recorded_level_to_dict(recorded.get(key)) if isinstance(recorded.get(key), dict) and (key != 'irc' or is_ts)
            else None
            for key in LEVEL_JOB_KEYS}


def _resolve_freq_scale_factor_entry(freq_level) -> tuple[str | None, str | None]:
    """
    Return the ``data/freq_scale_factors.yml`` entry key matched for ``freq_level``
    and the literature source that entry points at.

    The key is the ``freq_scale_factors`` mapping key the level resolved to. The
    source is that entry's ``source`` index resolved against the file's top-level
    ``sources`` mapping.

    Both are ``None`` when ``freq_level`` is ``None``, when the file cannot be read
    or parsed, and when the level has no entry in the file. The key is returned with
    a ``None`` source when the entry carries no ``source`` index or the index is not
    listed in ``sources``.
    """
    if freq_level is None:
        return None, None
    yml_path = os.path.join(ARC_PATH, 'data', 'freq_scale_factors.yml')
    try:
        data = read_yaml_file(yml_path)
    except (InputError, OSError, yaml.YAMLError):
        logger.debug(f'Failed to read the frequency scale factor database {yml_path!r}', exc_info=True)
        return None, None
    if not isinstance(data, dict):
        return None, None

    factors = data.get('freq_scale_factors')
    if not isinstance(factors, dict):
        return None, None
    level_key = freq_level if isinstance(freq_level, str) else str(freq_level)
    entry = factors.get(level_key)
    if not isinstance(entry, dict):
        return None, None

    source_key = entry.get('source')
    sources = data.get('sources')
    if source_key is None or not isinstance(sources, dict):
        return level_key, None
    return level_key, sources.get(source_key)


def _make_rel_path(path: str | None, project_directory: str) -> str | None:
    """Convert a path to one relative to ``project_directory``, or ``None``.

    Returns ``None`` for an empty path and for any path that does not lie
    inside ``project_directory``. An export is meant to be shared, and a file
    outside the project directory is not part of it: no relative path to it can
    be resolved on the reader's machine, and the ``../../..`` walk-up that
    ``os.path.relpath`` would produce describes the writer's own filesystem
    layout, home-directory name included. ``None`` is the contract's existing
    way of saying a log is unavailable.
    """
    if not path:
        return None
    try:
        relative = os.path.relpath(path, project_directory)
    except ValueError:
        logger.debug(f'Cannot relate {path!r} to the project directory {project_directory!r}.')
        return None
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        logger.debug(f'Omitting {path!r} from the export: it lies outside the project '
                     f'directory {project_directory!r}.')
        return None
    return relative


# TS-guess method strings → output.yml log-field name. Mirror of
# scheduler._TS_GUESS_METHOD_TO_PATHS_KEY but in this module's
# vocabulary (record-field names rather than ``paths`` slots). Kept
# local to avoid an output→scheduler import dependency.
_TS_GUESS_METHOD_TO_LOG_FIELD: dict[str, str] = {
    'orca_neb': 'neb_log',
    'xtb_gsm': 'gsm_log',
    'xtb-gsm': 'gsm_log',
}


def _ts_guess_log_field_for_method(method: object) -> str | None:
    """Return the output-record log-field name (``neb_log`` / ``gsm_log``)
    corresponding to a TSGuess method string, or ``None`` for
    geometry-only / unknown methods.

    Case- and whitespace-insensitive. Mirror of
    ``scheduler._ts_guess_paths_key`` so the output writer can apply a
    method-aware fallback when the scheduler's ``paths`` slot wasn't
    populated (e.g. restart-restored runs that bypass the TS-selection
    write sites).
    """
    if not isinstance(method, str):
        return None
    return _TS_GUESS_METHOD_TO_LOG_FIELD.get(method.strip().lower())


def _timedelta_to_seconds(value) -> float | None:
    """
    Convert a timedelta, a numeric value, or a str(timedelta) representation to float seconds.

    Accepted string formats are those produced by ``str(datetime.timedelta)``:
    ``'H:MM:SS'``, ``'H:MM:SS.ffffff'``, and ``'N day(s), H:MM:SS[.ffffff]'``.

    Returns:
        float | None: The total seconds (rounded to 2 decimals), or ``None`` if the value
                      is ``None`` or cannot be interpreted.
    """
    if value is None:
        return None
    if isinstance(value, datetime.timedelta):
        return round(value.total_seconds(), 2)
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if isinstance(value, str):
        try:
            days = 0.0
            time_part = value
            if 'day' in value:
                day_str, time_part = value.split(',', 1)
                days = float(day_str.split()[0])
                time_part = time_part.strip()
            parts = time_part.split(':')
            if len(parts) != 3:
                return None
            hours, minutes, seconds = float(parts[0]), float(parts[1]), float(parts[2])
            return round(days * 86400 + hours * 3600 + minutes * 60 + seconds, 2)
        except (ValueError, IndexError):
            return None
    return None


def _compute_cost_metrics(completed_job_records: list | None, t0: float | None) -> dict:
    """
    Aggregate per-job cost records into run-level cost metrics.

    Per-job records come from ``Scheduler.completed_job_records`` (collected at job
    completion and persisted in the restart file). Pipe-mode tasks are recorded into
    the same list at ingestion (``PipeCoordinator._record_pipe_task_cost``, with
    ``server='pipe'`` and the pipe engine as the ESS). Jobs missing run-time or
    core-count data are counted (``jobs_missing_time`` / ``jobs_missing_cores``)
    rather than silently dropped, so downstream analysis knows the coverage.

    Args:
        completed_job_records (list, optional): Entries are dicts with at least
            'job_adapter' (the ESS/software name), 'run_time_sec', and 'cpu_cores'.
        t0 (float, optional): The epoch timestamp when the ARC run started.

    Returns:
        dict: {'wall_time_hrs', 'total_job_count', 'total_execution_time_hrs',
               'total_core_hours', 'jobs_missing_time', 'jobs_missing_cores', 'per_ess'}.
    """
    records = completed_job_records or []
    per_ess: dict[str, dict] = {}
    jobs_missing_time, jobs_missing_cores = 0, 0
    for record in records:
        ess = record.get('job_adapter') or 'unknown'
        entry = per_ess.setdefault(ess, {'job_count': 0,
                                         'execution_time_hrs': 0.0,
                                         'core_hours': 0.0,
                                         'jobs_missing_time': 0,
                                         })
        entry['job_count'] += 1
        run_time_sec = _timedelta_to_seconds(record.get('run_time_sec'))
        cpu_cores = record.get('cpu_cores')
        if run_time_sec is None:
            jobs_missing_time += 1
            entry['jobs_missing_time'] += 1
            continue
        entry['execution_time_hrs'] += run_time_sec / 3600.0
        if cpu_cores is None:
            jobs_missing_cores += 1
        else:
            entry['core_hours'] += run_time_sec * cpu_cores / 3600.0
    for entry in per_ess.values():
        entry['execution_time_hrs'] = round(entry['execution_time_hrs'], 4)
        entry['core_hours'] = round(entry['core_hours'], 4)
    wall_time_hrs = round((datetime.datetime.now().timestamp() - t0) / 3600.0, 4) \
        if t0 is not None else None
    return {
        'wall_time_hrs': wall_time_hrs,
        'total_job_count': len(records),
        'total_execution_time_hrs': round(sum(e['execution_time_hrs'] for e in per_ess.values()), 4),
        'total_core_hours': round(sum(e['core_hours'] for e in per_ess.values()), 4),
        'jobs_missing_time': jobs_missing_time,
        'jobs_missing_cores': jobs_missing_cores,
        'per_ess': per_ess or None,
    }


def _parse_zpe(freq_path: str | None, project_directory: str) -> float | None:
    """
    Parse ZPE in Hartree from the freq log file.

    Uses the ESS adapter's ``parse_zpe_correction``, which reads the
    ``Zero-point correction=`` line (Gaussian) or equivalent.  The adapter
    returns kJ/mol; we convert back to Hartree.
    """
    if not freq_path:
        return None
    if not os.path.isabs(freq_path):
        freq_path = os.path.join(project_directory, freq_path)
    if not os.path.isfile(freq_path):
        return None
    try:
        zpe_kj = parse_zpe_correction(freq_path)
        return zpe_kj / E_h_kJmol if zpe_kj is not None else None
    except Exception:
        return None


def _parse_wavefunction_stability(stability_path: str | None, project_directory: str) -> dict | None:
    """
    Parse the wavefunction stability verdict from a stability analysis log.

    Returns ``None`` when no stability analysis was run, when its log is missing,
    or when the log holds no verdict. Otherwise returns the parsed verdict with the
    log's run-relative path added under ``'log'``.

    ``'log'`` names the analysis log, and every other key of an ESS that follows an
    instability rather than only reporting it describes one of TWO wavefunctions:
    ``verdict``, ``lowest_eigenvalue``, ``negative_eigenvectors`` and ``restricted``
    belong to the wavefunction under TEST, while the energies and spin expectation values
    that log also holds belong to the FOLLOWED solution the ESS relaxed into, which is a
    different wavefunction. A consumer reading a quantity out of that log rather than out
    of this block is reading the followed solution.

    Returns: dict | None
        ``{'verdict': 'stable' | 'internal_instability' | 'external_instability'
                      | 'unattributed_instability' | 'unknown',
           'internal_instability': bool | None, 'external_instability': bool | None,
           'relaxations': list[str], 'negative_eigenvectors': list[dict],
           'lowest_eigenvalue': float | None, 'restricted': bool | None,
           'invalidates_analytic_freq': bool | None, 'log': str}``, plus the keys an
        individual ESS reader adds. The ORCA reader adds ``'n_analyses'`` (int), the number
        of analyses the log holds, ``'followed_to_stable'`` (bool), whether the last of
        several ended stable, and ``'s_squared_after_follow'`` (float | None), the spin
        expectation value of the followed solution, from which the sector of a restricted
        reference's instability is measured.
        ``'unknown'`` means an analysis ran but its verdict could not be read, and is
        never to be treated as ``'stable'``. ``'unattributed_instability'`` means the
        wavefunction is unstable but the ESS did not report which sector the instability
        lies in, and is likewise never to be treated as ``'stable'``.
    """
    if not stability_path:
        return None
    path = stability_path if os.path.isabs(stability_path) else os.path.join(project_directory, stability_path)
    if not os.path.isfile(path):
        return None
    try:
        result = parse_wavefunction_stability(path)
    except Exception:
        logger.debug(f'Failed to parse a wavefunction stability verdict from {path!r}', exc_info=True)
        return None
    if not result:
        return None
    result = dict(result)
    result['log'] = _make_rel_path(path, project_directory)
    return result


def _scf_reference_block(spc, stability: dict | None, project_directory: str) -> dict:
    """
    Report which source decided a species' open-shell character and which SCF references its jobs used.

    ``source`` is ``'declared'`` when the user gave a ``number_of_radicals``, which always wins
    and is reported even where it contradicts the measured verdict; ``'derived'`` when the user
    gave nothing and a measured wavefunction-stability verdict made the species unrestricted;
    and ``None`` when the spin multiplicity alone decided. ``verdict`` and ``verdict_restricted``
    carry the measured picture whether or not it was the deciding one, and ``log`` names the
    stability analysis they were read from, following ``_parse_wavefunction_stability``.

    The analysis is run for a transition state and for any other species whose optimization ran
    restricted, but only a transition state's verdict is acted on. So a species carrying an
    external instability under a ``source`` of ``None`` is one whose restricted energy sits above
    a lower symmetry-broken solution that ARC measured and left alone, which the entry's ``is_ts``
    separates from a transition state whose identical verdict reads ``'derived'`` and did decide.

    ``sp_reference`` and ``freq_reference`` are the references those two jobs declared in the
    inputs they actually ran, and ``reference_mismatch`` is ``True`` when they differ, which
    means the species' E0 sums an electronic energy and a ZPE taken from two different surfaces.
    It is ``None``, not ``False``, whenever either reference is unknown: an sp job that was never
    submitted because the sp level equals the opt level, or a job carrying no memo, leaves nothing
    to compare, and reporting that as ``False`` would be indistinguishable from two references
    checked and found to agree.

    ``source`` is ``None`` where a declared ``number_of_radicals`` of zero or one blocked a measured
    verdict without itself attributing open-shell character. ``declared_number_of_radicals`` still
    carries the declaration, so the pair says what happened.

    ``measured_on_ts_guess`` is set only on a verdict carried over from an abandoned TS guess,
    and names the guess it was measured on; the guess reported elsewhere in the entry is a
    different one. A TS switch resets the stability path, so ``log`` falls back to the path the
    species' own verdict was read from: a ``source`` of ``'derived'`` names the analysis that
    decided it whether or not that analysis is still the surviving geometry's.

    ``verdict`` falls back to the stability log when the species carries none, as a restart
    written before the species held one does, while ``source`` never does: the reference
    decision reads the species and only the species, so a verdict that reached the log but not
    the species decided nothing and is reported without being credited with the decision.

    The block is emitted whether or not the species converged, like the ``wavefunction_stability``
    entry it explains and unlike the parsed results around it. It records a decision ARC made
    rather than a quantity a job produced, and a species that adopted a verdict and then failed to
    converge is exactly the case where knowing ARC changed its reference explains the failure.

    Args:
        spc (ARCSpecies): The species the block describes.
        stability (dict, optional): The parsed wavefunction stability verdict, where one was parsed.
        project_directory (str): The project directory, which the reported log path is relative to.

    Returns: dict
        A flat mapping of scalars; keys are always present, with ``None`` where unknown.
    """
    verdict = getattr(spc, 'derived_stability_verdict', None)
    if not isinstance(verdict, dict):
        verdict = stability if isinstance(stability, dict) else None
    log = stability.get('log') if isinstance(stability, dict) else None
    if log is None and verdict is not None:
        log = _make_rel_path(verdict.get('log'), project_directory)
    references = getattr(spc, 'scf_references', None)
    references = references if isinstance(references, dict) else dict()
    sp_reference, freq_reference = references.get('sp'), references.get('freq')
    return {'source': open_shell_character_source(spc),
            'declared_number_of_radicals': getattr(spc, 'number_of_radicals', None),
            'verdict': verdict.get('verdict') if verdict else None,
            'verdict_restricted': verdict.get('restricted') if verdict else None,
            'measured_on_ts_guess': verdict.get('measured_on_ts_guess') if verdict else None,
            'sp_reference': sp_reference,
            'freq_reference': freq_reference,
            'reference_mismatch': sp_reference != freq_reference
                                  if sp_reference is not None and freq_reference is not None else None,
            'log': log,
            }


def _parse_spin_diagnostic(sp_path: str | None,
                           freq_path: str | None,
                           opt_path: str | None,
                           multiplicity: int | None,
                           project_directory: str,
                           ) -> dict | None:
    """
    Parse the S**2 spin-contamination diagnostic for a species' single-point calc.

    The diagnostic is a property of the (unrestricted) wavefunction, so it is
    parsed from the sp job's log; when the sp energy reused the optimization
    output the sp log may be absent, so the first of the sp, freq and opt/geo
    logs that exists is the one read. A log that exists but yields no
    ``<S**2>`` ends the search: an ESS with no ``<S**2>`` reader, or a
    restricted reference, means this calc has no diagnostic, and reading a
    different job's wavefunction in its place would attribute another level of
    theory's value to the sp calc. The log actually read is recorded under
    ``'log'`` so the value can be traced to it.

    Restricted / closed-shell logs print no ``<S**2>`` and this returns ``None``
    for them, so the caller omits the block rather than emitting an all-null one.

    ``s_squared_expected`` is recomputed from ARC's own ``multiplicity`` when
    available, falling back to the value the ESS log reported.

    Returns: dict | None
        ``{'s_squared': float, 's_squared_expected': float | None (omitted if
        None), 's_squared_annihilated': float | None (omitted if None),
        'log': str}`` or ``None`` when no ``<S**2>`` could be parsed.
    """
    parsed, source_path = None, None
    for candidate in (sp_path, freq_path, opt_path):
        if not candidate:
            continue
        path = candidate if os.path.isabs(candidate) else os.path.join(project_directory, candidate)
        if not os.path.isfile(path):
            continue
        source_path = path
        try:
            parsed = parse_s_squared(path)
        except Exception:
            logger.debug(f'Failed to parse an S**2 spin diagnostic from {path!r}', exc_info=True)
            parsed = None
        break
    if parsed is None or parsed.get('s_squared') is None:
        return None
    result: dict = {'s_squared': float(parsed['s_squared'])}
    expected = s_squared_expected_from_multiplicity(multiplicity)
    if expected is None:
        expected = parsed.get('s_squared_expected')
    if expected is not None:
        result['s_squared_expected'] = float(expected)
    annihilated = parsed.get('s_squared_annihilated')
    if annihilated is not None:
        result['s_squared_annihilated'] = float(annihilated)
    result['log'] = _make_rel_path(source_path, project_directory)
    return result


def _parse_opt_log(geo_path: str | None, project_directory: str) -> tuple:
    """
    Parse n_steps, final electronic energy, and final geometry from an opt log.

    Returns a 4-tuple ``(n_steps, final_energy_hartree, final_xyz_str, final_isotopes)``;
    any element may be ``None`` if that piece couldn't be extracted (the
    others are still attempted independently). ``final_xyz_str`` is in
    the same atom-only format as ``xyz_to_str`` produces — symbol +
    coords, one atom per line, no count header. ``final_isotopes`` is the
    isotope list the log itself states (see ``_get_log_stated_isotopes``), ``None`` for a log that states none.

    The geometry is parsed via :func:`parse_geometry`, which dispatches
    to per-ESS adapters; logs from any supported ESS work without
    branching here. Used for both fine and coarse opt logs — the coarse
    geometry surfaces as ``coarse_opt_output_xyz`` in output.yml so the
    downstream consumers can reconstruct the ``opt_coarse → opt`` geometry
    lineage.
    """
    if not geo_path:
        return None, None, None, None
    if not os.path.isabs(geo_path):
        geo_path = os.path.join(project_directory, geo_path)
    if not os.path.isfile(geo_path):
        return None, None, None, None

    # Each parse is best-effort and independent — if one fails we still
    # surface the others. The final-energy parse historically failed-fast
    # for the whole tuple, but that's not the right shape now that the
    # geometry parse is also in the mix.
    n_steps = None
    e_hartree = None
    final_xyz = None
    final_isotopes = None
    try:
        n_steps = parse_opt_steps(geo_path)
    except Exception:
        logger.debug("Could not parse n_steps from %s", geo_path, exc_info=True)
    try:
        e_kj = parse_e_elect(geo_path)
        e_hartree = e_kj / E_h_kJmol if e_kj is not None else None
    except Exception:
        logger.debug("Could not parse final energy from %s", geo_path, exc_info=True)
    try:
        xyz_dict = parse_geometry(geo_path)
        if xyz_dict is not None:
            final_xyz = xyz_to_str(xyz_dict)
            final_isotopes = _get_log_stated_isotopes(geo_path, xyz_dict['symbols'])
    except Exception:
        logger.debug("Could not parse final geometry from %s", geo_path, exc_info=True)
    return n_steps, e_hartree, final_xyz, final_isotopes


def _get_log_stated_isotopes(log_path: str | None, symbols) -> list[int] | None:
    """
    The isotope mass numbers a Gaussian log states it used, one per atom of ``symbols``, ``None`` for any other log.

    Gaussian prints ``Atom N has atomic number Z and mass M`` for every atom of a job that runs a thermochemistry
    analysis; the mass number is ``M`` rounded. The last block of ``len(symbols)`` such lines is taken. The result is
    ``None`` when the log is not a Gaussian log, cannot be read, prints no such lines, prints a number of lines that
    is not a multiple of ``len(symbols)``, or prints an atomic number that is not that of the symbol at its position.
    Never raises.
    """
    if not isinstance(log_path, str) or not os.path.isfile(log_path) or not isinstance(symbols, (list, tuple)) \
            or not symbols or not all(symbol in NUMBER_BY_SYMBOL for symbol in symbols):
        return None
    if _identify_log_ess(log_path, os.path.dirname(log_path)) != 'gaussian':
        return None
    entries = list()
    try:
        with open(log_path, 'r', errors='ignore') as f:
            for line in f:
                match = GAUSSIAN_ATOM_MASS_REGEX.match(line)
                if match is not None:
                    entries.append((int(match.group(1)), float(match.group(2))))
    except OSError as exc:
        logger.debug(f"Could not read the isotopes of '{log_path}': {exc}")
        return None
    n_atoms = len(symbols)
    if len(entries) < n_atoms or len(entries) % n_atoms:
        return None
    block = entries[-n_atoms:]
    if any(atomic_number != NUMBER_BY_SYMBOL[symbol] for (atomic_number, _), symbol in zip(block, symbols)):
        return None
    return [int(round(mass)) for _, mass in block]


def _get_t1_diagnostic(sp_log_path: str | None, project_directory: str) -> float | None:
    """
    The T1 diagnostic the exported single-point log prints for its correlated method (for example coupled cluster, or
    the coupled-pair methods of ORCA, which also print one), parsed through ARC's parser for that log's program.

    Args:
        sp_log_path (str, optional): The path of the exported sp log, absolute or relative to ``project_directory``.
        project_directory (str): The directory a relative path resolves against.

    Returns:
        float | None: The T1 value; ``None`` when there is no log, the log prints none, the parsed value is not a
                      finite number, or the log cannot be parsed. Never raises.
    """
    abs_path = _abs_existing_path(sp_log_path, project_directory)
    if abs_path is None:
        return None
    try:
        t1 = parse_t1(abs_path)
    except Exception:
        logger.debug(f"Could not parse a T1 diagnostic from '{abs_path}'", exc_info=True)
        return None
    if isinstance(t1, bool) or not isinstance(t1, (int, float)) or not math.isfinite(t1):
        return None
    return float(t1)


def _str_or_none(value) -> str | None:
    """``value`` when it is a non-empty string, else ``None``."""
    return value if isinstance(value, str) and value.strip() else None


def _get_reversible(rxn) -> bool | None:
    """
    Whether a reaction is written as reversible: ``True`` for the ``<=>`` arrow, ``False`` for ``=>``, and ``None``
    for any other arrow or a reaction that states none.
    """
    arrow = getattr(rxn, 'arrow', None)
    return {'<=>': True, '=>': False}.get(arrow.strip()) if isinstance(arrow, str) else None


def _abs_existing_path(path: str | None, project_directory: str) -> str | None:
    """Return the absolute path of a calculation file that exists on disk, else ``None``.

    ``path`` may be absolute (as ``output_dict['paths']`` stores it) or relative to
    ``project_directory`` (as the fields this module emits store it); both forms are
    resolved here.
    """
    if not path or not isinstance(path, str):
        return None
    abs_path = path if os.path.isabs(path) else os.path.join(project_directory, path)
    return abs_path if os.path.isfile(abs_path) else None


def _parse_calc_constraints(
    input_rel_path: str | None,
    log_path: str | None,
    software: str | None,
    project_directory: str,
) -> list[dict]:
    """Best-effort parse of held-fixed constraints for one calculation.

    Prefers the ESS input deck (``input_rel_path``) over the log because
    the deck holds the exact ModRedundant / ``%geom Constraints`` block
    ARC emitted; the log echoes it but adds parser surface area. Returns
    an empty list when no deck/log is available, the software is
    unsupported, or the file can't be parsed.

    Never raises: any failure inside the parser is logged as a warning by
    the parser itself, and we shrug it off here so output.yml generation
    stays robust to malformed decks.
    """
    if not software:
        return []
    sw = str(software).lower()

    candidates = [path for path in (_abs_existing_path(input_rel_path, project_directory),
                                    _abs_existing_path(log_path, project_directory))
                  if path is not None]

    if not candidates:
        return []

    try:
        if sw == 'gaussian':
            for path in candidates:
                parsed = parse_gaussian_constraints(path)
                if parsed:
                    return parsed
            return []
        if sw == 'orca':
            for path in candidates:
                parsed = parse_orca_constraints(path)
                if parsed:
                    return parsed
            return []
    except Exception as exc:
        logger.warning("Constraint extraction failed for %s (software=%s): %s",
                       candidates[0], sw, exc)
    return []


def _input_filename_for(software: str | None) -> str | None:
    """Return the ESS-specific input deck filename, or None.

    Pulls from ``settings['input_filenames']`` so the mapping stays
    in one place. Software not in the map (e.g., ``gcn``, ``torchani``,
    ``mockter`` — generally not "real" ESS jobs) returns None and the
    caller emits no input-deck path for that job.
    """
    if not software:
        return None
    name = str(software).lower()
    return (settings.get('input_filenames') or {}).get(name)


def _derive_input_path(
    log_path: str | None,
    software: str | None,
    project_directory: str,
) -> str | None:
    """Return the input deck path (project-relative) for a given job log.

    The input deck is a sibling of the log file, named per
    ``input_filenames[software]``. Existence is checked on disk: if the
    file isn't there (e.g., archived runs that kept the log but discarded
    the deck), this returns None rather than emitting a ghost path.
    """
    if not log_path:
        return None
    fname = _input_filename_for(software)
    if not fname:
        return None
    abs_log = log_path if os.path.isabs(log_path) else os.path.join(project_directory, log_path)
    candidate = os.path.join(os.path.dirname(abs_log), fname)
    if not os.path.isfile(candidate):
        return None
    return _make_rel_path(candidate, project_directory)


def _gaussian_route_texts(path: str, first_only: bool = False) -> list[str]:
    """Return the Gaussian route sections of an input deck or log, each as a single line, in file order.

    A route starts at a line whose first non-blank character is ``#`` and ends at the following
    blank line (an input deck) or rule of dashes (a log's echoed route). The first route of the
    file is the first such line. Every later route must directly follow a rule of dashes, which is
    how a log opens each ``Link1`` step, so a ``#`` line wrapped out of an archive block is not
    read as a route. With ``first_only`` the file is read only as far as its first route.

    A log wraps the route mid-token at a fixed column and indents every echoed line by one
    space; its lines are joined after dropping that single leading space and without
    inserting one, so ``integral=(grid=ultr`` + ``afine, ...`` rejoins as written. A deck
    block has no wrapping, so its lines are stripped and joined by a single space.

    Returns an empty list when the file cannot be read or holds no route.
    """
    routes: list[str] = []
    collected: list[str] = []
    in_route = False
    previous_was_rule = False

    def finish(ended_on_dash_rule: bool) -> None:
        if ended_on_dash_rule:
            routes.append(''.join(part[1:] if part.startswith(' ') else part for part in collected))
        else:
            routes.append(' '.join(part.strip() for part in collected))

    try:
        with open(path, 'r', errors='ignore') as f:
            for line in f:
                stripped = line.strip()
                is_rule = bool(stripped) and set(stripped) == {'-'}
                if not in_route:
                    if stripped.startswith('#') and (not routes or previous_was_rule):
                        in_route = True
                        collected = [line.rstrip('\n')]
                    previous_was_rule = is_rule
                    continue
                if not stripped or is_rule:
                    finish(is_rule)
                    in_route = False
                    previous_was_rule = is_rule
                    if first_only:
                        return routes
                    continue
                collected.append(line.rstrip('\n'))
            if in_route:
                finish(False)
    except OSError as exc:
        logger.debug("Could not read a Gaussian route from '%s': %s", path, exc)
        return []
    return routes


def _gaussian_route_text(path: str) -> str | None:
    """Return the first Gaussian route section of an input deck or log as a single line.

    The joining rules are those of ``_gaussian_route_texts``. Only the first route of the file is
    read, so a ``--Link1--`` file yields its first job's route.

    Returns ``None`` when the file cannot be read or holds no route.
    """
    routes = _gaussian_route_texts(path, first_only=True)
    return routes[0] if routes else None


def _gaussian_freq_keyword_options(route: str) -> str | None:
    """Return the lowercase options attached to the route's ``freq`` keyword.

    ``''`` means the route requested a bare ``freq`` with no options, ``None`` means the
    route carries no ``freq`` keyword at all. ``freq=numer``, ``freq=(noraman,numer)``
    and ``freq(numer)`` all yield the text between the delimiters. A letter may not
    follow the keyword, so log prose such as ``Frequencies --`` does not match.
    """
    match = GAUSSIAN_FREQ_KEYWORD_REGEX.search(route)
    if match is None:
        return None
    return (match.group(1) or match.group(2) or '').lower()


def _gaussian_method_hessian_method(method: str, method_type: str | None) -> str | None:
    """Map a Gaussian method name onto how Gaussian builds its Hessian under a bare ``freq``.

    The method name is matched as a whole token, lowercased and stripped, against three
    explicit tables; a leading ``u`` or ``r`` spin prefix is removed when the remainder is
    a listed token, so ``uccsd(t)`` resolves as ``ccsd(t)``. An ``ro`` prefix is handled
    separately: ``rohf`` gets numerical frequencies, RO-MP*/RO-CC are energies-only, and
    every other ``ro`` method (RO-DFT included) is ``None``.

    A spelling that matches ``GAUSSIAN_CORRELATED_METHOD_REGEX`` without matching a table
    entry — ``mp4(full)``, ``mp4sdq(full)``, ``ccsd(full)`` — is ``None``.

    Methods outside the tables and outside that pattern fall through to ``method_type``:
    a ``'dft'`` level carrying none of the ``GAUSSIAN_NO_ANALYTIC_HESSIAN_DFT_MARKERS``
    substrings is analytic. Anything else — composite, semiempirical, force field, an
    unlisted wavefunction method — is ``None``.
    """
    normalized = method.lower().strip()
    if normalized.startswith('ro'):
        base = normalized[2:]
        if base == 'hf':
            return HESSIAN_METHOD_FD_GRADIENT
        if base in GAUSSIAN_RO_FD_ENERGY_METHODS:
            return HESSIAN_METHOD_FD_ENERGY
        return None
    candidates = [normalized]
    for prefix in ('u', 'r'):
        if normalized.startswith(prefix):
            candidates.append(normalized[len(prefix):])
    for candidate in candidates:
        if candidate in GAUSSIAN_FD_ENERGY_METHODS:
            return HESSIAN_METHOD_FD_ENERGY
        if candidate in GAUSSIAN_FD_GRADIENT_METHODS:
            return HESSIAN_METHOD_FD_GRADIENT
        if candidate in GAUSSIAN_ANALYTIC_METHODS:
            return HESSIAN_METHOD_ANALYTIC
    if GAUSSIAN_CORRELATED_METHOD_REGEX.match(normalized) is not None:
        return None
    if method_type == 'dft' and not any(marker in normalized
                                        for marker in GAUSSIAN_NO_ANALYTIC_HESSIAN_DFT_MARKERS):
        return HESSIAN_METHOD_ANALYTIC
    return None


def _gaussian_freq_hessian_method(deck_path: str | None,
                                  log_path: str | None,
                                  level,
                                  ) -> str | None:
    """Resolve the Hessian method of a Gaussian freq job from its route and its level.

    ``Freq=Numer`` (or ``Numerical``) and ``Freq=EnOnly`` state the answer in the route.
    A bare ``freq`` — what ``arc/job/adapters/gaussian.py`` writes — does not, and is
    resolved from the method by ``_gaussian_method_hessian_method``. Only the route is
    read; no other Gaussian log text is consulted.

    The route is taken from ``deck_path`` when that file yields one, and from
    ``log_path`` otherwise. Returns ``None`` when no route is available and when the
    route carries no ``freq`` keyword.
    """
    route = None
    for path in (deck_path, log_path):
        if path is None:
            continue
        route = _gaussian_route_text(path)
        if route:
            break
    if not route:
        return None
    options = _gaussian_freq_keyword_options(route)
    if options is None:
        return None
    if 'enonly' in options:
        return HESSIAN_METHOD_FD_ENERGY
    if 'numer' in options:
        return HESSIAN_METHOD_FD_GRADIENT
    method = getattr(level, 'method', None)
    if not method:
        return None
    return _gaussian_method_hessian_method(method, getattr(level, 'method_type', None))


def _orca_route_text(path: str) -> str | None:
    """Return the lowercase Orca keyword lines of an input deck or log, space-joined.

    Orca keyword lines start with ``!``; a log echoes the deck one line at a time behind
    an ``|  <n]> `` prefix, which is stripped here so a deck and a log yield the same
    text; a ``#`` comment, which ends at the next ``#`` or at the end of the line, is dropped. The scan stops at the ``****END OF INPUT****`` line that closes a log's echoed
    input block, and in any case after ``ORCA_ROUTE_SCAN_LINE_LIMIT`` lines. Returns
    ``None`` when the file cannot be read or holds no keyword line.
    """
    keyword_lines: list[str] = []
    try:
        with open(path, 'r', errors='ignore') as f:
            for line_number, line in enumerate(f):
                if line_number >= ORCA_ROUTE_SCAN_LINE_LIMIT:
                    break
                content = line.strip()
                if content.startswith('|') and '>' in content:
                    content = content.split('>', 1)[1].strip()
                if ORCA_INPUT_ECHO_END_MARKER in content:
                    break
                if content.startswith('!'):
                    keyword_lines.append(' '.join(re.sub(r'#[^#]*(#|$)', ' ', content[1:]).lower().split()))
    except OSError as exc:
        logger.debug("Could not read an Orca route from '%s': %s", path, exc)
        return None
    return ' '.join(keyword_lines) or None


def _orca_analytic_hessian_is_documented(route: str, level) -> bool:
    """Return whether Orca documents an analytic Hessian for this level and keyword line.

    True for HF and for a DFT functional that is not a double hybrid, and only when the
    keyword line carries no RI-JK approximation. Every other case — MP2, DLPNO, the
    coupled-cluster methods, a double hybrid, or an RI-JK run — is False.
    """
    if any(marker in route for marker in ORCA_NO_ANALYTIC_HESSIAN_MARKERS):
        return False
    method = (getattr(level, 'method', None) or '').lower().strip()
    if not method:
        return False
    if any(marker in method for marker in ORCA_DOUBLE_HYBRID_MARKERS):
        return False
    if method in ORCA_ANALYTIC_HESSIAN_HF_METHODS:
        return True
    return getattr(level, 'method_type', None) == 'dft'


def _file_contains_marker(path: str, marker: str, line_limit: int) -> bool:
    """Return whether ``marker`` appears within the first ``line_limit`` lines of ``path``.

    Returns ``False`` when the file cannot be read and when the marker lies beyond
    ``line_limit``.
    """
    try:
        with open(path, 'r', errors='ignore') as f:
            for line_number, line in enumerate(f):
                if line_number >= line_limit:
                    break
                if marker in line:
                    return True
    except OSError as exc:
        logger.debug("Could not scan '%s' for '%s': %s", path, marker, exc)
    return False


def _orca_freq_hessian_method(deck_path: str | None, log_path: str | None, level) -> str | None:
    """Resolve the Hessian method of an Orca freq job from its keyword line.

    ``NumFreq`` differentiates gradients, or energies when ``NumGrad`` is on the same
    keyword line. Orca documents ``Freq`` as an alias of ``AnFreq``, so both are analytic
    for the levels ``_orca_analytic_hessian_is_documented`` accepts and ``None``
    otherwise. When no keyword line can be read, an ``ORCA SCF HESSIAN`` header within the
    first ``ORCA_HESSIAN_MARKER_SCAN_LINE_LIMIT`` lines of the log is taken as analytic;
    no numerical header is relied on.
    """
    route = None
    for path in (deck_path, log_path):
        if path is None:
            continue
        route = _orca_route_text(path)
        if route:
            break
    if route is None:
        if log_path is not None and _file_contains_marker(log_path, ORCA_ANALYTIC_HESSIAN_MARKER,
                                                          ORCA_HESSIAN_MARKER_SCAN_LINE_LIMIT):
            return HESSIAN_METHOD_ANALYTIC
        return None
    if 'numfreq' in route:
        return HESSIAN_METHOD_FD_ENERGY if 'numgrad' in route else HESSIAN_METHOD_FD_GRADIENT
    if 'anfreq' in route or ORCA_FREQ_KEYWORD_REGEX.search(route) is not None:
        return HESSIAN_METHOD_ANALYTIC if _orca_analytic_hessian_is_documented(route, level) else None
    return None


def _pyscf_freq_hessian_method(spc) -> str | None:
    """Resolve the Hessian method of a PySCF freq job from the species' spin multiplicity.

    ``run_freq`` in ``arc/job/adapters/scripts/pyscf_script.py`` branches on the
    multiplicity alone: multiplicity 1 builds an analytic RKS Hessian, and any higher
    multiplicity builds a central finite difference of analytic gradients. Returns
    ``None`` when the species carries no multiplicity and when the multiplicity is not
    an integer value.
    """
    multiplicity = getattr(spc, 'multiplicity', None)
    if not is_str_int(multiplicity):
        return None
    return HESSIAN_METHOD_FD_GRADIENT if int(multiplicity) > 1 else HESSIAN_METHOD_ANALYTIC


def _get_freq_hessian_method(freq_input_path: str | None,
                             freq_log_path: str | None,
                             software: str | None,
                             level,
                             spc,
                             project_directory: str,
                             ) -> str | None:
    """Return how the frequency job's Hessian was built, or ``None``.

    The value is one of ``'analytic'``, ``'finite_difference_gradient'`` and
    ``'finite_difference_energy'``, and records the method ARC's own frequency request
    resolves to under the documented behaviour of the ESS that ran it. It is ``None``
    wherever that request does not settle the answer, and is never inferred from anything
    weaker.

    The input deck is read in preference to the log, the log being the fallback when the
    deck is not on disk. ``software`` is the freq job's ESS name as ``_get_ess_software``
    yields it, ``level`` is the freq level of theory and ``spc`` the species the job ran
    on. ``gaussian``, ``orca`` and ``pyscf`` are resolved by the helpers above; every
    other ESS returns ``None``, as does a freq job whose deck and log are both missing.
    """
    if not software:
        return None
    deck_path = _abs_existing_path(freq_input_path, project_directory)
    log_path = _abs_existing_path(freq_log_path, project_directory)
    if deck_path is None and log_path is None:
        return None
    sw = str(software).lower()
    if sw == 'gaussian':
        return _gaussian_freq_hessian_method(deck_path, log_path, level)
    if sw == 'orca':
        return _orca_freq_hessian_method(deck_path, log_path, level)
    if sw == 'pyscf':
        return _pyscf_freq_hessian_method(spc)
    return None


def _get_ess_route(log_path: str | None, project_directory: str) -> str | None:
    """
    The keyword line of an ESS job: the log's echoed route, or the route of the input deck next to the log when the log
    holds none. The program is the one the log itself states. For Gaussian the value is the route section as written
    (see ``_gaussian_route_text``); for Orca it is the lowercase keyword lines without their ``!``, space-joined (see
    ``_orca_route_text``). ``None`` for any other program, when the log is missing, and when neither file holds a
    keyword line. Never raises.
    """
    abs_log = _abs_existing_path(log_path, project_directory)
    if abs_log is None:
        return None
    software = _identify_log_ess(abs_log, project_directory)
    route_reader = {'gaussian': _gaussian_route_text, 'orca': _orca_route_text}.get(software)
    if route_reader is None:
        return None
    deck = _abs_existing_path(_derive_input_path(abs_log, software, project_directory), project_directory)
    for path in (abs_log, deck):
        if path is None:
            continue
        try:
            route = route_reader(path)
        except Exception:
            logger.debug(f"Could not read the route of '{path}'", exc_info=True)
            continue
        if route:
            return route
    return None


def _get_composite_step_routes(log_path: str | None, project_directory: str) -> list[str] | None:
    """
    The route of every internal step of a composite-method job, as the Gaussian log echoes them in file order.

    Gaussian prints one route section per ``Link1`` step, so a CBS-QB3 or G4 log yields one entry per step. Only the
    log is read, never the input deck beside it, and only a log the program identifies as Gaussian is read. ``None``
    for any other program, when the log is missing, and when it echoes no route. Never raises.
    """
    abs_log = _abs_existing_path(log_path, project_directory)
    if abs_log is None or _identify_log_ess(abs_log, project_directory) != 'gaussian':
        return None
    try:
        routes = _gaussian_route_texts(abs_log)
    except Exception:
        logger.debug(f"Could not read the step routes of '{abs_log}'", exc_info=True)
        return None
    return routes or None


def _get_gaussian_dipole_moment(log_path: str | None, project_directory: str) -> tuple[float | None, str | None]:
    """
    The total dipole moment in Debye of a Gaussian log and the density it was computed from.

    The value is that of the last ``Dipole moment (field-independent basis, Debye)`` block (read by
    ``parse_dipole_moment``) and the density is the word of the ``Population analysis using the <X> density`` header
    that precedes that block, as printed (``SCF``, ``current``, ...). Both are ``None`` for any other program, a
    missing log and a log that prints no block, and the density is ``None`` when no header precedes the block. Never
    raises.
    """
    abs_log = _abs_existing_path(log_path, project_directory)
    if abs_log is None or _identify_log_ess(abs_log, project_directory) != 'gaussian':
        return None, None
    try:
        value = parse_dipole_moment(abs_log)
    except Exception:
        logger.debug(f"Could not parse the dipole moment from '{abs_log}'", exc_info=True)
        return None, None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None, None
    density, last_block_density, header_density, expect_total, last_block_readable = None, None, None, False, True
    try:
        with open(abs_log, 'r', errors='ignore') as f:
            for line in f:
                match = GAUSSIAN_POPULATION_DENSITY_REGEX.search(line)
                if match is not None:
                    density = match.group(1)
                elif 'dipole moment' in line.lower() and 'debye' in line.lower():
                    header_density, expect_total = density, True
                elif expect_total:
                    last_block_readable = GAUSSIAN_DIPOLE_TOTAL_REGEX.search(line) is not None
                    last_block_density = header_density if last_block_readable else None
                    expect_total = False
    except OSError as exc:
        logger.debug(f"Could not read the dipole density of '{abs_log}': {exc}")
        return float(value), None
    if not last_block_readable:
        return None, None
    return float(value), last_block_density


def _get_gaussian_polarizability(log_path: str | None, project_directory: str) -> float | None:
    """
    The isotropic static polarizability in angstrom^3 of a Gaussian freq log.

    It is a third of the trace of the last ``Exact polarizability`` line (``xx xy yy xz yz zz``, atomic units, bohr^3,
    six fixed 8-character fields after the colon) the freq job printed, converted with ``bohr_to_angstrom``. ``None``
    for any other program, a missing log, a log that prints no such line, a last such line that cannot be read, and a
    last line whose six elements are all zero. Never raises.
    """
    abs_log = _abs_existing_path(log_path, project_directory)
    if abs_log is None or _identify_log_ess(abs_log, project_directory) != 'gaussian':
        return None
    tensor = None
    try:
        with open(abs_log, 'r', errors='ignore') as f:
            for line in f:
                if 'Exact polarizability' in line:
                    fields = line.rstrip('\r\n').split(':', 1)[-1]
                    tokens = [fields[i * 8:(i + 1) * 8].strip() for i in range(6)]
                    tensor = [float(token) for token in tokens] \
                        if len(fields) >= 48 and all(is_str_float(token) for token in tokens) else None
    except OSError as exc:
        logger.debug(f"Could not read the polarizability of '{abs_log}': {exc}")
        return None
    if tensor is None or not any(tensor):
        return None
    trace = tensor[0] + tensor[2] + tensor[5]
    return trace / 3.0 * bohr_to_angstrom ** 3


def _iter_ess_logs(paths: dict, project_directory: str):
    """
    Yield ``(job_type, absolute_log_path)`` for every ESS log referenced by ``paths``
    that exists on disk.

    The job types are ``'sp'``, ``'opt'``, ``'freq'``, ``'composite'``, ``'neb'`` and ``'irc'``, and define the
    key space shared by ``_get_ess_versions`` and ``_get_ess_software``. ``'irc'`` is yielded once per existing
    log of ``paths['irc']`` (forward and reverse). Relative paths are resolved against
    ``project_directory``; entries that are empty or not a file on disk are skipped.
    """
    key_map = {'sp': 'sp', 'geo': 'opt', 'freq': 'freq', 'composite': 'composite', 'neb': 'neb'}
    for path_key, label in key_map.items():
        log_path = _abs_existing_path(paths.get(path_key), project_directory)
        if log_path is not None:
            yield label, log_path
    for irc_path in _irc_log_paths(paths):
        log_path = _abs_existing_path(irc_path, project_directory)
        if log_path is not None:
            yield 'irc', log_path


def _irc_log_paths(paths: dict) -> list[str]:
    """The non-empty IRC log paths of ``paths``."""
    irc_paths = paths.get('irc')
    irc_paths = irc_paths if isinstance(irc_paths, (list, tuple)) else list()
    return [path for path in irc_paths if isinstance(path, str) and path]


def _collapse_ess_observations(observed: dict[str, list[str]], paths: dict) -> dict[str, str]:
    """
    Reduce per-log observations to one value per job type.

    ``'irc'`` keeps a value only when every recorded IRC log was observed and they all agree.
    """
    collapsed = {label: values[0] for label, values in observed.items() if label != 'irc' and values}
    irc_values = observed.get('irc') or list()
    if irc_values and len(irc_values) == len(_irc_log_paths(paths)) and len(set(irc_values)) == 1:
        collapsed['irc'] = irc_values[0]
    return collapsed


def _get_ess_versions(paths: dict, project_directory: str) -> dict[str, str] | None:
    """
    Parse ESS version strings from each available log file (sp, opt, freq, composite, neb, irc).

    Returns a dict like ``{'sp': 'ORCA 5.0.4', 'opt': 'Gaussian 16, Revision C.01'}``,
    keyed by job type. Each value is the full ESS banner as the program printed it, not a
    bare version number. The ``'irc'`` entry is present only when every IRC log states the same banner.
    Caches parsed versions to avoid re-parsing shared log files.
    Returns ``None`` if nothing could be parsed.
    """
    observed: dict[str, list[str]] = {}
    parsed_cache: dict[str, str] = {}
    for label, log_path in _iter_ess_logs(paths, project_directory):
        if log_path not in parsed_cache:
            try:
                version = parse_ess_version(log_path)
            except Exception:
                logger.debug(f"Failed to parse ESS version from log file '{log_path}'", exc_info=True)
                continue
            if not version:
                continue
            parsed_cache[log_path] = version
        observed.setdefault(label, list()).append(parsed_cache[log_path])
    return _collapse_ess_observations(observed, paths) or None


def _get_ess_software(paths: dict, project_directory: str) -> dict[str, str] | None:
    """
    Identify the ESS that produced each available log file (sp, opt, freq, composite, neb, irc).

    Returns a dict like ``{'sp': 'orca', 'opt': 'gaussian'}``, keyed by the same job types
    ``_get_ess_versions`` uses and read from the same log files, so a consumer can pair a
    version string with the program that actually produced it instead of with the program
    a level of theory declared. Values are ARC's lowercase ESS names. The ``'irc'`` entry is present only
    when every IRC log was identified as the same ESS.

    The key set is a superset of ``_get_ess_versions``' key set: a log whose ESS is
    identified but whose version banner cannot be parsed appears here and not there, while
    a parsed version always implies an identified ESS. Caches identifications to avoid
    re-reading shared log files. Returns ``None`` if no ESS could be identified.
    """
    observed: dict[str, list[str]] = {}
    identified_cache: dict[str, str] = {}
    for label, log_path in _iter_ess_logs(paths, project_directory):
        if log_path not in identified_cache:
            try:
                ess_name = determine_ess(log_file_path=log_path, raise_error=False)
            except Exception:
                logger.debug(f"Failed to identify the ESS of log file '{log_path}'", exc_info=True)
                continue
            if not ess_name:
                continue
            identified_cache[log_path] = ess_name
        observed.setdefault(label, list()).append(identified_cache[log_path])
    return _collapse_ess_observations(observed, paths) or None


def _identify_log_ess(log_path: str | None, project_directory: str) -> str | None:
    """The ESS name a single log states, ``None`` where the log is missing or states none. Never raises."""
    abs_path = _abs_existing_path(log_path, project_directory)
    if abs_path is None:
        return None
    try:
        return determine_ess(log_file_path=abs_path, raise_error=False) or None
    except Exception:
        logger.debug(f"Failed to identify the ESS of log file '{abs_path}'", exc_info=True)
        return None


def _observe_log_ess(log_path: str | None, project_directory: str) -> tuple[str | None, str | None]:
    """
    The ESS name and full version banner a single log states, ``(None, None)`` where the log is missing or
    states none. Never raises.
    """
    abs_path = _abs_existing_path(log_path, project_directory)
    if abs_path is None:
        return None, None
    software = _identify_log_ess(abs_path, project_directory)
    version = None
    try:
        version = parse_ess_version(abs_path) or None
    except Exception:
        logger.debug(f"Failed to parse ESS version from log file '{abs_path}'", exc_info=True)
    return software, version


class EnergyCorrections(NamedTuple):
    """The run-level AEC/BAC parameter tables together with the Arkane keys they came from.

    ``aec_key`` and ``bac_key`` are the exact Arkane database key strings that were
    fuzzy-matched in the atom-energy and bond-additivity sections respectively. They are
    matched independently and may legitimately differ, and either is ``None`` when no key
    was matched (or when the correction was not requested).
    """
    aec: dict | None
    bac: dict | None
    aec_key: str | None
    bac_key: str | None
    quantum_corrections_path: str | None = None


def _match_arkane_correction_keys(arkane_level_of_theory,
                                  bac_type: str | None,
                                  ) -> tuple[str | None, str | None]:
    """
    Fuzzy-match the Arkane database key strings for the AEC and the BAC sections separately.

    Returns ``(aec_key, bac_key)``. Each entry is ``None`` when no key matched, and
    ``bac_key`` is ``None`` when ``bac_type`` is not ``'p'`` or ``'m'``.
    """
    if arkane_level_of_theory is None:
        return None, None
    try:
        qm_corr_files = get_qm_corrections_files()
        aec_key = find_best_across_files(arkane_level_of_theory, qm_corr_files,
                                         AEC_SECTION_START, AEC_SECTION_END)
        bac_key = None
        if bac_type in ('p', 'm'):
            if bac_type == 'm':
                bac_start, bac_end = MBAC_SECTION_START, MBAC_SECTION_END
            else:
                bac_start, bac_end = PBAC_SECTION_START, PBAC_SECTION_END
            bac_key = find_best_across_files(arkane_level_of_theory, qm_corr_files, bac_start, bac_end)
        return aec_key, bac_key
    except Exception:
        logger.debug('Failed to match Arkane correction keys', exc_info=True)
        return None, None


def _get_energy_corrections(arkane_level_of_theory, bac_type: str | None) -> EnergyCorrections:
    """
    Look up the AEC (per-atom, Hartree) and BAC (per-bond, kcal/mol) values
    that Arkane used from the RMG database for the given level of theory.

    Finds the AEC and BAC keys independently via fuzzy matching in their
    respective database sections, then calls ``arc/scripts/get_qm_corrections.py``
    as a subprocess to extract the actual correction dicts. The matched keys are
    returned alongside the tables so that callers can report which Arkane key the
    corrections were computed from and whether the BAC table belongs to that key.

    Returns:
        EnergyCorrections: The AEC table, the BAC table, the AEC key and the BAC key, and the
        ``quantum_corrections_path`` the script reported, each ``None`` when unavailable.
    """
    aec_key, bac_key = _match_arkane_correction_keys(arkane_level_of_theory, bac_type)
    if aec_key is None and bac_key is None:
        return EnergyCorrections(None, None, None, None)
    result = _run_qm_corrections_script(aec_key, bac_key, bac_type)
    if result is None:
        return EnergyCorrections(None, None, aec_key, bac_key)
    reported_path = result.get('quantum_corrections_path')
    return EnergyCorrections(result.get('aec'), result.get('bac'), aec_key, bac_key,
                             reported_path if isinstance(reported_path, str) and reported_path else None)


def _safe(fn, default=None):
    """Call fn() and return *default* if any exception is raised."""
    try:
        return fn()
    except Exception:
        logger.debug(f'_safe() caught exception in {fn}', exc_info=True)
        return default


def _build_species_correction_inputs(species_dict: dict) -> list[dict]:
    """Build the per-species payloads for ``arc/scripts/get_species_corrections.py``.

    Species whose geometry is missing, empty, or carries an unknown element symbol
    are omitted.
    """
    species_inputs: list[dict] = []
    for label, spc in species_dict.items():
        xyz = _get_exported_xyz(spc)
        if xyz is None:
            continue
        symbols = list(xyz.get('symbols') or [])
        coords = [list(row) for row in (xyz.get('coords') or [])]
        if not symbols or not coords:
            continue
        nums = [NUMBER_BY_SYMBOL.get(s) for s in symbols]
        if any(n is None for n in nums):
            continue
        atoms: dict[str, int] = {}
        for s in symbols:
            atoms[s] = atoms.get(s, 0) + 1
        bonds = dict(getattr(spc, 'bond_corrections', None) or {})
        species_inputs.append({
            'label': label,
            'atoms': atoms,
            'bonds': bonds,
            'coords': coords,
            'nums': nums,
            'multiplicity': int(getattr(spc, 'multiplicity', None) or 1),
        })
    return species_inputs


def _bac_is_applied_to(spc, compute_thermo: bool) -> bool:
    """Whether ARC routes ``spc`` through an Arkane run that applies a BAC.

    A bond additivity correction is applied by the thermo statmech run and by the E0-only run of a species
    (for example a BDE fragment), both of which use the run's BAC type. Neither happens when the project does not
    compute thermo, and neither includes a transition state or, for the thermo run, a species whose own
    ``compute_thermo`` is off. The kinetics statmech run, the only Arkane run a rates-only project performs, is
    invoked with no BAC type, so nothing in such a run carries one.
    """
    if not compute_thermo:
        return False
    if getattr(spc, 'is_ts', False):
        return False
    if getattr(spc, 'e0_only', False) is True:
        return True
    return bool(getattr(spc, 'compute_thermo', True))


def _correction_switches(spc) -> tuple[bool | None, bool | None]:
    """The atom and bond correction switches of the Arkane run that produced the energy ``spc`` exports.

    That is the species' own E0 run for a transition state or an E0-only species, and the thermo run otherwise.
    A switch that is not known is ``None``.
    """
    if getattr(spc, 'is_ts', False) or getattr(spc, 'e0_only', False) is True:
        return (getattr(spc, 'e0_atom_corrections_applied', None),
                getattr(spc, 'e0_bond_corrections_applied', None))
    thermo = getattr(spc, 'thermo', None)
    return (getattr(thermo, 'atom_corrections_applied', None),
            getattr(thermo, 'bond_corrections_applied', None))


def _drop_unapplied_corrections(records: list[dict], atom_applied: bool | None, bond_applied: bool | None) -> list[dict]:
    """Keep only the correction records the Arkane run behind the exported energy is recorded as having applied.

    The records are recomputed from the Arkane key matched for the run's level, which says nothing about whether
    Arkane was told to apply them. The ``atom_energy`` records stay only when ``atom_applied`` is ``True`` and the
    ``bond_additivity`` records only when ``bond_applied`` is ``True``; a ``False`` or a ``None`` (unrecorded) switch
    drops its records, so no correction is stated that the run is not known to have applied."""
    kept = set()
    if atom_applied is True:
        kept.add('atom_energy')
    if bond_applied is True:
        kept.add('bond_additivity')
    return [record for record in records if record.get('correction_type') not in ('atom_energy', 'bond_additivity')
            or record.get('correction_type') in kept]


def _compute_species_corrections(
    species_dict: dict,
    aec_key: str | None,
    bac_type: str | None,
    project_directory: str,
) -> dict[str, dict]:
    """Compute per-species AEC/BAC totals + components by delegating to
    Arkane's correction functions through ``arc/scripts/get_species_corrections.py``
    in the RMG conda environment.

    ``aec_key`` is the atom-energy-section key ARC hands Arkane as its model
    chemistry. Both correction kinds are computed from it in a single invocation,
    and nothing is computed when it is ``None``.

    Returns a dict ``{label: {'aec': {...}, 'bac': {...}}}``. Species whose
    inputs are insufficient (no xyz, no bonds, etc.) are omitted; per-species
    Arkane errors land as ``aec_error``/``bac_error`` keys and are dropped
    silently when the result is consumed (the correction is omitted for that
    species). Returns ``{}`` on any whole-batch failure so that downstream
    output.yml writing proceeds without corrections rather than aborting.
    """
    if aec_key is None:
        return {}

    species_inputs = _build_species_correction_inputs(species_dict)
    if not species_inputs:
        return {}

    return _run_species_corrections_script(
        species_inputs=species_inputs,
        lot_str=aec_key,
        bac_type=bac_type,
        project_directory=project_directory,
    )


def _run_species_corrections_script(species_inputs: list[dict],
                                    lot_str: str,
                                    bac_type: str | None,
                                    project_directory: str,
                                    ) -> dict[str, dict]:
    """Run ``arc/scripts/get_species_corrections.py`` for a single Arkane key.

    Returns ``{label: {...}}`` with the raw blocks the script emitted, or ``{}`` when
    the invocation failed.
    """
    script_path = os.path.join(ARC_PATH, 'arc', 'scripts', 'get_species_corrections.py')

    tmp_dir = os.path.join(project_directory, 'output')
    os.makedirs(tmp_dir, exist_ok=True)
    fd_in, tmp_in = tempfile.mkstemp(dir=tmp_dir, suffix='.spc_corr_input.yml')
    fd_out, tmp_out = tempfile.mkstemp(dir=tmp_dir, suffix='.spc_corr_output.yml')
    try:
        os.close(fd_in)
        os.close(fd_out)
        save_yaml_file(path=tmp_in, content={
            'level_of_theory': lot_str,
            'bac_type': bac_type,
            'species': species_inputs,
        })

        command = rmg_env_command(py_args=[script_path, tmp_in, tmp_out])
        _, stderr = execute_command(command=command, shell=True, executable='/bin/bash')
        if stderr:
            logger.warning(f'get_species_corrections.py stderr: {stderr}')

        result = read_yaml_file(tmp_out) or {}
        out: dict[str, dict] = {}
        for entry in (result.get('species') or []):
            label = entry.get('label')
            if label is None:
                continue
            out[label] = {k: v for k, v in entry.items() if k != 'label'}
        return out
    except Exception as e:
        logger.warning(f'Per-species correction computation failed: {e}')
        return {}
    finally:
        for p in (tmp_in, tmp_out):
            try:
                os.unlink(p)
            except OSError:
                logger.debug(f'Failed to remove temporary file {p!r}', exc_info=True)


_AEC_TABLE_UNIT = 'hartree'
_BAC_TABLE_UNIT = 'kcal_mol'


def _flat_parameter_values(table: dict | None) -> dict[str, float] | None:
    """
    Return ``table`` as a sorted ``{name: float}`` mapping, or ``None``.

    ``None`` is returned for an absent table and for any table that is not a
    flat mapping of scalars — Arkane's Melius corrections are a nested
    ``{atom_corr: {...}, bond_corr_length: {...}, ...}`` structure that cannot
    be expressed as a parameter table. Checking the shape here keeps the
    caller's ``bac_type`` test a statement about which correction model carries
    a parameter table, rather than the only thing standing between a nested
    table and a ``TypeError`` inside ``write_output_yml``, whose caller logs and
    discards the exception along with the entire document.
    """
    if not table or not isinstance(table, dict):
        return None
    if any(isinstance(value, (dict, list, tuple)) for value in table.values()):
        return None
    try:
        return {str(key): float(value) for key, value in sorted(table.items())}
    except (TypeError, ValueError):
        logger.debug('Correction parameter table is not a flat numeric mapping', exc_info=True)
        return None


def _build_energy_corrections_for_species(
    label: str,
    species_corrections: dict[str, dict],
    arkane_level_of_theory,
    bac_type: str | None,
    *,
    aec_table: dict | None = None,
    bac_table: dict | None = None,
    aec_key: str | None = None,
    bac_key: str | None = None,
) -> list[dict]:
    """Build tool-neutral per-species correction facts for ``output.yml``.

    ARC records the correction model, total, native component decomposition,
    the level of theory ARC ran at, the Arkane database key ARC matched in the
    section this correction's parameters live in, and the parameter table it
    actually used. The atom-energy record reports ``aec_key`` as its
    ``matched_arkane_key`` and the bond-additivity record reports ``bac_key``;
    either is ``None`` when nothing matched in that section. The BAC
    ``parameter_table`` is emitted only when ``bac_key`` equals ``aec_key``, the
    key ARC hands Arkane as its model chemistry. Consumers own any
    database-specific scheme enums, roles, nesting, or reference rows. Failure rows
    from the helper script are omitted independently.

    Arkane's Petersson correction skips a bond type that has no parameter and applies the rest, so the
    ``components`` of a bond-additivity record are the bonds with a parameter, which sum to ``total``, and
    ``skipped_components`` lists the other bonds as ``{bond, count}``. It is ``None`` where it does not apply, for an
    atom-energy record and a Melius record.
    """
    entry = species_corrections.get(label) or {}
    applied: list[dict] = []
    lot_dict = _level_to_dict(arkane_level_of_theory)

    aec_block = entry.get('aec')
    if aec_block and aec_block.get('value') is not None:
        correction: dict = {
            'correction_type': 'atom_energy',
            'model': 'arkane_atom_energy',
            'level_of_theory': lot_dict,
            'matched_arkane_key': aec_key,
            'total': {
                'value': float(aec_block['value']),
                'unit': aec_block.get('value_unit', 'hartree'),
            },
            'components': aec_block.get('components') or [],
            'skipped_components': None,
        }
        if aec_table:
            # NOT per-atom corrections: these are Arkane's bare atomic
            # electronic energies, which Arkane *subtracts* as a reference. The
            # per-atom quantity that actually sums to ``total`` is
            # ``components[].contribution_value``, which additionally carries
            # the ``atom_hf - atom_thermal`` term absent here. Naming this
            # ``parameter_table`` invited consumers to reconstruct the
            # correction as ``sum(count * value)``, which gets both the sign and
            # ~10^5 kJ/mol wrong.
            correction['reference_atom_energies'] = {
                'unit': _AEC_TABLE_UNIT,
                'applied_as': 'subtracted',
                'values': {str(k): float(v) for k, v in sorted(aec_table.items())},
            }
        applied.append(correction)

    bac_block = entry.get('bac')
    if bac_block and bac_block.get('value') is not None and bac_type in ('p', 'm'):
        components = bac_block.get('components') or []
        skipped_components = [{'bond': c['key'], 'count': c['multiplicity']}
                              for c in components if c.get('parameter_value') is None]
        components = [c for c in components if c.get('parameter_value') is not None]
        correction = {
            'correction_type': 'bond_additivity',
            'model': {'p': 'petersson', 'm': 'melius'}[bac_type],
            'level_of_theory': lot_dict,
            'matched_arkane_key': bac_key,
            'total': {
                'value': float(bac_block['value']),
                'unit': bac_block.get('value_unit', 'kcal_mol'),
            },
            'components': components,
            'skipped_components': skipped_components if bac_type == 'p' else None,
        }
        if bac_type == 'p' and bac_key == aec_key:
            values = _flat_parameter_values(bac_table)
            if values:
                correction['parameter_table'] = {
                    'unit': _BAC_TABLE_UNIT,
                    'values': values,
                }
        applied.append(correction)

    return applied


def _compute_point_groups(species_dict: dict, project_directory: str) -> dict[str, str | None]:
    """
    Compute point groups for all species via the ``symmetry`` binary in the RMG env.

    Calls ``arc/scripts/get_point_groups.py`` as a subprocess in the RMG conda
    environment (same pattern as save_arkane_thermo.py).  Returns a dict mapping
    each species label to its point group string (e.g. ``'C2v'``) or ``None``.
    On any failure the function returns an empty dict so callers get ``None`` for
    every species rather than crashing the run.
    """
    script_path = os.path.join(ARC_PATH, 'arc', 'scripts', 'get_point_groups.py')

    # Build input dict: {label: {symbols: [...], coords: [...]}}
    pg_input: dict[str, Any] = {}
    for label, spc in species_dict.items():
        xyz = _get_exported_xyz(spc)
        if xyz is None:
            continue
        symbols = list(xyz.get('symbols', []))
        coords = [list(row) for row in xyz.get('coords', [])]
        if symbols and coords:
            pg_input[label] = {'symbols': symbols, 'coords': coords}

    if not pg_input:
        return {}

    tmp_dir = os.path.join(project_directory, 'output')
    os.makedirs(tmp_dir, exist_ok=True)
    fd_in, tmp_in = tempfile.mkstemp(dir=tmp_dir, suffix='.pg_input.yml')
    fd_out, tmp_out = tempfile.mkstemp(dir=tmp_dir, suffix='.pg_output.yml')
    try:
        os.close(fd_in)
        os.close(fd_out)
        save_yaml_file(path=tmp_in, content=pg_input)

        command = rmg_env_command(py_args=[script_path, tmp_in, tmp_out])
        _, stderr = execute_command(command=command, shell=True, executable='/bin/bash')
        if stderr:
            logger.warning(f'get_point_groups.py stderr: {stderr}')

        result = read_yaml_file(tmp_out) or {}
        return {str(k): (str(v) if v is not None else None) for k, v in result.items()}
    except Exception as e:
        logger.warning(f'Could not compute point groups: {e}')
        return {}
    finally:
        for p in (tmp_in, tmp_out):
            try:
                os.unlink(p)
            except OSError:
                logger.debug(f'Failed to remove temporary file {p!r}', exc_info=True)


def _spc_to_dict(spc, output_dict: dict, project_directory: str,
                  point_groups: dict | None = None, irc_requested: bool = True,
                  software_by_job: dict[str, str | None] | None = None,
                  freq_level=None) -> dict:
    """Build the per-species/TS section for output.yml.

    ``software_by_job`` is an optional ``{'opt': name, 'freq': name,
    'sp': name}`` map that lets this function emit per-job input-deck
    paths (``opt_input``, ``freq_input``, ``sp_input``) alongside the
    log paths. When omitted (the back-compat path), the input fields
    come out as ``None`` and downstream consumers proceed with logs only.

    ``freq_level`` is the frequency job's level of theory, needed to state
    ``freq_hessian_method`` for an ESS that does not print which Hessian it
    built; without it that field comes out as ``None``.
    """
    software_by_job = software_by_job or {}
    label = spc.label
    entry = output_dict.get(label, {})
    converged = entry.get('convergence') is True
    paths = entry.get('paths', {})

    d: dict[str, Any] = {
        'label': label,
        'original_label': spc.original_label,
        'charge': spc.charge,
        'multiplicity': spc.multiplicity,
        'converged': converged,
        'is_ts': spc.is_ts,
    }

    # ── molecular identity ───────────────────────────────────────────────────
    if spc.is_ts:
        d['smiles'] = None
        d['inchi'] = None
        d['inchi_key'] = None
        d['formula'] = _safe(lambda: spc.mol.get_formula()) if spc.mol is not None else None
    elif spc.mol is not None:
        mol_copy = spc.mol.copy(deep=True)
        d['smiles'] = _safe(lambda: mol_copy.to_smiles())
        d['inchi'] = _safe(lambda: mol_copy.to_inchi())
        d['inchi_key'] = _safe(lambda: mol_copy.to_inchi_key())
        d['formula'] = _safe(lambda: spc.mol.get_formula())
    else:
        d['smiles'] = None
        d['inchi'] = None
        d['inchi_key'] = None
        d['formula'] = None

    # ── final geometry ──────────────────────────────────────────────────────
    xyz = _get_exported_xyz(spc)
    if xyz is None and converged and not spc.is_ts \
            and spc.mol is not None and len(spc.mol.atoms) == 1:
        # Monoatomic species skip opt entirely (nothing to optimize), so neither
        # final_xyz nor initial_xyz is populated. Synthesize the trivial
        # one-atom geometry so output.yml carries a usable xyz for downstream
        # consumers.
        xyz = {'symbols': (spc.mol.atoms[0].element.symbol,),
               'coords': ((0.0, 0.0, 0.0),)}
    d['xyz'] = xyz_to_str(xyz) if xyz is not None else None
    d['xyz_isotopes'] = _get_log_stated_isotopes(
        _abs_existing_path(paths.get('geo') or paths.get('composite') or None, project_directory),
        xyz['symbols']) if xyz is not None and spc.final_xyz is not None else None

    raw_conformers = getattr(spc, 'conformers', None) or []
    if raw_conformers:
        d['conformers'] = [xyz_to_str(c) if not isinstance(c, str) else c
                           for c in raw_conformers]
        raw_energies = getattr(spc, 'conformer_energies', None) or []
        d['conformer_energies'] = list(raw_energies)
        d['conformers_isotopes'] = _get_conformers_isotopes(spc, raw_conformers, project_directory)
        d['conformer_ess_software'], d['conformer_ess_version'] = \
            _get_conformer_ess(spc, len(raw_conformers), project_directory)
    d['conformer_levels'], d['conformer_energy_kind'], d['conformer_energy_level'], d['conformer_force_field'] = \
        _conformer_provenance_to_fields(spc, len(raw_conformers))

    # ``opt_input_xyz`` and the coarse-opt xyz fields are populated below,
    # AFTER coarse-opt parsing (we need ``coarse_final_xyz`` to set the
    # fine opt's input correctly when coarse ran).

    # ── is monoatomic? (drives null-vs-value for freq/statmech) ─────────────
    is_mono = spc.is_monoatomic() is True

    # ── energies ────────────────────────────────────────────────────────────
    if converged and spc.e_elect is not None:
        d['sp_energy_hartree'] = spc.e_elect / E_h_kJmol
    else:
        d['sp_energy_hartree'] = None

    d['zpe_hartree'] = _parse_zpe(paths.get('freq') or None, project_directory) \
        if converged and not is_mono else None

    d['opt_converged'] = entry.get('job_types', {}).get('opt') if converged else None

    # ── coarse opt (null if fine grid wasn't used or job didn't run) ────────
    # When coarse ran, its parsed final geometry becomes both the coarse
    # opt's output and (semantically) the fine opt's input — see the
    # ``opt_input_xyz`` resolution below.
    coarse_path = paths.get('geo_coarse') or None
    coarse_final_xyz: str | None = None
    coarse_final_isotopes: list[int] | None = None
    if converged and coarse_path:
        d['coarse_opt_log'] = _make_rel_path(coarse_path, project_directory)
        d['coarse_opt_n_steps'], d['coarse_opt_final_energy_hartree'], coarse_final_xyz, coarse_final_isotopes = \
            _parse_opt_log(coarse_path, project_directory)
    else:
        d['coarse_opt_log'] = None
        d['coarse_opt_n_steps'] = None
        d['coarse_opt_final_energy_hartree'] = None

    # ── fine opt (or only opt if no fine grid) ─────────────────────────────
    # We discard the geometry here — ``xyz`` (set above) already carries
    # the fine opt's final geometry via ``spc.final_xyz``, and re-parsing
    # the log would just produce the same content with different rounding.
    if converged:
        d['opt_n_steps'], d['opt_final_energy_hartree'], _, fine_opt_isotopes = _parse_opt_log(
            paths.get('geo') or None, project_directory
        )
    else:
        d['opt_n_steps'], d['opt_final_energy_hartree'] = None, None
        fine_opt_isotopes = None

    # ── opt input/output geometry semantics ────────────────────────────────
    # When coarse ran (a real two-stage opt), the geometry chain is:
    #   spc.initial_xyz  →  coarse_opt  →  coarse_final_xyz  →  opt  →  xyz
    # When coarse didn't run (single-stage opt), there's no intermediate
    # and the chain is just:
    #   spc.initial_xyz  →  opt  →  xyz
    #
    # ``opt_input_xyz`` always means "what was submitted to the FINE opt"
    # — the coarse output if coarse ran, else the species' initial xyz.
    # ``coarse_opt_input_xyz`` and ``coarse_opt_output_xyz`` are non-null
    # only when a coarse opt actually ran AND its log was parseable.
    initial_xyz_str = xyz_to_str(spc.initial_xyz) if spc.initial_xyz is not None else None
    if coarse_final_xyz is not None:
        # Real two-stage opt with parseable coarse output.
        d['coarse_opt_input_xyz'] = initial_xyz_str
        d['coarse_opt_input_xyz_isotopes'] = coarse_final_isotopes
        d['coarse_opt_output_xyz'] = coarse_final_xyz
        d['coarse_opt_output_xyz_isotopes'] = coarse_final_isotopes
        d['opt_input_xyz'] = coarse_final_xyz
        d['opt_input_xyz_isotopes'] = fine_opt_isotopes
    else:
        # Either no coarse stage, or coarse ran but its geometry wasn't
        # parseable. Consumers cannot reconstruct the standalone coarse stage
        # without its output geometry. Honest-empty beats fake provenance.
        d['coarse_opt_input_xyz'] = None
        d['coarse_opt_input_xyz_isotopes'] = None
        d['coarse_opt_output_xyz'] = None
        d['coarse_opt_output_xyz_isotopes'] = None
        d['opt_input_xyz'] = initial_xyz_str
        d['opt_input_xyz_isotopes'] = fine_opt_isotopes if initial_xyz_str is not None else None

    # ── freq results ────────────────────────────────────────────────────────
    if is_mono or not converged:
        d['freq_n_imag'] = None
        d['imag_freq_cm1'] = None
        d['imaginary_frequencies_cm1'] = None
    else:
        imaginary_freqs = _get_imaginary_freqs(spc)
        d['freq_n_imag'] = len(imaginary_freqs) if imaginary_freqs is not None else None
        d['imag_freq_cm1'] = min(imaginary_freqs) if imaginary_freqs else None
        d['imaginary_frequencies_cm1'] = imaginary_freqs or None

    # ── log file paths (run-relative) ────────────────────────────────────────
    d['opt_log'] = _make_rel_path(paths.get('geo') or None, project_directory)
    d['freq_log'] = _make_rel_path(paths.get('freq') or None, project_directory)
    d['sp_log'] = _make_rel_path(paths.get('sp') or None, project_directory)
    d['composite_log'] = _make_rel_path(paths.get('composite') or None, project_directory)
    d['opt_route'] = _get_ess_route(paths.get('geo') or None, project_directory) if converged else None
    d['freq_route'] = _get_ess_route(paths.get('freq') or None, project_directory) if converged else None
    d['sp_route'] = _get_ess_route(paths.get('sp') or None, project_directory) if converged else None
    d['composite_route'] = _get_ess_route(paths.get('composite') or None, project_directory) if converged else None
    d['composite_step_routes'] = _get_composite_step_routes(paths.get('composite') or None, project_directory) \
        if converged else None
    d['opt_dipole_moment_debye'], d['opt_dipole_moment_density'] = _get_gaussian_dipole_moment(
        paths.get('geo') or None, project_directory) if converged and not is_mono and spc.charge == 0 \
        else (None, None)
    d['freq_polarizability_angstrom3'] = _get_gaussian_polarizability(
        paths.get('freq') or None, project_directory) if converged and not is_mono else None

    # ── wavefunction stability diagnostic (null unless the job ran) ──────────
    stability = _parse_wavefunction_stability(paths.get('stability') or None, project_directory)
    d['wavefunction_stability'] = stability

    # ── which source decided the open-shell character, and the references used ──
    d['scf_reference'] = _scf_reference_block(spc, stability, project_directory)

    # ── S**2 spin-contamination diagnostic (sp calc, open-shell only) ───────
    d['sp_spin_diagnostic'] = _parse_spin_diagnostic(
        paths.get('sp') or None,
        paths.get('freq') or None,
        paths.get('geo') or None,
        spc.multiplicity,
        project_directory,
    ) if converged else None
    # ── ESS input deck paths ────────────────────────────────────────────────
    # Same directory as the corresponding log, with the per-software
    # filename from settings['input_filenames']. None when the file isn't
    # on disk (the consumer treats that as "no deck available").
    d['opt_input'] = _derive_input_path(
        paths.get('geo') or None, software_by_job.get('opt'), project_directory,
    )
    d['freq_input'] = _derive_input_path(
        paths.get('freq') or None, software_by_job.get('freq'), project_directory,
    )
    d['sp_input'] = _derive_input_path(
        paths.get('sp') or None, software_by_job.get('sp'), project_directory,
    )
    d['composite_input'] = _derive_input_path(
        paths.get('composite') or None, _identify_log_ess(paths.get('composite') or None, project_directory),
        project_directory,
    )

    # ── held-fixed coordinate constraints ──────────────────────────────────
    # Best-effort: parse the input deck (preferred — exact ARC-emitted form)
    # falling back to the log when no deck is on disk. Failures here never
    # fail output.yml generation; they just emit ``[]`` for that calc.
    d['opt_constraints'] = _parse_calc_constraints(
        d.get('opt_input'), paths.get('geo') or None,
        software_by_job.get('opt'), project_directory,
    )
    d['freq_constraints'] = _parse_calc_constraints(
        d.get('freq_input'), paths.get('freq') or None,
        software_by_job.get('freq'), project_directory,
    )
    d['freq_hessian_method'] = _get_freq_hessian_method(
        d.get('freq_input'), paths.get('freq') or None,
        software_by_job.get('freq'), freq_level, spc, project_directory,
    ) if converged else None

    # ── per-calc final effective settings ──────────────────────────────────
    # These are the calc-specific scientific knobs (grid, scf_convergence,
    # opt_convergence, max_*_cycles, symmetry, guess, …) that defined the
    # FINAL job — distinct from ``level_of_theory`` identity and from
    # scheduler/operational fields (server, queue, runtime, ess_trsh —
    # all explicitly out of scope for the database).
    #
    # Today the producer can prove exactly one signal from observable run
    # state: which stage of ARC's two-stage opt convention this calc
    # represents. When ``coarse_opt_log`` is non-null, the workflow was
    # coarse-then-fine, so the fine opt is the ``"fine"`` stage and the
    # coarse opt is the ``"coarse"`` stage. We name the key
    # ``optimization_stage`` (rather than ``fine: bool``) so consumers
    # don't have to know ARC's convention to read it — and so it doesn't
    # collide with ESS-vendor "fine" keywords that may end up in the
    # same dict later.
    #
    # For single-stage opts and for freq/sp we have no equally reliable
    # signal from filesystem state alone, so the field is omitted
    # (``None``) — better than fabricating defaults. Future producers
    # (a JobAdapter→species-record handoff carrying the raw
    # ``self.fine`` / ``self.grid`` / ``self.args`` per job) can grow
    # these dicts without touching the adapter wiring.
    if coarse_final_xyz is not None:
        d['opt_final_settings'] = {'optimization_stage': 'fine'}
        d['coarse_opt_final_settings'] = {'optimization_stage': 'coarse'}
    else:
        d['opt_final_settings'] = None
        d['coarse_opt_final_settings'] = None
    d['freq_final_settings'] = None
    d['sp_final_settings'] = None
    d['sp_constraints'] = _parse_calc_constraints(
        d.get('sp_input'), paths.get('sp') or None,
        software_by_job.get('sp'), project_directory,
    )

    # ── ESS software version (from SP log, or fall back to geo/freq log) ──
    d['ess_versions'] = _get_ess_versions(paths, project_directory) if converged else None
    d['ess_software'] = _get_ess_software(paths, project_directory) if converged else None
    d['levels'] = _species_levels_to_dict(entry, spc.is_ts)
    d['sp_t1_diagnostic'] = _get_t1_diagnostic(paths.get('sp') or None, project_directory) if converged else None
    if not spc.is_ts:
        irc_label = getattr(spc, 'irc_label', None)
        endpoint_of = irc_label if isinstance(irc_label, str) and irc_label else None
        direction = entry.get('irc_direction') if isinstance(entry, dict) else None
        d['irc_endpoint_of'] = endpoint_of
        d['irc_endpoint_direction'] = direction if endpoint_of and direction in ('forward', 'reverse') else None

    if spc.is_ts:
        d['chosen_ts_method'] = getattr(spc, 'chosen_ts_method', None)
        d['successful_ts_methods'] = getattr(spc, 'successful_methods', None) or None
        d['neb_log'] = _make_rel_path(paths.get('neb') or None, project_directory)
        d['gsm_log'] = _make_rel_path(paths.get('gsm') or None, project_directory)

        chosen_index = getattr(spc, 'chosen_ts', None)
        chosen_guess = None
        if isinstance(chosen_index, int):
            chosen_guess = next(
                (guess for guess in (getattr(spc, 'ts_guesses', None) or [])
                 if getattr(guess, 'index', None) == chosen_index),
                None,
            )
        if chosen_guess is not None:
            method = getattr(chosen_guess, 'method', None)
            method = method.strip().lower() if isinstance(method, str) and method.strip() else None
            method_sources = []
            for source in (getattr(chosen_guess, 'method_sources', None) or []):
                if not isinstance(source, str) or not source.strip():
                    continue
                source = source.strip().lower()
                if source not in method_sources:
                    method_sources.append(source)
            if method is not None and method not in method_sources:
                method_sources.insert(0, method)
            # Only expose stable attribution fields. TSGuess.as_dict() also
            # contains absolute paths, timestamps, geometries, and diagnostic
            # state that do not belong in this result-contract seam.
            d['ts_guesses'] = [{
                'index': getattr(chosen_guess, 'index', chosen_index),
                'chosen': True,
                'method': method,
                'method_sources': method_sources,
            }]
            chosen_log = getattr(chosen_guess, 'log_path', None)
            log_field = _ts_guess_log_field_for_method(getattr(chosen_guess, 'method', None))
            if chosen_log and log_field and not d.get(log_field):
                d[log_field] = _make_rel_path(chosen_log, project_directory)
            if not d.get('neb_log') and not d.get('gsm_log'):
                source_paths = getattr(chosen_guess, 'method_source_paths', None) or {}
                for source in (getattr(chosen_guess, 'method_sources', None) or []):
                    source_field = _ts_guess_log_field_for_method(source)
                    source_log = source_paths.get(source)
                    if source_field and source_log:
                        d[source_field] = _make_rel_path(source_log, project_directory)
                        break
        else:
            d['ts_guesses'] = []

        irc_paths = list(paths.get('irc') or [])
        d['irc_logs'] = [_make_rel_path(path, project_directory) for path in irc_paths]
        d['irc_log_routes'] = [_get_ess_route(path or None, project_directory) for path in irc_paths]
        irc_directions = list(paths.get('irc_directions') or [])
        d['irc_log_directions'] = (
            irc_directions + [None] * (len(irc_paths) - len(irc_directions))
        )[:len(irc_paths)]
        irc_levels = list(paths.get('irc_levels') or [])
        d['irc_log_levels'] = [
            _recorded_level_to_dict(level)
            for level in (irc_levels + [None] * (len(irc_paths) - len(irc_levels)))[:len(irc_paths)]
        ]
        if not irc_requested:
            d['irc_converged'] = None
        else:
            d['irc_converged'] = entry.get('job_types', {}).get('irc', False)
        d['ts_checks'] = _ts_checks_to_dict(spc)
        d['irc_participant_mapping'] = _irc_participant_mapping_to_dict(spc)
        d['freq_frequencies_cm1_ess_order'], d['reaction_coordinate_mode_index'] = \
            _get_ts_frequencies_in_ess_order(spc, converged, d['freq_log'], project_directory)
        d['nmd_forced'] = _get_nmd_forced(spc)
        d['neb_succeeded'] = _get_neb_succeeded(spc)
        d['rxn_label'] = spc.rxn_label

    # ── thermochemistry (non-TS converged species only) ──────────────────────
    if not spc.is_ts and converged and spc.thermo is not None and spc.thermo.H298 is not None:
        d['thermo'] = _thermo_to_dict(spc.thermo)
    else:
        d['thermo'] = None

    rotor_scans = _build_rotor_scans(spc, project_directory) if not is_mono and converged else []

    if not is_mono and converged:
        pg = (point_groups or {}).get(label)
        d['statmech'] = _statmech_to_dict(
            spc, project_directory, point_group=pg,
            scan_keys={entry['key'] for entry in rotor_scans},
        )
    else:
        d['statmech'] = None

    # ── additional calculations (rotor scans, etc.) ─────────────────────────
    d['rotor_scans'] = rotor_scans

    return d


def _get_ts_frequencies_in_ess_order(spc,
                                     converged: bool,
                                     freq_log: str | None = None,
                                     project_directory: str | None = None,
                                     ) -> tuple[list[float] | None, int | None]:
    """
    Return the frequencies of a transition state as the frequency job printed them, and the 1-based index into that
    list of the reaction-coordinate mode.

    ``ARCSpecies.freqs`` is the list the frequency job reported, unsorted and with its imaginary modes in place. The
    normal mode displacement check indexes the frequencies it parsed from an output file, which need not be the same
    list: a parser may concatenate several frequency blocks (Gaussian does), and a composite log yields a prefix. The
    check therefore records the 0-based position of the mode it analysed (``nmd_record['mode_index']``), the number of
    modes it indexed (``n_modes``) and the output file it parsed (``freq_log_path``). The index here is that position
    plus one. It is given only when the check genuinely passed (``ts_checks['NMD']`` is ``True`` and was not forced by
    ``skip_nmd``), ``n_modes`` equals the number of listed frequencies, ``freq_log_path`` is the exported frequency
    log, the record holds an integer position within the listed frequencies, and the frequency at that position is
    imaginary, which the analysed mode always is. A record without ``n_modes`` or ``freq_log_path`` (an older
    restart) gives no index.

    Args:
        spc (ARCSpecies): The transition state.
        converged (bool): Whether the species converged.
        freq_log (str, optional): The exported frequency log, relative to ``project_directory`` or absolute.
        project_directory (str, optional): The directory a relative ``freq_log`` resolves against.

    Returns:
        tuple[list[float] | None, int | None]: The frequencies, ``None`` when the species is not converged or has
                                               none; and the index, ``None`` when it is not established.
    """
    freqs = getattr(spc, 'freqs', None)
    if not converged or freqs is None or not len(freqs):
        return None, None
    try:
        ess_order = [float(freq) for freq in freqs]
    except (TypeError, ValueError):
        logger.debug(f'Could not read the frequencies of {getattr(spc, "label", None)}', exc_info=True)
        return None, None
    if (getattr(spc, 'ts_checks', None) or {}).get('NMD') is not True:
        return ess_order, None
    record = getattr(spc, 'nmd_record', None)
    record = record if isinstance(record, dict) else dict()
    mode_index = record.get('mode_index')
    if record.get('forced') is True or not _is_int(mode_index) or not 0 <= mode_index < len(ess_order) \
            or ess_order[mode_index] >= 0:
        return ess_order, None
    n_modes = record.get('n_modes')
    if not _is_int(n_modes) or n_modes != len(ess_order):
        return ess_order, None
    recorded_log, exported_log = record.get('freq_log_path'), freq_log
    if not isinstance(recorded_log, str) or not recorded_log or not isinstance(exported_log, str) or not exported_log:
        return ess_order, None
    base = project_directory or os.getcwd()
    if os.path.normpath(os.path.join(base, recorded_log)) != os.path.normpath(os.path.join(base, exported_log)):
        return ess_order, None
    return ess_order, mode_index + 1


def _get_nmd_forced(spc) -> bool | None:
    """
    Whether the normal mode displacement verdict of a TS was forced to a pass, as the check's own record states.

    Args:
        spc (ARCSpecies): The transition state.

    Returns:
        bool | None: ``True`` when the check failed and ``skip_nmd`` forced ``ts_checks['NMD']`` to ``True``;
                     ``False`` when the check ran and was not forced; ``None`` when there is no check record or
                     ``ts_checks['NMD']`` is not a boolean.
    """
    verdict = (getattr(spc, 'ts_checks', None) or {}).get('NMD')
    record = getattr(spc, 'nmd_record', None)
    if not isinstance(verdict, bool) or not isinstance(record, dict) or not record:
        return None
    if record.get('forced') is True:
        return True if verdict else None
    return False


NEB_CONVERGENCE_BANNER = 'THE NEB OPTIMIZATION HAS CONVERGED'


def _neb_log_states_convergence(path: str) -> bool | None:
    """
    Whether an ORCA NEB log prints the line stating that the NEB optimization has converged.

    Args:
        path (str): The path to the NEB log.

    Returns:
        bool | None: ``True`` when the log prints the convergence line, ``False`` when it does not, and ``None`` when
                     the file cannot be read.
    """
    if not isinstance(path, str) or not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', errors='replace') as f:
            return any(NEB_CONVERGENCE_BANNER in line for line in f)
    except OSError:
        logger.debug(f'Could not read the NEB log {path}', exc_info=True)
        return None


def _get_neb_succeeded(spc) -> bool | None:
    """
    Whether the NEB log of an ``orca_neb`` guess recorded for a TS states that the NEB optimization converged.

    A guess counts when its own method or any of its merged ``method_sources`` is ``orca_neb``. The log read is the
    one recorded for that method on the guess (``log_path`` for a guess of that method, ``method_source_paths`` for a
    merged one). A guess with no recorded log, or a log that cannot be read, states nothing. Whether the guess holds
    a geometry is not consulted.

    Args:
        spc (ARCSpecies): The transition state.

    Returns:
        bool | None: ``True`` when the log of any recorded ``orca_neb`` guess prints ``THE NEB OPTIMIZATION HAS
                     CONVERGED``; ``False`` when ``orca_neb`` guesses were recorded with readable logs and none prints
                     it; ``None`` otherwise.
    """
    outcomes = list()
    for guess in getattr(spc, 'ts_guesses', None) or list():
        method = getattr(guess, 'method', None)
        sources = [method] + list(getattr(guess, 'method_sources', None) or list())
        if not any(isinstance(source, str) and source.strip().lower() == 'orca_neb' for source in sources):
            continue
        if isinstance(method, str) and method.strip().lower() == 'orca_neb':
            path = getattr(guess, 'log_path', None)
        else:
            path = (getattr(guess, 'method_source_paths', None) or dict()).get('orca_neb')
        if not path:
            continue
        outcome = _neb_log_states_convergence(path)
        if outcome is not None:
            outcomes.append(outcome)
    if not outcomes:
        return None
    return any(outcomes)


def _has_default_isotopes(xyz: dict) -> bool:
    """
    Whether every atom of ``xyz`` carries its most common isotope, or the geometry states no isotopes. ``False`` when
    the symbols or isotopes cannot be compared.
    """
    isotopes = xyz.get('isotopes')
    if not isotopes:
        return True
    symbols = xyz.get('symbols')
    if not isinstance(isotopes, (list, tuple)) or not isinstance(symbols, (list, tuple)):
        return False
    try:
        return all(isotope == get_most_common_isotope_for_element(symbol) for symbol, isotope in zip(symbols, isotopes))
    except (KeyError, TypeError, ValueError):
        logger.debug(f'Could not compare the isotopes {isotopes} with the most common ones of {symbols}', exc_info=True)
        return False


def _get_rigid_rotor_kind(spc, point_group: str | None = None) -> str | None:
    """
    The rigid-rotor kind of a species, read from its exported point group alone.

    A species with one atom is an ``atom``. Otherwise the point group fixes how many principal moments of inertia
    the symmetry forces to be equal, because the inertia tensor is invariant under every symmetry operation. An
    improper rotation S_n(theta) equals the inversion times the proper rotation C(theta + pi), and the inversion acts
    trivially on the tensor, so S_n acts on it as the proper rotation C(theta + pi) (a mirror plane, S_1, acts as a
    C2 rotation). The mapping is:

    - C∞v and D∞h (``Cinfv``, ``Dinfh``, ``C*v``, ``D*h``): ``linear``, one moment vanishes.
    - T, Td, Th, O, Oh, I, Ih: ``spherical_top``, the rotations transform as a triply degenerate irreducible
      representation, so the tensor is isotropic.
    - C_n, C_nv, C_nh, D_n, D_nh and D_nd with n >= 3: ``symmetric_top``, a proper C_n axis with n >= 3 equates the two
      moments perpendicular to it.
    - D2d and S_n with even n >= 4 (S4, S6, S8, ...): ``symmetric_top``, an S_n axis acts on the tensor as the
      rotation by 2*pi/n + pi, which has order 3 or more for every even n >= 4: S4 acts as C4 cubed (order 4), S6 as
      C3 inverse (order 3) and S8 with order 8, although none of these contains a proper C_n axis with n >= 3.
      Allene (D2d), whose S4 axis is the one that does this, is a prolate symmetric top.
    - C1, Cs, Ci (S2), C2, C2v, C2h, D2, D2h: ``asymmetric_top``, no axis of order 3 or more and no S4 axis, so nothing
      forces two moments to agree.

    A symmetric top here is a symmetry-enforced one; an accidental equality of moments is not stated.

    ``None`` when the point group is ``None`` or is not one of the above, and when the species carries non-default
    isotopes, because the point group is that of the geometry without isotopic labels and does not apply to a
    labelled one. No moment of inertia is computed and no tolerance is used.

    Args:
        spc: The species.
        point_group (str, optional): The exported point group symbol of the exported geometry.

    Returns:
        str | None: ``atom``, ``linear``, ``spherical_top``, ``symmetric_top``, ``asymmetric_top`` or ``None``.
    """
    if spc.is_monoatomic() is True:
        return 'atom'
    if not isinstance(point_group, str):
        return None
    xyz = _get_exported_xyz(spc)
    if xyz is not None and not _has_default_isotopes(xyz):
        return None
    if re.fullmatch(r'C(inf|\*|∞)v|D(inf|\*|∞)h', point_group):
        return 'linear'
    if re.fullmatch(r'[TOI][dh]?', point_group):
        return 'spherical_top'
    if point_group in ('C1', 'Cs', 'Ci', 'S2', 'C2', 'C2v', 'C2h', 'D2', 'D2h'):
        return 'asymmetric_top'
    match = re.fullmatch(r'([CDS])(\d+)([vhd]?)', point_group)
    if match is None:
        return None
    family, order, suffix = match.group(1), int(match.group(2)), match.group(3)
    if suffix not in {'C': ('', 'v', 'h'), 'D': ('', 'h', 'd'), 'S': ('',)}[family]:
        return None
    if family == 'S':
        return 'symmetric_top' if order >= 4 and order % 2 == 0 else None
    if family == 'D' and suffix == 'd':
        return 'symmetric_top' if order >= 2 else None
    return 'symmetric_top' if order >= 3 else None


def _get_imaginary_freqs(spc) -> list[float] | None:
    """
    Return every imaginary frequency (cm⁻¹) of a species, most negative first.

    Imaginary frequencies are reported as negative numbers. The negative frequencies of the completed
    freq job are used when it found any; otherwise the chosen TS guess's recorded imaginary
    frequencies are used, falling back to the freq job's (empty) result. The chosen guess is
    looked up by its ``TSGuess.index``, not by its position in ``ts_guesses``.

    Returns:
        Optional[list[float]]: The imaginary frequencies, an empty list when the species has none,
                               or ``None`` when no frequency data is available at all.
    """
    freqs = getattr(spc, 'freqs', None)
    negative_freqs = sorted(float(f) for f in freqs if f < 0) if freqs is not None and len(freqs) else None
    if negative_freqs:
        return negative_freqs
    try:
        chosen = spc.chosen_ts
        if chosen is not None and spc.ts_guesses:
            chosen_guess = next((tsg for tsg in spc.ts_guesses if tsg.index == chosen), None)
            im_freqs = chosen_guess.imaginary_freqs if chosen_guess is not None else None
            if im_freqs:
                return sorted(float(f) for f in im_freqs)
    except Exception as e:
        logger.debug('Failed to obtain imaginary frequencies from ts_guesses for %s, got %s: %s',
                     spc.label, type(e).__name__, e)
    return negative_freqs


def _thermo_to_dict(thermo) -> dict:
    """Convert a ThermoData object to a plain, unit-labeled dict.

    ``standard_state_pressure_pa`` is the standard state every entropy, free energy and
    NASA fit in the returned dict belongs to, including those inside ``thermo_points``.
    It is ``None`` when the thermo did not come from a statmech run that recorded one.

    ``atom_corrections_applied`` and ``bond_corrections_applied`` are the ``useAtomCorrections``
    and ``useBondCorrections`` values of the Arkane run that produced the thermo, and are
    ``None`` when that run did not record them. ``atom_corrections_level`` is the level whose
    atom energies that run subtracted, in the shape of the document's other levels; it is
    ``None`` unless ``atom_corrections_applied`` is ``True``.
    """
    def _scalar(x):
        """Extract the numeric value from a (value, units) tuple or a plain number."""
        if isinstance(x, (list, tuple)) and len(x) >= 1:
            return x[0]
        return x

    t: dict[str, Any] = {
        'h298_kj_mol': thermo.H298,
        's298_j_mol_k': thermo.S298,
        'tmin_k': _scalar(thermo.Tmin),
        'tmax_k': _scalar(thermo.Tmax),
        'standard_state_pressure_pa': getattr(thermo, 'standard_state_pressure_pa', None),
        'atom_corrections_applied': getattr(thermo, 'atom_corrections_applied', None),
        'bond_corrections_applied': getattr(thermo, 'bond_corrections_applied', None),
        'atom_corrections_level': _level_to_dict(getattr(thermo, 'atom_corrections_level', None))
        if getattr(thermo, 'atom_corrections_applied', None) is True else None,
    }

    # ── Per-temperature thermochemistry ──────────────────────────────────────
    # ``thermo_points`` carries explicit thermodynamic facts and units
    # (Cp + H + S + G at each tabulated T). Falls back to building a
    # Cp-only point list from RMG's ``Tdata``/``Cpdata`` when only that
    # legacy form is available — H/S/G stay omitted because we don't
    # have the polynomial in scope here to evaluate them. Consumers must accept
    # either shape.
    points = getattr(thermo, 'thermo_points', None)
    if points is not None:
        t['thermo_points'] = points
    elif thermo.Tdata is not None and thermo.Cpdata is not None:
        T_list = thermo.Tdata[0] if isinstance(thermo.Tdata, (list, tuple)) else thermo.Tdata
        Cp_list = thermo.Cpdata[0] if isinstance(thermo.Cpdata, (list, tuple)) else thermo.Cpdata
        t['thermo_points'] = [{'temperature_k': float(T), 'cp_j_mol_k': float(Cp)}
                              for T, Cp in zip(T_list, Cp_list)]
    else:
        t['thermo_points'] = None

    # ── NASA polynomials ─────────────────────────────────────────────────────
    t['nasa_low'] = getattr(thermo, 'nasa_low', None)
    t['nasa_high'] = getattr(thermo, 'nasa_high', None)

    return t


def _statmech_to_dict(spc,
                      project_directory: str,
                      point_group: str | None = None,
                      scan_keys: set[str] | None = None,
                      ) -> dict:
    """Build the statmech sub-section for a non-monoatomic converged species/TS.

    ``scan_keys`` is forwarded unchanged to :func:`_get_torsions`.
    """
    # Use the cached private attribute to avoid triggering a geometry re-read
    is_linear = spc._is_linear

    rotor_kind = _get_rigid_rotor_kind(spc, point_group)

    rotor_modes = getattr(spc, 'arkane_rotor_modes', None)
    freqs = getattr(spc, 'freqs', None)
    if freqs is not None and spc.is_ts:
        # Exclude the imaginary mode (most negative frequency)
        freqs = [f for f in freqs if f >= 0]

    return {
        'e0_kj_mol': spc.e0,
        'e0_atom_corrections_applied': getattr(spc, 'e0_atom_corrections_applied', None),
        'e0_bond_corrections_applied': getattr(spc, 'e0_bond_corrections_applied', None),
        'arkane_rotors_applied': len(rotor_modes) if rotor_modes is not None else None,
        'arkane_treatment': get_arkane_treatment(rotor_modes),
        'spin_multiplicity': spc.multiplicity,
        'optical_isomers': spc.optical_isomers,
        'is_linear': is_linear,
        'external_symmetry': spc.external_symmetry,
        'point_group': point_group,
        'rigid_rotor_kind': rotor_kind,
        'harmonic_frequencies_cm1': [float(f) for f in freqs] if freqs is not None else None,
        'torsions': _get_torsions(spc, project_directory, scan_keys=scan_keys),
        'rejected_torsions': _get_rejected_torsions(spc, project_directory),
    }


def _get_torsions(spc, project_directory: str, scan_keys: set[str] | None = None) -> list[dict]:
    """Build the torsions list from spc.rotors_dict.

    Each emitted torsion carries a ``source_scan_key`` (e.g.
    ``"scan_rotor_3"``) only when :func:`_build_rotor_scans` actually emits a
    record under that key, so a torsion can never reference a scan record that
    is absent from ``rotor_scans``. Rotors with no corresponding record get the
    field set to ``None`` rather than fabricating a key that points at no calc.

    Args:
        spc: The species whose ``rotors_dict`` is being rendered.
        project_directory (str): The run's project directory.
        scan_keys (Optional[set[str]]): The keys :func:`_build_rotor_scans`
            emitted for this species. When ``None`` the keys are derived here
            via the same builder, which costs an extra parse; callers that
            already built the records should pass them in.
    """
    if not getattr(spc, 'rotors_dict', None):
        return []
    if scan_keys is None:
        scan_keys = {entry['key'] for entry in _build_rotor_scans(spc, project_directory)}
    torsions = []
    treated_rotors = [rotor for rotor in spc.rotors_dict.values() if rotor.get('success') is True]
    rotor_modes = getattr(spc, 'arkane_rotor_modes', None)
    arkane_treatments = (
        [ARKANE_TORSION_TREATMENT_BY_MODE.get(mode) for mode in rotor_modes]
        if rotor_modes is not None and len(rotor_modes) == len(treated_rotors) else [None] * len(treated_rotors)
    )
    for rotor_index, rotor in spc.rotors_dict.items():
        if rotor.get('success') is not True:
            continue
        scan = rotor.get('scan')       # 4-atom dihedral defining atoms, 1-indexed
        pivots = rotor.get('pivots')   # 2-atom rotation axis, 1-indexed
        symmetry = rotor.get('symmetry', 1)
        treatment = arkane_treatments[len(torsions)]
        dimension = int(rotor.get('dimensions', 1) or 1)
        candidate_key = f'scan_rotor_{rotor_index}'
        scan_key = candidate_key if candidate_key in scan_keys else None
        torsions.append({
            'symmetry_number': symmetry,
            'treatment': treatment,
            'dimension': dimension,
            'atom_indices': scan,
            'pivot_atoms': pivots,
            'barrier_kj_mol': (_get_rotor_barrier(rotor, project_directory)
                               if dimension == 1 else None),
            'source_scan_key': scan_key,
        })
    return torsions


def _get_rejected_torsions(spc, project_directory: str) -> list[dict]:
    """Build the rejected-torsions list from spc.rotors_dict.

    ``_get_torsions`` above emits only rotors ARC actually treated
    (``success is True``); genuinely rejected rotors — ``success is
    False`` — simply vanished from the result contract. That's a real
    information loss: a species where ARC found 3 rotors and rejected 2
    looked identical, downstream, to one that genuinely has a single rotor.
    This is the complementary half of ``rotors_dict``: every entry
    ``_get_torsions`` skips *because ARC decided against it*, paired with
    *why* (``invalidation_reason``), so a consumer can tell "no rotors"
    from "rotors found and rejected, for these reasons."

    ``success`` is a three-state field (see ``arc/scheduler.py``'s rotor
    state table): ``None`` means pending — not yet started, mid
    troubleshooting, or scanning against a previous/lower conformer — and
    is *not* a rejection; ``False`` means ARC genuinely rejected the rotor
    (invalidated after convergence, or determined not to be a torsion at
    all); ``True`` means it succeeded. Only ``success is False`` belongs
    here. A species can converge with rotors still ``None`` (rotor
    convergence is not gated on species convergence), so treating
    "not True" as "rejected" would misreport ordinary pending rotors as
    reason-less rejections.

    ``invalidation_reason`` is carried verbatim, including the empty string
    it defaults to when ARC never recorded a specific reason — that's the
    honest signal ("rejected, reason unknown") rather than something to
    paper over. It is accumulated with ``+=`` across troubleshooting
    rounds (see ``arc/scheduler.py`` and ``arc/checks/ts.py``), so a
    single string here may concatenate more than one recorded reason.

    Each emitted rejection also carries ``source_log`` — the rotor's scan
    log path, relative to ``project_directory``, when one is found on disk
    at write time (:func:`_resolve_scan_path`), using the same
    relative-path treatment (:func:`_make_rel_path`)
    ``rotor_scans[].source_log`` gets — which is a bare ``os.path.relpath``
    with no containment check, so a log outside ``project_directory``
    yields a ``../..``-prefixed path revealing host layout (matches
    existing ``rotor_scans[].source_log`` behavior; not changed here).
    This deliberately does *not* go through :func:`_build_rotor_scan_entry`
    / ``rotor_scans``: that gate only admits ``success is True`` rotors
    (see its docstring), and a rejected rotor's log may itself be
    truncated or self-inconsistent (e.g. a crashed job's partial scan) —
    recording *where the evidence is* is safe and useful even when
    *parsing* it would not be. ``source_log`` is *omitted* (never ``null``)
    when ``rotors_dict`` records no on-disk scan for this rotor, so a
    consumer never receives a dangling reference.

    IMPORTANT: the *absence* of ``source_log`` is NOT a reliable signal
    that this rotor was "never a torsion." ``_resolve_scan_path`` requires
    the file to actually exist on disk at write time, and there are
    several reachable paths to ``success is False`` with no surviving log
    even though a scan genuinely ran:

    - A restart-restored rejected rotor's relative ``scan_path`` is only
      repaired for rotors where ``success`` is truthy (see
      ``arc/main.py``), so a rejected rotor's stale relative path
      routinely fails ``_resolve_scan_path``'s ``os.path.isfile`` check
      even though the rotor was genuinely scanned.
    - A failed directed scan has its ``scan_path`` deliberately blanked to
      ``''`` alongside a real, non-empty ``invalidation_reason``
      (``arc/scheduler.py``'s ``check_directed_scan``) — the scan ran and
      failed, but the record looks identical to "never scanned."
    - A TS reaction-zone pivot exclusion (``arc/checks/ts.py``) sets
      ``success = False`` for a policy reason, typically before any scan
      was attempted at all.

    ARC's ``rotors_dict`` does not currently distinguish these rejection
    stages from a genuine "not a torsion" determination —
    ``invalidation_reason`` is the only (free-text, imperfect) signal
    available, and ``source_log``'s presence or absence must not be used
    as a substitute discriminator.

    Args:
        spc: The species whose ``rotors_dict`` is being rendered.
        project_directory (str): The run's project directory.
    """
    if not getattr(spc, 'rotors_dict', None):
        return []
    rejected = []
    for rotor_index, rotor in spc.rotors_dict.items():
        if rotor.get('success') is not False:
            continue
        entry = {
            'rotor_index': rotor_index,
            'invalidation_reason': rotor.get('invalidation_reason', ''),
            'atom_indices': rotor.get('scan'),   # 4-atom dihedral defining atoms, 1-indexed
            'pivot_atoms': rotor.get('pivots'),  # 2-atom rotation axis, 1-indexed
            'dimension': int(rotor.get('dimensions', 1) or 1),
        }
        source_log = _make_rel_path(_resolve_scan_path(rotor, project_directory), project_directory)
        if source_log is not None:
            entry['source_log'] = source_log
        rejected.append(entry)
    return rejected


def _resolve_scan_path(rotor: dict, project_directory: str) -> str | None:
    """Return an absolute, on-disk scan log path for ``rotor``, or ``None``.

    Centralizes the rules used by both the barrier helper and the
    scan-calc builder so they agree on which rotors qualify as "has a
    scan log we can use."
    """
    return _abs_existing_path(rotor.get('scan_path', ''), project_directory)


def _get_rotor_barrier(rotor: dict, project_directory: str) -> float | None:
    """
    Return max(V) - min(V) in kJ/mol from the 1D scan output file.

    parse_1d_scan_energies already zeroes the minimum, so max(energies) is the
    barrier height directly.
    """
    scan_path = _resolve_scan_path(rotor, project_directory)
    if scan_path is None:
        return None
    try:
        energies, _ = parse_1d_scan_energies(log_file_path=scan_path)
        if energies is not None and len(energies):
            return float(max(energies))
    except Exception:
        logger.debug(f"Failed to parse 1D rotor scan energies from '{scan_path}'", exc_info=True)
    return None


def _build_rotor_scans(spc, project_directory: str) -> list[dict]:
    """Build tool-neutral parsed rotor-scan records for a species.

    Emits one record per successful 1D rotor whose scan log is available and
    parses cleanly. ARC reports source facts, native indices and explicit
    units; consumers decide how scans map into their calculation models.

    ``relaxed`` is ``None`` when the scan type is unrecognized, and
    ``zero_energy_reference_hartree`` is omitted entirely when the parser
    reports none.

        {
            'key': 'scan_rotor_<i>',
            'source_log': str,
            'result': {
                'dimension': 1,
                'relaxed': bool | None,
                'zero_energy_reference_hartree': float,
                'coordinate': {coordinate_type, atom_indices, index_base,
                               unit, symmetry_number, ...},
                'samples': [{source_index, angle_degrees,
                             electronic_energy_hartree,
                             relative_energy_kj_mol, geometry_xyz}],
            },
        }

    Rotors that don't produce a parseable scan_result are skipped — the
    matching torsion's ``source_scan_key`` will be ``None``
    so consumers don't end up with dangling references.

    Only 1D rotors are emitted here; ND rotors and any future
    multi-dimensional shapes go through a separate path (deferred).
    """
    if not getattr(spc, 'rotors_dict', None):
        return []
    # The species's converged opt geometry is the input the rotor scan
    # job was launched against — we pass it down so the scan-result
    # builder can compute ``start_value`` (the dihedral the user
    # requested as the scan's starting point). ``initial_xyz`` is a
    # fallback for species that haven't (yet) populated ``final_xyz``;
    # better to derive a start dihedral from the pre-opt geometry than
    # to leave the field null when something usable exists.
    input_xyz = getattr(spc, 'final_xyz', None) or getattr(spc, 'initial_xyz', None)
    out: list[dict] = []
    for rotor_index, rotor in spc.rotors_dict.items():
        entry = _build_rotor_scan_entry(
            rotor_index, rotor, project_directory, input_xyz=input_xyz,
        )
        if entry is not None:
            out.append(entry)
    return out


def _build_rotor_scan_entry(rotor_index,
                            rotor: dict,
                            project_directory: str,
                            input_xyz: dict | None = None,
                            ) -> dict | None:
    """Return the ``rotor_scans`` record for a single rotor, or ``None``.

    ``None`` is returned for an unsuccessful rotor, a rotor of more than one
    dimension, and a rotor whose scan log does not yield a scan result. This is
    the only place that decision is made; both :func:`_build_rotor_scans` and
    :func:`_get_torsions` reach it through :func:`_build_rotor_scans`.
    """
    if rotor.get('success') is not True:
        return None
    if rotor.get('dimensions', 1) != 1:
        return None
    scan_result = _build_scan_result_for_rotor(
        rotor, project_directory, input_xyz=input_xyz,
    )
    if scan_result is None:
        return None
    scan_path = _resolve_scan_path(rotor, project_directory)
    ess_software, ess_version = _observe_log_ess(scan_path, project_directory)
    entry = {
        'key': f'scan_rotor_{rotor_index}',
        'source_log': _make_rel_path(scan_path, project_directory),
        'ess_software': ess_software,
        'ess_version': ess_version,
        'result': scan_result,
    }
    scan_constraints = _parse_scan_constraints(rotor, project_directory)
    if scan_constraints:
        entry['constraints'] = scan_constraints
    return entry


def _parse_scan_constraints(rotor: dict, project_directory: str) -> list[dict]:
    """Best-effort extraction of held-fixed constraints for a rotor scan.

    Dispatches to the per-software constraint parser based on the
    rotor's ``scan_software`` hint. The scheduler stamps this field with
    ``job.job_adapter`` (e.g. ``'gaussian'``, ``'orca'``) every time it
    records a scan job's output path — in ``Scheduler.end_job`` when a
    scan/directed-scan job finishes, and in ``Scheduler.check_scan_job``
    when the scan is post-processed for symmetry/energies — so a
    completed scan's rotor dict always carries the software that
    actually produced the log. Returns the held-fixed constraints — the
    active scan coordinate is excluded; it lives in
    ``scan_result.coordinates[]``.

    Software dispatch:
        ``gaussian``      → :func:`parse_gaussian_constraints`
        ``orca``          → :func:`parse_orca_constraints`
        missing / empty   → fall back to Gaussian (the only software ARC
                            currently emits ModRedundant for; this keeps
                            restart files / pre-existing rotor dicts from
                            losing their constraints)
        anything else     → ``[]`` with a debug log (no noisy warning;
                            the consumer only cares about the absence)

    Never raises: parser failures degrade to ``[]`` so a malformed log
    doesn't break the other rotor-scan records in the run.
    """
    scan_path = _resolve_scan_path(rotor, project_directory)
    if scan_path is None:
        return []
    software = (rotor.get('scan_software') or '').strip().lower()
    parser_fn = _scan_constraint_parser_for(software)
    if parser_fn is None:
        logger.debug("Scan-constraint extraction: no parser for software=%r "
                     "at '%s'; emitting [].", software, scan_path)
        return []
    try:
        return parser_fn(scan_path)
    except Exception as exc:
        logger.warning("Scan-constraint extraction failed for '%s' "
                       "(software=%r): %s", scan_path, software, exc)
        return []


def _scan_constraint_parser_for(software: str):
    """Return the constraint-parser callable for ``software``, or None.

    Empty / missing ``software`` falls back to the Gaussian parser to
    preserve the prior best-effort behavior for rotor dicts that predate
    the ``scan_software`` field (older restart files, in-progress
    refactors). Anything explicitly unknown (e.g., ``'qchem'``) returns
    None so the caller can debug-log and emit an empty list without
    parsing surprise files.
    """
    if not software or software == 'gaussian':
        return parse_gaussian_constraints
    if software == 'orca':
        return parse_orca_constraints
    return None


def _safe_dihedral_for_scan_atoms(
    xyz: dict | None,
    scan_atoms: list[int],
    scan_path: str,
) -> float | None:
    """Compute the dihedral (degrees, 0-360) at the scan quartet, or ``None``.

    Wraps :func:`arc.species.vectors.calculate_dihedral_angle` with the
    failure-tolerant contract this helper needs: missing input → quiet
    ``None`` (caller decides what "unknown" means); raised exception or
    NaN return (colinear atoms) → ``None`` plus a warning that names
    the scan log so the operator can investigate. Never raises.

    ``scan_atoms`` must be 1-based per ARC convention; that's enforced
    upstream where ``rotor['scan']`` is validated.
    """
    if xyz is None:
        return None
    try:
        angle = calculate_dihedral_angle(coords=xyz, torsion=scan_atoms, index=1)
    except Exception as exc:
        logger.warning(
            "Scan dihedral calculation failed for atoms=%r in '%s': %s.",
            scan_atoms, scan_path, exc,
        )
        return None
    if math.isnan(angle):  # colinear atoms in the input geometry.
        logger.warning(
            "Scan dihedral is NaN (colinear quartet?) for atoms=%r in '%s'.",
            scan_atoms, scan_path,
        )
        return None
    return float(angle)


def _unwrap_angle_sequence(angles: list[float]) -> list[float]:
    """Make a sequence of angles continuous by folding each step into (-180, 180].

    The first value is kept as measured; every later value is the running sum of the
    folded steps. A sweep that passes a full turn therefore keeps climbing past 360
    instead of folding back towards 0, and a sweep that runs towards decreasing values
    goes below its start. Every value equals the corresponding input angle modulo 360.
    """
    unwrapped = [float(angles[0])]
    for previous, current in zip(angles, angles[1:]):
        step = (float(current) - float(previous) + 180.0) % 360.0 - 180.0
        unwrapped.append(unwrapped[-1] + step)
    return unwrapped


def _absolute_scan_dihedrals(geometries: list | None,
                             scan_atoms: list[int],
                             scan_path: str,
                             expected_count: int,
                             ) -> list[float] | None:
    """Measure one absolute dihedral per scan point, or return ``None``.

    Each value is the dihedral of ``scan_atoms`` measured on that point's *own*
    geometry, so it is the value of the internal coordinate at that point rather than a
    displacement from the scan's first sample, and it agrees by construction with what a
    consumer recomputes from the geometry published alongside it. The sequence is
    unwrapped, so a full rotation ends a turn away from where it started and a sweep
    towards decreasing values goes below its start; values are never folded back into a
    0-360 window.

    Returns ``None`` when the per-point geometries are absent, do not align 1:1 with the
    scan points, or contain a quartet whose dihedral cannot be measured. There is no
    absolute coordinate to publish in those cases and a displacement is not a substitute,
    so the caller drops the scan rather than publishing an axis of a different kind.
    """
    if not isinstance(geometries, list) or len(geometries) != expected_count:
        logger.warning(
            "Scan-point geometries are unavailable or do not align with the %d scan "
            "points of '%s'; omitting scan_result, as the samples' absolute dihedral "
            "cannot be measured.",
            expected_count, scan_path,
        )
        return None
    measured: list[float] = []
    for geometry in geometries:
        angle = _safe_dihedral_for_scan_atoms(geometry, scan_atoms, scan_path)
        if angle is None:
            logger.warning(
                "Could not measure the dihedral of atoms=%r at every point of '%s'; "
                "omitting scan_result.",
                scan_atoms, scan_path,
            )
            return None
        measured.append(angle)
    return _unwrap_angle_sequence(measured)


def _xyz_dict_to_output_text(xyz_dict: dict | None) -> str | None:
    """Serialize an ARC xyz dict using ARC's ordinary atom-line format."""
    if xyz_dict is None:
        return None
    bare = xyz_to_str(xyz_dict=xyz_dict)
    if not bare:
        return None
    return bare.strip() or None


def _scan_is_relaxed(rotor: dict) -> bool | None:
    """Return whether the rotor's scan geometries were relaxed at each point.

    An ESS-native torsion scan (``directed_scan_type`` absent or ``'ess'``)
    optimizes all remaining degrees of freedom at every step, and so does every
    ``*_opt`` directed scan. ARC also supports rigid scans — the
    ``brute_force_sp`` family evaluates single-point energies on unrelaxed
    geometries — for which claiming relaxation would be a false statement about
    the data. An unrecognized scan type yields ``None`` rather than a guess.

    Args:
        rotor (dict): The species' ``rotors_dict`` entry.

    Returns:
        bool | None: ``True`` for a relaxed scan, ``False`` for a rigid one,
                     ``None`` when the scan type is not recognized.
    """
    scan_type = rotor.get('directed_scan_type')
    if scan_type is None or not str(scan_type).strip():
        return True
    scan_type = str(scan_type).strip().lower()
    if scan_type == 'ess' or '_opt' in scan_type:
        return True
    if '_sp' in scan_type:
        return False
    return None


def _build_scan_result_for_rotor(
    rotor: dict,
    project_directory: str,
    *,
    input_xyz: dict | None = None,
) -> dict | None:
    """Parse a rotor scan into a tool-neutral scientific result.

    Returns ``None`` when:
      * no scan log on disk (covered by ``_resolve_scan_path``)
      * parser fails to extract angles or relative energies (the only
        two fields that drive the shape — absolute Hartree energies are
        nice-to-have)
      * atom indices are not a valid ARC-native 1-based dihedral quartet
      * the per-point geometries do not yield an absolute dihedral for
        every sample, since each sample's ``angle_degrees`` is measured
        on that sample's own geometry

    Per-sample geometries use ARC's ordinary atom-line XYZ text. On length
    mismatch or serialization failure they are dropped uniformly; partial
    coverage would imply an alignment ARC cannot verify.
    """
    scan_path = _resolve_scan_path(rotor, project_directory)
    if scan_path is None:
        return None

    parsed = parse_1d_scan_full_result(log_file_path=scan_path)

    angles = parsed.get('angles_deg')
    rel_energies = parsed.get('relative_energies_kj_mol')
    if not angles or not rel_energies:
        return None
    if len(angles) != len(rel_energies):
        logger.debug(
            "Skipping scan_result for '%s': angles/energies length mismatch (%d vs %d)",
            scan_path, len(angles), len(rel_energies),
        )
        return None

    abs_energies = parsed.get('absolute_energies_hartree')
    # absolute_energies and angles must align with relative_energies; if a
    # parser disagreed on the count, drop absolute rather than misalign.
    if abs_energies is not None and len(abs_energies) != len(rel_energies):
        abs_energies = None

    # Per-point geometries: the parser wrapper returns one xyz dict per
    # converged scan iteration. Attach to each point as
    # geometry text only when the list aligns 1:1 with the energy list;
    # otherwise drop wholesale —
    # mixing populated and missing entries would imply an alignment we
    # can't verify, and the schema accepts ``geometry`` as omitted but
    # not as ``null``.
    geometries = parsed.get('geometries')
    point_geometry_xyz_texts: list[str] | None = None
    point_geometry_isotopes: list[list[int] | None] = []
    if geometries is not None:
        if len(geometries) != len(rel_energies):
            logger.warning(
                "Scan-point geometry count (%d) does not match scan-point count (%d) "
                "for '%s'; omitting per-point geometries from scan_result.",
                len(geometries), len(rel_energies), scan_path,
            )
        else:
            try:
                point_geometry_xyz_texts = [
                    _xyz_dict_to_output_text(g) for g in geometries
                ]
                stated_isotopes: dict = dict()
                for g in geometries:
                    if isinstance(g, dict) and isinstance(g.get('symbols'), (list, tuple)):
                        symbols = tuple(g['symbols'])
                        if symbols not in stated_isotopes:
                            stated_isotopes[symbols] = _get_log_stated_isotopes(scan_path, symbols)
                point_geometry_isotopes = [
                    stated_isotopes.get(tuple(g['symbols'])) if isinstance(g, dict)
                    and isinstance(g.get('symbols'), (list, tuple)) else None
                    for g in geometries
                ]
            except Exception as exc:
                logger.warning(
                    "Scan-point geometry serialization failed for '%s': %s; "
                    "omitting per-point geometries from scan_result.",
                    scan_path, exc,
                )
                point_geometry_xyz_texts = None
            else:
                # If any single point produced an empty/None text, drop
                # all rather than emit asymmetric coverage.
                if any(t is None or not t for t in point_geometry_xyz_texts):
                    logger.warning(
                        "Scan-point geometry serialization yielded empty text for "
                        "at least one point in '%s'; omitting per-point geometries.",
                        scan_path,
                    )
                    point_geometry_xyz_texts = None

    scan_atoms = rotor.get('scan')
    if not (isinstance(scan_atoms, list) and len(scan_atoms) == 4
            and all(isinstance(a, int) and a >= 1 for a in scan_atoms)
            and len(set(scan_atoms)) == 4):
        return None

    point_dihedrals = _absolute_scan_dihedrals(
        geometries, scan_atoms, scan_path, len(rel_energies),
    )
    if point_dihedrals is None:
        return None

    symmetry = rotor.get('symmetry')
    coord: dict[str, Any] = {
        'coordinate_type': 'dihedral',
        'atom_indices': list(scan_atoms),
        'index_base': 1,
        'unit': 'degree',
        'sample_count': len(angles),
    }
    if isinstance(symmetry, int) and symmetry >= 1:
        coord['symmetry_number'] = symmetry

    # Requested grid metadata: ``parse_scan_args`` reads the
    # ModRedundant header that Gaussian echoes back into its log
    # (``D a b c d S <step_count> <step_size>``), giving us the exact
    # step size the user requested rather than one inferred from the
    # completed point spacing. ORCA / other ESS raise
    # ``NotImplementedError`` from the same parser; in those cases
    # the grid fields stay absent rather than guessed at.
    requested_step_size: float | None = None
    requested_steps: int | None = None
    try:
        scan_args = parse_scan_args(scan_path)
        raw_step_size = scan_args.get('step_size')
        if isinstance(raw_step_size, (int, float)) and raw_step_size != 0:
            requested_step_size = float(raw_step_size)
        raw_steps = scan_args.get('step')
        if isinstance(raw_steps, int) and raw_steps > 0:
            requested_steps = raw_steps
    except NotImplementedError:
        logger.debug(f"parse_scan_args does not support the log format of '{scan_path}'; "
                     f"leaving the requested-grid fields absent")
    except Exception:
        logger.debug(f"parse_scan_args failed for '{scan_path}'", exc_info=True)
    if requested_step_size is not None:
        coord['requested_step_size'] = requested_step_size

    # ``requested_start``/``requested_end`` describe the requested grid, not
    # the completed-point spacing. Computing them honestly requires
    # both the requested step size (above) AND the input dihedral
    # (the geometry the user pointed the scan at). The latter comes
    # from the species record — preferred — falling back to the
    # first parsed scan-iteration geometry, which for Gaussian
    # ModRedundant scans has the dihedral held fixed at the input
    # value by construction (so it's not "inferring from outputs",
    # it's reading a frozen DOF). ``end_value`` is then exact:
    # ``start + step_size * steps``. We deliberately do
    # NOT wrap into [-180, 180]: a full rotation must land at
    # ``start + 360°``, not back at ``start``; continuity is the scientific
    # fact downstream consumers need.
    #
    # ``steps`` is the count the user *requested*, read from the same
    # ModRedundant header as the step size. Deriving it from the number of
    # completed points instead would make a Gaussian-truncated scan report its
    # truncated span as the request — contradicting this field's whole purpose.
    # Completed points remain available as ``sample_count`` and ``samples``.
    requested_span_steps = (
        requested_steps if requested_steps is not None else len(rel_energies) - 1
    )
    if (
        requested_step_size is not None
        and requested_span_steps >= 0
    ):
        dihedral_source = input_xyz
        if dihedral_source is None:
            geom_list = parsed.get('geometries')
            if isinstance(geom_list, list) and geom_list:
                dihedral_source = geom_list[0]
        start_value = _safe_dihedral_for_scan_atoms(
            dihedral_source, scan_atoms, scan_path,
        )
        if start_value is not None:
            coord['requested_start'] = start_value
            coord['requested_end'] = (
                start_value + requested_step_size * requested_span_steps
            )

    samples: list[dict[str, Any]] = []
    for source_index, (angle, rel_e) in enumerate(zip(point_dihedrals, rel_energies)):
        sample: dict[str, Any] = {
            'source_index': source_index,
            'angle_degrees': float(angle),
            'relative_energy_kj_mol': float(rel_e),
        }
        if abs_energies is not None:
            sample['electronic_energy_hartree'] = float(abs_energies[source_index])
        if point_geometry_xyz_texts is not None:
            sample['geometry_xyz'] = point_geometry_xyz_texts[source_index]
            sample['geometry_isotopes'] = point_geometry_isotopes[source_index]
        samples.append(sample)

    scan_result: dict[str, Any] = {
        'dimension': 1,
        'relaxed': _scan_is_relaxed(rotor),
        'coordinate': coord,
        'samples': samples,
    }
    zero_ref = parsed.get('zero_energy_reference_hartree')
    if isinstance(zero_ref, (int, float)):
        scan_result['zero_energy_reference_hartree'] = float(zero_ref)

    return scan_result


def _ts_checks_to_dict(spc) -> dict:
    """Return the transition state's validation verdicts.

    ``ARCSpecies.ts_checks`` is where ARC records whether the transition state
    passed each validation it ran: the energy checks (``E0``, ``e_elect``), the
    IRC endpoint check, the single-imaginary-frequency check, and the normal
    mode displacement check. A verdict is ``True`` (passed), ``False`` (failed)
    or ``None`` (not run, or run without a conclusion) — ``None`` is not a
    failure. ``warnings`` accumulates free text emitted by those checks.

    These are the transition state's provenance: ``irc_converged`` next to them
    reports that the IRC *jobs completed*, not that the IRC validated anything,
    and only ``ts_checks['IRC']`` carries that verdict.

    Args:
        spc (ARCSpecies): The transition state species.

    Returns: dict
        ``{'E0', 'e_elect', 'IRC', 'freq', 'NMD'}`` mapped to ``bool | None``,
        plus ``'warnings'`` as a string.
    """
    checks = getattr(spc, 'ts_checks', None) or dict()
    result = {key: (checks.get(key) if isinstance(checks.get(key), bool) else None)
              for key in ('E0', 'e_elect', 'IRC', 'freq', 'NMD')}
    warnings = checks.get('warnings')
    result['warnings'] = warnings if isinstance(warnings, str) else ''
    return result


def _is_int(value) -> bool:
    """Whether ``value`` is an ``int`` and not a ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _irc_participant_mapping_to_dict(spc) -> dict | None:
    """
    Return which atoms of each IRC endpoint geometry belong to which participant species, or ``None``.

    The mapping is recorded only when the IRC check established its verdict by graph isomorphism of the perceived
    endpoint fragments, so it is ``None`` when the bond-list fallback decided, when IRC was not run or failed, and
    unless ``ts_checks['IRC']`` is ``True``. A recorded value that does not have the documented shape, or that
    violates the schema (positions and occurrences below 1, negative or repeated atom indices), is dropped rather
    than exported.

    Args:
        spc (ARCSpecies): The transition state species.

    Returns: dict | None
        ``{'reactants': side, 'products': side, 'sides_distinguishable': bool, 'atom_order_matches_ts': bool | None}``
        where ``side`` is ``{'endpoint_label': str | None, 'participants': [participant, ...]}`` and
        ``participant`` is ``{'label': str, 'position': int, 'occurrence': int, 'atom_indices': list[int]}``.
        ``endpoint_label`` is the label of the species the optimized endpoint geometry belongs to. ``position`` is 1-based in the order of
        ``atom_map_reactant_labels`` / ``atom_map_product_labels``. ``atom_indices`` are ascending 0-based
        indices into that optimized endpoint geometry. ``sides_distinguishable`` is ``False`` when the reactants and
        the products are graph-isomorphic to each other, and ``atom_order_matches_ts`` states whether the endpoint
        geometries follow the atom order of the TS geometry the IRC was run from (``None`` when not checked).
    """
    if (getattr(spc, 'ts_checks', None) or {}).get('IRC') is not True:
        return None
    mapping = getattr(spc, 'irc_participant_mapping', None)
    if not isinstance(mapping, dict) \
            or set(mapping) != {'reactants', 'products', 'sides_distinguishable', 'atom_order_matches_ts'} \
            or not isinstance(mapping['sides_distinguishable'], bool) \
            or not (mapping['atom_order_matches_ts'] is None or isinstance(mapping['atom_order_matches_ts'], bool)):
        return None
    result = {'sides_distinguishable': mapping['sides_distinguishable'],
              'atom_order_matches_ts': mapping['atom_order_matches_ts']}
    for well in ('reactants', 'products'):
        side = mapping[well]
        if not isinstance(side, dict) or not isinstance(side.get('participants'), list):
            return None
        label = side.get('endpoint_label')
        if label is not None and not isinstance(label, str):
            return None
        participants = list()
        for participant in side['participants']:
            if not isinstance(participant, dict) or not isinstance(participant.get('label'), str) \
                    or not all(_is_int(participant.get(key)) and participant[key] >= 1
                               for key in ('position', 'occurrence')) \
                    or not isinstance(participant.get('atom_indices'), list) \
                    or not all(_is_int(i) and i >= 0 for i in participant['atom_indices']) \
                    or len(set(participant['atom_indices'])) != len(participant['atom_indices']):
                return None
            participants.append({'label': participant['label'],
                                 'position': participant['position'],
                                 'occurrence': participant['occurrence'],
                                 'atom_indices': list(participant['atom_indices']),
                                 })
        result[well] = {'endpoint_label': label, 'participants': participants}
    return result


def _get_reaction_atom_map(rxn) -> dict:
    """
    Read the atom map a reaction already holds, without computing one, and state what it maps.

    The map is exported only if it is a permutation of the atoms of the expanded reactants that conserves the
    element: reactant atom ``i`` and product atom ``map[i]`` must have the same symbol. Reactant atoms are counted
    over ``get_reactants_and_products`` (``r_species`` order, one block per occurrence, each block in the exported
    geometry atom order of the species), and product atoms likewise over the products. The map counts atoms in the atom
    order of each species' ``mol``, so it is exported only if the exported geometry of every participant has that order
    (see ``_geometry_follows_mol_atom_order``); otherwise it is ``None`` and the reason is logged.

    Args:
        rxn (ARCReaction): The reaction.

    Returns: dict
        ``atom_map``, ``atom_map_reactant_labels`` and ``atom_map_product_labels`` (one label per occurrence, in map
        order), ``atom_map_source`` (``'declared'`` or ``'inferred'``) and ``atom_map_method``; every value is ``None``
        when the reaction holds no (well-formed, element-conserving) map, and the source and method are ``None`` when
        the origin of the map was not recorded.
    """
    nothing = {key: None for key in ('atom_map', 'atom_map_reactant_labels', 'atom_map_product_labels',
                                     'atom_map_source', 'atom_map_method')}
    atom_map = getattr(rxn, '_atom_map', None)
    if not isinstance(atom_map, (list, tuple)) or not all(_is_int(i) for i in atom_map):
        return nothing
    reactants, products = rxn.get_reactants_and_products(return_copies=False)
    xyzs = [_get_exported_xyz(spc) for spc in reactants + products]
    if any(not isinstance(xyz, dict) for xyz in xyzs):
        return nothing
    if _has_participant_off_mol_atom_order(reactants + products, xyzs):
        logger.warning(f'The atom map of {getattr(rxn, "label", None)} counts atoms in the atom order of the '
                       f'species mol, which the exported geometry of at least one participant does not follow; '
                       f'the atom map is not exported.')
        return nothing
    r_symbols = [symbol for xyz in xyzs[:len(reactants)] for symbol in xyz['symbols']]
    p_symbols = [symbol for xyz in xyzs[len(reactants):] for symbol in xyz['symbols']]
    if len(atom_map) != len(r_symbols) or len(r_symbols) != len(p_symbols) \
            or sorted(atom_map) != list(range(len(atom_map))) \
            or any(r_symbols[i] != p_symbols[j] for i, j in enumerate(atom_map)):
        return nothing
    source = getattr(rxn, '_atom_map_source', None)
    source = source if source in ATOM_MAP_SOURCES else None
    method = getattr(rxn, '_atom_map_method', None)
    return {'atom_map': list(atom_map),
            'atom_map_reactant_labels': [spc.label for spc in reactants],
            'atom_map_product_labels': [spc.label for spc in products],
            'atom_map_source': source,
            'atom_map_method': method if source == 'inferred' and isinstance(method, str) else None,
            }


def _get_ts_atom_map(rxn, atom_map_fields: dict) -> dict:
    """
    Read the TS atom map the IRC check recorded on the reaction's TS, and state whether it can be exported.

    The map is exported only if every one of these holds: the reaction has a TS with a geometry, its IRC check passed,
    the check recorded a map (not a reason it could not), the reaction exports an ``atom_map``, the recorded map is
    well formed (lengths, a bijection onto the TS atoms, element conservation against the TS geometry, and
    ``products[atom_map[i]] == reactants[i]``), and the geometry of every reactant and product has the atom order of
    its ``mol``, which ``atom_map`` and the bonds the map was built from count in. Anything malformed is dropped
    and logged.

    Args:
        rxn (ARCReaction): The reaction.
        atom_map_fields (dict): The result of ``_get_reaction_atom_map`` for the reaction.

    Returns: dict
        ``ts_atom_map`` (``{'ts_label', 'reactants', 'products', 'method', 'reactant_endpoint',
        'ts_atom_order_follows_reactants'}`` or
        ``None``) and ``ts_atom_map_unavailable_reason`` (one of ``TS_ATOM_MAP_UNAVAILABLE_REASONS``, ``None`` exactly
        when the map is exported).
    """
    def _unavailable(reason: str) -> dict:
        return {'ts_atom_map': None, 'ts_atom_map_unavailable_reason': reason}

    ts = getattr(rxn, 'ts_species', None)
    ts_xyz = _get_exported_xyz(ts) if ts is not None and hasattr(ts, 'final_xyz') else None
    if not isinstance(getattr(ts, 'label', None), str) or not isinstance(ts_xyz, dict):
        return _unavailable('no_ts')
    ts_checks = getattr(ts, 'ts_checks', None)
    if not isinstance(ts_checks, dict) or ts_checks.get('IRC') is not True:
        return _unavailable('irc_not_passed')
    recorded = getattr(ts, 'ts_atom_map', None)
    reason = getattr(ts, 'ts_atom_map_unavailable_reason', None)
    if not isinstance(recorded, dict):
        return _unavailable(reason if reason in TS_ATOM_MAP_UNAVAILABLE_REASONS
                            else 'no_atom_map' if atom_map_fields['atom_map'] is None else 'not_recorded')
    atom_map = atom_map_fields['atom_map']
    reactants, products = rxn.get_reactants_and_products(return_copies=False)
    xyzs = [_get_exported_xyz(spc) for spc in reactants + products]
    if atom_map is None:
        return _unavailable('species_atom_order_mismatch'
                            if isinstance(getattr(rxn, '_atom_map', None), (list, tuple))
                            and _has_participant_off_mol_atom_order(reactants + products, xyzs) else 'no_atom_map')
    if any(not isinstance(xyz, dict) for xyz in xyzs):
        return _unavailable('not_recorded')
    r_symbols = [symbol for xyz in xyzs[:len(reactants)] for symbol in xyz['symbols']]
    p_symbols = [symbol for xyz in xyzs[len(reactants):] for symbol in xyz['symbols']]
    if not _is_well_formed_ts_atom_map(recorded, ts.label, atom_map, r_symbols, p_symbols, list(ts_xyz['symbols'])):
        logger.warning(f'The TS atom map of {getattr(rxn, "label", None)} is malformed and is not exported.')
        return _unavailable('not_recorded')
    return {'ts_atom_map': {'ts_label': recorded['ts_label'],
                            'reactants': list(recorded['reactants']),
                            'products': list(recorded['products']),
                            'method': recorded['method'],
                            'reactant_endpoint': recorded['reactant_endpoint'],
                            'ts_atom_order_follows_reactants': recorded['ts_atom_order_follows_reactants'],
                            },
            'ts_atom_map_unavailable_reason': None,
            }


def _is_well_formed_ts_atom_map(recorded: dict,
                                ts_label: str,
                                atom_map: list[int],
                                r_symbols: list[str],
                                p_symbols: list[str],
                                ts_symbols: list[str],
                                ) -> bool:
    """
    Whether a recorded TS atom map has the documented shape and agrees with the reaction and the TS geometry.

    Args:
        recorded (dict): The recorded map.
        ts_label (str): The label of the TS.
        atom_map (list[int]): The reaction's atom map.
        r_symbols (list[str]): The element of every concatenated reactant atom.
        p_symbols (list[str]): The element of every concatenated product atom.
        ts_symbols (list[str]): The element of every TS atom.

    Returns:
        bool: ``True`` if the map is well formed.
    """
    n_atoms = len(atom_map)
    if set(recorded) != {'ts_label', 'reactants', 'products', 'method', 'reactant_endpoint',
                         'ts_atom_order_follows_reactants'} \
            or recorded['ts_label'] != ts_label or recorded['method'] != TS_ATOM_MAP_METHOD \
            or not _is_int(recorded['reactant_endpoint']) or recorded['reactant_endpoint'] not in (1, 2) \
            or not isinstance(recorded['ts_atom_order_follows_reactants'], bool):
        return False
    reactants, products = recorded['reactants'], recorded['products']
    if not all(isinstance(side, list) and len(side) == n_atoms and all(_is_int(i) for i in side)
               for side in (reactants, products)):
        return False
    if len(ts_symbols) != n_atoms or len(r_symbols) != n_atoms or len(p_symbols) != n_atoms:
        return False
    if sorted(reactants) != list(range(n_atoms)) or sorted(products) != list(range(n_atoms)):
        return False
    if any(r_symbols[i] != ts_symbols[reactants[i]] or p_symbols[atom_map[i]] != ts_symbols[products[atom_map[i]]]
           for i in range(n_atoms)):
        return False
    if any(products[atom_map[i]] != reactants[i] for i in range(n_atoms)):
        return False
    return recorded['ts_atom_order_follows_reactants'] == (reactants == list(range(n_atoms)))


def _geometry_follows_mol_atom_order(spc, xyz: dict) -> bool:
    """
    Whether the geometry of a species has the atom order of its ``mol``: the same number of atoms, the same
    element in every position, and every bond of the ``mol`` no longer than 1.2 times the single-bond length of its two
    elements in the geometry (``are_coords_compliant_with_graph``; bond orders are not considered).

    Args:
        spc: The species.
        xyz (dict): The geometry that is exported for the species.

    Returns:
        bool: ``True`` if the two orders agree.
    """
    mol = getattr(spc, 'mol', None)
    if mol is None or len(mol.atoms) != len(xyz['symbols']):
        return False
    return are_coords_compliant_with_graph(xyz=xyz, mol=mol)


def _has_participant_off_mol_atom_order(species: list, xyzs: list) -> bool:
    """
    Whether every species has an exported geometry and at least one geometry does not have the atom order of its
    species' ``mol``.

    Args:
        species (list): The reaction participants.
        xyzs (list): The exported geometry of each participant, ``None`` where it has none.

    Returns:
        bool: ``True`` if all geometries exist and any one does not follow its ``mol``.
    """
    if any(not isinstance(xyz, dict) for xyz in xyzs):
        return False
    return any(not _geometry_follows_mol_atom_order(spc, xyz) for spc, xyz in zip(species, xyzs))


def _rxn_to_dict(rxn) -> dict:
    """Convert an ARCReaction to a plain dict for output.yml."""
    reactants, products = rxn.get_reactants_and_products(return_copies=False)
    kinetics = rxn.kinetics
    kin_dict: dict | None = None
    if kinetics is not None:
        A = kinetics.get('A')
        Ea = kinetics.get('Ea')
        T0 = kinetics.get('T0')
        Tmin = kinetics.get('Tmin')
        Tmax = kinetics.get('Tmax')
        kin_dict = {
            'A': A[0] if isinstance(A, (tuple, list)) else A,
            'A_units': A[1] if isinstance(A, (tuple, list)) else None,
            'n': kinetics.get('n'),
            'Ea': Ea[0] if isinstance(Ea, (tuple, list)) else Ea,
            'Ea_units': Ea[1] if isinstance(Ea, (tuple, list)) else None,
            'T0_k': T0[0] if isinstance(T0, (tuple, list)) else T0,
            'Tmin_k': Tmin[0] if isinstance(Tmin, (tuple, list)) else Tmin,
            'Tmax_k': Tmax[0] if isinstance(Tmax, (tuple, list)) else Tmax,
            'dA': kinetics.get('dA'),
            'dn': kinetics.get('dn'),
            'dEa': kinetics.get('dEa'),
            'dEa_units': kinetics.get('dEa_units'),
            'n_data_points': kinetics.get('n_data_points'),
            # Only the Arkane run that fitted these A/n/Ea stamps this key
            # (arc/statmech/arkane.py). Kinetics from any other source —
            # user-supplied in the input YAML, restored from a restart file, or
            # from a future non-Arkane StatmechAdapter — carry no tunneling
            # correction, so the field stays null rather than defaulting from
            # the template constant. A provenance field that cannot be false
            # carries no information.
            'tunneling': kinetics.get('tunneling'),
            'atom_corrections_applied': kinetics.get('atom_corrections_applied'),
            'comment': _str_or_none(kinetics.get('comment')),
            'ts_validation': _str_or_none(kinetics.get('ts_validation')),
        }

    rxn_dict: dict = {
        'label': rxn.label,
        'reactant_labels': list(rxn.reactants),
        'product_labels': list(rxn.products),
        'reactant_species_labels': [spc.label for spc in reactants] or None,
        'product_species_labels': [spc.label for spc in products] or None,
        'family': rxn.family,
        'multiplicity': rxn.multiplicity,
        'ts_label': rxn.ts_label,
        'kinetics': kin_dict,
        'reversible': _get_reversible(rxn),
    }
    atom_map_fields = _get_reaction_atom_map(rxn)
    rxn_dict.update(atom_map_fields)
    rxn_dict.update(_get_ts_atom_map(rxn, atom_map_fields))
    long_kin_desc = getattr(rxn, 'long_kinetic_description', None)
    if long_kin_desc:
        rxn_dict['long_kinetic_description'] = long_kin_desc
    return rxn_dict
