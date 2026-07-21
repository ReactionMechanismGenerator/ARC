"""
Module for writing the consolidated output.yml at the end of an ARC run.

output.yml supersedes output/status.yml and <project>.info: it consolidates
all result data into a single file with run-relative paths so downstream
consumers (TCKDB, analysis scripts) need only one file.

Written atomically at the very end of a run. If the run is interrupted, the
file will not exist rather than be partially written.
"""

import datetime
import math
import os
import tempfile
from typing import Any
import uuid

from arc.common import ARC_PATH, VERSION, get_git_commit, get_logger, read_yaml_file, save_yaml_file
from arc.constants import E_h_kJmol
from arc.imports import settings
from arc.job.local import execute_command
from arc.parser.parser import (
    parse_1d_scan_energies,
    parse_1d_scan_full_result,
    parse_e_elect,
    parse_ess_version,
    parse_geometry,
    parse_opt_steps,
    parse_s_squared,
    parse_scan_args,
    parse_zpe_correction,
    s_squared_expected_from_multiplicity,
)
from arc.species.converter import xyz_to_str
from arc.species.vectors import calculate_dihedral_angle
from arc.statmech.arkane import (
    AEC_SECTION_START, AEC_SECTION_END,
    ARKANE_TUNNELING_METHOD,
    MBAC_SECTION_START, MBAC_SECTION_END,
    PBAC_SECTION_START, PBAC_SECTION_END,
    find_best_across_files, get_qm_corrections_files,
)
from arc.tckdb_evidence import (
    EVIDENCE_FILENAME,
    EVIDENCE_SCHEMA_NAME,
    EVIDENCE_SCHEMA_VERSION,
    build_tckdb_evidence,
    write_tckdb_evidence_atomic,
)


logger = get_logger()


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
    composite_method=None,
    freq_scale_factor: float | None = None,
    freq_scale_factor_user_provided: bool = False,
    bac_type: str | None = None,
    arkane_level_of_theory=None,
    irc_requested: bool = True,
    t0: float | None = None,
    completed_job_records: list | None = None,
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
        composite_method (Level, optional): Composite method (e.g., CBS-QB3, G4).
        freq_scale_factor (float, optional): The harmonic frequencies scaling factor used.
        freq_scale_factor_user_provided (bool): Whether the user explicitly set the scale factor.
        bac_type (str, optional): The BAC type ('p', 'm', or None).
        arkane_level_of_theory (Level, optional): The composite LOT Arkane uses for energy corrections.
            Recorded only when it differs from sp_level.
        irc_requested (bool): Whether IRC jobs were requested for this run.
        t0 (float, optional): The epoch timestamp when the ARC run started.
        completed_job_records (list, optional): Lightweight per-job cost records accumulated by the Scheduler
            (see Scheduler._record_completed_job), used for the run-level cost metrics.
    """
    doc: dict[str, Any] = {}

    # ---- header ----------------------------------------------------------------
    doc['schema_version'] = '1.1'
    arc_git, _ = get_git_commit(ARC_PATH)
    doc['project'] = project
    doc['arc_version'] = VERSION
    doc['arc_git_commit'] = arc_git or None
    doc['arkane_git_commit'] = _get_arkane_git_commit()
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
    doc['arkane_level_of_theory'] = _level_to_dict(arkane_level_of_theory)
    doc['freq_scale_factor'] = freq_scale_factor
    doc['freq_scale_factor_source'] = (
        None if freq_scale_factor_user_provided
        else _resolve_freq_scale_factor_source(freq_level)
    )
    doc['bac_type'] = bac_type
    aec, bac = _get_energy_corrections(arkane_level_of_theory, bac_type)
    doc['atom_energy_corrections'] = aec
    doc['bond_additivity_corrections'] = bac

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
        species_dict, arkane_level_of_theory, bac_type, project_directory,
    )
    doc['species'] = []
    doc['transition_states'] = []
    for spc in species_dict.values():
        d = _spc_to_dict(spc, output_dict, project_directory, point_groups,
                         irc_requested=irc_requested, software_by_job=software_by_job)
        d['applied_energy_corrections'] = _build_applied_corrections_for_species(
            spc.label, species_corrections, arkane_level_of_theory, bac_type,
            aec_table=aec, bac_table=bac,
        )
        if spc.is_ts:
            doc['transition_states'].append(d)
        else:
            doc['species'].append(d)

    # ---- reactions --------------------------------------------------------------
    doc['reactions'] = [_rxn_to_dict(rxn) for rxn in reactions]

    # ---- atomic write -----------------------------------------------------------
    out_dir = os.path.join(project_directory, 'output')
    os.makedirs(out_dir, exist_ok=True)
    document_id = uuid.uuid4().hex
    try:
        evidence_doc = build_tckdb_evidence(
            output_doc=doc,
            project_directory=project_directory,
            document_id=document_id,
        )
        evidence_path = write_tckdb_evidence_atomic(
            evidence_doc=evidence_doc,
            output_directory=out_dir,
        )
        doc['tckdb_evidence'] = {
            'path': EVIDENCE_FILENAME,
            'schema_name': EVIDENCE_SCHEMA_NAME,
            'schema_version': EVIDENCE_SCHEMA_VERSION,
            'document_id': document_id,
        }
        counts = {
            kind: {
                status: sum(
                    1 for record in evidence_doc['records']
                    if record.get(kind, {}).get('status') == status
                )
                for status in ('available', 'unavailable')
            }
            for kind in ('freq_hessian', 'irc', 'gsm')
        }
        logger.info('Wrote TCKDB evidence to %s (counts=%s)', evidence_path, counts)
    except Exception as exc:
        logger.warning('Could not build/write optional TCKDB evidence: %s', exc)
    out_path = os.path.join(out_dir, 'output.yml')
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix='.output.yml.tmp')
    try:
        os.close(fd)
        save_yaml_file(path=tmp_path, content=doc)
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.debug(f'Failed to remove temporary output file {tmp_path}', exc_info=True)
        raise
    logger.info(f'Wrote consolidated results to {out_path}')

# ── helpers ──────────────────────────────────────────────────────────────────


def _get_arkane_git_commit() -> str | None:
    """Return the HEAD commit hash of the RMG-Py repo (Arkane lives there), or None."""
    try:
        rmg_path = settings.get('RMG_PATH')
        if not rmg_path:
            return None
        head, _ = get_git_commit(rmg_path)
        return head or None
    except Exception:
        return None


def _level_to_dict(level) -> dict | None:
    """Convert a Level object to a dict with all non-None fields, or None."""
    if level is None:
        return None
    if hasattr(level, 'as_dict'):
        d = level.as_dict()
        d.pop('repr', None)
        d.pop('compatible_ess', None)
        return d
    return {
        'method': getattr(level, 'method', None),
        'basis': getattr(level, 'basis', None),
        'software': getattr(level, 'software', None),
    }


def _resolve_freq_scale_factor_source(freq_level) -> str | None:
    """
    Return the full literature citation for the freq scale factor, or None.

    Looks up the source index from the entry in data/freq_scale_factors.yml
    and resolves it against the top-level ``sources`` dict.
    Returns None when the level is not in the database (scale factor was
    user-supplied or computed).
    """
    if freq_level is None:
        return None
    yml_path = os.path.join(ARC_PATH, 'data', 'freq_scale_factors.yml')
    try:
        data = read_yaml_file(yml_path)
    except (OSError, Exception):
        return None

    sources = data.get('sources', {})
    factors = data.get('freq_scale_factors', {})
    level_key = str(freq_level) if not isinstance(freq_level, str) else freq_level
    entry = factors.get(level_key)
    if not isinstance(entry, dict):
        return None
    source_key = entry.get('source')
    if source_key is None:
        return None
    return sources.get(source_key)


def _make_rel_path(path: str | None, project_directory: str) -> str | None:
    """Convert an absolute path to one relative to project_directory, or None."""
    if not path:
        return None
    try:
        return os.path.relpath(path, project_directory)
    except ValueError:
        return path  # Windows: relpath can fail across drives


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


def _ts_guesses_to_list(spc, project_directory: str) -> list[dict]:
    """
    Build the per-TSGuess provenance list for a TS entry in output.yml.

    One entry per TSGuess of the TS species, recording the guess method and its
    provenance (method_sources after equivalent-guess clustering), success, relative
    energy among the guesses, execution time (seconds), the level of theory the
    guess-generating adapter ran its electronic structure at (``None`` for pure
    ML/template methods), the guess log file (run-relative), and whether this guess
    was the one chosen for the TS optimization.

    Args:
        spc (ARCSpecies): The TS species.
        project_directory (str): Root directory of this ARC project.

    Returns:
        list[dict]: One provenance dict per TSGuess.
    """
    ts_guesses = getattr(spc, 'ts_guesses', None) or []
    chosen_idx = getattr(spc, 'chosen_ts', None)
    chosen_guess = None
    if isinstance(chosen_idx, int) and 0 <= chosen_idx < len(ts_guesses):
        chosen_guess = ts_guesses[chosen_idx]
    entries = []
    for tsg in ts_guesses:
        method_sources = getattr(tsg, 'method_sources', None)
        entries.append({
            'index': tsg.index,
            'method': tsg.method,
            'method_sources': list(method_sources) if method_sources else None,
            'method_index': tsg.method_index,
            'method_direction': tsg.method_direction,
            'success': tsg.success,
            'energy_kj_mol': tsg.energy,  # relative energy among this TS's guesses
            'execution_time_sec': _timedelta_to_seconds(getattr(tsg, 'execution_time', None)),
            'level': getattr(tsg, 'level', None),
            'log_path': _make_rel_path(getattr(tsg, 'log_path', None), project_directory),
            'chosen': tsg is chosen_guess,
        })
    return entries


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


def _parse_spin_diagnostic(sp_path: str | None,
                           freq_path: str | None,
                           opt_path: str | None,
                           multiplicity: int | None,
                           project_directory: str,
                           ) -> dict | None:
    """
    Parse the S**2 spin-contamination diagnostic for a species' single-point calc.

    The diagnostic is a property of the (unrestricted) wavefunction, so it is
    parsed from the sp job's log; when the benchmark reuses the optimization
    output for the sp energy the sp log may be absent, so this falls back to the
    freq log and then the opt/geo log (all run at comparable open-shell
    references). Only emitted for unrestricted/open-shell calcs — restricted /
    closed-shell logs print no ``<S**2>`` and this returns ``None`` (so the
    caller omits the block entirely rather than fabricating an all-null one).

    ``s_squared_expected`` is authoritatively recomputed from ARC's own
    ``multiplicity`` (the source of truth) when available, falling back to the
    value the ESS log reported.

    Returns: dict | None
        ``{'s_squared': float, 's_squared_expected': float | None (omitted if
        None), 's_squared_annihilated': float | None (omitted if None)}`` or
        ``None`` when no ``<S**2>`` could be parsed.
    """
    parsed = None
    for candidate in (sp_path, freq_path, opt_path):
        if not candidate:
            continue
        path = candidate if os.path.isabs(candidate) else os.path.join(project_directory, candidate)
        if not os.path.isfile(path):
            continue
        try:
            parsed = parse_s_squared(path)
        except Exception:
            logger.debug(f'Failed to parse S**2 spin diagnostic from {path!r}', exc_info=True)
            parsed = None
        if parsed is not None and parsed.get('s_squared') is not None:
            break
        parsed = None
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
    return result


def _parse_opt_log(geo_path: str | None, project_directory: str) -> tuple:
    """
    Parse n_steps, final electronic energy, and final geometry from an opt log.

    Returns a 3-tuple ``(n_steps, final_energy_hartree, final_xyz_str)``;
    any element may be ``None`` if that piece couldn't be extracted (the
    others are still attempted independently). ``final_xyz_str`` is in
    the same atom-only format as ``xyz_to_str`` produces — symbol +
    coords, one atom per line, no count header.

    The geometry is parsed via :func:`parse_geometry`, which dispatches
    to per-ESS adapters; logs from any supported ESS work without
    branching here. Used for both fine and coarse opt logs — the coarse
    geometry surfaces as ``coarse_opt_output_xyz`` in output.yml so the
    TCKDB bundle can chain ``opt_coarse → opt`` with the right geometry
    on each side.
    """
    if not geo_path:
        return None, None, None
    if not os.path.isabs(geo_path):
        geo_path = os.path.join(project_directory, geo_path)
    if not os.path.isfile(geo_path):
        return None, None, None

    # Each parse is best-effort and independent — if one fails we still
    # surface the others. The final-energy parse historically failed-fast
    # for the whole tuple, but that's not the right shape now that the
    # geometry parse is also in the mix.
    n_steps = None
    e_hartree = None
    final_xyz = None
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
    except Exception:
        logger.debug("Could not parse final geometry from %s", geo_path, exc_info=True)
    return n_steps, e_hartree, final_xyz


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

    candidates: list[str] = []
    if input_rel_path:
        abs_input = os.path.join(project_directory, input_rel_path) \
            if not os.path.isabs(input_rel_path) else input_rel_path
        if os.path.isfile(abs_input):
            candidates.append(abs_input)
    if log_path and os.path.isfile(log_path):
        candidates.append(log_path)

    if not candidates:
        return []

    try:
        if sw == 'gaussian':
            from arc.parser.adapters.gaussian import parse_gaussian_constraints
            for path in candidates:
                parsed = parse_gaussian_constraints(path)
                if parsed:
                    return parsed
            return []
        if sw == 'orca':
            from arc.parser.adapters.orca import parse_orca_constraints
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


def _get_ess_versions(paths: dict, project_directory: str) -> dict[str, str] | None:
    """
    Parse ESS version strings from each available log file (sp, opt, freq, neb).

    Returns a dict like ``{'sp': 'ORCA 5.0.4', 'opt': 'Gaussian 16, Revision C.01'}``,
    keyed by job type.  Caches parsed versions to avoid re-parsing shared log files.
    Returns ``None`` if nothing could be parsed.
    """
    key_map = {'sp': 'sp', 'geo': 'opt', 'freq': 'freq', 'neb': 'neb'}
    versions: dict[str, str] = {}
    parsed_cache: dict[str, str] = {}
    for path_key, label in key_map.items():
        log_path = paths.get(path_key) or None
        if not log_path:
            continue
        if not os.path.isabs(log_path):
            log_path = os.path.join(project_directory, log_path)
        if not os.path.isfile(log_path):
            continue
        if log_path in parsed_cache:
            versions[label] = parsed_cache[log_path]
            continue
        try:
            version = parse_ess_version(log_path)
            if version:
                versions[label] = version
                parsed_cache[log_path] = version
        except Exception:
            logger.debug(f"Failed to parse ESS version from log file '{log_path}'", exc_info=True)
    return versions or None


def _get_energy_corrections(arkane_level_of_theory, bac_type: str | None) -> tuple:
    """
    Look up the AEC (per-atom, Hartree) and BAC (per-bond, kJ/mol) values
    that Arkane used from the RMG database for the given level of theory.

    Finds the AEC and BAC keys independently via fuzzy matching in their
    respective database sections, then calls ``arc/scripts/get_qm_corrections.py``
    as a subprocess to extract the actual correction dicts.

    Returns:
        (aec_dict_or_None, bac_dict_or_None)
    """
    if arkane_level_of_theory is None:
        return None, None
    try:
        qm_corr_files = get_qm_corrections_files()

        aec_key = find_best_across_files(arkane_level_of_theory, qm_corr_files,
                                          AEC_SECTION_START, AEC_SECTION_END)
        if aec_key is None:
            return None, None

        bac_key = None
        if bac_type in ('p', 'm'):
            if bac_type == 'm':
                bac_start, bac_end = MBAC_SECTION_START, MBAC_SECTION_END
            else:
                bac_start, bac_end = PBAC_SECTION_START, PBAC_SECTION_END
            bac_key = find_best_across_files(arkane_level_of_theory, qm_corr_files, bac_start, bac_end)

        script_path = os.path.join(ARC_PATH, 'arc', 'scripts', 'get_qm_corrections.py')
        rmg_env = settings.get('RMG_ENV_NAME', 'rmg_env')

        fd_in, tmp_in = tempfile.mkstemp(suffix='.qm_input.yml')
        fd_out, tmp_out = tempfile.mkstemp(suffix='.qm_output.yml')
        try:
            os.close(fd_in)
            os.close(fd_out)
            save_yaml_file(path=tmp_in, content={
                'aec_key': aec_key,
                'bac_key': bac_key,
                'bac_type': bac_type,
            })

            commands = [
                'bash -lc "set -euo pipefail; '
                'if command -v micromamba >/dev/null 2>&1; then '
                f'    micromamba run -n {rmg_env} python {script_path} {tmp_in} {tmp_out}; '
                'elif command -v conda >/dev/null 2>&1; then '
                f'    conda run -n {rmg_env} python {script_path} {tmp_in} {tmp_out}; '
                'elif command -v mamba >/dev/null 2>&1; then '
                f'    mamba run -n {rmg_env} python {script_path} {tmp_in} {tmp_out}; '
                'else '
                '    echo \'micromamba/conda/mamba required\' >&2; exit 1; '
                'fi"',
            ]
            _, stderr = execute_command(command=commands, executable='/bin/bash')
            if stderr:
                logger.warning(f'get_qm_corrections.py stderr: {stderr}')

            result = read_yaml_file(tmp_out) or {}
            return result.get('aec'), result.get('bac')
        finally:
            for p in (tmp_in, tmp_out):
                try:
                    os.unlink(p)
                except OSError:
                    logger.debug(f'Failed to remove temporary file {p!r}', exc_info=True)
    except Exception:
        return None, None


def _safe(fn, default=None):
    """Call fn() and return *default* if any exception is raised."""
    try:
        return fn()
    except Exception:
        logger.debug(f'_safe() caught exception in {fn}', exc_info=True)
        return default


_BAC_KIND_BY_TYPE = {'p': 'bac_petersson', 'm': 'bac_melius'}


def _compute_species_corrections(
    species_dict: dict,
    arkane_level_of_theory,
    bac_type: str | None,
    project_directory: str,
) -> dict[str, dict]:
    """Compute per-species AEC/BAC totals + components by delegating to
    Arkane's correction functions through ``arc/scripts/get_species_corrections.py``
    in the RMG conda environment.

    Returns a dict ``{label: {'aec': {...}, 'bac': {...}}}``. Species whose
    inputs are insufficient (no xyz, no bonds, etc.) are omitted; per-species
    Arkane errors land as ``aec_error``/``bac_error`` keys and are dropped
    silently when the result is consumed (the correction is omitted for that
    species). Returns ``{}`` on any whole-batch failure so that downstream
    output.yml writing proceeds without corrections rather than aborting.
    """
    if arkane_level_of_theory is None:
        return {}
    from arc.common import NUMBER_BY_SYMBOL

    # Resolve the same matched LevelOfTheory key string that
    # _get_energy_corrections uses, via fuzzy match against the RMG QM
    # corrections data files. This is what the rmg_env script's regex
    # parser knows how to reconstruct.
    try:
        qm_corr_files = get_qm_corrections_files()
        lot_str = find_best_across_files(
            arkane_level_of_theory, qm_corr_files,
            AEC_SECTION_START, AEC_SECTION_END,
        )
    except Exception:
        lot_str = None
    if not lot_str:
        return {}

    species_inputs: list[dict] = []
    for label, spc in species_dict.items():
        xyz = spc.final_xyz if spc.final_xyz is not None else spc.initial_xyz
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

    if not species_inputs:
        return {}

    rmg_env = settings.get('RMG_ENV_NAME', 'rmg_env')
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

        commands = [
            'bash -lc "set -euo pipefail; '
            'if command -v micromamba >/dev/null 2>&1; then '
            f'    micromamba run -n {rmg_env} python {script_path} {tmp_in} {tmp_out}; '
            'elif command -v conda >/dev/null 2>&1; then '
            f'    conda run -n {rmg_env} python {script_path} {tmp_in} {tmp_out}; '
            'elif command -v mamba >/dev/null 2>&1; then '
            f'    mamba run -n {rmg_env} python {script_path} {tmp_in} {tmp_out}; '
            'else '
            '    echo \'micromamba/conda/mamba required\' >&2; exit 1; '
            'fi"',
        ]
        _, stderr = execute_command(command=commands, executable='/bin/bash')
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


def _aec_atom_params(aec_table: dict | None) -> list[dict]:
    """Translate ARC's run-level ``atom_energy_corrections`` mapping into
    TCKDB ``SchemeAtomParamPayload`` shape, sorted by element for
    deterministic output.yml. Returns ``[]`` when the table is empty or
    missing — TCKDB then has no scheme params to persist, but the applied
    correction row still lands."""
    if not aec_table:
        return []
    return [
        {'element': str(elem), 'value': float(value)}
        for elem, value in sorted(aec_table.items())
    ]


def _pbac_bond_params(bac_table: dict | None) -> list[dict]:
    """Translate ARC's run-level ``bond_additivity_corrections`` mapping
    into TCKDB ``SchemeBondParamPayload`` shape, sorted by bond_key for
    deterministic output.yml."""
    if not bac_table:
        return []
    return [
        {'bond_key': str(key), 'value': float(value)}
        for key, value in sorted(bac_table.items())
    ]


def _build_applied_corrections_for_species(
    label: str,
    species_corrections: dict[str, dict],
    arkane_level_of_theory,
    bac_type: str | None,
    *,
    aec_table: dict | None = None,
    bac_table: dict | None = None,
) -> list[dict]:
    """Build the per-species ``applied_energy_corrections`` list for output.yml.

    Translates the rmg_env script's per-species totals + components into the
    output.yml shape (mirroring TCKDB's ``AppliedEnergyCorrectionUploadPayload``
    contract): each entry has ``application_role``, ``value``, ``value_unit``,
    ``scheme``, and (optional) ``components``. Failure rows from the script
    (``aec_error``/``bac_error``) are silently dropped — that species simply
    has the failing role omitted.

    ``aec_table`` and ``bac_table`` are the run-level parameter dicts ARC
    already retrieves from the RMG database (see ``_get_energy_corrections``).
    Attached to each scheme as ``atom_params`` / ``bond_params`` so TCKDB
    populates the ``energy_correction_scheme_atom_param`` /
    ``energy_correction_scheme_bond_param`` reference tables — without
    them the applied correction lands but the scheme rows that link to
    parameter values stay empty. mBAC schemes never carry params: the
    Melius parameter table doesn't fit ``SchemeBondParamPayload``'s
    bond-key shape (it's atom-pair indexed and includes mol-level
    corrections), and there's no safe coercion at the producer.
    """
    entry = species_corrections.get(label) or {}
    applied: list[dict] = []
    lot_dict = _level_to_dict(arkane_level_of_theory)

    aec_block = entry.get('aec')
    if aec_block and aec_block.get('value') is not None:
        scheme: dict = {
            'kind': 'atom_energy',
            'name': 'atom_energy',
            'level_of_theory': lot_dict,
            'units': aec_block.get('value_unit', 'hartree'),
            'version': None,
            'source_literature': None,
            'note': 'Per-species AEC computed by Arkane.',
        }
        atom_params = _aec_atom_params(aec_table)
        if atom_params:
            scheme['atom_params'] = atom_params
        applied.append({
            'application_role': 'aec_total',
            'value': float(aec_block['value']),
            'value_unit': aec_block.get('value_unit', 'hartree'),
            'scheme': scheme,
            'components': aec_block.get('components') or [],
        })

    bac_block = entry.get('bac')
    if bac_block and bac_block.get('value') is not None and bac_type in _BAC_KIND_BY_TYPE:
        scheme_kind = _BAC_KIND_BY_TYPE[bac_type]
        components = bac_block.get('components')
        # Drop components when any bond is missing a parameter — partial
        # decompositions would not sum to the total and would mislead
        # downstream consumers.
        if components and any(c.get('parameter_value') is None for c in components):
            components = None
        scheme = {
            'kind': scheme_kind,
            'name': scheme_kind,
            'level_of_theory': lot_dict,
            'units': bac_block.get('value_unit', 'kcal_mol'),
            'version': None,
            'source_literature': None,
            'note': f'Per-species BAC computed by Arkane (bac_type={bac_type}).',
        }
        # Petersson BAC has a clean (bond_key → value) parameter table;
        # Melius does not — its parameters are atom-pair / length /
        # neighbor / molecular and need ``component_params``, which ARC
        # doesn't currently surface. Per spec we omit params for mBAC
        # rather than fabricate or coerce.
        if bac_type == 'p':
            bond_params = _pbac_bond_params(bac_table)
            if bond_params:
                scheme['bond_params'] = bond_params
        applied.append({
            'application_role': 'bac_total',
            'value': float(bac_block['value']),
            'value_unit': bac_block.get('value_unit', 'kcal_mol'),
            'scheme': scheme,
            'components': components or [],
        })

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
    rmg_env = settings.get('RMG_ENV_NAME', 'rmg_env')
    script_path = os.path.join(ARC_PATH, 'arc', 'scripts', 'get_point_groups.py')

    # Build input dict: {label: {symbols: [...], coords: [...]}}
    pg_input: dict[str, Any] = {}
    for label, spc in species_dict.items():
        xyz = spc.final_xyz if spc.final_xyz is not None else spc.initial_xyz
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

        commands = [
            'bash -lc "set -euo pipefail; '
            'if command -v micromamba >/dev/null 2>&1; then '
            f'    micromamba run -n {rmg_env} python {script_path} {tmp_in} {tmp_out}; '
            'elif command -v conda >/dev/null 2>&1; then '
            f'    conda run -n {rmg_env} python {script_path} {tmp_in} {tmp_out}; '
            'elif command -v mamba >/dev/null 2>&1; then '
            f'    mamba run -n {rmg_env} python {script_path} {tmp_in} {tmp_out}; '
            'else '
            '    echo \'micromamba/conda/mamba required\' >&2; exit 1; '
            'fi"',
        ]
        _, stderr = execute_command(command=commands, executable='/bin/bash')
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
                  software_by_job: dict[str, str | None] | None = None) -> dict:
    """Build the per-species/TS section for output.yml.

    ``software_by_job`` is an optional ``{'opt': name, 'freq': name,
    'sp': name}`` map that lets this function emit per-job input-deck
    paths (``opt_input``, ``freq_input``, ``sp_input``) alongside the
    log paths. When omitted (the back-compat path), the input fields
    come out as ``None`` and downstream consumers proceed with logs only.
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
    xyz = spc.final_xyz if spc.final_xyz is not None else spc.initial_xyz
    if xyz is None and converged and not spc.is_ts \
            and spc.mol is not None and len(spc.mol.atoms) == 1:
        # Monoatomic species skip opt entirely (nothing to optimize), so neither
        # final_xyz nor initial_xyz is populated. Synthesize the trivial
        # one-atom geometry so output.yml carries a usable xyz for downstream
        # consumers (e.g. the TCKDB uploader).
        xyz = {'symbols': (spc.mol.atoms[0].element.symbol,),
               'coords': ((0.0, 0.0, 0.0),)}
    d['xyz'] = xyz_to_str(xyz) if xyz is not None else None

    # ── screened conformers ────────────────────────────────────────────────
    # ARC's conformer screen often produces multiple optimized geometries
    # at the same level. ``ARCSpecies.conformers`` holds them as xyz
    # dicts, with ``conformer_energies`` holding the post-screen E0 in
    # kJ/mol *relative* to the lowest. Surface both lists when populated
    # so downstream consumers (TCKDB bundle) can emit the full set
    # rather than only the selected conformer of record. Skipped (key
    # omitted) when ARC has no screened conformers — backward compatible.
    raw_conformers = getattr(spc, 'conformers', None) or []
    if raw_conformers:
        d['conformers'] = [xyz_to_str(c) if not isinstance(c, str) else c
                           for c in raw_conformers]
        raw_energies = getattr(spc, 'conformer_energies', None) or []
        d['conformer_energies'] = list(raw_energies)

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
    if converged and coarse_path:
        d['coarse_opt_log'] = _make_rel_path(coarse_path, project_directory)
        d['coarse_opt_n_steps'], d['coarse_opt_final_energy_hartree'], coarse_final_xyz = \
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
        d['opt_n_steps'], d['opt_final_energy_hartree'], _ = _parse_opt_log(
            paths.get('geo') or None, project_directory
        )
    else:
        d['opt_n_steps'], d['opt_final_energy_hartree'] = None, None

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
        d['coarse_opt_output_xyz'] = coarse_final_xyz
        d['opt_input_xyz'] = coarse_final_xyz
    else:
        # Either no coarse stage, or coarse ran but its geometry wasn't
        # parseable. Producers downstream (TCKDB bundle) won't emit a
        # standalone ``opt_coarse`` calc in that case — they need the
        # output geometry to chain. Honest-empty beats fake-provenance.
        d['coarse_opt_input_xyz'] = None
        d['coarse_opt_output_xyz'] = None
        d['opt_input_xyz'] = initial_xyz_str

    # ── freq results ────────────────────────────────────────────────────────
    if is_mono:
        d['freq_n_imag'] = None
        d['imag_freq_cm1'] = None
    elif converged and not spc.is_ts:
        d['freq_n_imag'] = 0
        d['imag_freq_cm1'] = None          # never meaningful for stable species
    elif converged and spc.is_ts:
        d['freq_n_imag'] = 1
        d['imag_freq_cm1'] = _get_ts_imag_freq(spc)
    else:
        d['freq_n_imag'] = None
        d['imag_freq_cm1'] = None

    # ── log file paths (run-relative) ────────────────────────────────────────
    d['opt_log'] = _make_rel_path(paths.get('geo') or None, project_directory)
    d['freq_log'] = _make_rel_path(paths.get('freq') or None, project_directory)
    d['sp_log'] = _make_rel_path(paths.get('sp') or None, project_directory)

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

    # ── held-fixed coordinate constraints (for TCKDB) ──────────────────────
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

    # ── per-calc final effective settings (for TCKDB) ──────────────────────
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

    # ── S**2 spin-contamination diagnostic (for TCKDB, sp calc) ─────────────
    # A property of the (unrestricted) wavefunction, parsed from the sp job's
    # log (falling back to freq/opt when the sp energy reused the opt output).
    # Only populated for open-shell/unrestricted calcs where the ESS reported
    # ``<S**2>``; ``None`` (key present, null value) for restricted /
    # closed-shell species — the TCKDB adapter omits the block in that case.
    d['sp_spin_diagnostic'] = _parse_spin_diagnostic(
        paths.get('sp') or None,
        paths.get('freq') or None,
        paths.get('geo') or None,
        spc.multiplicity,
        project_directory,
    ) if converged else None

    # ── ESS software version (from SP log, or fall back to geo/freq log) ──
    d['ess_versions'] = _get_ess_versions(paths, project_directory) if converged else None

    if spc.is_ts:
        d['chosen_ts_method'] = getattr(spc, 'chosen_ts_method', None)
        d['successful_ts_methods'] = getattr(spc, 'successful_methods', None) or None
        d['ts_guesses'] = _ts_guesses_to_list(spc, project_directory)
        d['neb_log'] = _make_rel_path(paths.get('neb') or None, project_directory)
        # Path-search log slot for xtb_gsm-chosen TS guesses. Distinct
        # from ``neb_log`` so the TCKDB adapter can dispatch a method-
        # aware ``path_search`` parent calc (``method=neb`` vs
        # ``method=gsm``) without inspecting the file. Null when the
        # chosen TS-guess method is not GSM (or when GSM ran but didn't
        # surface a stringfile path).
        d['gsm_log'] = _make_rel_path(paths.get('gsm') or None, project_directory)

        # Fallback: when ``paths['neb']`` / ``paths['gsm']`` is empty
        # but the chosen TSGuess carries a ``log_path``, route it to
        # the matching log field. Covers restart-restored runs where
        # the scheduler's TS-selection write sites
        # (``determine_most_likely_ts_conformer`` /
        # ``run_ts_conformer_jobs``) don't re-fire because
        # ``chosen_ts`` is already pinned. Method dispatch is strict —
        # no GSM-log-into-neb_log cross-pollination — and gated on the
        # *chosen* guess (per the calculation_dependency contract: a
        # log from a non-winning guess is irrelevant provenance).
        chosen_idx = getattr(spc, 'chosen_ts', None)
        ts_guesses = getattr(spc, 'ts_guesses', None) or []
        if (chosen_idx is not None
                and isinstance(chosen_idx, int)
                and 0 <= chosen_idx < len(ts_guesses)):
            chosen_guess = ts_guesses[chosen_idx]
            chosen_log = getattr(chosen_guess, 'log_path', None)
            chosen_method = (getattr(chosen_guess, 'method', None)
                             or d.get('chosen_ts_method'))
            log_field = _ts_guess_log_field_for_method(chosen_method)
            if chosen_log and log_field and not d.get(log_field):
                d[log_field] = _make_rel_path(chosen_log, project_directory)
            # Additive fallback for dedup-merged guesses: when the chosen
            # guess's own (primary) method is geometry-only but a
            # path-search method (xtb_gsm / orca_neb) merged into it during
            # equivalent-guess clustering, route that source's preserved log
            # (``method_source_paths``) into the matching field. Mirrors
            # ``scheduler._ts_guess_path_provenance`` so restart-restored
            # runs — where the scheduler's TS-selection write sites don't
            # re-fire — still surface the path-search log. Does not change
            # which method won selection. Only fires when neither path-log
            # field was populated above (i.e. the scheduler's ``paths`` write
            # didn't run), and sets exactly one field — first path source in
            # ``method_sources`` order — so output.yml keeps the scheduler's
            # single-slot, first-source-wins invariant and never claims two
            # path logs for one guess.
            if not d.get('neb_log') and not d.get('gsm_log'):
                source_paths = getattr(chosen_guess, 'method_source_paths', None) or {}
                for source in (getattr(chosen_guess, 'method_sources', None) or []):
                    source_field = _ts_guess_log_field_for_method(source)
                    source_log = source_paths.get(source)
                    if source_field and source_log:
                        d[source_field] = _make_rel_path(source_log, project_directory)
                        break
        irc_paths = list(paths.get('irc') or [])
        d['irc_logs'] = [_make_rel_path(p, project_directory) for p in irc_paths]
        # Per-log direction in lockstep with ``irc_logs``: 'forward' /
        # 'reverse' / null when the scheduler couldn't observe it (older
        # projects, restarts predating the parallel-list tracking).
        # Padded to ``len(irc_logs)`` so consumers can zip the two lists
        # without index-out-of-range checks.
        irc_dirs_raw = list(paths.get('irc_directions') or [])
        d['irc_log_directions'] = (
            irc_dirs_raw + [None] * (len(irc_paths) - len(irc_dirs_raw))
        )[:len(irc_paths)]
        if not irc_requested:
            d['irc_converged'] = None
        else:
            d['irc_converged'] = entry.get('job_types', {}).get('irc', False)
        d['rxn_label'] = spc.rxn_label

    # ── thermochemistry (non-TS converged species only) ──────────────────────
    if not spc.is_ts and converged and spc.thermo is not None and spc.thermo.H298 is not None:
        d['thermo'] = _thermo_to_dict(spc.thermo)
    else:
        d['thermo'] = None

    # ── statistical mechanics ────────────────────────────────────────────────
    if not is_mono and converged:
        pg = (point_groups or {}).get(label)
        d['statmech'] = _statmech_to_dict(spc, project_directory, point_group=pg)
    else:
        d['statmech'] = None

    # ── additional calculations (rotor scans, etc.) ─────────────────────────
    # Bundle-local calcs that aren't part of the opt → freq → sp chain. The
    # field is a list (possibly empty) so consumers can iterate without a
    # None-guard. Currently populated only for converged non-monoatomic
    # species with successful 1D rotor scans; everything else gets ``[]``.
    if not is_mono and converged:
        d['additional_calculations'] = _build_scan_calculations(spc, project_directory)
    else:
        d['additional_calculations'] = []

    return d


def _get_ts_imag_freq(spc) -> float | None:
    """Return the imaginary frequency (cm⁻¹) of the TS, or None."""
    # Primary: take the most negative frequency from spc.freqs (all freqs from the freq job)
    freqs = getattr(spc, 'freqs', None)
    if freqs:
        neg = [f for f in freqs if f < 0]
        if neg:
            return float(min(neg))
    # Fallback: try the chosen TS guess's imaginary_freqs
    try:
        chosen = spc.chosen_ts
        if chosen is not None and spc.ts_guesses and chosen < len(spc.ts_guesses):
            im_freqs = spc.ts_guesses[chosen].imaginary_freqs
            if im_freqs:
                return float(im_freqs[0])
    except Exception:
        logger.debug('Failed to obtain TS imaginary frequency from ts_guesses for %s', spc.label, exc_info=True)
    return None


def _thermo_to_dict(thermo) -> dict:
    """Convert a ThermoData object to a plain, unit-labeled dict."""
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
    }

    # ── Per-temperature thermochemistry ──────────────────────────────────────
    # ``thermo_points`` carries the full TCKDB ``thermo_point`` shape
    # (Cp + H + S + G at each tabulated T). Falls back to building a
    # Cp-only point list from RMG's ``Tdata``/``Cpdata`` when only that
    # legacy form is available — H/S/G stay omitted because we don't
    # have the polynomial in scope here to evaluate them. The TCKDB
    # adapter accepts either shape.
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


def _statmech_to_dict(spc, project_directory: str, point_group: str | None = None) -> dict:
    """Build the statmech sub-section for a non-monoatomic converged species/TS."""
    # Use the cached private attribute to avoid triggering a geometry re-read
    is_linear = spc._is_linear

    if spc.is_monoatomic() is True:
        rotor_kind = 'atom'
    elif is_linear:
        rotor_kind = 'linear'
    else:
        rotor_kind = 'asymmetric_top'

    freqs = getattr(spc, 'freqs', None)
    if freqs is not None and spc.is_ts:
        # Exclude the imaginary mode (most negative frequency)
        freqs = [f for f in freqs if f >= 0]

    return {
        'e0_kj_mol': spc.e0,
        'spin_multiplicity': spc.multiplicity,
        'optical_isomers': spc.optical_isomers,
        'is_linear': is_linear,
        'external_symmetry': spc.external_symmetry,
        'point_group': point_group,
        'rigid_rotor_kind': rotor_kind,
        'harmonic_frequencies_cm1': [float(f) for f in freqs] if freqs is not None else None,
        'torsions': _get_torsions(spc, project_directory),
    }


def _get_torsions(spc, project_directory: str) -> list[dict]:
    """Build the torsions list from spc.rotors_dict.

    Each emitted torsion carries a ``source_scan_calculation_key`` (e.g.
    ``"scan_rotor_3"``) when its scan log is on disk and parseable. The key
    matches the bundle-local key used by :func:`_build_scan_calculations`,
    so TCKDB consumers can resolve the torsion's underlying scan calc.
    Rotors whose scan log is missing or fails to parse get the field set to
    ``None`` rather than fabricating a key that points at no calc.
    """
    if not getattr(spc, 'rotors_dict', None):
        return []
    torsions = []
    for rotor_index, rotor in spc.rotors_dict.items():
        if rotor.get('success') is not True:
            continue
        scan = rotor.get('scan')       # 4-atom dihedral defining atoms, 1-indexed
        pivots = rotor.get('pivots')   # 2-atom rotation axis, 1-indexed
        symmetry = rotor.get('symmetry', 1)
        rotor_type = rotor.get('type', 'HinderedRotor')
        treatment = 'free_rotor' if 'Free' in str(rotor_type) else 'hindered_rotor'
        scan_key = (
            f'scan_rotor_{rotor_index}'
            if _scan_log_is_parseable(rotor, project_directory) else None
        )
        torsions.append({
            'symmetry_number': symmetry,
            'treatment': treatment,
            'atom_indices': scan,
            'pivot_atoms': pivots,
            'barrier_kj_mol': _get_rotor_barrier(rotor, project_directory),
            'source_scan_calculation_key': scan_key,
        })
    return torsions


def _resolve_scan_path(rotor: dict, project_directory: str) -> str | None:
    """Return an absolute, on-disk scan log path for ``rotor``, or ``None``.

    Centralizes the rules used by both the barrier helper and the
    scan-calc builder so they agree on which rotors qualify as "has a
    scan log we can use."
    """
    scan_path = rotor.get('scan_path', '')
    if not scan_path:
        return None
    if not os.path.isabs(scan_path):
        scan_path = os.path.join(project_directory, scan_path)
    if not os.path.isfile(scan_path):
        return None
    return scan_path


def _scan_log_is_parseable(rotor: dict, project_directory: str) -> bool:
    """Cheap presence-check for a usable 1D scan log on disk."""
    return _resolve_scan_path(rotor, project_directory) is not None


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


def _build_scan_calculations(spc, project_directory: str) -> list[dict]:
    """Build the species' ``additional_calculations`` list from rotor scans.

    Emits one ``type='scan'`` calc per successful 1D rotor whose scan log
    is on disk and parses cleanly. Each entry shape:

        {
            'key': 'scan_rotor_<i>',
            'type': 'scan',
            'scan_result': {
                'dimension': 1,
                'is_relaxed': True,
                'zero_energy_reference_hartree': float | None,
                'coordinates': [{coordinate_index, coordinate_kind, atom*_index,
                                 step_count, value_unit, symmetry_number}],
                'points': [{point_index, electronic_energy_hartree,
                            relative_energy_kj_mol,
                            coordinate_values: [{coordinate_index,
                                                 coordinate_value, value_unit}],
                            xyz: str | None}],
            },
        }

    Rotors that don't produce a parseable scan_result are skipped — the
    matching torsion's ``source_scan_calculation_key`` will be ``None``
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
        if rotor.get('success') is not True:
            continue
        if rotor.get('dimensions', 1) != 1:
            continue
        scan_result = _build_scan_result_for_rotor(
            rotor, project_directory, input_xyz=input_xyz,
        )
        if scan_result is None:
            continue
        scan_constraints = _parse_scan_constraints(rotor, project_directory)
        entry = {
            'key': f'scan_rotor_{rotor_index}',
            'type': 'scan',
            'scan_result': scan_result,
        }
        if scan_constraints:
            entry['constraints'] = scan_constraints
        out.append(entry)
    return out


def _parse_scan_constraints(rotor: dict, project_directory: str) -> list[dict]:
    """Best-effort extraction of held-fixed constraints for a rotor scan.

    Dispatches to the per-software constraint parser based on the
    rotor's ``scan_software`` hint (set by the scheduler when the scan
    job completes). Returns the held-fixed constraints — the active
    scan coordinate is excluded; it lives in ``scan_result.coordinates[]``.

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
    doesn't break ``additional_calculations`` for the rest of the run.
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
    if not software:
        from arc.parser.adapters.gaussian import parse_gaussian_constraints
        return parse_gaussian_constraints
    if software == 'gaussian':
        from arc.parser.adapters.gaussian import parse_gaussian_constraints
        return parse_gaussian_constraints
    if software == 'orca':
        from arc.parser.adapters.orca import parse_orca_constraints
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
    ``None`` (caller falls through to "unknown"); raised exception or
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
            "Scan start_value/end_value: dihedral calculation failed for "
            "atoms=%r in '%s': %s; omitting start_value/end_value.",
            scan_atoms, scan_path, exc,
        )
        return None
    if math.isnan(angle):  # colinear atoms in the input geometry.
        logger.warning(
            "Scan start_value/end_value: dihedral is NaN (colinear quartet?) "
            "for atoms=%r in '%s'; omitting start_value/end_value.",
            scan_atoms, scan_path,
        )
        return None
    return float(angle)


def _xyz_dict_to_tckdb_xyz_text(xyz_dict: dict | None) -> str | None:
    """Serialize an ARC xyz dict into TCKDB's count-headered xyz_text.

    Mirrors the conformer-side normalization the adapter applies to the
    species ``xyz`` field: bare atom-only string from
    ``arc.species.converter.xyz_to_str`` plus a TCKDB-required
    ``<n_atoms>\\n<comment>\\n<atoms>`` envelope. Returns ``None`` for
    null/empty input so the caller can decide whether to omit the
    field; serialization errors propagate as exceptions for the caller
    to handle (uniform-drop on failure rather than per-point drop).
    """
    if xyz_dict is None:
        return None
    bare = xyz_to_str(xyz_dict=xyz_dict)
    if not bare:
        return None
    text = bare.strip()
    if not text:
        return None
    return f"{len(text.splitlines())}\n\n{text}"


def _build_scan_result_for_rotor(
    rotor: dict,
    project_directory: str,
    *,
    input_xyz: dict | None = None,
) -> dict | None:
    """Parse the rotor's scan log and shape it into a TCKDB ``scan_result`` dict.

    Returns ``None`` when:
      * no scan log on disk (covered by ``_resolve_scan_path``)
      * parser fails to extract angles or relative energies (the only
        two fields that drive the shape — absolute Hartree and
        geometries are nice-to-have)
      * atom_indices isn't a 4-int 1-based dihedral quartet (the
        coordinates entry would be unusable downstream)

    On the happy path the dict carries plain Python primitives only, so
    YAML emission via ``save_yaml_file`` Just Works. Per-point
    geometries are emitted under ``points[i].geometry.xyz_text`` when
    the parser returns a per-step geometry list aligned with the
    energy/angle list (TCKDB resolves these server-side into
    ``calc_scan_point.geometry_id``); on length mismatch or
    serialization failure the geometries are dropped uniformly across
    all points and a warning is logged — partial coverage would imply
    a per-point alignment we can't actually verify.
    """
    scan_path = _resolve_scan_path(rotor, project_directory)
    if scan_path is None:
        return None

    try:
        parsed = parse_1d_scan_full_result(log_file_path=scan_path)
    except Exception:
        logger.debug(f"Failed to parse 1D rotor scan result from '{scan_path}'", exc_info=True)
        return None

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
    # ``geometry.xyz_text`` (TCKDB count-headered format) only when the
    # list aligns 1:1 with the energy list; otherwise drop wholesale —
    # mixing populated and missing entries would imply an alignment we
    # can't verify, and the schema accepts ``geometry`` as omitted but
    # not as ``null``.
    geometries = parsed.get('geometries')
    point_geometry_xyz_texts: list[str] | None = None
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
                    _xyz_dict_to_tckdb_xyz_text(g) for g in geometries
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
    a1, a2, a3, a4 = scan_atoms

    symmetry = rotor.get('symmetry')
    coord: dict[str, Any] = {
        'coordinate_index': 1,
        'coordinate_kind': 'dihedral',
        'atom1_index': a1,
        'atom2_index': a2,
        'atom3_index': a3,
        'atom4_index': a4,
        'step_count': len(angles),
        'value_unit': 'degree',
    }
    if isinstance(symmetry, int) and symmetry >= 1:
        coord['symmetry_number'] = symmetry

    # Requested grid metadata: ``parse_scan_args`` reads the
    # ModRedundant header that Gaussian echoes back into its log
    # (``D a b c d S <step_count> <step_size>``), giving us the exact
    # step size the user requested rather than one inferred from the
    # completed point spacing. ORCA / other ESS raise
    # ``NotImplementedError`` from the same parser; in those cases
    # the grid fields stay absent rather than guessed at — TCKDB
    # treats null as "unknown grid", which is honest.
    requested_step_size: float | None = None
    try:
        scan_args = parse_scan_args(scan_path)
        raw_step_size = scan_args.get('step_size')
        if isinstance(raw_step_size, (int, float)) and raw_step_size > 0:
            requested_step_size = float(raw_step_size)
    except NotImplementedError:
        # Non-Gaussian ESS: parser doesn't speak this log format yet.
        pass
    except Exception:
        logger.debug(f"parse_scan_args failed for '{scan_path}'", exc_info=True)
    if requested_step_size is not None:
        coord['step_size'] = requested_step_size
        # 1D dihedral torsion scans: the resolution IS the step size.
        # ND or non-dihedral scans would need a different mapping;
        # this whole helper is the 1D path so the equivalence holds.
        coord['resolution_degrees'] = requested_step_size

    # ``start_value``/``end_value`` describe the requested grid, not
    # the completed-point spacing. Computing them honestly requires
    # both the requested step size (above) AND the input dihedral
    # (the geometry the user pointed the scan at). The latter comes
    # from the species record — preferred — falling back to the
    # first parsed scan-iteration geometry, which for Gaussian
    # ModRedundant scans has the dihedral held fixed at the input
    # value by construction (so it's not "inferring from outputs",
    # it's reading a frozen DOF). ``end_value`` is then exact:
    # ``start + step_size * (step_count - 1)``. We deliberately do
    # NOT wrap into [-180, 180]: a full rotation must land at
    # ``start + 360°``, not back at ``start`` — TCKDB's column has
    # no range constraint, and continuity is what downstream
    # consumers (rotor-treatment plotters, etc.) expect.
    step_count_for_grid = len(rel_energies)
    if (
        requested_step_size is not None
        and step_count_for_grid >= 1
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
            coord['start_value'] = start_value
            coord['end_value'] = (
                start_value + requested_step_size * (step_count_for_grid - 1)
            )

    points: list[dict[str, Any]] = []
    for i, (angle, rel_e) in enumerate(zip(angles, rel_energies), start=1):
        point: dict[str, Any] = {
            'point_index': i,
            'relative_energy_kj_mol': float(rel_e),
            'coordinate_values': [{
                'coordinate_index': 1,
                'coordinate_value': float(angle),
                'value_unit': 'degree',
            }],
        }
        if abs_energies is not None:
            point['electronic_energy_hartree'] = float(abs_energies[i - 1])
        if point_geometry_xyz_texts is not None:
            point['geometry'] = {'xyz_text': point_geometry_xyz_texts[i - 1]}
        points.append(point)

    scan_result: dict[str, Any] = {
        'dimension': 1,
        'is_relaxed': True,
        'coordinates': [coord],
        'points': points,
    }
    zero_ref = parsed.get('zero_energy_reference_hartree')
    if isinstance(zero_ref, (int, float)):
        scan_result['zero_energy_reference_hartree'] = float(zero_ref)

    return scan_result


def _rxn_to_dict(rxn) -> dict:
    """Convert an ARCReaction to a plain dict for output.yml."""
    kinetics = rxn.kinetics
    kin_dict: dict | None = None
    if kinetics is not None:
        A = kinetics.get('A')
        Ea = kinetics.get('Ea')
        Tmin = kinetics.get('Tmin')
        Tmax = kinetics.get('Tmax')
        kin_dict = {
            'A': A[0] if isinstance(A, (tuple, list)) else A,
            'A_units': A[1] if isinstance(A, (tuple, list)) else None,
            'n': kinetics.get('n'),
            'Ea': Ea[0] if isinstance(Ea, (tuple, list)) else Ea,
            'Ea_units': Ea[1] if isinstance(Ea, (tuple, list)) else None,
            'Tmin_k': Tmin[0] if isinstance(Tmin, (tuple, list)) else Tmin,
            'Tmax_k': Tmax[0] if isinstance(Tmax, (tuple, list)) else Tmax,
            'dA': kinetics.get('dA'),
            'dn': kinetics.get('dn'),
            'dEa': kinetics.get('dEa'),
            'dEa_units': kinetics.get('dEa_units'),
            'n_data_points': kinetics.get('n_data_points'),
            # ARC always renders the same tunneling method into Arkane's
            # input template (currently 'Eckart'); record it here so
            # downstream consumers know which correction was applied to
            # the fitted A/n/Ea. If Arkane's parsed kinetics carries an
            # explicit tunneling marker in the future, prefer that;
            # otherwise fall back to the template constant.
            'tunneling': kinetics.get('tunneling') or ARKANE_TUNNELING_METHOD,
        }

    rxn_dict: dict = {
        'label': rxn.label,
        'reactant_labels': list(rxn.reactants),
        'product_labels': list(rxn.products),
        'family': rxn.family,
        'multiplicity': rxn.multiplicity,
        'ts_label': rxn.ts_label,
        'kinetics': kin_dict,
    }
    long_kin_desc = getattr(rxn, 'long_kinetic_description', None)
    if long_kin_desc:
        rxn_dict['long_kinetic_description'] = long_kin_desc
    return rxn_dict
