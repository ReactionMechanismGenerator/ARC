"""
Module for writing the consolidated output.yml at the end of an ARC run.

output.yml supersedes output/status.yml and <project>.info: it consolidates
all result data into a single file with run-relative paths so downstream
consumers (TCKDB, analysis scripts) need only one file.

Written atomically at the very end of a run. If the run is interrupted, the
file will not exist rather than be partially written.
"""

import datetime
import os
import tempfile
from typing import Any

from arc.common import ARC_PATH, VERSION, get_git_commit, get_logger, read_yaml_file, save_yaml_file
from arc.constants import E_h_kJmol
from arc.imports import settings
from arc.job.adapters.common import open_shell_character_source
from arc.job.env_run import rmg_env_command
from arc.job.local import execute_command
from arc.parser.parser import (parse_1d_scan_energies, parse_e_elect, parse_ess_version, parse_opt_steps,
                               parse_s_squared, parse_wavefunction_stability, parse_zpe_correction,
                               s_squared_expected_from_multiplicity)
from arc.species.converter import xyz_to_str
from arc.statmech.arkane import (
    AEC_SECTION_START, AEC_SECTION_END,
    MBAC_SECTION_START, MBAC_SECTION_END,
    PBAC_SECTION_START, PBAC_SECTION_END,
    find_best_across_files, get_qm_corrections_files,
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
    """
    doc: dict[str, Any] = {}

    # ---- header ----------------------------------------------------------------
    doc['schema_version'] = '1.0'
    arc_git, _ = get_git_commit(ARC_PATH)
    doc['project'] = project
    doc['arc_version'] = VERSION
    doc['arc_git_commit'] = arc_git or None
    doc['arkane_git_commit'] = _get_arkane_git_commit()
    doc['datetime_started'] = (
        datetime.datetime.fromtimestamp(t0).strftime('%Y-%m-%d %H:%M') if t0 is not None else None
    )
    doc['datetime_completed'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

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

    # ---- species and TSs --------------------------------------------------------
    point_groups = _compute_point_groups(species_dict, project_directory)
    doc['species'] = []
    doc['transition_states'] = []
    for spc in species_dict.values():
        d = _spc_to_dict(spc, output_dict, project_directory, point_groups, irc_requested=irc_requested)
        if spc.is_ts:
            doc['transition_states'].append(d)
        else:
            doc['species'].append(d)

    # ---- reactions --------------------------------------------------------------
    doc['reactions'] = [_rxn_to_dict(rxn) for rxn in reactions]

    # ---- atomic write -----------------------------------------------------------
    out_dir = os.path.join(project_directory, 'output')
    os.makedirs(out_dir, exist_ok=True)
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


def _scf_reference_block(spc, stability: dict | None) -> dict:
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
    different one.

    ``verdict`` falls back to the stability log when the species carries none, as a restart
    written before the species held one does, while ``source`` never does: the reference
    decision reads the species and only the species, so a verdict that reached the log but not
    the species decided nothing and is reported without being credited with the decision.

    The block is emitted whether or not the species converged, like the ``wavefunction_stability``
    entry it explains and unlike the parsed results around it. It records a decision ARC made
    rather than a quantity a job produced, and a species that adopted a verdict and then failed to
    converge is exactly the case where knowing ARC changed its reference explains the failure.

    Returns: dict
        A flat mapping of scalars; keys are always present, with ``None`` where unknown.
    """
    verdict = getattr(spc, 'derived_stability_verdict', None)
    if not isinstance(verdict, dict):
        verdict = stability if isinstance(stability, dict) else None
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
            'log': stability.get('log') if isinstance(stability, dict) else None,
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
    Parse opt_n_steps and opt_final_energy_hartree from the geometry opt log.

    Returns:
        (opt_n_steps, opt_final_energy_hartree), either could be None.
    """
    if not geo_path:
        return None, None
    if not os.path.isabs(geo_path):
        geo_path = os.path.join(project_directory, geo_path)
    if not os.path.isfile(geo_path):
        return None, None
    try:
        n_steps = parse_opt_steps(geo_path)
        e_elect_kj = parse_e_elect(geo_path)  # returns kJ/mol
        e_elect_hartree = e_elect_kj / E_h_kJmol if e_elect_kj is not None else None
        return n_steps, e_elect_hartree
    except Exception:
        return None, None


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

            command = rmg_env_command(py_args=[script_path, tmp_in, tmp_out])
            _, stderr = execute_command(command=command, shell=True, executable='/bin/bash')
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
                  point_groups: dict | None = None, irc_requested: bool = True) -> dict:
    """Build the per-species/TS section for output.yml."""
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
    d['xyz'] = xyz_to_str(xyz) if xyz is not None else None

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
    coarse_path = paths.get('geo_coarse') or None
    if converged and coarse_path:
        d['coarse_opt_log'] = _make_rel_path(coarse_path, project_directory)
        d['coarse_opt_n_steps'], d['coarse_opt_final_energy_hartree'] = \
            _parse_opt_log(coarse_path, project_directory)
    else:
        d['coarse_opt_log'] = None
        d['coarse_opt_n_steps'] = None
        d['coarse_opt_final_energy_hartree'] = None

    # ── fine opt (or only opt if no fine grid) ─────────────────────────────
    d['opt_n_steps'], d['opt_final_energy_hartree'] = _parse_opt_log(
        paths.get('geo') or None, project_directory) if converged else (None, None)

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

    # ── wavefunction stability diagnostic (null unless the job ran) ──────────
    stability = _parse_wavefunction_stability(paths.get('stability') or None, project_directory)
    d['wavefunction_stability'] = stability

    # ── which source decided the open-shell character, and the references used ──
    d['scf_reference'] = _scf_reference_block(spc, stability)

    # ── S**2 spin-contamination diagnostic (sp calc, open-shell only) ───────
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
        d['neb_log'] = _make_rel_path(paths.get('neb') or None, project_directory)
        d['irc_logs'] = [_make_rel_path(p, project_directory) for p in (paths.get('irc') or [])]
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

    return d


def _get_ts_imag_freq(spc) -> float | None:
    """Return the imaginary frequency (cm⁻¹) of the TS, or None.

    ``ARCSpecies.chosen_ts`` holds the chosen ``TSGuess.index``, which is that guess's stable
    identity, not its position in ``ts_guesses``: unsuccessful guesses are kept in the list and
    equivalent-guess clustering removes guesses while preserving the survivors' indices, so the
    two diverge. The chosen guess is therefore looked up by index.
    """
    # Primary: take the most negative frequency from spc.freqs (all freqs from the freq job)
    freqs = getattr(spc, 'freqs', None)
    if freqs:
        neg = [f for f in freqs if f < 0]
        if neg:
            return float(min(neg))
    # Fallback: try the chosen TS guess's imaginary_freqs
    try:
        chosen = spc.chosen_ts
        if chosen is not None and spc.ts_guesses:
            chosen_guess = next((tsg for tsg in spc.ts_guesses if tsg.index == chosen), None)
            im_freqs = chosen_guess.imaginary_freqs if chosen_guess is not None else None
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

    # ── Cp tabulation ────────────────────────────────────────────────────────
    cp = getattr(thermo, 'cp_data', None)
    if cp is not None:
        t['cp_data'] = cp
    elif thermo.Tdata is not None and thermo.Cpdata is not None:
        T_list = thermo.Tdata[0] if isinstance(thermo.Tdata, (list, tuple)) else thermo.Tdata
        Cp_list = thermo.Cpdata[0] if isinstance(thermo.Cpdata, (list, tuple)) else thermo.Cpdata
        t['cp_data'] = [{'temperature_k': float(T), 'cp_j_mol_k': float(Cp)}
                        for T, Cp in zip(T_list, Cp_list)]
    else:
        t['cp_data'] = None

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
    """Build the torsions list from spc.rotors_dict."""
    if not getattr(spc, 'rotors_dict', None):
        return []
    torsions = []
    for rotor in spc.rotors_dict.values():
        if rotor.get('success') is not True:
            continue
        scan = rotor.get('scan')       # 4-atom dihedral defining atoms, 1-indexed
        pivots = rotor.get('pivots')   # 2-atom rotation axis, 1-indexed
        symmetry = rotor.get('symmetry', 1)
        rotor_type = rotor.get('type', 'HinderedRotor')
        treatment = 'free_rotor' if 'Free' in str(rotor_type) else 'hindered_rotor'
        torsions.append({
            'symmetry_number': symmetry,
            'treatment': treatment,
            'atom_indices': scan,
            'pivot_atoms': pivots,
            'barrier_kj_mol': _get_rotor_barrier(rotor, project_directory),
        })
    return torsions


def _get_rotor_barrier(rotor: dict, project_directory: str) -> float | None:
    """
    Return max(V) - min(V) in kJ/mol from the 1D scan output file.

    parse_1d_scan_energies already zeroes the minimum, so max(energies) is the
    barrier height directly.
    """
    scan_path = rotor.get('scan_path', '')
    if not scan_path:
        return None
    if not os.path.isabs(scan_path):
        scan_path = os.path.join(project_directory, scan_path)
    if not os.path.isfile(scan_path):
        return None
    try:
        energies, _ = parse_1d_scan_energies(log_file_path=scan_path)
        if energies:
            return float(max(energies))
    except Exception:
        logger.debug(f"Failed to parse 1D rotor scan energies from '{scan_path}'", exc_info=True)
    return None


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
        }

    return {
        'label': rxn.label,
        'reactant_labels': list(rxn.reactants),
        'product_labels': list(rxn.products),
        'family': rxn.family,
        'multiplicity': rxn.multiplicity,
        'ts_label': rxn.ts_label,
        'kinetics': kin_dict,
    }
