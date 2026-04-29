#!/usr/bin/env python3
# encoding: utf-8

"""
ARC - Automatic Rate Calculator
"""

import argparse
import logging
import os

from arc.common import read_yaml_file
from arc.main import ARC
from arc.tckdb.config import TCKDBConfig, UPLOAD_MODE_COMPUTED_SPECIES


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by keywords.
    """
    parser = argparse.ArgumentParser(description='Automatic Rate Calculator (ARC)')
    parser.add_argument('file', metavar='FILE', type=str, nargs=1,
                        help='a file describing the job to execute')

    # Options for controlling the amount of information printed to the console
    # By default a moderate level of information is printed; you can either
    # ask for less (quiet), more (verbose), or much more (debug)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d', '--debug', action='store_true', help='print debug information')
    group.add_argument('-q', '--quiet', action='store_true', help='only print warnings and errors')

    args = parser.parse_args(command_line_args)
    args.file = args.file[0]

    return args


def main():
    """
    The main ARC executable function
    """
    args = parse_command_line_arguments()
    input_file = args.file
    project_directory = os.path.abspath(os.path.dirname(args.file))
    input_dict = read_yaml_file(path=input_file, project_directory=project_directory)
    if 'project' not in list(input_dict.keys()):
        raise ValueError('A project name must be provided!')

    verbose = logging.INFO
    if args.debug:
        verbose = logging.DEBUG
    elif args.quiet:
        verbose = logging.WARNING
    input_dict['verbose'] = input_dict['verbose'] if 'verbose' in input_dict else verbose
    if 'project_directory' not in input_dict or not input_dict['project_directory']:
        input_dict['project_directory'] = project_directory

    tckdb_config = TCKDBConfig.from_dict(input_dict.pop('tckdb', None))

    arc_object = ARC(**input_dict)
    arc_object.tckdb_config = tckdb_config
    if tckdb_config is not None:
        print(f'TCKDB integration enabled: {tckdb_config.base_url}')

    # Persistent SSH pool lives for the duration of the run; close it
    # explicitly on every exit path (success, error, ctrl-C) so we don't
    # leave paramiko Transports orphaned. Lazily instantiated on first
    # remote-queue job, so this is a no-op for fully-local runs.
    try:
        arc_object.execute()

        if tckdb_config is not None:
            from arc.tckdb.adapter import TCKDBAdapter
            adapter = TCKDBAdapter(tckdb_config, project_directory=arc_object.project_directory)
            _run_tckdb_upload_sweep(arc_object, adapter, tckdb_config)
    finally:
        from arc.job.ssh_pool import reset_default_pool
        reset_default_pool()


def _run_tckdb_upload_sweep(arc_object, adapter, tckdb_config):
    """End-of-run sweep: build/write/upload one TCKDB payload per converged species.

    Reads ``<project>/output/output.yml`` (the consolidated run summary
    from ``arc/output.py``) and dispatches per ``tckdb_config.upload_mode``:

    - ``"conformer"`` (default): one ``/uploads/conformers`` POST per
      species, followed by per-artifact POSTs to
      ``/calculations/{id}/artifacts`` for each configured kind.
    - ``"computed_species"``: one ``/uploads/computed-species`` bundle
      POST per species, with artifacts inlined under each calc; no
      separate artifact sweep.

    Both paths share the same per-species iteration, error handling,
    and summary print shape. TS records are deferred regardless of mode.
    """
    output_path = os.path.join(arc_object.project_directory, 'output', 'output.yml')
    if not os.path.exists(output_path):
        # Most common cause: the run was interrupted before
        # write_output_yml ran. Skip cleanly rather than scrape live
        # objects — the replay path expects output.yml as the contract.
        print(f'TCKDB upload skipped: {output_path} not found (run did not complete?)')
        return

    output_doc = read_yaml_file(path=output_path)
    species_records = list(output_doc.get('species') or [])
    ts_records = list(output_doc.get('transition_states') or [])
    # Both modes cover minima only; TS records are deferred to a future
    # TS-specific adapter method targeting /uploads/transition-states
    # (different schema, no SMILES requirement).
    n_ts_deferred = sum(1 for r in ts_records if r.get('converged'))

    is_bundle_mode = tckdb_config.upload_mode == UPLOAD_MODE_COMPUTED_SPECIES

    counts = {'uploaded': 0, 'skipped': 0, 'failed': 0}
    artifact_counts = {'uploaded': 0, 'skipped': 0, 'failed': 0}
    failures = []
    artifact_failures = []
    n_attempted = 0
    for record in species_records:
        label = record.get('label') or '<unlabeled>'
        if not record.get('converged'):
            continue
        n_attempted += 1
        try:
            if is_bundle_mode:
                # Single bundle carries species_entry + conformer +
                # opt/freq/sp + (optional) thermo + inlined artifacts.
                outcome = adapter.submit_computed_species_from_output(
                    output_doc=output_doc, species_record=record,
                )
            else:
                outcome = adapter.submit_from_output(
                    output_doc=output_doc, species_record=record,
                )
        except Exception as exc:
            counts['failed'] += 1
            failures.append((label, f'{type(exc).__name__}: {exc}'))
            continue
        if outcome is None:
            continue
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
        if outcome.status == 'failed':
            failures.append((label, outcome.error or 'unknown error'))
        elif (
            outcome.status == 'uploaded'
            and not is_bundle_mode
            and tckdb_config.artifacts.upload
        ):
            # Artifact sweep is conformer-mode only — the bundle path
            # carries artifacts inline under each calc.
            _sweep_artifacts_for_species(
                adapter=adapter,
                arc_object=arc_object,
                output_doc=output_doc,
                species_record=record,
                outcome=outcome,
                counts=artifact_counts,
                failures=artifact_failures,
                kinds=_implementable_kinds_from_config(tckdb_config),
            )

    mode_label = 'computed-species bundle' if is_bundle_mode else 'conformer/calculation'
    print(f'TCKDB v0 ({mode_label}, {n_attempted} converged species):')
    print(f'  uploaded: {counts["uploaded"]}  skipped: {counts["skipped"]}  failed: {counts["failed"]}')
    if not is_bundle_mode and tckdb_config.artifacts.upload:
        # Bundle mode rolls artifacts into the same upload, so a
        # standalone artifact summary line would be misleading.
        print(
            f'  artifacts: uploaded {artifact_counts["uploaded"]}  '
            f'skipped {artifact_counts["skipped"]}  failed {artifact_counts["failed"]}'
        )
    if n_ts_deferred:
        print(f'  ({n_ts_deferred} converged TS deferred — TS-specific adapter not yet implemented)')
    for label, err in failures:
        print(f'  failed: {label} — {err}')
    for label, kind, err in artifact_failures:
        print(f'  failed artifact: {label} ({kind}) — {err}')


_CALC_TYPE_TO_LOG_KEY = {
    'opt': 'opt_log',
    'freq': 'freq_log',
    'sp': 'sp_log',
}

# Companion mapping for input-deck paths, emitted by ``arc/output.py``
# alongside the log paths. Per-job, with per-job software → per-job
# filename, and only set when the deck file is on disk.
_CALC_TYPE_TO_INPUT_KEY = {
    'opt': 'opt_input',
    'freq': 'freq_input',
    'sp': 'sp_input',
}


def _implementable_kinds_from_config(tckdb_config):
    """Intersect user-configured kinds with ARC's IMPLEMENTED_ARTIFACT_KINDS.

    The config-parse step warns about valid-but-not-implemented kinds;
    this filter is the runtime side of the same gate, so the sweep
    silently skips them rather than calling the adapter (which would
    skip with a defensive log message anyway).
    """
    from arc.tckdb.config import IMPLEMENTED_ARTIFACT_KINDS
    return tuple(k for k in tckdb_config.artifacts.kinds if k in IMPLEMENTED_ARTIFACT_KINDS)


def _resolve_artifact_path(*, kind, calc_type, species_record, output_doc):
    """Resolve the local file path to upload for a (kind, calc_type) pair.

    Returns ``None`` if there's nothing to upload for this combination
    (e.g. unsupported calc type, file not on disk, engine unknown).

    For ``output_log``, the path is keyed off the species_record's
    log fields (``opt_log`` / ``freq_log`` / ``sp_log``).

    For ``input``, the input deck (``input.gjf``, ``ZMAT``, ``input.in``,
    etc.) is always written as a sibling of the output log, so we
    derive its name from ``arc.imports.settings['input_filenames']``
    keyed on the engine in ``output_doc['opt_level']['software']``.
    """
    log_key = _CALC_TYPE_TO_LOG_KEY.get(str(calc_type).lower())
    if log_key is None:
        return None
    log_path = species_record.get(log_key)
    if not log_path:
        return None
    if kind == 'output_log':
        return log_path
    if kind == 'input':
        # Prefer the path emitted directly by ``arc/output.py``: it's
        # per-job (so a Gaussian opt + Molpro sp run picks the right
        # deck per calc), and existence on disk has already been
        # verified at output-write time.
        input_field = _CALC_TYPE_TO_INPUT_KEY.get(str(calc_type).lower())
        if input_field:
            recorded = species_record.get(input_field)
            if recorded:
                return recorded
        # Back-compat: older output.yml files predating the
        # ``<calc>_input`` schema extension. Derive from the opt-level
        # software via settings['input_filenames']. Same logic as before
        # — kept so old runs can still upload input decks via the
        # primitive endpoint.
        from arc.imports import settings as _arc_settings
        opt_level = output_doc.get('opt_level') or {}
        engine = (opt_level.get('software') or '').lower() if isinstance(opt_level, dict) else ''
        input_filenames = _arc_settings.get('input_filenames', {})
        input_name = input_filenames.get(engine)
        if not input_name:
            return None
        return os.path.join(os.path.dirname(log_path), input_name)
    return None


def _sweep_artifacts_for_species(
    *,
    adapter,
    arc_object,
    output_doc,
    species_record,
    outcome,
    counts,
    failures,
    kinds,
):
    """For one converged species' conformer upload, push artifacts of each kind to each calc.

    Iterates the calc refs returned by the conformer upload (primary +
    additional) and, for each, iterates the configured kinds. Resolves
    the right local file path per (kind, calc_type) and dispatches to
    ``adapter.submit_artifacts_for_calculation``. Updates ``counts`` and
    ``failures`` in place.
    """
    label = species_record.get('label') or '<unlabeled>'
    refs = []
    if outcome.primary_calculation:
        refs.append(outcome.primary_calculation)
    refs.extend(outcome.additional_calculations or [])
    if not refs:
        # Older server response without calc refs — skip artifact upload
        # for this species rather than guess at IDs.
        return
    for ref in refs:
        calc_id = ref.get('calculation_id')
        calc_type = ref.get('type')
        if calc_id is None or calc_type is None:
            continue
        for kind in kinds:
            file_path = _resolve_artifact_path(
                kind=kind,
                calc_type=calc_type,
                species_record=species_record,
                output_doc=output_doc,
            )
            if file_path is None:
                counts['skipped'] = counts.get('skipped', 0) + 1
                continue
            try:
                art_outcome = adapter.submit_artifacts_for_calculation(
                    output_doc=output_doc,
                    species_record=species_record,
                    calculation_id=int(calc_id),
                    calculation_type=str(calc_type),
                    file_path=file_path,
                    kind=kind,
                )
            except Exception as exc:
                counts['failed'] = counts.get('failed', 0) + 1
                failures.append((label, kind, f'{type(exc).__name__}: {exc}'))
                continue
            if art_outcome is None:
                continue
            counts[art_outcome.status] = counts.get(art_outcome.status, 0) + 1
            if art_outcome.status == 'failed':
                failures.append((label, art_outcome.kind, art_outcome.error or 'unknown error'))


if __name__ == '__main__':
    main()
