"""End-of-run TCKDB upload sweep.

Reads ``<project>/output/output.yml`` and dispatches per
``TCKDBConfig.upload_mode``:

- ``conformer``: per-species ``/uploads/conformers`` POST + per-artifact
  POSTs to ``/calculations/{id}/artifacts``.
- ``computed_species``: one ``/uploads/computed-species`` bundle per
  species, with artifacts inlined under each calc.
- ``computed_reaction``: one ``/uploads/computed-reaction`` bundle per
  reaction. Species + TS + kinetics all ship in the reaction bundle;
  the per-species sweep is *not* run. Exception: for a *partial* reaction
  (TS missing/non-converged) whose bundle never live-POSTs, each
  individually-converged reactant/product species is still salvaged via
  the computed-species path so converged minima are not lost with a
  failed TS.
- ``computed_ts``: one ``/uploads/transition-states`` POST per converged
  transition state, with the reaction (reactants/products, family,
  reversibility) embedded from the TS's reaction record. Non-converged
  TSs are skipped, mirroring the species-sweep eligibility gate.

Lives in its own module so both the post-``execute()`` hook in
``ARC.py`` and the standalone CLI (``arc/tckdb/cli.py``) can call the
same code path. Functions take ``project_directory`` directly rather
than the live ARC object — output.yml is the contract.
"""

import copy
import os

from arc.common import get_logger, read_yaml_file
from arc.tckdb.config import (
    IMPLEMENTED_ARTIFACT_KINDS,
    UPLOAD_MODE_COMPUTED_REACTION,
    UPLOAD_MODE_COMPUTED_SPECIES,
    UPLOAD_MODE_COMPUTED_TS,
)


logger = get_logger()

# Canonical display handle for a transition state with no label (rare —
# ARC normally labels every TS). Kept as a single constant so log lines
# and the sweep summary never drift apart.
_UNLABELED_TS = '<unlabeled-ts>'


def run_upload_sweep(*, adapter, project_directory, tckdb_config):
    """Top-level dispatch: load output.yml, route per upload_mode, print summary.

    Returns ``None``. Side effects: writes payloads/sidecars under
    ``payload_dir``, optionally POSTs to TCKDB, and prints a one-line
    summary per upload kind.
    """
    output_path = os.path.join(project_directory, 'output', 'output.yml')
    if not os.path.exists(output_path):
        # Most common cause: the run was interrupted before
        # write_output_yml ran. Skip cleanly rather than scrape live
        # objects — the replay path expects output.yml as the contract.
        print(f'TCKDB upload skipped: {output_path} not found (run did not complete?)')
        return

    output_doc = read_yaml_file(path=output_path)

    if tckdb_config.upload_mode == UPLOAD_MODE_COMPUTED_REACTION:
        # Reaction mode is its own iteration shape: one POST per
        # reaction, species + TS + kinetics carried inline. Species
        # records reach TCKDB through whichever reaction(s) reference
        # them; standalone species uploads need a different mode.
        _run_reaction_sweep(
            adapter=adapter,
            output_doc=output_doc,
            tckdb_config=tckdb_config,
        )
        return

    if tckdb_config.upload_mode == UPLOAD_MODE_COMPUTED_TS:
        # TS mode is per-transition-state: one POST per converged TS to
        # /uploads/transition-states, reaction embedded from the TS's
        # reaction record. Species minima are out of scope here — use a
        # species mode (or computed_reaction) to upload those.
        _run_ts_sweep(
            adapter=adapter,
            output_doc=output_doc,
            tckdb_config=tckdb_config,
        )
        return

    _run_species_sweep(
        adapter=adapter, output_doc=output_doc, tckdb_config=tckdb_config,
    )


def _run_species_sweep(*, adapter, output_doc, tckdb_config):
    """Per-species iteration shared by ``conformer`` and ``computed_species`` modes."""
    species_records = list(output_doc.get('species') or [])
    # The species-only modes cover minima only. Converged transition
    # states are uploaded by the ``computed_ts`` mode
    # (``_run_ts_sweep`` -> /uploads/transition-states); computed_reaction
    # mode handles TSes inline. Neither is this sweep's concern.
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
    for label, err in failures:
        print(f'  failed: {label} — {err}')
    for label, kind, err in artifact_failures:
        print(f'  failed artifact: {label} ({kind}) — {err}')


def _run_reaction_sweep(*, adapter, output_doc, tckdb_config):
    """One ``/uploads/computed-reaction`` POST per ``output_doc['reactions']`` entry.

    Reactions skipped here:
        - no ``reactant_labels`` / ``product_labels``    → reaction
          builder raises ValueError; we count + log the failure.
        - ts_label points at a TS without a parsed xyz   → same.
        - reaction has no kinetics block                 → still
          uploaded (provenance value), with empty kinetics.
        - TS missing or non-converged with
          ``allow_partial_uploads=false`` → counted as ``skipped``,
          one log line per skip explains why.
        - TS missing or non-converged with
          ``allow_partial_uploads=true``  → ``ts_label`` and
          ``kinetics`` are stripped from a deepcopy of the record and
          submitted with ``is_partial=true``. The adapter writes a
          ``.partial`` sidecar and never live-POSTs in phase-1.

    For *either* partial case (regardless of ``allow_partial_uploads``),
    each reactant/product species that individually converged is
    salvaged via the computed-species path — the reaction bundle never
    live-POSTs when partial, so those converged minima would otherwise be
    dropped. This is independent of the reaction/kinetics provenance the
    flag governs; see ``_upload_partial_reaction_species``.
    """
    reaction_records = list(output_doc.get('reactions') or [])
    n_attempted = len(reaction_records)
    counts = {'uploaded': 0, 'skipped': 0, 'failed': 0}
    failures: list[tuple[str, str]] = []
    ts_index = {
        str(r.get('label') or r.get('original_label')): r
        for r in (output_doc.get('transition_states') or [])
        if isinstance(r, dict) and (r.get('label') or r.get('original_label'))
    }
    species_index = {
        str(r.get('label') or r.get('original_label')): r
        for r in (output_doc.get('species') or [])
        if isinstance(r, dict) and (r.get('label') or r.get('original_label'))
    }
    n_partial_written = 0
    n_partial_disabled_skip = 0
    # Converged reactant/product species salvaged from partial reactions
    # (their reaction bundle never live-POSTs). Deduped by label across
    # the whole sweep so a species shared by several failed reactions is
    # uploaded at most once.
    species_seen: set[str] = set()
    species_counts = {'uploaded': 0, 'skipped': 0, 'failed': 0}
    species_failures: list[tuple[str, str]] = []

    for record in reaction_records:
        label = record.get('label') or '<unlabeled-reaction>'
        ts_label = record.get('ts_label')
        ts_record = ts_index.get(str(ts_label)) if ts_label else None
        ts_converged = bool(ts_record and ts_record.get('converged'))
        # We treat both "ts_label set but TS missing/non-converged" and
        # "no ts_label at all" as the partial case. The latter is rare
        # (ARC normally emits a ts_label even on failure), but a
        # reaction record with reactants/products and no TS is exactly
        # the partial-shape we want to allow under the flag.
        is_partial = not ts_converged

        if is_partial:
            # The reaction bundle for a partial reaction never live-POSTs
            # (it is either skipped outright below or written sidecar-only
            # with is_partial=true). Its converged reactant/product species
            # would then never reach TCKDB. Upload each species that
            # *individually* converged via the computed-species path,
            # gated solely on that species' own convergence — independent
            # of allow_partial_uploads, which governs only the
            # reaction/kinetics provenance record, not the species minima.
            _upload_partial_reaction_species(
                adapter=adapter,
                output_doc=output_doc,
                reaction_record=record,
                species_index=species_index,
                seen=species_seen,
                counts=species_counts,
                failures=species_failures,
            )

        if is_partial and not tckdb_config.allow_partial_uploads:
            n_partial_disabled_skip += 1
            counts['skipped'] += 1
            logger.info(
                'TCKDB reaction %r skipped: TS %r is missing or non-converged '
                'and tckdb.allow_partial_uploads is false.',
                label, ts_label,
            )
            continue

        submit_record = record
        if is_partial:
            # Strip TS and kinetics from a deepcopy: the producer must
            # not silently upload kinetics fitted against a TS that did
            # not validate. Original output_doc / reaction_record are
            # left untouched so downstream consumers see real data.
            submit_record = copy.deepcopy(record)
            submit_record['ts_label'] = None
            submit_record['kinetics'] = None
            logger.info(
                'TCKDB reaction %r: TS %r missing/non-converged — writing '
                'partial sidecar (ts_label and kinetics stripped).',
                label, ts_label,
            )

        try:
            outcome = adapter.submit_computed_reaction_from_output(
                output_doc=output_doc,
                reaction_record=submit_record,
                is_partial=is_partial,
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
        if is_partial and outcome.status == 'skipped':
            n_partial_written += 1

    print(f'TCKDB v0 (computed-reaction bundle, {n_attempted} reactions):')
    print(f'  uploaded: {counts["uploaded"]}  skipped: {counts["skipped"]}  failed: {counts["failed"]}')
    if n_partial_written:
        print(
            f'  partial sidecars written (TS missing/non-converged, not POSTed): '
            f'{n_partial_written}'
        )
    if n_partial_disabled_skip:
        print(
            f'  partial reactions skipped (allow_partial_uploads=false): '
            f'{n_partial_disabled_skip}'
        )
    if any(species_counts.values()) or species_failures:
        print(
            f'  species salvaged from partial reactions: '
            f'uploaded {species_counts["uploaded"]}  '
            f'skipped {species_counts["skipped"]}  '
            f'failed {species_counts["failed"]}'
        )
    for slabel, serr in species_failures:
        print(f'  failed species: {slabel} — {serr}')
    if not reaction_records:
        # Common cause: ARC ran species jobs but kinetics fitting
        # didn't produce any reactions in output.yml. Surface this so
        # the user knows why nothing was uploaded.
        print('  (no reactions in output.yml — kinetics fit may not have run)')
    for label, err in failures:
        print(f'  failed: {label} — {err}')


def _run_ts_sweep(*, adapter, output_doc, tckdb_config):
    """One ``/uploads/transition-states`` POST per converged transition state.

    Iterates ``output_doc['transition_states']`` and uploads each TS whose
    own ``converged`` flag is true — the same eligibility gate
    ``_run_species_sweep`` uses for minima. Non-converged TSs are skipped.

    Each TS needs its reaction (reactants/products, family, reversibility)
    to fill the embedded ``reaction`` of the standalone payload; that comes
    from the ``output_doc['reactions']`` entry whose ``ts_label`` points at
    this TS. A converged TS with no such reaction can't produce a valid
    request (the schema requires at least one reactant and one product), so
    it is skipped with a logged reason rather than failed. Those semantic
    skips are counted separately from the adapter's own ``skipped``
    outcomes (a dry run with ``upload=false``) so the summary is honest
    about *why* nothing was sent.

    Routes through ``adapter.submit_computed_ts_from_output``, so
    readiness/retry, strict-mode, idempotency replay, and the recovery-log
    behavior apply automatically — same submit/_record_failure path as the
    other modes.
    """
    ts_records = list(output_doc.get('transition_states') or [])
    reaction_records = list(output_doc.get('reactions') or [])
    # ts_label -> reaction record. First reaction wins if two reference the
    # same TS (rare; ARC emits one reaction per TS in practice).
    reaction_by_ts_label: dict[str, dict] = {}
    for rxn in reaction_records:
        if not isinstance(rxn, dict):
            continue
        ts_label = rxn.get('ts_label')
        if ts_label:
            reaction_by_ts_label.setdefault(str(ts_label), rxn)

    # ``counts`` holds adapter outcomes only (uploaded / dry-run-skipped /
    # failed). Semantic skips before the adapter is even called are tracked
    # apart so the summary can name them distinctly.
    counts = {'uploaded': 0, 'skipped': 0, 'failed': 0}
    failures = []
    n_converged = 0
    n_no_reaction = 0
    for record in ts_records:
        raw_label = record.get('label') or record.get('original_label')
        label = raw_label or _UNLABELED_TS
        if not record.get('converged'):
            # Same gate as the species sweep: only converged stationary
            # points are uploadable. Non-converged TSs are skipped, never
            # uploaded.
            continue
        n_converged += 1
        reaction_record = (
            reaction_by_ts_label.get(str(raw_label)) if raw_label else None
        )
        if reaction_record is None:
            # No reaction references this TS (or the TS is unlabeled and
            # can't be keyed), so we can't build the embedded
            # reactants/products the endpoint requires. Skip loudly — kept
            # out of the adapter ``skipped`` bucket so a dry run and a
            # no-reaction skip are never conflated.
            n_no_reaction += 1
            logger.info(
                'TCKDB TS %r skipped: no reaction in output.yml references it '
                '(need reactants/products for the embedded reaction).',
                label,
            )
            continue
        try:
            outcome = adapter.submit_computed_ts_from_output(
                output_doc=output_doc,
                ts_record=record,
                reaction_record=reaction_record,
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

    print(f'TCKDB v0 (transition-state, {n_converged} converged TS):')
    print(f'  uploaded: {counts["uploaded"]}  skipped: {counts["skipped"]}  failed: {counts["failed"]}')
    if n_no_reaction:
        print(
            f'  skipped (no reaction references the TS): {n_no_reaction}'
        )
    if not ts_records:
        print('  (no transition states in output.yml)')
    for label, err in failures:
        print(f'  failed: {label} — {err}')


def _upload_partial_reaction_species(
    *,
    adapter,
    output_doc,
    reaction_record,
    species_index,
    seen,
    counts,
    failures,
):
    """Upload the individually-converged species of a partial reaction.

    A reaction whose TS is missing/non-converged is uploaded (if at all)
    only as a phase-1 ``.partial`` sidecar that never live-POSTs, so its
    reactant/product species — which may each have converged and met all
    requirements on their own — would otherwise be dropped. This salvages
    them through the existing computed-species path (SPECIES payload
    shape, ``/uploads/computed-species``), gated purely on each species'
    own ``converged`` flag, exactly as the standalone species sweep gates
    (see ``_run_species_sweep``).

    Kinetics and the TS are deliberately not touched here — only the
    minima are salvaged. Idempotency: ``submit_computed_species_from_output``
    builds a content-hashed idempotency key, so a re-POST replays
    server-side; the ``seen`` label set additionally suppresses redundant
    network calls for a species referenced by more than one failed
    reaction (or already handled earlier in this sweep). Updates
    ``counts`` and ``failures`` in place.
    """
    labels = list(reaction_record.get('reactant_labels') or [])
    labels += list(reaction_record.get('product_labels') or [])
    for label in labels:
        if not label:
            continue
        key = str(label)
        if key in seen:
            continue
        record = species_index.get(key)
        if record is None:
            # Reaction referenced a species absent from output_doc.species.
            # The reaction builder itself raises on this, so nothing to do
            # here — leave it for that path to report.
            continue
        if not record.get('converged'):
            # Same eligibility gate as the standalone species sweep: only
            # converged minima are uploadable. Non-converged species are
            # skipped, never uploaded.
            continue
        # Mark as handled before attempting: a build/upload failure for
        # this species should be reported once, not retried for every
        # other reaction that references it.
        seen.add(key)
        try:
            outcome = adapter.submit_computed_species_from_output(
                output_doc=output_doc, species_record=record,
            )
        except Exception as exc:
            counts['failed'] = counts.get('failed', 0) + 1
            failures.append((key, f'{type(exc).__name__}: {exc}'))
            continue
        if outcome is None:
            continue
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
        if outcome.status == 'failed':
            failures.append((key, outcome.error or 'unknown error'))


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
    output_doc,
    species_record,
    outcome,
    counts,
    failures,
    kinds,
):
    """For one converged species' conformer upload, push artifacts to each calc.

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
        artifact_items = []
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
            artifact_items.append((kind, file_path))
        if not artifact_items:
            continue
        try:
            art_outcomes = adapter.submit_artifact_batch_for_calculation(
                output_doc=output_doc,
                species_record=species_record,
                calculation_id=int(calc_id),
                calculation_type=str(calc_type),
                artifacts=artifact_items,
            )
        except Exception as exc:
            counts['failed'] = counts.get('failed', 0) + len(artifact_items)
            for kind, _ in artifact_items:
                failures.append((label, kind, f'{type(exc).__name__}: {exc}'))
            continue
        if art_outcomes is None:
            continue
        for art_outcome in art_outcomes:
            counts[art_outcome.status] = counts.get(art_outcome.status, 0) + 1
            if art_outcome.status == 'failed':
                failures.append((label, art_outcome.kind, art_outcome.error or 'unknown error'))


__all__ = ['run_upload_sweep']
