"""Adapter that turns an ARC ``output.yml`` record into a TCKDB conformer upload.

The adapter is the only ARC module that knows the shape of a TCKDB
upload. It builds a JSON payload matching ``ConformerUploadRequest``,
writes it to disk, and (optionally) hands it to ``tckdb-client``.

Source of truth for upload data is ``<project>/output/output.yml`` —
``arc/output.py`` was designed for this consumer (see its module
docstring). The adapter therefore takes two dicts: the full output
document (for top-level levels-of-theory, ARC version, etc.) and one
species record from ``output_doc['species']`` or
``output_doc['transition_states']``. This keeps the adapter decoupled
from ARC's live object model and makes a separate replay path (read
output.yml later, post payloads, no ARC needed) trivial.

Three guarantees:

1. If the adapter is disabled or no config is provided, it is a no-op.
2. The payload is on disk *before* any network call. Replay tooling
   only needs ``payload_file + endpoint + idempotency_key``.
3. By default, an upload failure is logged + recorded in the sidecar
   but does not raise. ``strict=True`` flips that.
"""

import base64
import hashlib
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tckdb_client import TCKDBClient

from arc.tckdb.config import IMPLEMENTED_ARTIFACT_KINDS, TCKDBConfig
from arc.tckdb.idempotency import (
    ArtifactIdempotencyInputs,
    IdempotencyInputs,
    build_artifact_idempotency_key,
    build_idempotency_key,
)
from arc.tckdb.payload_writer import (
    ArtifactSidecarMetadata,
    PayloadWriter,
    SidecarMetadata,
    WrittenArtifact,
    WrittenPayload,
    _utcnow_iso,
)


logger = logging.getLogger("arc")

CONFORMER_UPLOAD_ENDPOINT = "/uploads/conformers"
PAYLOAD_KIND = "conformer_calculation"
ARTIFACTS_ENDPOINT_TEMPLATE = "/calculations/{calculation_id}/artifacts"

# Computed-species bundle endpoint. One self-contained payload that
# carries species_entry + conformers + calcs + artifacts + thermo, with
# all cross-references expressed as local string keys (no DB ids).
COMPUTED_SPECIES_ENDPOINT = "/uploads/computed-species"
COMPUTED_SPECIES_KIND = "computed_species"

# Local calculation-key namespace within a computed-species bundle.
# These keys are referenced from `depends_on.parent_calculation_key` and
# `thermo.source_calculations[].calculation_key`. They have no relation
# to TCKDB-assigned calculation_ids — the bundle endpoint mints those
# server-side and returns them in the response.
_CALC_KEY_OPT = "opt"
_CALC_KEY_FREQ = "freq"
_CALC_KEY_SP = "sp"

# Per-calc record-field map for output_log artifacts. Same convention as
# `_LOG_FIELD_BY_CALC_KEY` in the existing artifact path: the species
# record carries `<job>_log` paths that come straight from
# `arc/output.py::_spc_to_dict`.
_LOG_FIELD_BY_CALC_KEY = {
    _CALC_KEY_OPT: "opt_log",
    _CALC_KEY_FREQ: "freq_log",
    _CALC_KEY_SP: "sp_log",
}

# Same shape, for input-deck paths. `arc/output.py` emits these per-job
# (with per-job software → per-job filename), only when the deck file
# actually exists on disk; null when the deck wasn't kept (archived runs).
_INPUT_FIELD_BY_CALC_KEY = {
    _CALC_KEY_OPT: "opt_input",
    _CALC_KEY_FREQ: "freq_input",
    _CALC_KEY_SP: "sp_input",
}

# (artifact_kind, record-field-map) pairs that the inline-artifact
# helper iterates per calc. Keeping the mapping data-driven so adding
# checkpoints (or any future kind) is one tuple, not a code branch.
_INLINE_ARTIFACT_SOURCES: tuple[tuple[str, dict[str, str]], ...] = (
    ("output_log", _LOG_FIELD_BY_CALC_KEY),
    ("input", _INPUT_FIELD_BY_CALC_KEY),
)


@dataclass(frozen=True)
class UploadOutcome:
    """Result of one adapter invocation; mirrors the sidecar status.

    ``primary_calculation`` and ``additional_calculations`` are populated
    on successful conformer uploads (status == ``uploaded``) and carry
    the server's :class:`CalculationUploadRef` shape — i.e. dicts with
    ``calculation_id``, ``type``, and (for additional calcs) ``request_index``.
    They are ``None`` / empty otherwise.
    """

    status: str  # pending | uploaded | failed | skipped
    payload_path: Path
    sidecar_path: Path
    idempotency_key: str
    error: str | None = None
    response: Any = None
    primary_calculation: dict[str, Any] | None = None
    additional_calculations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ArtifactUploadOutcome:
    """Result of one artifact upload attempt."""

    status: str  # uploaded | failed | skipped
    sidecar_path: Path | None
    idempotency_key: str | None
    calculation_id: int
    kind: str
    error: str | None = None
    response: Any = None
    skip_reason: str | None = None


class TCKDBAdapter:
    """Build, write, and optionally upload one conformer/calculation payload.

    The adapter holds the config and a payload writer. ``client_factory``
    is overridable so tests can inject a mocked ``TCKDBClient`` without
    touching the network.
    """

    def __init__(
        self,
        config: TCKDBConfig,
        *,
        project_directory: str | Path | None = None,
        client_factory=None,
    ):
        self._config = config
        self._project_directory = (
            Path(project_directory) if project_directory is not None else None
        )
        # Resolve payload_dir against the project directory if it's relative,
        # so payloads land under the active ARC project rather than CWD.
        payload_root = Path(config.payload_dir)
        if not payload_root.is_absolute() and project_directory is not None:
            payload_root = Path(project_directory) / payload_root
        self._writer = PayloadWriter(payload_root)
        self._client_factory = client_factory

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def submit_from_output(
        self,
        *,
        output_doc: Mapping[str, Any],
        species_record: Mapping[str, Any],
        conformer_index: int = 0,
        extra_label: str | None = None,
    ) -> UploadOutcome | None:
        """Build, write, and (if configured) upload one conformer payload.

        ``output_doc`` is the full parsed ``output.yml``. ``species_record``
        is one entry from ``output_doc['species']`` or
        ``output_doc['transition_states']``.

        Returns ``None`` if the adapter is disabled (so callers can write
        ``adapter.submit_from_output(...)`` without an enabled-check).
        """
        if not self._config.enabled:
            return None

        payload = self._build_payload(
            output_doc=output_doc,
            species_record=species_record,
        )

        species_label = species_record.get("label") or "unlabeled"
        conformer_label = extra_label or f"conf{conformer_index}"
        project_label = self._config.project_label or output_doc.get("project")
        idempotency_inputs = IdempotencyInputs.from_payload(
            project_label=project_label,
            species_label=species_label,
            conformer_label=conformer_label,
            payload_kind=PAYLOAD_KIND,
            payload=payload,
        )
        idempotency_key = build_idempotency_key(idempotency_inputs)

        written = self._writer.write(
            label=f"{species_label}.{conformer_label}",
            payload=payload,
            endpoint=CONFORMER_UPLOAD_ENDPOINT,
            idempotency_key=idempotency_key,
            payload_kind=PAYLOAD_KIND,
            base_url=self._config.base_url,
        )
        logger.info(
            "TCKDB payload written: %s (key=%s)",
            written.payload_path,
            idempotency_key,
        )

        if not self._config.upload:
            return self._finalize_skipped(written)

        return self._upload(written, payload)

    def submit_artifacts_for_calculation(
        self,
        *,
        output_doc: Mapping[str, Any],
        species_record: Mapping[str, Any],
        calculation_id: int,
        calculation_type: str,
        file_path: str | Path,
        kind: str = "output_log",
    ) -> ArtifactUploadOutcome | None:
        """Upload one local file as an artifact attached to a TCKDB calculation.

        ``output_doc`` and ``species_record`` mirror :meth:`submit_from_output`.
        ``file_path`` may be absolute or relative — relative paths are
        resolved against ``project_directory`` (passed at construction).
        ``kind`` selects the TCKDB ArtifactKind; defaults to
        ``"output_log"`` for back-compat with v1 callers.

        Returns ``None`` only if the adapter itself is disabled. Otherwise
        returns an :class:`ArtifactUploadOutcome` whose ``status`` is one
        of ``uploaded`` / ``failed`` / ``skipped``. Skip reasons:
        artifact upload disabled, kind not in config.kinds, kind not yet
        implemented in ARC, file path missing, or file exceeds
        ``max_size_mb``.

        ``calculation_type`` is recorded in the sidecar but does not feed
        the URL — the endpoint takes the calc id directly.
        """
        if not self._config.enabled:
            return None
        species_label = species_record.get("label") or "unlabeled"
        artifact_cfg = self._config.artifacts

        if not artifact_cfg.upload:
            return _skip(calculation_id, kind, "artifacts.upload is False")
        if kind not in artifact_cfg.kinds:
            return _skip(calculation_id, kind, f"kind {kind!r} not in config.kinds")
        # Defensive check: config-parse warns about not-yet-implemented
        # kinds but doesn't reject them. If a user lists e.g. 'checkpoint'
        # and somehow a caller routes it here, skip cleanly rather than
        # uploading bytes intended for a different code path.
        if kind not in IMPLEMENTED_ARTIFACT_KINDS:
            return _skip(
                calculation_id, kind,
                f"kind {kind!r} is server-accepted but ARC has no upload path yet",
            )

        resolved = self._resolve_local_path(file_path)
        if resolved is None or not resolved.is_file():
            return _skip(
                calculation_id, kind, f"file missing: {file_path!r}"
            )

        size_bytes = resolved.stat().st_size
        max_bytes = artifact_cfg.max_size_mb * 1024 * 1024
        if size_bytes > max_bytes:
            return _skip(
                calculation_id,
                kind,
                f"file {resolved.name} is {size_bytes} bytes "
                f"(>{artifact_cfg.max_size_mb} MB cap)",
            )

        with resolved.open("rb") as fh:
            content = fh.read()
        sha256 = hashlib.sha256(content).hexdigest()
        content_b64 = base64.b64encode(content).decode("ascii")

        project_label = self._config.project_label or output_doc.get("project")
        idempotency_inputs = ArtifactIdempotencyInputs(
            project_label=project_label,
            species_label=species_label,
            calculation_id=calculation_id,
            artifact_kind=kind,
            artifact_sha256=sha256,
        )
        idempotency_key = build_artifact_idempotency_key(idempotency_inputs)
        endpoint = ARTIFACTS_ENDPOINT_TEMPLATE.format(calculation_id=calculation_id)

        written_artifact = self._writer.write_artifact_sidecar(
            species_label=species_label,
            calculation_id=calculation_id,
            kind=kind,
            filename=resolved.name,
            sha256=sha256,
            bytes_=size_bytes,
            endpoint=endpoint,
            idempotency_key=idempotency_key,
            source_path=str(resolved),
            base_url=self._config.base_url,
        )

        request_body = {
            "artifacts": [
                {
                    "kind": kind,
                    "filename": resolved.name,
                    "content_base64": content_b64,
                    "sha256": sha256,
                    "bytes": size_bytes,
                }
            ]
        }
        return self._upload_artifact(
            written=written_artifact,
            request_body=request_body,
            endpoint=endpoint,
        )

    # ------------------------------------------------------------------
    # Computed-species bundle path (POST /uploads/computed-species)
    # ------------------------------------------------------------------

    def submit_computed_species_from_output(
        self,
        *,
        output_doc: Mapping[str, Any],
        species_record: Mapping[str, Any],
        conformer_index: int = 0,
        extra_label: str | None = None,
    ) -> UploadOutcome | None:
        """Build, write, and (if configured) upload one computed-species bundle.

        Bundles species_entry + conformer geometry + opt/freq/sp + thermo
        + (optionally) inline artifacts into a single
        ``ComputedSpeciesUploadRequest`` and POSTs to
        ``/uploads/computed-species``. Returns ``None`` if the adapter is
        disabled, mirroring :meth:`submit_from_output`.

        Build failures (e.g. missing opt level, missing xyz) raise; the
        caller in scheduler/processor is responsible for wrapping the
        per-species call in a try/except so one bad species doesn't take
        down the rest of the run — same shape as the conformer path.
        """
        if not self._config.enabled:
            return None

        species_label = species_record.get("label") or "unlabeled"
        conformer_label = extra_label or f"conf{conformer_index}"
        project_label = self._config.project_label or output_doc.get("project")

        payload = self._build_computed_species_payload(
            output_doc=output_doc,
            species_record=species_record,
            conformer_key=conformer_label,
        )

        idempotency_inputs = IdempotencyInputs.from_payload(
            project_label=project_label,
            species_label=species_label,
            conformer_label=conformer_label,
            payload_kind=COMPUTED_SPECIES_KIND,
            payload=payload,
        )
        idempotency_key = build_idempotency_key(idempotency_inputs)

        written = self._writer.write(
            label=f"{species_label}.{conformer_label}",
            payload=payload,
            endpoint=COMPUTED_SPECIES_ENDPOINT,
            idempotency_key=idempotency_key,
            payload_kind=COMPUTED_SPECIES_KIND,
            base_url=self._config.base_url,
            subdir=PayloadWriter.COMPUTED_SPECIES_SUBDIR,
        )
        logger.info(
            "TCKDB computed-species payload written: %s (key=%s)",
            written.payload_path,
            idempotency_key,
        )

        if not self._config.upload:
            return self._finalize_skipped(written)

        return self._upload(written, payload, endpoint=COMPUTED_SPECIES_ENDPOINT)

    def _build_computed_species_payload(
        self,
        *,
        output_doc: Mapping[str, Any],
        species_record: Mapping[str, Any],
        conformer_key: str,
    ) -> dict[str, Any]:
        """Compose one ComputedSpeciesUploadRequest dict.

        Reuses the existing per-calc and species-entry shapers; the only
        bundle-specific surface is the conformer wrapper, dependency
        edges (declared by local calc keys), inline artifacts, and the
        optional thermo block.
        """
        included_keys, conformer_block = self._build_conformer_block(
            output_doc=output_doc,
            species_record=species_record,
            conformer_key=conformer_key,
        )

        bundle: dict[str, Any] = {
            "species_entry": self._species_entry_payload(species_record),
            "conformers": [conformer_block],
        }

        thermo_block = _build_thermo_block(
            species_record.get("thermo"),
            included_calc_keys=included_keys,
        )
        if thermo_block is not None:
            bundle["thermo"] = thermo_block

        # Workflow-tool release at bundle level (in addition to per-calc):
        # mirrors what the conformer adapter records and lets the server
        # tag the species_entry with the producer.
        arc_version = output_doc.get("arc_version")
        arc_git_commit = output_doc.get("arc_git_commit")
        if arc_version or arc_git_commit:
            wt: dict[str, Any] = {"name": "ARC"}
            if arc_version:
                wt["version"] = str(arc_version)
            if arc_git_commit:
                wt["git_commit"] = str(arc_git_commit)
            bundle["workflow_tool_release"] = wt

        return bundle

    def _build_conformer_block(
        self,
        *,
        output_doc: Mapping[str, Any],
        species_record: Mapping[str, Any],
        conformer_key: str,
    ) -> tuple[list[str], dict[str, Any]]:
        """Build one ConformerInBundle and return (included_calc_keys, block).

        The keys list is what's actually present in the bundle's calc
        namespace, used to drive thermo's source_calculations links.
        """
        primary_calc = self._build_calc_in_bundle(
            output_doc=output_doc,
            species_record=species_record,
            calc_key=_CALC_KEY_OPT,
            calc_type="opt",
            level_kind="opt",
            ess_job_key="opt",
            result_field="opt_result",
            result_payload=_opt_result_payload(species_record),
            depends_on=None,
            tckdb_origin=None,
        )

        included: list[str] = [_CALC_KEY_OPT]
        additional: list[dict[str, Any]] = []

        freq_result = _freq_result_payload(species_record)
        if freq_result is not None:
            try:
                additional.append(self._build_calc_in_bundle(
                    output_doc=output_doc,
                    species_record=species_record,
                    calc_key=_CALC_KEY_FREQ,
                    calc_type="freq",
                    level_kind="freq",
                    ess_job_key="freq",
                    result_field="freq_result",
                    result_payload=freq_result,
                    depends_on=[{"parent_calculation_key": _CALC_KEY_OPT, "role": "freq_on"}],
                    tckdb_origin=None,
                ))
                included.append(_CALC_KEY_FREQ)
            except ValueError as exc:
                logger.warning(
                    "TCKDB computed-species: freq calculation skipped for label=%s: %s",
                    species_record.get("label"), exc,
                )

        sp_result = _sp_result_payload(species_record)
        if sp_result is not None:
            try:
                additional.append(self._build_calc_in_bundle(
                    output_doc=output_doc,
                    species_record=species_record,
                    calc_key=_CALC_KEY_SP,
                    calc_type="sp",
                    level_kind="sp",
                    ess_job_key="sp",
                    result_field="sp_result",
                    result_payload=sp_result,
                    depends_on=[{"parent_calculation_key": _CALC_KEY_OPT, "role": "single_point_on"}],
                    tckdb_origin=(
                        _reused_origin("opt") if _sp_is_reused_from_opt(output_doc) else None
                    ),
                ))
                included.append(_CALC_KEY_SP)
            except ValueError as exc:
                logger.warning(
                    "TCKDB computed-species: sp calculation skipped for label=%s: %s",
                    species_record.get("label"), exc,
                )

        block: dict[str, Any] = {
            "key": conformer_key,
            "geometry": {"xyz_text": _require_xyz_text(species_record)},
            "primary_calculation": primary_calc,
            "additional_calculations": additional,
        }
        label = species_record.get("label")
        if label:
            block["label"] = str(label)[:64]
        return included, block

    def _build_calc_in_bundle(
        self,
        *,
        output_doc: Mapping[str, Any],
        species_record: Mapping[str, Any],
        calc_key: str,
        calc_type: str,
        level_kind: str,
        ess_job_key: str,
        result_field: str,
        result_payload: Mapping[str, Any] | None,
        depends_on: list[Mapping[str, Any]] | None,
        tckdb_origin: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Build one CalculationInBundle dict.

        Reuses :meth:`_calculation_payload` for the level/software/result
        plumbing, then layers on the bundle-specific fields: ``key``,
        ``depends_on``, and inline ``artifacts``.
        """
        level = _resolve_level(output_doc, level_kind)
        calc = self._calculation_payload(
            output_doc, species_record,
            calc_type=calc_type,
            level=level,
            ess_job_key=ess_job_key,
            result_field=result_field,
            result_payload=result_payload,
            tckdb_origin=tckdb_origin,
        )
        calc["key"] = calc_key
        if depends_on:
            calc["depends_on"] = [dict(d) for d in depends_on]
        artifacts = self._inline_artifacts_for_calc(species_record, calc_key=calc_key)
        # Schema defaults `artifacts: []`. Emit explicitly only when we have
        # bytes to send (or when artifact upload is enabled and we want to
        # signal "no log available" with an empty list); omit otherwise.
        if artifacts:
            calc["artifacts"] = artifacts
        return calc

    def _inline_artifacts_for_calc(
        self,
        species_record: Mapping[str, Any],
        *,
        calc_key: str,
    ) -> list[dict[str, Any]]:
        """Return the inline artifact list for one calc within a bundle.

        Iterates ``_INLINE_ARTIFACT_SOURCES`` (currently ``output_log``
        and ``input``) and emits one ArtifactIn dict per kind whose
        record path resolves to a real file on disk. Each kind is
        independently gated on ``config.artifacts.kinds``, so a user
        can opt into logs but not decks (or vice versa).

        Skip rules per (calc_key, kind):
            - artifacts globally disabled              → entire list = []
            - kind not in config.artifacts.kinds       → that kind only
            - record-field path missing / null         → that kind only
            - resolved file not on disk                → that kind only
            - file > artifacts.max_size_mb             → that kind only (warn)
        """
        artifact_cfg = self._config.artifacts
        if not artifact_cfg.upload:
            return []
        artifacts: list[dict[str, Any]] = []
        for kind, field_map in _INLINE_ARTIFACT_SOURCES:
            if kind not in artifact_cfg.kinds:
                continue
            artifact = self._read_inline_artifact(
                species_record,
                calc_key=calc_key,
                kind=kind,
                record_field=field_map.get(calc_key),
            )
            if artifact is not None:
                artifacts.append(artifact)
        return artifacts

    def _read_inline_artifact(
        self,
        species_record: Mapping[str, Any],
        *,
        calc_key: str,
        kind: str,
        record_field: str | None,
    ) -> dict[str, Any] | None:
        """Resolve, read, hash, and base64-encode one artifact for the bundle.

        Returns ``None`` (with a debug or warning log) on any of:
        unknown calc_key, missing/null record path, file not on disk, or
        file exceeding ``max_size_mb``. Otherwise returns the
        ``ArtifactIn``-shaped dict ready to drop into ``calc.artifacts``.
        """
        if record_field is None:
            return None
        path_value = species_record.get(record_field)
        if not path_value:
            return None
        resolved = self._resolve_local_path(path_value)
        if resolved is None or not resolved.is_file():
            logger.debug(
                "TCKDB computed-species: %s %s artifact missing on disk for %s (path=%s)",
                calc_key, kind, species_record.get("label"), path_value,
            )
            return None
        size_bytes = resolved.stat().st_size
        max_bytes = self._config.artifacts.max_size_mb * 1024 * 1024
        if size_bytes > max_bytes:
            logger.warning(
                "TCKDB computed-species: %s %s %s skipped (%s bytes > %s MB cap)",
                calc_key, kind, resolved.name, size_bytes,
                self._config.artifacts.max_size_mb,
            )
            return None
        with resolved.open("rb") as fh:
            content = fh.read()
        return {
            "kind": kind,
            "filename": resolved.name,
            "content_base64": base64.b64encode(content).decode("ascii"),
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": size_bytes,
        }

    def _resolve_local_path(self, file_path: str | Path) -> Path | None:
        """Resolve a local file path against project_directory if it's relative."""
        if file_path is None:
            return None
        path = Path(file_path)
        if path.is_absolute():
            return path
        if self._project_directory is not None:
            return Path(self._project_directory) / path
        return path.resolve()

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    @staticmethod
    def _species_entry_payload(record: Mapping[str, Any]) -> dict[str, Any]:
        smiles = record.get("smiles")
        if not smiles:
            raise ValueError(
                f"output.yml record for label={record.get('label')!r} has no smiles; "
                "TCKDB upload requires a SMILES on the species_entry."
            )
        is_ts = bool(record.get("is_ts"))
        return {
            "molecule_kind": "molecule",
            "smiles": str(smiles),
            "charge": int(record.get("charge", 0) or 0),
            "multiplicity": int(record.get("multiplicity", 1) or 1),
            "species_entry_kind": "transition_state" if is_ts else "minimum",
        }

    def _build_payload(
        self,
        *,
        output_doc: Mapping[str, Any],
        species_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        species_entry = self._species_entry_payload(species_record)
        geometry_payload = {"xyz_text": _require_xyz_text(species_record)}
        primary, additional = self._build_calculations(output_doc, species_record)

        payload: dict[str, Any] = {
            "species_entry": species_entry,
            "geometry": geometry_payload,
            "calculation": primary,
            "scientific_origin": "computed",
        }
        if additional:
            payload["additional_calculations"] = additional
        label = species_record.get("label")
        if label:
            payload["label"] = str(label)[:64]
        return payload

    @classmethod
    def _build_calculations(
        cls,
        output_doc: Mapping[str, Any],
        record: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Return (primary opt calculation, [freq, sp] additional calculations).

        Additional calculations are skipped (with a warning) when their
        result fields are absent or malformed, or when no level of theory
        is available. Skipping an optional calc never fails the upload.
        """
        primary = cls._calculation_payload(
            output_doc,
            record,
            calc_type="opt",
            level=_resolve_level(output_doc, "opt"),
            ess_job_key="opt",
            result_field="opt_result",
            result_payload=_opt_result_payload(record),
        )

        additional: list[dict[str, Any]] = []
        freq_result = _freq_result_payload(record)
        if freq_result is not None:
            freq_level = _resolve_level(output_doc, "freq")
            try:
                additional.append(
                    cls._calculation_payload(
                        output_doc,
                        record,
                        calc_type="freq",
                        level=freq_level,
                        ess_job_key="freq",
                        result_field="freq_result",
                        result_payload=freq_result,
                    )
                )
            except ValueError as exc:
                logger.warning(
                    "TCKDB freq additional calculation skipped for label=%s: %s",
                    record.get("label"), exc,
                )

        sp_result = _sp_result_payload(record)
        if sp_result is not None:
            sp_level = _resolve_level(output_doc, "sp")
            sp_origin = _reused_origin("opt") if _sp_is_reused_from_opt(output_doc) else None
            try:
                additional.append(
                    cls._calculation_payload(
                        output_doc,
                        record,
                        calc_type="sp",
                        level=sp_level,
                        ess_job_key="sp",
                        result_field="sp_result",
                        result_payload=sp_result,
                        tckdb_origin=sp_origin,
                    )
                )
            except ValueError as exc:
                logger.warning(
                    "TCKDB sp additional calculation skipped for label=%s: %s",
                    record.get("label"), exc,
                )

        return primary, additional

    @staticmethod
    def _calculation_payload(
        output_doc: Mapping[str, Any],
        record: Mapping[str, Any],
        *,
        calc_type: str,
        level: Mapping[str, Any] | None,
        ess_job_key: str,
        result_field: str | None = None,
        result_payload: Mapping[str, Any] | None = None,
        tckdb_origin: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(level, Mapping):
            raise ValueError(
                f"no level of theory available for {calc_type} calculation; "
                "cannot build TCKDB calculation payload."
            )
        method = level.get("method")
        if not method:
            raise ValueError(
                f"level of theory for {calc_type} is missing method; "
                "cannot build TCKDB calculation payload."
            )
        software_name = level.get("software")
        if not software_name:
            raise ValueError(
                f"level of theory for {calc_type} is missing software; "
                "cannot identify the ESS for TCKDB."
            )

        level_of_theory: dict[str, Any] = {"method": str(method)}
        basis = level.get("basis")
        if basis:
            level_of_theory["basis"] = str(basis)

        software_release: dict[str, Any] = {"name": str(software_name)}
        ess_versions = record.get("ess_versions")
        if isinstance(ess_versions, Mapping):
            # ess_versions is keyed by job type ('opt', 'freq', 'sp', 'neb'),
            # not by software name. Fall back to opt's version if the
            # job-specific entry is missing (often the case for combined
            # opt+freq runs or shared sp/freq logs).
            ess_version = ess_versions.get(ess_job_key) or ess_versions.get("opt")
            if ess_version:
                software_release["version"] = str(ess_version)

        calc: dict[str, Any] = {
            "type": calc_type,
            "quality": "raw",
            "software_release": software_release,
            "level_of_theory": level_of_theory,
        }

        arc_version = output_doc.get("arc_version")
        arc_git_commit = output_doc.get("arc_git_commit")
        if arc_version or arc_git_commit:
            wt: dict[str, Any] = {"name": "ARC"}
            if arc_version:
                wt["version"] = str(arc_version)
            if arc_git_commit:
                wt["git_commit"] = str(arc_git_commit)
            calc["workflow_tool_release"] = wt

        if result_field and result_payload:
            calc[result_field] = dict(result_payload)

        # tckdb_origin is qualifier metadata (e.g. "this SP row is reused
        # from opt, not an independently executed ESS job"). It rides
        # under parameters_json so server-side schema doesn't need to
        # grow a column for an ARC-side concern.
        if tckdb_origin is not None:
            calc["parameters_json"] = {"tckdb_origin": dict(tckdb_origin)}

        return calc

    # ------------------------------------------------------------------
    # Upload + sidecar finalization
    # ------------------------------------------------------------------

    def _finalize_skipped(self, written: WrittenPayload) -> UploadOutcome:
        sc = written.sidecar
        sc.status = "skipped"
        self._writer.update_sidecar(written.sidecar_path, sc)
        logger.info("TCKDB upload skipped (upload=false): %s", written.payload_path)
        return UploadOutcome(
            status="skipped",
            payload_path=written.payload_path,
            sidecar_path=written.sidecar_path,
            idempotency_key=sc.idempotency_key,
        )

    def _upload(
        self,
        written: WrittenPayload,
        payload: dict[str, Any],
        *,
        endpoint: str = CONFORMER_UPLOAD_ENDPOINT,
    ) -> UploadOutcome:
        sc = written.sidecar
        api_key = self._config.resolve_api_key()
        if not api_key:
            msg = (
                f"TCKDB API key env var '{self._config.api_key_env}' is not set; "
                "skipping network call and recording sidecar as failed."
            )
            return self._record_failure(written, msg, raised=ValueError(msg))

        try:
            client = self._make_client(api_key)
        except Exception as exc:  # pragma: no cover - defensive
            return self._record_failure(written, f"client init failed: {exc}", exc)

        try:
            response = client.request_json(
                "POST",
                endpoint,
                json=payload,
                idempotency_key=sc.idempotency_key,
            )
        except Exception as exc:
            _close_quietly(client, "after upload failure")
            return self._record_failure(written, str(exc), exc)
        else:
            _close_quietly(client, "after upload success")

        sc.status = "uploaded"
        sc.uploaded_at = _utcnow_iso()
        sc.response_status_code = getattr(response, "status_code", None)
        response_data = getattr(response, "data", None)
        sc.response_body = _summarize_response_body(response_data)
        sc.idempotency_replayed = bool(getattr(response, "idempotency_replayed", False))
        sc.last_error = None
        self._writer.update_sidecar(written.sidecar_path, sc)
        if sc.idempotency_replayed:
            logger.info(
                "TCKDB upload replayed (idempotent): %s key=%s",
                written.payload_path,
                sc.idempotency_key,
            )
        else:
            logger.info(
                "TCKDB upload succeeded: %s key=%s",
                written.payload_path,
                sc.idempotency_key,
            )
        primary, additional = _extract_calc_refs(response_data)
        return UploadOutcome(
            status="uploaded",
            payload_path=written.payload_path,
            sidecar_path=written.sidecar_path,
            idempotency_key=sc.idempotency_key,
            response=sc.response_body,
            primary_calculation=primary,
            additional_calculations=additional,
        )

    def _record_failure(
        self, written: WrittenPayload, message: str, raised: BaseException
    ) -> UploadOutcome:
        sc = written.sidecar
        sc.status = "failed"
        sc.last_error = message
        self._writer.update_sidecar(written.sidecar_path, sc)
        logger.warning(
            "TCKDB upload failed (strict=%s): %s key=%s err=%s",
            self._config.strict,
            written.payload_path,
            sc.idempotency_key,
            message,
        )
        if self._config.strict:
            raise raised
        return UploadOutcome(
            status="failed",
            payload_path=written.payload_path,
            sidecar_path=written.sidecar_path,
            idempotency_key=sc.idempotency_key,
            error=message,
        )

    def _make_client(self, api_key: str):
        if self._client_factory is not None:
            return self._client_factory(self._config, api_key)
        return TCKDBClient(
            self._config.base_url,
            api_key=api_key,
            timeout=self._config.timeout_seconds,
        )

    def _upload_artifact(
        self,
        *,
        written: WrittenArtifact,
        request_body: dict[str, Any],
        endpoint: str,
    ) -> ArtifactUploadOutcome:
        sc = written.sidecar
        api_key = self._config.resolve_api_key()
        if not api_key:
            msg = (
                f"TCKDB API key env var '{self._config.api_key_env}' is not set; "
                "skipping artifact network call."
            )
            return self._record_artifact_failure(written, msg, ValueError(msg))

        try:
            client = self._make_client(api_key)
        except Exception as exc:  # pragma: no cover - defensive
            return self._record_artifact_failure(
                written, f"client init failed: {exc}", exc
            )

        try:
            response = client.request_json(
                "POST",
                endpoint,
                json=request_body,
                idempotency_key=sc.idempotency_key,
            )
        except Exception as exc:
            _close_quietly(client, "after artifact upload failure")
            return self._record_artifact_failure(written, str(exc), exc)
        else:
            _close_quietly(client, "after artifact upload success")

        sc.status = "uploaded"
        sc.uploaded_at = _utcnow_iso()
        sc.response_status_code = getattr(response, "status_code", None)
        sc.response_body = _summarize_response_body(getattr(response, "data", None))
        sc.idempotency_replayed = bool(getattr(response, "idempotency_replayed", False))
        sc.last_error = None
        self._writer.update_artifact_sidecar(written.sidecar_path, sc)
        if sc.idempotency_replayed:
            logger.info(
                "TCKDB artifact upload replayed (idempotent): calc=%s kind=%s key=%s",
                sc.calculation_id, sc.kind, sc.idempotency_key,
            )
        else:
            logger.info(
                "TCKDB artifact upload succeeded: calc=%s kind=%s key=%s",
                sc.calculation_id, sc.kind, sc.idempotency_key,
            )
        return ArtifactUploadOutcome(
            status="uploaded",
            sidecar_path=written.sidecar_path,
            idempotency_key=sc.idempotency_key,
            calculation_id=sc.calculation_id,
            kind=sc.kind,
            response=sc.response_body,
        )

    def _record_artifact_failure(
        self,
        written: WrittenArtifact,
        message: str,
        raised: BaseException,
    ) -> ArtifactUploadOutcome:
        sc = written.sidecar
        sc.status = "failed"
        sc.last_error = message
        self._writer.update_artifact_sidecar(written.sidecar_path, sc)
        logger.warning(
            "TCKDB artifact upload failed (strict=%s): calc=%s kind=%s err=%s",
            self._config.strict, sc.calculation_id, sc.kind, message,
        )
        if self._config.strict:
            raise raised
        return ArtifactUploadOutcome(
            status="failed",
            sidecar_path=written.sidecar_path,
            idempotency_key=sc.idempotency_key,
            calculation_id=sc.calculation_id,
            kind=sc.kind,
            error=message,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _sp_is_reused_from_opt(output_doc: Mapping[str, Any]) -> bool:
    """Whether ARC's SP energy is reused from the opt calculation.

    The signal is structural equality between ``sp_level`` and
    ``opt_level``. Two flavors both count:

    - ``sp_level`` absent / null in output.yml — the common ARC case
      where the user only declares ``opt_level=`` and ARC reuses the
      opt energy as the SP energy.
    - ``sp_level`` explicitly set to a dict structurally equal to
      ``opt_level`` — same outcome, no separate ESS job.

    A ``sp_level`` that differs from ``opt_level`` (different method,
    basis, software, etc.) means a real SP job was executed, so we
    return False and the calculation gets no reused-result marker.
    """
    opt_level = output_doc.get("opt_level")
    if not isinstance(opt_level, Mapping):
        return False
    sp_level = output_doc.get("sp_level")
    if sp_level is None:
        return True
    if not isinstance(sp_level, Mapping):
        return False
    return dict(sp_level) == dict(opt_level)


def _reused_origin(reused_from_calc_type: str) -> dict[str, Any]:
    """Build the ``tckdb_origin`` payload for a reused-result calculation.

    Lives under ``parameters_json.tckdb_origin`` on the calculation row.
    The DAG edge between calculations carries the relational link
    (e.g. ``opt -> sp`` with role ``single_point_on``); this dict
    carries the qualifier — *this* row's energy is reused, not freshly
    computed — so downstream consumers can tell aggregate-from-opt SP
    rows apart from independently executed SP jobs.
    """
    return {
        "origin_kind": "reused_result",
        "reused_from": {"calculation_type": reused_from_calc_type},
        "reason": (
            f"sp_level equals {reused_from_calc_type}_level; "
            f"{reused_from_calc_type} electronic energy reused as SP energy"
        ),
        "independent_ess_job": False,
        "producer": "ARC",
    }


def _resolve_level(
    output_doc: Mapping[str, Any], job_kind: str
) -> Mapping[str, Any] | None:
    """Return the level-of-theory dict for ``job_kind`` ('opt'/'freq'/'sp').

    The opt level is always required by the primary calculation. For
    freq/sp, ARC users very often run all three at the same level and
    only declare ``opt_level=`` — ``output.yml`` then writes ``freq_level:
    null`` / ``sp_level: null``. To avoid silently dropping the freq/sp
    additional calculations in the common case, fall back to ``opt_level``
    when the job-specific level is absent. A present-but-distinct
    ``freq_level`` / ``sp_level`` is treated as authoritative.
    """
    if job_kind == "opt":
        level = output_doc.get("opt_level")
        return level if isinstance(level, Mapping) else None
    job_level = output_doc.get(f"{job_kind}_level")
    if isinstance(job_level, Mapping):
        return job_level
    opt_level = output_doc.get("opt_level")
    return opt_level if isinstance(opt_level, Mapping) else None


def _opt_result_payload(record: Mapping[str, Any]) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    if record.get("opt_n_steps") is not None:
        out["n_steps"] = record["opt_n_steps"]
    if record.get("opt_final_energy_hartree") is not None:
        out["final_energy_hartree"] = record["opt_final_energy_hartree"]
    return out or None


_FREQ_FIELD_SPECS = (
    # (record_key, payload_key, coerce)
    ("freq_n_imag", "n_imag", int),
    ("imag_freq_cm1", "imag_freq_cm1", float),
    ("zpe_hartree", "zpe_hartree", float),
)


def _freq_result_payload(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build a FreqResultPayload-shaped dict from an output.yml record.

    Returns ``None`` when no freq fields are populated. Returns ``None``
    and logs a warning when any present field cannot be coerced to the
    expected numeric type — the task spec mandates skipping the whole
    additional calculation rather than uploading a partial freq row.
    """
    if all(record.get(rkey) is None for rkey, _, _ in _FREQ_FIELD_SPECS):
        return None
    out: dict[str, Any] = {}
    for record_key, payload_key, coerce in _FREQ_FIELD_SPECS:
        value = record.get(record_key)
        if value is None:
            continue
        try:
            out[payload_key] = coerce(value)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "TCKDB freq additional calculation skipped for label=%s: "
                "malformed %s=%r (%s)",
                record.get("label"), record_key, value, exc,
            )
            return None
    return out or None


def _sp_result_payload(record: Mapping[str, Any]) -> dict[str, Any] | None:
    # ``sp_energy_hartree`` is ARC's record key; ``electronic_energy_hartree``
    # is the TCKDB-side field name (some records may carry it directly).
    record_key = "sp_energy_hartree" if record.get("sp_energy_hartree") is not None \
        else "electronic_energy_hartree"
    energy = record.get(record_key)
    if energy is None:
        return None
    try:
        return {"electronic_energy_hartree": float(energy)}
    except (TypeError, ValueError) as exc:
        logger.warning(
            "TCKDB sp additional calculation skipped for label=%s: "
            "malformed %s=%r (%s)",
            record.get("label"), record_key, energy, exc,
        )
        return None


def _build_thermo_block(
    thermo_record: Any,
    *,
    included_calc_keys: list[str],
) -> dict[str, Any] | None:
    """Build a ThermoInBundle dict from ``output.yml`` thermo data.

    Returns ``None`` when no usable thermo content can be assembled. The
    server-side ``ThermoInBundle.validate_has_scientific_content``
    rejects empty thermo blocks, so emitting one with nothing in it
    would just produce a 422; better to omit at the producer.

    Mapping (from ``arc/output.py::_thermo_to_dict``):
        h298_kj_mol  → h298_kj_mol
        s298_j_mol_k → s298_j_mol_k
        tmin_k       → tmin_k
        tmax_k       → tmax_k
        nasa_low.coeffs  → nasa.a1..a7
        nasa_high.coeffs → nasa.b1..b7
        nasa_low.tmin_k  → nasa.t_low
        nasa_low.tmax_k  → nasa.t_mid (cross-checked vs nasa_high.tmin_k)
        nasa_high.tmax_k → nasa.t_high
        cp_data          → points (per-point validation; bad points dropped)

    ``source_calculations`` are populated from ``included_calc_keys``:
    each of ``opt`` / ``freq`` / ``sp`` that actually made it into the
    bundle gets a link with the matching role, since
    ``ThermoCalculationRole`` accepts those literals directly.
    """
    if not isinstance(thermo_record, Mapping):
        return None

    block: dict[str, Any] = {}

    h298 = thermo_record.get("h298_kj_mol")
    if h298 is not None:
        try:
            block["h298_kj_mol"] = float(h298)
        except (TypeError, ValueError) as exc:
            logger.warning("TCKDB thermo: malformed h298_kj_mol=%r (%s)", h298, exc)

    s298 = thermo_record.get("s298_j_mol_k")
    if s298 is not None:
        try:
            block["s298_j_mol_k"] = float(s298)
        except (TypeError, ValueError) as exc:
            logger.warning("TCKDB thermo: malformed s298_j_mol_k=%r (%s)", s298, exc)

    for tk in ("tmin_k", "tmax_k"):
        v = thermo_record.get(tk)
        if v is not None:
            try:
                block[tk] = float(v)
            except (TypeError, ValueError) as exc:
                logger.warning("TCKDB thermo: malformed %s=%r (%s)", tk, v, exc)

    nasa = _build_nasa_block(thermo_record.get("nasa_low"), thermo_record.get("nasa_high"))
    if nasa is not None:
        block["nasa"] = nasa

    points = _build_thermo_points(thermo_record.get("cp_data"))
    if points:
        block["points"] = points

    # ThermoCalculationRole accepts opt/freq/sp/composite/imported. We
    # link freq and sp when present (they're the calcs that physically
    # produced the thermo); we also link opt if it's the only calc in
    # the bundle (unusual, but covers thermo-from-opt-only edge cases).
    sources: list[dict[str, str]] = []
    for key in (_CALC_KEY_FREQ, _CALC_KEY_SP):
        if key in included_calc_keys:
            sources.append({"calculation_key": key, "role": key})
    if not sources and _CALC_KEY_OPT in included_calc_keys:
        sources.append({"calculation_key": _CALC_KEY_OPT, "role": _CALC_KEY_OPT})
    if sources:
        block["source_calculations"] = sources

    has_scalar = "h298_kj_mol" in block or "s298_j_mol_k" in block
    has_nasa = "nasa" in block
    has_points = "points" in block
    if not (has_scalar or has_nasa or has_points):
        # Server would 422 us; nothing usable here.
        return None
    return block


def _build_nasa_block(
    nasa_low: Any, nasa_high: Any
) -> dict[str, Any] | None:
    """Map ARC's two NASA blocks to ``ThermoNASACreate``.

    Returns ``None`` if either block is missing or fails any of the
    structural checks. Per spec, malformed NASA must skip the NASA block
    only — scalar thermo and Cp points are kept by the caller.
    """
    if not isinstance(nasa_low, Mapping) or not isinstance(nasa_high, Mapping):
        return None
    low_coeffs = nasa_low.get("coeffs")
    high_coeffs = nasa_high.get("coeffs")
    if not isinstance(low_coeffs, list) or len(low_coeffs) != 7:
        logger.warning(
            "TCKDB thermo: NASA block skipped — nasa_low.coeffs must be a list of 7 floats."
        )
        return None
    if not isinstance(high_coeffs, list) or len(high_coeffs) != 7:
        logger.warning(
            "TCKDB thermo: NASA block skipped — nasa_high.coeffs must be a list of 7 floats."
        )
        return None
    t_low = nasa_low.get("tmin_k")
    t_mid_low = nasa_low.get("tmax_k")
    t_mid_high = nasa_high.get("tmin_k")
    t_high = nasa_high.get("tmax_k")
    if None in (t_low, t_mid_low, t_mid_high, t_high):
        logger.warning(
            "TCKDB thermo: NASA block skipped — temperature bounds incomplete."
        )
        return None
    try:
        t_low_f, t_mid_low_f, t_mid_high_f, t_high_f = (
            float(t_low), float(t_mid_low), float(t_mid_high), float(t_high)
        )
    except (TypeError, ValueError) as exc:
        logger.warning("TCKDB thermo: NASA block skipped — non-numeric bounds (%s).", exc)
        return None
    if t_mid_low_f != t_mid_high_f:
        logger.warning(
            "TCKDB thermo: NASA block skipped — nasa_low.tmax_k=%s != nasa_high.tmin_k=%s.",
            t_mid_low_f, t_mid_high_f,
        )
        return None
    try:
        low_floats = [float(c) for c in low_coeffs]
        high_floats = [float(c) for c in high_coeffs]
    except (TypeError, ValueError) as exc:
        logger.warning("TCKDB thermo: NASA block skipped — non-numeric coefficient (%s).", exc)
        return None
    block: dict[str, Any] = {
        "t_low": t_low_f,
        "t_mid": t_mid_low_f,
        "t_high": t_high_f,
    }
    for i, c in enumerate(low_floats, start=1):
        block[f"a{i}"] = c
    for i, c in enumerate(high_floats, start=1):
        block[f"b{i}"] = c
    return block


def _build_thermo_points(cp_data: Any) -> list[dict[str, Any]]:
    """Map ARC's ``cp_data`` list to ``ThermoPointCreate`` dicts.

    Each entry must carry ``temperature_k``; other fields are optional.
    Malformed individual points (missing/non-numeric temperature, or a
    Cp value that won't coerce) are dropped with a warning so a single
    bad row doesn't take out the whole thermo upload.
    """
    if not isinstance(cp_data, list):
        return []
    seen_temps: set[float] = set()
    points: list[dict[str, Any]] = []
    for i, raw in enumerate(cp_data):
        if not isinstance(raw, Mapping):
            logger.warning("TCKDB thermo: cp_data[%d] skipped — not a mapping.", i)
            continue
        t = raw.get("temperature_k")
        if t is None:
            logger.warning("TCKDB thermo: cp_data[%d] skipped — missing temperature_k.", i)
            continue
        try:
            t_f = float(t)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "TCKDB thermo: cp_data[%d] skipped — non-numeric temperature_k=%r (%s).",
                i, t, exc,
            )
            continue
        if t_f <= 0:
            logger.warning(
                "TCKDB thermo: cp_data[%d] skipped — temperature_k must be > 0 (got %s).",
                i, t_f,
            )
            continue
        if t_f in seen_temps:
            # Server enforces uniqueness by temperature_k; skip duplicates here.
            logger.warning(
                "TCKDB thermo: cp_data[%d] skipped — duplicate temperature_k=%s.",
                i, t_f,
            )
            continue
        seen_temps.add(t_f)
        point: dict[str, Any] = {"temperature_k": t_f}
        for src_key, dst_key in (
            ("cp_j_mol_k", "cp_j_mol_k"),
            ("h_kj_mol", "h_kj_mol"),
            ("s_j_mol_k", "s_j_mol_k"),
            ("g_kj_mol", "g_kj_mol"),
        ):
            v = raw.get(src_key)
            if v is None:
                continue
            try:
                point[dst_key] = float(v)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "TCKDB thermo: cp_data[%d].%s dropped — non-numeric %r (%s).",
                    i, src_key, v, exc,
                )
        points.append(point)
    return points


def _require_xyz_text(record: Mapping[str, Any]) -> str:
    """Pull the xyz string out of an output.yml species record.

    Converts ARC's atom-only xyz (``"C 0.0 0.0 0.0\\nH 1.0 0.0 0.0"``,
    emitted by ``xyz_to_str``) into the standard XYZ format that TCKDB
    expects (``"<n_atoms>\\n<comment>\\n<atom lines>"``). If the input
    already has an integer atom-count header, it is passed through
    untouched. This is the format-translation boundary between ARC's
    internal convention and the TCKDB schema.
    """
    xyz = record.get("xyz")
    if not xyz:
        raise ValueError(
            f"output.yml record for label={record.get('label')!r} has no xyz; "
            "cannot build geometry payload."
        )
    text = str(xyz).strip()
    if not text:
        raise ValueError(
            f"output.yml record for label={record.get('label')!r} has empty xyz."
        )

    lines = text.splitlines()
    try:
        int(lines[0].strip())
        return text
    except (ValueError, IndexError):
        pass
    label = record.get("label") or ""
    return f"{len(lines)}\n{label}\n{text}"


def _summarize_response_body(body: Any, *, max_chars: int = 2000) -> Any:
    """Truncate huge bodies so the sidecar stays small and grep-friendly."""
    if body is None:
        return None
    if isinstance(body, (dict, list)):
        return body
    text = str(body)
    if len(text) > max_chars:
        return text[:max_chars] + "...<truncated>"
    return text


def _extract_calc_refs(
    response_data: Any,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Pull primary/additional CalculationUploadRef dicts from a server response.

    Two response shapes are recognized:

    1. **Conformer upload** (``/uploads/conformers``): ``primary_calculation``
       and ``additional_calculations`` sit at the top level::

           {"primary_calculation": {"calculation_id": ..., "type": ...},
            "additional_calculations": [...]}

    2. **Computed-species bundle** (``/uploads/computed-species``): the
       same fields are nested under ``conformers[0]``::

           {"species_entry_id": ..., "conformers": [
              {"key": "...", "primary_calculation": {...},
               "additional_calculations": [...]}, ...]}

       Only the first conformer's refs are surfaced — today's ARC bundles
       carry exactly one conformer per species, so multi-conformer
       responses would need a richer outcome shape, which we'd add when
       the producer side starts emitting them.

    Older server builds omit these fields entirely; the caller treats
    their absence as "no artifact targets known".
    """
    if not isinstance(response_data, Mapping):
        return None, []
    if "conformers" in response_data and isinstance(response_data["conformers"], list):
        confs = response_data["conformers"]
        if confs and isinstance(confs[0], Mapping):
            return _extract_calc_refs(confs[0])
        return None, []
    primary = response_data.get("primary_calculation")
    if not isinstance(primary, Mapping):
        primary = None
    else:
        primary = dict(primary)
    additional_raw = response_data.get("additional_calculations") or []
    additional = [dict(ref) for ref in additional_raw if isinstance(ref, Mapping)]
    return primary, additional


def _skip(
    calculation_id: int, kind: str, reason: str
) -> "ArtifactUploadOutcome":
    """Build a skipped outcome and log the reason once."""
    logger.info(
        "TCKDB artifact upload skipped: calc=%s kind=%s reason=%s",
        calculation_id, kind, reason,
    )
    return ArtifactUploadOutcome(
        status="skipped",
        sidecar_path=None,
        idempotency_key=None,
        calculation_id=calculation_id,
        kind=kind,
        skip_reason=reason,
    )


def _close_quietly(client: Any, context: str) -> None:
    """Close a TCKDB client and swallow close errors with a debug log."""
    close = getattr(client, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:  # pragma: no cover - close errors swallowed
        logger.debug("TCKDB client close errored %s", context, exc_info=True)


__all__ = [
    "ARTIFACTS_ENDPOINT_TEMPLATE",
    "ArtifactUploadOutcome",
    "COMPUTED_SPECIES_ENDPOINT",
    "COMPUTED_SPECIES_KIND",
    "CONFORMER_UPLOAD_ENDPOINT",
    "PAYLOAD_KIND",
    "TCKDBAdapter",
    "UploadOutcome",
]
