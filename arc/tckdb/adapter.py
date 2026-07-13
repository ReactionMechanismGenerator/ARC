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
import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tckdb_client import TCKDBClient
from tckdb_client.errors import TCKDBError

from arc.common import get_logger
from arc.tckdb.config import (
    IMPLEMENTED_ARTIFACT_KINDS,
    TCKDBConfig,
    UPLOAD_MODE_COMPUTED_REACTION,
    UPLOAD_MODE_COMPUTED_SPECIES,
)

# Upload modes whose bundle payload already carries input/output_log
# artifacts inline under each calculation. Standalone artifact sidecars
# in these modes would (a) duplicate uploads and (b) lock in
# server-assigned calculation IDs that go stale on DB resets — see
# `submit_artifacts_for_calculation` for the gating logic.
_BUNDLE_MODES_WITH_INLINE_ARTIFACTS = frozenset({
    UPLOAD_MODE_COMPUTED_SPECIES,
    UPLOAD_MODE_COMPUTED_REACTION,
})
from arc.tckdb.idempotency import (
    ArtifactIdempotencyInputs,
    IdempotencyInputs,
    build_artifact_idempotency_key,
    build_idempotency_key,
)
from arc.tckdb.constraints import serialize_constraints
from arc.tckdb.payload_writer import (
    ArtifactSidecarMetadata,
    PayloadWriter,
    SidecarMetadata,
    WrittenArtifact,
    WrittenPayload,
    _utcnow_iso,
)


def _serialize_calc_constraints(source) -> list[dict]:
    """Translate a parser-shaped constraint list into the TCKDB payload shape.

    ``source`` is whatever ``arc/output.py`` attached to the record (a
    list of parser dicts) or what the caller passed explicitly. Empty /
    None / unrecognised input produces ``[]``. Wraps
    ``arc.tckdb.constraints.serialize_constraints`` so per-calc parse
    failures never bubble up into payload generation.
    """
    if not source:
        return []
    if not isinstance(source, (list, tuple)):
        logger.warning("TCKDB constraints: expected list, got %s; emitting [].",
                       type(source).__name__)
        return []
    try:
        return serialize_constraints(source)
    except Exception as exc:  # defensive — serializer logs internally too
        logger.warning("TCKDB constraints: serialization failed: %s; emitting [].",
                       exc)
        return []


logger = get_logger()

CONFORMER_UPLOAD_ENDPOINT = "/uploads/conformers"
PAYLOAD_KIND = "conformer_calculation"
ARTIFACTS_ENDPOINT_TEMPLATE = "/calculations/{calculation_id}/artifacts"

# Computed-species bundle endpoint. One self-contained payload that
# carries species_entry + conformers + calcs + artifacts + thermo, with
# all cross-references expressed as local string keys (no DB ids).
COMPUTED_SPECIES_ENDPOINT = "/uploads/computed-species"
COMPUTED_SPECIES_KIND = "computed_species"

# Computed-reaction bundle endpoint. Like computed-species but adds
# reactant/product species blocks, an inline transition_state, and a
# modified-Arrhenius kinetics block with producer-declared
# source_calculations.
COMPUTED_REACTION_ENDPOINT = "/uploads/computed-reaction"
COMPUTED_REACTION_KIND = "computed_reaction"

# Readiness-probe (/readyz) retry policy. During a long ARC run the TCKDB
# server can be briefly not-ready (restart, rolling deploy, momentary
# load). Rather than fail the in-run upload on the first blip, the
# preflight probe retries with exponential backoff before giving up.
#
# PREFLIGHT_MAX_ATTEMPTS counts the initial probe plus retries (5 → 1
# initial + 4 retries). Delays follow PREFLIGHT_BASE_DELAY_SECONDS * 2**i
# (1s, 2s, 4s, 8s) capped at PREFLIGHT_MAX_DELAY_SECONDS, so the worst
# case waits ~1+2+4+8 = 15s across the four gaps — well under a minute so
# a genuinely-down server doesn't stall the whole run. Tunable here.
PREFLIGHT_MAX_ATTEMPTS = 5
PREFLIGHT_BASE_DELAY_SECONDS = 1.0
PREFLIGHT_MAX_DELAY_SECONDS = 8.0

# Local calculation-key namespace within a computed-species bundle.
# These keys are referenced from `depends_on.parent_calculation_key` and
# `thermo.source_calculations[].calculation_key`. They have no relation
# to TCKDB-assigned calculation_ids — the bundle endpoint mints those
# server-side and returns them in the response.
_CALC_KEY_OPT = "opt"
_CALC_KEY_OPT_COARSE = "opt_coarse"
_CALC_KEY_FREQ = "freq"
_CALC_KEY_SP = "sp"

# Version tag stamped onto every parsed Hessian payload. Bump when the
# Cartesian-Hessian parsing in ``arc/parser/adapters/{gaussian,orca}.py``
# changes in a way that would alter the emitted lower-triangle values.
_HESSIAN_PARSER_VERSION = "arc-hessian-1"

# ESS name (from ``arc.parser.parser.determine_ess``) → TCKDB
# ``HessianSource`` enum value. Gaussian prints the Hessian inside the log
# (``parsed_log``); Orca writes it to a sibling ``.hess`` file
# (``parsed_hess``). Other ESS engines have no Cartesian-Hessian parser
# wired up, so they are absent and the Hessian is simply skipped.
_HESSIAN_SOURCE_BY_ESS = {
    "gaussian": "parsed_log",
    "orca": "parsed_hess",
}

# Per-calc record-field map for output_log artifacts. Same convention as
# `_LOG_FIELD_BY_CALC_KEY` in the existing artifact path: the species
# record carries `<job>_log` paths that come straight from
# `arc/output.py::_spc_to_dict`.
_LOG_FIELD_BY_CALC_KEY = {
    _CALC_KEY_OPT: "opt_log",
    _CALC_KEY_OPT_COARSE: "coarse_opt_log",
    _CALC_KEY_FREQ: "freq_log",
    _CALC_KEY_SP: "sp_log",
    # ts_guess role is method-dispatched at lookup time (see
    # ``_resolve_log_field``) because the producing TS-guess adapter
    # determines which record field carries the log path
    # (``neb_log`` for orca_neb, ``gsm_log`` for xtb_gsm). The gate in
    # ``_build_ts_block`` is what decides whether the calc is emitted
    # at all.
}

# Same shape, for input-deck paths. `arc/output.py` emits these per-job
# (with per-job software → per-job filename), only when the deck file
# actually exists on disk; null when the deck wasn't kept (archived runs).
# (No coarse_opt_input field today — coarse jobs share the engine's
# input filename with the fine opt; the deck is keyed off the log dir.)
_INPUT_FIELD_BY_CALC_KEY = {
    _CALC_KEY_OPT: "opt_input",
    _CALC_KEY_FREQ: "freq_input",
    _CALC_KEY_SP: "sp_input",
}

# Held-fixed coordinate constraints emitted onto the species record by
# ``arc/output.py::_spc_to_dict`` (one list per ESS-job calc). Scan calcs
# carry their constraints inline on the ``additional_calculations`` entry
# rather than via species_record, so they're not in this map.
_CONSTRAINTS_FIELD_BY_CALC_KEY = {
    _CALC_KEY_OPT: "opt_constraints",
    _CALC_KEY_FREQ: "freq_constraints",
    _CALC_KEY_SP: "sp_constraints",
}

# (artifact_kind, record-field-map) pairs that the inline-artifact
# helper iterates per calc. Keeping the mapping data-driven so adding
# checkpoints (or any future kind) is one tuple, not a code branch.
_INLINE_ARTIFACT_SOURCES: tuple[tuple[str, dict[str, str]], ...] = (
    ("output_log", _LOG_FIELD_BY_CALC_KEY),
    ("input", _INPUT_FIELD_BY_CALC_KEY),
)

# TS-side calc role; reaction bundles can also carry an irc on the TS,
# but ARC may not always have one parsed cleanly, so it's optional.
_CALC_KEY_IRC = "irc"

# TS guess parent calculation (only emitted when the chosen guess was
# itself a real path-search calculation — orca_neb or xtb_gsm).
# Geometry-only guesses (heuristics, AutoTST/KinBot wrappers, GCN,
# user-supplied XYZ) stay as ``calculation_input_geometry`` and never
# produce a calc node here, per the calculation_dependency contract:
# parent must be a real calc.
_CALC_KEY_TS_GUESS = "ts_guess"

# ARC's TSGuess.method strings → TCKDB ``path_search_result.method``
# enum value. Lookup is case- and whitespace-insensitive at the call
# site (producer strings are stable identifiers but historically have
# casing variations like ``orca_neb`` vs ``ORCA_NEB`` and module-name
# vs class-name forms ``xtb_gsm`` vs ``xTB-GSM``). Geometry-only
# methods (heuristics, AutoTST, KinBot, GCN, user XYZ) are deliberately
# absent — they have no path-search log to anchor a parent calc.
_TS_GUESS_PATH_SEARCH_METHODS: dict[str, str] = {
    "orca_neb": "neb",
    "xtb_gsm": "gsm",
    "xtb-gsm": "gsm",
}

# Per-method record-field name carrying the path-search log path.
# ``neb_log`` is populated by ``arc/output.py`` from ``paths['neb']``
# (set by the scheduler from ``tsg.log_path`` when orca_neb wins) and
# ``gsm_log`` from ``paths['gsm']`` (xtb_gsm wins). Both fields are
# read by ``_resolve_ts_guess_path_search`` to gate emission of the
# ``ts_guess`` parent calc.
_TS_GUESS_LOG_FIELD_BY_METHOD: dict[str, str] = {
    "neb": "neb_log",
    "gsm": "gsm_log",
}

# Per-method gate for emitting the path-search log as an inline
# ``output_log`` artifact. Gated by what the backend's ESS-signature
# validator accepts: NEB logs are real ORCA output files (pass); GSM
# stringfiles are multi-frame XYZ trajectories (fail signature check —
# backend supports cfour/gaussian/molpro/nwchem/orca/psi4/qchem/turbomole
# only). For GSM, the per-frame geometries already ride inside
# ``path_search_result.points``, so the raw stringfile would be
# redundant provenance even if the backend accepted it.
_TS_GUESS_UPLOAD_LOG_AS_ARTIFACT: dict[str, bool] = {
    "neb": True,
    "gsm": False,
}

# Per-method static properties of the path-search algorithm itself.
# These are intrinsic to the *method*, not the run — both ARC-supported
# adapters (orca_neb, xtb_gsm) are double-ended path-search methods that
# consume two endpoint geometries (reactant + product). Single-ended
# methods (growing-string, freezing-string) would set is_double_ended
# to False and source_endpoint_count to 1; not currently exposed by
# any ARC adapter, so absent here.
_PATH_SEARCH_METHOD_PROPERTIES: dict[str, dict[str, Any]] = {
    "neb": {"is_double_ended": True, "source_endpoint_count": 2},
    "gsm": {"is_double_ended": True, "source_endpoint_count": 2},
}


def _resolve_ts_guess_path_search(method: object) -> str | None:
    """Return the TCKDB ``path_search_result.method`` enum value for an
    ARC TSGuess.method string, or ``None`` for geometry-only / unknown
    methods.

    Conservative: only matches strings explicitly listed in
    ``_TS_GUESS_PATH_SEARCH_METHODS`` (case- and whitespace-insensitive).
    Non-string and unknown values return ``None`` — the caller falls
    back to geometry-only TS provenance (no parent calc emitted).
    """
    if not isinstance(method, str):
        return None
    return _TS_GUESS_PATH_SEARCH_METHODS.get(method.strip().lower())


def _resolve_log_field(role: str, record: Mapping[str, Any]) -> str | None:
    """Resolve the species-record field carrying the output-log path for ``role``.

    Ordinary roles dispatch through ``_LOG_FIELD_BY_CALC_KEY``. The
    ``ts_guess`` role is method-dispatched: the field name depends on
    which TS-guess adapter produced the chosen guess (orca_neb →
    ``neb_log``, xtb_gsm → ``gsm_log``). Returns ``None`` when the role
    has no log mapping or the TS-guess method is geometry-only.
    """
    if role == _CALC_KEY_TS_GUESS:
        method_enum = _resolve_ts_guess_path_search(record.get("chosen_ts_method"))
        if method_enum is None:
            return None
        # Some path-search methods (notably GSM) produce log artifacts
        # the backend's ESS-signature validator rejects. Skip the
        # artifact upload for those — the geometric data still rides
        # inside ``path_search_result.points``.
        if not _TS_GUESS_UPLOAD_LOG_AS_ARTIFACT.get(method_enum, False):
            return None
        return _TS_GUESS_LOG_FIELD_BY_METHOD.get(method_enum)
    return _LOG_FIELD_BY_CALC_KEY.get(role)


# Backend ``KIND_ALLOWED_EXTENSIONS`` for ``output_log`` is
# ``{.log, .out, .orca}`` (see TCKDB_v2 fragments/artifact.py).
# Non-traditional log artifacts (e.g. the GSM ``stringfile.xyz0000``)
# need their declared filename adapted to satisfy the allowlist while
# preserving the original basename for human inspection. The backend
# treats ``filename`` as provenance metadata only — storage is content-
# addressed and the filename does not influence the URI.
_OUTPUT_LOG_ALLOWED_EXTS: frozenset[str] = frozenset({".log", ".out", ".orca"})


def _coerce_artifact_filename(name: str, kind: str) -> str:
    """Adapt an artifact ``filename`` to satisfy the backend's per-kind
    extension allowlist. Currently only ``output_log`` is coerced —
    other kinds either match upstream (``input``, ``checkpoint``) or
    aren't emitted by ARC.

    For ``output_log`` files whose extension is already permitted
    (``.log`` / ``.out`` / ``.orca``), the name is returned verbatim.
    For non-conforming names (e.g. GSM ``stringfile.xyz0000``), a
    ``.log`` suffix is appended so the original basename is preserved
    as the prefix (``stringfile.xyz0000.log``). The coercion is a
    no-op for any other kind.
    """
    if kind != "output_log":
        return name
    ext = os.path.splitext(name)[1].lower()
    if ext in _OUTPUT_LOG_ALLOWED_EXTS:
        return name
    return f"{name}.log"


# Bundle-local calc role for rotor scans. Per-rotor keys are minted as
# ``f"{_CALC_KEY_SCAN}_rotor_{i}"`` by ``arc/output.py``; the adapter
# accepts whatever key the species record provides without re-minting,
# so the source-of-truth for naming stays on the producer side.
_CALC_KEY_SCAN = "scan"

# ARC kinetics-unit string → TCKDB enum string. Source of truth for the
# enum values is TCKDB's ``ArrheniusAUnits`` and ``ActivationEnergyUnits``
# (backend/app/db/models/common.py). The ARC side normalizes whitespace
# and quotes before lookup so cosmetic variations don't miss.
_ARC_TO_TCKDB_A_UNITS: dict[str, str] = {
    "s^-1": "per_s",
    "1/s": "per_s",
    "cm^3/(mol*s)": "cm3_mol_s",
    "cm^3/(molecule*s)": "cm3_molecule_s",
    "m^3/(mol*s)": "m3_mol_s",
    "cm^6/(mol^2*s)": "cm6_mol2_s",
    "cm^6/(molecule^2*s)": "cm6_molecule2_s",
    "m^6/(mol^2*s)": "m6_mol2_s",
}

_ARC_TO_TCKDB_EA_UNITS: dict[str, str] = {
    "j/mol": "j_mol",
    "kj/mol": "kj_mol",
    "cal/mol": "cal_mol",
    "kcal/mol": "kcal_mol",
}


def _normalize_unit_key(text: str | None) -> str | None:
    """Lowercase + strip whitespace so ``" kJ/mol "`` matches ``"kj/mol"``.

    The ARC unit strings come from RMG and aren't perfectly consistent in
    case/whitespace; the TCKDB enums are. This is the boundary helper.
    """
    if text is None:
        return None
    s = str(text).strip().lower()
    return s or None


def arc_to_tckdb_a_units(arc_units: str | None) -> str | None:
    """Map an ARC ``A_units`` string to a TCKDB ``ArrheniusAUnits`` enum value.

    Returns ``None`` for null/empty input or unrecognized strings — the
    caller decides whether to omit the field or raise. Unrecognized
    units log a debug line so a producer with a typo can spot it
    without spamming warnings on every reaction.
    """
    key = _normalize_unit_key(arc_units)
    if key is None:
        return None
    enum_value = _ARC_TO_TCKDB_A_UNITS.get(key)
    if enum_value is None:
        logger.debug(
            "TCKDB kinetics: unrecognized A_units %r; field will be omitted.",
            arc_units,
        )
    return enum_value


def arc_to_tckdb_ea_units(arc_units: str | None) -> str | None:
    """Map an ARC ``Ea_units`` (or ``dEa_units``) string to a TCKDB ``ActivationEnergyUnits`` enum value.

    Same null-or-unknown → ``None`` policy as :func:`arc_to_tckdb_a_units`.
    """
    key = _normalize_unit_key(arc_units)
    if key is None:
        return None
    enum_value = _ARC_TO_TCKDB_EA_UNITS.get(key)
    if enum_value is None:
        logger.debug(
            "TCKDB kinetics: unrecognized Ea_units %r; field will be omitted.",
            arc_units,
        )
    return enum_value


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


@dataclass(frozen=True)
class _PreparedArtifactUpload:
    """ARC-local artifact upload plan item plus its sidecar handle."""

    written: WrittenArtifact
    calculation_key: str
    calculation_id: int
    path: Path
    kind: str
    label: str | None
    sha256: str
    bytes: int
    filename: str


class TCKDBReadinessError(RuntimeError):
    """Raised when the TCKDB readyz preflight fails before upload."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_json: Any = None,
        response_text: str | None = None,
        headers: Mapping[str, str] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_json = response_json
        self.response_text = response_text
        self.headers = dict(headers) if headers is not None else None


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
        self._preflight_checked = False
        self._preflight_metadata: dict[str, Any] | None = None
        self._preflight_error: TCKDBReadinessError | None = None

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
        outcomes = self.submit_artifact_batch_for_calculation(
            output_doc=output_doc,
            species_record=species_record,
            calculation_id=calculation_id,
            calculation_type=calculation_type,
            artifacts=[(kind, file_path)],
        )
        if outcomes is None:
            return None
        return outcomes[0] if outcomes else None

    def submit_artifact_batch_for_calculation(
        self,
        *,
        output_doc: Mapping[str, Any],
        species_record: Mapping[str, Any],
        calculation_id: int,
        calculation_type: str,
        artifacts: list[tuple[str, str | Path]],
    ) -> list[ArtifactUploadOutcome] | None:
        """Upload artifacts for one calculation with client-side batch grouping."""
        if not self._config.enabled:
            return None

        species_label = species_record.get("label") or "unlabeled"
        artifact_cfg = self._config.artifacts
        outcomes: list[ArtifactUploadOutcome] = []
        prepared: list[_PreparedArtifactUpload] = []

        for kind, file_path in artifacts:
            prepared_item = self._prepare_artifact_upload(
                output_doc=output_doc,
                species_label=species_label,
                calculation_id=calculation_id,
                kind=kind,
                file_path=file_path,
                artifact_cfg=artifact_cfg,
            )
            if isinstance(prepared_item, ArtifactUploadOutcome):
                outcomes.append(prepared_item)
            else:
                prepared.append(prepared_item)

        if not prepared:
            return outcomes

        batch_digest = _artifact_batch_digest(prepared)
        first = prepared[0]
        prepared[0] = _PreparedArtifactUpload(
            written=first.written,
            calculation_key=f"calc{calculation_id}-{batch_digest}",
            calculation_id=first.calculation_id,
            path=first.path,
            kind=first.kind,
            label=first.label,
            sha256=first.sha256,
            bytes=first.bytes,
            filename=first.filename,
        )

        batch_outcomes = self._upload_artifact_batch(
            prepared=prepared,
            idempotency_key_prefix=_artifact_batch_idempotency_prefix(
                self._config.project_label or output_doc.get("project"),
                species_label,
            ),
        )
        return outcomes + batch_outcomes

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

        # Additional screened conformers, if ARC surfaced them in the
        # output.yml record. Only the selected conformer carries full
        # opt/freq/sp/thermo provenance — the rest land as honest
        # "geometry + bare opt" observations so the conformer screen
        # isn't lost to the database. Selected stays first for stable
        # ordering and so consumers reading conformers[0] still get the
        # canonical conformer.
        selected_xyz = conformer_block["geometry"]["xyz_text"]
        alt_blocks = self._build_alt_conformer_blocks(
            output_doc=output_doc,
            species_record=species_record,
            selected_xyz_text=selected_xyz,
            selected_key=conformer_key,
        )

        bundle: dict[str, Any] = {
            "species_entry": self._species_entry_payload(species_record),
            "conformers": [conformer_block, *alt_blocks],
        }

        applied_corrections = _build_applied_energy_corrections(
            species_record.get("applied_energy_corrections") or [],
            source_calculation_key=(
                _CALC_KEY_SP if _CALC_KEY_SP in included_keys else None
            ),
        )
        if applied_corrections:
            bundle["applied_energy_corrections"] = applied_corrections

        thermo_block = _build_thermo_block(
            species_record.get("thermo"),
            included_calc_keys=included_keys,
        )
        if thermo_block is not None:
            bundle["thermo"] = thermo_block

        # Workflow-tool release at bundle level (in addition to per-calc):
        # mirrors what the conformer adapter records and lets the server
        # tag the species_entry with the producer.
        arc_wt = _arc_workflow_tool_release(output_doc)
        if arc_wt is not None:
            bundle["workflow_tool_release"] = arc_wt

        # Statmech block: carries frequency-scale-factor provenance,
        # plus richer base statmech metadata (external_symmetry,
        # is_linear, rigid_rotor_kind, statmech_treatment, point_group)
        # and slim torsion summaries when ARC populated them in the
        # species record's ``statmech`` subdict (written by
        # ``arc/output.py::_statmech_to_dict``). The container is omitted
        # entirely when nothing useful resolves — no empty containers.
        # Computed-species emits unscoped role keys (``opt`` / ``freq``
        # / ``sp``) directly as the bundle-local calc keys.
        species_calc_keys_by_role: dict[str, str] = {
            role: role for role in (_CALC_KEY_OPT, _CALC_KEY_FREQ, _CALC_KEY_SP)
            if role in included_keys
        }
        statmech_block = _build_statmech_block_for_species(
            output_doc=output_doc,
            species_record=species_record,
            calc_keys_by_role=species_calc_keys_by_role,
            workflow_tool_release=arc_wt,
        )
        if statmech_block is not None:
            bundle["statmech"] = statmech_block

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

        The conformer's reference xyz (the optimized geometry) is
        normalized once and threaded through to ``_build_calc_in_bundle``
        so freq + sp can declare it as their explicit input geometry —
        ARC's invariant guarantees they ran on this geometry. Server
        auto-fill would also reach the same answer for freq/sp, but
        being explicit makes the bundle self-describing.
        """
        # Required: the bundle's ConformerInBundle.geometry, also reused
        # as the explicit input_geometries entry on freq + sp.
        conformer_xyz_text = _require_xyz_text(species_record)

        # Coarse-opt provenance. Emitted only when ``coarse_opt_log``
        # exists AND ``coarse_opt_output_xyz`` is populated — the
        # output xyz is what chains the fine opt to the coarse opt
        # (it's both opt_coarse's output and the fine opt's declared
        # input). When the coarse log existed but its geometry didn't
        # parse cleanly, ``arc/output.py`` deliberately leaves
        # ``coarse_opt_output_xyz`` null so we fall back to single-stage
        # bundle shape rather than emit a half-described opt_coarse.
        opt_coarse_calc = self._build_opt_coarse_calc(
            output_doc=output_doc, species_record=species_record,
        )

        # Fine opt's depends_on: gain an ``optimized_from → opt_coarse``
        # edge when coarse was emitted. Otherwise no upstream calc.
        fine_opt_depends_on: list[Mapping[str, Any]] | None = None
        if opt_coarse_calc is not None:
            fine_opt_depends_on = [
                {"parent_calculation_key": _CALC_KEY_OPT_COARSE,
                 "role": "optimized_from"}
            ]

        primary_calc = self._build_calc_in_bundle(
            output_doc=output_doc,
            species_record=species_record,
            calc_key=_CALC_KEY_OPT,
            calc_type="opt",
            level_kind="opt",
            ess_job_key="opt",
            result_field="opt_result",
            result_payload=_opt_result_payload(species_record),
            depends_on=fine_opt_depends_on,
            tckdb_origin=None,
            conformer_xyz_text=conformer_xyz_text,
        )

        included: list[str] = [_CALC_KEY_OPT]
        additional: list[dict[str, Any]] = []
        # Coarse opt is an *additional* calc, not primary — fine opt is
        # the geometry of record. Order matters for thermo-source-link
        # determinism: coarse goes first so the bundle reads
        # opt → opt_coarse → freq → sp in additional_calculations.
        if opt_coarse_calc is not None:
            additional.append(opt_coarse_calc)
            included.append(_CALC_KEY_OPT_COARSE)

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
                    conformer_xyz_text=conformer_xyz_text,
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
                    conformer_xyz_text=conformer_xyz_text,
                ))
                included.append(_CALC_KEY_SP)
            except ValueError as exc:
                logger.warning(
                    "TCKDB computed-species: sp calculation skipped for label=%s: %s",
                    species_record.get("label"), exc,
                )

        # Rotor scans. ``arc/output.py`` populates the species record's
        # ``additional_calculations`` list with one entry per successful
        # 1D rotor whose log was parseable. Each entry already carries a
        # bundle-local ``key`` (``scan_rotor_<i>``) and a TCKDB-shaped
        # ``scan_result`` dict; the adapter only needs to layer level /
        # software / workflow_tool_release on top, with opt-level
        # fallback because rotors_dict has no per-scan level field. The
        # ``depends_on`` edge points back to opt — the scan is a series
        # of constrained reoptimizations from that geometry.
        for scan_entry in (species_record.get("additional_calculations") or []):
            if not isinstance(scan_entry, Mapping):
                continue
            if scan_entry.get("type") != _CALC_KEY_SCAN:
                continue
            scan_key = scan_entry.get("key")
            scan_result = scan_entry.get("scan_result")
            if not isinstance(scan_key, str) or not scan_key:
                continue
            if not isinstance(scan_result, Mapping):
                continue
            try:
                additional.append(self._build_calc_in_bundle(
                    output_doc=output_doc,
                    species_record=species_record,
                    calc_key=scan_key,
                    calc_type=_CALC_KEY_SCAN,
                    level_kind="opt",
                    ess_job_key="opt",
                    result_field="scan_result",
                    result_payload=scan_result,
                    depends_on=[{"parent_calculation_key": _CALC_KEY_OPT,
                                 "role": "scan_parent"}],
                    tckdb_origin=None,
                    conformer_xyz_text=conformer_xyz_text,
                    calc_role=_CALC_KEY_SCAN,
                    source_constraints=scan_entry.get("constraints"),
                ))
                included.append(scan_key)
            except ValueError as exc:
                logger.warning(
                    "TCKDB computed-species: scan calculation %s skipped for label=%s: %s",
                    scan_key, species_record.get("label"), exc,
                )

        block: dict[str, Any] = {
            "key": conformer_key,
            "geometry": {"xyz_text": conformer_xyz_text},
            "primary_calculation": primary_calc,
            "additional_calculations": additional,
        }
        label = species_record.get("label")
        if label:
            block["label"] = str(label)[:64]
        return included, block

    def _build_alt_conformer_blocks(
        self,
        *,
        output_doc: Mapping[str, Any],
        species_record: Mapping[str, Any],
        selected_xyz_text: str,
        selected_key: str,
    ) -> list[dict[str, Any]]:
        """Build additional ``ConformerInBundle`` dicts for screened conformers.

        ARC's conformer screen often optimizes several geometries at the
        same level before picking one as the conformer of record. The
        ``output.yml`` species record carries those alongside the
        selected one as ``conformers`` (xyz strings).

        For each unique alt geometry we emit a minimal
        ``ConformerInBundle``:

        * deterministic key ``alt0``, ``alt1``, … (indexed by position
          in the dedup'd input list, never by iteration order over an
          unsorted dict). Collisions with ``selected_key`` are resolved
          by skipping any candidate whose key would clash.
        * geometry pulled from ``conformers[i]`` after normalization.
        * a bare opt ``primary_calculation`` carrying the same level /
          software as the selected conformer's opt, with **no**
          ``opt_result`` (no absolute hartree we can attest to here)
          and **no** relative-energy annotation.

        Note on ``conformer_energies``: ARC's
        ``species.conformer_energies`` are workflow-local relative E0
        values (the reference is "lowest in this screening set", which
        depends on which conformers ARC chose to screen and on its
        ranking procedure). Those semantics do not survive a round-trip
        through TCKDB, which is intentionally workflow-tool agnostic
        and stores final scientific records — not ARC's internal
        ranking context. The adapter therefore IGNORES
        ``conformer_energies`` and never emits a relative-energy field
        on a screened conformer (not in ``note``, not in
        ``parameters_json``). The raw value still lives in ARC's
        artifacts for debugging.

        Skips: empty ``conformers`` list, candidates whose xyz fails to
        normalize, exact xyz duplicates of the selected geometry or of
        an earlier alt, and candidates whose minimal opt calc cannot be
        built (e.g. opt level not resolvable). Failures log + continue;
        they never fail the whole bundle.
        """
        raw_conformers = species_record.get("conformers")
        if not isinstance(raw_conformers, (list, tuple)) or not raw_conformers:
            return []

        label = species_record.get("label")
        opt_level = _resolve_level(output_doc, "opt")

        seen_xyz: set[str] = {selected_xyz_text}
        blocks: list[dict[str, Any]] = []
        for src_idx, raw_xyz in enumerate(raw_conformers):
            normalized = _normalize_xyz_text(raw_xyz, label)
            if normalized is None:
                continue
            if normalized in seen_xyz:
                continue
            seen_xyz.add(normalized)

            alt_idx = len(blocks)
            alt_key = f"alt{alt_idx}"
            if alt_key == selected_key:
                # Caller picked a selected key in our alt-key namespace
                # (e.g. conformer_index=0 → "alt0"). Skip rather than
                # silently shadow the selected entry — never produces
                # in practice with the conformer label scheme but keeps
                # the namespace contract loud.
                logger.warning(
                    "TCKDB computed-species: alt conformer #%d for label=%s "
                    "would collide with selected key %r; skipping.",
                    src_idx, label, selected_key,
                )
                continue

            try:
                opt_calc = self._calculation_payload(
                    output_doc, species_record,
                    calc_type="opt",
                    level=opt_level,
                    ess_job_key="opt",
                    result_field="opt_result",
                    result_payload=None,
                    # ``tckdb_origin`` lands under ``parameters_json``
                    # and tags the row as a screened-conformer anchor,
                    # NOT a parsed opt job. This is the loud signal —
                    # without it, an alt opt row would look like an
                    # ordinary opt that happened to lack result data.
                    tckdb_origin=_screened_conformer_origin(),
                )
            except ValueError as exc:
                logger.warning(
                    "TCKDB computed-species: alt conformer #%d for label=%s "
                    "skipped (opt calc not buildable): %s",
                    src_idx, label, exc,
                )
                continue

            opt_calc["key"] = f"{alt_key}_opt"
            # Anchor opt's output_geometries to the alt xyz explicitly:
            # backend's auto-fill would otherwise pin every conformer's
            # opt output to the selected conformer's geometry of record.
            opt_calc["output_geometries"] = [
                {"geometry": {"xyz_text": normalized}, "role": "final"},
            ]

            block: dict[str, Any] = {
                "key": alt_key,
                "geometry": {"xyz_text": normalized},
                "primary_calculation": opt_calc,
                "additional_calculations": [],
            }
            if label:
                block["label"] = str(label)[:64]
            # ``conformer_energies`` are workflow-local relative E0
            # values and are deliberately dropped at this seam — see the
            # method docstring. Do not re-introduce a ``note`` /
            # ``parameters_json`` carrier for them.

            blocks.append(block)

        return blocks

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
        conformer_xyz_text: str | None = None,
        calc_role: str | None = None,
        source_constraints: list | None = None,
    ) -> dict[str, Any]:
        """Build one CalculationInBundle dict.

        ``calc_key`` is the bundle-local identity (e.g. ``"opt"`` for a
        single-species bundle, ``"r0_opt"`` for a reaction bundle).
        ``calc_role`` is the chemistry role used for policy lookups
        (input/output geometry rules, log-field map). When ``calc_role``
        is ``None`` it defaults to ``calc_key`` — that's the
        single-species case where the two are identical and the existing
        callers don't have to change.

        Reuses :meth:`_calculation_payload` for the level/software/result
        plumbing, then layers on the bundle-specific fields: ``key``,
        ``depends_on``, ``input_geometries``, and inline ``artifacts``.

        Input-geometry policy (matches TCKDB v0 backend):
        - ``opt``: emit ``input_geometries`` only when ARC has the
          actual pre-opt xyz on the record (``opt_input_xyz``). Never
          fall back to the conformer's optimized geometry — that's
          opt's *output*, not its input. If absent, the field is
          omitted; backend has no auto-fill for opt.
        - ``freq`` / ``sp``: emit ``input_geometries`` set to the
          conformer's optimized geometry (passed in via
          ``conformer_xyz_text``). ARC's invariant guarantees these
          ran on that geometry. Backend would auto-fill the same value
          if we omitted, but explicit is better — keeps the bundle
          self-describing and removes ambiguity for any consumer that
          doesn't replicate the auto-fill rule.
        """
        level = _resolve_level(output_doc, level_kind)
        role = calc_role if calc_role is not None else calc_key
        final_settings = _final_settings_for_calc(
            species_record=species_record,
            calc_role=role,
        )
        calc = self._calculation_payload(
            output_doc, species_record,
            calc_type=calc_type,
            level=level,
            ess_job_key=ess_job_key,
            result_field=result_field,
            result_payload=result_payload,
            tckdb_origin=tckdb_origin,
            final_settings=final_settings,
        )
        calc["key"] = calc_key
        if depends_on:
            calc["depends_on"] = [dict(d) for d in depends_on]
        input_geometries = self._input_geometries_for_calc(
            calc_role=role,
            species_record=species_record,
            conformer_xyz_text=conformer_xyz_text,
        )
        if input_geometries:
            calc["input_geometries"] = input_geometries
        output_geometries = self._output_geometries_for_calc(
            calc_role=role,
            species_record=species_record,
            conformer_xyz_text=conformer_xyz_text,
        )
        if output_geometries:
            calc["output_geometries"] = output_geometries
        # Optional inline Cartesian Hessian for freq calcs (DR-0030). Parsed
        # from the freq job's ESS output; skipped silently when unavailable so
        # it can never break payload construction.
        if role == _CALC_KEY_FREQ:
            hessian = self._build_freq_hessian_payload(
                species_record=species_record,
                geometry_xyz_text=conformer_xyz_text,
            )
            if hessian is not None:
                calc["hessian"] = hessian
        artifacts = self._inline_artifacts_for_calc(species_record, calc_role=role)
        # Schema defaults `artifacts: []`. Emit explicitly only when we have
        # bytes to send (or when artifact upload is enabled and we want to
        # signal "no log available" with an empty list); omit otherwise.
        if artifacts:
            calc["artifacts"] = artifacts

        # Held-fixed coordinate constraints. ``source_constraints`` wins
        # when caller supplies it explicitly (scan calcs pull from the
        # ``additional_calculations`` entry). Otherwise we pick up the
        # per-job list ``arc/output.py`` populated on the species record.
        constraints_source = source_constraints
        if constraints_source is None:
            field = _CONSTRAINTS_FIELD_BY_CALC_KEY.get(role)
            if field:
                constraints_source = species_record.get(field)
        constraint_payload = _serialize_calc_constraints(constraints_source)
        if constraint_payload:
            calc["constraints"] = constraint_payload
        return calc

    @staticmethod
    def _output_geometries_for_calc(
        *,
        calc_role: str,
        species_record: Mapping[str, Any],
        conformer_xyz_text: str | None,
    ) -> list[dict[str, Any]]:
        """Compute the ``output_geometries`` list for one calc, per-kind policy.

        Each entry is shaped ``{"geometry": {"xyz_text": ...}, "role":
        "final"}`` per TCKDB's ``OutputGeometryEntry``. Returns ``[]``
        when nothing should be emitted; the caller drops empty lists.

        Policy mirrors what the calc actually produced:

        - ``opt`` (fine): produced the conformer's geometry of record.
          Emit ``conformer_xyz_text`` with ``role=final``. If absent
          (e.g., bundle-build path that skipped requiring xyz), the
          backend still has its single-stage fallback that links opt
          to the conformer geometry — but that fallback is server-
          side and shouldn't be relied on once we're declaring outputs
          explicitly elsewhere.
        - ``opt_coarse``: produced ``coarse_opt_output_xyz``. Emit it
          with ``role=final``.
        - ``freq`` / ``sp``: don't move atoms; we don't surface a
          standalone "freq output geometry" or "sp output geometry"
          today. Backend's freq/sp fallback now creates zero output
          rows for these (per the new contract), so omitting is correct.
        """
        if calc_role == _CALC_KEY_OPT:
            if not conformer_xyz_text:
                return []
            return [{"geometry": {"xyz_text": conformer_xyz_text}, "role": "final"}]
        if calc_role == _CALC_KEY_OPT_COARSE:
            coarse_out = species_record.get("coarse_opt_output_xyz")
            if not coarse_out:
                return []
            normalized = _normalize_xyz_text(coarse_out, species_record.get("label"))
            if not normalized:
                return []
            return [{"geometry": {"xyz_text": normalized}, "role": "final"}]
        # freq / sp / irc / others: no output_geometries today.
        return []

    @staticmethod
    def _input_geometries_for_calc(
        *,
        calc_role: str,
        species_record: Mapping[str, Any],
        conformer_xyz_text: str | None,
    ) -> list[dict[str, Any]]:
        """Compute the ``input_geometries`` list for one calc, per-kind policy.

        Returns ``[]`` when nothing should be emitted. The caller checks
        truthiness — empty lists are dropped from the payload so we don't
        send a zero-length array where None is more accurate.
        """
        if calc_role == _CALC_KEY_OPT:
            opt_input_xyz = species_record.get("opt_input_xyz")
            if not opt_input_xyz:
                return []
            normalized = _normalize_xyz_text(opt_input_xyz, species_record.get("label"))
            if not normalized:
                return []
            return [{"xyz_text": normalized}]
        if calc_role == _CALC_KEY_OPT_COARSE:
            # Coarse opt's input is the species' truly-initial xyz —
            # ``coarse_opt_input_xyz`` from arc/output.py. The caller
            # only invokes this when the coarse stage actually ran, so
            # an absent value here is a real bug; skip rather than
            # fabricate (the parent ``_build_opt_coarse_calc`` short-
            # circuits before reaching us if either coarse field is
            # missing).
            coarse_in = species_record.get("coarse_opt_input_xyz")
            if not coarse_in:
                return []
            normalized = _normalize_xyz_text(coarse_in, species_record.get("label"))
            return [{"xyz_text": normalized}] if normalized else []
        if calc_role in (_CALC_KEY_FREQ, _CALC_KEY_SP, _CALC_KEY_IRC):
            # ARC invariant: freq, sp, and (TS) irc all run on the
            # conformer's optimized xyz. Surface it explicitly rather
            # than relying on backend auto-fill — keeps the bundle
            # self-describing.
            if not conformer_xyz_text:
                return []
            return [{"xyz_text": conformer_xyz_text}]
        return []

    def _build_opt_coarse_calc(
        self,
        *,
        output_doc: Mapping[str, Any],
        species_record: Mapping[str, Any],
        calc_key: str = _CALC_KEY_OPT_COARSE,
    ) -> dict[str, Any] | None:
        """Build the coarse opt's CalculationInBundle dict, or return None.

        ``calc_key`` is the bundle-local identity. Computed-species uses
        the bare role (``"opt_coarse"``); computed-reaction passes a
        species-namespaced variant (``"r0_opt_coarse"``, ``"p1_opt_coarse"``)
        so it stays globally unique across the bundle. The chemistry
        role passed to ``_build_calc_in_bundle`` stays
        ``_CALC_KEY_OPT_COARSE`` either way — it's what drives the
        ``coarse_opt_log`` / ``coarse_opt_input_xyz`` lookups on the
        species record.

        Returns ``None`` when:
        - ``coarse_opt_log`` is absent (no coarse stage ran)
        - ``coarse_opt_output_xyz`` is null (coarse log existed but its
          final geometry couldn't be parsed; modeling the calc without
          its output geometry would create a half-described provenance
          row, so we drop the whole opt_coarse rather than mislead)

        Note on output geometry: TCKDB's bundle workflow auto-anchors
        every calc's ``CalculationOutputGeometry`` to the conformer
        geometry (the FINE opt's output). Until the bundle schema
        gains an ``output_geometries`` field on ``CalculationInBundle``,
        opt_coarse's output-geometry row will incorrectly point at
        the fine geometry server-side. The input chain
        (``calculation_input_geometry``) is correct — that's what was
        empty before this change and is what the producer can fix.
        """
        if not species_record.get("coarse_opt_log"):
            return None
        if not species_record.get("coarse_opt_output_xyz"):
            logger.debug(
                "TCKDB: coarse_opt_log present but coarse_opt_output_xyz "
                "is null for label=%s — falling back to single-stage opt "
                "bundle.",
                species_record.get("label"),
            )
            return None
        result_payload = _coarse_opt_result_payload(species_record)
        # Reuse the standard calc-in-bundle builder. opt_coarse takes
        # the same level/software as the fine opt (ARC runs both
        # stages at the configured opt level — the difference is
        # convergence criterion, not method/basis). No depends_on
        # (it's the chain head). No tckdb_origin (it's a real ESS run,
        # not a reuse of another calc's result).
        try:
            return self._build_calc_in_bundle(
                output_doc=output_doc,
                species_record=species_record,
                calc_key=calc_key,
                calc_role=_CALC_KEY_OPT_COARSE,
                calc_type="opt",
                level_kind="opt",
                ess_job_key="opt",
                result_field="opt_result",
                result_payload=result_payload,
                depends_on=None,
                tckdb_origin=None,
                conformer_xyz_text=None,  # opt_coarse's input is its own xyz, not the conformer
            )
        except ValueError as exc:
            logger.warning(
                "TCKDB: opt_coarse calculation skipped for label=%s "
                "(calc_key=%s): %s",
                species_record.get("label"), calc_key, exc,
            )
            return None

    def _inline_artifacts_for_calc(
        self,
        species_record: Mapping[str, Any],
        *,
        calc_role: str,
    ) -> list[dict[str, Any]]:
        """Return the inline artifact list for one calc within a bundle.

        ``calc_role`` is the chemistry role (``opt``/``freq``/``sp``/
        ``opt_coarse``/``irc``) used to look up the matching record
        field name (``opt_log`` etc.). The bundle-local key — e.g.
        ``r0_opt`` for a reaction — is irrelevant here; ARC stores the
        log paths on the species record under role-keyed names.

        Iterates ``_INLINE_ARTIFACT_SOURCES`` (currently ``output_log``
        and ``input``) and emits one ArtifactIn dict per kind whose
        record path resolves to a real file on disk. Each kind is
        independently gated on ``config.artifacts.kinds``, so a user
        can opt into logs but not decks (or vice versa).

        Skip rules per (calc_role, kind):
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
            # ``output_log`` for ``ts_guess`` is method-dispatched
            # (NEB → ``neb_log``, GSM → ``gsm_log``); other (kind, role)
            # combinations look up in the static map.
            if kind == "output_log":
                record_field = _resolve_log_field(calc_role, species_record)
            else:
                record_field = field_map.get(calc_role)
            artifact = self._read_inline_artifact(
                species_record,
                calc_role=calc_role,
                kind=kind,
                record_field=record_field,
            )
            if artifact is not None:
                artifacts.append(artifact)
        return artifacts

    def _read_inline_artifact(
        self,
        species_record: Mapping[str, Any],
        *,
        calc_role: str,
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
                "TCKDB bundle: %s %s artifact missing on disk for %s (path=%s)",
                calc_role, kind, species_record.get("label"), path_value,
            )
            return None
        size_bytes = resolved.stat().st_size
        max_bytes = self._config.artifacts.max_size_mb * 1024 * 1024
        if size_bytes > max_bytes:
            logger.warning(
                "TCKDB bundle: %s %s %s skipped (%s bytes > %s MB cap)",
                calc_role, kind, resolved.name, size_bytes,
                self._config.artifacts.max_size_mb,
            )
            return None
        with resolved.open("rb") as fh:
            content = fh.read()
        return {
            "kind": kind,
            "filename": _coerce_artifact_filename(resolved.name, kind),
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

    def _build_freq_hessian_payload(
        self,
        *,
        species_record: Mapping[str, Any],
        geometry_xyz_text: str | None,
    ) -> dict[str, Any] | None:
        """Build the optional ``hessian`` payload for a freq calculation.

        Parses the Cartesian Hessian from the freq job's ESS output (Gaussian
        log block or Orca sibling ``.hess`` file) via the parser-layer adapters,
        and packs it into the TCKDB ``HessianPayload`` shape:
        ``{"geometry": {"xyz_text": ...}, "lower_triangle_hartree_bohr2":
        [...], "source": "parsed_log"|"parsed_hess", "parser_version": ...}``.

        The Hessian is native atomic units (hartree/bohr²) straight from the
        parser — no unit conversion is applied here. ``geometry_xyz_text`` is
        the freq calc's input geometry (the conformer's optimized xyz, already
        normalized to standard XYZ), which is the configuration the Hessian was
        computed at.

        Returns ``None`` — never raises — when the Hessian is unavailable for
        any reason (no geometry, no freq log on disk, unknown/unsupported ESS,
        missing FC block / ``.hess``, monatomic species, or any parse error).
        The Hessian is strictly optional and must never break payload
        construction.
        """
        if not geometry_xyz_text:
            return None
        log_path = species_record.get(_LOG_FIELD_BY_CALC_KEY[_CALC_KEY_FREQ])
        if not log_path:
            return None
        resolved = self._resolve_local_path(log_path)
        if resolved is None or not resolved.is_file():
            return None
        try:
            # Late import: keeps the ESS parser adapters (heavyweight) off the
            # adapter's import path unless a Hessian is actually being built.
            from arc.parser.parser import determine_ess
            from arc.parser.factory import ess_factory

            ess_name = determine_ess(str(resolved), raise_error=False)
            source = _HESSIAN_SOURCE_BY_ESS.get(ess_name) if ess_name else None
            if source is None:
                return None
            ess_adapter = ess_factory(str(resolved), ess_name)
            parse = getattr(ess_adapter, "parse_cartesian_hessian_lower_triangle", None)
            if parse is None:
                return None
            triangle = parse()
        except Exception as exc:  # noqa: BLE001 — Hessian is best-effort only.
            logger.debug(
                "TCKDB bundle: Hessian parse skipped for label=%s (%s)",
                species_record.get("label"), exc,
            )
            return None
        if not triangle:
            return None
        return {
            "geometry": {"xyz_text": geometry_xyz_text},
            "lower_triangle_hartree_bohr2": triangle,
            "source": source,
            "parser_version": _HESSIAN_PARSER_VERSION,
        }

    def _parse_irc_trajectories(
        self,
        ts_record: Mapping[str, Any],
    ) -> list[dict[str, Any]] | None:
        """Parse each IRC log into a trajectory dict for payload assembly.

        Returns one dict per parsed log:
            ``{
                "direction": "forward"|"reverse"|None,
                "rich_points": [<rich-point dict>, ...] | None,
                "geom_points": [<xyz_dict>, ...] | None,
            }``

        Each log is first attempted with the rich parser
        (:func:`arc.parser.parser.parse_irc_path`) which carries energies,
        gradients, reaction coordinates, and per-point direction labels.
        On rich-parser failure (or for ESS backends that haven't
        implemented it), the geometry-only :func:`parse_irc_traj` runs
        as a fallback so the upload retains the geometry-only IRC payload
        behavior described in the task spec.

        Direction-resolution order (used when a per-point direction
        isn't present in the rich data):
        1. Rich parser's per-point ``direction`` (Gaussian's
           ``FORWARD/REVERSE path direction.`` announcement).
        2. ``ts_record['irc_log_directions'][i]`` — the authoritative
           value the scheduler captured from ``job.irc_direction`` and
           ``arc/output.py`` paired with ``irc_logs``. Production ARC
           IRC log filenames are just ``output.log`` inside an
           ``irc_<server_id>`` folder, so filename detection alone
           always returns None on real runs.
        3. ``_detect_irc_direction(filename)`` — back-compat fallback
           for output.yml files written before the paired-list
           tracking, and for test fixtures that hand-craft directional
           filenames.
        4. ``None`` — last resort. The trajectory is still emitted;
           per-point direction is omitted (the schema allows nullable
           ``IRCDirection`` on each point).

        Returns ``None`` when no logs resolve, no logs exist on disk, or
        every parse attempt failed. The caller uses ``None`` as the
        signal to omit ``irc_result`` (partial-data fallback per spec).
        """
        log_paths = ts_record.get("irc_logs") or []
        if not log_paths:
            return None
        # Lazy import: parser imports drag in heavyweight ESS adapters;
        # the adapter module is loaded even when IRC isn't in play.
        from arc.parser.parser import parse_irc_path, parse_irc_traj

        log_directions = list(ts_record.get("irc_log_directions") or [])
        trajectories: list[dict[str, Any]] = []
        for i, log_path in enumerate(log_paths):
            resolved = self._resolve_local_path(log_path)
            if resolved is None or not resolved.is_file():
                logger.debug(
                    "TCKDB computed-reaction: IRC log path not found on disk: %r",
                    log_path,
                )
                continue
            # Resolve the trajectory-level direction once: scheduler-
            # tracked first, filename heuristic second, None last. This
            # is used as a per-point fallback when the rich parser
            # doesn't carry FORWARD/REVERSE labels (currently only
            # Gaussian does).
            direction = log_directions[i] if i < len(log_directions) else None
            if direction not in (_IRC_DIRECTION_FORWARD, _IRC_DIRECTION_REVERSE):
                direction = _detect_irc_direction(str(log_path))
            rich_points: list[dict[str, Any]] | None = None
            try:
                rich_points = parse_irc_path(log_file_path=str(resolved))
            except Exception as exc:
                logger.debug(
                    "TCKDB computed-reaction: parse_irc_path failed for %s: %s",
                    resolved, exc,
                )
            geom_points: list[dict[str, Any]] | None = None
            if not rich_points:
                # Geometry-only fallback so non-Gaussian logs (and
                # malformed Gaussian logs) keep producing a usable IRC
                # payload, matching the pre-rich-parser behavior.
                try:
                    geom_points = parse_irc_traj(log_file_path=str(resolved))
                except Exception as exc:
                    logger.debug(
                        "TCKDB computed-reaction: parse_irc_traj failed for %s: %s",
                        resolved, exc,
                    )
                    continue
                if not geom_points:
                    continue
            trajectories.append({
                "direction": direction,
                "rich_points": rich_points,
                "geom_points": geom_points,
            })
        return trajectories or None

    # ------------------------------------------------------------------
    # Computed-reaction bundle path (POST /uploads/computed-reaction)
    # ------------------------------------------------------------------

    def submit_computed_reaction_from_output(
        self,
        *,
        output_doc: Mapping[str, Any],
        reaction_record: Mapping[str, Any],
        is_partial: bool = False,
    ) -> UploadOutcome | None:
        """Build, write, and (if configured) upload one computed-reaction bundle.

        Walks the reaction's ``reactant_labels``/``product_labels`` against
        ``output_doc['species']`` (and ``output_doc['transition_states']``
        for ``ts_label``) and assembles a self-contained
        ``ComputedReactionUploadRequest`` covering species, TS, and
        modified-Arrhenius kinetics — all cross-referenced by local
        string keys, no DB ids.

        ``is_partial=True`` marks the bundle as a deliberately
        incomplete record (phase-1 partial sidecar for a reaction whose
        TS search failed). The caller is responsible for stripping
        ``ts_label`` and ``kinetics`` from ``reaction_record`` before
        calling — the adapter does not edit the input. Partial bundles
        are sidecar-only in phase-1: the network POST is skipped
        regardless of ``config.upload``, the payload + sidecar land on
        disk with a ``.partial`` filename infix and ``is_partial=true``
        metadata, and the returned :class:`UploadOutcome` has
        ``status="skipped"``.

        Returns ``None`` if the adapter is disabled. Build failures (e.g.
        missing reactant species, missing opt level) raise; the caller is
        responsible for wrapping the per-reaction call in a try/except so
        one bad reaction doesn't take down the rest of the run.
        """
        if not self._config.enabled:
            return None

        reaction_label = reaction_record.get("label") or "unlabeled"
        project_label = self._config.project_label or output_doc.get("project")

        payload = self._build_computed_reaction_payload(
            output_doc=output_doc,
            reaction_record=reaction_record,
        )

        idempotency_inputs = IdempotencyInputs.from_payload(
            project_label=project_label,
            species_label=reaction_label,
            # The reaction has no conformer concept at the bundle level —
            # just one fit per upload — so the conformer slot in the key
            # carries the TS label (or "noTS"). This keeps the key shape
            # consistent with the species path while still uniquely
            # identifying the reaction within a project.
            conformer_label=str(reaction_record.get("ts_label") or "noTS"),
            payload_kind=COMPUTED_REACTION_KIND,
            payload=payload,
        )
        idempotency_key = build_idempotency_key(idempotency_inputs)

        written = self._writer.write(
            label=reaction_label,
            payload=payload,
            endpoint=COMPUTED_REACTION_ENDPOINT,
            idempotency_key=idempotency_key,
            payload_kind=COMPUTED_REACTION_KIND,
            base_url=self._config.base_url,
            subdir=PayloadWriter.COMPUTED_REACTION_SUBDIR,
            is_partial=is_partial,
        )
        if is_partial:
            # Phase-1 policy: partial reaction sidecars never POST. The
            # server has not been verified to accept transition_state=null
            # bundles, and partial-record semantics on the server side
            # are out of scope until a follow-up. Sidecar lands on disk
            # with the .partial infix; status reuses "skipped" because
            # is_partial=true on the sidecar already disambiguates this
            # from an upload=false skip.
            logger.info(
                "TCKDB partial computed-reaction sidecar written; "
                "live upload skipped: %s (key=%s)",
                written.payload_path,
                idempotency_key,
            )
            return self._finalize_skipped(written)

        logger.info(
            "TCKDB computed-reaction payload written: %s (key=%s)",
            written.payload_path,
            idempotency_key,
        )

        if not self._config.upload:
            return self._finalize_skipped(written)

        return self._upload(written, payload, endpoint=COMPUTED_REACTION_ENDPOINT)

    def _build_computed_reaction_payload(
        self,
        *,
        output_doc: Mapping[str, Any],
        reaction_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Compose one ``ComputedReactionUploadRequest`` dict.

        Resolves reactant/product/TS records from ``output_doc``,
        delegates per-actor block construction (species + TS) to the
        shared per-actor helpers, and stitches in a single
        modified-Arrhenius kinetics fit when ARC produced one.
        """
        species_index = _index_species(output_doc)
        ts_index = _index_transition_states(output_doc)

        reactant_labels = list(reaction_record.get("reactant_labels") or [])
        product_labels = list(reaction_record.get("product_labels") or [])
        if not reactant_labels:
            raise ValueError(
                f"reaction label={reaction_record.get('label')!r} has no reactant_labels."
            )
        if not product_labels:
            raise ValueError(
                f"reaction label={reaction_record.get('label')!r} has no product_labels."
            )

        # Build reactant/product species blocks under namespaced keys.
        species_blocks: list[dict[str, Any]] = []
        reactant_keys: list[str] = []
        product_keys: list[str] = []
        # Per-actor calc-role → bundle-key map; consumed by the kinetics
        # builder so source_calculations references resolve to the same
        # local keys the species blocks declared.
        actor_calc_keys: dict[str, dict[str, str]] = {}

        for i, label in enumerate(reactant_labels):
            actor_key = _local_key_for_actor("r", i, label)
            calc_prefix = _calc_prefix_for_actor("r", i)
            record = species_index.get(label)
            if record is None:
                raise ValueError(
                    f"reaction {reaction_record.get('label')!r}: reactant "
                    f"label {label!r} not found in output_doc.species."
                )
            block, calc_keys = self._build_reaction_species_block(
                output_doc=output_doc,
                species_record=record,
                actor_key=actor_key,
                calc_prefix=calc_prefix,
            )
            species_blocks.append(block)
            reactant_keys.append(actor_key)
            actor_calc_keys[actor_key] = calc_keys

        for j, label in enumerate(product_labels):
            actor_key = _local_key_for_actor("p", j, label)
            calc_prefix = _calc_prefix_for_actor("p", j)
            record = species_index.get(label)
            if record is None:
                raise ValueError(
                    f"reaction {reaction_record.get('label')!r}: product "
                    f"label {label!r} not found in output_doc.species."
                )
            block, calc_keys = self._build_reaction_species_block(
                output_doc=output_doc,
                species_record=record,
                actor_key=actor_key,
                calc_prefix=calc_prefix,
            )
            species_blocks.append(block)
            product_keys.append(actor_key)
            actor_calc_keys[actor_key] = calc_keys

        # TS block (inline). Optional — a reaction with no TS still
        # carries kinetics but server-side it's a thinner record.
        ts_label = reaction_record.get("ts_label")
        ts_block: dict[str, Any] | None = None
        ts_calc_keys: dict[str, str] = {}
        if ts_label:
            ts_record = ts_index.get(ts_label)
            if ts_record is None:
                raise ValueError(
                    f"reaction {reaction_record.get('label')!r}: ts_label "
                    f"{ts_label!r} not found in output_doc.transition_states."
                )
            ts_block, ts_calc_keys = self._build_ts_block(
                output_doc=output_doc,
                ts_record=ts_record,
                ts_label=ts_label,
                reaction_multiplicity=reaction_record.get("multiplicity"),
                unmapped_smiles=_ts_unmapped_smiles_handle(
                    ts_record=ts_record,
                    reaction_record=reaction_record,
                    species_index=species_index,
                ),
            )

        # Kinetics. ARC produces at most one fit per reaction today.
        kinetics_payload = reaction_record.get("kinetics")
        kinetics_blocks: list[dict[str, Any]] = []
        if isinstance(kinetics_payload, Mapping):
            kinetics_block = _build_kinetics_block(
                kinetics_record=kinetics_payload,
                reactant_keys=reactant_keys,
                product_keys=product_keys,
                actor_calc_keys=actor_calc_keys,
                ts_calc_keys=ts_calc_keys,
                long_kinetic_description=reaction_record.get(
                    "long_kinetic_description"
                ),
            )
            if kinetics_block is not None:
                kinetics_blocks.append(kinetics_block)

        bundle: dict[str, Any] = {
            "species": species_blocks,
            "reactant_keys": reactant_keys,
            "product_keys": product_keys,
        }
        # Emit ``reversible`` only when the producer set it explicitly.
        # ARCReaction has no first-class ``reversible`` attribute yet,
        # so this is normally ``None`` and we fall back to the schema's
        # default of True. The pass-through is here so a future ARC
        # change that sets ``reversible`` on the reaction (e.g., from an
        # RMG import) lands on the wire without a second adapter edit.
        reversible = reaction_record.get("reversible")
        if reversible is not None:
            bundle["reversible"] = bool(reversible)
        if ts_block is not None:
            bundle["transition_state"] = ts_block
        if kinetics_blocks:
            bundle["kinetics"] = kinetics_blocks

        # Final pass: flatten every calc's wrapped result into the
        # network_pdep flat fields the computed-reaction endpoint
        # expects. Keeping this as a single end-of-build walker means
        # any new calc-emit site automatically gets the right shape.
        _flatten_all_reaction_calcs(bundle)

        family = reaction_record.get("family")
        if family:
            bundle["reaction_family"] = str(family)
            # The server validates against a canonical-family list and
            # demands a source_note when the supplied name is unknown.
            # We don't have that list at the producer, so always tag
            # the source — a no-op for canonical names, a safety net
            # for non-canonical ones.
            bundle["reaction_family_source_note"] = "ARC-reported family"

        arc_version = output_doc.get("arc_version")
        arc_git_commit = output_doc.get("arc_git_commit")
        if arc_version or arc_git_commit:
            wt: dict[str, Any] = {"name": "ARC"}
            if arc_version:
                wt["version"] = str(arc_version)
            if arc_git_commit:
                wt["git_commit"] = str(arc_git_commit)
            bundle["workflow_tool_release"] = wt

        analysis_release = _arc_analysis_software_release(output_doc)
        if analysis_release is not None:
            bundle["analysis_software_release"] = analysis_release

        return bundle

    def _build_reaction_species_block(
        self,
        *,
        output_doc: Mapping[str, Any],
        species_record: Mapping[str, Any],
        actor_key: str,
        calc_prefix: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Build one ``BundleSpeciesIn`` dict + a calc-role → bundle-key map.

        ``actor_key`` (e.g. ``"r0_CHO"``) becomes the species block's
        ``key`` and the geometry/conformer key tails. ``calc_prefix``
        (e.g. ``"r0"``) is the namespace for calculation keys —
        deliberately shorter than ``actor_key`` so source_calculations
        references stay compact.

        The returned map (e.g. ``{"opt": "r0_opt", "freq": "r0_freq",
        "sp": "r0_sp"}``) is what the kinetics builder uses to wire
        ``source_calculations`` back to the freshly-minted local keys.
        It only contains the roles whose calculation actually made it
        into the bundle.
        """
        conformer_xyz_text = _require_xyz_text(species_record)
        opt_key = f"{calc_prefix}_{_CALC_KEY_OPT}"
        opt_coarse_key = f"{calc_prefix}_{_CALC_KEY_OPT_COARSE}"
        freq_key = f"{calc_prefix}_{_CALC_KEY_FREQ}"
        sp_key = f"{calc_prefix}_{_CALC_KEY_SP}"
        geom_key = f"{actor_key}_geom"
        conf_key = f"{actor_key}_conf0"

        # Coarse-opt provenance, parallel to the computed-species path.
        # Same gate (both ``coarse_opt_log`` and ``coarse_opt_output_xyz``
        # must be present) — when only one resolves, we fall back to
        # single-stage rather than emit a half-described opt_coarse.
        # Namespacing the calc key with ``calc_prefix`` keeps it globally
        # unique across reactant/product/TS, which the bundle's
        # ``validate_unique_keys`` requires.
        opt_coarse_calc = self._build_opt_coarse_calc(
            output_doc=output_doc,
            species_record=species_record,
            calc_key=opt_coarse_key,
        )
        fine_opt_depends_on: list[Mapping[str, Any]] | None = None
        if opt_coarse_calc is not None:
            fine_opt_depends_on = [
                {"parent_calculation_key": opt_coarse_key,
                 "role": "optimized_from"}
            ]

        primary_calc = self._build_calc_in_bundle(
            output_doc=output_doc,
            species_record=species_record,
            calc_key=opt_key,
            calc_role=_CALC_KEY_OPT,
            calc_type="opt",
            level_kind="opt",
            ess_job_key="opt",
            result_field="opt_result",
            result_payload=_opt_result_payload(species_record),
            depends_on=fine_opt_depends_on,
            tckdb_origin=None,
            conformer_xyz_text=conformer_xyz_text,
        )

        calc_keys: dict[str, str] = {_CALC_KEY_OPT: opt_key}
        additional: list[dict[str, Any]] = []
        # opt_coarse is type=opt, so the schema validator at
        # computed_reaction_upload.py:446 exempts it from needing
        # geometry_key — leave it bare.
        if opt_coarse_calc is not None:
            additional.append(opt_coarse_calc)
            calc_keys[_CALC_KEY_OPT_COARSE] = opt_coarse_key

        freq_result = _freq_result_payload(species_record)
        if freq_result is not None:
            try:
                freq_calc = self._build_calc_in_bundle(
                    output_doc=output_doc,
                    species_record=species_record,
                    calc_key=freq_key,
                    calc_role=_CALC_KEY_FREQ,
                    calc_type="freq",
                    level_kind="freq",
                    ess_job_key="freq",
                    result_field="freq_result",
                    result_payload=freq_result,
                    depends_on=[{"parent_calculation_key": opt_key, "role": "freq_on"}],
                    tckdb_origin=None,
                    conformer_xyz_text=conformer_xyz_text,
                )
                # Server requires non-opt species calcs to reference a
                # conformer geometry by key (BundleSpeciesIn validator
                # validate_calc_geometry_keys).
                freq_calc["geometry_key"] = geom_key
                additional.append(freq_calc)
                calc_keys[_CALC_KEY_FREQ] = freq_key
            except ValueError as exc:
                logger.warning(
                    "TCKDB computed-reaction: %s freq calculation skipped: %s",
                    actor_key, exc,
                )

        sp_result = _sp_result_payload(species_record)
        if sp_result is not None:
            try:
                sp_calc = self._build_calc_in_bundle(
                    output_doc=output_doc,
                    species_record=species_record,
                    calc_key=sp_key,
                    calc_role=_CALC_KEY_SP,
                    calc_type="sp",
                    level_kind="sp",
                    ess_job_key="sp",
                    result_field="sp_result",
                    result_payload=sp_result,
                    depends_on=[{"parent_calculation_key": opt_key, "role": "single_point_on"}],
                    tckdb_origin=(
                        _reused_origin("opt") if _sp_is_reused_from_opt(output_doc) else None
                    ),
                    conformer_xyz_text=conformer_xyz_text,
                )
                sp_calc["geometry_key"] = geom_key
                additional.append(sp_calc)
                calc_keys[_CALC_KEY_SP] = sp_key
            except ValueError as exc:
                logger.warning(
                    "TCKDB computed-reaction: %s sp calculation skipped: %s",
                    actor_key, exc,
                )

        # Rotor scans. Mirrors ``_build_conformer_block``'s scan loop,
        # with two reaction-path-specific deltas:
        #
        #  - Calc keys are namespaced (``r0_scan_rotor_0``, etc.). The
        #    bundle schema's ``validate_unique_keys`` requires *globally*
        #    unique calc keys across all species + the TS, so two
        #    reactants both reporting ``scan_rotor_0`` would collide.
        #    The torsion's ``source_scan_calculation_key`` gets rewritten
        #    in lockstep via ``scan_key_renames`` so the reference still
        #    resolves.
        #  - ``geometry_key`` is set: non-opt species calcs in reaction
        #    bundles must point at the conformer geometry (same reason
        #    freq/sp set it above).
        scan_key_renames: dict[str, str] = {}
        for scan_entry in (species_record.get("additional_calculations") or []):
            if not isinstance(scan_entry, Mapping):
                continue
            if scan_entry.get("type") != _CALC_KEY_SCAN:
                continue
            original_scan_key = scan_entry.get("key")
            scan_result = scan_entry.get("scan_result")
            if not isinstance(original_scan_key, str) or not original_scan_key:
                continue
            if not isinstance(scan_result, Mapping):
                continue
            namespaced_scan_key = f"{calc_prefix}_{original_scan_key}"
            try:
                scan_calc = self._build_calc_in_bundle(
                    output_doc=output_doc,
                    species_record=species_record,
                    calc_key=namespaced_scan_key,
                    calc_role=_CALC_KEY_SCAN,
                    calc_type=_CALC_KEY_SCAN,
                    level_kind="opt",
                    ess_job_key="opt",
                    result_field="scan_result",
                    result_payload=scan_result,
                    depends_on=[{"parent_calculation_key": opt_key,
                                 "role": "scan_parent"}],
                    tckdb_origin=None,
                    conformer_xyz_text=conformer_xyz_text,
                    source_constraints=scan_entry.get("constraints"),
                )
                scan_calc["geometry_key"] = geom_key
                additional.append(scan_calc)
                scan_key_renames[original_scan_key] = namespaced_scan_key
            except ValueError as exc:
                logger.warning(
                    "TCKDB computed-reaction: %s scan calculation %s skipped: %s",
                    actor_key, original_scan_key, exc,
                )

        species_block: dict[str, Any] = {
            "key": actor_key,
            "species_entry": self._species_entry_payload(species_record),
            "conformers": [
                {
                    "key": conf_key,
                    "geometry": {"key": geom_key, "xyz_text": conformer_xyz_text},
                    "calculation": primary_calc,
                }
            ],
            "calculations": additional,
        }
        thermo_block = _build_thermo_block(
            species_record.get("thermo"),
            included_calc_keys=[
                # Pass the bundle-local keys so source_calculations links
                # match the calc keys actually emitted in this species
                # block. _build_thermo_block expects role names — we
                # pre-translate to keys via the role→key map below.
                # Order kept deterministic: opt, freq, sp.
                calc_keys[role]
                for role in (_CALC_KEY_OPT, _CALC_KEY_FREQ, _CALC_KEY_SP)
                if role in calc_keys
            ],
        )
        if thermo_block is not None:
            species_block["thermo"] = thermo_block

        # Per-species AEC/BAC corrections target this species's resolved
        # species_entry. Anchor each correction to this species's own SP
        # calc (e.g. r0_sp / p1_sp); never to ts_sp or to a sibling
        # species's SP key — the server enforces ownership in
        # ``_persist_species_applied_corrections`` and would 422 on a
        # cross-species reference.
        applied_corrections = _build_applied_energy_corrections(
            species_record.get("applied_energy_corrections") or [],
            source_calculation_key=calc_keys.get(_CALC_KEY_SP),
        )
        if applied_corrections:
            species_block["applied_energy_corrections"] = applied_corrections

        # Statmech block: carries frequency-scale-factor provenance plus
        # the schema-supported base statmech fields for this reactant/
        # product (external_symmetry, is_linear, rigid_rotor_kind,
        # statmech_treatment, point_group, slim torsions, and
        # source_calculations referencing this species's own opt/freq/
        # sp calcs). ``BundleStatmechIn`` accepts the same field set as
        # ``StatmechInBundle``, so the shared builder produces the same
        # shape; the only mode-specific input is the calc-key namespace
        # — we pass the *species-scoped* ``calc_keys`` (e.g.
        # ``{"opt": "r0_opt", ...}``) so the server-side ownership check
        # sees only this species's own calculations. Sibling species
        # and the TS use disjoint namespaces (``r1_*``/``p0_*``/``ts_*``)
        # and therefore can't leak in here. The helper returns ``None``
        # (and we skip the whole statmech block) when nothing useful
        # resolves, keeping payloads backward-compatible with FSF-less
        # runs.
        species_statmech = _build_statmech_block_for_species(
            output_doc=output_doc,
            species_record=species_record,
            calc_keys_by_role=calc_keys,
            workflow_tool_release=_arc_workflow_tool_release(output_doc),
            scan_key_renames=scan_key_renames or None,
        )
        if species_statmech is not None:
            species_block["statmech"] = species_statmech

        return species_block, calc_keys

    def _build_ts_block(
        self,
        *,
        output_doc: Mapping[str, Any],
        ts_record: Mapping[str, Any],
        ts_label: str,
        reaction_multiplicity: int | None,
        unmapped_smiles: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Build one ``BundleTransitionStateIn`` dict + a calc-role → key map.

        ARC stores TS records in ``output_doc['transition_states']`` with
        the same shape as species records (the TS is just a stationary
        point with ``is_ts: true``); the multiplicity falls back to the
        reaction's multiplicity when the TS record doesn't carry one.

        ``unmapped_smiles`` is a deterministic textual handle for the
        TS, computed by :func:`_ts_unmapped_smiles_handle` at the
        caller. ``None`` is the common case (TS has no Lewis structure
        and no upstream textual identifier was derivable); the field is
        omitted from the payload, leaving the server to store NULL.
        The producer never fabricates a normal-molecule SMILES from
        the TS geometry — the ``mol`` field stays absent regardless.

        IRC provenance: when ARC has ``irc_logs`` populated, emit a
        ``ts_irc`` calc with ``depends_on(role=irc_start)`` pointing at
        ``ts_opt`` — IRC is seeded from the optimized TS saddle, so
        ``ts_opt`` is its primary geometry-producing parent (the TS freq
        validates the saddle but isn't the geometry source). When the
        IRC log files parse cleanly, attach a structured ``irc_result``
        with forward/reverse points and producer-declared
        ``output_geometries`` (``irc_forward``/``irc_reverse`` endpoints).
        Forward/reverse are ESS path-direction labels — the producer
        does NOT infer reactant/product side from them.
        """
        conformer_xyz_text = _require_xyz_text(ts_record)
        ts_opt_key = f"ts_{_CALC_KEY_OPT}"
        # ``_CALC_KEY_TS_GUESS`` already starts with ``ts_``; using it
        # bare keeps the bundle key as ``ts_guess`` (don't double-prefix).
        ts_guess_key = _CALC_KEY_TS_GUESS
        ts_freq_key = f"ts_{_CALC_KEY_FREQ}"
        ts_sp_key = f"ts_{_CALC_KEY_SP}"
        ts_irc_key = f"ts_{_CALC_KEY_IRC}"
        ts_geom_key = "ts_geom"

        # Path-search ts_guess provenance. Emit a parent calculation
        # only when the chosen TS guess is itself a real path-search
        # calculation (orca_neb → method=neb, xtb_gsm → method=gsm) AND
        # the producer recorded the corresponding log path
        # (``neb_log`` / ``gsm_log``). Heuristics / AutoTST / KinBot /
        # GCN / user-supplied guesses stay geometry-only on the
        # ts_opt: ``calculation_dependency`` requires a real parent
        # calculation, never a geometry. The artifact path picks up
        # the log via ``_resolve_log_field`` when artifact upload is
        # enabled.
        ts_guess_calc: dict[str, Any] | None = None
        ts_guess_method = _resolve_ts_guess_path_search(
            ts_record.get("chosen_ts_method"),
        )
        ts_guess_log_field = (
            _TS_GUESS_LOG_FIELD_BY_METHOD.get(ts_guess_method)
            if ts_guess_method else None
        )
        if (
            ts_guess_method
            and ts_guess_log_field
            and ts_record.get(ts_guess_log_field)
        ):
            # Resolve the on-disk log path (gsm_log/neb_log are stored
            # run-relative on the record) so the trajectory parser can
            # read it. Falls back to ``opt_input_xyz`` for the single-
            # point branch when the trajectory parser is unavailable
            # (NEB today; any GSM run whose stringfile didn't parse).
            log_local = self._resolve_local_path(
                ts_record.get(ts_guess_log_field),
            )
            # The patched ``ograd`` writes preserved per-node files
            # into ``<gsm_run_dir>/gsm_node_outputs/`` — the run
            # directory is the parent of the stringfile log.
            node_outputs_dir: Path | None = None
            if log_local is not None:
                candidate = log_local.parent / "gsm_node_outputs"
                if candidate.is_dir():
                    node_outputs_dir = candidate
            path_search_payload = _build_path_search_result_payload(
                method=ts_guess_method,
                log_path=str(log_local) if log_local is not None else None,
                fallback_xyz_text=ts_record.get("opt_input_xyz"),
                node_outputs_dir=node_outputs_dir,
            )
            if path_search_payload is None:
                logger.warning(
                    "TCKDB computed-reaction: ts_guess (%s) calc skipped — "
                    "could not build path_search_result.points (no parseable "
                    "log and no fallback xyz).",
                    ts_guess_method,
                )
            else:
                try:
                    ts_guess_calc = self._build_calc_in_bundle(
                        output_doc=output_doc,
                        species_record=ts_record,
                        calc_key=ts_guess_key,
                        calc_role=_CALC_KEY_TS_GUESS,
                        calc_type="path_search",
                        # No ts_guess_level in output_doc today; fall back
                        # to opt_level (consistent with how irc resolves).
                        level_kind="opt",
                        ess_job_key="opt",
                        result_field="path_search_result",
                        result_payload=path_search_payload,
                        depends_on=None,
                        tckdb_origin=None,
                        conformer_xyz_text=None,
                    )
                except ValueError as exc:
                    logger.warning(
                        "TCKDB computed-reaction: ts_guess (%s) calc skipped: %s",
                        ts_guess_method, exc,
                    )

        ts_opt_depends_on: list[Mapping[str, Any]] | None = None
        if ts_guess_calc is not None:
            ts_opt_depends_on = [
                {"parent_calculation_key": ts_guess_key,
                 "role": "optimized_from"}
            ]

        # ARC's selected TS-guess method (heuristics, AutoTST, KinBot,
        # GCN, user, …) is workflow narrative — TCKDB has no
        # schema-reviewed slot for "what generated the guess that fed
        # ts_opt". The optimized TS itself (geometry, freq imag mode,
        # IRC connectivity to reactant/product) is the scientific
        # evidence the database stores; the upstream guess source is
        # ARC-internal context that lives in ARC's artifacts. Do not
        # re-introduce ``selected_ts_guess`` as a ``tckdb_origin``
        # marker on ts_opt unless TCKDB grows a dedicated field.
        # NEB/GSM is a separate case: ``path_search_result`` ships
        # actual scientific path-search data (per-image energies,
        # geometries) on the ts_guess calc, accepted by the schema.
        primary_calc = self._build_calc_in_bundle(
            output_doc=output_doc,
            species_record=ts_record,
            calc_key=ts_opt_key,
            calc_role=_CALC_KEY_OPT,
            calc_type="opt",
            level_kind="opt",
            ess_job_key="opt",
            result_field="opt_result",
            result_payload=_opt_result_payload(ts_record),
            depends_on=ts_opt_depends_on,
            tckdb_origin=None,
            conformer_xyz_text=conformer_xyz_text,
        )

        calc_keys: dict[str, str] = {_CALC_KEY_OPT: ts_opt_key}
        additional: list[dict[str, Any]] = []
        # Order: ts_guess (parent) first, then freq/sp/irc descend
        # from ts_opt. ``BundleTransitionStateIn`` doesn't enforce a
        # geometry_key on additional calcs (unlike the species-side
        # validator), so type=path_search stays bare.
        if ts_guess_calc is not None:
            additional.append(ts_guess_calc)
            calc_keys[_CALC_KEY_TS_GUESS] = ts_guess_key

        freq_result = _freq_result_payload(ts_record)
        if freq_result is not None:
            try:
                additional.append(self._build_calc_in_bundle(
                    output_doc=output_doc,
                    species_record=ts_record,
                    calc_key=ts_freq_key,
                    calc_role=_CALC_KEY_FREQ,
                    calc_type="freq",
                    level_kind="freq",
                    ess_job_key="freq",
                    result_field="freq_result",
                    result_payload=freq_result,
                    depends_on=[{"parent_calculation_key": ts_opt_key, "role": "freq_on"}],
                    tckdb_origin=None,
                    conformer_xyz_text=conformer_xyz_text,
                ))
                calc_keys[_CALC_KEY_FREQ] = ts_freq_key
            except ValueError as exc:
                logger.warning("TCKDB computed-reaction: ts freq calc skipped: %s", exc)

        sp_result = _sp_result_payload(ts_record)
        if sp_result is not None:
            try:
                additional.append(self._build_calc_in_bundle(
                    output_doc=output_doc,
                    species_record=ts_record,
                    calc_key=ts_sp_key,
                    calc_role=_CALC_KEY_SP,
                    calc_type="sp",
                    level_kind="sp",
                    ess_job_key="sp",
                    result_field="sp_result",
                    result_payload=sp_result,
                    depends_on=[{"parent_calculation_key": ts_opt_key, "role": "single_point_on"}],
                    tckdb_origin=(
                        _reused_origin("opt") if _sp_is_reused_from_opt(output_doc) else None
                    ),
                    conformer_xyz_text=conformer_xyz_text,
                ))
                calc_keys[_CALC_KEY_SP] = ts_sp_key
            except ValueError as exc:
                logger.warning("TCKDB computed-reaction: ts sp calc skipped: %s", exc)

        # IRC: emit a calc when ARC has irc_logs populated. Always carry
        # depends_on(role=irc_start) → ts_opt, because IRC is seeded from
        # the optimized TS saddle. When the logs parse cleanly, attach a
        # structured irc_result with forward/reverse points and producer-
        # declared output_geometries for the irc_forward/irc_reverse
        # endpoints. When parsing fails we still emit the type=irc calc
        # (so kinetics.source_calculations(role=irc) and the dependency
        # edge can stand on their own) but omit irc_result rather than
        # fabricating incomplete structured data.
        if ts_record.get("irc_logs"):
            try:
                irc_calc = self._build_calc_in_bundle(
                    output_doc=output_doc,
                    species_record=ts_record,
                    calc_key=ts_irc_key,
                    calc_role=_CALC_KEY_IRC,
                    calc_type="irc",
                    level_kind="opt",  # ARC runs IRC at the opt level
                    ess_job_key="opt",
                    result_field=None,
                    result_payload=None,
                    depends_on=[{"parent_calculation_key": ts_opt_key, "role": "irc_start"}],
                    tckdb_origin=None,
                    conformer_xyz_text=conformer_xyz_text,
                )
                # IRC output geometries are derived by the server from
                # ``irc_result.points``: ``_persist_irc_result`` writes a
                # ``calculation_output_geometry`` row (role=irc_forward /
                # irc_reverse) for every forward/reverse point. Emitting
                # explicit ``output_geometries`` here would double-claim
                # those geometries and the
                # ``attach_calculation_output_geometries`` uniqueness check
                # would 422. We therefore attach only ``irc_result`` and
                # let the server own the output-geometry links.
                irc_parsed = self._parse_irc_trajectories(ts_record)
                if irc_parsed is not None:
                    zero_ref = _resolve_irc_zero_energy_reference(
                        output_doc=output_doc,
                        ts_record=ts_record,
                    )
                    # Build the optional TS marker from the seed-saddle
                    # record. ``conformer_xyz_text`` is the same TS xyz
                    # the TS block already uses for ``ts_opt``'s output
                    # geometry, so the marker points at the canonical
                    # saddle — no separate xyz_to_str pass needed.
                    ts_marker: dict[str, Any] | None = None
                    if conformer_xyz_text:
                        ts_marker = {
                            "xyz_text": conformer_xyz_text,
                            # Pair the marker's energy with the same
                            # zero_ref the per-point relatives are
                            # computed against — keeps relative_energy
                            # internally consistent (TS marker → 0).
                            "electronic_energy_hartree": zero_ref,
                        }
                    irc_result = _build_irc_result_payload(
                        irc_parsed,
                        zero_energy_reference_hartree=zero_ref,
                        ts_marker=ts_marker,
                    )
                    if irc_result is not None:
                        irc_calc["irc_result"] = irc_result
                additional.append(irc_calc)
                calc_keys[_CALC_KEY_IRC] = ts_irc_key
            except ValueError as exc:
                logger.debug("TCKDB computed-reaction: ts irc calc skipped: %s", exc)

        # TS multiplicity: the TS record's own field is authoritative;
        # fall back to the reaction's multiplicity when missing (some
        # ARC runs only carry it on the reaction).
        ts_mult = ts_record.get("multiplicity")
        if ts_mult is None:
            ts_mult = reaction_multiplicity
        if ts_mult is None:
            raise ValueError(
                f"transition state {ts_label!r} has no multiplicity and the "
                "reaction record has none either; cannot build TS block."
            )

        ts_block: dict[str, Any] = {
            "charge": int(ts_record.get("charge", 0) or 0),
            "multiplicity": int(ts_mult),
            "geometry": {"key": ts_geom_key, "xyz_text": conformer_xyz_text},
            "calculation": primary_calc,
            "calculations": additional,
            "label": str(ts_label)[:64],
        }
        if unmapped_smiles:
            ts_block["unmapped_smiles"] = str(unmapped_smiles)

        # TS-side AEC/BAC corrections target the TS entry directly via
        # ``target_transition_state_entry_id`` server-side — never via the
        # reaction entry. Anchor to ts_sp specifically; species-side SP
        # keys (r0_sp / p1_sp) belong to other species and a cross-owner
        # reference would 422.
        applied_corrections = _build_applied_energy_corrections(
            ts_record.get("applied_energy_corrections") or [],
            source_calculation_key=calc_keys.get(_CALC_KEY_SP),
        )
        if applied_corrections:
            ts_block["applied_energy_corrections"] = applied_corrections

        return ts_block, calc_keys

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
        smiles_text = str(smiles)
        is_ts = bool(record.get("is_ts"))
        # ARC has no excited-state workflow, so every uploaded species_entry
        # is asserted as the electronic ground state. Sending this explicitly
        # (rather than relying on the server's column default) keeps the
        # payload self-describing for replay tooling and protects against
        # silent server-default drift. The other identity-disambiguation
        # fields (stereo_label, electronic_state_label, term_symbol[_raw],
        # isotopologue_label) are intentionally left null: ARC has no
        # reliable source for them, the TCKDB backend derives stereo_label
        # from the uploaded 3D geometry when applicable, and the dedupe
        # uniqueness key uses ``nulls_not_distinct`` so paired nulls collapse
        # to the same row rather than fragmenting it.
        entry: dict[str, Any] = {
            "molecule_kind": "molecule",
            "smiles": smiles_text,
            "charge": int(record.get("charge", 0) or 0),
            "multiplicity": int(record.get("multiplicity", 1) or 1),
            "species_entry_kind": "transition_state" if is_ts else "minimum",
            "electronic_state_kind": "ground",
        }
        # Optional ``unmapped_smiles``: TCKDB carries a free-form
        # alternate identity string for cases where the primary
        # ``smiles`` is something special (e.g. an atom-mapped form
        # from a future producer path). The adapter only forwards an
        # explicit value the producer surfaces — it never derives by
        # string manipulation, never strips atom maps from arbitrary
        # strings. Omitted when missing/empty/whitespace, and also
        # omitted when identical to the main ``smiles`` (the schema
        # would accept the duplicate but storing the same string twice
        # adds no identity information).
        unmapped_raw = record.get("unmapped_smiles")
        if isinstance(unmapped_raw, str):
            unmapped_text = unmapped_raw.strip()
            if unmapped_text and unmapped_text != smiles_text:
                entry["unmapped_smiles"] = unmapped_text
        return entry

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
        final_settings: Mapping[str, Any] | None = None,
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

        level_of_theory = _arc_level_to_tckdb_lot(level)
        if level_of_theory is None:
            # method was already validated above; this is defensive against
            # a future change to _arc_level_to_tckdb_lot that drops the row.
            raise ValueError(
                f"could not project level of theory for {calc_type} onto "
                "TCKDB LevelOfTheoryRef shape."
            )

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

        # ``parameters_json`` is the single per-calc slot for free-form
        # qualifier metadata. Two writers feed it: ``tckdb_origin``
        # (provenance qualifier — reused-result / screened-conformer)
        # and ``final_settings`` (the calc's effective scientific
        # settings, e.g. ``{"fine": True}``, distinct from LoT-identity
        # keywords). The merge helper drops the field entirely when
        # both sources are empty.
        parameters_json = _merge_parameters_json(
            tckdb_origin=tckdb_origin,
            final_settings=final_settings,
        )
        if parameters_json is not None:
            calc["parameters_json"] = parameters_json

        return calc

    # ------------------------------------------------------------------
    # Upload + sidecar finalization
    # ------------------------------------------------------------------

    def _finalize_skipped(self, written: WrittenPayload) -> UploadOutcome:
        sc = written.sidecar
        sc.status = "skipped"
        self._writer.update_sidecar(written.sidecar_path, sc)
        # Two distinct reasons land here: (a) ``upload=false`` for a
        # complete bundle, and (b) phase-1 partial sidecars that are
        # intentionally never live-POSTed. Pick the right log line so
        # operators reading logs don't have to grep ``is_partial`` to
        # reverse-engineer the cause.
        if sc.is_partial:
            logger.info(
                "TCKDB partial sidecar finalized (status=skipped, is_partial=true; "
                "phase-1 policy: partial bundles are sidecar-only, never POSTed): %s",
                written.payload_path,
            )
        else:
            logger.info(
                "TCKDB upload skipped (upload=false): %s", written.payload_path,
            )
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
                f"TCKDB API key not configured (tried: "
                f"{self._config.describe_api_key_sources()}); "
                "skipping network call and recording sidecar as failed."
            )
            return self._record_failure(written, msg, raised=ValueError(msg))

        try:
            client = self._make_client(api_key)
        except Exception as exc:  # pragma: no cover - defensive
            return self._record_failure(written, f"client init failed: {exc}", exc)

        try:
            self._ensure_ready(client)
            _attach_preflight(sc, self._preflight_metadata)
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
        sc.public_refs = _extract_tckdb_public_refs(response_data)
        _append_request_id(sc, "upload", response)
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
        # Pull structured HTTP details off the exception when present
        # (TCKDBClient attaches ``status_code``, ``response_json``, and
        # ``response_text`` on its HTTP-error subclasses). FastAPI 422
        # bodies otherwise collapse to "HTTP 422" via the exception's
        # default message, hiding the field-level rejection reason.
        # Non-HTTP failures (timeouts, connection errors, etc.) lack
        # these attrs and fall back to the original ``message`` —
        # preserving the prior log shape for those paths.
        status_code = getattr(raised, "status_code", None)
        response_json = getattr(raised, "response_json", None)
        response_text = getattr(raised, "response_text", None)
        if response_json is not None:
            sc.response_body = _summarize_response_body(response_json)
        elif response_text:
            sc.response_body = _summarize_response_body(response_text)
        if status_code is not None:
            sc.response_status_code = status_code
        if isinstance(raised, TCKDBReadinessError):
            sc.preflight = {
                "ready": False,
                "status_code": status_code,
                "request_id": _request_id_from(raised),
            }
            _append_request_id(sc, "readyz", raised)
        else:
            _append_request_id(sc, "upload", raised)
        detail_for_message: Any = (
            response_json
            if response_json is not None
            else (response_text or message)
        )
        sc.last_error = (
            f"HTTP {status_code}: {detail_for_message}"
            if status_code is not None
            else message
        )
        self._writer.update_sidecar(written.sidecar_path, sc)
        logger.warning(
            "TCKDB upload failed (strict=%s): %s key=%s err=%s",
            self._config.strict,
            written.payload_path,
            sc.idempotency_key,
            sc.last_error,
        )
        if isinstance(raised, TCKDBReadinessError):
            self._log_readiness_recovery()
        if self._config.strict:
            raise raised
        return UploadOutcome(
            status="failed",
            payload_path=written.payload_path,
            sidecar_path=written.sidecar_path,
            idempotency_key=sc.idempotency_key,
            error=sc.last_error,
        )

    def _log_readiness_recovery(self) -> None:
        """Tell the user the payloads are on disk and how to re-run manually.

        A readiness failure means the science is done and the payload was
        written — only the network POST didn't happen. The upload is fully
        replayable via the standalone CLI once the server is back, so emit
        the exact command on its own line for copy-paste. The input.yml
        path isn't in scope at this layer, so we give the invariant form
        (``<project_dir>/input.yml``, the CLI's default location) with the
        real project directory and upload mode we do have.
        """
        payload_dir = self._writer.root
        project_dir = self._project_directory
        mode = self._config.upload_mode
        if project_dir is not None:
            input_ref = f"{project_dir}/input.yml"
            recovery_cmd = (
                f"python -m arc.tckdb.cli {input_ref} "
                f"-p {project_dir} --upload-mode {mode}"
            )
        else:
            # No project directory in scope — give the invariant shape with
            # a clear placeholder rather than a wrong absolute path.
            recovery_cmd = (
                f"python -m arc.tckdb.cli <input.yml> --upload-mode {mode}"
            )
        logger.warning(
            "TCKDB server was not ready after %d attempts; payloads were "
            "written to %s. Re-run the upload once it is up:\n%s",
            PREFLIGHT_MAX_ATTEMPTS, payload_dir, recovery_cmd,
        )

    def _make_client(self, api_key: str):
        if self._client_factory is not None:
            return self._client_factory(self._config, api_key)
        return TCKDBClient(
            self._config.base_url,
            api_key=api_key,
            timeout=self._config.timeout_seconds,
        )

    def _ensure_ready(self, client: Any) -> None:
        """Run the optional TCKDB readyz preflight once per adapter instance.

        The probe is retried up to :data:`PREFLIGHT_MAX_ATTEMPTS` times
        with exponential backoff (see the module constants) so a transient
        not-ready blip mid-run doesn't drop an upload. Both failure modes
        are retried: a request exception (server unreachable / 5xx) and a
        200 body that reports not-ready. The first attempt that reports
        ready wins; only after all attempts are exhausted is
        ``self._preflight_error`` set and raised. The single-check-per-
        instance semantics are preserved — the whole retry loop runs once,
        and the final error is cached so later calls re-raise it cheaply.
        """
        if not self._config.preflight:
            return
        if self._preflight_error is not None:
            raise self._preflight_error
        if self._preflight_checked:
            return
        self._preflight_checked = True

        last_error: TCKDBReadinessError | None = None
        for attempt in range(PREFLIGHT_MAX_ATTEMPTS):
            try:
                response = client.request_json("GET", "/readyz", authenticated=False)
            except TCKDBError as exc:
                # Server unreachable / timeout / 5xx during a blip. Build
                # the readiness error now so, if this is the last attempt,
                # we raise a message consistent with the not-ready path.
                last_error = _build_readiness_error(exc)
            else:
                data = getattr(response, "data", None)
                ready = _readyz_body_is_ready(data)
                metadata = {
                    "ready": ready,
                    "status_code": getattr(response, "status_code", None),
                    "request_id": _request_id_from(response),
                }
                if isinstance(data, Mapping):
                    for key in ("status", "code", "alembic_revision"):
                        if data.get(key) is not None:
                            metadata[key] = data.get(key)
                self._preflight_metadata = metadata
                if ready:
                    return
                last_error = TCKDBReadinessError(
                    _format_readiness_message(
                        status_code=metadata.get("status_code"),
                        body=data,
                        request_id=metadata.get("request_id"),
                    ),
                    status_code=metadata.get("status_code"),
                    response_json=data if isinstance(data, Mapping) else None,
                    response_text=None if isinstance(data, Mapping) else str(data),
                    headers=getattr(response, "headers", None),
                )

            # Not ready (error or not-ready body). Back off before the
            # next attempt unless this was the final one.
            if attempt < PREFLIGHT_MAX_ATTEMPTS - 1:
                delay = min(
                    PREFLIGHT_BASE_DELAY_SECONDS * (2 ** attempt),
                    PREFLIGHT_MAX_DELAY_SECONDS,
                )
                logger.info(
                    "TCKDB /readyz not ready (attempt %d/%d); retrying in %.1fs.",
                    attempt + 1, PREFLIGHT_MAX_ATTEMPTS, delay,
                )
                _preflight_sleep(delay)

        # All attempts exhausted. ``last_error`` is always set here because
        # every loop iteration that reaches this point set it.
        self._preflight_error = last_error
        raise self._preflight_error

    def _prepare_artifact_upload(
        self,
        *,
        output_doc: Mapping[str, Any],
        species_label: str,
        calculation_id: int,
        kind: str,
        file_path: str | Path,
        artifact_cfg: Any,
    ) -> _PreparedArtifactUpload | ArtifactUploadOutcome:
        # Defense-in-depth: in bundle modes the bundle payload already
        # carries input/output_log artifacts inline under each calc.
        if (
            self._config.upload_mode in _BUNDLE_MODES_WITH_INLINE_ARTIFACTS
            and kind in IMPLEMENTED_ARTIFACT_KINDS
        ):
            return _skip(
                calculation_id, kind,
                f"upload_mode={self._config.upload_mode!r} carries kind={kind!r} "
                "inline in the bundle; standalone artifact upload suppressed",
            )

        if not artifact_cfg.upload:
            return _skip(calculation_id, kind, "artifacts.upload is False")
        if kind not in artifact_cfg.kinds:
            return _skip(calculation_id, kind, f"kind {kind!r} not in config.kinds")
        if kind not in IMPLEMENTED_ARTIFACT_KINDS:
            return _skip(
                calculation_id, kind,
                f"kind {kind!r} is server-accepted but ARC has no upload path yet",
            )

        resolved = self._resolve_local_path(file_path)
        if resolved is None or not resolved.is_file():
            return _skip(calculation_id, kind, f"file missing: {file_path!r}")

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

        project_label = self._config.project_label or output_doc.get("project")
        idempotency_key = build_artifact_idempotency_key(ArtifactIdempotencyInputs(
            project_label=project_label,
            species_label=species_label,
            calculation_id=calculation_id,
            artifact_kind=kind,
            artifact_sha256=sha256,
        ))
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
        return _PreparedArtifactUpload(
            written=written_artifact,
            calculation_key=f"{kind}-{sha256[:16]}",
            calculation_id=calculation_id,
            path=resolved,
            kind=kind,
            label=species_label,
            sha256=sha256,
            bytes=size_bytes,
            filename=resolved.name,
        )

    def _upload_artifact_batch(
        self,
        *,
        prepared: list[_PreparedArtifactUpload],
        idempotency_key_prefix: str,
    ) -> list[ArtifactUploadOutcome]:
        api_key = self._config.resolve_api_key()
        if not api_key:
            msg = (
                f"TCKDB API key not configured (tried: "
                f"{self._config.describe_api_key_sources()}); "
                "skipping artifact network call."
            )
            return self._record_artifact_batch_failure(prepared, msg, ValueError(msg))

        try:
            client = self._make_client(api_key)
        except Exception as exc:  # pragma: no cover - defensive
            return self._record_artifact_batch_failure(
                prepared, f"client init failed: {exc}", exc
            )

        try:
            self._ensure_ready(client)
            for item in prepared:
                _attach_preflight(item.written.sidecar, self._preflight_metadata)
            batch_results = client.upload_artifacts(
                prepared,
                idempotency_key_prefix=idempotency_key_prefix,
                batch_by_calculation=True,
            )
        except (AttributeError, TypeError) as exc:
            _close_quietly(client, "after artifact upload failure")
            msg = (
                "Installed tckdb-client does not support "
                "batch_by_calculation artifact uploads. Upgrade tckdb-client "
                "to the version expected by this ARC branch."
            )
            return self._record_artifact_batch_failure(prepared, msg, exc)
        except Exception as exc:
            _close_quietly(client, "after artifact upload failure")
            return self._record_artifact_batch_failure(prepared, str(exc), exc)
        else:
            _close_quietly(client, "after artifact upload success")

        batch_summary = _summarize_artifact_batch_results(batch_results)
        outcomes: list[ArtifactUploadOutcome] = []
        for item in prepared:
            sc = item.written.sidecar
            sc.status = "uploaded"
            sc.uploaded_at = _utcnow_iso()
            sc.response_status_code = _artifact_batch_status_code(batch_results)
            sc.response_body = _summarize_response_body(batch_summary)
            sc.public_refs = _extract_tckdb_public_refs(batch_summary)
            for result in _artifact_batch_result_items(batch_results):
                _append_request_id(sc, "artifact_upload", result)
            sc.idempotency_replayed = _artifact_batch_replayed(batch_results)
            sc.last_error = None
            self._writer.update_artifact_sidecar(item.written.sidecar_path, sc)
            logger.info(
                "TCKDB artifact batch upload succeeded: calc=%s kind=%s key=%s",
                sc.calculation_id, sc.kind, sc.idempotency_key,
            )
            outcomes.append(ArtifactUploadOutcome(
                status="uploaded",
                sidecar_path=item.written.sidecar_path,
                idempotency_key=sc.idempotency_key,
                calculation_id=sc.calculation_id,
                kind=sc.kind,
                response=sc.response_body,
            ))
        return outcomes

    def _record_artifact_batch_failure(
        self,
        prepared: list[_PreparedArtifactUpload],
        message: str,
        raised: BaseException,
    ) -> list[ArtifactUploadOutcome]:
        outcomes = [
            self._record_artifact_failure(item.written, message, raised, raise_on_strict=False)
            for item in prepared
        ]
        if self._config.strict:
            raise raised
        return outcomes

    def _record_artifact_failure(
        self,
        written: WrittenArtifact,
        message: str,
        raised: BaseException,
        *,
        raise_on_strict: bool = True,
    ) -> ArtifactUploadOutcome:
        sc = written.sidecar
        sc.status = "failed"
        sc.last_error = message
        status_code = getattr(raised, "status_code", None)
        response_json = getattr(raised, "response_json", None)
        response_text = getattr(raised, "response_text", None)
        if status_code is not None:
            sc.response_status_code = status_code
        if response_json is not None:
            sc.response_body = _summarize_response_body(response_json)
        elif response_text:
            sc.response_body = _summarize_response_body(response_text)
        if isinstance(raised, TCKDBReadinessError):
            sc.preflight = {
                "ready": False,
                "status_code": getattr(raised, "status_code", None),
                "request_id": _request_id_from(raised),
            }
            _append_request_id(sc, "readyz", raised)
        else:
            _append_request_id(sc, "artifact_upload", raised)
        self._writer.update_artifact_sidecar(written.sidecar_path, sc)
        logger.warning(
            "TCKDB artifact upload failed (strict=%s): calc=%s kind=%s err=%s",
            self._config.strict, sc.calculation_id, sc.kind, message,
        )
        if self._config.strict and raise_on_strict:
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


def _artifact_batch_digest(items: list[_PreparedArtifactUpload]) -> str:
    """Content digest for a calculation-scoped artifact batch idempotency key."""
    h = hashlib.sha256()
    for item in items:
        h.update(str(item.calculation_id).encode("utf-8"))
        h.update(b"\0")
        h.update(item.kind.encode("utf-8"))
        h.update(b"\0")
        h.update(item.filename.encode("utf-8"))
        h.update(b"\0")
        h.update(item.sha256.encode("ascii"))
        h.update(b"\0")
        h.update(str(item.bytes).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _artifact_batch_idempotency_prefix(
    project_label: Any,
    species_label: Any,
) -> str:
    """Stable prefix consumed by ``client.upload_artifacts`` batch mode."""
    def clean(value: Any, *, limit: int) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._:-]+", "-", str(value or "unknown"))
        cleaned = cleaned.strip(":-.") or "unknown"
        return cleaned[:limit]

    return f"arc:{clean(project_label, limit=48)}:{clean(species_label, limit=64)}:artifact"


def _summarize_artifact_batch_results(batch_results: Any) -> Any:
    """Make tckdb-client ``ArtifactUploadBatchResult`` objects JSON-shaped."""
    if isinstance(batch_results, list):
        summarized = []
        for result in batch_results:
            response = getattr(result, "response", result)
            response_data = getattr(response, "data", response)
            summarized.append({
                "calculation_id": getattr(result, "calculation_id", None),
                "calculation_keys": list(getattr(result, "calculation_keys", ()) or ()),
                "artifact_count": getattr(result, "artifact_count", None),
                "response": response_data,
            })
        return summarized
    return batch_results


def _artifact_batch_status_code(batch_results: Any) -> int | None:
    if not isinstance(batch_results, list):
        return getattr(batch_results, "status_code", None)
    for result in batch_results:
        response = getattr(result, "response", None)
        status_code = getattr(result, "status_code", None)
        if status_code is None:
            status_code = getattr(response, "status_code", None)
        if status_code is not None:
            return status_code
    return None


def _artifact_batch_replayed(batch_results: Any) -> bool | None:
    if not isinstance(batch_results, list):
        replayed = getattr(batch_results, "idempotency_replayed", None)
        return bool(replayed) if replayed is not None else None
    replay_values: list[bool] = []
    for result in batch_results:
        response = getattr(result, "response", None)
        replayed = getattr(result, "idempotency_replayed", None)
        if replayed is None:
            replayed = getattr(response, "idempotency_replayed", None)
        if replayed is not None:
            replay_values.append(bool(replayed))
    return any(replay_values) if replay_values else None


def _artifact_batch_result_items(batch_results: Any) -> list[Any]:
    return batch_results if isinstance(batch_results, list) else [batch_results]


def _headers_from(obj: Any) -> Mapping[str, str] | None:
    headers = getattr(obj, "headers", None)
    if headers is None:
        response = getattr(obj, "response", None)
        headers = getattr(response, "headers", None)
    return headers if isinstance(headers, Mapping) else None


def _request_id_from(obj: Any) -> str | None:
    headers = _headers_from(obj)
    if not headers:
        return None
    for name, value in headers.items():
        if str(name).lower() == "x-request-id" and value:
            return str(value)
    return None


def _append_request_id(sidecar: Any, operation: str, obj: Any) -> None:
    request_id = _request_id_from(obj)
    if not request_id:
        return
    entry = {
        "operation": operation,
        "request_id": request_id,
        "status_code": getattr(obj, "status_code", None),
    }
    if entry["status_code"] is None:
        response = getattr(obj, "response", None)
        entry["status_code"] = getattr(response, "status_code", None)
    sidecar.request_ids.append(entry)


def _attach_preflight(sidecar: Any, metadata: dict[str, Any] | None) -> None:
    if metadata is None:
        return
    sidecar.preflight = dict(metadata)
    request_id = metadata.get("request_id")
    if request_id:
        sidecar.request_ids.append({
            "operation": "readyz",
            "request_id": request_id,
            "status_code": metadata.get("status_code"),
        })


def _preflight_sleep(seconds: float) -> None:
    """Sleep between readiness-probe retries.

    Thin indirection over :func:`time.sleep` so tests can patch out the
    real backoff wait (``mock.patch('arc.tckdb.adapter._preflight_sleep')``)
    and run instantly without changing the retry logic.
    """
    time.sleep(seconds)


def _readyz_body_is_ready(body: Any) -> bool:
    if not isinstance(body, Mapping):
        return False
    if body.get("ready") is True:
        return True
    status = body.get("status")
    return isinstance(status, str) and status.lower() in {"ready", "ok"}


def _build_readiness_error(exc: BaseException) -> TCKDBReadinessError:
    response_json = getattr(exc, "response_json", None)
    response_text = getattr(exc, "response_text", None)
    status_code = getattr(exc, "status_code", None)
    headers = getattr(exc, "headers", None)
    return TCKDBReadinessError(
        _format_readiness_message(
            status_code=status_code,
            body=response_json if response_json is not None else response_text,
            request_id=_request_id_from(exc),
        ),
        status_code=status_code,
        response_json=response_json,
        response_text=response_text,
        headers=headers,
    )


def _format_readiness_message(
    *,
    status_code: Any,
    body: Any,
    request_id: Any,
) -> str:
    status = None
    code = None
    if isinstance(body, Mapping):
        status = body.get("status")
        code = body.get("code")
    parts = ["TCKDB readiness check failed before upload:"]
    if status_code is not None:
        parts.append(f"status_code={status_code}")
    if status is not None:
        parts.append(f"status={status}")
    if code is not None:
        parts.append(f"code={code}")
    if request_id is not None:
        parts.append(f"request_id={request_id}")
    if len(parts) == 1 and body:
        parts.append(str(body))
    return " ".join(parts)


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


# TCKDB's ``CalculationWithResultsPayload.origin_kind`` is a *validated*
# enum. The backend hoists ``parameters_json.tckdb_origin.origin_kind``
# into that top-level field (that is why the ``reused_result`` marker,
# which happens to be a valid member, round-trips cleanly), so any value
# ARC emits for ``origin_kind`` — even nested under the "free-form"
# parameters_json — MUST be one of these. ARC-specific provenance detail
# that is NOT an enum member (e.g. "screened_conformer") is carried on a
# separate ``origin_detail`` key, which the backend leaves opaque.
VALID_TCKDB_ORIGIN_KINDS: frozenset[str] = frozenset(
    {"executed", "reused_result", "imported", "derived"}
)


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


# Per-calc-role mapping to the ``<role>_final_settings`` field name on the
# species record. Roles not in the map have no current producer-side
# source of final-settings data; the helper returns ``None`` for them
# and the adapter omits ``parameters_json.final_settings`` from the calc.
_FINAL_SETTINGS_FIELD_BY_CALC_ROLE: Mapping[str, str] = {
    "opt": "opt_final_settings",
    "opt_coarse": "coarse_opt_final_settings",
    "freq": "freq_final_settings",
    "sp": "sp_final_settings",
    "irc": "irc_final_settings",
}


def _final_settings_for_calc(
    *,
    species_record: Mapping[str, Any],
    calc_role: str,
) -> dict[str, Any] | None:
    """Return the calc's ``final_settings`` dict from the species record.

    ``final_settings`` carries the final effective scientific/numerical
    knobs that defined this calculation — distinct from
    ``level_of_theory.keywords`` (which carries LoT-identity settings)
    and from operational/scheduler metadata (server, queue, runtime,
    job ids — explicitly out of scope).

    Today the producer (``arc/output.py``) populates these honestly
    from observable run state (e.g. ``opt_final_settings.fine = True``
    when a coarse stage ran, since that proves the fine opt ran with
    ``fine=True``). Calls return ``None`` when:
    * the calc role has no producer-side source today,
    * the field is missing from the species record (older output.yml),
    * the field is present but empty / not a mapping.
    """
    field = _FINAL_SETTINGS_FIELD_BY_CALC_ROLE.get(calc_role)
    if field is None:
        return None
    raw = species_record.get(field)
    if not isinstance(raw, Mapping) or not raw:
        return None
    return {k: v for k, v in raw.items() if v is not None}


def _merge_parameters_json(
    *,
    existing: Mapping[str, Any] | None = None,
    tckdb_origin: Mapping[str, Any] | None = None,
    final_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Merge optional calculation metadata into ``parameters_json``.

    ``parameters_json`` is the single calc-level slot TCKDB exposes for
    free-form qualifier metadata (per-row, opaque to the schema). Today
    two writers populate it:

    * ``tckdb_origin`` — origin/qualifier markers (see :func:`_reused_origin`,
      :func:`_screened_conformer_origin`). Identifies non-standard rows
      so consumers don't mistake them for independently executed jobs.
    * ``final_settings`` — the final effective scientific/numerical
      settings of the calculation (e.g. ``{"fine": True}``). Distinct
      from ``level_of_theory.keywords``, which carries LoT-identity
      settings that participate in dedup hashing.

    ``existing`` lets callers fold both writers on top of a caller-
    provided base. Keys collide explicitly: the merge intentionally
    does NOT silently overwrite a pre-existing ``tckdb_origin`` /
    ``final_settings`` — that would mask bugs. Callers should pass
    only one source for each key.

    Returns ``None`` when the merged dict is empty, so the caller can
    omit the field entirely rather than emit an empty container.
    """
    merged: dict[str, Any] = dict(existing) if existing else {}
    if tckdb_origin:
        merged["tckdb_origin"] = dict(tckdb_origin)
    if final_settings:
        merged["final_settings"] = dict(final_settings)
    return merged or None


def _screened_conformer_origin() -> dict[str, Any]:
    """Build the ``tckdb_origin`` payload for an alt-conformer's opt row.

    TCKDB's ``ConformerInBundle`` requires every conformer to carry a
    ``primary_calculation`` of type ``opt``. For screened conformers
    that ARC didn't select, no parsed ESS opt log lands on the species
    record — we only have a converged geometry plus a relative kJ/mol
    energy from the conformer screen. This marker makes that explicit
    on the wire so downstream consumers can tell screened-conformer
    anchor rows apart from independently executed opt jobs.

    ``origin_kind`` maps to ``derived``: the backend hoists this value
    into the validated ``CalculationWithResultsPayload.origin_kind`` enum
    (``{executed, reused_result, imported, derived}``), and a screened
    conformer's geometry is *derived* from ARC's conformer screen rather
    than produced by an independently executed ESS opt job. The
    ARC-specific ``screened_conformer`` distinction is preserved verbatim
    under ``origin_detail`` (opaque to the backend enum) so downstream
    consumers can still single these rows out. Emitting
    ``origin_kind="screened_conformer"`` here previously caused a 422 —
    it is not an enum member.
    """
    return {
        "origin_kind": "derived",
        "origin_detail": "screened_conformer",
        "reason": (
            "alt conformer geometry anchored to the bundle; ARC did "
            "not parse an independent opt job for this conformer"
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
    # opt_converged is emitted by arc/output.py::_spc_to_dict (line 541) and
    # accepted by TCKDB's OptResultPayload as ``converged: bool | None``.
    # Without this mapping the calc_opt_result.converged column lands NULL
    # even for known-converged species. Coerce to bool defensively — ARC
    # writes True/False but the schema is strict about the type.
    if record.get("opt_converged") is not None:
        out["converged"] = bool(record["opt_converged"])
    if record.get("opt_n_steps") is not None:
        out["n_steps"] = record["opt_n_steps"]
    if record.get("opt_final_energy_hartree") is not None:
        out["final_energy_hartree"] = record["opt_final_energy_hartree"]
    return out or None


def _coarse_opt_result_payload(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Like :func:`_opt_result_payload` but reads the ``coarse_opt_*`` fields.

    The coarse opt is treated as ``converged=True`` whenever it ran to
    completion — output.yml only records ``coarse_opt_log`` after a
    successful coarse run (see ``_spc_to_dict`` line 555: ``if converged
    and coarse_path``). A separate ``coarse_opt_converged`` field would
    be slightly more honest, but ARC doesn't currently emit one and the
    convention "the coarse log only exists if it converged" holds.
    """
    out: dict[str, Any] = {}
    # The presence of the coarse log indicates a successful coarse stage.
    out["converged"] = True
    if record.get("coarse_opt_n_steps") is not None:
        out["n_steps"] = record["coarse_opt_n_steps"]
    if record.get("coarse_opt_final_energy_hartree") is not None:
        out["final_energy_hartree"] = record["coarse_opt_final_energy_hartree"]
    return out


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
    statmech = record.get("statmech") or {}
    raw_freqs = statmech.get("harmonic_frequencies_cm1")
    has_modes_source = bool(raw_freqs)
    if not has_modes_source and all(
        record.get(rkey) is None for rkey, _, _ in _FREQ_FIELD_SPECS
    ):
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
    if has_modes_source:
        try:
            modes = []
            for f in raw_freqs:
                freq = float(f)
                modes.append({
                    "frequency_cm1": freq,
                    "is_imaginary": freq < 0,
                })
        except (TypeError, ValueError) as exc:
            logger.warning(
                "TCKDB freq additional calculation skipped for label=%s: "
                "malformed harmonic_frequencies_cm1=%r (%s)",
                record.get("label"), raw_freqs, exc,
            )
            return None
        # ARC's statmech ``harmonic_frequencies_cm1`` lists only the REAL
        # vibrational modes; a transition state's imaginary (reaction-
        # coordinate) frequency is carried separately in ``imag_freq_cm1``.
        # Re-insert it as an imaginary mode so ``modes`` is internally
        # consistent with ``n_imag`` — the TCKDB FreqResultPayload validator
        # requires count(is_imaginary) == n_imag.
        n_imag = out.get("n_imag")
        imag = record.get("imag_freq_cm1")
        if n_imag and imag is not None and not any(
            m["is_imaginary"] for m in modes
        ):
            try:
                modes.insert(0, {
                    "frequency_cm1": -abs(float(imag)),
                    "is_imaginary": True,
                })
            except (TypeError, ValueError):
                pass
        imag_count = sum(1 for m in modes if m["is_imaginary"])
        if n_imag is not None and imag_count != n_imag:
            # Cannot reconcile modes with n_imag (e.g. a higher-order saddle
            # with a single stored imag_freq_cm1). Emit the scalar
            # n_imag/imag_freq_cm1 without ``modes`` rather than a payload the
            # backend validator would reject.
            logger.warning(
                "TCKDB freq modes omitted for label=%s: could not reconcile "
                "n_imag=%s with %d imaginary mode(s) from statmech.",
                record.get("label"), n_imag, imag_count,
            )
        else:
            for i, m in enumerate(modes, start=1):
                m["mode_index"] = i
            out["modes"] = modes
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


_APPLIED_CORRECTION_COMPONENT_FIELDS = (
    "component_kind", "key", "multiplicity", "parameter_value", "contribution_value",
)

# TCKDB's LevelOfTheoryRef primary-key set — used by _level_keys_match
# for conservative LoT equality. Note the names here are TCKDB's
# (post-translation), not ARC's (output.yml writes `auxiliary_basis` /
# `cabs`); _level_keys_match compares projected dicts.
_TCKDB_LOT_REF_FIELDS = ("method", "basis", "aux_basis", "cabs_basis")

# Field-name translation from ARC's Level (output.yml shape) to TCKDB's
# LevelOfTheoryRef. ARC's Level.as_dict() emits `auxiliary_basis` / `cabs` /
# `solvation_method`; TCKDB's LoT uses `aux_basis` / `cabs_basis` /
# `solvent_model`. `software` / `software_version` belong on
# `software_release`, not the LoT, and are intentionally dropped here.
# `method_type` / `year` / `solvation_scheme_level` / `compatible_ess` have
# no TCKDB LoT counterpart and are also dropped.
_ARC_TO_TCKDB_LOT_FIELDS = {
    "method": "method",
    "basis": "basis",
    "auxiliary_basis": "aux_basis",
    "cabs": "cabs_basis",
    "dispersion": "dispersion",
    "solvent": "solvent",
    "solvation_method": "solvent_model",
}


def _arc_args_to_keywords(args: Any) -> str | None:
    """Flatten ARC's nested ``args`` dict to TCKDB's flat ``keywords`` string.

    ARC stores ESS/runtime options as a nested mapping, commonly with
    categories such as ``keyword`` and ``block``. TCKDB stores the projected
    level-of-theory options in a single deterministic string that participates
    in ``lot_hash`` deduplication.

    The serialization is intentionally category-prefixed and sorted so that
    equivalent dictionaries produce identical strings regardless of insertion
    order.
    """
    if not isinstance(args, Mapping):
        return None

    parts: list[str] = []

    for category in sorted(args):
        entries = args.get(category)
        if not isinstance(entries, Mapping) or not entries:
            continue

        for key in sorted(entries):
            value = entries[key]
            if value is None:
                continue

            value_text = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            parts.append(f"{category}:{key}={value_text}")

    if not parts:
        return None

    return "; ".join(parts)


def _arc_level_to_tckdb_lot(level: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Project ARC's per-job level dict (output.yml shape) onto TCKDB's
    ``LevelOfTheoryRef`` shape, applying field-name translation and
    flattening ``args`` into ``keywords``.

    Returns ``None`` if ``level`` is missing or has no ``method`` —
    callers decide whether to error or skip.
    """
    if not isinstance(level, Mapping):
        return None
    if not level.get("method"):
        return None
    out: dict[str, Any] = {}
    for src, dst in _ARC_TO_TCKDB_LOT_FIELDS.items():
        v = level.get(src)
        if v:
            out[dst] = str(v)
    keywords = _arc_args_to_keywords(level.get("args"))
    if keywords:
        out["keywords"] = keywords
    return out


def _scheme_level_of_theory(scheme: Mapping[str, Any]) -> dict[str, Any] | None:
    """Project ARC's per-species scheme.level_of_theory dict onto TCKDB's
    ``LevelOfTheoryRef`` shape. The scheme dict comes from the same
    ``_level_to_dict(arkane_level_of_theory)`` producer as opt/freq/sp
    levels, so the same field-name translation applies."""
    return _arc_level_to_tckdb_lot(scheme.get("level_of_theory"))


def _build_applied_energy_corrections(
    applied_records: Any,
    *,
    source_calculation_key: str | None = None,
) -> list[dict[str, Any]]:
    """Translate ``output.yml`` per-species ``applied_energy_corrections``
    into the TCKDB ``AppliedEnergyCorrectionUploadPayload`` shape.

    The output.yml shape is already very close to the upload schema (same
    ``application_role`` / ``value`` / ``value_unit`` / ``scheme`` fields),
    so this is mostly a passthrough plus three small adaptations:

    1. Drop output.yml-only fields from components (e.g. ``parameter_unit``)
       — TCKDB's ``AppliedCorrectionComponentPayload`` rejects unknowns.
    2. Drop component rows whose ``parameter_value`` is null — the upload
       schema requires a real number, and the producer marks them null
       precisely when reconstruction wasn't reliable.
    3. Attach ``source_calculation_key`` when the caller has resolved it
       to a real SP key in this bundle. AEC and BAC are corrections to the
       electronic-energy reference, so SP is the right anchor; the field
       is omitted when no key is supplied rather than guessed.

    The caller resolves the SP key against the relevant namespace —
    bundle-global (``"sp"``) for computed-species, scoped (``"r0_sp"`` /
    ``"p0_sp"`` / ``"ts_sp"``) for computed-reaction. The helper does
    not know about modes; it just takes the resolved key (or ``None``)
    and stamps it on every emitted entry.
    """
    if not isinstance(applied_records, list):
        return []

    out: list[dict[str, Any]] = []
    for rec in applied_records:
        if not isinstance(rec, Mapping):
            continue
        if rec.get("value") is None or rec.get("scheme") is None:
            continue

        components_in = rec.get("components") or []
        components_out: list[dict[str, Any]] = []
        for c in components_in:
            if not isinstance(c, Mapping):
                continue
            if c.get("parameter_value") is None or c.get("contribution_value") is None:
                continue
            components_out.append({
                k: c[k] for k in _APPLIED_CORRECTION_COMPONENT_FIELDS if k in c
            })

        scheme_in = rec["scheme"]
        scheme_out = {k: scheme_in[k] for k in scheme_in if k != "level_of_theory"}
        lot_ref = _scheme_level_of_theory(scheme_in)
        if lot_ref is not None:
            scheme_out["level_of_theory"] = lot_ref

        payload: dict[str, Any] = {
            "application_role": rec["application_role"],
            "value": float(rec["value"]),
            "value_unit": rec["value_unit"],
            "scheme": scheme_out,
            "components": components_out,
        }
        if source_calculation_key is not None:
            payload["source_calculation_key"] = source_calculation_key

        note = rec.get("note")
        if note is not None:
            payload["note"] = note

        out.append(payload)

    return out


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
        thermo_points    → points (per-point validation; bad points dropped)

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

    points = _build_thermo_points(thermo_record.get("thermo_points") or thermo_record.get("cp_data"))
    if points:
        block["points"] = points

    # Thermo provenance: link every calculation that actually contributed
    # to the thermo result, not just the "physically produced" ones.
    # For ARC's standard opt/freq/sp pipeline that means:
    #   opt  — geometry that freq + sp were run on
    #   freq — vibrational modes / ZPE / thermal corrections
    #   sp   — electronic-energy reference for H298
    # The links should be self-sufficient; we don't want consumers to have
    # to traverse `calculation_dependency` to recover the opt calc.
    # Order is fixed (opt, freq, sp) so payloads are deterministic — same
    # inputs hash to the same idempotency key across runs.
    sources: list[dict[str, str]] = []
    for key in (_CALC_KEY_OPT, _CALC_KEY_FREQ, _CALC_KEY_SP):
        if key in included_calc_keys:
            sources.append({"calculation_key": key, "role": key})
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


def _build_thermo_points(thermo_points: Any) -> list[dict[str, Any]]:
    """Map ARC's ``thermo_points`` list to ``ThermoPointCreate`` dicts.

    Each entry must carry ``temperature_k``; ``cp_j_mol_k``, ``h_kj_mol``,
    ``s_j_mol_k``, and ``g_kj_mol`` are optional and forwarded when present
    and numeric. Malformed individual points (missing/non-numeric
    temperature, or a per-quantity value that won't coerce) are dropped
    with a warning so a single bad row doesn't take out the whole
    thermo upload.
    """
    if not isinstance(thermo_points, list):
        return []
    seen_temps: set[float] = set()
    points: list[dict[str, Any]] = []
    for i, raw in enumerate(thermo_points):
        if not isinstance(raw, Mapping):
            logger.warning("TCKDB thermo: thermo_points[%d] skipped — not a mapping.", i)
            continue
        t = raw.get("temperature_k")
        if t is None:
            logger.warning("TCKDB thermo: thermo_points[%d] skipped — missing temperature_k.", i)
            continue
        try:
            t_f = float(t)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "TCKDB thermo: thermo_points[%d] skipped — non-numeric temperature_k=%r (%s).",
                i, t, exc,
            )
            continue
        if t_f <= 0:
            logger.warning(
                "TCKDB thermo: thermo_points[%d] skipped — temperature_k must be > 0 (got %s).",
                i, t_f,
            )
            continue
        if t_f in seen_temps:
            # Server enforces uniqueness by temperature_k; skip duplicates here.
            logger.warning(
                "TCKDB thermo: thermo_points[%d] skipped — duplicate temperature_k=%s.",
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
                    "TCKDB thermo: thermo_points[%d].%s dropped — non-numeric %r (%s).",
                    i, src_key, v, exc,
                )
        points.append(point)
    return points


_KEY_PART_RE = re.compile(r"[^A-Za-z0-9]+")


def _safe_key_part(label: str | None) -> str:
    """Sanitize a label for use as part of a bundle local key.

    Bundle keys ride into ``GeometryIn.key`` / ``CalculationIn.key``
    where the schema only requires ``min_length=1``, but downstream
    consumers (and the idempotency-key sanitizer) prefer
    ``[A-Za-z0-9._:-]``. We keep alphanumerics, drop everything else,
    cap to 32 chars, and fall back to ``"x"`` for an all-junk label so
    we never produce an empty key segment.
    """
    if not label:
        return "x"
    cleaned = _KEY_PART_RE.sub("", str(label))[:32]
    return cleaned or "x"


def _local_key_for_actor(prefix: str, index: int, label: str | None) -> str:
    """Build a deterministic per-actor *species* key like ``"r0_CHO"`` / ``"p1_CH3"``.

    The numeric ``index`` is what guarantees uniqueness across actors
    that happen to share a chemical label (e.g. H + H ⇌ H2 has two
    ``r*_H`` slots). The sanitized label tail is purely informational —
    a human reading the JSON should be able to tell ``r0_CHO`` from
    ``r1_CH4`` without cross-referencing. Calc keys derive from
    :func:`_calc_prefix_for_actor` instead, which omits the label so
    calc keys stay short (e.g. ``r0_opt`` not ``r0_CHO_opt``).
    """
    return f"{prefix}{index}_{_safe_key_part(label)}"


def _calc_prefix_for_actor(prefix: str, index: int) -> str:
    """Build the per-actor calc-key prefix (e.g. ``"r0"`` / ``"p1"``).

    The chemical label is intentionally omitted: calculation keys ride
    into ``kinetics.source_calculations`` and are referenced verbatim,
    so shorter is better as long as uniqueness is preserved (the
    role-letter + index combo guarantees it).
    """
    return f"{prefix}{index}"


def _index_species(output_doc: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Build a label → species record map from ``output_doc['species']``.

    Later occurrences win on collision — that matches the implicit
    contract of ARC's output.yml (one record per label) but lets the
    builder fail loudly when a record is genuinely missing rather than
    silently picking a stale duplicate.
    """
    out: dict[str, Mapping[str, Any]] = {}
    for record in output_doc.get("species") or []:
        if not isinstance(record, Mapping):
            continue
        label = record.get("label") or record.get("original_label")
        if label:
            out[str(label)] = record
    return out


def _index_transition_states(
    output_doc: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Build a ts-label → record map from ``output_doc['transition_states']``."""
    out: dict[str, Mapping[str, Any]] = {}
    for record in output_doc.get("transition_states") or []:
        if not isinstance(record, Mapping):
            continue
        label = record.get("label") or record.get("original_label")
        if label:
            out[str(label)] = record
    return out


def _arc_workflow_tool_release(
    output_doc: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build the ARC ``WorkflowToolReleaseRef``-shaped dict, or ``None``.

    Single source of truth for the ARC release identity used in the
    bundle. Returns ``None`` when neither version nor git commit is
    available (rare — usually at least one is set).
    """
    arc_version = output_doc.get("arc_version")
    arc_git_commit = output_doc.get("arc_git_commit")
    if not (arc_version or arc_git_commit):
        return None
    wt: dict[str, Any] = {"name": "ARC"}
    if arc_version:
        wt["version"] = str(arc_version)
    if arc_git_commit:
        wt["git_commit"] = str(arc_git_commit)
    return wt


def _arc_analysis_software_release(
    output_doc: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build the Arkane ``SoftwareReleaseRef``-shaped dict, or ``None``.

    ARC drives Arkane for kinetics/thermo fitting; the only durable
    Arkane provenance ``output.yml`` carries today is the RMG-Py HEAD
    commit (Arkane lives in that repo). No Arkane version string is
    captured, so we emit ``name`` + ``revision`` only and let the
    schema's other ``SoftwareReleaseRef`` fields remain absent.
    """
    arkane_git_commit = output_doc.get("arkane_git_commit")
    if not arkane_git_commit:
        return None
    return {"name": "Arkane", "revision": str(arkane_git_commit)}


# StatmechCalculationRole values that ARC's three-stage opt/freq/sp
# workflow can declare. Order is fixed (opt → freq → sp) so the
# emitted source_calculations list is byte-stable across runs of the
# same content. ``opt_coarse`` is intentionally excluded — it's an
# intermediate optimization stage, not a statmech input (the schema's
# StatmechCalculationRole enum has no ``opt_coarse`` value, and the
# task spec is explicit that opt_coarse is not a statmech source).
_STATMECH_CALC_ROLES: tuple[tuple[str, str], ...] = (
    (_CALC_KEY_OPT, "opt"),
    (_CALC_KEY_FREQ, "freq"),
    (_CALC_KEY_SP, "sp"),
)


def _build_freq_scale_factor_ref(
    output_doc: Mapping[str, Any],
    *,
    workflow_tool_release: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a ``FreqScaleFactorRef``-shaped dict, or ``None``.

    Maps ARC run-level frequency-scale metadata to the unified TCKDB
    reference shape. Returns ``None`` when ARC didn't apply a scale
    factor for this run, or when the level of theory's ``method``
    isn't available — ``LevelOfTheoryRef.method`` is required by the
    schema, and emitting a ref with a missing method would fail
    server-side validation.

    Source-attribution policy (from
    ``FreqScaleFactorRef`` docstring + ARC's audit):

    * ``value`` ← ``output_doc['freq_scale_factor']``
    * ``level_of_theory`` ← ``freq_level`` (preferred — actual freq LOT),
      falling back to ``arkane_level_of_theory`` when freq_level is
      absent (common single-LOT runs).
    * ``software`` ← ``freq_level.software`` (or fallback chain to opt).
    * ``scale_kind`` ← ``"fundamental"``. ARC doesn't distinguish
      ZPE/enthalpy/entropy/Cp scale factors today; ``fundamental`` is
      the ``FrequencyScaleKind`` default.
    * ``note`` ← ``freq_scale_factor_source`` when present (it's a bare
      citation/URL string, not structured literature).
    * ``source_literature`` ← always ``None``. The schema explicitly
      forbids synthesizing literature rows from raw citation strings.
    * ``workflow_tool_release`` ← ARC release **only when ARC's
      curated data file was the proximate source** (i.e., the source
      string is non-null). When the user supplied the factor directly,
      omit ``workflow_tool_release`` — claiming ARC's release would
      fork the dedupe identity tuple ``(level, software, scale_kind,
      value, source_literature, workflow_tool_release)`` and create
      duplicate registry rows.
    """
    value = output_doc.get("freq_scale_factor")
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        logger.warning(
            "TCKDB statmech: malformed freq_scale_factor=%r; omitting FSF ref.",
            value,
        )
        return None
    if value_f <= 0:
        logger.warning(
            "TCKDB statmech: freq_scale_factor=%r is non-positive; omitting "
            "FSF ref (server requires gt 0).", value,
        )
        return None

    # Prefer freq_level, fall back to arkane_level_of_theory. Don't
    # fall further back to opt_level — opt and freq levels are
    # genuinely independent in some workflows, and an FSF mislabeled
    # against opt_level would dedupe incorrectly.
    level_source = output_doc.get("freq_level") or output_doc.get("arkane_level_of_theory")
    level_of_theory = _arc_level_to_tckdb_lot(level_source)
    if level_of_theory is None:
        logger.debug(
            "TCKDB statmech: freq/arkane level missing or has no 'method'; "
            "cannot build FreqScaleFactorRef.level_of_theory.",
        )
        return None

    ref: dict[str, Any] = {
        "level_of_theory": level_of_theory,
        "scale_kind": "fundamental",
        "value": value_f,
    }

    # Software comes from freq_level when present; otherwise fall back
    # to opt_level's software (matches the existing ess_versions
    # fallback in _calculation_payload — opt and freq usually share an
    # ESS in practice).
    freq_software = (
        (level_source.get("software") if isinstance(level_source, Mapping) else None)
        or _opt_level_software(output_doc)
    )
    if freq_software:
        ref["software"] = {"name": str(freq_software)}

    source_string = output_doc.get("freq_scale_factor_source")
    if source_string:
        # Bare citation string lands in note. Never synthesized into
        # source_literature.
        ref["note"] = str(source_string)
        # Only tag ARC as the proximate source when our data file was
        # actually the lookup origin (source_string non-null implies
        # _resolve_freq_scale_factor_source matched a row in
        # data/freq_scale_factors.yml). User-supplied factors leave
        # workflow_tool_release null to avoid forking the registry.
        if workflow_tool_release is not None:
            ref["workflow_tool_release"] = dict(workflow_tool_release)

    return ref


def _opt_level_software(output_doc: Mapping[str, Any]) -> str | None:
    opt_level = output_doc.get("opt_level")
    if isinstance(opt_level, Mapping):
        sw = opt_level.get("software")
        if sw:
            return str(sw)
    return None


# TCKDB ``RigidRotorKind`` enum values (live as of this writing — see
# ``app/db/models/common.py``). ARC's ``_statmech_to_dict`` currently
# only ever emits ``atom`` / ``linear`` / ``asymmetric_top``; the other
# two are listed for forward compatibility if ARC adds the analysis.
_TCKDB_RIGID_ROTOR_KINDS = frozenset({
    "atom", "linear", "spherical_top", "symmetric_top", "asymmetric_top",
})

# TCKDB ``TorsionTreatmentKind`` values that ARC produces today. The
# remaining values (``rigid_top``, ``hindered_rotor_dos``) aren't in
# ARC's lexicon, so they're silently dropped per spec rather than
# guessed at.
_ARC_TO_TCKDB_TORSION_TREATMENTS = frozenset({"free_rotor", "hindered_rotor"})


def _build_statmech_block_for_species(
    *,
    output_doc: Mapping[str, Any],
    species_record: Mapping[str, Any] | None = None,
    calc_keys_by_role: Mapping[str, str],
    workflow_tool_release: Mapping[str, Any] | None,
    scan_key_renames: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Build a ``StatmechInBundle``-/``BundleStatmechIn``-shaped dict, or ``None``.

    Pulls per-species statmech metadata from
    ``species_record['statmech']`` (the dict that ``arc/output.py::
    _statmech_to_dict`` writes into ``output.yml``) and projects it onto
    the upload schema. The frequency-scale-factor handling is unchanged.

    Both bundle endpoints (``StatmechInBundle`` for computed-species,
    ``BundleStatmechIn`` for computed-reaction per-species) now accept
    the same field set: ``external_symmetry``, ``is_linear``,
    ``rigid_rotor_kind``, ``statmech_treatment``, ``point_group``,
    ``freq_scale_factor``, ``torsions``, and ``source_calculations``.
    Mode-specific filtering is therefore a no-op; the only thing the
    caller varies is the calc-key namespace (unscoped for computed-
    species, ``r0_*``/``p0_*``/``ts_*`` for computed-reaction).

    ``calc_keys_by_role`` is the role-to-bundle-local-key map for the
    *owning* species block. Computed-species passes the unscoped keys
    (``{"opt": "opt", ...}``); computed-reaction passes the species-
    scoped equivalents (``{"opt": "r0_opt", ...}``). The helper does
    not synthesize keys — it only writes a source_calculation entry per
    role that the caller declared.

    Returns ``None`` when no field survives filtering — emitting an
    empty statmech container would just create a useless server-side
    row. Per the project convention, no empty containers.
    """
    block: dict[str, Any] = {}

    fsf_ref = _build_freq_scale_factor_ref(
        output_doc, workflow_tool_release=workflow_tool_release,
    )
    if fsf_ref is not None:
        block["freq_scale_factor"] = fsf_ref

    statmech_input = (
        species_record.get("statmech") if isinstance(species_record, Mapping) else None
    )
    if isinstance(statmech_input, Mapping):
        external_symmetry = statmech_input.get("external_symmetry")
        if isinstance(external_symmetry, int) and external_symmetry >= 1:
            block["external_symmetry"] = external_symmetry

        is_linear = statmech_input.get("is_linear")
        if isinstance(is_linear, bool):
            block["is_linear"] = is_linear

        rotor_kind = statmech_input.get("rigid_rotor_kind")
        if isinstance(rotor_kind, str) and rotor_kind in _TCKDB_RIGID_ROTOR_KINDS:
            block["rigid_rotor_kind"] = rotor_kind

        torsions_input = statmech_input.get("torsions")
        treatment = _classify_statmech_treatment(torsions_input)
        if treatment is not None:
            block["statmech_treatment"] = treatment

        slim_torsions = _build_slim_torsions(
            torsions_input, scan_key_renames=scan_key_renames,
        )
        if slim_torsions:
            block["torsions"] = slim_torsions

        point_group = statmech_input.get("point_group")
        if isinstance(point_group, str) and point_group.strip():
            block["point_group"] = point_group.strip()

    # Source calculations are provenance for actual statmech metadata,
    # not statmech metadata in themselves — emitting them on an
    # otherwise-empty block produces a "useless container" that the
    # project convention forbids. Only attach them when the block
    # already carries at least one substantive field (FSF or any
    # species-derived field), which preserves the prior behavior of
    # omitting the whole block on FSF-less runs that lack a statmech
    # subdict. The keys are taken straight from ``calc_keys_by_role``;
    # the caller is responsible for passing the correct namespace
    # (unscoped for computed-species, ``r0_*``/``p0_*`` for computed-
    # reaction species blocks) so the server-side ownership check sees
    # only calculations owned by the same species entry.
    if block:
        sources = _build_statmech_source_calculations(
            calc_keys_by_role=calc_keys_by_role,
        )
        if sources:
            block["source_calculations"] = sources

    return block or None


def _classify_statmech_treatment(
    torsions: Any,
) -> str | None:
    """Map ARC's torsion list to a TCKDB ``StatmechTreatmentKind`` value.

    Rules:

    * ``None`` torsions input (no statmech subdict on the species
      record) → ``None``: ARC didn't emit a statmech evaluation, so the
      treatment is genuinely unknown. Don't fabricate one.
    * Empty list (statmech ran, no successful rotors) → ``"rrho"``.
    * ≥1 1D rotor (each entry's ``atom_indices`` is a flat 4-int list)
      and no ND → ``"rrho_1d"``.
    * ≥1 ND rotor (entry's ``atom_indices`` is a list of 4-int lists)
      and no 1D → ``"rrho_nd"``.
    * Mix of 1D and ND → ``"rrho_1d_nd"``.
    * Anything we can't classify confidently → ``None`` (omitted).

    The ``rrho_ad``/``rrao`` enum values are reserved for treatments
    ARC doesn't currently produce; we never emit them.
    """
    if torsions is None:
        return None
    if not isinstance(torsions, list):
        return None
    if not torsions:
        return "rrho"
    has_1d = False
    has_nd = False
    for t in torsions:
        if not isinstance(t, Mapping):
            return None
        atom_indices = t.get("atom_indices")
        if (
            isinstance(atom_indices, list)
            and len(atom_indices) == 4
            and all(isinstance(x, int) for x in atom_indices)
        ):
            has_1d = True
        elif (
            isinstance(atom_indices, list)
            and atom_indices
            and all(isinstance(x, list) for x in atom_indices)
        ):
            has_nd = True
        else:
            return None
    if has_1d and has_nd:
        return "rrho_1d_nd"
    if has_nd:
        return "rrho_nd"
    if has_1d:
        return "rrho_1d"
    return None


def _build_slim_torsions(
    torsions: Any,
    *,
    scan_key_renames: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build ``BundleStatmechTorsionIn`` entries, with coordinate quartets when available.

    ARC's per-rotor dict (built by ``_get_torsions`` in ``arc/output.py``)
    carries:

      * ``symmetry_number`` (int)
      * ``treatment`` (``"hindered_rotor"`` / ``"free_rotor"``)
      * ``atom_indices`` (1-based 4-int dihedral defining atom quartet,
        or for ND scans a list of 4-int lists)
      * ``pivot_atoms`` (1-based 2-int axis — bundle has no column)
      * ``barrier_kj_mol`` (fitted barrier — bundle has no column)

    The bundle ``StatmechTorsionInBundle`` schema accepts:

      * ``torsion_index`` (1-based, allocated by emission order)
      * ``symmetry_number``
      * ``treatment_kind``
      * ``dimension`` (default 1)
      * ``coordinates`` (list of ``StatmechTorsionCoordinateIn`` —
        each with ``coordinate_index`` plus ``atom1_index``..``atom4_index``,
        all 1-based; the four atoms must be distinct)
      * ``source_scan_calculation_key`` (optional, must resolve to a
        bundle-local calc of type ``scan`` — deferred until ARC emits
        scan calcs)

    Coordinate emission rules:

      * 1D rotors with a flat 4-int ``atom_indices`` → emit one
        coordinate, ``dimension=1``.
      * ND rotors with a list-of-lists ``atom_indices`` → emit one
        coordinate per inner list, ``dimension=N``. The bundle schema
        requires ``len(coordinates) == dimension`` and contiguous
        ``coordinate_index`` values 1..N.
      * Missing or malformed ``atom_indices`` → log a warning and emit
        the summary fields only (no ``coordinates``, no ``dimension``
        override). Producers must never fabricate atom quartets.

    Rotors whose treatment isn't a recognized TCKDB value (e.g. ARC
    might add new types in the future) are omitted entirely rather than
    emitted with a missing treatment_kind — the latter would produce a
    torsion entry that's effectively meaningless to consumers.

    ``pivot_atoms`` and ``barrier_kj_mol`` are deliberately not emitted:
    the bundle schema rejects ``pivot_atoms`` and has no destination
    column for the fitted barrier. Both stay out of the payload.
    """
    if not isinstance(torsions, list):
        return []
    out: list[dict[str, Any]] = []
    next_index = 1
    for entry in torsions:
        if not isinstance(entry, Mapping):
            continue
        treatment = entry.get("treatment")
        if treatment not in _ARC_TO_TCKDB_TORSION_TREATMENTS:
            continue
        slim: dict[str, Any] = {
            "torsion_index": next_index,
            "treatment_kind": treatment,
        }
        sym = entry.get("symmetry_number")
        if isinstance(sym, int) and sym >= 1:
            slim["symmetry_number"] = sym

        coordinates = _coerce_torsion_coordinates(entry.get("atom_indices"))
        if coordinates is not None:
            slim["dimension"] = len(coordinates)
            slim["coordinates"] = coordinates
        else:
            atom_indices = entry.get("atom_indices")
            if atom_indices is not None:
                logger.warning(
                    "TCKDB statmech: torsion #%d has unusable atom_indices=%r; "
                    "emitting torsion summary without coordinates.",
                    next_index, atom_indices,
                )
        # Link the torsion to its underlying scan calc when ARC provided a
        # bundle-local key. Computed-reaction bundles namespace scan calcs
        # per-species (``r0_scan_rotor_0``, etc.) because the server
        # enforces global calc-key uniqueness across the whole bundle —
        # the caller passes a ``scan_key_renames`` map so the torsion's
        # reference matches the namespaced calc key. Computed-species
        # bundles have a single species and pass ``None``: the original
        # un-prefixed key is already unique.
        scan_key = entry.get("source_scan_calculation_key")
        if isinstance(scan_key, str) and scan_key:
            if scan_key_renames is not None:
                scan_key = scan_key_renames.get(scan_key, scan_key)
            slim["source_scan_calculation_key"] = scan_key
        out.append(slim)
        next_index += 1
    return out


def _coerce_torsion_coordinates(
    atom_indices: Any,
) -> list[dict[str, int]] | None:
    """Validate and project ARC's atom_indices into TCKDB coordinate dicts.

    Returns the coordinate list when ``atom_indices`` is well-formed:

      * Flat list of 4 distinct positive ints → one coordinate.
      * List of N flat-4-int lists (each distinct, 4 atoms each) → N
        coordinates with ``coordinate_index`` running 1..N.

    Returns ``None`` when the input is missing, malformed, contains
    non-positive integers, or has duplicate atoms within a quartet —
    callers fall back to a coordinate-less torsion summary so a single
    bad rotor never blocks the whole upload. The 4-distinct-atoms check
    mirrors the server-side ``StatmechTorsionCoordinateIn`` validator;
    failing here gives a clearer producer-side message than letting the
    server 422.
    """
    if atom_indices is None:
        return None
    if not isinstance(atom_indices, list) or not atom_indices:
        return None
    # 1D shape: a flat list of exactly 4 ints.
    if all(isinstance(x, int) for x in atom_indices):
        if len(atom_indices) != 4:
            return None
        if not all(x >= 1 for x in atom_indices):
            return None
        if len(set(atom_indices)) != 4:
            return None
        a1, a2, a3, a4 = atom_indices
        return [{
            "coordinate_index": 1,
            "atom1_index": a1, "atom2_index": a2,
            "atom3_index": a3, "atom4_index": a4,
        }]
    # ND shape: a list of 4-int sub-lists.
    if all(isinstance(x, list) for x in atom_indices):
        coords: list[dict[str, int]] = []
        for i, quartet in enumerate(atom_indices, start=1):
            if not all(isinstance(x, int) for x in quartet):
                return None
            if len(quartet) != 4:
                return None
            if not all(x >= 1 for x in quartet):
                return None
            if len(set(quartet)) != 4:
                return None
            a1, a2, a3, a4 = quartet
            coords.append({
                "coordinate_index": i,
                "atom1_index": a1, "atom2_index": a2,
                "atom3_index": a3, "atom4_index": a4,
            })
        return coords or None
    return None


def _build_statmech_source_calculations(
    *,
    calc_keys_by_role: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Compose statmech ``source_calculations`` from the bundle's emitted calcs.

    Walks ``_STATMECH_CALC_ROLES`` in order (opt → freq → sp) and emits
    one entry per role whose key is present in the caller-supplied
    ``calc_keys_by_role`` map. ``opt_coarse`` is excluded by design —
    it's not a ``StatmechCalculationRole`` enum value.

    The role keys (``"opt"`` / ``"freq"`` / ``"sp"``) are the
    ``StatmechCalculationRole`` enum strings. Values are the bundle-local
    calculation keys the server should resolve. Both computed-species
    and computed-reaction call this with their own scoped values; this
    helper is mode-agnostic.
    """
    return [
        {"calculation_key": calc_keys_by_role[role], "role": role}
        for _, role in _STATMECH_CALC_ROLES
        if role in calc_keys_by_role and calc_keys_by_role[role]
    ]


def _ts_unmapped_smiles_handle(
    *,
    ts_record: Mapping[str, Any],
    reaction_record: Mapping[str, Any],
    species_index: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """Derive a deterministic textual handle for the TS, or ``None``.

    Resolution order, lifted directly from the spec:

    1. ``ts_record['smiles']`` when ARC already attached a textual TS
       identifier (rare in production — TS records usually have SMILES
       null because there's no Lewis structure).
    2. A canonical reaction-SMILES handle ``"<r1>.<r2>>><p1>.<p2>"``
       built from the reactant/product species' SMILES looked up
       through ``species_index``. ``.`` joins species; ``>>`` separates
       reactant and product sides — the standard interchange format
       and unambiguous from ARC's data. This is *not* a claim that the
       TS itself is a single molecule; it's the same data the field
       intends to carry — a traceability handle.
    3. ``None`` when any reactant or product lacks a SMILES, or when
       the reaction has no reactant/product labels. The producer
       refuses to emit a misleading half-handle; TCKDB stores NULL.

    Determinism: the helper is a pure function of the input mappings.
    No paths, timestamps, or run identifiers leak in. Idempotency keys
    will move only if the upstream species SMILES change.
    """
    direct = ts_record.get("smiles")
    if direct:
        text = str(direct).strip()
        if text:
            return text

    def _smiles_for_labels(labels: list[str] | None) -> list[str] | None:
        out: list[str] = []
        for label in labels or []:
            record = species_index.get(label)
            if not isinstance(record, Mapping):
                return None
            smiles = record.get("smiles")
            if not smiles:
                return None
            out.append(str(smiles).strip())
        return out or None

    reactant_smiles = _smiles_for_labels(reaction_record.get("reactant_labels"))
    if reactant_smiles is None:
        return None
    product_smiles = _smiles_for_labels(reaction_record.get("product_labels"))
    if product_smiles is None:
        return None

    return f"{'.'.join(reactant_smiles)}>>{'.'.join(product_smiles)}"


# Result-shape adapter: ``CalculationInBundle`` (computed-species)
# wraps results in nested ``opt_result``/``freq_result``/``sp_result``
# dicts; the network_pdep ``CalculationIn`` (which the computed-reaction
# endpoint extends) carries the same data as flat fields. The mapping
# below is the full v0 translation; ``_calculation_payload`` produces
# the wrapped shape, and :func:`_flatten_result_fields` rewrites it to
# the flat shape in place when building reaction bundles.
_REACTION_FLAT_RESULT_FIELDS: dict[str, dict[str, str]] = {
    "opt_result": {
        "converged": "opt_converged",
        "n_steps": "opt_n_steps",
        "final_energy_hartree": "opt_final_energy_hartree",
    },
    "freq_result": {
        "n_imag": "freq_n_imag",
        "imag_freq_cm1": "freq_imag_freq_cm1",
        "zpe_hartree": "freq_zpe_hartree",
    },
    "sp_result": {
        "electronic_energy_hartree": "sp_electronic_energy_hartree",
    },
}


def _flatten_result_fields(calc: dict[str, Any]) -> None:
    """Convert wrapped ``opt_result``/``freq_result``/``sp_result`` into flat fields.

    Mutates ``calc`` in place: each wrapped result dict is removed and
    its values are promoted to the network_pdep-style flat field names.
    A no-op for IRC and any other calc type without a wrapped result —
    those carry their data through ``parameters_json`` instead.
    """
    for wrapped_key, field_map in _REACTION_FLAT_RESULT_FIELDS.items():
        result = calc.pop(wrapped_key, None)
        if not isinstance(result, Mapping):
            continue
        for src, dst in field_map.items():
            if src in result:
                calc[dst] = result[src]
        if wrapped_key == "freq_result":
            modes = result.get("modes")
            if modes:
                calc["freq_frequencies_cm1"] = [m["frequency_cm1"] for m in modes]


def _flatten_all_reaction_calcs(bundle: dict[str, Any]) -> None:
    """Walk a computed-reaction bundle and flatten every calc in place.

    Hits each species block's primary opt (under ``conformers[*].calculation``)
    and additionals (under ``calculations``), then the TS's primary
    (``transition_state.calculation``) and additionals
    (``transition_state.calculations``). One pass at the bundle root
    keeps the per-builder code free of shape concerns.
    """
    for species in bundle.get("species") or []:
        for conf in species.get("conformers") or []:
            calc = conf.get("calculation")
            if isinstance(calc, dict):
                _flatten_result_fields(calc)
        for calc in species.get("calculations") or []:
            if isinstance(calc, dict):
                _flatten_result_fields(calc)
    ts = bundle.get("transition_state")
    if isinstance(ts, dict):
        primary = ts.get("calculation")
        if isinstance(primary, dict):
            _flatten_result_fields(primary)
        for calc in ts.get("calculations") or []:
            if isinstance(calc, dict):
                _flatten_result_fields(calc)


# Roles defined by TCKDB's ``KineticsCalculationRole`` enum; mirrored
# here so the adapter stays loud about unknown roles instead of
# bouncing through a 422 at upload time.
_KINETICS_ROLE_REACTANT_ENERGY = "reactant_energy"
_KINETICS_ROLE_PRODUCT_ENERGY = "product_energy"
_KINETICS_ROLE_TS_ENERGY = "ts_energy"
_KINETICS_ROLE_TS_FREQ = "freq"
_KINETICS_ROLE_TS_IRC = "irc"


def _coerce_optional_float(value: object) -> float | None:
    """Return value as float when possible, otherwise None."""
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# Preferred lookup order for extracting a canonical tunneling-model
# label out of a structured value. Mappings are checked first by key,
# then arbitrary objects by attribute. Keep the two lists aligned;
# ``class_name`` is the attribute analogue of the ``class`` mapping
# key (which would be a builtin shadow on attribute access).
_TUNNELING_MODEL_KEYS = ("method", "model", "type", "name", "class")
_TUNNELING_MODEL_ATTRS = ("method", "model", "type", "name", "class_name")


def _stringify_tunneling_model(value: object) -> str | None:
    """Return a concise, stable tunneling-model label for TCKDB.

    Tunneling metadata reaches the adapter from output.yml as a free-form
    value (string today, possibly a structured object tomorrow). TCKDB
    stores ``tunneling_model`` as a free-form ``str | None``, so the
    adapter's job is to turn whatever ARC supplies into a stable,
    human-meaningful label without ever crashing payload construction
    on a malformed entry.
    """
    if not value:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, Mapping):
        for key in _TUNNELING_MODEL_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    else:
        for attr in _TUNNELING_MODEL_ATTRS:
            candidate = getattr(value, attr, None)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        encoded = str(value).strip()
    return encoded or None


def _build_kinetics_block(
    *,
    kinetics_record: Mapping[str, Any],
    reactant_keys: list[str],
    product_keys: list[str],
    actor_calc_keys: Mapping[str, Mapping[str, str]],
    ts_calc_keys: Mapping[str, str],
    long_kinetic_description: str | None = None,
) -> dict[str, Any] | None:
    """Build a ``BundleKineticsIn``-shaped dict from ARC kinetics.

    Mapping (ARC → TCKDB):
        A           → a
        A_units     → a_units (via :func:`arc_to_tckdb_a_units`)
        n           → n
        Ea          → reported_ea
        Ea_units    → reported_ea_units (via :func:`arc_to_tckdb_ea_units`)
        Tmin_k      → tmin_k
        Tmax_k      → tmax_k
        dn          → n_uncertainty
        dEa         → d_reported_ea (only when dEa_units == Ea_units;
                      the live schema has no d_reported_ea_units field,
                      so dEa is only safe to emit when its units agree
                      with Ea's)
        dA          → a_uncertainty (+ a_uncertainty_kind="multiplicative")
                      Arkane/ARC dA is a multiplicative factor f, with
                      the true value bracketed by [A/f, A*f]. TCKDB's
                      ``KineticsUncertaintyKind`` exposes that semantic
                      explicitly, so the producer preserves dA verbatim
                      rather than dropping or re-encoding it. dA < 1.0
                      is omitted (server rejects multiplicative factors
                      below 1).

    ``source_calculations`` is populated from each actor's emitted
    sp/ts_sp/ts_freq/ts_irc keys. Reactant/product freq calcs are
    deliberately *not* linked as ``role=freq`` — in v0 kinetics that
    role is reserved for the TS frequency.

    Returns ``None`` when neither A nor Ea is populated — TCKDB's
    ``BundleKineticsIn`` allows empty kinetics, but a totally empty
    record is just noise and easier to omit than to send.
    """
    has_substantive_field = any(
        kinetics_record.get(k) is not None
        for k in ("A", "n", "Ea", "Tmin_k", "Tmax_k")
    )
    if not has_substantive_field:
        return None

    block: dict[str, Any] = {
        "reactant_keys": list(reactant_keys),
        "product_keys": list(product_keys),
        "model_kind": "modified_arrhenius",
    }

    a = kinetics_record.get("A")
    if a is not None:
        try:
            block["a"] = float(a)
        except (TypeError, ValueError) as exc:
            logger.warning("TCKDB kinetics: malformed A=%r (%s)", a, exc)
    a_units = arc_to_tckdb_a_units(kinetics_record.get("A_units"))
    if a_units:
        block["a_units"] = a_units

    n = kinetics_record.get("n")
    if n is not None:
        try:
            block["n"] = float(n)
        except (TypeError, ValueError) as exc:
            logger.warning("TCKDB kinetics: malformed n=%r (%s)", n, exc)

    ea = kinetics_record.get("Ea")
    ea_units = arc_to_tckdb_ea_units(kinetics_record.get("Ea_units"))
    if ea is not None:
        try:
            block["reported_ea"] = float(ea)
        except (TypeError, ValueError) as exc:
            logger.warning("TCKDB kinetics: malformed Ea=%r (%s)", ea, exc)
    if ea_units:
        block["reported_ea_units"] = ea_units

    for arc_key, payload_key in (("Tmin_k", "tmin_k"), ("Tmax_k", "tmax_k")):
        v = kinetics_record.get(arc_key)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError) as exc:
            logger.warning("TCKDB kinetics: malformed %s=%r (%s)", arc_key, v, exc)
            continue
        if f <= 0:
            logger.warning(
                "TCKDB kinetics: %s=%r is non-positive; field will be omitted "
                "(server requires gt 0).", arc_key, v,
            )
            continue
        block[payload_key] = f

    dn = kinetics_record.get("dn")
    if dn is not None:
        try:
            block["n_uncertainty"] = float(dn)
        except (TypeError, ValueError) as exc:
            logger.warning("TCKDB kinetics: malformed dn=%r (%s)", dn, exc)

    # dEa policy: ARC's dEa_units may differ from Ea_units. TCKDB has
    # only one Ea unit per kinetics row (no separate d_reported_ea_units
    # column today), so we can only safely emit d_reported_ea when the
    # producer reported it in the same units as Ea — otherwise the
    # number would be silently misinterpreted as the wrong unit.
    dea = kinetics_record.get("dEa")
    if dea is not None:
        dea_units_raw = kinetics_record.get("dEa_units")
        dea_units = arc_to_tckdb_ea_units(dea_units_raw) if dea_units_raw else ea_units
        if dea_units is not None and ea_units is not None and dea_units == ea_units:
            try:
                block["d_reported_ea"] = float(dea)
            except (TypeError, ValueError) as exc:
                logger.warning("TCKDB kinetics: malformed dEa=%r (%s)", dea, exc)
        elif dea_units is None and ea_units is None:
            # Best-effort: both unitless; pass through.
            try:
                block["d_reported_ea"] = float(dea)
            except (TypeError, ValueError) as exc:
                logger.warning("TCKDB kinetics: malformed dEa=%r (%s)", dea, exc)
        else:
            logger.debug(
                "TCKDB kinetics: dEa units (%r) differ from Ea units (%r); "
                "omitting d_reported_ea to avoid unit ambiguity.",
                dea_units_raw, kinetics_record.get("Ea_units"),
            )

    # dA policy: Arkane/ARC dA is a multiplicative uncertainty factor
    # (true A in [A/f, A*f]). TCKDB's ``KineticsUncertaintyKind.multiplicative``
    # encodes exactly that, so we forward dA verbatim — never re-derive
    # it as A*(dA-1) or fold it into an additive band. Pair the value
    # with ``a_uncertainty_kind="multiplicative"`` (the schema requires
    # both fields to appear together, and rejects multiplicative factors
    # below 1.0).
    da = kinetics_record.get("dA")
    if da is not None:
        try:
            da_f = float(da)
        except (TypeError, ValueError) as exc:
            logger.warning("TCKDB kinetics: malformed dA=%r (%s)", da, exc)
        else:
            if da_f < 1.0:
                logger.debug(
                    "TCKDB kinetics: dA=%r is below 1.0; multiplicative factors "
                    "must be >= 1, so a_uncertainty will be omitted.",
                    da,
                )
            else:
                block["a_uncertainty"] = da_f
                block["a_uncertainty_kind"] = "multiplicative"

    # Tunneling method ARC applied to the fit (currently always Eckart;
    # surfaced through output.yml so the adapter doesn't have to hardcode
    # the constant alongside the producer template). Absent → omit the
    # field for backward compat with older output.yml versions.
    tunneling_label = _stringify_tunneling_model(kinetics_record.get("tunneling"))
    if tunneling_label:
        block["tunneling_model"] = tunneling_label

    # Reaction-path degeneracy. TCKDB's ``BundleKineticsIn.degeneracy``
    # is ``float | None`` with ``gt=0``; the server rejects zero and
    # negative values. We mirror that constraint locally so the
    # producer never ships a value the server would 422 — and so the
    # absence of degeneracy on the wire stays distinct from a "zero"
    # value (the schema treats missing as NULL, *not* 1.0, and the
    # producer must too: don't default to 1, don't infer from
    # stoichiometry). Non-numeric / None / non-positive → omit.
    degeneracy = _coerce_optional_float(kinetics_record.get("degeneracy"))
    if degeneracy is not None and degeneracy > 0:
        block["degeneracy"] = degeneracy

    # Free-form note. Prefer an explicit ``note`` on the kinetics record
    # (future-proofing) over the reaction-level
    # ``long_kinetic_description`` ARC carries today. Both routes feed
    # the same TCKDB ``KineticsCreate.note`` slot.
    note_candidates = (
        kinetics_record.get("note"),
        long_kinetic_description,
    )
    for candidate in note_candidates:
        if isinstance(candidate, str) and candidate.strip():
            block["note"] = candidate.strip()
            break

    sources = _build_kinetics_source_calculations(
        reactant_keys=reactant_keys,
        product_keys=product_keys,
        actor_calc_keys=actor_calc_keys,
        ts_calc_keys=ts_calc_keys,
    )
    if sources:
        block["source_calculations"] = sources

    return block


def _build_kinetics_source_calculations(
    *,
    reactant_keys: list[str],
    product_keys: list[str],
    actor_calc_keys: Mapping[str, Mapping[str, str]],
    ts_calc_keys: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Compose the kinetics ``source_calculations`` list explicitly.

    Producer policy: declare exactly the links we have evidence for.

    - Each reactant/product contributes its sp calc (if present) under
      ``reactant_energy``/``product_energy``.
    - The TS contributes ts_sp under ``ts_energy``.
    - The TS contributes ts_freq under ``freq`` (kinetics ``freq`` role
      is the *TS* frequency in v0, not reactant freq).
    - The TS contributes ts_irc under ``irc`` only when ARC produced
      and persisted an IRC calc.

    Anything missing is omitted — the spec is explicit that a missing
    source link is preferable to a fabricated one.
    """
    links: list[dict[str, Any]] = []
    for actor_key in reactant_keys:
        sp_key = actor_calc_keys.get(actor_key, {}).get(_CALC_KEY_SP)
        if sp_key:
            links.append({"calculation_key": sp_key, "role": _KINETICS_ROLE_REACTANT_ENERGY})
    for actor_key in product_keys:
        sp_key = actor_calc_keys.get(actor_key, {}).get(_CALC_KEY_SP)
        if sp_key:
            links.append({"calculation_key": sp_key, "role": _KINETICS_ROLE_PRODUCT_ENERGY})
    ts_sp = ts_calc_keys.get(_CALC_KEY_SP)
    if ts_sp:
        links.append({"calculation_key": ts_sp, "role": _KINETICS_ROLE_TS_ENERGY})
    ts_freq = ts_calc_keys.get(_CALC_KEY_FREQ)
    if ts_freq:
        links.append({"calculation_key": ts_freq, "role": _KINETICS_ROLE_TS_FREQ})
    ts_irc = ts_calc_keys.get(_CALC_KEY_IRC)
    if ts_irc:
        links.append({"calculation_key": ts_irc, "role": _KINETICS_ROLE_TS_IRC})
    return links


# IRC direction labels mirror TCKDB's ``IRCDirection`` enum. They are
# ESS path-direction labels only — the producer does not infer
# reactant/product side from them.
_IRC_DIRECTION_FORWARD = "forward"
_IRC_DIRECTION_REVERSE = "reverse"
_IRC_DIRECTION_BOTH = "both"


def _detect_irc_direction(log_path: str) -> str | None:
    """Detect IRC direction (forward/reverse) from a log filename.

    ARC writes per-direction IRC logs whose filenames carry an explicit
    direction infix. The two patterns seen in production runs:

    - ``..._irc_f.log`` / ``..._irc_r.log``   (compact runtime convention)
    - ``..._forward.log`` / ``..._reverse.log`` (long-form / test fixtures)

    Returns ``None`` if the filename matches neither — caller still
    emits the trajectory but omits the per-point direction. ``None``
    is also the right answer when ARC has consolidated forward+reverse
    into a single log; current ARC adapters don't produce that shape,
    but the caller treats it correctly anyway.
    """
    name = Path(str(log_path)).name.lower()
    if "irc_f" in name or "forward" in name:
        return _IRC_DIRECTION_FORWARD
    if "irc_r" in name or "reverse" in name:
        return _IRC_DIRECTION_REVERSE
    return None


# Hartree → kJ/mol conversion. Matches CODATA 2018 within the
# truncation TCKDB downstream code uses for relative energies; defining
# it inline keeps this helper free of further imports at module top.
_HARTREE_TO_KJ_MOL = 2625.4996


def _build_irc_result_payload(
    trajectories: list[dict[str, Any]],
    zero_energy_reference_hartree: float | None = None,
    ts_marker: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a TCKDB ``IRCResultPayload``-shaped dict from parsed trajectories.

    Each trajectory is a
    ``{"direction": str|None,
       "rich_points": [<rich-point dict>, ...] | None,
       "geom_points": [<xyz_dict>, ...] | None}``
    as produced by :meth:`TCKDBAdapter._parse_irc_trajectories`. When
    ``rich_points`` is populated, per-point ``electronic_energy_hartree``,
    ``reaction_coordinate``, ``max_gradient``, ``rms_gradient``, and (when
    Gaussian provides them) per-point ``direction`` flow through. When
    only ``geom_points`` is available the result reduces to the original
    geometry-only IRC payload behavior — energies/gradients are simply
    omitted, the schema makes them optional.

    ``point_index`` is allocated globally across both branches so the
    server-side uniqueness invariant holds. Direction labels are passed
    straight through — forward/reverse are ESS path labels, not
    reactant/product designators.

    ``zero_energy_reference_hartree`` is the TS reference energy at the
    IRC level of theory, resolved by
    :func:`_resolve_irc_zero_energy_reference`. When non-null, per-point
    ``relative_energy_kj_mol`` is computed against it and the result-
    level field is stamped on the payload. When null, both stay absent
    rather than fabricated.

    ``ts_marker`` is an optional dict
    ``{"xyz_text": <normalized xyz>, "electronic_energy_hartree": float|None}``
    representing the *seed* TS of the IRC path — the optimized saddle
    that ARC handed to the IRC job. When provided, a single synthesized
    point is appended after all trajectory points with ``is_ts=True``,
    ``reaction_coordinate=0.0`` (the TS is the origin by definition),
    ``relative_energy_kj_mol=0.0`` (when ``zero_energy_reference_hartree``
    is set), and the supplied geometry/energy. The result's
    ``ts_point_index`` is then set to this synthesized point's index.

    This is distinct from inferring the TS from per-point trajectory
    data (which ARC's parsers don't reliably do): the TS marker is
    ARC's own seed for the IRC, not a derivative of the parsed log.
    Ordering: the marker is *appended* rather than inserted between
    branches, so existing trajectory indices stay stable and ARC's
    ESS-emitted trajectory order is not reshuffled.
    """
    if not trajectories:
        return None
    # Late-imported to avoid pulling species converter at adapter import
    # time when the producer is configured but disabled.
    from arc.species.converter import xyz_to_str

    points: list[dict[str, Any]] = []
    has_forward = False
    has_reverse = False
    for traj in trajectories:
        traj_direction = traj.get("direction")
        rich_points = traj.get("rich_points") or []
        geom_points = traj.get("geom_points") or []
        # Rich points carry per-point direction (Gaussian's FORWARD/
        # REVERSE announcement). Geometry-only points inherit the
        # trajectory-level direction resolved upstream from the
        # scheduler list / filename heuristic.
        if rich_points:
            iter_records = (
                {
                    "xyz": rp.get("xyz"),
                    "direction": rp.get("direction") or traj_direction,
                    "electronic_energy_hartree": rp.get("electronic_energy_hartree"),
                    "reaction_coordinate": rp.get("reaction_coordinate"),
                    "max_gradient": rp.get("max_gradient"),
                    "rms_gradient": rp.get("rms_gradient"),
                }
                for rp in rich_points
            )
        else:
            iter_records = (
                {
                    "xyz": xyz,
                    "direction": traj_direction,
                    "electronic_energy_hartree": None,
                    "reaction_coordinate": None,
                    "max_gradient": None,
                    "rms_gradient": None,
                }
                for xyz in geom_points
            )

        for record in iter_records:
            direction = record["direction"]
            if direction == _IRC_DIRECTION_FORWARD:
                has_forward = True
            elif direction == _IRC_DIRECTION_REVERSE:
                has_reverse = True
            point: dict[str, Any] = {"point_index": len(points)}
            if direction in (_IRC_DIRECTION_FORWARD, _IRC_DIRECTION_REVERSE):
                point["direction"] = direction
            xyz_dict = record["xyz"]
            if xyz_dict is not None:
                try:
                    xyz_str = xyz_to_str(xyz_dict=xyz_dict)
                except Exception as exc:
                    logger.debug(
                        "TCKDB computed-reaction: IRC point xyz_to_str failed: %s",
                        exc,
                    )
                    xyz_str = None
                normalized = _normalize_xyz_text(xyz_str, None)
                if normalized:
                    point["geometry"] = {"xyz_text": normalized}
            energy = record["electronic_energy_hartree"]
            if energy is not None:
                point["electronic_energy_hartree"] = float(energy)
                if zero_energy_reference_hartree is not None:
                    point["relative_energy_kj_mol"] = (
                        (float(energy) - zero_energy_reference_hartree)
                        * _HARTREE_TO_KJ_MOL
                    )
            rc = record["reaction_coordinate"]
            if rc is not None:
                point["reaction_coordinate"] = float(rc)
            max_grad = record["max_gradient"]
            if max_grad is not None:
                point["max_gradient"] = float(max_grad)
            rms_grad = record["rms_gradient"]
            if rms_grad is not None:
                point["rms_gradient"] = float(rms_grad)
            points.append(point)
    if not points:
        return None
    if has_forward and has_reverse:
        overall = _IRC_DIRECTION_BOTH
    elif has_forward:
        overall = _IRC_DIRECTION_FORWARD
    elif has_reverse:
        overall = _IRC_DIRECTION_REVERSE
    else:
        # Direction couldn't be detected from filenames. The schema
        # requires a direction enum value — default to ``both`` since
        # ARC's invariant is that any IRC run with multiple logs covers
        # both branches. With a single unlabeled log we still pick
        # ``both`` rather than guessing one side; the per-point
        # direction is simply omitted.
        overall = _IRC_DIRECTION_BOTH

    # TS marker: synthesized point identifying the optimized saddle the
    # IRC was seeded from. ``direction`` is intentionally omitted (the
    # schema treats null direction as the TS marker); ``is_ts=True``
    # and a matching ``ts_point_index`` on the result let downstream
    # consumers locate the TS without scanning energies. Only emit
    # when we actually have a TS xyz to attach — the marker is
    # geometry-anchored, not a pure flag.
    ts_point_index: int | None = None
    if ts_marker and ts_marker.get("xyz_text"):
        ts_index = len(points)
        ts_point: dict[str, Any] = {
            "point_index": ts_index,
            "is_ts": True,
            "reaction_coordinate": 0.0,
            "geometry": {"xyz_text": str(ts_marker["xyz_text"])},
        }
        ts_energy = ts_marker.get("electronic_energy_hartree")
        if ts_energy is not None:
            try:
                ts_energy_f = float(ts_energy)
            except (TypeError, ValueError):
                ts_energy_f = None
            if ts_energy_f is not None:
                ts_point["electronic_energy_hartree"] = ts_energy_f
                if zero_energy_reference_hartree is not None:
                    ts_point["relative_energy_kj_mol"] = (
                        (ts_energy_f - zero_energy_reference_hartree)
                        * _HARTREE_TO_KJ_MOL
                    )
        points.append(ts_point)
        ts_point_index = ts_index

    payload: dict[str, Any] = {
        "direction": overall,
        "has_forward": has_forward,
        "has_reverse": has_reverse,
        "point_count": len(points),
        "points": points,
    }
    if ts_point_index is not None:
        payload["ts_point_index"] = ts_point_index
    if zero_energy_reference_hartree is not None:
        payload["zero_energy_reference_hartree"] = float(
            zero_energy_reference_hartree
        )
    return payload


def _level_keys_match(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> bool:
    """Conservative level-of-theory equality check.

    Compares only the fields TCKDB treats as primary keys for
    ``LevelOfTheoryRef`` (method/basis/aux_basis/cabs_basis). Differences
    in software, dispersion, or solvation are intentionally not enough
    to declare equality here — but they're also not enough to declare
    inequality, since ARC frequently leaves them null on one side
    (the opt level) and populated on the other (the sp level) without
    that meaning the *energies* are at different levels.
    """
    if a is None or b is None:
        return False
    for field in _TCKDB_LOT_REF_FIELDS:
        av = a.get(field)
        bv = b.get(field)
        if av is None and bv is None:
            continue
        if av is None or bv is None:
            return False
        if str(av).strip().lower() != str(bv).strip().lower():
            return False
    return True


def _parse_xtb_turbomole_energy_file(path: str | Path) -> float | None:
    """Parse the Turbomole-format ``energy`` file written by ``xtb --grad``.

    Provenance is xTB; ``Turbomole`` here names only the on-disk text
    format xTB emits when invoked with ``--grad`` (the same shape
    Turbomole's gradient driver writes — xTB borrows the layout to
    plug into Turbomole-compatible tooling like ``tm2orca.py``).

    File shape (one cycle per line, terminated by ``$end``)::

        $energy      SCF              SCFKIN            SCFPOT
             1     -28.12345678901   0.0   0.0
        $end

    The relevant value is on the line before ``$end``, second
    whitespace-separated field. Mirrors ``tm2orca.py``'s parse
    convention so the two stay in lockstep. Returns ``None`` for
    missing/malformed files (the caller treats that as
    "no energy known for this node").
    """
    try:
        with open(path) as fh:
            lines = fh.readlines()
        return float(lines[-2].strip().split()[1])
    except (OSError, ValueError, IndexError):
        return None


def _parse_xtb_turbomole_gradient_file(
    path: str | Path,
) -> tuple[float | None, float | None]:
    """Parse the Turbomole-format ``gradient`` file written by ``xtb --grad``
    → ``(max_grad, rms_grad)`` in Hartree/Bohr. Returns ``(None, None)``
    for missing/malformed input.

    Provenance is xTB; ``Turbomole`` names only the on-disk text
    format xTB emits with ``--grad``.

    File shape::

        $grad   cycle = 1  SCF energy = ...
           x  y  z  symbol      ← N coord lines
           ...
           gx  gy  gz           ← N gradient-component lines (D-exponent)
           ...
        $end

    ``max_grad`` is the largest absolute gradient component; ``rms_grad``
    is ``sqrt(mean(g_i^2))`` over all ``3N`` components. Both are
    derived from the *gradient* (``dE/dx``), not the force — schema
    distinguishes the two and the source data is unambiguously the
    gradient (xTB writes its output in Turbomole's gradient layout
    when called with ``--grad``).
    """
    try:
        with open(path) as fh:
            lines = fh.readlines()
        natoms = int((len(lines) - 3) / 2)
        if natoms < 1:
            return None, None
        grad_lines = lines[2 + natoms : 2 + 2 * natoms]
        components: list[float] = []
        for line in grad_lines:
            parts = line.strip().replace("D", "E").split()
            if len(parts) < 3:
                continue
            components.extend(float(p) for p in parts[:3])
        if not components:
            return None, None
        import math
        max_g = max(abs(c) for c in components)
        rms_g = math.sqrt(sum(c * c for c in components) / len(components))
        return max_g, rms_g
    except (OSError, ValueError, IndexError):
        return None, None


def _read_gsm_node_outputs(
    outputs_dir: str | Path,
) -> dict[int, dict[str, float]]:
    """Scan a ``gsm_node_outputs/`` directory for preserved per-node data
    written by ``xtb --grad`` (in Turbomole's on-disk format).

    The patched ``ograd`` wrapper writes files named like::

        gsm_node_outputs/
          0000.01.energy        ← xTB-generated, Turbomole-format
          0000.01.gradient      ← xTB-generated, Turbomole-format
          0000.01.xtbout        ← xTB stdout (preserved for inspection)
          0000.02.energy
          ...

    where the ``0000`` prefix is GSM's iteration-round id and the
    ``.NN`` suffix is the per-round node label (1-based). This helper
    returns ``{label_int: {electronic_energy_hartree, max_gradient,
    rms_gradient}}`` for every label whose ``.energy`` file parses
    cleanly. Nodes missing both files (or whose files don't parse) are
    omitted; the caller decides whether to leave the corresponding
    point's metadata null.

    The ESS provenance for the resulting calculation remains xTB /
    xTB-GSM; ``Turbomole`` names only the on-disk text shape these
    files use.
    """
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.is_dir():
        return {}
    by_label: dict[int, dict[str, float]] = {}
    for energy_file in sorted(outputs_dir.glob("*.energy")):
        # Filename is ``<round>.<NN>.energy`` — pull the trailing
        # ``NN`` integer (the per-round node label).
        stem = energy_file.stem  # e.g. "0000.01"
        try:
            label_int = int(stem.split(".")[-1])
        except (ValueError, IndexError):
            continue
        e_h = _parse_xtb_turbomole_energy_file(energy_file)
        if e_h is None:
            continue
        node_data: dict[str, float] = {"electronic_energy_hartree": e_h}
        gradient_file = outputs_dir / f"{stem}.gradient"
        if gradient_file.is_file():
            max_g, rms_g = _parse_xtb_turbomole_gradient_file(gradient_file)
            if max_g is not None:
                node_data["max_gradient"] = max_g
            if rms_g is not None:
                node_data["rms_gradient"] = rms_g
        by_label[label_int] = node_data
    return by_label


def _build_path_search_result_payload(
    *,
    method: str,
    log_path: str | Path | None,
    fallback_xyz_text: str | None,
    node_outputs_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Build a TCKDB ``PathSearchResultPayload``-shaped dict for the
    chosen TS-guess parent calc.

    The backend requires ``points`` with ``min_length=1`` (see
    ``backend/app/schemas/fragments/calculation.py``), so a bare
    ``{method: ...}`` payload won't validate.

    Strategy by method:

    * ``method == 'gsm'``: parse the GSM stringfile via
      :func:`arc.parser.parser.parse_trajectory` (handles the
      ``stringfile.xyz0000`` multi-frame XYZ format). Each frame
      becomes one point with ``geometry``. ``selected_ts_point_index``
      and the corresponding point's ``is_ts_guess`` flag track the
      middle-ish frame xtb_gsm picks as the TS guess (mirroring
      ``xTBGSMAdapter.process_run``).

    * ``method == 'neb'``: NEB log image extraction is a future parser
      lift; for now emit a single TS-guess point from
      ``fallback_xyz_text`` (the chosen guess's geometry, also the
      ts_opt's input XYZ). This is provenance the producer already
      committed to — the same geometry that became ``ts_opt`` input —
      not invented data. Marked with ``is_ts_guess=True``.

    * Other methods or unparseable logs: fall back to the single-point
      shape, or return ``None`` if no fallback geometry is available.

    Returns ``None`` (caller skips the calc) when even the fallback
    can't be built — the conservative gate then leaves ``ts_opt``
    edge-less rather than emitting a path_search calc with no points.
    """
    # Late-imported to keep the adapter's import surface lean when
    # path-search emission isn't exercised.
    from arc.constants import E_h_kJmol
    from arc.parser.parser import parse_trajectory
    from arc.species.converter import xyz_to_str

    # Per-node parsed metadata from preserved xtb outputs, when the
    # producer wrote them. Empty dict for older runs (wrapper predates
    # preservation) — points stay geometry-only, no inventing.
    node_metadata: dict[int, dict[str, float]] = (
        _read_gsm_node_outputs(node_outputs_dir)
        if node_outputs_dir is not None and method == "gsm"
        else {}
    )

    points: list[dict[str, Any]] = []
    selected_index: int | None = None

    if method == "gsm" and log_path is not None:
        path_str = str(log_path)
        try:
            traj = parse_trajectory(path_str)
        except Exception as exc:
            logger.warning(
                "TCKDB path_search: GSM trajectory parse failed for %s: %s",
                path_str, exc,
            )
            traj = None
        if traj:
            # xtb_gsm.process_run picks the TS guess at the middle
            # frame: ``int((len(traj) - 1) / 2) + 1``. Mirror that
            # here so the selected point matches the geometry that
            # actually became ``ts_opt``'s input.
            selected_index = int((len(traj) - 1) / 2) + 1
            for i, frame in enumerate(traj):
                try:
                    atom_only = xyz_to_str(frame)
                except Exception:
                    continue
                # TCKDB's ``GeometryPayload.xyz_text`` requires the
                # canonical XYZ-file shape (atom-count header + comment
                # + atom lines). ``xyz_to_str`` emits atom-lines only,
                # so route through ``_normalize_xyz_text`` to add the
                # header — same translation the species-side calcs use.
                xyz_text = _normalize_xyz_text(
                    atom_only, label=f"gsm_point_{i}",
                )
                if not xyz_text:
                    continue
                # path_coordinate intentionally left null: GSM node
                # ordering is represented by point_index; ARC does not
                # currently preserve GSM's internal reaction-coordinate
                # metric.
                point: dict[str, Any] = {
                    "point_index": i,
                    "geometry": {"xyz_text": xyz_text},
                }
                if i == selected_index:
                    point["is_ts_guess"] = True
                # Per-node energy / gradient (when preserved by the
                # patched ograd wrapper). The producer-side label
                # convention is that the on-disk node label NN maps
                # directly to the stringfile frame index — verified
                # against the smoke-test fixture; if a future GSM
                # version shifts this mapping, only mismatched indices
                # are silently skipped here, never invented.
                meta = node_metadata.get(i)
                if meta:
                    if "electronic_energy_hartree" in meta:
                        point["electronic_energy_hartree"] = meta[
                            "electronic_energy_hartree"
                        ]
                    if "max_gradient" in meta:
                        point["max_gradient"] = meta["max_gradient"]
                    if "rms_gradient" in meta:
                        point["rms_gradient"] = meta["rms_gradient"]
                points.append(point)

    if not points:
        # Single-point fallback: emit the chosen guess's geometry as
        # one TS-guess point. Honest about scope (lossy: we know one
        # point, not the full path) without faking other points.
        normalized_fallback = _normalize_xyz_text(
            fallback_xyz_text, label="ts_guess_point",
        )
        if not normalized_fallback:
            return None
        points = [{
            "point_index": 0,
            "geometry": {"xyz_text": normalized_fallback},
            "is_ts_guess": True,
        }]
        selected_index = 0

    # Relative-energy convention: all-or-none. Compute relative
    # energies for every point only when *every* point has an
    # electronic energy parsed; partial profiles are misleading
    # ("the dip at point 5 is real or just missing data?") so we
    # leave them all null in that case.
    energies = [
        p.get("electronic_energy_hartree") for p in points
    ]
    zero_e_h: float | None = None
    if energies and all(e is not None for e in energies):
        zero_e_h = min(energies)
        for p in points:
            e_h = p["electronic_energy_hartree"]
            p["relative_energy_kj_mol"] = (e_h - zero_e_h) * E_h_kJmol

    payload: dict[str, Any] = {
        "method": method,
        # ``converged: True`` is reliable here — the caller's gate
        # only invokes this builder when the chosen TSGuess has a
        # ``log_path``, which the producer sets only on
        # ``tsg.success=True`` (see ``xtb_gsm.process_run`` /
        # ``orca_neb`` adapters).
        "converged": True,
        "n_points": len(points),
        "points": points,
    }
    if zero_e_h is not None:
        payload["zero_energy_reference_hartree"] = zero_e_h
    # Static method properties (is_double_ended, source_endpoint_count)
    # — intrinsic to the algorithm, not the run. See
    # ``_PATH_SEARCH_METHOD_PROPERTIES``.
    payload.update(_PATH_SEARCH_METHOD_PROPERTIES.get(method, {}))
    if selected_index is not None and any(
        p["point_index"] == selected_index for p in points
    ):
        payload["selected_ts_point_index"] = selected_index
    return payload


def _resolve_irc_zero_energy_reference(
    *,
    output_doc: Mapping[str, Any],
    ts_record: Mapping[str, Any],
) -> float | None:
    """Pick the TS reference electronic energy for an IRC path.

    ARC runs IRC at the opt level (see ``level_kind="opt"`` at the IRC
    calculation construction site). The reference must therefore live at
    that same level, or the relative energies it produces would mix two
    levels of theory.

    Resolution order, conservative by design:

    1. ``ts_sp_result.electronic_energy_hartree`` *only if* the project
       SP level matches the project opt level (LoT keys equal).
    2. ``ts_record['opt_final_energy_hartree']`` — the TS opt's converged
       SCF, which is by construction at the opt level.
    3. ``None`` — never fabricate a reference. Downstream consumers
       interpret a null ``zero_energy_reference_hartree`` as "relative
       energies unavailable" and skip the ``relative_energy_kj_mol``
       per-point field accordingly.
    """
    opt_level = _resolve_level(output_doc, "opt")
    sp_level = output_doc.get("sp_level")
    sp_level = sp_level if isinstance(sp_level, Mapping) else None
    sp_energy = ts_record.get("sp_energy_hartree")
    if sp_energy is None:
        sp_energy = ts_record.get("electronic_energy_hartree")
    if sp_energy is not None and _level_keys_match(opt_level, sp_level):
        try:
            return float(sp_energy)
        except (TypeError, ValueError):
            pass
    opt_energy = ts_record.get("opt_final_energy_hartree")
    if opt_energy is not None:
        try:
            return float(opt_energy)
        except (TypeError, ValueError):
            pass
    return None


def _normalize_xyz_text(xyz: str | None, label: str | None) -> str | None:
    """Convert an ARC atom-only xyz string into TCKDB's standard XYZ format.

    Input shape (what ``xyz_to_str`` emits):
        ``"C 0.0 0.0 0.0\\nH 1.0 0.0 0.0"``
    Output shape (what TCKDB ``GeometryPayload.xyz_text`` expects):
        ``"<n_atoms>\\n<comment>\\n<atom lines>"``

    If the input already has a valid integer atom-count header, it's
    returned untouched. Returns ``None`` for null/empty input — the
    format-translation boundary between ARC's internal convention and
    the TCKDB schema, with no requirement that input be present.
    """
    if not xyz:
        return None
    text = str(xyz).strip()
    if not text:
        return None
    lines = text.splitlines()
    try:
        int(lines[0].strip())
        return text
    except (ValueError, IndexError):
        pass
    return f"{len(lines)}\n{label or ''}\n{text}"


def _require_xyz_text(record: Mapping[str, Any]) -> str:
    """Pull and normalize the species record's reference xyz, raising if absent.

    Wraps :func:`_normalize_xyz_text` for the conformer-level geometry
    case where missing xyz is a fatal error (the bundle requires a
    ``ConformerInBundle.geometry``). For optional per-calc input
    geometries, call ``_normalize_xyz_text`` directly so a missing xyz
    just yields ``None`` and the caller omits the field.
    """
    xyz = record.get("xyz")
    if not xyz:
        raise ValueError(
            f"output.yml record for label={record.get('label')!r} has no xyz; "
            "cannot build geometry payload."
        )
    text = _normalize_xyz_text(xyz, record.get("label"))
    if text is None:
        raise ValueError(
            f"output.yml record for label={record.get('label')!r} has empty xyz."
        )
    return text


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


_PUBLIC_REF_ECHO_KEYS = frozenset({"request", "requests", "payload", "payloads"})


def _extract_tckdb_public_refs(obj: Any) -> dict[str, list[str]]:
    """Collect returned TCKDB ``*_ref`` strings from a nested response body.

    The helper stores all returned refs it sees outside obvious echoed
    request/payload blocks. Values are grouped by pluralized key name and
    deduped in first-seen order, e.g. ``species_entry_ref`` becomes
    ``species_entry_refs``.
    """
    refs: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}

    def bucket_for(key: str) -> str:
        return key[:-4] + "_refs"

    def add(key: str, value: str) -> None:
        bucket = bucket_for(key)
        bucket_seen = seen.setdefault(bucket, set())
        if value in bucket_seen:
            return
        bucket_seen.add(value)
        refs.setdefault(bucket, []).append(value)

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_str = str(key)
                if key_str in _PUBLIC_REF_ECHO_KEYS:
                    continue
                if key_str.endswith("_ref") and isinstance(child, str):
                    add(key_str, child)
                    continue
                if child is None:
                    continue
                walk(child)
            return
        if isinstance(value, list):
            for item in value:
                walk(item)

    walk(obj)
    return refs


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
    "COMPUTED_REACTION_ENDPOINT",
    "COMPUTED_REACTION_KIND",
    "COMPUTED_SPECIES_ENDPOINT",
    "COMPUTED_SPECIES_KIND",
    "CONFORMER_UPLOAD_ENDPOINT",
    "PAYLOAD_KIND",
    "TCKDBAdapter",
    "UploadOutcome",
    "arc_to_tckdb_a_units",
    "arc_to_tckdb_ea_units",
]
