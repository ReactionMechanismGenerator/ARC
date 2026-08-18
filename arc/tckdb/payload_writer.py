"""Write conformer/calculation payloads + sidecar metadata to disk.

Two invariants:

1. The payload JSON is written *once* and never rewritten — replay
   tooling needs the exact bytes that were (or would have been) sent.
2. The sidecar JSON is written eagerly as ``status="pending"`` *before*
   any network call, then updated in-place when the upload resolves.
   That way a crash mid-upload leaves a clear ``pending`` record on
   disk rather than no trace at all.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SAFE_LABEL = re.compile(r"[^A-Za-z0-9._-]+")


def _utcnow_iso() -> str:
    """Z-suffixed UTC timestamp; the standard for sidecar timestamps."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_label(label: str) -> str:
    """Sanitize a label for use as a filename component."""
    cleaned = _SAFE_LABEL.sub("-", label).strip("-.") or "unlabeled"
    return cleaned[:120]


def _fs_safe_key(key: str) -> str:
    """Sanitize an archive key (timestamp/idempotency-key) for a filename.

    Critically strips colons out of ISO-8601 timestamps (``:`` is illegal
    on some filesystems and confuses path tooling) by mapping every
    character outside ``[A-Za-z0-9._-]`` to ``-``.
    """
    cleaned = _SAFE_LABEL.sub("-", key).strip("-.") or "unkeyed"
    return cleaned[:120]


#: Bundle format the sidecars on disk conform to. Read by
#: ``tckdb-client``'s replay tool, which only attempts an upload when
#: this value is in its ``supported_format_versions`` list. Bump *only*
#: as a coordinated change with the client.
BUNDLE_FORMAT_VERSION = "0"


@dataclass
class SidecarMetadata:
    """On-disk record of one upload attempt; updated in place after upload."""

    payload_file: str
    endpoint: str
    idempotency_key: str
    payload_kind: str
    bundle_format_version: str = BUNDLE_FORMAT_VERSION
    created_at: str = field(default_factory=_utcnow_iso)
    uploaded_at: str | None = None
    status: str = "pending"
    response_status_code: int | None = None
    response_body: Any = None
    public_refs: dict[str, list[str]] = field(default_factory=dict)
    request_ids: list[dict[str, Any]] = field(default_factory=list)
    preflight: dict[str, Any] | None = None
    idempotency_replayed: bool | None = None
    last_error: str | None = None
    base_url: str | None = None
    # Phase-1 partial-upload marker. False for any record that
    # represents a complete TCKDB-uploadable bundle. True for records
    # the producer knows are deliberately incomplete (e.g. a
    # computed-reaction sidecar built after a TS-search failure, with
    # ts_label and kinetics stripped). Pairs with the ``.partial.``
    # filename infix; both must agree.
    is_partial: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WrittenPayload:
    """Handle returned by :meth:`PayloadWriter.write` for downstream upload + sidecar updates."""

    payload_path: Path
    sidecar_path: Path
    sidecar: SidecarMetadata


@dataclass
class ArtifactSidecarMetadata:
    """On-disk record of one artifact upload event; updated in place after upload.

    Distinct from :class:`SidecarMetadata`: artifact uploads target a
    concrete TCKDB calculation row (calculation_id) and carry the bytes
    hash so a replay tool can re-verify before retransmission.

    ``payload_kind`` is fixed to ``"calculation_artifact"`` so the replay
    tool can dispatch on the same field that conformer sidecars use —
    without it, every artifact sidecar lands in the ``__unknown__``
    bucket and is skipped.
    """

    endpoint: str
    idempotency_key: str
    calculation_id: int
    kind: str
    filename: str
    sha256: str
    bytes: int
    source_path: str | None = None
    payload_kind: str = "calculation_artifact"
    bundle_format_version: str = BUNDLE_FORMAT_VERSION
    created_at: str = field(default_factory=_utcnow_iso)
    uploaded_at: str | None = None
    status: str = "pending"
    response_status_code: int | None = None
    response_body: Any = None
    public_refs: dict[str, list[str]] = field(default_factory=dict)
    request_ids: list[dict[str, Any]] = field(default_factory=list)
    preflight: dict[str, Any] | None = None
    idempotency_replayed: bool | None = None
    last_error: str | None = None
    base_url: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WrittenArtifact:
    """Handle returned by :meth:`PayloadWriter.write_artifact` for downstream sidecar updates."""

    sidecar_path: Path
    sidecar: ArtifactSidecarMetadata


class PayloadWriter:
    """File-system surface for the conformer-calculation upload payload.

    Layout::

        <root>/conformer_calculation/<safe-label>.payload.json
        <root>/conformer_calculation/<safe-label>.meta.json

    The writer is intentionally dumb: no schema awareness, no upload
    logic. The adapter composes the payload, hands it here, then drives
    the upload separately.
    """

    SUBDIR = "conformer_calculation"
    ARTIFACT_SUBDIR = "calculation_artifacts"
    COMPUTED_SPECIES_SUBDIR = "computed_species"
    COMPUTED_REACTION_SUBDIR = "computed_reaction"
    TRANSITION_STATE_SUBDIR = "transition_state"
    PAYLOAD_SUFFIX = ".payload.json"
    SIDECAR_SUFFIX = ".meta.json"
    ARTIFACT_SIDECAR_SUFFIX = ".artifact.meta.json"
    # Inserted before PAYLOAD_SUFFIX/SIDECAR_SUFFIX when ``is_partial``
    # is true so a partial bundle is visually distinguishable from a
    # complete one on disk (in addition to the metadata flag).
    PARTIAL_INFIX = ".partial"
    # Subdirectory (under each payload bucket) into which a prior
    # payload+sidecar pair is moved before a re-run overwrites the
    # canonical path. Kept as a nested dir so directory-based discovery
    # tooling can exclude it with a single path check. NEVER holds the
    # current bundle — only historical ones. Discovery/replay tooling
    # MUST skip this subdir (the canonical path always holds the live
    # bundle).
    ARCHIVE_SUBDIR = "archive"

    def __init__(
        self,
        root_dir: str | os.PathLike[str],
        *,
        archive_previous: bool = True,
    ):
        self._root = Path(root_dir)
        # When True (default), an existing payload/sidecar pair is moved
        # into ``ARCHIVE_SUBDIR`` before a re-run overwrites it, so the
        # trail mapping "which local run produced which DB record" is
        # preserved (the TCKDB backend appends on re-upload; the local
        # side must not destroy the prior bundle in place).
        self._archive_previous = archive_previous
        # Monotonic last-resort disambiguator for the rare case where a
        # prior sidecar is unreadable AND the payload mtime is
        # unavailable AND the mtime-derived name still collides.
        self._archive_counter = 0

    @property
    def root(self) -> Path:
        return self._root

    def write(
        self,
        *,
        label: str,
        payload: Any,
        endpoint: str,
        idempotency_key: str,
        payload_kind: str = "conformer_calculation",
        base_url: str | None = None,
        subdir: str | None = None,
        is_partial: bool = False,
    ) -> WrittenPayload:
        """Write payload JSON and an initial ``pending`` sidecar atomically.

        ``subdir`` selects the on-disk bucket (``conformer_calculation``
        for the conformer endpoint, ``computed_species`` for the bundle
        endpoint). When omitted, the default ``SUBDIR`` is used to keep
        existing callers untouched.

        ``is_partial`` marks the record as a deliberately incomplete
        bundle (phase-1 partial computed-reaction sidecars). When true,
        the on-disk filenames gain a ``.partial`` infix and the sidecar
        metadata's ``is_partial`` flag is set. Default false preserves
        existing filenames and metadata exactly.

        Returns a :class:`WrittenPayload` carrying both paths and the
        in-memory sidecar dataclass. Callers update the sidecar via
        :meth:`update_sidecar` after the upload resolves.
        """
        directory = self._root / (subdir or self.SUBDIR)
        directory.mkdir(parents=True, exist_ok=True)
        safe = _safe_label(label)
        infix = self.PARTIAL_INFIX if is_partial else ""
        payload_path = directory / f"{safe}{infix}{self.PAYLOAD_SUFFIX}"
        sidecar_path = directory / f"{safe}{infix}{self.SIDECAR_SUFFIX}"

        # Preserve the prior run's trail: on a genuine overwrite (the
        # canonical payload already exists), MOVE the existing pair into
        # ``archive/`` before writing the new one. Move-then-write means a
        # crash can never lose the prior payload — it is either in place
        # or already archived, never gone. First writes archive nothing.
        if self._archive_previous and payload_path.exists():
            self._archive_existing(directory, safe, infix, payload_path, sidecar_path)

        self._write_json_atomic(payload_path, payload)

        sidecar = SidecarMetadata(
            payload_file=str(payload_path),
            endpoint=endpoint,
            idempotency_key=idempotency_key,
            payload_kind=payload_kind,
            base_url=base_url,
            is_partial=is_partial,
        )
        self._write_json_atomic(sidecar_path, sidecar.to_json())

        return WrittenPayload(
            payload_path=payload_path,
            sidecar_path=sidecar_path,
            sidecar=sidecar,
        )

    def update_sidecar(self, sidecar_path: Path, sidecar: SidecarMetadata) -> None:
        """Rewrite the sidecar in place with the latest status."""
        self._write_json_atomic(sidecar_path, sidecar.to_json())

    def _archive_existing(
        self,
        directory: Path,
        safe: str,
        infix: str,
        payload_path: Path,
        sidecar_path: Path,
    ) -> None:
        """Move an existing payload+sidecar pair into ``archive/``.

        The archived pair is named ``<safe><infix>.<key>.payload.json``
        (and the matching ``.meta.json``) where ``<key>`` is derived from
        the PREVIOUS sidecar's identity so the archived files line up with
        the DB record they produced:

        * the prior sidecar's ``uploaded_at`` timestamp if present (that
          is the moment the record was accepted server-side), else
        * the prior sidecar's ``idempotency_key`` (always present, and the
          exact key the server dedupes/appends on), else
        * — if the sidecar is missing/unreadable/corrupt — the old
          payload file's mtime, else a monotonic counter.

        The key is made filesystem-safe (no colons). If the target name
        would collide, a numeric suffix disambiguates. Files are *moved*
        (``os.replace``), never copied.
        """
        archive_dir = directory / self.ARCHIVE_SUBDIR
        key = self._archive_key(sidecar_path, payload_path)
        archive_dir.mkdir(parents=True, exist_ok=True)

        base = f"{safe}{infix}.{key}"
        archive_payload = archive_dir / f"{base}{self.PAYLOAD_SUFFIX}"
        archive_sidecar = archive_dir / f"{base}{self.SIDECAR_SUFFIX}"
        # Disambiguate against an existing archived pair sharing this key
        # (e.g. two re-runs before any upload, so both carry the same
        # idempotency key). Never overwrite an already-archived bundle.
        suffix = 1
        while archive_payload.exists() or archive_sidecar.exists():
            disambiguated = f"{base}.{suffix}"
            archive_payload = archive_dir / f"{disambiguated}{self.PAYLOAD_SUFFIX}"
            archive_sidecar = archive_dir / f"{disambiguated}{self.SIDECAR_SUFFIX}"
            suffix += 1

        # Move the sidecar first, then the payload. Ordering is safe
        # either way (the archived name is unique and distinct from the
        # canonical name), but moving the payload last means the canonical
        # payload is the last thing to disappear before the new write.
        if sidecar_path.exists():
            os.replace(sidecar_path, archive_sidecar)
        os.replace(payload_path, archive_payload)

    def _archive_key(self, sidecar_path: Path, payload_path: Path) -> str:
        """Derive a filesystem-safe archive key from the prior sidecar.

        Prefers the prior upload timestamp, falls back to the idempotency
        key, then the payload mtime, then a monotonic counter — never
        raises, so a corrupt sidecar can never lose the payload file.
        """
        key: str | None = None
        try:
            data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict):
            uploaded_at = data.get("uploaded_at")
            if isinstance(uploaded_at, str) and uploaded_at.strip():
                key = uploaded_at
            else:
                idempotency_key = data.get("idempotency_key")
                if isinstance(idempotency_key, str) and idempotency_key.strip():
                    key = idempotency_key
        if key is None:
            # Sidecar missing/unreadable/corrupt or lacking usable
            # identity: fall back to the payload's mtime.
            try:
                mtime = payload_path.stat().st_mtime
                key = datetime.fromtimestamp(mtime, timezone.utc).strftime(
                    "%Y%m%dT%H%M%S"
                )
            except OSError:
                key = None
        if key is None:
            self._archive_counter += 1
            key = f"prev{self._archive_counter}"
        return _fs_safe_key(key)

    def write_artifact_sidecar(
        self,
        *,
        species_label: str,
        calculation_id: int,
        kind: str,
        filename: str,
        sha256: str,
        bytes_: int,
        endpoint: str,
        idempotency_key: str,
        source_path: str | None = None,
        base_url: str | None = None,
    ) -> WrittenArtifact:
        """Write a ``pending`` artifact sidecar before the network call.

        One sidecar per (species, calculation, kind) tuple. The same
        species can have multiple sidecars across calculations (opt /
        freq / sp) and across kinds (output_log / checkpoint). The
        kind+calc-id+species combination keeps filenames distinct and
        makes the replay path scriptable.
        """
        directory = self._root / self.ARTIFACT_SUBDIR
        directory.mkdir(parents=True, exist_ok=True)
        safe_species = _safe_label(species_label)
        safe_kind = _safe_label(kind)
        sidecar_name = f"{safe_species}.calc{calculation_id}.{safe_kind}{self.ARTIFACT_SIDECAR_SUFFIX}"
        sidecar_path = directory / sidecar_name

        sidecar = ArtifactSidecarMetadata(
            endpoint=endpoint,
            idempotency_key=idempotency_key,
            calculation_id=calculation_id,
            kind=kind,
            filename=filename,
            sha256=sha256,
            bytes=bytes_,
            source_path=source_path,
            base_url=base_url,
        )
        self._write_json_atomic(sidecar_path, sidecar.to_json())
        return WrittenArtifact(sidecar_path=sidecar_path, sidecar=sidecar)

    def update_artifact_sidecar(
        self, sidecar_path: Path, sidecar: ArtifactSidecarMetadata
    ) -> None:
        """Rewrite the artifact sidecar in place with the latest status."""
        self._write_json_atomic(sidecar_path, sidecar.to_json())

    @staticmethod
    def _write_json_atomic(path: Path, data: Any) -> None:
        """Write JSON via tmp+rename so a crash mid-write cannot leave a partial file."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
        os.replace(tmp, path)


def should_replay_sidecar(
    meta: "SidecarMetadata | dict[str, Any] | None",
    *,
    include_partial: bool = False,
) -> bool:
    """Replay-tool gate: should this sidecar be re-POSTed to TCKDB?

    The contract a discovery-based replay tool MUST honor: if a
    sidecar is marked ``is_partial: true`` (or the on-disk file's name
    carries the ``.partial`` infix), do **not** replay it as a
    complete bundle. Partial computed-reaction sidecars omit
    ``transition_state`` and ``kinetics`` and have not been validated
    against the TCKDB server's acceptance of ``transition_state=null``
    payloads. Replaying one as a normal upload would either silently
    create an incomplete record server-side or error out at validation.

    ``meta`` accepts either an in-memory :class:`SidecarMetadata` or
    the parsed JSON dict from disk, so callers can use this either
    after :meth:`PayloadWriter.write` or after re-reading a
    ``*.meta.json`` file in a separate process.

    ``include_partial=True`` is an explicit opt-in for a future
    server-supported partial-replay path. It exists as a kwarg today
    so the contract is greppable, but no replay tool inside this repo
    sets it. External tools (e.g. ``tckdb-client``) MUST default to
    ``False`` and require an explicit user flag (e.g.
    ``--include-partial``) before passing ``True``.

    Returns ``True`` when the record is safe to POST as-is, ``False``
    when the replay path should skip and log instead.
    """
    if meta is None:
        return False
    is_partial = (
        meta.is_partial
        if isinstance(meta, SidecarMetadata)
        else bool(meta.get("is_partial", False))
    )
    # Producer keeps ``is_partial`` and the filename infix in sync (see
    # SidecarMetadata docstring), but a hand-edited or corrupted sidecar
    # could disagree. Either signal alone marks the record partial —
    # bias toward the safe "skip" answer rather than re-POSTing an
    # incomplete bundle.
    payload_file = (
        meta.payload_file
        if isinstance(meta, SidecarMetadata)
        else str(meta.get("payload_file") or "")
    )
    if PayloadWriter.PARTIAL_INFIX in payload_file:
        is_partial = True
    if is_partial and not include_partial:
        return False
    return True


__all__ = [
    "ArtifactSidecarMetadata",
    "PayloadWriter",
    "SidecarMetadata",
    "WrittenArtifact",
    "WrittenPayload",
    "should_replay_sidecar",
]
