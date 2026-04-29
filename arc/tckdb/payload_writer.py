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
    idempotency_replayed: bool | None = None
    last_error: str | None = None
    base_url: str | None = None

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
    PAYLOAD_SUFFIX = ".payload.json"
    SIDECAR_SUFFIX = ".meta.json"
    ARTIFACT_SIDECAR_SUFFIX = ".artifact.meta.json"

    def __init__(self, root_dir: str | os.PathLike[str]):
        self._root = Path(root_dir)

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
    ) -> WrittenPayload:
        """Write payload JSON and an initial ``pending`` sidecar atomically.

        ``subdir`` selects the on-disk bucket (``conformer_calculation``
        for the conformer endpoint, ``computed_species`` for the bundle
        endpoint). When omitted, the default ``SUBDIR`` is used to keep
        existing callers untouched.

        Returns a :class:`WrittenPayload` carrying both paths and the
        in-memory sidecar dataclass. Callers update the sidecar via
        :meth:`update_sidecar` after the upload resolves.
        """
        directory = self._root / (subdir or self.SUBDIR)
        directory.mkdir(parents=True, exist_ok=True)
        safe = _safe_label(label)
        payload_path = directory / f"{safe}{self.PAYLOAD_SUFFIX}"
        sidecar_path = directory / f"{safe}{self.SIDECAR_SUFFIX}"

        self._write_json_atomic(payload_path, payload)

        sidecar = SidecarMetadata(
            payload_file=str(payload_path),
            endpoint=endpoint,
            idempotency_key=idempotency_key,
            payload_kind=payload_kind,
            base_url=base_url,
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


__all__ = [
    "ArtifactSidecarMetadata",
    "PayloadWriter",
    "SidecarMetadata",
    "WrittenArtifact",
    "WrittenPayload",
]
