"""Adapter that turns ARC objects into TCKDB conformer-upload payloads.

The adapter is the only ARC module that knows the shape of a TCKDB
upload. It builds a JSON payload matching ``ConformerUploadRequest``,
writes it to disk, and (optionally) hands it to ``tckdb-client``.

Three guarantees:

1. If the adapter is disabled or no config is provided, it is a no-op.
2. The payload is on disk *before* any network call. Replay tooling
   only needs ``payload_file + endpoint + idempotency_key``.
3. By default, an upload failure is logged + recorded in the sidecar
   but does not raise. ``strict=True`` flips that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from arc.tckdb.config import TCKDBConfig
from arc.tckdb.idempotency import IdempotencyInputs, build_idempotency_key
from arc.tckdb.payload_writer import (
    PayloadWriter,
    SidecarMetadata,
    WrittenPayload,
)


logger = logging.getLogger("arc")

CONFORMER_UPLOAD_ENDPOINT = "/uploads/conformers"
PAYLOAD_KIND = "conformer_calculation"


@dataclass(frozen=True)
class UploadOutcome:
    """Result of one adapter invocation; mirrors the sidecar status."""

    status: str  # pending | uploaded | failed | skipped
    payload_path: Path
    sidecar_path: Path
    idempotency_key: str
    error: str | None = None
    response: Any = None


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

    def submit_conformer(
        self,
        *,
        species,
        level,
        xyz: str | Mapping[str, Any] | None = None,
        conformer_index: int = 0,
        calculation_type: str = "opt",
        calculation_quality: str = "raw",
        opt_result: Mapping[str, Any] | None = None,
        freq_result: Mapping[str, Any] | None = None,
        sp_result: Mapping[str, Any] | None = None,
        arc_version: str | None = None,
        arc_git_commit: str | None = None,
        extra_label: str | None = None,
    ) -> UploadOutcome | None:
        """Build, write, and (if configured) upload one conformer payload.

        Returns ``None`` if the adapter is disabled (so callers can write
        ``adapter.submit_conformer(...)`` without an enabled-check).
        """
        if not self._config.enabled:
            return None

        payload = self._build_payload(
            species=species,
            level=level,
            xyz=xyz,
            calculation_type=calculation_type,
            calculation_quality=calculation_quality,
            opt_result=opt_result,
            freq_result=freq_result,
            sp_result=sp_result,
            arc_version=arc_version,
            arc_git_commit=arc_git_commit,
        )

        species_label = getattr(species, "label", None) or "unlabeled"
        conformer_label = extra_label or f"conf{conformer_index}"
        idempotency_inputs = IdempotencyInputs.from_payload(
            project_label=self._config.project_label,
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

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        *,
        species,
        level,
        xyz,
        calculation_type: str,
        calculation_quality: str,
        opt_result: Mapping[str, Any] | None,
        freq_result: Mapping[str, Any] | None,
        sp_result: Mapping[str, Any] | None,
        arc_version: str | None,
        arc_git_commit: str | None,
    ) -> dict[str, Any]:
        """Construct the JSON payload accepted by ``ConformerUploadRequest``.

        Designed to fail soft on missing optional ARC attributes — many
        ARC species objects in mid-flight don't carry every field.
        Required-by-schema fields (smiles, charge, multiplicity, xyz_text,
        method, software name) come from the species/level objects.
        """
        species_entry = self._species_entry_payload(species)
        geometry_payload = {"xyz_text": _coerce_xyz_text(xyz, species)}
        calculation_payload = self._calculation_payload(
            level=level,
            calculation_type=calculation_type,
            calculation_quality=calculation_quality,
            opt_result=opt_result,
            freq_result=freq_result,
            sp_result=sp_result,
            arc_version=arc_version,
            arc_git_commit=arc_git_commit,
        )

        payload: dict[str, Any] = {
            "species_entry": species_entry,
            "geometry": geometry_payload,
            "calculation": calculation_payload,
            "scientific_origin": "computed",
        }
        label = getattr(species, "label", None)
        if label:
            payload["label"] = str(label)[:64]
        return payload

    @staticmethod
    def _species_entry_payload(species) -> dict[str, Any]:
        smiles = _resolve_smiles(species)
        if not smiles:
            raise ValueError(
                "TCKDB upload requires a SMILES on the species; "
                f"got species.label={getattr(species, 'label', None)!r} with no resolvable SMILES."
            )
        is_ts = bool(getattr(species, "is_ts", False))
        return {
            "molecule_kind": "molecule",
            "smiles": smiles,
            "charge": int(getattr(species, "charge", 0) or 0),
            "multiplicity": int(getattr(species, "multiplicity", 1) or 1),
            "species_entry_kind": "transition_state" if is_ts else "minimum",
        }

    @staticmethod
    def _calculation_payload(
        *,
        level,
        calculation_type: str,
        calculation_quality: str,
        opt_result: Mapping[str, Any] | None,
        freq_result: Mapping[str, Any] | None,
        sp_result: Mapping[str, Any] | None,
        arc_version: str | None,
        arc_git_commit: str | None,
    ) -> dict[str, Any]:
        method = getattr(level, "method", None) if level is not None else None
        if not method:
            raise ValueError("TCKDB upload requires a level-of-theory method.")
        software_name = getattr(level, "software", None) if level is not None else None
        if not software_name:
            raise ValueError("TCKDB upload requires the ESS name (level.software).")

        level_of_theory: dict[str, Any] = {"method": method}
        for src, dst in (
            ("basis", "basis"),
            ("auxiliary_basis", "aux_basis"),
            ("cabs", "cabs_basis"),
            ("dispersion", "dispersion"),
            ("solvent", "solvent"),
            ("solvation_method", "solvent_model"),
        ):
            value = getattr(level, src, None)
            if value:
                level_of_theory[dst] = value

        software_release: dict[str, Any] = {"name": software_name}
        version = getattr(level, "software_version", None)
        if version is not None:
            software_release["version"] = str(version)

        calc: dict[str, Any] = {
            "type": calculation_type,
            "quality": calculation_quality,
            "software_release": software_release,
            "level_of_theory": level_of_theory,
        }

        if arc_version or arc_git_commit:
            wt: dict[str, Any] = {"name": "ARC"}
            if arc_version:
                wt["version"] = arc_version
            if arc_git_commit:
                wt["git_commit"] = arc_git_commit
            calc["workflow_tool_release"] = wt

        if calculation_type == "opt" and opt_result:
            calc["opt_result"] = dict(opt_result)
        if calculation_type == "freq" and freq_result:
            calc["freq_result"] = dict(freq_result)
        if calculation_type == "sp" and sp_result:
            calc["sp_result"] = dict(sp_result)

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

    def _upload(self, written: WrittenPayload, payload: dict[str, Any]) -> UploadOutcome:
        from arc.tckdb.payload_writer import _utcnow_iso  # local import keeps deps tidy

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
                CONFORMER_UPLOAD_ENDPOINT,
                json=payload,
                idempotency_key=sc.idempotency_key,
            )
        except Exception as exc:
            client_close = getattr(client, "close", None)
            if callable(client_close):
                try:
                    client_close()
                except Exception:  # pragma: no cover - close errors swallowed
                    logger.debug("TCKDB client close errored after upload failure", exc_info=True)
            return self._record_failure(written, str(exc), exc)
        else:
            client_close = getattr(client, "close", None)
            if callable(client_close):
                try:
                    client_close()
                except Exception:  # pragma: no cover - close errors swallowed
                    logger.debug("TCKDB client close errored after upload success", exc_info=True)

        sc.status = "uploaded"
        sc.uploaded_at = _utcnow_iso()
        sc.response_status_code = getattr(response, "status_code", None)
        sc.response_body = _summarize_response_body(getattr(response, "data", None))
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
        return UploadOutcome(
            status="uploaded",
            payload_path=written.payload_path,
            sidecar_path=written.sidecar_path,
            idempotency_key=sc.idempotency_key,
            response=sc.response_body,
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
        # Lazy import: keep tckdb_client out of the import path when the
        # adapter is unused.
        from tckdb_client import TCKDBClient

        return TCKDBClient(
            self._config.base_url,
            api_key=api_key,
            timeout=self._config.timeout_seconds,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_smiles(species) -> str | None:
    """Best-effort SMILES extraction from an ARC species or duck-typed stand-in."""
    smiles = getattr(species, "smiles", None)
    if smiles:
        return str(smiles)
    mol = getattr(species, "mol", None)
    if mol is None:
        return None
    to_smiles = getattr(mol, "to_smiles", None) or getattr(mol, "smiles", None)
    if callable(to_smiles):
        try:
            return str(to_smiles())
        except Exception:
            logger.debug("mol.to_smiles() raised; falling back to None", exc_info=True)
            return None
    if isinstance(to_smiles, str):
        return to_smiles
    return None


def _coerce_xyz_text(xyz, species) -> str:
    """Accept xyz as a string or ARC xyz dict; fall back to species attrs."""
    candidate = xyz
    if candidate is None:
        candidate = (
            getattr(species, "final_xyz", None)
            or getattr(species, "initial_xyz", None)
        )
    if candidate is None:
        raise ValueError("TCKDB upload requires xyz coordinates.")
    if isinstance(candidate, str):
        text = candidate.strip()
        if not text:
            raise ValueError("TCKDB upload requires non-empty xyz coordinates.")
        return text
    # ARC xyz dict: {'symbols': (...), 'coords': ((x,y,z),...), ...}
    symbols = candidate.get("symbols") if isinstance(candidate, Mapping) else None
    coords = candidate.get("coords") if isinstance(candidate, Mapping) else None
    if not symbols or not coords:
        raise ValueError(f"Unrecognized xyz container: {type(candidate).__name__}")
    lines = [
        f"{sym} {x:.8f} {y:.8f} {z:.8f}"
        for sym, (x, y, z) in zip(symbols, coords)
    ]
    return "\n".join(lines)


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


__all__ = ["TCKDBAdapter", "UploadOutcome", "CONFORMER_UPLOAD_ENDPOINT", "PAYLOAD_KIND"]
