"""Configuration for the ARC TCKDB adapter.

The adapter is opt-in. If no ``tckdb`` block is present in the ARC input
(or ``enabled`` is false), :func:`TCKDBConfig.from_dict` returns ``None``
and the adapter is a no-op.

API keys are never read from input files. The config carries only the
*name* of the env var (``api_key_env``); the adapter resolves the key
at upload time.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger("arc")


DEFAULT_PAYLOAD_DIR = "tckdb_payloads"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_API_KEY_ENV = "TCKDB_API_KEY"

# Upload-mode switch. ``conformer`` (default) keeps the existing
# /uploads/conformers + per-artifact path. ``computed_species`` builds
# one self-contained bundle and posts it to /uploads/computed-species.
# A run can use either; mixing per-species is intentionally not
# supported — pick one mode per ARC run.
UPLOAD_MODE_CONFORMER = "conformer"
UPLOAD_MODE_COMPUTED_SPECIES = "computed_species"
VALID_UPLOAD_MODES = frozenset({UPLOAD_MODE_CONFORMER, UPLOAD_MODE_COMPUTED_SPECIES})

# Mirrors the server-side ArtifactKind enum
# (backend/app/db/models/common.py:147 in TCKDB_v2). Keeping the source
# of truth here keeps the adapter loud-failing on unknown kinds at
# config parse time rather than at HTTP 422 time.
VALID_ARTIFACT_KINDS = frozenset({
    "input",
    "output_log",
    "checkpoint",
    "formatted_checkpoint",
    "ancillary",
})

# Subset of VALID_ARTIFACT_KINDS that ARC actually has the codepath to
# produce. Listing a valid-but-not-implemented kind in input.yml is
# permitted (so users can opt into future kinds early) but warned at
# parse time so the silent-zero-uploads outcome is visible.
#
# As new upload paths land, add their kinds here. Today:
#   - output_log: ESS log file, from record["opt_log"]/["freq_log"]/["sp_log"]
#   - input:      ESS input deck (input.gjf / ZMAT / input.in), sibling of opt_log
IMPLEMENTED_ARTIFACT_KINDS = frozenset({"output_log", "input"})

DEFAULT_ARTIFACT_KINDS: tuple[str, ...] = ("output_log",)
DEFAULT_ARTIFACT_MAX_SIZE_MB = 50


@dataclass(frozen=True)
class TCKDBArtifactConfig:
    """Per-artifact upload knobs.

    Defaults are conservative: artifact upload is opt-in (``upload=False``)
    and only ``output_log`` is in scope. Users opt in by adding an
    ``artifacts:`` sub-block to the ``tckdb`` config.
    """

    upload: bool = False
    kinds: tuple[str, ...] = DEFAULT_ARTIFACT_KINDS
    max_size_mb: int = DEFAULT_ARTIFACT_MAX_SIZE_MB

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "TCKDBArtifactConfig":
        if not raw:
            return cls()
        kinds_raw = raw.get("kinds", DEFAULT_ARTIFACT_KINDS)
        if isinstance(kinds_raw, str):
            kinds_raw = (kinds_raw,)
        kinds = tuple(str(k) for k in kinds_raw)
        unknown = [k for k in kinds if k not in VALID_ARTIFACT_KINDS]
        if unknown:
            raise ValueError(
                f"tckdb.artifacts.kinds contains unknown kind(s): {unknown}. "
                f"Valid kinds: {sorted(VALID_ARTIFACT_KINDS)}."
            )
        not_implemented = [k for k in kinds if k not in IMPLEMENTED_ARTIFACT_KINDS]
        if not_implemented:
            logger.warning(
                "tckdb.artifacts.kinds includes kind(s) the TCKDB server accepts "
                "but ARC doesn't yet produce uploads for: %s. Currently implemented: %s. "
                "These will be silently skipped at upload time.",
                not_implemented, sorted(IMPLEMENTED_ARTIFACT_KINDS),
            )
        max_size_mb = int(raw.get("max_size_mb", DEFAULT_ARTIFACT_MAX_SIZE_MB))
        if max_size_mb <= 0:
            raise ValueError(
                f"tckdb.artifacts.max_size_mb must be > 0; got {max_size_mb}."
            )
        return cls(
            upload=bool(raw.get("upload", False)),
            kinds=kinds,
            max_size_mb=max_size_mb,
        )


@dataclass(frozen=True)
class TCKDBConfig:
    """ARC-side TCKDB adapter configuration.

    ``enabled`` is the master switch. ``upload`` controls whether the
    adapter contacts the network at all — when false, payloads are
    written to disk and the sidecar is marked ``skipped``.

    Strict mode raises on upload failure; the default is to log a
    warning, record the error in the sidecar, and let ARC continue.
    """

    enabled: bool = False
    base_url: str | None = None
    api_key_env: str = DEFAULT_API_KEY_ENV
    payload_dir: str = DEFAULT_PAYLOAD_DIR
    upload: bool = True
    strict: bool = False
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    project_label: str | None = field(
        default=None,
        metadata={"help": "Optional ARC project/run label baked into idempotency keys."},
    )
    upload_mode: str = UPLOAD_MODE_CONFORMER
    artifacts: TCKDBArtifactConfig = field(default_factory=TCKDBArtifactConfig)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "TCKDBConfig | None":
        """Build a config from a raw mapping, or return ``None`` when disabled.

        Returning ``None`` for the disabled case lets callers write
        ``if cfg is None: return`` rather than checking a flag.
        """
        if not raw:
            return None
        if not raw.get("enabled", False):
            return None
        base_url = raw.get("base_url")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("tckdb.base_url is required when tckdb.enabled is true.")
        upload_mode = str(raw.get("upload_mode", UPLOAD_MODE_CONFORMER))
        if upload_mode not in VALID_UPLOAD_MODES:
            raise ValueError(
                f"tckdb.upload_mode must be one of {sorted(VALID_UPLOAD_MODES)}; "
                f"got {upload_mode!r}."
            )
        return cls(
            enabled=True,
            base_url=base_url,
            api_key_env=str(raw.get("api_key_env", DEFAULT_API_KEY_ENV)),
            payload_dir=str(raw.get("payload_dir", DEFAULT_PAYLOAD_DIR)),
            upload=bool(raw.get("upload", True)),
            strict=bool(raw.get("strict", False)),
            timeout_seconds=float(raw.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)),
            project_label=raw.get("project_label"),
            upload_mode=upload_mode,
            artifacts=TCKDBArtifactConfig.from_dict(raw.get("artifacts")),
        )

    def resolve_api_key(self) -> str | None:
        """Read the API key from the configured env var. Never logged."""
        return os.environ.get(self.api_key_env)
