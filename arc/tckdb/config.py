"""Configuration for the ARC TCKDB adapter.

The adapter is opt-in. If no ``tckdb`` block is present in the ARC input
(or ``enabled`` is false), :func:`TCKDBConfig.from_dict` returns ``None``
and the adapter is a no-op.

API keys are never read from input files. The config carries only the
*name* of the env var (``api_key_env``); the adapter resolves the key
at upload time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping


DEFAULT_PAYLOAD_DIR = "tckdb_payloads"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_API_KEY_ENV = "TCKDB_API_KEY"


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
        return cls(
            enabled=True,
            base_url=base_url,
            api_key_env=str(raw.get("api_key_env", DEFAULT_API_KEY_ENV)),
            payload_dir=str(raw.get("payload_dir", DEFAULT_PAYLOAD_DIR)),
            upload=bool(raw.get("upload", True)),
            strict=bool(raw.get("strict", False)),
            timeout_seconds=float(raw.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)),
            project_label=raw.get("project_label"),
        )

    def resolve_api_key(self) -> str | None:
        """Read the API key from the configured env var. Never logged."""
        return os.environ.get(self.api_key_env)
