"""Idempotency-key generation for ARC -> TCKDB uploads.

Wraps :func:`tckdb_client.make_idempotency_key`. The composition rules
(*which* ARC-level facts go into the key) are intentionally kept here in
ARC, because they decide what "the same logical upload" means. Stable
across retries, distinct across logically-different inputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IdempotencyInputs:
    """Stable inputs that identify one logical conformer/calculation upload.

    Every field here MUST be deterministic for the same ARC output —
    no timestamps, no PIDs, no random suffixes. The combination is what
    the TCKDB server uses to deduplicate replays.
    """

    project_label: str | None
    species_label: str
    conformer_label: str
    payload_kind: str
    payload_hash: str

    @classmethod
    def from_payload(
        cls,
        *,
        project_label: str | None,
        species_label: str,
        conformer_label: str,
        payload_kind: str,
        payload: Any,
    ) -> "IdempotencyInputs":
        """Hash the canonical-JSON payload to produce a stable identity tail.

        Sorted-key JSON is used so dict ordering does not change the
        hash; this lets a re-built payload with the same content produce
        the same key on retry.
        """
        canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()[:16]
        return cls(
            project_label=project_label,
            species_label=species_label,
            conformer_label=conformer_label,
            payload_kind=payload_kind,
            payload_hash=digest,
        )


def build_idempotency_key(inputs: IdempotencyInputs) -> str:
    """Compose a stable idempotency key from :class:`IdempotencyInputs`.

    Shape: ``arc:<project>:<species>:<conformer>:<kind>:<payload-hash>``.

    - ``"arc"`` namespaces the key against other producers (e.g. RMG).
    - ``project_label`` keeps two runs of the same species under
      different projects from replaying each other.
    - The trailing ``payload_hash`` makes content-changes produce a
      *different* key, so edited geometries do not silently replay the
      old upload.

    ``make_idempotency_key`` sanitizes illegal characters and validates
    the result against the server constraint
    ``^[A-Za-z0-9._:-]{16,200}$``; callers don't need to pre-clean parts.
    """
    # Lazy import so arc.tckdb is importable when the adapter is unused
    # and tckdb-client is not installed.
    from tckdb_client import make_idempotency_key

    parts: list[str] = ["arc"]
    if inputs.project_label:
        parts.append(inputs.project_label)
    parts.extend(
        [
            inputs.species_label,
            inputs.conformer_label,
            inputs.payload_kind,
            inputs.payload_hash,
        ]
    )
    return make_idempotency_key(*parts)


__all__ = ["IdempotencyInputs", "build_idempotency_key"]
