# ARC ↔ TCKDB integration (v0)

This is the v0 of the ARC-side TCKDB adapter. It builds a
**conformer/calculation upload payload** from ARC objects, writes it to
disk, and (optionally) uploads it via
[`tckdb-client`](https://github.com/tckdb).

## What v0 does

1. Builds a JSON payload matching TCKDB's `ConformerUploadRequest`.
2. Writes the payload to disk *before* attempting any network call.
3. Writes a sidecar `*.meta.json` with the endpoint, idempotency key,
   timestamps, and upload status.
4. Optionally uploads via `tckdb-client`, sending a stable idempotency
   key.
5. Records success/failure in the sidecar.
6. Does **not** fail ARC by default if the upload errors — strict mode
   is opt-in.

## What v0 does not do

- thermo upload
- kinetics upload
- bundle submission
- hosted contribution flow
- background retry daemon
- direct DB access
- chemistry logic in `tckdb-client`

These are deferred to later milestones.

## Configuration

Add a `tckdb` block to your ARC input. The block is **optional** — if
absent or `enabled: false`, the adapter is a no-op.

```yaml
tckdb:
  enabled: true
  base_url: "http://localhost:8000/api/v1"
  api_key_env: "TCKDB_API_KEY"
  payload_dir: "tckdb_payloads"
  upload: true
  strict: false
  timeout_seconds: 30
  project_label: "my-project"   # optional; baked into idempotency key
```

| Field             | Default              | Notes                                                     |
| ----------------- | -------------------- | --------------------------------------------------------- |
| `enabled`         | `false`              | Master switch.                                            |
| `base_url`        | _(required)_         | TCKDB API root.                                           |
| `api_key_env`     | `TCKDB_API_KEY`      | Env var holding the API key. Never store the key in YAML. |
| `payload_dir`     | `tckdb_payloads`     | Relative paths resolve under the ARC project directory.   |
| `upload`          | `true`               | If `false`, write payload only and mark sidecar `skipped`.|
| `strict`          | `false`              | If `true`, upload failure raises.                         |
| `timeout_seconds` | `30`                 | Per-request timeout.                                      |
| `project_label`   | `null`               | Optional run/project tag baked into the idempotency key.  |

### API key

The adapter reads `os.environ[api_key_env]`. If the var is unset and
`upload: true`, the adapter records a failed sidecar (or raises in
strict mode) and never contacts the network.

```bash
export TCKDB_API_KEY="tck_replace_me"
```

## Local TCKDB example

Run TCKDB locally, then point ARC at it:

```bash
# in TCKDB_v2/
docker compose -f docker-compose.local.yml up -d

# in your ARC input:
tckdb:
  enabled: true
  base_url: "http://localhost:8000/api/v1"
  api_key_env: "TCKDB_API_KEY"
```

## Payload directory layout

```
<project>/tckdb_payloads/
  conformer_calculation/
    <species>.<conformer>.payload.json
    <species>.<conformer>.meta.json
```

- The `*.payload.json` file is **immutable** after first write — it is
  the exact JSON that was (or would have been) sent.
- The `*.meta.json` sidecar is written eagerly as `pending` and updated
  in place once the upload resolves.

## Sidecar shape

```json
{
  "payload_file": ".../ethanol.conf0.payload.json",
  "endpoint": "/uploads/conformers",
  "idempotency_key": "arc:my-project:ethanol:conf0:conformer_calculation:abc1234567890def",
  "payload_kind": "conformer_calculation",
  "created_at": "2026-04-26T12:00:00Z",
  "uploaded_at": "2026-04-26T12:00:01Z",
  "status": "uploaded",
  "response_status_code": 201,
  "response_body": {"...": "..."},
  "idempotency_replayed": false,
  "last_error": null,
  "base_url": "http://localhost:8000/api/v1"
}
```

`status` ∈ `pending | uploaded | failed | skipped`.

## Idempotency

Keys have the shape

```
arc:<project>:<species>:<conformer>:<kind>:<payload-hash>
```

and are generated via `tckdb_client.make_idempotency_key`. The
`payload-hash` tail is a SHA-256 prefix over the canonical-JSON payload,
so:

- **same** logical payload → **same** key (server replays the previous
  response, marking `idempotency_replayed: true`).
- **changed** geometry/level/results → **different** key (a real new
  upload).

The key contains no timestamps, no PIDs, no random suffixes.

## Upload failure behavior

| Mode             | On upload error            |
| ---------------- | -------------------------- |
| `strict: false`  | Log a warning, set sidecar `status=failed` with `last_error`, return an `UploadOutcome(status="failed", ...)`. ARC continues. |
| `strict: true`   | Same sidecar update, then re-raise the underlying exception.            |

## Replay idea

There is no built-in retry daemon yet. A future replay tool only needs:

```text
sidecar.payload_file   +
sidecar.endpoint       +
sidecar.idempotency_key
```

plus a configured `base_url` and API key. Because the idempotency key is
stable and the payload is byte-identical to the first attempt, a replay
will produce either a fresh write or a server-side replay.

## Programmatic use

```python
from arc.tckdb import TCKDBAdapter, TCKDBConfig

cfg = TCKDBConfig.from_dict(arc_input.get("tckdb"))
if cfg is not None:
    adapter = TCKDBAdapter(cfg, project_directory=run.project_directory)
    adapter.submit_conformer(
        species=species,
        level=level,
        xyz=species.final_xyz,
        conformer_index=0,
        calculation_type="opt",
        opt_result={"converged": True, "final_energy_hartree": -154.5},
        arc_version=ARC_VERSION,
    )
```

`submit_conformer` returns `None` when the adapter is disabled, or an
`UploadOutcome` carrying the resulting status, payload path, sidecar
path, and idempotency key.

## Dependency policy

The adapter is an **optional** integration. The dependency policy is:

- **ARC core does not require `tckdb-client`.** Importing `arc.tckdb` is
  safe in environments that don't have it installed; both
  `tckdb_client` references are lazy.
- **The TCKDB integration requires `tckdb-client`.** This is true even
  when `upload: false`, because ARC still generates an idempotency key
  via `tckdb_client.make_idempotency_key` and records it in the sidecar
  for later replay. In other words: as soon as you set
  `tckdb.enabled: true`, the package becomes a hard requirement.
- **Until `tckdb-client` is published**, install it from the local
  TCKDB checkout:

  ```bash
  conda run -n arc_env pip install -e /path/to/TCKDB_v2/clients/python/tckdb-client
  ```

- **Once published**, ARC's `environment.yml` should declare it as an
  optional integration dependency:

  ```yaml
    - pip:
        - "tckdb-client>=0.1,<0.2"
  ```

  The `>=0.1,<0.2` range is the v0 client API; bump deliberately on a
  major-version client release.

ARC's unit tests for the adapter inject a stub client via
`TCKDBAdapter(..., client_factory=...)`. The idempotency-key tests still
need `tckdb-client` because key construction is part of the
adapter contract, but no test contacts the network.
