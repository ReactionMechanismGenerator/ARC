# ARC ↔ TCKDB integration (v0)

This is the v0 of the ARC-side TCKDB adapter. It builds a
**conformer/calculation upload payload** from ARC objects, writes it to
disk, and (optionally) uploads it via
[`tckdb-client`](https://github.com/tckdb).

For the semantic boundary between ARC scientific outputs and ARC
workflow narrative, see [`tckdb_upload_boundary.md`](tckdb_upload_boundary.md).
In short, TCKDB receives final scientific records and
reproducibility-critical evidence, not ARC's internal workflow history.

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

## Calculation constraints

ARC emits held-fixed coordinate constraints (Gaussian ModRedundant `F`
lines and ORCA `%geom Constraints { … C }` blocks) into each calculation
payload's top-level `constraints[]` field. Producer behavior:

- Gaussian input decks are the preferred parse source (the literal block
  ARC wrote); the corresponding log file is used as a fallback.
- Only frozen coordinates become `constraints[]`. Active scan
  coordinates remain in `scan_result.coordinates[]` — never both.
- Atom indices are 1-based per TCKDB convention. ORCA decks are 0-based
  on disk and converted at the parser boundary.
- Supported `constraint_kind` values: `cartesian_atom`, `bond`, `angle`,
  `dihedral`. Gaussian `D` always emits `dihedral` — ARC does not
  invent `improper` classification heuristics from a plain
  `D i j k l F` line.
- Constraint extraction is best-effort: parse failures emit a warning
  and an empty list, never failing ARC job generation or TCKDB payload
  emission.

ORCA support is currently defensive — ARC's ORCA adapter does not
itself emit `%geom Constraints` blocks today, so the parser primarily
exists to round-trip user-supplied decks.

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
  api_key_file: "/home/me/code/TCKDB_v2/backend/.tckdb_api_key"   # optional
  api_key_env_file: "/home/me/code/TCKDB_v2/backend/.tckdb_auth.env"  # optional
  payload_dir: "tckdb_payloads"
  upload: true
  preflight: true
  strict: false
  timeout_seconds: 30
  project_label: "my-project"   # optional; baked into idempotency key
  artifacts:                    # optional sub-block; opt-in
    upload: true
    kinds: ["output_log", "input"]
    max_size_mb: 50
```

| Field              | Default              | Notes                                                     |
| ------------------ | -------------------- | --------------------------------------------------------- |
| `enabled`          | `false`              | Master switch.                                            |
| `base_url`         | _(required)_         | TCKDB API root.                                           |
| `api_key_env`      | `TCKDB_API_KEY`      | Env var holding the API key. Never store the key in YAML. |
| `api_key_file`     | _(unset)_            | Optional path to a plain text file containing the raw key. |
| `api_key_env_file` | _(unset)_            | Optional path to a shell/dotenv-style file with `TCKDB_API_KEY=...`. Parsed, never executed. |
| `payload_dir`      | `tckdb_payloads`     | Relative paths resolve under the ARC project directory.   |
| `upload`          | `true`               | If `false`, write payload only and mark sidecar `skipped`.|
| `preflight`       | `true`               | If `true`, run one `GET /readyz` check before real uploads. Skipped when `upload: false`. |
| `strict`          | `false`              | If `true`, upload failure raises.                         |
| `timeout_seconds` | `30`                 | Per-request timeout.                                      |
| `project_label`   | `null`               | Optional run/project tag baked into the idempotency key.  |
| `artifacts`       | _(see below)_        | Optional sub-block controlling per-file attachments.      |
| `allow_partial_uploads` | `true`         | Computed-reaction mode only. See [Partial reaction sidecars](#partial-reaction-sidecars). |

### Partial reaction sidecars

In `computed_reaction` mode, ARC's reaction sweep checks each
reaction's `ts_label` against `output_doc['transition_states']` and
asks whether the TS converged. Two cases for a non-converged (or
missing) TS:

- **`allow_partial_uploads: true`** *(default)*: the sweep deepcopies
  the reaction record, sets `ts_label = null` and `kinetics = null`,
  and submits it as a **partial** bundle. The partial payload omits
  `transition_state` and `kinetics` entirely, the on-disk filenames
  gain a `.partial` infix (e.g.
  `CHO+CH4__CH2O+CH3.partial.payload.json`), and the sidecar is
  marked `is_partial: true`. **Partial bundles are never live-POSTed
  in phase-1**, regardless of `upload`. They land on disk with
  `status: "skipped"` and the live network path is bypassed.
- **`allow_partial_uploads: false`**: the sweep skips the reaction
  entirely. One log line per skip explains that the TS was missing or
  non-converged.

What the producer **does not** do for partial bundles, by design:

- Upload kinetics if the TS that produced them did not validate.
- Mark a partial reaction as complete on the wire.
- Include failed/candidate TS attempts as if they had been validated.

Reactions whose **identity** is malformed (e.g. a reactant species
absent from `output_doc.species`) are *not* treated as partial — they
flow through the normal failure path so the cause is visible in the
sweep summary rather than silently downgraded.

There is no `live_upload_partial` knob today. Live POST of partial
records is gated until a follow-up confirms the TCKDB server's
behavior on `transition_state: null` payloads.

#### Consumer contract for replay tools

Any tool that walks `tckdb_payloads/` and POSTs sidecars to TCKDB
(today, that's `tckdb-client`'s replay daemon — *not* anything
shipped inside ARC) **MUST** honor the following:

1. **Skip `metadata.is_partial == true` sidecars by default.**
   Partial bundles intentionally omit `transition_state` and
   `kinetics`; replaying one as a complete `computed-reaction`
   upload either silently creates an incomplete server-side record
   or errors at validation. The `.partial` filename infix is the
   second, redundant signal — either is sufficient on its own.
2. **Require an explicit user opt-in to ever replay partials.**
   A future flag (e.g. `--include-partial`) is reserved for once
   the TCKDB server formally accepts `transition_state: null`. No
   replay tool should infer "partial means incomplete-but-please-try"
   without that opt-in.
3. **Never strip the `.partial` infix or the `is_partial` flag** when
   copying / uploading / archiving sidecars. The two markers must
   stay in sync; ARC writes both.

ARC publishes a small in-tree helper for tools that import its
modules:

```python
from arc.tckdb.payload_writer import should_replay_sidecar

if not should_replay_sidecar(meta_dict_or_object):
    continue   # partial sidecar, do not POST
```

The helper accepts either a parsed JSON dict from a `*.meta.json`
file or a live `SidecarMetadata` instance. Its `include_partial`
kwarg is the documented opt-in point.

ARC's own re-sweep CLI (`python -m arc.tckdb.cli`) does **not** scan
`tckdb_payloads/` — it re-reads `output/output.yml` and goes back
through the normal sweep, so partial sidecars are produced fresh
from the source of truth on each invocation, never re-POSTed by
file discovery.

### Artifact sub-block

Artifacts are files (ESS logs, input decks, …) attached to an existing
TCKDB calculation row. The conformer payload is uploaded first; on
success the adapter can then push artifacts against the returned
`calculation_id`s.

```yaml
tckdb:
  ...
  artifacts:
    upload: true
    kinds: ["output_log", "input"]
    max_size_mb: 50
```

| Field          | Default            | Notes                                                                             |
| -------------- | ------------------ | --------------------------------------------------------------------------------- |
| `upload`       | `false`            | Opt-in switch. When `false`, no artifact network calls.                           |
| `kinds`        | `["output_log"]`   | Which `ArtifactKind`s to upload. Validated at config-parse time.                  |
| `max_size_mb`  | `50`               | Per-file cap. Larger files are skipped (sidecar `skipped`, reason recorded).      |

**Valid kinds** (mirror the server-side `ArtifactKind` enum):
`input`, `output_log`, `checkpoint`, `formatted_checkpoint`, `ancillary`.

**Currently implemented in ARC:**
- `output_log` — ESS log from `record["opt_log"] / ["freq_log"] / ["sp_log"]`.
- `input` — ESS input deck (`input.gjf` / `ZMAT` / `input.in`), sibling of `opt_log`.

Listing a valid-but-not-implemented kind (e.g. `checkpoint`) is allowed
so users can opt in early. Config-parse logs a warning, and the adapter
skips cleanly at upload time rather than 422-ing.

#### Artifact endpoint and idempotency

- Endpoint: `POST /calculations/{calculation_id}/artifacts` with the
  file bytes base64-encoded in the request body.
- Artifact uploads are batched by calculation through `tckdb-client`
  when multiple selected files target the same `calculation_id`.
- Idempotency key shape:
  ```
  arc:<project>:<species>:artifact:<calc_id>:<kind>:<sha256-prefix>
  ```
  Identical retry → server replays. Different bytes for the same
  `(calc_id, kind)` → new key, new upload event.

#### Artifact sidecar layout

Artifact sidecars live alongside conformer payloads:

```
<project>/tckdb_payloads/
  artifacts/
    <species>.calc<calculation_id>.<kind>.meta.json
```

#### Artifact skip / failure semantics

`status` ∈ `uploaded | failed | skipped`. Documented skip reasons:

- `artifacts.upload is False`
- `kind 'X' not in config.kinds`
- `kind 'X' is server-accepted but ARC has no upload path yet`
- `file missing: '...'`
- `file <name> is <bytes> bytes (><N> MB cap)`

Strict / non-strict failure behavior matches the conformer path:
non-strict logs a warning and records the error in the sidecar; strict
re-raises after updating the sidecar.

### API key

API keys are never stored in `input.yml`. The adapter resolves the key
at upload time, in this order:

1. `os.environ[api_key_env]` (default `TCKDB_API_KEY`).
2. `tckdb.api_key_file` — a plain text file whose entire contents
   (whitespace-stripped) are the API key.
3. `tckdb.api_key_env_file` — a shell/dotenv-style file containing a
   `TCKDB_API_KEY=...` assignment. The file is parsed, **never sourced
   or executed**; only literal assignment lines (with optional
   `export ` prefix and `'`/`"` quoting) are recognized, and `$VAR`
   references are not interpolated.

If the env var is set it always wins, so a one-off `TCKDB_API_KEY=...`
on the command line still overrides whatever is in `input.yml`.

If a configured file path is missing, unreadable, empty, or doesn't
define the variable, the adapter raises an `InputError` so a
misconfigured `input.yml` fails loudly rather than silently producing
zero uploads. If neither the env var nor any file is configured and
`upload: true`, the adapter records a failed sidecar (or raises in
strict mode) and never contacts the network.

```bash
# Option A — env var
export TCKDB_API_KEY="tck_replace_me"

# Option B — point ARC at the helper file written by tckdb dev login
#   tckdb:
#     api_key_file: /home/me/code/TCKDB_v2/backend/.tckdb_api_key
```

> The credential helper files (e.g. `.tckdb_api_key`,
> `.tckdb_auth.env`) hold a live secret. Do not commit them — keep
> them outside the repo or in `.gitignore`.

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
  "public_refs": {"calculation_refs": ["calc_..."]},
  "request_ids": [
    {"operation": "readyz", "request_id": "req_abc", "status_code": 200},
    {"operation": "upload", "request_id": "req_def", "status_code": 201}
  ],
  "preflight": {"ready": true, "status_code": 200, "request_id": "req_abc"},
  "idempotency_replayed": false,
  "last_error": null,
  "base_url": "http://localhost:8000/api/v1",
  "is_partial": false
}
```

When `upload: true` and `preflight: true`, ARC runs one readiness
preflight per adapter session before the first upload request. The
preflight calls `GET /readyz`; failures are recorded on the sidecar and
prevent the upload request. Payload-only/offline modes skip the
preflight because no network request will be sent.

Sidecars store `X-Request-ID` values when TCKDB returns them and the
client response exposes headers. Use these IDs to correlate ARC-side
upload attempts with TCKDB backend logs. The current `tckdb-client`
batch artifact helper returns artifact batch bodies without response
headers, so standalone artifact batch request IDs are recorded only
once that client metadata is exposed.

`status` ∈ `pending | uploaded | failed | skipped`.

`is_partial: true` marks a deliberately incomplete bundle (today, only
phase-1 partial computed-reaction sidecars). Filenames for partial
records carry a `.partial` infix in addition to the metadata flag.

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
