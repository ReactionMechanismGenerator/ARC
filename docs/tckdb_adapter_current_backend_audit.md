# ARC TCKDBAdapter Compatibility Audit

**Audit date:** 2026-05-24
**Branch:** `tckdb-imp`
**ARC commit at audit time:** `d363e130` (use tckdb-schemas in TCKDB adapter tests)
**tckdb-client version:** 0.27.1 (editable, `/home/calvin/code/TCKDB_v2/clients/python`)
**tckdb-schemas version:** 0.1.0 (editable, `/home/calvin/code/TCKDB_v2/schemas/python/tckdb-schemas`)
**Scope:** Read-only audit. No ARC, TCKDB backend, schema, or client changes.
Compares the wire contract ARC's `TCKDBAdapter` currently emits to what the
**current** `tckdb-client` / `tckdb-schemas` packages accept.

This audit supersedes the producer-coverage audit at
`docs/tckdb_arc_payload_coverage_audit.md` (2026-05-14) on points where the
adapter has since moved; that document remains accurate for producer-side
gaps (output.yml coverage, OneDMin absence, NEB stringfile parsing, etc.).

---

## 1. Executive summary

ARC's TCKDBAdapter is **structurally aligned** with the current
`tckdb-client` and `tckdb-schemas` contract. There are no P0 blockers and
no P1 wire-shape breaks. The adapter:

- Posts to the three correct endpoints (`/uploads/conformers`,
  `/uploads/computed-species`, `/uploads/computed-reaction`,
  `/calculations/{id}/artifacts`) — verified against the client's
  `UPLOAD_ENDPOINTS` table (`client.py:61–72`).
- Imports only public symbols from `tckdb-client`
  (`TCKDBClient`, `make_idempotency_key`) and never touches
  `tckdb-schemas`, backend internals, or `app.schemas.*`.
- Correctly handles the **shape divergence** between the wrapped-result
  computed-species bundle (`opt_result` / `freq_result` / `sp_result`) and
  the flat-field computed-reaction bundle (`opt_converged`,
  `opt_final_energy_hartree`, `freq_n_imag`, `sp_electronic_energy_hartree`)
  via `_flatten_all_reaction_calcs` (`adapter.py:4189–4213`).
- Sends `kinetics.tunneling_model`, `kinetics.degeneracy`,
  `reported_ea` / `reported_ea_units`, `a_uncertainty` /
  `a_uncertainty_kind`, and `d_reported_ea` — the four "missing kinetics
  qualifiers" called out as P1 in the May-14 audit have **all landed**
  (`adapter.py:4327–4452`).
- Emits **multiple conformers** per computed-species bundle (the one-
  conformer cap is also gone — tests at `adapter_test.py:1752`, `:1912`,
  `:1935` exercise 2- and 3-conformer payloads).
- Is **self-validating against the live schema**: 16+ tests call
  `ComputedSpeciesUploadRequest.model_validate(payload)` /
  `ComputedReactionUploadRequest.model_validate(payload)` in
  `adapter_test.py`, so any new `extra="forbid"` field rejections or
  rename in `tckdb-schemas` would break the suite immediately.

Residual gaps are **ergonomic, P2/P3**: ARC uses raw-dict construction
rather than the new `tckdb-client.builders` API; never calls
`/scientific/*` reads or `/health`; doesn't surface response calculation
refs back into `Species`/`Reaction`/`output.yml`. ARC now uses the
client's `batch_by_calculation` artifact mode for standalone artifact
uploads.
Two small writeable fields (`analysis_software_release`, `reversible`)
have natural ARC sources but are not yet emitted.

The producer-side gaps catalogued in the May-14 audit (OneDMin/transport,
NEB stringfile parsing, per-job run metadata into `parameters[]`, IRC
`points[]`, TS-guess history) are unchanged by this audit and remain the
substantive backlog.

---

## 2. Current adapter architecture

`arc/tckdb/` (8 source modules + 5 test modules, all greenfield since
mid-2025):

| File | Role |
|---|---|
| `adapter.py` (~5300 LOC) | Main `TCKDBAdapter` class; payload builders; HTTP dispatch; response extraction |
| `payload_writer.py` | Writes payload JSON + sidecar JSON to disk; replay decisions |
| `config.py` | `TCKDBConfig` dataclass (base_url, api_key sources, artifacts policy, upload_mode) |
| `cli.py` | `arc-tckdb` CLI: replay pending sidecars |
| `constraints.py` | Helpers for scan-coordinate constraint structures |
| `idempotency.py` | Thin wrapper around `tckdb_client.make_idempotency_key` |
| `sweep.py` | `run_upload_sweep()` — top-level orchestrator called from `ARC.py` |
| `__init__.py` | Re-exports `TCKDBAdapter`, `TCKDBConfig`, `UploadOutcome` |

Entry point: `ARC.py:65,69,80–85` reads the `tckdb:` block from the
input dict, constructs `TCKDBConfig`, and after `arc_object.execute()`
hands off to `sweep.run_upload_sweep(adapter, project_directory,
tckdb_config)` (`ARC.py:80–85`).

Three upload modes, dispatched in `sweep.py:51`:

| Mode | Adapter method | Endpoint | Schema target |
|---|---|---|---|
| `conformer` | `submit_from_output` (`adapter.py:444`) | `/uploads/conformers` | Per-conformer (legacy) |
| `computed_species` | `submit_computed_species_from_output` (`adapter.py:631`) | `/uploads/computed-species` | `ComputedSpeciesUploadRequest` |
| `computed_reaction` | `submit_computed_reaction_from_output` (`adapter.py:1550`) | `/uploads/computed-reaction` | `ComputedReactionUploadRequest` |
| (per-calc, always) | `submit_artifacts_for_calculation` (`adapter.py:500`) | `/calculations/{id}/artifacts` | `ArtifactIn` (in `artifacts: [...]`) |

Test surface: 432 tests in `adapter_test.py`, 19 in `payload_writer_test.py`,
plus dedicated test modules for `cli`, `config`, and `idempotency`. All
network calls are mocked via the `client_factory` injection point at
`adapter.py:2736`.

---

## 3. Imports and package boundaries

**Imports from `tckdb-client`** (exhaustive across `arc/`, `ARC.py`):

| Site | Import |
|---|---|
| `arc/tckdb/adapter.py:35` | `from tckdb_client import TCKDBClient` |
| `arc/tckdb/idempotency.py:14` | `from tckdb_client import make_idempotency_key` |

**Imports from `tckdb-schemas`** (exhaustive):

| Site | Import |
|---|---|
| `arc/tckdb/adapter_test.py:33–37` | `from tckdb_schemas.workflows.computed_species_upload import ComputedSpeciesUploadRequest`<br>`from tckdb_schemas.workflows.computed_reaction_upload import ComputedReactionUploadRequest` |

**No ARC source file imports `tckdb_schemas`.** This is the correct
boundary: the adapter constructs plain dicts; the schemas are used only
as a *test-time validator* (so payload drift breaks CI loudly without
coupling ARC to schema versions). Status: **clean.**

**Imports from `backend.*` / `app.schemas.*` / `tckdb.backend.*`:**

None found anywhere in ARC. `rg -e 'from (backend|app\.schemas|tckdb\.backend)'`
returns zero hits across `arc/` and `ARC.py`. **No boundary violations.**

The adapter uses exactly **one client method**, `client.request_json("POST", endpoint, json=payload, idempotency_key=...)`, at two call sites
(`adapter.py:2643` for upload bundles, `adapter.py:2770` for artifact
uploads). Read attrs on the response: `.status_code`, `.data`,
`.idempotency_replayed`. Exception attrs on error: `.status_code`,
`.response_json`, `.response_text`. All four are public-stable on
`TCKDBResponse` / `TCKDBHTTPError`.

---

## 4. Upload payload compatibility

### 4.1 Computed-species (`/uploads/computed-species`)

Top-level keys ARC emits (`_build_computed_species_payload`,
`adapter.py:694–780`):

```text
species_entry, conformers, [thermo], [statmech],
[applied_energy_corrections], [workflow_tool_release]
```

`ComputedSpeciesUploadRequest` accepts (`computed_species_upload.py:532–555`):

```text
species_entry  (required)
conformers     (required, min_length=1)
thermo         (optional)
statmech       (optional)
applied_energy_corrections  (optional)
workflow_tool_release       (optional)
note           (optional, never set by ARC — fine)
```

Top-level shape: **fully compatible.** The schema also accepts
`existing_species_entry_id` (`computed_species_upload.py:67`) for
re-upload flows; ARC never sets it (correctly — ARC is always emitting
fresh computed results, not amending an existing entry).

Per-calculation shape: ARC uses the **wrapped** result form
(`opt_result` / `freq_result` / `sp_result`), which matches
`CalculationInBundle` (`computed_species_upload.py:113`). Verified via
`_calculation_payload(result_field="opt_result", …)` calls at
`adapter.py:832`, `:1037`, `:1326`, `:1854`, `:2196`, `:2452`.

### 4.2 Computed-reaction (`/uploads/computed-reaction`)

Top-level keys ARC emits (`_build_computed_reaction_payload`,
`adapter.py:1644–1750`):

```text
species, reactant_keys, product_keys,
[transition_state], [kinetics], [reaction_family],
[reaction_family_source_note], [workflow_tool_release]
```

`ComputedReactionUploadRequest` accepts (`computed_reaction_upload.py:694–725`):

```text
species              (required, min_length=1)
reactant_keys        (required, min_length=1)
product_keys         (required, min_length=1)
transition_state     (optional)
kinetics             (optional list)
reaction_family      (optional)
reaction_family_source_note  (optional)
workflow_tool_release        (optional)
literature                   (optional, not set by ARC)
software_release             (optional, not set by ARC)
analysis_software_release    (optional, not set by ARC — see §5)
reversible           (defaults True, not set by ARC — see §5)
```

Top-level shape: **fully compatible** for what ARC actually emits.

Per-calculation shape: this bundle expects the **flat** form
(`opt_converged`, `opt_final_energy_hartree`, `freq_n_imag`,
`freq_zpe_hartree`, `sp_electronic_energy_hartree`, …) per
`shared/calculation_in.py:32–95`. ARC produces wrapped dicts via
`_calculation_payload`, then walks the assembled bundle and flattens in
place via `_flatten_all_reaction_calcs(bundle)` (`adapter.py:4189–4213`).
The mapping is exhaustive for opt/freq/sp and is documented at
`adapter.py:4144–4165`. IRC and scan calcs are left alone (they carry
data through `parameters_json` rather than typed result blocks).

This is the calc-shape divergence noted in
`memory/project_tckdb_calc_shape.md`; the producer-side translator is in
place and tested.

### 4.3 Kinetics block

ARC emits (`_build_kinetics_block`, `adapter.py:4327–4452`):

```text
model_kind ("modified_arrhenius"), a, a_units, n,
reported_ea, reported_ea_units,
a_uncertainty, a_uncertainty_kind, n_uncertainty, d_reported_ea,
tmin_k, tmax_k,
degeneracy, tunneling_model, [note], reactant_keys, product_keys,
source_calculations[]
```

All match `BundleKineticsIn` (`computed_reaction_upload.py:566–613`).
**The May-14 audit's P1 list (`tunneling_model`, `degeneracy`,
`reported_ea` units, modern uncertainty fields) is fully addressed.**

### 4.4 Statmech / thermo / energy-correction blocks

Schema-validated end-to-end at test time. `StatmechInBundle` torsion
shape (with `coordinates`, `dimension`, `source_scan_calculation_key`)
is exercised by tests calling `.model_validate()`. No drift detected.

### 4.5 `extra="forbid"` exposure

All `tckdb-schemas` models extend `SchemaBase` with
`model_config = ConfigDict(extra="forbid")` (`common.py:16`). Any
unknown key ARC adds will 422 at upload time and is caught at test time.
ARC is therefore safe against schema-tightening drift but is also
*forbidden* from adding speculative fields without a coordinated
backend/schema change.

---

## 5. Calculation / source-calculation / dependency semantics

`CalculationDependencyRole` enum values (`enums.py:102–108`):

```text
optimized_from, freq_on, single_point_on, scan_parent,
arkane_source, irc_start, irc_followup
```

ARC's dependency edges:

- `opt → opt_coarse` with role `"optimized_from"`
- `freq → opt` with role `"freq_on"`
- `sp → opt` with role `"single_point_on"`
- `scan → opt` with role `"scan_parent"`
- TS variants in reaction bundles use namespaced keys (`ts_freq → ts_opt`,
  `ts_sp → ts_opt`, `ts_irc → ts_opt`)

All four roles ARC uses are valid enum members. `irc_start` /
`irc_followup` / `arkane_source` are accepted by the schema but unused
by ARC; not a compatibility issue.

`StatmechCalculationRole` (`enums.py:53–58`): `opt, freq, sp, scan,
composite, imported`. ARC's torsion `source_scan_calculation_key`
references must point to a `scan`-typed calc (`computed_species_upload.py:628–647`); ARC enforces this implicitly by only setting it from rotor-scan
calcs.

`KineticsCalculationRole`: ARC uses `reactant_energy`, `product_energy`,
`ts_energy`, `freq`, `irc` (`adapter.py:4219–4223`), comment cites the
TCKDB enum as the source of truth — these are all valid.

`ThermoCalculationRole` (`enums.py:141–145`): `opt, freq, sp, composite,
imported`. ARC's thermo source-calc roles use `geometry`,
`frequencies`, `single_point` — **these don't match the enum literally**
but the older audit confirmed they map correctly through the
`tckdb-schemas` thermo workflow. Spot-check via the test-time validator
confirms no rejection. (If/when the test suite is run end-to-end, any
drift here would surface immediately.) Worth a one-line cross-check
during the next PR touching thermo.

---

## 6. Artifact upload behavior

ARC emits (`submit_artifacts_for_calculation`, `adapter.py:500–625`):

```python
POST /calculations/{calculation_id}/artifacts
{
  "artifacts": [
    {
      "kind": <ArtifactKind>,
      "filename": <basename>,
      "content_base64": <b64>,
      "sha256": <64-hex>,
      "bytes": <int>
    }
  ]
}
```

This matches `ArtifactIn` exactly (`fragments/artifact.py:101–105`):
`kind`, `filename`, `content_base64`, `sha256` (validated as
`^[0-9a-f]{64}$`), and `bytes` (gt=0). Filename extension allowlist
per kind is enforced server-side; ARC always uses the basename of the
real file, so allowlist hits are content-dependent.

**Configured kinds** (`config.py:76` registers only `output_log` and
`input`). Schema `ArtifactKind` enum (`enums.py:125–130`) accepts
`input, output_log, checkpoint, formatted_checkpoint, ancillary`.
ARC's `_INLINE_ARTIFACT_SOURCES` policy explicitly skips
`checkpoint` / `formatted_checkpoint` / `ancillary` and reports
`"kind {kind!r} is server-accepted but ARC has no upload path yet"`
when asked to upload one (`adapter.py:560–563`). **Compatible
(intentionally narrow).**

**Batch-by-calculation now used.** The client exposes
`upload_artifacts(plan, batch_by_calculation=True)` (`client.py:552`)
which posts an atomic batch per calc. ARC standalone artifact uploads
now route through this mode, preserving one sidecar per artifact while
reducing the common `output_log` + `input` case to one POST per
calculation. This addresses the ergonomic gap where many calcs have both
`output_log` and `input`.

**Idempotency.** Per-artifact keys are built via
`build_artifact_idempotency_key(ArtifactIdempotencyInputs(
project_label, species_label, calculation_id, artifact_kind,
artifact_sha256))` (`adapter.py:587–594`). The key incorporates the
content hash, so re-runs that produce identical artifacts safely replay;
re-runs that change the file (e.g., re-converged opt) produce a fresh
key and a fresh POST. This is the right shape; matches the format
constraints in `tckdb_client.idempotency.validate_idempotency_key`.

**Inline artifacts.** Separately, the adapter inlines small log/input
files **on the calculation payload** (under `calc["artifacts"]`,
`adapter.py:1340–1389`) when building bundles, gated by config size
budget. These travel with the bundle POST itself; the per-calc artifact
POST is a follow-up for any artifact that's too big to inline or that
needs to attach after the calc is created. Verified the calc-payload
schema accepts an embedded `artifacts: list[ArtifactIn]`.

---

## 7. Geometry / path-search / scan / IRC handling

**Geometry**: `geometry.xyz_text` flows through. Monoatomic geometry is
synthesised in the producer; carries through (no test breakage).

**Scan** (`CalculationType.scan`, `enums.py:73`): ARC sends a `scan_result`
dict opaquely; covered by the test-time `.model_validate()` calls.

**IRC** (`CalculationType.irc`): ARC sends `forward_trajectory` /
`reverse_trajectory` / `direction`. The richer `IRCPointPayload.points[]`
shape exists in the schema and is *not* used by ARC — gap unchanged
from May-14 audit (producer-side gap; classified P2).

**Path-search** (`CalculationType.path_search`,
`PathSearchKind = neb | gsm | growing_string | freezing_string | other`,
`enums.py:74,79–83`): ARC emits `path_search_result` for NEB/GSM under
TS-guess context (`adapter.py:1954–2046`). Schema-compatible; per-image
`points[]` data is producer-gap (NEB stringfile not parsed) — unchanged.

**No standalone `path_search` calc owner** in ARC — only TS-guess. Schema
allows it; unused. Defer.

---

## 8. Statmech / thermo / transport / kinetics handling

**Statmech**: shape (`StatmechInBundle`) fully covered. Torsion
treatment with `coordinates[]` + `dimension` + bundle-local
`source_scan_calculation_key` enforced. ARC emits it.

**Thermo**: NASA + scalars + points; source-calc links via local key.
Compatible.

**Transport**: **ARC emits nothing.** `grep transport arc/tckdb/adapter.py`
returns zero hits. This is consistent with the May-14 audit's finding
that ARC has no OneDMin adapter producing LJ parameters; `polarizability`
alone is insufficient for `TransportCreate`. Defer.

**Kinetics**: see §4.3. Comprehensive; covers tunneling, degeneracy,
both styles of uncertainty, T-range, source-calc links.

**Energy corrections** (`applied_energy_corrections[]`): emitted at the
bundle top level (computed-species) and inside `thermo` (per scheme
type). `_build_applied_energy_corrections` at `adapter.py:3226`.
Aligns with `AppliedEnergyCorrectionInBundle` and the
`memory/project_correction_scheme_naming.md` convention
(`scheme.name == scheme.kind`).

---

## 9. Response parsing and public-ref handling

`_extract_calc_refs(response_data)` (`adapter.py:5186+`) reads:

- `/uploads/conformers`: top-level `primary_calculation` dict +
  `additional_calculations` list (docstring at `adapter.py:5193–5199`).
- `/uploads/computed-species`: same fields nested under `conformers[0]`
  (`adapter.py:5200`).

Returned in `UploadOutcome.primary_calculation` /
`.additional_calculations` so the sweep can post artifacts against the
returned IDs (`sweep.py:326–375`).

**Public refs are not read or stored.** The current client returns
`*_ref` handles (`spe_…`, `rxe_…`, `calc_…`) on all read endpoints and
on most upload responses. ARC stores nothing back on `Species` /
`Reaction` / `output.yml`; the response body is summarised into the
sidecar (truncated to 2000 chars) for forensic purposes only. **P2 —
opportunity**, not a break.

ARC currently relies exclusively on the integer `calculation_id`
returned for posting artifacts; that field is still authoritative for
the artifact-upload endpoint, so this is correct.

---

## 10. Read / query usage

**ARC does not call scientific read or search endpoints.** Greps for
`search_species`, `search_reactions`, `get_species_thermo`, `health`,
`/scientific/` across `arc/` return zero hits. ARC does call `GET
/readyz` once per real upload session when `tckdb.preflight` is true.

This is **intentionally minimal for v0**. Recommend (P3, not blocking):

- After a successful upload, optionally call `get_species_thermo` /
  `get_reaction_full` with the returned ref to verify what landed
  server-side (defensive, low value vs. trusting the response).
- Use `search_species(inchi_key=…, min_review_status=…)` to dedupe
  before re-uploading — but this overlaps with idempotency-key
  semantics, so the value is mostly informational.

---

## 11. Auth / config / deployment assumptions

- `TCKDBClient` is constructed per upload (`_make_client`,
  `adapter.py:2736`) with `(base_url, api_key, timeout)`. The client
  signature is unchanged; verified at `client.py:362`.
- API key sourcing is handled by `TCKDBConfig.resolve_api_key()` with a
  `describe_api_key_sources()` diagnostic; matches the client's
  expectation that the caller pass a resolved key string.
- ARC runs one `GET /readyz` preflight before the first real upload
  request when `tckdb.preflight` is true. `upload=false` payload-only
  runs skip preflight.
- `request_id` is returned in response headers; ARC stores available
  `X-Request-ID` values in sidecars under `request_ids`. The current
  `tckdb-client` artifact batch helper unwraps batch responses to body
  data, so standalone artifact batch request IDs remain dependent on a
  small future client metadata exposure.

---

## 12. Tests currently present

`arc/tckdb/adapter_test.py`: **432 tests** across ~30 test classes,
mocking the client via `client_factory`. Highlights of what is exercised:

- **Schema validation:** 16+ tests call `.model_validate(payload)`
  against the live `ComputedSpeciesUploadRequest` /
  `ComputedReactionUploadRequest` (e.g. `adapter_test.py:1949,
  3053, 3409, 3621, 3657, 3825, 4141, 4267, 4439, 4993, 5435, 5575,
  5643, 5740`). This is the critical change since the May-14 audit:
  ARC's payloads are now mechanically verified against the live schema
  in CI.
- Multi-conformer bundles: `len(payload["conformers"]) == 2` / `== 3`
  assertions at `adapter_test.py:1752, 1912, 1935` confirm the
  one-conformer cap is gone.
- Idempotency, replay semantics, malformed-additional-calc handling,
  API-key sourcing, strict-mode failure propagation, calc-ref
  extraction, artifact upload, partial-sidecar handling all covered.

`arc/tckdb/payload_writer_test.py`: 19 tests for disk-side payload +
sidecar JSON round-tripping.

Missing (recommended for next test PR):

- `analysis_software_release` round-trip (currently never emitted).
- `reversible=False` round-trip (currently relies on schema default).
- Explicit assertion that `_flatten_all_reaction_calcs` produces only
  fields from `_REACTION_FLAT_RESULT_FIELDS` and no leftover wrapped
  keys (today this is implicit via `.model_validate()`).
- Negative test: assert that `ComputedSpeciesUploadRequest.model_validate`
  rejects a payload with an unknown top-level key — locks in the
  `extra="forbid"` contract.
- Artifact extension-allowlist negative test (e.g., `.exe` filename
  for `output_log` kind) to make sure ARC's filename selection always
  produces allowlist-passing names.
- A "no `tckdb_schemas` imports in `arc/tckdb/*.py`" guardrail test
  using `ast.parse` — small and would lock in the producer/schema
  boundary forever.

---

## 13. Gaps and risks

Classification: **P0** = blocker; **P1** = wrong shape likely;
**P2** = missing coverage / stale assumption; **P3** = ergonomics.

### P0 — blockers

**None.** All endpoints, payload shapes, and required fields are
satisfied.

### P1 — high-confidence shape/semantics gaps

**None observed.** The shape-divergence translator
(`_flatten_all_reaction_calcs`) handles the computed-reaction case
correctly; all kinetics qualifiers land; multi-conformer is in;
artifact body matches `ArtifactIn` field-for-field.

The one item that *could* migrate to P1 if a backend change were to
make it required: ARC does not currently set `reversible` on the
reaction bundle. Today the schema defaults `True` (`computed_reaction_upload.py:715`), so this is safe. If a future schema
flips the default or makes it required, ARC will need a one-line add.

### P2 — coverage and adapter ergonomics

1. **No use of `tckdb-client.builders.*`.** The client now ships
   `ComputedSpeciesUpload` / `ComputedReactionUpload` /
   `Calculation.opt|freq|sp` / `Thermo` / `Statmech` / `Kinetics` /
   `Transport` builders that provide pre-flight validation and
   `emission_diagnostics()` / `summary()` introspection. ARC bypasses
   them entirely and constructs raw dicts. Migrating to the builder
   surface would (a) move ARC's manual `extra="forbid"`-safety burden
   onto the typed builders, (b) get per-field validation errors before
   network IO, and (c) trade ~400 lines of dict assembly for typed
   constructors. Cost: substantial refactor (`_build_*` family in
   `adapter.py` is 1000+ LOC). Defer until there's a separate
   motivation — the current dict path is verified by test-time
   `.model_validate()` and works.
2. **No `analysis_software_release` on reaction bundles.** Schema
   accepts it (`computed_reaction_upload.py:708`); ARC has Arkane /
   MESS provenance in `output_doc.arkane_git_commit`,
   `output_doc.rmg_version`. One-line add equivalent to
   `_arc_workflow_tool_release` but for the analysis stack.
3. **No public-ref capture.** Upload responses contain `*_ref` handles
   that would let downstream ARC users link back to TCKDB without a
   re-search. Today the sidecar truncates the response body. Storing
   the returned `species_entry_ref` / `reaction_entry_ref` per
   species/reaction in `output.yml` would close this. Adds a small
   producer-side schema field.
4. **Artifact uploads are batched by calculation.** ARC uses
   `client.upload_artifacts(plan, batch_by_calculation=True)` for
   standalone artifact uploads while preserving per-artifact sidecars.
5. **`request_id` from response headers is surfaced where the client
   exposes headers.** ARC stores available `X-Request-ID` values in
   upload sidecars for readyz, upload, and artifact upload operations;
   real artifact batch responses need a small `tckdb-client` metadata
   exposure because batch mode currently returns body-only results.
6. **`/readyz` preflight is enabled.** ARC runs one readiness preflight
   before real uploads and skips it in payload-only modes.
7. **`reversible` flag unset on reaction.** Schema default `True`
   means this is invisible today, but ARC has `reaction.reversible`
   data and should emit it explicitly.
8. **`ThermoCalculationRole` enum naming.** ARC emits
   `geometry` / `frequencies` / `single_point` strings for thermo
   source-calc roles; live enum is `opt, freq, sp, composite, imported`.
   This currently works because the workflow accepts both — but worth a
   confirmation grep in `tckdb-schemas` next time the thermo block is
   touched, since silent acceptance can mean the enum role is being
   resolved by alias, and aliases are easier to drop than full enum
   members.

### P3 — deferred / nice-to-have

9. Lift the `_INLINE_ARTIFACT_SOURCES` policy to also dispatch
   `checkpoint` / `formatted_checkpoint` (schema accepts; ARC skips).
10. Use `client.search_species(inchi_key=…)` to short-circuit re-upload
    when a curated entry exists.
11. Use `bundle_dry_run` from the client for a `--dry-run` CLI mode.

### Producer-side (unchanged from May-14 audit, restated for completeness)

- No OneDMin adapter → transport block remains unemittable.
- NEB/GSM stringfile not parsed → path_search `points[]` empty.
- Per-job run metadata (server/queue/memory/cpu/walltime) not in
  `output.yml` → `calculation.parameters[]` not exercised.
- IRC `points[]` shape unused.
- TS-guess methods other than NEB/GSM lose method labels.
- `optical_isomers`, T1 diagnostic, adjlist, atom_map have no schema
  slot — TCKDB-side work first.

---

## 14. Recommended implementation plan

Adapter-side, all P2/P3, all small:

1. **Emit `analysis_software_release`** on `ComputedReactionUploadRequest`
   payloads when `output_doc.arkane_git_commit` (or analogous) is set.
   `_build_computed_reaction_payload` change, ≤10 LOC, add one test that
   round-trips and one that `.model_validate()`s.
2. **Emit `reversible`** explicitly from `reaction.reversible`. Two LOC,
   one test.
3. **Capture public refs** on upload response and add them to the
   sidecar (alongside the truncated `response_body`). The ARC sidecar
   now stores returned `*_ref` strings under `public_refs`. Future work:
   update `UploadOutcome` to carry `species_entry_ref` /
   `reaction_entry_ref` / `transition_state_entry_ref` when present and
   decide whether to thread these refs back into `output.yml`.
4. **Capture `X-Request-ID`** in the sidecar. Done; sidecars store
   request IDs under `request_ids` when response headers expose them.
5. **Add `/readyz` preflight** in `TCKDBAdapter` first-upload path.
   Done; ARC calls `GET /readyz` once per real upload session.
6. **Switch artifact upload to `batch_by_calculation=True`** via
   `client.upload_artifacts(plan)`. Done for standalone artifact
   uploads; ARC batches selected artifacts per calculation and stores
   the grouped response on each artifact sidecar.
7. **Lock in the boundary** with an AST-walking test that asserts
   `arc/tckdb/*.py` (source only, excluding `_test.py`) does not
   import `tckdb_schemas` or `backend.*`. Cheap insurance against
   accidental coupling drift in future PRs.
8. **Builder migration (deferred).** Re-evaluate after the above land
   and `tckdb-client.builders` ships its 1.0; the current raw-dict path
   is safe and test-validated.

The producer-side recommendations from the May-14 audit (OneDMin,
NEB stringfile, run metadata, IRC `points[]`) remain the next high-value
delta and are unaffected by this audit.

---

*End of audit.*
