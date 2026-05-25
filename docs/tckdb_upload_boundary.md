# ARC → TCKDB Semantic Upload Boundary

## Purpose

This document defines what ARC should and should not upload to TCKDB. It exists
so that the line between *scientific data* and *workflow narrative* stays sharp
as ARC's TCKDB adapter evolves. The rule is meant to be applied at the producer
(ARC) side. The adapter under `arc/tckdb/adapter.py` and its tests in
`arc/tckdb/adapter_test.py` are the enforcement surface; the live TCKDB
schemas under `TCKDB_v2/backend/app/schemas/` are the contract.

Companion docs:
- `docs/tckdb-integration.md` — how ARC talks to TCKDB at the wire level.
- `docs/tckdb_arc_payload_coverage_audit.md` — running audit of which TCKDB
  fields ARC currently populates.

## Core rule

> **TCKDB stores final scientific records and reproducibility-critical
> evidence. ARC stores ARC workflow history.**

If a value would only make sense to someone debugging an ARC run, it is
workflow history. If a value would help any thermochemistry or kinetics
consumer reinterpret, recompute, or cite the result, it is scientific data.

When in doubt, *omit* from the TCKDB payload and keep it in ARC artifacts.
TCKDB is easier to extend later (typed field, schema review) than to clean
up after free-text noise has accumulated.

## Data ARC should upload

The following are in-scope for the ARC → TCKDB upload contract:

- `workflow_tool_release` identifying ARC (version + release metadata).
- `software_release` for each electronic-structure / statmech tool actually
  run (Gaussian, Orca, Arkane, …).
- `level_of_theory` for every calculation that carries one.
- Species / reaction identity (SMILES, InChI, adjacency list, formula,
  charge, multiplicity).
- Input geometries (`initial_xyz`, conformer geometries actually submitted).
- Output geometries (`final_xyz`, optimized conformer geometries).
- Calculation results: energies, gradients, Hessians, frequencies, ZPE,
  rotational constants — anything that is the *result* of a job.
- Thermo records: `h298_kj_mol`, `s298_j_mol_k`, `tmin_k`, `tmax_k`, NASA
  polynomial coefficients, and tabulated `thermo_points`
  (`temperature_k` + `cp_j_mol_k` + optional `h_kj_mol`, `s_j_mol_k`,
  `g_kj_mol`).
- Statmech records: `e0_kj_mol`, `spin_multiplicity`, `optical_isomers`,
  `is_linear`, rotor kind, frequencies, frequency scale factor (with
  source citation when present).
- Kinetics records: Arrhenius / modified-Arrhenius parameters, fit
  temperature range.
- `kinetics.degeneracy` when explicitly known. Degeneracy modifies the
  meaning of the rate expression itself; it is not workflow history.
- Tunneling model (`Eckart`, `Wigner`, etc.) when used.
- `source_calculations` links from thermo/kinetics records back to the
  opt / freq / sp calculations that produced them, with the correct role.
- `calculation_dependency` links (e.g. freq depends on opt geometry) so the
  scientific provenance chain is reconstructable.
- Structured IRC / scan / path-search points *when they are scientific
  result data* (per-point energies, geometries, mode labels). The IRC
  payload supports a TS marker — see Borderline cases.
- Artifacts / logs as evidence, using the existing artifact slot when the
  log is the canonical record of a result (not a debug trail).

## Data ARC should not upload

The following are out-of-scope. They belong in ARC's project directory,
restart YAML, and log files, not in TCKDB structured payloads:

- ARC troubleshooting history (which methods were tried, in what order,
  what failed and why ARC moved on).
- Operational metadata: `server`, `queue`, `job_id`, `run_time`,
  `walltime`, `scratch_path`, `local_path`, attempted queues.
- Failed conformer attempts and conformer-screening logs.
- ARC's internal conformer ranking and `relative_e0_kj_mol` from
  screening — only the *chosen* conformers and their *absolute* results
  should appear.
- `successful_methods` / `unsuccessful_methods` lists.
- `chosen_ts_list`, `selected_ts_guess` markers, and any TS-guess history
  (`tsg_spawned`, attempted families, etc.).
- `ts_report` text (ARC's own narrative summary of the TS workflow).
- Reaction-template internals: `reaction_template`, `template_labels`,
  family-matching scratch.
- `atom_map` and mapping history. Atom-map tracking is an ARC mechanism,
  not a chemical fact.
- `long_thermo_description`. This accumulator is ARC/RMG export bookkeeping
  and either duplicates typed TCKDB fields (symmetry, geometry, BAC scheme)
  or carries workflow narrative (TS report, "no rotors considered"). See
  Borderline cases.
- RMG / Arkane library-export `longDesc` text (the `entry(... longDesc=...)`
  free-text block emitted to `.py` library files).

## Borderline cases and decisions

Decisions that were close enough to deserve a written rationale:

- **`kinetics.degeneracy`** — *Keep.* Degeneracy is part of kinetics-
  expression interpretation: the same A, n, Ea mean different rates at
  different degeneracies. It is a scientific fact about the reaction, not
  workflow history.

- **`screened_conformer` `tckdb_origin` marker** — *Keep.* TCKDB requires
  a `primary_calculation` for each conformer; ARC's screening produces a
  minimal anchor calculation that is *not* an independent ESS job. The
  marker exists so consumers do not mistake that anchor for a real job.
  This is the narrow exception, not a precedent for stashing workflow
  state under `tckdb_origin`.

- **`final_settings.optimization_stage`** — *Keep.* Records the final
  effective calculation stage the result corresponds to (e.g. fine-grid
  re-opt) where multiple opt stages exist in one chain. Do *not* expand
  this into troubleshooting history (earlier attempts, why a stage was
  rerun, etc.).

- **IRC TS marker** — *Keep.* `IRCPointPayload` / `IRCResultPayload`
  explicitly support TS markers via `is_ts` and `ts_point_index`. The TS
  geometry along an IRC is scientifically meaningful, not workflow narrative.

- **`unmapped_smiles`** — *Keep only* as an explicit producer-supplied
  auxiliary chemical identity string. Do *not* derive it inside the
  adapter by stripping atom maps from a mapped SMILES, and do *not* emit
  `atom_map` itself. The principle: TCKDB can hold the human-readable
  identity, but not the ARC-internal mapping that produced it.

- **`long_thermo_description`** — *Keep out.* See the audit in this branch.
  Persisted content is `Bond corrections: {…}` (duplicates BAC scheme
  provenance already on the calculation). Post-export content adds
  symmetry / optical isomers / geometry (already typed in statmech +
  geometry) and, for TS, full `ts_report` (workflow narrative).

## Current accepted ARC-specific markers

The following `parameters_json` / origin conventions are currently allowed
in TCKDB payloads, and *only* these:

- `tckdb_origin.reused_result` — flags a calculation row that re-points at
  a previously uploaded result rather than re-running. Preserves
  idempotency without re-uploading the underlying ESS job.
- `tckdb_origin.screened_conformer` — flags the minimal anchor calculation
  attached to a screened-only conformer (see Borderline cases).
- `final_settings.optimization_stage` — final effective optimization stage
  for the calculation chain (see Borderline cases).

Adding a new `origin_kind` (or any new ARC-specific `parameters_json` key)
requires a schema/boundary review first. The default answer is *no*; the
burden is on the proposer to show that the value is scientifically
meaningful and that no typed TCKDB slot can hold it.

## Guardrail tests

The adapter test suite (`arc/tckdb/adapter_test.py`) carries full-payload
walkers that assert *absence* of forbidden keys, in addition to the
positive-shape tests. Specifically:

- Workflow-narrative keys never appear anywhere in an emitted payload:
  `successful_methods`, `unsuccessful_methods`, `chosen_ts_list`,
  `ts_report`, `selected_ts_guess`.
- Operational fields never appear: `server`, `queue`, `job_id`,
  `run_time`, `walltime`, `scratch_path`, `local_path`.
- Atom-map / mapping-history / reaction-template keys never appear:
  `atom_map`, `mapping_history`, `reaction_template`, `template_labels`.
- `relative_e0_kj_mol` never appears (only absolute energies escape ARC's
  conformer screening into TCKDB).

When a future change adds a new emission path, the corresponding walker
assertion should be extended so the boundary stays enforced by tests, not
by reviewer memory.

## Future schema-change rule

If a value is scientifically meaningful but has no typed TCKDB slot:

1. **Do not hide it** in `note`, `parameters_json`, or any other free-text
   slot by default. Doing so creates an unreviewed schema-by-precedent.
2. **First decide whether TCKDB should own the concept.** Ask: is this a
   fact about the result that any consumer would want to reinterpret /
   recompute / cite? Or is it ARC-specific bookkeeping?
3. **If yes** — TCKDB should own it — propose a schema/API field. Update
   `TCKDB_v2/backend/app/schemas/`, regenerate the client, then teach the
   adapter to populate the new field.
4. **If no** — keep it in ARC artifacts / logs / restart YAML. The
   project directory is the right home for ARC-private state.

This rule applies symmetrically: removing a field from TCKDB also begins
with a schema/boundary review, not a producer-side patch.
