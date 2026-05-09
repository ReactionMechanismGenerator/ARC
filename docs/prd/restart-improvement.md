# PRD: Restart improvement via fixture-driven mockter and structural restart-dict invariants

## Problem Statement

ARC's restart feature is fragile and under-tested. Multiple production bugs (#632, #624, #622, #358) crash with `KeyError` during `save_restart_dict()` or after `arcrestart`, costing user runs and engineering time. The root cause is that ARC's state machinery — the contract between `running_jobs`, `job_dict`, on-disk artifacts, and the persisted `restart.yml` — is implicit and untested end-to-end. The existing `mockter` adapter, intended to enable end-to-end testing, returns dummy YAML that Arkane cannot parse, so it has never been used to drive a real pipeline test. Without an end-to-end test instrument, every restart fix is a point patch validated by manual testing on real cluster jobs, which is slow, expensive, and incomplete.

## Solution

Overhaul `mockter` into a deterministic, fixture-driven test instrument that emits skeletal but Arkane-parseable Gaussian-format log files, route it through ARC's existing `Level` mechanism via `mockter1`, `mockter2`, … method names, and use it to build three tiers of regression tests covering: (Tier-1) single-run restart-dict invariants, (Tier-2) interrupt-and-resume scenarios for cross-run bugs, (Tier-3) end-to-end thermo and kinetics equivalence between mockter-driven runs and runs over real DFT log files. Each test scenario is a per-scenario fixture YAML committed alongside real Gaussian log files, generated once from a real DFT run by an extraction script. The invariant violations behind #632/#624 are encoded as a pure function and run as a precondition of `save_restart_dict()`, turning the bug class into a structural impossibility rather than a series of point fixes.

## User Stories

1. As an ARC developer, I want a fast, deterministic, FF-free mock ESS adapter, so that I can run the full ARC pipeline end-to-end in unit-test time without using cluster resources.
2. As an ARC developer, I want mockter to emit skeletal Gaussian-format `.log` files that Arkane parses without modification, so that the ESS-output pipeline (`parser.py`, `output_dict`, Arkane `Log`) is exercised by tests instead of bypassed.
3. As an ARC developer, I want to select a fixture file via the level-of-theory string (`mockter2/def2tzvp`), so that the choice survives subprocess boundaries and integrates with ARC's existing job-routing without new `input.yml` fields.
4. As an ARC developer, I want a fixture YAML schema that captures conformer XYZs, opt/fine-opt geometries, freqs, Hessians, scans, sp energies, TS guesses, IRC trajectories, with a versioned schema and provenance block, so that fixtures are auditable and regeneratable.
5. As an ARC developer, I want one fixture file per scenario (not one global file), so that scenarios are independently editable, merge conflicts are localized, and a broken fixture only breaks its own test.
6. As an ARC developer, I want a fixture extraction script that reads a real ARC project directory and uses ARC's parser to compose the fixture YAML, so that I can iterate "run real DFT once → extract → commit" and never hand-edit numerical values.
7. As an ARC developer, I want mockter to fall back to today's hand-rolled values when a fixture key is missing and to log a warning + write a `mockter_fallback.flag` marker, so that tests can assert no fallback occurred and silent fallback isn't a hidden test gap.
8. As an ARC developer, I want fixture entries to optionally carry a `raise:` clause (`crash`, `oom`, `timeout`, `scf_nonconvergence`, `sigterm`), so that I can deterministically trigger ESS failure paths from the fixture without polluting `scheduler.py` with test hooks.
9. As an ARC developer, I want a `note:` field on fixture conformer entries, so that "this xyz is deliberately isobutane to test n-butane isomorphism handling" is documented inline.
10. As an ARC developer, I want mockter to validate at fixture-load time that any conformer marked `isomorphic: true` actually matches the species's expected graph, so that fixture data integrity is enforced and "wrong XYZ but flagged isomorphic" is impossible.
11. As an ARC developer, I want mockter to emit Gaussian logs with only scientific content (geometry blocks, `SCF Done`, freqs, ZPE, Hessian, normal termination), no banner / authors / runtime / citations, so that the forging is purely about parser-equivalence and stays small.
12. As an ARC developer, I want a Gaussian log renderer that is a pure function (data dict → log text) with no ARC dependencies, so that I can unit-test it by round-tripping through Arkane's `GaussianLog` and asserting bit-equal extraction.
13. As an ARC developer, I want the renderer to emit Hessian blocks in every freq log (and to add `iop(2/9=2000)` to the route line in composite logs that combine opt+freq), so that Arkane's >13-atom Hessian check passes for arbitrarily sized test species.
14. As an ARC developer, I want a `restart_invariants` checker as a pure function — given a restart-state snapshot, return a list of invariant violations between `running_jobs` and `job_dict` — so that the same function powers Tier-1 tests and acts as a precondition inside `save_restart_dict()`.
15. As an ARC developer, I want `save_restart_dict()` to call the invariants checker before writing and either repair or refuse-and-log, so that #632 and #624 become structurally impossible rather than fixed at one calling site.
16. As an ARC developer, I want a Tier-1 test that runs ARC end-to-end on a multi-conformer species via mockter and asserts every `save_restart_dict()` call produces an invariant-clean `restart.yml`, so that #632 and #624 stay fixed.
17. As an ARC developer, I want a stop-and-restart test harness (pytest fixture) that runs ARC in a subprocess, lets mockter abort that subprocess at a fixture-controlled point, then re-launches ARC from the saved `restart.yml`, so that Tier-2 tests are isolated and the test runner doesn't die when ARC dies.
18. As an ARC developer, I want a Tier-2 test for the conformer-isomorphism-troubleshoot path (#622) where all conformers are non-isomorphic on first run and ARC restarts to find an empty `job_dict`, so that the empty-job-dict-but-results-on-disk path is fixed and stays fixed.
19. As an ARC developer, I want a Tier-2 test for the composite-then-freq phase ordering (#358) where a composite job is interrupted mid-flight and on restart freq must wait for composite completion, so that the phase-gate enforcement is regression-protected.
20. As an ARC developer, I want a Tier-3 thermo equivalence test that runs ARC twice on the same chemistry — once parsing real Gaussian logs directly, once via mockter rendering the same numbers — and asserts bit-equal output dicts and Arkane outputs, so that mockter is provably parser-equivalent for the thermo path.
21. As an ARC developer, I want a Tier-3 kinetics equivalence test for one full reaction (TS + IRC + kinetics) under the same protocol, so that mockter is provably parser-equivalent for the kinetics path.
22. As an ARC developer, I want a rotor-scan restart scenario, so that state-heavy rotor restart bugs are surfaced even though they aren't in the current ticket list.
23. As an ARC researcher, I want my hand-curated DFT-derived fixtures (from real Gaussian runs at e.g. wb97xd/def2tzvp + CCSD(T)-F12) committed alongside the real Gaussian log files, so that the numbers are auditable and regeneratable later.
24. As an ARC researcher running real species through the production pipeline, I don't want the mockter overhaul to change anything about non-mockter levels of theory, so that production runs are unaffected by test infrastructure.
25. As an ARC researcher who hits a restart crash on the cluster, I want the test suite to have already exercised the failure mode against a deterministic mock ESS, so that crashes I see are actual production bugs rather than known restart-machinery flakes.
26. As an ARC contributor, I don't want mockter to have any FF dependencies (RDKit MMFF94s, OpenBabel UFF, xtb), so that test outputs don't drift when those libraries upgrade.
27. As an ARC contributor adding a new test scenario, I want a documented procedure (run DFT → extraction script → commit fixture + real logs) in `arc/testing/mockter_fixtures/README.md`, so that I can add a scenario without reverse-engineering the schema.
28. As an ARC maintainer reviewing PRs, I want fixture YAMLs schema-validated at load time so a malformed fixture fails the test loudly and points at the offending file, rather than producing a confusing crash deep inside the renderer.
29. As an ARC maintainer, I want SIGTERM safety (signal handler that triggers `save_restart_dict()` cleanly) as a deferred follow-on, so that "Ctrl-C is now safe" is delivered after the higher-impact restart fixes land.
30. As an ARC contributor, I want every test scenario to specify which fixture key, if any, should `raise` and what kind, so that interrupt-and-resume tests are reproducible across machines and CI workers.
31. As an ARC developer, I don't want mockter to write to `scheduler.py` or any other production module via test hooks; the abort mechanism lives entirely in `mockter.py`, so that test concerns don't leak into production code paths.
32. As an ARC developer, I want the existing `output.yml` mockter artifact retained as a debug mirror alongside the canonical `output.log`, so that a developer inspecting a failed mockter run can see structured data without re-parsing the forged log.

## Implementation Decisions

### Architecture
- Mockter stops being an opaque dummy; it becomes a fixture-driven test instrument with three internal layers: fixture loader/lookup, Gaussian-format renderer, and the adapter glue. The renderer has zero ARC dependencies and is unit-testable in isolation.
- Fixture selection rides on `Level.method`. A new regex `^mockter(\d{1,2})$` (allowing one or two digits) routes to the mockter adapter and identifies which fixture file to load. Composite scenarios use `mockter_CBS-QB3_N` (no basis), routed via existing composite-method handling. No new top-level `input.yml` field, no env vars — subprocess-safe.
- The fixture format is one YAML file per scenario. Schema is versioned (`schema_version: 1`); a malformed or wrong-version fixture fails to load loudly. Real Gaussian log files used as the source of truth for fixture extraction are committed alongside.
- The forged Gaussian log is purely scientific: route section, "Input orientation" geometry blocks, "SCF Done" energy line (or CCSD/CCSD(T) variants), "Frequencies --" blocks, ZPE line, "Force constants in Cartesian coordinates" Hessian block, "Normal termination of Gaussian". No banner, citations, runtime, or machine info. Hessian appears in every freq log; for composite (combined opt+freq) logs the route section includes `iop(2/9=2000)` to satisfy Arkane's >13-atom check.
- The fixture lookup key is `(label, job_type, conformer | tsg | torsions | irc_direction)` as appropriate for the job. On miss, mockter logs a `WARN`, writes a `mockter_fallback.flag` file in the job's local path, and renders with today's hand-rolled values so a missing fixture entry doesn't crash a partially-keyed test scenario but is detectable.
- A fixture entry may carry a `raise:` clause (`crash | oom | timeout | scf_nonconvergence | sigterm`). On encountering one, mockter raises a `MockterAbort` with the indicated kind, or for `sigterm` calls `os.kill(os.getpid(), signal.SIGTERM)`. The abort mechanism lives entirely in mockter; `scheduler.py` is not modified for test purposes. Tests run ARC in a subprocess so an abort doesn't take the test runner down.
- A new `restart_invariants` module provides a pure function `check_restart_dict_consistency(restart_dict) -> list[InvariantViolation]` that asserts (a) every name in `running_jobs[label]` resolves to a key in `job_dict[label][...]`; (b) every conformer index in `running_jobs` matches a populated entry in `job_dict[label]['conformers']`; (c) every TS-guess index matches `job_dict[label]['tsg']`; (d) per-job `job_type` keys are mutually exclusive with the conformer/tsg name conventions. The same function is invoked from `save_restart_dict()` as a precondition: violations are repaired (preferred — re-populate `job_dict` from on-disk artifacts) or, if irrecoverable, the save is refused with a clear error rather than producing a malformed `restart.yml`.
- Bug fixes are scoped per-issue but framed by the invariants checker:
  - **#632 / #624**: `process_conformers` wipes-and-repopulates `job_dict[label]['conformers']` interleaved with `save_restart_dict()` calls. Fix: complete the wipe-and-repopulate atomically before the first save, OR rebuild `job_dict` from disk before save when an invariant violation is detected.
  - **#622**: `troubleshoot_conformer_isomorphism` indexes `job_dict[label]['conformers'][0]` blindly. Fix: detect "all conformers complete on disk but `job_dict` empty" state on restart and rebuild `job_dict[label]['conformers']` from the on-disk artifacts before troubleshooting.
  - **#358**: scheduler's "what's next" doesn't gate freq on composite completion when restart reconstructs state. Fix: enforce phase-gate in the scheduler's job-spawning predicate using `output_dict[label]['paths']['composite']` populated-and-real as a hard precondition for spawning freq.

### Modules
- **Gaussian log renderer** (new). Pure function module. Inputs: data dict + route flags. Output: log text. No imports from `arc.scheduler`, `arc.job.adapters` (other than for unit tests), `arc.species`. Round-trip testable against Arkane's `GaussianLog`. Owns the canonical Gaussian-format knowledge.
- **Fixture loader** (new). Owns the YAML schema. `Fixture.load(path)`, `fixture.lookup(...)`, `entry.is_raise()`, `entry.raise_kind()`, `entry.payload()`. Schema validation at load time, including the "isomorphic-true XYZs really are isomorphic" cross-check.
- **Restart invariants** (new). Pure function `check_restart_dict_consistency(restart_dict) -> list[InvariantViolation]` and a small `InvariantViolation` data class. Used by tests AND by `save_restart_dict()`.
- **Fixture builder script** (new). CLI that reads an ARC project directory, uses `arc.parser`, writes a schema-validated fixture YAML.
- **Mockter adapter rewrite** (modified). New `execute_incore` orchestrates fixture lookup → render → write; the canonical mockter `output_filenames` switches to `output.log` with `output.yml` retained as a debug mirror. Falls back deterministically on miss.
- **Test harness** (new). pytest fixtures providing subprocess-isolated ARC runs with restart cycling. Used by Tier-2 tests.
- **Level routing** (small change to `arc/level.py`). Add `mockter\d{1,2}` recognition to `Level.deduce_software`.
- **Settings** (small change). Update mockter `output_filenames` and ensure the mockter-N composite-method routing is recognized by the existing composite-method comparison.
- **Bug-fix patches** (modifications to `arc/scheduler.py`). Three localized fixes for #632/#624, #622, #358, with the invariants checker as backstop.

### Schema (fixture v1)
```
schema_version: 1
provenance:
  generated_at: ISO date
  source_project: path to the ARC project that produced the data
  generated_by: extraction-script identifier
  ess: gaussian
  level_of_theory: e.g. wb97xd/def2tzvp for opt/freq, CCSD(T)-F12/cc-pVTZ-F12 for sp
species:
  <label>:
    smiles: ...
    multiplicity: ...
    charge: ...
    conformers:
      - xyz: <yaml literal block>
        e_elect: <Hartree>
        isomorphic: <bool>
        note: <optional string>
      - raise: { type: <kind>, message: <string> }   # interrupt entry
    opt: { xyz, e_elect }
    fine_opt: { xyz, e_elect }
    freq: { freqs (cm^-1), zpe (Hartree), hessian (J/m^2 matrix or null) }
    sp: { e_elect (Hartree), t1_diagnostic (or null) }
    scans:
      - { torsions: [[...]], energies (Hartree), xyzs }
    irc:
      forward: { xyzs, energies (Hartree) }
      reverse: { xyzs, energies (Hartree) }
reactions:
  <reaction_label>:
    ts_label: ...
    multiplicity: ...
ts:
  <ts_label>: same shape as a species, plus 'guesses: [{xyz, method}]' indexed by tsg
```

### Test scenarios
- `mockter1_conformer_save_invariants` — n-butane multi-conformer, all isomorphic, single-run; covers #632 / #624.
- `mockter2_isomorphism_trsh` — n-butane multi-conformer with isobutane xyz injected on 1–2 conformers; `raise: sigterm` after isomorphism check fails; covers #622.
- `mockter3_composite_phase_ordering` — CH3OH at CBS-QB3 (or similar composite); `raise: oom` mid-composite; covers #358.
- `mockter4_thermo_equivalence` — C2H6, OH, H2O thermo; Tier-3 equivalence check.
- `mockter5_kinetics_equivalence` — C2H6 + OH H-abstraction reaction with TS + IRC; Tier-3 equivalence check.
- `mockter6_rotor_scan_restart` — propanol rotor scan with `raise: timeout` mid-scan; rotor-state coverage.

### Out-of-band coordination
Per-scenario ARC input files will be created under `arc/testing/mockter_fixtures/_input_files/scenarioN/` as a temp folder for the maintainer to run real DFT against. For interrupt scenarios the input files include a comment naming the post-job interrupt point so the recorded outputs match what mockter will replay before raising. All scenarios use Gaussian for opt/freq; sp uses the user's preferred WF level — the real outputs become the forging reference.

## Testing Decisions

### What makes a good test here
- **External behavior, not implementation**: tests assert on `output_dict` content, `restart.yml` content, Arkane outputs (NASA polynomials, k(T)), and invariants between data structures. They never inspect mockter or scheduler internals beyond those public outputs.
- **Deterministic both sides**: equivalence tests assert *bit-equal* outputs between real-DFT-driven and mockter-driven runs. Numerical input is identical (extracted from the same real Gaussian logs); both pipelines must produce identical numerical output. Any drift indicates a forging/parsing equivalence bug.
- **Subprocess isolation for Tier-2**: stop-and-restart tests run ARC in a subprocess so `MockterAbort` (or signal) doesn't kill the test runner. Restart cycling is driven by re-launching ARC with the saved `restart.yml`.
- **Invariants over enumeration**: prefer asserting that an invariant *holds* (e.g. "every name in `running_jobs` resolves in `job_dict`") over enumerating specific KeyError-or-not cases. The invariants checker is the assertion vehicle.

### Modules to test
- **Gaussian log renderer (high)**: unit + property-based round-trip vs. Arkane's `GaussianLog.load_geometry`, `load_energy`, `load_zero_point_energy`, `load_conformer`, `load_force_constant_matrix`, `load_negative_frequency`, `load_scan_energies`. For each, render → parse → assert bit-equal.
- **Restart invariants (high)**: unit tests on synthetic restart dicts. Each known bug pattern from #632 / #624 has a corresponding violation case; clean states pass; ambiguous edge cases (in-flight jobs vs. completed jobs) have explicit expected outcomes.
- **Fixture loader (medium)**: schema validation cases (missing required field, wrong schema version, isomorphic-true-but-graph-disagrees, malformed `raise:` clause).
- **Mockter runner integration (medium)**: via the test harness, run mockter end-to-end on a tiny fixture, assert `output.log` is parsed-equivalent to the fixture data.
- **Tier-1 (high)**: `test_restart_save_invariants` — runs ARC on fixture #1, asserts the invariants checker passes at every `save_restart_dict()`.
- **Tier-2 (high)**: `test_isomorphism_trsh_restart`, `test_composite_phase_ordering_restart` — subprocess harness, stop-and-restart cycle, asserts ARC behavior on resume.
- **Tier-3 (high)**: `test_thermo_equivalence`, `test_kinetics_equivalence` — bit-equal comparison between real-log and mockter-driven runs.
- **Fixture builder (low)**: one integration test on a tiny pre-recorded ARC project dir.

### Prior art
- Existing `functional/restart_test.py::TestRestart.test_restart_thermo` is the model for end-to-end restart testing. New tests follow that pattern for the ARC-instantiation-and-execute mechanics, and add the invariants/equivalence/subprocess-harness layers on top.
- `arc/job/adapters/mockter_test.py` is the existing test file for mockter; it'll be substantially expanded to cover the renderer and runner.
- `functional/test_mockter_*.py` is the proposed home for Tier-1, Tier-2, Tier-3 tests, organized per scenario.

## Out of Scope

- **Real DFT data generation**: the maintainer runs real Gaussian on the per-scenario input files; this PRD is about consuming the resulting outputs into fixtures.
- **SIGTERM signal handler in `ARC.execute()`**: deferred to a follow-on stage. The fixture `raise: sigterm` will work imperfectly until then (it kills the subprocess, but no clean save happens before death). For the bug families in this PRD, `raise: crash | oom | timeout | scf_nonconvergence` are sufficient.
- **Forging Orca / Molpro / QChem logs**: only Gaussian is targeted. Other ESS-format mocking is a future extension if the test suite needs to validate non-Gaussian parser branches.
- **Production runs with `mockter` levels**: mockter is a test instrument; it is not intended for any user workflow outside the test suite.
- **FF-based mockter physics** (RDKit MMFF94s, OpenBabel UFF, xtb): explicitly rejected. Determinism beats realism for a regression instrument.
- **Renderer support for IRC and rotor scans in the v1 schema beyond what scenarios 5 and 6 require**: schema includes the fields, but renderer/parser equivalence will be validated for these only via the relevant scenarios; broader IRC/scan coverage is incremental.
- **Performance optimization of mockter**: it's already incore; staging concerns are correctness and determinism.
- **Backwards compatibility for old `restart.yml` files written by current ARC**: the invariants checker accepts any schema-valid `restart.yml`; older files predating fields like `adaptive_levels` are out of scope unless the checker refuses them.
- **Cross-platform testing**: tests run on Linux only (CI baseline).

## Further Notes

Stages and dependencies (proposed):
- **Stage 0** — Mockter overhaul (renderer, fixture loader, runner, fixture builder, level routing, settings; renderer unit tests).
- **Stage 1** — Tier-1 test for #632 / #624 + invariants checker + bug fix.
- **Stage 2** — Tier-3 thermo equivalence (validates mockter forging).
- **Stage 3** — Tier-2 test for #622 + bug fix (depends on Stage 0 `raise:` semantics).
- **Stage 4** — Tier-2 test for #358 + bug fix.
- **Stage 5** — Tier-3 kinetics equivalence.
- **Stage 6** — Rotor scan / scenario #6.
- **Stage 7** — SIGTERM signal handler + matching test.

Stages 1 and 2 can run in parallel after Stage 0. Stages 3 and 4 are independent of each other but both depend on Stage 0.

Per-scenario ARC input files for the maintainer to run will be staged under `arc/testing/mockter_fixtures/_input_files/scenarioN/` with brief notes describing which post-job point (if any) ARC should be interrupted at.

Bugs are intentionally addressed by structural fixes, not just point patches: the invariants checker as `save_restart_dict()` precondition makes #632 / #624 a closed bug class. #622 and #358 are addressed by specific fixes plus regression tests; the same patterns may apply to similar latent bugs (rotor scan restart, IRC restart) that future scenarios will surface.
