# Mockter fixture input files

These are real ARC input files used to generate the source-of-truth DFT data
that mockter fixtures will replay. Run each `input.yml` against your normal
ARC setup; the resulting `Projects/<project>/calcs/` tree is what the fixture
extraction script (Stage 0) will consume.

## Conventions

- **Server-agnostic**: no `ess_settings` block. ARC will use `global_ess_settings`
  from `~/.arc/settings.py`.
- **Levels**:
  - opt / freq / scan: `wb97xd/def2tzvp` (Gaussian).
  - sp: `CCSD(T)/cc-pVTZ` (Gaussian; not Arkane-AEC-recognized — by design).
  - `arkane_level_of_theory: CCSD(T)-F12/cc-pVTZ-F12` — a recognized Arkane
    entry used as a "dummy model_chemistry" so Arkane applies AECs/BACs.
    The actual sp values come from the real Gaussian CCSD(T) run.
  - composite scenarios use `cbs-qb3` (Gaussian native).
- **Output target**: real Gaussian `.log` files. Mockter will mimic these.
- **Interrupt scenarios**: when a scenario calls for an interrupted real ARC
  run, the input file's leading comment names the post-job point at which
  ARC should be killed (Ctrl-C + `arcrestart` later, or just terminate).
  The interrupt is needed only if you want to capture a real `restart.yml`
  alongside; mockter's `raise:` mechanism is the test-time interrupt and
  doesn't require a real interrupted run to seed the fixture.
- **Cost**: each scenario should be ≤ a few hours on a small queue. Sized
  to be a regression-test fixture, not a research run.

## Scenarios

| Dir | Bug coverage | Tier |
|---|---|---|
| `scenario1_conformer_save_invariants/` | #632, #624 | 1 |
| `scenario2_isomorphism_trsh/` | #622 | 2 |
| `scenario3_composite_phase_ordering/` | #358 | 2 |
| `scenario4_thermo_equivalence/` | mockter validation (thermo) | 3 |
| `scenario5_kinetics_equivalence/` | mockter validation (kinetics) | 3 |
| `scenario6_rotor_scan_restart/` | rotor-state restart | 2 |

## After running

For each scenario, expected outputs the fixture builder will harvest:
- per-conformer Gaussian `.log` (geometry, energy)
- opt + fine_opt `.log`
- freq `.log` (geometry, freqs, ZPE, force constants)
- sp `.log` (electronic energy)
- scan `.log` for any rotor scans
- TS opt/freq/sp `.log` and IRC `.log` (scenario 5 only)

Commit the resulting `Projects/<project>/calcs/` tree under
`arc/testing/mockter_fixtures/_real_runs/scenarioN/` — the fixture builder
will read from there.
