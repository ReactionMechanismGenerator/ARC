# Mockter fixtures

Deterministic test fixtures for the `mockter` adapter. Each `mockterN.yml`
captures real DFT output data (geometry, energies, frequencies, ZPE, force
constants, scans, IRC where applicable) extracted from a one-time real ARC
run. Mockter looks up entries by `(species_label, job_type, ...)` and replays
them as skeletal Gaussian-format `.log` files for ARC to parse.

## Files

| File | Purpose |
|---|---|
| `mockter1.yml` … `mockter6.yml` | Per-scenario fixture data (v1 schema) |
| `_reference_outputs/sN_output.yml` | Snapshot of ARC's `output/output.yml` from the real run that produced each fixture — used as ground truth for Tier-3 equivalence tests |
| `_input_files/scenarioN/input.yml` | The ARC input file used to generate the real DFT data |

## Schema (v1)

```
schema_version: 1
provenance:
  generated_at, source_project, arc_git_commit, arkane_git_commit
  opt_level, freq_level, sp_level, composite_method, arkane_level_of_theory
  freq_scale_factor
species:
  <label>:
    smiles, multiplicity, charge, inchi
    conformers: [{xyz, e_elect (Hartree), isomorphic, note?}]
    opt:        {xyz, e_elect}            # coarse opt
    fine_opt:   {xyz, e_elect}            # final / fine opt
    freq:       {freqs (cm^-1), zpe (Hartree), hessian_block (verbatim Gaussian text), n_imag, imag_freq_cm1}
    sp:         {e_elect (Hartree), t1_diagnostic}
    composite:  {xyz, e_elect, hessian_block}    # only when composite_method is set
    scans: [{torsions, energies (Hartree, relative), xyzs}]
    rotors_meta: list of torsion metadata (pivots, symmetry, treatment)
reactions:
  <label>: {family, multiplicity, reactants, products, ts_label}
ts:
  <label>: same shape as a species (opt, fine_opt, freq, sp, irc, scans),
           plus rxn_label, converged, chosen_ts_method
```

## Units

- All electronic and ZPE energies in **Hartree**.
- Scan energies relative to the first scan point, in **Hartree**.
- Frequencies in **cm⁻¹**.
- `hessian_block`: verbatim text of Gaussian's `Force constants in Cartesian
  coordinates:` block, included so mockter can re-emit it byte-identically
  without re-rendering the matrix.

## Scenarios

| Fixture | Bug coverage | Tier | Species / TS |
|---|---|---|---|
| `mockter1.yml` | #632, #624 | 1 (single-run) | n_butane (3 conformers, 3 rotor scans) |
| `mockter2.yml` | #622 | 2 (stop-and-restart) | n_butane + iso_butane (wrong-isomer source) |
| `mockter3.yml` | #358 | 2 (stop-and-restart) | methanol (CBS-QB3 composite) |
| `mockter4.yml` | mockter validation | 3 (thermo equivalence) | C2H6, OH, H2O |
| `mockter5.yml` | mockter validation | 3 (kinetics equivalence) | C2H6, OH, C2H5, H2O + TS0 (PARTIAL — TS opt did not converge due to a normal-mode-displacement check; see scenario 5 notes) |
| `mockter6.yml` | rotor-state restart | 2 (stop-and-restart) | propanol (4 conformers, 3 rotor scans) |

## Caveats

- **Scenario 5 TS** is partially populated: opt and freq logs were captured,
  IRC trajectories are 1-point only, and the Molpro CC sp did not finish
  because ARC's NMD check rejected the TS geometry. Believed to be a false
  negative in the NMD check, tracked separately. The kinetics equivalence
  test will need either a re-run of scenario 5 with the NMD bug fixed, or
  a workaround that bypasses the NMD gate.
- The fixture builder (`arc/testing/scripts/build_mockter_fixture.py`) is
  one-shot: it consumed the source DFT runs and wrote these YAMLs. To
  regenerate, re-run the input files in `_input_files/` and re-execute the
  builder pointing at the new project directories.
