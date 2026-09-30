# output.yml Schema Reference

The consolidated `output.yml` is written atomically to `<project_directory>/output/output.yml`
at the end of an ARC run. It contains all result data in a single file with run-relative paths
so downstream consumers (TCKDB, analysis scripts) need only one file.

Fields marked **nullable** may be `null` when the species did not converge,
the job was not requested, or the data is not applicable (e.g. monoatomic species).
Other fields are instead **omitted** — the key is absent from the mapping rather
than present with a `null` value — which matters to a consumer reading the
document with `.get()`. Which of the two applies is stated per field in the
tables below.

The distinction is enforced mechanically by `arc/schemas/output_yml_schema.json`,
validated in `arc/output_schema_test.py` against documents built by the real
writer. That schema, not this prose, is the authority: if the two disagree, the
document is a bug in this file.

---

## Quick Overview

```
output.yml
├── schema_version: "1.3"
├── project
├── arc_version
├── arc_git_commit?
├── arkane_version?
├── arkane_git_commit?
├── rmg_database: {path_kind, git_commit?, version?, quantum_corrections_sha256?}
├── arc_aec_yml_sha256?
├── datetime_started?
├── datetime_completed
│
├── composite_method?
├── opt_level?
├── freq_level?
├── sp_level?
├── scan_level?, irc_level?, conformer_opt_level?, conformer_sp_level?, ts_guess_level?, gsm_level?
├── neb_level (omitted unless the run has a TS or a reaction and the orca_neb TS adapter was configured)
├── arkane_level_of_theory?
├── adaptive_levels?
├── freq_scale_factor?
├── freq_scale_factor_key?
├── freq_scale_factor_source?
├── bac_type?
├── atom_energy_corrections?
├── bond_additivity_corrections?
├── parser_evidence?
│   └── path, schema_name, schema_version, document_id
│
├── species: []
│   └── label, original_label, charge, multiplicity, converged
│       ├── smiles?, inchi?, inchi_key?, formula?
│       ├── xyz?, xyz_isotopes?
│       ├── conformers, conformer_energies, conformers_isotopes   (all omitted when none were screened)
│       ├── conformer_ess_software, conformer_ess_version   (omitted with conformers)
│       ├── conformer_levels?, conformer_energy_kind?, conformer_energy_level?, conformer_force_field?
│       ├── sp_energy_hartree?, zpe_hartree?, opt_converged?
│       ├── coarse_opt_log?, coarse_opt_n_steps?, coarse_opt_final_energy_hartree?
│       ├── opt_n_steps?, opt_final_energy_hartree?
│       ├── coarse_opt_input_xyz?, coarse_opt_output_xyz?, opt_input_xyz?
│       ├── coarse_opt_input_xyz_isotopes?, coarse_opt_output_xyz_isotopes?, opt_input_xyz_isotopes?
│       ├── freq_n_imag?, imag_freq_cm1?, imaginary_frequencies_cm1?
│       ├── opt_log?, freq_log?, sp_log?, composite_log?
│       ├── opt_route?, freq_route?, sp_route?, composite_route?, composite_step_routes?
│       ├── sp_t1_diagnostic?, opt_dipole_moment_debye?, opt_dipole_moment_density?
│       ├── freq_polarizability_angstrom3?
│       ├── sp_spin_diagnostic?
│       │   └── {s_squared, s_squared_expected?, s_squared_annihilated?, log?}
│       ├── wavefunction_stability?
│       │   └── {verdict, internal_instability, external_instability, relaxations,
│       │        negative_eigenvectors, lowest_eigenvalue, restricted,
│       │        invalidates_analytic_freq, log}
│       ├── scf_reference
│       │   └── {source, declared_number_of_radicals, verdict, verdict_restricted,
│       │        measured_on_ts_guess, sp_reference, freq_reference,
│       │        reference_mismatch, log}
│       ├── opt_input?, freq_input?, sp_input?, composite_input?
│       ├── opt_constraints: [], freq_constraints: [], sp_constraints: []
│       │   └── [{coordinate_type, atom_indices, index_base,
│       │        target_value?, target_value_units?}, ...]
│       ├── freq_hessian_method?
│       ├── opt_final_settings?, coarse_opt_final_settings?
│       ├── freq_final_settings?, sp_final_settings?
│       ├── rotor_scans: []
│       │   └── key, source_log?, ess_software?, ess_version?, constraints?, result
│       │       ├── dimension, relaxed?, zero_energy_reference_hartree?
│       │       ├── coordinate: {coordinate_type, atom_indices, index_base, unit,
│       │       │                sample_count, symmetry_number?,
│       │       │                requested_step_size?,
│       │       │                requested_start?, requested_end?}
│       │       └── samples: [{source_index, angle_degrees,
│       │                      relative_energy_kj_mol,
│       │                      electronic_energy_hartree?, geometry_xyz?, geometry_isotopes?}, ...]
│       ├── energy_corrections: []
│       │   └── correction_type, model, level_of_theory?, matched_arkane_key?,
│       │       total, components, skipped_components, reference_atom_energies?, parameter_table?
│       ├── ess_versions?, ess_software?
│       ├── levels: {opt?, freq?, sp?, composite?, irc?}
│       ├── thermo?
│       │   ├── h298_kj_mol, s298_j_mol_k, tmin_k, tmax_k
│       │   ├── standard_state_pressure_pa?
│       │   ├── atom_corrections_applied?, bond_corrections_applied?, atom_corrections_level?
│       │   ├── thermo_points?: [{temperature_k, cp_j_mol_k, h_kj_mol, s_j_mol_k, g_kj_mol}, ...]
│       │   ├── nasa_low?: {tmin_k, tmax_k, coeffs}
│       │   └── nasa_high?: {tmin_k, tmax_k, coeffs}
│       ├── irc_endpoint_of?, irc_endpoint_direction?   (species records only)
│       └── statmech?
│           ├── e0_kj_mol?, e0_atom_corrections_applied?, e0_bond_corrections_applied?
│           ├── arkane_rotors_applied?, arkane_treatment?
│           ├── spin_multiplicity, optical_isomers
│           ├── is_linear, external_symmetry, point_group?
│           ├── rigid_rotor_kind, harmonic_frequencies_cm1?
│           ├── torsions: []
│           │   └── symmetry_number, treatment?, atom_indices, pivot_atoms,
│           │       barrier_kj_mol?, source_scan_key?
│           └── rejected_torsions?: []
│               └── rotor_index, invalidation_reason, atom_indices, pivot_atoms,
│                   dimension, source_log?
│
├── transition_states: []
│   └── (all species fields, plus:)
│       ├── chosen_ts_method?, successful_ts_methods?
│       ├── ts_guesses: []
│       ├── neb_log?, gsm_log?, irc_logs: [], irc_log_routes: [], irc_log_directions: [], irc_log_levels: [], irc_converged?
│       ├── freq_frequencies_cm1_ess_order?, reaction_coordinate_mode_index?, nmd_forced?, neb_succeeded?
│       ├── ts_checks: {E0?, e_elect?, IRC?, freq?, NMD?, warnings}
│       ├── irc_participant_mapping?
│       └── rxn_label
│
└── reactions: []
    └── label, reactant_labels, product_labels, reactant_species_labels?, product_species_labels?, family?, multiplicity, ts_label, reversible?, atom_map?, atom_map_reactant_labels?, atom_map_product_labels?, atom_map_source?, atom_map_method?, ts_atom_map?, ts_atom_map_unavailable_reason?
        ├── long_kinetic_description   (omitted when empty)
        └── kinetics?
            └── A, A_units?, n, Ea, Ea_units?, Tmin_k, Tmax_k
                dA?, dn?, dEa?, dEa_units?, n_data_points?, tunneling, atom_corrections_applied?,
                comment?, ts_validation?
```

`?` in the tree above means "not always a value" and does **not** distinguish
nullable from omitted. That distinction is contractual and is stated per field in
the tables below, and is enforced mechanically by
`arc/schemas/output_yml_schema.json`: a field documented as **omitted** is a
validation error if present as `null`, and vice versa. When this document and the
schema disagree, the schema is authoritative — it is validated against documents
produced by the real writer in `arc/output_schema_test.py`.

TS entries share all species fields but add IRC/NEB/method fields and always have
`thermo: null`.

---

## Top-level

| Field | Type | Description |
|---|---|---|
| `schema_version` | `str` | Output contract version. `1.3` adds, in every record, `levels`, `conformer_levels`, `conformer_energy_kind`, `conformer_energy_level`, `conformer_force_field`, `conformer_ess_software`, `conformer_ess_version`, `composite_log`, `composite_input`, `opt_route`, `freq_route`, `sp_route`, `composite_route`, `composite_step_routes`, `opt_dipole_moment_debye`, `opt_dipole_moment_density`, `freq_polarizability_angstrom3`, `sp_t1_diagnostic`, the isotope lists (`xyz_isotopes`, `conformers_isotopes`, `coarse_opt_input_xyz_isotopes`, `coarse_opt_output_xyz_isotopes`, `opt_input_xyz_isotopes`, a scan sample's `geometry_isotopes`) and a rotor scan's own `ess_software` and `ess_version`; in species records `irc_endpoint_of` and `irc_endpoint_direction`; in TS records `irc_log_levels`, `irc_log_routes`, `freq_frequencies_cm1_ess_order`, `reaction_coordinate_mode_index`, `nmd_forced`, `neb_succeeded` and `irc_participant_mapping`; in the header `adaptive_levels`, `scan_level`, `irc_level`, `conformer_opt_level`, `conformer_sp_level`, `ts_guess_level`, `gsm_level`, `rmg_database` and `arc_aec_yml_sha256`, and the `composite`, `irc` and `gsm` keys of `ess_versions` and `ess_software`; in `statmech`, `e0_atom_corrections_applied`, `e0_bond_corrections_applied`, `arkane_rotors_applied` and `arkane_treatment`; in `thermo`, `atom_corrections_applied`, `bond_corrections_applied` and `atom_corrections_level`; in `energy_corrections` records, the bonds a Petersson BAC applied as `components` and the unparametrized ones as `skipped_components`; and in reactions `reversible`, `reactant_species_labels`, `product_species_labels`, `atom_map`, `atom_map_reactant_labels`, `atom_map_product_labels`, `atom_map_source`, `atom_map_method`, `ts_atom_map`, `ts_atom_map_unavailable_reason`, and in `kinetics` `comment`, `ts_validation` and `atom_corrections_applied`. `1.3` changes `statmech.rigid_rotor_kind` (which now also takes `symmetric_top` and `spherical_top`, read from the point group), `statmech.torsions[].treatment` (nullable, read from Arkane's output), `energy_corrections` (a record is kept only where the run behind the exported energy is recorded as having applied it), the `conformer_energies` description (absolute energies, not relative ones), and `neb_level` (exported in runs that use the default TS adapters and have a TS or a reaction). Every new value is `null` when it was not observed, and none is reconstructed from a level of theory. `1.1` adds the optional `parser_evidence` descriptor, renames the thermo block's `cp_data` to `thermo_points` (widened from Cp-only to Cp/H/S/G per temperature; there is no `cp_data` alias), renames the AEC `parameter_table` to `reference_atom_energies`, adds `imaginary_frequencies_cm1`, the per-correction `matched_arkane_key` and `freq_scale_factor_key`, adds `ess_software` so each `ess_versions` banner can be paired with the program that actually produced it, adds `arkane_version`, read together with `arkane_git_commit` from one install so the run carries an Arkane software identity even where there is no git repository to report a commit from, adds `freq_hessian_method`, whose three tokens are TCKDB's `HessianMethod` vocabulary and record the provenance of the frequency method ARC requested, adds the optional `statmech.rejected_torsions`, adds the TS-only `ts_checks`, adds `kinetics.T0_k` and the constraints' `target_value_units`, adds the thermo block's `standard_state_pressure_pa`, which names the standard state its entropy, free energy and NASA fit belong to, and adds the `rotor_scans` block, whose samples carry `angle_degrees` — the **absolute** dihedral measured on each sample's own geometry, never a displacement — and whose `requested_step_size` is signed, so a reversed scan records its direction instead of losing the whole requested grid. Emitted from a single constant shared with the evidence sidecar's `output_schema_version`, so the two cannot drift |
| `project` | `str` | ARC project name |
| `arc_version` | `str` | ARC version string |
| `arc_git_commit` | `str?` | ARC repo HEAD commit hash |
| `arkane_version` | `str?` | Version of the Arkane (RMG-Py) install ARC invoked for this run, whether or not that run produced corrections; `null` when it cannot be resolved |
| `arkane_git_commit` | `str?` | RMG-Py (Arkane) repo HEAD commit hash, from the same install as `arkane_version` |
| `rmg_database` | `dict` | The identity of the quantum corrections tables Arkane used: `{path_kind, git_commit, version, quantum_corrections_sha256}`. Arkane reads `quantum_corrections/data.py` from the database its own `rmgpy` settings point at (`arkane.encorr.data.quantum_corrections_path`, reported by the RMG conda environment), which is not necessarily the file under ARC's `RMG_DB_PATH`; `quantum_corrections_sha256` is the SHA-256 of that file's bytes. `path_kind` is `"git"` when it lies in a git checkout (`git_commit` is then its `HEAD`, read from the `.git` files without running git), `"package"` when it lies inside a conda prefix that lists the `rmgdatabase` package (`version` is then that package's version, and there is no commit), and `"unknown"` otherwise. `path_kind` is `"git"` only when the database root, three directories above that file (`<root>/input/quantum_corrections/data.py`), holds a `.git`; an unrelated repository further up is not taken for it. The tables themselves come from the file Arkane loaded. When the file ARC matches its correction keys against under `RMG_DB_PATH` has another digest, a matched key may be missing or different in Arkane's file, and ARC logs a warning. Each value but `path_kind` is `null` when it cannot be determined. The digest identifies the `data.py` tables only: atom energies ARC renders from its own `data/AEC.yml` are identified by `arc_aec_yml_sha256`, and two runs with the same digest can still have used different atom energies. Always emitted, never raises |
| `arc_aec_yml_sha256` | `str?` | The SHA-256 of ARC's own `data/AEC.yml`, recorded by an Arkane adapter at the moment it rendered `atomEnergies` from that file into an Arkane input during the run. The digests come from the adapters of the final processing and from the adapter that computed the E0 of every exported species and TS, so the E0 computed by the TS check is covered: its digest is recorded with the E0 and travels with it when it is copied onto the TS and the wells. It is exported only when exactly one distinct such digest was recorded. `null` when none was (the atom energies came from Arkane's own database, no Arkane input was rendered, or the run was restarted from a version that did not record it) and, with a warning in the log, when several differing digests were |
| `datetime_started` | `str?` | Run start timestamp (`YYYY-MM-DD HH:MM`); `null` when ARC has no recorded start time |
| `datetime_completed` | `str` | Completion timestamp (`YYYY-MM-DD HH:MM`) |
| `parser_evidence` | `dict` | Descriptor for `parser_evidence.json`. **Omitted** (key absent, never `null`) unless the sidecar was written successfully |

The descriptor binds `output.yml` to the parser-neutral evidence sidecar with
`path`, `schema_name`, `schema_version`, and a shared UUID4 `document_id`.
ARC writes the sidecar first and `output.yml` second so consumers can reject a
stale or interrupted pair by comparing document IDs. The evidence file contains
best-effort, versioned Hessian, IRC, and GSM facts parsed by ARC; it contains no
TCKDB upload request objects.

The sidecar's `output_schema_version` is emitted from the same constant as
`schema_version` above, so a consumer gating on it cannot be misled by drift
between the two files.

The sidecar's `producer` block identifies the software the evidence came from:
`name` (always `ARC`), `version` and `git_commit` for ARC itself, and
`arkane_version` and `arkane_git_commit` copied from the fields of the same name
in `output.yml`. All four version and commit fields are always present and are
`null` when `output.yml` carries no value for them, including when it predates
the Arkane fields entirely. Naming the Arkane install in the sidecar lets a reader
identify the software ARC invoked without holding `output.yml` alongside it; both
files report it from the single resolution described under
[Arkane provenance](#arkane-provenance), so they cannot disagree.

### Arkane provenance

`arkane_version` and `arkane_git_commit` identify the Arkane (RMG-Py) install ARC
invoked for this run, the way `arc_version` and `arc_git_commit` identify ARC itself.

**They name the software that ran, not a claim about what it produced.** Arkane prints
the header they are read from when it starts, before it computes anything, and ARC does
not act on whether that run then failed. So a project whose Arkane run died still
carries the identity of the Arkane that died — which is what a reader diagnosing the
failure needs — and `atom_energy_corrections` and `bond_additivity_corrections` can be
empty while these two fields are populated. Neither field is evidence that the
corrections in the document came from a completed Arkane run; the corrections
themselves are.

**Both come from one install.** They are read together, never resolved separately, so
they cannot name two different RMG-Py trees in the same document. The first source is
the header Arkane prints at the top of its own log, at
`calcs/statmech/<thermo|kinetics>/{arkane.log,stdout.log}` — it carries both the version
and the RMG-Py HEAD of the Arkane that actually ran this project. A log last modified
before the run started is skipped, because that directory survives between runs and an
older log describes an earlier one. A log holding more than one header is read from the
first one, so the version and the commit always come from the same run. If no such log
yields a version, both fields come instead from `RMG_PATH`: `__version__` in
`rmgpy/version.py`, and that repository's HEAD.

`arkane_git_commit` alone is `null` whenever the source reports no repository, which is
the normal case for a pip- or conda-installed RMG-Py; `arkane_version` still names the
software there, which is the point of the pair.

Both are `null` when no Arkane log is readable and `RMG_PATH` does not point at an
RMG-Py source tree. `RMG_PATH` is discovered from `$RMG_PATH`, `$GITHUB_WORKSPACE`,
`sys.path` entries and `~/Code/RMG-Py` only — it has no site-packages search — so on a
conda install with nothing exported the fallback contributes nothing and the log is the
only source.

### Hessian evidence and Cartesian frames

Each `freq_hessian` record pairs `lower_triangle` (packed lower triangle,
row-major including the diagonal, in hartree/bohr²) with `geometry_xyz_text`
and a `frame` label naming the Cartesian frame **both** share.

This matters more than it looks. A Hessian is only meaningful against the
geometry it was evaluated at, in the same frame — mass-weighting and projecting
out translation and rotation both mix the two. Gaussian prints its force
constants in the *input* orientation while reporting geometries in the
*standard* orientation, and the two are a pure rigid-body rotation apart:
identical internal distances, different Cartesian coordinates. Pairing them
reconstructs a materially different spectrum — invented low-frequency modes and
errors of hundreds of wavenumbers, with a TS's imaginary mode moving by an order
of magnitude — and **no size, length, or finiteness check can detect it**,
because a rotation changes none of those.

ARC therefore reads the geometry from the Hessian's own source: Gaussian's last
`Input orientation:` table (`frame: "gaussian_input_orientation"`) or Orca's
`.hess` `$atoms` block (`frame: "orca_hess_atoms"`). When that frame-matched
geometry cannot be recovered, the record degrades to
`{"status": "unavailable", "reason": "hessian_frame_unavailable"}` rather than
shipping a mismatched pair. `parser_version` is `arc-hessian-1`.

`frame` is an ARC-local field. Consumers that forward this data to a schema
which fixes the frame by contract rather than by label (TCKDB among them) should
use it to validate, then drop it during payload translation — as with every
other ARC-native name here, translation is the consumer's job, not ARC's.

### GSM path evidence and the two reaction coordinates

Each `gsm` record's points carry the stringfile geometry, the stringfile's own
relative energy (`stringfile_relative_energy_kcal_mol`), and
`cumulative_com_superposed_displacement_angstrom`.

That coordinate is the running sum, zero at the first frame, of the displacement
between consecutive frames after each pair is superposed: translated so their
centers of **mass** coincide, then optimally rotated. Each term is the Frobenius
norm of the superposed coordinate difference — `sqrt(sum_atoms |dr|^2)`, which is
`sqrt(N_atoms)` x RMSD. The norm itself is unweighted, but the superposition
frame is mass-dependent, so the value is not invariant to isotopic substitution
and is an upper bound on the freely-superposed minimum. The name states which
superposition produced the number.

It is **not** a reaction coordinate. The IRC records' `reaction_coordinate_sqrt_amu_bohr`
is the mass-weighted intrinsic reaction coordinate, in bohr*sqrt(amu) and signed about
the TS: Gaussian prints a per-branch cumulative arc length, which is non-negative on
both branches, and ARC negates it on every point whose `direction` is `reverse`, so the
two branches of one reaction concatenate into a single coordinate increasing from
reactants to products. `abs()` recovers the magnitude Gaussian printed, and a point
whose `direction` could not be read keeps that magnitude because its branch is unknown.
The two coordinates share no units, no sign convention, and no origin; they must never
be compared or plotted on one axis.

**Per-node energies are attached by geometry, never by id arithmetic.** GSM runs
every gradient through `./ograd <run>.<slot> <ncpu>`, and ARC's queue wrapper
archives each result as `gsm_node_outputs/<run>.<slot>.{energy,gradient,xtbout}`.
`slot` is GSM's ICoord array index plus `runend` (1), ICoords that are not string
nodes write into the same id space (the exact-TS optimizer uses `runend - 1`, the
string's own gradient object uses `runend` itself), and `cp -p` overwrites — so a
file holds whichever ICoord evaluated under that id last, and the id does not
identify a frame arithmetically. On a real run, `0000.01` holds the *last* frame's
geometry, not the first.

The `.gradient` file does, however, record the geometry its gradient was evaluated
at. ARC parses that geometry and superposes it against every frame, attaching the
energy and gradient norms only on a unique match within
`1e-3` Angstrom that no other invocation also claims. Attached points carry
`geometry_matched_ograd_invocation_id` and `geometry_match_displacement_angstrom`
alongside `electronic_energy_hartree` and the gradient norms; observed residuals
are ~1e-5 Angstrom against a nearest-non-match of ~0.3 Angstrom, four orders of
margin. Points with no verified match carry none of those keys.

`ograd_invocations` remains the verbatim record of what was archived — a list of
`{invocation_id, electronic_energy_hartree, max_gradient_hartree_per_bohr?,
rms_gradient_hartree_per_bohr?}`, ordered numerically by `(run, slot)`. It asserts
no correspondence to any `source_point_index`; the key is absent when nothing was
archived.

This matters because the xTB/GSM build writes `0.000000` into every stringfile
comment line, so `stringfile_relative_energy_kcal_mol` is identically zero on real
runs and the geometry-verified attachment is the only energy a GSM point carries.

`parser_version` is `arc-gsm-stringfile-1`.

## Levels of Theory

Every level field is a **level dict**: `Level.as_dict()` with the `repr` and
`compatible_ess` keys removed. Only the level's non-`None` attributes appear, so the
key set varies between runs; the possible keys are `method`, `basis`,
`auxiliary_basis`, `dispersion`, `cabs`, `method_type` (e.g. `dft`, `wavefunction`,
`composite`, `force_field`), `software`, `software_version`, `solvation_method`,
`solvent`, `solvation_scheme_level`, `args`, and `year`.

`year` appears only when the level carries one. On `arkane_level_of_theory` it selects
which RMG-database parameterisation of the atom-energy and bond-additivity corrections
Arkane matched, so two runs with the same `method` and `basis` but different `year` can
legitimately report different correction tables; the field is what tells them apart.

A level dict inside a species or TS record (`levels`, `conformer_levels`, `conformer_energy_level`) has no
`software` key and no `solvation_scheme_level`: a level's `software` is only ARC's deduction of which program would run it.
`ess_software` states the program of the opt, freq, sp, composite, IRC and xtb_gsm logs, and a rotor scan states its own `ess_software`; the programs of
conformer jobs are not stated. The header levels are requested
levels and keep `software`.

`solvation_scheme_level` is itself a nested level dict, held to the same key set —
`Level.as_dict()` leaves a `Level` object there, which ARC converts recursively so
the document contains only plain types that `yaml.safe_load` accepts. `args` is the
one free-form value (arbitrary ESS keywords).

The header `opt_level`, `freq_level` and `sp_level` are the levels of the run, not
necessarily the levels any one species was computed at: with `adaptive_levels` each
species' jobs run at a level chosen by its heavy-atom count. Use the record's `levels`
(see [Levels](#levels)) for the level a species' logs were computed at, and the header
`adaptive_levels` to tell that they differ by design.

| Field | Type | Description |
|---|---|---|
| `composite_method` | `dict?` | Composite method level (e.g. CBS-QB3, G4); `null` when no composite method was used |
| `opt_level` | `dict?` | Geometry optimization level |
| `freq_level` | `dict?` | Frequency calculation level |
| `sp_level` | `dict?` | Single-point energy level |
| `scan_level` | `dict?` | The requested level of the rotor scans (the level the scan jobs are submitted at, before troubleshooting). `null` when rotor scans were not requested (`job_types.rotors` false). It is a run-level value: a species without rotors ran no scan at this level. `null` also when an `adaptive_levels` entry names `scan`, since the scan level then depends on the species |
| `irc_level` | `dict?` | The requested level of the IRC jobs. `null` when IRC was not requested (`job_types.irc` false) or the run has no transition state. The level the IRC jobs of one TS ran at is that record's `levels.irc`. `null` also when an `adaptive_levels` entry names `irc` |
| `conformer_opt_level` | `dict?` | The requested level of the conformer optimizations of non-TS species. `null` when none could run: conformer optimization was not requested (`job_types.conf_opt` false) and no conformer was optimized anyway (ARC still optimizes conformers of a species that has no 3D structure), or the run has no computed non-monoatomic species. `null` also when an `adaptive_levels` entry names `conf_opt`. The level of each exported conformer is in `conformer_levels` |
| `conformer_sp_level` | `dict?` | The requested level of the conformer single points that follow the conformer optimizations. `null` unless they were requested (`job_types.conf_sp` true), conformer optimization could run, and the level differs from the run's conformer optimization level (ARC skips a conformer single point at the level of the optimization) and no `adaptive_levels` entry names `conf_sp` |
| `ts_guess_level` | `dict?` | The requested level at which the guesses of a TS are optimized and compared. `null` when the run has no TS. It is used only when more than one TS guess succeeded; a TS with a single successful guess is optimized directly at the opt level (`levels.opt`). The guesses are optimized as `conf_opt` jobs, so the level is `null` when an `adaptive_levels` entry names `conf_opt` |
| `gsm_level` | `dict?` | The level of the `xtb_gsm` path searches, `{method: gfn2, software: xtb}`. Stated only when the run has a GSM log and every archived per-node xtb output (`gsm_node_outputs/*.xtbout`) beside every GSM log shows the GFN2-xTB Hamiltonian and a program call with the charge and the number of unpaired electrons (multiplicity minus one) of the TS record. `null` when the run has no GSM log, when a GSM log has no archived xtb output, or when an output shows another method, charge or spin |
| `neb_level` | `dict` | NEB TS search level. **Omitted** (key absent) unless the run has at least one TS or reaction, the `orca_neb` TS adapter is among the adapters the scheduler uses (the user's `ts_adapters`, or ARC's default list when the user gave none) and `orca_neb_settings['level']` is set — it does not indicate that an NEB job actually ran |
| `arkane_level_of_theory` | `dict?` | Composite level Arkane uses for energy corrections |
| `adaptive_levels` | `list?` | The run's adaptive levels of theory in the list form ARC writes to `restart.yml`, or `null` when the run does not use `adaptive_levels`. Each entry is `{atom_range: [min, max], levels: {"<job types>": level dict}}`; `atom_range` is a heavy-atom count range whose last upper bound is the string `"inf"`, and job types that share a level are joined by a space (`"opt freq"`). Under adaptive levels a species' levels are chosen by its heavy-atom count (a reaction's participants by the reaction-wide count, which may add a `<label>_TS<i>` copy of a participant), so the header `opt_level`, `freq_level` and `sp_level` are run-level defaults only: the level a species' logs were computed at is that record's `levels`. An adaptive entry may name any job type; the header `scan_level`, `irc_level`, `conformer_opt_level`, `conformer_sp_level` and `ts_guess_level` are `null` when an entry names their job type (`scan`, `irc`, `conf_opt`, `conf_sp`, and `conf_opt` for `ts_guess_level`). |
| `freq_scale_factor` | `float?` | Harmonic frequency scaling factor. Resolved from the header `freq_level`; under `adaptive_levels` a species' frequencies may be from another level, so compare with the record's `levels.freq` |
| `freq_scale_factor_key` | `str?` | The entry key in ARC's `data/freq_scale_factors.yml` that `freq_level` resolved to — e.g. `"wb97xd/def2tzvp, software: gaussian"`. This is the block the factor was read from, so a consumer can check the factor against its own copy of the file. `null` when the user supplied the factor, when `freq_level` has no entry in that file, or when the file cannot be read. Under `adaptive_levels` it is resolved from the header `freq_level`, not from each species' own `levels.freq` |
| `freq_scale_factor_source` | `str?` | The literature source that the `freq_scale_factor_key` entry points at. Each entry in `data/freq_scale_factors.yml` carries an explicit integer `source` index which is resolved against the file's top-level `sources` mapping; this field is that resolved string. `null` whenever `freq_scale_factor_key` is `null`, and also when the matched entry carries no resolvable index |
| `bac_type` | `str?` | Bond additivity correction type the run **requested**: `"p"`, `"m"`, or `null`. Requested is not applied: ARC applies a BAC only in the thermo statmech run, so on a rates-only run (`compute_rates` without `compute_thermo`), on every transition state, and on a species whose own `compute_thermo` is off, no BAC is applied and no `bond_additivity` record appears under that species' `energy_corrections` |
| `atom_energy_corrections` | `dict?` | Arkane's **reference atomic electronic energies** in Hartree (`{element: value, ...}`), the table Arkane looked up for `arkane_level_of_theory`. These are *not* per-atom corrections and do not sum to the applied correction: Arkane **subtracts** them and additionally applies an `atom_hf - atom_thermal` term not represented here. For the per-atom quantities that do sum to the applied total, see `energy_corrections[].components[].contribution_value` |
| `bond_additivity_corrections` | `dict?` | Arkane's BAC parameter table for `arkane_level_of_theory`, in the native unit of Arkane's tables — **kcal/mol**. (Earlier revisions of this document said kJ/mol; the data was always kcal/mol.) The shape follows `bac_type`: for `"p"` (Petersson) a flat `{bond: float}` map; for `"m"` (Melius) Arkane's nested `{atom_corr, bond_corr_length, bond_corr_neighbor, mol_corr}` structure, where the first three are `{element: float}` maps and `mol_corr` is a float |

## Species

`species` is a list of entries, one per non-TS species.

Calculation constraints, rotor scans, and energy corrections are deliberately
tool-neutral. Constraint and scan coordinates retain their source atom indices
with an explicit `index_base`; scan samples retain source ordering and explicit
units; correction records retain the applied model, total, components, level
of theory, and native parameter table. Consumers own any database enum,
one-based-index, calculation-DAG, or payload nesting conversion.

### Identity

| Field | Type | Description |
|---|---|---|
| `label` | `str` | Species label |
| `original_label` | `str?` | Original user-provided label |
| `charge` | `int` | Molecular charge |
| `multiplicity` | `int?` | Spin multiplicity |
| `converged` | `bool` | Whether all requested jobs converged |
| `is_ts` | `false` | Always `false` for species |
| `smiles` | `str?` | SMILES string |
| `inchi` | `str?` | InChI identifier |
| `inchi_key` | `str?` | InChI key |
| `formula` | `str?` | Molecular formula |
| `xyz` | `str?` | Final (or initial) geometry as an XYZ block |
| `xyz_isotopes` | `list[int]?` | The isotope mass number of each atom of `xyz`, in its atom order (`xyz` itself carries none), **as the log of that geometry states it**: the opt log, or the composite log of a composite run, and only when it printed `Atom N has atomic number Z and mass M` for every atom (a Gaussian freq or composite job does; an opt-only Gaussian log does not). ARC's own geometry dicts are never read for isotopes, because the converter fills them with the most common isotope and ARC writes no isotope into an ESS input deck. `null` for every other program, for a log that states none, when `xyz` is not the final geometry of that log's job, and for the synthesized one-atom geometry of a monoatomic species. `conformers_isotopes` and the input-geometry lists below follow the same rule |

### Screened Conformers

`conformers`, `conformer_energies`, `conformers_isotopes`, `conformer_ess_software` and `conformer_ess_version` are **omitted** (absent, not `null`) when ARC
screened no conformers for the species, so a consumer must use `.get()` rather than
indexing. `conformer_levels`, `conformer_energy_kind`, `conformer_energy_level` and `conformer_force_field` are always
present and are `null` in that case.

| Field | Type | Description |
|---|---|---|
| `conformers` | `list[str]` | Screened conformer geometries as XYZ blocks, in ARC's conformer order |
| `conformers_isotopes` | `list[list[int]?]` | In lockstep with `conformers`: the isotopes each conformer's own optimization log (`conformer_logs`) states, as for `xyz_isotopes`. An entry is `null` for a conformer with no recorded log, for a conformer ARC holds only as text, and for a log that is missing or states no masses. Omitted with `conformers` |
| `conformer_energies` | `list[float?]` | One **absolute** energy per conformer (not relative to the lowest), in lockstep with `conformers`. An entry is `null` for a conformer that has no energy: a consumer must upload no energy for it. `conformer_energy_kind`, `conformer_energy_level` and `conformer_force_field` describe only the entries that are not `null`; a `null` entry never changes them, so `conformer_energy_kind` can be set while some entries are `null`. The unit depends on where each energy came from: **kcal/mol** for a force-field energy of the conformer screen, **kJ/mol** for an electronic energy parsed from a conformer optimization (or, when it ran, conformer single-point) log. A conformer whose optimization failed keeps its force-field geometry and force-field energy, so one list can mix the two units; `conformer_energy_kind` says which applies. Use `sp_energy_hartree` / `statmech.e0_kj_mol` for the energy of the species itself |
| `conformer_ess_software` | `list[str?]` | In lockstep with `conformers`: the ESS that ran the conformer optimization that produced each conformer geometry, identified from that optimization log (recorded in `conformer_logs`) by the same identification as `ess_software`. A conformer single point does not change it: when conformer single points ran, the scheduler may write the single point's geometry into `conformers[i]`, while this entry still names the optimization log, so it is not the program of the single-point energy (`conformer_energy_level` states that level). An entry is `null` for a conformer whose geometry came from no ESS optimization (a force-field geometry, a conformer whose optimization failed, a user-supplied one), whose log was not recorded (a restart written before the log was recorded), or whose log is missing from disk or states no program. Omitted with `conformers` |
| `conformer_ess_version` | `list[str?]` | In lockstep with `conformers`: the full banner the program printed in that same log, as for `ess_versions`. `null` wherever the matching `conformer_ess_software` entry is `null`, and when the optimization log states no banner. Omitted with `conformers` |
| `conformer_levels` | `list[dict?]?` | In lockstep with `conformers` (always the same length): the level of the conformer optimization job that produced each geometry. An entry is `null` for a geometry that was never optimized (a force-field geometry, including the one kept by a conformer whose optimization failed, or a user-supplied one) and for one whose level was not recorded (a restart written before this key existed). It is the level of the job that produced the geometry, so conformer troubleshooting at another level shows up here. A conformer single point does not change it. The whole key is `null` when the record has no conformers, which includes every transition state (TS guesses are not exported as conformers) |
| `conformer_energy_kind` | `str?` | `"force_field_kcal_mol"` when every energy held in `conformer_energies` is a force-field energy known to be in kcal/mol for the force field and backend that computed it; `"electronic_kj_mol"` when every one is an electronic energy in kJ/mol parsed from an ESS log. `null` when there is no energy, when the list mixes the two kinds (some conformers optimized, others still holding force-field energies), when the origin of an energy was not recorded (older restarts, energies read from a user-supplied conformers file), when an energy is a placeholder that no force field computed (the cheat-sheet, monoatomic and diatomic species), and when the force field's energy unit is not known to be kcal/mol (OpenBabel's UFF and Ghemical energies are kJ/mol) |
| `conformer_force_field` | `str?` | The force field and backend that produced `conformer_energies`, e.g. `"MMFF94s (rdkit)"`. RDKit falls back from MMFF to UFF when MMFF cannot be set up, and then it reads `"UFF (rdkit)"`. Set when every energy held came from one force field and backend, including one whose unit is not known to be kcal/mol (`conformer_energy_kind` is then `null`). `null` when there is no energy, for electronic energies, for placeholder energies, when the energies came from different force fields or backends, and when it was not recorded (older restarts) |
| `conformer_energy_level` | `dict?` | The level the electronic energies were computed at: the conformer single-point level when single points overwrote the optimization energies, else the conformer optimization level. `null` unless `conformer_energy_kind` is `"electronic_kj_mol"` and all energies held were computed at one level (so `null` for force-field energies, and when only some conformers got a single point) |

### Energies

| Field | Type | Description |
|---|---|---|
| `sp_energy_hartree` | `float?` | Single-point electronic energy (Hartree), **as the ESS reported it** — no atom-energy or bond-additivity correction has been applied. The corrections are reported separately under `energy_corrections`, which is also where their sign and unit conventions are stated. With a solvation scheme, it is the energy of the original sp job (`levels.sp`), not of the extra scheme jobs |
| `zpe_hartree` | `float?` | Zero-point energy (Hartree), **unscaled** — `freq_scale_factor` has *not* been applied, exactly as for `statmech.harmonic_frequencies_cm1`. It *has* been applied to `statmech.e0_kj_mol`, so `sp_energy_hartree + zpe_hartree` will not reproduce `e0_kj_mol`: the two differ by the applied energy corrections and by the unscaled ZPE excess `(1 - freq_scale_factor) * ZPE`. `null` for monoatomic |
| `opt_converged` | `bool?` | Whether geometry optimization converged |

### Optimization Details

| Field | Type | Description |
|---|---|---|
| `coarse_opt_log` | `str?` | Run-relative path to coarse optimization log |
| `coarse_opt_n_steps` | `int?` | Number of coarse optimization steps |
| `coarse_opt_final_energy_hartree` | `float?` | Final energy from coarse optimization |
| `opt_n_steps` | `int?` | Number of (fine) optimization steps |
| `opt_final_energy_hartree` | `float?` | Final energy from (fine) optimization |

**Optimization geometry provenance.** ARC's two-stage convention is
`initial_xyz → coarse opt → coarse output → fine opt → xyz`. When no coarse stage
ran, the chain collapses to `initial_xyz → opt → xyz`.

| Field | Type | Description |
|---|---|---|
| `coarse_opt_input_xyz` | `str?` | Geometry submitted to the coarse optimization; `null` unless a coarse stage ran and its output geometry parsed |
| `coarse_opt_output_xyz` | `str?` | Geometry the coarse optimization produced; `null` under the same condition |
| `opt_input_xyz` | `str?` | Geometry submitted to the fine optimization: the coarse output when a coarse stage ran and parsed, otherwise the species' initial geometry |
| `coarse_opt_input_xyz_isotopes` | `list[int]?` | The isotopes of `coarse_opt_input_xyz`, stated by the coarse opt log (the job that read that geometry) and taken only from a log that states the masses it used (a Gaussian log that printed 'Atom N has atomic number Z and mass M' for every atom, as a freq or composite job does); `null` otherwise and whenever `coarse_opt_input_xyz` is `null` |
| `coarse_opt_output_xyz_isotopes` | `list[int]?` | The isotopes of `coarse_opt_output_xyz`, which is parsed from the coarse opt log: taken only from a log that states the masses it used (a Gaussian log that printed 'Atom N has atomic number Z and mass M' for every atom, as a freq or composite job does); `null` for every other program and for a log that states none (an opt-only Gaussian log does not) |
| `opt_input_xyz_isotopes` | `list[int]?` | The isotopes of `opt_input_xyz`, stated by the opt log, the job that read that geometry, and taken only from a log that states the masses it used (a Gaussian log that printed 'Atom N has atomic number Z and mass M' for every atom, as a freq or composite job does); `null` otherwise and whenever `opt_input_xyz` is `null`. Most are `null`: an opt log rarely states its masses |

### Frequency Results

| Field | Type | Description |
|---|---|---|
| `freq_n_imag` | `int?` | Number of imaginary frequencies; `0` for a clean stable species, `null` for monoatomic or non-converged |
| `imag_freq_cm1` | `float?` | The most negative imaginary frequency (cm-1), or `null` when there are none. Normally `null` for a non-TS species, but a stable species that converged with a spurious imaginary mode reports it rather than hiding it |
| `imaginary_frequencies_cm1` | `list[float]?` | **All** imaginary frequencies (cm-1), not just the most negative; `null` when there are none, or when the species is monoatomic or non-converged |

### Log File Paths

All paths are relative to the project directory.

| Field | Type | Description |
|---|---|---|
| `opt_log` | `str?` | Geometry optimization log |
| `freq_log` | `str?` | Frequency calculation log |
| `sp_log` | `str?` | Single-point energy log |
| `composite_log` | `str?` | The composite-method (CBS-QB3, G4, ...) job log, from which a composite run's geometry and energy are read; `null` for a run that is not a composite run |
| `opt_input` | `str?` | Geometry optimization input deck; `null` when the deck is not on disk |
| `freq_input` | `str?` | Frequency calculation input deck; `null` when the deck is not on disk |
| `sp_input` | `str?` | Single-point energy input deck; `null` when the deck is not on disk |
| `composite_input` | `str?` | Composite-method input deck, a sibling of `composite_log` named for the program the log identifies; `null` when the deck is not on disk |
| `ess_versions` | `dict?` | ESS version banners, keyed by job type (`{'opt'|'freq'|'sp'|'composite'|'neb'|'irc'|'gsm': banner_str, ...}`). Each value is the **full banner** as the program printed it (e.g. `'Gaussian 16, Revision C.01'`, `'ORCA 6.0.0'`), not a bare version number — nothing trims it. A job type is absent when its log is missing or its banner could not be parsed; `irc` covers every recorded IRC log and is present only when they all state the same banner; `gsm` is the xtb banner of a TS's archived GSM node outputs, present only when they all state the same one; the whole field is `null` for a non-converged species or when nothing could be parsed |
| `ess_software` | `dict?` | The ESS that produced each log, keyed by the same job types and read from the same log files (`{'opt': 'gaussian', 'sp': 'orca', ...}`). Values are ARC's lowercase ESS names. Pair `ess_versions[job]` with `ess_software[job]` — a run may use different programs for different job types, so the level of theory's declared software is not a safe stand-in. The key set is a **superset** of `ess_versions`': a log whose ESS is identified but whose banner cannot be parsed appears here only, `irc` is present only when every recorded IRC log was identified as the same program, and `gsm` (`xtb`) only when every archived GSM node output was. `null` under the same conditions as `ess_versions` |

Each input deck sits in the same directory as its log, under the ESS-specific
filename from `settings['input_filenames']`. Software with no entry in that map
(`gcn`, `torchani`, `mockter`, ...) yields `null`.

### Effective ESS Keywords

`opt_route`, `freq_route`, `sp_route`, `composite_route` and `composite_step_routes` state the keyword line the job ran with, **observed, never rebuilt from a level of theory or from `job.args`**. The log's echoed route is read in preference to the input deck beside it, since the log is what ran; the deck is the fallback when the log holds none. The program is the one the log itself states (not the program a level requested). They are `null` for a species that did not converge, for any program other than Gaussian and Orca, and when no keyword line is found. When the sp result was taken from the opt log, `sp_route` is the opt job's line.

Gaussian reports the first route section of the file, so a `--Link1--` file yields its first job's route; a log's wrapped route lines are rejoined as written and a deck's lines are joined by single spaces. Orca reports the `!` keyword lines only, lowercased, without their `!` and any `#` comment, joined by single spaces; `%` blocks are not included.

| Field | Type | Description |
|---|---|---|
| `opt_route` | `str?` | Keyword line of the opt job |
| `freq_route` | `str?` | The same for the freq job |
| `sp_route` | `str?` | The same for the sp job |
| `composite_route` | `str?` | The same for the composite-method job |
| `composite_step_routes` | `list[str]?` | The route of every internal step of the composite-method job, in the order the composite log echoes them. Gaussian prints one route section per `Link1` step, so a CBS-QB3 log lists one route per step, and `composite_route` is the first entry. Read from the Gaussian log only (never from the input deck or from the composite method name), with the joining rules of `opt_route`; only a route that directly follows a rule of dashes counts, so a `#` line wrapped out of an archive block is not one. `null` for a species that did not converge, with no composite log, for a program other than Gaussian (ORCA included, since its log does not state composite steps), and when the log echoes no route. No geometry level is stated or deduced from it |
| `irc_log_routes` | `list[str?]` | Transition states only: the keyword line of each IRC job, in lockstep with `irc_logs`; `null` for a log that is missing, from another program, or without a keyword line |

### Parsed Molecular Properties

`sp_t1_diagnostic` is whatever ARC's parser reads from the exported sp log, the T1 diagnostic that log prints for its correlated method; the dipole moment and polarizability are read from the exported Gaussian logs and are `null` for any other program. All are `null` for a species that did not converge and when the log prints no value. No rotational constants are exported: ARC has no parser for them (Arkane computes them from the geometry and ARC reads only its symmetry numbers from its output).

| Field | Type | Description |
|---|---|---|
| `sp_t1_diagnostic` | `float?` | The T1 diagnostic (unitless) that the exported sp log prints for its correlated method (for example coupled cluster, or the coupled-pair methods of ORCA, which also print one), parsed from that log at export time by ARC's parser for its program. No method name is matched. `null` when there is no sp log, when it prints none, when it cannot be parsed, and for a species that did not converge |
| `opt_dipole_moment_debye` | `float?` | The total dipole moment in **Debye** of the last `Dipole moment (field-independent basis, Debye)` block of the Gaussian opt log, at the final opt geometry. `null` for a charged species (the dipole of an ion is origin-dependent), for a monoatomic species and when the log prints no block |
| `opt_dipole_moment_density` | `str?` | The density that dipole was computed from, as the word of the Gaussian header `Population analysis using the <X> density` preceding the block (`SCF`, `current`, ...). `null` when the header is absent or the dipole is `null` |
| `freq_polarizability_angstrom3` | `float?` | The isotropic static polarizability in **angstrom^3** at the freq level: a third of the trace of the last `Exact polarizability` line (atomic units, bohr^3) of the Gaussian freq log, converted with ARC's `bohr_to_angstrom`. `null` also for a monoatomic species (no freq job, so not computed) |

### Levels

| Field | Type | Description |
|---|---|---|
| `levels` | `dict` | The level of theory (requested, not observed) of the job whose log the record exports: `{opt, freq, sp, composite, irc}`. Present on every species and TS record with all five keys; each value is a level dict or `null`. `opt`, `freq` and `sp` are the levels of the jobs that wrote `opt_log`, `freq_log` and `sp_log` (where the single-point energy is read from the optimization log, `sp` repeats `opt`; for a composite method `sp` repeats `composite`, since the energy is read from the composite log), `composite` is the composite job's level, and `irc` is the level of a TS's IRC jobs. Each is captured from the job when the scheduler recorded its log path, so it is the level that job actually ran at under `adaptive_levels`, a reaction-wide override or troubleshooting, and never a recomputation from the run settings. `null` means the level was not recorded: no such job ran, or the run or restart predates level recording. `irc` is `null` on every record that is not a TS, and on a TS whose forward and reverse IRC jobs ran at different levels. A level dict here carries no `software`, which is only a deduction; `ess_software` states the program of the opt, freq and sp logs, and the programs of composite, IRC and conformer jobs are not stated. When the single point is not requested, `sp_energy_hartree` is read from the optimization log and pairs with `levels.opt`; `sp_log` and `levels.sp` are then `null`. For a composite job the log also holds the frequencies of the composite method's own geometry level, which the composite level does not name, so `freq` is `null` and a `freq_log` that is the composite log pairs with `levels.composite`. A monoatomic species has no opt job, so its `opt` is `null` although its single-point log is also its `opt_log`. The scheme level of a `solvation_scheme_level` is not recorded. |
| `irc_endpoint_of` | `str?` | **Species records only** (absent on transition states). The label of the TS whose IRC produced this species, for an IRC endpoint species (labeled `IRC_<ts label>_<n>`, optimized from an end of the IRC path); `null` for every ordinary species. An endpoint is the geometry the IRC reached, not a well ARC was asked about, so a consumer that treats species records as wells should skip a record where this is not `null`. The TS record lists no endpoints; its own `irc_logs` are the IRC logs. |
| `irc_endpoint_direction` | `str?` | **Species records only.** `"forward"` or `"reverse"`: the direction of the IRC job whose log produced this endpoint. `null` for an ordinary species and for an endpoint whose direction was not recorded. The `<n>` in the label is the order in which the IRC jobs finished, not the direction. `forward` and `reverse` name the sign of the IRC job's transition vector in the ESS and carry no reactant or product meaning. |

### Held-Fixed Constraints

`opt_constraints`, `freq_constraints` and `sp_constraints` each hold the
coordinates that were frozen for that calculation, parsed from the ESS input
deck or log. Each is `[]` when the deck records none or is not on disk. The
scanned coordinate of a rotor scan is *not* a constraint and never appears
here; a rotor scan's own frozen coordinates are under `rotor_scans[].constraints`.

| Field | Type | Description |
|---|---|---|
| `coordinate_type` | `str` | `"cartesian"`, `"distance"`, `"angle"` or `"dihedral"` |
| `atom_indices` | `list[int]` | The atoms defining the coordinate, in the source deck's own numbering |
| `index_base` | `int` | The numbering convention of `atom_indices`: `1` for Gaussian, `0` for Orca. ARC does **not** renumber them |
| `target_value` | `float?` | The value the coordinate was held at, exactly as the deck states it and never converted. `null` when the deck froze the coordinate without naming a value |
| `target_value_units` | `str?` | The unit `target_value` is in, never converted. `"degree"` for `angle` and `dihedral` on every ESS. For `distance` and `cartesian` it is `"angstrom"` for both a Gaussian ModRedundant coordinate and an Orca `%geom Constraints` entry — the unit Orca reads by default and the one ARC writes. Orca does **not** echo the unit its block was written in, so a hand-written Orca deck using Bohr carries a value this field does not describe |

### Frequency Hessian Method

| Field | Type | Description |
|---|---|---|
| `freq_hessian_method` | `str?` | `"analytic"`, `"finite_difference_gradient"`, `"finite_difference_energy"`, or `null` when ARC cannot state it |

`freq_hessian_method` records how the frequency job's Hessian was built, in TCKDB's
closed `HessianMethod` vocabulary. It is the provenance of the method **ARC
requested**, resolved through the documented behaviour of the ESS that ran it: ARC
states it only where the request settles the answer, and emits `null` everywhere
else rather than inferring one from weaker evidence.

The freq input deck is read in preference to the log, the deck being ARC's request
rather than an echo of it; the log is the fallback when the deck is not on disk. The
field is `null` when the species did not converge and when neither the deck nor the
log is on disk.

| ESS | Signal | Result |
|---|---|---|
| `gaussian` | route's `freq` carries `numer`/`numerical` | `finite_difference_gradient` |
| `gaussian` | route's `freq` carries `enonly` | `finite_difference_energy` |
| `gaussian` | bare `freq`, method `hf`/`rhf`/`uhf`, `mp2`, `cis`, `casscf` | `analytic` |
| `gaussian` | bare `freq`, R/U DFT functional (`b2plyp`, `b2plypd3`, `mpw2plyp` included) | `analytic` |
| `gaussian` | bare `freq`, method `mp3`, `mp4(sdq)`, `mp4(dq)`, `ccd`, `ccsd`, `cid`, `cisd`, `qcisd`, `bd` | `finite_difference_gradient` |
| `gaussian` | bare `freq`, method `rohf` | `finite_difference_gradient` |
| `gaussian` | bare `freq`, method `mp4` (i.e. MP4(SDTQ)), `mp5`, `ccsd(t)`, `qcisd(t)`, `qcisd(tq)`, `bd(t)`, `bd(tq)` | `finite_difference_energy` |
| `gaussian` | bare `freq`, `ro`-prefixed MP2/MP3/MP4 or coupled cluster | `finite_difference_energy` |
| `gaussian` | bare `freq`, `ro`-prefixed DFT, a double hybrid Gaussian does not document (see below), an option-bearing correlated spelling such as `mp4(full)`, any unlisted method, or a composite / semiempirical / force-field level | `null` |
| `orca` | keyword line carries `NumFreq` | `finite_difference_gradient` |
| `orca` | keyword line carries `NumFreq` **and** `NumGrad` | `finite_difference_energy` |
| `orca` | `AnFreq` or bare `Freq`, method HF or a non-double-hybrid DFT functional, no `RI-JK` | `analytic` |
| `orca` | `AnFreq` or bare `Freq`, any other method (MP2, DLPNO, coupled cluster, a double hybrid) or an `RI-JK` keyword line | `null` |
| `orca` | no keyword line readable, log carries an `ORCA SCF HESSIAN` header | `analytic` |
| `orca` | no keyword line readable, no such header | `null` |
| `pyscf` | multiplicity 1 | `analytic` |
| `pyscf` | multiplicity > 1 | `finite_difference_gradient` |
| any other ESS | — | `null` |

The Gaussian route is read free-format, as Gaussian reads it: `Freq=Numer`,
`Freq = Numer`, `Freq=(NoRaman, Numer)` and `Freq (Numer)` are one keyword with one
option list. A keyword the `freq` letters merely appear inside — `freqchk`,
`anharmonicfreq`, `opt=calcnumerfc` — is not that keyword, and a parenthesised group
belonging to a *following* keyword (`freq scf=(numer)`, `freq IOp(7/33=1)`) is not
that option list.

Gaussian never prints its analytic default, so a bare `freq` route is resolved from
the method name, matched as a whole token with a leading `u`/`r` spin prefix
stripped. `RO` is handled apart from `R`/`U`: Gaussian computes numerical
frequencies for ROHF, is energies-only for RO-MP*/RO-CC, and documents neither for
RO-DFT, which is therefore `null`. `mp4` alone means MP4(SDTQ).

A method name outside those tables is checked against a correlated-method pattern
(`mp3`-`mp9`, `cc`, `qci`, `ci`, `bd`, `cas` and `hf`, optionally spin-prefixed)
before the DFT rule is applied, and yields `null` when it matches. Gaussian's option
syntax admits many spellings the tables do not carry — `mp4(full)`, `MP4(SDTQ,Full)`,
`mp4sdq(full)`, `mp5(full)` — and `Level.deduce_method_type` types every one of them
as DFT, since its wavefunction list stops at `mp3`. Without the pattern those
spellings would be reported `analytic`, which is wrong for every one of them; with it
they are `null`. The pattern is deliberately anchored so that a DFT functional whose
name opens with the same letters keeps its analytic answer: `hfs` and `hfb` (Gaussian's
Slater and Becke88 exchange functionals), `hf3c`, `mpw3lyp`, `mpw2plyp` and `b2plyp`
are all left to the DFT rule.

The double hybrids Gaussian does not document an analytic Hessian for are excluded
from the DFT rule by **substring** match, matching how the Orca branch excludes the
same family: a name carrying `dsd`, `pwpb95`, `qidh`, `pbe0dh`, `pbe0-dh`, `b2gpplyp`
or `wb97x-2` is `null`. This covers spelling variants — `dsd-pbep86`, `revdsdpbep86`,
`dsd-blyp`, `pbe-qidh` — that an exact-match list missed. `b2plyp`, `b2plypd3` and
`mpw2plyp`, which Gaussian *does* implement with analytic second derivatives, are not
matched by any of those substrings and stay `analytic`.

A **composite** level yields `null`: the internal frequency step of a composite
method is the method's own choice, not ARC's request. Gaussian *log* text is not
consulted — the `Differentiating once with respect to nuclear coordinates` line is
printed by the analytic CPHF link and appears in every Gaussian frequency fixture,
so it distinguishes nothing.

The whole Gaussian branch keys on `Level.method_type`, so a method that ARC's type
deduction misclassifies yields no value: `cam-b3lyp` contains `am` and is typed
semiempirical, so it is reported `null` rather than `analytic`. That is the
conservative direction — a missing value, never a wrong one — but it is a limitation
of the type deduction rather than a statement about the functional.

On the Orca side the keyword line is read from the head of the file: a log's echoed
input block ends at `****END OF INPUT****` and the scan stops there, and in any case
after a few thousand lines, so a multi-hundred-megabyte log is not read through. When
no keyword line can be read at all — an archived log truncated to its tail — an
`ORCA SCF HESSIAN` header found within the first 50,000 lines is taken as analytic.
No numerical header is relied on for the reverse direction.

Orca documents `Freq` as an alias of `AnFreq`, which is why a bare `Freq` — what ARC
writes unless `use_num_freq` is set — is analytic, but only for the levels Orca
documents an analytic Hessian for. `NumFreq` differentiates gradients, or energies
when the gradients are themselves numerical (`NumGrad`). When no keyword line can be
read at all, an `ORCA SCF HESSIAN` header in the log is accepted as analytic; no
numerical header is relied on, none having been observed in an ARC fixture.

The PySCF rule is not re-derived here: `run_freq` in
`arc/job/adapters/scripts/pyscf_script.py` branches on the spin multiplicity alone —
multiplicity 1 builds an analytic RKS Hessian, and anything higher a central finite
difference of analytic gradients.

### Spin Diagnostic

`sp_spin_diagnostic` reports the S² spin contamination of the single-point
wavefunction. It is parsed from the sp log, falling back to the freq log and then
the opt/geo log (ARC may reuse the optimization output for the sp energy). It is
`null` when the species did not converge, and for restricted/closed-shell
calculations, whose logs print no `<S**2>` at all.

| Field | Type | Description |
|---|---|---|
| `s_squared` | `float` | The `<S**2>` value the ESS reported. Always present when the block is non-`null` |
| `s_squared_expected` | `float?` | The exact `<S**2>` for a pure spin state, recomputed from ARC's own `multiplicity` and falling back to the value in the log. **Omitted** when neither source yields a value |
| `s_squared_annihilated` | `float?` | The `<S**2>` after spin annihilation, when the ESS reports one. **Omitted** otherwise |
| `log` | `str` | Run-relative path of the ESS log the value was read from |

### Wavefunction Stability and the SCF Reference

`wavefunction_stability` carries the verdict of a wavefunction stability analysis,
and is `null` when no analysis was run, when its log is missing, or when the log
holds no verdict. `verdict` is one of `stable`, `internal_instability`,
`external_instability`, `unattributed_instability` or `unknown`.

`log` names the analysis log. Every other field of an ESS that follows an
instability rather than only reporting it describes one of two wavefunctions:
`verdict`, `lowest_eigenvalue`, `negative_eigenvectors` and `restricted` belong to
the wavefunction under test, while the energies and spin expectation values that
log also holds belong to the followed solution the ESS relaxed into.

| Field | Type | Description |
|---|---|---|
| `verdict` | `str` | The analysis verdict |
| `internal_instability` | `bool?` | Whether a spin-conserving instability was found |
| `external_instability` | `bool?` | Whether a spin-symmetry-breaking instability was found. `null` when the sector was not tested |
| `relaxations` | `[str]` | The relaxed constraint each external verdict names, e.g. `RHF -> UHF` |
| `negative_eigenvectors` | `[{label, eigenvalue}]` | The negative roots of the stability matrix |
| `lowest_eigenvalue` | `float?` | The smallest reported eigenvalue, negative or not |
| `restricted` | `bool?` | The reference that was tested |
| `invalidates_analytic_freq` | `bool?` | Whether the verdict puts the analytic frequencies outside their range of validity |
| `n_analyses` | `int?` | Number of analyses the log holds (ORCA) |
| `followed_to_stable` | `bool?` | Whether the ESS relaxed into a stable solution (ORCA) |
| `s_squared_after_follow` | `float?` | `<S**2>` of the followed solution (ORCA) |
| `measured_on_ts_guess` | `int?` | Index of the abandoned TS guess a carried-over verdict was measured on |
| `log` | `str?` | Run-relative path of the stability analysis log |

`scf_reference` reports which source decided the species' open-shell character and
which SCF references the sp and freq jobs actually ran with. It is always present.

| Field | Type | Description |
|---|---|---|
| `source` | `str?` | `declared` when the user declared more than one radical, `derived` when a measured stability verdict ARC acts on called for an unrestricted reference, `null` when the multiplicity alone decided |
| `declared_number_of_radicals` | `int?` | The user's declaration, when there is one |
| `verdict` | `str?` | The stability verdict the reference was derived from |
| `verdict_restricted` | `bool?` | The reference that verdict was measured on |
| `measured_on_ts_guess` | `int?` | Index of the abandoned TS guess the verdict was measured on |
| `sp_reference` | `str?` | The SCF reference the sp job ran with |
| `freq_reference` | `str?` | The SCF reference the freq job ran with |
| `reference_mismatch` | `bool?` | Whether the two differ. `null` when either is unknown |
| `log` | `str?` | Run-relative path of the log the verdict came from |

### Final Calculation Settings

These carry the calc-specific scientific knobs that defined the final job, as
distinct from level-of-theory identity and from scheduler/operational fields.
Today ARC can prove exactly one such setting from run state — which stage of the
two-stage optimization convention a calc represents — so the dicts hold only
`optimization_stage`, and the fields are `null` rather than fabricated whenever
that signal is absent.

| Field | Type | Description |
|---|---|---|
| `opt_final_settings` | `dict?` | `{'optimization_stage': 'fine'}` when a coarse stage ran and parsed; `null` for a single-stage optimization |
| `coarse_opt_final_settings` | `dict?` | `{'optimization_stage': 'coarse'}` under the same condition; `null` otherwise |
| `freq_final_settings` | `null` | Always `null`; ARC has no equally reliable signal for freq jobs yet |
| `sp_final_settings` | `null` | Always `null`; ARC has no equally reliable signal for sp jobs yet |

### Energy Corrections

`energy_corrections` is always present and is always a list — possibly empty when no
corrections were applied or the correction helper produced no usable rows. Atom-energy
and bond-additivity records are built independently, so either may be absent from the
list on its own.

Every record here is a correction ARC **applied** to this species, never one it merely
configured. A `bond_additivity` record therefore appears only where the thermo statmech
run — the only Arkane run that applies a BAC — actually covered the species: never on a
run that computes rates without thermo, never on a transition state, and never on a
species whose own `compute_thermo` is off. Read the run-level `bac_type` as the request
and this list as the outcome.

The records are recomputed after the run from the Arkane key matched for
`arkane_level_of_theory`, not read back from the run, so ARC keeps a record only where the
run behind the exported energy is recorded as having applied it: an `atom_energy` record
appears only when the switch (`thermo.atom_corrections_applied`, or `e0_atom_corrections_applied`
for a transition state or an E0-only species) is `true`, and a `bond_additivity` record only
when `bond_corrections_applied` (or `e0_bond_corrections_applied`) is `true`. A `false` switch
and a `null` one alike mean no correction row is stated: a `null` switch (no thermo, a run not
stamped by Arkane, an older restart) is not read as "applied".

The converse does not hold: a missing `atom_energy` record does not show that Arkane ran
without atom energy corrections. The list is recomputed after the run from the matched
Arkane key, so it is empty when the corrections came from ARC's own `data/AEC.yml` rather
than Arkane's database, and a failed recomputation drops its rows. Read
`thermo.atom_corrections_applied` for whether the thermo carries the correction.

| Field | Type | Description |
|---|---|---|
| `correction_type` | `str` | `"atom_energy"` or `"bond_additivity"` |
| `model` | `str` | `"arkane_atom_energy"`, or `"petersson"` / `"melius"` for BAC |
| `level_of_theory` | `dict?` | The ARC level of theory (a level dict) whose energies the correction was applied to — ARC's own `arkane_level_of_theory`, **not** an Arkane database key; `null` when unknown |
| `matched_arkane_key` | `str?` | The Arkane database key string (e.g. `"LevelOfTheory(method='wb97xd',basis='def2tzvp',software='gaussian')"`) that ARC matched in the section this correction's parameters live in: the atom-energy section for `atom_energy`, the Petersson or Melius bond-additivity section for `bond_additivity`. The two sections are searched independently, so the two records for one species may carry different keys — a key's method can also carry a refit vintage the run's own level does not (`b3lyp2023` for a job run at `b3lyp`). The atom-energy key is additionally the model chemistry ARC hands Arkane, so it is the key the totals in **both** records were computed under. `null` when nothing matched in that record's section |
| `total` | `dict` | `{value: float, unit: str}` — the applied correction total. `unit` is a closed vocabulary tied to `correction_type`: `hartree` for `atom_energy`, `kcal_mol` for `bond_additivity`. Same for `components[].parameter_unit`, by `component_kind` |
| `components` | `list` | Native per-atom / per-bond decomposition. Always present; `[]` when no decomposition is available. For a Petersson BAC these are the bonds Arkane applied, which sum to `total`: Arkane skips a bond type that has no parameter in the matched table and applies the rest (`arkane/encorr/bac.py`, `_get_petersson_correction`) |
| `skipped_components` | `list?` | The bonds of a Petersson BAC that have no parameter in the matched table, as `{bond: str, count: int}`; they contributed nothing to `total`. `[]` when every bond has a parameter. `null` where it does not apply: an atom-energy record and a Melius record. Only a `petersson` record can list bonds |

Each **`components[]`** entry (from `arc/scripts/get_species_corrections.py`) is:

| Field | Type | Description |
|---|---|---|
| `component_kind` | `str` | `"atom"` for an AEC component, `"bond"` for a BAC component |
| `key` | `str` | The element symbol (AEC) or bond descriptor such as `"C-H"` (BAC) |
| `multiplicity` | `int` | How many times this atom or bond occurs in the species |
| `parameter_value` | `float?` | The raw table parameter for this key; `null` when the key is not parameterized |
| `parameter_unit` | `str` | The unit of `parameter_value` |
| `contribution_value` | `float?` | This component's signed contribution to `total`, in `total`'s unit; `null` when not computable. **This is the quantity that sums to `total`** — not `multiplicity * parameter_value`, which for an AEC differs from it in both sign and magnitude |
| `reference_atom_energies` | `dict?` | **AEC only.** `{unit: "hartree", applied_as: "subtracted", values: {element: float, ...}}` — Arkane's bare atomic electronic energies for this level. Reconstructing the correction as `sum(count * value)` is wrong in both sign and magnitude; use `components[].contribution_value`. **Omitted** when the run has no such table |
| `parameter_table` | `dict?` | **BAC only.** `{unit: "kcal_mol", values: {bond: float, ...}}` — the per-bond BAC parameters ARC actually used. **Omitted** when the run has no such table, unless `bac_type == 'p'` (so a Melius run never carries one), and unless the key matched in the BAC section is also the atom-energy key the totals were computed under |

### Rotor Scans

`rotor_scans` is always a list, `[]` for monoatomic or non-converged species. It holds
one record per successful 1D rotor whose scan log parsed cleanly; rotors that fail any
of those conditions — unsuccessful, multi-dimensional, or unparseable — are skipped
entirely, so the list can be shorter than `statmech.torsions` and the matching
torsion's `source_scan_key` is then `null`. `constraints` (held-fixed coordinates,
excluding the scanned coordinate itself) is **omitted** when none were found, and
`result.zero_energy_reference_hartree` is **omitted** when the parser reports none.

Each record also states `ess_software` (ARC's lowercase ESS name) and `ess_version` (the full banner), both identified from the scan log itself and `null` when the log is missing or states none.

**`result`** itself carries:

| Field | Type | Description |
|---|---|---|
| `dimension` | `int` | Always `1` — only 1D rotors are emitted here |
| `relaxed` | `bool?` | Whether the remaining degrees of freedom were optimized at each scan point. `true` for an ESS-native scan and for any `*_opt` directed scan, `false` for the rigid `brute_force_sp` family, and `null` when the scan type is not recognized. Not a constant: asserting `true` unconditionally would misdescribe every rigid scan |
| `zero_energy_reference_hartree` | `float?` | The absolute energy the relative curve is zeroed against. **Omitted** when the parser reports none |
| `coordinate` | `dict` | The scanned coordinate and requested grid, below |
| `samples` | `list` | One entry per parsed scan point |

**`result.coordinate`** describes the scanned coordinate and the requested grid:

| Field | Type | Description |
|---|---|---|
| `coordinate_type` | `str` | Always `"dihedral"` |
| `atom_indices` | `list[int]` | The 4 atoms defining the dihedral, in source order |
| `index_base` | `int` | Always `1` — the atom indices are 1-based |
| `unit` | `str` | Always `"degree"` |
| `sample_count` | `int` | Number of parsed scan points, matching `len(samples)` |
| `symmetry_number` | `int` | The rotor's symmetry number. **Omitted** unless ARC determined an integer symmetry of at least 1 |
| `requested_step_size` | `float?` | The step size the user requested, in degrees, read back from the ESS log rather than inferred from point spacing. **Signed**: negative for a scan that sweeps toward decreasing dihedral values, so the direction of a reversed scan is recorded rather than lost. Never `0` — a zero step is 'unknown', not a literal zero-degree grid. Gaussian only — other ESS raise `NotImplementedError` from the same parser, so this is **omitted** for them |
| `requested_start` | `float?` | The requested starting dihedral in degrees, taken from the geometry the scan was launched against. Grid metadata describing the extent that was *asked for* — **not** an anchor to add to a sample's `angle_degrees`, which is already absolute. **Omitted** without a `requested_step_size` or a computable dihedral |
| `requested_end` | `float?` | `requested_start + requested_step_size * steps`, where `steps` is the step count **requested** in the scan's ModRedundant header — not the number of completed points, so a truncated scan still reports the span that was asked for. Falls back to `sample_count - 1` only when the requested count is unavailable. Because `requested_step_size` is signed, a reversed scan ends **below** `requested_start`. Deliberately not wrapped into `[-180, 180]`, so a full rotation ends at `start ± 360` rather than back at `start`. **Omitted** under the same condition as `requested_start` |

#### `result.samples[]`

One entry per parsed scan point, in the order the ESS reported them.

| Field | Type | Description |
|-------|------|-------------|
| `source_index` | `int` | The point's 0-based position in the parsed scan, so a sample can be traced back to the log |
| `angle_degrees` | `float` | **The absolute dihedral, not a displacement.** The value of the scanned internal coordinate at this point, in degrees, measured on *this point's own geometry* at `coordinate.atom_indices` — so it agrees by construction with what a consumer recomputes from the `geometry_xyz` published beside it, and nothing has to be added back to it. The sequence is unwrapped and never folded into a `0-360` window: a full rotation that starts at `59.867` ends at `419.867` (the same angle modulo 360) so the curve stays monotone across the closing step, and a scan that sweeps toward decreasing values goes below its start and may go negative. A scan whose points do not all yield a measurable dihedral is **not published at all** — there is no absolute coordinate for it and a displacement is not a substitute |
| `relative_energy_kj_mol` | `float` | Electronic energy in kJ/mol relative to the lowest point of *this* scan, so the largest value is the torsional barrier. The zero is `result.zero_energy_reference_hartree` when that field is present |
| `electronic_energy_hartree` | `float` | The point's absolute electronic energy in Hartree. **Omitted** (key absent, never `null`) when the ESS adapter has no Hartree-preserving parse or its list did not align 1:1 with the energies |
| `geometry_xyz` | `str` | The point's geometry as an XYZ block. **Omitted** (key absent, never `null`) when per-point geometries were unavailable or did not align 1:1 with the energies; coverage is all-or-nothing, never partial |
| `geometry_isotopes` | `list[int]?` | The isotopes of `geometry_xyz`, taken only from a scan log that states the masses it used (Gaussian, as above); `null` for every other program and for a log that states none. Emitted together with `geometry_xyz` and omitted with it |

### Thermochemistry

`thermo` is `null` for non-converged species or species without thermo data.

| Field | Type | Description |
|---|---|---|
| `h298_kj_mol` | `float` | Standard enthalpy at 298 K (kJ/mol) |
| `s298_j_mol_k` | `float?` | Standard entropy at 298 K (J/(mol K)) |
| `tmin_k` | `float?` | Minimum temperature (K) |
| `tmax_k` | `float?` | Maximum temperature (K) |
| `standard_state_pressure_pa` | `float?` | Standard-state pressure (Pa) that `s298_j_mol_k`, the NASA polynomials and every `thermo_points` entropy and free energy belong to. Recovered by `arc/scripts/save_arkane_thermo.py` from RMG's translational partition function — the one place the standard state enters a statmech result — rather than restated as a literal. Arkane runs at 1 atm (101325 Pa) and exposes no way to change it (`IdealGasTranslation.get_partition_function` in RMG-Py's `rmgpy/statmech/translation.pyx` divides by the literal `101325.`, and `get_entropy` is built on it), so every record of a run carries the same value; a consumer that assumes 1 bar is wrong by about 0.11 J/(mol K) in S. **Per path:** (1) thermo of an `ArkaneAdapter` run whose `thermo.yaml` was written by `save_arkane_thermo.py` carries the value that script measured from the partition function; (2) thermo that `ArkaneAdapter` parsed from the run's `output.py` because `thermo.yaml` is missing or carries no pressure records `101325.0`, the constant the code above applies, since the run is the same Arkane statmech run; (3) `null` for a thermo no Arkane run of ARC produced, which no ARC path does, and for every `output.yml` written before this field existed. Nothing else defaults it |
| `atom_corrections_applied` | `bool?` | Whether the Arkane run that produced this thermo had its atom energy correction switch (`useAtomCorrections`) on. This is the switch ARC wrote into the Arkane input, and nothing more. ARC turns it off when neither Arkane's database nor ARC's `data/AEC.yml` has atom energies for `arkane_level_of_theory` (matched with its dispersion correction folded into the method, whether carried in the method string or the separate `dispersion` field; a level with a `solvation_method` matches none, since Arkane has no corrections for solvated levels); Arkane then subtracts no atom energies, so `h298_kj_mol`, the NASA polynomials and every `thermo_points` enthalpy and free energy are absolute electronic-structure energies (roughly -1e5 kJ/mol per heavy atom), **not** formation enthalpies. `true` means only that Arkane subtracted the atom energies of `atom_corrections_level` (it stops with an error rather than produce thermo when it cannot find them); it does **not** mean `h298_kj_mol` is a formation enthalpy. That also needs `atom_corrections_level` to be the level the species' energies were computed at, which ARC does not require: a dummy `arkane_level_of_theory` (as in `examples/Stationary/bde`) is applied to energies from another level, and ARC only logs a warning. `null` when the thermo did not come from an Arkane run that recorded it: every `output.yml` written before this field existed, and a species Arkane loaded from its own YAML file (`yml_path`), which Arkane takes as-is without applying any correction |
| `bond_corrections_applied` | `bool?` | The `useBondCorrections` value of the same Arkane run. It is `true` only when the run requested a `bac_type` and atom energy corrections were on, so it is always `false` when `atom_corrections_applied` is `false`. `true` means the BAC model was applied, not that every bond had a parameter: Arkane's Petersson model skips, with a warning, a bond type its table lacks. `null` exactly when `atom_corrections_applied` is `null`. `true` implies the header `bac_type` is `"p"` or `"m"` (the schema enforces it, and likewise for `statmech.e0_bond_corrections_applied: true` of a species or TS record, which is the same adapter switch): the flag is the adapter's `bac_type is not None and use_aec`, and the adapter receives the same `bac_type` the header exports |
| `atom_corrections_level` | `dict?` | The level whose atom energies Arkane subtracted: the level ARC matched Arkane's model chemistry for, or whose `data/AEC.yml` entry it passed as `atomEnergies` (always `arkane_level_of_theory`). Same shape as `sp_level` and `composite_method`. **Before treating `h298_kj_mol` or the NASA polynomials as formation enthalpies, compare this with the level the species' energies were computed at** (`composite_method` if set, else `sp_level`): if they differ, the enthalpies mix two levels and are neither formation enthalpies nor absolute energies. Compare method (with dispersion folded in) and basis after normalizing them as ARC does when matching Arkane's database: ignore case, hyphens and spaces, and strip a trailing four-digit refit year from the method (`wB97X-D/def2-TZVP` is `wb97xd/def2tzvp`); a level may carry its dispersion correction in the method string or in a separate `dispersion` field, and `b3lyp` with `dispersion: gd3bj`, `b3lyp-d3bj` and `b3lyp-d3(bj)` are one method, while `wb97xd` and `wb97xd3` are two. ARC's Arkane correction matching folds the dispersion correction into the method in the same way, so the atom energies applied carry the dispersion correction of `atom_corrections_level`. Arkane has no corrections for solvated levels, so a level with a `solvation_method` is never `atom_corrections_level`: when the energy level has a `solvation_method` and this field is not `null`, it is a gas-phase `arkane_level_of_theory` set explicitly, its gas-phase atom energies were subtracted from solvated energies, and `h298_kj_mol` and the NASA polynomials should not be treated as formation enthalpies (`formation_298k`). Under `adaptive_levels` each species' energy level is chosen by its heavy-atom count while Arkane gets the one `arkane_level_of_theory`, so compare with the record's `levels.composite`, else `levels.sp`, rather than with the header `sp_level`. When not `null`, the value always repeats `arkane_level_of_theory`; what this field adds is where it is `null`, marking the species (an Arkane YAML species, or `false` thermo) whose atom energies were not subtracted. The Arkane database key that level matched, which may carry a refit year, is `energy_corrections[].matched_arkane_key`. `null` unless `atom_corrections_applied` is `true`. When the sp method is DLPNO, a hydrogen, deuterium or tritium atom's single point runs at an HF fallback level (`levels.sp`), so it differs from `arkane_level_of_theory` and is not a mismatch; an `adaptive_levels` sp entry replaces that fallback |
| `thermo_points` | `list?` | Tabulated per-temperature thermochemistry (see below) |
| `nasa_low` | `dict?` | Low-temperature NASA polynomial |
| `nasa_high` | `dict?` | High-temperature NASA polynomial |

**`thermo_points`** entries (one per evaluation temperature; `temperature_k` is required, all others are optional but emitted by default when produced via `arc/scripts/save_arkane_thermo.py`):

| Field | Type | Description |
|---|---|---|
| `temperature_k` | `float` | Temperature (K) |
| `cp_j_mol_k` | `float?` | Heat capacity at constant pressure (J/(mol K)) |
| `h_kj_mol`    | `float?` | Enthalpy at this temperature (kJ/mol) |
| `s_j_mol_k`   | `float?` | Entropy at this temperature (J/(mol K)) |
| `g_kj_mol`    | `float?` | Gibbs free energy at this temperature (kJ/mol) |

**`nasa_low` / `nasa_high`**:

| Field | Type | Description |
|---|---|---|
| `tmin_k` | `float` | Polynomial validity range minimum (K) |
| `tmax_k` | `float` | Polynomial validity range maximum (K) |
| `coeffs` | `list[float]` | 7 NASA polynomial coefficients |

### Statistical Mechanics

`statmech` is `null` for monoatomic or non-converged species.

| Field | Type | Description |
|---|---|---|
| `e0_kj_mol` | `float?` | Arkane's E0 (kJ/mol), the electronic energy plus scaled ZPE. With atom corrections on, Arkane also adds `atom_hf - atom_thermal` for every atom, so E0 is the 0 K enthalpy of formation minus `sum(n_i * atom_thermal_i)` and **not** the 0 K formation enthalpy (for CH4 it is lower by about 18.0 kJ/mol); a TS's E0 is never a formation enthalpy. The switches below say which corrections the run applied |
| `e0_atom_corrections_applied` | `bool?` | Whether the Arkane run that wrote `e0_kj_mol` had its atom energy correction switched on (`useAtomCorrections`). When `false`, `e0_kj_mol` is an absolute electronic energy plus ZPE and not a formation enthalpy. `e0_kj_mol` is written by three different runs: the thermo run and the E0-only run of a species (bond correction on when the run has a `bac_type`), the kinetics run that writes a TS's E0 (no bond correction), and the TS-check E0-only run (no bond correction), whose E0 and switches `copy_e0_values` copies to the reaction. As currently computed, a well's E0 can therefore carry a BAC that its TS's does not. `null` when the run's switches are not known (an E0 restored from a restart that predates them, or supplied by the user) and for a species loaded from an Arkane YAML, whose energy Arkane takes as-is |
| `e0_bond_corrections_applied` | `bool?` | Whether that same Arkane run had its bond additivity correction switched on (`useBondCorrections`); `null` under the same conditions |
| `arkane_rotors_applied` | `int?` | The number of rotor modes (`HinderedRotor`, `FreeRotor`, `HinderedRotor2D`, `HinderedRotorClassicalND`, and `Mode`, which is how Arkane writes a multi-dimensional rotor) in the `conformer(modes=[...])` block of the `output.py` of the last Arkane run that wrote this species' conformer block, parsed before the statmech directory is reused. It counts modes, not torsional degrees of freedom. The kinetics run overwrites the modes of the wells it declares, not their E0. Arkane drops every rotor, and treats the species as a rigid rotor and harmonic oscillator, when the frequency log has no force-constant matrix, so it is `0` where ARC found rotors, and it can be lower than `len(torsions)`. `null` when that output was not parsed |
| `arkane_treatment` | `str?` | The treatment Arkane applied, as TCKDB's `StatmechTreatmentKind`, derived from the rotor modes of that block: `"rrho"` with no rotor mode, `"rrho_1d"` when every rotor mode is a `HinderedRotor` or `FreeRotor`, `"rrho_nd"` when every one is a `HinderedRotor2D` or `HinderedRotorClassicalND`, `"rrho_1d_nd"` when both kinds are present. `null` when the output was not parsed, and when any rotor mode is of unknown kind (a `Mode` entry, which is what a multi-dimensional rotor is written as). `null` `arkane_rotors_applied` implies `null` here, `"rrho"` implies `0` rotors, and any rotor treatment implies at least `1` |
| `spin_multiplicity` | `int?` | Spin multiplicity |
| `optical_isomers` | `int?` | Number of optical isomers |
| `is_linear` | `bool?` | Whether the molecule is linear |
| `external_symmetry` | `int?` | External symmetry number |
| `point_group` | `str?` | Point group (e.g. `C2v`) |
| `rigid_rotor_kind` | `str?` | The rigid-rotor kind, read from the exported `point_group` alone; no moment of inertia is computed and no tolerance is used. `"atom"` for a single atom (never reached: `statmech` is `null` for monoatomic species). `"linear"` for C∞v and D∞h. `"spherical_top"` for the cubic groups T, Td, Th, O, Oh, I and Ih, whose rotations transform as a triply degenerate representation so the inertia tensor is isotropic. `"symmetric_top"` for any other group with a proper C_n axis of n >= 3 (C_n, C_nv, C_nh, D_n, D_nh, D_nd with n >= 3), which equates the two moments perpendicular to it, and for D2d and S_n with even n >= 4 (S4, S6, S8, ...), because an improper rotation S_n(θ) is the inversion times the proper rotation C(θ+π), and the inversion acts trivially on the inertia tensor, so S_n acts on it as the rotation by 2π/n + π, which has order 3 or more for every even n >= 4 (S4 acts as C4³, order 4; S6 as C3⁻¹, order 3), although D2d and S_n have no proper axis above C2 (allene is a prolate symmetric top). `"asymmetric_top"` for C1, Cs, Ci, C2, C2v, C2h, D2 and D2h, where nothing forces two moments to agree. Benzene (D6h) and NH3 (C3v) are symmetric tops, CH4 (Td) and SF6 (Oh) spherical tops, H2O (C2v) an asymmetric top, CO2 (D∞h) linear. The kind is symmetry-enforced: an accidental equality of moments is not stated. `null` when `point_group` is `null` or not one of the groups above, and when the species carries non-default isotopes, because the point group is that of the geometry without isotopic labels and does not apply to a labelled one. The builder has an `"atom"` branch, but `statmech` is `null` for monoatomic species, so it never reaches `output.yml`. A geometric label only: Arkane treats every non-linear species as a `NonlinearRotor` |
| `harmonic_frequencies_cm1` | `list[float]?` | Harmonic frequencies (cm-1) as parsed from the ESS, **unscaled** — `freq_scale_factor` has *not* been applied, although it has been applied to `statmech.e0_kj_mol`, so recomputing ZPE from this list will not reproduce `e0_kj_mol` unless you scale first. For TSs **every** negative (imaginary) frequency is dropped, not only the reaction mode — a TS that legitimately carries additional small imaginary modes (which `check_imaginary_frequencies` permits) loses those too, so the list can be shorter than the species' true mode count. Non-TS species are not filtered at all |
| `torsions` | `list` | Internal rotation data (see below) |
| `rejected_torsions` | `list` | Rotors ARC evaluated and rejected (see below). **Optional in the schema** — not in `statmech`'s `required` list — unlike every other field in this table. This version of the writer always emits it (`[]` when there is nothing to report, never omitted), but a consumer reading documents from other producers, or from before this key existed, must tolerate its absence rather than assuming `[]` |

**`torsions`** entries (only successful rotors):

| Field | Type | Description |
|---|---|---|
| `symmetry_number` | `int?` | Torsional symmetry number |
| `treatment` | `str?` | `"hindered_rotor"` or `"free_rotor"`, read from the mode Arkane's own output holds for this rotor. `null` when that is not known: Arkane's output was not parsed, Arkane kept a different number of rotors than this list holds (it drops all of them when the frequency log has no force-constant matrix, and then `statmech.arkane_treatment` is `"rrho"`), or the rotor's mode is a multi-dimensional one (`Mode`). ARC's own rotor record does not decide it, and it is never defaulted to `"hindered_rotor"` |
| `dimension` | `int` | Rotor dimensionality. `1` for an ordinary rotor. ND directed rotors are reported here with their real dimensionality — `rotor_scans` remains 1D-only, so an ND torsion always has `source_scan_key: null` |
| `atom_indices` | `list[int]?` | 4-atom dihedral defining atoms (1-indexed). For an ND rotor (`dimension > 1`) this is a **list of lists**, one quartet per dimension |
| `pivot_atoms` | `list[int]?` | 2-atom rotation axis (1-indexed). For an ND rotor this is a **list of pairs**, one per dimension |
| `barrier_kj_mol` | `float?` | Torsional barrier height (kJ/mol). `null` for an ND rotor (`dimension > 1`), whose barrier cannot be derived from a 1D scan parse |
| `source_scan_key` | `str?` | The `rotor_scans[].key` (e.g. `"scan_rotor_3"`) this torsion was derived from. `null` when `rotor_scans` holds no record for that rotor, so the reference can never dangle |

**`rejected_torsions`** entries (rotors ARC evaluated and rejected, i.e. `rotors_dict` entries whose `success` is `False`):

A rotor's `success` field is three-state: `None` means pending — not yet
started, mid-troubleshooting, or scanning against a previous/lower
conformer — and is *not* a rejection, so pending rotors are simply **absent**
from `rejected_torsions`, the same way they are absent from `torsions`
today; there is no separate "was this rotor attempted" signal. `False` means
ARC genuinely rejected the rotor, either after convergence invalidated it or
because it determined the coordinate isn't a torsion at all. Only
`success is False` rotors appear here.

A rejected rotor never gets a `rotor_scans` record — `rotor_scans` (see above)
holds only *successful* 1D rotors — so a rejection's evidence is recorded
directly on the entry rather than by reference, as `source_log`, whenever a
scan log for the rotor is found on disk when `output.yml` is written.

**`source_log`'s presence or absence is *not* a reliable discriminator
between rejection causes.** An earlier version of this document claimed
that presence of `source_log` meant "we scanned this rotor and rejected the
result" and absence meant "this coordinate was never treated as a torsion."
That claim is false: `source_log` is populated only when
[`_resolve_scan_path`](../arc/output.py) finds the file *actually present on
disk* at write time, and there are several reachable ways for a genuinely
scanned, genuinely rejected rotor to have no surviving log:

- A restart-restored rejected rotor's stale relative `scan_path` is only
  repaired for rotors where `success` is truthy (see `arc/main.py`), so a
  rejected rotor's path routinely fails the on-disk check even though the
  rotor was scanned.
- A failed directed scan has its `scan_path` deliberately blanked to `''`
  while still recording a real, non-empty `invalidation_reason` (see
  `Scheduler.check_directed_scan` in `arc/scheduler.py`) — the scan ran and
  failed, but the record looks identical to "never scanned."
- A TS reaction-zone pivot exclusion (see `arc/checks/ts.py`) sets
  `success = False` for a policy reason, typically before any scan is
  attempted at all — a different kind of rejection than either of the above.

ARC's `rotors_dict` does not currently distinguish these rejection stages
from a genuine "not a torsion" determination. `invalidation_reason` is the
only signal available for telling them apart, and it is free-text and
imperfect — do not rely on `source_log`'s presence or absence as a
substitute.

| Field | Type | Description |
|---|---|---|
| `rotor_index` | `int` | The rotor's key in `rotors_dict` — its identifying index among the species' rotors, not an atom index |
| `invalidation_reason` | `str` | ARC's recorded reason(s) the rotor was rejected, verbatim. An empty string means ARC recorded no specific reason — it is not fabricated into something more descriptive. This is accumulated with `+=` across troubleshooting rounds, so it may hold more than one concatenated reason rather than a single discrete one |
| `atom_indices` | `list[int]?` | 4-atom dihedral defining atoms (1-indexed). For an ND rotor (`dimension > 1`) this is a **list of lists**, one quartet per dimension. `null` when `rotors_dict` has no `scan` entry for this rotor |
| `pivot_atoms` | `list[int]?` | 2-atom rotation axis (1-indexed). For an ND rotor this is a **list of pairs**, one per dimension. `null` when `rotors_dict` has no `pivots` entry for this rotor |
| `dimension` | `int` | Rotor dimensionality, exactly as the successful torsion shape's `dimension`. `1` for an ordinary 1D rotor; higher for an ND directed rotor |
| `source_log` | `str?` | The rejected rotor's scan log path, relative to the project directory, recorded directly from `rotors_dict`'s `scan_path` rather than via a `rotor_scans` key, since this rotor has no `rotor_scans` record. A log that lies outside the project directory is not part of the export and yields no path at all (same treatment `rotor_scans[].source_log` gets), so this field is absent for it too. **Omitted** (never `null`) when `rotors_dict` records no on-disk scan for this rotor at write time. See the note above: absence does not mean the rotor was never a torsion |

---

## Transition States

`transition_states` is a list of entries that include **all species fields above**, plus:

| Field | Type | Description |
|---|---|---|
| `is_ts` | `true` | Always `true` |
| `freq_n_imag` | `int?` | The **number** of imaginary frequencies found, `null` when non-converged or monoatomic. Usually `1` for a well-behaved TS, but ARC permits additional small imaginary modes, so `2` or more is a legitimate value — do not treat this field as a converged-TS flag |
| `imag_freq_cm1` | `float?` | The most negative imaginary frequency (cm-1), i.e. the reaction mode |
| `imaginary_frequencies_cm1` | `list[float]?` | All imaginary frequencies (cm-1), including any additional small ones beyond the reaction mode |
| `chosen_ts_method` | `str?` | The TS search method that was selected |
| `successful_ts_methods` | `list[str]?` | All TS methods that succeeded |
| `ts_guesses` | `list[dict]` | Sanitized provenance for the chosen guess: `index`, `chosen`, `method`, and merged `method_sources` |
| `neb_log` | `str?` | Run-relative path to the NEB log. Taken from the run's `neb` path slot, falling back to the chosen TS guess's log when that guess's method is `orca_neb` |
| `gsm_log` | `str?` | Run-relative path to the selected GSM stringfile. Taken from the run's `gsm` path slot, falling back to the chosen TS guess's log when that guess's method is `xtb_gsm` |
| `irc_logs` | `list[str]` | Run-relative paths to IRC logs |
| `irc_log_routes` | `list[str?]` | The keyword line each IRC job ran with, in lockstep with `irc_logs` (see Effective ESS Keywords) |
| `irc_log_directions` | `list[str?]` | Which branch of the reaction path each log traversed, in lockstep with `irc_logs`. A closed vocabulary — `"forward"`, `"reverse"`, or `null` when ARC recorded no direction for that log — because this is the one field where a wrong value silently swaps reactant and product |
| `irc_log_levels` | `list[dict?]` | The level of the IRC job that produced each log, in lockstep with `irc_logs` (`job.level`, as recorded when the IRC path was stored; a level dict without `software`, since `ess_software.irc` is the observed program). `null` for a log whose job level was not recorded (an older run or restart). `levels.irc` is `null` when the two logs ran at different levels; this field keeps each log's own level |
| `freq_frequencies_cm1_ess_order` | `list[float]?` | Every frequency (cm-1) of the frequency job as printed by the ESS (ascending for Gaussian, ORCA and Q-Chem), the imaginary modes (negative) in place. `imaginary_frequencies_cm1` is re-sorted most-negative first and `statmech.harmonic_frequencies_cm1` drops the imaginary modes, so this is the only list that keeps each mode's position. `null` when the frequencies were not recorded and for a TS that did not converge. Transition states only |
| `reaction_coordinate_mode_index` | `int?` | The 1-based index, into `freq_frequencies_cm1_ess_order`, of the mode the normal mode displacement check validated as the reaction coordinate. `null` unless `ts_checks.NMD` is `true` and the check genuinely passed (a failed check that `skip_nmd` forced to pass leaves it `null`), and unless the check's record holds the position of the mode it analysed within the frequencies it parsed from an output file, the number of modes it indexed equals the length of this list, the output file it parsed is the exported `freq_log`, and the frequency at that position is imaginary. The parsed list is not always the listed one (Gaussian's displacement parser concatenates every frequency block, and a composite log yields a prefix), which is why both are checked; a record without the number of modes or the file (an older restart) gives `null`. Transition states only |
| `nmd_forced` | `bool?` | Whether `ts_checks.NMD` was forced to `true` after the normal mode displacement check failed (the `skip_nmd` option), as the check's own record (`nmd_record['forced']`) states. `true`: the check failed and ARC forced the verdict to a pass. `false`: the check ran and ARC did not force it, whatever its verdict. `null` when there is no check record (the check did not run, or an older run did not save the record) and when `ts_checks.NMD` is not a boolean. It is a top-level key rather than a member of `ts_checks` because `ts_checks` holds only verdicts. Transition states only |
| `neb_succeeded` | `bool?` | Whether the NEB log of an `orca_neb` guess ARC recorded for this TS states that the NEB optimization converged, read from the TS guesses the species holds (a guess whose `method` or merged `method_sources` include `orca_neb`) and the log path each records for `orca_neb`. `true`: the log of a recorded `orca_neb` guess prints `THE NEB OPTIMIZATION HAS CONVERGED`. `false`: `orca_neb` guesses were recorded with readable logs and none prints that line. `null` when ARC holds no `orca_neb` guess for the TS (the job did not run, the TS came from a YAML file, or no record survived), a guess records no log path, or no recorded log can be read. Whether the guess holds a geometry is not consulted: an in-core run parses a geometry from any log that exists, including one from a NEB that stopped early. It is never inferred from `neb_level`, a guess's `success` flag or the configured TS adapters. It is a separate key because `ts_guesses` lists only the chosen guess. Transition states only |
| `irc_converged` | `bool?` | **Whether the IRC jobs completed, not whether the IRC validated the TS.** It becomes `true` once both IRC directions finished, whatever their endpoints turned out to be, and is `null` when the run did not request IRC. The validation verdict is `ts_checks.IRC` |
| `ts_checks` | `dict` | ARC's own verdicts on this transition state (see below) |
| `irc_participant_mapping` | `dict?` | Which atoms of each optimized IRC endpoint geometry belong to which participant species: `{reactants: side, products: side, sides_distinguishable, atom_order_matches_ts}`, where a `side` is `{endpoint_label, participants}`. `endpoint_label` is the label of that endpoint species (`null` if not told). Each participant is `{label, position, occurrence, atom_indices}`: its species label, its 1-based position in the order of `atom_map_reactant_labels` / `atom_map_product_labels` (not of `reactant_labels`; when those are `null` because the reaction has no exported map, the position follows the `get_reactants_and_products` order: `r_species` order for the reactants and `p_species` order for the products, a repeated species expanded into one entry per occurrence), the 1-based count of that label up to it (so `2 CH3` gives occurrences 1 and 2), and the **0-based, ascending** atom indices into the optimized geometry of the endpoint species. Occurrence `k` is independent of the `k`-th block of `atom_map`, and which endpoint fragment is which occurrence of a repeated species is arbitrary. Only atom-set membership is recorded, not the atom-to-atom correspondence inside a participant. Membership is the distance-threshold connectivity of the optimized endpoint geometry, matched to the species by graph isomorphism, which ignores stereochemistry, so stereoisomers with isomorphic graphs (E/Z isomers) may have their labels swapped. `sides_distinguishable` is `false` when the reactants and the products are graph-isomorphic to each other (`CH3 + CH4 <=> CH4 + CH3`, degenerate rearrangements), in which case which endpoint is called the reactants is a convention. `atom_order_matches_ts` is `true` only when the first geometry of every IRC log is congruent with the TS geometry atom for atom, each endpoint's optimization log starts from the last geometry of its IRC log and ends at the endpoint geometry, and both endpoint geometries have the element sequence of the TS (so atom `i` of an endpoint is atom `i` of the TS, given that the ESS keeps the atom order of its input within a log); `false` when any of those is contradicted, including endpoint element sequences that differ from the TS's; and `null` when there was no TS geometry or a log or geometry needed for the check could not be read (only Gaussian logs are read); the indices are TS atom indices only when it is `true`. Recorded only when the IRC verdict was established by graph isomorphism of the perceived fragments; `null` when only the bond-list fallback decided, when IRC was not run or did not validate the TS (`ts_checks.IRC` is not `true`), and when not recorded. Transition states only |
| `rxn_label` | `str?` | Reaction label this TS belongs to |
| `thermo` | `null` | Always `null` for transition states |

### TS Validation Verdicts

`ts_checks` is the transition state's provenance: one verdict per validation ARC
runs. `true` is a pass, `false` a failure, and `null` means the check did not run
or reached no conclusion — **`null` is not a failure**, and a TS whose checks are
all `null` has not been shown to be wrong, only left unvalidated.

| Field | Type | Description |
|---|---|---|
| `E0` | `bool?` | The zero-point-corrected energy of the TS lies above both reaction wells |
| `e_elect` | `bool?` | The electronic energy of the TS lies above both reaction wells |
| `IRC` | `bool?` | The intrinsic reaction coordinate connects this TS to the reaction's own reactants and products. This — not `irc_converged` — is the IRC verdict |
| `freq` | `bool?` | The frequency calculation yielded exactly one imaginary mode, of a magnitude consistent with the reaction |
| `NMD` | `bool?` | The imaginary mode's normal mode displacement moves the atoms whose bonds the reaction forms and breaks |
| `warnings` | `str` | Free text accumulated by the checks above; `""` when none raised anything |

---

## Reactions

`reactions` is a list of entries, one per reaction.

| Field | Type | Description |
|---|---|---|
| `label` | `str` | Reaction label |
| `reactant_labels` | `list[str]` | Species labels of reactants, sorted and de-duplicated: `HO2 + HO2 <=> H2O2 + O2` lists `HO2` once. It is not the stoichiometry; use `reactant_species_labels` |
| `product_labels` | `list[str]` | Species labels of products, sorted and de-duplicated; use `product_species_labels` for the stoichiometry |
| `reactant_species_labels` | `list[str]?` | The reactant species with **one entry per occurrence**, in the order of `ARCReaction.get_reactants_and_products` (`["HO2", "HO2"]`). `null` when the reaction holds no reactant species objects (a reaction restored or built from labels alone, e.g. a restart file written before reactions carried their species), because the occurrences are counted from those objects and are not reconstructed from the label; `reactant_labels` still states the labels. When `atom_map` is not `null` it equals `atom_map_reactant_labels` |
| `product_species_labels` | `list[str]?` | The same for the products, `null` under the same condition. When `atom_map` is not `null` it equals `atom_map_product_labels` |
| `family` | `str?` | Reaction family |
| `multiplicity` | `int?` | Reaction spin multiplicity |
| `ts_label` | `str?` | Label of the associated transition state |
| `kinetics` | `dict?` | Fitted kinetics (see below); `null` if not computed |
| `reversible` | `bool?` | `true` for the `<=>` arrow, `false` for `=>`, `null` when the arrow is unknown. ARC writes every reaction with `<=>`, so every reaction ARC exports is `true`; it states the notation of the label, not a measured or computed reversibility |
| `atom_map` | `list[int]?` | ARC's reactant-to-product atom map as the reaction holds it; export never computes one, so it is `null` when ARC had not mapped the reaction, and it is also `null` unless the map is a permutation that conserves the element (reactant atom `i` and product atom `atom_map[i]` have the same element), and unless the exported geometry of every reactant and product has the atom order of that species' molecule, which the map counts atoms in. That is decided by ARC's `are_coords_compliant_with_graph`: the geometry must have as many atoms as the molecule, the element of the molecule's atom in every position, and every bond of the molecule no longer than 1.2 times the single-bond length of its two elements (bond orders are not considered), so a geometry with a bond stretched past that, such as a loosely bound complex, also fails it. The check can only turn the map into `null` (a warning is logged); it never reorders or alters a map. When the map is `null` for this reason, `ts_atom_map_unavailable_reason` is `species_atom_order_mismatch` only if the TS has a geometry, its IRC check passed and that check recorded a TS atom map; otherwise it is `no_ts`, `irc_not_passed`, or the reason the IRC check recorded. Entry `i` is the **0-based** index of the product atom that reactant atom `i` becomes. Reactant atoms are counted over `atom_map_reactant_labels` in that order, each occurrence contributing the atoms of that species in the atom order of its exported geometry, one block per occurrence (`CH3 + CH3` has two blocks of four); product atoms are counted the same way over `atom_map_product_labels`. That is the order of the reaction's species lists, **not** that of `reactant_labels`, which is sorted and de-duplicated. Among symmetry-equivalent atoms (which hydrogen of CH4, which CH3 block) the choice is arbitrary and not specific to the reactive site. It says nothing about the atom order of the transition state; `ts_atom_map` states the TS atom of every atom |
| `atom_map_reactant_labels` | `list[str]?` | The label of each reactant in the order `atom_map` counts their atoms, one entry per occurrence (`["CH3", "CH3"]`). `null` exactly when `atom_map` is `null` |
| `atom_map_product_labels` | `list[str]?` | The same for the products |
| `atom_map_source` | `str?` | `"inferred"` when ARC computed the map with its own mapping algorithm (a restart preserves this) and `"declared"` when it was explicitly declared as a user-given map. `null` when `atom_map` is `null` and when the origin was not recorded: an input or restart dict that states no source (ARC cannot tell the two apart), or an unknown value. A user declares a map by giving `atom_map_source: declared` next to `atom_map` in the reaction's input or restart dict |
| `atom_map_method` | `str?` | The algorithm that computed an inferred map (ARC's mapping driver, with the reaction family when known). `null` unless `atom_map_source` is `"inferred"` |
| `ts_atom_map` | `dict?` | The atom of the transition state that every atom of the reaction is: `{ts_label, reactants, products, method, reactant_endpoint, ts_atom_order_follows_reactants}`. `reactants` has one **0-based TS atom index** per concatenated reactant atom, in the order of `atom_map_reactant_labels` (each occurrence contributing the atoms of its species in the atom order of its exported geometry), and `products` has one per concatenated product atom, in the order of `atom_map_product_labels`, so `products[atom_map[i]] == reactants[i]` and each list is a bijection onto the TS atoms. `method` is always `"irc_endpoint_cgr_isomorphism"`: ARC builds the **condensed graph of reaction** of the reaction (the reactant atoms as nodes labelled by element; an edge for every bond of the reactants or the products, labelled by which of the two it is in, from `atom_map`) and that of the TS (the TS atoms; the bonds of the two IRC endpoint geometries, labelled the same way), and takes a label-preserving isomorphism of the first onto the second, the identity when it is one. The TS-indexed graph takes its bonds from the molecule fragments the IRC verdict perceived on the two endpoints (connectivity and element only, bond orders ignored), not from a second distance perception; the fragment atoms are tied to the endpoint atoms by exact coordinate equality, and a fragment atom without coordinates, or one that is not tied to exactly one endpoint atom of its element, refuses the map. **The correspondence is a constitutional (2D) one.** Symmetry-equivalent atoms, including diastereotopic ones, follow a deterministic convention (the identity when it is a valid map, otherwise the first match the graph matcher finds), which is not canonical across networkx versions, and the number of equivalent maps is not stated. **What is verified:** the first geometry of every IRC log (Gaussian's first `Input orientation`) is congruent with the TS geometry atom for atom, meaning the same elements in the same order and, without permuting any atom, the same coordinates after a rigid rotation and translation (a Kabsch superposition by a proper rotation after giving the parsed geometry the isotopes of the reference, so a mirror image is refused) with a root of the summed squared deviations of at most 1e-3 Angstrom, which covers coordinate print precision and not chemistry; each endpoint's optimization log starts from a geometry congruent with the last geometry of the IRC log of the same index, and the endpoint geometry is exactly the final geometry of that optimization log; `atom_map` is a permutation; and the endpoint graphs from the IRC verdict's perception are coupled through `atom_map` by the isomorphism. **What rests on construction:** the ESS keeps the atom order of its input within a log, and ARC's parsing and writing keep it, so endpoint atom `i` is TS atom `i`. Only Gaussian IRC logs are read; there is no verified first-geometry format for another program, so any other log gives `irc_start_geometry_unavailable`. No ARC TS method guarantees that the TS lists its atoms in reactant order, and `ts_atom_order_follows_reactants` is `true` exactly when `reactants` is `[0, 1, ..., n-1]`, so a consumer reads the order from the map and does not infer it from the TS method. `reactant_endpoint` is the IRC endpoint, `1` or `2` in the order the IRC check received them, whose bonds served as the reactants. To get the slice of one participant, count the atoms of the participants in `atom_map_reactant_labels` order (or `atom_map_product_labels` for the products): participant `k` owns the entries from the sum of the atom counts before it, and the entry at its own atom `a` is the TS atom of the atom `a` of that participant's exported geometry. Recorded only when the IRC verdict came from graph isomorphism (the same path as `irc_participant_mapping`), when the endpoint geometries have the element sequence of the TS, and when every IRC log starts from the TS geometry (so that endpoint atom `i` is TS atom `i`); it never changes `atom_map`. The atom order of a participant's geometry is the order its `mol` counts in, which `atom_map` and the bonds the isomorphism uses are counted in. A species given coordinates has its `mol` perceived from them, so the two agree, but a species that keeps its own `mol` (`keep_mol`) may not, and the export checks each reactant and product (same elements in the same positions, and every bond of the `mol` at a bonded distance in the geometry); a species that fails gives `species_atom_order_mismatch`. `null` with `ts_atom_map_unavailable_reason` otherwise |
| `ts_atom_map_unavailable_reason` | `str?` | Why `ts_atom_map` is `null`; `null` exactly when it is not. One of: `no_ts` (the reaction has no TS, or the TS has no geometry), `irc_not_passed` (`ts_checks.IRC` is not `true`), `irc_fallback_path` (the IRC verdict came from the bond-list fallback, which has no atom correspondence), `no_atom_map` (the reaction exports no `atom_map`; ARC does not compute one for this), `endpoint_perception_mismatch` (the connectivity of the fragments the IRC verdict perceived on an endpoint is not the bond graph of its side, or those fragments cannot be tied atom for atom to the endpoint geometry and the TS atoms), `atom_map_contradicts_ts` (the bonds the endpoints form and break are not those of `atom_map`, so no isomorphism exists: the map is not altered and a warning is logged), `species_atom_order_mismatch` (a reactant or product geometry does not have the atom order of its molecule), `irc_start_geometry_unavailable` (an IRC log is missing, or the first geometry of one cannot be read; only Gaussian logs are read), `irc_start_geometry_differs` (the first geometry of an IRC log is not congruent with the TS geometry atom for atom), `irc_endpoint_geometry_differs` (an endpoint's optimization did not start from the last geometry of its IRC log, or the endpoint geometry is not the final geometry of that optimization log), `computation_failed` (the computation raised an error, which is logged), `not_recorded` (a restart written before the map was recorded, or a record that is malformed: wrong length, not a bijection, an element or the `atom_map` invariant violated, which is dropped and logged) |
| `long_kinetic_description` | `str` | ARC's verbose description of how the rate coefficient was obtained. **Omitted** (key absent) when the reaction carries no such description |

**`kinetics`**:

| Field | Type | Description |
|---|---|---|
| `A` | `float?` | Pre-exponential factor |
| `A_units` | `str?` | Units of `A`, verbatim from the fit. Not pinned by the schema because it follows the reaction's molecularity (`s^-1` unimolecular, `cm^3/(mol*s)` bimolecular, and so on) — read it, never assume it |
| `n` | `float?` | Temperature exponent |
| `Ea` | `float?` | Activation energy |
| `Ea_units` | `str?` | Units of `Ea`, verbatim from the fit (`kJ/mol` or `kcal/mol`). Not pinned for the same reason as `A_units`: ARC reports the source's unit rather than converting |
| `T0_k` | `float?` | The reference temperature (K) of `k = A (T/T0)^n exp(-Ea/RT)`. `A` and `n` cannot be interpreted without it |
| `Tmin_k` | `float?` | Minimum fitted temperature (K) |
| `Tmax_k` | `float?` | Maximum fitted temperature (K) |
| `dA` | `float?` | **Multiplicative** uncertainty factor on `A`, dimensionless, as Arkane reports it (`dA = *|/ 1.48466`). The one-sigma band is `[A / dA, A * dA]` — **not** `A ± dA`. This is why it has no units sibling: unlike `dEa` there is no unit to carry |
| `dn` | `float?` | **Additive** uncertainty on `n`: the band is `n ± dn`. Dimensionless |
| `dEa` | `float?` | **Additive** uncertainty on `Ea`: the band is `Ea ± dEa`, in `dEa_units` |
| `dEa_units` | `str?` | Units of dEa |
| `n_data_points` | `int?` | Number of data points used in fitting |
| `atom_corrections_applied` | `bool?` | Whether the Arkane kinetics run that fitted these parameters had its atom energy correction switched on; it decides what the TS's `statmech.e0_kj_mol` is. The kinetics run applies no bond correction. `null` when the kinetics did not come from such a run, and when the TS or any well of the reaction was loaded from an Arkane YAML, which Arkane takes as-is |
| `comment` | `str?` | The comment Arkane attached to the fitted Arrhenius expression, verbatim, plus, for a rate whose TS failed the IRC check, the `ts_validation` marker on a further line. `null` when the kinetics carry none (user-supplied or restored kinetics) |
| `ts_validation` | `str?` | A human-readable summary ARC stamps when it computed this rate from a TS that positively failed its IRC check (`ts_checks.IRC` is `false`); read the TS record's `ts_checks` for the structured verdict. `null` when ARC found no such failure, which includes an IRC check that was not performed |
| `tunneling` | `str?` | The tunneling correction applied to the fitted `A`/`n`/`Ea`, stamped by the Arkane run that produced them (currently `"Eckart"`). `null` when the kinetics did not come from that run — user-supplied in the input YAML, restored from a restart file, or produced by a non-Arkane statmech adapter — because those carry no tunneling correction. Never defaulted from the template constant: a consumer must be able to tell "Eckart was applied" from "nothing is known" |
