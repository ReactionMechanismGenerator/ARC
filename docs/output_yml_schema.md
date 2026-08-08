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
├── schema_version: "1.1"
├── project
├── arc_version
├── arc_git_commit?
├── arkane_git_commit?
├── datetime_started?
├── datetime_completed
│
├── composite_method?
├── opt_level?
├── freq_level?
├── sp_level?
├── neb_level (omitted unless the orca_neb TS adapter was configured)
├── arkane_level_of_theory?
├── freq_scale_factor?
├── freq_scale_factor_source?
├── bac_type?
├── atom_energy_corrections?
├── bond_additivity_corrections?
├── tckdb_evidence?
│   └── path, schema_name, schema_version, document_id
│
├── species: []
│   └── label, original_label, charge, multiplicity, converged
│       ├── smiles?, inchi?, inchi_key?, formula?
│       ├── xyz?
│       ├── conformers, conformer_energies   (both omitted when none were screened)
│       ├── sp_energy_hartree?, zpe_hartree?, opt_converged?
│       ├── coarse_opt_log?, coarse_opt_n_steps?, coarse_opt_final_energy_hartree?
│       ├── opt_n_steps?, opt_final_energy_hartree?
│       ├── coarse_opt_input_xyz?, coarse_opt_output_xyz?, opt_input_xyz?
│       ├── freq_n_imag?, imag_freq_cm1?, imaginary_frequencies_cm1?
│       ├── opt_log?, freq_log?, sp_log?
│       ├── sp_spin_diagnostic?
│       │   └── {s_squared, s_squared_expected?, s_squared_annihilated?}
│       ├── opt_input?, freq_input?, sp_input?
│       ├── opt_constraints: [], freq_constraints: [], sp_constraints: []
│       │   └── [{coordinate_type, atom_indices, index_base,
│       │        target_value?, target_value_units?}, ...]
│       ├── opt_final_settings?, coarse_opt_final_settings?
│       ├── freq_final_settings?, sp_final_settings?
│       ├── rotor_scans: []
│       │   └── key, source_log?, constraints?, result
│       │       ├── dimension, relaxed?, zero_energy_reference_hartree?
│       │       ├── coordinate: {coordinate_type, atom_indices, index_base, unit,
│       │       │                sample_count, symmetry_number?,
│       │       │                requested_step_size?,
│       │       │                requested_start?, requested_end?}
│       │       └── samples: [{source_index, angle_degrees,
│       │                      relative_energy_kj_mol,
│       │                      electronic_energy_hartree?, geometry_xyz?}, ...]
│       ├── energy_corrections: []
│       │   └── correction_type, model, level_of_theory?, matched_arkane_key?,
│       │       total, components, reference_atom_energies?, parameter_table?
│       ├── ess_versions?, ess_software?
│       ├── thermo?
│       │   ├── h298_kj_mol, s298_j_mol_k, tmin_k, tmax_k
│       │   ├── thermo_points?: [{temperature_k, cp_j_mol_k, h_kj_mol, s_j_mol_k, g_kj_mol}, ...]
│       │   ├── nasa_low?: {tmin_k, tmax_k, coeffs}
│       │   └── nasa_high?: {tmin_k, tmax_k, coeffs}
│       └── statmech?
│           ├── e0_kj_mol?, spin_multiplicity, optical_isomers
│           ├── is_linear, external_symmetry, point_group?
│           ├── rigid_rotor_kind, harmonic_frequencies_cm1?
│           ├── torsions: []
│           │   └── symmetry_number, treatment, atom_indices, pivot_atoms,
│           │       barrier_kj_mol?, source_scan_key?
│           └── rejected_torsions?: []
│               └── rotor_index, invalidation_reason, atom_indices, pivot_atoms,
│                   dimension, source_log?
│
├── transition_states: []
│   └── (all species fields, plus:)
│       ├── chosen_ts_method?, successful_ts_methods?
│       ├── ts_guesses: []
│       ├── neb_log?, gsm_log?, irc_logs: [], irc_log_directions: [], irc_converged?
│       ├── ts_checks: {E0?, e_elect?, IRC?, freq?, NMD?, warnings}
│       └── rxn_label
│
└── reactions: []
    └── label, reactant_labels, product_labels, family?, multiplicity, ts_label
        ├── long_kinetic_description   (omitted when empty)
        └── kinetics?
            └── A, A_units?, n, Ea, Ea_units?, Tmin_k, Tmax_k
                dA?, dn?, dEa?, dEa_units?, n_data_points?, tunneling
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
| `schema_version` | `str` | Output contract version. `1.1` adds the optional `tckdb_evidence` descriptor, renames the thermo block's `cp_data` to `thermo_points` (widened from Cp-only to Cp/H/S/G per temperature; there is no `cp_data` alias), renames the AEC `parameter_table` to `reference_atom_energies`, adds `imaginary_frequencies_cm1` and the per-correction `matched_arkane_key`, adds `ess_software` so each `ess_versions` banner can be paired with the program that actually produced it, adds the optional `statmech.rejected_torsions`, adds the TS-only `ts_checks`, adds `kinetics.T0_k` and the constraints' `target_value_units`, and adds the `rotor_scans` block, whose samples carry `angle_degrees` — the **absolute** dihedral measured on each sample's own geometry, never a displacement — and whose `requested_step_size` is signed, so a reversed scan records its direction instead of losing the whole requested grid. Emitted from a single constant shared with the evidence sidecar's `output_schema_version`, so the two cannot drift |
| `project` | `str` | ARC project name |
| `arc_version` | `str` | ARC version string |
| `arc_git_commit` | `str?` | ARC repo HEAD commit hash |
| `arkane_git_commit` | `str?` | RMG-Py (Arkane) repo HEAD commit hash |
| `datetime_started` | `str?` | Run start timestamp (`YYYY-MM-DD HH:MM`); `null` when ARC has no recorded start time |
| `datetime_completed` | `str` | Completion timestamp (`YYYY-MM-DD HH:MM`) |
| `tckdb_evidence` | `dict` | Descriptor for `tckdb_evidence.json`. **Omitted** (key absent, never `null`) unless the sidecar was written successfully |

The descriptor binds `output.yml` to the parser-neutral evidence sidecar with
`path`, `schema_name`, `schema_version`, and a shared UUID4 `document_id`.
ARC writes the sidecar first and `output.yml` second so consumers can reject a
stale or interrupted pair by comparing document IDs. The evidence file contains
best-effort, versioned Hessian, IRC, and GSM facts parsed by ARC; it contains no
TCKDB upload request objects.

The sidecar's `output_schema_version` is emitted from the same constant as
`schema_version` above, so a consumer gating on it cannot be misled by drift
between the two files.

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
shipping a mismatched pair. `parser_version` is `arc-hessian-2`; records written
by `arc-hessian-1` are **not** frame-consistent and must not be trusted for any
re-derivation.

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
is Gaussian's own mass-weighted intrinsic reaction coordinate, in bohr*sqrt(amu) and
signed about the TS. The two share no units, no sign convention, and no origin;
they must never be compared or plotted on one axis.

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

`parser_version` is `arc-gsm-stringfile-3`. Records written by
`arc-gsm-stringfile-1` carry `path_coordinate_angstrom`, a `node_label` equal to
the frame index, and per-point energies taken from `<run>.<slot>` files indexed as
though `slot` were the frame index — those per-point values are wrong and must not
be trusted. `arc-gsm-stringfile-2` carries the same quantity under the name
`cumulative_cartesian_displacement_angstrom`, which overstates it by asserting a
plain Cartesian displacement, and attaches no per-point energies at all.

## Levels of Theory

Every level field is a **level dict**: `Level.as_dict()` with the `repr` and
`compatible_ess` keys removed. Only the level's non-`None` attributes appear, so the
key set varies between runs; the possible keys are `method`, `basis`,
`auxiliary_basis`, `dispersion`, `cabs`, `method_type` (e.g. `dft`, `wavefunction`,
`composite`, `force_field`), `software`, `software_version`, `solvation_method`,
`solvent`, `solvation_scheme_level`, `args`, and `year`.

`solvation_scheme_level` is itself a nested level dict, held to the same key set —
`Level.as_dict()` leaves a `Level` object there, which ARC converts recursively so
the document contains only plain types that `yaml.safe_load` accepts. `args` is the
one free-form value (arbitrary ESS keywords).

| Field | Type | Description |
|---|---|---|
| `composite_method` | `dict?` | Composite method level (e.g. CBS-QB3, G4); `null` when no composite method was used |
| `opt_level` | `dict?` | Geometry optimization level |
| `freq_level` | `dict?` | Frequency calculation level |
| `sp_level` | `dict?` | Single-point energy level |
| `neb_level` | `dict` | NEB TS search level. **Omitted** (key absent) unless the `orca_neb` TS adapter was configured for the run and `orca_neb_settings['level']` is set — it does not indicate that an NEB job actually ran |
| `arkane_level_of_theory` | `dict?` | Composite level Arkane uses for energy corrections |
| `freq_scale_factor` | `float?` | Harmonic frequency scaling factor |
| `freq_scale_factor_source` | `str?` | Source of the scaling factor (`null` if user-provided) |
| `bac_type` | `str?` | Bond additivity correction type: `"p"`, `"m"`, or `null` |
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

### Screened Conformers

Both keys are **omitted** (absent, not `null`) when ARC screened no conformers for
the species, so a consumer must use `.get()` rather than indexing.

| Field | Type | Description |
|---|---|---|
| `conformers` | `list[str]` | Screened conformer geometries as XYZ blocks, in ARC's conformer order |
| `conformer_energies` | `list[float?]` | One energy per conformer relative to the lowest, in lockstep with `conformers`. An entry is `null` for a conformer whose energy has not been filled in yet: the list is pre-allocated to the conformer count and populated one job at a time, so a partly-`null` list is an ordinary mid-run state, not an error. **The unit is stage-dependent and nothing in this document distinguishes the two stages**: it is kcal/mol while the list still holds force-field energies from the conformer screen, and kJ/mol once the conformers have been optimized at a quantum-chemical level. Treat this list as an ordering aid, and use `sp_energy_hartree` / `statmech.e0_kj_mol` when an unambiguous scale is required |

### Energies

| Field | Type | Description |
|---|---|---|
| `sp_energy_hartree` | `float?` | Single-point electronic energy (Hartree), **as the ESS reported it** — no atom-energy or bond-additivity correction has been applied. The corrections are reported separately under `energy_corrections`, which is also where their sign and unit conventions are stated |
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
| `opt_input` | `str?` | Geometry optimization input deck; `null` when the deck is not on disk |
| `freq_input` | `str?` | Frequency calculation input deck; `null` when the deck is not on disk |
| `sp_input` | `str?` | Single-point energy input deck; `null` when the deck is not on disk |
| `ess_versions` | `dict?` | ESS version banners, keyed by job type (`{'opt'|'freq'|'sp'|'neb': banner_str, ...}`). Each value is the **full banner** as the program printed it (e.g. `'Gaussian 16, Revision C.01'`, `'ORCA 6.0.0'`), not a bare version number — nothing trims it. A job type is absent when its log is missing or its banner could not be parsed; the whole field is `null` for a non-converged species or when nothing could be parsed |
| `ess_software` | `dict?` | The ESS that produced each log, keyed by the same job types and read from the same log files (`{'opt': 'gaussian', 'sp': 'orca', ...}`). Values are ARC's lowercase ESS names. Pair `ess_versions[job]` with `ess_software[job]` — a run may use different programs for different job types, so the level of theory's declared software is not a safe stand-in. The key set is a **superset** of `ess_versions`': a log whose ESS is identified but whose banner cannot be parsed appears here only. `null` under the same conditions as `ess_versions` |

Each input deck sits in the same directory as its log, under the ESS-specific
filename from `settings['input_filenames']`. Software with no entry in that map
(`gcn`, `torchani`, `mockter`, ...) yields `null`.

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

| Field | Type | Description |
|---|---|---|
| `correction_type` | `str` | `"atom_energy"` or `"bond_additivity"` |
| `model` | `str` | `"arkane_atom_energy"`, or `"petersson"` / `"melius"` for BAC |
| `level_of_theory` | `dict?` | The ARC level of theory (a level dict) whose energies the correction was applied to — ARC's own `arkane_level_of_theory`, **not** an Arkane database key; `null` when unknown |
| `matched_arkane_key` | `str?` | The Arkane database key string (e.g. `"LevelOfTheory(method='wb97xd',basis='def2tzvp',software='gaussian')"`) whose parameters produced this record's numbers: the atom-energy-section key ARC hands Arkane as its model chemistry, so both records for a species carry the same value. `null` when no atom-energy key was matched |
| `total` | `dict` | `{value: float, unit: str}` — the applied correction total. `unit` is a closed vocabulary tied to `correction_type`: `hartree` for `atom_energy`, `kcal_mol` for `bond_additivity`. Same for `components[].parameter_unit`, by `component_kind` |
| `components` | `list` | Native per-atom / per-bond decomposition. Always present; `[]` when no decomposition is available, and deliberately emptied for a BAC whose bonds are not all parameterized (a partial decomposition would not sum to `total`) |

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
| `parameter_table` | `dict?` | **BAC only.** `{unit: "kcal_mol", values: {bond: float, ...}}` — the per-bond BAC parameters ARC actually used. **Omitted** when the run has no such table, unless `bac_type == 'p'` (so a Melius run never carries one), and unless the key matched in the BAC section is `matched_arkane_key` |

### Rotor Scans

`rotor_scans` is always a list, `[]` for monoatomic or non-converged species. It holds
one record per successful 1D rotor whose scan log parsed cleanly; rotors that fail any
of those conditions — unsuccessful, multi-dimensional, or unparseable — are skipped
entirely, so the list can be shorter than `statmech.torsions` and the matching
torsion's `source_scan_key` is then `null`. `constraints` (held-fixed coordinates,
excluding the scanned coordinate itself) is **omitted** when none were found, and
`result.zero_energy_reference_hartree` is **omitted** when the parser reports none.

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

### Thermochemistry

`thermo` is `null` for non-converged species or species without thermo data.

| Field | Type | Description |
|---|---|---|
| `h298_kj_mol` | `float` | Standard enthalpy at 298 K (kJ/mol) |
| `s298_j_mol_k` | `float?` | Standard entropy at 298 K (J/(mol K)) |
| `tmin_k` | `float?` | Minimum temperature (K) |
| `tmax_k` | `float?` | Maximum temperature (K) |
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
| `e0_kj_mol` | `float?` | Ground-state energy (kJ/mol) |
| `spin_multiplicity` | `int?` | Spin multiplicity |
| `optical_isomers` | `int?` | Number of optical isomers |
| `is_linear` | `bool?` | Whether the molecule is linear |
| `external_symmetry` | `int?` | External symmetry number |
| `point_group` | `str?` | Point group (e.g. `C2v`) |
| `rigid_rotor_kind` | `str` | `"linear"` or `"asymmetric_top"`. The builder has a third `"atom"` branch, but `statmech` is `null` for monoatomic species, so it never reaches `output.yml` |
| `harmonic_frequencies_cm1` | `list[float]?` | Harmonic frequencies (cm-1) as parsed from the ESS, **unscaled** — `freq_scale_factor` has *not* been applied, although it has been applied to `statmech.e0_kj_mol`, so recomputing ZPE from this list will not reproduce `e0_kj_mol` unless you scale first. For TSs **every** negative (imaginary) frequency is dropped, not only the reaction mode — a TS that legitimately carries additional small imaginary modes (which `check_imaginary_frequencies` permits) loses those too, so the list can be shorter than the species' true mode count. Non-TS species are not filtered at all |
| `torsions` | `list` | Internal rotation data (see below) |
| `rejected_torsions` | `list` | Rotors ARC evaluated and rejected (see below). **Optional in the schema** — not in `statmech`'s `required` list — unlike every other field in this table. This version of the writer always emits it (`[]` when there is nothing to report, never omitted), but a consumer reading documents from other producers, or from before this key existed, must tolerate its absence rather than assuming `[]` |

**`torsions`** entries (only successful rotors):

| Field | Type | Description |
|---|---|---|
| `symmetry_number` | `int?` | Torsional symmetry number |
| `treatment` | `str` | `"hindered_rotor"` or `"free_rotor"` |
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
| `source_log` | `str?` | The rejected rotor's scan log path, relative to the project directory, recorded directly from `rotors_dict`'s `scan_path` rather than via a `rotor_scans` key, since this rotor has no `rotor_scans` record. This is a bare `os.path.relpath` with no containment check (same treatment `rotor_scans[].source_log` gets) — a log outside the project directory yields a `../..`-prefixed path revealing host layout. **Omitted** (never `null`) when `rotors_dict` records no on-disk scan for this rotor at write time. See the note above: absence does not mean the rotor was never a torsion |

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
| `irc_log_directions` | `list[str?]` | Which branch of the reaction path each log traversed, in lockstep with `irc_logs`. A closed vocabulary — `"forward"`, `"reverse"`, or `null` when ARC recorded no direction for that log — because this is the one field where a wrong value silently swaps reactant and product |
| `irc_converged` | `bool?` | **Whether the IRC jobs completed, not whether the IRC validated the TS.** It becomes `true` once both IRC directions finished, whatever their endpoints turned out to be, and is `null` when the run did not request IRC. The validation verdict is `ts_checks.IRC` |
| `ts_checks` | `dict` | ARC's own verdicts on this transition state (see below) |
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
| `reactant_labels` | `list[str]` | Species labels of reactants |
| `product_labels` | `list[str]` | Species labels of products |
| `family` | `str?` | Reaction family |
| `multiplicity` | `int?` | Reaction spin multiplicity |
| `ts_label` | `str?` | Label of the associated transition state |
| `kinetics` | `dict?` | Fitted kinetics (see below); `null` if not computed |
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
| `tunneling` | `str?` | The tunneling correction applied to the fitted `A`/`n`/`Ea`, stamped by the Arkane run that produced them (currently `"Eckart"`). `null` when the kinetics did not come from that run — user-supplied in the input YAML, restored from a restart file, or produced by a non-Arkane statmech adapter — because those carry no tunneling correction. Never defaulted from the template constant: a consumer must be able to tell "Eckart was applied" from "nothing is known" |
