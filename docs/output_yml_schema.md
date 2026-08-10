# output.yml Schema Reference

The consolidated `output.yml` is written atomically to `<project_directory>/output/output.yml`
at the end of an ARC run. It contains all result data in a single file with run-relative paths
so downstream consumers (TCKDB, analysis scripts) need only one file.

Fields marked **nullable** may be `null` when the species did not converge,
the job was not requested, or the data is not applicable (e.g. monoatomic species).
A handful of fields are instead **omitted** — the key is absent from the mapping
rather than present with a `null` value — which matters to a consumer reading the
document with `.get()`. Those fields are called out explicitly below; everything
else marked `?` is present with a `null` value.

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
├── cost_metrics
│   ├── wall_time_hrs?
│   ├── total_job_count, total_execution_time_hrs, total_core_hours
│   ├── jobs_missing_time, jobs_missing_cores
│   └── per_ess?: {<ess>: {job_count, execution_time_hrs, core_hours, jobs_missing_time}, ...}
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
│       ├── freq_n_imag?, imag_freq_cm1?
│       ├── opt_log?, freq_log?, sp_log?
│       ├── sp_spin_diagnostic?
│       │   └── {s_squared, s_squared_expected?, s_squared_annihilated?}
│       ├── opt_input?, freq_input?, sp_input?
│       ├── opt_constraints: [], freq_constraints: [], sp_constraints: []
│       │   └── [{coordinate_type, atom_indices, index_base, target_value?}, ...]
│       ├── opt_final_settings?, coarse_opt_final_settings?
│       ├── freq_final_settings?, sp_final_settings?
│       ├── rotor_scans: []
│       │   └── key, source_log?, constraints?, result
│       │       ├── dimension, relaxed, zero_energy_reference_hartree?
│       │       ├── coordinate: {coordinate_type, atom_indices, index_base, unit,
│       │       │                sample_count, symmetry_number?,
│       │       │                requested_step_size?,
│       │       │                requested_start?, requested_end?}
│       │       └── samples: [{source_index, angle_degrees,
│       │                      relative_energy_kj_mol,
│       │                      electronic_energy_hartree?, geometry_xyz?}, ...]
│       ├── energy_corrections: []
│       │   └── correction_type, model, level_of_theory?, total,
│       │       components, parameter_table?
│       ├── ess_versions?
│       ├── thermo?
│       │   ├── h298_kj_mol, s298_j_mol_k, tmin_k, tmax_k
│       │   ├── thermo_points?: [{temperature_k, cp_j_mol_k, h_kj_mol, s_j_mol_k, g_kj_mol}, ...]
│       │   ├── nasa_low?: {tmin_k, tmax_k, coeffs}
│       │   └── nasa_high?: {tmin_k, tmax_k, coeffs}
│       └── statmech?
│           ├── e0_kj_mol?, spin_multiplicity, optical_isomers
│           ├── is_linear, external_symmetry, point_group?
│           ├── rigid_rotor_kind, harmonic_frequencies_cm1?
│           └── torsions: []
│               └── symmetry_number, treatment, atom_indices, pivot_atoms,
│                   barrier_kj_mol?, source_scan_key?
│
├── transition_states: []
│   └── (all species fields, plus:)
│       ├── chosen_ts_method?, successful_ts_methods?
│       ├── ts_guesses: []
│       ├── neb_log?, gsm_log?, irc_logs: [], irc_log_directions: [], irc_converged?
│       └── rxn_label
│
└── reactions: []
    └── label, reactant_labels, product_labels, family?, multiplicity, ts_label
        ├── long_kinetic_description   (omitted when empty)
        └── kinetics?
            └── A, A_units?, n, Ea, Ea_units?, Tmin_k, Tmax_k
                dA?, dn?, dEa?, dEa_units?, n_data_points?, tunneling
```

`?` = nullable; a parenthesised note marks a key that is omitted entirely rather than
emitted as `null`. TS entries share all species fields but add IRC/NEB/method fields
and always have `thermo: null`.

---

## Top-level

| Field | Type | Description |
|---|---|---|
| `schema_version` | `str` | Output contract version; `1.1` adds the optional evidence descriptor |
| `project` | `str` | ARC project name |
| `arc_version` | `str` | ARC version string |
| `arc_git_commit` | `str?` | ARC repo HEAD commit hash |
| `arkane_git_commit` | `str?` | RMG-Py (Arkane) repo HEAD commit hash |
| `datetime_started` | `str?` | Run start timestamp (`YYYY-MM-DD HH:MM`); `null` when ARC has no recorded start time |
| `datetime_completed` | `str` | Completion timestamp (`YYYY-MM-DD HH:MM`) |
| `tckdb_evidence` | `dict?` | Descriptor for `tckdb_evidence.json`, present only after the sidecar was written successfully |

The descriptor binds `output.yml` to the parser-neutral evidence sidecar with
`path`, `schema_name`, `schema_version`, and a shared UUID4 `document_id`.
ARC writes the sidecar first and `output.yml` second so consumers can reject a
stale or interrupted pair by comparing document IDs. The evidence file contains
best-effort, versioned Hessian, IRC, and GSM facts parsed by ARC; it contains no
TCKDB upload request objects.

## Cost Metrics

`cost_metrics` records the computational cost of the run, aggregated from per-job
records collected as jobs complete (persisted in the restart file, so restarted runs
keep their history). Jobs with unavailable run time or core count are **counted**, not
silently dropped, so analysis scripts know the coverage. Wall time is queue-confounded
and should be treated as a secondary metric; ESS execution time and core-hours are the
primary cost measures. Pipe-mode tasks are not individually tracked and are therefore
not included in the per-job aggregates.

| Field | Type | Description |
|---|---|---|
| `wall_time_hrs` | `float?` | Wall-clock duration of the run in hours (also derivable from the `datetime_*` pair, which only has minute resolution) |
| `total_job_count` | `int` | Total number of completed jobs recorded |
| `total_execution_time_hrs` | `float` | Summed ESS job execution time (hours), over jobs with a known run time |
| `total_core_hours` | `float` | Summed execution time x CPU cores (hours), over jobs with known run time and core count |
| `jobs_missing_time` | `int` | Jobs ARC recorded with no run time. They are excluded from both `total_execution_time_hrs` and `total_core_hours`, so a non-zero value means **both totals understate the run**, by this many jobs |
| `jobs_missing_cores` | `int` | Jobs that have a run time but no core count. They still count toward `total_execution_time_hrs`, but are excluded from `total_core_hours`, so a non-zero value means **core-hours alone understate the run**, by this many jobs |
| `per_ess` | `dict?` | Per-ESS-software aggregates, keyed by the job adapter name (e.g. `gaussian`, `orca`, `xtb`); records with no adapter recorded are bucketed under `unknown`. `null` when no jobs were recorded |

Each `per_ess` entry: `{job_count: int, execution_time_hrs: float, core_hours: float, jobs_missing_time: int}`.

## Levels of Theory

Every level field is a **level dict**: `Level.as_dict()` with the `repr` and
`compatible_ess` keys removed. Only the level's non-`None` attributes appear, so the
key set varies between runs; the possible keys are `method`, `basis`,
`auxiliary_basis`, `dispersion`, `cabs`, `method_type` (e.g. `dft`, `wavefunction`,
`composite`, `force_field`), `software`, `software_version`, `solvation_method`,
`solvent`, `solvation_scheme_level`, `args`, and `year`.

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
| `atom_energy_corrections` | `dict?` | Per-atom corrections in Hartree (`{element: value, ...}`) |
| `bond_additivity_corrections` | `dict?` | Per-bond corrections in kcal/mol (`{bond: value, ...}`) |

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
| `original_label` | `str` | Original user-provided label |
| `charge` | `int` | Molecular charge |
| `multiplicity` | `int` | Spin multiplicity |
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
| `conformer_energies` | `list[float]` | Post-screen energies (kJ/mol) relative to the lowest conformer, in lockstep with `conformers` |

### Energies

| Field | Type | Description |
|---|---|---|
| `sp_energy_hartree` | `float?` | Single-point electronic energy (Hartree) |
| `zpe_hartree` | `float?` | Zero-point energy (Hartree); `null` for monoatomic |
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
| `freq_n_imag` | `int?` | Number of imaginary frequencies; `0` for stable species, `null` for monoatomic |
| `imag_freq_cm1` | `null` | Always `null` for non-TS species |

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
| `ess_versions` | `dict?` | ESS software versions, keyed by job type (`{sp\|opt\|freq\|neb: version_str, ...}`) |

Each input deck sits in the same directory as its log, under the ESS-specific
filename from `settings['input_filenames']`. Software with no entry in that map
(`gcn`, `torchani`, `mockter`, ...) yields `null`.

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
| `level_of_theory` | `dict?` | The Arkane level of theory the correction was applied at (a level dict); `null` when unknown |
| `total` | `dict` | `{value: float, unit: str}` — the applied correction total, in the unit the correction helper reported (defaulting to `hartree` for AEC and `kcal_mol` for BAC) |
| `components` | `list` | Native per-atom / per-bond decomposition. Always present; `[]` when no decomposition is available, and deliberately emptied for a BAC whose bonds are not all parameterized (a partial decomposition would not sum to `total`) |
| `parameter_table` | `dict?` | `{unit: str, values: {name: float, ...}}` — the parameter table ARC actually used. **Omitted** when the run has no such table, and for BAC additionally omitted unless `bac_type == 'p'`, so a Melius run never carries a BAC `parameter_table` |

### Rotor Scans

`rotor_scans` is always a list, `[]` for monoatomic or non-converged species. It holds
one record per successful 1D rotor whose scan log parsed cleanly; rotors that fail any
of those conditions — unsuccessful, multi-dimensional, or unparseable — are skipped
entirely, so the list can be shorter than `statmech.torsions` and the matching
torsion's `source_scan_key` is then `null`. `constraints` (held-fixed coordinates,
excluding the scanned coordinate itself) is **omitted** when none were found, and
`result.zero_energy_reference_hartree` is **omitted** when the parser reports none.

**`result.coordinate`** describes the scanned coordinate and the requested grid:

| Field | Type | Description |
|---|---|---|
| `coordinate_type` | `str` | Always `"dihedral"` |
| `atom_indices` | `list[int]` | The 4 atoms defining the dihedral, in source order |
| `index_base` | `int` | Always `1` — the atom indices are 1-based |
| `unit` | `str` | Always `"degree"` |
| `sample_count` | `int` | Number of parsed scan points, matching `len(samples)` |
| `symmetry_number` | `int?` | The rotor's symmetry number. **Omitted** unless ARC determined an integer symmetry of at least 1 |
| `requested_step_size` | `float?` | The step size the user requested, read back from the ESS log rather than inferred from point spacing. Gaussian only — other ESS raise `NotImplementedError` from the same parser, so this is **omitted** for them |
| `requested_start` | `float?` | The requested starting dihedral, taken from the geometry the scan was launched against. **Omitted** without a `requested_step_size` or a computable dihedral |
| `requested_end` | `float?` | `requested_start + requested_step_size * (sample_count - 1)`. Deliberately not wrapped into `[-180, 180]`, so a full rotation ends at `start + 360` rather than back at `start`. **Omitted** under the same condition as `requested_start` |

### Thermochemistry

`thermo` is `null` for non-converged species or species without thermo data.

| Field | Type | Description |
|---|---|---|
| `h298_kj_mol` | `float` | Standard enthalpy at 298 K (kJ/mol) |
| `s298_j_mol_k` | `float` | Standard entropy at 298 K (J/(mol K)) |
| `tmin_k` | `float` | Minimum temperature (K) |
| `tmax_k` | `float` | Maximum temperature (K) |
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
| `spin_multiplicity` | `int` | Spin multiplicity |
| `optical_isomers` | `int` | Number of optical isomers |
| `is_linear` | `bool` | Whether the molecule is linear |
| `external_symmetry` | `int` | External symmetry number |
| `point_group` | `str?` | Point group (e.g. `C2v`) |
| `rigid_rotor_kind` | `str` | `"linear"` or `"asymmetric_top"`. The builder has a third `"atom"` branch, but `statmech` is `null` for monoatomic species, so it never reaches `output.yml` |
| `harmonic_frequencies_cm1` | `list[float]?` | Harmonic frequencies (cm-1) as parsed. For TSs **every** negative (imaginary) frequency is dropped, not only the reaction mode — a TS that legitimately carries additional small imaginary modes (which `check_imaginary_frequencies` permits) loses those too, so the list can be shorter than the species' true mode count. Non-TS species are not filtered at all |
| `torsions` | `list` | Internal rotation data (see below) |

**`torsions`** entries (only successful rotors):

| Field | Type | Description |
|---|---|---|
| `symmetry_number` | `int` | Torsional symmetry number |
| `treatment` | `str` | `"hindered_rotor"` or `"free_rotor"` |
| `atom_indices` | `list[int]` | 4-atom dihedral defining atoms (1-indexed) |
| `pivot_atoms` | `list[int]` | 2-atom rotation axis (1-indexed) |
| `barrier_kj_mol` | `float?` | Torsional barrier height (kJ/mol) |
| `source_scan_key` | `str?` | The `rotor_scans[].key` (e.g. `"scan_rotor_3"`) this torsion was derived from. `null` when `rotor_scans` holds no record for that rotor, so the reference can never dangle |

---

## Transition States

`transition_states` is a list of entries that include **all species fields above**, plus:

| Field | Type | Description |
|---|---|---|
| `is_ts` | `true` | Always `true` |
| `freq_n_imag` | `int?` | `1` when converged, `null` otherwise |
| `imag_freq_cm1` | `float?` | Imaginary frequency (cm-1) |
| `chosen_ts_method` | `str?` | The TS search method that was selected |
| `successful_ts_methods` | `list[str]?` | All TS methods that succeeded |
| `ts_guesses` | `list[dict]` | Sanitized provenance for the chosen guess: `index`, `chosen`, `method`, and merged `method_sources` |
| `neb_log` | `str?` | Run-relative path to the NEB log. Taken from the run's `neb` path slot, falling back to the chosen TS guess's log when that guess's method is `orca_neb` |
| `gsm_log` | `str?` | Run-relative path to the selected GSM stringfile. Taken from the run's `gsm` path slot, falling back to the chosen TS guess's log when that guess's method is `xtb_gsm` |
| `irc_logs` | `list[str]` | Run-relative paths to IRC logs |
| `irc_log_directions` | `list[str?]` | Forward/reverse direction in lockstep with `irc_logs` |
| `irc_converged` | `bool?` | Whether IRC converged (`null` if IRC was not requested) |
| `rxn_label` | `str` | Reaction label this TS belongs to |
| `thermo` | `null` | Always `null` for transition states |

---

## Reactions

`reactions` is a list of entries, one per reaction.

| Field | Type | Description |
|---|---|---|
| `label` | `str` | Reaction label |
| `reactant_labels` | `list[str]` | Species labels of reactants |
| `product_labels` | `list[str]` | Species labels of products |
| `family` | `str?` | Reaction family |
| `multiplicity` | `int` | Reaction spin multiplicity |
| `ts_label` | `str` | Label of the associated transition state |
| `kinetics` | `dict?` | Fitted kinetics (see below); `null` if not computed |
| `long_kinetic_description` | `str` | ARC's verbose description of how the rate coefficient was obtained. **Omitted** (key absent) when the reaction carries no such description |

**`kinetics`**:

| Field | Type | Description |
|---|---|---|
| `A` | `float` | Pre-exponential factor |
| `A_units` | `str?` | Units of A |
| `n` | `float` | Temperature exponent |
| `Ea` | `float` | Activation energy |
| `Ea_units` | `str?` | Units of Ea |
| `Tmin_k` | `float` | Minimum fitted temperature (K) |
| `Tmax_k` | `float` | Maximum fitted temperature (K) |
| `dA` | `float?` | Uncertainty in A |
| `dn` | `float?` | Uncertainty in n |
| `dEa` | `float?` | Uncertainty in Ea |
| `dEa_units` | `str?` | Units of dEa |
| `n_data_points` | `int?` | Number of data points used in fitting |
| `tunneling` | `str` | The tunneling correction applied to the fitted `A`/`n`/`Ea`. Always present whenever `kinetics` is non-`null`: Arkane's parsed value when it carries one, otherwise the method ARC renders into the Arkane input template (currently `"Eckart"`) |
