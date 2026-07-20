# output.yml Schema Reference

The consolidated `output.yml` is written atomically to `<project_directory>/output/output.yml`
at the end of an ARC run. It contains all result data in a single file with run-relative paths
so downstream consumers (TCKDB, analysis scripts) need only one file.

Fields marked **nullable** may be `null` when the species did not converge,
the job was not requested, or the data is not applicable (e.g. monoatomic species).

---

## Quick Overview

```
output.yml
├── schema_version: "1.1"
├── project
├── arc_version
├── arc_git_commit?
├── arkane_git_commit?
├── datetime_completed
│
├── opt_level?
├── freq_level?
├── sp_level?
├── neb_level?
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
│       ├── sp_energy_hartree?, zpe_hartree?, opt_converged?
│       ├── coarse_opt_log?, coarse_opt_n_steps?, coarse_opt_final_energy_hartree?
│       ├── opt_n_steps?, opt_final_energy_hartree?
│       ├── freq_n_imag?, imag_freq_cm1?
│       ├── opt_log?, freq_log?, sp_log?
│       ├── opt_constraints?, freq_constraints?, sp_constraints?
│       │   └── [{coordinate_type, atom_indices, index_base, target_value?}, ...]
│       ├── rotor_scans?: []
│       │   └── key, source_log?, constraints?, result
│       │       ├── dimension, relaxed, zero_energy_reference_hartree?
│       │       ├── coordinate: {coordinate_type, atom_indices, index_base, unit, ...}
│       │       └── samples: [{source_index, angle_degrees,
│       │                      relative_energy_kj_mol,
│       │                      electronic_energy_hartree?, geometry_xyz?}, ...]
│       ├── energy_corrections?: []
│       │   └── correction_type, model, level_of_theory?, total,
│       │       components?, parameter_table?
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
│       ├── neb_log?, gsm_log?, irc_logs: [], irc_log_directions: [], irc_converged?
│       ├── neb_log?, gsm_log?, irc_logs: [], irc_converged?
│       └── rxn_label
│
└── reactions: []
    └── label, reactant_labels, product_labels, family?, multiplicity, ts_label
        └── kinetics?
            └── A, A_units?, n, Ea, Ea_units?, Tmin_k, Tmax_k
                dA?, dn?, dEa?, dEa_units?, n_data_points?
```

`?` = nullable. TS entries share all species fields but add IRC/NEB/method fields and always have `thermo: null`.

---

## Top-level

| Field | Type | Description |
|---|---|---|
| `schema_version` | `str` | Output contract version; `1.1` adds the optional evidence descriptor |
| `project` | `str` | ARC project name |
| `arc_version` | `str` | ARC version string |
| `arc_git_commit` | `str?` | ARC repo HEAD commit hash |
| `arkane_git_commit` | `str?` | RMG-Py (Arkane) repo HEAD commit hash |
| `datetime_completed` | `str` | Completion timestamp (`YYYY-MM-DD HH:MM`) |
| `tckdb_evidence` | `dict?` | Descriptor for `tckdb_evidence.json`, present only after the sidecar was written successfully |

The descriptor binds `output.yml` to the parser-neutral evidence sidecar with
`path`, `schema_name`, `schema_version`, and a shared UUID4 `document_id`.
ARC writes the sidecar first and `output.yml` second so consumers can reject a
stale or interrupted pair by comparing document IDs. The evidence file contains
best-effort, versioned Hessian, IRC, and GSM facts parsed by ARC; it contains no
TCKDB upload request objects.

## Levels of Theory

| Field | Type | Description |
|---|---|---|
| `opt_level` | `dict?` | Geometry optimization level (`method`, `basis`, `software`, ...) |
| `freq_level` | `dict?` | Frequency calculation level |
| `sp_level` | `dict?` | Single-point energy level |
| `neb_level` | `dict?` | NEB TS search level (only present if NEB was used) |
| `arkane_level_of_theory` | `dict?` | Composite level Arkane uses for energy corrections |
| `freq_scale_factor` | `float?` | Harmonic frequency scaling factor |
| `freq_scale_factor_source` | `str?` | Source of the scaling factor (`null` if user-provided) |
| `bac_type` | `str?` | Bond additivity correction type: `"p"`, `"m"`, or `null` |
| `atom_energy_corrections` | `dict?` | Per-atom corrections in Hartree (`{element: value, ...}`) |
| `bond_additivity_corrections` | `dict?` | Per-bond corrections in kJ/mol (`{bond: value, ...}`) |

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
| `ess_versions` | `dict?` | ESS software versions (`{label: version_str, ...}`) |

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
| `rigid_rotor_kind` | `str` | `"linear"` or `"asymmetric_top"` |
| `harmonic_frequencies_cm1` | `list[float]?` | Real harmonic frequencies (cm-1); imaginary mode excluded for TSs |
| `torsions` | `list` | Internal rotation data (see below) |

**`torsions`** entries (only successful rotors):

| Field | Type | Description |
|---|---|---|
| `symmetry_number` | `int` | Torsional symmetry number |
| `treatment` | `str` | `"hindered_rotor"` or `"free_rotor"` |
| `atom_indices` | `list[int]` | 4-atom dihedral defining atoms (1-indexed) |
| `pivot_atoms` | `list[int]` | 2-atom rotation axis (1-indexed) |
| `barrier_kj_mol` | `float?` | Torsional barrier height (kJ/mol) |

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
| `neb_log` | `str?` | Run-relative path to NEB log |
| `gsm_log` | `str?` | Run-relative path to the selected GSM stringfile |
| `neb_log` | `str?` | Run-relative path to NEB log (set when chosen TS method is `orca_neb`) |
| `gsm_log` | `str?` | Run-relative path to GSM stringfile (set when chosen TS method is `xtb_gsm`) |
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
