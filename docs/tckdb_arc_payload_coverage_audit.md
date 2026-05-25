# ARC → TCKDB Payload Coverage Audit

**Audit date:** 2026-05-14
**Branch:** `tckdb-imp`
**Scope:** Read-only audit. No code or schema changes. Compares what ARC has at
upload time (in `ARCSpecies` / `ARCReaction` / `JobAdapter` and in
`output.yml`) against what `arc/tckdb/adapter.py` actually sends to TCKDB, and
against what the TCKDB write API will accept (`backend/app/schemas/entities/*`
and `backend/app/schemas/fragments/*`).

`tckdb-client` is intentionally a thin transport/replay/auth layer and is **out
of scope**. All chemistry-aware mapping is in ARC.

---

## Summary

The ARC TCKDB adapter is mature for the **opt → freq → sp → thermo → kinetics**
spine, with first-class support for inline artifacts (logs + input), depends-on
graphs, level-of-theory projection, statmech (FSF + torsions), and
NEB/GSM-as-TS-guess `path_search_result` emission. Idempotency is solid.

There are, however, several places where ARC already holds scientifically
meaningful data that **never reaches TCKDB**, despite TCKDB having a typed
slot for it. The biggest gaps are:

1. **All-but-one conformer** — the bundle is hard-capped at one conformer; the
   `conformers` / `conformer_energies` list on `ARCSpecies` is dropped.
2. **Calculation run metadata** — server, queue, memory, cpu_cores, walltime,
   `ess_trsh_methods`, `attempted_queues`, fine/grid: not emitted, despite
   `calculation.parameters_json` / `parameters[]` being designed for exactly
   this.
3. **Kinetics qualifiers** — `tunneling_model` and `degeneracy` are accepted
   by `KineticsCreate` and present in ARC (`reaction.kinetics['tunneling']`,
   `reaction.family_own_reverse` + RMG family degeneracy) but not sent.
4. **TS auxiliaries** — `ts_checks`, `successful_methods`/`unsuccessful_methods`,
   `ts_report`, `chosen_ts_method`/`chosen_ts_list` provenance.
5. **Identity adjuncts** — `adjlist`, `unmapped_smiles` (canonical-vs-mapped),
   `optical_isomers` (only via statmech.is_linear/rigid_rotor_kind; chirality
   itself is dropped).
6. **T1 diagnostic** — used by ARC to flag multireference character, ignored
   on the wire.
7. **Transport — partial only.** `ARCSpecies.transport_data` exists as a
   `TransportData()` placeholder (`species.py:368`) but **no OneDMin adapter
   ships with ARC today** (`arc/job/adapters/` has no `onedmin.py`). The only
   field actually populated is `polarizability`, parsed as a side-effect of
   freq jobs (`scheduler.py:2677`). Sigma, epsilon, dipole, and rotational
   relaxation are never computed in the current codebase, so end-to-end
   transport upload is *not* a near-term win — it requires a OneDMin adapter
   first.

These are all *adapter-side or producer-side* gaps; TCKDB already accepts the
shapes.

---

## Current upload modes found

Three payload shapes are built by `arc/tckdb/adapter.py`:

| Mode | Entry function | Adapter lines | Endpoint | Used when |
|------|----------------|--------------|----------|-----------|
| **conformer** (legacy single-calc) | `submit_from_output()` → `_build_payload()` | 444–498, 2216–2237 | `/uploads/conformers` | Per-conformer, simplest shape. Always emits `species_entry` + `geometry` + a primary `opt` calc + optional freq/sp as `additional_calculations`. |
| **computed_species** (bundle) | `_build_computed_species_payload()` | 694–764 | `/uploads/conformers` (bundle route) | Full converged-species submission. Adds thermo, statmech, applied energy corrections, full depends-on graph, inline artifacts, rotor scans. Hard-capped at **one** `conformers[]` entry (test asserts `len(payload["conformers"]) == 1`, adapter_test.py:1226). |
| **computed_reaction** (bundle) | `_build_computed_reaction_payload()` | 1497–1645 | `/uploads/conformers` (bundle route) | Reactants + products as `species[]`, optional `transition_state{}` block, optional `kinetics[]` (modified-Arrhenius), reaction family, workflow tool release. TS-guess NEB/GSM emits `path_search_result`. |

Idempotency: `arc:<project>:<species>:<conformer>:<kind>:<sha256-payload>[:16]`
(`idempotency.py:18–90`); bundle local keys are unscoped for species (`opt`,
`freq`, `sp`, `opt_coarse`) and actor-scoped for reactions (`r0_opt`,
`p0_freq`, `ts_irc`).

---

## Coverage matrix

Status legend: **sent** · **adapter-gap** (ARC has it, adapter drops it) ·
**producer-gap** (ARC has it in memory but `output.yml` does not carry it) ·
**schema-blocker** (TCKDB cannot accept) · **intentional** · **unclear**.

| Domain | ARC source | TCKDB target | Status | Evidence: ARC | Evidence: adapter/TCKDB | Recommendation | Priority |
|---|---|---|---|---|---|---|---|
| Species identity | `record.smiles` | `species_entry.smiles` | sent | output.yml builder | adapter.py:2209 | — | — |
| Species charge | `record.charge` (default 0) | `species_entry.charge` | sent | species.py:365 | adapter.py:2210 | — | — |
| Species multiplicity | `record.multiplicity` | `species_entry.multiplicity` | sent | species.py:358 | adapter.py:2211 | — | — |
| Species TS flag | `record.is_ts` | `species_entry.species_entry_kind` | sent | species.py:393 | adapter.py:2212 | — | — |
| Adjacency list | `species.adjlist` | (no schema field — `species_entry.unmapped_smiles` is closest) | schema-blocker | species.py:357 | n/a | Note as future TCKDB feature if downstream consumers need adjlists. | P3 |
| Unmapped SMILES | derivable from `mol`/`atom_map` | `species_entry.unmapped_smiles` | adapter-gap | species.py via mol.to_smiles() | identity.py:36 (accepted) | Emit when mapped/canonical SMILES differ (TS atom-mapped reactions). | P2 |
| Stereo / chirality | `species.optical_isomers` (1 or 2) | `species_entry.stereo_kind` / `stereo_label` | adapter-gap (partial) | species.py:364 | adapter.py:2200–2206 comments "ARC has no reliable source; TCKDB backend derives stereo from 3D geometry" | Tighten the comment: ARC *does* have `optical_isomers`; even if `stereo_label` is left to the backend, surface chirality via `statmech` (already routes through `optical_isomers`-equivalent fields). Verify via test. | P2 |
| Final geometry | `record.opt_xyz` | `geometry.xyz_text` + `conformers[0].geometry.xyz_text` | sent | scheduler/output | adapter.py:2223, :787 | — | — |
| Initial geometry | `record.opt_input_xyz` | `calc.input_geometries[0]` (opt only) | sent (opt only) | output.yml | adapter.py:1088–1090 | Confirmed intentional: freq/sp use the optimized geom as input (1091–1106). | — |
| All conformers | `species.conformers[]` + `species.conformer_energies[]` | `conformers[]` (list, schema allows multiple) | adapter-gap | species.py:67–90 docstring; conformer fields populated by Scheduler | bundle hard-caps at one (adapter_test.py:1226) | Lift the one-conformer cap. Send each conformer with its `key=conf<idx>`, its own opt result + `final_energy_hartree`, and a `selected_conformer` marker (or `note: "selected"`) on the lowest. | **P1** |
| Conformer energies | `species.conformer_energies` (kJ/mol) | `calc.opt_result.final_energy_hartree` per conformer | adapter-gap | species.py:67–90 | only the selected conformer's final energy is sent | Convert kJ/mol→hartree at the boundary; one calc per conformer (depends_on coarse if applicable). | P1 |
| Monoatomic xyz | synthesized in output.yml | `geometry.xyz_text` | sent | output.py monoatomic special-case | adapter writes it through | Add regression test (see "Tests to add"). | P2 |
| Z-matrix / internal coords | `species.zmat` | (no current schema field) | schema-blocker | species.py | n/a | Note for future schema enhancement; XYZ is canonical so this is low-value. | P3 |
| **Opt calculation** | `record.opt_*` fields | `calculation{type=opt}` + `opt_result` | sent | output.yml | adapter.py:2708–2721 | — | — |
| Opt convergence | `record.opt_converged` | `opt_result.converged` | sent | output.py:939 | adapter.py:2716 | — | — |
| Opt final energy | `record.opt_final_energy_hartree` | `opt_result.final_energy_hartree` | sent | output.py:940 | adapter.py:2720 | — | — |
| Opt n_steps | `record.opt_n_steps` | `opt_result.n_steps` | sent | output.yml | adapter.py:2708–2721 | — | — |
| Opt coarse stage | `record.coarse_opt_*` | second `calculation{type=opt_coarse}` + `depends_on` | sent | output.yml | adapter.py:797–831 | — | — |
| **Freq calculation** | `record.freq_*` + `record.statmech.harmonic_frequencies_cm1` | `calculation{type=freq}` + `freq_result` | sent | output.yml | adapter.py:2752–2799 | — | — |
| n_imag | `record.freq_n_imag` | `freq_result.n_imag` | sent | output.yml | adapter.py:2746 | — | — |
| Imaginary freq | `record.imag_freq_cm1` | `freq_result.imag_freq_cm1` | sent | output.yml | adapter.py:2747 | — | — |
| ZPE | `record.zpe_hartree` | `freq_result.zpe_hartree` | sent | output.yml | adapter.py:2768 | — | — |
| Full freq list | `record.statmech.harmonic_frequencies_cm1` | `freq_result.modes[].frequency_cm1` | sent | output.py:1182 | adapter.py:2784–2790 | — | — |
| Mode metadata (reduced mass, IR intensity, symmetry label) | available in ESS log; not parsed into ARC | `freq_result.modes[].reduced_mass_amu` / `ir_intensity_km_mol` / `symmetry_label` | producer-gap | n/a | calculation.py freq result accepts these | Future enrichment via parser; deferable. | P3 |
| **SP calculation** | `record.sp_energy_hartree` | `calculation{type=sp}` + `sp_result` | sent | output.yml | adapter.py:2802–2818 | — | — |
| SCF stability check | `record` (not currently captured; some ESS expose it) | `calc.scf_stability` (status enum + lowest_eigenvalue) | producer-gap | not captured by ARC parsers | calculation.py:CalculationWithResultsPayload | If ARC starts emitting SCF stability (Gaussian `stable=opt`), wire it through. Defer. | P3 |
| **Rotor scans** | `species.rotors_dict` + per-scan `scan_result` mapping | `calculation{type=scan}` + `scan_result` (caller dict) | sent | species.py:67–90 | adapter.py:891–924, 1797–1812 | — | — |
| Rotor torsion fingerprint | `species.rotors_dict[*]['symmetry']`/`pivots`/`scan` | `statmech.torsions[]` with `coordinates[]`, `symmetry_number`, `treatment_kind`, `source_scan_calculation_id` | sent | output.py:1191–1224 | adapter.py:3601–3698 | — | — |
| Rotor barrier (`max_e` rel. to min, kJ/mol) | `species.rotors_dict[*]['max_e']` | (no direct schema field on `StatmechTorsionCreate`) | unclear | species.py:67–90 | statmech.py | If torsions carry `note`, surface "barrier=X kJ/mol" there; otherwise propose a `barrier_kj_mol` field on `StatmechTorsionCreate`. | P2 |
| Directed/ND rotor scans | `species.directed_rotors` + `directed_scan` dict | `scan_result` extension or new `scan_result.directed=true` | producer-gap | species.py:401 | adapter has no path | Defer; ND scan plumbing is unfinished upstream. | P3 |
| **IRC** | `irc_log` + parsed forward/reverse trajectories | `irc_result` (forward_trajectory + reverse_trajectory + direction) | sent | output.yml | adapter.py:4177–4275 | — | — |
| IRC per-point energies / coords | parsed into `forward_trajectory` / `reverse_trajectory` | `irc_result.points[]` (richer point-by-point schema) | adapter-gap | output.yml emits trajectories | calculation.py `IRCPointPayload` accepts `point_index`, `reaction_coordinate`, `electronic_energy_hartree`, `geometry`, `is_ts` | If output.yml IRC payload already has per-point data, emit `points[]` instead of (or in addition to) the trajectory pair, so TCKDB consumers can plot the IRC profile. | P2 |
| **NEB / GSM (TS-guess)** | `output['paths']['neb']` / `gsm` + `record.ts_guesses` chosen method | `calculation{type=path_search}` + `path_search_result{method, points[]}` (TS-guess only) | sent (TS-guess) | output.yml | adapter.py:1954–2046 | — | — |
| NEB/GSM image energies/coords | written to log paths; **not parsed** by ARC into structured `points[]` | `path_search_result.points[].electronic_energy_hartree` + `geometry` | producer-gap | output[label]['paths']['neb'/'gsm'] is just a log path | adapter.py emits `path_search` with bare `points` from log file | Add a producer-side parser for NEB/GSM stringfile → per-image `xyz` + energy. | P2 |
| NEB / GSM as standalone calc (not just TS-guess) | not wired | `calculation{type=path_search}` standalone | adapter-gap | — | path_search only emitted under TS-guess context | Document as intentional for v0; lift later when standalone path-search is meaningful for ARC. | P3 |
| **Level of theory** | `output_doc[<calc_level>]` (`method`, `basis`, `auxiliary_basis`, `cabs`, `dispersion`, `solvent`, `solvation_method`, `args`) | `calculation.level_of_theory.{method,basis,aux_basis,cabs_basis,dispersion,solvent,solvent_model,keywords}` | sent | output.yml level dicts | adapter.py:2890–2910 (projection); 2849–2888 (`args` → flattened `keywords`) | — | — |
| Method type / year / scheme level | ARC level dict has `method_type`, `year`, `solvation_scheme_level`, `compatible_ess` | (none of these have direct LoT slots) | intentional drop | output.yml | adapter.py:2890–2910 explicitly drops them | Confirm intent; consider routing `method_type` into a future LoT note. | P3 |
| **Software / version** | `output_doc[<level>].software`, `record.ess_versions[job_key]` | `calculation.software_release.{name,version}` | sent | output.yml | adapter.py:2347–2356 | — | — |
| ESS build/revision/release_date | not parsed by ARC | `SoftwareReleaseRef.{revision,build,release_date}` | producer-gap | n/a | software.py | Defer. | P3 |
| **Workflow tool release** | `output_doc.arc_version` + `output_doc.arc_git_commit` | `calculation.workflow_tool_release.{name="ARC",version,git_commit}` | sent | output.py:91–100 | adapter.py:2365–2373 | — | — |
| Arkane / RMG git commit | `arkane_git_commit` in output.yml | — (no second workflow tool field) | unclear | output.py:96 | adapter does not currently surface this | Either attach as `note` on the workflow_tool_release or accept as a second workflow tool release; defer. | P3 |
| **Calc run metadata** | `JobAdapter.{server, queue, attempted_queues, job_memory_gb, cpu_cores, max_job_time, run_time, fine, grid, ess_trsh_methods, job_id, job_status, restarted, times_rerun, additional_job_info}` | `calculation.parameters_json` *or* `calculation.parameters[]` (raw_key/raw_value structured observations) | adapter-gap (mostly producer-gap) | adapter.py:114–210 (set on JobAdapter at runtime) | adapter.py only emits `tckdb_origin` for sp-reused-from-opt (2382–2383); nothing else lands in `parameters_json` | output.yml producer needs to persist per-job: server, queue, memory_gb, cpu_cores, walltime, fine, ess_trsh_methods, restarted, times_rerun, job_id. Adapter then projects these into `parameters[]` with structured `(canonical_key, canonical_value, unit)`. | **P1** |
| Job adapter / engine name | `JobAdapter.job_adapter` | `calculation.parameters[]` `(canonical_key="job_adapter", …)` | adapter-gap | adapter.py | — | Same as above. | P1 |
| Job type | `JobAdapter.job_type` | already encoded in `calculation.type` | sent (indirectly) | — | — | — | — |
| Troubleshooting history | `JobAdapter.ess_trsh_methods` + `attempted_queues` | `calculation.parameters[]` or `calculation.note` (no schema slot) | adapter-gap / weak-schema | adapter.py:151, 182 | — | Surface as structured parameters; valuable for retrospectives. | P2 |
| Input / output / checkpoint paths on disk | `JobAdapter.{local_path,input_file_path,output_file_path,checkpoint_path}` | `calculation.artifacts[]` (binary content) | partial (logs+input only) | adapter.py base | adapter.py:1193–1242, 161–175 (config.py only registers `output_log`, `input`) | Extend `_INLINE_ARTIFACT_SOURCES` to also emit `checkpoint` (.chk) when present, gated by size budget. | P2 |
| Submit script / submit_time / end_time | `JobAdapter.{initial_time, final_time, run_time, submit_script_memory}` | `calculation.parameters[]` | adapter-gap | adapter.py:135–185 | — | Defer to the same producer-side metadata pass as above. | P2 |
| Fine grid / DFT integration grid | `JobAdapter.fine`, `JobAdapter.grid` | `calculation.level_of_theory.keywords` (where ARC currently routes `args`) **or** `parameters_json` | adapter-gap | adapter.py:155 | level_of_theory projection sees only `args` | Verify whether `fine=True`/grid setting is being threaded into the level `args` dict. If not, emit explicit parameters. | P2 |
| **Thermo H298/S298/Cp** | `record.thermo.{h298_kj_mol,s298_j_mol_k,thermo_points}` | `thermo.{h298_kj_mol,s298_j_mol_k,points[]}` | sent | output.py:1121–1158 | adapter.py:3029–3082 | — | — |
| Thermo NASA | `record.thermo.{nasa_low, nasa_high}` | `thermo.nasa.{a1..a7, b1..b7, t_low, t_mid, t_high}` | sent | output.py | adapter.py:3085–3147 | — | — |
| Thermo Tmin/Tmax | `record.thermo.{tmin_k,tmax_k}` | `thermo.{tmin_k,tmax_k}` | sent | output.py | adapter.py:3043–3049 | — | — |
| Thermo uncertainties | derivable from RMG/ARC, not currently set | `thermo.{h298_uncertainty_kj_mol, s298_uncertainty_j_mol_k}` | producer-gap | — | thermo.py:ThermoCreate | Defer. | P3 |
| Thermo source calculations | bundle-internal opt/freq/sp keys | `thermo.source_calculations[]` (`role`: `frequencies`, `single_point`, `geometry`) | sent | adapter.py:3069–3074 | — | — | — |
| RMG-estimated thermo | `species.rmg_thermo` | (no schema field) | intentional / schema-blocker | species.py:349 | n/a | Defer; comparison data, not provenance. | P3 |
| Long thermo description | `species.long_thermo_description` | `thermo.note` | adapter-gap | species.py:414 | thermo.py | One-line add; informational. | P3 |
| Applied energy corrections | `record.applied_energy_corrections` | `applied_energy_corrections[]` | sent | output.yml | adapter.py:2921–2989 | — | — |
| BAC / AEC values per bond/atom | `species.bond_corrections` and Arkane-applied AECs | embedded in `applied_energy_corrections[]` aggregate | sent (aggregate) | species.py:424 | adapter.py:2921–2989 | Per-bond breakdown could be surfaced as note; defer. | P3 |
| **Statmech base** | `record.statmech.{external_symmetry,is_linear,rigid_rotor_kind,statmech_treatment,point_group}` | `statmech.{external_symmetry,is_linear,rigid_rotor_kind,statmech_treatment,point_group}` | sent | output.py:1161–1188 | adapter.py:3496–3527 | — | — |
| Optical isomers / chirality | `species.optical_isomers` | currently routed into `statmech` indirectly via `rigid_rotor_kind`; no dedicated field | adapter-gap (semantic) | species.py:364 | adapter only sets the statmech rotational params | Either (a) accept `optical_isomers` on `StatmechCreate` (schema change) or (b) round-trip via `note`. Today the field is *dropped*. | P2 |
| Frequency scale factor | `output_doc.freq_scale_factor` + `freq_scale_factor_source` | `statmech.frequency_scale_factor_ref` | sent | output.py:56, 110–114 | adapter.py:3324–3423 | — | — |
| Torsion treatments | `record.statmech.torsions[]` | `statmech.torsions[]` (with `coordinates[]`, `symmetry_number`, `treatment_kind`, `source_scan_calculation_id`) | sent | output.py:1191–1224 | adapter.py:3601–3698 | — | — |
| Torsion barrier height | `species.rotors_dict[*]['max_e']` (kJ/mol) | (no schema field) | schema-blocker | species.py | statmech.py:StatmechTorsionCreate | See "Rotor barrier" row. | P2 |
| **Reaction topology** | `reaction.r_species[]` / `p_species[]` ordered | `reaction.species[]` with `key`, `role`, `participant_index` ordering | sent | reaction.py:107–108 | adapter.py:1610–1612, 1647–1820 | — | — |
| Reaction family | `reaction.family` | `reaction.reaction_family` | sent | reaction.py:113 | adapter.py:1625–1633 | — | — |
| Reaction family-own-reverse | `reaction.family_own_reverse` | (no schema slot) | schema-blocker | reaction.py:116 | — | Defer; routes into degeneracy logic. | P3 |
| Reaction degeneracy | derivable from `reaction.product_dicts` or RMG family | `kinetics.degeneracy: float` | adapter-gap | reaction.py:121 | kinetics.py:KineticsCreate | Emit if `reaction.kinetics` or RMG family supplies it. | **P1** |
| Tunneling model | `reaction.kinetics['tunneling']` (default Eckart) | `kinetics.tunneling_model: str` | adapter-gap | output.py:1652–1691; reaction.py:109 | kinetics.py:KineticsCreate | One-line add. | **P1** |
| Atom map | `reaction.atom_map` | (no schema slot today) | schema-blocker | reaction.py:122 | — | Useful for downstream consumers; propose schema add. | P2 |
| Reaction long description | `reaction.long_kinetic_description` | `kinetics.note` | adapter-gap | reaction.py:111 | kinetics.py | One-line add. | P3 |
| dH_rxn(298) | `reaction.dh_rxn298` (J/mol) | (no schema slot on `KineticsCreate`) | schema-blocker | reaction.py:118 | — | Computed by TCKDB from species thermo; not needed on the wire. | P3 |
| **TS xyz / charge / multiplicity** | `reaction.ts_species.{opt_xyz,charge,multiplicity}` | `transition_state{}` block | sent | reaction.py:132 | adapter.py:1583–1593 | — | — |
| TS guesses (all) | `reaction.ts_species.ts_guesses[]` | `path_search_result.points[]` (per-guess) | partial (only chosen method) | species.py:342 | adapter.py:1954–2046 | Emit one `path_search_result.point` per `TSGuess` (xyz + energy + `is_ts_guess=True`) rather than just the chosen method. | P2 |
| TS chosen method | `reaction.ts_species.chosen_ts_method` | encoded in `path_search_result.method` (NEB/GSM only) | sent (partial) | species.py:409 | adapter.py:1954 | For non-NEB/GSM methods (autotst, kinbot, heuristic), the method name is dropped. Route via `note`. | P2 |
| TS guesses tried (`chosen_ts_list`) | `species.chosen_ts_list` | (no slot) | schema-blocker | species.py:410 | — | Could be a `transition_state_entry.note`. Defer. | P3 |
| TS checks | `record.ts_species.ts_checks` (dict[str,bool]) | (no slot) | schema-blocker | species.py:375 | — | Propose a `TransitionStateEntry.checks` JSON column or notes-string. P2 ergonomically, but schema work first. | P2 |
| Successful / unsuccessful TS methods | `species.successful_methods`, `unsuccessful_methods` | (no slot) | schema-blocker | species.py:407–408 | — | Route into a TS-entry `note`; defer. | P3 |
| TS report | `species.ts_report` | (no slot) | schema-blocker | species.py:416 | — | Route via TS-entry `note`. | P3 |
| IRC endpoints / link to reactants&products | `species.irc_label` | `irc_result.points[]` + reaction-side linkage via TS↔reaction | partial | species.py:362 | — | Verify both forward and reverse endpoints are matched to the right side of the reaction (not just the TS). | P2 |
| **Kinetics A / n / Ea / units** | `reaction.kinetics.{A,n,Ea,A_units,Ea_units}` | `kinetics.{a,n,ea_kj_mol,a_units}` (Ea always kJ/mol via `arc_to_tckdb_ea_units`) | sent | reaction.py:109; output.py:1652–1691 | adapter.py:3975–4000 | — | — |
| Kinetics T-range | `reaction.kinetics.{Tmin_k,Tmax_k}` | `kinetics.{tmin_k,tmax_k}` | sent | output.py | adapter.py:4001–4009 | — | — |
| Kinetics uncertainties | `reaction.kinetics.{dA,dn,dEa,dEa_units}` | `kinetics.{a_uncertainty,n_uncertainty,ea_uncertainty_kj_mol}` | sent (conditional) | output.py | adapter.py:4010–4035 | — | — |
| Kinetics source calculations | bundle keys for reactant/product SPs + TS opt/freq/sp/irc | `kinetics.source_calculations[]` (`role`: reactant_energy, product_energy, ts_energy, freq, irc) | sent | adapter.py:4036–4137 | — | — | — |
| RMG library kinetics | `reaction.rmg_kinetics` | (no schema slot) | intentional / producer-gap | reaction.py:110 | — | Defer; reference rather than provenance. | P3 |
| **Transport (LJ collision data)** | `species.transport_data` is a `TransportData()` placeholder (`species.py:368`); only `polarizability` is actually populated, as a side-effect of freq parsing (`scheduler.py:2677`). **No OneDMin adapter ships in `arc/job/adapters/` today** — sigma, epsilon, dipole, rotrelaxcollnum are never computed. | `transport.{sigma_angstrom,epsilon_over_k_k,dipole_debye,polarizability_angstrom3,rotational_relaxation}` | **producer-side missing data + adapter-gap** | species.py:368, 2639–2698; scheduler.py:2677 | transport.py:TransportCreate — fully accepted; adapter has **no** transport code path (grep on adapter.py:transport → 0 hits) | Not a near-term win. Requires (i) a OneDMin job adapter to actually compute LJ params, then (ii) producer serialization, then (iii) adapter wiring. Today only `polarizability` is real data, and on its own (without sigma/epsilon) it would be a partial transport block. **Defer until OneDMin adapter exists.** | P2 |
| Transport source calculations | OneDMin job (does not exist as an adapter today) | `transport.source_calculations[]` | producer-gap | — | transport.py | Same OneDMin-adapter prerequisite. | P3 |
| **ARC version** | `arc.common.VERSION` = "1.1.0" | `workflow_tool_release.version` | sent | common.py:43 | adapter.py:2370 | — | — |
| ARC git commit | `output_doc.arc_git_commit` | `workflow_tool_release.git_commit` | sent | output.py:92 | adapter.py:2372 | — | — |
| Datetime started / completed | `output_doc.datetime_started`, `datetime_completed` | (no schema slot in workflow_tool_release; could go in calc parameters) | adapter-gap | output.py:97–100 | — | Surface project-level run window if useful. Defer. | P3 |
| **T1 diagnostic** | `species.t1` (Molpro) | (no schema slot) | schema-blocker | species.py:338 | — | Useful multireference indicator; propose `species_entry.t1_diagnostic` or route via statmech `note`. | P2 |
| Active orbitals / n_radicals | `species.active`, `species.number_of_radicals` | (no slot) | schema-blocker | species.py:360, 363 | — | Defer. | P3 |
| Conformer torsion fingerprint | derivable from `species.rotors_dict` per conformer | (no slot) | schema-blocker | species.py | — | Defer; would help dedupe at search time. | P3 |
| **Artifacts: output_log** | `record.{opt,freq,sp,coarse_opt,scan_rotors,irc,neb,gsm}_log` | `calc.artifacts[] {kind=output_log}` | sent | adapter.py:246–278 | adapter.py:1193–1242 | — | — |
| Artifacts: input | `record.opt_input_xyz_file` (opt only) | `calc.artifacts[] {kind=input}` (opt only) | sent (opt) | adapter.py:175 | adapter.py:1193–1242 | Extend to freq/sp/scan input files when those are persisted. | P3 |
| Artifacts: checkpoint (.chk, .npz) | `species.checkfile` + `JobAdapter.checkpoint_path` | `calc.artifacts[] {kind=checkpoint}` (would need a `ArtifactKind` enum entry; verify) | adapter-gap (and possibly schema) | species.py:367 | config.py:76 only registers `output_log`, `input` | Decide if checkpoint upload is worth the disk cost; gate by size. | P2 |
| Artifacts: NEB/GSM stringfile | `output['paths']['neb']`, `gsm` | `calc.artifacts[] {kind=output_log}` (already covered by output_log) | sent | — | — | — | — |

---

## Species and species_entry coverage

What flows: `smiles`, `charge`, `multiplicity`, `species_entry_kind`,
`electronic_state_kind="ground"`. What is dropped: `unmapped_smiles`,
`stereo_kind`/`stereo_label` (deliberately deferred to backend per
adapter.py:2200–2206; revisit since ARC *does* carry `optical_isomers`),
`adjlist` (no schema slot — schema-blocker), `term_symbol`,
`isotopologue_label`, `t1` (no schema slot).

The strongest existing producer-side data with no slot:
- **adjacency list** (schema-blocker)
- **T1 diagnostic** (schema-blocker)
- **electronic active-space metadata** (schema-blocker)

The strongest adapter-side gap:
- **unmapped_smiles** — accepted today by `SpeciesEntryIdentityPayload` and
  trivially derivable when ARC has an atom map.

---

## Geometry and conformer coverage

**Bundle hard-caps at one conformer** (`adapter_test.py:1226` asserts
`len(payload["conformers"]) == 1`). `ARCSpecies.conformers[]` and
`ARCSpecies.conformer_energies[]` carry the full set of screened conformers,
including their relative energies. None of them other than the selected one
reach TCKDB.

Recommendation: lift the cap. Emit each `ARCSpecies.conformers[i]` as a
separate `conformers[]` entry with key `conf<i>`, opt result populated from
its conformer-energy entry, and a marker on the selected one. Keys remain
deterministic and idempotent.

Initial vs final geometry: opt's `input_geometries[0]` is populated when
`record.opt_input_xyz` exists; freq/sp use the optimized geometry as input
(intentional, adapter.py:1091–1106).

Monoatomic special-case is wired through the producer's synthesized xyz; add
a regression test (see below).

---

## Calculation provenance coverage

What's sent on every calculation:

| Field | Source |
|---|---|
| `software_release.name` / `.version` | `output_doc[<level>].software` + `record.ess_versions[<job_key>]` |
| `workflow_tool_release.name` ("ARC") / `.version` / `.git_commit` | `output_doc.arc_version`, `.arc_git_commit` |
| `level_of_theory.method` / `.basis` / `.aux_basis` / `.cabs_basis` / `.dispersion` / `.solvent` / `.solvent_model` / `.keywords` (flattened `args`) | ARC level dict, projected via `_arc_level_to_tckdb_lot()` |
| `type`, `quality="raw"` | adapter-internal |
| `depends_on[]` | bundle-internal (opt_coarse → opt, freq → opt, sp → opt, scan → opt) |

**Missing despite TCKDB having a slot:**

- `parameters_json` / `parameters[]` is essentially empty (the adapter only
  fills `tckdb_origin` when sp is reused from an opt run). The schema accepts
  arbitrary `(raw_key, raw_value, canonical_key, canonical_value, unit)`
  observations — designed for exactly the kind of run metadata ARC has on the
  JobAdapter object (server, queue, memory, cpu, walltime, fine, grid,
  troubleshooting methods, job_id, restarted, times_rerun, additional_job_info).
- This is the single largest "structured data exists on both sides, nobody is
  carrying it" gap.

`software_release.{revision,build,release_date}`, `literature_id`,
`scf_stability`, and Arkane git commit are accepted by the schema but ARC
does not currently capture them.

---

## Calculation result coverage

**Opt** (`opt_result`): `converged`, `n_steps`, `final_energy_hartree` — all sent.

**Freq** (`freq_result`): `n_imag`, `imag_freq_cm1`, `zpe_hartree`, `modes[]` with
`mode_index`, `frequency_cm1`, `is_imaginary` — all sent.
Per-mode `reduced_mass_amu`, `force_constant_mdyne_angstrom`,
`ir_intensity_km_mol`, `raman_activity`, `symmetry_label` are accepted by the
schema but not parsed by ARC (producer-gap, low priority).

**SP** (`sp_result`): `electronic_energy_hartree` — sent.

**IRC** (`irc_result`): forward/reverse trajectory + direction are sent. The
richer `points[]` shape (`point_index`, `reaction_coordinate`,
`electronic_energy_hartree`, `geometry`, `is_ts`) is available and would let
TCKDB consumers plot the profile; ARC's output.yml emits the trajectory data
but the adapter does not transcode it into `points[]`. **Adapter-gap (P2).**

**Scan** (`scan_result`): caller-supplied dict, passed through opaquely. OK
for v0; if/when ARC structures scan output further, project into a typed
shape.

**Path-search** (`path_search_result`): wired for NEB/GSM **only as TS
guesses**, not as standalone calcs. NEB/GSM image energies + geometries
require a producer-side stringfile parser to populate `points[]` (currently
the log path is just attached as an artifact). **Producer-gap (P2).**

**SCF stability**: accepted by the schema; not captured by ARC.

---

## Dependency/link coverage

`depends_on[]` is well-modeled:
- `opt → opt_coarse` (`role="optimized_from"`) when both stages exist
- `freq → opt` (`role="freq_on"`)
- `sp → opt` (`role="single_point_on"`)
- `scan → opt` (`role="scan_parent"`)
- `ts_freq → ts_opt`, `ts_sp → ts_opt`, `ts_irc → ts_opt` (in reaction
  bundles, actor-scoped keys)

Source-calculation links:
- `thermo.source_calculations[]`: opt/freq/sp keys → roles
  (`geometry`/`frequencies`/`single_point`)
- `kinetics.source_calculations[]`: reactant_energy, product_energy,
  ts_energy, freq, irc
- `statmech.torsions[*].source_scan_calculation_id`: rotor scan parent
- `transport.source_calculations[]`: **never emitted** (transport block is
  never built)

---

## Thermo/statmech coverage

Thermo: H298/S298, Cp points, NASA low+high polys, Tmin/Tmax,
source_calculations — all sent. Uncertainties and `note` are unfilled
(`long_thermo_description` could populate `note`; adapter-gap P3).

Statmech: external_symmetry, is_linear, rigid_rotor_kind, statmech_treatment,
point_group, freq_scale_factor (via FSF ref to registry), torsions with
coordinates+symmetry_number+treatment_kind+source_scan_calculation_id — all
sent.

Statmech gaps:
- `optical_isomers` (chirality 1/2) — dropped at the adapter; not modeled in
  `StatmechCreate`. Either accept as a schema add or route via `note`.
- Torsion **barrier height** (`rotors_dict[i]['max_e']` kJ/mol) — accepted
  nowhere; schema add needed.
- `uses_projected_frequencies` — accepted by schema; not set by ARC.

---

## Kinetics/reaction/TS coverage

Kinetics fields sent: `a`, `a_units`, `n`, `ea_kj_mol`, `tmin_k`, `tmax_k`,
`a_uncertainty`, `n_uncertainty`, `ea_uncertainty_kj_mol`, `source_calculations`.

**Adapter-gaps (all P1, all one-liners):**
- `tunneling_model`: `reaction.kinetics['tunneling']` (default Eckart) is in
  the output.yml producer but not picked up by `_build_kinetics_block`.
- `degeneracy`: ARC reaction has `family_own_reverse` and access to RMG family
  degeneracy via `product_dicts`; nothing reaches the wire.
- `note`: `reaction.long_kinetic_description` is the natural fit.

TS coverage: TS xyz + charge + multiplicity + the (chosen) `path_search_result`
for NEB/GSM are sent. All other TS-guess methods (autotst, kinbot, heuristic)
have their method names dropped — adapter-gap P2. Per-guess xyz+energy
(`TSGuess[*].xyz` / `.energy`) is also lost. TS checks, TS report, successful
vs unsuccessful methods, `chosen_ts_list` — all dropped (no schema slot today;
TS-entry `note` is the closest place).

---

## IRC/scan/NEB/path-search coverage

| Mechanism | ARC has | adapter sends | gap |
|---|---|---|---|
| IRC trajectory | forward + reverse with energies/coords | forward_trajectory + reverse_trajectory | richer `points[]` shape not used |
| IRC endpoint reconciliation | `species.irc_label` | implicit via TS↔reaction | verify both endpoints map to correct reactant/product side |
| Rotor scans | per-rotor coords + energies via output.yml | `scan_result` (caller-supplied dict) + statmech torsions | none material at v0 |
| Directed/ND scans | `directed_rotors` + `directed_scan` | not emitted | producer + adapter both gap |
| NEB / GSM | log path only | `path_search_result` for TS-guess; no `points[]` data | producer-side parser missing |
| GSM stringfile | log file only | attached as artifact | structural parse missing |

---

## Artifact coverage

Implemented kinds (config.py:76): `output_log`, `input`.

Per-calc dispatch (`_resolve_log_field`, adapter.py:246–278):
- opt → `opt_log`
- freq → `freq_log`
- sp → `sp_log`
- opt_coarse → `coarse_opt_log`
- irc → `irc_log` (or `neb_log` / `gsm_log` for TS-guess)
- scan → `scan_rotors_log`

For `input`: only `opt_input_xyz_file` is wired. freq/sp/scan have no input
file attached. Adding them would be cheap.

Checkpoint files (Gaussian `.chk`, frequency `.npz`, etc.) are not attached
despite ARC keeping a `checkfile` field on `ARCSpecies`. Whether to attach
them is a size/value tradeoff (probably gate behind config.size_budget).

---

## Known producer-side issues

1. **No OneDMin adapter** — `arc/job/adapters/` has no `onedmin.py`. `ARCSpecies.transport_data` is created as a default `TransportData()` placeholder and only `polarizability` is populated (from freq parsing, `scheduler.py:2677`). End-to-end transport upload is blocked by the absence of the LJ-parameter producer, not by the adapter.
2. **Only one conformer is persisted** — `species.conformers[]` / `conformer_energies[]` are in memory but only the selected conformer's geometry/energy lands in output.yml in a TCKDB-shaped form.
3. **NEB/GSM stringfile not parsed** — only the log path is recorded; no structured `points[]` data.
4. **Adjacency list not in output.yml** — `species.adjlist` is in memory; not serialized (also no TCKDB slot, so this is dual-gapped).
5. **Per-job run metadata not in output.yml** — server, queue, memory, cpu, walltime, fine, ess_trsh_methods, attempted_queues, job_id, restarted, times_rerun, additional_job_info exist only on the in-memory `JobAdapter` object.
6. **IRC point-level structured data** — output.yml emits trajectories; per-point `(reaction_coordinate, energy, geometry, is_ts)` is not surfaced.
7. **Rotor barrier height (`max_e`)** — set on `species.rotors_dict[i]` but not pulled into the output.yml torsions block.
8. **Frequency-mode auxiliaries** (reduced mass, IR intensity, symmetry label) — present in ESS logs but not parsed.
9. **T1 diagnostic** — read from Molpro into `species.t1`; not serialized.
10. **TSGuess geometries/energies** — `species.ts_guesses[].xyz` and `.energy` are in memory; only the chosen method's NEB/GSM path is structured.

---

## Adapter-side issues

1. **Conformer cap** — bundle emits exactly one `conformers[]` entry even when the schema accepts a list; lift the cap to send all converged conformers with a "selected" marker.
2. **`calculation.parameters_json` is essentially unused** — `tckdb_origin` is the only key the adapter currently sets. The whole JobAdapter run-metadata surface (memory, cpu, walltime, server, queue, fine/grid, troubleshooting, restarted) has a place to land and isn't being routed there.
3. **Kinetics block drops `tunneling_model` and `degeneracy`** — both accepted by `KineticsCreate`, both present (or trivially derivable) in ARC. One-liners.
4. **`unmapped_smiles` never set** — accepted on `SpeciesEntryIdentityPayload`; computable from `mol` + `atom_map`.
5. **Stereo/chirality routed nowhere** — `ARCSpecies.optical_isomers` is in memory; statmech block does not surface it.
6. **TS-guess method dropped for non-NEB/GSM** — autotst/kinbot/heuristic guess methods leave no trace on the wire. Easiest fix is a `note` on the TS entry.
7. **IRC `points[]` not emitted** — adapter only emits `forward_trajectory` / `reverse_trajectory`. The richer schema would help downstream plotting.
8. **`input` artifact only for opt** — freq/sp/scan input files are skipped.
9. **`workflow_tool_release` is single-valued** — `arkane_git_commit` from output.yml does not surface; either add as note or accept that Arkane provenance lives in `software_release` of the freq calc.
10. **No standalone `path_search` calc type** — only TS-guess. If ARC ever supports standalone NEB/GSM scans, the path is closed today.

---

## TCKDB-side blockers (schema-only)

These are real gaps that *cannot* be closed by ARC alone — TCKDB schema work
is required first.

1. **Adjacency list** — `species_entry` has no `adjlist` slot. `unmapped_smiles` is close but not equivalent.
2. **T1 diagnostic** — no slot on `species_entry`; would help multireference flagging.
3. **Z-matrix / internal coords** — no schema slot.
4. **Atom map** — no slot on `reaction` / `reaction_entry_structure_participant`.
5. **TS checks / TS report / TS guess history** — no structured slot on `transition_state_entry`; today must round-trip via `note`.
6. **Torsion barrier height (`max_e` in kJ/mol)** — `StatmechTorsionCreate` has no `barrier_kj_mol`.
7. **`optical_isomers` (chirality 1/2)** — not on `StatmechCreate` and not first-class on `species_entry` (only `stereo_kind`/`stereo_label`).
8. **Active orbitals / number of radicals** — no slot.
9. **Reaction `family_own_reverse`** — no slot (informational; routes into degeneracy logic).
10. **Standalone path_search owner** — `path_search_result` is currently wired only under TS-guess; if path-search is ever its own first-class calc, the owner side needs a story.

---

## Recommended implementation order

1. **PR #1 — Kinetics qualifiers (P1, quick win).** Wire `tunneling_model` (from `reaction.kinetics['tunneling']`) and `degeneracy` (from RMG family / `reaction.product_dicts`) into `_build_kinetics_block`. Add `note` from `reaction.long_kinetic_description`. ~10-line adapter change.
2. **PR #2 — All conformers, not one (P1).** Lift the bundle's one-conformer cap. Send each `ARCSpecies.conformers[i]` as a `conformers[i]` entry with its own opt result + energy. Mark the selected conformer (`note: "selected"` or a dedicated `is_selected` flag if added schema-side). Update `adapter_test.py:1226` to expect N conformers. Verify first that output.yml carries per-conformer data — if not, this becomes a dual producer+adapter change.
3. **PR #3 — Run-metadata into `parameters[]` (P1).** Producer threads JobAdapter metadata (server, queue, memory_gb, cpu_cores, walltime, fine, ess_trsh_methods, attempted_queues, job_id, restarted, times_rerun) into output.yml per job. Adapter projects into `calculation.parameters[]` with stable canonical_keys.
4. **PR #4 — Identity & stereo (P2).** Set `species_entry.unmapped_smiles` when available. Route `ARCSpecies.optical_isomers` (and `is_linear` / `rigid_rotor_kind` consistency) through statmech.
5. **PR #5 — IRC and TS richness (P2).** Emit `irc_result.points[]` from the IRC trajectories the producer already has. Emit a `path_search_result.point` per `TSGuess` (xyz + energy + `is_ts_guess=True`) regardless of method; route TS checks / chosen method / report into TS-entry `note`.
6. **PR #6 — Producer parsers (P2).** NEB/GSM stringfile → structured `points[]`. Rotor-scan `max_e` → torsion barrier (gated on schema add). Frequency-mode aux fields.
7. **PR #7+ — Schema requests on TCKDB side.** Adjlist, T1, atom map, TS checks, torsion barrier, optical_isomers, standalone path_search owner.
8. **Deferred — Transport end-to-end.** Requires a OneDMin job adapter to land in `arc/job/adapters/` first. Until then, `transport_data` carries only polarizability (which is not sufficient on its own — `TransportCreate` requires `sigma_angstrom` and `epsilon_over_k_k` to be set together or both omitted).

---

## Tests to add

For each P0/P1 gap, propose a regression test in
`arc/tckdb/adapter_test.py` (or, where producer-side, in
`arc/output_test.py`):

```text
test_all_conformers_emitted_in_bundle:
    species.conformers has N>1 entries → payload["conformers"] has N entries,
    keys are conf0..conf{N-1}, exactly one carries the "selected" marker, each
    has its own opt_result.final_energy_hartree.

test_monoatomic_converged_species_emits_xyz:
    monoatomic species → geometry.xyz_text non-empty + conformers[0]
    geometry present, opt_result.final_energy_hartree set.

test_calculation_run_metadata_in_parameters:
    after producer thread-through, each calculation has parameters[] entries
    with canonical_keys = {server, queue, memory_gb, cpu_cores, walltime_h,
    fine, ess_trsh_methods, restarted}. Stable values across reruns.

test_kinetics_tunneling_and_degeneracy_emitted:
    reaction with kinetics['tunneling']='Eckart' and degeneracy=2 → kinetics
    block has tunneling_model='Eckart' and degeneracy=2.

test_unmapped_smiles_set_when_mapping_present:
    reaction TS with atom_map → reactant species_entry.unmapped_smiles is set
    when it differs from species_entry.smiles.

test_ts_guess_methods_all_emit_path_search_points:
    reaction with N TSGuesses across [autotst, kinbot, gsm, neb] → each guess
    appears as a path_search_result.point (xyz + energy + is_ts_guess=True),
    regardless of method.

test_thermo_uploads_link_source_calculations:
    thermo.source_calculations[] roles cover {geometry, frequencies,
    single_point} and reference bundle keys that exist in conformers[*].

test_freq_n_imag_and_imag_freq_emitted:
    freq calc → freq_result.n_imag and imag_freq_cm1 set; modes[] has full
    frequency list.

test_irc_emits_points_block:
    after PR #6, payload["transition_state"]["additional_calculations"][irc]
    .irc_result.points[] is non-empty with point_index, electronic_energy,
    geometry, and at least one is_ts=True point.

test_artifacts_attach_to_correct_calculation:
    each artifact's owner key matches the calc that produced it (opt_log
    only on opt calc; freq_log only on freq; etc.). No duplicates.

test_adapter_emits_stable_local_keys_for_replay:
    two adapter runs on the same output.yml produce identical local keys
    (opt, freq, sp; r0_opt, p0_freq, ts_irc) and identical idempotency keys.
```

---

## Top 10 gaps

1. **Only one conformer per bundle** (P1) — drops the screened conformer set; TCKDB `conformers[]` already accepts a list.
2. **`calculation.parameters_json` / `parameters[]` empty** (P1) — JobAdapter run metadata (server/queue/memory/cpu/walltime/trsh/fine) never lands despite the schema being designed for it.
3. **Kinetics `tunneling_model` not sent** (P1) — one-liner, schema accepts it.
4. **Kinetics `degeneracy` not sent** (P1) — derivable from RMG family.
5. **`unmapped_smiles` not sent** (P2) — accepted by schema; trivial when atom_map exists.
6. **`optical_isomers` / chirality dropped** (P2) — adapter drops it explicitly.
7. **IRC `points[]` not emitted** (P2) — only trajectories.
8. **TS-guess methods other than NEB/GSM dropped** (P2) — autotst/kinbot/heuristic leave no trace.
9. **No `input` artifact for freq/sp/scan** (P3) — only opt has one.
10. **Transport block never sent** (P2, deferred) — TCKDB accepts it, but ARC has no OneDMin adapter to produce LJ params; only `polarizability` is populated today. Not actionable until a OneDMin adapter exists.

## Top 5 quick wins

1. **`kinetics.tunneling_model` + `degeneracy` + `note`** — one PR, three lines each in `_build_kinetics_block`.
2. **`thermo.note` from `long_thermo_description`** — one-line.
3. **`species_entry.unmapped_smiles`** — derivable at the boundary; small.
4. **Stop dropping `optical_isomers`** — even if the schema slot is debated, surface it via `statmech.note` today.
5. **Reaction `note` from `long_kinetic_description`** — one-line.

## Top 5 schema/API blockers (real TCKDB-side work)

1. **`species_entry.adjlist`** — currently no slot.
2. **`statmech_torsion.barrier_kj_mol`** — already-computed data with nowhere to land.
3. **`species_entry.t1_diagnostic`** (or analogous multireference flag).
4. **`reaction_entry.atom_map`** — needed for cross-tool consumers.
5. **`transition_state_entry.checks` + `ts_report` (JSON or note column)** — current TS-entry shape can't represent the search history.

## Recommended first PR

**Kinetics qualifiers.** Adapter-only change in `_build_kinetics_block`
(`adapter.py:3921–4041`):
- `tunneling_model` ← `reaction.kinetics['tunneling']` (default `Eckart`
  when missing).
- `degeneracy` ← either `reaction.kinetics['degeneracy']` if populated by
  the producer, or recover from `reaction.product_dicts` / RMG family.
- `note` ← `reaction.long_kinetic_description` when present.

Two regression tests in `adapter_test.py`:
`test_kinetics_tunneling_and_degeneracy_emitted` and
`test_kinetics_note_from_long_description`. Zero schema work. ~10 LOC change.
Smallest possible step that ships real value.

## Recommended second PR

**Lift the one-conformer cap.** Verify first that `output.yml` carries
per-conformer geometry+energy data (it currently emits only the selected
conformer's structured fields; if so, that becomes a small producer-side
prerequisite). Then adapter-side: emit each `ARCSpecies.conformers[i]` as
its own `conformers[]` entry with a deterministic key (`conf{i}`),
per-conformer opt result populated from `ARCSpecies.conformer_energies[i]`,
and a marker on the selected conformer.  Update `adapter_test.py:1226` and
add `test_all_conformers_emitted_in_bundle`. Idempotency keys remain stable
because conformer keys are deterministic.

---

*End of audit.*
