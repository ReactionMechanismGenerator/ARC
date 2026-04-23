# `sp_composite` examples

These inputs demonstrate the four YAML forms accepted by ARC's `sp_composite`
feature — composite single-point protocols for refined electronic energies
(HEAT-style focal-point analysis and CBS extrapolation).

| File | Demonstrates |
|---|---|
| `heat345q_preset/input.yml` | **Form 1** — preset by name (`HEAT-345Q`). Smallest possible composite input. |
| `heat345q_partial_override/input.yml` | **Form 2** — preset with partial override: swap one basis set on a single term. |
| `explicit_fpa/input.yml` | **Form 3** — fully explicit recipe, including a `cbs_extrapolation` term (Helgaker 2-pt correlation). |
| `per_species_override/input.yml` | **Form 4** — per-species override: one species keeps the project default, one opts out via `null`, one uses a species-specific protocol. |

## Running

Activate the ARC conda environment (`environment.yml`), then from the repo root:

    python ARC.py examples/Composite/heat345q_preset/input.yml

After the run finishes, a provenance notebook is generated at
`<project_directory>/output/sp_composite.ipynb`. Open it in Jupyter or VS Code
and select **Run All** — each section re-parses the actual QM output files via
`arc.parser.parse_e_elect` and re-evaluates the `CompositeProtocol` to verify
the final `e_elect` matches what ARC recorded in `output.yml`.

## A note on cost

The HEAT-style examples include `CCSDT` and `CCSDT(Q)` post-(T) corrections
that require the CFOUR (NCC module) or Molpro adapters to actually execute.
These are *illustrative*: the recipes are scientifically meaningful for small
molecules (4–6 atoms, tight TSs) but become prohibitive quickly. The minimal
`heat345q_preset` example uses `H2` and `O` as smoke-test species; adapt the
level of theory (or drop expensive terms via overrides) for larger systems.

For small methodological demos that do not require an expensive post-(T)
reference calculation, see `explicit_fpa/input.yml`, which shows the CBS
extrapolation form using only CCSD(T)/cc-pV{T,Q}Z.

## Units

`species.e_elect` is stored in kJ/mol throughout. The notebook and ARC log
display Hartree only at boundaries via division by `E_h_kJmol`
(≈ 2625.4996 kJ/mol/Hartree). The Arkane species file (under
`<project>/output/Species/<label>/arkane/species.py`) is rendered with a bare
`energy = <Hartree>` assignment when `sp_composite` is active — matching
Arkane's numeric-energy convention.

## More

Full documentation: `docs/source/advanced.rst`, section
*Composite single-point protocols (sp_composite)*.
