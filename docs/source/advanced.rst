.. _advanced:

Advanced Usage
==============

This page collects the controls most users need after the first successful run:
levels of theory, job selection, ESS routing, resource control, rotor scans,
transition-state adapters, restarts, and troubleshooting.

.. _flexXYZ:

Flexible Coordinate Input
-------------------------

The ``xyz`` field of an ``ARCSpecies`` can be:

* a multiline XYZ string;
* a list of XYZ strings;
* a path to an XYZ file;
* a path to a supported ESS input or output file;
* a path to ARC conformer files generated before or after optimization.

Example:

.. code-block:: yaml

   species:
     - label: TS1
       is_ts: true
       xyz:
         - guesses/ts1_guess_1.gjf
         - guesses/ts1_guess_2.out
         - |
           C      0.000000    0.000000    0.000000
           H      0.000000    0.000000    1.089000

Job Types
---------

ARC recognizes these current job type keys:

* ``conf_opt`` - conformer optimization;
* ``conf_sp`` - conformer single-point jobs;
* ``opt`` - geometry optimization;
* ``fine`` - fine-grid optimization;
* ``freq`` - frequency calculation;
* ``sp`` - single-point energy;
* ``rotors`` - rotor scans;
* ``irc`` - intrinsic reaction coordinate;
* ``orbitals`` - molecular orbitals;
* ``onedmin`` - Lennard-Jones / OneDMin workflow;
* ``bde`` - bond dissociation energy workflow.

Older input aliases are still normalized in code: ``fine_grid`` maps to
``fine`` and ``lennard_jones`` maps to ``onedmin``. Prefer the current names in
new inputs.

Run one job family:

.. code-block:: yaml

   specific_job_type: sp

When ``specific_job_type`` is set, it takes precedence over ``job_types``.

.. _levels:

Levels of Theory
----------------

The fastest way to specify a common workflow is ``level_of_theory``:

.. code-block:: yaml

   level_of_theory: CCSD(T)-F12/cc-pVTZ-F12//wb97xd/def2tzvp

This means:

* optimize, frequency, and scan jobs use ``wb97xd/def2tzvp``;
* single-point jobs use ``CCSD(T)-F12/cc-pVTZ-F12``.

A single non-composite method applies to opt, freq, scan, and sp:

.. code-block:: yaml

   level_of_theory: wb97xd/def2svp

A composite method is specified without a slash:

.. code-block:: yaml

   level_of_theory: CBS-QB3

Use job-specific keys when you need more control:

.. code-block:: yaml

   conformer_opt_level:
     method: b3lyp
     basis: 6-31g(d,p)
     dispersion: empiricaldispersion=gd3bj

   opt_level: wb97xd/def2tzvp
   freq_level: wb97xd/def2tzvp
   sp_level:
     method: DLPNO-CCSD(T)-F12
     basis: cc-pVTZ-F12
     auxiliary_basis: aug-cc-pVTZ/C
     cabs: cc-pVTZ-F12-CABS
     software: orca

Do not put Arkane correction years into QC method names such as
``wb97xd32023``. Use ``arkane_level_of_theory.year`` when you need a specific
Arkane correction year:

.. code-block:: yaml

   arkane_level_of_theory:
     method: b97d3
     basis: def2tzvp
     year: 2023

ESS-Specific Arguments
----------------------

Use ``args`` for extra ESS keywords or blocks:

.. code-block:: yaml

   opt_level:
     method: wb97xd
     basis: def2tzvp
     software: gaussian
     args:
       keyword:
         general: iop(99/33=1)

For multiline blocks:

.. code-block:: yaml

   sp_level:
     method: dlpno-ccsd(t)
     basis: def2tzvp
     software: orca
     args:
       block:
         general: |
           %scf
             MaxIter 500
           end

Multireference Methods (MRCI)
-----------------------------

To request a multireference calculation such as MRCI, specify any of the
following on ``sp_level``. A "simple" MRCI computation:

.. code-block:: yaml

   sp_level: MRCI/cc-pVTZ

Explicitly correlated (F12) calculations improve basis-set convergence and are
only available through Molpro:

.. code-block:: yaml

   sp_level:
     method: MRCI-F12
     basis: cc-pVTZ-F12

You can also specify a chain of jobs (supported in Molpro and Orca) so that the
MRCI calculation uses the orbitals of the previous job. For example, to perform an
MRCI calculation on CASSCF orbitals:

.. code-block:: yaml

   sp_level:
     method: MP2_CASSCF_MRCI
     basis: aug-cc-pVTZ

This chain, separated by underscores, performs an HF calculation (by default, no
need to specify), an MP2 calculation, then a CASSCF calculation, and finally an
MRCI calculation on the CASSCF orbitals. Requesting an MRCI job causes ARC to first
automatically spawn a Molpro CCSD/cc-pVDZ job to identify the active space for the
MRCI calculation. If the subsequent job is spawned in Orca, the active space is
used; if it is spawned in Molpro, the entire space is currently considered (the
active space is not determined explicitly). It is therefore recommended to set the
``levels_ess`` dict in settings so that ``MRCI`` jobs run in Orca, and ``F12`` and
``CCSD`` jobs run in Molpro.

ARC extracts active-space parameters from the Molpro CCSD output file to guide the
subsequent calculation:

* **Active electrons** are obtained by subtracting the net charge and the core
  electrons (estimated as 2 per heavy atom) from the total nuclear charge:

  .. math:: N_{active} = Z_{total} - Q_{net} - (2 \times N_{heavy})

* **Active orbitals** are determined by summing the counts of "closed-shell" and
  "active" orbitals reported in the output.

The active-space routine returns a dictionary containing the ``'e_o'`` tuple
(electrons, orbitals) alongside lists of occupied (``'occ'``) and closed-shell
(``'closed'``) orbitals per irreducible representation.

Solvation
---------

Solvation is specified on a level of theory with the top-level fields
``solvation_method`` and ``solvent``:

.. code-block:: yaml

   opt_level:
     method: wb97xd
     basis: def2tzvp
     software: gaussian
     solvation_method: pcm
     solvent: diethylether

Support is adapter-dependent. Gaussian, ORCA, and xTB currently have solvation
handling in their job adapters; always choose method and solvent names in the
format expected by the selected ESS.

Composite single-point protocols (``sp_composite``)
---------------------------------------------------

``sp_composite`` expresses the final electronic energy of each stationary point
as a sum of contributions computed at *different* levels of theory — a
HEAT-style focal-point analysis. This is distinct from the legacy
``composite_method`` (which means a Gaussian-style single-job composite like
``CBS-QB3``); the two are mutually exclusive at the project level.

**When is it for you?**
When a single level of theory is insufficient for the accuracy you need on a
transition state. A typical motivation: ``CCSD(T)-F12/cc-pVTZ-F12`` wells agree
with ATcT, but TS barriers miss experiment by several kJ/mol. Adding small
post-(T) corrections (``δ[CCSDT]``, ``δ[CCSDT(Q)]``), plus core-valence and
scalar-relativistic terms, closes the gap without any empirical fitting.

**YAML forms.**
Forms 1–4 below are the core input shapes (preset by name, preset with partial
override, fully explicit recipe, and per-species override); forms 5–8 are
worked variants of these shapes for specific protocols.

**Form 1 — preset by name.** The quickest path::

    project: h2o_heat345q
    sp_composite: HEAT-345Q
    species:
      - label: H2O
        smiles: O

ARC ships the following presets in ``arc/level/presets.yml``:

*HEAT family (Tajti / Bomble / Stanton lineage):*

* ``HEAT-345`` — HEAT-style recipe inspired by Tajti et al. (see references).
  Includes δ[CCSDT], δ_CV (core-valence, all-electron CCSD(T)/cc-pCVTZ vs
  frozen-core), and δ_rel (DKH2 scalar-relativistic CCSD(T)/cc-pVTZ-DK).
* ``HEAT-345Q`` — HEAT-345 plus a δ[CCSDT(Q)] correction.
* ``HEAT-345_noC`` / ``HEAT-345Q_noC`` — same as the corresponding HEAT
  variant but with the **δ_CV** (core-valence) correction omitted. The omission
  is part of the preset name and reference string so users can cite the
  protocol honestly when the all-electron leg is unavailable on their ESS.
  Use these when targeting an ESS without a clean Molpro-style
  ``core,...`` directive (or when the core-valence contribution is known to
  be negligible — typically < 0.5 kJ/mol for first-row systems).
* ``HEAT-345QP`` — HEAT-345Q extended with full quadruples (δ[CCSDTQ]) and
  perturbative pentuples (δ[CCSDTQ(P)]). The δ_QQ and δ_P legs route
  through the MRCC interface — modern Molpro builds with MRCC linked in
  accept ``ccsdtq`` and ``ccsdtq(p)`` via the same path used for
  ``ccsdt`` / ``ccsdt(q)`` in HEAT-345Q. CFOUR-NCC is an alternative back
  end. A plain Molpro install without MRCC cannot run these sub-jobs.
* ``HEAT-456Q`` — same correction stack as ``HEAT-345Q`` but with a tighter
  base. The published HEAT-456 series uses cardinals {Q,5,6} for the HF /
  CCSD(T) CBS reference; the ARC adaptation pins the anchor to
  ``CCSD(T)-F12/cc-pVQZ-F12`` (single-anchor approximation of that CBS limit).

*W\ :sub:`n` family (Karton/Martin / Boese):*

* ``W2`` / ``W2-F12`` — high-quality CCSD(T) anchor + δ_CV + δ_rel. The
  ``-F12`` variant uses ``CCSD(T)-F12/cc-pVQZ-F12`` for near-CBS quality
  from a single SP. The non-F12 variant uses ``CCSD(T)/aug-cc-pVQZ``.
* ``W3`` / ``W3-F12`` — W2 + δ[CCSDT]. *Note:* there is no canonical
  primary publication titled "W3-F12"; ARC's preset is an extension by
  analogy to the published W2-F12 (see references below). Cite as
  "W3-F12 (ARC adaptation)".
* ``W4`` / ``W4-F12`` — W3 + δ[CCSDT(Q)] + δ[CCSDTQ]. The δ_QQ leg goes
  through the MRCC interface (Molpro-with-MRCC or CFOUR-NCC) — see the
  note under HEAT-345QP above; the same back-end requirement applies.

*Focal-point analysis:*

* ``FPA-min`` — minimal focal-point recipe using **CBS-as-base**: the absolute
  energy is a two-point Helgaker ``X^-3`` extrapolation of the
  ``CCSD(T)/cc-pVTZ`` + ``cc-pVQZ`` **total** energies, with a δ[CCSDT]
  correction on top. Note the ``X^-3`` form was derived for the correlation
  energy; applying it to total energies technically mis-treats the
  exponentially-converging HF component. At {T,Q} cardinals the residual is
  small and this is common practice — the preset's reference string states
  this explicitly so users can cite it honestly.

.. note::

   The W\ :sub:`n` family in ARC is a **single-anchor adaptation** of the
   canonical Karton/Martin protocols: the W\ :sub:`n` HF/CCSD/(T) basis-cardinal
   CBS extrapolations are absorbed into the anchor SP rather than being
   evaluated as separate stacked terms. This is faithful to the W\ :sub:`n`
   spirit (high-quality CCSD(T) anchor + post-(T) / CV / rel corrections)
   but not byte-identical to the published prescription. When citing, use
   "W2 (ARC adaptation)" / "W4-F12 (ARC adaptation)" rather than the
   bare protocol name to avoid implying a strict reproduction.

**ESS syntax for δ_CV and δ_rel.** The HEAT presets shipped here target the
**Molpro** adapter:

* δ_CV — all-electron CCSD(T)/cc-pCVTZ via Molpro's ``core,0,...`` directive
  (``args.keyword.core: 'core,0,0,0,0,0,0,0,0;'``). Trailing zeros are
  harmless for lower-symmetry point groups.
* δ_rel — DKH2 scalar-relativistic CCSD(T)/cc-pVTZ-DK via the canonical
  Molpro directive ``SET,DKHO=2`` (passed as
  ``args.keyword.dkho: 'SET,DKHO=2;'``). The Molpro manual
  (https://www.molpro.net/manual/doku.php?id=relativistic_corrections)
  explicitly recommends ``DKHO`` over the legacy ``DKROLL``. The directive
  must appear *before* ``int;`` so the integrals are evaluated with the
  DK-transformed Hamiltonian.

Other ESSes need different keywords; pointing a HEAT-345 / HEAT-345Q preset
at, say, the CFOUR or Orca adapter for those SPs will write the wrong
directive. Until per-ESS preset families ship, either supply an explicit
recipe or use a ``_noC`` variant.

**Form 2 — preset with partial override.** Replace specific fields of named
terms in the preset::

    sp_composite:
      preset: HEAT-345Q
      overrides:
        delta_T:
          high: {method: ccsdt, basis: cc-pVTZ}

The override dict keys are term labels (``base``, ``delta_T``, ``delta_Q``,
``delta_CV``, ``delta_rel``, ...). Unknown target labels raise ``InputError``.

**Form 3 — fully explicit recipe, with a CBS extrapolation as the base.** No
preset, complete control. This is the canonical focal-point shape: the
absolute energy is the CBS-extrapolated value, and δ-corrections stack on
top::

    sp_composite:
      reference: "My recipe; DOI: 10.1234/example"
      base:
        label: base
        type: cbs_extrapolation
        formula: helgaker_corr_2pt
        components: total      # only "total" is currently supported
        levels:
          - {method: ccsd(t), basis: cc-pVTZ}
          - {method: ccsd(t), basis: cc-pVQZ}
      corrections:
        - label: delta_T
          type: delta
          high: {method: ccsdt,   basis: cc-pVDZ}
          low:  {method: ccsd(t), basis: cc-pVDZ}

The ``base`` may equally be a plain level (``base: {method: ccsd(t)-f12,
basis: cc-pVTZ-f12}`` or a ``"method/basis"`` string) when a single anchor SP
is preferred over an extrapolation.

Term types:

* ``single_point`` — one absolute SP (only the ``base`` is usually one).
* ``delta`` — ``E[high] − E[low]`` between two levels (same basis typically).
* ``cbs_extrapolation`` — CBS extrapolation from ≥2 levels with the same
  method but different basis cardinalities. Built-in formulas:
  ``helgaker_hf_2pt`` (Halkier et al. 1998), ``helgaker_corr_2pt``
  (Helgaker et al. 1997), ``martin_3pt`` (Martin 1996). Alternatively,
  supply a user formula string referencing ``X``, ``Y``, ``Z`` (cardinals)
  and ``E_X``, ``E_Y``, ``E_Z`` (energies); it is parsed through a
  whitelisted AST evaluator — no ``eval()``. **A** ``cbs_extrapolation``
  **term is accepted only as the** ``base``: with ``components: total`` (the
  only supported value) it evaluates to an absolute energy, so listing it
  under ``corrections`` would double-count the base — ARC rejects this with
  an ``InputError`` pointing at CBS-as-base usage or the HEAT / W\ :sub:`n`
  presets. The CBS base's sub-jobs are tracked under the deterministic
  sub_labels ``base__card_<X>`` (one per cardinal).

.. note::

   Applying a two-point ``X^-3`` formula to **total** energies technically
   mis-treats the HF component, which converges exponentially with cardinal
   number rather than as ``X^-3``. At {T,Q} cardinals the residual is small
   and extrapolating totals is common practice. ARC currently parses only
   total electronic energies; component-wise extrapolation
   (``components: hf`` / ``corr``) awaits adapter-level component parsing.

**Form 4 — per-species override.** Three states are distinguishable::

    project: mixed
    sp_composite: HEAT-345Q          # applies by default to every species
    species:
      - label: H2O                   # inherits the project-wide protocol
        smiles: O
      - label: H2O_uncorrected
        smiles: O
        sp_composite: null           # opt out — use plain sp_level
      - label: TS1
        xyz: ...
        sp_composite:                # species-specific override
          base: {method: mp2, basis: cc-pVTZ}
          corrections: []

Internally each species is in one of three states: ``"inherit"`` (key absent),
``"opt_out"`` (explicit ``null``), ``"explicit"`` (preset name or recipe).
These three survive ``as_dict`` / ``from_dict`` and restart-dict round-trip.

.. note::

   A per-species ``"explicit"`` protocol affects only that species' composite
   energy: Arkane's atom energy correction (AEC) lookup still uses the
   project-global protocol's primary base level (the single base level, or
   the largest-cardinal CBS leg), not the species-level base. If a species
   needs AEC at a different level, set ``arkane_level_of_theory`` explicitly
   for the project.

**Form 5 — W\ :sub:`n` family for high-accuracy anchor energies.** When
δ-corrections beyond CCSD(T) are *not* the bottleneck and you mainly want a
near-CBS CCSD(T) reference with the canonical core-valence and scalar-
relativistic corrections, the W2/W3 family is a good fit::

    project: barriers_w3f12
    sp_composite: W3-F12
    species:
      - label: TS1
        xyz: ...

This is cheaper than a HEAT-345Q and converges quickly because the F12 anchor
already absorbs most of the CBS basis-set limit. ``W3-F12`` adds δ[CCSDT] on
top, which is typically the largest post-(T) effect for small organic TSs.

**Form 6 — HEAT-456Q for tighter CBS reference on small molecules.** For
small molecules where HF and CCSD(T) basis incompleteness matters, swap the
``cc-pVTZ-F12`` anchor for the ``cc-pVQZ-F12`` anchor::

    sp_composite: HEAT-456Q

This preset has the same correction stack as ``HEAT-345Q`` (δ[CCSDT],
δ[CCSDT(Q)], δ_CV, δ_rel) but a more accurate base, mirroring the published
HEAT-456 series whose HF/CCSD(T) CBS uses cardinals {Q,5,6}.

**Form 7 — preset + per-term basis upgrade.** Combine a published preset
with a partial override to refine just the term you care about::

    sp_composite:
      preset: HEAT-345Q
      overrides:
        delta_T:
          high: {method: ccsdt,   basis: cc-pVTZ}
          low:  {method: ccsd(t), basis: cc-pVTZ}

This keeps the inexpensive δ[CCSDT(Q)]/cc-pVDZ leg, the cheap δ_CV/cc-pCVTZ
core-valence pair, and the standard δ_rel — but moves only the δ[CCSDT]
correction to a tighter basis. Useful when one term is responsible for most
of the residual basis-set error in a barrier.

**Form 8 — explicit recipe with W\ :sub:`n`-style stacked deltas.** For
direct control of the entire ladder, write the recipe out::

    sp_composite:
      reference: "W3-style stack with custom anchor; DOI: 10.1063/1.1638736"
      base:
        method: ccsd(t)-f12
        basis: cc-pVQZ-f12
      corrections:
        - label: delta_T
          type: delta
          high: {method: ccsdt,    basis: cc-pVDZ}
          low:  {method: ccsd(t),  basis: cc-pVDZ}
        - label: delta_CV
          type: delta
          high: {method: ccsd(t),  basis: cc-pCVTZ,
                 args: {keyword: {core: 'core,0,0,0,0,0,0,0,0;'}, block: {}}}
          low:  {method: ccsd(t),  basis: cc-pCVTZ}
        - label: delta_rel
          type: delta
          high: {method: ccsd(t),  basis: cc-pVTZ-DK,
                 args: {keyword: {dkho: 'SET,DKHO=2;'}, block: {}}}
          low:  {method: ccsd(t),  basis: cc-pVTZ}

This is essentially what ``W3-F12`` expands to internally — useful as a
template when you want to deviate from a shipped preset.

**Interactions with other parameters.**

* **``sp_level``** — coexists. If you omit ``sp_level`` while setting
  ``sp_composite``, ARC derives ``sp_level`` from the protocol's primary base
  level so downstream code that reads ``sp_level`` (opt-out species, legacy
  paths) keeps working. For a single-SP base this is simply the base level;
  for a CBS base — which has no single level — it is the **largest-cardinal**
  leg of the extrapolation (e.g. ``ccsd(t)/cc-pvqz`` for ``FPA-min``). If you
  supply ``sp_level`` explicitly, it is preserved.
* **``composite_method`` (legacy)** — mutually exclusive with ``sp_composite``.
  Project fails to start with ``InputError`` if both are set.
* **``adaptive_levels``** — mutually exclusive in the current release. Raises
  ``InputError``. A future release may allow compatible combinations.
* **``conformer_sp_level``** — unaffected. Conformer ranking stays at its own
  level; ``sp_composite`` kicks in only at the final SP stage on the
  optimized geometry.

**AEC / BAC behavior.**
When ``sp_composite`` is active, ARC automatically routes Arkane's AEC lookup
through the protocol's primary base level (the single base level, or the
largest-cardinal CBS leg). The BAC lookup is **skipped entirely** with a
single warning — BAC was derived for a single LoT and is not meaningful on
top of a δ-corrected composite. If you need BAC, compute it externally
against the base level and add it as a literal term in the recipe.

Known limitation: per-species AEC is *not* implemented. When species carry
mixed per-species protocols, the global AEC lookup uses the *project-level*
protocol's primary base level. Users who need per-species AEC should set
``arkane_level_of_theory`` explicitly per project.

**Restart behavior.**
Composite sub-jobs are tracked in the persistent output dict
(``output[label]['paths']['sp_composite']: {sub_label → path}``). Restart
re-runs only the sub-jobs missing from that dict. On init the scheduler
*validates* every recorded path (file exists, ``parse_e_elect`` returns a
number); invalidated entries are pushed back to pending with a warning. After
seeding, the scheduler kick-starts any pending sub-jobs for species with prior
composite progress, so a restart with no other events still makes forward
progress.

**Provenance notebook.**
Every time a composite finalizes, ARC regenerates a single project-level
Jupyter notebook at ``<project>/output/sp_composite.ipynb``. It is
**unexecuted on write**: it contains cell sources but no outputs. The user
opens the notebook and runs "Run All" to independently verify the result —
each section reconstructs its ``CompositeProtocol`` from a literal recipe
dict, re-parses every sub-job QM output via ``arc.parser.parse_e_elect``, and
re-evaluates the total. Citations (with DOI when supplied) carry through
from ``presets.yml`` (or from the user's explicit ``reference:`` key) into the
notebook's markdown.

**Units.**
``arc.parser.parse_e_elect`` returns kJ/mol. ``CompositeProtocol.evaluate``
is a pass-through sum and preserves whatever units its inputs use. ARC always
stores ``species.e_elect`` in kJ/mol. Hartree is used only at display /
logging boundaries (division by ``arc.constants.E_h_kJmol``) and in the
Arkane species-file renderer, which converts once when writing the numeric
``energy = <Hartree>`` assignment.

**Known limitations.**

* **MRCC adapter**: ARC does not ship a dedicated standalone MRCC adapter.
  Methods that route through MRCC (``CCSDT``, ``CCSDT(Q)``, ``CCSDTQ``,
  ``CCSDTQ(P)``) work today through the Molpro adapter when Molpro is built
  with the MRCC interface, or through CFOUR-NCC.
* **Per-species AEC/BAC**: see the AEC/BAC section above.
* **``adaptive_levels`` interaction**: currently rejected; may relax later.

**References.**

* Allen, East, Császár — focal-point analysis review (general FPA methodology).
* Tajti, Szalay, Császár, Kállay, Gauss, Valeev, Flowers, Vázquez, Stanton,
  *J. Chem. Phys.* **121**, 11599 (2004). DOI: 10.1063/1.1811608 — HEAT-345 protocol.
* Bomble, Vázquez, Kállay, Michauk, Szalay, Császár, Gauss, Stanton,
  *J. Chem. Phys.* **125**, 064108 (2006). DOI: 10.1063/1.2206789 — HEAT-345(Q)
  and HEAT-456 series.
* Martin, de Oliveira, *J. Chem. Phys.* **111**, 1843 (1999).
  DOI: 10.1063/1.479454 — W1 / W2 protocols.
* Boese, Oren, Atasoylu, Martin, Kállay, Gauss, *J. Chem. Phys.* **120**, 4129
  (2004). DOI: 10.1063/1.1638736 — W3 protocol.
* Karton, Rabinovich, Martin, Ruscic, *J. Chem. Phys.* **125**, 144108 (2006).
  DOI: 10.1063/1.2348881 — W4 protocol.
* Karton, Martin, *J. Chem. Phys.* **136**, 124114 (2012).
  DOI: 10.1063/1.3697678 — W1-F12 and W2-F12 protocols. ARC's ``W3-F12``
  preset is an adaptation by analogy (no canonical primary publication
  titled "W3-F12"): it stacks δ[CCSDT] on top of the W2-F12 anchor in the
  spirit of how W3 (Boese et al. 2004) extended W2.
* Sylvetsky, Peterson, Karton, Martin, *J. Chem. Phys.* **144**, 214101
  (2016). DOI: 10.1063/1.4952410 — W4-F12 protocol.
* Helgaker, Klopper, Koch, Noga, *J. Chem. Phys.* **106**, 9639 (1997).
  DOI: 10.1063/1.473863 — two-point correlation CBS extrapolation.
* Halkier, Helgaker, Jørgensen, Klopper, Koch, Olsen, Wilson,
  *Chem. Phys. Lett.* **286**, 243-252 (1998). DOI: 10.1016/S0009-2614(98)00111-0
  — extends the two-point correlation-energy CBS extrapolation to Ne, N\ :sub:`2`,
  and H\ :sub:`2`\ O.
* Halkier, Helgaker, Jørgensen, Klopper, Olsen, *Chem. Phys. Lett.* **302**,
  437-446 (1999). DOI: 10.1016/S0009-2614(99)00179-7 — two-point HF-energy CBS
  extrapolation; source of the fitted ``α = 1.63`` exponential decay parameter
  used by ``helgaker_hf_2pt``.
* Martin, *Chem. Phys. Lett.* **259**, 669-678 (1996). DOI: 10.1016/0009-2614(96)00898-6
  — three-point Schwartz-style extrapolation.
* Dunning, *J. Chem. Phys.* **90**, 1007 (1989). DOI: 10.1063/1.456153 —
  correlation-consistent basis-set families; cardinal-number convention used
  by ``cardinal_from_basis``.


Adaptive Levels
---------------

Use ``adaptive_levels`` to change methods by molecule size. ARC expects tuple
keys for the heavy-atom ranges and tuple keys for grouped job types. In an
``input.yml`` file, write tuple keys with YAML's ``!!python/tuple`` tag:

.. code-block:: yaml

   adaptive_levels:
     ? !!python/tuple [1, 5]
     :
       ? !!python/tuple [opt, freq]
       : wb97xd/6-311+g(2d,2p)
       sp: ccsd(t)-f12/aug-cc-pvtz-f12
     ? !!python/tuple [6, 15]
     :
       ? !!python/tuple [opt, freq]
       : b3lyp/cbsb7
       sp: dlpno-ccsd(t)/def2-tzvp
     ? !!python/tuple [16, inf]
     :
       ? !!python/tuple [opt, freq]
       : b3lyp/6-31g(d,p)
       sp: wb97xd/6-311+g(2d,2p)

When using ARC from Python, pass regular Python tuples:

.. code-block:: python

   adaptive_levels = {
       (1, 5): {
           ('opt', 'freq'): 'wb97xd/6-311+g(2d,2p)',
           'sp': 'ccsd(t)-f12/aug-cc-pvtz-f12',
       },
       (6, 15): {
           ('opt', 'freq'): 'b3lyp/cbsb7',
           'sp': 'dlpno-ccsd(t)/def2-tzvp',
       },
       (16, 'inf'): {
           ('opt', 'freq'): 'b3lyp/6-31g(d,p)',
           'sp': 'wb97xd/6-311+g(2d,2p)',
       },
   }

Cover the full heavy-atom range without gaps.

Memory, CPUs, and Wall Time
---------------------------

Set defaults per project:

.. code-block:: yaml

   job_memory: 32
   max_job_time: 48

Server entries can also define node limits:

.. code-block:: python

   servers = {
       'my_slurm': {
           'cluster_soft': 'Slurm',
           'address': 'login.cluster.edu',
           'un': 'my_user',
           'key': '/home/my_user/.ssh/id_rsa',
           'cpus': 32,
           'memory': 128,
       },
   }

ARC may increase resources during troubleshooting, bounded by server and default
job settings. By default, troubleshooting will not request more than 95% of a
server node's configured memory.

.. _directory:

Project Directories
-------------------

By default, command-line runs use the directory containing the input file as the
project directory, while API runs create projects under ``ARC/Projects``. Set
``project_directory`` when you want outputs elsewhere:

.. code-block:: yaml

   project: ethanol_thermo
   project_directory: /scratch/my_user/arc_projects/ethanol_thermo

Remote project files are created on the server selected for each job. If a server
entry defines ``path``, ARC uses that path as the base for remote project
storage.

Routing ESS Jobs
----------------

Use ``ess_settings`` to override global software routing for a project:

.. code-block:: yaml

   ess_settings:
     gaussian:
       - high_memory_cluster
       - local
     orca: local
     molpro: server2

The order matters when a list is supplied; ARC tries the listed servers in
priority order.

Current supported ESS keys include ``cfour``, ``gaussian``, ``mockter``,
``molpro``, ``orca``, ``qchem``, ``terachem``, ``onedmin``, ``xtb``,
``torchani``, and ``openbabel``. Some additional adapters, such as TS-search
adapters, are configured through their own settings.

Fine-Grid Optimizations
-----------------------

The ``fine`` job type is enabled by default. If ``fine`` is true and ``opt`` is
false, ARC still runs optimization jobs but treats them as fine-grid jobs from
the start.

.. code-block:: yaml

   job_types:
     opt: false
     fine: true

Although this argument is called ``fine`` in ARC, in practice it directs the ESS to
use an **ultrafine** grid. See, for example, `this study`__ describing the
importance of the DFT grid.

__ DFTGridStudy_

In Gaussian, ``fine`` adds the following directive::

    scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)

In QChem, it adds the following directives::

    GEOM_OPT_TOL_GRADIENT     15
    GEOM_OPT_TOL_DISPLACEMENT 60
    GEOM_OPT_TOL_ENERGY       5
    XC_GRID                   3

In TeraChem, it adds the following directives::

    dftgrid 4
    dynamicgrid yes

Rotor Scans
-----------

``rotors`` is enabled by default. ARC identifies internal rotors and runs scans
for valid torsions. The default scan resolution is controlled by
``rotor_scan_resolution`` in settings.

Disable rotor scans for a project:

.. code-block:: yaml

   job_types:
     rotors: false

Use ``directed_rotors`` or ``preserve_param_in_scan`` on species when you need
more control over scan definitions and constrained internal coordinates.

ND Rotor Scans
--------------

ARC also supports ND (N-dimensional, N >= 1) rotor scans. There are seven different
ND types to execute:

- A1. Generate all geometries in advance (brute force), and calculate single-point
  energies (nested or diagonalized).
- A2. Generate all geometries in advance (brute force), and run constraint
  optimizations (nested or diagonalized).
- B. Derive the geometry from the previous point (continuous) and run constraint
  optimizations (nested or diagonalized).
- C. Let the ESS guide the optimizations.

Each of the options above (A or B) can be either "nested" (considering all ND
dihedral combinations) or "diagonal" (resulting in a unique 1D rotor scan across
several dimensions). The seventh option (C) allows the ESS to control the ND scan,
which is similar in principle to option B, but not directly controlled by ARC.

The optional primary keys are:

- ``brute_force_sp``
- ``brute_force_opt``
- ``cont_opt``
- ``ess``

The brute-force methods generate all the geometries in advance and submit all
relevant jobs simultaneously. The continuous method waits for the previous job to
terminate, and uses its geometry as the initial guess for the next job.

Another set of three keys is allowed, adding ``_diagonal`` to each of the above keys.
The secondary keys are therefore:

- ``brute_force_sp_diagonal``
- ``brute_force_opt_diagonal``
- ``cont_opt_diagonal``

Specifying ``_diagonal`` increments all the respective dihedrals together, resulting
in a 1D scan instead of an ND scan. Values are nested lists. Each value is a list
where the entries are either pivot lists (e.g., ``[1, 5]``) or lists of pivot lists
(e.g., ``[[1, 5], [6, 8]]``), or a mix (e.g., ``[[4, 8], [[6, 9], [3, 4]]]``). The
requested directed scan type is executed separately for each list entry. A list entry
that contains only two pivots results in a 1D scan, while a list entry with N pivots
considers all of them and results in an ND scan (if ``_diagonal`` is not specified).
Note that indices are 1-indexed.

ARC generates geometries using the ``rotor_scan_resolution`` argument in
``settings.py``. An ``'all'`` string entry is also allowed in the value list,
triggering a directed internal-rotation scan for all torsions in the molecule. If
``'all'`` is specified within a second-level list, all the dihedrals are considered
together. Currently ARC does not automatically identify torsions to be treated as ND,
so this attribute must be specified by the user.

To execute ND rotor scans, first set the ``rotors`` job type to ``True``, then set
the ``directed_rotors`` attribute of the relevant species. Below are several examples.

To run all dihedral scans of a species separately using brute-force sp (each as 1D)::

    spc1 = ARCSpecies(label='some_label', smiles='species_smiles', directed_rotors={'brute_force_sp': ['all']})

To run all dihedral scans of a species as a conjugated scan (ND, N = the number of
torsions)::

    spc1 = ARCSpecies(label='some_label', smiles='species_smiles', directed_rotors={'cont_opt': [['all']]})

Note the change in list level (``all`` is either within one or two nested lists) in
the above examples.

To run specific dihedrals as ND (here all 2D combinations for a species with 3
torsions)::

    spc1 = ARCSpecies(label='C4O2', smiles='[O]CCCC=O', xyz=xyz,
                      directed_rotors={'brute_force_opt': [[[5, 3], [3, 4]], [[3, 4], [4, 6]], [[5, 3], [4, 6]]]})

- Note: ND rotors are still **not** incorporated into the molecular partition
  function, so they currently do not affect thermo or rates.
- Note: Any torsion defined as part of an ND rotor scan will **not** be spawned for
  that species as a separate 1D scan.
- Warning: Job arrays have not been incorporated into ARC yet. Spawning ND rotor
  scans will result in **many** individual jobs being submitted to your server queue
  system.

Transition-State Search Adapters
--------------------------------

ARC can use several TS adapters when configured and installed, including
heuristics, linear, AutoTST, KinBot, GCN, xTB-GSM, and ORCA-NEB. See
:ref:`TS_search` for a description of each. Select adapters per project:

.. code-block:: yaml

   ts_adapters:
     - heuristics
     - xtb_gsm
     - orca_neb

User-supplied ``ts_xyz_guess`` values are always a useful fallback because they
make the calculation less dependent on automated TS guess generation.

Pipe Mode
---------

Pipe mode is ARC's opt-in distributed execution path for large homogeneous job
batches on HPC systems. It is disabled by default:

.. code-block:: python

   pipe_settings = {
       'enabled': True,
       'min_tasks': 10,
       'lease_duration_hrs': 1,
   }

Enable it in ``~/.arc/settings.py`` only after your normal scheduler submission
works. ARC considers pipe mode for eligible batches once ``min_tasks`` is met.
Transition-state guess generation is not currently wired through pipe mode, so
do not rely on pipe mode for TS-guess orchestration.

Troubleshooting Controls
------------------------

ARC attempts ESS and rotor troubleshooting by default. Disable these only when
you need strict no-resubmission behavior:

.. code-block:: yaml

   trsh_ess_jobs: false
   trsh_rotors: false

Use ``keep_checks: true`` when Gaussian checkfiles or other retained files are
needed for manual diagnosis.

At times a user might know in advance that a particular additional keyword is
required for the calculation. In such cases, pass the relevant keyword in the
``initial_trsh`` dictionary (``trsh`` stands for troubleshooting), keyed by ESS:

.. code-block:: yaml

   initial_trsh:
     gaussian:
       - iop(1/18=1)
     molpro:
       - shift,-1.0,-0.5;
     qchem:
       - GEOM_OPT_MAX_CYCLES 250

Batch Delete ARC Jobs
---------------------

.. warning::

   DANGER ZONE: make sure you understand what you're doing before running this
   script. Data of running jobs will be lost.

ARC has a feature that deletes all ARC-spawned jobs from selected servers and
projects. To delete all ARC jobs, run the following in the ARC code folder after
activating ``arc_env``::

    python arc/utils/delete.py -a

You can also delete jobs from a specific server by specifying its name after the
``-s`` flag::

    python arc/utils/delete.py -s server1 -a

To delete jobs from a specific ARC project, pass the project's name after the
``-p`` flag::

    python arc/utils/delete.py -p project1

Alternatively (since project names might be long and not always shown in full when
requesting the server job status), you can supply an ARC job ID, and ALL jobs
related to the project of the given job ID will be deleted (NOT only the given
job!)::

    python arc/utils/delete.py -j a_54836

Note that either a ``-a``, a ``-p``, or a ``-j`` flag must be given. All flags can
be combined with the optional ``-s`` flag.

Writing an ARC Input File Using the API
---------------------------------------

Writing YAML by hand isn't very intuitive for many users. You can instead use ARC's
API to define your objects, then dump them into a YAML file that ARC can read as an
input::

    from arc.species.species import ARCSpecies
    from arc.common import save_yaml_file

    input_dict = dict()

    input_dict['project'] = 'Demo_project_input_file_from_API'

    input_dict['job_types'] = {'conf_opt': True,
                               'opt': True,
                               'fine': True,
                               'freq': True,
                               'sp': True,
                               'rotors': True,
                               'conf_sp': False,
                               'orbitals': False,
                               'lennard_jones': False,
                              }

    spc1 = ARCSpecies(label='NO', smiles='[N]=O')

    adj1 = """multiplicity 2
    1 C u0 p0 c0 {2,D} {4,S} {5,S}
    2 C u0 p0 c0 {1,D} {3,S} {6,S}
    3 O u1 p2 c0 {2,S}
    4 H u0 p0 c0 {1,S}
    5 H u0 p0 c0 {1,S}
    6 H u0 p0 c0 {2,S}"""

    xyz2 = [
        """O       1.35170118   -1.00275231   -0.48283333
           C      -0.67437022    0.01989281    0.16029161
           C       0.62797113   -0.03193934   -0.15151370
           H      -1.14812497    0.95492850    0.42742905
           H      -1.27300665   -0.88397696    0.14797321
           H       1.11582953    0.94384729   -0.10134685""",
        """O       1.49847909   -0.87864716    0.21971764
           C      -0.69134542   -0.01812252    0.05076812
           C       0.64534929    0.00412787   -0.04279617
           H      -1.19713983   -0.90988817    0.40350584
           H      -1.28488154    0.84437992   -0.22108130
           H       1.02953840    0.95815005   -0.41011413"""]

    spc2 = ARCSpecies(label='vinoxy', xyz=xyz2, adjlist=adj1)

    spc_list = [spc1, spc2]

    input_dict['species'] = [spc.as_dict() for spc in spc_list]

    save_yaml_file(path='some/path/to/desired/folder/input.yml', content=input_dict)

The above code generates the following input file::

    project: Demo_project_input_file_from_API

    job_types:
      rotors: true
      conf_opt: true
      fine: true
      freq: true
      lennard_jones: false
      opt: true
      orbitals: false
      sp: true

    species:
    - E0: null
      arkane_file: null
      bond_corrections:
        N=O: 1
      charge: 0
      external_symmetry: null
      force_field: MMFF94
      generate_thermo: true
      is_ts: false
      label: 'NO'
      mol: |
        multiplicity 2
        1 N u1 p1 c0 {2,D}
        2 O u0 p2 c0 {1,D}
      multiplicity: 2
      number_of_rotors: 0
    - E0: null
      arkane_file: null
      bond_corrections:
        C-H: 3
        C-O: 1
        C=C: 1
      charge: 0
      conformers:
      - |-
        O       1.35170118   -1.00275231   -0.48283333
        C      -0.67437022    0.01989281    0.16029161
        C       0.62797113   -0.03193934   -0.15151370
        H      -1.14812497    0.95492850    0.42742905
        H      -1.27300665   -0.88397696    0.14797321
        H       1.11582953    0.94384729   -0.10134685
      - |-
        O       1.49847909   -0.87864716    0.21971764
        C      -0.69134542   -0.01812252    0.05076812
        C       0.64534929    0.00412787   -0.04279617
        H      -1.19713983   -0.90988817    0.40350584
        H      -1.28488154    0.84437992   -0.22108130
        H       1.02953840    0.95815005   -0.41011413
      force_field: MMFF94
      generate_thermo: true
      is_ts: false
      label: vinoxy
      mol: |
        multiplicity 2
        1 O u1 p2 c0 {3,S}
        2 C u0 p0 c0 {3,D} {4,S} {5,S}
        3 C u0 p0 c0 {1,S} {2,D} {6,S}
        4 H u0 p0 c0 {2,S}
        5 H u0 p0 c0 {2,S}
        6 H u0 p0 c0 {3,S}
      multiplicity: 2
      number_of_rotors: 0

Restarts
--------

Restart files are normal ARC inputs with more state. To restart:

.. code-block:: bash

   conda activate arc_env
   python /path/to/ARC/ARC.py restart.yml

Keep the project directory and server-side job files available when restarting;
ARC uses them to collect and continue previously submitted work.

.. include:: links.txt
