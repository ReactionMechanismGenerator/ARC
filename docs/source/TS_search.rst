.. _TS_search:

Transition State Search
=======================

ARC can automatically search for and validate transition states (TSs) for a wide
range of reaction types, from fast heuristic builders to machine-learning and
reaction-path approaches. This section describes the currently supported TS-search
methods and how to use them through ARC input files.

All methods are registered as **TS adapters** and configured via the ``ts_adapters``
list in the ARC input file. ARC tries each adapter in order and collects all
resulting TS guesses for downstream optimization and validation (energy, frequency,
and IRC).

ARC-native methods
------------------

These methods are implemented directly inside ARC and do not require any external
package beyond the standard ARC environment.

Heuristics adapter
^^^^^^^^^^^^^^^^^^

ARC includes an internal TS-guess adapter named ``heuristics``. The implementation
lives in ``arc/job/adapters/ts/heuristics.py`` and runs incore; it generates
candidate TS geometries directly from the mapped reactant and product wells and the
RMG reaction-family template rather than submitting a separate external TS-search
program. It does not perform any electronic-structure calculation; TS construction is
purely geometric.

The current heuristic adapter supports:

- ``H_Abstraction`` reactions
- ``carbonyl_based_hydrolysis`` reactions
- ``ether_hydrolysis`` reactions
- ``nitrile_hydrolysis`` reactions

Use it by listing ``heuristics`` under ``ts_adapters``:

.. code-block:: yaml

   ts_adapters:
     - heuristics

.. graphviz::
   :caption: How the heuristic TS adapter fits into an ARC run.

   digraph heuristic_ts {
      graph [rankdir=LR, bgcolor="transparent"];
      node [shape=box, style="rounded,filled", fillcolor="#f5f7fa", color="#9fb3c8", fontname="Helvetica"];
      edge [color="#5b6676", fontname="Helvetica"];

      rxn [label="ARCReaction\nmapped wells"];
      family [label="Reaction family"];
      heuristics [label="heuristics.py"];
      guesses [label="TSGuess objects\nXYZ geometries"];
      opt [label="TS optimization"];
      validate [label="frequency / IRC\nvalidation"];

      rxn -> family -> heuristics -> guesses -> opt -> validate;
   }

Hydrogen abstraction heuristic TS search
""""""""""""""""""""""""""""""""""""""""

For RMG ``H_Abstraction`` reactions, the heuristic adapter constructs TS guesses
for reactions of the form:

.. code-block:: text

   R1-H + R2 <=> R1 + R2-H

The algorithm identifies the transferred hydrogen and the two reacting centers
from the mapped reaction, places the abstracted hydrogen between the donor and
acceptor heavy atoms at Pauling partial-bond distances, combines reactant/product
geometries, stretches the forming and breaking H bonds, and scans the approach
dihedral at a configurable increment to generate multiple rotamer guesses. The
``dihedral_increment`` keyword controls the rotational scan resolution (default 30°;
smaller values yield more guesses). Duplicate and colliding geometries are filtered
before ARC stores the remaining guesses as ``TSGuess(method='Heuristics')`` entries.

At minimum, the reaction must have:

- A recognized ``H_Abstraction`` family assignment.
- 3D coordinates for all reactant and product wells.
- Atom mapping between reactants and products, including the transferred H.
- ``heuristics`` enabled in ``ts_adapters``.

Example input pattern:

.. code-block:: yaml

   project: h_abstraction_example

   ts_adapters:
     - heuristics

   species:
     - label: methane
       smiles: C
     - label: OH
       smiles: "[OH]"
     - label: methyl
       smiles: "[CH3]"
     - label: water
       smiles: O

   reactions:
     - label: methane + OH <=> methyl + water
       reactants:
         - methane
         - OH
       products:
         - methyl
         - water

ARC then uses the heuristic guesses as normal TS candidates: it optimizes them,
checks for a single meaningful imaginary frequency, optionally runs IRCs, and
uses the successful TS in kinetics processing.

Neutral hydrolysis TS search
""""""""""""""""""""""""""""

ARC supports automated TS generation and validation for **neutral hydrolysis reactions**.
This capability is designed to start from a high-level reaction definition (e.g., SMILES-defined reactants/products)
and proceed through TS generation, optimization, and validation without requiring manual TS construction.

Supported sub-families
~~~~~~~~~~~~~~~~~~~~~~~
The current implementation supports the following neutral hydrolysis sub-families:

- Ester hydrolysis
- Amide hydrolysis
- Acyl halide hydrolysis
- Ether hydrolysis
- Nitrile hydrolysis

How it is used
~~~~~~~~~~~~~~
To run neutral hydrolysis TS search, define the reacting species and the overall reaction in the input file
(see ARC's examples folder for an input file that executes a neutral hydrolysis TS search).
At minimum, specify:

- The participating species (e.g., SMILES, xyz, InChI, or adjacency list)
- A reaction string connecting the species labels (e.g., ``A + H2O <=> products``)
- The TS generation adapter(s) under ``ts_adapters`` (in this case, use: ``['heuristics']``)
- The electronic structure levels used for optimization/validation (e.g., ``opt_level``, ``freq_level``, ``irc_level``)

What ARC does
~~~~~~~~~~~~~
For neutral hydrolysis reactions, ARC performs the following general steps:

1. Identify the relevant reactive atoms based on the reaction family definition.
2. Generate one or more chemically reasonable TS guesses for the hydrolysis transformation.
3. Optimize the TS candidates that pass internal filtration.
4. Validate the TS using vibrational frequency and IRC calculations.

Linear interpolation adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``linear`` adapter is an in-core adapter that generates TS guess geometries by
interpolating internal coordinates (Z-matrices) between reactant and product. It
handles both isomerization (A ⇌ B) and addition/dissociation (A ⇌ B + C) reactions,
covering many families that ``heuristics`` does not support.

For **isomerization** reactions, a strategy pipeline is executed for each reaction
path identified from the RMG template:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Strategy
     - Description
   * - Ring scission
     - Folds the reactant chain into a ring, then stretches breaking bonds.
       Used for ring-opening reactions discovered in reverse.
   * - Direct contraction
     - Moves a terminal group toward its forming-bond partner. Useful for
       radical ring-closure reactions (e.g., Intra_R_Add_Exocyclic).
   * - Ring closure
     - Rotates backbone torsions to close a forming bond into a ring.
   * - Z-matrix interpolation
     - The core method. Builds two Z-matrix chimeras (Type R from the
       reactant topology, Type P from the product topology), blends them at
       the interpolation weight, and converts back to Cartesian coordinates.
       Only coordinates referencing reactive atoms are interpolated; spectator
       coordinates are kept from the source geometry.
   * - 3-center shift
     - Repositions a migrating atom (e.g., halogen, sulfur) between its
       donor and acceptor for 1,2-shift reactions.

For **addition/dissociation** reactions, the adapter starts from the unimolecular
species and:

1. Identifies which bonds to cut using the RMG template or combinatorial
   fragmentation.
2. Stretches the fragments apart to Pauling TS-estimate distances.
3. Migrates atoms (typically H) between fragments when the product
   composition requires it.
4. For concerted multi-bond eliminations (e.g., XY_elimination producing
   C=C + H₂ + CO₂), a concerted builder simultaneously stretches breaking
   bonds and contracts forming bonds.

**Dedicated family builders:**

* **XY elimination hydroxyl** — builds a 6-membered ring TS by folding the
  molecule through three dihedral rotations, then setting element-specific
  Pauling distances (H–H short, H–O shorter than H–C, C–C long).

**Post-processing:** every guess goes through family-specific post-processing
(forming-bond triangulation for H-transfer, donor H staggering, umbrella inversion
for migrating groups, reactive-bond distance adjustment, H orientation correction)
and validation (collision detection, detached-atom checks, fragment counting,
backbone drift, family-specific motif filters).

Set ``ts_adapters: ['heuristics', 'linear']`` to run both native adapters. The
linear adapter is complementary to heuristics.

External-package methods
------------------------

These methods rely on external packages that must be installed separately. See
:ref:`installation` for setup instructions.

AutoTST (``'autotst'``)
^^^^^^^^^^^^^^^^^^^^^^^^

Uses the `AutoTST <https://github.com/ReactionMechanismGenerator/AutoTST>`_
package to generate TS guesses from RMG reaction templates. AutoTST performs
systematic conformer searches of the TS using distance-geometry embedding and
RDKit force-field optimization, guided by the reaction family template distances.

Runs as a subprocess. Requires the ``autotst`` conda environment.

KinBot (``'kinbot'``)
^^^^^^^^^^^^^^^^^^^^^

Uses the `KinBot <https://github.com/zadorlab/KinBot>`_ package, which performs
automated reaction discovery and TS search using semiempirical or DFT methods.
KinBot explores the potential energy surface starting from a given species and
locates TS geometries for elementary reactions.

Runs as a subprocess. Requires the ``kinbot`` conda environment.

TS-GCN (``'gcn'``)
^^^^^^^^^^^^^^^^^^

Uses a graph-convolutional neural network (TS-GCN) trained on DFT-optimized TS
geometries to predict 3D TS structures directly from the reactant and product
graphs. This is the fastest external method but is limited to the atom types and
reaction classes in its training data.

Runs as a subprocess. Requires the ``ts_gcn`` conda environment.

xTB-GSM (``'xtb_gsm'``)
^^^^^^^^^^^^^^^^^^^^^^^

Uses the `Growing String Method (GSM)
<https://github.com/ZimmermanGroup/molecularGSM>`_ with the GFN2-xTB
semiempirical method to locate approximate TS geometries along the
minimum-energy path between reactant and product. This is a reaction-path method
rather than a guess-based method, so it tends to produce higher-quality initial TS
geometries at the cost of longer compute time.

Runs as a subprocess. Requires ``xtb`` and ``gsm`` executables.

ORCA NEB (``'orca_neb'``)
^^^^^^^^^^^^^^^^^^^^^^^^^

Uses ORCA's nudged elastic band (NEB) implementation to find the minimum-energy
path and locate the TS as the highest-energy image. This is a DFT-level
reaction-path method and produces high-quality TS geometries, but is significantly
more expensive than the heuristic methods.

Requires a configured ORCA installation and server access.

General workflow
----------------

Regardless of which adapter(s) are used, ARC follows the same general workflow
for each reaction:

1. **TS guess generation** — each adapter produces one or more candidate TS
   geometries.
2. **Clustering** — near-duplicate guesses are removed.
3. **Optimization** — each surviving guess is optimized at the specified level
   of theory.
4. **Validation** — frequency analysis confirms exactly one meaningful imaginary
   frequency, and IRC calculations verify that the TS connects the correct
   reactant and product wells.

Multiple adapters can be combined (e.g., ``ts_adapters: ['heuristics', 'linear',
'gcn', 'kinbot']``) to maximize coverage across reaction families.

Outputs and validation
----------------------
Validated TS results are reported in the project output (log files and generated artifacts),
together with the supporting calculations (optimization, frequency, and IRC).
ARC does not require TS geometries to be isomorphic with a stored 2D adjacency list, since a TS does not have a
single strict graph representation. Instead, TS validation relies on TS-specific checks such as the imaginary
frequency, normal mode displacement analysis, IRC results, and energetic consistency.

References
----------

[1] C. Pieters, A. Grinberg Dana, *"Learning Rates: Predicting Rate Coefficients for Hydrogen Abstraction Reactions"*,
Digital Discovery 2026.

[2] L. Fahoum, A. Grinberg Dana, *"Automated reaction transition state search for bimolecular liquid-phase reactions
using internal coordinates: a test case for neutral hydrolysis"*, Digital Discovery 2026, 5, 1372-1387,
DOI: 10.1039/D5DD00506J.

CREST
^^^^^

CREST is an external conformational sampling tool used by ARC as a TS-search wrapper stage.
In ARC's current flow, CREST is applied to TS seeds generated by base TS search methods and uses
family-specific constraints from ARC.

Current ARC family support for CREST:

- ``H_Abstraction`` only (RMG family reference:
  `H_Abstraction <https://rmg.mit.edu/database/kinetics/families/H_Abstraction/>`_).

External references:

- `CREST documentation <https://crest-lab.github.io/crest-docs/>`_
- `CREST constrained sampling example <https://crest-lab.github.io/crest-docs/page/examples/example_4.html>`_

Wrapper Extension Guide
"""""""""""""""""""""""

Use this guide when extending CREST-based TS workflows in ARC (for example, adding hydrolysis support to CREST,
or allowing CREST to wrap a new TS seed source adapter).

ARC uses a neutral wrapper hub API for TS seed generation and wrapper-specific constraints:

- ``arc.job.adapters.ts.seed_hub.get_ts_seeds(...)``
- ``arc.job.adapters.ts.seed_hub.get_wrapper_constraints(...)``

Current status
""""""""""""""

- ``CrestAdapter`` requests seeds using ``base_adapter='heuristics'``.
- ``CrestAdapter`` requests constraints using ``wrapper='crest'``.
- CREST constraints are currently implemented for ``H_Abstraction`` only.
- Hydrolysis seeds can be generated by heuristics, but CREST constraints for hydrolysis are not implemented yet.

Seed schema contract
""""""""""""""""""""

``get_ts_seeds(...)`` returns a list of seed dictionaries with the following fields:

- ``xyz``: Cartesian coordinates dictionary.
- ``family``: Reaction family associated with the seed.
- ``method``: Method label for provenance.
- ``source_adapter``: TS-search adapter id that generated the seed.
- ``metadata``: Optional adapter-specific metadata dictionary.

Extension instructions: Add a new family to CREST
"""""""""""""""""""""""""""""""""""""""""""""""""

1. Update ``get_ts_seeds(...)`` logic in ``arc/job/adapters/ts/seed_hub.py`` only if the seed generation path changes.
2. Add family-specific CREST constraints in ``_get_crest_constraints(...)`` (or family helper it calls) in
   ``arc/job/adapters/ts/seed_hub.py``.
3. Add/update tests in ``arc/job/adapters/ts/heuristics_test.py`` (``TestHeuristicsHub``).
4. Update ``ts_adapters_by_rmg_family`` mapping if CREST should be enabled for that family.

Extension instructions: Let CREST wrap a new TS seed adapter
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

1. Add a ``base_adapter`` branch in ``get_ts_seeds(...)``.
2. Ensure the returned seed objects satisfy the seed schema contract.
3. Reuse ``get_wrapper_constraints(wrapper='crest', ...)`` with those seeds.
4. Add tests for the new adapter branch and constraints compatibility.

Minimal usage pattern
"""""""""""""""""""""

.. code-block:: python

    from arc.job.adapters.ts.seed_hub import get_ts_seeds, get_wrapper_constraints

    seeds = get_ts_seeds(
        reaction=rxn,
        base_adapter='heuristics',
        dihedral_increment=30,
    )
    for seed in seeds:
        crest_constraints = get_wrapper_constraints(
            wrapper='crest',
            reaction=rxn,
            seed=seed,
        )
        if crest_constraints is None:
            continue
        # run CREST with crest_constraints["A"], crest_constraints["H"], crest_constraints["B"]

.. include:: links.txt
