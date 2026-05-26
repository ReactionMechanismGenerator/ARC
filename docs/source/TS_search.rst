.. _TS_search:

Transition State Search
=======================

ARC can automatically search for and validate transition states (TSs) for selected reaction types.
This section describes currently supported TS-search workflows and how to use them through ARC input files.

Heuristics adapter
^^^^^^^^^^^^^^^^^^

ARC includes an internal TS-guess adapter named ``heuristics``. The implementation
lives in ``arc/job/adapters/ts/heuristics.py`` and runs incore; it generates
candidate TS geometries directly from the mapped reactant and product wells
rather than submitting a separate external TS-search program.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For RMG ``H_Abstraction`` reactions, the heuristic adapter constructs TS guesses
for reactions of the form:

.. code-block:: text

   R1-H + R2 <=> R1 + R2-H

The algorithm identifies the transferred hydrogen and the two reacting centers
from the mapped reaction, combines reactant/product geometries, stretches the
forming and breaking H bonds, and samples relevant dihedral angles. Duplicate
and colliding geometries are filtered before ARC stores the remaining guesses as
``TSGuess(method='Heuristics')`` entries.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ARC supports automated TS generation and validation for **neutral hydrolysis reactions**.
This capability is designed to start from a high-level reaction definition (e.g., SMILES-defined reactants/products)
and proceed through TS generation, optimization, and validation without requiring manual TS construction.

Supported sub-families
""""""""""""""""""""""
The current implementation supports the following neutral hydrolysis sub-families:

- Ester hydrolysis
- Amide hydrolysis
- Acyl halide hydrolysis
- Ether hydrolysis
- Nitrile hydrolysis

How it is used
""""""""""""""
To run neutral hydrolysis TS search, define the reacting species and the overall reaction in the input file
(see ARC's examples folder for an input file that executes a neutral hydrolysis TS search).
At minimum, specify:

- The participating species (e.g., SMILES, xyz, InChI, or adjacency list)
- A reaction string connecting the species labels (e.g., ``A + H2O <=> products``)
- The TS generation adapter(s) under ``ts_adapters`` (in this case, use: ``['heuristics']``)
- The electronic structure levels used for optimization/validation (e.g., ``opt_level``, ``freq_level``, ``irc_level``)

What ARC does
""""""""""""""""""""""""""
For neutral hydrolysis reactions, ARC performs the following general steps:

1. Identify the relevant reactive atoms based on the reaction family definition.
2. Generate one or more chemically reasonable TS guesses for the hydrolysis transformation.
3. Optimize the TS candidates that pass internal filtration.
4. Validate the TS using vibrational frequency and IRC calculations.

Outputs and validation
""""""""""""""""""""""
Validated TS results are reported in the project output (log files and generated artifacts),
together with the supporting calculations (optimization, frequency, and IRC).
ARC does not require TS geometries to be isomorphic with a stored 2D adjacency list, since a TS does not have a
single strict graph representation. Instead, TS validation relies on TS-specific checks such as the imaginary
frequency, normal mode displacement analysis, IRC results, and energetic consistency.

Reference
"""""""""
A detailed description of the methodology, design choices, and validation benchmarks is provided in:
L. Fahoum, A. Grinberg Dana, *“Automated Reaction Transition State Search for Neutral Hydrolysis Reactions”*,
Digital Discovery, 2026.

.. include:: links.txt
