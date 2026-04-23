.. _TS_search:

Transition State Search
=======================

ARC can automatically search for and validate transition states (TSs) for selected reaction types.
This section describes currently supported TS-search workflows and how to use them through ARC input files.

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
3. Optimize the TS candidates that passes internal filtration.
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
