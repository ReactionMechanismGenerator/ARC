.. _TS_search:

Transition State Search
=======================

ARC can automatically search for and validate transition states (TSs) for a wide
range of reaction types.  It supports several TS-search methods, from fast
heuristic builders to machine-learning and reaction-path approaches.

All methods are registered as **TS adapters** and configured via the
``ts_adapters`` list in the ARC input file.  ARC tries each adapter in order
and collects all resulting TS guesses for downstream optimization and
validation (frequency + IRC).


ARC-native methods
------------------

These methods are implemented directly inside ARC and do not require any
external package beyond the standard ARC environment.


Heuristics (``'heuristics'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A rule-based TS builder that constructs chemically reasonable TS guess
geometries from the reactant/product structures and the RMG reaction-family
template.  It does not perform any electronic-structure calculation; geometry
construction is purely geometric.

**Supported families:**

*Hydrogen abstraction*
  For bimolecular ``A + B ⇌ C + D`` H-abstraction reactions, the adapter
  places the abstracted hydrogen between the donor and acceptor heavy atoms
  at Pauling partial-bond distances, then scans the approach dihedral at a
  configurable increment (default 90°) to generate multiple rotamer guesses.

*Neutral hydrolysis*
  For neutral hydrolysis reactions (ester, amide, acyl halide, ether, and
  nitrile hydrolysis), the adapter identifies the reactive atoms from the
  family definition and constructs one or more TS guesses for the water
  attack / leaving-group departure mechanism.  See reference below.

**Usage:**
Set ``ts_adapters: ['heuristics']`` in the input file.  For hydrogen
abstraction, the ``dihedral_increment`` keyword controls the rotational
scan resolution (smaller values yield more guesses).

**Reference:**
L. Fahoum, A. Grinberg Dana, *"Automated Reaction Transition State Search
for Neutral Hydrolysis Reactions"*, Digital Discovery, 2026.


Linear interpolation (``'linear'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An in-core adapter that generates TS guess geometries by interpolating
internal coordinates (Z-matrices) between reactant and product.  It handles
both isomerization (A ⇌ B) and addition/dissociation (A ⇌ B + C) reactions.

For **isomerization** reactions, a strategy pipeline is executed for each
reaction path identified from the RMG template:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Strategy
     - Description
   * - Ring scission
     - Folds the reactant chain into a ring, then stretches breaking bonds.
       Used for ring-opening reactions discovered in reverse.
   * - Direct contraction
     - Moves a terminal group toward its forming-bond partner.  Useful for
       radical ring-closure reactions (e.g., Intra_R_Add_Exocyclic).
   * - Ring closure
     - Rotates backbone torsions to close a forming bond into a ring.
   * - Z-matrix interpolation
     - The core method.  Builds two Z-matrix chimeras (Type R from the
       reactant topology, Type P from the product topology), blends them at
       the interpolation weight, and converts back to Cartesian coordinates.
       Only coordinates referencing reactive atoms are interpolated; spectator
       coordinates are kept from the source geometry.
   * - 3-center shift
     - Repositions a migrating atom (e.g., halogen, sulfur) between its
       donor and acceptor for 1,2-shift reactions.

For **addition/dissociation** reactions, the adapter starts from the
unimolecular species and:

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

**Post-processing:**

Every guess goes through family-specific post-processing (forming-bond
triangulation for H-transfer, donor H staggering, umbrella inversion for
migrating groups, reactive-bond distance adjustment, H orientation
correction) and validation (collision detection, detached-atom checks,
fragment counting, backbone drift, family-specific motif filters).

**Usage:**
Set ``ts_adapters: ['heuristics', 'linear']`` to run both native adapters.
The linear adapter is complementary to heuristics — it covers many families
that heuristics does not support.


External-package methods
------------------------

These methods rely on external packages that must be installed separately.
See the ARC installation guide for setup instructions.


AutoTST (``'autotst'``)
^^^^^^^^^^^^^^^^^^^^^^^^

Uses the `AutoTST <https://github.com/ReactionMechanismGenerator/AutoTST>`_
package to generate TS guesses from RMG reaction templates.  AutoTST
performs systematic conformer searches of the TS using distance-geometry
embedding and RDKit force-field optimization, guided by the reaction family
template distances.

Runs as a subprocess.  Requires the ``autotst`` conda environment.


KinBot (``'kinbot'``)
^^^^^^^^^^^^^^^^^^^^^

Uses the `KinBot <https://github.com/zadorlab/KinBot>`_ package, which
performs automated reaction discovery and TS search using semiempirical
or DFT methods.  KinBot explores the potential energy surface starting from
a given species and locates TS geometries for elementary reactions.

Runs as a subprocess.  Requires the ``kinbot`` conda environment.


TS-GCN (``'gcn'``)
^^^^^^^^^^^^^^^^^^^

Uses a graph-convolutional neural network (TS-GCN) trained on DFT-optimized
TS geometries to predict 3D TS structures directly from the reactant and
product graphs.  This is the fastest external method but is limited to the
atom types and reaction classes in its training data.

Runs as a subprocess.  Requires the ``ts_gcn`` conda environment.


xTB-GSM (``'xtb_gsm'``)
^^^^^^^^^^^^^^^^^^^^^^^^

Uses the `Growing String Method (GSM)
<https://github.com/ZimmermanGroup/molecularGSM>`_ with the GFN2-xTB
semiempirical method to locate approximate TS geometries along the
minimum-energy path between reactant and product.  This is a reaction-path
method rather than a guess-based method, so it tends to produce higher-quality
initial TS geometries at the cost of longer compute time.

Runs as a subprocess.  Requires ``xtb`` and ``gsm`` executables.


ORCA NEB (``'orca_neb'``)
^^^^^^^^^^^^^^^^^^^^^^^^^

Uses ORCA's nudged elastic band (NEB) implementation to find the
minimum-energy path and locate the TS as the highest-energy image.
This is a DFT-level reaction-path method and produces high-quality TS
geometries, but is significantly more expensive than the heuristic methods.

Requires a configured ORCA installation and server access.


General workflow
----------------

Regardless of which adapter(s) are used, ARC follows the same general
workflow for each reaction:

1. **TS guess generation** — each adapter produces one or more candidate
   TS geometries.
2. **Clustering** — near-duplicate guesses are removed.
3. **Optimization** — each surviving guess is optimized at the specified
   level of theory.
4. **Validation** — frequency analysis confirms exactly one imaginary
   frequency, and IRC calculations verify that the TS connects the correct
   reactant and product wells.

Multiple adapters can be combined (e.g., ``ts_adapters: ['heuristics',
'linear', 'gcn', 'kinbot']``) to maximize coverage across reaction families.


.. include:: links.txt
