.. _tools:

Standalone Tools
================

ARC ships several Jupyter notebooks under ``ipython/Tools``. They are not
embedded web apps in the documentation site; they are notebooks you open and run
locally from the repository after activating ``arc_env``.

Run them from the repository root:

.. code-block:: bash

   conda activate arc_env
   jupyter notebook ipython/Tools

The notebooks are useful for diagnostics, geometry inspection, conversion, and
small calculations that reuse ARC internals without launching a full ARC project.

Available Notebooks
-------------------

``ARC ESS diagnostics.ipynb``
    Checks configured servers and reports which electronic structure software ARC
    can find. Use this after editing ``~/.arc/settings.py`` and ``~/.arc/submit.py``.

    GitHub: `ARC ESS diagnostics notebook <https://github.com/ReactionMechanismGenerator/ARC/blob/main/ipython/Tools/ARC%20ESS%20diagnostics.ipynb>`__

``conformers.ipynb``
    Runs ARC conformer-generation utilities for a molecule and lets you inspect
    candidate conformers before launching expensive ESS jobs.

    GitHub: `Conformers notebook <https://github.com/ReactionMechanismGenerator/ARC/blob/main/ipython/Tools/conformers.ipynb>`__

``coordinates conversions.ipynb``
    Demonstrates conversions between XYZ strings, ARC XYZ dictionaries,
    Z-matrices, and related coordinate formats.

    GitHub: `Coordinate conversions notebook <https://github.com/ReactionMechanismGenerator/ARC/blob/main/ipython/Tools/coordinates%20conversions.ipynb>`__

``Plot xyz.ipynb``
    Visualizes a 3D geometry from XYZ-style input.

    GitHub: `Plot XYZ notebook <https://github.com/ReactionMechanismGenerator/ARC/blob/main/ipython/Tools/Plot%20xyz.ipynb>`__

``Visualize 1D Torsion scan.ipynb``
    Plots a one-dimensional rotor scan from ARC scan data.

    GitHub: `Visualize 1D torsion scan notebook <https://github.com/ReactionMechanismGenerator/ARC/blob/main/ipython/Tools/Visualize%201D%20Torsion%20scan.ipynb>`__

``Visualize 2D Torsion scan.ipynb``
    Plots a two-dimensional rotor scan from the YAML files ARC creates for scan
    surfaces. It can also show the conformer closest to selected dihedral values,
    and lets you change the colormap and resolution of the resulting image.

    GitHub: `Visualize 2D torsion scan notebook <https://github.com/ReactionMechanismGenerator/ARC/blob/main/ipython/Tools/Visualize%202D%20Torsion%20scan.ipynb>`__

``External symmetry and optical isomers.ipynb``
    Calculates external symmetry and optical isomer counts from 3D coordinates.

    GitHub: `External symmetry notebook <https://github.com/ReactionMechanismGenerator/ARC/blob/main/ipython/Tools/External%20symmetry%20and%20optical%20isomers.ipynb>`__

``Determine harmonic frequencies scaling factor.ipynb``
    Uses Truhlar's method to determine a harmonic frequency scaling factor for a
    selected level of theory.

    GitHub: `Frequency scaling notebook <https://github.com/ReactionMechanismGenerator/ARC/blob/main/ipython/Tools/Determine%20harmonic%20frequencies%20scaling%20factor.ipynb>`__

``Perceive_xyz_(xyz_to_smiles).ipynb``
    Attempts to infer connectivity and return a SMILES representation from 3D
    coordinates.

    GitHub: `XYZ perception notebook <https://github.com/ReactionMechanismGenerator/ARC/blob/main/ipython/Tools/Perceive_xyz_(xyz_to_smiles).ipynb>`__

``OneDMin script.ipynb``
    Helps choose ``r_min`` and ``r_max`` for OneDMin calculations by visualizing
    bath gas placement around the target species.

    GitHub: `OneDMin notebook <https://github.com/ReactionMechanismGenerator/ARC/blob/main/ipython/Tools/OneDMin%20script.ipynb>`__

``TS Guesses.ipynb``
    Demonstrates transition-state guess utilities and inspection workflows.

    GitHub: `TS guesses notebook <https://github.com/ReactionMechanismGenerator/ARC/blob/main/ipython/Tools/TS%20Guesses.ipynb>`__

``delete all arc jobs.ipynb``
    Deletes jobs on a selected server whose names start with ``a_``.

    .. warning::

       This notebook can cause unintentional loss of data (running jobs). It might
       delete non-ARC-related jobs, and it will delete ARC jobs of all projects you
       are currently running on the selected server(s). Use it only when you
       understand exactly what it does and the selected server, user, and job-name
       filter.

    GitHub: `Delete jobs notebook <https://github.com/ReactionMechanismGenerator/ARC/blob/main/ipython/Tools/delete%20all%20arc%20jobs.ipynb>`__

Demo notebooks are also available under ``ipython/Demo``:

* `ARC thermo demo <https://github.com/ReactionMechanismGenerator/ARC/blob/main/ipython/Demo/ARC%20thermo%20demo.ipynb>`__
* `ARC reaction demo using YAML <https://github.com/ReactionMechanismGenerator/ARC/blob/main/ipython/Demo/ARC%20rxn%20demo%20using%20YAML.ipynb>`__

.. include:: links.txt
