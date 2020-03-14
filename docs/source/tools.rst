.. _tools:

Standalone tools
================

The ARC repository includes various stand-alone scripts, able to perform useful tasks using ARC's functions and methods
without spawning actual ESS jobs. These scripts are available as iPython Jupyter notebooks under the ``ipython`` folder.


ARC ESS diagnostics
^^^^^^^^^^^^^^^^^^^

This tool will search all servers defined in ARC's settings file, and report on the different ESS it finds.


Conformers
^^^^^^^^^^

This tool uses ARC's method to generate force field conformers.


Delete all ARC jobs
^^^^^^^^^^^^^^^^^^^

This tool will delete all jobs on the selected server that have job names starting with ``a_``.

**WARNING**: This script might cause unintentional loss of data (running jobs).
Make sure you understand exactly what you are doing and what it does before executing it.
I might delete non-ARC related jobs, and it will delete ARC jobs of all projects you are
currently running on that server/s.


Determine harmonic frequencies scaling factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This tool uses `Truhlar's method`__ to determine frequency scaling factors for the user's choice of level of theory.

__ truhlar_


External symmetry and optical isomers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This tool calculates external symmetry and number of optical isomers (1 or 2) from 3D coordinates.


OneDMin script
^^^^^^^^^^^^^^

This tool assist in determination of r_min and r_max for a OneDMin calculation by visualizing the bath gas
at these distances from the center of mass of the target species.


Perceive_xyz_(xyz_to_smiles)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This perception tool returns SMILES for given 3D coordinates.


Plot xyz
^^^^^^^^

This is a plotter tool for 3D coordinates.


Visualize 1D Torsion scan
^^^^^^^^^^^^^^^^^^^^^^^^^

This is a plotter tool for 1D torsion scans.


Visualize 2D Torsion scan
^^^^^^^^^^^^^^^^^^^^^^^^^

This is a plotter tool for 2D torsion scans created by ARC
(the YAML file created by ARC is necessary to visuallize this map).
The user can also change the color pam and resolution of the resulting image.

Specifying a combination of two pivots will show a 3D representation of the respective conformer
with the closest dihedrals to the ones specified.


.. include:: links.txt
