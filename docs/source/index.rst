ARC Documentation
=================

.. image:: arc.png
   :align: center
   :width: 180px
   :alt: ARC logo

**ARC** (Automated Rate Calculator) automates electronic structure calculations for
chemical kinetics. It turns species and reaction definitions into managed quantum
chemistry jobs, tracks those jobs on local or remote compute resources, and
processes the results into thermodynamic and kinetic data.

ARC is useful when you need reproducible thermochemistry, high-pressure-limit
rate coefficients, transition-state workflows, rotor scans, or structured
post-processing around electronic structure software.

What ARC Does
-------------

ARC accepts molecular inputs such as SMILES_, InChI_, RMG adjacency lists, XYZ
coordinates, and existing electronic-structure input or output files. It can then:

* generate and optimize conformers;
* run optimizations, frequencies, single-point energies, rotor scans, IRC jobs,
  BDE workflows, and selected transition-state search adapters;
* submit jobs locally or over SSH to servers using OGE/SGE, Slurm, PBS, or
  HTCondor;
* recover from many failed jobs using built-in troubleshooting logic;
* run Arkane for thermochemistry and kinetics when the required RMG/Arkane
  environment is available.

Start Here
----------

New users should read these pages in order:

1. :ref:`installation` - create the ``arc_env`` environment, install optional
   dependencies, and configure personal settings.
2. :ref:`running` - run ARC from YAML, Python scripts, notebooks, local machines,
   remote servers, and HPC clusters.
3. :ref:`how_it_works` - understand the workflow, core objects, job lifecycle,
   data flow, and output structure.
4. :ref:`input_reference` - see every accepted ``input.yml`` key and how nested
   species, reaction, level, and job-type dictionaries are shaped.
5. :ref:`examples` - copy known-good input patterns for species, reactions,
   transition states, and the Python API.
6. :ref:`advanced` - customize levels of theory, job types, memory, ESS routing,
   rotor scans, troubleshooting, and restarts.

Reference
---------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   docker
   running
   how_it_works
   input_reference
   examples
   TS_search
   advanced
   output
   tools

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   media
   release
   credits
   contribute
   cite
   licence

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: links.txt
