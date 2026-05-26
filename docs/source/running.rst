.. _running:

Running ARC
===========

ARC can be run from a YAML input file, from Python, or from notebooks. The same
configuration concepts apply in all modes: define species and reactions, choose
job types and levels of theory, and map electronic structure software to local
or remote compute resources.

Activate ARC First
------------------

Use ``arc_env`` for development and execution:

.. code-block:: bash

   conda activate arc_env

Run from YAML
-------------

Create an ``input.yml``:

.. code-block:: yaml

   project: ethanol_demo

   species:
     - label: ethanol
       smiles: CCO

Run it from the repository checkout:

.. code-block:: bash

   python ARC.py input.yml

Or from elsewhere:

.. code-block:: bash

   python /path/to/ARC/ARC.py /path/to/input.yml

ARC writes project files under ``Projects/<project>`` by default unless
``project_directory`` is supplied.
All parameters of :ref:`arc.main.ARC <main>` are legal top-level input file
keywords. Entries under ``species`` and ``reactions`` define
:ref:`ARCSpecies <species>` and :ref:`ARCReaction <reaction>` objects. See
:ref:`input_reference` for the full input-key checklist.

Restart a Project
-----------------

ARC writes ``restart.yml`` files that contain the current state of the project,
including submitted jobs. To restart, run ARC with the restart file:

.. code-block:: bash

   python /path/to/ARC/ARC.py restart.yml

In restart mode, ARC can collect finished jobs, keep waiting for jobs that are
still running, and continue work that was not completed.

Run from Python
---------------

Use the Python API when you want to generate inputs programmatically, integrate
ARC into scripts, or work in a notebook:

.. code-block:: python

   from arc import ARC
   from arc.species import ARCSpecies

   ethanol = ARCSpecies(label='ethanol', smiles='CCO')

   arc = ARC(
       project='ethanol_api_demo',
       species=[ethanol],
       job_types={
           'conf_opt': True,
           'opt': True,
           'fine': True,
           'freq': True,
           'sp': True,
           'rotors': True,
       },
       level_of_theory='wb97xd/def2svp',
       ess_settings={'gaussian': 'local'},
   )

   arc.execute()

``ARC`` also accepts species and reaction dictionaries, so YAML-style inputs can
be converted into API calls without manually constructing every object.

Run Locally
-----------

Use local mode when ARC and the relevant electronic structure software are on
the same machine or cluster login environment. Configure ``servers`` with the
reserved ``local`` key and route software to ``local``:
Here, ``local`` means ARC does not use SSH for that server. The job is still
submitted using the configured scheduler and submit template.

.. code-block:: python

   servers = {
       'local': {
           'cluster_soft': 'Slurm',
           'un': 'my_user',
       },
   }

   global_ess_settings = {
       'gaussian': 'local',
       'orca': 'local',
       'xtb': 'local',
   }

Run over SSH
------------

Use SSH mode when ARC runs on your workstation but submits jobs on one or more
remote servers. Configure each remote server with ``address``, ``un``, and
``key``, then route ESS names to those servers:

.. code-block:: python

   servers = {
       'cluster_a': {
           'cluster_soft': 'Slurm',
           'address': 'login.cluster.edu',
           'un': 'my_user',
           'key': '/home/my_user/.ssh/id_rsa',
       },
   }

   global_ess_settings = {
       'gaussian': 'cluster_a',
       'molpro': 'cluster_a',
   }

Run on HPC
----------

On HPC systems, ARC usually runs on a login or workflow node and submits ESS jobs
through the scheduler. The important site-specific pieces are:

* ``cluster_soft`` in each server entry;
* submit command paths in ``settings.py`` if the defaults do not match your site;
* submit script templates in ``submit.py``;
* scratch paths, queue names, wall time, memory, and CPU limits.

Keep the scheduler template variables such as ``{memory}``, ``{cpus}``,
``{name}``, and ``{input_file}`` intact so ARC can fill them at runtime.

Run Arkane Independently
------------------------

ARC runs Arkane automatically for supported statmech workflows. You can also run
Arkane directly:

.. code-block:: bash

   conda run -n rmg_env python -m arkane input.py

For a source installation, make sure ``RMG_PY_PATH`` and ``RMG_DB_PATH`` point to
valid checkouts before running Arkane.

.. include:: links.txt
