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

Transition-State Search Adapters
--------------------------------

ARC can use several TS adapters when configured and installed, including
heuristics, AutoTST, GCN, xTB-GSM, and ORCA-NEB. Select adapters per project:

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

Restarts
--------

Restart files are normal ARC inputs with more state. To restart:

.. code-block:: bash

   conda activate arc_env
   python /path/to/ARC/ARC.py restart.yml

Keep the project directory and server-side job files available when restarting;
ARC uses them to collect and continue previously submitted work.

.. include:: links.txt
