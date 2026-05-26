.. _how_it_works:

How ARC Works
=============

ARC is organized around a small set of core objects and a scheduler that turns
chemical intent into electronic structure jobs. This page gives the system map:
what happens after ARC reads an input file, which objects own which data, and
where outputs come from.

High-Level Workflow
-------------------

An ARC run follows this general path:

1. Read an ``input.yml`` or restart file.
2. Build an :ref:`ARC <main>` project object.
3. Convert ``species`` entries into :ref:`ARCSpecies <species>` objects.
4. Convert ``reactions`` entries into :ref:`ARCReaction <reaction>` objects.
5. Resolve levels of theory, job types, ESS routing, and project directories.
6. Start the scheduler.
7. Generate conformers, TS guesses, rotor scans, and ESS jobs as requested.
8. Submit jobs locally or through SSH/scheduler adapters.
9. Parse completed outputs, troubleshoot failures, and update ``restart.yml``.
10. Run statmech processing and write output libraries, plots, status files, and
    processed geometries.

.. graphviz::
   :caption: The main ARC run loop.

   digraph arc_workflow {
      graph [rankdir=LR, bgcolor="transparent"];
      node [shape=box, style="rounded,filled", fillcolor="#f5f7fa", color="#9fb3c8", fontname="Helvetica"];
      edge [color="#5b6676", fontname="Helvetica"];

      input [label="input.yml\nor restart.yml"];
      arc [label="ARC project"];
      objects [label="ARCSpecies\nARCReaction"];
      scheduler [label="Scheduler"];
      jobs [label="JobAdapter\nESS jobs"];
      parse [label="Parsers\nTroubleshooting"];
      statmech [label="Arkane/statmech\nprocessing"];
      output [label="restart.yml\nstatus.yml\nlibraries\nplots"];

      input -> arc -> objects -> scheduler -> jobs -> parse -> scheduler;
      parse -> statmech -> output;
      scheduler -> output [label="updates"];
   }

Core Objects
------------

``ARC``
    The project entry point. It owns user-facing project options such as
    ``project``, ``species``, ``reactions``, ``job_types``, levels of theory,
    ESS routing, statmech settings, resource settings, and restart state.

``ARCSpecies``
    The species representation. It stores labels, graph representations,
    charge/multiplicity, XYZ data, conformers, rotor information, thermo data,
    TS metadata, BDE requests, and output-related state.

``ARCReaction``
    The reaction representation. It stores reactant/product labels or species
    objects, atom mapping state, reaction charge/multiplicity, TS guesses, TS
    labels, family information, and kinetics data.

``Scheduler``
    The central job orchestrator. It decides which jobs are needed, creates job
    adapters, submits jobs, polls job status, parses results, applies
    troubleshooting, and advances species/reaction state.

``Level``
    The level-of-theory representation. It normalizes method, basis, ESS,
    auxiliary basis, F12 CABS, solvation, arguments, and Arkane correction-year
    metadata.

Job Lifecycle
-------------

Most ESS jobs move through the same lifecycle:

1. ARC determines that a job type is needed for a species, TS, reaction, rotor,
   or conformer.
2. The scheduler calls the job factory to create the correct adapter.
3. The adapter writes an ESS input file and a submit script.
4. ARC submits the job through the configured local or SSH execution path.
5. The scheduler periodically checks job status.
6. When the job finishes, ARC parses geometry, energy, frequencies, scan data,
   or other output as appropriate.
7. If the job failed and troubleshooting is enabled, ARC classifies the failure,
   modifies settings or input, and resubmits when a recovery path exists.
8. ARC updates output state and ``restart.yml`` so the project can continue
   after interruption.

.. graphviz::
   :caption: A typical ESS job lifecycle.

   digraph job_lifecycle {
      graph [rankdir=LR, bgcolor="transparent"];
      node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#9fb3c8", fontname="Helvetica"];
      edge [color="#5b6676", fontname="Helvetica"];

      needed [label="Job needed"];
      factory [label="job_factory()"];
      adapter [label="JobAdapter"];
      submit [label="write input\nsubmit script"];
      run [label="scheduler/ESS run"];
      done [label="parse output"];
      trsh [label="troubleshoot\nif failed"];
      state [label="update state\nrestart.yml"];

      needed -> factory -> adapter -> submit -> run -> done -> state;
      done -> trsh [label="failed"];
      trsh -> submit [label="resubmit"];
   }

Execution Paths
---------------

ARC distinguishes where it runs from where ESS jobs run:

``local``
    The reserved server name for jobs submitted from the same host or login
    environment where ARC is running. This still normally uses a scheduler such
    as Slurm, OGE/SGE, PBS, or HTCondor.

Remote SSH server
    ARC connects to a configured SSH host, writes project/job files remotely,
    submits jobs through that host's scheduler, and downloads outputs for
    parsing.

Pipe mode
    An opt-in execution path for large homogeneous HPC batches. It stages many
    ready tasks into a shared task directory and uses scheduler array workers to
    claim and execute tasks.

Data Flow
---------

The most common data path is:

.. graphviz::
   :caption: Data flow from user input to processed outputs.

   digraph data_flow {
      graph [rankdir=TB, bgcolor="transparent"];
      node [shape=box, style="rounded,filled", fillcolor="#f5f7fa", color="#9fb3c8", fontname="Helvetica"];
      edge [color="#5b6676", fontname="Helvetica"];

      input [label="input.yml"];
      arc [label="ARC(...)"];
      species [label="ARCSpecies"];
      reaction [label="ARCReaction"];
      scheduler [label="Scheduler"];
      adapter [label="JobAdapter"];
      ess [label="ESS input/output"];
      parsed [label="Parsed geometry,\nenergy, freq, scans"];
      outputs [label="status.yml\nrestart.yml\ngeometry files\nRMG libraries\nplots"];

      input -> arc;
      arc -> species;
      arc -> reaction;
      species -> scheduler;
      reaction -> scheduler;
      scheduler -> adapter -> ess -> parsed -> scheduler;
      scheduler -> outputs;
   }

Restart files are not a separate format. They are expanded ARC input files that
also include accumulated state such as running jobs, output paths, and parsed
results.

Important Data Structures
-------------------------

XYZ dictionary
    ARC's internal Cartesian coordinate format. It contains ``symbols``,
    ``isotopes``, and ``coords`` tuples. Most coordinate strings and files are
    normalized into this structure.

    Template:

    .. code-block:: python

       xyz = {
           'symbols': ('C', 'O', 'H', 'H', 'H', 'H'),
           'isotopes': (12, 16, 1, 1, 1, 1),
           'coords': (
               (0.000000, 0.000000, 0.000000),
               (1.420000, 0.000000, 0.000000),
               (-0.540000, 0.935000, 0.000000),
               (-0.540000, -0.467000, 0.809000),
               (-0.540000, -0.467000, -0.809000),
               (1.780000, 0.000000, 0.960000),
           ),
       }

Z-matrix dictionary
    Internal-coordinate representation used by geometry and rotor workflows. It
    stores symbols, coordinate parameter names, parameter values, and atom maps.

    Template:

    .. code-block:: python

       zmat = {
           'symbols': ('C', 'O', 'H'),
           'coords': (
               (None, None, None),
               ('R_1_0', None, None),
               ('R_2_0', 'A_2_0_1', None),
           ),
           'vars': {
               'R_1_0': 1.42,
               'R_2_0': 1.09,
               'A_2_0_1': 109.5,
           },
           'map': {0: 0, 1: 1, 2: 2},
       }

Conformer dictionaries
    Candidate conformers contain an XYZ dictionary plus metadata such as source,
    index, force-field energy, torsion information, chirality information, and
    distance-matrix data where relevant.

    Template:

    .. code-block:: python

       conformer = {
           'xyz': xyz,
           'index': 0,
           'FF energy': 0.0,
           'source': 'RDKit',
           'torsion_dihedrals': {(1, 2, 3, 4): 180.0},
           'chirality': {},
           'dmat': None,
       }

Rotor dictionaries
    Species rotor information is stored by rotor index. Entries include pivots,
    tops, scan definitions, torsions, scan paths, invalidation reasons, symmetry,
    and scan results.

    Template:

    .. code-block:: python

       rotors_dict = {
           0: {
               'pivots': [1, 2],
               'top': [2, 3, 4],
               'scan': [1, 2, 3, 4],
               'torsion': [0, 1, 2, 3],
               'success': None,
               'invalidation_reason': '',
               'scan_path': '/path/to/scan/output.out',
               'max_e': None,
               'trsh_counter': 0,
               'trsh_methods': [],
               'symmetry': None,
               'dimensions': 1,
               'directed_scan': {},
           },
       }

Status dictionaries
    ARC writes structured status information for species, TSs, reactions, job
    convergence, paths, warnings, and errors so users can inspect project state
    without reading every output file.

    Representative ``output/status.yml`` shape:

    .. code-block:: yaml

       species:
         ethanol:
           converged: true
           isomorphism: true
           paths:
             geo: /path/to/output/Species/ethanol/geometry/ethanol.xyz
             freq: /path/to/calcs/Species/ethanol/freq_a123/output.out
           warnings: []
           errors: []
       reactions:
         ethanol + OH <=> products:
           converged: false
           warnings:
             - TS search incomplete

Job dictionaries
    The scheduler tracks active and completed jobs by species/reaction label and
    job type. This is the state that lets ARC restart after interruption.

    Simplified shape:

    .. code-block:: python

       job_dict = {
           'ethanol': {
               'conf_opt': {0: '<JobAdapter>'},
               'opt': {'opt_a123': '<JobAdapter>'},
               'freq': {'freq_a124': '<JobAdapter>'},
               'sp': {'sp_a125': '<JobAdapter>'},
           },
       }

.. graphviz::
   :caption: Common ARC data structures and where they are used.

   digraph data_structures {
      graph [rankdir=LR, bgcolor="transparent"];
      node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#9fb3c8", fontname="Helvetica"];
      edge [color="#5b6676", fontname="Helvetica"];

      xyz [label="XYZ dict"];
      zmat [label="Z-matrix dict"];
      conformer [label="Conformer dict"];
      rotor [label="Rotor dict"];
      species [label="ARCSpecies"];
      scheduler [label="Scheduler"];
      output [label="status.yml\nrestart.yml"];

      xyz -> species;
      zmat -> species;
      conformer -> species;
      rotor -> species;
      species -> scheduler;
      scheduler -> output;
   }

Where Files Are Written
-----------------------

Local project outputs are written under ``project_directory``. If no explicit
directory is given, command-line runs use the input file directory and API runs
default to ``ARC/Projects/<project>``.

Inside a project, the key paths are:

* ``arc.log`` - human-readable execution log.
* ``restart.yml`` - restartable ARC state.
* ``calcs/`` - job inputs, submit scripts, raw ESS outputs, and auxiliary files.
* ``output/status.yml`` - project status summary.
* ``output/Species`` - processed species outputs and geometries.
* ``output/rxns`` - processed reaction outputs.
* ``output/RMG libraries`` - thermo and kinetics libraries when generated.

Remote jobs use project directories on the selected server. The optional
``path`` key in a server definition controls the base path for remote project
storage.

Extension Points
----------------

ARC is intentionally adapter-driven:

* ESS support lives in job adapters.
* Parser support lives in parser adapters and parser helpers.
* TS search methods are registered adapters.
* Statmech processing is selected through statmech adapters.
* Site-specific scheduler and submit behavior is configured in ``settings.py``
  and ``submit.py`` overrides.

For normal users, these extension points mostly appear as input/settings choices.
For developers, they are the places to add new ESS integrations, TS generation
methods, statmech backends, or scheduler behavior.

Related Pages
-------------

* :ref:`input_reference` for accepted YAML keys.
* :ref:`running` for execution modes.
* :ref:`advanced` for levels, job types, resources, troubleshooting, and pipe mode.
* :ref:`output` for generated files and project structure.
* :ref:`api` for the detailed Python API.

.. include:: links.txt
