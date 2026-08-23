.. _input_reference:

Input YAML Reference
====================

ARC input files are YAML dictionaries passed directly to ``ARC(**input_dict)``.
Top-level keys must match arguments accepted by :ref:`arc.main.ARC <main>`.
Entries under ``species`` are converted to ``ARCSpecies`` objects, and entries
under ``reactions`` are converted to ``ARCReaction`` objects.

Use this page as a practical checklist of accepted keys. For deeper behavior,
see the API reference pages for :ref:`ARC <main>`, :ref:`ARCSpecies <species>`,
:ref:`ARCReaction <reaction>`, and :ref:`Level <level>`.

Minimal Shape
-------------

.. code-block:: yaml

   project: my_project

   species:
     - label: ethanol
       smiles: CCO

   level_of_theory: wb97xd/def2svp
   ess_settings:
     gaussian: local

Top-Level Keys
--------------

Project and execution:

* ``project`` - project name. Required for new projects.
* ``project_directory`` - local directory for project output.
* ``verbose`` - logging level.
* ``running_jobs`` - restart state for submitted jobs.
* ``output`` - restart/status state for species.
* ``output_multi_spc`` - restart/status state for multi-species workflows.

Species and reactions:

* ``species`` - list of species dictionaries.
* ``reactions`` - list of reaction dictionaries.
* ``dont_gen_confs`` - species labels for which conformer generation should be avoided when XYZ is supplied.
* ``n_confs`` - number of low-energy force-field conformers to keep. Default: ``10``.
* ``e_confs`` - conformer energy window in kJ/mol. Default: ``5.0``.
* ``allow_nonisomorphic_2d`` - allow optimization even when generated 3D geometry is not isomorphic to the 2D graph.

Levels of theory:

* ``level_of_theory`` - shortcut for composite or ``sp//opt`` style levels.
* ``composite_method`` - composite method level.
* ``conformer_level`` - alias for ``conformer_opt_level``.
* ``conformer_opt_level`` - conformer optimization level.
* ``conformer_sp_level`` - conformer single-point level.
* ``ts_guess_level`` - level for TS guess comparison/optimization.
* ``opt_level`` - geometry optimization level.
* ``freq_level`` - frequency level.
* ``sp_level`` - single-point energy level.
* ``scan_level`` - rotor scan level.
* ``irc_level`` - IRC level.
* ``orbitals_level`` - orbitals level.
* ``arkane_level_of_theory`` - level used for Arkane atom and bond corrections.
* ``adaptive_levels`` - level choices by heavy-atom count.

Job selection and resources:

* ``job_types`` - dictionary of job type booleans.
* ``specific_job_type`` - run one job family; takes precedence over ``job_types``.
* ``job_memory`` - memory per job in GB.
* ``max_job_time`` - wall-time limit in hours.
* ``keep_checks`` - keep ESS checkfiles, both locally and on the servers.
* ``trsh_ess_jobs`` - troubleshoot failed ESS jobs. Default: ``true``.
* ``trsh_rotors`` - troubleshoot failed rotor scans. Default: ``true``.
* ``skip_nmd`` - skip normal-mode-displacement checks.
* ``report_e_elect`` - report electronic energy.

Software and statmech:

* ``ess_settings`` - map ESS names to server names for this project.
* ``thermo_adapter`` - thermochemistry adapter. Default: ``Arkane``.
* ``kinetics_adapter`` - kinetics adapter. Default: ``Arkane``.
* ``compare_to_rmg`` - compare calculated data to RMG database values. Default: ``true``.
* ``compute_thermo`` - compute thermodynamic properties. Default: ``true``.
* ``compute_rates`` - compute rate coefficients. Default: ``true``.
* ``compute_transport`` - compute transport properties. Default: ``false``.
* ``bac_type`` - bond additivity correction type: ``p``, ``m``, or ``null``.
* ``freq_scale_factor`` - harmonic frequency scaling factor.
* ``calc_freq_factor`` - calculate a scaling factor if one cannot be found. Default: ``true``.
* ``bath_gas`` - bath gas for OneDMin/Lennard-Jones workflows.

Kinetics conditions and TS search:

* ``T_min`` - minimum temperature tuple, e.g. ``[500, K]``.
* ``T_max`` - maximum temperature tuple, e.g. ``[3000, K]``.
* ``T_count`` - number of temperature points. Default: ``50``.
* ``ts_adapters`` - TS search adapters to try, such as ``heuristics``, ``AutoTST``, ``GCN``, ``xtb_gsm``, and ``orca_neb``.

Species Entries
---------------

Each entry under ``species`` may use the following keys:

Identity and structure:

* ``label`` - species label.
* ``smiles`` - SMILES string.
* ``inchi`` - InChI string.
* ``adjlist`` - RMG adjacency list.
* ``xyz`` - XYZ string, XYZ dictionary, list of XYZ entries, or path to a supported coordinate/input/output file.
* ``yml_path`` - path to a species YAML file.
* ``mol`` - RMG Molecule object when using the Python API.
* ``species_dict`` - dictionary representation used in restarts/API workflows.

Charge, spin, and TS metadata:

* ``charge`` - integer charge.
* ``multiplicity`` - spin multiplicity.
* ``number_of_radicals`` - radical count override.
* ``is_ts`` - whether this species is a transition state.
* ``ts_number`` - TS index.
* ``rxn_label`` - associated reaction label.
* ``rxn_index`` - associated reaction index.
* ``irc_label`` - IRC label.

Thermo and output behavior:

* ``compute_thermo`` - compute thermo for this species.
* ``include_in_thermo_lib`` - include species in output thermo library.
* ``e0_only`` - compute/store only E0 where applicable.
* ``bond_corrections`` - custom bond correction data.
* ``bdes`` - bond dissociation energy atom-pair list.
* ``checkfile`` - ESS checkfile path.
* ``run_time`` - restart/runtime metadata.

Conformers, rotors, and geometry controls:

* ``force_field`` - force field for conformer generation. Default: ``MMFF94s``.
* ``directed_rotors`` - directed rotor scan requests.
* ``preserve_param_in_scan`` - internal coordinates to preserve during scans.
* ``consider_all_diastereomers`` - consider all diastereomers during conformer generation.
* ``fragments`` - fragment atom indices.
* ``active`` - active-space metadata.
* ``external_symmetry`` - external symmetry number.
* ``optical_isomers`` - number of optical isomers.
* ``multi_species`` - multi-species label.
* ``keep_mol`` - preserve the molecule object.
* ``project_directory`` - species-specific project directory metadata.

Reaction Entries
----------------

Each entry under ``reactions`` may use the following keys:

* ``label`` - reaction label, commonly ``A + B <=> C + D``.
* ``reactants`` - list of reactant labels.
* ``products`` - list of product labels.
* ``r_species`` - reactant ``ARCSpecies`` objects when using the Python API.
* ``p_species`` - product ``ARCSpecies`` objects when using the Python API.
* ``ts_label`` - associated TS species label.
* ``ts_xyz_guess`` - TS guess or list of guesses.
* ``xyz`` - alias for ``ts_xyz_guess``.
* ``family`` - RMG reaction family.
* ``multiplicity`` - reaction PES multiplicity.
* ``charge`` - reaction PES charge.
* ``preserve_param_in_scan`` - scan constraints for the reaction/TS.
* ``kinetics`` - user-supplied kinetics data.
* ``reaction_dict`` - dictionary representation used in restarts/API workflows.
* ``species_list`` - species object list used when reconstructing from dictionaries.

Level Dictionaries
------------------

Any level key can be a string:

.. code-block:: yaml

   opt_level: wb97xd/def2tzvp

or a dictionary:

.. code-block:: yaml

   sp_level:
     method: DLPNO-CCSD(T)-F12
     basis: cc-pVTZ-F12
     auxiliary_basis: aug-cc-pVTZ/C
     cabs: cc-pVTZ-F12-CABS
     software: orca

Accepted level dictionary keys are:

* ``method``
* ``basis``
* ``auxiliary_basis``
* ``dispersion``
* ``cabs``
* ``method_type``
* ``software``
* ``software_version``
* ``compatible_ess``
* ``solvation_method``
* ``solvent``
* ``solvation_scheme_level``
* ``args``
* ``year``

Use ``year`` only for Arkane correction matching, not inside QC method names.

Job Types
---------

Current job type keys are:

* ``conf_opt``
* ``conf_sp``
* ``opt``
* ``fine``
* ``freq``
* ``sp``
* ``rotors``
* ``irc``
* ``orbitals``
* ``stability``
* ``onedmin``
* ``bde``

Legacy aliases are normalized by ARC:

* ``fine_grid`` -> ``fine``
* ``lennard_jones`` -> ``onedmin``

Example:

.. code-block:: yaml

   job_types:
     conf_opt: true
     opt: true
     fine: true
     freq: true
     sp: true
     rotors: false
     stability: false

``conf_opt``, ``opt``, ``fine``, ``freq``, ``sp``, ``rotors`` and ``irc`` default to
``true`` when omitted; ``conf_sp``, ``orbitals``, ``stability``, ``onedmin`` and ``bde``
default to ``false``.

Wavefunction Stability Analysis
-------------------------------

``stability`` is off by default and is opt-in through ``job_types``. ARC has implemented
the analysis for Gaussian and for ORCA so far; other ESSs are not wired up yet, and a run
whose optimization jobs go to another ESS is told so once per ESS in the log rather than
silently producing nothing. The two ESSs test the same space and agree on the verdict; in
ORCA the analysis always follows an instability it finds, since ORCA 6.0.0 aborts rather
than merely reporting one, and the sector of a restricted reference's instability is read
off the solution it relaxes into.

It runs once per species, as soon as that species' optimization converges and before the
frequency job, the single point, the IRC and the rotor scans, so that those jobs inherit
the reference and the geometry the verdict settles on. It runs only for a transition state
or for a species whose optimization actually ran with a restricted reference - a restricted
reference is the only one the analysis can inform, since a restricted solution gives the
same energy as an unrestricted one if and only if it is stable. It is further limited to
DFT and Hartree-Fock optimization levels, and needs the checkfile the optimization job
wrote, so that the SCF under test is the converged one the later jobs build on.

What it buys you: a verdict recorded in the log and in ``output.yml`` saying whether the
converged wavefunction is a genuine minimum in the space of orbital rotations, together
with the label and eigenvalue of any negative stability-matrix root, and whether the
analytic frequencies are invalidated by it. For a transition state the verdict can also
decide the restricted-versus-unrestricted reference of the jobs that follow. See
:ref:`Advanced Features <advanced>` for the full treatment, including which instabilities
invalidate analytic frequencies and what an adopted verdict does and does not correct.

``specific_job_type`` cannot be used to request it. That key replaces ``job_types``
wholesale with a dictionary in which only the named job type is ``true``, so
``specific_job_type: stability`` switches off the ``opt``, ``freq`` and ``sp`` jobs the
analysis is spawned from and nothing runs at all (``bde`` is special-cased to re-enable
them; ``stability`` is not). Any other value of ``specific_job_type`` likewise sets
``stability`` to ``false``. Request it through ``job_types``.

ESS Settings
------------

``ess_settings`` maps software names to configured server names:

.. code-block:: yaml

   ess_settings:
     gaussian:
       - local
       - cluster_a
     orca: local
     molpro: cluster_b

Supported ESS keys include ``cfour``, ``gaussian``, ``mockter``, ``molpro``,
``orca``, ``qchem``, ``terachem``, ``onedmin``, ``xtb``, ``torchani``, and
``openbabel``. TS adapters such as ``heuristics``, ``AutoTST``, ``GCN``,
``xtb_gsm``, and ``orca_neb`` are configured through ``ts_adapters`` and related
installation/settings.

Complete Example
----------------

.. code-block:: yaml

   project: ethanol_reference_demo
   project_directory: /tmp/arc_ethanol_reference_demo

   ess_settings:
     gaussian: local

   job_types:
     conf_opt: true
     opt: true
     fine: true
     freq: true
     sp: true
     rotors: true
     irc: false

   level_of_theory: wb97xd/def2svp
   sp_level:
     method: wb97xd
     basis: def2tzvp
     software: gaussian

   job_memory: 14
   max_job_time: 24
   compute_thermo: true
   compute_rates: false
   bac_type: null

   species:
     - label: ethanol
       smiles: CCO
       multiplicity: 1
       charge: 0

.. include:: links.txt
