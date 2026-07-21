.. _examples:

Examples
========

This page shows compact, current input patterns. Use them as starting points,
then adapt levels of theory, software routing, and server settings to your site.

Minimal Species Thermochemistry
-------------------------------

.. code-block:: yaml

   project: ethanol_thermo

   species:
     - label: ethanol
       smiles: CCO

   level_of_theory: wb97xd/def2svp
   ess_settings:
     gaussian: local

This asks ARC to use the default job types, which include conformer optimization,
optimization, fine-grid optimization, frequency, single-point, rotor, and IRC
where applicable.

Several Species and Coordinate Formats
--------------------------------------

.. code-block:: yaml

   project: species_inputs

   ess_settings:
     gaussian:
       - local
       - server1
     orca: local

   job_types:
     conf_opt: true
     opt: true
     fine: true
     freq: true
     sp: true
     rotors: true

   level_of_theory: CCSD(T)-F12/cc-pVTZ-F12//wb97xd/def2tzvp

   species:
     - label: propane
       smiles: CCC

     - label: propanol
       inchi: InChI=1S/C3H8O/c1-2-3-4/h4H,2-3H2,1H3

     - label: vinoxy
       xyz: |
         O       1.35170118   -1.00275231   -0.48283333
         C      -0.67437022    0.01989281    0.16029161
         C       0.62797113   -0.03193934   -0.15151370
         H      -1.14812497    0.95492850    0.42742905
         H      -1.27300665   -0.88397696    0.14797321
         H       1.11582953    0.94384729   -0.10134685

Reaction with a TS Guess
------------------------

This is a reaction-definition fragment. Add ``level_of_theory`` and
``ess_settings`` for the ESS you intend to use before running it.

.. code-block:: yaml

   project: reaction_demo

   species:
     - label: N2H4
       smiles: NN
     - label: NH
       smiles: '[NH]'
     - label: N2H3
       smiles: N[NH]
     - label: NH2
       smiles: '[NH2]'

   reactions:
     - label: N2H4 + NH <=> N2H3 + NH2
       ts_xyz_guess:
         - |
           N      -0.4465194713     0.6830090994    -0.0932618217
           H      -0.4573825998     1.1483344874     0.8104886823
           H       0.6773598975     0.3820642106    -0.2197000290
           N      -1.2239012380    -0.4695695875    -0.0069891203
           H      -1.8039356973    -0.5112019151     0.8166872835
           H      -1.7837217777    -0.5685801608    -0.8405154279
           N       1.9039017235    -0.1568337145    -0.0766247796
           H       1.7333130781    -0.8468572038     0.6711695415

Transition-State Species
------------------------

Use ``is_ts: true`` when the species entry itself is a transition state:
This example is a fragment; add ``ess_settings`` for your configured ESS before
running it.

.. code-block:: yaml

   project: ts_opt

   level_of_theory: apfd/def2TZVPP
   freq_scale_factor: 0.992

   species:
     - label: TS1
       is_ts: true
       multiplicity: 2
       xyz:
         - TS1/guess_1.gjf
         - TS1/guess_2.gjf
         - TS1/guess_3.gjf

ARC can read coordinate guesses from multiline XYZ strings, XYZ files, Gaussian
input files, and supported ESS output files.

Python API
----------

.. code-block:: python

   from arc import ARC
   from arc.species import ARCSpecies
   from arc.reaction import ARCReaction

   n2h4 = ARCSpecies(label='N2H4', smiles='NN')
   nh = ARCSpecies(label='NH', smiles='[NH]')
   n2h3 = ARCSpecies(label='N2H3', smiles='N[NH]')
   nh2 = ARCSpecies(label='NH2', smiles='[NH2]')

   reaction = ARCReaction(
       label='N2H4 + NH <=> N2H3 + NH2',
       r_species=[n2h4, nh],
       p_species=[n2h3, nh2],
   )

   arc = ARC(
       project='api_reaction_demo',
       species=[n2h4, nh, n2h3, nh2],
       reactions=[reaction],
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

Use ``ARCSpecies`` for species definitions, ``ARCReaction`` for reactions, and
``ARC`` for project execution. The same keyword names used in YAML are generally
accepted by the Python API.

Job-Type Shortcuts
------------------

To run only one job family, use ``specific_job_type``:
This example is a fragment; add ``level_of_theory`` and ``ess_settings`` before
running it.

.. code-block:: yaml

   project: bde_demo
   specific_job_type: bde

   species:
     - label: ethanol
       smiles: CCO

For BDE workflows, ARC automatically enables the supporting optimization,
fine-grid, frequency, and single-point jobs required by the workflow.

Where to Find More Inputs
-------------------------

The repository includes more examples and fixtures:

* ``examples`` - maintained YAML examples for minimal, stationary-species,
  reaction, BDE, TS, and developer/mock workflows;
* ``ipython/Demo`` - tutorial notebooks and YAML examples;
* ``ipython/Tools`` - notebook tools that exercise ARC utilities;
* ``arc/testing/yml_testing`` - YAML fixtures used by tests;
* ``arc/testing/restart`` - restart-file examples.

.. include:: links.txt
