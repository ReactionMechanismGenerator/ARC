.. _advanced:

Advanced Features
=================

Flexible coordinates (xyz) input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The xyz attribute of an :ref:`ARCSpecies <species>` objects (whether TS or not) is extremely flexible.
It could be a multiline string containing the coordinates, or a list of several multiline strings.
It could also contain valid file paths to ESS input files, output files,
`XYZ format`__ files, or ARC's conformers (before/after optimization) files.
See :ref:`the examples <examples>`.

__ xyz_format_


Using a fine grid for optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This option is turned on by default. If you'd like to turn it off,
set ``fine`` in the ``job_types`` dictionary to `False`.

If turned on, ARC will spawn another optimization job with a fine grid
using the already optimized geometry.

In Gaussian, this will add the keywords::

    scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)

In QChem, this will add the following directives::

   GEOM_OPT_TOL_GRADIENT 15
   GEOM_OPT_TOL_DISPLACEMENT 60
   GEOM_OPT_TOL_ENERGY 5

It has no effect for Molpro optimization jobs.


Rotor scans
^^^^^^^^^^^

This option is turned on by default. If you'd like to turn it off,
set ``scan_rotors`` in the ``job_types`` dictionary to ``False``.

ARC will perform 1D rotor scans to all possible unique hindered rotors in the species,

The rotor scan resolution is 8 degrees by default (scanning 360 degrees overall).
Rotors are invalidated (not used for thermo / rate calculations) if at least one barrier
is above a maximum threshold (40 kJ/mol by defaut), if the scan is inconsistent by more than 30%
between two consecutive points, or if the scan is inconsistent by more than 5 kJ/mol
between the initial anf final points.
All of the above settings can be modified in the settings.py file.


Electronic Structure Software Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ARC currently supports the following electronic structure software (ESS):

    - `Gaussian`__
    - `Molpro <https://www.molpro.net/>`_
    - `QChem <https://www.q-chem.com/>`_

__ gaussian_

ARC also supports the following (non-ESS) software:

    - `OneDMin <https://tcg.cse.anl.gov/papr/codes/onedmin.html>`_ for Lennard-Jones transport coefficient calculations.
    - `Gromacs <http://www.gromacs.org/>`_ for molecular dynamics simulations.

You may pass an ESS settings dictionary to direct ARC where to find each software::

    ess_settings:
      gaussian:
      - server1
      - server2
      gromacs:
      - server1
      molpro:
      - server1
      onedmin:
      - server2
      qchem:
      - server1



Troubleshooting
^^^^^^^^^^^^^^^

ARC is has fairly good auto-troubleshooting methods.

However, at times a user might know in advance that a particular additional keywork
is required for the calculation. In such cases, simply pass the relevant keyword
in the ``initial_trsh`` (`trsh` stands for `troubleshooting`) dictionary passed to ARC::

    initial_trsh:
      gaussian:
      - iop(1/18=1)
      molpro:
      - shift,-1.0,-0.5;
      qchem:
      - GEOM_OPT_MAX_CYCLES 250



Gaussian check files
^^^^^^^^^^^^^^^^^^^^

ARC copies check files from previous `Gaussian`__ jobs, and uses them when spawning additional
jobs for the same species. When ARC terminates it will attempt to delete all downloaded checkfiles
(remote copies remain). To keep the check files set the ``keep_checks`` attribute to ``True`` (it is
``False`` by default).

__ gaussian_


Frequency scaling factors
^^^^^^^^^^^^^^^^^^^^^^^^^

ARC will look for appropriate available frequency scaling factors in `Arkane`_
for the respective ``freq_level``. If a frequency scaling factor isn't available, ARC will attempt
to determine it using `Truhlar's method`__. This involves spawning fine optimizations anf frequency
calculations for a dataset of 15 small molecules. To avoid this, either pass a known frequency scaling
factor using the ``freq_scale_factor`` attribute (see :ref:`examples <examples>`), or set the
``calc_freq_factor`` attribute to ``False`` (it is ``True`` by default).

__ Truhlar_


Adaptive levels of theory
^^^^^^^^^^^^^^^^^^^^^^^^^

Often we'd like to adapt the levels of theory to the size of the molecule.
To do so, pass the ``adaptive_levels`` attribute, which is a dictionary of
levels of theory for ranges of the number of heavy (non-hydrogen) atoms in the
molecule. Keys are tuples of (`min_num_atoms`, `max_num_atoms`), values are
dictionaries with ``optfreq`` and ``sp`` as keys and levels of theory as values.
Don't forget to bound the entire range between ``1`` and ``inf``, also make sure
there aren't any gaps in the heavy atom ranges. The below is in Python (not YAML) format::

    adaptive_levels = {(1, 5):      {'optfreq': 'wb97xd/6-311+g(2d,2p)',
                                     'sp': 'ccsd(t)-f12/aug-cc-pvtz-f12'},
                       (6, 15):     {'optfreq': 'b3lyp/cbsb7',
                                     'sp': 'dlpno-ccsd(t)/def2-tzvp/c'},
                       (16, 30):    {'optfreq': 'b3lyp/6-31g(d,p)',
                                     'sp': 'wb97xd/6-311+g(2d,2p)'},
                       (31, 'inf'): {'optfreq': 'b3lyp/6-31g(d,p)',
                                     'sp': 'b3lyp/6-311+g(d,p)'}}



Isomorphism checks
^^^^^^^^^^^^^^^^^^

When a species is defined using a 2D graph (i.e., SMILES, AdjList, or InChI), an isomorphism check
is performed on the optimized geometry. If the molecule perceived from the 3D coordinate is not isomorphic
with the input 2D graph, ARC will not spawn any additional jobs for the species, and will not use it further
(for thermo and/or rates). However, sometimes the perception algorithm doesn't work as expected (e.g.,
issues with charged species and triplets are known). To continue spawning jobs for all species in an ARC
project, pass `True` to the ``allow_nonisomorphic_2d`` argument (it is `False` by default).


.. _directory:

Using a non-default project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ARC is run from the terminal with an input/restart file
then the folder in which that file is located becomes the Project's folder.
If ARC is run using the API, a folder with the Project's name is created under ``ARC/Projects/``.
To change this behavior, you may request a specific project folder. Simply pass the desired project
folder path using the ``project_directory`` argument. If the folder does not exist, ARC will create it
(and all parent folders, if necessary).


Visualizing molecular orbitals (MOs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are various ways to visualize MOs.
One way is to open a `Gaussian`__ output file
using `GaussView <https://gaussian.com/gaussview6/>`_.

__ gaussian_

ARC supports an additional way to generate high quality and good looking MOs.
Simply set the ``orbitals`` entry of the ``job_types`` dictionary to `True` (it is `False` by default`).
ARC will spawn a `QChem <https://www.q-chem.com/>`_ job with the
``PRINT_ORBITALS TRUE`` directive using `NBO <http://nbo.chem.wisc.edu/>`_,
and will copy the resulting FCheck output file.
Make sure you set the `orbitals` level of theory to the desired level in ``default_levels_of_theory``
in ``settings.py``.
Open the resulting FCheck file using `IQMol <http://iqmol.org/>`_
to post process and save images.


.. include:: links.txt
