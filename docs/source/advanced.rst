.. _advanced:

Advanced Features
=================

.. _flexXYZ:

Flexible coordinates (xyz) input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The xyz attribute of an :ref:`ARCSpecies <species>` objects (whether TS or not) is extremely flexible.
It could be a multiline string containing the coordinates, or a list of several multiline strings.
It could also contain valid file paths to ESS input files, output files,
`XYZ format`__ files, or ARC's conformers (before/after optimization) files.
See :ref:`the examples <examples>`.

__ xyz_format_


Using a fine DFT grid for optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This option is turned on by default. If you'd like to turn it off,
set ``fine`` in the ``job_types`` dictionary to `False`.

If turned on, ARC will spawn another optimization job after the first one converges
with a fine grid settings, using the already optimized geometry.

Note that this argument is called ``fine`` in ARC, although in practice
it directs the ESS to use an **ultrafine** grid. See, for example, `this study`__
describing the importance of a DFT grid.

__ DFTGridStudy_

In Gaussian, this will add the following keywords::

    scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)

In QChem, this will add the following directives::

   GEOM_OPT_TOL_GRADIENT     15
   GEOM_OPT_TOL_DISPLACEMENT 60
   GEOM_OPT_TOL_ENERGY       5
   XC_GRID                   3


Rotor scans
^^^^^^^^^^^

This option is turned on by default. If you'd like to turn it off,
set ``1d_rotors`` in the ``job_types`` dictionary to `False`.

ARC will perform 1D rotor scans for all possible unique internal rotors in the species,

The rotor scan resolution is 8 degrees by default (scanning 360 degrees overall).
This can be changed via the ``rotor_scan_resolution`` parameter in the settings.
Rotors are invalidated (not used for thermo / rate calculations) if at least one barrier
is above a maximum threshold (40 kJ/mol by default), if the scan is inconsistent by more than 30%
between two consecutive points, or if the scan is inconsistent by more than 5 kJ/mol
between the initial anf final points.
All of the above settings can be modified in the settings.py file.


Directed rotor scans
^^^^^^^^^^^^^^^^^^^^

ARC can also spawn a series of constrained (fixed dihedral) optimization jobs for an internal rotor scan.
This feature is mainly useful to conduct scans in ESS that do not offer a robust scan feature,
or do not offer one at all. There are prncipally two approaches here, a "continuous" directed scan,
and a "brute" directed scan. The former ("continuous") approach spawns the optimization jobs in serial where each job
relays on the final optimization geometry of the previous point in the scan. The latter ("brute" force) spawn all
the optimization jobs in parallel, all stemming from the global minimum conformation of the species (each job
has a different dihedral angle which is kept frozen for the desired pivot).
The "continuous" approach is closer to the phisical reality, while the "brute" approach is more robust.
Note that the brute force approach is also in ARC's arsenal of internal rotation troubleshooting methods.

To use directed internal rotation scans, make sure ``1d_rotors`` in the ``job_types`` dictionary is set to `True`
(this is the default value). Next, specify the specific pivots in the species for which a directed scan should
be spawned using the ``directed_rotors`` attribute of the species class. Note that instead of specifying pivots,
you can also specify ``all`` to treat all internal rotations of the species using this method.
Here are two (simple) examples::


    species:

    - label: ethane_1
      smiles: CC
      xyz: |
        C      -0.76058300    0.01581500    0.05881300
        C       0.76058300   -0.01581500   -0.05881200
        H      -1.22350200   -0.74245600   -0.57963900
        H      -1.08284500   -0.17477700    1.08677700
        H      -1.16115500    0.98931900   -0.23902800
        H       1.22350200    0.74245600    0.57964000
        H       1.16115500   -0.98931900    0.23902900
        H       1.08284500    0.17477700   -1.08677700
      directed_rotors:
        cont:
        - 1
        - 2

    - label: ethane_2
      smiles: CC
      xyz: |
        C      -0.76058300    0.01581500    0.05881300
        C       0.76058300   -0.01581500   -0.05881200
        H      -1.22350200   -0.74245600   -0.57963900
        H      -1.08284500   -0.17477700    1.08677700
        H      -1.16115500    0.98931900   -0.23902800
        H       1.22350200    0.74245600    0.57964000
        H       1.16115500   -0.98931900    0.23902900
        H       1.08284500    0.17477700   -1.08677700
      directed_rotors:
        brute:
        - all


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

However, at times a user might know in advance that a particular additional keyword
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
is performed on the optimized geometry (all conformers and final optimization).
If the molecule perceived from the 3D coordinate is not isomorphic
with the input 2D graph, ARC will not spawn any additional jobs for the species, and will not use it further
(for thermo and/or rates calculations). However, sometimes the perception algorithm doesn't work as expected (e.g.,
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


Don't generate conformers for specific species
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``conformers`` entry in the ``job_types`` dictionary is a global flag,
affecting conformer generation of all species in the project.

If you'd like to avoid generating conformers just for selected species,
pass their labels to ARC in the ``dont_gen_confs`` list, e.g.::

    project: arc_demo_selective_confs

    dont_gen_confs:
    - propanol

    species:

    - label: propane
      smiles: CCC
      xyz: |
        C   0.0000000   0.0000000   0.5863560
        C   0.0000000   1.2624760  -0.2596090
        C   0.0000000  -1.2624760  -0.2596090
        H   0.8743630   0.0000000   1.2380970
        H  -0.8743630   0.0000000   1.2380970
        H   0.0000000   2.1562580   0.3624930
        H   0.0000000  -2.1562580   0.3624930
        H   0.8805340   1.2981830  -0.9010030
        H  -0.8805340   1.2981830  -0.9010030
        H  -0.8805340  -1.2981830  -0.9010030
        H   0.8805340  -1.2981830  -0.9010030

    - label: propanol
      smiles: CCCO
      xyz: |
        C  -1.4392250   1.2137610   0.0000000
        C   0.0000000   0.7359250   0.0000000
        C   0.0958270  -0.7679350   0.0000000
        O   1.4668240  -1.1155780   0.0000000
        H  -1.4886150   2.2983600   0.0000000
        H  -1.9711060   0.8557990   0.8788010
        H  -1.9711060   0.8557990  -0.8788010
        H   0.5245130   1.1136730   0.8740840
        H   0.5245130   1.1136730  -0.8740840
        H  -0.4095940  -1.1667640   0.8815110
        H  -0.4095940  -1.1667640  -0.8815110
        H   1.5267840  -2.0696580   0.0000000

In the above example, ARC will only generate conformers for propane (not for propanol).
For propane, it will compare the selected conformers against the user-given xyz guess using the
conformer level DFT method, and will take the most stable structure for the rest of the calculations,
regardless of its source (ARC's conformers or the user guess). For propanol, on the other hand,
ARC will not attempt to generate conformers, and will simply use the user guess.

Note: If a species label is added to the ``dont_gen_confs`` list, but the species has no 3D
coordinates, ARC **will** generate conformers for it.


Writing an ARC input file using the API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Writing in YAML isn't very intuitive for many, especially without a good editor.
You could use ARC's API to define your objects, and then dump it all in a YAML file
which could be read as an input in ARC::

    from arc.species.species import ARCSpecies
    from arc.common import save_yaml_file

    input_dict = dict()

    input_dict['project'] = 'Demo_project_input_file_from_API'

    input_dict['job_types'] = {'conformers': True,
                               'opt': True,
                               'fine_grid': True,
                               'freq': True,
                               'sp': True,
                               '1d_rotors': True,
                               'orbitals': False,
                               'lennard_jones': False,
                              }

    spc1 = ARCSpecies(label='NO', smiles='[N]=O')

    adj1 = """multiplicity 2
    1 C u0 p0 c0 {2,D} {4,S} {5,S}
    2 C u0 p0 c0 {1,D} {3,S} {6,S}
    3 O u1 p2 c0 {2,S}
    4 H u0 p0 c0 {1,S}
    5 H u0 p0 c0 {1,S}
    6 H u0 p0 c0 {2,S}"""

    xyz2 = [
        """O       1.35170118   -1.00275231   -0.48283333
           C      -0.67437022    0.01989281    0.16029161
           C       0.62797113   -0.03193934   -0.15151370
           H      -1.14812497    0.95492850    0.42742905
           H      -1.27300665   -0.88397696    0.14797321
           H       1.11582953    0.94384729   -0.10134685""",
        """O       1.49847909   -0.87864716    0.21971764
           C      -0.69134542   -0.01812252    0.05076812
           C       0.64534929    0.00412787   -0.04279617
           H      -1.19713983   -0.90988817    0.40350584
           H      -1.28488154    0.84437992   -0.22108130
           H       1.02953840    0.95815005   -0.41011413"""]

    spc2 = ARCSpecies(label='vinoxy', xyz=xyz2, adjlist=adj1)

    spc_list = [spc1, spc2]

    input_dict['species'] = [spc.as_dict() for spc in spc_list]

    save_yaml_file(path='some/path/to/desired/folder/input.yml', content=input_dict)


The above code generated the following input file::

    project: Demo_project_input_file_from_API

    job_types:
      1d_rotors: true
      conformers: true
      fine_grid: true
      freq: true
      lennard_jones: false
      opt: true
      orbitals: false
      sp: true

    species:
    - E0: null
      arkane_file: null
      bond_corrections:
        N=O: 1
      charge: 0
      external_symmetry: null
      force_field: MMFF94
      generate_thermo: true
      is_ts: false
      label: 'NO'
      long_thermo_description: 'Bond corrections: {''N=O'': 1}

        '
      mol: |
        multiplicity 2
        1 N u1 p1 c0 {2,D}
        2 O u0 p2 c0 {1,D}
      multiplicity: 2
      neg_freqs_trshed: []
      number_of_rotors: 0
      optical_isomers: null
      rotors_dict: {}
      t1: null
    - E0: null
      arkane_file: null
      bond_corrections:
        C-H: 3
        C-O: 1
        C=C: 1
      charge: 0
      conformer_energies:
      - null
      - null
      conformers:
      - |-
        O       1.35170118   -1.00275231   -0.48283333
        C      -0.67437022    0.01989281    0.16029161
        C       0.62797113   -0.03193934   -0.15151370
        H      -1.14812497    0.95492850    0.42742905
        H      -1.27300665   -0.88397696    0.14797321
        H       1.11582953    0.94384729   -0.10134685
      - |-
        O       1.49847909   -0.87864716    0.21971764
        C      -0.69134542   -0.01812252    0.05076812
        C       0.64534929    0.00412787   -0.04279617
        H      -1.19713983   -0.90988817    0.40350584
        H      -1.28488154    0.84437992   -0.22108130
        H       1.02953840    0.95815005   -0.41011413
      external_symmetry: null
      force_field: MMFF94
      generate_thermo: true
      is_ts: false
      label: vinoxy
      long_thermo_description: 'Bond corrections: {''C-O'': 1, ''C=C'': 1, ''C-H'': 3}

        '
      mol: |
        multiplicity 2
        1 O u1 p2 c0 {3,S}
        2 C u0 p0 c0 {3,D} {4,S} {5,S}
        3 C u0 p0 c0 {1,S} {2,D} {6,S}
        4 H u0 p0 c0 {2,S}
        5 H u0 p0 c0 {2,S}
        6 H u0 p0 c0 {3,S}
      multiplicity: 2
      neg_freqs_trshed: []
      number_of_rotors: 0
      optical_isomers: null
      rotors_dict: {}
      t1: null


Calculating BDEs (bond dissociation energies)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To direct ARC to calculate BDEs for a species, set the ``bde`` job type to `True`,
and set the requested atom indices (1-indexed) in the species ``.bdes`` argument
as a list of tuples representing indices of bonded atoms (via a single bond)
between which the BDE will be calculated (a list of lists is also allowed and will be converted),
E.g., a species can be defined as::

    spc1 = ARCSpecies(label='label1', smiles='CC(NO)C', xyz=xyz1, bdes=[(1, 2), (5, 8)])

Note that the `bdes` species argument also accepts the string ``'all_h'`` as one of the entries
in the list, directing ARC to calculate BDEs for all hydrogen atoms in the species.
Below is an example requesting all hydrogen BDEs in ethanol including the `C--O` BDE::

    project: ethanol_BDEs

    job_types:
      1d_rotors: true
      conformers: true
      fine_grid: true
      freq: true
      opt: true
      sp: true
      bde: true

    species:
    - label: ethanol
      smiles: CCO
      xyz: |
        O       1.20823797   -0.43654321    0.79046266
        C       0.38565457    0.37473766   -0.03466399
        C      -0.94122817   -0.32248828   -0.24592109
        H       0.89282946    0.53292877   -0.99112072
        H       0.23767951    1.34108205    0.45660206
        H      -0.79278514   -1.30029213   -0.71598886
        H      -1.43922693   -0.50288055    0.71249177
        H      -1.60098471    0.27712988   -0.87920708
        H       2.04982343    0.03632579    0.90734524
      bdes:
      - all_h
      - - 1
        - 2

Note: The BDEs are determined based on `E0`, therefore both ``sp`` and ``freq`` jobs
must be spawned (and successfully terminated for all species and fragments).

The calculated BDEs are reported in the log file as well as in a designated `BDE_report.yml`
file under the `output` directory in the project's folder. Units are kJ/mol.


.. include:: links.txt
