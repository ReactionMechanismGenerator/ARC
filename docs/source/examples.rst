.. _examples:

Examples
========

ARC's examples folder (`ARC/examples`__) is an excellent resource for up-to-date sample input files,
demonstrating different features. Below are examples for common basic tasks performed using ARC.

__ examplesGitHub_


Calculating thermodynamic properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example below shows how to set the levels of theory and job types.
Multiple species are defined using SMILES / an XYZ list / InChI::

    project: arc_demo_1

    ess_settings:
      gaussian:
      - local
      - server1
      molpro:
      - server1
      qchem:
      - server2

    job_types:
      rotors: true
      conformers: true
      fine: true
      freq: true
      opt: true
      sp: true

    max_job_time: 24

    level_of_theory: CCSD(T)-F12/cc-pVTZ-F12//wb97xd/def2tzvp
    scan_level: wb97xd/def2tzvp
    conformer_level: b3lyp/6-311+g(d,p)

    species:

    - label: propane
      smiles: CCC

    - label: vinoxy
      xyz:
      - |
        O       1.35170118   -1.00275231   -0.48283333
        C      -0.67437022    0.01989281    0.16029161
        C       0.62797113   -0.03193934   -0.15151370
        H      -1.14812497    0.95492850    0.42742905
        H      -1.27300665   -0.88397696    0.14797321
        H       1.11582953    0.94384729   -0.10134685
      - |
        O       1.49847909   -0.87864716    0.21971764
        C      -0.69134542   -0.01812252    0.05076812
        C       0.64534929    0.00412787   -0.04279617
        H      -1.19713983   -0.90988817    0.40350584
        H      -1.28488154    0.84437992   -0.22108130
        H       1.02953840    0.95815005   -0.41011413
      - |
        O      -1.15888237    1.27653343    0.30527086
        C       0.63650559   -0.15873769   -0.22478280
        C      -0.59108268    0.16116823    0.20729283
        H       1.31343166    0.61755575   -0.56336439
        H       0.97181468   -1.18699193   -0.24372402
        H      -1.17178687   -0.70952779    0.51930751

    - label: propanol
      inchi: InChI=1S/C3H8O/c1-2-3-4/h4H,2-3H2,1H3

In the above example, ``level_of_theory`` is a directive that specifies in one line the
`sp_method/sp_basis_set//opt_method/opt_basis_set`. Specifying the ``level_of_theory``
also sets the ``freq_level`` as equal to the ``opt_level``. Note that the ``scan_level``
isn't set, and ARC will use the default in settings.py unless directed otherwise.

To specify a composite method, simply define something like::

    level_of_theory: CBS-QB3

Note that for composite methods the default ``freq_level`` and ``scan_level`` may have different
default values than for non-composite methods (defined in settings.py). Yes, an independent
frequencies calculation job is executed after a composite job just so that the Hamiltonian will
be outputted.

The same example as above ran via the API (e.g., in `Jupyter notebooks`__) would look like the following::

    from arc import ARC
    from arc.species import ARCSpecies
    from IPython.display import display
    %matplotlib notebook

    spc1 = ARCSpecies(label='propane', smiles='CCC')

    xyz_list = ["""O       1.35170118   -1.00275231   -0.48283333
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
                   H       1.02953840    0.95815005   -0.41011413""",
                """O      -1.15888237    1.27653343    0.30527086
                   C       0.63650559   -0.15873769   -0.22478280
                   C      -0.59108268    0.16116823    0.20729283
                   H       1.31343166    0.61755575   -0.56336439
                   H       0.97181468   -1.18699193   -0.24372402
                   H      -1.17178687   -0.70952779    0.51930751"""]

    spc2 = ARCSpecies(label='vinoxy', xyz=xyz_list)

    spc3 = ARCSpecies(label='propanol', inchi='InChI=1S/C3H8O/c1-2-3-4/h4H,2-3H2,1H3')

    arc0 = ARC(project='arc_demo_1',
               ess_settings={'gaussian': ['local', 'server1'], 'molpro': 'server1', 'qchem': 'server2'},
               job_types={'rotors': True, 'conformers': True, 'fine': True, 'freq': True, 'opt': True, sp: True},
               max_job_time=24,
               level_of_theory='CCSD(T)-F12/cc-pVTZ-F12//wb97xd/def2tzvp',
               scan_level='wb97xd/def2tzvp',
               conformer_level='b3lyp/6-311+g(d,p)')

    arc0.execute()

__ jupyter_


Optimizing transition states
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's an example for optimizing a transition state (TS) after generating several guesses, say in GaussView::

    project: BNdTSs

    level_of_theory: apfd/def2TZVPP
    freq_scale_factor: 0.992

    species:
      - label: TS1
        is_ts: true
        multiplicity: 2
        xyz:
          - TS1/1.gjf
          - TS1/2.gjf
          - TS1/3.gjf
          - TS1/4.gjf
          - TS1/5.gjf
          - TS1/6.gjf
      - label: TS3
        is_ts: true
        multiplicity: 2
        xyz:
          - TS3/1.gjf
          - TS3/2.gjf
          - TS3/3.gjf
          - TS3/4.gjf

In the above example we're using `.gjf` files (Gaussian job files) that contain the coordinate guesses.
You could define ``xyz`` using various forms, see :ref:`Flexible coordinates (xyz) input <flexXYZ>`.

Note that the main difference is the ``is_ts`` flag which is set to `True` (it is `False` by default).

.. include:: links.txt
