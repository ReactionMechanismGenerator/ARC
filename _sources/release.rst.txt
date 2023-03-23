.. _release:

Release notes
=============

ARC 1.1.0
^^^^^^^^^

- New features:

  - Project name validity check #58
  - Added Free Rotors #59
  - Report the git head in the log file #63
  - Determine rotor symmetry using a cyclic energy scan list #66
  - Improve SGE submit file #68
  - Improved restart #64
  - Parse xyz for a species from file #70
  - Troubleshoot conformer jobs if they all fail #77
  - Changed rotor inconsistency_ab #78
  - Don't crush ARC if it couldn't troubleshoot a job #79
  - Make xyz to 2D more robust (catch errors) #80
  - Check isomorphism for conformers  #84
  - Improve submit files #81
  - Troubleshoot rotor scans #85
  - Generate resonance structures before checking molecule isomorphism #87
  - Standardize xyz #88
  - Copy the frequency output file for a TS #89
  - Save the generated conformers in the species geometry folder #90
  - Allow nonisomorphic 2d #91
  - Improved management and report of job time #94
  - Don't use applyAtomEnergyCorrections for kinetics #95
  - Fix to xyz/2D loading and atom reordering in ARCSpecies #96
  - Preserve multiplicity throughout Species representations #99
  - Archive log and restart files #97
  - Run and save molecular orbitals #102
  - Allow ARCSpecies to be initialized with a conformer list file #104
  - Set default max stepsize for optimization in Gaussian #108
  - Save unconverged species (labels) #98
  - Use the Gaussian check file #100
  - Allow birad singlet to be defined and run as unrestricted #114
  - Update determine_qm_software to new Arkane #111
  - Calculating Lennard-Jones coefficients #117
  - Allow the user to define adaptive levels of theory (by heavy atoms) #119
  - Set the 1d_rotors settings directive to True by default #121
  - Updated submit scripts #123
  - Allow defining a TS species simply with .xyz guesses #124
  - Save conformer jobs in the restart file #125
  - Parse Molpro frequencies #127
  - Run ARC on a server #128
  - Activate the job_types flags for opt, freq, and sp #132
  - Modifications to how the model chemistry is treated to be consistent with Arkane #134
  - Organized job memory handling #137
  - Consider the requested job types for species already defined with xyz #139
  - Added a ZPE/freq scaling factor script to utils #136
  - Added a function to save an input file (and not run ARC on spot) #140
  - Added some Job methods and reorganized init #142
  - Added NIST to kinetics plots (off by default) #146
  - A new way of thinking about conformers #143
  - Improved ESS troubleshooting #147
  - Adapt Processor to the new Arkane output #148

- Documentation:

  - Added ARC's documentation #152

- Bug fixes:

  - Plot kinetics even if no RMG reaction matches #62
  - Treat same scan point at 0 and 360 deg correctly #67
  - Fix bug with loading molecule statmech during Rate jobs #73
  - If statement in model chemistry #101
  - Don't lose track of jobs in 'qw' status on SGE #103
  - Don't overwrite self.mol if self.mol_from_xyz is None #106
  - Remove extra parentheses in gaussian input file #120
  - Don't run a fine opt in Molpro #122
  - Set the memory per cpu slightly higher than the ESS memory #144
  - File compatibility with MAC OS (skip .DS_Store files) #145
  - Improved ARC's sleeping habits when experiencing connection errors #149
  - Minor conformers/xyz files fixes #150
  - Correcly move the Arkane YAML file #151

- Miscellaneous:

  - #109:

    - Added `examples` to .gitignore
    - Allow users to specify the number of cpu's to use per server
    - Download additional info re failed jobs (out.txt, err.txt, slurm.out)
    - Don't run orbitals jobs by default
    - Make servers in ess_settings a list
    - Allow users to specify a `levels_ess` dictionary associating levels of theory (or partial phrases of methods or basis sets) with an ESS
    - Added ess_settings to settings.py instead of the API
    - Tests: functions in main
    - Additional minor fixes

  - #112:

    - Load ess_settings correctly from dict
    - Improved species logging in project info file
    - Don't plot kinetics if it is None
    - Improved error message if rxn energetics are problematic
    - Added min_list() to Scheduler
    - Improve converter xyz functions
    - Improved xyz handling in Species (`xyz` can now be a list)

  - #133:

    - Updated isomorphic species check method name in rmgdb
    - Corrected local server check
    - Updated ESS check function name in iPython notebook

  - #141:

    - Added an iPy notebook with an xyz to SMILES script
    - Don't load the RMG database if not necessary

- New dependencies:

  - scikit-learn #116


ARC 1.0.0
^^^^^^^^^

This is the first stable version of ARC. See the API.

It has reasonable capabilities to calculate thermodynamic properties for arbitrary species.
It has the capability of calculating reaction rates, yet a user guess for the
transition state geometry should be provided.
(ARC can use `AutoTST`__ to generate such guesses for hydrogen abstraction reactions only,
though not all chemical elements are supported by AutoTST)

__ autotst_


Version style
^^^^^^^^^^^^^

ARC uses `Semantic Versioning`__

__ semantic_


.. include:: links.txt
