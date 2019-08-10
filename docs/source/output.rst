.. _output:

Output
======

If ARC is run from the terminal with an input/restart file
then the folder in which that file is located becomes the Project's folder.
If ARC is run using the API, a folder with the Project's name is created under ``ARC/Projects/``.
It is therefore recommended to give different names for different projects you run.
A specific output folder can also be asked, see :ref:`Advanced Features <directory>`.

A respective project folder is also created on each of the servers ARC uses for a
specific Project. This remote folder can be found under ``~/runs/ARC_Projects/`` on each server.
It is helpful to look at these files in case calculations aren't performing as expected (crush).
Some additional information ARC has not downloaded may be found on the server.

After running a Project, the local Project folder will contain the following directory tree
(**bold** face represents folders, *italics* face represents files):

- *arc.log*: Details of all project execution procedures.
- *<Project_name>.info*: A quick record of the levels of theory used for the project,
  the servers used and the Species / TSs / Reactions calculated with respective execution times
  (where <Project_name> is the actual Project name).
  Note the execution time also considers the server's queue wait time.
- *restart.yml*: A restart file for the Project, constantly updates while ARC is running.
- **calcs**

    - **Species**

      - **<Species label>**

        - **<Job name>**

          - *<input file>*
          - *<submit file>*
          - *<output.out>*
          - other optional files such as *check.chk*

    - **TSs**

      - **<TS label>**

        - **<Job name>**

          - *<input file>*
          - *<submit file>*
          - *<output.out>*
          - other optional files such as *check.chk*

- **output**

  - *thermo.info*: Sources of thermoproperties determined by RMG for the parity plots for H298 and S298.
  - *thermo_parity_plots.pdf*: Parity plots of ARC's calculation and RMG's values.
  - *rate_plots.pdf.pdf*: ARC's calculated rate coefficient and RMG's values (log k vs. 1000/T).
  - *status.yml*: A status file for all species and TSs (includes important paths, warnings, errors,
    and job convergence by job type)
  - **RMG libraries**

    - **thermo**

      - *<Project_name>*: An RMG thermo library of species calculated in the present project.

    - **kinetics**: An RMG kinetics library of Reactions calculated in the present project.

      - *reactions.py*
      - *dictionary.txt*

  - **Species**

    - *<species_name>_arkane_input.py*: The Arkane species file (NOT an Arkane input file).
    - *<species_name>_arkane_output.py*: The Arkane output file.
    - *<species_name>.yml*: Arkane's species YAML file.
    - *chem.inp*: A Chemkin format NASA polynomial of the thermo data for the species.
    - *species_dictionary.txt*: The adjacency list for the species.
    - **geometry**

        - *<species_name>.gjf*: A Gaussian job file for visualizing the species' coordinates in GaussView.
        - *<species_name>.xyz*: An XYZ format file for visualizing the species' coordinates, e.g., in Avogadro.
        - *freq.out*: The frequency job output file for visualizing vibrational modes.
        - *geometry.png*: An image of the species in an arbitrary orientation for quick viewing.
        - **conformers**

          - *conformers_before_optimization.txt*
          - *conformers_after_optimization.txt*
          - *conformer torsions <n>.png*

        - **rotors**

          - *[<pivot1>, [pivot2>].png*: Rotor scan images named by atom label (1-indexed) pivots.

  - **rxns**
    - <a similar structure as described for **Species** above>
- **log_and_restart_archive**
  - <archived log and restart files renamed with the date they were archived>


.. include:: links.txt
