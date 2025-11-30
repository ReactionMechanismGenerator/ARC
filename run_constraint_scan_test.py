
import shutil

import os

from arc import ARC

from arc.species import ARCSpecies

from arc.reaction import ARCReaction

from arc.species.species import TSGuess



print("Script started.")



# 1. Define Species

# Using simple SMILES/XYZ. 

# We use a "Bad Guess" for the TS: Reactants 5 Angstroms apart.

# This will optimize to a Van der Waals complex (0 imaginary freqs), failing the TS check.

xyz_bad_guess = """

C       0.00000000    0.00000000    0.00000000

H       0.62911800    0.62911800    0.62911800

H      -0.62911800   -0.62911800    0.62911800

H      -0.62911800    0.62911800   -0.62911800

H       0.62911800   -0.62911800   -0.62911800

O       0.00000000    0.00000000    5.00000000

H       0.00000000    0.00000000    5.96000000

"""



print("Defining species...")

spc_ch4 = ARCSpecies(label='CH4', smiles='C')

spc_oh = ARCSpecies(label='OH', smiles='[OH]')

spc_ch3 = ARCSpecies(label='CH3', smiles='[CH3]')

spc_h2o = ARCSpecies(label='H2O', smiles='O')

print("Species defined.")



# 2. Define Reaction

# We explicitly name the TS so we can define it beforehand

print("Defining reaction...")

rxn = ARCReaction(label='CH4 + OH <=> CH3 + H2O',

                  r_species=[spc_ch4, spc_oh],

                  p_species=[spc_ch3, spc_h2o],

                  ts_label='TS_CH4_OH')

print("Reaction defined.")



# 3. Define TS Species with Bad Guess

# Create a TSGuess that looks like it came from an automated method

print("Defining TS species with bad guess...")

bad_guess = TSGuess(method='fake_method',

                    xyz=xyz_bad_guess,

                    success=True,

                    index=0)



ts_species = ARCSpecies(label='TS_CH4_OH',

                        is_ts=True,

                        charge=0,

                        multiplicity=2,

                        xyz=xyz_bad_guess) # Initial xyz



ts_species.ts_guesses = [bad_guess]

ts_species.ts_conf_spawned = True 

ts_species.tsg_spawned = True

ts_species.ts_guess_priority = False # CRITICAL: Ensure it's NOT priority

ts_species.chosen_ts = 0

print("TS species defined.")



# 4. Initialize ARC

project = 'ConstraintScanTest'



# Configure servers... (kept as is)

arc_servers = {

    'zeus': {

        'cluster_soft': 'PBS', 

        'address': 'zeus.technion.ac.il',

        'un': 'calvin.p',

        'key': '/home/calvin/.ssh/id_ed25519', 

    },

}

arc_ess_settings = {'gaussian': ['zeus']}



print("Initializing ARC object...")

arc_obj = ARC(project=project,

              species=[spc_ch4, spc_oh, spc_ch3, spc_h2o, ts_species], # Pass TS species here

              reactions=[rxn],

              opt_level='b3lyp/6-31g(d)',

              freq_level='b3lyp/6-31g(d)',

              ts_guess_level='b3lyp/6-31g(d)',

              ts_adapters=[], 

              job_types={'opt': True, 'freq': True, 'sp': False, 'rotors': False},

              allow_nonisomorphic_2d=True,

              compare_to_rmg=False,

              )

print("ARC object initialized.")



print("\nSETUP COMPLETE: Injected bad TS guess via species list. Starting ARC execution...")

print("Expectation: ")

print("1. 'fake_method' guess will fail freq check (0 imag freqs).")

print("2. switch_ts will be called.")

print("3. Constraint Scan will be triggered.")

print("4. New TS guess will be created from scan max.")

print("5. Optimization of new guess will start.")



# 5. Run ARC

arc_obj.execute()

print("ARC execution finished.")
