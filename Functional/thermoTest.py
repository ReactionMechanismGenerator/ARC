from ARC import ARC
from arc.species import ARCSpecies

smiles_dict = {'propane': 'CCC',
               'Ethylamine': 'NCC',
               'butanol': 'CCCCO',
              }

species_list = list()
for label, smiles in smiles_dict.items():
    species = ARCSpecies(label=label, smiles=smiles)  # ARC also accepts InChI just XYZ coordinates
    species_list.append(species)

job_types = {'conformers': True,
             'opt': True,
             'fine_grid': False,
             'freq': True,
             'sp': True,
             'rotors': False,
             'irc': False,
            }

arc_object = ARC(project='FunctionalTesting',
                 species=species_list,
                 job_types=job_types,
                 conformer_level='gfn2',
                 level_of_theory='gfn2',
                 freq_scale_factor=1.0,
                )

arc_object.execute()

print("########", arc_object.summary(), "########")