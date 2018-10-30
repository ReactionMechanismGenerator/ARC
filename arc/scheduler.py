#!/usr/bin/env python
# encoding: utf-8

from rmgpy.species import Species
from rmgpy.reaction import Reaction

from arc.molecule.conformer import ConformerSearch
from arc.molecule.rotors import Rotors
from arc.job.job import Job
from arc.exceptions import SpeciesError

##################################################################


class Scheduler(object):
    """
    ARC Scheduler class. Creates jobs, submits, checks status, troubleshoots.
    Each species in `species_list` has to have a unique label.

    The attributes are:

    ================ =================== ===============================================================================
    Attribute        Type                Description
    ================ =================== ===============================================================================
    `project`         ``str``            The project's name. Used for naming the directory.
    `species_list`    ``list``           Contains ``RMG.Species`` objects.
    `rxn_list`        ``list``           Contains `RMG.Reaction`` objects.
    `level_of_theory` ``str``            *FULL* level of theory, e.g. 'CBS-QB3',
                                           'CCSD(T)-F12a/aug-cc-pVTZ//B3LYP/6-311++G(3df,3pd)'...
    `species_dict`    ``dict``           A dictionary of species (dictionary structure specified below)
    `ts_dict`         ``dict``           A dictionary of transition states (dictionary structure specified below)
    ================ =================== ===============================================================================

    The structure of species_dict is the following:
    species_dict = {'species_label_1': {'species_object': ``RMG.Species``,
                                        'species_status': 'init', 'opt' / 'sp, rotors' / 'done' / 'errored: <reason>',
                                        'number_of_rotors': ``int``,
                                        'rotors_dict: {1: {'pivots': pivot_list,
                                                           'top': top_list,
                                                           'scan': scan_list},
                                                       2: {}, ...
                                                      }
                                        'conformers': ``list`` of <xyz matrix with element labels>,
                                        'initial_xyz': <xyz matrix with element labels>,
                                        'final_xyz': <xyz matrix with element labels>,
                                        'num_opt_jobs': ``int``,
                                        'opt_jobs': {1: ``Job``,
                                                     2: ``Job``,...},
                                        'num_sp_jobs': ``int``,
                                        'sp_jobs': {1: ``Job``,
                                                    2: ``Job``,...},
                                        'rotors': {1: {'num_scan_jobs': ``int``,  # rotor number corresponds to rotors_dict
                                                       'scan_jobs': {1: ``Job``,
                                                                     2: ``Job``,...},
                                                      }, ...
                                                   2: {}, ...
                                                  },
                                        },
                    'species_label_2': {}, ...
                    }

    """
    def __init__(self, project, species_list, rxn_list, level_of_theory):
        self.project = project
        for species in species_list:
            if not isinstance(species, Species):
                raise ValueError('`species_list` must be a list of RMG.Species objects. Got {0}'.format(type(species)))
        self.species_list = species_list
        for rxn in rxn_list:
            if not isinstance(rxn, Reaction):
                raise ValueError('`rxn_list` must be a list of RMG.Reaction objects. Got {0}'.format(type(rxn)))
        self.rxn_list = rxn_list
        self.level_of_theory = level_of_theory.lower()
        self.job_dict = dict()
        self.species_dict = dict()
        self.unique_species_labels = list()
        for species in species_list:
            if len(species.label) == 0 or species.label in self.unique_species_labels:
                raise SpeciesError('Each species in `species_list` has to have a unique label.')
            self.unique_species_labels.append(species.label)
            self.species_dict[species.label] = dict()
            self.species_dict[species.label]['species_object'] = species
            self.species_dict[species.label]['species_status'] = 'init'
            self.species_dict[species.label]['number_of_rotors'] = 0
            self.species_dict[species.label]['rotors_dict'] = 0
            self.species_dict[species.label]['conformers'] = list()  # xyzs of all conformers
            self.species_dict[species.label]['initial_xyz'] = ''  # xyz of selected conformer
            self.species_dict[species.label]['final_xyz'] = ''  # xyz of final geometry optimization
            self.species_dict[species.label]['num_opt_jobs'] = 0
            self.species_dict[species.label]['opt_jobs'] = dict()
            self.species_dict[species.label]['num_sp_jobs'] = 0
            self.species_dict[species.label]['sp_jobs'] = dict()
            self.species_dict[species.label]['rotors'] = dict()  # rotor scan jobs per rotor
        self.ts_dict = dict()
        self.unique_ts_labels = list()
        # check that TS labels aren't in unique_species_labels AND unique_ts_labels
        self.generate_localized_structures()
        self.generate_species_conformers()
        self.determine_species_rotors()
        self.schedule_jobs()

    def schedule_jobs(self):
        """
        The main job scheduling block
        """
        self.run_conformer_jobs()



        # write_completed_job_to_csv_file

    def generate_localized_structures(self):
        """
        Generate localized (resonance) structures of each species.
        """
        for label in self.unique_species_labels:
            self.species_dict[label]['species_object'].generate_resonance_structures(keep_isomorphic=False,
                                                                                     filter_structures=True)

    def generate_species_conformers(self):
        """
        Generate conformers using RDKit and OpenBabel for all representative localized structures of each species
        """
        for label in self.unique_species_labels:
            for mol in self.species_dict[label]['species_object'].molecule:
                confs = ConformerSearch(mol)
                for xyz in confs.xyzs:
                    self.species_dict[label]['conformers'].append(xyz)

    def run_conformer_jobs(self):
        """
        Select the most stable conformer for each species by spawning opt jobs at B3LYP/6-311++(d,p).
        The resulting conformer is saved in <xyz matrix with element labels> format
        in self.species_dict[species.label]['initial_xyz']
        """
        for label in self.unique_species_labels:
            self.job_dict[label] = dict()
            self.job_dict[label]['conformers'] = dict()
            for i, xyz in enumerate(self.species_dict[label]['conformers']):
                job = Job(project=self.project, species_name=label, xyz=xyz, job_type='opt',
                          level_of_theory='B3LYP/6-311++(d,p)',
                          multiplicity=self.species_dict[label]['species_object'].molecule[0].multiplicity,
                          charge=self.species_dict[label]['species_object'].molecule[0].getNetCharge(), conformer=i,
                          fine=False, software=None, is_ts=False)
                self.job_dict[label]['conformers'][i] = job
                self.job_dict[label]['conformers'][i].run()

    def determine_species_rotors(self):
        """
        Determine possible unique rotors in the species to be treated as hindered rotors,
        taking into account all localized structures.
        The resulting rotors are saved in {'pivots': [1, 3], 'top': [3, 7], 'scan': [2, 1, 3, 7]} format
        in self.species_dict[species.label]['rotors_dict']. Also updates 'number_of_rotors'.
        """
    # def determine_rotors(self):
    #     for mol in self.species.molecule:
    #         r = Rotors(mol)
    #         for new_rotor in r.rotors:
    #             for existing_rotor in self.rotors:
    #                 if existing_rotor['pivots'] == new_rotor['pivots']:
    #                     break
    #             else:
    #                 self.rotors.append(new_rotor)
    #
    # self.scan_method, self.scan_basis = 'B3LYP', '6-311++G(3df,3pd)'



    def check_all_done(self):
        """
        Check that all species and TSs have status 'done'
        """
        pass

