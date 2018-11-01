#!/usr/bin/env python
# encoding: utf-8

import logging
import time
import os

from rmgpy.cantherm.statmech import Log, assign_frequency_scale_factor

from arc.molecule.conformer import ConformerSearch
from arc.molecule.rotors import Rotors
from arc.job.job import Job
from arc.exceptions import SpeciesError
from arc.settings import output_filename

##################################################################


class Scheduler(object):
    """
    ARC Scheduler class. Creates jobs, submits, checks status, troubleshoots.
    Each species in `species_list` has to have a unique label.

    The attributes are:

    ====================== =================== =========================================================================
    Attribute              Type                Description
    ====================== =================== =========================================================================
    `project`               ``str``            The project's name. Used for naming the directory.
    `species_list`          ``list``           Contains ``RMG.Species`` objects.
    'unique_species_labels' ``list``           A list of species labels (checked for duplicates)
    'unique_ts_labels'      ``list``           A list of TS labels (checked for duplicates)
    `rxn_list`              ``list``           Contains `RMG.Reaction`` objects.
    `level_of_theory`       ``str``            *FULL* level of theory, e.g. 'CBS-QB3',
                                                 'CCSD(T)-F12a/aug-cc-pVTZ//B3LYP/6-311++G(3df,3pd)'...
    'frequency_factor'      ``float``          The frequency scaling factor for the ZPE correction
    `species_dict`          ``dict``           A dictionary of species (dictionary structure specified below)
    `ts_dict`               ``dict``           A dictionary of transition states (dictionary structure specified below)
    `job_dict`              ``dict``           A dictionary of all scheduled jobs. Keys are species / TS labels.
    'running_jobs'          ``dict``           A dictionary of currently running jobs. Keys are labels,
                                                 value is a list of job types.
    'running_job_ids'       ``list``           A list of relevant job IDs currently running on the server
    ====================== =================== =========================================================================

    The structure of species_dict is the following:
    species_dict = {'species_label_1': {'species_object': ``RMG.Species``,
                                        'species_status': 'init', 'opt' / 'sp, rotors' / 'done' / 'errored: <reason>',
                                        'number_of_rotors': ``int``,
                                        'rotors_dict: {1: {'pivots': pivot_list,
                                                           'top': top_list,
                                                           'scan': scan_list},
                                                       2: {}, ...
                                                      }
                                        'successful_rotors': <structure like 'rotors_dict'>,
                                        'unsuccessful_rotors': <structure like 'rotors_dict'>,
                                        'conformers': ``list`` of <xyz matrix with element labels>,
                                        'conformer_energies': ``list`` of respective E0 values,
                                        'initial_xyz': <xyz matrix with element labels> of most stable conformer,
                                        'final_xyz': <xyz matrix with element labels> after geometry optimization,
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

    The structure of ts_dict is the following:
    -- similar, add irc, gsm, conformers are optional TS structures found using different methods

    species_dict = {'species_label_1': {'species_object': ``RMG.Species``,
                                        'species_status': 'init'/'conf'/'optfreq'/'opt'/'freq'/'sp, rotors'/'troubleshoot: <details>'/'errored: <reason>'/'done',
                                        'selected_conformer': ``int``,
                                        'number_of_rotors': ``int``,
                                        'rotors_dict: {1: {'pivots': pivot_list,
                                                           'top': top_list,
                                                           'scan': scan_list},
                                                       2: {}, ...
                                                      }
                                        'conformer_dict': {1: {'initial_xyz': <xyz matrix with element labels>,
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
                                                                        },
                                                               'conf_status': 'init', 'opt' / 'sp, rotors' / 'done' / 'errored: <reason>',
                                                               },
                                                           2: {}, ...
                                                          },
                                        },
                    'species_label_2': {}, ...
                    }

    """
    def __init__(self, project, species_list, rxn_list, level_of_theory, frequency_factor=None):
        self.project = project
        self.species_list = species_list
        self.rxn_list = rxn_list
        self.level_of_theory = level_of_theory.lower()
        if frequency_factor is not None:
            self.frequency_factor = frequency_factor
        else:
            self.frequency_factor = assign_frequency_scale_factor(self.level_of_theory)
        self.job_dict = dict()
        self.running_jobs = dict()
        self.running_job_ids = list()
        self.species_dict = dict()
        self.unique_species_labels = list()
        self.unique_ts_labels = list()  # not used yet
        for species in self.species_list:
            if len(species.label) == 0 or species.label in self.unique_species_labels:
                raise SpeciesError('Each species in `species_list` has to have a unique label.')
            self.unique_species_labels.append(species.label)
            self.job_dict[species.label] = dict()  # values are dicts with job names as keys (e.g., 'opt1') and Job objects as values
            self.species_dict[species.label] = dict()
            self.species_dict[species.label]['species_object'] = species
            self.species_dict[species.label]['species_status'] = 'init'
            self.species_dict[species.label]['number_of_rotors'] = 0
            self.species_dict[species.label]['rotors_dict'] = dict()
            self.species_dict[species.label]['successful_rotors'] = dict()
            self.species_dict[species.label]['unsuccessful_rotors'] = dict()
            self.species_dict[species.label]['conformers'] = list()  # xyzs of all conformers
            self.species_dict[species.label]['conformer_energies'] = list()  # conformer E0 (Hartree), respectively
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
        logging.info('generating conformer jobs for all species')
        self.run_conformer_jobs()  # also writes jobs to self.running_jobs
        # TODO: get xyz guesses for all TSs
        while self.running_jobs != {}:  # loop while jobs are still running
            self.get_running_jobs_ids()
            for label in self.unique_species_labels:  # (TODO: add TS) loop through species, look for completed jobs and jobs to run
                for key, val in self.running_jobs.iteritems():
                    # key is a species/TS label; val is a list of all job names (e.g., 'optfreq2') of the species/TS
                    if key == label:  # found a species/TS with active jobs
                        for job_name in val:
                            i = int(job_name[9:])  # the conformer number (integer). parsed from a string like 'conformer12'.
                            if 'conformer' in job_name\
                                    and not self.job_dict[label]['conformers'][i].job_id in self.running_job_ids:
                                # this is a completed conformer job
                                self.download_job_output(job=self.job_dict[label]['conformers'][i])
                                self.job_dict[label]['conformers'][i].write_completed_job_to_csv_file()
                                self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
                                self.parse_conformer_energy(species_label=label, i=i)
                                # Just terminated a conformer job.
                                # Are there additional conformer optimization jobs currently running for this species?
                                species_running_jobs = self.running_jobs[label]
                                for spec_run_job in species_running_jobs:
                                    if 'conformer' in spec_run_job:
                                        break
                                else:
                                    # All conformer optimizations terminated. Run opt on most stable conformer geometry.
                                    logging.info('\n\nAl conformer jobs successfully terminated')
                                    self.determine_most_stable_conformer(label)
                                    self.run_opt_job(label)  # or run composite method, depends on self.level_of_theory
                            elif 'opt' in job_name and not self.job_dict[label][job_name].job_id in self.running_job_ids:
                                # val is 'opt1', 'opt2', etc. or 'optfreq1', optfreq2', etc.
                                self.download_job_output(job=self.job_dict[label][job_name])
                                self.job_dict[label][job_name].write_completed_job_to_csv_file()
                                self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
                                success = self.parse_opt_geo(label=label)
                                if success:
                                    if not 'freq' in job_name:
                                        self.run_freq_job(label)
                                    self.run_sp_job(label)
                                    self.run_scan_jobs(label)
                            elif 'freq' in job_name and not self.job_dict[label][job_name].job_id in self.running_job_ids:  # this is not an 'optfreq' job
                                self.download_job_output(job=self.job_dict[label][job_name])
                                self.job_dict[label][job_name].write_completed_job_to_csv_file()
                                self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
                                self.check_freq_job(label=label)
                            elif 'composite' in job_name and not self.job_dict[label][job_name].job_id in self.running_job_ids:
                                self.download_job_output(job=self.job_dict[label][job_name])
                                self.job_dict[label][job_name].write_completed_job_to_csv_file()
                                self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
                                success = self.check_composite_job(label=label)
                                if success:
                                    self.run_freq_job(label)
                                    self.run_scan_jobs(label)
                            elif 'sp' in job_name and not self.job_dict[label][job_name].job_id in self.running_job_ids:
                                self.download_job_output(job=self.job_dict[label][job_name])
                                self.job_dict[label][job_name].write_completed_job_to_csv_file()
                                self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
                                self.check_sp_job(label=label)
                            elif 'scan' in job_name and not self.job_dict[label][job_name].job_id in self.running_job_ids:
                                self.download_job_output(job=self.job_dict[label][job_name])
                                self.job_dict[label][job_name].write_completed_job_to_csv_file()
                                self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
                                self.check_scan_job(label=label)
                            # TODO: GSM, IRC
            time.sleep(60)  # wait a minute before bugging the servers again.
        # When exiting the while loop, make sure we got all we asked for. Otherwise spawn and rerun schedule_jobs()

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
                    self.species_dict[label]['conformer_energies'].append(0.0)  # a placeholder (lists are synced)

    def run_conformer_jobs(self):
        """
        Select the most stable conformer for each species by spawning opt jobs at B3LYP/6-311++(d,p).
        The resulting conformer is saved in <xyz matrix with element labels> format
        in self.species_dict[species.label]['initial_xyz']
        """
        for label in self.unique_species_labels:
            self.running_jobs[label] = list()
            self.job_dict[label]['conformers'] = dict()
            for i, xyz in enumerate(self.species_dict[label]['conformers']):
                job = Job(project=self.project, species_name=label, xyz=xyz, job_type='opt',
                          level_of_theory='B3LYP/6-311++(d,p)',
                          multiplicity=self.species_dict[label]['species_object'].molecule[0].multiplicity,
                          charge=self.species_dict[label]['species_object'].molecule[0].getNetCharge(),
                          conformer=i, fine=False, software=None, is_ts=False)
                self.job_dict[label]['conformers'][i] = job
                self.running_jobs[label].append('conformer{0}'.format(i))
                self.job_dict[label]['conformers'][i].run()

    def parse_conformer_energy(self, species_label, i):
        """
        Parse E0 (Hartree) from the conformer opt output file, saves it in the 'conformer_energies' attribute.
        Troubleshoot if job crashed by running CBS-QB3 which is often more robust (but more time-consuming).
        """
        job = self.job_dict[species_label]['conformers'][i]
        local_path_to_output_file = os.path.join(job.local_path, output_filename[job.software])
        log = Log(path='')
        log.determine_qm_software(fullpath=local_path_to_output_file)
        E0 = log.software_log.loadEnergy(frequencyScaleFactor=self.frequency_factor,
                                         modelChemistry=self.level_of_theory)
        self.species_dict[species_label]['conformer_energies'][i] = E0

    def parse_opt_geo(self, label):
        """
        Check that an opt or optfreq job converged successfully, and parse the geometry into `final_xyz`.
        If the job is 'optfreq', also checks (QA) that no imaginary frequencies were assigned for stable species,
        and that exactly one imaginary frequency was assigned for a TS.
        Returns ``True`` if the job (or both jobs) converged successfully, ``False`` otherwise and troubleshoots opt.
        """
        success = False
        return success

    def check_freq_job(self, label):
        """
        Check that a freq job converged successfully. Also checks (QA) that no imaginary frequencies were assigned for
        stable species, and that exactly one imaginary frequency was assigned for a TS. Also troubleshoots freq.
        """

    def check_sp_job(self, label):
        """
        Check that a single point job converged successfully. Also troubleshoots.
        """

    def check_scan_job(self, label):
        """
        Check that a rotor scan job converged successfully. Also checks (QA) whether the scan is relatively "smooth",
        and recommends whether or not to use this rotor using the 'successful_rotors' and 'unsuccessful_rotors'
        attributes.
        """

    def check_composite_job(self, label):
        """
        Check that a composite job converged successfully, and parse the geometry into `final_xyz`.
        Also checks that no imaginary frequencies were assigned for stable species, and that exactly one imaginary
        frequency was assigned for a TS.
        Returns ``True`` if the job converged successfully, ``False`` otherwise and troubleshoots.
        """
        success = False
        return success

    def determine_most_stable_conformer(self, species_label):
        """
        Determine the most stable conformer of species. Save the resulting xyz as `initial_xyz`
        """
        # look in self.species_dict[species_label]['conformer_energies']
        # output to self.species_dict[species_label]['initial_xyz']

    def run_opt_job(self, label):
        """
        Spawn a geometry optimization job. If no optimization jobs were run and label represents a species, use the
        xyz of the most stable conformer, and if label represents a TS, use a provided xyz or spawn GSM.
        If an optimization job previously ran for this species and failed, troubleshoot it and spawn a new one.
        The opt job uses the `initial_xyz` as the initial guess if this is a species.
        If self.level_of_theory represents a composite method, spawn the requested job instead.
        """

    def run_freq_job(self, label):
        """
        Spawn a freq job on the last converged optimization geometry.
        If this was originally a composite job (but not a composite job as a troubleshooter method), run an appropriate
        freq job outputting the Hessian.
        """

    def run_sp_job(self, label):
        """
        Spawn a single point job on the last converged optimization geometry.
        """

    def run_scan_jobs(self, label):
        """
        Spawn rotor scan jobs on the last converged optimization geometry.
        """

    def get_running_jobs_ids(self):
        """
        Check status on all active servers, return a list of relevant running job IDs
        """

    def download_job_output(self, job):
        """
        Download relevant output file from the server to the local `job` directory
        """

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


# TODO: TSs, troubleshooting convergence problems (do CBS-QB3, molpro tricks)
# TODO: a log file for each species with sucsessfull runs/rotors/ etc. all statuses

"""

    # An additional freq job has to be spawned since gaussian can only be forced to output the complete
    # Hessian for the first part of the its job.
Maybe define (in this file) an ARCSpecies object with all properties? (inc. TS?)

generate and explore all conformers
get all species
(bother with TS's later)
ask what additional jobs are needed for the species
parse output
write completed jobs to csv file
actually run the jobs

check (using Arkane) if the rotors start at the minimal energy. if not, re opt.
"""



"""
troubleshoot geomerty unconverged in gaussian: add Opt=CalcFC
troubleshoot sp in molpro: use a lower but reasonable basis set, then the higher one

QA: check number of negative frequencies

check spin contamination (in molpro, this is in the output.log file(!) as "Spin contamination <S**2-Sz**2-Sz>     0.00000000")
check T3 or other MR indication
make visuallization files
"""
