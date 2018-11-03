#!/usr/bin/env python
# encoding: utf-8

import logging
import time
import os

from rmgpy.cantherm.statmech import Log, assign_frequency_scale_factor
from rmgpy.molecule.element import getElement

from arc.molecule.conformer import ConformerSearch, get_xyz_matrix
from arc.molecule.rotors import Rotors
from arc.job.job import Job
from arc.exceptions import SpeciesError, SchedulerError
from arc.settings import output_filename
from arc.ssh import check_servers_jobs_ids

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
    'servers'               ''list''           A list of servers used for the present project
    `species_list`          ``list``           Contains ``RMG.Species`` objects.
    'unique_species_labels' ``list``           A list of species labels (checked for duplicates)
    'unique_ts_labels'      ``list``           A list of TS labels (checked for duplicates)
    `rxn_list`              ``list``           Contains `RMG.Reaction`` objects.
    `level_of_theory`       ``str``            *FULL* level of theory, e.g. 'CBS-QB3',
                                                 'CCSD(T)-F12a/aug-cc-pVTZ//B3LYP/6-311++G(3df,3pd)'...
    'frequency_factor'      ``float``          The frequency scaling factor for the ZPE correction
    `species_dict`          ``dict``           A dictionary of species (dictionary structure specified below)
    `ts_dict`               ``dict``           A dictionary of transition states (dictionary structure specified below)
    `job_dict`              ``dict``           A dictionary of all scheduled jobs. Keys are species / TS labels,
                                                 values are dictionaries where keys are job names (corresponding to
                                                 'running_jobs' if job is running) and values are the Job objects.
    'running_jobs'          ``dict``           A dictionary of currently running jobs (a subset of `job_dict`).
                                                 Keys are species/TS label, values are lists of job names
                                                 (e.g. 'conformer3', 'opt_a123').
    'servers_jobs_ids'       ``list``           A list of relevant job IDs currently running on the server
    ====================== =================== =========================================================================

    Dictionary structures:

*   species_dict = {'species_label_1': {'species_object': ``RMG.Species``,
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
                                        'rotors': {1: {'num_scan_jobs': ``int``,  # rotor number corresponds to rotors_dict
                                                       'scan_jobs': {1: ``Job``,
                                                                     2: ``Job``,...},
                                                      }, ...
                                                   2: {}, ...
                                                  },
                                        },
                    'species_label_2': {}, ...
                    }

*   ts_dict = {'ts_label_1': {need to have xyz guess, xyz_final, multiplicity, charge ...

*   job_dict = {'label_1': {'conformers':       {0: Job1,
                                                 1: Job2, ...},
                            'opt':             {job_name1: Job1,
                                                job_name2: Job2, ...},
                            'sp':              {job_name1: Job1,
                                                job_name2: Job2, ...},
                            'freq':            {job_name1: Job1,
                                                job_name2: Job2, ...},
                            'composite':       {job_name1: Job1,
                                                job_name2: Job2, ...},
                            'scan': {pivots_1: {job_name1: Job1,
                                                job_name2: Job2, ...},
                                     pivots_2: {job_name1: Job1,
                                                job_name2: Job2, ...},
                            }
                }
    """
    def __init__(self, project, species_list, rxn_list, level_of_theory, frequency_factor=None):
        self.project = project
        self.servers = list()
        self.species_list = species_list
        self.rxn_list = rxn_list
        self.level_of_theory = level_of_theory.lower()
        if frequency_factor is not None:
            self.frequency_factor = frequency_factor
        else:
            self.frequency_factor = assign_frequency_scale_factor(self.level_of_theory)
        self.job_dict = dict()
        self.running_jobs = dict()
        self.servers_jobs_ids = list()
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
            self.get_servers_jobs_ids()  # updates `self.servers_jobs_ids`
            for label in self.unique_species_labels:  # (TODO: add TS) loop through species, look for completed jobs and jobs to run
                for key, val in self.running_jobs.iteritems():
                    # key is a species/TS label; val is a list of all job names (e.g., 'optfreq2') of the species/TS
                    if key == label:  # found a species/TS with active jobs
                        for job_name in val:
                            if 'conformer' in job_name:
                                i = int(job_name[9:])  # the conformer number. parsed from a string like 'conformer12'.
                                if not self.job_dict[label]['conformers'][i].job_id in self.servers_jobs_ids:
                                    # this is a completed conformer job
                                    self.download_job_output(job=self.job_dict[label]['conformers'][i])
                                    self.job_dict[label]['conformers'][i].write_completed_job_to_csv_file()
                                    self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
                                    self.parse_conformer_energy(species_label=label, i=i)
                                    # Just terminated a conformer job.
                                    # Are there additional conformer jobs currently running for this species?
                                    species_running_jobs = self.running_jobs[label]
                                    for spec_run_job in species_running_jobs:
                                        if 'conformer' in spec_run_job:
                                            break
                                    else:
                                        # All conformer jobs terminated. Run opt on most stable conformer geometry.
                                        logging.info('\n\nAl conformer jobs successfully terminated')
                                        self.determine_most_stable_conformer(label)
                                        self.run_opt_job(label)  # or run composite method, depends on self.level_of_theory
                            elif 'opt' in job_name\
                                    and not self.job_dict[label][job_name].job_id in self.servers_jobs_ids:
                                # val is 'opt1', 'opt2', etc., or 'optfreq1', optfreq2', etc.
                                self.download_job_output(job=self.job_dict[label][job_name])
                                self.job_dict[label][job_name].write_completed_job_to_csv_file()
                                self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
                                success = self.parse_opt_geo(label=label, job=self.job_dict[label][job_name])
                                if success:
                                    if not 'freq' in job_name:
                                        self.run_freq_job(label)
                                    self.run_sp_job(label)
                                    self.run_scan_jobs(label)
                                else:
                                    # opt job failed, troubleshoot
                                    self.run_opt_job(label)
                            elif 'freq' in job_name\
                                    and not self.job_dict[label][job_name].job_id in self.servers_jobs_ids:
                                    # this is not an 'optfreq' job
                                self.download_job_output(job=self.job_dict[label][job_name])
                                self.job_dict[label][job_name].write_completed_job_to_csv_file()
                                self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
                                self.check_freq_job(label=label)
                            elif 'composite' in job_name\
                                    and not self.job_dict[label][job_name].job_id in self.servers_jobs_ids:
                                self.download_job_output(job=self.job_dict[label][job_name])
                                self.job_dict[label][job_name].write_completed_job_to_csv_file()
                                self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
                                success = self.check_composite_job(label=label)
                                if success:
                                    self.run_freq_job(label)
                                    self.run_scan_jobs(label)
                            elif 'sp' in job_name\
                                    and not self.job_dict[label][job_name].job_id in self.servers_jobs_ids:
                                self.download_job_output(job=self.job_dict[label][job_name])
                                self.job_dict[label][job_name].write_completed_job_to_csv_file()
                                self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
                                self.check_sp_job(label=label)
                            elif 'scan' in job_name\
                                    and not self.job_dict[label][job_name].job_id in self.servers_jobs_ids:
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

    def determine_species_rotors(self):
        """
        Determine possible unique rotors in the species to be treated as hindered rotors,
        taking into account all localized structures.
        The resulting rotors are saved in {'pivots': [1, 3], 'top': [3, 7], 'scan': [2, 1, 3, 7]} format
        in self.species_dict[species.label]['rotors_dict']. Also updates 'number_of_rotors'.
        """
        for label in self.unique_species_labels:
            for mol in self.species_dict[label]['species_object'].molecule:
                r = Rotors(mol)
                for new_rotor in r.rotors:
                    for key, existing_rotor in self.species_dict[label]['rotors_dict'].iteritems():
                        if existing_rotor['pivots'] == new_rotor['pivots']:
                            break
                    else:
                        self.species_dict[label]['number_of_rotors'] += 1
                        self.species_dict[label]['rotors_dict'][self.species_dict[label]['number_of_rotors']] =\
                            new_rotor

    def run_job(self, label, xyz, level_of_theory, job_type, fine=False, software=None, is_ts=False,
                     shift='', trsh='', memory=1000, conformer=-1):
        """
        A helper function for running jobs
        """
        if label in self.unique_species_labels:
            multiplicity = self.species_dict[label]['species_object'].molecule[0].multiplicity
            charge = self.species_dict[label]['species_object'].molecule[0].getNetCharge()
        elif label in self.unique_ts_labels:
            multiplicity = self.ts_dict[label]['multiplicity']
            charge = self.ts_dict[label]['charge']
        job = Job(project=self.project, species_name=label, xyz=xyz, job_type=job_type, level_of_theory=level_of_theory,
                  multiplicity=multiplicity, charge=charge, fine=fine, shift=shift, software=software, is_ts=is_ts,
                  memory=memory, trsh=trsh, conformer=conformer)
        if conformer < 0:
            name = job.job_name
            self.running_jobs[label].append(job_type + '_' + job.job_name)  # mark as a running job
            self.job_dict[label][job_type][name] = job
            self.job_dict[label][job_type][name].run()
        else:
            self.running_jobs[label].append('conformer{0}'.format(conformer))  # mark as a running job
            self.job_dict[label]['conformers'][conformer] = job  # save job object
            self.job_dict[label]['conformers'][conformer].run()  # run the job
        if job.server not in self.servers:
            self.servers.append(job.server)

    def run_conformer_jobs(self):
        """
        Select the most stable conformer for each species by spawning opt jobs at B3LYP/6-311++(d,p).
        The resulting conformer is saved in <xyz matrix with element labels> format
        in self.species_dict[species.label]['initial_xyz']
        """
        for label in self.unique_species_labels:
            self.running_jobs[label] = list()  # initialize running_jobs for all species TODO: also TSs
            self.job_dict[label]['conformers'] = dict()
            for i, xyz in enumerate(self.species_dict[label]['conformers']):
                self.run_job(label=label, xyz=xyz, level_of_theory='B3LYP/6-311++(d,p)', job_type='conformer',
                             conformer=i)

    def run_opt_job(self, label):
        """
        Spawn a geometry optimization job. The initial guess is taken from the `initial_xyz` attribute.
        If an optimization job previously ran for this species and failed, troubleshoot it and spawn a new one.
        The opt job uses the `initial_xyz` as the initial guess if this is a species.

        The job server status is in job.status[0] and can be either 'initializing' / 'running' / 'errored' / 'done'
        The job ess (electronic structure software calculation) status is in  job.status[0] and can be
        either `initializing` / `running` / `errored: {error type / message}` / `unconverged` / `done`
        """
        if 'opt' not in self.job_dict[label]:  # Check whether or not opt jobs has been spawned yet
            # we're spawning the first opt
            self.job_dict[label]['opt'] = dict()
            level_of_theory = self.level_of_theory.split('//')[1]
            if label in self.unique_species_labels:  # this is a species, not a TS
                initial_xyz = self.species_dict[label]['initial_xyz']
                self.run_job(label=label, xyz=initial_xyz, level_of_theory=level_of_theory, job_type='opt', fine=False)
        else:
            # we're troubleshooting for opt jobs.
            # First check for server status and troubleshoot if needed. Then check for ESS status and troubleshoot
            # if needed. Finally, check whether or not the last job had fine=True, add if it didn't run with fine.
            if '//' in self.level_of_theory:
                level_of_theory = self.level_of_theory.split('//')[1]
            else:
                level_of_theory = self.level_of_theory
            previous_job_num = -1
            latest_job_num = -1
            job = None
            for job_name in self.job_dict[label]['opt'].iterkeys():  # get latest Job object
                job_name_int = int(job_name[5:])
                if job_name_int > latest_job_num:
                    previous_job_num = latest_job_num
                    latest_job_num = job_name_int
                    job = self.job_dict[label]['opt'][job_name]
            if job.job_status[0] == 'done':
                if job.job_status[1] == 'done':
                    if job.fine:
                        # run_opt_job should not be called if all looks good...
                        logging.error('opt job for {label} seems right, yet "run_opt_job"'
                                      ' was called.'.format(label=label))
                        raise SchedulerError('opt job for {label} seems right, yet "run_opt_job"'
                                             ' was called.'.format(label=label))
                    else:
                        # Run opt again using a finer grid.
                        self.parse_opt_geo(label=label, job=job, parse_anyway=True)
                        if label in self.unique_species_labels:
                            xyz = self.species_dict[label]['final_xyz']
                            self.species_dict[label]['initial_xyz'] = xyz  # save for troubleshooting
                        elif label in self.unique_ts_labels:
                            xyz = self.ts_dict[label]['final_xyz']
                            self.ts_dict[label]['initial_xyz'] = xyz  # save for troubleshooting
                        self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory,
                                          job_type='opt', fine=True)
                else:
                    # job passes on the server, but failed in ESS calculation
                    if previous_job_num >= 0 and job.fine:
                        previous_job = self.job_dict[label]['opt']['opt_a'+str(previous_job_num)]
                        if not previous_job.fine and previous_job.job_status[0] == 'done'\
                                and previous_job.job_status[1] == 'done':
                            # The present job with a fine grid failed in the ESS calculation.
                            # A *previous* job without a fine grid terminated successfully on the server and ESS.
                            # So use the xyz determined w/o the fine grid, and output an error message to alert users.
                            logging.error('Optimization job for {label} with a fine grid terminated successfully'
                                          ' on the server, but crashed during calculation. NOT running with fine'
                                          ' grid again.')
                            self.parse_opt_geo(label=label, job=previous_job, parse_anyway=True)
                    else:
                        self.troubleshoot_ess(label=label, job=job, level_of_theory=level_of_theory, job_type='opt')
            else:
                job.troubleshoot_server()

    def parse_conformer_energy(self, species_label, i):
        """
        Parse E0 (Hartree) from the conformer opt output file, saves it in the 'conformer_energies' attribute.
        Troubleshoot if job crashed by running CBS-QB3 which is often more robust (but more time-consuming).
        """
        job = self.job_dict[species_label]['conformers'][i]
        local_path_to_output_file = os.path.join(job.local_path, output_filename[job.software])
        log = Log(path='')
        log.determine_qm_software(fullpath=local_path_to_output_file)
        e0 = log.software_log.loadEnergy(frequencyScaleFactor=self.frequency_factor,
                                         modelChemistry=self.level_of_theory)
        self.species_dict[species_label]['conformer_energies'][i] = e0

    def determine_most_stable_conformer(self, species_label):
        """
        Determine the most stable conformer of species. Save the resulting xyz as `initial_xyz`
        """
        e_min = self.species_dict[species_label]['conformer_energies'][0]
        i_min = 0
        for i, ei in enumerate(self.species_dict[species_label]['conformer_energies']):
            if ei < e_min:
                e_min = ei
                i_min = i
        self.species_dict[species_label]['initial_xyz'] = self.species_dict[species_label]['conformers'][i_min]

    def parse_opt_geo(self, label, job, parse_anyway=False):
        """
        Check that an 'opt' or 'optfreq' job converged successfully, and parse the geometry into `final_xyz`.
        If the job is 'optfreq', also checks (QA) that no imaginary frequencies were assigned for stable species,
        and that exactly one imaginary frequency was assigned for a TS.
        Returns ``True`` if the job (or both jobs) converged successfully, ``False`` otherwise and troubleshoots opt.
        """
        if (job.fine and job.job_status[0] == 'done' and job.job_status[1] == 'done') or parse_anyway:
            local_path_to_output_file = os.path.join(job.local_path, output_filename[job.software])
            log = Log(path='')
            log.determine_qm_software(fullpath=local_path_to_output_file)
            coord, number, mass = log.software_log.loadGeometry()
            xyz = get_xyz_matrix(xyz=coord, from_arkane=True, number=number)
            if label in self.unique_species_labels:
                self.species_dict[label]['final_xyz'] = xyz
            elif label in self.unique_ts_labels:
                self.ts_dict[label]['final_xyz'] = xyz
            return True
        return False

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
    def get_servers_jobs_ids(self):
        """
        Check status on all active servers, return a list of relevant running job IDs
        """
        self.servers_jobs_ids = check_servers_jobs_ids(self.servers)

    def download_job_output(self, job):
        """
        Download relevant output file from the server to the local `job` directory
        """

    def troubleshoot_ess(self, label, job, level_of_theory, job_type):
        """
        """
        logging.info('\n\n')
        logging.warn('Troubleshooting {job_type} job for {label} which failed with status'
                     ' {stat} in {soft}.'.format(job_type=job_type, label=label, stat=job.status[1], soft=job.software))
        if label in self.unique_species_labels:
            xyz = self.species_dict[label]['initial_xyz']
        elif label in self.unique_ts_labels:
            xyz = self.ts_dict[label]['initial_xyz']
        if job.software == 'gaussian03':
            if not 'cbs-qb3' in job.ess_trsh_methods and self.level_of_theory != 'cbs-qb3':
                # try running CBS-QB3, which is relatively robust
                job.ess_trsh_methods.append('cbs-qb3')
                level_of_theory = 'cbs-qb3'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory,
                              job_type=job_type, fine=False)
            elif 'scf=qc' not in job.ess_trsh_methods:
                # calls a quadratically convergent SCF procedure
                job.ess_trsh_methods.append('scf=qc')
                trsh = 'scf=qc'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory,
                              job_type=job_type, fine=False, trsh=trsh)
            elif 'scf=nosymm' not in job.ess_trsh_methods:
                # calls a quadratically convergent SCF procedure
                job.ess_trsh_methods.append('scf=nosymm')
                trsh = 'scf=nosymm'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory,
                              job_type=job_type, fine=False, trsh=trsh)
            elif 'memory' not in job.ess_trsh_methods:
                # Increase memory allocation
                job.ess_trsh_methods.append('memory')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory,
                              job_type=job_type, fine=False, memory=2500)
            elif 'scf=(qc,nosymm)' not in job.ess_trsh_methods:
                # try both qc and nosymm
                job.ess_trsh_methods.append('scf=(qc,nosymm)')
                trsh = 'scf=(qc,nosymm)'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory,
                              job_type=job_type, fine=False, trsh=trsh)
            elif self.level_of_theory != 'cbs-qb3':
                # try both qc and nosymm with CBS-QB3
                job.ess_trsh_methods.append('scf=(qc,nosymm) & CBS-QB3')
                level_of_theory = 'cbs-qb3'
                trsh = 'scf=(qc,nosymm)'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory,
                              job_type=job_type, fine=False, trsh=trsh)
            elif 'qchem' not in job.ess_trsh_methods:
                # Try QChem
                job.ess_trsh_methods.append('qchem')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory,
                              job_type=job_type, fine=False, software='qchem')
            elif 'molpro2012' not in job.ess_trsh_methods:
                # Try molpro
                job.ess_trsh_methods.append('molpro2012')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory,
                              job_type=job_type, fine=False, software='molpro2012')
            else:
                logging.error('Could not troubleshoot geometry optimization for {label}! Tried'
                              ' troubleshooting with the following methods: {methods}'.format(
                    label=label, methods=job.ess_trsh_methods))
                raise SchedulerError('Could not troubleshoot geometry optimization for {label}! Tried'
                                     ' troubleshooting with the following methods: {methods}'.format(
                    label=label, methods=job.ess_trsh_methods))
        elif job.software == 'qchem':
            if 'max opt cycles reached' in job.job_status[1] and 'max_cycles' not in job.ess_trsh_methods:
                # this is a common error, increase max cycles and continue running from last geometry
                job.ess_trsh_methods.append('max_cycles')  # avoids infinite looping
                trsh = '\nGEOM_OPT_MAX_CYCLES 250'  # default is 50
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory,
                              job_type=job_type, fine=False, trsh=trsh)
            elif 'b3lyp' not in job.ess_trsh_methods:
                job.ess_trsh_methods.append('b3lyp')
                # try converging with B3LYP
                level_of_theory = 'b3lyp/6-311+g**'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory,
                              job_type=job_type, fine=False)
            elif 'gaussian03' not in job.ess_trsh_methods:
                # Try Gaussian
                job.ess_trsh_methods.append('gaussian03')
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory,
                              job_type=job_type, fine=False, software='gaussian03')
            elif 'molpro' not in job.ess_trsh_methods:
                # Try molpro
                job.ess_trsh_methods.append('molpro')
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory,
                              job_type=job_type, fine=False, software='molpro2012')
            else:
                logging.error('Could not troubleshoot geometry optimization for {label}! Tried'
                              ' troubleshooting with the following methods: {methods}'.format(
                    label=label, methods=job.ess_trsh_methods))
                raise SchedulerError('Could not troubleshoot geometry optimization for {label}! Tried'
                                     ' troubleshooting with the following methods: {methods}'.format(
                    label=label, methods=job.ess_trsh_methods))
        elif 'molpro' in job.software:
            if 'shift' not in job.ess_trsh_methods:
                # try adding a level shift for alpha- and beta-spin orbitals
                job.ess_trsh_methods.append('shift')
                shift = 'shift,-1.0,-0.5;'
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory,
                              job_type=job_type, fine=False, shift=shift)
            elif 'memory' not in job.ess_trsh_methods:
                # Increase memory allocation, also run with a shift
                job.ess_trsh_methods.append('memory')
                shift = 'shift,-1.0,-0.5;'
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory,
                              job_type=job_type, fine=False, shift=shift, memory=5000)
            elif 'gaussian03' not in job.ess_trsh_methods:
                # Try Gaussian
                job.ess_trsh_methods.append('gaussian03')
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory,
                              job_type=job_type, fine=False, software='gaussian03')
            elif 'qchem' not in job.ess_trsh_methods:
                # Try QChem
                job.ess_trsh_methods.append('qchem')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory,
                              job_type=job_type, fine=False, software='qchem')
            else:
                logging.error('Could not troubleshoot geometry optimization for {label}! Tried'
                              ' troubleshooting with the following methods: {methods}'.format(
                    label=label, methods=job.ess_trsh_methods))
                raise SchedulerError('Could not troubleshoot geometry optimization for {label}! Tried'
                                     ' troubleshooting with the following methods: {methods}'.format(
                    label=label, methods=job.ess_trsh_methods))

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
