#!/usr/bin/env python
# encoding: utf-8

import logging
import time
import os

import cclib

from rmgpy.cantherm.statmech import Log

from arc.molecule.conformer import ConformerSearch, get_xyz_matrix
from arc.molecule.rotors import Rotors
from arc.job.job import Job
from arc.exceptions import SpeciesError, SchedulerError
from arc.job.ssh import SSH_Client

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
    'composite'             ``bool``           Whether level_of_theory represents a composite method or not
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
                                        'number_of_rotors': ``int``,
                                        'rotors_dict: {1: {'pivots': pivot_list,
                                                           'top': top_list,
                                                           'scan': scan_list,
                                                           'success: ''bool''},
                                                       2: {}, ...
                                                      }
                                        'conformers': ``list`` of <xyz matrix with element labels>,
                                        'conformer_energies': ``list`` of respective E0 values,
                                        'initial_xyz': <xyz matrix with element labels> of most stable conformer,
                                        'final_xyz': <xyz matrix with element labels> after geometry optimization,
                                        },
                    'species_label_2': {}, ...
                    }

*   ts_dict = {'ts_label_1': {'multiplicity': ``int``,
                              'charge': ``int``,
                              'number_of_rotors': ``int``,
                              'rotors_dict: {1: {'pivots': pivot_list,
                                                 'top': top_list,
                                                 'scan': scan_list,
                                                 'success: ''bool''},
                                             2: {}, ...
                                            }
                              'initial_xyz': <xyz matrix with element labels> of initial xyz guess,
                              'final_xyz': <xyz matrix with element labels> after TS geometry optimization,
                              },
                'ts_label_2': {}, ...
               }

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
    # TODO: a dict of job names and output files of final geo, freq, sp, scans. the final result returned
    # TODO: also track failed jobs/species like unconverged opt or neg freq
    # TODO: adding could be done in check functions
    def __init__(self, project, species_list, rxn_list, level_of_theory, freq_level=''):
        self.project = project
        self.servers = list()
        self.species_list = species_list
        self.rxn_list = rxn_list
        self.level_of_theory = level_of_theory.lower()
        self.freq_level = freq_level
        self.composite = not '//' in self.level_of_theory
        self.job_dict = dict()
        self.running_jobs = dict()
        self.servers_jobs_ids = list()
        self.species_dict = dict()
        self.unique_species_labels = list()
        self.unique_ts_labels = list()  # not used yet
        # TODO: repeat the below loop for rxns and create TS labels and an empty job_dict[label] = dict() for each
        for species in self.species_list:
            if len(species.label) == 0 or species.label in self.unique_species_labels:
                raise SpeciesError('Each species in `species_list` has to have a unique label.')
            self.unique_species_labels.append(species.label)
            self.job_dict[species.label] = dict()  # values are dicts with job names as keys (e.g., 'opt1') and Job objects as values
            self.species_dict[species.label] = dict()
            self.species_dict[species.label]['species_object'] = species
            self.species_dict[species.label]['number_of_rotors'] = 0
            self.species_dict[species.label]['rotors_dict'] = dict()
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
        self.timer = True
        self.schedule_jobs()

    def schedule_jobs(self):
        """
        The main job scheduling block
        """
        logging.info('generating conformer jobs for all species')
        self.run_conformer_jobs()  # also writes jobs to self.running_jobs
        # TODO: get xyz guesses for all TSs
        while self.running_jobs != {}:  # loop while jobs are still running
            self.timer = True
            for label in self.unique_species_labels + self.unique_ts_labels:  # look for completed jobs and jobs to run
                self.get_servers_jobs_ids()  # updates `self.servers_jobs_ids`
                try:
                    job_list = self.running_jobs[label]
                    for job_name in job_list:
                        if 'conformer' in job_name:
                            i = int(job_name[9:])  # the conformer number. parsed from a string like 'conformer12'.
                            if self.job_dict[label]['conformers'][i].job_id not in self.servers_jobs_ids:
                                # this is a completed conformer job
                                job = self.job_dict[label]['conformers'][i]
                                successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                                if successful_server_termination:
                                    self.parse_conformer_energy(job=job, label=label, i=i)
                                # Just terminated a conformer job.
                                # Are there additional conformer jobs currently running for this species?
                                for spec_jobs in job_list:
                                    if 'conformer' in spec_jobs:
                                        break
                                else:
                                    # All conformer jobs terminated. Run opt on most stable conformer geometry.
                                    logging.info('\nConformer jobs for {0} successfully terminated.'.format(
                                        label))
                                    self.determine_most_stable_conformer(label)
                                    if not self.composite:
                                        self.run_opt_job(label)
                                    else:
                                        self.run_composite_job(label)
                                self.timer = False
                                break
                        elif 'opt' in job_name\
                                and not self.job_dict[label]['opt'][job_name].job_id in self.servers_jobs_ids:
                            # val is 'opt1', 'opt2', etc., or 'optfreq1', optfreq2', etc.
                            job = self.job_dict[label]['opt'][job_name]
                            successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                            if successful_server_termination:
                                success = self.parse_opt_geo(label=label, job=job)
                                if success:
                                    if self.composite:
                                        # This was originally a composite method, probably troubleshooted as 'opt'
                                        self.run_composite_job(label)
                                    else:
                                        if 'freq' not in job_name:
                                            self.run_freq_job(label)
                                        self.run_sp_job(label)
                                        self.run_scan_jobs(label)
                            self.timer = False
                            break
                        elif 'freq' in job_name\
                                and not self.job_dict[label]['freq'][job_name].job_id in self.servers_jobs_ids:
                            # this is not an 'optfreq' job
                            job = self.job_dict[label]['freq'][job_name]
                            successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                            if successful_server_termination:
                                self.check_freq_job(label=label, job=job)
                            self.timer = False
                            break
                        elif 'sp' in job_name\
                                and not self.job_dict[label]['sp'][job_name].job_id in self.servers_jobs_ids:
                            job = self.job_dict[label]['sp'][job_name]
                            successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                            if successful_server_termination:
                                self.check_sp_job(label=label, job=job)
                            self.timer = False
                            break
                        elif 'composite' in job_name\
                                and not self.job_dict[label]['composite'][job_name].job_id in self.servers_jobs_ids:
                            job = self.job_dict[label]['composite'][job_name]
                            successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                            if successful_server_termination:
                                success = self.parse_opt_geo(label=label, job=job)
                                if success:
                                    if not self.composite:
                                        # This wasn't originally a composite method, probably troubleshooted as such
                                        self.run_opt_job(label)
                                    else:
                                        self.run_freq_job(label)
                                        self.run_scan_jobs(label)
                            self.timer = False
                            break
                        elif 'scan' in job_name\
                                and not self.job_dict[label]['scan'][job_name].job_id in self.servers_jobs_ids:
                            job = self.job_dict[label]['scan'][job_name]
                            successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                            if successful_server_termination:
                                self.check_scan_job(label=label, job=job)
                            self.timer = False
                            break
                        # TODO: GSM, IRC
                except KeyError:
                    pass
                if not self.running_jobs[label]:
                    self.check_all_done(label)
                    logging.info('All jobs of species {0} are done'.format(label))
                    del self.running_jobs[label]

            if self.timer:
                logging.debug('zzz... setting timer for 1 minute... zzz')
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
                conformers = ConformerSearch(mol)
                for xyz in conformers.xyzs:
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
                     shift='', trsh='', memory=1000, conformer=-1, ess_trsh_methods=list(), scan='', pivots=list()):
        """
        A helper function for running jobs
        """
        if self.is_label_ts(label):
            multiplicity = self.ts_dict[label]['multiplicity']
            charge = self.ts_dict[label]['charge']
        else:
            multiplicity = self.species_dict[label]['species_object'].molecule[0].multiplicity
            charge = self.species_dict[label]['species_object'].molecule[0].getNetCharge()
        job = Job(project=self.project, species_name=label, xyz=xyz, job_type=job_type, level_of_theory=level_of_theory,
                  multiplicity=multiplicity, charge=charge, fine=fine, shift=shift, software=software, is_ts=is_ts,
                  memory=memory, trsh=trsh, conformer=conformer, ess_trsh_methods=ess_trsh_methods, scan=scan,
                  pivots=pivots)
        if conformer < 0:
            self.running_jobs[label].append(job.job_name)  # mark as a running job
            try:
                self.job_dict[label][job_type]
            except KeyError:
                # Jobs of this type haven't been spawned for label, this is probably a troubleshooting job
                self.job_dict[label][job_type] = dict()
            self.job_dict[label][job_type][job.job_name] = job
            self.job_dict[label][job_type][job.job_name].run()
        else:
            # Running a conformer job. Append differently to job_dict.
            self.running_jobs[label].append('conformer{0}'.format(conformer))  # mark as a running job
            self.job_dict[label]['conformers'][conformer] = job  # save job object
            self.job_dict[label]['conformers'][conformer].run()  # run the job
        if job.server not in self.servers:
            self.servers.append(job.server)

    def end_job(self, job, label, job_name):
        """
        A helper function for checking job status, saving in cvs file, and downloading output files.
        Returns ``True`` if job terminated successfully on the server, ``False`` otherwise
        """
        logging.info('ending job {0}'.format(job.job_name))
        job.determine_job_status()  # also downloads output file
        job.write_completed_job_to_csv_file()
        self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
        self.timer = False
        if job.job_status[0] != 'done':
            job.troubleshoot_server()
            return False
        return True

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
                self.run_job(label=label, xyz=xyz, level_of_theory='b3lyp/6-311++g(d,p)', job_type='conformer',
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
        if 'opt' not in self.job_dict[label]:  # Check whether or not opt jobs have been spawned yet
            # we're spawning the first opt job
            self.job_dict[label]['opt'] = dict()
            if not self.composite:
                level_of_theory = self.level_of_theory.split('//')[1]
            else:
                # this is probably a troubleshooting opt job for a composite method
                level_of_theory = 'b3lyp/6-311++g(d,p)'
            if self.is_label_ts(label):
                initial_xyz = self.ts_dict[label]['initial_xyz']
                self.run_job(label=label, xyz=initial_xyz, level_of_theory=level_of_theory, job_type='opt', is_ts=True,
                             fine=False)
            else:
                initial_xyz = self.species_dict[label]['initial_xyz']
                self.run_job(label=label, xyz=initial_xyz, level_of_theory=level_of_theory, job_type='opt', is_ts=False,
                             fine=False)
        else:
            self.troubleshoot_opt_jobs(label=label)

    def run_composite_job(self, label):
        """
        Spawn a composite job (e.g., CBS-QB3) using 'final_xyz' for species ot TS 'label'.
        """
        if not self.composite:
            logging.error('Cannot run {0} as a composite method'.format(self.level_of_theory))
            raise SchedulerError('Cannot run {0} as a composite method'.format(self.level_of_theory))
        if 'composite' not in self.job_dict[label]:  # Check whether or not composite jobs have been spawned yet
            # we're spawning the first composite job
            self.job_dict[label]['composite'] = dict()
        if self.is_label_ts(label):
            is_ts = True
            if self.species_dict[label]['final_xyz'] != '':
                # If not empty, then opt was probably troubleshooted, use 'final_xyz'
                xyz = self.ts_dict[label]['final_xyz']
            else:
                xyz = self.ts_dict[label]['initial_xyz']
        else:
            is_ts = False
            if self.species_dict[label]['final_xyz'] != '':
                # If not empty, then opt was probably troubleshooted, use 'final_xyz'
                xyz = self.species_dict[label]['final_xyz']
            else:
                xyz = self.species_dict[label]['initial_xyz']
        self.run_job(label=label, xyz=xyz, level_of_theory=self.level_of_theory, job_type='composite', is_ts=is_ts,
                     fine=True)

    def run_freq_job(self, label):
        """
        Spawn a freq job using 'final_xyz' for species ot TS 'label'.
        If this was originally a composite job, run an appropriate separate freq job outputting the Hessian.
        """
        if 'freq' not in self.job_dict[label]:  # Check whether or not freq jobs have been spawned yet
            # we're spawning the first freq job
            self.job_dict[label]['freq'] = dict()
        if self.freq_level:
            level_of_theory = self.freq_level
        elif self.composite:
            level_of_theory = 'b3lyp/cbsb7'  # this is the freq basis set used in CBS-QB3
        else:
            level_of_theory = self.level_of_theory.split('//')[1]  # use the geometry optimization's level of theory
        if self.is_label_ts(label):
            xyz = self.ts_dict[label]['final_xyz']
        else:
            xyz = self.species_dict[label]['final_xyz']
        self.run_job(label, xyz, level_of_theory, job_type='freq')

    def run_sp_job(self, label):
        """
        Spawn a single point job using 'final_xyz' for species ot TS 'label'.
        """
        if 'sp' not in self.job_dict[label]:  # Check whether or not single point jobs have been spawned yet
            # we're spawning the first sp job
            self.job_dict[label]['sp'] = dict()
        if self.composite:
            raise SchedulerError('run_sp_job() was called for {0} which has a composite method level of theory'.format(
                label))
        else:
            level_of_theory = self.level_of_theory.split('//')[0]
        if self.is_label_ts(label):
            xyz = self.ts_dict[label]['final_xyz']
        else:
            xyz = self.species_dict[label]['final_xyz']
        self.run_job(label, xyz, level_of_theory, job_type='sp')

    def run_scan_jobs(self, label):
        """
        Spawn rotor scan jobs using 'final_xyz' for species ot TS 'label'.
        """
        if 'scan' not in self.job_dict[label]:  # Check whether or not rotor scan jobs have been spawned yet
            # we're spawning the first scan job
            self.job_dict[label]['scan'] = dict()
        level_of_theory = 'b3lyp/6-311++g(d,p)'  # fixed level for rotor scans
        if self.is_label_ts(label):
            xyz = self.ts_dict[label]['final_xyz']
            for i in xrange(self.ts_dict['number_of_rotors']):
                scan = self.ts_dict['rotors_dict'][i]['scan']
                pivots = self.ts_dict['rotors_dict'][i]['pivots']
                self.run_job(label, xyz, level_of_theory, job_type='scan', scan=scan, pivots=pivots)
        else:
            xyz = self.species_dict[label]['final_xyz']
            for i in xrange(self.species_dict['number_of_rotors']):
                scan = self.species_dict['rotors_dict'][i]['scan']
                pivots = self.species_dict['rotors_dict'][i]['pivots']
                self.run_job(label, xyz, level_of_theory, job_type='scan', scan=scan, pivots=pivots)

    def parse_conformer_energy(self, job, label, i):
        """
        Parse E0 (Hartree) from the conformer opt output file, saves it in the 'conformer_energies' attribute.
        Troubleshoot if job crashed by running CBS-QB3 which is often more robust (but more time-consuming).
        """
        if job.job_status[1] == 'done':
            job = self.job_dict[label]['conformers'][i]
            local_path_to_output_file = os.path.join(job.local_path, 'output.out')
            log = Log(path='')
            log.determine_qm_software(fullpath=local_path_to_output_file)
            e0 = log.software_log.loadEnergy()
            self.species_dict[label]['conformer_energies'][i] = e0
        else:
            self.troubleshoot_ess(label=label, job=job, level_of_theory='b3lyp/6-311++g(d,p)',
                                  job_type='conformer', conformer=i)

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
        if (job.job_status[1] == 'done' and ('opt' in job.job_type and job.fine) or job.job_type == 'composite')\
                or parse_anyway:
            local_path_to_output_file = os.path.join(job.local_path, 'output.out')
            log = Log(path='')
            log.determine_qm_software(fullpath=local_path_to_output_file)
            coord, number, mass = log.software_log.loadGeometry()
            xyz = get_xyz_matrix(xyz=coord, from_arkane=True, number=number)
            if self.is_label_ts(label):
                self.ts_dict[label]['final_xyz'] = xyz
            else:
                self.species_dict[label]['final_xyz'] = xyz
            return True
        elif job.job_status[1] != 'done':
            self.troubleshoot_opt_jobs(label=label)
        return False  # return ``False``, so no freq / sp / scan jobs are initiated for this unoptimized geometry

    def check_freq_job(self, label, job):
        """
        Check that a freq job converged successfully. Also checks (QA) that no imaginary frequencies were assigned for
        stable species, and that exactly one imaginary frequency was assigned for a TS.
        """
        if job.job_status[1] != 'done':
            self.troubleshoot_ess(label=label, job=job, level_of_theory=job.level_of_theory, job_type='freq')
        local_path_to_output_file = os.path.join(job.local_path, 'output.out')
        parser = cclib.io.ccopen(local_path_to_output_file)
        data = parser.parse()
        neg_freq_counter = 0
        for freq in data.vibfreqs:
            if freq < 0:
                neg_freq_counter += 1
        if self.is_label_ts(label) and neg_freq_counter != 1:
                logging.error('TS {0} has {1} imaginary frequencies, should have exactly 1.'.format(label,
                                                                                                    neg_freq_counter))
        elif neg_freq_counter != 0:
                logging.error('species {0} has {1} imaginary frequencies, should have exactly 0.'.format(label,
                                                                                                         neg_freq_counter))

    def check_sp_job(self, label, job):
        """
        Check that a single point job converged successfully.
        """
        if job.job_status[1] != 'done':
            self.troubleshoot_ess(label=label, job=job, level_of_theory=job.level_of_theory, job_type='sp')

    def check_scan_job(self, label, job):
        """
        Check that a rotor scan job converged successfully. Also checks (QA) whether the scan is relatively "smooth",
        and recommends whether or not to use this rotor using the 'successful_rotors' and 'unsuccessful_rotors'
        attributes.
        """
        if job.job_status[1] != 'done':
            if self.is_label_ts(label):
                for i in xrange(self.species_dict[label]['number_of_rotors']):
                    if self.species_dict[label]['rotors_dict'][i]['pivots'] == job.pivots:
                        self.species_dict[label]['rotors_dict'][i]['success'] = False
                        break
                else:
                    raise SchedulerError('Could not match rotor with pivots {0} in TS {1}'.format(job.pivots, label))
            else:
                for i in xrange(self.ts_dict[label]['number_of_rotors']):
                    if self.ts_dict[label]['rotors_dict'][i]['pivots'] == job.pivots:
                        self.ts_dict[label]['rotors_dict'][i]['success'] = False
                        break
                else:
                    raise SchedulerError('Could not match rotor with pivots {0} in species {1}'.format(job.pivots, label))
        else:
            # ESS converged. Is rotor smooth?
            # TODO is rotor smooth?
            pass

    def is_label_ts(self, label):
        """
        Helper function to differentiate between a species and a TS
        """
        if label in self.unique_species_labels:
            is_ts = False
        elif label in self.unique_ts_labels:
            is_ts = True
        else:
            raise ValueError('Cannot identify label {0}'.format(label))
        return is_ts

    def get_servers_jobs_ids(self):
        """
        Check status on all active servers, return a list of relevant running job IDs
        """
        self.servers_jobs_ids = list()
        for server in self.servers:
            ssh = SSH_Client(server)
            self.servers_jobs_ids.extend(ssh.check_running_jobs_ids())

    def troubleshoot_opt_jobs(self, label):
        """
        we're troubleshooting for opt jobs.
        First check for server status and troubleshoot if needed. Then check for ESS status and troubleshoot
        if needed. Finally, check whether or not the last job had fine=True, add if it didn't run with fine.
        """
        if not self.composite:
            # Take only the geometry part for level of theory
            level_of_theory = self.level_of_theory.split('//')[1]
        else:
            # This is a composite method
            level_of_theory = self.level_of_theory
        previous_job_num = -1
        latest_job_num = -1
        job = None
        for job_name in self.job_dict[label]['opt'].iterkeys():  # get latest Job object for the species / TS
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
                    if self.is_label_ts(label):
                        xyz = self.ts_dict[label]['final_xyz']
                        self.ts_dict[label]['initial_xyz'] = xyz  # save for troubleshooting
                    else:
                        xyz = self.species_dict[label]['final_xyz']
                        self.species_dict[label]['initial_xyz'] = xyz  # save for troubleshooting
                    self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory,
                                 job_type='opt', is_ts=job.is_ts, fine=True)
            else:
                # job passed on the server, but failed in ESS calculation
                if previous_job_num >= 0 and job.fine:
                    previous_job = self.job_dict[label]['opt']['opt_a' + str(previous_job_num)]
                    if not previous_job.fine and previous_job.job_status[0] == 'done' \
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

    def troubleshoot_ess(self, label, job, level_of_theory, job_type, conformer=-1):
        """
        Troubleshoot issues related to the electronic structure software, such as conversion
        """
        logging.info('\n')
        logging.warn('Troubleshooting {job_type} job for {label} which failed with status'
                     ' {stat} in {soft}.'.format(job_type=job_type, label=label, stat=job.job_status[1],
                                                 soft=job.software))
        is_ts = self.is_label_ts(label)
        if is_ts:
            xyz = self.ts_dict[label]['initial_xyz']
        else:
            xyz = self.species_dict[label]['initial_xyz']
        if job.software == 'gaussian03':
            if 'cbs-qb3' not in job.ess_trsh_methods and self.level_of_theory != 'cbs-qb3':
                # try running CBS-QB3, which is relatively robust
                logging.info('Troubleshooting {type} job in {software} using CBS-QB3'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('cbs-qb3')
                level_of_theory = 'cbs-qb3'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory,
                             job_type='composite', is_ts=is_ts, fine=False, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'scf=qc' not in job.ess_trsh_methods:
                # calls a quadratically convergent SCF procedure
                logging.info('Troubleshooting {type} job in {software} using scf=qc'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('scf=qc')
                trsh = 'scf=qc'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, is_ts=is_ts, fine=False, trsh=trsh,
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'scf=nosymm' not in job.ess_trsh_methods:
                # calls a quadratically convergent SCF procedure
                logging.info('Troubleshooting {type} job in {software} using scf=nosymm'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('scf=nosymm')
                trsh = 'scf=nosymm'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, is_ts=is_ts, fine=False, trsh=trsh,
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'memory' not in job.ess_trsh_methods:
                # Increase memory allocation
                logging.info('Troubleshooting {type} job in {software} using memory'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('memory')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, is_ts=is_ts, fine=False, memory=2500,
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'scf=(qc,nosymm)' not in job.ess_trsh_methods:
                # try both qc and nosymm
                logging.info('Troubleshooting {type} job in {software} using scf=(qc,nosymm)'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('scf=(qc,nosymm)')
                trsh = 'scf=(qc,nosymm)'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, is_ts=is_ts, fine=False, trsh=trsh,
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif self.level_of_theory != 'cbs-qb3':
                # try both qc and nosymm with CBS-QB3
                logging.info('Troubleshooting {type} job in {software} using oth qc and nosymm with CBS-QB3'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('scf=(qc,nosymm) & CBS-QB3')
                level_of_theory = 'cbs-qb3'
                trsh = 'scf=(qc,nosymm)'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, is_ts=is_ts, fine=False, trsh=trsh,
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'qchem' not in job.ess_trsh_methods:
                # Try QChem
                logging.info('Troubleshooting {type} job in {software} using qchem'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('qchem')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory,
                             job_type=job_type, is_ts=is_ts, fine=False, software='qchem',
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'molpro_2012' not in job.ess_trsh_methods:
                # Try molpro
                logging.info('Troubleshooting {type} job in {software} using molpro_2012'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('molpro_2012')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory,
                             job_type=job_type, is_ts=is_ts, fine=False, software='molpro_2012',
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
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
                logging.info('Troubleshooting {type} job in {software} using max_cycles'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('max_cycles')  # avoids infinite looping
                trsh = '\nGEOM_OPT_MAX_CYCLES 250'  # default is 50
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, is_ts=is_ts, fine=False, trsh=trsh,
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'b3lyp' not in job.ess_trsh_methods:
                logging.info('Troubleshooting {type} job in {software} using b3lyp'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('b3lyp')
                # try converging with B3LYP
                level_of_theory = 'b3lyp/6-311++g(d,p)'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, is_ts=is_ts, fine=False, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'gaussian03' not in job.ess_trsh_methods:
                # Try Gaussian
                logging.info('Troubleshooting {type} job in {software} using gaussian03'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('gaussian03')
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory,
                             job_type=job_type, is_ts=is_ts, fine=False, software='gaussian03',
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'molpro' not in job.ess_trsh_methods:
                # Try molpro
                logging.info('Troubleshooting {type} job in {software} using molpro'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('molpro')
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory,
                             job_type=job_type, is_ts=is_ts, fine=False, software='molpro_2012',
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            else:
                logging.error('Could not troubleshoot geometry optimization for {label}! Tried'
                              ' troubleshooting with the following methods: {methods}'.format(
                    label=label, methods=job.ess_trsh_methods))
                raise SchedulerError('Could not troubleshoot geometry optimization for {label}! Tried'
                                     ' troubleshooting with the following methods: {methods}'.format(
                    label=label, methods=job.ess_trsh_methods))
        elif 'molpro' in job.software:
            if 'vdz' not in job.ess_trsh_methods:
                # try adding a level shift for alpha- and beta-spin orbitals
                logging.info('Troubleshooting {type} job in {software} using vdz'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('vdz')
                trsh = 'vdz'
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, is_ts=is_ts, fine=False, trsh=trsh,
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'shift' not in job.ess_trsh_methods:
                # try adding a level shift for alpha- and beta-spin orbitals
                logging.info('Troubleshooting {type} job in {software} using shift'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('shift')
                shift = 'shift,-1.0,-0.5;'
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, is_ts=is_ts, fine=False, shift=shift,
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'vdz' not in job.ess_trsh_methods:
                # try adding a level shift for alpha- and beta-spin orbitals
                logging.info('Troubleshooting {type} job in {software} using vdz'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('vdz')
                trsh = 'vdz'
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, is_ts=is_ts, fine=False, trsh=trsh,
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'memory' not in job.ess_trsh_methods:
                # Increase memory allocation, also run with a shift
                logging.info('Troubleshooting {type} job in {software} using memory'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('memory')
                shift = 'shift,-1.0,-0.5;'
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, is_ts=is_ts, fine=False, shift=shift, memory=5000,
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'gaussian03' not in job.ess_trsh_methods:
                # Try Gaussian
                logging.info('Troubleshooting {type} job in {software} using gaussian03'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('gaussian03')
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory,
                             job_type=job_type, is_ts=is_ts, fine=False, software='gaussian03',
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'qchem' not in job.ess_trsh_methods:
                # Try QChem
                logging.info('Troubleshooting {type} job in {software} using qchem'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('qchem')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory,
                             job_type=job_type, is_ts=is_ts, fine=False, software='qchem',
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            else:
                logging.error('Could not troubleshoot geometry optimization for {label}! Tried'
                              ' troubleshooting with the following methods: {methods}'.format(
                    label=label, methods=job.ess_trsh_methods))
                raise SchedulerError('Could not troubleshoot geometry optimization for {label}! Tried'
                                     ' troubleshooting with the following methods: {methods}'.format(
                    label=label, methods=job.ess_trsh_methods))

    def check_all_done(self, label):
        """
        Check that we have all required data for the species/TS in ``label``
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
