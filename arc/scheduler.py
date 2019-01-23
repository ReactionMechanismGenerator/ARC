#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)
import logging
import time
import os
import datetime
import numpy as np
import math

import cclib

import yaml
try:
    from yaml import CDumper as Dumper, CSafeDumper as SafeDumper, CLoader as Loader, CSafeLoader as SafeLoader
except ImportError:
    from yaml import Dumper, Loader, SafeLoader, SafeDumper

from arkane.statmech import Log

from arc import plotter
from arc import parser
from arc.job.job import Job
from arc.exceptions import SpeciesError, SchedulerError
from arc.job.ssh import SSH_Client
from arc.species import ARCSpecies, get_xyz_string
from arc.settings import rotor_scan_resolution, inconsistency_ab, inconsistency_az, maximum_barrier

##################################################################


class Scheduler(object):
    """
    ARC Scheduler class. Creates jobs, submits, checks status, troubleshoots.
    Each species in `species_list` has to have a unique label.

    The attributes are:

    ======================= ========= ==================================================================================
    Attribute               Type               Description
    ======================= ========= ==================================================================================
    `project`               ``str``   The project's name. Used for naming the working directory.
    'servers'               ''list''  A list of servers used for the present project
    `species_list`          ``list``  Contains input ``ARCSpecies`` objects (both species and TSs).
    `species_dict`          ``dict``  Keys are labels, values are ARCSpecies objects
    'unique_species_labels' ``list``  A list of species labels (checked for duplicates)
    `level_of_theory`       ``str``   *FULL* level of theory, e.g. 'CBS-QB3',
                                        'CCSD(T)-F12a/aug-cc-pVTZ//B3LYP/6-311++G(3df,3pd)'...
    'composite'             ``bool``    Whether level_of_theory represents a composite method or not
    `job_dict`              ``dict``  A dictionary of all scheduled jobs. Keys are species / TS labels,
                                        values are dictionaries where keys are job names (corresponding to
                                        'running_jobs' if job is running) and values are the Job objects.
    'running_jobs'          ``dict``  A dictionary of currently running jobs (a subset of `job_dict`).
                                        Keys are species/TS label, values are lists of job names
                                        (e.g. 'conformer3', 'opt_a123').
    'servers_jobs_ids'      ``list``  A list of relevant job IDs currently running on the server
    'fine'                  ``bool``  Whether or not to use a fine grid for opt jobs (spawns an additional job)
    'output'                ``dict``  Output dictionary with status and final QM file paths for all species
    `settings`              ``dict``  A dictionary of available servers and software
    `initial_trsh`          ``dict``  Troubleshooting methods to try by default. Keys are server names, values are trshs
    `restart_dict`          ``dict``  A restart dictionary parsed from a YAML restart file
    `project_directory`     ``str``   Folder path for the project: the input file path or ARC/Projects/project-name
    `save_restart`          ``bool``  Whether to start saving a restart file. ``True`` only after all species are loaded
                                        (otherwise saves a partial file and may cause loss of information)
    ======================= ========= ==================================================================================

    Dictionary structures:

*   job_dict = {label_1: {'conformers': {0: Job1,
                                         1: Job2, ...},
                          'opt':        {job_name1: Job1,
                                         job_name2: Job2, ...},
                          'sp':         {job_name1: Job1,
                                         job_name2: Job2, ...},
                          'freq':       {job_name1: Job1,
                                         job_name2: Job2, ...},
                          'composite':  {job_name1: Job1,
                                         job_name2: Job2, ...},
                          'scan':       {job_name1: Job1,
                                         job_name2: Job2, ...},
                          }
                label_2: {...},
                }

*   output = {label_1: {'status': ``str``,  # 'converged', or 'Error: <reason>'
                        'geo': <path to geometry optimization output file>,
                        'freq': <path to freq output file>,
                        'sp': <path to sp output file>,
                        'composite': <path to composite output file>,
             label_2: {...},
             }
    # Note that rotor scans are located under Species.rotors_dict
    """
    def __init__(self, project, settings, species_list, composite_method, conformer_level, opt_level, freq_level,
                 sp_level, scan_level, project_directory, fine=False, generate_conformers=True, scan_rotors=True,
                 initial_trsh=None, restart_dict=None):
        self.restart_dict = restart_dict
        self.species_list = species_list
        self.project = project
        self.settings = settings
        self.project_directory = project_directory
        self.job_dict = dict()
        self.servers_jobs_ids = list()
        self.running_jobs = dict()
        if self.restart_dict is not None:
            self.output = self.restart_dict['output']
            if 'running_jobs' in self.restart_dict:
                self.restore_running_jobs()
        else:
            self.output = dict()
        self.yaml_path = os.path.join(self.project_directory, 'restart.yml')
        self.report_time = time.time()  # init time for reporting status every 1 hr
        self.servers = list()
        self.composite_method = composite_method
        self.conformer_level = conformer_level
        self.opt_level = opt_level
        self.freq_level = freq_level
        self.sp_level = sp_level
        self.scan_level = scan_level
        self.fine = fine
        self.generate_conformers = generate_conformers
        self.scan_rotors = scan_rotors
        self.species_dict = dict()
        self.unique_species_labels = list()
        self.initial_trsh = initial_trsh if initial_trsh is not None else dict()
        self.save_restart = False
        for species in self.species_list:
            if not isinstance(species, ARCSpecies):
                raise SpeciesError('Each species in `species_list` must be an ARCSpecies object.'
                                   ' Got type {0} for {1}'.format(type(species), species.label))
            if species.label in self.unique_species_labels:
                raise SpeciesError('Each species in `species_list` has to have a unique label.'
                                   ' Label of species {0} is not unique.'.format(species.label))
            self.unique_species_labels.append(species.label)
            if species.label not in self.output:
                self.output[species.label] = dict()
                self.output[species.label]['status'] = ''
            else:
                self.output[species.label]['status'] += 'Restarted ARC at {0}; '.format(datetime.datetime.now())
            if species.label not in self.job_dict:
                self.job_dict[species.label] = dict()
            self.species_dict[species.label] = species
            if self.scan_rotors and not self.species_dict[species.label].number_of_rotors:
                self.species_dict[species.label].determine_rotors()
            if species.label not in self.running_jobs:
                self.running_jobs[species.label] = list()  # initialize before running the first job
            if self.species_dict[species.label].number_of_atoms == 1\
                    and 'sp' not in self.output[species.label] \
                    and 'composite' not in self.output[species.label]:
                # don't bother checking for already ran jobs for monoatomic species if restarting
                logging.debug('Species {0} is monoatomic'.format(species.label))
                if not self.species_dict[species.label].initial_xyz:
                    # generate a simple "Symbol   0.0   0.0   0.0" xyz matrix
                    if self.species_dict[species.label].mol is not None:
                        symbol = self.species_dict[species.label].mol.atoms[0].symbol
                    else:
                        symbol = species.label
                        logging.warning('Could not determine element of monoatomic species {0}.'
                                        ' Assuming element is {1}'.format(species.label, symbol))
                    self.species_dict[species.label].initial_xyz = symbol + '   0.0   0.0   0.0'
                    self.species_dict[species.label].final_xyz = symbol + '   0.0   0.0   0.0'
                # No need to run any job for a monoatomic species other than sp (or composite if relevant)
                if self.composite_method:
                    self.run_composite_job(species.label)
                else:
                    self.run_sp_job(label=species.label)
            elif self.species_dict[species.label].initial_xyz or self.species_dict[species.label].final_xyz:
                # For restarting purposes: check before running jobs whether they were already terminated
                # (check self.output) or whether they are currently running (check self.job_dict)
                if self.composite_method and 'composite' not in self.output[species.label]\
                        and 'composite' not in self.job_dict[species.label]:
                    self.run_composite_job(species.label)
                elif self.composite_method and 'freq' not in self.output[species.label]\
                        and 'freq' not in self.job_dict[species.label]:
                    self.run_freq_job(species.label)
                elif 'opt converged' not in self.output[species.label]['status']\
                        and 'opt' not in self.job_dict[species.label]:
                    self.run_opt_job(species.label)
                elif 'opt converged' in self.output[species.label]['status']:
                    # opt is done
                    if 'freq' not in self.output[species.label] and 'freq' not in self.job_dict[species.label]:
                        if self.species_dict[species.label].number_of_atoms > 1:
                            self.run_freq_job(species.label)
                    if 'sp' not in self.output[species.label] and 'sp' not in self.job_dict[species.label]:
                        self.run_sp_job(species.label)
                    if self.scan_rotors:
                        # restart-related check are performed in run_scan_jobs()
                        self.run_scan_jobs(species.label)
                if self.species_dict[species.label].t0 is None:
                    self.species_dict[species.label].t0 = time.time()
            elif not self.species_dict[species.label].is_ts and self.generate_conformers\
                    and 'geo' not in self.output[species.label]:
                self.species_dict[species.label].generate_conformers()
        self.save_restart = True
        self.timer = True
        self.schedule_jobs()

    def schedule_jobs(self):
        """
        The main job scheduling block
        """
        if self.generate_conformers:
            self.run_conformer_jobs()
        while self.running_jobs != {}:  # loop while jobs are still running
            logging.debug('Currently running jobs:\n{0}'.format(self.running_jobs))
            self.timer = True
            for label in self.unique_species_labels:
                # look for completed jobs and decide what jobs to run next
                self.get_servers_jobs_ids()  # updates `self.servers_jobs_ids`
                try:
                    job_list = self.running_jobs[label]
                except KeyError:
                    continue
                for job_name in job_list:
                    if 'conformer' in job_name:
                        i = int(job_name[9:])  # the conformer number. parsed from a string like 'conformer12'.
                        job = self.job_dict[label]['conformers'][i]
                        max_time = datetime.timedelta(seconds=max(1800, job.n_atoms ** 4))
                        if datetime.datetime.now() - job.date_time > max_time:
                            # resubmit this conformer job
                            logging.error('Conformer job {name} for {label} is taking too long (already {delta}). '
                                          'Terminating job and re-submitting'.format(name=job.job_name, label=label,
                                                                                     delta=max_time))
                            job.delete()
                            self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
                            self.run_job(label=label, xyz=job.xyz, level_of_theory=self.conformer_level,
                                         job_type='conformer', conformer=job.conformer)
                        elif self.job_dict[label]['conformers'][i].job_id not in self.servers_jobs_ids:
                            # this is a completed conformer job
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
                                logging.info('\nConformer jobs for {0} successfully terminated.\n'.format(
                                    label))
                                self.determine_most_stable_conformer(label)
                                if not self.composite_method:
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
                                if self.composite_method:
                                    # This was originally a composite method, probably troubleshooted as 'opt'
                                    self.run_composite_job(label)
                                else:
                                    if self.species_dict[label].number_of_atoms > 1:
                                        if 'freq' not in job_name:
                                            self.run_freq_job(label)
                                        else:  # this is an 'optfreq' job type
                                            self.check_freq_job(label=label, job=job)
                                    self.run_sp_job(label)
                                    self.run_scan_jobs(label)
                        self.timer = False
                        break
                    elif 'freq' in job_name\
                            and not self.job_dict[label]['freq'][job_name].job_id in self.servers_jobs_ids:
                        # this is NOT an 'optfreq' job
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
                            success = self.parse_composite_geo(label=label, job=job)
                            if success:
                                if not self.composite_method:
                                    # This wasn't originally a composite method, probably troubleshooted as such
                                    self.run_opt_job(label)
                                else:
                                    if self.species_dict[label].number_of_atoms > 1:
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
                if not job_list:  # if it's an empty dictionary
                    self.check_all_done(label)
                    if not self.running_jobs[label]:
                        # delete the label only if it represents an empty dictionary
                        del self.running_jobs[label]
            if self.timer:
                time.sleep(30)  # wait 30 sec before bugging the servers again.
            t = time.time() - self.report_time
            if t > 3600:
                self.report_time = time.time()
                logging.info('Currently running jobs:\n{0}'.format(self.running_jobs))

    def run_job(self, label, xyz, level_of_theory, job_type, fine=False, software=None, shift='', trsh='', memory=1500,
                conformer=-1, ess_trsh_methods=None, scan='', pivots=None, occ=None):
        """
        A helper function for running (all) jobs
        """
        ess_trsh_methods = ess_trsh_methods if ess_trsh_methods is not None else list()
        pivots = pivots if pivots is not None else list()
        if self.species_dict[label].t0 is None:
            self.species_dict[label].t0 = time.time()
        species = self.species_dict[label]
        job = Job(project=self.project, settings=self.settings, species_name=label, xyz=xyz, job_type=job_type,
                  level_of_theory=level_of_theory, multiplicity=species.multiplicity, charge=species.charge, fine=fine,
                  shift=shift, software=software, is_ts=species.is_ts, memory=memory, trsh=trsh, conformer=conformer,
                  ess_trsh_methods=ess_trsh_methods, scan=scan, pivots=pivots, occ=occ, initial_trsh=self.initial_trsh,
                  project_directory=self.project_directory)
        if conformer < 0:
            # this is NOT a conformer job
            self.running_jobs[label].append(job.job_name)  # mark as a running job
            try:
                self.job_dict[label][job_type]
            except KeyError:
                # Jobs of this type haven't been spawned for label, this could be a troubleshooting job
                self.job_dict[label][job_type] = dict()
            self.job_dict[label][job_type][job.job_name] = job
            self.job_dict[label][job_type][job.job_name].run()
            self.save_restart_dict()
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
        try:
            job.determine_job_status()  # also downloads output file
        except IOError:
            logging.warn('Tried to determine status of job {0}, but it seems like the job never ran.'
                         ' Re-running job.'.format(job.job_name))
            self.run_job(label=label, xyz=job.xyz, level_of_theory=job.level_of_theory, job_type=job.job_type,
                         fine=job.fine, software=job.software, shift=job.shift, trsh=job.trsh, memory=job.memory,
                         conformer=job.conformer, ess_trsh_methods=job.ess_trsh_methods, scan=job.scan,
                         pivots=job.pivots, occ=job.occ)
        if job.job_status[0] != 'running' and job.job_status[1] != 'running':
            self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
            self.timer = False
            job.run_time = str(datetime.datetime.now() - job.date_time).split('.')[0]
            job.write_completed_job_to_csv_file()
            logging.info('  Ending job {name} for {label} ({time})'.format(name=job.job_name, label=label,
                                                                           time=job.run_time))
            if job.job_status[0] != 'done':
                return False
            self.save_restart_dict()
            return True

    def run_conformer_jobs(self):
        """
        Select the most stable conformer for each species by spawning opt jobs at a selected low level DFT.
        The resulting conformer is saved in a <xyz matrix with element labels> format
        in self.species_dict[species.label]['initial_xyz']
        """
        for label in self.unique_species_labels:
            if not self.species_dict[label].is_ts and 'opt converged' not in self.output[label]['status']\
                        and 'opt' not in self.job_dict[label]:
                if len(self.species_dict[label].conformers) > 1:
                    self.job_dict[label]['conformers'] = dict()
                    for i, xyz in enumerate(self.species_dict[label].conformers):
                        self.run_job(label=label, xyz=xyz, level_of_theory=self.conformer_level, job_type='conformer',
                                     conformer=i)
                else:
                    if 'opt' not in self.job_dict[label] and 'composite' not in self.job_dict[label]\
                            and self.species_dict[label].number_of_atoms > 1:
                        # proceed only if opt (/composite) not already spawned
                        logging.info('Only one conformer is available for species {0},'
                                     ' using it for geometry optimization'.format(label))
                        self.species_dict[label].initial_xyz = self.species_dict[label].conformers[0]
                        if not self.composite_method:
                            self.run_opt_job(label)
                        else:
                            self.run_composite_job(label)

    def run_opt_job(self, label):
        """
        Spawn a geometry optimization job. The initial guess is taken from the `initial_xyz` attribute.
        """
        if 'opt' not in self.job_dict[label]:  # Check whether or not opt jobs have been spawned yet
            # we're spawning the first opt job for this species
            self.job_dict[label]['opt'] = dict()
        self.run_job(label=label, xyz=self.species_dict[label].initial_xyz, level_of_theory=self.opt_level,
                     job_type='opt', fine=False)

    def run_composite_job(self, label):
        """
        Spawn a composite job (e.g., CBS-QB3) using 'final_xyz' for species ot TS 'label'.
        """
        if not self.composite_method:
            raise SchedulerError('Cannot run {0} as a composite method, without specifying a method.'.format(label))
        if 'composite' not in self.job_dict[label]:  # Check whether or not composite jobs have been spawned yet
            # we're spawning the first composite job for this species
            self.job_dict[label]['composite'] = dict()
        if self.species_dict[label].final_xyz != '':
            xyz = self.species_dict[label].final_xyz
        else:
            xyz = self.species_dict[label].initial_xyz
        self.run_job(label=label, xyz=xyz, level_of_theory=self.composite_method, job_type='composite', fine=False)

    def run_freq_job(self, label):
        """
        Spawn a freq job using 'final_xyz' for species ot TS 'label'.
        If this was originally a composite job, run an appropriate separate freq job outputting the Hessian.
        """
        if 'freq' not in self.job_dict[label]:  # Check whether or not freq jobs have been spawned yet
            # we're spawning the first freq job for this species
            self.job_dict[label]['freq'] = dict()
        self.run_job(label=label, xyz=self.species_dict[label].final_xyz,
                     level_of_theory=self.freq_level, job_type='freq')

    def run_sp_job(self, label):
        """
        Spawn a single point job using 'final_xyz' for species ot TS 'label'.
        If the method is MRCI, first spawn a simple CCSD job, and use orbital determination to run the MRCI job
        """
        # determine_occ(label=self.label, xyz=self.xyz, charge=self.charge)
        if 'sp' not in self.job_dict[label]:  # Check whether or not single point jobs have been spawned yet
            # we're spawning the first sp job for this species
            self.job_dict[label]['sp'] = dict()
        if self.composite_method:
            raise SchedulerError('run_sp_job() was called for {0} which has a composite method level of theory'.format(
                label))
        if 'mrci' in self.sp_level:
            if self.job_dict[label]['sp']:
                # Parse orbital information from the CCSD job, then run MRCI
                job0 = None
                jobname0 = 0
                for job_name, job in self.job_dict[label]['sp']:
                    if int(job_name.split('_a')[-1]) > jobname0:
                        jobname0 = int(job_name.split('_a')[-1])
                        job0 = job
                with open(job0.local_path_to_output_file, 'rb') as f:
                    lines = f.readlines()
                    core = val = 0, 0
                    for line in lines:
                        if 'NUMBER OF CORE ORBITALS' in line:
                            core = int(line.split()[4])
                        elif 'NUMBER OF VALENCE ORBITALS' in line:
                            val = int(line.split()[4])
                        if val * core:
                            break
                    else:
                        raise SchedulerError('Could not determine number of core and valence orbitals from CCSD'
                                             ' sp calculation for {label}'.format(label=label))
                occ = val + core  # the occupied orbitals are the core and valence orbitals
                self.run_job(label=label, xyz=self.species_dict[label].final_xyz, level_of_theory='ccsd/vdz',
                             job_type='sp', occ=occ)
            else:
                # MRCI was requested but no sp job ran for this species, run CCSD first
                logging.info('running a CCSD job for {0} before MRCI'.format(label))
                self.run_job(label=label, xyz=self.species_dict[label].final_xyz, level_of_theory='ccsd/vdz',
                             job_type='sp')
        self.run_job(label=label, xyz=self.species_dict[label].final_xyz, level_of_theory=self.sp_level, job_type='sp')

    def run_scan_jobs(self, label):
        """
        Spawn rotor scan jobs using 'final_xyz' for species or TS 'label'.
        """
        if self.scan_rotors:
            if 'scan' not in self.job_dict[label]:  # Check whether or not rotor scan jobs have been spawned yet
                # we're spawning the first scan job for this species
                self.job_dict[label]['scan'] = dict()
            for i in range(self.species_dict[label].number_of_rotors):
                scan = self.species_dict[label].rotors_dict[i]['scan']
                pivots = self.species_dict[label].rotors_dict[i]['pivots']
                if 'scan_path' not in self.species_dict[label].rotors_dict[i]\
                        or not self.species_dict[label].rotors_dict[i]['scan_path']:
                    # check this job isn't already running on the server (from a restarted project):
                    for scan_job in self.job_dict[label]['scan'].values():
                        if scan_job.pivots == pivots and scan_job.job_name in self.running_jobs[label]:
                            break
                    else:
                        self.run_job(label=label, xyz=self.species_dict[label].final_xyz,
                                     level_of_theory=self.scan_level, job_type='scan', scan=scan, pivots=pivots)

    def parse_conformer_energy(self, job, label, i):
        """
        Parse E0 (Hartree) from the conformer opt output file, saves it in the 'conformer_energies' attribute.
        Troubleshoot if job crashed by running CBS-QB3 which is often more robust (but more time-consuming).
        """
        if job.job_status[1] == 'done':
            job = self.job_dict[label]['conformers'][i]
            log = Log(path='')
            log.determine_qm_software(fullpath=job.local_path_to_output_file)
            e0 = log.software_log.loadEnergy()
            self.species_dict[label].conformer_energies[i] = e0
        else:
            logging.warn('Conformer {i} for {label} did not converge!'.format(i=i, label=label))

    def determine_most_stable_conformer(self, label):
        """
        Determine the most stable conformer of species. Save the resulting xyz as `initial_xyz`
        """
        if all(e == 0.0 for e in self.species_dict[label].conformer_energies):
            logging.error('No conformer converged for species {0}. Will try to optimize the first conformer'
                          ' anyway'.format(label))
            self.output[label]['status'] += 'No conformers; '
        e_min = self.species_dict[label].conformer_energies[0]
        i_min = 0
        for i, ei in enumerate(self.species_dict[label].conformer_energies):
            if ei < e_min:
                e_min = ei
                i_min = i
        self.species_dict[label].initial_xyz = self.species_dict[label].conformers[i_min]

    def parse_composite_geo(self, label, job):
        """
        Check that a 'composite' job converged successfully, and parse the geometry into `final_xyz`.
        Also checks (QA) that no imaginary frequencies were assigned for stable species,
        and that exactly one imaginary frequency was assigned for a TS.
        Returns ``True`` if the job converged successfully, ``False`` otherwise and troubleshoots.
        """
        logging.debug('parsing composite geo for {0}'.format(job.job_name))
        freq_ok = False
        if job.job_status[1] == 'done':
            log = Log(path='')
            log.determine_qm_software(fullpath=job.local_path_to_output_file)
            coord, number, _ = log.software_log.loadGeometry()
            self.species_dict[label].final_xyz = get_xyz_string(xyz=coord, number=number)
            self.output[label]['status'] += 'composite converged; '
            self.output[label]['composite'] = os.path.join(job.local_path, 'output.out')
            logging.info('\nOptimized geometry for {label} at {level}:\n{xyz}'.format(label=label,
                        level=job.level_of_theory, xyz=self.species_dict[label].final_xyz))
            if self.species_dict[label].number_of_atoms > 2:
                success = plotter.show_sticks(xyz=self.species_dict[label].final_xyz)
                if not success:
                    plotter.plot_3d_mol_as_scatter(xyz=self.species_dict[label].final_xyz, path=None)
            # Check frequencies (using cclib crashed for CBS-QB3 output, using an explicit parser here)
            frequencies = parser.parse_frequencies(job.local_path_to_output_file, job.software)
            freq_ok = self.check_negative_freq(label=label, job=job, vibfreqs=frequencies)
            if freq_ok:
                # Update restart dictionary and save the yaml restart file:
                self.save_restart_dict()
                return True  # run freq / scan jobs on this optimized geometry
            elif not self.species_dict[label].is_ts:
                self.troubleshoot_negative_freq(label=label, job=job)
            self.species_dict[label].opt_level = self.composite_method
        if job.job_status[1] != 'done' or not freq_ok:
            self.troubleshoot_ess(label=label, job=job, level_of_theory=job.level_of_theory, job_type='composite')
        return False  # return ``False``, so no freq / scan jobs are initiated for this unoptimized geometry

    def parse_opt_geo(self, label, job):
        """
        Check that an 'opt' or 'optfreq' job converged successfully, and parse the geometry into `final_xyz`.
        If the job is 'optfreq', also checks (QA) that no imaginary frequencies were assigned for stable species,
        and that exactly one imaginary frequency was assigned for a TS.
        Returns ``True`` if the job (or both jobs) converged successfully, ``False`` otherwise and troubleshoots opt.
        """
        logging.debug('parsing opt geo for {0}'.format(job.job_name))
        if job.job_status[1] == 'done':
            log = Log(path='')
            log.determine_qm_software(fullpath=job.local_path_to_output_file)
            coord, number, _ = log.software_log.loadGeometry()
            self.species_dict[label].final_xyz = get_xyz_string(xyz=coord, number=number)
            if not job.fine and self.fine:
                # Run opt again using a finer grid.
                xyz = self.species_dict[label].final_xyz
                self.species_dict[label].initial_xyz = xyz  # save for troubleshooting, since trsh goes by initial
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, job_type='opt', fine=True)
            else:
                if 'optfreq' in job.job_name:
                    self.check_freq_job(label, job)
                self.output[label]['status'] += 'opt converged; '
                logging.info('\nOptimized geometry for {label} at {level}:\n{xyz}'.format(label=label,
                            level=job.level_of_theory, xyz=self.species_dict[label].final_xyz))
                if self.species_dict[label].number_of_atoms > 2:
                    success = plotter.show_sticks(xyz=self.species_dict[label].final_xyz)
                    if not success:
                        plotter.plot_3d_mol_as_scatter(xyz=self.species_dict[label].final_xyz, path=None)
                self.species_dict[label].opt_level = self.opt_level
                # Update restart dictionary and save the yaml restart file:
                self.save_restart_dict()
                return True  # run freq / sp / scan jobs on this optimized geometry
        else:
            self.troubleshoot_opt_jobs(label=label)
        return False  # return ``False``, so no freq / sp / scan jobs are initiated for this unoptimized geometry

    def check_freq_job(self, label, job):
        """
        Check that a freq job converged successfully. Also checks (QA) that no imaginary frequencies were assigned for
        stable species, and that exactly one imaginary frequency was assigned for a TS.
        """
        if job.job_status[1] == 'done':
            if not os.path.isfile(job.local_path_to_output_file):
                raise SchedulerError('Called check_freq_job with no output file')
            ccparser = cclib.io.ccopen(str(job.local_path_to_output_file))
            try:
                data = ccparser.parse()
                vibfreqs = data.vibfreqs
            except AssertionError:
                """
                In cclib/parser/qchemparser.py there's an assertion of `assert 'Beta MOs' in line`
                which sometimes fails (CClib issue https://github.com/cclib/cclib/issues/678)
                """
                vibfreqs = parser.parse_frequencies(path=str(job.local_path_to_output_file), software=job.software)
            freq_ok = self.check_negative_freq(label=label, job=job, vibfreqs=vibfreqs)
            if not self.species_dict[label].is_ts and not freq_ok:
                self.troubleshoot_negative_freq(label=label, job=job)
        else:
            self.troubleshoot_ess(label=label, job=job, level_of_theory=job.level_of_theory, job_type='freq')

    def check_negative_freq(self, label, job, vibfreqs):
        """
        A helper function for determining the number of negative frequencies. Also logs appropriate errors.
        Returns ``True`` if the number of negative frequencies is as excepted, ``False`` otherwise.
        """
        neg_freq_counter = 0
        neg_fre = None
        for freq in vibfreqs:
            if freq < 0:
                neg_freq_counter += 1
                neg_fre = freq
        if self.species_dict[label].is_ts and neg_freq_counter != 1:
                logging.error('TS {0} has {1} imaginary frequencies,'
                              ' should have exactly 1.'.format(label, neg_freq_counter))
                self.output[label]['status'] += 'Warning: {0} imaginary freq for TS; '.format(neg_freq_counter)
                return False
        elif not self.species_dict[label].is_ts and neg_freq_counter != 0:
                logging.error('species {0} has {1} imaginary frequencies,'
                              ' should have exactly 0.'.format(label, neg_freq_counter))
                self.output[label]['status'] += 'Warning: {0} imaginary freq for stable species; '.format(
                    neg_freq_counter)
                return False
        else:
            if self.species_dict[label].is_ts:
                logging.info('TS {0} has exactly one imaginary frequency: {1}'.format(label, neg_fre))
            self.output[label]['status'] += 'freq converged; '
            self.output[label]['geo'] = job.local_path_to_output_file
            self.output[label]['freq'] = job.local_path_to_output_file
            # Update restart dictionary and save the yaml restart file:
            self.save_restart_dict()
            return True

    def check_sp_job(self, label, job):
        """
        Check that a single point job converged successfully.
        """
        if 'mrci' in self.sp_level and 'mrci' not in job.level_of_theory:
            # This is a CCSD job ran before MRCI. Spawn MRCI
            self.run_sp_job(label)
        elif job.job_status[1] == 'done':
            self.output[label]['status'] += 'sp converged; '
            self.output[label]['sp'] = os.path.join(job.local_path, 'output.out')
            # Update restart dictionary and save the yaml restart file:
            self.save_restart_dict()
        else:
            self.troubleshoot_ess(label=label, job=job, level_of_theory=job.level_of_theory, job_type='sp')

    def check_scan_job(self, label, job):
        """
        Check that a rotor scan job converged successfully. Also checks (QA) whether the scan is relatively "smooth",
        and whether the optimized geometry indeed represents the minimum energy conformer.
        Recommends whether or not to use this rotor using the 'successful_rotors' and 'unsuccessful_rotors' attributes.
        * rotors_dict structure (attribute of ARCSpecies):
            rotors_dict: {1: {'pivots': pivots_list,
                              'top': top_list,
                              'scan': scan_list,
                              'success: ''bool'',
                              'times_dihedral_set': ``int``
                              'scan_path': <path to scan output file>},
                          2: {}, ...
                         }
        """
        for i in range(self.species_dict[label].number_of_rotors):
            message = ''
            if self.species_dict[label].rotors_dict[i]['pivots'] == job.pivots:
                invalidate = False
                if job.job_status[1] == 'done':
                    # ESS converged. Get PES using Arkane:
                    log = Log(path='')
                    log.determine_qm_software(fullpath=job.local_path_to_output_file)
                    plot_scan = True
                    try:
                        v_list, angle = log.software_log.loadScanEnergies()
                    except ZeroDivisionError:
                        logging.error('Energies from rotor scan of {label} between pivots {pivots} could not'
                                      'be read. Invalidating rotor.'.format(label=label, pivots=job.pivots))
                        invalidate = True
                        plot_scan = False
                    else:
                        v_list = np.array(v_list, np.float64)
                        v_list = v_list * 0.001  # convert to kJ/mol
                        # 1. Check smoothness:
                        if abs(v_list[-1] - v_list[0]) > inconsistency_az:
                            # initial and final points differ by more than `inconsistency_az` kJ/mol.
                            # seems like this rotor broke the conformer. Invalidate
                            error_message = 'Rotor scan of {label} between pivots {pivots} is inconsistent by more' \
                                            ' than {inconsistency} kJ/mol between initial and final positions.' \
                                            ' Invalidating rotor.\nv_list[0] = {v0}, v_list[-1] = {vneg1}'.format(
                                             label=label, pivots=job.pivots, inconsistency=inconsistency_az,
                                             v0=v_list[0], vneg1=v_list[-1])
                            logging.error(error_message)
                            message += error_message + '; '
                            invalidate = True
                        if not invalidate:
                            v_last = v_list[-1]
                            for v in v_list:
                                if abs(v - v_last) > inconsistency_ab:
                                    # Two consecutive points on the scan differ by more than `inconsistency_ab` kJ/mol.
                                    # This is a serious inconsistency. Invalidate
                                    error_message = 'Rotor scan of {label} between pivots {pivots} is inconsistent by' \
                                                    'more than {inconsistency} kJ/mol between two consecutive points.' \
                                                    ' Invalidating rotor.'.format(label=label, pivots=job.pivots,
                                                                                  inconsistency=inconsistency_ab)
                                    logging.error(error_message)
                                    message += error_message + '; '
                                    invalidate = True
                                    break
                                if abs(v - v_list[0]) > maximum_barrier:
                                    # The barrier for the hinderd rotor is higher than `maximum_barrier` kJ/mol.
                                    # Invalidate
                                    warn_message = 'Rotor scan of {label} between pivots {pivots} has a barrier larger' \
                                                   ' than {maximum_barrier} kJ/mol. Invalidating rotor.'.format(
                                                    label=label, pivots=job.pivots, maximum_barrier=maximum_barrier)
                                    logging.warn(warn_message)
                                    message += warn_message + '; '
                                    invalidate = True
                                    break
                                v_last = v
                        # 2. Check conformation:
                        if not invalidate:
                            v_diff = (v_list[0] - np.min(v_list))
                            if v_diff >= 2:
                                self.species_dict[label].rotors_dict[i]['success'] = False
                                logging.info('Species {label} is not oriented correctly around pivots {pivots}, searching'
                                             ' for a better conformation...'.format(label=label, pivots=job.pivots))
                                # Find the rotation dihedral in degrees to the closest minimum:
                                min_v = v_list[0]
                                min_index = 0
                                for j, v in enumerate(v_list):
                                    if v < min_v - 2:
                                        min_v = v
                                        min_index = j
                                self.species_dict[label].set_dihedral(scan=self.species_dict[label].rotors_dict[i]['scan'],
                                                                      pivots=self.species_dict[label].rotors_dict[i]['pivots'],
                                                                      deg_increment=min_index*rotor_scan_resolution)
                                self.delete_all_species_jobs(label)
                                self.run_opt_job(label)  # run opt on newly generated initial_xyz with the desired dihedral
                            else:
                                self.species_dict[label].rotors_dict[i]['success'] = True
                        if plot_scan:
                            invalidated = ''
                            if invalidate:
                                invalidated = '*INVALIDATED* '
                            message += invalidated
                            logging.info('{invalidated}Rotor scan between pivots {pivots} for {label} is:'.format(
                                invalidated=invalidated, pivots=self.species_dict[label].rotors_dict[i]['pivots'],
                                label=label))
                            folder_name = 'TSs' if job.is_ts else 'Species'
                            rotor_path = os.path.join(self.project_directory, 'output', folder_name,
                                                      job.species_name, 'rotors')
                            plotter.plot_rotor_scan(angle, v_list, path=rotor_path, pivots=job.pivots, comment=message)
                else:
                    # scan job crashed
                    invalidate = True
                if invalidate:
                    self.species_dict[label].rotors_dict[i]['success'] = False

                else:
                    self.species_dict[label].rotors_dict[i]['success'] = True
                    self.species_dict[label].rotors_dict[i]['scan_path'] = job.local_path_to_output_file
                break  # `job` has only one pivot. Break if found, otherwise raise an error.
        else:
            raise SchedulerError('Could not match rotor with pivots {0} in species {1}'.format(job.pivots, label))
        self.save_restart_dict()

    def get_servers_jobs_ids(self):
        """
        Check status on all active servers, return a list of relevant running job IDs
        """
        self.servers_jobs_ids = list()
        for server in self.servers:
            ssh = SSH_Client(server)
            self.servers_jobs_ids.extend(ssh.check_running_jobs_ids())

    def troubleshoot_negative_freq(self, label, job):
        """
        Troubleshooting cases where stable species (not TS's) have negative frequencies.
        We take  +/-1.1 displacements, generating several initial geometries, and running them as conformers
        """
        factor = 1.1
        ccparser = cclib.io.ccopen(str(job.local_path_to_output_file))
        data = ccparser.parse()
        vibfreqs = data.vibfreqs
        vibdisps = data.vibdisps
        atomnos = data.atomnos
        atomcoords = data.atomcoords
        if len(self.species_dict[label].neg_freqs_trshed) > 10:
            logging.error('Species {0} was troubleshooted for negative frequencies too many times.')
            if not self.scan_rotors:
                logging.error('The `scan_rotors` parameter is turned off, cannot troubleshoot geometry using'
                              ' dihedral modifications.')
                self.output[label]['status'] = 'scan_rotors = False; '
            logging.error('Invalidating species.')
            self.output[label]['status'] = 'Error: Encountered negative frequencies too many times; '
            return
        neg_freqs_idx = list()  # store indices w.r.t. vibfreqs
        largest_neg_freq_idx = 0  # index in vibfreqs
        for i, freq in enumerate(vibfreqs):
            if freq < 0:
                neg_freqs_idx.append(i)
                if vibfreqs[i] < vibfreqs[largest_neg_freq_idx]:
                    largest_neg_freq_idx = i
            else:
                # assuming frequencies are ordered, break after the first positive freq encounter
                break
        if vibfreqs[largest_neg_freq_idx] >= 0 or len(neg_freqs_idx) == 0:
            raise SchedulerError('Could not determine negative frequency in species {0} while troubleshooting for'
                                 ' negative frequencies'.format(label))
        if len(neg_freqs_idx) == 1 and len(self.species_dict[label].neg_freqs_trshed) == 0:
            # species has one negative frequency, and has not been troubleshooted for it before
            logging.info('Species {0} has a negative frequencies ({1}). Perturbing its geometry using the respective '
                         'vibrational displacements'.format(label, vibfreqs[largest_neg_freq_idx]))
            neg_freqs_idx = [largest_neg_freq_idx]  # indices of the negative frequencies to troubleshoot for
        elif len(neg_freqs_idx) == 1 and len(self.species_dict[label].neg_freqs_trshed) == 0:
            # species has one negative frequency, and has been troubleshooted for it before
            factor = 1.3
            logging.info('Species {0} has a negative frequencies ({1}). Perturbing its geometry using the respective '
                         'vibrational displacements, this time using a larger factor ({2})'.format(
                label, vibfreqs[largest_neg_freq_idx], factor))
            neg_freqs_idx = [largest_neg_freq_idx]  # indices of the negative frequencies to troubleshoot for
        elif len(neg_freqs_idx) > 1 and len(self.species_dict[label].neg_freqs_trshed) == 0:
            # species has more than one negative frequency, and has not been troubleshooted for it before
            logging.info('Species {0} has {1} negative frequencies. Perturbing its geometry using the vibrational '
                         'displacements of its largest negative frequency, {2}'.format(label, len(neg_freqs_idx),
                                                                                       vibfreqs[largest_neg_freq_idx]))
            neg_freqs_idx = [largest_neg_freq_idx]  # indices of the negative frequencies to troubleshoot for
        elif len(neg_freqs_idx) > 1 and len(self.species_dict[label].neg_freqs_trshed) > 0:
            # species has more than one negative frequency, and has been troubleshooted for it before
            logging.info('Species {0} has {1} negative frequencies. Perturbing its geometry using the vibrational'
                         ' displacements of ALL negative frequencies'.format(label, len(neg_freqs_idx)))
        self.species_dict[label].neg_freqs_trshed.extend([round(vibfreqs[i], 2) for i in neg_freqs_idx])  # record freqs
        logging.info('Deleting all currently running jobs for species {0} before troubleshooting for'
                     ' negative frequency...'.format(label))
        self.delete_all_species_jobs(label)
        self.species_dict[label].conformers = list()  # initialize the conformer list
        self.species_dict[label].conformer_energies = list()
        atomcoords = atomcoords[-1]  # it's a list within a list, take the last geometry
        for neg_freq_idx in neg_freqs_idx:
            displacement = vibdisps[neg_freq_idx]
            xyz1 = atomcoords + factor * displacement
            xyz2 = atomcoords - factor * displacement
            self.species_dict[label].conformers.append(get_xyz_string(xyz=xyz1, number=atomnos))
            self.species_dict[label].conformers.append(get_xyz_string(xyz=xyz2, number=atomnos))
            self.species_dict[label].conformer_energies.extend([0.0, 0.0])  # a placeholder (lists are synced)
        self.job_dict[label]['conformers'] = dict()  # initialize the conformer job dictionary
        for i, xyz in enumerate(self.species_dict[label].conformers):
            self.run_job(label=label, xyz=xyz, level_of_theory=self.conformer_level, job_type='conformer', conformer=i)

    def troubleshoot_opt_jobs(self, label):
        """
        we're troubleshooting for opt jobs.
        First check for server status and troubleshoot if needed. Then check for ESS status and troubleshoot
        if needed. Finally, check whether or not the last job had fine=True, add if it didn't run with fine.
        """
        previous_job_num = -1
        latest_job_num = -1
        job = None
        for job_name in self.job_dict[label]['opt'].keys():  # get latest Job object for the species / TS
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
                    self.parse_opt_geo(label=label, job=job)
                    xyz = self.species_dict[label].final_xyz
                    self.species_dict[label].initial_xyz = xyz  # save for troubleshooting, since trsh goes by initial
                    self.run_job(label=label, xyz=xyz, level_of_theory=self.opt_level, job_type='opt', fine=True)
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
                                      ' grid again.'.format(label=label))
                        self.parse_opt_geo(label=label, job=previous_job)
                else:
                    self.troubleshoot_ess(label=label, job=job, level_of_theory=self.opt_level, job_type='opt')
        else:
            job.troubleshoot_server()

    def troubleshoot_ess(self, label, job, level_of_theory, job_type, conformer=-1):
        """
        Troubleshoot issues related to the electronic structure software, such as conversion
        """
        logging.info('\n')
        logging.warn('Troubleshooting {label} job {job_name} which failed with status "{stat}" in {soft}.'.format(
            job_name=job.job_name, label=label, stat=job.job_status[1], soft=job.software))
        xyz = self.species_dict[label].initial_xyz
        if 'Unknown reason' in job.job_status[1] and 'change_node' not in job.ess_trsh_methods:
            job.ess_trsh_methods.append('change_node')
            job.troubleshoot_server()
            if job.job_name not in self.running_jobs[label]:
                self.running_jobs[label].append(job.job_name)  # mark as a running job
        elif job.software == 'gaussian':
            if 'scf=(qc,nosymm)' not in job.ess_trsh_methods:
                # try both qc and nosymm
                logging.info('Troubleshooting {type} job in {software} using scf=(qc,nosymm)'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('scf=(qc,nosymm)')
                trsh = 'scf=(qc,nosymm)'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'scf=(NDump=30)' not in job.ess_trsh_methods:
                # Allows dynamic dumping for up to N SCF iterations (slower conversion)
                logging.info('Troubleshooting {type} job in {software} using scf=(NDump=30)'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('scf=(NDump=30)')
                trsh = 'scf=(NDump=30)'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'scf=NoDIIS' not in job.ess_trsh_methods:
                # Switching off Pulay's Direct Inversion
                logging.info('Troubleshooting {type} job in {software} using scf=NoDIIS'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('scf=NoDIIS')
                trsh = 'scf=NoDIIS'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'int=(Acc2E=14)' not in job.ess_trsh_methods:  # does not work in g03
                # Change integral accuracy (skip everything up to 1E-14 instead of 1E-12)
                logging.info('Troubleshooting {type} job in {software} using int=(Acc2E=14)'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('int=(Acc2E=14)')
                trsh = 'int=(Acc2E=14)'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'cbs-qb3' not in job.ess_trsh_methods and self.composite_method != 'cbs-qb3':
                # try running CBS-QB3, which is relatively robust
                logging.info('Troubleshooting {type} job in {software} using CBS-QB3'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('cbs-qb3')
                level_of_theory = 'cbs-qb3'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, job_type='composite',
                             fine=job.fine, ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'scf=nosymm' not in job.ess_trsh_methods:
                # calls a quadratically convergent SCF procedure
                logging.info('Troubleshooting {type} job in {software} using scf=nosymm'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('scf=nosymm')
                trsh = 'scf=nosymm'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'memory' not in job.ess_trsh_methods:
                # Increase memory allocation
                memory = 3000
                logging.info('Troubleshooting {type} job in {software} using memory: {mem} MB'.format(
                    type=job_type, software=job.software, mem=memory))
                job.ess_trsh_methods.append('memory')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, memory=memory, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif self.composite_method != 'cbs-qb3' and 'scf=(qc,nosymm) & CBS-QB3' not in job.ess_trsh_methods:
                # try both qc and nosymm with CBS-QB3
                logging.info('Troubleshooting {type} job in {software} using oth qc and nosymm with CBS-QB3'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('scf=(qc,nosymm) & CBS-QB3')
                level_of_theory = 'cbs-qb3'
                trsh = 'scf=(qc,nosymm)'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'qchem' not in job.ess_trsh_methods:
                # Try QChem
                logging.info('Troubleshooting {type} job using qchem instead of {software}'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('qchem')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, job_type=job_type, fine=job.fine,
                             software='qchem', ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'molpro' not in job.ess_trsh_methods:
                # Try molpro
                logging.info('Troubleshooting {type} job using molpro instead of {software}'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('molpro')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, job_type=job_type, fine=job.fine,
                             software='molpro', ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
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
                job.ess_trsh_methods.append('max_cycles')
                trsh = '\n   GEOM_OPT_MAX_CYCLES 250'  # default is 50
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'SCF failed' in job.job_status[1] and 'DIIS_GDM' not in job.ess_trsh_methods:
                # change the SCF algorithm and increase max SCF cycles
                logging.info('Troubleshooting {type} job in {software} using the DIIS_GDM SCF algorithm'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('DIIS_GDM')
                trsh = '\n   SCF_ALGORITHM DIIS_GDM\n   MAX_SCF_CYCLES 250'  # default is 50
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'SYM_IGNORE' not in job.ess_trsh_methods:  # symmetry - look in manual, no symm if fails
                # change the SCF algorithm and increase max SCF cycles
                logging.info('Troubleshooting {type} job in {software} using SYM_IGNORE'
                             ' as well as the DIIS_GDM SCF algorithm'.format(type=job_type, software=job.software))
                job.ess_trsh_methods.append('SYM_IGNORE')
                trsh = '\n   SCF_ALGORITHM DIIS_GDM\n   MAX_SCF_CYCLES 250\n   SYM_IGNORE     True'
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'b3lyp' not in job.ess_trsh_methods:
                logging.info('Troubleshooting {type} job in {software} using b3lyp'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('b3lyp')
                # try converging with B3LYP
                level_of_theory = 'b3lyp/6-311++g(d,p)'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'gaussian' not in job.ess_trsh_methods:
                # Try Gaussian
                logging.info('Troubleshooting {type} job using gaussian instead of {software}'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('gaussian')
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, job_type=job_type, fine=job.fine,
                             software='gaussian', ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'molpro' not in job.ess_trsh_methods:
                # Try molpro
                logging.info('Troubleshooting {type} job using molpro instead of {software}'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('molpro')
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, job_type=job_type, fine=job.fine,
                             software='molpro', ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            else:
                logging.error('Could not troubleshoot geometry optimization for {label}! Tried'
                              ' troubleshooting with the following methods: {methods}'.format(
                    label=label, methods=job.ess_trsh_methods))
                raise SchedulerError('Could not troubleshoot geometry optimization for {label}! Tried'
                                     ' troubleshooting with the following methods: {methods}'.format(
                    label=label, methods=job.ess_trsh_methods))
        elif 'molpro' in job.software:
            if 'additional memory (mW) required' in job.job_status[1]:
                # Increase memory allocation.
                # job.job_status[1] will be for example `'errored: additional memory (mW) required: 996.31'`.
                # The number is the ADDITIONAL memory required
                job.ess_trsh_methods.append('memory')
                add_mem = float(job.job_status[1].split()[-1])  # parse Molpro's requirement
                add_mem = int(math.ceil(add_mem / 100.0)) * 100  # round up to the next hundred
                add_mem += 250  # be conservative
                memory = job.memory + add_mem
                if memory < 5000:
                    memory = 5000
                logging.info('Troubleshooting {type} job in {software} using memory: {mw} MW'.format(
                    type=job_type, software=job.software, mw=memory))
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, shift=job.shift, memory=memory,
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'shift' not in job.ess_trsh_methods:
                # try adding a level shift for alpha- and beta-spin orbitals
                logging.info('Troubleshooting {type} job in {software} using shift'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('shift')
                shift = 'shift,-1.0,-0.5;'
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, shift=shift, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'vdz' not in job.ess_trsh_methods:
                # try adding a level shift for alpha- and beta-spin orbitals
                logging.info('Troubleshooting {type} job in {software} using vdz'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('vdz')
                trsh = 'vdz'
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'vdz & shift' not in job.ess_trsh_methods:
                # try adding a level shift for alpha- and beta-spin orbitals
                logging.info('Troubleshooting {type} job in {software} using vdz'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('vdz & shift')
                shift = 'shift,-1.0,-0.5;'
                trsh = 'vdz'
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, shift=shift, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'memory' not in job.ess_trsh_methods:
                # Increase memory allocation, also run with a shift
                job.ess_trsh_methods.append('memory')
                memory = 5000
                logging.info('Troubleshooting {type} job in {software} using memory: {mw} MW'.format(
                    type=job_type, software=job.software, mw=memory))
                shift = 'shift,-1.0,-0.5;'
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, shift=shift, memory=memory,
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'gaussian' not in job.ess_trsh_methods:
                # Try Gaussian
                logging.info('Troubleshooting {type} job using gaussian instead of {software}'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('gaussian')
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, job_type=job_type, fine=job.fine,
                             software='gaussian', ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'qchem' not in job.ess_trsh_methods:
                # Try QChem
                logging.info('Troubleshooting {type} job using qchem instead of {software}'.format(
                    type=job_type, software=job.software))
                job.ess_trsh_methods.append('qchem')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, job_type=job_type, fine=job.fine,
                             software='qchem', ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
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
        status = self.output[label]['status']
        if 'error' not in status and ('composite converged' in status or ('sp converged' in status and
                (self.species_dict[label].number_of_atoms == 1 or ('freq converged' in status and 'opt converged' in status)))):
            d, h, m, s = time_lapse(t0=self.species_dict[label].t0)
            self.species_dict[label].execution_time = '{0}{1:02.0f}:{2:02.0f}:{3:02.0f}'.format(d, h, m, s)
            logging.info('\nAll jobs for species {0} successfully converged.'
                         ' Elapsed time: {1}'.format(label, self.species_dict[label].execution_time))
            self.output[label]['status'] += '; ALL converged'
            plotter.save_geo(species=self.species_dict[label], project_directory=self.project_directory)
        elif not self.output[label]['status']:
            self.output[label]['status'] = 'nothing converged'
            logging.error('species {0} did not converge. Status is: {1}'.format(label, status))
            # Update restart dictionary and save the yaml restart file:
            self.save_restart_dict()

    def delete_all_species_jobs(self, label):
        """
        Delete all jobs of species/TS represented by `label`
        """
        logging.debug('Deleting all jobs for species {0}'.format(label))
        for job_dict in self.job_dict[label].values():
            for job_name, job in job_dict.items():
                if job_name in self.running_jobs[label]:
                    logging.debug('Deleted job {0}'.format(job_name))
                    job.delete()
        self.running_jobs[label] = list()

    def restore_running_jobs(self):
        """
        Make Job objects for jobs which were running in the previous session.
        Important for the restart feature so long jobs won't be ran twice.
        """
        jobs = self.restart_dict['running_jobs']
        for spc_label in jobs.keys():
            if spc_label not in self.running_jobs:
                self.running_jobs[spc_label] = list()
            for job_description in jobs[spc_label]:
                self.running_jobs[spc_label].append(job_description['job_name'])
                for species in self.species_list:
                    if species.label == spc_label:
                        break
                else:
                    raise SchedulerError('Could not find species {0} in the restart file'.format(spc_label))
                job = Job(project=self.project, settings=self.settings, species_name=spc_label,
                          xyz=job_description['xyz'], job_type=job_description['job_type'],
                          level_of_theory=job_description['level_of_theory'], multiplicity=species.multiplicity,
                          charge=species.charge, fine=job_description['fine'], shift=job_description['shift'],
                          is_ts=species.is_ts, memory=job_description['memory'], trsh=job_description['trsh'],
                          ess_trsh_methods=job_description['ess_trsh_methods'], scan=job_description['scan'],
                          pivots=job_description['pivots'], occ=job_description['occ'],
                          project_directory=job_description['project_directory'], job_num=job_description['job_num'],
                          job_server_name=job_description['job_server_name'], job_name=job_description['job_name'],
                          job_id=job_description['job_id'], server=job_description['server'],
                          run_time=job_description['run_time'], date_time=job_description['date_time'])
                if spc_label not in self.job_dict:
                    self.job_dict[spc_label] = dict()
                if job_description['job_type'] not in self.job_dict[spc_label]:
                    self.job_dict[spc_label][job_description['job_type']] = dict()
                self.job_dict[spc_label][job_description['job_type']][job_description['job_name']] = job
                self.servers_jobs_ids.append(job.job_id)
        if self.job_dict:
            content = 'Restarting ARC, tracking the following jobs spawned in a previous session:'
            for spc_label in self.job_dict.keys():
                content += '\n' + spc_label + ': '
                for tob_type in self.job_dict[spc_label].keys():
                    for job_name in self.job_dict[spc_label][tob_type].keys():
                        content += job_name + ', '
            content += '\n\n'
            logging.info(content)

    def save_restart_dict(self):
        """
        Update the restart_dict and save the restart.yml file
        """
        if self.save_restart:
            logging.debug('Creating a restart file...')
            self.restart_dict['output'] = self.output
            self.restart_dict['species'] = [spc.as_dict() for spc in self.species_dict.values()]
            self.restart_dict['running_jobs'] = dict()
            for spc in self.species_dict.values():
                if spc.label in self.running_jobs:
                    self.restart_dict['running_jobs'][spc.label] =\
                        [self.job_dict[spc.label][job_name.split('_')[0]][job_name].as_dict()
                         for job_name in self.running_jobs[spc.label] if 'conformer' not in job_name]
            content = yaml.safe_dump(data=self.restart_dict, encoding='utf-8', allow_unicode=True)
            with open(self.yaml_path, 'w') as f:
                f.write(content)
            logging.debug('Dumping restart dictionary:\n{0}'.format(self.restart_dict))


def time_lapse(t0):
    """A helper function returning the elapsed time since t0"""
    t = time.time() - t0
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        d = str(d) + ' days, '
    else:
        d = ''
    return d, h, m, s
