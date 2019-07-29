#!/usr/bin/env python
# encoding: utf-8

"""
A module for scheduling jobs
Includes spawning, terminating, checking, and troubleshooting various jobs
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import time
import os
import datetime
import numpy as np
import math
import shutil
import logging
from IPython.display import display

import cclib

from arkane.statmech import determine_qm_software
from rmgpy.reaction import Reaction
from rmgpy.exceptions import InputError as RMGInputError, AtomTypeError

from arc.common import get_logger, read_yaml_file, save_yaml_file, get_ordinal_indicator, min_list
import arc.rmgdb as rmgdb
from arc import plotter
from arc import parser
from arc.job.job import Job
from arc.arc_exceptions import SpeciesError, SchedulerError, TSError, SanitizationError
from arc.job.ssh import SSHClient
from arc.job.local import check_running_jobs_ids
from arc.species.species import ARCSpecies, TSGuess, determine_rotor_symmetry
from arc.species.converter import get_xyz_string, molecules_from_xyz, check_isomorphism, standardize_xyz_string
import arc.species.conformers as conformers
from arc.ts.atst import autotst
from arc.settings import rotor_scan_resolution, inconsistency_ab, inconsistency_az, maximum_barrier, default_job_types,\
    servers

##################################################################

logger = get_logger()


class Scheduler(object):
    """
    ARC Scheduler class. Creates jobs, submits, checks status, troubleshoots.
    Each species in `species_list` has to have a unique label.

    The attributes are:

    ======================= ========= ==================================================================================
    Attribute               Type               Description
    ======================= ========= ==================================================================================
    `project`               ``str``   The project's name. Used for naming the working directory.
    `servers`               ''list''  A list of servers used for the present project
    `species_list`          ``list``  Contains input ``ARCSpecies`` objects (both species and TSs)
    `species_dict`          ``dict``  Keys are labels, values are ARCSpecies objects
    `rxn_list`              ``list``  Contains input ``ARCReaction`` objects
    `unique_species_labels` ``list``  A list of species labels (checked for duplicates)
    `level_of_theory`       ``str``   *FULL* level of theory, e.g. 'CBS-QB3',
                                        'CCSD(T)-F12a/aug-cc-pVTZ//B3LYP/6-311++G(3df,3pd)'...
    `adaptive_levels`       ``dict``  A dictionary of levels of theory for ranges of the number of heavy atoms in the
                                        molecule. Keys are tuples of (min_num_atoms, max_num_atoms), values are
                                        dictionaries with 'optfreq' and 'sp' as keys and levels of theory as values.
    `composite`             ``bool``    Whether level_of_theory represents a composite method or not
    `job_dict`              ``dict``  A dictionary of all scheduled jobs. Keys are species / TS labels,
                                        values are dictionaries where keys are job names (corresponding to
                                        'running_jobs' if job is running) and values are the Job objects.
    `running_jobs`          ``dict``  A dictionary of currently running jobs (a subset of `job_dict`).
                                        Keys are species/TS label, values are lists of job names
                                        (e.g. 'conformer3', 'opt_a123').
    `servers_jobs_ids`      ``list``  A list of relevant job IDs currently running on the server
    `output`                ``dict``  Output dictionary with status and final QM file paths for all species
    `ess_settings`          ``dict``  A dictionary of available ESS and a correcponding server list
    `initial_trsh`          ``dict``  Troubleshooting methods to try by default. Keys are ESS software, values are trshs
    `restart_dict`          ``dict``  A restart dictionary parsed from a YAML restart file
    `project_directory`     ``str``   Folder path for the project: the input file path or ARC/Projects/project-name
    `save_restart`          ``bool``  Whether to start saving a restart file. ``True`` only after all species are loaded
                                        (otherwise saves a partial file and may cause loss of information)
    `restart_path`          ``str``   Path to the `restart.yml` file to be saved
    `max_job_time`          ``int``   The maximal allowed job time on the server in hours
    `testing`               ``bool``  Used for internal ARC testing (generating the object w/o executing it)
    `rmgdb`                 ``RMGDatabase``  The RMG database object
    `allow_nonisomorphic_2d` ``bool`` Whether to optimize species even if they do not have a 3D conformer that is
                                        isomorphic to the 2D graph representation
    `dont_gen_confs`        ``list``  A list of species labels for which conformer jobs were loaded from a restart file,
                                        and additional conformer generation should be avoided
    `confs_to_dft`          ``int``   The number of lowest MD conformers to DFT at the conformers_level.
    `memory`                ``int``   The total allocated job memory in GB (14 by default)
    `job_types`             ``dict``  A dictionary of job types to execute. Keys are job types, values are boolean
    `bath_gas`              ``str``   A bath gas. Currently used in OneDMin to calc L-J parameters.
                                        Allowed values are He, Ne, Ar, Kr, H2, N2, O2
    ======================= ========= ==================================================================================

    Dictionary structures:

*   job_dict = {label_1: {'conformers': {0: Job1,
                                         1: Job2, ...},  # TS guesses are considered `conformers` as well
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
                          ...
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
    def __init__(self, project, ess_settings, species_list, project_directory, composite_method='', conformer_level='',
                 opt_level='', freq_level='', sp_level='', scan_level='', ts_guess_level='', orbitals_level='',
                 adaptive_levels=None, rmgdatabase=None, job_types=None, initial_trsh=None, rxn_list=None, bath_gas=None,
                 restart_dict=None, max_job_time=120, allow_nonisomorphic_2d=False, memory=14, testing=False,
                 dont_gen_confs=None, confs_to_dft=5):
        self.rmgdb = rmgdatabase
        self.restart_dict = restart_dict
        self.species_list = species_list
        self.rxn_list = rxn_list if rxn_list is not None else list()
        self.project = project
        self.max_job_time = max_job_time
        self.ess_settings = ess_settings
        self.project_directory = project_directory
        self.job_dict = dict()
        self.servers_jobs_ids = list()
        self.running_jobs = dict()
        self.allow_nonisomorphic_2d = allow_nonisomorphic_2d
        self.testing = testing
        self.memory = memory
        self.bath_gas = bath_gas
        self.adaptive_levels = adaptive_levels
        self.confs_to_dft = confs_to_dft
        self.dont_gen_confs = dont_gen_confs or list()

        self.species_dict = dict()
        for species in self.species_list:
            self.species_dict[species.label] = species
        if self.restart_dict is not None:
            self.output = self.restart_dict['output']
            if 'running_jobs' in self.restart_dict:
                self.restore_running_jobs()
        else:
            self.output = dict()

        self.restart_path = os.path.join(self.project_directory, 'restart.yml')
        self.report_time = time.time()  # init time for reporting status every 1 hr
        self.servers = list()
        self.composite_method = composite_method
        self.conformer_level = conformer_level
        self.ts_guess_level = ts_guess_level
        self.opt_level = opt_level
        self.freq_level = freq_level
        self.sp_level = sp_level
        self.scan_level = scan_level
        self.orbitals_level = orbitals_level
        self.job_types = job_types if job_types is not None else default_job_types
        self.unique_species_labels = list()
        self.initial_trsh = initial_trsh if initial_trsh is not None else dict()
        self.save_restart = False

        if len(self.rxn_list):
            rxn_info_path = self.make_reaction_labels_info_file()
            logger.info("\nLoading RMG's families...")
            rmgdb.load_families_only(self.rmgdb)
            for rxn in self.rxn_list:
                logger.info('\n\n')
                # update the ARCReaction object and generate an ARCSpecies object for its TS
                rxn.r_species, rxn.p_species = list(), list()
                for spc in self.species_list:
                    if spc.label in rxn.reactants:
                        rxn.r_species.append(spc)
                    elif spc.label in rxn.products:
                        rxn.p_species.append(spc)
                rxn.rmg_reaction_from_arc_species()
                rxn.check_attributes()
                rxn.determine_family(self.rmgdb)
                family_text = ''
                if rxn.family is not None:
                    family_text = 'identified as belonging to RMG family {0}'.format(rxn.family.label)
                    if rxn.family_own_reverse:
                        family_text += ", which is its own reverse"
                logger.info('Considering reaction: {0}'.format(rxn.label))
                if family_text:
                    logger.info('({0})'.format(family_text))
                if rxn.rmg_reaction is not None:
                    display(rxn.rmg_reaction)
                rxn.determine_rxn_charge()
                rxn.determine_rxn_multiplicity()
                rxn.ts_label = rxn.ts_label if rxn.ts_label is not None else 'TS{0}'.format(rxn.index)
                with open(rxn_info_path, 'a') as f:
                    f.write(str('{0}: {1}'.format(rxn.ts_label, rxn.label)))
                    if family_text:
                        family_text = '\n(' + family_text + ')'
                        f.write(str(family_text))
                    f.write(str('\n\n'))
                if len(rxn.ts_xyz_guess) == 1 and 'user guess' not in rxn.ts_methods:
                    rxn.ts_methods.append('user guess')
                elif len(rxn.ts_xyz_guess) > 1 and all(['user guess' not in method for method in rxn.ts_methods]):
                    rxn.ts_methods.append('{0} user guesses'.format(len(rxn.ts_xyz_guess)))
                auto_tst = False
                reverse_auto_tst = False
                for method in rxn.ts_methods:
                    if method == 'autotst':
                        auto_tst = True
                    elif method == 'reverse_autotst':
                        reverse_auto_tst = True
                if rxn.family_own_reverse and auto_tst and not reverse_auto_tst:
                    rxn.ts_methods.append('reverse_autotst')
                if not any([spc.label == rxn.ts_label for spc in self.species_list]):
                    ts_species = ARCSpecies(is_ts=True, label=rxn.ts_label, rxn_label=rxn.label,
                                            multiplicity=rxn.multiplicity, charge=rxn.charge, generate_thermo=False,
                                            ts_methods=rxn.ts_methods, ts_number=rxn.index)
                    ts_species.number_of_atoms = sum(reactant.number_of_atoms for reactant in rxn.r_species)
                    self.species_list.append(ts_species)
                else:
                    # The TS species was already loaded from a restart dict or an Arkane YAML file
                    for spc in self.species_list:
                        if spc.label == rxn.ts_label:
                            ts_species = spc
                            break
                    if ts_species.rxn_label is None:
                        ts_species.rxn_label = rxn.label
                rxn.ts_species = ts_species
                # Generate TSGuess objects for all methods, start with the user guesses
                for i, user_guess in enumerate(rxn.ts_xyz_guess):  # this is a list of guesses, could be empty
                    ts_species.ts_guesses.append(TSGuess(method='user guess {0}'.format(i), xyz=user_guess,
                                                         rmg_reaction=rxn.rmg_reaction))
                for tsm in rxn.ts_methods:
                    # loop through all ts methods of this reaction, generate a TSGuess object if not a user guess
                    if 'user guess' not in tsm:
                        rmg_reaction = rxn.rmg_reaction
                        if tsm == 'reverse_autotst':
                            rmg_reaction = Reaction(reactants=rxn.rmg_reaction.products,
                                                    products=rxn.rmg_reaction.reactants)
                        family = rxn.family.label if rxn.family is not None else None
                        ts_species.ts_guesses.append(TSGuess(method=tsm, family=family, rmg_reaction=rmg_reaction))
                for ts_guess in ts_species.ts_guesses:
                    # Execute the TS guess methods that don't require optimized reactants and products
                    if 'autotst' in ts_guess.method and ts_guess.xyz is None:
                        reverse = ' in the reverse direction' if 'reverse' in ts_guess.method else ''
                        logger.info('Trying to generating a TS guess for {0} reaction {1} using AutoTST{2}...'.format(
                            ts_guess.family, rxn.label, reverse))
                        ts_guess.t0 = datetime.datetime.now()
                        try:
                            ts_guess.xyz = autotst(rmg_reaction=ts_guess.rmg_reaction, reaction_family=ts_guess.family)
                        except TSError as e:
                            logger.error('Could not generate an AutoTST guess for reaction {0}.\nGot: {1}'.format(
                                rxn.label, e.message))
                        ts_guess.success = True if ts_guess.xyz is not None else False
                        ts_guess.execution_time = str(datetime.datetime.now() - ts_guess.t0).split('.')[0]
                    else:
                        # spawn other methods as needed when they are implemented (job_type = 'ts_guess');
                        # add to job_dict only spawn if `ts_guess.xyz is None` (restart)
                        pass

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
                self.output[species.label]['status'] += '; Restarted ARC at {0}; '.format(datetime.datetime.now())
            if species.label not in self.job_dict:
                self.job_dict[species.label] = dict()
            if species.yml_path is None:
                if self.job_types['1d_rotors'] and not self.species_dict[species.label].number_of_rotors:
                    self.species_dict[species.label].determine_rotors()
                if not self.job_types['opt'] and self.species_dict[species.label].final_xyz is not None:
                    self.output[species.label]['status'] += 'opt converged; '
                if species.label not in self.running_jobs:
                    self.running_jobs[species.label] = list()  # initialize before running the first job
                if not species.is_ts and species.number_of_atoms == 1:
                    logger.debug('Species {0} is monoatomic'.format(species.label))
                    if not self.species_dict[species.label].initial_xyz:
                        # generate a simple "Symbol   0.0   0.0   0.0" xyz matrix
                        if self.species_dict[species.label].mol is not None:
                            symbol = self.species_dict[species.label].mol.atoms[0].symbol
                        else:
                            symbol = species.label
                            logger.warning('Could not determine element of monoatomic species {0}.'
                                           ' Assuming element is {1}'.format(species.label, symbol))
                        self.species_dict[species.label].initial_xyz = symbol + '   0.0   0.0   0.0'
                        self.species_dict[species.label].final_xyz = symbol + '   0.0   0.0   0.0'
                    if 'sp' not in self.output[species.label] and 'composite' not in self.output[species.label] \
                            and 'sp' not in self.job_dict[species.label]\
                            and 'composite' not in self.job_dict[species.label]:
                        # No need to run opt/freq jobs for a monoatomic species other than sp (or composite if relevant)
                        if self.composite_method:
                            self.run_composite_job(species.label)
                        else:
                            self.run_sp_job(label=species.label)
                        if self.job_types['onedmin']:
                            self.run_onedmin_job(species.label)
                elif (self.species_dict[species.label].initial_xyz is not None
                        or self.species_dict[species.label].final_xyz is not None) and not self.testing:
                    # For restarting purposes: check before running jobs whether they were already terminated
                    # (check self.output) or whether they are "currently running" (check self.job_dict)
                    # This section takes care of restarting a Species (including a TS), but does not
                    # deal with conformers nor with ts_guesses
                    if self.composite_method and 'composite' not in self.output[species.label]\
                            and 'composite' not in self.job_dict[species.label]:
                        self.run_composite_job(species.label)
                    elif self.composite_method and 'freq' not in self.output[species.label]\
                            and 'freq' not in self.job_dict[species.label]\
                            and 'composite' not in self.job_dict[species.label]:
                        self.run_freq_job(species.label)
                    elif 'opt converged' not in self.output[species.label]['status']\
                            and 'opt' not in self.job_dict[species.label] and not self.composite_method \
                            and 'geo' not in self.output[species.label]:
                        self.run_opt_job(species.label)
                    elif 'opt converged' in self.output[species.label]['status'] and not self.composite_method:
                        # opt is done
                        if 'freq' not in self.output[species.label] and 'freq' not in self.job_dict[species.label]:
                            if self.species_dict[species.label].is_ts\
                                    or self.species_dict[species.label].number_of_atoms > 1:
                                self.run_freq_job(species.label)
                        if 'sp' not in self.output[species.label] and 'sp' not in self.job_dict[species.label]:
                            self.run_sp_job(species.label)
                        if self.job_types['1d_rotors']:
                            # restart-related checks are performed in run_scan_jobs()
                            self.run_scan_jobs(species.label)
            else:
                # Species is loaded from an Arkane YAML file (no need to execute any job)
                self.output[species.label]['status'] = 'ALL converged'
                if species.is_ts:
                    # This is a TS loaded from a YAML file
                    species.ts_conf_spawned = True
        self.save_restart = True
        self.timer = True
        if not self.testing:
            self.schedule_jobs()

    def schedule_jobs(self):
        """
        The main job scheduling block
        """
        for species in self.species_dict.values():
            if species.initial_xyz is None and species.final_xyz is None and species.conformers\
                    and any([e is not None for e in species.conformer_energies]):
                # the species has no xyz, but has conformers and at least one of the conformers has energy
                self.determine_most_stable_conformer(species.label)
                if species.initial_xyz is not None:
                    if self.composite_method:
                        self.run_composite_job(species.label)
                    else:
                        self.run_opt_job(species.label)
        self.run_conformer_jobs()
        while self.running_jobs != {}:  # loop while jobs are still running
            logger.debug('Currently running jobs:\n{0}'.format(self.running_jobs))
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
                        if self.job_dict[label]['conformers'][i].job_id not in self.servers_jobs_ids:
                            # this is a completed conformer job
                            successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                            if successful_server_termination:
                                self.parse_conformer(job=job, label=label, i=i)
                            # Just terminated a conformer job.
                            # Are there additional conformer jobs currently running for this species?
                            for spec_jobs in job_list:
                                if 'conformer' in spec_jobs:
                                    break
                            else:
                                # All conformer jobs terminated.
                                # Check isomorphism and run opt on most stable conformer geometry.
                                logger.info('\nConformer jobs for {0} successfully terminated.\n'.format(
                                    label))
                                if self.species_dict[label].is_ts:
                                    self.determine_most_likely_ts_conformer(label)
                                else:
                                    self.determine_most_stable_conformer(label)  # also checks isomorphism
                                if self.species_dict[label].initial_xyz is not None:
                                    # if initial_xyz is None, then we're probably troubleshooting conformers, don't opt
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
                                    if self.species_dict[label].is_ts\
                                            or self.species_dict[label].number_of_atoms > 1:
                                        if 'freq' not in job_name:
                                            self.run_freq_job(label)
                                        else:  # this is an 'optfreq' job type
                                            self.check_freq_job(label=label, job=job)
                                    self.run_sp_job(label)
                                    self.run_scan_jobs(label)
                                    self.run_orbitals_job(label)
                                    if self.job_types['onedmin'] and not self.species_dict[label].is_ts:
                                        self.run_onedmin_job(label)
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
                                    if self.species_dict[label].is_ts\
                                            or self.species_dict[label].number_of_atoms > 1:
                                        self.run_freq_job(label)
                                    self.run_scan_jobs(label)
                                    if self.job_types['onedmin'] and not self.species_dict[label].is_ts\
                                            and self.composite_method:
                                        self.run_onedmin_job(label)
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
                    elif 'orbitals' in job_name\
                            and not self.job_dict[label]['orbitals'][job_name].job_id in self.servers_jobs_ids:
                        job = self.job_dict[label]['orbitals'][job_name]
                        successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                        if successful_server_termination:
                            # copy the orbitals file to the species / TS output folder
                            folder_name = 'rxns' if self.species_dict[label].is_ts else 'Species'
                            orbitals_path = os.path.join(self.project_directory, 'output', folder_name, label,
                                                         'geometry', 'orbitals.fchk')
                            if os.path.isfile(job.local_path_to_orbitals_file):
                                shutil.copyfile(job.local_path_to_orbitals_file, orbitals_path)
                        self.timer = False
                        break
                    elif 'onedmin' in job_name\
                            and not self.job_dict[label]['onedmin'][job_name].job_id in self.servers_jobs_ids:
                        job = self.job_dict[label]['onedmin'][job_name]
                        successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                        if successful_server_termination:
                            # copy the lennard_jones file to the species output folder (TS's don't have L-J data)
                            lj_output_path = os.path.join(self.project_directory, 'output', 'Species', label,
                                                          'lennard_jones.dat')
                            if os.path.isfile(job.local_path_to_lj_file):
                                shutil.copyfile(job.local_path_to_lj_file, lj_output_path)
                                self.output[label]['status'] += 'OneDMin converged; '
                                self.species_dict[label].set_transport_data(
                                    lj_path=os.path.join(self.project_directory, 'output', 'Species', label,
                                                         'lennard_jones.dat'),
                                    opt_path=self.output[label]['geo'], bath_gas=job.bath_gas, opt_level=self.opt_level)
                        self.timer = False
                        break
                    elif 'ff_param_fit' in job_name\
                            and not self.job_dict[label]['ff_param_fit'][job_name].job_id in self.servers_jobs_ids:
                        job = self.job_dict[label]['ff_param_fit'][job_name]
                        successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                        mmff94_fallback = False
                        if successful_server_termination and job.job_status[1] == 'done':
                            # copy the fitting file to the species output folder
                            ff_param_fit_path = os.path.join(self.project_directory, 'calcs', 'Species', label,
                                                             'ff_param_fit')
                            if not os.path.isdir(ff_param_fit_path):
                                os.makedirs(ff_param_fit_path)
                            ff_param_fit_path = os.path.join(ff_param_fit_path, 'gaussian.out')
                            if os.path.isfile(job.local_path_to_output_file):
                                shutil.copyfile(job.local_path_to_output_file, ff_param_fit_path)
                                self.output[label]['status'] += 'ff_param_fit converged; '
                                self.spawn_md_jobs(label)
                            else:
                                mmff94_fallback = True
                        else:
                            mmff94_fallback = True
                        if mmff94_fallback:
                            logger.error('Force field parameter fitting job in Gaussian failed. Generating standard '
                                         'MMFF94 conformers instead of fitting a force field for species {0}, although '
                                         'its force_field attribute was set to "fit".'.format(label))
                            self.species_dict[label].force_field = 'MMFF94'
                            self.species_dict[label].generate_conformers(confs_to_dft=self.confs_to_dft,
                                                                         plot_path=os.path.join(self.project_directory,
                                                                                                'output', 'Species',
                                                                                                label, 'geometry',
                                                                                                'conformers'))
                            self.process_conformers(label)
                        self.timer = False
                        break
                    elif 'gromacs' in job_name\
                            and not self.job_dict[label]['gromacs'][job_name].job_id in self.servers_jobs_ids:
                        job = self.job_dict[label]['gromacs'][job_name]
                        successful_server_termination = self.end_job(job=job, label=label, job_name=job_name)
                        if successful_server_termination:
                            self.check_md_job(label=label, job=job)
                        self.timer = False
                        break

                if self.species_dict[label].is_ts and not self.species_dict[label].ts_conf_spawned\
                        and not any([tsg.success is None for tsg in self.species_dict[label].ts_guesses]):
                    # This is a TS Species for which conformers haven't been spawned, and all .success flags
                    # contain a values (whether ``True`` or ``False``)
                    # We're ready to spawn conformers for this TS Species
                    self.species_dict[label].generate_conformers()
                    self.run_ts_conformer_jobs(label=label)
                    self.species_dict[label].ts_conf_spawned = True

                if not job_list and not(self.species_dict[label].is_ts
                                        and not self.species_dict[label].ts_conf_spawned):
                    self.check_all_done(label)
                    if not self.running_jobs[label]:
                        # delete the label only if it represents an empty dictionary
                        del self.running_jobs[label]

            if self.timer and job_list:
                time.sleep(30)  # wait 30 sec before bugging the servers again.
            t = time.time() - self.report_time
            if t > 3600:
                self.report_time = time.time()
                logger.info('Currently running jobs:\n{0}'.format(self.running_jobs))

        # After exiting the Scheduler while loop, append all YAML species not directly calculated to the species_dict:
        for spc in self.species_list:
            if spc.yml_path is not None:
                self.species_dict[spc.label] = spc

    def run_job(self, label, xyz, level_of_theory, job_type, fine=False, software=None, shift='', trsh='', memory=None,
                conformer=-1, ess_trsh_methods=None, scan='', pivots=None, occ=None, scan_trsh='', scan_res=None,
                max_job_time=None, confs=None, radius=None):
        """
        A helper function for running (all) jobs
        """
        max_job_time = max_job_time or self.max_job_time  # if it's None, set to default
        ess_trsh_methods = ess_trsh_methods if ess_trsh_methods is not None else list()
        pivots = pivots if pivots is not None else list()
        species = self.species_dict[label]
        memory = memory if memory is not None else self.memory
        checkfile = self.species_dict[label].checkfile  # defaults to None
        if self.species_dict[label].most_stable_conformer is not None and job_type in ['opt', 'composite', 'optfreq'] \
                and self.species_dict[label].conformer_checkfiles:
            checkfile = self.species_dict[label].conformer_checkfiles[self.species_dict[label].most_stable_conformer]
        if self.adaptive_levels is not None:
            level_of_theory = self.determine_adaptive_level(original_level_of_theory=level_of_theory, job_type=job_type,
                                                            heavy_atoms=self.species_dict[label].number_of_heavy_atoms)
        job = Job(project=self.project, ess_settings=self.ess_settings, species_name=label, xyz=xyz, job_type=job_type,
                  level_of_theory=level_of_theory, multiplicity=species.multiplicity, charge=species.charge, fine=fine,
                  shift=shift, software=software, is_ts=species.is_ts, memory=memory, trsh=trsh,
                  ess_trsh_methods=ess_trsh_methods, scan=scan, pivots=pivots, occ=occ, initial_trsh=self.initial_trsh,
                  project_directory=self.project_directory, max_job_time=max_job_time, scan_trsh=scan_trsh,
                  scan_res=scan_res, conformer=conformer, checkfile=checkfile, bath_gas=self.bath_gas,
                  number_of_radicals=species.number_of_radicals, conformers=confs, radius=radius)
        if job.software is not None:
            if conformer < 0:
                # this is NOT a conformer DFT job
                self.running_jobs[label].append(job.job_name)  # mark as a running job
                if job_type not in self.job_dict[label]:
                    # Jobs of this type haven't been spawned for label
                    self.job_dict[label][job_type] = dict()
                self.job_dict[label][job_type][job.job_name] = job
                self.job_dict[label][job_type][job.job_name].run()
            else:
                # Running a conformer DFT job. Append differently to job_dict.
                self.running_jobs[label].append('conformer{0}'.format(conformer))  # mark as a running job
                self.job_dict[label]['conformers'][conformer] = job  # save job object
                self.job_dict[label]['conformers'][conformer].run()  # run the job
            self.save_restart_dict()
            if job.server not in self.servers:
                self.servers.append(job.server)

    def end_job(self, job, label, job_name):
        """
        A helper function for checking job status, saving in csv file, and downloading output files.
        Returns ``True`` if job terminated successfully on the server, ``False`` otherwise
        """
        try:
            job.determine_job_status()  # also downloads output file
        except IOError:
            if job.job_type not in ['orbitals']:
                logger.warning('Tried to determine status of job {0}, but it seems like the job never ran.'
                               ' Re-running job.'.format(job.job_name))
                self.run_job(label=label, xyz=job.xyz, level_of_theory=job.level_of_theory, job_type=job.job_type,
                             fine=job.fine, software=job.software, shift=job.shift, trsh=job.trsh, memory=job.memory_gb,
                             conformer=job.conformer, ess_trsh_methods=job.ess_trsh_methods, scan=job.scan,
                             pivots=job.pivots, occ=job.occ, scan_trsh=job.scan_trsh, scan_res=job.scan_res,
                             max_job_time=job.max_job_time)
            if job_name in self.running_jobs[label]:
                self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
        if job.job_status[0] != 'running' and job.job_status[1] != 'running':
            if job_name in self.running_jobs[label]:
                self.running_jobs[label].pop(self.running_jobs[label].index(job_name))
            self.timer = False
            job.write_completed_job_to_csv_file()
            logger.info('  Ending job {name} for {label} (run time: {time})'.format(name=job.job_name, label=label,
                                                                                    time=job.run_time))
            if job.job_status[0] != 'done':
                return False
            self.save_restart_dict()
            if job.software.lower() == 'gaussian' and os.path.isfile(os.path.join(job.local_path, 'check.chk'))\
                    and job.job_type in ['conformer', 'opt', 'optfreq', 'composite']:
                check_path = os.path.join(job.local_path, 'check.chk')
                if os.path.isfile(check_path):
                    if job.job_type == 'conformer':
                        self.species_dict[label].conformer_checkfiles[job.conformer] = check_path
                    else:
                        self.species_dict[label].checkfile = check_path
            return True

    def run_conformer_jobs(self):
        """
        Select the most stable conformer for each species using molecular dynamics (force fields) and subsequently
        spawning opt jobs at the conformer level of theory, usually a reasonable yet cheap DFT, e.g., b97d3/6-31+g(d,p).
        The resulting conformer is saved in a string format xyz in the Species initial_xyz attribute.
        """
        log_info_printed = False
        for label in self.unique_species_labels:
            if not self.species_dict[label].is_ts and 'opt converged' not in self.output[label]['status'] \
                    and 'opt' not in self.job_dict[label] and 'composite' not in self.job_dict[label] \
                    and all([e is None for e in self.species_dict[label].conformer_energies]) \
                    and self.species_dict[label].number_of_atoms > 1 and label not in self.dont_gen_confs \
                    and 'geo' not in self.output[label] \
                    and (self.job_types['conformers'] or (not self.job_types['conformers']
                                                          and self.species_dict[label].initial_xyz is None
                                                          and self.species_dict[label].final_xyz is None
                                                          and not self.species_dict[label].conformers)):
                # This is not a TS, opt (/composite) did not converged nor running, and conformer energies were not set
                # (and it's not in self.dont_gen_confs). Also, either 'conformers' are set to True in job_types,
                # or they are set to False but the species has no 3D information. Generate conformers.
                if not log_info_printed:
                    logger.info('\nStarting (non-TS) species conformational analysis...\n')
                    log_info_printed = True
                if self.species_dict[label].force_field == 'fit':
                    # first run a Gaussian blyp/svp/svpfit job for force field parameters fitting
                    if self.species_dict[label].cheap_conformer is None:
                        self.species_dict[label].get_cheap_conformer()
                    self.run_force_field_fit_job(label)
                else:
                    if self.species_dict[label].force_field == 'cheap':
                        # just embed in RDKit and use MMFF94 for opt and energies
                        if self.species_dict[label].initial_xyz is None:
                            self.species_dict[label].initial_xyz = self.species_dict[label].get_xyz()
                    else:
                        # run the combinatorial method w/o fitting a force field
                        self.species_dict[label].generate_conformers(confs_to_dft=self.confs_to_dft,
                                                                     plot_path=os.path.join(self.project_directory,
                                                                                            'output', 'Species',
                                                                                            label, 'geometry',
                                                                                            'conformers'))
                    self.process_conformers(label)
            elif not self.job_types['conformers']:
                # we're not running conformer jobs
                if self.species_dict[label].initial_xyz is not None or self.species_dict[label].final_xyz is not None:
                    pass
                elif self.species_dict[label].conformers:
                    # the species was defined with xyz's
                    self.process_conformers(label)

    def run_ts_conformer_jobs(self, label):
        """
        Spawn opt jobs at the ts_guesses level of theory for the TS guesses
        """
        plotter.save_conformers_file(project_directory=self.project_directory, label=label,
                                     xyzs=[tsg.xyz for tsg in self.species_dict[label].ts_guesses],
                                     level_of_theory=self.ts_guess_level,
                                     multiplicity=self.species_dict[label].multiplicity,
                                     charge=self.species_dict[label].charge, is_ts=True,
                                     ts_methods=[tsg.method for tsg in self.species_dict[label].ts_guesses])
        self.job_dict[label]['conformers'] = dict()
        for i, tsg in enumerate(self.species_dict[label].ts_guesses):
            self.run_job(label=label, xyz=tsg.xyz, level_of_theory=self.ts_guess_level, job_type='conformer',
                         conformer=i)

    def run_opt_job(self, label):
        """
        Spawn a geometry optimization job. The initial guess is taken from the `initial_xyz` attribute.
        """
        if 'opt' not in self.job_dict[label]:  # Check whether or not opt jobs have been spawned yet
            # we're spawning the first opt job for this species
            self.job_dict[label]['opt'] = dict()
        if self.species_dict[label].initial_xyz is None:
            raise SpeciesError('Cannot execute opt job for {0} without xyz (got None for Species.initial_xyz)'.format(
                label))
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
        if self.species_dict[label].final_xyz is not None:
            xyz = self.species_dict[label].final_xyz
        else:
            xyz = self.species_dict[label].initial_xyz
        self.run_job(label=label, xyz=xyz, level_of_theory=self.composite_method, job_type='composite',
                     fine=self.job_types['fine'])

    def run_freq_job(self, label):
        """
        Spawn a freq job using 'final_xyz' for species ot TS 'label'.
        If this was originally a composite job, run an appropriate separate freq job outputting the Hessian.
        """
        if 'freq' not in self.job_dict[label]:  # Check whether or not freq jobs have been spawned yet
            # we're spawning the first freq job for this species
            self.job_dict[label]['freq'] = dict()
        if self.job_types['freq']:
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
                logger.info('running a CCSD job for {0} before MRCI'.format(label))
                self.run_job(label=label, xyz=self.species_dict[label].final_xyz, level_of_theory='ccsd/vdz',
                             job_type='sp')
        if self.job_types['sp']:
            xyz = self.species_dict[label].final_xyz or self.species_dict[label].initial_xyz
            if xyz is None:
                xyz = self.species_dict[label].conformers[0]
            self.run_job(label=label, xyz=xyz, level_of_theory=self.sp_level, job_type='sp')

    def run_scan_jobs(self, label):
        """
        Spawn rotor scan jobs using 'final_xyz' for species or TS 'label'.
        """
        if self.job_types['1d_rotors']:
            if 'scan' not in self.job_dict[label]:  # Check whether or not rotor scan jobs have been spawned yet
                # we're spawning the first scan job for this species
                self.job_dict[label]['scan'] = dict()
            for i in range(self.species_dict[label].number_of_rotors):
                scan = self.species_dict[label].rotors_dict[i]['scan']
                pivots = self.species_dict[label].rotors_dict[i]['pivots']
                if 'scan_path' not in self.species_dict[label].rotors_dict[i]\
                        or not self.species_dict[label].rotors_dict[i]['scan_path']:
                    # check this job isn't already running on the server or completed (from a restarted project):
                    for scan_job in self.job_dict[label]['scan'].values():
                        if scan_job.pivots == pivots and scan_job.job_name in self.running_jobs[label]:
                            break
                    else:
                        self.run_job(label=label, xyz=self.species_dict[label].final_xyz,
                                     level_of_theory=self.scan_level, job_type='scan', scan=scan, pivots=pivots)

    def run_orbitals_job(self, label):
        """
        Spawn orbitals job used for molecular orbital visualization
        Currently supporting QChem for printing the orbitals, the output could be visualized using IQMol
        """
        if self.job_types['orbitals'] and 'orbitals' not in self.job_dict[label]:
            self.run_job(label=label, xyz=self.species_dict[label].final_xyz, level_of_theory=self.orbitals_level,
                         job_type='orbitals')

    def run_onedmin_job(self, label):
        """
        Spawn a lennard-jones calculation using OneDMin
        """
        if 'onedmin' not in self.ess_settings:
            logger.error('Cannot execute a Lennard Jones job without the OneDMin software')
        elif 'onedmin' not in self.job_dict[label]:
            self.run_job(label=label, xyz=self.species_dict[label].final_xyz, job_type='onedmin',
                         level_of_theory='')

    def run_force_field_fit_job(self, label):
        """
        Spawn a force field parameter fitting job (currently only Gaussian is supported for this task).
        """
        if self.species_dict[label].svpfit_output_file is not None\
                and os.path.isfile(self.species_dict[label].svpfit_output_file):
            # a force field parameter fit job was already spawned, use this file
            ff_param_fit_path = os.path.join(self.project_directory, 'calcs', 'Species', label, 'ff_param_fit')
            if not os.path.isdir(ff_param_fit_path):
                os.makedirs(ff_param_fit_path)
            ff_param_fit_path = os.path.join(ff_param_fit_path, 'gaussian.out')
            shutil.copyfile(self.species_dict[label].svpfit_output_file, ff_param_fit_path)
            self.output[label]['status'] += 'ff_param_fit converged; '
            self.spawn_md_jobs(label)
        elif 'gaussian' not in self.ess_settings:
            logger.error('Cannot execute a force field parameter fitting job in Gaussian. Gaussian  is missing from '
                         'the ess_settings dictionary. Generating standard MMFF94 conformers instead for '
                         'species {0}, although its force_field attribute was set to "fit".'.format(label))
            self.species_dict[label].force_field = 'MMFF94'
            self.species_dict[label].generate_conformers(confs_to_dft=self.confs_to_dft,
                                                         plot_path=os.path.join(self.project_directory, 'output',
                                                                                'Species', label, 'geometry',
                                                                                'conformers'))
            self.process_conformers(label)
        else:
            if 'ff_param_fit' not in self.job_dict[label]:
                self.run_job(label=label, xyz=self.species_dict[label].get_xyz(), job_type='ff_param_fit',
                             level_of_theory='blyp/svp/svpfit')

    def run_gromacs_job(self, label, confs):
        """
        Run a Gromacs MD job.

        Args:
            label (str, unicode): The species label.
            confs (str, unicode): The path to a YAML file with array-format coordinates to optimize.
        """
        if 'gromacs' not in self.ess_settings:
            logger.error('Cannot execute a Gromacs MD job without the Gromacs software')
        else:
            self.run_job(label=label, xyz=None, job_type='gromacs', level_of_theory='', confs=confs,
                         radius=self.species_dict[label].radius)

    def spawn_md_jobs(self, label, prev_conf_list=None, num_confs=None):
        """
        Embed conformers and run a molecular dynamics optimization using a fitted force field.
        Then generate conformers using combinations of the detected torsion wells, and re-run until converging.

        Args:
            label (str, unicode): The species label.
            prev_conf_list (list, optional): The previous conformers (entries are two length lists, not dicts).
                                             If not given, a first Gromacs job will be spawned.
            num_confs (int, optional): The number of conformers to generate.
        """
        if not self.species_dict[label].rotors_dict:
            self.species_dict[label].determine_rotors()
        torsions, tops = list(), list()
        for rotor_dict in self.species_dict[label].rotors_dict.values():
            torsions.append(rotor_dict['scan'])
            tops.append(rotor_dict['top'])

        if prev_conf_list is None:
            # first time spawning MD jobs for this species
            if self.species_dict[label].mol_list is not None:
                self.species_dict[label].mol_list = [conformers.update_mol(mol)
                                                     for mol in self.species_dict[label].mol_list]
                number_of_heavy_atoms = len([atom for atom in self.species_dict[label].mol_list[0].atoms
                                             if atom.isNonHydrogen()])
            else:
                xyz = self.species_dict[label].get_xyz()
                number_of_heavy_atoms = 0
                for line in xyz.splitlines():
                    if line and line.split()[0] != 'H':
                        number_of_heavy_atoms += 1
            num_confs = num_confs\
                or conformers.determine_number_of_conformers_to_generate(heavy_atoms=number_of_heavy_atoms,
                                                                         torsion_num=len(torsions), label=label)
            coords = list()
            for mol in self.species_dict[label].mol_list:
                # embed conformers (but don't optimize)
                rd_mol, rd_index_map = conformers.embed_rdkit(label=label, mol=mol, num_confs=num_confs, xyz=None)
                for i in range(rd_mol.GetNumConformers()):
                    conf, coord = rd_mol.GetConformer(i), list()
                    for j in range(conf.GetNumAtoms()):
                        pt = conf.GetAtomPosition(j)
                        coord.append([pt.x, pt.y, pt.z])
                    if rd_index_map is not None:
                        coord = [coord[rd_index_map[j]] for j, _ in enumerate(coord)]  # reorder
                    coords.append(coord)
            embedded_confs_path = os.path.join(self.project_directory, 'calcs', 'Species', label,
                                               'ff_param_fit', 'embedded_conformers.yml')  # list of lists
            save_yaml_file(path=embedded_confs_path, content=coords)
            self.run_gromacs_job(label, confs=embedded_confs_path)
        else:
            # a previous Gromacs job was submitted, generate specific conformers via deduce_new_conformers()
            confs = list()
            confs_path = os.path.join(self.project_directory, 'calcs', 'Species', label,
                                      'ff_param_fit', 'conformers.yml')  # list of dicts
            if os.path.isfile(confs_path):
                confs = read_yaml_file(confs_path)
            for prev_conf in prev_conf_list:
                confs.append({'xyz': prev_conf[0], 'FF energy': prev_conf[1], 'source': 'Gromacs', 'index': len(confs)})
            save_yaml_file(path=confs_path, content=confs)  # save for the next iteration and for archiving

            confs = conformers.determine_dihedrals(confs, torsions)
            new_conformers, _ = conformers.deduce_new_conformers(label=label, conformers=confs, torsions=torsions,
                                                                 tops=tops, mol_list=self.species_dict[label].mol_list,
                                                                 plot_path=False)
            new_confs_path = os.path.join(self.project_directory, 'calcs', 'Species', label,
                                          'ff_param_fit', 'new_conformers.yml')  # list of lists
            coords = [new_conf['xyz'] for new_conf in new_conformers]
            save_yaml_file(path=new_confs_path, content=coords)
            self.run_gromacs_job(label, confs=new_confs_path)

    def process_conformers(self, label):
        """
        Process the generated conformers and spawn DFT jobs at the conformer_level.
        """
        plotter.save_conformers_file(project_directory=self.project_directory, label=label,
                                     xyzs=self.species_dict[label].conformers, level_of_theory=self.conformer_level,
                                     multiplicity=self.species_dict[label].multiplicity,
                                     charge=self.species_dict[label].charge, is_ts=False)  # before optimization
        if self.species_dict[label].initial_xyz is None and self.species_dict[label].final_xyz is None \
                and not self.testing:
            self.job_dict[label]['conformers'] = dict()
            for i, xyz in enumerate(self.species_dict[label].conformers):
                self.run_job(label=label, xyz=xyz, level_of_theory=self.conformer_level,
                             job_type='conformer', conformer=i)

    def parse_conformer(self, job, label, i):
        """
        Parse E0 (kJ/mol) from the conformer opt output file.
        For species, save it in the Species.conformer_energies attribute.
        Fot TSs, save it in the TSGuess.energy attribute, and also parse the geometry.
        """
        if job.job_status[1] == 'done':
            log = determine_qm_software(fullpath=job.local_path_to_output_file)
            coords, number, _ = log.loadGeometry()
            if self.species_dict[label].is_ts:
                self.species_dict[label].ts_guesses[i].energy = log.loadEnergy() * 0.001  # in kJ/mol
                self.species_dict[label].ts_guesses[i].opt_xyz = get_xyz_string(coords=coords, numbers=number)
                self.species_dict[label].ts_guesses[i].index = i
                logger.debug('Energy for TSGuess {0} of {1} is {2:.2f}'.format(
                    i, self.species_dict[label].label, self.species_dict[label].ts_guesses[i].energy))
            else:
                self.species_dict[label].conformer_energies[i] = log.loadEnergy() * 0.001  # in kJ/mol
                self.species_dict[label].conformers[i] = get_xyz_string(coords=coords, numbers=number)
                logger.debug('Energy for conformer {0} of {1} is {2:.2f}'.format(
                    i, self.species_dict[label].label, self.species_dict[label].conformer_energies[i]))
        else:
            logger.warning('Conformer {i} for {label} did not converge!'.format(i=i, label=label))

    def determine_most_stable_conformer(self, label):
        """
        Determine the most stable conformer for a species (which is not a TS).
        Also run an isomorphism check.
        Save the resulting xyz as `initial_xyz`
        """
        if self.species_dict[label].is_ts:
            raise SchedulerError('The determine_most_stable_conformer() method does not deal with transition'
                                 ' state guesses.')
        if 'conformers' in self.job_dict[label] and all(e is None for e in self.species_dict[label].conformer_energies):
            logger.error('No conformer converged for species {0}! Trying to troubleshoot conformer jobs...'.format(
                label))
            for i, job in self.job_dict[label]['conformers'].items():
                self.troubleshoot_ess(label, job, level_of_theory=job.level_of_theory, job_type='conformer',
                                      conformer=job.conformer)
        else:
            conformer_xyz = None
            xyzs = list()
            if self.species_dict[label].conformer_energies:
                xyzs = self.species_dict[label].conformers
            else:
                for job in self.job_dict[label]['conformers'].values():
                    log = determine_qm_software(fullpath=job.local_path_to_output_file)
                    try:
                        coords, number, _ = log.loadGeometry()
                    except RMGInputError:
                        xyzs.append(None)
                    else:
                        xyzs.append(get_xyz_string(coords=coords, numbers=number))
            xyzs_in_original_order = xyzs
            energies, xyzs = (list(t) for t in zip(*sorted(zip(self.species_dict[label].conformer_energies, xyzs))))
            plotter.save_conformers_file(project_directory=self.project_directory, label=label,
                                         xyzs=self.species_dict[label].conformers, level_of_theory=self.conformer_level,
                                         multiplicity=self.species_dict[label].multiplicity,
                                         charge=self.species_dict[label].charge, is_ts=False,
                                         energies=self.species_dict[label].conformer_energies)  # after optimization
            # Run isomorphism checks if a 2D representation is available
            if self.species_dict[label].mol is not None:
                for i, xyz in enumerate(xyzs):
                    try:
                        b_mol = molecules_from_xyz(xyz, multiplicity=self.species_dict[label].multiplicity,
                                                   charge=self.species_dict[label].charge)[1]
                    except SanitizationError:
                        b_mol = None
                    if b_mol is not None:
                        try:
                            is_isomorphic = check_isomorphism(self.species_dict[label].mol, b_mol)
                        except (ValueError, AtomTypeError) as e:
                            if self.species_dict[label].charge:
                                logger.error('Could not determine isomorphism for charged species {0}. '
                                             'Optimizing the most stable conformer anyway. Got the '
                                             'following error:\n{1}'.format(label, e))
                            else:
                                logger.error('Could not determine isomorphism for (non-charged) species {0}. '
                                             'Optimizing the most stable conformer anyway. Got the '
                                             'following error:\n{1}'.format(label, e))
                            conformer_xyz = xyzs[0]
                            break
                        if is_isomorphic:
                            if i == 0:
                                logger.info('Most stable conformer for species {0} was found to be isomorphic '
                                            'with the 2D graph representation {1}\n'.format(label, b_mol.toSMILES()))
                                conformer_xyz = xyz
                                self.output[label]['status'] += 'passed isomorphism check; '
                            else:
                                if energies[i] is not None:
                                    logger.info('A conformer for species {0} was found to be isomorphic '
                                                'with the 2D graph representation {1}. This conformer is {2:.2f} kJ/mol'
                                                ' above the most stable one which corresponds to  {3} (and is not '
                                                'isomorphic). Using the isomorphic conformer for further geometry '
                                                'optimization.'.format(
                                                 label, self.species_dict[label].mol.toSMILES(),
                                                 (energies[i] - min_list(energies)) * 0.001,
                                                 molecules_from_xyz(xyzs[0],
                                                                    multiplicity=self.species_dict[label].multiplicity,
                                                                    charge=self.species_dict[label].charge)[1]))
                                conformer_xyz = xyz
                                self.output[label]['status'] += 'passed isomorphism check but not for the most stable' \
                                                                ' conformer; '
                            break
                        else:
                            if i == 0:
                                logger.warning('Most stable conformer for species {0} with structure {1} was found to '
                                               'be NON-isomorphic with the 2D graph representation {2}. Searching for '
                                               'a different conformer that is isomorphic...'.format(
                                                label, b_mol.toSMILES(), self.species_dict[label].mol.toSMILES()))
                else:
                    smiles_list = list()
                    for xyz in xyzs:
                        try:
                            b_mol = molecules_from_xyz(xyz, multiplicity=self.species_dict[label].multiplicity,
                                                       charge=self.species_dict[label].charge)[1]
                            smiles_list.append(b_mol.toSMILES())
                        except (SanitizationError, AttributeError):
                            smiles_list.append('Could not perceive molecule')
                    if self.allow_nonisomorphic_2d or self.species_dict[label].charge:
                        # we'll optimize the most stable conformer even if it is not isomorphic to the 2D graph
                        logger.error('No conformer for {0} was found to be isomorphic with the 2D graph representation'
                                     ' {1} (got: {2}). Optimizing the most stable conformer anyway.'.format(
                                      label, self.species_dict[label].mol.toSMILES(), smiles_list))
                        self.output[label]['status'] += 'No conformer was found to be isomorphic with the 2D' \
                                                        ' graph representation; '
                        if self.species_dict[label].charge:
                            logger.warning('Isomorphism check cannot be done for charged species {0}'.format(label))
                        conformer_xyz = xyzs[0]
                    else:
                        logger.error('No conformer for {0} was found to be isomorphic with the 2D graph representation'
                                     ' {1} (got: {2}). NOT optimizing this species.'.format(
                                      label, self.species_dict[label].mol.toSMILES(), smiles_list))
                        self.output[label]['status'] += 'Error: No conformer was found to be isomorphic with the 2D' \
                                                        ' graph representation!; '
            else:
                logger.warning('Could not run isomorphism check for species {0} due to missing 2D graph '
                               'representation. Using the most stable conformer for further geometry'
                               ' optimization.'.format(label))
                conformer_xyz = xyzs[0]
            if conformer_xyz is not None:
                self.species_dict[label].initial_xyz = conformer_xyz
                self.species_dict[label].most_stable_conformer = xyzs_in_original_order.index(conformer_xyz)

    def determine_most_likely_ts_conformer(self, label):
        """
        Determine the most likely TS conformer.
        Save the resulting xyz as `initial_xyz`
        """
        if not self.species_dict[label].is_ts:
            raise SchedulerError('determine_most_likely_ts_conformer() method only processes transition state guesses.')
        for tsg in self.species_dict[label].ts_guesses:
            if tsg.success:
                self.species_dict[label].successful_methods.append(tsg.method)
            else:
                self.species_dict[label].unsuccessful_methods.append(tsg.method)
        message = '\nAll TS guesses for {0} terminated.'.format(label)
        if self.species_dict[label].successful_methods and not self.species_dict[label].unsuccessful_methods:
            message += '\n All methods were successful: {0}'.format(self.species_dict[label].successful_methods)
        elif self.species_dict[label].successful_methods:
            message += ' Successful methods: {0}'.format(self.species_dict[label].successful_methods)
        elif self.species_dict[label].yml_path is not None and self.species_dict[label].final_xyz is not None:
            message += ' Geometry parsed from YAML file.'
        else:
            message += ' No method has converged!'
            logger.error('No TS methods for {0} have converged!'.format(label))
        if self.species_dict[label].unsuccessful_methods:
            message += ' Unsuccessful methods: {0}'.format(self.species_dict[label].unsuccessful_methods)
        logger.info(message)
        logger.info('\n')

        if all(tsg.energy is None for tsg in self.species_dict[label].ts_guesses):
            logger.error('No guess converged for TS {0}!'.format(label))
            # for i, job in self.job_dict[label]['conformers'].items():
            #     self.troubleshoot_ess(label, job, level_of_theory=job.level_of_theory, job_type='conformer',
            #                           conformer=job.conformer)
        else:
            # currently we take the most stable guess. We'll need to implement additional checks here:
            # - normal displacement mode of the imaginary frequency
            # - IRC isomorphism checks
            rxn_txt = '' if self.species_dict[label].rxn_label is None\
                else ' of reaction {0}'.format(self.species_dict[label].rxn_label)
            logger.info('\n\nGeometry *guesses* of successful TS guesses for {0}{1}:'.format(label, rxn_txt))
            e_min = min_list([tsg.energy for tsg in self.species_dict[label].ts_guesses])
            i_min = None
            for tsg in self.species_dict[label].ts_guesses:
                if tsg.energy is not None and tsg.energy == e_min:
                    i_min = tsg.index
            for tsg in self.species_dict[label].ts_guesses:
                if tsg.index == i_min:
                    self.species_dict[label].chosen_ts = i_min  # change this if selecting a better TS later
                    self.species_dict[label].chosen_ts_method = tsg.method  # change if selecting a better TS later
                    self.species_dict[label].initial_xyz = tsg.xyz
                if tsg.success and tsg.energy is not None:  # guess method and ts_level opt were both successful
                    tsg.energy = tsg.energy - e_min
                    logger.info('TS guess {0} for {1}. Method: {2}, relative energy: {3:.2f} kJ/mol, guess execution '
                                'time: {4}'.format(tsg.index, label, tsg.method, tsg.energy, tsg.execution_time))
                    # for TSs, only use `draw_3d()`, not `show_sticks()` which gets connectivity wrong:
                    plotter.draw_structure(xyz=tsg.xyz, method='draw_3d')
            if self.species_dict[label].chosen_ts is None:
                raise SpeciesError('Could not pair most stable conformer {0} of {1} to a respective '
                                   'TS guess'.format(i_min, label))
            plotter.save_conformers_file(project_directory=self.project_directory, label=label,
                                         xyzs=[tsg.opt_xyz for tsg in self.species_dict[label].ts_guesses],
                                         level_of_theory=self.ts_guess_level,
                                         multiplicity=self.species_dict[label].multiplicity,
                                         charge=self.species_dict[label].charge, is_ts=True,
                                         energies=[tsg.energy for tsg in self.species_dict[label].ts_guesses],
                                         ts_methods=[tsg.method for tsg in self.species_dict[label].ts_guesses])

    def parse_composite_geo(self, label, job):
        """
        Check that a 'composite' job converged successfully, and parse the geometry into `final_xyz`.
        Also checks (QA) that no imaginary frequencies were assigned for stable species,
        and that exactly one imaginary frequency was assigned for a TS.
        Returns ``True`` if the job converged successfully, ``False`` otherwise and troubleshoots.
        """
        logger.debug('parsing composite geo for {0}'.format(job.job_name))
        freq_ok = False
        if job.job_status[1] == 'done':
            log = determine_qm_software(fullpath=job.local_path_to_output_file)
            coords, number, _ = log.loadGeometry()
            self.species_dict[label].final_xyz = get_xyz_string(coords=coords, numbers=number)
            self.output[label]['status'] += 'composite converged; '
            self.output[label]['composite'] = os.path.join(job.local_path, 'output.out')
            self.species_dict[label].opt_level = self.composite_method
            rxn_str = ''
            if self.species_dict[label].is_ts:
                rxn_str = ' of reaction {0}'.format(self.species_dict[label].rxn_label)
            logger.info('\nOptimized geometry for {label}{rxn} at {level}:\n{xyz}'.format(
                label=label, rxn=rxn_str, level=job.level_of_theory, xyz=self.species_dict[label].final_xyz))
            if not job.is_ts:
                plotter.draw_structure(species=self.species_dict[label], project_directory=self.project_directory)
            else:
                # for TSs, only use `draw_3d()`, not `show_sticks()` which gets connectivity wrong:
                plotter.draw_structure(species=self.species_dict[label], project_directory=self.project_directory,
                                       method='draw_3d')
            # Check frequencies (using cclib crashes for CBS-QB3 output, so using an explicit parser here)
            frequencies = parser.parse_frequencies(job.local_path_to_output_file, job.software)
            freq_ok = self.check_negative_freq(label=label, job=job, vibfreqs=frequencies)
            if freq_ok:
                # Update restart dictionary and save the yaml restart file:
                self.save_restart_dict()
                return True  # run freq / scan jobs on this optimized geometry
            elif not self.species_dict[label].is_ts:
                self.troubleshoot_negative_freq(label=label, job=job)
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
        success = False
        logger.debug('parsing opt geo for {0}'.format(job.job_name))
        if job.job_status[1] == 'done':
            log = determine_qm_software(fullpath=job.local_path_to_output_file)
            coords, number, _ = log.loadGeometry()
            self.species_dict[label].final_xyz = get_xyz_string(coords=coords, numbers=number)
            if not job.fine and self.job_types['fine'] and not job.software == 'molpro':
                # Run opt again using a finer grid.
                xyz = self.species_dict[label].final_xyz
                self.species_dict[label].initial_xyz = xyz  # save for troubleshooting, since trsh goes by initial
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, job_type='opt', fine=True)
            else:
                success = True
                if 'optfreq' in job.job_name:
                    self.check_freq_job(label, job)
                self.output[label]['status'] += 'opt converged; '
                self.species_dict[label].opt_level = self.opt_level
                if self.species_dict[label].is_ts:
                    rxn_str = ' of reaction {0}'.format(self.species_dict[label].rxn_label)
                else:
                    rxn_str = ''
                    # Update restart dictionary and save the yaml restart file:
                    # This is the final geometry of a stable species. Determine whether the species participates in
                    # a reaction (or several), if so update its geometry in the respective Species representing the
                    # reaction's TS
                    for rxn in self.rxn_list:
                        reactant = True if label in rxn.reactants else False
                        product = True if label in rxn.products else False
                        if reactant or product:
                            ts = self.species_dict[rxn.ts_label]
                            for tsg in ts.ts_guesses:
                                if reactant and\
                                        not any([reactant_xyz[0] == label for reactant_xyz in tsg.reactants_xyz]):
                                    # This species is a reactant of rxn,
                                    # and its geometry wasn't saved in the TSGuess objects
                                    tsg.reactants_xyz.append((label, self.species_dict[label].final_xyz))
                                if product and\
                                        not any([product_xyz[0] == label for product_xyz in tsg.products_xyz]):
                                    # This species is a product of rxn,
                                    # and its geometry wasn't saved in the TSGuess objects
                                    tsg.products_xyz.append((label, self.species_dict[label].final_xyz))
                logger.info('\nOptimized geometry for {label}{rxn} at {level}:\n{xyz}'.format(
                    label=label, rxn=rxn_str, level=job.level_of_theory, xyz=self.species_dict[label].final_xyz))
                self.save_restart_dict()
                self.output[label]['geo'] = job.local_path_to_output_file  # will be overwritten when freq is done
            if not job.is_ts:
                plotter.draw_structure(species=self.species_dict[label], project_directory=self.project_directory)
            else:
                # for TSs, only use `draw_3d()`, not `show_sticks()` which gets connectivity wrong:
                plotter.draw_structure(species=self.species_dict[label], project_directory=self.project_directory,
                                       method='draw_3d')
        else:
            self.troubleshoot_opt_jobs(label=label)
        if success:
            return True  # run freq / sp / scan jobs on this optimized geometry
        else:
            return False  # return ``False``, so no freq / sp / scan jobs are initiated for this unoptimized geometry

    def check_freq_job(self, label, job):
        """
        Check that a freq job converged successfully. Also checks (QA) that no imaginary frequencies were assigned for
        stable species, and that exactly one imaginary frequency was assigned for a TS.
        """
        if job.job_status[1] == 'done':
            if not os.path.isfile(job.local_path_to_output_file):
                raise SchedulerError('Called check_freq_job with no output file')
            vibfreqs = parser.parse_frequencies(path=str(job.local_path_to_output_file), software=job.software)
            freq_ok = self.check_negative_freq(label=label, job=job, vibfreqs=vibfreqs)
            if not self.species_dict[label].is_ts and not freq_ok:
                self.troubleshoot_negative_freq(label=label, job=job)
            if freq_ok:
                # copy the frequency file to the species / TS output folder
                folder_name = 'rxns' if self.species_dict[label].is_ts else 'Species'
                freq_path = os.path.join(self.project_directory, 'output', folder_name, label, 'geometry', 'freq.out')
                shutil.copyfile(job.local_path_to_output_file, freq_path)
                # set species.polarizability
                polarizability = parser.parse_polarizability(job.local_path_to_output_file)
                if polarizability is not None:
                    self.species_dict[label].transport_data.polarizability = (polarizability, str('angstroms^3'))
                    if self.species_dict[label].transport_data.comment:
                        self.species_dict[label].transport_data.comment +=\
                            str('\nPolarizability calculated at the {0} level of theory'.format(self.freq_level))
                    else:
                        self.species_dict[label].transport_data.comment =\
                            str('Polarizability calculated at the {0} level of theory'.format(self.freq_level))
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
            logger.error('TS {0} has {1} imaginary frequencies,'
                         ' should have exactly 1.'.format(label, neg_freq_counter))
            self.output[label]['status'] += 'Warning: {0} imaginary freq for TS; '.format(neg_freq_counter)
            return False
        elif not self.species_dict[label].is_ts and neg_freq_counter != 0:
            logger.error('Species {0} has {1} imaginary frequencies,'
                         ' should have exactly 0.'.format(label, neg_freq_counter))
            self.output[label]['status'] += 'Warning: {0} imaginary freq for stable species; '.format(
                neg_freq_counter)
            return False
        else:
            if self.species_dict[label].is_ts:
                logger.info('{0} has exactly one imaginary frequency: {1}'.format(label, neg_fre))
            self.output[label]['status'] += 'freq converged; '
            self.output[label]['geo'] = job.local_path_to_output_file
            self.output[label]['freq'] = job.local_path_to_output_file
            if not self.testing:
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
            self.species_dict[label].t1 = parser.parse_t1(self.output[label]['sp'])
            zpe_scale_factor = 0.99 if self.composite_method.lower() == 'cbs-qb3' else 1.0
            self.species_dict[label].e_elect = parser.parse_e_elect(self.output[label]['sp'],
                                                                    zpe_scale_factor=zpe_scale_factor)
            if self.species_dict[label].t1 is not None:
                txt = ''
                if self.species_dict[label].t1 > 0.02:
                    txt += ". Looks like it requires multireference treatment, I wouldn't trust it's calculated energy!"
                    self.output[label]['status'] += 'T1 = {0}; '.format(self.species_dict[label].t1)
                elif self.species_dict[label].t1 > 0.015:
                    txt += ". It might have multireference characteristic."
                    self.output[label]['status'] += 'T1 = {0}; '.format(self.species_dict[label].t1)
                logger.info('Species {0} has a T1 diagnostic parameter of {1}{2}'.format(
                    label, self.species_dict[label].t1, txt))
            # Update restart dictionary and save the yaml restart file:
            self.save_restart_dict()
            if self.species_dict[label].number_of_atoms == 1:
                # save the geometry from the sp job for monoatomic species for which no opt/freq jobs will be spawned
                self.output[label]['geo'] = job.local_path_to_output_file
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
                              'success': ``bool``,
                              'invalidation_reason': ``str``,
                              'times_dihedral_set': ``int``,
                              'scan_path': <path to scan output file>,
                              'max_e': ``float``,  # in kJ/mol},
                              'symmetry': ``int``}
                          2: {}, ...
                         }
        """
        for i in range(self.species_dict[label].number_of_rotors):
            message = ''
            invalidation_reason = ''
            trsh = False
            if self.species_dict[label].rotors_dict[i]['pivots'] == job.pivots:
                invalidate = False
                if job.job_status[1] == 'done':
                    # ESS converged. Get PES scan using Arkane:
                    log = determine_qm_software(fullpath=job.local_path_to_output_file)
                    try:
                        v_list, angle = log.loadScanEnergies()
                    except ZeroDivisionError:
                        logger.error('Energies from rotor scan of {label} between pivots {pivots} could not'
                                     'be read. Invalidating rotor.'.format(label=label, pivots=job.pivots))
                        invalidate = True
                        invalidation_reason = 'could not read energies'
                    else:
                        v_list = np.array(v_list, np.float64)
                        v_list *= 0.001  # convert to kJ/mol
                        # 1. Check smoothness:
                        if abs(v_list[-1] - v_list[0]) > inconsistency_az:
                            # initial and final points differ by more than `inconsistency_az` kJ/mol.
                            # seems like this rotor broke the conformer. Invalidate
                            error_message = 'Rotor scan of {label} between pivots {pivots} is inconsistent by more' \
                                            ' than {incons_az:.2f} kJ/mol between initial and final positions.' \
                                            ' Invalidating rotor.\nv_list[0] = {v0}, v_list[-1] = {vneg1}'.format(
                                             label=label, pivots=job.pivots, incons_az=inconsistency_az,
                                             v0=v_list[0], vneg1=v_list[-1])
                            logger.error(error_message)
                            message += error_message + '; '
                            invalidate = True
                            invalidation_reason = 'initial and final points are inconsistent by more than {0:.2f} ' \
                                                  'kJ/mol'.format(inconsistency_az)
                            if not job.scan_trsh:
                                logger.info('Trying to troubleshoot rotor {0} of {1}...'.format(job.pivots, label))
                                trsh = True
                                self.troubleshoot_scan_job(job=job)
                        if not invalidate:
                            v_last = v_list[-1]
                            for v in v_list:
                                if abs(v - v_last) > inconsistency_ab * max(v_list):
                                    # Two consecutive points on the scan differ by more than `inconsistency_ab` kJ/mol.
                                    # This is a serious inconsistency. Invalidate
                                    error_message = 'Rotor scan of {label} between pivots {pivots} is inconsistent ' \
                                                    'by more than {incons_ab:.2f} kJ/mol between two consecutive ' \
                                                    'points. Invalidating rotor.'.format(
                                                     label=label, pivots=job.pivots,
                                                     incons_ab=inconsistency_ab * max(v_list))
                                    logger.error(error_message)
                                    message += error_message + '; '
                                    invalidate = True
                                    invalidation_reason = 'two consecutive points are inconsistent by more than ' \
                                                          '{0:.2f} kJ/mol'.format(inconsistency_ab)
                                    if not job.scan_trsh:
                                        logger.info('Trying to troubleshoot rotor {0} of {1}...'.format(
                                            job.pivots, label))
                                        trsh = True
                                        self.troubleshoot_scan_job(job=job)
                                    break
                                if abs(v - v_list[0]) > maximum_barrier:
                                    # The barrier for the hinderd rotor is higher than `maximum_barrier` kJ/mol.
                                    # Invalidate
                                    warn_message = 'Rotor scan of {label} between pivots {pivots} has a barrier ' \
                                                   'larger than {max_barrier:.2f} kJ/mol. Invalidating rotor.'.format(
                                                    label=label, pivots=job.pivots, max_barrier=maximum_barrier)
                                    logger.warning(warn_message)
                                    message += warn_message + '; '
                                    invalidate = True
                                    invalidation_reason = 'scan has a barrier larger than {0}' \
                                                          ' kJ/mol'.format(maximum_barrier)
                                    break
                                v_last = v
                        # 2. Check conformation:
                        invalidated = ''
                        if not invalidate and not trsh:
                            v_diff = (v_list[0] - np.min(v_list))
                            if v_diff >= 2 or v_diff > 0.5 * (max(v_list) - min(v_list)):
                                self.species_dict[label].rotors_dict[i]['success'] = False
                                logger.info('Species {label} is not oriented correctly around pivots {pivots},'
                                            ' searching for a better conformation...'.format(label=label,
                                                                                             pivots=job.pivots))
                                # Find the rotation dihedral in degrees to the closest minimum:
                                min_v = v_list[0]
                                min_index = 0
                                for j, v in enumerate(v_list):
                                    if v < min_v - 2:
                                        min_v = v
                                        min_index = j
                                self.species_dict[label].set_dihedral(
                                    pivots=self.species_dict[label].rotors_dict[i]['pivots'],
                                    scan=self.species_dict[label].rotors_dict[i]['scan'],
                                    deg_increment=min_index*rotor_scan_resolution)
                                self.delete_all_species_jobs(label)
                                self.run_opt_job(label)  # run opt on new initial_xyz with the desired dihedral
                            else:
                                self.species_dict[label].rotors_dict[i]['success'] = True
                        elif invalidate:
                            invalidated = '*INVALIDATED* '
                        if self.species_dict[label].rotors_dict[i]['success']:
                            self.species_dict[label].rotors_dict[i]['symmetry'], _ = determine_rotor_symmetry(
                                rotor_path=job.local_path_to_output_file, label=label,
                                pivots=self.species_dict[label].rotors_dict[i]['pivots'])
                            symmetry = ' has symmetry {0}'.format(self.species_dict[label].rotors_dict[i]['symmetry'])

                            logger.info('{invalidated}Rotor scan {scan} between pivots {pivots}'
                                        ' for {label}{symmetry}'.format(
                                         invalidated=invalidated, scan=self.species_dict[label].rotors_dict[i]['scan'],
                                         pivots=self.species_dict[label].rotors_dict[i]['pivots'],
                                         label=label, symmetry=symmetry))
                            folder_name = 'rxns' if job.is_ts else 'Species'
                            rotor_path = os.path.join(self.project_directory, 'output', folder_name,
                                                      job.species_name, 'rotors')
                            message += invalidated
                            plotter.plot_rotor_scan(angle, v_list, path=rotor_path, pivots=job.pivots, comment=message)
                else:
                    # scan job crashed
                    invalidate = True
                    invalidation_reason = 'scan job crashed'
                if not trsh:
                    if invalidate:
                        self.species_dict[label].rotors_dict[i]['success'] = False
                    else:
                        self.species_dict[label].rotors_dict[i]['success'] = True
                    self.species_dict[label].rotors_dict[i]['scan_path'] = job.local_path_to_output_file
                    self.species_dict[label].rotors_dict[i]['invalidation_reason'] = invalidation_reason
                break  # A job object has only one pivot. Break if found, otherwise raise an error.
        else:
            raise SchedulerError('Could not match rotor with pivots {0} in species {1}'.format(job.pivots, label))
        self.save_restart_dict()

    def check_md_job(self, label, job, max_iterations=10):
        """
        Check whether the MD spawning algorithm converged on a single structure.
        If not converged, spawn another MD job (up to a maximum number of jobs).
        If it did converge, save the resulting lowest conformers and DFT them.

        Args:
             label (str, unicode): The species label.
             job (Job): The Gromacs MD job to check.
             max_iterations (int, optional): The maximal number of MD trials per species.
        """
        done = False
        conf_list = read_yaml_file(job.local_path_to_output_file)
        lowest_conf = conformers.get_lowest_confs(label=label, confs=conf_list)[0]
        if self.species_dict[label].recent_md_conformer is None:
            self.species_dict[label].recent_md_conformer = lowest_conf + [0]
        else:
            if self.species_dict[label].recent_md_conformer[2] >= max_iterations:
                logger.error('Could not converge on a single conformer using Gromacs for species {0} '
                             'even after {1} iterations. Using the latest conformer.'.format(label, max_iterations))
                done = True
            elif lowest_conf[1] < self.species_dict[label].recent_md_conformer[1]:
                # unconverged
                self.species_dict[label].recent_md_conformer = lowest_conf\
                                                               + [self.species_dict[label].recent_md_conformer[2] + 1]
            elif lowest_conf[1] == self.species_dict[label].recent_md_conformer[1]:
                if conformers.compare_xyz(lowest_conf[0], self.species_dict[label].recent_md_conformer[0]):
                    # converged
                    done = True
                else:
                    # same energy but different conformer? for now we'll consider it as converged
                    logger.warning('MD jobs for {0} converged with same energy conformer by different xyz:\n'
                                   '{1}\n\nand\n\n{2}'.format(label, self.species_dict[label].recent_md_conformer[0],
                                                              lowest_conf[0]))
                    done = True  # Todo: reconsider
            else:
                # why did we found a higher conformer?
                logger.error('Could not converge on a single conformer using Gromacs for species {0}, got a higher'
                             'energy conformer. Using the latest lowest conformer.'.format(label))
                done = True
        if done:
            # process conformers and DFT them
            logger.info('Final conformer for {0}:\n{1}'.format(label, lowest_conf[0]))
            plotter.draw_structure(xyz=lowest_conf[0], species=self.species_dict[label])
            lowest_confs = conformers.get_lowest_confs(label=label, confs=conf_list, n=self.confs_to_dft)
            self.species_dict[label].conformers.extend(standardize_xyz_string(conf[0]) for conf in lowest_confs)
            self.species_dict[label].conformer_energies = [None] * len(lowest_confs)
            self.process_conformers(label=label)
        else:
            # spawn a new MD simulation
            ordinal = get_ordinal_indicator(self.species_dict[label].recent_md_conformer[2] + 1)
            logger.info('{0}{1} conformer for {2}:\n{3}'.format(self.species_dict[label].recent_md_conformer[2] + 1,
                                                                ordinal, label, lowest_conf[0]))
            plotter.draw_structure(xyz=lowest_conf[0], species=self.species_dict[label])
            ordinal = get_ordinal_indicator(self.species_dict[label].recent_md_conformer[2] + 2)
            logger.info('Spawning the {0}{1} round of MD simulations for {2}'.format(
                self.species_dict[label].recent_md_conformer[2] + 2, ordinal, label))
            self.spawn_md_jobs(label, prev_conf_list=conf_list)

    def check_all_done(self, label):
        """
        Check that we have all required data for the species/TS in ``label``
        """
        status = self.output[label]['status']
        if 'error' not in status and ('composite converged' in status or ('sp converged' in status and
                (self.species_dict[label].is_ts or self.species_dict[label].number_of_atoms == 1 or
                ('freq converged' in status and 'opt converged' in status)))):
            self.output[label]['status'] += 'ALL converged'
            plotter.save_geo(species=self.species_dict[label], project_directory=self.project_directory)
            if self.species_dict[label].is_ts:
                self.species_dict[label].make_ts_report()
                logger.info(self.species_dict[label].ts_report + '\n')
            zero_delta = datetime.timedelta(0)
            conf_time = max([job.run_time for job in self.job_dict[label]['conformers'].values()])\
                if 'conformers' in self.job_dict[label] else zero_delta
            opt_time = sum_time_delta([job.run_time for job in self.job_dict[label]['opt'].values()])\
                if 'opt' in self.job_dict[label] else zero_delta
            comp_time = sum_time_delta([job.run_time for job in self.job_dict[label]['composite'].values()])\
                if 'composite' in self.job_dict[label] else zero_delta
            other_time = max([sum_time_delta([job.run_time for job in job_dictionary.values()])
                              for job_type, job_dictionary in self.job_dict[label].items()
                              if job_type not in ['conformers', 'opt', 'composite']])\
                if any([job_type not in ['conformers', 'opt', 'composite']
                        for job_type in self.job_dict[label].keys()]) else zero_delta
            self.species_dict[label].run_time = self.species_dict[label].run_time\
                                                or (conf_time or zero_delta) + (opt_time or zero_delta)\
                                                + (comp_time or zero_delta) + (other_time or zero_delta)
            logger.info('\nAll jobs for species {0} successfully converged.'
                        ' Run time: {1}'.format(label, self.species_dict[label].run_time))
        elif not self.output[label]['status']:
            self.output[label]['status'] = 'nothing converged'
            logger.error('Species {0} did not converge. Status is: {1}'.format(label, status))
        # Update restart dictionary and save the yaml restart file:
        self.save_restart_dict()

    def get_servers_jobs_ids(self):
        """
        Check status on all active servers, return a list of relevant running job IDs
        """
        self.servers_jobs_ids = list()
        for server in self.servers:
            if server != 'local':
                ssh = SSHClient(server)
                self.servers_jobs_ids.extend(ssh.check_running_jobs_ids())
            else:
                self.servers_jobs_ids.extend(check_running_jobs_ids())

    def troubleshoot_negative_freq(self, label, job):
        """
        Troubleshooting cases where stable species (not TS's) have negative frequencies.
        We take  +/-1.1 displacements, generating several initial geometries, and running them as conformers

        Todo:
            * get all torsions of the molecule (if weren't already generated),
              identify atom/s with largest displacements (top 2)
              determine torsions with unique PIVOTS where these atoms are in the "scan" and "top" but not pivotal
              generate a 360 scal using 30 deg increments and append all 12 results as conformers
              (consider rotor symmetry to append less conformers?)
        """
        factor = 1.1
        ccparser = cclib.io.ccopen(str(job.local_path_to_output_file), logging.CRITICAL)
        data = ccparser.parse()
        vibfreqs = data.vibfreqs
        vibdisps = data.vibdisps
        atomnos = data.atomnos
        atomcoords = data.atomcoords
        if len(self.species_dict[label].neg_freqs_trshed) > 10:
            logger.error('Species {0} was troubleshooted for negative frequencies too many times.'.format(label))
            if not self.job_types['1d_rotors']:
                logger.error('The rotor scans feature is turned off, '
                             'cannot troubleshoot geometry using dihedral modifications.')
                self.output[label]['status'] = '1d_rotors = False; '
            logger.error('Invalidating species.')
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
        if len(neg_freqs_idx) == 1 and not len(self.species_dict[label].neg_freqs_trshed):
            # species has one negative frequency, and has not been troubleshooted for it before
            logger.info('Species {0} has a negative frequency ({1}). Perturbing its geometry using the respective '
                        'vibrational displacements'.format(label, vibfreqs[largest_neg_freq_idx]))
            neg_freqs_idx = [largest_neg_freq_idx]  # indices of the negative frequencies to troubleshoot for
        elif len(neg_freqs_idx) == 1 and len(self.species_dict[label].neg_freqs_trshed):
            # species has one negative frequency, and has been troubleshooted for it before
            factor = 1 + 0.1 * (len(self.species_dict[label].neg_freqs_trshed) + 1)
            logger.info('Species {0} has a negative frequency ({1}) for the {2} time. Perturbing its geometry using '
                        'the respective vibrational displacements, this time using a larger factor (x {3})'.format(
                         label, vibfreqs[largest_neg_freq_idx], len(self.species_dict[label].neg_freqs_trshed), factor))
            neg_freqs_idx = [largest_neg_freq_idx]  # indices of the negative frequencies to troubleshoot for
        elif len(neg_freqs_idx) > 1 and not len(self.species_dict[label].neg_freqs_trshed):
            # species has more than one negative frequency, and has not been troubleshooted for it before
            logger.info('Species {0} has {1} negative frequencies. Perturbing its geometry using the vibrational '
                        'displacements of its largest negative frequency, {2}'.format(label, len(neg_freqs_idx),
                                                                                      vibfreqs[largest_neg_freq_idx]))
            neg_freqs_idx = [largest_neg_freq_idx]  # indices of the negative frequencies to troubleshoot for
        elif len(neg_freqs_idx) > 1 and len(self.species_dict[label].neg_freqs_trshed):
            # species has more than one negative frequency, and has been troubleshooted for it before
            logger.info('Species {0} has {1} negative frequencies. Perturbing its geometry using the vibrational'
                        ' displacements of ALL negative frequencies'.format(label, len(neg_freqs_idx)))
        self.species_dict[label].neg_freqs_trshed.extend([round(vibfreqs[i], 2) for i in neg_freqs_idx])  # record freqs
        logger.info('Deleting all currently running jobs for species {0} before troubleshooting for'
                    ' negative frequency...'.format(label))
        self.delete_all_species_jobs(label)
        self.species_dict[label].conformers = list()  # initialize the conformer list
        self.species_dict[label].conformer_energies = list()
        atomcoords = atomcoords[-1]  # it's a list within a list, take the last geometry
        for neg_freq_idx in neg_freqs_idx:
            displacement = vibdisps[neg_freq_idx]
            xyz1 = atomcoords + factor * displacement
            xyz2 = atomcoords - factor * displacement
            self.species_dict[label].conformers.append(get_xyz_string(coords=xyz1, numbers=atomnos))
            self.species_dict[label].conformers.append(get_xyz_string(coords=xyz2, numbers=atomnos))
            self.species_dict[label].conformer_energies.extend([None, None])  # a placeholder (lists are synced)
        self.job_dict[label]['conformers'] = dict()  # initialize the conformer job dictionary
        for i, xyz in enumerate(self.species_dict[label].conformers):
            self.run_job(label=label, xyz=xyz, level_of_theory=self.conformer_level, job_type='conformer', conformer=i)

    def troubleshoot_scan_job(self, job):
        """
        Try freezing all dihedrals other than the scan's pivots for this job
        """
        label = job.species_name
        species_scan_lists = [rotor_dict['scan'] for rotor_dict in self.species_dict[label].rotors_dict.values()]
        if job.scan not in species_scan_lists:
            raise SchedulerError('Could not find the dihedral to troubleshoot for in the dcan list of species'
                                 ' {0}'.format(label))
        species_scan_lists.pop(species_scan_lists.index(job.scan))
        if len(species_scan_lists):
            scan_trsh = '\n'
            for scan in species_scan_lists:
                scan_trsh += 'D ' + ''.join([str(num) + ' ' for num in scan]) + 'F\n'
            scan_res = min(4, int(job.scan_res / 2))
            # make sure mod(360, scan res) is 0:
            if scan_res not in [4, 2, 1]:
                scan_res = min([4, 2, 1], key=lambda x: abs(x - scan_res))
            self.run_job(label=label, xyz=job.xyz, level_of_theory=job.level_of_theory, job_type='scan',
                         scan=job.scan, pivots=job.pivots, scan_trsh=scan_trsh, scan_res=4)

    def troubleshoot_opt_jobs(self, label):
        """
        We're troubleshooting for opt jobs.
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
                    logger.error('opt job for {label} seems right, yet "run_opt_job"'
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
                        logger.error('Optimization job for {label} with a fine grid terminated successfully'
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
        logger.info('\n')
        logger.warning('Troubleshooting {label} job {job_name} which failed with status "{stat}" in {soft}.'.format(
            job_name=job.job_name, label=label, stat=job.job_status[1], soft=job.software))
        if conformer != -1:
            xyz = self.species_dict[label].conformers[conformer]
        else:
            xyz = self.species_dict[label].final_xyz or self.species_dict[label].initial_xyz
        if 'Unknown reason' in job.job_status[1] and 'change_node' not in job.ess_trsh_methods:
            job.ess_trsh_methods.append('change_node')
            job.troubleshoot_server()
            if job.job_name not in self.running_jobs[label]:
                self.running_jobs[label].append(job.job_name)  # mark as a running job
        elif 'Ran out of disk space' in job.job_status[1]:
            self.output[label]['status'] += '; Error: Could not troubleshoot {job_type} for {label}! ' \
                                            ' The job ran out of disc space on {server}'.format(
                job_type=job_type, label=label, methods=job.ess_trsh_methods, server=job.server)
            logger.error('Could not troubleshoot {job_type} for {label}! The job ran out of disc space on '
                         '{server}'.format(job_type=job_type, label=label, methods=job.ess_trsh_methods,
                                           server=job.server))
        elif job.software == 'gaussian':
            if self.species_dict[label].checkfile is None:
                self.species_dict[label].checkfile = job.checkfile
            if 'Basis set data is not on the checkpoint file' in job.job_status[1]\
                    and 'checkfie=None' not in job.ess_trsh_methods:
                # The checkfile doesn't match the new basis set, remove it and rerun the job
                logger.info('Troubleshooting {type} job in {software} for {label} that failed with '
                            '"Basis set data is not on the checkpoint file" by removing the checkfile.'.format(
                             type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('checkfie=None')
                self.species_dict[label].checkfile = None
                self.species_dict[label].conformer_checkfiles = dict()
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'l103 internal coordinate error' in job.job_status[1]\
                    and 'cartesian' not in job.ess_trsh_methods and job_type == 'opt':
                # try both cartesian and nosymm
                logger.info('Troubleshooting {type} job in {software} for {label} using opt=cartesian with'
                            ' nosyym'.format(type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('cartesian')
                trsh = 'opt=(cartesian,nosymm)'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'unconverged' in job.job_status[1] and 'fine' not in job.ess_trsh_methods and not job.fine:
                # try a fine grid for SCF and integral
                logger.info('Troubleshooting {type} job in {software} for {label} using a fine grid'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('fine')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=True, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'scf=(qc,nosymm)' not in job.ess_trsh_methods:
                # try both qc and nosymm
                logger.info('Troubleshooting {type} job in {software} for {label} using scf=(qc,nosymm)'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('scf=(qc,nosymm)')
                trsh = 'scf=(qc,nosymm)'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'scf=(NDump=30)' not in job.ess_trsh_methods:
                # Allows dynamic dumping for up to N SCF iterations (slower conversion)
                logger.info('Troubleshooting {type} job in {software} for {label} using scf=(NDump=30)'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('scf=(NDump=30)')
                trsh = 'scf=(NDump=30)'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'scf=NoDIIS' not in job.ess_trsh_methods:
                # Switching off Pulay's Direct Inversion
                logger.info('Troubleshooting {type} job in {software} for {label} using scf=NoDIIS'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('scf=NoDIIS')
                trsh = 'scf=NoDIIS'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'int=(Acc2E=14)' not in job.ess_trsh_methods:  # does not work in g03
                # Change integral accuracy (skip everything up to 1E-14 instead of 1E-12)
                logger.info('Troubleshooting {type} job in {software} for {label} using int=(Acc2E=14)'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('int=(Acc2E=14)')
                trsh = 'int=(Acc2E=14)'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'cbs-qb3' not in job.ess_trsh_methods and self.composite_method != 'cbs-qb3':
                # try running CBS-QB3, which is relatively robust
                logger.info('Troubleshooting {type} job in {software} for {label} using CBS-QB3'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('cbs-qb3')
                level_of_theory = 'cbs-qb3'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, job_type='composite',
                             fine=job.fine, ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'scf=nosymm' not in job.ess_trsh_methods:
                # try running w/o considering symmetry
                logger.info('Troubleshooting {type} job in {software} for {label} using scf=nosymm'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('scf=nosymm')
                trsh = 'scf=nosymm'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'memory' not in job.ess_trsh_methods:
                # Increase memory allocation
                max_mem = servers[job.server].get('memory', 128)  # Node memory in GB, default to 128 if not specified
                memory = job.memory_gb * 2 if job.memory_gb * 2 < max_mem * 0.9 else max_mem * 0.9
                logger.info('Troubleshooting {type} job in {software} for {label} using memory: {mem} GB instead of'
                            ' {old} GB'.format(type=job_type, software=job.software, mem=memory, old=job.memory_gb,
                                               label=label))
                job.ess_trsh_methods.append('memory')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, memory=memory, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif self.composite_method != 'cbs-qb3' and 'scf=(qc,nosymm) & CBS-QB3' not in job.ess_trsh_methods:
                # try both qc and nosymm with CBS-QB3
                logger.info('Troubleshooting {type} job in {software} for {label} using scf=(qc,nosymm) with '
                            'CBS-QB3'.format(type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('scf=(qc,nosymm) & CBS-QB3')
                level_of_theory = 'cbs-qb3'
                trsh = 'scf=(qc,nosymm)'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'qchem' not in job.ess_trsh_methods and not job.job_type == 'composite' and\
                    'qchem' in [ess.lower() for ess in self.ess_settings.keys()]:
                # Try QChem
                logger.info('Troubleshooting {type} job using qchem instead of {software} for {label}'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('qchem')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, job_type=job_type, fine=job.fine,
                             software='qchem', ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'molpro' not in job.ess_trsh_methods and not job.job_type == 'composite' \
                    and 'molpro' in [ess.lower() for ess in self.ess_settings.keys()]:
                # Try molpro
                logger.info('Troubleshooting {type} job using molpro instead of {software} for {label}'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('molpro')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, job_type=job_type, fine=job.fine,
                             software='molpro', ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            else:
                logger.error('Could not troubleshoot geometry optimization for {label}! Tried'
                             ' troubleshooting with the following methods: {methods}'.format(
                              label=label, methods=job.ess_trsh_methods))
                self.output[label]['status'] += '; Error: Could not troubleshoot {job_type} for {label}! ' \
                                                ' Tried troubleshooting with the following methods: {methods}'.format(
                                                job_type=job_type, label=label, methods=job.ess_trsh_methods)
                self.save_restart_dict()
        elif job.software == 'qchem':
            if 'max opt cycles reached' in job.job_status[1] and 'max_cycles' not in job.ess_trsh_methods:
                # this is a common error, increase max cycles and continue running from last geometry
                logger.info('Troubleshooting {type} job in {software} for {label} using max_cycles'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('max_cycles')
                trsh = '\n   GEOM_OPT_MAX_CYCLES 250'  # default is 50
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'SCF failed' in job.job_status[1] and 'DIIS_GDM' not in job.ess_trsh_methods:
                # change the SCF algorithm and increase max SCF cycles
                logger.info('Troubleshooting {type} job in {software} for {label} using the DIIS_GDM SCF algorithm'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('DIIS_GDM')
                trsh = '\n   SCF_ALGORITHM DIIS_GDM\n   MAX_SCF_CYCLES 1000'  # default is 50
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'SYM_IGNORE' not in job.ess_trsh_methods:  # symmetry - look in manual, no symm if fails
                # change the SCF algorithm and increase max SCF cycles
                logger.info('Troubleshooting {type} job in {software} for {label} using SYM_IGNORE as well as the '
                            'DIIS_GDM SCF algorithm'.format(type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('SYM_IGNORE')
                trsh = '\n   SCF_ALGORITHM DIIS_GDM\n   MAX_SCF_CYCLES 250\n   SYM_IGNORE     True'
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'b3lyp' not in job.ess_trsh_methods:
                logger.info('Troubleshooting {type} job in {software} for {label} using b3lyp'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('b3lyp')
                # try converging with B3LYP
                level_of_theory = 'b3lyp/6-311++g(d,p)'
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'gaussian' not in job.ess_trsh_methods\
                    and 'gaussian' in [ess.lower() for ess in self.ess_settings.keys()]:
                # Try Gaussian
                logger.info('Troubleshooting {type} job using gaussian instead of {software} for {label}'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('gaussian')
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, job_type=job_type,
                             fine=job.fine, software='gaussian', ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'molpro' not in job.ess_trsh_methods \
                    and 'molpro' in [ess.lower() for ess in self.ess_settings.keys()]:
                # Try molpro
                logger.info('Troubleshooting {type} job using molpro instead of {software} for {label}'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('molpro')
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, job_type=job_type,
                             fine=job.fine, software='molpro', ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            else:
                logger.error('Could not troubleshoot geometry optimization for {label}! Tried'
                             ' troubleshooting with the following methods: {methods}'.format(
                              label=label, methods=job.ess_trsh_methods))
                self.output[label]['status'] += '; Error: Could not troubleshoot {job_type} for {label}! ' \
                                                ' Tried troubleshooting with the following methods: {methods}'.format(
                                                job_type=job_type, label=label, methods=job.ess_trsh_methods)
                self.save_restart_dict()
        elif 'molpro' in job.software:
            if 'additional memory (mW) required' in job.job_status[1]:
                # Increase memory allocation.
                # job.job_status[1] will be for example `'errored: additional memory (mW) required: 996.31'`.
                # The number is the ADDITIONAL memory required
                job.ess_trsh_methods.append('memory')
                add_mem = float(job.job_status[1].split()[-1])  # parse Molpro's requirement in MW
                add_mem = int(math.ceil(add_mem / 100.0)) * 100  # round up to the next hundred
                memory = job.memory_gb + add_mem / 128. + 5  # convert MW to GB, add 5 extra GB (be conservative)
                logger.info('Troubleshooting {type} job in {software} for {label} using memory: {mem} GB instead of '
                            '{old} GB'.format(type=job_type, software=job.software, mem=memory, old=job.memory_gb,
                                              label=label))
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, shift=job.shift, memory=memory,
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'shift' not in job.ess_trsh_methods:
                # try adding a level shift for alpha- and beta-spin orbitals
                logger.info('Troubleshooting {type} job in {software} for {label} using shift'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('shift')
                shift = 'shift,-1.0,-0.5;'
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, shift=shift, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'vdz' not in job.ess_trsh_methods:
                # degrade the basis set
                logger.info('Troubleshooting {type} job in {software} for {label} using vdz'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('vdz')
                trsh = 'vdz'
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, trsh=trsh, ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'vdz & shift' not in job.ess_trsh_methods:
                # try adding a level shift for alpha- and beta-spin orbitals
                logger.info('Troubleshooting {type} job in {software} for {label} using vdz'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('vdz & shift')
                shift = 'shift,-1.0,-0.5;'
                trsh = 'vdz'
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, shift=shift, trsh=trsh,
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'memory' not in job.ess_trsh_methods:
                # Increase memory allocation, also run with a shift
                job.ess_trsh_methods.append('memory')
                memory = servers[job.server]['memory']  # set memory to the value of an entire node (in GB)
                logger.info('Troubleshooting {type} job in {software} for {label} using memory: {mem} GB instead of '
                            '{old} GB'.format(type=job_type, software=job.software, mem=memory, old=job.memory_gb,
                                              label=label))
                shift = 'shift,-1.0,-0.5;'
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, software=job.software,
                             job_type=job_type, fine=job.fine, shift=shift, memory=memory,
                             ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            elif 'gaussian' not in job.ess_trsh_methods\
                    and 'gaussian' in [ess.lower() for ess in self.ess_settings.keys()]:
                # Try Gaussian
                logger.info('Troubleshooting {type} job using gaussian instead of {software} for {label}'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('gaussian')
                self.run_job(label=label, xyz=xyz, level_of_theory=job.level_of_theory, job_type=job_type,
                             fine=job.fine, software='gaussian', ess_trsh_methods=job.ess_trsh_methods,
                             conformer=conformer)
            elif 'qchem' not in job.ess_trsh_methods and 'qchem' in [ess.lower() for ess in self.ess_settings.keys()]:
                # Try QChem
                logger.info('Troubleshooting {type} job using qchem instead of {software} for {label}'.format(
                    type=job_type, software=job.software, label=label))
                job.ess_trsh_methods.append('qchem')
                self.run_job(label=label, xyz=xyz, level_of_theory=level_of_theory, job_type=job_type, fine=job.fine,
                             software='qchem', ess_trsh_methods=job.ess_trsh_methods, conformer=conformer)
            else:
                logger.error('Could not troubleshoot geometry optimization for {label}! Tried'
                             ' troubleshooting with the following methods: {methods}'.format(
                              label=label, methods=job.ess_trsh_methods))
                self.output[label]['status'] += '; Error: Could not troubleshoot {job_type} for {label}! ' \
                                                ' Tried troubleshooting with the following methods: {methods}'.format(
                                                job_type=job_type, label=label, methods=job.ess_trsh_methods)
                self.save_restart_dict()

    def delete_all_species_jobs(self, label):
        """
        Delete all jobs of species/TS represented by `label`
        """
        logger.debug('Deleting all jobs for species {0}'.format(label))
        for job_dict in self.job_dict[label].values():
            for job_name, job in job_dict.items():
                if job_name in self.running_jobs[label]:
                    logger.debug('Deleted job {0}'.format(job_name))
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
                if 'conformer' not in job_description or job_description['conformer'] < 0:
                    self.running_jobs[spc_label].append(job_description['job_name'])
                else:
                    self.running_jobs[spc_label].append('conformer{0}'.format(job_description['conformer']))
                for species in self.species_list:
                    if species.label == spc_label:
                        break
                else:
                    raise SchedulerError('Could not find species {0} in the restart file'.format(spc_label))
                conformer = job_description['conformer'] if 'conformer' in job_description else -1
                job = Job(project=self.project, ess_settings=self.ess_settings, species_name=spc_label,
                          xyz=job_description['xyz'], job_type=job_description['job_type'],
                          level_of_theory=job_description['level_of_theory'], multiplicity=species.multiplicity,
                          charge=species.charge, fine=job_description['fine'], shift=job_description['shift'],
                          is_ts=species.is_ts, memory=job_description['memory'], trsh=job_description['trsh'],
                          ess_trsh_methods=job_description['ess_trsh_methods'], scan=job_description['scan'],
                          pivots=job_description['pivots'], occ=job_description['occ'],
                          project_directory=job_description['project_directory'], job_num=job_description['job_num'],
                          job_server_name=job_description['job_server_name'], job_name=job_description['job_name'],
                          job_id=job_description['job_id'], server=job_description['server'],
                          initial_time=job_description['initial_time'], conformer=conformer,
                          software=job_description['software'], comments=job_description['comments'],
                          scan_trsh=job_description['scan_trsh'], initial_trsh=job_description['initial_trsh'],
                          max_job_time=job_description['max_job_time'], scan_res=job_description['scan_res'],
                          number_of_radicals=species.number_of_radicals)
                if spc_label not in self.job_dict:
                    self.job_dict[spc_label] = dict()
                if job_description['job_type'] not in self.job_dict[spc_label]:
                    if 'conformer' not in job_description or job_description['conformer'] < 0:
                        self.job_dict[spc_label][job_description['job_type']] = dict()
                    elif 'conformers' not in self.job_dict[spc_label]:
                        self.job_dict[spc_label]['conformers'] = dict()
                if 'conformer' not in job_description or job_description['conformer'] < 0:
                    self.job_dict[spc_label][job_description['job_type']][job_description['job_name']] = job
                else:
                    self.job_dict[spc_label]['conformers'][int(job_description['conformer'])] = job
                    # don't generate additional conformers for this species
                    self.dont_gen_confs.append(spc_label)
                self.servers_jobs_ids.append(job.job_id)
        if self.job_dict:
            content = 'Restarting ARC, tracking the following jobs spawned in a previous session:'
            for spc_label in self.job_dict.keys():
                content += '\n' + spc_label + ': '
                for job_type in self.job_dict[spc_label].keys():
                    for job_name in self.job_dict[spc_label][job_type].keys():
                        if job_type != 'conformers':
                            content += job_name + ', '
                        else:
                            content += self.job_dict[spc_label][job_type][job_name].job_name\
                                       + ' (conformer' + str(job_name) + ')' + ', '
            content += '\n\n'
            logger.info(content)

    def save_restart_dict(self):
        """
        Update the restart_dict and save the restart.yml file
        """
        if self.save_restart and self.restart_dict is not None:
            logger.debug('Creating a restart file...')
            self.restart_dict['output'] = self.output
            self.restart_dict['species'] = [spc.as_dict() for spc in self.species_dict.values()]
            self.restart_dict['running_jobs'] = dict()
            for spc in self.species_dict.values():
                if spc.label in self.running_jobs:
                    self.restart_dict['running_jobs'][spc.label] =\
                        [self.job_dict[spc.label][job_name.rsplit('_', 1)[0]][job_name].as_dict()
                         for job_name in self.running_jobs[spc.label] if 'conformer' not in job_name]\
                        + [self.job_dict[spc.label]['conformers'][int(job_name.split('mer')[1])].as_dict()
                           for job_name in self.running_jobs[spc.label] if 'conformer' in job_name]
            logger.debug('Dumping restart dictionary:\n{0}'.format(self.restart_dict))
            save_yaml_file(path=self.restart_path, content=self.restart_dict)

    def make_reaction_labels_info_file(self):
        """A helper function for creating the `reactions labels.info` file"""
        rxn_info_path = os.path.join(self.project_directory, 'output', 'rxns', 'reaction labels.info')
        old_file_path = os.path.join(os.path.join(self.project_directory, 'output', 'rxns', 'reaction labels.old.info'))
        if os.path.isfile(rxn_info_path):
            if os.path.isfile(old_file_path):
                os.remove(old_file_path)
            shutil.copy(rxn_info_path, old_file_path)
            os.remove(rxn_info_path)
        if not os.path.exists(os.path.dirname(rxn_info_path)):
            os.makedirs(os.path.dirname(rxn_info_path))
        with open(rxn_info_path, 'w') as f:
            f.write(str('Reaction labels and respective TS labels:\n\n'))
        return rxn_info_path

    def determine_adaptive_level(self, original_level_of_theory, job_type, heavy_atoms):
        """
        Determine the level of theory to be used according to the job type and number of heavy atoms.
        self.adaptive_levels is a dictionary of levels of theory for ranges of the number of heavy atoms in the
        molecule. Keys are tuples of (min_num_atoms, max_num_atoms), values are dictionaries with 'optfreq' and 'sp'
        as keys and levels of theory as values. 'inf' is accepted an max_num_atoms.
        """
        if self.adaptive_levels is not None:
            for constraint, level_dict in self.adaptive_levels.items():
                if constraint[1] == 'inf' and heavy_atoms >= constraint[0]:
                    break
                elif constraint[1] >= heavy_atoms >= constraint[0]:
                    break
            else:
                raise SchedulerError('Could not determine adaptive level of theory for {0} heavy atoms using '
                                     'the following adaptive levels:\n{1}'.format(heavy_atoms, self.adaptive_levels))
            if job_type in ['opt', 'freq', 'optfreq', 'composite']:
                if 'optfreq' not in level_dict:
                    raise SchedulerError("Could not find the 'optfreq' key in the adaptive levels dictionary for "
                                         "{0} heavy atoms. Got:\n{1}".format(heavy_atoms, self.adaptive_levels))
                return level_dict['optfreq']
            elif job_type == 'sp':
                if 'sp' not in level_dict:
                    raise SchedulerError("Could not find the 'sp' key in the adaptive levels dictionary for "
                                         "{0} heavy atoms. Got:\n{1}".format(heavy_atoms, self.adaptive_levels))
                return level_dict['sp']
            else:
                # for any other job type use the original level of theory regardless of the number of atoms
                return original_level_of_theory


def sum_time_delta(timedelta_list):
    """A helper function for summing datetime.timedelta objects"""
    result = datetime.timedelta(0)
    for timedelta in timedelta_list:
        if timedelta is not None:
            result += timedelta
    return result
