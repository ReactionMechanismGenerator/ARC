#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)
import logging
import sys
import os
import time
import re
import shutil
from distutils.spawn import find_executable
from IPython.display import display

from rmgpy.species import Species
from rmgpy.reaction import Reaction

import arc.rmgdb as rmgdb
from arc.settings import arc_path, default_levels_of_theory, check_status_command, servers
from arc.scheduler import Scheduler, time_lapse
from arc.exceptions import InputError, SettingsError, SpeciesError
from arc.species import ARCSpecies
from arc.reaction import ARCReaction
from arc.processor import Processor
from arc.job.ssh import SSH_Client

##################################################################


class ARC(object):
    """
    Main ARC object.
    The software is currently configured to run on a local computer, sending jobs / commands to one or more servers.

    The attributes are:

    ====================== ========== ==================================================================================
    Attribute              Type       Description
    ====================== ========== ==================================================================================
    `project`              ``str``    The project's name. Used for naming the working directory.
    `project_directory`    ``str``    The path to the project directory
    `arc_species_list`     ``list``   A list of ARCSpecies objects (each entry should represent either a stable well,
                                        TS guesses are given in the arc_rxn_list)
    'arc_rxn_list`         ``list``   A list of ARCReaction objects
    `conformer_level`      ``str``    Level of theory for conformer searches
    `ts_guess_level`       ``str``    Level of theory for comparisons of TS guesses between different methods
    `composite_method'     ``str``    Composite method
    `opt_level`            ``str``    Level of theory for geometry optimization
    `freq_level`           ``str``    Level of theory for frequency calculations
    `sp_level`             ``str``    Level of theory for single point calculations
    `scan_level`           ``str``    Level of theory for rotor scans
    `output`               ``dict``   Output dictionary with status and final QM file paths for all species
                                        Only used for restarting, the actual object used is in the Scheduler class
    `fine`                 ``bool``   Whether or not to use a fine grid for opt jobs (spawns an additional job)
    `generate_conformers`  ``bool``   Whether or not to generate conformers when an initial geometry is given
    `scan_rotors`          ``bool``   Whether or not to perform rotor scans
    `use_bac`              ``bool``   Whether or not to use bond additivity corrections for thermo calculations
    `model_chemistry`      ``list``   The model chemistry in Arkane for energy corrections (AE, BAC).
                                        This can be usually determined automatically.
    `settings`             ``dict``   A dictionary of available servers and software
    `ess_settings`         ``dict``   An optional input parameter: a dictionary relating ESS to servers
    `initial_trsh`         ``dict``   Troubleshooting methods to try by default. Keys are server names, values are trshs
    't0'                   ``float``  Initial time when the project was spawned
    `execution_time`       ``str``    Overall execution time
    `lib_long_desc`        ``str``    A multiline description of levels of theory for the outputted RMG libraries
    `running_jobs`         ``dict``   A dictionary of jobs submitted in a precious ARC instance, used for restarting ARC
    `t_min`                ``tuple``  The minimum temperature for kinetics computations, e.g., (500, str('K'))
    `t_max`                ``tuple``  The maximum temperature for kinetics computations, e.g., (3000, str('K'))
    `t_count`              ``int``    The number of temperature points between t_min and t_max for kinetics computations
    `rmgdb`                ``RMGDatabase``  The RMG database object
    ====================== ========== ==================================================================================

    `level_of_theory` is a string representing either sp//geometry levels or a composite method, e.g. 'CBS-QB3',
                                                 'CCSD(T)-F12a/aug-cc-pVTZ//B3LYP/6-311++G(3df,3pd)'...
    """
    def __init__(self, input_dict=None, project=None, arc_species_list=None, arc_rxn_list=None, level_of_theory='',
                 conformer_level='', composite_method='', opt_level='', freq_level='', sp_level='', scan_level='',
                 ts_guess_level='', fine=True, generate_conformers=True, scan_rotors=True, use_bac=True,
                 model_chemistry='', ess_settings=None, initial_trsh=None, t_min=None, t_max=None, t_count=None,
                 verbose=logging.INFO, project_directory=None):

        self.__version__ = '0.1'
        self.verbose = verbose
        self.ess_settings = ess_settings
        self.settings = dict()
        self.output = dict()
        self.running_jobs = dict()
        self.lib_long_desc = ''
        self.unique_species_labels = list()
        self.rmgdb = rmgdb.make_rmg_database_object()

        if input_dict is None:
            if project is None:
                raise ValueError('A project name must be provided for a new project')
            self.project = project
            self.t_min = t_min
            self.t_max = t_max
            self.t_count = t_count
            self.project_directory = project_directory if project_directory is not None\
                else os.path.join(arc_path, 'Projects', self.project)
            if not os.path.exists(self.project_directory):
                os.makedirs(self.project_directory)
            self.initialize_log(verbose=self.verbose, log_file=os.path.join(self.project_directory, 'arc.log'))
            self.t0 = time.time()  # init time
            self.execution_time = None
            self.initial_trsh = initial_trsh if initial_trsh is not None else dict()
            self.determine_remote()
            self.fine = fine
            self.generate_conformers = generate_conformers
            self.scan_rotors = scan_rotors
            self.use_bac = use_bac
            self.model_chemistry = model_chemistry
            if self.model_chemistry:
                logging.info('Using {0} as model chemistry for energy corrections in Arkane'.format(self.model_chemistry))
            if not self.fine:
                logging.info('\n')
                logging.warning('Not using a fine grid for geometry optimization jobs')
                logging.info('\n')
            if not self.scan_rotors:
                logging.info('\n')
                logging.warning("Not running rotor scans."
                                " This might compromise finding the best conformer, as dihedral angles won't be"
                                " corrected. Also, entropy won't be accurate.")
                logging.info('\n')

            if level_of_theory.count('//') > 1:
                raise InputError('Level of theory seems wrong. It should either be a composite method (like CBS-QB3)'
                                 ' or be of the form sp//geometry, e.g., CCSD(T)-F12/avtz//wB97x-D3/6-311++g**.'
                                 ' Got: {0}'.format(level_of_theory))

            if conformer_level:
                logging.info('Using {0} for refined conformer searches (after filtering via force fields)'.format(
                    conformer_level))
                self.conformer_level = conformer_level.lower()
            elif self.generate_conformers:
                self.conformer_level = default_levels_of_theory['conformer'].lower()
                logging.info('Using default level {0} for refined conformer searches (after filtering via force'
                             ' fields)'.format(default_levels_of_theory['conformer']))
            else:
                self.conformer_level = ''
            if ts_guess_level:
                logging.info('Using {0} for TS guesses comparison of different methods'.format(ts_guess_level))
                self.ts_guess_level = ts_guess_level.lower()
            else:
                self.ts_guess_level = default_levels_of_theory['ts_guesses'].lower()
                logging.info('Using default level {0} for TS guesses comparison of different methods'.format(
                    default_levels_of_theory['ts_guesses']))

            if level_of_theory:
                if '/' not in level_of_theory:  # assume this is a composite method
                    self.composite_method = level_of_theory.lower()
                    logging.info('Using composite method {0}'.format(self.composite_method))
                    if freq_level:
                        self.freq_level = freq_level.lower()
                        logging.info('Using {0} for frequency calculations'.format(self.freq_level))
                    else:
                        # This is a composite method
                        self.freq_level = default_levels_of_theory['freq_for_composite'].lower()
                        logging.info('Using default level {0} for frequency calculations after composite jobs'.format(
                            self.freq_level))
                elif '//' in level_of_theory:
                    self.composite_method = ''
                    self.opt_level = level_of_theory.lower().split('//')[1]
                    self.freq_level = level_of_theory.lower().split('//')[1]
                    self.sp_level = level_of_theory.lower().split('//')[0]
                    logging.info('Using {0} for geometry optimizations'.format(level_of_theory.split('//')[1]))
                    logging.info('Using {0} for frequency calculations'.format(level_of_theory.split('//')[1]))
                    logging.info('Using {0} for single point calculations'.format(level_of_theory.split('//')[0]))
                elif '/' in level_of_theory and '//' not in level_of_theory:
                    # assume this is not a composite method, and the user meant to run opt, freq and sp at this level.
                    # running an sp after opt at the same level is meaningless, but doesn't matter much also...
                    self.composite_method = ''
                    self.opt_level = level_of_theory.lower()
                    self.freq_level = level_of_theory.lower()
                    self.sp_level = level_of_theory.lower()
                    logging.info('Using {0} for geometry optimizations'.format(level_of_theory))
                    logging.info('Using {0} for frequency calculations'.format(level_of_theory))
                    logging.info('Using {0} for single point calculations'.format(level_of_theory))
            else:
                self.composite_method = composite_method.lower()
                if self.composite_method:
                    if level_of_theory and level_of_theory.lower != self.composite_method:
                        raise InputError('Specify either composite_method or level_of_theory')
                    logging.info('Using composite method {0}'.format(composite_method))
                    if self.composite_method == 'cbs-qb3':
                        self.model_chemistry = self.composite_method
                        logging.info('Using {0} as model chemistry for energy corrections in Arkane'.format(
                            self.model_chemistry))
                    elif self.use_bac:
                        raise InputError('Could not determine model chemistry to use for composite method {0}'.format(
                            self.composite_method))

                if opt_level:
                    self.opt_level = opt_level.lower()
                    logging.info('Using {0} for geometry optimizations'.format(self.opt_level))
                elif not self.composite_method:
                    # self.opt_level = 'wb97x-d3/def2-tzvpd'
                    # logging.info('Using wB97x-D3/def2-TZVPD for geometry optimizations')
                    self.opt_level = default_levels_of_theory['opt'].lower()
                    logging.info('Using default level {0} for geometry optimizations'.format(self.opt_level))
                else:
                    self.opt_level = ''

                if freq_level:
                    self.freq_level = freq_level.lower()
                    logging.info('Using {0} for frequency calculations'.format(self.freq_level))
                elif not self.composite_method:
                    if opt_level:
                        self.freq_level = opt_level.lower()
                        logging.info('Using user-defined opt level {0} for frequency calculations as well'.format(
                            self.freq_level))
                    else:
                        # self.freq_level = 'wb97x-d3/def2-tzvpd'
                        # logging.info('Using wB97x-D3/def2-TZVPD for frequency calculations')
                        self.freq_level = default_levels_of_theory['freq'].lower()
                        logging.info('Using default level {0} for frequency calculations'.format(self.freq_level))
                else:
                    # This is a composite method
                    self.freq_level = default_levels_of_theory['freq_for_composite'].lower()
                    logging.info('Using default level {0} for frequency calculations after composite jobs'.format(
                        self.freq_level))

                if sp_level:
                    self.sp_level = sp_level.lower()
                    logging.info('Using {0} for single point calculations'.format(self.sp_level))
                elif not self.composite_method:
                    self.sp_level = default_levels_of_theory['sp'].lower()
                    logging.info('Using default level {0} for single point calculations'.format(self.sp_level))
                else:
                    # It's a composite method, no need in explicit sp
                    self.sp_level = ''

            if scan_level:
                self.scan_level = scan_level.lower()
                if self.scan_rotors:
                    logging.info('Using {0} for rotor scans'.format(self.scan_level))
            elif self.scan_rotors:
                if not self.composite_method:
                    self.scan_level = default_levels_of_theory['scan'].lower()
                    logging.info('Using default level {0} for rotor scans'.format(self.scan_level))
                else:
                    # This is a composite method
                    self.freq_level = default_levels_of_theory['scan_for_composite'].lower()
                    logging.info('Using default level {0} for scan calculations after composite jobs'.format(
                        self.freq_level))
            else:
                self.scan_level = ''

            if self.composite_method:
                self.opt_level = ''
                self.sp_level = ''

            self.arc_species_list = arc_species_list if arc_species_list is not None else list()
            converted_species_list = list()
            indices_to_pop = []
            for i, spc in enumerate(self.arc_species_list):
                if isinstance(spc, Species):
                    if not spc.label:
                        raise InputError('Missing label on RMG Species object {0}'.format(spc))
                    indices_to_pop.append(i)
                    arc_spc = ARCSpecies(is_ts=False, rmg_species=spc)  # assuming an RMG Species is not a TS
                    converted_species_list.append(arc_spc)
                elif not isinstance(spc, ARCSpecies):
                    raise ValueError('A species should either be an `ARCSpecies` object or an RMG `Species` object.'
                                     ' Got: {0} for {1}'.format(type(spc), spc.label))
            for i in reversed(range(len(self.arc_species_list))):  # pop from the end, so other indices won't change
                if i in indices_to_pop:
                    self.arc_species_list.pop(i)
            self.arc_species_list.extend(converted_species_list)
            for arc_spc in self.arc_species_list:
                if arc_spc.label not in self.unique_species_labels:
                    self.unique_species_labels.append(arc_spc.label)
                else:
                    raise ValueError('Species label {0} is not unique'.format(arc_spc.label))
            self.arc_rxn_list = arc_rxn_list if arc_rxn_list is not None else list()
            converted_rxn_list = list()
            indices_to_pop = []
            for i, rxn in enumerate(self.arc_rxn_list):
                if isinstance(rxn, Reaction):
                    if not rxn.reactants or not rxn.products:
                        raise InputError('Missing reactants and/or products in RMG Reaction object {0}'.format(rxn))
                    indices_to_pop.append(i)
                    arc_rxn = ARCReaction(rmg_reaction=rxn)
                    converted_rxn_list.append(arc_rxn)
                    for spc in rxn.reactants + rxn.products:
                        if not isinstance(spc, Species):
                            raise InputError('All reactants and procucts of an RMG Reaction have to be RMG Species'
                                             ' objects. Got: {0} in reaction {1}'.format(type(spc), rxn))
                        if not spc.label:
                            raise InputError('Missing label on RMG Species object {0} in reaction {1}'.format(
                                spc, rxn))
                        if spc.label not in self.unique_species_labels:
                            # Add species participating in an RMG Reaction to arc_species_list if not already there
                            # We assume each species has a unique label
                            self.arc_species_list.append(ARCSpecies(is_ts=False, rmg_species=spc))
                            self.unique_species_labels.append(spc.label)
                elif not isinstance(rxn, ARCReaction):
                    raise ValueError('A reaction should either be an `ARCReaction` object or an RMG `Reaction` object.'
                                     ' Got: {0} for {1}'.format(type(rxn), rxn.label))
            for i in reversed(range(len(self.arc_rxn_list))):  # pop from the end, so other indices won't change
                if i in indices_to_pop:
                    self.arc_rxn_list.pop(i)
            self.arc_rxn_list.extend(converted_rxn_list)
            rxn_index = 0
            for arc_rxn in self.arc_rxn_list:
                arc_rxn.index = rxn_index
                rxn_index += 1

        else:
            # ARC is run from an input or a restart file.
            # Read the input_dict
            self.from_dict(input_dict=input_dict, project=project, project_directory=project_directory)
        self.restart_dict = self.as_dict()
        self.determine_model_chemistry()
        self.scheduler = None

        # make a backup copy of the restart file if it exists (but don't save an updated one just yet)
        if os.path.isfile(os.path.join(self.project_directory, 'restart.yml')):
            shutil.copy(os.path.join(self.project_directory, 'restart.yml'),
                        os.path.join(self.project_directory, 'restart.old.yml'))

    def as_dict(self):
        """
        A helper function for dumping this object as a dictionary in a YAML file for restarting ARC
        """
        restart_dict = dict()
        restart_dict['project'] = self.project
        restart_dict['ess_settings'] = self.settings
        restart_dict['fine'] = self.fine
        restart_dict['generate_conformers'] = self.generate_conformers
        restart_dict['scan_rotors'] = self.scan_rotors
        restart_dict['use_bac'] = self.use_bac
        restart_dict['model_chemistry'] = self.model_chemistry
        restart_dict['composite_method'] = self.composite_method
        restart_dict['conformer_level'] = self.conformer_level
        restart_dict['ts_guess_level'] = self.ts_guess_level
        restart_dict['scan_level'] = self.scan_level
        if not self.composite_method:
            restart_dict['opt_level'] = self.opt_level
            restart_dict['freq_level'] = self.freq_level
            restart_dict['sp_level'] = self.sp_level
        if self.initial_trsh:
            restart_dict['initial_trsh'] = self.initial_trsh
        restart_dict['species'] = [spc.as_dict() for spc in self.arc_species_list]
        restart_dict['reactions'] = [rxn.as_dict() for rxn in self.arc_rxn_list]
        restart_dict['output'] = self.output  # if read from_dict then it has actual values
        restart_dict['running_jobs'] = self.running_jobs  # if read from_dict then it has actual values
        restart_dict['t_min'] = self.t_min
        restart_dict['t_max'] = self.t_max
        restart_dict['t_count'] = self.t_count
        return restart_dict

    def from_dict(self, input_dict, project=None, project_directory=None):
        """
        A helper function for loading this object from a dictionary in a YAML file for restarting ARC
        If `project` name and `ess_settings` are given as well to __init__, they will override the respective values
        in the restart dictionary.
        """
        if project is None and 'project' not in input_dict:
            raise InputError('A project name must be given')
        self.project = project if project is not None else input_dict['project']
        self.project_directory = project_directory if project_directory is not None \
            else os.path.join(arc_path, 'Projects', self.project)
        if not os.path.exists(self.project_directory):
            os.makedirs(self.project_directory)
        self.initialize_log(verbose=self.verbose, log_file=os.path.join(self.project_directory, 'arc.log'))
        self.t0 = time.time()  # init time
        self.execution_time = None
        self.verbose = input_dict['verbose'] if 'verbose' in input_dict else self.verbose

        if self.ess_settings is not None:
            self.settings['ssh'] = True
            for ess, server in self.ess_settings.items():
                if ess.lower() not in ['gaussian', 'qchem', 'molpro']:
                    raise SettingsError('Recognized ESS software are Gaussian, QChem or Molpro. Got: {0}'.format(ess))
                if server.lower() not in servers:
                    server_names = [name for name in servers]
                    raise SettingsError('Recognized servers are {0}. Got: {1}'.format(server_names, servers))
                self.settings[ess.lower()] = server.lower()
        elif 'ess_settings' in input_dict:
            self.settings = input_dict['ess_settings']
            self.settings['ssh'] = True
        else:
            self.determine_remote()
        logging.info('\nUsing the following settings: {0}\n'.format(self.settings))

        self.output = input_dict['output'] if 'output' in input_dict else dict()
        if self.output:
            for label, spc_output in self.output.items():
                for key, val in spc_output.items():
                    if key in ['geo', 'freq', 'sp', 'composite']:
                        if not os.path.isfile(val):
                            raise SpeciesError('Could not find {0} output file for species {1}'.format(key, label))
                    elif key == 'rotors':
                        for rotor_num, rotor_dict in val.items():
                            if not os.path.isfile(val['path']):
                                raise SpeciesError('Could not find {0} output file for rotor {1} of species {2}'.format(
                                    key, rotor_num, label))
        self.running_jobs = input_dict['running_jobs'] if 'running_jobs' in input_dict else dict()
        logging.debug('output dictionary successfully parsed:\n{0}'.format(self.output))
        self.t_min = input_dict['t_min'] if 't_min' in input_dict else None
        self.t_max = input_dict['t_max'] if 't_max' in input_dict else None
        self.t_count = input_dict['t_count'] if 't_count' in input_dict else None

        self.initial_trsh = input_dict['initial_trsh'] if 'initial_trsh' in input_dict else dict()
        self.fine = input_dict['fine'] if 'fine' in input_dict else True
        self.generate_conformers = input_dict['generate_conformers'] if 'generate_conformers' in input_dict else True
        self.scan_rotors = input_dict['scan_rotors'] if 'scan_rotors' in input_dict else True
        self.use_bac = input_dict['use_bac'] if 'use_bac' in input_dict else True
        self.model_chemistry = input_dict['model_chemistry'] if 'use_bac' in input_dict\
                                                                and input_dict['use_bac'] else ''
        if not self.fine:
            logging.info('\n')
            logging.warning('Not using a fine grid for geometry optimization jobs')
            logging.info('\n')
        if not self.scan_rotors:
            logging.info('\n')
            logging.warning("Not running rotor scans."
                            " This might compromise finding the best conformer, as dihedral angles won't be"
                            " corrected. Also, entropy won't be accurate.")
            logging.info('\n')

        if 'conformer_level' in input_dict:
            self.conformer_level = input_dict['conformer_level'].lower()
            logging.info('Using {0} for refined conformer searches (after filtering via force fields)'.format(
                self.conformer_level))
        elif self.generate_conformers:
            self.conformer_level = default_levels_of_theory['conformer'].lower()
            logging.info('Using default level {0} for refined conformer searches (after filtering via force'
                         ' fields)'.format(default_levels_of_theory['conformer']))

        if 'ts_guess_level' in input_dict:
            self.ts_guess_level = input_dict['ts_guess_level'].lower()
            logging.info('Using {0} for TS guesses comparison of different methods'.format(self.ts_guess_level))
        else:
            self.ts_guess_level = default_levels_of_theory['ts_guesses'].lower()
            logging.info('Using default level {0} for TS guesses comparison of different methods'.format(
                default_levels_of_theory['ts_guesses']))

        self.composite_method = input_dict['composite_method'].lower() if 'composite_method' in input_dict else ''
        if self.composite_method:
            logging.info('Using composite method {0}'.format(self.composite_method))
            if self.composite_method == 'cbs-qb3':
                self.model_chemistry = self.composite_method
                logging.info('Using {0} as model chemistry for energy corrections in Arkane'.format(
                    self.model_chemistry))
            elif self.use_bac:
                raise InputError('Could not determine model chemistry to use for composite method {0}'.format(
                    self.composite_method))

        if 'scan_level' in input_dict:
            self.scan_level = input_dict['scan_level'].lower()
            if self.scan_rotors:
                logging.info('Using {0} for rotor scans'.format(self.scan_level))
        elif self.scan_rotors:
            if not self.composite_method:
                self.scan_level = default_levels_of_theory['scan'].lower()
                logging.info('Using default level {0} for rotor scans'.format(self.scan_level))
            else:
                # This is a composite method
                self.freq_level = default_levels_of_theory['scan_for_composite'].lower()
                logging.info('Using default level {0} for scan calculations after composite jobs'.format(
                    self.freq_level))
        else:
            self.scan_level = ''

        if 'opt_level' in input_dict:
            self.opt_level = input_dict['opt_level'].lower()
            logging.info('Using {0} for geometry optimizations'.format(self.opt_level))
        elif not self.composite_method:
            self.opt_level = default_levels_of_theory['opt'].lower()
            logging.info('Using default level {0} for geometry optimizations'.format(self.opt_level))
        else:
            self.opt_level = ''

        if 'freq_level' in input_dict:
            self.freq_level = input_dict['freq_level'].lower()
        elif not self.composite_method:
            if 'opt_level' in input_dict:
                self.freq_level = input_dict['opt_level'].lower()
                logging.info('Using user-defined opt level {0} for frequency calculations as well'.format(
                    self.freq_level))
            else:
                self.freq_level = default_levels_of_theory['freq'].lower()
                logging.info('Using default level {0} for frequency calculations'.format(self.freq_level))
        else:
            # This is a composite method
            self.freq_level = default_levels_of_theory['freq_for_composite'].lower()
            logging.info('Using default level {0} for frequency calculations after composite jobs'.format(
                self.freq_level))

        if 'sp_level' in input_dict:
            self.sp_level = input_dict['sp_level'].lower()
            logging.info('Using {0} for single point calculations'.format(self.sp_level))
        elif not self.composite_method:
            self.sp_level = default_levels_of_theory['sp'].lower()
            logging.info('Using default level {0} for single point calculations'.format(self.sp_level))
        else:
            # It's a composite method, no need in explicit sp
            self.sp_level = ''
        if 'species' in input_dict:
            self.arc_species_list = [ARCSpecies(species_dict=spc_dict) for spc_dict in input_dict['species']]
        else:
            self.arc_species_list = list()
        if 'reactions' in input_dict:
            self.arc_rxn_list = [ARCReaction(reaction_dict=rxn_dict) for rxn_dict in input_dict['reactions']]
            for i, rxn in enumerate(self.arc_rxn_list):
                rxn.index = i
        else:
            self.arc_rxn_list = list()

    def execute(self):
        logging.info('\n')
        for species in self.arc_species_list:
            if not isinstance(species, ARCSpecies):
                raise ValueError('All species in arc_species_list must be ARCSpecies objects.'
                                 ' Got {0}'.format(type(species)))
            if species.is_ts:
                logging.info('Considering transition state: {0}'.format(species.label))
            else:
                logging.info('Considering species: {0}'.format(species.label))
                if species.mol is not None:
                    display(species.mol)
        logging.info('\n')
        for rxn in self.arc_rxn_list:
            if not isinstance(rxn, ARCReaction):
                raise ValueError('All reactions in arc_rxn_list must be ARCReaction objects.'
                                 ' Got {0}'.format(type(rxn)))

        self.scheduler = Scheduler(project=self.project, species_list=self.arc_species_list, rxn_list=self.arc_rxn_list,
                                   composite_method=self.composite_method, conformer_level=self.conformer_level,
                                   opt_level=self.opt_level, freq_level=self.freq_level, sp_level=self.sp_level,
                                   scan_level=self.scan_level, ts_guess_level=self.ts_guess_level ,fine=self.fine,
                                   settings=self.settings, generate_conformers=self.generate_conformers,
                                   scan_rotors=self.scan_rotors, initial_trsh=self.initial_trsh, rmgdatabase=self.rmgdb,
                                   restart_dict=self.restart_dict, project_directory=self.project_directory)
        prc = Processor(project=self.project, project_directory=self.project_directory,
                        species_dict=self.scheduler.species_dict, rxn_list=self.scheduler.rxn_list,
                        output=self.scheduler.output, use_bac=self.use_bac, model_chemistry=self.model_chemistry,
                        lib_long_desc=self.lib_long_desc, rmgdatabase=self.rmgdb, t_min=self.t_min, t_max=self.t_max,
                        t_count=self.t_count)
        prc.process()
        self.save_project_info_file()
        self.summary()
        self.log_footer()

    def save_project_info_file(self):
        d, h, m, s = time_lapse(t0=self.t0)
        self.execution_time = '{0}{1:02.0f}:{2:02.0f}:{3:02.0f}'.format(d, h, m, s)
        path = os.path.join(self.project_directory, '{0}.info'.format(self.project))
        if os.path.exists(path):
            os.remove(path)
        if self.fine:
            fine_txt = '(using a fine grid)'
        else:
            fine_txt = '(NOT using a fine grid)'

        txt = ''
        txt += 'ARC project {0}\n\nLevels of theory used:\n\n'.format(self.project)
        txt += 'Conformers:       {0}\n'.format(self.conformer_level)
        txt += 'TS guesses:       {0}\n'.format(self.ts_guess_level)
        if self.composite_method:
            txt += 'Composite method: {0} {1}\n'.format(self.composite_method, fine_txt)
            txt += 'Frequencies:      {0}\n'.format(self.freq_level)
        else:
            txt += 'Optimization:     {0} {1}\n'.format(self.opt_level, fine_txt)
            txt += 'Frequencies:      {0}\n'.format(self.freq_level)
            txt += 'Single point:     {0}\n'.format(self.sp_level)
        if self.scan_rotors:
            txt += 'Rotor scans:      {0}\n'.format(self.scan_level)
        else:
            txt += 'Not scanning rotors\n'
        if self.use_bac:
            txt += 'Using bond additivity corrections for thermo\n'
        else:
            txt += 'NOT using bond additivity corrections for thermo\n'
        if self.initial_trsh:
            txt += 'Using an initial troubleshooting method "{0}"'.format(self.initial_trsh)
        txt += '\nUsing the following settings: {0}\n'.format(self.settings)
        txt += '\nConsidered the following species and TSs:\n'
        for species in self.arc_species_list:
            if species.is_ts:
                if species.execution_time is not None:
                    txt += 'TS {0} (execution time: {1})\n'.format(species.label, species.execution_time)
                else:
                    txt += 'TS {0} (Failed)\n'.format(species.label)
            else:
                if species.execution_time is not None:
                    txt += 'Species {0} (execution time: {1})\n'.format(species.label, species.execution_time)
                else:
                    txt += 'Species {0} (Failed!)\n'.format(species.label)
        if self.arc_rxn_list:
            for rxn in self.arc_rxn_list:
                txt += 'Considered reaction: {0}\n'.format(rxn.label)
        txt += '\nOverall execution time: {0}'.format(self.execution_time)
        txt += '\n'

        with open(path, 'w') as f:
            f.write(txt)
        self.lib_long_desc = txt

    def summary(self):
        """
        Report status and data of all species / reactions
        """
        logging.info('\n\n\nAll jobs terminated. Summary for project {0}:\n'.format(self.project))
        for label, output in self.scheduler.output.items():
            if 'ALL converged' in output['status']:
                logging.info('Species {0} converged successfully'.format(label))
            else:
                logging.info('Species {0} failed with status:\n  {1}'.format(label, output['status']))

    def initialize_log(self, verbose=logging.INFO, log_file=None):
        """
        Set up a logger for ARC to use to print output to stdout.
        The `verbose` parameter is an integer specifying the amount of log text seen
        at the console; the levels correspond to those of the :data:`logging` module.
        """
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(verbose)

        # Use custom level names for cleaner log output
        logging.addLevelName(logging.CRITICAL, 'Critical: ')
        logging.addLevelName(logging.ERROR, 'Error: ')
        logging.addLevelName(logging.WARNING, 'Warning: ')
        logging.addLevelName(logging.INFO, '')
        logging.addLevelName(logging.DEBUG, '')
        logging.addLevelName(0, '')

        # Create formatter and add to handlers
        formatter = logging.Formatter('%(levelname)s%(message)s')

        # Remove old handlers before adding ours
        while logger.handlers:
            logger.removeHandler(logger.handlers[0])

        # Create console handler; send everything to stdout rather than stderr
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(verbose)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Create file handler
        if log_file:
            if os.path.isfile(log_file):
                old_file = os.path.join(os.path.dirname(log_file), 'arc.old.log')
                if os.path.isfile(old_file):
                    os.remove(old_file)
                shutil.copy(log_file, old_file)
                os.remove(log_file)
            fh = logging.FileHandler(filename=log_file)
            fh.setLevel(verbose)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            self.log_header()

    def log_header(self, level=logging.INFO):
        """
        Output a header containing identifying information about CanTherm to the log.
        """
        logging.log(level, 'ARC execution initiated on {0}'.format(time.asctime()))
        logging.log(level, '')
        logging.log(level, '###############################################################')
        logging.log(level, '#                                                             #')
        logging.log(level, '#                 Automatic Rate Calculator                   #')
        logging.log(level, '#                            ARC                              #')
        logging.log(level, '#                                                             #')
        logging.log(level, '#   Version: {0}{1}                                       #'.format(
            self.__version__, ' ' * (10 - len(self.__version__))))
        logging.log(level, '#                                                             #')
        logging.log(level, '###############################################################')
        logging.log(level, '')
        logging.info('Starting project {0}'.format(self.project))

    def log_footer(self, level=logging.INFO):
        """
        Output a footer to the log.
        """
        logging.log(level, '')
        logging.log(level, 'Total execution time: {0}'.format(self.execution_time))
        logging.log(level, 'ARC execution terminated on {0}'.format(time.asctime()))

    def determine_model_chemistry(self):
        if self.model_chemistry:
            self.model_chemistry = self.model_chemistry.lower()
            if self.model_chemistry not in ['cbs-qb3', 'cbs-qb3-paraskevas', 'ccsd(t)-f12/cc-pvdz-f12',
                                            'ccsd(t)-f12/cc-pvtz-f12', 'ccsd(t)-f12/cc-pvqz-f12',
                                            'b3lyp/cbsb7', 'b3lyp/6-311g(2d,d,p)', 'b3lyp/6-311+g(3df,2p)',
                                            'b3lyp/6-31g**']:
                logging.warn('No bond additivity corrections (BAC) are available in Arkane for "model chemistry"'
                             ' {0}. As a result, thermodynamic parameters are expected to be inaccurate. Make sure that'
                             ' atom energy corrections (AEC) were supplied or are available in Arkane to avoid'
                             ' error.'.format(self.model_chemistry))
        else:
            # model chemistry was not given, try to determine it from the sp_level
            model_chemistry = ''
            if not self.composite_method:
                sp_level = self.sp_level.lower()
            else:
                sp_level = self.composite_method
            sp_level = sp_level.replace('f12a', 'f12').replace('f12b', 'f12')
            if sp_level in ['ccsd(t)-f12/cc-pvdz', 'ccsd(t)-f12/cc-pvtz', 'ccsd(t)-f12/cc-pvqz']:
                logging.warning('Using model chemistry {0} based on sp level {1}.'.format(
                    sp_level + '-f12', sp_level))
                model_chemistry = sp_level + '-f12'
            elif not model_chemistry and sp_level in ['cbs-qb3', 'cbs-qb3-paraskevas', 'ccsd(t)-f12/cc-pvdz-f12',
                                                      'ccsd(t)-f12/cc-pvtz-f12', 'ccsd(t)-f12/cc-pvqz-f12',
                                                      'b3lyp/cbsb7', 'b3lyp/6-311g(2d,d,p)', 'b3lyp/6-311+g(3df,2p)',
                                                      'b3lyp/6-31g**']:
                model_chemistry = sp_level
            elif self.use_bac:
                raise InputError('Could not determine appropriate model chemistry to be used in Arkane for'
                                 ' thermochemical parameter calculations. Either turn off the "use_bac" flag'
                                 ' (and BAC will not be used), or specify a correct model chemistry. For a'
                                 ' comprehensive model chemistry list allowed in Arkane, see the Arkane documentation'
                                 ' on the RMG website, rmg.mit.edu.')
            else:
                # use_bac is False, and no model chemistry was specified
                if sp_level in ['m06-2x/cc-pvtz', 'g3', 'm08so/mg3s*', 'klip_1', 'klip_2', 'klip_3', 'klip_2_cc',
                                'ccsd(t)-f12/cc-pvdz-f12_h-tz', 'ccsd(t)-f12/cc-pvdz-f12_h-qz',
                                'ccsd(t)-f12/cc-pvdz-f12', 'ccsd(t)-f12/cc-pvtz-f12', 'ccsd(t)-f12/cc-pvqz-f12',
                                'ccsd(t)-f12/cc-pcvdz-f12', 'ccsd(t)-f12/cc-pcvtz-f12', 'ccsd(t)-f12/cc-pcvqz-f12',
                                'ccsd(t)-f12/cc-pvtz-f12(-pp)', 'ccsd(t)/aug-cc-pvtz(-pp)', 'ccsd(t)-f12/aug-cc-pvdz',
                                'ccsd(t)-f12/aug-cc-pvtz', 'ccsd(t)-f12/aug-cc-pvqz', 'b-ccsd(t)-f12/cc-pvdz-f12',
                                'b-ccsd(t)-f12/cc-pvtz-f12', 'b-ccsd(t)-f12/cc-pvqz-f12', 'b-ccsd(t)-f12/cc-pcvdz-f12',
                                'b-ccsd(t)-f12/cc-pcvtz-f12', 'b-ccsd(t)-f12/cc-pcvqz-f12', 'b-ccsd(t)-f12/aug-cc-pvdz',
                                'b-ccsd(t)-f12/aug-cc-pvtz', 'b-ccsd(t)-f12/aug-cc-pvqz', 'mp2_rmp2_pvdz',
                                'mp2_rmp2_pvtz', 'mp2_rmp2_pvqz', 'ccsd-f12/cc-pvdz-f12',
                                'ccsd(t)-f12/cc-pvdz-f12_noscale', 'g03_pbepbe_6-311++g_d_p', 'fci/cc-pvdz',
                                'fci/cc-pvtz', 'fci/cc-pvqz','bmk/cbsb7', 'bmk/6-311g(2d,d,p)', 'b3lyp/6-31g**',
                                'b3lyp/6-311+g(3df,2p)', 'MRCI+Davidson/aug-cc-pV(T+d)Z']:
                    model_chemistry = sp_level
            self.model_chemistry = model_chemistry
            logging.debug('Using {0} as model chemistry for energy corrections in Arkane'.format(
                self.model_chemistry))
            if not self.model_chemistry:
                logging.warn('Could not determine a Model Chemistry to be used in Arkane, NOT calculating thermodata')
                for spc in self.arc_species_list:
                    spc.generate_thermo = False
        if self.model_chemistry:
            logging.info('Using {0} as model chemistry for energy corrections in Arkane'.format(
                self.model_chemistry))

    def determine_remote(self, diagnostics=False):
        """
        Determine whether ARC is executed remotely
        and if so the available ESS software and the cluster software of the server
        if `diagnostics` is True, this method will not raise errors, and will print its findings
        """
        if self.ess_settings is not None:
            self.settings['ssh'] = True
            for ess, server in self.ess_settings.items():
                if ess.lower() not in ['gaussian', 'qchem', 'molpro']:
                    raise SettingsError('Recognized ESS software are Gaussian, QChem or Molpro. Got: {0}'.format(ess))
                if server.lower() not in servers:
                    server_names = [name for name in servers]
                    raise SettingsError('Recognized servers are {0}. Got: {1}'.format(server_names, servers))
                self.settings[ess.lower()] = server.lower()
            logging.info('\nUsing the following user input: {0}\n'.format(self.settings))
            return
        if diagnostics:
            logging.info('\n\n\n ***** Running ESS diagnostics: *****\n')
        # os.system('. ~/.bashrc')  # TODO This might be a security risk - rethink it
        if 'SSH_CONNECTION' in os.environ:
            # ARC is executed on a server, proceed
            logging.info('\n\nExecuting QM jobs locally.')
            if diagnostics:
                logging.info('ARC is being excecuted on a server (found "SSH_CONNECTION" in the os.environ dictionary')
                logging.info('Using distutils.spawn.find_executable() to find ESS')
            self.settings['ssh'] = False
            g03 = find_executable('g03')
            g09 = find_executable('g09')
            g16 = find_executable('g16')
            if g03 or g09 or g16:
                if diagnostics:
                    logging.info('Found Gaussian: g03={0}, g09={1}, g16={2}'.format(g03, g09, g16))
                self.settings['gaussian'] = True
            else:
                if diagnostics:
                    logging.info('Did NOT find Gaussian: g03={0}, g09={1}, g16={2}'.format(g03, g09, g16))
                self.settings['gaussian'] = False
            qchem = find_executable('qchem')
            if qchem:
                self.settings['qchem'] = True
            else:
                if diagnostics:
                    logging.info('Did not find QChem')
                self.settings['qchem'] = False
            molpro = find_executable('molpro')
            if molpro:
                self.settings['molpro'] = True
            else:
                if diagnostics:
                    logging.info('Did not find Molpro')
                self.settings['molpro'] = False
            if self.settings['gaussian']:
                logging.info('Found Gaussian')
            if self.settings['qchem']:
                logging.info('Found QChem')
            if self.settings['molpro']:
                logging.info('Found Molpro')
        else:
            # ARC is executed locally, communication with a server needs to be established
            if diagnostics:
                logging.info('ARC is being excecuted on a PC'
                             ' (did not find "SSH_CONNECTION" in the os.environ dictionary')
            self.settings['ssh'] = True
            logging.info('\n\nExecuting QM jobs remotely. Mapping servers...')
            # map servers
            self.settings['gaussian'], self.settings['qchem'], self.settings['molpro'] = None, None, None
            for server in servers.keys():
                if diagnostics:
                    logging.info('Trying {0}'.format(server))
                ssh = SSH_Client(server)
                cmd = '. ~/.bashrc; which g03'
                g03, _ = ssh.send_command_to_server(cmd)
                cmd = '. ~/.bashrc; which g09'
                g09, _ = ssh.send_command_to_server(cmd)
                cmd = '. ~/.bashrc; which g16'
                g16, _ = ssh.send_command_to_server(cmd)
                if g03 or g09 or g16:
                    if diagnostics:
                        logging.info('Found Gaussian on {3}: g03={0}, g09={1}, g16={2}'.format(g03, g09, g16, server))
                    if self.settings['gaussian'] is None or 'precedence' in servers[server]\
                            and servers[server]['precedence'] == 'gaussian':
                        self.settings['gaussian'] = server
                elif diagnostics:
                    logging.info('Did NOT find Gaussian on {3}: g03={0}, g09={1}, g16={2}'.format(g03, g09, g16, server))
                cmd = '. ~/.bashrc; which qchem'
                qchem, _ = ssh.send_command_to_server(cmd)
                if qchem:
                    if diagnostics:
                        logging.info('Found QChem on {0}'.format(server))
                    if self.settings['qchem'] is None or 'precedence' in servers[server]\
                            and servers[server]['precedence'] == 'qchem':
                        self.settings['qchem'] = server
                elif diagnostics:
                    logging.info('Did NOT find QChem on {0}'.format(server))
                cmd = '. .bashrc; which molpro'
                molpro, _ = ssh.send_command_to_server(cmd)
                if molpro:
                    if diagnostics:
                        logging.info('Found Molpro on {0}'.format(server))
                    if self.settings['molpro'] is None or 'precedence' in servers[server]\
                            and servers[server]['precedence'] == 'molpro':
                        self.settings['molpro'] = server
                elif diagnostics:
                    logging.info('Did NOT find Molpro on {0}'.format(server))
            if diagnostics:
                logging.info('\n')
            if self.settings['gaussian']:
                logging.info('Using Gaussian on {0}'.format(self.settings['gaussian']))
            if self.settings['qchem']:
                logging.info('Using QChem on {0}'.format(self.settings['qchem']))
            if self.settings['molpro']:
                logging.info('Using Molpro on {0}'.format(self.settings['molpro']))
            logging.info('\n')
        if not self.settings['gaussian'] and not self.settings['qchem'] and not self.settings['molpro']\
                and not diagnostics:
            raise SettingsError('Could not find any ESS. Check your .bashrc definitions on the server.\n'
                                'Alternatively, you could pass a software-server dictionary to arc as `ess_settings`')
        elif diagnostics:
            logging.info('ESS diagnostics completed')


def delete_all_arc_jobs(server_list):
    """
    Delete all ARC-spawned jobs (with job name starting with `a` and a digit) from :list:servers
    (`servers` could also be a string of one server name)
    Make sure you know what you're doing, so unrelated jobs won't be deleted...
    Useful when terminating ARC while some (ghost) jobs are still running.
    """
    if isinstance(server_list, str):
        server_list = [server_list]
    for server in server_list:
        print('\nDeleting all ARC jobs from {0}...'.format(server))
        cmd = check_status_command[servers[server]['cluster_soft']] + ' -u ' + servers[server]['un']
        ssh = SSH_Client(server)
        stdout, _ = ssh.send_command_to_server(cmd)
        for status_line in stdout:
            s = re.search(' a\d+', status_line)
            if s is not None:
                if servers[server]['cluster_soft'].lower() == 'slurm':
                    job_id = s.group()[1:]
                    server_job_id = status_line.split()[0]
                    ssh.delete_job(server_job_id)
                    print('deleted job {0} ({1} on server)'.format(job_id, server_job_id))
                elif servers[server]['cluster_soft'].lower() == 'oge':
                    job_id = s.group()[1:]
                    ssh.delete_job(job_id)
                    print('deleted job {0}'.format(job_id))
    print('\ndone.')
