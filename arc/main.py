#!/usr/bin/env python
# encoding: utf-8

"""
ARC's main module.
To run ARC through its API, first make an instance of the ARC class, then call the .execute() method. For example::

  arc0 = ARC(project='ArcDemo', arc_species_list=[spc0, spc1, spc2])
  arc0.execute()

Where ``spc0``, ``spc1``, and ``spc2`` in the above example are :ref:`ARCSpecies <species>` objects.

"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import logging
import os
import time
import datetime
import shutil
from distutils.spawn import find_executable
from IPython.display import display

from rmgpy.species import Species
from rmgpy.reaction import Reaction
from arkane.statmech import assign_frequency_scale_factor

import arc.rmgdb as rmgdb
from arc.settings import arc_path, default_levels_of_theory, servers, valid_chars, default_job_types
from arc.scheduler import Scheduler
from arc.common import VERSION, read_yaml_file, time_lapse, check_ess_settings, initialize_log, log_footer, get_logger,\
    save_yaml_file, initialize_job_types
from arc.arc_exceptions import InputError, SettingsError, SpeciesError
from arc.species.species import ARCSpecies
from arc.reaction import ARCReaction
from arc.processor import Processor
from arc.job.ssh import SSHClient
from arc.utils.scale import determine_scaling_factors

try:
    from arc.settings import global_ess_settings
except ImportError:
    global_ess_settings = None

##################################################################

logger = get_logger()


class ARC(object):
    """
    The main ARC class.

    Args:
        input_dict (dict, str, optional): Either a dictionary from which to recreate this object, or the path to an ARC
                                          input/restart YAML file.
        project (str, optional): The project's name. Used for naming the working directory.
        arc_species_list (list, optional): A list of :ref:`ARCSpecies <species>` objects.
        arc_rxn_list (list, optional): A list of :ref:`ARCReaction <reaction>` objects.
        level_of_theory (str, optional): A string representing either sp//geometry levels or a composite method, e.g.
                                         'CBS-QB3', 'CCSD(T)-F12a/aug-cc-pVTZ//B3LYP/6-311++G(3df,3pd)'...
        conformer_level (str, optional): Level of theory for conformer searches.
        composite_method (str, optional): Composite method.
        opt_level (str, optional): Level of theory for geometry optimization.
        freq_level (str, optional): Level of theory for frequency calculations.
        sp_level (str, optional): Level of theory for single point calculations.
        scan_level (str, optional): Level of theory for rotor scans.
        ts_guess_level (str, optional): Level of theory for comparisons of TS guesses between different methods.
        use_bac (bool, optional): Whether or not to use bond additivity corrections for thermo calculations.
        job_types (dict, optional): A dictionary of job types to execute. Keys are job types, values are boolean.
        model_chemistry (str, optional): The model chemistry in Arkane for energy corrections (AE, BAC) and
                                         frequencies/ZPE scaling factor. Can usually be determined automatically.
        initial_trsh (dict, optional): Troubleshooting methods to try by default. Keys are ESS software, values are
                                       trshs.
        t_min (tuple, optional): The minimum temperature for kinetics computations, e.g., (500, str('K')).
        t_max (tuple, optional): The maximum temperature for kinetics computations, e.g., (3000, str('K')).
        t_count (int, optional): The number of temperature points between t_min and t_max for kinetics computations.
        verbose (int, optional): The logging level to use.
        project_directory (str, optional): The path to the project directory.
        max_job_time (int, optional): The maximal allowed job time on the server in hours.
        allow_nonisomorphic_2d (bool, optional): Whether to optimize species even if they do not have a 3D conformer
                                                 that is isomorphic to the 2D graph representation.
        job_memory (int, optional): The total allocated job memory in GB (14 by default to be lower than 90% * 16 GB).
        ess_settings (dict, optional): A dictionary of available ESS (keys) and a corresponding server list (values).
        bath_gas (str, optional): A bath gas. Currently used in OneDMin to calc L-J parameters.
                                  Allowed values are He, Ne, Ar, Kr, H2, N2, O2.
        adaptive_levels (dict, optional): A dictionary of levels of theory for ranges of the number of heavy atoms in
                                          the molecule. Keys are tuples of (min_num_atoms, max_num_atoms), values are
                                          dictionaries with ``optfreq`` and ``sp`` as keys and levels of theory as
                                          values.
        freq_scale_factor (float, optional): The harmonic frequencies scaling factor. Could be automatically determined
                                             if not available in Arkane and not provided by the user.
        calc_freq_factor (bool, optional): Whether to calculate the frequencies scaling factor using Truhlar's method if
                                           it was not given by the user and could not be determined by Arkane. True to
                                           calculate, False to use user input / Arkane's value / Arkane's default.
        confs_to_dft (int, optional): The number of lowest MD conformers to DFT at the conformers_level.
        keep_checks (bool, optional): Whether to keep all Gaussian checkfiles when ARC terminates. True to keep,
                                      default is False.
        dont_gen_confs (list, optional): A list of species labels for which conformer generation should be avoided
                                         if xyz is given.

    Attributes:
        project (str): The project's name. Used for naming the working directory.
        project_directory (str): The path to the project directory.
        arc_species_list (list): A list of :ref:`ARCSpecies <species>` objects.
        arc_rxn_list (list): A list of :ref:`ARCReaction <reaction>` objects.
        conformer_level (str): Level of theory for conformer searches.
        composite_method (str): Composite method.
        opt_level (str): Level of theory for geometry optimization.
        freq_level (str): Level of theory for frequency calculations.
        sp_level (str): Level of theory for single point calculations.
        scan_level (str): Level of theory for rotor scans.
        ts_guess_level (str): Level of theory for comparisons of TS guesses between different methods.
        adaptive_levels (dict): A dictionary of levels of theory for ranges of the number of heavy atoms in the
                                molecule. Keys are tuples of (min_num_atoms, max_num_atoms), values are dictionaries
                                with ``optfreq`` and ``sp`` as keys and levels of theory as values.
        output (dict): Output dictionary with status and final QM file paths for all species. Only used for restarting,
                         the actual object used is in the Scheduler class.
        use_bac (bool): Whether or not to use bond additivity corrections for thermo calculations.
        model_chemistry (str): The model chemistry in Arkane for energy corrections (AE, BAC) and frequencies/ZPE
                               scaling factor. Can usually be determined automatically.
        freq_scale_factor (float): The harmonic frequencies scaling factor. Could be automatically determined if not
                                   available in Arkane and not provided by the user.
        calc_freq_factor (bool): Whether to calculate the frequencies scaling factor using Truhlar's method if it was
                                 not given by the user and could not be determined by Arkane. True to calculate, False
                                 to use user input / Arkane's value / Arkane's default.
        ess_settings (dict): A dictionary of available ESS (keys) and a corresponding server list (values).
        initial_trsh (dict): Troubleshooting methods to try by default. Keys are ESS software, values are trshs.
        t0 (float): Initial time when the project was spawned.
        confs_to_dft (int): The number of lowest MD conformers to DFT at the conformers_level.
        execution_time (str): Overall execution time.
        lib_long_desc (str): A multiline description of levels of theory for the outputted RMG libraries.
        running_jobs (dict): A dictionary of jobs submitted in a precious ARC instance, used for restarting ARC.
        t_min (tuple): The minimum temperature for kinetics computations, e.g., (500, str('K')).
        t_max (tuple): The maximum temperature for kinetics computations, e.g., (3000, str('K')).
        t_count (int): The number of temperature points between t_min and t_max for kinetics computations.
        max_job_time (int): The maximal allowed job time on the server in hours.
        rmgdb (RMGDatabase): The RMG database object.
        allow_nonisomorphic_2d (bool): Whether to optimize species even if they do not have a 3D conformer that is
                                       isomorphic to the 2D graph representation.
        memory (int): The total allocated job memory in GB (14 by default to be lower than 90% * 16 GB).
        job_types (dict): A dictionary of job types to execute. Keys are job types, values are boolean.
        bath_gas (str): A bath gas. Currently used in OneDMin to calc L-J parameters.
                        Allowed values are He, Ne, Ar, Kr, H2, N2, O2.
        keep_checks (bool): Whether to keep all Gaussian checkfiles when ARC terminates. True to keep, default is False.
        dont_gen_confs (list): A list of species labels for which conformer generation should be avoided
                               if xyz is given.

    """

    def __init__(self, input_dict=None, project=None, arc_species_list=None, arc_rxn_list=None, level_of_theory='',
                 conformer_level='', composite_method='', opt_level='', freq_level='', sp_level='', scan_level='',
                 ts_guess_level='', use_bac=True, job_types=None, model_chemistry='', initial_trsh=None, t_min=None,
                 t_max=None, t_count=None, verbose=logging.INFO, project_directory=None, max_job_time=120,
                 allow_nonisomorphic_2d=False, job_memory=14, ess_settings=None, bath_gas=None,
                 adaptive_levels=None, freq_scale_factor=None, calc_freq_factor=True, confs_to_dft=5,
                 keep_checks=False, dont_gen_confs=None):
        self.__version__ = VERSION
        self.verbose = verbose
        self.output = dict()
        self.running_jobs = dict()
        self.lib_long_desc = ''
        self.unique_species_labels = list()
        self.rmgdb = rmgdb.make_rmg_database_object()
        self.max_job_time = max_job_time
        self.allow_nonisomorphic_2d = allow_nonisomorphic_2d
        self.memory = job_memory
        self.orbitals_level = default_levels_of_theory['orbitals'].lower()
        self.ess_settings = dict()
        self.calc_freq_factor = calc_freq_factor
        self.keep_checks = keep_checks

        if input_dict is None:
            if project is None:
                raise ValueError('A project name must be provided for a new project')
            self.project = project
            self.t_min = t_min
            self.t_max = t_max
            self.t_count = t_count
            self.job_types = initialize_job_types(job_types)
            self.bath_gas = bath_gas
            self.confs_to_dft = confs_to_dft
            self.adaptive_levels = adaptive_levels
            self.project_directory = project_directory if project_directory is not None\
                else os.path.join(arc_path, 'Projects', self.project)
            if not os.path.exists(self.project_directory):
                os.makedirs(self.project_directory)
            initialize_log(log_file=os.path.join(self.project_directory, 'arc.log'), project=self.project,
                           project_directory=self.project_directory, verbose=self.verbose)
            self.dont_gen_confs = dont_gen_confs if dont_gen_confs is not None else list()
            self.t0 = time.time()  # init time
            self.execution_time = None
            self.initial_trsh = initial_trsh if initial_trsh is not None else dict()
            self.use_bac = use_bac
            self.model_chemistry = model_chemistry
            self.freq_scale_factor = freq_scale_factor
            if self.model_chemistry:
                logger.info('Using {0} as model chemistry for energy corrections in Arkane'.format(
                    self.model_chemistry))
            if not self.job_types['fine']:
                logger.info('\n')
                logger.warning('Not using a fine grid for geometry optimization jobs')
                logger.info('\n')
            if not self.job_types['1d_rotors']:
                logger.info('\n')
                logger.warning("Not running rotor scans."
                               " This might compromise finding the best conformer, as dihedral angles won't be"
                               " corrected. Also, entropy won't be accurate.")
                logger.info('\n')

            if level_of_theory.count('//') > 1:
                raise InputError('Level of theory seems wrong. It should either be a composite method (like CBS-QB3)'
                                 ' or be of the form sp//geometry, e.g., CCSD(T)-F12/avtz//wB97x-D3/6-311++g**.'
                                 ' Got: {0}'.format(level_of_theory))

            if conformer_level:
                logger.info('Using {0} for refined conformer searches (after filtering via force fields)'.format(
                    conformer_level))
                self.conformer_level = conformer_level.lower()
            else:
                self.conformer_level = default_levels_of_theory['conformer'].lower()
                logger.info('Using default level {0} for refined conformer searches (after filtering via force'
                            ' fields)'.format(default_levels_of_theory['conformer']))
            if ts_guess_level:
                logger.info('Using {0} for TS guesses comparison of different methods'.format(ts_guess_level))
                self.ts_guess_level = ts_guess_level.lower()
            else:
                self.ts_guess_level = default_levels_of_theory['ts_guesses'].lower()
                logger.info('Using default level {0} for TS guesses comparison of different methods'.format(
                    default_levels_of_theory['ts_guesses']))

            if level_of_theory:
                if '/' not in level_of_theory:  # assume this is a composite method
                    self.composite_method = level_of_theory.lower()
                    logger.info('Using composite method {0}'.format(self.composite_method))
                    self.opt_level = ''
                    self.sp_level = ''
                    if freq_level:
                        self.freq_level = freq_level.lower()
                        logger.info('Using {0} for frequency calculations'.format(self.freq_level))
                    else:
                        self.freq_level = default_levels_of_theory['freq_for_composite'].lower()
                        logger.info('Using default level {0} for frequency calculations after composite jobs'.format(
                            self.freq_level))
                elif '//' in level_of_theory:
                    self.composite_method = ''
                    self.opt_level = level_of_theory.lower().split('//')[1]
                    self.freq_level = level_of_theory.lower().split('//')[1]
                    self.sp_level = level_of_theory.lower().split('//')[0]
                    logger.info('Using {0} for geometry optimizations'.format(level_of_theory.split('//')[1]))
                    logger.info('Using {0} for frequency calculations'.format(level_of_theory.split('//')[1]))
                    logger.info('Using {0} for single point calculations'.format(level_of_theory.split('//')[0]))
                elif '/' in level_of_theory and '//' not in level_of_theory:
                    # assume this is not a composite method, and the user meant to run opt, freq and sp at this level.
                    # running an sp after opt at the same level is meaningless, but doesn't matter much also...
                    self.composite_method = ''
                    self.opt_level = level_of_theory.lower()
                    self.freq_level = level_of_theory.lower()
                    self.sp_level = level_of_theory.lower()
                    logger.info('Using {0} for geometry optimizations'.format(level_of_theory))
                    logger.info('Using {0} for frequency calculations'.format(level_of_theory))
                    logger.info('Using {0} for single point calculations'.format(level_of_theory))
            else:
                self.composite_method = composite_method.lower()
                if self.composite_method:
                    if level_of_theory and level_of_theory.lower != self.composite_method:
                        raise InputError('Specify either composite_method or level_of_theory')
                    logger.info('Using composite method {0}'.format(composite_method))
                    if self.composite_method == 'cbs-qb3':
                        self.model_chemistry = self.composite_method
                        logger.info('Using {0} as model chemistry for energy corrections in Arkane'.format(
                            self.model_chemistry))
                    elif self.use_bac:
                        raise InputError('Could not determine model chemistry to use for composite method {0}'.format(
                            self.composite_method))

                if opt_level:
                    self.opt_level = opt_level.lower()
                    logger.info('Using {0} for geometry optimizations'.format(self.opt_level))
                elif not self.composite_method:
                    self.opt_level = default_levels_of_theory['opt'].lower()
                    logger.info('Using default level {0} for geometry optimizations'.format(self.opt_level))
                else:
                    self.opt_level = ''

                if freq_level:
                    self.freq_level = freq_level.lower()
                    logger.info('Using {0} for frequency calculations'.format(self.freq_level))
                elif not self.composite_method:
                    if opt_level:
                        self.freq_level = opt_level.lower()
                        logger.info('Using user-defined opt level {0} for frequency calculations as well'.format(
                            self.freq_level))
                    else:
                        # self.freq_level = 'wb97x-d3/def2-tzvpd'
                        # logger.info('Using wB97x-D3/def2-TZVPD for frequency calculations')
                        self.freq_level = default_levels_of_theory['freq'].lower()
                        logger.info('Using default level {0} for frequency calculations'.format(self.freq_level))
                else:
                    # This is a composite method
                    self.freq_level = default_levels_of_theory['freq_for_composite'].lower()
                    logger.info('Using default level {0} for frequency calculations after composite jobs'.format(
                        self.freq_level))

                if sp_level:
                    self.sp_level = sp_level.lower()
                    logger.info('Using {0} for single point calculations'.format(self.sp_level))
                elif not self.composite_method:
                    self.sp_level = default_levels_of_theory['sp'].lower()
                    logger.info('Using default level {0} for single point calculations'.format(self.sp_level))
                else:
                    # It's a composite method, no need in explicit sp
                    self.sp_level = ''

            if scan_level:
                self.scan_level = scan_level.lower()
                if self.job_types['1d_rotors']:
                    logger.info('Using {0} for rotor scans'.format(self.scan_level))
            elif self.job_types['1d_rotors']:
                if not self.composite_method:
                    self.scan_level = default_levels_of_theory['scan'].lower()
                    logger.info('Using default level {0} for rotor scans'.format(self.scan_level))
                else:
                    # This is a composite method
                    self.scan_level = default_levels_of_theory['scan_for_composite'].lower()
                    logger.info('Using default level {0} for rotor scans after composite jobs'.format(
                        self.scan_level))
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
            if self.job_types['bde']:
                self.add_hydrogen_for_bde()
            self.determine_unique_species_labels()
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
            project_directory = project_directory if project_directory is not None\
                else os.path.abspath(os.path.dirname(input_dict))
            self.from_dict(input_dict=input_dict, project=project, project_directory=project_directory)
        if self.adaptive_levels is not None:
            logger.info('Using the following adaptive levels of theory:\n{0}'.format(self.adaptive_levels))
        if not self.ess_settings:
            # don't override self.ess_settings if determined from an input dictionary
            self.ess_settings = check_ess_settings(ess_settings or global_ess_settings)
        if not self.ess_settings:
            self.determine_ess_settings()
        self.determine_model_chemistry()
        self.scheduler = None
        self.check_project_name()
        self.check_freq_scaling_factor()
        self.restart_dict = self.as_dict()

        # make a backup copy of the restart file if it exists (but don't save an updated one just yet)
        if os.path.isfile(os.path.join(self.project_directory, 'restart.yml')):
            if not os.path.isdir(os.path.join(self.project_directory, 'log_and_restart_archive')):
                os.mkdir(os.path.join(self.project_directory, 'log_and_restart_archive'))
            local_time = datetime.datetime.now().strftime("%H%M%S_%b%d_%Y")
            restart_backup_name = 'restart.old.' + local_time + '.yml'
            shutil.copy(os.path.join(self.project_directory, 'restart.yml'),
                        os.path.join(self.project_directory, 'log_and_restart_archive', restart_backup_name))

    def as_dict(self):
        """
        A helper function for dumping this object as a dictionary in a YAML file for restarting ARC.
        """
        restart_dict = dict()
        restart_dict['project'] = self.project
        if self.bath_gas is not None:
            restart_dict['bath_gas'] = self.bath_gas
        if self.adaptive_levels is not None:
            restart_dict['adaptive_levels'] = self.adaptive_levels
        restart_dict['job_types'] = self.job_types
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
        if self.freq_scale_factor is not None:
            restart_dict['freq_scale_factor'] = self.freq_scale_factor
        restart_dict['calc_freq_factor'] = self.calc_freq_factor
        if self.dont_gen_confs:
            restart_dict['dont_gen_confs'] = self.dont_gen_confs
        restart_dict['species'] = [spc.as_dict() for spc in self.arc_species_list]
        restart_dict['reactions'] = [rxn.as_dict() for rxn in self.arc_rxn_list]
        restart_dict['output'] = self.output  # if read from_dict then it has actual values
        restart_dict['running_jobs'] = self.running_jobs  # if read from_dict then it has actual values
        restart_dict['t_min'] = self.t_min
        restart_dict['t_max'] = self.t_max
        restart_dict['t_count'] = self.t_count
        restart_dict['max_job_time'] = self.max_job_time
        restart_dict['allow_nonisomorphic_2d'] = self.allow_nonisomorphic_2d
        restart_dict['ess_settings'] = self.ess_settings
        restart_dict['job_memory'] = self.memory
        restart_dict['confs_to_dft'] = self.confs_to_dft
        if self.keep_checks:
            restart_dict['keep_checks'] = self.keep_checks
        return restart_dict

    def from_dict(self, input_dict, project=None, project_directory=None):
        """
        A helper function for loading this object from a dictionary in a YAML file for restarting ARC.
        If `project` name and `ess_settings` are given as well to __init__, they will override the respective values
        in the restart dictionary.
        """
        if isinstance(input_dict, (str, unicode)):
            input_dict = read_yaml_file(input_dict)
        if project is None and 'project' not in input_dict:
            raise InputError('A project name must be given')
        self.project = project if project is not None else input_dict['project']
        self.project_directory = project_directory if project_directory is not None \
            else os.path.join(arc_path, 'Projects', self.project)
        if not os.path.exists(self.project_directory):
            os.makedirs(self.project_directory)
        initialize_log(log_file=os.path.join(self.project_directory, 'arc.log'), project=self.project,
                       project_directory=self.project_directory, verbose=self.verbose)
        self.t0 = time.time()  # init time
        self.execution_time = None
        self.verbose = input_dict['verbose'] if 'verbose' in input_dict else self.verbose
        self.max_job_time = input_dict['max_job_time'] if 'max_job_time' in input_dict else self.max_job_time
        self.memory = input_dict['job_memory'] if 'job_memory' in input_dict else self.memory
        self.bath_gas = input_dict['bath_gas'] if 'bath_gas' in input_dict else None
        self.confs_to_dft = input_dict['confs_to_dft'] if 'confs_to_dft' in input_dict else 5
        self.adaptive_levels = input_dict['adaptive_levels'] if 'adaptive_levels' in input_dict else None
        self.keep_checks = input_dict['keep_checks'] if 'keep_checks' in input_dict else False
        self.allow_nonisomorphic_2d = input_dict['allow_nonisomorphic_2d']\
            if 'allow_nonisomorphic_2d' in input_dict else False
        self.output = input_dict['output'] if 'output' in input_dict else dict()
        self.freq_scale_factor = input_dict['freq_scale_factor'] if 'freq_scale_factor' in input_dict else None
        if self.output:
            for label, spc_output in self.output.items():
                if 'paths' in spc_output:
                    for key, val in spc_output['paths'].items():
                        if key in ['geo', 'freq', 'sp', 'composite']:
                            if val and not os.path.isfile(val):
                                # try correcting relative paths
                                if os.path.isfile(os.path.join(arc_path, val)):
                                    self.output[label]['paths'][key] = os.path.join(arc_path, val)
                                elif os.path.isfile(os.path.join(arc_path, 'Projects', val)):
                                    self.output[label]['paths'][key] = os.path.join(arc_path, 'Projects', val)
                                else:
                                    raise SpeciesError('Could not find {0} output file for species {1}: {2}'.format(
                                        key, label, val))
        self.running_jobs = input_dict['running_jobs'] if 'running_jobs' in input_dict else dict()
        logger.debug('output dictionary successfully parsed:\n{0}'.format(self.output))
        self.t_min = input_dict['t_min'] if 't_min' in input_dict else None
        self.t_max = input_dict['t_max'] if 't_max' in input_dict else None
        self.t_count = input_dict['t_count'] if 't_count' in input_dict else None

        self.initial_trsh = input_dict['initial_trsh'] if 'initial_trsh' in input_dict else dict()
        self.job_types = input_dict['job_types'] if 'job_types' in input_dict else default_job_types
        self.job_types = initialize_job_types(self.job_types)
        self.use_bac = input_dict['use_bac'] if 'use_bac' in input_dict else True
        self.calc_freq_factor = input_dict['calc_freq_factor'] if 'calc_freq_factor' in input_dict else True
        self.model_chemistry = input_dict['model_chemistry'] if 'use_bac' in input_dict\
                                                                and input_dict['use_bac'] else ''
        ess_settings = input_dict['ess_settings'] if 'ess_settings' in input_dict else global_ess_settings
        self.ess_settings = check_ess_settings(ess_settings)
        self.dont_gen_confs = input_dict['dont_gen_confs'] if 'dont_gen_confs' in input_dict else list()
        if not self.job_types['fine']:
            logger.info('\n')
            logger.warning('Not using a fine grid for geometry optimization jobs')
            logger.info('\n')
        if not self.job_types['1d_rotors']:
            logger.info('\n')
            logger.warning("Not running rotor scans."
                           " This might compromise finding the best conformer, as dihedral angles won't be"
                           " corrected. Also, entropy won't be accurate.")
            logger.info('\n')

        if 'conformer_level' in input_dict:
            self.conformer_level = input_dict['conformer_level'].lower()
            logger.info('Using {0} for refined conformer searches (after filtering via force fields)'.format(
                self.conformer_level))
        else:
            self.conformer_level = default_levels_of_theory['conformer'].lower()
            logger.info('Using default level {0} for refined conformer searches (after filtering via force'
                        ' fields)'.format(default_levels_of_theory['conformer']))

        if 'ts_guess_level' in input_dict:
            self.ts_guess_level = input_dict['ts_guess_level'].lower()
            logger.info('Using {0} for TS guesses comparison of different methods'.format(self.ts_guess_level))
        else:
            self.ts_guess_level = default_levels_of_theory['ts_guesses'].lower()
            logger.info('Using default level {0} for TS guesses comparison of different methods'.format(
                default_levels_of_theory['ts_guesses']))

        if 'level_of_theory' in input_dict:
            if '/' not in input_dict['level_of_theory']:  # assume this is a composite method
                self.composite_method = input_dict['level_of_theory'].lower()
                logger.info('Using composite method {0}'.format(self.composite_method))
                self.opt_level = ''
                self.sp_level = ''
                if 'freq_level' in input_dict:
                    self.freq_level = input_dict['freq_level'].lower()
                    logger.info('Using {0} for frequency calculations'.format(self.freq_level))
                else:
                    self.freq_level = default_levels_of_theory['freq_for_composite'].lower()
                    logger.info('Using default level {0} for frequency calculations after composite jobs'.format(
                        self.freq_level))
            elif '//' in input_dict['level_of_theory']:
                self.composite_method = ''
                self.opt_level = input_dict['level_of_theory'].lower().split('//')[1]
                self.freq_level = input_dict['level_of_theory'].lower().split('//')[1]
                self.sp_level = input_dict['level_of_theory'].lower().split('//')[0]
                logger.info('Using {0} for geometry optimizations'.format(
                    input_dict['level_of_theory'].split('//')[1]))
                logger.info('Using {0} for frequency calculations'.format(
                    input_dict['level_of_theory'].split('//')[1]))
                logger.info('Using {0} for single point calculations'.format(
                    input_dict['level_of_theory'].split('//')[0]))
            elif '/' in input_dict['level_of_theory'] and '//' not in input_dict['level_of_theory']:
                # assume this is not a composite method, and the user meant to run opt, freq and sp at this level.
                # running an sp after opt at the same level is meaningless, but doesn't matter much also...
                self.composite_method = ''
                self.opt_level = input_dict['level_of_theory'].lower()
                self.freq_level = input_dict['level_of_theory'].lower()
                self.sp_level = input_dict['level_of_theory'].lower()
                logger.info('Using {0} for geometry optimizations'.format(input_dict['level_of_theory']))
                logger.info('Using {0} for frequency calculations'.format(input_dict['level_of_theory']))
                logger.info('Using {0} for single point calculations'.format(input_dict['level_of_theory']))

        else:
            self.composite_method = input_dict['composite_method'].lower() if 'composite_method' in input_dict else ''
            if self.composite_method:
                logger.info('Using composite method {0}'.format(self.composite_method))
                if self.composite_method == 'cbs-qb3':
                    self.model_chemistry = self.composite_method
                    logger.info('Using {0} as model chemistry for energy corrections in Arkane'.format(
                        self.model_chemistry))
                elif self.use_bac:
                    raise InputError('Could not determine model chemistry to use for composite method {0}'.format(
                        self.composite_method))

            if 'opt_level' in input_dict:
                self.opt_level = input_dict['opt_level'].lower()
                logger.info('Using {0} for geometry optimizations'.format(self.opt_level))
            elif not self.composite_method:
                self.opt_level = default_levels_of_theory['opt'].lower()
                logger.info('Using default level {0} for geometry optimizations'.format(self.opt_level))
            else:
                self.opt_level = ''

            if 'freq_level' in input_dict:
                self.freq_level = input_dict['freq_level'].lower()
            elif not self.composite_method:
                if 'opt_level' in input_dict:
                    self.freq_level = input_dict['opt_level'].lower()
                    logger.info('Using user-defined opt level {0} for frequency calculations as well'.format(
                        self.freq_level))
                else:
                    self.freq_level = default_levels_of_theory['freq'].lower()
                    logger.info('Using default level {0} for frequency calculations'.format(self.freq_level))
            else:
                # This is a composite method
                self.freq_level = default_levels_of_theory['freq_for_composite'].lower()
                logger.info('Using default level {0} for frequency calculations after composite jobs'.format(
                    self.freq_level))

            if 'sp_level' in input_dict:
                self.sp_level = input_dict['sp_level'].lower()
                logger.info('Using {0} for single point calculations'.format(self.sp_level))
            elif not self.composite_method:
                self.sp_level = default_levels_of_theory['sp'].lower()
                logger.info('Using default level {0} for single point calculations'.format(self.sp_level))
            else:
                # It's a composite method, no need in explicit sp
                self.sp_level = ''

        if 'scan_level' in input_dict:
            self.scan_level = input_dict['scan_level'].lower()
            if '1d_rotors' in self.job_types:
                logger.info('Using {0} for rotor scans'.format(self.scan_level))
        elif '1d_rotors' in self.job_types:
            if not self.composite_method:
                self.scan_level = default_levels_of_theory['scan'].lower()
                logger.info('Using default level {0} for rotor scans'.format(self.scan_level))
            else:
                # This is a composite method
                self.scan_level = default_levels_of_theory['scan_for_composite'].lower()
                logger.info('Using default level {0} for rotor scans after composite jobs'.format(
                    self.scan_level))
        else:
            self.scan_level = ''

        if 'species' in input_dict:
            self.arc_species_list = [ARCSpecies(species_dict=spc_dict) for spc_dict in input_dict['species']]
            for spc in self.arc_species_list:
                for rotor_num, rotor_dict in spc.rotors_dict.items():
                    if not os.path.isfile(rotor_dict['scan_path']) and rotor_dict['success']:
                        # try correcting relative paths
                        if os.path.isfile(os.path.join(arc_path, rotor_dict['scan_path'])):
                            spc.rotors_dict[rotor_num]['scan_path'] = os.path.join(arc_path, rotor_dict['scan_path'])
                        elif os.path.isfile(os.path.join(arc_path, 'Projects', rotor_dict['scan_path'])):
                            spc.rotors_dict[rotor_num]['scan_path'] =\
                                os.path.join(arc_path, 'Projects', rotor_dict['scan_path'])
                        else:
                            raise SpeciesError('Could not find rotor scan output file for rotor {0} of species {1}:'
                                               ' {2}'.format(rotor_num, spc.label, rotor_dict['scan_path']))
        else:
            self.arc_species_list = list()
        if self.job_types['bde']:
            self.add_hydrogen_for_bde()
        self.determine_unique_species_labels()
        if 'reactions' in input_dict:
            self.arc_rxn_list = [ARCReaction(reaction_dict=rxn_dict) for rxn_dict in input_dict['reactions']]
            for i, rxn in enumerate(self.arc_rxn_list):
                rxn.index = i
        else:
            self.arc_rxn_list = list()

    def write_input_file(self, path=None):
        """
        Save the current attributes as an ARC input file.

        Args:
             path (str, optional): The full path for the generated input file.
        """
        if path is None:
            path = os.path.join(self.project_directory, 'input.yml')
        base_path = os.path.dirname(path)
        if not os.path.isdir(base_path):
            os.makedirs(base_path)
        logger.info('\n\nWriting input file to {0}'.format(path))
        save_yaml_file(path=path, content=self.restart_dict)

    def execute(self):
        """
        Execute ARC.
        """
        logger.info('\n')
        for species in self.arc_species_list:
            if not isinstance(species, ARCSpecies):
                raise ValueError('All species in arc_species_list must be ARCSpecies objects.'
                                 ' Got {0}'.format(type(species)))
            if species.is_ts:
                logger.info('Considering transition state: {0}'.format(species.label))
            else:
                logger.info('Considering species: {0}'.format(species.label))
                if species.mol is not None:
                    display(species.mol)
        logger.info('\n')
        for rxn in self.arc_rxn_list:
            if not isinstance(rxn, ARCReaction):
                raise ValueError('All reactions in arc_rxn_list must be ARCReaction objects.'
                                 ' Got {0}'.format(type(rxn)))
        self.scheduler = Scheduler(project=self.project, species_list=self.arc_species_list, rxn_list=self.arc_rxn_list,
                                   composite_method=self.composite_method, conformer_level=self.conformer_level,
                                   opt_level=self.opt_level, freq_level=self.freq_level, sp_level=self.sp_level,
                                   scan_level=self.scan_level, ts_guess_level=self.ts_guess_level,
                                   ess_settings=self.ess_settings, job_types=self.job_types, bath_gas=self.bath_gas,
                                   initial_trsh=self.initial_trsh, rmgdatabase=self.rmgdb,
                                   restart_dict=self.restart_dict, project_directory=self.project_directory,
                                   max_job_time=self.max_job_time, allow_nonisomorphic_2d=self.allow_nonisomorphic_2d,
                                   memory=self.memory, orbitals_level=self.orbitals_level,
                                   adaptive_levels=self.adaptive_levels, confs_to_dft=self.confs_to_dft,
                                   dont_gen_confs=self.dont_gen_confs)

        save_yaml_file(path=os.path.join(self.project_directory, 'output', 'status.yml'), content=self.scheduler.output)

        if not self.keep_checks:
            self.delete_check_files()

        self.save_project_info_file()

        prc = Processor(project=self.project, project_directory=self.project_directory,
                        species_dict=self.scheduler.species_dict, rxn_list=self.scheduler.rxn_list,
                        output=self.scheduler.output, use_bac=self.use_bac, model_chemistry=self.model_chemistry,
                        lib_long_desc=self.lib_long_desc, rmgdatabase=self.rmgdb, t_min=self.t_min, t_max=self.t_max,
                        t_count=self.t_count, freq_scale_factor=self.freq_scale_factor)
        prc.process()
        self.summary()
        log_footer(execution_time=self.execution_time)

    def save_project_info_file(self):
        """
        Save a project info file.
        """
        self.execution_time = time_lapse(t0=self.t0)
        path = os.path.join(self.project_directory, '{0}.info'.format(self.project))
        if os.path.exists(path):
            os.remove(path)
        if self.job_types['fine']:
            fine_txt = '(using a fine grid)'
        else:
            fine_txt = '(NOT using a fine grid)'

        txt = ''
        txt += 'ARC v{0}\n'.format(self.__version__)
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
        if '1d_rotors' in self.job_types:
            txt += 'Rotor scans:      {0}\n'.format(self.scan_level)
        else:
            txt += 'Not scanning rotors\n'
        if self.use_bac:
            txt += 'Using bond additivity corrections for thermo\n'
        else:
            txt += 'NOT using bond additivity corrections for thermo\n'
        if self.initial_trsh:
            txt += 'Using an initial troubleshooting method "{0}"'.format(self.initial_trsh)
        txt += '\nUsing the following ESS settings: {0}\n'.format(self.ess_settings)
        txt += '\nConsidered the following species and TSs:\n'
        for species in self.arc_species_list:
            descriptor = 'TS' if species.is_ts else 'Species'
            failed = '' if self.scheduler.output[species.label]['convergence'] else ' (Failed!)'
            txt += '{descriptor} {label}{failed} (run time: {time})\n'.format(
                descriptor=descriptor, label=species.label, failed=failed, time=species.run_time)
        if self.arc_rxn_list:
            for rxn in self.arc_rxn_list:
                txt += 'Considered reaction: {0}\n'.format(rxn.label)
        txt += '\nOverall time since project initiation: {0}'.format(self.execution_time)
        txt += '\n'

        with open(path, 'w') as f:
            f.write(str(txt))
        self.lib_long_desc = txt

    def summary(self):
        """
        Report status and data of all species / reactions.
        """
        logger.info('\n\n\nAll jobs terminated. Summary for project {0}:\n'.format(self.project))
        for label, output in self.scheduler.output.items():
            if output['convergence']:
                logger.info('Species {0} converged successfully\n'.format(label))
            else:
                job_type_status = {key: val for key, val in self.output[label]['job_types'].items()
                                   if key in self.job_types and self.job_types[key]}
                logger.info('Species {0} failed with status:\n  {1}'.format(label, job_type_status))
                keys = ['conformers', 'isomorphism', 'info']
                for key in keys:
                    if key in output and output[key]:
                        logger.info(output[key])
                if 'warnings' in output and output['warnings']:
                    logger.info('\n and warnings: {0}'.format(output['warnings']))
                if 'errors' in output and output['errors']:
                    logger.info('\n and errors: {0}'.format(output['errors']))
                logger.info('\n')

    def determine_model_chemistry(self):
        """
        Determine the model_chemistry to be used in Arkane.

        Todo:
            * Determine whether the model chemistry exists in Arkane automatically instead of hard coding
        """
        if self.model_chemistry:
            self.model_chemistry = self.model_chemistry.lower()
            if self.model_chemistry.split('//')[0] not in [
                    'cbs-qb3', 'cbs-qb3-paraskevas', 'ccsd(t)-f12/cc-pvdz-f12', 'ccsd(t)-f12/cc-pvtz-f12',
                    'ccsd(t)-f12/cc-pvqz-f12', 'b3lyp/cbsb7', 'b3lyp/6-311g(2d,d,p)', 'b3lyp/6-311+g(3df,2p)',
                    'b3lyp/6-31g**']:
                logger.warning('No bond additivity corrections (BAC) are available in Arkane for "model chemistry"'
                               ' {0}. As a result, thermodynamic parameters are expected to be inaccurate. Make sure'
                               ' that atom energy corrections (AEC) were supplied or are available in Arkane to avoid'
                               ' error.'.format(self.model_chemistry))
        else:
            # model chemistry was not given, try to determine it from the sp_level and freq_level
            model_chemistry = ''
            if self.composite_method:
                self.model_chemistry = self.composite_method.lower()
            else:
                sp_level = self.sp_level.replace('f12a', 'f12').replace('f12b', 'f12').lower()
                freq_level = self.freq_level.replace('f12a', 'f12').replace('f12b', 'f12').lower()
                if sp_level in ['ccsd(t)-f12/cc-pvdz', 'ccsd(t)-f12/cc-pvtz', 'ccsd(t)-f12/cc-pvqz']:
                    logger.warning('Using model chemistry {0} based on sp level {1}.'.format(
                        sp_level + '-f12', sp_level))
                    sp_level += '-f12'
                if sp_level not in ['ccsd(t)-f12/cc-pvdz-f12', 'ccsd(t)-f12/cc-pvtz-f12', 'ccsd(t)-f12/cc-pvqz-f12',
                                    'b3lyp/cbsb7', 'b3lyp/6-311g(2d,d,p)', 'b3lyp/6-311+g(3df,2p)', 'b3lyp/6-31g**']\
                        and self.use_bac:
                    logger.info('\n\n')
                    logger.warning('Could not determine appropriate Model Chemistry to be used in Arkane for '
                                   'thermochemical parameter calculations.\nNot using atom energy corrections and '
                                   'bond additivity corrections!\n\n')
                    self.use_bac = False
                elif sp_level not in ['m06-2x/cc-pvtz', 'g3', 'm08so/mg3s*', 'klip_1', 'klip_2', 'klip_3', 'klip_2_cc',
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
                                      'fci/cc-pvtz', 'fci/cc-pvqz', 'bmk/cbsb7', 'bmk/6-311g(2d,d,p)', 'b3lyp/6-31g**',
                                      'b3lyp/6-311+g(3df,2p)', 'MRCI+Davidson/aug-cc-pV(T+d)Z']:
                    logger.warning('Could not determine a Model Chemistry to be used in Arkane, '
                                   'NOT calculating thermodata')
                    for spc in self.arc_species_list:
                        spc.generate_thermo = False
                self.model_chemistry = sp_level + '//' + freq_level
        if self.model_chemistry:
            logger.info('Using {0} as a model chemistry in Arkane'.format(self.model_chemistry))

    def determine_ess_settings(self, diagnostics=False):
        """
        Determine where each ESS is available, locally (in running on a server) and/or on remote servers.
        if `diagnostics` is True, this method will not raise errors, and will print its findings.
        """
        if self.ess_settings is not None and not diagnostics:
            self.ess_settings = check_ess_settings(self.ess_settings)
            return

        if diagnostics:
            t0 = time.time()
            logger.info('\n\n\n ***** Running ESS diagnostics: *****\n')

        # os.system('. ~/.bashrc')  # TODO This might be a security risk - rethink it

        for software in ['gaussian', 'molpro', 'qchem', 'orca', 'onedmin']:
            self.ess_settings[software] = list()

        # first look for ESS locally (e.g., when running ARC itself on a server)
        if 'SSH_CONNECTION' in os.environ and diagnostics:
            logger.info('Found "SSH_CONNECTION" in the os.environ dictionary, '
                        'using distutils.spawn.find_executable() to find ESS')
        if 'local' in servers:
            g03 = find_executable('g03')
            g09 = find_executable('g09')
            g16 = find_executable('g16')
            if g03 or g09 or g16:
                if diagnostics:
                    logger.info('Found Gaussian: g03={0}, g09={1}, g16={2}'.format(g03, g09, g16))
                self.ess_settings['gaussian'] = ['local']
            qchem = find_executable('qchem')
            if qchem:
                self.ess_settings['qchem'] = ['local']
            qchem = find_executable('orca')
            if qchem:
                self.ess_settings['orca'] = ['local']
            molpro = find_executable('molpro')
            if molpro:
                self.ess_settings['molpro'] = ['local']
            if any([val for val in self.ess_settings.values()]):
                if diagnostics:
                    logger.info('Found the following ESS on the local machine:')
                    logger.info([software for software, val in self.ess_settings.items() if val])
                    logger.info('\n')
                else:
                    logger.info('Did not find ESS on the local machine\n\n')
        else:
            logger.info("\nNot searching for ESS locally ('local' wasn't specified in the servers dictionary)\n")

        # look for ESS on remote servers ARC has access to
        logger.info('\n\nMapping servers...\n')
        for server in servers.keys():
            if server == 'local':
                continue
            if diagnostics:
                logger.info('\nTrying {0}'.format(server))
            ssh = SSHClient(server)

            cmd = '. ~/.bashrc; which g03'
            g03 = ssh.send_command_to_server(cmd)[0]
            cmd = '. ~/.bashrc; which g09'
            g09 = ssh.send_command_to_server(cmd)[0]
            cmd = '. ~/.bashrc; which g16'
            g16 = ssh.send_command_to_server(cmd)[0]
            if g03 or g09 or g16:
                if diagnostics:
                    logger.info('  Found Gaussian on {3}: g03={0}, g09={1}, g16={2}'.format(g03, g09, g16, server))
                self.ess_settings['gaussian'].append(server)
            elif diagnostics:
                logger.info('  Did NOT find Gaussian on {0}'.format(server))

            cmd = '. ~/.bashrc; which qchem'
            qchem = ssh.send_command_to_server(cmd)[0]
            if qchem:
                if diagnostics:
                    logger.info('  Found QChem on {0}'.format(server))
                self.ess_settings['qchem'].append(server)
            elif diagnostics:
                logger.info('  Did NOT find QChem on {0}'.format(server))

            cmd = '. ~/.bashrc; which orca'
            orca = ssh.send_command_to_server(cmd)[0]
            if orca:
                if diagnostics:
                    logger.info('  Found Orca on {0}'.format(server))
                self.ess_settings['orca'].append(server)
            elif diagnostics:
                logger.info('  Did NOT find Orca on {0}'.format(server))

            cmd = '. .bashrc; which molpro'
            molpro = ssh.send_command_to_server(cmd)[0]
            if molpro:
                if diagnostics:
                    logger.info('  Found Molpro on {0}'.format(server))
                self.ess_settings['molpro'].append(server)
            elif diagnostics:
                logger.info('  Did NOT find Molpro on {0}'.format(server))
        if diagnostics:
            logger.info('\n\n')
        if 'gaussian' in self.ess_settings.keys():
            logger.info('Using Gaussian on {0}'.format(self.ess_settings['gaussian']))
        if 'qchem' in self.ess_settings.keys():
            logger.info('Using QChem on {0}'.format(self.ess_settings['qchem']))
        if 'orca' in self.ess_settings.keys():
            logger.info('Using Orca on {0}'.format(self.ess_settings['orca']))
        if 'molpro' in self.ess_settings.keys():
            logger.info('Using Molpro on {0}'.format(self.ess_settings['molpro']))
        logger.info('\n')

        if 'gaussian' not in self.ess_settings.keys() and 'qchem' not in self.ess_settings.keys() \
                and 'orca' not in self.ess_settings.keys() and 'molpro' not in self.ess_settings.keys()\
                and 'onedmin' not in self.ess_settings.keys() and not diagnostics:
            raise SettingsError('Could not find any ESS. Check your .bashrc definitions on the server.\n'
                                'Alternatively, you could pass a software-server dictionary to arc as `ess_settings`')
        elif diagnostics:
            logger.info('ESS diagnostics completed (elapsed time: {0})'.format(time_lapse(t0)))

    def check_project_name(self):
        """
        Check the validity of the project name.
        """
        for char in self.project:
            if char not in valid_chars:
                raise InputError('A project name (used to naming folders) must contain only valid characters.'
                                 ' Got {0} in {1}.'.format(char, self.project))
            if char == ' ':  # space IS a valid character for other purposes, but isn't valid in project names
                raise InputError('A project name (used to naming folders) must not contain spaces.'
                                 ' Got {0}.'.format(self.project))

    def check_freq_scaling_factor(self):
        """
        Check that the harmonic frequencies scaling factor is known,
        otherwise spawn a calculation for it if calc_freq_factor is set to True.
        """
        if self.freq_scale_factor is None:
            # the user did not specify a scaling factor, see if Arkane has it
            level = self.freq_level if not self.composite_method else self.composite_method
            freq_scale_factor = assign_frequency_scale_factor(level)
            if freq_scale_factor != 1:
                # Arkane has this harmonic frequencies scaling factor (if not found, the factor is set to exactly 1)
                self.freq_scale_factor = freq_scale_factor
            else:
                logger.info('Could not determine the harmonic frequencies scaling factor for {0} from '
                            'Arkane.'.format(level))
                if self.calc_freq_factor:
                    logger.info("Calculating it using Truhlar's method:\n\n")
                    self.freq_scale_factor = determine_scaling_factors(
                        level, ess_settings=self.ess_settings, init_log=False)[0]
                else:
                    logger.info('Not calculating it, assuming a frequencies scaling factor of 1.')

    def delete_check_files(self):
        """
        Delete the Gaussian checkfiles, the usually take up lots of space and are not needed after ARC terminates.
        Pass True to the keep_checks flag to avoid deleting check files.
        """
        logged = False
        calcs_path = os.path.join(self.project_directory, 'calcs')
        for (root, _, files) in os.walk(calcs_path):
            for file_ in files:
                if file_ == 'check.chk' and os.path.isfile(os.path.join(root, file_)):
                    if not logged:
                        logger.info('deleting all Gaussian check files...')
                        logged = True
                    os.remove(os.path.join(root, file_))

    def determine_unique_species_labels(self):
        """
        Determine unique species labels.

        Raises:
            ValueError: If a non-unique species is found.
        """
        for arc_spc in self.arc_species_list:
            if arc_spc.label not in self.unique_species_labels:
                self.unique_species_labels.append(arc_spc.label)
            else:
                raise ValueError('Species label {0} is not unique'.format(arc_spc.label))

    def add_hydrogen_for_bde(self):
        """
        Make sure ARC has a hydrogen species labeled as 'H' for the final processing of bde jobs (if not, create one).
        """
        if any([spc.bdes is not None for spc in self.arc_species_list]):
            for species in self.arc_species_list:
                if species.label == 'H':
                    if species.number_of_atoms == 1 and species.get_xyz(get_cheap=True).split()[0][0] == 'H':
                        break
                    else:
                        raise SpeciesError('A species with label "H" was defined, but does not seem to be '
                                           'the hydrogen atom species. Cannot calculate bond dissociation energies.')
            else:
                # no H species defined, make one
                h = ARCSpecies(label='H', smiles='[H]', generate_thermo=False)
                self.arc_species_list.append(h)
